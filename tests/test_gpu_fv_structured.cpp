// dns flow --solver simple vs what is known without OpenFOAM
// (docs/fvstructured.md). Default: the gates, under a minute. A mode argument
// prints a ladder: `poiseuille`, `ras <model> <reTau> <beta> <ny>...`,
// `couette <n>...`, `cyl <rFarD> <iters> <n>...`.
//  V1 periodic plane Poiseuille, uniform mesh, FV closed form to rounding:
//     the wall face's half-cell gradient offsets the whole parabola by
//     f h^2 / (8 nu), every interior row exact
//  V2 stretched and focal Poiseuille: wall stress balances the force to
//     rounding, the profile converges at order 2
//  V3 SA and SST channel on the oracle's own y mesh vs rans_channel1d
//  V4 MRF spinning cylinder in a fixed circle vs the exact Couette profile
//  V5 uniform freestream preserved on a focal box
//  V6 determinism: two runs byte-identical
#include "dns/cli.hpp"
#include "dns/config.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvsimple.hpp"
#include "dns/fvstructured.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/fv_simple.hpp"
#include "dns/la/graph.hpp"
#include "dns/rans_channel1d.hpp"

#include "check.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <numbers>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using dnstest::finish;

namespace {

constexpr double kPi = std::numbers::pi;

FvStructured structured(std::vector<std::string> args)
{
    std::vector<std::string> store = {"dns", "flow", "--solver", "simple", "--fv-precision", "fp64"};
    store.insert(store.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& s : store) argv.push_back(s.data());
    auto raw = cli::parse(int(argv.size()), argv.data());
    if (!raw) {
        std::printf("parse: %s\n", raw.error().message.c_str());
        std::exit(2);
    }
    auto c = validate(*raw);
    if (!c) {
        std::printf("validate: %s %s\n", c.error().field.c_str(), c.error().message.c_str());
        std::exit(2);
    }
    return c->fvs;
}

Sdf bodySdf(const FvStructured& s)
{
    const double r = 0.5 * kPi / 4.0;
    if (s.revolved) return [r](double x, double y, double z) { return std::sqrt(x * x + y * y + z * z) - r; };
    return [r](double x, double y, double) { return std::hypot(x, y) - r; };
}

struct Result
{
    FaceMesh fm;
    gpu::FvState st;
    gpu::FvForceSums fo{};
    int iters = 0;
    double seconds = 0, last = 0; // last: the residual that stopped the march
};

// Marches until the sampled monitor stops moving (relative change < tol over
// `every` iterations) or maxIter. texts may be patched by the caller.
std::expected<Result, std::string> solve(const FvStructured& s, FoamCaseTexts t, int maxIter, int every, double tol,
                                         const FvNumerics& num = {})
{
    auto pm = fvStructuredMesh(s, bodySdf(s));
    if (!pm) return std::unexpected(pm.error());
    auto fc = parseFoamCase(t, pm->man);
    if (!fc) return std::unexpected(fc.error());
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Source);
    if (!fm) return std::unexpected(fm.error());
    const auto g = la::graphOf(*fm);
    const auto pat = la::ellPattern(g);
    auto setup = buildSimpleSetup(*fm, *fc, g, pat);
    if (!setup) return std::unexpected(setup.error());
    if (fc->ras.model != FvTurbModel::Laminar)
        if (auto r = buildRasSetup(*pm, *fm, *fc, *setup); !r) return std::unexpected(r.error());
    auto ds = gpu::DeviceSimple::make(*setup, *fc, g, num);
    if (!ds) return std::unexpected(ds.error());
    const auto t0 = std::chrono::steady_clock::now();
    Result R;
    double prev = 0, prevF = 0;
    for (int it = 1; it <= maxIter; ++it) {
        auto lg = ds->iterate();
        if (!lg) return std::unexpected(std::format("iteration {}: {}", it, lg.error()));
        R.iters = it;
        if (it % every != 0 && it != maxIter) continue;
        auto st = ds->state();
        if (!st) return std::unexpected(st.error());
        double m = 0, v = 0;
        for (std::size_t c = 0; c < std::size_t(fm->nC); ++c) {
            const double sp = std::hypot(st->U[c], st->U[std::size_t(fm->nC) + c]);
            m += fm->V[c] * sp;
            v += fm->V[c];
        }
        m /= v;
        R.last = std::abs(m - prev) / std::max(std::abs(m), 1e-300);
        if (fc->forces) { // the mean speed is far-field; the wall has its own clock
            auto f = ds->forces();
            if (!f) return std::unexpected(f.error());
            const double Fx = std::hypot(f->fp[0] + f->fv[0], f->fp[1] + f->fv[1]);
            R.last = std::max(R.last, std::abs(Fx - prevF) / std::max(std::abs(Fx), 1e-300));
            prevF = Fx;
        }
        const bool done = R.last < tol && it > every;
        prev = m;
        if (done || it == maxIter) {
            R.st = std::move(*st);
            break;
        }
    }
    if (fc->forces) {
        auto f = ds->forces();
        if (!f) return std::unexpected(f.error());
        R.fo = *f;
    }
    R.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    R.fm = std::move(*fm);
    return R;
}

// The wall-normal rows of a 2D channel (bottom half and top), area-mean per row.
struct Rows
{
    std::vector<double> y, u, k, w, nut;
};

Rows rowsOf(const Result& R, int ny)
{
    Rows o;
    const std::size_t nC = std::size_t(R.fm.nC);
    const int nx = int(nC) / ny;
    o.y.assign(std::size_t(ny), 0);
    o.u.assign(std::size_t(ny), 0);
    const bool sst = !R.st.s[1].empty();
    if (sst) {
        o.k.assign(std::size_t(ny), 0);
        o.w.assign(std::size_t(ny), 0);
    }
    if (!R.st.nut.empty()) o.nut.assign(std::size_t(ny), 0);
    for (std::size_t c = 0; c < nC; ++c) {
        const std::size_t j = (c / std::size_t(nx)) % std::size_t(ny);
        o.y[j] += R.fm.C[1][c] / nx;
        o.u[j] += R.st.U[c] / nx;
        if (sst) {
            o.k[j] += R.st.s[0][c] / nx;
            o.w[j] += R.st.s[1][c] / nx;
        }
        if (!R.st.nut.empty()) o.nut[j] += R.st.nut[c] / nx;
    }
    return o;
}

std::vector<std::string> channelArgs(int nx, int ny, double reTau, const std::vector<std::string>& more, int iters)
{
    std::vector<std::string> a = {"--body", "none", "--far", "periodic", "--noslip", "--re-tau", std::format("{}", reTau),
                                  "--n", std::to_string(nx), "--ny", std::to_string(ny), "--max-iters", std::to_string(iters)};
    a.insert(a.end(), more.begin(), more.end());
    return a;
}

// V1
void poiseuilleUniform()
{
    std::printf("V1 periodic Poiseuille, uniform mesh, Re_tau 10\n");
    const int ny = 32;
    const FvStructured s = structured(channelArgs(4, ny, 10, {}, 1500));
    auto R = solve(s, fvStructuredTexts(s), 1500, 50, 1e-15);
    if (!R) {
        check(false, R.error());
        return;
    }
    const double nu = 0.1, h = 2.0 / ny;
    const Rows o = rowsOf(*R, ny);
    double d = 0, peak = 0, v = 0;
    for (int j = 0; j < ny; ++j) {
        const double y = o.y[std::size_t(j)], ex = y * (2 - y) / (2 * nu) + h * h / (8 * nu);
        d = std::max(d, std::abs(o.u[std::size_t(j)] - ex));
        peak = std::max(peak, ex);
    }
    for (std::size_t c = 0; c < std::size_t(R->fm.nC); ++c) v = std::max(v, std::abs(R->st.U[std::size_t(R->fm.nC) + c]));
    check(d <= 1e-12 * peak, std::format("u vs FV closed form {:.2e} of peak (it {}), max |v| {:.2e}", d / peak, R->iters, v));
    const double tau = -(R->fo.fp[0] + R->fo.fv[0]) / (2 * 2 * kPi * 1.0);
    check(std::abs(std::abs(tau) - 1) <= 1e-11, std::format("wall stress {:.14f} vs the force balance 1", std::abs(tau)));
}

// Max error of the converged profile vs the exact parabola (at cell centres).
double parabolaError(const std::vector<std::string>& args, int ny, double reTau, int iters, double* tau = nullptr)
{
    const FvStructured s = structured(args);
    auto R = solve(s, fvStructuredTexts(s), iters, 100, 1e-13);
    if (!R) {
        std::printf("    %s\n", R.error().c_str());
        return 1e300;
    }
    const Rows o = rowsOf(*R, ny);
    double d = 0;
    for (int j = 0; j < ny; ++j) {
        const double y = o.y[std::size_t(j)], ex = reTau * y * (2 - y) / 2;
        d = std::max(d, std::abs(o.u[std::size_t(j)] - ex));
    }
    if (tau) *tau = std::abs(R->fo.fp[0] + R->fo.fv[0]) / (2 * 2 * kPi * 1.0);
    return d / (reTau / 2);
}

// V2 (printLadder: the long form)
void poiseuilleMapped(bool longForm)
{
    std::printf("V2 Poiseuille on --stretch and --focal, order from the ladder\n");
    const std::vector<int> ny = longForm ? std::vector<int>{64, 128, 256, 512} : std::vector<int>{64, 128};
    for (const char* kind : {"stretch", "focal"}) {
        std::vector<double> e;
        for (const int n : ny) {
            std::vector<std::string> more = kind == std::string("stretch")
                                                ? std::vector<std::string>{"--stretch", "2"}
                                                : std::vector<std::string>{"--focal", "0,0,0.75,1.25,0,0", "--focal-ratio", "2", "--focal-blend", "1"};
            double tau = 0;
            const int iters = 10000;
            e.push_back(parabolaError(channelArgs(4, n, 10, more, iters), n, 10, iters, &tau));
            std::printf("    %s ny %d: max |u - exact| / peak %.4e, wall stress - 1 = %.2e\n", kind, n, e.back(), tau - 1);
        }
        const double p = std::log2(e[e.size() - 2] / e.back());
        check(p > 1.8 && p < 2.3, std::format("{}: order {:.3f}", kind, p));
    }
}

struct RasRow
{
    double ub, ubOracle, dMax, tau;
};

std::expected<RasRow, std::string> rasChannel(const char* model, double reTau, double beta, int ny, int iters)
{
    const FvStructured s = structured(channelArgs(4, ny, reTau, {"--rans", model, "--stretch", std::format("{}", beta)}, iters));
    auto R = solve(s, fvStructuredTexts(s), iters, 200, 1e-11);
    if (!R) return std::unexpected(R.error());
    const Rows o = rowsOf(*R, ny);
    Channel1dCase cc{.ny = ny, .beta = beta, .reTau = reTau, .mach = 0.05,
                     .model = std::string(model) == "sa-neg" ? RansModel::SaNeg : RansModel::Sst2003m, .tol = ny > 96 ? 1e-9 : 1e-11};
    auto orc = solveChannel1d(cc);
    if (!orc) return std::unexpected("oracle: " + orc.error());
    double ub = 0, d = 0, A = 0, ubO = 0;
    for (std::size_t c = 0; c < std::size_t(R->fm.nC); ++c) {
        ub += R->fm.V[c] * R->st.U[c];
        A += R->fm.V[c];
    }
    ub /= A;
    ubO = orc->ub;
    for (int j = 0; j < ny; ++j) d = std::max(d, std::abs(o.u[std::size_t(j)] - orc->u[std::size_t(j)]));
    const double tau = std::abs(R->fo.fp[0] + R->fo.fv[0]) / (2 * 2 * kPi * 1.0);
    std::printf("    %s Re_tau %g beta %g ny %d: it %d (monitor %.1e, %.1f s), ub %.6f oracle %.6f (%+.3f%%), max |du|/u_c %.3e, "
                "tau_w %.10f\n",
                model, reTau, beta, ny, R->iters, R->last, R->seconds, ub, ubO, 100 * (ub / ubO - 1),
                d / orc->u[std::size_t(ny / 2)], tau);
    return RasRow{ub, ubO, d / orc->u[std::size_t(ny / 2)], tau};
}

// V3
void rasGate()
{
    std::printf("V3 RAS channel vs rans_channel1d, oracle's y mesh\n");
    for (const char* m : {"sa-neg", "sst-2003m"}) {
        auto r = rasChannel(m, 180, 2, 48, 6000);
        if (!r) {
            check(false, std::string(m) + ": " + r.error());
            continue;
        }
        check(std::abs(r->tau - 1) < 1e-8, std::format("{} wall stress balances the force: {:.2e}", m, r->tau - 1));
        check(std::abs(r->ub / r->ubOracle - 1) < 0.05, std::format("{} ub within 5% of the oracle's closure ({:+.2f}%): "
                                                                    "named model differences, docs/fvstructured.md",
                                                                    m, 100 * (r->ub / r->ubOracle - 1)));
    }
}

// V4: a spinning cylinder of radius a inside a fixed noSlip circle R: the MRF
// zone r < D, the wall in it Omega x r, u_theta(r) = A r + B / r exactly.
double couette(int n, int iters, double* uErr = nullptr)
{
    const FvStructured s = structured({"--ogrid", "--body", "cylinder", "--n", std::to_string(n), "--ny", std::to_string(n / 2 + 2),
                                       "--ogrid-far", "4", "--ogrid-beta", "0", "--re", "10", "--rotor-mach", "0.2", "--mach", "0.2",
                                       "--rotor-frame", "mrf", "--rotor-axis", "z", "--max-iters", std::to_string(iters)});
    FoamCaseTexts t = fvStructuredTexts(s);
    // The far circle becomes a resting noSlip wall of the same mesh: a patch
    // type the O-grid does not emit, so the mesh patch stays `patch`.
    auto swap = [](std::string& x, std::string_view a, std::string_view b) {
        const auto at = x.find(a);
        if (at != std::string::npos) x.replace(at, a.size(), b);
    };
    swap(t["0/U"], "type freestreamVelocity; freestreamValue uniform (1 0 0);", "type noSlip;");
    swap(t["0/U"], "internalField   uniform (1 0 0);", "internalField   uniform (0 0 0);");
    swap(t["0/p"], "type freestreamPressure; freestreamValue uniform 0;", "type zeroGradient;");
    auto R = solve(s, t, iters, 200, 1e-13);
    if (!R) {
        std::printf("    %s\n", R.error().c_str());
        return 1e300;
    }
    const double a = 0.5 * kPi / 4.0, Rf = s.ogrid.rFar, Om = s.omega;
    const double B = Om * a * a * Rf * Rf / (Rf * Rf - a * a), Ac = -Om * a * a / (Rf * Rf - a * a);
    double d = 0, e = 0, A = 0;
    const std::size_t nC = std::size_t(R->fm.nC);
    for (std::size_t c = 0; c < nC; ++c) {
        const double x = R->fm.C[0][c], y = R->fm.C[1][c], r = std::hypot(x, y);
        const double ut = Ac * r + B / r, ux = -ut * y / r, uy = ut * x / r;
        const double dd = std::hypot(R->st.U[c] - ux, R->st.U[nC + c] - uy);
        d = std::max(d, dd);
        e += R->fm.V[c] * dd * dd;
        A += R->fm.V[c];
    }
    if (uErr) *uErr = std::sqrt(e / A) / (std::abs(Om) * a);
    return d / (std::abs(Om) * a);
}

void freestreamFocal()
{
    std::printf("V5 uniform stream on a focal box\n");
    const FvStructured s = structured({"--body", "none", "--n", "64", "--ny", "64", "--focal", "2,4,2,4,0,0", "--focal-ratio", "2",
                                       "--re", "100", "--max-iters", "20"});
    auto R = solve(s, fvStructuredTexts(s), 20, 5, 1e-30);
    if (!R) {
        check(false, R.error());
        return;
    }
    double d = 0;
    const std::size_t nC = std::size_t(R->fm.nC);
    for (std::size_t c = 0; c < nC; ++c) d = std::max({d, std::abs(R->st.U[c] - 1), std::abs(R->st.U[nC + c])});
    check(d < 1e-11, std::format("max |U - (1, 0)| = {:.2e} after 20 iterations", d));
}

void determinism()
{
    std::printf("V6 determinism\n");
    const FvStructured s = structured({"--ogrid", "--body", "cylinder", "--n", "48", "--ny", "26", "--re", "40", "--max-iters", "60"});
    auto a = solve(s, fvStructuredTexts(s), 60, 60, 0, fvNumerics(FvPrecision::Fp32));
    auto b = solve(s, fvStructuredTexts(s), 60, 60, 0, fvNumerics(FvPrecision::Fp32));
    check(a && b && a->st.U == b->st.U && a->st.p == b->st.p, "O-grid FP32 runs byte-identical");
}

// `cyl`: cd of the O-grid cylinder ladder at fixed far radius, r_w / r_far grid-similar.
void cylLadder(double rFarD, int iters, const std::vector<int>& ns, const char* re, const char* beta)
{
    std::vector<double> cd;
    for (const int n : ns) {
        const FvStructured s = structured({"--ogrid", "--body", "cylinder", "--n", std::to_string(n), "--ny", std::to_string(n / 2 + 2),
                                           "--ogrid-far", std::format("{}", rFarD), "--ogrid-beta", beta, "--re", re,
                                           "--max-iters", std::to_string(iters)});
        FoamCaseTexts t = fvStructuredTexts(s);
        if (std::getenv("FV_DIV_LINEAR")) { // the central scheme of the exported case, edited
            std::string& sch = t["system/fvSchemes"];
            const std::string a = "bounded Gauss linearUpwind grad(U)";
            sch.replace(sch.find(a), a.size(), "bounded Gauss linear");
        }
        auto R = solve(s, t, iters, 200, 1e-12);
        if (!R) {
            std::printf("n %d: %s\n", n, R.error().c_str());
            return;
        }
        const double q = 0.5 * s.refArea;
        cd.push_back((R->fo.fp[0] + R->fo.fv[0]) / q);
        std::printf("n %d x %d: cd %.8f cdp %.8f cdf %.8f it %d monitor %.1e %.1f s\n", n, n / 2, cd.back(), R->fo.fp[0] / q,
                    R->fo.fv[0] / q, R->iters, R->last, R->seconds);
        if (cd.size() >= 3) {
            const std::size_t k = cd.size();
            const double p = std::log2((cd[k - 3] - cd[k - 2]) / (cd[k - 2] - cd[k - 1]));
            std::printf("  order %.3f, Richardson cd %.8f\n", p, cd[k - 1] + (cd[k - 1] - cd[k - 2]) / (std::pow(2, p) - 1));
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (!gpu::deviceAvailable()) dnstest::skip("no GPU device");
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "poiseuille") {
        poiseuilleMapped(true);
        return finish();
    }
    if (mode == "ras" && argc > 5) {
        for (int i = 5; i < argc; ++i)
            if (auto r = rasChannel(argv[2], std::atof(argv[3]), std::atof(argv[4]), std::atoi(argv[i]), 60000); !r)
                check(false, std::format("ny {}: {}", argv[i], r.error()));
        return finish();
    }
    if (mode == "couette") {
        double prev = 0, prevL2 = 0;
        for (int i = 2; i < argc; ++i) {
            double l2 = 0;
            const double e = couette(std::atoi(argv[i]), 6000, &l2);
            std::printf("n %s: max |U - exact| / U_wall %.4e, L2 %.4e", argv[i], e, l2);
            if (prev > 0) std::printf(", order max %.3f L2 %.3f", std::log2(prev / e), std::log2(prevL2 / l2));
            std::printf("\n");
            prev = e;
            prevL2 = l2;
        }
        return finish();
    }
    if (mode == "cyl" && argc > 6) {
        std::vector<int> ns;
        for (int i = 6; i < argc; ++i) ns.push_back(std::atoi(argv[i]));
        cylLadder(std::atof(argv[2]), std::atoi(argv[3]), ns, argv[4], argv[5]);
        return finish();
    }
    poiseuilleUniform();
    poiseuilleMapped(false);
    rasGate();
    freestreamFocal();
    {
        std::printf("V4 MRF Couette\n");
        double l1 = 0, l2 = 0;
        const double e1 = couette(32, 2500, &l1), e2 = couette(64, 2500, &l2);
        const double p = std::log2(l1 / l2);
        check(l2 < 1e-2 && p > 1.9, std::format("L2 |U - exact| / U_wall {:.3e} -> {:.3e}, order {:.3f} (max {:.3e} -> {:.3e})",
                                                l1, l2, p, e1, e2));
    }
    determinism();
    return finish();
}
