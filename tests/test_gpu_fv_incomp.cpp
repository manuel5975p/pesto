// The time-accurate incompressible FV solver (docs/fvincomp.md), its gates on
// the device. No OpenFOAM here: the analytic 2D Taylor-Green vortex, the
// discrete energy identities and the stored simpleFoam fixed points are the
// oracles. FV vs MAC ladders are longer than a ctest (docs/fvincomp.md).
//   T1  2D TGV, order on the uniform and on the smooth perturbed mesh
//   T2  inviscid KE: the convective rate is 0 to rounding on every mesh
//   T3  projection: |div phi| under the solver's tolerance, FP32 vs FP64
//   T4  rerun byte-identical; 2D w exactly 0; a full case on cyl2d_snappy
//   T5  LES: sigma and Smagorinsky, Cs = 0 bitwise DNS, sigma 0 on 2D
//   T6  simpleFoam's stored fixed point: the march stays near it (t6long: from rest)
//   T7  a short step on a steady state: the memory of c = cb dt, and the end-time rule
#include "dns/foamdict.hpp"
#include "dns/foamio.hpp"
#include "dns/fvincomp.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/runner_fvincomp.hpp"
#include "dns/la/graph.hpp"
#include "dns/polymesh.hpp"

#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <numbers>
#include <string>
#include <vector>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif
#ifndef DNS_FV_DATA
#error "DNS_FV_DATA must point at tests/data/fv"
#endif

using namespace dns;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

constexpr double kPi = std::numbers::pi;

std::vector<double> tgvField(const FaceMesh& fm, bool twoD)
{
    const std::size_t nC = std::size_t(fm.nC);
    std::vector<double> U(3 * nC, 0.0);
    for (std::size_t q = 0; q < nC; ++q) {
        const double x = fm.C[0][q], y = fm.C[1][q], cz = twoD ? 1.0 : std::cos(fm.C[2][q]);
        U[q] = std::sin(x) * std::cos(y) * cz;
        U[nC + q] = -std::cos(x) * std::sin(y) * cz;
    }
    return U;
}

FaceMesh box(int n, int nz, FvPerturb kind, double amp)
{
    const Grid g = Grid::make(n, n, nz, 2 * kPi, 2 * kPi, nz == 1 ? 1.0 : 2 * kPi);
    return *fvPeriodicBox(g, kind, amp);
}

// The L2 error of the 2D TGV at t.
double tgvError(const FaceMesh& fm, const std::vector<double>& U, double nu, double t)
{
    const std::size_t nC = std::size_t(fm.nC);
    const double decay = std::exp(-2.0 * nu * t);
    double e2 = 0, vol = 0;
    for (std::size_t q = 0; q < nC; ++q) {
        const double x = fm.C[0][q], y = fm.C[1][q];
        const double du = U[q] - std::sin(x) * std::cos(y) * decay, dv = U[nC + q] + std::cos(x) * std::sin(y) * decay;
        e2 += fm.V[q] * (du * du + dv * dv);
        vol += fm.V[q];
    }
    return std::sqrt(e2 / vol);
}

struct Device
{
    FvIncompSetup s;
    gpu::DeviceFvIncomp d;
};

std::expected<Device, std::string> device(const FaceMesh& fm, const FvIncompBcs& b, const FvIncompControls& c,
                                          const std::vector<double>& U0)
{
    auto s = buildFvIncompSetup(fm, b);
    if (!s) return std::unexpected(s.error());
    const auto g = la::graphOf(fm);
    auto d = gpu::DeviceFvIncomp::make(*s, g, c, U0);
    if (!d) return std::unexpected(d.error());
    return Device{std::move(*s), std::move(*d)};
}

// March to tEnd; the state. Error text on failure.
std::expected<gpu::FvIncompState, std::string> march(Device& D, double cfl, double tEnd, int* steps = nullptr,
                                                     bool watch = false)
{
    double t = 0;
    int n = 0;
    double k0 = 0;
    while (t < tEnd) {
        auto dt = D.d.stableDt(cfl);
        if (!dt) return std::unexpected(dt.error());
        const double rem = tEnd - t;
        if (rem <= 1e-9 * *dt) break; // arrived: the runner's rule, equal steps to tEnd
        const double h = rem / std::ceil(rem / *dt);
        auto lg = D.d.step(h);
        if (!lg) return std::unexpected(lg.error());
        if (watch && (lg->iterations > 30 || lg->unconverged || n % 100 == 0)) {
            auto dg = D.d.diagnostics();
            const bool jump = k0 > 0 && std::abs(dg->K - k0) > 1e-7 * k0;
            if (lg->iterations > 30 || lg->unconverged || jump)
                std::printf("    step %d: iters %d unconverged %d ratio %.2e K %.10e divMax %.2e\n", n, lg->iterations,
                            lg->unconverged, lg->divRatio, dg->K, dg->divMax);
            k0 = dg->K;
        }
        t += h;
        ++n;
    }
    if (steps) *steps = n;
    return D.d.state();
}

void t1()
{
    std::printf("T1 2D TGV order, nu 0.05, t 1, cfl 0.1, FP64\n");
    const double nu = 0.05;
    FvIncompControls c{.nu = nu, .tol = 1e-9};
    const std::pair<const char*, std::pair<FvPerturb, double>> meshes[] = {
        {"uniform", {FvPerturb::None, 0.0}}, {"sine 0.04", {FvPerturb::Sine, 0.04}}};
    for (const auto& [name, m] : meshes) {
        double err[2] = {0, 0};
        for (int i = 0; i < 2; ++i) {
            const FaceMesh fm = box(32 << i, 1, m.first, m.second);
            auto D = device(fm, {}, c, tgvField(fm, true));
            if (!D) {
                check(false, D.error());
                return;
            }
            auto st = march(*D, 0.1, 1.0);
            if (!st) {
                check(false, st.error());
                return;
            }
            err[i] = tgvError(fm, st->U, nu, 1.0);
        }
        const double p = std::log2(err[0] / err[1]);
        std::printf("  %-10s L2 %.3e %.3e order %.3f\n", name, err[0], err[1], p);
        check(p > 1.8, std::format("{}: order 32 -> 64 > 1.8", name));
    }
}

void t2()
{
    std::printf("T2 inviscid KE\n");
    FvIncompControls c{.nu = 0, .tol = 1e-10};
    auto rm = readPolyMesh(fs::path(DNS_MESH_DATA) / "box_refined" / "constant" / "polyMesh");
    if (!rm) {
        check(false, rm.error());
        return;
    }
    auto refined = FaceMesh::fromPolyMesh(*rm, CellOrder::Morton);
    if (!refined) {
        check(false, refined.error());
        return;
    }
    struct M
    {
        const char* name;
        FaceMesh fm;
    };
    const M meshes[] = {{"uniform 16^2", box(16, 1, FvPerturb::None, 0)},
                        {"random 0.25, 16^2", box(16, 1, FvPerturb::Random, 0.25)},
                        {"sine 0.05, 12^3", box(12, 12, FvPerturb::Sine, 0.05)},
                        {"random 0.15, 12^3", box(12, 12, FvPerturb::Random, 0.15)},
                        {"box_refined", *refined}};
    for (const M& m : meshes) {
        const bool twoD = m.fm.emptyAxis >= 0;
        // box_refined: 1 x 0.75 x 0.5, the field's own period is not the box's.
        std::vector<double> U0 = tgvField(m.fm, twoD);
        if (std::strcmp(m.name, "box_refined") == 0) {
            const std::size_t nC = std::size_t(m.fm.nC);
            const double k = 2 * kPi;
            for (std::size_t q = 0; q < nC; ++q) {
                const double x = m.fm.C[0][q], y = m.fm.C[1][q] / 0.75, z = m.fm.C[2][q] / 0.5;
                U0[q] = std::sin(k * x) * std::cos(k * y) * std::cos(k * z);
                U0[nC + q] = -0.75 * std::cos(k * x) * std::sin(k * y) * std::cos(k * z);
                U0[2 * nC + q] = 0;
            }
        }
        auto D = device(m.fm, {}, c, U0);
        if (!D) {
            check(false, D.error());
            return;
        }
        auto d0 = D->d.diagnostics();
        double dt0 = *D->d.stableDt(0.2);
        double worst = 0, press = 0;
        for (int s = 0; s < 8; ++s) {
            auto lg = D->d.step(dt0);
            if (!lg) {
                check(false, lg.error());
                return;
            }
            auto bu = D->d.budget();
            if (!bu) {
                check(false, bu.error());
                return;
            }
            worst = std::max(worst, std::abs(bu->conv) / bu->grossConv);
            press = std::max(press, std::abs(bu->press));
        }
        auto d1 = D->d.diagnostics();
        const double drift = (d1->K - d0->K) / d0->K;
        std::printf("  %-20s conv/gross %.2e  max|press work|/K %.2e  K drift %.2e over 8 steps\n", m.name, worst,
                    press / d0->K, drift);
        check(worst < 1e-13, std::format("{}: the convective rate is 0 to rounding", m.name));
        check(drift < 1e-2 * std::abs(1.0), std::format("{}: inviscid K bounded", m.name));
    }
}

void t3()
{
    std::printf("T3 divergence vs tolerance\n");
    const FaceMesh fm = box(32, 1, FvPerturb::Sine, 0.04);
    for (const FvPrecision prec : {FvPrecision::Fp64, FvPrecision::Fp32}) {
        for (const double tol : {1e-3, 1e-5, 1e-8}) {
            FvIncompControls c{.nu = 0.01, .tol = tol, .precision = prec};
            auto D = device(fm, {}, c, tgvField(fm, true));
            if (!D) {
                check(false, D.error());
                return;
            }
            double ratio = 0;
            int iters = 0;
            for (int s = 0; s < 10; ++s) {
                auto dt = D->d.stableDt(0.5);
                auto lg = D->d.step(*dt);
                if (!lg) {
                    check(false, lg.error());
                    return;
                }
                ratio = std::max(ratio, lg->divRatio);
                iters += lg->iterations;
            }
            auto dg = D->d.diagnostics();
            std::printf("  %s tol %.0e: est |r|/|b| %.2e, true max|div| %.2e rms %.2e, %d PCG iterations\n",
                        fvPrecisionName(prec), tol, ratio, dg->divMax, dg->divRms, iters);
            if (prec == FvPrecision::Fp64)
                check(ratio <= tol * (1 + 1e-9) || tol < 1e-12, std::format("FP64: |r|/|b| <= tol at {:.0e}", tol));
            else
                check(dg->divMax < 1e-3, "FP32: max|div phi| < 1e-3");
        }
    }
}

void t4()
{
    std::printf("T4 cyl2d_snappy end to end, bytes, 2D\n");
    auto pm = readPolyMesh(fs::path(DNS_MESH_DATA) / "cyl2d_snappy" / "constant" / "polyMesh");
    if (!pm) {
        check(false, pm.error());
        return;
    }
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Morton);
    if (!fm) {
        check(false, fm.error());
        return;
    }
    auto b = parseFvIncompBcs(R"(
[reference]
u = 1.0
length = 1.0
area = 0.1
[initial]
u = [1, 0, 0]
[patch.inlet]
kind = "inflow"
u = [1, 0, 0]
[patch.outlet]
kind = "convective"
uc = 1.0
[patch.lower]
kind = "slip"
[patch.upper]
kind = "far"
u = [1, 0, 0]
[patch.cylinder]
kind = "wall"
omega = [0, 0, 0.2]
origin = [0, 0, 0]
force = true
)");
    check(b.has_value(), "the table parses");
    if (!b) return;
    if (auto e = bindFvIncompBcs(*b, *fm); !e) {
        check(false, e.error());
        return;
    }
    const std::size_t nC = std::size_t(fm->nC);
    std::vector<double> U0(3 * nC, 0.0);
    std::fill_n(U0.begin(), nC, 1.0);
    const FvIncompControls c{.nu = 1.0 / 40.0, .tol = 1e-6, .precision = FvPrecision::Fp64};
    std::vector<double> ref[2];
    double cd = 0;
    for (int i = 0; i < 2; ++i) {
        auto D = device(*fm, *b, c, U0);
        if (!D) {
            check(false, D.error());
            return;
        }
        int n = 0;
        auto st = march(*D, 0.5, 1.0, &n);
        if (!st) {
            check(false, st.error());
            return;
        }
        ref[i] = st->U;
        ref[i].insert(ref[i].end(), st->p.begin(), st->p.end());
        ref[i].insert(ref[i].end(), st->phi.begin(), st->phi.end());
        auto f = D->d.forces();
        if (!f) {
            check(false, f.error());
            return;
        }
        cd = (f->fp[0] + f->fv[0]) / (0.5 * b->area);
        if (i == 0) {
            auto dg = D->d.diagnostics();
            std::printf("  %d steps, cd %.4f, K %.5g, max|div| %.2e, nonOrth %.1f deg\n", n, cd, dg->K, dg->divMax,
                        D->s.nonOrthMaxDeg);
            bool w0 = true;
            for (std::size_t q = 0; q < nC; ++q) w0 = w0 && st->U[2 * nC + q] == 0.0;
            check(w0, "2D: w exactly 0 after the run");
            check(std::isfinite(cd) && cd > 0 && cd < 100, "a finite positive drag");
        }
    }
    check(ref[0].size() == ref[1].size() &&
              std::memcmp(ref[0].data(), ref[1].data(), ref[0].size() * sizeof(double)) == 0,
          "rerun byte-identical (U, p, phi)");
}

void t5()
{
    std::printf("T5 LES\n");
    const FaceMesh fm = box(16, 16, FvPerturb::None, 0);
    const auto U0 = tgvField(fm, false);
    double k[3] = {0, 0, 0};
    const FvIncompControls cs[3] = {{.nu = 1.0 / 1600, .tol = 1e-6},
                                    {.nu = 1.0 / 1600, .tol = 1e-6, .les = 0.17},
                                    {.nu = 1.0 / 1600, .tol = 1e-6, .les = 1.35, .sgs = SgsModel::Sigma}};
    for (int i = 0; i < 3; ++i) {
        auto D = device(fm, {}, cs[i], U0);
        if (!D) {
            check(false, D.error());
            return;
        }
        auto st = march(*D, 0.5, 3.0);
        if (!st) {
            check(false, st.error());
            return;
        }
        k[i] = D->d.diagnostics()->K;
    }
    std::printf("  K(t = 3): DNS %.6f  Smagorinsky %.6f  sigma %.6f\n", k[0], k[1], k[2]);
    check(k[1] < k[0], "Smagorinsky dissipates more than the molecular run");
    check(k[2] < k[0], "sigma dissipates more than the molecular run");
    check(k[2] > k[1], "sigma is smaller than Smagorinsky in the laminar TGV start");
    // A 2D state: D_sigma is exactly 0, so sigma LES is the molecular run bit for bit.
    const FaceMesh f2 = box(16, 1, FvPerturb::Random, 0.1);
    const auto U2 = tgvField(f2, true);
    FvIncompControls a{.nu = 0.01, .tol = 1e-7}, l = a;
    l.les = 1.35;
    l.sgs = SgsModel::Sigma;
    auto Da = device(f2, {}, a, U2), Dl = device(f2, {}, l, U2);
    if (!Da || !Dl) {
        check(false, "2D device");
        return;
    }
    auto sa = march(*Da, 0.5, 0.5), sl = march(*Dl, 0.5, 0.5);
    check(sa && sl && sa->U == sl->U, "2D sigma LES is the molecular run bit for bit");
}

// U vs simpleFoam, in-plane components: max and rms / max|U|, where the max sits.
struct FoamDiff
{
    double max = 0, rms = 0;
    double x = 0, y = 0;
};

// out(q): cell q left out of max and rms.
template <class Out>
FoamDiff foamDiff(const FaceMesh& fm, const std::vector<double>& U, const std::vector<double>& ref, Out out)
{
    const std::size_t nC = std::size_t(fm.nC);
    FoamDiff o;
    double umax = 0, e2 = 0, vol = 0;
    for (std::size_t q = 0; q < nC; ++q) {
        const std::size_t i = std::size_t(fm.srcCell[q]);
        for (std::size_t j = 0; j < 2; ++j) umax = std::max(umax, std::abs(ref[3 * i + j]));
        if (out(q)) continue;
        for (std::size_t j = 0; j < 2; ++j) {
            const double d = std::abs(U[j * nC + q] - ref[3 * i + j]);
            if (d > o.max) {
                o.max = d;
                o.x = fm.C[0][q];
                o.y = fm.C[1][q];
            }
            e2 += fm.V[q] * d * d;
        }
        vol += fm.V[q];
    }
    o.max /= umax;
    o.rms = std::sqrt(e2 / vol) / umax;
    return o;
}

FoamDiff foamDiff(const FaceMesh& fm, const std::vector<double>& U, const std::vector<double>& ref)
{
    return foamDiff(fm, U, ref, [](std::size_t) { return false; });
}

// The cells within k rings of a lid corner of the square cavity [0, L]^2.
bool nearLidCorner(const FaceMesh& fm, std::size_t q, double L, double h, int k)
{
    const double x = fm.C[0][q], y = fm.C[1][q];
    return (L - y) < k * h && (x < k * h || L - x < k * h);
}

// A cavity case at any N: simpleFoam's fixed point (`dns flow --polymesh`, its
// U file) as the start, the march's fixed point vs it, per corner ring excluded.
void cavityLadder(const fs::path& dir, const fs::path& refU, double tEnd, double cfl)
{
    auto pm = readPolyMesh(dir / "constant" / "polyMesh");
    if (!pm) {
        check(false, pm.error());
        return;
    }
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Source);
    if (!fm) {
        check(false, fm.error());
        return;
    }
    auto b = parseFvIncompBcs("[patch.movingWall]\nkind = \"wall\"\nu = [1, 0, 0]\n[patch.fixedWalls]\nkind = \"wall\"\n");
    if (auto e = bindFvIncompBcs(*b, *fm); !e) {
        check(false, e.error());
        return;
    }
    const std::size_t nC = std::size_t(fm->nC);
    auto ref = readFoamInternalField(refU, 3, std::int64_t(nC));
    if (!ref) {
        check(false, ref.error());
        return;
    }
    std::vector<double> U0(3 * nC);
    for (std::size_t q = 0; q < nC; ++q)
        for (std::size_t j = 0; j < 3; ++j) U0[j * nC + q] = (*ref)[3 * std::size_t(fm->srcCell[q]) + j];
    const double L = 0.1, h = L / std::sqrt(double(nC));
    auto D = device(*fm, *b, FvIncompControls{.nu = 1e-3, .tol = 1e-7}, U0);
    if (!D) {
        check(false, D.error());
        return;
    }
    std::vector<double> prev;
    for (int part = 0; part < 4; ++part) {
        int steps = 0;
        auto st = march(*D, cfl, tEnd / 4, &steps);
        if (!st) {
            check(false, st.error());
            return;
        }
        double move = 0;
        for (std::size_t i = 0; i < prev.size(); ++i) move = std::max(move, std::abs(st->U[i] - prev[i]));
        prev = st->U;
        std::printf("  N %g t %g (%d steps) moved %.3e:", L / h, tEnd * (part + 1) / 4, steps, move);
        for (const int k : {0, 1, 2, 4, 8}) {
            const FoamDiff d = foamDiff(*fm, st->U, *ref, [&](std::size_t q) { return nearLidCorner(*fm, q, L, h, k); });
            std::printf("  k%d max %.3e rms %.3e", k, d.max, d.rms);
        }
        std::printf("\n");
    }
}

// Same mesh, same BCs: simpleFoam's fixed point vs the march's, started on it.
// The force three ways: the solver's tally (p_b = p_P + G.r), zero-gradient p
// (simpleFoam's), and the viscous part alone. Host arithmetic on the state.
void fixedPoint(const fs::path& dir, const fs::path& bcFile, const fs::path& refU, double nu, double tEnd, double cfl,
                double tol)
{
    auto pm = readPolyMesh(dir / "constant" / "polyMesh");
    if (!pm) {
        check(false, pm.error());
        return;
    }
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Morton);
    if (!fm) {
        check(false, fm.error());
        return;
    }
    std::ifstream in(bcFile);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto b = parseFvIncompBcs(text);
    if (!b) {
        check(false, b.error());
        return;
    }
    if (auto e = bindFvIncompBcs(*b, *fm); !e) {
        check(false, e.error());
        return;
    }
    const std::size_t nC = std::size_t(fm->nC);
    auto ref = readFoamInternalField(refU, 3, std::int64_t(nC));
    if (!ref) {
        check(false, ref.error());
        return;
    }
    std::vector<double> U0(3 * nC);
    for (std::size_t q = 0; q < nC; ++q)
        for (std::size_t j = 0; j < 3; ++j) U0[j * nC + q] = (*ref)[3 * std::size_t(fm->srcCell[q]) + j];
    const FvIncompControls c{.nu = nu, .tol = tol};
    auto D = device(*fm, *b, c, U0);
    if (!D) {
        check(false, D.error());
        return;
    }
    for (const auto& pa : fm->patches) {
        double smin = 1e300, smax = -1e300;
        for (int i = 0; i < pa.count; ++i) {
            const double sx = fm->S[0][std::size_t(fm->nI) + std::size_t(pa.first + i)];
            smin = std::min(smin, sx);
            smax = std::max(smax, sx);
        }
        std::printf("  patch %-12s %6d faces, S.x in [%.4g, %.4g]\n", pa.name.c_str(), pa.count, smin, smax);
    }
    const double q0 = 0.5 * b->uRef * b->uRef * b->area;
    const auto& S = D->s;
    const std::size_t nB = std::size_t(S.nB);
    std::printf("  nonOrth %.2f, cells %zu, dt %.3e\n", S.nonOrthMaxDeg, nC, *D->d.stableDt(cfl));
    std::vector<double> prev;
    const int parts = 4;
    for (int part = 0; part < parts; ++part) {
        int steps = 0;
        auto st = march(*D, cfl, tEnd / parts, &steps);
        if (!st) {
            check(false, st.error());
            return;
        }
        double move = 0;
        for (std::size_t i = 0; i < prev.size(); ++i) move = std::max(move, std::abs(st->U[i] - prev[i]));
        prev = st->U;
        auto f = D->d.forces();
        // The control volume: inflow of x momentum + pressure over every other
        // boundary (Fx = -sum(phi u_x + p S_x)); p_b on an open face, else the cell's.
        double fp0 = 0, fcv = 0;
        for (std::size_t i = 0; i < nB; ++i) {
            const double pc = st->p[std::size_t(S.bOwn[i])];
            if (S.bForce[i]) {
                fp0 += pc * S.bS[i];
                continue;
            }
            const double pb = S.bCode[i] == std::uint8_t(FvIncompFaceBc::Outflow) ? S.bP[i] : pc;
            fcv -= st->phiB[i] * st->Ub[i] + pb * S.bS[i];
        }
        const FoamDiff d = foamDiff(*fm, st->U, *ref);
        std::printf("  t %g (%d steps) moved %.3e  cd %.6f  (cdp %.6f, cdf %.6f)  cd(cv) %.6f  |U - simple| max %.3e at "
                    "(%.4g, %.4g) rms %.3e\n",
                    tEnd * (part + 1) / parts, steps, move, (f->fp[0] + f->fv[0]) / q0, f->fp[0] / q0, f->fv[0] / q0,
                    fcv / q0, d.max, d.x, d.y, d.rms);
    }
}

// fromRef: start ON simpleFoam's fixed point and report how far the march walks
// off it in tEnds (the ctest form). Else march from rest: the plain comparison,
// tEnds a list of report times, longer than a ctest.
void t6(bool fromRef, bool watch = false)
{
    std::printf("T6 steady fixed point vs simpleFoam (stored), %s\n", fromRef ? "from the stored field" : "from rest");
    struct Case
    {
        const char* name;
        const char* toml;
        double nu, cfl;
        std::vector<double> tEnds;
    };
    const Case cases[] = {
        {"cavity", R"(
[patch.movingWall]
kind = "wall"
u = [1, 0, 0]
[patch.fixedWalls]
kind = "wall"
)",
         1e-3, 0.6, fromRef ? std::vector<double>{0.5} : watch ? std::vector<double>{30, 60, 120} : std::vector<double>{30, 60, 120}},
        {"channel", R"(
[initial]
u = [1, 0, 0]
[patch.inlet]
kind = "inflow"
u = [1, 0, 0]
[patch.outlet]
kind = "outflow"
[patch.floor]
kind = "wall"
[patch.lid]
kind = "slip"
)",
         0.002, 0.6, fromRef ? std::vector<double>{0.5} : std::vector<double>{15, 30, 60}}};
    for (const Case& k : cases) {
        const fs::path dir = fs::path(DNS_FV_DATA) / k.name;
        auto pm = readPolyMesh(dir / "constant" / "polyMesh");
        if (!pm) {
            check(false, pm.error());
            return;
        }
        auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Source);
        if (!fm) {
            check(false, fm.error());
            return;
        }
        auto b = parseFvIncompBcs(k.toml);
        if (!b) {
            check(false, b.error());
            return;
        }
        if (auto e = bindFvIncompBcs(*b, *fm); !e) {
            check(false, e.error());
            return;
        }
        const std::size_t nC = std::size_t(fm->nC);
        auto ref = readFoamInternalField(dir / "ref" / "400" / "U", 3, std::int64_t(nC));
        if (!ref) {
            check(false, ref.error());
            return;
        }
        std::vector<double> U0(3 * nC, 0.0);
        std::fill_n(U0.begin(), nC, b->u0[0]);
        if (fromRef)
            for (std::size_t q = 0; q < nC; ++q)
                for (std::size_t j = 0; j < 3; ++j) U0[j * nC + q] = (*ref)[3 * std::size_t(fm->srcCell[q]) + j];
        const FvIncompControls c{.nu = k.nu, .tol = 1e-7};
        auto D = device(*fm, *b, c, U0);
        if (!D) {
            check(false, D.error());
            return;
        }
        double t = 0;
        FoamDiff last;
        std::vector<double> prev;
        for (const double tEnd : k.tEnds) {
            int steps = 0;
            auto st = march(*D, k.cfl, tEnd - t, &steps, watch);
            if (!st) {
                check(false, st.error());
                return;
            }
            t = tEnd;
            last = foamDiff(*fm, st->U, *ref);
            double move = 0;
            for (std::size_t i = 0; i < prev.size(); ++i) move = std::max(move, std::abs(st->U[i] - prev[i]));
            prev = st->U;
            std::printf("  %-8s t %-5g (%d steps): |U - simpleFoam| / max|U| max %.3e at (%.4g, %.4g), rms %.3e, moved %.3e\n",
                        k.name, t, steps, last.max, last.x, last.y, last.rms, move);
            std::fflush(stdout);
        }
        if (!fromRef) continue;
        // The lid corners are a velocity jump: pressure singular, every cell there a
        // scheme number that does not shrink (6.1e-2, 5.4e-2, 5.1e-2 at N 24, 48, 96),
        // while the same physical region out shrinks at order 1.8 (docs/fvincomp.md).
        // So the gate reads outside two cells of them.
        const bool cavity = std::strcmp(k.name, "cavity") == 0;
        const double L = 0.1, h = L / std::sqrt(double(nC));
        auto st = D->d.state();
        const FoamDiff out =
            foamDiff(*fm, st->U, *ref, [&](std::size_t q) { return cavity && nearLidCorner(*fm, q, L, h, 2); });
        std::printf("  %-8s the gate's cells: max %.3e rms %.3e (all cells: max %.3e rms %.3e)\n", k.name, out.max, out.rms,
                    last.max, last.rms);
        check(out.max < 0.05, std::format("{}: max within the scheme difference", k.name));
    }
}

// Issue 79c: a step much shorter than the last one, at a steady state. The
// approximate projection's fixed point carries c = cb dt, so a step moves u by
// (c_prev - c) G A^-1 div(linear(G) - fp), bounded by c_prev, whatever the shortness.
void t7()
{
    std::printf("T7 a short step on the cavity's steady state\n");
    const fs::path dir = fs::path(DNS_FV_DATA) / "cavity";
    auto pm = readPolyMesh(dir / "constant" / "polyMesh");
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Source);
    auto b = parseFvIncompBcs("[patch.movingWall]\nkind = \"wall\"\nu = [1, 0, 0]\n[patch.fixedWalls]\nkind = \"wall\"\n");
    if (auto e = bindFvIncompBcs(*b, *fm); !e) {
        check(false, e.error());
        return;
    }
    const std::size_t nC = std::size_t(fm->nC);
    auto D = device(*fm, *b, FvIncompControls{.nu = 1e-3, .tol = 1e-7}, std::vector<double>(3 * nC, 0.0));
    if (!D) {
        check(false, D.error());
        return;
    }
    auto settle = [&](double dt, int n) {
        for (int s = 0; s < n; ++s)
            if (auto lg = D->d.step(dt); !lg) return false;
        return true;
    };
    const double dt = *D->d.stableDt(0.6);
    if (!settle(dt, 6000)) return;
    auto moved = [&](const std::vector<double>& a, const std::vector<double>& c) {
        double m = 0;
        for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - c[i]));
        return m;
    };
    auto base = D->d.state();
    settle(dt, 200);
    auto again = D->d.state();
    const double flat = moved(base->U, again->U);
    std::printf("  dt %.3e, t %.3g: 200 equal steps move U %.3e\n", dt, 6200 * dt, flat);
    check(flat < 1e-9, "equal steps hold the steady state");
    double full = 0, half = 0;
    for (const double f : {0.5, 0.1, 1e-3, 1e-6}) {
        auto a = D->d.state();
        D->d.step(f * dt);
        auto c = D->d.state();
        const double mv = moved(a->U, c->U);
        settle(dt, 400);
        auto e = D->d.state();
        std::printf("  a step of %.0e dt moves U %.3e; 400 steps later U is %.3e off\n", f, mv, moved(a->U, e->U));
        full = f < 2e-6 ? mv : full;
        half = f == 0.5 ? mv : half;
        check(moved(a->U, e->U) < 1e-6, "the state returns after a short step");
    }
    check(full > 1e-2 && half < full, "a short step moves U, bounded as its shortness grows: the memory of c");
    // The end-time rule (runner, march): equal steps, 200.37 dt.
    auto a = D->d.state();
    auto st = march(*D, 0.6, 200.37 * dt);
    if (!st) {
        check(false, st.error());
        return;
    }
    // 201 steps of f dt, f = 200.37 / 201: the move is (1 - f) of the short step's.
    const double bound = (1.0 - 200.37 / 201.0) * full;
    std::printf("  equal steps to a non-multiple end time move U %.3e, (1 - f) full %.3e\n", moved(a->U, st->U), bound);
    check(moved(a->U, st->U) < 1.5 * bound, "an end time in equal steps moves U by (1 - f) of the short step's move");
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (!gpu::deviceAvailable()) dnstest::skip("no GPU device");
    const std::string mode = argc > 1 ? argv[1] : "all";
    if (mode == "all" || mode == "t1") t1();
    if (mode == "all" || mode == "t2") t2();
    if (mode == "all" || mode == "t3") t3();
    if (mode == "all" || mode == "t4") t4();
    if (mode == "all" || mode == "t5") t5();
    if (mode == "all" || mode == "t6") t6(true);
    if (mode == "t6long") t6(false, true);
    if (mode == "all" || mode == "t7") t7();
    if (mode == "fixedpoint" && argc >= 8)
        fixedPoint(argv[2], argv[3], argv[4], std::atof(argv[5]), std::atof(argv[6]), std::atof(argv[7]),
                   argc > 8 ? std::atof(argv[8]) : 1e-7);
    if (mode == "cavity" && argc == 6) cavityLadder(argv[2], argv[3], std::atof(argv[4]), std::atof(argv[5]));
    return dnstest::finish();
}
