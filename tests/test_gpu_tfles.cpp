// LES species and the thickened flame of `dns flame` (docs/chem.md, "LES
// species and TFLES"; dns/tfles.hpp, gpu/tfles.hpp).
//
//   bits     (binary) --tfles 1 (uniform, and dynamic) == no --tfles bitwise:
//            1D, and a 3D turbulent box (--turb 4) for ~200 steps -- frames
//            (FP32 binary) byte for byte, stats on every common column;
//            a 3D planar laminar LES + TFLES flame with --efficiency
//            charlette == none byte for byte (curl u == 0 -> u' 0 -> E 1).
//   sensor   (API) the dynamic sensor on the thickened oracle (F 4 at 1.875
//            cells, F 2 at 3 cells): Omega >= 0.99 where c in [0.01, 0.99]
//            and <= 0.01 past 2 F delta_th beyond either end; the separable
//            max filter == a brute-force host max bitwise (3D, odd sizes,
//            both the windowed and the whole-line branch).
//   species  (API) gpu_flame_les_species: (a) a uniform power-of-two Y stays
//            exactly Y rho under the Eddy and Thick kernels; (b) mass, rho Y
//            and E conserved in a periodic box (FP64 sums, rounding level);
//            (c) mutCopyOpenXHook: mu_t on the open x rows == their
//            neighbours bitwise and <= the interior max (raw value reported).
//   teeth    (API) Shvab-Zeldovich teeth: a periodic non-reacting hot/cold
//            box stirred by a two-scale ABC flow, Smagorinsky Cs 2: max
//            |T/Tu + (sigma-1) Y - sigma| with Sc_t = Pr_t <= the molecular
//            DNS's own (MUSCL/central scheme) floor, and >= 10x larger with
//            the TEST-ONLY Sc_t = 0.45; TFLES + LES within 2x the floor
//            (the plan's absolute 1e-2 is printed, not gated: the floor
//            alone is 1.6e-2).
//   lam      (binary) gpu_tfles_1d: laminar TFLES. L1a/L1b: 1.875 cells,
//            F 4, uniform / dynamic sensor, tt 80, window [40, 80]; L1c: the
//            DNS at 7.5 cells in the F-scaled box, tt 20, window [10, 20]:
//            |S_L(L1a) - S_L(L1c)| <= 2e-3, |S_L(L1b)/S_L(L1a) - 1| <= 1%.
//            L2: L1b in 3D (nz = ny = 16) with LES sigma + Charlette:
//            |S_L - S_L(L1b)| <= 2e-4; the Smagorinsky twin's shift reported.
//   dl       (binary) gpu_tfles_dl: thickening stretches the Darrieus-Landau
//            dispersion relation by F, s_F(k) = s_1(F k) / F (docs/chem.md,
//            "Why TFLES misses the U2 DNS"; issues.md L8). One 2D wrinkle
//            mode at sigma 5 (tt 20, frames every 0.5): the U2 box width
//            (15 delta_th = 24 delta_D, about 1.15 lambda_c) GROWS in the DNS
//            at 4 cells and DECAYS under TFLES F 2, while twice that width at
//            F 2 grows at half the DNS rate. The rate s is the slope of ln a
//            over t in [4, 16], a = mode 1 of x_I(y) = sum_i Y (FP64 DFT),
//            as tools/chem/wrinkle_rate.py.
//
// Usage: test_gpu_tfles bits|sensor|species|teeth|lam|dl. No device -> 77.
#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/flame1d.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/chem.hpp"
#include "dns/gpu/tfles.hpp"
#include "dns/tfles.hpp"
#include "dns/transport.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <numbers>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

using namespace dns;
using dnstest::check;

namespace {

Case caseOf(std::vector<std::string> args)
{
    auto c = dnstest::runCli(std::move(args));
    if (!c) {
        std::printf("case rejected: %s\n", c.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return *c;
}

// The runner's TflesSetup of a validated case (apps/dns_main.cpp).
TflesSetup tflesOf(const Case& c)
{
    TflesSetup s;
    s.F = c.flameTfles;
    s.uniform = c.flameTflesUniform;
    s.charlette = c.flameCharlette;
    s.deltaL = c.flameEig.dthOverDD;
    s.nuU = c.prandtl;
    const double thetaS = kTflesTaFraction * c.flameModel.theta();
    s.TaS = thetaS * c.flameTu;
    s.lnAs = -tflesSensorLogMax(c.flameModel.sigma, thetaS);
    s.dilate = c.flameTflesDilate;
    return s;
}

std::string slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

// --- bits --------------------------------------------------------------------

std::string runDns(const std::string& name, const std::string& args)
{
    const std::string prefix = "/tmp/dns_test_tfles_" + name;
    const std::string cmd = "\"" DNS_BIN "\" flame " + args + " --quiet --out " +
                            prefix;
    const int rc = valtest::runBinary(cmd);
    check(rc == 0, name + ": run exits 0");
    return prefix;
}

// a vs b: every frame/mask .bin byte for byte, and every stats column of a
// (the reference, without TFLES columns) equal in b as text.
void sameRun(const std::string& what, const std::string& a,
             const std::string& b)
{
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(a).parent_path();
    const std::string sa = fs::path(a).filename().string() + "_";
    int nBin = 0;
    bool binSame = true;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string fn = e.path().filename().string();
        if (fn.rfind(sa, 0) != 0 || e.path().extension() != ".bin") continue;
        const std::string other = b + "_" + fn.substr(sa.size());
        const bool same = slurp(e.path().string()) == slurp(other);
        if (!same) std::printf("    differs: %s\n", fn.c_str());
        binSame = binSame && same;
        ++nBin;
    }
    const auto ta = valtest::loadCsv(a + "_stats.csv");
    const auto tb = valtest::loadCsv(b + "_stats.csv");
    bool colsSame = ta && tb && ta->rows() == tb->rows() && ta->rows() > 0;
    if (colsSame)
        for (const auto& n : ta->names) colsSame = colsSame && ta->col(n) == tb->col(n);
    std::printf("  %s: %d .bin files compared, %zu stats rows\n", what.c_str(),
                nBin, ta ? ta->rows() : std::size_t(0));
    check(nBin > 0 && binSame, what + ": frames byte for byte");
    check(colsSame, what + ": every common stats column equal");
}

void bits()
{
    std::printf("bits: --tfles 1 == no --tfles\n");
    const std::string d1 = "--cells-per-delta 8 --mach 0.02 --tt 2 "
                           "--inflow-speed 0.98 --ny 4 --emit stats --emit bin "
                           "--fdt 0.5";
    const auto a = runDns("b1_ref", d1);
    const auto b = runDns("b1_uni", d1 + " --tfles 1 --tfles-sensor uniform");
    const auto c = runDns("b1_dyn", d1 + " --tfles 1");
    sameRun("1D uniform F 1", a, b);
    sameRun("1D dynamic F 1", a, c);
    const std::string d3 = "--nz 64 --ny 64 --cells-per-delta 4 --mach 0.02 "
                           "--turb 4 --turb-length 2.5 --tt 0.3 --emit stats "
                           "--emit volume --fdt 0.15";
    const auto e = runDns("b3_ref", d3);
    const auto f = runDns("b3_uni", d3 + " --tfles 1 --tfles-sensor uniform");
    sameRun("3D turbulent uniform F 1", e, f);
    std::printf("bits: planar laminar LES + TFLES, charlette == none\n");
    const std::string p3 = "--nz 16 --ny 16 --cells-per-delta 2 --tfles 4 "
                           "--upstream 30 --downstream 20 --les 1.35 --sgs "
                           "sigma --tt 3 --emit stats --emit volume --fdt 1.5";
    const auto g = runDns("p3_none", p3 + " --efficiency none");
    const auto h = runDns("p3_ch", p3 + " --efficiency charlette");
    sameRun("planar 3D charlette vs none", g, h);
}

// --- sensor ------------------------------------------------------------------

void sensorCase(const std::vector<std::string>& args)
{
    const Case c = caseOf(args);
    const Grid& g = c.grid;
    const double Tu = c.flameTu, sig = c.flameModel.sigma, F = c.flameTfles;
    const double dth = c.flameEig.dthOverDD;
    const auto tr = Transport::air(c.prandtl, Tu, c.tinfK, c.prandtl, kGammaAir,
                                   kRgasBox, false);
    auto dev = gpu::DeviceComp::make(g, tr);
    auto ds = gpu::DeviceCompState::make(g.size(), 0, false, 1);
    auto tf = gpu::DeviceTfles::make(g, tflesOf(c));
    if (!dev || !ds || !tf) std::exit(EXIT_FAILURE);
    CompState h = CompState::zeros(g);
    std::vector<double> ry;
    flameIC(c, h, ry);
    if (ds->upload(h) != hipSuccess || ds->uploadSpecies(0, ry) != hipSuccess)
        std::exit(EXIT_FAILURE);
    dev->primitives(*ds);
    tf->update(*ds, dev->u(), dev->v(), dev->w(), dev->T());
    std::vector<float> om(g.size());
    if (tf->omField().download(om.data()) != hipSuccess) std::exit(EXIT_FAILURE);
    // c along the j = 0 row (the IC is y-uniform).
    std::vector<double> cx(g.nx), ox(g.nx);
    for (int i = 0; i < g.nx; ++i) {
        const std::size_t q = g.idx(i, 0, 0);
        const double r = h.rho[q], u = h.mx[q] / r;
        const double p = (kGammaAir - 1) * (h.E[q] - 0.5 * r * u * u);
        cx[std::size_t(i)] = (p / (r * kRgasBox) - Tu) / ((sig - 1) * Tu);
        ox[std::size_t(i)] = om[q];
    }
    double x01 = NAN, x99 = NAN, minIn = 1, maxOut = 0;
    for (int i = 0; i < g.nx; ++i) {
        const double x = (i + 0.5) * g.hx;
        if (std::isnan(x01) && cx[std::size_t(i)] >= 0.01) x01 = x;
        if (std::isnan(x99) && cx[std::size_t(i)] >= 0.99) x99 = x;
    }
    const double far = 2 * F * dth;
    double lo99 = NAN, hi99 = NAN; // the Omega >= 0.99 extent
    for (int i = 0; i < g.nx; ++i) {
        const double x = (i + 0.5) * g.hx, cc = cx[std::size_t(i)];
        const double o = ox[std::size_t(i)];
        if (cc >= 0.01 && cc <= 0.99) minIn = std::min(minIn, o);
        if (x < x01 - far || x > x99 + far) maxOut = std::max(maxOut, o);
        if (o >= 0.99) {
            if (std::isnan(lo99)) lo99 = x;
            hi99 = x;
        }
    }
    // c at the upstream edge of Omega >= 0.99, and the extents in F delta_th.
    double cLo = NAN;
    for (int i = 0; i < g.nx; ++i)
        if ((i + 0.5) * g.hx >= lo99) {
            cLo = cx[std::size_t(i)];
            break;
        }
    std::printf("  F %g, %g cells/delta_th, dilation %d cells: c in [0.01, "
                "0.99] at x [%.3f, %.3f]; Omega >= 0.99 on x [%.3f, %.3f] "
                "(upstream edge at c = %.3g, %.3f F delta_th before c = 0.01; "
                "%.3f F delta_th past c = 0.99)\n",
                F, c.flameCells, c.flameTflesDilate, x01, x99, lo99, hi99, cLo,
                (x01 - lo99) / (F * dth), (hi99 - x99) / (F * dth));
    std::printf("  min Omega where c in [0.01, 0.99] = %.6f; max Omega beyond "
                "2 F delta_th = %.3g\n",
                minIn, maxOut);
    check(minIn >= 0.99, "Omega >= 0.99 wherever c in [0.01, 0.99]");
    check(maxOut <= 0.01, "Omega <= 0.01 at 2 F delta_th beyond either end");
}

// The T-gated separable max filter against a host composition of the same
// three gated line passes (x clamped, y and z periodic), and -- with T
// uniform, where the gate is open -- against the brute-force box max.
void dilationBrute(int r, bool gated)
{
    const Grid g = Grid::make(37, 11, 9, 37.0, 11.0, 9.0);
    TflesSetup s;
    s.F = 2;
    s.dilate = r;
    auto tf = gpu::DeviceTfles::make(g, s);
    auto raw = gpu::DeviceScalar::make(g.size());
    auto tt = gpu::DeviceScalar::make(g.size());
    auto out = gpu::DeviceScalar::make(g.size());
    if (!tf || !raw || !tt || !out) std::exit(EXIT_FAILURE);
    std::mt19937 rng(1234 + r);
    std::uniform_int_distribution<int> d(0, 999);
    std::vector<float> h(g.size()), T(g.size()), o(g.size());
    for (auto& v : h) v = d(rng) < 950 ? 0.0f : float(d(rng)) / 1000.0f;
    for (auto& v : T) v = gated ? float(d(rng) % 50) : 1.0f; // ties included
    if (raw->upload(h.data()) != hipSuccess || tt->upload(T.data()) != hipSuccess)
        std::exit(EXIT_FAILURE);
    tf->dilate(*raw, *tt, *out);
    if (out->download(o.data()) != hipSuccess) std::exit(EXIT_FAILURE);
    // Host: the same passes.
    auto pass = [&](const std::vector<float>& in, int axis) {
        std::vector<float> res(in.size());
        const int n[3] = {g.nx, g.ny, g.nz};
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    const int q[3] = {i, j, k};
                    const std::size_t c = g.idx(i, j, k);
                    float m = in[c];
                    auto take = [&](int p) {
                        int a[3] = {i, j, k};
                        a[axis] = p;
                        const std::size_t e = g.idx(a[0], a[1], a[2]);
                        if (T[e] >= T[c]) m = std::max(m, in[e]);
                    };
                    if (axis == 0) {
                        for (int p = std::max(0, i - r); p <= std::min(g.nx - 1, i + r); ++p)
                            take(p);
                    } else if (2 * r + 1 >= n[axis]) {
                        for (int p = 0; p < n[axis]; ++p) take(p);
                    } else {
                        for (int dd = -r; dd <= r; ++dd)
                            take(((q[axis] + dd) % n[axis] + n[axis]) % n[axis]);
                    }
                    res[c] = m;
                }
        return res;
    };
    const std::vector<float> ref = pass(pass(pass(h, 0), 1), 2);
    long bad = 0, badBox = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                bad += ref[g.idx(i, j, k)] != o[g.idx(i, j, k)];
                if (gated) continue;
                float m = 0;
                for (int a = std::max(0, i - r); a <= std::min(g.nx - 1, i + r); ++a)
                    for (int b = -r; b <= r; ++b)
                        for (int cc = -r; cc <= r; ++cc) {
                            const int jj = ((j + b) % g.ny + g.ny) % g.ny;
                            const int kk = ((k + cc) % g.nz + g.nz) % g.nz;
                            m = std::max(m, h[g.idx(a, jj, kk)]);
                        }
                badBox += m != o[g.idx(i, j, k)];
            }
    std::printf("  radius %d on 37 x 11 x 9, %s: %ld cells differ from the host "
                "passes%s\n",
                r, gated ? "random T (gated)" : "uniform T (gate open)", bad,
                gated ? "" : std::format(", {} from the brute-force box max",
                                         badBox).c_str());
    check(bad == 0, "the device filter == the host gated passes bitwise");
    if (!gated) check(badBox == 0, "uniform T: == the brute-force box max");
}

void sensor()
{
    std::printf("sensor coverage on the thickened oracle\n");
    sensorCase({"flame", "--tfles", "4", "--cells-per-delta", "1.875",
                "--upstream", "30", "--downstream", "20"});
    sensorCase({"flame", "--tfles", "2", "--cells-per-delta", "3",
                "--upstream", "30", "--downstream", "20"});
    std::printf("dilation vs the host passes / brute force\n");
    dilationBrute(3, false);
    dilationBrute(7, false); // 2r + 1 >= ny, nz: the whole-line branch
    dilationBrute(3, true);
    dilationBrute(7, true);
}

// --- the periodic mixing box (species, teeth) ----------------------------------

constexpr double kSig = 5.0;

struct Box
{
    Grid g = Grid::make(1, 1, 1, 1, 1, 1);
    CompState q = CompState::zeros(Grid::make(1, 1, 1, 1, 1, 1));
    std::vector<double> ry;
    double Tu = 1;
};

// Periodic hot/cold box at flame Mach M (Tu = p = 1/(gamma M^2)): T/Tu = 1 +
// (sigma - 1) sin^2(pi x/Lx), Y on the Shvab-Zeldovich line (yUniform < 0)
// or uniform, stirred by two ABC modes (k1 amplitude A, k2 amplitude A/2).
Box mixingBox(int nx, int ny, int nz, double h, double A, double yUniform,
              double M = 0.01)
{
    Box b;
    b.g = Grid::make(nx, ny, nz, nx * h, ny * h, nz * h);
    const Grid& g = b.g;
    b.q = CompState::zeros(g);
    b.ry.assign(g.size(), 0.0);
    b.Tu = 1.0 / (kGammaAir * M * M);
    const double Tu = b.Tu;
    const double kx = 2 * std::numbers::pi / (nx * h),
                 ky = 2 * std::numbers::pi / (ny * h),
                 kz = 2 * std::numbers::pi / (nz * h);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double x = (i + 0.5) * h, y = (j + 0.5) * h,
                             z = (k + 0.5) * h;
                const double s = std::sin(std::numbers::pi * x / (nx * h));
                const double t = 1 + (kSig - 1) * s * s;
                const double rho = 1.0 / t;
                double u = 0, v = 0, w = 0;
                for (int m = 1; m <= 2; ++m) {
                    const double a = A / m;
                    u += a * (std::sin(m * kz * z) + std::cos(m * ky * y));
                    v += a * (std::sin(m * kx * x) + std::cos(m * kz * z));
                    w += a * (std::sin(m * ky * y) + std::cos(m * kx * x));
                }
                const std::size_t c = g.idx(i, j, k);
                b.q.rho[c] = rho;
                b.q.mx[c] = rho * u;
                b.q.my[c] = rho * v;
                b.q.mz[c] = rho * w;
                b.q.E[c] = Tu / (kGammaAir - 1) + 0.5 * rho * (u * u + v * v + w * w);
                const double Y = yUniform >= 0 ? yUniform : (kSig - t) / (kSig - 1);
                b.ry[c] = rho * Y;
            }
    return b;
}

struct BoxRun
{
    CompState q = CompState::zeros(Grid::make(1, 1, 1, 1, 1, 1));
    std::vector<double> ry;
    std::vector<float> nut;
    double defect = 0;        // max |T/Tu + (sigma-1) Y - sigma|
    double sums0[3] = {}, sums1[3] = {}; // rho, rho Y, E (FP64) before / after
    double t = 0;             // simulated time
};

struct BoxOpts
{
    int steps = 100;
    double cs = 1.35;         // 0 = molecular DNS (the Molecular kernel)
    SgsModel sgs = SgsModel::Sigma;
    double tflesF = 0;        // 0 = no TFLES
    double invSct = 0;        // SpeciesSetup::testInvSct
};

BoxRun runBox(const Box& b, const BoxOpts& o)
{
    const Grid& g = b.g;
    const double Tu = b.Tu;
    const auto tr = Transport::air(0.72, Tu, 288.15, 0.72, kGammaAir, kRgasBox,
                                   false);
    auto dev = gpu::DeviceComp::make(g, tr, o.cs, o.sgs);
    auto red = gpu::ReduceScratch::make();
    auto ds = gpu::DeviceCompState::make(g.size(), 0, false, 1);
    if (!dev || !red || !ds) std::exit(EXIT_FAILURE);
    std::optional<gpu::DeviceTfles> tf;
    gpu::SpeciesSetup sp;
    sp.nY = 1;
    sp.lnA = -HUGE_VAL; // no reaction: pure mixing
    sp.Ta = 0;
    sp.q = tr.cp() * (kSig - 1) * Tu;
    sp.invPr = 1.0 / 0.72;
    sp.testInvSct = o.invSct;
    if (o.tflesF > 0) {
        TflesSetup s;
        s.F = o.tflesF;
        s.charlette = o.cs > 0;
        s.deltaL = 1.60165;
        s.nuU = 0.72;
        const double theta = 8.0 * kSig * kSig / (kSig - 1);
        s.TaS = kTflesTaFraction * theta * Tu;
        s.lnAs = -tflesSensorLogMax(kSig, kTflesTaFraction * theta);
        s.dilate = int(std::ceil(kTflesDilate * o.tflesF * 1.60165 / g.hx));
        auto m = gpu::DeviceTfles::make(g, s);
        if (!m) std::exit(EXIT_FAILURE);
        tf = std::move(*m);
        sp.tflesFE = tf->fe();
        sp.tflesEoF = tf->eof();
        sp.tflesOm = tf->om();
    }
    if (!dev->enableSpecies(sp)) std::exit(EXIT_FAILURE);
    if (ds->upload(b.q) != hipSuccess || ds->uploadSpecies(0, b.ry) != hipSuccess)
        std::exit(EXIT_FAILURE);
    BoxRun r;
    auto sums = [&](double* s) {
        CompState q = CompState::zeros(g);
        std::vector<double> ry;
        if (ds->download(q) != hipSuccess || ds->downloadSpecies(0, ry) != hipSuccess)
            std::exit(EXIT_FAILURE);
        s[0] = s[1] = s[2] = 0;
        for (std::size_t c = 0; c < g.size(); ++c) {
            s[0] += q.rho[c];
            s[1] += ry[c];
            s[2] += q.E[c];
        }
        r.q = std::move(q);
        r.ry = std::move(ry);
    };
    sums(r.sums0);
    for (int n = 0; n < o.steps; ++n) {
        if (tf) {
            dev->primitives(*ds);
            tf->update(*ds, dev->u(), dev->v(), dev->w(), dev->T());
        }
        double dt = dev->stableDt(*ds, *red, 0.4);
        if (tf) dt = std::min(dt, gpu::tflesStableDt(*dev, *ds, tf->feField(), *red));
        dev->step(*ds, gpu::Real(dt));
        r.t += dt;
    }
    sums(r.sums1);
    for (std::size_t c = 0; c < g.size(); ++c) {
        const double rho = r.q.rho[c];
        const double ke = 0.5 * (r.q.mx[c] * r.q.mx[c] + r.q.my[c] * r.q.my[c] +
                                 r.q.mz[c] * r.q.mz[c]) / rho;
        const double T = (kGammaAir - 1) * (r.q.E[c] - ke) / (rho * kRgasBox);
        r.defect = std::max(r.defect, std::abs(T / Tu + (kSig - 1) * r.ry[c] / rho -
                                               kSig));
    }
    if (o.cs > 0) {
        dev->updateSgsNut(*ds);
        r.nut.resize(g.size());
        if (dev->mut().download(r.nut.data()) != hipSuccess) std::exit(EXIT_FAILURE);
    }
    return r;
}

double relChange(const BoxRun& r, int q)
{
    return std::abs(r.sums1[q] - r.sums0[q]) / std::abs(r.sums0[q]);
}

// mu_t on the open x rows of a box whose u jumps by 4 across the middle and
// across the periodic wrap (the flame's fresh/burnt jump), raw and imaged.
void openRows(SgsModel sgs, double cs)
{
    const Grid g = Grid::make(32, 8, 8, 32.0, 8.0, 8.0);
    Box b = mixingBox(32, 8, 8, 1.0, 0.2, 0.5);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = g.nx / 2; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double r = b.q.rho[c];
                const double u0 = b.q.mx[c] / r;
                b.q.E[c] += 0.5 * r * ((u0 + 4) * (u0 + 4) - u0 * u0);
                b.q.mx[c] = r * (u0 + 4);
            }
    const auto tr = Transport::air(0.72, b.Tu, 288.15, 0.72, kGammaAir,
                                   kRgasBox, false);
    auto dev = gpu::DeviceComp::make(g, tr, cs, sgs);
    auto ds = gpu::DeviceCompState::make(g.size(), 0, false, 0);
    if (!dev || !ds || ds->upload(b.q) != hipSuccess) std::exit(EXIT_FAILURE);
    std::vector<float> raw(g.size()), img(g.size());
    dev->updateSgsNut(*ds);
    if (dev->mut().download(raw.data()) != hipSuccess) std::exit(EXIT_FAILURE);
    dev->setMutHook(gpu::mutCopyOpenXHook(gpu::GpuGrid::from(g)));
    dev->updateSgsNut(*ds);
    if (dev->mut().download(img.data()) != hipSuccess) std::exit(EXIT_FAILURE);
    double rawEdge = 0, edge = 0, interior = 0;
    bool copy = true;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 1; i < g.nx - 1; ++i)
                interior = std::max(interior, double(img[g.idx(i, j, k)]));
            for (int i : {0, g.nx - 1}) {
                rawEdge = std::max(rawEdge, double(raw[g.idx(i, j, k)]));
                edge = std::max(edge, double(img[g.idx(i, j, k)]));
            }
            copy = copy && img[g.idx(0, j, k)] == img[g.idx(1, j, k)] &&
                   img[g.idx(g.nx - 1, j, k)] == img[g.idx(g.nx - 2, j, k)];
        }
    const std::string nm = sgs == SgsModel::Sigma ? "sigma" : "smag";
    std::printf("  %s: open rows raw (wrap-differenced) max mu_t %.4g, with the "
                "hook %.4g; interior max %.4g\n",
                nm.c_str(), rawEdge, edge, interior);
    check(copy, nm + ": mu_t(0) == mu_t(1), mu_t(nx-1) == mu_t(nx-2) bitwise");
    check(edge <= interior, nm + ": open-row mu_t <= the interior max");
}

void species()
{
    std::printf("(a) uniform Y = 0.5 exact under the Eddy and Thick kernels\n");
    const Box u = mixingBox(32, 16, 16, 1.0, 2.0, 0.5);
    for (double F : {0.0, 3.0}) {
        const BoxRun r = runBox(u, {.steps = 60, .tflesF = F});
        long bad = 0;
        for (std::size_t c = 0; c < u.g.size(); ++c)
            bad += r.ry[c] != 0.5 * r.q.rho[c];
        std::printf("  %s: %ld cells with rho Y != rho / 2\n",
                    F > 0 ? "Thick (LES + TFLES F 3)" : "Eddy (LES)", bad);
        check(bad == 0, std::string(F > 0 ? "Thick" : "Eddy") +
                            ": rho Y == rho / 2 bitwise after 60 steps");
    }
    // The FP32 state does not conserve to the bit (each cell rounds its own
    // update; the mean flow alone drifts ~4e-9 / step, docs: SSP-RK3 ca + cb
    // == 1). The eddy / thickened fluxes are conservative iff they add
    // nothing to the molecular DNS's drift.
    std::printf("(b) conservation in the periodic box, 200 steps, against the "
                "molecular DNS's own FP32 drift\n");
    const Box m = mixingBox(32, 16, 16, 1.0, 2.0, -1);
    const BoxRun dns = runBox(m, {.steps = 200, .cs = 0});
    const char* nm[3] = {"mass", "rho Y", "E"};
    for (double F : {0.0, 3.0}) {
        const BoxRun r = runBox(m, {.steps = 200, .tflesF = F});
        bool ok = true;
        for (int q = 0; q < 3; ++q) {
            std::printf("  %s %s: relative change %.3e (DNS %.3e)\n",
                        F > 0 ? "Thick" : "Eddy", nm[q], relChange(r, q),
                        relChange(dns, q));
            ok = ok && relChange(r, q) <= 2 * relChange(dns, q) + 1e-7;
        }
        check(ok, std::string(F > 0 ? "Thick" : "Eddy") +
                      ": mass, rho Y, E drift <= 2 x the DNS's + 1e-7");
    }
    std::printf("(c) mu_t on the open x rows (mutCopyOpenXHook)\n");
    openRows(SgsModel::Sigma, 1.35);
    openRows(SgsModel::Smagorinsky, 0.17);
}

// The Shvab-Zeldovich defect of a non-reacting stirred hot/cold box: T/Tu +
// (sigma - 1) Y = sigma holds initially and is transported exactly by the
// continuous Le = 1 equations at low Mach (energy and species diffuse alike,
// the eddy fluxes too iff Sc_t = Pr_t). The DISCRETE scheme is not exact: the
// species face value is MC-limited MUSCL, the energy flux central, and the
// limiter clips at the stirred extrema -- the molecular DNS's own defect
// (measured 1.6e-2 here). So the teeth are relative: the eddy fluxes at Sc_t
// = Pr_t add nothing to that floor, at Sc_t 0.45 they add >= 10x. A strong
// Smagorinsky (Cs 2, mean mu_t/mu ~ 14) makes the SGS flux dominate while
// the stirring stays short (300 steps, t ~ 1): the test is of the flux
// coupling, not of the model.
void teeth()
{
    const int steps = 300;
    const Box m = mixingBox(32, 16, 16, 4.0, 2.0, -1, 0.005);
    std::printf("Shvab-Zeldovich teeth (non-reacting periodic box 32 x 16 x 16, "
                "h 4, ABC amplitude 2, M 0.005, %d steps, Smagorinsky Cs 2)\n",
                steps);
    const SgsModel sg = SgsModel::Smagorinsky;
    const BoxRun dns = runBox(m, {.steps = steps, .cs = 0});
    const BoxRun ok = runBox(m, {.steps = steps, .cs = 2, .sgs = sg});
    const BoxRun bad = runBox(m, {.steps = steps, .cs = 2, .sgs = sg,
                                  .invSct = 1.0 / 0.45});
    double nutMax = 0, nutMean = 0;
    for (float v : ok.nut) {
        nutMax = std::max(nutMax, double(v));
        nutMean += v;
    }
    nutMean /= double(ok.nut.size());
    std::printf("  t = %.3g; DNS defect %.3e; LES (Eddy): %.3e at Sc_t = Pr_t, "
                "%.3e at Sc_t 0.45 (x %.1f); mu_t / mu max %.3f, mean %.3f\n",
                ok.t, dns.defect, ok.defect, bad.defect, bad.defect / ok.defect,
                nutMax / 0.72, nutMean / 0.72);
    std::printf("  (the plan's absolute gate, defect <= 1e-2 at Sc_t = Pr_t: "
                "%s -- the DNS floor is %.3e)\n",
                ok.defect <= 1e-2 ? "met" : "NOT met", dns.defect);
    check(ok.defect <= dns.defect, "Sc_t = Pr_t: the eddy fluxes add no defect "
                                   "to the DNS's scheme floor");
    check(bad.defect >= 10 * ok.defect, "Sc_t 0.45: defect >= 10x larger");
    const BoxRun tok = runBox(m, {.steps = steps, .cs = 2, .sgs = sg,
                                  .tflesF = 2});
    const BoxRun tbad = runBox(m, {.steps = steps, .cs = 2, .sgs = sg,
                                   .tflesF = 2, .invSct = 1.0 / 0.45});
    std::printf("  LES + TFLES F 2 (Thick): defect %.3e at Sc_t = Pr_t, %.3e at "
                "Sc_t 0.45 (x %.1f; the SGS species flux is off where Omega "
                "= 1)\n",
                tok.defect, tbad.defect, tbad.defect / tok.defect);
    check(tok.defect <= 2 * dns.defect, "TFLES + LES, Sc_t = Pr_t: defect "
                                        "within 2x the DNS floor");
    check(tbad.defect > tok.defect, "TFLES + LES, Sc_t 0.45: defect larger "
                                    "(where Omega < 1)");
}

// --- lam ---------------------------------------------------------------------

struct Lam
{
    double SL = NAN, SLa = NAN, SLb = NAN, Ebar = NAN;
};

Lam lamRun(const std::string& name, const std::string& args, double t0,
           double t1)
{
    Lam l;
    const std::string prefix = "/tmp/dns_test_tfles_" + name;
    const std::string cmd = "\"" DNS_BIN "\" flame " + args +
                            " --inflow-speed 0.98 --mach 0.01 --emit stats "
                            "--quiet --out " + prefix;
    const int rc = valtest::runBinary(cmd);
    check(rc == 0, name + ": run exits 0");
    if (rc != 0) return l;
    const auto t = valtest::loadCsv(prefix + "_stats.csv");
    if (!t) return l;
    const auto& tc = t->col("t");
    const auto& sl = t->col("SL");
    double sa = 0, sb = 0;
    int na = 0, nb = 0;
    const double tm = 0.5 * (t0 + t1);
    for (std::size_t i = 0; i < tc.size(); ++i) {
        if (tc[i] < t0) continue;
        if (tc[i] < tm) { sa += sl[i]; ++na; }
        else { sb += sl[i]; ++nb; }
    }
    if (na && nb) {
        l.SLa = sa / na;
        l.SLb = sb / nb;
        l.SL = (sa + sb) / (na + nb);
    }
    if (std::ranges::find(t->names, "Ebar") != t->names.end())
        l.Ebar = t->col("Ebar").back();
    std::printf("  %-6s S_L %.6f  halves %.6f / %.6f%s\n", name.c_str(), l.SL,
                l.SLa, l.SLb,
                std::isnan(l.Ebar) ? "" : std::format("  Ebar {:.6f}", l.Ebar).c_str());
    return l;
}

void lam()
{
    std::printf("laminar TFLES (M 0.01, --inflow-speed 0.98)\n");
    const std::string box = "--upstream 30 --downstream 20 --cells-per-delta 1.875 "
                            "--tfles 4";
    const Lam a = lamRun("L1a", box + " --tfles-sensor uniform --tt 80", 40, 80);
    const Lam b = lamRun("L1b", box + " --tt 80", 40, 80);
    const Lam c = lamRun("L1c", "--upstream 7.5 --downstream 5 "
                                "--cells-per-delta 7.5 --tt 20",
                         10, 20);
    std::printf("  S_L(L1a) - S_L(L1c) = %+.3e;  S_L(L1b)/S_L(L1a) - 1 = %+.3e\n",
                a.SL - c.SL, b.SL / a.SL - 1);
    check(std::abs(a.SL - c.SL) <= 2e-3, "|S_L(TFLES F 4, uniform) - S_L(DNS, "
                                         "F-scaled grid)| <= 2e-3");
    check(std::abs(b.SL / a.SL - 1) <= 0.01, "dynamic sensor within 1% of "
                                             "uniform");
    const std::string d3 = box + " --nz 16 --ny 16 --tt 80 --les ";
    const Lam s = lamRun("L2sig", d3 + "1.35 --sgs sigma", 40, 80);
    const Lam m = lamRun("L2smag", d3 + "0.17 --sgs smag", 40, 80);
    std::printf("  3D sigma - L1b = %+.3e;  3D Smagorinsky - L1b = %+.3e "
                "(%+.3f%%)\n",
                s.SL - b.SL, m.SL - b.SL, 100 * (m.SL / b.SL - 1));
    check(std::abs(s.SL - b.SL) <= 2e-4, "3D planar LES sigma + Charlette == "
                                         "the 1D TFLES S_L to 2e-4");
}

// --- dl ----------------------------------------------------------------------

// The number after "key": in the run's flat .json; NaN if absent.
double jsonNumber(const std::string& js, const std::string& key)
{
    const std::string k = "\"" + key + "\":";
    const std::size_t p = js.find(k);
    if (p == std::string::npos) return NAN;
    return std::strtod(js.c_str() + p + k.size(), nullptr);
}

// Growth rate of a single 2D wrinkle mode of ny cells' wavelength.
double wrinkleRate(const std::string& name, const std::string& args, int ny)
{
    const std::string prefix = runDns(
        "dl_" + name, "--upstream 16 --downstream 16 --inflow-speed 1 --tt 20 "
                      "--mach 0.02 --wrinkle 0.5 --wrinkle-modes 1 --emit stats "
                      "--emit bin --fdt 0.5 --ny " +
                          std::to_string(ny) + " " + args);
    const std::string js = slurp(prefix + ".json");
    const double nxd = jsonNumber(js, "nx"), fdt = jsonNumber(js, "frameDt");
    const std::string raw = slurp(prefix + "_fuel.bin");
    if (!(nxd > 0 && fdt > 0)) {
        check(false, name + ": nx and frameDt in the .json");
        return NAN;
    }
    const std::size_t nx = std::size_t(nxd), per = nx * std::size_t(ny);
    const std::size_t nf = raw.size() / (sizeof(float) * per);
    std::vector<float> Y(nf * per);
    std::memcpy(Y.data(), raw.data(), Y.size() * sizeof(float));
    // Least squares of ln a(t) over the frames with t in [4, 16].
    double st = 0, sa = 0, stt = 0, sta = 0, a0 = NAN, aEnd = NAN;
    int n = 0;
    for (std::size_t f = 0; f < nf; ++f) {
        double re = 0, im = 0;
        for (int j = 0; j < ny; ++j) {
            double xi = 0;
            for (std::size_t i = 0; i < nx; ++i)
                xi += Y[f * per + std::size_t(j) * nx + i];
            const double ph = 2 * std::numbers::pi * j / ny;
            re += xi * std::cos(ph);
            im -= xi * std::sin(ph);
        }
        const double a = 2 * std::hypot(re, im) / ny;
        // Frame f is the first step with t >= f fdt (cadence starts at
        // frameStart 0; 41 frames over tt 20 at fdt 0.5).
        const double t = fdt * double(f);
        if (f == 0) a0 = a;
        aEnd = a;
        if (t < 4 || t > 16) continue;
        const double la = std::log(a);
        st += t;
        sa += la;
        stt += t * t;
        sta += t * la;
        ++n;
    }
    const double s = (n * sta - st * sa) / (n * stt - st * st);
    std::printf("  %-6s ny %3d: %zu frames, a(t_end)/a(t_0) = %.3f, s = %+.4f / "
                "tau over [4, 16] (%d frames)\n",
                name.c_str(), ny, nf, aEnd / a0, s, n);
    check(n >= 20, name + ": >= 20 frames in the fit window");
    return s;
}

void dl()
{
    std::printf("Darrieus-Landau under thickening (sigma 5, one wrinkle mode)\n");
    const double dns = wrinkleRate("m_dns4", "--cells-per-delta 4 --nscbc-sink 11",
                                   60);
    const double f2 = wrinkleRate("m_f2", "--cells-per-delta 3 --nscbc-sink 8 "
                                          "--tfles 2",
                                  45);
    const double w2 = wrinkleRate("W_f2", "--cells-per-delta 3 --nscbc-sink 8 "
                                          "--tfles 2",
                                  90);
    const double ratio = w2 / (dns / 2);
    std::printf("  s(W_f2) / (s(m_dns4) / 2) = %.3f (s_F(k) = s_1(F k) / F: 1)\n",
                ratio);
    check(dns >= 0.02, "the DNS's U2-width mode grows (s >= +0.02 / tau)");
    check(f2 <= -0.10, "TFLES F 2 damps the same mode (s <= -0.10 / tau)");
    check(ratio >= 0.6 && ratio <= 1.4,
          "twice the width at F 2 grows at half the DNS rate (ratio in [0.6, 1.4])");
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator("/tmp", ec))
        if (e.path().filename().string().rfind("dns_test_tfles_dl_", 0) == 0)
            fs::remove(e.path(), ec);
}

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "bits") bits();
    else if (mode == "sensor") sensor();
    else if (mode == "species") species();
    else if (mode == "teeth") teeth();
    else if (mode == "lam") lam();
    else if (mode == "dl") dl();
    else {
        std::printf("usage: test_gpu_tfles bits|sensor|species|teeth|lam|dl\n");
        return EXIT_FAILURE;
    }
    return dnstest::finish();
}
