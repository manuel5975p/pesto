// DeviceLowMach through its API (gpu/lowmach.hpp; masterplan real_bunsen S4,
// docs/chem.md "Low-Mach"). Non-reacting (lnA = -inf) variable-density boxes:
// 2D 64 x 32 and 3D 32 x 16 x 16, a hot blob (T 1..4) advected through the
// inflow face at u_in 1 with a swirl of 0.3 in y (not divergence-free: the
// first projection makes it so), 100 steps at the solver's own dt.
//
//   passive  a uniform power-of-two Y (C0: Y 0.5; two reactants: Y_F 0.25,
//            Y_O 0.5, inflow the same) stays EXACTLY Y rho, hi and lo words
//            (rho Y and rho ride one projected flux, D7 face formula);
//   mass     sum (rho hi + lo) dV equals the FP64 face-flux tally to 1e-10
//            (telescoping fluxes, compensated words: what is left is the lo
//            words' own rounding, ~ulp(hi) eps per cell and stage; measured
//            7.9e-12 2D, 4.0e-11 3D);
//   rest     a uniform state at rest (m_in 0) stays bitwise at rest: rho, Y,
//            m, p unchanged over 100 steps of dt 0.05;
//   rerun    two fresh solvers on the same case: every field byte-identical;
//   project  after 100 steps, max |div m - D| / max |D| <= 1e-4 (the FP32
//            floor of the spectral solve, issue 75's 1e-5 - 1e-4 of |b|) and
//            the EOS defect max |rho T* - 1| <= 1e-6.
//   species  (S6) rho Y_k = the FP64 species boundary-flux tally to 1e-10.
//
// Open laterals (S6, LowMachSetup::openY/openZ):
//   passive  the three passive cases with open y (z): uniform Y exact (the
//            ambient the same Y), mass and species = their tallies 1e-10,
//            projection, EOS, rerun byte-identical (hi faces included);
//   rest     a quiescent open box (2D open y; 3D open y, z): u == 0, p == 0
//            exactly over 1000 steps;
//   balance  a pure-fuel box with a hot blob and a fuel jet, AIR ambient
//            (2D, 3D, 300 steps): mass, fuel, oxygen = their boundary
//            tallies to 1e-6 of the throughput, the laterals entrain, the
//            oxygen in the box is what the inflow faces carried (Y_O in [0,
//            Y_O2]);
//   backflow a nearly inviscid vortex straddling the open y-hi face (2000
//            steps): the kinetic energy never grows with the stabilization
//            (beta 1); without it (beta 0) it grows 4.4x (measured).
//
// Usage: test_gpu_lowmach. No HIP device -> skip (77).
#include "dns/gpu/lowmach.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/reductions.hpp"
#include "gpu_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace dns;
using namespace dns::gpu;
using dnstest::check;

struct LmCase
{
    Grid g;
    LowMachSetup sp;
    LowMachState ic;
};

// The hot-blob box: T = 1 + 3 exp(-r^2/4) around (Lx/3, Ly/2[, Lz/2]), m_x =
// mIn on every x face, m_y = 0.3 rho sin(2 pi x/Lx) cos(2 pi z/Lz), p = 0.
LmCase blob(int nx, int ny, int nz, int nY, double mIn, double yF, double yO)
{
    LmCase c;
    const double h = 0.25;
    c.g = Grid::make(nx, ny, nz, nx * h, ny * h, nz == 1 ? 1.0 : nz * h);
    LowMachSetup& s = c.sp;
    s.tr = Transport::air(0.72, 1.0, 0.0, 0.72, kGammaAir, kRgasBox, false);
    s.prGas = 1;
    s.nY = nY;
    s.fuel = 0;
    s.ox = nY == 2 ? 1 : -1;
    s.lnA = -HUGE_VAL; // no reaction
    s.Ta = 50;
    s.qcp = 4;
    s.mIn = mIn;
    s.TIn = 1;
    s.yIn[0] = yF;
    s.yIn[1] = yO;
    const Grid& g = c.g;
    LowMachState& q = c.ic;
    q.nY = nY;
    q.rho.assign(g.size(), 0);
    q.p.assign(g.size(), 0);
    q.my.assign(g.size(), 0);
    q.mz.assign(g.size(), 0);
    q.mx.assign(std::size_t(g.nx + 1) * g.ny * g.nz, mIn);
    q.rhoY[0].assign(g.size(), 0);
    if (nY == 2) q.rhoY[1].assign(g.size(), 0);
    const double Lx = g.nx * g.hx, Ly = g.ny * g.hy, Lz = g.nz * g.hz;
    const double tau = 2 * std::numbers::pi;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy,
                             z = (k + 0.5) * g.hz;
                double r2 = (x - Lx / 3) * (x - Lx / 3) + (y - Ly / 2) * (y - Ly / 2);
                if (g.nz > 1) r2 += (z - Lz / 2) * (z - Lz / 2);
                const double T = 1 + 3 * std::exp(-r2 / 4);
                const std::size_t c0 = g.idx(i, j, k);
                q.rho[c0] = 1 / T;
                q.rhoY[0][c0] = q.rho[c0] * yF;
                if (nY == 2) q.rhoY[1][c0] = q.rho[c0] * yO;
                if (mIn != 0)
                    q.my[c0] = 0.3 * q.rho[c0] * std::sin(tau * x / Lx) *
                               (g.nz > 1 ? std::cos(tau * z / Lz) : 1.0);
            }
    return c;
}

struct Result
{
    bool ok = false;
    LowMachState hi, full;
    double mass = 0, tally = 0, divRel = 0, eos = 0;
    double ymass[2] = {0, 0}, ytally[2] = {0, 0};
    double throughput = 0; // sum over steps of dt (|mIn| + |mOut| + |mLat|)
    double keMax = 0, ke0 = 0;
    LmBoundarySums bs;
};

// Kinetic energy of a host state: the face mass fluxes squared over the
// two-cell mean density (x interior and lateral lo faces; FP64).
double kinetic(const Grid& g, const LowMachState& s)
{
    double e = 0;
    auto rc = [&](int i, int j, int k) {
        return s.rho[g.idx(i, (j + g.ny) % g.ny, (k + g.nz) % g.nz)];
    };
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                if (i > 0) {
                    const double m = s.mx[std::size_t(i) + std::size_t(g.nx + 1) *
                                                               (std::size_t(j) + std::size_t(g.ny) * k)];
                    e += m * m / (0.5 * (rc(i - 1, j, k) + rc(i, j, k)));
                }
                if (j > 0) e += s.my[c] * s.my[c] / (0.5 * (rc(i, j - 1, k) + rc(i, j, k)));
                if (k > 0) e += s.mz[c] * s.mz[c] / (0.5 * (rc(i, j, k - 1) + rc(i, j, k)));
            }
    return 0.5 * e * g.cellVolume();
}

// steps steps at the solver's dt (or dtFixed > 0). keEvery > 0: the kinetic
// energy's max over every keEvery-th step (a host download).
Result run(const LmCase& c, int steps, double dtFixed = 0, int keEvery = 0)
{
    Result r;
    auto d = DeviceLowMach::make(c.g, c.sp);
    auto red = ReduceScratch::make();
    if (!d || !red) {
        std::printf("    make failed: %s\n", d ? "reduce" : d.error().c_str());
        return r;
    }
    if (auto e = d->upload(c.ic); !e) {
        std::printf("    upload failed: %s\n", e.error().c_str());
        return r;
    }
    if (keEvery > 0) r.ke0 = r.keMax = kinetic(c.g, c.ic);
    for (int n = 0; n < steps; ++n) {
        double dt = dtFixed;
        if (dt <= 0)
            dt = 0.95 * std::min(d->convectiveDt(*red, 0.8), d->diffusiveDt(*red));
        if (auto e = d->step(dt); !e) {
            std::printf("    step failed: %s\n", e.error().c_str());
            return r;
        }
        const LmBoundarySums b = d->boundarySums();
        r.throughput += dt * (std::abs(b.mIn) + std::abs(b.mOut) + std::abs(b.mLat));
        if (keEvery > 0 && (n + 1) % keEvery == 0) {
            auto s = d->download(true);
            if (!s) return r;
            const double ke = kinetic(c.g, *s);
            if (!std::isfinite(ke)) {
                r.keMax = HUGE_VAL;
                break;
            }
            r.keMax = std::max(r.keMax, ke);
        }
    }
    r.divRel = d->divErr(*red) / d->divTargetMax(*red);
    r.eos = d->eosErr(*red);
    r.bs = d->boundarySums();
    auto h = d->download(true), f = d->download(false);
    if (!h || !f) return r;
    r.hi = std::move(*h);
    r.full = std::move(*f);
    const double dV = c.g.cellVolume();
    for (double v : r.full.rho) r.mass += v * dV;
    r.tally = d->massTally();
    for (int k = 0; k < c.sp.nY; ++k) {
        for (double v : r.full.rhoY[std::size_t(k)]) r.ymass[k] += v * dV;
        r.ytally[k] = d->speciesTally(k);
    }
    r.ok = hipDeviceSynchronize() == hipSuccess;
    return r;
}

bool sameState(const LowMachState& a, const LowMachState& b)
{
    bool s = a.rho == b.rho && a.p == b.p && a.mx == b.mx && a.my == b.my &&
             a.mz == b.mz && a.myHi == b.myHi && a.mzHi == b.mzHi;
    for (int k = 0; k < a.nY; ++k) s = s && a.rhoY[std::size_t(k)] == b.rhoY[std::size_t(k)];
    return s;
}

void passive(const std::string& name, const LmCase& c, double yF, double yO)
{
    const Result r = run(c, 100);
    check(r.ok, name + ": 100 steps ran");
    if (!r.ok) return;
    long bad = 0;
    for (std::size_t q = 0; q < r.hi.rho.size(); ++q) {
        bad += r.hi.rhoY[0][q] != yF * r.hi.rho[q];
        bad += r.full.rhoY[0][q] != yF * r.full.rho[q];
        if (c.sp.nY == 2) {
            bad += r.hi.rhoY[1][q] != yO * r.hi.rho[q];
            bad += r.full.rhoY[1][q] != yO * r.full.rho[q];
        }
    }
    check(bad == 0, std::format("{}: rho Y == Y rho bitwise (hi and hi + lo), "
                                "{} mismatches",
                                name, bad));
    // The blob moved: the test is not of a frozen state.
    double rmin = 1e300, rmax = 0;
    for (double v : r.full.rho) {
        rmin = std::min(rmin, v);
        rmax = std::max(rmax, v);
    }
    const double rel = std::abs(r.mass - r.tally) / r.tally;
    std::printf("  %s: rho in [%.4f, %.4f], mass %.15g tally %.15g (rel %.2e), "
                "divErr/max|D| %.2e, eosErr %.2e\n",
                name.c_str(), rmin, rmax, r.mass, r.tally, rel, r.divRel, r.eos);
    check(rel <= 1e-10, name + ": mass = the FP64 face-flux tally to 1e-10");
    double yrel = 0;
    for (int k = 0; k < c.sp.nY; ++k)
        yrel = std::max(yrel, std::abs(r.ymass[k] - r.ytally[k]) / r.ytally[k]);
    std::printf("    species inventory vs the FP64 boundary-flux tally: rel %.2e\n",
                yrel);
    check(yrel <= 1e-10, name + ": rho Y_k = the species boundary-flux tally to "
                                "1e-10");
    check(r.eos <= 1e-6, name + ": EOS defect max |rho T* - 1| <= 1e-6");
    check(r.divRel <= 1e-4, name + ": projected max |div m - D| / max |D| <= 1e-4");
    // Rerun: a fresh solver, byte-identical.
    const Result r2 = run(c, 100);
    check(r2.ok && sameState(r.hi, r2.hi) && sameState(r.full, r2.full),
          name + ": rerun byte-identical");
}

// The flame index on a drawn field (S5, gpu/derived flameIndexCentred): xi
// of every cell, the floor 2^-24/h.
std::vector<float> flameIndexOf(const Grid& g, const std::vector<float>& yf,
                                const std::vector<float>& yo)
{
    auto a = DeviceScalar::make(g.size()), b = DeviceScalar::make(g.size()),
         o = DeviceScalar::make(g.size());
    std::vector<float> x(g.size(), NAN);
    if (!a || !b || !o || a->upload(yf.data()) != hipSuccess ||
        b->upload(yo.data()) != hipSuccess)
        return x;
    flameIndexCentred(GpuGrid::from(g), *a, *b, *o);
    if (o->download(x.data()) != hipSuccess) x.assign(g.size(), NAN);
    return x;
}

// Host |grad| of a drawn field with the kernel's stencil (FP64): x one-sided
// at the ends, y and z periodic central.
double gradNorm(const Grid& g, const std::vector<float>& f, int i, int j, int k)
{
    const int il = i > 0 ? i - 1 : i, ir = i + 1 < g.nx ? i + 1 : i;
    auto at = [&](int a, int b, int c) {
        return double(f[g.idx(a, (b + g.ny) % g.ny, (c + g.nz) % g.nz)]);
    };
    const double fx = (at(ir, j, k) - at(il, j, k)) / ((ir - il) * g.hx);
    const double fy = (at(i, j + 1, k) - at(i, j - 1, k)) / (2 * g.hy);
    const double fz = (at(i, j, k + 1) - at(i, j, k - 1)) / (2 * g.hz);
    return std::sqrt(fx * fx + fy * fy + fz * fz);
}

// xi of the drawn pair: within tol of want wherever both gradients are well
// above the noise floor (|grad Y| >= 2^-12/h: rounding moves the angle by
// <= 2^-12, xi by ~1e-8), and every cell 0 or within [-1, 1] + 1e-6; at
// least minCount strong cells.
void indexIs(const std::string& name, const Grid& g, const std::vector<float>& yf,
             const std::vector<float>& yo, double want, double tol,
             std::size_t minCount)
{
    const auto xi = flameIndexOf(g, yf, yo);
    const double strong = std::ldexp(1.0, -12) / g.hx;
    std::size_t n = 0, bad = 0;
    double dev = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double v = xi[g.idx(i, j, k)];
                bad += !(std::abs(v) <= 1 + 1e-6);
                if (gradNorm(g, yf, i, j, k) < strong ||
                    gradNorm(g, yo, i, j, k) < strong)
                    continue;
                ++n;
                dev = std::max(dev, std::abs(v - want));
                bad += !(std::abs(v - want) <= tol);
            }
    std::printf("  %s: %zu strong cells, max |xi - %+g| = %.2e\n", name.c_str(), n,
                want, dev);
    check(bad == 0 && n >= minCount,
          std::format("{}: xi = {:+g} to {:g} ({} strong cells, >= {})", name,
                      want, tol, n, minCount));
}

void findexTests()
{
    std::printf("findex (Takeno: premixed +1, diffusion sheet -1, noise 0)\n");
    for (const int nz : {1, 8}) {
        const double h = 0.5;
        const Grid g = Grid::make(32, 16, nz, 32 * h, 16 * h, nz == 1 ? 1.0 : nz * h);
        const std::string d = nz == 1 ? "2D" : "3D";
        std::vector<float> yf(g.size()), yo(g.size());
        auto fill = [&](auto&& F, auto&& O) {
            for (int k = 0; k < g.nz; ++k)
                for (int j = 0; j < g.ny; ++j)
                    for (int i = 0; i < g.nx; ++i) {
                        const double x = (i + 0.5) * h, y = (j + 0.5) * h,
                                     z = (k + 0.5) * h;
                        yf[g.idx(i, j, k)] = float(F(x, y, z));
                        yo[g.idx(i, j, k)] = float(O(x, y, z));
                    }
        };
        auto layer = [](double s) { return 0.5 * (1 - std::tanh((s - 8) / 1.5)); };
        // A planar layer along x: both reactants consumed toward +x
        // (premixed), and a Burke-Schumann sheet (fuel from -x, oxidizer
        // from +x). Pure x gradients: xi exactly +-1.
        fill([&](double x, double, double) { return 0.07 * layer(x); },
             [&](double x, double, double) { return 0.22 * layer(x); });
        indexIs(d + " premixed layer (x)", g, yf, yo, 1.0, 0.0, 32);
        fill([&](double x, double, double) { return 0.07 * layer(x); },
             [&](double x, double, double) { return 0.233 * (1 - layer(x)); });
        indexIs(d + " diffusion sheet (x)", g, yf, yo, -1.0, 0.0, 32);
        // A wavy front s = x + 1.5 sin(2 pi y/Ly) (+ z in 3D): the two
        // reactants functions of the same s, so the gradients are parallel
        // (premixed) or antiparallel (sheet) away from rounding.
        const double Ly = g.ny * h, Lz = g.nz * g.hz;
        auto s = [&](double x, double y, double z) {
            return x + 1.5 * std::sin(6.283185307179586 * y / Ly) +
                   (nz > 1 ? 1.0 * std::cos(6.283185307179586 * z / Lz) : 0.0);
        };
        fill([&](double x, double y, double z) { return 0.07 * layer(s(x, y, z)); },
             [&](double x, double y, double z) { return 0.22 * layer(s(x, y, z)); });
        indexIs(d + " premixed layer (wavy)", g, yf, yo, 1.0, 1e-6, 32);
        fill([&](double x, double y, double z) { return 0.07 * layer(s(x, y, z)); },
             [&](double x, double y, double z) {
                 return 0.233 * (1 - layer(s(x, y, z)));
             });
        indexIs(d + " diffusion sheet (wavy)", g, yf, yo, -1.0, 1e-6, 32);
        // Noise: 0.25 + {0, 1 ulp} (2^-25) in a fixed pattern: every gradient
        // under the floor 2^-24/h, xi 0 everywhere.
        for (std::size_t q = 0; q < g.size(); ++q) {
            yf[q] = 0.25f + float((q * 2654435761u >> 7) & 1) * std::ldexp(1.0f, -25);
            yo[q] = 0.25f + float((q * 40503u >> 5) & 1) * std::ldexp(1.0f, -25);
        }
        const auto xn = flameIndexOf(g, yf, yo);
        bool zero = true;
        for (const float v : xn) zero = zero && v == 0;
        check(zero, d + " noise floor: xi 0 everywhere");
    }
}

// The open-lateral variant of a case (S6): y open (z too in 3D), the
// ambient at T 1 with the given Y (the passive tests: the uniform Y itself).
LmCase opened(LmCase c, double yF, double yO)
{
    c.sp.openY = true;
    c.sp.openZ = c.g.nz > 1;
    c.sp.ambT = 1;
    c.sp.ambY[0] = yF;
    c.sp.ambY[1] = yO;
    return c;
}

void openTests()
{
    std::printf("open laterals (S6): passive / mass / species tally / rerun\n");
    passive("2D C0 open y", opened(blob(64, 32, 1, 1, 1.0, 0.5, 0), 0.5, 0), 0.5, 0);
    passive("2D two reactants open y",
            opened(blob(64, 32, 1, 2, 1.0, 0.25, 0.5), 0.25, 0.5), 0.25, 0.5);
    passive("3D C0 open y, z", opened(blob(32, 16, 16, 1, 1.0, 0.5, 0), 0.5, 0), 0.5,
            0);

    std::printf("open rest (quiescent box, 1000 steps of dt 0.05)\n");
    for (const int nz : {1, 8}) {
        LmCase c = opened(blob(32, 16, nz, 2, 0.0, 0.25, 0.5), 0.25, 0.5);
        for (double& v : c.ic.rho) v = 1;
        for (double& v : c.ic.rhoY[0]) v = 0.25;
        for (double& v : c.ic.rhoY[1]) v = 0.5;
        const Result r = run(c, 1000, 0.05);
        bool rest = r.ok;
        for (std::size_t q = 0; rest && q < r.hi.rho.size(); ++q)
            rest = r.hi.rho[q] == 1 && r.hi.rhoY[0][q] == 0.25 &&
                   r.hi.rhoY[1][q] == 0.5 && r.hi.p[q] == 0 && r.hi.my[q] == 0 &&
                   r.hi.mz[q] == 0 && r.full.rho[q] == 1;
        for (double v : r.hi.mx) rest = rest && v == 0;
        for (double v : r.hi.myHi) rest = rest && v == 0;
        for (double v : r.hi.mzHi) rest = rest && v == 0;
        check(rest && r.hi.myHi.size() == std::size_t(32) * nz &&
                  r.hi.mzHi.size() == (nz > 1 ? std::size_t(32) * 16 : 0),
              std::format("rest {}: u == 0 and p == 0 exactly over 1000 steps "
                          "(every face, the hi faces too)",
                          nz == 1 ? "2D open y" : "3D open y, z"));
    }

    // Balance: a pure-fuel box (Y_F 1, Y_O 0, a hot blob: dilatation pushes
    // gas out sideways) fed by a cold fuel jet through the middle third of
    // x-lo (the rest a wall); the ambient is AIR (Y_O 0.233). Every oxygen
    // atom in the box came in through an open face carrying the ambient.
    std::printf("open balance (fuel box, fuel jet, air ambient)\n");
    for (const int nz : {1, 8}) {
        LmCase c = blob(48, 32, nz, 2, 0.0, 1.0, 0.0);
        c = opened(c, 0.0, 0.233);
        const std::size_t nr = std::size_t(c.g.ny) * c.g.nz;
        c.sp.rowM.assign(nr, 0.0);
        c.sp.rowT.assign(nr, 1.0);
        c.sp.rowY[0].assign(nr, 1.0);
        c.sp.rowY[1].assign(nr, 0.0);
        for (int k = 0; k < c.g.nz; ++k)
            for (int j = c.g.ny / 3; j < 2 * c.g.ny / 3; ++j) {
                const bool in = nz == 1 || (k >= c.g.nz / 4 && k < 3 * c.g.nz / 4);
                if (in) c.sp.rowM[std::size_t(j) + std::size_t(c.g.ny) * k] = 2.0;
            }
        for (int k = 0; k < c.g.nz; ++k)
            for (int j = 0; j < c.g.ny; ++j)
                for (int i = 0; i <= c.g.nx; ++i)
                    c.ic.mx[std::size_t(i) + std::size_t(c.g.nx + 1) *
                                                 (std::size_t(j) + std::size_t(c.g.ny) * k)] =
                        c.sp.rowM[std::size_t(j) + std::size_t(c.g.ny) * k];
        const Result r = run(c, 300);
        check(r.ok, "balance: 300 steps ran");
        if (!r.ok) continue;
        const double dm = std::abs(r.mass - r.tally) / r.throughput;
        const double dF = std::abs(r.ymass[0] - r.ytally[0]) / r.throughput;
        const double dO = std::abs(r.ymass[1] - r.ytally[1]) / (0.233 * r.throughput);
        std::printf("  %s: throughput %.4g, |mass - tally| %.2e, |fuel - tally| "
                    "%.2e, |O - tally| %.2e (of throughput); O inventory %.4g, "
                    "mIn %.4g mOut %.4g mLat %+.4g, uLat [%.4g, %.4g]\n",
                    nz == 1 ? "2D" : "3D", r.throughput, dm, dF, dO, r.ymass[1],
                    r.bs.mIn, r.bs.mOut, r.bs.mLat, r.bs.uLatMin, r.bs.uLatMax);
        const std::string d = nz == 1 ? "balance 2D: " : "balance 3D: ";
        check(dm <= 1e-6 && dF <= 1e-6 && dO <= 1e-6,
              d + "mass and both species = their FP64 boundary tallies to 1e-6 "
                  "of the throughput");
        check(r.bs.uLatMin < 0 && r.ymass[1] > 1e-3 * 0.233 * r.throughput,
              d + "the laterals entrain (uLatMin < 0) and the oxygen they carry "
                  "is the ambient's (inventory > 0, = the tally)");
        double yoMin = 1, yoMax = 0;
        for (std::size_t q = 0; q < r.full.rho.size(); ++q) {
            const double yo = r.full.rhoY[1][q] / r.full.rho[q];
            yoMin = std::min(yoMin, yo);
            yoMax = std::max(yoMax, yo);
        }
        check(yoMin >= -1e-6 && yoMax <= 0.233 + 1e-6,
              std::format("{}Y_O within [0, Y_O2] ({:.3g} .. {:.6g})", d, yoMin,
                          yoMax));
    }

    // Backflow: a vortex centred 3 cells inside the y-hi face of a box at
    // rest (x-lo a wall) sweeps gas out of and back into the open face. The
    // kinetic energy must not grow (viscous decay; the stabilization removes
    // the incoming kinetic-energy flux); beta = 0 is reported.
    std::printf("backflow (a vortex straddling the open y-hi face)\n");
    for (const double beta : {1.0, 0.0}) {
        LmCase c = opened(blob(64, 48, 1, 1, 0.0, 0.5, 0), 0.5, 0);
        c.sp.backflow = beta;
        // Nearly inviscid (mu 0.01: cell Reynolds ~ 40 at the vortex core):
        // the central scheme carries no dissipation of its own.
        c.sp.tr = Transport::air(0.01, 1.0, 0.0, 0.72, kGammaAir, kRgasBox, false);
        const Grid& g = c.g;
        for (double& v : c.ic.rho) v = 1;
        for (double& v : c.ic.rhoY[0]) v = 0.5;
        const double xc = 0.5 * g.nx * g.hx, yc = (g.ny - 3) * g.hy, rc = 2.0,
                     G = 30.0;
        auto vel = [&](double x, double y, double& u, double& v) {
            const double dx = x - xc, dy = y - yc, r2 = dx * dx + dy * dy;
            const double f = r2 > 0 ? G / (2 * std::numbers::pi * r2) *
                                          (1 - std::exp(-r2 / (rc * rc)))
                                    : 0.0;
            u = -f * dy;
            v = f * dx;
        };
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                double u, v;
                if (i > 0) {
                    vel(i * g.hx, (j + 0.5) * g.hy, u, v);
                    c.ic.mx[std::size_t(i) + std::size_t(g.nx + 1) * j] = u;
                }
                vel((i + 0.5) * g.hx, j * g.hy, u, v);
                c.ic.my[g.idx(i, j, 0)] = v;
            }
        c.ic.myHi.assign(std::size_t(g.nx), 0.0);
        for (int i = 0; i < g.nx; ++i) {
            double u, v;
            vel((i + 0.5) * g.hx, g.ny * g.hy, u, v);
            c.ic.myHi[std::size_t(i)] = v;
        }
        const Result r = run(c, 2000, 0.01, 100);
        std::printf("  beta %g: KE max/KE0 %.6f (KE0 %.4g), end KE/KE0 %.4f\n", beta,
                    r.keMax / r.ke0, r.ke0, kinetic(g, r.hi) / r.ke0);
        if (beta == 1.0)
            check(r.ok && r.keMax <= r.ke0 * (1 + 1e-6),
                  "backflow beta 1: the kinetic energy never grows (2000 steps)");
        else // measured 4.42: the stabilization is load-bearing
            check(r.keMax > 1.5 * r.ke0,
                  "backflow beta 0: the open face injects kinetic energy (KE "
                  "grows > 1.5x: why the stabilization is on)");
    }
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    std::printf("passive / mass / rerun (2D 64 x 32, 3D 32 x 16 x 16; C0 and "
                "two reactants)\n");
    passive("2D C0", blob(64, 32, 1, 1, 1.0, 0.5, 0), 0.5, 0);
    passive("2D two reactants", blob(64, 32, 1, 2, 1.0, 0.25, 0.5), 0.25, 0.5);
    passive("3D C0", blob(32, 16, 16, 1, 1.0, 0.5, 0), 0.5, 0);
    // rows (S5): the per-row inflow profile filled with the scalars' values
    // is the scalar path bit for bit (the device reads rows either way).
    std::printf("rows (uniform per-row inflow == the scalar targets)\n");
    {
        LmCase a = blob(64, 32, 1, 2, 1.0, 0.25, 0.5);
        LmCase b = a;
        const std::size_t nr = std::size_t(b.g.ny) * b.g.nz;
        b.sp.rowM.assign(nr, b.sp.mIn);
        b.sp.rowT.assign(nr, b.sp.TIn);
        b.sp.rowY[0].assign(nr, b.sp.yIn[0]);
        b.sp.rowY[1].assign(nr, b.sp.yIn[1]);
        const Result ra = run(a, 50), rb = run(b, 50);
        check(ra.ok && rb.ok && sameState(ra.full, rb.full),
              "rows: uniform rows == scalars, every field bitwise after 50 steps");
        LmCase bad = b;
        bad.sp.rowT.pop_back();
        check(!DeviceLowMach::make(bad.g, bad.sp).has_value(),
              "rows: a profile of the wrong length is an error by name");
    }
    findexTests();
    std::printf("rest (uniform, m_in 0, dt 0.05)\n");
    {
        LmCase c = blob(32, 16, 1, 1, 0.0, 1.0, 0);
        for (double& v : c.ic.rho) v = 1;
        for (double& v : c.ic.rhoY[0]) v = 1;
        const Result r = run(c, 100, 0.05);
        bool rest = r.ok;
        for (std::size_t q = 0; rest && q < r.hi.rho.size(); ++q)
            rest = r.hi.rho[q] == 1 && r.hi.rhoY[0][q] == 1 && r.hi.p[q] == 0 &&
                   r.hi.my[q] == 0 && r.hi.mz[q] == 0 && r.full.rho[q] == 1;
        for (double v : r.hi.mx) rest = rest && v == 0;
        check(rest, "rest: rho, Y, m, p unchanged bitwise over 100 steps");
    }
    openTests();
    return dnstest::finish();
}
