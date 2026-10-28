// The two-reactant step of `dns flame --mode diffusion` (docs/chem.md),
// device API level, on an unsteady diffusion layer (fuel slab | oxidizer)
// whose mixture fraction is the analytic erf solution.
//
// mix:  isothermal (no heat release: rho, T, u stay uniform), a fuel slab on
//       a periodic L x L box TILTED 45 deg ((x + y) is the layer normal) and
//       advected by u = (1, 0) -- the shared limiter is active on both axes.
//       Three species: Y_F, Y_O and a passive tracer carrying Z.
//   (a) coupling function: Z from (Y_F, Y_O) equals the tracer to rounding,
//       reacting AND passive; the per-species limiter (the premixed kernel
//       on the same passive IC) breaks it at truncation size (the contrast);
//   (b) bounds: Z, Y_F, Y_O in [0, 1] (to 1e-6);
//   (c) Burke-Schumann: the peak of omega sits at the analytic Z = Z_st
//       surface x_f(t) (two flames, erf images), within a band in cells,
//       and the fuel burnt per flame matches the diffusive flux into the
//       sheet rho D Y_F1 |dZ/dn| / (1 - Z_st);
// heat: heated (sigma 4, beta 8) 1D layer on an open x box (NSCBC at both
//       faces: isobaric), Da and 8 Da: the peak temperature against the
//       flame-sheet T_ad = sigma T0, the Z at the peak of omega against Z_st,
//       the deficit falling with Da; the tracer and bounds again.
#include "dns/chem.hpp"
#include "dns/comp.hpp"
#include "dns/diffusion.hpp"
#include "dns/gpu/chem.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/transport.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>
#include <vector>

using namespace dns;

namespace {

using dnstest::check;

constexpr double gam = 1.4, Pr = 0.72;
const double kSqrt2 = std::sqrt(2.0);

struct Fields
{
    std::vector<double> rho, yf, yo, z, T, w;
};

// Download rho, Y_F, Y_O, the tracer (nY == 3), T and omega of state s.
Fields fetch(const Grid& g, gpu::DeviceComp& dev,
             const gpu::DeviceCompState& s, int nY)
{
    dev.updateChem(s);
    Fields f;
    CompState h = CompState::zeros(g);
    if (s.download(h) != hipSuccess) std::exit(EXIT_FAILURE);
    f.rho = h.rho;
    std::vector<double> ry[3];
    for (int k = 0; k < nY; ++k)
        if (s.downloadSpecies(k, ry[k]) != hipSuccess) std::exit(EXIT_FAILURE);
    const std::size_t n = f.rho.size();
    std::vector<float> T(n), w(n);
    if (dev.T().download(T.data()) != hipSuccess ||
        dev.omega().download(w.data()) != hipSuccess)
        std::exit(EXIT_FAILURE);
    f.yf.resize(n);
    f.yo.resize(n);
    f.z.resize(n);
    f.T.assign(T.begin(), T.end());
    f.w.assign(w.begin(), w.end());
    for (std::size_t c = 0; c < n; ++c) {
        f.yf[c] = ry[0][c] / f.rho[c];
        f.yo[c] = ry[1][c] / f.rho[c];
        f.z[c] = nY > 2 ? ry[2][c] / f.rho[c] : 0.0;
    }
    return f;
}

struct Checks
{
    double dz = 0;          // max |Z(Y_F, Y_O) - tracer|
    double zMin = HUGE_VAL, zMax = -HUGE_VAL, yMin = HUGE_VAL, yfMax = 0,
           yoMax = 0;
};

Checks scalarChecks(const DiffusionModel& m, const Fields& f)
{
    Checks c;
    for (std::size_t i = 0; i < f.rho.size(); ++i) {
        const double z = mixtureFraction(m, f.yf[i], f.yo[i]);
        c.dz = std::max(c.dz, std::abs(z - f.z[i]));
        c.zMin = std::min({c.zMin, z, f.z[i]});
        c.zMax = std::max({c.zMax, z, f.z[i]});
        c.yMin = std::min({c.yMin, f.yf[i], f.yo[i]});
        c.yfMax = std::max(c.yfMax, f.yf[i]);
        c.yoMax = std::max(c.yoMax, f.yo[i]);
    }
    return c;
}

// Parabolic peak of v[k] around index k0 (periodic of length n): returns
// the fractional index.
double parabolicPeak(const std::vector<double>& v, int k0)
{
    const int n = int(v.size());
    const double a = v[(k0 - 1 + n) % n], b = v[k0], c = v[(k0 + 1) % n];
    const double den = a - 2 * b + c;
    return den < 0 ? k0 + 0.5 * (a - c) / den : double(k0);
}

// ---- mix -----------------------------------------------------------------

constexpr double kL = 64, kU = 1.0, kT0Virtual = 2.0, kRun = 20.0;
constexpr int kN = 256;

// Analytic mixture fraction of the slab [a, b] (period P along s) shifted by
// x0, diffusivity D, at diffusion age t (step IC at t = 0), images -3..3.
double slabZ(double s, double t, double x0, double a, double b, double P,
             double D)
{
    const double d = 2 * std::sqrt(D * t);
    double z = 0;
    for (int k = -3; k <= 3; ++k)
        z += 0.5 * (std::erf((s - a - x0 + k * P) / d) -
                    std::erf((s - b - x0 + k * P) / d));
    return z;
}

struct MixOut
{
    Checks ch;
    double errL = 0, errR = 0;      // flame position error, cells along s
    double burnL = 0, burnR = 0;    // burnt / analytic diffusive flux
    double zPeakL = 0, zPeakR = 0;  // tracer Z at the omega peaks
    double zErr = 0;                // max |tracer - analytic erf Z| on row 0
    long steps = 0;
};

// reacting: Da at lnA = log(Da) (isothermal: Ta = 0); ox: -1 runs the
// premixed (per-species limiter) kernel with no reaction, the contrast. D:
// the diffusivity (mu = Pr D); D < 1 raises the cell Peclet U h / D so the
// limiter acts (the contrast runs at D = 0.02, cell Peclet 12.5).
MixOut runMix(const DiffusionModel& m, bool reacting, int ox, double D = 1)
{
    const double h = kL / kN, M = 0.1, T0 = 1 / (gam * M * M);
    const Grid g = Grid::make(kN, kN, 1, kL, kL, 1.0);
    const auto tr = Transport::air(Pr * D, T0, 288.15, Pr, gam, 1.0, false);
    auto dev = gpu::DeviceComp::make(g, tr);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !red) std::exit(EXIT_FAILURE);
    gpu::SpeciesSetup sp;
    sp.nY = 3;
    sp.fuel = 0;
    sp.ox = ox;
    sp.stoich = m.s;
    sp.lnA = reacting ? std::log(m.Da)
                      : -std::numeric_limits<double>::infinity();
    sp.Ta = 0;
    sp.q = 0;
    sp.invPr = 1 / Pr;
    if (!dev->enableSpecies(sp)) std::exit(EXIT_FAILURE);
    const double P = kL / kSqrt2, a = 0.25 * P, b = 0.75 * P, cs = kU / kSqrt2;
    CompState ic = CompState::zeros(g);
    std::vector<double> ry[3];
    for (auto& v : ry) v.assign(g.size(), 0.0);
    for (int j = 0; j < kN; ++j)
        for (int i = 0; i < kN; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double s = ((i + 0.5) * h + (j + 0.5) * h) / kSqrt2;
            const double Z = slabZ(s, kT0Virtual, 0.0, a, b, P, D);
            const SheetState st = burkeSchumann(m, Z);
            ic.rho[c] = 1;
            ic.mx[c] = kU;
            ic.E[c] = T0 / (gam - 1) + 0.5 * kU * kU;
            ry[0][c] = st.YF;
            ry[1][c] = st.YO;
            ry[2][c] = mixtureFraction(m, st.YF, st.YO);
        }
    auto ds = gpu::DeviceCompState::make(g.size(), 0, false, 3);
    if (!ds || ds->upload(ic) != hipSuccess) std::exit(EXIT_FAILURE);
    for (int k = 0; k < 3; ++k)
        if (ds->uploadSpecies(k, ry[k]) != hipSuccess) std::exit(EXIT_FAILURE);

    MixOut o;
    double t = 0, dt = 0;
    while (t < kRun) {
        if (o.steps % 16 == 0)
            dt = 0.95 * std::min(dev->stableDt(*ds, *red, 0.4),
                                 gpu::chemStableDt(*dev, *ds, *red));
        const double st = std::min(dt, kRun - t);
        dev->step(*ds, gpu::Real(st));
        t += st;
        ++o.steps;
    }
    const Fields f = fetch(g, *dev, *ds, 3);
    o.ch = scalarChecks(m, f);
    if (!reacting) return o;

    // Row j = 0: s_i = (i + 1) h / sqrt 2 (the field depends on i + j only).
    const double ds_ = h / kSqrt2, tv = kT0Virtual + kRun;
    std::vector<double> w(kN), z(kN);
    for (int i = 0; i < kN; ++i) {
        w[i] = f.w[g.idx(i, 0, 0)];
        z[i] = f.z[g.idx(i, 0, 0)];
    }
    auto sOf = [&](double k) { return (k + 1) * ds_; };
    const double zs = m.zst();
    // Analytic flame positions: Z = zst on each side of the slab centre.
    const double mid = std::fmod(0.5 * (a + b) + cs * kRun, P);
    auto zAt = [&](double s) { return slabZ(s, tv, cs * kRun, a, b, P, D); };
    for (int i = 0; i < kN; ++i)
        o.zErr = std::max(o.zErr, std::abs(z[i] - zAt(sOf(i))));
    if (std::getenv("DNS_DIFF_DUMP"))
        for (int i = 0; i < kN; ++i)
            std::printf("dump %d %.6f %.6e %.6e %.6e %.6e %.6e\n", i, sOf(i),
                        w[i], z[i], zAt(sOf(i)), f.yf[g.idx(i, 0, 0)],
                        f.yo[g.idx(i, 0, 0)]);
    auto bisect = [&](double lo, double hi) {
        // zAt(lo) - zs and zAt(hi) - zs of opposite sign.
        const double flo = zAt(lo) - zs;
        for (int k = 0; k < 200; ++k) {
            const double md = 0.5 * (lo + hi);
            ((zAt(md) - zs) * flo > 0 ? lo : hi) = md;
        }
        return 0.5 * (lo + hi);
    };
    const double sR = bisect(mid, mid + 0.5 * P), sL = bisect(mid - 0.5 * P, mid);
    for (int side = 0; side < 2; ++side) {
        const double sa = side == 0 ? sL : sR;
        // Nearest cell, then the max of omega within +- 3 units.
        const int kc = int(std::lround(sa / ds_ - 1));
        const int win = int(3.0 / ds_);
        int kb = kc;
        for (int d = -win; d <= win; ++d) {
            const int k = ((kc + d) % kN + kN) % kN;
            if (w[k] > w[(kb % kN + kN) % kN]) kb = kc + d;
        }
        const int kbm = (kb % kN + kN) % kN;
        const double kp = parabolicPeak(w, kbm) + (kb - kbm);
        const double err = (sOf(kp) - sa) / ds_;
        // Burnt per unit area: omega over +- P/4 around the flame.
        double burn = 0;
        for (int d = -int(0.25 * P / ds_); d <= int(0.25 * P / ds_); ++d)
            burn += w[((kc + d) % kN + kN) % kN] * ds_;
        const double eps = 1e-4;
        const double dZ = std::abs(zAt(sa + eps) - zAt(sa - eps)) / (2 * eps);
        const double flux = D * m.yF1 * dZ / (1 - zs);
        // Z at the omega peak (linear in the tracer).
        const double fr = kp - std::floor(kp);
        const int k0 = ((int(std::floor(kp)) % kN) + kN) % kN;
        const double zp = (1 - fr) * z[k0] + fr * z[(k0 + 1) % kN];
        (side == 0 ? o.errL : o.errR) = err;
        (side == 0 ? o.burnL : o.burnR) = burn / flux;
        (side == 0 ? o.zPeakL : o.zPeakR) = zp;
    }
    return o;
}

// ---- heat ----------------------------------------------------------------

struct HeatOut
{
    Checks ch;
    double tPeak = 0, tPeakP0 = 0, zAtPeak = 0, pRel = 0;
    long steps = 0;
};

HeatOut runHeat(const DiffusionModel& m, double tEnd)
{
    const double Lx = 128, h = 0.125, M = 0.01, T0 = 1 / (gam * M * M);
    const int nx = int(Lx / h), ny = 4;
    const Grid g = Grid::make(nx, ny, 1, Lx, ny * h, 1.0);
    const auto tr = Transport::air(Pr, T0, 288.15, Pr, gam, 1.0, false);
    auto dev = gpu::DeviceComp::make(g, tr);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !red) std::exit(EXIT_FAILURE);
    gpu::SpeciesSetup sp;
    sp.nY = 3;
    sp.fuel = 0;
    sp.ox = 1;
    sp.stoich = m.s;
    sp.lnA = m.lnA();
    sp.Ta = m.theta() * T0;
    sp.q = tr.cp() * T0 * m.qOverCpT0();
    sp.invPr = 1 / Pr;
    sp.openX = true;
    sp.yIn[0] = 0;
    sp.yIn[1] = m.yO2;
    sp.yIn[2] = 0;
    if (!dev->enableSpecies(sp)) std::exit(EXIT_FAILURE);
    gpu::DeviceCompNscbc::Config nc;
    nc.rhoInf = 1.0;
    nc.uInf = 0.0;
    nc.pInf = T0;
    nc.gamma = gam;
    nc.Rgas = 1.0;
    nc.TInf = T0;
    // A stiff pressure anchor: the default 0.25 relaxes over L/(sigma c) ~ 25
    // time units, so the heat release's volume flux held the whole box 6%
    // above p0 (adiabatic compression +1.7% of T); 4 relaxes in ~1.6.
    nc.sigma = 4.0;
    auto nsc = gpu::DeviceCompNscbc::make(g, nc);
    if (!nsc) std::exit(EXIT_FAILURE);
    nsc->setFreestream(0.0);
    dev->setRhsHook([&](const gpu::DeviceCompState& s, gpu::DeviceCompState& o,
                        const gpu::StageCombine& cmb) {
        nsc->apply(*dev, s, o, cmb);
        dev->speciesOpenConsistency(s, o, cmb);
    });
    // Oxidizer on the left (the held inflow side), fuel on the right; the
    // interface at Lx/2, erf profile of a layer aged 4 (cold D = 1; aged 1
    // the start's scalar dissipation extinguished Da 200).
    CompState ic = CompState::zeros(g);
    std::vector<double> ry[3];
    for (auto& v : ry) v.assign(g.size(), 0.0);
    const double x0 = 0.5 * Lx;
    for (int i = 0; i < nx; ++i) {
        const double x = (i + 0.5) * h;
        const double Z = 0.5 * std::erfc(-(x - x0) / 4.0);
        const SheetState st = burkeSchumann(m, Z);
        const double rho = 1 / st.t;
        for (int j = 0; j < ny; ++j) {
            const std::size_t c = g.idx(i, j, 0);
            ic.rho[c] = rho;
            ic.E[c] = T0 / (gam - 1);
            ry[0][c] = rho * st.YF;
            ry[1][c] = rho * st.YO;
            ry[2][c] = rho * mixtureFraction(m, st.YF, st.YO);
        }
    }
    auto ds = gpu::DeviceCompState::make(g.size(), 0, false, 3);
    if (!ds || ds->upload(ic) != hipSuccess) std::exit(EXIT_FAILURE);
    for (int k = 0; k < 3; ++k)
        if (ds->uploadSpecies(k, ry[k]) != hipSuccess) std::exit(EXIT_FAILURE);
    HeatOut o;
    double t = 0, dt = 0;
    while (t < tEnd) {
        if (o.steps % 16 == 0)
            dt = 0.95 * std::min(dev->stableDt(*ds, *red, 0.4),
                                 gpu::chemStableDt(*dev, *ds, *red));
        const double st = std::min(dt, tEnd - t);
        dev->step(*ds, gpu::Real(st));
        t += st;
        ++o.steps;
    }
    const Fields f = fetch(g, *dev, *ds, 3);
    o.ch = scalarChecks(m, f);
    std::vector<double> w(nx), z(nx);
    double Tm = 0;
    int iT = 0;
    for (int i = 0; i < nx; ++i) {
        w[i] = f.w[g.idx(i, 0, 0)];
        z[i] = f.z[g.idx(i, 0, 0)];
        if (f.T[g.idx(i, 0, 0)] > Tm) {
            Tm = f.T[g.idx(i, 0, 0)];
            iT = i;
        }
    }
    const int kb = int(std::max_element(w.begin(), w.end()) - w.begin());
    const double kp = parabolicPeak(w, kb);
    const int k0 = int(std::floor(kp));
    const double fr = kp - k0;
    o.zAtPeak = (1 - fr) * z[k0] + fr * z[k0 + 1];
    o.tPeak = Tm / (m.sigma * T0);
    double pm = 0;
    CompState hq = CompState::zeros(g);
    if (ds->download(hq) != hipSuccess) std::exit(EXIT_FAILURE);
    auto pAt = [&](std::size_t c) {
        const double ke = 0.5 * (hq.mx[c] * hq.mx[c]) / hq.rho[c];
        return (gam - 1) * (hq.E[c] - ke) / T0;
    };
    for (std::size_t c = 0; c < g.size(); ++c)
        pm = std::max(pm, std::abs(pAt(c) - 1));
    // The start's box acoustic mode (p within ~1% of p0 at M 0.01) is
    // isentropic: T_ad at p0 is the measured peak / (p/p0)^((gamma-1)/gamma).
    o.tPeakP0 = o.tPeak * std::pow(pAt(g.idx(iT, 0, 0)), -(gam - 1) / gam);
    if (std::getenv("DNS_DIFF_DUMP"))
        for (int i = 0; i < nx; i += 4) {
            const std::size_t c = g.idx(i, 0, 0);
            const double ke = 0.5 * (hq.mx[c] * hq.mx[c]) / hq.rho[c];
            std::printf("hdump %d %.4f T %.5f p %.6f u %.5f w %.4e z %.5f\n", i,
                        (i + 0.5) * h, f.T[c] / T0,
                        (gam - 1) * (hq.E[c] - ke) / T0, hq.mx[c] / hq.rho[c],
                        f.w[c], f.z[c]);
        }
    o.pRel = pm;
    return o;
}

void mix()
{
    DiffusionModel m;
    m.s = 3;
    m.yF1 = 1;
    m.yO2 = 1;
    m.sigma = 1;
    m.Da = 50;
    const MixOut r = runMix(m, true, 1);
    const MixOut p2 = runMix(m, false, 1, 0.02);
    const MixOut p1 = runMix(m, false, -1, 0.02);
    std::printf("  mix: Z_st %.4f, Da %g, h %.4f, %ld steps to t = %g\n",
                m.zst(), m.Da, kL / kN, r.steps, kRun);
    std::printf("  (a) max|Z(Y_F,Y_O) - tracer|: reacting %.3e, passive shared "
                "%.3e, passive per-species limiter %.3e\n",
                r.ch.dz, p2.ch.dz, p1.ch.dz);
    std::printf("  (b) Z in [%.3e, 1%+.3e], Y min %.3e, Y_F max-1 %+.3e, Y_O "
                "max-1 %+.3e\n",
                r.ch.zMin, r.ch.zMax - 1, r.ch.yMin, r.ch.yfMax - 1,
                r.ch.yoMax - 1);
    std::printf("  (d) max|Z tracer - analytic erf|: %.3e\n", r.zErr);
    std::printf("  (c) omega peak - analytic x_f: left %+.4f, right %+.4f "
                "cells (s); Z at the peaks %.5f / %.5f (Z_st %.5f)\n",
                r.errL, r.errR, r.zPeakL, r.zPeakR, m.zst());
    std::printf("      burnt / rho D Y_F1 |dZ/dn| / (1 - Z_st): left %.5f, "
                "right %.5f\n",
                r.burnL, r.burnR);
    // FP32 rounding walk: 4 ulp of 1 per stage, sqrt(3 steps) stages.
    auto walk = [](long n) { return 4 * std::ldexp(1.0, -23) * std::sqrt(3.0 * n); };
    std::printf("      rounding-walk bands: reacting %.3e (%ld steps), passive "
                "%.3e (%ld steps)\n",
                walk(r.steps), r.steps, walk(p2.steps), p2.steps);
    check(r.ch.dz < walk(r.steps) && p2.ch.dz < walk(p2.steps),
          "(a) Z(Y_F, Y_O) equals the passive tracer within the FP32 "
          "rounding walk 4 ulp sqrt(3 steps) (shared limiter), reacting and "
          "passive");
    check(p1.ch.dz > 30 * p2.ch.dz,
          "(a) contrast: per-species limiting breaks it by > 30x");
    check(r.ch.zMin > -1e-6 && r.ch.zMax < 1 + 1e-6 && r.ch.yMin > -1e-6 &&
              r.ch.yfMax < 1 + 1e-6 && r.ch.yoMax < 1 + 1e-6,
          "(b) Z, Y_F, Y_O in [-1e-6, 1 + 1e-6]");
    check(r.zErr < 3e-3, "(d) the tracer's Z within 3e-3 of the analytic erf "
                         "(row 0 max; limiter clipping at the slab centre)");
    check(std::abs(r.errL) < 1.0 && std::abs(r.errR) < 1.0,
          "(c) omega peak within 1 cell of the analytic x_f(t)");
    check(std::abs(r.burnL - 1) < 0.02 && std::abs(r.burnR - 1) < 0.02,
          "(c) fuel burnt per flame within 2% of the Burke-Schumann flux");
}

void heat()
{
    DiffusionModel m;
    m.s = 3;
    m.yF1 = 1;
    m.yO2 = 1;
    m.sigma = 4;
    m.beta = 8;
    const double tEnd = 10;
    m.Da = 500;
    const HeatOut a = runHeat(m, tEnd);
    m.Da = 4000;
    const HeatOut b = runHeat(m, tEnd);
    std::printf("  heat: sigma %g beta %g Z_st %.4f t %g\n", m.sigma, m.beta,
                m.zst(), tEnd);
    for (const auto& [o, da] : {std::pair{a, 500.0}, std::pair{b, 4000.0}})
        std::printf("    Da %6g: %ld steps, Tmax/T_ad %.5f (at p0: %.5f), Z at "
                    "omega peak %.5f, max|p/p0 - 1| %.2e, |Z - tracer| %.2e, "
                    "Z [%.2e, 1%+.2e], Y min %.2e\n",
                    da, o.steps, o.tPeak, o.tPeakP0, o.zAtPeak, o.pRel,
                    o.ch.dz, o.ch.zMin, o.ch.zMax - 1, o.ch.yMin);
    // Linan's diffusion-flame regime: the peak-T deficit scales Da^(-1/3),
    // so 8x Da halves it and T(Da -> inf) = T_b + (T_b - T_a).
    const double da = 1 - a.tPeakP0, db = 1 - b.tPeakP0;
    const double tInf = b.tPeakP0 + (b.tPeakP0 - a.tPeakP0);
    std::printf("    deficit ratio Da 500 / Da 4000 = %.4f (Da^-1/3: 2), "
                "extrapolated T(Da -> inf)/T_ad = %.5f\n",
                da / db, tInf);
    auto walk = [](long n) { return 4 * std::ldexp(1.0, -23) * std::sqrt(3.0 * n); };
    check(a.ch.dz < walk(a.steps) && b.ch.dz < walk(b.steps),
          "heated: Z(Y_F, Y_O) equals the tracer within the rounding walk");
    check(a.ch.zMin > -1e-6 && a.ch.zMax < 1 + 1e-6 && a.ch.yMin > -1e-6 &&
              b.ch.zMin > -1e-6 && b.ch.zMax < 1 + 1e-6 && b.ch.yMin > -1e-6,
          "heated: Z, Y in bounds");
    check(db > 0 && db < 0.04, "Da 4000: Tmax 0-4% below T_ad");
    check(da / db > 1.6 && da / db < 2.5,
          "deficit ratio for 8x Da in [1.6, 2.5] (Da^-1/3 scaling: 2)");
    check(std::abs(tInf - 1) < 0.01,
          "extrapolated T(Da -> inf) within 1% of the flame-sheet T_ad");
    check(std::abs(b.zAtPeak - m.zst()) < 0.01,
          "Da 4000: omega peaks at Z_st +- 0.01");
}

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();
    const char* mode = argc > 1 ? argv[1] : "mix";
    if (!std::strcmp(mode, "mix")) mix();
    else if (!std::strcmp(mode, "heat")) heat();
    else {
        std::fprintf(stderr, "usage: test_gpu_diffusion_layer mix|heat\n");
        return 2;
    }
    return dnstest::finish();
}
