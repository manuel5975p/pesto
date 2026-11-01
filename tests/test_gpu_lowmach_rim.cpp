// The immersed isothermal tube rim on the low-Mach solver (masterplan
// real_bunsen S8; docs/chem.md "Low-Mach rim"). channel / cond / imperm drive
// DeviceLowMach through its API with LowMachSetup::solid (the staircase of
// dns/lowmach.hpp lmTubeSolid); anchor runs the dns binary.
//
//   channel  a 2D inert channel between two lips (W 8 = 16 cells, lips 4
//            cells, tube 96 of 128 cells, Re U W/nu 11, periodic laterals,
//            room draft 0.25): the developed Poiseuille rows (lmPoiseuille:
//            the scheme's own discrete solution on the staircase) preserved
//            inside the tube (columns 8..40, >= 3.5 W from the exit) to 1e-4
//            of U, v to 1e-5; dp/dx = the profile's discrete curvature mu
//            D2 u to 1e-3 and = -12 mu U / W^2 to 1%; the wall shear 2 mu u_1
//            / h balances -dp/dx W / 2 per wall to 1e-3 (and 6 mu U / W to
//            1%); every face touching the tube carries m == 0 EXACTLY; mass =
//            the FP64 tally to 1e-6.
//   cond     conduction against the isothermal wall: 2D, open laterals, a
//            solid slab across y (i 16..19 of 32), the x-lo face at T 2 with
//            m 0, the slab at T_w 1, constant mu: the LINEAR profile T = 2 -
//            x/L is the discrete fixed point (the wall face's rho D(T_w)
//            (T_c - T_w)/(h/2) is exact for it) -- after 500 steps T within
//            2e-6 of it, |m| <= 1e-6; qRim (lmWallHeat: the stage's face
//            flux) = lambda dT/L Ly = 3.5 to 2e-6 and equals the inflow face's
//            conduction (closure); with T_w 1.5 the same with dT 0.5.
//   imperm   3D round tube (D 4 = 8 cells, lip 3 cells, 16 cells tall) with
//            a HOT Poiseuille jet (T 2), open y and z, gravity 0.5, a uniform
//            power-of-two passive Y = 0.5 everywhere fluid, 300 steps: every
//            face touching the tube m == 0 EXACTLY (x, y, z, the open hi
//            faces); the tube's rho and rho Y unchanged BITWISE; rho Y == 0.5
//            rho bitwise in every fluid cell (the wall faces carry exactly
//            nothing and the face values next to them stay exact); mass and
//            species inventories = their FP64 tallies to 1e-6 / 1e-6; the PCG
//            warm gate <= 15 iterations per stage solve after the first step;
//            a rerun byte-identical.
//   anchor   the plan's anchoring case through the binary: 2D rich phi 1.3
//            slot (C2b, sigma 6.88, Sutherland 298 K, Pr 0.41: mapping M1), W
//            14 delta_th, U 6, lip 1.4, 4 cells, open laterals, tt 40, the
//            burner's default dt rule (S9: --dt-rule cell; the S8 default,
//            now --dt-rule bounds, overshot T to 7.14 and counted 6400 Y hits
//            here -- LM10): anchored ON the rim
//            (xRimBase in (0, 4 delta_th], window [30, 40] halves within 1
//            cell), no flash-back (exit 0, the base above the rim), the base
//            radius over the lip (rBase in [R_i, R_o + 1 delta_th]), pcgIt <=
//            15 for t > 1, yHits 0, heat into the tube qRim > 0; the same
//            with --rim-temp 1.5 (standoff and qRim reported).
//
// Usage: test_gpu_lowmach_rim channel|cond|imperm|anchor. No HIP device -> 77.
#include "dns/bunsen.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/lowmach.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/lowmach.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using namespace dns;
using namespace dns::gpu;
using dnstest::check;

std::size_t fx(const Grid& g, int i, int j, int k)
{
    return std::size_t(i) + std::size_t(g.nx + 1) * (std::size_t(j) + std::size_t(g.ny) * k);
}

// An inert setup (lnA = -inf), mu = Pr = 0.72 constant, one species.
LowMachSetup inert()
{
    LowMachSetup s;
    s.tr = Transport::air(0.72, 1.0, 0.0, 0.72, kGammaAir, kRgasBox, false);
    s.prGas = 1;
    s.nY = 1;
    s.fuel = 0;
    s.lnA = -HUGE_VAL;
    s.Ta = 50;
    s.qcp = 4;
    s.TIn = 1;
    s.yIn[0] = 1;
    return s;
}

struct Runner
{
    std::optional<DeviceLowMach> d;
    std::optional<ReduceScratch> red;
    long steps = 0;
    int pcgFirst = -1, pcgWarmMax = 0;
    bool ok = false;

    Runner(const Grid& g, const LowMachSetup& s, const LowMachState& ic)
    {
        auto m = DeviceLowMach::make(g, s);
        auto r = ReduceScratch::make();
        if (!m || !r) {
            std::printf("    make failed: %s\n", m ? "reduce" : m.error().c_str());
            return;
        }
        d.emplace(std::move(*m));
        red.emplace(std::move(*r));
        if (auto e = d->upload(ic); !e) {
            std::printf("    upload failed: %s\n", e.error().c_str());
            return;
        }
        ok = true;
    }
    bool advance(int n, double dtFixed = 0)
    {
        for (int q = 0; ok && q < n; ++q) {
            double dt = dtFixed;
            if (dt <= 0)
                dt = 0.95 * std::min(d->convectiveDt(*red, 0.8), d->diffusiveDt(*red));
            if (auto e = d->step(dt); !e) {
                std::printf("    step failed: %s\n", e.error().c_str());
                ok = false;
            }
            const int it = d->pcgIterations();
            if (steps == 0) pcgFirst = it;
            else pcgWarmMax = std::max(pcgWarmMax, it);
            ++steps;
        }
        return ok;
    }
    // The current state, hi words only (nullopt if the run failed).
    std::optional<LowMachState> state() const
    {
        if (!ok) return std::nullopt;
        auto s = d->download(true);
        if (!s) return std::nullopt;
        return std::move(*s);
    }
};

// Every MAC face touching a solid cell carries m == 0 exactly. Lat: open
// axes clamp (the open boundary faces: closed iff their cell is solid).
bool closedFacesZero(const Grid& g, const std::vector<std::uint8_t>& solid,
                     const LowMachState& s, bool openY, bool openZ, long& nClosed)
{
    auto sol = [&](int i, int j, int k) {
        if (i < 0 || i >= g.nx) return false;
        j = openY ? std::clamp(j, 0, g.ny - 1) : (j + g.ny) % g.ny;
        k = openZ ? std::clamp(k, 0, g.nz - 1) : (k + g.nz) % g.nz;
        return solid[g.idx(i, j, k)] != 0;
    };
    bool ok = true;
    nClosed = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 0; i <= g.nx; ++i)
                if (sol(i - 1, j, k) || sol(i, j, k)) {
                    ++nClosed;
                    ok = ok && s.mx[fx(g, i, j, k)] == 0.0;
                }
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                if (sol(i, j - 1, k) || sol(i, j, k)) {
                    ++nClosed;
                    ok = ok && s.my[c] == 0.0;
                }
                if (g.nz > 1 && (sol(i, j, k - 1) || sol(i, j, k))) {
                    ++nClosed;
                    ok = ok && s.mz[c] == 0.0;
                }
                if (openY && j == g.ny - 1 && sol(i, j, k)) {
                    ++nClosed;
                    ok = ok && s.myHi[std::size_t(i) + std::size_t(g.nx) * k] == 0.0;
                }
                if (openZ && k == g.nz - 1 && sol(i, j, k)) {
                    ++nClosed;
                    ok = ok && s.mzHi[std::size_t(i) + std::size_t(g.nx) * j] == 0.0;
                }
            }
        }
    return ok;
}

// The tube's IC on a grid: rho = 1/T (T from tAt for fluid, T_w solid), rho Y
// = Y rho (0 solid), every x face of a fluid inflow row the row's m, 0 on
// faces touching the tube and on the tube's rows; laterals 0.
LowMachState tubeIC(const Grid& g, const std::vector<std::uint8_t>& solid,
                    const LowMachSetup& s, double Tw, double Y,
                    const std::vector<double>& rowM,
                    const std::function<double(int, int, int)>& tAt)
{
    LowMachState q;
    q.nY = 1;
    q.rho.assign(g.size(), 0);
    q.rhoY[0].assign(g.size(), 0);
    q.p.assign(g.size(), 0);
    q.my.assign(g.size(), 0);
    q.mz.assign(g.size(), 0);
    q.mx.assign(std::size_t(g.nx + 1) * g.ny * g.nz, 0);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            const std::size_t row = std::size_t(j) + std::size_t(g.ny) * k;
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double t = solid[c] ? Tw : tAt(i, j, k);
                q.rho[c] = 1.0 / t;
                q.rhoY[0][c] = solid[c] ? 0.0 : Y / t;
            }
            for (int i = 0; i <= g.nx; ++i) {
                const bool touch = (i > 0 && solid[g.idx(i - 1, j, k)]) ||
                                   (i < g.nx && solid[g.idx(i, j, k)]);
                q.mx[fx(g, i, j, k)] = touch || solid[g.idx(0, j, k)] ? 0.0 : rowM[row];
            }
        }
    (void)s;
    return q;
}

// --- channel -------------------------------------------------------------------

int channel()
{
    const double h = 0.5, U = 1.0, uCo = 0.25;
    const Grid g = Grid::make(128, 32, 1, 128 * h, 32 * h, 1.0);
    BunsenGeom b;
    b.W = 8;
    b.rimLip = 2;
    b.xRim = 96 * h;
    b.yc = 16 * h;
    const auto solid = lmTubeSolid(b, g);
    const auto f = lmPoiseuille(b, g);
    LowMachSetup s = inert();
    s.solid = solid;
    s.wallT = 1;
    const double ro = 0.5 * b.W + b.rimLip;
    for (int j = 0; j < g.ny; ++j) {
        const double r = std::abs((j + 0.5) * h - b.yc);
        const double m = f[std::size_t(j)] > 0 ? U * f[std::size_t(j)] : r > ro ? uCo : 0.0;
        s.rowM.push_back(m);
        s.rowT.push_back(1.0);
        s.rowY[0].push_back(1.0);
    }
    const LowMachState ic =
        tubeIC(g, solid, s, 1.0, 1.0, s.rowM, [](int, int, int) { return 1.0; });
    Runner run(g, s, ic);
    run.advance(3000);
    auto st = run.state();
    check(st.has_value(), std::format("channel: 3000 steps ran (PCG first step {}, "
                                      "warm max {})",
                                      run.pcgFirst, run.pcgWarmMax));
    if (!st) return dnstest::finish();
    // (a) the profile inside the tube, columns 8..40 (the faces between).
    double du = 0, dv = 0;
    for (int j = 0; j < g.ny; ++j) {
        if (f[std::size_t(j)] == 0) continue;
        for (int i = 8; i <= 40; ++i) {
            du = std::max(du, std::abs(st->mx[fx(g, i, j, 0)] - s.rowM[std::size_t(j)]));
            dv = std::max(dv, std::abs(st->my[g.idx(i, j, 0)]));
        }
    }
    // (b) dp/dx on the axis (the two rows either side), least squares.
    std::vector<double> xs, ps;
    for (int i = 8; i <= 40; ++i) {
        xs.push_back((i + 0.5) * h);
        ps.push_back(0.5 * (st->p[g.idx(i, 15, 0)] + st->p[g.idx(i, 16, 0)]));
    }
    double mx = 0, my = 0;
    for (std::size_t q = 0; q < xs.size(); ++q) {
        mx += xs[q];
        my += ps[q];
    }
    mx /= double(xs.size());
    my /= double(xs.size());
    double sxx = 0, sxy = 0;
    for (std::size_t q = 0; q < xs.size(); ++q) {
        sxx += (xs[q] - mx) * (xs[q] - mx);
        sxy += (xs[q] - mx) * (ps[q] - my);
    }
    const double dpdx = sxy / sxx;
    const double mu = 0.72;
    const double lap = (f[15] - 2 * f[16] + f[17]) / (h * h); // the curvature
    const double we = 16 * h;
    const double dpdxDisc = mu * U * lap, dpdxPois = -12 * mu * U / (we * we);
    // (c) the wall shear: 2 mu u_1/h at the first jet row (j 8 and 23).
    const double u1 = 0.5 * (st->mx[fx(g, 24, 8, 0)] + st->mx[fx(g, 24, 23, 0)]);
    const double tauW = 2 * mu * u1 / h;
    std::printf("  channel: max |m - m_row| %.3e, max |v| %.3e, dp/dx %.6f (discrete "
                "%.6f, 12 mu U/W^2 %.6f), wall shear %.6f (dp/dx W/2 %.6f, 6 mu U/W "
                "%.6f)\n",
                du, dv, dpdx, dpdxDisc, dpdxPois, tauW, -dpdx * we / 2,
                6 * mu * U / we);
    check(du <= 1e-4 * U, std::format("channel: Poiseuille preserved in the tube "
                                      "(max |m - m_row| {:.2e} <= 1e-4 U)",
                                      du));
    check(dv <= 1e-5 * U, std::format("channel: v ~ 0 in the tube ({:.2e})", dv));
    check(std::abs(dpdx / dpdxDisc - 1) <= 1e-3,
          std::format("channel: dp/dx = mu D2 u of the rows (ratio {:.6f})",
                      dpdx / dpdxDisc));
    check(std::abs(dpdx / dpdxPois - 1) <= 1e-2,
          std::format("channel: dp/dx = -12 mu U/W^2 within 1% (ratio {:.6f})",
                      dpdx / dpdxPois));
    check(std::abs(tauW / (-dpdx * we / 2) - 1) <= 1e-3,
          std::format("channel: wall shear balances the pressure drop (ratio {:.6f})",
                      tauW / (-dpdx * we / 2)));
    check(std::abs(tauW / (6 * mu * U / we) - 1) <= 1e-2,
          std::format("channel: wall shear = 6 mu U/W within 1% (ratio {:.6f})",
                      tauW / (6 * mu * U / we)));
    long nc = 0;
    check(closedFacesZero(g, solid, *st, false, false, nc),
          std::format("channel: all {} faces touching the tube carry m == 0 exactly",
                      nc));
    auto full = run.d->download(false);
    if (full) {
        double mass = 0;
        for (double v : full->rho) mass += v;
        mass *= g.hx * g.hy * g.hz;
        const double tally = run.d->massTally();
        check(std::abs(mass - tally) <= 1e-6 * tally,
              std::format("channel: mass = the FP64 tally ({:.2e})",
                          (mass - tally) / tally));
    }
    return dnstest::finish();
}

// --- cond ------------------------------------------------------------------------

void condCase(double Tw)
{
    const double h = 0.5, Tin = 2.0;
    const Grid g = Grid::make(32, 16, 1, 32 * h, 16 * h, 1.0);
    std::vector<std::uint8_t> solid(g.size(), 0);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 16; i < 20; ++i) solid[g.idx(i, j, 0)] = 1;
    LowMachSetup s = inert();
    s.solid = solid;
    s.wallT = Tw;
    s.openY = true;
    s.ambT = Tw; // the room above the slab at the wall's T: nothing moves there
    s.ambY[0] = 1;
    s.mIn = 0;
    s.TIn = Tin;
    const double L = 16 * h;
    auto tLin = [&](int i) { return Tin - (Tin - Tw) * (i + 0.5) * h / L; };
    const LowMachState ic = tubeIC(g, solid, s, Tw, 1.0,
                                   std::vector<double>(std::size_t(g.ny), 0.0),
                                   [&](int i, int, int) { return i < 16 ? tLin(i) : Tw; });
    Runner run(g, s, ic);
    run.advance(500, 0.02);
    auto st = run.state();
    check(st.has_value(), std::format("cond T_w {}: 500 steps ran", Tw));
    if (!st) return;
    double dT = 0, mMax = 0;
    std::vector<float> T(g.size());
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            T[c] = float(1.0 / st->rho[c]);
            if (!solid[c]) dT = std::max(dT, std::abs(T[c] - (i < 16 ? tLin(i) : Tw)));
        }
    for (double v : st->mx) mMax = std::max(mMax, std::abs(v));
    for (double v : st->my) mMax = std::max(mMax, std::abs(v));
    for (double v : st->myHi) mMax = std::max(mMax, std::abs(v));
    const double lambda = 3.5 * 0.72 / 0.72; // cp mu/Pr
    const double q = lmWallHeat(g, solid, T, lambda, Tw, true, false);
    const double qExact = lambda * (Tin - Tw) / L * g.ny * h;
    // The inflow face's conduction into the box (the stage's form).
    double qIn = 0;
    for (int j = 0; j < g.ny; ++j)
        qIn += lambda * (Tin - double(T[g.idx(0, j, 0)])) / (0.5 * h) * h;
    std::printf("  cond T_w %g: max |T - T_lin| %.3e, max |m| %.3e, qRim %.9f (exact "
                "%.9f), inflow conduction %.9f\n",
                Tw, dT, mMax, q, qExact, qIn);
    check(dT <= 2e-6, std::format("cond T_w {}: the linear profile is the fixed "
                                  "point ({:.2e})",
                                  Tw, dT));
    check(mMax <= 1e-6, std::format("cond T_w {}: at rest ({:.2e})", Tw, mMax));
    check(std::abs(q / qExact - 1) <= 2e-6,
          std::format("cond T_w {}: qRim = lambda dT/L Ly ({:.2e})", Tw, q / qExact - 1));
    check(std::abs(qIn / q - 1) <= 2e-6,
          std::format("cond T_w {}: closure, heat in = heat into the wall ({:.2e})", Tw,
                      qIn / q - 1));
}

int cond()
{
    condCase(1.0);
    condCase(1.5);
    return dnstest::finish();
}

// --- imperm -----------------------------------------------------------------------

struct Imperm
{
    Grid g;
    std::vector<std::uint8_t> solid;
    LowMachSetup s;
    LowMachState ic;
};

Imperm impermCase()
{
    Imperm c;
    const double h = 0.5, U = 1.0, Tj = 2.0, uCo = 0.2, Y = 0.5;
    c.g = Grid::make(40, 24, 24, 40 * h, 24 * h, 24 * h);
    const Grid& g = c.g;
    BunsenGeom b;
    b.round = true;
    b.W = 4;
    b.rimLip = 1.5;
    b.xRim = 16 * h;
    b.yc = b.zc = 12 * h;
    c.solid = lmTubeSolid(b, g);
    const auto f = lmPoiseuille(b, g);
    LowMachSetup& s = c.s;
    s = inert();
    s.solid = c.solid;
    s.wallT = 1;
    s.openY = s.openZ = true;
    s.ambT = 1;
    s.ambY[0] = Y;
    s.gravity = 0.5;
    s.yIn[0] = Y;
    const double ro = 0.5 * b.W + b.rimLip;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            const std::size_t q = std::size_t(j) + std::size_t(g.ny) * k;
            const double r = std::hypot((j + 0.5) * h - b.yc, (k + 0.5) * h - b.zc);
            const bool jet = f[q] > 0;
            s.rowT.push_back(jet ? Tj : 1.0);
            s.rowM.push_back(jet ? U * f[q] / Tj : r > ro ? uCo : 0.0);
            s.rowY[0].push_back(Y);
        }
    auto tAt = [&](int i, int j, int k) {
        const std::size_t q = std::size_t(j) + std::size_t(g.ny) * k;
        return f[q] > 0 && i < 16 ? Tj : 1.0;
    };
    c.ic = tubeIC(g, c.solid, s, 1.0, Y, s.rowM, tAt);
    return c;
}

int imperm()
{
    const Imperm c = impermCase();
    const Grid& g = c.g;
    Runner run(g, c.s, c.ic);
    run.advance(300);
    auto st = run.state();
    check(st.has_value(), "imperm: 300 steps ran");
    if (!st) return dnstest::finish();
    long nc = 0;
    check(closedFacesZero(g, c.solid, *st, true, true, nc),
          std::format("imperm: all {} faces touching the tube carry m == 0 EXACTLY "
                      "(x, y, z, open hi faces)",
                      nc));
    bool solidSame = true, yExact = true;
    long nSolid = 0;
    double mMax = 0;
    for (std::size_t q = 0; q < g.size(); ++q) {
        if (c.solid[q]) {
            ++nSolid;
            solidSame = solidSame && float(st->rho[q]) == float(c.ic.rho[q]) &&
                        st->rhoY[0][q] == 0.0;
        } else {
            yExact = yExact && float(st->rhoY[0][q]) == 0.5f * float(st->rho[q]);
        }
    }
    for (double v : st->mx) mMax = std::max(mMax, std::abs(v));
    check(solidSame, std::format("imperm: the tube's {} cells keep rho and rho Y "
                                 "(0) bitwise",
                                 nSolid));
    check(yExact, "imperm: rho Y == 0.5 rho bitwise in every fluid cell (the wall "
                  "faces carry nothing; the faces beside them stay exact)");
    auto full = run.d->download(false);
    if (full) {
        double mass = 0, inv = 0;
        for (double v : full->rho) mass += v;
        for (double v : full->rhoY[0]) inv += v;
        const double dV = g.hx * g.hy * g.hz;
        mass *= dV;
        inv *= dV;
        const double mt = run.d->massTally(), yt = run.d->speciesTally(0);
        std::printf("  imperm: mass - tally %.2e, species - tally %.2e (rel), max |m_x| "
                    "%.3f, PCG first step %d, warm max %d\n",
                    (mass - mt) / mt, (inv - yt) / yt, mMax, run.pcgFirst,
                    run.pcgWarmMax);
        check(std::abs(mass - mt) <= 1e-6 * mt, "imperm: mass = the FP64 tally");
        check(std::abs(inv - yt) <= 1e-6 * yt,
              "imperm: species inventory = the FP64 boundary tally (nothing through "
              "the wall)");
    }
    check(run.pcgWarmMax <= 15,
          std::format("imperm: PCG warm gate <= 15 iterations per stage solve (max "
                      "{}, first step {})",
                      run.pcgWarmMax, run.pcgFirst));
    // A rerun is byte-identical (deterministic PCG, warm starts, kernels).
    Runner again(g, c.s, c.ic);
    again.advance(300);
    auto s2 = again.state();
    check(s2 && s2->rho == st->rho && s2->mx == st->mx && s2->my == st->my &&
              s2->mz == st->mz && s2->p == st->p && s2->rhoY[0] == st->rhoY[0],
          "imperm: a rerun is byte-identical");
    return dnstest::finish();
}

// --- anchor -----------------------------------------------------------------------

struct Win
{
    double mean = NAN, a = NAN, b = NAN, mx = -HUGE_VAL;
};
Win window(const valtest::CsvTable& t, const std::string& c, double t0, double t1)
{
    double s = 0, sa = 0, sb = 0;
    int n = 0, na = 0, nb = 0;
    Win w;
    const auto& tt = t.col("t");
    const auto& v = t.col(c);
    for (std::size_t i = 0; i < tt.size(); ++i) {
        if (tt[i] < t0 - 1e-9 || tt[i] > t1 + 1e-9) continue;
        s += v[i];
        ++n;
        w.mx = std::max(w.mx, v[i]);
        if (tt[i] < 0.5 * (t0 + t1)) {
            sa += v[i];
            ++na;
        } else {
            sb += v[i];
            ++nb;
        }
    }
    w.mean = n ? s / n : NAN;
    w.a = na ? sa / na : NAN;
    w.b = nb ? sb / nb : NAN;
    return w;
}

const std::string kAnchor =
    "flame --mode bunsen --physics lowmach --equivalence 1.3 --expansion 6.88 "
    "--visc sutherland --tinf 298 --prandtl 0.41 --slot-width 14 --jet-speed 6 "
    "--rim 1.4 --laterals open --coflow-width 6 --cells-per-delta 4 "
    "--tt 40 --emit stats ";

int anchor()
{
    const double dth = 4.70964, h = dth / 4;
    const double ri = 7 * dth, ro = 8.4 * dth;
    for (const double tw : {1.0, 1.5}) {
        const std::string pre = std::format("/tmp/dns_test_lm_rim_anchor_{:g}", tw);
        const int rc = valtest::runBinary(std::format("\"{}\" {} --rim-temp {:g} --quiet "
                                                      "--out {}",
                                                      DNS_BIN, kAnchor, tw, pre));
        check(rc == 0, std::format("anchor T_w {}: exits 0 (anchored, no flash-back, "
                                   "PCG converged)",
                                   tw));
        auto t = valtest::loadCsv(pre + "_stats.csv");
        check(t.has_value(), std::format("anchor T_w {}: stats readable", tw));
        if (rc != 0 || !t) continue;
        const Win base = window(*t, "xRimBase", 30, 40);
        const Win rb = window(*t, "rBase", 30, 40);
        const Win q = window(*t, "qRim", 30, 40);
        const Win pre2 = window(*t, "preFrac", 30, 40);
        const Win pcg = window(*t, "pcgIt", 1, 40);
        std::printf("  anchor T_w %g: xRimBase %.3f (%.3f delta_th; halves %.3f / "
                    "%.3f), rBase %.3f (R_i %.2f, R_o %.2f), qRim %.3f, preFrac %.4f, "
                    "pcgIt max %g\n",
                    tw, base.mean, base.mean / dth, base.a, base.b, rb.mean, ri, ro,
                    q.mean, pre2.mean, pcg.mx);
        check(base.mean > 0 && base.mean <= 4 * dth,
              std::format("anchor T_w {}: the base sits on the rim, 0 < xRimBase "
                          "<= 4 delta_th ({:.3f})",
                          tw, base.mean));
        check(std::abs(base.a - base.b) <= h,
              std::format("anchor T_w {}: the base is flat in time (halves within "
                          "one cell)",
                          tw));
        check(rb.mean >= ri && rb.mean <= ro + dth,
              std::format("anchor T_w {}: the base radius over the lip ({:.3f})", tw,
                          rb.mean));
        check(pcg.mx <= 15, std::format("anchor T_w {}: PCG <= 15 iterations per "
                                        "stage solve (max {:g})",
                                        tw, pcg.mx));
        const Win hits = window(*t, "yHits", 0, 40);
        check(hits.mx == 0, std::format("anchor T_w {}: Y bounded at the default "
                                        "dt rule (yHits max {:g})",
                                        tw, hits.mx));
        if (tw == 1.0)
            check(q.mean > 0, "anchor T_w 1: heat flows into the cold tube");
    }
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    const std::string m = argc > 1 ? argv[1] : "channel";
    if (m == "cond") return cond();
    if (m == "imperm") return imperm();
    if (m == "anchor") return anchor();
    return channel();
}
