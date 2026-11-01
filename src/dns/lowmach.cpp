#include "dns/lowmach.hpp"

#include "dns/bunsen.hpp"
#include "dns/chem.hpp"
#include "dns/flame1d.hpp"

#include <cassert>
#include <cmath>

namespace dns {

bool lmFftSmooth(int n)
{
    assert(n >= 1);
    for (int p : {2, 3, 5, 7})
        while (n % p == 0) n /= p;
    return n == 1;
}

int lmNextSmooth(int n)
{
    assert(n >= 1);
    while (!lmFftSmooth(n)) ++n;
    return n;
}

int lmPrevSmooth(int n)
{
    assert(n >= 1);
    while (n > 1 && !lmFftSmooth(n)) --n;
    return n;
}

LmMix lmPartialPremixMix(const PartialPremixModel& pm, double a, double b,
                        double tB, double cAir, double tAir)
{
    // The products + air part, burnt to completion (Burke-Schumann): s Y_F -
    // Y_O and the enthalpy T + (q/cp) Y_F are kept.
    double yF = pm.yFrem() * b, yO = pm.yO2 * cAir, t = tB + tAir;
    const double qcp = pm.qOverCpTu();
    if (pm.s * yF >= yO) {
        yF -= yO / pm.s;
        t += qcp * yO / pm.s;
        yO = 0;
    } else {
        yO -= pm.s * yF;
        t += qcp * yF;
        yF = 0;
    }
    return {a + t, pm.yF1() * a + yF, pm.yO1() * a + yO};
}

namespace {

// The burner's radial coordinate (bunsen.cpp's rule): |y - yc| (slot) or
// hypot(y - yc, z - zc) (round).
double radiusAt(const BunsenGeom& b, double y, double z)
{
    return b.round ? std::hypot(y - b.yc, z - b.zc) : std::abs(y - b.yc);
}

// The unscaled Poiseuille shape at radius r < R_i (lmPoiseuille's header).
double poiseuilleRaw(const BunsenGeom& b, double h, double r)
{
    const double ri = 0.5 * b.W;
    if (b.round) return 1.0 - (r / ri) * (r / ri);
    // The slot's staircase wall face: the jet cells n = 1.. with centre (n -
    // 1/2) h < R_i (yc is a cell face).
    const int nIn = int(std::ceil(ri / h + 0.5)) - 1;
    const double re = nIn * h, we = 2.0 * re;
    const double d = 0.5 * (std::sqrt(we * we + h * h) - we);
    return (re + d) * (re + d) - r * r;
}

} // namespace

double lmPoiseuilleScale(const BunsenGeom& b, const Grid& g)
{
    double s = 0;
    long n = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            const double r = radiusAt(b, (j + 0.5) * g.hy, (k + 0.5) * g.hz);
            if (!(r < 0.5 * b.W)) continue;
            s += poiseuilleRaw(b, g.hy, r);
            ++n;
        }
    assert(n > 0 && s > 0);
    return double(n) / s;
}

double lmTubeSdf(const BunsenGeom& b, double x, double y, double z)
{
    assert(b.rimLip > 0);
    const double r = radiusAt(b, y, z);
    const double ri = 0.5 * b.W, ro = ri + b.rimLip;
    const double dr = std::max(ri - r, r - ro); // <= 0 within the wall radially
    const double dx = x - b.xRim;               // <= 0 below the rim
    if (dr <= 0 && dx <= 0) return std::max(dr, dx);
    return std::hypot(std::max(dr, 0.0), std::max(dx, 0.0));
}

std::vector<std::uint8_t> lmTubeSolid(const BunsenGeom& b, const Grid& g)
{
    if (!(b.rimLip > 0)) return {};
    std::vector<std::uint8_t> s(g.size(), 0);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                s[g.idx(i, j, k)] =
                    lmTubeSdf(b, (i + 0.5) * g.hx, (j + 0.5) * g.hy, (k + 0.5) * g.hz) <=
                            0.0
                        ? 1
                        : 0;
    return s;
}

std::vector<double> lmPoiseuille(const BunsenGeom& b, const Grid& g)
{
    assert(b.rimLip > 0);
    const double sc = lmPoiseuilleScale(b, g);
    std::vector<double> f(std::size_t(g.ny) * g.nz, 0.0);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            const double r = radiusAt(b, (j + 0.5) * g.hy, (k + 0.5) * g.hz);
            if (r < 0.5 * b.W)
                f[std::size_t(j) + std::size_t(g.ny) * k] = sc * poiseuilleRaw(b, g.hy, r);
        }
    return f;
}

double lmWallHeat(const Grid& g, std::span<const std::uint8_t> solid,
                  std::span<const float> T, double lambdaW, double Tw, bool openY,
                  bool openZ)
{
    assert(solid.size() == g.size() && T.size() == g.size());
    double q = 0;
    const double aX = g.hy * g.hz, aY = g.hx * g.hz, aZ = g.hx * g.hy;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                if (solid[c]) continue;
                const double f = lambdaW * (double(T[c]) - Tw);
                auto face = [&](std::size_t nb, double h, double area) {
                    if (solid[nb]) q += f / (0.5 * h) * area;
                };
                if (i > 0) face(g.idx(i - 1, j, k), g.hx, aX);
                if (i + 1 < g.nx) face(g.idx(i + 1, j, k), g.hx, aX);
                if (j > 0 || !openY) face(g.idx(i, j > 0 ? j - 1 : g.ny - 1, k), g.hy, aY);
                if (j + 1 < g.ny || !openY) face(g.idx(i, j + 1 < g.ny ? j + 1 : 0, k), g.hy, aY);
                if (g.nz > 1) {
                    if (k > 0 || !openZ) face(g.idx(i, j, k > 0 ? k - 1 : g.nz - 1), g.hz, aZ);
                    if (k + 1 < g.nz || !openZ) face(g.idx(i, j, k + 1 < g.nz ? k + 1 : 0), g.hz, aZ);
                }
            }
    return q;
}

LmInflowRow lmInflowRow(const Case& c, double y, double z)
{
    assert(c.flameBunsen && c.physics == Physics::LowMach);
    if (c.lmRim) {
        // S8: sharp rows -- the tube's Poiseuille jet, its closed wall, the
        // room draft of cold air.
        const BunsenGeom& b = c.bunsen;
        const double r = radiusAt(b, y, z), ri = 0.5 * b.W;
        LmInflowRow w{};
        if (r < ri) {
            const double u = b.uJet * b.rimScale * poiseuilleRaw(b, c.grid.hy, r);
            w.t = 1.0;
            w.yF = c.flamePremix2 ? c.premix.yF1() : 1.0;
            w.yO = c.flamePremix2 ? c.premix.yO1() : 0.0;
            w.jet = 1;
            w.m = u / w.t;
        } else if (r <= ri + b.rimLip) {
            w.t = b.tRim; // the tube: a closed face
            w.m = 0;
        } else {
            w.t = b.tCo;
            w.yO = c.flamePremix2 ? c.premix.yO2 : 0.0;
            w.co = 1;
            w.m = b.uCo / w.t;
        }
        return w;
    }
    const BunsenInflowRow in = bunsenInflowAt(c.bunsen, y, z);
    LmInflowRow r;
    r.jet = in.jet;
    r.pilot = in.pilot;
    r.co = in.co;
    if (c.flamePremix2) {
        const BunsenGeom& b = c.bunsen;
        const LmMix m = lmPartialPremixMix(c.premix, in.jet, in.pilot,
                                           b.tPilot * in.pilot, in.co,
                                           b.tCo * in.co);
        r.t = m.t;
        r.yF = m.yF;
        r.yO = m.yO;
    } else {
        r.t = in.t;
        r.yF = in.Y;
        r.yO = 0;
    }
    r.m = in.u / r.t; // rho = 1/t (P0 = R = Tu = 1)
    return r;
}

namespace {

// lowmachBunsenIC with --rim (S8; the header's rule).
LowMachState rimIC(const Case& c)
{
    const Grid& g = c.grid;
    const BunsenGeom& b = c.bunsen;
    const PartialPremixModel& pm = c.premix;
    const bool two = c.flamePremix2;
    const double sig = two ? pm.sigmaAt(pm.phi) : c.flameModel.sigma;
    const double ri = 0.5 * b.W;
    const double xTop = b.xRim + b.coneH + kLmRimColumn * c.flameEig.dthOverDD;
    const std::vector<std::uint8_t> solid = lmTubeSolid(b, g);
    LowMachState s;
    s.nY = two ? 2 : 1;
    s.rho.assign(g.size(), 0.0);
    s.p.assign(g.size(), 0.0);
    s.my.assign(g.size(), 0.0);
    s.mz.assign(g.size(), 0.0);
    s.mx.assign(std::size_t(g.nx + 1) * g.ny * g.nz, 0.0);
    s.rhoY[0].assign(g.size(), 0.0);
    if (two) s.rhoY[1].assign(g.size(), 0.0);
    auto put = [&](std::size_t cc, double t, double yF, double yO) {
        const double rho = 1.0 / t;
        s.rho[cc] = rho;
        s.rhoY[0][cc] = rho * yF;
        if (two) s.rhoY[1][cc] = rho * yO;
    };
    const double yO2 = two ? pm.yO2 : 0.0;
    for (int k = 0; k < g.nz; ++k) {
        const double z = (k + 0.5) * g.hz;
        for (int j = 0; j < g.ny; ++j) {
            const double y = (j + 0.5) * g.hy;
            const LmInflowRow in = lmInflowRow(c, y, z);
            const double r = b.round ? std::hypot(y - b.yc, z - b.zc) : std::abs(y - b.yc);
            const std::size_t row = std::size_t(j) + std::size_t(g.ny) * k;
            // Every x face of a jet or draft row its inflow m; 0 on the
            // tube's rows (wall and the wake above the lip) and on any face
            // touching the tube.
            for (int i = 0; i <= g.nx; ++i) {
                const bool touch = (i > 0 && solid[g.idx(i - 1, j, k)]) ||
                                   (i < g.nx && solid[g.idx(i, j, k)]);
                s.mx[std::size_t(i) + std::size_t(g.nx + 1) * row] =
                    touch || solid[g.idx(0, j, k)] ? 0.0 : in.m;
            }
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t cc = g.idx(i, j, k);
                const double x = (i + 0.5) * g.hx;
                if (solid[cc]) {
                    put(cc, b.tRim, 0.0, 0.0);
                    continue;
                }
                // Below the rim: the row's own gas (fresh premix in the tube,
                // room air outside); --reaction off: everywhere (the lip rows
                // above the rim room air at rest).
                if (x < b.xRim || c.lmInert) {
                    if (r < ri || r > ri + b.rimLip)
                        put(cc, in.t, in.yF, in.yO);
                    else
                        put(cc, b.tCo, 0.0, yO2);
                    continue;
                }
                const double P = 0.25 * (1.0 - std::tanh((r - ri) / b.edge)) *
                                 (1.0 - std::tanh((x - xTop) / b.edge));
                const double tf =
                    flameProfile(c.flameEig.tab, bunsenConeDistanceAt(b, x - b.xRim, y, z));
                const double sp = (sig - tf) / (sig - 1.0);
                if (two) {
                    const LmMix mx = lmPartialPremixMix(pm, P * sp, P * (1 - sp),
                                                        sig * P * (1 - sp), 1.0 - P,
                                                        b.tCo * (1.0 - P));
                    put(cc, mx.t, mx.yF, mx.yO);
                } else {
                    put(cc, tf * P + b.tCo * (1.0 - P), sp * P, 0.0);
                }
            }
        }
    }
    return s;
}

} // namespace

LowMachState lowmachBunsenIC(const Case& c)
{
    assert(c.sub == Subcommand::Flame && c.physics == Physics::LowMach &&
           c.flameBunsen && !c.flameEig.tab.T.empty() && c.flameTu == 1.0);
    if (c.lmRim) return rimIC(c);
    const Grid& g = c.grid;
    const BunsenGeom& b = c.bunsen;
    const double sig = c.flamePremix2 ? c.premix.sigmaAt(c.premix.phi)
                                      : c.flameModel.sigma;
    LowMachState s;
    s.nY = c.flamePremix2 ? 2 : 1;
    s.rho.assign(g.size(), 0.0);
    s.p.assign(g.size(), 0.0);
    s.my.assign(g.size(), 0.0);
    s.mz.assign(g.size(), 0.0);
    s.mx.assign(std::size_t(g.nx + 1) * g.ny * g.nz, 0.0);
    s.rhoY[0].assign(g.size(), 0.0);
    if (s.nY == 2) s.rhoY[1].assign(g.size(), 0.0);
    const PartialPremixModel& pm = c.premix;
    for (int k = 0; k < g.nz; ++k) {
        const double z = (k + 0.5) * g.hz;
        for (int j = 0; j < g.ny; ++j) {
            const double y = (j + 0.5) * g.hy;
            const LmInflowRow in = lmInflowRow(c, y, z);
            const std::size_t row = std::size_t(j) + std::size_t(g.ny) * k;
            for (int i = 0; i <= g.nx; ++i)
                s.mx[std::size_t(i) + std::size_t(g.nx + 1) * row] = in.m;
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t cc = g.idx(i, j, k);
                if (c.lmInert) {
                    // --reaction off (S6): the inflow row extruded, no cone.
                    const double rho = 1.0 / in.t;
                    s.rho[cc] = rho;
                    s.rhoY[0][cc] = rho * in.yF;
                    if (s.nY == 2) s.rhoY[1][cc] = rho * in.yO;
                    continue;
                }
                const double x = (i + 0.5) * g.hx;
                const double tf =
                    flameProfile(c.flameEig.tab, bunsenConeDistanceAt(b, x, y, z));
                const double sp = (sig - tf) / (sig - 1.0); // the jet's progress
                if (s.nY == 2) {
                    // Fresh jet a = J s_p; products b = J (1 - s_p) + P at
                    // sigma / tPilot; air C: its overlap with the products
                    // laid burnt (the outer sheet's Burke-Schumann state).
                    const LmMix mx = lmPartialPremixMix(
                        pm, in.jet * sp, in.jet * (1 - sp) + in.pilot,
                        sig * in.jet * (1 - sp) + b.tPilot * in.pilot, in.co,
                        b.tCo * in.co);
                    const double rho = 1.0 / mx.t;
                    s.rho[cc] = rho;
                    s.rhoY[0][cc] = rho * mx.yF;
                    s.rhoY[1][cc] = rho * mx.yO;
                } else {
                    const double t = tf * in.jet + b.tPilot * in.pilot + b.tCo * in.co;
                    const double rho = 1.0 / t;
                    s.rho[cc] = rho;
                    s.rhoY[0][cc] = rho * sp * in.jet;
                }
            }
        }
    }
    return s;
}

LowMachState lowmachIC(const Case& c)
{
    if (c.flameBunsen) return lowmachBunsenIC(c);
    assert(c.sub == Subcommand::Flame && c.physics == Physics::LowMach &&
           !c.flameEig.tab.T.empty() && c.flameTu == 1.0);
    const Grid& g = c.grid;
    const FlameEigen& tab = c.flameEig.tab;
    const double sigma = c.flamePremix2 ? c.premix.sigmaAt(c.premixBurnPhi())
                                        : c.flameModel.sigma;
    const double A = c.flameWrinkle * c.flameEig.dthOverDD;
    const double Ly = g.ny * g.hy;
    const double m = c.flameUIn; // rho_u u_in, rho_u = 1
    LowMachState s;
    s.nY = c.flamePremix2 ? 2 : 1;
    s.rho.assign(g.size(), 0.0);
    s.p.assign(g.size(), 0.0);
    s.my.assign(g.size(), 0.0);
    s.mz.assign(g.size(), 0.0);
    s.mx.assign(std::size_t(g.nx + 1) * g.ny * g.nz, m);
    std::vector<double> rhoY[2];
    rhoY[1].assign(g.size(), 0.0);
    std::vector<double> T(g.nx);
    for (int j = 0; j < g.ny; ++j) {
        const double y = (j + 0.5) * g.hy;
        const double xf =
            c.flameXf0 + flameWrinkle(A, c.flameWrinkleModes, y, Ly);
        for (int i = 0; i < g.nx; ++i)
            T[std::size_t(i)] = flameProfile(tab, (i + 0.5) * g.hx - xf);
        const double tLast = T[std::size_t(g.nx - 1)];
        for (int i = 0; i < g.nx; ++i) {
            const double t = T[std::size_t(i)];
            const double rho = 1.0 / t;
            for (int k = 0; k < g.nz; ++k) {
                const std::size_t cc = g.idx(i, j, k);
                s.rho[cc] = rho;
                s.p[cc] = m * m * (tLast - t);
                rhoY[1][cc] = rho * ((sigma - t) / (sigma - 1));
            }
        }
    }
    if (c.flamePremix2) {
        premix2Reactants(c, s.rho, rhoY);
        s.rhoY[0] = std::move(rhoY[0]);
        s.rhoY[1] = std::move(rhoY[1]);
    } else {
        s.rhoY[0] = std::move(rhoY[1]);
    }
    return s;
}

} // namespace dns
