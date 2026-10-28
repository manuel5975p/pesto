// `dns flame --mode chamber` geometry layer (docs/chem.md, "Chamber"): a 2D
// closed combustion chamber as an immersed ghost-wall body -- back wall,
// straight section, converging nozzle -- with a bluff wedge flame holder
// inside, fed premixed reactants by an injector band across the whole back
// wall. Pure host: polygon SDFs, the per-cell region codes the runner's
// balances read, the IC, and the 1D isentropic nozzle estimate. No GPU, no
// filesystem.
#pragma once

#include "dns/comp.hpp"
#include "dns/grid.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace dns {

// The chamber in box units (delta_D), axis along +x at y = yc. Inner
// cavity: back face xb() = x0 + tw, straight section [xb, xn] of height Hc,
// linear converging nozzle [xn, xe] down to height He. Walls tw thick (the
// outer contour is the inner one shifted by tw vertically; the lip at xe is
// a vertical face tw tall). Holder: a solid wedge, apex at (xg, yc) pointing
// upstream, base Hg tall at xg + Lg. Injector: cavity cells with x < xb + wInj.
struct ChamberSpec
{
    double x0 = 0, yc = 0, tw = 0;
    double Hc = 0, Lc = 0, Ln = 0, He = 0;
    double xg = 0, Lg = 0, Hg = 0;
    double wInj = 0;

    double xb() const { return x0 + tw; }
    double xn() const { return xb() + Lc; }
    double xe() const { return xn() + Ln; }
    double xBase() const { return xg + Lg; } // holder base = IC interface
};

using Point2 = std::array<double, 2>;

// Signed distance to a simple closed polygon (vertices in order, either
// orientation): the exact Euclidean distance to its edges, negative inside
// (even-odd crossing rule). Pre: poly.size() >= 3, no self-intersection.
double polygonSdf(std::span<const Point2> poly, double x, double y);

// The C-shaped wall section (outer contour + inner contour, joined at the
// lips), the holder wedge, and the cavity (inner contour closed by the exit
// plane x = xe).
std::vector<Point2> chamberWallPolygon(const ChamberSpec& s);
std::vector<Point2> holderPolygon(const ChamberSpec& s);
std::vector<Point2> cavityPolygon(const ChamberSpec& s);

// sdf > 0 in fluid, < 0 in solid (wall U holder): what DeviceCompGhost
// classifies: min of the two polygon distances (negative inside either), the
// exact distance to the union of the two disjoint bodies.
double chamberSdf(const ChamberSpec& s, double x, double y);

// Per-cell region codes, read by the runner's balances (one code per cell,
// gidx order). Cavity = Cavity..Exit.
enum class ChamberRegion : std::uint8_t
{
    Solid = 0,        // sdf < 0 at the centre: ghost or deep, never fluid
    Ambient = 1,      // fluid outside the cavity
    Cavity = 2,       // fluid inside the cavity, none of the below
    Injector = 3,     // the injector band: x < xb + wInj
    FreshProbe = 4,   // x in [xb + wInj + h, xg - h]: the fresh plenum
    NozzleProbe = 5,  // x in [xn - 1.6, xn): just ahead of the contraction
    Exit = 6,         // the last cavity column with centre x < xe
    AmbientProbe = 7, // ambient cells of that same column (beside the lips)
};
inline constexpr int kChamberRegions = 8;
inline bool inCavity(std::uint8_t r)
{
    return r >= std::uint8_t(ChamberRegion::Cavity) &&
           r <= std::uint8_t(ChamberRegion::Exit);
}

// The region of every cell of g (centres at (i+0.5)h). Pre: g.nz == 1.
std::vector<std::uint8_t> chamberRegions(const Grid& g, const ChamberSpec& s);

// The chamber IC, all at p0 = Tu (rho_u = R = 1): cavity fresh (T = Tu,
// Y = 1, at rest) upstream of the holder base and burnt downstream, joined by
// the oracle profile theta(x - xBase) = T/Tu in [1, sigma] (Y = (sigma -
// theta)/(sigma - 1), rho = p0/T); ambient cold inert gas (Tu, Y = 0) moving
// at uCo; solid cells as ambient at rest with rhoY = 0 (the ghost fill owns
// their mean flow; rhoY stays 0 there for the whole run).
// Pre: regions from chamberRegions(g, s), theta maps into [1, sigma].
void chamberIC(const Grid& g, const std::vector<std::uint8_t>& regions,
               double gamma, double Tu, double sigma, double uCo,
               const std::function<double(double)>& theta, double xBase,
               CompState& q, std::vector<double>& rhoY);

// 1D isentropic nozzle: the exit Mach Me and the total pressure p0 that pass
// mass flux mdot (per unit depth) of total temperature T0 through exit area
// A at exit static pressure pa: mdot/A = pa Me sqrt(gamma (1 + (gamma-1)/2
// Me^2) / (R T0)), p0 = pa (1 + (gamma-1)/2 Me^2)^(gamma/(gamma-1)).
// Error when the flux exceeds the sonic one (choked: no subsonic solution).
// Pre: all arguments finite and > 0, gamma > 1.
struct NozzleEstimate
{
    double Me = 0, p0 = 0;
};
std::expected<NozzleEstimate, std::string>
isentropicNozzle(double mdot, double T0, double A, double pa, double gamma,
                 double R);

} // namespace dns
