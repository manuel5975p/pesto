// Analytic solid shapes in world units and their signed-distance functions.
//
// This header OWNS the shape structs. It used to include cutcell.hpp to get
// them, which is the wrong direction: every consumer that wanted an Sdf pulled
// in CutPoissonSolver and its CG workspace with it. cutcell.hpp now includes
// this one -- it needs the shapes for its exact closed-form CutGeometry
// constructors, not the other way round.
#pragma once

#include "dns/sdf.hpp"

#include <array>
#include <span>
#include <vector>

namespace dns {

// Axis-aligned solid slab in z crossed with an L1 ball (diamond) in x-y, i.e.
// a square prism rotated 45 deg about z -- a "rotated cube". cx,cy,R,z0,z1 in
// world units. Interior: |x-cx|+|y-cy| <= R and z0 <= z <= z1.
struct RotatedCube
{
    double cx, cy, r, z0, z1;
};

// Solid ball of radius r centred at (cx,cy,cz), world units. Must not touch
// the periodic images (geometry is built unwrapped).
struct Sphere
{
    double cx, cy, cz, r;
};

// Solid circular cylinder along z (infinite: spans the whole grid), axis at
// (cx,cy), radius r, world units. Interior: (x-cx)^2 + (y-cy)^2 <= r^2.
struct CylinderZ
{
    double cx, cy, r;
};

// Two overlapping disks of half-width w and half-height hh (w > hh): a lens
// with sharp leading/trailing tips at x = cx +- w. Interior: both disks.
// Convex, z-invariant.
struct Lens
{
    double cx, cy, w, hh;
    double e() const { return (w * w - hh * hh) / (2 * hh); } // disk-centre offset
    double ra() const { return e() + hh; }                    // disk radius
};

// Wall-mounted blocker: axis-aligned rectangle of half-thickness w about
// x = cx, solid from below the domain floor (semi-infinite downward, so it is
// sealed through the wall/ghost rows -- no under-gap jet) up to y = yTop.
// z-invariant. The 2D shear-layer rig: a laminar tunnel inflow separates at
// the sharp top edge.
struct Fence
{
    double cx, w, yTop;
};

// Raised solid floor: a slab spanning the ENTIRE streamwise (and spanwise)
// extent, solid for y <= y_s(x), sealed semi-infinitely downward. The cut
// surface is a no-slip plate whose leading edge sits in the inflow plane --
// the wall-boundary-layer rig. Optional roughness: n triangular SPIKES of
// height amp and half-base halfw, one per pitch, centred at
// x0 + (m + 1/2) pitch (m = 0..n-1):
//   y_s(x) = yTop + max(0, amp (1 - |x - centre| / halfw)).
// Sharp tips are fine for the sampled cut (lens-tip precedent); the plate is
// exactly FLAT for at least pitch/2 - halfw next to the inlet/outlet planes,
// so the boundary cut rows never seal into a pocket (a ripple reaching a
// plane ratcheted the prescribed inflow into a sliver pocket -- measured
// dt collapse). amp = 0 is the exact flat plate. z-invariant.
struct Floor
{
    double yTop;
    double amp = 0, halfw = 1, pitch = 1, x0 = 0;
    int n = 0;
};

Sdf sdfOf(const Sphere&);
Sdf sdfOf(const CylinderZ&);
Sdf sdfOf(const RotatedCube&);
Sdf sdfOf(const Lens&);
Sdf sdfOf(const Fence&);
Sdf sdfOf(const Floor&);

// base minus spherical-cap pits: sdf = max(base, -pits). Pit footprint radius
// rd and depth are in WORLD units; the caller sizes them in grid units so they
// stay resolvable (5h / 2.5h in the validated golf runs -- a real ball's 0.01D
// dimples vanish on any affordable grid).
// zInvariant drops the z term of the pit distance, so a z-aligned cylinder's
// dimples become infinite grooves (the 2D golf disk) rather than isolated pits.
// Pre: rd > 0, centres non-empty.
Sdf dimpled(Sdf base, std::span<const std::array<double, 3>> centres, double rd,
            bool zInvariant = false);

// Pit sphere radius for a footprint of half-width a and recess depth.
// Pre: depth > 0.
double pitRadius(double a, double depth);

// Deterministic dimple lattices, centres of the subtracted pit spheres.
// 3D: golden-angle Fibonacci on the sphere. 2D: irrational phase around the
// circle (no dimple lands axis-aligned), z-replicated by the caller.
// Pre: n > 0, depth > 0.
std::vector<std::array<double, 3>> fibonacciCentres(const Sphere&, int n, double rd,
                                                    double depth);
std::vector<std::array<double, 3>> ringCentres(const CylinderZ&, int n, double rd,
                                               double depth);

} // namespace dns
