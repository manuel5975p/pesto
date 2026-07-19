// Analytic solid shapes in world units and their signed-distance functions.
// The shapes themselves are declared in cutcell.hpp (which owns the exact
// closed-form CutGeometry constructors); this header adds their Sdf views, the
// lens, and the dimple lattices.
#pragma once

#include "dns/cutcell.hpp"
#include "dns/sdf.hpp"

#include <array>
#include <span>
#include <vector>

namespace dns {

// Two overlapping disks of half-width w and half-height hh (w > hh): a lens
// with sharp leading/trailing tips at x = cx +- w. Interior: both disks.
// Convex, z-invariant.
struct Lens
{
    double cx, cy, w, hh;
    double e() const { return (w * w - hh * hh) / (2 * hh); } // disk-centre offset
    double ra() const { return e() + hh; }                    // disk radius
};

Sdf sdfOf(const Sphere&);
Sdf sdfOf(const CylinderZ&);
Sdf sdfOf(const RotatedCube&);
Sdf sdfOf(const Lens&);

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
