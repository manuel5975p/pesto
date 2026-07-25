// Second-order MAC finite-difference operators: divergence, gradient,
// Laplacian, convection. All fields periodic; no allocation inside kernels.
#pragma once

#include "dns/field.hpp"
#include "dns/grid.hpp"

namespace dns {

// div(U) at cell centres. out.size() == g.size()
void divergence(const Grid& g, const VectorField& U, Scalar& out);

// U -= scale * grad(p), gradient evaluated on the corresponding faces.
void subtractGradient(const Grid& g, const Scalar& p, double scale, VectorField& U);

// out = lap(q) component-wise on the native face grid.
void laplacian(const Grid& g, const VectorField& U, VectorField& out);

// out = -(U . grad)U in conservative (divergence) form on the face grids.
void convection(const Grid& g, const VectorField& U, VectorField& out);

// Volume-integrated kinetic energy 0.5*int(|U|^2) via face-squared averages.
double kineticEnergy(const Grid& g, const VectorField& U);

// max over cells of |u|/hx + |v|/hy + |w|/hz (sum-CFL for 3D transport).
double maxCourantPerDt(const Grid& g, const VectorField& U);

// Volume-averaged velocity per component (conserved momentum / volume).
struct Vec3 { double x, y, z; };
Vec3 meanVelocity(const Grid& g, const VectorField& U);

// U -= m componentwise; used to pin the mean mode under linear forcing.
void subtractMean(const Grid& g, Vec3 m, VectorField& U);

// Specific fluctuation energy k' = 0.5 <|U - <U>|^2> (per unit mass).
double fluctuationEnergy(const Grid& g, const VectorField& U);

// <sum_ij (du_i/dx_j)^2> from compact staggered differences; eps = nu * this.
double velocityGradientSq(const Grid& g, const VectorField& U);

// max |div(U)| over cells (diagnostic; needs no scratch).
double maxAbsDivergence(const Grid& g, const VectorField& U);

// MAC-aware trilinear prolongation of a staggered field from a coarse grid
// (gc, spacing hc) onto a fine one (gf, hf) -- the grid-sequencing (--seq)
// transfer. Each component samples its OWN face map along the face-normal axis
// and the centre map along the other two; a collocated interpolation here
// would smear the staggering and seed the fine level with a divergent field.
// Both grids carry the one-cell ghost ring. nz == 1 on either grid keeps w
// exactly 0 and the z sampling collapsed.
VectorField prolongMac(const Grid& gc, double hc, const VectorField& C,
                       const Grid& gf, double hf);

} // namespace dns
