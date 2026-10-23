// Blade-element momentum theory, the port of tools/bemt.py: axial and
// tangential induction, Prandtl tip x hub loss, Buhl's turbulent-wake branch,
// and Ning's (2014) one-variable residual in phi solved by a bracketed brentq,
// so a root is guaranteed wherever a bracket changes sign. J = 0 is the same
// equation (static momentum), not a special case.
#pragma once

#include "dns/prop/blade.hpp"
#include "dns/prop/curve.hpp"
#include "dns/prop/polar.hpp"
#include "dns/prop/spec.hpp"

#include <string>
#include <vector>

namespace dns::prop {

// The annuli, dimensional: cosine-clustered to root and tip, from the hub
// SURFACE (the root inside the hub is buried).
struct BemtBlade
{
    int B = 0;
    double D = 0, R = 0, rh = 0;
    std::vector<double> r, dr, c, beta, sigma;   // sigma' = B c / (2 pi r)
};

// Pre: blades >= 1, 0 < hubRR < 1, elements >= 4, diameter > 0.
BemtBlade bemtBlade(const Planform& pf, int blades, double hubRR, int elements,
                    double diameter);

// Per-element solution at one J (NaN where no bracket changed sign).
struct BemtSpan
{
    std::vector<double> phi, alpha, cl, cd, cn, ct, F, k, kp, W, Va, Vt;
    std::vector<double> re, mach, dTdr, dQdr, a, ap;
};

struct BemtResult
{
    BemtSpan span;
    double T = 0, Q = 0, P = 0;   // N, N m, W (propulsive)
    double re75 = 0;
    double momErr = 0;            // max |dT_blade - dT_momentum| / peak dT/dr
    int unsolved = 0;             // annuli with no bracket (dropped)
};

// All annuli at advance ratio J. Section Re (and Mach) frozen per solve and
// refreshed from the solved W: three passes for a tabulated polar, one for the
// Re-free thin polar.
BemtResult bemtSolve(double J, const BemtBlade& bl, const Operating& op,
                     const Polar& polar);

// The whole curve over c.js, plus the advisories bemt.py prints.
struct BemtCurve
{
    std::vector<CurvePoint> rows;
    double worstMomErr = 0;
    int unsolved = 0;
    double machHi = 0;   // 1.15 x the largest geometric station Mach
};
BemtCurve bemtCurve(const Case& c, const BemtBlade& bl, const Polar& polar);

// The --dump-span table at J (bemt.py's format).
std::string bemtSpanTable(double J, const BemtBlade& bl, const BemtResult& res);

} // namespace dns::prop
