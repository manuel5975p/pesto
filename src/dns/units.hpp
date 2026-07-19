// SI <-> box-unit conversion and HIT parameter design (h = 1.5 eta sizing).
// Internally the solver runs in box units (L = 2pi, u' = 1) so FP32 sees O(1);
// SI appears only in configuration and reporting.
#pragma once

#include <cassert>
#include <cmath>
#include <numbers>

namespace dns {

// Reference scales: one box unit of length/velocity equals lRef/uRef in SI.
struct BoxUnits
{
    double lRef, uRef; // > 0

    double tRef() const { return lRef / uRef; }
    double nuToBox(double nuSI) const { return nuSI / (lRef * uRef); }
    double epsToBox(double epsSI) const { return epsSI * lRef / (uRef * uRef * uRef); }
    double epsToSI(double eps) const { return eps * uRef * uRef * uRef / lRef; }
    double lengthToSI(double l) const { return l * lRef; }
    double velocityToSI(double u) const { return u * uRef; }
    double timeToSI(double t) const { return t * tRef(); }
};

// Stationary linear-forcing HIT sized so h = 1.5 eta exactly:
// eps = u'^3/(0.19 L)  =>  L = (1.5 N)^(4/3) * 0.19^(1/3) * nu/u'.
struct HitDesign
{
    int n;            // grid points per side
    double nuSI;      // kinematic viscosity [m^2/s]
    double uPrimeSI;  // target one-component rms velocity [m/s]
    double lBoxSI;    // box side [m]
    double epsSI;     // dissipation [W/kg]
    double etaSI;     // Kolmogorov length [m]
    double tEddySI;   // large-eddy turnover K/eps [s]
    double reLambda;  // Taylor-scale Reynolds number (depends on n only)

    // Pre: n > 0, nuSI > 0, uPrimeSI > 0.
    static HitDesign make(int n, double nuSI, double uPrimeSI)
    {
        assert(n > 0 && nuSI > 0 && uPrimeSI > 0);
        HitDesign d{};
        d.n = n;
        d.nuSI = nuSI;
        d.uPrimeSI = uPrimeSI;
        d.lBoxSI = std::pow(1.5 * n, 4.0 / 3.0) * std::cbrt(0.19) * nuSI / uPrimeSI;
        d.epsSI = std::pow(uPrimeSI, 3) / (0.19 * d.lBoxSI);
        d.etaSI = std::pow(nuSI * nuSI * nuSI / d.epsSI, 0.25);
        d.tEddySI = 1.5 * uPrimeSI * uPrimeSI / d.epsSI;
        const double reL = 0.19 * d.lBoxSI * uPrimeSI / nuSI;
        d.reLambda = std::sqrt(15.0 * reL);
        return d;
    }

    // Box-unit view: L = 2pi, u' = 1.
    BoxUnits units() const { return {lBoxSI / (2.0 * std::numbers::pi), uPrimeSI}; }
    double nuBox() const { return units().nuToBox(nuSI); }
    double epsBox() const { return units().epsToBox(epsSI); }
    double tEddyBox() const { return tEddySI / units().tRef(); }
};

} // namespace dns
