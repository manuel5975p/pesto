// Physical (SI) framing for a case that the solver runs dimensionless.
//
// The solver's box units are fixed: uInf = 1, rho_inf = 1, and the body
// diameter D comes from --dr (box length measured in D). Re and Mach are its
// only physical inputs. That is efficient and it is what FP32 wants, but it is
// not how anyone thinks about a wind tunnel -- so the editor takes the
// tangible quantities (a material, a body size in metres, a freestream in m/s)
// and DERIVES Re and Mach from them:
//
//     Re   = uInf * dBody / nu
//     Mach = uInf / c
//
// This does not make the UI own any physics (doc.hpp's rule). Re and Mach are
// handed to the same RawConfig -> dns::validate() path as before; validate()
// still decides what is legal, and the exported command line still reads
// --re/--mach, because that is what `dns` accepts. The SI numbers are the
// INPUT face, not a second source of truth.
//
// Length conversion has a pleasant property worth knowing: scene object
// offsets and diameters are already expressed in units of D (scene.hpp), so
// their metre values are just `value * dBody` -- no grid or box factor enters.
// Grid spacings and the domain DO need the box factor, metresPerBox().
#pragma once

#include <cmath>
#include <cstddef>
#include <span>

namespace ui {

// A working fluid. nu and c are what the solver needs; rho is carried because
// it is what turns a dimensionless coefficient back into newtons, and it costs
// one field to keep the table honest.
struct Material
{
    const char* name;
    double nu;   // kinematic viscosity [m^2/s]
    double c;    // speed of sound [m/s]
    double rho;  // density [kg/m^3]
};

// Sea-level / 20 C values, spanning four decades of nu so the Re read-out
// actually moves when you switch. The last entry is "custom": its numbers are
// ignored and the document's own nu/c/rho are used instead.
inline std::span<const Material> materials()
{
    static constexpr Material kTable[] = {
        {"air", 1.516e-5, 343.2, 1.204},
        {"water", 1.004e-6, 1481.0, 998.2},
        {"seawater", 1.050e-6, 1521.0, 1025.0},
        {"glycerin", 1.120e-3, 1920.0, 1261.0},
        {"custom", 1.516e-5, 343.2, 1.204},
    };
    return {kTable, sizeof kTable / sizeof kTable[0]};
}

inline int customMaterialIndex() { return int(materials().size()) - 1; }

// The SI face of a case. Held by DocState; every metre/second shown in the UI
// goes through this.
struct Physical
{
    double nu = 1.516e-5;   // [m^2/s]
    double c = 343.2;       // [m/s]
    double rho = 1.204;     // [kg/m^3]

    // The default case: a 1 mm wire in a 3 m/s airflow. Chosen so it is a
    // thing that exists (hot-wire anemometry) AND lands at Re ~ 198, which is
    // where this tool has always opened -- a cylinder just past the onset of
    // vortex shedding, resolvable on the default grid. Defaulting to a 0.1 m
    // body at 30 m/s would be equally tangible and put the editor at Re 2e5,
    // i.e. "under-resolved" before the user has touched anything.
    double dBody = 0.001;   // reference body diameter [m]
    double uInf = 3.0;      // freestream [m/s]

    // The two numbers the solver actually takes. Guarded against a zero
    // denominator: the fields are validated on entry (dragDouble clamps them
    // above zero), but this is called every frame from the UI thread and a
    // NaN would propagate into the panel text before validate() ever saw it.
    double re() const { return nu > 0 ? uInf * dBody / nu : 0.0; }
    double mach() const { return c > 0 ? uInf / c : 0.0; }

    // Metres per box length unit, given the body diameter the RESOLVED case
    // realised (Case::placement.D or CaseGeometry::D -- they can differ from
    // the requested one by the grid snap, and the honest factor is the
    // realised one). Returns 0 when dBox is not usable, and callers treat a
    // zero factor as "no SI view yet" rather than dividing by it.
    double metresPerBox(double dBox) const
    {
        return dBox > 0 ? dBody / dBox : 0.0;
    }

    // --- convection (dns conv) ---------------------------------------------
    // The same idea one step further: a convection cell has no body and no
    // freestream, so its tangible face is a LAYER HEIGHT, an imposed
    // temperature drop and a mean temperature, and the Rayleigh number is
    // derived from them exactly as Re is derived above.
    //
    //   Ra = g alpha dT H^3 / (nu kappa),  alpha = 1/T for an ideal gas,
    //                                      kappa = nu / Pr
    //      = g dT H^3 Pr / (T nu^2)
    //
    // Pre: all of dT, H, T, nu, Pr > 0; returns 0 otherwise rather than an
    // inf, because this is called every frame from the UI thread and a panel
    // must never print a NaN the validator has not seen yet.
    static constexpr double gravity = 9.80665;   // [m/s^2]

    double rayleigh(double dT, double h, double tMean, double pr) const
    {
        if (!(dT > 0 && h > 0 && tMean > 0 && pr > 0 && nu > 0)) return 0.0;
        return gravity * dT * h * h * h * pr / (tMean * nu * nu);
    }

    // The free-fall velocity sqrt(g alpha dT H) and the time H/U it defines --
    // the unit `--tt` counts in, so the panel can say how long a run is in
    // seconds.
    double freeFall(double dT, double h, double tMean) const
    {
        if (!(dT > 0 && h > 0 && tMean > 0)) return 0.0;
        return std::sqrt(gravity * (dT / tMean) * h);
    }
    double freeFallSeconds(double dT, double h, double tMean) const
    {
        const double u = freeFall(dT, h, tMean);
        return u > 0 ? h / u : 0.0;
    }
    // The Mach number the layer ACTUALLY convects at. It is ~1e-3 for air,
    // which is why the compressible branch cannot be run at it and treats its
    // own --mach as a cost knob instead.
    double convMachPhysical(double dT, double h, double tMean) const
    {
        const double u = freeFall(dT, h, tMean);
        return (u > 0 && c > 0) ? u / c : 0.0;
    }

    // One flow-through in seconds: the box reports tEnd = ft * L / uInf, and
    // L is a length, so the SI value is the same expression in SI.
    double flowThroughSeconds(double lBox, double dBox) const
    {
        const double m = metresPerBox(dBox);
        return (m > 0 && uInf > 0) ? lBox * m / uInf : 0.0;
    }
};

} // namespace ui
