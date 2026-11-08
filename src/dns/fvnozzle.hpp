// The nozzle's library half (docs/nozzle.md): the rest start, the ideal mass flow, the stats
// row of a run on a Total inlet. Host, no GPU, no physics in the driver.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvref.hpp"

#include <array>
#include <span>
#include <vector>

namespace dns {

// Uniform rest at p, T: the IC of a nozzle run, raised by a ramp on the Total row's p0.
// Pre: p, T > 0.
FvState fvRestState(const FaceMesh& fm, double p, double T, double gamma, double Rgas);

// Ideal choked mass flow through astar of a gas at stagnation p0, T0 (what Cd divides by).
// Pre: all > 0.
double fvIdealMassFlow(double p0, double T0, double astar, double gamma, double Rgas);

// The stats row's definitions: astar, axis (unit), p0 and t0 stagnation in box units.
struct FvNozzleRun
{
    double astar = 0;
    std::array<double, 3> axis{1, 0, 0};
    double pRef = 0, p0 = 0, t0 = 0;
};

// mdotIn, mdotOut > 0 through the Total and Outflow patches; Fwall the gauge wall force
// along the axis; Fin, Fexit the outward patch fluxes of momentum along it; Cd = mdotOut /
// ideal, Cf = Fexit / (p0 A*).
struct FvNozzleRow
{
    double mdotIn = 0, mdotOut = 0, Fwall = 0, Fin = 0, Fexit = 0, Cd = 0, Cf = 0;
};

// bc: a row per patch. mass, mom[3]: DeviceFvComp::patchFlux of rows 0, 1..3, per patch.
// fw: the tallied force. Pre: spans sized alike; run.astar > 0.
FvNozzleRow fvNozzleRow(std::span<const FvBc> bc, std::span<const double> mass, const std::array<std::vector<double>, 3>& mom,
                        const std::array<double, 3>& fw, const FvNozzleRun& run, double gamma, double Rgas);

} // namespace dns
