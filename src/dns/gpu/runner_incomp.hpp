// The one incompressible RK3 + projection time loop. Every `dns flow` run with
// --physics incompressible goes through here, whatever the body and whichever
// far field: the five app copies of this loop differed only in the geometry
// they were handed.
//
// GPU-only header.
#pragma once

#include "dns/config.hpp"
#include "dns/cutcell.hpp"
#include "dns/gpu/bodyforce.hpp"
#include "dns/gpu/index.hpp"
#include "dns/output.hpp"
#include "dns/sdf.hpp"

#include <expected>
#include <string>

namespace dns::gpu {

// Everything the loop needs that is not the grid, the geometry or the sink.
struct IncompRun
{
    double nu = 0;            // molecular viscosity; the ONLY DNS dissipation
    double uInf = 1.0;
    double qRef = 1.0;        // 0.5 rho uInf^2 * frontal area
    double D = 1.0;           // body diameter, sets the frame colour scales
    double cfl = 0.8;
    double tEnd = 0;
    double tol = 1e-4;        // CG per-cell RMS residual target
    int maxIters = 500;
    double les = 0;           // Smagorinsky Cs; 0 = DNS. Requires nz == 1.

    // Far field. Fringe = fully periodic box + cosine sponge (FFT pressure);
    // otherwise the confined tunnel (Dirichlet inlet, convective outlet,
    // free-slip lateral walls, DCT-preconditioned CG on the confined box).
    bool fringe = false;
    int fringeCells = 0;      // 0 = clamp(0.1 nx, 8, nx/2 - 1)
    int fringeAxes = 0;       // 0 = nz == 1 ? 2 : 3

    BodyBox box{};            // force-tally index box
    int statsEvery = 20;      // steps between force/diagnostic rows
};

struct RunStats
{
    long steps = 0;
    double msPerStep = 0;
    double cd = 0, cl = 0, cs = 0, cdf = 0;   // last measured
};

// Runs to completion, writing through `out`. `cut` must already be sanitized
// (and, for the tunnel far field, closed on the box boundary); `sdf` is the
// body's signed distance, used once to build the ghost-fill sharp wall.
// Errors carry the process exit code the masterplan assigns (6 resource,
// 7 numerical, 8 solver).
struct RunError
{
    int code;
    std::string message;
};

std::expected<RunStats, RunError>
runIncompressible(const Grid&, const CutGeometry& cut, const Sdf& sdf,
                  const VectorField& ic, const IncompRun&, RunOutput&);

} // namespace dns::gpu
