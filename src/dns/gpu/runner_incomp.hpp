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
#include <vector>

namespace dns::gpu {

// Everything the loop needs that is not the grid, the geometry or the sink.
struct IncompRun
{
    double nu = 0;            // molecular viscosity; the ONLY DNS dissipation
    double uInf = 1.0;
    double qRef = 1.0;        // 0.5 rho uInf^2 * frontal area
    double D = 1.0;           // body diameter, sets the frame colour scales
    double cfl = 0.8;
    // Steps between stable-dt recomputes. Every recompute is a max-reduce over
    // the whole field AND the one mandatory host synchronization of the step,
    // so N > 1 buys the pipeline a run of N steps it never has to drain --
    // at the price of holding dt fixed across them, which CHANGES THE
    // TRAJECTORY. 1 = recompute every step, today's behaviour and the default,
    // so nothing moves unless the caller asks for it. Same field and same
    // meaning as the compressible runner's --dt-recompute.
    int dtRecompute = 1;
    // Safety factor on a held dt, the companion of dtRecompute: a bound
    // measured N steps ago is stale and the flow may have accelerated since.
    // 1.0 = no derating, bit-for-bit today's dt, and only safe at
    // dtRecompute == 1; raising dtRecompute must lower this with it (the
    // compressible runner pairs 16 with 0.95).
    double dtMargin = 1.0;
    double tEnd = 0;
    double tol = 1e-4;        // CG per-cell RMS residual target
    int maxIters = 500;
    double les = 0;           // SGS constant; 0 = DNS
    SgsModel sgs = SgsModel::Smagorinsky;   // nut operator, used iff les > 0
    bool cutFlux = false;     // --cut-flux: apertured momentum RHS (gpu/cutflux.hpp)
    double penalize = 0;      // --penalize: Brinkman band decay time eta; 0 = off
    double ramp = 0;          // --ramp: quiet start, freestream smoothstepped
                              // over [0, ramp]; 0 = impulsive (historical)
    double vortMax = 0;       // vorticity frame clip (absolute); 0 = 8 uInf/D
    bool vortAsinh = false;   // png: asinh(v/toe), toe = vortMax/32
    bool speedDelta = false;  // frames carry |u| - uInf (diverging)

    // Far field. Fringe = fully periodic box + cosine sponge (FFT pressure);
    // otherwise the confined tunnel (Dirichlet inlet, convective outlet,
    // free-slip lateral walls, DCT-preconditioned CG on the confined box).
    // open turns the tunnel's lateral walls into sponge-free zero-gradient
    // open faces whose flux enters the projection (see DeviceTunnelBc).
    bool fringe = false;
    bool open = false;
    // Potential-corrected inlet (requires open): Dirichlet inlet = freestream
    // + wake source (EMA of measured drag) + displacement doublet about the
    // body centre.
    bool pot = false;
    double cx = 0, cy = 0, cz = 0; // body centre (world coords)
    // Full-width solid floor crossing the inlet/outlet planes (--body floor):
    // fully-solid interior row count and the cut row's open face fraction,
    // measured from the body cut BEFORE the boundary ring is closed.
    // -1 = no floor (bit-identical historical tunnel).
    int floorRows = -1;
    double floorFrac = 1.0;
    int fringeCells = 0;      // 0 = clamp(0.1 nx, 8, nx/2 - 1)
    int fringeAxes = 0;       // 0 = nz == 1 ? 2 : 3

    BodyBox box{};            // force-tally index box (the whole scene/body)

    // Per-object force integration (scenes): one extra tally per object,
    // written as the "forces" CSV series (t, then cd_/cl_/cdf_ per label) at
    // the same cadence as "stats". Boxes must be pairwise disjoint -- an
    // overlapping pair would each count the shared surface -- and the caller
    // (dns_main) enforces that, downgrading to the total-only tally instead.
    struct Tally
    {
        std::string name;
        BodyBox box;
    };
    std::vector<Tally> tallies;

    int statsEvery = 20;      // steps between force/diagnostic rows

    // Stationary mode: stop early once the field stops changing. The residual
    // is R = max|u(t+dt) - u(t)| / dt, non-dimensionalized by uInf^2/D, so the
    // threshold is dt- and grid-independent. 0 = disabled (march to tEnd).
    // tEnd then acts as the give-up cap: reaching it is a non-convergence
    // error (exit 8), never a silent "converged".
    double steadyTol = 0;
};

struct RunStats
{
    long steps = 0;
    double msPerStep = 0;
    double cd = 0, cl = 0, cs = 0, cdf = 0;   // last measured
    double resid = 0;         // last steady residual, uInf^2/D units
    double tStop = 0;         // time actually reached
    bool converged = false;   // steady mode only: resid fell below steadyTol
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

// `finalField`, when non-null, receives the converged/final velocity field --
// grid sequencing prolongs it onto the next level as that level's initial
// guess. Must be sized for this grid.
std::expected<RunStats, RunError>
runIncompressible(const Grid&, const CutGeometry& cut, const Sdf& sdf,
                  const VectorField& ic, const IncompRun&, RunOutput&,
                  VectorField* finalField = nullptr);

} // namespace dns::gpu
