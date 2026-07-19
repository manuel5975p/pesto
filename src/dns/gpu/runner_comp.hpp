// The one compressible SSP-RK3 time loop. Every `dns flow` run with
// --physics compressible goes through here, whatever the body and whichever
// far field (sponge / NSCBC, walled or open).
//
// GPU-only header.
#pragma once

#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/gpu/runner_incomp.hpp"   // RunStats, RunError
#include "dns/output.hpp"
#include "dns/sdf.hpp"

#include <expected>

namespace dns::gpu {

struct CompRun
{
    double gamma = 1.4;
    double mu = 0;
    double uInf = 1.0, rhoInf = 1.0, pInf = 1.0;
    double mach = 0.2;
    double D = 1.0;
    double qRef = 1.0;
    double cfl = 0.4;
    double tEnd = 0;
    double les = 0;              // shared 3D-valid core LES; 0 = molecular DNS

    // Far field.
    FarField far = FarField::Sponge;
    bool open = false;           // lateral faces open too
    int spongeCells = 0;         // 0 = clamp(0.1 nx, 8, nx/2 - 1)
    double nscbcBeta = 0.5;      // Yoo-Im transverse correction, in [0,1)
    int nscbcStrip = 0;          // absorbing strip inside the NSCBC rows

    // Quiet start. ramp > 0: the IC is at REST (exactly consistent with the
    // no-slip body) and the freestream is raised by uniform Galilean boosts
    // over [0, ramp]; measured startup |p'|/pInf 12.7% -> 0.56%.
    double ramp = 2.0;

    // stableDt costs two full-grid reductions plus host syncs. The acoustic
    // CFL moves slowly, so the compressible cylinder path recomputes it every
    // 16 steps with a 0.95 margin. That changes trajectories but not
    // statistics, deliberately -- do not "fix" it back to every step.
    int dtRecompute = 16;
    double dtMargin = 0.95;

    bool signedVorticity = true; // 2D frames use signed wz + a diverging map
    double curlMax = 0;          // 0 = 8/D signed, 15/D magnitude
    int statsEvery = 200;        // steps between force rows
};

// `ic` is the initial conservative state; `sdf` is the body, used to build the
// sharp ghost wall. The caller applies its own trip/taper to `ic`.
std::expected<RunStats, RunError>
runCompressible(const Grid&, const CompState& ic, const Sdf& sdf,
                const std::array<double, 6>* bbox, const CompRun&, RunOutput&);

} // namespace dns::gpu
