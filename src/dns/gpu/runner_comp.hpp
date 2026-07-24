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
    SgsModel sgs = SgsModel::Smagorinsky;   // nut operator, used iff les > 0
    // Aperture geometry for --cut-flux / --penalize; nullptr when both are
    // off. Uniform mesh + body only (validator-enforced); must outlive the run.
    const CutGeometry* cutGeom = nullptr;
    bool cutFlux = false;        // sharp apertured fluxes (attachCutFlux)
    double penalize = 0;         // Brinkman band decay time eta; 0 = off

    // Wall-normal tanh clustering along y (dns/remap.hpp TanhStretchY):
    // 0 = uniform mesh; > 0 switches the solver to the curvilinear
    // finite-volume path on CurvMesh::stretchedY. The validator guarantees
    // les == 0, no immersed body and far == Sponge with this (NSCBC's
    // transverse terms and the ghost wall still assume uniform spacing).
    double stretchBeta = 0;
    // Focal clustering (dns/remap.hpp AxisRemap3): a fine plateau box with a
    // coarse far field, one independent map per axis. Mutually exclusive with
    // stretchBeta; same guarantees from the validator (les == 0, no immersed
    // body, and the map is the exact identity within the boundary margin).
    // focalActive == false leaves the mesh uniform bit-for-bit.
    bool focalActive = false;
    double focalLo[3] = {0, 0, 0}, focalHi[3] = {0, 0, 0}, focalW[3] = {0, 0, 0};
    double focalRatio = 1;
    // Closed lateral y walls become adiabatic no-slip mirrors (boundary
    // layers grow on them) instead of the default free-slip.
    bool noslipY = false;

    // Far field. Sponge is never a default (it is an explicit escape hatch fallback);
    // the physical characteristic boundary is the default here too.
    FarField far = FarField::Nscbc;
    bool open = false;           // lateral faces open too
    // Full-width solid floor (--body floor) crossing the NSCBC x planes:
    // interior rows j <= floorRows are fully solid there (the ghost wall
    // owns them; NSCBC must not overwrite their rhs). -1 = no floor.
    int floorRows = -1;
    int spongeCells = 0;         // 0 = clamp(0.1 nx, 8, nx/2 - 1)
    double nscbcBeta = 0.5;      // Yoo-Im transverse correction, in [0,1)
    int nscbcStrip = 0;          // absorbing strip inside the NSCBC rows
    // Potential-corrected NSCBC targets: wake source from the EMA-smoothed
    // measured drag + displacement doublet about the body centre.
    bool nscbcPot = false;
    double cx = 0, cy = 0, cz = 0; // body centre (world coords)

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
    bool vortAsinh = false;      // png: asinh(v/toe), toe = curlMax/32
    bool speedDelta = false;     // frames carry |u| - uInf (diverging)
    int statsEvery = 200;        // steps between force rows
};

// `ic` is the initial conservative state; `sdf` is the body, used to build the
// sharp ghost wall. The caller applies its own trip/taper to `ic`.
std::expected<RunStats, RunError>
runCompressible(const Grid&, const CompState& ic, const Sdf& sdf,
                const std::array<double, 6>* bbox, const CompRun&, RunOutput&);

} // namespace dns::gpu
