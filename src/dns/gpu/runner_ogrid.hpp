// The compressible time loop on a body-fitted O-grid (dns/ogrid.hpp): the body
// IS the inner boundary, so there is no immersed-body machinery here at all --
// no ghost wall, no cut cells, no --penalize, no --far box. The mesh is the
// geometry, the wall is a mesh face row, and the far field is the annulus'
// outer circle (gpu/ogrid_bc.hpp).
//
// This is a FORK of runner_comp, not a branch of it: gpu/internal/run_loop.hpp
// says the loop bodies are deliberately not merged, because their differences
// are physics. What is shared is only the ordering-free scaffolding
// (internal::recomputeDt, internal::finishTiming, the collapse threshold).
//
// GPU-only header.
#pragma once

#include "dns/comp.hpp"
#include "dns/ogrid.hpp"
#include "dns/output.hpp"
#include "dns/rotor.hpp"
#include "dns/sgs.hpp"
#include "dns/transport.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "dns/gpu/run_result.hpp" // RunStats, RunError

#include <expected>
#include <string>
#include <vector>

namespace dns::gpu {

// Everything the loop needs that is not the grid, the mesh or the sink.
struct OGridRun
{
    // The gas: gamma, R, Pr and the viscosity law (dns/transport.hpp). Passed
    // whole rather than as (gamma, mu) because the isothermal wall needs R and
    // the wall heat flux needs kappa(T); a Sutherland law therefore reaches the
    // boundary kernels for free.
    Transport tr{};

    // Freestream, along +x. pInf sets the Mach number; |uInf| must be subsonic.
    double rhoInf = 1.0, uInf = 1.0, pInf = 1.0;

    // Inner wall thermal condition.
    OGridThermal thermal = OGridThermal::Adiabatic;
    double Twall = 1.0;          // isothermal only

    // --rotor-mach with --rotor-frame mrf: the body spins about the z shaft
    // through the O-grid centre, solved in the rotating frame on the static
    // body-fitted mesh (the frozen-rotor approximation; exact when the
    // section is a circle). The wall row mirrors about u = omega x r, the
    // rhs carries the node-psi MRF zone (DeviceComp::setMrfCurv), and the
    // stats gain thrust/torque/ct/cq exactly as runner_comp's rotor does.
    // The MOVING-geometry rotor has no O-grid counterpart: spinning the body
    // would spin the mesh, so the runner rejects rotor.active() && !rotorMrf.
    // Inactive rotor = bit-for-bit the static runner.
    Rotor rotor{};
    bool rotorMrf = false;

    // Far-field row: Rudy-Strikwerda pressure anchor and the outflow/inflow
    // blend dead band (gpu/ogrid_bc.hpp). There is deliberately no sponge
    // option -- a sponge is never a default anywhere, and on an annulus whose
    // outer boundary is a characteristic circle there is nothing for it to do.
    double farSigma = 1.0;
    double farBand = 0.01;

    // Reference scales for the reported coefficients: qRef = 0.5 rhoInf uInf^2
    // D (per unit span; the loop divides by the span itself), D the body
    // diameter, which also sets the frame colour scales.
    double D = 1.0;
    double qRef = 1.0;

    // Shared 3D-valid core LES; 0 = molecular DNS, bit-for-bit. The SGS
    // operators read the mesh metrics, so they work on the O-grid exactly as on
    // any other curvilinear mesh; the wall tally stays molecular.
    double les = 0;
    SgsModel sgs = SgsModel::Smagorinsky;

    double cfl = 0.4;
    double tEnd = 0;

    // Quiet start. ramp > 0: the IC is at REST (exactly consistent with the
    // no-slip wall) and the freestream is raised by uniform Galilean boosts
    // over [0, ramp], the far-field target following each boost.
    double ramp = 2.0;

    // stableDt costs a full-grid reduction plus a host sync; the acoustic CFL
    // moves slowly. Same meaning and same defaults as runner_comp's, and the
    // same deliberate trajectory change.
    int dtRecompute = 16;
    double dtMargin = 0.95;

    int statsEvery = 200;        // steps between force rows

    // Frame colour scales.
    bool signedVorticity = true; // 2D frames use signed wz + a diverging map
    double curlMax = 0;          // 0 = 8/D signed, 15/D magnitude
    bool vortAsinh = false;      // png: asinh(v/toe), toe = curlMax/32
    bool speedDelta = false;     // frames carry |u| - uInf (diverging)
    // --temp: a third frame channel carrying T - Tinf on the diverging map.
    // Off by default (it changes what every artefact of the run holds); see
    // the same field on CompRun.
    bool tempFrames = false;

    // Body centre: the origin of the surface CSV's angle (and, on a revolved
    // mesh, the axis of revolution through it along x).
    double cx = 0, cy = 0, cz = 0;

    // RANS (docs/rans.md, "dns flow"). None = the DNS/LES runner bit for
    // bit. wallDist: d at every cell (ogridWallDistance; ghosts their source
    // row's), also given without RANS for a steady run or --emit surface
    // (then it only feeds d1 and y+). phiInf: the freestream SPECIFIC
    // scalars (SA nu~ | SST k, omega). icTurb: the initial CONSERVED
    // scalars, g.size() each, for the model's nT.
    RansModel rans = RansModel::None;
    std::vector<double> wallDist;
    double phiInf[2] = {0, 0};
    std::vector<double> icTurb[2];

    // Compressible --steady (> 0): the explicit local-time-stepping pseudo-
    // time march on the compensated state (gpu/internal/steady_march.hpp),
    // stopped on cd, cdf, the wake deficit and the turbulence integrals;
    // steadySteps caps it (exit 8). tEnd and ramp are not read then.
    double steadyTol = 0;
    long steadySteps = 0;
    // The Newton-Krylov march (gpu/nk) instead of the explicit LTS one.
    // Pre: nz == 1 or the revolved wedge.
    bool steadyImplicit = false;
    // --compensated: the time march on the compensated state too (the
    // explicit steady march always has it). Pre: !steadyImplicit.
    bool compensated = false;

    // The run label (dns/rans_label): the progress line's tail and the
    // surface CSV's header comment. Empty on a DNS/LES time march.
    std::string label;
};

// `ic` is the initial conservative state on the O-grid's logical grid (a rest
// state for a quiet start); both ghost rows are overwritten by the boundary
// hook before the first step. `og` must be the mesh `g` was sized for
// (dns::ogridGrid), and its CurvMetrics are built here.
//
// WHAT A FRAME IS. The mesh is body-fitted, so the frame arrays are the
// INDEX-SPACE raster of the annulus: column i is the angular cell (clockwise,
// theta = -2 pi (i + 1/2) / nx) and row j the radial cell, wall at j = 1 and
// far boundary at j = ny-2. It is an unrolled boundary-layer view, correct
// per cell and honest about it -- it is NOT a picture of physical space, and
// dns/framemap.hpp's PhysicalRaster cannot fix that: that resampler is per-axis
// separable and an O-grid's map is not. A physical picture needs a scattered
// resample onto a Cartesian raster, which nothing in the tree does yet. The
// FIELDS are physical: the vorticity is taken through the mesh metrics, so the
// values are true curl components, only their placement is index space. The two
// ghost rows are written as zero rather than left holding a wrapped stencil.
std::expected<RunStats, RunError>
runOGrid(const Grid& g, const OGrid& og, const CompState& ic, const OGridRun& r,
         RunOutput& out);

} // namespace dns::gpu
