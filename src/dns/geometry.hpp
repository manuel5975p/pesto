// Geometry sources: any Sdf (analytic, dimpled, or MeshSdf) -> CutGeometry.
// Exact closed-form geometry is used where one exists (cutcell.hpp's
// CutGeometry::make overloads); otherwise the cut is built here by
// band-classified supersampling with a Lipschitz block prefilter.
#pragma once

#include "dns/bodies.hpp"
#include "dns/cutcell.hpp"
#include "dns/grid.hpp"
#include "dns/remap.hpp"
#include "dns/sdf.hpp"

#include <expected>
#include <string>

namespace dns {

// Sampling budget. The validated bodies use DIFFERENT budgets and they are
// carried through explicitly, NOT unified -- a changed budget shifts the cut:
//   mesh / golf sphere   nCell 4, nFace 4          (3D lattices)
//   golf disk            nCell 8, nFace 8, zFlat   (one x-y layer, replicated)
//   lens                 exact chord apertures + 64^2 lambda (exactLensCut)
struct SampleOpts
{
    int nCell = 4;
    int nFace = 4;
    // Sample and store one x-y layer and replicate it over k. Exact for a
    // z-invariant body and the only correct form when the sdf is z-invariant
    // but the grid is not.
    bool zInvariant = false;
    // Band half-width beyond which a cell is trivially fluid/solid, in world
    // units, relative to sdf(centre). 0 = half the cell diagonal.
    double band = 0.0;
    // Host workers, <= 0 = hardware concurrency. Never changes a value.
    int threads = 0;
};

// lambda / ax / ay / az by supersampling `sdf` (> 0 is fluid). Cells whose
// block-centre Lipschitz bound proves them wholly fluid or wholly solid skip
// the inner samples entirely -- the dominant cost on large grids.
// nz == 1 or zInvariant: the z pass is skipped and az mirrors lambda.
// Threaded over block rows, one sdf copy per worker (sdf must tolerate
// concurrent copies): bit-identical for any o.threads (host_threads).
// Pre: grid dims > 0, sdf approximately a true distance near the wall.
// Post: every value in [0,1].
std::expected<CutGeometry, std::string>
sampleCut(const Grid&, const Sdf&, SampleOpts = {});

// The cut of the body the SOLVER sees on a --focal mesh, in index space: cell
// (i,j,k) samples the WORLD sdf at the MAPPED positions of its reference
// sub-samples -- the frame the ghost wall classifies in. For display and
// diagnostics of a clustered mesh (dns-ui); qRef keeps the world-lattice
// sampleCut (see ReferenceQ). The pulled-back field is divided by the map's
// largest slope so it stays 1-Lipschitz in reference coordinates, which is
// what sampleCut's prefilter and band assume (fractions are sign counts, so
// the scale changes no value).
// Pre: as sampleCut. Post: all-inactive remap = sampleCut, bit for bit.
std::expected<CutGeometry, std::string>
sampleCutMapped(const Grid&, const Sdf&, const AxisRemap3&, SampleOpts = {});

// Exact chord apertures + supersampled lambda for the lens: its tips are
// sub-cell and pure sampling loses them.
// Pre: b.w > b.hh, lambdaSamples > 0.
std::expected<CutGeometry, std::string>
exactLensCut(const Grid&, const Lens& b, int lambdaSamples = 64);

// --- reference dynamic pressure ---------------------------------------------
//
// qRef = 0.5 rho uInf^2 A, the divisor of every reported cd / cl / cs. A is the
// body's FRONTAL AREA, measured from the cut apertures: a (j,k) stream column
// is blocked by 1 - min_i ax, exact for a convex body and honest for the rest.
//
// --focal: this measurement does not see the solver mesh at all, and that is
// why it is right. sampleCut() evaluates the WORLD sdf on the uniform lattice
// (i + 0.5) h -- a uniform grid of spacing h in world units -- so the cut is
// an independent sampling of the body, at its true size D, and the reference
// spacings h are that lattice's own weights. The SOLVED body (the ghost wall,
// classified against MAPPED centres) is the same world body on the clustered
// mesh; the two agree on A because both are the sdf's body, not because any
// factor cancels. Weighting the reference cut's rows by the focal map's
// PHYSICAL extents mixes the two lattices and makes A low by the plateau
// slope c*r -- measured 2026-07-25, cylinder n = 256:
//
//   R    A with reference spacings   A with physical spacings   true D
//   1    0.785228                    0.785228                   0.785398
//   2    0.785228                    0.523485                   0.785398
//   4    0.785228                    0.314091                   0.785398
//
// So: DO NOT "fix" this by mapping the spacings. The resolution of A is the
// reference h (coarser than the plateau), which is ample for a frontal area.
struct ReferenceQ
{
    double q = 0.0;        // 0.5 rho uInf^2 A (with the physics normalisation)
    double area = 0.0;     // the measured frontal area A itself
    bool measured = false; // false = no solid cells found; q fell back to D
};

// The two physics keep DISTINCT normalisations and index ranges, and merging
// them would silently rescale compressible cd:
//   incompressible: grids carry a ghost ring, so the sweep is the interior
//                   only (k,j from 1, i from 2) and A is the raw area.
//   compressible:   no ghost ring, the caller passes an already-sanitized cut,
//                   the sweep is the full extent, and A is divided by the z
//                   span so cd is per unit depth. No solid cells falls back
//                   to D.
ReferenceQ referenceQIncomp(const Grid& g, const CutGeometry& cut,
                            double rhoInf, double uInf);
ReferenceQ referenceQComp(const Grid& g, const CutGeometry& cut, double rhoInf,
                          double uInf, double D);

// --- --body floor: the solid rows at the inlet/outlet planes ----------------
//
// The raised floor crosses the x boundary planes, so the boundary treatment
// has to know how much of the inlet column is wall. Measured from the BODY cut
// -- before closeBoxBoundary zeroes those very apertures.
//
// The sliver rule is load-bearing and MEASURED: a cut row less than half open
// at the plane is promoted to wall (no Dirichlet, no prescribed flux, no
// characteristic inflow) rather than jetting uInf into a mostly-solid cell --
// that ratcheted at 2048^2. Both the incompressible tunnel and the
// compressible boundary used to implement this rule separately, and the
// justification was written down in only one of them.
struct FloorRows
{
    int rows = -1;      // fully-solid interior rows at the plane
    double frac = 1.0;  // open fraction of the first non-solid row
};

FloorRows measureFloorRows(const Grid& g, const CutGeometry& cut);

} // namespace dns
