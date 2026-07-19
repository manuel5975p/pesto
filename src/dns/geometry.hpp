// Geometry sources: any Sdf (analytic, dimpled, or MeshSdf) -> CutGeometry.
// Exact closed-form geometry is used where one exists (cutcell.hpp's
// CutGeometry::make overloads); otherwise the cut is built here by
// band-classified supersampling with a Lipschitz block prefilter.
#pragma once

#include "dns/bodies.hpp"
#include "dns/cutcell.hpp"
#include "dns/grid.hpp"
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
};

// lambda / ax / ay / az by supersampling `sdf` (> 0 is fluid). Cells whose
// block-centre Lipschitz bound proves them wholly fluid or wholly solid skip
// the inner samples entirely -- the dominant cost on large grids.
// nz == 1 or zInvariant: the z pass is skipped and az mirrors lambda.
// Pre: grid dims > 0, sdf approximately a true distance near the wall.
// Post: every value in [0,1].
std::expected<CutGeometry, std::string>
sampleCut(const Grid&, const Sdf&, SampleOpts = {});

// Exact chord apertures + supersampled lambda for the lens: its tips are
// sub-cell and pure sampling loses them.
// Pre: b.w > b.hh, lambdaSamples > 0.
std::expected<CutGeometry, std::string>
exactLensCut(const Grid&, const Lens& b, int lambdaSamples = 64);

} // namespace dns
