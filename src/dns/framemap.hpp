// Rendering a curvilinear run on a PHYSICAL-uniform raster.
//
// The solver's arrays live in REFERENCE (index) space: cell (i,j,k) is one
// entry whatever physical volume it occupies. A pixel-per-cell image is
// therefore a picture of the INDEX space -- on a clustered mesh (--focal,
// --stretch) the fine plateau is blown up and the coarse far field squashed,
// so the body comes out the wrong shape and the wake reads at the wrong
// scale. PhysicalRaster undoes the mesh map for display only: pixel p samples
// the field at the reference point that maps to p's uniform PHYSICAL position.
//
// The raster pitch is the FINEST spacing on each axis, so clustering never
// costs displayed detail -- the whole point of putting cells there. The render
// grid is therefore LARGER than the solver grid (by up to --focal-ratio per
// clustered axis, capped); RunOutput is constructed with it, and every field
// and the body mask are resampled into it. The identity map is inactive: the
// render grid IS the solver grid and apply() is a plain copy, so uniform-mesh
// runs are bit-identical.
//
// Display only. Nothing here touches the state, the fluxes or the statistics.
#pragma once

#include "dns/grid.hpp"

#include <cstddef>
#include "dns/remap.hpp"

#include <functional>
#include <vector>

namespace dns {

// One axis of a separable mesh map, as the renderer needs it. `inverse` is the
// physical -> reference map (host-only; Focal1D::inverse, TanhStretchY::
// invMapY). A null inverse, or hFine >= the uniform spacing, is the identity.
struct AxisRemapView
{
    double L = 1.0;                        // physical period of the axis
    double hFine = 0;                      // finest physical spacing; 0 = uniform
    std::function<double(double)> inverse; // null = identity
};

class PhysicalRaster
{
public:
    // Inactive: render grid == g, apply() copies.
    static PhysicalRaster identity(const Grid& g);

    // Per-axis resampling tables. A degenerate axis (n == 1, the nz == 1
    // restriction) is always left alone. The render count per axis is
    // round(L / hFine) clamped to [n, growthCap * n]; if the product still
    // exceeds cellCap, every grown axis shrinks by one common factor (a 3D
    // mesh clustered on all three axes asks for ratio^3 times the cells).
    // Past the cap the plateau displays below its native resolution -- the
    // geometry stays true, only the detail is capped.
    static PhysicalRaster make(const Grid& g, const AxisRemapView v[3],
                               int growthCap = 8,
                               std::size_t cellCap = 128u << 20);

    bool active() const { return active_; }
    const Grid& renderGrid() const { return rg_; }

    // dst (renderGrid().size(), Grid::idx order) <- trilinear samples of src
    // (the solver grid). Inactive rasters copy.
    void apply(const float* src, float* dst) const;

    // Convenience for the one-shot fields (the body mask).
    std::vector<float> apply(const std::vector<float>& src) const;

private:
    struct Axis
    {
        std::vector<int> i0, i1;  // wrapped source indices
        std::vector<float> w;     // weight of i1
    };
    Grid src_{}, rg_{};
    Axis ax_[3];
    bool active_ = false;
};

// Build the display raster for a compressible run's active mesh maps (both the
// exact identity when off). See framemap.cpp for why this exists once.
PhysicalRaster rasterFor(const Grid& g, const AxisRemap3& focal,
                         const TanhStretchY& stretch);

} // namespace dns
