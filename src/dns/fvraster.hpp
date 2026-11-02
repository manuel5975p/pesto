// A uniform plane through a FaceMesh for the frame sinks (docs/fvcomp.md): an
// unstructured mesh has no raster, so every FV frame on any mesh is the
// containing cell's value at the pixel centre. Piecewise constant, no blending.
// Host, geometry layer. Deterministic at any thread count.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/grid.hpp"

#include <array>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns {

struct FvRasterSpec
{
    int longSide = 512;            // pixels along the window's longer side
    std::array<double, 4> box{};   // lo0 hi0 lo1 hi1 in the plane's axes; all 0 = the mesh bbox
};

class FvRaster
{
public:
    // The plane: normal to emptyAxis (2D) or to z at the bbox centre (3D).
    // Errors: an empty or inverted window.
    static std::expected<FvRaster, std::string> make(const FaceMesh& fm, const FvRasterSpec& sp, int threads = 0);

    // The render grid (nz == 1) in pixels.
    const Grid& grid() const { return g_; }
    // 1 inside a cell, 0 outside the mesh (a body, a hole).
    const std::vector<float>& mask() const { return mask_; }
    // out[pixel] = cellValue[containing cell], fill outside. Pre: sizes match.
    void sample(std::span<const double> cellValue, std::span<float> out, float fill = 0.0f) const;

private:
    Grid g_{};
    std::vector<int> cell_;
    std::vector<float> mask_;
};

} // namespace dns
