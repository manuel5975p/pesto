// PNG export of 2D field frames (RGB8, zlib-deflated) so a running 2D
// simulation can be watched live from a png_<run> directory. CPU-only.
#pragma once

#include <string>

namespace dns {

// Diverging blue-white-red map of field/scale clipped to [-1, 1]; cells with
// mask < 0.5 (solid) render dark grey, mask may be null. Row j = 0 is the
// bottom of the image. w, h > 0, scale > 0; false on I/O failure.
bool writeDivergingPng(const std::string& path, int w, int h,
                       const float* field, float scale, const float* mask);

// Sequential viridis map of (field - lo) / (hi - lo) clipped to [0, 1];
// same mask/orientation conventions. hi > lo; false on I/O failure.
bool writeSequentialPng(const std::string& path, int w, int h,
                        const float* field, float lo, float hi,
                        const float* mask);

} // namespace dns
