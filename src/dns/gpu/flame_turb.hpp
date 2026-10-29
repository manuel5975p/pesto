// The turbulent inflow of `dns flame --turb` (docs/chem.md, "3D and
// turbulence"): the frozen synthetic-HIT box (dns/turb_ic.hpp) on the device,
// sampled once per step at the slice the inflow face sees after sweeping a
// distance X = integral of u_in dt (Taylor's hypothesis: the box is advected
// past the face at the inflow speed). The rows feed
// DeviceCompNscbc::setTurbulentInflow. GPU-only header.
#pragma once

#include "dns/gpu/device_buffer.hpp"

#include <array>
#include <cmath>
#include <expected>
#include <vector>

namespace dns::gpu {

// The box column the inflow cell (x = h/2) sees after a sweep X: fluid there
// came from x_b = h/2 - X, i.e. the fractional cell index s = (-X/h) mod nbx;
// i0 = floor(s), i1 = i0 + 1 mod nbx, f = s - i0 rounded to FP32 once.
// At X = 0: i0 = 0, f = 0 (the IC's own column, flame1d addTurbulence).
// Pre: h > 0, nbx >= 1, X finite.
struct TurbSlice
{
    int i0, i1;
    float f;
};
TurbSlice turbSliceOf(double X, double h, int nbx);

// The row value between two box columns: fmaf(f, b - a, a), contracted
// explicitly so the device kernel and this host twin round identically.
inline float turbLerp(float a, float b, float f) { return std::fmaf(f, b - a, a); }

class DeviceTurbInflow
{
public:
    // box: turbBox(spec) components u, v, w over nbx x ny x nz cells, i
    // fastest (i + nbx (j + ny k)), narrowed to FP32 once.
    // Pre: every component has nbx * ny * nz entries.
    static std::expected<DeviceTurbInflow, hipError_t>
    make(int nbx, int ny, int nz, double h,
         const std::array<std::vector<double>, 3>& box);

    // du, dv, dw <- the box at the sweep X (one launch, a row per thread).
    void sample(double X);

    const Real* du() const { return du_.data(); }
    const Real* dv() const { return dv_.data(); }
    const Real* dw() const { return dw_.data(); }
    // The three row arrays (ny * nz each), for tests.
    const DeviceScalar& rows(int c) const { return c == 0 ? du_ : c == 1 ? dv_ : dw_; }

private:
    int nbx_ = 0, nrows_ = 0;
    double h_ = 1;
    DeviceScalar tu_, tv_, tw_, du_, dv_, dw_;
};

} // namespace dns::gpu
