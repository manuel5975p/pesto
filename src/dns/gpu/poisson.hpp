// Device pressure Poisson solver: exact spectral inversion of the 2nd-order
// FD Laplacian via hipFFT. Periodic mode uses the full-grid R2C transform;
// NeumannBox solves the zero-flux (mirror) problem on the interior of a grid
// carrying a one-cell ghost ring, via per-axis DCT-II built from same-length
// FFTs (Makhoul permutation + twiddle -- no padding, no extra memory scale).
// 2D: nz == 1 is the exact 2D restriction (z stencils cancel identically under
// periodic wrap); both modes then transform x/y only. GPU-only header; hipFFT
// types stay inside the .hip TU.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"

#include <expected>
#include <memory>

namespace dns::gpu {

// Boundary handling of the spectral solve.
//  Periodic:   all axes periodic, transform over the full grid.
//  NeumannBox: homogeneous Neumann on the faces of the interior box [1, n-2]
//              per non-degenerate axis (an nz == 1 axis has no ghosts and
//              stays untransformed); ghost-ring entries of rhs are ignored
//              and p's ghost ring is written as zero.
enum class PoissonBc { Periodic, NeumannBox };

class DevicePoissonSolver
{
public:
    // Builds the FFT plans once; expensive, amortize over the run.
    static std::expected<DevicePoissonSolver, hipError_t>
    make(const GpuGrid& g, PoissonBc bc = PoissonBc::Periodic);

    DevicePoissonSolver(DevicePoissonSolver&&) noexcept;
    DevicePoissonSolver& operator=(DevicePoissonSolver&&) noexcept;
    ~DevicePoissonSolver();

    // Solve lap(p) = rhs; p overwritten, zero mean (over the interior box in
    // NeumannBox mode). rhs/p distinct device buffers of full grid size.
    void solve(const DeviceScalar& rhs, DeviceScalar& p);

private:
    struct Impl;
    explicit DevicePoissonSolver(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace dns::gpu
