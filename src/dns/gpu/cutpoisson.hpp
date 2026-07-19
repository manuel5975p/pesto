// Cut-cell pressure solve: the CG SOLVER axis. Aperture-weighted Poisson on the
// fluid subspace, preconditioned by the spectral (FFT / Makhoul-DCT) inverse
// Laplacian from gpu/poisson.hpp. Split out of gpu/cutcell.hpp, which keeps the
// geometry axis -- the apertured operators this solver calls.
// GPU-only header.
#pragma once

#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/poisson.hpp"
#include "dns/gpu/reductions.hpp"

#include <expected>

namespace dns::gpu {

// Aperture-weighted pressure Poisson M(p) = rhs by CG on the fluid subspace
// (constant mode projected out), preconditioned by the periodic FFT inverse
// Laplacian: the apertured operator differs from the constant-coefficient one
// only at the body, so the FFT solve clusters the spectrum and CG converges in
// tens of iterations. FP32 fields, FP64 dot products.
class DeviceCutPoissonSolver
{
public:
    // cut must outlive the solver. tol is the per-cell RMS residual target
    // (same semantics as the CPU CutPoissonSolver); FP32 floors near ~1e-6.
    // bc selects the spectral preconditioner: NeumannBox expects the grid to
    // carry a one-cell ghost ring with all boundary apertures closed (see
    // closeBoxBoundary); the apertured operator itself is unchanged.
    static std::expected<DeviceCutPoissonSolver, hipError_t>
    make(const GpuGrid& g, const DeviceCutGeometry& cut, int maxIters, double tol,
         PoissonBc bc = PoissonBc::Periodic);

    DeviceCutPoissonSolver(DeviceCutPoissonSolver&&) noexcept = default;
    DeviceCutPoissonSolver& operator=(DeviceCutPoissonSolver&&) noexcept = default;

    // Solve M(p) = rhs; p has zero mean over fluid cells (zero in solid) on
    // return. warmStart reuses the incoming p as initial guess. Returns the
    // iterations taken.
    int solve(const DeviceScalar& rhs, DeviceScalar& p, bool warmStart = false);

private:
    DeviceCutPoissonSolver(const GpuGrid& g, const DeviceCutGeometry& cut,
                           int maxIters, double tol, DevicePoissonSolver fft,
                           DeviceScalar r, DeviceScalar z, DeviceScalar d,
                           DeviceScalar q, DeviceScalar t, ReduceScratch red,
                           DeviceBuffer<double> sc, DeviceBuffer<Real> ab);

    void applyA(const DeviceScalar& x, DeviceScalar& y); // y = -M(x), SPD on fluid
    void precondition(const DeviceScalar& r, DeviceScalar& z); // z ~ A^-1 r
    void projectMean(DeviceScalar& x); // subtract fluid mean, zero solid cells
    // r,r-style fluid dot product left in device slot dst (no host sync).
    void fluidDotInto(const DeviceScalar& a, const DeviceScalar& b, double* dst);

    GpuGrid g_;
    const DeviceCutGeometry* cut_;
    int maxIters_;
    double tol_;
    DevicePoissonSolver fft_;
    DeviceScalar r_, z_, d_, q_, t_; // PCG workspace + FFT staging
    ReduceScratch red_;
    // Device-resident CG scalars so alpha/beta/mean never round-trip to the host.
    // sc_: [rr, dq, rz, rzNew, sum, bb]; ab_: [alpha, beta] (Real for the updates).
    DeviceBuffer<double> sc_;
    DeviceBuffer<Real> ab_;
};

// One cut-cell projection: solve M(phi) = divCut(U)/dt, U -= dt grad(phi) on
// open faces, then re-mask the solid. Warm-starts phi from the previous call.
struct DeviceCutProjectionScratch
{
    DeviceScalar div, phi;

    static std::expected<DeviceCutProjectionScratch, hipError_t> make(std::size_t n)
    {
        auto d = DeviceScalar::make(n);
        if (!d) return std::unexpected(d.error());
        auto p = DeviceScalar::make(n);
        if (!p) return std::unexpected(p.error());
        return DeviceCutProjectionScratch{std::move(*d), std::move(*p)};
    }
};

// Returns the CG iterations taken by the pressure solve. Pre: dt > 0.
int projectCut(const GpuGrid& g, const DeviceCutGeometry& cut,
               DeviceCutPoissonSolver& poisson, Real dt, DeviceField& U,
               DeviceCutProjectionScratch& s);

} // namespace dns::gpu
