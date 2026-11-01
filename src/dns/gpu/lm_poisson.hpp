// Low-Mach pressure solve (masterplan real_bunsen D9 / S3, docs/la.md
// "Separable Poisson"): L_ap phi = rhs on the FLUID cells of a SepBox whose
// faces are binary apertures from a solid map (a face between two fluid cells
// is open, every face touching a solid cell is closed: zero flux). Without a
// solid the operator IS the box Laplacian: one direct SeparablePoisson solve.
// With one: PCG on gpu/la Krylov<float> (FP32 iterate, batched FP64 dots),
// preconditioned by the box's spectral solve restricted to the fluid cells.
// GPU-only header.
#pragma once

#include "dns/gpu/la/krylov.hpp"
#include "dns/gpu/poisson.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>

namespace dns::gpu {

// Per-cell face code (device, one byte per cell; faceCodes()): bit set = that
// face couples the cell to its neighbour (both fluid; interior face, or the
// wrap face of a periodic axis). A boundary face of a non-periodic axis never
// couples: its kind decides (Neumann: no flux; Dirichlet p = 0: 2 x_c / h^2).
// The ONE face-open definition the low-Mach flux kernels read too.
inline constexpr std::uint8_t kLmFaceXLo = 1u << 0, kLmFaceXHi = 1u << 1,
                              kLmFaceYLo = 1u << 2, kLmFaceYHi = 1u << 3,
                              kLmFaceZLo = 1u << 4, kLmFaceZHi = 1u << 5,
                              kLmSolid = 1u << 7;

struct LmPressureOptions
{
    // Stop: recursive |b - A x|_2 <= tol |b|_2 (the rhs norm, so a warm start
    // pays only for what it lacks). Measured FP32 floor: docs/la.md.
    double tol = 1e-6;
    int maxIter = 200;
};

struct LmSolveResult
{
    int iterations = 0;     // PCG iterations; 0 on the direct path
    bool direct = false;    // no solid: one spectral solve
    bool converged = false; // direct: always true
    double bNorm = 0;       // |b|_2 over the fluid cells (PCG path)
    double relResidual = 0; // recursive |r| / |b| (PCG path)
    // FP32 floor estimate eps_f32 |A| |x| / |b| with |A| <= 4 sum 1/h^2
    // (Gershgorin); a tol below it cannot certify (CLAUDE.md "FP32 floor").
    double floorRel = 0;
};

class LmPressure
{
public:
    // solid: host, box.size() bytes, nonzero = solid; empty = no solid (direct
    // path). Error by name: size mismatch, solid with no Dirichlet face, fluid
    // cells enclosed by solid away from every Dirichlet face (phi undetermined
    // there), invalid box, device failure.
    static std::expected<LmPressure, std::string>
    make(const SepBox& box, std::span<const std::uint8_t> solid, const LmPressureOptions& opt = {});

    LmPressure(LmPressure&&) noexcept;
    LmPressure& operator=(LmPressure&&) noexcept;
    ~LmPressure();

    // L_ap phi = rhs on fluid cells, phi = 0 on solid cells. warm: start from
    // the incoming phi (its solid entries are zeroed), else from 0. rhs's solid
    // entries are ignored. Pre: device arrays of box().size(), not aliased.
    // Not converged within maxIter is a result (converged false), not an error.
    std::expected<LmSolveResult, std::string> solve(const Real* rhs, Real* phi, bool warm);

    // y = A x: A = -L_ap on fluid rows, identity on solid rows (SPD). The FP64
    // instantiation is the reference operator. Pre: device arrays, no alias.
    void apply(const float* x, float* y) const;
    void apply(const double* x, double* y) const;

    const std::uint8_t* faceCodes() const; // device, see kLmFace*
    const std::uint8_t* fluid() const;     // device, 1 fluid / 0 solid
    std::size_t solidCells() const;
    bool immersed() const { return solidCells() > 0; }
    SeparablePoisson& spectral();
    const SepBox& box() const;

private:
    struct Impl;
    explicit LmPressure(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace dns::gpu
