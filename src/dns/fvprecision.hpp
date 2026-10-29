// Arithmetic of the incompressible FV path (`dns flow --polymesh`, docs/fv.md
// "Precision"): FP32 default, FP64 the parity mode, and the FP32 solve
// floors. Pure, host-includable (cli/config and gpu/fv_simple read it).
#pragma once

#include <cstdint>

namespace dns {

// Fp32: fields, geometry, matrices, Krylov vectors, AMG and the kernels in
// float, every reduction (dots, norms, normFactor, adjustPhi, continuity,
// forces) accumulated in FP64. Fp64: everything in double (parity mode).
enum class FvPrecision : std::uint8_t { Fp32, Fp64 };

// Solve floors in OpenFOAM's normalised-residual units (sum|b - Ax| /
// normFactor): a solve stops at max(fvSolution tolerance, floor), relTol
// unchanged. 0 = none.
struct FvFloors
{
    double U = 0; // the U components
    double p = 0;
    double s = 0; // turbulence scalars (k, omega, nuTilda)
};

// The FP32 floors, MEASURED (docs/fv.md "Precision"): with no floor the
// initial residual of a converged FP32 run sits at a noise level (median of
// the last 100 iterations, rotor2 L1 / L1 SST / F1): U 3.1e-7 / 7.9e-7 /
// 4.6e-7, p 4.6e-7 / 6.6e-7 / 6.8e-7, k 2.2e-8, omega 7.3e-10 (SST). Each floor
// is the first decade above its field's noise, so a converged solve costs no
// iteration instead of chasing noise; no solve reached maxIter at any floor
// tried (0, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6): the stall criterion alone gives 0.
inline constexpr FvFloors kFp32Floors{1e-6, 1e-6, 1e-7};

struct FvNumerics
{
    FvPrecision precision = FvPrecision::Fp64;
    FvFloors floors{};
};

// The numerics of a precision: Fp64 no floors, Fp32 kFp32Floors.
constexpr FvNumerics fvNumerics(FvPrecision p)
{
    return {p, p == FvPrecision::Fp32 ? kFp32Floors : FvFloors{}};
}

inline const char* fvPrecisionName(FvPrecision p) { return p == FvPrecision::Fp32 ? "FP32" : "FP64"; }

} // namespace dns
