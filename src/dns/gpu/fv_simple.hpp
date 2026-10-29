// The incompressible FV solver on FaceMesh (masterplan_meshsolver P5b,
// docs/fv.md): OpenFOAM v2606 simpleFoam's SIMPLE / SIMPLEC iteration on the
// identical polyMesh, FP64, GPU-resident, linear solves on gpu/la (U:
// BiCGStab + multicolour GS, p: AMG-PCG) with OpenFOAM's normalised-residual
// stop. One SIMPLE iteration = UEqn.H + pEqn.H of simpleFoam verbatim:
// MRF.correctBoundaryVelocity, fvm::div(phi,U) [bounded] Gauss linear |
// linearUpwind | upwind, MRF Coriolis, -fvc::div(nu dev2(T(grad U))) -
// fvm::laplacian(nu, U) (corrected), UEqn.relax, the momentum predictor,
// rAU / HbyA / constrainHbyA / phiHbyA, MRF.makeRelative, adjustPhi,
// SIMPLEC (consistent), the non-orthogonal pressure correctors with
// setReference, phi = phiHbyA - pEqn.flux(), p.relax, the momentum corrector.
// FP64 (parity) or FP32 (FvPrecision). Deterministic: one thread per
// row/face, CSR face order, no FP atomics, fixed-order FP64 reductions.
// GPU-only header.
#pragma once

#include "dns/fvcase.hpp"
#include "dns/fvprecision.hpp"
#include "dns/fvsimple.hpp"
#include "dns/la/amg.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace dns::gpu {

// One linear solve's OpenFOAM-style account. floorLimited: it stopped on the
// FvFloors floor alone (its residual above both the case's tolerance and
// relTol x initial) -- not an error, reported in the log.
struct FvSolveLog
{
    double initial = 0, final = 0;
    int iterations = 0;
    bool floorLimited = false;
};

struct FvIterLog
{
    int iter = 0;
    std::array<FvSolveLog, 3> U{};       // components; unsolved (2D) left zero
    std::vector<FvSolveLog> p;           // one per non-orthogonal corrector
    double contLocal = 0, contGlobal = 0;
    // RAS: the scalar solves in solve order (SST: omega, k; SA: nuTilda) and
    // whether bound() acted on them.
    int nT = 0;
    std::array<FvSolveLog, 2> T{};
    std::array<bool, 2> bounded{};
    // A coupled outer iteration (kIt >= 0; SIMPLE: -1): U[j].initial / p[0].initial
    // are the PROBE (simpleFoam's initial residuals at the iteration's start
    // state), kIt / kRel the FGMRES iterations and relative residual, alphaC the
    // pseudo-time relaxation used, ms* the stage times (synchronised), jac the
    // G-JAC relative errors (U rows, continuity rows; < 0: not checked).
    int kIt = -1;
    double kRel = 0, alphaC = 0, msRes = 0, msSetup = 0, msKrylov = 0, msUpdate = 0;
    std::array<double, 2> jac{-1.0, -1.0};
};

// EXPERIMENTAL coupled pressure-velocity prototype (plans/plan_perf_5_coupled.md
// P0b, docs/fv.md "Coupled prototype"): inexact Picard on simpleFoam's own
// fixed-point residual (Rhie-Chow coefficient = the case's alpha_U-relaxed
// rAU), composed Jacobian, one linearised SIMPLEC step as preconditioner,
// FGMRES(30). eta: FGMRES relTol per outer iteration; alpha: initial
// pseudo-time relaxation alpha_c (1 = none), ramped to 1 by SER on the probe;
// check: G-JAC (J v against a frozen-coefficient difference of R*) at the
// first outer iteration.
struct FvCoupledOpts
{
    double eta = 0.1, alpha = 1.0;
    bool check = false;
    int maxKrylov = 300;
};

// forces function object: pressure and viscous force and moment sums.
struct FvForceSums
{
    std::array<double, 3> fp{}, fv{}, mp{}, mv{};
};

// Host copies of the state in FaceMesh numbering.
struct FvState
{
    std::vector<double> U, Ub, p, pb, phi, phiB; // [3 nC] [3 nB] [nC] [nB] [nI] [nB]
    // RAS: s[q] / sb[q] the transported scalars (SST k, omega; SA nuTilda), nut.
    std::array<std::vector<double>, 2> s, sb;
    std::vector<double> nut, nutB;
};

class DeviceSimple
{
public:
    // Uploads setup s and the case controls; builds the p AMG hierarchy (host,
    // once). RAS (s.ras.model != Laminar, buildRasSetup done): the model's
    // fields, then validate(). num: arithmetic and solve floors (default:
    // FP64, none). Error: device allocation, AMG setup.
    static std::expected<DeviceSimple, std::string> make(const FvSimpleSetup& s, const FoamCase& c,
                                                         const dns::la::CellGraph& g, const FvNumerics& num = {});

    // One SIMPLE iteration. Error: a non-finite residual, a solver failure,
    // adjustPhi's "continuity error cannot be removed".
    std::expected<FvIterLog, std::string> iterate();
    // One EXPERIMENTAL coupled outer iteration (FvCoupledOpts) from the current
    // state (U, p, phi). Pre: laminar. Error: a solver failure, a non-finite
    // residual.
    std::expected<FvIterLog, std::string> coupledIterate(const FvCoupledOpts& o);
    // OpenFOAM's forces function object on the current state.
    std::expected<FvForceSums, std::string> forces();
    std::expected<FvState, std::string> state() const;
    int iteration() const;

    DeviceSimple(DeviceSimple&&) noexcept;
    DeviceSimple& operator=(DeviceSimple&&) noexcept;
    ~DeviceSimple();

    struct Impl;

private:
    DeviceSimple();
    std::unique_ptr<Impl> p_;
};

} // namespace dns::gpu
