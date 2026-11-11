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
#include "dns/fvmotion.hpp"
#include "dns/fvprecision.hpp"
#include "dns/fvsimple.hpp"
#include "dns/la/amg.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
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

// One actuationDiskSource's row of an iteration (OpenFOAM's
// actuationDiskSource.dat columns; diskDir is FvDisk::dir).
struct FvDiskLog
{
    std::array<double, 3> Uref{}; // monitor-cell mean of the iteration's start U
    double Cp = 0, Ct = 0, a = 0, T = 0;
};

struct FvIterLog
{
    int iter = 0;
    std::vector<FvDiskLog> disks;        // per FoamCase::disks entry
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
    // pimpleFoam (one FvIterLog per outer corrector): continuityErrs.H's (sum
    // local, global) after each pressure corrector; p holds every corrector's
    // non-orthogonal solves in order. SIMPLE: empty.
    std::vector<std::array<double, 2>> cont;
    // Moving mesh, the outer corrector that moved it (the first): CorrectPhi's
    // pcorr solves and its continuityErrs.H (correctPhi yes), meshCourantNo.H
    // (checkMeshCourantNo yes; meshCo[0] mean, [1] max). msSetup: the device's
    // share of the mesh update (uploads, pattern, matrices, AMG; synchronised).
    std::vector<FvSolveLog> pcorr;
    std::optional<std::array<double, 2>> contCorr, meshCo;
};

// A mesh update handed to timeStep (dns/fvmotion FvMeshMotion::update): the
// setup rebuilt on the moved points and the motion's flux / wall velocity.
struct FvMovedMesh
{
    const FvSimpleSetup* s = nullptr;
    const FvMeshMove* mv = nullptr;
    // cyclicAMI on a moving mesh: the new matrix graph (the AMI couplings
    // changed); the device rebuilds its matrices, GS and AMG on it. Null: the
    // pattern is unchanged.
    const dns::la::CellGraph* g = nullptr;
};

// CourantNo.H on the current phi: 0.5 max(sum|phi| / V) deltaT and
// 0.5 (sum sum|phi| / sum V) deltaT.
struct FvCourant
{
    double mean = 0, max = 0;
};

// One pimpleFoam time step: its outer correctors in order.
struct FvStepLog
{
    std::vector<FvIterLog> outer;
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
    // Moving mesh (empty otherwise): the face velocity Uf [3 nI] [3 nB] and the
    // mesh flux [nI] [nB]. phi above is then the RELATIVE flux, as pimpleFoam
    // writes it.
    std::vector<double> Uf, UfB, meshPhi, meshPhiB;
};

class DeviceSimple
{
public:
    // Uploads setup s and the case controls; builds the p AMG hierarchy (host,
    // once). RAS (s.ras.model != Laminar, buildRasSetup done): the model's
    // fields, then validate(). num: arithmetic and solve floors (default:
    // FP64, none). c.disks non-empty: s.disks from buildDiskSetup. Error:
    // device allocation, AMG setup, disks unbound.
    static std::expected<DeviceSimple, std::string> make(const FvSimpleSetup& s, const FoamCase& c,
                                                         const dns::la::CellGraph& g, const FvNumerics& num = {});

    // One SIMPLE iteration. Error: a non-finite residual, a solver failure,
    // adjustPhi's "continuity error cannot be removed".
    std::expected<FvIterLog, std::string> iterate();
    // One EXPERIMENTAL coupled outer iteration (FvCoupledOpts) from the current
    // state (U, p, phi). Pre: laminar. Error: a solver failure, a non-finite
    // residual.
    std::expected<FvIterLog, std::string> coupledIterate(const FvCoupledOpts& o);
    // pimpleFoam (FoamCase::pimple.on; docs/fvpimple.md, gpu/fv_pimple.hip).
    // courant: CourantNo.H of the current flux at deltaT. timeStep: one time
    // step of size deltaT > 0 -- the old-time fields are the state at the call,
    // then pimpleFoam.C's outer / pressure / non-orthogonal corrector loops
    // (UEqn.H, pEqn.H, turbulence->correct()) with the Final-iteration entries.
    // Error: not a pimpleFoam case, a solver failure, a non-finite residual.
    // Moving mesh (FoamCase::motion.on): `moved` is required -- the mesh update
    // of this step (the new time's points), applied at the first outer
    // corrector as mesh.update() is: V0, geometry, meshPhi, movingWallVelocity,
    // then correctPhi.H, fvc::ddtCorr(U, Uf), correctUf, makeRelative. Static
    // mesh: `moved` must be null.
    std::expected<FvCourant, std::string> courant(double deltaT);
    std::expected<FvStepLog, std::string> timeStep(double deltaT, const FvMovedMesh* moved = nullptr);
    // The HOST part of the matrix rebuild a moved cyclicAMI mesh needs (ELL pattern, colours, AMG hierarchy),
    // ahead of timeStep: pure host work, so FvMeshMotion::update runs it beside the wall distance. Same values
    // as timeStep computes without it. Pre: s is the moved setup (s.ras unread), g the FvMovedMesh::g object.
    void prepareMatrices(const FvSimpleSetup& s, const dns::la::CellGraph& g);
    // OpenFOAM's forces function object FoamCase::forces[k] on the current
    // state (k out of range: zeros).
    std::expected<FvForceSums, std::string> forces(int k = 0);
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
