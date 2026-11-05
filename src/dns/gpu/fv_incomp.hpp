// Time-accurate incompressible FV solver on any FaceMesh (docs/fvincomp.md):
// the MAC solver's SSP-RK3 with one pressure projection per stage, collocated.
//
//   State      cell velocity u, face flux phi (own -> nei), boundary values.
//   Convection skew-symmetric central, R_P = -(1/V) sum_f phi_f u_N / 2: a
//              skew matrix for ANY phi, so its energy is 0 to rounding.
//   Viscosity  compact (u_N - u_P) d.S/|d|^2 + AlongD correction; LES adds
//              nut_f (grad u + grad u^T).S. A wall face: OpenFOAM's whole stress
//              nu (g_b + dev2(g_b^T)).S, g_b one-sided along the normal.
//   Predictor  u* = ca a + cb b + cr R(b), phi* = S.(linear interpolation of u*).
//   Projection A p = -div(phi*)/c, A the compact |S|^2/(S.d) Laplacian (fixed,
//              AMG built once), c = cb dt; phi = phi* - c fp, u = u* - c G,
//              fp the face gradient flux, G its Perot reconstruction
//              (1/V) sum_f +-r_f fp_f, exact for linear p. Non-orthogonal
//              meshes: fp carries the deferred k . (least-squares grad p), nCorr
//              solves, phi closes on the last whichever nCorr.
//
// Temperature (make given T0; dns conv --physics incompressible): T rides the
// same stage on the same phi, skew-symmetric central + the compact diffusion.
// The Boussinesq force is a FACE quantity, never a cell one: phi* gets c B_f,
// B_f = (T_f - Tref) betaG.S, and the reconstruction reads fp - B_f, so a
// hydrostatic state cancels face by face and leaves u exactly (Mencinger; the
// cell form leaves a wall cell half a hydrostatic gradient per stage). A wall
// face stays Neumann and carries no net force. Absent: none of it runs. betaG 0: a
// passive tracer, the momentum bit for bit the run without T. LES + T (mci): the eddy
// heat flux kappa + nut_f / Pr_t (kLesPrt) on the internal faces, the boundary faces
// molecular; nut 0 = the molecular T bit for bit. Modules and pairs: dns/fvimodules,
// checked by make.
//
// Deterministic: CSR ascending gather, each pass writes its own index, fixed-
// order FP64 reductions, no FP atomics. One source for R in {double, float}.
// GPU-only header.
#pragma once

#include "dns/fvincomp.hpp"
#include "dns/fvlowmach.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/la/graph.hpp"

#include <array>
#include <cmath>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace dns {
class Checkpoint; // dns/checkpoint.hpp
}

namespace dns::gpu {

// One step's pressure account (all stages).
struct FvIncompStepLog
{
    int solves = 0, iterations = 0;
    int unconverged = 0;       // solves that hit maxIter before their target
    double divRatio = 0;       // worst |div phi|_2 / |div phi*|_2 estimate of the step
};

// Volume integrals of the current state.
struct FvIncompDiag
{
    double K = 0;       // sum V |u|^2 / 2
    double eps = 0;     // molecular viscous KE loss, nu sum_f corr.S |du|^2 (u.L u)
    double divMax = 0;  // max_c |sum phi| / V
    double divRms = 0;  // V-weighted rms of the same
    double uMax = 0;
};

// The semi-discrete KE rates of the current state (inviscid budget gate).
struct FvIncompBudget
{
    double conv = 0;     // sum V u . C(u): 0 to rounding by construction
    double grossConv = 0; // sum |V u_c C_c|: the scale of conv
    double press = 0;    // -sum V u . G p with the last p
    double convT = 0, grossConvT = 0; // sum V T C(T) and its scale (temperature only)
};

// Force on the `force` patches (components of Fp + Fv): pressure with p_b = p_P,
// the value the momentum equation applies; viscous with the wall stress.
struct FvIncompForces
{
    std::array<double, 3> fp{}, fv{};
};

// Host copies, FaceMesh numbering.
struct FvIncompState
{
    std::vector<double> U, Ub, p, phi, phiB; // [3 nC] [3 nB] [nC] [nI] [nB]
    std::vector<double> T, Tb;               // [nC] [nB]; empty without temperature
};

class DeviceFvIncomp
{
public:
    // Uploads s, builds and factors the pressure hierarchy, sets the state from
    // U0 ([3 nC]; boundary values from s), projects it once. Pre: s from
    // buildFvIncompSetup; g = la::graphOf of the same mesh. Error: device
    // allocation, AMG setup, the projection's, a module set dns/fvimodules refuses
    // (gravity without T). T0 ([nC]) non-empty: temperature on (c.kappa, c.betaG,
    // s.bT); empty: the solver without it, bit for bit.
    static std::expected<DeviceFvIncomp, std::string> make(const FvIncompSetup& s, const dns::la::CellGraph& g,
                                                           const FvIncompControls& c, const std::vector<double>& U0,
                                                           const std::vector<double>& T0 = {});

    // cfl / max_c(max(conv_c, diff_c)), conv_c = sum_f |phi_f| / (2 V), diff_c =
    // sum_f (nu + 2 nut_f) corr.S / V: the MAC runner's rule on a uniform mesh.
    // With temperature nu reads max(nu, kappa) (2 nut_f >= the eddy heat nut_f / Pr_t).
    // Error: a device failure.
    std::expected<double, std::string> stableDt(double cfl);
    // One SSP-RK3 step. Pre: dt within a small factor of the last step's: the
    // approximate projection's fixed point carries cb dt, so a step of f dt moves
    // u by ~(1 - f) of that memory for ANY f << 1 and p by 1/f (docs/fvincomp.md,
    // gpu_fv_incomp T7). Reach an end time in equal steps. Error: a non-finite
    // residual, a solver failure.
    std::expected<FvIncompStepLog, std::string> step(double dt);
    std::expected<FvIncompDiag, std::string> diagnostics();
    std::expected<FvIncompBudget, std::string> budget();
    std::expected<FvIncompForces, std::string> forces();
    std::expected<FvIncompState, std::string> state() const;
    // Heat flux out of the fluid per boundary face, [nB]. Pre: temperature on.
    std::expected<std::vector<double>, std::string> wallHeat();
    // Cell gradient of u, [9 nC], (3 i + q) nC + c = d_i u_q. Pre: temperature on.
    std::expected<std::vector<double>, std::string> gradU();

    DeviceFvIncomp(DeviceFvIncomp&&) noexcept;
    DeviceFvIncomp& operator=(DeviceFvIncomp&&) noexcept;
    ~DeviceFvIncomp();

    struct Impl;

private:
    DeviceFvIncomp();
    std::unique_ptr<Impl> p_;
};

// --- Low-Mach -------------------------------------------------------------
// The reacting low-Mach model of gpu/lowmach on any FaceMesh (docs/fvincomp.md
// "Low-Mach"), on this file's mesh view, gradients and projection.
//
//   State      rho, rho Y_k (cells; FP32 compensated: hi + Kahan lo), rho u (cells),
//              the face mass flux m, p. T = P0/(R rho), Y = rho Y / rho derived.
//   Stage      gpu/lowmach's order, its algebra verbatim:
//              1 T, Y, omega, u from b; 2 T* and the divergence target D;
//              3 (rho u)* = a + (cb (b - a) + cr R), R the conservative central
//              momentum flux m (u_o + u_n)/2 and the viscous stress, no pressure;
//              4 m* = S . linear((rho u)*), A p = (V D - div m*)/cr + kg,
//              m = m* - cr fp, rho u -= cr G; 5 rho, rho Y on the PROJECTED m.
//   Pressure   A is the incompressible GEOMETRIC matrix, no density in it: the
//              projection acts on the MASS flux, so 1/rho never appears (structured:
//              LmPressure's constant coefficient). Built and factored once.
//   Limiter    one theta per face over Y_k and T (fvThetaVote): T + (q/cp) Y keeps
//              one face formula; a face whose upwind cell has no second neighbour
//              goes central (FvLowMachTables::first).
//   Open       (FvLowMachTables::bOpen) ONE rule per face by the sign of its own
//              flux: out zero gradient, in the ambient at rest (T_a, Y_a, rho_a,
//              u_t 0) with p = -beta rho_a u_n^2/2; no diffusion; the tangential
//              traction only. Absent: none of it runs.
//   Gravity    (model.gravity) the face force (rho_f - rho_a) g . S in m* and out of
//              the reconstruction (fp - B): hydrostatics in p, a hydrostatic state
//              cancels face by face. Absent: none of it runs.
//   LES        (model.cs, M4) gpu/lowmach's eddy terms on faces: mu_t = rho (cs Delta)^2
//              rate(least-squares grad u) per cell and stage; mu + mu_t,f in the stress,
//              rho D + mu_t,f/Sc_t on internal faces; boundary faces molecular.
//   TFLES      (model.tfles, C0) F E rho D (+ (1 - Omega) mu_t/Sc_t) there, omega E/F;
//              the sensor dilates along FvLowMachTables::walk. Absent: the M3 kernels.
//
// Mass telescopes whatever the solver tolerance; a uniform Y is kept to the bit
// (rho Y and rho ride one m through one source). Energy: rho T = P0/R pointwise,
// so the semi-discrete balance holds to rounding (budget()); the march's balance
// closes to the time scheme's order, m in the T predictor being the stage's input.
struct FvLowMachDt
{
    double conv = 0, diff = 0, chem = 0; // cfl / max rate, 1 / max rate, K / max J
    double pos = HUGE_VAL;               // --dt-rule cell: posK / max_c (2 C_c + D_c); posK 0: not evaluated
};

// The current state's boundary face fluxes (the stage kernels' expressions on it):
// phiB [nB] mass, fy[s] [nB] species s (convective + Fick), outward positive; heat
// [nB] rho D grad T . S of a fixed-T face (inflow, isothermal wall: Fourier over cp,
// into the mesh positive), 0 elsewhere.
struct FvLowMachFaces
{
    std::vector<double> phiB;
    std::array<std::vector<double>, kFvLmMaxY> fy;
    std::vector<double> heat;
};

// The inventories and their FP64 tallies. A tally = the uploaded inventory plus every
// stage's net boundary flux with the RK weights. Rounding: mass, y (y + burn == yTally).
struct FvLowMachTotals
{
    double mass = 0, massTally = 0;
    std::array<double, kFvLmMaxY> y{}, yTally{}; // no chemistry in yTally
    double burn = 0;                             // time integral of Sum V omega
    // Sum V rho T (constant by the EOS), and that plus the time integral of the net
    // boundary enthalpy flux + (q/cp) Sum V omega.
    double enthalpy = 0, enthalpyTally = 0;
    double divErr = 0, eosErr = 0;               // max |div m - D|, max |rho T* R/P0 - 1|
};

// The semi-discrete energy balance of the current state: Sum V (rho R_T + T R_rho) -
// (boundary enthalpy flux in + (q/cp) Sum V omega) and the scale of its terms.
struct FvLowMachBudget
{
    double defect = 0, gross = 0;
};

// Host copies, FaceMesh numbering; rho, ry hi + lo in FP64. ry[1] empty at nY == 1.
struct FvLowMachCells
{
    std::vector<double> rho, ru, p, T, omega, phi, phiB;
    std::vector<double> mut; // M4: mu_t of this state (empty without LES)
    std::array<std::vector<double>, kFvLmMaxY> ry;
};

// M4: the eddy fields of the current state: max mu_t, max F E, the Omega-weighted mean E
// (NaN without TFLES or flame). gpu/lowmach's LmEddyStats.
struct FvLowMachEddy
{
    double mutMax = 0, feMax = 1, eBar = NAN;
};

class DeviceFvLowMach
{
public:
    // Uploads the mesh, builds and factors the pressure hierarchy once, lays the
    // flux of ic. Pre: s, t of the same bound table; g = la::graphOf of the mesh; ic
    // sized by it; c.precision picks R. Error: no p-fixed face, a device allocation,
    // AMG setup.
    static std::expected<DeviceFvLowMach, std::string> make(const FvIncompSetup& s, const FvLowMachTables& t,
                                                            const dns::la::CellGraph& g, const FvLowMachModel& model,
                                                            FvIncompControls c, const FvLowMachIC& ic);

    // The bounds on the current state: conv = cfl / max_c sum_f |m|/(2 V min(rho_f,
    // rho_c)); diff = 1 / max_c mu(T_c) max(4/3, 1/Pr) sum_f (|S|^2/S.d) / (rho_c V), the
    // structured 0.25 h^2 rho/mu on a uniform 2D mesh; chem = K / max J, gpu/lowmach's J.
    // posK > 0 (--dt-rule cell, gpu/lowmach's LmPosRate on faces): pos = posK / max_c
    // (2 conv_c + diff_c) with this rule's per-cell rates; 0: pos not evaluated.
    std::expected<FvLowMachDt, std::string> stableDt(double cfl, double K, double posK = 0);
    // Boundary fluxes of the current state. Error: a device failure.
    std::expected<FvLowMachFaces, std::string> faces();
    // The flame index xi = grad Y_F . grad Y_O / (|grad Y_F| |grad Y_O|) per cell on the
    // least-squares gradients, 0 where either |grad Y| < floor (gpu/derived's
    // flameIndexCentred on any mesh). Pre: nY == 2.
    std::expected<std::vector<double>, std::string> flameIndex(double floor);
    // One SSP-RK3 step. Pre: dt within a small factor of the last step's
    // (DeviceFvIncomp::step's memory of c). Error: non-finite pressure rhs, solver.
    std::expected<FvIncompStepLog, std::string> step(double dt);
    std::expected<FvLowMachTotals, std::string> totals();
    std::expected<FvLowMachBudget, std::string> budget();
    std::expected<FvLowMachCells, std::string> cells();
    // M4: mu_t, F E, E of the current state (model.cs, model.tfles). Error: a device failure.
    std::expected<FvLowMachEddy, std::string> eddyStats();
    // --checkpoint-dt: every word a step carries to the next, raw (F32 or F64 by
    // precision): the state "state.fvlm.rho", "rlo", "ru", "y<s>", "ylo<s>"; the
    // projection's warm start and memory "state.fvlm.p" (the PCG's x0), "phi",
    // "phiB" (the last stage's face mass flux: its Rhie-Chow term of c); the FP64
    // tallies "state.fvlm.tally" and the uploaded "state.fvlm.enthalpy0". Eddy
    // and TFLES fields are of the current state (refreshEddy), not carried.
    // Errors: exit 6 (download).
    std::expected<void, RunError> save(Checkpoint& ck) const;
    // --restart: save's records into this solver (same mesh, model, precision).
    // Errors: a record missing or mis-sized exit 3, an upload exit 6.
    std::expected<void, RunError> load(const Checkpoint& ck);
    // --turb (M5): replace the boundary velocity and mass flux tables (FvIncompSetup::bU,
    // FvLowMachTables::rUb, [3 nB] each, the make() layout) for the next steps: the
    // swept box's inflow rows. Only inflow faces read them. Error: a size, an upload.
    std::expected<void, std::string> setInflow(const std::vector<double>& bU, const std::vector<double>& rUb);

    DeviceFvLowMach(DeviceFvLowMach&&) noexcept;
    DeviceFvLowMach& operator=(DeviceFvLowMach&&) noexcept;
    ~DeviceFvLowMach();

    struct Impl;

private:
    DeviceFvLowMach();
    std::unique_ptr<Impl> p_;
};

} // namespace dns::gpu
