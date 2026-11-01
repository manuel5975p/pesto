// The low-Mach (zero-Mach, variable-density, acoustically filtered) flame
// solver (masterplan real_bunsen section 2 / S4; docs/chem.md "Low-Mach").
//
// Model: calorically perfect, equal W, Le = 1, thermodynamic pressure P0
// CONSTANT (open domain), EOS rho T = P0/R; the dynamic pressure p is the
// projection's Lagrange multiplier, O(rho S_L^2). State: rho, rho Y_k (cells),
// the MAC mass fluxes m = rho u (faces; dns/lowmach.hpp's layout), p (cells).
// T = P0/(R rho) and Y_k = rho Y_k / rho are DERIVED.
//
// One SSP-RK3 stage (ca, cb, cr = StageCombine's weights, cr includes dt),
// from (a = the step's start, b = the stage input):
//   1. cell pass on b: T, Y_k, omega (arrheniusRate / arrheniusRate2 / the
//      C2b law: the shipped rate templates), mu(T), div u;
//   2. temperature predictor T* = ca T_a + cb T_b + cr R_T(b), R_T = [-sum_f
//      m_f (T_f - T_c) + div(lambda/cp grad T) + (q/cp) omega] / rho_b, T_f on
//      the species' SHARED limiter (H = T + (q/cp) Y_F rides one linear face
//      formula). T VOTES in it (S5, issues LM8): a shared theta nobody votes
//      for is 2, the DOWNWIND value -- harmless on a uniform species, but T
//      carried that way in a jet core whose species sit below the vote
//      threshold (Y_F 0.07: ulp 7e-9 < 2^-24) grew an odd-even T wiggle to a
//      blow-up; rho* = P0/(R T*), the divergence target D =
//      (ca rho_a + cb rho_b - rho*) / cr;
//   3. momentum predictor m* = ca m_a + cb m_b + cr [-div(m u) + div tau -
//      grad p_b] (central conservative form, mass fluxes x mean velocities:
//      kinetic-energy consistent with the discrete continuity; no added
//      dissipation; full stress with the -2/3 div u part);
//   4. projection on the MASS FLUX: L phi = (div m* - D)/cr (constant
//      coefficient: LmPressure, one spectral solve without solids), m = m* -
//      cr G phi, p = p_b + phi;
//   5. continuity and species on the PROJECTED flux: rho = ca rho_a + cb rho_b
//      - cr div m, rho Y_k likewise with F = m Y_f - rho D grad Y (shared
//      limiter) and the sinks -omega (fuel), -s omega (oxidizer).
// So mass is conserved by telescoping fluxes whatever the solver tolerance, a
// uniform Y is kept to the bit (rho Y and rho ride one m), and the EOS defect
// rho T* - P0/R is the projection's residual (eosErr).
//
// Boundaries (S4; per row since S5): x-lo inflow FACE (m, T, Y of the row:
// convective + diffusive face flux, tangential u = 0, phi Neumann); x-hi
// outflow (p = phi
// = 0 on the face, m from the projection, scalars zero-gradient both ways,
// no diffusion, zero normal-stress gradient); y, z periodic.
//
// Gravity (S7, LowMachSetup::gravity > 0; its own momentum instantiation,
// g = 0 the S6 kernel bit for bit): g along -x, the momentum predictor of
// every x face adds -(rho_f - rho_a) g (rho_f the face density the face's
// velocity reads: the two-cell mean; the outflow face the last cell's, the
// ambient's on backflow), rho_a = P0/(R T_amb). Hydrostatics live in p.
//
// Immersed tube (S8, LowMachSetup::solid non-empty; the Wall instantiation
// of every flux kernel, no solid = the S7 kernels bit for bit): a STAIRCASE
// of solid cells. LmPressure's face codes are the face-open definition: a
// MAC face touching a solid cell is CLOSED, m = 0 in every stage (the
// momentum predictor writes 0, the correction skips it, the apertured PCG
// sees no flux there) -- the wall is impermeable to mass and, riding m,
// carries no convective momentum, enthalpy or species flux. No-slip: a
// closed tangential neighbour face reads as the antimirror of the face
// itself (u = 0 on the wall; the inflow face's tangential rule).
// Isothermal at wallT: the conduction rho D(T_w) (T_c - T_w)/(h/2) through
// each wall face (the inflow face's form). Species-tight: zero diffusive
// flux too. A face whose upwind second neighbour is solid goes central in the
// shared limiter (no second neighbour). Solid cells carry their rho (=
// P0/(R T_w)) and rho Y (0) unchanged: R_T = 0, no reaction. The pressure
// solve is the apertured FP32 PCG preconditioned by the box's spectral
// solve, warm-started per stage from that stage's increment of the previous
// step (phiW_, checkpointed).
//
// Open laterals (S6, LowMachSetup::openY/openZ; their own kernel
// instantiations, periodic = the S5 kernels bit for bit): y (z) faces are
// TRACTION-FREE p = phi = 0 faces (the pressure solve's DD kind), m_n from
// the projection; ONE ghost rule per face, decided by the sign of the face's
// own mass flux: OUTFLOW zero gradient (every cell-centred read across the
// face is the boundary cell's), INFLOW the quiescent AMBIENT (the gas
// entering is room air: T_amb, Y_amb, rho_amb = P0/(R T_amb); tangential
// velocity 0 on the face; face pressure -beta rho_amb u_n^2 / 2 -- the
// total-pressure inflow condition p + rho u_n^2/2 = 0, Moghadam et al.
// 2011's backflow stabilization at beta = 1, removing the incoming kinetic-
// energy flux). Scalars: upwind (ambient | own), no diffusion through an open
// face. With open laterals the x-hi outflow's BACKFLOW takes the same rule
// (ambient scalars, rho_amb, the stabilization). The open faces' mass fluxes
// live beside the lo-face arrays: my's hi face (j = ny) in myHi (nx nz), mz's
// in mzHi (nx ny).
//
// Deterministic: own-index writes, no FP atomics, FP64 reductions min/max
// only (rocPRIM) and the mass tally a fixed-order single-block tree.
// GPU-only header.
#pragma once

#include "dns/checkpoint.hpp"
#include "dns/chem.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/lm_poisson.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/grid.hpp"
#include "dns/lowmach.hpp"
#include "dns/transport.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace dns::gpu {

inline constexpr int kLmMaxY = 2;
// Per-x-column FP64 sums of DeviceLowMach::columnStats: [0] rho, [1] cell
// mean of m_x, [2] T, [3] rho Y_fuel, [4] omega, [5] max |H - H_in| (H = T +
// (q/cp) Y_F, the Shvab-Zeldovich coupling function), [6] cell mean of u, [7]
// p, [8] fuel Y outside [-1e-6, 1 + 1e-6] (count), [9] min Y_F, [10] max
// Y_F, [11] min Z, [12] max Z (two reactants: the mixture fraction; else 0).
inline constexpr int kLmSums = 13;

struct LowMachSetup
{
    Transport tr;              // mu = Pr at Tref = Tu = 1 (box units)
    double prGas = 1;          // P0/R: T = prGas / rho
    int nY = 1, fuel = 0, ox = -1;
    double stoich = 1;
    double lnA = 0, Ta = 0;    // lnA = -inf: no reaction
    double qcp = 0;            // q / cp per unit fuel mass
    PhiRateLaw<double> law;    // n > 0: the C2b A(phi) law (two reactants)
    double mIn = 1, TIn = 1;   // inflow face mass flux and temperature
    double yIn[kLmMaxY] = {1, 0};
    // Per-row inflow (masterplan real_bunsen S5, the Bunsen): the face state
    // of row j + ny k (ny nz entries each; rowY[k] for k < nY). Empty: every
    // row the scalars above. The device ALWAYS reads per-row arrays (FP32
    // copies of the FP64 values, rho = P0/(R T) per row): a uniform profile
    // and the scalars are the same bits (gpu_lowmach_bunsen "uniform").
    std::vector<double> rowM, rowT, rowY[kLmMaxY];
    // H_in = TIn + qcp yIn[fuel]; mixture fraction Z = (zS Y_F - Y_O + zO2)
    // / zDen (PartialPremixModel::mixtureFraction; zDen 0: not reported).
    double zS = 0, zO2 = 0, zDen = 0;
    // Open lateral faces (S6; header comment): false = periodic, the S5
    // solver bit for bit. openZ needs nz > 1. The ambient the open faces
    // (and, when any is open, the x-hi backflow) draw in: T, Y_k; backflow =
    // the stabilization's beta (1: the total-pressure inflow; 0 switches the
    // term off -- an API knob for the stability test, the CLI keeps 1).
    bool openY = false, openZ = false;
    double ambT = 1;
    double ambY[kLmMaxY] = {0, 0};
    double backflow = 1;
    // Gravity (S7): g >= 0 along -x (the jet axis points UP). The ONE
    // buoyancy term of the model, (rho_f - rho_a) g on the x faces' momentum,
    // rho_a = prGas / ambT (the quiescent ambient): the ambient's hydrostatic
    // pressure lives in p, so the ambient at rest is p = 0 exactly and the
    // open faces' p = 0 stays its pressure. 0 = the S6 kernels bit for bit.
    double gravity = 0;
    // The immersed tube (S8; header comment "Immersed tube"): the host solid
    // map (g.size() bytes, nonzero = solid; dns/lowmach.hpp lmTubeSolid),
    // empty = no solid, the S7 kernels bit for bit. wallT = its isothermal T;
    // pcgTol = the apertured PCG's stop |r| <= tol |b| (LmPressureOptions).
    std::vector<std::uint8_t> solid;
    double wallT = 1;
    double pcgTol = 1e-6;
};

// FP64 boundary sums of the CURRENT state (DeviceLowMach::boundarySums;
// fixed-order single-block reductions): face mass fluxes times areas, the
// lateral faces' OUTWARD normal velocity m_n / rho_f (rho_f the upwind
// density: the ambient's on an inflow face), and each species' convective
// flux OUT through x-hi and the open faces with the ambient rule (upwind:
// the cell's Y outflowing, Y_amb inflowing).
struct LmBoundarySums
{
    double mIn = 0;      // through x-lo (in)
    double mOut = 0;     // through x-hi (out; negative = net backflow)
    double mLat = 0;     // net IN through the open lateral faces (0: periodic)
    double uLatMin = 0;  // min outward normal velocity there (< 0: entrainment)
    double uLatMax = 0;  // max (> 0: something leaves sideways)
    double yOut[kLmMaxY] = {0, 0}; // species out through x-hi + the open faces
};

class DeviceLowMach
{
public:
    // Allocates the state (three RK buffers), the scratch and the pressure
    // solve (SepBox x ND, y/z periodic or DD; s.solid: the apertured PCG and
    // its three warm-start buffers). Errors by name (a solid map of the
    // wrong size, wall T <= 0, LmPressure's: no Dirichlet face, enclosed
    // fluid).
    // Pre: g.nx even, s.nY in [1, kLmMaxY], ox < 0 or ox != fuel.
    static std::expected<DeviceLowMach, std::string> make(const Grid& g,
                                                          const LowMachSetup& s);

    // The host state (FP64, rounded to FP32 once) into the step buffers; the
    // mass tally starts at the uploaded mass (FP64). Pre: s matches the grid;
    // with a solid, every face touching it carries 0 (lowmachIC's do; the
    // first stage's predictor zeroes them in its output either way).
    std::expected<void, std::string> upload(const LowMachState& s);

    // One SSP-RK3 step of dt. A pressure-solve failure is an error by name
    // (pcgFailed(): the apertured PCG ran out of iterations -- exit 8).
    std::expected<void, std::string> step(double dt);
    // The tube's apertured PCG (S8): the most iterations of the last step's
    // three stage solves (0: no solid, the direct solve), and whether a solve
    // failed to converge.
    int pcgIterations() const;
    bool pcgFailed() const { return pcgFailed_; }
    bool immersed() const { return pr_.immersed(); }

    // The bounds (FP64 reductions over the current state): convective
    // cfl / max_c sum_a |u_a|/h_a (face maxima per cell); diffusive 0.25 h^2
    // rho_min / (mu(T_max) max(4/3, 1/Pr)), T_max = P0/(R rho_min) (Sutherland
    // monotone); chemistry K / max J (the cp-form Jacobian of chemStableDt).
    double convectiveDt(ReduceScratch& r, double cfl) const;
    double diffusiveDt(ReduceScratch& r) const;
    double chemDt(ReduceScratch& r, double K) const;
    // `--dt-rule cell` (S9, issues LM10): K / max_c (2 C_c + D_c), the
    // per-cell sum of the convective (x2: MC) and diffusive rates (LmPosRate in
    // lowmach.hip): the explicit transport step's positivity bound for T and
    // Y_k at the cell where both are large (the hot, fast products).
    double positivityDt(ReduceScratch& r, double K) const;

    // The cell pass on the current state (T, Y, omega fresh for the stats and
    // frames below).
    void updateCells();
    // kLmSums per x column, one thread per column, (k, j) ascending, FP64.
    // Pre: updateCells() since the last step.
    hipError_t columnStats(DeviceBuffer<double>& scratch,
                           std::vector<double>& host) const;
    // max |div m - D| and max |rho T* - P0/R| / (P0/R) of the last stage of
    // the last step (0 before the first step).
    double divErr(ReduceScratch& r) const;
    double eosErr(ReduceScratch& r) const;
    // max |D| of that stage (divErr's scale).
    double divTargetMax(ReduceScratch& r) const;
    // The FP64 mass tally: the uploaded mass plus every stage's net face flux
    // (in - out) with the RK weights. Synchronous download.
    double massTally() const;
    // The FP64 species tally of rho Y_k (S6): the uploaded inventory plus every
    // stage's net BOUNDARY flux (x-lo convective + diffusive, x-hi and the open
    // faces convective), the exact face values the stage used. It carries no
    // chemistry: with reaction, inventory - tally = -(integral of the sink).
    // Pre: k < nY.
    double speciesTally(int k) const;
    // The current state's boundary sums (LmBoundarySums). Synchronous.
    LmBoundarySums boundarySums() const;

    const DeviceScalar& T() const { return T_; }
    const DeviceScalar& Y(int k) const { return Y_[k]; }
    const DeviceScalar& omega() const { return om_; }
    // The current density (the hi word) and the x-face mass fluxes (nx + 1
    // per row).
    const DeviceScalar& rho() const { return s_.rho; }
    const DeviceScalar& mx() const { return s_.mx; }
    // The dynamic pressure (cells; the jet profiles' integral, S6).
    const DeviceScalar& p() const { return s_.p; }
    // Cell-centred velocity of the current state: the mean of the cell's two
    // face velocities m_f/rho_f per axis (the cell pass's div u faces; at nz
    // == 1 w is the one z face's, 0). For frames and the cone's flank flux.
    void cellVelocity(DeviceScalar& u, DeviceScalar& v, DeviceScalar& w) const;
    const GpuGrid& grid() const { return g_; }
    const LowMachSetup& setup() const { return sp_; }

    // Host copies of the current state (tests): rho, rho Y_k (hi + lo in
    // FP64), mx, my, mz, p. hiOnly: rho, rho Y_k as their hi words. Open
    // laterals: myHi, mzHi too (empty when periodic).
    std::expected<LowMachState, std::string> download(bool hiOnly = false) const;

    // "state.rho", "state.y0..", "state.mx", "state.my", "state.mz",
    // "state.p", the low words "state.rlo", "state.ylo0.." (raw FP32),
    // "state.tally" and "state.ytally" (F64), open laterals "state.myhi" /
    // "state.mzhi", the tube's PCG warm starts "state.phi0..2": the
    // checkpoint's fields.
    std::expected<void, RunError> save(Checkpoint& ck) const;
    std::expected<void, RunError> load(const Checkpoint& ck);

    DeviceLowMach(DeviceLowMach&&) noexcept = default;
    DeviceLowMach& operator=(DeviceLowMach&&) noexcept = default;

    // rho and rho Y_k are COMPENSATED (hi + Kahan lo word; the kernels read
    // hi): FP32 dropped the preheat zone's sub-ulp increments (lowmach.hip,
    // LmWords). m and p are plain (the projection re-determines m each
    // stage).
    struct State
    {
        DeviceScalar rho, p, mx, my, mz;
        DeviceScalar myHi, mzHi; // open laterals: the hi faces (else empty)
        DeviceScalar y[kLmMaxY];
        DeviceScalar rlo, ylo[kLmMaxY];
        DeviceBuffer<double> tally;  // 1 entry: the FP64 mass tally
        DeviceBuffer<double> ytally; // kLmMaxY entries: the species tallies
    };

private:
    DeviceLowMach(GpuGrid g, LowMachSetup s, LmPressure pr)
        : g_(g), sp_(std::move(s)), pr_(std::move(pr))
    {
    }
    // si: the stage index (0..2; the tube's warm-start buffer).
    void stage(const State& a, const State& b, State& out, double ca,
               double cb, double cr, int si, std::string& err);
    // The inflow row buffers (m, rho, T, Y_0, Y_1 or null).
    std::array<const Real*, 5> inflowRows() const;

    GpuGrid g_;
    LowMachSetup sp_;
    LmPressure pr_;
    State s_, q1_, q2_;
    DeviceScalar T_, om_, mu_, divu_, Tstar_, D_, rhs_, phi_;
    DeviceScalar Y_[kLmMaxY];
    // The inflow face per row (ny nz): m, rho = P0/(R T), T, Y_k.
    DeviceScalar inM_, inRho_, inT_, inY_[kLmMaxY];
    // The stage's boundary face fluxes per species (kLmScalars writes, the
    // species tally sums): x-lo, x-hi (ny nz each), y-lo, y-hi (nx nz), z-lo,
    // z-hi (nx ny); periodic axes' segments unused.
    DeviceScalar bflux_;
    mutable DeviceBuffer<double> bsums_; // boundarySums' scratch
    double area_ = 0, inflowArea_ = 0; // hy hz, and sum of mIn over the face
    // S8: per stage the pressure increment = the next step's PCG warm start
    // (allocated with a solid only; checkpointed "state.phi0..2").
    DeviceScalar phiW_[3];
    int pcgIt_[3] = {0, 0, 0};
    bool pcgFailed_ = false;
};

} // namespace dns::gpu
