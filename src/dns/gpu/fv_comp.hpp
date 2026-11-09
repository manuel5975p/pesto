// The compressible FV family on FaceMesh (docs/fvcomp.md): explicit DNS/LES of
// the FvRef scheme on the GPU, any mesh a FaceMesh holds. The face physics IS
// dns/fvflux.hpp (the same templates, instantiated for device R); the pass
// structure IS FvRef's: primitives -> boundary states -> least-squares cell
// gradients -> [eddy viscosity] -> materialized face fluxes -> per-cell gather
// of its CSR faces in ascending face order, fused into the SSP-RK3 stage
// out = ca a + cb b + cr R(b), ca = 1 - cb.
//
// One source, R in {float, double} (FvPrecision: FP32 default, FP64 parity with
// FvRef). Deterministic: one thread per cell or face, every pass writes its own
// index only, no FP atomics; dt is an exact min; tallies are downloaded and
// summed in ascending face order in FP64, so reruns are byte-identical.
//
// Species (FvChem, docs/fvcomp.md "Species"): rows 5.. of the state, their own
// kernels after the mean-flow ones, nY == 0 launches none of them. Species LES and
// TFLES ("Species LES and TFLES"): the eddy / thickened instantiations of the species
// face kernel, the TFLES coefficient passes at each step's start; off = none of them.
//
// RANS (FvCompConfig::rans, dns/fvrans.hpp, docs/fvcomp.md "RANS"): the state
// gains rows 5.. (rho nu~ | rho k, rho omega); mu_t rides the LES slot, so the
// mean-flow kernels are the LES instantiation; the scalars have their own
// passes after the mean gather, point-implicit in the stage. None = no pass, no
// buffer, the solver bit for bit.
//
// Gravity (dns conv) is two additive passes, never a kernel edit: its face terms
// after the fluxes, their sum on the stage after the gather.
//
// Shock capture (FvCompConfig::shock, docs/nozzle.md): the sensor per step, one face
// pass on the materialized F. No hot kernel is edited.
//
// A boundary face indexes a row of the FvBc table: one row per patch, or
// FvCompConfig::bcRow's per-face rows (the jet's inflow). GPU-only header.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvprecision.hpp"
#include "dns/fvref.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/sgs.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace dns {
class Checkpoint; // dns/checkpoint.hpp
}

namespace dns::gpu {

struct FvCompConfig
{
    Transport tr;
    std::vector<FvBc> bc{};  // the FvBc table: a row per patch, then per-face rows (FvRef::make's rules)
    std::vector<int> bcRow{}; // empty: a face reads its patch's row; else per boundary face, its row
    FvPrecision precision = FvPrecision::Fp32;
    // LES: csLes > 0 adds mu_t = rho (C Delta)^2 op(grad u) per cell (dns/sgs.hpp,
    // the structured operators and constants), face mean into mu, cp/prt scaled
    // into kappa. Delta^2 = V^(2/3) in 3D, the 2D cell area on a 2D mesh. A wall
    // face carries the molecular stress only (mu_t = 0 at the wall). csLes == 0:
    // no eddy kernel runs.
    double csLes = 0;
    SgsModel sgs = SgsModel::Smagorinsky;
    double prt = 0.9;
    // FP32 only: march the two-word state (hi + lo, gpu/internal/stage_write.hpp).
    // A plain FP32 state drops updates below ulp(q)/2: cd of a steady cylinder
    // wanders 0.5% (docs/fvcomp.md). Costs the two extra low-word stage writes.
    bool compensated = false;
    // Species rows and the one-step chemistry (FvRef::make's rules), nY == 0 off.
    // With LES (premixed and two reactants): + nut_f / Sc_t on every internal face, Sc_t =
    // Pr_t = prt (docs/fvcomp.md "Species LES and TFLES").
    FvChem chem{};
    // The thickened flame (FvConfig::tfles's rules and terms): F E, E/F, Omega from
    // each step's input state (step(); rhs, evaluate, stableDt and speciesFields form
    // them from the current state), held over the three stages. Empty: off, no
    // TFLES kernel, the species kernels bit for bit.
    std::optional<TflesSetup> tfles{};
    // RANS closure (exclusive with csLes): FvRef's rules. Pr_t = kPrandtlTurb.
    FvRansConfig rans{};
    // Constant force per volume, rhs of m + f, of E + f.u; 0 = no pass.
    std::array<double, 3> bodyForce{};
    // Per patch: its faces enter the surface tally. Empty: every Wall patch.
    std::vector<std::uint8_t> tallied{};
    // The gauge of the tallied pressure force (subtracted per face, so a closed
    // body's constant pressure never rounds into the force).
    double pRef = 0;
    // Acceleration vector (FvConfig::gravity, the same rules). Zero: no gravity
    // pass runs. dns conv's buoyancy: rho(T) through the EOS, no Boussinesq term.
    std::array<double, 3> gravity{};
    // The hydrostatic ambient its open patches' rows are (FvConfig::ambient: checked, not
    // read by a kernel; the rows are the table's).
    std::optional<FvAmbient> ambient{};
    // --turb (M5): FvConfig::turb (the inventory's module; the rows are setBcRows').
    bool turb = false;
    // Shock capture (FvConfig::shock's terms, docs/nozzle.md): the sensor at each step's
    // start (kFvShockSense; rhs, evaluate, patchFlux, shockCells form it from the current
    // state), one face pass after the fluxes. None: no launch, no buffer.
    FvShock shock{};
};

// Force and heat on the tallied faces, FP64 sums in ascending face order.
struct FvSurface
{
    std::array<double, 3> fp{}, fv{}; // gauge pressure and viscous force on the body
    double heat = 0;                  // conduction, fluid into body
    double area = 0;
    std::vector<double> heatPatch{};  // heat per FaceMesh::patches entry
};

// FP64 cell sums in ascending cell order.
struct FvTotals
{
    double mass = 0, energy = 0, kinetic = 0, machMax = 0;
    std::array<double, 3> mom{};
};

class DeviceFvComp
{
public:
    // Pre: fm outlives nothing (every table is copied). Errors (user input, exit
    // 3 at the caller): whatever FvRef::make rejects, the inventory's precision pairs
    // (dns/fvmodules). Errors (device): allocation.
    static std::expected<DeviceFvComp, std::string> make(const FaceMesh& fm, const FvCompConfig& cfg,
                                                         int threads = 0);

    // Pre: s sized for the mesh with nV == 5 + chem.nY + the closure's scalars. FP32 narrows.
    std::expected<void, std::string> upload(const dns::FvState& s);
    std::expected<void, std::string> download(dns::FvState& s) const;

    // dq/dt of the current state, FvRef::rhs's meaning.
    std::expected<void, std::string> rhs(dns::FvState& out);
    // rhs plus every intermediate (FvEval's arrays; bnd and bphi on active
    // boundary faces only, F on active faces, the rest 0). Widened to double.
    std::expected<void, std::string> evaluate(FvEval& ev, dns::FvState& out);
    // One SSP-RK3 step of the current state; dt within stableDt. Failures of the
    // launches surface at the next stableDt / check.
    void step(double dt);
    // min over cells of FvRef's bound with mu_t added: momentum factor max(4/3,
    // gamma/Pr) on mu, max(4/3, gamma/Pr_t) on mu_t. Errors: rho or p not
    // positive-finite, RANS mu_t not finite (the diverged cell by name), a
    // launch failure.
    std::expected<double, std::string> stableDt(FvDtLimits lim = {});
    // Local time stepping (pseudo-time, not time-accurate): every later step(dtRef)
    // advances cell c by dtRef lts[c], lts[c] = dt_c/dtRef >= 1 from stableDt's
    // per-cell bound of the current state; every stage writer takes it. Returns
    // dtRef. Errors: stableDt's; the inventory's time-axis refusals (dns/fvmodules:
    // species, gravity, LES). Never calling it keeps step() bit for bit.
    std::expected<double, std::string> localDt(FvDtLimits lim = {});
    // chemDt of FvRef on the current state (FP64 from the state words): k / max J,
    // +inf without a reaction. Errors: a launch failure.
    std::expected<double, std::string> chemDt(double k);
    // State and the ONE omega of the current state (nC values; empty if nY == 0).
    std::expected<void, std::string> speciesFields(dns::FvState& s, std::vector<double>& omega);
    // Flux of row r (0 .. nV - 1) of the current state through each patch, outward,
    // FP64 sum in ascending face order. Errors: a launch failure.
    std::expected<std::vector<double>, std::string> patchFlux(int r);
    // Tally of the current state.
    std::expected<FvSurface, std::string> surface();
    // Sums of the current state.
    std::expected<FvTotals, std::string> totals() const;
    // --shock: cells with a nonzero byte, and the largest byte, of the current state's
    // sensor. Errors: no --shock, a launch failure.
    std::expected<long, std::string> shockCells();
    std::expected<int, std::string> shockByteMax();
    // Stage writes the SST guard clipped (k >= 0, omega >= floor rho). 0 otherwise.
    std::expected<unsigned long long, std::string> floorHits() const;
    // Per-cell mu_t of the current state (0 without LES or RANS).
    std::expected<std::vector<double>, std::string> eddyViscosity();
    // TFLES: Ebar = sum V Omega E / sum V Omega (NaN if Omega == 0), FEmax = max F E
    // of the current state's coefficients, FP64 in ascending cell order. Errors:
    // no TFLES, a launch failure.
    struct TflesStats
    {
        double Ebar = 0, FEmax = 0;
    };
    std::expected<TflesStats, std::string> tflesStats();
    // u, v, w, |grad-free| primitives and the velocity gradient of the current
    // state for frames: rows rho u v w e p T, gradient [12][nC] as FvEval.
    std::expected<void, std::string> cellFields(std::array<std::vector<double>, 7>& prim,
                                                std::array<std::vector<double>, 12>& grad);
    // Exact Galilean boost of the current state by du (the quiet start).
    void boost(const std::array<double, 3>& du);
    // Replace one row of the FvBc table. Pre: patch valid, kind unchanged.
    std::expected<void, std::string> setBc(int patch, const FvBc& bc);
    // --checkpoint-dt: every word a step carries to the next, raw (F32 or F64 by
    // precision): "state.fv.q", the compensated low words "state.fv.lo", the [thrust]
    // ledger "state.fv.ledger.{I,lo,om,omlo}" (F64, only while setLedger is on), the SST
    // guard's count "state.fv.hits". Nothing else survives a step (eddy, TFLES and
    // gradient fields are of the step's input state; the ledger's ledS/ledPart are
    // rewritten before every read). Errors: exit 6 (download).
    std::expected<void, RunError> save(Checkpoint& ck) const;
    // --restart: save's records into this solver (same mesh, closure, precision);
    // after it the next step is the checkpointed run's bit for bit. Pre: under [thrust],
    // setLedger(true) first (it zero-allocates the words load overwrites). Errors: a
    // record missing or mis-sized exit 3, an upload exit 6.
    std::expected<void, RunError> load(const Checkpoint& ck);
    // [thrust]'s impulse ledger (FvLedger, docs/nozzle.md "Thrust"): on zeroes it, and every later
    // step() adds per stage a fixed-order FP64 sum of each patch's materialized boundary F (one
    // block per patch, ascending partials, no atomics) and of V omega, with the weights of that
    // step's stage writes. Off (the default): no launch, the buffers freed. equalWeights (TEST
    // ONLY): dt/3 per stage, G2.2's control. Errors: gravity, body force, RANS (sources that are
    // no boundary flux), local time stepping (no one dt), an allocation.
    std::expected<void, std::string> setLedger(bool on, bool equalWeights = false);
    // The ledger since setLedger(true). Errors: off, a download.
    std::expected<FvLedger, std::string> ledger() const;
    // Replace rows first .. first + rows.size() of the FvBc table in one upload (M5: the
    // turbulent inflow's per-face rows, rewritten between steps). Pre: the range is in
    // the table, every kind unchanged (else an error by name, nothing written).
    std::expected<void, std::string> setBcRows(int first, std::span<const FvBc> rows);

    int cells() const;
    FvPrecision precision() const;
    // Device bytes of the solver (state, stage buffers, fields, tables).
    std::size_t deviceBytes() const;

    DeviceFvComp(DeviceFvComp&&) noexcept;
    DeviceFvComp& operator=(DeviceFvComp&&) noexcept;
    ~DeviceFvComp();

    struct Impl;

private:
    DeviceFvComp();
    std::unique_ptr<Impl> p_;
};

} // namespace dns::gpu
