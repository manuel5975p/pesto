// The module inventory of the compressible FV stage (docs/fvcomp.md "Combinations",
// docs/masterplan_multiphysics.md "Composition, not pairs"): every module of the one
// stage as data, and for every pair of modules ONE decision (live | refused + reason).
// The ONE definition FvRef::make, DeviceFvComp (make, localDt), the CLI routes and the
// generated docs table read. Pure data and lookups: no mesh, no state, no GPU.
//
// A module is a set of additive face / cell terms (or a stage-writer switch) of the
// stage. Within one axis the modules are alternatives (Exclusive); across axes a pair
// is Live unless a rule refuses it by name. Requirements (TFLES needs premixed
// species, the Charlette efficiency needs LES) are single-module rules.
#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>

namespace dns {

enum class FvModule : std::uint8_t
{
    LesSmag,
    LesSigma,
    TflesUniform,
    TflesDynamic,
    TflesEfficiency, // Charlette's efficiency on a TFLES flame
    RansSa,
    RansSst,
    ChemC0, // premixed one-step
    ChemC1, // two reactants (C1; --equivalence --sl-law onestep, C2, is this stage)
    ChemC2b, // two reactants with the GRI-3.0 A(phi) law (FvChem::law)
    Gravity,
    BodyForce, // constant force per volume (the channel's drive)
    Turb,      // --turb: the synthetic box in the IC and the swept per-face inflow rows (M5)
    Steady,    // local time stepping (DeviceFvComp::localDt)
    Compensated,
    Fp64,
    Shock, // --shock hll|sense on a polyMesh (dns/fvshock, docs/nozzle.md)
    Count
};
inline constexpr std::size_t kFvModuleCount = std::size_t(FvModule::Count);
using FvModuleSet = std::bitset<kFvModuleCount>;

enum class FvAxis : std::uint8_t
{
    Turbulence, // eddy viscosity: LES or RANS, one slot
    Flame,      // thickening
    Efficiency,
    Chemistry,
    Gravity,
    Drive,
    Inflow, // the turbulent inflow: IC and per-face rows, no stage term
    Time,
    Precision,
    Capture // shock capture: the sensor, the HLL face term, the background term
};

struct FvModuleInfo
{
    FvModule id;
    FvAxis axis;
    std::string_view name;     // short, the docs' and the harness' column
    std::string_view spelling; // where a user names it
};

enum class FvPairDecision : std::uint8_t
{
    Live,
    Refused,
    Exclusive // two choices of one axis
};

struct FvPairRule
{
    FvPairDecision decision = FvPairDecision::Live;
    std::string_view message{}; // the refusal's error text (empty if live)
    std::string_view why{};     // the reason (physics or a missing piece)
    std::string_view gate{};    // live: the gates besides gpu_fv_matrix that cover it
};

struct FvRequirement
{
    FvModule module;
    FvModuleSet anyOf; // at least one of these present
    std::string_view message;
};

// Every module, in FvModule order.
std::span<const FvModuleInfo> fvModules();
const FvModuleInfo& fvModuleInfo(FvModule m);
// Pre: a != b. Symmetric.
FvPairRule fvPair(FvModule a, FvModule b);
std::span<const FvRequirement> fvRequirements();

FvModuleSet fvModuleSet(std::initializer_list<FvModule> ms);
// The first refusal of s: pairs by ascending (a, b), then the requirements in table
// order. Deterministic; empty set and every live set pass.
std::expected<void, std::string> fvModulesCheck(const FvModuleSet& s);
// s plus what its requirements need (the first module of each anyOf), to a fixed point.
FvModuleSet fvModulesClosure(FvModuleSet s);
// "LES smag + C0"; "laminar" for the empty set.
std::string fvModuleSetName(const FvModuleSet& s);
// The decision part of docs/fvcomp.md "Combinations": the module list, the pair
// matrix, the refusals with their reasons, the live pairs' hand gates.
std::string fvCombinationsMarkdown();

} // namespace dns
