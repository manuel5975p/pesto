// The module inventory of the low-Mach flame stage (M4; docs/fvincomp.md "Low-Mach
// combinations", docs/masterplan_multiphysics.md "Composition, not pairs"): every module of
// gpu/lowmach's stage and its FaceMesh twin (gpu/fv_incomp DeviceFvLowMach) as data, and ONE
// decision per pair (live | refused + reason). dns/fvmodules' shape; the compressible family
// keeps its own. Read by the CLI (dns/config/flame_scope.cpp lowMachModules: dns flame --physics lowmach), the solvers'
// make() guards and the harness (test_gpu_lm_matrix). Pure data: no mesh, no state, no GPU.
#pragma once

#include "dns/fvmodules.hpp"

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>

namespace dns {

enum class LmModule : std::uint8_t
{
    LesSmag,
    LesSigma,
    TflesUniform,
    TflesDynamic,
    TflesEfficiency, // Charlette's efficiency
    ChemC0,          // premixed one-step
    ChemC2,          // two reactants, --equivalence (one-step A)
    ChemC2b,         // two reactants, --sl-law gri30 (A(phi))
    Gravity,         // --gravity (the burner)
    Open,            // --laterals open / an `open` patch
    Rim,             // --rim (the immersed tube; FV: an isothermal Wall patch)
    Turb,            // --turb (M5): the synthetic box in the IC and the swept inflow rows
    Count
};
inline constexpr std::size_t kLmModuleCount = std::size_t(LmModule::Count);
using LmModuleSet = std::bitset<kLmModuleCount>;

enum class LmAxis : std::uint8_t
{
    Turbulence,
    Flame,
    Efficiency,
    Chemistry,
    Gravity,
    Open,
    Rim,
    Inflow
};

struct LmModuleInfo
{
    LmModule id;
    LmAxis axis;
    std::string_view name;     // the docs' and the harness' column
    std::string_view spelling; // where a user names it
};

struct LmRequirement
{
    LmModule module;
    LmModuleSet anyOf;
    std::string_view message;
};

// Every module, in LmModule order.
std::span<const LmModuleInfo> lmModules();
const LmModuleInfo& lmModuleInfo(LmModule m);
// Pre: a != b. Symmetric. FvPairRule: decision, error text, reason, hand gates.
FvPairRule lmPair(LmModule a, LmModule b);
std::span<const LmRequirement> lmRequirements();

LmModuleSet lmModuleSet(std::initializer_list<LmModule> ms);
// The first refusal of s: pairs by ascending (a, b), then the requirements in table order.
std::expected<void, std::string> lmModulesCheck(const LmModuleSet& s);
// s plus what its requirements need (the first of each anyOf), to a fixed point.
LmModuleSet lmModulesClosure(LmModuleSet s);
// "LES sigma + C0"; "laminar" for the empty set.
std::string lmModuleSetName(const LmModuleSet& s);
// docs/fvincomp.md "Low-Mach combinations": modules, pair matrix, refusals, hand gates.
std::string lmCombinationsMarkdown();

} // namespace dns
