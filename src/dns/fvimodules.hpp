// The module inventory of the time-accurate incompressible FV stage (gpu/fv_incomp
// DeviceFvIncomp; docs/fvincomp.md "Incompressible combinations", docs/
// masterplan_multiphysics.md "Composition, not pairs"): every module of the stage as
// data, ONE decision per pair (live | refused + reason), and the modules refused alone
// (not built on this family, with the reason). dns/fvmodules' shape (the compressible
// family) and dns/lmmodules' (low-Mach) keep their own. Read by the CLI (dns/config/common.cpp:
// dns flow / tgv / conv on the incompressible FV route), DeviceFvIncomp::make and the
// harness (test_gpu_fvi_matrix). Pure data: no mesh, no state, no GPU.
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

enum class FviModule : std::uint8_t
{
    LesSmag,
    LesSigma,
    RansSa,      // refused alone: not built on this family
    RansSst,     // refused alone
    Temperature, // the scalar row T (a passive tracer without Gravity)
    Gravity,     // the Boussinesq face force (T - Tref) betaG . S
    ChemC0,      // refused alone: constant-density reacting scalars
    ChemC1,      // refused alone
    ChemC2b,     // refused alone
    Count
};
inline constexpr std::size_t kFviModuleCount = std::size_t(FviModule::Count);
using FviModuleSet = std::bitset<kFviModuleCount>;

enum class FviAxis : std::uint8_t
{
    Turbulence,
    Scalar,
    Force,
    Chemistry
};

struct FviModuleInfo
{
    FviModule id;
    FviAxis axis;
    std::string_view name;     // the docs' and the harness' column
    std::string_view spelling; // where a user names it
    std::string_view refused{}; // non-empty: not built on this family; the error text
    std::string_view why{};     // the reason of a refusal
};

struct FviRequirement
{
    FviModule module;
    FviModuleSet anyOf;
    std::string_view message;
};

// Every module, in FviModule order.
std::span<const FviModuleInfo> fviModules();
const FviModuleInfo& fviModuleInfo(FviModule m);
// Pre: a != b. Symmetric. A pair holding a module refused alone is Refused with that
// module's text.
FvPairRule fviPair(FviModule a, FviModule b);
std::span<const FviRequirement> fviRequirements();

FviModuleSet fviModuleSet(std::initializer_list<FviModule> ms);
// The first refusal of s: modules refused alone (ascending), pairs by ascending (a, b),
// then the requirements in table order. Deterministic; the empty set passes.
std::expected<void, std::string> fviModulesCheck(const FviModuleSet& s);
// s plus what its requirements need (the first of each anyOf), to a fixed point.
FviModuleSet fviModulesClosure(FviModuleSet s);
// "LES sigma + T + gravity"; "laminar" for the empty set.
std::string fviModuleSetName(const FviModuleSet& s);
// docs/fvincomp.md "Incompressible combinations": modules, pair matrix, refusals,
// requirements, hand gates.
std::string fviCombinationsMarkdown();

} // namespace dns
