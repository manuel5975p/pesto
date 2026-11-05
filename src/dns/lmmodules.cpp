#include "dns/lmmodules.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <format>

namespace dns {
namespace {

using M = LmModule;
using D = FvPairDecision;

constexpr std::array<LmModuleInfo, kLmModuleCount> kModules{{
    {M::LesSmag, LmAxis::Turbulence, "LES smag", "--les Cs --sgs smag"},
    {M::LesSigma, LmAxis::Turbulence, "LES sigma", "--les C --sgs sigma"},
    {M::TflesUniform, LmAxis::Flame, "TFLES uniform", "--tfles F --tfles-sensor uniform"},
    {M::TflesDynamic, LmAxis::Flame, "TFLES dynamic", "--tfles F [--tfles-sensor dynamic]"},
    {M::TflesEfficiency, LmAxis::Efficiency, "Charlette", "--efficiency charlette"},
    {M::ChemC0, LmAxis::Chemistry, "C0", "dns flame --physics lowmach (premixed, bunsen)"},
    {M::ChemC2, LmAxis::Chemistry, "C2", "--equivalence PHI --sl-law onestep"},
    {M::ChemC2b, LmAxis::Chemistry, "C2b", "--equivalence PHI [--sl-law gri30]"},
    {M::Gravity, LmAxis::Gravity, "gravity", "--gravity G (--mode bunsen)"},
    {M::Open, LmAxis::Open, "open", "--laterals open; type = open in --bc"},
    {M::Rim, LmAxis::Rim, "rim", "--rim LIP, an isothermal wall patch"},
    {M::Turb, LmAxis::Inflow, "turb", "--turb U (the planar flame, --nz >= 8)"},
}};

struct Rule
{
    M a, b;
    FvPairRule r;
};

constexpr std::string_view kTfC2 =
    "--tfles with --equivalence: the thickened flame is premixed C0 only (low-Mach TFLES of two reactants is "
    "refused)";
constexpr std::string_view kTfC2Why =
    "F keeps one S_L and E wrinkles one (S_L, delta_L) pair: the C2 burner carries an outer Burke-Schumann "
    "diffusion sheet whose structure thickening does not preserve, and the sensor's normalisation is the one-step "
    "C0 manifold's (tflesSensorLogMax)";
constexpr std::string_view kTfOpen =
    "--tfles with --laterals open: the sensor's T-gated max filter wraps y and z periodic (gpu/tfles)";
constexpr std::string_view kTfOpenWhy = "an open lateral is no wrap: the dilation would read across the box";
constexpr std::string_view kLesRim =
    "--les with --rim: not built (no wall model for mu_t at the immersed staircase)";
constexpr std::string_view kLesRimWhy =
    "the cell-centred velocity differences read the solid cells' zero velocity: mu_t peaks at the lip with no "
    "damping (the incompressible cut-cell LES caveat)";
constexpr std::string_view kTfRim = "--tfles with --rim: not built (the sensor dilates through the tube's solid cells)";
constexpr std::string_view kTfRimWhy = "the max filter has no solid gate";
constexpr std::string_view kTurbOpen =
    "--turb with --laterals open: not built (the synthetic box is periodic in y and z)";
constexpr std::string_view kTurbOpenWhy = "an open lateral is no period of the box; its rows would wrap across it";
constexpr std::string_view kTurbBurner =
    "--turb on the low-Mach burner (--rim, --gravity): not built (the low-Mach --turb is the planar flame box's)";
constexpr std::string_view kTurbBurnerWhy =
    "the burner's inflow is the jet's per-row profile (lmInflowRow); a turbulent jet weight J(y) is the compressible "
    "slot's and has no low-Mach counterpart";
constexpr std::string_view kTurbC2 = "--turb with --sl-law onestep (model C2): not built (no turbulent number)";
constexpr std::string_view kTurbC2Why =
    "the turbulent box is measured on C0 and C2b; the uncorrected one-step C2 is the S1 diagnostic step";
constexpr std::string_view kTurbGate = "gpu_lm_turb (IC, rows, structured vs FV), tools/chem/turb_fv.sh lm, lmpairs (S_c vs structured)";
constexpr std::string_view kGate =
    "gpu_lm_matrix; gpu_lowmach_les (structured X, laminar limit)";

constexpr FvPairRule refuse(std::string_view m, std::string_view w) { return {.decision = D::Refused, .message = m, .why = w}; }
constexpr FvPairRule live(std::string_view gate) { return {.decision = D::Live, .gate = gate}; }

constexpr std::array kRules{
    Rule{M::TflesUniform, M::ChemC2, refuse(kTfC2, kTfC2Why)},
    Rule{M::TflesUniform, M::ChemC2b, refuse(kTfC2, kTfC2Why)},
    Rule{M::TflesDynamic, M::ChemC2, refuse(kTfC2, kTfC2Why)},
    Rule{M::TflesDynamic, M::ChemC2b, refuse(kTfC2, kTfC2Why)},
    Rule{M::TflesEfficiency, M::ChemC2, refuse(kTfC2, kTfC2Why)},
    Rule{M::TflesEfficiency, M::ChemC2b, refuse(kTfC2, kTfC2Why)},
    Rule{M::TflesUniform, M::Open, refuse(kTfOpen, kTfOpenWhy)},
    Rule{M::TflesDynamic, M::Open, refuse(kTfOpen, kTfOpenWhy)},
    Rule{M::TflesEfficiency, M::Open, refuse(kTfOpen, kTfOpenWhy)},
    Rule{M::LesSmag, M::Rim, refuse(kLesRim, kLesRimWhy)},
    Rule{M::LesSigma, M::Rim, refuse(kLesRim, kLesRimWhy)},
    Rule{M::TflesUniform, M::Rim, refuse(kTfRim, kTfRimWhy)},
    Rule{M::TflesDynamic, M::Rim, refuse(kTfRim, kTfRimWhy)},
    Rule{M::TflesEfficiency, M::Rim, refuse(kTfRim, kTfRimWhy)},
    Rule{M::Turb, M::Open, refuse(kTurbOpen, kTurbOpenWhy)},
    Rule{M::Turb, M::Rim, refuse(kTurbBurner, kTurbBurnerWhy)},
    Rule{M::Turb, M::Gravity, refuse(kTurbBurner, kTurbBurnerWhy)},
    Rule{M::Turb, M::ChemC2, refuse(kTurbC2, kTurbC2Why)},
    Rule{M::Turb, M::ChemC0, live(kTurbGate)},
    Rule{M::Turb, M::ChemC2b, live(kTurbGate)},
    Rule{M::Turb, M::LesSmag, live(kTurbGate)},
    Rule{M::Turb, M::LesSigma, live(kTurbGate)},
    Rule{M::Turb, M::TflesUniform, live(kTurbGate)},
    Rule{M::Turb, M::TflesDynamic, live(kTurbGate)},
    Rule{M::Turb, M::TflesEfficiency, live(kTurbGate)},
    // Live pairs with a hand gate beside the matrix.
    Rule{M::LesSigma, M::ChemC0, live("gpu_lowmach_les L1 (sigma on a planar flame = DNS bitwise, both solvers)")},
    Rule{M::LesSmag, M::ChemC0, live("gpu_lowmach_les X2 (FV vs structured: the scheme gap shrinks)")},
    Rule{M::TflesUniform, M::ChemC0, live("gpu_lowmach_les X1 (FV vs structured: rounding), S_L of F")},
    Rule{M::TflesDynamic, M::ChemC0, live("gpu_lowmach_les X1 (FV vs structured: rounding)")},
    Rule{M::LesSmag, M::ChemC2b, live("gpu_lm_matrix (b): Z == 1 kept under the one-Sc_t eddy flux")},
    Rule{M::LesSigma, M::ChemC2b, live("gpu_lm_matrix (b): Z == 1 kept under the one-Sc_t eddy flux")},
};

constexpr LmModuleSet bits(std::initializer_list<M> ms)
{
    LmModuleSet s;
    for (const M m : ms) s.set(std::size_t(m));
    return s;
}

const std::array<LmRequirement, 4> kRequires{{
    {M::TflesEfficiency, bits({M::LesSmag, M::LesSigma}),
     "--efficiency charlette needs --les (the sub-grid velocity of a filtered field)"},
    {M::TflesEfficiency, bits({M::TflesDynamic, M::TflesUniform}), "--efficiency needs --tfles F (>= 1)"},
    {M::TflesUniform, bits({M::ChemC0}), "--tfles: the premixed C0 flame"},
    {M::TflesDynamic, bits({M::ChemC0}), "--tfles: the premixed C0 flame"},
}};

std::string_view decisionWord(D d) { return d == D::Live ? "live" : d == D::Refused ? "refused" : "exclusive"; }

} // namespace

std::span<const LmModuleInfo> lmModules() { return kModules; }

const LmModuleInfo& lmModuleInfo(LmModule m)
{
    assert(m < LmModule::Count);
    return kModules[std::size_t(m)];
}

FvPairRule lmPair(LmModule a, LmModule b)
{
    assert(a != b && a < LmModule::Count && b < LmModule::Count);
    const auto it = std::ranges::find_if(kRules, [&](const Rule& r) { return (r.a == a && r.b == b) || (r.a == b && r.b == a); });
    if (it != kRules.end()) return it->r;
    if (lmModuleInfo(a).axis == lmModuleInfo(b).axis)
        return {.decision = D::Exclusive, .message = "low-Mach modules: one choice per axis", .why = "two choices of one axis"};
    return live(kGate);
}

std::span<const LmRequirement> lmRequirements() { return kRequires; }

LmModuleSet lmModuleSet(std::initializer_list<LmModule> ms) { return bits(ms); }

std::expected<void, std::string> lmModulesCheck(const LmModuleSet& s)
{
    for (std::size_t a = 0; a < kLmModuleCount; ++a) {
        if (!s.test(a)) continue;
        for (std::size_t b = a + 1; b < kLmModuleCount; ++b) {
            if (!s.test(b)) continue;
            const FvPairRule r = lmPair(LmModule(a), LmModule(b));
            if (r.decision != D::Live) return std::unexpected(std::string(r.message));
        }
    }
    for (const LmRequirement& q : kRequires)
        if (s.test(std::size_t(q.module)) && (s & q.anyOf).none()) return std::unexpected(std::string(q.message));
    return {};
}

LmModuleSet lmModulesClosure(LmModuleSet s)
{
    for (bool grew = true; grew;) {
        grew = false;
        for (const LmRequirement& q : kRequires) {
            if (!s.test(std::size_t(q.module)) || (s & q.anyOf).any()) continue;
            for (std::size_t m = 0; m < kLmModuleCount; ++m)
                if (q.anyOf.test(m)) {
                    s.set(m);
                    grew = true;
                    break;
                }
        }
    }
    return s;
}

std::string lmModuleSetName(const LmModuleSet& s)
{
    std::string n;
    for (std::size_t m = 0; m < kLmModuleCount; ++m) {
        if (!s.test(m)) continue;
        if (!n.empty()) n += " + ";
        n += kModules[m].name;
    }
    return n.empty() ? std::string("laminar") : n;
}

std::string lmCombinationsMarkdown()
{
    std::string o = "<!-- generated: dns::lmCombinationsMarkdown (src/dns/lmmodules.cpp); test_gpu_lm_matrix markdown -->\n";
    o += "| Module | Axis | Spelling |\n|---|---|---|\n";
    static constexpr std::string_view axes[] = {"turbulence", "flame", "efficiency", "chemistry",
                                                "gravity",    "open",  "rim",        "inflow"};
    for (const LmModuleInfo& m : kModules)
        o += std::format("| {} | {} | `{}` |\n", m.name, axes[std::size_t(m.axis)], m.spelling);
    o += "\nPair matrix: L live, R refused, x one axis (exclusive).\n\n|   |";
    for (const LmModuleInfo& m : kModules) o += std::format(" {} |", m.name);
    o += "\n|---|";
    for (std::size_t i = 0; i < kLmModuleCount; ++i) o += "---|";
    o += "\n";
    for (const LmModuleInfo& a : kModules) {
        o += std::format("| {} |", a.name);
        for (const LmModuleInfo& b : kModules) {
            if (a.id == b.id) {
                o += " . |";
                continue;
            }
            const D d = lmPair(a.id, b.id).decision;
            o += d == D::Live ? " L |" : d == D::Refused ? " R |" : " x |";
        }
        o += "\n";
    }
    o += "\n| Pair | Decision | Error text | Why |\n|---|---|---|---|\n";
    for (std::size_t a = 0; a < kLmModuleCount; ++a)
        for (std::size_t b = a + 1; b < kLmModuleCount; ++b) {
            const FvPairRule r = lmPair(LmModule(a), LmModule(b));
            if (r.decision != D::Refused) continue;
            o += std::format("| {} + {} | {} | `{}` | {} |\n", kModules[a].name, kModules[b].name, decisionWord(r.decision),
                             r.message, r.why);
        }
    for (const LmRequirement& q : kRequires) {
        std::string need;
        for (std::size_t m = 0; m < kLmModuleCount; ++m)
            if (q.anyOf.test(m)) need += (need.empty() ? "" : " or ") + std::string(kModules[m].name);
        o += std::format("| {} without {} | requirement | `{}` | |\n", lmModuleInfo(q.module).name, need, q.message);
    }
    o += "\n| Live pair | Hand gates |\n|---|---|\n";
    for (std::size_t a = 0; a < kLmModuleCount; ++a)
        for (std::size_t b = a + 1; b < kLmModuleCount; ++b) {
            const FvPairRule r = lmPair(LmModule(a), LmModule(b));
            if (r.decision == D::Live && !r.gate.empty() && r.gate != kGate)
                o += std::format("| {} + {} | {} |\n", kModules[a].name, kModules[b].name, r.gate);
        }
    return o;
}

} // namespace dns
