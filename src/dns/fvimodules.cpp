#include "dns/fvimodules.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <format>

namespace dns {
namespace {

using M = FviModule;
using D = FvPairDecision;

constexpr std::string_view kRans =
    "--rans on the time-accurate incompressible FV solver: not built (steady RANS on a FaceMesh is simpleFoam: dns "
    "flow --polymesh CASE or --solver simple)";
constexpr std::string_view kRansWhy =
    "the family's RAS is gpu/fv_simple + gpu/fv_ras, OpenFOAM v2606 kOmegaSST / SpalartAllmaras verbatim and gated "
    "to 1e-9 (gpu_fv_simple IF-T); URANS on the projection stage would be a second implementation of the closures "
    "with its own wall treatment and no consumer";
constexpr std::string_view kChem =
    "species on the incompressible FV solver: not built (constant-density reacting scalars; the flame on a FaceMesh is "
    "dns flame --physics lowmach --solver fv)";
constexpr std::string_view kChemWhy =
    "decided on evidence (issues.md FI1): no oracle (neither structured incompressible solver, gpu/incomp nor "
    "gpu/boussinesq, carries a scalar beyond T, so gate 5 has no counterpart); no consumer (every flame model, C0, C1, "
    "C2b, is calibrated by its heat release: S_L, T_ad; constant density removes it, DeviceFvLowMach is the "
    "variable-density flame on this mesh view); a passive tracer is the T row without gravity (gpu_fvi_matrix: "
    "momentum bitwise, conv T energy 0), so a scalar row would add no kernel";

constexpr std::array<FviModuleInfo, kFviModuleCount> kModules{{
    {M::LesSmag, FviAxis::Turbulence, "LES smag", "--les Cs [--sgs smag]"},
    {M::LesSigma, FviAxis::Turbulence, "LES sigma", "--les C --sgs sigma (3D)"},
    {M::RansSa, FviAxis::Turbulence, "RANS SA", "--rans sa-neg", kRans, kRansWhy},
    {M::RansSst, FviAxis::Turbulence, "RANS SST", "--rans sst-2003m, sst-vm", kRans, kRansWhy},
    {M::Temperature, FviAxis::Scalar, "T", "dns conv --physics incompressible --solver fv; DeviceFvIncomp::make(T0)"},
    {M::Gravity, FviAxis::Force, "gravity", "dns conv (betaG = g alpha, free-fall scaling)"},
    {M::ChemC0, FviAxis::Chemistry, "C0", "(premixed one-step)", kChem, kChemWhy},
    {M::ChemC1, FviAxis::Chemistry, "C1", "(two reactants)", kChem, kChemWhy},
    {M::ChemC2b, FviAxis::Chemistry, "C2b", "(--equivalence, GRI-3.0 A(phi))", kChem, kChemWhy},
}};

struct Rule
{
    M a, b;
    FvPairRule r;
};

constexpr std::string_view kGate = "gpu_fvi_matrix";

constexpr FvPairRule live(std::string_view gate) { return {.decision = D::Live, .gate = gate}; }

constexpr std::array kRules{
    Rule{M::LesSmag, M::Temperature,
         live("gpu_fvi_matrix_phys P2 (Ra 1e5 3D: Nu +8.7%, +5.6% of the same-grid DNS at 32, 48 cells, shrinking with h)")},
    Rule{M::LesSigma, M::Temperature,
         live("gpu_fvi_matrix (a: sigma on a 2D mesh = molecular bitwise, momentum and T); gpu_fvi_matrix_phys P2 (Nu "
              "+3.4%, +3.1%)")},
    Rule{M::LesSmag, M::Gravity, live("gpu_fvi_matrix_phys P1 (2D Ra 5e4: Nu +1.6%), P2")},
    Rule{M::LesSigma, M::Gravity, live("gpu_fvi_matrix_phys P2 (DNS < sigma < Smagorinsky Nu at both grids)")},
    Rule{M::Temperature, M::Gravity, live("gpu_fv_conv C3-C8, long (FV vs MAC Ra 5e4)")},
};

constexpr FviModuleSet bits(std::initializer_list<M> ms)
{
    FviModuleSet s;
    for (const M m : ms) s.set(std::size_t(m));
    return s;
}

const std::array<FviRequirement, 1> kRequires{{
    {M::Gravity, bits({M::Temperature}), "the Boussinesq force needs the temperature row (dns conv)"},
}};

std::string_view decisionWord(D d) { return d == D::Live ? "live" : d == D::Refused ? "refused" : "exclusive"; }

} // namespace

std::span<const FviModuleInfo> fviModules() { return kModules; }

const FviModuleInfo& fviModuleInfo(FviModule m)
{
    assert(m < FviModule::Count);
    return kModules[std::size_t(m)];
}

FvPairRule fviPair(FviModule a, FviModule b)
{
    assert(a != b && a < FviModule::Count && b < FviModule::Count);
    for (const M m : {a, b})
        if (const FviModuleInfo& i = fviModuleInfo(m); !i.refused.empty())
            return {.decision = D::Refused, .message = i.refused, .why = i.why};
    if (fviModuleInfo(a).axis == fviModuleInfo(b).axis)
        return {.decision = D::Exclusive, .message = "incompressible FV modules: one choice per axis",
                .why = "two choices of one axis"};
    const auto it = std::ranges::find_if(kRules, [&](const Rule& r) { return (r.a == a && r.b == b) || (r.a == b && r.b == a); });
    if (it != kRules.end()) return it->r;
    return live(kGate);
}

std::span<const FviRequirement> fviRequirements() { return kRequires; }

FviModuleSet fviModuleSet(std::initializer_list<FviModule> ms) { return bits(ms); }

std::expected<void, std::string> fviModulesCheck(const FviModuleSet& s)
{
    for (const FviModuleInfo& i : kModules)
        if (s.test(std::size_t(i.id)) && !i.refused.empty()) return std::unexpected(std::string(i.refused));
    for (std::size_t a = 0; a < kFviModuleCount; ++a) {
        if (!s.test(a)) continue;
        for (std::size_t b = a + 1; b < kFviModuleCount; ++b) {
            if (!s.test(b)) continue;
            const FvPairRule r = fviPair(FviModule(a), FviModule(b));
            if (r.decision != D::Live) return std::unexpected(std::string(r.message));
        }
    }
    for (const FviRequirement& q : kRequires)
        if (s.test(std::size_t(q.module)) && (s & q.anyOf).none()) return std::unexpected(std::string(q.message));
    return {};
}

FviModuleSet fviModulesClosure(FviModuleSet s)
{
    for (bool grew = true; grew;) {
        grew = false;
        for (const FviRequirement& q : kRequires) {
            if (!s.test(std::size_t(q.module)) || (s & q.anyOf).any()) continue;
            for (std::size_t m = 0; m < kFviModuleCount; ++m)
                if (q.anyOf.test(m)) {
                    s.set(m);
                    grew = true;
                    break;
                }
        }
    }
    return s;
}

std::string fviModuleSetName(const FviModuleSet& s)
{
    std::string n;
    for (std::size_t m = 0; m < kFviModuleCount; ++m) {
        if (!s.test(m)) continue;
        if (!n.empty()) n += " + ";
        n += kModules[m].name;
    }
    return n.empty() ? std::string("laminar") : n;
}

std::string fviCombinationsMarkdown()
{
    std::string o = "<!-- generated: dns::fviCombinationsMarkdown (src/dns/fvimodules.cpp); test_gpu_fvi_matrix markdown -->\n";
    o += "| Module | Axis | Spelling | Built |\n|---|---|---|---|\n";
    static constexpr std::string_view axes[] = {"turbulence", "scalar", "force", "chemistry"};
    for (const FviModuleInfo& m : kModules)
        o += std::format("| {} | {} | `{}` | {} |\n", m.name, axes[std::size_t(m.axis)], m.spelling,
                         m.refused.empty() ? "yes" : "refused");
    o += "\nPair matrix: L live, R refused, x one axis (exclusive).\n\n|   |";
    for (const FviModuleInfo& m : kModules) o += std::format(" {} |", m.name);
    o += "\n|---|";
    for (std::size_t i = 0; i < kFviModuleCount; ++i) o += "---|";
    o += "\n";
    for (const FviModuleInfo& a : kModules) {
        o += std::format("| {} |", a.name);
        for (const FviModuleInfo& b : kModules) {
            if (a.id == b.id) {
                o += " . |";
                continue;
            }
            const D d = fviPair(a.id, b.id).decision;
            o += d == D::Live ? " L |" : d == D::Refused ? " R |" : " x |";
        }
        o += "\n";
    }
    o += "\n| Refused | Error text | Why |\n|---|---|---|\n";
    for (const FviModuleInfo& m : kModules)
        if (!m.refused.empty()) o += std::format("| {} | `{}` | {} |\n", m.name, m.refused, m.why);
    for (std::size_t a = 0; a < kFviModuleCount; ++a)
        for (std::size_t b = a + 1; b < kFviModuleCount; ++b) {
            if (!kModules[a].refused.empty() || !kModules[b].refused.empty()) continue;
            const FvPairRule r = fviPair(FviModule(a), FviModule(b));
            if (r.decision != D::Refused) continue;
            o += std::format("| {} + {} ({}) | `{}` | {} |\n", kModules[a].name, kModules[b].name,
                             decisionWord(r.decision), r.message, r.why);
        }
    for (const FviRequirement& q : kRequires) {
        std::string need;
        for (std::size_t m = 0; m < kFviModuleCount; ++m)
            if (q.anyOf.test(m)) need += (need.empty() ? "" : " or ") + std::string(kModules[m].name);
        o += std::format("| {} without {} (requirement) | `{}` | |\n", fviModuleInfo(q.module).name, need, q.message);
    }
    o += "\n| Live pair | Hand gates |\n|---|---|\n";
    for (std::size_t a = 0; a < kFviModuleCount; ++a)
        for (std::size_t b = a + 1; b < kFviModuleCount; ++b) {
            const FvPairRule r = fviPair(FviModule(a), FviModule(b));
            if (r.decision == D::Live && !r.gate.empty() && r.gate != kGate)
                o += std::format("| {} + {} | {} |\n", kModules[a].name, kModules[b].name, r.gate);
        }
    return o;
}

} // namespace dns
