#include "dns/fvmodules.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <format>

namespace dns {
namespace {

using M = FvModule;
using D = FvPairDecision;

constexpr std::array<FvModuleInfo, kFvModuleCount> kModules{{
    {M::LesSmag, FvAxis::Turbulence, "LES smag", "--les Cs [--sgs smag]"},
    {M::LesSigma, FvAxis::Turbulence, "LES sigma", "--les C --sgs sigma"},
    {M::TflesUniform, FvAxis::Flame, "TFLES uniform", "--tfles F --tfles-sensor uniform"},
    {M::TflesDynamic, FvAxis::Flame, "TFLES dynamic", "--tfles F [--tfles-sensor dynamic]"},
    {M::TflesEfficiency, FvAxis::Efficiency, "Charlette", "--efficiency charlette"},
    {M::RansSa, FvAxis::Turbulence, "RANS SA", "--rans sa-neg"},
    {M::RansSst, FvAxis::Turbulence, "RANS SST", "--rans sst-2003m, --rans sst-vm"},
    {M::ChemC0, FvAxis::Chemistry, "C0", "dns flame --mode premixed"},
    {M::ChemC1, FvAxis::Chemistry, "C1", "dns flame --mode diffusion; --equivalence PHI --sl-law onestep (C2)"},
    {M::ChemC2b, FvAxis::Chemistry, "C2b", "dns flame --equivalence PHI [--sl-law gri30] [--mix-phi PHI]"},
    {M::Gravity, FvAxis::Gravity, "gravity",
     "dns conv; dns flame --mode candle (FvConfig::gravity; open patches: FvConfig::ambient rows)"},
    {M::BodyForce, FvAxis::Drive, "bodyForce", "the channel's drive (FvConfig::bodyForce)"},
    {M::Turb, FvAxis::Inflow, "turb", "dns flame --turb U --solver fv (FvConfig::turb)"},
    {M::Steady, FvAxis::Time, "steady LTS", "--steady TOL (DeviceFvComp::localDt)"},
    {M::Compensated, FvAxis::State, "compensated", "--compensated"},
    {M::Fp64, FvAxis::Precision, "FP64", "--fv-precision fp64"},
    {M::Shock, FvAxis::Capture, "shock", "--shock hll|sense --polymesh"},
}};

struct Rule
{
    M a, b;
    FvPairRule r;
};

constexpr std::string_view kSpRans = "FV species with RANS: not built (no turbulent scalar flux, two users of rows 5..)";
constexpr std::string_view kSpRansWhy =
    "a RANS flame needs a mean reaction-rate closure (Arrhenius of the mean state is not the mean rate) and a "
    "turbulent scalar flux; neither is built";
constexpr std::string_view kSpLts = "FV local time stepping with species: not built";
constexpr std::string_view kSpLtsWhy = "kFvChem takes the bare cr: under LTS the fixed point would move with lts";
constexpr std::string_view kLesLts = "FV LES with local time stepping: LES is time-accurate (pseudo-time has no filtered field)";
constexpr std::string_view kLesLtsWhy =
    "an LES statistic is a time average of a filtered unsteady field; the structured validator refuses the same pair";
constexpr std::string_view kTfC1 = "FV TFLES: the thickened flame is premixed species only (FvChem ox < 0)";
constexpr std::string_view kTfC1Why = "the thickening and its sensor are the premixed step's (kChemCellPass<true>)";
constexpr std::string_view kTfC2b =
    "FV TFLES with --equivalence: the thickened flame is premixed species only (FvChem ox < 0)";
constexpr std::string_view kTfC2bWhy =
    "the thickening, its sensor (the one-step rate at Ta/2) and Charlette's E are the premixed step's; a two-reactant "
    "TFLES would thicken the partially premixed jet's outer diffusion sheet too, which no model here defines (the "
    "structured --equivalence refuses --tfles alike)";
constexpr std::string_view kTfRans = "FV TFLES with RANS: the thickened flame is a resolved (DNS / LES) flame";
constexpr std::string_view kTfRansWhy = "TFLES thickens a flame the grid carries; a RANS mean flame has none";
constexpr std::string_view kEffRans = "FV TFLES efficiency (Charlette) with RANS: it reads an LES sub-grid velocity";
constexpr std::string_view kEffRansWhy = "Charlette's u' is the filtered field's (Colin's operator), RANS has none";
constexpr std::string_view kLesRans = "FV: LES and RANS put two eddy viscosities in one slot";
constexpr std::string_view kLesRansWhy = "mu_t has one slot (nut); one model owns it";
constexpr std::string_view kGravRans = "FV RANS with gravity: not built (no buoyancy term in the closures)";
constexpr std::string_view kGravRansWhy =
    "the closures carry no buoyancy production: an unstable stratification produces no k or nu~, so a RANS "
    "convection cell (gravity's customer, dns conv) would decay toward laminar. Wrong physics, not a small gap";
constexpr std::string_view kGravLts = "FV local time stepping with gravity: not built";
constexpr std::string_view kGravLtsWhy = "kFvGravity takes the bare cr: under LTS the fixed point would move with lts";
constexpr std::string_view kTurbRans = "FV --turb with RANS: synthetic fluctuations on an ensemble mean";
constexpr std::string_view kTurbRansWhy =
    "a RANS field is the ensemble mean: resolved fluctuations at its inflow are no RANS input (that is k and "
    "omega / nu~ at the patch), and the RANS flame is refused anyway";
constexpr std::string_view kTurbC1 =
    "FV --turb with model C1 / C2 (--mode diffusion, --sl-law onestep): not built (the turbulent box is measured on "
    "C0 and C2b)";
constexpr std::string_view kTurbC1Why =
    "the slot jet takes no --turb on any solver; the one-step C2 premix is C1's stage and has no turbulent number";
constexpr std::string_view kTurbGrav = "FV --turb with gravity: not built (the box's inflow is an open patch)";
constexpr std::string_view kTurbGravWhy =
    "gravity needs a closed domain (no hydrostatic ambient on open patches); --turb needs an open inflow";
constexpr std::string_view kTurbDrive = "FV --turb with a body force: not built (the drive's channel has no inflow)";
constexpr std::string_view kTurbDriveWhy = "the box is swept through an inflow patch; the driven channel is closed";
constexpr std::string_view kTurbLts = "FV --turb with local time stepping: the swept inflow is time-accurate";
constexpr std::string_view kTurbLtsWhy = "the rows follow X = sum U dt; pseudo-time has no physical sweep";
// masterplan_combclose R6 (N16): the plain FP64 stage-3 write ca a + RN(cb b) rounds on a
// lattice (+0.11 ulp per write on mantissas [1, 1.5)); the double-double state is its fix.
constexpr std::string_view kCompFp64 =
    "gpu_fv_matrix (compensated FP64: FvRef 1e-13, numbering, rerun, pow2 Y, mass/energy, the silent-float guard), "
    "gpu_fv_nozzle_react N16b, gpu_restart_fv fp64c";

constexpr std::string_view kShC1 = "FV --shock with two reactants: not built (no gate; docs/nozzle.md \"Reacting\")";
constexpr std::string_view kShC2b = "FV --shock with --equivalence (C2b): not built (no gate; docs/nozzle.md \"Reacting\")";
constexpr std::string_view kShTwoWhy =
    "Shock x C0 is gated (masterplan_combnozzle G3.1-G3.3); the two-reactant step's shared limiter under the hybrid "
    "mass flux is not (P8)";
constexpr std::string_view kShC0 = "gpu_fv_matrix, gpu_fv_matrix_phys (G3.3), fvref_shockspecies (G3.2), gpu_fv_flame_shock";
constexpr std::string_view kShLes = "FV --shock with LES: not built (no gate)";
constexpr std::string_view kShLesWhy = "no nozzle rung runs LES; the background term has no filter width";
constexpr std::string_view kShTfles = "FV --shock with TFLES: not built (no gate)";
constexpr std::string_view kShTflesWhy = "no gate thickens a flame under the hybrid flux (Shock x C0 is gated on the laminar flame)";
constexpr std::string_view kShRans = "FV --shock with RANS: not built (no gate; the scalars ride F[0])";
constexpr std::string_view kShRansWhy = "F[0] would carry the hybrid mass flux under the scalars: ungated";
constexpr std::string_view kShGrav = "FV --shock with gravity: not built (no consumer)";
constexpr std::string_view kShGravWhy = "no rung has both; the hydrostatic balance is unproven under D3";
constexpr std::string_view kShDrive = "FV --shock with a body force: not built (no consumer)";
constexpr std::string_view kShDriveWhy = "the driven channel is subsonic: nothing to capture";
constexpr std::string_view kShTurb = "FV --shock with --turb: not built (no consumer)";
constexpr std::string_view kShTurbWhy = "--turb is the flags' flame box's (--solver fv); --shock is the polyMesh route's";
constexpr std::string_view kShLts = "FV --shock with local time stepping: not built (the sensor flickers; the structured steady march refuses it too)";
constexpr std::string_view kShLtsWhy =
    "a frozen or live sensor both left the host LTS residual flat at 3.7e-3 on a Total inlet (issues.md Z7); the time march is the route";
constexpr std::string_view kShNozzle = "gpu_fv_nozzle";

constexpr FvPairRule refused(std::string_view msg, std::string_view why)
{
    return {.decision = D::Refused, .message = msg, .why = why};
}
constexpr FvPairRule live(std::string_view gate) { return {.decision = D::Live, .gate = gate}; }

constexpr std::string_view kFlameLes = "gpu_fv_flame_les L1-L5, X1";
constexpr std::string_view kSpPrec = "gpu_fv_flame S3, S4";
constexpr std::string_view kRansPrec = "gpu_fv_rans";
constexpr std::string_view kRich = "gpu_fv_flame_rich R1, R2, gpu_fv_flame_rich1d";
constexpr std::string_view kRichLes = "gpu_fv_flame_rich_les (3D laminar LES vs DNS)";
constexpr std::string_view kCandle = "gpu_fv_candle_rest, _g0, _height (the candle vs structured)";
constexpr std::string_view kTurbGate = "gpu_fv_turb (IC, rows, trajectory X), tools/chem/turb_fv.sh comp, pairs (S_c, area vs structured)";

// Explicit rules; every other cross-axis pair is live on gpu_fv_matrix alone.
constexpr std::array kRules{
    // turbulence x flame / efficiency / chemistry
    Rule{M::LesSmag, M::TflesUniform, live(kFlameLes)},
    Rule{M::LesSmag, M::TflesDynamic, live(kFlameLes)},
    Rule{M::LesSigma, M::TflesUniform, live(kFlameLes)},
    Rule{M::LesSigma, M::TflesDynamic, live(kFlameLes)},
    Rule{M::LesSmag, M::TflesEfficiency, live(kFlameLes)},
    Rule{M::LesSigma, M::TflesEfficiency, live(kFlameLes)},
    Rule{M::LesSmag, M::ChemC0, live(kFlameLes)},
    Rule{M::LesSigma, M::ChemC0, live(kFlameLes)},
    Rule{M::LesSmag, M::Steady, refused(kLesLts, kLesLtsWhy)},
    Rule{M::LesSigma, M::Steady, refused(kLesLts, kLesLtsWhy)},
    Rule{M::LesSmag, M::RansSa, {.decision = D::Exclusive, .message = kLesRans, .why = kLesRansWhy}},
    Rule{M::LesSmag, M::RansSst, {.decision = D::Exclusive, .message = kLesRans, .why = kLesRansWhy}},
    Rule{M::LesSigma, M::RansSa, {.decision = D::Exclusive, .message = kLesRans, .why = kLesRansWhy}},
    Rule{M::LesSigma, M::RansSst, {.decision = D::Exclusive, .message = kLesRans, .why = kLesRansWhy}},
    Rule{M::TflesUniform, M::RansSa, refused(kTfRans, kTfRansWhy)},
    Rule{M::TflesUniform, M::RansSst, refused(kTfRans, kTfRansWhy)},
    Rule{M::TflesDynamic, M::RansSa, refused(kTfRans, kTfRansWhy)},
    Rule{M::TflesDynamic, M::RansSst, refused(kTfRans, kTfRansWhy)},
    Rule{M::TflesEfficiency, M::RansSa, refused(kEffRans, kEffRansWhy)},
    Rule{M::TflesEfficiency, M::RansSst, refused(kEffRans, kEffRansWhy)},
    Rule{M::RansSa, M::ChemC0, refused(kSpRans, kSpRansWhy)},
    Rule{M::RansSa, M::ChemC1, refused(kSpRans, kSpRansWhy)},
    Rule{M::RansSst, M::ChemC0, refused(kSpRans, kSpRansWhy)},
    Rule{M::RansSst, M::ChemC1, refused(kSpRans, kSpRansWhy)},
    Rule{M::RansSa, M::ChemC2b, refused(kSpRans, kSpRansWhy)},
    Rule{M::RansSst, M::ChemC2b, refused(kSpRans, kSpRansWhy)},
    Rule{M::LesSmag, M::ChemC2b, live(kRichLes)},
    Rule{M::LesSigma, M::ChemC2b, live(kRichLes)},
    Rule{M::RansSa, M::Gravity, refused(kGravRans, kGravRansWhy)},
    Rule{M::RansSst, M::Gravity, refused(kGravRans, kGravRansWhy)},
    Rule{M::RansSa, M::BodyForce, live("gpu_fv_rans R8 (the channel)")},
    Rule{M::RansSst, M::BodyForce, live("gpu_fv_rans R8 (the channel)")},
    Rule{M::RansSa, M::Steady, live(kRansPrec)},
    Rule{M::RansSst, M::Steady, live(kRansPrec)},
    Rule{M::RansSa, M::Compensated, live(kRansPrec)},
    Rule{M::RansSst, M::Compensated, live(kRansPrec)},
    Rule{M::RansSa, M::Fp64, live(kRansPrec)},
    Rule{M::RansSst, M::Fp64, live(kRansPrec)},
    Rule{M::LesSmag, M::Fp64, live("gpu_fv_cross X2")},
    Rule{M::LesSigma, M::Fp64, live("gpu_fv_cross X2")},
    // flame / efficiency
    Rule{M::TflesUniform, M::TflesEfficiency, live(kFlameLes)},
    Rule{M::TflesDynamic, M::TflesEfficiency, live(kFlameLes)},
    Rule{M::TflesUniform, M::ChemC0, live(kFlameLes)},
    Rule{M::TflesDynamic, M::ChemC0, live(kFlameLes)},
    Rule{M::TflesEfficiency, M::ChemC0, live(kFlameLes)},
    Rule{M::TflesUniform, M::ChemC1, refused(kTfC1, kTfC1Why)},
    Rule{M::TflesDynamic, M::ChemC1, refused(kTfC1, kTfC1Why)},
    Rule{M::TflesEfficiency, M::ChemC1, refused(kTfC1, kTfC1Why)},
    Rule{M::TflesUniform, M::ChemC2b, refused(kTfC2b, kTfC2bWhy)},
    Rule{M::TflesDynamic, M::ChemC2b, refused(kTfC2b, kTfC2bWhy)},
    Rule{M::TflesEfficiency, M::ChemC2b, refused(kTfC2b, kTfC2bWhy)},
    Rule{M::TflesUniform, M::Steady, refused(kSpLts, kSpLtsWhy)},
    Rule{M::TflesDynamic, M::Steady, refused(kSpLts, kSpLtsWhy)},
    Rule{M::TflesEfficiency, M::Steady, refused(kSpLts, kSpLtsWhy)},
    // chemistry
    Rule{M::ChemC0, M::Steady, refused(kSpLts, kSpLtsWhy)},
    Rule{M::ChemC1, M::Steady, refused(kSpLts, kSpLtsWhy)},
    Rule{M::ChemC2b, M::Steady, refused(kSpLts, kSpLtsWhy)},
    Rule{M::ChemC0, M::Compensated, live(kSpPrec)},
    Rule{M::ChemC1, M::Compensated, live("gpu_fv_flame_jet")},
    Rule{M::ChemC0, M::Fp64, live(kSpPrec)},
    Rule{M::ChemC1, M::Fp64, live("gpu_fv_flame_jet")},
    Rule{M::ChemC2b, M::Compensated, live(kRich)},
    Rule{M::ChemC2b, M::Fp64, live(kRich)},
    // gravity
    Rule{M::Gravity, M::Steady, refused(kGravLts, kGravLtsWhy)},
    Rule{M::Gravity, M::ChemC1, live(kCandle)},
    Rule{M::Gravity, M::Compensated, live("gpu_fv_conv")},
    Rule{M::Gravity, M::Fp64, live("gpu_fv_conv")},
    Rule{M::BodyForce, M::Steady, live(kRansPrec)},
    // turb (M5): the IC and the per-face rows, no stage term
    Rule{M::Turb, M::RansSa, refused(kTurbRans, kTurbRansWhy)},
    Rule{M::Turb, M::RansSst, refused(kTurbRans, kTurbRansWhy)},
    Rule{M::Turb, M::ChemC1, refused(kTurbC1, kTurbC1Why)},
    Rule{M::Turb, M::Gravity, refused(kTurbGrav, kTurbGravWhy)},
    Rule{M::Turb, M::BodyForce, refused(kTurbDrive, kTurbDriveWhy)},
    Rule{M::Turb, M::Steady, refused(kTurbLts, kTurbLtsWhy)},
    Rule{M::Turb, M::ChemC0, live(kTurbGate)},
    Rule{M::Turb, M::ChemC2b, live(kTurbGate)},
    Rule{M::Turb, M::LesSmag, live(kTurbGate)},
    Rule{M::Turb, M::LesSigma, live(kTurbGate)},
    Rule{M::Turb, M::TflesUniform, live(kTurbGate)},
    Rule{M::Turb, M::TflesDynamic, live(kTurbGate)},
    Rule{M::Turb, M::TflesEfficiency, live(kTurbGate)},
    Rule{M::Turb, M::Compensated, live(kTurbGate)},
    Rule{M::Turb, M::Fp64, live(kTurbGate)},
    Rule{M::Compensated, M::Fp64, live(kCompFp64)},
    // shock capture: every refusal written (the default of a cross-axis pair is live)
    Rule{M::Shock, M::Fp64, live(kShNozzle)},
    Rule{M::Shock, M::Compensated, live(kShNozzle)},
    Rule{M::Shock, M::ChemC0, live(kShC0)},
    Rule{M::Shock, M::ChemC1, refused(kShC1, kShTwoWhy)},
    Rule{M::Shock, M::ChemC2b, refused(kShC2b, kShTwoWhy)},
    Rule{M::Shock, M::LesSmag, refused(kShLes, kShLesWhy)},
    Rule{M::Shock, M::LesSigma, refused(kShLes, kShLesWhy)},
    Rule{M::Shock, M::TflesUniform, refused(kShTfles, kShTflesWhy)},
    Rule{M::Shock, M::TflesDynamic, refused(kShTfles, kShTflesWhy)},
    Rule{M::Shock, M::TflesEfficiency, refused(kShTfles, kShTflesWhy)},
    Rule{M::Shock, M::RansSa, refused(kShRans, kShRansWhy)},
    Rule{M::Shock, M::RansSst, refused(kShRans, kShRansWhy)},
    Rule{M::Shock, M::Gravity, refused(kShGrav, kShGravWhy)},
    Rule{M::Shock, M::BodyForce, refused(kShDrive, kShDriveWhy)},
    Rule{M::Shock, M::Turb, refused(kShTurb, kShTurbWhy)},
    Rule{M::Shock, M::Steady, refused(kShLts, kShLtsWhy)},
};

// Pre-built bit sets (constexpr bitset construction from an integer).
constexpr unsigned long long bit(M m) { return 1ull << unsigned(m); }

constexpr std::array<FvRequirement, 5> kRequires{{
    {M::Turb, FvModuleSet(bit(M::ChemC0) | bit(M::ChemC2b)),
     "FV --turb: the premixed flame box (C0, C2b; dns flame --mode premixed --solver fv)"},
    {M::TflesUniform, FvModuleSet(bit(M::ChemC0)), kTfC1},
    {M::TflesDynamic, FvModuleSet(bit(M::ChemC0)), kTfC1},
    {M::TflesEfficiency, FvModuleSet(bit(M::TflesUniform) | bit(M::TflesDynamic)),
     "FV TFLES efficiency (Charlette) is an option of the thickened flame: needs TFLES"},
    {M::TflesEfficiency, FvModuleSet(bit(M::LesSmag) | bit(M::LesSigma)),
     "FV TFLES efficiency (Charlette) needs LES: the sub-grid velocity of a filtered field"},
}};

std::string_view decisionWord(D d)
{
    switch (d) {
    case D::Live: return "live";
    case D::Refused: return "refused";
    case D::Exclusive: return "exclusive";
    }
    return "";
}

} // namespace

std::span<const FvModuleInfo> fvModules() { return kModules; }

const FvModuleInfo& fvModuleInfo(FvModule m)
{
    assert(m < FvModule::Count);
    return kModules[std::size_t(m)];
}

FvPairRule fvPair(FvModule a, FvModule b)
{
    assert(a != b && a < FvModule::Count && b < FvModule::Count);
    const auto it = std::ranges::find_if(kRules, [&](const Rule& r) { return (r.a == a && r.b == b) || (r.a == b && r.b == a); });
    if (it != kRules.end()) return it->r;
    if (fvModuleInfo(a).axis == fvModuleInfo(b).axis)
        return {.decision = D::Exclusive, .message = "FV modules: one choice per axis", .why = "two choices of one axis"};
    return live({});
}

std::span<const FvRequirement> fvRequirements() { return kRequires; }

FvModuleSet fvModuleSet(std::initializer_list<FvModule> ms)
{
    FvModuleSet s;
    for (const FvModule m : ms) s.set(std::size_t(m));
    return s;
}

std::expected<void, std::string> fvModulesCheck(const FvModuleSet& s)
{
    for (std::size_t a = 0; a < kFvModuleCount; ++a) {
        if (!s.test(a)) continue;
        for (std::size_t b = a + 1; b < kFvModuleCount; ++b) {
            if (!s.test(b)) continue;
            const FvPairRule r = fvPair(FvModule(a), FvModule(b));
            if (r.decision != D::Live) return std::unexpected(std::string(r.message));
        }
    }
    for (const FvRequirement& q : kRequires)
        if (s.test(std::size_t(q.module)) && (s & q.anyOf).none()) return std::unexpected(std::string(q.message));
    return {};
}

FvModuleSet fvModulesClosure(FvModuleSet s)
{
    for (bool grew = true; grew;) {
        grew = false;
        for (const FvRequirement& q : kRequires) {
            if (!s.test(std::size_t(q.module)) || (s & q.anyOf).any()) continue;
            for (std::size_t m = 0; m < kFvModuleCount; ++m)
                if (q.anyOf.test(m)) {
                    s.set(m);
                    grew = true;
                    break;
                }
        }
    }
    return s;
}

std::string fvModuleSetName(const FvModuleSet& s)
{
    std::string n;
    for (std::size_t m = 0; m < kFvModuleCount; ++m) {
        if (!s.test(m)) continue;
        if (!n.empty()) n += " + ";
        n += kModules[m].name;
    }
    return n.empty() ? std::string("laminar") : n;
}

std::string fvCombinationsMarkdown()
{
    std::string o = "<!-- generated: dns::fvCombinationsMarkdown (src/dns/fvmodules.cpp); test_gpu_fv_matrix markdown -->\n";
    o += "| Module | Axis | Spelling |\n|---|---|---|\n";
    static constexpr std::string_view axes[] = {"turbulence", "flame", "efficiency", "chemistry", "gravity", "drive",
                                                "inflow",     "time",  "precision",  "capture",   "state"};
    for (const FvModuleInfo& m : kModules)
        o += std::format("| {} | {} | `{}` |\n", m.name, axes[std::size_t(m.axis)], m.spelling);
    o += "\nPair matrix: L live, R refused, x one axis (exclusive).\n\n|   |";
    for (const FvModuleInfo& m : kModules) o += std::format(" {} |", m.name);
    o += "\n|---|";
    for (std::size_t i = 0; i < kFvModuleCount; ++i) o += "---|";
    o += "\n";
    for (const FvModuleInfo& a : kModules) {
        o += std::format("| {} |", a.name);
        for (const FvModuleInfo& b : kModules) {
            if (a.id == b.id) {
                o += " . |";
                continue;
            }
            const D d = fvPair(a.id, b.id).decision;
            o += d == D::Live ? " L |" : d == D::Refused ? " R |" : " x |";
        }
        o += "\n";
    }
    o += "\n| Pair | Decision | Error text | Why |\n|---|---|---|---|\n";
    for (std::size_t a = 0; a < kFvModuleCount; ++a)
        for (std::size_t b = a + 1; b < kFvModuleCount; ++b) {
            const FvPairRule r = fvPair(FvModule(a), FvModule(b));
            if (r.decision == D::Live || r.message == "FV modules: one choice per axis") continue;
            o += std::format("| {} + {} | {} | `{}` | {} |\n", kModules[a].name, kModules[b].name, decisionWord(r.decision),
                             r.message, r.why);
        }
    for (const FvRequirement& q : kRequires) {
        std::string need;
        for (std::size_t m = 0; m < kFvModuleCount; ++m)
            if (q.anyOf.test(m)) need += (need.empty() ? "" : " or ") + std::string(kModules[m].name);
        o += std::format("| {} without {} | requirement | `{}` | |\n", fvModuleInfo(q.module).name, need, q.message);
    }
    o += "\n| Live pair | Hand gates besides gpu_fv_matrix |\n|---|---|\n";
    for (std::size_t a = 0; a < kFvModuleCount; ++a)
        for (std::size_t b = a + 1; b < kFvModuleCount; ++b) {
            const FvPairRule r = fvPair(FvModule(a), FvModule(b));
            if (r.decision == D::Live && !r.gate.empty())
                o += std::format("| {} + {} | {} |\n", kModules[a].name, kModules[b].name, r.gate);
        }
    return o;
}

} // namespace dns
