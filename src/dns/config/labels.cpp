// The run labels (DNS/LES/RANS, FV, flame, chamber, wick) and their --dry-run helpers.
#include "dns/config.hpp"
#include "dns/config/internal.hpp"

#include "dns/lowmach.hpp"

#include "dns/rans_label.hpp"
#include "dns/tfles.hpp"
#include "dns/transport.hpp"
#include "dns/turb_ic.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <string>

using std::numbers::pi;

namespace dns {
using namespace cfg;
} // namespace dns

namespace dns {

double channelUbEstimate(double reTau)
{
    return 2.5 * std::log(std::max(reTau, 3.0)) + 2.6;
}

double channelPRef(double reTau, double mach)
{
    const double ub = channelUbEstimate(reTau);
    return (ub / mach) * (ub / mach) / kGammaAir;
}

// What FV --shock appends: the background term runs on a silent run too.
inline constexpr const char* kFvShockSuffix =
    " [shock capturing: pressure-sensed two-point HLL (Davis speeds) + background D4 eps {:g} |u.S|: NUMERICAL "
    "dissipation everywhere, not DNS; shock thickness is the mesh's]";

std::string fvCompLabel(const Case& c, bool twoD)
{
    const std::string mesh =
        c.fvMode == FvCompMode::PolyMesh ? std::format("polyMesh {}", c.fvMeshDir)
        : c.ogrid                        ? std::string("body-fitted O-grid via FaceMesh::fromCurvMesh")
        : c.focalActive                  ? std::string("focal box via FaceMesh::fromCurvMesh")
        : c.stretchBeta > 0              ? std::string("stretched box via FaceMesh::fromCurvMesh")
                                         : std::string("uniform box via FaceMesh::fromCurvMesh");
    if (c.rans != RansModel::None) // a mean-model label: no DNS/LES text, no 2D caveat
        return std::format("[FV path, {}, {}{}] compressible {} [exact wall distance, scalars first-order upwind]: "
                           "Kennedy-Gruber face flux, least-squares gradients, SSP-RK3",
                           mesh, fvPrecisionName(c.fvPrecision),
                           c.compensated ? fvCompensatedSuffix(c.fvPrecision, true) : c.compSteady ? " compensated" : "",
                           ransRunLabel(c.rans, c.compSteady ? RansMode::SteadyExplicit : RansMode::Urans));
    const std::string model =
        c.compSteady ? ransRunLabel(RansModel::None, RansMode::SteadyExplicit)
        : c.lesLabel ? lesRunLabel(c.sgsModel, c.cs, twoD)
        : c.shock == ShockMode::Hll ? std::string(c.inviscid ? "Euler" : "laminar")
                   : std::string("DNS") + (twoD ? " (2D: qualitative only, no vortex stretching)" : "");
    const std::string sense = c.shock == ShockMode::Sense ? " [--shock sense: sensor only, the central solver bit for bit]" : "";
    const std::string hll = c.shock == ShockMode::Hll ? std::vformat(kFvShockSuffix, std::make_format_args(c.shockEps4)) : "";
    const std::string gam = c.fvGamma != kGammaAir ? std::format(" gamma {:g}", c.fvGamma) : "";
    return std::format("[FV path, {}, {}{}] compressible{} {}: Kennedy-Gruber face flux, least-squares gradients, "
                       "SSP-RK3{}{}",
                       mesh, fvPrecisionName(c.fvPrecision), fvCompensatedSuffix(c.fvPrecision, c.compensated), gam, model, sense, hll);
}

} // namespace dns

namespace dns::cfg {
namespace {

// M5: the FV routes' --turb (dns/fvturb): what the IC and the inflow rows read.
static std::string fvTurbText(const Case& c)
{
    if (!(c.flameTurb > 0)) return {};
    return "; --turb: the box at the cell centroids (IC) and swept past the xlo faces as per-face inflow rows (FP64 "
           "point sums at the face centres, rewritten every step)";
}

} // namespace
} // namespace dns::cfg

namespace dns {

std::string fvFlameLabel(const Case& c)
{
    const std::string mesh = c.fvMode == FvCompMode::PolyMesh ? std::format("polyMesh {}", c.fvMeshDir)
                                                              : std::string("uniform box via FaceMesh::fromCurvMesh");
    if (c.physics == Physics::LowMach) {
        // The model part of the structured label; the scheme is this path's own.
        const std::string base = flameRunLabel(c);
        const std::size_t bar = base.find(" | ");
        return base.substr(0, base.find(" MAC staggered")) +
               std::format(" FV path, {}, {}: collocated SSP-RK3, mass-flux projection per stage on the geometric "
                           "pressure matrix (AMG-PCG), T and species: MC-limited, ONE limiter shared by the species "
                           "and T, species and rho on the PROJECTED face mass flux (mean flow central)",
                           mesh, c.fvPrecision == FvPrecision::Fp32 ? "fp32, rho and rho Y compensated" : "fp64") +
               fvTurbText(c) + (bar == std::string::npos ? std::string() : base.substr(bar));
    }
    // --shock hll (masterplan_combnozzle P3, Shock x C0): numerical dissipation everywhere, never DNS.
    if (c.shock == ShockMode::Hll)
        return std::format("laminar reacting [premixed single-step C0, calorically perfect gamma {:g}, Le 1, M_flame {:g} "
                           "(inflated)] species: MC-limited MUSCL on the FINAL face mass flux (KG + HLL + D3) | FV path, "
                           "{}, {}{}: Kennedy-Gruber face flux, least-squares gradients, SSP-RK3; open faces are FV "
                           "boundary fluxes, no NSCBC, no sink",
                           c.fvGamma, c.mach, mesh, fvPrecisionName(c.fvPrecision), fvCompensatedSuffix(c.fvPrecision, c.compensated)) +
               std::vformat(kFvShockSuffix, std::make_format_args(c.shockEps4));
    return flameRunLabel(c) +
           std::format(" | FV path, {}, {}{}: Kennedy-Gruber face flux, least-squares gradients, SSP-RK3; species "
                       "virtual-upwind MC {}on the face mass flux; open faces are FV boundary fluxes, no NSCBC, no sink{}",
                       mesh, fvPrecisionName(c.fvPrecision), fvCompensatedSuffix(c.fvPrecision, c.compensated),
                       c.flameDiffusion || c.flamePremix2 ? "shared by the species " : "",
                       c.flameCandle && c.flameGravity > 0
                           ? "; gravity the well-balanced face form, open faces on the hydrostatic ambient's rows "
                             "(p_a at the owner centroid)"
                           : "") +
           fvTurbText(c);
}

namespace {

// What --shock appends to a run label: DNS holds only where sigma = 0.
inline constexpr const char* kShockSuffix =
    " [shock capturing: pressure-sensed two-point HLL (Davis speeds), NUMERICAL "
    "dissipation in sensed cells, shock thickness is the mesh's; central KG elsewhere]";
inline constexpr const char* kShockSenseSuffix = " [--shock sense: sensor only, the central solver bit for bit]";

std::string shockSuffix(const Case& c)
{
    return c.shock == ShockMode::Hll     ? kShockSuffix
           : c.shock == ShockMode::Sense ? kShockSenseSuffix
                                         : "";
}

} // namespace

std::string flowRunLabel(const Case& c)
{
    const bool rans = c.rans != RansModel::None;
    // The wedge's suffix says what it cannot resolve; the full revolution's
    // says its resolved odd-mode trip (and wedge start), which change what a
    // run can converge to, so they are never silent.
    const bool wedge = c.ogridRevolved && c.ogridNphi == 0;
    const bool full = c.ogridRevolved && c.ogridNphi > 0;
    std::string geo;
    if (wedge) geo = ransWedgeSuffix(c.ogridWedgeDeg, rans);
    if (full)
        geo = std::format(
            " [full revolution {}: {}{}]", c.ogridNphi,
            c.ogridTrip > 0
                ? std::format("odd-mode trip {:g} U", c.ogridTrip)
                : std::string("untripped, exactly pi-symmetric"),
            c.ogridInitFrom.empty() ? std::string()
                                    : ", started from the wedge " +
                                          c.ogridInitFrom);
    // A tripped cylinder O-grid starts off the symmetric state: never silent.
    if (c.ogrid && !c.ogridRevolved && c.ogridTrip > 0)
        geo = std::format(" [O-grid trip {:g} U]", c.ogridTrip);
    if (rans || c.compSteady)
    {
        const RansMode mode =
            !c.compSteady ? RansMode::Urans
            : c.steadyScheme == SteadyScheme::Implicit
                ? RansMode::SteadyImplicitNk
                : RansMode::SteadyExplicit;
        // The frozen rotor's steady march (immersed, laminar): the MRF note
        // ahead of the wall's.
        const std::string where =
            rans && !c.ogrid && c.fvMode == FvCompMode::Off ? std::string(kImmersedSuffix)
            : c.rotorMrf && !c.ogrid
                ? std::string(kMrfSteadySuffix) + std::string(kImmersedSuffix)
                : geo;
        std::string l = ransRunLabel(c.rans, mode, 0, where);
        if (wedge && !c.compSteady) l += kAxisymUnsteady;
        if (c.thinSeal) l += kThinSealSuffix;
        if (c.precondK > 0) l += precondSuffix(c.precondK);
        return l;
    }
    if (wedge) return "DNS" + geo + std::string(kAxisymUnsteady);
    if (!geo.empty()) return "DNS" + geo;
    if (c.shock != ShockMode::None)
        return (c.lesLabel ? lesRunLabel(c.sgsModel, c.cs, c.grid.nz == 1) : std::string("DNS")) + shockSuffix(c);
    return {};
}

} // namespace dns

namespace dns::cfg {
namespace {

// " | 2D slot Bunsen: ..." -- the burner and its anchoring, named.
// lz: the span (box units) at nz > 1, 0 = the 2D burner (its text verbatim).
// air: the --equivalence burner (low-Mach, S5): the coflow is air, the pilot
// strips the jet's burnt products.
std::string bunsenLabel(const BunsenGeom& b, double sigma, double lz,
                        bool air = false, bool open = false)
{
    // The y-z box of the round nozzle: periodic, or OPEN faces (--laterals
    // open, low-Mach S6; the S5 text verbatim when periodic).
    const char* box = open ? "open" : "periodic";
    auto kind = [&](double t) {
        return t >= sigma ? "burnt" : t > 1 ? "warm inert" : "cold inert";
    };
    if (air)
        return std::format(
            " | {}: U_jet {:g} S_L, AIR coflow (T/Tu {:g}) at {:g} S_L{}",
            b.round ? std::format("3D round-nozzle Bunsen (diameter D, {} "
                                  "square y-z box {:.3g} D)",
                                  box, lz / b.W)
            : lz > 0 ? std::format("3D slot Bunsen (spanwise periodic L_z={:.3g} "
                                   "W)",
                                   lz / b.W)
                     : std::string("2D slot Bunsen"),
            b.uJet, b.tCo, b.uCo,
            b.pilot > 0 ? std::format(", rich-product pilot strips (T/Tu {:g}) "
                                      "at {:g} S_L",
                                      b.tPilot, b.uPilot)
                        : std::string());
    return std::format(
        " | {}: U_jet {:g} S_L, coflow {} (T/Tu {:g}) at {:g} "
        "S_L{}",
        b.round ? std::format("3D round-nozzle Bunsen (diameter D, {} "
                              "square y-z box {:.3g} D)",
                              box, lz / b.W)
        : lz > 0 ? std::format("3D slot Bunsen (spanwise periodic L_z={:.3g} W)",
                               lz / b.W)
                 : std::string("2D slot Bunsen"),
        b.uJet, kind(b.tCo), b.tCo, b.uCo,
        b.pilot > 0 ? std::format(", {} pilot strips (T/Tu {:g}) at {:g} S_L",
                                  kind(b.tPilot), b.tPilot, b.uPilot)
                    : std::string());
}

} // namespace
} // namespace dns::cfg

namespace dns {

std::string flameRunLabel(const Case& c)
{
    assert(c.sub == Subcommand::Flame);
    if (c.flameDiffusion) {
        const DiffusionModel& m = c.diff;
        const BunsenGeom& b = c.bunsen;
        return std::format(
            "DNS [diffusion single-step F + {:g} O -> {:g} P, calorically "
            "perfect, "
            "Le=1, Y_F1={:g}, Y_O2={:g}, Z_st={:.4g}, T_ad/T0={:g}, Da={:g}, "
            "M_jet={:g}] species: MC-limited MUSCL, ONE limiter shared by the "
            "species (coupling functions exact), on the central mass flux "
            "(mean flow central) | 2D slot jet: Pe=W={:g}, coflow {:g} U_jet{}; "
            "inflow = the Burke-Schumann sheet of its mixture fraction",
            m.s, 1 + m.s, m.yF1, m.yO2, m.zst(), m.sigma, m.Da, c.mach, b.W,
            b.uCo,
            b.pilot > 0 ? std::format(", burnt Z_st pilot strips {:g} at {:g} "
                                      "U_jet",
                                      b.pilot, b.uPilot)
                        : std::string()) +
               (c.flameCandle
                    ? std::format(" | candle: gravity g={:g} U^2/l along -x "
                                  "(Fr=U^2/(gW)={:.3g}), hydrostatic "
                                  "ambient column{}",
                                  c.flameGravity,
                                  c.flameGravity > 0
                                      ? 1.0 / (c.flameGravity * b.W)
                                      : HUGE_VAL,
                                  c.flameQuiescent ? "; QUIESCENT (no fuel, "
                                                     "inflow at rest)"
                                                   : "")
                    : std::string());
    }
    // 3D (nz > 1) and --turb add to the bracket; nz == 1 reads as it did.
    std::string dim, kind = "DNS";
    if (c.grid.nz > 1) {
        dim = std::format("; 3D nz={}", c.grid.nz);
        if (c.flameTurb > 0) {
            const double dth = c.flameEig.dthOverDD;
            const TurbNumbers tn =
                turbNumbers(c.flameTurb, c.flameTurbL, c.prandtl, dth);
            dim += std::format(
                ", synthetic HIT {}inflow u'/S_L={:g}, l_t={:.4g} delta_th "
                "(frozen hash-seeded box{}, Taylor-swept, seed {}; Re_t={:.3g}, "
                "h/eta={:.3g} at the inflow){}",
                c.flameBunsen ? "jet " : "", c.flameTurb, c.flameTurbL / dth,
                c.flameBunsen ? " x jet weight J" : "", c.seed, tn.ReT,
                c.h / tn.eta,
                c.flameHold ? "; flame held by inflow control" : "");
            if (c.h / tn.eta > 2 && !c.lesLabel)
                kind = std::format("UNDER-RESOLVED DNS (h/eta={:.3g})",
                                   c.h / tn.eta);
        }
    }
    // LES and the thickened flame replace "DNS" (docs/chem.md, "LES species
    // and TFLES"): the SGS model with its constant, the species SGS flux,
    // then F, the sensor and the efficiency.
    const std::string les =
        c.lesLabel
            ? std::format("LES [{}={:g}; species SGS flux mu_t/Sc_t, "
                          "Sc_t=Pr_t=0.9{}]",
                          c.sgsModel == SgsModel::Sigma ? "sigma-model Csigma"
                                                        : "Smagorinsky Cs",
                          c.cs,
                          c.fvMode != FvCompMode::Off
                              ? "; mu_t off the least-squares gradient on "
                                "every cell, no SGS flux on boundary faces"
                          : c.physics == Physics::LowMach
                              // M4: the low-Mach MAC solver (gpu/lowmach kLmNut).
                              ? "; T eddy Fourier too, mu_t off central "
                                "differences of the cell-centred MAC "
                                "velocity, no SGS flux on boundary faces"
                              : "; mu_t on the open x rows copied from the "
                                "neighbour row")
            : std::string();
    const std::string tfles =
        c.flameTfles > 0
            ? std::format(
                  "TFLES [F={:g}, {}, {}{}]", c.flameTfles,
                  c.flameTflesUniform
                      ? "uniform sensor (Omega=1)"
                      : std::format("dynamic sensor (rate at Ta/2, tanh {:g}, "
                                    "spread {:g} F delta_th toward colder gas)",
                                    kTflesSensorBeta, kTflesDilate),
                  c.flameCharlette
                      ? std::format("Charlette 2002 efficiency beta={:g} "
                                    "(delta_L=delta_th), u'_Delta by Colin's "
                                    "operator",
                                    kCharletteBeta)
                      : std::string("no efficiency (E=1)"),
                  c.lesLabel ? "" : ", no SGS model")
            : std::string();
    if (!les.empty() && !tfles.empty()) kind = les + " + " + tfles;
    else if (!les.empty()) kind = les;
    else if (!tfles.empty()) kind = "thickened flame " + tfles;
    // sigma and beta are named only when not the model's default, so every
    // default label reads byte for byte as before (TPRI lesgap).
    const PremixedModel defModel{};
    // The Mach item: the compressible solver's inflated flame Mach, or the
    // low-Mach solver's M -> 0 limit (masterplan real_bunsen S4).
    const bool lm = c.physics == Physics::LowMach;
    const std::string machText =
        lm ? std::string("LOW-MACH M->0 limit (zero-Mach variable-density, "
                         "acoustically filtered, P0 constant)")
           : std::format("M_flame={:g} (inflated; CH4/air 1.15e-3)", c.mach);
    const std::string schemeText =
        lm ? std::string("MAC staggered, mass-flux projection per RK3 stage ") +
                 (c.lmRim ? "(apertured FP32 PCG Poisson around the tube, "
                            "spectral preconditioner)"
                          : "(spectral direct Poisson)") +
                 std::string(", temperature predictor; "
                         "species and T: MC-limited MUSCL, ONE limiter shared "
                         "by the species and T, species and rho on the "
                         "PROJECTED mass flux (mean flow central)")
           : std::string();
    // The low-Mach burner's S6 items (empty otherwise: every S5 label byte
    // for byte).
    const std::string lmTail =
        (c.lmOpenY
             ? std::format(" | OPEN laterals ({}): traction-free p=0 faces "
                           "entraining the quiescent ambient (T=Tu, {}), "
                           "backflow-stabilized (face p=-rho u_n^2/2)",
                           c.lmOpenZ ? "y, z" : "y",
                           c.flamePremix2 ? "air" : "inert Y=0")
             : std::string()) +
        (c.lmInert ? std::string(" [--reaction off: INERT jet, no chemistry]")
                   : std::string()) +
        (c.lmGravity > 0
             ? std::format(" | GRAVITY g={:g} S_L^2/delta_D along -x (buoyancy "
                           "(rho - rho_a) g, rho_a the Tu ambient; hydrostatics "
                           "in p)",
                           c.lmGravity)
             : std::string()) +
        (c.lmRim
             ? std::format(" | RIM: immersed isothermal tube T_w={:g}Tu, ID {:g} "
                           "lip {:g} delta_th, {:g} delta_th below the exit "
                           "(STAIRCASE wall on the cell faces: no-slip, "
                           "impermeable, species-tight, first-order wall "
                           "position where it cuts cells); Poiseuille inflow, "
                           "room draft {:g} S_L, no pilot",
                           c.bunsen.tRim, c.bunsen.W / c.flameEig.dthOverDD,
                           c.bunsen.rimLip / c.flameEig.dthOverDD,
                           c.bunsen.xRim / c.flameEig.dthOverDD, c.bunsen.uCo)
             : std::string()) +
        (c.siDeltaD > 0
             ? std::format(" | SI (echo, no physics): delta_D={:.5g} mm, S_L={:.5g} "
                           "m/s, tau={:.4g} ms; D={:.4g} mm, U={:.4g} m/s",
                           1e3 * c.siDeltaD, c.siSL, 1e3 * c.siDeltaD / c.siSL,
                           1e3 * c.siDeltaD * c.bunsen.W, c.siSL * c.bunsen.uJet)
             : std::string());
    if (c.flamePremix2) {
        const PartialPremixModel& pm = c.premix;
        const std::string law =
            c.premixLaw.n > 0
                ? std::format("S_L(phi) law C2b: A C(xi) calibrated to "
                              "GRI-3.0 CH4/air ({} nodes phi {:g}..{:g}, "
                              "C(jet)=1, C(Z_st)={:.4g}), ",
                              c.premixLaw.n, ch4Gri30Speeds().front().phi,
                              ch4Gri30Speeds().back().phi,
                              phiRateC(c.premixLaw, pm, 1.0))
                : std::string("uncorrected one-step S_L(phi) (rises with "
                              "phi rich, CH4 falls: issues.md 73), ");
        const std::string mix =
            c.premixMixPhi > 0
                ? std::format("burning the jet diluted to phi={:g} (S_L "
                              "{:.4g} of the jet's), ",
                              c.premixMixPhi, c.flameEig.tab.m)
                : std::string();
        return std::format(
            "{} [partially premixed two-reactant F + {:g} O -> {:g} P (rich "
            "jet phi={:g}, s={:g}, Y_O2={:g}: Y_F1={:.4g}, Y_O1={:.4g}, "
            "Y_F,rem={:.4g}, Z_st={:.4g}), unit S_L at phi, {}{}calorically "
            "perfect, Le=1, sigma={:g}, Zeldovich beta={:g}, {}{}] {}",
            kind, pm.s, 1 + pm.s, pm.phi, pm.s, pm.yO2, pm.yF1(), pm.yO1(),
            pm.yFrem(), pm.zst(), law, mix, c.flameModel.sigma,
            c.flameModel.beta, machText, dim,
            lm ? schemeText
               : std::string("species: MC-limited MUSCL, ONE limiter shared "
                             "by the species (coupling functions exact), on "
                             "the central mass flux (mean flow central)")) +
               (c.flameBunsen
                    ? bunsenLabel(c.bunsen, c.flameModel.sigma,
                                  c.grid.nz > 1 ? c.grid.nz * c.grid.hz : 0.0,
                                  true, c.lmOpenY) +
                          "; flame index (Takeno) separates premixed and "
                          "diffusion burning"
                    : std::string()) +
               lmTail;
    }
    std::string model;
    if (c.flameModel.sigma != defModel.sigma)
        model += std::format(", sigma={:g}", c.flameModel.sigma);
    if (c.flameModel.beta != defModel.beta)
        model += std::format(", Zeldovich beta={:g}", c.flameModel.beta);
    // --gamma (FV polyMesh C0 only): named only off 1.4, so every other label reads as before.
    const std::string gam = c.fvGamma != kGammaAir
                                ? std::format(", gamma {:g} for fresh AND burnt gas, frozen", c.fvGamma)
                                : std::string();
    return std::format(
        "{} [premixed single-step, calorically perfect{}, Le=1{}, {}{}] {}{}",
        kind, gam, model, machText, dim,
        lm ? schemeText
           : std::string("species: MC-limited MUSCL on the central mass flux "
                         "(bounded scalar transport; mean flow central)"),
        c.flameBunsen
            ? bunsenLabel(c.bunsen, c.flameModel.sigma,
                          c.grid.nz > 1 ? c.grid.nz * c.grid.hz : 0.0, false,
                          c.lmOpenY)
            : std::string()) +
           lmTail + (c.chamberMode ? std::string() : shockSuffix(c));
}

std::string chamberRunLabel(const Case& c)
{
    assert(c.chamberMode);
    return flameRunLabel(c) +
           " chamber: immersed adiabatic ghost wall [first-order wall "
           "position, NOT body-fitted; species flux zero through wall faces], "
           "injector = volumetric mass-flow source band" +
           (c.chamberReaction ? std::string()
                              : std::string(" [--reaction off: inert]")) +
           shockSuffix(c);
}

std::string wickRunLabel(const Case& c)
{
    assert(c.flameWick);
    const DiffusionModel& m = c.diff;
    const WickSpec& s = c.wick;
    return std::format(
        "DNS 2D [diffusion single-step F + {:g} O -> {:g} P, calorically "
        "perfect, Le=1, Y_O2={:g}, Z_st={:.4g}, T_ad/T0={:g}, Da={:g}, "
        "M={:g} (INFLATED: the buoyant plume runs at Mach ~M |u|max)] "
        "species: MC-limited MUSCL, ONE limiter shared by fuel and oxidizer, "
        "on the central mass flux (mean flow central) | WICK CANDLE: gravity "
        "g={:g} U^2/l along -x, hydrostatic ambient column, immersed "
        "isothermal ghost wall [first-order wall position, NOT body-fitted; "
        "wick {:g} x {:g} l at T={:g} T0, body {:g} l wide at T0; species "
        "flux zero through wall faces], fuel = volumetric source band on the "
        "wick surface at the FIXED rate Q={:g} (not tied to the wick heat "
        "flux){}",
        m.s, 1 + m.s, m.yO2, m.zst(), m.sigma, m.Da, c.mach, c.flameGravity,
        s.wWick, s.hWick, s.tWick, s.wCandle, s.Q,
        c.wickSoot
            ? std::format(" | SOOT: one-equation semi-empirical (formation "
                          "Da_f={:g} e^-{:g}T0/T on rho Y_F, oxidation "
                          "Da_ox={:g} e^-{:g}T0/T on rho^2 Y_s Y_O; constants "
                          "TUNED FOR APPEARANCE; no diffusion, no "
                          "thermophoresis, no radiation)",
                          c.soot.Daf, c.soot.thetaF, c.soot.Daox,
                          c.soot.thetaOx)
            : std::string(" | no soot"));
}

} // namespace dns

namespace dns::cfg {

std::string describeWick(const Case& c)
{
    const DiffusionModel& m = c.diff;
    const WickSpec& s = c.wick;
    const double h = c.h, sig = m.sigma;
    const double Lb = c.grid.nx * h;
    const CandleScales cs = candleScales(m, c.flameGravity, s.wWick, Lb, c.mach);
    const double uMax = std::max(c.bunsen.uCo, cs.uFreeFall);
    const double dtAc = c.cfl * h / (std::sqrt(sig) / c.mach + sig * uMax);
    const double dtTh = 0.25 * h * h * (1.0 / sig) /
                        (c.prandtl * std::max(1.0, kGammaAir / c.prandtl));
    const double perim = 2 * s.hWick + s.wWick;
    return std::format(
        "case:\n"
        "  mode       2D WICK CANDLE: an immersed candle body + wick, fuel "
        "vapour from the wick surface, gravity -x (dns flame --mode candle "
        "--wick)\n"
        "  model      single step F + s O -> (1+s) P, omega = A rho^2 Y_F Y_O "
        "e^(-Ta/T); calorically perfect (gamma 1.4), equal W, Le = 1\n"
        "             s = {:g}, Y_O2 = {:g} -> Z_st = {:.6g}; T_ad/T0 = {:.6g}, "
        "beta = {:.6g}, Ta/T0 = {:.6g}, Da = {:.6g} (ln A = {:.6g})\n"
        "  soot       {}\n"
        "  units      rho0 = R = U = 1, l = D0/U = 1 (mu = Pr = {:.4g}), T0 = "
        "p0 = 1/(gamma M^2) = {:.6g}\n"
        "  grid       {} x {} x 1  h = {:.6g} = l/{:.4g}  box {:.6g} x {:.6g}  "
        "[x up, y periodic]\n"
        "  candle     body x in [{:.4g}, {:.4g}] (wax top), {:.4g} l wide, "
        "centred at y = {:.6g}; isothermal at T0\n"
        "  wick       {:.4g} l wide, {:.4g} l above the wax (tip x = {:.4g}); "
        "isothermal at {:.4g} T0 (vaporisation)\n"
        "  fuel       Q = {:.4g} per unit depth (FIXED), Y_F = {:g}, at "
        "{:.4g} T0 through the source band beside the wick (blowing ~{:.3g} "
        "U)\n"
        "  gravity    g = {:.6g} U^2/l along -x; p_a(L)/p0 = {:.6g}; free-fall "
        "speed {:.4g} U (buoyant Mach {:.3g})\n"
        "  air        coflow {:.4g} U from below, {:.4g} l per side beside the "
        "body; y periodic\n"
        "  boundary   x-lo NSCBC air inflow, x-hi outflow p_a(x_out) (anchor "
        "sigma {:g}), both with the hydrostatic waves; air drawn down the top "
        "is ambient; sink {} cells, amp {:.3g}\n"
        "  IC         hydrostatic column; Z = e^(-d/0.5 l) around the wick up "
        "to 4 l above its tip, as its Burke-Schumann state (a closed sheet)\n"
        "  run        tt = {:.6g} l/U  cfl = {:.3g}  dt est {:.3e} (acoustic "
        "{:.3e}, thermal {:.3e}; chemistry at run time)\n"
        "  label      {}\n"
        "  out        {}\n",
        m.s, m.yO2, m.zst(), sig, m.beta, m.theta(), m.Da, m.lnA(),
        c.wickSoot
            ? std::format("one equation, rho Y_s (species 2): formation "
                          "Da_f = {:g} (Ta_f = {:g} T0, ln A_f = {:.6g}), "
                          "oxidation Da_ox = {:g} (Ta_ox = {:g} T0, ln A_ox "
                          "= {:.6g}); appearance-tuned",
                          c.soot.Daf, c.soot.thetaF, c.soot.lnAf(sig),
                          c.soot.Daox, c.soot.thetaOx, c.soot.lnAox(sig))
            : std::string("off"),
        c.prandtl, c.flameTu, c.grid.nx, c.grid.ny, h, c.flameCells,
        c.grid.nx * h, c.grid.ny * h, s.xBase, s.xTop, s.wCandle, s.yc,
        s.wWick, s.hWick, s.xTip(), s.tWick, s.Q, s.yF, s.tWick,
        s.Q * s.tWick / perim, c.flameGravity, 1 - cs.hydroDrop,
        cs.uFreeFall, cs.machBuoyant, c.bunsen.uCo, c.bunsen.wCo,
        kDiffOutflowSigma, c.nscbcSink, c.nscbcSinkAmp, c.tEnd, c.cfl,
        std::min(dtAc, dtTh), dtAc, dtTh, wickRunLabel(c), c.outPrefix);
}

std::string describeChamber(const Case& c)
{
    const ChamberSpec& s = c.chamber;
    const double sig = c.flameModel.sigma, dth = c.flameEig.dthOverDD;
    const double Tu = c.flameTu;
    const auto nz = isentropicNozzle(c.chamberMdot, sig * Tu, s.He, Tu,
                                     kGammaAir, kRgasBox);
    const double cb = std::sqrt(sig) / c.mach;
    const std::string estimate =
        nz ? std::format("  estimate   isentropic exit (T0 = Tb, p_e = p0): Me = {:.4g}, u_e = {:.4g}, p0/p_amb = {:.6g}\n",
                         nz->Me, nz->Me * cb, nz->p0 / Tu)
           : std::string("  estimate   choked: no subsonic isentropic exit (captured, --shock hll)\n");
    return std::format(
        "case:\n"
        "  mode       closed 2D combustion chamber (dns flame --mode chamber)\n"
        "  model      single step F -> P, calorically perfect (gamma 1.4), "
        "equal W, Le = 1\n"
        "             sigma = {:.6g}  beta = {:.6g}  Pr = {:.4g}  Lambda = "
        "{:.10g}  delta_th/delta_D = {:.6g}\n"
        "  units      rho_u = S_L = R = 1, delta_D = 1, Tu = p0 = {:.6g}\n"
        "  grid       {} x {} x 1  h = {:.6g} = delta_th/{:.4g}  box {:.6g} x "
        "{:.6g}  [native 2D, y periodic]\n"
        "  chamber    inner {:.4g} x {:.4g} (back face x = {:.4g}), wall "
        "{:.4g}, nozzle {:.4g} long to exit {:.4g} at x = {:.4g} (ratio "
        "{:.3g})\n"
        "  holder     wedge apex x = {:.4g}, base {:.4g} tall at x = {:.4g}\n"
        "  injector   band x < {:.4g}: premixed Y = 1 at Tu, {:.4g} S_L bulk "
        "(mdot {:.4g} per unit depth), ramp {:.4g}\n"
        "  ambient    cold inert gas (Tu, Y = 0), coflow {:.4g} S_L, x NSCBC "
        "(etaIn {:.4g}), p = p0; sink {} cells, amp {:.3g}\n"
        "  IC         cavity fresh up to the holder base, burnt behind it "
        "(oracle profile), at rest, p0\n"
        "{}"
        "  run        tt = {:.6g} flame times  cfl = {:.3g}\n"
        "  label      {}\n"
        "  out        {}\n",
        sig, c.flameModel.beta, c.prandtl, c.flameEig.Lambda, dth, Tu,
        c.grid.nx, c.grid.ny, c.h, c.flameCells, c.grid.nx * c.h,
        c.grid.ny * c.h, s.Lc + s.Ln, s.Hc, s.xb(), s.tw, s.Ln, s.He, s.xe(),
        s.Hc / s.He, s.xg, s.Hg, s.xBase(), s.xb() + s.wInj, c.flameUIn,
        c.chamberMdot, c.chamberRamp, c.chamberCoflow, c.nscbcEtaIn,
        c.nscbcSink, c.nscbcSinkAmp, estimate, c.ft, c.cfl, chamberRunLabel(c), c.outPrefix);
}

std::string describeLowMachBunsen(const Case& c)
{
    const BunsenGeom& b = c.bunsen;
    const Grid& g = c.grid;
    const double dth = c.flameEig.dthOverDD, h = c.h;
    const bool two = c.flamePremix2;
    const PartialPremixModel& pm = c.premix;
    const double sig = two ? pm.sigma : c.flameModel.sigma;
    // The dt bounds at the burnt state (u = sigma U_jet, rho = 1/sigma,
    // mu(T_b)); the chemistry bound is printed at run time.
    const double muB = c.prandtl * c.flameModel.lambdaRatio(sig);
    const double dtCv = c.cfl * h / (sig * b.uJet);
    const double dtDf = 0.25 * h * h * (1.0 / sig) /
                        (muB * std::max(4.0 / 3.0, 1.0 / c.prandtl));
    const std::string kind = b.round ? "3D round-nozzle Bunsen"
                             : g.nz > 1 ? "3D slot Bunsen (z periodic)"
                                        : "2D slot Bunsen";
    const std::string model =
        two ? std::format(
                  "two-reactant F + {:g} O -> {:g} P (model C2{}), rich jet "
                  "phi = {:g}: Y_F1 = {:.6g}  Y_O1 = {:.6g}  Y_F,rem = {:.6g}\n"
                  "             Z_st = {:.6g}  sigma_in = {:.6g}  sigma_st = "
                  "{:.6g}  Lambda2 = {:.10g}\n",
                  pm.s, 1 + pm.s, c.premixLaw.n > 0 ? "b, S_L(phi) GRI-3.0" : "",
                  pm.phi, pm.yF1(), pm.yO1(), pm.yFrem(), pm.zst(), pm.sigma,
                  pm.sigmaSt(), c.flameEig.Lambda)
            : std::format("single step F -> P, Lambda = {:.10g}\n",
                          c.flameEig.Lambda);
    const std::string jetSp =
        two ? std::format("fresh rich premix Y_F {:.4g}, Y_O {:.4g}", pm.yF1(),
                          pm.yO1())
            : std::string("Y = 1");
    const std::string coSp = two ? std::format("AIR Y_O = {:.4g}", pm.yO2)
                                 : std::string("Y = 0");
    const std::string pilot =
        b.pilot > 0
            ? std::format("\n             pilot strips {:.4g} delta_th per side: "
                          "{}, T = {:.4g} Tu, u = {:.4g} S_L",
                          b.pilot / dth,
                          two ? std::format("the jet's burnt products (Y_F,rem "
                                            "{:.4g}, Y_O 0)",
                                            pm.yFrem())
                              : std::string("Y = 0"),
                          b.tPilot, b.uPilot)
        : c.lmRim
            ? std::format(
                  "\n             TUBE (--rim): lip {:.4g} delta_th ({:.3g} cells), "
                  "ID {:.4g} / OD {:.4g} delta_th, standing on the inflow face "
                  "to x = {:.6g} ({} cells, {:.4g} delta_th), wall T = {:.4g} Tu "
                  "(isothermal); {} solid cells (staircase: cell centre inside)"
                  "\n             jet rows Poiseuille (developed, bulk {:.4g} S_L, "
                  "centre {:.6g} x bulk); the tube's rows closed (m = 0); "
                  "outside the room draft",
                  b.rimLip / dth, b.rimLip / h, b.W / dth, (b.W + 2 * b.rimLip) / dth,
                  b.xRim, int(std::lround(b.xRim / h)), b.xRim / dth, b.tRim,
                  std::ranges::count(lmTubeSolid(b, g), std::uint8_t(1)), b.uJet,
                  std::ranges::max(lmPoiseuille(b, g)))
            : std::string();
    const double alpha = std::asin(1.0 / b.uJet) * 180.0 / std::numbers::pi;
    const bool lat = c.lmOpenY;
    // S7: gravity (only with --gravity > 0: every other dry-run unchanged).
    // Fr = U^2/(g D), Ri = g D (1 - 1/sigma)/U^2 (burnt gas against the
    // ambient), u_ff = sqrt(g D (1 - 1/sigma)), and over the box length L.
    const double gv = c.lmGravity, dRho = 1.0 - 1.0 / sig, Lx = g.nx * h;
    const std::string grav =
        gv > 0 ? std::format(
                     "\n  gravity    g = {:.6g} S_L^2/delta_D along -x (real CH4/air "
                     "phi 1.3 at this delta_th/delta_D: {:.4g}); buoyancy (rho - "
                     "rho_a) g, rho_a = the Tu ambient, hydrostatics in p\n"
                     "             Fr = U^2/(g D) = {:.4g}  Ri = g D (1 - 1/sigma)/U^2 "
                     "= {:.4g}  free-fall u_ff = sqrt(g D (1 - 1/sigma)) = {:.4g} "
                     "S_L (over the box length {:.4g} S_L)",
                     gv, lmRealGravity(dth), b.uJet * b.uJet / (gv * b.W),
                     gv * b.W * dRho / (b.uJet * b.uJet), std::sqrt(gv * b.W * dRho),
                     std::sqrt(gv * Lx * dRho))
               : std::string();
    // S9: the SI echo (--si-delta-d, --si-sl; no physics): lengths x
    // delta_D, speeds x S_L, times x delta_D/S_L; nu = Pr S_L delta_D (mu =
    // Pr at Tu, rho_u = 1); g in m/s^2 = g S_L^2/delta_D.
    const double dD = c.siDeltaD, sl = c.siSL, mm = 1e3 * dD;
    const std::string si =
        dD > 0
            ? std::format(
                  "\n  SI         delta_D = {:.5g} mm, S_L = {:.5g} m/s, tau = "
                  "delta_D/S_L = {:.5g} ms, nu_u = Pr S_L delta_D = {:.4g} m^2/s"
                  "{}\n             D = {:.4g} mm, U = {:.4g} m/s, h = {:.4g} mm, "
                  "box {:.4g} x {:.4g} x {:.4g} mm{}; Re_D = U D/nu_u = {:.4g}; "
                  "tt = {:.4g} ms",
                  mm, sl, 1e3 * dD / sl, c.prandtl * sl * dD,
                  gv > 0 ? std::format(", g = {:.4g} m/s^2", gv * sl * sl / dD)
                         : std::string(),
                  b.W * mm, b.uJet * sl, h * mm, Lx * mm, g.ny * h * mm,
                  g.nz > 1 ? g.nz * h * mm : 0.0,
                  c.lmRim ? std::format(" (above the rim {:.4g} mm; tube ID {:.4g} "
                                        "/ OD {:.4g} mm)",
                                        (Lx - b.xRim) * mm, b.W * mm,
                                        (b.W + 2 * b.rimLip) * mm)
                          : std::string(),
                  b.uJet * b.W / c.prandtl, c.ft * 1e3 * dD / sl)
            : std::string();
    const std::string gravSi = grav + si;
    return std::format(
        "case:\n"
        "  mode       {} (dns flame --mode bunsen --physics lowmach)\n"
        "  model      {}"
        "             calorically perfect, Le = 1, sigma = {:.6g}  beta = "
        "{:.6g}  theta = {:.6g}  Pr = {:.4g}  delta_th/delta_D = {:.6g}\n"
        "  solver     LOW-MACH (M -> 0): P0 constant, rho T = P0/R, MAC "
        "staggered, SSP-RK3, mass-flux projection per stage (SeparablePoisson "
        "x ND, {}, {})\n"
        "  units      rho_u = S_L = R = 1, delta_D = 1, Tu = P0 = 1 (no "
        "acoustic scale), cp = 3.5\n"
        "  transport  {}\n"
        "  grid       {} x {} x {}  h = {:.6g} = delta_th/{:.4g}  box {:.6g} x "
        "{:.6g}{}{}{}\n"
        "  burner     {} {:.4g} delta_th centred at y = {:.6g}{}; jet {}, T = "
        "Tu, u = {:.4g} S_L\n"
        "             coflow {:.4g} delta_th per side (+{} FFT cells), {}, T = "
        "{:.4g} Tu, u = {:.4g} S_L{}\n"
        "             edges tanh half-width {:.4g} delta_th ({:.3g} cells)\n"
        "  cone       ideal sin(alpha) = S_L/U: alpha = {:.4g} deg, height "
        "{:.4g} delta_th; box length {:.4g} delta_th\n"
        "  boundary   x-lo inflow FACE per row (m, T, Y fixed; convective + "
        "diffusive face flux; tangential u = 0); x-hi p = 0, scalars upwind, "
        "no diffusion; {}\n"
        "  stats      {}\n"
        "  frames     temp, fuel{}, omega{}, speed\n"
        "  run        tt = {:.6g} flame times  cfl = {:.3g}  dt est {:.3e} "
        "(convective {:.3e}, diffusive {:.3e}; chemistry --chem-dt {:g} / J "
        "at run time{})\n"
        "  label      {}\n"
        "  out        {}\n",
        kind, model, sig, c.flameModel.beta, c.flameModel.theta(), c.prandtl,
        dth, lat ? (c.lmOpenZ ? "y/z DD (open)" : "y DD (open), z periodic")
                 : "y/z periodic",
        c.lmRim ? "the tube's cells SOLID: apertured FP32 PCG preconditioned by "
                  "that spectral solve, warm-started per stage"
                : "one direct solve",
        c.sutherland ? std::format("Sutherland mu(T), lambda(T), rho D(T), S/Tu "
                                   "= {:.6g} (Tu = {:.6g} K)",
                                   c.flameModel.sOverTu, c.tinfK)
                     : std::string("constant mu = Pr, lambda = cp, rho D = 1"),
        g.nx, g.ny, g.nz, h, c.flameCells, g.nx * h, g.ny * h,
        g.nz > 1 ? std::format(" x {:.6g}", g.nz * h) : std::string(),
        c.lmNxPad > 0 ? std::format("\n             nx rounded up by {} cells "
                                    "downstream (FFT size: nx/2 7-smooth)",
                                    c.lmNxPad)
                      : std::string(),
        c.lmNyPad > 0 ? std::format("\n             ny widened by {} cells "
                                    "(half per side, into the coflow: ny "
                                    "7-smooth)",
                                    c.lmNyPad)
                      : std::string(),
        b.round ? "round nozzle, diameter D =" : "slot W =", b.W / dth, b.yc,
        b.round ? std::format(", z = {:.6g}", b.zc) : std::string(), jetSp,
        b.uJet, b.wCo / dth, c.lmNyPad / 2, coSp, b.tCo, b.uCo, pilot,
        b.edge / dth, b.edge / h, alpha, b.coneH / dth, g.nx * h / dth,
        lat ? std::format(
                  "y{} OPEN (--laterals open): traction-free p = 0 faces, "
                  "m_n from the projection; inflow = the quiescent ambient (T "
                  "= Tu, {}; tangential u = 0, face p = -rho u_n^2/2: the "
                  "backflow stabilization), outflow zero gradient, no "
                  "diffusion; the x-hi backflow takes the ambient too{}",
                  c.lmOpenZ ? ", z" : "", two ? "air" : "Y = 0",
                  c.lmOpenZ || g.nz == 1 ? "" : "; z periodic (the slot's span)") +
                  gravSi
            : std::format("y{} periodic", g.nz > 1 ? ", z" : "") + gravSi,
        b.round ? "the cone on the AXIAL plane z = zc (mean of the two planes "
                  "either side); integrals over the box; ST = burn / (pi R "
                  "hypot(H, R))"
        : g.nz > 1 ? "the cone and every integral on the span (z) mean, per "
                     "unit span; ST = burn / (2 flank)"
                   : "the cone (measureCone) and integrals per unit depth",
        two ? ", oxid" : "", two ? ", findex (Takeno)" : "", c.ft, c.cfl,
        std::min(dtCv, dtDf), dtCv, dtDf, c.lmChemDt,
        c.lmDtCell ? "; --dt-rule cell: and the per-cell positivity bound 1 / "
                     "max (2 C + D)"
                   : "; --dt-rule bounds",
        flameRunLabel(c), c.outPrefix);
}

} // namespace dns::cfg

namespace dns {

double hFine(const Case& c)
{
    return c.focalActive ? hMin(c.grid.nz, c.focalHFine[0], c.focalHFine[1],
                                c.focalHFine[2])
                         : hMin(c.grid);
}

std::string fvConvLabel(const Case& c, bool twoD)
{
    const std::string mesh =
        !c.fvMeshDir.empty() ? std::format("polyMesh {}", c.fvMeshDir)
        : c.fvPerturb == 0   ? std::string("uniform box via FaceMesh::fromCurvMesh")
                             : std::format("{} {:g} box via FaceMesh::fromCurvMesh", c.fvPerturb == 1 ? "sine" : "random",
                                           c.fvPerturbAmp);
    const std::string model = c.physics == Physics::Incompressible ? "Boussinesq, face buoyancy, skew-symmetric T"
                                                                   : "compressible, buoyancy = rho(T) g through the EOS";
    const std::string what =
        c.cs > 0 ? std::format("LES {} {} {:g}, eddy heat flux Pr_t {:g}", c.sgsModel == SgsModel::Sigma ? "sigma" : "Smagorinsky",
                               c.sgsModel == SgsModel::Sigma ? "Csigma" : "Cs", c.cs, kLesPrt)
                 : std::string("DNS");
    return std::format("{}{} [FV path, {}, {}{}] Rayleigh-Benard {}", what, twoD ? " (2D: qualitative only)" : "", mesh,
                       fvPrecisionName(c.fvPrecision), fvCompensatedSuffix(c.fvPrecision, c.compensated), model);
}

} // namespace dns
