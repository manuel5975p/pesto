// validate(): dns flame --mode chamber, the bunsen geometry, --wick, diffusion / candle.
#include "dns/config.hpp"
#include "dns/config/internal.hpp"

#include "dns/lowmach.hpp"

#include "dns/transport.hpp"
#include "dns/turb_ic.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <optional>
#include <string>
#include <string_view>

namespace dns::cfg {

std::expected<Case, ConfigError> finishChamber(const RawConfig& r, Case c)
{
    const auto& f = r.chamber;
    const double cells =
        r.given.contains("--cells-per-delta") ? r.cellsPerDelta : 4.0;
    const double down = r.given.contains("--downstream") ? r.downstream : 40.0;
    if (!(down >= 10))
        return bad(2, "--downstream", "must be >= 10 thermal thicknesses "
                                      "(the jet region past the nozzle)");
    if (!(finitePos(f.injectSpeed) && f.injectSpeed <= 20))
        return bad(2, "--inject-speed", "must be in (0, 20] (x S_L)");
    if (!(finitePos(f.coflow) && f.coflow <= 20))
        return bad(2, "--coflow", "must be in (0, 20] (x S_L): a static "
                                  "ambient makes the NSCBC inflow an acoustic "
                                  "mirror");
    if (!(std::isfinite(f.height) && f.height >= 8))
        return bad(2, "--chamber-height", "must be >= 8 thermal thicknesses");
    if (!(std::isfinite(f.length) && f.length >= 10))
        return bad(2, "--chamber-length", "must be >= 10 thermal thicknesses");
    if (!(f.nozzleRatio >= 1.5 && f.nozzleRatio <= 6))
        return bad(2, "--nozzle-ratio", "must be in [1.5, 6] (Hc / He)");
    if (!(f.holder >= 1 && f.holder <= f.height - 4))
        return bad(2, "--holder-height", "must be in [1, chamber height - 4] "
                                         "thermal thicknesses");
    if (!(std::isfinite(f.clearance) && f.clearance >= 4))
        return bad(2, "--clearance", "must be >= 4 thermal thicknesses");
    if (!(std::isfinite(f.ramp) && f.ramp >= 0))
        return bad(2, "--inject-ramp", "must be finite and >= 0");

    const double dth = c.flameEig.dthOverDD;
    const double h = dth / cells;
    const int nb = c.nscbcSink;
    ChamberSpec s;
    s.tw = 1.5 * dth;
    s.Hc = f.height * dth;
    s.Lc = f.length * dth;
    s.He = s.Hc / f.nozzleRatio;
    s.Ln = 0.5 * s.Hc;
    s.Hg = f.holder * dth;
    s.Lg = 0.75 * s.Hg;
    s.wInj = 1.25 * dth;
    s.x0 = nb * h + 4 * dth;
    s.xg = s.xb() + s.wInj + 2.5 * dth;
    if (s.xBase() > s.xn() - 2 * dth)
        return bad(2, "--chamber-length", "too short for the injector, the "
                                          "holder and 2 thermal thicknesses "
                                          "behind it");
    if (s.He / h < 8)
        return bad(2, "--nozzle-ratio", std::format(
                       "leaves the nozzle exit {:.3g} cells tall (< 8): raise "
                       "--cells-per-delta or --chamber-height",
                       s.He / h));
    const int nx = int(std::ceil((s.xe() + down * dth) / h - 1e-9)) + nb;
    const int ny =
        int(std::ceil((s.Hc + 2 * s.tw + 2 * f.clearance * dth) / h - 1e-9));
    auto box = makeBox(nx, ny, 1, nx * h, ny * h, 1.0);
    if (!box) return std::unexpected(box.error());
    s.yc = 0.5 * ny * h;
    c.grid = *box;
    c.n = nx;
    c.h = h;
    c.L = nx * h;
    c.flameCells = cells;
    c.flameDown = down;
    c.flameTu = 1.0 / (kGammaAir * c.mach * c.mach);
    c.chamberMode = true;
    c.chamber = s;
    c.chamberMdot = f.injectSpeed * s.Hc; // rho_u = 1
    c.chamberCoflow = f.coflow;
    c.chamberRamp = f.ramp;
    c.chamberReaction = f.reaction == 1;
    c.flameUIn = f.injectSpeed;
    c.uInf = f.coflow;
    c.re = 1.0 / c.prandtl;
    c.nu = c.prandtl;
    c.far = FarField::Nscbc;
    c.bodyKind = BodyKind::None;

    // The nozzle at the burnt total temperature, exit at the ambient p0: a
    // choked estimate has no subsonic solution and the central flux has no
    // shock capturing.
    const double Tu = c.flameTu, sig = c.flameModel.sigma;
    auto nz = isentropicNozzle(c.chamberMdot, sig * Tu, s.He, Tu, kGammaAir,
                               kRgasBox);
    if (!nz)
        return bad(2, "--inject-speed", std::format(
                       "chokes the nozzle ({}): lower --inject-speed or "
                       "--nozzle-ratio, or raise --mach",
                       nz.error()));
    if (nz->Me > 0.7)
        c.warnings.push_back(std::format(
            "the isentropic exit Mach is {:.3g} (> 0.7): the central flux has "
            "no shock capturing", nz->Me));

    if (auto e = resolveCfl(r, c, 0.4); e) return std::unexpected(*e);
    if (!finitePos(r.tt)) return bad(2, "--tt", "must be finite and > 0");
    c.ft = r.given.contains("--tt") ? r.tt : 60.0;
    c.tEnd = c.ft;
    if (auto e = resolveDtRecompute(r, c, 16); e) return std::unexpected(*e);
    if (auto e = validateFrameOutputs(r, c); e) return std::unexpected(*e);
    if (auto e = resolveEmit(c, {Emit::Stats}); e) return std::unexpected(*e);
    if (c.mach > 0.02)
        c.warnings.push_back(std::format(
            "--mach {:.4g} is above the measured ladder top 0.02", c.mach));
    if (nb == 0)
        c.warnings.push_back("--nscbc-sink 0: the bare characteristic "
                             "boundary (nothing drains mesh-scale content)");
    if (!c.chamberReaction)
        c.warnings.push_back("--reaction off: inert species transport (the "
                             "wall-impermeability check), no flame");
    c.meta = {.D = s.Hc,
              .re = c.re,
              .mach = c.mach,
              .frameDt = c.fdt,
              .body = "combustion chamber",
              .farField = "nscbc",
              .physics = name(c.physics),
              .kick = 0.0,
              .sgs = "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return c;
}

std::expected<void, ConfigError> bunsenGeometry(const RawConfig& r, Case& c,
                                                int nb)
{
    const double dth = c.flameEig.dthOverDD, cells = c.flameCells;
    const double h = dth / cells, sig = c.flameModel.sigma;
    if (!(std::isfinite(r.slotWidth) && r.slotWidth >= 4))
        return bad(2, "--slot-width", "must be >= 4 thermal thicknesses (a "
                                      "cone needs a straight flank)");
    if (!(std::isfinite(r.jetSpeed) && r.jetSpeed >= 1.5 && r.jetSpeed <= 20))
        return bad(2, "--jet-speed", "must be in [1.5, 20] S_L (a cone "
                                     "needs U > S_L; above 20 it outgrows "
                                     "any affordable box)");
    if (r.jetSpeed * c.mach > 0.3)
        return bad(2, "--jet-speed", std::format(
                                         "jet Mach {:.3g} = U M exceeds 0.3: "
                                         "lower --mach or the jet speed",
                                         r.jetSpeed * c.mach));
    // --rim (S8; the validator admitted it on the low-Mach burner only): the
    // coflow is the ROOM DRAFT (D15: default 0.4 S_L), cold room air.
    const bool rim = r.given.contains("--rim");
    const double uco = rim && !r.given.contains("--coflow-speed") ? kLmRoomDraft
                                                                   : r.coflowSpeed;
    if (!(std::isfinite(uco) && uco >= 0 && uco <= r.jetSpeed))
        return bad(2, "--coflow-speed", "must be in [0, --jet-speed] S_L");
    const double wco = r.coflowWidth < 0 ? r.slotWidth : r.coflowWidth;
    if (!(std::isfinite(wco) && wco >= 2))
        return bad(2, "--coflow-width", "must be >= 2 thermal thicknesses per "
                                        "side (y is periodic: the coflow "
                                        "separates the jet from its image)");
    // --equivalence (low-Mach, S5): the coflow is AIR, cold by default; the
    // single-reactant burner's default is the burnt coflow (the rim's: cold).
    const double tco = rim ? 1.0
                       : r.coflowTemp < 0 && !r.given.contains("--coflow-temp")
                           ? (c.flamePremix2 ? 1.0 : sig)
                           : r.coflowTemp;
    double lip = 0, tube = 0;
    if (rim) {
        if (!(std::isfinite(r.rim) && r.rim * cells >= 4 - 1e-9))
            return bad(2, "--rim", std::format(
                                       "must be >= 4 cells ({:.4g} thermal "
                                       "thicknesses at --cells-per-delta {:g}): "
                                       "the lip thickness, two cells per side",
                                       4.0 / cells, cells));
        tube = r.tubeLength < 0 ? kLmTubeLengthDefault : r.tubeLength;
        if (!(std::isfinite(tube) && tube * cells >= 8 - 1e-9))
            return bad(2, "--tube-length",
                       std::format("must be >= 8 cells ({:.4g} thermal thicknesses "
                                   "at --cells-per-delta {:g}): the developed "
                                   "tube flow below the exit",
                                   8.0 / cells, cells));
        if (!(std::isfinite(r.rimTemp) && r.rimTemp >= 1 && r.rimTemp <= 3))
            return bad(2, "--rim-temp", "must be in [1, 3] (T_wall/Tu)");
        lip = r.rim;
    }
    if (!(std::isfinite(tco) && tco >= 1 && tco <= sig))
        return bad(2, "--coflow-temp", std::format(
                                           "must be in [1, --expansion = {:g}] "
                                           "(T/Tu: 1 cold inert, sigma the "
                                           "burnt pilot)",
                                           sig));
    if (!(std::isfinite(r.pilotWidth) && r.pilotWidth >= 0))
        return bad(2, "--pilot-width", "must be finite and >= 0 thermal "
                                       "thicknesses");
    if (!(r.pilotWidth > 0))
        if (auto e = rejectGiven(r, {"--pilot-temp", "--pilot-speed"},
                                 "needs --pilot-width > 0");
            e)
            return std::unexpected(*e);
    const double tp = r.pilotTemp < 0 && !r.given.contains("--pilot-temp")
                          ? sig
                          : r.pilotTemp;
    if (!(std::isfinite(tp) && tp >= 1 && tp <= sig))
        return bad(2, "--pilot-temp", std::format(
                                          "must be in [1, --expansion = {:g}] "
                                          "(T/Tu)",
                                          sig));
    const double up = r.pilotSpeed < 0 && !r.given.contains("--pilot-speed")
                          ? uco
                          : r.pilotSpeed;
    if (!(std::isfinite(up) && up >= 0 && up <= r.jetSpeed))
        return bad(2, "--pilot-speed", "must be in [0, --jet-speed] S_L");
    if (!(std::isfinite(r.slotEdge) && r.slotEdge * cells >= 1))
        return bad(2, "--slot-edge", std::format(
                                         "must be >= one cell ({:.4g} "
                                         "thermal thicknesses at "
                                         "--cells-per-delta {:g})",
                                         1.0 / cells, cells));
    // --turb: the cone of Damkoehler's S_T = S_L (1 + u'/S_L) -- an UPPER
    // estimate of S_T (a shorter cone), so the IC and the --length bound are
    // a floor; the run measures its own tip (stats xTip, exit 8 past the
    // outflow band). Without --turb the laminar cone, as before.
    const double sT = r.turb > 0 ? damkohlerST(r.turb) : 1.0;
    if (r.turb > 0 && !(r.jetSpeed > 1.25 * sT))
        return bad(2, "--turb", std::format(
                                    "the turbulent flame speed estimate S_T = "
                                    "S_L (1 + u'/S_L) = {:.4g} leaves no cone "
                                    "at --jet-speed {:g} (needs U_jet > 1.25 "
                                    "S_T): lower --turb or raise --jet-speed",
                                    sT, r.jetSpeed));
    const double coneH =
        r.turb > 0
            ? 0.5 * r.slotWidth *
                  std::sqrt((r.jetSpeed / sT) * (r.jetSpeed / sT) - 1.0)
            : 0.5 * r.slotWidth *
                  std::sqrt(r.jetSpeed * r.jetSpeed - 1.0); // delta_th
    const double len = r.flameLength < 0 ? std::ceil(1.25 * coneH + 8)
                                         : r.flameLength;
    if (!(std::isfinite(len) && len >= coneH + 4))
        return bad(2, "--length", std::format(
                                      "must be >= the {} cone height + 4 = "
                                      "{:.4g} thermal thicknesses (the tip "
                                      "and its burnt gas must fit)",
                                      r.turb > 0 ? "turbulent (Damkoehler "
                                                   "S_T)"
                                                 : "ideal",
                                      coneH + 4));
    if (!(std::isfinite(r.pulse) && r.pulse >= 0 && r.pulse <= 0.5))
        return bad(2, "--pulse", "must be in [0, 0.5] (fraction of the jet "
                                 "speed)");
    if (r.pulse > 0 && !(std::isfinite(r.pulseFreq) && r.pulseFreq > 0))
        return bad(2, "--pulse-freq", "must be finite and > 0 with --pulse "
                                      "(1/tau)");
    if (r.given.contains("--pulse-freq") && !(r.pulse > 0))
        return bad(2, "--pulse-freq", "needs --pulse > 0");

    int half = int(std::ceil((0.5 * r.slotWidth + r.pilotWidth + lip + wco) *
                             cells - 1e-9));
    // --rim: the tube stands on the inflow face below the exit; --length is
    // the box ABOVE the rim. The rim top snaps to a cell face (the staircase).
    const int nTube = rim ? int(std::lround(tube * cells)) : 0;
    int nx = int(std::ceil(len * cells - 1e-9)) + 2 * nb + nTube;
    if (c.physics == Physics::LowMach) {
        // The pressure solve's FFT sizes (K11): ny = 2 half 7-smooth (the
        // cells widen the coflow, both sides), nx/2 7-smooth (cells added
        // downstream).
        const int half0 = half, nx0 = nx;
        while (!lmFftSmooth(2 * half)) ++half;
        nx = 2 * lmNextSmooth((nx + 1) / 2);
        c.lmNyPad = 2 * (half - half0);
        c.lmNxPad = nx - nx0;
    }
    const int ny = 2 * half;
    // --nozzle round: the jet axis at (yc, zc) of a square periodic y-z box;
    // zc a cell face like yc (nz = ny even).
    const bool round = r.nozzle == 1;
    if (round && r.nz != ny)
        return bad(2, "--nozzle", std::format(
                                      "round needs --nz = ny = {} (a square "
                                      "periodic y-z box around the round jet; "
                                      "ny is set by --slot-width, "
                                      "--pilot-width, --coflow-width)",
                                      ny));
    // 3D (--nz >= 8): z periodic at hz = h -- one span period of an infinite
    // slot; nz == 1 keeps hz = 1.
    auto box = makeBox(nx, ny, r.nz, nx * h, ny * h,
                       r.nz == 1 ? 1.0 : r.nz * h);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    if (r.turb > 0) {
        if (auto e = flameTurbSetup(r, c, r.jetSpeed, "U_jet"); e)
            return std::unexpected(*e);
        c.warnings.push_back(std::format(
            "--turb {:g} with --mode bunsen: the IC cone and the --length "
            "bound use Damkoehler's S_T = {:.4g} S_L (an upper estimate: a "
            "slower-burning cone is taller -- watch xTip; exit 8 when the tip "
            "reaches the outflow band)",
            r.turb, sT));
    }
    BunsenGeom& b = c.bunsen;
    b.W = r.slotWidth * dth;
    b.uJet = r.jetSpeed;
    b.uCo = uco;
    b.rimLip = lip * dth;
    b.xRim = nTube * h;
    b.tRim = rim ? r.rimTemp : 1.0;
    b.wCo = wco * dth;
    b.tCo = tco;
    b.pilot = r.pilotWidth * dth;
    b.tPilot = tp;
    b.uPilot = up;
    b.edge = r.slotEdge * dth;
    b.yc = half * h;
    b.coneH = coneH * dth;
    b.pulse = r.pulse;
    b.pulseFreq = r.pulseFreq;
    b.round = round;
    b.zc = round ? half * h : 0.0;
    if (rim) b.rimScale = lmPoiseuilleScale(b, c.grid);
    c.flameBunsen = true;
    c.uInf = r.jetSpeed;
    return {};
}

} // namespace dns::cfg

namespace dns::cfg {
namespace {

// --mode diffusion defaults that differ from the shared flame flags' (those
// are premixed numbers in delta_th and S_L): the jet Mach, the slot width
// (the jet Peclet number), the edge half-width, all in l = D0/U_jet.
constexpr double kDiffMachDefault = 0.05;

constexpr double kDiffSlotDefault = 16;

// Edge 3: measured (2026-10-28, W 16, s 1, beta 5, Da 1000, M 0.05, 4
// cells/l, tt 40) edge 1 exits 8 at t = 0.48 (the thin IC sheet goes out);
// edge 2 and edge 1 + 2 l burnt Z_st pilots anchor; edge 3 is steady (tip
// 51.2 l, TmaxIn/T_ad 0.986). docs/chem.md, anchoring table.
constexpr double kDiffEdgeDefault = 3.0;

// beta 5: measured (same case, edge 3, outflow sigma 4, the inflow pin) beta 8
// at Da 1000 extinguishes the IC sheet by t = 1.8 (exit 8); beta 5 anchors
// (tip 51.2 l, Tmax/T_ad 0.986, steady by t ~ 60).
constexpr double kDiffBetaDefault = 5.0;

// --mode candle defaults (docs/chem.md, candle section), in l = D0/U_fuel and
// U_fuel. Paraffin/air: s ~3.4 (mass, with O2), Y_O2 0.233 -> Z_st 0.064.
// Fr = U^2/(g W) = 1/(g W) = 0.071 at the defaults (buoyancy-controlled).
// Measured (2026-10-28, scouts): in the 2D buoyancy regime the flame's
// aspect follows the fuel Peclet number W (H/w ~ Q/D): W 2 at g 7 burned a
// needle (tip 24 l, half-width 2 l), W 1 at g 14 a teardrop (tip ~9.5 l,
// half-width 1.4 l) -- hence the low-Peclet wick on a finer grid.
constexpr double kCandleGravityDefault = 14;

constexpr double kCandleSlotDefault = 1;

constexpr double kCandleStoichDefault = 3.4;

constexpr double kCandleOxYDefault = 0.233;

constexpr double kCandleEdgeDefault = 0.5;

constexpr double kCandleCellsDefault = 8;

constexpr double kCandleCoflowWidthDefault = 12;

constexpr double kCandleLengthDefault = 32;

// Da 5000: at 1000 the W 1 candle's ignition kernel went out (exit 8 at t
// 0.56, Tmax past the band 2.99 T0).
constexpr double kCandleDaDefault = 5000;

// The IC lights a short column (a closed Burke-Schumann sheet 4 l tall), not
// the whole box: the extruded column of the jet IC is a box-tall thermal that
// rolled up into vortices descending from the top (scout, W 4: outflow
// backflow -44 U, diverged at t 10.7).
constexpr double kCandleIgnitionDefault = 4;

// M = U_fuel/c0: the buoyant flow runs ~35 U_fuel (measured |u|max), so the
// fuel Mach is inflated far less than the jet's: buoyant Mach ~0.18.
constexpr double kCandleMachDefault = 0.005;

// --mode candle --wick defaults (docs/chem.md, "Wick candle"), in l and T0.
// Wick 0.75 l = 6 cells at 8 cells/l: the ghost wall's two mirror cells per
// side with room to spare; 2 l tall (about 1/5 of the flame, a real candle's
// proportion). T_wick 2 T0: paraffin vapour boils at ~600-650 K (T0 300 K).
// Fuel rate 1 = the slot candle's Q = rho U W at W 1 (tip ~11 l). Body 6 x 4 l
// with 9.5 l of air per side: the slot candle's 25 l periodic width.
constexpr double kWickWidthDefault = 0.75;

constexpr double kWickHeightDefault = 2;

constexpr double kWickTempDefault = 2;

constexpr double kWickFuelRateDefault = 1;

constexpr double kCandleBodyWidthDefault = 6;

constexpr double kCandleBodyHeightDefault = 4;

constexpr double kWickCoflowWidthDefault = 9.5;

// Air below the candle body past the inflow sink band (the coflow passes
// around it), l; at least 4 cells.
constexpr double kWickGapBelow = 1;

} // namespace
} // namespace dns::cfg

namespace dns::cfg {

std::optional<ConfigError> wickFlagScope(const RawConfig& r, bool candle)
{
    if (!candle)
        return rejectGiven(r,
                           {"--wick", "--wick-width", "--wick-height",
                            "--wick-temp", "--fuel-rate", "--candle-width",
                            "--candle-height", "--soot", "--soot-form",
                            "--soot-ox"},
                           "only applies to --mode candle --wick");
    if (r.wick)
        return rejectGiven(r,
                           {"--slot-width", "--slot-edge", "--pilot-width",
                            "--pilot-speed", "--pulse", "--pulse-freq",
                            "--ignition", "--quiescent"},
                           "does not apply with --wick (the wick is an "
                           "immersed body; fuel vapour enters through its "
                           "surface, the IC lights a sheet around it)");
    return rejectGiven(r,
                       {"--wick-width", "--wick-height", "--wick-temp",
                        "--fuel-rate", "--candle-width", "--candle-height",
                        "--soot", "--soot-form", "--soot-ox"},
                       "only applies to --mode candle --wick");
}

} // namespace dns::cfg

namespace dns::cfg {
namespace {

// The wick candle on top of a validated --mode candle case: the body, the
// wick, the fuel source, the soot model and the box (dns/wick.hpp).
std::expected<Case, ConfigError> finishWick(const RawConfig& r, Case c)
{
    auto val = [&](const char* flag, double v, double dflt) {
        return r.given.contains(flag) ? v : dflt;
    };
    const double cells = c.flameCells, h = c.h;
    if (!r.given.contains("--damkohler")) c.diff.Da = kWickDaDefault;
    WickSpec& s = c.wick;
    s.wWick = val("--wick-width", r.wickWidth, kWickWidthDefault);
    if (!(std::isfinite(s.wWick) && s.wWick * cells >= 4 - 1e-9 &&
          s.wWick <= 20))
        return bad(2, "--wick-width",
                   std::format("must be in [4 cells, 20] l ({:.4g} l at "
                               "--cells-per-unit {:g}): the ghost wall needs "
                               "two mirror cells per side",
                               4.0 / cells, cells));
    s.hWick = val("--wick-height", r.wickHeight, kWickHeightDefault);
    if (!(std::isfinite(s.hWick) && s.hWick * cells >= 2 - 1e-9 &&
          s.hWick <= 50))
        return bad(2, "--wick-height", "must be in [2 cells, 50] l");
    s.tWick = val("--wick-temp", r.wickTemp, kWickTempDefault);
    if (!(std::isfinite(s.tWick) && s.tWick >= 1 && s.tWick <= c.diff.sigma))
        return bad(2, "--wick-temp", std::format("must be in [1, sigma = {:g}] "
                                                 "(the wick wall T/T0: the "
                                                 "fuel's vaporisation "
                                                 "temperature)",
                                                 c.diff.sigma));
    s.tCandle = 1;
    s.Q = val("--fuel-rate", r.fuelRate, kWickFuelRateDefault);
    if (!(finitePos(s.Q) && s.Q <= 20))
        return bad(2, "--fuel-rate", "must be in (0, 20] (fuel vapour mass "
                                     "per unit time and depth, rho0 U l)");
    s.yF = c.diff.yF1;
    s.wCandle = val("--candle-width", r.candleWidth, kCandleBodyWidthDefault);
    if (!(std::isfinite(s.wCandle) && s.wCandle >= s.wWick + 4 * h &&
          s.wCandle <= 200))
        return bad(2, "--candle-width", "must be in [wick width + 4 cells, "
                                        "200] l");
    const double hC = val("--candle-height", r.candleHeight,
                          kCandleBodyHeightDefault);
    if (!(std::isfinite(hC) && hC * cells >= 4 - 1e-9 && hC <= 100))
        return bad(2, "--candle-height", "must be in [4 cells, 100] l");
    const double wco = r.coflowWidth < 0 ? kWickCoflowWidthDefault
                                         : r.coflowWidth;
    if (!(std::isfinite(wco) && wco >= 2))
        return bad(2, "--coflow-width", "must be >= 2 l per side (y is "
                                        "periodic: the air separates the "
                                        "candle from its image)");
    const int nb = c.nscbcSink;
    const double gap = std::max(kWickGapBelow, 4 * h);
    const double len = c.grid.nx * h - 2.0 * nb * h; // validateDiffusion's
    if (!(gap + hC + s.hWick + 8 <= len))
        return bad(2, "--length", std::format("must be >= {:.4g} l: the air "
                                              "below the body, the body, the "
                                              "wick and 8 l of flame",
                                              gap + hC + s.hWick + 8));
    if (!r.soot)
        if (auto e = rejectGiven(r, {"--soot-form", "--soot-ox"},
                                 "does not apply with --soot off");
            e)
            return std::unexpected(*e);
    c.wickSoot = r.soot != 0;
    c.soot.Daf = val("--soot-form", r.sootForm, kSootDafDefault);
    c.soot.Daox = val("--soot-ox", r.sootOx, kSootDaoxDefault);
    if (!(finitePos(c.soot.Daf) && c.soot.Daf <= 1e6))
        return bad(2, "--soot-form", "must be in (0, 1e6] (A_f e^(-Ta_f/T_ad) "
                                     "l/U)");
    if (!(finitePos(c.soot.Daox) && c.soot.Daox <= 1e6))
        return bad(2, "--soot-ox", "must be in (0, 1e6] (A_ox rho0 "
                                   "e^(-Ta_ox/T_ad) l/U)");

    const int half = int(std::ceil((0.5 * s.wCandle + wco) * cells - 1e-9));
    const int ny = 2 * half, nx = c.grid.nx;
    auto box = makeBox(nx, ny, 1, nx * h, ny * h, 1.0);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    s.yc = half * h;
    s.xBase = nb * h + gap;
    s.xTop = s.xBase + hC;
    c.bunsen.yc = s.yc;
    c.bunsen.wCo = wco;
    c.flameWick = true;
    c.flameIgnition = 0; // the wick IC lights its own sheet
    c.n = c.grid.nx;
    c.L = c.grid.nx * h;
    c.meta.body = c.wickSoot ? "wick candle diffusion flame with soot"
                             : "wick candle diffusion flame";
    c.meta.setGrid(c.grid);
    // Blowing speed of the vapour off the wick: Q / (rho_v perimeter), rho_v
    // = p0/(R T_wick) = 1/tWick.
    const double uBlow = s.Q * s.tWick / (2 * s.hWick + s.wWick);
    if (uBlow > 2)
        c.warnings.push_back(std::format(
            "--fuel-rate {:g}: the vapour leaves the wick at ~{:.3g} U "
            "(Q T_wick/(rho0 T0 perimeter)): a jet, not a candle",
            s.Q, uBlow));
    return c;
}

} // namespace
} // namespace dns::cfg

namespace dns::cfg {

std::expected<Case, ConfigError> validateDiffusion(const RawConfig& r, Case c)
{
    const bool candle = r.flameMode == 3;
    if (auto e = chamberFlagScope(r, false); e) return std::unexpected(*e);
    if (auto e = wickFlagScope(r, candle); e) return std::unexpected(*e);
    if (auto e = candle ? rejectGiven(r, {"--pilot-width", "--pilot-speed"},
                                      "does not apply to --mode candle (the "
                                      "flame anchors on its inflow sheet: no "
                                      "pilot)")
                        : rejectGiven(r, {"--gravity", "--quiescent"},
                                      "only applies to --mode candle");
        e)
        return std::unexpected(*e);
    if (candle && r.quiescent)
        if (auto e = rejectGiven(r,
                                 {"--slot-width", "--slot-edge",
                                  "--coflow-speed", "--pulse", "--pulse-freq"},
                                 "does not apply with --quiescent (no fuel, "
                                 "every inflow row at rest)");
            e)
            return std::unexpected(*e);
    // A candle default where the shared flag (a premixed / jet number) was
    // not given.
    auto pick = [&](const char* flag, double given, double dflt) {
        return candle && !r.given.contains(flag) ? dflt : given;
    };
    if (auto e = rejectGiven(r, {"--re"}, "does not apply: the jet is set by "
                                          "--slot-width (its Peclet number "
                                          "U W / D0) and --prandtl");
        e)
        return std::unexpected(*e);
    if (auto e = rejectGiven(r, {"--ft"}, "does not apply: the run length is "
                                          "--tt (in l/U_jet)");
        e)
        return std::unexpected(*e);
    if (auto e = rejectGiven(r, {"--n"}, "does not apply: the grid is "
                                         "--cells-per-unit over the burner "
                                         "and --length");
        e)
        return std::unexpected(*e);
    if (c.physics != Physics::Compressible)
        return bad(2, "--physics", "a flame is compressible only (no low-Mach "
                                   "branch until P7; docs/chem.md)");
    if (auto e = rejectGiven(r,
                             {"--upstream", "--downstream", "--inflow-speed",
                              "--wrinkle", "--wrinkle-modes", "--ny",
                              "--profile-dt", "--cells-per-delta",
                              "--jet-speed", "--coflow-temp", "--pilot-temp"},
                             "does not apply to --mode diffusion (the jet "
                             "speed is the unit, the inflow state is the "
                             "Burke-Schumann sheet of its mixture fraction, "
                             "the grid is --cells-per-unit)");
        e)
        return std::unexpected(*e);
    if (auto e = turbFlagScope(r, false, false); e) return std::unexpected(*e);
    if (auto e = lesTflesFlagScope(r, false); e) return std::unexpected(*e);
    if (r.nz != 1)
        return bad(2, "--nz", "must be 1: 3D --mode diffusion/candle is not "
                              "built");
    c.mach = r.given.contains("--mach")
                 ? r.mach
                 : (candle ? kCandleMachDefault : kDiffMachDefault);
    if (!(finitePos(c.mach) && c.mach <= 0.2))
        return bad(2, "--mach", "must be in (0, 0.2] for --mode diffusion "
                                "and candle: the jet Mach number U_jet/c0");
    DiffusionModel md;
    md.s = pick("--stoich", r.stoich, kCandleStoichDefault);
    if (!(std::isfinite(md.s) && md.s > 0 && md.s <= 100))
        return bad(2, "--stoich", "must be in (0, 100] (oxidizer mass per "
                                  "unit fuel mass)");
    md.yF1 = r.fuelY;
    if (!(md.yF1 > 0 && md.yF1 <= 1))
        return bad(2, "--fuel-y", "must be in (0, 1] (the fuel stream's Y_F; "
                                  "balance inert)");
    md.yO2 = pick("--ox-y", r.oxY, kCandleOxYDefault);
    if (!(md.yO2 > 0 && md.yO2 <= 1))
        return bad(2, "--ox-y", "must be in (0, 1] (the oxidizer stream's "
                                "Y_O; balance inert)");
    md.sigma = r.expansion;
    if (!(md.sigma > 1 && md.sigma <= 10))
        return bad(2, "--expansion", "must be in (1, 10] (sigma = T_ad(Z_st) "
                                     "/ T0, the flame-sheet temperature)");
    md.beta = r.given.contains("--zeldovich") ? r.zeldovich : kDiffBetaDefault;
    if (!(md.beta >= 2 && md.beta <= 20))
        return bad(2, "--zeldovich", "must be in [2, 20]");
    md.Da = pick("--damkohler", r.damkohler, kCandleDaDefault);
    if (!(finitePos(md.Da) && md.Da <= 1e6))
        return bad(2, "--damkohler", "must be in (0, 1e6] (A rho0 "
                                     "e^(-Ta/T_ad) l/U_jet)");
    c.prandtl = r.prandtl;
    if (!finitePos(c.prandtl)) return bad(2, "--prandtl", "must be finite and > 0");
    c.sutherland = r.visc == 1;
    c.tinfK = r.tinf;
    if (c.sutherland && !(r.given.contains("--tinf") && finitePos(c.tinfK)))
        return bad(2, "--tinf", "is required with --visc sutherland: T0 in "
                                "kelvin (Sutherland's 110.4 K needs a "
                                "dimensional anchor)");
    if (!c.sutherland && r.given.contains("--tinf"))
        return bad(2, "--tinf", "only anchors --visc sutherland");
    const double cells =
        pick("--cells-per-unit", r.cellsPerUnit, kCandleCellsDefault);
    if (!(std::isfinite(cells) && cells >= 2 && cells <= 64))
        return bad(2, "--cells-per-unit", "must be in [2, 64] (cells per l = "
                                          "D0/U_jet)");
    const double h = 1.0 / cells;
    const double W = r.given.contains("--slot-width")
                         ? r.slotWidth
                         : (candle ? kCandleSlotDefault : kDiffSlotDefault);
    // The candle's wick is a low-Peclet slot (the default Pe is 1).
    const double wMin = candle ? 0.5 : 4;
    if (!(std::isfinite(W) && W >= wMin && W <= 200))
        return bad(2, "--slot-width", std::format("must be in [{:g}, 200] l "
                                                  "(the jet Peclet number U W "
                                                  "/ D0)",
                                                  wMin));
    if (!(std::isfinite(r.coflowSpeed) && r.coflowSpeed >= 0 &&
          r.coflowSpeed <= 2))
        return bad(2, "--coflow-speed", "must be in [0, 2] U_jet");
    const double wco = r.coflowWidth < 0
                           ? (candle ? kCandleCoflowWidthDefault : W)
                           : r.coflowWidth;
    if (!(std::isfinite(wco) && wco >= 2))
        return bad(2, "--coflow-width", "must be >= 2 l per side (y is "
                                        "periodic: the coflow separates the "
                                        "jet from its image)");
    if (!(std::isfinite(r.pilotWidth) && r.pilotWidth >= 0))
        return bad(2, "--pilot-width", "must be finite and >= 0 l");
    if (!(r.pilotWidth > 0))
        if (auto e = rejectGiven(r, {"--pilot-speed"}, "needs --pilot-width > 0");
            e)
            return std::unexpected(*e);
    const double up = r.pilotSpeed < 0 && !r.given.contains("--pilot-speed")
                          ? r.coflowSpeed
                          : r.pilotSpeed;
    if (!(std::isfinite(up) && up >= 0 && up <= 2))
        return bad(2, "--pilot-speed", "must be in [0, 2] U_jet");
    const double edge = r.given.contains("--slot-edge")
                            ? r.slotEdge
                            : (candle ? kCandleEdgeDefault : kDiffEdgeDefault);
    if (!(std::isfinite(edge) && edge * cells >= 1))
        return bad(2, "--slot-edge", std::format("must be >= one cell ({:.4g} "
                                                 "l at --cells-per-unit {:g})",
                                                 1.0 / cells, cells));
    const double tipCold = sheetTipConstantDensity(W, md.zst());
    const double len =
        r.flameLength >= 0
            ? r.flameLength
            : (candle ? kCandleLengthDefault
                      : std::ceil(std::min(1.25 * tipCold + 8, 20.0 * W)));
    if (!(std::isfinite(len) && len >= W))
        return bad(2, "--length", "must be >= the slot width (l)");
    const double ign = r.given.contains("--ignition")
                           ? r.ignition
                           : (candle ? kCandleIgnitionDefault : 0.0);
    if (r.given.contains("--ignition") &&
        !(std::isfinite(ign) && ign >= 1 && ign <= len))
        return bad(2, "--ignition", "must be in [1, --length] l (the IC's "
                                    "burning column)");
    if (candle && r.quiescent && r.given.contains("--ignition"))
        return bad(2, "--ignition", "does not apply with --quiescent (no "
                                    "fuel: the IC is the ambient at rest)");
    const double grav =
        candle ? (r.given.contains("--gravity") ? r.gravity
                                                : kCandleGravityDefault)
               : 0.0;
    if (!(std::isfinite(grav) && grav >= 0 && grav <= 1000))
        return bad(2, "--gravity", "must be in [0, 1000] U_jet^2/l (g along "
                                   "-x; 0 = the diffusion jet bit for bit)");
    if (!(std::isfinite(r.pulse) && r.pulse >= 0 && r.pulse <= 0.5))
        return bad(2, "--pulse", "must be in [0, 0.5] (fraction of the jet "
                                 "speed)");
    if (r.pulse > 0 && !(std::isfinite(r.pulseFreq) && r.pulseFreq > 0))
        return bad(2, "--pulse-freq", "must be finite and > 0 with --pulse "
                                      "(U_jet/l)");
    if (r.given.contains("--pulse-freq") && !(r.pulse > 0))
        return bad(2, "--pulse-freq", "needs --pulse > 0");
    const double uMax = std::max({1.0, r.coflowSpeed, up}) * (1 + r.pulse);
    if (uMax * c.mach > 0.3)
        return bad(2, "--mach", std::format("the fastest inflow stream's Mach "
                                            "{:.3g} exceeds 0.3",
                                            uMax * c.mach));
    if (candle && grav > 0) {
        const CandleScales cs = candleScales(md, grav, W, len, c.mach);
        if (cs.machBuoyant > 0.3)
            return bad(2, "--mach", std::format(
                                        "the buoyant Mach M sqrt(2 g L (sigma "
                                        "- 1)/sigma) = {:.3g} exceeds 0.3: "
                                        "lower --mach, --gravity or --length",
                                        cs.machBuoyant));
        if (cs.machBuoyant > 0.15)
            c.warnings.push_back(std::format(
                "buoyant Mach {:.3g} (free-fall speed {:.3g} U): O(M^2) "
                "compressibility in the plume",
                cs.machBuoyant, cs.uFreeFall));
        if (!r.quiescent && cs.froude > 0.3)
            c.warnings.push_back(std::format(
                "Froude number U^2/(g W) = {:.3g}: the fuel jet's momentum is "
                "not negligible against buoyancy (not a candle)",
                cs.froude));
        if (cs.adiabatic > 0.02)
            c.warnings.push_back(std::format(
                "adiabatic fraction (g/cp) L/(T_ad - T0) = {:.3g}: the "
                "inflated Mach makes the column's lapse rate visible",
                cs.adiabatic));
    }
    c.nscbcEtaIn = r.nscbcEtaIn;
    if (!finitePos(c.nscbcEtaIn))
        return bad(2, "--nscbc-etain", "must be finite and > 0");
    c.nscbcThermalInflow = r.nscbcTinflow;
    c.nscbcSink = r.given.contains("--nscbc-sink") ? r.nscbcSink : 16;
    if (c.nscbcSink < 0) return bad(2, "--nscbc-sink", "must be >= 0 (0 = off)");
    c.nscbcSinkAmp = r.nscbcSinkAmp;
    if (!(c.nscbcSinkAmp > 0 && c.nscbcSinkAmp <= 2))
        return bad(2, "--nscbc-sink-amp", "must be in (0, 2] (the SSP-RK3 "
                                          "real-axis bound)");

    const int nb = c.nscbcSink;
    const int half = int(std::ceil((0.5 * W + r.pilotWidth + wco) * cells - 1e-9));
    const int ny = 2 * half;
    const int nx = int(std::ceil(len * cells - 1e-9)) + 2 * nb;
    auto box = makeBox(nx, ny, 1, nx * h, ny * h, 1.0);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    BunsenGeom& b = c.bunsen;
    b.W = W;
    b.uJet = 1;
    b.uCo = r.coflowSpeed;
    b.wCo = wco;
    b.pilot = r.pilotWidth;
    b.uPilot = up;
    b.tCo = b.tPilot = 1; // unused: the inflow is the Burke-Schumann state
    b.edge = edge;
    b.yc = half * h;
    b.coneH = 0;
    b.pulse = r.pulse;
    b.pulseFreq = r.pulseFreq;
    c.diff = md;
    c.flameDiffusion = true;
    c.flameCandle = candle;
    c.flameGravity = grav;
    c.flameQuiescent = candle && r.quiescent;
    c.flameIgnition = c.flameQuiescent ? 0.0 : ign;
    c.flameTu = 1.0 / (kGammaAir * c.mach * c.mach);
    c.flameCells = cells;
    c.n = c.grid.nx;
    c.h = h;
    c.L = c.grid.nx * h;
    c.re = W / c.prandtl; // U W rho0 / mu, mu = Pr
    c.nu = c.prandtl;
    c.uInf = 1.0;
    c.far = FarField::Nscbc;
    c.bodyKind = BodyKind::None;

    if (auto e = resolveCfl(r, c, 0.4); e) return std::unexpected(*e);
    if (!finitePos(r.tt)) return bad(2, "--tt", "must be finite and > 0");
    c.ft = r.tt;
    c.tEnd = r.tt;
    if (auto e = resolveDtRecompute(r, c, 16); e) return std::unexpected(*e);
    if (auto e = validateFrameOutputs(r, c); e) return std::unexpected(*e);
    if (auto e = resolveEmit(c, {Emit::Stats}); e) return std::unexpected(*e);

    if (c.mach > 0.1)
        c.warnings.push_back(std::format(
            "--mach {:.4g}: the heat release's dilatation carries an O(M^2) "
            "compressibility error (gamma M^2 (sigma - 1) = {:.3g})",
            c.mach, kGammaAir * c.mach * c.mach * (md.sigma - 1)));
    if (nb == 0)
        c.warnings.push_back("--nscbc-sink 0: the bare characteristic "
                             "boundary (nothing drains mesh-scale content)");
    c.meta = {.D = 1.0,
              .re = c.re,
              .mach = c.mach,
              .frameDt = c.fdt,
              .body = candle ? (c.flameQuiescent ? "candle box at rest"
                                                 : "candle diffusion flame")
                             : "slot jet diffusion flame",
              .farField = "nscbc",
              .physics = name(c.physics),
              .kick = 0.0,
              .sgs = "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    if (candle && r.wick) return finishWick(r, std::move(c));
    return c;
}

} // namespace dns::cfg
