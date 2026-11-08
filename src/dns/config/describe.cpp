// Case::describe / describeCase: the resolved case for --dry-run.
#include "dns/config.hpp"
#include "dns/config/internal.hpp"

#include "dns/facemesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/parallel.hpp"

#include "dns/pseudo_time.hpp"
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
#include <string_view>

using std::numbers::pi;

namespace dns {
using namespace cfg;
} // namespace dns

namespace dns {

// ---------------------------------------------------------------------------

std::string Case::describe() const
{
    std::string s = describeCase();
    if (shock == ShockMode::Hll && fvMode == FvCompMode::PolyMesh)
        s += std::format("  shock      hll (mesh lines, stride 2, {:g}..{:g}, per step) + D4 eps {:g} |u.S|\n", kShockTh1,
                         kShockTh2, shockEps4);
    else if (shock == ShockMode::Hll)
        s += std::format("  shock      hll (pressure-sensed two-point HLL, sensor {:g}..{:g}, per step)\n", kShockTh1,
                         kShockTh2);
    if (shock == ShockMode::Sense)
        s += std::format("  shock      sense (sensor only, {:g}..{:g}, per step: the central solver bit for bit)\n",
                         kShockTh1, kShockTh2);
    if (fvMode == FvCompMode::Structured && sub == Subcommand::Flame)
        s += std::format("  solver     {}\n  patches    {}\n  frames     uniform plane, containing cell per pixel, long "
                         "side {}\n",
                         fvFlameLabel(*this),
                         physics == Physics::LowMach && flameBunsen
                             ? std::string("x-lo inflow face, one row per face (the structured rows: lmInflowRow of "
                                           "the owner's row), x-hi outflow p = 0, ") +
                                   (lmOpenY ? "y `open` faces (one rule by the sign of the face flux: out zero "
                                              "gradient, in the ambient at rest, p = -rho_a u_n^2/2; the x-hi "
                                              "backflow too)"
                                            : "y cyclic") +
                                   (lmRim ? ", the tube's staircase cells removed: its faces the isothermal `rim` "
                                            "Wall patch"
                                          : "") +
                                   (lmGravity > 0 ? ", gravity the face force (rho_f - rho_a) g . S" : "") +
                                   "; cone measured on a pixel = cell plane (same mesh)"
                         : physics == Physics::LowMach
                             ? "x-lo inflow face, x-hi outflow p = 0 (the structured faces above; same mesh)"
                         : flameCandle && flameGravity > 0
                             ? std::string(flameQuiescent ? "x-lo Inflow at rest (air), one fixed-face row per face"
                                                          : "x-lo Inflow, one fixed-face row per face (the "
                                                            "Burke-Schumann sheet)") +
                                   ", x-hi Far (characteristic), one row per face: the hydrostatic ambient column's "
                                   "air at T0 at its owner's centroid, moving with the coflow; no NSCBC and no sink "
                                   "(any NSCBC line above is the structured solver's; same mesh)"
                         : flameDiffusion
                             ? "x-lo Inflow, one fixed-face row per face (the Burke-Schumann sheet), x-hi Outflow at "
                               "p0 carrying its own Y, no NSCBC and no sink (any NSCBC line above is the structured "
                               "solver's; same mesh)"
                             : "x-lo Far at the fresh state, x-hi Outflow at the burnt p, no NSCBC and no sink (any "
                               "NSCBC line above is the structured solver's; same mesh)",
                         fvFrameN);
    else if (fvMode == FvCompMode::Structured)
    {
        s += std::format("  solver     {}\n  patches    {}\n  frames     uniform plane, containing cell per pixel, "
                         "long side {}\n",
                         fvCompLabel(*this, grid.nz == 1),
                         ogrid ? "wall row: Wall; outer circle: Far (1D characteristic, not the structured far row)"
                               : "box faces: Far (1D characteristic: no NSCBC, no sink), --noslip y: Wall, --far "
                                 "periodic: cyclic",
                         fvFrameN);
    }
    if (checkpointDt > 0 || !restartPrefix.empty()) {
        s += checkpointDt > 0
                 ? std::format("  checkpoint every {:g} (model time) -> {}_ckpt.bin{}\n",
                               checkpointDt, outPrefix,
                               wallLimit >= 0
                                   ? std::format("; wall limit {:g} s", wallLimit)
                                   : std::string())
                 : std::string("  checkpoint none\n");
        if (!restartPrefix.empty())
            s += std::format("  restart    from {}_ckpt.bin (outputs truncated to "
                             "its lengths, then appended)\n",
                             restartPrefix);
    }
    return s;
}

std::string Case::describeCase() const
{
    if (sub == Subcommand::Prop) return describeProp(prop);
    if (sub == Subcommand::Conv && fvConv)
        return std::format("case:\n"
                           "  mode       Rayleigh-Benard on a FaceMesh\n"
                           "  label      {}\n"
                           "  mesh       {}\n"
                           "  sides      {}\n"
                           "  driving    Ra={:.6g}  Pr={:.4g}  floor T={:.6g}, lid T={:.6g}, gravity {:.6g} in -y\n"
                           "  derived    nu={:.6g}  kappa={:.6g}  Uff={:.6g}  cfl={:.3g}\n"
                           "  run        tt={:.6g} free-fall times (tEnd={:.6g})  kick={:.3g}  height bins {}\n"
                           "  out        {}\n",
                           fvConvLabel(*this, fvMeshDir.empty() && grid.nz == 1),
                           fvMeshDir.empty() ? std::format("{} x {} x {} fluid cells", nxInterior, grid.ny - 2, nzInterior)
                                             : std::format("polyMesh {}, --bc {}", fvMeshDir, fvBcFile),
                           closedSides ? "closed: no-slip adiabatic FaceMesh walls" : "periodic: cyclic faces", rayleigh,
                           prandtl, tBot, tTop, gravity, nu, kappaDiff, uFreeFall, cfl, ft, tEnd, kick, fvConvBins,
                           outPrefix);
    if (sub == Subcommand::Mesh)
        return std::format("case:\n"
                           "  mode       FaceMesh report of an OpenFOAM polyMesh (host-only, no GPU)\n"
                           "  polymesh   {}\n"
                           "  order      {}\n",
                           polyMeshDir, cellOrderNames()[std::size_t(meshOrder)]);
    if (sub == Subcommand::Flow && fvMode == FvCompMode::PolyMesh)
        return std::format("case:\n"
                           "  mode       compressible FV path on an OpenFOAM polyMesh, GPU [FV path, body-fitted]\n"
                           "  polymesh   {}\n"
                           "  physics    Re {:g} on L {:g}, Mach {:g}, Pr {:g}, viscosity {}\n"
                           "  sgs        {}\n"
                           "  precision  {}\n"
                           "  time       tt {:g} after ramp {:g} (L/U), cfl {:g}\n"
                           "  frames     uniform plane, containing cell per pixel, long side {}\n"
                           "  out        {}\n{}",
                           fvMeshDir, re, fvBc.length, mach, prandtl, sutherland ? "sutherland" : "constant",
                           lesLabel                  ? lesRunLabel(sgsModel, cs, false)
                           : shock == ShockMode::Hll ? std::string(inviscid ? "Euler" : "laminar")
                                                     : std::string("DNS"),
                           fvPrecisionName(fvPrecision), tEnd, ramp, cfl, fvFrameN, outPrefix, fvBc.describe());
    if (fvIncomp)
    {
        const std::string mesh =
            sub == Subcommand::Tgv
                ? std::format("periodic box {} x {} x {}{}", grid.nx, grid.ny, grid.nz,
                              fvPerturb == 0 ? std::string()
                                             : std::format(", {} perturbation {:g}", fvPerturb == 1 ? "sine" : "random",
                                                           fvPerturbAmp))
            : !fvMeshDir.empty() ? std::format("polyMesh {}, --bc {}", fvMeshDir, fvBcFile)
                                 : std::format("O-grid {} x {} x {} (fromCurvMesh)", grid.nx, grid.ny - 2, grid.nz);
        return std::format("case:\n"
                           "  mode       incompressible time-accurate FV solver on a FaceMesh: collocated SSP-RK3 + "
                           "per-stage projection, skew-symmetric central convection, AMG-PCG [{}]\n"
                           "  mesh       {}\n"
                           "  label      {}\n"
                           "  run        {} = {:g}  cfl {:g}  pressure tol {:g}, maxIter {}\n"
                           "  precision  {}\n"
                           "  out        {}\n",
                           sub == Subcommand::Tgv ? "dns tgv" : "dns flow", mesh,
                           cs > 0 ? std::format("LES {} C {:g}", sgsModel == SgsModel::Sigma ? "sigma" : "Smagorinsky", cs)
                                  : std::string("DNS"),
                           fvMeshDir.empty() ? "tEnd" : "--ft", fvMeshDir.empty() ? tEnd : ft, cfl, fvTol, fvMaxPcg,
                           fvPrecisionName(fvPrecision), outPrefix);
    }
    if (sub == Subcommand::Flow && !fvCase.empty())
    {
        const FvNumerics num = fvNumerics(fvPrecision);
        const std::string prec =
            fvPrecision == FvPrecision::Fp64
                ? std::string("FP64 (--fv-precision fp64: the parity mode, solves to the fvSolution tolerances)")
                : std::format("FP32 (default; FP64 reductions; solve floors U {:g}, p {:g}, turbulence {:g}: "
                              "absolute tolerance max(fvSolution, floor), relTol unchanged)",
                              num.floors.U, num.floors.p, num.floors.s);
        return std::format("case:\n"
                           "  mode       incompressible FV path: OpenFOAM simpleFoam (SIMPLE/SIMPLEC) on the "
                           "identical polyMesh, {}, GPU [FV path, body-fitted]\n"
                           "  precision  {}\n"
                           "  polymesh   {}\n"
                           "  iterations {}\n"
                           "  out        {}\n{}",
                           fvPrecisionName(fvPrecision), prec, fvCase,
                           fvMaxIters > 0 ? std::to_string(fvMaxIters) : std::string("controlDict endTime"),
                           outPrefix,
                           fvCoupled < 0
                               ? std::string()
                               : std::format("  algorithm  coupled p-U PROTOTYPE, EXPERIMENTAL (plan_perf_5 P0b) "
                                             "after {} SIMPLE iterations: FGMRES(30) relTol {:g}, alpha_c {:g} "
                                             "(SER){}\n",
                                             fvCoupled, fvCoupledEta, fvCoupledAlpha,
                                             fvCoupledCheck ? ", G-JAC check" : ""));
    }
    if (sub == Subcommand::Hit) return describeHit(*this);
    if (sub == Subcommand::Flame && flameWick) return describeWick(*this);
    if (sub == Subcommand::Flame && flameDiffusion)
    {
        const DiffusionModel& m = diff;
        const BunsenGeom& b = bunsen;
        const double sig = m.sigma;
        double uMax = std::max({1.0, b.uCo, b.uPilot}) * (1 + b.pulse);
        std::string mode = "2D slot jet diffusion flame (dns flame --mode "
                           "diffusion)";
        std::string grav;
        if (flameCandle) {
            mode = "2D candle: the diffusion flame under gravity, pointing "
                   "+x (dns flame --mode candle)";
            const double Lb = grid.nx * h;
            if (flameGravity > 0) {
                const CandleScales cs =
                    candleScales(m, flameGravity, b.W, Lb, mach);
                uMax = std::max(uMax, cs.uFreeFall);
                grav = std::format(
                    "  gravity    g = {:.6g} U^2/l along -x; ambient column "
                    "isothermal T0, p_a(x) = p0 exp(-g x/(R T0)), p_a(L)/p0 = "
                    "{:.6g}\n"
                    "             Fr = U^2/(g W) = {:.4g}; free-fall speed "
                    "sqrt(2 g L (sigma-1)/sigma) = {:.4g} U (buoyant Mach "
                    "{:.3g})\n"
                    "             adiabatic fraction (g/cp) L/(T_ad - T0) = "
                    "{:.3g}; flicker scale 0.479 sqrt(g/W) = {:.4g} U/l\n",
                    flameGravity, 1 - cs.hydroDrop, cs.froude, cs.uFreeFall,
                    cs.machBuoyant, cs.adiabatic, cs.flicker);
            } else
                grav = "  gravity    g = 0: the diffusion jet bit for bit\n";
            if (flameQuiescent)
                grav += "  quiescent  NO fuel, every inflow row at rest (Z = "
                        "0, u = 0, T0): the hydrostatic rest check\n";
        }
        const double dtAc = cfl * h / (std::sqrt(sig) / mach + sig * uMax);
        const double dtTh = 0.25 * h * h * (1.0 / sig) /
                            (prandtl * std::max(1.0, kGammaAir / prandtl));
        return std::format(
            "case:\n"
            "  mode       {}\n"
            "{}"
            "  model      single step F + s O -> (1+s) P, omega = A rho^2 Y_F "
            "Y_O e^(-Ta/T); calorically perfect\n"
            "             (gamma 1.4), equal W, Le = 1; s = {:g}, Y_F1 = {:g}, "
            "Y_O2 = {:g} -> Z_st = {:.6g}\n"
            "             T_ad/T0 = {:.6g}, q/(cp T0) = {:.6g} per unit fuel, "
            "beta = {:.6g}, Ta/T0 = {:.6g}, Da = {:.6g} (ln A = {:.6g})\n"
            "  units      rho0 = R = U_jet = 1, l = D0/U_jet = 1 (mu = Pr = "
            "{:.4g}), T0 = p0 = 1/(gamma M^2) = {:.6g}\n"
            "  transport  {}\n"
            "  grid       {} x {} x 1  h = {:.6g} = l/{:.4g}  box {:.6g} x "
            "{:.6g}  [x jet axis, y periodic]\n"
            "  burner     slot W = {:.4g} l (Pe = Re Pr = {:.4g}, Re = {:.4g}) "
            "centred at y = {:.6g}, fuel Z = 1, u = 1 (jet Mach {:.3g})\n"
            "             oxidizer coflow {:.4g} l per side, Z = 0, u = {:.4g}"
            "{}\n"
            "             edges tanh half-width {:.4g} l ({:.3g} cells){}\n"
            "             inflow state: the Burke-Schumann sheet of Z(y) (the "
            "jet edge burns at the exit plane)\n"
            "  sheet      constant-density Burke-Schumann tip W^2/(16 "
            "erfinv(Z_st)^2) = {:.4g} l (scale only; D(T) shortens it); box "
            "length {:.4g} l past the bands\n"
            "  boundary   x-lo NSCBC per-row inflow (u, rho), etaIn {:.4g}, "
            "species pinned to the sheet; x-hi outflow p = {} (anchor sigma "
            "{:g}), backflow carries the cell's own Y\n"
            "             sink {} cells, amp {:.3g}\n"
            "  run        tt = {:.6g} l/U_jet  cfl = {:.3g}  dt est {:.3e} "
            "(acoustic {:.3e}, thermal {:.3e}; chemistry at run time)\n"
            "  label      {}\n"
            "  out        {}\n",
            mode, grav,
            m.s, m.yF1, m.yO2, m.zst(), sig, m.qOverCpT0(), m.beta, m.theta(),
            m.Da, m.lnA(), prandtl, flameTu,
            sutherland ? std::format("Sutherland mu(T), T0 = {:.6g} K", tinfK)
                       : std::string("constant mu = Pr, rho D = mu/Pr"),
            grid.nx, grid.ny, h, flameCells, grid.nx * h, grid.ny * h, b.W,
            b.W, b.W / prandtl, b.yc, mach, b.wCo, b.uCo,
            b.pilot > 0 ? std::format("\n             pilot strips {:.4g} l "
                                      "per side: burnt gas at Z_st (T_ad), u "
                                      "= {:.4g}",
                                      b.pilot, b.uPilot)
                        : std::string(),
            b.edge, b.edge / h,
            b.pulse > 0 ? std::format("\n             jet pulse u_jet (1 + "
                                      "{:.3g} sin(2 pi {:.4g} t))",
                                      b.pulse, b.pulseFreq)
                        : std::string(),
            sheetTipConstantDensity(b.W, m.zst()),
            grid.nx * h - 2.0 * nscbcSink * h, nscbcEtaIn,
            flameCandle && flameGravity > 0
                ? "p_a(x_out), both faces' incoming acoustic waves carrying "
                  "the ambient hydrostatic gradient; air drawn DOWN through "
                  "the top relaxes to the ambient rho_a, v = 0"
                : "p0",
            kDiffOutflowSigma,
            nscbcSink, nscbcSinkAmp, tEnd, cfl, std::min(dtAc, dtTh), dtAc, dtTh,
            flameRunLabel(*this), outPrefix);
    }
    // The premixed modes' turbulence / LES / TFLES lines (--mode premixed and
    // bunsen; empty without those flags).
    auto flameTurbLesText = [&]() {
        const double dth = flameEig.dthOverDD;
        std::string turbText;
        if (flameTurb > 0) {
            const TurbNumbers tn = turbNumbers(flameTurb, flameTurbL, prandtl, dth);
            const double kp = turbPeak(flameTurbL);
            turbText = std::format(
                "  turbulence frozen synthetic HIT box {} x {} x {} (L_bx = L_z), "
                "seed {}, swept through the inflow at {} (Taylor)\n"
                "             u'/S_L = {:.4g}  l_t = {:.4g} delta_th = {:.4g} "
                "delta_D ({:.3g} cells)  kp L_z/(2 pi) = {:.3g}\n"
                "             Re_t = u' l_t/nu = {:.4g}  eta = {:.4g}  h/eta = "
                "{:.3g}  tau_t = l_t/u' = {:.4g}\n"
                "             Ka = (u'/S_L)^1.5 (l_t/delta_th)^-0.5 = {:.4g}  Da "
                "= (l_t/delta_th)/(u'/S_L) = {:.4g}  (inflow values; u' decays "
                "toward the flame)\n"
                "             S_T/S_L: Peters 1999 {:.4g}, Damkoehler 1 + u'/S_L "
                "{:.4g} (report-only, at the inflow u')\n"
                "             {}; {}\n",
                flameTurbNbx, grid.ny, grid.nz, seed,
                !flameBunsen   ? "u_in"
                : bunsen.round ? "U_jet, times the jet weight J(r)"
                               : "U_jet, times the jet weight J(y)",
                flameTurb, flameTurbL / dth, flameTurbL, flameTurbL / h,
                kp * grid.nz * h / (2 * std::numbers::pi), tn.ReT, tn.eta,
                h / tn.eta, tn.tauT, tn.Ka, tn.Da,
                petersST(flameTurb, flameTurbL, 1.0), damkohlerST(flameTurb),
                flameBunsen ? "IC fresh gas carries the box times Y (the "
                              "fresh fraction)"
                            : "IC fresh gas carries the box up to x_f0 - 1.5 l_t",
                flameBunsen
                    ? std::format("Re_jet = U W/nu = {:.4g}",
                                  bunsen.uJet * bunsen.W / prandtl)
                    : std::string(flameHold
                                      ? "--hold-flame: u_in tracks the "
                                        "consumption speed, the fresh-mass "
                                        "thickness held at its first value"
                                      : "u_in fixed"));
        }
        // LES / TFLES (docs/chem.md, "LES species and TFLES").
        const bool fvl = fvMode != FvCompMode::Off;
        if (lesLabel)
            turbText += std::format(
                "  les        mu_t = rho (C Delta)^2 op(grad u), C = {:g} ({}), "
                "Delta = h; eddy conductivity mu_t cp/Pr_t, species SGS flux "
                "mu_t/Sc_t, Sc_t = Pr_t = 0.9;\n"
                "             {}\n",
                cs, sgsModel == SgsModel::Sigma ? "sigma" : "Smagorinsky",
                fvl ? "FV: mu_t off the least-squares gradient on every cell "
                      "(Delta^2 = V^(2/3)), no SGS flux on boundary faces"
                    : "mu_t on i = 0, nx-1 copied from i = 1, nx-2");
        if (flameTfles > 0)
            turbText += std::format(
                "  tfles      F_max = {:g}: {:.4g} cells across the thickened "
                "flame (F --cells-per-delta); sensor {}\n"
                "             efficiency {}; IC = the oracle thickened by "
                "F_max; {}\n",
                flameTfles, flameTfles * flameCells,
                flameTflesUniform
                    ? std::string("uniform (Omega = 1)")
                    : std::format("dynamic: tanh({:g} w_s/w_s,max), w_s the "
                                  "rate at Ta/2, spread toward colder gas by "
                                  "a T-gated max filter of radius {} cells",
                                  kTflesSensorBeta, flameTflesDilate),
                flameCharlette ? std::format("Charlette 2002 (beta {:g}, "
                                             "delta_L = delta_th, Re_Delta on "
                                             "nu_u = Pr), u'_Delta = Colin "
                                             "2 h^3 |lap curl u| (Delta/h)^(1/3)",
                                             kCharletteBeta)
                               : std::string("none (E = 1)"),
                fvl ? "FV: the sensor's filter walks faces (fvTflesWalk), the "
                      "Sigma-form dt takes mu max(4/3, F E gamma/Pr)"
                    : "dt also bounded by 0.25 h^2 rho_min/(gamma/Pr (mu F E "
                      "+ mu_t))");
        return turbText;
    };
    if (sub == Subcommand::Flame && flameBunsen && physics == Physics::LowMach)
        return describeLowMachBunsen(*this);
    if (sub == Subcommand::Flame && flameBunsen && grid.nz > 1)
    {
        // 3D slot Bunsen (docs/chem.md, "3D turbulent slot Bunsen").
        const double sig = flameModel.sigma, dth = flameEig.dthOverDD;
        const BunsenGeom& b = bunsen;
        const double dtAc = cfl * h / (std::sqrt(sig) / mach + sig * b.uJet);
        const double dtTh = 0.25 * h * h * (1.0 / sig) /
                            (prandtl * std::max(1.0, kGammaAir / prandtl));
        return std::format(
            "case:\n"
            "  mode       3D {} Bunsen flame{} (dns flame --mode bunsen "
            "--nz {})\n"
            "  model      single step F -> P, calorically perfect (gamma 1.4), "
            "equal W, Le = 1\n"
            "             sigma = Tb/Tu = {:.6g}  beta = {:.6g}  theta = Ta/Tu "
            "= {:.6g}  Pr = {:.4g}\n"
            "  oracle     Lambda = {:.10g}  delta_th/delta_D = {:.6g}\n"
            "  units      rho_u = S_L = R = 1, delta_D = 1, Tu = p0 = {:.6g}\n"
            "  grid       {} x {} x {} = {:.4g} M cells  h = {:.6g} = "
            "delta_th/{:.4g}  box {:.6g} x {:.6g} x {:.6g}  [x jet axis, y "
            "periodic, z periodic {} = {:.3g} W]\n"
            "  burner     {} = {:.4g} delta_th; jet u = {:.4g} S_L (jet "
            "Mach {:.3g}); coflow {:.4g} delta_th per side, T = {:.4g} Tu, u "
            "= {:.4g} S_L\n"
            "  cone       IC/--length cone height {:.4g} delta_th{}; box length "
            "{:.4g} delta_th past the bands\n"
            "{}"
            "{}"
            "  boundary   x-lo NSCBC per-row inflow, etaIn {:.4g}; x-hi p = "
            "p0; sink {} cells\n"
            "  run        tt = {:.6g} flame times  cfl = {:.3g}  dt est {:.3e} "
            "(acoustic {:.3e}, thermal {:.3e})\n"
            "  label      {}\n"
            "  out        {}\n",
            b.round ? "round-nozzle" : "slot",
            flameTurb > 0 ? " in synthetic turbulence" : "", grid.nz, sig,
            flameModel.beta, flameModel.theta(), prandtl, flameEig.Lambda, dth,
            flameTu, grid.nx, grid.ny, grid.nz,
            double(grid.nx) * grid.ny * grid.nz * 1e-6, h, flameCells,
            grid.nx * h, grid.ny * h, grid.nz * h,
            b.round ? "(square y-z box) L_z" : "span", grid.nz * h / b.W,
            b.round ? "round nozzle, diameter D" : "slot W", b.W / dth, b.uJet,
            b.uJet * mach, b.wCo / dth, b.tCo, b.uCo, b.coneH / dth,
            flameTurb > 0 ? " (Damkoehler S_T = S_L (1 + u'/S_L))" : " (ideal)",
            grid.nx * h / dth - 2.0 * nscbcSink / flameCells,
            flameTurbLesText(),
            b.round ? "  stats      the AXIAL plane z = zc (mean of the two planes "
                      "either side) feeds the cone measurement; ST = burn / (pi R "
                      "hypot(H, R)), R = D/2, area = integral |grad c| / that, I0 = "
                      "ST/area; SLflank NaN (per unit span)\n"
                    : "  stats      span means (z) feed the cone measurement; ST = "
                      "burn / (2 flank L_z), area = integral |grad c| / (2 flank "
                      "L_z), I0 = ST/area\n",
            nscbcEtaIn, nscbcSink, tEnd, cfl,
            std::min(dtAc, dtTh), dtAc, dtTh, flameRunLabel(*this), outPrefix);
    }
    if (sub == Subcommand::Flame && flameBunsen)
    {
        const double sig = flameModel.sigma, dth = flameEig.dthOverDD;
        const BunsenGeom& b = bunsen;
        const double dtAc = cfl * h / (std::sqrt(sig) / mach + sig * b.uJet);
        const double dtTh = 0.25 * h * h * (1.0 / sig) /
                            (prandtl * std::max(1.0, kGammaAir / prandtl));
        const double alpha =
            std::asin(1.0 / b.uJet) * 180.0 / std::numbers::pi;
        return std::format(
            "case:\n"
            "  mode       2D slot Bunsen flame (dns flame --mode bunsen)\n"
            "  model      single step F -> P, calorically perfect (gamma 1.4), "
            "equal W, Le = 1\n"
            "             sigma = Tb/Tu = {:.6g}  beta = {:.6g}  theta = Ta/Tu "
            "= {:.6g}  Pr = {:.4g}\n"
            "  oracle     Lambda = {:.10g}  delta_th/delta_D = {:.6g}\n"
            "  units      rho_u = S_L = R = 1, delta_D = 1, Tu = p0 = {:.6g}\n"
            "  transport  {}\n"
            "  grid       {} x {} x 1  h = {:.6g} = delta_th/{:.4g}  box "
            "{:.6g} x {:.6g}  [x jet axis, y periodic]\n"
            "  burner     slot W = {:.4g} delta_th centred at y = {:.6g}; jet "
            "Y = 1, T = Tu, u = {:.4g} S_L (jet Mach {:.3g})\n"
            "             coflow {:.4g} delta_th per side, Y = 0, T = {:.4g} Tu, "
            "u = {:.4g} S_L{}\n"
            "             edges tanh half-width {:.4g} delta_th ({:.3g} cells)"
            "{}\n"
            "  cone       ideal sin(alpha) = S_L/U: alpha = {:.4g} deg, height "
            "{:.4g} delta_th; box length {:.4g} delta_th past the bands\n"
            "  boundary   x-lo NSCBC per-row inflow (u, rho, Y), etaIn {:.4g}; "
            "x-hi outflow p = p0; q omega re-added on the open rows\n"
            "             sink {} cells, amp {:.3g}\n"
            "  run        tt = {:.6g} flame times  cfl = {:.3g}  dt est {:.3e} "
            "(acoustic {:.3e}, thermal {:.3e})\n"
            "  label      {}\n"
            "  out        {}\n",
            sig, flameModel.beta, flameModel.theta(), prandtl, flameEig.Lambda,
            dth, flameTu,
            sutherland ? std::format("Sutherland lambda(T), S/Tu = {:.6g} "
                                     "(Tu = {:.6g} K)",
                                     flameModel.sOverTu, tinfK)
                       : std::string("constant mu = Pr, lambda = cp"),
            grid.nx, grid.ny, h, flameCells, grid.nx * h, grid.ny * h,
            b.W / dth, b.yc, b.uJet, b.uJet * mach, b.wCo / dth, b.tCo, b.uCo,
            b.pilot > 0 ? std::format("\n             pilot strips {:.4g} "
                                      "delta_th per side: Y = 0, T = {:.4g} "
                                      "Tu, u = {:.4g} S_L",
                                      b.pilot / dth, b.tPilot, b.uPilot)
                        : std::string(),
            b.edge / dth, b.edge / h,
            b.pulse > 0 ? std::format("\n             jet pulse u_jet (1 + "
                                      "{:.3g} sin(2 pi {:.4g} t))",
                                      b.pulse, b.pulseFreq)
                        : std::string(),
            alpha, b.coneH / dth, grid.nx * h / dth - 2.0 * nscbcSink / flameCells,
            nscbcEtaIn, nscbcSink, nscbcSinkAmp, tEnd, cfl,
            std::min(dtAc, dtTh), dtAc, dtTh, flameRunLabel(*this), outPrefix);
    }
    if (sub == Subcommand::Flame && chamberMode) return describeChamber(*this);
    if (sub == Subcommand::Flame && fvMode == FvCompMode::PolyMesh) {
        // Compressible --equivalence: the premix every open patch carries (dns/drive/flame_fv.cpp runFvFlame).
        std::string premixText;
        if (flamePremix2 && physics != Physics::LowMach) {
            PartialPremixModel b = premix;
            b.phi = premixBurnPhi();
            premixText = std::format("  model      {} two-reactant F + {:g} O, phi {:g} (burnt premix phi {:g}): fresh "
                                     "Y_F {:.6g}, Y_O {:.6g} on every far / inflow patch; burnt Y_F {:.6g}, Y_O {:.6g} "
                                     "on every outflow; Tb/Tu {:.6g}\n",
                                     premixLaw.n > 0 ? "C2b" : "C2", premix.s, premix.phi, b.phi, b.yF1(), b.yO1(),
                                     b.phi > 1 ? b.yFrem() : 0.0, b.phi > 1 ? 0.0 : b.yOrem(), premix.sigmaAt(b.phi));
        }
        return std::format("dns flame: premixed, FV path, polyMesh {}\n  flame      sigma {:g} beta {:g} Pr {:g} M {:g}, "
                           "Lambda {:.10g}, delta_th/delta_D {:.6g}\n{}  run        tt {:g}, cfl {:g}, {}{}, x_f at s = {:g} "
                           "from the origin, u_in {:g}\n  output     {}\n{}",
                           fvMeshDir, flameModel.sigma, flameModel.beta, prandtl, mach, flameEig.Lambda,
                           flameEig.dthOverDD, premixText, tEnd, cfl, fvPrecisionName(fvPrecision),
                           compensated ? " compensated" : "", flameXf0, flameUIn, outPrefix, fvBc.describe());
    }
    if (sub == Subcommand::Flame)
    {
        const double sig = flameModel.sigma, dth = flameEig.dthOverDD;
        // dt estimate: the burnt acoustic bound and the thermal one (the
        // runner's stableDt; the chemistry bound is printed at run time).
        const double dtAc = cfl * h / (std::sqrt(sig) / mach + sig * flameUIn);
        const double dtTh = 0.25 * h * h * (1.0 / sig) /
                            (prandtl * std::max(1.0, kGammaAir / prandtl));
        // 3D (nz > 1): the grid's z, and the turbulence groups with --turb.
        const bool d3 = grid.nz > 1;
        const std::string turbText = flameTurbLesText();
        // Model C2 (--equivalence): the two-reactant jet's lines; model C0's
        // read as before.
        const PartialPremixModel& pm = premix;
        std::string modelText =
            "single step F -> P, calorically perfect (gamma 1.4), equal W, "
            "Le = 1\n";
        std::string oracleText = std::format(
            "Lambda = {:.10g} (FP64 phase plane; A burns at S_L = 1)  "
            "delta_th/delta_D = {:.6g}\n",
            flameEig.Lambda, dth);
        std::string qText = std::format("q = cp (sigma-1) Tu = {:.6g}",
                                        kGammaAir / (kGammaAir - 1) *
                                            (sig - 1) * flameTu);
        std::string yInText = "Y = 1";
        if (flamePremix2) {
            modelText = std::format(
                "two-reactant step F + {:g} O -> {:g} P (model C2: rich "
                "premixed jet), calorically perfect (gamma 1.4), equal W, "
                "Le = 1\n"
                "             phi = {:g}  s = {:g}  Y_O2(air) = {:g}: Y_F1 = "
                "{:.6g}  Y_O1 = {:.6g}  Y_F,rem = {:.6g}\n"
                "             Z_st = {:.6g} (the leftover fuel against air)  "
                "sigma_st = {:.6g} (Burke-Schumann sheet T/Tu)\n",
                pm.s, 1 + pm.s, pm.phi, pm.s, pm.yO2, pm.yF1(), pm.yO1(),
                pm.yFrem(), pm.zst(), pm.sigmaSt());
            // S_L(phi)/S_L(jet) at fixed A, q, Ta: the uncorrected step and
            // (--sl-law gri30) model C2b; one item per (phi, law), threaded
            // (own-index writes).
            constexpr double phis[] = {0.6, 0.7, 0.8, 0.9, 1.05, 1.1,
                                       1.2, 1.3, 1.4, 1.5, 1.6};
            constexpr std::size_t nP = std::size(phis);
            const PhiRateLaw<double>* law =
                premixLaw.n > 0 ? &premixLaw : nullptr;
            const std::size_t nItems = law ? 2 * nP : nP;
            std::vector<double> ratio(nItems, NAN);
            forItems(workerCount(0, nItems), nItems,
                     [&](int, std::size_t k) {
                         const auto r = premixedSpeedRatio(
                             pm, phis[k % nP], k < nP ? nullptr : law);
                         if (r) ratio[k] = *r;
                     });
            auto row = [&](std::size_t off) {
                std::string t;
                for (std::size_t k = 0; k < nP; ++k)
                    t += std::isfinite(ratio[off + k])
                             ? std::format(" {:g}:{:.4g}", phis[k],
                                           ratio[off + k])
                             : std::format(" {:g}:-", phis[k]);
                return t;
            };
            std::string lawText;
            if (law)
                lawText = std::format(
                    "             S_L law C2b (--sl-law gri30): omega x C(xi), "
                    "ln C clamped cubic spline on {} GRI-3.0 nodes phi "
                    "{:g}..{:g}; C(jet) = 1, C(phi 1 = Z_st) = {:.4g}\n"
                    "             C2b S_L(phi)/S_L(jet) (= GRI-3.0's at the "
                    "nodes):{}\n",
                    premixLaw.n, ch4Gri30Speeds().front().phi,
                    ch4Gri30Speeds().back().phi, phiRateC(premixLaw, pm, 1.0),
                    row(nP));
            if (premixMixPhi > 0) {
                PartialPremixModel b = pm;
                b.phi = premixMixPhi;
                lawText += std::format(
                    "             --mix-phi {:g}: burns the jet diluted with "
                    "air, Y_F = {:.6g}  Y_O = {:.6g}  Tb/Tu = {:.6g}  S_L/S_L"
                    "(jet) = {:.6g} (oracle; delta_th below is this flame's)\n",
                    premixMixPhi, b.yF1(), b.yO1(), pm.sigmaAt(premixMixPhi),
                    flameEig.tab.m);
            }
            oracleText = std::format(
                "Lambda2 = {:.10g} (FP64 phase plane, omega = A rho^2 Y_F Y_O "
                "e^(-Ta/T); A burns the jet at S_L = 1)  delta_th/delta_D = "
                "{:.6g}\n"
                "             uncorrected one-step S_L(phi)/S_L(jet) at fixed "
                "A, q, Ta:{}\n{}",
                flameEig.Lambda, dth, row(0), lawText);
            qText = std::format("q = cp Tu (sigma-1) s/Y_O1 = {:.6g} per unit "
                                "fuel",
                                kGammaAir / (kGammaAir - 1) * flameTu *
                                    pm.qOverCpTu());
            PartialPremixModel b = pm;
            b.phi = premixBurnPhi();
            yInText = std::format("Y_F = {:.6g}, Y_O = {:.6g}", b.yF1(),
                                  b.yO1());
        }
        if (flameEig.tab.tCut > 0)
            oracleText += std::format(
                "             cold cutoff: rate off below T/Tu = {:.8g} (the "
                "unburnt mixture's own rate is not negligible; issues.md "
                "72)\n",
                flameEig.tab.tCut);
        if (physics == Physics::LowMach) {
            // The run's bounds at the burnt state (u = sigma u_in, rho =
            // 1/sigma, mu(T_b)); the chemistry bound is printed at run time.
            const double muB = prandtl * flameModel.lambdaRatio(sig);
            const double dtCv = cfl * h / (sig * flameUIn);
            const double dtDf = 0.25 * h * h * (1.0 / sig) /
                                (muB * std::max(4.0 / 3.0, 1.0 / prandtl));
            return std::format(
                "case:\n"
                "  mode       {} (dns flame --mode premixed --physics lowmach)\n"
                "  model      {}"
                "             sigma = Tb/Tu = {:.6g}  beta = {:.6g}  theta = "
                "Ta/Tu = {:.6g}  Pr = {:.4g}\n"
                "  oracle     {}"
                "  solver     LOW-MACH (M -> 0): P0 constant, rho T = P0/R, "
                "MAC staggered, SSP-RK3 with a mass-flux projection per stage "
                "(SeparablePoisson: x ND, y/z periodic, one direct solve)\n"
                "  units      rho_u = S_L = R = 1, delta_D = 1, Tu = P0 = 1 "
                "(no acoustic scale), cp = 3.5, {}\n"
                "  transport  {}\n"
                "  grid       {} x {} x {}  h = {:.6g} = delta_th/{:.4g}  box "
                "{:.6g} x {:.6g}{}{}\n"
                "  flame      x_f0 = {:.6g} ({:.4g} delta_th fresh), {:.4g} "
                "delta_th burnt{}\n"
                "  boundary   x-lo inflow FACE: m = {:.4g} rho_u S_L, T = Tu, {} "
                "(convective + diffusive face flux); x-hi outflow p = 0, "
                "scalars upwind, no diffusion\n"
                "  run        tt = {:.6g} flame times  cfl = {:.3g}  dt est "
                "{:.3e} (convective {:.3e}, diffusive {:.3e}; chemistry "
                "--chem-dt {:g} / J at run time{})\n"
                "  label      {}\n"
                "  out        {}\n",
                d3 ? "3D planar premixed flame"
                   : grid.ny > 4 || flameWrinkle > 0
                       ? "2D premixed flame"
                       : "1D freely propagating premixed flame",
                modelText, sig, flameModel.beta, flameModel.theta(), prandtl,
                oracleText, qText,
                sutherland ? std::format("Sutherland mu(T), lambda(T), rho D(T), "
                                         "S/Tu = {:.6g} (Tu = {:.6g} K)",
                                         flameModel.sOverTu, tinfK)
                           : std::string("constant mu = Pr, lambda = cp, rho D = 1"),
                grid.nx, grid.ny, grid.nz, h, flameCells, grid.nx * h,
                grid.ny * h,
                d3 ? std::format(" x {:.6g}  [3D, y and z periodic, hz = h]",
                                 grid.nz * h)
                   : std::string("  [native 2D, y periodic]"),
                lmNxPad > 0 ? std::format("\n             nx rounded up by {} "
                                          "cells downstream (FFT size: nx/2 "
                                          "7-smooth)",
                                          lmNxPad)
                            : std::string(),
                flameXf0, flameUp, flameDown + lmNxPad / flameCells,
                flameWrinkle > 0
                    ? std::format("\n             wrinkle {:.4g} delta_th over "
                                  "modes 1..{} (shortest {:.4g} delta_th)",
                                  flameWrinkle, flameWrinkleModes,
                                  grid.ny * h / flameWrinkleModes / dth)
                    : std::string(),
                flameUIn, yInText, ft, cfl, std::min(dtCv, dtDf), dtCv, dtDf,
                lmChemDt,
                lmDtCell ? "; --dt-rule cell: and the per-cell positivity "
                           "bound 1 / max (2 C + D)"
                         : "",
                flameRunLabel(*this), outPrefix);
        }
        return std::format(
            "case:\n"
            "  mode       {} (dns flame "
            "--mode premixed)\n"
            "  model      {}"
            "             sigma = Tb/Tu = {:.6g}  beta = {:.6g}  theta = Ta/Tu "
            "= {:.6g}  Pr = {:.4g}\n"
            "  oracle     {}"
            "  units      rho_u = S_L = R = 1, delta_D = 1, Tu = p0 = "
            "1/(gamma M^2) = {:.6g}, {}\n"
            "  transport  {}\n"
            "  grid       {} x {} x {}  h = {:.6g} = delta_th/{:.4g}  box "
            "{:.6g} x {:.6g}{}\n"
            "  flame      x_f0 = {:.6g} ({:.4g} delta_th fresh + {} sink "
            "cells), {:.4g} delta_th burnt{}\n"
            "{}"
            "  boundary   x NSCBC: inflow u = {:.4g} S_L (etaIn {:.4g}), "
            "outflow p = p_b = p0 - {:.4g}; species carried in at {}\n"
            "             sink {} cells, amp {:.3g}\n"
            "  run        tt = {:.6g} flame times (tEnd = {:.6g})  cfl = {:.3g}  "
            "dt est {:.3e} (acoustic {:.3e}, thermal {:.3e})\n"
            "  label      {}\n"
            "  out        {}\n",
            d3 ? (flameTurb > 0 ? "3D premixed flame in synthetic turbulence"
                                : "3D planar premixed flame")
               : "1D freely propagating premixed flame",
            modelText, sig, flameModel.beta, flameModel.theta(), prandtl,
            oracleText, flameTu, qText,
            sutherland ? std::format("Sutherland lambda(T), S/Tu = {:.6g} "
                                     "(Tu = {:.6g} K)",
                                     flameModel.sOverTu, tinfK)
                       : std::string("constant mu = Pr, lambda = cp"),
            grid.nx, grid.ny, grid.nz, h, flameCells, grid.nx * h, grid.ny * h,
            d3 ? std::format(" x {:.6g}  [3D, y and z periodic, hz = h]",
                             grid.nz * h)
               : std::string("  [native 2D, y periodic]"),
            flameXf0, flameUp, nscbcSink, flameDown,
            flameWrinkle > 0
                ? std::format("\n             wrinkle {:.4g} delta_th over "
                              "modes 1..{} (shortest {:.4g} delta_th)",
                              flameWrinkle, flameWrinkleModes,
                              grid.ny * h / flameWrinkleModes / dth)
                : std::string(),
            turbText, flameUIn, nscbcEtaIn,
            flameUIn * flameUIn * (sig - 1), yInText, nscbcSink, nscbcSinkAmp,
            ft, tEnd, cfl, std::min(dtAc, dtTh), dtAc, dtTh,
            flameRunLabel(*this), outPrefix);
    }
    // Shared by both conv branches.
    const std::string patchLabel =
        hotPatch >= 1.0
            ? std::string("uniformly heated")
            : std::format("heated on a centred patch of side fraction {:.4g}"
                          " ({:.1f}% of the area); the rest sits at the lid"
                          " temperature", hotPatch,
                          100.0 * (grid.nz == 1 ? hotPatch
                                                : hotPatch * hotPatch));
    // The lateral faces, in the terms the run actually has: which axes closed
    // and what a closed one is (a no-slip adiabatic wall, ghost ring and all).
    const std::string sidesLabel =
        !closedSides
            ? std::string("periodic (unbounded layer sampled over one box)")
            : std::format("CLOSED: no-slip adiabatic walls on {} (+1 ghost "
                          "column each side); fluid extent unchanged",
                          grid.nz == 1 ? "x (z stays the exact 2D wrap)"
                                       : "x and z");
    if (sub == Subcommand::Conv && physics == Physics::Incompressible)
    {
        return std::format(
            "case:\n"
            "  mode       Rayleigh-Benard convection, BOUSSINESQ "
            "(incompressible)\n"
            "  grid       {} x {} x {} ({} fluid rows + 2 ghost)  "
            "box {:.6g} x {:.6g} x {:.6g}{}\n"
            "  walls      no-slip isothermal plates at T = {:+.3g} / {:+.3g} "
            "(drop 1 over H=1)\n"
            "  sides      {}\n"
            "  floor      {}\n"
            "  driving    Ra={:.6g}  Pr={:.4g}   (free-fall scaling: "
            "H = dT = g*alpha = 1)\n"
            "  derived    nu={:.6g}  kappa={:.6g}  Uff=1  Re_ff={:.6g}  t_ff=1  "
            "cfl={:.3g}\n"
            "  limit      no sound speed, no acoustic dt, no adiabatic "
            "gradient: Ra and Nu share the full drop\n"
            "  run        tt={:.6g} free-fall times (tEnd={:.6g})  kick={:.3g}  "
            "profile-dt={:.3g}\n"
            "  label      DNS{}\n"
            "  out        {}\n",
            grid.nx, grid.ny, grid.nz, grid.ny - 2,
            lxBox, convH, lzBox, grid.nz == 1 ? "  [native 2D]" : "",
            tBot, tTop, sidesLabel, patchLabel,
            rayleigh, prandtl, nu, kappaDiff, re, cfl,
            ft, tEnd, kick, profileDt,
            grid.nz == 1 ? " (2D: qualitative only, no vortex stretching)"
                         : "",
            outPrefix);
    }
    if (sub == Subcommand::Conv)
    {
        return std::format(
            "case:\n"
            "  mode       Rayleigh-Benard convection (no-slip isothermal y "
            "walls)\n"
            "  grid       {} x {} x {} ({} fluid rows + 2 ghost)  "
            "box {:.6g} x {:.6g} x {:.6g}{}\n"
            "  walls      floor T={:.6g}, lid T={:.6g} (imposed drop {:.6g} over "
            "H={:.6g})\n"
            "  sides      {}\n"
            "  floor      {}\n"
            "  gravity    g={:.6g} in -y; adiabatic drop (g/cp)H={:.6g} "
            "({:.1f}% of the imposed)\n"
            "  driving    Ra={:.6g} on the superadiabatic drop {:.6g}  Pr={:.4g}  "
            "Mach={:.3g}\n"
            "  derived    nu={:.6g}  kappa={:.6g}  Uff={:.6g}  Re_ff={:.6g}  "
            "t_ff={:.6g}  cfl={:.3g}\n"
            "  transport  {}\n"
            "  run        tt={:.6g} free-fall times (tEnd={:.6g})  kick={:.3g}  "
            "profile-dt={:.3g}\n"
            "  label      DNS{}\n"
            "  out        {}\n",
            grid.nx, grid.ny, grid.nz, grid.ny - 2,
            lxBox, convH, lzBox, grid.nz == 1 ? "  [native 2D]" : "",
            tBot, tTop, dtemp, convH, sidesLabel, patchLabel,
            gravity, dtemp - dtempSuper,
            dtemp > 0 ? 100.0 * (dtemp - dtempSuper) / dtemp : 0.0,
            rayleigh, dtempSuper, prandtl, mach,
            nu, prandtl > 0 ? nu / prandtl : 0.0, uFreeFall, re,
            uFreeFall > 0 ? convH / uFreeFall : 0.0, cfl,
            sutherland
                ? "Sutherland mu(T), kappa(T) = mu cp / Pr"
                : "constant mu, kappa",
            ft, tEnd, kick, profileDt,
            grid.nz == 1 ? " (2D: qualitative only, no vortex stretching)"
                         : "",
            outPrefix);
    }
    if (sub == Subcommand::Channel)
    {
        // RANS: the label names the closure and the mode, and a RANS mean
        // really IS 1D -- it never carries the 2D "qualitative" suffix.
        const bool ransOn = rans != RansModel::None;
        const bool implicitSteady =
            steadyTol > 0 && steadyScheme == SteadyScheme::Implicit;
        // The ramp's three constants, spelled as the help text spells them.
        static_assert(kImplicitCfl0 == 100 && kImplicitCflGrowth == 2 &&
                      kImplicitCflMax == 1e8);
        const std::string runLine =
            steadyTol > 0
                ? std::format("steady TOL={:.6g} cap {} iterations{}",
                              steadyTol, steadySteps,
                              implicitSteady ? ", implicit Newton (CFL 100 x2 "
                                               "-> 1e8, total mass pinned)"
                                             : "")
                : std::format("tt={:.6g} eddy turnovers (tEnd={:.6g})  "
                              "kick={:.3g}  profile-dt={:.3g}{}",
                              ft, tEnd, kick, profileDt,
                              compensated ? kCompensatedNote : "");
        const RansMode mode = steadyTol <= 0   ? RansMode::Urans
                              : implicitSteady ? RansMode::SteadyImplicitLine
                                               : RansMode::SteadyExplicit;
        const std::string label =
            ransOn     ? ransRunLabel(rans, mode)
            : lesLabel ? lesRunLabel(sgsModel, cs, grid.nz == 1)
                       : std::string("DNS") +
                             (grid.nz == 1
                                  ? " (2D: qualitative only, no vortex "
                                    "stretching)"
                                  : "");
        const std::string wallScalars =
            ransSst(rans) ? "k_wall = 0 (antimirrored ghost), omega_wall = 60 "
                            "nu/(beta1 d1^2) (TMR; Dirichlet image),\n"
                            "             "
                          : "nu~_wall = 0 (antimirrored ghost), ";
        const std::string turb =
            ransOn ? std::format("  turbulence {}mu_t = 0 on the wall face, "
                                 "Pr_t = {:.3g}, wall-resolved (y+_1 = {:.3g} "
                                 "est.)\n",
                                 wallScalars, kPrandtlTurb, chanH1 * reTau)
            : lesLabel
                ? std::format("  sgs        mu_t = rho (C Delta)^2 op(grad u), "
                              "Delta = vol^(1/3) per cell (2D: area^(1/2));\n"
                              "             odd image on both ghost rows: "
                              "mu_t = 0 on the wall face{}\n",
                              sgsModel == SgsModel::Smagorinsky
                                  ? "; smag: no wall damping"
                                  : "")
                : std::string();
        return std::format(
            "case:\n"
            "  mode       compressible turbulent channel (periodic x/z, "
            "no-slip y walls)\n"
            "  grid       {} x {} x {} ({} fluid rows + 2 ghost)  "
            "box {:.6g} x {:.6g} x {:.6g}{}\n"
            "  mesh       y tanh-stretch beta={:.4g}: first cell {:.3e} "
            "(y+ = {:.3g}), growth <= {:.4f}\n"
            "  physics    Re_tau={:.6g}  nu={:.6g}  Mach_bulk={:.3g}  "
            "(u_bulk est {:.3g}, p={:.6g})  cfl={:.3g}\n"
            "  walls      {} no-slip mirrors; force f=1 (constant), "
            "u_tau target 1\n"
            "{}"
            "  run        {}\n"
            "  label      {}\n"
            "  out        {}\n",
            grid.nx, grid.ny, grid.nz, grid.ny - 2,
            lxBox, 2.0, lzBox, grid.nz == 1 ? "  [native 2D]" : "",
            stretchBeta, chanH1, chanH1 * reTau, stretchGrowth,
            reTau, nu, mach, ubEst, pRef, cfl,
            wallIso ? "isothermal" : "ADIABATIC (no heat sink -- short "
                                     "transients only)",
            turb, runLine, label, outPrefix);
    }
    // A mesh or scene path can be any length, which is why this is std::format
    // and not a fixed buffer: snprintf TRUNCATED the report of exactly the runs
    // whose body most needed naming.
    const std::string bodyPath = bodyKind == BodyKind::Scene ? " " + scenePath
                                 : meshPath.empty()          ? std::string()
                                                             : " " + meshPath;
    std::string s = std::format(
        "case:\n"
        "  physics    {}\n"
        "  body       {}{}{}\n"
        "  far field  {}{}\n"
        "  grid       {} x {} x {}  h={:.6g}  L={:.6g}{}\n"
        "  placement  cx={:.6g} cy={:.6g} cz={:.6g} D={:.6g} ({:.1f} cells)\n"
        "  Re={:.6g}  nu={:.6g}  Mach={:.6g}  cfl={:.6g}  Cs={:.6g} ({})\n"
        "  ft={:.6g}  tEnd={:.6g}  fdt={:.6g}\n"
        "  mode       {}\n"
        "  out        {}\n",
        name(physics), name(bodyKind), bodyPath,
        cutFlux ? " [cut-flux]" : "",
        // c.far stays Nscbc internally on the O-grid (nothing on that path
        // reads it); the report names the boundary the run actually has.
        ogrid ? "o-grid outer row (characteristic LODI, gpu/ogrid_bc)"
              : name(far),
        open ?(farPot ? " (lateral open, potential-corrected)"
                       : " (lateral open)")
             : (farPot ? " (potential-corrected)" : ""),
        grid.nx, grid.ny, grid.nz, h, L, grid.nz == 1 ? "  [native 2D]" : "",
        placement.cx, placement.cy, placement.cz, placement.D,
        h > 0 ? placement.D / h : 0.0,
        re, nu, mach, cfl, cs,
        lesLabel ? (sgsModel == SgsModel::Sigma ? "LES sigma" : "LES smag")
        : rans != RansModel::None ? "RANS"
                                  : "DNS",
        ft, tEnd, fdt,
        compSteady      ? "steady pseudo-time (stop on the integrals, "
                          "--steady-steps cap)"
        : steadyTol > 0 ? "stationary (stop at steady residual, tEnd = give-up cap)"
        : compensated   ? std::string("time-march to tEnd") + kCompensatedNote
                        : "time-march to tEnd",
        outPrefix);
    // Derived outflow coefficient: printed only when asked for, so a default
    // case's --dry-run reads exactly as it always did. It CHANGES the boundary
    // arithmetic, so like the DNS/LES label it must be visible.
    if (nscbcAbc != NscbcAbc::YooIm)
        s += std::string("  nscbc      outflow transverse coefficient derived ")
             + (nscbcAbc == NscbcAbc::Modal ? "per transverse DCT mode"
                                            : "per face cell")
             + " (--nscbc-abc; --nscbc-beta caps it)\n";
    // The sink is an explicit energy sink and is reported like the DNS/LES
    // label: a run that carries one says so in its resolved case.
    if (nscbcSink > 0)
        s += std::format(
            "  nscbc      scale-selective sink: {}-cell band, sin^4 cutoff "
            "(~86% at 4dx, ~1% at 16dx), sigma_max = {:.3g} c/h\n",
            nscbcSink, nscbcSinkAmp);
    // Thermal model: printed only when a run actually asked for one, so a
    // mechanical case's --dry-run reads exactly as it always did.
    if (physics == Physics::Compressible &&
        (sutherland || bodyIsothermal || nscbcThermalInflow))
    {
        const double Tinf = mach > 0 ? 1.0 / (kGammaAir * mach * mach) : 0.0;
        s += std::format(
            "  thermal    visc={}  Pr={:.6g}  Tinf={:.6g} box ({:.6g} K)\n"
            "             body wall {}{}  nscbc inflow holds {}\n",
            sutherland ? "sutherland" : "constant", prandtl, Tinf, tinfK,
            bodyIsothermal ? "isothermal" : "adiabatic",
            bodyIsothermal ? " Twall/Tinf=" + std::to_string(twallRatio)
                           : std::string(),
            nscbcThermalInflow ? "T" : "rho");
    }
    if (ogrid && !ogridRevolved)
    {
        s += std::format(
            "  o-grid     BODY-FITTED: {} angular x {} radial (2 ghost rows)"
            " x {} spanwise\n"
            "             rFar={:.6g} ({:.6g} D)  beta={:.6g}  span={:.6g}  "
            "h1≈{:.3e} (nominal circle, beta={:.6g})\n"
            "             the wall is a MESH FACE -- no cut cells, no immersed"
            " body,\n"
            "             forces are an exact surface integral\n",
            grid.nx, grid.ny, grid.nz, ogridRFar,
            placement.D > 0 ? ogridRFar / placement.D : 0.0, ogridBeta,
            ogridSpan, ogridH1, ogridBeta);
        if (ogridTrip > 0)
            s += std::format("             trip {:g} U (solenoidal vortex, 1 D "
                             "behind the body, D/4 off the axis{})\n",
                             ogridTrip,
                             grid.nz > 1 ? ", centred in the span" : "");
    }
    if (ogridRevolved && ogridNphi > 0)
    {
        const double dphi = 2.0 * std::numbers::pi / ogridNphi;
        s += std::format(
            "  o-grid     REVOLVED about x: {} meridional (theta 0..pi, + 2 "
            "ghost columns past the axis) x {} radial\n"
            "             (2 ghost rows), FULL revolution {} azimuthal cells "
            "({:.4g} deg; the axis ghosts\n"
            "             copy k + {}); polygonal body, frontal area "
            "{:.4f} of the circle's\n"
            "             axis faces degenerate (S = 0, limit metric); qRef on "
            "pi D^2/4\n"
            "             odd-mode trip {}\n"
            "             initial state {}\n",
            grid.nx - 2, grid.ny, ogridNphi, dphi * 180.0 / std::numbers::pi,
            ogridNphi / 2, std::sin(dphi) / dphi,
            ogridTrip > 0 ? std::format("{:g} U (solenoidal, 1 D behind the "
                                        "body{})",
                                        ogridTrip,
                                        ogridTrip == kOgridTripDefault
                                            ? "; the default"
                                            : "")
                          : std::string("none (exactly pi-symmetric: no odd "
                                        "azimuthal modes)"),
            ogridInitFrom.empty()
                ? std::string(ramp > 0 ? "rest (quiet-start ramp)"
                                       : "the uniform freestream")
                : std::format("the converged wedge {}_final.bin revolved over "
                              "the {} planes (mesh checked against "
                              "{}_mesh.bin), then the trip",
                              ogridInitFrom, ogridNphi, ogridInitFrom));
    }
    else if (ogridRevolved)
    {
        s += std::format(
            "  o-grid     REVOLVED about x: {} meridional (theta 0..pi, + 2 "
            "ghost columns past the axis) x {} radial\n"
            "             (2 ghost rows), axisymmetric WEDGE {:g} deg (1 real "
            "plane + 2 ghost planes)\n"
            "             axis faces degenerate (S = 0, limit metric); qRef on "
            "pi D^2/4\n",
            grid.nx - 2, grid.ny, ogridWedgeDeg);
    }
    // RANS / compressible steady / the wedge: the label and what the closure
    // resolved (RANS-off Cartesian or z-extruded time marches print none of
    // these lines).
    if (const std::string label = flowRunLabel(*this); !label.empty())
        s += std::format("  label      {}\n", label);
    if (rans != RansModel::None || compSteady)
    {
        if (rans != RansModel::None)
        {
            s += std::format("  turbulence {}, Pr_t = {:.3g}\n",
                             ransInflowText(*this), kPrandtlTurb);
            if (!ogrid)
                s += ransSst(rans)
                         ? "  wall       immersed ghost wall: k_g = a^2 k_s "
                           "(even, quadratic), omega_g = (1+a)\n"
                           "             omega_w - a omega_s, omega_w = 60 "
                           "nu/(beta1 d_s^2) (an UNVERIFIED sub-cell choice),\n"
                           "             mu_t_g = -min(a mu_t_s, fluid face "
                           "neighbours' mu_t) (no face below mu)\n"
                         : "  wall       immersed ghost wall, the velocity's "
                           "linear image: nu~_g = -a nu~_s,\n"
                           "             mu_t_g = -min(a mu_t_s, fluid face "
                           "neighbours' mu_t) (no face below mu)\n";
            else
                s += ransSst(rans)
                         ? "  wall       k_wall = 0 (antimirrored ghost), "
                           "omega_wall = 60 nu/(beta1 d1^2) per wall face\n"
                           "             (TMR; Dirichlet image), mu_t = 0 on "
                           "the wall face\n"
                         : "  wall       nu~_wall = 0 (antimirrored ghost), "
                           "mu_t = 0 on the wall face\n";
            s += std::format(
                "  y+         first cell ~{:.3g} (estimate: bluff-body peak "
                "u_tau = 0.38 Re^-0.1 U; the run measures it)\n",
                ransYplusEstimate(ransFirstCellDistance(*this), placement.D,
                                  re));
        }
        // The ramp's constants, spelled as the help text spells them.
        static_assert(kNkCfl0 == 10 && kNkCflGrowth == 2 && kNkCflMax == 1e8);
        s += !compSteady ? std::string()
             : steadyScheme == SteadyScheme::Implicit
                 ? std::format("  run        steady TOL={:.6g} cap {} "
                               "iterations, implicit Newton-Krylov (CFL 10 "
                               "x2 -> 1e8, GMRES({}) + line-SGS on a "
                               "first-order Rusanov P)\n",
                               steadyTol, steadySteps, kNkRestart)
                 : std::format("  run        steady TOL={:.6g} cap {} "
                               "iterations, local time stepping cfl={:.3g}, "
                               "localDt every {}{}\n",
                               steadyTol, steadySteps, cfl, dtRecompute,
                               (steadySample == 200
                                    ? std::string()
                                    : std::format(", stop sample every {}",
                                                  steadySample)) +
                                   (ltsMax > 0
                                        ? std::format(", lts ratio <= {:g}",
                                                      ltsMax)
                                        : std::string()) +
                                   (precondK > 0
                                        ? std::format(", Weiss-Smith low-Mach "
                                                      "preconditioner K={:g}",
                                                      precondK)
                                        : std::string()));
        s += compSteady ? std::string()
                 : std::string("  run        URANS: SSP-RK3 on the global "
                               "dt (acoustic + per-cell RANS diffusive "
                               "bound)\n");
    }
    if (stretchBeta > 0)
    {
        s += std::format(
            "  mesh       y tanh-stretch beta={:.4g}: first interior cell "
            "{:.3e} (hy/{:.1f}), growth <= {:.4f}\n"
            "             (acoustic dt shrinks with the wall cell)\n",
            stretchBeta, stretchH1,
            stretchH1 > 0 ? grid.hy / stretchH1 : 0.0, stretchGrowth);
    }
    if (focalActive)
    {
        const char* ax[3] = {"x", "y", "z"};
        std::string per;
        for (int a = 0; a < 3; ++a)
        {
            if (focalHi[a] <= focalLo[a]) continue;
            per += std::format(
                "             {} physical [{:.4g},{:.4g}] blend {:.4g}: fine "
                "{:.3e} coarse {:.3e}\n",
                ax[a], focalLo[a], focalHi[a], focalW[a], focalHFine[a],
                focalHCoarse[a]);
        }
        s += std::format("  mesh       focal clustering ratio {:.4g}, growth "
                         "<= {:.4f}, dt factor {:.3f}\n",
                         focalRatio, focalGrowth, focalDtFactor);
        s += per;
    }
    if (rotor.active())
    {
        const char* ax[3] = {"x", "y", "z"};
        const double hFine = dns::hFine(*this);
        const double tRev = 2.0 * std::numbers::pi / std::abs(rotor.omega);
        const char* shaft = ax[rotorAxisIndex(rotor.axis)];
        s += std::format(
            "  rotor      SPINNING body: shaft along {} through the placement "
            "centre\n"
            "             tip Mach {:.4g} (freestream {:.4g}, sum {:.4g} < 1)  "
            "omega={:.6g}  R={:.6g}\n"
            "             t_rev={:.6g} -> {:.2f} revolutions in tEnd={:.6g}\n",
            shaft, rotorTipMach, mach, rotorTipMach + mach, rotor.omega,
            rotor.radius, tRev, tRev > 0 ? tEnd / tRev : 0.0, tEnd);
        if (rotorMrf)
            s += std::format(
                "             frame: MRF (frozen rotor, an APPROXIMATION): "
                "geometry static at\n"
                "             azimuth 0, wall moves at omega x r, rotating-"
                "frame equations in a\n"
                "             blended cylinder around the swept volume (zone "
                "printed at run time).\n"
                "             Exact for an axisymmetric body; time-accurate "
                "blade passage needs\n"
                "             --rotor-frame moving\n");
        else
            s += std::format(
                "             azimuth {} bins (natural {} = 2 pi R / h_fine); "
                "geometry is\n"
                "             precomputed per bin and replayed -- the time "
                "loop never rebuilds it\n",
                rotor.bins, rotorNaturalBins(rotor, hFine));
        s += std::format(
            "             wall is MOVING no-slip (omega x r, tip speed "
            "{:.6g}); run reports\n"
            "             thrust and shaft torque about {}\n",
            std::abs(rotor.omega) * rotor.radius, shaft);
        if (!ogrid)
            s += std::format(
                "             blade must be >= 4 fine cells thick "
                "(h_fine={:.3e}, D/h_fine={:.1f}):\n"
                "             >= {:.4g} in world units, or the two surfaces' "
                "mirror bands overlap\n",
                hFine, hFine > 0 ? placement.D / hFine : 0.0, 4.0 * hFine);
        else
            s += "             wall is a mesh face row: no mirror band, no "
                 "thickness gate\n";
        if (rotor.thicken > 0)
            s += std::format(
                "  blade      THICKENED: surface dilated outward by {:.4g} "
                "fine cells ({:.4g} world).\n"
                "             This is a DIFFERENT BODY -- chord grows by "
                "{:.4g}, edges round.\n"
                "             Report it with every result.\n",
                bladeThickenCells, rotor.thicken, 2.0 * rotor.thicken);
    }
    if (thinSeal)
        s += "  wall       THIN SECTIONS SEALED: a fluid/fluid face whose "
             "centre segment crosses\n"
             "             the body is a wall face of both cells (zero-"
             "thickness sheet; body\n"
             "             unchanged). Label: [thin sections sealed: sub-cell "
             "sheets imaged as\n"
             "             zero-thickness walls]\n";
    if (noslipY) s += "  walls      y walls no-slip (adiabatic mirror)\n";
    return s;
}

} // namespace dns
