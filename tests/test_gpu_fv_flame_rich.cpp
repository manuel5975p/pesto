// Air-methane (models C2, C2b: `--equivalence`, `--sl-law gri30`) on the compressible FV
// family (docs/fvcomp.md "Air-methane", masterplan_multiphysics M2). The generic gates
// of the C2b module (FvRef, numberings, rest, pow2 Y, Z == tracer, inert) are
// gpu_fv_matrix's; here the structured counterpart and the physics:
//   R1 FV vs the structured two-reactant kernels (kSpeciesStage2, kChemCellPass2 /
//      kChemCellPass2Phi) on a uniform periodic box, same dt, 400 steps, a burning rich
//      slab in a stratified premix (phi 1.17..1.3: C(xi) 1..3; A / 10: at the flame's A the
//      non-equilibrium slab explodes, T 2 Tb, chaotic in all three solvers): rounding,
//      <= 3 x the FP32 run's own distance from FP64; C2 (no law) and C2b
//   R2 the law's on-device rate: FP64 omega vs a host restatement of
//      phiRateLn + arrheniusRate2 per cell, and C(xi) = 1 at the jet's xi
//   rich1d (`test_gpu_fv_flame_rich rich1d`): the binary route, the jet (phi 1.3) and
//      --mix-phi 1.2 at 15 cells, FP64: S within 5e-3 of the oracle ratio, Z within 5e-4
//      of the premix's, yHits 0, the law's shape (S_mix/S_jet)/r within 3e-3
//      (gpu_flame_rich1d_law's gates)
//   les (`test_gpu_fv_flame_rich les`): 3D laminar LES (sigma 1.35, Smagorinsky 0.17)
//      vs the DNS on the same 3D box, C2b, FP64: sigma |S_LES/S_DNS - 1| < 1e-6 (D_sigma
//      of a planar field is 0); Smagorinsky's |S| of the dilatation is not: faster, < 5%
#include "fv_comp_test_util.hpp"

#include "dns/chem.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <cstring>
#include <unistd.h>

using namespace dnstest::fvc;
using dnstest::check;

namespace {

// The structured gpu_flame_rich1d model: CH4's phi 1.3, sigma 6.88, beta 8.
PartialPremixModel jetModel()
{
    PartialPremixModel pm;
    pm.sigma = 6.88;
    return pm;
}

struct Box
{
    Grid g;
    Transport tr;
    double Tu = 0, q = 0, lnA = 0, Ta = 0;
    CompState ic;
    std::vector<double> ry[2];
};

// A uniform periodic 128 x 32 box at M 0.05, a hot slab across x in a premix whose
// air dilution a(y) = 0.05 (1 + cos(2 pi y/Ly)) spans xi from the jet's (phi 1.3)
// to phi ~1.17 (C 1..2.7); inside the slab the deficient reactant is consumed by the progress.
Box richBox(const PartialPremixModel& pm, double lambda2, double dth)
{
    Box b;
    const double M = 0.05;
    b.Tu = 1.0 / (kGamma * M * M);
    const double h = dth / 12.0;
    b.g = Grid::make(128, 32, 1, 128 * h, 32 * h, 1.0);
    b.tr = Transport::air(pm.Pr, b.Tu, 288.15, pm.Pr, kGamma, 1.0, false);
    b.q = b.tr.cp() * b.Tu * pm.qOverCpTu();
    b.lnA = std::log(lambda2 / 10.0); // a tenth of the flame's A: the non-equilibrium slab must not explode
    b.Ta = pm.theta() * b.Tu;
    b.ic = CompState::zeros(b.g);
    b.ry[0].assign(b.g.size(), 0.0);
    b.ry[1].assign(b.g.size(), 0.0);
    const double Lx = b.g.nx * b.g.hx, Ly = b.g.ny * b.g.hy, w = 1.5 * dth, sig = pm.sigma;
    for (int j = 0; j < b.g.ny; ++j)
        for (int i = 0; i < b.g.nx; ++i) {
            const std::size_t c = b.g.idx(i, j, 0);
            const double x = (i + 0.5) * b.g.hx, y = (j + 0.5) * b.g.hy;
            const double f = 0.3 * Lx + 2 * h * std::cos(2 * std::numbers::pi * y / Ly);
            const double t = 1.0 + 0.5 * (sig - 1.0) * (std::tanh((x - f) / w) - std::tanh((x - 0.7 * Lx) / w));
            const double sp = (sig - t) / (sig - 1.0); // progress: 1 fresh, 0 burnt
            const double a = 0.05 * (1.0 + std::cos(2 * std::numbers::pi * y / Ly));
            const double yF = (1.0 - a) * pm.yF1(), yO = (1.0 - a) * pm.yO1() + a * pm.yO2;
            const double e = (1.0 - sp) * std::min(yF, yO / pm.s);
            const double rho = 1.0 / t;
            b.ic.rho[c] = rho;
            b.ic.E[c] = b.Tu / (kGamma - 1.0);
            b.ry[0][c] = rho * (yF - e);
            b.ry[1][c] = rho * (yO - pm.s * e);
        }
    return b;
}

// R1: one model (law null = C2) through both solvers.
void structuredCase(const char* name, const PartialPremixModel& pm, const PhiRateLaw<double>* law, double lambda2,
                    double dth)
{
    const Box b = richBox(pm, lambda2, dth);
    constexpr int steps = 400;
    auto dev = gpu::DeviceComp::make(b.g, b.tr);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !red) std::exit(EXIT_FAILURE);
    gpu::SpeciesSetup ss;
    ss.nY = 2;
    ss.fuel = 0;
    ss.ox = 1;
    ss.stoich = pm.s;
    ss.lnA = b.lnA;
    ss.Ta = b.Ta;
    ss.q = b.q;
    ss.invPr = 1.0 / pm.Pr;
    if (law) ss.phiRate = *law;
    if (!dev->enableSpecies(ss)) std::exit(EXIT_FAILURE);
    auto ds = gpu::DeviceCompState::make(b.g.size(), 0, false, 2);
    if (!ds || ds->upload(b.ic) != hipSuccess || ds->uploadSpecies(0, b.ry[0]) != hipSuccess ||
        ds->uploadSpecies(1, b.ry[1]) != hipSuccess)
        std::exit(EXIT_FAILURE);
    FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(b.g), periodicSpec(b.g), CellOrder::Morton), "box");
    const std::vector<FvBc> bc = bcFor(fm, [](const MeshPatch&) { return FvBc{}; });
    FvChem chem{.nY = 2, .fuel = 0, .lnA = b.lnA, .Ta = b.Ta, .q = b.q, .invPr = 1.0 / pm.Pr, .ox = 1, .stoich = pm.s};
    if (law) chem.law = *law;
    constexpr int nV = kFvMeanVars + 2;
    const std::vector<double>* src[nV] = {&b.ic.rho, &b.ic.mx, &b.ic.my, &b.ic.mz, &b.ic.E, &b.ry[0], &b.ry[1]};
    FvState s = FvState::zeros(fm.nC, nV), st = FvState::zeros(fm.nC, nV);
    for (int r = 0; r < nV; ++r)
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) s.var(r)[c] = (*src[r])[std::size_t(fm.srcCell[c])];
    // dt: half the structured flow bound, and the chemistry's 0.5 / max J of the IC (the
    // runners' rule; C(xi) up to 3 makes it bind where the flow bound alone left C2 stable).
    double dtChem = 0.0;
    {
        DeviceFvComp d = orDie(DeviceFvComp::make(fm, {.tr = b.tr, .bc = bc, .precision = FvPrecision::Fp64, .chem = chem}),
                               "fv");
        orDie(d.upload(s), "upload");
        dtChem = orDie(d.chemDt(0.5), "chemDt");
    }
    const float dt = std::min(0.5f * float(dev->stableDt(*ds, *red, 0.4)), float(dtChem));
    for (int n = 0; n < steps; ++n) dev->step(*ds, gpu::Real(dt));
    auto ho = CompState::zeros(b.g);
    std::vector<double> yo[2];
    if (ds->download(ho) != hipSuccess || ds->downloadSpecies(0, yo[0]) != hipSuccess ||
        ds->downloadSpecies(1, yo[1]) != hipSuccess)
        std::exit(EXIT_FAILURE);
    const std::vector<double>* hv[nV] = {&ho.rho, &ho.mx, &ho.my, &ho.mz, &ho.E, &yo[0], &yo[1]};
    for (int r = 0; r < nV; ++r)
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) st.var(r)[c] = (*hv[r])[std::size_t(fm.srcCell[c])];
    FvState o32, o64;
    for (const FvPrecision pr : {FvPrecision::Fp32, FvPrecision::Fp64}) {
        DeviceFvComp d = orDie(DeviceFvComp::make(fm, {.tr = b.tr, .bc = bc, .precision = pr, .chem = chem}), "fv");
        orDie(d.upload(s), "upload");
        for (int n = 0; n < steps; ++n) d.step(double(dt));
        orDie(d.download(pr == FvPrecision::Fp32 ? o32 : o64), "download");
    }
    double ox1 = 0.0, ox0 = 0.0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        ox1 += o64.var(6)[c];
        ox0 += s.var(6)[c];
    }
    const double a = relDiff(o32.q, st.q, nV, fm.nC), own = relDiff(o32.q, o64.q, nV, fm.nC),
                 st64 = relDiff(st.q, o64.q, nV, fm.nC);
    std::printf("  %s: %d x %d, %d steps of %.3e (chem bound %.3e), oxidizer burnt %.3f%%; FV FP32 vs structured %.3e, "
                "structured vs FV FP64 %.3e, FV FP32 vs FP64 %.3e\n",
                name, b.g.nx, b.g.ny, steps, double(dt), dtChem, 100.0 * (1.0 - ox1 / ox0), a, st64, own);
    check(ox1 < 0.995 * ox0, std::format("{}: the slab burns (>= 0.5% of the oxidizer)", name));
    check(a <= 3.0 * own, std::format("{}: FV FP32 vs structured {:.2e} <= 3 x own {:.2e}", name, a, own));
    check(st64 <= 3.0 * own, std::format("{}: structured vs FV FP64 {:.2e} <= 3 x own {:.2e}", name, st64, own));
}

void gateStructured(const PartialPremixModel& pm, const PhiRateLaw<double>& law, double lambda2, double dth)
{
    std::printf("== R1 FV vs the structured two-reactant kernels, uniform periodic, stratified premix, same dt\n");
    structuredCase("C2 (one-step)", pm, nullptr, lambda2, dth);
    structuredCase("C2b (GRI-3.0 law)", pm, &law, lambda2, dth);
}

// R2: the device's FP64 omega against phiRateLn + arrheniusRate2 restated per cell.
void gateRate(const PartialPremixModel& pm, const PhiRateLaw<double>& law, double lambda2, double dth)
{
    std::printf("== R2 the C2b rate on the device vs its host restatement, FP64\n");
    const Box b = richBox(pm, lambda2, dth);
    FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(b.g), periodicSpec(b.g), CellOrder::Morton), "box");
    const std::vector<FvBc> bc = bcFor(fm, [](const MeshPatch&) { return FvBc{}; });
    FvChem chem{.nY = 2, .fuel = 0, .lnA = b.lnA, .Ta = b.Ta, .q = b.q, .invPr = 1.0 / pm.Pr, .ox = 1, .stoich = pm.s};
    chem.law = law;
    constexpr int nV = kFvMeanVars + 2;
    const std::vector<double>* src[nV] = {&b.ic.rho, &b.ic.mx, &b.ic.my, &b.ic.mz, &b.ic.E, &b.ry[0], &b.ry[1]};
    FvState s = FvState::zeros(fm.nC, nV);
    for (int r = 0; r < nV; ++r)
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) s.var(r)[c] = (*src[r])[std::size_t(fm.srcCell[c])];
    DeviceFvComp d = orDie(DeviceFvComp::make(fm, {.tr = b.tr, .bc = bc, .precision = FvPrecision::Fp64, .chem = chem}), "fv");
    orDie(d.upload(s), "upload");
    FvState q;
    std::vector<double> om;
    orDie(d.speciesFields(q, om), "species");
    double e = 0.0, cMin = HUGE_VAL, cMax = 0.0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double rho = s.var(0)[c], yf = s.var(5)[c] / rho, yo = s.var(6)[c] / rho;
        const double T = (kGamma - 1.0) * s.var(4)[c] / rho; // R = 1, at rest
        const double lnC = phiRateLn(law, yf, yo);
        cMin = std::min(cMin, std::exp(lnC));
        cMax = std::max(cMax, std::exp(lnC));
        const double w = arrheniusRate2(b.lnA + lnC, b.Ta, rho, yf, yo, T);
        e = std::max(e, std::abs(om[c] - w) / std::max(std::abs(w), 1e-300));
    }
    const double cJet = std::exp(phiRateLn(law, pm.yF1(), pm.yO1()));
    std::printf("  omega vs restatement: max relative %.2e over %d cells; C(xi) in [%.4g, %.4g]; C(jet) - 1 = %.2e\n", e,
                fm.nC, cMin, cMax, cJet - 1.0);
    check(e <= 1e-13, "C2b omega = arrheniusRate2(lnA + phiRateLn) to 1e-13");
    check(cMax / cMin > 2.0, "the stratified box spans C(xi) (max / min > 2)");
    check(std::abs(cJet - 1.0) <= 1e-12, "C(jet) = 1 (the law's normalisation)");
}

struct Flame
{
    bool ok = false;
    double SL = NAN, SLa = NAN, SLb = NAN, hits = 0, zMin = NAN, zMax = NAN;
};

constexpr double kTw0 = 10, kTw1 = 20;

// gpu_flame_rich1d's protocol through the FV binary route.
Flame runFlame(const std::string& name, const std::string& args)
{
    Flame f;
    const std::string prefix = "/tmp/dns_test_fvrich_" + name + "_" + std::to_string(::getpid());
    const std::string cmd = "\"" DNS_BIN "\" flame --solver fv --equivalence 1.3 --expansion 6.88 --visc sutherland "
                            "--tinf 298 --mach 0.01 --inflow-speed 0.98 --tt 20 --emit stats --quiet " +
                            args + " --out " + prefix;
    const int rc = valtest::runBinary(cmd);
    check(rc == 0, name + ": run exits 0");
    if (rc != 0) return f;
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    std::filesystem::remove(prefix + "_stats.csv");
    std::filesystem::remove(prefix + ".json");
    check(tbl.has_value(), name + ": stats readable");
    if (!tbl) return f;
    const auto& t = tbl->col("t");
    const auto& sl = tbl->col("SL");
    double sa = 0, sb = 0;
    int na = 0, nb = 0;
    f.zMin = HUGE_VAL;
    f.zMax = -HUGE_VAL;
    for (std::size_t i = 0; i < t.size(); ++i) {
        f.hits += tbl->col("yHits")[i];
        f.zMin = std::min(f.zMin, tbl->col("zMin")[i]);
        f.zMax = std::max(f.zMax, tbl->col("zMax")[i]);
        if (t[i] < kTw0) continue;
        (t[i] < 0.5 * (kTw0 + kTw1) ? sa : sb) += sl[i];
        ++(t[i] < 0.5 * (kTw0 + kTw1) ? na : nb);
    }
    if (na == 0 || nb == 0) return f;
    f.SLa = sa / na;
    f.SLb = sb / nb;
    f.SL = (sa + sb) / (na + nb);
    f.ok = std::isfinite(f.SL);
    std::printf("  %-12s S %.6f  halves %.6f / %.6f  yHits %g  Z in [%.7f, %.7f]\n", name.c_str(), f.SL, f.SLa, f.SLb,
                f.hits, f.zMin, f.zMax);
    check(f.ok, name + ": stats finite");
    check(std::abs(f.SLa - f.SLb) < 1e-3, name + ": S flat (halves to 1e-3)");
    check(f.hits == 0, name + ": yHits 0");
    return f;
}

int rich1d()
{
    PartialPremixModel pm = jetModel();
    pm.sutherland = true;
    pm.sOverTu = Transport::sutherlandK / 298.0;
    const auto law = phiRateLaw(pm);
    check(law.has_value(), "phiRateLaw resolves");
    if (!law) return dnstest::finish();
    const auto r12 = premixedSpeedRatio(pm, 1.2, &*law);
    check(r12.has_value(), "oracle ratio at phi 1.2");
    if (!r12) return dnstest::finish();
    PartialPremixModel b = pm;
    b.phi = 1.2;
    const double zJet = pm.mixtureFraction(pm.yF1(), pm.yO1()), zMix = pm.mixtureFraction(b.yF1(), b.yO1());
    const Flame jet = runFlame("jet", "--cells-per-delta 15 --fv-precision fp64");
    const Flame mix = runFlame("mix1.2", "--cells-per-delta 15 --fv-precision fp64 --mix-phi 1.2");
    if (!jet.ok || !mix.ok) return dnstest::finish();
    const double ratio = (mix.SL / jet.SL) / *r12 - 1.0;
    std::printf("  jet S - 1 %+.3e; mix S/r - 1 %+.3e (r %.6f); (S_mix/S_jet)/r - 1 %+.3e; Z - Z(phi): jet [%+.2e, "
                "%+.2e], mix [%+.2e, %+.2e]\n",
                jet.SL - 1.0, mix.SL / *r12 - 1.0, *r12, ratio, jet.zMin - zJet, jet.zMax - zJet, mix.zMin - zMix,
                mix.zMax - zMix);
    check(std::abs(jet.SL - 1.0) < 5e-3, "jet: |S - 1| < 5e-3 at 15 cells");
    check(std::abs(mix.SL / *r12 - 1.0) < 5e-3, "mix 1.2: |S/r - 1| < 5e-3");
    check(std::abs(ratio) < 3e-3, "the law's shape: |(S_mix/S_jet)/r - 1| < 3e-3");
    check(std::max(std::abs(jet.zMin - zJet), std::abs(jet.zMax - zJet)) < 5e-4, "jet: Z within 5e-4 of Z(1.3)");
    check(std::max(std::abs(mix.zMin - zMix), std::abs(mix.zMax - zMix)) < 5e-4, "mix: Z within 5e-4 of Z(1.2)");
    return dnstest::finish();
}

int les()
{
    const std::string box = "--cells-per-delta 7.5 --ny 8 --nz 8 --fv-precision fp64";
    const Flame dnsF = runFlame("dns3d", box);
    const Flame sig = runFlame("les-sigma", box + " --les 1.35 --sgs sigma");
    const Flame sma = runFlame("les-smag", box + " --les 0.17 --sgs smag");
    if (!dnsF.ok || !sig.ok || !sma.ok) return dnstest::finish();
    std::printf("  S_LES/S_DNS - 1: sigma %+.3e, Smagorinsky 0.17 %+.3e (DNS S - 1 %+.3e)\n", sig.SL / dnsF.SL - 1.0,
                sma.SL / dnsF.SL - 1.0, dnsF.SL - 1.0);
    // sigma: D_sigma of a planar field is 0 up to the cells' summation-order noise.
    // Smagorinsky: |S| of the dilatation is not, mu_t thickens the laminar flame (the
    // structured flame's reason for an explicit --sgs): measured, its sign checked.
    check(std::abs(sig.SL / dnsF.SL - 1.0) < 1e-6, "laminar LES sigma: S within 1e-6 of the DNS");
    check(sma.SL > dnsF.SL && sma.SL / dnsF.SL - 1.0 < 0.05, "laminar LES Smagorinsky 0.17: faster than the DNS, < 5%");
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();
    if (argc > 1 && std::string(argv[1]) == "rich1d") return rich1d();
    if (argc > 1 && std::string(argv[1]) == "les") return les();
    const PartialPremixModel pm = jetModel();
    const auto law = orDie(phiRateLaw(pm), "law");
    const PremixedEigen eig = orDie(premixedEigen2(pm), "oracle");
    gateStructured(pm, law, eig.Lambda, eig.dthOverDD);
    gateRate(pm, law, eig.Lambda, eig.dthOverDD);
    return dnstest::finish();
}
