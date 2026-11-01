// dns flame --equivalence end to end (model C2, docs/chem.md; masterplan
// real_bunsen S1): the rich two-reactant premixed jet's 1D flame on the
// compressible solver against the FP64 oracle premixedEigen2 that defines its
// S_L = 1, beside the single-reactant flame at the same sigma and grid.
//
// Case: CH4's phi 1.3 (s 3.99, Y_O2 0.233), sigma 6.88, beta 8, Sutherland
// at Tu 298 K, M 0.01, --inflow-speed 0.98, tt 20, window t in [10, 20]
// (gpu_flame1d's protocol), 7.5 and 15 cells per delta_th.
//   each run   exit 0; S flat (window halves agree to 1e-3); omegaEdge <
//              1e-10; yHits 0; Tmax/(sigma Tu) - 1 < 1e-3.
//   rich       |S_rich - S_single| <= 2e-3 at both grids (measured
//              2026-10-31: 7.5 cells 1.015612 vs 1.015058, 15 cells 1.002544
//              vs 1.002695 -- the single-reactant ladder's +1.5% / +0.27%;
//              with the S1b law, the --equivalence default since then:
//              1.015555 / 1.002132); |S_rich(15) - 1| < 5e-3.
//   Z          the mixture fraction (coupling function Y_F - Y_O/s) within
//              5e-4 of 1 over the box and the run (measured +-1.7e-4 at both
//              grids, 1.2e-4 at M 0.02, 2.4e-5 at tt 2: an FP32 transport walk
//              of the NON-power-of-two mass fractions, amplified by s/(s
//              Y_F,rem + Y_O2) = 13; the plan's 1e-5 is below it).
//
// `law` (masterplan real_bunsen S1b, model C2b: --sl-law gri30, the
// --equivalence default): the jet (phi 1.3) and --mix-phi 0.7 / 1.05 / 1.2
// (the rim's air-diluted mixtures; A, q, Ta the jet's) at 15 cells, inflow
// 0.98 x the oracle ratio r = premixedSpeedRatio(phi, phiRateLaw) (= GRI-3.0's
// S_L(phi)/S_L(1.3) at these nodes: 0.8221 / 1.6336 / 1.4175), plus 1.2 on
// the uncorrected step (--sl-law onestep, r 0.9174: the wrong-sign S1 model).
//   each run   as above (flat, edge, yHits 0, Tmax/Tb(phi) - 1 < 1e-3);
//              |S/r - 1| < 5e-3 (measured 2026-10-31: jet +0.21%, 0.7
//              +0.39% (+0.08% at 30 cells: discretization), 1.05 +0.14%, 1.2
//              +0.13%, 1.2 uncorrected +0.30%); Z within 5e-4 of the
//              premix's Z(phi).
//   ratio      |(S_mix/S_jet)/r - 1| < 3e-3: the law's shape on the GPU
//              (FP32 kernel) is the FP64 oracle's (measured +1.8e-3 / -7.2e-4
//              / -8.1e-4 at 0.7 / 1.05 / 1.2). Not tighter: the bias does not
//              cancel between sigma 5.25 and 6.88, and the jet's S carries
//              the FP32 Z walk through C (d ln S/dZ = -7.3 at phi 1.3: 30
//              cells read 0.9965 with Z up to 1 + 4.6e-4, -2.3e-3 vs the
//              uncorrected step; docs/chem.md "Model C2b").
//
// Usage: test_gpu_flame_rich1d [law]. No HIP device -> skip (77).
#include "dns/chem.hpp"
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <string>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;

struct Flame
{
    bool ok = false;
    double SL = NAN, SLa = NAN, SLb = NAN, edge = 0, hits = 0, tmax = 0;
    double zMin = NAN, zMax = NAN;
};

constexpr double kSigma = 6.88, kMach = 0.01, kTw0 = 10, kTw1 = 20;

// sigma: the burnt premix's Tb/Tu (Tmax gate); zRef: its mixture fraction.
Flame runCase(const std::string& name, const std::string& args,
              double sigma = kSigma, double zRef = 1.0)
{
    Flame f;
    const std::string prefix = "/tmp/dns_test_rich1d_" + name;
    const std::string cmd =
        "\"" DNS_BIN "\" flame --expansion 6.88 --visc sutherland --tinf 298 "
        "--mach 0.01 --inflow-speed 0.98 --tt 20 --emit stats --quiet " +
        args + " --out " + prefix;
    const int rc = valtest::runBinary(cmd);
    check(rc == 0, name + ": run exits 0");
    if (rc != 0) return f;
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    check(tbl.has_value(), name + ": stats readable");
    if (!tbl) return f;
    const auto& t = tbl->col("t");
    const auto& sl = tbl->col("SL");
    const bool z = std::ranges::find(tbl->names, "zMin") != tbl->names.end();
    double sa = 0, sb = 0;
    int na = 0, nb = 0;
    f.zMin = z ? HUGE_VAL : NAN;
    f.zMax = z ? -HUGE_VAL : NAN;
    for (std::size_t i = 0; i < t.size(); ++i) {
        f.hits += tbl->col("yHits")[i];
        if (z) {
            f.zMin = std::min(f.zMin, tbl->col("zMin")[i]);
            f.zMax = std::max(f.zMax, tbl->col("zMax")[i]);
        }
        if (t[i] < kTw0) continue;
        f.edge = std::max(f.edge, tbl->col("omegaEdge")[i]);
        f.tmax = std::max(f.tmax, tbl->col("Tmax")[i]);
        if (t[i] < 0.5 * (kTw0 + kTw1)) {
            sa += sl[i];
            ++na;
        } else {
            sb += sl[i];
            ++nb;
        }
    }
    if (na == 0 || nb == 0) return f;
    f.SLa = sa / na;
    f.SLb = sb / nb;
    f.SL = (sa + sb) / (na + nb);
    f.ok = std::isfinite(f.SL);
    const double Tu = 1 / (1.4 * kMach * kMach);
    std::printf("  %-10s S %.6f (%+.3f%%)  halves %.6f / %.6f  omegaEdge "
                "%.2e  yHits %g  Tmax/(sigma Tu) - 1 %+.2e",
                name.c_str(), f.SL, 100 * (f.SL - 1), f.SLa, f.SLb, f.edge,
                f.hits, f.tmax / (sigma * Tu) - 1);
    if (z)
        std::printf("  Z - Zref in [%+.2e, %+.2e]", f.zMin - zRef,
                    f.zMax - zRef);
    std::printf("\n");
    check(f.ok, name + ": stats finite");
    check(std::abs(f.SLa - f.SLb) < 1e-3, name + ": S flat (halves to 1e-3)");
    check(f.edge < 1e-10, name + ": omegaEdge < 1e-10");
    check(f.hits == 0, name + ": yHits 0");
    check(f.tmax / (sigma * Tu) - 1 < 1e-3, name + ": Tmax/(sigma Tu) - 1 < 1e-3");
    return f;
}

// The C2b law end to end (header).
int lawCases()
{
    dns::PartialPremixModel pm; // phi 1.3, s 3.99, Y_O2 0.233
    pm.sigma = kSigma;
    pm.sutherland = true;
    pm.sOverTu = 110.4 / 298;
    const auto law = dns::phiRateLaw(pm);
    check(law.has_value(), "law: phiRateLaw resolves");
    if (!law) return dnstest::finish();
    const std::string base = "--cells-per-delta 15 --equivalence 1.3";
    const Flame jet = runCase("law_jet", base + " --inflow-speed 0.98");
    check(std::abs(jet.SL - 1) < 5e-3, "law_jet: |S - 1| < 5e-3");
    struct Mix
    {
        double phi;
        bool onestep;
    };
    for (const Mix m : {Mix{0.7, false}, Mix{1.05, false}, Mix{1.2, false},
                        Mix{1.2, true}}) {
        const auto r = dns::premixedSpeedRatio(pm, m.phi,
                                               m.onestep ? nullptr : &*law);
        check(r.has_value(), "law: oracle ratio resolves");
        if (!r) continue;
        dns::PartialPremixModel b = pm;
        b.phi = m.phi;
        const double zRef = pm.mixtureFraction(b.yF1(), b.yO1());
        const std::string name =
            std::format("law_mix{:g}{}", m.phi, m.onestep ? "_onestep" : "");
        const Flame f = runCase(
            name,
            std::format("{} --mix-phi {:g} --inflow-speed {:.6g}{}", base,
                        m.phi, 0.98 * *r,
                        m.onestep ? " --sl-law onestep" : ""),
            pm.sigmaAt(m.phi), zRef);
        const double q = (f.SL / jet.SL) / *r - 1;
        std::printf("  %-18s oracle r %.6f  S/r - 1 %+.3e  (S/S_jet)/r - 1 "
                    "%+.3e\n",
                    name.c_str(), *r, f.SL / *r - 1, q);
        check(std::abs(f.SL / *r - 1) < 5e-3, name + ": |S/r - 1| < 5e-3");
        check(std::abs(q) < 3e-3, name + ": |(S/S_jet)/r - 1| < 3e-3");
        check(std::abs(f.zMin - zRef) < 5e-4 && std::abs(f.zMax - zRef) < 5e-4,
              name + ": Z within 5e-4 of the premix's");
    }
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    if (argc > 1 && std::string(argv[1]) == "law") return lawCases();
    const Flame s7 = runCase("single7.5", "--cells-per-delta 7.5");
    const Flame r7 = runCase("rich7.5", "--cells-per-delta 7.5 --equivalence 1.3");
    const Flame s15 = runCase("single15", "--cells-per-delta 15");
    const Flame r15 = runCase("rich15", "--cells-per-delta 15 --equivalence 1.3");
    std::printf("  rich - single: 7.5 cells %+.3e, 15 cells %+.3e\n",
                r7.SL - s7.SL, r15.SL - s15.SL);
    check(std::abs(r7.SL - s7.SL) <= 2e-3 && std::abs(r15.SL - s15.SL) <= 2e-3,
          "rich S within 2e-3 of the single-reactant flame at both grids");
    check(std::abs(r15.SL - 1) < 5e-3, "|S_rich(15 cells) - 1| < 5e-3");
    for (const Flame* f : {&r7, &r15})
        check(std::abs(f->zMin - 1) < 5e-4 && std::abs(f->zMax - 1) < 5e-4,
              "Z within 5e-4 of 1 (the FP32 walk; measured 1.7e-4)");
    return dnstest::finish();
}
