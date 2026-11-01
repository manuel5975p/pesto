// dns flame --mode bunsen --physics lowmach end to end (masterplan
// real_bunsen S5; docs/chem.md "Low-Mach Bunsen"). Driven through the dns
// binary.
//
//   g2a      the G2a burner (sigma 5 beta 8 C0, W 12 delta_th, U 5, burnt
//            coflow, 8 cells, tt 30, window [20, 25]) on the low-Mach solver
//            (--length 49: the 392 x 288 mesh of the compressible run and of
//            reactingFoam) against the compressible solver at M 0.01 and 0.02
//            (--nscbc-tinflow --nscbc-etain 1000, D10's foam-agreeing
//            variant), M^2-extrapolated, and against reactingFoam's G2a
//            (foam_mclin_c8, the same window, tools/chem/foam_flame.py
//            compare G2a; constants below). Gates (measured 2026-10-31):
//            low-Mach steady (halves to 1e-3), burn = fuel inflow to 1e-3,
//            Tmax = sigma, yHits 0; SLflank, mFlank, burnRatio within 1% of
//            the compressible M -> 0; xTip and sinRatio within 3% of it (the
//            plan's 1% is NOT met: -1.9% / -2.6%, the compressible's own gap
//            to foam is +2.4% / +2.2% while the low-Mach sits +0.5% / -0.4%
//            from foam -- docs, issues LM9); xTip and sinRatio within 1% of
//            foam, SLflank within 1.5%.
//   extrude  the low-Mach burner at nz 8 (z periodic, the IC extruded) equals
//            the 2D run byte for byte: the mid-span frames and every stats
//            column but mass, massErr (box totals), dtDiff (the bound itself)
//            and ST (3D only); C0 and
//            the --equivalence burner. Byte identity needs the same dt: the
//            3D diffusive bound is 1/6 h^2/D (positivity; 2D 1/4), so the
//            runs pin dt by a binding convective bound (--cfl 0.1; the
//            convective rate of an extruded field is the 2D one exactly)
//            under --dt-rule bounds (S9: the burner's default cell rule sums
//            the cell's diffusion over 3 axes in 3D, 2 in 2D -- a different
//            dt by construction).
//   rich     the partially premixed slot (phi 1.3 C2b, sigma 6.88,
//            Sutherland 298 K, Pr 0.52; W 4, U 3, 1.5 delta_th rich-product
//            pilots at the jet speed, air coflow, 8 cells, length 50, tt 40,
//            window [30, 40]): two flames (burnDif > 0.1 burn), preFrac in
//            [0.72, 0.80], the outer cone closed (xTipOut finite, steady to
//            1e-3, past the inner tip), burnRatio within 1e-3 of 1, Z in [0,
//            1] +- 1e-5, yHits 0. (The plan's W 12, U 5, 3 delta_th pilots
//            has its outer tip beyond ~180 delta_th: the mixing length ~ (W/2
//            + pilot)^2 U / D -- measured open at 45, still growing at 90.)
//   round    3D round nozzle smoke (nz 64, D 8 delta_th, 4 cells, the
//            --equivalence burner with pilots, the default dt rule, tt 3): exits 0,
//            the axial-plane cone measured (xTip finite, base anchored), T
//            <= sigma_st and yHits 0 (LM10: at the S8 default dt the 4-cell
//            tip was not bounded), volumes written for every channel, findex in
//            [-1, 1].
//
//   bickley  (S6) the inert 2D planar jet on open laterals vs Bickley (Re 60),
//            the clearance halved (no confinement), periodic (confined).
//   openrich (S6) the partially premixed slot with open laterals to t 200:
//            closed, steady, balanced; its outer cone LONGER than periodic.
//
// Usage: test_gpu_lowmach_bunsen g2a|extrude|rich|round|bickley|openrich. No
// HIP device -> 77.
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;

std::vector<char> readAll(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Window mean and halves of a stats column over [t0, t1].
struct Win
{
    double mean = NAN, a = NAN, b = NAN;
};
Win window(const valtest::CsvTable& t, const std::string& c, double t0, double t1)
{
    double s = 0, sa = 0, sb = 0;
    int n = 0, na = 0, nb = 0;
    const auto& tt = t.col("t");
    const auto& v = t.col(c);
    for (std::size_t i = 0; i < tt.size(); ++i) {
        if (tt[i] < t0 - 1e-9 || tt[i] > t1 + 1e-9) continue;
        s += v[i];
        ++n;
        if (tt[i] < 0.5 * (t0 + t1)) {
            sa += v[i];
            ++na;
        } else {
            sb += v[i];
            ++nb;
        }
    }
    return {n ? s / n : NAN, na ? sa / na : NAN, nb ? sb / nb : NAN};
}

std::optional<valtest::CsvTable> run(const std::string& prefix,
                                     const std::string& args)
{
    const int rc = valtest::runBinary("\"" DNS_BIN "\" " + args + " --quiet --out " +
                                      prefix);
    check(rc == 0, prefix + ": exits 0");
    if (rc != 0) return std::nullopt;
    auto t = valtest::loadCsv(prefix + "_stats.csv");
    check(t.has_value(), prefix + ": stats readable");
    return t;
}

const std::string kG2a = "flame --mode bunsen --expansion 5 --zeldovich 8 "
                         "--prandtl 0.72 --cells-per-delta 8 --slot-width 12 "
                         "--jet-speed 5 --tt 30 --emit stats ";

int g2a()
{
    const double t0 = 20, t1 = 25;
    const auto lm = run("/tmp/dns_test_lmb_g2a_lm",
                        kG2a + "--physics lowmach --length 49");
    const auto c1 = run("/tmp/dns_test_lmb_g2a_m01",
                        kG2a + "--mach 0.01 --nscbc-tinflow --nscbc-etain 1000");
    const auto c2 = run("/tmp/dns_test_lmb_g2a_m02",
                        kG2a + "--mach 0.02 --nscbc-tinflow --nscbc-etain 1000");
    if (!lm || !c1 || !c2) return dnstest::finish();
    // reactingFoam G2a (foam_mclin_c8, window [20, 25], the port on its
    // fields: runs/flames/foam/G2a/compare.txt, 2026-10-30).
    struct Q
    {
        const char* name;
        double foam;
    };
    const Q qs[] = {{"xTip", 43.71554}, {"tanA", 0.21025}, {"sinRatio", 1.02876},
                    {"SLflank", 1.00392}, {"mFlank", 1.00304},
                    {"burnRatio", 1.00318}};
    std::printf("  %-9s %10s %10s %10s %10s %10s | lm/ext-1  lm/foam-1\n", "",
                "low-Mach", "M 0.01", "M 0.02", "M->0", "foam");
    for (const Q& q : qs) {
        const Win l = window(*lm, q.name, t0, t1);
        const double m1 = window(*c1, q.name, t0, t1).mean;
        const double m2 = window(*c2, q.name, t0, t1).mean;
        const double ex = (4 * m1 - m2) / 3; // M^2 extrapolation
        const double re = l.mean / ex - 1, rf = l.mean / q.foam - 1;
        std::printf("  %-9s %10.5f %10.5f %10.5f %10.5f %10.5f | %+.2e  %+.2e\n",
                    q.name, l.mean, m1, m2, ex, q.foam, re, rf);
        check(std::abs(l.a - l.b) <= 1e-3 * std::abs(l.mean),
              std::format("low-Mach {} steady (halves to 1e-3)", q.name));
        const std::string n = q.name;
        if (n == "SLflank" || n == "mFlank" || n == "burnRatio")
            check(std::abs(re) <= 0.01,
                  n + " within 1% of the compressible M -> 0");
        if (n == "xTip" || n == "sinRatio") {
            check(std::abs(re) <= 0.03,
                  n + " within 3% of the compressible M -> 0 (plan 1%: LM9)");
            check(std::abs(rf) <= 0.01, n + " within 1% of reactingFoam");
        }
        if (n == "SLflank")
            check(std::abs(rf) <= 0.015, "SLflank within 1.5% of reactingFoam");
    }
    const Win br = window(*lm, "burnRatio", t0, t1);
    check(std::abs(br.mean - 1) <= 1e-3, "low-Mach burn = fuel inflow to 1e-3");
    // The IC is not the projected flow: Y_F undershoots to -8e-4 in the first
    // tau and recovers by t ~ 5 (measured); the window is clean.
    double tmax = 0, hits = 0;
    for (std::size_t i = 0; i < lm->col("t").size(); ++i) {
        if (lm->col("t")[i] < t0) continue;
        tmax = std::max(tmax, lm->col("Tmax")[i]);
        hits += lm->col("yHits")[i];
    }
    check(std::abs(tmax / 5 - 1) <= 2e-4, "low-Mach Tmax = sigma Tu to 2e-4 (window)");
    check(hits == 0, "low-Mach yHits 0 over the window");
    return dnstest::finish();
}

// 2D vs nz 8: frames and stats columns byte for byte.
void extrudeOne(const std::string& name, const std::string& args,
                const std::vector<std::string>& frames)
{
    const std::string p2 = "/tmp/dns_test_lmb_x2_" + name;
    const std::string p3 = "/tmp/dns_test_lmb_x3_" + name;
    const auto t2 = run(p2, args);
    const auto t3 = run(p3, args + " --nz 8");
    if (!t2 || !t3) return;
    for (const std::string& f : frames) {
        const auto a = readAll(p2 + "_" + f + ".bin");
        const auto b = readAll(p3 + "_" + f + ".bin");
        check(!a.empty() && a == b,
              std::format("{}: {} frames (mid-span plane) byte-identical to 2D "
                          "({} / {} bytes)",
                          name, f, a.size(), b.size()));
    }
    int cols = 0;
    bool all = true;
    for (const std::string& k : t2->names) {
        // mass, massErr: box totals; dtDiff: the bound itself (1/6 vs 1/4).
        if (k == "mass" || k == "massErr" || k == "dtDiff") continue;
        const auto& a = t2->col(k);
        const auto& b = t3->col(k);
        bool same = a.size() == b.size() && !a.empty();
        for (std::size_t i = 0; same && i < a.size(); ++i)
            same = (a[i] == b[i]) || (std::isnan(a[i]) && std::isnan(b[i]));
        if (!same) std::printf("    %s: column %s differs\n", name.c_str(), k.c_str());
        all = all && same;
        ++cols;
    }
    check(all && cols > 20,
          std::format("{}: {} stats columns (all but mass, massErr, dtDiff) "
                      "equal the 2D run exactly ({} rows)",
                      name, cols, t2->col("t").size()));
}

int extrude()
{
    extrudeOne("c0",
               "flame --mode bunsen --physics lowmach --cells-per-delta 4 "
               "--slot-width 6 --coflow-width 3 --jet-speed 3 --cfl 0.1 --tt 2 "
               "--dt-rule bounds --emit stats --emit bin --fdt 0.5",
               {"temp", "fuel", "omega", "speed"});
    extrudeOne("c2",
               "flame --mode bunsen --physics lowmach --equivalence 1.3 "
               "--expansion 6.88 --visc sutherland --tinf 298 --prandtl 0.52 "
               "--cells-per-delta 4 --slot-width 4 --jet-speed 3 --pilot-width "
               "1.5 --pilot-speed 3 --coflow-width 3 --cfl 0.1 --tt 2 "
               "--dt-rule bounds --emit stats --emit bin --fdt 0.5",
               {"temp", "fuel", "oxid", "omega", "findex", "speed"});
    return dnstest::finish();
}

int rich()
{
    const double t0 = 30, t1 = 40;
    const auto t = run("/tmp/dns_test_lmb_rich",
                       "flame --mode bunsen --physics lowmach --equivalence 1.3 "
                       "--expansion 6.88 --visc sutherland --tinf 298 --prandtl "
                       "0.52 --slot-width 4 --jet-speed 3 --pilot-width 1.5 "
                       "--pilot-speed 3 --coflow-width 6 --length 50 --tt 40 "
                       "--emit stats");
    if (!t) return dnstest::finish();
    const Win burn = window(*t, "burn", t0, t1), dif = window(*t, "burnDif", t0, t1);
    const Win pre = window(*t, "preFrac", t0, t1);
    const Win out = window(*t, "xTipOut", t0, t1), tip = window(*t, "xTip", t0, t1);
    const Win br = window(*t, "burnRatio", t0, t1);
    double zMin = 1, zMax = 0, hits = 0;
    for (std::size_t i = 0; i < t->col("t").size(); ++i) {
        if (t->col("t")[i] < t0) continue;
        zMin = std::min(zMin, t->col("zMin")[i]);
        zMax = std::max(zMax, t->col("zMax")[i]);
        hits += t->col("yHits")[i];
    }
    std::printf("  inner tip %.4f, outer tip %.4f (halves %.4f / %.4f), burn %.5f "
                "(premixed %.5f, diffusion %.5f), preFrac %.5f, burnRatio %.6f, "
                "Z [%+.2e, 1%+.2e], yHits %g\n",
                tip.mean, out.mean, out.a, out.b, burn.mean, burn.mean - dif.mean,
                dif.mean, pre.mean, br.mean, zMin, zMax - 1, hits);
    check(dif.mean > 0.1 * burn.mean, "two flames: burnDif > 0.1 burn");
    check(pre.mean >= 0.72 && pre.mean <= 0.80, "preFrac in [0.72, 0.80]");
    check(std::isfinite(out.mean) && std::abs(out.a - out.b) <= 1e-3 * out.mean &&
              out.mean > tip.mean,
          "the outer cone closed (xTipOut finite, steady to 1e-3, past the inner "
          "tip)");
    check(std::abs(br.mean - 1) <= 1e-3, "burn = fuelIn - fuelOut to 1e-3");
    check(zMin >= -1e-5 && zMax <= 1 + 1e-5, "Z in [0, 1] +- 1e-5");
    check(hits == 0, "yHits 0 over the window");
    return dnstest::finish();
}

int round()
{
    const std::string p = "/tmp/dns_test_lmb_round";
    const auto t = run(p, "flame --mode bunsen --physics lowmach --nozzle round "
                          "--nz 64 --equivalence 1.3 --expansion 6.88 --visc "
                          "sutherland --tinf 298 --prandtl 0.52 --cells-per-delta 4 "
                          "--slot-width 8 --pilot-width 2 --pilot-speed 5 "
                          "--coflow-width 2 --jet-speed 5 --tt 3 "
                          "--emit stats --emit volume --fdt 1");
    if (!t) return dnstest::finish();
    const double tip = t->col("xTip").back(), base = t->col("xBase").back();
    std::printf("  round: xTip %.4f, xBase %.4f, ST %.4f, preFrac %.4f, uMax %.4f\n",
                tip, base, t->col("ST").back(), t->col("preFrac").back(),
                t->col("uMax").back());
    check(std::isfinite(tip) && base < 0.5 * tip,
          "round: axial-plane cone measured, base anchored");
    double tmax = 0, hits = 0;
    for (std::size_t i = 0; i < t->col("t").size(); ++i) {
        tmax = std::max(tmax, t->col("Tmax")[i]);
        hits += t->col("yHits")[i];
    }
    // sigma_st 6.9773 (the outer sheet's Burke-Schumann T): bounded at 4
    // cells by the burner's default --dt-rule cell (S9; the S8 default,
    // diffusion-bound dt 0.0091, over/undershot at the inner tip and S5 ran
    // --chem-dt 0.5: issues LM10).
    check(tmax <= 6.9773 * (1 + 1e-3) && hits == 0,
          std::format("round: bounded (Tmax {:.5f} <= sigma_st, yHits {:g})", tmax,
                      hits));
    const std::size_t n = std::size_t(140) * 64 * 64;
    for (const char* f : {"temp", "fuel", "oxid", "omega", "findex", "speed"}) {
        const auto b = readAll(p + "_" + f + ".bin");
        check(b.size() == 4 * n * 4,
              std::format("round: {} volumes written (4 frames, {} bytes)", f,
                          b.size()));
        if (std::string(f) != "findex" || b.size() != 4 * n * 4) continue;
        const float* x = reinterpret_cast<const float*>(b.data());
        bool in = true;
        for (std::size_t q = 0; q < 4 * n; ++q)
            in = in && std::abs(x[q]) <= 1.0f + 1e-6f;
        check(in, "round: findex in [-1, 1]");
    }
    return dnstest::finish();
}

// --- S6: open laterals --------------------------------------------------------

struct Line
{
    double a = 0, b = 0, rms = 0;
};

Line fit(const std::vector<double>& x, const std::vector<double>& y)
{
    const double n = double(x.size());
    double mx = 0, my = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;
    double sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        sxx += (x[i] - mx) * (x[i] - mx);
        sxy += (x[i] - mx) * (y[i] - my);
    }
    Line l;
    l.b = sxy / sxx;
    l.a = my - l.b * mx;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double r = y[i] - (l.a + l.b * x[i]);
        l.rms += r * r;
    }
    l.rms = std::sqrt(l.rms / n);
    return l;
}

// The log-log slope of v against x - x0.
double powerExp(const std::vector<double>& x, const std::vector<double>& v, double x0)
{
    std::vector<double> lx, lv;
    for (std::size_t i = 0; i < x.size(); ++i) {
        lx.push_back(std::log(x[i] - x0));
        lv.push_back(std::log(v[i]));
    }
    return fit(lx, lv).b;
}

// The jet profile at the LAST profile time of PREFIX_jet.csv, columns by name.
struct JetProf
{
    std::vector<double> x, uc, J, P, Q, Qj, b;
    double t = NAN;
};

std::optional<JetProf> lastJet(const std::string& prefix)
{
    auto t = valtest::loadCsv(prefix + "_jet.csv");
    check(t.has_value(), prefix + ": jet profiles readable");
    if (!t) return std::nullopt;
    JetProf p;
    const auto& tt = t->col("t");
    p.t = tt.back();
    for (std::size_t i = 0; i < tt.size(); ++i) {
        if (tt[i] != p.t) continue;
        p.x.push_back(t->col("x")[i]);
        p.uc.push_back(t->col("uc")[i]);
        p.J.push_back(t->col("J")[i]);
        p.P.push_back(t->col("P")[i]);
        p.Q.push_back(t->col("Q")[i]);
        p.Qj.push_back(t->col("Qj")[i]);
        p.b.push_back(t->col("b")[i]);
    }
    return p;
}

// The 2D planar jet (Bickley 1937) on the inert open burner: W 4 delta_th =
// 6.40660 delta_D, U 3, nu = Pr = 0.32033 (Re_jet = U W / nu = 60), 60 W
// long, 4 cells / delta_th (16 cells across W), tt 1200 (steady: u_c moves
// 7e-5 over the last 50). Bickley with K = the FAR-FIELD momentum flux (the
// window mean of J): a wall-issued plane jet hands part of J to the wall
// suction at its base (Schneider 1985) -- the box conserves J + P (P = int
// p), not J. Window x in [20, 50] W. Measured 2026-10-31 (Re 60 / Re 30):
// u_c^-3 slope / Bickley 0.984 / 0.989, exponent (that origin) -0.3330 /
// -0.3316, free origin -0.341 / -0.374; b^1.5 0.976 / 0.954; Qj^3 (Q within
// 3 b) 0.967 / 0.908, dQj/dx exponent -0.680 / -0.767; J +-0.38% / 1.04%;
// J + P along the box 0.36% / 0.55% -- every residual halves from Re 30 to
// 60 (the O(1/Re) higher-order boundary-layer terms), so the test runs Re 60.
// No confinement: the clearance 15 W -> 7.5 W moves u_c <= 0.36%, b <= 0.6%,
// J <= 1.0% over [5, 50] W. Periodic laterals (the same box): net Q constant
// (no lateral entrainment: the jet feeds on its own recirculation), J +-9.6%.
// --dt-rule bounds (S9): the isothermal inert jet has no T/Y extremum for
// the burner's cell rule to protect, and that rule's smaller dt took this
// test from 132 s to 231 s against its 300 s timeout (gates unchanged).
const std::string kBickley =
    "flame --mode bunsen --physics lowmach --reaction off --coflow-temp 1 "
    "--coflow-speed 0 --slot-width 4 --jet-speed 3 --prandtl 0.32033 "
    "--cells-per-delta 4 --length 240 --tt 1200 --profile-dt 50 --emit stats "
    "--dt-rule bounds ";

int bickley()
{
    const double W = 4 * 1.60165, nu = 0.32033;
    const std::string p15 = "/tmp/dns_test_lmb_bick15", p75 = "/tmp/dns_test_lmb_bick75",
                      pp = "/tmp/dns_test_lmb_bickper";
    const auto s15 = run(p15, kBickley + "--coflow-width 60 --laterals open");
    const auto s75 = run(p75, kBickley + "--coflow-width 30 --laterals open");
    const auto sp = run(pp, kBickley + "--coflow-width 60");
    const auto a = lastJet(p15), b = lastJet(p75), c = lastJet(pp);
    if (!s15 || !s75 || !sp || !a || !b || !c) return dnstest::finish();
    std::vector<std::size_t> sel;
    for (std::size_t i = 0; i < a->x.size(); ++i)
        if (a->x[i] >= 20 * W && a->x[i] <= 50 * W) sel.push_back(i);
    auto pick = [&](const std::vector<double>& v) {
        std::vector<double> o;
        for (const std::size_t i : sel) o.push_back(v[i]);
        return o;
    };
    const auto xs = pick(a->x), uc = pick(a->uc), J = pick(a->J), Qj = pick(a->Qj),
               bw = pick(a->b);
    double K = 0;
    for (const double j : J) K += j;
    K /= double(J.size());
    auto map = [](const std::vector<double>& v, double e) {
        std::vector<double> o;
        for (const double q : v) o.push_back(std::pow(q, e));
        return o;
    };
    const Line lu = fit(xs, map(uc, -3)), lq = fit(xs, map(Qj, 3)), lb = fit(xs, map(bw, 1.5));
    const double x0 = -lu.a / lu.b;
    const double ru = lu.b / (32 * nu / (3 * K * K));
    const double rq = lq.b / (36 * K * nu * std::pow(std::tanh(3 * 0.8814), 3));
    const double rb = lb.b / (std::pow(0.8814, 1.5) * std::sqrt(48 * nu * nu / K));
    const double nU = powerExp(xs, uc, x0);
    std::vector<double> dQ;
    for (const std::size_t i : sel) dQ.push_back((a->Qj[i + 1] - a->Qj[i - 1]) / (a->x[i + 1] - a->x[i - 1]));
    const double nD = powerExp(xs, dQ, x0);
    double jDev = 0, jpMin = HUGE_VAL, jpMax = -HUGE_VAL;
    for (const double j : J) jDev = std::max(jDev, std::abs(j / K - 1));
    for (std::size_t i = 0; i < a->x.size(); ++i) {
        jpMin = std::min(jpMin, a->J[i] + a->P[i]);
        jpMax = std::max(jpMax, a->J[i] + a->P[i]);
    }
    std::printf("  open 15 W, Re 60 (t %g): K %.5g; u_c^-3 slope / Bickley %.4f (origin "
                "%+.2f W), exponent %+.4f; b^1.5 %.4f; Qj^3 %.4f, dQj/dx exponent "
                "%+.4f; J +-%.2e; J + P [%.5g, %.5g]\n",
                a->t, K, ru, x0 / W, nU, rb, rq, nD, jDev, jpMin, jpMax);
    check(std::abs(ru - 1) <= 0.03, "bickley: u_c^-3 slope within 3% of 32 nu/(3 K^2)");
    check(std::abs(nU + 1.0 / 3) <= 0.02, "bickley: u_c ~ x^(-1/3) (+-0.02)");
    check(std::abs(rb - 1) <= 0.05, "bickley: b^1.5 slope within 5% (half-width ~ x^(2/3))");
    check(std::abs(rq - 1) <= 0.06, "bickley: Qj^3 slope within 6% of 36 K nu");
    check(std::abs(nD + 2.0 / 3) <= 0.05, "bickley: entrainment dQ/dx ~ x^(-2/3) (+-0.05)");
    check(jDev <= 0.02, "bickley: momentum flux J conserved +-2% over the window");
    check((jpMax - jpMin) / jpMax <= 0.01, "bickley: J + P conserved to 1% along the box");
    // Steady and balanced: the last stats row's mass fluxes (in + lat = out).
    const double mi = s15->col("mdotIn").back(), mo = s15->col("mdotOut").back(),
                 ml = s15->col("mdotLat").back();
    check(std::abs(mi + ml - mo) <= 1e-3 * mo && ml > 0 && s15->col("uLatMin").back() < 0,
          std::format("bickley: steady mass balance in {:.5g} + lateral {:.5g} = out "
                      "{:.5g} (the laterals entrain)",
                      mi, ml, mo));
    // No confinement: the clearance halved.
    double du = 0, db = 0, dj = 0;
    for (std::size_t i = 0; i < a->x.size(); ++i) {
        if (a->x[i] < 5 * W || a->x[i] > 50 * W) continue;
        du = std::max(du, std::abs(b->uc[i] / a->uc[i] - 1));
        db = std::max(db, std::abs(b->b[i] / a->b[i] - 1));
        dj = std::max(dj, std::abs(b->J[i] / a->J[i] - 1));
    }
    std::printf("  clearance 7.5 W vs 15 W over [5, 50] W: u_c %.2e, b %.2e, J %.2e\n",
                du, db, dj);
    check(du <= 0.005 && db <= 0.01 && dj <= 0.015,
          "bickley: no confinement (clearance halved: u_c 0.5%, b 1%, J 1.5%)");
    // Periodic (report + the confinement it shows).
    double qMin = HUGE_VAL, qMax = -HUGE_VAL, jMin = HUGE_VAL, jMax = -HUGE_VAL;
    for (const std::size_t i : sel) {
        qMin = std::min(qMin, c->Q[i]);
        qMax = std::max(qMax, c->Q[i]);
        jMin = std::min(jMin, c->J[i]);
        jMax = std::max(jMax, c->J[i]);
    }
    std::printf("  periodic: net Q over the window [%.5g, %.5g] (open: %.5g -> %.5g), J "
                "spread %.3f\n",
                qMin, qMax, a->Q[sel.front()], a->Q[sel.back()], (jMax - jMin) / jMax);
    check((qMax - qMin) / qMax <= 1e-3 && (jMax - jMin) / jMax > 0.05,
          "bickley: periodic laterals confine (net Q constant, no entrainment; J "
          "not conserved)");
    return dnstest::finish();
}

// The partially premixed slot with OPEN laterals (S6, issues LM11): rich2d's
// burner (W 4, U 3, pilots 1.5 at U, air coflow 1 S_L at x-lo, clearance 6,
// 8 cells, length 50) to t 200 (the starting-jet head leaves the box by ~100:
// measured), window [150, 200]. Measured 2026-10-31: outer tip 90.47 (periodic
// 80.09: the dilatation pushes the coflow air OUT through the laterals, net
// -32 of mdotIn 116), preFrac 0.7268, burnRatio 1.0000, steady to 1e-6.
int openRich()
{
    const double t0 = 150, t1 = 200;
    const auto t = run("/tmp/dns_test_lmb_openrich",
                       "flame --mode bunsen --physics lowmach --laterals open "
                       "--equivalence 1.3 --expansion 6.88 --visc sutherland "
                       "--tinf 298 --prandtl 0.52 --slot-width 4 --jet-speed 3 "
                       "--pilot-width 1.5 --pilot-speed 3 --coflow-width 6 "
                       "--length 50 --tt 200 --emit stats");
    if (!t) return dnstest::finish();
    const Win out = window(*t, "xTipOut", t0, t1), br = window(*t, "burnRatio", t0, t1);
    const Win pre = window(*t, "preFrac", t0, t1), mi = window(*t, "mdotIn", t0, t1),
              mo = window(*t, "mdotOut", t0, t1), ml = window(*t, "mdotLat", t0, t1);
    double zMin = 1, zMax = 0, hits = 0, tmax = 0;
    for (std::size_t i = 0; i < t->col("t").size(); ++i) {
        if (t->col("t")[i] < t0) continue;
        zMin = std::min(zMin, t->col("zMin")[i]);
        zMax = std::max(zMax, t->col("zMax")[i]);
        hits += t->col("yHits")[i];
        tmax = std::max(tmax, t->col("Tmax")[i]);
    }
    std::printf("  open: outer tip %.4f (halves %.4f / %.4f; periodic 80.09), preFrac "
                "%.5f, burnRatio %.6f, mdot in %.5g + lateral %.5g -> out %.5g, Tmax "
                "%.5f, Z [%+.2e, 1%+.2e], yHits %g\n",
                out.mean, out.a, out.b, pre.mean, br.mean, mi.mean, ml.mean, mo.mean,
                tmax, zMin, zMax - 1, hits);
    check(std::isfinite(out.mean) && std::abs(out.a - out.b) <= 1e-4 * out.mean,
          "openrich: the outer cone closed and steady (1e-4)");
    check(out.mean > 80.09, "openrich: the open box's outer cone is LONGER than the "
                            "periodic one's (80.09; LM11)");
    check(std::abs(mi.mean + ml.mean - mo.mean) <= 1e-3 * mo.mean && ml.mean < 0,
          "openrich: steady mass balance, the laterals a net OUTflow");
    check(std::abs(br.mean - 1) <= 1e-3, "openrich: burn = fuelIn - fuelOut to 1e-3");
    check(pre.mean >= 0.70 && pre.mean <= 0.80, "openrich: preFrac in [0.70, 0.80]");
    check(zMin >= -1e-5 && zMax <= 1 + 1e-5 && hits == 0 && tmax <= 6.9773 * 1.001,
          "openrich: Z in [0, 1], yHits 0, Tmax <= sigma_st");
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    const std::string m = argc > 1 ? argv[1] : "g2a";
    if (m == "extrude") return extrude();
    if (m == "rich") return rich();
    if (m == "round") return round();
    if (m == "bickley") return bickley();
    if (m == "openrich") return openRich();
    return g2a();
}
