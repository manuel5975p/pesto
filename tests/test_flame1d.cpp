// P0 chemistry oracle (dns/chem.hpp, docs/chem.md): the FP64 phase-plane
// eigenvalue of the 1D freely propagating single-step flame. (a) step-count
// convergence, (b) the exact sqrt(A) similarity, (c) the dimensional form
// against Cantera 3.2.0 on the SAME model (tests/data/chem, written by
// tools/chem/cantera_flame.py; Cantera is never a build dependency), (d) the
// profile table's monotonicity and thickness, (e) the bracket: Lambda(sigma)
// smooth over sigma 2..10 (issues.md 72), (f) the reported artefacts, (g)
// shipped bits, (h) the cold cutoff, (i) model C2 (premixedEigen2) and the
// GRI-3.0 reference, (j) model C2b: the A(phi) law against GRI-3.0's S_L(phi)
// (nodes, held-out rows), its smoothness, FP32 = FP64, the lean branch, T_ad.
#include "dns/chem.hpp"

#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using dnstest::check;

double rel(double a, double b) { return std::abs(a / b - 1); }

// The number after "key": in a flat JSON document (the tool's own output).
double jsonNumber(const std::string& doc, const std::string& key)
{
    const auto p = doc.find("\"" + key + "\":");
    if (p == std::string::npos) return NAN;
    return std::stod(doc.substr(p + key.size() + 3));
}

} // namespace

int main()
{
    using namespace dns;
    const PremixedModel md{}; // sigma 5, beta 8, constant lambda

    // (a) step count: N vs 4N.
    auto e1 = premixedEigen(md, 20000);
    auto e4 = premixedEigen(md, 80000);
    check(e1 && e4, "premixedEigen converges at N 2e4 and 8e4");
    if (!e1 || !e4) return dnstest::finish();
    std::printf("  Lambda N2e4 %.10f  N8e4 %.10f  rel %.3e\n", e1->Lambda,
                e4->Lambda, rel(e1->Lambda, e4->Lambda));
    std::printf("  delta_th/delta_D %.8f\n", e1->dthOverDD);
    check(rel(e1->Lambda, e4->Lambda) < 1e-9, "(a) Lambda(N) = Lambda(4N) to 1e-9");
    // The scouting value (RK4 in T, scratch fd/phase.py): 3656401.098, 1.60165.
    check(rel(e1->Lambda, 3656401.098) < 1e-6, "Lambda = 3656401.098 (scouting)");
    check(rel(e1->dthOverDD, 1.60165) < 1e-5, "delta_th/delta_D = 1.60165");
    {
        PremixedModel b10 = md;
        b10.beta = 10;
        PremixedModel b6 = md;
        b6.beta = 6;
        auto a = premixedEigen(b10), b = premixedEigen(b6);
        check(a && rel(a->Lambda, 69343676.77) < 1e-6 &&
                  rel(a->dthOverDD, 1.49422) < 1e-5,
              "beta 10: Lambda 69343676.77, delta_th 1.49422");
        check(b && rel(b->Lambda, 169306.6266) < 1e-6 &&
                  rel(b->dthOverDD, 1.78079) < 1e-5,
              "beta 6: Lambda 169306.6266, delta_th 1.78079");
    }

    // (b) m(4A) = 2 m(A) on the raw problem.
    {
        FlameProblem f;
        f.Tu = 1;
        f.Tb = md.sigma;
        f.cp = 1;
        f.Ta = md.theta();
        f.rho = [](double T) { return 1 / T; };
        f.lambda = [](double) { return 1.0; };
        f.A = 1e6;
        auto m1 = flameEigen(f);
        f.A = 4e6;
        auto m4 = flameEigen(f);
        check(m1 && m4 && std::abs(m4->m / (2 * m1->m) - 1) < 1e-10,
              "(b) m(4A) = 2 m(A) to 1e-10");
        if (m1 && m4)
            std::printf("  m(1e6) %.15g  m(4e6)/2 %.15g\n", m1->m, m4->m / 2);
    }

    // (d) the profile table: x and T strictly increasing; delta_th from a
    // fine resampling of the Hermite profile equals (sigma - 1)/max dT/dx.
    {
        const auto& t = e1->tab;
        bool mono = true;
        for (std::size_t k = 1; k < t.x.size(); ++k)
            mono = mono && t.x[k] > t.x[k - 1] && t.T[k] > t.T[k - 1];
        check(mono, "(d) x(T) table strictly monotone");
        double gmax = 0;
        const double hs = 1e-4;
        for (double x = -5; x < 5; x += hs)
            gmax = std::max(gmax, (flameProfile(t, x + hs) - flameProfile(t, x)) / hs);
        const double dthFd = (md.sigma - 1) / gmax;
        std::printf("  delta_th: table %.8f  resampled %.8f\n", e1->dthOverDD,
                    dthFd);
        check(rel(dthFd, e1->dthOverDD) < 1e-6,
              "(d) resampled thickness = (sigma-1)/max(p/lambda) to 1e-6");
        check(std::abs(flameProfile(t, 0) - 0.5 * (1 + md.sigma)) < 1e-12,
              "(d) T(0) = (Tu+Tb)/2");
        check(flameProfile(t, -1e3) == t.T.front() &&
                  flameProfile(t, 1e3) == t.T.back(),
              "(d) ends clamp to the table temperatures");
    }

    // (e) the bracket (issues.md 72): ln Lambda(sigma) smooth over sigma
    // 2..10 step 0.02, constant lambda and Sutherland (Tu 298 K). The old
    // doubling/halving bisected spurious verdicts (Lambda = 2^14 at sigma 6.5,
    // 42008 at 6.88): a factor >= 2 between neighbours, |d2 ln Lambda| >=
    // 0.69. Measured |d2| <= 5.6e-3 (sigma 2: d ln Lambda = 0.16 per step,
    // so the plan's "3% between neighbours" is not a property of the model).
    {
        constexpr int nS = 401;
        for (const bool suth : {false, true}) {
            std::vector<double> L(nS, NAN);
            std::vector<std::thread> pool;
            const int nt = int(std::clamp(std::thread::hardware_concurrency(), 1u, 16u));
            for (int t = 0; t < nt; ++t)
                pool.emplace_back([&, t] {
                    for (int i = t; i < nS; i += nt) {
                        PremixedModel m;
                        m.sigma = 2 + 0.02 * i;
                        m.sutherland = suth;
                        m.sOverTu = suth ? 110.4 / 298 : 0.0;
                        auto e = premixedEigen(m);
                        if (e) L[std::size_t(i)] = e->Lambda;
                    }
                });
            for (auto& th : pool) th.join();
            bool all = true;
            double d2max = 0;
            int iMax = 0;
            for (int i = 0; i < nS; ++i) all = all && std::isfinite(L[std::size_t(i)]);
            for (int i = 1; all && i + 1 < nS; ++i) {
                const double d2 = std::abs(std::log(L[std::size_t(i + 1)]) -
                                           2 * std::log(L[std::size_t(i)]) +
                                           std::log(L[std::size_t(i - 1)]));
                if (d2 > d2max) {
                    d2max = d2;
                    iMax = i;
                }
            }
            std::printf("  (e) %s: max |d2 ln Lambda| %.3e at sigma %.2f\n",
                        suth ? "Sutherland" : "constant", d2max, 2 + 0.02 * iMax);
            check(all, std::string("(e) every sigma in 2..10 resolves, ") +
                           (suth ? "Sutherland" : "constant lambda"));
            check(all && d2max < 0.02,
                  std::string("(e) Lambda(sigma) smooth: |d2 ln Lambda| < 0.02, ") +
                      (suth ? "Sutherland" : "constant lambda"));
        }
    }

    // (f) the reported artefacts, now: sigma 6.5 / 6.88 with Sutherland at
    // Tu 298 K (old Lambda 16383.95 = 2^14 / 42008.4, delta_th/delta_D 5.13 /
    // 45.5).
    {
        PremixedModel s65, s688;
        s65.sigma = 6.5;
        s688.sigma = 6.88;
        for (auto* m : {&s65, &s688}) {
            m->sutherland = true;
            m->sOverTu = 110.4 / 298;
        }
        auto a = premixedEigen(s65), b = premixedEigen(s688);
        check(a && b, "(f) sigma 6.5 and 6.88 (Sutherland) resolve");
        if (a && b) {
            std::printf("  (f) sigma 6.5: Lambda %.10g dth %.6g; 6.88: Lambda "
                        "%.10g dth %.6g\n",
                        a->Lambda, a->dthOverDD, b->Lambda, b->dthOverDD);
            check(rel(a->Lambda, 941442.5603) < 1e-9 &&
                      rel(b->Lambda, 882657.4055) < 1e-9,
                  "(f) Lambda 941442.5603 (6.5), 882657.4055 (6.88)");
            check(b->dthOverDD >= 4.4 && b->dthOverDD <= 4.7,
                  "(f) sigma 6.88 Sutherland: delta_th/delta_D in [4.4, 4.7]");
        }
    }

    // (g) every well-posed problem keeps its bits: the scan reproduces the
    // shipped [lo, 1] / [1, hi] bracket, the bisection its midpoints.
    {
        struct Pin
        {
            double sigma, beta, tinf, Lambda;
        };
        const Pin pins[] = {{5, 8, 0, 0x1.be5688c9b03fap+21},
                            {5, 10, 0, 0x1.08866f31604bdp+26},
                            {5, 6, 0, 0x1.4aad5039474dcp+17},
                            {7.5, 8, 300, 0x1.8c6197062553ap+19},
                            {3, 8, 0, 0x1.cee18b0666651p+23}};
        bool same = true;
        for (const Pin& p : pins) {
            PremixedModel m;
            m.sigma = p.sigma;
            m.beta = p.beta;
            m.sutherland = p.tinf > 0;
            m.sOverTu = p.tinf > 0 ? 110.4 / p.tinf : 0.0;
            auto e = premixedEigen(m);
            same = same && e && e->Lambda == p.Lambda && e->tab.tCut == 0;
        }
        check(same, "(g) shipped Lambda bit for bit (sigma 5 beta 6/8/10, "
                    "7.5 Sutherland, 3)");
        // rateFactor == 1 is the empty hook bit for bit (a pure multiply).
        FlameProblem f;
        f.Tu = 1;
        f.Tb = 5;
        f.Ta = md.theta();
        f.rho = [](double T) { return 1 / T; };
        f.lambda = [](double) { return 1.0; };
        auto e0 = flameEigen(f);
        f.rateFactor = [](double, double) { return 1.0; };
        auto e1 = flameEigen(f);
        check(e0 && e1 && e0->m == e1->m && e0->T == e1->T && e0->x == e1->x &&
                  e0->dTdx == e1->dTdx,
              "(g) rateFactor = 1 equals the empty hook bit for bit");
    }

    // (h) the cold-boundary difficulty (kColdCut): sigma 2 beta 8 burns its
    // own unburnt mixture (rate(Tu)/rate(Tb) = e^(-beta sigma)); the uncut
    // march drifted as 1/n (Lambda 4.72e8 / 1.24e8 at n 2e4 / 8e4). With the
    // cutoff: n-converged, tCut just above Tu. beta 2 sigma 1.5 never gets an
    // inert preheat zone, beta 4 sigma 2.6 cuts too high: errors by name.
    {
        PremixedModel s2;
        s2.sigma = 2;
        auto a = premixedEigen(s2, 20000), b = premixedEigen(s2, 80000);
        check(a && b, "(h) sigma 2 resolves at n 2e4 and 8e4");
        if (a && b) {
            std::printf("  (h) sigma 2: Lambda %.10g (n 8e4: %.10g, rel "
                        "%.2e) tCut %.8g dth %.6g\n",
                        a->Lambda, b->Lambda, rel(a->Lambda, b->Lambda),
                        a->tab.tCut, a->dthOverDD);
            check(rel(a->Lambda, b->Lambda) < 1e-6 && a->tab.tCut > 1 &&
                      a->tab.tCut < 1.001,
                  "(h) sigma 2: cutoff active just above Tu, Lambda(n) = "
                  "Lambda(4n) to 1e-6");
        }
        PremixedModel bad;
        bad.sigma = 1.5;
        bad.beta = 2;
        auto e = premixedEigen(bad);
        check(!e && e.error().find("cold-boundary difficulty") != std::string::npos,
              "(h) beta 2 sigma 1.5: a cold-boundary-difficulty error");
        // A cut above kColdCutMaxRise of the rise: the cutoff's eigenvalue.
        PremixedModel b4;
        b4.sigma = 2.6;
        b4.beta = 4;
        auto e4 = premixedEigen(b4);
        check(!e4 && e4.error().find("is cut at") != std::string::npos,
              "(h) beta 4 sigma 2.6 (cut 1.1e-3 of the rise above Tu): an "
              "error by name");
    }

    // (i) model C2: the rich two-reactant jet (premixedEigen2).
    {
        PartialPremixModel pm; // phi 1.3, s 3.99, Y_O2 0.233, sigma 6.88
        pm.sutherland = true;
        pm.sOverTu = 110.4 / 298;
        std::printf("  (i) C2 phi 1.3: Y_F1 %.6f Y_O1 %.6f Y_F,rem %.6f Z_st "
                    "%.6f sigma_st %.6f\n",
                    pm.yF1(), pm.yO1(), pm.yFrem(), pm.zst(), pm.sigmaSt());
        check(std::abs(pm.yF1() - 0.0705584) < 1e-6 &&
                  std::abs(pm.yO1() - 0.21656) < 1e-5 &&
                  std::abs(pm.yFrem() - 0.0162827) < 1e-6 &&
                  std::abs(pm.zst() - 0.781963) < 1e-6 &&
                  std::abs(pm.sigmaSt() - 6.97733) < 1e-5,
              "(i) C2 derived: Y_F1 0.07056, Y_O1 0.2166, Y_F,rem 0.01628, Z_st "
              "0.782, sigma_st 6.977");
        check(std::abs(pm.yF1() - (pm.yFrem() + pm.yO1() / pm.s)) < 1e-15 &&
                  std::abs(pm.mixtureFraction(pm.yF1(), pm.yO1()) - 1) < 1e-14 &&
                  std::abs(pm.mixtureFraction(pm.yFrem(), 0) - 1) < 1e-14 &&
                  std::abs(pm.mixtureFraction(0, pm.yO2)) < 1e-14,
              "(i) Z = 1 fresh and burnt jet, 0 in air; Y_F1 = Y_F,rem + Y_O1/s");
        auto e2 = premixedEigen2(pm);
        check(e2.has_value(), "(i) premixedEigen2 resolves (phi 1.3, sigma 6.88)");
        if (e2) {
            std::printf("  (i) Lambda2 %.10g delta_th/delta_D %.6g\n",
                        e2->Lambda, e2->dthOverDD);
            check(e2->dthOverDD > 4.4 && e2->dthOverDD < 5.0,
                  "(i) two-reactant delta_th/delta_D in (4.4, 5.0)");
        }
        auto r1 = premixedSpeedRatio(pm, pm.phi);
        check(r1 && *r1 == 1.0, "(i) S_L(phi_jet)/S_L(phi_jet) = 1 exactly");
        // m(4A) = 2 m(A) with the rate factor (the sqrt(A) similarity that
        // makes Lambda2 = 1/m(A = 1)^2).
        FlameProblem f;
        f.Tu = 1;
        f.Tb = pm.sigma;
        f.Ta = pm.theta();
        f.rho = [](double T) { return 1 / T; };
        f.lambda = [&pm](double T) { return pm.lambdaRatio(T); };
        const double sY = pm.s * pm.yFrem(), yO1 = pm.yO1();
        f.rateFactor = [sY, yO1](double T, double sp) {
            return (1 / T) * (sY + yO1 * sp);
        };
        f.A = 1e7;
        auto m1 = flameEigen(f);
        f.A = 4e7;
        auto m4 = flameEigen(f);
        check(m1 && m4 && std::abs(m4->m / (2 * m1->m) - 1) < 1e-10,
              "(i) two-reactant m(4A) = 2 m(A) to 1e-10");
        // phi -> large: Y_F ~ Y_F,rem everywhere, the rate is A rho^2 s
        // Y_F,rem Y_O = (s Y_F,rem) x the one-reactant rate with rho(T) as
        // its factor; relative O(Y_O1 / (s Y_F,rem)) ~ 1/phi.
        PartialPremixModel big = pm;
        big.phi = 1e6;
        auto eb = premixedEigen2(big);
        FlameProblem g = f;
        g.A = 1;
        g.rateFactor = [](double T, double) { return 1 / T; };
        auto ref = flameEigen(g);
        if (eb && ref) {
            const double mBig = 1 / std::sqrt(eb->Lambda);
            const double mRef = ref->m * std::sqrt(big.s * big.yFrem());
            std::printf("  (i) phi 1e6: m %.12g vs sqrt(s Y_F,rem) m_1 %.12g "
                        "(rel %.2e)\n",
                        mBig, mRef, rel(mBig, mRef));
            check(rel(mBig, mRef) < 1e-6,
                  "(i) phi -> large: the one-reactant eigenvalue x sqrt(s "
                  "Y_F,rem) to 1e-6");
        } else {
            check(false, "(i) phi 1e6 and its one-reactant limit resolve");
        }
        // The model's own S_L(phi) at fixed A, q, Ta beside GRI-3.0
        // (tests/data/chem/ch4_gri30.csv, tools/chem/cantera_flame.py
        // --real): printed, not gated (docs/chem.md, D4).
        std::ifstream cs(std::string(DNS_CHEM_DATA) + "/ch4_gri30.csv");
        check(cs.good(), "(i) GRI-3.0 reference present (ch4_gri30.csv)");
        std::vector<double> ph, sl, sad;
        std::string line;
        std::vector<std::string> head;
        while (std::getline(cs, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> f2;
            std::stringstream ls(line);
            for (std::string c; std::getline(ls, c, ',');) f2.push_back(c);
            if (head.empty()) {
                head = f2;
                continue;
            }
            ph.push_back(std::stod(f2[0]));
            sl.push_back(std::stod(f2[1]));
            sad.push_back(std::stod(f2[5]));
        }
        double s13 = NAN;
        for (std::size_t k = 0; k < ph.size(); ++k)
            if (ph[k] == 1.3) s13 = sl[k];
        check(head.size() > 5 && head[0] == "phi" && head[1] == "SL" &&
                  head[5] == "sigma_ad" && std::isfinite(s13),
              "(i) ch4_gri30.csv: phi, SL, sigma_ad columns with the phi 1.3 "
              "row");

        // (j) model C2b (masterplan real_bunsen S1b): the A(phi) law.
        auto law = phiRateLaw(pm);
        check(law.has_value(), "(j) phiRateLaw resolves (phi 1.3, sigma 6.88, "
                               "Sutherland)");
        if (!law) return dnstest::finish();
        const PhiRateLaw<double>& L = *law;
        // The embedded nodes are the csv's rows (one source of numbers).
        bool same = true;
        for (const Ch4SpeedNode& v : ch4Gri30Speeds()) {
            bool hit = false;
            for (std::size_t k = 0; k < ph.size(); ++k)
                hit = hit || (ph[k] == v.phi && sl[k] == v.SL);
            same = same && hit;
        }
        check(same && L.n == int(ch4Gri30Speeds().size()) - 1,
              "(j) the law's GRI nodes are ch4_gri30.csv rows; phi 1 skipped "
              "(26 nodes)");
        check(std::abs(phiRateC(L, pm, pm.phi) - 1) < 1e-12,
              "(j) C(jet) = 1 to 1e-12");
        // Every csv row: the uncorrected and the C2b oracle ratio vs GRI's.
        // phi 1 (no hot-end start) is read at 1 + 1e-4. Threaded: own-index
        // writes.
        const std::size_t nr = ph.size();
        std::vector<double> rOne(nr, NAN), rLaw(nr, NAN);
        {
            std::vector<std::thread> pool;
            const int nt = int(std::clamp(std::thread::hardware_concurrency(), 1u, 16u));
            for (int t = 0; t < nt; ++t)
                pool.emplace_back([&, t] {
                    for (std::size_t k = std::size_t(t); k < nr; k += std::size_t(nt)) {
                        const double p = ph[k] == 1 ? 1 + 1e-4 : ph[k];
                        if (auto a = premixedSpeedRatio(pm, p)) rOne[k] = *a;
                        if (auto b = premixedSpeedRatio(pm, p, &L)) rLaw[k] = *b;
                    }
                });
            for (auto& th : pool) th.join();
        }
        double nodeErr = 0, offErr = 0, sigErr = 0, offPhi = 0;
        bool allOk = true;
        for (std::size_t k = 0; k < nr; ++k) {
            bool node = ph[k] != 1;
            if (node) {
                node = false;
                for (const Ch4SpeedNode& v : ch4Gri30Speeds())
                    node = node || v.phi == ph[k];
            }
            const double g = sl[k] / s13, e = rLaw[k] / g - 1;
            allOk = allOk && std::isfinite(rOne[k]) && std::isfinite(rLaw[k]);
            const double ms = pm.sigmaAt(ph[k]);
            const bool range = ph[k] >= 0.6 && ph[k] <= 1.6;
            if (range) sigErr = std::max(sigErr, std::abs(ms / sad[k] - 1));
            if (node) nodeErr = std::max(nodeErr, std::abs(e));
            else if (range && std::abs(e) > offErr) {
                offErr = std::abs(e);
                offPhi = ph[k];
            }
            std::printf("  (j) phi %5.3f %s GRI-3.0 S_L/S_L(1.3) %.4f  "
                        "one-step %.4f  C2b %.4f (%+.2e)  C %.4f  sigma model "
                        "%.3f GRI %.3f\n",
                        ph[k], node ? "node" : "off ", g, rOne[k], rLaw[k], e,
                        phiRateC(L, pm, ph[k]), ms, sad[k]);
        }
        std::printf("  (j) max |err| nodes %.2e, off-node %.2e (phi %.3f); "
                    "max |sigma_model/sigma_GRI - 1| over 0.6..1.6 %.3f\n",
                    nodeErr, offErr, offPhi, sigErr);
        check(allOk, "(j) every csv row resolves, uncorrected and C2b");
        check(nodeErr < 1e-9, "(j) C2b = GRI-3.0's S_L(phi)/S_L(1.3) at the "
                              "nodes to 1e-9");
        check(offErr < 5e-3, "(j) C2b within 5e-3 of GRI-3.0 at the held-out "
                             "rows 0.575..1.575 and phi 1 (measured 3.3e-3 at "
                             "1.575)");
        // The sign issue 73 is about: uncorrected rises 1.3 -> 1.6, C2b falls.
        double u13 = NAN, u16 = NAN, c16 = NAN;
        for (std::size_t k = 0; k < nr; ++k) {
            if (ph[k] == 1.3) u13 = rOne[k];
            if (ph[k] == 1.6) {
                u16 = rOne[k];
                c16 = rLaw[k];
            }
        }
        check(u16 > u13 && c16 < 0.35,
              "(j) issue 73: one-step S_L rises 1.3 -> 1.6, C2b falls to GRI's "
              "0.342");
        // T_ad(phi): the one-step's complete combustion on the jet's q,
        // stated not fitted (low lean: the rich jet's q is short of CH4's
        // lean heat; high rich: no CO/H2) -- bounded here.
        check(sigErr < 0.17, "(j) model Tb/Tu within 17% of GRI-3.0's T_ad/Tu "
                             "over phi 0.6..1.6 (measured 0.164 at 0.6)");
        // Smooth, no wiggles: ln C monotone (GRI's and the model's ratio
        // both are) on a dense xi grid from the lean end to the rich end,
        // C1 at every node (one-sided slopes agree), zero slope at the ends.
        {
            bool mono = true;
            double prev = HUGE_VAL;
            const double x0 = L.xi[0], x1 = L.xi[L.n - 1];
            for (int i = 0; i <= 4000; ++i) {
                const double xi = x0 + (x1 - x0) * i / 4000.0;
                // Y_F = xi, Y_O = (1 - xi) Y_O2: the fresh premix at xi.
                const double v = phiRateLn(L, xi, (1 - xi) * L.yO2);
                mono = mono && v <= prev;
                prev = v;
            }
            double c1 = 0;
            const double d = 1e-9; // truncation m2 d/2 <= 4e-5
            auto at = [&](double xi) {
                return phiRateLn(L, xi, (1 - xi) * L.yO2);
            };
            for (int k = 0; k < L.n; ++k) {
                const double x = L.xi[k];
                const double sl2 = (at(x) - at(x - d)) / d;
                const double sr2 = (at(x + d) - at(x)) / d;
                const double scale = k == 0 || k == L.n - 1 ? 1.0 : std::abs(sl2) + 1;
                c1 = std::max(c1, std::abs(sr2 - sl2) / scale);
            }
            std::printf("  (j) ln C monotone %d, max slope jump at nodes "
                        "%.2e (rel)\n",
                        int(mono), c1);
            check(mono, "(j) ln C(xi) monotone over the node range");
            check(c1 < 1e-3, "(j) ln C C1 at every node, zero slope at the "
                             "ends (one-sided slopes to 1e-3)");
        }
        // The kernel's FP32 law equals the oracle's FP64 one to FP32 noise.
        {
            const PhiRateLaw<float> Lf = L.as<float>();
            double e = 0;
            for (int i = 0; i <= 2000; ++i) {
                const double phi = 0.45 + 1.3 * i / 2000.0;
                PartialPremixModel b = pm;
                b.phi = phi;
                const double v = phiRateLn(L, b.yF1(), b.yO1());
                const float w = phiRateLn(Lf, float(b.yF1()), float(b.yO1()));
                e = std::max(e, std::abs(double(w) - v));
            }
            std::printf("  (j) FP32 vs FP64 ln C: max |diff| %.2e\n", e);
            check(e < 1e-4, "(j) FP32 law within 1e-4 of FP64 in ln C");
        }
        // The lean branch meets the rich one at phi 1 (both reaction
        // orders -> 2, both Tb -> the same): m(1 - 1e-4) ~ m(1 + 1e-4).
        {
            auto a = premixedSpeedRatio(pm, 1 - 1e-4), b = premixedSpeedRatio(pm, 1 + 1e-4);
            check(a && b && std::abs(*a / *b - 1) < 1e-3,
                  "(j) lean and rich oracles meet at phi 1 (1e-3; slope 1.47)");
        }
        // premixedEigen2At at the jet's phi is premixedEigen2 bit for bit.
        {
            auto a = premixedEigen2(pm), b = premixedEigen2At(pm, pm.phi);
            check(a && b && a->Lambda == b->Lambda && a->tab.m == b->tab.m &&
                      a->tab.x == b->tab.x && a->tab.T == b->tab.T &&
                      a->tab.dTdx == b->tab.dTdx && a->dthOverDD == b->dthOverDD,
                  "(j) premixedEigen2At(jet) == premixedEigen2 bit for bit");
            // --mix-phi 0.8: Lambda the jet's, tab.m the ratio.
            auto m = premixedEigen2At(pm, 0.8, &L);
            check(a && m && m->Lambda == a->Lambda &&
                      std::abs(m->tab.m / (0.26711442 / s13) - 1) < 1e-9,
                  "(j) premixedEigen2At(0.8, law): the jet's Lambda, tab.m = "
                  "GRI's 1.1552");
        }
    }

    // (c) the dimensional model against Cantera 3.2.0.
    {
        const std::string dir = DNS_CHEM_DATA;
        std::ifstream js(dir + "/cantera_premixed.json"),
            cs(dir + "/cantera_premixed.csv");
        check(js.good() && cs.good(), "Cantera reference data present");
        if (!js || !cs) return dnstest::finish();
        std::stringstream ss;
        ss << js.rdbuf();
        const std::string doc = ss.str();
        std::vector<double> TT, lam;
        std::string line;
        std::getline(cs, line); // header
        while (std::getline(cs, line)) {
            const auto c = line.find(',');
            TT.push_back(std::stod(line.substr(0, c)));
            lam.push_back(std::stod(line.substr(c + 1)));
        }
        FlameProblem f;
        f.Tu = jsonNumber(doc, "Tu");
        f.Tb = jsonNumber(doc, "Tb");
        f.cp = jsonNumber(doc, "cp");
        f.A = jsonNumber(doc, "A");
        f.Ta = jsonNumber(doc, "Ta");
        const double P = jsonNumber(doc, "P"), Rs = jsonNumber(doc, "Rs");
        f.rho = [P, Rs](double T) { return P / (Rs * T); };
        // np.interp's linear table, as the scouting oracle read it.
        f.lambda = [&TT, &lam](double T) {
            const auto it = std::ranges::upper_bound(TT, T);
            if (it == TT.begin()) return lam.front();
            if (it == TT.end()) return lam.back();
            const std::size_t k = std::size_t(it - TT.begin());
            const double w = (T - TT[k - 1]) / (TT[k] - TT[k - 1]);
            return lam[k - 1] + w * (lam[k] - lam[k - 1]);
        };
        auto e = flameEigen(f);
        check(e.has_value(), "dimensional oracle converges");
        if (e) {
            const double SL = e->m / f.rho(f.Tu);
            const double fine = jsonNumber(doc, "SL_finest");
            const double rich = jsonNumber(doc, "SL_richardson");
            std::printf("  S_L oracle %.6f m/s  Cantera finest %.6f (%+.3f%%)  "
                        "Richardson %.6f (%+.3f%%)  delta_th %.4e m\n",
                        SL, fine, 100 * (SL / fine - 1), rich,
                        100 * (SL / rich - 1), e->dth);
            check(rel(SL, fine) < 5e-3,
                  "(c) |S_oracle/S_cantera_finest - 1| < 0.5%");
        }
    }
    return dnstest::finish();
}
