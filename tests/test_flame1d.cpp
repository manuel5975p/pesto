// P0 chemistry oracle (dns/chem.hpp, docs/chem.md): the FP64 phase-plane
// eigenvalue of the 1D freely propagating single-step flame. (a) step-count
// convergence, (b) the exact sqrt(A) similarity, (c) the dimensional form
// against Cantera 3.2.0 on the SAME model (tests/data/chem, written by
// tools/chem/cantera_flame.py; Cantera is never a build dependency), (d) the
// profile table's monotonicity and thickness.
#include "dns/chem.hpp"

#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
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
