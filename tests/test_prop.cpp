// dns prop, library level: the thin-airfoil polar and the planform against
// their closed forms, BEMT's own consistency identities, the lifting-line
// self-test, CSV and CLI rejections, and EQUIVALENCE with the Python codes it
// ports (the retired tools/bemt.py, tools/liftline.py) on pinned fixtures.
//
// The fixtures in tests/data/prop were produced by the Python tools on
// exactly these planform/polar CSVs (reduced XFOIL tables, 2 Re each, 7
// significant digits; the thin polar rebuilt in-process). Measured at
// port time: BEMT agrees to <= 1e-15 relative, the lifting line to <= 2e-14
// (the same Newton / hybrd / pseudo-transient path). The tolerances below
// leave room for another libm, not for another algorithm.

#include "check.hpp"
#include "cli_test_util.hpp"

#include "dns/prop/bemt.hpp"
#include "dns/prop/liftline.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#ifndef DNS_PROP_DATA
#error "DNS_PROP_DATA must be defined (tests/data/prop)"
#endif

namespace {

using namespace dns;
using dnstest::check;
using dnstest::rejected;
using std::numbers::pi;

std::string data(const std::string& f) { return std::string(DNS_PROP_DATA) + "/" + f; }

std::string slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

// A Python reference curve: J,CT,CP,CQ,eta,T_N,Q_Nm,P_W,Re75,converged with
// repr floats ('nan' allowed).
std::vector<std::vector<double>> readRef(const std::string& path)
{
    std::vector<std::vector<double>> rows;
    std::istringstream in(slurp(path));
    std::string line;
    std::getline(in, line);   // header
    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        std::vector<double> r;
        std::istringstream ls(line);
        std::string tok;
        while (std::getline(ls, tok, ','))
            r.push_back(tok == "nan" ? std::numeric_limits<double>::quiet_NaN() : std::stod(tok));
        rows.push_back(std::move(r));
    }
    return rows;
}

bool near(double a, double b, double tol, double scale)
{
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return std::fabs(a - b) <= tol * std::max(std::fabs(b), scale);
}

// --- closed forms ------------------------------------------------------------

void thinPolar()
{
    const double a0 = -2.0, cd0 = 0.02, cdm = 1.2;
    const prop::Polar p = prop::Polar::thinAirfoil(a0, cd0, cdm);
    double err = 0.0, cdErr = 0.0;
    for (double deg : {-11.9, -8.0, -2.0, 0.0, 5.0, 7.9})
    {
        const double a = deg * pi / 180.0;
        const prop::ClCd v = p.at(a, 1e5);
        err = std::max(err, std::fabs(v.cl - 2 * pi * (deg - a0) * pi / 180.0));
        cdErr = std::max(cdErr, std::fabs(v.cd - cd0));
    }
    check(err < 1e-12, std::format("thin polar: cl = 2 pi (alpha - alpha0) on the "
                                   "linear range (max err {:.1e})", err));
    check(cdErr < 1e-15, "thin polar: cd = cd0 on the linear range");
    const prop::ClCd a = p.at(0.3, 1e4), b = p.at(0.3 + 2 * pi, 3e6), c = p.at(0.3 - 4 * pi, 1);
    check(std::fabs(a.cl - b.cl) < 1e-12 && std::fabs(a.cl - c.cl) < 1e-12
              && std::fabs(a.cd - b.cd) < 1e-12,
          "thin polar: 2 pi periodic in alpha and Re-independent");
    check(std::fabs(p.at(pi / 2, 1).cd - cdm) < 1e-12,
          "thin polar: Viterna flat-plate drag cdMax at 90 deg");
    check(std::fabs(p.at(-pi, 1).cl) < 1e-12, "thin polar: cl = 0 at 180 deg");
    // Continuity at the linear-range ends (the Viterna hand-over).
    double jump = 0.0;
    for (double deg : {a0 - 10.0, a0 + 10.0, 90.0, -90.0})
    {
        const double e = 1e-9, r = deg * pi / 180.0;
        jump = std::max(jump, std::fabs(p.at(r + e, 1).cl - p.at(r - e, 1).cl));
        jump = std::max(jump, std::fabs(p.at(r + e, 1).cd - p.at(r - e, 1).cd));
    }
    check(jump < 1e-6, std::format("thin polar: continuous at the stall hand-over and "
                                   "+-90 deg (max jump {:.1e})", jump));
    check(std::fabs(prop::viternaCdMax(10) - (1.11 + 0.18)) < 1e-15
              && std::fabs(prop::viternaCdMax(80) - (1.11 + 0.9)) < 1e-15,
          "viternaCdMax = 1.11 + 0.018 AR, AR capped at 50");
}

void planform()
{
    auto p = prop::Planform::fromCsv("r_R,c_R,beta_deg\n0.2,0.2,40\n0.6,0.1,20\n1,0.1,10\n");
    check(bool(p), "planform: a valid table parses");
    if (!p) return;
    const prop::Section s0 = p->at(0.1), s1 = p->at(0.2), sx = p->at(0.5), sTip = p->at(0.6);
    check(s0.chord == 0.1 && std::fabs(s0.beta - 40 * pi / 180) < 1e-15,
          "planform: held at the root row inside it (c/R -> c/D halves)");
    check(std::fabs(s1.chord - 0.075) < 1e-15 && std::fabs(s1.beta - 30 * pi / 180) < 1e-15,
          "planform: linear between rows");
    check(sx.chord == 0.05 && sTip.chord == 0.05, "planform: held past the tip row");
    auto flat = prop::Planform::fromCsv("r_R,c_R,beta_deg\n0.1,0.2,10\n1,0.2,10\n");
    check(flat && std::fabs(flat->aspectRatio(0.2) - (0.5 - 0.1) / 0.1) < 1e-12,
          "planform: aspect ratio of a constant chord = (R - rh) / c");
    check(p->rootRR() == 0.2, "planform: rootRR is the first r/R");
}

// --- consistency of the BEMT solution -------------------------------------------

std::optional<Case> cli(std::vector<std::string> args)
{
    auto c = dnstest::runCli(std::move(args));
    if (!c) std::printf("    cli: %s\n", c.error().message.c_str());
    return c ? std::optional<Case>(std::move(*c)) : std::nullopt;
}

void bemtIdentities()
{
    auto c = cli({"prop", "--blades", "2", "--planform", data("t4412_planform.csv"),
                  "--polar", "thin", "--thin-alpha0", "-4.154480845734293", "--j",
                  "0:0.9:0.05"});
    check(bool(c), "bemt: the t4412 thin case validates");
    if (!c) return;
    auto pf = prop::Planform::fromCsv(slurp(c->prop.planformPath));
    if (!pf) { check(false, "bemt: fixture planform parses"); return; }
    prop::Case pc = c->prop;
    pc.hubRR = pf->rootRR();
    const auto polar = prop::Polar::thinAirfoil(pc.thinAlpha0Deg, pc.thinCd0,
                                                prop::viternaCdMax(pf->aspectRatio(pc.hubRR)));
    const auto bl = prop::bemtBlade(*pf, pc.blades, pc.hubRR, pc.stations, pc.op.diameter);
    const auto curve = prop::bemtCurve(pc, bl, polar);
    bool cq = true, eta = true, froude = true, mono = true;
    for (std::size_t k = 0; k < curve.rows.size(); ++k)
    {
        const auto& r = curve.rows[k];
        cq = cq && std::fabs(r.CQ - r.CP / (2 * pi)) <= 1e-15 * std::fabs(r.CP);
        if (r.J > 0 && r.CP > 0)
            eta = eta && std::fabs(r.eta - r.J * r.CT / r.CP) <= 1e-15 * std::fabs(r.eta);
        if (r.J > 0 && r.CT > 0)
            froude = froude && r.eta < 2.0 / (1.0 + std::sqrt(1.0 + 8.0 * r.CT / (pi * r.J * r.J)));
        // Below J ~ 0.3 the thin polar's +-10 deg linear range has stalled the
        // inboard blade, so CT RISES with J there (measured); past it, it falls.
        if (k > 0 && r.J >= 0.4) mono = mono && r.CT < curve.rows[k - 1].CT;
    }
    check(cq, "bemt: CQ = CP / (2 pi)");
    check(eta, "bemt: eta = J CT / CP");
    check(froude, "bemt: eta below the actuator-disk (Froude) bound at every J");
    check(mono, "bemt: CT falls monotonically with J once the blade is unstalled");
    check(curve.rows.front().CT > 0 && std::isnan(curve.rows.front().eta),
          "bemt: static thrust positive, eta(J = 0) is nan");
    check(curve.unsolved == 0, "bemt: every annulus found a bracket");
    check(curve.worstMomErr < 1e-8,
          std::format("bemt: blade loads = momentum loads in the momentum region "
                      "(max {:.1e} of peak dT/dr)", curve.worstMomErr));
    // J = 0 is the same equation, not a special case: the curve is continuous
    // into it.
    const auto r0 = prop::bemtSolve(0.0, bl, pc.op, polar);
    const auto r1 = prop::bemtSolve(1e-4, bl, pc.op, polar);
    check(std::fabs(r0.T - r1.T) < 1e-3 * r0.T, "bemt: T(J) continuous into J = 0");
}

// --- the lifting-line self-test ----------------------------------------------------

void selftest()
{
    for (const auto& k : prop::liftlineSelfTest())
        check(k.pass, "liftline self-test " + k.name + ": " + k.message);
}

// --- equivalence with the Python codes ------------------------------------------

struct Fixture
{
    const char* name;
    std::vector<std::string> op;   // blade + operating point flags
    const char* alpha0;            // the section's thin-airfoil zero-lift angle
};

void equivalence()
{
    const Fixture fx[] = {
        {"t4412", {"--blades", "2", "--diameter", "0.254", "--rpm", "6000", "--nu", "1.5e-5"},
         "-4.154480845734293"},
        {"da4002",
         {"--blades", "2", "--diameter", "0.2286", "--rpm", "5028", "--nu", "1.46e-5",
          "--hub-radius", "0.15"},
         "-1.3392506322921527"},
    };
    for (const Fixture& f : fx)
        for (const char* method : {"bemt", "liftline"})
            for (const bool thin : {false, true})
            {
                std::vector<std::string> a{"prop", "--method", method, "--planform",
                                           data(std::string(f.name) + "_planform.csv"),
                                           "--j", "0:0.95:0.05"};
                a.insert(a.end(), f.op.begin(), f.op.end());
                if (thin) a.insert(a.end(), {"--polar", "thin", "--thin-alpha0", f.alpha0});
                else a.insert(a.end(), {"--polar", data(std::string(f.name) + "_polar.csv")});
                const std::string tag = std::format("{} {} {}", f.name, method,
                                                    thin ? "thin" : "table");
                auto c = cli(a);
                if (!c) { check(false, tag + ": validates"); continue; }
                prop::Case pc = c->prop;
                auto pf = prop::Planform::fromCsv(slurp(pc.planformPath));
                if (!pf) { check(false, tag + ": planform parses"); continue; }
                if (pc.hubRR <= 0) pc.hubRR = pf->rootRR();
                std::optional<prop::Polar> polar;
                if (thin)
                    polar = prop::Polar::thinAirfoil(
                        pc.thinAlpha0Deg, pc.thinCd0,
                        prop::viternaCdMax(pf->aspectRatio(pc.hubRR)));
                else if (auto p = prop::Polar::fromCsv(slurp(pc.polarPath)))
                    polar = std::move(*p);
                if (!polar) { check(false, tag + ": polar parses"); continue; }
                std::vector<prop::CurvePoint> rows;
                const bool ll = pc.method == prop::Method::Liftline;
                if (ll)
                    rows = prop::liftlineCurve(
                               pc,
                               prop::liftGeometry(*pf, pc.hubRR, pc.ll.panels, pc.ll.hub,
                                                  pc.op.diameter),
                               *polar)
                               .rows;
                else
                    rows = prop::bemtCurve(pc,
                                           prop::bemtBlade(*pf, pc.blades, pc.hubRR,
                                                           pc.stations, pc.op.diameter),
                                           *polar)
                               .rows;
                const auto ref = readRef(data(std::format("{}_{}_{}.csv", f.name, method,
                                                          thin ? "thin" : "table")));
                if (ref.size() != rows.size())
                {
                    check(false, tag + ": same J count as the Python reference");
                    continue;
                }
                double scaleT = 0, scaleP = 0, scaleTN = 0, scaleQ = 0;
                for (const auto& r : ref)
                {
                    scaleT = std::max(scaleT, std::fabs(r[1]));
                    scaleP = std::max(scaleP, std::fabs(r[2]));
                    scaleTN = std::max(scaleTN, std::fabs(r[5]));
                    scaleQ = std::max(scaleQ, std::fabs(r[6]));
                }
                // BEMT is the same arithmetic; the lifting line differs only in
                // summation order (BLAS vs loops) inside Newton.
                const double tol = ll ? 1e-8 : 1e-10;
                double worst = 0;
                bool ok = true;
                for (std::size_t k = 0; k < rows.size(); ++k)
                {
                    const auto& r = ref[k];
                    const auto& c2 = rows[k];
                    ok = ok && r[0] == c2.J && near(c2.CT, r[1], tol, 1e-3 * scaleT)
                         && near(c2.CP, r[2], tol, 1e-3 * scaleP)
                         && near(c2.CQ, r[3], tol, 1e-3 * scaleP / (2 * pi))
                         && near(c2.T, r[5], tol, 1e-3 * scaleTN)
                         && near(c2.Q, r[6], tol, 1e-3 * scaleQ)
                         && near(c2.re75, r[8], tol, 0) && double(c2.converged) == r[9];
                    worst = std::max({worst, std::fabs(c2.CT - r[1]) / scaleT,
                                      std::fabs(c2.CP - r[2]) / scaleP});
                }
                check(ok, std::format("{}: matches the Python port over {} J (max "
                                      "|dCT|,|dCP| {:.1e} of peak)",
                                      tag, rows.size(), worst));
            }
}

// --- rejections ------------------------------------------------------------------

void csvRejections()
{
    using prop::Planform;
    using prop::Polar;
    const char* hdr = "r_R,c_R,beta_deg\n";
    struct Bad
    {
        std::string text, what;
    };
    for (const Bad& b : {Bad{"", "an empty file"},
                         Bad{"r,c,beta\n0.2,0.1,20\n1,0.1,10\n", "a wrong header"},
                         Bad{std::string(hdr) + "0.2,0.1,20\n", "a single row"},
                         Bad{std::string(hdr) + "0.5,0.1,20\n0.4,0.1,10\n", "r_R decreasing"},
                         Bad{std::string(hdr) + "0.5,0.1,20\n0.5,0.1,10\n", "r_R repeated"},
                         Bad{std::string(hdr) + "0.2,0.1,20\n1.2,0.1,10\n", "r_R > 1"},
                         Bad{std::string(hdr) + "0,0.1,20\n1,0.1,10\n", "r_R = 0"},
                         Bad{std::string(hdr) + "0.2,0,20\n1,0.1,10\n", "c_R = 0"},
                         Bad{std::string(hdr) + "0.2,0.1,90\n1,0.1,10\n", "beta = 90 deg"},
                         Bad{std::string(hdr) + "0.2,0.1,x\n1,0.1,10\n", "a non-number"},
                         Bad{std::string(hdr) + "0.2,0.1,nan\n1,0.1,10\n", "a NaN"},
                         Bad{std::string(hdr) + "0.2,0.1\n1,0.1,10\n", "a missing column"},
                         Bad{std::string(hdr) + "0.2,0.1,20,4\n1,0.1,10\n", "an extra column"}})
        check(!Planform::fromCsv(b.text), "planform rejects " + b.what);

    // A minimal valid two-Re polar, then single corruptions of it.
    auto table = [](double re, double cl) {
        return std::format("{},-180,0,0.1\n{},0,{},0.01\n{},180,0,0.1\n", re, re, cl, re);
    };
    const std::string ph = "re,alpha_deg,cl,cd\n";
    check(bool(Polar::fromCsv(ph + table(1e4, 0.3) + table(1e5, 0.5))),
          "polar: a valid two-Re table parses");
    auto two = Polar::fromCsv(ph + table(1e4, 0.3) + table(1e5, 0.5));
    if (two)
    {
        const double mid = std::sqrt(1e4 * 1e5);
        check(std::fabs(two->at(0, mid).cl - 0.4) < 1e-12,
              "polar: linear in log Re between tables");
        check(two->at(0, 1.0).cl == 0.3 && two->at(0, 1e9).cl == 0.5,
              "polar: Re clamped to the table range, never extrapolated");
    }
    for (const Bad& b :
         {Bad{"", "an empty file"}, Bad{"re,alpha,cl,cd\n" + table(1e4, 0.3), "a wrong header"},
          Bad{ph + table(1e5, 0.3) + table(1e4, 0.5), "Re descending"},
          Bad{ph + table(1e4, 0.3) + table(1e5, 0.5) + table(1e4, 0.3), "a split Re group"},
          Bad{ph + "1e4,-180,0,0.1\n1e4,0,0.3,0.01\n1e4,-10,0,0.1\n1e4,180,0,0.1\n",
              "alpha not increasing"},
          Bad{ph + "1e4,-170,0,0.1\n1e4,180,0,0.1\n", "alpha not reaching -180"},
          Bad{ph + "1e4,-180,0,0.1\n1e4,170,0,0.1\n", "alpha not reaching +180"},
          Bad{ph + "1e4,-180,0,0.1\n1e4,0,0.3,0\n1e4,180,0,0.1\n", "cd = 0"},
          Bad{ph + table(0, 0.3), "Re = 0"},
          Bad{ph + "1e4,-180,0,0.1\n1e4,0,inf,0.01\n1e4,180,0,0.1\n", "an infinite cl"},
          Bad{ph, "no data rows"}})
        check(!Polar::fromCsv(b.text), "polar rejects " + b.what);
}

void cliRejections()
{
    const std::vector<std::string> ok{"prop", "--planform", "p.csv", "--polar", "thin"};
    auto with = [&](std::initializer_list<std::string> extra) {
        std::vector<std::string> v = ok;
        v.insert(v.end(), extra);
        return v;
    };
    auto c = dnstest::runCli(ok);
    check(bool(c), "cli: a minimal prop case validates without touching the files");
    if (c)
    {
        check(c->prop.js.size() == 21 && c->prop.js.front() == 0.0 && c->prop.js.back() == 1.0,
              "cli: bemt's default J list is 0:1:0.05 (21 points)");
        check(c->prop.outCsv.empty(), "cli: no --out = no CSV");
    }
    auto cl = dnstest::runCli(with({"--method", "liftline"}));
    check(cl && cl->prop.js.size() == 11, "cli: liftline's default J list is 0:1.0:0.1");
    auto cj = dnstest::runCli(with({"--j", "0.3,0.1,0.3"}));
    check(cj && cj->prop.js == std::vector<double>{0.1, 0.3}, "cli: a J list is sorted and "
                                                               "de-duplicated");
    auto cr = dnstest::runCli(with({"--j", "0:0.3:0.1"}));
    check(cr && cr->prop.js == std::vector<double>{0.0, 0.1, 0.2, 0.3},
          "cli: a J range includes its stop, rounded like Python's round(x, 12)");

    rejected(with({"--bogus"}), 1, "cli: an unknown flag is exit 1");
    rejected(with({"--backend", "cpu"}), 1, "cli: prop takes no --backend (never a GPU)");
    rejected(with({"--n", "64"}), 1, "cli: prop takes no grid flags");
    rejected(with({"--seed", "2"}), 1, "cli: prop takes no --seed");
    rejected(with({"--emit", "stats"}), 1, "cli: prop takes no --emit");
    rejected(with({"--method", "vortex"}), 1, "cli: an unknown --method is exit 1");
    rejected(with({"--hub", "ring", "--method", "liftline"}), 1, "cli: an unknown --hub");
    rejected(with({"--rpm", "fast"}), 1, "cli: a non-numeric value is exit 1");
    rejected({"prop", "--polar", "thin"}, 2, "cli: missing --planform is exit 2");
    rejected({"prop", "--planform", "p.csv"}, 2, "cli: missing --polar is exit 2");
    rejected(with({"--j", "-0.1"}), 2, "cli: J < 0 is exit 2");
    rejected(with({"--j", "0,-0.5,1"}), 2, "cli: a negative J in a list is exit 2");
    rejected(with({"--j", "1:0:0.1"}), 2, "cli: a reversed J range is exit 2");
    rejected(with({"--j", "0:1:0"}), 2, "cli: a zero J step is exit 2");
    rejected(with({"--j", "0:1"}), 2, "cli: a two-field J range is exit 2");
    rejected(with({"--j", "0,,1"}), 2, "cli: an empty J list item is exit 2");
    rejected(with({"--j", "zero"}), 2, "cli: a non-numeric J is exit 2");
    rejected(with({"--j", "0:1e9:1e-9"}), 2, "cli: an absurd J count is exit 2");
    rejected(with({"--panels", "20"}), 2, "cli: a lifting-line flag with bemt is exit 2");
    rejected(with({"--method", "liftline", "--stations-bemt", "20"}), 2,
             "cli: a bemt flag with liftline is exit 2");
    rejected({"prop", "--selftest"}, 2, "cli: --selftest needs --method liftline");
    rejected({"prop", "--method", "liftline", "--selftest", "--planform", "p.csv"}, 2,
             "cli: --selftest rejects the inputs it would not read");
    check(bool(dnstest::runCli({"prop", "--method", "liftline", "--selftest"})),
          "cli: --method liftline --selftest validates alone");
    rejected({"prop", "--planform", "p.csv", "--polar", "x.csv", "--thin-cd0", "0.03"}, 2,
             "cli: --thin-cd0 without --polar thin is exit 2");
    rejected(with({"--thin-cd0", "0"}), 2, "cli: --thin-cd0 0 is exit 2");
    rejected(with({"--thin-alpha0", "12"}), 2, "cli: --thin-alpha0 outside (-10,10) is exit 2");
    rejected(with({"--hub-radius", "1.2"}), 2, "cli: --hub-radius >= 1 is exit 2");
    rejected(with({"--hub-radius", "0"}), 2, "cli: --hub-radius 0 is exit 2");
    rejected(with({"--blades", "0"}), 2, "cli: --blades 0 is exit 2");
    for (const char* f : {"--diameter", "--rpm", "--rho", "--nu", "--sound"})
        rejected(with({f, "0"}), 2, std::string("cli: ") + f + " 0 is exit 2");
    rejected(with({"--stations-bemt", "3"}), 2, "cli: --stations-bemt < 4 is exit 2");
    rejected(with({"--method", "liftline", "--panels", "3"}), 2, "cli: --panels < 4 is exit 2");
    rejected(with({"--method", "liftline", "--seg-deg", "40"}), 2,
             "cli: --seg-deg > 30 is exit 2");
    rejected(with({"--method", "liftline", "--align-iters", "-1"}), 2,
             "cli: --align-iters < 0 is exit 2");
    rejected(with({"--method", "liftline", "--wake-length", "0"}), 2,
             "cli: --wake-length 0 is exit 2");
    rejected(with({"--method", "liftline", "--core", "-0.1"}), 2, "cli: --core < 0 is exit 2");
    rejected(with({"--dump-span", "-1"}), 2, "cli: --dump-span < 0 is exit 2");
    rejected(with({"--out", ""}), 2, "cli: an empty --out is exit 2");
}

} // namespace

int main()
{
    thinPolar();
    planform();
    bemtIdentities();
    selftest();
    equivalence();
    csvRejections();
    cliRejections();
    return dnstest::finish();
}
