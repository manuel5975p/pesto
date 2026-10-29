// dns flame parse + validate (docs/chem.md): the resolved grid in oracle
// thermal thicknesses, the dry-run text (label, Lambda), every rejection BY
// NAME with code 2, and the CLI-level ones (unknown --mode value, the
// unregistered --lewis) with code 1.
#include "dns/config.hpp"
#include "dns/flame1d.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using dnstest::check;
using dnstest::runCli;

// Rejected with `code`, and the error names `flag`.
void rejects(std::vector<std::string> args, int code, const std::string& flag,
             const std::string& what)
{
    auto c = runCli(std::move(args));
    const bool ok = !c && c.error().code == code &&
                    (c.error().field == flag ||
                     c.error().message.find(flag) != std::string::npos);
    if (!ok && !c)
        std::printf("    got code %d field '%s': %s\n", c.error().code,
                    c.error().field.c_str(), c.error().message.c_str());
    check(ok, what);
}

} // namespace

int main()
{
    using namespace dns;
    {
        auto c = runCli({"flame"});
        check(c.has_value(), "defaults validate");
        if (c) {
            const double dth = c->flameEig.dthOverDD;
            check(std::abs(c->flameEig.Lambda / 3656401.098 - 1) < 1e-6,
                  "Lambda = 3656401.098 (sigma 5, beta 8)");
            check(c->grid.nx == 22 * 15 + 2 * 16 && c->grid.ny == 4 &&
                      c->grid.nz == 1,
                  "grid 362 x 4 x 1 (22 delta_th at 15 cells + 2 x 16 sink)");
            check(std::abs(c->grid.hx - dth / 15) < 1e-15 &&
                      std::abs(c->grid.hy - c->grid.hx) < 1e-15,
                  "h = delta_th / 15, square cells");
            check(std::abs(c->flameXf0 - (16 * c->grid.hx + 12 * dth)) < 1e-12,
                  "x_f0 = 16 h + 12 delta_th");
            check(std::abs(c->flameTu - 1 / (1.4 * 1e-4)) < 1e-9,
                  "Tu = 1/(gamma M^2) at M 0.01");
            check(c->nscbcSink == 16 && c->nscbcEtaIn == 100,
                  "sink 16, etaIn 100 by default");
            const std::string d = c->describe();
            check(d.find(flameRunLabel(*c)) != std::string::npos &&
                      d.find("MC-limited MUSCL") != std::string::npos &&
                      d.find("(inflated; CH4/air 1.15e-3)") != std::string::npos,
                  "dry-run carries the label (model, inflated M, scheme)");
            check(d.find("3656401.098") != std::string::npos,
                  "dry-run prints Lambda");
            std::printf("%s", d.c_str());
            check(c->warnings.empty(), "no warning at the defaults");

            // The IC: the oracle flame on the grid, fresh Y = 1, burnt p.
            CompState q = CompState::zeros(c->grid);
            std::vector<double> ry;
            flameIC(*c, q, ry);
            const double T0 = c->flameTu, sig = 5;
            const std::size_t i0 = 0, iN = std::size_t(c->grid.nx - 1);
            auto Tat = [&](std::size_t i) {
                const double r = q.rho[i], u = q.mx[i] / r;
                const double p = 0.4 * (q.E[i] - 0.5 * r * u * u);
                return p / r;
            };
            check(std::abs(Tat(i0) / T0 - 1) < 1e-6 &&
                      std::abs(ry[i0] / q.rho[i0] - 1) < 1e-6,
                  "IC fresh end: T = Tu, Y = 1");
            check(std::abs(Tat(iN) / (sig * T0) - 1) < 1e-6 &&
                      std::abs(ry[iN]) < 1e-6,
                  "IC burnt end: T = sigma Tu, Y = 0");
            check(std::abs(q.mx[i0] - 1) < 1e-12 && std::abs(q.mx[iN] - 1) < 1e-12,
                  "IC mass flux = 1 (S_L) everywhere");
        }
    }
    {
        auto c = runCli({"flame", "--mach", "0.03"});
        check(c && c->warnings.size() == 1, "--mach 0.03 warns (above 0.02)");
        auto s = runCli({"flame", "--visc", "sutherland", "--tinf", "300"});
        check(s && s->sutherland && std::abs(s->flameModel.sOverTu - 0.368) < 1e-12,
              "sutherland with --tinf 300: S/Tu = 110.4/300");
        auto b = runCli({"flame", "--cells-per-delta", "30", "--zeldovich", "10"});
        check(b && b->grid.nx == 22 * 30 + 32 &&
                  std::abs(b->flameEig.Lambda / 69343676.77 - 1) < 1e-6,
              "--cells-per-delta 30 --zeldovich 10 resolve");
    }
    {
        // --wrinkle: off is exactly the planar IC; on displaces the front by
        // flameWrinkle(y), mode sum with the documented amplitude.
        check(flameWrinkle(0, 8, 3.0, 10.0) == 0, "wrinkle A = 0 is exactly 0");
        double s2 = 0;
        const int M = 4096;
        for (int j = 0; j < M; ++j) {
            const double w = flameWrinkle(1.0, 16, (j + 0.5) * 10.0 / M, 10.0);
            s2 += w * w;
        }
        check(std::abs(std::sqrt(s2 / M) - std::sqrt(0.5)) < 1e-9,
              "wrinkle rms = A/sqrt(2) over the box height");
        check(std::abs(flameWrinkle(1.0, 16, 0.3, 10.0) -
                       flameWrinkle(1.0, 16, 10.3, 10.0)) < 1e-12,
              "wrinkle is Ly-periodic");

        auto p = runCli({"flame", "--ny", "128"});
        auto w = runCli({"flame", "--ny", "128", "--wrinkle", "0.5",
                         "--wrinkle-modes", "4"});
        check(p && w && p->flameWrinkle == 0 && w->flameWrinkle == 0.5 &&
                  w->flameWrinkleModes == 4,
              "--wrinkle 0.5 --wrinkle-modes 4 resolve");
        if (p && w) {
            CompState qp, qw;
            std::vector<double> rp, rw;
            flameIC(*p, qp, rp);
            flameIC(*w, qw, rw);
            const Grid& g = p->grid;
            bool rowsEqual = true, rowsDiffer = false;
            for (int i = 0; i < g.nx; ++i) {
                rowsEqual = rowsEqual && qp.rho[g.idx(i, 0, 0)] == qp.rho[g.idx(i, 77, 0)];
                rowsDiffer = rowsDiffer || qw.rho[g.idx(i, 0, 0)] != qw.rho[g.idx(i, 77, 0)];
            }
            check(rowsEqual && rowsDiffer, "planar IC is y-uniform, wrinkled is not");
            check(p->describe().find("wrinkle") == std::string::npos &&
                      w->describe().find("wrinkle 0.5 delta_th over modes 1..4") !=
                          std::string::npos,
                  "dry-run names the wrinkle only when on");
        }
    }

    rejects({"flame", "--re", "100"}, 2, "--re", "--re rejected by name");
    rejects({"flame", "--ft", "3"}, 2, "--ft", "--ft rejected by name");
    rejects({"flame", "--n", "256"}, 2, "--n", "--n rejected by name");
    rejects({"flame", "--physics", "incompressible"}, 2, "--physics",
            "--physics incompressible rejected (no low-Mach branch)");
    rejects({"flame", "--nz", "4"}, 2, "--nz", "--nz 4 rejected (1 or >= 8)");
    rejects({"flame", "--ny", "3"}, 2, "--ny", "--ny 3 rejected");
    rejects({"flame", "--mach", "0.06"}, 2, "--mach", "--mach 0.06 rejected");
    rejects({"flame", "--mach", "0"}, 2, "--mach", "--mach 0 rejected");
    rejects({"flame", "--expansion", "1"}, 2, "--expansion", "--expansion 1 rejected");
    rejects({"flame", "--expansion", "11"}, 2, "--expansion", "--expansion 11 rejected");
    rejects({"flame", "--zeldovich", "1"}, 2, "--zeldovich", "--zeldovich 1 rejected");
    rejects({"flame", "--cells-per-delta", "3"}, 2, "--cells-per-delta",
            "--cells-per-delta 3 rejected");
    rejects({"flame", "--upstream", "5"}, 2, "--upstream", "--upstream 5 rejected");
    rejects({"flame", "--downstream", "3"}, 2, "--downstream",
            "--downstream 3 rejected");
    rejects({"flame", "--inflow-speed", "0"}, 2, "--inflow-speed",
            "--inflow-speed 0 rejected");
    rejects({"flame", "--visc", "sutherland"}, 2, "--tinf",
            "sutherland without --tinf rejected");
    rejects({"flame", "--tinf", "300"}, 2, "--tinf",
            "--tinf without sutherland rejected");
    rejects({"flame", "--prandtl", "0"}, 2, "--prandtl", "--prandtl 0 rejected");
    rejects({"flame", "--nscbc-sink", "-1"}, 2, "--nscbc-sink",
            "--nscbc-sink -1 rejected");
    rejects({"flame", "--nscbc-sink-amp", "3"}, 2, "--nscbc-sink-amp",
            "--nscbc-sink-amp 3 rejected");
    rejects({"flame", "--nscbc-etain", "0"}, 2, "--nscbc-etain",
            "--nscbc-etain 0 rejected");
    rejects({"flame", "--tt", "0"}, 2, "--tt", "--tt 0 rejected");
    rejects({"flame", "--ny", "128", "--wrinkle", "-1"}, 2, "--wrinkle",
            "--wrinkle -1 rejected");
    rejects({"flame", "--ny", "128", "--wrinkle-modes", "4"}, 2,
            "--wrinkle-modes", "--wrinkle-modes without --wrinkle rejected");
    rejects({"flame", "--ny", "128", "--wrinkle", "0.5", "--wrinkle-modes", "17"},
            2, "--wrinkle-modes", "--wrinkle-modes 17 > ny/8 rejected");
    rejects({"flame", "--ny", "128", "--wrinkle", "0.5", "--wrinkle-modes", "0"},
            2, "--wrinkle-modes", "--wrinkle-modes 0 rejected");
    rejects({"flame", "--ny", "128", "--wrinkle", "2", "--wrinkle-modes", "16"},
            2, "--wrinkle", "--wrinkle 2 x sqrt(16) past half of --downstream rejected");
    {
        // --mode bunsen: the box from the burner, the dry-run geometry, the
        // inflow profile and IC (dns/bunsen.hpp).
        auto c = runCli({"flame", "--mode", "bunsen"});
        check(c && c->flameBunsen, "--mode bunsen validates at its defaults");
        if (c) {
            const double dth = c->flameEig.dthOverDD, h = c->grid.hx;
            const BunsenGeom& b = c->bunsen;
            check(std::abs(h - dth / 8) < 1e-15, "bunsen default: 8 cells/delta_th");
            check(c->grid.ny == 2 * 18 * 8 && c->grid.nx == 45 * 8 + 32,
                  "bunsen grid 392 x 288 (W 12 + 2 x 12 coflow; 45 delta_th "
                  "+ 2 x 16 sink)");
            check(std::abs(b.W - 12 * dth) < 1e-12 && b.uJet == 5 &&
                      b.uCo == 1 && b.tCo == 5 && b.pilot == 0 &&
                      std::abs(b.yc - 0.5 * c->grid.ny * h) < 1e-12,
                  "bunsen defaults: W 12, U 5, burnt coflow at 1, centred");
            check(std::abs(b.coneH - 6 * std::sqrt(24.0) * dth) < 1e-9,
                  "ideal cone height (W/2) sqrt(U^2 - 1)");
            const std::string d = c->describe();
            check(d.find("2D slot Bunsen") != std::string::npos &&
                      d.find("alpha = 11.54 deg") != std::string::npos &&
                      d.find("burnt") != std::string::npos,
                  "dry-run prints the burner, the ideal angle, the pilot");
            CompState q;
            std::vector<double> ry;
            bunsenIC(*c, q, ry);
            const Grid& g = c->grid;
            const std::size_t jc = std::size_t(g.ny / 2);
            const std::size_t a = g.idx(0, int(jc), 0), o = g.idx(0, 0, 0);
            const std::size_t t = g.idx(g.nx - 1, int(jc), 0);
            // The axis sits (W/2) cos(alpha) = 9.4 delta_D inside the flank:
            // the preheat tail e^-9.4 still shows at 1e-4.
            check(std::abs(ry[a] / q.rho[a] - 1) < 1e-3 &&
                      std::abs(q.mx[a] / q.rho[a] - 5) < 5e-3,
                  "IC at the slot exit: Y = 1, u = U_jet (to the preheat tail)");
            check(ry[o] / q.rho[o] < 1e-9 &&
                      std::abs(q.rho[o] * 5 - 1) < 1e-9,
                  "IC in the coflow: Y = 0, rho = rho_u / sigma (burnt)");
            check(ry[t] / q.rho[t] < 1e-6 && std::abs(q.mx[t] - 5) < 1e-9,
                  "IC past the tip: burnt, the column's mass flux");
        }
        auto p = runCli({"flame", "--mode", "bunsen", "--coflow-temp", "1",
                         "--pilot-width", "3", "--pilot-speed", "5"});
        check(p && p->bunsen.tCo == 1 && p->bunsen.tPilot == 5 &&
                  p->bunsen.uPilot == 5 &&
                  p->grid.ny == 2 * int(std::ceil((6 + 3 + 12) * 8.0)),
              "pilot strips: temp defaults to sigma, box widened by them");
        auto e = runCli({"flame", "--mode", "bunsen", "--expansion", "7"});
        check(e && e->bunsen.tCo == 7, "--coflow-temp default follows sigma");
        auto L = runCli({"flame", "--mode", "bunsen", "--jet-speed", "3",
                         "--cells-per-delta", "12", "--length", "30"});
        check(L && L->grid.nx == 30 * 12 + 32, "--length 30 at 12 cells");
    }
    for (const char* f : {"--upstream", "--downstream", "--inflow-speed",
                          "--wrinkle", "--wrinkle-modes", "--ny", "--profile-dt"})
        rejects({"flame", "--mode", "bunsen", f, "1"}, 2, f,
                std::string(f) + " rejected by name with --mode bunsen");
    for (const char* f : {"--slot-width", "--jet-speed", "--coflow-speed",
                          "--coflow-width", "--coflow-temp", "--pilot-width",
                          "--pilot-temp", "--pilot-speed", "--slot-edge",
                          "--length", "--pulse", "--pulse-freq"})
        rejects({"flame", f, "1"}, 2, f,
                std::string(f) + " rejected by name without --mode bunsen");
    auto B = [](std::vector<std::string> extra) {
        std::vector<std::string> a{"flame", "--mode", "bunsen"};
        a.insert(a.end(), extra.begin(), extra.end());
        return a;
    };
    rejects(B({"--slot-width", "3"}), 2, "--slot-width", "--slot-width 3 rejected");
    rejects(B({"--jet-speed", "1.2"}), 2, "--jet-speed", "--jet-speed 1.2 rejected");
    rejects(B({"--jet-speed", "8", "--mach", "0.05"}), 2, "--jet-speed",
            "jet Mach 0.4 rejected");
    rejects(B({"--coflow-speed", "6"}), 2, "--coflow-speed",
            "--coflow-speed above the jet rejected");
    rejects(B({"--coflow-width", "1"}), 2, "--coflow-width",
            "--coflow-width 1 rejected");
    rejects(B({"--coflow-temp", "0.5"}), 2, "--coflow-temp",
            "--coflow-temp 0.5 rejected");
    rejects(B({"--coflow-temp", "6"}), 2, "--coflow-temp",
            "--coflow-temp above sigma rejected");
    rejects(B({"--pilot-temp", "3"}), 2, "--pilot-temp",
            "--pilot-temp without --pilot-width rejected");
    rejects(B({"--pilot-width", "2", "--pilot-speed", "9"}), 2, "--pilot-speed",
            "--pilot-speed above the jet rejected");
    rejects(B({"--slot-edge", "0.1"}), 2, "--slot-edge",
            "--slot-edge below one cell rejected");
    rejects(B({"--length", "20"}), 2, "--length",
            "--length shorter than the cone rejected");
    rejects(B({"--pulse", "0.6", "--pulse-freq", "1"}), 2, "--pulse",
            "--pulse 0.6 rejected");
    rejects(B({"--pulse", "0.1"}), 2, "--pulse-freq",
            "--pulse without --pulse-freq rejected");
    rejects(B({"--pulse-freq", "1"}), 2, "--pulse-freq",
            "--pulse-freq without --pulse rejected");
    rejects({"flame", "--mode", "counterflow"}, 1, "premixed",
            "an unbuilt --mode is a CLI error listing premixed");
    rejects({"flame", "--lewis", "0.8"}, 1, "--lewis",
            "--lewis is not registered (no Le != 1 oracle)");
    rejects({"flame", "--body", "cylinder"}, 1, "--body",
            "--body is not a flame flag");

    // 3D premixed and --turb (docs/chem.md, "3D and turbulence").
    {
        auto c = runCli({"flame", "--nz", "16", "--ny", "16",
                         "--cells-per-delta", "8"});
        check(c && c->grid.nz == 16 && std::abs(c->grid.hz - c->grid.hx) < 1e-15,
              "--nz 16: a 3D premixed box, hz = h");
        if (c) {
            const std::string d = c->describe();
            check(d.find("3D planar premixed flame") != std::string::npos &&
                      d.find("[3D, y and z periodic, hz = h]") != std::string::npos,
                  "3D dry-run: mode and grid lines");
            check(flameRunLabel(*c).find("; 3D nz=16]") != std::string::npos &&
                      flameRunLabel(*c).starts_with("DNS ["),
                  "3D label: DNS, '3D nz=16' in the bracket");
        }
        auto t = runCli({"flame", "--nz", "120", "--ny", "120",
                         "--cells-per-delta", "8", "--mach", "0.02",
                         "--upstream", "10", "--downstream", "8", "--turb",
                         "4", "--hold-flame", "--seed", "3",
                         "--inflow-speed", "2.5"});
        check(t.has_value(), "the T2 case validates");
        if (t) {
            const double dth = t->flameEig.dthOverDD;
            check(t->flameTurb == 4 && std::abs(t->flameTurbL - 2.5 * dth) < 1e-12 &&
                      t->flameTurbNbx == 120 && t->flameHold && t->seed == 3,
                  "--turb 4: u', l_t = 2.5 delta_th (default), L_bx = L_z, hold, "
                  "seed");
            const std::string l = flameRunLabel(*t);
            check(l.find("synthetic HIT inflow u'/S_L=4, l_t=2.5 delta_th") !=
                          std::string::npos &&
                      l.find("seed 3") != std::string::npos &&
                      l.find("flame held by inflow control") != std::string::npos,
                  "turbulent label: u', l_t, seed, hold");
            const std::string d = t->describe();
            check(d.find("Re_t = u' l_t/nu = 22.25") != std::string::npos &&
                      d.find("Peters 1999") != std::string::npos,
                  "turbulent dry-run: Re_t and the correlations");
            check(t->warnings.empty(), "T2 resolved: no h/eta warning");
        }
        auto u = runCli({"flame", "--nz", "60", "--ny", "60",
                         "--cells-per-delta", "4", "--mach", "0.005", "--turb",
                         "16", "--turb-length", "2.5"});
        check(u && !u->warnings.empty() &&
                  flameRunLabel(*u).starts_with("UNDER-RESOLVED DNS"),
              "h/eta > 2: warned and labelled UNDER-RESOLVED DNS");
    }
    rejects({"flame", "--nz", "6"}, 2, "--nz", "--nz 6 rejected (1 or >= 8)");
    rejects({"flame", "--mode", "bunsen", "--nz", "6"}, 2, "--nz",
            "--nz 6 rejected with --mode bunsen (1 or >= 8)");
    rejects({"flame", "--mode", "chamber", "--nz", "16"}, 2, "--nz",
            "--nz 16 rejected with --mode chamber");
    // 3D slot Bunsen (docs/chem.md, "3D turbulent slot Bunsen").
    {
        auto b = runCli({"flame", "--mode", "bunsen", "--nz", "16"});
        check(b && b->grid.nz == 16 && b->grid.hz == b->h &&
                  b->grid.ny == 288 && b->flameTurb == 0,
              "--mode bunsen --nz 16: z periodic at hz = h, the 2D y extent");
        if (b)
            check(flameRunLabel(*b).find("3D slot Bunsen (spanwise periodic "
                                         "L_z=") != std::string::npos &&
                      b->describe().find("3D slot Bunsen flame (dns flame") !=
                          std::string::npos,
                  "3D bunsen: label and dry-run name the 3D burner");
        auto t = runCli({"flame", "--mode", "bunsen", "--jet-speed", "20",
                         "--slot-width", "24", "--coflow-width", "16",
                         "--mach", "0.01", "--les", "1.35", "--sgs", "sigma",
                         "--turb", "3", "--turb-length", "6",
                         "--cells-per-delta", "2.5", "--tfles", "3", "--nz",
                         "120", "--length", "100", "--slot-edge", "0.5",
                         "--seed", "2"});
        check(t.has_value(), "the P3 turbulent slot Bunsen LES validates");
        if (t) {
            const double dth = t->flameEig.dthOverDD;
            check(t->grid.nx == 282 && t->grid.ny == 140 && t->grid.nz == 120 &&
                      t->flameTurb == 3 &&
                      std::abs(t->flameTurbL - 6 * dth) < 1e-12 &&
                      t->flameTurbNbx == 120 && t->seed == 2 && t->lesLabel &&
                      t->flameTfles == 3 && t->flameCharlette,
                  "P3 case: 282 x 140 x 120, u' 3, l_t 6, L_bx = L_z, seed, "
                  "LES + TFLES 3 + Charlette");
            // Damkoehler S_T = 4: cone (W/2) sqrt((U/4)^2 - 1) = 12 sqrt(24).
            check(std::abs(t->bunsen.coneH / dth - 12 * std::sqrt(24.0)) < 1e-9,
                  "--turb: the IC cone uses Damkoehler's S_T = 1 + u'");
            const std::string l = flameRunLabel(*t);
            check(l.starts_with("LES [sigma-model Csigma=1.35") &&
                      l.find("TFLES [F=3") != std::string::npos &&
                      l.find("synthetic HIT jet inflow u'/S_L=3") !=
                          std::string::npos &&
                      l.find("x jet weight J") != std::string::npos &&
                      l.find("M_flame=0.01 (inflated") != std::string::npos &&
                      l.find("3D slot Bunsen") != std::string::npos,
                  "P3 label: LES model + Cs, TFLES F, jet turbulence, Mach "
                  "inflation, 3D burner");
            const std::string d = t->describe();
            check(d.find("Re_jet = U W/nu = 1068") != std::string::npos &&
                      d.find("  tfles      F_max = 3") != std::string::npos &&
                      d.find("  les        mu_t") != std::string::npos,
                  "P3 dry-run: Re_jet, the tfles and les lines");
            check(std::ranges::any_of(t->warnings,
                                      [](const std::string& w) {
                                          return w.find("Damkoehler") !=
                                                 std::string::npos;
                                      }),
                  "--turb with bunsen warns that the cone uses Damkoehler's "
                  "S_T");
        }
    }
    rejects({"flame", "--mode", "bunsen", "--nz", "16", "--turb", "3",
             "--jet-speed", "4.5"},
            2, "--turb", "bunsen --turb: U_jet <= 1.25 S_T (no cone) rejected");
    rejects({"flame", "--mode", "bunsen", "--nz", "16", "--turb", "2",
             "--turb-length", "6"},
            2, "--turb-length", "bunsen --turb: L_z < 6 l_t rejected");
    rejects({"flame", "--mode", "bunsen", "--nz", "16", "--turb", "2",
             "--length", "10"},
            2, "--length", "bunsen --turb: --length below the turbulent cone "
                           "rejected");
    rejects({"flame", "--mode", "bunsen", "--nz", "200", "--jet-speed", "20",
             "--mach", "0.0125", "--turb", "4"},
            2, "--turb", "bunsen --turb: M (U_jet + 3 u') > 0.3 rejected");
    rejects({"flame", "--mode", "bunsen", "--nz", "16", "--les", "1.35"}, 2,
            "--les", "bunsen --les without an explicit --sgs rejected");
    rejects({"flame", "--mode", "diffusion", "--nz", "16"}, 2, "--nz",
            "--nz 16 rejected with --mode diffusion");
    rejects({"flame", "--turb", "2"}, 2, "--turb", "--turb at nz 1 rejected");
    rejects({"flame", "--nz", "16", "--ny", "16", "--turb", "-1"}, 2, "--turb",
            "--turb -1 rejected");
    rejects({"flame", "--nz", "16", "--ny", "16", "--turb-length", "1"}, 2,
            "--turb-length", "--turb-length without --turb rejected");
    rejects({"flame", "--nz", "16", "--ny", "16", "--hold-flame"}, 2,
            "--hold-flame", "--hold-flame without --turb rejected");
    rejects({"flame", "--seed", "3"}, 2, "--seed",
            "--seed without --turb rejected");
    rejects({"flame", "--nz", "64", "--ny", "64", "--cells-per-delta", "8",
             "--turb", "2", "--turb-length", "0.4"},
            2, "--turb-length", "l_t below 4 cells rejected");
    rejects({"flame", "--nz", "64", "--ny", "64", "--cells-per-delta", "8",
             "--turb", "2", "--turb-length", "2"},
            2, "--turb-length", "L_y = 4 l_t (< 6 l_t) rejected");
    rejects({"flame", "--nz", "120", "--ny", "120", "--cells-per-delta", "8",
             "--mach", "0.05", "--turb", "2"},
            2, "--turb", "M (U_in + 3 u') > 0.3 rejected");
    for (const char* m : {"bunsen", "diffusion", "candle", "chamber"})
        for (const char* f : {"--turb", "--turb-length", "--seed"})
            rejects({"flame", "--mode", m, f, "1"}, 2, f,
                    std::string(f) + " rejected by name with --mode " + m);
    rejects({"flame", "--mode", "bunsen", "--hold-flame"}, 2, "--hold-flame",
            "--hold-flame rejected with --mode bunsen");
    rejects({"flame", "--frame-start", "1"}, 2, "--frame-start",
            "--frame-start without a frame --emit rejected");
    rejects({"flame", "--frame-start", "20", "--emit", "bin"}, 2,
            "--frame-start", "--frame-start past --tt rejected");
    {
        auto f = runCli({"flame", "--frame-start", "2", "--emit", "bin"});
        check(f && f->frameStart == 2, "--frame-start 2 with --emit bin");
        auto z = runCli({"flame"});
        check(z && z->frameStart == 0, "frames start at 0 by default");
    }
    rejects({"flow", "--body", "cylinder", "--turb", "1"}, 1, "--turb",
            "--turb is not a flow flag");

    // LES species and TFLES (docs/chem.md, "LES species and TFLES").
    {
        auto l = runCli({"flame", "--nz", "16", "--ny", "16", "--les", "1.35",
                         "--sgs", "sigma", "--tfles", "4", "--cells-per-delta",
                         "1.875", "--upstream", "30", "--downstream", "20"});
        check(l.has_value(), "LES sigma + TFLES 4 at 1.875 cells validates");
        if (l) {
            check(l->lesLabel && l->cs == 1.35 &&
                      l->sgsModel == SgsModel::Sigma && l->flameTfles == 4 &&
                      !l->flameTflesUniform && l->flameCharlette &&
                      l->flameTflesDilate == 19,
                  "cs 1.35 sigma, F 4, dynamic, charlette by default with "
                  "--les, dilation ceil(2.5 F cells) = 19");
            const std::string lb = flameRunLabel(*l);
            check(lb.starts_with("LES [sigma-model Csigma=1.35; species SGS "
                                 "flux mu_t/Sc_t, Sc_t=Pr_t=0.9") &&
                      lb.find("+ TFLES [F=4, dynamic sensor") !=
                          std::string::npos &&
                      lb.find("Charlette 2002 efficiency beta=0.5") !=
                          std::string::npos &&
                      lb.find("M_flame=0.01 (inflated") != std::string::npos,
                  "label: LES model + Cs + species SGS + TFLES F, sensor, "
                  "efficiency, Mach inflation");
            const std::string d = l->describe();
            check(d.find("  les        mu_t = rho (C Delta)^2") !=
                          std::string::npos &&
                      d.find("  tfles      F_max = 4: 7.5 cells across") !=
                          std::string::npos,
                  "dry-run: les and tfles lines");
            check(l->meta.sgs == "sigma", "manifest sgs = sigma");
        }
        auto t = runCli({"flame", "--tfles", "4", "--cells-per-delta", "1.875",
                         "--tfles-sensor", "uniform"});
        check(t && t->flameTfles == 4 && t->flameTflesUniform &&
                  !t->flameCharlette && !t->lesLabel,
              "1D --tfles 4 uniform: no efficiency without --les");
        if (t) {
            check(flameRunLabel(*t).starts_with(
                      "thickened flame TFLES [F=4, uniform sensor (Omega=1), "
                      "no efficiency (E=1), no SGS model]"),
                  "TFLES without LES: labelled thickened flame, no SGS model");
            check(t->warnings.size() == 0, "7.5 thickened cells: no warning");
        }
        auto w = runCli({"flame", "--tfles", "2", "--cells-per-delta", "2.5"});
        check(w && !w->warnings.empty(),
              "5 cells across the thickened flame: warned (below 6)");
        auto b = runCli({"flame", "--mode", "bunsen", "--tfles", "2",
                         "--cells-per-delta", "4"});
        check(b && b->flameTfles == 2 &&
                  flameRunLabel(*b).find("TFLES [F=2") != std::string::npos,
              "--mode bunsen --tfles 2 (2D) validates");
        auto s = runCli({"flame", "--nz", "16", "--ny", "16", "--les", "0.17",
                         "--sgs", "smag", "--tfles", "2", "--cells-per-delta",
                         "4", "--efficiency", "none"});
        check(s && !s->flameCharlette &&
                  flameRunLabel(*s).starts_with("LES [Smagorinsky Cs=0.17"),
              "Smagorinsky label; --efficiency none with --les");
        auto o = runCli({"flame", "--nz", "16", "--ny", "16"});
        check(o && !o->lesLabel && o->flameTfles == 0 &&
                  flameRunLabel(*o).starts_with("DNS ["),
              "no flag: DNS, no TFLES");
    }
    rejects({"flame", "--nz", "16", "--ny", "16", "--les", "1.35"}, 2, "--les",
            "--les without an explicit --sgs rejected (names sigma 1.35)");
    rejects({"flame", "--les", "1.35", "--sgs", "sigma"}, 2, "--les",
            "--les at nz 1 rejected");
    rejects({"flame", "--nz", "16", "--ny", "16", "--les", "-1", "--sgs",
             "sigma"},
            2, "--les", "--les -1 rejected");
    rejects({"flame", "--sgs", "sigma"}, 2, "--sgs", "--sgs without --les rejected");
    rejects({"flame", "--mode", "bunsen", "--les", "1.35", "--sgs", "sigma"}, 2,
            "--les", "--les with 2D --mode bunsen (nz 1) rejected");
    rejects({"flame", "--tfles", "0.5"}, 2, "--tfles", "--tfles 0.5 rejected");
    rejects({"flame", "--tfles", "21"}, 2, "--tfles", "--tfles 21 rejected");
    rejects({"flame", "--tfles-sensor", "uniform"}, 2, "--tfles-sensor",
            "--tfles-sensor without --tfles rejected");
    rejects({"flame", "--efficiency", "none"}, 2, "--efficiency",
            "--efficiency without --tfles rejected");
    rejects({"flame", "--tfles", "4", "--cells-per-delta", "2", "--efficiency",
             "charlette"},
            2, "--efficiency", "--efficiency charlette without --les rejected");
    rejects({"flame", "--tfles", "2", "--cells-per-delta", "1.5"}, 2,
            "--cells-per-delta", "F --cells-per-delta < 4 rejected");
    rejects({"flame", "--tfles-sensor", "maybe"}, 1, "--tfles-sensor",
            "--tfles-sensor maybe: not a value");
    for (const char* m : {"diffusion", "candle", "chamber"})
        for (const char* f : {"--les", "--tfles"})
            rejects({"flame", "--mode", m, f, "1"}, 2, f,
                    std::string(f) + " rejected by name with --mode " + m);
    rejects({"flame", "--mode", "diffusion", "--sgs", "sigma"}, 2, "--sgs",
            "--sgs rejected by name with --mode diffusion");
    rejects({"flame", "--mode", "candle", "--efficiency", "none"}, 2,
            "--efficiency", "--efficiency rejected by name with --mode candle");
    rejects({"conv", "--tfles", "2"}, 1, "--tfles", "--tfles is not a conv flag");
    return dnstest::finish();
}
