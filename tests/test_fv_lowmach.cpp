// The host side of the low-Mach FV route (docs/fvincomp.md "Low-Mach"): the CLI route
// and what stays rejected by name, the --bc table, the first-side table, the shared
// limiter vs the structured stencil, the probe.
#include "dns/config.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvcomp_config.hpp"
#include "dns/fvflame.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvincomp.hpp"
#include "dns/fvlowmach.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <format>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace dns;
using dnstest::check;
using dnstest::runCli;

void rejects(std::vector<std::string> a, const std::string& flag)
{
    auto c = runCli(std::move(a));
    const bool ok = !c && (c.error().field == flag || c.error().message.find(flag) != std::string::npos);
    if (!ok && !c) std::printf("    got %d %s: %s\n", c.error().code, c.error().field.c_str(), c.error().message.c_str());
    check(ok, std::format("rejected by name: {}", flag));
}

void routeGate()
{
    const std::vector<std::string> fl = {"flame", "--physics", "lowmach", "--solver", "fv"};
    auto plus = [&](std::vector<std::string> b) {
        std::vector<std::string> a = fl;
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    auto c = runCli(fl);
    check(c.has_value(), "dns flame --physics lowmach --solver fv validates");
    if (c) {
        check(c->fvMode == FvCompMode::Structured && c->physics == Physics::LowMach, "the route's mode");
        const std::string d = c->describe();
        check(d.find("mass-flux projection per stage") != std::string::npos, "the dry run names the FV solver");
        check(d.find("LOW-MACH") != std::string::npos, "the dry run names the model");
        check(d.find("fp32, rho and rho Y compensated") != std::string::npos, "the label names the state");
    }
    check(runCli(plus({"--tol", "1e-6", "--chem-dt", "0.5", "--fv-precision", "fp64"})).has_value(), "its own flags pass");
    rejects(plus({"--compensated"}), "--compensated");
    rejects(plus({"--mach", "0.01"}), "--mach");
    rejects(plus({"--equivalence", "1.3"}), "--equivalence");
    rejects(plus({"--mode", "bunsen"}), "--mode");
    rejects(plus({"--gravity", "1"}), "--gravity");
    rejects(plus({"--les", "0.2"}), "--les");
    rejects(plus({"--tfles", "2"}), "--tfles");
    rejects(plus({"--turb", "1"}), "--turb");
    rejects(plus({"--dt-rule", "cell"}), "--dt-rule");
    rejects(plus({"--checkpoint-dt", "1"}), "--checkpoint-dt");
    rejects({"flame", "--physics", "lowmach", "--tol", "1e-6"}, "--tol");
    rejects({"flame", "--physics", "lowmach", "--fv-precision", "fp64"}, "--fv-precision");
}

void bcGate()
{
    auto spec = [](const std::string& t) {
        auto s = parseFvBc(t);
        check(s.has_value(), "the --bc text parses");
        return *s;
    };
    auto fails = [&](const std::string& t, const std::string& word) {
        auto b = fvLowMachBcs(spec(t), 1.0, 1.0);
        check(!b && b.error().find(word) != std::string::npos, std::format("--bc rejected by name: {}", word));
    };
    const std::string in = "[patch.a]\ntype = \"inflow\"\ny = [1.0]\n";
    fails("[patch.a]\ntype = \"far\"\ny = [1.0]\n[patch.b]\ntype = \"outflow\"\n", "far");
    fails(in + "[patch.b]\ntype = \"outflow\"\nburnt = true\n", "burnt");
    fails(in + "[patch.b]\ntype = \"outflow\"\np = 0.9\n", "p:");
    fails(in + "[patch.b]\ntype = \"outflow\"\ntback = 2.0\n", "tback");
    fails(in + "[patch.b]\ntype = \"outflow\"\ny = [0.5]\n", "y on an outflow");
    fails("[patch.a]\ntype = \"inflow\"\n[patch.b]\ntype = \"outflow\"\n", "y = [");
    fails(in + "[patch.b]\ntype = \"wall\"\n", "no outflow");
    auto ok = fvLowMachBcs(spec(in + "[patch.b]\ntype = \"outflow\"\n[patch.c]\ntype = \"wall\"\nthermal = \"isothermal\"\ntwall = 2.0\n"),
                           0.5, 1.0);
    check(ok.has_value(), "an inflow, outflow, isothermal wall table passes");
    if (ok) {
        check(ok->flow.patches[0].hasT && ok->flow.patches[0].T == 1.0, "inflow T fixed");
        check(ok->flow.patches[0].u[0] == 0.5, "inflow u: the freestream along direction");
        check(ok->flow.patches[2].hasT && ok->flow.patches[2].T == 2.0, "wall T fixed");
    }
}

// A column of 6 cells: the first and last have no second neighbour; the y cyclic
// faces of the same cells keep theirs.
void tableGate()
{
    const Grid g = Grid::make(6, 4, 1, 6.0, 4.0, 1.0);
    auto fm = fvLowMachBox(g);
    check(fm.has_value(), "box mesh");
    if (!fm) return;
    FvBcSpec sp = fvLowMachBoxBc();
    auto b = fvLowMachBcs(sp, 1.0, 1.0);
    check(b.has_value(), "box bcs");
    if (!b) return;
    check(bindFvIncompBcs(b->flow, *fm).has_value(), "bound");
    auto s = buildFvIncompSetup(*fm, b->flow);
    check(s.has_value(), "setup");
    if (!s) return;
    const FvLowMachTables t = buildFvLowMachTables(*fm, *b, *s, 1.0);
    const std::size_t nI = std::size_t(s->nI);
    int xFirst = 0, yFirst = 0;
    for (std::size_t f = 0; f < nI; ++f) {
        const bool xFace = std::abs(s->S[f]) > 0.5;
        (xFace ? xFirst : yFirst) += t.first[f] ? 1 : 0;
    }
    check(xFirst == 2 * g.ny, "x faces beside either end carry the first-side bit");
    check(yFirst == 0, "tangential faces keep the limiter");
    check(!s->pureNeumann, "the outflow fixes p");
    int inflow = 0;
    for (std::size_t q = 0; q < std::size_t(s->nB); ++q) inflow += t.bYfix[q];
    check(inflow == g.ny, "one fixed-Y face per row");
}

// The structured stencil on a uniform mesh: d1 = Y_up - Y_upup, theta MC / d2.
void limiterGate()
{
    const double h = 0.5, d[3] = {h, 0, 0};
    const double Y[4] = {0.1, 0.4, 0.45, 0.9}, T[4] = {1.0, 1.2, 2.0, 2.1};
    auto grad = [&](const double* f, int i) { return (f[i + 1] - f[i - 1]) / (2 * h); };
    const double gYP[3] = {grad(Y, 1), 0, 0}, gYN[3] = {grad(Y, 2), 0, 0};
    const double gTP[3] = {grad(T, 1), 0, 0}, gTN[3] = {grad(T, 2), 0, 0};
    double th = 2.0;
    th = fvThetaVote(th, 1.0, Y[1], Y[2], gYP, gYN, d);
    th = fvThetaVote(th, 1.0, T[1], T[2], gTP, gTN, d);
    const double thY = fvMcSlope(Y[1] - Y[0], Y[2] - Y[1]) / (Y[2] - Y[1]);
    const double thT = fvMcSlope(T[1] - T[0], T[2] - T[1]) / (T[2] - T[1]);
    check(std::abs(th - std::min(thY, thT)) < 1e-14, "theta = min over voters of the structured MC ratio");
    const double back = fvThetaVote(2.0, -1.0, Y[1], Y[2], gYP, gYN, d);
    const double thB = fvMcSlope(Y[2] - Y[3], Y[1] - Y[2]) / (Y[1] - Y[2]);
    check(std::abs(back - thB) < 1e-14, "the other upwind side");
    const double zero[3] = {0, 0, 0};
    check(fvThetaVote(2.0, 1.0, 0.25, 0.25, zero, zero, d) == 2.0, "a uniform field does not vote");
    check(fvFaceYShared(1.0, 0.5, 0.5, 2.0) == 0.5 && fvFaceYShared(-1.0, 0.5, 0.5, 1.0) == 0.5, "uniform: the bit");
    check(fvFaceYShared(1.0, 1.0, 3.0, 1.0) == 2.0, "theta 1: central");
}

// Cartesian slabs: the means of the cells, the dynamic p.
void probeGate()
{
    const Grid g = Grid::make(8, 2, 1, 8.0, 2.0, 1.0);
    auto fm = fvLowMachBox(g);
    check(fm.has_value(), "box");
    if (!fm) return;
    const FvFlameSpec fl{.dir = {1, 0, 0}, .Tu = 1, .sigma = 5, .dth = 1, .gamma = 1.4, .Rgas = 1, .q = 14.0, .fuel = 0};
    auto pr = FvFlameProbe::make(*fm, fl);
    check(pr.has_value(), "probe");
    if (!pr) return;
    const std::size_t n = std::size_t(fm->nC);
    std::vector<double> rho(n), ry(n), ru(3 * n, 0.0), T(n), p(n), om(n, 0.5);
    for (std::size_t c = 0; c < n; ++c) {
        T[c] = 1.0 + std::floor(fm->C[0][c]) * 4.0 / 7.0;
        rho[c] = 1.0 / T[c];
        ry[c] = rho[c] * (5.0 - T[c]) / 4.0;
        ru[c] = 1.0;
        p[c] = 5.0 - T[c];
    }
    const FvFlameMeasure m = pr->measureLowMach({.rho = &rho, .ry = &ry, .ru = &ru, .T = &T, .p = &p, .omega = &om});
    check(m.slabs.size() == 8, "one slab per column");
    check(std::abs(m.slabs[0].u - 1.0) < 1e-14 && std::abs(m.slabs[7].u - 5.0) < 1e-14, "u = rho u / rho");
    check(std::abs(m.slabs[0].p - 4.0) < 1e-14, "the slab's dynamic p");
    check(std::abs(m.burn - 0.5 * 16.0) < 1e-13, "burn");
    check(std::abs(m.mass - std::accumulate(rho.begin(), rho.end(), 0.0)) < 1e-13, "mass");
}

} // namespace

int main()
{
    routeGate();
    bcGate();
    tableGate();
    limiterGate();
    probeGate();
    return dnstest::finish();
}
