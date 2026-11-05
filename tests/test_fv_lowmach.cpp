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
    // Model C2 / C2b on faces (M3 step 1): the structured validator's case.
    {
        auto r = runCli(plus({"--equivalence", "1.3", "--expansion", "6.88"}));
        check(r.has_value(), "--equivalence passes on the low-Mach FV route");
        if (r) check(r->flamePremix2 && r->premixLaw.n > 0, "--equivalence: C2b law (gri30 default) resolved");
        auto o = runCli(plus({"--equivalence", "1.3", "--expansion", "6.88", "--sl-law", "onestep", "--mix-phi", "1.2"}));
        check(o.has_value() && o->premixLaw.n == 0, "--sl-law onestep, --mix-phi pass");
        rejects(plus({"--equivalence", "1.3", "--les", "0.2"}), "--les");
    }
    rejects(plus({"--sl-law", "gri30"}), "--sl-law");
    // The 2D slot burner (M3 step 2), open laterals, gravity, the rim (steps 3-5); 3D: not built here.
    {
        auto b = runCli(plus({"--mode", "bunsen", "--cells-per-delta", "4"}));
        check(b.has_value() && b->flameBunsen && b->lmDtCell, "--mode bunsen passes, --dt-rule cell its default");
        if (b) check(b->describe().find("one row per face") != std::string::npos, "the dry run names the rows");
        auto b2 = runCli(plus({"--mode", "bunsen", "--equivalence", "1.3", "--expansion", "6.88", "--pilot-width", "1.5",
                               "--dt-rule", "bounds"}));
        check(b2.has_value() && !b2->lmDtCell, "the partially premixed burner passes, --dt-rule bounds");
        rejects(plus({"--mode", "bunsen", "--nz", "8"}), "--nz");
        auto o = runCli(plus({"--mode", "bunsen", "--laterals", "open", "--gravity", "0.0241"}));
        check(o.has_value() && o->lmOpenY && o->lmGravity == 0.0241, "--laterals open, --gravity pass on the burner");
        if (o) check(o->describe().find("OPEN laterals") != std::string::npos, "the dry run names the open faces");
        auto rm = runCli(plus({"--mode", "bunsen", "--equivalence", "1.3", "--expansion", "6.88", "--slot-width", "14",
                               "--jet-speed", "6", "--rim", "1.4", "--laterals", "open", "--coflow-width", "6"}));
        check(rm.has_value() && rm->lmRim, "--rim passes on the burner (the tube: an isothermal rim Wall patch)");
        if (rm) check(rm->describe().find("RIM") != std::string::npos, "the dry run names the rim");
        rejects(plus({"--mode", "bunsen", "--nozzle", "round"}), "--nozzle");
        rejects(plus({"--mode", "bunsen", "--profile-dt", "1"}), "--profile-dt");
    }
    rejects(plus({"--mode", "diffusion"}), "--mode");
    rejects(plus({"--gravity", "1"}), "--gravity");
    // M4: LES and TFLES are built on the route; their refused pairs (dns/lmmodules) by name.
    rejects(plus({"--les", "0.2"}), "--les"); // 2D: needs --nz >= 8
    {
        auto l = runCli(plus({"--nz", "8", "--les", "0.17", "--sgs", "smag"}));
        check(l.has_value() && l->lesLabel && l->cs == 0.17, "M4: --les passes on the low-Mach FV route (3D)");
        if (l) check(l->describe().find("LES [Smagorinsky Cs=0.17") != std::string::npos, "M4: the label names Cs and model");
        auto t = runCli(plus({"--tfles", "2"}));
        check(t.has_value() && t->flameTfles == 2, "M4: --tfles passes on the low-Mach FV route");
        auto e = runCli(plus({"--equivalence", "1.3", "--nz", "8", "--les", "1.35", "--sgs", "sigma"}));
        check(e.has_value(), "M4: two-reactant LES passes (one Sc_t: Z stays linear)");
    }
    rejects(plus({"--equivalence", "1.3", "--tfles", "2"}), "--tfles");
    rejects(plus({"--mode", "bunsen", "--laterals", "open", "--tfles", "2"}), "--tfles");
    rejects(plus({"--mode", "bunsen", "--rim", "1", "--tfles", "2"}), "--tfles");
    rejects(plus({"--turb", "1"}), "--turb");
    rejects(plus({"--dt-rule", "cell"}), "--dt-rule");
    // M5: checkpoints are built on the low-Mach FV route (gpu_restart_fvlm).
    {
        auto c = runCli(plus({"--checkpoint-dt", "1"}));
        check(c.has_value() && c->checkpointDt == 1, "M5: --checkpoint-dt passes on the low-Mach FV route");
    }
    rejects(plus({"--wall-limit", "60"}), "--wall-limit");
    // A polyMesh burner: open faces and the tube are the mesh's patches, not flags.
    rejects({"flame", "--physics", "lowmach", "--mode", "bunsen", "--polymesh", "/nonexistent", "--bc", "/nonexistent",
             "--laterals", "open"},
            "--laterals");
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
    // `open` (M3 step 3): p = 0 as an outflow, the ambient one per run.
    {
        auto o = fvLowMachBcs(spec(in + "[patch.b]\ntype = \"open\"\nt = 1.5\ny = [0.0, 0.233]\n[patch.c]\ntype = "
                                        "\"open\"\nt = 1.5\ny = [0.0, 0.233]\n"),
                              1.0, 2.0);
        check(o.has_value(), "two open patches at one ambient pass (no outflow needed)");
        if (o) {
            check(o->open && o->openPatches.size() == 2, "the open patches are named");
            check(o->ambT == 3.0 && o->ambY[1] == 0.233, "the ambient: t in Tu, y");
            check(o->flow.patches[1].kind == FvIncompBc::Outflow && o->flow.patches[1].p == 0.0, "open = a p = 0 face");
        }
        fails(in + "[patch.b]\ntype = \"open\"\nt = 1.5\n[patch.c]\ntype = \"open\"\n", "one ambient");
        auto key = parseFvBc("[patch.b]\ntype = \"open\"\n");
        check(key && fvBcFlameKey(*key).find("open") != std::string::npos, "open is a dns flame key (flow, conv reject it)");
        check(!parseFvBc("[patch.b]\ntype = \"open\"\np = 0.5\n"), "open reads t and y only");
    }
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

// The box with open laterals and a solid strip (M3 steps 3, 5): the strip's cells go,
// its fluid/solid faces are the `rim` patch with S out of the fluid, srcCell stays
// Grid::idx, the open faces' table marks the open patches 2 and the outflow 1.
void rimMeshGate()
{
    const Grid g = Grid::make(8, 6, 1, 8.0, 6.0, 1.0);
    std::vector<std::uint8_t> solid(g.size(), 0);
    for (int i = 0; i < 3; ++i) solid[g.idx(i, 2, 0)] = 1;
    auto fm = fvLowMachBox(g, true, solid);
    check(fm.has_value(), "box minus a strip");
    if (!fm) return;
    check(fm->nC == 45, "45 fluid cells");
    double vol = 0;
    for (double v : fm->V) vol += v;
    check(std::abs(vol - 45.0) < 1e-12, "the volume of the fluid cells");
    bool src = true;
    for (auto s : fm->srcCell) src = src && s >= 0 && std::size_t(s) < g.size() && !solid[std::size_t(s)];
    check(src, "srcCell = Grid::idx of a fluid cell");
    const auto rim = std::ranges::find(fm->patches, std::string("rim"), &MeshPatch::name);
    check(rim != fm->patches.end() && rim->count == 7, "7 rim faces: the strip's top, bottom and end");
    bool out = rim != fm->patches.end();
    for (int f = out ? rim->first : 0; out && f < rim->first + rim->count; ++f) {
        const std::size_t b = std::size_t(fm->nI + f), c = std::size_t(fm->own[b]);
        double dot = 0;
        for (int k = 0; k < 3; ++k) dot += fm->S[std::size_t(k)][b] * (fm->Cf[std::size_t(k)][b] - fm->C[std::size_t(k)][c]);
        out = dot > 0;
    }
    check(out, "every rim face's S points out of its fluid cell");
    FvBcSpec spec = fvLowMachBoxBc(true, 1.0, {0.0, 0.233});
    spec.patches.push_back({.name = "rim", .kind = FvPatchKind::Wall, .isothermal = true, .tWall = 1.5});
    auto b = fvLowMachBcs(spec, 1.0, 1.0);
    check(b.has_value() && b->open, "the box's open bcs");
    if (!b) return;
    check(bindFvIncompBcs(b->flow, *fm).has_value(), "bound");
    auto s = buildFvIncompSetup(*fm, b->flow);
    check(s.has_value(), "setup");
    if (!s) return;
    const FvLowMachTables t = buildFvLowMachTables(*fm, *b, *s, 1.0);
    int n1 = 0, n2 = 0;
    for (auto v : t.bOpen) (v == 1 ? n1 : n2) += v != 0;
    check(n1 == g.ny && n2 == 2 * g.nx, "bOpen: 1 on the outflow, 2 on the open patches");
    check(t.ambY[1] == 0.233 && t.ambT == 1.0, "the ambient in the tables");
}

} // namespace

int main()
{
    routeGate();
    bcGate();
    tableGate();
    rimMeshGate();
    limiterGate();
    probeGate();
    return dnstest::finish();
}
