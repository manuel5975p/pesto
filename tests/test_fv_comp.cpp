// Compressible FV family, host side (docs/fvcomp.md): the --bc file, its binding
// against a patch list, the case layer's FvBc table, and the CLI -> validate
// rules (routing, flags rejected by name, exit codes).
#include "check.hpp"
#include "dns/cli.hpp"
#include "dns/curvmesh.hpp"
#include "dns/config.hpp"
#include "dns/fvcomp_case.hpp"
#include "dns/fvcomp_config.hpp"
#include "dns/fvflame.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvincomp.hpp"
#include "dns/fvraster.hpp"
#include "dns/fvref.hpp"
#include "dns/curvmesh.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <random>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

constexpr const char* kGood = R"(
[reference]
length = 2.0
direction = [1, 0, 0]
lift = [0, 1, 0]
[patch.body]
type = "wall"
thermal = "isothermal"
twall = 1.2
[patch.in]
type = "inflow"
t = 1.1
[patch.out]
type = "outflow"
p = 0.95
tback = 1.05
[patch.far]
type = "far"
[patch.side]
type = "slip"
)";

bool errIs(const std::expected<FvBcSpec, std::string>& e, std::string_view what)
{
    return !e && e.error().find(what) != std::string::npos;
}

void parseGate()
{
    auto s = parseFvBc(kGood);
    check(s.has_value(), "the full table parses");
    if (!s) return;
    check(s->patches.size() == 5 && s->length == 2.0, "five patches, length 2");
    const auto& w = s->patches[0]; // table order: ascending names
    check(w.name == "body" && w.isothermal && w.tWall == 1.2 && w.tallied, "wall: isothermal, tallied by default");
    check(errIs(parseFvBc("[patch.a]\ntype = \"wal\"\n"), "unknown patch type"), "unknown kind by name");
    check(errIs(parseFvBc("[patch.a]\ntype = \"far\"\nu = [1,0,0]\n"), "key 'u'"), "foreign key by name");
    check(errIs(parseFvBc("[nope]\n[patch.a]\ntype = \"far\"\n"), "unknown top-level key"), "unknown table");
    check(errIs(parseFvBc("[patch.a]\ntype = \"wall\"\nthermal = \"isothermal\"\n"), "twall"),
          "isothermal wall without twall");
    check(errIs(parseFvBc("[patch.a]\ntype = \"wall\"\ntwall = 2\n"), "twall"), "twall on an adiabatic wall");
    check(errIs(parseFvBc("[patch.a]\ntype = \"outflow\"\np = -1\n"), "must be > 0"), "negative pressure ratio");
    check(errIs(parseFvBc("[reference]\nlift = [1,0,0]\n[patch.a]\ntype = \"far\"\n"), "normal"),
          "lift not normal to direction");
    check(errIs(parseFvBc(""), "no [patch.NAME]"), "empty file");
    check(errIs(parseFvBc("[patch.a\n"), "line"), "TOML syntax error names the line");
}

void bind()
{
    auto s = parseFvBc(kGood);
    if (!s) return;
    std::vector<FvMeshPatch> m = {{"body", "wall"}, {"in", "patch"}, {"out", "patch"}, {"far", "patch"},
                                  {"side", "patch"}, {"back", "empty"}, {"front", "empty"}, {"cl", "cyclic"}};
    check(checkFvBc(*s, m).has_value(), "bound; empty and cyclic patches need no table");
    check(fvMeshIs2D(m), "empty patch = 2D");
    auto m2 = m;
    m2.push_back({"lost", "patch"});
    auto e = checkFvBc(*s, m2);
    check(!e && e.error().find("`lost`") != std::string::npos, "an unbound patch by name");
    auto m3 = m;
    m3.erase(m3.begin() + 3);
    e = checkFvBc(*s, m3);
    check(!e && e.error().find("`far`") != std::string::npos, "a binding with no patch by name");
    auto m4 = m;
    m4[0].type = "cyclic";
    check(!checkFvBc(*s, m4).has_value(), "binding a cyclic patch is rejected");
    auto m5 = m;
    m5[1].type = "wedge";
    check(!checkFvBc(*s, m5).has_value(), "wedge is rejected");
    auto m3d = m;
    m3d.erase(m3d.begin() + 5, m3d.begin() + 7);
    e = checkFvBc(*s, m3d);
    check(!e && e.error().find("reference.area") != std::string::npos, "3D tally needs a reference area");
}

void table()
{
    auto s = parseFvBc(kGood);
    if (!s) return;
    const FvFreestream f = fvFreestreamOf(0.2, 1.4, 1.0, {1, 0, 0});
    check(std::abs(f.p - 17.857142857142858) < 1e-12 && f.T == f.p, "p = 1/(gamma M^2), T = p at R = 1");
    auto of = [&](const char* n) { return *std::ranges::find(s->patches, n, &FvPatchSpec::name); };
    const FvBc w = fvBcOf(of("body"), f, 1.0);
    check(w.kind == FvBcKind::Wall && w.isothermal && std::abs(w.tWall - 1.2 * f.T) < 1e-12, "wall ratio -> T");
    const FvBc in = fvBcOf(of("in"), f, 0.5);
    check(in.uIn[0] == 0.5 && std::abs(in.tIn - 1.1 * f.T) < 1e-12, "inflow follows the ramp");
    const FvBc o = fvBcOf(of("out"), f, 1.0);
    check(std::abs(o.pOut - 0.95 * f.p) < 1e-12, "outflow pressure ratio");
    const FvBc ff = fvBcOf(of("far"), f, 1.0);
    check(ff.qFar[1] == 1.0 && ff.qFar[4] == f.p / (f.gamma - 1.0) + 0.5, "far = the conserved freestream");
}

std::expected<Case, ConfigError> runCli(std::vector<std::string> a, bool withBc = false)
{
    std::string prog = "dns";
    std::vector<char*> v = {prog.data()};
    for (auto& s : a) v.push_back(s.data());
    auto r = cli::parse(int(v.size()), v.data());
    if (!r) return std::unexpected(ConfigError{1, "cli", r.error().message});
    if (withBc) r->fvBc = parseFvBc(kGood).value();
    return validate(*r);
}

void cliGate()
{
    const std::vector<std::string> sf = {"flow", "--physics", "compressible", "--solver", "fv"};
    auto with = [&](std::vector<std::string> x) {
        auto v = sf;
        v.insert(v.end(), x.begin(), x.end());
        return runCli(v);
    };
    auto ok = with({"--ogrid", "--body", "cylinder", "--n", "64", "--ny", "32", "--re", "40", "--mach", "0.2"});
    check(ok && ok->fvMode == FvCompMode::Structured && ok->ogrid, "--solver fv --ogrid routes structured");
    if (ok) check(fvCompLabel(*ok, true).starts_with("[FV path, body-fitted O-grid"), "the label says FV path");
    auto rej = [&](std::vector<std::string> x, std::string_view why) {
        auto r = with(std::move(x));
        check(!r && r.error().code == 2 && r.error().message.find(why) != std::string::npos,
              std::format("rejected, exit 2: {}", why));
    };
    rej({"--body", "cylinder", "--n", "64"}, "no immersed wall");
    rej({"--ogrid", "--body", "sphere", "--n", "64", "--ny", "32"}, "P8");
    rej({"--ogrid", "--body", "cylinder", "--n", "64", "--ny", "32", "--pot"}, "--pot");
    rej({"--ogrid", "--body", "cylinder", "--n", "64", "--ny", "32", "--emit", "vtk"}, "not built");
    // RANS: labelled, off the mesh-free flags' route, rejected by name elsewhere.
    auto ra = with({"--ogrid", "--body", "cylinder", "--n", "64", "--ny", "32", "--re", "1e4", "--mach", "0.2", "--rans",
                    "sst-2003m"});
    check(ra && ra->rans == RansModel::Sst2003m && ra->ransPhiInf[1] > 0.0, "--solver fv --ogrid --rans routes");
    if (ra)
        check(fvCompLabel(*ra, true).find("RANS SST-2003m -- URANS (time-accurate)") != std::string::npos,
              "the FV label says RANS and the mode");
    rej({"--ogrid", "--body", "cylinder", "--n", "64", "--ny", "32", "--rans", "sa-neg", "--les", "0.1"}, "--les");
    rej({"--body", "none", "--n", "64", "--rans", "sa-neg"}, "needs a wall");
    auto rb = with({"--body", "none", "--n", "32", "--noslip", "--re", "100", "--rans", "sa-neg"});
    check(rb && rb->rans == RansModel::SaNeg, "--solver fv --body none --noslip --rans routes (Wall patches)");
    auto rp = runCli({"flow", "--physics", "compressible", "--polymesh", "/tmp/x", "--bc", "b", "--re", "1e4", "--mach",
                      "0.2", "--tt", "1", "--rans", "sa-neg", "--rans-nutinf", "4"}, true);
    check(rp && rp->rans == RansModel::SaNeg && std::abs(rp->ransPhiInf[0] - 4.0 * 2.0 / 1e4) < 1e-15,
          "polyMesh route: nu~inf = R U L / Re");
    if (rp) check(rp->meta.rans == "SA-neg", "polyMesh route: meta says the closure");
    auto rq = runCli({"flow", "--physics", "compressible", "--polymesh", "/tmp/x", "--bc", "b", "--re", "1e4", "--mach",
                      "0.2", "--tt", "1", "--rans", "sa-neg", "--rans-kinf", "1e-7"}, true);
    check(!rq && rq.error().field == "--rans-kinf", "polyMesh route: an SST flag on SA");
    auto rs = with({"--ogrid", "--body", "cylinder", "--n", "64", "--ny", "32", "--re", "1e4", "--mach", "0.2", "--rans",
                    "sa-neg", "--steady", "1e-3"});
    check(rs && rs->compSteady, "--solver fv --steady routes");
    if (rs)
        check(fvCompLabel(*rs, true).find("steady (local time stepping: pseudo-time, not time-accurate)") !=
                  std::string::npos,
              "the FV label says pseudo-time");
    auto rt = runCli({"flow", "--physics", "compressible", "--polymesh", "/tmp/x", "--bc", "b", "--re", "1e4", "--mach",
                      "0.2", "--tt", "1", "--rans", "sa-neg", "--steady", "1e-3"}, true);
    check(!rt && rt.error().field == "--tt", "polyMesh route: --tt with --steady");
    auto rl = runCli({"flow", "--physics", "compressible", "--polymesh", "/tmp/x", "--bc", "b", "--re", "1e4", "--mach",
                      "0.2", "--tt", "1", "--rans", "sa-neg", "--les", "0.1"}, true);
    check(!rl && rl.error().field == "--rans", "polyMesh route: LES and RANS");
    auto inc = runCli({"flow", "--solver", "fv", "--ogrid", "--body", "cylinder"});
    check(inc && inc->fvIncomp && inc->fvMode == FvCompMode::Off, "--solver fv dispatches on --physics: incompressible");
    auto sim = runCli({"flow", "--polymesh", "/tmp/x"});
    check(sim && sim->fvMode == FvCompMode::Off && sim->fvCase == "/tmp/x", "--polymesh alone stays simpleFoam");
    auto noBc = runCli({"flow", "--physics", "compressible", "--polymesh", "/tmp/x", "--re", "40", "--mach", "0.2",
                        "--tt", "1"});
    check(!noBc && noBc.error().message.find("--bc") != std::string::npos, "compressible polyMesh needs --bc");
    auto pm = runCli({"flow", "--physics", "compressible", "--polymesh", "/tmp/x", "--bc", "b", "--re", "40",
                      "--mach", "0.2", "--tt", "3"}, true);
    check(pm && pm->fvMode == FvCompMode::PolyMesh && pm->tEnd == 6.0, "polyMesh route: tt in L/U, L = 2");
    auto fl = runCli({"flow", "--physics", "compressible", "--polymesh", "/tmp/x", "--bc", "b", "--re", "40",
                      "--mach", "0.2", "--tt", "3", "--ogrid"}, true);
    check(!fl && fl.error().message.find("--ogrid") != std::string::npos, "a structured flag on the polyMesh route");
    auto stray = runCli({"flow", "--body", "cylinder", "--bc", "b"});
    check(!stray && stray.error().code == 2, "--bc on the structured path");
    auto prec = runCli({"flow", "--physics", "compressible", "--body", "cylinder", "--fv-precision", "fp64"});
    check(!prec && prec.error().code == 2, "--fv-precision without an FV route");

    // The routes of --solver / --polymesh / --bc: one rule, each path's own list.
    auto cli1 = [](std::vector<std::string> a) { return runCli(std::move(a)); };
    auto incPoly = cli1({"flow", "--polymesh", "/tmp/x", "--bc", "b", "--re", "40", "--ft", "1"});
    check(incPoly && incPoly->fvIncomp && incPoly->fvMode == FvCompMode::Off, "--bc alone: --physics picks incompressible");
    auto fvPoly = runCli({"flow", "--solver", "fv", "--physics", "compressible", "--polymesh", "/tmp/x", "--bc", "b",
                          "--re", "40", "--mach", "0.2", "--tt", "3"}, true);
    check(fvPoly && fvPoly->fvMode == FvCompMode::PolyMesh, "--solver fv is implied on the compressible polyMesh");
    const auto reject = [&](std::vector<std::string> a, std::string_view why) {
        auto r = cli1(std::move(a));
        check(!r && r.error().code == 2 && (r.error().field + " " + r.error().message).find(why) != std::string::npos,
              std::format("route rejected by name: {}", why));
    };
    reject({"flow", "--polymesh", "/tmp/x", "--bc", "b", "--re", "40", "--ft", "1", "--compensated"}, "--compensated");
    reject({"flow", "--polymesh", "/tmp/x", "--bc", "b", "--re", "40", "--ft", "1", "--tt", "1"}, "--tt");
    reject({"flow", "--polymesh", "/tmp/x", "--bc", "b", "--re", "40", "--ft", "1", "--frame-n", "64"}, "--frame-n");
    reject({"flow", "--polymesh", "/tmp/x", "--bc", "b", "--re", "40", "--ft", "1", "--emit", "slices"}, "--emit");
    reject({"flow", "--polymesh", "/tmp/x", "--tt", "1"}, "--tt");
    reject({"flow", "--polymesh", "/tmp/x", "--compensated"}, "--compensated");
    reject({"flow", "--polymesh", "/tmp/x", "--solver", "structured"}, "--solver");
    reject({"flow", "--polymesh", "/tmp/x", "--solver", "simple"}, "--solver");
    reject({"flow", "--solver", "simple", "--body", "none", "--compensated"}, "--compensated");
    reject({"flow", "--solver", "simple", "--body", "none", "--bc", "b"}, "--bc");
    reject({"flow", "--solver", "simple", "--physics", "compressible", "--body", "none"}, "--physics");
    reject({"flow", "--physics", "compressible", "--solver", "fv", "--body", "none", "--export", "c"}, "--export");
    reject({"flow", "--physics", "compressible", "--solver", "fv", "--body", "none", "--bc", "b"}, "--bc");
    reject({"flow", "--solver", "fv", "--ogrid", "--body", "cylinder", "--compensated"}, "--compensated");
    reject({"flow", "--solver", "fv", "--ogrid", "--body", "cylinder", "--emit", "slices"}, "--emit");
    reject({"flow", "--body", "cylinder", "--re-tau", "180"}, "--re-tau");
    check(!cli1({"tgv", "--solver", "simple"}), "tgv has no --solver simple");

    // --physics picks the --bc parser: the other schema's key is an error by name.
    constexpr const char* kIncompBc = "[patch.w]\nkind = \"wall\"\n";
    constexpr const char* kCompBc = "[patch.w]\ntype = \"wall\"\n";
    auto foreignC = parseFvBc(kIncompBc);
    check(!foreignC && foreignC.error().find("'kind'") != std::string::npos, "compressible --bc rejects kind by name");
    auto foreignI = parseFvIncompBcs(kCompBc);
    check(!foreignI && foreignI.error().find("'type'") != std::string::npos, "incompressible --bc rejects type by name");
    auto refC = parseFvBc("[reference]\nu = 1\n[patch.w]\ntype = \"wall\"\n");
    check(!refC && refC.error().find("'u'") != std::string::npos, "compressible --bc rejects reference.u by name");
    auto refI = parseFvIncompBcs("[reference]\ndirection = [1, 0, 0]\n[patch.w]\nkind = \"wall\"\n");
    check(!refI && refI.error().find("'direction'") != std::string::npos,
          "incompressible --bc rejects reference.direction by name");
    auto iniC = parseFvBc("[initial]\nu = [1, 0, 0]\n[patch.w]\ntype = \"wall\"\n");
    check(!iniC && iniC.error().find("'initial'") != std::string::npos, "compressible --bc rejects [initial] by name");
}

// dns flame --solver fv: the route, the --bc keys, what stays rejected; the face
// value against the structured stencil; the slab probe on a uniform box.
// FvJetProbe on a Cartesian mesh IS measureJet: the same synthetic fields.
void jetProbe()
{
    const Grid g = Grid::make(64, 48, 1, 16.0, 12.0, 1.0);
    const StructuredSpec ss{.lo = {0, 0, 0},
                            .hi = {g.nx - 1, g.ny - 1, 0},
                            .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                            .type = {FoamPatchType::Patch, FoamPatchType::Patch, FoamPatchType::Cyclic, FoamPatchType::Cyclic,
                                     FoamPatchType::Empty, FoamPatchType::Empty}};
    auto fm = FaceMesh::fromCurvMesh(CurvMesh::uniform(g), ss, CellOrder::Morton);
    check(fm.has_value(), "jet probe mesh");
    if (!fm) return;
    DiffusionModel md;
    md.s = 2;
    const double T0 = 285.0, yc = 6.0, h = 0.25;
    BunsenGeom b;
    b.W = 4;
    b.yc = yc;
    const FvJetSpec js{.origin = {0, yc, 0}, .model = md, .geom = b, .T0 = T0, .q = 3.0, .skip = 4};
    auto probe = FvJetProbe::make(*fm, js);
    check(probe.has_value(), "jet probe made");
    if (!probe) return;
    const std::size_t nC = std::size_t(fm->nC);
    FvState s = FvState::zeros(fm->nC, 7);
    std::vector<double> om(nC);
    const std::size_t n = g.size();
    std::vector<float> YF(n), YO(n), W(n);
    for (std::size_t c = 0; c < nC; ++c) {
        const double x = fm->C[0][c], y = fm->C[1][c] - yc, w = 2.0 - 0.15 * x;
        const double Z = 0.5 * (1.0 - std::tanh((std::abs(y) - w) / 1.5)) * std::exp(-0.04 * x);
        const SheetState st = burkeSchumann(md, Z);
        s.var(0)[c] = 1.0;
        s.var(4)[c] = T0 / 0.4;
        s.var(5)[c] = st.YF;
        s.var(6)[c] = st.YO;
        om[c] = std::exp(-(Z - md.zst()) * (Z - md.zst()) / 0.01) * std::exp(-0.05 * x);
        const std::size_t i = std::size_t(fm->srcCell[c]);
        YF[i] = float(st.YF);
        YO[i] = float(st.YO);
        W[i] = float(om[c]);
    }
    const FvJetMeasure a = probe->measure(s, om);
    const JetMeasure e = measureJet(md, YF, YO, W, g.nx, g.ny, h, yc);
    std::printf("  jet probe: tip %.6f / %.6f, half %.6f / %.6f, base %.3f / %.3f\n", a.xTip, e.xTip, a.half, e.half, a.xBase, e.xBase);
    check(std::abs(a.xTip - e.xTip) < 1e-5, "probe tip == measureJet (FP32 fields)");
    check(std::abs(a.half - e.half) < 1e-5, "probe half-width == measureJet");
    check(a.xBase == e.xBase, "probe base == measureJet");
    check(a.hits == e.hits, "probe hits == measureJet");
    check(std::abs(a.pMean - 1.0) < 1e-12, "pMean reads p / p0");
}

// A characteristic Outflow at p_out == p_P is the plain one in rho, u, p; below p_out
// it slows the leaving flow and keeps the owner's entropy and outgoing invariant.
void charOutflow()
{
    FvBc bc;
    bc.kind = FvBcKind::Outflow;
    bc.pOut = 2.0;
    bc.tBack = 1.0;
    bc.characteristic = true;
    const double S[3] = {1, 0, 0}, Cf[3] = {0, 0, 0}, gam = 1.4;
    const FvPrim<double> P = fvPrimOf(1.0, 0.3, 0.1, 0.0, 2.0 / 0.4 + 0.05, gam, 1.0);
    bc.pOut = P.p; // the state's own p: 2.0 rounds through fvPrimOf
    const FvPrim<double> a = fvBoundaryState(bc, P, S, Cf, gam, 1.0);
    check(a.rho == P.rho && a.u[0] == P.u[0] && a.u[1] == P.u[1] && a.p == P.p, "characteristic outflow at p_out: the owner");
    bc.pOut = 3.0;
    const FvPrim<double> b = fvBoundaryState(bc, P, S, Cf, gam, 1.0);
    check(b.u[0] < P.u[0] && b.u[1] == P.u[1], "above the owner's p the leaving flow slows, the tangential u holds");
    check(std::abs(b.p / std::pow(b.rho, gam) - P.p / std::pow(P.rho, gam)) < 1e-14, "the owner's entropy");
    const double ci = P.u[0] + 2.0 / (gam - 1.0) * std::sqrt(gam * P.p / P.rho),
                 cb = b.u[0] + 2.0 / (gam - 1.0) * std::sqrt(gam * b.p / b.rho);
    check(std::abs(ci - cb) < 1e-14, "the outgoing Riemann invariant");
}

// The shared face value is linear across species; noise below 2^-24 does not vote.
void sharedLimiter()
{
    const double d[3] = {0.1, 0.0, 0.0};
    const double mdot = 1.0, g0[3] = {0, 0, 0};
    const double gF[3] = {2.0, 0, 0}, gO[3] = {-0.3, 0, 0};
    double th = 2.0;
    th = fvThetaVote(th, mdot, 0.4, 0.45, gF, g0, d);
    th = fvThetaVote(th, mdot, 0.7, 0.68, gO, g0, d);
    const double a = fvFaceYShared(mdot, 0.4, 0.45, th), b = fvFaceYShared(mdot, 0.7, 0.68, th);
    const double c = fvFaceYShared(mdot, 2 * 0.4 - 3 * 0.7, 2 * 0.45 - 3 * 0.68, th);
    check(th >= 0.0 && th <= 2.0, "theta in [0, 2]");
    check(std::abs(c - (2 * a - 3 * b)) < 1e-15, "the shared face value is linear in Y");
    const double gN[3] = {-1e-3, 0, 0};
    check(fvThetaVote(th, mdot, 1e-9, 2e-9, gN, g0, d) == th, "|d2| < 2^-24 does not vote");
    check(fvThetaVote(th, mdot, 1e-5, 2e-5, gN, g0, d) == 0.0, "|d2| >= 2^-24 votes");
}

void flameGate()
{
    auto cli1 = [](std::vector<std::string> a, bool bc = false) { return runCli(std::move(a), bc); };
    auto box = cli1({"flame", "--solver", "fv", "--cells-per-delta", "10"});
    check(box && box->fvMode == FvCompMode::Structured, "flame --solver fv routes the flags' box");
    if (box) check(fvFlameLabel(*box).find("FV path, uniform box") != std::string::npos, "the label says FV path");
    auto plain = cli1({"flame", "--cells-per-delta", "10"});
    check(plain && plain->fvMode == FvCompMode::Off, "flame alone stays structured");
    auto poly = cli1({"flame", "--polymesh", "/tmp/x", "--bc", "b", "--upstream", "8"}, true);
    check(poly && poly->fvMode == FvCompMode::PolyMesh && poly->fvMeshDir == "/tmp/x", "flame --polymesh + --bc");
    if (poly) check(poly->flameXf0 == 8.0 * poly->flameEig.dthOverDD, "--upstream is the plain distance on a mesh");
    const auto reject = [&](std::vector<std::string> a, std::string_view why) {
        auto r = cli1(std::move(a));
        check(!r && r.error().code == 2 && (r.error().field + " " + r.error().message).find(why) != std::string::npos,
              std::format("flame rejected by name: {}", why));
    };
    for (const char* f : {"--compensated"}) reject({"flame", f}, "--compensated");
    reject({"flame", "--fv-precision", "fp64"}, "--fv-precision");
    reject({"flame", "--bc", "b"}, "--bc");
    reject({"flame", "--frame-n", "64"}, "--frame-n");
    reject({"flame", "--solver", "fv", "--mode", "bunsen"}, "--mode");
    reject({"flame", "--solver", "fv", "--physics", "lowmach", "--compensated"}, "--compensated"); // the route: test_fv_lowmach
    reject({"flame", "--solver", "fv", "--physics", "lowmach", "--mode", "diffusion"}, "--mode");
    reject({"flame", "--solver", "fv", "--mode", "candle"}, "--mode");
    reject({"flame", "--solver", "fv", "--mode", "chamber"}, "--mode");
    reject({"flame", "--solver", "fv", "--mode", "diffusion", "--pulse", "0.1"}, "--pulse");
    reject({"flame", "--solver", "fv", "--mode", "diffusion", "--nscbc-sink", "8"}, "--nscbc-sink");
    reject({"flame", "--solver", "fv", "--slot-width", "8"}, "--slot-width");
    reject({"flame", "--polymesh", "/tmp/x", "--mode", "diffusion", "--cells-per-unit", "4"}, "--bc");
    auto dj = cli1({"flame", "--solver", "fv", "--mode", "diffusion", "--slot-width", "8"});
    check(dj && dj->fvMode == FvCompMode::Structured && dj->flameDiffusion, "--mode diffusion routes the FV box");
    if (dj) check(fvFlameLabel(*dj).find("shared by the species") != std::string::npos, "the jet's label says the limiter");
    auto ju = parseFvBc("[patch.i]\ntype = \"inflow\"\njet = true\n[patch.o]\ntype = \"outflow\"\n");
    check(ju && ju->patches[0].jet && fvBcFlameKey(*ju) == "patch.i.jet", "jet = true parses, a flame key");
    auto jy = parseFvBc("[patch.i]\ntype = \"inflow\"\njet = true\ny = [1.0]\n");
    check(!jy && jy.error().find("jet = true") != std::string::npos, "jet = true excludes y");
    auto jw = parseFvBc("[patch.i]\ntype = \"wall\"\njet = true\n");
    check(!jw && jw.error().find("'jet'") != std::string::npos, "a wall carries no jet");
    jetProbe();
    sharedLimiter();
    charOutflow();
    reject({"flame", "--solver", "fv", "--nscbc-etain", "10"}, "--nscbc-etain");
    reject({"flame", "--solver", "fv", "--nscbc-sink", "8"}, "--nscbc-sink");
    reject({"flame", "--solver", "fv", "--nscbc-tinflow"}, "--nscbc-tinflow");
    reject({"flame", "--solver", "fv", "--equivalence", "1.3"}, "--equivalence");
    reject({"flame", "--solver", "fv", "--nz", "8", "--turb", "2"}, "--turb");
    reject({"flame", "--solver", "fv", "--nz", "8", "--les", "0.2"}, "--les");
    reject({"flame", "--solver", "fv", "--tfles", "2"}, "--tfles");
    reject({"flame", "--solver", "fv", "--checkpoint-dt", "1"}, "--checkpoint-dt");
    reject({"flame", "--solver", "fv", "--bc", "b"}, "--bc");
    reject({"flame", "--polymesh", "/tmp/x", "--cells-per-delta", "10"}, "--bc");

    auto ok = parseFvBc("[reference]\norigin = [1, 2, 3]\n[patch.i]\ntype = \"far\"\ny = [1.0, 0.25]\n"
                        "[patch.o]\ntype = \"outflow\"\nburnt = true\n");
    check(ok && ok->patches[0].y[1] == 0.25 && ok->patches[1].burnt && ok->hasOrigin, "the flame keys parse");
    if (ok) {
        const FvFreestream f{.p = 2.0, .T = 2.0, .pBurnt = 1.9, .tBurnt = 10.0};
        const FvBc in = fvBcOf(ok->patches[0], f, 1.0), o = fvBcOf(ok->patches[1], f, 1.0);
        check(in.y[0] == 1.0 && in.y[1] == 0.25, "fvBcOf carries y");
        check(o.pOut == 1.9 && o.tBack == 10.0, "a burnt Outflow reads the burnt state");
    }
    auto wy = parseFvBc("[patch.w]\ntype = \"wall\"\ny = [1.0]\n");
    check(!wy && wy.error().find("'y'") != std::string::npos, "a wall carries no y");
    auto by = parseFvBc("[patch.i]\ntype = \"far\"\ny = [1.5]\n");
    check(!by && by.error().find("[0, 1]") != std::string::npos, "y out of [0, 1]");
    auto bp = parseFvBc("[patch.o]\ntype = \"outflow\"\nburnt = true\np = 0.9\n");
    check(!bp, "burnt excludes p");
    auto fy = parseFvBc("[patch.i]\ntype = \"far\"\ny = [1.0]\n");
    check(fy && fvBcFlameKey(*fy) == "patch.i.y", "the first flame key is named");

    // Uniform mesh: LSQ's grad Y . d = (Y_R - Y_LL)/2, so fvFaceY IS the structured
    // faceY's value, either wind.
    std::mt19937 gen(5);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const auto mc = [](double d1, double d2) {
        if (!(d1 * d2 > 0.0)) return 0.0;
        const double m = std::min({2.0 * std::abs(d1), 2.0 * std::abs(d2), 0.5 * std::abs(d1 + d2)});
        return d1 > 0.0 ? m : -m;
    };
    double worst = 0.0;
    const double d[3] = {0.1, 0.0, 0.0};
    for (int n = 0; n < 2000; ++n) {
        const double LL = uni(gen), L = uni(gen), R = uni(gen), RR = uni(gen);
        const double gL[3] = {(R - LL) / 0.2, 0, 0}, gR[3] = {(RR - L) / 0.2, 0, 0};
        worst = std::max(worst, std::abs(fvFaceY(1.0, L, R, gL, gR, d) - (L + 0.5 * mc(L - LL, R - L))));
        worst = std::max(worst, std::abs(fvFaceY(-1.0, L, R, gL, gR, d) - (R + 0.5 * mc(R - RR, L - R))));
    }
    check(worst <= 1e-15, std::format("fvFaceY = the structured stencil, {:.1e}", worst));
    FvBc bb;
    bb.kind = FvBcKind::Far;
    bb.y[0] = 0.3;
    check(fvBoundaryY(bb, 0, 0.7, -1.0) == 0.3 && fvBoundaryY(bb, 0, 0.7, 1.0) == 0.7, "boundary Y: the row in, the owner out");
    bb.kind = FvBcKind::Wall;
    check(fvBoundaryY(bb, 0, 0.7, -1.0) == 0.7, "a wall keeps the owner's");

    // The probe on a uniform box: slabs = columns, a tanh T profile read back
    // (x = 3 is a face: the peak slab slope is the difference quotient across it).
    const Grid g = Grid::make(64, 4, 1, 6.4, 0.4, 0.2);
    const StructuredSpec sp{.lo = {0, 0, 0},
                            .hi = {63, 3, 0},
                            .name = {"a", "b", "c", "d", "e", "f"},
                            .type = {FoamPatchType::Patch, FoamPatchType::Patch, FoamPatchType::Cyclic,
                                     FoamPatchType::Cyclic, FoamPatchType::Empty, FoamPatchType::Empty}};
    auto fm = FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::Morton);
    check(fm.has_value(), "probe mesh");
    if (!fm) return;
    const FvFlameSpec fs{.dir = {1, 0, 0}, .Tu = 1.0, .sigma = 5.0, .dth = 0.5, .gamma = 1.4, .Rgas = 1.0, .q = 2.0};
    auto pr = FvFlameProbe::make(*fm, fs);
    check(pr && std::abs(pr->slab() - 0.1) < 1e-12 && std::abs(pr->area() - 0.08) < 1e-12, "slab h, area");
    if (!pr) return;
    FvState st = fvFlameState(*fm, fs, {0, 0, 0}, 0.0, 1, [](double x) { return 3.0 + 2.0 * std::tanh(2.0 * (x - 3.0)); });
    std::vector<double> om(std::size_t(fm->nC), 1.0);
    const FvFlameMeasure m = pr->measure(st, om);
    check(m.slabs.size() == 64, "64 slabs");
    check(std::abs(m.xf - 3.0) < 1e-12, std::format("x_f {:.12f} at the mid temperature", m.xf));
    check(std::abs(m.dth - 0.1 / std::tanh(0.1)) < 1e-9, std::format("delta_th {:.9f} of the tanh", m.dth));
    check(std::abs(m.burn - 6.4 * 0.08) < 1e-12 && std::abs(m.SL * m.slabs[10].ry - 6.4) < 1e-12,
          "burn, S_L = burn / (A rho Y of the slab 2 delta_th in)");
    check(m.hits == 0 && m.yMin >= 0.0 && m.yMax <= 1.0, "Y bounds");

    // FvRef's species passes: byte-identical at any thread count.
    const Transport tr = Transport::constant(1.4, 0.05);
    const FvChem chem{.nY = 1, .fuel = 0, .lnA = 3.0, .Ta = 2.0, .q = 1.0, .invPr = 1.4};
    std::vector<FvBc> bc;
    for (const MeshPatch& pt : fm->patches) {
        FvBc b;
        b.kind = pt.type == FoamPatchType::Empty ? FvBcKind::Empty : FvBcKind::Slip;
        bc.push_back(b);
    }
    FvState a = st, b = st;
    auto r1 = FvRef::make(*fm, {.tr = tr, .bc = bc, .chem = chem}, 1), r4 = FvRef::make(*fm, {.tr = tr, .bc = bc, .chem = chem}, 4);
    check(r1 && r4, "FvRef with species");
    if (!r1 || !r4) return;
    r1->march(a, 1e-3, 5);
    r4->march(b, 1e-3, 5);
    check(a.q == b.q, "FvRef species march: 1 and 4 threads byte-identical");
    check(!FvRef::make(*fm, {.tr = tr, .bc = bc, .chem = {.nY = 5}}), "nY past the cap rejected");
    check(!FvRef::make(*fm, {.tr = tr, .bc = bc, .chem = {.nY = 1, .fuel = 1}}), "fuel past nY rejected");
}

// docs/fvcomp.md "Combinations": each pair of species, RANS, gravity is refused,
// on every spelling the CLI has and at FvRef::make; never a silent path.
void comboGate()
{
    auto refused = [](std::vector<std::string> a, std::string_view flag) {
        auto r = runCli(std::move(a));
        check(!r && (r.error().field + " " + r.error().message).find(flag) != std::string::npos,
              std::format("combination refused by name: {}", flag));
    };
    const std::vector<std::string> fl = {"flame", "--solver", "fv"}, cv = {"conv", "--solver", "fv"};
    auto plus = [](std::vector<std::string> a, std::vector<std::string> b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    // species + gravity; species + LES, as the structured flame spells them.
    refused(plus(fl, {"--gravity", "1"}), "--gravity");
    refused(plus(fl, {"--les", "0.2"}), "--les");
    // Options the two routes have no flag for: --rans, --steady are unknown there.
    refused(plus(fl, {"--rans", "sa-neg"}), "--rans");
    refused(plus(fl, {"--steady", "1e-3"}), "--steady");
    refused(plus(cv, {"--rans", "sa-neg"}), "--rans");
    refused(plus(cv, {"--steady", "1e-3"}), "--steady");
    refused(plus(cv, {"--gravity", "1"}), "--gravity");
    refused(plus(cv, {"--les", "0.2"}), "--les");
    // dns flow has no gravity flag; species keys of a --bc file: flameGate, cvBc below.
    refused({"flow", "--physics", "compressible", "--solver", "fv", "--ogrid", "--body", "cylinder", "--gravity", "1"},
            "--gravity");
    auto cvBc = parseFvBc("[reference]\norigin = [0, 0, 0]\n[patch.floor]\ntype = \"wall\"\n");
    check(cvBc && fvBcFlameKey(*cvBc) == "reference.origin", "a conv --bc file: the flame key is named");

    const Grid g = Grid::make(4, 4, 1, 1.0, 1.0, 0.1);
    const StructuredSpec sp{.lo = {0, 0, 0},
                            .hi = {3, 3, 0},
                            .name = {"a", "b", "c", "d", "e", "f"},
                            .type = {FoamPatchType::Wall, FoamPatchType::Wall, FoamPatchType::Wall, FoamPatchType::Wall,
                                     FoamPatchType::Empty, FoamPatchType::Empty}};
    auto fm = FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::Morton);
    check(fm.has_value(), "combo mesh");
    if (!fm) return;
    std::vector<FvBc> bc;
    for (const MeshPatch& pt : fm->patches) {
        FvBc b;
        b.kind = pt.type == FoamPatchType::Empty ? FvBcKind::Empty : FvBcKind::Wall;
        bc.push_back(b);
    }
    const Transport tr = Transport::constant(1.4, 0.05);
    const FvChem chem{.nY = 1, .fuel = 0, .lnA = 3.0, .Ta = 2.0, .q = 1.0, .invPr = 1.4};
    const FvRansConfig rans{.model = RansModel::SaNeg, .wallDist = std::vector<double>(std::size_t(fm->nC), 0.1)};
    const std::array<double, 3> gv{0.0, -1.0, 0.0}, f{0.1, 0.0, 0.0};
    auto no = [&](const FvConfig& c, std::string_view why) {
        auto r = FvRef::make(*fm, c);
        check(!r && r.error().find(why) != std::string::npos, std::format("FvRef::make refuses: {}", why));
    };
    no({.tr = tr, .bc = bc, .chem = chem, .rans = rans}, "species with RANS");
    no({.tr = tr, .bc = bc, .gravity = gv, .chem = chem}, "species with gravity");
    no({.tr = tr, .bc = bc, .chem = chem, .bodyForce = f}, "species with bodyForce");
    no({.tr = tr, .bc = bc, .gravity = gv, .rans = rans}, "RANS with gravity");
    no({.tr = tr, .bc = bc, .gravity = gv, .bodyForce = f}, "gravity with bodyForce");
    check(FvRef::make(*fm, {.tr = tr, .bc = bc, .gravity = gv}).has_value(), "gravity alone passes");
    check(FvRef::make(*fm, {.tr = tr, .bc = bc, .rans = rans, .bodyForce = f}).has_value(), "RANS + drive passes");
    check(FvRef::make(*fm, {.tr = tr, .bc = bc, .chem = chem}).has_value(), "species alone passes");
}

void raster()
{
    // A periodic-free unit box 8 x 8 x 1: every pixel inside, the containing cell's value.
    const Grid g = Grid::make(8, 8, 1, 1.0, 1.0, 0.1);
    const StructuredSpec sp{.lo = {0, 0, 0},
                            .hi = {7, 7, 0},
                            .name = {"a", "b", "c", "d", "e", "f"},
                            .type = {FoamPatchType::Patch, FoamPatchType::Patch, FoamPatchType::Patch,
                                     FoamPatchType::Patch, FoamPatchType::Empty, FoamPatchType::Empty}};
    auto fm = FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::Morton);
    check(fm.has_value(), "box mesh");
    if (!fm) return;
    auto r = FvRaster::make(*fm, {.longSide = 16, .box = {0.0, 1.0, 0.0, 1.0}});
    check(r.has_value() && r->grid().nx == 16 && r->grid().ny == 16, "16 x 16 plane");
    if (!r) return;
    std::vector<double> x(std::size_t(fm->nC));
    for (std::size_t c = 0; c < x.size(); ++c) x[c] = fm->C[0][c];
    std::vector<float> o(r->grid().size());
    r->sample(x, o);
    double worst = 0.0;
    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 16; ++i) {
            const double px = (i + 0.5) / 16.0, want = (std::floor(px * 8.0) + 0.5) / 8.0;
            worst = std::max(worst, std::abs(double(o[std::size_t(j) * 16 + std::size_t(i)]) - want));
        }
    check(worst < 1e-6 && std::ranges::all_of(r->mask(), [](float m) { return m == 1.0f; }),
          std::format("pixel -> containing cell, {:.1e}", worst));
    auto out = FvRaster::make(*fm, {.longSide = 16, .box = {-1.0, 0.0, 0.0, 1.0}});
    check(out && std::ranges::all_of(out->mask(), [](float m) { return m == 0.0f; }), "outside the mesh: mask 0");
}

} // namespace

int main()
{
    parseGate();
    bind();
    table();
    cliGate();
    flameGate();
    comboGate();
    raster();
    return dnstest::finish();
}
