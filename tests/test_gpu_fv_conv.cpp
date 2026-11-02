// dns conv on a FaceMesh (docs/fvcomp.md, docs/fvincomp.md), both models.
//   C1 gravity: double instantiation vs FvRef, rhs and 3 steps <= 1e-13, sine box
//   C2 compensated FP32 gravity tracks FP64 where plain drops sub-ulp updates
//   C3 Boussinesq: conv T energy 0 to rounding on uniform and random meshes
//   C4 rest: hydrostatic conduction state, residual velocity stated, both physics,
//      uniform and non-orthogonal (bounds in docs: measured)
//   C5 subcritical Ra 1000 from a trip: Nu 1.00 to 1%, the layer stops
//   C6 --polymesh is the box route: a binary polyMesh of the same mesh, byte-equal stats
//   C7 adiabatic closed sides: FV wall heat 0, the balance closes
//   C8 well balanced: the conduction state on a random mesh, FP64, to rounding
//   long (test_gpu_fv_conv long): Ra 5e4 2D 96 x 48: FV vs structured same model,
//      FV compressible vs FV Boussinesq (gpu_rb's 15%), walls vs interior 5%,
//      up-down symmetry broken, |div| < 1e-4
#include "fv_comp_test_util.hpp"

#include "dns/fvconv.hpp"
#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/la/graph.hpp"
#include "dns/polymesh.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <cstring>
#include <fstream>
#include <iterator>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

using namespace dnstest::fvc;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

Grid convGrid(int n, int ny)
{
    const double H = 1.0, hy = H / ny;
    return Grid::make(n, ny + 2, 1, 2.0, H + 2 * hy, 1.0);
}

std::vector<FvBc> plates(const FaceMesh& fm, double tb, double tt)
{
    return bcFor(fm, [&](const MeshPatch& p) {
        FvBc b;
        b.kind = FvBcKind::Wall;
        b.isothermal = true;
        b.tWall = p.name == "floor" ? tb : tt;
        return b;
    });
}

// Hydrostatic-ish state with a shear, the physics of every term.
FvState wobbly(const FaceMesh& fm)
{
    FvState s = FvState::zeros(fm.nC);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double x = fm.C[0][c], y = fm.C[1][c];
        const double rho = 1.0 + 0.1 * std::cos(3 * y) * std::sin(std::numbers::pi * x);
        const double u = 0.1 * std::sin(std::numbers::pi * y), v = 0.05 * std::cos(std::numbers::pi * x);
        s.var(0)[c] = rho;
        s.var(1)[c] = rho * u;
        s.var(2)[c] = rho * v;
        s.var(4)[c] = (1.0 - 0.1 * y) / (kGamma - 1.0) + 0.5 * rho * (u * u + v * v);
    }
    return s;
}

void gravityOracle()
{
    const Grid g = convGrid(24, 12);
    const FaceMesh fm = orDie(fvConvBox(g, false, FvPerturb::Sine, 0.08), "mesh");
    const Transport tr = Transport::constant(kGamma, 0.02);
    const std::vector<FvBc> bc = plates(fm, 1.05, 0.95);
    const std::array<double, 3> gv = {0.0, -0.3, 0.0};
    const FvRef ref = orDie(FvRef::make(fm, {.tr = tr, .bc = bc, .gravity = gv}), "FvRef");
    auto dev = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = bc, .precision = FvPrecision::Fp64, .gravity = gv}), "device");
    const FvState s0 = wobbly(fm);
    orDie(dev.upload(s0), "upload");
    FvState a, b;
    ref.rhs(s0, a);
    orDie(dev.rhs(b), "rhs");
    double e = 0, sc = 0;
    for (std::size_t i = 0; i < a.q.size(); ++i) {
        e = std::max(e, std::abs(a.q[i] - b.q[i]));
        sc = std::max(sc, std::abs(a.q[i]));
    }
    std::printf("  C1 rhs: device vs FvRef %.3e of %.3e\n", e, sc);
    check(e <= 1e-13 * sc, "C1 gravity rhs: device == FvRef to 1e-13");

    FvState o = s0;
    double dt = orDie(ref.stableDt(s0), "dt");
    ref.march(o, dt, 3);
    orDie(dev.upload(s0), "upload");
    for (int i = 0; i < 3; ++i) dev.step(dt);
    orDie(dev.download(b), "download");
    e = 0, sc = 0;
    for (std::size_t i = 0; i < o.q.size(); ++i) {
        e = std::max(e, std::abs(o.q[i] - b.q[i]));
        sc = std::max(sc, std::abs(o.q[i]));
    }
    std::printf("  C1 3 steps: device vs FvRef %.3e of %.3e\n", e, sc);
    check(e <= 1e-13 * sc, "C1 gravity march: device == FvRef to 1e-13");

    // Open patches and moving walls have no hydrostatic ambient.
    std::vector<FvBc> moved = bc;
    moved[0].uWall[0] = 0.1;
    check(!FvRef::make(fm, {.tr = tr, .bc = moved, .gravity = gv}), "C1 gravity rejects a moving wall");
    std::vector<FvBc> open = bc;
    open[0].kind = FvBcKind::Far;
    open[0].isothermal = false;
    open[0].qFar[4] = 2.5;
    check(!FvRef::make(fm, {.tr = tr, .bc = open, .gravity = gv}), "C1 gravity rejects an open patch");
}

void compensated()
{
    const Grid g = Grid::make(8, 8, 1, 1.0, 1.0, 1.0);
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "mesh");
    const Transport tr = Transport::constant(kGamma, 0.0);
    const std::vector<FvBc> bc = bcFor(fm, [](const MeshPatch&) { return FvBc{}; });
    const std::array<double, 3> gv = {0.0, -1e-7, 0.0}; // cr g below ulp(m)/2 on m ~ 1
    FvState s = FvState::zeros(fm.nC);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        s.var(0)[c] = 1.0;
        s.var(2)[c] = 1.0;
        s.var(4)[c] = 3.0;
    }
    auto run = [&](FvPrecision pr, bool comp) {
        auto d = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = bc, .precision = pr, .compensated = comp, .gravity = gv}),
                       "device");
        orDie(d.upload(s), "upload");
        for (int i = 0; i < 40; ++i) d.step(1e-2);
        FvState o;
        orDie(d.download(o), "download");
        return o;
    };
    auto refused = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = bc, .gravity = gv}), "device");
    const auto lts = refused.localDt({.cfl = 0.4});
    check(!lts && lts.error().find("with gravity") != std::string::npos, "local time stepping refused with gravity");
    const auto les = DeviceFvComp::make(fm, {.tr = tr, .bc = bc, .csLes = 0.17, .gravity = gv});
    check(!les && les.error().find("gravity with LES") != std::string::npos, "gravity + LES refused");
    const FvState d64 = run(FvPrecision::Fp64, false), cmp = run(FvPrecision::Fp32, true), pln = run(FvPrecision::Fp32, false);
    const double e64 = d64.var(2)[3] - 1.0, eC = cmp.var(2)[3] - 1.0, eP = pln.var(2)[3] - 1.0;
    std::printf("  C2 dm after 40 steps: FP64 %.4e, compensated %.4e, plain %.4e\n", e64, eC, eP);
    check(std::abs(eC - e64) < 0.02 * std::abs(e64), "C2 compensated gravity tracks FP64 to 2%");
    check(eP == 0.0, "C2 plain FP32 drops it (why --compensated exists)");
}

void thermalBudget()
{
    for (FvPerturb kind : {FvPerturb::None, FvPerturb::Random}) {
        const Grid g = convGrid(32, 16);
        const FaceMesh fm = orDie(fvConvBox(g, false, kind, 0.3), "mesh");
        FvIncompBcs b;
        b.patches = {{.name = "floor", .kind = FvIncompBc::Wall, .hasT = true, .T = 0.5},
                     {.name = "lid", .kind = FvIncompBc::Wall, .hasT = true, .T = -0.5}};
        orDie(bindFvIncompBcs(b, fm), "bind");
        const FvIncompSetup s = orDie(buildFvIncompSetup(fm, b), "setup");
        const std::size_t nC = std::size_t(fm.nC);
        std::vector<double> U(3 * nC), T(nC);
        for (std::size_t c = 0; c < nC; ++c) {
            const double x = fm.C[0][c], y = fm.C[1][c], pi = std::numbers::pi;
            U[c] = 0.2 * std::sin(pi * x) * std::cos(pi * y);
            U[nC + c] = -0.2 * std::cos(pi * x) * std::sin(pi * y);
            T[c] = 0.5 - y + 0.1 * std::sin(pi * x);
        }
        auto d = orDie(gpu::DeviceFvIncomp::make(s, la::graphOf(fm),
                                                 {.nu = 0.01, .kappa = 0.01, .betaG = {0.0, 1.0, 0.0}}, U, T),
                       "device");
        check(d.step(1e-3).has_value(), "C3 a step runs");
        const auto bd = orDie(d.budget(), "budget");
        std::printf("  C3 conv T energy %.3e of gross %.3e\n", bd.convT, bd.grossConvT);
        check(std::abs(bd.convT) < 1e-13 * bd.grossConvT, "C3 conv T energy is 0 to rounding");
    }
}

struct Tail
{
    bool ok = false;
    double nuBot = 0, nuTop = 0, nuFlux = 0, nuBotF = 0, nuTopF = 0, urms = 0, uLast = 0, maxDiv = 0, maxWallGap = 0;
};

double spreadOf(const Tail& t)
{
    return std::max({std::abs(t.nuBot - t.nuTop), std::abs(t.nuBot - t.nuFlux), std::abs(t.nuTop - t.nuFlux)}) / t.nuBot;
}

// Time mean of the last 60% of rows, gpu_rb's window.
Tail tailOf(const std::string& prefix)
{
    Tail r;
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    if (!tbl) return r;
    const auto& nb = tbl->col("nuBot");
    const auto& nt = tbl->col("nuTop");
    const auto& nf = tbl->col("nuFlux");
    const auto& ur = tbl->col("urms");
    if (nb.size() < 4) return r;
    const std::size_t i0 = nb.size() * 2 / 5, n = nb.size() - i0;
    for (std::size_t i = i0; i < nb.size(); ++i) {
        r.nuBot += nb[i] / double(n);
        r.nuTop += nt[i] / double(n);
        r.nuFlux += nf[i] / double(n);
        r.urms += ur[i] / double(n);
    }
    for (std::size_t i = 0; i < nb.size(); ++i) r.maxWallGap = std::max(r.maxWallGap, std::abs(nb[i] - nt[i])); // gpu_rb's: whole run
    r.uLast = ur.back();
    const auto hasCol = [&](const char* n) { return std::ranges::find(tbl->names, n) != tbl->names.end(); };
    if (hasCol("nuBotF")) {
        const auto& fb = tbl->col("nuBotF");
        const auto& ft = tbl->col("nuTopF");
        for (std::size_t i = i0; i < nb.size(); ++i) {
            r.nuBotF += fb[i] / double(n);
            r.nuTopF += ft[i] / double(n);
        }
    }
    if (hasCol("maxDiv"))
        for (double d : tbl->col("maxDiv")) r.maxDiv = std::max(r.maxDiv, d);
    r.ok = true;
    return r;
}

Tail run(const std::string& name, const std::string& args)
{
    const std::string prefix = "/tmp/dns_test_fvconv_" + name;
    const std::string cmd = "\"" DNS_BIN "\" conv " + args + " --emit stats --quiet --out " + prefix;
    const int rc = valtest::runBinary(cmd);
    check(rc == 0, name + ": run exits 0");
    return rc == 0 ? tailOf(prefix) : Tail{};
}

const char* const kBox2D = " --nz 1 --n 64 --ny 32 ";

void rest()
{
    struct R
    {
        const char* name;
        const char* args;
        double bound; // measured, docs/fvcomp.md and docs/fvincomp.md (urms / Uff at the end)
    };
    const R rows[] = {
        {"restC", "--solver fv --physics compressible --kick 0 --rayleigh 1000 --tt 4", 2e-6},
        {"restCs", "--solver fv --physics compressible --kick 0 --rayleigh 1000 --tt 4 --mesh-perturb 0.08", 2e-5},
        {"restB", "--solver fv --physics incompressible --kick 0 --rayleigh 1000 --tt 4", 1e-8},
        {"restBs", "--solver fv --physics incompressible --kick 0 --rayleigh 1000 --tt 4 --mesh-perturb 0.08", 3e-4},
    };
    for (const R& r : rows) {
        const Tail t = run(r.name, std::string(kBox2D) + r.args);
        std::printf("  C4 %s: urms at the end %.3e, nu %.5f / %.5f\n", r.name, t.uLast, t.nuBot, t.nuTop);
        check(t.ok && t.uLast < r.bound, std::string("C4 rest residual: ") + r.name);
    }
}

// C8 well balanced: the conduction state on the random mesh, FP64: urms at rounding,
// plane statistics exact (the old face mean read nuFlux 0.61, urms 1.1e-2 Uff).
void wellBalanced()
{
    const Tail t = run("restW", std::string(kBox2D) + "--solver fv --physics compressible --kick 0 --rayleigh 1000 --tt 4 "
                                                      "--fv-precision fp64 --mesh-perturb 0.3 --mesh-perturb-kind random");
    std::printf("  C8 random mesh, FP64: urms %.3e, nuFlux %.6f, face %.6f / %.6f\n", t.uLast, t.nuFlux, t.nuBotF, t.nuTopF);
    check(t.ok && t.uLast < 1e-13, "C8 rest on the random mesh: urms < 1e-13");
    check(t.ok && std::abs(t.nuFlux - 1.0) < 1e-6, "C8 nuFlux 1 on the random mesh");
}

void subcritical()
{
    for (const char* ph : {"compressible", "incompressible"}) {
        const Tail t = run(std::string("sub") + ph, std::string(kBox2D) + "--solver fv --rayleigh 1000 --tt 40 --physics " + ph);
        std::printf("  C5 %s: Nu %.4f / %.4f flux %.4f, face %.4f / %.4f, urms end %.3e\n", ph, t.nuBot, t.nuTop, t.nuFlux,
                    t.nuBotF, t.nuTopF, t.uLast);
        check(t.ok && std::abs(t.nuBot - 1) < 0.01 && std::abs(t.nuTop - 1) < 0.01, "C5 subcritical: Nu 1 to 1%");
        check(t.ok && t.uLast < 1e-3, "C5 subcritical: the layer stops");
    }
}

std::string readAll(const std::string& p)
{
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

// The rows without the label comment: a mesh path names itself there.
std::string rows(const std::string& p)
{
    std::string s = readAll(p);
    const auto at = s.find("t,dt,");
    return at == std::string::npos ? s : s.substr(at);
}

void polyRoute()
{
    const fs::path dir = "/tmp/dns_test_fvconv_mesh";
    fs::remove_all(dir);
    const Grid g = convGrid(32, 16);
    const CurvMesh m = fvConvCurv(g, false, FvPerturb::Sine, 0.05);
    const PolyMesh pm = orDie(FaceMesh::polyOf(m, fvConvSpec(g, false)), "polyOf");
    orDie(writePolyMesh(dir, pm, true), "write");
    const std::string common = "--rayleigh 5e4 --tt 1 --aspect 2 ";
    {
        std::ofstream(dir / "c.toml") << "[patch.floor]\ntype = \"wall\"\nthermal = \"isothermal\"\ntwall = 1.05\n"
                                         "[patch.lid]\ntype = \"wall\"\nthermal = \"isothermal\"\ntwall = 0.95\n"
                                         "[patch.back]\ntype = \"empty\"\n[patch.front]\ntype = \"empty\"\n";
        std::ofstream(dir / "b.toml") << "[patch.floor]\nkind = \"wall\"\nt = 0.5\n[patch.lid]\nkind = \"wall\"\nt = -0.5\n";
    }
    const std::string box = "--solver fv --nz 1 --n 32 --ny 16 --mesh-perturb 0.05 ", poly = "--polymesh " + dir.string() + " --ny 16 ";
    for (auto [ph, toml] : {std::pair{"compressible", "c.toml"}, {"incompressible", "b.toml"}}) {
        const std::string a = std::string("polyA") + ph, b = std::string("polyB") + ph;
        run(a, box + common + "--physics " + ph);
        run(b, poly + common + "--physics " + ph + " --bc " + (dir / toml).string());
        const std::string ra = rows("/tmp/dns_test_fvconv_" + a + "_stats.csv"), rb = rows("/tmp/dns_test_fvconv_" + b + "_stats.csv");
        check(!ra.empty() && ra == rb, std::string("C6 --polymesh == the box route, byte for byte: ") + ph);
    }
}

void closedSides()
{
    const Tail c = run("closedC", std::string(kBox2D) + "--solver fv --sides closed --rayleigh 5e4 --tt 2 --physics compressible");
    const Tail b = run("closedB", std::string(kBox2D) + "--solver fv --sides closed --rayleigh 5e4 --tt 12 --physics incompressible");
    check(c.ok && b.ok, "C7 --sides closed runs on both physics");
}

void longGates()
{
    const std::string s = " --nz 1 --n 96 --ny 48 --rayleigh 5e4 --tt 40 ";
    const Tail fb = run("lfb", "--solver fv --physics incompressible --fv-precision fp64" + s); // FP32: |div| floor 1.4e-4
    const Tail fc = run("lfc", "--solver fv --physics compressible" + s);
    const Tail sb = run("lsb", "--physics incompressible" + s);
    const Tail sc = run("lsc", "--physics compressible" + s);
    auto rel = [](double a, double b) { return std::abs(a - b) / (0.5 * (a + b)); };
    std::printf("  FV B %.3f/%.3f/%.3f, FV C %.3f/%.3f/%.3f, MAC %.3f, structured C %.3f\n", fb.nuBot, fb.nuTop, fb.nuFlux,
                fc.nuBot, fc.nuTop, fc.nuFlux, sb.nuBot, sc.nuBot);
    check(fb.ok && fc.ok && rel(fb.nuBot, fc.nuBot) < 0.15, "L FV Boussinesq vs FV compressible Nu to 15%");
    check(fb.ok && spreadOf(fb) < 0.05, "L Boussinesq walls vs interior Nu to 5%");
    check(fc.ok && spreadOf(fc) < 0.05, "L compressible walls vs interior Nu to 5%");
    check(fb.ok && fb.maxDiv < 1e-4, "L |div| < 1e-4");
    check(fb.ok && fb.maxWallGap > 0.05 * fb.nuBot, "L the up-down symmetry is broken");
    check(fb.ok && sb.ok && rel(fb.nuBot, sb.nuBot) < 0.02, "L FV vs MAC Boussinesq Nu to 2%");
    check(fc.ok && sc.ok && rel(fc.nuBot, sc.nuBot) < 0.02, "L FV vs structured compressible Nu to 2%");
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    if (argc > 1 && std::strcmp(argv[1], "long") == 0) {
        longGates();
        return dnstest::finish();
    }
    gravityOracle();
    compensated();
    thermalBudget();
    rest();
    wellBalanced();
    subcritical();
    polyRoute();
    closedSides();
    return dnstest::finish();
}
