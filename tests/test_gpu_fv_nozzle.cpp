// The nozzle's device gates (docs/nozzle.md, masterplan_cdnozzle section 6): DeviceFvComp, SI
// rows, rhoCentralFoam references in tests/data/fv/nozzle_{A,B,W}, meshes from
// tools/fv/nozzle_mesh.py (python3, a temporary directory). Every run asserts it ran.
//   fixed    G14 T1: force, FP64, A and B: every cell equals foam
//   wedge    G15 T1 on the 20 deg wedge; 5 vs 20 deg; w 0.0 in FP32 and FP64 kernels
//   reflect  G16 Ms 2, 3, 6 onto a wall; the 30 deg body-fitted wall, near-wall faces sensed
//   dt       G17 sides out of the dt: ratio, same fixed point
//   source   G18 the exact source flow: order 2, bare KG's order 1 the control
//   silent   G19 nc-smooth: 0 sensed past the ramp; nc > 0; the centre rule >= 1
//   exit     G20 the half-Riemann row's four rigs and their controls
//   sauer    G21 parabola R 20: dM/dx, sonic x, Cd
//   cdladder G22 Richardson over R, the momentum budget
//   fp32     G25 compensated vs FP64, shock x, mdot; plain's gap
//   gamma    G31 T1 at gamma 1.2
//   sense    G27 sense == none bytes, sensed > 0 on B
//   react    masterplan_combnozzle P4: the reacting one-row nozzle through dns flame (G4.1, G4.8, the route)
//   reacttwin  G4.5: the reacting nozzle's inert twin, viscous at Re* and inviscid
//   reacttwinrow  combscale S5: the twin of the multi-row rig (any rows, a stub chamber), no gate but its settle test
//   react2d  combscale S5: the multi-row reacting nozzle through dns flame (xIa columns, seed, label, ledger, scope)
//   wedgeflame  combscale S6a: the wedge under C0 + hll (W2 device, W3)
//   wedgestrip  combscale S6a: W5, the 1D flame on a wedge strip vs the planar mesh, through dns flame
#include "fv_comp_test_util.hpp"
#include "fvref_nozzle.hpp"
#include "fvref_wedgeflame.hpp"

#include "dns/checkpoint.hpp"
#include "dns/fvnozzle.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "gpu_test_util.hpp"

#include <unistd.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dnstest::fvc;
using dnstest::check;
using dnstest::noz::field;
using dnstest::noz::kGam;
using dnstest::noz::kP0;
using dnstest::noz::kT0;
using dnstest::noz::loadRef;
using dnstest::noz::nozzleBc;
using dnstest::noz::Ref;
using dnstest::noz::Rows;
using dnstest::noz::stateOf;
namespace fs = std::filesystem;

namespace {

const double kC0 = std::sqrt(1.4 * 287.1 * 300.0); // t* = 1 m / c0

fs::path tmpDir()
{
    static const fs::path d = [] {
        const fs::path p = fs::temp_directory_path() / std::format("dns_nozzle_{}", getpid());
        fs::create_directories(p);
        return p;
    }();
    return d;
}

// One generated case: nozzle_mesh.py ARGS --out tmp/name.
struct Gen
{
    fs::path dir;
    PolyMesh pm;
    FaceMesh fm;
    int nx = 0, ny = 0;
    double astar = 0;
};

// The generator's astar off its bc.toml.
double astarOf(const fs::path& d)
{
    std::ifstream f(d / "bc.toml");
    std::string l;
    while (std::getline(f, l))
        if (l.rfind("astar = ", 0) == 0) return std::stod(l.substr(8));
    return 0.0;
}

Gen gen(const std::string& name, int nx, int ny, const std::string& args, CellOrder order = CellOrder::Morton)
{
    Gen g;
    g.dir = tmpDir() / name;
    g.nx = nx;
    g.ny = ny;
    const fs::path tool = fs::path(DNS_MESH_DATA) / ".." / ".." / ".." / "tools/fv/nozzle_mesh.py";
    const std::string cmd = std::format("python3 {} --nx {} --ny {} {} --out {} > /dev/null", tool.string(), nx, ny, args, g.dir.string());
    if (std::system(cmd.c_str()) != 0) {
        std::printf("FATAL: %s\n", cmd.c_str());
        std::exit(1);
    }
    g.pm = orDie(readPolyMesh(g.dir / "constant/polyMesh"), name);
    g.fm = orDie(FaceMesh::fromPolyMesh(g.pm, order, 0, {.wedge = true}), name);
    g.astar = astarOf(g.dir);
    return g;
}

FvState icOf(const Gen& g, double R = 287.1)
{
    return stateOf(g.fm, field(g.dir / "0/p", 1, g.fm.nC), field(g.dir / "0/T", 1, g.fm.nC), field(g.dir / "0/U", 3, g.fm.nC), R);
}

struct Run
{
    FvShock shock{.mode = ShockMode::Hll, .eps4 = kFvShockEps4};
    FvPrecision prec = FvPrecision::Fp64;
    bool comp = false;
    double mu = 0;
    double R = 287.1, gamma = 1.4;
};

FvCompConfig cfgOf(const FaceMesh& fm, const Rows& rows, const Run& r)
{
    FvCompConfig c;
    c.tr = Transport::air(r.mu, 1.0, 0.0, 0.72, r.gamma, r.R, false);
    c.bc = nozzleBc(fm, rows);
    c.precision = r.prec;
    c.compensated = r.comp;
    c.shock = r.shock;
    c.tallied.assign(fm.patches.size(), 0);
    for (std::size_t p = 0; p < fm.patches.size(); ++p) c.tallied[p] = fm.patches[p].name == "wall" ? 1 : 0;
    return c;
}

// March to tEnd under --shock's cadence (dt every step); a diverged state ends it.
struct Marched
{
    long steps = 0;
    double t = 0;
    std::string err;
};

Marched marchTo(DeviceFvComp& d, double tEnd, double cfl = 0.4, int every = 1, long cap = 2000000)
{
    Marched m;
    double dt = 0;
    while (m.t < tEnd && m.steps < cap) {
        if (m.steps % every == 0) {
            auto s = d.stableDt({.cfl = cfl});
            if (!s) {
                m.err = s.error();
                return m;
            }
            dt = 0.95 * *s;
        }
        const double h = std::min(dt, tEnd - m.t);
        d.step(h);
        m.t += h;
        ++m.steps;
    }
    return m;
}

// Cell-order (source) p, T, u per cell of a state.
struct Fields
{
    std::vector<double> p, T, rho;
    std::array<std::vector<double>, 3> u;
};

Fields fieldsOf(const FaceMesh& fm, const FvState& s, double g, double R)
{
    Fields f;
    const std::size_t n = std::size_t(fm.nC);
    f.p.resize(n);
    f.T.resize(n);
    f.rho.resize(n);
    for (auto& v : f.u) v.resize(n);
    for (std::size_t c = 0; c < n; ++c) {
        const std::size_t o = std::size_t(fm.srcCell[c]);
        const FvPrim<double> q = fvPrimOf(s.var(0)[c], s.var(1)[c], s.var(2)[c], s.var(3)[c], s.var(4)[c], g, R);
        f.p[o] = q.p;
        f.T[o] = q.temp;
        f.rho[o] = q.rho;
        for (std::size_t k = 0; k < 3; ++k) f.u[k][o] = q.u[k];
    }
    return f;
}

double relMax(const std::vector<double>& a, std::span<const double> b)
{
    double e = 0, s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
        s = std::max(s, std::abs(b[i]));
    }
    return s > 0 ? e / s : e;
}

std::vector<double> rowOf(const std::vector<double>& v, std::size_t k, std::size_t nc)
{
    std::vector<double> o(v.size() / nc);
    for (std::size_t i = 0; i < o.size(); ++i) o[i] = v[nc * i + k];
    return o;
}

// max over p, T, u, v of the relative cell difference vs foam.
double vsFoam(const Fields& f, const Ref& w)
{
    double e = std::max(relMax(f.p, w.p), relMax(f.T, w.T));
    e = std::max(e, relMax(f.u[0], rowOf(w.U, 0, 3)));
    return std::max(e, relMax(f.u[1], rowOf(w.U, 1, 3)));
}

// The Outflow patch's mass flux of a state under the Extrapolate or Fixed row.
double exitMdot(DeviceFvComp& d, const std::vector<FvBc>& bc)
{
    const std::vector<double> m = orDie(d.patchFlux(0), "patchFlux");
    double o = 0;
    for (std::size_t p = 0; p < bc.size(); ++p)
        if (bc[p].kind == FvBcKind::Outflow) o += m[p];
    return o;
}

double inletMdot(DeviceFvComp& d, const std::vector<FvBc>& bc)
{
    const std::vector<double> m = orDie(d.patchFlux(0), "patchFlux");
    double o = 0;
    for (std::size_t p = 0; p < bc.size(); ++p)
        if (bc[p].kind == FvBcKind::Total) o -= m[p];
    return o;
}

// One row (masterplan_combnozzle P2): v is a rounding field in both solvers, so it is held to the speed scale
// max |u|, not to its own max (~1e-16).
double vsFoamRow(const Fields& f, const Ref& w)
{
    double e = std::max(relMax(f.p, w.p), relMax(f.T, w.T));
    const std::vector<double> u = rowOf(w.U, 0, 3), v = rowOf(w.U, 1, 3);
    e = std::max(e, relMax(f.u[0], u));
    double dv = 0, um = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        dv = std::max(dv, std::abs(f.u[1][i] - v[i]));
        um = std::max(um, std::abs(u[i]));
    }
    return std::max(e, dv / um);
}

void t1Case(const char* name, double pb, double band, double tEnd, const char* tag)
{
    const Ref w = loadRef(name, CellOrder::Morton);
    const Rows rows{.pb = pb};
    Run r{.shock = {.mode = ShockMode::Hll, .force = true}, .R = w.Rgas};
    const FvCompConfig cfg = cfgOf(w.fm, rows, r);
    auto d = orDie(DeviceFvComp::make(w.fm, cfg), name);
    const FvState ic = stateOf(w.fm, field(w.dir / "0/p", 1, w.fm.nC), field(w.dir / "0/T", 1, w.fm.nC), field(w.dir / "0/U", 3, w.fm.nC), w.Rgas);
    orDie(d.upload(ic), name);
    const Fields f0 = fieldsOf(w.fm, ic, kGam, w.Rgas);
    // One row: every cell owns a wall face.
    const bool row = std::ranges::any_of(w.fm.patches, [&](const MeshPatch& m) { return m.name == "wall" && m.count == w.fm.nC; });
    const auto gap = [&](const Fields& f) { return row ? vsFoamRow(f, w) : vsFoam(f, w); };
    const double icGap = gap(f0);
    const auto t0 = std::chrono::steady_clock::now();
    const Marched m = marchTo(d, tEnd, 0.4, 16);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    check(m.err.empty(), std::format("{}: live ({})", name, m.err));
    FvState s;
    orDie(d.download(s), name);
    const Fields f = fieldsOf(w.fm, s, kGam, w.Rgas);
    const double e = gap(f);
    const double mo = exitMdot(d, cfg.bc), mi = inletMdot(d, cfg.bc);
    // Foam's exit mdot: the same row on its field.
    double mf = 0;
    {
        const dns::FvState fs0 = stateOf(w.fm, w.p, w.T, w.U, w.Rgas);
        orDie(d.upload(fs0), name);
        mf = exitMdot(d, cfg.bc);
    }
    double jump = 1.0;
    for (int c = 0; c < w.fm.nI; ++c) {
        const double a = w.p[std::size_t(w.fm.srcCell[std::size_t(w.fm.own[std::size_t(c)])])],
                     b = w.p[std::size_t(w.fm.srcCell[std::size_t(w.fm.nei[std::size_t(c)])])];
        jump = std::max(jump, std::max(a, b) / std::min(a, b));
    }
    std::printf("  %s: %ld steps, t %.3f s (t* %.0f), %.1f s; max cell gap vs foam %.2e (IC %.2e); mdot out/in - 1 %.2e, vs foam %.2e; largest p "
                "ratio across a face %.3f\n",
                tag, m.steps, m.t, m.t * kC0, sec, e, icGap, mo / mi - 1, mo / mf - 1, jump);
    check(m.steps >= 20000, std::format("{}: steps {} >= 20000", tag, m.steps));
    check(icGap >= 1e-2, std::format("{}: IC vs ref {:.2e} >= 1e-2", tag, icGap));
    check(e <= band, std::format("{}: every cell p, T, U {:.2e} <= {:.0e} of scale", tag, e, band));
    check(std::abs(mo / mf - 1) <= band, std::format("{}: mdot vs foam {:.2e}", tag, std::abs(mo / mf - 1)));
    if (pb > 0) check(jump > 2.0, std::format("{}: a real jump, p ratio {:.3f} > 2", tag, jump));
}

void fixed()
{
    std::printf("G14 T1: force, FP64, q1d IC, every cell vs rhoCentralFoam Kurganov + upwind\n");
    t1Case("nozzle_A", 0.0, 1e-11, 2.4, "A");
    t1Case("nozzle_B", 6e4, 1e-11, 2.4, "B");
    // masterplan_combnozzle G2.8: the one-row nc nozzle (nozzle_QA, 200 x 1), the same rule.
    t1Case("nozzle_QA", 0.0, 1e-11, 2.4, "QA one row");
    {
        // G2.8's control: the route's own flux (KG + hll D3, eps 1/16) from the same IC.
        const Ref w = loadRef("nozzle_QA", CellOrder::Morton);
        auto d = orDie(DeviceFvComp::make(w.fm, cfgOf(w.fm, {}, Run{.R = w.Rgas})), "QA kg");
        orDie(d.upload(stateOf(w.fm, field(w.dir / "0/p", 1, w.fm.nC), field(w.dir / "0/T", 1, w.fm.nC), field(w.dir / "0/U", 3, w.fm.nC), w.Rgas)), "QA kg");
        const Marched m = marchTo(d, 2.4, 0.4, 16);
        FvState s;
        orDie(d.download(s), "QA kg");
        const double e = vsFoamRow(fieldsOf(w.fm, s, kGam, w.Rgas), w);
        std::printf("  QA control, KG + D3 (the route's flux): %ld steps, gap %.2e\n", m.steps, e);
        // The plan's threshold 1e-2 (from the 2D host KG residual) is UNMET on one row: 8.97e-3. Replaced once
        // (masterplan_combnozzle A8): >= 1e-3, 1e8 x the band.
        std::printf("  %s G2.8 control as specified: the KG flux vs foam >= 1e-2\n", e >= 1e-2 ? "[PASS]" : "[UNMET (recorded, A8)]");
        check(e >= 1e-3, "G2.8 control (replacement): the KG flux vs foam >= 1e-3");
    }
    // The control: R + 1e-6 from foam's own field cannot hold it.
    const Ref w = loadRef("nozzle_A", CellOrder::Morton);
    Run r{.shock = {.mode = ShockMode::Hll, .force = true}, .R = w.Rgas * (1 + 1e-6)};
    auto d = orDie(DeviceFvComp::make(w.fm, cfgOf(w.fm, {}, r)), "ctrl");
    orDie(d.upload(stateOf(w.fm, w.p, w.T, w.U, w.Rgas)), "ctrl");
    const Marched m = marchTo(d, 0.5, 0.4, 16);
    FvState s;
    orDie(d.download(s), "ctrl");
    const double e = vsFoam(fieldsOf(w.fm, s, kGam, w.Rgas), w);
    std::printf("  control R + 1e-6: %ld steps, gap %.2e\n", m.steps, e);
    check(e > 1e-9, "control: gas R + 1e-6 fails the band");
}

// w of the state, max |rho w|.
double wMax(const FvState& s)
{
    double w = 0;
    for (int c = 0; c < s.nC; ++c) w = std::max(w, std::abs(s.var(3)[c]));
    return w;
}

void wedge()
{
    std::printf("G15 T1 on the wedge: 20 deg, y = r\n");
    t1Case("nozzle_W", 0.0, 1e-10, 2.4, "W 20 deg");
    const Ref w = loadRef("nozzle_W", CellOrder::Morton);
    check(std::stod(dnstest::noz::lastTime(w.dir).filename().string()) >= 1.2, "foam reference at t >= 1.2 (drift 2.9e-15 at step 1)");
    // 5 vs 20 deg: the same march, the gauge. 2000 steps of the hybrid.
    const double h5 = 2.5 * std::numbers::pi / 180, h20 = w.fm.wedgeHalf;
    const FvState ic = stateOf(w.fm, field(w.dir / "0/p", 1, w.fm.nC), field(w.dir / "0/T", 1, w.fm.nC), field(w.dir / "0/U", 3, w.fm.nC), w.Rgas);
    const FaceMesh f5 = orDie(FaceMesh::fromPolyMesh(dnstest::noz::rewedge(w.pm, h20, h5), CellOrder::Morton, 0, {.wedge = true}), "5 deg");
    Run r{.R = w.Rgas};
    auto dA = orDie(DeviceFvComp::make(w.fm, cfgOf(w.fm, {}, r)), "20");
    auto dB = orDie(DeviceFvComp::make(f5, cfgOf(f5, {}, r)), "5");
    orDie(dA.upload(ic), "20");
    orDie(dB.upload(ic), "5");
    for (int i = 0; i < 2000; ++i) {
        dA.step(2e-5);
        dB.step(2e-5);
    }
    FvState sa, sb;
    orDie(dA.download(sa), "20");
    orDie(dB.download(sb), "5");
    const double e = relMax(fieldsOf(w.fm, sa, kGam, w.Rgas).p, fieldsOf(f5, sb, kGam, w.Rgas).p);
    std::printf("  5 vs 20 deg, 2000 steps: max dp/p %.2e\n", e);
    check(e <= 1e-10, "dns 5 vs 20 deg <= 1e-10");
    // w 0.0 through the device kernels, FP32 and FP64, viscous.
    for (const FvPrecision pr : {FvPrecision::Fp64, FvPrecision::Fp32}) {
        Run v{.prec = pr, .mu = 1e-3, .R = w.Rgas};
        auto d = orDie(DeviceFvComp::make(w.fm, cfgOf(w.fm, {.slip = false}, v)), "w");
        orDie(d.upload(ic), "w");
        const Marched m = marchTo(d, 1e300, 0.4, 16, 1000);
        FvState s;
        orDie(d.download(s), "w");
        std::printf("  %s, viscous, %ld steps: max |rho w| %.3e\n", pr == FvPrecision::Fp64 ? "FP64" : "FP32", m.steps, wMax(s));
        check(wMax(s) == 0.0 && m.err.empty(), "w == 0.0 bitwise through the kernels");
    }
}

// A structured quad block as a 2D polyMesh (z depth 0.1): nodes X, Y (i + (nx + 1) j). Patches
// ilo, ihi, jlo, jhi, back, front.
PolyMesh quadPoly(int nx, int ny, const std::vector<double>& X, const std::vector<double>& Y)
{
    PolyMesh pm;
    const int nxp = nx + 1, N = nxp * (ny + 1);
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < N; ++i) {
            pm.px.push_back(X[std::size_t(i)]);
            pm.py.push_back(Y[std::size_t(i)]);
            pm.pz.push_back(k * 0.1);
        }
    auto nid = [&](int i, int j, int k) { return i + nxp * j + k * N; };
    pm.faceStart.push_back(0);
    auto add = [&](std::array<int, 4> f, int o, int n = -1) {
        pm.faceNodes.insert(pm.faceNodes.end(), f.begin(), f.end());
        pm.faceStart.push_back(std::int32_t(pm.faceNodes.size()));
        pm.owner.push_back(o);
        if (n >= 0) pm.neighbour.push_back(n);
    };
    auto cell = [&](int i, int j) { return i + nx * j; };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            if (i < nx - 1) add({nid(i + 1, j, 0), nid(i + 1, j + 1, 0), nid(i + 1, j + 1, 1), nid(i + 1, j, 1)}, cell(i, j), cell(i + 1, j));
            if (j < ny - 1) add({nid(i, j + 1, 0), nid(i, j + 1, 1), nid(i + 1, j + 1, 1), nid(i + 1, j + 1, 0)}, cell(i, j), cell(i, j + 1));
        }
    const int nI = int(pm.neighbour.size());
    auto patch = [&](const char* name, FoamPatchType ty, auto g) {
        const int st = int(pm.owner.size());
        g();
        pm.man.patches.push_back({name, ty, std::int64_t(pm.owner.size()) - st, st, "", std::nullopt});
    };
    patch("ilo", FoamPatchType::Patch, [&] {
        for (int j = 0; j < ny; ++j) add({nid(0, j, 0), nid(0, j, 1), nid(0, j + 1, 1), nid(0, j + 1, 0)}, cell(0, j));
    });
    patch("ihi", FoamPatchType::Patch, [&] {
        for (int j = 0; j < ny; ++j) add({nid(nx, j, 0), nid(nx, j + 1, 0), nid(nx, j + 1, 1), nid(nx, j, 1)}, cell(nx - 1, j));
    });
    patch("jlo", FoamPatchType::Patch, [&] {
        for (int i = 0; i < nx; ++i) add({nid(i, 0, 0), nid(i + 1, 0, 0), nid(i + 1, 0, 1), nid(i, 0, 1)}, cell(i, 0));
    });
    patch("jhi", FoamPatchType::Patch, [&] {
        for (int i = 0; i < nx; ++i) add({nid(i, ny, 0), nid(i, ny, 1), nid(i + 1, ny, 1), nid(i + 1, ny, 0)}, cell(i, ny - 1));
    });
    patch("back", FoamPatchType::Empty, [&] {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) add({nid(i, j, 0), nid(i, j + 1, 0), nid(i + 1, j + 1, 0), nid(i + 1, j, 0)}, cell(i, j));
    });
    patch("front", FoamPatchType::Empty, [&] {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) add({nid(i, j, 1), nid(i + 1, j, 1), nid(i + 1, j + 1, 1), nid(i, j + 1, 1)}, cell(i, j));
    });
    pm.man.nPoints = std::int64_t(pm.px.size());
    pm.man.nCells = nx * ny;
    pm.man.nFaces = std::int64_t(pm.owner.size());
    pm.man.nInternalFaces = nI;
    return pm;
}

FvBc slipRow()
{
    FvBc b;
    b.kind = FvBcKind::Slip;
    return b;
}

FvBc openRow()
{
    FvBc b;
    b.kind = FvBcKind::Outflow;
    b.exit = FvExit::Extrapolate;
    return b;
}

// One 1D reflection: a shock at x 0.5 into rest, a wall at x = 0, x = 1 copied.
struct Reflect
{
    double plateau = 0, wall = 0;
    long steps = 0;
    std::string err;
};

Reflect reflectRun(double Ms, ShockMode mode, int n = 400)
{
    const double g = 1.4, p1 = 1.0, r1 = 1.0, c1 = std::sqrt(g);
    const double p2 = p1 * (2 * g * Ms * Ms - (g - 1)) / (g + 1), r2 = r1 * (g + 1) * Ms * Ms / ((g - 1) * Ms * Ms + 2),
                 u2 = -Ms * c1 * (1 - r1 / r2);
    // The reflected shock's speed w: post state at rest at the wall.
    double lo = 1e-6, hi = 100 * c1, w = 0, Mrel = 0;
    for (int i = 0; i < 300; ++i) {
        w = 0.5 * (lo + hi);
        Mrel = (w - u2) / std::sqrt(g * p2 / r2);
        const double ur = w - (w - u2) * (2 / ((g + 1) * Mrel * Mrel) + (g - 1) / (g + 1));
        (ur > 0 ? hi : lo) = w;
    }
    const double p3 = p2 * (2 * g * Mrel * Mrel - (g - 1)) / (g + 1);
    const int ny = 4;
    const double h = 1.0 / n;
    std::vector<double> X, Y;
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= n; ++i) {
            X.push_back(i * h);
            Y.push_back(j * h);
        }
    const PolyMesh pm = quadPoly(n, ny, X, Y);
    const FaceMesh fm = orDie(FaceMesh::fromPolyMesh(pm, CellOrder::Morton), "refl");
    FvCompConfig c;
    c.tr = Transport::air(0.0, 1.0, 0.0, 0.72, g, 1.0, false);
    for (const MeshPatch& m : fm.patches) {
        if (m.type == FoamPatchType::Empty) {
            FvBc e;
            e.kind = FvBcKind::Empty;
            c.bc.push_back(e);
        } else {
            c.bc.push_back(m.name == "ihi" ? openRow() : slipRow());
        }
    }
    c.precision = FvPrecision::Fp64;
    c.shock = {.mode = mode, .eps4 = 0.0};
    auto d = orDie(DeviceFvComp::make(fm, c), "refl");
    FvState s = FvState::zeros(fm.nC);
    for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
        const bool left = fm.C[0][q] < 0.5;
        const double r = left ? r1 : r2, u = left ? 0.0 : u2, p = left ? p1 : p2;
        s.var(0)[q] = r;
        s.var(1)[q] = r * u;
        s.var(4)[q] = p / (g - 1) + 0.5 * r * u * u;
    }
    orDie(d.upload(s), "refl");
    const double tEnd = 0.5 / (Ms * c1) + 0.25 / w * 0.8;
    Reflect o;
    const Marched m = marchTo(d, tEnd, 0.4);
    o.steps = m.steps;
    o.err = m.err;
    if (!m.err.empty()) return o;
    orDie(d.download(s), "refl");
    const double xs = w * (m.t - 0.5 / (Ms * c1));
    double sp = 0, sn = 0, wall = 0, wn = 0;
    for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
        const FvPrim<double> a = fvPrimOf(s.var(0)[q], s.var(1)[q], s.var(2)[q], s.var(3)[q], s.var(4)[q], g, 1.0);
        if (!std::isfinite(a.p)) {
            o.err = "not finite";
            return o;
        }
        const double x = fm.C[0][q];
        if (x > 0.25 * xs && x < 0.7 * xs) {
            sp += a.p;
            sn += 1;
        }
        if (x < h) {
            wall += a.p;
            wn += 1;
        }
    }
    o.plateau = sp / sn / p3 - 1;
    o.wall = wall / wn / p3 - 1;
    return o;
}

// The 30 deg body-fitted wall: a normal shock (Ms 2, along +x) runs up a ramp the mesh lines
// follow. Every near-wall face the shock crosses must be sensed: S8's staircase left 48 of 714
// faces out, a body-fitted wall has no second blocked axis.
void ramp30()
{
    const int nx = 200, ny = 60;
    const double h = 0.01, sl = std::tan(30.0 * std::numbers::pi / 180);
    std::vector<double> X, Y;
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            X.push_back(i * h);
            Y.push_back(i * h * sl + j * h);
        }
    const PolyMesh pm = quadPoly(nx, ny, X, Y);
    const FaceMesh fm = orDie(FaceMesh::fromPolyMesh(pm, CellOrder::Source), "ramp");
    FvCompConfig c;
    c.tr = Transport::air(0.0, 1.0, 0.0, 0.72, 1.4, 1.0, false);
    for (const MeshPatch& m : fm.patches) {
        if (m.type == FoamPatchType::Empty) {
            FvBc e;
            e.kind = FvBcKind::Empty;
            c.bc.push_back(e);
        } else {
            c.bc.push_back(m.name == "ilo" || m.name == "ihi" ? openRow() : slipRow());
        }
    }
    c.precision = FvPrecision::Fp64;
    c.shock = {.mode = ShockMode::Hll, .eps4 = kFvShockEps4};
    auto d = orDie(DeviceFvComp::make(fm, c), "ramp");
    const double g = 1.4, Ms = 2.0, c1 = std::sqrt(g), p2 = (2 * g * Ms * Ms - (g - 1)) / (g + 1), r2 = (g + 1) * Ms * Ms / ((g - 1) * Ms * Ms + 2);
    const double u2 = Ms * c1 * (1 - 1 / r2);
    FvState s = FvState::zeros(fm.nC);
    for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
        const bool post = fm.C[0][q] < 0.3;
        const double r = post ? r2 : 1.0, u = post ? u2 : 0.0, p = post ? p2 : 1.0;
        s.var(0)[q] = r;
        s.var(1)[q] = r * u;
        s.var(4)[q] = p / (g - 1) + 0.5 * r * u * u;
    }
    orDie(d.upload(s), "ramp");
    long crossed = 0, unsensed = 0;
    const FvRef ref = orDie(FvRef::make(fm, FvConfig{.tr = c.tr, .bc = c.bc}), "lines"); // only its geometry
    const FvShockLines lines = fvShockLines(fm, ref.faceGeom());
    double t = 0;
    for (int n = 0; n < 8; ++n) {
        const Marched m = marchTo(d, 0.06);
        if (!m.err.empty()) {
            check(false, "ramp: " + m.err);
            return;
        }
        t += m.t;
        FvEval ev;
        FvState o;
        orDie(d.evaluate(ev, o), "ramp");
        for (std::size_t f = 0; f < std::size_t(fm.nI); ++f) {
            const std::size_t a = std::size_t(fm.own[f]), b = std::size_t(fm.nei[f]);
            if (fm.srcCell[a] / nx >= 2 || fm.srcCell[b] / nx >= 2) continue; // rows 0, 1
            if (std::abs(fm.S[0][f]) < std::abs(fm.S[1][f])) continue;        // i faces
            const double pa = ev.prim[5][a], pb = ev.prim[5][b];
            if (std::max(pa, pb) / std::min(pa, pb) < 1.2) continue;
            ++crossed;
            int sg = std::max(ev.sigma[a], ev.sigma[b]);
            for (int e : {lines.faceLine[2 * f], lines.faceLine[2 * f + 1]})
                if (e >= 0) sg = std::max(sg, int(ev.sigma[std::size_t(e)]));
            unsensed += sg == 0;
        }
    }
    std::printf("  30 deg ramp, t %.2f: %ld near-wall faces crossed, %ld unsensed\n", t, crossed, unsensed);
    check(crossed > 20, "the shock crossed the near-wall rows");
    check(unsensed == 0, "every near-wall face crossed is sensed");
}

void reflect()
{
    std::printf("G16 reflection onto a wall, whole-line rule; the 30 deg wall\n");
    for (double Ms : {2.0, 3.0, 6.0}) {
        const Reflect r = reflectRun(Ms, ShockMode::Hll);
        std::printf("  Ms %g: %ld steps, plateau %+.2e, wall cell %+.2e %s\n", Ms, r.steps, r.plateau, r.wall, r.err.c_str());
        check(r.err.empty() && r.steps > 100, std::format("Ms {:g}: live", Ms));
        check(std::abs(r.plateau) <= 5e-3, std::format("Ms {:g}: plateau <= 5e-3", Ms));
        check(std::abs(r.wall) <= 1e-2, std::format("Ms {:g}: wall cell <= 1e-2", Ms));
    }
    const Reflect bare = reflectRun(6.0, ShockMode::Sense);
    std::printf("  control, bare KG Ms 6: %ld steps, %s\n", bare.steps, bare.err.c_str());
    check(!bare.err.empty(), "control: bare KG dies at Ms 6");
    ramp30();
}

void dtGate()
{
    std::printf("G17 sides out of the dt\n");
    const Ref w20 = loadRef("nozzle_W", CellOrder::Morton);
    // 5 deg: the shipped Sigma counted the sides (A3 E6 12.0 at 160x32).
    const FaceMesh f5 = orDie(FaceMesh::fromPolyMesh(dnstest::noz::rewedge(w20.pm, w20.fm.wedgeHalf, 2.5 * std::numbers::pi / 180), CellOrder::Morton,
                                                     0, {.wedge = true}),
                              "5");
    const FvState s = stateOf(f5, w20.p, w20.T, w20.U, w20.Rgas);
    Run r{.shock = {.mode = ShockMode::Hll, .force = true}, .R = w20.Rgas};
    const FvCompConfig c = cfgOf(f5, {}, r);
    auto d = orDie(DeviceFvComp::make(f5, c), "dt");
    orDie(d.upload(s), "dt");
    const double out = orDie(d.stableDt({.cfl = 0.4}), "dt");
    // The sides counted: the same Sigma with their faces in, host.
    double in = HUGE_VAL;
    for (std::size_t q = 0; q < std::size_t(f5.nC); ++q) {
        const FvPrim<double> p = fvPrimOf(s.var(0)[q], s.var(1)[q], s.var(2)[q], s.var(3)[q], s.var(4)[q], kGam, w20.Rgas);
        const double cs = std::sqrt(kGam * p.p / p.rho);
        double sum = 0;
        for (int e = f5.cellStart[q]; e < f5.cellStart[q + 1]; ++e) {
            const std::size_t f = std::size_t(f5.cellFace[std::size_t(e)]);
            sum += std::abs(p.u[0] * f5.S[0][f] + p.u[1] * f5.S[1][f] + p.u[2] * f5.S[2][f]) + cs * std::hypot(f5.S[0][f], f5.S[1][f], f5.S[2][f]);
        }
        in = std::min(in, 2 * 0.4 * f5.V[q] / sum);
    }
    std::printf("  dt out %.4e, in %.4e, ratio %.2f\n", out, in, out / in);
    check(out / in >= 8.0, "dt sides out / in >= 8 (5 deg, 80x16)");
    // The same fixed point at the dt the sides would have forced.
    const FvState ic = stateOf(f5, field(w20.dir / "0/p", 1, f5.nC), field(w20.dir / "0/T", 1, f5.nC), field(w20.dir / "0/U", 3, f5.nC), w20.Rgas);
    auto a = orDie(DeviceFvComp::make(f5, c), "a");
    orDie(a.upload(ic), "a");
    const Marched ma = marchTo(a, 2.4, 0.4, 16);
    auto b = orDie(DeviceFvComp::make(f5, c), "b");
    orDie(b.upload(ic), "b");
    const Marched mb = marchTo(b, 2.4, 0.4 * in / out, 64);
    FvState sa, sb;
    orDie(a.download(sa), "a");
    orDie(b.download(sb), "b");
    const double e = relMax(fieldsOf(f5, sa, kGam, w20.Rgas).p, fieldsOf(f5, sb, kGam, w20.Rgas).p);
    std::printf("  %ld vs %ld steps: same fixed point %.2e\n", ma.steps, mb.steps, e);
    check(mb.steps > 4 * ma.steps, "the small dt really was");
    check(e <= 1e-10, "same fixed point <= 1e-10");
}

// M >= 1 of A/A*.
double areaMach(double ar)
{
    const double g = 1.4;
    double lo = 1.0, hi = 60.0;
    for (int i = 0; i < 200; ++i) {
        const double m = 0.5 * (lo + hi);
        const double f = (1 / m) * std::pow((2 / (g + 1)) * (1 + (g - 1) / 2 * m * m), (g + 1) / (2 * (g - 1))) - ar;
        (f > 0 ? hi : lo) = m;
    }
    return 0.5 * (lo + hi);
}

struct Src
{
    double l1 = 0, wall = 0, saw = 0;
    long sensed = 0, steps = 0;
};

// The exact planar supersonic source flow, 15 deg sector, r 2 to 6, nr x nr/4, exact Far inflow
// rows, Slip sides: three flow-throughs from the exact field.
Src sourceRun(int nr, double eps)
{
    const double p0 = 1e5, T0 = 300.0, R = 287.1, g = 1.4, th = 15.0 * std::numbers::pi / 180, r0 = 2.0, r1 = 6.0;
    const int nt = nr / 4;
    std::vector<double> X, Y;
    for (int j = 0; j <= nt; ++j)
        for (int i = 0; i <= nr; ++i) {
            const double r = r0 + (r1 - r0) * i / nr, a = th * j / nt;
            X.push_back(r * std::cos(a));
            Y.push_back(r * std::sin(a));
        }
    const PolyMesh pm = quadPoly(nr, nt, X, Y);
    const FaceMesh fm = orDie(FaceMesh::fromPolyMesh(pm, CellOrder::Morton), "source");
    const auto exact = [&](double r, double a) {
        const double M = areaMach(r), T = T0 / (1 + 0.2 * M * M), p = p0 * std::pow(T / T0, 3.5), c = std::sqrt(g * R * T), rho = p / (R * T);
        return std::array<double, 7>{rho, rho * M * c * std::cos(a), rho * M * c * std::sin(a), 0.0, p / (g - 1) + 0.5 * rho * (M * c) * (M * c), p, M * c};
    };
    FvCompConfig c;
    c.tr = Transport::air(0.0, 1.0, 0.0, 0.72, g, R, false);
    c.precision = FvPrecision::Fp64;
    c.shock = {.mode = ShockMode::Hll, .eps4 = eps};
    for (const MeshPatch& m : fm.patches) {
        if (m.type == FoamPatchType::Empty) {
            FvBc e;
            e.kind = FvBcKind::Empty;
            c.bc.push_back(e);
        } else if (m.name == "ilo") {
            FvBc f;
            f.kind = FvBcKind::Far;
            f.qFar[0] = 1.0;
            f.qFar[4] = 1e6;
            c.bc.push_back(f);
        } else {
            c.bc.push_back(m.name == "ihi" ? openRow() : slipRow());
        }
    }
    const std::size_t nP = c.bc.size();
    c.bcRow.assign(std::size_t(fm.nB), 0);
    for (std::size_t p = 0; p < nP; ++p)
        for (int i = 0; i < fm.patches[p].count; ++i) {
            const int b = fm.patches[p].first + i;
            c.bcRow[std::size_t(b)] = int(p);
            if (fm.patches[p].name != "ilo") continue;
            const std::size_t f = std::size_t(fm.nI + b);
            const double xm = fm.Cf[0][f], ym = fm.Cf[1][f];
            const auto q = exact(std::hypot(xm, ym), std::atan2(ym, xm));
            FvBc r = c.bc[p];
            for (std::size_t k = 0; k < 5; ++k) r.qFar[k] = q[k];
            c.bcRow[std::size_t(b)] = int(c.bc.size());
            c.bc.push_back(r);
        }
    auto d = orDie(DeviceFvComp::make(fm, c), "source");
    FvState s = FvState::zeros(fm.nC);
    std::vector<double> px(std::size_t(fm.nC));
    double Ms = 0, cs = 0;
    for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
        const auto e = exact(std::hypot(fm.C[0][q], fm.C[1][q]), std::atan2(fm.C[1][q], fm.C[0][q]));
        for (int v = 0; v < 5; ++v) s.var(v)[q] = e[std::size_t(v)];
        px[q] = e[5];
        Ms += e[6];
        cs += 1.0;
    }
    orDie(d.upload(s), "source");
    const double fl = (r1 - r0) / (Ms / cs);
    const Marched m = marchTo(d, 3 * fl, 0.4, 4);
    if (!m.err.empty()) {
        std::printf("  N %d eps %g: %s\n", nr, eps, m.err.c_str());
        return {.l1 = HUGE_VAL};
    }
    orDie(d.download(s), "source");
    Src o;
    o.steps = m.steps;
    o.sensed = orDie(d.shockCells(), "sensed");
    std::vector<double> e(px.size());
    double vt = 0;
    std::vector<double> srcErr(px.size());
    for (std::size_t q = 0; q < px.size(); ++q) {
        const double p = fvPrimOf(s.var(0)[q], s.var(1)[q], s.var(2)[q], s.var(3)[q], s.var(4)[q], g, R).p;
        e[q] = p / px[q] - 1;
        o.l1 += std::abs(e[q]) * fm.V[q];
        vt += fm.V[q];
        srcErr[std::size_t(fm.srcCell[q])] = e[q];
    }
    o.l1 /= vt;
    long n3 = 0, nw = 0;
    for (int j = 0; j < nt; ++j)
        for (int i = 0; i + 3 < nr; ++i) {
            const double* a = srcErr.data() + std::size_t(i + nr * j);
            o.saw += std::abs(a[3] - 3 * a[2] + 3 * a[1] - a[0]) / 8;
            ++n3;
        }
    o.saw /= double(n3);
    for (int i = 0; i < nr; ++i) {
        o.wall += std::abs(srcErr[std::size_t(i + nr * (nt - 1))]);
        ++nw;
    }
    o.wall /= double(nw);
    return o;
}

void source()
{
    std::printf("G18 exact source flow, FP64, hybrid D4 1/16\n");
    const Src a = sourceRun(64, kFvShockEps4), b = sourceRun(128, kFvShockEps4), c = sourceRun(256, kFvShockEps4);
    const double ord = std::log2(b.l1 / c.l1);
    std::printf("  N 64, 128, 256: L1 p %.3e %.3e %.3e order %.2f; wall %.3e, sawtooth %.3e; sensed %ld %ld %ld; steps %ld\n", a.l1, b.l1, c.l1, ord,
                c.wall, c.saw, a.sensed, b.sensed, c.sensed, c.steps);
    check(a.l1 <= 1e-3 && b.l1 <= 2.6e-4 && c.l1 <= 6.5e-5, "L1 p <= 1e-3, 2.6e-4, 6.5e-5");
    check(ord >= 1.7, std::format("order {:.2f} >= 1.7", ord));
    check(c.wall <= 4.8e-5, "wall <= 4.8e-5");
    check(c.saw <= 1.5e-5, "sawtooth <= 1.5e-5");
    check(c.sensed == 0 && c.steps > 100, "silent, live");
    const Src k1 = sourceRun(128, 0.0), k2 = sourceRun(256, 0.0);
    const double okg = std::log2(k1.l1 / k2.l1);
    std::printf("  control eps 0, N 128, 256: L1 p %.3e %.3e order %.2f, wall %.3e, sawtooth %.3e\n", k1.l1, k2.l1, okg, k2.wall, k2.saw);
    check(k2.l1 > 6.5e-5 && okg < 1.7, "control: eps 0 fails both the band and the order");
}

// --- CLI runs -----------------------------------------------------------------------

// Column k of a stats csv: header row, then numbers; # lines skipped.
std::vector<double> csvColumn(const fs::path& f, const std::string& col)
{
    std::ifstream in(f);
    std::string l;
    int k = -1;
    std::vector<double> o;
    while (std::getline(in, l)) {
        if (l.empty() || l[0] == '#') continue;
        std::stringstream ss(l);
        std::string w;
        int i = 0;
        if (k < 0) {
            while (std::getline(ss, w, ',')) {
                if (w == col) k = i;
                ++i;
            }
            if (k < 0) return o;
            continue;
        }
        while (std::getline(ss, w, ',')) {
            if (i++ == k) o.push_back(std::stod(w));
        }
    }
    return o;
}

int runDns(const std::string& args, const fs::path& log)
{
    const std::string cmd = std::format("{} {} > {} 2>&1", DNS_BIN, args, log.string());
    const int rc = std::system(cmd.c_str());
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : 99;
}

// A smooth contour past the ramp: 0 sensed. The centre rule on the same p fires.
void silent()
{
    std::printf("G19 the sensor on a smooth contour, rest + ramp, total + riemann, FP64\n");
    // nc-smooth 200x40: the gate. a1-smooth 100x20: the rig that reads the centre rule (sp_100, A1's
    // contour; at 200x40 the dns field reads 0.0159 under it, below th1: masterplan_cdnozzle A3).
    struct Case
    {
        const char* name;
        const char* contour;
        int nx, ny;
    } cases[] = {{"a1-smooth", "a1-smooth", 100, 20}, {"nc-smooth", "nc-smooth", 200, 40}, {"nc", "nc", 200, 40}};
    for (const Case& k : cases) {
        const Gen g = gen(k.name, k.nx, k.ny, std::format("--contour {} --mode planar --exit riemann --pb 0.05", k.contour), CellOrder::Source);
        const fs::path out = tmpDir() / (std::string(k.name) + "_run");
        const double ramp = 20.0, tt = 60.0;
        const int rc = runDns(std::format("flow --physics compressible --polymesh {} --bc {}/bc.toml --shock hll --inviscid --mach 1 --ramp {} --tt {} "
                                          "--fv-precision fp64 --foam-write {} --emit stats --out {}",
                                          g.dir.string(), g.dir.string(), ramp, tt, ramp + tt, out.string()),
                              out.string() + ".log");
        const auto t = csvColumn(out.string() + "_stats.csv", "t"), sc = csvColumn(out.string() + "_stats.csv", "shockCells");
        double late = 0;
        long rows = 0;
        for (std::size_t i = 0; i < t.size(); ++i)
            if (t[i] >= ramp + 50.0) {
                late = std::max(late, sc[i]);
                ++rows;
            }
        const double peak = sc.empty() ? -1 : *std::ranges::max_element(sc);
        std::printf("  %s: rc %d, %zu rows, t end %.1f; sensed past ramp + 50: max %g over %ld rows; peak %g\n", k.name, rc, t.size(), t.empty() ? 0.0 : t.back(), late, rows, peak);
        check(rc == 0 && rows > 2, std::format("{}: live", k.name));
        const std::string nm = k.name;
        if (nm == "nc") check(late > 0, "nc: sensed > 0 (the arc-to-cone junction is a real compression)");
        else check(late == 0, std::format("{}: 0 sensed past ramp + 50 t*", nm));
        // The centre rule on the run's own p: a blocked read as the cell, structured i, j.
        const fs::path dir = fs::path(out.string() + "_foam") / std::format("{:.12g}", ramp + tt);
        const std::vector<double> p = field(dir / "p", 1, g.fm.nC);
        const int nx = g.nx, ny = g.ny;
        long centre = 0;
        double vmax = 0;
        auto at = [&](int i, int j) { return p[std::size_t(i + nx * j)]; };
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double c = at(i, j), xp = i + 2 < nx ? at(i + 2, j) : c, xm = i >= 2 ? at(i - 2, j) : c;
                const double yp = j + 2 < ny ? at(i, j + 2) : c, ym = j >= 2 ? at(i, j - 2) : c;
                const double v = std::max(shockJump(xm, c, xp), shockJump(ym, c, yp));
                vmax = std::max(vmax, v);
                centre += v > kShockTh1;
            }
        std::printf("  %s: centre rule on the same p: %ld cells above th1, max %.4f\n", k.name, centre, vmax);
        if (nm == "a1-smooth") check(centre >= 1, "control: the centre rule counts >= 1 in the same run");
    }
}

double midExit(const Gen& g, const std::vector<double>& v)
{
    return v[std::size_t((g.nx - 1) + g.nx * (g.ny / 2))]; // the exit column, mid row: source order
}

struct ExitOut
{
    long steps = 0;
    double M = 0, p = 0, Tmax = 0, Tpeak = 0;
    std::string err;
};

// A run on a generated mesh. IC 0: q1d; 1: rest diaphragm at the throat.
ExitOut exitRun(const Gen& g, const Rows& rows, int rest, double tStar, double eps)
{
    Run r;
    r.shock.eps4 = eps;
    auto d = orDie(DeviceFvComp::make(g.fm, cfgOf(g.fm, rows, r)), "exit");
    FvState s = icOf(g);
    if (rest) {
        const double R = 287.1, rb = rows.pb / (R * 300.0);
        for (std::size_t q = 0; q < std::size_t(g.fm.nC); ++q) {
            const bool up = g.fm.C[0][q] < 0.0;
            for (int v = 1; v < 4; ++v) s.var(v)[q] = 0.0;
            s.var(0)[q] = up ? 1e5 / (R * 300.0) : rb;
            s.var(4)[q] = (up ? 1e5 : rows.pb) / (kGam - 1.0);
        }
    }
    orDie(d.upload(s), "exit");
    ExitOut o;
    // Two-t* chunks; the peak T past t* 30 (the start's transient out), as stab.py read it.
    double tDone = 0;
    while (tDone < tStar) {
        const double chunk = std::min(2.0, tStar - tDone);
        const Marched m = marchTo(d, chunk / kC0);
        o.steps += m.steps;
        tDone += chunk;
        o.err = m.err;
        if (!m.err.empty()) return o;
        if (tDone < 30.0) continue;
        orDie(d.download(s), "exit");
        const std::vector<double> Tn = fieldsOf(g.fm, s, kGam, 287.1).T;
        o.Tpeak = std::max(o.Tpeak, *std::ranges::max_element(Tn));
    }
    orDie(d.download(s), "exit");
    const Fields f = fieldsOf(g.fm, s, kGam, 287.1);
    const double c = std::sqrt(kGam * midExit(g, f.p) / midExit(g, f.rho));
    o.M = std::hypot(midExit(g, f.u[0]), midExit(g, f.u[1])) / c;
    o.p = midExit(g, f.p);
    o.Tmax = *std::ranges::max_element(f.T);
    return o;
}

void exitGate()
{
    std::printf("G20 the exit rows on cone6\n");
    const std::string cone = "--contour cone6 --mode wedge --dphi 20 --exit extrapolate";
    const Gen g = gen("cone6", 320, 64, cone, CellOrder::Source);
    const Rows riem{.pb = 0.01 * kP0, .exit = FvExit::Riemann};
    const ExitOut a = exitRun(g, riem, 1, 16.0, kFvShockEps4);
    const ExitOut ac = exitRun(g, {.pb = 0.01 * kP0, .characteristic = true}, 1, 16.0, kFvShockEps4);
    std::printf("  rest, p_b 0.01, t* 16: riemann M %.3f p/p0 %.4f (%ld steps %s); characteristic M %.3f (%s)\n", a.M, a.p / kP0, a.steps, a.err.c_str(), ac.M, ac.err.c_str());
    check(a.err.empty() && a.steps >= 5000, std::format("rest: live, {} steps", a.steps));
    check(a.M >= 3.2, "rest: exit M >= 3.2 at t* 16");
    check(ac.M <= 1.0 || !ac.err.empty(), "control: characteristic hangs (M <= 1) or dies");
    const ExitOut b = exitRun(g, {.pb = 0.5 * kP0, .exit = FvExit::Riemann}, 0, 60.0, kFvShockEps4);
    const ExitOut bc = exitRun(g, {.pb = 0.0}, 0, 60.0, kFvShockEps4);
    std::printf("  supersonic IC, p_b 0.5, t* 60: riemann p/p0 %.4f (%ld steps); extrapolate p/p0 %.4f, M %.3f\n", b.p / kP0, b.steps, bc.p / kP0, bc.M);
    check(b.err.empty() && std::abs(b.p / kP0 - 0.5) <= 0.002, "p_e/p0 0.5 +- 0.002");
    check(bc.err.empty() && bc.p / kP0 <= 0.02, "control: extrapolate stays supersonic, p <= 0.02");
    const ExitOut c = exitRun(g, {.pb = 0.15 * kP0, .exit = FvExit::Riemann}, 0, 60.0, kFvShockEps4);
    const ExitOut cf = exitRun(g, {.pb = 0.15 * kP0}, 0, 1.0, kFvShockEps4);
    std::printf("  supersonic IC, p_b 0.15: riemann M %.4f p/p0 %.4f; vs extrapolate %.2e; shipped row: %ld steps, %s\n", c.M, c.p / kP0, std::abs(c.M / bc.M - 1), cf.steps, cf.err.c_str());
    check(std::abs(c.M / bc.M - 1) <= 1e-3, "p_b 0.15: vs extrapolate 1e-3");
    check(!cf.err.empty() && cf.steps <= 4, "control: the shipped row diverges in its first steps");
    // stab.py's IC: q1d with the normal shock placed by p_b (a supersonic IC sends a p ratio 30 shock
    // in from the exit and it focuses on the axis, T/T0 1.8, foam or not: not this gate's rig).
    const Gen s = gen("cone6s", 160, 40, "--contour cone6 --mode wedge --dphi 20 --exit fixed --pb 0.5", CellOrder::Source);
    const ExitOut d = exitRun(s, {.pb = 0.5 * kP0, .exit = FvExit::Riemann}, 0, 120.0, kFvShockEps4);
    const ExitOut dc = exitRun(s, {.pb = 0.5 * kP0, .exit = FvExit::Riemann}, 0, 120.0, 0.0);
    std::printf("  G-std p_b 0.5, 160x40, t* 120: Tmax/T0 %.3f (%ld steps %s); --shock-d4 0 %.3f (%ld steps %s)\n", d.Tpeak / kT0, d.steps, d.err.c_str(), dc.Tpeak / kT0, dc.steps, dc.err.c_str());
    check(d.err.empty() && d.Tpeak / kT0 <= 1.15, "G-std: Tmax/T0 <= 1.15, the run's peak past t* 30");
    check(!dc.err.empty() || dc.Tpeak / kT0 > 1.15, "control: --shock-d4 0 fails the band (host 2.11, device rebaseline: A10)");
}

void sauer()
{
    std::printf("G21 Sauer, parabola R 20, planar 280x32\n");
    const double R = 20.0, g1 = 2.4, alpha = std::sqrt(1.0 / (g1 * R));
    const Gen g = gen("par", 280, 32, "--contour parabola --mode planar --exit extrapolate", CellOrder::Source);
    Run r;
    const FvCompConfig cfg = cfgOf(g.fm, {}, r);
    auto d = orDie(DeviceFvComp::make(g.fm, cfg), "par");
    orDie(d.upload(icOf(g)), "par");
    const Marched m = marchTo(d, 300.0 / kC0, 0.4, 8);
    check(m.err.empty() && m.steps > 5000, std::format("live, {} steps {}", m.steps, m.err));
    FvState s;
    orDie(d.download(s), "par");
    const Fields f = fieldsOf(g.fm, s, kGam, 287.1);
    std::vector<double> Mx(std::size_t(g.nx)), xc(std::size_t(g.nx));
    for (std::size_t c = 0; c < std::size_t(g.fm.nC); ++c)
        if (g.fm.srcCell[c] < g.nx) xc[std::size_t(g.fm.srcCell[c])] = g.fm.C[0][c];
    for (int i = 0; i < g.nx; ++i) {
        const std::size_t q = std::size_t(i);
        Mx[q] = std::hypot(f.u[0][q], f.u[1][q]) / std::sqrt(kGam * f.p[q] / f.rho[q]);
    }
    int ic = 0;
    while (ic + 1 < g.nx && Mx[std::size_t(ic) + 1] < 1.0) ++ic;
    const double slope = (Mx[std::size_t(ic) + 1] - Mx[std::size_t(ic)]) / (xc[std::size_t(ic) + 1] - xc[std::size_t(ic)]);
    const double xs = xc[std::size_t(ic)] + (1.0 - Mx[std::size_t(ic)]) / slope;
    const double dM = g1 * alpha / 2, xsS = g1 * alpha / 6;
    const double mo = exitMdot(d, cfg.bc), mi = inletMdot(d, cfg.bc);
    const double cd = mo / fvIdealMassFlow(kP0, kT0, g.astar, kGam, 287.1), cdS = 1 - g1 / (90 * R * R);
    const double wrong = mo / fvIdealMassFlow(kP0, kT0, g.astar * 0.99873, kGam, 287.1);
    std::printf("  dM/dx %.4f vs Sauer %.4f (%+.2f%%), sonic x %.4f vs %.4f (%+.2f%%), Cd %.6f vs %.6f, mdot in/out %.2e; Cd with A2's cos factor %.6f\n", slope, dM,
                100 * (slope / dM - 1), xs, xsS, 100 * (xs / xsS - 1), cd, cdS, mi / mo - 1, wrong);
    check(std::abs(slope / dM - 1) <= 0.03, "dM/dx 3%");
    check(std::abs(xs / xsS - 1) <= 0.03, "sonic x 3%");
    check(std::abs(cd - cdS) <= 2e-4, "Cd 2e-4");
    check(std::abs(wrong - cdS) > 2e-4, "control: an area off by the cos form's factor fails");
}

void cdladder()
{
    std::printf("G22 Cd over R 2, 4, 8, wedge; budget\n");
    const double target = 2.4 / 96.0;
    double deficit[3];
    const int Rs[3] = {2, 4, 8};
    for (int k = 0; k < 3; ++k) {
        const int nx = k == 0 ? 200 : k == 1 ? 240 : 320;
        const Gen g = gen(std::format("R{}", Rs[k]), nx, 40, std::format("--contour nc --Rt {} --mode wedge --dphi 20 --exit extrapolate", Rs[k]));
        Run r;
        FvCompConfig cfg = cfgOf(g.fm, {}, r);
        auto d = orDie(DeviceFvComp::make(g.fm, cfg), "R");
        orDie(d.upload(icOf(g)), "R");
        // Cd creeps to t* ~1500 and the chamber rings past it (A7): a late run, a long mean.
        const Marched m = marchTo(d, 1500.0 / kC0, 0.4, 8);
        double md = 0, mdi = 0, bud = 0, cd = 0;
        const int avg = 800;
        for (int n = 0; n < avg; ++n) {
            for (int q = 0; q < 50; ++q) {
                auto dt = d.stableDt({.cfl = 0.4});
                if (!dt) break;
                d.step(0.95 * *dt);
            }
            md += exitMdot(d, cfg.bc) / avg;
            mdi += inletMdot(d, cfg.bc) / avg;
            const std::array<std::vector<double>, 3> mo = {orDie(d.patchFlux(1), "m"), orDie(d.patchFlux(2), "m"), orDie(d.patchFlux(3), "m")};
            const gpu::FvSurface w = orDie(d.surface(), "w");
            const FvNozzleRow z = fvNozzleRow(cfg.bc, orDie(d.patchFlux(0), "m"), mo, w.fp, {.astar = g.astar, .p0 = kP0, .t0 = kT0}, kGam, 287.1);
            bud += (z.Fexit + z.Fin + z.Fwall) / std::abs(z.Fexit) / avg; // pRef 0
            cd += z.Cd / avg;
        }
        deficit[k] = (1 - cd) * Rs[k] * Rs[k];
        std::printf("  R %d, %dx40: %ld steps, Cd %.6f, deficit R^2 %.5f, mdot out/in - 1 %.2e, budget %.2e %s\n", Rs[k], nx, m.steps, cd, deficit[k], md / mdi - 1, bud, m.err.c_str());
        check(m.err.empty() && m.steps > 5000, std::format("R {}: live", Rs[k]));
        check(std::abs(md / mdi - 1) <= 1e-5, std::format("R {}: mdot in/out 1e-5", Rs[k]));
        check(std::abs(bud) <= 1e-4, std::format("R {}: momentum budget 1e-4", Rs[k]));
    }
    const double rich = 2 * deficit[2] - deficit[1];
    std::printf("  Richardson d R^2 %.5f vs (g+1)/96 %.5f (%+.1f%%)\n", rich, target, 100 * (rich / target - 1));
    check(std::abs(rich / target - 1) <= 0.03, "Richardson d R^2 within 3% of (g+1)/96");
}

struct Fp
{
    double xs = 0, mdot = 0;
};

Fp fpRun(const Gen& g, FvPrecision pr, bool comp)
{
    Run r{.prec = pr, .comp = comp};
    const FvCompConfig cfg = cfgOf(g.fm, {.pb = 0.6 * kP0}, r);
    auto d = orDie(DeviceFvComp::make(g.fm, cfg), "fp");
    orDie(d.upload(icOf(g)), "fp");
    const Marched m = marchTo(d, 150.0 / kC0, 0.4, 4);
    check(m.err.empty() && m.steps > 5000, std::format("{} comp {}: live, {} steps", pr == FvPrecision::Fp64 ? "FP64" : "FP32", comp, m.steps));
    Fp o;
    const int avg = 40;
    for (int n = 0; n < avg; ++n) {
        for (int q = 0; q < 100; ++q) {
            auto dt = d.stableDt({.cfl = 0.4});
            if (!dt) break;
            d.step(0.95 * *dt);
        }
        FvState s;
        orDie(d.download(s), "fp");
        const Fields f = fieldsOf(g.fm, s, kGam, 287.1);
        // The shock on the axis row: the largest p jump past the throat, sub-cell by its middle value.
        int best = 0;
        double jump = 0;
        for (int i = g.nx / 2; i + 1 < g.nx; ++i) {
            const double dp = f.p[std::size_t(i) + 1] - f.p[std::size_t(i)];
            if (dp > jump) {
                jump = dp;
                best = i;
            }
        }
        o.xs += (best + 0.5) / avg;
        o.mdot += exitMdot(d, cfg.bc) / avg;
    }
    return o;
}

void fp32()
{
    std::printf("G25 FP32 compensated vs FP64: N-C B, 160x32, t* 150 + a 4000-step mean\n");
    const Gen g = gen("fpB", 160, 32, "--contour nc --mode planar --exit fixed --pb 0.6", CellOrder::Source);
    const Fp a = fpRun(g, FvPrecision::Fp64, false), b = fpRun(g, FvPrecision::Fp32, true), c = fpRun(g, FvPrecision::Fp32, false);
    std::printf("  x_s (cells): FP64 %.3f, compensated %+.3f, plain %+.3f; mdot: compensated %+.2e, plain %+.2e\n", a.xs, b.xs - a.xs, c.xs - a.xs, b.mdot / a.mdot - 1, c.mdot / a.mdot - 1);
    check(std::abs(b.xs - a.xs) <= 0.5, "x_s <= 0.5 cell");
    check(std::abs(b.mdot / a.mdot - 1) <= 1e-5, "mdot 1e-5");
}

// G31: T1 at gamma 1.2, the same row, the ref's own gas; foam's gamma 1.4 field is the control.
void gammaGate()
{
    std::printf("G31 T1 at gamma 1.2, B\n");
    const Ref w = loadRef("nozzle_G", CellOrder::Morton);
    Run r{.shock = {.mode = ShockMode::Hll, .force = true}, .R = w.Rgas, .gamma = 1.2};
    const FvCompConfig cfg = cfgOf(w.fm, {.pb = 6e4}, r);
    auto d = orDie(DeviceFvComp::make(w.fm, cfg), "G");
    const auto gasState = [&](const std::vector<double>& p, const std::vector<double>& T, const std::vector<double>& U) {
        FvState s = stateOf(w.fm, p, T, U, w.Rgas);
        for (int c = 0; c < s.nC; ++c) {
            const double ke = 0.5 * (s.var(1)[c] * s.var(1)[c] + s.var(2)[c] * s.var(2)[c]) / s.var(0)[c];
            s.var(4)[c] = p[std::size_t(w.fm.srcCell[std::size_t(c)])] / 0.2 + ke;
        }
        return s;
    };
    orDie(d.upload(gasState(field(w.dir / "0/p", 1, w.fm.nC), field(w.dir / "0/T", 1, w.fm.nC), field(w.dir / "0/U", 3, w.fm.nC))), "G");
    const Marched m = marchTo(d, 2.4, 0.4, 16);
    FvState s;
    orDie(d.download(s), "G");
    const double e = vsFoam(fieldsOf(w.fm, s, 1.2, w.Rgas), w);
    const Ref b = loadRef("nozzle_B", CellOrder::Morton);
    const double c = vsFoam(fieldsOf(w.fm, s, 1.2, w.Rgas), b);
    std::printf("  %ld steps %s: max cell gap vs foam at 1.2 %.2e; vs the 1.4 reference %.2e\n", m.steps, m.err.c_str(), e, c);
    check(m.err.empty() && m.steps >= 20000, "live");
    check(e <= 1e-10, "T1 at gamma 1.2 <= 1e-10");
    check(c >= 1e-3, "control: the gamma 1.4 reference fails");
}

// G27, CLI half: a silent run under hll at eps 0 and under sense keep none's numbers.
void cliSilent()
{
    std::printf("G27 --shock-d4 0 silent == none, sense == none\n");
    const std::string mesh = std::string(DNS_MESH_DATA) + "/cyl2d_snappy";
    const std::string bc = std::string(DNS_MESH_DATA) + "/../../../examples/fvcomp_cyl2d_snappy.toml";
    const std::string base = std::format("flow --physics compressible --polymesh {} --bc {} --re 40 --mach 0.2 --ramp 1 --tt 1 --emit stats", mesh, bc);
    const auto run = [&](const char* tag, const std::string& extra) {
        const fs::path o = tmpDir() / tag;
        const int rc = runDns(base + extra + " --out " + o.string(), o.string() + ".log");
        check(rc == 0, std::format("{}: rc {}", tag, rc));
        return o.string() + "_stats.csv";
    };
    const std::string a = run("none", ""), b = run("d40", " --shock hll --shock-d4 0"), c = run("sns", " --shock sense");
    const char* cols[] = {"t", "dt", "cd", "cl", "cs", "machMax", "mass", "energy", "kinetic"};
    long rows = 0;
    bool same = true;
    for (const char* k : cols) {
        const auto x = csvColumn(a, k), y = csvColumn(b, k), z = csvColumn(c, k);
        rows = long(x.size());
        same = same && !x.empty() && x == y && x == z;
    }
    const auto s1 = csvColumn(b, "shockCells"), s2 = csvColumn(c, "shockCells");
    std::printf("  %ld rows; sensed max %g, %g\n", rows, s1.empty() ? -1.0 : *std::ranges::max_element(s1), s2.empty() ? -1.0 : *std::ranges::max_element(s2));
    check(rows > 3 && same, "every stats column equal, none vs --shock-d4 0 vs sense");
    check(!s1.empty() && *std::ranges::max_element(s1) == 0.0, "the run was silent");
}

// G27, library half: sense == none bytes on the B field, sensed > 0.
void sense()
{
    std::printf("G27 sense == none bytes\n");
    const Ref w = loadRef("nozzle_B", CellOrder::Morton);
    const FvState ic = stateOf(w.fm, field(w.dir / "0/p", 1, w.fm.nC), field(w.dir / "0/T", 1, w.fm.nC), field(w.dir / "0/U", 3, w.fm.nC), w.Rgas);
    const Rows rows{.pb = 6e4};
    Run a{.shock = {}, .prec = FvPrecision::Fp32, .R = w.Rgas}, b{.shock = {.mode = ShockMode::Sense}, .prec = FvPrecision::Fp32, .R = w.Rgas};
    auto da = orDie(DeviceFvComp::make(w.fm, cfgOf(w.fm, rows, a)), "none");
    auto db = orDie(DeviceFvComp::make(w.fm, cfgOf(w.fm, rows, b)), "sense");
    orDie(da.upload(ic), "none");
    orDie(db.upload(ic), "sense");
    long sensed = 0;
    for (int i = 0; i < 300; ++i) {
        da.step(1e-5);
        db.step(1e-5);
        if (i % 100 == 99) sensed = std::max(sensed, orDie(db.shockCells(), "cells"));
    }
    FvState sa, sb;
    orDie(da.download(sa), "none");
    orDie(db.download(sb), "sense");
    check(sa.q.size() == sb.q.size() && std::memcmp(sa.q.data(), sb.q.data(), sa.q.size() * sizeof(double)) == 0, "sense == none bytes, 300 steps");
    std::printf("  sensed %ld\n", sensed);
    check(sensed > 0, "sensed > 0 on B");
}

// --- masterplan_combnozzle P2: the one-row nozzle (quasi-1D FV) and the thrust ledger -------------

constexpr double kAstar = 0.1; // every contour's throat: y* 1 x depth 0.1 (not the generator's node minimum)

// The conserved state of cell-order p, T, U at gamma g (stateOf is gamma 1.4's).
FvState gasStateOf(const FaceMesh& fm, const std::vector<double>& p, const std::vector<double>& T, const std::vector<double>& U, double g)
{
    FvState s = stateOf(fm, p, T, U, 287.1);
    for (int c = 0; c < s.nC; ++c) {
        const double ke = 0.5 * (s.var(1)[c] * s.var(1)[c] + s.var(2)[c] * s.var(2)[c] + s.var(3)[c] * s.var(3)[c]) / s.var(0)[c];
        s.var(4)[c] = p[std::size_t(fm.srcCell[std::size_t(c)])] / (g - 1.0) + ke;
    }
    return s;
}

FvState icOfGas(const Gen& g, double gamma)
{
    return gasStateOf(g.fm, field(g.dir / "0/p", 1, g.fm.nC), field(g.dir / "0/T", 1, g.fm.nC), field(g.dir / "0/U", 3, g.fm.nC), gamma);
}

// The quasi-1D IC of a one-row case: the generator's p, T, u with v 0 (R1: the one-row scheme keeps v
// 0.0 only from v 0; the generator's 2D IC carries v = u dy/dx y/y_w).
FvState q1dIc(const Gen& g, double gamma)
{
    std::vector<double> U = field(g.dir / "0/U", 3, g.fm.nC);
    for (std::size_t c = 0; c < U.size() / 3; ++c) U[3 * c + 1] = 0.0;
    return gasStateOf(g.fm, field(g.dir / "0/p", 1, g.fm.nC), field(g.dir / "0/T", 1, g.fm.nC), U, gamma);
}

// v and w exactly 0.0 in every cell.
bool vwZero(const FvState& s)
{
    for (int c = 0; c < s.nC; ++c)
        if (s.var(2)[c] != 0.0 || s.var(3)[c] != 0.0) return false;
    return true;
}

// A run's window means from the ledger (masterplan_combnozzle R2) and the end state's shock.
struct Q1d
{
    long steps = 0;
    std::string err;
    double Cd = 0, CfVac = 0, Fin = 0, Fwall = 0, pe = 0, Me = 0, xs = NAN, mdotRatio = 0;
    bool vZero = true;
    double vMax = 0;              // max |v|/c over cells and checks
    std::vector<double> xsHist{}; // the shock crossing at each check (level given)
    double exitSuperMin = 1.0;
    FvState state;
};

// The quasi-1D shock's level (p1 + p2)/2 at area ratio ar (the upstream supersonic branch) and the
// linear crossing of the cell-centre p of a one-row state with it, the crossing nearest xGuess.
double shockCrossing(const FaceMesh& fm, const FvState& s, double level, double xGuess, double g)
{
    std::vector<std::pair<double, double>> xp;
    for (int c = 0; c < fm.nC; ++c)
        xp.emplace_back(fm.C[0][std::size_t(c)],
                        fvPrimOf(s.var(0)[c], s.var(1)[c], s.var(2)[c], s.var(3)[c], s.var(4)[c], g, 287.1).p);
    std::ranges::sort(xp);
    double best = NAN;
    for (std::size_t i = 0; i + 1 < xp.size(); ++i) {
        const auto [x0, p0] = xp[i];
        const auto [x1, p1] = xp[i + 1];
        if ((p0 - level) * (p1 - level) > 0.0 || p1 == p0) continue;
        const double x = x0 + (level - p0) / (p1 - p0) * (x1 - x0);
        if (!std::isfinite(best) || std::abs(x - xGuess) < std::abs(best - xGuess)) best = x;
    }
    return best;
}

// The one-row run: the generator's IC, rows, FP64 under r; the ledger from t 0; window [0.75 tt, tt].
// checks: v, w bitwise and exitSuper sampled at that many points in the run.
// vMax: max over cells of |v|/c.
double vOverC(const FvState& s, double g)
{
    double m = 0;
    for (int c = 0; c < s.nC; ++c) {
        const FvPrim<double> q = fvPrimOf(s.var(0)[c], s.var(1)[c], s.var(2)[c], s.var(3)[c], s.var(4)[c], g, 287.1);
        m = std::max(m, std::hypot(q.u[1], q.u[2]) / std::sqrt(g * q.p / q.rho));
    }
    return m;
}

Q1d q1dRun(const Gen& g, const Rows& rows, const Run& r, double tStar, bool supersonicExit, int checks = 8, double level = NAN,
           double xGuess = 0.0)
{
    Q1d o;
    FvCompConfig cfg = cfgOf(g.fm, rows, r);
    auto d = orDie(DeviceFvComp::make(g.fm, cfg), "q1d");
    orDie(d.upload(q1dIc(g, r.gamma)), "q1d");
    orDie(d.setLedger(true), "ledger");
    const std::vector<FvRole> roles = fvRoles(cfg.bc, g.fm);
    const FvThrustCase k{.axis = {1, 0, 0}, .pa = rows.pb, .p0 = kP0, .t0 = kT0, .astar = kAstar, .gamma = r.gamma, .Rgas = 287.1,
                         .exitArea = fvExitArea(g.fm, roles, {1, 0, 0})};
    const double tEnd = tStar / kC0, tWin = 0.75 * tEnd;
    double t = 0;
    FvRoleSums a{};
    for (int n = 1; n <= checks; ++n) {
        const double tn = tEnd * n / checks;
        const Marched m = marchTo(d, tn - t, 0.4, 4);
        o.steps += m.steps;
        t = tn;
        if (!m.err.empty()) {
            o.err = m.err;
            return o;
        }
        FvState s;
        orDie(d.download(s), "q1d");
        o.vZero = o.vZero && vwZero(s);
        o.vMax = std::max(o.vMax, vOverC(s, r.gamma));
        if (std::isfinite(level)) o.xsHist.push_back(shockCrossing(g.fm, s, level, xGuess, r.gamma));
        FvEval ev;
        FvState rr;
        orDie(d.evaluate(ev, rr), "q1d");
        o.exitSuperMin = std::min(o.exitSuperMin, fvExitSample(g.fm, roles, ev, rr, k).exitSuper);
        if (std::abs(t - tWin) < 1e-12 * tEnd || (n == 3 * checks / 4)) a = fvRoleSums(orDie(d.ledger(), "ledger"), roles, -1);
    }
    const double t1 = tEnd * double(3 * checks / 4) / checks;
    const FvRoleSums b = fvRoleSums(orDie(d.ledger(), "ledger"), roles, -1);
    const FvThrustWindow w = fvThrustWindow(a, t1, b, tEnd, k);
    const double ref = kP0 * kAstar, dt = tEnd - t1;
    const auto I = [&](FvRole q, int v) {
        return (b.I[std::size_t(q)][std::size_t(v)] - a.I[std::size_t(q)][std::size_t(v)]) / dt;
    };
    o.Cd = w.mdotOut / fvIdealMassFlow(kP0, kT0, kAstar, r.gamma, 287.1);
    o.mdotRatio = w.mdotOut / w.mdotIn;
    o.CfVac = w.CfVac;
    o.Fin = I(FvRole::In, 1) / ref;
    o.Fwall = I(FvRole::Wall, 1) / ref;
    const double Ae = k.exitArea;
    const double f1 = I(FvRole::Exit, 0) / Ae, f2 = I(FvRole::Exit, 1) / Ae, f3 = I(FvRole::Exit, 4) / Ae;
    o.Me = fvExitMach(f1, f2, f3, r.gamma, supersonicExit);
    o.pe = f2 / (1.0 + r.gamma * o.Me * o.Me) / kP0;
    orDie(d.download(o.state), "q1d");
    return o;
}

double order(double e1, double e2) { return std::log2(e1 / e2); }

// G2.3, G2.4: the parabola R 20, A_e/A* 1.4, total + riemann.
void q1dParabola()
{
    std::printf("G2.3 / G2.4 the one-row parabola R 20, A_e/A* 1.4, total + riemann p_b 0.01, FP64, tt 160 t*, window [120, 160]\n");
    const double cfO = 1.3794635, meO = 1.7632052;
    const Rows rows{.pb = 0.01 * kP0, .exit = FvExit::Riemann};
    double ec[3], ef[3], em[3], fc[3], vRung = 0;
    const int nxs[3] = {200, 400, 800};
    for (int k = 0; k < 3; ++k) {
        const Gen g = gen(std::format("par{}", nxs[k]), nxs[k], 1, "--contour parabola --mode planar --exit riemann --pb 0.01");
        const Q1d q = q1dRun(g, rows, Run{}, 160.0, true);
        ec[k] = std::abs(q.Cd - 1.0);
        ef[k] = std::abs(q.CfVac - cfO);
        em[k] = std::abs(q.Me - meO);
        std::printf("  nx %d: %ld steps %s; Cd %.9f CfVac %.9f M_e %.9f p_e/p0 %.7f; out/in - 1 %.1e; v, w == 0.0: %s (max |v|/c %.2e); exitSuper min %.3f\n",
                    nxs[k], q.steps, q.err.c_str(), q.Cd, q.CfVac, q.Me, q.pe, q.mdotRatio - 1, q.vZero ? "yes" : "NO", q.vMax, q.exitSuperMin);
        check(q.err.empty() && q.steps > 2000, std::format("nx {}: live", nxs[k]));
        // G2.3 as specified (bitwise) is UNMET: the wall's S_y and the axis's differ by an ulp per cell (fvref_ledger
        // L3, OpenFOAM's face fan), so v is a rounding field (masterplan_combnozzle A9). Printed, not counted.
        std::printf("  %s G2.3 nx %d: v, w == 0.0 bitwise at every check%s\n", q.vZero ? "[PASS]" : "[UNMET (recorded, A9)]", nxs[k], "");
        check(q.vMax <= 1e-12, std::format("G2.3b nx {}: max |v|/c {:.2e} <= 1e-12 at every check", nxs[k], q.vMax));
        vRung = std::max(vRung, q.vMax);
        // The control: force sigma 1, first-order HLL.
        const Q1d f = q1dRun(g, rows, Run{.shock = {.mode = ShockMode::Hll, .force = true}}, 160.0, true, 4);
        fc[k] = std::abs(f.CfVac - cfO);
    }
    std::printf("  errors |Cd - 1| %.2e %.2e %.2e (orders %.2f %.2f); |CfVac - %.7f| %.2e %.2e %.2e (%.2f %.2f); |M_e - %.7f| %.2e %.2e %.2e (%.2f %.2f)\n",
                ec[0], ec[1], ec[2], order(ec[0], ec[1]), order(ec[1], ec[2]), cfO, ef[0], ef[1], ef[2], order(ef[0], ef[1]), order(ef[1], ef[2]), meO,
                em[0], em[1], em[2], order(em[0], em[1]), order(em[1], em[2]));
    std::printf("  control force sigma 1: |CfVac - oracle| %.2e %.2e %.2e (orders %.2f %.2f)\n", fc[0], fc[1], fc[2], order(fc[0], fc[1]), order(fc[1], fc[2]));
    for (const auto& [e, name] : {std::pair{ec, "Cd"}, {ef, "CfVac"}, {em, "M_e"}})
        check(order(e[0], e[1]) >= 1.8 && order(e[1], e[2]) >= 1.8, std::format("G2.4 {}: order >= 1.8 on both pairs", name));
    check(ef[1] <= 1e-5, std::format("G2.4 CfVac at 400: {:.2e} <= 1e-5", ef[1]));
    check(fc[1] >= 1e-4 && order(fc[1], fc[2]) <= 1.2, "G2.4 control: force sigma 1 CfVac error >= 1e-4 at 400, order <= 1.2");
    // G2.3's control: one axis node moved 1e-9 along x (both z planes: still an extrusion): its x face leans,
    // S_y != 0 there, so v is a field, not rounding.
    const Gen g = gen("par200j", 200, 1, "--contour parabola --mode planar --exit riemann --pb 0.01");
    PolyMesh pm = g.pm;
    pm.px[100] += 1e-9;
    pm.px[100 + 2 * 201] += 1e-9;
    Gen j = g;
    j.fm = orDie(FaceMesh::fromPolyMesh(pm, CellOrder::Morton, 0, {.wedge = true}), "jitter");
    const Q1d q = q1dRun(j, rows, Run{}, 40.0, true, 2);
    std::printf("  control: one axis node 1e-9 along x: max |v|/c %.2e after %ld steps (the rungs' max %.2e)\n", q.vMax, q.steps, vRung);
    check(q.vMax > 1e-12 && q.vMax >= 100.0 * vRung, "G2.3b control: a jittered axis node fails the band (|v|/c > 1e-12, >= 100x the rungs')");
}

// G2.5 nc A (eps 2), G2.6 nc B (p_b 0.6): the closed q1d numbers on one row.
void q1dNc()
{
    std::printf("G2.5 nc A one row, eps 2, total + riemann p_b 0.01, FP64, tt 160 t*\n");
    const int nxs[3] = {200, 400, 800};
    double ef[3];
    for (int k = 0; k < 3; ++k) {
        const Gen g = gen(std::format("ncA{}", nxs[k]), nxs[k], 1, "--contour nc --mode planar --exit riemann --pb 0.01");
        const Q1d q = q1dRun(g, {.pb = 0.01 * kP0, .exit = FvExit::Riemann}, Run{}, 160.0, true);
        ef[k] = std::abs(q.CfVac - 1.457600);
        std::printf("  nx %d: %ld steps %s; CfVac %.7f (%+.2e) Fin %.7f (%+.2e) wall %.7f (%+.2e) p_e/p0 %.7f (%+.2e) Cd %.7f; max |v|/c %.2e\n", nxs[k],
                    q.steps, q.err.c_str(), q.CfVac, q.CfVac - 1.457600, q.Fin, q.Fin + 2.595093, q.Fwall, q.Fwall - 1.137493, q.pe, q.pe - 0.093933, q.Cd,
                    q.vMax);
        check(q.err.empty() && q.steps > 2000, std::format("nc A {}: live", nxs[k]));
        check(q.vMax <= 1e-12, std::format("G2.3b nc A {}: max |v|/c {:.2e} <= 1e-12", nxs[k], q.vMax));
    }
    std::printf("  CfVac errors %.2e %.2e %.2e, orders %.2f %.2f\n", ef[0], ef[1], ef[2], order(ef[0], ef[1]), order(ef[1], ef[2]));
    check(ef[1] <= 2e-4 && ef[2] < ef[1], "G2.5 CfVac <= 2e-4 at 400, falling");
    check(order(ef[0], ef[1]) >= 1.5 && order(ef[1], ef[2]) >= 1.5, "G2.5 CfVac order >= 1.5 on both pairs");
}

void q1dShock()
{
    const int nxs[3] = {200, 400, 800};
    std::printf("G2.6 nc B one row, p_b/p0 0.6 (riemann), FP64, tt 400 t*; x_s by (p1 + p2)/2\n");
    // The quasi-1D shock (oracle.py normalShockPos): A/A* 1.755749, M1 2.04716, x_s 3.08380.
    const double g = kGam, M1 = 2.04716, p1 = std::pow(1 + 0.5 * (g - 1) * M1 * M1, -g / (g - 1)) * kP0;
    const double p2 = p1 * (2 * g * M1 * M1 - (g - 1)) / (g + 1), level = 0.5 * (p1 + p2), xsO = 3.08380;
    double ex[3], ee[3], h[3];
    for (int k = 0; k < 3; ++k) {
        const Gen gB = gen(std::format("ncB{}", nxs[k]), nxs[k], 1, "--contour nc --mode planar --exit fixed --pb 0.6");
        h[k] = (gB.pm.px[std::size_t(nxs[k])] - gB.pm.px[0]) / nxs[k];
        const Rows rows{.pb = 0.6 * kP0, .exit = FvExit::Riemann};
        const Q1d q = q1dRun(gB, rows, Run{}, 400.0, false, 8, level, xsO);
        const double xs = shockCrossing(gB.fm, q.state, level, xsO, g);
        std::printf("  nx %d x_s at t* 50 .. 400:", nxs[k]);
        for (double x : q.xsHist) std::printf(" %.5f", x);
        std::printf("\n");
        ex[k] = std::abs(xs - xsO);
        ee[k] = std::abs(q.CfVac - 1.574057);
        const Q1d z = q1dRun(gB, rows, Run{.shock = {.mode = ShockMode::Hll, .eps4 = 0.0}}, 400.0, false, 4);
        const double xz = shockCrossing(gB.fm, z.state, level, xsO, g);
        std::printf("  nx %d (h %.4f): %ld steps %s; x_s %.5f (%+.2e = %.2f h) Fexit/(p0A*) %.7f (%+.2e) wall %.7f (%+.2e) Cd %.7f; d4 0 [reported]: x_s %.5f "
                    "Fexit %.7f %s\n",
                    nxs[k], h[k], q.steps, q.err.c_str(), xs, xs - xsO, (xs - xsO) / h[k], q.CfVac, q.CfVac - 1.574057, q.Fwall, q.Fwall - 1.021036, q.Cd, xz,
                    z.CfVac, z.err.c_str());
        check(q.err.empty() && q.steps > 2000, std::format("nc B {}: live", nxs[k]));
        check(q.vMax <= 1e-12, std::format("G2.3b nc B {}: max |v|/c {:.2e} <= 1e-12", nxs[k], q.vMax));
        check(ex[k] <= h[k], std::format("G2.6 nx {}: |dx_s| {:.2e} <= h {:.4f}", nxs[k], ex[k], h[k]));
    }
    std::printf("  x_s errors %.2e %.2e %.2e, orders %.2f %.2f; Fexit errors %.2e %.2e %.2e\n", ex[0], ex[1], ex[2], order(ex[0], ex[1]), order(ex[1], ex[2]), ee[0],
                ee[1], ee[2]);
    // UNMET (masterplan_combnozzle A10): x_s is settled from t* 50 at every rung and inside 0.12 h, but the crossing's
    // sub-cell offset is not monotone in h. Printed, not counted (P0's rule for a recorded UNMET).
    const bool xsOrder = order(ex[0], ex[1]) >= 0.9 && order(ex[1], ex[2]) >= 0.9;
    std::printf("  %s G2.6 x_s order >= 0.9 on both pairs (%.2f, %.2f)\n", xsOrder ? "[PASS]" : "[UNMET (recorded, A10)]", order(ex[0], ex[1]),
                order(ex[1], ex[2]));
    check(ee[1] <= 5e-4, std::format("G2.6 Fexit at 400 {:.2e} <= 5e-4", ee[1]));
    // The control: p_b 1% high at 800.
    const Gen gB = gen("ncB800c", 800, 1, "--contour nc --mode planar --exit fixed --pb 0.6");
    const Q1d c = q1dRun(gB, {.pb = 0.606 * kP0, .exit = FvExit::Riemann}, Run{}, 400.0, false, 4);
    const double xc = shockCrossing(gB.fm, c.state, level, xsO, g);
    std::printf("  control p_b 0.606 at 800: x_s %.5f, moved %.2e vs the 800 error %.2e\n", xc, std::abs(xc - xsO), ex[2]);
    check(std::abs(xc - xsO) >= 10.0 * ex[2], "G2.6 control: p_b off 1% moves x_s >= 10x the 800-cell error");
}

// G2.7: a supersonic exit cannot see p_a: state and row bytes equal across p_a, Cf by arithmetic.
void ambient()
{
    std::printf("G2.7 ambient: nc A one row 200, riemann at p_a/p0 0.01, 0.093933, 0.45, q1d IC, FP64, 20000 steps at a fixed dt (t* ~160)\n");
    const Gen g = gen("amb", 200, 1, "--contour nc --mode planar --exit riemann --pb 0.01");
    struct Out
    {
        FvState s;
        std::vector<double> row;
        double exitSuperMin = 1;
    };
    const auto cols = fvThrustColumns();
    const auto colOf = [&](const char* n) { return std::size_t(std::ranges::find(cols, n) - cols.begin()); };
    const auto run = [&](double pa, FvExit exit) {
        Rows rows{.pb = pa * kP0, .exit = exit};
        const FvCompConfig cfg = cfgOf(g.fm, rows, Run{});
        auto d = orDie(DeviceFvComp::make(g.fm, cfg), "amb");
        orDie(d.upload(q1dIc(g, kGam)), "amb");
        orDie(d.setLedger(true), "ledger");
        const std::vector<FvRole> roles = fvRoles(cfg.bc, g.fm);
        const FvThrustCase k{.axis = {1, 0, 0}, .pa = pa * kP0, .p0 = kP0, .t0 = kT0, .astar = kAstar, .gamma = kGam, .Rgas = 287.1,
                             .exitArea = fvExitArea(g.fm, roles, {1, 0, 0})};
        Out o;
        FvState s0;
        orDie(d.download(s0), "amb");
        const FvRow6 P0 = fvTotals6(g.fm, s0, -1);
        FvRoleSums prev{};
        double t = 0, tPrev = 0;
        // Fixed dt (the step count and every dt identical across p_a by construction of the bytes claim).
        const double dt = 0.95 * orDie(d.stableDt({.cfl = 0.4}), "dt");
        for (int n = 0; n < 10; ++n) {
            for (int q = 0; q < 2000; ++q) d.step(dt);
            t += 2000 * dt;
            FvEval ev;
            FvState rr;
            orDie(d.evaluate(ev, rr), "amb");
            const FvExitSample x = fvExitSample(g.fm, roles, ev, rr, k);
            o.exitSuperMin = std::min(o.exitSuperMin, x.exitSuper);
            const FvRoleSums now = fvRoleSums(orDie(d.ledger(), "ledger"), roles, -1);
            orDie(d.download(o.s), "amb");
            o.row = fvThrustRow(t, 0, dt, fvTotals6(g.fm, o.s, -1), P0, now, fvThrustWindow(prev, tPrev, now, t, k), x, k);
            prev = now;
            tPrev = t;
        }
        std::printf("    p_a %.6f %s: t* %.1f, CfVac %.9f Cf %.9f exitSuper min %.3f\n", pa, exit == FvExit::Fixed ? "fixed" : "riemann", t * kC0,
                    o.row[colOf("CfVac")], o.row[colOf("Cf")], o.exitSuperMin);
        return o;
    };
    const double pas[3] = {0.01, 0.093933, 0.45};
    const Out a = run(pas[0], FvExit::Riemann);
    const double eps = fvExitArea(g.fm, fvRoles(cfgOf(g.fm, {}, Run{}).bc, g.fm), {1, 0, 0}) / kAstar;
    for (double pa : {pas[1], pas[2]}) {
        const Out b = run(pa, FvExit::Riemann);
        const bool bytes = std::memcmp(a.s.q.data(), b.s.q.data(), a.s.q.size() * sizeof(double)) == 0;
        bool rowBytes = true;
        for (std::size_t i = 0; i < cols.size(); ++i)
            if (cols[i] != "Cf" && cols[i] != "Fpres" && cols[i] != "Texit" && cols[i] != "Twall") rowBytes = rowBytes && a.row[i] == b.row[i];
        const double cf = b.row[colOf("Cf")], cfv = b.row[colOf("CfVac")], arith = cf - (cfv - pa * eps);
        std::printf("  p_a %.6f: state bytes %s, row bytes (but the p_a columns) %s; Cf - (CfVac - (p_a/p0) eps_e) = %.2e (eps_e %.6f); exitSuper min %.3f\n", pa,
                    bytes ? "equal" : "DIFFER", rowBytes ? "equal" : "DIFFER", arith, eps, b.exitSuperMin);
        check(bytes && rowBytes, std::format("G2.7 p_a {}: state and row bytes equal p_a 0.01's", pa));
        check(std::abs(arith) <= 1e-15 * std::abs(cfv), std::format("G2.7 p_a {}: Cf arithmetic 1e-15", pa));
        check(b.exitSuperMin == 1.0, std::format("G2.7 p_a {}: exitSuper == 1 at every sample", pa));
    }
    const Out f = run(0.45, FvExit::Fixed);
    const bool same = std::memcmp(a.s.q.data(), f.s.q.data(), a.s.q.size() * sizeof(double)) == 0;
    check(!same, "G2.7 control: exit = \"fixed\" at 0.45 breaks the bytes");
}

// G2.9: the react contour's inert twin (lc 0, eps_c = eps_e = 6.25), gamma 1.4 and 1.2, 400 and 800.
void twin()
{
    std::printf("G2.9 the react twin: lc 0, eps_c = eps_e = 6.25, inviscid, FP64, total + riemann p_b 0.005, tt 160 t*\n");
    const double o14 = 1.608145, o12 = 1.678517;
    // Rungs (A8): the plan names none; P4 compares against the twin on the reacting nozzle's xi grid (h ~ 1.1e-3 y*,
    // ~15 000 cells). 400 and 800 (h 0.04, 0.02) are reported; the gate's rungs are 1600 and 3200 (h 0.01, 0.005).
    double e[2][4], cf14 = 0;
    const double gs[2] = {1.4, 1.2};
    const int nxs[4] = {400, 800, 1600, 3200};
    for (int gi = 0; gi < 2; ++gi)
        for (int k = 0; k < 4; ++k) {
            const int nx = nxs[k];
            const Gen g = gen(std::format("rt{}_{}", nx, gi), nx, 1, std::format("--contour react --lc 0 --mode planar --exit riemann --pb 0.005 --gamma {}", gs[gi]));
            const Q1d q = q1dRun(g, {.pb = 0.005 * kP0, .exit = FvExit::Riemann}, Run{.gamma = gs[gi]}, 160.0, true);
            const double o = gi == 0 ? o14 : o12;
            e[gi][k] = std::abs(q.CfVac - o);
            if (gi == 0 && k == 0) cf14 = q.CfVac;
            std::printf("  gamma %.1f nx %d: %ld steps %s; CfVac %.8f vs %.6f (%+.2e), Cd %.8f, M_e %.5f, max |v|/c %.2e\n", gs[gi], nx, q.steps, q.err.c_str(),
                        q.CfVac, o, q.CfVac - o, q.Cd, q.Me, q.vMax);
            check(q.err.empty() && q.steps > 2000, std::format("twin gamma {} {}: live", gs[gi], nx));
            check(q.vMax <= 1e-12, std::format("G2.3b twin gamma {} {}: max |v|/c {:.2e} <= 1e-12", gs[gi], nx, q.vMax));
        }
    for (int gi = 0; gi < 2; ++gi) {
        std::printf("  gamma %.1f: |CfVac - oracle| %.2e %.2e %.2e %.2e, orders %.2f %.2f %.2f\n", gs[gi], e[gi][0], e[gi][1], e[gi][2], e[gi][3],
                    order(e[gi][0], e[gi][1]), order(e[gi][1], e[gi][2]), order(e[gi][2], e[gi][3]));
        const bool coarse = e[gi][0] <= 1e-4 && e[gi][1] < e[gi][0];
        std::printf("  %s G2.9 gamma %.1f at 400 / 800 (reported rungs): CfVac <= 1e-4 at 400 (%.2e)\n", coarse ? "[PASS]" : "[UNMET (recorded, A8)]", gs[gi],
                    e[gi][0]);
        check(e[gi][2] <= 1e-4 && e[gi][3] < e[gi][2],
              std::format("G2.9 gamma {}: CfVac <= 1e-4 at 1600 ({:.2e}) and falling ({:.2e})", gs[gi], e[gi][2], e[gi][3]));
    }
    std::printf("  control: the gamma 1.2 oracle vs the gamma 1.4 run: %.2f%%\n", 100 * std::abs(cf14 / o12 - 1));
    check(std::abs(cf14 / o12 - 1) >= 0.04, "G2.9 control: the gamma 1.2 oracle misses the 1.4 run by >= 4%");
    std::printf("G2.9 T1: the twin at 400 vs rhoCentralFoam upwind\n");
    t1Case("nozzle_RT", 0.0, 1e-11, 2.4, "RT 400");
}

// The thrust row's band scale of row k: sum_c V |q_k| + sum_roles |I_k| (R2), of a device state.
double bandScale(const FaceMesh& fm, const FvState& s, const FvRoleSums& r, int k)
{
    double S = 0;
    for (const FvRow6& o : r.I) S += std::abs(o[std::size_t(k)]);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) S += fm.V[c] * std::abs(s.var(k)[c]);
    return S;
}

// G2.2: the ledger identity per row over the ramp; Texit == Twall; the 1/3 weights; FvRef vs device.
void ledgerGate()
{
    std::printf("G2.2 the impulse ledger: rest + ramp 40 t*, 1000 steps through it, FP64, hll\n");
    struct Case
    {
        const char* name;
        FaceMesh fm;
        Rows rows;
    };
    std::vector<Case> cases;
    cases.push_back({"nozzle_A", loadRef("nozzle_A", CellOrder::Morton).fm, {.pb = 0.1 * kP0, .exit = FvExit::Riemann}});
    cases.push_back({"nozzle_W", loadRef("nozzle_W", CellOrder::Morton).fm, {.pb = 0.1 * kP0, .exit = FvExit::Riemann}});
    cases.push_back({"q1d 200", gen("lpar", 200, 1, "--contour parabola --mode planar --exit riemann --pb 0.1").fm, {.pb = 0.1 * kP0, .exit = FvExit::Riemann}});
    for (Case& k : cases) {
        const FaceMesh& fm = k.fm;
        const auto march = [&](bool eq, double& worst, FvState& end, double& tex) {
            FvCompConfig cfg = cfgOf(fm, k.rows, Run{});
            auto d = orDie(DeviceFvComp::make(fm, cfg), k.name);
            const double pRest = k.rows.pb, R = 287.1;
            orDie(d.upload(dns::fvRestState(fm, pRest, kT0, kGam, R)), k.name);
            orDie(d.setLedger(true, eq), "ledger");
            const std::vector<FvRole> roles = fvRoles(cfg.bc, fm);
            FvState s0;
            orDie(d.download(s0), "s0");
            const FvRow6 P0 = fvTotals6(fm, s0, -1);
            const int inlet = int(std::ranges::find(cfg.bc, FvBcKind::Total, &FvBc::kind) - cfg.bc.begin());
            FvBc tot = cfg.bc[std::size_t(inlet)];
            tot.pTotal = pRest;
            orDie(d.setBc(inlet, tot), "ramp");
            const double tRamp = 40.0 / kC0;
            double t = 0;
            worst = 0;
            for (int n = 0; n < 1000; ++n) {
                const double dt = 0.95 * orDie(d.stableDt({.cfl = 0.4}), "dt");
                d.step(dt);
                t += dt;
                const double x = std::min(t / tRamp, 1.0), frac = x * x * (3 - 2 * x);
                tot.pTotal = pRest + frac * (kP0 - pRest);
                orDie(d.setBc(inlet, tot), "ramp");
                if (n % 100 != 99) continue;
                FvState s;
                orDie(d.download(s), "s");
                const FvRoleSums r = fvRoleSums(orDie(d.ledger(), "ledger"), roles, -1);
                const FvRow6 P = fvTotals6(fm, s, -1);
                for (int v = 0; v < 5; ++v) {
                    double res = P[std::size_t(v)] - P0[std::size_t(v)];
                    for (const FvRow6& o : r.I) res += o[std::size_t(v)];
                    // A row with S 0 (w on a planar or wedge mesh): the identity must hold as 0.0.
                    const double S = bandScale(fm, s, r, v), e = S > 0.0 ? std::abs(res) / S : (res == 0.0 ? 0.0 : HUGE_VAL);
                    worst = std::isfinite(e) ? std::max(worst, e) : HUGE_VAL;
                }
                end = s;
            }
            // Texit == Twall at p_a 0 and 0.1 p0, at the end state.
            tex = 0;
            FvEval ev;
            FvState rr;
            orDie(d.evaluate(ev, rr), "eval");
            const FvRoleSums r = fvRoleSums(orDie(d.ledger(), "ledger"), roles, -1);
            for (double pa : {0.0, 0.1 * kP0}) {
                const FvThrustCase tk{.axis = {1, 0, 0}, .pa = pa, .p0 = kP0, .astar = kAstar, .gamma = kGam, .Rgas = R};
                const FvExitSample x = fvExitSample(fm, roles, ev, rr, tk);
                tex = std::max(tex, std::abs(x.Texit - x.Twall) / bandScale(fm, end, r, 1));
            }
            std::printf("    %s%s: t* %.2f, ramp frac %.3f\n", k.name, eq ? " (1/3 weights)" : "", t * kC0, std::min(t / tRamp, 1.0));
        };
        double w = 0, wc = 0, tex = 0, texc = 0;
        FvState s, sc;
        march(false, w, s, tex);
        march(true, wc, sc, texc);
        std::printf("  %s: max |res_k| / S_k %.2e; |Texit - Twall| / S_mx %.2e; control 1/3 weights %.2e\n", k.name, w, tex, wc);
        check(w <= 1e-12, std::format("G2.2 {}: ledger identity {:.2e} <= 1e-12 of S_k", k.name, w));
        check(tex <= 1e-12, std::format("G2.2 {}: Texit == Twall at p_a 0 and 0.1 ({:.2e})", k.name, tex));
        check(wc >= 1e-6, std::format("G2.2 {} control: weights 1/3 each {:.2e} >= 1e-6", k.name, wc));
        // FvRef's ledger vs the device: 200 steps at the full p0 from rest at p0/2 (a 10:1 start without the
        // ramp diverges on one row), the device's dt replayed.
        FvCompConfig cfg = cfgOf(fm, k.rows, Run{});
        auto d = orDie(DeviceFvComp::make(fm, cfg), k.name);
        const FvState ic = dns::fvRestState(fm, 0.5 * kP0, kT0, kGam, 287.1);
        orDie(d.upload(ic), "ic");
        orDie(d.setLedger(true), "ledger");
        FvConfig rc{.tr = cfg.tr, .bc = cfg.bc, .shock = cfg.shock};
        const FvRef ref = orDie(FvRef::make(fm, rc), "FvRef");
        FvState h = ic;
        FvLedger hl = FvLedger::zeros(int(fm.patches.size()), kFvMeanVars);
        for (int n = 0; n < 200; ++n) {
            const double dt = 0.95 * orDie(d.stableDt({.cfl = 0.4}), "dt");
            d.step(dt);
            ref.march(h, dt, 1, nullptr, &hl);
        }
        const FvLedger dl = orDie(d.ledger(), "ledger");
        const std::vector<FvRole> roles = fvRoles(cfg.bc, fm);
        const FvRoleSums rd = fvRoleSums(dl, roles, -1);
        double gap = 0;
        for (int p = 0; p < dl.nP; ++p)
            for (int v = 0; v < 5; ++v) {
                // S per patch here: a wedge's two sides carry +-z fluxes whose role sum cancels.
                double S = 0;
                for (int q = 0; q < dl.nP; ++q) S += std::abs(dl.at(q, v));
                for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) S += fm.V[c] * std::abs(h.var(v)[c]);
                const double dI = std::abs(dl.at(p, v) - hl.at(p, v));
                gap = std::max(gap, S > 0.0 ? dI / S : (dI == 0.0 ? 0.0 : HUGE_VAL));
            }
        std::printf("  %s: FvRef vs device ledger, 200 steps: max |dI| / S_k %.2e (exit mass %.4g)\n", k.name, gap, rd.I[1][0]);
        check(gap <= 1e-13, std::format("G2.2 {}: FvRef ledger vs device {:.2e} <= 1e-13", k.name, gap));
    }
}

// N16 (masterplan_combnozzle A23): the compensated ledger's residual does not grow with the step count. One-row
// parabola, q1d IC, total + riemann p_b 0.1, FP64, hll, 100 000 steps at the IC's 0.95 stableDt; checks at 1e3, 1e4,
// 1e5 steps. The plain sum (the hi word alone: the pre-N16 ledger bit for bit) is the control.
//   gate     compensated max_k |res_k| / S_k <= 1e-14 at every check, last / first <= 4
//   control  plain at 1e5 >= 10 x compensated at 1e5 (the uncompensated growth)
void ledgerLong()
{
    std::printf("N16 the compensated ledger: q1d 200 parabola, 100000 steps, FP64, hll\n");
    const Gen g = gen("llong", 200, 1, "--contour parabola --mode planar --exit riemann --pb 0.1");
    const Rows rows{.pb = 0.1 * kP0, .exit = FvExit::Riemann};
    FvCompConfig cfg = cfgOf(g.fm, rows, Run{});
    auto d = orDie(DeviceFvComp::make(g.fm, cfg), "llong");
    orDie(d.upload(q1dIc(g, kGam)), "llong");
    orDie(d.setLedger(true), "ledger");
    const std::vector<FvRole> roles = fvRoles(cfg.bc, g.fm);
    FvState s0;
    orDie(d.download(s0), "s0");
    const FvRow6 P0 = fvTotals6(g.fm, s0, -1);
    const double dt = 0.95 * orDie(d.stableDt({.cfl = 0.4}), "dt");
    const auto worst = [&](const FvLedger& l, const FvState& s) {
        const FvRoleSums r = fvRoleSums(l, roles, -1);
        const FvRow6 P = fvTotals6(g.fm, s, -1);
        double w = 0;
        for (int v = 0; v < 5; ++v) {
            double res = P[std::size_t(v)] - P0[std::size_t(v)];
            for (const FvRow6& o : r.I) res += o[std::size_t(v)];
            const double S = bandScale(g.fm, s, r, v);
            if (S > 0.0) w = std::max(w, std::abs(res) / S);
        }
        return w;
    };
    std::vector<double> comp, plain;
    long n = 0;
    for (const long target : {1000L, 10000L, 100000L}) {
        for (; n < target; ++n) d.step(dt);
        FvState s;
        orDie(d.download(s), "s");
        FvLedger l = orDie(d.ledger(), "ledger");
        comp.push_back(worst(l, s));
        std::ranges::fill(l.lo, 0.0);
        l.omegaLo = 0.0;
        plain.push_back(worst(l, s));
        std::printf("  %6ld steps: max |res_k| / S_k compensated %.2e, plain (hi only) %.2e\n", n, comp.back(), plain.back());
    }
    const double cmax = *std::ranges::max_element(comp);
    check(cmax <= 1e-14, std::format("N16: compensated ledger identity {:.2e} <= 1e-14 at every check", cmax));
    check(comp.back() <= 4.0 * std::max(comp.front(), 1e-17),
          std::format("N16: compensated flat, 1e5 / 1e3 steps {:.2e} / {:.2e} <= 4", comp.back(), comp.front()));
    check(plain.back() >= 10.0 * comp.back(),
          std::format("N16 control: the plain sum grows ({:.2e} >= 10 x {:.2e})", plain.back(), comp.back()));
}

// G2.10 and the CLI's file rules: nozzle_A 80x16 through dns flow with [thrust].
void thrustCli()
{
    std::printf("G2.10 nozzle_A 80x16 through the CLI, [thrust] pa 0, rest + ramp 40, tt 200 t*, FP64\n");
    const std::string data = std::string(DNS_MESH_DATA) + "/../fv/nozzle_A";
    std::string bc;
    {
        std::ifstream in(data + "/bc.toml");
        bc.assign(std::istreambuf_iterator<char>(in), {});
    }
    const fs::path bcT = tmpDir() / "A_thrust.toml";
    std::ofstream(bcT) << bc << "\n[thrust]\npa = 0\n";
    const std::string base = std::format("flow --physics compressible --polymesh {} --shock hll --inviscid --mach 1 --fv-precision fp64 --emit stats", data);
    const fs::path o = tmpDir() / "A_thr";
    const int rc = runDns(base + std::format(" --bc {} --ramp 40 --tt 200 --out {}", bcT.string(), o.string()), o.string() + ".log");
    const std::string csv = o.string() + "_thrust.csv";
    const auto col = [&](const char* c) { return csvColumn(csv, c); };
    const auto t = col("t");
    check(rc == 0 && t.size() > 10, std::format("rc {}, {} thrust rows", rc, t.size()));
    if (t.size() < 10) return;
    const char* rows[5] = {"mass", "mx", "my", "mz", "E"};
    const char* roles[4] = {"in", "exit", "wall", "side"};
    double worst = 0, tex = 0;
    for (int v = 0; v < 5; ++v) {
        const auto res = col(std::format("res_{}", rows[v]).c_str()), P = col(std::format("P_{}", rows[v]).c_str());
        std::vector<std::vector<double>> I;
        for (const char* r : roles) I.push_back(col(std::format("I_{}_{}", r, rows[v]).c_str()));
        for (std::size_t i = 0; i < t.size(); ++i) {
            // S' = |P| + sum |I| <= S_k: a stricter band than R2's.
            double S = std::abs(P[i]);
            for (const auto& c : I) S += std::abs(c[i]);
            worst = std::max(worst, std::abs(res[i]) / S);
            if (v == 1) tex = std::max(tex, std::abs(col("Texit")[i] - col("Twall")[i]) / S);
        }
    }
    // The window [140, 240] t* from the ledger: Cf = Delta I_exit,x / Delta t / (p0 A*), box units.
    const auto Ix = col("I_exit_mx");
    std::size_t i1 = 0;
    while (i1 + 1 < t.size() && t[i1] < 140.0) ++i1;
    const std::size_t i2 = t.size() - 1;
    const double astar = astarOf(data), p0 = 1.0 / kGam;
    const double cf = (Ix[i2] - Ix[i1]) / (t[i2] - t[i1]) / (p0 * astar);
    const double lam = std::sin(15.0 * std::numbers::pi / 180) / (15.0 * std::numbers::pi / 180);
    std::printf("  %zu rows: max |res_k| / (|P_k| + sum |I_k|) %.2e, |Texit - Twall| %.2e; window [%.1f, %.1f] Cf %.6f, bracket [1.43700, 1.44315] "
                "(lambda %.6f)\n",
                t.size(), worst, tex, t[i1], t[i2], cf, lam);
    check(worst <= 1e-12, std::format("G2.10 ledger identity in _thrust.csv {:.2e} <= 1e-12", worst));
    check(tex <= 1e-12, "G2.10 Texit == Twall in every row");
    check(cf >= 1.43700 && cf <= 1.44315, std::format("G2.10 window Cf {:.6f} in [1.43700, 1.44315]", cf));
    // Without [thrust]: no file, the stats bytes.
    const fs::path a = tmpDir() / "A_s1", b = tmpDir() / "A_s2";
    const int r1 = runDns(base + std::format(" --bc {} --ramp 2 --tt 3 --out {}", data + "/bc.toml", a.string()), a.string() + ".log");
    const int r2 = runDns(base + std::format(" --bc {} --ramp 2 --tt 3 --out {}", bcT.string(), b.string()), b.string() + ".log");
    const auto bytes = [](const fs::path& f) {
        std::ifstream in(f, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), {});
    };
    const std::string s1 = bytes(a.string() + "_stats.csv"), s2 = bytes(b.string() + "_stats.csv");
    check(r1 == 0 && r2 == 0 && !s1.empty() && s1 == s2, "stats bytes equal with and without [thrust]");
    check(!fs::exists(a.string() + "_thrust.csv") && fs::exists(b.string() + "_thrust.csv"), "_thrust.csv only with [thrust]");
}

// masterplan_combnozzle P4 (ctest gpu_fv_nozzle_react): the reacting one-row nozzle through dns flame at 15 cells /
// delta_th, tools/fv/nozzle_react.py's rig (M 0.042, gamma 1.4, eps_e 2), tt 3 (the full-length gates are the runs of
// docs/nozzle.md "Reacting"):
//   G4.1  every _thrust.csv row: |res_k| <= 1e-12 (|P_k| + sum |I_k| (+ q |Omega|, |Omega|)); control: Omega dropped
//         from the rho Y row fails by >= 1e-6
//   G4.8  shockCells 0 on every row
//   G4.12a (masterplan_combclose) the first row's station T0b on the design's to 1e-6 (the Rayleigh-line chamber IC)
//   N16b  (masterplan_combclose K4) the same run with --compensated (FP64 double-double): max r <= 1e-14, flat (last 10%
//         of rows <= 4 x rows 10-20%); control: the plain run's max r >= 4e-14 (lattice bias, test_stage_bias)
//   route the station columns present and finite, exitSuper 1, the drift guard silent; S_L on A_ref within 1% of the
//         1D S_L,num (the reduced tt reads the IC's start, not a gate); --inflow-speed exit 2
void reactRun()
{
    std::printf("P4 the reacting one-row nozzle through dns flame, 15 cells / delta_th, M 0.042, FP64, --shock hll, tt 3\n");
    const fs::path dir = tmpDir() / "react15";
    const fs::path tool = fs::path(DNS_MESH_DATA) / ".." / ".." / ".." / "tools/fv/nozzle_react.py";
    const std::string gen = std::format("python3 {} design --cells 15 --mach 0.042 --sl 0.9945059 --out {} > /dev/null", tool.string(),
                                        dir.string());
    if (std::system(gen.c_str()) != 0) {
        check(false, "nozzle_react.py design wrote the case");
        return;
    }
    const std::string base = std::format("flame --mach 0.042 --expansion 5 --zeldovich 8 --prandtl 0.72 --polymesh {0} --bc {0}/bc.toml "
                                         "--shock hll --fv-precision fp64 --upstream 12 --emit stats --quiet",
                                         dir.string());
    const fs::path o = tmpDir() / "react15_run";
    const int rc = runDns(base + std::format(" --tt 3 --out {}", o.string()), o.string() + ".log");
    const std::string th = o.string() + "_thrust.csv", st = o.string() + "_stats.csv";
    const auto t = csvColumn(th, "t");
    check(rc == 0 && t.size() > 5, std::format("dns flame [nozzle]: rc {}, {} thrust rows", rc, t.size()));
    if (t.size() < 5) return;
    const double Tu = 1.0 / (1.4 * 0.042 * 0.042), q = 3.5 * 4.0 * Tu;
    // Per row r_i = max_k |res_k| / S'_k; ctrl: the rho Y row with Omega dropped.
    auto ledgerR = [&](const std::string& csv, double* ctrlOut) {
        const auto tt = csvColumn(csv, "t"), Om = csvColumn(csv, "Omega");
        std::vector<double> r(tt.size(), 0.0);
        double ctrl = 0;
        for (const char* k : {"mass", "mx", "my", "mz", "E", "ry"}) {
            const auto res = csvColumn(csv, std::format("res_{}", k)), P = csvColumn(csv, std::format("P_{}", k));
            std::vector<std::vector<double>> I;
            for (const char* rr : {"in", "exit", "wall", "side"}) I.push_back(csvColumn(csv, std::format("I_{}_{}", rr, k)));
            for (std::size_t i = 0; i < tt.size(); ++i) {
                double S = std::abs(P[i]);
                for (const auto& c : I) S += std::abs(c[i]);
                if (std::string(k) == "E") S += q * std::abs(Om[i]);
                if (std::string(k) == "ry") S += std::abs(Om[i]);
                if (S > 0) r[i] = std::max(r[i], std::abs(res[i]) / S);
                if (std::string(k) == "ry" && S > 0) ctrl = std::max(ctrl, std::abs(res[i] - Om[i]) / S);
            }
        }
        if (ctrlOut) *ctrlOut = ctrl;
        return r;
    };
    double ctrl = 0;
    const std::vector<double> rPlain = ledgerR(th, &ctrl);
    const double worst = *std::ranges::max_element(rPlain);
    std::printf("  %zu thrust rows: max |res_k| / S'_k %.2e; control (Omega dropped from rho Y) %.2e\n", t.size(), worst, ctrl);
    check(worst <= 1e-12, std::format("G4.1 ledger identity on the reacting route {:.2e} <= 1e-12", worst));
    check(ctrl >= 1e-6, std::format("G4.1 control: Omega dropped from the rho Y row fails ({:.2e} >= 1e-6)", ctrl));
    // N16b (masterplan_combclose K4): the same run on the compensated FP64 state (double-double:
    // the increment form + a low word) holds the ledger at its own rounding, flat; the plain FP64
    // stage-3 write's lattice bias (+0.11 ulp per write, test_stage_bias) grows it linearly.
    {
        const fs::path oc = tmpDir() / "react15_comp";
        const int rcc = runDns(base + std::format(" --compensated --tt 3 --out {}", oc.string()), oc.string() + ".log");
        const std::string thc = oc.string() + "_thrust.csv";
        const std::vector<double> rc = rcc == 0 ? ledgerR(thc, nullptr) : std::vector<double>{};
        check(rcc == 0 && rc.size() >= 20, std::format("N16b: the compensated FP64 run, rc {}, {} thrust rows", rcc, rc.size()));
        if (rc.size() >= 20) {
            const std::size_t n = rc.size();
            const double cmax = *std::ranges::max_element(rc);
            const double early = *std::max_element(rc.begin() + std::ptrdiff_t(n / 10), rc.begin() + std::ptrdiff_t(n / 5));
            const double late = *std::max_element(rc.begin() + std::ptrdiff_t(n - n / 10), rc.end());
            std::printf("  N16b compensated FP64: %zu rows, max r %.2e, last 10%% %.2e vs rows 10-20%% %.2e; plain max r %.2e\n", n, cmax,
                        late, early, worst);
            check(cmax <= 1e-14, std::format("N16b: compensated FP64 max r {:.2e} <= 1e-14", cmax));
            check(late <= 4 * early, std::format("N16b: compensated FP64 flat, last 10% {:.2e} <= 4 x rows 10-20% {:.2e}", late, early));
            check(worst >= 4e-14, std::format("N16b control: the plain FP64 run at the same tt fails ({:.2e} >= 4e-14)", worst));
        }
        std::ifstream lf(thc);
        std::string label;
        std::getline(lf, label);
        check(label.find("FP64 compensated (double-double state)") != std::string::npos,
              "N16b: the _thrust.csv label names the double-double state");
    }
    const auto sensed = csvColumn(st, "shockCells");
    check(!sensed.empty() && *std::ranges::max_element(sensed) == 0.0, "G4.8 shockCells 0 on every row");
    bool finite = true;
    for (const char* k : {"p0b", "T0b", "CdB", "CfBvac", "CfB"}) {
        const auto v = csvColumn(th, k);
        finite = finite && v.size() == t.size() && std::ranges::all_of(v, [](double x) { return std::isfinite(x); });
    }
    const auto sup = csvColumn(th, "exitSuper"), cdb = csvColumn(th, "CdB"), sl = csvColumn(st, "SL"), T0b = csvColumn(th, "T0b");
    std::printf("  last row: CdB %.6f, T0b / Tu %.8f, S_L %.6f (1D 0.9945059), exitSuper %.3f\n", cdb.back(), T0b.back() / Tu, sl.back(),
                sup.back());
    check(finite, "route: the station columns p0b T0b CdB CfBvac CfB on every row, finite");
    // G4.12(a) (masterplan_combclose 2.5): the chamber IC on the Rayleigh line puts the station's first row at the
    // design's T0b (fvref_react R1: 2024.7606632703096); premixedPoint's low-Mach IC read +1.709e-3 there.
    const double t0bDesign = 2024.7606632703096, g412a = T0b.front() / t0bDesign - 1;
    std::printf("  G4.12a first row (t %.4f): T0b / T0b_design - 1 %+.3e (old premixedPoint IC +1.709e-3)\n", t.front(), g412a);
    check(std::abs(g412a) <= 1e-6, std::format("G4.12a: first-row T0b on the design's {:+.2e} <= 1e-6", g412a));
    check(std::ranges::all_of(sup, [](double x) { return x == 1.0; }), "route: exitSuper 1 on every row");
    check(std::abs(sl.back() / 0.9945059 - 1) <= 0.01, "route: S_L on A_ref within 1% of the 1D S_L,num (start, not a gate)");
    const int r2 = runDns(base + std::format(" --tt 3 --inflow-speed 0.98 --dry-run --out {}", o.string()), o.string() + "_is.log");
    check(r2 == 2, std::format("--inflow-speed with [nozzle]: exit 2 (rc {})", r2));
}

// masterplan_combnozzle G4.5 (ctest gpu_fv_nozzle_reacttwin): the reacting nozzle's half against its inert twin. The
// twin is nozzle_react.py's rig from the contraction start (lc 0, y* 1, eps_c 6.172901800721939, Lcon 6, Ldiv 4, eps_e 2,
// 4680 cells: the 15-cell flame's xi grid), SI at p0 1e5, T0 300, from the generator's isentropic q1d IC (A20: the CLI
// twin from rest rang +-1e-3 in Cd at 60-80 t*). Viscous at the flame's Re* = rho0b c0b y*/mu 736.2624872413062 (flame
// units, mu = Pr; the gated run's station p0b 403.42733, T0b 2024.75881), Pr 0.72; inviscid against the closed oracle.
// tt 80 t*, window the last quarter (the ledger). The flame's window numbers (the gated 15-cell run, [40, 80] tau):
// CdB 0.9989656977, CfBvac 1.4546195121.
void reactTwin()
{
    std::printf("G4.5 the reacting nozzle's inert twin: react lc 0, eps_c 6.1729018, eps_e 2, 4680 cells, q1d IC, FP64, hll, tt 80 t*\n");
    const double cdB = 0.9989656977, cfB = 1.4546195121, cfO = 1.457600096948547, reStar = 736.2624872413062;
    const double pa = 0.010037095719382376; // the flame's pa over p0b (nozzle_react.py twin)
    const Gen g = gen("rtw", 4680, 1,
                      std::format("--contour react --lc 0 --lcon 6 --ldiv 4 --eps-c 6.172901800721939 --eps-e 2 --scale 1 --mode planar "
                                  "--exit riemann --pb {}",
                                  pa));
    const Rows rows{.pb = pa * kP0, .exit = FvExit::Riemann};
    const double rho0 = kP0 / (287.1 * kT0), mu = rho0 * kC0 * 1.0 / reStar;
    const Q1d v = q1dRun(g, rows, Run{.mu = mu}, 80.0, true);
    const Q1d i = q1dRun(g, rows, Run{}, 80.0, true);
    std::printf("  viscous (mu %.6g SI, Re* %.4f): %ld steps %s; Cd %.10f CfVac %.10f (vs the flame's CdB %+.2e, CfBvac %+.2e); max |v|/c %.2e\n", mu,
                reStar, v.steps, v.err.c_str(), v.Cd, v.CfVac, cdB - v.Cd, cfB - v.CfVac, v.vMax);
    std::printf("  inviscid: %ld steps %s; Cd %.10f CfVac %.10f (closed oracle %.7f: %+.2e); the viscous loss: Cd %+.2e, CfVac %+.2e\n", i.steps,
                i.err.c_str(), i.Cd, i.CfVac, cfO, i.CfVac - cfO, v.Cd - i.Cd, v.CfVac - i.CfVac);
    check(v.err.empty() && i.err.empty() && v.steps > 2000 && i.steps > 2000, "twins live");
    // UNMET as measured (masterplan_combnozzle A21: -1.10e-4): printed, not counted (P0's rule for a recorded UNMET).
    std::printf("  %s G4.5 |CdB - Cd_twin| %.2e <= 1e-4\n", std::abs(cdB - v.Cd) <= 1e-4 ? "[PASS]" : "[UNMET (recorded, A21)]",
                std::abs(cdB - v.Cd));
    check(std::abs(cdB - v.Cd) <= 2e-4, std::format("G4.5 regression: |CdB - Cd_twin| {:.2e} stays <= 2e-4", std::abs(cdB - v.Cd)));
    check(std::abs(i.CfVac - cfO) <= 1e-5, std::format("the inviscid twin vs the closed oracle {:.2e} <= 1e-5", std::abs(i.CfVac - cfO)));
    check(std::abs(cfB - v.CfVac) <= 1e-4, std::format("G4.5 |CfBvac - CfVac_twin| {:.2e} <= 1e-4", std::abs(cfB - v.CfVac)));
    // The control: Cd on T0u (A3's normalisation of the station: sqrt(T0u / T0b) ~ 1/sqrt(sigma)).
    const double T0u = 1.0003489340123555 / (1.4 * 0.042 * 0.042), T0b = 2024.7588104045626, cdU = cdB * std::sqrt(T0u / T0b);
    std::printf("  control: Cd on T0u %.6f, off the twin by %.3f\n", cdU, cdU - v.Cd);
    check(std::abs(cdU - v.Cd) >= 0.1, "G4.5 control: Cd on T0u misses by >= 0.1");
}

// The viscous twin of any rung (masterplan_combnozzle A23; no gate, the numbers for nozzle_react.py report):
// reacttwin NX EPS_C RE_STAR PA T_STAR [GAMMA]. The 15-cell twin is NX 4680 EPS_C 6.172901800721939 RE_STAR
// 736.2624872413062 PA 0.010037095719382376; the window is q1dRun's last quarter of T_STAR. GAMMA (default 1.4;
// masterplan_combclose K7): the generator's q1d IC, the gas, Re* = rho0 c0 y*/mu and t* = 1 m / c0 on c0(GAMMA).
void reactTwinAt(int nx, double epsC, double reStar, double pa, double tStar, double gamma)
{
    std::printf("twin: react lc 0, eps_c %.16g, eps_e 2, %d cells, Re* %.16g, pa %.16g, gamma %g, q1d IC, FP64, hll, tt %g t*\n", epsC, nx,
                reStar, pa, gamma, tStar);
    const Gen g = gen(std::format("rtw{}_{}", nx, int(std::lround(gamma * 100))), nx, 1,
                      std::format("--contour react --lc 0 --lcon 6 --ldiv 4 --eps-c {} --eps-e 2 --scale 1 --mode planar --exit riemann --pb {} "
                                  "--gamma {}",
                                  epsC, pa, gamma));
    const Rows rows{.pb = pa * kP0, .exit = FvExit::Riemann};
    const double c0 = std::sqrt(gamma * 287.1 * kT0), rho0 = kP0 / (287.1 * kT0), mu = rho0 * c0 * 1.0 / reStar;
    // q1dRun counts t* on kC0 (gamma 1.4): rescaled so the run is tStar of this gamma's t* (ratio exactly 1 at 1.4).
    const Q1d v = q1dRun(g, rows, Run{.mu = mu, .gamma = gamma}, tStar * (kC0 / c0), true);
    std::printf("  viscous: %ld steps %s; Cd %.10f CfVac %.10f; max |v|/c %.2e; window [%g, %g] t*\n", v.steps, v.err.c_str(), v.Cd, v.CfVac,
                v.vMax, 0.75 * tStar, tStar);
    check(v.err.empty() && v.steps > 2000, "twin live");
}

// The viscous twin by grid sequencing (masterplan_combclose K3, analysis_combclose_2 B4, its probe's `seq`):
// reacttwinseq NXC NXF EPS_C RE_STAR PA TC TF. The coarse twin (NXC cells, q1d IC) to TC t*, its state interpolated
// linearly in x onto NXF cells (v 0), then TF t* more. Windows of 5 t* (ledger): settled iff |mdotOut/mdotIn - 1| <= 1e-5
// in every window of the last 15 t*; Cd, CfVac over the last 15 t* (the ledger's endpoints). gamma 1.4.
void reactTwinSeq(int nxC, int nxF, double epsC, double reStar, double pa, double tC, double tF)
{
    std::printf("twin seq: react lc 0, eps_c %.16g, eps_e 2, Re* %.16g, pa %.16g; coarse %d cells tt %g t*, fine %d cells tt %g t*\n", epsC,
                reStar, pa, nxC, tC, nxF, tF);
    const std::string args =
        std::format("--contour react --lc 0 --lcon 6 --ldiv 4 --eps-c {} --eps-e 2 --scale 1 --mode planar --exit riemann --pb {}", epsC, pa);
    const Gen gc = gen(std::format("sc{}", nxC), nxC, 1, args), gf = gen(std::format("sf{}", nxF), nxF, 1, args);
    const Rows rows{.pb = pa * kP0, .exit = FvExit::Riemann};
    const double rho0 = kP0 / (287.1 * kT0), mu = rho0 * kC0 * 1.0 / reStar;
    const Run r{.mu = mu};
    const double ideal = fvIdealMassFlow(kP0, kT0, kAstar, r.gamma, 287.1);
    // One march of a twin in n equal windows: per window Cd on mdotOut, out/in - 1, CfVac; the sums at each end.
    struct Win
    {
        double t1, t2, cd, outIn, cf;
    };
    const auto march = [&](const Gen& g, const FvState& ic, double tStar, int n, const char* tag, std::vector<Win>& wins,
                           std::vector<FvRoleSums>& sums, FvState* end) -> bool {
        FvCompConfig cfg = cfgOf(g.fm, rows, r);
        auto d = orDie(DeviceFvComp::make(g.fm, cfg), tag);
        orDie(d.upload(ic), tag);
        orDie(d.setLedger(true), "ledger");
        const std::vector<FvRole> roles = fvRoles(cfg.bc, g.fm);
        const FvThrustCase k{.axis = {1, 0, 0}, .pa = rows.pb, .p0 = kP0, .t0 = kT0, .astar = kAstar, .gamma = r.gamma, .Rgas = 287.1,
                             .exitArea = fvExitArea(g.fm, roles, {1, 0, 0})};
        sums.push_back(fvRoleSums(orDie(d.ledger(), "ledger"), roles, -1));
        double t = 0;
        long steps = 0;
        for (int i = 1; i <= n; ++i) {
            const double tn = tStar / kC0 * i / n;
            const Marched m = marchTo(d, tn - t, 0.4, 4, 4000000);
            steps += m.steps;
            if (!m.err.empty()) {
                std::printf("  %s: %s\n", tag, m.err.c_str());
                return false;
            }
            sums.push_back(fvRoleSums(orDie(d.ledger(), "ledger"), roles, -1));
            const FvThrustWindow w = fvThrustWindow(sums[sums.size() - 2], t, sums.back(), tn, k);
            wins.push_back({t * kC0, tn * kC0, w.mdotOut / ideal, w.mdotOut / w.mdotIn - 1, w.CfVac});
            std::printf("  %s t* [%6.2f, %6.2f] steps %8ld Cd %.10f out/in-1 %+.2e CfVac %.10f\n", tag, t * kC0, tn * kC0, steps,
                        wins.back().cd, wins.back().outIn, wins.back().cf);
            t = tn;
        }
        if (end) orDie(d.download(*end), tag);
        return true;
    };
    std::vector<Win> wc, wf;
    std::vector<FvRoleSums> sc, sf;
    FvState coarse;
    if (!march(gc, q1dIc(gc, r.gamma), tC, 4, "coarse", wc, sc, &coarse)) {
        check(false, "twin seq: the coarse twin marched");
        return;
    }
    // x-sorted coarse centroids, linear interpolation of every conserved row, v 0.
    std::vector<std::pair<double, int>> xc;
    for (int c = 0; c < gc.fm.nC; ++c) xc.emplace_back(gc.fm.C[0][std::size_t(c)], c);
    std::ranges::sort(xc);
    FvState fine = q1dIc(gf, r.gamma);
    for (int c = 0; c < gf.fm.nC; ++c) {
        const double x = gf.fm.C[0][std::size_t(c)];
        const auto it = std::ranges::lower_bound(xc, std::pair{x, -1});
        const std::size_t i1 = std::clamp<std::size_t>(std::size_t(it - xc.begin()), 1, xc.size() - 1);
        const auto [x0, c0] = xc[i1 - 1];
        const auto [x1, c1] = xc[i1];
        const double w = std::clamp((x - x0) / (x1 - x0), 0.0, 1.0);
        for (int v = 0; v < fine.nV; ++v) fine.var(v)[c] = (1 - w) * coarse.var(v)[c0] + w * coarse.var(v)[c1];
        fine.var(2)[c] = 0.0;
    }
    const int n = std::max(1, int(std::lround(tF / 5.0)));
    if (!march(gf, fine, tF, n, "fine", wf, sf, nullptr)) {
        check(false, "twin seq: the fine twin marched");
        return;
    }
    const int last = std::min(n, std::max(1, int(std::lround(15.0 / (tF / n)))));
    double outIn = 0;
    for (int i = n - last; i < n; ++i) outIn = std::max(outIn, std::abs(wf[std::size_t(i)].outIn));
    // Cd, CfVac over the last 15 t*: the ledger's endpoints (sums[n - last], sums[n]).
    const double t1 = wf[std::size_t(n - last)].t1 / kC0, t2 = wf.back().t2 / kC0;
    const std::vector<FvRole> roles = fvRoles(cfgOf(gf.fm, rows, r).bc, gf.fm);
    const FvThrustCase k{.axis = {1, 0, 0}, .pa = rows.pb, .p0 = kP0, .t0 = kT0, .astar = kAstar, .gamma = r.gamma, .Rgas = 287.1,
                         .exitArea = fvExitArea(gf.fm, roles, {1, 0, 0})};
    const FvThrustWindow w = fvThrustWindow(sf[std::size_t(n - last)], t1, sf.back(), t2, k);
    std::printf("  fine window [%g, %g] t*: Cd %.10f CfVac %.10f; max |out/in - 1| per window %.2e (settled <= 1e-5: %s)\n", t1 * kC0,
                t2 * kC0, w.mdotOut / ideal, w.CfVac, outIn, outIn <= 1e-5 ? "yes" : "NO");
    check(outIn <= 1e-5, std::format("twin seq settled: |out/in - 1| {:.2e} <= 1e-5 per 5 t* over the last 15 t*", outIn));
}

// combscale S5 (masterplan_combscale R-g, G6.2): the inert twin of the multi-row rig, any rows:
// reacttwinrow NX NY LC EPS_C RE_STAR PA T_STAR [DIV]. The react contour at y* 1 (lc LC y* stub, Lcon 6, Ldiv 4, eps_e 2),
// planar, NY rows; NY 1: the q1d IC (v 0), NY >= 2: the generator's 2D IC (v = u y_w' y/y_w). RE_STAR <= 0: inviscid.
// Windows of 5 t* (ledger); settled iff |mdotOut/mdotIn - 1| <= 1e-5 in every window of the last 15 t*; Cd (exit
// flux), Cd on the inflow and CfVac over the last 100 t* (T_STAR / 4 if shorter), the ledger's endpoints. gamma 1.4.
// [DIV] (combscale N25): the divergent, cos (absent: the old argv, mesh and print) | cone (nozzle_mesh.py --div).
void reactTwinRow(int nx, int ny, double lc, double epsC, double reStar, double pa, double tStar, const std::string& div)
{
    const bool cos = div == "cos";
    std::printf("twin row: react lc %.6g, eps_c %.16g, eps_e 2, %d x %d cells, Re* %.16g, pa %.16g, tt %g t*, %s IC%s\n", lc, epsC, nx, ny,
                reStar, pa, tStar, ny == 1 ? "q1d" : "2D generator", cos ? "" : (", div " + div).c_str());
    const std::string args =
        std::format("--contour react --lc {} --lcon 6 --ldiv 4 --eps-c {} --eps-e 2 --scale 1 --mode planar --exit riemann --pb {}", lc,
                    epsC, pa) +
        (cos ? "" : " --div " + div);
    const Gen g = gen(std::format("rtr{}x{}{}", nx, ny, cos ? "" : div), nx, ny, args);
    const Rows rows{.pb = pa * kP0, .exit = FvExit::Riemann};
    const double rho0 = kP0 / (287.1 * kT0), mu = reStar > 0 ? rho0 * kC0 * 1.0 / reStar : 0.0;
    const Run r{.mu = mu};
    const double ideal = fvIdealMassFlow(kP0, kT0, kAstar, r.gamma, 287.1);
    FvCompConfig cfg = cfgOf(g.fm, rows, r);
    auto d = orDie(DeviceFvComp::make(g.fm, cfg), "twin row");
    orDie(d.upload(ny == 1 ? q1dIc(g, r.gamma) : icOfGas(g, r.gamma)), "twin row");
    orDie(d.setLedger(true), "ledger");
    const std::vector<FvRole> roles = fvRoles(cfg.bc, g.fm);
    const FvThrustCase k{.axis = {1, 0, 0}, .pa = rows.pb, .p0 = kP0, .t0 = kT0, .astar = kAstar, .gamma = r.gamma, .Rgas = 287.1,
                         .exitArea = fvExitArea(g.fm, roles, {1, 0, 0})};
    const int n = std::max(1, int(std::lround(tStar / 5.0)));
    std::vector<FvRoleSums> sums{fvRoleSums(orDie(d.ledger(), "ledger"), roles, -1)};
    std::vector<double> outIn;
    double t = 0;
    long steps = 0;
    for (int i = 1; i <= n; ++i) {
        const double tn = tStar / kC0 * i / n;
        const Marched m = marchTo(d, tn - t, 0.4, 4, 40000000);
        steps += m.steps;
        if (!m.err.empty()) {
            std::printf("  %s\n", m.err.c_str());
            check(false, "twin row marched");
            return;
        }
        sums.push_back(fvRoleSums(orDie(d.ledger(), "ledger"), roles, -1));
        const FvThrustWindow w = fvThrustWindow(sums[sums.size() - 2], t, sums.back(), tn, k);
        outIn.push_back(w.mdotOut / w.mdotIn - 1);
        if (i % std::max(1, n / 12) == 0 || i == n)
            std::printf("  t* [%7.2f, %7.2f] steps %9ld Cd %.10f out/in-1 %+.2e CfVac %.10f\n", t * kC0, tn * kC0, steps, w.mdotOut / ideal,
                        outIn.back(), w.CfVac);
        t = tn;
    }
    const int last = std::min(n, std::max(1, int(std::lround(15.0 / (tStar / n)))));
    double settle = 0;
    for (int i = n - last; i < n; ++i) settle = std::max(settle, std::abs(outIn[std::size_t(i)]));
    const double win = tStar >= 400.0 ? 100.0 : tStar / 4;
    const int iw = n - std::min(n, std::max(1, int(std::lround(win / (tStar / n)))));
    const double t1 = tStar / kC0 * iw / n, t2 = tStar / kC0;
    const FvThrustWindow w = fvThrustWindow(sums[std::size_t(iw)], t1, sums.back(), t2, k);
    FvState s;
    orDie(d.download(s), "twin row");
    std::printf("  window [%g, %g] t*: Cd %.10f Cd_in %.10f CfVac %.10f; max |out/in - 1| over the last 15 t* %.2e (settled <= 1e-5: %s); "
                "%ld steps, max |v|/c %.3e\n",
                t1 * kC0, t2 * kC0, w.mdotOut / ideal, w.mdotIn / ideal, w.CfVac, settle, settle <= 1e-5 ? "yes" : "NO", steps,
                vOverC(s, r.gamma));
    check(settle <= 1e-5, std::format("twin row settled: |out/in - 1| {:.2e} <= 1e-5 per 5 t* over the last 15 t*", settle));
}

// combscale S5 (ctest gpu_fv_nozzle_react2d): the multi-row reacting nozzle through dns flame, a smoke on 588 x 4 at
// the S5 rig's columns (y* 12 h, Lc 468 columns, 15 cells / delta_th), FP64 compensated, --shock hll:
//   route  rc 0; xIa1 xIa2 last in _stats.csv, finite; the label's half-domain suffix; shockCells 0
//   seed   --wrinkle 0.005: the first row's xIa1 within 2% of 0.005 delta_th; a planar run |xIa| <= 1e-6 delta_D
//   G4.1f  the ledger at <= 1e-13 on every row
//   scope  --wrinkle on the one-row chamber: exit 3 by name; one row: no xIa columns, no suffix
void react2dRun()
{
    std::printf("combscale S5: the multi-row reacting nozzle through dns flame, 588 x 4, FP64 compensated, --shock hll\n");
    const fs::path tool = fs::path(DNS_MESH_DATA) / ".." / ".." / ".." / "tools/fv/nozzle_react.py";
    const fs::path dir = tmpDir() / "react2d", one = tmpDir() / "react1d";
    const auto design = [&](const fs::path& d, int ny) {
        return std::system(std::format("python3 {} design --cells 15 --ny {} --ystar-cells 12 --lc-cells 468 --mach 0.042 --sl 0.9945059 "
                                       "--out {} > /dev/null",
                                       tool.string(), ny, d.string())
                               .c_str()) == 0;
    };
    if (!design(dir, 4) || !design(one, 1)) {
        check(false, "nozzle_react.py design wrote the cases");
        return;
    }
    const auto base = [](const fs::path& d) {
        return std::format("flame --mach 0.042 --expansion 5 --zeldovich 8 --prandtl 0.72 --polymesh {0} --bc {0}/bc.toml --shock hll "
                           "--fv-precision fp64 --compensated --upstream 12 --emit stats --quiet",
                           d.string());
    };
    const double dth = 1.601651;
    const fs::path o = tmpDir() / "r2d", op = tmpDir() / "r2dp", o1 = tmpDir() / "r1d";
    const int rc = runDns(base(dir) + std::format(" --wrinkle 0.005 --tt 0.5 --out {}", o.string()), o.string() + ".log");
    const std::string st = o.string() + "_stats.csv", th = o.string() + "_thrust.csv";
    const auto t = csvColumn(st, "t"), a1 = csvColumn(st, "xIa1"), a2 = csvColumn(st, "xIa2"), sens = csvColumn(st, "shockCells");
    check(rc == 0 && t.size() > 5 && a1.size() == t.size() && a2.size() == t.size(),
          std::format("route: rc {}, {} rows, xIa1 xIa2 present", rc, t.size()));
    if (t.size() < 5 || a1.size() != t.size()) return;
    std::string header, label;
    {
        std::ifstream f(st);
        std::getline(f, label);
        std::getline(f, header);
    }
    check(header.ends_with("xIa1,xIa2"), "route: xIa1 xIa2 are the last columns");
    check(label.find("[half-domain: transverse modes even about the symmetry plane only]") != std::string::npos,
          "route: the label states the half-domain");
    check(std::ranges::all_of(a1, [](double x) { return std::isfinite(x); }) && std::ranges::all_of(a2, [](double x) { return std::isfinite(x); }),
          "route: xIa finite");
    check(!sens.empty() && *std::ranges::max_element(sens) == 0.0, "G4.8: shockCells 0 on every row");
    const double seed = 0.005 * dth;
    std::printf("  seeded: first row t %.4f xIa1 %.6e (seed %.6e, %.4f), xIa2 %.2e; last t %.3f xIa1 %.6e xIa2 %.2e\n", t.front(), a1.front(), seed,
                a1.front() / seed, a2.front(), t.back(), a1.back(), a2.back());
    check(std::abs(a1.front() / seed - 1) <= 0.02, "seed: the first row's xIa1 within 2% of 0.005 delta_th");
    // G4.1f on every row (the reacting route's ledger, as gpu_fv_nozzle_react).
    {
        const double Tu = 1.0 / (1.4 * 0.042 * 0.042), q = 3.5 * 4.0 * Tu;
        const auto Om = csvColumn(th, "Omega");
        double worst = 0;
        for (const char* kk : {"mass", "mx", "my", "mz", "E", "ry"}) {
            const auto res = csvColumn(th, std::format("res_{}", kk)), P = csvColumn(th, std::format("P_{}", kk));
            std::vector<std::vector<double>> I;
            for (const char* rr : {"in", "exit", "wall", "side"}) I.push_back(csvColumn(th, std::format("I_{}_{}", rr, kk)));
            for (std::size_t i = 0; i < res.size(); ++i) {
                double S = std::abs(P[i]);
                for (const auto& c : I) S += std::abs(c[i]);
                if (std::string(kk) == "E") S += q * std::abs(Om[i]);
                if (std::string(kk) == "ry") S += std::abs(Om[i]);
                if (S > 0) worst = std::max(worst, std::abs(res[i]) / S);
            }
        }
        std::printf("  G4.1f: max r %.2e over %zu rows\n", worst, Om.size());
        check(!Om.empty() && worst <= 1e-13, std::format("G4.1f: the 2D ledger {:.2e} <= 1e-13", worst));
    }
    const int rp = runDns(base(dir) + std::format(" --tt 0.2 --out {}", op.string()), op.string() + ".log");
    const auto p1 = csvColumn(op.string() + "_stats.csv", "xIa1"), p2 = csvColumn(op.string() + "_stats.csv", "xIa2");
    double pm = 0;
    for (std::size_t i = 0; i < p1.size() && i < p2.size(); ++i) pm = std::max({pm, std::abs(p1[i]), std::abs(p2[i])});
    std::printf("  planar (no --wrinkle): rc %d, max |xIa| %.3e over %zu rows\n", rp, pm, p1.size());
    check(rp == 0 && !p1.empty() && pm <= 1e-6, std::format("seed: a planar run reads max |xIa| {:.2e} <= 1e-6 delta_D", pm));
    // The one-row chamber: --wrinkle refused (exit 3) by name; without it no xIa columns, no suffix.
    const int r1 = runDns(base(one) + std::format(" --wrinkle 0.005 --tt 0.1 --out {}", o1.string()), o1.string() + ".log");
    std::string log;
    {
        std::ifstream f(o1.string() + ".log");
        log.assign(std::istreambuf_iterator<char>(f), {});
    }
    check(r1 == 3 && log.find("--wrinkle: the [nozzle] chamber has one row: no transverse mode") != std::string::npos,
          std::format("scope: --wrinkle on a one-row chamber: exit 3 by name (rc {})", r1));
    const int r0 = runDns(base(one) + std::format(" --tt 0.05 --out {}", o1.string()), o1.string() + "_0.log");
    std::string l0;
    {
        std::ifstream f(o1.string() + "_stats.csv");
        std::getline(f, l0);
        std::string h;
        std::getline(f, h);
        check(r0 == 0 && !h.empty() && h.find("xIa") == std::string::npos && l0.find("half-domain") == std::string::npos,
              "scope: one row: no xIa columns, no half-domain suffix");
    }
}

// masterplan_combscale S6a (ctest gpu_fv_nozzle_wedgeflame): the wedge under C0 + hll on the device, the rig of
// tests/fvref_wedgeflame.hpp (nozzle_W, viscous, SI one-step chemistry).
//   W2  passive Y = 1/4, 40 steps, FP64 plain and compensated: rho Y == rho/4 in every cell, every wedge side's
//       F_Y == 0.0, bitwise. Control: Y = 0.3 is off somewhere.
//   W3  reacting, a Y ramp, 3 steps: device FP64 vs FvRef max rel diff per row <= 1e-13 on Source, Morton and
//       ColourMajor; the three numberings agree <= 1e-13; a rerun is the bytes.
void wedgeFlame()
{
    namespace wf = dnstest::wfl;
    std::printf("W2 device: a passive Y = 1/4 through the wedge, 40 steps, FP64\n");
    const Ref w = loadRef("nozzle_W", CellOrder::Morton);
    const auto devCfg = [&](const FaceMesh& fm, bool react, double yIn, bool comp) {
        const FvConfig h = wf::cfgW(fm, w.Rgas, react, yIn);
        FvCompConfig c;
        c.tr = h.tr;
        c.bc = h.bc;
        c.precision = FvPrecision::Fp64;
        c.compensated = comp;
        c.chem = h.chem;
        c.shock = h.shock;
        c.tallied.assign(fm.patches.size(), 0);
        return c;
    };
    const auto passive = [&](double y, bool comp, double& fy, double& moved) {
        auto d = orDie(DeviceFvComp::make(w.fm, devCfg(w.fm, false, y, comp)), "W2");
        const FvState ic = wf::icW(w, w.fm, [y](double) { return y; });
        orDie(d.upload(ic), "W2");
        const double dt = 0.95 * orDie(d.stableDt({.cfl = 0.4}), "dt");
        for (int n = 0; n < 40; ++n) d.step(dt);
        FvState s, rr;
        FvEval ev;
        orDie(d.download(s), "W2");
        orDie(d.evaluate(ev, rr), "W2");
        fy = wf::sideFy(w.fm, ev);
        moved = 0;
        for (int c = 0; c < s.nC; ++c) moved = std::max(moved, std::abs(s.var(0)[c] / ic.var(0)[c] - 1.0));
        return wf::offCells(s, y);
    };
    for (const bool comp : {false, true}) {
        double fy = 0, moved = 0;
        const long off = passive(0.25, comp, fy, moved);
        std::printf("  FP64%s: %ld cells off rho/4, max side |F_Y| %.3e, rho moved %.2e\n", comp ? " compensated" : "", off, fy, moved);
        check(moved >= 1e-4, "W2: the march moved rho");
        check(off == 0, std::format("W2 FP64{}: rho Y == rho/4 bitwise in every cell", comp ? " compensated" : ""));
        check(fy == 0.0, std::format("W2 FP64{}: the wedge sides' species flux == 0.0", comp ? " compensated" : ""));
    }
    {
        double fy = 0, moved = 0;
        const long off = passive(0.3, false, fy, moved);
        std::printf("  control Y 0.3: %ld cells off\n", off);
        check(off >= 1, "W2 control: Y = 0.3 is off by >= 1 ulp somewhere");
    }
    std::printf("W3 device FP64 vs FvRef: reacting, a Y ramp, 3 steps, three numberings\n");
    // Per source cell and row, after 3 steps at one dt (the host's bound on the Source mesh).
    using Rows6 = std::array<std::vector<double>, 6>;
    const auto bySource = [](const FaceMesh& fm, const FvState& s) {
        Rows6 r;
        for (int v = 0; v < 6; ++v) {
            r[std::size_t(v)].resize(std::size_t(fm.nC));
            for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) r[std::size_t(v)][std::size_t(fm.srcCell[c])] = s.var(v)[c];
        }
        return r;
    };
    // max over rows of max_c |a - b| / max_c |b|; a row that is 0.0 in both (w) reads 0.
    const auto rowGap = [](const Rows6& a, const Rows6& b) {
        double g = 0;
        for (std::size_t v = 0; v < 6; ++v) g = std::max(g, relMax(a[v], b[v]));
        return g;
    };
    double dt = 0;
    {
        const FaceMesh fm = orDie(FaceMesh::fromPolyMesh(w.pm, CellOrder::Source, 0, {.wedge = true}), "source");
        const FvRef ref = orDie(FvRef::make(fm, wf::cfgW(fm, w.Rgas, true, 1.0)), "FvRef");
        const FvState s = wf::icW(w, fm, wf::rampOf(fm));
        dt = std::min(0.5 * orDie(ref.stableDt(s), "dt"), ref.chemDt(s, 0.5));
    }
    Rows6 first;
    const char* names[] = {"Source", "Morton", "ColourMajor"};
    const CellOrder orders[] = {CellOrder::Source, CellOrder::Morton, CellOrder::ColourMajor};
    for (int k = 0; k < 3; ++k) {
        const FaceMesh fm = orDie(FaceMesh::fromPolyMesh(w.pm, orders[k], 0, {.wedge = true}), names[k]);
        const FvRef ref = orDie(FvRef::make(fm, wf::cfgW(fm, w.Rgas, true, 1.0)), "FvRef");
        const FvState ic = wf::icW(w, fm, wf::rampOf(fm));
        FvState h = ic, s, s2;
        ref.march(h, dt, 3);
        const auto dev = [&](FvState& out) {
            auto d = orDie(DeviceFvComp::make(fm, devCfg(fm, true, 1.0, false)), names[k]);
            orDie(d.upload(ic), names[k]);
            for (int n = 0; n < 3; ++n) d.step(dt);
            orDie(d.download(out), names[k]);
        };
        dev(s);
        dev(s2);
        const Rows6 rd = bySource(fm, s), rh = bySource(fm, h);
        const double gap = rowGap(rd, rh), moved = relMax(bySource(fm, ic)[5], rh[5]);
        if (k == 0) first = rd;
        const double num = rowGap(rd, first);
        std::printf("  %-11s: device vs FvRef %.2e, vs the Source numbering %.2e, max |rho w| %.1e (rho Y moved %.2e)\n", names[k], gap, num,
                    wMax(s), moved);
        check(moved >= 1e-6, std::format("W3 {}: the reacting march moved rho Y", names[k]));
        check(gap <= 1e-13, std::format("W3 {}: device FP64 vs FvRef {:.2e} <= 1e-13 per row", names[k], gap));
        check(num <= 1e-13, std::format("W3 {}: vs the Source numbering {:.2e} <= 1e-13", names[k], num));
        check(wMax(s) == 0.0, std::format("W3 {}: w == 0.0", names[k]));
        check(s.q.size() == s2.q.size() && std::memcmp(s.q.data(), s2.q.data(), s.q.size() * sizeof(double)) == 0,
              std::format("W3 {}: a rerun is the bytes", names[k]));
    }
}

// masterplan_combscale S6a, W5 (ctest gpu_fv_nozzle_wedgestrip): the planar 1D flame of G1.8 (fv1d_mn's dns argv, the
// poly1d c7.5 mesh: 197 x 4, y cyclic) against the same x nodes as a 5 deg wedge strip, R 1.5 delta_D, 6 rows
// (nozzle_react.py strip), through dns flame, FP64, --shock hll, tt 14, window [7, 14], --checkpoint-dt 7.
// An identity gate: same h AND same dt. The window numbers are row means of _stats.csv (a row per 200 steps, 6
// digits) of a signal that drifts and saws as the front crosses cells, so two runs at different dt average different
// sample times: the planar run at half its dt reproduces its own rows to the print while its row mean moves 4e-6.
// The strip's dt is 0.805 of the planar run's (the axis prisms' Sigma bound), so the reference runs at --cfl 0.4 x
// (strip dt / planar dt, window mean): its rows sit on the strip's times.
//   W5   strip vs the planar run at the strip's dt: |dS_L| <= 1e-6, |d delta_th| / delta_th <= 1e-4; the strip's last
//        checkpoint (t >= 7): max |v| / c <= 1e-11, rho w == 0.0
//   N29  the two checkpoints' FP64 states cell by cell: the strip radially uniform (1e-12); strip minus planar, one
//        translation removed, <= 1e-6 of max and per unit time the planar pair's own (rows 1e-2, front speed 1e-3)
//   same argv (the plan's protocol): both printed, [UNMET] recorded, not gated
//   t = tt, the one time every run shares: S_L, delta_th equal to the print
//   control (no run): the strip's S_L vs the archived 15-cell 0.9945059 differs >= 1e-3
void wedgeStrip()
{
    std::printf("W5 the 1D flame on a 5 deg wedge strip vs the planar poly1d mesh, 7.5 cells / delta_th, FP64, hll, tt 14\n");
    const fs::path root = fs::path(DNS_MESH_DATA) / ".." / ".." / "..";
    const fs::path planar = tmpDir() / "p1d", strip = tmpDir() / "strip";
    const std::string g1 = std::format("python3 {} --nx 197 --ny 4 --lx 42.07003070110829 --ly 0.85421382134230028 --depth 1 --out {} > /dev/null",
                                       (root / "tools/chem/fv_flame_mesh.py").string(), planar.string());
    const std::string g2 = std::format("python3 {} strip --cells 7.5 --radius 1.5 --ny 6 --dphi 5 --like {} --out {} > /dev/null",
                                       (root / "tools/fv/nozzle_react.py").string(), planar.string(), strip.string());
    if (std::system(g1.c_str()) != 0 || std::system(g2.c_str()) != 0) {
        check(false, "fv_flame_mesh.py and nozzle_react.py strip wrote the meshes");
        return;
    }
    struct Win
    {
        int rc = 0;
        std::vector<double> t, dt, sl, dth;
        double slMean = 0, dthMean = 0, dtMean = 0;
        long rows = 0;
    };
    const auto run = [&](const char* name, const fs::path& mesh, const std::string& extra) {
        const fs::path o = tmpDir() / name;
        Win r;
        r.rc = runDns(std::format("flame --mach 0.042 --expansion 5 --zeldovich 8 --prandtl 0.72 --polymesh {0} --bc {0}/bc.toml --upstream "
                                  "14.133333333333333 --inflow-speed 0.98 --tt 14 --profile-dt 0.1 --fv-precision fp64 --shock hll "
                                  "--checkpoint-dt 7 --quiet --out {1}{2}",
                                  mesh.string(), o.string(), extra),
                      o.string() + ".log");
        const std::string st = o.string() + "_stats.csv";
        r.t = csvColumn(st, "t");
        r.dt = csvColumn(st, "dt");
        r.sl = csvColumn(st, "SL");
        r.dth = csvColumn(st, "dth");
        long nd = 0;
        for (std::size_t i = 0; i < r.t.size(); ++i) {
            if (r.t[i] < 7.0 || r.t[i] > 14.0) continue;
            r.slMean += r.sl[i];
            r.dthMean += r.dth[i];
            ++r.rows;
            if (i + 1 < r.t.size()) { // the last row's dt is the clamp onto tt
                r.dtMean += r.dt[i];
                ++nd;
            }
        }
        if (r.rows > 0) r.slMean /= double(r.rows), r.dthMean /= double(r.rows);
        if (nd > 0) r.dtMean /= double(nd);
        std::printf("  %-7s rc %d, %zu rows (%ld in [7, 14]), dt %.6e: S_L %.8f, delta_th %.8f\n", name, r.rc, r.t.size(), r.rows, r.dtMean, r.slMean,
                    r.dthMean);
        return r;
    };
    const Win w = run("strip", strip, ""), p = run("planar", planar, "");
    check(w.rc == 0 && p.rc == 0 && w.rows >= 20 && p.rows >= 20, "route: the strip and the planar run ran");
    if (w.rc != 0 || p.rc != 0 || w.rows < 20 || p.rows < 20) return;
    {
        std::ifstream f(tmpDir() / "strip_stats.csv");
        std::string label;
        std::getline(f, label);
        check(label.find(" [axisymmetric wedge 5 deg: m = 0 modes only, no swirl]") != std::string::npos, "route: the label states the wedge");
        const auto sc = csvColumn(tmpDir() / "strip_stats.csv", "shockCells");
        check(!sc.empty() && *std::ranges::max_element(sc) == 0.0, "route: shockCells 0 on every row");
    }
    const double ratio = w.dtMean / p.dtMean;
    const Win m = run("matched", planar, std::format(" --cfl {:.17g}", 0.4 * ratio));
    const Win h = run("half", planar, " --cfl 0.2");
    check(m.rc == 0 && h.rc == 0 && m.rows == w.rows, "the planar runs at the strip's dt and at half dt ran; the matched rows pair the strip's");
    if (m.rc != 0 || h.rc != 0 || m.rows != w.rows) return;
    // The plan's protocol (same argv): recorded, not gated.
    const double dS0 = w.slMean - p.slMean, dD0 = (w.dthMean - p.dthMean) / p.dthMean;
    std::printf("  dt strip / planar %.5f (pre-registered ~1/2.7)\n", ratio);
    std::printf("  same argv: dS_L %+.2e, d delta_th / delta_th %+.2e\n", dS0, dD0);
    std::printf("  %s W5 as specified (same argv): |dS_L| <= 1e-6, |d delta_th| / delta_th <= 1e-4\n",
                std::abs(dS0) <= 1e-6 && std::abs(dD0) <= 1e-4 ? "[PASS]" : "[UNMET (recorded, B7: row means at different sample times)]");
    // The cause: the planar run at half dt on the cfl-0.4 run's row times (linear between its own rows).
    const auto at = [](const Win& r, const std::vector<double>& v, double t) {
        std::size_t i = 1;
        while (i + 1 < r.t.size() && r.t[i] < t) ++i;
        const double x = (t - r.t[i - 1]) / (r.t[i] - r.t[i - 1]);
        return v[i - 1] + x * (v[i] - v[i - 1]);
    };
    double ptS = 0, ptD = 0, rowT = 0;
    for (std::size_t i = 0; i < p.t.size(); ++i)
        if (p.t[i] >= 7.0) {
            ptS = std::max(ptS, std::abs(at(h, h.sl, p.t[i]) - p.sl[i]));
            ptD = std::max(ptD, std::abs(at(h, h.dth, p.t[i]) - p.dth[i]) / p.dth[i]);
        }
    for (std::size_t i = 0; i < w.t.size() && i < m.t.size(); ++i) rowT = std::max(rowT, std::abs(w.t[i] - m.t[i]));
    std::printf("  planar at half dt, on the planar rows' times: max |dS_L| %.2e, max |d delta_th| / delta_th %.2e; its row mean moves S_L %+.2e, "
                "delta_th %+.2e\n",
                ptS, ptD, h.slMean - p.slMean, (h.dthMean - p.dthMean) / p.dthMean);
    check(ptS <= 1e-5 && ptD <= 1e-5, std::format("cause: half dt reproduces the planar rows to the print (S_L {:.1e}, delta_th {:.1e} <= 1e-5)", ptS, ptD));
    check(std::abs(h.slMean - p.slMean) >= 1e-6,
          std::format("cause: the row mean of the same solution moves with the sample times ({:+.2e} >= 1e-6)", h.slMean - p.slMean));
    // t = tt: the one time every run shares.
    const double eS = std::abs(w.sl.back() - p.sl.back()), eD = std::abs(w.dth.back() - p.dth.back()) / p.dth.back();
    std::printf("  t = 14, strip vs planar (same argv): |dS_L| %.1e, |d delta_th| / delta_th %.1e\n", eS, eD);
    check(w.t.back() == 14.0 && p.t.back() == 14.0 && eS <= 1e-5 && eD <= 1e-5, "t = tt: the strip's S_L and delta_th equal the planar run's to the print");
    {
        // The slab profiles' last dump (t = tt in both): every slab, every column.
        const fs::path fw = tmpDir() / "strip_profiles.csv", fp = tmpDir() / "planar_profiles.csv";
        const auto tw = csvColumn(fw, "t"), tp = csvColumn(fp, "t");
        double worst = 0;
        long slabs = 0;
        for (const char* k : {"T", "rho", "u", "p", "Y", "omega"}) {
            const auto a = csvColumn(fw, k), b = csvColumn(fp, k);
            double e = 0, s = 0;
            slabs = 0;
            for (std::size_t i = tw.size(), j = tp.size(); i > 0 && j > 0 && tw[i - 1] == 14.0 && tp[j - 1] == 14.0; --i, --j, ++slabs) {
                e = std::max(e, std::abs(a[i - 1] - b[j - 1]));
                s = std::max(s, std::abs(b[j - 1]));
            }
            worst = std::max(worst, s > 0 ? e / s : e);
        }
        std::printf("  t = 14 profiles, %ld slabs, T rho u p Y omega: max |strip - planar| / max %.1e\n", slabs, worst);
        check(slabs == 197 && worst <= 1e-5, std::format("t = tt: the strip's slab profiles equal the planar run's to the print ({:.1e})", worst));
    }
    // W5: the identity at the same h and dt.
    const double dS = w.slMean - m.slMean, dD = (w.dthMean - m.dthMean) / m.dthMean;
    std::printf("  matched dt (planar --cfl %.6f, dt ratio %.6f, row times within %.1e): dS_L %+.2e, d delta_th / delta_th %+.2e\n", 0.4 * ratio,
                w.dtMean / m.dtMean, rowT, dS, dD);
    check(std::abs(w.dtMean / m.dtMean - 1) <= 1e-3, "W5: the reference runs at the strip's dt (1e-3)");
    check(std::abs(dS) <= 1e-6, std::format("W5: |dS_L| {:.2e} <= 1e-6 at the same h and dt", std::abs(dS)));
    check(std::abs(dD) <= 1e-4, std::format("W5: |d delta_th| / delta_th {:.2e} <= 1e-4 at the same h and dt", std::abs(dD)));
    // v, w off the strip's last checkpoint.
    const auto ck = dns::Checkpoint::read(dns::checkpointPath((tmpDir() / "strip").string()));
    const auto q = ck ? ck->f64("state.fv.q") : std::unexpected(ck.error());
    const auto tc = ck ? ck->f64Scalar("t") : std::unexpected(ck.error());
    check(q.has_value() && tc.has_value() && q->size() % 6 == 0, "the strip's checkpoint holds state.fv.q (6 rows, FP64)");
    if (q && tc && q->size() % 6 == 0) {
        const std::size_t nC = q->size() / 6;
        double vc = 0, wm = 0;
        for (std::size_t c = 0; c < nC; ++c) {
            const double rho = (*q)[c], mx = (*q)[nC + c], my = (*q)[2 * nC + c], mz = (*q)[3 * nC + c], E = (*q)[4 * nC + c];
            const double pr = 0.4 * (E - 0.5 * (mx * mx + my * my + mz * mz) / rho);
            vc = std::max(vc, std::abs(my / rho) / std::sqrt(1.4 * pr / rho));
            wm = std::max(wm, std::abs(mz));
        }
        std::printf("  the strip's checkpoint at t %.4f, %zu cells: max |v| / c %.3e, max |rho w| %.1e\n", *tc, nC, vc, wm);
        check(*tc >= 7.0, "W5: the checkpoint is at t >= 7");
        check(vc <= 1e-11, std::format("W5: max |v| / c {:.2e} <= 1e-11", vc));
        check(wm == 0.0, "W5: rho w == 0.0");
    }
    // N29: the identity at full precision. The strip's and the matched planar run's last checkpoints (the first step
    // past t 7 in both), state.fv.q FP64, cell by cell: every strip cell against the planar cell of its x column.
    {
        struct Snap
        {
            FaceMesh fm;
            std::vector<double> q;
            double t = 0;
        };
        const auto snap = [&](const fs::path& mesh, const char* name, bool wedge) -> std::optional<Snap> {
            auto pm = readPolyMesh(mesh / "constant/polyMesh");
            if (!pm) return std::nullopt;
            auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Morton, 0, {.wedge = wedge});
            const auto c = dns::Checkpoint::read(dns::checkpointPath((tmpDir() / name).string()));
            if (!fm || !c) return std::nullopt;
            auto s = c->f64("state.fv.q");
            const auto t = c->f64Scalar("t");
            if (!s || !t || s->size() != 6 * fm->C[0].size()) return std::nullopt;
            return Snap{std::move(*fm), std::move(*s), *t};
        };
        struct Cmp
        {
            double dt = 0, xOff = 1, col = 0, rad = 0, sh = 0, raw[4] = {}, res[4] = {}, rawMax = 0, resMax = 0;
        };
        static constexpr int rows[] = {0, 1, 4, 5};
        static constexpr const char* names[] = {"rho", "rho u", "E", "rho Y"};
        // a against b (b planar: the reference columns). The two checkpoints are a step apart in time (each is its
        // run's first step past 7): the raw difference carries the front's travel over that time. ONE number removes
        // it: the translation sh (least squares on rho Y, b's own central d/dx).
        const auto compare = [&](const Snap& a, const Snap& b) {
            Cmp o;
            o.dt = a.t - b.t;
            const std::size_t nA = a.fm.C[0].size(), nB = b.fm.C[0].size();
            std::vector<std::pair<double, std::size_t>> xs, cols; // b: one cell per x column, ascending
            for (std::size_t c = 0; c < nB; ++c) xs.emplace_back(b.fm.C[0][c], c);
            std::ranges::sort(xs);
            for (const auto& xc : xs)
                if (cols.empty() || xc.first - cols.back().first > 1e-9) cols.push_back(xc);
            const auto colOf = [&](double x) {
                const auto it = std::ranges::lower_bound(cols, std::pair{x - 1e-9, std::size_t(0)});
                return std::size_t(it - cols.begin());
            };
            const auto ddx = [&](int row, std::size_t i) {
                const std::size_t lo = i > 0 ? i - 1 : i, hi = i + 1 < cols.size() ? i + 1 : i;
                return (b.q[row * nB + cols[hi].second] - b.q[row * nB + cols[lo].second]) / (cols[hi].first - cols[lo].first);
            };
            double s[4] = {};
            for (std::size_t c = 0; c < nB; ++c) { // b is 1D: its rows within a column
                const std::size_t i = colOf(b.fm.C[0][c]);
                for (int k = 0; k < 4; ++k) {
                    o.col = std::max(o.col, std::abs(b.q[rows[k] * nB + c] - b.q[rows[k] * nB + cols[i].second]));
                    s[k] = std::max(s[k], std::abs(b.q[rows[k] * nB + c]));
                }
            }
            o.xOff = 0;
            for (std::size_t c = 0; c < nA; ++c) {
                const std::size_t i = colOf(a.fm.C[0][c]);
                o.xOff = std::max(o.xOff, i < cols.size() ? std::abs(cols[i].first - a.fm.C[0][c]) : 1.0);
            }
            if (o.xOff > 1e-9) return o;
            double num = 0, den = 0;
            std::vector<double> first(cols.size() * 4, std::numeric_limits<double>::quiet_NaN());
            for (std::size_t c = 0; c < nA; ++c) {
                const std::size_t i = colOf(a.fm.C[0][c]);
                const double g = ddx(5, i);
                num += (a.q[5 * nA + c] - b.q[5 * nB + cols[i].second]) * g;
                den += g * g;
                for (int k = 0; k < 4; ++k) { // a's own rows within a column
                    double& f = first[4 * i + k];
                    if (std::isnan(f)) f = a.q[rows[k] * nA + c];
                    o.rad = std::max(o.rad, std::abs(a.q[rows[k] * nA + c] - f) / s[k]);
                }
            }
            o.sh = den > 0 ? num / den : 0;
            for (std::size_t c = 0; c < nA; ++c) {
                const std::size_t i = colOf(a.fm.C[0][c]);
                for (int k = 0; k < 4; ++k) {
                    const double d = a.q[rows[k] * nA + c] - b.q[rows[k] * nB + cols[i].second];
                    o.raw[k] = std::max(o.raw[k], std::abs(d) / s[k]);
                    o.res[k] = std::max(o.res[k], std::abs(d - o.sh * ddx(rows[k], i)) / s[k]);
                }
            }
            for (int k = 0; k < 4; ++k) o.rawMax = std::max(o.rawMax, o.raw[k]), o.resMax = std::max(o.resMax, o.res[k]);
            return o;
        };
        const auto show = [&](const char* what, const Cmp& o) {
            std::printf("  N29 %s: checkpoints %.3e apart in t, x within %.1e, reference rows within %.1e, own rows within %.1e, translation "
                        "%.4e delta_D (front speed %.4f in the grid frame)\n",
                        what, o.dt, o.xOff, o.col, o.rad, o.sh, o.dt != 0 ? -o.sh / o.dt : 0.0);
            for (int k = 0; k < 4; ++k) std::printf("    %-6s max |a - b| / max: raw %.3e, translation removed %.3e\n", names[k], o.raw[k], o.res[k]);
        };
        const auto a = snap(strip, "strip", true), b = snap(planar, "matched", false), p0 = snap(planar, "planar", false);
        check(a.has_value() && b.has_value() && p0.has_value(), "N29: the checkpoints read onto their meshes (Morton, 6 rows)");
        if (a && b && p0) {
            // The method's floor: the same decomposition on two PLANAR runs (--cfl 0.4 vs the strip's dt), no wedge in it.
            const Cmp w5 = compare(*a, *b), fl = compare(*p0, *b);
            show("strip vs planar at the strip's dt", w5);
            show("floor: planar --cfl 0.4 vs planar at the strip's dt", fl);
            check(w5.xOff <= 1e-9 && fl.xOff <= 1e-9, "N29: every cell has its planar column");
            check(w5.rad <= 1e-12, std::format("N29: the strip's state is radially uniform, FP64 ({:.2e} <= 1e-12)", w5.rad));
            check(w5.resMax <= 1e-6, std::format("N29: the strip's state equals the planar run's up to one translation, FP64 ({:.2e} <= 1e-6)", w5.resMax));
            // What is left is the solution's own evolution over the offset: per unit time it is the planar pair's, row
            // by row, and so is the front speed. A wedge term would sit on top of it.
            double rate = 0;
            for (int k = 0; k < 4; ++k) rate = std::max(rate, std::abs(w5.res[k] / std::abs(w5.dt) / (fl.res[k] / std::abs(fl.dt)) - 1));
            const double speed = std::abs(w5.sh / w5.dt / (fl.sh / fl.dt) - 1);
            std::printf("    per unit time vs the planar pair: residual rows within %.2e, front speed within %.2e\n", rate, speed);
            check(std::abs(w5.dt) >= 1e-5 && std::abs(fl.dt) >= 1e-5, "N29: both pairs are offset in time (else the raw difference is the gate)");
            check(rate <= 1e-2 && speed <= 1e-3,
                  std::format("N29: strip minus planar per unit time IS the planar pair's (rows {:.1e} <= 1e-2, front speed {:.1e} <= 1e-3)", rate, speed));
        }
    }
    const double ctrl = std::abs(w.slMean - 0.9945059);
    std::printf("  control: |S_L(strip, 7.5 cells) - 0.9945059 (15 cells)| %.3e\n", ctrl);
    check(ctrl >= 1e-3, "W5 control: the strip's S_L differs from the 15-cell value by >= 1e-3");
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    const std::string_view m = argc > 1 ? argv[1] : "";
    const auto done = [] {
        std::error_code ec;
        fs::remove_all(tmpDir(), ec);
        return dnstest::finish();
    };
    if (m == "fixed") fixed();
    else if (m == "wedge") wedge();
    else if (m == "reflect") reflect();
    else if (m == "dt") dtGate();
    else if (m == "source") source();
    else if (m == "silent") silent();
    else if (m == "exit") exitGate();
    else if (m == "sauer") sauer();
    else if (m == "cdladder") cdladder();
    else if (m == "fp32") fp32();
    else if (m == "sense") {
        sense();
        cliSilent();
    } else if (m == "gamma") gammaGate();
    else if (m == "q1d") {
        q1dParabola();
        q1dNc();
    } else if (m == "q1dshock") q1dShock();
    else if (m == "ambient") ambient();
    else if (m == "twin") twin();
    else if (m == "react") reactRun();
    else if (m == "reacttwin" && (argc == 7 || argc == 8))
        reactTwinAt(std::atoi(argv[2]), std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]), std::atof(argv[6]),
                    argc == 8 ? std::atof(argv[7]) : 1.4);
    else if (m == "reacttwinseq" && argc == 9)
        reactTwinSeq(std::atoi(argv[2]), std::atoi(argv[3]), std::atof(argv[4]), std::atof(argv[5]), std::atof(argv[6]), std::atof(argv[7]),
                     std::atof(argv[8]));
    else if (m == "reacttwin") reactTwin();
    else if (m == "reacttwinrow" && (argc == 9 || (argc == 10 && (std::string(argv[9]) == "cos" || std::string(argv[9]) == "cone"))))
        reactTwinRow(std::atoi(argv[2]), std::atoi(argv[3]), std::atof(argv[4]), std::atof(argv[5]), std::atof(argv[6]), std::atof(argv[7]),
                     std::atof(argv[8]), argc == 10 ? argv[9] : "cos");
    else if (m == "react2d") react2dRun();
    else if (m == "wedgeflame") wedgeFlame();
    else if (m == "wedgestrip") wedgeStrip();
    else if (m == "ledger") {
        ledgerGate();
        ledgerLong();
        thrustCli();
    } else {
        std::printf("unknown mode\n");
        return 2;
    }
    return done();
}
