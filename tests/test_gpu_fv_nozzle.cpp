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
#include "fv_comp_test_util.hpp"
#include "fvref_nozzle.hpp"

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
    const double icGap = vsFoam(f0, w);
    const auto t0 = std::chrono::steady_clock::now();
    const Marched m = marchTo(d, tEnd, 0.4, 16);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    check(m.err.empty(), std::format("{}: live ({})", name, m.err));
    FvState s;
    orDie(d.download(s), name);
    const Fields f = fieldsOf(w.fm, s, kGam, w.Rgas);
    const double e = vsFoam(f, w);
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
    else {
        std::printf("unknown mode\n");
        return 2;
    }
    return done();
}
