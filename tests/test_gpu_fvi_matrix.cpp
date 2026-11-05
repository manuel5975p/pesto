// mci: the generic gate matrix of the time-accurate incompressible FV stage (gpu/fv_incomp
// DeviceFvIncomp; docs/fvincomp.md "Incompressible combinations", docs/
// masterplan_multiphysics.md "Composition, not pairs"). Every LIVE set of the inventory
// (dns/fvimodules: each module alone and each pair, closed under the requirements) on
// four 3D meshes -- a uniform walled box, the same box rigidly rotated, a random +-0.25 h
// box, a cylinder O-grid -- each in Source / Morton / ColourMajor numbering:
//   (a) inert = the stage without the module, byte for byte: T (and gravity at betaG 0)
//       leaves momentum (U, p, phi) bitwise the cell without T, FP32 and FP64; LES sigma
//       on a 2D mesh (D_sigma == 0) is the molecular run bitwise, momentum and T, on the
//       non-orthogonal 2D meshes (an orthogonal molecular run skips the gradient, another
//       instantiation: n/a there). Other modules have no inert setting: their off IS
//       absence (tools/fv/comp_bit_identity.sh against the base binary)
//   (b) invariants on the closed set: rest (U 0) stays rest; with gravity the hydrostatic
//       state stays at rest (boxes: the conduction profile between the plates; O-grid:
//       uniform T != Tref, a linear p) -- uniform, rotated, O-grid to the solver's
//       tolerance, random stated; the divergence floor after 10 steps of a vortical state;
//       the skew convection's KE rate sum V u.C(u) and T's sum V T C(T) 0 to rounding
//   (c) FP32 vs FP64: stated, bounded
//   (d) Source / Morton / ColourMajor: 3 FP64 steps at tol 1e-12, mapped by srcCell, to
//       the solver's summation order (the CSR and the dot order move), bitwise stated
//   (e) rerun byte-identical, FP32 and FP64, 10 steps
// One row per (set, mesh). `markdown` prints the generated docs section; `physics` runs
// the live pairs' physics numbers (gpu_fvi_matrix_phys: Ra-Nu with LES).
#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvconv.hpp"
#include "dns/fvimodules.hpp"
#include "dns/fvincomp.hpp"
#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/la/graph.hpp"
#include "dns/ogrid.hpp"
#include "dns/polymesh.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

using namespace dns;
using dnstest::check;

namespace {

constexpr double kPi = std::numbers::pi;
constexpr std::array<CellOrder, 3> kOrders{CellOrder::Source, CellOrder::Morton, CellOrder::ColourMajor};
constexpr double kNu = 0.01, kKappa = 0.014, kCsSmag = 0.3, kCsSigma = 1.35;

template <class T>
T must(std::expected<T, std::string> e, const std::string& what)
{
    if (!e) {
        std::printf("FATAL: %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
    return std::move(*e);
}

using Mat = std::array<std::array<double, 3>, 3>;
constexpr Mat kId{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};

// Rodrigues: rotation by a about the unit axis n.
Mat rotation(std::array<double, 3> n, double a)
{
    const double l = std::hypot(n[0], n[1], n[2]);
    for (double& x : n) x /= l;
    const double c = std::cos(a), s = std::sin(a), t = 1 - c;
    return {{{t * n[0] * n[0] + c, t * n[0] * n[1] - s * n[2], t * n[0] * n[2] + s * n[1]},
             {t * n[0] * n[1] + s * n[2], t * n[1] * n[1] + c, t * n[1] * n[2] - s * n[0]},
             {t * n[0] * n[2] - s * n[1], t * n[1] * n[2] + s * n[0], t * n[2] * n[2] + c}}};
}

std::array<double, 3> rot(const Mat& R, const std::array<double, 3>& x)
{
    std::array<double, 3> y{};
    for (int i = 0; i < 3; ++i) y[i] = R[i][0] * x[0] + R[i][1] * x[1] + R[i][2] * x[2];
    return y;
}

std::array<double, 3> rotT(const Mat& R, const std::array<double, 3>& x)
{
    std::array<double, 3> y{};
    for (int i = 0; i < 3; ++i) y[i] = R[0][i] * x[0] + R[1][i] * x[1] + R[2][i] * x[2];
    return y;
}

struct MeshCase
{
    std::string name;
    std::vector<FaceMesh> fm; // kOrders (3D), or Morton only (the 2D inert meshes)
    Mat R = kId;              // the rotated box: reference coordinates = R^T x
    bool box = true;          // plates floor (T 0.5) / lid (T -0.5) at y_ref = 0 / 1; else the O-grid
    double Lx = 1, Lz = 1;
};

// conv's box (periodic x, z; plates at y 0 and 1), 3D unless nz == 1.
CurvMesh boxCurv(int nx, int ny, int nz, FvPerturb kind, double amp, Grid& g)
{
    const double hy = 1.0 / ny;
    g = Grid::make(nx, ny + 2, nz, 1.5, 1.0 + 2 * hy, nz == 1 ? 1.0 : 1.0);
    return fvConvCurv(g, false, kind, amp);
}

FaceMesh rotatedBox(const CurvMesh& cm, const Grid& g, const Mat& R, CellOrder o)
{
    PolyMesh pm = must(FaceMesh::polyOf(cm, fvConvSpec(g, false)), "polyOf");
    for (std::size_t i = 0; i < pm.px.size(); ++i) {
        const auto y = rot(R, {pm.px[i], pm.py[i], pm.pz[i]});
        pm.px[i] = y[0];
        pm.py[i] = y[1];
        pm.pz[i] = y[2];
    }
    for (FoamPatch& p : pm.man.patches)
        if (p.separation) p.separation = rot(R, *p.separation);
    return must(FaceMesh::fromPolyMesh(pm, o), "rotated box");
}

std::expected<OGrid, std::string> ogrid(int nz)
{
    const Grid g = ogridGrid(16, 8, nz, 0.5, 4.0, 1.0);
    return buildOGrid(g, [](double x, double y, double) { return std::hypot(x, y) - 0.5; },
                      {.rFar = 4.0, .beta = 1.5, .span = 1.0});
}

FaceMesh ogridMesh(const OGrid& og, CellOrder o)
{
    const Grid& g = og.mesh.g;
    const FoamPatchType zt = g.nz == 1 ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    const StructuredSpec sp{.lo = {0, 1, 0},
                            .hi = {g.nx - 1, g.ny - 2, g.nz - 1},
                            .name = {"ilo", "ihi", "wall", "far", "zlo", "zhi"},
                            .type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Wall,
                                     FoamPatchType::Patch, zt, zt}};
    return must(FaceMesh::fromCurvMesh(og.mesh, sp, o), "O-grid");
}

std::vector<MeshCase> meshes3d()
{
    std::vector<MeshCase> v;
    Grid g;
    const CurvMesh uni = boxCurv(8, 6, 4, FvPerturb::None, 0, g);
    MeshCase u{.name = "uniform box 8x6x4", .fm = {}, .Lx = 1.5, .Lz = 1.0};
    for (const CellOrder o : kOrders) u.fm.push_back(must(FaceMesh::fromCurvMesh(uni, fvConvSpec(g, false), o), "box"));
    v.push_back(std::move(u));
    MeshCase r{.name = "rotated box (0.7 rad about (1,2,3))", .fm = {}, .R = rotation({1, 2, 3}, 0.7), .Lx = 1.5, .Lz = 1.0};
    for (const CellOrder o : kOrders) r.fm.push_back(rotatedBox(uni, g, r.R, o));
    v.push_back(std::move(r));
    const CurvMesh rnd = boxCurv(8, 6, 4, FvPerturb::Random, 0.25, g);
    MeshCase q{.name = "random 0.25 h box", .fm = {}, .Lx = 1.5, .Lz = 1.0};
    for (const CellOrder o : kOrders) q.fm.push_back(must(FaceMesh::fromCurvMesh(rnd, fvConvSpec(g, false), o), "random"));
    v.push_back(std::move(q));
    const OGrid og = must(ogrid(4), "O-grid");
    MeshCase c{.name = "O-grid 16x8x4", .fm = {}, .box = false};
    for (const CellOrder o : kOrders) c.fm.push_back(ogridMesh(og, o));
    v.push_back(std::move(c));
    return v;
}

std::vector<MeshCase> meshes2d()
{
    std::vector<MeshCase> v;
    Grid g;
    const CurvMesh rnd = boxCurv(10, 8, 1, FvPerturb::Random, 0.25, g);
    v.push_back({.name = "random 0.25 h box 2D", .fm = {must(FaceMesh::fromCurvMesh(rnd, fvConvSpec(g, false), CellOrder::Morton), "r2")},
                 .Lx = 1.5});
    const CurvMesh sine = boxCurv(10, 8, 1, FvPerturb::Sine, 0.08, g);
    v.push_back({.name = "sine 0.08 box 2D", .fm = {must(FaceMesh::fromCurvMesh(sine, fvConvSpec(g, false), CellOrder::Morton), "s2")},
                 .Lx = 1.5});
    const OGrid og = must(ogrid(1), "O-grid 2D");
    v.push_back({.name = "O-grid 2D", .fm = {ogridMesh(og, CellOrder::Morton)}, .box = false});
    return v;
}

// The --bc table. Boxes: the plates. O-grid open: a hot wall (T 0.5) and a far field u
// (1, 0, 0); closed: an adiabatic wall and a slip far circle.
FvIncompBcs bcsOf(const MeshCase& m, const FaceMesh& fm, bool closed)
{
    FvIncompBcs b;
    if (m.box)
        b.patches = {{.name = "floor", .kind = FvIncompBc::Wall, .hasT = true, .T = 0.5},
                     {.name = "lid", .kind = FvIncompBc::Wall, .hasT = true, .T = -0.5}};
    else if (closed)
        b.patches = {{.name = "wall", .kind = FvIncompBc::Wall}, {.name = "far", .kind = FvIncompBc::Slip}};
    else
        b.patches = {{.name = "wall", .kind = FvIncompBc::Wall, .hasT = true, .T = 0.5},
                     {.name = "far", .kind = FvIncompBc::Far, .u = {1.0, 0.0, 0.0}}};
    must(bindFvIncompBcs(b, fm).transform([] { return 0; }), "bind");
    return b;
}

bool has(const FviModuleSet& s, FviModule m) { return s.test(std::size_t(m)); }

FvIncompControls controlsOf(const FviModuleSet& s, const MeshCase& m, FvPrecision p, double tol)
{
    FvIncompControls c{.nu = kNu, .tol = tol, .maxIter = 400, .precision = p, .kappa = kKappa, .Tref = 0.0};
    if (has(s, FviModule::LesSmag)) c.les = kCsSmag;
    if (has(s, FviModule::LesSigma)) {
        c.les = kCsSigma;
        c.sgs = SgsModel::Sigma;
    }
    if (has(s, FviModule::Gravity)) c.betaG = rot(m.R, {0.0, 1.0, 0.0});
    return c;
}

enum class Ic { Vortical, Rest };

// Position only (reference coordinates), so every numbering gets the same numbers.
void icOf(const MeshCase& m, const FaceMesh& fm, Ic ic, std::vector<double>& U, std::vector<double>& T)
{
    const std::size_t nC = std::size_t(fm.nC);
    U.assign(3 * nC, 0.0);
    T.assign(nC, 0.0);
    for (std::size_t q = 0; q < nC; ++q) {
        const auto x = rotT(m.R, {fm.C[0][q], fm.C[1][q], fm.C[2][q]});
        std::array<double, 3> u{};
        if (m.box) {
            const double ax = 2 * kPi * x[0] / m.Lx, az = 2 * kPi * x[2] / m.Lz, ay = kPi * x[1];
            T[q] = 0.5 - x[1];
            if (ic == Ic::Vortical) {
                T[q] += 0.1 * std::sin(ax) * std::sin(ay) * (1 + 0.3 * std::cos(az));
                u = {0.3 * std::cos(ax) * std::sin(ay) + 0.05 * std::sin(az), -0.3 * std::sin(ax) * std::cos(ay),
                     0.1 * std::sin(ax + 0.5) * std::sin(ay)};
            }
        } else {
            T[q] = 0.25;
            if (ic == Ic::Vortical) {
                T[q] += 0.1 * std::sin(x[0]) * std::cos(x[1]);
                u = {1.0 + 0.1 * std::sin(x[1]) * std::cos(2 * kPi * x[2]), 0.1 * std::cos(x[0]),
                     0.05 * std::sin(x[0] + x[1])};
            }
        }
        const auto w = rot(m.R, u);
        for (std::size_t k = 0; k < 3; ++k) U[k * nC + q] = w[k];
    }
}

struct Run
{
    gpu::FvIncompState st;
    gpu::FvIncompDiag dg;
    gpu::FvIncompBudget bd;
    bool orthogonal = true;
};

// steps fixed steps of dt from ic; thermal: T rides (make given T0).
Run run(const MeshCase& m, int order, const FviModuleSet& s, FvPrecision p, double tol, Ic ic, bool closed, double dt,
        int steps, bool thermal, const FvIncompControls* override = nullptr)
{
    const FaceMesh& fm = m.fm[std::size_t(order)];
    const FvIncompBcs b = bcsOf(m, fm, closed);
    const FvIncompSetup su = must(buildFvIncompSetup(fm, b), "setup");
    std::vector<double> U, T;
    icOf(m, fm, ic, U, T);
    const FvIncompControls c = override ? *override : controlsOf(s, m, p, tol);
    auto d = must(gpu::DeviceFvIncomp::make(su, la::graphOf(fm), c, U, thermal ? T : std::vector<double>{}), "make");
    for (int i = 0; i < steps; ++i) must(d.step(dt).transform([](auto) { return 0; }), "step");
    return {.st = must(d.state(), "state"), .dg = must(d.diagnostics(), "diag"), .bd = must(d.budget(), "budget"),
            .orthogonal = su.orthogonal};
}

bool sameMomentum(const gpu::FvIncompState& a, const gpu::FvIncompState& b)
{
    return a.U == b.U && a.p == b.p && a.phi == b.phi && a.phiB == b.phiB;
}

double maxAbs(const std::vector<double>& v)
{
    double m = 0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

// One dt per mesh: the most demanding live set's bound, so every run of the mesh shares it.
double meshDt(const MeshCase& m)
{
    const FaceMesh& fm = m.fm[std::size_t(m.fm.size() > 1 ? 1 : 0)];
    const FvIncompBcs b = bcsOf(m, fm, false);
    const FvIncompSetup su = must(buildFvIncompSetup(fm, b), "setup");
    std::vector<double> U, T;
    icOf(m, fm, Ic::Vortical, U, T);
    FvIncompControls c = controlsOf(fviModuleSet({FviModule::LesSmag, FviModule::Temperature, FviModule::Gravity}), m,
                                    FvPrecision::Fp64, 1e-10);
    auto d = must(gpu::DeviceFvIncomp::make(su, la::graphOf(fm), c, U, T), "make");
    must(d.step(1e-4).transform([](auto) { return 0; }), "step"); // nut of a stage
    return 0.5 * must(d.stableDt(0.5), "dt");
}

std::vector<FviModuleSet> liveSets()
{
    std::vector<FviModuleSet> v;
    auto add = [&](FviModuleSet s) {
        s = fviModulesClosure(s);
        if (!fviModulesCheck(s)) return;
        if (std::ranges::find(v, s) == v.end()) v.push_back(s);
    };
    add({});
    for (std::size_t a = 0; a < kFviModuleCount; ++a) {
        FviModuleSet one;
        one.set(a);
        add(one);
        for (std::size_t b = a + 1; b < kFviModuleCount; ++b) {
            FviModuleSet two = one;
            two.set(b);
            add(two);
        }
    }
    std::ranges::sort(v, {}, [](const FviModuleSet& s) { return s.to_ullong(); });
    return v;
}

int runMatrix()
{
    const std::vector<FviModuleSet> sets = liveSets();
    std::vector<MeshCase> ms = meshes3d();
    const std::vector<MeshCase> m2 = meshes2d();
    constexpr int kSteps = 10;
    int rows = 0;
    std::printf("| Set | Mesh | (a) inert | (b) rest, div, KE | (c) fp32 | (d) numbering | (e) rerun |\n|---|---|---|---|---|---|---|\n");
    std::vector<double> dts;
    for (const MeshCase& m : ms) dts.push_back(meshDt(m));
    for (const FviModuleSet& s : sets) {
        const bool thermal = has(s, FviModule::Temperature), grav = has(s, FviModule::Gravity);
        const bool sigma = has(s, FviModule::LesSigma);
        for (std::size_t mi = 0; mi < ms.size(); ++mi) {
            const MeshCase& m = ms[mi];
            const double dt = dts[mi];
            const std::string tag = std::format("[{} on {}]", fviModuleSetName(s), m.name);
            std::string a, bcol, ccol, dcol, ecol;
            // (a) T, gravity at betaG 0: momentum bitwise the cell without them.
            if (thermal) {
                FviModuleSet base = s;
                base.reset(std::size_t(FviModule::Temperature));
                base.reset(std::size_t(FviModule::Gravity));
                bool all = true;
                for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64}) {
                    FvIncompControls c0 = controlsOf(s, m, p, 1e-6);
                    c0.betaG = {0.0, 0.0, 0.0};
                    const Run with = run(m, 1, s, p, 1e-6, Ic::Vortical, false, dt, kSteps, true, &c0);
                    const Run without = run(m, 1, base, p, 1e-6, Ic::Vortical, false, dt, kSteps, false);
                    const bool eq = sameMomentum(with.st, without.st);
                    if (!eq) {
                        double du = 0;
                        for (std::size_t i = 0; i < with.st.U.size(); ++i)
                            du = std::max(du, std::abs(with.st.U[i] - without.st.U[i]));
                        std::printf("    T passive %s: max |dU| %.3e\n", p == FvPrecision::Fp64 ? "fp64" : "fp32", du);
                    }
                    all = all && eq;
                }
                check(all, "(a) passive T leaves momentum bitwise, FP32 and FP64 " + tag);
                a += all ? "T passive: momentum bitwise" : "T passive: DIFFERS";
            }
            // (a) LES sigma on 2D (nut == 0): the molecular run, on the 2D meshes (the row of
            // the first 3D mesh carries them).
            if (sigma && mi == 0) {
                FviModuleSet base = s;
                base.reset(std::size_t(FviModule::LesSigma));
                for (const MeshCase& q : m2) {
                    const double d2 = meshDt(q);
                    const Run l = run(q, 0, s, FvPrecision::Fp32, 1e-6, Ic::Vortical, false, d2, kSteps, thermal);
                    const Run n = run(q, 0, base, FvPrecision::Fp32, 1e-6, Ic::Vortical, false, d2, kSteps, thermal);
                    if (n.orthogonal) {
                        a += std::format("; {}: n/a (orthogonal)", q.name);
                        continue;
                    }
                    const bool eq = sameMomentum(l.st, n.st) && l.st.T == n.st.T;
                    check(eq, std::format("(a) sigma on a 2D state is the molecular run bitwise [{} on {}]",
                                          fviModuleSetName(s), q.name));
                    a += std::format("; sigma 2D {}: {}", q.name, eq ? "bitwise" : "DIFFERS");
                }
            }
            if (a.empty()) a = "n/a (off = absence)";
            // (b) rest on the closed set.
            {
                const Run r = run(m, 1, s, FvPrecision::Fp64, 1e-12, Ic::Rest, true, dt, kSteps, thermal);
                const double um = maxAbs(r.st.U);
                const bool exactRest = !grav;
                const bool tight = m.name.find("random") == std::string::npos;
                const double bound = exactRest ? 0.0 : tight ? 1e-12 : 5e-3; // random: the stated residual (measured 1.6e-3)
                check(um <= bound, std::format("(b) rest stays rest (|u| {:.1e} <= {:.0e}) {}", um, bound, tag));
                bcol += std::format("rest max u {:.1e}{}", um, !exactRest && !tight ? " (stated)" : "");
            }
            // (b) divergence floor, KE; (c) fp32 vs fp64; (e) rerun.
            Run r64, r32;
            {
                r64 = run(m, 1, s, FvPrecision::Fp64, 1e-10, Ic::Vortical, false, dt, kSteps, thermal);
                const Run again = run(m, 1, s, FvPrecision::Fp64, 1e-10, Ic::Vortical, false, dt, kSteps, thermal);
                r32 = run(m, 1, s, FvPrecision::Fp32, 1e-6, Ic::Vortical, false, dt, kSteps, thermal);
                const Run again32 = run(m, 1, s, FvPrecision::Fp32, 1e-6, Ic::Vortical, false, dt, kSteps, thermal);
                const bool rr = sameMomentum(r64.st, again.st) && r64.st.T == again.st.T &&
                                sameMomentum(r32.st, again32.st) && r32.st.T == again32.st.T;
                check(rr, "(e) rerun byte-identical, FP32 and FP64 " + tag);
                ecol = rr ? "=" : "DIFFERS";
                const double div64 = r64.dg.divMax / std::max(r64.dg.uMax, 1e-30), div32 = r32.dg.divMax / std::max(r32.dg.uMax, 1e-30);
                check(div64 <= 1e-9, std::format("(b) FP64 divergence floor {:.1e} <= 1e-9 {}", div64, tag));
                check(div32 <= 1e-4, std::format("(b) FP32 divergence floor {:.1e} <= 1e-4 {}", div32, tag));
                // The energy identities hold on a closed boundary set (an open face carries KE in or
                // out): the O-grid's slip far circle.
                const Run kr = m.box ? r64 : run(m, 1, s, FvPrecision::Fp64, 1e-10, Ic::Vortical, true, dt, kSteps, thermal);
                const double ke = std::abs(kr.bd.conv) / std::max(kr.bd.grossConv, 1e-300);
                check(ke <= 1e-13, std::format("(b) conv KE {:.1e} of gross <= 1e-13 {}", ke, tag));
                bcol += std::format(", div/u {:.1e} (fp32 {:.1e}), KE {:.1e}", div64, div32, ke);
                if (thermal) {
                    const double kt = std::abs(kr.bd.convT) / std::max(kr.bd.grossConvT, 1e-300);
                    check(kt <= 1e-13, std::format("(b) conv T energy {:.1e} of gross <= 1e-13 {}", kt, tag));
                    bcol += std::format(", T {:.1e}", kt);
                }
                double du = 0;
                for (std::size_t i = 0; i < r64.st.U.size(); ++i) du = std::max(du, std::abs(r64.st.U[i] - r32.st.U[i]));
                du /= maxAbs(r64.st.U);
                check(du <= 1e-4, std::format("(c) FP32 vs FP64 |dU|/U {:.1e} <= 1e-4 {}", du, tag));
                ccol = std::format("{:.1e}", du);
            }
            // (d) numbering, FP64, 3 steps, by srcCell.
            {
                std::array<Run, 3> o;
                for (int k = 0; k < 3; ++k) o[std::size_t(k)] = run(m, k, s, FvPrecision::Fp64, 1e-12, Ic::Vortical, false, dt, 3, thermal);
                const std::size_t nC = o[1].st.p.size();
                // srcCell is Grid::idx on a structured mesh (ghost rows counted): rank it.
                std::vector<std::int64_t> keys(m.fm[1].srcCell.begin(), m.fm[1].srcCell.end());
                std::ranges::sort(keys);
                auto byKey = [&](int k) {
                    std::vector<double> v(4 * nC, 0.0);
                    const FaceMesh& fm = m.fm[std::size_t(k)];
                    const Run& r = o[std::size_t(k)];
                    for (std::size_t q = 0; q < nC; ++q) {
                        const std::size_t i = std::size_t(std::ranges::lower_bound(keys, fm.srcCell[q]) - keys.begin());
                        for (std::size_t j = 0; j < 3; ++j) v[4 * i + j] = r.st.U[j * nC + q];
                        v[4 * i + 3] = thermal ? r.st.T[q] : 0.0;
                    }
                    return v;
                };
                const auto ref = byKey(1);
                double dm = 0;
                bool bit = true;
                for (int k : {0, 2}) {
                    const auto v = byKey(k);
                    for (std::size_t i = 0; i < v.size(); ++i) {
                        dm = std::max(dm, std::abs(v[i] - ref[i]));
                        bit = bit && v[i] == ref[i];
                    }
                }
                dm /= maxAbs(ref);
                check(dm <= 1e-10, std::format("(d) numbering {:.1e} <= 1e-10 {}", dm, tag));
                dcol = bit ? "bitwise" : std::format("{:.1e}", dm);
            }
            std::printf("| %s | %s | %s | %s | %s | %s | %s |\n", fviModuleSetName(s).c_str(), m.name.c_str(), a.c_str(),
                        bcol.c_str(), ccol.c_str(), dcol.c_str(), ecol.c_str());
            std::fflush(stdout);
            ++rows;
        }
    }
    std::printf("gpu_fvi_matrix: %zu live sets, %d rows\n", sets.size(), rows);
    // The refusals reach DeviceFvIncomp::make: gravity without T.
    {
        const MeshCase& m = ms[0];
        const FaceMesh& fm = m.fm[1];
        const FvIncompSetup su = must(buildFvIncompSetup(fm, bcsOf(m, fm, true)), "setup");
        std::vector<double> U, T;
        icOf(m, fm, Ic::Rest, U, T);
        const FvIncompControls c{.nu = kNu, .betaG = {0.0, 1.0, 0.0}};
        const auto d = gpu::DeviceFvIncomp::make(su, la::graphOf(fm), c, U);
        check(!d && d.error().find("needs the temperature row") != std::string::npos,
              "make refuses gravity without T by the inventory's text");
    }
    return dnstest::finish();
}

// --- physics numbers of the live pairs ------------------------------------------------

struct Nu
{
    bool ok = false;
    double bot = 0, top = 0, face = 0;
};

// Time mean of the last 60% of rows (gpu_rb's window): nuBot, nuTop, (nuBotF + nuTopF)/2.
Nu nuOf(const std::string& prefix)
{
    Nu r;
    const auto t = valtest::loadCsv(prefix + "_stats.csv");
    if (!t) return r;
    const auto &nb = t->col("nuBot"), &nt = t->col("nuTop"), &fb = t->col("nuBotF"), &ft = t->col("nuTopF");
    if (nb.size() < 4) return r;
    const std::size_t i0 = nb.size() * 2 / 5, n = nb.size() - i0;
    for (std::size_t i = i0; i < nb.size(); ++i) {
        r.bot += nb[i] / double(n);
        r.top += nt[i] / double(n);
        r.face += 0.5 * (fb[i] + ft[i]) / double(n);
    }
    r.ok = true;
    return r;
}

Nu conv(const std::string& name, const std::string& args)
{
    const std::string prefix = "/tmp/dns_test_fvi_" + name;
    const std::string cmd = "\"" DNS_BIN "\" conv --solver fv --physics incompressible " + args +
                            " --emit stats --quiet --out " + prefix;
    const int rc = valtest::runBinary(cmd);
    check(rc == 0, name + ": run exits 0");
    return rc == 0 ? nuOf(prefix) : Nu{};
}

int runPhysics()
{
    // P1 (LES smag x gravity, 2D: qualitative): Ra 5e4, Pr 0.71, t 40, 48 x 24, Smagorinsky
    // 0.17 vs the same-grid DNS.
    const std::string r2 = " --rayleigh 5e4 --nz 1 --tt 40 --kick 0.02 --n 48 --ny 24 ";
    const Nu d2 = conv("d2", r2), l2 = conv("l2", r2 + "--les 0.17");
    std::printf("P1 Ra 5e4 2D 48x24: nuBot DNS %.4f, Smagorinsky 0.17 %.4f (%+.2f%%); face Nu %.4f %.4f\n", d2.bot, l2.bot,
                100 * (l2.bot / d2.bot - 1), d2.face, l2.face);
    for (const auto& [n, x] : {std::pair{"DNS", d2}, {"LES", l2}})
        check(x.ok && std::abs(x.bot - x.top) <= 0.05 * x.bot,
              std::format("P1 {}: the walls balance (nuBot {:.4f}, nuTop {:.4f}) to 5%", n, x.bot, x.top));
    check(l2.ok && std::abs(l2.bot - d2.bot) <= 0.05 * d2.bot, "P1 2D Smagorinsky within 5% of the same-grid DNS");
    // P2 (LES x T x gravity, 3D): Ra 1e5, aspect 2, t 60, 32 x 16 x 32 and 48 x 24 x 48: DNS,
    // sigma 1.35, Smagorinsky 0.17. Measured (docs/fvincomp.md): both LES raise Nu (the eddy
    // heat flux in the thermal layer), sigma less (D_sigma vanishes in the wall shear, no
    // damping in Smagorinsky), and the shift shrinks with h (Delta ~ h).
    const std::string r3 = " --rayleigh 1e5 --tt 60 --kick 0.05 --aspect 2 ";
    struct Rung
    {
        Nu d, s, m;
    };
    std::array<Rung, 2> rg;
    const char* grid[] = {"--n 32 --ny 16 --nz 32", "--n 48 --ny 24 --nz 48"};
    for (int k = 0; k < 2; ++k) {
        const std::string g = r3 + grid[k], t = std::to_string(k);
        rg[std::size_t(k)] = {.d = conv("d3" + t, g), .s = conv("s3" + t, g + " --les 1.35 --sgs sigma"),
                              .m = conv("m3" + t, g + " --les 0.17")};
        const Rung& r = rg[std::size_t(k)];
        std::printf("P2 Ra 1e5 3D %s: nuBot DNS %.4f, sigma %.4f (%+.2f%%), Smagorinsky %.4f (%+.2f%%); face %.4f %.4f %.4f\n",
                    grid[k], r.d.bot, r.s.bot, 100 * (r.s.bot / r.d.bot - 1), r.m.bot, 100 * (r.m.bot / r.d.bot - 1),
                    r.d.face, r.s.face, r.m.face);
        for (const auto& [n, x] : {std::pair{"DNS", r.d}, {"sigma", r.s}, {"Smagorinsky", r.m}})
            check(x.ok && std::abs(x.bot - x.top) <= 0.05 * x.bot,
                  std::format("P2 {} {}: the walls balance (nuBot {:.4f}, nuTop {:.4f}) to 5%", n, grid[k], x.bot, x.top));
        check(r.d.bot < r.s.bot && r.s.bot < r.m.bot, std::format("P2 {}: DNS < sigma < Smagorinsky Nu", grid[k]));
        check(r.m.bot <= 1.12 * r.d.bot, std::format("P2 {}: LES within 12% of the same-grid DNS", grid[k]));
    }
    check(rg[1].s.bot / rg[1].d.bot < rg[0].s.bot / rg[0].d.bot && rg[1].m.bot / rg[1].d.bot < rg[0].m.bot / rg[0].d.bot,
          "P2 the LES shift shrinks with h (sigma and Smagorinsky)");
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "markdown") {
        std::printf("%s", fviCombinationsMarkdown().c_str());
        return 0;
    }
    test::requireDeviceOrSkip();
    if (argc > 1 && std::string(argv[1]) == "physics") return runPhysics();
    return runMatrix();
}
