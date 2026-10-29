// FV face physics and the CPU FP64 oracle (dns/fvflux.hpp, dns/fvgrad,
// dns/fvref; masterplan_meshsolver P2, docs/mesh.md):
//   G1  uniform periodic box, hx != hy != hz: FvRef::rhs == Comp::rhs (1e-13)
//   G2  inviscid KE: sum V (u.rhs_m - |u|^2/2 rhs_rho) == discrete pressure
//       work (0 at uniform p), perturbed periodic hexes and box_refined
//   G3  freestream on cyl2d_snappy: |rhs| <= 1e-13
//   G4  (`test_fvref mms`) MMS on 3 nested sinusoidal periodic 2D meshes,
//       CPU-marched to steady: solution error order >= 1.8
//   G5  checkerboard decay of the face Laplacian: AlongD vs alpha-damping
//   G6  2D: w and every z flux exactly 0 on cyl2d_snappy
//   G7  adiabatic wall heat flux exactly 0.0 on a skewed walled box
//   E1  gradients exact for linear fields   E2  threads byte-identical
//   E3  Sigma-form dt: checkerboard decays at the bound, grows past it
//   E4  cell orders agree   E5  Dual / float instantiations of fvflux
//   E6  u checkerboard on 100:1 cells, Pr 0.72/2/7: the momentum 4/3
#include "dns/fvref.hpp"

#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/dual.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvgrad.hpp"
#include "check.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <functional>
#include <numbers>
#include <random>
#include <string>
#include <thread>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dns;
using dnstest::check;
namespace fs = std::filesystem;
using std::numbers::pi;

namespace {

constexpr double kGamma = 1.4;

template <class T> T orDie(std::expected<T, std::string> e, const std::string& what)
{
    if (!e) {
        std::printf("FATAL: %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
    return std::move(*e);
}

FaceMesh loadMesh(const char* name, CellOrder order = CellOrder::Morton)
{
    const PolyMesh pm = orDie(readPolyMesh(fs::path(DNS_MESH_DATA) / name / "constant/polyMesh"), name);
    return orDie(FaceMesh::fromPolyMesh(pm, order), name);
}

// Cyclic on x and y, z cyclic (3D) or empty (nz == 1).
StructuredSpec periodicSpec(const Grid& g)
{
    const FoamPatchType cz = g.nz == 1 ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    return {{0, 0, 0},
            {g.nx - 1, g.ny - 1, g.nz - 1},
            {"xlo", "xhi", "ylo", "yhi", "back", "front"},
            {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic, cz, cz}};
}

// One FvBc per patch: Empty on `empty` patches, else pick(patch).
std::vector<FvBc> bcFor(const FaceMesh& fm, const std::function<FvBc(const MeshPatch&)>& pick)
{
    std::vector<FvBc> v;
    for (const MeshPatch& p : fm.patches) {
        if (p.type == FoamPatchType::Empty) {
            FvBc e;
            e.kind = FvBcKind::Empty;
            v.push_back(e);
        } else
            v.push_back(pick(p));
    }
    return v;
}

FvConfig config(Transport tr, std::vector<FvBc> bc, FvGradScheme gs = FvGradScheme::LeastSquares,
                FvFaceGradRule rule = FvFaceGradRule::AlongD, double alpha = 1.0)
{
    FvConfig c;
    c.tr = tr;
    c.grad = gs;
    c.rule = rule;
    c.alpha = alpha;
    c.bc = std::move(bc);
    return c;
}

void setPrim(FvState& s, std::size_t c, double rho, const double u[3], double p)
{
    s.var(0)[c] = rho;
    s.var(1)[c] = rho * u[0];
    s.var(2)[c] = rho * u[1];
    s.var(3)[c] = rho * u[2];
    s.var(4)[c] = p / (kGamma - 1.0) + 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
}

std::array<int, 3> ijk(const Grid& g, std::int64_t idx)
{
    return {int(idx % g.nx), int((idx / g.nx) % g.ny), int(idx / (std::int64_t(g.nx) * g.ny))};
}

CurvMesh perturbed(const Grid& g, double frac, std::uint64_t seed)
{
    CurvMesh m = CurvMesh::uniform(g);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (std::size_t c = 0; c < m.x.size(); ++c) {
        m.x[c] += frac * g.hx * U(rng);
        m.y[c] += frac * g.hy * U(rng);
        if (g.nz > 1) m.z[c] += frac * g.hz * U(rng);
    }
    return m;
}

double maxAbs(const double* a, std::size_t n)
{
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::abs(a[i]));
    return m;
}

// ---------------------------------------------------------------------------
// G1: scheme equality with dns::Comp on a uniform periodic box.

void g1()
{
    std::printf("G1 FvRef::rhs vs Comp::rhs on uniform periodic boxes (hx != hy != hz)\n");
    struct Case
    {
        const char* tag;
        int nz;
        Transport tr;
        FvGradScheme gs;
        CellOrder order;
    };
    const Case cases[] = {
        {"3D mu 0.05, LSQ, Morton", 5, Transport::constant(kGamma, 0.05), FvGradScheme::LeastSquares, CellOrder::Morton},
        {"3D Sutherland, LSQ, colour", 5, Transport::air(0.05, 0.9, 288.15, 0.72, kGamma, 1.0, true),
         FvGradScheme::LeastSquares, CellOrder::ColourMajor},
        {"3D mu 0.05, Green-Gauss, source", 5, Transport::constant(kGamma, 0.05), FvGradScheme::GreenGauss,
         CellOrder::Source},
        {"3D inviscid", 5, Transport::constant(kGamma, 0.0), FvGradScheme::LeastSquares, CellOrder::Morton},
        {"2D (nz 1) mu 0.05, LSQ", 1, Transport::constant(kGamma, 0.05), FvGradScheme::LeastSquares,
         CellOrder::Morton},
    };
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> ph(0.0, 2.0 * pi);
    for (const Case& cs : cases) {
        const Grid g = cs.nz > 1 ? Grid::make(8, 6, 5, 8 * 0.3, 6 * 0.45, 5 * 0.2)
                                 : Grid::make(8, 6, 1, 8 * 0.3, 6 * 0.45, 0.35);
        Comp comp = Comp::make(g, kGamma, cs.tr.muRef);
        comp.tr = cs.tr;
        const FaceMesh fm =
            orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), cs.order), cs.tag);
        const FvRef ref = orDie(FvRef::make(fm, config(cs.tr, bcFor(fm, {}), cs.gs)), cs.tag);
        double a[15];
        for (double& x : a) x = ph(rng);
        const double lx = g.nx * g.hx, ly = g.ny * g.hy, lz = g.nz * g.hz;
        CompState q = CompState::zeros(g);
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    const double X = 2 * pi * (i + 0.5) * g.hx / lx, Y = 2 * pi * (j + 0.5) * g.hy / ly;
                    const double Z = g.nz > 1 ? 2 * pi * (k + 0.5) * g.hz / lz : 0.0;
                    const double rho = 1.0 + 0.2 * std::sin(X + a[0]) * std::cos(Y + a[1]) * std::cos(Z + a[2]);
                    const double u[3] = {0.4 + 0.3 * std::cos(X + a[3]) * std::sin(Y + a[4]) * std::cos(Z + a[5]),
                                         -0.2 + 0.25 * std::sin(X + Y + a[6]) * std::cos(Z + a[7]),
                                         g.nz > 1 ? 0.2 * std::sin(Z + a[8]) * std::cos(X + a[9]) : 0.0};
                    const double p = 1.0 + 0.1 * std::cos(X + a[10]) * std::cos(Y + a[11]) * std::sin(Z + a[12]);
                    const std::size_t c = g.idx(i, j, k);
                    q.rho[c] = rho;
                    q.mx[c] = rho * u[0];
                    q.my[c] = rho * u[1];
                    q.mz[c] = rho * u[2];
                    q.E[c] = p / (kGamma - 1.0) + 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                }
        FvState s = FvState::zeros(fm.nC), out;
        for (int n = 0; n < fm.nC; ++n) {
            const std::size_t c = std::size_t(fm.srcCell[std::size_t(n)]);
            s.var(0)[n] = q.rho[c];
            s.var(1)[n] = q.mx[c];
            s.var(2)[n] = q.my[c];
            s.var(3)[n] = q.mz[c];
            s.var(4)[n] = q.E[c];
        }
        CompState r = CompState::zeros(g);
        comp.rhs(q, r);
        ref.rhs(s, out);
        const std::vector<double>* rv[5] = {&r.rho, &r.mx, &r.my, &r.mz, &r.E};
        double worst = 0.0;
        for (int v = 0; v < 5; ++v) {
            const double scale = std::max(maxAbs(rv[v]->data(), rv[v]->size()), 1e-300);
            double e = 0.0;
            for (int n = 0; n < fm.nC; ++n)
                e = std::max(e, std::abs(out.var(v)[n] - (*rv[v])[std::size_t(fm.srcCell[std::size_t(n)])]));
            // A row that is identically zero in both (w at nz == 1) counts as exact.
            worst = std::max(worst, maxAbs(rv[v]->data(), rv[v]->size()) == 0.0 ? e : e / scale);
        }
        check(worst <= 1e-13, std::format("{}: max relative |FvRef - Comp| = {:.2e} <= 1e-13", cs.tag, worst));
    }
}

// ---------------------------------------------------------------------------
// G2: discrete kinetic-energy budget of the inviscid KG flux.

void keBudget(const FaceMesh& fm, const std::string& tag)
{
    const FvRef ref = orDie(FvRef::make(fm, config(Transport::constant(kGamma, 0.0), bcFor(fm, {}))), tag);
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (const bool uniformP : {true, false}) {
        FvState s = FvState::zeros(fm.nC), out;
        std::vector<double> pc(std::size_t(fm.nC));
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            const double u[3] = {0.5 * U(rng), 0.5 * U(rng), 0.5 * U(rng)};
            pc[c] = uniformP ? 1.0 : 1.0 + 0.2 * U(rng);
            setPrim(s, c, 1.0 + 0.2 * U(rng), u, pc[c]);
        }
        ref.rhs(s, out);
        auto vel = [&](std::size_t c, int k) { return s.var(1 + k)[c] / s.var(0)[c]; };
        double ke = 0.0, scale = 0.0;
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            const double V = fm.V[c];
            double uu = 0.0, um = 0.0, am = 0.0;
            for (int k = 0; k < 3; ++k) {
                uu += vel(c, k) * vel(c, k);
                um += vel(c, k) * out.var(1 + k)[c];
                am += std::abs(vel(c, k) * out.var(1 + k)[c]);
            }
            ke += V * (um - 0.5 * uu * out.var(0)[c]);
            scale += V * (am + 0.5 * uu * std::abs(out.var(0)[c]));
        }
        double pw = 0.0; // sum_f pbar S . (u_nei - u_own)
        for (std::size_t f = 0; f < std::size_t(fm.nI); ++f) {
            const std::size_t o = std::size_t(fm.own[f]), n = std::size_t(fm.nei[f]);
            double du = 0.0;
            for (int k = 0; k < 3; ++k) du += fm.S[std::size_t(k)][f] * (vel(n, k) - vel(o, k));
            pw += 0.5 * (pc[o] + pc[n]) * du;
        }
        const double err = std::abs(ke - pw) / scale;
        check(err <= 1e-13, std::format("{} ({}): |KE budget - pressure work| / scale = {:.2e} <= 1e-13 "
                                        "(pressure work {:.3e}, scale {:.3e})",
                                        tag, uniformP ? "uniform p" : "random p", err, pw, scale));
        double cons = 0.0;
        for (int v = 0; v < 5; ++v) {
            double sum = 0.0, abs = 0.0;
            for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
                sum += fm.V[c] * out.var(v)[c];
                abs += fm.V[c] * std::abs(out.var(v)[c]);
            }
            cons = std::max(cons, std::abs(sum) / abs);
        }
        check(cons <= 1e-13, std::format("{} ({}): mass/momentum/energy totals conserved, worst {:.2e}", tag,
                                         uniformP ? "uniform p" : "random p", cons));
    }
}

void g2()
{
    std::printf("G2 inviscid kinetic-energy budget (KG)\n");
    const Grid g = Grid::make(8, 7, 6, 1.0, 0.9, 0.8);
    keBudget(orDie(FaceMesh::fromCurvMesh(perturbed(g, 0.15, 3), periodicSpec(g), CellOrder::Morton), "pert"),
             "perturbed periodic hexes 8x7x6 (+-0.15h, warped faces)");
    keBudget(loadMesh("box_refined"), "box_refined (split hexes, all cyclic)");
}

// ---------------------------------------------------------------------------
// G3: freestream preservation on the snappy cylinder.

FvPrim<double> primOfState(const double q[5]) { return fvPrimOf(q[0], q[1], q[2], q[3], q[4], kGamma, 1.0); }

void g3()
{
    std::printf("G3 freestream on cyl2d_snappy\n");
    const FaceMesh fm = loadMesh("cyl2d_snappy");
    for (const bool mixed : {false, true}) {
        const double u[3] = {0.6, mixed ? 0.0 : 0.2, 0.0}, rho = 1.0, p = 1.0 / kGamma;
        double qInf[5] = {rho, rho * u[0], rho * u[1], rho * u[2],
                          p / (kGamma - 1.0) + 0.5 * rho * (u[0] * u[0] + u[1] * u[1])};
        const FvPrim<double> P = primOfState(qInf);
        auto pick = [&](const MeshPatch& pt) {
            FvBc b;
            b.kind = FvBcKind::Far;
            std::copy(qInf, qInf + 5, b.qFar);
            if (!mixed) return b;
            if (pt.name == "inlet") {
                b.kind = FvBcKind::Inflow;
                std::copy(u, u + 3, b.uIn);
                b.tIn = P.temp;
            } else if (pt.name == "outlet") {
                b.kind = FvBcKind::Outflow;
                b.pOut = P.p;
                b.tBack = P.temp;
            } else if (pt.name == "lower" || pt.name == "upper") {
                b.kind = FvBcKind::Slip;
            }
            return b;
        };
        const FvRef ref = orDie(FvRef::make(fm, config(Transport::constant(kGamma, 0.01), bcFor(fm, pick))), "g3");
        FvState s = FvState::zeros(fm.nC), out;
        for (int v = 0; v < 5; ++v) std::fill_n(s.var(v), fm.nC, qInf[v]);
        ref.rhs(s, out);
        const double worst = maxAbs(out.q.data(), out.q.size());
        check(worst <= 1e-13,
              std::format("{}: max |rhs| = {:.2e} <= 1e-13 (rho 1, |u| {:.3f}, p 1/gamma, mu 0.01)",
                          mixed ? "inlet Inflow, outlet Outflow, lower/upper Slip, cylinder Far"
                                : "every patch Far",
                          worst, std::hypot(u[0], u[1])));
    }
}

// ---------------------------------------------------------------------------
// G5: the face Laplacian's checkerboard damping, per correction rule.

struct Laplacian
{
    const FaceMesh& fm;
    FvFaceGeom geom;
    FvGradWeights w;

    // out = (1/V) sum_f g_f . S_f (periodic meshes: no boundary faces).
    void apply(const std::vector<double>& phi, std::vector<double>& out) const
    {
        const std::size_t nC = std::size_t(fm.nC), nI = std::size_t(fm.nI);
        std::array<std::vector<double>, 3> g;
        for (auto& x : g) x.assign(nC, 0.0);
        fvCellGradient(fm, geom, w, phi, {}, {g[0].data(), g[1].data(), g[2].data()}, 0, nC);
        std::vector<double> fl(nI);
        for (std::size_t f = 0; f < nI; ++f) {
            const std::size_t o = std::size_t(fm.own[f]), n = std::size_t(fm.nei[f]);
            const double go[3] = {g[0][o], g[1][o], g[2][o]}, gn[3] = {g[0][n], g[1][n], g[2][n]};
            const double d[3] = {fm.d[0][f], fm.d[1][f], fm.d[2][f]};
            const double cr[3] = {geom.corr[0][f], geom.corr[1][f], geom.corr[2][f]};
            double gf[3];
            fvFaceGrad(go, gn, phi[o], phi[n], d, cr, gf);
            fl[f] = gf[0] * fm.S[0][f] + gf[1] * fm.S[1][f] + gf[2] * fm.S[2][f];
        }
        out.assign(nC, 0.0);
        for (std::size_t c = 0; c < nC; ++c) {
            double a = 0.0;
            for (int q = fm.cellStart[c]; q < fm.cellStart[c + 1]; ++q) {
                const std::size_t f = std::size_t(fm.cellFace[std::size_t(q)]);
                if (f >= nI) continue;
                a += std::size_t(fm.own[f]) == c ? fl[f] : -fl[f];
            }
            out[c] = a / fm.V[c];
        }
    }
};

double dotV(const FaceMesh& fm, const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c) s += fm.V[c] * a[c] * b[c];
    return s;
}

struct DampResult
{
    double rhoCb, lamMax, eff, decay;
};

DampResult damping(const FaceMesh& fm, const Grid& g, FvFaceGradRule rule, double alpha)
{
    Laplacian L{fm, orDie(fvFaceGeom(fm, rule, alpha), "geom"), {}};
    L.w = orDie(fvGradWeights(fm, L.geom, FvGradScheme::LeastSquares), "lsq");
    const std::size_t nC = std::size_t(fm.nC);
    std::vector<double> cb(nC), Lx;
    for (std::size_t c = 0; c < nC; ++c) {
        const auto I = ijk(g, fm.srcCell[c]);
        cb[c] = (I[0] + I[1] + I[2]) % 2 ? -1.0 : 1.0;
    }
    L.apply(cb, Lx);
    const double rhoCb = -dotV(fm, cb, Lx) / dotV(fm, cb, cb);
    // Spectral radius by power iteration (V-weighted norm).
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> x(nC);
    for (double& v : x) v = U(rng);
    double lam = 0.0;
    for (int it = 0; it < 300; ++it) {
        L.apply(x, Lx);
        const double nx = std::sqrt(dotV(fm, x, x)), nl = std::sqrt(dotV(fm, Lx, Lx));
        lam = nl / nx;
        for (std::size_t c = 0; c < nC; ++c) x[c] = Lx[c] / nl;
    }
    // Explicit SSP-RK3 at dt = 2 / lambda_max (the dt the operator allows),
    // 40 steps from the checkerboard: its remaining projection.
    const double dt = 2.0 / lam;
    std::vector<double> q = cb, q1(nC), q2(nC);
    for (int n = 0; n < 40; ++n) {
        L.apply(q, Lx);
        for (std::size_t c = 0; c < nC; ++c) q1[c] = q[c] + dt * Lx[c];
        L.apply(q1, Lx);
        for (std::size_t c = 0; c < nC; ++c) q2[c] = 0.75 * q[c] + 0.25 * (q1[c] + dt * Lx[c]);
        L.apply(q2, Lx);
        for (std::size_t c = 0; c < nC; ++c) q[c] = q[c] / 3.0 + 2.0 / 3.0 * (q2[c] + dt * Lx[c]);
    }
    const double decay = std::abs(dotV(fm, q, cb) / dotV(fm, cb, cb));
    return {rhoCb, lam, rhoCb / lam, decay};
}

void g5()
{
    std::printf("G5 checkerboard damping of the face Laplacian (LSQ cell gradients)\n");
    struct M
    {
        const char* tag;
        Grid g;
        CurvMesh m;
    };
    const Grid g2 = Grid::make(32, 32, 1, 1.0, 1.0, 1.0 / 32);
    const Grid g3 = Grid::make(12, 12, 12, 1.0, 1.0, 1.0);
    const M meshes[] = {
        {"uniform 2D 32^2", g2, CurvMesh::uniform(g2)},
        {"random +-0.25h 2D 32^2", g2, perturbed(g2, 0.25, 17)},
        {"sinusoidal amp 0.06 2D 32^2", g2, CurvMesh::sinusoidal(g2, 0.06)},
        {"random +-0.2h 3D 12^3", g3, perturbed(g3, 0.2, 19)},
    };
    struct R
    {
        const char* tag;
        FvFaceGradRule rule;
        double alpha;
    };
    const R rules[] = {{"AlongD a=1 (plain correction)", FvFaceGradRule::AlongD, 1.0},
                       {"AlongNormal a=4/3 (alpha-damping)", FvFaceGradRule::AlongNormal, 4.0 / 3.0},
                       {"AlongNormal a=1 (OpenFOAM corrected)", FvFaceGradRule::AlongNormal, 1.0}};
    for (const M& m : meshes) {
        const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(m.m, periodicSpec(m.g), CellOrder::Morton), m.tag);
        for (const R& r : rules) {
            const DampResult d = damping(fm, m.g, r.rule, r.alpha);
            std::printf("  %-28s %-38s rho_cb h^2 %.4f  lambda_max h^2 %.4f  efficiency %.4f  "
                        "cb after 40 RK3 steps at 2/lambda: %.3e\n",
                        m.tag, r.tag, d.rhoCb * m.g.hx * m.g.hx, d.lamMax * m.g.hx * m.g.hx, d.eff, d.decay);
            if (r.rule == FvFaceGradRule::AlongD)
                check(d.decay < 1e-3 && d.eff > 0.5,
                      std::format("{}: the pinned rule (AlongD) damps the checkerboard (efficiency {:.3f}, "
                                  "remaining {:.2e})",
                                  m.tag, d.eff, d.decay));
        }
    }
}

// ---------------------------------------------------------------------------
// G6: the 2D restriction is exact on the snappy cylinder.

struct CylCase
{
    FaceMesh fm;
    FvConfig cfg;
    FvState s;
};

CylCase cylCase(CellOrder order = CellOrder::Morton)
{
    CylCase cc{loadMesh("cyl2d_snappy", order), {}, {}};
    // M 0.2 (c = 1): an impulsive M 0.5 start goes transonic at the shoulders.
    const double u[3] = {0.2, 0.02, 0.0}, rho = 1.0, p = 1.0 / kGamma;
    double qInf[5] = {rho, rho * u[0], rho * u[1], 0.0, p / (kGamma - 1.0) + 0.5 * rho * (u[0] * u[0] + u[1] * u[1])};
    const FvPrim<double> P = primOfState(qInf);
    auto pick = [&](const MeshPatch& pt) {
        FvBc b;
        if (pt.name == "inlet") {
            b.kind = FvBcKind::Inflow;
            std::copy(u, u + 3, b.uIn);
            b.tIn = P.temp;
        } else if (pt.name == "outlet") {
            b.kind = FvBcKind::Outflow;
            b.pOut = P.p;
            b.tBack = P.temp;
        } else if (pt.name == "cylinder") {
            b.kind = FvBcKind::Wall;
            b.isothermal = true;
            b.tWall = 1.1 * P.temp;
            b.omega[2] = 0.3; // spinning about the cylinder axis
        } else {
            b.kind = FvBcKind::Slip;
        }
        return b;
    };
    cc.cfg = config(Transport::constant(kGamma, 0.02), bcFor(cc.fm, pick));
    // Freestream plus a deterministic per-SOURCE-cell perturbation, so every
    // cell order holds the same physical state.
    cc.s = FvState::zeros(cc.fm.nC);
    for (std::size_t c = 0; c < std::size_t(cc.fm.nC); ++c) {
        std::mt19937_64 rng(1000 + std::uint64_t(cc.fm.srcCell[c]));
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        const double uc[3] = {u[0] + 0.05 * U(rng), u[1] + 0.05 * U(rng), 0.0};
        setPrim(cc.s, c, rho * (1.0 + 0.05 * U(rng)), uc, p * (1.0 + 0.05 * U(rng)));
    }
    return cc;
}

void g6()
{
    std::printf("G6 2D restriction on cyl2d_snappy (wall isothermal + spinning, inflow, outflow, slip)\n");
    const CylCase cc = cylCase();
    const FaceMesh& fm = cc.fm;
    check(fm.dim == 2 && fm.emptyAxis == 2, "cyl2d_snappy is 2D, empty axis z");
    const FvRef ref = orDie(FvRef::make(fm, cc.cfg), "g6");
    FvEval ev;
    FvState out;
    ref.evaluate(cc.s, ev, out);
    const std::size_t nF = std::size_t(fm.nI + fm.nB);
    bool fz = true, gz = true, bz = true, rz = true;
    for (std::size_t f = 0; f < nF; ++f) fz = fz && ev.F[3][f] == 0.0;
    for (int k = 0; k < 4; ++k)
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) gz = gz && ev.grad[std::size_t(3 * k + 2)][c] == 0.0;
    for (const auto& b : ev.bnd) bz = bz && b.u[2] == 0.0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) rz = rz && out.var(3)[c] == 0.0;
    check(fz, "every face's z-momentum flux is exactly 0.0 (interior, wall, slip, inflow, outflow)");
    check(gz, "every cell gradient's z component is exactly 0.0 (u, v, w, T)");
    check(bz, "every boundary state has w exactly 0.0");
    check(rz, "rhs of rho w is exactly 0.0 in every cell");
    FvState s = cc.s;
    const double dt = orDie(ref.stableDt(s), "dt");
    ref.march(s, dt, 20);
    bool wz = true;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) wz = wz && s.var(3)[c] == 0.0;
    double rmin = 1e300;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) rmin = std::min(rmin, s.var(0)[c]);
    check(wz && rmin > 0.0, std::format("20 SSP-RK3 steps at dt {:.3e}: rho w exactly 0.0 everywhere (rho min {:.4f})",
                                        dt, rmin));
}

// ---------------------------------------------------------------------------
// G7: an adiabatic wall leaks no heat, on a skewed (sinusoidal) walled box.

void g7()
{
    std::printf("G7 adiabatic wall heat flux on a skewed walled box (sinusoidal amp 0.06, 8x6x4)\n");
    const Grid g = Grid::make(8, 6, 4, 2.0, 1.2, 0.6);
    const StructuredSpec sp{{0, 0, 0},
                            {7, 5, 3},
                            {"xlo", "xhi", "lower", "upper", "zlo", "zhi"},
                            {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Wall, FoamPatchType::Wall,
                             FoamPatchType::Cyclic, FoamPatchType::Cyclic}};
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::sinusoidal(g, 0.06), sp, CellOrder::Morton), "g7");
    double maxNonOrth = 0.0;
    for (const MeshPatch& p : fm.patches)
        for (int i = 0; i < p.count; ++i) {
            const std::size_t f = std::size_t(fm.nI + p.first + i);
            const double S[3] = {fm.S[0][f], fm.S[1][f], fm.S[2][f]}, d[3] = {fm.d[0][f], fm.d[1][f], fm.d[2][f]};
            const double c = (S[0] * d[0] + S[1] * d[1] + S[2] * d[2]) /
                             (std::hypot(S[0], S[1], S[2]) * std::hypot(d[0], d[1], d[2]));
            maxNonOrth = std::max(maxNonOrth, std::acos(std::min(1.0, c)) * 180.0 / pi);
        }
    const double T0 = 1.0 / kGamma;
    for (const bool iso : {false, true}) {
        auto pick = [&](const MeshPatch&) {
            FvBc b;
            b.kind = FvBcKind::Wall;
            b.isothermal = iso;
            b.tWall = T0;
            return b;
        };
        const FvRef ref = orDie(FvRef::make(fm, config(Transport::constant(kGamma, 0.05), bcFor(fm, pick))), "g7");
        FvState s = FvState::zeros(fm.nC), out;
        std::mt19937_64 rng(23);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            const double T = T0 * (1.0 + 0.1 * std::sin(2.0 * pi * fm.C[0][c] / 2.0)); // tangential gradient
            const double u[3] = {0.1 * U(rng), 0.1 * U(rng), 0.1 * U(rng)};
            setPrim(s, c, 1.0, u, T);
        }
        FvEval ev;
        ref.evaluate(s, ev, out);
        bool zero = true;
        double mx = 0.0;
        for (std::size_t f = std::size_t(fm.nI); f < std::size_t(fm.nI + fm.nB); ++f) {
            zero = zero && ev.F[4][f] == 0.0 && ev.F[0][f] == 0.0;
            mx = std::max(mx, std::abs(ev.F[4][f]));
        }
        double sum = 0.0, abs = 0.0;
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            sum += fm.V[c] * out.var(4)[c];
            abs += fm.V[c] * std::abs(out.var(4)[c]);
        }
        if (!iso) {
            check(zero, std::format("adiabatic: wall mass and energy face fluxes exactly 0.0 on all {} wall faces "
                                    "(wall non-orthogonality up to {:.1f} deg, T varies along the wall)",
                                    fm.nB, maxNonOrth));
            check(std::abs(sum) <= 1e-14 * abs,
                  std::format("adiabatic: total energy rhs sum V rhs_E / sum V |rhs_E| = {:.2e} (no leak)",
                              std::abs(sum) / abs));
        } else {
            check(mx > 1e-6, std::format("isothermal contrast: wall energy flux max {:.3e} != 0", mx));
        }
    }
}

// ---------------------------------------------------------------------------
// E1: least squares is exact for linear fields on any mesh (boundary faces
// carry the exact face value).

void e1()
{
    std::printf("E1 cell gradients of a linear field\n");
    struct M
    {
        const char* tag;
        FaceMesh fm;
    };
    const Grid g = Grid::make(8, 6, 4, 2.0, 1.2, 0.6);
    StructuredSpec walls{{0, 0, 0}, {7, 5, 3}, {"a", "b", "c", "d", "e", "f"}, {}};
    walls.type.fill(FoamPatchType::Wall);
    M meshes[] = {{"cyl2d_snappy", loadMesh("cyl2d_snappy")},
                  {"sinusoidal 3D walled box", orDie(FaceMesh::fromCurvMesh(CurvMesh::sinusoidal(g, 0.06), walls,
                                                                            CellOrder::Morton),
                                                     "walled")}};
    for (const M& m : meshes) {
        const FaceMesh& fm = m.fm;
        const double a[3] = {2.0, -3.0, fm.dim == 2 ? 0.0 : 0.5};
        auto phiAt = [&](double x, double y, double z) { return 1.0 + a[0] * x + a[1] * y + a[2] * z; };
        std::vector<double> phi(std::size_t(fm.nC)), phiB(std::size_t(fm.nB));
        for (std::size_t c = 0; c < phi.size(); ++c) phi[c] = phiAt(fm.C[0][c], fm.C[1][c], fm.C[2][c]);
        for (std::size_t b = 0; b < phiB.size(); ++b) {
            const std::size_t f = std::size_t(fm.nI) + b;
            phiB[b] = phiAt(fm.Cf[0][f], fm.Cf[1][f], fm.Cf[2][f]);
        }
        for (const FvGradScheme gs : {FvGradScheme::LeastSquares, FvGradScheme::GreenGauss}) {
            const FvFaceGeom geom = orDie(fvFaceGeom(fm, FvFaceGradRule::AlongD, 1.0), m.tag);
            const FvGradWeights w = orDie(fvGradWeights(fm, geom, gs), m.tag);
            std::array<std::vector<double>, 3> gr;
            for (auto& x : gr) x.assign(phi.size(), 0.0);
            fvCellGradient(fm, geom, w, phi, phiB, {gr[0].data(), gr[1].data(), gr[2].data()}, 0, phi.size());
            double err = 0.0;
            for (std::size_t c = 0; c < phi.size(); ++c)
                err = std::max(err, std::hypot(gr[0][c] - a[0], gr[1][c] - a[1], gr[2][c] - a[2]));
            err /= std::hypot(a[0], a[1], a[2]);
            if (gs == FvGradScheme::LeastSquares)
                check(err <= 1e-12, std::format("{}: least squares exact, max |g - a| / |a| = {:.2e}", m.tag, err));
            else
                std::printf("  %s: Green-Gauss max |g - a| / |a| = %.3e (not exact off the line of centres; "
                            "reported)\n",
                            m.tag, err);
        }
    }
}

// ---------------------------------------------------------------------------
// E2: byte-identical for any thread count.

void e2()
{
    std::printf("E2 threads 1/2/3/7/hw byte-identical (rhs and 3 steps)\n");
    const Grid g = Grid::make(96, 96, 1, 1.0, 1.0, 1.0 / 96);
    const FaceMesh fm =
        orDie(FaceMesh::fromCurvMesh(CurvMesh::sinusoidal(g, 0.05), periodicSpec(g), CellOrder::Morton), "e2");
    std::mt19937_64 rng(29);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    FvState s0 = FvState::zeros(fm.nC);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double u[3] = {0.3 + 0.05 * U(rng), 0.05 * U(rng), 0.0};
        setPrim(s0, c, 1.0 + 0.05 * U(rng), u, 1.0 / kGamma * (1.0 + 0.05 * U(rng)));
    }
    std::vector<double> ref0, ref1;
    const int hw = int(std::max(1u, std::thread::hardware_concurrency()));
    bool same = true;
    for (const int t : {1, 2, 3, 7, hw}) {
        const FvRef ref = orDie(FvRef::make(fm, config(Transport::constant(kGamma, 0.01), bcFor(fm, {})), t), "e2");
        FvState out, s = s0;
        ref.rhs(s, out);
        const double dt = orDie(ref.stableDt(s), "dt");
        ref.march(s, dt, 3);
        if (ref0.empty()) {
            ref0 = out.q;
            ref1 = s.q;
            continue;
        }
        same = same && std::memcmp(ref0.data(), out.q.data(), ref0.size() * sizeof(double)) == 0 &&
               std::memcmp(ref1.data(), s.q.data(), ref1.size() * sizeof(double)) == 0;
    }
    check(same, std::format("9216-cell mesh: rhs and 3-step state byte-identical for threads 1/2/3/7/{}", hw));
}

// ---------------------------------------------------------------------------
// E3: the Sigma-form diffusive bound is the checkerboard's own (issue 15).

void e3()
{
    std::printf("E3 Sigma-form diffusive dt on a 3D isotropic periodic box (T checkerboard, mu 1)\n");
    const Grid g = Grid::make(12, 12, 12, 1.0, 1.0, 1.0);
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "e3");
    const FvRef ref = orDie(FvRef::make(fm, config(Transport::constant(kGamma, 1.0), bcFor(fm, {}))), "e3");
    const double T0 = 1.0 / kGamma, eps = 1e-3;
    FvState s0 = FvState::zeros(fm.nC);
    std::vector<double> cb(std::size_t(fm.nC));
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const auto I = ijk(g, fm.srcCell[c]);
        cb[c] = (I[0] + I[1] + I[2]) % 2 ? -1.0 : 1.0;
        const double u[3] = {0, 0, 0};
        setPrim(s0, c, 1.0, u, T0 * (1.0 + eps * cb[c]));
    }
    const double dt = orDie(ref.stableDt(s0, {.cfl = 0.4, .diff = 1.0}), "dt");
    auto amp = [&](const FvState& s) {
        double a = 0.0;
        for (std::size_t c = 0; c < cb.size(); ++c) {
            const FvPrim<double> p = fvPrimOf(s.var(0)[c], s.var(1)[c], s.var(2)[c], s.var(3)[c], s.var(4)[c],
                                              kGamma, 1.0);
            a += cb[c] * (p.temp - T0);
        }
        return std::abs(a) / (double(cb.size()) * T0 * eps);
    };
    for (const double f : {1.0, 1.3}) {
        FvState s = s0;
        ref.march(s, f * dt, 20);
        const double a = amp(s);
        if (f == 1.0)
            check(a < 1e-6, std::format("at the bound (lambda dt = -2): checkerboard x {:.2e} after 20 steps", a));
        else
            check(a > 1.0, std::format("at 1.3x the bound (lambda dt = -2.6, past SSP-RK3's -2.51): grows x {:.2e} "
                                       "(the bound has no hidden slack)",
                                       a));
    }
}

// ---------------------------------------------------------------------------
// E6: the momentum checkerboard's 4/3 (normal stress). A longitudinal u
// checkerboard on 100:1 cells at Pr 2 and 7 sits at lambda dt = -2 only with
// max(4/3, gamma/Pr); max(1, .) put it at -2.67 (x1.8e6 in 60 steps).

void e6()
{
    std::printf("E6 Sigma-form dt, u checkerboard on 100:1 2D cells, Pr 0.72 / 2 / 7\n");
    const Grid g = Grid::make(64, 4, 1, 1.0, 4 * 100.0 / 64, 1.0 / 64);
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "e6");
    const double T0 = 1.0 / kGamma, eps = 1e-3;
    FvState s0 = FvState::zeros(fm.nC);
    std::vector<double> cb(std::size_t(fm.nC));
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        cb[c] = ijk(g, fm.srcCell[c])[0] % 2 ? -1.0 : 1.0;
        const double u[3] = {eps * cb[c], 0, 0};
        setPrim(s0, c, 1.0, u, T0);
    }
    auto amp = [&](const FvState& s) {
        double a = 0.0;
        for (std::size_t c = 0; c < cb.size(); ++c) a += cb[c] * s.var(1)[c] / s.var(0)[c];
        return std::abs(a) / (double(cb.size()) * eps);
    };
    for (const double pr : {0.72, 2.0, 7.0}) {
        Transport tr = Transport::constant(kGamma, 1.0);
        tr.Pr = pr;
        const FvRef ref = orDie(FvRef::make(fm, config(tr, bcFor(fm, {}))), "e6");
        const double dt = orDie(ref.stableDt(s0, {.cfl = 0.4, .diff = 1.0}), "dt");
        for (const double f : {1.0, 1.3}) {
            if (pr == 0.72 && f > 1.0) continue; // energy-bound there: the u mode has slack
            FvState s = s0;
            ref.march(s, f * dt, 60);
            const double a = amp(s);
            if (f == 1.0)
                check(a < 1e-6, std::format("Pr {}: at the bound the u checkerboard decays x {:.2e} in 60 steps", pr, a));
            else
                check(a > 1.0, std::format("Pr {}: at 1.3x the bound it grows x {:.2e} (the 4/3 has no slack)", pr, a));
        }
    }
}

// ---------------------------------------------------------------------------
// E4: Source / Morton / colour-major give the same rhs (to summation order).

void e4()
{
    std::printf("E4 cell orders on cyl2d_snappy\n");
    std::vector<double> base;
    double worst = 0.0;
    for (const CellOrder o : {CellOrder::Source, CellOrder::Morton, CellOrder::ColourMajor}) {
        const CylCase cc = cylCase(o);
        const FvRef ref = orDie(FvRef::make(cc.fm, cc.cfg), "e4");
        FvState out;
        ref.rhs(cc.s, out);
        std::vector<double> bySrc(out.q.size());
        for (int v = 0; v < 5; ++v)
            for (std::size_t c = 0; c < std::size_t(cc.fm.nC); ++c)
                bySrc[std::size_t(v) * std::size_t(cc.fm.nC) + std::size_t(cc.fm.srcCell[c])] = out.var(v)[c];
        if (base.empty()) {
            base = bySrc;
            continue;
        }
        const double scale = maxAbs(base.data(), base.size());
        for (std::size_t i = 0; i < base.size(); ++i) worst = std::max(worst, std::abs(bySrc[i] - base[i]) / scale);
    }
    check(worst <= 1e-13, std::format("Morton and colour-major rhs == source-order rhs to {:.2e} (<= 1e-13)", worst));
}

// ---------------------------------------------------------------------------
// E5: the templates instantiate on Dual<N> (implicit Jacobians, P7) and float
// (the GPU path, P4).

void e5()
{
    std::printf("E5 fvflux on Dual<10> (vs FP64 central FD) and float\n");
    const Transport tr = Transport::constant(kGamma, 0.03);
    const double S[3] = {0.3, -0.7, 0.2};
    const double qL[5] = {1.1, 0.4, -0.1, 0.05, 2.3}, qR[5] = {0.95, 0.35, 0.02, -0.03, 2.1};
    const double gu[3][3] = {{0.1, -0.2, 0.05}, {0.3, 0.02, -0.1}, {0.0, 0.07, 0.2}}, gT[3] = {0.4, -0.1, 0.2};
    auto fluxD = [&](const double* a, const double* b, double F[5]) {
        const FvPrim<double> A = fvPrimOf(a[0], a[1], a[2], a[3], a[4], kGamma, 1.0);
        const FvPrim<double> B = fvPrimOf(b[0], b[1], b[2], b[3], b[4], kGamma, 1.0);
        double mu, kap;
        fvTransportAt(tr, 0.5 * (A.temp + B.temp), mu, kap);
        const FvFlux<double> f = fvMeanFlux(A, B, S, gu, gT, mu, kap);
        F[0] = f.rho, F[1] = f.m[0], F[2] = f.m[1], F[3] = f.m[2], F[4] = f.E;
    };
    using D = Dual<10>;
    D a[5], b[5];
    for (int v = 0; v < 5; ++v) {
        a[v] = D::var(qL[v], v);
        b[v] = D::var(qR[v], 5 + v);
    }
    const FvPrim<D> A = fvPrimOf(a[0], a[1], a[2], a[3], a[4], kGamma, 1.0);
    const FvPrim<D> B = fvPrimOf(b[0], b[1], b[2], b[3], b[4], kGamma, 1.0);
    D gud[3][3], gTd[3];
    for (int i = 0; i < 3; ++i) {
        gTd[i] = D(gT[i]);
        for (int j = 0; j < 3; ++j) gud[i][j] = D(gu[i][j]);
    }
    D mu, kap;
    fvTransportAt(tr, D(0.5) * (A.temp + B.temp), mu, kap);
    const FvFlux<D> fd = fvMeanFlux(A, B, S, gud, gTd, mu, kap);
    const D Fd[5] = {fd.rho, fd.m[0], fd.m[1], fd.m[2], fd.E};
    double worst = 0.0, F0[5];
    fluxD(qL, qR, F0);
    for (int k = 0; k < 10; ++k) {
        double ap[5], am[5], bp[5], bm[5], Fp[5], Fm[5];
        std::copy(qL, qL + 5, ap), std::copy(qL, qL + 5, am), std::copy(qR, qR + 5, bp), std::copy(qR, qR + 5, bm);
        const double h = 1e-6;
        (k < 5 ? ap[k] : bp[k - 5]) += h;
        (k < 5 ? am[k] : bm[k - 5]) -= h;
        fluxD(ap, bp, Fp);
        fluxD(am, bm, Fm);
        for (int r = 0; r < 5; ++r) {
            const double fdv = (Fp[r] - Fm[r]) / (2 * h);
            worst = std::max(worst, std::abs(Fd[r].d[k] - fdv) / (std::abs(fdv) + 1.0));
        }
    }
    double vErr = 0.0;
    for (int r = 0; r < 5; ++r) vErr = std::max(vErr, std::abs(Fd[r].v - F0[r]) / (std::abs(F0[r]) + 1.0));
    check(vErr <= 1e-14 && worst <= 1e-8,
          std::format("fvMeanFlux<Dual<10>>: value vs double {:.2e} <= 1e-14, d/dq vs central FD {:.2e} <= 1e-8",
                      vErr, worst));

    // Far boundary state through Dual (pow on Dual) vs FD.
    FvBc far;
    far.kind = FvBcKind::Far;
    const double qf[5] = {1.0, 0.5, 0.1, 0.0, 1.0 / (kGamma - 1.0) / kGamma + 0.5 * 0.26};
    std::copy(qf, qf + 5, far.qFar);
    const double Cf[3] = {0, 0, 0};
    auto farFlux = [&](const double* q, double F[5]) {
        const FvPrim<double> P = fvPrimOf(q[0], q[1], q[2], q[3], q[4], kGamma, 1.0);
        const FvPrim<double> Bs = fvBoundaryState(far, P, S, Cf, kGamma, 1.0);
        const FvFlux<double> f = fvBoundaryFlux(far, P, Bs, S, gu, gT, 0.0, 0.0);
        F[0] = f.rho, F[1] = f.m[0], F[2] = f.m[1], F[3] = f.m[2], F[4] = f.E;
    };
    const FvPrim<D> Pd = A;
    const FvPrim<D> Bd = fvBoundaryState(far, Pd, S, Cf, kGamma, 1.0);
    const FvFlux<D> ffd = fvBoundaryFlux(far, Pd, Bd, S, gud, gTd, D(0.0), D(0.0));
    const D Ff[5] = {ffd.rho, ffd.m[0], ffd.m[1], ffd.m[2], ffd.E};
    double wf = 0.0;
    for (int k = 0; k < 5; ++k) {
        double qp[5], qm[5], Fp[5], Fm[5];
        std::copy(qL, qL + 5, qp), std::copy(qL, qL + 5, qm);
        qp[k] += 1e-6, qm[k] -= 1e-6;
        farFlux(qp, Fp);
        farFlux(qm, Fm);
        for (int r = 0; r < 5; ++r) {
            const double fdv = (Fp[r] - Fm[r]) / 2e-6;
            wf = std::max(wf, std::abs(Ff[r].d[k] - fdv) / (std::abs(fdv) + 1.0));
        }
    }
    check(wf <= 1e-8, std::format("far-field boundary flux on Dual (subsonic characteristic state): d/dq vs FD "
                                  "{:.2e} <= 1e-8",
                                  wf));

    // float: the same templates, single precision.
    const float Sf[3] = {0.3f, -0.7f, 0.2f};
    const FvPrim<float> Af = fvPrimOf(float(qL[0]), float(qL[1]), float(qL[2]), float(qL[3]), float(qL[4]), kGamma, 1.0);
    const FvPrim<float> Bf = fvPrimOf(float(qR[0]), float(qR[1]), float(qR[2]), float(qR[3]), float(qR[4]), kGamma, 1.0);
    float guf[3][3], gTf[3], muf, kapf;
    for (int i = 0; i < 3; ++i) {
        gTf[i] = float(gT[i]);
        for (int j = 0; j < 3; ++j) guf[i][j] = float(gu[i][j]);
    }
    fvTransportAt(tr, 0.5f * (Af.temp + Bf.temp), muf, kapf);
    const FvFlux<float> ff = fvMeanFlux(Af, Bf, Sf, guf, gTf, muf, kapf);
    const float Fflt[5] = {ff.rho, ff.m[0], ff.m[1], ff.m[2], ff.E};
    double fe = 0.0;
    for (int r = 0; r < 5; ++r) fe = std::max(fe, std::abs(double(Fflt[r]) - F0[r]) / (std::abs(F0[r]) + 1e-3));
    check(fe <= 1e-5, std::format("fvMeanFlux<float, float> vs double: {:.2e} <= 1e-5", fe));
}

} // namespace

int mms(); // G4, below

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "mms") return mms();
    g1();
    g2();
    g3();
    g5();
    g6();
    g7();
    e1();
    e2();
    e3();
    e4();
    e5();
    e6();
    return dnstest::finish();
}

// ---------------------------------------------------------------------------
// G4: method of manufactured solutions. A steady 2D compressible NS solution
// on the unit periodic square, held by a source; the FV source is the EXACT
// face-integrated flux of the manufactured field (3-point Gauss along each
// straight face), so the exact cell balance closes and the discrete steady
// state differs from the field by the scheme's error alone. Three nested
// sinusoidal meshes (amp 0.06: ~20 deg non-orthogonal) marched to steady.

namespace {

struct Mms
{
    // q = a + b sin(k x + px) sin(k y + py), k = 2 pi.
    struct F
    {
        double a, b, px, py;
        double v(double x, double y) const { return a + b * std::sin(2 * pi * x + px) * std::sin(2 * pi * y + py); }
        double dx(double x, double y) const
        {
            return b * 2 * pi * std::cos(2 * pi * x + px) * std::sin(2 * pi * y + py);
        }
        double dy(double x, double y) const
        {
            return b * 2 * pi * std::sin(2 * pi * x + px) * std::cos(2 * pi * y + py);
        }
    };
    F rho{1.0, 0.1, 0.0, 0.3}, u{0.3, 0.1, pi / 2, 0.7}, v{0.2, 0.08, 1.1, pi / 2},
        p{1.0 / kGamma, 0.07, 0.4, 1.9};
    Transport tr = Transport::constant(kGamma, 0.01);

    void conserved(double x, double y, double q[5]) const
    {
        const double r = rho.v(x, y), uu = u.v(x, y), vv = v.v(x, y), pp = p.v(x, y);
        q[0] = r;
        q[1] = r * uu;
        q[2] = r * vv;
        q[3] = 0.0;
        q[4] = pp / (kGamma - 1.0) + 0.5 * r * (uu * uu + vv * vv);
    }

    // Exact F . n (unit n, n_z = 0).
    void flux(double x, double y, const double n[2], double F5[5]) const
    {
        const double r = rho.v(x, y), uu = u.v(x, y), vv = v.v(x, y), pp = p.v(x, y);
        const double T = pp / r, e = pp / ((kGamma - 1.0) * r) + 0.5 * (uu * uu + vv * vv);
        const double U = uu * n[0] + vv * n[1];
        const double ux = u.dx(x, y), uy = u.dy(x, y), vx = v.dx(x, y), vy = v.dy(x, y);
        const double Tx = (p.dx(x, y) - T * rho.dx(x, y)) / r, Ty = (p.dy(x, y) - T * rho.dy(x, y)) / r;
        const double mu = tr.muRef, kappa = tr.kappa(T), div = ux + vy;
        const double txx = mu * (2 * ux) - 2.0 / 3.0 * mu * div, tyy = mu * (2 * vy) - 2.0 / 3.0 * mu * div;
        const double txy = mu * (uy + vx);
        const double tn[2] = {txx * n[0] + txy * n[1], txy * n[0] + tyy * n[1]};
        F5[0] = r * U;
        F5[1] = r * U * uu + pp * n[0] - tn[0];
        F5[2] = r * U * vv + pp * n[1] - tn[1];
        F5[3] = 0.0;
        F5[4] = r * U * e + pp * U - (uu * tn[0] + vv * tn[1]) - kappa * (Tx * n[0] + Ty * n[1]);
    }
};

struct Level
{
    int N;
    double err[4];
    long steps;
    double seconds;
    double change;  // last chunk's |dq| / |error|
    bool converged;
};

Level mmsLevel(const Mms& mm, int N, const FvState* coarse, const FaceMesh* fmCoarse, int threads,
               FvFaceGradRule rule, double alpha, FvGradScheme gs, FvState* keep)
{
    const Grid g = Grid::make(N, N, 1, 1.0, 1.0, 1.0 / N);
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::sinusoidal(g, 0.06), periodicSpec(g), CellOrder::Morton),
                              "mms");
    const FvRef ref = orDie(FvRef::make(fm, config(mm.tr, bcFor(fm, {}), gs, rule, alpha), threads), "mms");
    const std::size_t nC = std::size_t(fm.nC);
    // Source: (1/V) sum_out of the exactly integrated face flux.
    FvState src = FvState::zeros(fm.nC), ex = FvState::zeros(fm.nC);
    const double gx[3] = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)}, gw[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
    for (std::size_t f = 0; f < std::size_t(fm.nI); ++f) {
        const double S[2] = {fm.S[0][f], fm.S[1][f]}, mS = std::hypot(S[0], S[1]);
        const double n[2] = {S[0] / mS, S[1] / mS}, t[2] = {-n[1], n[0]};
        const double half = 0.5 * mS / g.hz; // half edge length (the face is the edge extruded by hz)
        double I[5] = {0, 0, 0, 0, 0};
        for (int q = 0; q < 3; ++q) {
            double F5[5];
            mm.flux(fm.Cf[0][f] + half * gx[q] * t[0], fm.Cf[1][f] + half * gx[q] * t[1], n, F5);
            for (int v = 0; v < 5; ++v) I[v] += gw[q] * F5[v];
        }
        for (int v = 0; v < 5; ++v) {
            I[v] *= 0.5 * mS;
            src.var(v)[fm.own[f]] += I[v];
            src.var(v)[fm.nei[f]] -= I[v];
        }
    }
    for (std::size_t c = 0; c < nC; ++c) {
        double q[5];
        mm.conserved(fm.C[0][c], fm.C[1][c], q);
        for (int v = 0; v < 5; ++v) {
            src.var(v)[c] /= fm.V[c];
            ex.var(v)[c] = q[v];
        }
    }
    // Start: the exact field, plus (when a coarser steady state exists) its
    // error / 4 from the parent cell, re-centred so the conserved totals stay
    // the exact field's (they fix the periodic steady state).
    FvState s = ex;
    if (coarse) {
        std::vector<std::size_t> parentOf(std::size_t(fmCoarse->nC));
        for (std::size_t c = 0; c < std::size_t(fmCoarse->nC); ++c) parentOf[std::size_t(fmCoarse->srcCell[c])] = c;
        const int Nc = N / 2;
        for (int v = 0; v < 5; ++v) {
            double shift = 0.0, vol = 0.0;
            for (std::size_t c = 0; c < nC; ++c) {
                const auto I = ijk(g, fm.srcCell[c]);
                const std::size_t pc = parentOf[std::size_t((I[1] / 2) * Nc + I[0] / 2)];
                double qc[5];
                mm.conserved(fmCoarse->C[0][pc], fmCoarse->C[1][pc], qc);
                const double add = 0.25 * (coarse->var(v)[pc] - qc[v]);
                s.var(v)[c] += add;
                shift += fm.V[c] * add;
                vol += fm.V[c];
            }
            for (std::size_t c = 0; c < nC; ++c) s.var(v)[c] -= shift / vol;
        }
    }
    const double dt = orDie(ref.stableDt(s, {.cfl = 0.9, .diff = 0.9}), "dt");
    // Slowest decay: the k = 2 pi acoustic mode, 0.944 nu k^2 (Pr 0.72).
    const double efold = 1.0 / (0.944 * mm.tr.muRef * 4 * pi * pi);
    const int chunk = std::max(50, int(0.25 * efold / dt));
    auto errs = [&](const FvState& q, double e[4]) {
        const int vars[4] = {0, 1, 2, 4};
        for (int k = 0; k < 4; ++k) {
            double s2 = 0.0, vol = 0.0;
            for (std::size_t c = 0; c < nC; ++c) {
                const double d = q.var(vars[k])[c] - ex.var(vars[k])[c];
                s2 += fm.V[c] * d * d;
                vol += fm.V[c];
            }
            e[k] = std::sqrt(s2 / vol);
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    long steps = 0;
    double e[4], change = 1.0;
    bool converged = false;
    // Stop when one chunk (~1/4 of the slowest e-fold) moves the state by
    // <= 0.4% of its error.
    for (int it = 0; it < 400 && !converged; ++it) {
        const FvState prev = s;
        ref.march(s, dt, chunk, &src);
        steps += chunk;
        errs(s, e);
        double d2 = 0.0, e2 = 0.0;
        for (int v : {0, 1, 2, 4})
            for (std::size_t c = 0; c < nC; ++c) {
                const double dd = s.var(v)[c] - prev.var(v)[c], ee = s.var(v)[c] - ex.var(v)[c];
                d2 += fm.V[c] * dd * dd;
                e2 += fm.V[c] * ee * ee;
            }
        change = std::sqrt(d2 / e2);
        converged = change <= 0.004;
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (keep) *keep = s;
    return {N, {e[0], e[1], e[2], e[3]}, steps, sec, change, converged};
}

void mmsSeries(const Mms& mm, const char* tag, FvFaceGradRule rule, double alpha, FvGradScheme gs,
               std::initializer_list<int> Ns, bool gate)
{
    std::printf("  %s\n", tag);
    std::vector<Level> lv;
    FvState prev;
    FaceMesh prevMesh;
    for (const int N : Ns) {
        FvState keep;
        const Level L = mmsLevel(mm, N, lv.empty() ? nullptr : &prev, lv.empty() ? nullptr : &prevMesh, 0, rule,
                                 alpha, gs, &keep);
        std::printf("    N %4d: L2 error rho %.4e  rho u %.4e  rho v %.4e  E %.4e   (%ld steps, last chunk "
                    "|dq|/|e| %.1e, %.1f s)\n",
                    N, L.err[0], L.err[1], L.err[2], L.err[3], L.steps, L.change, L.seconds);
        if (!L.converged) check(false, std::format("{} N {}: not steady after {} steps", tag, N, L.steps));
        lv.push_back(L);
        prev = std::move(keep);
        const Grid g = Grid::make(N, N, 1, 1.0, 1.0, 1.0 / N);
        prevMesh =
            orDie(FaceMesh::fromCurvMesh(CurvMesh::sinusoidal(g, 0.06), periodicSpec(g), CellOrder::Morton), "mms");
    }
    for (std::size_t i = 1; i < lv.size(); ++i) {
        double omin = 1e300;
        std::string row;
        for (int k = 0; k < 4; ++k) {
            const double o = std::log2(lv[i - 1].err[k] / lv[i].err[k]);
            omin = std::min(omin, o);
            row += std::format(" {:.3f}", o);
        }
        const bool finest = i + 1 == lv.size();
        if (gate && finest)
            check(omin >= 1.8, std::format("{} {} -> {}: orders (rho, rho u, rho v, E){} >= 1.8", tag, lv[i - 1].N,
                                           lv[i].N, row));
        else
            std::printf("    %d -> %d: orders (rho, rho u, rho v, E)%s\n", lv[i - 1].N, lv[i].N, row.c_str());
    }
}

} // namespace

int mms()
{
    std::printf("G4 MMS: 2D compressible NS, mu 0.01 (Re ~ 36), M ~ 0.36, sinusoidal amp 0.06 periodic meshes\n");
    const Mms mm;
    mmsSeries(mm, "LSQ + AlongD (the pinned scheme)", FvFaceGradRule::AlongD, 1.0, FvGradScheme::LeastSquares,
              {32, 64, 128}, true);
    mmsSeries(mm, "Green-Gauss + AlongD (reported)", FvFaceGradRule::AlongD, 1.0, FvGradScheme::GreenGauss, {32, 64},
              false);
    mmsSeries(mm, "LSQ + AlongNormal alpha 4/3 (reported)", FvFaceGradRule::AlongNormal, 4.0 / 3.0,
              FvGradScheme::LeastSquares, {32, 64}, false);
    return dnstest::finish();
}
