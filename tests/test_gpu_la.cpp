// gpu/la (docs/la.md, masterplan_meshsolver P3), device half:
//   la1   LA1: SpMV, residual, block Jacobi, multicolour GS, line solves vs the
//         host FP64 restatements (tests/la_test_util.hpp) on structured 5/7-point
//         (2D, 3D, cyclic) and polyhedral graphs (cyl2d_snappy, box_refined), every
//         compiled shape, float (2e-6 of the row scale; lines 1e-6 of the line max)
//         and double (1e-12); batched FP64 dots vs a host long-double sum.
//   la2   LA2: FGMRES and BiCGStab on 2D convection-diffusion (Pe 1e3), PCG on
//         Poisson (2D, 3D), plus B = 3 and B = 5 systems: true residual of the
//         solved operator vs the recursive estimate, |gap| <= 1e-5 (in units of
//         |r0|); two runs byte-identical.
//   perf  LA-perf: B = 5 SpMV at 1M cells <= 1.3 ns/cell; batched lines
//         8192 x 128 (B = 5) <= 0.8 ms per apply. Run it through simjob --bench.
#include "gpu_test_util.hpp"
#include "la_test_util.hpp"

#include "dns/facemesh.hpp"
#include "dns/gpu/la/blockell.hpp"
#include "dns/gpu/la/krylov.hpp"
#include "dns/gpu/la/smooth.hpp"
#include "dns/la/graph.hpp"
#include "dns/la/structured.hpp"
#include "dns/polymesh.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dns;
using namespace dns::la;
using namespace dns::gpu::la;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

template <class E>
auto orDie(E&& e, const std::string& what)
{
    if (!e) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(e.error())>, hipError_t>)
            std::printf("FATAL %s: %s\n", what.c_str(), hipGetErrorString(e.error()));
        else std::printf("FATAL %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
    if constexpr (!std::is_void_v<typename std::remove_cvref_t<E>::value_type>) return std::move(*e);
}

template <class T>
gpu::DeviceBuffer<T> upload(const std::vector<double>& h)
{
    auto b = orDie(gpu::DeviceBuffer<T>::make(std::max<std::size_t>(1, h.size())), "alloc");
    std::vector<T> t(h.begin(), h.end());
    if (!t.empty() && b.upload(t.data()) != hipSuccess) orDie(std::expected<int, std::string>(std::unexpected("upload")), "");
    return b;
}

template <class T>
std::vector<double> download(const gpu::DeviceBuffer<T>& b, std::size_t n)
{
    std::vector<T> t(std::max<std::size_t>(1, b.size()));
    if (b.download(t.data()) != hipSuccess) orDie(std::expected<int, std::string>(std::unexpected("download")), "");
    return std::vector<double>(t.begin(), t.begin() + std::ptrdiff_t(n));
}

template <class T>
const char* tname() { return sizeof(T) == 4 ? "float" : "double"; }

FaceMesh loadMesh(const char* name, CellOrder order)
{
    const PolyMesh pm = orDie(readPolyMesh(fs::path(DNS_MESH_DATA) / name / "constant/polyMesh"), name);
    return orDie(FaceMesh::fromPolyMesh(pm, order), name);
}

// Chains on any graph: from each unassigned row ascending, walk to the lowest
// unassigned neighbour, at most 12 rows (a deterministic polyhedral line set).
LineSet walkLines(const CellGraph& g)
{
    std::vector<char> used(std::size_t(g.n), 0);
    std::vector<std::vector<int>> ch;
    for (int s = 0; s < g.n; ++s) {
        if (used[std::size_t(s)]) continue;
        std::vector<int> c{s};
        used[std::size_t(s)] = 1;
        while (c.size() < 12) {
            int nx = -1;
            for (int k = g.rowPtr[std::size_t(c.back())]; k < g.rowPtr[std::size_t(c.back()) + 1]; ++k)
                if (!used[std::size_t(g.col[std::size_t(k)])]) {
                    nx = g.col[std::size_t(k)];
                    break;
                }
            if (nx < 0) break;
            c.push_back(nx);
            used[std::size_t(nx)] = 1;
        }
        ch.push_back(std::move(c));
    }
    return orDie(lineSet(g, ch), "walk lines");
}

struct GraphCase
{
    std::string tag;
    CellGraph g;
    LineSet lines;
};

std::vector<GraphCase> la1Graphs()
{
    std::vector<GraphCase> v;
    v.push_back({"2D 5-point 37x23", structuredGraph({37, 23, 1}, {false, false, false}), gridLines({37, 23, 1}, 1)});
    v.push_back({"3D 7-point 13x9x7", structuredGraph({13, 9, 7}, {false, false, false}), gridLines({13, 9, 7}, 0)});
    v.push_back({"3D cyclic 8x6x5", structuredGraph({8, 6, 5}, {true, true, true}), gridLines({8, 6, 5}, 2)});
    v.push_back({"2D cyclic 2x40", structuredGraph({2, 40, 1}, {true, true, false}), gridLines({2, 40, 1}, 1)});
    for (const auto& [name, ord, oname] :
         {std::tuple{"cyl2d_snappy", CellOrder::Morton, "Morton"}, std::tuple{"cyl2d_snappy", CellOrder::ColourMajor, "ColourMajor"},
          std::tuple{"box_refined", CellOrder::Morton, "Morton"}}) {
        const FaceMesh fm = loadMesh(name, ord);
        CellGraph g = graphOf(fm);
        LineSet ls = walkLines(g);
        v.push_back({std::string(name) + " " + oname, std::move(g), std::move(ls)});
    }
    return v;
}

std::vector<BlockShape> allShapes()
{
    std::vector<BlockShape> s;
    for (const int B : supportedBlockSizes()) {
        s.push_back(*denseShape(B));
        if (B > 5) s.push_back(*blockShape(B, 5, true));
    }
    s.push_back(*blockShape(11, 5, false)); // per-row scalars
    return s;
}

template <class T>
void la1Case(const GraphCase& gc, BlockShape s, unsigned seed, double& wSpmv, double& wRes, double& wJac,
             double& wGs, double& wLine)
{
    const double tol = sizeof(T) == 4 ? 2e-6 : 1e-12, tolLine = sizeof(T) == 4 ? 1e-6 : 1e-12;
    const CellGraph& g = gc.g;
    const BlockMatrix m = latest::roundTo<T>(latest::randomMatrix(g, s, seed));
    const std::size_t N = std::size_t(g.n) * std::size_t(s.B);
    const std::string tag = gc.tag + " B" + std::to_string(s.B) + "/" + std::to_string(s.Bd) + "/" +
                            std::to_string(s.nS) + " " + tname<T>();
    auto A = orDie(BlockEll<T>::make(ellPattern(g), s), tag);
    orDie(A.upload(m), tag);
    const auto x = latest::roundVec<T>(latest::randomVec(N, seed + 1)),
               b = latest::roundVec<T>(latest::randomVec(N, seed + 2));
    auto dx = upload<T>(x), db = upload<T>(b);
    auto dy = upload<T>(std::vector<double>(N, 0.0)), dz = upload<T>(std::vector<double>(N, 0.0));
    // SpMV and residual
    std::vector<double> y, sc;
    latest::spmv(m, x, y, &sc);
    A.spmv(dx.data(), dy.data());
    const double eS = latest::maxRel(download(dy, N), y, sc);
    std::vector<double> r(N), rs(N);
    for (std::size_t i = 0; i < N; ++i) {
        r[i] = b[i] - y[i];
        rs[i] = std::abs(b[i]) + sc[i];
    }
    A.residual(db.data(), dx.data(), dy.data());
    const double eR = latest::maxRel(download(dy, N), r, rs);
    // block Jacobi
    auto J = orDie(BlockJacobi<T>::make(A), tag);
    orDie(J.factor(), tag + " jacobi");
    J.apply(db.data(), dz.data());
    std::vector<double> z, zs;
    latest::jacobi(m, b, z, zs);
    const double eJ = latest::maxRel(download(dz, N), z, zs);
    // multicolour GS: one symmetric sweep from x
    const std::vector<std::uint8_t> col = greedyColour(g);
    auto G = orDie(MultiColourGS<T>::make(A, col), tag);
    orDie(G.factor(), tag + " gs");
    auto dg = upload<T>(x);
    G.smooth(db.data(), dg.data(), 1);
    std::vector<double> xg = x, s1, s2;
    latest::mcgsSweep(m, col, b, xg, true, s1);
    latest::mcgsSweep(m, col, b, xg, false, s2);
    for (std::size_t i = 0; i < N; ++i) s2[i] = std::max(s1[i], s2[i]);
    const double eG = latest::maxRel(download(dg, N), xg, s2);
    // lines
    auto L = orDie(LineSmoother<T>::make(A, gc.lines), tag);
    orDie(L.factor(), tag + " lines");
    L.apply(db.data(), dz.data());
    std::vector<double> zl, lm;
    latest::lineSolve(m, gc.lines, b, zl, lm, s.B <= 7);
    const double eL = latest::maxRel(download(dz, N), zl, lm);
    wSpmv = std::max(wSpmv, eS);
    wRes = std::max(wRes, eR);
    wJac = std::max(wJac, eJ);
    wGs = std::max(wGs, eG);
    wLine = std::max(wLine, eL);
    const bool ok = eS <= tol && eR <= tol && eJ <= tol && eG <= tol && eL <= tolLine;
    if (!ok)
        std::printf("    %s: spmv %.2e res %.2e jac %.2e gs %.2e line %.2e\n", tag.c_str(), eS, eR, eJ, eG, eL);
    check(ok, tag);
}

template <class T>
void dotsCase()
{
    const std::size_t N = 100003;
    auto K = orDie(Krylov<T>::make(N, 0), "krylov");
    std::vector<gpu::DeviceBuffer<T>> bufs;
    std::vector<std::vector<double>> hv;
    for (int k = 0; k < 2 * kMaxDots; ++k) {
        hv.push_back(latest::roundVec<T>(latest::randomVec(N, 100u + unsigned(k))));
        bufs.push_back(upload<T>(hv.back()));
    }
    std::vector<DotPair<T>> pairs;
    for (int k = 0; k < kMaxDots; ++k) pairs.push_back({bufs[std::size_t(2 * k)].data(), bufs[std::size_t(2 * k + 1)].data()});
    std::vector<double> out(kMaxDots), out2(kMaxDots);
    orDie(K.dots(pairs, out), "dots");
    orDie(K.dots(pairs, out2), "dots");
    double worst = 0;
    for (int k = 0; k < kMaxDots; ++k) {
        long double s = 0, a = 0;
        for (std::size_t i = 0; i < N; ++i) {
            const long double t = (long double)hv[std::size_t(2 * k)][i] * (long double)hv[std::size_t(2 * k + 1)][i];
            s += t;
            a += t < 0 ? -t : t;
        }
        worst = std::max(worst, double(std::abs((long double)out[std::size_t(k)] - s) / a));
    }
    check(worst <= 1e-13, std::string("batched dots (") + tname<T>() + ", 66 x 100003): rel to sum|ab| " +
                              std::to_string(worst));
    check(std::memcmp(out.data(), out2.data(), out.size() * sizeof(double)) == 0, "batched dots byte-identical rerun");
}

int la1()
{
    std::printf("LA1: device vs host FP64 restatement\n");
    const auto graphs = la1Graphs();
    const auto shapes = allShapes();
    for (const auto& gc : graphs) {
        int multi = 0;
        for (int l = 0; l < gc.lines.count(); ++l) multi += gc.lines.start[std::size_t(l) + 1] - gc.lines.start[std::size_t(l)] > 1;
        std::printf("  graph %s: %d rows, nnz %zu, %d lines (%d multi-row, max %d)\n", gc.tag.c_str(), gc.g.n,
                    gc.g.col.size(), gc.lines.count(), multi, gc.lines.maxLength());
    }
    for (const bool dbl : {false, true}) {
        double wS = 0, wR = 0, wJ = 0, wG = 0, wL = 0;
        unsigned seed = 1;
        for (const auto& gc : graphs)
            for (const BlockShape s : shapes) {
                if (dbl) la1Case<double>(gc, s, seed++ * 7u, wS, wR, wJ, wG, wL);
                else la1Case<float>(gc, s, seed++ * 7u, wS, wR, wJ, wG, wL);
            }
        std::printf("  LA1 %s worst: spmv %.3e residual %.3e jacobi %.3e mcgs %.3e lines %.3e\n", dbl ? "double" : "float",
                    wS, wR, wJ, wG, wL);
    }
    dotsCase<float>();
    dotsCase<double>();
    return dnstest::finish();
}

// ---- LA2 ------------------------------------------------------------------

// 2D/3D finite-volume operators on the unit box, Dirichlet boundaries (h/2
// wall distance), diffusion nu per face, first-order upwind convection,
// Pe = |u| L / nu = 1 / nu. Flow (2D only):
//  Cell     the recirculating cell psi = sin(pi x) sin(pi y) / pi (|u| <= 1,
//           psi = 0 on the walls), face fluxes as node differences of psi --
//           discretely divergence-free, no boundary flux;
//  Uniform  u = (cos 30, sin 30): inflow faces feed the rhs (diagonal
//           untouched), outflow faces add F to the diagonal.
enum class Flow { None, Cell, Uniform };

BlockMatrix scalarOperator(std::array<int, 3> n, double nu, Flow flow)
{
    const CellGraph g = structuredGraph(n, {false, false, false});
    BlockMatrix m = BlockMatrix::zeros(g, *denseShape(1));
    const double h = 1.0 / n[0], pi = std::acos(-1.0);
    const int dims = n[2] == 1 ? 2 : 3;
    const double ux = std::cos(pi / 6), uy = std::sin(pi / 6);
    auto psi = [&](double x, double y) { return std::sin(pi * x) * std::sin(pi * y) / pi; };
    // Outward flux of cell (i, j) through its face in direction (ax, s).
    auto flux = [&](int i, int jj, int ax, int s) {
        if (flow == Flow::Uniform) return s * (ax == 0 ? ux : uy) * h;
        if (flow == Flow::None || ax == 2) return 0.0;
        if (ax == 0) {
            const double xf = (i + (s > 0 ? 1 : 0)) * h;
            return s * (psi(xf, (jj + 1) * h) - psi(xf, jj * h)); // int u dy
        }
        const double yf = (jj + (s > 0 ? 1 : 0)) * h;
        return -s * (psi((i + 1) * h, yf) - psi(i * h, yf)); // int v dx, v = -dpsi/dx
    };
    const int stride[3] = {1, n[0], n[0] * n[1]};
    for (int c = 0; c < g.n; ++c) {
        const int ijk[3] = {c % n[0], (c / n[0]) % n[1], c / (n[0] * n[1])};
        double diag = 0;
        for (int ax = 0; ax < dims; ++ax)
            for (const int s : {-1, 1}) {
                const double F = flux(ijk[0], ijk[1], ax, s);
                const int m2 = ijk[ax] + s;
                if (m2 < 0 || m2 >= n[ax]) { // boundary face
                    diag += 2.0 * nu + std::max(F, 0.0);
                    continue;
                }
                const int j = c + s * stride[ax];
                const int k = int(std::lower_bound(g.col.begin() + g.rowPtr[std::size_t(c)],
                                                   g.col.begin() + g.rowPtr[std::size_t(c) + 1], j) -
                                  g.col.begin());
                m.off[std::size_t(k)] = -(nu + std::max(-F, 0.0));
                diag += nu + std::max(F, 0.0);
            }
        m.diag[std::size_t(c)] = diag;
    }
    return m;
}

// B-block operator from a scalar one: off = a_cj I (dense), diagonal
// a_cc (I + K) with K SKEW (a Coriolis-like coupling, |K_rq| <= 0.2): the
// symmetric part stays the scalar operator's, so the coupled system is as
// well posed as the scalar one (a non-skew K on a weakly dominant operator
// is indefinite).
BlockMatrix blockOperator(const BlockMatrix& s, int B, unsigned seed)
{
    const CellGraph g{s.n, s.rowPtr, s.col, {}, {}};
    BlockMatrix m = BlockMatrix::zeros(g, *denseShape(B));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> K(std::size_t(B * B), 0.0);
    for (int r = 0; r < B; ++r)
        for (int q = r + 1; q < B; ++q) {
            K[std::size_t(r * B + q)] = 0.2 * u(rng);
            K[std::size_t(q * B + r)] = -K[std::size_t(r * B + q)];
        }
    for (int c = 0; c < s.n; ++c)
        for (int r = 0; r < B; ++r)
            for (int q = 0; q < B; ++q)
                m.diag[(std::size_t(c) * std::size_t(B) + std::size_t(r)) * std::size_t(B) + std::size_t(q)] =
                    s.diag[std::size_t(c)] * ((r == q ? 1.0 : 0.0) + K[std::size_t(r * B + q)]);
    for (std::size_t k = 0; k < s.nnz(); ++k)
        for (int r = 0; r < B; ++r) m.off[(k * std::size_t(B) + std::size_t(r)) * std::size_t(B) + std::size_t(r)] = s.off[k];
    return m;
}

// Anisotropic diffusion (y couplings ay x the x couplings), 2D, Dirichlet.
BlockMatrix anisotropic(std::array<int, 3> n, double ay)
{
    const CellGraph g = structuredGraph(n, {false, false, false});
    BlockMatrix m = BlockMatrix::zeros(g, *denseShape(1));
    for (int c = 0; c < g.n; ++c) {
        double d = 0;
        int nx = 0, ny = 0;
        for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k) {
            const bool yn = std::abs(g.col[std::size_t(k)] - c) == n[0];
            m.off[std::size_t(k)] = yn ? -ay : -1.0;
            d += yn ? ay : 1.0;
            (yn ? ny : nx)++;
        }
        d += (2 - nx) * 2.0 + (2 - ny) * 2.0 * ay;
        m.diag[std::size_t(c)] = d;
    }
    return m;
}

int gRestart = 30, gMaxIter = 3000; // FGMRES cycle length, iteration cap

enum class Method { Fgmres, Bicgstab, Pcg };
enum class Prec { None, Jacobi, Mcgs, Lines };

std::string mname(Method m)
{
    return m == Method::Fgmres ? "FGMRES(" + std::to_string(gRestart) + ")" : m == Method::Bicgstab ? "BiCGStab" : "PCG";
}
const char* pname(Prec p)
{
    return p == Prec::None ? "none" : p == Prec::Jacobi ? "block-Jacobi" : p == Prec::Mcgs ? "MC-SGS" : "lines";
}

struct SolveOut
{
    KrylovResult res;
    std::vector<unsigned char> bytes;
    double trueRel = 0;
};

template <class T>
SolveOut solveOnce(const BlockMatrix& m0, const CellGraph& g, const LineSet& lines, const std::vector<double>& b,
                   Method meth, Prec prec, double relTol)
{
    const BlockMatrix m = latest::roundTo<T>(m0);
    auto A = orDie(BlockEll<T>::make(ellPattern(g), m.shape), "A");
    orDie(A.upload(m), "A");
    const std::size_t N = A.size();
    auto db = upload<T>(b), dx = upload<T>(std::vector<double>(N, 0.0));
    auto J = orDie(BlockJacobi<T>::make(A), "J");
    auto G = orDie(MultiColourGS<T>::make(A, greedyColour(g)), "G");
    auto L = orDie(LineSmoother<T>::make(A, lines), "L");
    LinOp<T> Mop;
    if (prec == Prec::Jacobi) {
        orDie(J.factor(), "J");
        Mop = [&](const T* r, T* z) { J.apply(r, z); };
    } else if (prec == Prec::Mcgs) {
        orDie(G.factor(), "G");
        Mop = [&](const T* r, T* z) { G.apply(r, z); };
    } else if (prec == Prec::Lines) {
        orDie(L.factor(), "L");
        Mop = [&](const T* r, T* z) { L.apply(r, z); };
    }
    const LinOp<T> Aop = [&](const T* x, T* y) { A.spmv(x, y); };
    auto K = orDie(Krylov<T>::make(N, gRestart), "K");
    KrylovOptions o;
    o.restart = gRestart;
    o.maxIter = gMaxIter;
    o.relTol = relTol;
    const auto r = meth == Method::Fgmres ? K.fgmres(Aop, Mop, db.data(), dx.data(), o)
                   : meth == Method::Bicgstab ? K.bicgstab(Aop, Mop, db.data(), dx.data(), o)
                                              : K.pcg(Aop, Mop, db.data(), dx.data(), o);
    SolveOut out;
    out.res = orDie(r, "solve");
    std::vector<T> xt(N);
    if (dx.download(xt.data()) != hipSuccess) std::exit(1);
    out.bytes.resize(N * sizeof(T));
    std::memcpy(out.bytes.data(), xt.data(), out.bytes.size());
    const std::vector<double> x(xt.begin(), xt.end());
    std::vector<double> y;
    latest::spmv(m, x, y);
    double rn = 0;
    const std::vector<double> bt = latest::roundVec<T>(b);
    for (std::size_t i = 0; i < N; ++i) rn += (bt[i] - y[i]) * (bt[i] - y[i]);
    out.trueRel = std::sqrt(rn) / out.res.r0; // x0 = 0: r0 = |b| as the solver measured it
    return out;
}

// What a case gates: nothing (report + determinism), the estimate's
// honesty |true - estimate| <= 1e-5, or that plus convergence.
enum class Gate { Report, Gap, Converged };

template <class T>
void la2Case(const std::string& prob, const BlockMatrix& m, const CellGraph& g, const LineSet& lines, Method meth,
             Prec prec, double relTol, Gate gate, int restart = 30, int maxIt = 3000)
{
    gRestart = restart;
    gMaxIter = maxIt;
    const std::vector<double> b = latest::randomVec(std::size_t(g.n) * std::size_t(m.shape.B), 77u);
    const SolveOut a = solveOnce<T>(m, g, lines, b, meth, prec, relTol);
    const SolveOut c = solveOnce<T>(m, g, lines, b, meth, prec, relTol);
    const double gap = std::abs(a.trueRel - a.res.relResidual);
    const std::string tag = prob + " " + mname(meth) + "/" + pname(prec) + " " + tname<T>();
    std::printf("  %-34s %-10s %-12s %-6s it %5d est %.3e true %.3e gap %.2e%s%s\n", prob.c_str(), mname(meth).c_str(),
                pname(prec), tname<T>(), a.res.iterations, a.res.relResidual, a.trueRel, gap,
                a.res.restarts ? (" shadow restarts " + std::to_string(a.res.restarts)).c_str() : "",
                a.res.converged ? "" : a.res.breakdown ? " BREAKDOWN" : " NOT CONVERGED");
    const bool same = a.bytes == c.bytes && a.res.iterations == c.res.iterations &&
                      std::memcmp(&a.res.relResidual, &c.res.relResidual, sizeof(double)) == 0;
    check(same, tag + ": two runs byte-identical");
    if (gate == Gate::Gap) check(gap <= 1e-5, tag + ": |true - estimate| <= 1e-5");
    if (gate == Gate::Converged) check(a.res.converged && gap <= 1e-5, tag + ": converged, |true - estimate| <= 1e-5");
}

int la2()
{
    std::printf("LA2: Krylov true residual vs recursive estimate, determinism\n");
    const std::array<int, 3> n2{128, 128, 1}, n3{32, 32, 32}, n8{8, 8, 1};
    const CellGraph g2 = structuredGraph(n2, {false, false, false}), g3 = structuredGraph(n3, {false, false, false}),
                    g8 = structuredGraph(n8, {false, false, false});
    const LineSet lx2 = gridLines(n2, 0), lx3 = gridLines(n3, 0), lx8 = gridLines(n8, 0);
    const BlockMatrix uf = scalarOperator(n2, 1e-3, Flow::Uniform); // Pe 1e3, through-flow
    const BlockMatrix cd = scalarOperator(n2, 1e-3, Flow::Cell);    // Pe 1e3, recirculating
    const BlockMatrix po2 = scalarOperator(n2, 1.0, Flow::None);    // Poisson
    const BlockMatrix po3 = scalarOperator(n3, 1.0, Flow::None);
    // Finite termination: full GMRES on 64 unknowns converges in <= 64 steps.
    la2Case<double>("conv-diff 8^2 Pe 1e3 full GMRES", scalarOperator(n8, 1e-3, Flow::Cell), g8, lx8, Method::Fgmres,
                    Prec::None, 1e-10, Gate::Converged, 64, 64);
    for (const Method mt : {Method::Fgmres, Method::Bicgstab})
        for (const Prec p : {Prec::Jacobi, Prec::Mcgs, Prec::Lines}) {
            la2Case<double>("conv-diff 128^2 Pe 1e3 uniform B1", uf, g2, lx2, mt, p, 1e-8, Gate::Converged);
            la2Case<float>("conv-diff 128^2 Pe 1e3 uniform B1", uf, g2, lx2, mt, p, 1e-5, Gate::Report);
        }
    for (const Method mt : {Method::Fgmres, Method::Bicgstab})
        for (const Prec p : {Prec::Jacobi, Prec::Mcgs, Prec::Lines}) {
            // restarted GMRES stagnates on the closed streamlines (measured, any
            // restart <= 60): the estimate is gated, convergence only for BiCGStab
            la2Case<double>("conv-diff 128^2 Pe 1e3 cell B1", cd, g2, lx2, mt, p, 1e-8,
                            mt == Method::Bicgstab ? Gate::Converged : Gate::Gap);
            la2Case<float>("conv-diff 128^2 Pe 1e3 cell B1", cd, g2, lx2, mt, p, 1e-5, Gate::Report);
        }
    for (const Prec p : {Prec::None, Prec::Jacobi, Prec::Mcgs, Prec::Lines}) {
        la2Case<double>("Poisson 128^2 B1", po2, g2, lx2, Method::Pcg, p, 1e-8, Gate::Converged);
        la2Case<float>("Poisson 128^2 B1", po2, g2, lx2, Method::Pcg, p, 1e-5, Gate::Report);
    }
    la2Case<double>("Poisson 32^3 B1", po3, g3, lx3, Method::Pcg, Prec::Mcgs, 1e-8, Gate::Converged);
    la2Case<float>("Poisson 32^3 B1", po3, g3, lx3, Method::Pcg, Prec::Mcgs, 1e-5, Gate::Report);
    // block systems (skew, Coriolis-like coupling)
    const BlockMatrix uf3 = blockOperator(uf, 3, 5u), cd3 = blockOperator(cd, 3, 5u);
    la2Case<double>("conv-diff uniform B3 coupled", uf3, g2, lx2, Method::Fgmres, Prec::Mcgs, 1e-8, Gate::Converged);
    la2Case<double>("conv-diff uniform B3 coupled", uf3, g2, lx2, Method::Bicgstab, Prec::Mcgs, 1e-8, Gate::Converged);
    la2Case<float>("conv-diff uniform B3 coupled", uf3, g2, lx2, Method::Bicgstab, Prec::Mcgs, 1e-5, Gate::Report);
    la2Case<double>("conv-diff cell B3 coupled", cd3, g2, lx2, Method::Bicgstab, Prec::Mcgs, 1e-8, Gate::Converged);
    const std::array<int, 3> na{64, 256, 1};
    const CellGraph ga = structuredGraph(na, {false, false, false});
    const BlockMatrix an5 = blockOperator(anisotropic(na, 1e3), 5, 9u);
    la2Case<double>("aniso diffusion 64x256 ay 1e3 B5", an5, ga, gridLines(na, 1), Method::Fgmres, Prec::Mcgs, 1e-8,
                    Gate::Gap);
    la2Case<double>("aniso diffusion 64x256 ay 1e3 B5", an5, ga, gridLines(na, 1), Method::Fgmres, Prec::Lines, 1e-8,
                    Gate::Converged);
    la2Case<double>("aniso diffusion 64x256 ay 1e3 B5", an5, ga, gridLines(na, 1), Method::Bicgstab, Prec::Lines, 1e-8,
                    Gate::Converged);
    la2Case<float>("aniso diffusion 64x256 ay 1e3 B5", an5, ga, gridLines(na, 1), Method::Fgmres, Prec::Lines, 1e-5,
                   Gate::Report);
    return dnstest::finish();
}

// ---- LA-perf ----------------------------------------------------------------

template <class F>
double timeMs(F&& f, int reps)
{
    for (int i = 0; i < 3; ++i) f();
    (void)hipDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    (void)hipDeviceSynchronize();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

template <class T>
double spmvNs(const CellGraph& g, BlockShape s, double* gsRatio = nullptr)
{
    const BlockMatrix m = latest::randomMatrix(g, s, 3u);
    auto A = orDie(BlockEll<T>::make(ellPattern(g), s), "A");
    orDie(A.upload(m), "A");
    const std::size_t N = A.size();
    auto x = upload<T>(latest::randomVec(N, 4u)), y = upload<T>(std::vector<double>(N, 0.0));
    const double ms = timeMs([&] { A.spmv(x.data(), y.data()); }, 50);
    const double ns = ms * 1e6 / g.n;
    std::printf("  SpMV B%d/%d %-6s %8d rows: %.3f ms = %.3f ns/cell\n", s.B, s.Bd, tname<T>(), g.n, ms, ns);
    if (gsRatio) {
        auto G = orDie(MultiColourGS<T>::make(A, greedyColour(g)), "G");
        orDie(G.factor(), "G");
        const double gms = timeMs([&] { G.sweep(y.data(), x.data(), true); }, 20);
        *gsRatio = gms / ms;
        std::printf("  MC-GS one forward sweep (%d colours): %.3f ms = %.2f SpMV\n", G.colours(), gms, *gsRatio);
    }
    return ns;
}

int perf()
{
    std::printf("LA-perf (gfx timing via hipDeviceSynchronize; run under simjob --bench)\n");
    const CellGraph g = structuredGraph({100, 100, 100}, {true, true, true});
    const double ns5 = spmvNs<float>(g, *denseShape(5));
    spmvNs<float>(g, *denseShape(1));
    spmvNs<float>(g, *denseShape(3));
    spmvNs<double>(g, *denseShape(1));
    spmvNs<double>(g, *denseShape(3));
    spmvNs<float>(g, *blockShape(7, 5, true));
    spmvNs<float>(g, *blockShape(11, 5, true));
    double gsRatio = 0;
    spmvNs<float>(g, *denseShape(5), &gsRatio);
    {   // MC-GS in colour-major numbering: each colour contiguous
        const auto col = greedyColour(g);
        const std::vector<int> perm = colourMajorOrder(col); // perm[new] = old
        std::vector<int> inv(perm.size());
        for (std::size_t i = 0; i < perm.size(); ++i) inv[std::size_t(perm[i])] = int(i);
        CellGraph pg;
        pg.n = g.n;
        pg.rowPtr.push_back(0);
        for (int nw = 0; nw < g.n; ++nw) {
            const int o = perm[std::size_t(nw)];
            std::vector<int> r;
            for (int k = g.rowPtr[std::size_t(o)]; k < g.rowPtr[std::size_t(o) + 1]; ++k)
                r.push_back(inv[std::size_t(g.col[std::size_t(k)])]);
            std::ranges::sort(r);
            pg.col.insert(pg.col.end(), r.begin(), r.end());
            pg.rowPtr.push_back(int(pg.col.size()));
        }
        double cmRatio = 0;
        spmvNs<float>(pg, *denseShape(5), &cmRatio);
        std::printf("  (colour-major numbering above)\n");
    }
    check(ns5 <= 1.3, "LA-perf: B5 float SpMV at 1M cells <= 1.3 ns/cell (" + std::to_string(ns5) + ")");

    // batched lines 8192 x 128, B = 5
    const std::array<int, 3> nl{8192, 128, 1};
    const CellGraph gl = structuredGraph(nl, {false, false, false});
    const BlockShape s5 = *denseShape(5);
    const BlockMatrix m = latest::randomMatrix(gl, s5, 5u);
    auto A = orDie(BlockEll<float>::make(ellPattern(gl), s5), "A");
    orDie(A.upload(m), "A");
    auto L = orDie(LineSmoother<float>::make(A, gridLines(nl, 1)), "L");
    const double fms = timeMs([&] { (void)L.factor(); }, 5);
    const std::size_t N = A.size();
    auto r = upload<float>(latest::randomVec(N, 6u)), z = upload<float>(std::vector<double>(N, 0.0));
    const double lms = timeMs([&] { L.apply(r.data(), z.data()); }, 50);
    std::printf("  lines 8192 x 128 B5 float: factor %.3f ms, apply %.3f ms (%.3f ns/cell)\n", fms, lms,
                lms * 1e6 / gl.n);
    check(lms <= 0.8, "LA-perf: batched lines 8192 x 128 B5 apply <= 0.8 ms (" + std::to_string(lms) + ")");

    // batched dots: one CGS pass at j = 30 (31 distinct basis vectors against
    // w), B = 5 at 1M cells: N = 5.24M per vector, 650 MB of basis.
    auto K = orDie(Krylov<float>::make(N, 0), "K");
    auto V = orDie(gpu::DeviceBuffer<float>::make(31 * N), "basis");
    (void)hipMemset(V.data(), 0, V.size() * sizeof(float));
    std::vector<DotPair<float>> pairs;
    for (std::size_t k = 0; k < 31; ++k) pairs.push_back({V.data() + k * N, z.data()});
    std::vector<double> out(31);
    const double dms = timeMs([&] { (void)K.dots(pairs, out); }, 20);
    std::printf("  batched dots 31 distinct x %zu float: %.3f ms incl. the one host copy (%.0f GB/s of basis)\n", N,
                dms, 31.0 * double(N) * 4 / (dms * 1e6));
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "la1";
    if (mode == "la1") return la1();
    if (mode == "la2") return la2();
    if (mode == "perf") return perf();
    std::printf("usage: test_gpu_la la1|la2|perf\n");
    return 1;
}
