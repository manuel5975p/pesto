// gpu/la AMG (docs/la.md, masterplan_meshsolver P5), device half:
//   refresh  the device coarse values == the host Galerkin restatement of the
//            level above, BIT FOR BIT, on structured and polyhedral graphs and
//            every compiled shape (float, double); the dense coarsest inverse
//            vs a host dense solve (also with constantNullSpace); determinism.
//   la3      LA3: AMG-PCG on 2D Poisson 64^2 .. 1024^2, iterations grow <= 1.5x
//            per 16x cells; 3D 64^3 z-stretched aspect 100 (uniform and graded)
//            <= 2x the isotropic count; block B = 3 coupled systems (FGMRES).
//   la4      LA4: AMG-PCG vs the FFT-PCG cut-cell pressure solve
//            (gpu/cutpoisson, periodic 256^2, cylinder): same p; both counts.
//   meshes   AMG-PCG on OpenFOAM's Laplacian of the checked-in polyhedral
//            meshes (+ any polyMesh dirs given after `meshes`): converged in
//            <= 100 iterations, byte-identical reruns; `--out FILE` writes x in
//            source cell order (for the OpenFOAM GAMG comparison), `--l1hist N`
//            the OpenFOAM-normalised L1 residual after 1..N iterations.
//   fp32     the FV path's p solve (scaled V, Jacobi 2) in float with OpenFOAM's
//            measure at the FP32 floor: converged, vs FP64, byte-identical, and
//            the true-residual floor eps_f32 sum|A||x|/sum|b| (fp32Mode).
//   tune     report only (not a ctest): iterations and solve time per cycle /
//            smoother / coarse-size choice on Poisson 1024^2, 96^3 and any
//            polyMesh dirs given -- the measurement behind the AmgOptions defaults.
#include "gpu_test_util.hpp"
#include "la_test_util.hpp"

#include "dns/bodies.hpp"
#include "dns/cutcell.hpp"
#include "dns/grid.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvprecision.hpp"
#include "dns/gpu/cutpoisson.hpp"
#include "dns/gpu/la/amg.hpp"
#include "dns/gpu/la/blockell.hpp"
#include "dns/gpu/la/krylov.hpp"
#include "dns/la/amg.hpp"
#include "dns/la/graph.hpp"
#include "dns/la/structured.hpp"
#include "dns/polymesh.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    t.resize(b.size());
    if (b.upload(t.data()) != hipSuccess) std::exit(1);
    return b;
}

template <class T>
std::vector<double> fetch(const T* d, std::size_t n)
{
    std::vector<T> t(std::max<std::size_t>(1, n));
    if (n && hipMemcpy(t.data(), d, n * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess) std::exit(1);
    return std::vector<double>(t.begin(), t.begin() + std::ptrdiff_t(n));
}

template <class T>
const char* tname() { return sizeof(T) == 4 ? "float" : "double"; }

std::string sci(double v)
{
    char b[32];
    std::snprintf(b, sizeof b, "%.3e", v);
    return b;
}

double ms(std::chrono::steady_clock::time_point t0)
{
    (void)hipDeviceSynchronize();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

FaceMesh loadMesh(const fs::path& dir, CellOrder order)
{
    const PolyMesh pm = orDie(readPolyMesh(dir), dir.string());
    return orDie(FaceMesh::fromPolyMesh(pm, order), dir.string());
}

// The device matrix back on the host (BlockMatrix layout on g).
template <class T>
BlockMatrix download(const BlockEll<T>& A, const CellGraph& g)
{
    const EllView<T> v = A.view();
    const BlockShape s = A.shape();
    BlockMatrix m = BlockMatrix::zeros(g, s);
    const std::size_t n = std::size_t(v.n), nE = std::size_t(v.nE), B = std::size_t(s.B);
    const auto diag = fetch(v.diag, n * B * B), off = fetch(v.off, nE * std::size_t(s.offDense())),
               sc = fetch(v.sc, nE * std::size_t(s.nS));
    for (std::size_t c = 0; c < n; ++c)
        for (std::size_t e = 0; e < B * B; ++e) m.diag[c * B * B + e] = diag[e * n + c];
    for (std::size_t q = 0; q < g.col.size(); ++q) {
        const std::size_t E = std::size_t(A.pattern().slotPos[q]);
        for (std::size_t e = 0; e < std::size_t(s.offDense()); ++e) m.off[q * std::size_t(s.offDense()) + e] = off[e * nE + E];
        for (std::size_t e = 0; e < std::size_t(s.nS); ++e) m.sc[q * std::size_t(s.nS) + e] = sc[e * nE + E];
    }
    return m;
}

double maxAbsDiff(const BlockMatrix& a, const BlockMatrix& b, double& scale)
{
    double e = 0;
    scale = 0;
    for (auto [x, y] : {std::pair{&a.diag, &b.diag}, std::pair{&a.off, &b.off}, std::pair{&a.sc, &b.sc}})
        for (std::size_t i = 0; i < x->size(); ++i) {
            e = std::max(e, std::abs((*x)[i] - (*y)[i]));
            scale = std::max(scale, std::abs((*y)[i]));
        }
    return e;
}

std::vector<BlockShape> allShapes()
{
    std::vector<BlockShape> s;
    for (const int B : supportedBlockSizes()) {
        s.push_back(*denseShape(B));
        if (B > 5) s.push_back(*blockShape(B, 5, true));
    }
    s.push_back(*blockShape(11, 5, false));
    return s;
}

// ---- one AMG-PCG / FGMRES solve ------------------------------------------------

struct Cfg
{
    AmgSmoother sm = AmgSmoother::MultiColourGS, csm = AmgSmoother::Jacobi; // level 0, levels >= 1 (the defaults)
    int inner = 2, sweeps = 1, coarse = 128, innerMinRows = 10000;
    double omega = 0.67, lineRatio = 0;
    bool nullSpace = false;
    enum class Outer { Pcg, Fgmres, Bicgstab } outer = Outer::Pcg;
};

const char* smName(AmgSmoother s)
{
    return s == AmgSmoother::Jacobi ? "Jacobi" : s == AmgSmoother::MultiColourGS ? "MC-SGS" : "lines";
}

std::string cfgName(const Cfg& c)
{
    const char* cy = c.inner == 0 ? "V" : c.inner == 1 ? "scaledV" : "K";
    const char* ou = c.outer == Cfg::Outer::Pcg ? (c.inner == 2 ? "FCG" : "PCG") : c.outer == Cfg::Outer::Fgmres ? "FGMRES" : "BiCGStab";
    std::string s = std::string(ou) + "/" + cy + "/" + smName(c.sm);
    if (c.csm != c.sm) s += "+" + std::string(smName(c.csm));
    if (c.inner > 0 && c.innerMinRows != 10000) s += " Kmin " + std::to_string(c.innerMinRows);
    if (c.coarse != 128) s += " c" + std::to_string(c.coarse);
    return s;
}

Cfg smoothers(AmgSmoother fine, AmgSmoother coarse)
{
    Cfg c;
    c.sm = fine;
    c.csm = coarse;
    return c;
}

struct Run
{
    KrylovResult res;
    double trueRel = 0, l1 = 0, hostMs = 0, makeMs = 0, refreshMs = 0, solveMs = 0, opc = 0;
    int levels = 0;
    std::string rows;
    std::vector<double> x;
    std::vector<unsigned char> bytes;
};

// normFactor > 0: OpenFOAM's measure and absTol its `tolerance` (KrylovOptions).
template <class T>
Run amgSolve(const BlockMatrix& m0, const CellGraph& g, const std::vector<double>& b, const Cfg& c, double relTol,
             int maxIt = 500, double normFactor = 0, double absTol = 0)
{
    const BlockMatrix m = latest::roundTo<T>(m0);
    Run out;
    AmgSetup su;
    su.coarseUnknowns = c.coarse;
    su.lineRatio = c.lineRatio;
    auto t0 = std::chrono::steady_clock::now();
    const AmgHierarchy h = orDie(amgHierarchy(g, amgStrength(m), m.shape.B, su), "hierarchy");
    out.hostMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    out.levels = h.levels();
    out.opc = h.operatorComplexity();
    for (const int r : h.rows) out.rows += (out.rows.empty() ? "" : " ") + std::to_string(r);
    auto A = orDie(BlockEll<T>::make(ellPattern(g), m.shape), "A");
    orDie(A.upload(m), "A");
    AmgOptions o;
    o.smoother = c.sm;
    o.coarseSmoother = c.csm;
    o.inner = c.inner;
    o.sweeps = c.sweeps;
    o.omega = c.omega;
    o.innerMinRows = c.innerMinRows;
    o.constantNullSpace = c.nullSpace;
    t0 = std::chrono::steady_clock::now();
    auto amg = orDie(Amg<T>::make(A, h, o), "amg make");
    out.makeMs = ms(t0);
    t0 = std::chrono::steady_clock::now();
    orDie(amg.refresh(), "amg refresh");
    out.refreshMs = ms(t0);
    const std::size_t N = A.size();
    auto db = upload<T>(b), dx = upload<T>(std::vector<double>(N, 0.0));
    auto K = orDie(Krylov<T>::make(N, c.outer == Cfg::Outer::Fgmres ? 30 : 0), "krylov");
    KrylovOptions ko;
    ko.relTol = relTol;
    ko.maxIter = maxIt;
    ko.flexible = c.inner == 2;
    ko.normFactor = normFactor;
    ko.absTol = absTol;
    const LinOp<T> Aop = [&](const T* x, T* y) { A.spmv(x, y); };
    const LinOp<T> Mop = [&](const T* r, T* z) { amg.apply(r, z); };
    t0 = std::chrono::steady_clock::now();
    auto r = c.outer == Cfg::Outer::Pcg      ? K.pcg(Aop, Mop, db.data(), dx.data(), ko)
             : c.outer == Cfg::Outer::Fgmres ? K.fgmres(Aop, Mop, db.data(), dx.data(), ko)
                                             : K.bicgstab(Aop, Mop, db.data(), dx.data(), ko);
    out.solveMs = ms(t0);
    out.res = orDie(r, "solve");
    std::vector<T> xt(N);
    if (dx.download(xt.data()) != hipSuccess) std::exit(1);
    out.bytes.resize(N * sizeof(T));
    std::memcpy(out.bytes.data(), xt.data(), out.bytes.size());
    out.x.assign(xt.begin(), xt.end());
    std::vector<double> y;
    latest::spmv(m, out.x, y);
    const std::vector<double> bt = latest::roundVec<T>(b);
    double r2 = 0, b2 = 0, r1 = 0, b1 = 0;
    for (std::size_t i = 0; i < N; ++i) {
        r2 += (bt[i] - y[i]) * (bt[i] - y[i]);
        b2 += bt[i] * bt[i];
        r1 += std::abs(bt[i] - y[i]);
        b1 += std::abs(bt[i]);
    }
    out.trueRel = std::sqrt(r2 / b2);
    out.l1 = r1 / b1; // OpenFOAM's normalised residual at x0 = 0
    return out;
}

void printRun(const std::string& prob, const Cfg& c, const Run& r, const char* tn)
{
    std::printf("  %-34s %-22s %-6s levels %2d op %.2f  it %4d  est %.2e true %.2e L1 %.2e  host %.1f ms refresh %.2f ms "
                "solve %.2f ms%s\n",
                prob.c_str(), cfgName(c).c_str(), tn, r.levels, r.opc, r.res.iterations, r.res.relResidual, r.trueRel,
                r.l1, r.hostMs, r.refreshMs, r.solveMs, r.res.converged ? "" : r.res.breakdown ? " BREAKDOWN" : " NOT CONVERGED");
}

// ---- refresh ------------------------------------------------------------------

template <class T>
void refreshCase(const std::string& tag0, const CellGraph& g, BlockShape s, unsigned seed, double& worst)
{
    const std::string tag = tag0 + " B" + std::to_string(s.B) + "/" + std::to_string(s.Bd) + "/" + std::to_string(s.nS) +
                            " " + tname<T>();
    const BlockMatrix m = latest::roundTo<T>(latest::randomMatrix(g, s, seed));
    AmgSetup su;
    su.coarseUnknowns = std::max(8, 2 * s.B);
    const AmgHierarchy h = orDie(amgHierarchy(g, amgStrength(m), s.B, su), tag);
    auto A = orDie(BlockEll<T>::make(ellPattern(g), s), tag);
    orDie(A.upload(m), tag);
    AmgOptions o;
    o.smoother = AmgSmoother::Jacobi;
    auto amg = orDie(Amg<T>::make(A, h, o), tag);
    orDie(amg.refresh(), tag);
    double e = 0;
    BlockMatrix prev = m;
    for (int l = 1; l < amg.levels(); ++l) {
        const CellGraph& gc = h.graphs[std::size_t(l - 1)];
        const BlockMatrix dev = download(amg.matrix(l), gc);
        const BlockMatrix ref = latest::roundTo<T>(amgGalerkin(prev, gc, h.transfers[std::size_t(l - 1)]));
        double sc = 0;
        e = std::max(e, maxAbsDiff(dev, ref, sc));
        prev = dev;
    }
    worst = std::max(worst, e);
    check(e == 0.0 && amg.levels() >= 2, tag + ": " + std::to_string(amg.levels()) + " levels, device coarse == host Galerkin bit for bit");
}

// One-level hierarchy: apply == the dense inverse.
template <class T>
void denseCase(int B, bool nullSpace)
{
    const std::array<int, 3> n{9, 7, 1};
    const CellGraph g = structuredGraph(n, {false, false, false});
    const std::string tag = std::string("dense coarsest 9x7 B") + std::to_string(B) + (nullSpace ? " null space" : "") +
                            " " + tname<T>();
    BlockMatrix m = latest::roundTo<T>(latest::randomMatrix(g, *denseShape(B), 21u + unsigned(B)));
    if (nullSpace) { // Neumann Laplacian: zero row sums, constant null space
        m = BlockMatrix::zeros(g, *denseShape(1));
        for (int c = 0; c < g.n; ++c)
            for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k) {
                m.off[std::size_t(k)] = -1.0 - 0.25 * ((c + k) % 3);
                m.diag[std::size_t(c)] += 0;
            }
        for (int c = 0; c < g.n; ++c) // symmetric: a_cj = a_jc
            for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k) {
                const int j = g.col[std::size_t(k)];
                if (j < c) m.off[std::size_t(k)] = m.off[std::size_t(latest::slotOf(m, j, c))];
            }
        for (int c = 0; c < g.n; ++c)
            for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k) m.diag[std::size_t(c)] -= m.off[std::size_t(k)];
    }
    AmgSetup su;
    su.coarseUnknowns = 1024;
    const AmgHierarchy h = orDie(amgHierarchy(g, amgStrength(m), m.shape.B, su), tag);
    auto A = orDie(BlockEll<T>::make(ellPattern(g), m.shape), tag);
    orDie(A.upload(m), tag);
    AmgOptions o;
    o.constantNullSpace = nullSpace;
    auto amg = orDie(Amg<T>::make(A, h, o), tag);
    orDie(amg.refresh(), tag);
    const std::size_t N = A.size();
    std::vector<double> r = latest::roundVec<T>(latest::randomVec(N, 5u));
    if (nullSpace) {
        double mean = 0;
        for (const double v : r) mean += v;
        mean /= double(N);
        for (double& v : r) v = double(T(v - mean));
    }
    auto dr = upload<T>(r), dz = upload<T>(std::vector<double>(N, 0.0)), dz2 = upload<T>(std::vector<double>(N, 0.0));
    amg.apply(dr.data(), dz.data());
    amg.apply(dr.data(), dz2.data());
    const auto z = fetch(dz.data(), N), z2 = fetch(dz2.data(), N);
    std::vector<double> y;
    latest::spmv(m, z, y);
    double e = 0, sc = 0;
    for (std::size_t i = 0; i < N; ++i) {
        e = std::max(e, std::abs(y[i] - r[i]));
        sc = std::max(sc, std::abs(r[i]));
    }
    const double tol = sizeof(T) == 4 ? 1e-4 : 1e-11;
    check(amg.levels() == 1 && e / sc <= tol, tag + ": |A z - r| / |r| " + sci(e / sc));
    check(z == z2, tag + ": two applies byte-identical");
    if (!nullSpace) { // a singular matrix without the flag is an error by name
        BlockMatrix sing = m;
        for (int r0 = 0; r0 < m.shape.B; ++r0)
            for (int q = 0; q < m.shape.B; ++q) sing.diag[std::size_t(r0 * m.shape.B + q)] = 0;
        for (int k = sing.rowPtr[0]; k < sing.rowPtr[1]; ++k)
            for (std::size_t e2 = 0; e2 < std::size_t(m.shape.offDense()); ++e2)
                sing.off[std::size_t(k) * std::size_t(m.shape.offDense()) + e2] = 0;
        orDie(A.upload(sing), tag);
        const auto f = amg.refresh();
        check(!f && f.error().find("singular") != std::string::npos, tag + ": a zero row is rejected by name");
    }
}

int refreshMode()
{
    std::printf("AMG refresh: device coarse values vs the host Galerkin restatement\n");
    std::vector<std::pair<std::string, CellGraph>> gs;
    gs.emplace_back("2D 37x23", structuredGraph({37, 23, 1}, {false, false, false}));
    gs.emplace_back("3D 13x9x7", structuredGraph({13, 9, 7}, {false, false, false}));
    gs.emplace_back("3D cyclic 8x6x5", structuredGraph({8, 6, 5}, {true, true, true}));
    gs.emplace_back("cyl2d_snappy", graphOf(loadMesh(fs::path(DNS_MESH_DATA) / "cyl2d_snappy/constant/polyMesh", CellOrder::Morton)));
    gs.emplace_back("box_refined", graphOf(loadMesh(fs::path(DNS_MESH_DATA) / "box_refined/constant/polyMesh", CellOrder::Morton)));
    for (const bool dbl : {false, true}) {
        double worst = 0;
        unsigned seed = 1;
        for (const auto& [tag, g] : gs)
            for (const BlockShape s : allShapes()) {
                if (dbl) refreshCase<double>(tag, g, s, seed++ * 13u, worst);
                else refreshCase<float>(tag, g, s, seed++ * 13u, worst);
            }
        std::printf("  refresh %s: worst |device - host| = %.3e over %zu cases\n", dbl ? "double" : "float", worst,
                    gs.size() * allShapes().size());
    }
    for (const int B : {1, 3, 5}) {
        denseCase<double>(B, false);
        denseCase<float>(B, false);
    }
    denseCase<double>(1, true);
    denseCase<float>(1, true);
    // determinism of a full cycle on a polyhedral mesh: two hierarchies, two Amg objects
    {
        const CellGraph& g = gs[3].second;
        const BlockMatrix m = latest::randomMatrix(g, *denseShape(5), 99u);
        std::vector<unsigned char> first;
        for (int rep = 0; rep < 2; ++rep) {
            AmgSetup su;
            su.coarseUnknowns = 64;
            const AmgHierarchy h = orDie(amgHierarchy(g, amgStrength(m), 5, su), "det");
            auto A = orDie(BlockEll<double>::make(ellPattern(g), m.shape), "det");
            orDie(A.upload(m), "det");
            auto amg = orDie(Amg<double>::make(A, h, AmgOptions{}), "det");
            orDie(amg.refresh(), "det");
            auto r = upload<double>(latest::randomVec(A.size(), 4u)), z = upload<double>(std::vector<double>(A.size(), 0));
            amg.apply(r.data(), z.data());
            const auto zz = fetch(z.data(), A.size());
            std::vector<unsigned char> bytes(zz.size() * 8);
            std::memcpy(bytes.data(), zz.data(), bytes.size());
            if (rep == 0) first = bytes;
            else check(bytes == first, "cyl2d_snappy B5 K-cycle: two independent setups + cycles byte-identical");
        }
    }
    return dnstest::finish();
}

// ---- LA3 --------------------------------------------------------------------------

// The LA2 convection-diffusion operators (uniform flow / recirculating cell),
// scalar then lifted to B blocks with a skew coupling (test_gpu_la.cpp).
enum class Flow { Uniform, Cell };

BlockMatrix scalarOperator(std::array<int, 3> n, double nu, Flow flow)
{
    const CellGraph g = structuredGraph(n, {false, false, false});
    BlockMatrix m = BlockMatrix::zeros(g, *denseShape(1));
    const double h = 1.0 / n[0], pi = std::acos(-1.0);
    const double ux = std::cos(pi / 6), uy = std::sin(pi / 6);
    auto psi = [&](double x, double y) { return std::sin(pi * x) * std::sin(pi * y) / pi; };
    auto flux = [&](int i, int jj, int ax, int s) {
        if (flow == Flow::Uniform) return s * (ax == 0 ? ux : uy) * h;
        if (ax == 0) {
            const double xf = (i + (s > 0 ? 1 : 0)) * h;
            return s * (psi(xf, (jj + 1) * h) - psi(xf, jj * h));
        }
        const double yf = (jj + (s > 0 ? 1 : 0)) * h;
        return -s * (psi((i + 1) * h, yf) - psi(i * h, yf));
    };
    const int stride[2] = {1, n[0]};
    for (int c = 0; c < g.n; ++c) {
        const int ij[2] = {c % n[0], c / n[0]};
        double diag = 0;
        for (int ax = 0; ax < 2; ++ax)
            for (const int s : {-1, 1}) {
                const double F = flux(ij[0], ij[1], ax, s);
                const int m2 = ij[ax] + s;
                if (m2 < 0 || m2 >= n[std::size_t(ax)]) {
                    diag += 2.0 * nu + std::max(F, 0.0);
                    continue;
                }
                m.off[std::size_t(latest::slotOf(m, c, c + s * stride[ax]))] = -(nu + std::max(-F, 0.0));
                diag += nu + std::max(F, 0.0);
            }
        m.diag[std::size_t(c)] = diag;
    }
    return m;
}

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

std::array<std::vector<double>, 3> uniformH(std::array<int, 3> n, std::array<double, 3> h)
{
    std::array<std::vector<double>, 3> w;
    for (int a = 0; a < 3; ++a) w[std::size_t(a)].assign(std::size_t(n[std::size_t(a)]), h[std::size_t(a)]);
    return w;
}

int la3Mode()
{
    std::printf("LA3: AMG-PCG iterations (double, relTol 1e-8, b random)\n");
    std::vector<Cfg> cfgs(5);
    cfgs[1].inner = 0;
    cfgs[2].inner = 1;
    cfgs[3] = cfgs[4] = smoothers(AmgSmoother::Jacobi, AmgSmoother::Jacobi);
    cfgs[4].inner = 1;
    std::vector<std::vector<int>> its(cfgs.size());
    for (const int n : {64, 256, 1024}) {
        const std::array<int, 3> nn{n, n, 1};
        const CellGraph g = structuredGraph(nn, {false, false, false});
        const BlockMatrix m = latest::tensorLaplacian(g, nn, uniformH(nn, {1.0 / n, 1.0 / n, 1.0}));
        const std::vector<double> b = latest::randomVec(std::size_t(g.n), 7u);
        for (std::size_t k = 0; k < cfgs.size(); ++k) {
            const Run r = amgSolve<double>(m, g, b, cfgs[k], 1e-8);
            printRun("Poisson " + std::to_string(n) + "^2 Dirichlet", cfgs[k], r, "double");
            its[k].push_back(r.res.converged ? r.res.iterations : 100000);
            if (k == 0) {
                const Run r2 = amgSolve<double>(m, g, b, cfgs[k], 1e-8);
                check(r2.bytes == r.bytes && r2.res.iterations == r.res.iterations,
                      "Poisson " + std::to_string(n) + "^2: two solves byte-identical");
                check(r.res.converged && std::abs(r.trueRel - r.res.relResidual) <= 1e-8,
                      "Poisson " + std::to_string(n) + "^2: converged, true == estimate");
            }
        }
    }
    for (std::size_t k = 0; k < cfgs.size(); ++k) {
        const double g1 = double(its[k][1]) / its[k][0], g2 = double(its[k][2]) / its[k][1];
        std::printf("  growth per 16x cells %-22s: %d -> %d -> %d  (x%.3f, x%.3f)\n", cfgName(cfgs[k]).c_str(), its[k][0],
                    its[k][1], its[k][2], g1, g2);
        if (k == 0)
            check(g1 <= 1.5 && g2 <= 1.5, "LA3: default AMG-PCG iterations grow <= 1.5x per 16x cells (" + std::to_string(g1) +
                                              ", " + std::to_string(g2) + ")");
    }
    // FP32 hierarchy and iterate, relTol 1e-5 (report: the FP32 floor)
    {
        const std::array<int, 3> nn{1024, 1024, 1};
        const CellGraph g = structuredGraph(nn, {false, false, false});
        const BlockMatrix m = latest::tensorLaplacian(g, nn, uniformH(nn, {1.0 / 1024, 1.0 / 1024, 1.0}));
        const Run r = amgSolve<float>(m, g, latest::randomVec(std::size_t(g.n), 7u), cfgs[0], 1e-5);
        printRun("Poisson 1024^2 Dirichlet", cfgs[0], r, "float");
    }

    std::printf("LA3: 3D 64^3, z-stretched aspect 100\n");
    const std::array<int, 3> n3{64, 64, 64};
    const CellGraph g3 = structuredGraph(n3, {false, false, false});
    const double h = 1.0 / 64;
    std::array<std::vector<double>, 3> iso = uniformH(n3, {h, h, h}), uni = uniformH(n3, {h, h, h / 100}), grad = iso;
    // graded: hz from h / 100 at the k = 0 wall, geometric to h at the top
    const double ratio = std::pow(100.0, 1.0 / 63.0);
    for (int k = 0; k < 64; ++k) grad[2][std::size_t(k)] = h / 100 * std::pow(ratio, k);
    const std::vector<double> b3 = latest::randomVec(std::size_t(g3.n), 8u);
    Cfg mc, ln = smoothers(AmgSmoother::Lines, AmgSmoother::Lines), lnu;
    ln.lineRatio = 4;
    lnu = ln;
    lnu.omega = 1.0;
    int itIso = 0;
    for (const auto& [tag, hh] : {std::pair{"iso", &iso}, std::pair{"aspect 100 uniform", &uni},
                                  std::pair{"aspect 100 graded", &grad}}) {
        const BlockMatrix m = latest::tensorLaplacian(g3, n3, *hh);
        for (const Cfg* c : {&mc, &ln, &lnu}) {
            const Run r = amgSolve<double>(m, g3, b3, *c, 1e-8);
            printRun(std::string("64^3 ") + tag + (c == &lnu ? " (omega 1)" : ""), *c, r, "double");
            const std::string t = tag;
            if (t == "iso" && c == &mc) itIso = r.res.iterations;
            if (t != "iso" && c == &ln)
                check(r.res.converged && r.res.iterations <= 2 * itIso,
                      std::string("LA3: 64^3 ") + tag + " with lines: " + std::to_string(r.res.iterations) +
                          " <= 2 x isotropic " + std::to_string(itIso));
        }
    }

    std::printf("block systems (AMG on B = 3 blocks, strength = |block|_F)\n");
    const std::array<int, 3> n2{128, 128, 1};
    const CellGraph g2 = structuredGraph(n2, {false, false, false});
    const BlockMatrix uf3 = blockOperator(scalarOperator(n2, 1e-3, Flow::Uniform), 3, 5u),
                      cd3 = blockOperator(scalarOperator(n2, 1e-3, Flow::Cell), 3, 5u);
    const std::vector<double> bb = latest::randomVec(std::size_t(g2.n) * 3, 77u);
    for (const auto& [tag, mm] : {std::pair{"conv-diff uniform B3 coupled", &uf3}, std::pair{"conv-diff cell B3 coupled", &cd3}}) {
        Cfg f;
        f.outer = Cfg::Outer::Fgmres;
        const Run r = amgSolve<double>(*mm, g2, bb, f, 1e-8, 3000);
        printRun(tag, f, r, "double");
        check(r.res.converged, std::string(tag) + ": FGMRES + AMG K-cycle converged");
        Cfg bi;
        bi.outer = Cfg::Outer::Bicgstab;
        bi.inner = 0;
        const Run r2 = amgSolve<double>(*mm, g2, bb, bi, 1e-8, 3000);
        printRun(tag, bi, r2, "double");
        check(r2.res.converged, std::string(tag) + ": BiCGStab + AMG V-cycle converged");
        // convection: GS on the coarse levels too
        for (Cfg c : {f, bi}) {
            c.csm = AmgSmoother::MultiColourGS;
            const Run r3 = amgSolve<double>(*mm, g2, bb, c, 1e-8, 3000);
            printRun(tag, c, r3, "double");
            check(r3.res.converged, std::string(tag) + ": " + cfgName(c) + " converged");
        }
    }
    return dnstest::finish();
}

// ---- LA4 --------------------------------------------------------------------------

int la4Mode()
{
    std::printf("LA4: AMG-PCG vs FFT-PCG on the cut-cell pressure operator\n");
    const int n = 256;
    const Grid g = Grid::make(n, n, 1, 1.0, 1.0, 1.0 / n);
    const CylinderZ cyl{0.5, 0.5, 0.1};
    CutGeometry cut = CutGeometry::make(g, cyl);
    sanitizeCut(g, cut);
    // fluid compaction and the host operator A = -M (SPD on fluid)
    std::vector<int> id(g.size(), -1), cell;
    for (std::size_t c = 0; c < g.size(); ++c)
        if (cut.lambda[c] > 0) {
            id[c] = int(cell.size());
            cell.push_back(int(c));
        }
    const int nf = int(cell.size());
    const double ih2 = double(n) * double(n);
    CellGraph cg;
    cg.n = nf;
    cg.rowPtr.push_back(0);
    std::vector<double> diag(std::size_t(nf), 0.0), offv;
    for (int r = 0; r < nf; ++r) {
        const int c = cell[std::size_t(r)], i = c % n, j = c / n;
        const int nb[4] = {j * n + (i + n - 1) % n, j * n + (i + 1) % n, ((j + n - 1) % n) * n + i, ((j + 1) % n) * n + i};
        const double ap[4] = {cut.ax[std::size_t(c)], cut.ax[std::size_t(nb[1])], cut.ay[std::size_t(c)], cut.ay[std::size_t(nb[3])]};
        std::vector<std::pair<int, double>> row;
        for (int f = 0; f < 4; ++f) {
            if (!(ap[f] > 0)) continue;
            const double coef = ap[f] * ih2;
            diag[std::size_t(r)] += coef;
            if (id[std::size_t(nb[f])] >= 0) row.emplace_back(id[std::size_t(nb[f])], -coef);
        }
        std::ranges::sort(row);
        for (const auto& [col, v] : row) {
            cg.col.push_back(col);
            offv.push_back(v);
        }
        cg.rowPtr.push_back(int(cg.col.size()));
    }
    BlockMatrix m = BlockMatrix::zeros(cg, *denseShape(1));
    m.diag = diag;
    m.off = offv;
    // rhs: smooth, fluid-mean free
    std::vector<double> rhs(g.size(), 0.0);
    const double pi = std::acos(-1.0);
    double mean = 0;
    for (int r = 0; r < nf; ++r) {
        const int c = cell[std::size_t(r)];
        const double x = (c % n + 0.5) / n, y = (c / n + 0.5) / n;
        rhs[std::size_t(c)] = double(float(std::sin(2 * pi * x) * std::cos(4 * pi * y) + 0.3 * std::cos(6 * pi * x)));
        mean += rhs[std::size_t(c)];
    }
    mean /= nf;
    std::vector<double> b(static_cast<std::size_t>(nf));
    double bmean = 0;
    for (int r = 0; r < nf; ++r) {
        rhs[std::size_t(cell[std::size_t(r)])] = double(float(rhs[std::size_t(cell[std::size_t(r)])] - mean));
        b[std::size_t(r)] = -rhs[std::size_t(cell[std::size_t(r)])];
        bmean += b[std::size_t(r)];
    }
    for (double& v : b) v -= bmean / nf; // consistent to FP64 rounding
    std::printf("  grid %d^2 periodic, cylinder r 0.1 at the centre: %d fluid cells, nnz %zu\n", n, nf, cg.col.size());
    // FFT-PCG
    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = orDie(gpu::DeviceCutGeometry::make(cut), "cut");
    auto solver = orDie(gpu::DeviceCutPoissonSolver::make(gg, dCut, 500, 1e-6), "fft pcg");
    std::vector<float> rf(rhs.begin(), rhs.end());
    auto drhs = orDie(gpu::DeviceScalar::make(g.size()), "rhs"), dp = orDie(gpu::DeviceScalar::make(g.size()), "p");
    if (drhs.upload(rf.data()) != hipSuccess) return 1;
    auto t0 = std::chrono::steady_clock::now();
    const gpu::PcgResult fr = solver.solve(drhs, dp, false);
    const double fftMs = ms(t0);
    std::vector<float> pf(g.size());
    if (dp.download(pf.data()) != hipSuccess) return 1;
    std::vector<double> pfft(static_cast<std::size_t>(nf));
    for (int r = 0; r < nf; ++r) pfft[std::size_t(r)] = pf[std::size_t(cell[std::size_t(r)])];
    std::printf("  FFT-PCG (FP32, tol 1e-6 of |b|): %d iterations, converged %d, %.2f ms\n", fr.iters, int(fr.converged), fftMs);
    auto meanFree = [](std::vector<double> v) {
        double s = 0;
        for (const double x : v) s += x;
        s /= double(v.size());
        for (double& x : v) x -= s;
        return v;
    };
    Cfg c;
    c.nullSpace = true;
    const Run ref = amgSolve<double>(m, cg, b, c, 1e-12);
    printRun("cut 256^2 (reference)", c, ref, "double");
    const Run r6 = amgSolve<double>(m, cg, b, c, 1e-6);
    printRun("cut 256^2 (tol 1e-6)", c, r6, "double");
    const Run f6 = amgSolve<float>(m, cg, b, c, 1e-6);
    printRun("cut 256^2 (tol 1e-6)", c, f6, "float");
    Cfg v = c;
    v.inner = 0;
    const Run v6 = amgSolve<double>(m, cg, b, v, 1e-6);
    printRun("cut 256^2 (tol 1e-6)", v, v6, "double");
    const std::vector<double> pr = meanFree(ref.x), pa = meanFree(pfft);
    double d2 = 0, n2 = 0, dmax = 0, pmax = 0;
    for (int r = 0; r < nf; ++r) {
        const double d = pa[std::size_t(r)] - pr[std::size_t(r)];
        d2 += d * d;
        n2 += pr[std::size_t(r)] * pr[std::size_t(r)];
        dmax = std::max(dmax, std::abs(d));
        pmax = std::max(pmax, std::abs(pr[std::size_t(r)]));
    }
    // the FFT result's own residual in the FP64 operator
    std::vector<double> y;
    latest::spmv(m, pfft, y);
    double rr = 0, bb = 0;
    for (int r = 0; r < nf; ++r) {
        rr += (b[std::size_t(r)] - y[std::size_t(r)]) * (b[std::size_t(r)] - y[std::size_t(r)]);
        bb += b[std::size_t(r)] * b[std::size_t(r)];
    }
    std::printf("  p(FFT-PCG) vs p(AMG-PCG 1e-12): rel L2 %.3e, max %.3e of max|p| %.3e; FFT result's FP64 residual %.3e\n",
                std::sqrt(d2 / n2), dmax / pmax, pmax, std::sqrt(rr / bb));
    check(fr.converged && ref.res.converged && r6.res.converged && f6.res.converged, "LA4: every solve converged");
    check(std::sqrt(d2 / n2) <= 1e-4, "LA4: same p to the FP32 FFT-PCG's tolerance (rel L2 <= 1e-4: " +
                                          sci(std::sqrt(d2 / n2)) + ")");
    std::printf("  LA4 iterations at relTol 1e-6: FFT-PCG %d, AMG-PCG double %d, float %d, V-cycle double %d\n", fr.iters,
                r6.res.iterations, f6.res.iterations, v6.res.iterations);
    return dnstest::finish();
}

// ---- meshes ------------------------------------------------------------------------

int meshesMode(int argc, char** argv)
{
    std::printf("AMG-PCG on OpenFOAM's Laplacian (Gauss linear corrected, gamma 1) of polyhedral meshes\n");
    std::vector<fs::path> dirs{fs::path(DNS_MESH_DATA) / "cyl2d_snappy/constant/polyMesh",
                               fs::path(DNS_MESH_DATA) / "box_refined/constant/polyMesh"};
    std::string outFile;
    int l1hist = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) outFile = argv[++i];
        else if (a == "--l1hist" && i + 1 < argc) l1hist = std::atoi(argv[++i]);
        else dirs.emplace_back(a);
    }
    for (std::size_t di = 0; di < dirs.size(); ++di) {
        const fs::path& dir = dirs[di];
        auto t0 = std::chrono::steady_clock::now();
        const FaceMesh fm = loadMesh(dir, CellOrder::Morton);
        const double readS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const CellGraph g = graphOf(fm);
        std::vector<std::string> dir0;
        for (const auto& p : fm.patches)
            if (p.name == "outlet") dir0.push_back(p.name);
        const BlockMatrix m = latest::foamLaplacian(fm, g, dir0);
        const bool singular = dir0.empty();
        // b: unit source V (Dirichlet outlet), or V (x - mean_V x) (pure Neumann)
        std::vector<double> b(std::size_t(fm.nC));
        double vx = 0, vs = 0;
        for (int c = 0; c < fm.nC; ++c) {
            vx += fm.V[std::size_t(c)] * fm.C[0][std::size_t(c)];
            vs += fm.V[std::size_t(c)];
        }
        for (int c = 0; c < fm.nC; ++c)
            b[std::size_t(c)] = singular ? fm.V[std::size_t(c)] * (fm.C[0][std::size_t(c)] - vx / vs) : fm.V[std::size_t(c)];
        std::printf("  %s: %d cells, nnz %zu, %s, read %.1f s\n", dir.string().c_str(), fm.nC, g.col.size(),
                    singular ? "pure Neumann (constant null space)" : "outlet fixedValue 0", readS);
        const std::string tag = dir.parent_path().parent_path().filename().string();
        std::vector<Cfg> cfgs(4);
        cfgs[1].inner = 0;
        cfgs[2].inner = 1;
        cfgs[3] = smoothers(AmgSmoother::Jacobi, AmgSmoother::Jacobi);
        for (auto& c : cfgs) c.nullSpace = singular;
        for (std::size_t k = 0; k < cfgs.size(); ++k) {
            const Run r = amgSolve<double>(m, g, b, cfgs[k], 1e-8);
            printRun(tag + " relTol 1e-8", cfgs[k], r, "double");
            if (k == 0) {
                const Run r2 = amgSolve<double>(m, g, b, cfgs[k], 1e-8);
                check(r.res.converged && r.res.iterations <= 100, tag + ": AMG-PCG converged to 1e-8 in <= 100 iterations (" +
                                                                       std::to_string(r.res.iterations) + ")");
                check(r2.bytes == r.bytes, tag + ": two solves byte-identical");
                const Run r1 = amgSolve<double>(m, g, b, cfgs[k], 1e-2);
                printRun(tag + " relTol 1e-2", cfgs[k], r1, "double");
                const Run f = amgSolve<float>(m, g, b, cfgs[k], 1e-6);
                printRun(tag + " relTol 1e-6", cfgs[k], f, "float");
                if (!outFile.empty() && di + 1 == dirs.size()) {
                    const Run t = amgSolve<double>(m, g, b, cfgs[k], 1e-13, 2000);
                    std::vector<double> src(std::size_t(fm.nC));
                    for (int c = 0; c < fm.nC; ++c) src[std::size_t(fm.srcCell[std::size_t(c)])] = t.x[std::size_t(c)];
                    std::ofstream os(outFile, std::ios::binary);
                    os.write(reinterpret_cast<const char*>(src.data()), std::streamsize(src.size() * sizeof(double)));
                    std::printf("  wrote x (relTol 1e-13, %d it, true %.2e) in source order to %s\n", t.res.iterations,
                                t.trueRel, outFile.c_str());
                }
                for (int it = 1; it <= l1hist; ++it) {
                    const Run h = amgSolve<double>(m, g, b, cfgs[k], 1e-30, it);
                    std::printf("    L1 history %s: it %d L1 %.4e L2 %.4e\n", cfgName(cfgs[k]).c_str(), it, h.l1, h.trueRel);
                }
            }
        }
    }
    return dnstest::finish();
}

// ---- tune (report only: cycle cost vs iterations) ------------------------------------

int tuneMode(int argc, char** argv)
{
    std::printf("AMG tuning: iterations and solve time per configuration (double, relTol 1e-8)\n");
    struct Prob
    {
        std::string tag;
        CellGraph g;
        BlockMatrix m;
        std::vector<double> b;
        bool singular = false;
    };
    std::vector<Prob> ps;
    {
        const std::array<int, 3> nn{1024, 1024, 1};
        CellGraph g = structuredGraph(nn, {false, false, false});
        BlockMatrix m = latest::tensorLaplacian(g, nn, uniformH(nn, {1.0 / 1024, 1.0 / 1024, 1.0}));
        ps.push_back({"Poisson 1024^2", std::move(g), std::move(m), latest::randomVec(1024 * 1024, 7u)});
    }
    {
        const std::array<int, 3> nn{96, 96, 96};
        CellGraph g = structuredGraph(nn, {false, false, false});
        BlockMatrix m = latest::tensorLaplacian(g, nn, uniformH(nn, {1.0 / 96, 1.0 / 96, 1.0 / 96}));
        ps.push_back({"Poisson 96^3", std::move(g), std::move(m), latest::randomVec(96 * 96 * 96, 7u)});
    }
    for (int i = 2; i < argc; ++i) {
        const FaceMesh fm = loadMesh(argv[i], CellOrder::Morton);
        CellGraph g = graphOf(fm);
        std::vector<std::string> d0;
        for (const auto& p : fm.patches)
            if (p.name == "outlet") d0.push_back(p.name);
        BlockMatrix m = latest::foamLaplacian(fm, g, d0);
        ps.push_back({argv[i], std::move(g), std::move(m), fm.V, d0.empty()});
    }
    using S = AmgSmoother;
    for (const auto& p : ps) {
        for (const auto& [fs, cs] : {std::pair{S::MultiColourGS, S::Jacobi}})
            for (const int coarse : {64, 128, 256})
                for (const int kmin : {-1, 4000, 10000, 20000}) {
                    Cfg c = smoothers(fs, cs);
                    c.coarse = coarse;
                    c.nullSpace = p.singular;
                    c.inner = kmin == -1 ? 0 : kmin == -2 ? 1 : 2;
                    c.innerMinRows = std::max(kmin, 0);
                    const Run r = amgSolve<double>(p.m, p.g, p.b, c, 1e-8);
                    std::printf("  %-28s %-40s it %4d  solve %8.2f ms  %6.2f ms/it  refresh %6.2f ms%s\n", p.tag.c_str(),
                                cfgName(c).c_str(), r.res.iterations, r.solveMs, r.solveMs / std::max(1, r.res.iterations),
                                r.refreshMs, r.res.converged ? "" : " NOT CONVERGED");
                }
    }
    return 0;
}

// FP32 round (docs/fv.md "Precision"): the FV path's p solve -- AMG-PCG,
// scaled V, 2 damped Jacobi sweeps per level -- in float with OpenFOAM's
// normalised residual (normFactor = sum |b| at x0 = 0) on a 512^2 Dirichlet
// Poisson problem. Gates: at absTol = the FP32 p floor the float solve
// converges (no maxIter), agrees with the FP64 solve to 1e-5 of the solution
// scale, two float solves are byte-identical; below the floor (absTol 1e-10)
// it still terminates -- the recursive estimate keeps falling -- while the
// TRUE residual stalls at eps_f32 sum|A||x| / sum|b| (gated within [0.1, 10]x):
// on this problem ~1e-5, ABOVE the FV floor 1e-6, so an FP32 estimate below
// that bound is not a certificate (the FV path re-measures r0 each iteration).
int fp32Mode()
{
    std::printf("FP32: the FV p solve in float, OpenFOAM's normalised residual\n");
    const std::array<int, 3> nn{512, 512, 1};
    const CellGraph g = structuredGraph(nn, {false, false, false});
    const BlockMatrix m = latest::tensorLaplacian(g, nn, uniformH(nn, {1.0 / 512, 1.0 / 512, 1.0}));
    const std::vector<double> b = latest::randomVec(std::size_t(g.n), 11u);
    Cfg c = smoothers(AmgSmoother::Jacobi, AmgSmoother::Jacobi);
    c.inner = 1;
    c.sweeps = 2;
    auto nf = [&]<class T>(T) {
        double s = 0;
        for (const double v : latest::roundVec<T>(b)) s += std::abs(v);
        return s + 1e-20;
    };
    const double fl = dns::kFp32Floors.p;
    const Run d = amgSolve<double>(m, g, b, c, 0.0, 1000, nf(0.0), fl);
    printRun("Poisson 512^2 absTol floor", c, d, "double");
    const Run f = amgSolve<float>(m, g, b, c, 0.0, 1000, nf(0.0f), fl);
    const Run f2 = amgSolve<float>(m, g, b, c, 0.0, 1000, nf(0.0f), fl);
    printRun("Poisson 512^2 absTol floor", c, f, "float");
    std::printf("  floor %.1e: float it %d est %.3e true L1 %.3e; double it %d true L1 %.3e\n", fl, f.res.iterations,
                f.res.residual, f.l1, d.res.iterations, d.l1);
    check(f.res.converged && f.res.iterations < 1000, "float AMG-PCG at absTol = the FP32 p floor converges (" +
                                                          std::to_string(f.res.iterations) + " it)");
    // The arithmetic's floor eps_f32 sum|A||x| / sum|b| (the float matrix, the float iterate).
    const BlockMatrix mf = latest::roundTo<float>(m);
    double ax = 0;
    for (int r = 0; r < g.n; ++r) {
        ax += std::abs(mf.diag[std::size_t(r)]) * std::abs(f.x[std::size_t(r)]);
        for (int k = g.rowPtr[std::size_t(r)]; k < g.rowPtr[std::size_t(r) + 1]; ++k)
            ax += std::abs(mf.off[std::size_t(k)]) * std::abs(f.x[std::size_t(g.col[std::size_t(k)])]);
    }
    const double bound = std::ldexp(ax / nf(0.0f), -24);
    std::printf("  eps_f32 sum|A||x| / sum|b| = %.3e\n", bound);
    double dx = 0, sc = 0;
    for (std::size_t i = 0; i < f.x.size(); ++i) {
        dx = std::max(dx, std::abs(f.x[i] - d.x[i]));
        sc = std::max(sc, std::abs(d.x[i]));
    }
    check(dx <= 1e-5 * sc, "float vs double solution at the floor: " + sci(dx / sc) + " of scale (gate 1e-5)");
    check(f2.bytes == f.bytes && f2.res.iterations == f.res.iterations, "float: two solves byte-identical");
    for (const double t : {1e-7, 1e-8, 1e-10}) {
        const Run u = amgSolve<float>(m, g, b, c, 0.0, 1000, nf(0.0f), t);
        std::printf("  float absTol %.0e: it %d converged %d est %.3e true L1 %.3e\n", t, u.res.iterations,
                    int(u.res.converged), u.res.residual, u.l1);
        if (t == 1e-10) {
            check(u.res.iterations < 1000, "float below the floor (absTol 1e-10) terminates before maxIter (" +
                                                std::to_string(u.res.iterations) + " it, true L1 " + sci(u.l1) + ")");
            check(u.l1 >= 0.1 * bound && u.l1 <= 10 * bound,
                  "float: the TRUE residual stalls at the arithmetic's floor eps_f32 sum|A||x|/sum|b| (ratio " +
                      sci(u.l1 / bound) + ", gate [0.1, 10]) while the estimate falls to " + sci(u.res.residual));
        }
    }
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "refresh";
    if (mode == "fp32") return fp32Mode();
    if (mode == "refresh") return refreshMode();
    if (mode == "la3") return la3Mode();
    if (mode == "la4") return la4Mode();
    if (mode == "meshes") return meshesMode(argc, argv);
    if (mode == "tune") return tuneMode(argc, argv);
    std::printf("usage: test_gpu_amg refresh|la3|la4|meshes [POLYMESH_DIR...] [--out FILE] [--l1hist N]\n");
    return 1;
}
