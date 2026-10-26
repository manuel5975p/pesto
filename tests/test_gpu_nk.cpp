// The flow Newton-Krylov pieces (gpu/nk, dns/nk_precond; masterplan_flowrans
// P3, docs/rans.md "Newton-Krylov pseudo-time"):
//   (dual) the dual-number blocks of the preconditioner flux equal a host
//          4th-order central difference of the same flux to 1e-10
//   (N1)   J v: forward eps vs central 2 eps agree to 1e-2 per field
//          (scaled) at kNkFdStep, on a mid-transient and a near-converged
//          state; the sweep 2^-10/-12/-14 is printed
//   (N2)   the GPU-assembled P blocks (FP32 storage) equal a host FP64
//          restatement built from central differences of F_P to 2e-6 of the
//          block scale, every real cell of a 16 x 12 O-grid (laminar and SA)
//   (N3)   the GPU line-SGS application equals a host dense block
//          Gauss-Seidel on the downloaded P, same ordering, to 1e-5
//   (2D)   rho w stays exactly 0 through Newton-Krylov iterations
// Seconds.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/nk_precond.hpp"
#include "dns/ogrid.hpp"
#include "dns/pseudo_time.hpp"
#include "dns/rans.hpp"
#include "dns/walldist.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/nk.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "dns/gpu/reductions.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

constexpr double kGam = 1.4, kRgas = 1.0, kMach = 0.2;
constexpr double kRhoInf = 1.0, kUInf = 1.0;
const double kPInf = kRhoInf * kUInf * kUInf / (kGam * kMach * kMach);
const double kTInf = kPInf / (kRhoInf * kRgas);
constexpr double kD = 0.785398163397448, kRFar = 20 * kD;
constexpr int kF = 5;

struct Rig
{
    Grid g{1, 1, 1, 1, 1, 1};
    std::unique_ptr<OGrid> og;
    std::unique_ptr<CurvMetrics> m;
    std::vector<double> d;
    std::vector<std::uint8_t> mask;
    Transport tr{};
    std::unique_ptr<gpu::DeviceComp> dev;
    std::unique_ptr<gpu::DeviceCompState> ds;
    std::unique_ptr<gpu::DeviceOGridBc> bc;
    std::unique_ptr<gpu::ReduceScratch> red;
    std::unique_ptr<gpu::DeviceNk> nk;
    RansModel model = RansModel::None;
    double phi = 0;
};

std::unique_ptr<Rig> makeRig(RansModel model, int nx, int ny, int steps)
{
    auto r = std::make_unique<Rig>();
    r->model = model;
    r->g = ogridGrid(nx, ny, 1, 0.5 * kD, kRFar, 1.0);
    const Sdf circle = [](double x, double y, double) {
        return std::hypot(x, y) - 0.5 * kD;
    };
    auto og = buildOGrid(r->g, circle, {.rFar = kRFar, .beta = 3.0, .span = 1.0});
    if (!og) return nullptr;
    r->og = std::make_unique<OGrid>(std::move(*og));
    auto m = CurvMetrics::build(r->og->mesh);
    if (!m) return nullptr;
    r->m = std::make_unique<CurvMetrics>(std::move(*m));
    r->d = ogridWallDistance(r->og->mesh);
    const double mu = kRhoInf * kUInf * kD / 40.0;
    if (model == RansModel::SaNeg)
        r->phi = nuTildaInf(kRansNutInfRatio, mu / kRhoInf);
    r->tr = Transport::air(mu, kTInf, 300.0, 0.72, kGam, kRgas, false);
    auto dev = gpu::DeviceComp::make(*r->m, r->tr);
    auto ds = gpu::DeviceCompState::make(r->g.size(), ransScalars(model));
    gpu::DeviceOGridBc::Config bcc{};
    bcc.gamma = kGam;
    bcc.Rgas = kRgas;
    bcc.rhoInf = kRhoInf;
    bcc.uInf = kUInf;
    bcc.pInf = kPInf;
    bcc.tr = r->tr;
    bcc.wallD1 = wallFirstCellDistance(r->g, r->d);
    bcc.rans = model;
    bcc.phiInf[0] = r->phi;
    auto bc = gpu::DeviceOGridBc::make(r->g, bcc);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !bc || !red) return nullptr;
    r->dev = std::make_unique<gpu::DeviceComp>(std::move(*dev));
    r->ds = std::make_unique<gpu::DeviceCompState>(std::move(*ds));
    r->bc = std::make_unique<gpu::DeviceOGridBc>(std::move(*bc));
    r->red = std::make_unique<gpu::ReduceScratch>(std::move(*red));
    r->mask = gpu::rowMask(r->g, 1, r->g.ny - 2);
    if (model != RansModel::None) {
        if (!r->dev->enableRans({.model = model,
                                 .wallDist = r->d,
                                 .mutHook = r->bc->mutHook(),
                                 .fluidMask = r->mask}))
            return nullptr;
    } else if (!r->dev->enableSteady(r->mask)) {
        return nullptr;
    }
    Rig* p = r.get();
    p->dev->setStageHook([p](gpu::DeviceCompState& s) { p->bc->apply(s); });
    // Impulsive freestream, then `steps` explicit steps: a transient state.
    auto h = CompState::zeros(r->g);
    for (std::size_t c = 0; c < r->g.size(); ++c) {
        h.rho[c] = kRhoInf;
        h.mx[c] = kRhoInf * kUInf;
        h.E[c] = kPInf / (kGam - 1.0) + 0.5 * kRhoInf * kUInf * kUInf;
    }
    (void)r->ds->upload(h);
    if (r->ds->nT)
        (void)r->ds->uploadTurb(0, std::vector<double>(r->g.size(),
                                                       kRhoInf * r->phi));
    r->bc->apply(*r->ds);
    for (int s = 0; s < steps; ++s) {
        const double dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
        r->dev->step(*r->ds, gpu::Real(dt));
        r->bc->apply(*r->ds);
    }
    gpu::DeviceNk::Config nc;
    nc.lay = {.iLo = 0, .iHi = r->g.nx - 1, .kLo = 0, .kHi = 0,
              .cyclicI = true, .skipK = true, .wedge = false, .dphi = 0};
    const double cInf = std::sqrt(kGam * kPInf / kRhoInf);
    nc.sc.rho = kRhoInf;
    nc.sc.mom = kRhoInf * (kUInf + cInf);
    nc.sc.E = kRhoInf * cInf * cInf;
    nc.sc.t[0] = kRhoInf * std::max(r->phi, mu / kRhoInf);
    nc.model = model;
    nc.gamma = kGam;
    nc.Rgas = kRgas;
    auto nk = gpu::DeviceNk::make(*r->dev, r->mask, nc,
                                  [p](gpu::DeviceCompState& s) { p->bc->apply(s); });
    if (!nk) return nullptr;
    r->nk = std::make_unique<gpu::DeviceNk>(std::move(*nk));
    if (r->nk->load(*r->ds) != hipSuccess) return nullptr;
    r->nk->store(*r->ds);
    (void)r->nk->residual(*r->ds, *r->red);
    return r;
}

std::vector<double> down(const gpu::DeviceBuffer<gpu::Real>& b)
{
    std::vector<gpu::Real> h(b.size());
    (void)b.download(h.data());
    return {h.begin(), h.end()};
}
std::vector<double> downD(const gpu::DeviceBuffer<double>& b)
{
    std::vector<double> h(b.size());
    (void)b.download(h.data());
    return h;
}

// (dual): duals vs a 4th-order central difference, random face data.
void dualBlocks()
{
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    double worst = 0;
    for (int trial = 0; trial < 20; ++trial) {
        nk::FaceFrozen f{};
        for (int x = 0; x < 3; ++x) {
            f.S[x] = U(rng);
            f.Ji[x] = 3 * U(rng);
        }
        f.lambda = 2 + U(rng);
        f.mu = trial % 4 == 0 ? 0.0 : 0.05 * (1.5 + U(rng));
        f.kappa = 0.07 * (1.5 + U(rng));
        f.gamma = kGam;
        f.Rgas = kRgas;
        double q[2][5];
        for (auto& s : q) {
            const double rho = 1 + 0.2 * U(rng), u = U(rng), v = U(rng),
                         w = U(rng), p = 10 + U(rng);
            s[0] = rho;
            s[1] = rho * u;
            s[2] = rho * v;
            s[3] = rho * w;
            s[4] = p / (kGam - 1) + 0.5 * rho * (u * u + v * v + w * w);
        }
        double AL[5][5], AR[5][5];
        nk::precondBlocks(q[0], q[1], f, AL, AR);
        auto flux = [&](const double (&a)[2][5], double F[5]) {
            nk::precondFlux<double>(a[0], a[1], f, F);
        };
        double scale = 0;
        for (int r = 0; r < 5; ++r)
            for (int c = 0; c < 5; ++c)
                scale = std::max({scale, std::abs(AL[r][c]), std::abs(AR[r][c])});
        for (int side = 0; side < 2; ++side)
            for (int c = 0; c < 5; ++c) {
                const double h = 1e-3 * (std::abs(q[side][c]) + 1.0);
                auto D = [&](double hh, int r) {
                    double a[2][5], b[2][5], Fa[5], Fb[5];
                    std::copy(&q[0][0], &q[0][0] + 10, &a[0][0]);
                    std::copy(&q[0][0], &q[0][0] + 10, &b[0][0]);
                    a[side][c] += hh;
                    b[side][c] -= hh;
                    flux(a, Fa);
                    flux(b, Fb);
                    return (Fa[r] - Fb[r]) / (2 * hh);
                };
                for (int r = 0; r < 5; ++r) {
                    const double fd = (4 * D(0.5 * h, r) - D(h, r)) / 3;
                    const double ad = side ? AR[r][c] : AL[r][c];
                    worst = std::max(worst, std::abs(fd - ad) / scale);
                }
            }
    }
    std::printf("  dual blocks vs 4th-order central FD: max rel %.3e\n", worst);
    check(worst <= 1e-10, "(dual) precondBlocks == central FD of F_P to 1e-10");
}

// Per-field scaled norms of a F x n vector over the mask.
std::vector<double> fieldNorms(const Rig& r, const std::vector<double>& v, int F)
{
    const std::size_t n = r.g.size();
    std::vector<double> s(F, 0.0);
    for (int f = 0; f < F; ++f)
        for (std::size_t c = 0; c < n; ++c)
            if (r.mask[c]) s[f] += v[f * n + c] * v[f * n + c];
    for (double& x : s) x = std::sqrt(x);
    return s;
}

// (N1) at the rig's current iterate: forward eps against central 2 eps,
// per field (scaled norms), over the plan's sweep {2^-10, 2^-12, 2^-14}; the
// gate is at the adopted kNkFdStep (2^-10 failed on SA's nu~, pseudo_time.hpp).
void jvConsistency(Rig& r, const std::string& what)
{
    const int F = r.nk->fields();
    const std::size_t n = r.g.size();
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<gpu::Real> hv(F * n, 0);
    for (std::size_t c = 0; c < n; ++c)
        if (r.mask[c])
            for (int f = 0; f < F; ++f) hv[f * n + c] = gpu::Real(U(rng));
    auto v = gpu::DeviceBuffer<gpu::Real>::make(F * n);
    auto a = gpu::DeviceBuffer<gpu::Real>::make(F * n);
    auto b = gpu::DeviceBuffer<gpu::Real>::make(F * n);
    if (!v || !a || !b) { check(false, "(N1) buffers"); return; }
    (void)v->upload(hv.data());
    bool atStep = false;
    for (const double eps : {0x1p-10, 0x1p-12, 0x1p-14}) {
        r.nk->jacVec(*v, false, eps, *a, *r.red);
        r.nk->jacVec(*v, true, 2 * eps, *b, *r.red);
        const auto fa = down(*a), fb = down(*b);
        std::vector<double> diff(fa.size());
        for (std::size_t i = 0; i < fa.size(); ++i) diff[i] = fa[i] - fb[i];
        const auto nd = fieldNorms(r, diff, F), nb = fieldNorms(r, fb, F);
        double worst = 0;
        std::string per;
        for (int f = 0; f < F; ++f) {
            if (f == 3) continue; // rho w: J v is 0 in 2D (both exactly)
            worst = std::max(worst, nd[f] / nb[f]);
            per += std::format(" {:.2e}", nd[f] / nb[f]);
        }
        std::printf("  %s eps 2^%d: |Jv_fwd - Jv_ctr| / |Jv_ctr| per field%s\n",
                    what.c_str(), int(std::lround(std::log2(eps))), per.c_str());
        if (eps == kNkFdStep) atStep = worst <= 1e-2;
    }
    check(atStep, "(N1) " + what + ": forward vs central J v agree to 1e-2 "
                                   "per field at kNkFdStep (2^-14)");
}

// The host restatement of kAssemble, built from CENTRAL DIFFERENCES of F_P
// (nk::precondFlux) on the FP32 inputs the GPU sees.
struct Host
{
    const Rig& r;
    std::vector<double> rho, u, v, w, p, T, mut;
    std::vector<double> ds, dtc;
    double cfl;
};

void fdBlocks(const double qa[5], const double qb[5], const nk::FaceFrozen& f,
              double AL[5][5], double AR[5][5])
{
    for (int side = 0; side < 2; ++side)
        for (int c = 0; c < 5; ++c) {
            const double base = side ? qb[c] : qa[c];
            const double h = 1e-3 * (std::abs(base) + 1.0);
            auto D = [&](double hh, int row) {
                double a[5], b[5], c2[5], d2[5], Fa[5], Fb[5];
                std::copy(qa, qa + 5, a);
                std::copy(qb, qb + 5, b);
                std::copy(qa, qa + 5, c2);
                std::copy(qb, qb + 5, d2);
                (side ? b : a)[c] += hh;
                (side ? d2 : c2)[c] -= hh;
                nk::precondFlux<double>(a, b, f, Fa);
                nk::precondFlux<double>(c2, d2, f, Fb);
                return (Fa[row] - Fb[row]) / (2 * hh);
            };
            for (int row = 0; row < 5; ++row) {
                const double fd = (4 * D(0.5 * h, row) - D(h, row)) / 3;
                (side ? AR : AL)[row][c] = fd;
            }
        }
}

// (N2) every real cell: GPU blocks vs the host restatement.
void assembly(Rig& r, const std::string& what)
{
    const double cfl = 100;
    if (auto e = r.nk->assemble(*r.ds, cfl, *r.red); !e) {
        check(false, "(N2) assemble: " + e.error());
        return;
    }
    const gpu::GpuGrid gg = gpu::GpuGrid::from(r.g);
    const std::size_t n = r.g.size();
    const auto Am = down(r.nk->meanBlocks());
    const auto ds = downD(r.nk->scales());
    const auto dtc = downD(r.dev->localDtCells());
    const auto rho = down(r.ds->rho);
    const auto U = down(r.dev->u()), V = down(r.dev->v()), W = down(r.dev->w()),
               P = down(r.dev->p()), T = down(r.dev->T());
    std::vector<double> mut(n, 0.0);
    if (r.model != RansModel::None) mut = down(r.dev->mut());
    const double mu0 = double(float(r.tr.muRef));
    const double kap0 = double(float(r.tr.muRef * r.tr.cp() / r.tr.Pr));
    const double kapScale = r.tr.cp() / 0.9;
    auto f32 = [](double x) { return double(float(x)); };
    double worst = 0;
    for (int j = 1; j <= r.g.ny - 2; ++j)
        for (int i = 0; i < r.g.nx; ++i) {
            const std::size_t c = r.g.idx(i, j, 0);
            const auto o = gpu::offsetsAt(gg, i, j, 0);
            const double iv = f32(1.0 / r.m->vol[c]);
            double Jd[5][5] = {}, Jo[6][5][5] = {};
            for (int a = 0; a < 2; ++a)
                for (int side = 0; side < 2; ++side) {
                    const std::size_t nb = side ? c + o.p[a] : c + o.m[a];
                    const std::size_t fa = side ? c : nb, fb = side ? nb : c;
                    nk::FaceFrozen fr{};
                    for (int x = 0; x < 3; ++x) {
                        fr.S[x] = f32(r.m->S[a][x][fa]);
                        fr.Ji[x] = f32(r.m->Ji[a][a * 3 + x][fa]);
                    }
                    const double Sn = std::sqrt(fr.S[0] * fr.S[0] +
                                                fr.S[1] * fr.S[1] +
                                                fr.S[2] * fr.S[2]);
                    auto cons = [&](std::size_t x, double q[5]) {
                        q[0] = rho[x];
                        q[1] = rho[x] * U[x];
                        q[2] = rho[x] * V[x];
                        q[3] = rho[x] * W[x];
                        q[4] = P[x] / (kGam - 1) +
                               0.5 * rho[x] *
                                   (U[x] * U[x] + V[x] * V[x] + W[x] * W[x]);
                    };
                    auto lam = [&](std::size_t x) {
                        const double Un = U[x] * fr.S[0] + V[x] * fr.S[1] +
                                          W[x] * fr.S[2];
                        return std::abs(Un) + std::sqrt(kGam * P[x] / rho[x]) * Sn;
                    };
                    fr.lambda = std::max(lam(fa), lam(fb));
                    const double mt = 0.5 * (mut[fa] + mut[fb]);
                    fr.mu = mu0 + mt;
                    fr.kappa = kap0 + mt * kapScale;
                    fr.gamma = kGam;
                    fr.Rgas = kRgas;
                    double qa[5], qb[5], AL[5][5], AR[5][5];
                    cons(fa, qa);
                    cons(fb, qb);
                    fdBlocks(qa, qb, fr, AL, AR);
                    double Bd[5][5], Bo[5][5];
                    for (int x = 0; x < 5; ++x)
                        for (int y = 0; y < 5; ++y) {
                            Bd[x][y] = side ? -AL[x][y] * iv : AR[x][y] * iv;
                            Bo[x][y] = side ? -AR[x][y] * iv : AL[x][y] * iv;
                        }
                    const int slot = a * 2 + side;
                    const bool real = r.mask[nb] != 0;
                    const bool wall = a == 1 && side == 0 && j == 1;
                    if (wall)
                        for (int x = 0; x < 5; ++x)
                            for (int y = 1; y <= 3; ++y) Bo[x][y] = -Bo[x][y];
                    for (int x = 0; x < 5; ++x)
                        for (int y = 0; y < 5; ++y) {
                            Jd[x][y] += Bd[x][y] + (wall ? Bo[x][y] : 0.0);
                            if (real) Jo[slot][x][y] += Bo[x][y];
                        }
                }
            // Compare: P = I/dtau - J, scaled by ds.
            double scale = 0, err = 0;
            for (int b = 0; b < 7; ++b)
                for (int x = 0; x < 5; ++x)
                    for (int y = 0; y < 5; ++y) {
                        double hostP;
                        if (b == 0) {
                            hostP = -Jd[x][y] +
                                    (x == y ? 1.0 / (cfl * dtc[c]) : 0.0);
                            hostP *= ds[y * n + c] / ds[x * n + c];
                        } else {
                            const int slot = b - 1, a = slot / 2,
                                      side = slot % 2;
                            if (a == 2) {
                                hostP = 0;
                            } else {
                                const std::size_t nb =
                                    side ? c + o.p[a] : c + o.m[a];
                                hostP = r.mask[nb] ? -Jo[slot][x][y] *
                                                         ds[y * n + nb] /
                                                         ds[x * n + c]
                                                   : 0.0;
                            }
                        }
                        const double gpuP = Am[std::size_t(b * 25 + x * 5 + y) * n + c];
                        scale = std::max(scale, std::abs(hostP));
                        err = std::max(err, std::abs(hostP - gpuP));
                    }
            worst = std::max(worst, err / scale);
        }
    std::printf("  %s: GPU P blocks vs host FD restatement: max rel %.3e\n",
                what.c_str(), worst);
    check(worst <= 2e-6, "(N2) " + what +
                             ": assembled P == host central-FD restatement "
                             "(FP32 storage, 2e-6 of the block scale)");
}

// Dense solve of a small system (partial pivoting), in place.
void denseSolve(std::vector<double>& A, std::vector<double>& b, int m)
{
    for (int c = 0; c < m; ++c) {
        int pr = c;
        for (int r = c + 1; r < m; ++r)
            if (std::abs(A[r * m + c]) > std::abs(A[pr * m + c])) pr = r;
        if (pr != c) {
            for (int k = 0; k < m; ++k) std::swap(A[c * m + k], A[pr * m + k]);
            std::swap(b[c], b[pr]);
        }
        for (int r = c + 1; r < m; ++r) {
            const double f = A[r * m + c] / A[c * m + c];
            if (f == 0) continue;
            for (int k = c; k < m; ++k) A[r * m + k] -= f * A[c * m + k];
            b[r] -= f * b[c];
        }
    }
    for (int c = m - 1; c >= 0; --c) {
        double s = b[c];
        for (int k = c + 1; k < m; ++k) s -= A[c * m + k] * b[k];
        b[c] = s / A[c * m + c];
    }
}

// (N3) z = P^-1 r by one symmetric lexicographic line GS sweep on the host.
void lineSgs(Rig& r, const std::string& what)
{
    if (auto e = r.nk->assemble(*r.ds, 1000.0, *r.red); !e) {
        check(false, "(N3) assemble: " + e.error());
        return;
    }
    const int F = r.nk->fields(), nT = F - kF;
    const std::size_t n = r.g.size();
    const gpu::GpuGrid gg = gpu::GpuGrid::from(r.g);
    const auto Am = down(r.nk->meanBlocks());
    const auto At = down(r.nk->scalarBlocks());
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<gpu::Real> hr(F * n, 0);
    for (std::size_t c = 0; c < n; ++c)
        if (r.mask[c])
            for (int f = 0; f < F; ++f) hr[f * n + c] = gpu::Real(U(rng));
    auto rb = gpu::DeviceBuffer<gpu::Real>::make(F * n);
    auto zb = gpu::DeviceBuffer<gpu::Real>::make(F * n);
    if (!rb || !zb) { check(false, "(N3) buffers"); return; }
    (void)rb->upload(hr.data());
    r.nk->precond(*rb, *zb);
    const auto zg = down(*zb);

    std::vector<double> z(F * n, 0.0);
    const int nf = r.g.ny - 2, B = F;
    auto line = [&](int i) {
        const int m = nf * B;
        std::vector<double> A(std::size_t(m) * m, 0.0), b(m, 0.0);
        for (int jj = 0; jj < nf; ++jj) {
            const int j = 1 + jj;
            const std::size_t c = r.g.idx(i, j, 0);
            const auto o = gpu::offsetsAt(gg, i, j, 0);
            for (int f = 0; f < F; ++f) b[jj * B + f] = hr[f * n + c];
            for (int slot : {0, 1}) {
                const std::size_t nb = slot ? c + o.p[0] : c + o.m[0];
                for (int x = 0; x < 5; ++x)
                    for (int y = 0; y < 5; ++y)
                        b[jj * B + x] -=
                            Am[std::size_t((1 + slot) * 25 + x * 5 + y) * n + c] *
                            z[y * n + nb];
                for (int e = 0; e < nT; ++e)
                    b[jj * B + 5 + e] -= At[std::size_t(e * 7 + 1 + slot) * n + c] *
                                         z[(5 + e) * n + nb];
            }
            for (int blk : {0, 3, 4}) {
                const int jn = blk == 0 ? jj : blk == 3 ? jj - 1 : jj + 1;
                if (jn < 0 || jn >= nf) continue;
                for (int x = 0; x < 5; ++x)
                    for (int y = 0; y < 5; ++y)
                        A[std::size_t(jj * B + x) * m + jn * B + y] =
                            Am[std::size_t(blk * 25 + x * 5 + y) * n + c];
                for (int e = 0; e < nT; ++e)
                    A[std::size_t(jj * B + 5 + e) * m + jn * B + 5 + e] =
                        At[std::size_t(e * 7 + blk) * n + c];
            }
        }
        denseSolve(A, b, m);
        for (int jj = 0; jj < nf; ++jj)
            for (int f = 0; f < F; ++f)
                z[f * n + r.g.idx(i, 1 + jj, 0)] = b[jj * B + f];
    };
    for (int i = 0; i < r.g.nx; ++i) line(i);
    for (int i = r.g.nx - 2; i >= 0; --i) line(i);
    double err = 0, scale = 0;
    for (std::size_t c = 0; c < n; ++c)
        if (r.mask[c])
            for (int f = 0; f < F; ++f) {
                err = std::max(err, std::abs(z[f * n + c] - zg[f * n + c]));
                scale = std::max(scale, std::abs(z[f * n + c]));
            }
    std::printf("  %s: GPU line-SGS vs host dense block GS: max rel %.3e\n",
                what.c_str(), err / scale);
    check(err <= 1e-5 * scale, "(N3) " + what +
                                   ": line-SGS == host dense block GS (same "
                                   "order) to 1e-5");
}

// (2D) rho w == 0 exactly through Newton-Krylov iterations; the iterate
// stays finite and the residual falls.
void newton2d(Rig& r)
{
    PseudoTime pt(kNkPolicy);
    const double r0 = r.nk->residual(*r.ds, *r.red);
    double rn = r0;
    bool zero = true, finite = true;
    for (int it = 0; it < 6; ++it) {
        auto lin = r.nk->solve(*r.ds, pt.cfl(), *r.red);
        if (!lin) { check(false, "(2D) solve: " + lin.error()); return; }
        const double a = r.nk->stepLength(*r.red);
        r.nk->trial(a, *r.ds);
        const double rt = r.nk->residual(*r.ds, *r.red);
        finite = finite && std::isfinite(rt);
        r.nk->accept();
        pt.accepted(a);
        rn = rt;
        const auto mz = down(r.ds->mz);
        zero = zero && std::ranges::all_of(mz, [](double x) { return x == 0.0; });
    }
    std::printf("  Newton-Krylov 6 iterations: scaled residual %.3e -> %.3e\n",
                r0, rn);
    check(zero, "(2D) rho w == 0 exactly through 6 Newton-Krylov iterations");
    check(finite && rn < 0.5 * r0, "(2D) the scaled residual falls");
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    dualBlocks();
    for (RansModel m : {RansModel::None, RansModel::SaNeg}) {
        const std::string n = m == RansModel::None ? "laminar" : "SA";
        auto r = makeRig(m, 16, 12, 100);
        if (!r) { check(false, n + " rig"); continue; }
        assembly(*r, n);
        lineSgs(*r, n);
        jvConsistency(*r, n + " mid-transient");
        newton2d(*r);
        jvConsistency(*r, n + " after 6 Newton-Krylov iterations");
    }
    return dnstest::finish();
}
