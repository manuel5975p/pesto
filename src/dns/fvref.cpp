#include "dns/fvref.hpp"

#include "dns/parallel.hpp"

#include <algorithm>
#include <barrier>
#include <cassert>
#include <cmath>
#include <format>
#include <optional>

namespace dns {
namespace {

using V3 = std::array<double, 3>;

V3 at(const std::array<std::vector<double>, 3>& a, std::size_t i) { return {a[0][i], a[1][i], a[2][i]}; }

// Largest diffusivity on mu/rho the FV checkerboard sees: energy gamma/Pr, momentum
// 4/3 (the normal stress of a longitudinal checkerboard; 1 grew x1.8e6 in 60 steps at
// Pr 2 on 100:1 cells, E6). FV-only: the structured solvers keep diffusivityFactor.
double fvDiffusivityFactor(const Transport& tr) { return std::max(4.0 / 3.0, tr.gamma / tr.Pr); }

std::pair<std::size_t, std::size_t> part(std::size_t n, int w, int nw)
{
    return {n * std::size_t(w) / std::size_t(nw), n * std::size_t(w + 1) / std::size_t(nw)};
}

// body(w, nw, sync) on nw persistent workers; sync() is a barrier across all
// of them (a no-op on one worker).
template <class Body>
void onWorkers(int nw, Body&& body)
{
    if (nw <= 1) {
        body(0, 1, [] {});
        return;
    }
    std::barrier sync(nw);
    forBlocks(nw, std::size_t(nw), [&](int w, std::size_t, std::size_t) {
        body(w, nw, [&] { sync.arrive_and_wait(); });
    });
}

bool finite3(const double v[3]) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); }

std::expected<void, std::string> checkBc(const FvBc& bc, const MeshPatch& p, int emptyAxis, const Transport& tr)
{
    const bool empty = p.type == FoamPatchType::Empty;
    if (empty != (bc.kind == FvBcKind::Empty))
        return std::unexpected(std::format("patch `{}`: {}", p.name,
                                           empty ? "an `empty` patch takes the Empty kind only"
                                                 : "the Empty kind is for `empty` patches only"));
    const int e = emptyAxis;
    switch (bc.kind) {
    case FvBcKind::Wall:
        if (bc.isothermal && !(bc.tWall > 0.0 && std::isfinite(bc.tWall)))
            return std::unexpected(std::format("patch `{}`: isothermal wall temperature must be positive, got {}",
                                               p.name, bc.tWall));
        if (!finite3(bc.uWall) || !finite3(bc.omega) || !finite3(bc.origin))
            return std::unexpected(std::format("patch `{}`: wall motion must be finite", p.name));
        if (e >= 0) {
            bool inPlane = bc.uWall[e] == 0.0;
            for (int a = 0; a < 3; ++a)
                if (a != e && bc.omega[a] != 0.0) inPlane = false;
            if (!inPlane)
                return std::unexpected(std::format(
                    "patch `{}`: a 2D mesh needs wall motion in its plane (no velocity along the empty "
                    "axis, rotation about it only)",
                    p.name));
        }
        return {};
    case FvBcKind::Inflow:
        if (!(bc.tIn > 0.0 && std::isfinite(bc.tIn)) || !finite3(bc.uIn))
            return std::unexpected(std::format("patch `{}`: inflow needs a finite velocity and T > 0", p.name));
        if (e >= 0 && bc.uIn[e] != 0.0)
            return std::unexpected(std::format("patch `{}`: inflow velocity along the empty axis on a 2D mesh",
                                               p.name));
        return {};
    case FvBcKind::Outflow:
        if (!(bc.pOut > 0.0 && std::isfinite(bc.pOut)) || !(bc.tBack > 0.0 && std::isfinite(bc.tBack)))
            return std::unexpected(std::format("patch `{}`: outflow needs p > 0 and a backflow T > 0", p.name));
        return {};
    case FvBcKind::Far: {
        const FvPrim<double> F =
            fvPrimOf(bc.qFar[0], bc.qFar[1], bc.qFar[2], bc.qFar[3], bc.qFar[4], tr.gamma, tr.Rgas);
        if (!(F.rho > 0.0 && F.p > 0.0 && std::isfinite(F.rho) && std::isfinite(F.p) && finite3(F.u)))
            return std::unexpected(std::format("patch `{}`: far-field state needs rho > 0 and p > 0", p.name));
        if (e >= 0 && bc.qFar[1 + e] != 0.0)
            return std::unexpected(std::format("patch `{}`: far-field velocity along the empty axis on a 2D mesh",
                                               p.name));
        return {};
    }
    case FvBcKind::Slip:
    case FvBcKind::Empty: return {};
    }
    return {};
}

} // namespace

struct FvRef::Stage
{
    const double* a = nullptr; // null: rhs mode, out = R
    const double* b = nullptr;
    double* out = nullptr;
    double ca = 0, cb = 0, cr = 1;
    const double* src = nullptr;
};

FvState FvState::zeros(int nC, int nV)
{
    FvState s;
    s.nV = nV;
    s.nC = nC;
    s.q.assign(std::size_t(nV) * std::size_t(nC), 0.0);
    return s;
}

std::expected<FvRef, std::string> FvRef::make(const FaceMesh& fm, FvConfig cfg, int threads)
{
    const Transport& tr = cfg.tr;
    if (!(tr.gamma > 1.0) || !(tr.muRef >= 0.0) || !(tr.Rgas > 0.0) || !(tr.Pr > 0.0))
        return std::unexpected("FV gas: gamma > 1, mu >= 0, R > 0 and Pr > 0 required");
    if (cfg.bc.size() != fm.patches.size())
        return std::unexpected(std::format("FV boundary conditions: {} given for {} patches", cfg.bc.size(),
                                           fm.patches.size()));
    for (std::size_t p = 0; p < fm.patches.size(); ++p)
        if (auto ok = checkBc(cfg.bc[p], fm.patches[p], fm.emptyAxis, tr); !ok)
            return std::unexpected(std::move(ok.error()));

    FvRef r;
    r.fm_ = &fm;
    auto geom = fvFaceGeom(fm, cfg.rule, cfg.alpha, threads);
    if (!geom) return std::unexpected(std::move(geom.error()));
    r.geom_ = std::move(*geom);
    auto gw = fvGradWeights(fm, r.geom_, cfg.grad, threads);
    if (!gw) return std::unexpected(std::move(gw.error()));
    r.gw_ = std::move(*gw);
    r.cfg_ = std::move(cfg);

    r.bcOf_.assign(std::size_t(fm.nB), -1);
    for (std::size_t p = 0; p < fm.patches.size(); ++p)
        std::fill_n(r.bcOf_.begin() + fm.patches[p].first, fm.patches[p].count, int(p));
    r.diffCoef_.assign(std::size_t(fm.nC), 0.0);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        double s = 0.0;
        for (int q = fm.cellStart[c]; q < fm.cellStart[c + 1]; ++q) {
            const std::size_t f = std::size_t(fm.cellFace[std::size_t(q)]);
            if (!r.geom_.active[f]) continue;
            s += r.geom_.corr[0][f] * fm.S[0][f] + r.geom_.corr[1][f] * fm.S[1][f] +
                 r.geom_.corr[2][f] * fm.S[2][f];
        }
        r.diffCoef_[c] = s;
    }
    r.workers_ = workerCount(threads, std::max<std::size_t>(1, std::size_t(fm.nC) / 256));
    return r;
}

void FvRef::prims(const double* q, FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const std::size_t nC = std::size_t(fm_->nC);
    const double g = cfg_.tr.gamma, R = cfg_.tr.Rgas;
    for (std::size_t c = lo; c < hi; ++c) {
        const FvPrim<double> p = fvPrimOf(q[c], q[nC + c], q[2 * nC + c], q[3 * nC + c], q[4 * nC + c], g, R);
        ev.prim[0][c] = p.rho;
        ev.prim[1][c] = p.u[0];
        ev.prim[2][c] = p.u[1];
        ev.prim[3][c] = p.u[2];
        ev.prim[4][c] = p.e;
        ev.prim[5][c] = p.p;
        ev.prim[6][c] = p.temp;
    }
}

namespace {

FvPrim<double> primAt(const FvEval& ev, std::size_t c)
{
    return {ev.prim[0][c], {ev.prim[1][c], ev.prim[2][c], ev.prim[3][c]}, ev.prim[4][c], ev.prim[5][c],
            ev.prim[6][c]};
}

} // namespace

void FvRef::bndStates(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nI = std::size_t(m.nI);
    for (std::size_t b = lo; b < hi; ++b) {
        const std::size_t f = nI + b;
        if (!geom_.active[f]) continue;
        const FvPrim<double> P = primAt(ev, std::size_t(m.own[f]));
        const V3 S = at(m.S, f), Cf = at(m.Cf, f);
        const FvPrim<double> B = fvBoundaryState(cfg_.bc[std::size_t(bcOf_[b])], P, S.data(), Cf.data(),
                                                 cfg_.tr.gamma, cfg_.tr.Rgas);
        ev.bnd[b] = B;
        ev.bphi[0][b] = B.u[0];
        ev.bphi[1][b] = B.u[1];
        ev.bphi[2][b] = B.u[2];
        ev.bphi[3][b] = B.temp;
    }
}

void FvRef::grads(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const int src[4] = {1, 2, 3, 6}; // u, v, w, T rows of prim
    for (int k = 0; k < 4; ++k)
        fvCellGradient(*fm_, geom_, gw_, ev.prim[std::size_t(src[k])], ev.bphi[std::size_t(k)],
                       {ev.grad[std::size_t(3 * k)].data(), ev.grad[std::size_t(3 * k + 1)].data(),
                        ev.grad[std::size_t(3 * k + 2)].data()},
                       lo, hi);
}

void FvRef::faces(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nI = std::size_t(m.nI);
    const Transport& tr = cfg_.tr;
    const bool viscous = tr.muRef > 0.0;
    auto cellGrad = [&](std::size_t c, int k, double g[3]) {
        for (int a = 0; a < 3; ++a) g[a] = ev.grad[std::size_t(3 * k + a)][c];
    };
    for (std::size_t f = lo; f < hi; ++f) {
        FvFlux<double> F{0.0, {0.0, 0.0, 0.0}, 0.0};
        const V3 S = at(m.S, f), d = at(m.d, f), corr = at(geom_.corr, f);
        double gu[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, gT[3] = {0, 0, 0};
        const std::size_t o = std::size_t(m.own[f]);
        const FvPrim<double> A = primAt(ev, o);
        if (f < nI) {
            const std::size_t n = std::size_t(m.nei[f]);
            const FvPrim<double> B = primAt(ev, n);
            double mu = 0.0, kappa = 0.0;
            if (viscous) {
                double ga[3], gb[3];
                for (int k = 0; k < 3; ++k) {
                    cellGrad(o, k, ga);
                    cellGrad(n, k, gb);
                    fvFaceGrad(ga, gb, A.u[k], B.u[k], d.data(), corr.data(), gu[k]);
                }
                cellGrad(o, 3, ga);
                cellGrad(n, 3, gb);
                fvFaceGrad(ga, gb, A.temp, B.temp, d.data(), corr.data(), gT);
                fvTransportAt(tr, 0.5 * (A.temp + B.temp), mu, kappa);
            }
            F = fvMeanFlux(A, B, S.data(), gu, gT, mu, kappa);
        } else if (geom_.active[f]) {
            const std::size_t b = f - nI;
            const FvBc& bc = cfg_.bc[std::size_t(bcOf_[b])];
            const FvPrim<double>& B = ev.bnd[b];
            double mu = 0.0, kappa = 0.0;
            if (viscous && (bc.kind == FvBcKind::Wall || bc.kind == FvBcKind::Slip)) {
                double ga[3];
                for (int k = 0; k < 3; ++k) {
                    cellGrad(o, k, ga);
                    fvFaceGradBnd(ga, A.u[k], B.u[k], d.data(), corr.data(), gu[k]);
                }
                cellGrad(o, 3, ga);
                fvFaceGradBnd(ga, A.temp, B.temp, d.data(), corr.data(), gT);
                const double Tf = bc.kind == FvBcKind::Wall && bc.isothermal ? 0.5 * (A.temp + B.temp) : A.temp;
                fvTransportAt(tr, Tf, mu, kappa);
            }
            F = fvBoundaryFlux(bc, A, B, S.data(), gu, gT, mu, kappa);
        }
        ev.F[0][f] = F.rho;
        ev.F[1][f] = F.m[0];
        ev.F[2][f] = F.m[1];
        ev.F[3][f] = F.m[2];
        ev.F[4][f] = F.E;
    }
}

void FvRef::gather(const FvEval& ev, const Stage& st, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nC = std::size_t(m.nC);
    const int nI = m.nI;
    for (std::size_t c = lo; c < hi; ++c) {
        double acc[kFvMeanVars] = {0, 0, 0, 0, 0};
        for (int q = m.cellStart[c]; q < m.cellStart[c + 1]; ++q) {
            const std::size_t f = std::size_t(m.cellFace[std::size_t(q)]);
            if (int(f) >= nI && !geom_.active[f]) continue;
            if (std::size_t(m.own[f]) == c)
                for (int v = 0; v < kFvMeanVars; ++v) acc[v] += ev.F[std::size_t(v)][f];
            else
                for (int v = 0; v < kFvMeanVars; ++v) acc[v] -= ev.F[std::size_t(v)][f];
        }
        const double V = m.V[c];
        for (int v = 0; v < kFvMeanVars; ++v) {
            const std::size_t i = std::size_t(v) * nC + c;
            double R = -acc[v] / V;
            if (st.src) R += st.src[i];
            st.out[i] = st.a ? st.ca * st.a[i] + st.cb * st.b[i] + st.cr * R : R;
        }
    }
}

namespace {

void sizeEval(FvEval& ev, std::size_t nC, std::size_t nB, std::size_t nF)
{
    for (auto& p : ev.prim) p.assign(nC, 0.0);
    ev.bnd.assign(nB, FvPrim<double>{});
    for (auto& p : ev.bphi) p.assign(nB, 0.0);
    for (auto& g : ev.grad) g.assign(nC, 0.0);
    for (auto& F : ev.F) F.assign(nF, 0.0);
}

} // namespace

void FvRef::evaluate(const FvState& s, FvEval& ev, FvState& out, const FvState* src) const
{
    const FaceMesh& m = *fm_;
    assert(s.nV == kFvMeanVars && s.nC == m.nC && (!src || (src->nV == s.nV && src->nC == s.nC)));
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB), nF = std::size_t(m.nI) + nB;
    sizeEval(ev, nC, nB, nF);
    if (out.nV != s.nV || out.nC != s.nC) out = FvState::zeros(s.nC, s.nV);
    const bool viscous = cfg_.tr.muRef > 0.0;
    Stage st;
    st.out = out.q.data();
    st.src = src ? src->q.data() : nullptr;
    onWorkers(workers_, [&](int w, int nw, auto&& sync) {
        auto [c0, c1] = part(nC, w, nw);
        auto [b0, b1] = part(nB, w, nw);
        auto [f0, f1] = part(nF, w, nw);
        prims(s.q.data(), ev, c0, c1);
        sync();
        bndStates(ev, b0, b1);
        sync();
        if (viscous) {
            grads(ev, c0, c1);
            sync();
        }
        faces(ev, f0, f1);
        sync();
        gather(ev, st, c0, c1);
    });
}

void FvRef::rhs(const FvState& s, FvState& out, const FvState* src) const
{
    FvEval ev;
    evaluate(s, ev, out, src);
}

void FvRef::march(FvState& s, double dt, int steps, const FvState* src) const
{
    const FaceMesh& m = *fm_;
    assert(s.nV == kFvMeanVars && s.nC == m.nC && (!src || (src->nV == s.nV && src->nC == s.nC)));
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB), nF = std::size_t(m.nI) + nB;
    FvEval ev;
    sizeEval(ev, nC, nB, nF);
    FvState q1 = FvState::zeros(s.nC, s.nV), q2 = FvState::zeros(s.nC, s.nV);
    const bool viscous = cfg_.tr.muRef > 0.0;
    // SSP-RK3 in the fused form out = ca a + cb b + cr R(b), ca = 1 - cb.
    const double cbs[3] = {0.0, 0.25, 2.0 / 3.0};
    const double* bs[3] = {s.q.data(), q1.q.data(), q2.q.data()};
    double* outs[3] = {q1.q.data(), q2.q.data(), s.q.data()};
    onWorkers(workers_, [&](int w, int nw, auto&& sync) {
        auto [c0, c1] = part(nC, w, nw);
        auto [b0, b1] = part(nB, w, nw);
        auto [f0, f1] = part(nF, w, nw);
        for (int n = 0; n < steps; ++n)
            for (int k = 0; k < 3; ++k) {
                prims(bs[k], ev, c0, c1);
                sync();
                bndStates(ev, b0, b1);
                sync();
                if (viscous) {
                    grads(ev, c0, c1);
                    sync();
                }
                faces(ev, f0, f1);
                sync();
                Stage st;
                st.a = s.q.data();
                st.b = bs[k];
                st.out = outs[k];
                st.cb = cbs[k];
                st.ca = 1.0 - st.cb;
                st.cr = k == 0 ? dt : st.cb * dt;
                st.src = src ? src->q.data() : nullptr;
                gather(ev, st, c0, c1);
                sync();
            }
    });
}

std::expected<double, std::string> FvRef::stableDt(const FvState& s, FvDtLimits lim,
                                                   std::vector<double>* local) const
{
    const FaceMesh& m = *fm_;
    assert(s.nV == kFvMeanVars && s.nC == m.nC);
    const std::size_t nC = std::size_t(m.nC);
    const Transport& tr = cfg_.tr;
    if (local) local->assign(nC, 0.0);
    const int nw = workerCount(workers_, nC);
    std::vector<double> mins(std::size_t(nw), INFINITY);
    std::vector<std::optional<std::string>> err(static_cast<std::size_t>(nw));
    forBlocks(nw, nC, [&](int w, std::size_t lo, std::size_t hi) {
        for (std::size_t c = lo; c < hi; ++c) {
            const FvPrim<double> p = fvPrimOf(s.var(0)[c], s.var(1)[c], s.var(2)[c], s.var(3)[c], s.var(4)[c],
                                              tr.gamma, tr.Rgas);
            if (!(p.rho > 0.0) || !std::isfinite(p.rho) || !(p.p > 0.0) || !std::isfinite(p.p)) {
                err[std::size_t(w)] = std::format("FV state diverged: rho {} p {} at cell {} (source cell {})", p.rho,
                                                  p.p, c, m.srcCell[c]);
                return;
            }
            const double cs = std::sqrt(tr.gamma * p.p / p.rho);
            double sum = 0.0;
            for (int q = m.cellStart[c]; q < m.cellStart[c + 1]; ++q) {
                const std::size_t f = std::size_t(m.cellFace[std::size_t(q)]);
                if (!geom_.active[f]) continue;
                const double Sx = m.S[0][f], Sy = m.S[1][f], Sz = m.S[2][f];
                sum += std::abs(p.u[0] * Sx + p.u[1] * Sy + p.u[2] * Sz) + cs * std::sqrt(Sx * Sx + Sy * Sy + Sz * Sz);
            }
            double dt = 2.0 * lim.cfl * m.V[c] / sum;
            if (tr.muRef > 0.0) {
                double mu, kappa;
                fvTransportAt(tr, p.temp, mu, kappa);
                dt = std::min(dt, lim.diff * p.rho * m.V[c] / (mu * fvDiffusivityFactor(tr) * diffCoef_[c]));
            }
            if (local) (*local)[c] = dt;
            mins[std::size_t(w)] = std::min(mins[std::size_t(w)], dt);
        }
    });
    for (auto& e : err)
        if (e) return std::unexpected(std::move(*e));
    const double dt = *std::min_element(mins.begin(), mins.end());
    if (!(dt > 0.0) || !std::isfinite(dt)) return std::unexpected("FV stable dt is not positive-finite");
    return dt;
}

} // namespace dns
