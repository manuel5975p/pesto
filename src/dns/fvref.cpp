#include "dns/fvref.hpp"

#include "dns/chem.hpp"
#include "dns/diffusion.hpp"
#include "dns/parallel.hpp"

#include <algorithm>
#include <barrier>
#include <cassert>
#include <cmath>
#include <format>
#include <optional>
#include <span>

namespace dns {
namespace {

using V3 = std::array<double, 3>;

V3 at(const std::array<std::vector<double>, 3>& a, std::size_t i) { return {a[0][i], a[1][i], a[2][i]}; }

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
    const bool wedge = p.type == FoamPatchType::Wedge;
    if (wedge != (bc.kind == FvBcKind::Slip && bc.wedge))
        return std::unexpected(std::format("patch `{}`: {}", p.name,
                                           wedge ? "a `wedge` patch takes the wedge Slip row only"
                                                 : "the wedge Slip row is for `wedge` patches only"));
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
        if (bc.exit == FvExit::Extrapolate) return {};
        if (!(bc.pOut > 0.0 && std::isfinite(bc.pOut)) || !(bc.tBack > 0.0 && std::isfinite(bc.tBack)))
            return std::unexpected(std::format("patch `{}`: outflow needs p > 0 and a backflow T > 0", p.name));
        return {};
    case FvBcKind::Total:
        if (!(bc.pTotal > 0.0 && std::isfinite(bc.pTotal)) || !(bc.tTotal > 0.0 && std::isfinite(bc.tTotal)))
            return std::unexpected(std::format("patch `{}`: total needs p0 > 0 and t0 > 0", p.name));
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

FvModuleSet fvModulesOf(const FvConfig& cfg)
{
    const auto nonzero = [](const std::array<double, 3>& v) {
        return std::ranges::any_of(v, [](double x) { return x != 0.0; });
    };
    FvModuleSet s;
    const auto on = [&](FvModule m, bool b) { s.set(std::size_t(m), b); };
    const bool lesOn = cfg.les.cs > 0.0;
    on(FvModule::LesSmag, lesOn && cfg.les.sgs != SgsModel::Sigma);
    on(FvModule::LesSigma, lesOn && cfg.les.sgs == SgsModel::Sigma);
    on(FvModule::TflesUniform, cfg.tfles && cfg.tfles->uniform);
    on(FvModule::TflesDynamic, cfg.tfles && !cfg.tfles->uniform);
    on(FvModule::TflesEfficiency, cfg.tfles && cfg.tfles->charlette);
    on(FvModule::RansSa, cfg.rans.model != RansModel::None && !ransSst(cfg.rans.model));
    on(FvModule::RansSst, ransSst(cfg.rans.model));
    on(FvModule::ChemC0, cfg.chem.nY > 0 && cfg.chem.ox < 0);
    on(FvModule::ChemC1, cfg.chem.nY > 0 && cfg.chem.ox >= 0 && cfg.chem.law.n == 0);
    on(FvModule::ChemC2b, cfg.chem.nY > 0 && cfg.chem.ox >= 0 && cfg.chem.law.n > 0);
    on(FvModule::Gravity, nonzero(cfg.gravity));
    on(FvModule::BodyForce, nonzero(cfg.bodyForce));
    on(FvModule::Turb, cfg.turb);
    on(FvModule::Shock, cfg.shock.mode != ShockMode::None);
    return s;
}

double fvAmbientP(const FvAmbient& a, const std::array<double, 3>& g, const double x[3], double Rgas)
{
    assert(a.p0 > 0.0 && a.T0 > 0.0 && Rgas > 0.0);
    const double gm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    if (!(gm > 0.0)) return a.p0;
    const V3 up = {-g[0] / gm, -g[1] / gm, -g[2] / gm};
    const double h = (x[0] - a.x0[0]) * up[0] + (x[1] - a.x0[1]) * up[1] + (x[2] - a.x0[2]) * up[2];
    return candleAmbientP(a.p0, gm, Rgas * a.T0, h);
}

std::array<double, 5> fvAmbientConserved(const FvAmbient& a, const std::array<double, 3>& g, const double x[3],
                                         const double u[3], double gamma, double Rgas)
{
    const double p = fvAmbientP(a, g, x, Rgas), rho = p / (Rgas * a.T0);
    return {rho, rho * u[0], rho * u[1], rho * u[2], p / (gamma - 1.0) + 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])};
}

FvBcTable fvAmbientRows(const FaceMesh& fm, FvBcTable t, const FvAmbient& a, const std::array<double, 3>& g,
                        double gamma, double Rgas)
{
    if (t.faceRow.empty()) {
        t.faceRow.assign(std::size_t(fm.nB), 0);
        for (std::size_t p = 0; p < fm.patches.size(); ++p)
            std::fill_n(t.faceRow.begin() + fm.patches[p].first, fm.patches[p].count, int(p));
    }
    for (std::size_t b = 0; b < std::size_t(fm.nB); ++b) {
        const FvBc& r = t.bc[std::size_t(t.faceRow[b])];
        if (r.kind != FvBcKind::Far && r.kind != FvBcKind::Outflow) continue;
        const std::size_t c = std::size_t(fm.own[std::size_t(fm.nI) + b]);
        const double x[3] = {fm.C[0][c], fm.C[1][c], fm.C[2][c]};
        FvBc n = r;
        if (r.kind == FvBcKind::Far) {
            const double u[3] = {r.qFar[1] / r.qFar[0], r.qFar[2] / r.qFar[0], r.qFar[3] / r.qFar[0]};
            const std::array<double, 5> q = fvAmbientConserved(a, g, x, u, gamma, Rgas);
            std::ranges::copy(q, n.qFar);
        } else {
            n.pOut = fvAmbientP(a, g, x, Rgas);
        }
        t.faceRow[b] = int(t.bc.size());
        t.bc.push_back(n);
    }
    return t;
}

FvState fvAmbientState(const FaceMesh& fm, const FvAmbient& a, const std::array<double, 3>& g, double gamma,
                       double Rgas, int nV)
{
    FvState s = FvState::zeros(fm.nC, nV);
    const double u[3] = {0.0, 0.0, 0.0};
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double x[3] = {fm.C[0][c], fm.C[1][c], fm.C[2][c]};
        const std::array<double, 5> q = fvAmbientConserved(a, g, x, u, gamma, Rgas);
        for (int v = 0; v < kFvMeanVars; ++v) s.var(v)[c] = q[std::size_t(v)];
    }
    return s;
}

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
    const FvChem& ch = cfg.chem;
    if (ch.nY < 0 || ch.nY > kFvMaxSpecies || (ch.nY > 0 && (ch.fuel < 0 || ch.fuel >= ch.nY)))
        return std::unexpected(std::format("FV species: nY must be in [0, {}] and fuel one of them", kFvMaxSpecies));
    if (ch.nY > 0 && (!(ch.invPr > 0.0) || !(ch.q >= 0.0) || !(ch.Ta >= 0.0) || std::isnan(ch.lnA)))
        return std::unexpected("FV chemistry: needs 1/Pr > 0, q >= 0, Ta >= 0");
    if (ch.ox >= 0 && (ch.ox >= ch.nY || ch.ox == ch.fuel || !(ch.stoich > 0.0)))
        return std::unexpected("FV chemistry: the oxidizer is a species other than the fuel, stoich > 0");
    if (const PhiRateLaw<double>& L = ch.law; L.n != 0) {
        const auto xi = std::span(L.xi, std::size_t(std::clamp(L.n, 0, PhiRateLaw<double>::kMaxNodes)));
        const bool nodes = L.n >= 2 && L.n <= PhiRateLaw<double>::kMaxNodes &&
                           std::ranges::adjacent_find(xi, std::greater_equal<>{}) == xi.end();
        if (ch.ox < 0 || !nodes || !(L.invDen > 0.0) || !std::isfinite(L.invDen))
            return std::unexpected("FV chemistry: the C2b law needs two reactants and 2 to 32 strictly ascending nodes");
    }
    const FvLes& les = cfg.les;
    if (!(les.cs >= 0.0) || !std::isfinite(les.cs) || !(les.prt > 0.0))
        return std::unexpected("FV LES: needs Cs >= 0 and Pr_t > 0");
    if (!(cfg.shock.eps4 >= 0.0 && cfg.shock.eps4 <= kFvShockEps4Max)) return std::unexpected("FV --shock-d4: must be in [0, 0.125]");
    const bool lesOn = les.cs > 0.0;
    // Every pair and requirement of the configured modules: the inventory (dns/fvmodules,
    // docs/fvcomp.md "Combinations"), by name.
    if (auto ok = fvModulesCheck(fvModulesOf(cfg)); !ok) return std::unexpected(std::move(ok.error()));
    if (cfg.tfles) {
        const TflesSetup& t = *cfg.tfles;
        if (!(t.F >= 1.0) || !(t.deltaL > 0.0) || !(t.nuU > 0.0) || t.dilate < 0)
            return std::unexpected("FV TFLES: needs F >= 1, delta_L > 0, nu_u > 0, a dilation radius >= 0");
        if (!(tr.muRef > 0.0)) return std::unexpected("FV TFLES needs mu > 0 (it thickens the molecular fluxes)");
    }
    if (cfg.bcRow.empty() ? cfg.bc.size() != fm.patches.size() : cfg.bc.size() < fm.patches.size())
        return std::unexpected(std::format("FV boundary conditions: {} given for {} patches", cfg.bc.size(),
                                           fm.patches.size()));
    for (std::size_t p = 0; p < fm.patches.size(); ++p)
        if (auto ok = checkBc(cfg.bc[p], fm.patches[p], fm.emptyAxis, tr); !ok)
            return std::unexpected(std::move(ok.error()));
    if (!cfg.bcRow.empty()) {
        if (cfg.bcRow.size() != std::size_t(fm.nB))
            return std::unexpected(std::format("FV boundary rows: {} given for {} boundary faces", cfg.bcRow.size(), fm.nB));
        for (std::size_t p = 0; p < fm.patches.size(); ++p)
            for (int i = 0; i < fm.patches[p].count; ++i) {
                const int row = cfg.bcRow[std::size_t(fm.patches[p].first + i)];
                if (row < 0 || std::size_t(row) >= cfg.bc.size() || cfg.bc[std::size_t(row)].kind != cfg.bc[p].kind)
                    return std::unexpected(std::format("patch `{}`: a face row must exist and keep the patch's kind",
                                                       fm.patches[p].name));
                if (auto ok = checkBc(cfg.bc[std::size_t(row)], fm.patches[p], fm.emptyAxis, tr); !ok)
                    return std::unexpected(std::move(ok.error()));
            }
    }

    const bool gravOn = std::ranges::any_of(cfg.gravity, [](double v) { return v != 0.0; });
    if (cfg.ambient && !gravOn) return std::unexpected("FV hydrostatic ambient without gravity: nothing to balance");
    if (cfg.ambient && !(cfg.ambient->p0 > 0.0 && cfg.ambient->T0 > 0.0 && std::isfinite(cfg.ambient->p0) &&
                         std::isfinite(cfg.ambient->T0)))
        return std::unexpected("FV hydrostatic ambient: needs p0 > 0 and T0 > 0");
    if (gravOn) {
        if (fm.emptyAxis >= 0 && cfg.gravity[std::size_t(fm.emptyAxis)] != 0.0)
            return std::unexpected("FV gravity: along the empty axis of a 2D mesh");
        for (std::size_t p = 0; p < fm.patches.size(); ++p) {
            const FvBc& b = cfg.bc[p];
            const bool moving = std::ranges::any_of(b.uWall, [](double v) { return v != 0.0; }) ||
                                std::ranges::any_of(b.omega, [](double v) { return v != 0.0; });
            if (b.kind == FvBcKind::Wall && moving)
                return std::unexpected(std::format("patch `{}`: gravity takes resting walls only", fm.patches[p].name));
            if (b.kind != FvBcKind::Far && b.kind != FvBcKind::Outflow) continue;
            // An open face's reference pressure is the ambient column's at its owner's
            // height: without it the face is a pressure mismatch of rho g h (docs/fvcomp.md
            // "Hydrostatic ambient").
            const MeshPatch& mp = fm.patches[p];
            if (!cfg.ambient)
                return std::unexpected(std::format("patch `{}`: an open patch under gravity needs the hydrostatic "
                                                   "ambient (FvConfig::ambient, rows of fvAmbientRows)",
                                                   mp.name));
            for (int i = 0; i < mp.count; ++i) {
                const std::size_t f = std::size_t(mp.first + i);
                const FvBc& row = cfg.bc[std::size_t(cfg.bcRow.empty() ? int(p) : cfg.bcRow[f])];
                const std::size_t c = std::size_t(fm.own[std::size_t(fm.nI) + f]);
                const double x[3] = {fm.C[0][c], fm.C[1][c], fm.C[2][c]};
                const double pa = fvAmbientP(*cfg.ambient, cfg.gravity, x, tr.Rgas);
                double pr = row.pOut, Tr = cfg.ambient->T0;
                if (row.kind == FvBcKind::Far) {
                    const FvPrim<double> F = fvPrimOf(row.qFar[0], row.qFar[1], row.qFar[2], row.qFar[3], row.qFar[4],
                                                      tr.gamma, tr.Rgas);
                    pr = F.p;
                    Tr = F.temp;
                }
                if (!(std::abs(pr / pa - 1.0) <= 1e-12) || !(std::abs(Tr / cfg.ambient->T0 - 1.0) <= 1e-12))
                    return std::unexpected(std::format("patch `{}`: face {} is not the hydrostatic ambient's row at its "
                                                       "owner's centroid (p {:.17g} vs p_a {:.17g}; fvAmbientRows)",
                                                       mp.name, i, pr, pa));
            }
        }
    }

    FvRef r;
    r.fm_ = &fm;
    auto geom = fvFaceGeom(fm, cfg.rule, cfg.alpha, threads);
    if (!geom) return std::unexpected(std::move(geom.error()));
    r.geom_ = std::move(*geom);
    auto gw = fvGradWeights(fm, r.geom_, cfg.grad, threads);
    if (!gw) return std::unexpected(std::move(gw.error()));
    r.gw_ = std::move(*gw);
    r.cfg_ = std::move(cfg);

    r.patchOf_.assign(std::size_t(fm.nB), -1);
    for (std::size_t p = 0; p < fm.patches.size(); ++p)
        std::fill_n(r.patchOf_.begin() + fm.patches[p].first, fm.patches[p].count, int(p));
    r.bcOf_ = r.cfg_.bcRow.empty() ? r.patchOf_ : r.cfg_.bcRow;
    r.diffCoef_.assign(std::size_t(fm.nC), 0.0);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        double s = 0.0;
        for (int q = fm.cellStart[c]; q < fm.cellStart[c + 1]; ++q) {
            const std::size_t f = std::size_t(fm.cellFace[std::size_t(q)]);
            if (!r.geom_.active[f]) continue;
            if (f >= std::size_t(fm.nI) && r.cfg_.bc[std::size_t(r.bcOf_[f - std::size_t(fm.nI)])].wedge) {
                // d_z, the cell's half width: d's in-plane part is a centroid offset, 1/6 dr on a
                // prism, and it would hide the term at a small angle.
                const double sh = std::sin(fm.wedgeHalf);
                s += fm.V[c] * sh * sh / (fm.d[2][f] * fm.d[2][f]);
                continue;
            }
            s += r.geom_.corr[0][f] * fm.S[0][f] + r.geom_.corr[1][f] * fm.S[1][f] +
                 r.geom_.corr[2][f] * fm.S[2][f];
        }
        r.diffCoef_[c] = s;
    }
    r.workers_ = workerCount(threads, std::max<std::size_t>(1, std::size_t(fm.nC) / 256));
    const FvRansConfig& rc = r.cfg_.rans;
    if (rc.model != RansModel::None) {
        if (!(r.cfg_.tr.muRef > 0.0)) return std::unexpected("FV RANS needs mu > 0");
        if (rc.wallDist.size() != std::size_t(fm.nC) ||
            !std::ranges::all_of(rc.wallDist, [](double v) { return v > 0.0 && std::isfinite(v); }))
            return std::unexpected("FV RANS: the wall distance must be positive on every cell");
    }
    r.nV_ = kFvMeanVars + r.cfg_.chem.nY + ransScalars(rc.model);
    if (r.cfg_.shock.mode != ShockMode::None) r.lines_ = fvShockLines(fm, r.geom_);
    if (lesOn || r.cfg_.tfles) r.delta2_ = fvLesDelta2(fm);
    if (r.cfg_.tfles) {
        auto w = fvTflesWalk(fm, r.geom_);
        if (!w) return std::unexpected(std::move(w.error()));
        r.walk_ = std::move(*w);
    }
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
    const bool les = cfg_.les.cs > 0.0;
    const bool viscous = tr.muRef > 0.0 || les, rans = cfg_.rans.model != RansModel::None || les;
    const double kapEddy = tr.cp() / (les ? cfg_.les.prt : kPrandtlTurb);
    const auto& gv = cfg_.gravity;
    const bool hasG = gv[0] != 0.0 || gv[1] != 0.0 || gv[2] != 0.0;
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
            if (rans) {
                const double nf = 0.5 * (ev.nut[o] + ev.nut[n]);
                mu += nf;
                kappa += nf * kapEddy;
            }
            F = fvMeanFlux(A, B, S.data(), gu, gT, mu, kappa);
        } else if (geom_.active[f]) {
            const std::size_t b = f - nI;
            const FvBc& bc = cfg_.bc[std::size_t(bcOf_[b])];
            const FvPrim<double>& B = ev.bnd[b];
            double mu = 0.0, kappa = 0.0;
            if (viscous && (bc.kind == FvBcKind::Wall || bc.kind == FvBcKind::Slip || bc.diffusive)) {
                double ga[3];
                for (int k = 0; k < 3; ++k) {
                    cellGrad(o, k, ga);
                    fvFaceGradBnd(ga, A.u[k], B.u[k], d.data(), corr.data(), gu[k]);
                }
                cellGrad(o, 3, ga);
                fvFaceGradBnd(ga, A.temp, B.temp, d.data(), corr.data(), gT);
                const double Tf = (bc.kind == FvBcKind::Wall && bc.isothermal) || bc.diffusive ? 0.5 * (A.temp + B.temp) : A.temp;
                fvTransportAt(tr, Tf, mu, kappa);
                if (rans && bc.kind == FvBcKind::Slip) { // the even image; a Wall face: mu_t 0
                    mu += ev.nut[o];
                    kappa += ev.nut[o] * kapEddy;
                }
            }
            F = fvBoundaryFlux(bc, A, B, S.data(), gu, gT, mu, kappa);
        }
        if (hasG && f < nI) {
            double o[4];
            fvGravityFace(fvHydroRho(A, primAt(ev, std::size_t(m.nei[f])), tr.Rgas), F.rho, gv.data(), d.data(), S.data(), o);
            for (std::size_t k = 0; k < 4; ++k) ev.Fg[k][f] = o[k];
        }
        ev.F[0][f] = F.rho;
        ev.F[1][f] = F.m[0];
        ev.F[2][f] = F.m[1];
        ev.F[3][f] = F.m[2];
        ev.F[4][f] = F.E;
    }
}

void FvRef::shockSense(FvEval& ev) const
{
    if (cfg_.shock.force) {
        std::ranges::fill(ev.sigma, std::uint8_t(255));
        return;
    }
    fvShockSense(lines_, ev.prim[5], ev.sigma);
}

// One write per face, ascending: conservative by construction (docs/nozzle.md, kFvShockFace).
void FvRef::shockFaces(const double* q, FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nI = std::size_t(m.nI), nC = std::size_t(m.nC);
    const double eps = cfg_.shock.eps4, g = cfg_.tr.gamma;
    for (std::size_t f = lo; f < std::min(hi, nI); ++f) {
        const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
        const int pm = lines_.faceLine[2 * f], np = lines_.faceLine[2 * f + 1];
        std::uint8_t sg = std::max(ev.sigma[o], ev.sigma[n]);
        if (pm >= 0) sg = std::max(sg, ev.sigma[std::size_t(pm)]);
        if (np >= 0) sg = std::max(sg, ev.sigma[std::size_t(np)]);
        if (sg == 0 && eps == 0.0) continue;
        const FvPrim<double> A = primAt(ev, o), B = primAt(ev, n);
        const V3 S = at(m.S, f);
        const double sigma = double(sg) / 255.0;
        if (sg > 0) {
            double d[5];
            fvShockFlux(sigma, A, B, S.data(), g, d);
            for (std::size_t k = 0; k < 5; ++k) ev.F[k][f] += d[k];
        }
        if (pm < 0 || np < 0 || sg == 255 || eps == 0.0) continue;
        const double l = fvD4Rate(eps, sigma, A, B, S.data());
        for (std::size_t k = 0; k < 5; ++k) {
            const double a = l * fvD4Row(q[k * nC + std::size_t(pm)], q[k * nC + o], q[k * nC + n], q[k * nC + std::size_t(np)]);
            if (a != 0.0) ev.F[k][f] += a; // a uniform row keeps F's bytes, -0.0 included
        }
    }
}

// Y_k = rhoY_k / rho and the ONE omega of this stage, read by the species row
// and the heat release.
void FvRef::speciesCells(const double* q, FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const FvChem& ch = cfg_.chem;
    const std::size_t nC = std::size_t(fm_->nC);
    for (std::size_t c = lo; c < hi; ++c) {
        const double rho = q[c];
        for (std::size_t s = 0; s < std::size_t(ch.nY); ++s)
            ev.Y[s][c] = q[(std::size_t(kFvMeanVars) + s) * nC + c] / rho;
        const double yf = ev.Y[std::size_t(ch.fuel)][c], T = ev.prim[6][c];
        // C2b: ln C of the cell's unfloored Y_F, Y_O (kChemCellPass2Phi's arguments).
        const double yo = ch.ox >= 0 ? ev.Y[std::size_t(ch.ox)][c] : 0.0;
        const double lnA = ch.law.n > 0 ? ch.lnA + phiRateLn(ch.law, yf, yo) : ch.lnA;
        const double w = ch.ox >= 0 ? arrheniusRate2(lnA, ch.Ta, rho, yf, yo, T) : arrheniusRate(ch.lnA, ch.Ta, rho, yf, T);
        ev.omega[c] = cfg_.tfles ? w * ev.eof[c] : w; // TFLES: (E/F) omega, kChemCellPass<true>
    }
}

// mu_t = rho (C Delta)^2 op(grad u): DeviceFvComp's kFvNut, Delta^2 = fvLesDelta2.
void FvRef::lesCells(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const double cs = cfg_.les.cs;
    const int sigma = cfg_.les.sgs == SgsModel::Sigma ? 1 : 0;
    for (std::size_t c = lo; c < hi; ++c) {
        double gd[3][3];
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) gd[i][j] = ev.grad[3 * i + j][c];
        ev.nut[c] = ev.prim[0][c] * (cs * cs * delta2_[c]) * sgsRate(sigma, gd);
    }
}

// The raw sensor tanh(beta w_s) into tfWork[0] (w_s the rate at TaS, gpu/tfles's
// kTflesSensorRaw) and, for the efficiency, curl u off the cell gradients.
void FvRef::tflesRaw(const double* q, FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const TflesSetup& t = *cfg_.tfles;
    const std::size_t nC = std::size_t(fm_->nC), iF = std::size_t(kFvMeanVars + cfg_.chem.fuel);
    for (std::size_t c = lo; c < hi; ++c) {
        if (!t.uniform) {
            const double r = q[c];
            ev.tfWork[0][c] = std::tanh(t.sensorBeta * arrheniusRate(t.lnAs, t.TaS, r, q[iF * nC + c] / r, ev.prim[6][c]));
        }
        if (t.charlette) { // grad rows [3 field + axis]; fields u, v, w
            ev.curl[0][c] = ev.grad[7][c] - ev.grad[5][c];
            ev.curl[1][c] = ev.grad[2][c] - ev.grad[6][c];
            ev.curl[2][c] = ev.grad[3][c] - ev.grad[1][c];
        }
    }
}

// Pass p of the T-gated max filter: tfWork[p & 1] -> tfWork[(p + 1) & 1] along walked
// axis p, dilate steps each way; a cell takes the cells it reaches with T >= its own.
void FvRef::tflesDilate(FvEval& ev, int pass, std::size_t lo, std::size_t hi) const
{
    const std::size_t nC = std::size_t(fm_->nC);
    const std::vector<double>& in = ev.tfWork[std::size_t(pass & 1)];
    std::vector<double>& out = ev.tfWork[std::size_t((pass + 1) & 1)];
    const std::vector<double>& T = ev.prim[6];
    const int rad = cfg_.tfles->dilate;
    for (std::size_t c = lo; c < hi; ++c) {
        double m = in[c];
        for (std::size_t dir = 0; dir < 2; ++dir) {
            const int* nx = walk_.next.data() + (std::size_t(pass) * 2 + dir) * nC;
            int n = int(c);
            for (int k = 0; k < rad; ++k) {
                n = nx[std::size_t(n)];
                if (n < 0) break;
                if (T[std::size_t(n)] >= T[c]) m = std::max(m, in[std::size_t(n)]);
            }
        }
        out[c] = m;
    }
}

// Colin's grid-scale u' = c2 h^3 |lap(curl u)|, h = sqrt(Delta^2), the compact
// Laplacian (1/V) sum_f (phi_N - phi_P) corr.S over internal faces.
void FvRef::tflesColin(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    for (std::size_t c = lo; c < hi; ++c) {
        double s2 = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
            double l = 0.0;
            for (int q = m.cellStart[c]; q < m.cellStart[c + 1]; ++q) {
                const std::size_t f = std::size_t(m.cellFace[std::size_t(q)]);
                if (f >= std::size_t(m.nI)) continue;
                const std::size_t o = std::size_t(m.own[f]) == c ? std::size_t(m.nei[f]) : std::size_t(m.own[f]);
                l += walk_.lapW[std::size_t(q)] * (ev.curl[k][o] - ev.curl[k][c]);
            }
            l /= m.V[c];
            s2 += l * l;
        }
        const double h = std::sqrt(delta2_[c]);
        ev.uh[c] = kColinC2 * h * h * h * std::sqrt(s2);
    }
}

// F = 1 + (F_max - 1) Omega, E = Charlette off the edge layers: kTflesCombine.
void FvRef::tflesCombine(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const TflesSetup& t = *cfg_.tfles;
    const std::vector<double>& W = ev.tfWork[std::size_t(walk_.nAxes & 1)];
    for (std::size_t c = lo; c < hi; ++c) {
        const double w = t.uniform ? 1.0 : W[c];
        const double F = 1.0 + (t.F - 1.0) * w;
        double E = 1.0;
        if (t.charlette && !walk_.edge[c]) {
            const double D = F * t.deltaL;
            const double up = colinUprime(ev.uh[c], D, std::sqrt(delta2_[c]));
            E = charletteEfficiency(F, up, up * D / t.nuU, t.effBeta);
        }
        ev.fe[c] = F * E;
        ev.eof[c] = E / F;
        ev.om[c] = w;
    }
}

template <class Sync>
void FvRef::tflesFields(const double* q, FvEval& ev, std::size_t c0, std::size_t c1, Sync&& sync) const
{
    const TflesSetup& t = *cfg_.tfles;
    tflesRaw(q, ev, c0, c1);
    sync();
    if (!t.uniform)
        for (int p = 0; p < walk_.nAxes; ++p) {
            tflesDilate(ev, p, c0, c1);
            sync();
        }
    if (t.charlette) {
        tflesColin(ev, c0, c1);
        sync();
    }
    tflesCombine(ev, c0, c1);
    sync();
}

void FvRef::speciesBnd(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nI = std::size_t(m.nI);
    for (std::size_t b = lo; b < hi; ++b) {
        const std::size_t f = nI + b;
        if (!geom_.active[f]) continue;
        const std::size_t o = std::size_t(m.own[f]);
        const FvPrim<double>& B = ev.bnd[b];
        const double mdot = B.rho * (B.u[0] * m.S[0][f] + B.u[1] * m.S[1][f] + B.u[2] * m.S[2][f]);
        for (std::size_t s = 0; s < ev.Y.size(); ++s)
            ev.bY[s][b] = fvBoundaryY(cfg_.bc[std::size_t(bcOf_[b])], int(s), ev.Y[s][o], mdot);
    }
}

void FvRef::speciesGrads(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    for (std::size_t s = 0; s < ev.Y.size(); ++s)
        fvCellGradient(*fm_, geom_, gw_, ev.Y[s], ev.bY[s],
                       {ev.gradY[3 * s].data(), ev.gradY[3 * s + 1].data(), ev.gradY[3 * s + 2].data()}, lo, hi);
}

// After faces(): F[0] is the mass flux the species ride. rho D at the face's own
// mu (fvTransportAt on the mean T, the one Sutherland evaluation of the face).
void FvRef::speciesFaces(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nI = std::size_t(m.nI), nY = ev.Y.size();
    const Transport& tr = cfg_.tr;
    const bool les = cfg_.les.cs > 0.0, thick = cfg_.tfles.has_value();
    const bool viscous = tr.muRef > 0.0 || les, shared = cfg_.chem.ox >= 0;
    const double kapLes = tr.cp() / cfg_.les.prt;
    auto cellGrad = [&](std::size_t c, std::size_t s, double g[3]) {
        for (std::size_t a = 0; a < 3; ++a) g[a] = ev.gradY[3 * s + a][c];
    };
    for (std::size_t f = lo; f < hi; ++f) {
        const double mdot = ev.F[0][f];
        if (f >= nI) {
            const std::size_t b = f - nI;
            const FvBc& bc = cfg_.bc[std::size_t(bcOf_[b])];
            for (std::size_t s = 0; s < nY; ++s) ev.Fy[s][f] = geom_.active[f] ? mdot * ev.bY[s][b] : 0.0;
            if (!bc.diffusive || !viscous || !geom_.active[f]) continue;
            const std::size_t o = std::size_t(m.own[f]);
            const V3 S = at(m.S, f), d = at(m.d, f), corr = at(geom_.corr, f);
            double mu, kappa;
            fvTransportAt(tr, 0.5 * (ev.prim[6][o] + ev.bnd[b].temp), mu, kappa);
            for (std::size_t s = 0; s < nY; ++s) {
                double ga[3], g[3];
                cellGrad(o, s, ga);
                fvFaceGradBnd(ga, ev.Y[s][o], ev.bY[s][b], d.data(), corr.data(), g);
                ev.Fy[s][f] = fvSpeciesFlux(mdot, ev.bY[s][b], mu * cfg_.chem.invPr, g, S.data());
            }
            continue;
        }
        const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
        const V3 S = at(m.S, f), d = at(m.d, f), corr = at(geom_.corr, f);
        double rhoD = 0.0, kapf = 0.0;
        if (viscous) {
            double mu;
            fvTransportAt(tr, 0.5 * (ev.prim[6][o] + ev.prim[6][n]), mu, kapf);
            rhoD = mu * cfg_.chem.invPr;
        }
        // LES, TFLES (docs/fvcomp.md "Species LES and TFLES"): the eddy and thickened rho D,
        // the thickened energy flux added to F_E.
        if (les || thick) {
            const double nutf = les ? 0.5 * (ev.nut[o] + ev.nut[n]) : 0.0, invSct = 1.0 / cfg_.les.prt;
            if (!thick) {
                rhoD = fvEddyRhoD(rhoD, nutf, invSct);
            } else {
                const double FEf = 0.5 * (ev.fe[o] + ev.fe[n]), Omf = 0.5 * (ev.om[o] + ev.om[n]);
                rhoD = fvThickRhoD(rhoD, FEf, Omf, nutf, invSct, les);
                double ga[3], gb[3], gT[3];
                for (std::size_t a = 0; a < 3; ++a) {
                    ga[a] = ev.grad[9 + a][o];
                    gb[a] = ev.grad[9 + a][n];
                }
                fvFaceGrad(ga, gb, ev.prim[6][o], ev.prim[6][n], d.data(), corr.data(), gT);
                const double gTS = gT[0] * S[0] + gT[1] * S[1] + gT[2] * S[2];
                ev.F[4][f] = ev.F[4][f] + fvThickEnergyFlux(FEf, Omf, nutf, kapf, kapLes, gTS, les);
            }
        }
        double th = 2.0;
        for (std::size_t s = 0; shared && s < nY; ++s) {
            double ga[3], gb[3];
            cellGrad(o, s, ga);
            cellGrad(n, s, gb);
            th = fvThetaVote(th, mdot, ev.Y[s][o], ev.Y[s][n], ga, gb, d.data());
        }
        for (std::size_t s = 0; s < nY; ++s) {
            double ga[3], gb[3], g[3];
            cellGrad(o, s, ga);
            cellGrad(n, s, gb);
            const double YP = ev.Y[s][o], YN = ev.Y[s][n];
            fvFaceGrad(ga, gb, YP, YN, d.data(), corr.data(), g);
            const double Yf = shared ? fvFaceYShared(mdot, YP, YN, th) : fvFaceY(mdot, YP, YN, ga, gb, d.data());
            ev.Fy[s][f] = fvSpeciesFlux(mdot, Yf, rhoD, g, S.data());
        }
    }
}

void FvRef::gather(const FvEval& ev, const Stage& st, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nC = std::size_t(m.nC);
    const int nI = m.nI;
    const FvChem& ch = cfg_.chem;
    const int nV = kFvMeanVars + ch.nY;
    const std::size_t iFuel = std::size_t(kFvMeanVars + ch.fuel);
    const std::size_t iOx = ch.ox >= 0 ? std::size_t(kFvMeanVars + ch.ox) : std::size_t(nV);
    auto flux = [&](int v, std::size_t f) {
        return v < kFvMeanVars ? ev.F[std::size_t(v)][f] : ev.Fy[std::size_t(v - kFvMeanVars)][f];
    };
    const std::array<double, 3>& f3 = cfg_.bodyForce;
    const bool force = f3[0] != 0.0 || f3[1] != 0.0 || f3[2] != 0.0;
    const auto& g = cfg_.gravity;
    const bool hasG = g[0] != 0.0 || g[1] != 0.0 || g[2] != 0.0;
    for (std::size_t c = lo; c < hi; ++c) {
        double acc[kFvMeanVars + kFvMaxSpecies] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        double ag[4] = {0, 0, 0, 0};
        for (int q = m.cellStart[c]; q < m.cellStart[c + 1]; ++q) {
            const std::size_t f = std::size_t(m.cellFace[std::size_t(q)]);
            if (int(f) >= nI && !geom_.active[f]) continue;
            if (hasG && int(f) < nI)
                for (std::size_t k = 0; k < 4; ++k) ag[k] += ev.Fg[k][f];
            if (std::size_t(m.own[f]) == c)
                for (int v = 0; v < nV; ++v) acc[v] += flux(v, f);
            else
                for (int v = 0; v < nV; ++v) acc[v] -= flux(v, f);
        }
        const double V = m.V[c];
        for (int v = 0; v < nV; ++v) {
            const std::size_t i = std::size_t(v) * nC + c;
            double R = -acc[v] / V;
            if (ch.nY > 0) {
                if (v == 4) R += ch.q * ev.omega[c];
                if (std::size_t(v) == iFuel) R -= ev.omega[c];
                if (std::size_t(v) == iOx) R -= ch.stoich * ev.omega[c];
            }
            if (st.src) R += st.src[i];
            if (force && v >= 1 && v <= 3) R += f3[std::size_t(v - 1)];
            if (force && v == 4) R += f3[0] * ev.prim[1][c] + f3[1] * ev.prim[2][c] + f3[2] * ev.prim[3][c];
            if (hasG && v >= 1 && v <= 4) R += ag[std::size_t(v - 1)] / V;
            st.out[i] = st.a ? st.ca * st.a[i] + st.cb * st.b[i] + st.cr * R : R;
        }
    }
}

void FvRef::ransFields(const double* q, FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const std::size_t nC = std::size_t(fm_->nC);
    const bool sst = ransSst(cfg_.rans.model);
    for (std::size_t c = lo; c < hi; ++c) {
        const double rho = ev.prim[0][c];
        ev.tfld[0][c] = q[5 * nC + c] / rho;
        ev.tfld[1][c] = sst ? q[6 * nC + c] / rho : rho;
    }
}

void FvRef::ransBnd(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nI = std::size_t(m.nI);
    const FvRansConfig& rc = cfg_.rans;
    const bool sst = ransSst(rc.model);
    for (std::size_t b = lo; b < hi; ++b) {
        const std::size_t f = nI + b, o = std::size_t(m.own[f]);
        if (!geom_.active[f]) continue;
        const FvBc& bc = cfg_.bc[std::size_t(bcOf_[b])];
        const FvPrim<double> A = primAt(ev, o);
        double mu = 0.0, kappa = 0.0;
        if (bc.kind == FvBcKind::Wall) fvTransportAt(cfg_.tr, fvWallTemp(bc, A, ev.bnd[b]), mu, kappa);
        const V3 S = at(m.S, f);
        const double pA[2] = {ev.tfld[0][o], ev.tfld[1][o]};
        double v[2];
        fvRansBndFields(sst, bc, ev.bnd[b], S.data(), pA, rc.phiInf.data(), mu, rc.wallDist[o], cfg_.tr.Rgas, v);
        ev.tbnd[0][b] = v[0];
        ev.tbnd[1][b] = v[1];
    }
}

void FvRef::ransGrads(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    for (std::size_t k = 0; k < std::size_t(kFvRansFields); ++k)
        fvCellGradient(*fm_, geom_, gw_, ev.tfld[k], ev.tbnd[k],
                       {ev.tgrad[3 * k].data(), ev.tgrad[3 * k + 1].data(), ev.tgrad[3 * k + 2].data()}, lo, hi);
}

void FvRef::ransCells(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const FvRansConfig& rc = cfg_.rans;
    const bool sst = ransSst(rc.model);
    for (std::size_t c = lo; c < hi; ++c) {
        double mu, kappa, gd[3][3], g0[3], g1[3];
        fvTransportAt(cfg_.tr, ev.prim[6][c], mu, kappa);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) gd[i][j] = ev.grad[3 * i + j][c];
            g0[i] = ev.tgrad[i][c];
            g1[i] = ev.tgrad[3 + i][c];
        }
        const double rho = ev.prim[0][c], d = rc.wallDist[c];
        const FvRansCell<double> o = sst ? fvSstCell(rc.model, rho, ev.tfld[0][c], ev.tfld[1][c], mu, gd, d, g0, g1)
                                         : fvSaCell(rho, ev.tfld[0][c], mu, gd, d, g0, g1);
        ev.nut[c] = o.mut;
        ev.f1[c] = o.f1;
        ev.tsrc[0][c] = rc.transportOnly ? 0.0 : o.R[0];
        ev.tsrc[1][c] = rc.transportOnly ? 0.0 : o.R[1];
        ev.tsrc[2][c] = rc.transportOnly ? 0.0 : o.J[0];
        ev.tsrc[3][c] = rc.transportOnly ? 0.0 : o.J[1];
    }
}

void FvRef::ransFaces(FvEval& ev, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nI = std::size_t(m.nI);
    const FvRansConfig& rc = cfg_.rans;
    const bool sst = ransSst(rc.model), only = rc.transportOnly;
    const int nT = ransScalars(rc.model);
    const Transport& tr = cfg_.tr;
    auto cellGrad = [&](std::size_t c, int k, double g[3]) {
        for (int a = 0; a < 3; ++a) g[a] = ev.tgrad[std::size_t(3 * k + a)][c];
    };
    auto dotS = [](const double g[3], const V3& S) { return g[0] * S[0] + g[1] * S[1] + g[2] * S[2]; };
    for (std::size_t f = lo; f < hi; ++f) {
        if (!geom_.active[f]) continue;
        const std::size_t o = std::size_t(m.own[f]);
        const V3 S = at(m.S, f), d = at(m.d, f), corr = at(geom_.corr, f);
        const double mdot = ev.F[0][f];
        if (f < nI) {
            const std::size_t n = std::size_t(m.nei[f]);
            double mu, kappa;
            fvTransportAt(tr, 0.5 * (ev.prim[6][o] + ev.prim[6][n]), mu, kappa);
            const double rf = 0.5 * (ev.prim[0][o] + ev.prim[0][n]);
            const double F1f = 0.5 * (ev.f1[o] + ev.f1[n]), mutf = 0.5 * (ev.nut[o] + ev.nut[n]);
            for (int e = 0; e < nT; ++e) {
                const std::size_t q = std::size_t(e);
                const double pa = ev.tfld[q][o], pb = ev.tfld[q][n];
                if (only) {
                    ev.Ft[q][f] = fvUpwind(mdot, pa, pb);
                    continue;
                }
                double ga[3], gb[3], g[3];
                cellGrad(o, e, ga);
                cellGrad(n, e, gb);
                fvFaceGrad(ga, gb, pa, pb, d.data(), corr.data(), g);
                const double gS = dotS(g, S);
                if (!sst) {
                    ev.Ft[q][f] = fvSaFaceFlux(mdot, pa, pb, mu, rf, gS);
                    continue;
                }
                const double sig = e ? sstBlendC(F1f, sst::sigmaW1, sst::sigmaW2)
                                     : sstBlendC(F1f, sst::sigmaK1, sst::sigmaK2);
                ev.Ft[q][f] = fvSstFaceFlux(mdot, pa, pb, mu, sig, mutf, gS);
            }
            continue;
        }
        const std::size_t b = f - nI;
        const FvBc& bc = cfg_.bc[std::size_t(bcOf_[b])];
        for (int e = 0; e < nT; ++e) {
            const std::size_t q = std::size_t(e);
            if (bc.kind == FvBcKind::Slip) {
                ev.Ft[q][f] = 0.0;
            } else if (bc.kind == FvBcKind::Wall) {
                if (only) {
                    ev.Ft[q][f] = 0.0;
                    continue;
                }
                double mu, kappa, ga[3], g[3];
                fvTransportAt(tr, fvWallTemp(bc, primAt(ev, o), ev.bnd[b]), mu, kappa);
                cellGrad(o, e, ga);
                fvFaceGradBnd(ga, ev.tfld[q][o], ev.tbnd[q][b], d.data(), corr.data(), g);
                ev.Ft[q][f] = -(sst ? mu : (1.0 / sa::sigma) * mu) * dotS(g, S);
            } else {
                ev.Ft[q][f] = mdot * ev.tbnd[q][b];
            }
        }
    }
}

void FvRef::ransGather(const FvEval& ev, const Stage& st, std::size_t lo, std::size_t hi) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nC = std::size_t(m.nC);
    const FvRansConfig& rc = cfg_.rans;
    const int nT = ransScalars(rc.model);
    const bool sst = ransSst(rc.model);
    for (std::size_t c = lo; c < hi; ++c) {
        double acc[2] = {0, 0};
        for (int q = m.cellStart[c]; q < m.cellStart[c + 1]; ++q) {
            const std::size_t f = std::size_t(m.cellFace[std::size_t(q)]);
            if (!geom_.active[f]) continue;
            const bool own = std::size_t(m.own[f]) == c;
            for (int e = 0; e < nT; ++e) {
                if (own) acc[e] += ev.Ft[std::size_t(e)][f];
                else acc[e] -= ev.Ft[std::size_t(e)][f];
            }
        }
        const double V = m.V[c];
        for (int e = 0; e < nT; ++e) {
            const std::size_t i = std::size_t(5 + e) * nC + c, q = std::size_t(e);
            const double R = -acc[e] / V + ev.tsrc[q][c];
            if (!st.a) {
                st.out[i] = R;
                continue;
            }
            const double comb = st.ca * st.a[i] + st.cb * st.b[i];
            double o = pointImplicit(comb, st.cr, R, ev.tsrc[2 + q][c], st.b[i]);
            if (sst) { // the structured guard, against this stage's output density
                const double floorV = e == 0 ? 0.0 : rc.omegaFloor * st.out[c];
                if (!(o >= floorV)) o = floorV;
            }
            st.out[i] = o;
        }
    }
}

void FvRef::sizeEval(FvEval& ev) const
{
    const FaceMesh& m = *fm_;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB), nF = std::size_t(m.nI) + nB;
    const std::size_t nY = std::size_t(cfg_.chem.nY);
    ev.Y.assign(nY, std::vector<double>(nC, 0.0));
    ev.bY.assign(nY, std::vector<double>(nB, 0.0));
    ev.gradY.assign(3 * nY, std::vector<double>(nC, 0.0));
    ev.Fy.assign(nY, std::vector<double>(nF, 0.0));
    ev.omega.assign(nY > 0 ? nC : 0, 0.0);
    for (auto& p : ev.prim) p.assign(nC, 0.0);
    const auto& gv = cfg_.gravity;
    if (gv[0] != 0.0 || gv[1] != 0.0 || gv[2] != 0.0)
        for (auto& v : ev.Fg) v.assign(std::size_t(m.nI), 0.0);
    ev.bnd.assign(nB, FvPrim<double>{});
    for (auto& p : ev.bphi) p.assign(nB, 0.0);
    for (auto& g : ev.grad) g.assign(nC, 0.0);
    for (auto& F : ev.F) F.assign(nF, 0.0);
    if (cfg_.les.cs > 0.0) ev.nut.assign(nC, 0.0);
    if (cfg_.shock.mode != ShockMode::None) ev.sigma.assign(nC, 0);
    if (cfg_.tfles) {
        for (auto* v : {&ev.fe, &ev.eof, &ev.om, &ev.uh, &ev.tfWork[0], &ev.tfWork[1], &ev.curl[0], &ev.curl[1], &ev.curl[2]})
            v->assign(nC, 0.0);
    }
    if (cfg_.rans.model == RansModel::None) return;
    for (auto& p : ev.tfld) p.assign(nC, 0.0);
    for (auto& p : ev.tbnd) p.assign(nB, 0.0);
    for (auto& p : ev.Ft) p.assign(nF, 0.0);
    for (auto& g : ev.tgrad) g.assign(nC, 0.0);
    for (auto& g : ev.tsrc) g.assign(nC, 0.0);
    ev.nut.assign(nC, 0.0);
    ev.f1.assign(nC, 0.0);
}

void FvRef::evaluate(const FvState& s, FvEval& ev, FvState& out, const FvState* src) const
{
    const FaceMesh& m = *fm_;
    assert(s.nV == nV_ && s.nC == m.nC && (!src || (src->nV == s.nV && src->nC == s.nC)));
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB), nF = std::size_t(m.nI) + nB;
    sizeEval(ev);
    if (out.nV != s.nV || out.nC != s.nC) out = FvState::zeros(s.nC, s.nV);
    const bool les = cfg_.les.cs > 0.0, thick = cfg_.tfles.has_value();
    const bool viscous = cfg_.tr.muRef > 0.0 || les, species = cfg_.chem.nY > 0, rans = cfg_.rans.model != RansModel::None;
    const bool sensed = cfg_.shock.mode != ShockMode::None, shockFlux = cfg_.shock.mode == ShockMode::Hll;
    Stage st;
    st.out = out.q.data();
    st.src = src ? src->q.data() : nullptr;
    onWorkers(workers_, [&](int w, int nw, auto&& sync) {
        auto [c0, c1] = part(nC, w, nw);
        auto [b0, b1] = part(nB, w, nw);
        auto [f0, f1] = part(nF, w, nw);
        prims(s.q.data(), ev, c0, c1);
        sync();
        if (sensed) {
            if (w == 0) shockSense(ev);
            sync();
        }
        bndStates(ev, b0, b1);
        sync();
        // The cell gradients first: mu_t and TFLES's coefficients read them, the species
        // pass reads E/F (independent passes, so the order moves no number).
        if (viscous) {
            grads(ev, c0, c1);
            sync();
        }
        if (les) {
            lesCells(ev, c0, c1);
            sync();
        }
        if (thick) tflesFields(s.q.data(), ev, c0, c1, sync);
        if (species) {
            speciesCells(s.q.data(), ev, c0, c1);
            sync();
            speciesBnd(ev, b0, b1);
            sync();
            speciesGrads(ev, c0, c1);
            sync();
        }
        if (rans) {
            ransFields(s.q.data(), ev, c0, c1);
            sync();
            ransBnd(ev, b0, b1);
            sync();
            ransGrads(ev, c0, c1);
            sync();
            ransCells(ev, c0, c1);
            sync();
        }
        faces(ev, f0, f1);
        sync();
        if (shockFlux) {
            shockFaces(s.q.data(), ev, f0, f1);
            sync();
        }
        if (species) {
            speciesFaces(ev, f0, f1);
            sync();
        }
        gather(ev, st, c0, c1);
        if (!rans) return;
        sync();
        ransFaces(ev, f0, f1);
        sync();
        ransGather(ev, st, c0, c1);
    });
}

void FvRef::rhs(const FvState& s, FvState& out, const FvState* src) const
{
    FvEval ev;
    evaluate(s, ev, out, src);
}

void FvRef::march(FvState& s, double dt, int steps, const FvState* src, FvLedger* ledger) const
{
    const FaceMesh& m = *fm_;
    assert(s.nV == nV_ && s.nC == m.nC && (!src || (src->nV == s.nV && src->nC == s.nC)));
    const int nVl = kFvMeanVars + cfg_.chem.nY, nP = int(m.patches.size());
    assert(!ledger || (ledger->nV == nVl && ledger->nP == nP && !src && cfg_.rans.model == RansModel::None &&
                       std::ranges::all_of(cfg_.gravity, [](double g) { return g == 0.0; }) &&
                       std::ranges::all_of(cfg_.bodyForce, [](double g) { return g == 0.0; })));
    // The ledger's stage sums: [stage][patch][row], sum V omega per stage.
    std::vector<double> ledS(ledger ? 3 * std::size_t(nP) * std::size_t(nVl) : 0);
    double ledOm[3] = {0, 0, 0};
    const auto ledgerStage = [&](const FvEval& ev, int k) {
        double* S = ledS.data() + std::size_t(k) * std::size_t(nP) * std::size_t(nVl);
        for (int p = 0; p < nP; ++p) {
            const MeshPatch& mp = m.patches[std::size_t(p)];
            for (int v = 0; v < nVl; ++v) {
                const std::vector<double>& F = v < kFvMeanVars ? ev.F[std::size_t(v)] : ev.Fy[std::size_t(v - kFvMeanVars)];
                double a = 0.0;
                for (int b = mp.first; b < mp.first + mp.count; ++b) {
                    const std::size_t f = std::size_t(m.nI + b);
                    if (geom_.active[f]) a += F[f];
                }
                S[std::size_t(p) * std::size_t(nVl) + std::size_t(v)] = a;
            }
        }
        double o = 0.0;
        if (cfg_.chem.nY > 0)
            for (std::size_t c = 0; c < std::size_t(m.nC); ++c) o += m.V[c] * ev.omega[c];
        ledOm[k] = o;
    };
    // The R2 weights: the cr each stage write used (ca + cb = 1 exactly).
    const double w01 = (2.0 / 3.0) * (0.25 * dt), w2 = (2.0 / 3.0) * dt;
    const auto ledgerStep = [&] {
        const std::size_t n = std::size_t(nP) * std::size_t(nVl);
        for (std::size_t i = 0; i < n; ++i) ledger->I[i] += (w01 * ledS[i] + w01 * ledS[n + i]) + w2 * ledS[2 * n + i];
        ledger->omega += (w01 * ledOm[0] + w01 * ledOm[1]) + w2 * ledOm[2];
    };
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB), nF = std::size_t(m.nI) + nB;
    FvEval ev;
    sizeEval(ev);
    FvState q1 = FvState::zeros(s.nC, s.nV), q2 = FvState::zeros(s.nC, s.nV);
    const bool les = cfg_.les.cs > 0.0, thick = cfg_.tfles.has_value();
    const bool viscous = cfg_.tr.muRef > 0.0 || les, species = cfg_.chem.nY > 0, rans = cfg_.rans.model != RansModel::None;
    const bool sensed = cfg_.shock.mode != ShockMode::None, shockFlux = cfg_.shock.mode == ShockMode::Hll;
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
                // The sensor: of the step's INPUT state, held over its stages.
                if (sensed && k == 0) {
                    if (w == 0) shockSense(ev);
                    sync();
                }
                bndStates(ev, b0, b1);
                sync();
                if (viscous) {
                    grads(ev, c0, c1);
                    sync();
                }
                if (les) {
                    lesCells(ev, c0, c1);
                    sync();
                }
                // TFLES: F, E, Omega of the step's INPUT state, held over its stages.
                if (thick && k == 0) tflesFields(bs[k], ev, c0, c1, sync);
                if (species) {
                    speciesCells(bs[k], ev, c0, c1);
                    sync();
                    speciesBnd(ev, b0, b1);
                    sync();
                    speciesGrads(ev, c0, c1);
                    sync();
                }
                if (rans) {
                    ransFields(bs[k], ev, c0, c1);
                    sync();
                    ransBnd(ev, b0, b1);
                    sync();
                    ransGrads(ev, c0, c1);
                    sync();
                    ransCells(ev, c0, c1);
                    sync();
                }
                faces(ev, f0, f1);
                sync();
                if (shockFlux) {
                    shockFaces(bs[k], ev, f0, f1);
                    sync();
                }
                if (species) {
                    speciesFaces(ev, f0, f1);
                    sync();
                }
                // The ledger reads F and omega only (the gather writes the state): worker 0, before
                // its gather; the next stage's faces come after the gather's sync.
                if (ledger && w == 0) {
                    ledgerStage(ev, k);
                    if (k == 2) ledgerStep();
                }
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
                if (!rans) continue;
                ransFaces(ev, f0, f1);
                sync();
                ransGather(ev, st, c0, c1);
                sync();
            }
    });
}

double FvRef::chemDt(const FvState& s, double k) const
{
    assert(s.nV == nV() && s.nC == fm_->nC);
    const FvChem& ch = cfg_.chem;
    if (ch.nY == 0 || !std::isfinite(ch.lnA)) return HUGE_VAL;
    const Transport& tr = cfg_.tr;
    const double cv = tr.Rgas / (tr.gamma - 1.0);
    double J = 0.0;
    for (std::size_t c = 0; c < std::size_t(s.nC); ++c) {
        const FvPrim<double> p = fvPrimOf(s.var(0)[c], s.var(1)[c], s.var(2)[c], s.var(3)[c], s.var(4)[c], tr.gamma,
                                          tr.Rgas);
        const double Y = s.var(kFvMeanVars + ch.fuel)[c] / p.rho;
        if (ch.ox >= 0) {
            // ChemRate2Phi with the law: ln C in FP64 of the unfloored Y_F, Y_O.
            const double YO = s.var(kFvMeanVars + ch.ox)[c] / p.rho;
            const double r = std::exp(ch.lnA + (ch.law.n > 0 ? phiRateLn(ch.law, Y, YO) : 0.0) - ch.Ta / p.temp);
            const double yf = std::max(Y, 0.0), yo = std::max(YO, 0.0);
            J = std::max(J, r * p.rho * (yo + ch.stoich * yf + ch.q * yf * yo * ch.Ta / (cv * p.temp * p.temp)));
            continue;
        }
        const double r = std::exp(ch.lnA - ch.Ta / p.temp);
        J = std::max(J, r * (1.0 + ch.q * (Y > 0.0 ? Y : 0.0) * ch.Ta / (cv * p.temp * p.temp)));
    }
    return J > 0.0 ? k / J : HUGE_VAL;
}

std::expected<double, std::string> FvRef::stableDt(const FvState& s, FvDtLimits lim,
                                                   std::vector<double>* local) const
{
    const FaceMesh& m = *fm_;
    assert(s.nV == nV_ && s.nC == m.nC);
    const std::size_t nC = std::size_t(m.nC);
    const Transport& tr = cfg_.tr;
    if (local) local->assign(nC, 0.0);
    const bool les = cfg_.les.cs > 0.0, thick = cfg_.tfles.has_value();
    const bool rans = cfg_.rans.model != RansModel::None;
    FvEval ev;
    if (rans || les || thick) {
        FvState o;
        evaluate(s, ev, o);
    }
    const double fMol = fvDiffusivityFactor(tr.gamma, tr.Pr),
                 fEddy = fvDiffusivityFactor(tr.gamma, les ? cfg_.les.prt : kPrandtlTurb);
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
                if (f >= std::size_t(m.nI) && cfg_.bc[std::size_t(bcOf_[f - std::size_t(m.nI)])].wedge) continue;
                const double Sx = m.S[0][f], Sy = m.S[1][f], Sz = m.S[2][f];
                sum += std::abs(p.u[0] * Sx + p.u[1] * Sy + p.u[2] * Sz) + cs * std::sqrt(Sx * Sx + Sy * Sy + Sz * Sz);
            }
            double dt = 2.0 * lim.cfl * m.V[c] / sum;
            if (tr.muRef > 0.0 || les) {
                double mu, kappa;
                fvTransportAt(tr, p.temp, mu, kappa);
                // TFLES: the thickened energy and species carry FE gamma/Pr on mu.
                double den = mu * (thick ? std::max(4.0 / 3.0, ev.fe[c] * (tr.gamma / tr.Pr)) : fMol);
                if (rans || les) den += ev.nut[c] * fEddy;
                if (cfg_.rans.model == RansModel::SaNeg)
                    den = std::max(den, p.rho * saDtDiffusivity(mu / p.rho, ev.tfld[0][c]) / sa::sigma);
                dt = std::min(dt, lim.diff * p.rho * m.V[c] / (den * diffCoef_[c]));
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
