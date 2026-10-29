#include "dns/fvgrad.hpp"

#include "dns/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>

namespace dns {
namespace {

using V3 = std::array<double, 3>;

double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

V3 at(const std::array<std::vector<double>, 3>& a, std::size_t i) { return {a[0][i], a[1][i], a[2][i]}; }

void put(std::array<std::vector<double>, 3>& a, std::size_t i, const V3& v)
{
    for (int k = 0; k < 3; ++k) a[std::size_t(k)][i] = v[std::size_t(k)];
}

// Per-block first error, merged lowest block first: the serial loop's error
// for any worker count.
template <class Body>
std::expected<void, std::string> sweep(int threads, std::size_t n, Body&& body)
{
    const int nw = workerCount(threads, n);
    std::vector<std::optional<std::string>> err(static_cast<std::size_t>(nw));
    forBlocks(nw, n, [&](int w, std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi && !err[std::size_t(w)]; ++i)
            if (auto e = body(i); !e) err[std::size_t(w)] = std::move(e.error());
    });
    for (auto& e : err)
        if (e) return std::unexpected(std::move(*e));
    return {};
}

// Inverse of the symmetric least-squares matrix over the solution axes
// (emptyAxis row and column stay 0). Returns false when singular.
bool invertLsq(const double G[3][3], int emptyAxis, double out[3][3])
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out[r][c] = 0.0;
    const double tr = G[0][0] + G[1][1] + G[2][2];
    if (emptyAxis >= 0) {
        const int a = emptyAxis == 0 ? 1 : 0, b = emptyAxis == 2 ? 1 : 2;
        const double det = G[a][a] * G[b][b] - G[a][b] * G[b][a];
        if (!(std::abs(det) > 1e-12 * tr * tr)) return false;
        out[a][a] = G[b][b] / det;
        out[b][b] = G[a][a] / det;
        out[a][b] = -G[a][b] / det;
        out[b][a] = -G[b][a] / det;
        return true;
    }
    const double c00 = G[1][1] * G[2][2] - G[1][2] * G[2][1];
    const double c01 = G[1][2] * G[2][0] - G[1][0] * G[2][2];
    const double c02 = G[1][0] * G[2][1] - G[1][1] * G[2][0];
    const double det = G[0][0] * c00 + G[0][1] * c01 + G[0][2] * c02;
    if (!(std::abs(det) > 1e-12 * tr * tr * tr)) return false;
    out[0][0] = c00 / det;
    out[1][0] = c01 / det;
    out[2][0] = c02 / det;
    out[0][1] = (G[0][2] * G[2][1] - G[0][1] * G[2][2]) / det;
    out[1][1] = (G[0][0] * G[2][2] - G[0][2] * G[2][0]) / det;
    out[2][1] = (G[0][1] * G[2][0] - G[0][0] * G[2][1]) / det;
    out[0][2] = (G[0][1] * G[1][2] - G[0][2] * G[1][1]) / det;
    out[1][2] = (G[0][2] * G[1][0] - G[0][0] * G[1][2]) / det;
    out[2][2] = (G[0][0] * G[1][1] - G[0][1] * G[1][0]) / det;
    return true;
}

V3 apply(const double M[3][3], const V3& v)
{
    return {M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
            M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
            M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]};
}

} // namespace

std::expected<FvFaceGeom, std::string> fvFaceGeom(const FaceMesh& fm, FvFaceGradRule rule, double alpha,
                                                  int threads)
{
    if (!(alpha > 0.0) || !std::isfinite(alpha))
        return std::unexpected(std::format("face-gradient alpha must be positive, got {}", alpha));
    const std::size_t nI = std::size_t(fm.nI), nF = nI + std::size_t(fm.nB);
    FvFaceGeom g;
    g.rule = rule;
    g.alpha = alpha;
    for (int k = 0; k < 3; ++k) {
        g.rOwn[std::size_t(k)].assign(nF, 0.0);
        g.corr[std::size_t(k)].assign(nF, 0.0);
    }
    g.active.assign(nF, 1);
    for (const MeshPatch& p : fm.patches)
        if (p.type == FoamPatchType::Empty)
            std::fill_n(g.active.begin() + std::ptrdiff_t(nI) + p.first, p.count, std::uint8_t(0));

    auto ok = sweep(threads, nF, [&](std::size_t f) -> std::expected<void, std::string> {
        const V3 Cf = at(fm.Cf, f), d = at(fm.d, f), S = at(fm.S, f);
        const V3 Co = at(fm.C, std::size_t(fm.own[f]));
        V3 r = {Cf[0] - Co[0], Cf[1] - Co[1], Cf[2] - Co[2]};
        if (f < nI) {
            // Cyclic face with Cf in the neighbour's frame: translate by the
            // image offset T = C_own + d - C_nei (0 to rounding elsewhere).
            const V3 Cn = at(fm.C, std::size_t(fm.nei[f]));
            const V3 T = {Co[0] + d[0] - Cn[0], Co[1] + d[1] - Cn[1], Co[2] + d[2] - Cn[2]};
            if (dot(T, T) > 1e-16 * dot(d, d)) {
                const V3 rt = {r[0] + T[0], r[1] + T[1], r[2] + T[2]};
                if (dot(rt, rt) < dot(r, r)) r = rt;
            }
        }
        put(g.rOwn, f, r);
        if (!g.active[f]) return {};
        const double Sd = dot(S, d);
        if (!(Sd > 0.0))
            return std::unexpected(std::format("FV face geometry: S.d = {} <= 0 on face {} (source face {})",
                                               Sd, f, fm.srcFace[f]));
        V3 c;
        if (rule == FvFaceGradRule::AlongD) {
            const double s = alpha / dot(d, d);
            c = {s * d[0], s * d[1], s * d[2]};
        } else {
            const double s = alpha / Sd;
            c = {s * S[0], s * S[1], s * S[2]};
        }
        put(g.corr, f, c);
        return {};
    });
    if (!ok) return std::unexpected(std::move(ok.error()));
    return g;
}

std::expected<FvGradWeights, std::string> fvGradWeights(const FaceMesh& fm, const FvFaceGeom& geom,
                                                        FvGradScheme scheme, int threads)
{
    const std::size_t nC = std::size_t(fm.nC), nI = std::size_t(fm.nI), nF = nI + std::size_t(fm.nB);
    FvGradWeights w;
    w.scheme = scheme;
    for (int k = 0; k < 3; ++k) {
        w.own[std::size_t(k)].assign(nF, 0.0);
        w.nei[std::size_t(k)].assign(nI, 0.0);
    }
    if (scheme == FvGradScheme::GreenGauss) {
        const int nw = workerCount(threads, nF);
        forBlocks(nw, nF, [&](int, std::size_t lo, std::size_t hi) {
            for (std::size_t f = lo; f < hi; ++f) {
                if (!geom.active[f]) continue;
                const V3 S = at(fm.S, f);
                const double Vo = fm.V[std::size_t(fm.own[f])];
                if (f >= nI) {
                    put(w.own, f, {S[0] / Vo, S[1] / Vo, S[2] / Vo});
                    continue;
                }
                // OpenFOAM surfaceInterpolation::makeWeights, in the owner's frame.
                const V3 r = at(geom.rOwn, f), d = at(fm.d, f);
                const double sfdOwn = std::abs(dot(S, r));
                const double sfdNei = std::abs(dot(S, V3{d[0] - r[0], d[1] - r[1], d[2] - r[2]}));
                const double lambda = sfdNei / (sfdOwn + sfdNei);
                const double Vn = fm.V[std::size_t(fm.nei[f])];
                const double so = (1.0 - lambda) / Vo, sn = -lambda / Vn;
                put(w.own, f, {so * S[0], so * S[1], so * S[2]});
                put(w.nei, f, {sn * S[0], sn * S[1], sn * S[2]});
            }
        });
        return w;
    }

    // Least squares: G_c^-1 per cell, then the per-face weights.
    std::vector<std::array<double, 9>> Ginv(nC);
    auto ok = sweep(threads, nC, [&](std::size_t c) -> std::expected<void, std::string> {
        double G[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (int q = fm.cellStart[c]; q < fm.cellStart[c + 1]; ++q) {
            const std::size_t f = std::size_t(fm.cellFace[std::size_t(q)]);
            if (!geom.active[f]) continue;
            const V3 d = at(fm.d, f);
            const double iw = 1.0 / dot(d, d);
            for (int r = 0; r < 3; ++r)
                for (int s = 0; s < 3; ++s) G[r][s] += iw * d[std::size_t(r)] * d[std::size_t(s)];
        }
        double inv[3][3];
        if (!invertLsq(G, fm.emptyAxis, inv))
            return std::unexpected(std::format(
                "least-squares gradient matrix singular at cell {} (source cell {}): its faces do not span "
                "the solution space",
                c, fm.srcCell[c]));
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s) Ginv[c][std::size_t(r * 3 + s)] = inv[r][s];
        return {};
    });
    if (!ok) return std::unexpected(std::move(ok.error()));
    const int nw = workerCount(threads, nF);
    forBlocks(nw, nF, [&](int, std::size_t lo, std::size_t hi) {
        auto mat = [&](int c, double M[3][3]) {
            for (int r = 0; r < 3; ++r)
                for (int s = 0; s < 3; ++s) M[r][s] = Ginv[std::size_t(c)][std::size_t(r * 3 + s)];
        };
        for (std::size_t f = lo; f < hi; ++f) {
            if (!geom.active[f]) continue;
            const V3 d = at(fm.d, f);
            const double iw = 1.0 / dot(d, d);
            double M[3][3];
            mat(fm.own[f], M);
            const V3 wo = apply(M, {iw * d[0], iw * d[1], iw * d[2]});
            put(w.own, f, wo);
            if (f >= nI) continue;
            mat(fm.nei[f], M);
            put(w.nei, f, apply(M, {-iw * d[0], -iw * d[1], -iw * d[2]}));
        }
    });
    return w;
}

void fvCellGradient(const FaceMesh& fm, const FvFaceGeom& geom, const FvGradWeights& w,
                    std::span<const double> phi, std::span<const double> phiB,
                    const std::array<double*, 3>& g, std::size_t lo, std::size_t hi)
{
    const int nI = fm.nI;
    for (std::size_t c = lo; c < hi; ++c) {
        double gx = 0.0, gy = 0.0, gz = 0.0;
        const double pc = phi[c];
        for (int q = fm.cellStart[c]; q < fm.cellStart[c + 1]; ++q) {
            const int f = fm.cellFace[std::size_t(q)];
            const std::size_t fs = std::size_t(f);
            double dphi;
            const std::array<std::vector<double>, 3>* wt = &w.own;
            if (f < nI) {
                if (std::size_t(fm.own[fs]) == c) {
                    dphi = phi[std::size_t(fm.nei[fs])] - pc;
                } else {
                    dphi = phi[std::size_t(fm.own[fs])] - pc;
                    wt = &w.nei;
                }
            } else {
                if (!geom.active[fs]) continue;
                dphi = phiB[fs - std::size_t(nI)] - pc;
            }
            gx += (*wt)[0][fs] * dphi;
            gy += (*wt)[1][fs] * dphi;
            gz += (*wt)[2][fs] * dphi;
        }
        g[0][c] = gx;
        g[1][c] = gy;
        g[2][c] = gz;
    }
}

} // namespace dns
