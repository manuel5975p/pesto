#include "dns/gpu/fv_comp.hpp"

#include "dns/chem.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvrans.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/internal/ckpt_words.hpp"
#include "dns/gpu/internal/reduce.hpp"
#include "dns/gpu/internal/stage_write.hpp"
#include "dns/gpu/reductions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <optional>
#include <type_traits>
#include <utility>

namespace dns::gpu {

namespace {

constexpr int kT = 256;
unsigned blocksOf(std::size_t n) { return unsigned((n + kT - 1) / kT); }

// Row widths: primitives rho u v w e p T (FvRef's prim order), four gradients
// of three, the mean-flow flux, the tally row (fp, fv, q).
constexpr int kPrim = 7, kGrad = 12, kTally = 7;
constexpr int kFieldRow[4] = {1, 2, 3, 6}; // u, v, w, T in a primitive row

template <class R>
struct FaceRow
{
    int own, nei;
    R S[3], d[3], corr[3];
};

// bc: the row of the FvBc table (one per patch today; per-face data = one per face).
template <class R>
struct BndRow
{
    int own, bc;
    R S[3], Cf[3], d[3], corr[3];
};

// One CSR entry of the gradient: other >= 0 the neighbour cell, else ~b.
template <class R>
struct GradEnt
{
    int other;
    R w[3];
};

template <class R>
struct View
{
    int nC, nI, nB, nV;
    double gamma, Rgas;
    const FaceRow<R>* face;
    const BndRow<R>* bnd;
    const FvBc* bcs;
    const R* V;
    const int* cStart;
    const int* gFace; // f << 1 | (neighbour side), f compact (active faces)
    const GradEnt<R>* gEnt;
};

// Molecular transport: constant-law values folded host-side (fvTransportAt's
// result for any Tf), so a Sutherland-free face reads two scalars.
template <class R>
struct Gas
{
    Transport tr;
    R mu, kappa, kapEddy;
};

template <class R>
__device__ inline FvPrim<R> primOf(const R* P, int c)
{
    const R* p = P + std::size_t(kPrim) * std::size_t(c);
    return {p[0], {p[1], p[2], p[3]}, p[4], p[5], p[6]};
}

template <class R>
__device__ inline void transportAt(const Gas<R>& g, R Tf, R& mu, R& kappa)
{
    if (g.tr.sutherland) {
        fvTransportAt(g.tr, Tf, mu, kappa);
        return;
    }
    mu = g.mu;
    kappa = g.kappa;
}

template <class R>
__global__ void kFvPrim(View<R> m, const R* q, R* P)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    const FvPrim<R> p = fvPrimOf(q[cc], q[nC + cc], q[2 * nC + cc], q[3 * nC + cc], q[4 * nC + cc], m.gamma, m.Rgas);
    R* o = P + std::size_t(kPrim) * cc;
    o[0] = p.rho;
    o[1] = p.u[0];
    o[2] = p.u[1];
    o[3] = p.u[2];
    o[4] = p.e;
    o[5] = p.p;
    o[6] = p.temp;
}

// Boundary states: FvRef::bndStates.
template <class R>
__global__ void kFvBnd(View<R> m, const R* P, R* B)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const BndRow<R>& f = m.bnd[b];
    const FvPrim<R> A = primOf(P, f.own);
    const FvPrim<R> s = fvBoundaryState(m.bcs[f.bc], A, f.S, f.Cf, m.gamma, m.Rgas);
    R* o = B + std::size_t(kPrim) * std::size_t(b);
    o[0] = s.rho;
    o[1] = s.u[0];
    o[2] = s.u[1];
    o[3] = s.u[2];
    o[4] = s.e;
    o[5] = s.p;
    o[6] = s.temp;
}

// Cell gradients of u, v, w, T: fvCellGradient's sum, CSR order.
template <class R>
__global__ void __launch_bounds__(kT) kFvGrad(View<R> m, const R* P, const R* B, R* G)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const R* pc = P + std::size_t(kPrim) * std::size_t(c);
    R g[kGrad];
    R phiC[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) phiC[k] = pc[kFieldRow[k]];
#pragma unroll
    for (int k = 0; k < kGrad; ++k) g[k] = R(0.0);
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const GradEnt<R>& n = m.gEnt[e];
        const R* po = n.other >= 0 ? P + std::size_t(kPrim) * std::size_t(n.other)
                                   : B + std::size_t(kPrim) * std::size_t(~n.other);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const R dphi = po[kFieldRow[k]] - phiC[k];
            g[3 * k + 0] += n.w[0] * dphi;
            g[3 * k + 1] += n.w[1] * dphi;
            g[3 * k + 2] += n.w[2] * dphi;
        }
    }
    R* o = G + std::size_t(kGrad) * std::size_t(c);
#pragma unroll
    for (int k = 0; k < kGrad; ++k) o[k] = g[k];
}

// mu_t = rho (C Delta)^2 op(grad u): kCompSmagNutCurv's formula on the FV
// gradient. cd[c] = C^2 Delta^2.
template <class R>
__global__ void kFvNut(int nC, int sigma, const R* cd, const R* P, const R* G, R* nut)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    const R* g = G + std::size_t(kGrad) * std::size_t(c);
    R gd[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) gd[i][j] = g[3 * i + j];
    nut[c] = P[std::size_t(kPrim) * std::size_t(c)] * cd[c] * sgsRate(sigma, gd);
}

template <class R>
__device__ inline void cellGrad(const R* G, int c, int k, R (&g)[3])
{
    const R* p = G + std::size_t(kGrad) * std::size_t(c) + std::size_t(3 * k);
    g[0] = p[0];
    g[1] = p[1];
    g[2] = p[2];
}

// Internal face fluxes: FvRef::faces, plus the eddy part on the face mean of
// mu_t. Visc: gradients exist (mu > 0 or LES).
template <class R, bool Visc, bool Les>
__global__ void __launch_bounds__(kT) kFvFace(View<R> m, Gas<R> gas, const R* P, const R* G, const R* nut, R* F)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const FaceRow<R>& r = m.face[f];
    const FvPrim<R> A = primOf(P, r.own), B = primOf(P, r.nei);
    R gu[3][3] = {{R(0.0), R(0.0), R(0.0)}, {R(0.0), R(0.0), R(0.0)}, {R(0.0), R(0.0), R(0.0)}};
    R gT[3] = {R(0.0), R(0.0), R(0.0)};
    R mu = R(0.0), kappa = R(0.0);
    if constexpr (Visc) {
        R ga[3], gb[3];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            cellGrad(G, r.own, k, ga);
            cellGrad(G, r.nei, k, gb);
            fvFaceGrad(ga, gb, A.u[k], B.u[k], r.d, r.corr, gu[k]);
        }
        cellGrad(G, r.own, 3, ga);
        cellGrad(G, r.nei, 3, gb);
        fvFaceGrad(ga, gb, A.temp, B.temp, r.d, r.corr, gT);
        transportAt(gas, R(0.5) * (A.temp + B.temp), mu, kappa);
    }
    if constexpr (Les) {
        const R nf = R(0.5) * (nut[r.own] + nut[r.nei]);
        mu += nf;
        kappa += nf * gas.kapEddy;
    }
    const FvFlux<R> o = fvMeanFlux(A, B, r.S, gu, gT, mu, kappa);
    R* w = F + std::size_t(m.nV) * std::size_t(f);
    w[0] = o.rho;
    w[1] = o.m[0];
    w[2] = o.m[1];
    w[3] = o.m[2];
    w[4] = o.E;
}

// The Wall / Slip face gradients and transport of FvRef::faces. Les: a Slip
// face takes the owner's mu_t (the even image); a Wall face the molecular
// stress only (mu_t is odd there, 0 on the face).
template <class R, bool Les>
__device__ inline void bndViscous(const BndRow<R>& r, const FvBc& bc, const Gas<R>& gas, const FvPrim<R>& A,
                                  const FvPrim<R>& B, const R* G, const R* nut, R (&gu)[3][3], R (&gT)[3],
                                  R& mu, R& kappa)
{
    R ga[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        cellGrad(G, r.own, k, ga);
        fvFaceGradBnd(ga, A.u[k], B.u[k], r.d, r.corr, gu[k]);
    }
    cellGrad(G, r.own, 3, ga);
    fvFaceGradBnd(ga, A.temp, B.temp, r.d, r.corr, gT);
    const bool iso = (bc.kind == FvBcKind::Wall && bc.isothermal) || bc.diffusive;
    transportAt(gas, iso ? R(0.5) * (A.temp + B.temp) : A.temp, mu, kappa);
    if constexpr (Les)
        if (bc.kind == FvBcKind::Slip) {
            mu += nut[r.own];
            kappa += nut[r.own] * gas.kapEddy;
        }
}

template <class R, bool Visc, bool Les>
__global__ void __launch_bounds__(kT) kFvBndFace(View<R> m, Gas<R> gas, const R* P, const R* Bs, const R* G,
                                                 const R* nut, R* F)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const BndRow<R>& r = m.bnd[b];
    const FvBc& bc = m.bcs[r.bc];
    const FvPrim<R> A = primOf(P, r.own), B = primOf(Bs, b);
    R gu[3][3] = {{R(0.0), R(0.0), R(0.0)}, {R(0.0), R(0.0), R(0.0)}, {R(0.0), R(0.0), R(0.0)}};
    R gT[3] = {R(0.0), R(0.0), R(0.0)};
    R mu = R(0.0), kappa = R(0.0);
    if constexpr (Visc)
        if (bc.kind == FvBcKind::Wall || bc.kind == FvBcKind::Slip || bc.diffusive)
            bndViscous<R, Les>(r, bc, gas, A, B, G, nut, gu, gT, mu, kappa);
    const FvFlux<R> o = fvBoundaryFlux(bc, A, B, r.S, gu, gT, mu, kappa);
    R* w = F + std::size_t(m.nV) * (std::size_t(m.nI) + std::size_t(b));
    w[0] = o.rho;
    w[1] = o.m[0];
    w[2] = o.m[1];
    w[3] = o.m[2];
    w[4] = o.E;
}

// The gather fused with the stage: ascending CSR faces, own index only. Comp:
// the compensated two-word state (R = float: comp.hpp's Word2; R = double: the
// double-double Word2d of gpu/word2d.hpp), hi + lo accumulates what the plain
// expression rounds away (gpu/internal/stage_write.hpp, a compile-time choice).
// Rows First .. First + Count: the mean rows, or the species rows through this same
// source (one block of stage writes: another shape fuses cb X + cr R into other
// FMAs, and a uniform Y would walk off rho's compensated bits, S3).
template <class R, bool Comp, bool Lts, int First = 0, int Count = kFvMeanVars>
__global__ void __launch_bounds__(kT) kFvGather(View<R> m, const R* F, const R* a, const R* al, const R* b,
                                                const R* bl, R* out, R* ol, R ca, R cb, R cr0, const R* lts)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R acc[Count];
#pragma unroll
    for (int v = 0; v < Count; ++v) acc[v] = R(0.0);
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.gFace[e];
        const R* w = F + std::size_t(m.nV) * std::size_t(code >> 1) + First;
        if (code & 1) {
#pragma unroll
            for (int v = 0; v < Count; ++v) acc[v] -= w[v];
        } else {
#pragma unroll
            for (int v = 0; v < Count; ++v) acc[v] += w[v];
        }
    }
    const R Vc = m.V[c];
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    R cr = cr0;
    if constexpr (Lts) cr *= lts[c];
#pragma unroll
    for (int v = 0; v < Count; ++v) {
        const std::size_t i = std::size_t(v + First) * nC + cc;
        const R rr = -acc[v] / Vc;
        if constexpr (Comp) {
            const Word2Of<R> w2 = internal::combineWord2(cb, a[i], al[i], b[i], bl[i], cr * rr);
            out[i] = w2.hi;
            ol[i] = w2.lo;
        } else {
            out[i] = ca * a[i] + cb * b[i] + cr * rr;
        }
    }
}

// Gravity's face terms [4 per internal face], after the fluxes: fvGravityFace.
template <class R>
__global__ void __launch_bounds__(kT) kFvGravFace(View<R> m, double gx, double gy, double gz, const R* P, const R* F, R* Gf)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const FaceRow<R>& r = m.face[f];
    const double g[3] = {gx, gy, gz};
    fvGravityFace(fvHydroRho(primOf(P, r.own), primOf(P, r.nei), m.Rgas), F[std::size_t(m.nV) * std::size_t(f)], g, r.d, r.S,
                  Gf + std::size_t(4) * std::size_t(f));
}

// The term on the stage output, ascending CSR faces, both sides add. Comp: the
// two-word write.
template <class R, bool Comp>
__global__ void __launch_bounds__(kT) kFvGravity(View<R> m, R cr, const R* Gf, R* out, R* ol)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R acc[4] = {R(0.0), R(0.0), R(0.0), R(0.0)};
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int f = m.gFace[e] >> 1;
        if (f >= m.nI) continue;
        const R* w = Gf + std::size_t(4) * std::size_t(f);
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[k] += w[k];
    }
    const R Vc = m.V[c];
    const std::size_t n = std::size_t(m.nC), cc = std::size_t(c);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const std::size_t i = std::size_t(k + 1) * n + cc;
        const R inc = cr * (acc[k] / Vc);
        if constexpr (Comp) {
            const Word2Of<R> w = addWord2(out[i], ol[i], inc);
            out[i] = w.hi;
            ol[i] = w.lo;
        } else {
            out[i] += inc;
        }
    }
}

// The sensor of the state P was formed from: fvShockSense, whole lines only.
template <class R>
__global__ void __launch_bounds__(kT) kFvShockSense(int nC, const int* lineStart, const int* line, const R* P, std::uint8_t* sb)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    const R pc = P[std::size_t(kPrim) * std::size_t(c) + 5];
    R v = R(0.0);
    for (int l = lineStart[c]; l < lineStart[c + 1]; ++l) {
        const R jump = shockJump(P[std::size_t(kPrim) * std::size_t(line[2 * l]) + 5], pc, P[std::size_t(kPrim) * std::size_t(line[2 * l + 1]) + 5]);
        v = jump > v || jump != jump ? jump : v; // a NaN sticks
    }
    sb[c] = shockByte(v);
}

// Test only: sigma 1 on every cell.
__global__ void kFvShockForce(int nC, std::uint8_t* sb)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    sb[c] = 255;
}

// FvRef::shockFaces: F += |S| D + eps (1 - sigma) |u_f.S| D3, one write per face, so
// conservation holds by construction. s: the state P was formed from.
template <class R>
__global__ void __launch_bounds__(kT) kFvShockFace(View<R> m, double gamma, R eps, const int* fLine, const std::uint8_t* sb,
                                                   const R* s, const R* P, R* F)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const FaceRow<R>& r = m.face[f];
    const int pm = fLine[2 * f], np = fLine[2 * f + 1];
    int sg = sb[r.own] > sb[r.nei] ? sb[r.own] : sb[r.nei];
    if (pm >= 0 && sb[pm] > sg) sg = sb[pm];
    if (np >= 0 && sb[np] > sg) sg = sb[np];
    if (sg == 0 && eps == R(0.0)) return;
    const FvPrim<R> A = primOf(P, r.own), B = primOf(P, r.nei);
    const R sigma = R(sg) / R(255.0);
    R* w = F + std::size_t(m.nV) * std::size_t(f);
    if (sg > 0) {
        R d[5];
        fvShockFlux(sigma, A, B, r.S, gamma, d);
#pragma unroll
        for (int k = 0; k < 5; ++k) w[k] += d[k];
    }
    if (pm < 0 || np < 0 || sg == 255 || eps == R(0.0)) return;
    const R l = fvD4Rate(eps, sigma, A, B, r.S);
    const std::size_t nC = std::size_t(m.nC);
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const std::size_t o = std::size_t(k) * nC;
        const R a = l * fvD4Row(s[o + std::size_t(pm)], s[o + std::size_t(r.own)], s[o + std::size_t(r.nei)], s[o + std::size_t(np)]);
        if (a != R(0.0)) w[k] += a; // a uniform row keeps F's bytes, -0.0 included
    }
}

// Per-cell dt (FvRef::stableDt's bound); a non-positive or non-finite rho, p
// writes -1. Les: mu_t adds at its own factor. Rans: 1 SA (its scalar's bound on
// the first field), 2 SST; a non-finite mu_t writes -1.
template <class R, bool Les, int Rans, bool Thick = false>
__global__ void __launch_bounds__(kT) kFvDt(View<R> m, Gas<R> gas, double cfl, double diff, R fMol, R fEddy,
                                            const R* P, const R* nut, const R* Ph, const R* entS, const R* diffC,
                                            R* dtc, const R* fe = nullptr, R gPr = R(0.0))
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const FvPrim<R> p = primOf(P, c);
    if (!(p.rho > R(0.0)) || !(p.p > R(0.0)) || !isfinite(p.rho) || !isfinite(p.p)) {
        dtc[c] = R(-1.0);
        return;
    }
    if constexpr (Rans != 0)
        if (!isfinite(nut[c])) {
            dtc[c] = R(-1.0);
            return;
        }
    const R cs = sqrt(R(m.gamma) * p.p / p.rho);
    R sum = R(0.0);
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const R* s = entS + 3 * std::size_t(e);
        sum += fabs(p.u[0] * s[0] + p.u[1] * s[1] + p.u[2] * s[2]) + cs * sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    }
    R dt = R(2.0 * cfl) * m.V[c] / sum;
    if (gas.tr.muRef > 0.0 || Les) {
        R mu, kappa;
        transportAt(gas, p.temp, mu, kappa);
        R den = mu * fMol;
        if constexpr (Thick) { // the thickened energy and species: FE gamma/Pr on mu
            const R a = fe[c] * gPr, b = R(4.0 / 3.0);
            den = mu * (a > b ? a : b);
        }
        if constexpr (Les) den += nut[c] * fEddy;
        if constexpr (Rans == 1) {
            const R ds = p.rho * saDtDiffusivity(mu / p.rho, Ph[std::size_t(kFvRansFields) * std::size_t(c)]) /
                         R(sa::sigma);
            if (ds > den) den = ds;
        }
        if (den > R(0.0)) dt = fmin(dt, R(diff) * p.rho * m.V[c] / (den * diffC[c]));
    }
    dtc[c] = dt;
}

// --- RANS (dns/fvrans.hpp): FvRef's rans* passes. --------------------------

constexpr int kRansSrc = 4, kRansGrad = 3 * kFvRansFields; // R0 R1 J0 J1; two gradients

// The gradient fields of the state s (FvRef::ransFields). Sst: k, omega; SA: nu~, rho.
template <class R, bool Sst>
__global__ void kFvRansPhi(View<R> m, const R* s, const R* P, R* Ph)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    const R rho = P[std::size_t(kPrim) * cc];
    R* o = Ph + std::size_t(kFvRansFields) * cc;
    o[0] = s[5 * nC + cc] / rho;
    if constexpr (Sst) o[1] = s[6 * nC + cc] / rho;
    else o[1] = rho;
}

// Their values on the boundary faces (FvRef::ransBnd).
template <class R, bool Sst>
__global__ void __launch_bounds__(kT) kFvRansBnd(View<R> m, Gas<R> gas, R inf0, R inf1, const R* P, const R* Bs,
                                                 const R* Ph, const R* dw, R* Bh)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const BndRow<R>& r = m.bnd[b];
    const FvBc& bc = m.bcs[r.bc];
    const FvPrim<R> A = primOf(P, r.own), B = primOf(Bs, b);
    R mu = R(0.0), kappa = R(0.0);
    if (bc.kind == FvBcKind::Wall) transportAt(gas, fvWallTemp(bc, A, B), mu, kappa);
    const R pA[2] = {Ph[std::size_t(kFvRansFields) * std::size_t(r.own)],
                     Ph[std::size_t(kFvRansFields) * std::size_t(r.own) + 1]};
    const R inf[2] = {inf0, inf1};
    R v[2];
    fvRansBndFields<R, R>(Sst, bc, B, r.S, pA, inf, mu, dw[r.own], m.Rgas, v);
    R* o = Bh + std::size_t(kFvRansFields) * std::size_t(b);
    o[0] = v[0];
    o[1] = v[1];
}

// Cell gradients of the two fields: kFvGrad's sum.
template <class R>
__global__ void __launch_bounds__(kT) kFvRansGrad(View<R> m, const R* Ph, const R* Bh, R* Gh)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const R* pc = Ph + std::size_t(kFvRansFields) * std::size_t(c);
    R g[kRansGrad];
#pragma unroll
    for (int k = 0; k < kRansGrad; ++k) g[k] = R(0.0);
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const GradEnt<R>& n = m.gEnt[e];
        const R* po = n.other >= 0 ? Ph + std::size_t(kFvRansFields) * std::size_t(n.other)
                                   : Bh + std::size_t(kFvRansFields) * std::size_t(~n.other);
#pragma unroll
        for (int k = 0; k < kFvRansFields; ++k) {
            const R dphi = po[k] - pc[k];
            g[3 * k + 0] += n.w[0] * dphi;
            g[3 * k + 1] += n.w[1] * dphi;
            g[3 * k + 2] += n.w[2] * dphi;
        }
    }
    R* o = Gh + std::size_t(kRansGrad) * std::size_t(c);
#pragma unroll
    for (int k = 0; k < kRansGrad; ++k) o[k] = g[k];
}

// mu_t, F1 and the pointwise terms (FvRef::ransCells).
template <class R, bool Sst>
__global__ void __launch_bounds__(kT) kFvRansCell(View<R> m, Gas<R> gas, RansModel model, int only, const R* P,
                                                  const R* G, const R* Ph, const R* Gh, const R* dw, R* nut, R* f1,
                                                  R* Sr)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t cc = std::size_t(c);
    const FvPrim<R> p = primOf(P, c);
    const R* g = G + std::size_t(kGrad) * cc;
    R gd[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) gd[i][j] = g[3 * i + j];
    const R* ph = Ph + std::size_t(kFvRansFields) * cc;
    const R* gh = Gh + std::size_t(kRansGrad) * cc;
    R mu, kappa;
    transportAt(gas, p.temp, mu, kappa);
    FvRansCell<R> o;
    if constexpr (Sst) o = fvSstCell(model, p.rho, ph[0], ph[1], mu, gd, dw[c], gh, gh + 3);
    else o = fvSaCell(p.rho, ph[0], mu, gd, dw[c], gh, gh + 3);
    nut[c] = o.mut;
    if constexpr (Sst) f1[c] = o.f1;
    R* w = Sr + std::size_t(kRansSrc) * cc;
    if (only) {
#pragma unroll
        for (int k = 0; k < kRansSrc; ++k) w[k] = R(0.0);
        return;
    }
    w[0] = o.R[0];
    w[1] = o.R[1];
    w[2] = o.J[0];
    w[3] = o.J[1];
}

template <class R>
__device__ inline R dotS(const R g[3], const R S[3])
{
    return g[0] * S[0] + g[1] * S[1] + g[2] * S[2];
}

// Internal scalar face fluxes on the mean mass flux F[0] (FvRef::ransFaces).
template <class R, bool Sst>
__global__ void __launch_bounds__(kT) kFvRansFace(View<R> m, Gas<R> gas, int only, const R* P, const R* F,
                                                  const R* Ph, const R* Gh, const R* nut, const R* f1, R* Ft)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const FaceRow<R>& r = m.face[f];
    const R mdot = F[std::size_t(m.nV) * std::size_t(f)];
    const R* pa = Ph + std::size_t(kFvRansFields) * std::size_t(r.own);
    const R* pb = Ph + std::size_t(kFvRansFields) * std::size_t(r.nei);
    R* w = Ft + std::size_t(kFvRansFields) * std::size_t(f);
    if (only) {
        w[0] = fvUpwind(mdot, pa[0], pb[0]);
        if constexpr (Sst) w[1] = fvUpwind(mdot, pa[1], pb[1]);
        return;
    }
    const FvPrim<R> A = primOf(P, r.own), B = primOf(P, r.nei);
    R muf, kappa;
    transportAt(gas, R(0.5) * (A.temp + B.temp), muf, kappa);
    const R* ga = Gh + std::size_t(kRansGrad) * std::size_t(r.own);
    const R* gb = Gh + std::size_t(kRansGrad) * std::size_t(r.nei);
    R g[3];
    fvFaceGrad(ga, gb, pa[0], pb[0], r.d, r.corr, g);
    if constexpr (Sst) {
        const R F1f = R(0.5) * (f1[r.own] + f1[r.nei]), mutf = R(0.5) * (nut[r.own] + nut[r.nei]);
        w[0] = fvSstFaceFlux(mdot, pa[0], pb[0], muf, sstBlendC(F1f, sst::sigmaK1, sst::sigmaK2), mutf, dotS(g, r.S));
        fvFaceGrad(ga + 3, gb + 3, pa[1], pb[1], r.d, r.corr, g);
        w[1] = fvSstFaceFlux(mdot, pa[1], pb[1], muf, sstBlendC(F1f, sst::sigmaW1, sst::sigmaW2), mutf, dotS(g, r.S));
    } else {
        w[0] = fvSaFaceFlux(mdot, pa[0], pb[0], muf, R(0.5) * (A.rho + B.rho), dotS(g, r.S));
    }
}

// Boundary scalar fluxes: Wall diffusion on the molecular mu toward the wall
// value, Slip nothing, open faces the mean mass flux carrying the upwind value.
template <class R, bool Sst>
__global__ void __launch_bounds__(kT) kFvRansBndFace(View<R> m, Gas<R> gas, int only, const R* P, const R* Bs,
                                                     const R* F, const R* Ph, const R* Bh, const R* Gh, R* Ft)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const BndRow<R>& r = m.bnd[b];
    const FvBc& bc = m.bcs[r.bc];
    const std::size_t fc = std::size_t(m.nI) + std::size_t(b);
    R* w = Ft + std::size_t(kFvRansFields) * fc;
    const R* bv = Bh + std::size_t(kFvRansFields) * std::size_t(b);
    if (bc.kind == FvBcKind::Slip || (bc.kind == FvBcKind::Wall && only)) {
        w[0] = R(0.0);
        w[1] = R(0.0);
        return;
    }
    if (bc.kind != FvBcKind::Wall) {
        const R mdot = F[std::size_t(m.nV) * fc];
        w[0] = mdot * bv[0];
        if constexpr (Sst) w[1] = mdot * bv[1];
        return;
    }
    const FvPrim<R> A = primOf(P, r.own), B = primOf(Bs, b);
    R mu, kappa;
    transportAt(gas, fvWallTemp(bc, A, B), mu, kappa);
    const R* pa = Ph + std::size_t(kFvRansFields) * std::size_t(r.own);
    const R* ga = Gh + std::size_t(kRansGrad) * std::size_t(r.own);
    R g[3];
    fvFaceGradBnd(ga, pa[0], bv[0], r.d, r.corr, g);
    if constexpr (Sst) {
        w[0] = -mu * dotS(g, r.S);
        fvFaceGradBnd(ga + 3, pa[1], bv[1], r.d, r.corr, g);
        w[1] = -mu * dotS(g, r.S);
    } else {
        w[0] = -(R(1.0 / sa::sigma) * mu) * dotS(g, r.S);
    }
}

// The scalar gather and its stage: ascending CSR faces, own index only.
// residual: out = R. Else the point-implicit combination of kSstStageCurv /
// kSaStageCurv, SST's counted guard against the stage's output density.
template <class R, bool Sst, bool Comp, bool Lts>
__global__ void __launch_bounds__(kT) kFvRansGather(View<R> m, const R* Ft, const R* Sr, const R* a, const R* al,
                                                    const R* b, const R* bl, R* out, R* ol, R ca, R cb, R cr0,
                                                    int residual, R omegaFloor, unsigned long long* hits, const R* lts)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    constexpr int nT = Sst ? 2 : 1;
    R acc[nT];
#pragma unroll
    for (int e = 0; e < nT; ++e) acc[e] = R(0.0);
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.gFace[e];
        const R* w = Ft + std::size_t(kFvRansFields) * std::size_t(code >> 1);
        if (code & 1) {
#pragma unroll
            for (int v = 0; v < nT; ++v) acc[v] -= w[v];
        } else {
#pragma unroll
            for (int v = 0; v < nT; ++v) acc[v] += w[v];
        }
    }
    const R Vc = m.V[c];
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    const R* sr = Sr + std::size_t(kRansSrc) * cc;
    R cr = cr0;
    if constexpr (Lts) cr *= lts[c];
#pragma unroll
    for (int e = 0; e < nT; ++e) {
        const std::size_t i = std::size_t(5 + e) * nC + cc;
        const R rr = -acc[e] / Vc + sr[e];
        if (residual) {
            out[i] = rr;
            continue;
        }
        const R comb = ca * a[i] + cb * b[i];
        R hi, lo = R(0.0);
        if constexpr (Comp) {
            const Word2Of<R> w2 = internal::combineWord2(cb, a[i], al[i], b[i], bl[i],
                                                    pointImplicitDelta(comb, cr, rr, sr[2 + e], b[i]));
            hi = w2.hi;
            lo = w2.lo;
        } else {
            hi = pointImplicit(comb, cr, rr, sr[2 + e], b[i]);
        }
        if constexpr (Sst) {
            const R floorV = e == 0 ? R(0.0) : omegaFloor * out[cc];
            if (!(hi >= floorV)) {
                hi = floorV;
                lo = R(0.0);
                atomicAdd(hits, 1ull);
            }
        }
        out[i] = hi;
        if constexpr (Comp) ol[i] = lo;
    }
}

// The constant force per volume on a written stage: m + cr f, E + cr f.u, u of
// the stage's input.
template <class R, bool Comp, bool Lts>
__global__ void kFvForce(View<R> m, R fx, R fy, R fz, R cr0, const R* P, R* out, R* ol, const R* lts)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R cr = cr0;
    if constexpr (Lts) cr *= lts[c];
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    const R* p = P + std::size_t(kPrim) * cc;
    const R f[3] = {fx, fy, fz};
    const R pw = fx * p[1] + fy * p[2] + fz * p[3];
#pragma unroll
    for (int v = 1; v <= 4; ++v) {
        const std::size_t i = std::size_t(v) * nC + cc;
        const R x = cr * (v < 4 ? f[v - 1] : pw);
        if constexpr (Comp) {
            const Word2Of<R> w = addWord2(out[i], ol[i], x);
            out[i] = w.hi;
            ol[i] = w.lo;
        } else {
            out[i] += x;
        }
    }
}

// lts[c] = dt_c / dtRef.
template <class R>
__global__ void kFvLts(int nC, double dtRef, const R* dtc, R* lts)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    lts[c] = R(double(dtc[c]) / dtRef);
}

// The tally of one boundary face: gauge pressure force, viscous force, heat
// into the body. The stress and heat of kFvBndFace's flux; a Slip face carries
// only its normal stress.
template <class R, bool Les>
__global__ void __launch_bounds__(kT) kFvTally(View<R> m, Gas<R> gas, int n, const int* list, R pRef, const R* P,
                                               const R* Bs, const R* G, const R* nut, R* out)
{
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i >= n) return;
    const int b = list[i];
    const BndRow<R>& r = m.bnd[b];
    const FvBc& bc = m.bcs[r.bc];
    const FvPrim<R> A = primOf(P, r.own), B = primOf(Bs, b);
    R gu[3][3], gT[3], mu = R(0.0), kappa = R(0.0);
    bndViscous<R, Les>(r, bc, gas, A, B, G, nut, gu, gT, mu, kappa);
    R ts[3] = {R(0.0), R(0.0), R(0.0)};
    if (mu != R(0.0)) fvTraction(gu, mu, r.S, ts);
    if (bc.kind == FvBcKind::Slip) {
        const R S2 = r.S[0] * r.S[0] + r.S[1] * r.S[1] + r.S[2] * r.S[2];
        const R tn = (ts[0] * r.S[0] + ts[1] * r.S[1] + ts[2] * r.S[2]) / S2;
#pragma unroll
        for (int k = 0; k < 3; ++k) ts[k] = tn * r.S[k];
    }
    const R dp = A.p - pRef;
    R* w = out + std::size_t(kTally) * std::size_t(i);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        w[k] = dp * r.S[k];
        w[3 + k] = -ts[k];
    }
    w[6] = bc.kind == FvBcKind::Wall && bc.isothermal && kappa != R(0.0)
               ? -(kappa * (gT[0] * r.S[0] + gT[1] * r.S[1] + gT[2] * r.S[2]))
               : R(0.0);
}

// --- species (FvChem; FvRef's species passes). nY == 0 launches none. ---------

// The species kernels' fixed arguments. iY: the first species row of a state or
// flux row.
template <class R>
struct SpView
{
    int nY, iY, fuel, ox;
    R lnA, Ta, q, invPr, stoich;
    // LES, TFLES (Diff != 0 instantiations): 1/Sc_t, the eddy conductivity scale
    // cp/Pr_t, LES on (the Thick kernel's SGS term).
    R invSct, kapLes;
    int les;
};

// The species face diffusivity of kFvSpFace, one instantiation each: Molecular (the
// shipped kernel), Eddy (LES), Thick (TFLES, with or without LES).
enum class FvSpDiff : int
{
    Molecular = 0,
    Eddy = 1,
    Thick = 2
};

// Y_k = rhoY_k / rho [nY per cell] and the ONE omega of this stage. Two: model C1.
// Thick: (E/F) omega, eof the step's per-cell E/F (kChemCellPass<true>).
template <class R, bool Two, bool Thick = false>
__global__ void kFvSpCell(View<R> m, SpView<R> sp, const R* q, const R* P, R* Y, R* om, const R* eof)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    const R rho = q[cc];
    R* y = Y + std::size_t(sp.nY) * cc;
    for (int s = 0; s < sp.nY; ++s) y[s] = q[std::size_t(sp.iY + s) * nC + cc] / rho;
    if constexpr (Two) om[c] = arrheniusRate2(sp.lnA, sp.Ta, rho, y[sp.fuel], y[sp.ox], P[std::size_t(kPrim) * cc + 6]);
    else if constexpr (Thick) om[c] = arrheniusRate(sp.lnA, sp.Ta, rho, y[sp.fuel], P[std::size_t(kPrim) * cc + 6]) * eof[c];
    else om[c] = arrheniusRate(sp.lnA, sp.Ta, rho, y[sp.fuel], P[std::size_t(kPrim) * cc + 6]);
}

// kFvSpCell<R, true> with model C2b (FvChem::law): arrheniusRate2(lnA + ln C(xi), ...),
// xi of the cell's unfloored Y_F, Y_O (the structured kChemCellPass2Phi). Its own kernel:
// the C1 instantiation stays the shipped one.
template <class R>
__global__ void kFvSpCellPhi(View<R> m, SpView<R> sp, PhiRateLaw<R> law, const R* q, const R* P, R* Y, R* om)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    const R rho = q[cc];
    R* y = Y + std::size_t(sp.nY) * cc;
    for (int s = 0; s < sp.nY; ++s) y[s] = q[std::size_t(sp.iY + s) * nC + cc] / rho;
    const R yf = y[sp.fuel], yo = y[sp.ox];
    om[c] = arrheniusRate2(sp.lnA + phiRateLn(law, yf, yo), sp.Ta, rho, yf, yo, P[std::size_t(kPrim) * cc + 6]);
}

// Boundary Y [nY per compact face]: fvBoundaryY on the boundary state's mass flux.
template <class R>
__global__ void kFvSpBnd(View<R> m, SpView<R> sp, const R* Y, const R* Bs, R* Yb)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const BndRow<R>& f = m.bnd[b];
    const FvBc& bc = m.bcs[f.bc];
    const FvPrim<R> B = primOf(Bs, b);
    const R mdot = B.rho * (B.u[0] * f.S[0] + B.u[1] * f.S[1] + B.u[2] * f.S[2]);
    const R* y = Y + std::size_t(sp.nY) * std::size_t(f.own);
    R* o = Yb + std::size_t(sp.nY) * std::size_t(b);
    for (int s = 0; s < sp.nY; ++s) o[s] = fvBoundaryY(bc, s, y[s], mdot);
}

// Cell gradients of Y [3 nY per cell]: kFvGrad's sum, CSR order.
template <class R>
__global__ void __launch_bounds__(kT) kFvGradY(View<R> m, int nY, const R* Y, const R* Yb, R* GY)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const R* yc = Y + std::size_t(nY) * std::size_t(c);
    R g[3 * kFvMaxSpecies];
    for (int k = 0; k < 3 * nY; ++k) g[k] = R(0.0);
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const GradEnt<R>& n = m.gEnt[e];
        const R* yo = n.other >= 0 ? Y + std::size_t(nY) * std::size_t(n.other) : Yb + std::size_t(nY) * std::size_t(~n.other);
        for (int s = 0; s < nY; ++s) {
            const R dphi = yo[s] - yc[s];
            g[3 * s + 0] += n.w[0] * dphi;
            g[3 * s + 1] += n.w[1] * dphi;
            g[3 * s + 2] += n.w[2] * dphi;
        }
    }
    R* o = GY + std::size_t(3 * nY) * std::size_t(c);
    for (int k = 0; k < 3 * nY; ++k) o[k] = g[k];
}

// Internal species fluxes: the F[0] of this face times the limited face value,
// minus Fick. Runs after kFvFace. Visc: mu > 0. Two: one limiter shared by all
// species (the premixed kernel is the other instantiation, bit for bit). Diff
// (FvSpDiff, premixed, Visc): Eddy rho D + nut_f/Sc_t; Thick fvThickRhoD and the
// thickened energy flux added to this face's F_E (G the cell gradients, fe, omc the
// step's F E and Omega). Molecular: the kernel as it was.
template <class R, bool Visc, bool Two, int Diff = 0>
__global__ void __launch_bounds__(kT) kFvSpFace(View<R> m, SpView<R> sp, Gas<R> gas, const R* P, const R* Y, const R* GY,
                                                const R* G, const R* nut, const R* fe, const R* omc, R* F)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const FaceRow<R>& r = m.face[f];
    R* w = F + std::size_t(m.nV) * std::size_t(f);
    const R mdot = w[0];
    R rhoD = R(0.0);
    if constexpr (Visc) {
        R mu, kappa;
        transportAt(gas, R(0.5) * (P[std::size_t(kPrim) * std::size_t(r.own) + 6] + P[std::size_t(kPrim) * std::size_t(r.nei) + 6]),
                    mu, kappa);
        rhoD = mu * sp.invPr;
        if constexpr (Diff == int(FvSpDiff::Eddy)) {
            rhoD = fvEddyRhoD(rhoD, R(0.5) * (nut[r.own] + nut[r.nei]), sp.invSct);
        } else if constexpr (Diff == int(FvSpDiff::Thick)) {
            const R nutf = sp.les ? R(0.5) * (nut[r.own] + nut[r.nei]) : R(0.0);
            const R FEf = R(0.5) * (fe[r.own] + fe[r.nei]), Omf = R(0.5) * (omc[r.own] + omc[r.nei]);
            rhoD = fvThickRhoD(rhoD, FEf, Omf, nutf, sp.invSct, sp.les != 0);
            R ga[3], gb[3], gT[3];
            cellGrad(G, r.own, 3, ga);
            cellGrad(G, r.nei, 3, gb);
            fvFaceGrad(ga, gb, P[std::size_t(kPrim) * std::size_t(r.own) + 6], P[std::size_t(kPrim) * std::size_t(r.nei) + 6], r.d,
                       r.corr, gT);
            const R gTS = gT[0] * r.S[0] + gT[1] * r.S[1] + gT[2] * r.S[2];
            w[4] = w[4] + fvThickEnergyFlux(FEf, Omf, nutf, kappa, sp.kapLes, gTS, sp.les != 0);
        }
    }
    const R* yP = Y + std::size_t(sp.nY) * std::size_t(r.own);
    const R* yN = Y + std::size_t(sp.nY) * std::size_t(r.nei);
    const R* gP = GY + std::size_t(3 * sp.nY) * std::size_t(r.own);
    const R* gN = GY + std::size_t(3 * sp.nY) * std::size_t(r.nei);
    R th = R(2.0);
    if constexpr (Two)
        for (int s = 0; s < sp.nY; ++s) {
            const R ga[3] = {gP[3 * s], gP[3 * s + 1], gP[3 * s + 2]};
            const R gb[3] = {gN[3 * s], gN[3 * s + 1], gN[3 * s + 2]};
            th = fvThetaVote(th, mdot, yP[s], yN[s], ga, gb, r.d);
        }
    for (int s = 0; s < sp.nY; ++s) {
        const R ga[3] = {gP[3 * s], gP[3 * s + 1], gP[3 * s + 2]};
        const R gb[3] = {gN[3 * s], gN[3 * s + 1], gN[3 * s + 2]};
        R g[3];
        fvFaceGrad(ga, gb, yP[s], yN[s], r.d, r.corr, g);
        if constexpr (Two) w[sp.iY + s] = fvSpeciesFlux(mdot, fvFaceYShared(mdot, yP[s], yN[s], th), rhoD, g, r.S);
        else w[sp.iY + s] = fvSpeciesFlux(mdot, fvFaceY(mdot, yP[s], yN[s], ga, gb, r.d), rhoD, g, r.S);
    }
}

// Boundary species fluxes: F[0] Y_b, no diffusion. Two: a `diffusive` Inflow row adds
// Fick (premixed never has one: its instantiation is the kernel bit for bit).
template <class R, bool Two>
__global__ void __launch_bounds__(kT) kFvSpBndFace(View<R> m, SpView<R> sp, Gas<R> gas, const R* P, const R* Bs, const R* Y,
                                                   const R* GY, const R* Yb, R* F)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    R* w = F + std::size_t(m.nV) * (std::size_t(m.nI) + std::size_t(b));
    const R* y = Yb + std::size_t(sp.nY) * std::size_t(b);
    for (int s = 0; s < sp.nY; ++s) w[sp.iY + s] = w[0] * y[s];
    if constexpr (Two) {
        const BndRow<R>& r = m.bnd[b];
        if (!m.bcs[r.bc].diffusive || gas.tr.muRef == 0.0) return;
        R mu, kappa;
        transportAt(gas, R(0.5) * (P[std::size_t(kPrim) * std::size_t(r.own) + 6] + Bs[std::size_t(kPrim) * std::size_t(b) + 6]), mu,
                    kappa);
        const R* yc = Y + std::size_t(sp.nY) * std::size_t(r.own);
        const R* gc = GY + std::size_t(3 * sp.nY) * std::size_t(r.own);
        for (int s = 0; s < sp.nY; ++s) {
            const R ga[3] = {gc[3 * s], gc[3 * s + 1], gc[3 * s + 2]};
            R g[3];
            fvFaceGradBnd(ga, yc[s], y[s], r.d, r.corr, g);
            w[sp.iY + s] = fvSpeciesFlux(w[0], y[s], mu * sp.invPr, g, r.S);
        }
    }
}

// The step, after every row's gather: the fuel loses cr omega, E gains cr q omega,
// on the ONE omega (kSpeciesStage's E, a stage add). omega == 0 moves no bit. Two:
// the oxidizer loses cr s omega.
template <class R, bool Comp, bool Two>
__global__ void kFvChem(View<R> m, SpView<R> sp, const R* om, R* out, R* ol, R cr)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    const std::size_t iF = std::size_t(sp.iY + sp.fuel) * nC + cc, iE = 4 * nC + cc;
    if constexpr (Comp) {
        const Word2Of<R> f = addWord2(out[iF], ol[iF], -(cr * om[c]));
        const Word2Of<R> e = addWord2(out[iE], ol[iE], cr * (sp.q * om[c]));
        out[iF] = f.hi;
        ol[iF] = f.lo;
        out[iE] = e.hi;
        ol[iE] = e.lo;
    } else {
        out[iF] -= cr * om[c];
        out[iE] += cr * (sp.q * om[c]);
    }
    if constexpr (Two) {
        const std::size_t iO = std::size_t(sp.iY + sp.ox) * nC + cc;
        if constexpr (Comp) {
            const Word2Of<R> o = addWord2(out[iO], ol[iO], -(cr * (sp.stoich * om[c])));
            out[iO] = o.hi;
            ol[iO] = o.lo;
        } else {
            out[iO] -= cr * (sp.stoich * om[c]);
        }
    }
}

// --- TFLES (FvCompConfig::tfles; FvRef's tfles* passes, gpu/tfles.hip on faces). ---

// The fixed arguments of the TFLES kernels.
template <class R>
struct TfView
{
    int iF, rad, uniform, charlette;
    R lnAs, TaS, beta, Fmax, deltaL, nuU, effBeta;
};

// Raw sensor tanh(beta w_s) (w_s the rate at TaS, kTflesSensorRaw) and, for the
// efficiency, curl u off the cell gradients [3][nC].
template <class R>
__global__ void kFvTflesRaw(View<R> m, TfView<R> t, const R* q, const R* P, const R* G, R* raw, R* curl)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    if (!t.uniform) {
        const R r = q[cc];
        raw[c] = tanh(t.beta * arrheniusRate(t.lnAs, t.TaS, r, q[std::size_t(t.iF) * nC + cc] / r, P[std::size_t(kPrim) * cc + 6]));
    }
    if (t.charlette) {
        const R* g = G + std::size_t(kGrad) * cc;
        curl[cc] = g[7] - g[5];
        curl[nC + cc] = g[2] - g[6];
        curl[2 * nC + cc] = g[3] - g[1];
    }
}

// One pass of the T-gated max filter along walked axis slot `pass`: out = max of in
// over the cells the face walk reaches in rad steps each way with T >= T_c (c itself
// included). next: fvTflesWalk's table, -1 stops.
template <class R>
__global__ void kFvTflesDilate(View<R> m, int rad, const int* next, const R* P, const R* in, R* out)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC);
    const R Tc = P[std::size_t(kPrim) * std::size_t(c) + 6];
    R mx = in[c];
    for (int dir = 0; dir < 2; ++dir) {
        const int* nx = next + std::size_t(dir) * nC;
        int n = c;
        for (int k = 0; k < rad; ++k) {
            n = nx[n];
            if (n < 0) break;
            if (P[std::size_t(kPrim) * std::size_t(n) + 6] >= Tc) mx = fmax(mx, in[n]);
        }
    }
    out[c] = mx;
}

// Colin's grid u' = c2 h^3 |lap(curl u)| (coef = c2 h^3 per cell), the compact
// Laplacian (1/V) sum (phi_N - phi_P) corr.S over the cell's internal CSR entries.
template <class R>
__global__ void __launch_bounds__(kT) kFvTflesColin(View<R> m, const R* lapW, const R* coef, const R* curl, R* uh)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC);
    R s2 = R(0.0);
    for (int k = 0; k < 3; ++k) {
        const R* f = curl + std::size_t(k) * nC;
        R l = R(0.0);
        for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
            const int o = m.gEnt[e].other;
            if (o < 0) continue;
            l += lapW[e] * (f[o] - f[c]);
        }
        l /= m.V[c];
        s2 += l * l;
    }
    uh[c] = coef[c] * sqrt(s2);
}

// F = 1 + (F_max - 1) Omega, E Charlette off the edge layers (kTflesCombine): fe =
// F E, eof = E / F, om = Omega. W null: the uniform sensor (Omega 1).
template <class R>
__global__ void kFvTflesCombine(int nC, TfView<R> t, const R* W, const R* uh, const R* hc, const std::uint8_t* edge, R* fe,
                                R* eof, R* om)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    const R w = W ? W[c] : R(1.0);
    const R F = R(1.0) + (t.Fmax - R(1.0)) * w;
    R E = R(1.0);
    if (t.charlette && !edge[c]) {
        const R D = F * t.deltaL;
        const R up = colinUprime(uh[c], D, hc[c]);
        E = charletteEfficiency(F, up, up * D / t.nuU, t.effBeta);
    }
    fe[c] = F * E;
    eof[c] = E / F;
    om[c] = w;
}

// J_c of FvRef::chemDt, FP64 from the state words.
template <class R>
struct ChemJ
{
    const R* s;
    std::size_t n;
    int iFuel;
    double gamma, Rgas, lnA, Ta, q, cv;
    __device__ double operator()(int c) const
    {
        const std::size_t cc = std::size_t(c);
        const FvPrim<double> p = fvPrimOf(double(s[cc]), double(s[n + cc]), double(s[2 * n + cc]), double(s[3 * n + cc]),
                                          double(s[4 * n + cc]), gamma, Rgas);
        const double Y = double(s[std::size_t(iFuel) * n + cc]) / p.rho;
        const double J = exp(lnA - Ta / p.temp) * (1.0 + q * (Y > 0.0 ? Y : 0.0) * Ta / (cv * p.temp * p.temp));
        return isfinite(J) ? J : HUGE_VAL; // never dropped by the max
    }
};

// J_c of the two-reactant step: ChemRate2's expression, FP64 from the state words.
template <class R>
struct ChemJ2
{
    const R* s;
    std::size_t n;
    int iFuel, iOx;
    double gamma, Rgas, lnA, Ta, q, cv, stoich;
    __device__ double operator()(int c) const
    {
        const std::size_t cc = std::size_t(c);
        const FvPrim<double> p = fvPrimOf(double(s[cc]), double(s[n + cc]), double(s[2 * n + cc]), double(s[3 * n + cc]),
                                          double(s[4 * n + cc]), gamma, Rgas);
        const double yf = fmax(double(s[std::size_t(iFuel) * n + cc]) / p.rho, 0.0);
        const double yo = fmax(double(s[std::size_t(iOx) * n + cc]) / p.rho, 0.0);
        const double J = exp(lnA - Ta / p.temp) * p.rho * (yo + stoich * yf + q * yf * yo * Ta / (cv * p.temp * p.temp));
        return isfinite(J) ? J : HUGE_VAL;
    }
};

// ChemJ2 with the C2b law: ChemRate2Phi's expression (ln C in FP64 of the unfloored
// Y_F, Y_O), FP64 from the state words. Its own functor: ChemJ2 stays the shipped one.
template <class R>
struct ChemJ2Phi
{
    const R* s;
    std::size_t n;
    int iFuel, iOx;
    double gamma, Rgas, lnA, Ta, q, cv, stoich;
    PhiRateLaw<double> law;
    __device__ double operator()(int c) const
    {
        const std::size_t cc = std::size_t(c);
        const FvPrim<double> p = fvPrimOf(double(s[cc]), double(s[n + cc]), double(s[2 * n + cc]), double(s[3 * n + cc]),
                                          double(s[4 * n + cc]), gamma, Rgas);
        const double YF = double(s[std::size_t(iFuel) * n + cc]) / p.rho, YO = double(s[std::size_t(iOx) * n + cc]) / p.rho;
        const double yf = fmax(YF, 0.0), yo = fmax(YO, 0.0);
        const double J =
            exp(lnA + phiRateLn(law, YF, YO) - Ta / p.temp) * p.rho * (yo + stoich * yf + q * yf * yo * Ta / (cv * p.temp * p.temp));
        return isfinite(J) ? J : HUGE_VAL;
    }
};

// boostX's arithmetic in any direction.
template <class R>
__global__ void kFvBoost(int nC, R ux, R uy, R uz, R* q)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    const std::size_t n = std::size_t(nC), cc = std::size_t(c);
    const R rho = q[cc];
    const R mx = q[n + cc], my = q[2 * n + cc], mz = q[3 * n + cc];
    q[n + cc] = mx + rho * ux;
    q[2 * n + cc] = my + rho * uy;
    q[3 * n + cc] = mz + rho * uz;
    q[4 * n + cc] += mx * ux + my * uy + mz * uz + R(0.5) * rho * (ux * ux + uy * uy + uz * uz);
}

// The boost on the two-word state: the increments land in the low words' pair.
template <class R>
__global__ void kFvBoostPair(int nC, R ux, R uy, R uz, R* q, R* l)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    const std::size_t n = std::size_t(nC), cc = std::size_t(c);
    const R rho = q[cc];
    const R mx = q[n + cc], my = q[2 * n + cc], mz = q[3 * n + cc];
    const R du[3] = {ux, uy, uz};
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const std::size_t i = std::size_t(k + 1) * n + cc;
        const Word2Of<R> w = addWord2(q[i], l[i], rho * du[k]);
        q[i] = w.hi;
        l[i] = w.lo;
    }
    const std::size_t e = 4 * n + cc;
    const Word2Of<R> w = addWord2(q[e], l[e], mx * ux + my * uy + mz * uz + R(0.5) * rho * (ux * ux + uy * uy + uz * uz));
    q[e] = w.hi;
    l[e] = w.lo;
}

template <class R>
struct MinCell
{
    const R* d;
    __device__ double operator()(int c) const { return double(d[c]); }
};

// --- The impulse ledger ([thrust], FvLedger; docs/nozzle.md "Thrust"). ---

// A block's fixed-order FP64 sum: the shared-memory tree, no shuffles (wave32 = wave64). Every thread
// calls it; thread 0 gets the sum.
__device__ inline double blockSum(double v, double* sh)
{
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = kT / 2; s > 0; s >>= 1) {
        if (int(threadIdx.x) < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const double o = sh[0];
    __syncthreads();
    return o;
}

// One block per patch: S[p][v] = sum of F row v over the patch's compact boundary faces seg[p] .. seg[p
// + 1], thread t summing faces t, t + kT, ... in ascending order, then the tree.
template <class R>
__global__ void __launch_bounds__(kT) kFvLedger(int nI, int nV, const int* seg, const R* F, double* S)
{
    __shared__ double sh[kT];
    const int p = int(blockIdx.x);
    for (int v = 0; v < nV; ++v) {
        double a = 0.0;
        for (int b = seg[p] + int(threadIdx.x); b < seg[p + 1]; b += kT)
            a += double(F[std::size_t(nV) * (std::size_t(nI) + std::size_t(b)) + std::size_t(v)]);
        const double s = blockSum(a, sh);
        if (threadIdx.x == 0) S[std::size_t(p) * std::size_t(nV) + std::size_t(v)] = s;
    }
}

// Block partials of sum_c V omega, cells in ascending blocks.
template <class R>
__global__ void __launch_bounds__(kT) kFvLedgerOmega(int nC, const R* V, const R* om, double* part)
{
    __shared__ double sh[kT];
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    const double s = blockSum(c < nC ? double(V[c]) * double(om[c]) : 0.0, sh);
    if (threadIdx.x == 0) part[blockIdx.x] = s;
}

// The step: I += (w0 S0 + w1 S1) + w2 S2 per patch and row; omega likewise from the partials summed in
// ascending block order per stage. Compensated (fvLedgerAdd): I, om the plain sums bit for bit, lo, omLo
// their low words. One thread.
__global__ void kFvLedgerStep(int n, int nOm, const double* S, const double* part, double w0, double w1, double w2, double* I,
                              double* lo, double* om, double* omLo)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < n; ++i) fvLedgerAdd(I[i], lo[i], (w0 * S[i] + w1 * S[n + i]) + w2 * S[2 * n + i]);
    if (nOm == 0) return;
    double o[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
        for (int b = 0; b < nOm; ++b) o[k] += part[k * nOm + b];
    fvLedgerAdd(om[0], omLo[0], (w0 * o[0] + w1 * o[1]) + w2 * o[2]);
}

std::string hipErr(const char* what, hipError_t e)
{
    return std::format("FV device: {}: {}", what, hipGetErrorString(e));
}

} // namespace

struct DeviceFvComp::Impl
{
    virtual ~Impl() = default;
    virtual std::expected<void, std::string> upload(const dns::FvState&) = 0;
    virtual std::expected<void, std::string> download(dns::FvState&) const = 0;
    virtual std::expected<void, std::string> rhs(dns::FvState&) = 0;
    virtual std::expected<void, std::string> evaluate(FvEval&, dns::FvState&) = 0;
    virtual void step(double) = 0;
    virtual std::expected<double, std::string> localDt(FvDtLimits) = 0;
    virtual std::expected<double, std::string> stableDt(FvDtLimits) = 0;
    virtual std::expected<double, std::string> chemDt(double) = 0;
    virtual std::expected<void, std::string> speciesFields(dns::FvState&, std::vector<double>&) = 0;
    virtual std::expected<std::vector<double>, std::string> patchFlux(int) = 0;
    virtual std::expected<FvSurface, std::string> surface() = 0;
    virtual std::expected<std::vector<double>, std::string> eddyViscosity() = 0;
    virtual std::expected<DeviceFvComp::TflesStats, std::string> tflesStats() = 0;
    virtual std::expected<unsigned long long, std::string> floorHits() const = 0;
    virtual std::expected<long, std::string> shockCells() = 0;
    virtual std::expected<int, std::string> shockByteMax() = 0;
    virtual std::expected<void, std::string> cellFields(std::array<std::vector<double>, 7>&,
                                                        std::array<std::vector<double>, 12>&) = 0;
    virtual void boost(const std::array<double, 3>&) = 0;
    virtual std::expected<void, std::string> setBc(int, const FvBc&) = 0;
    virtual std::expected<void, RunError> save(Checkpoint&) const = 0;
    virtual std::expected<void, RunError> load(const Checkpoint&) = 0;
    virtual std::expected<void, std::string> setBcRows(int, std::span<const FvBc>) = 0;
    virtual std::expected<void, std::string> setLedger(bool, bool) = 0;
    virtual std::expected<FvLedger, std::string> ledger() const = 0;
    virtual std::size_t bytes() const = 0;
    int nC = 0;
    FvPrecision precision{};
    std::vector<double> V, C[3]; // host copies for sums
    double gamma = 1.4, Rgas = 1;
};

namespace {

template <class T>
std::expected<DeviceBuffer<T>, std::string> sized(std::size_t n, const char* what)
{
    auto b = DeviceBuffer<T>::make(std::max<std::size_t>(n, 1));
    if (!b) return std::unexpected(hipErr(what, b.error()));
    return std::move(*b);
}

template <class T>
std::expected<DeviceBuffer<T>, std::string> uploaded(const std::vector<T>& h, const char* what)
{
    auto b = sized<T>(h.size(), what);
    if (!b) return b;
    if (!h.empty())
        if (const hipError_t e = b->upload(h.data()); e != hipSuccess) return std::unexpected(hipErr(what, e));
    return b;
}

template <class R>
struct FvCompImpl final : DeviceFvComp::Impl
{
    int nI = 0, nB = 0, nF = 0, nEnt = 0, nV = kFvMeanVars, nTally = 0;
    bool visc = false, les = false, compensated = false, hasG = false;
    int sigma = 0;
    // RANS: nState = 5 + nT rows; force: the channel's drive (dns/fvrans.hpp).
    int nState = kFvMeanVars, nT = 0;
    bool rans = false, sst = false, ransOnly = false, force = false, useLts = false;
    FvModuleSet mods{}; // the inventory's modules of this solver (localDt adds Steady)
    RansModel ransModel = RansModel::None;
    R phiInf[2] = {R(0.0), R(0.0)}, omegaFloor = R(0.0), fb[3] = {R(0.0), R(0.0), R(0.0)};
    std::array<double, 3> grav{};
    double muRef = 0;
    R fMol = 1, fEddy = 1, pRef = 0;
    std::vector<FvBc> bcHost;
    std::vector<int> bndFace;  // compact boundary -> FaceMesh boundary index
    std::vector<int> bcRows;
    std::vector<int> tallyPatch; // tally entry -> FaceMesh::patches index
    std::vector<int> bndPatch;   // compact boundary -> FaceMesh::patches index

    DeviceBuffer<FaceRow<R>> face;
    DeviceBuffer<BndRow<R>> bnd;
    DeviceBuffer<FvBc> bcs;
    DeviceBuffer<R> Vd, diffC, cd, entS;
    DeviceBuffer<int> cStart, gFace, tallyList;
    DeviceBuffer<GradEnt<R>> gEnt;
    DeviceBuffer<R> q, q1, q2, l, l1, l2, P, B, G, F, Gf, nut, dtc, tally;
    DeviceBuffer<R> Y, Yb, GY, om; // species; 1 element when nY == 0
    SpView<R> sp{};
    FvChem chem{};
    DeviceBuffer<R> dw, Ph, Bh, Gh, f1, Sr, Ft, lts; // RANS; local time stepping
    DeviceBuffer<unsigned long long> hits;
    // TFLES: the step's F E, E/F, Omega; the sensor's ping-pong, curl, Colin's u';
    // the walk, edge layers, Laplacian weights per CSR entry, h and c2 h^3 per cell.
    bool thick = false;
    TfView<R> tf{};
    int nAx = 0;
    R gPr = R(0.0);
    DeviceBuffer<R> fe, eof, omT, tfA, tfB, curl, uh, lapW, hc, coef;
    DeviceBuffer<int> walk;
    DeviceBuffer<std::uint8_t> edge;
    // --shock: the sensor bytes, the mesh lines; shockFlux: the face pass runs (hll).
    bool shockOn = false, shockFlux = false, shockForce = false;
    R eps4 = R(0.0);
    DeviceBuffer<std::uint8_t> sb;
    DeviceBuffer<int> lineStart, line, fLine;
    mutable ReduceScratch red;
    Gas<R> gas{};
    // [thrust]: the impulse ledger. seg: per patch its compact boundary range; S [3][nP][nV] the
    // stage sums, part [3][nOm] the stage's V omega partials, I [nP][nV], om the integral; Lo their
    // two-sum low words (fvLedgerAdd).
    bool ledgerOn = false, ledgerEq = false;
    int nOm = 0;
    DeviceBuffer<int> ledSeg;
    DeviceBuffer<double> ledS, ledPart, ledI, ledLo, ledOm, ledOmLo;

    View<R> view() const
    {
        return {.nC = nC,
                .nI = nI,
                .nB = nB,
                .nV = nV,
                .gamma = gamma,
                .Rgas = Rgas,
                .face = face.data(),
                .bnd = bnd.data(),
                .bcs = bcs.data(),
                .V = Vd.data(),
                .cStart = cStart.data(),
                .gFace = gFace.data(),
                .gEnt = gEnt.data()};
    }

    void primsOf(const R* s)
    {
        kFvPrim<R><<<blocksOf(std::size_t(nC)), kT>>>(view(), s, P.data());
        if (nB) kFvBnd<R><<<blocksOf(std::size_t(nB)), kT>>>(view(), P.data(), B.data());
    }

    template <bool Sst>
    void ransCellAs(const R* s)
    {
        const unsigned bc = blocksOf(std::size_t(nC));
        kFvRansPhi<R, Sst><<<bc, kT>>>(view(), s, P.data(), Ph.data());
        if (nB)
            kFvRansBnd<R, Sst><<<blocksOf(std::size_t(nB)), kT>>>(view(), gas, phiInf[0], phiInf[1], P.data(), B.data(),
                                                                  Ph.data(), dw.data(), Bh.data());
        kFvRansGrad<R><<<bc, kT>>>(view(), Ph.data(), Bh.data(), Gh.data());
        kFvRansCell<R, Sst><<<bc, kT>>>(view(), gas, ransModel, int(ransOnly), P.data(), G.data(), Ph.data(), Gh.data(),
                                        dw.data(), nut.data(), f1.data(), Sr.data());
    }

    // s: the state P was formed from.
    void gradsAndNut(const R* s)
    {
        if (!visc) return;
        kFvGrad<R><<<blocksOf(std::size_t(nC)), kT>>>(view(), P.data(), B.data(), G.data());
        if (les)
            kFvNut<R><<<blocksOf(std::size_t(nC)), kT>>>(nC, sigma, cd.data(), P.data(), G.data(), nut.data());
        if (!rans) return;
        if (sst) ransCellAs<true>(s);
        else ransCellAs<false>(s);
    }

    template <bool Visc, bool Les>
    void facesAs()
    {
        kFvFace<R, Visc, Les><<<blocksOf(std::size_t(nI)), kT>>>(view(), gas, P.data(), G.data(), nut.data(), F.data());
        if (nB)
            kFvBndFace<R, Visc, Les><<<blocksOf(std::size_t(nB)), kT>>>(view(), gas, P.data(), B.data(), G.data(),
                                                                        nut.data(), F.data());
    }

    // s: the state P was formed from.
    void faces(const R* s)
    {
        if (les || rans) facesAs<true, true>();
        else if (visc) facesAs<true, false>();
        else facesAs<false, false>();
        if (shockFlux)
            kFvShockFace<R><<<blocksOf(std::size_t(nI)), kT>>>(view(), gamma, eps4, fLine.data(), sb.data(), s, P.data(), F.data());
        if (hasG)
            kFvGravFace<R><<<blocksOf(std::size_t(nI)), kT>>>(view(), grav[0], grav[1], grav[2], P.data(), F.data(), Gf.data());
    }

    // The sensor bytes of state s (FvRef::shockSense): primitives, then the lines.
    void shockUpdate(const R* s)
    {
        const unsigned nc = blocksOf(std::size_t(nC));
        if (shockForce) {
            kFvShockForce<<<nc, kT>>>(nC, sb.data());
            return;
        }
        kFvPrim<R><<<nc, kT>>>(view(), s, P.data());
        kFvShockSense<R><<<nc, kT>>>(nC, lineStart.data(), line.data(), P.data(), sb.data());
    }

    // The species passes before the mean-flow gradients: Y, omega, boundary Y.
    void speciesCells(const R* s)
    {
        const unsigned nc = blocksOf(std::size_t(nC));
        if (chem.ox >= 0 && chem.law.n > 0)
            kFvSpCellPhi<R><<<nc, kT>>>(view(), sp, chem.law.as<R>(), s, P.data(), Y.data(), om.data());
        else if (chem.ox >= 0) kFvSpCell<R, true><<<nc, kT>>>(view(), sp, s, P.data(), Y.data(), om.data(), nullptr);
        else if (thick) kFvSpCell<R, false, true><<<nc, kT>>>(view(), sp, s, P.data(), Y.data(), om.data(), eof.data());
        else kFvSpCell<R, false><<<nc, kT>>>(view(), sp, s, P.data(), Y.data(), om.data(), nullptr);
        if (nB) kFvSpBnd<R><<<blocksOf(std::size_t(nB)), kT>>>(view(), sp, Y.data(), B.data(), Yb.data());
    }

    // TFLES: F E, E/F, Omega of state s (FvRef::tflesFields): primitives and the
    // velocity gradients of s, the raw sensor, a dilation pass per walked axis (raw ->
    // tfB -> tfA -> tfB), Colin's u', the combine. A uniform sensor without the
    // efficiency is constant: formed at make, nothing runs.
    void tflesUpdate(const R* s)
    {
        if (tf.uniform && !tf.charlette) return;
        const unsigned nc = blocksOf(std::size_t(nC));
        primsOf(s);
        if (tf.charlette) kFvGrad<R><<<nc, kT>>>(view(), P.data(), B.data(), G.data());
        kFvTflesRaw<R><<<nc, kT>>>(view(), tf, s, P.data(), G.data(), tfA.data(), curl.data());
        const R* W = nullptr;
        if (!tf.uniform) {
            R* buf[2] = {tfA.data(), tfB.data()};
            for (int p = 0; p < nAx; ++p)
                kFvTflesDilate<R><<<nc, kT>>>(view(), tf.rad, walk.data() + std::size_t(2 * p) * std::size_t(nC), P.data(),
                                              buf[p & 1], buf[(p + 1) & 1]);
            W = buf[nAx & 1];
        }
        if (tf.charlette) kFvTflesColin<R><<<nc, kT>>>(view(), lapW.data(), coef.data(), curl.data(), uh.data());
        kFvTflesCombine<R><<<nc, kT>>>(nC, tf, W, uh.data(), hc.data(), edge.data(), fe.data(), eof.data(), omT.data());
    }

    // After faces(): gradients of Y, then the species fluxes on F[0].
    void speciesFaces()
    {
        kFvGradY<R><<<blocksOf(std::size_t(nC)), kT>>>(view(), sp.nY, Y.data(), Yb.data(), GY.data());
        const unsigned nf = blocksOf(std::size_t(nI));
        const R *g = G.data(), *nt = nut.data(), *f = fe.data(), *o = omT.data();
        if (chem.ox >= 0 && les)
            kFvSpFace<R, true, true, int(FvSpDiff::Eddy)><<<nf, kT>>>(view(), sp, gas, P.data(), Y.data(), GY.data(), g, nt, f, o,
                                                                       F.data());
        else if (chem.ox >= 0 && visc)
            kFvSpFace<R, true, true><<<nf, kT>>>(view(), sp, gas, P.data(), Y.data(), GY.data(), g, nt, f, o, F.data());
        else if (chem.ox >= 0)
            kFvSpFace<R, false, true><<<nf, kT>>>(view(), sp, gas, P.data(), Y.data(), GY.data(), g, nt, f, o, F.data());
        else if (thick)
            kFvSpFace<R, true, false, int(FvSpDiff::Thick)><<<nf, kT>>>(view(), sp, gas, P.data(), Y.data(), GY.data(), g, nt, f, o,
                                                                         F.data());
        else if (les)
            kFvSpFace<R, true, false, int(FvSpDiff::Eddy)><<<nf, kT>>>(view(), sp, gas, P.data(), Y.data(), GY.data(), g, nt, f, o,
                                                                        F.data());
        else if (visc)
            kFvSpFace<R, true, false><<<nf, kT>>>(view(), sp, gas, P.data(), Y.data(), GY.data(), g, nt, f, o, F.data());
        else kFvSpFace<R, false, false><<<nf, kT>>>(view(), sp, gas, P.data(), Y.data(), GY.data(), g, nt, f, o, F.data());
        if (nB && chem.ox >= 0)
            kFvSpBndFace<R, true><<<blocksOf(std::size_t(nB)), kT>>>(view(), sp, gas, P.data(), B.data(), Y.data(), GY.data(), Yb.data(), F.data());
        else if (nB)
            kFvSpBndFace<R, false><<<blocksOf(std::size_t(nB)), kT>>>(view(), sp, gas, P.data(), B.data(), Y.data(), GY.data(), Yb.data(), F.data());
    }

    // The species rows through kFvGather, then the step. Never under LTS (localDt
    // refuses): kFvChem takes the bare cr.
    template <bool Comp>
    void speciesGather(const R* a, const R* al, const R* b, const R* bl, R* out, R* ol, R ca, R cb, R cr)
    {
        const unsigned n = blocksOf(std::size_t(nC));
        auto rows = [&]<int Count>() {
            kFvGather<R, Comp, false, kFvMeanVars, Count><<<n, kT>>>(view(), F.data(), a, al, b, bl, out, ol, ca, cb, cr,
                                                                      nullptr);
        };
        if (sp.nY == 1) rows.template operator()<1>();
        else if (sp.nY == 2) rows.template operator()<2>();
        else if (sp.nY == 3) rows.template operator()<3>();
        else rows.template operator()<4>();
        if (chem.ox >= 0) kFvChem<R, Comp, true><<<n, kT>>>(view(), sp, om.data(), out, ol, cr);
        else kFvChem<R, Comp, false><<<n, kT>>>(view(), sp, om.data(), out, ol, cr);
    }

    // The turbulence scalars' stage after the mean gather.
    template <bool Sst, bool Comp, bool Lts>
    void ransStage(const R* a, const R* al, const R* b, const R* bl, R* out, R* ol, R ca, R cb, R cr, bool residual)
    {
        kFvRansFace<R, Sst><<<blocksOf(std::size_t(nI)), kT>>>(view(), gas, int(ransOnly), P.data(), F.data(), Ph.data(),
                                                               Gh.data(), nut.data(), f1.data(), Ft.data());
        if (nB)
            kFvRansBndFace<R, Sst><<<blocksOf(std::size_t(nB)), kT>>>(view(), gas, int(ransOnly), P.data(), B.data(),
                                                                      F.data(), Ph.data(), Bh.data(), Gh.data(),
                                                                      Ft.data());
        kFvRansGather<R, Sst, Comp, Lts><<<blocksOf(std::size_t(nC)), kT>>>(view(), Ft.data(), Sr.data(), a, al, b, bl,
                                                                            out, ol, ca, cb, cr, int(residual),
                                                                            omegaFloor, hits.data(), lts.data());
    }

    // The mean gather and what follows it: gravity, the drive, the species, the scalars.
    // Runtime-empty off.
    template <bool Comp, bool Lts>
    void gatherStage(const R* a, const R* al, const R* b, const R* bl, R* out, R* ol, R ca, R cb, R cr, bool residual)
    {
        kFvGather<R, Comp, Lts><<<blocksOf(std::size_t(nC)), kT>>>(view(), F.data(), a, al, b, bl, out, ol, ca, cb, cr,
                                                                   lts.data());
        if (hasG)
            kFvGravity<R, Comp><<<blocksOf(std::size_t(nC)), kT>>>(view(), cr, Gf.data(), out, ol);
        if (force)
            kFvForce<R, Comp, Lts><<<blocksOf(std::size_t(nC)), kT>>>(view(), fb[0], fb[1], fb[2], cr, P.data(), out, ol,
                                                                      lts.data());
        if (sp.nY) speciesGather<Comp>(a, al, b, bl, out, ol, ca, cb, cr);
        if (!rans) return;
        if (sst) ransStage<true, Comp, Lts>(a, al, b, bl, out, ol, ca, cb, cr, residual);
        else ransStage<false, Comp, Lts>(a, al, b, bl, out, ol, ca, cb, cr, residual);
    }

    // Every flux of the state b.
    void rhsPasses(const R* b)
    {
        primsOf(b);
        if (sp.nY) speciesCells(b);
        gradsAndNut(b);
        faces(b);
        if (sp.nY) speciesFaces();
    }

    // One fused stage: the state being differentiated is b.
    void stage(const R* a, const R* b, R* out, R ca, R cb, R cr, bool residual = false)
    {
        rhsPasses(b);
        if (useLts && !residual) gatherStage<false, true>(a, nullptr, b, nullptr, out, nullptr, ca, cb, cr, residual);
        else gatherStage<false, false>(a, nullptr, b, nullptr, out, nullptr, ca, cb, cr, residual);
    }

    // Stage on the two-word state (FP32 hi + lo, or FP64 double-double); ca is implied (1 - cb exactly).
    void stagePair(const R* a, const R* al, const R* b, const R* bl, R* out, R* ol, R cb, R cr)
    {
        rhsPasses(b);
        if (useLts) gatherStage<true, true>(a, al, b, bl, out, ol, R(1.0) - cb, cb, cr, false);
        else gatherStage<true, false>(a, al, b, bl, out, ol, R(1.0) - cb, cb, cr, false);
    }

    // The ledger's sums of stage k: F of the stage (the gather left it), omega of the stage.
    void ledgerStage(int k)
    {
        const std::size_t nP = ledSeg.size() - 1, n = nP * std::size_t(nV);
        if (nP) kFvLedger<R><<<unsigned(nP), kT>>>(nI, nV, ledSeg.data(), F.data(), ledS.data() + std::size_t(k) * n);
        if (nOm)
            kFvLedgerOmega<R><<<unsigned(nOm), kT>>>(nC, Vd.data(), om.data(), ledPart.data() + std::size_t(k) * std::size_t(nOm));
    }

    // R2's weights: w0 = w1 = cb2 (cb1 dtR), w2 = R(cb2 dtR), the cr each stage write used (ca + cb = 1).
    void ledgerStep(R cb1, R cb2, R dtR)
    {
        const double w01 = ledgerEq ? double(dtR) / 3.0 : double(cb2) * double(cb1 * dtR);
        const double w2 = ledgerEq ? double(dtR) / 3.0 : double(R(cb2 * dtR));
        const int n = int((ledSeg.size() - 1) * std::size_t(nV));
        kFvLedgerStep<<<1, 1>>>(n, nOm, ledS.data(), ledPart.data(), w01, w01, w2, ledI.data(), ledLo.data(), ledOm.data(),
                                ledOmLo.data());
    }

    void step(double dt) override
    {
        const R cb1 = R(0.25), cb2 = R(2.0 / 3.0);
        const R dtR = R(dt);
        if (thick) tflesUpdate(q.data()); // F, E, Omega of the step's input state
        if (shockOn) shockUpdate(q.data()); // the sensor of the step's input state
        if (compensated) {
            stagePair(q.data(), l.data(), q.data(), l.data(), q1.data(), l1.data(), R(0.0), dtR);
            if (ledgerOn) ledgerStage(0);
            stagePair(q.data(), l.data(), q1.data(), l1.data(), q2.data(), l2.data(), cb1, cb1 * dtR);
            if (ledgerOn) ledgerStage(1);
            stagePair(q.data(), l.data(), q2.data(), l2.data(), q1.data(), l1.data(), cb2, cb2 * dtR);
            if (ledgerOn) {
                ledgerStage(2);
                ledgerStep(cb1, cb2, dtR);
            }
            std::swap(q, q1);
            std::swap(l, l1);
            return;
        }
        stage(q.data(), q.data(), q1.data(), R(1.0) - R(0.0), R(0.0), dtR);
        if (ledgerOn) ledgerStage(0);
        stage(q.data(), q1.data(), q2.data(), R(1.0) - cb1, cb1, cb1 * dtR);
        if (ledgerOn) ledgerStage(1);
        // q1 is dead here: stage 3 writes it and swaps, so the output never aliases a.
        stage(q.data(), q2.data(), q1.data(), R(1.0) - cb2, cb2, cb2 * dtR);
        if (ledgerOn) {
            ledgerStage(2);
            ledgerStep(cb1, cb2, dtR);
        }
        std::swap(q, q1);
    }

    std::expected<void, std::string> setLedger(bool on, bool equalWeights) override
    {
        (void)hipDeviceSynchronize();
        ledgerOn = false;
        ledSeg = DeviceBuffer<int>();
        ledS = DeviceBuffer<double>();
        ledPart = DeviceBuffer<double>();
        ledI = DeviceBuffer<double>();
        ledLo = DeviceBuffer<double>();
        ledOm = DeviceBuffer<double>();
        ledOmLo = DeviceBuffer<double>();
        nOm = 0;
        if (!on) return {};
        if (hasG || force || rans) return std::unexpected("FV ledger: gravity, a body force or RANS adds sources that are no boundary flux");
        if (useLts) return std::unexpected("FV ledger: local time stepping has no one dt");
        // Per patch its compact boundary range: compact faces keep FaceMesh's patch-major order.
        const std::size_t nP = patchFirst.size();
        std::vector<int> seg(nP + 1, 0);
        for (std::size_t p = 0; p < nP; ++p) {
            if (patchActive[p] > 0 && patchFirst[p] != seg[p]) return std::unexpected("FV ledger: boundary faces not patch-major");
            seg[p + 1] = seg[p] + patchActive[p];
        }
        nOm = sp.nY ? int(blocksOf(std::size_t(nC))) : 0;
        const std::size_t n = nP * std::size_t(nV);
        std::optional<std::string> err;
        auto take = [&](auto& dst, auto&& made) {
            if (!made) {
                if (!err) err = std::move(made.error());
                return;
            }
            dst = std::move(*made);
        };
        take(ledSeg, uploaded(seg, "ledger"));
        take(ledS, sized<double>(3 * n, "ledger"));
        take(ledPart, sized<double>(3 * std::size_t(std::max(nOm, 1)), "ledger"));
        take(ledI, uploaded(std::vector<double>(std::max<std::size_t>(n, 1), 0.0), "ledger"));
        take(ledLo, uploaded(std::vector<double>(std::max<std::size_t>(n, 1), 0.0), "ledger"));
        take(ledOm, uploaded(std::vector<double>{0.0}, "ledger"));
        take(ledOmLo, uploaded(std::vector<double>{0.0}, "ledger"));
        if (err) return std::unexpected(std::move(*err));
        ledgerEq = equalWeights;
        ledgerOn = true;
        return {};
    }

    std::expected<FvLedger, std::string> ledger() const override
    {
        if (!ledgerOn) return std::unexpected("FV ledger: off ([thrust] absent)");
        (void)hipDeviceSynchronize();
        FvLedger o = FvLedger::zeros(int(ledSeg.size() - 1), nV);
        if (!o.I.empty()) {
            if (const hipError_t e = ledI.download(o.I.data()); e != hipSuccess) return std::unexpected(hipErr("ledger", e));
            if (const hipError_t e = ledLo.download(o.lo.data()); e != hipSuccess) return std::unexpected(hipErr("ledger", e));
        }
        if (const hipError_t e = ledOm.download(&o.omega); e != hipSuccess) return std::unexpected(hipErr("ledger", e));
        if (const hipError_t e = ledOmLo.download(&o.omegaLo); e != hipSuccess) return std::unexpected(hipErr("ledger", e));
        return o;
    }

    std::expected<void, std::string> check(const char* what) const
    {
        const hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) return std::unexpected(hipErr(what, e));
        if (const hipError_t l = hipGetLastError(); l != hipSuccess) return std::unexpected(hipErr(what, l));
        if (red.err != hipSuccess) return std::unexpected(hipErr(what, red.err));
        return {};
    }

    std::expected<void, std::string> put(DeviceBuffer<R>& d, const std::vector<double>& h, const char* what)
    {
        std::vector<R> t(h.size());
        std::ranges::transform(h, t.begin(), [](double v) { return R(v); });
        if (const hipError_t e = d.upload(t.data()); e != hipSuccess) return std::unexpected(hipErr(what, e));
        return {};
    }

    std::expected<void, std::string> upload(const dns::FvState& s) override
    {
        if (s.nV != nState || s.nC != nC) return std::unexpected("FV upload: state does not match the mesh");
        if (compensated) {
            if (auto r = put(l, std::vector<double>(s.q.size(), 0.0), "state upload"); !r) return r;
        }
        return put(q, s.q, "state upload");
    }

    std::expected<void, std::string> fetch(const DeviceBuffer<R>& d, std::vector<double>& h, const char* what) const
    {
        (void)hipDeviceSynchronize();
        std::vector<R> t(d.size());
        if (const hipError_t e = d.download(t.data()); e != hipSuccess) return std::unexpected(hipErr(what, e));
        h.resize(t.size());
        std::ranges::transform(t, h.begin(), [](R v) { return double(v); });
        return {};
    }

    std::expected<void, std::string> download(dns::FvState& s) const override
    {
        s.nV = nState;
        s.nC = nC;
        if (auto r = fetch(q, s.q, "state download"); !r) return r;
        if (!compensated) return {};
        std::vector<double> lo; // hi + lo, exact in FP64
        if (auto r = fetch(l, lo, "state download"); !r) return r;
        for (std::size_t i = 0; i < lo.size(); ++i) s.q[i] += lo[i];
        return {};
    }

    std::expected<void, std::string> rhs(dns::FvState& out) override
    {
        if (thick) tflesUpdate(q.data());
        if (shockOn) shockUpdate(q.data());
        stage(q.data(), q.data(), q1.data(), R(0.0), R(0.0), R(1.0), true);
        if (auto c = check("rhs"); !c) return c;
        out.nV = nState;
        out.nC = nC;
        return fetch(q1, out.q, "rhs download");
    }

    std::expected<void, std::string> evaluate(FvEval& ev, dns::FvState& out) override
    {
        if (auto r = rhs(out); !r) return r;
        const std::size_t n = std::size_t(nC), nFull = std::size_t(nI) + bndFull;
        std::vector<double> t;
        if (auto r = fetch(P, t, "primitives"); !r) return r;
        for (std::size_t k = 0; k < 7; ++k) {
            ev.prim[k].resize(n);
            for (std::size_t c = 0; c < n; ++c) ev.prim[k][c] = t[7 * c + k];
        }
        for (auto& g : ev.grad) g.assign(n, 0.0);
        if (visc) {
            if (auto r = fetch(G, t, "gradients"); !r) return r;
            for (std::size_t k = 0; k < 12; ++k)
                for (std::size_t c = 0; c < n; ++c) ev.grad[k][c] = t[12 * c + k];
        }
        for (auto& f : ev.F) f.assign(nFull, 0.0);
        if (auto r = fetch(F, t, "fluxes"); !r) return r;
        const std::size_t w = std::size_t(nV), nY = std::size_t(sp.nY);
        ev.Fy.assign(nY, std::vector<double>(nFull, 0.0));
        for (std::size_t k = 0; k < w; ++k) {
            std::vector<double>& to = k < 5 ? ev.F[k] : ev.Fy[k - 5];
            for (std::size_t f = 0; f < std::size_t(nI); ++f) to[f] = t[w * f + k];
            for (std::size_t b = 0; b < std::size_t(nB); ++b)
                to[std::size_t(nI) + std::size_t(bndFace[b])] = t[w * (std::size_t(nI) + b) + k];
        }
        ev.bnd.assign(bndFull, FvPrim<double>{});
        for (auto& p : ev.bphi) p.assign(bndFull, 0.0);
        if (nB) {
            if (auto r = fetch(B, t, "boundary states"); !r) return r;
            for (std::size_t b = 0; b < std::size_t(nB); ++b) {
                const std::size_t i = 7 * b, j = std::size_t(bndFace[b]);
                ev.bnd[j] = {t[i], {t[i + 1], t[i + 2], t[i + 3]}, t[i + 4], t[i + 5], t[i + 6]};
                ev.bphi[0][j] = t[i + 1];
                ev.bphi[1][j] = t[i + 2];
                ev.bphi[2][j] = t[i + 3];
                ev.bphi[3][j] = t[i + 6];
            }
        }
        ev.Y.assign(nY, std::vector<double>(n, 0.0));
        ev.gradY.assign(3 * nY, std::vector<double>(n, 0.0));
        ev.bY.assign(nY, std::vector<double>(bndFull, 0.0));
        ev.omega.clear();
        if (nY) {
            if (auto r = fetch(om, ev.omega, "omega"); !r) return r;
            if (auto r = fetch(Y, t, "species"); !r) return r;
            for (std::size_t s = 0; s < nY; ++s)
                for (std::size_t c = 0; c < n; ++c) ev.Y[s][c] = t[nY * c + s];
            if (auto r = fetch(GY, t, "species gradients"); !r) return r;
            for (std::size_t k = 0; k < 3 * nY; ++k)
                for (std::size_t c = 0; c < n; ++c) ev.gradY[k][c] = t[3 * nY * c + k];
            if (auto r = fetch(Yb, t, "boundary species"); !r) return r;
            for (std::size_t s = 0; s < nY; ++s)
                for (std::size_t b = 0; b < std::size_t(nB); ++b) ev.bY[s][std::size_t(bndFace[b])] = t[nY * b + s];
        }
        if (les)
            if (auto r = fetch(nut, ev.nut, "eddy viscosity"); !r) return r;
        ev.sigma.clear();
        if (shockOn) {
            ev.sigma.resize(n);
            if (const hipError_t e = sb.download(ev.sigma.data()); e != hipSuccess) return std::unexpected(hipErr("sensor", e));
        }
        if (thick) {
            if (auto r = fetch(fe, ev.fe, "TFLES F E"); !r) return r;
            if (auto r = fetch(eof, ev.eof, "TFLES E/F"); !r) return r;
            if (auto r = fetch(omT, ev.om, "TFLES sensor"); !r) return r;
        }
        if (!rans) return {};
        for (auto& v : ev.tfld) v.assign(n, 0.0);
        for (auto& v : ev.tbnd) v.assign(bndFull, 0.0);
        for (auto& v : ev.Ft) v.assign(nFull, 0.0);
        for (auto& v : ev.tgrad) v.assign(n, 0.0);
        for (auto& v : ev.tsrc) v.assign(n, 0.0);
        if (auto r = fetch(Ph, t, "scalars"); !r) return r;
        for (std::size_t k = 0; k < 2; ++k)
            for (std::size_t c = 0; c < n; ++c) ev.tfld[k][c] = t[2 * c + k];
        if (auto r = fetch(Gh, t, "scalar gradients"); !r) return r;
        for (std::size_t k = 0; k < 6; ++k)
            for (std::size_t c = 0; c < n; ++c) ev.tgrad[k][c] = t[6 * c + k];
        if (auto r = fetch(Sr, t, "scalar sources"); !r) return r;
        for (std::size_t k = 0; k < 4; ++k)
            for (std::size_t c = 0; c < n; ++c) ev.tsrc[k][c] = t[4 * c + k];
        if (auto r = fetch(nut, ev.nut, "eddy viscosity"); !r) return r;
        if (sst) {
            if (auto r = fetch(f1, ev.f1, "F1"); !r) return r;
        } else {
            ev.f1.assign(n, 0.0);
        }
        if (auto r = fetch(Ft, t, "scalar fluxes"); !r) return r;
        for (std::size_t k = 0; k < std::size_t(nT); ++k) {
            for (std::size_t f = 0; f < std::size_t(nI); ++f) ev.Ft[k][f] = t[2 * f + k];
            for (std::size_t b = 0; b < std::size_t(nB); ++b)
                ev.Ft[k][std::size_t(nI) + std::size_t(bndFace[b])] = t[2 * (std::size_t(nI) + b) + k];
        }
        if (nB) {
            if (auto r = fetch(Bh, t, "scalar boundary"); !r) return r;
            for (std::size_t b = 0; b < std::size_t(nB); ++b)
                for (std::size_t k = 0; k < 2; ++k) ev.tbnd[k][std::size_t(bndFace[b])] = t[2 * b + k];
        }
        return {};
    }

    std::expected<double, std::string> chemDt(double k) override
    {
        if (!sp.nY || !std::isfinite(chem.lnA)) return HUGE_VAL;
        if (chem.ox >= 0 && chem.law.n > 0) {
            const double J2 = internal::reduce(red, std::size_t(nC),
                                               ChemJ2Phi<R>{q.data(), std::size_t(nC), sp.iY + sp.fuel, sp.iY + sp.ox, gamma,
                                                            Rgas, chem.lnA, chem.Ta, chem.q, Rgas / (gamma - 1.0), chem.stoich,
                                                            chem.law},
                                               rocprim::maximum<double>(), 0.0);
            if (auto c = check("chemDt"); !c) return std::unexpected(c.error());
            return J2 > 0.0 ? k / J2 : HUGE_VAL;
        }
        if (chem.ox >= 0) {
            const double J2 = internal::reduce(red, std::size_t(nC),
                                               ChemJ2<R>{q.data(), std::size_t(nC), sp.iY + sp.fuel, sp.iY + sp.ox, gamma, Rgas,
                                                         chem.lnA, chem.Ta, chem.q, Rgas / (gamma - 1.0), chem.stoich},
                                               rocprim::maximum<double>(), 0.0);
            if (auto c = check("chemDt"); !c) return std::unexpected(c.error());
            return J2 > 0.0 ? k / J2 : HUGE_VAL;
        }
        const double J = internal::reduce(red, std::size_t(nC),
                                          ChemJ<R>{q.data(), std::size_t(nC), sp.iY + sp.fuel, gamma, Rgas, chem.lnA, chem.Ta,
                                                   chem.q, Rgas / (gamma - 1.0)},
                                          rocprim::maximum<double>(), 0.0);
        if (auto c = check("chemDt"); !c) return std::unexpected(c.error());
        return J > 0.0 ? k / J : HUGE_VAL;
    }

    std::expected<void, std::string> speciesFields(dns::FvState& s, std::vector<double>& omega) override
    {
        omega.clear();
        if (auto r = download(s); !r) return r;
        if (!sp.nY) return {};
        if (thick) tflesUpdate(q.data()); // omega is (E/F) omega of the current state's E, F
        primsOf(q.data());
        speciesCells(q.data());
        if (auto c = check("species fields"); !c) return c;
        return fetch(om, omega, "omega");
    }

    std::expected<std::vector<double>, std::string> patchFlux(int row) override
    {
        if (shockOn) shockUpdate(q.data());
        rhsPasses(q.data());
        if (auto c = check("patch flux"); !c) return std::unexpected(c.error());
        std::vector<double> t, o(bcHost.size(), 0.0);
        if (auto r = fetch(F, t, "fluxes"); !r) return std::unexpected(r.error());
        for (std::size_t b = 0; b < std::size_t(nB); ++b)
            o[std::size_t(bndPatch[b])] += t[std::size_t(nV) * (std::size_t(nI) + b) + std::size_t(row)];
        return o;
    }

    std::expected<double, std::string> stableDt(FvDtLimits lim) override
    {
        if (thick) tflesUpdate(q.data());
        primsOf(q.data());
        gradsAndNut(q.data());
        const unsigned bc = blocksOf(std::size_t(nC));
        if (thick && les)
            kFvDt<R, true, 0, true><<<bc, kT>>>(view(), gas, lim.cfl, lim.diff, fMol, fEddy, P.data(), nut.data(), Ph.data(),
                                                entS.data(), diffC.data(), dtc.data(), fe.data(), gPr);
        else if (thick)
            kFvDt<R, false, 0, true><<<bc, kT>>>(view(), gas, lim.cfl, lim.diff, fMol, fEddy, P.data(), nut.data(), Ph.data(),
                                                 entS.data(), diffC.data(), dtc.data(), fe.data(), gPr);
        else if (sst)
            kFvDt<R, true, 2><<<bc, kT>>>(view(), gas, lim.cfl, lim.diff, fMol, fEddy, P.data(), nut.data(), Ph.data(),
                                          entS.data(), diffC.data(), dtc.data());
        else if (rans)
            kFvDt<R, true, 1><<<bc, kT>>>(view(), gas, lim.cfl, lim.diff, fMol, fEddy, P.data(), nut.data(), Ph.data(),
                                          entS.data(), diffC.data(), dtc.data());
        else if (les)
            kFvDt<R, true, 0><<<bc, kT>>>(view(), gas, lim.cfl, lim.diff, fMol, fEddy, P.data(), nut.data(), Ph.data(),
                                          entS.data(), diffC.data(), dtc.data());
        else
            kFvDt<R, false, 0><<<bc, kT>>>(view(), gas, lim.cfl, lim.diff, fMol, fEddy, P.data(), nut.data(), Ph.data(),
                                           entS.data(), diffC.data(), dtc.data());
        const double dt = internal::reduce(red, std::size_t(nC), MinCell<R>{dtc.data()}, rocprim::minimum<double>(),
                                           1e300);
        if (auto c = check("stableDt"); !c) return std::unexpected(c.error());
        if (dt < 0.0) {
            std::vector<double> d, p;
            if (auto r = fetch(dtc, d, "dt"); !r) return std::unexpected(r.error());
            if (auto r = fetch(P, p, "primitives"); !r) return std::unexpected(r.error());
            const auto it = std::ranges::find_if(d, [](double v) { return v < 0.0; });
            const std::size_t c = std::size_t(it - d.begin());
            return std::unexpected(std::format("FV state diverged: rho {} p {} at cell {}", p[7 * c], p[7 * c + 5], c));
        }
        if (!(dt > 0.0) || !std::isfinite(dt)) return std::unexpected("FV stable dt is not positive-finite");
        return dt;
    }

    std::expected<double, std::string> localDt(FvDtLimits lim) override
    {
        // The time axis of the inventory: kFvChem, kFvGravity take the bare cr, LES is time-accurate.
        FvModuleSet withLts = mods;
        withLts.set(std::size_t(FvModule::Steady));
        if (auto ok = fvModulesCheck(withLts); !ok) return std::unexpected(std::move(ok.error()));
        auto d = stableDt(lim);
        if (!d) return d;
        if (lts.size() == 0) {
            auto b = sized<R>(std::size_t(nC), "local dt");
            if (!b) return std::unexpected(std::move(b.error()));
            lts = std::move(*b);
        }
        kFvLts<R><<<blocksOf(std::size_t(nC)), kT>>>(nC, *d, dtc.data(), lts.data());
        useLts = true;
        return d;
    }

    std::expected<FvSurface, std::string> surface() override
    {
        FvSurface s;
        if (!nTally) return s;
        primsOf(q.data());
        gradsAndNut(q.data());
        if (les || rans)
            kFvTally<R, true><<<blocksOf(std::size_t(nTally)), kT>>>(view(), gas, nTally, tallyList.data(), pRef, P.data(),
                                                                     B.data(), G.data(), nut.data(), tally.data());
        else
            kFvTally<R, false><<<blocksOf(std::size_t(nTally)), kT>>>(view(), gas, nTally, tallyList.data(), pRef,
                                                                      P.data(), B.data(), G.data(), nut.data(),
                                                                      tally.data());
        if (auto c = check("surface tally"); !c) return std::unexpected(c.error());
        std::vector<double> t;
        if (auto r = fetch(tally, t, "surface tally"); !r) return std::unexpected(r.error());
        s.heatPatch.assign(bcHost.size(), 0.0);
        for (std::size_t i = 0; i < std::size_t(nTally); ++i) {
            for (std::size_t k = 0; k < 3; ++k) {
                s.fp[k] += t[kTally * i + k];
                s.fv[k] += t[kTally * i + 3 + k];
            }
            s.heat += t[kTally * i + 6];
            s.heatPatch[std::size_t(tallyPatch[i])] += t[kTally * i + 6];
        }
        s.area = tallyArea;
        return s;
    }

    // The bytes of the current state, host copy.
    std::expected<std::vector<std::uint8_t>, std::string> shockBytes()
    {
        if (!shockOn) return std::unexpected("FV sensor: no --shock");
        shockUpdate(q.data());
        if (auto c = check("sensor"); !c) return std::unexpected(c.error());
        std::vector<std::uint8_t> b;
        b.resize(std::size_t(nC));
        if (const hipError_t e = sb.download(b.data()); e != hipSuccess) return std::unexpected(hipErr("sensor", e));
        return b;
    }

    std::expected<long, std::string> shockCells() override
    {
        auto b = shockBytes();
        if (!b) return std::unexpected(b.error());
        return long(std::ranges::count_if(*b, [](std::uint8_t v) { return v != 0; }));
    }

    std::expected<int, std::string> shockByteMax() override
    {
        auto b = shockBytes();
        if (!b) return std::unexpected(b.error());
        return int(*std::ranges::max_element(*b));
    }

    std::expected<std::vector<double>, std::string> eddyViscosity() override
    {
        std::vector<double> o(std::size_t(nC), 0.0);
        if (!les && !rans) return o;
        primsOf(q.data());
        gradsAndNut(q.data());
        if (auto c = check("eddy viscosity"); !c) return std::unexpected(c.error());
        if (auto r = fetch(nut, o, "eddy viscosity"); !r) return std::unexpected(r.error());
        return o;
    }

    std::expected<DeviceFvComp::TflesStats, std::string> tflesStats() override
    {
        if (!thick) return std::unexpected("FV TFLES stats: no --tfles");
        tflesUpdate(q.data());
        if (auto c = check("TFLES stats"); !c) return std::unexpected(c.error());
        std::vector<double> f, e, w;
        if (auto r = fetch(fe, f, "TFLES F E"); !r) return std::unexpected(r.error());
        if (auto r = fetch(eof, e, "TFLES E/F"); !r) return std::unexpected(r.error());
        if (auto r = fetch(omT, w, "TFLES sensor"); !r) return std::unexpected(r.error());
        double so = 0.0, soe = 0.0, mx = 0.0;
        for (std::size_t c = 0; c < f.size(); ++c) {
            so += V[c] * w[c];
            soe += V[c] * w[c] * std::sqrt(f[c] * e[c]); // E = sqrt(FE E/F)
            mx = std::max(mx, f[c]);
        }
        return DeviceFvComp::TflesStats{.Ebar = so > 0.0 ? soe / so : NAN, .FEmax = mx};
    }

    std::expected<unsigned long long, std::string> floorHits() const override
    {
        if (hits.size() == 0) return 0ull;
        (void)hipDeviceSynchronize();
        unsigned long long h = 0;
        if (const hipError_t e = hits.download(&h); e != hipSuccess) return std::unexpected(hipErr("floor hits", e));
        return h;
    }

    // The carried words: q, the compensated l, the SST guard's count.
    std::expected<void, RunError> save(Checkpoint& ck) const override
    {
        if (auto e = internal::putWords(ck, "state.fv.q", q); !e) return e;
        if (compensated)
            if (auto e = internal::putWords(ck, "state.fv.lo", l); !e) return e;
        if (hits.size() == 0) return {};
        unsigned long long h = 0;
        if (hits.download(&h) != hipSuccess) return std::unexpected(RunError{6, "checkpoint: floor hits download failed"});
        ck.putI64("state.fv.hits", std::int64_t(h));
        return {};
    }

    std::expected<void, RunError> load(const Checkpoint& ck) override
    {
        if (auto e = internal::getWords(ck, "state.fv.q", q); !e) return e;
        if (compensated)
            if (auto e = internal::getWords(ck, "state.fv.lo", l); !e) return e;
        if (hits.size() == 0) return {};
        auto h = ck.i64Scalar("state.fv.hits");
        if (!h) return std::unexpected(RunError{3, "--restart: " + h.error()});
        const auto v = static_cast<unsigned long long>(*h);
        if (hits.upload(&v) != hipSuccess) return std::unexpected(RunError{6, "--restart: floor hits upload failed"});
        return {};
    }

    std::expected<void, std::string> cellFields(std::array<std::vector<double>, 7>& prim,
                                                std::array<std::vector<double>, 12>& grad) override
    {
        primsOf(q.data());
        kFvGrad<R><<<blocksOf(std::size_t(nC)), kT>>>(view(), P.data(), B.data(), G.data());
        if (auto c = check("cell fields"); !c) return c;
        std::vector<double> t;
        const std::size_t n = std::size_t(nC);
        if (auto r = fetch(P, t, "primitives"); !r) return r;
        for (std::size_t k = 0; k < 7; ++k) {
            prim[k].resize(n);
            for (std::size_t c = 0; c < n; ++c) prim[k][c] = t[7 * c + k];
        }
        if (auto r = fetch(G, t, "gradients"); !r) return r;
        for (std::size_t k = 0; k < 12; ++k) {
            grad[k].resize(n);
            for (std::size_t c = 0; c < n; ++c) grad[k][c] = t[12 * c + k];
        }
        return {};
    }

    void boost(const std::array<double, 3>& du) override
    {
        if (compensated) {
            kFvBoostPair<R><<<blocksOf(std::size_t(nC)), kT>>>(nC, R(du[0]), R(du[1]), R(du[2]), q.data(), l.data());
            return;
        }
        kFvBoost<R><<<blocksOf(std::size_t(nC)), kT>>>(nC, R(du[0]), R(du[1]), R(du[2]), q.data());
    }

    std::expected<void, std::string> setBc(int patch, const FvBc& bc) override
    {
        if (patch < 0 || std::size_t(patch) >= bcHost.size() || bcHost[std::size_t(patch)].kind != bc.kind)
            return std::unexpected("FV setBc: bad patch or kind change");
        bcHost[std::size_t(patch)] = bc;
        (void)hipDeviceSynchronize();
        if (const hipError_t e = bcs.upload(bcHost.data()); e != hipSuccess)
            return std::unexpected(hipErr("boundary table upload", e));
        return {};
    }

    std::expected<void, std::string> setBcRows(int first, std::span<const FvBc> rows) override
    {
        if (first < 0 || std::size_t(first) + rows.size() > bcHost.size())
            return std::unexpected("FV setBcRows: rows past the table");
        for (std::size_t i = 0; i < rows.size(); ++i)
            if (bcHost[std::size_t(first) + i].kind != rows[i].kind)
                return std::unexpected("FV setBcRows: a row changes its kind");
        if (rows.empty()) return {};
        std::ranges::copy(rows, bcHost.begin() + first);
        (void)hipDeviceSynchronize();
        if (const hipError_t e = hipMemcpy(bcs.data() + first, rows.data(), rows.size() * sizeof(FvBc), hipMemcpyHostToDevice);
            e != hipSuccess)
            return std::unexpected(hipErr("boundary rows upload", e));
        return {};
    }

    std::size_t bytes() const override
    {
        std::size_t s = face.size() * sizeof(FaceRow<R>) + bnd.size() * sizeof(BndRow<R>) +
                        bcs.size() * sizeof(FvBc) + gEnt.size() * sizeof(GradEnt<R>) +
                        (cStart.size() + gFace.size() + tallyList.size() + lineStart.size() + line.size() + fLine.size()) * sizeof(int) +
                        sb.size();
        for (const auto* b : {&Vd, &diffC, &cd, &entS, &q, &q1, &q2, &l, &l1, &l2, &P, &B, &G, &F, &Gf, &nut, &dtc, &tally, &Y,
                              &Yb, &GY, &om, &dw, &Ph, &Bh, &Gh, &f1, &Sr, &Ft, &lts, &fe, &eof, &omT, &tfA, &tfB, &curl, &uh,
                              &lapW, &hc, &coef})
            s += b->size() * sizeof(R);
        return s + walk.size() * sizeof(int) + edge.size();
    }

    std::size_t bndFull = 0;
    double tallyArea = 0;
    std::vector<int> patchFirst, patchActive; // per FaceMesh patch: first compact boundary face, count
};

template <class R>
std::expected<std::unique_ptr<DeviceFvComp::Impl>, std::string> makeImpl(const FaceMesh& fm, const FvCompConfig& cfg,
                                                                         int threads)
{
    const FvConfig rc{.tr = cfg.tr,
                      .bc = cfg.bc,
                      .bcRow = cfg.bcRow,
                      .gravity = cfg.gravity,
                      .ambient = cfg.ambient,
                      .chem = cfg.chem,
                      .rans = cfg.rans,
                      .bodyForce = cfg.bodyForce,
                      .les = {.cs = cfg.csLes, .sgs = cfg.sgs, .prt = cfg.prt},
                      .tfles = cfg.tfles,
                      .turb = cfg.turb,
                      .shock = cfg.shock};
    // The model pairs (dns/fvmodules): FvRef::make refuses each by name.
    auto ref = FvRef::make(fm, rc, threads);
    if (!ref) return std::unexpected(std::move(ref.error()));
    // The precision axis of the inventory.
    FvModuleSet mods = fvModulesOf(rc);
    mods.set(std::size_t(FvModule::Compensated), cfg.compensated);
    mods.set(std::size_t(FvModule::Fp64), !std::is_same_v<R, Real>);
    if (auto ok = fvModulesCheck(mods); !ok) return std::unexpected(std::move(ok.error()));
    const bool withRans = cfg.rans.model != RansModel::None;
    if (!cfg.tallied.empty() && cfg.tallied.size() != fm.patches.size())
        return std::unexpected("FV tally: one flag per patch required");
    const FvFaceGeom& fg = ref->faceGeom();
    const FvGradWeights& gw = ref->gradWeights();
    const std::size_t nCs = std::size_t(fm.nC), nIs = std::size_t(fm.nI), nBs = std::size_t(fm.nB);

    auto im = std::make_unique<FvCompImpl<R>>();
    FvCompImpl<R>& I = *im;
    I.nC = fm.nC;
    I.mods = mods;
    I.nI = fm.nI;
    I.nV = kFvMeanVars + cfg.chem.nY;
    I.chem = cfg.chem;
    I.sp = {.nY = cfg.chem.nY,
            .iY = kFvMeanVars,
            .fuel = cfg.chem.fuel,
            .ox = cfg.chem.ox,
            .lnA = R(cfg.chem.lnA),
            .Ta = R(cfg.chem.Ta),
            .q = R(cfg.chem.q),
            .invPr = R(cfg.chem.invPr),
            .stoich = R(cfg.chem.stoich),
            .invSct = R(1.0 / cfg.prt),
            .kapLes = R(cfg.tr.cp() / cfg.prt),
            .les = cfg.csLes > 0.0 ? 1 : 0};
    I.thick = cfg.tfles.has_value();
    if (I.thick) {
        const TflesSetup& t = *cfg.tfles;
        I.tf = {.iF = kFvMeanVars + cfg.chem.fuel,
                .rad = t.dilate,
                .uniform = t.uniform ? 1 : 0,
                .charlette = t.charlette ? 1 : 0,
                .lnAs = R(t.lnAs),
                .TaS = R(t.TaS),
                .beta = R(t.sensorBeta),
                .Fmax = R(t.F),
                .deltaL = R(t.deltaL),
                .nuU = R(t.nuU),
                .effBeta = R(t.effBeta)};
        I.nAx = ref->tflesWalk().nAxes;
        I.gPr = R(cfg.tr.gamma / cfg.tr.Pr);
    }
    I.gamma = cfg.tr.gamma;
    I.Rgas = cfg.tr.Rgas;
    I.precision = cfg.precision;
    I.bcHost = cfg.bc;
    I.bndFull = nBs;
    I.les = cfg.csLes > 0.0;
    I.compensated = cfg.compensated;
    I.grav = cfg.gravity;
    I.hasG = std::ranges::any_of(cfg.gravity, [](double v) { return v != 0.0; });
    I.visc = cfg.tr.muRef > 0.0 || I.les;
    I.rans = withRans;
    I.ransModel = cfg.rans.model;
    I.sst = ransSst(cfg.rans.model);
    I.ransOnly = cfg.rans.transportOnly;
    I.nT = ransScalars(cfg.rans.model);
    I.nState = I.nV + I.nT;
    I.shockOn = cfg.shock.mode != ShockMode::None;
    I.shockFlux = cfg.shock.mode == ShockMode::Hll;
    I.shockForce = cfg.shock.force;
    I.eps4 = R(cfg.shock.eps4);
    I.phiInf[0] = R(cfg.rans.phiInf[0]);
    I.phiInf[1] = R(cfg.rans.phiInf[1]);
    I.omegaFloor = R(cfg.rans.omegaFloor);
    for (std::size_t k = 0; k < 3; ++k) {
        I.fb[k] = R(cfg.bodyForce[k]);
        if (cfg.bodyForce[k] != 0.0) I.force = true;
    }
    I.sigma = cfg.sgs == SgsModel::Sigma ? 1 : 0;
    I.muRef = cfg.tr.muRef;
    I.pRef = R(cfg.pRef);
    I.fMol = R(fvDiffusivityFactor(cfg.tr.gamma, cfg.tr.Pr));
    const double prt = withRans ? kPrandtlTurb : cfg.prt;
    I.fEddy = R(fvDiffusivityFactor(cfg.tr.gamma, prt));
    I.V = fm.V;
    for (int k = 0; k < 3; ++k) I.C[k] = fm.C[std::size_t(k)];
    I.gas.tr = cfg.tr;
    fvTransportAt(cfg.tr, R(cfg.tr.Tref), I.gas.mu, I.gas.kappa);
    I.gas.kapEddy = R(cfg.tr.cp() / prt);

    // Active faces: every internal face, the non-empty boundary faces. Compact
    // numbering keeps FaceMesh's order, so a cell's CSR stays ascending.
    std::vector<int> compact(nBs, -1);
    for (std::size_t b = 0; b < nBs; ++b)
        if (fg.active[nIs + b]) {
            compact[b] = int(I.bndFace.size());
            I.bndFace.push_back(int(b));
        }
    I.nB = int(I.bndFace.size());
    I.nF = I.nI + I.nB;
    // The ledger's per-patch ranges: each patch's active boundary faces, contiguous in compact order.
    I.patchActive.assign(fm.patches.size(), 0);
    I.patchFirst.assign(fm.patches.size(), -1);
    for (std::size_t p = 0; p < fm.patches.size(); ++p)
        for (int b = fm.patches[p].first; b < fm.patches[p].first + fm.patches[p].count; ++b) {
            if (compact[std::size_t(b)] < 0) continue;
            if (I.patchFirst[p] < 0) I.patchFirst[p] = compact[std::size_t(b)];
            ++I.patchActive[p];
        }

    std::vector<FaceRow<R>> rows(nIs);
    for (std::size_t f = 0; f < nIs; ++f) {
        FaceRow<R>& r = rows[f];
        r.own = fm.own[f];
        r.nei = fm.nei[f];
        for (std::size_t k = 0; k < 3; ++k) {
            r.S[k] = R(fm.S[k][f]);
            r.d[k] = R(fm.d[k][f]);
            r.corr[k] = R(fg.corr[k][f]);
        }
    }
    std::vector<BndRow<R>> brows(I.bndFace.size());
    for (std::size_t i = 0; i < brows.size(); ++i) {
        const std::size_t b = std::size_t(I.bndFace[i]), f = nIs + b;
        BndRow<R>& r = brows[i];
        r.own = fm.own[f];
        r.bc = ref->bcOf()[b];
        for (std::size_t k = 0; k < 3; ++k) {
            r.S[k] = R(fm.S[k][f]);
            r.Cf[k] = R(fm.Cf[k][f]);
            r.d[k] = R(fm.d[k][f]);
            r.corr[k] = R(fg.corr[k][f]);
        }
    }

    // CSR without the inactive faces; per entry the gather code, the gradient
    // operand and the area vector of the dt sum.
    std::vector<int> start(nCs + 1, 0), code;
    std::vector<GradEnt<R>> ents;
    std::vector<R> entS;
    code.reserve(fm.cellFace.size());
    ents.reserve(fm.cellFace.size());
    entS.reserve(3 * fm.cellFace.size());
    for (std::size_t c = 0; c < nCs; ++c) {
        for (int q = fm.cellStart[c]; q < fm.cellStart[c + 1]; ++q) {
            const std::size_t f = std::size_t(fm.cellFace[std::size_t(q)]);
            if (!fg.active[f]) continue;
            const bool own = std::size_t(fm.own[f]) == c;
            const bool internal = f < nIs;
            const int cf = internal ? int(f) : I.nI + compact[f - nIs];
            code.push_back((cf << 1) | (own ? 0 : 1));
            GradEnt<R> e;
            e.other = internal ? (own ? fm.nei[f] : fm.own[f]) : ~compact[f - nIs];
            const auto& w = own ? gw.own : gw.nei;
            const bool side = !internal && cfg.bc[std::size_t(ref->bcOf()[f - nIs])].wedge; // out of the dt sum
            for (std::size_t k = 0; k < 3; ++k) {
                e.w[k] = R(w[k][f]);
                entS.push_back(side ? R(0.0) : R(fm.S[k][f]));
            }
            ents.push_back(e);
        }
        start[c + 1] = int(code.size());
    }
    I.nEnt = int(code.size());

    // Tally list: ascending compact boundary faces.
    std::vector<int> tlist;
    for (std::size_t i = 0; i < I.bndFace.size(); ++i) {
        const std::size_t b = std::size_t(I.bndFace[i]), p = std::size_t(ref->patchOf()[b]);
        I.bndPatch.push_back(int(p));
        const bool on = cfg.tallied.empty() ? cfg.bc[p].kind == FvBcKind::Wall : cfg.tallied[p] != 0;
        if (!on) continue;
        if (cfg.bc[p].kind != FvBcKind::Wall && cfg.bc[p].kind != FvBcKind::Slip)
            return std::unexpected(std::format("patch `{}`: only wall and slip patches enter the force tally",
                                               fm.patches[p].name));
        tlist.push_back(int(i));
        I.tallyPatch.push_back(int(p));
        const std::size_t f = nIs + b;
        I.tallyArea += std::hypot(fm.S[0][f], fm.S[1][f], fm.S[2][f]);
    }
    I.nTally = int(tlist.size());

    std::vector<R> vr(nCs), dc(nCs), cdv(I.les ? nCs : 0);
    for (std::size_t c = 0; c < nCs; ++c) {
        vr[c] = R(fm.V[c]);
        dc[c] = R(ref->diffCoef()[c]);
        // kCompSmagNutCurv's C^2 curvDelta2(vol): fvLesDelta2, FvRef's.
        if (I.les) cdv[c] = R(cfg.csLes * cfg.csLes * ref->lesDelta2()[c]);
    }
    // TFLES: the walk, edge layers, Laplacian weights per compact CSR entry (FvRef's
    // per FaceMesh entry, the inactive faces dropped), h = sqrt Delta^2, c2 h^3.
    std::vector<R> lw, hcv, cov;
    if (I.thick) {
        const FvTflesWalk& wk = ref->tflesWalk();
        for (std::size_t c = 0; c < nCs; ++c)
            for (int q = fm.cellStart[c]; q < fm.cellStart[c + 1]; ++q)
                if (fg.active[std::size_t(fm.cellFace[std::size_t(q)])]) lw.push_back(R(wk.lapW[std::size_t(q)]));
        for (std::size_t c = 0; c < nCs; ++c) {
            const double h = std::sqrt(ref->lesDelta2()[c]);
            hcv.push_back(R(h));
            cov.push_back(R(kColinC2 * h * h * h));
        }
    }

    const std::size_t nFs = std::size_t(I.nF), nV = std::size_t(I.nV), nS = std::size_t(I.nState);
    std::optional<std::string> err;
    auto take = [&](auto& dst, auto&& made) {
        if (!made) {
            if (!err) err = std::move(made.error());
            return;
        }
        dst = std::move(*made);
    };
    take(I.face, uploaded(rows, "face table"));
    take(I.bnd, uploaded(brows, "boundary table"));
    take(I.bcs, uploaded(cfg.bc, "boundary conditions"));
    take(I.Vd, uploaded(vr, "volumes"));
    take(I.diffC, uploaded(dc, "dt table"));
    take(I.cd, uploaded(cdv, "LES table"));
    take(I.entS, uploaded(entS, "dt table"));
    take(I.cStart, uploaded(start, "CSR"));
    take(I.gFace, uploaded(code, "CSR"));
    take(I.gEnt, uploaded(ents, "CSR"));
    take(I.tallyList, uploaded(tlist, "tally list"));
    take(I.q, sized<R>(nS * nCs, "state"));
    take(I.q1, sized<R>(nS * nCs, "state"));
    take(I.q2, sized<R>(nS * nCs, "state"));
    if (I.compensated) {
        take(I.l, sized<R>(nS * nCs, "state"));
        take(I.l1, sized<R>(nS * nCs, "state"));
        take(I.l2, sized<R>(nS * nCs, "state"));
    }
    take(I.P, sized<R>(std::size_t(kPrim) * nCs, "primitives"));
    take(I.B, sized<R>(std::size_t(kPrim) * std::size_t(I.nB), "boundary states"));
    take(I.G, sized<R>(std::size_t(kGrad) * nCs, "gradients"));
    take(I.F, sized<R>(nV * nFs, "fluxes"));
    take(I.Gf, sized<R>(I.hasG ? std::size_t(4) * nIs : 1, "gravity"));
    take(I.nut, sized<R>(I.les || I.rans ? nCs : 1, "eddy viscosity"));
    if (I.rans) {
        const std::size_t nf2 = std::size_t(kFvRansFields);
        std::vector<R> dr(nCs);
        std::ranges::transform(cfg.rans.wallDist, dr.begin(), [](double v) { return R(v); });
        take(I.dw, uploaded(dr, "wall distance"));
        take(I.Ph, sized<R>(nf2 * nCs, "scalar fields"));
        take(I.Bh, sized<R>(nf2 * std::size_t(I.nB), "scalar fields"));
        take(I.Gh, sized<R>(std::size_t(kRansGrad) * nCs, "scalar gradients"));
        take(I.f1, sized<R>(I.sst ? nCs : 1, "F1"));
        take(I.Sr, sized<R>(std::size_t(kRansSrc) * nCs, "scalar sources"));
        take(I.Ft, sized<R>(nf2 * nFs, "scalar fluxes"));
        take(I.hits, uploaded(std::vector<unsigned long long>{0ull}, "floor hits"));
    }
    take(I.dtc, sized<R>(nCs, "dt"));
    take(I.tally, sized<R>(std::size_t(kTally) * tlist.size(), "tally"));
    const std::size_t nYs = std::size_t(cfg.chem.nY);
    take(I.Y, sized<R>(nYs * nCs, "species"));
    take(I.Yb, sized<R>(nYs * std::size_t(I.nB), "species"));
    take(I.GY, sized<R>(3 * nYs * nCs, "species"));
    take(I.om, sized<R>(nYs ? nCs : 1, "species"));
    if (I.thick) {
        const FvTflesWalk& wk = ref->tflesWalk();
        const bool tc = cfg.tfles->charlette;
        for (DeviceBuffer<R>* b : {&I.fe, &I.eof, &I.omT, &I.tfA, &I.tfB, &I.uh}) take(*b, sized<R>(nCs, "TFLES fields"));
        take(I.curl, sized<R>(tc ? 3 * nCs : 1, "TFLES curl"));
        take(I.walk, uploaded(wk.next, "TFLES walk"));
        take(I.edge, uploaded(wk.edge, "TFLES edge layers"));
        take(I.lapW, uploaded(tc ? lw : std::vector<R>{}, "TFLES Laplacian"));
        take(I.hc, uploaded(hcv, "TFLES h"));
        take(I.coef, uploaded(cov, "TFLES Colin"));
    }
    if (I.shockOn) {
        const FvShockLines& sl = ref->shockLines();
        take(I.sb, sized<std::uint8_t>(nCs, "sensor"));
        take(I.lineStart, uploaded(sl.lineStart, "sensor lines"));
        take(I.line, uploaded(sl.line, "sensor lines"));
        take(I.fLine, uploaded(sl.faceLine, "sensor lines"));
    }
    if (err) return std::unexpected(std::move(*err));
    // A uniform sensor without the efficiency is constant: formed here, once.
    if (I.thick && cfg.tfles->uniform && !cfg.tfles->charlette)
        kFvTflesCombine<R><<<blocksOf(nCs), kT>>>(I.nC, I.tf, nullptr, I.uh.data(), I.hc.data(), I.edge.data(), I.fe.data(),
                                                  I.eof.data(), I.omT.data());
    auto rs = ReduceScratch::make();
    if (!rs) return std::unexpected(hipErr("reduction scratch", rs.error()));
    I.red = std::move(*rs);
    return im;
}

} // namespace

DeviceFvComp::DeviceFvComp() = default;
DeviceFvComp::DeviceFvComp(DeviceFvComp&&) noexcept = default;
DeviceFvComp& DeviceFvComp::operator=(DeviceFvComp&&) noexcept = default;
DeviceFvComp::~DeviceFvComp() = default;

std::expected<DeviceFvComp, std::string> DeviceFvComp::make(const FaceMesh& fm, const FvCompConfig& cfg, int threads)
{
    auto im = cfg.precision == FvPrecision::Fp64 ? makeImpl<double>(fm, cfg, threads) : makeImpl<float>(fm, cfg, threads);
    if (!im) return std::unexpected(std::move(im.error()));
    DeviceFvComp d;
    d.p_ = std::move(*im);
    return d;
}

std::expected<void, std::string> DeviceFvComp::upload(const dns::FvState& s) { return p_->upload(s); }
std::expected<void, std::string> DeviceFvComp::download(dns::FvState& s) const { return p_->download(s); }
std::expected<void, std::string> DeviceFvComp::rhs(dns::FvState& out) { return p_->rhs(out); }
std::expected<void, std::string> DeviceFvComp::evaluate(FvEval& ev, dns::FvState& out) { return p_->evaluate(ev, out); }
void DeviceFvComp::step(double dt) { p_->step(dt); }
std::expected<double, std::string> DeviceFvComp::localDt(FvDtLimits lim) { return p_->localDt(lim); }
std::expected<double, std::string> DeviceFvComp::stableDt(FvDtLimits lim) { return p_->stableDt(lim); }
std::expected<double, std::string> DeviceFvComp::chemDt(double k) { return p_->chemDt(k); }
std::expected<void, std::string> DeviceFvComp::speciesFields(dns::FvState& s, std::vector<double>& omega)
{
    return p_->speciesFields(s, omega);
}
std::expected<std::vector<double>, std::string> DeviceFvComp::patchFlux(int row) { return p_->patchFlux(row); }
std::expected<FvSurface, std::string> DeviceFvComp::surface() { return p_->surface(); }
std::expected<std::vector<double>, std::string> DeviceFvComp::eddyViscosity() { return p_->eddyViscosity(); }
std::expected<DeviceFvComp::TflesStats, std::string> DeviceFvComp::tflesStats() { return p_->tflesStats(); }
std::expected<unsigned long long, std::string> DeviceFvComp::floorHits() const { return p_->floorHits(); }
std::expected<long, std::string> DeviceFvComp::shockCells() { return p_->shockCells(); }
std::expected<int, std::string> DeviceFvComp::shockByteMax() { return p_->shockByteMax(); }
std::expected<void, std::string> DeviceFvComp::cellFields(std::array<std::vector<double>, 7>& prim,
                                                          std::array<std::vector<double>, 12>& grad)
{
    return p_->cellFields(prim, grad);
}
void DeviceFvComp::boost(const std::array<double, 3>& du) { p_->boost(du); }
std::expected<void, std::string> DeviceFvComp::setBc(int patch, const FvBc& bc) { return p_->setBc(patch, bc); }
std::expected<void, RunError> DeviceFvComp::save(Checkpoint& ck) const { return p_->save(ck); }
std::expected<void, RunError> DeviceFvComp::load(const Checkpoint& ck) { return p_->load(ck); }
std::expected<void, std::string> DeviceFvComp::setBcRows(int first, std::span<const FvBc> rows)
{
    return p_->setBcRows(first, rows);
}
std::expected<void, std::string> DeviceFvComp::setLedger(bool on, bool equalWeights) { return p_->setLedger(on, equalWeights); }
std::expected<FvLedger, std::string> DeviceFvComp::ledger() const { return p_->ledger(); }
int DeviceFvComp::cells() const { return p_->nC; }
FvPrecision DeviceFvComp::precision() const { return p_->precision; }
std::size_t DeviceFvComp::deviceBytes() const { return p_->bytes(); }

std::expected<FvTotals, std::string> DeviceFvComp::totals() const
{
    dns::FvState s;
    if (auto r = p_->download(s); !r) return std::unexpected(r.error());
    const std::size_t n = std::size_t(s.nC);
    const double* q = s.q.data();
    FvTotals t;
    for (std::size_t c = 0; c < n; ++c) {
        const double V = p_->V[c], rho = q[c], mx = q[n + c], my = q[2 * n + c], mz = q[3 * n + c], E = q[4 * n + c];
        t.mass += rho * V;
        t.mom[0] += mx * V;
        t.mom[1] += my * V;
        t.mom[2] += mz * V;
        t.energy += E * V;
        const double ke = 0.5 * (mx * mx + my * my + mz * mz) / rho;
        t.kinetic += ke * V;
        const double p = (p_->gamma - 1.0) * (E - ke);
        t.machMax = std::max(t.machMax, std::sqrt(2.0 * ke / rho / (p_->gamma * p / rho)));
    }
    return t;
}

} // namespace dns::gpu
