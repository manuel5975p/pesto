#include "dns/gpu/fv_incomp.hpp"

#include "dns/chem.hpp"
#include "dns/fvflux.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/internal/ckpt_words.hpp"
#include "dns/gpu/internal/fv_device.hpp"
#include "dns/gpu/internal/stage_write.hpp"
#include "dns/gpu/la/amg.hpp"
#include "dns/gpu/la/blockell.hpp"
#include "dns/gpu/la/internal/dots.hpp"
#include "dns/gpu/la/krylov.hpp"
#include "dns/la/amg.hpp"
#include "dns/la/ell.hpp"
#include "dns/sgs.hpp"
#include "dns/tfles.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <type_traits>
#include <utility>

namespace dns::gpu {

using namespace fvd;

namespace {

// A launch of n items (n == 0, a mesh with no boundary: one block that exits).
inline unsigned grid1(std::size_t n) { return std::max(1u, grid(n)); }

constexpr u8 kWall = u8(FvIncompFaceBc::Wall), kSlip = u8(FvIncompFaceBc::Slip), kInflow = u8(FvIncompFaceBc::Inflow),
             kOutflow = u8(FvIncompFaceBc::Outflow), kConvective = u8(FvIncompFaceBc::Convective);

// Device mesh view; layouts as FvIncompSetup.
template <class R>
struct IMesh
{
    int nC = 0, nI = 0, nB = 0;
    const int *own = nullptr, *nei = nullptr, *bOwn = nullptr, *cStart = nullptr, *cEnt = nullptr;
    const R *S = nullptr, *d = nullptr, *rO = nullptr, *id2 = nullptr, *cS = nullptr, *wO = nullptr, *a = nullptr,
            *k = nullptr;
    const R *bS = nullptr, *bR = nullptr, *bN = nullptr, *bCs = nullptr, *bA = nullptr, *bDn = nullptr, *bMagS = nullptr,
            *bU = nullptr, *bP = nullptr, *bUc = nullptr;
    const R *V = nullptr, *gO = nullptr, *gN = nullptr;
    const u8 *bCode = nullptr, *bForce = nullptr, *bTfix = nullptr;
    const R* bT = nullptr;
};

// CSR entry -> internal face and its sign seen from the cell.
__device__ inline std::size_t faceOf(int code) { return std::size_t(code >> 1); }

template <class R>
__device__ inline R signOf(int code)
{
    return (code & 1) ? R(-1.0) : R(1.0);
}

// Least-squares cell gradient on differences: G[(3 i + q) nC + c] = d_i u_q.
template <class R>
__global__ void __launch_bounds__(256) kGrad(IMesh<R> m, const R* U, const R* Ub, R* G)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB), cc = std::size_t(c);
    const std::size_t nF = nI + nB;
    R g[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    R u[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) u[q] = U[std::size_t(q) * nC + cc];
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        R w[3], dq[3];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            const bool isNei = code & 1;
            const std::size_t other = std::size_t(isNei ? m.own[f] : m.nei[f]);
#pragma unroll
            for (int i = 0; i < 3; ++i) w[i] = isNei ? m.gN[std::size_t(i) * nI + f] : m.gO[std::size_t(i) * nF + f];
#pragma unroll
            for (int q = 0; q < 3; ++q) dq[q] = U[std::size_t(q) * nC + other] - u[q];
        } else {
            const std::size_t b = std::size_t(-code - 1);
#pragma unroll
            for (int i = 0; i < 3; ++i) w[i] = m.gO[std::size_t(i) * nF + nI + b];
#pragma unroll
            for (int q = 0; q < 3; ++q) dq[q] = Ub[std::size_t(q) * nB + b] - u[q];
        }
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int q = 0; q < 3; ++q) g[i][q] += w[i] * dq[q];
    }
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int q = 0; q < 3; ++q) G[std::size_t(3 * i + q) * nC + cc] = g[i][q];
}

// Least-squares gradient of p, GL[q nC + c]: a Neumann face has zero difference,
// a fixed-p face enters with its value. Feeds the deferred k . grad p only: a
// gradient from fp itself would be a recursion that diverges past ~35 deg.
// Hydro: a Neumann face's difference is dpB, the wall's hydrostatic image.
template <class R, bool Hydro = false>
__global__ void __launch_bounds__(256) kGradP(IMesh<R> m, const R* p, const R* dpB, R* GL)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    const std::size_t nF = nI + nB;
    const R pc = p[c];
    R g[3] = {0, 0, 0};
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            const bool isNei = code & 1;
            const R dp = p[isNei ? m.own[f] : m.nei[f]] - pc;
#pragma unroll
            for (int i = 0; i < 3; ++i)
                g[i] += (isNei ? m.gN[std::size_t(i) * nI + f] : m.gO[std::size_t(i) * nF + f]) * dp;
        } else {
            const std::size_t b = std::size_t(-code - 1);
            R dp;
            if (m.bCode[b] == kOutflow) dp = m.bP[b] - pc;
            else if constexpr (Hydro) dp = dpB[b];
            else continue;
#pragma unroll
            for (int i = 0; i < 3; ++i) g[i] += m.gO[std::size_t(i) * nF + nI + b] * dp;
        }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) GL[std::size_t(i) * nC + std::size_t(c)] = g[i];
}

// nut = (C Delta)^2 rate (dns/sgs.hpp), frozen over the stage.
template <class R>
__global__ void __launch_bounds__(256) kNut(int nC, R c2, int sigma, const R* G, const R* delta2, R* nut)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    const std::size_t n = std::size_t(nC), cc = std::size_t(c);
    R gd[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int q = 0; q < 3; ++q) gd[q][i] = G[std::size_t(3 * i + q) * n + cc];
    nut[c] = c2 * delta2[c] * sgsRate<R>(sigma, gd);
}

// Internal face flux of momentum, phi (u_o + u_n)/2 - viscous: Corr adds the
// AlongD correction; Les the eddy stress nut_f (grad u + grad u^T).S.
template <class R, bool Corr, bool Les>
__global__ void __launch_bounds__(256) kFace(IMesh<R> m, R nu, const R* U, const R* G, const R* phi, const R* nut, R* F)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    const R ph = phi[f], cs = m.cS[f];
    R uo[3], un[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        uo[q] = U[std::size_t(q) * nC + o];
        un[q] = U[std::size_t(q) * nC + n];
    }
    if constexpr (!Corr) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
            F[std::size_t(j) * nI + ff] = ph * (R(0.5) * (uo[j] + un[j])) - nu * (cs * (un[j] - uo[j]));
    } else {
        R S[3], d[3], gb[3][3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            S[i] = m.S[std::size_t(i) * nI + ff];
            d[i] = m.d[std::size_t(i) * nI + ff];
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                const std::size_t at = std::size_t(3 * i + q) * nC;
                gb[i][q] = R(0.5) * (G[at + o] + G[at + n]);
            }
        }
        R gf[3][3];
        R visc[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const R du = un[j] - uo[j];
            const R gdS = gb[0][j] * S[0] + gb[1][j] * S[1] + gb[2][j] * S[2];
            const R gdd = gb[0][j] * d[0] + gb[1][j] * d[1] + gb[2][j] * d[2];
            const R def = du - gdd;
            visc[j] = nu * (gdS + def * cs);
            if constexpr (Les) {
                const R c = def * m.id2[f];
#pragma unroll
                for (int i = 0; i < 3; ++i) gf[i][j] = gb[i][j] + c * d[i];
            }
        }
        if constexpr (Les) {
            const R nf = R(0.5) * (nut[o] + nut[n]);
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                R s = 0;
#pragma unroll
                for (int i = 0; i < 3; ++i) s += (gf[i][j] + gf[j][i]) * S[i];
                visc[j] += nf * s;
            }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j)
            F[std::size_t(j) * nI + ff] = ph * (R(0.5) * (uo[j] + un[j])) - visc[j];
    }
}

// The rest of the wall face's viscous stress, per nu: (dev2(T(grad u)))_b . S in
// component j, grad u_b = G_P with its normal part replaced by the one-sided
// (u_b - u_P)/dn: OpenFOAM's face value. The Laplacian alone left Re 40 cd 1.4%
// high on the tunnel mesh and its budget open (docs/fvincomp.md, "Wall stress").
template <class R>
__device__ inline R wallStressRest(const IMesh<R>& m, const R* U, const R* Ub, const R* G, std::size_t P, std::size_t b,
                                   int j)
{
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC);
    const R nj = m.bN[std::size_t(j) * nB + b];
    R r = 0, tr = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const R ni = m.bN[std::size_t(i) * nB + b];
        R ng = 0;
#pragma unroll
        for (int k = 0; k < 3; ++k) ng += m.bN[std::size_t(k) * nB + b] * G[std::size_t(3 * k + i) * nC + P];
        const R sn = (Ub[std::size_t(i) * nB + b] - U[std::size_t(i) * nC + P]) / m.bDn[b]; // one-sided d_n u_i
        r += ni * (G[std::size_t(3 * j + i) * nC + P] + nj * (sn - ng));
        tr += G[std::size_t(3 * i + i) * nC + P] + ni * (sn - ng);
    }
    return m.bMagS[b] * (r - R(2.0 / 3.0) * tr * nj);
}

// Boundary face flux of momentum: phi u_b, plus the viscous flux of a face that
// fixes u (wall, inflow; a slip face only its normal stress, u_b being reflected).
template <class R, bool Corr>
__global__ void __launch_bounds__(256) kBnd(IMesh<R> m, R nu, const R* U, const R* Ub, const R* G, const R* phiB, R* Fb)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const std::size_t P = std::size_t(m.bOwn[b]);
    const u8 code = m.bCode[b];
    const R ph = phiB[b], cs = m.bCs[b];
    const bool visc = code == kWall || code == kInflow || code == kSlip;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const R ub = Ub[std::size_t(j) * nB + bb], up = U[std::size_t(j) * nC + P];
        R v = 0;
        if (visc) {
            const R du = ub - up;
            if constexpr (Corr) {
                if (code == kSlip) v = du * cs;
                else {
                    R gS = 0, gr = 0;
#pragma unroll
                    for (int i = 0; i < 3; ++i) {
                        const R g = G[std::size_t(3 * i + j) * nC + P];
                        gS += g * m.bS[std::size_t(i) * nB + bb];
                        gr += g * m.bR[std::size_t(i) * nB + bb];
                    }
                    v = gS + (du - gr) * cs;
                    if (code == kWall) v += wallStressRest(m, U, Ub, G, P, bb, j);
                }
            } else
                v = du * cs;
        }
        Fb[std::size_t(j) * nB + bb] = ph * ub - nu * v;
    }
}

// u* = ca a + cb b + cr R, R = (-sum +-F + u_P sum(+-phi) / 2) / V: the flux
// form plus the skew-symmetrizing half of u_P div(phi). a may alias b.
template <class R>
__global__ void __launch_bounds__(256) kGather(IMesh<R> m, R ca, R cb, R cr, const R* a, const R* b, const R* F,
                                               const R* Fb, const R* phi, const R* phiB, R* out)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB), cc = std::size_t(c);
    R sum[3] = {0, 0, 0}, dv = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            if (code & 1) {
                dv -= phi[f];
#pragma unroll
                for (int j = 0; j < 3; ++j) sum[j] -= F[std::size_t(j) * nI + f];
            } else {
                dv += phi[f];
#pragma unroll
                for (int j = 0; j < 3; ++j) sum[j] += F[std::size_t(j) * nI + f];
            }
        } else {
            const std::size_t bf = std::size_t(-code - 1);
            dv += phiB[bf];
#pragma unroll
            for (int j = 0; j < 3; ++j) sum[j] += Fb[std::size_t(j) * nB + bf];
        }
    }
    const R V = m.V[c];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const std::size_t i = std::size_t(j) * nC + cc;
        const R ub = b[i];
        const R rate = (R(0.5) * (ub * dv) - sum[j]) / V;
        out[i] = ca * a[i] + cb * ub + cr * rate;
    }
}

// The convective face's own RK stage: du/dt = -uc (u_b - u_P) / dn.
template <class R>
__global__ void kConvOut(IMesh<R> m, R ca, R cb, R cr, const R* U, const R* a, const R* b, R* out)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nB || m.bCode[f] != kConvective) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(f);
    const std::size_t P = std::size_t(m.bOwn[f]);
    const R rate = -m.bUc[f] / m.bDn[f];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const std::size_t i = std::size_t(j) * nB + bb;
        out[i] = ca * a[i] + cb * b[i] + cr * (rate * (b[i] - U[std::size_t(j) * nC + P]));
    }
}

// phi* = S . (w u_o + (1 - w) u_n)
template <class R>
__global__ void kFlux(IMesh<R> m, const R* U, R* phiS)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    const R w = m.wO[f];
    R s = 0;
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        const R uo = U[std::size_t(q) * nC + o], un = U[std::size_t(q) * nC + n];
        s += m.S[std::size_t(q) * nI + ff] * (w * (uo - un) + un);
    }
    phiS[f] = s;
}

// Boundary phi*: impermeable faces 0, inflow and convective from their value,
// outflow from the cell.
template <class R>
__global__ void kFluxB(IMesh<R> m, const R* U, const R* Ub, R* phiSB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB), bb = std::size_t(b);
    const u8 code = m.bCode[b];
    if (code == kWall || code == kSlip) {
        phiSB[b] = 0;
        return;
    }
    R s = 0;
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        const R v = code == kOutflow ? U[std::size_t(q) * nC + std::size_t(m.bOwn[b])] : Ub[std::size_t(q) * nB + bb];
        s += m.bS[std::size_t(q) * nB + bb] * v;
    }
    phiSB[b] = s;
}

// Closes the net flux on the convective faces (the MAC tunnel's outlet
// correction): phi_b += dU |S|, u_b += dU n.
template <class R>
__global__ void kConvAdjust(IMesh<R> m, R dU, R* phiSB, R* Ub)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB || m.bCode[b] != kConvective) return;
    const std::size_t nB = std::size_t(m.nB), bb = std::size_t(b);
    phiSB[b] += dU * m.bMagS[b];
#pragma unroll
    for (int q = 0; q < 3; ++q) Ub[std::size_t(q) * nB + bb] += dU * m.bN[std::size_t(q) * nB + bb];
}

// kg_f = k . (w GL_o + (1 - w) GL_n): the deferred non-orthogonal flux.
template <class R>
__global__ void kKg(IMesh<R> m, const R* Gp, R* kg)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    const R w = m.wO[f];
    R s = 0;
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        const R go = Gp[std::size_t(q) * nC + o], gn = Gp[std::size_t(q) * nC + n];
        s += m.k[std::size_t(q) * nI + ff] * (w * (go - gn) + gn);
    }
    kg[f] = s;
}

// A p = rhs: -div(phi*)/c + div(kg) + fixed-p boundary terms. gross = sum|phi*|/c.
// Tgt (low-Mach): the divergence target, tgt = V D per cell, sits inside the bracket:
// a difference of near-equal terms is read once, before the scale by 1/c.
template <class R, bool NonOrth, bool Tgt = false>
__global__ void __launch_bounds__(256) kRhs(IMesh<R> m, R ic, const R* phiS, const R* phiSB, const R* kg, const R* tgt,
                                            R* rhs, R* gross)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R dv = 0, gr = 0, dk = 0, bp = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            const R p = phiS[f];
            gr += fabs(p);
            if (code & 1) {
                dv -= p;
                if constexpr (NonOrth) dk -= kg[f];
            } else {
                dv += p;
                if constexpr (NonOrth) dk += kg[f];
            }
        } else {
            const int b = -code - 1;
            const R p = phiSB[b];
            dv += p;
            gr += fabs(p);
            if (m.bCode[b] == kOutflow) bp += m.bA[b] * m.bP[b];
        }
    }
    if constexpr (Tgt) rhs[c] = bp + dk - ic * (dv - tgt[c]);
    else rhs[c] = bp + dk - ic * dv;
    gross[c] = ic * gr;
}

template <class R>
__global__ void kShift(int n, R s, R* x)
{
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i < n) x[i] -= s;
}

// fp_f = a (p_n - p_o) + kg_f
template <class R, bool NonOrth>
__global__ void kFp(IMesh<R> m, const R* p, const R* kg, R* fp)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    R s = m.a[f] * (p[m.nei[f]] - p[m.own[f]]);
    if constexpr (NonOrth) s += kg[f];
    fp[f] = s;
}

template <class R>
__global__ void kFpB(IMesh<R> m, const R* p, R* fpB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    fpB[b] = m.bCode[b] == kOutflow ? m.bA[b] * (m.bP[b] - p[m.bOwn[b]]) : R(0.0);
}

// G_c = (1/V) sum_f r_f fp_f (outward): the Perot reconstruction.
template <class R>
__global__ void __launch_bounds__(256) kGp(IMesh<R> m, const R* fp, const R* fpB, R* Gp)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    R g[3] = {0, 0, 0};
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            const R v = fp[f];
            if (code & 1) { // neighbour side: r_n = r_o - d, outward flux -fp
#pragma unroll
                for (int q = 0; q < 3; ++q)
                    g[q] += (m.d[std::size_t(q) * nI + f] - m.rO[std::size_t(q) * nI + f]) * v;
            } else {
#pragma unroll
                for (int q = 0; q < 3; ++q) g[q] += m.rO[std::size_t(q) * nI + f] * v;
            }
        } else {
            const std::size_t b = std::size_t(-code - 1);
            const R v = fpB[b];
#pragma unroll
            for (int q = 0; q < 3; ++q) g[q] += m.bR[std::size_t(q) * nB + b] * v;
        }
    }
    const R V = m.V[c];
#pragma unroll
    for (int q = 0; q < 3; ++q) Gp[std::size_t(q) * nC + std::size_t(c)] = g[q] / V;
}

// out[i] = x[i] - c y[i]
template <class R>
__global__ void kSubScaled(std::size_t n, R c, const R* x, const R* y, R* out)
{
    const std::size_t i = std::size_t(blockIdx.x) * kT + threadIdx.x;
    if (i < n) out[i] = x[i] - c * y[i];
}

// Boundary values of the cell field U. Convective faces keep theirs.
template <class R>
__global__ void kEvalB(IMesh<R> m, const R* U, const R* phiB, R* Ub)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const std::size_t P = std::size_t(m.bOwn[b]);
    const u8 code = m.bCode[b];
    if (code == kConvective) return;
    if (code == kWall || code == kInflow) {
#pragma unroll
        for (int q = 0; q < 3; ++q) Ub[std::size_t(q) * nB + bb] = m.bU[std::size_t(q) * nB + bb];
        return;
    }
    R up[3], n[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        up[q] = U[std::size_t(q) * nC + P];
        n[q] = m.bN[std::size_t(q) * nB + bb];
    }
    const R un = up[0] * n[0] + up[1] * n[1] + up[2] * n[2];
    if (code == kSlip) {
#pragma unroll
        for (int q = 0; q < 3; ++q) Ub[std::size_t(q) * nB + bb] = up[q] - un * n[q];
    } else if (phiB[b] >= 0) { // outflow
#pragma unroll
        for (int q = 0; q < 3; ++q) Ub[std::size_t(q) * nB + bb] = up[q];
    } else { // backflow: the normal part only
#pragma unroll
        for (int q = 0; q < 3; ++q) Ub[std::size_t(q) * nB + bb] = un * n[q];
    }
}

// Per-cell max(conv, diff) of the dt rule.
template <class R, bool Les>
__global__ void __launch_bounds__(256) kRate(IMesh<R> m, R nu, const R* phi, const R* phiB, const R* nut, R* rate)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R conv = 0, diff = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            conv += fabs(phi[f]);
            R nf = nu;
            if constexpr (Les) nf += nut[m.own[f]] + nut[m.nei[f]]; // 2 nut_f: the eddy normal stress
            diff += nf * m.cS[f];
        } else {
            const int b = -code - 1;
            conv += fabs(phiB[b]);
            if (m.bCode[b] == kWall || m.bCode[b] == kInflow || m.bCode[b] == kSlip) diff += nu * m.bCs[b];
        }
    }
    const R V = m.V[c];
    rate[c] = fmax(R(0.5) * conv / V, diff / V);
}

// Per-cell KE, |div|, V div^2, |u|.
template <class R>
__global__ void __launch_bounds__(256) kCellStats(IMesh<R> m, const R* U, const R* phi, const R* phiB, R* ke, R* dv,
                                                  R* dv2, R* um)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    R s = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) s += signOf<R>(code) * phi[faceOf(code)];
        else s += phiB[-code - 1];
    }
    const R V = m.V[c];
    R u2 = 0;
#pragma unroll
    for (int q = 0; q < 3; ++q) u2 += U[std::size_t(q) * nC + cc] * U[std::size_t(q) * nC + cc];
    const R d = s / V;
    ke[c] = R(0.5) * V * u2;
    dv[c] = fabs(d);
    dv2[c] = V * (d * d);
    um[c] = sqrt(u2);
}

// corr.S |du|^2 over the internal faces.
template <class R>
__global__ void kEpsF(IMesh<R> m, const R* U, R* e)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nC = std::size_t(m.nC);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    R s = 0;
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        const R du = U[std::size_t(q) * nC + n] - U[std::size_t(q) * nC + o];
        s += du * du;
    }
    e[f] = m.cS[f] * s;
}

// The same over the faces that fix u.
template <class R>
__global__ void kEpsB(IMesh<R> m, const R* U, const R* Ub, R* e)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB), P = std::size_t(m.bOwn[b]);
    if (m.bCode[b] != kWall && m.bCode[b] != kInflow) {
        e[b] = 0;
        return;
    }
    R s = 0;
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        const R du = Ub[std::size_t(q) * nB + std::size_t(b)] - U[std::size_t(q) * nC + P];
        s += du * du;
    }
    e[b] = m.bCs[b] * s;
}

// out[c] = V u . x (x: a per-volume vector field).
template <class R>
__global__ void kCellDot(IMesh<R> m, const R* U, const R* x, R* out)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    R s = 0;
#pragma unroll
    for (int q = 0; q < 3; ++q) s += U[std::size_t(q) * nC + cc] * x[std::size_t(q) * nC + cc];
    out[c] = m.V[c] * s;
}

// Forces on the `force` patches: [fp 3, fv 3] x nB. Wall p is the cell's: the
// value the momentum equation itself applies there (fp = 0 on a no-flux face in
// G IS p_b = p_P on an orthogonal wall). p_P + G.r reads half a normal gradient
// that carries the stage's c: cd moved with dt (docs/fvincomp.md).
template <class R, bool Corr>
__global__ void kForce(IMesh<R> m, R nu, const R* U, const R* Ub, const R* G, const R* p, R* out)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const std::size_t P = std::size_t(m.bOwn[b]);
    if (!m.bForce[b]) {
#pragma unroll
        for (int k = 0; k < 6; ++k) out[std::size_t(k) * nB + bb] = 0;
        return;
    }
    const R pb = p[P];
    const R cs = m.bCs[b];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const R du = Ub[std::size_t(j) * nB + bb] - U[std::size_t(j) * nC + P];
        R v = du * cs;
        if constexpr (Corr) {
            R gS = 0, gr = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const R g = G[std::size_t(3 * i + j) * nC + P];
                gS += g * m.bS[std::size_t(i) * nB + bb];
                gr += g * m.bR[std::size_t(i) * nB + bb];
            }
            v = gS + (du - gr) * cs;
            if (m.bCode[b] == kWall) v += wallStressRest(m, U, Ub, G, P, bb, j);
        }
        out[std::size_t(j) * nB + bb] = pb * m.bS[std::size_t(j) * nB + bb];
        out[std::size_t(3 + j) * nB + bb] = -nu * v;
    }
}

template <class R>
__global__ void kFillR(std::size_t n, R v, R* x)
{
    const std::size_t i = std::size_t(blockIdx.x) * kT + threadIdx.x;
    if (i < n) x[i] = v;
}

// Least-squares gradient of the scalar T, GT[q nC + c].
template <class R>
__global__ void __launch_bounds__(256) kGradT(IMesh<R> m, const R* T, const R* Tb, R* GT)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    const std::size_t nF = nI + nB;
    const R tc = T[c];
    R g[3] = {0, 0, 0};
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            const bool isNei = code & 1;
            const R dt = T[isNei ? m.own[f] : m.nei[f]] - tc;
#pragma unroll
            for (int i = 0; i < 3; ++i)
                g[i] += (isNei ? m.gN[std::size_t(i) * nI + f] : m.gO[std::size_t(i) * nF + f]) * dt;
        } else {
            const std::size_t b = std::size_t(-code - 1);
            const R dt = Tb[b] - tc;
#pragma unroll
            for (int i = 0; i < 3; ++i) g[i] += m.gO[std::size_t(i) * nF + nI + b] * dt;
        }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) GT[std::size_t(i) * nC + std::size_t(c)] = g[i];
}

// Internal face flux of T, phi (T_o + T_n)/2 - kappa g.S: kFace's scalar. Les (mci,
// docs/fvincomp.md "Incompressible combinations"): the eddy heat flux, kappa + nut_f /
// Pr_t on the face, nut_f the cells' mean (kFace's), Pr_t = kLesPrt; boundary faces
// stay molecular (kBndT), the wall's nut_f = 0. nut 0: kappa + 0 = kappa, bit for bit.
template <class R, bool Corr, bool Les = false>
__global__ void kFaceT(IMesh<R> m, R kap, const R* T, const R* GT, const R* phi, R* F, const R* nut = nullptr,
                       R invPrt = R(0))
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    const R to = T[o], tn = T[n], cs = m.cS[f];
    R diff = cs * (tn - to);
    if constexpr (Corr) {
        R gS = 0, gd = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const R gb = R(0.5) * (GT[std::size_t(i) * nC + o] + GT[std::size_t(i) * nC + n]);
            gS += gb * m.S[std::size_t(i) * nI + ff];
            gd += gb * m.d[std::size_t(i) * nI + ff];
        }
        diff = gS + (tn - to - gd) * cs;
    }
    if constexpr (Les) kap += (R(0.5) * (nut[o] + nut[n])) * invPrt;
    F[f] = phi[f] * (R(0.5) * (to + tn)) - kap * diff;
}

// Boundary face flux of T: phi T_b, plus conduction on a fixed-T face. A
// zero-flux face carries EXACTLY 0.
template <class R, bool Corr>
__global__ void kBndT(IMesh<R> m, R kap, const R* T, const R* Tb, const R* GT, const R* phiB, R* Fb)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const std::size_t P = std::size_t(m.bOwn[b]);
    const R adv = phiB[b] * Tb[b];
    if (!m.bTfix[b]) {
        Fb[b] = adv;
        return;
    }
    const R dT = Tb[b] - T[P], cs = m.bCs[b];
    R v = dT * cs;
    if constexpr (Corr) {
        R gS = 0, gr = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const R g = GT[std::size_t(i) * nC + P];
            gS += g * m.bS[std::size_t(i) * nB + bb];
            gr += g * m.bR[std::size_t(i) * nB + bb];
        }
        v = gS + (dT - gr) * cs;
    }
    Fb[b] = adv - kap * v;
}

// T* = ca a + cb b + cr R, the scalar of kGather: the same skew-symmetrizing half.
template <class R>
__global__ void __launch_bounds__(256) kGatherT(IMesh<R> m, R ca, R cb, R cr, const R* a, const R* b, const R* F,
                                                const R* Fb, const R* phi, const R* phiB, R* out)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R sum = 0, dv = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            if (code & 1) {
                dv -= phi[f];
                sum -= F[f];
            } else {
                dv += phi[f];
                sum += F[f];
            }
        } else {
            const std::size_t bf = std::size_t(-code - 1);
            dv += phiB[bf];
            sum += Fb[bf];
        }
    }
    const R tb = b[c];
    out[c] = ca * a[c] + cb * tb + cr * ((R(0.5) * (tb * dv) - sum) / m.V[c]);
}

// Boundary values of T: the fixed value, or the cell's (zero flux).
template <class R>
__global__ void kEvalT(IMesh<R> m, const R* T, R* Tb)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    Tb[b] = m.bTfix[b] ? m.bT[b] : T[m.bOwn[b]];
}

// B_f = (T_f - Tref) betaG.S (linear T_f), and phi* += c B_f.
template <class R>
__global__ void kBuoyF(IMesh<R> m, R bx, R by, R bz, R Tref, R c, const R* T, R* phiS, R* Bf)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nI = std::size_t(m.nI), ff = std::size_t(f);
    const R w = m.wO[f], to = T[m.own[f]], tn = T[m.nei[f]];
    const R gS = bx * m.S[ff] + by * m.S[nI + ff] + bz * m.S[2 * nI + ff];
    const R B = (w * (to - tn) + tn - Tref) * gS;
    Bf[f] = B;
    phiS[f] += c * B;
}

// The hydrostatic image of a wall face: p_b - p_P = (T_b - Tref) betaG.r.
template <class R>
__global__ void kBuoyB(IMesh<R> m, R bx, R by, R bz, R Tref, const R* Tb, R* dpB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), bb = std::size_t(b);
    dpB[b] = (Tb[b] - Tref) * (bx * m.bR[bb] + by * m.bR[nB + bb] + bz * m.bR[2 * nB + bb]);
}

// out[c] = V T r
template <class R>
__global__ void kCellMulV(IMesh<R> m, const R* T, const R* r, R* out)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    out[c] = m.V[c] * (T[c] * r[c]);
}

// Exact max of x over [0, N): the dots' block scheme with max, so independent
// of everything but the values.
template <class R>
__global__ void __launch_bounds__(256) kMaxPartial(std::size_t N, const R* x, double* part)
{
    __shared__ double sh[kT];
    const int t = int(threadIdx.x);
    double s = 0.0;
    const std::size_t stride = std::size_t(gridDim.x) * kT;
    for (std::size_t i = std::size_t(blockIdx.x) * kT + std::size_t(t); i < N; i += stride) s = fmax(s, double(x[i]));
    sh[t] = s;
    __syncthreads();
    for (int w = kT / 2; w > 0; w >>= 1) {
        if (t < w) sh[t] = fmax(sh[t], sh[t + w]);
        __syncthreads();
    }
    if (t == 0) part[blockIdx.x] = sh[0];
}

} // namespace

struct DeviceFvIncomp::Impl
{
    virtual ~Impl() = default;
    virtual std::expected<double, std::string> stableDt(double cfl) = 0;
    virtual std::expected<FvIncompStepLog, std::string> step(double dt) = 0;
    virtual std::expected<FvIncompDiag, std::string> diagnostics() = 0;
    virtual std::expected<FvIncompBudget, std::string> budget() = 0;
    virtual std::expected<FvIncompForces, std::string> forces() = 0;
    virtual std::expected<FvIncompState, std::string> state() const = 0;
    virtual std::expected<std::vector<double>, std::string> wallHeat() = 0;
    virtual std::expected<std::vector<double>, std::string> gradU() = 0;
};

namespace {

template <class R>
struct IncompImpl final : DeviceFvIncomp::Impl
{
    IMesh<R> m;
    FvIncompControls ctl;
    bool orthogonal = true, pureNeumann = true, hasConvective = false, hasWall = false, thermal = false;
    bool buoyT = false; // T with a force (betaG != 0); betaG 0: a passive tracer, momentum untouched
    double volume = 0, convArea = 0;
    int nCorr = 1;
    // mesh
    DeviceBuffer<int> own, nei, bOwn, cStart, cEnt;
    DeviceBuffer<R> S, d, rO, id2, cS, wO, a, k, bS, bR, bN, bCs, bA, bDn, bMagS, bU, bP, bUc, V, gO, gN, delta2;
    DeviceBuffer<u8> bCode, bForce, bTfix;
    // temperature: Tc current, Ta / Ts the RK buffers, Tb* boundary values
    DeviceBuffer<R> Tc, Ta, Ts, Tb, Tba, Tbs, GT, FT, FTb, Bf, bT, dpB;
    // state: U is the current field, U0 / Us the RK buffers (they swap)
    DeviceBuffer<R> U, U0, Us, Ub, Ub0, Ubs, phi, phiB, p;
    // work
    DeviceBuffer<R> G, Gp, GL, nut, F, Fb, phiS, phiSB, fp, fpB, kg, rhs, gross, w0, w1, w2, wf, wb, wU, ones;
    DeviceBuffer<double> part, outv;
    std::optional<la::BlockEll<R>> A;
    std::optional<la::Amg<R>> amg;
    std::optional<la::Krylov<R>> kr;
    FvIncompStepLog log;

    // The low-Mach stage's V D: project() closes div phi on it instead of on 0.
    const R* target = nullptr;
    // Low-Mach gravity: phi* carries c Bf (internal) and c BfB (the p-fixed faces),
    // the reconstruction reads fp - Bf, fpB - BfB, kGradP the walls' image dpB.
    bool buoyant = false;
    DeviceBuffer<R> BfB;

    bool wantG() const { return !orthogonal || ctl.les > 0 || hasWall || thermal || ctl.lowMach; }
    std::expected<void, std::string> check(const char* what)
    {
        if (auto e = hipGetLastError(); e != hipSuccess)
            return std::unexpected(std::format("fv incomp: {}: {}", what, hipGetErrorString(e)));
        return {};
    }

    // sum_i a_k[i] b_k[i] in FP64, fixed order (b_k null: the L1 sum).
    std::expected<std::array<double, 8>, std::string> dots(std::size_t N, int K, const R* const* x, const R* const* y)
    {
        la::internal::DotArgs<R> da{};
        for (int i = 0; i < K; ++i) {
            da.a[i] = x[i];
            da.b[i] = y[i];
        }
        la::internal::deviceDots(N, da, K, part.data(), outv.data());
        std::array<double, 8> h{};
        if (auto e = hipMemcpy(h.data(), outv.data(), std::size_t(K) * sizeof(double), hipMemcpyDeviceToHost);
            e != hipSuccess)
            return std::unexpected(std::string("fv incomp: reduction: ") + hipGetErrorString(e));
        return h;
    }

    std::expected<double, std::string> maxOf(std::size_t N, const R* x)
    {
        const std::size_t P = std::size_t(la::internal::dotParts(N));
        kMaxPartial<R><<<unsigned(P), kT>>>(N, x, part.data());
        std::vector<double> h(P);
        if (auto e = hipMemcpy(h.data(), part.data(), P * sizeof(double), hipMemcpyDeviceToHost); e != hipSuccess)
            return std::unexpected(std::string("fv incomp: reduction: ") + hipGetErrorString(e));
        return *std::ranges::max_element(h);
    }

    void gradient(const R* u, const R* ub) { kGrad<R><<<grid1(std::size_t(m.nC)), kT>>>(m, u, ub, G.data()); }

    // phi, Gp, u -= c Gp from the stage's predicted flux (phiS, phiSB).
    std::expected<void, std::string> project(R* u, R c)
    {
        const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
        const unsigned gC = grid1(nC), gI = grid1(nI), gB = grid1(nB);
        const R ic = R(1.0) / c;
        for (int it = 0; it < nCorr; ++it) {
            if (!orthogonal) {
                (buoyT || buoyant ? kGradP<R, true> : kGradP<R, false>)<<<gC, kT>>>(m, p.data(), dpB.data(),
                                                                                    GL.data());
                kKg<R><<<gI, kT>>>(m, GL.data(), kg.data());
            }
            if (target)
                (orthogonal ? kRhs<R, false, true> : kRhs<R, true, true>)<<<gC, kT>>>(
                    m, ic, phiS.data(), phiSB.data(), kg.data(), target, rhs.data(), gross.data());
            else
                (orthogonal ? kRhs<R, false> : kRhs<R, true>)<<<gC, kT>>>(m, ic, phiS.data(), phiSB.data(), kg.data(),
                                                                          nullptr, rhs.data(), gross.data());
            const R* xs[3] = {rhs.data(), rhs.data(), gross.data()};
            const R* ys[3] = {ones.data(), rhs.data(), gross.data()};
            auto dt = dots(nC, 3, xs, ys);
            if (!dt) return std::unexpected(dt.error());
            const double sumB = (*dt)[0], normB = std::sqrt((*dt)[1]), normG = std::sqrt((*dt)[2]);
            if (!std::isfinite(normB) || !std::isfinite(normG)) return std::unexpected("fv incomp: non-finite pressure rhs");
            if (pureNeumann) kShift<R><<<gC, kT>>>(int(nC), R(sumB / double(nC)), rhs.data());
            // Stop on the removed divergence, or on the arithmetic's noise: below
            // ~eps gross the FP32 estimate keeps falling on nothing (CLAUDE.md).
            la::KrylovOptions o;
            o.maxIter = ctl.maxIter;
            o.relTol = 0;
            o.absTol = std::max(ctl.tol * normB, 4.0 * double(std::numeric_limits<R>::epsilon()) * normG);
            o.flexible = true;
            const la::LinOp<R> Aop = [this](const R* in, R* out) { A->spmv(in, out); };
            const la::LinOp<R> Mop = [this](const R* in, R* out) { amg->apply(in, out); };
            auto r = kr->pcg(Aop, Mop, rhs.data(), p.data(), o);
            if (!r) return std::unexpected("fv incomp: pressure: " + r.error());
            ++log.solves;
            log.iterations += r->iterations;
            log.unconverged += r->converged ? 0 : 1;
            log.divRatio = std::max(log.divRatio, normB > 0 ? r->residual / normB : 0.0);
            (orthogonal ? kFp<R, false> : kFp<R, true>)<<<gI, kT>>>(m, p.data(), kg.data(), fp.data());
            kFpB<R><<<gB, kT>>>(m, p.data(), fpB.data());
        }
        if (buoyT) { // the net face force: at rest fp = B_f and G is exactly 0
            kSubScaled<R><<<gI, kT>>>(nI, R(1.0), fp.data(), Bf.data(), wf.data());
            kGp<R><<<gC, kT>>>(m, wf.data(), fpB.data(), Gp.data());
        } else if (buoyant) { // low-Mach: the p-fixed faces carry a force too
            kSubScaled<R><<<gI, kT>>>(nI, R(1.0), fp.data(), Bf.data(), wf.data());
            kSubScaled<R><<<gB, kT>>>(nB, R(1.0), fpB.data(), BfB.data(), wb.data());
            kGp<R><<<gC, kT>>>(m, wf.data(), wb.data(), Gp.data());
        } else {
            kGp<R><<<gC, kT>>>(m, fp.data(), fpB.data(), Gp.data());
        }
        kSubScaled<R><<<gI, kT>>>(nI, c, phiS.data(), fp.data(), phi.data());
        kSubScaled<R><<<gB, kT>>>(nB, c, phiSB.data(), fpB.data(), phiB.data());
        for (int q = 0; q < 3; ++q) {
            const std::size_t o = std::size_t(q) * nC;
            kSubScaled<R><<<gC, kT>>>(nC, c, u + o, Gp.data() + o, u + o);
        }
        return {};
    }

    // Closes the net boundary flux of phi*SB on the convective faces.
    std::expected<void, std::string> closeConvective(R* ub)
    {
        const std::size_t nB = std::size_t(m.nB);
        const R* xs[1] = {phiSB.data()};
        const R* ys[1] = {ones.data()};
        auto s = dots(nB, 1, xs, ys);
        if (!s) return std::unexpected(s.error());
        kConvAdjust<R><<<grid1(nB), kT>>>(m, R(-(*s)[0] / convArea), phiSB.data(), ub);
        return {};
    }

    // One RK stage: out = ca a + cb b + cb dt R(b, phi), projected.
    std::expected<void, std::string> stage(const R* ua, const R* uba, const R* ub, const R* ubb, R* out, R* outB, double cb,
                                           double dt, const R* ta = nullptr, const R* tb = nullptr,
                                           const R* tbb = nullptr, R* tOut = nullptr, R* tOutB = nullptr)
    {
        const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
        const unsigned gC = grid1(nC), gI = grid1(nI), gB = grid1(nB);
        const R rcb = R(cb), rca = R(1) - rcb, cr = R(cb * dt), nu = R(ctl.nu);
        const bool les = ctl.les > 0;
        if (wantG()) gradient(ub, ubb);
        if (les)
            kNut<R><<<gC, kT>>>(m.nC, R(ctl.les * ctl.les), ctl.sgs == SgsModel::Sigma ? 1 : 0, G.data(), delta2.data(),
                                nut.data());
        if (les) kFace<R, true, true><<<gI, kT>>>(m, nu, ub, G.data(), phi.data(), nut.data(), F.data());
        else if (!orthogonal) kFace<R, true, false><<<gI, kT>>>(m, nu, ub, G.data(), phi.data(), nullptr, F.data());
        else kFace<R, false, false><<<gI, kT>>>(m, nu, ub, nullptr, phi.data(), nullptr, F.data());
        (wantG() ? kBnd<R, true> : kBnd<R, false>)<<<gB, kT>>>(m, nu, ub, ubb, G.data(), phiB.data(), Fb.data());
        kGather<R><<<gC, kT>>>(m, rca, rcb, cr, ua, ub, F.data(), Fb.data(), phi.data(), phiB.data(), out);
        if (hasConvective) kConvOut<R><<<gB, kT>>>(m, rca, rcb, cr, ub, uba, ubb, outB);
        if (thermal) {
            scalarFluxes(R(ctl.kappa), tb, tbb);
            kGatherT<R><<<gC, kT>>>(m, rca, rcb, cr, ta, tb, FT.data(), FTb.data(), phi.data(), phiB.data(), tOut);
        }
        kFlux<R><<<gI, kT>>>(m, out, phiS.data());
        kFluxB<R><<<gB, kT>>>(m, out, outB, phiSB.data());
        if (hasConvective)
            if (auto r = closeConvective(outB); !r) return r;
        if (buoyT) {
            const R bx = R(ctl.betaG[0]), by = R(ctl.betaG[1]), bz = R(ctl.betaG[2]), tr = R(ctl.Tref);
            kBuoyF<R><<<gI, kT>>>(m, bx, by, bz, tr, cr, tb, phiS.data(), Bf.data());
            kBuoyB<R><<<gB, kT>>>(m, bx, by, bz, tr, tbb, dpB.data());
        }
        if (auto r = project(out, cr); !r) return r;
        kEvalB<R><<<gB, kT>>>(m, out, phiB.data(), outB);
        if (thermal) kEvalT<R><<<gB, kT>>>(m, tOut, tOutB);
        return {};
    }

    // FT, FTb of the temperature t with boundary values tb. LES: the eddy heat flux of
    // this stage's nut on the internal faces.
    void scalarFluxes(R kap, const R* t, const R* tb)
    {
        const unsigned gC = grid1(std::size_t(m.nC)), gI = grid1(std::size_t(m.nI)), gB = grid1(std::size_t(m.nB));
        if (!orthogonal) kGradT<R><<<gC, kT>>>(m, t, tb, GT.data());
        if (ctl.les > 0)
            (orthogonal ? kFaceT<R, false, true> : kFaceT<R, true, true>)<<<gI, kT>>>(
                m, kap, t, GT.data(), phi.data(), FT.data(), nut.data(), R(1.0 / kLesPrt));
        else
            (orthogonal ? kFaceT<R, false> : kFaceT<R, true>)<<<gI, kT>>>(m, kap, t, GT.data(), phi.data(), FT.data(),
                                                                           nullptr, R(0));
        (orthogonal ? kBndT<R, false> : kBndT<R, true>)<<<gB, kT>>>(m, kap, t, tb, GT.data(), phiB.data(), FTb.data());
    }

    std::expected<double, std::string> stableDt(double cfl) override
    {
        const std::size_t nC = std::size_t(m.nC);
        const bool les = ctl.les > 0;
        (les ? kRate<R, true> : kRate<R, false>)<<<grid1(nC), kT>>>(m, R(thermal ? std::max(ctl.nu, ctl.kappa) : ctl.nu),
                                                                   phi.data(), phiB.data(), nut.data(), w0.data());
        auto mx = maxOf(nC, w0.data());
        if (!mx) return std::unexpected(mx.error());
        if (auto r = check("stable dt"); !r) return std::unexpected(r.error());
        return cfl / *mx;
    }

    std::expected<FvIncompStepLog, std::string> step(double dt) override
    {
        log = {};
        std::swap(U0, U);
        std::swap(Ub0, Ub);
        std::swap(Ta, Tc);
        std::swap(Tba, Tb);
        if (auto r = stage(U0.data(), Ub0.data(), U0.data(), Ub0.data(), U.data(), Ub.data(), 1.0, dt, Ta.data(), Ta.data(),
                           Tba.data(), Tc.data(), Tb.data());
            !r)
            return std::unexpected(r.error());
        if (auto r = stage(U0.data(), Ub0.data(), U.data(), Ub.data(), Us.data(), Ubs.data(), 0.25, dt, Ta.data(), Tc.data(),
                           Tb.data(), Ts.data(), Tbs.data());
            !r)
            return std::unexpected(r.error());
        if (auto r = stage(U0.data(), Ub0.data(), Us.data(), Ubs.data(), U.data(), Ub.data(), 2.0 / 3.0, dt, Ta.data(),
                           Ts.data(), Tbs.data(), Tc.data(), Tb.data());
            !r)
            return std::unexpected(r.error());
        if (auto r = check("step"); !r) return std::unexpected(r.error());
        return log;
    }

    std::expected<FvIncompDiag, std::string> diagnostics() override
    {
        const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
        kCellStats<R><<<grid1(nC), kT>>>(m, U.data(), phi.data(), phiB.data(), w0.data(), w1.data(), w2.data(), wf.data());
        const R* xs[2] = {w0.data(), w2.data()};
        const R* ys[2] = {nullptr, nullptr};
        auto s = dots(nC, 2, xs, ys);
        if (!s) return std::unexpected(s.error());
        auto dm = maxOf(nC, w1.data());
        if (!dm) return std::unexpected(dm.error());
        auto um = maxOf(nC, wf.data());
        if (!um) return std::unexpected(um.error());
        FvIncompDiag out{.K = (*s)[0], .divMax = *dm, .divRms = std::sqrt((*s)[1] / volume), .uMax = *um};
        kEpsF<R><<<grid1(nI), kT>>>(m, U.data(), wf.data());
        const R* ef[1] = {wf.data()};
        const R* no[1] = {nullptr};
        auto e0 = dots(nI, 1, ef, no);
        if (!e0) return std::unexpected(e0.error());
        out.eps = ctl.nu * (*e0)[0];
        if (nB > 0) {
            kEpsB<R><<<grid1(nB), kT>>>(m, U.data(), Ub.data(), wf.data());
            auto e1 = dots(nB, 1, ef, no);
            if (!e1) return std::unexpected(e1.error());
            out.eps += ctl.nu * (*e1)[0];
        }
        if (auto r = check("diagnostics"); !r) return std::unexpected(r.error());
        return out;
    }

    std::expected<FvIncompBudget, std::string> budget() override
    {
        const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
        const unsigned gC = grid1(nC);
        kFace<R, false, false><<<grid1(nI), kT>>>(m, R(0.0), U.data(), nullptr, phi.data(), nullptr, F.data());
        kBnd<R, false><<<grid1(nB), kT>>>(m, R(0.0), U.data(), Ub.data(), nullptr, phiB.data(), Fb.data());
        kGather<R><<<gC, kT>>>(m, R(0.0), R(0.0), R(1.0), U.data(), U.data(), F.data(), Fb.data(), phi.data(), phiB.data(),
                               wU.data());
        kCellDot<R><<<gC, kT>>>(m, U.data(), wU.data(), w0.data());
        kCellDot<R><<<gC, kT>>>(m, U.data(), Gp.data(), w1.data());
        const R* xs[3] = {w0.data(), w0.data(), w1.data()};
        const R* ys[3] = {ones.data(), nullptr, ones.data()};
        auto s = dots(nC, 3, xs, ys);
        if (!s) return std::unexpected(s.error());
        FvIncompBudget o{.conv = (*s)[0], .grossConv = (*s)[1], .press = -(*s)[2]};
        if (thermal) {
            kFaceT<R, false><<<grid1(nI), kT>>>(m, R(0.0), Tc.data(), nullptr, phi.data(), FT.data(), nullptr, R(0));
            kBndT<R, false><<<grid1(nB), kT>>>(m, R(0.0), Tc.data(), Tb.data(), nullptr, phiB.data(), FTb.data());
            kGatherT<R><<<gC, kT>>>(m, R(0.0), R(0.0), R(1.0), Tc.data(), Tc.data(), FT.data(), FTb.data(), phi.data(),
                                    phiB.data(), w1.data());
            kCellMulV<R><<<gC, kT>>>(m, Tc.data(), w1.data(), w0.data());
            const R* xt[2] = {w0.data(), w0.data()};
            const R* yt[2] = {ones.data(), nullptr};
            auto st = dots(nC, 2, xt, yt);
            if (!st) return std::unexpected(st.error());
            o.convT = (*st)[0];
            o.grossConvT = (*st)[1];
        }
        if (auto r = check("budget"); !r) return std::unexpected(r.error());
        return o;
    }

    std::expected<FvIncompForces, std::string> forces() override
    {
        const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
        FvIncompForces o;
        if (nB == 0) return o;
        const bool corr = wantG();
        if (corr) gradient(U.data(), Ub.data());
        (corr ? kForce<R, true> : kForce<R, false>)<<<grid1(nB), kT>>>(m, R(ctl.nu), U.data(), Ub.data(), G.data(), p.data(),
                                                                      wb.data());
        (void)nC;
        const R* xs[6];
        const R* ys[6];
        for (int k = 0; k < 6; ++k) {
            xs[k] = wb.data() + std::size_t(k) * nB;
            ys[k] = ones.data();
        }
        auto s = dots(nB, 6, xs, ys);
        if (!s) return std::unexpected(s.error());
        for (std::size_t j = 0; j < 3; ++j) {
            o.fp[j] = (*s)[j];
            o.fv[j] = (*s)[3 + j];
        }
        if (auto r = check("forces"); !r) return std::unexpected(r.error());
        return o;
    }

    std::expected<FvIncompState, std::string> state() const override
    {
        const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
        FvIncompState s;
        s.U.resize(3 * nC);
        s.Ub.resize(3 * nB);
        s.p.resize(nC);
        s.phi.resize(nI);
        s.phiB.resize(nB);
        auto get = [](std::vector<double>& h, const DeviceBuffer<R>& dv) -> std::expected<void, std::string> {
            if (h.empty()) return {};
            std::vector<R> t(h.size());
            if (auto e = hipMemcpy(t.data(), dv.data(), t.size() * sizeof(R), hipMemcpyDeviceToHost); e != hipSuccess)
                return std::unexpected(std::string("fv incomp: download: ") + hipGetErrorString(e));
            std::ranges::transform(t, h.begin(), [](R x) { return double(x); });
            return {};
        };
        for (auto [h, dv] : {std::pair{&s.U, &U}, {&s.Ub, &Ub}, {&s.p, &p}, {&s.phi, &phi}, {&s.phiB, &phiB}})
            if (auto r = get(*h, *dv); !r) return std::unexpected(r.error());
        if (thermal) {
            s.T.resize(nC);
            s.Tb.resize(nB);
            if (auto r = get(s.T, Tc); !r) return std::unexpected(r.error());
            if (auto r = get(s.Tb, Tb); !r) return std::unexpected(r.error());
        }
        return s;
    }

    std::expected<std::vector<double>, std::string> fetchR(const DeviceBuffer<R>& dv, std::size_t n)
    {
        std::vector<R> t(n);
        if (auto e = hipMemcpy(t.data(), dv.data(), n * sizeof(R), hipMemcpyDeviceToHost); e != hipSuccess)
            return std::unexpected(std::string("fv incomp: download: ") + hipGetErrorString(e));
        std::vector<double> h(n);
        std::ranges::transform(t, h.begin(), [](R x) { return double(x); });
        return h;
    }

    std::expected<std::vector<double>, std::string> wallHeat() override
    {
        if (!thermal) return std::unexpected("fv incomp: no temperature in this run");
        scalarFluxes(R(ctl.kappa), Tc.data(), Tb.data());
        if (auto r = check("wall heat"); !r) return std::unexpected(r.error());
        return fetchR(FTb, std::size_t(m.nB));
    }

    std::expected<std::vector<double>, std::string> gradU() override
    {
        if (!thermal) return std::unexpected("fv incomp: no temperature in this run");
        gradient(U.data(), Ub.data());
        if (auto r = check("velocity gradient"); !r) return std::unexpected(r.error());
        return fetchR(G, 9 * std::size_t(m.nC));
    }
};

template <class R>
std::expected<void, std::string> makeImpl(IncompImpl<R>& I, const FvIncompSetup& s, const dns::la::CellGraph& g,
                                          const FvIncompControls& c, const std::vector<double>& U0,
                                          const std::vector<double>& T0)
{
    const std::size_t nC = std::size_t(s.nC), nI = std::size_t(s.nI), nB = std::size_t(s.nB);
    I.ctl = c;
    I.orthogonal = s.orthogonal;
    I.pureNeumann = s.pureNeumann;
    I.hasConvective = s.hasConvective;
    I.hasWall = std::ranges::any_of(s.bCode, [](std::uint8_t b) { return b == kWall; });
    I.thermal = !T0.empty();
    I.buoyT = I.thermal && std::ranges::any_of(c.betaG, [](double b) { return b != 0.0; });
    I.volume = s.volume;
    I.convArea = s.convArea;
    I.nCorr = c.nCorr > 0 ? c.nCorr : (s.orthogonal ? 1 : 2);
#define FVI_UP(dst, src)                                                                                               \
    {                                                                                                                  \
        auto t_ = upload(src, #src);                                                                                   \
        if (!t_) return std::unexpected(t_.error());                                                                   \
        dst = std::move(*t_);                                                                                          \
    }
#define FVI_UPR(dst, src)                                                                                              \
    {                                                                                                                  \
        auto t_ = uploadAs<R>(src, #src);                                                                              \
        if (!t_) return std::unexpected(t_.error());                                                                   \
        dst = std::move(*t_);                                                                                          \
    }
#define FVI_ZERO(dst, T, n)                                                                                            \
    {                                                                                                                  \
        auto t_ = zeros<T>(n, #dst);                                                                                   \
        if (!t_) return std::unexpected(t_.error());                                                                   \
        dst = std::move(*t_);                                                                                          \
    }
    FVI_UP(I.own, s.own) FVI_UP(I.nei, s.nei) FVI_UP(I.bOwn, s.bOwn) FVI_UP(I.cStart, s.cellStart)
    FVI_UP(I.cEnt, s.cellEnt) FVI_UP(I.bCode, s.bCode) FVI_UP(I.bForce, s.bForce)
    FVI_UPR(I.S, s.S) FVI_UPR(I.d, s.d) FVI_UPR(I.rO, s.rO) FVI_UPR(I.id2, s.id2) FVI_UPR(I.cS, s.cS)
    FVI_UPR(I.wO, s.wO) FVI_UPR(I.a, s.a) FVI_UPR(I.k, s.k) FVI_UPR(I.bS, s.bS) FVI_UPR(I.bR, s.bR)
    FVI_UPR(I.bN, s.bN) FVI_UPR(I.bCs, s.bCs) FVI_UPR(I.bA, s.bA) FVI_UPR(I.bDn, s.bDn) FVI_UPR(I.bMagS, s.bMagS)
    FVI_UPR(I.bU, s.bU) FVI_UPR(I.bP, s.bP) FVI_UPR(I.bUc, s.bUc) FVI_UPR(I.V, s.V) FVI_UPR(I.gO, s.gO)
    FVI_UPR(I.gN, s.gN) FVI_UPR(I.delta2, s.delta2)
    FVI_UPR(I.U, U0)
    FVI_ZERO(I.U0, R, 3 * nC) FVI_ZERO(I.Us, R, 3 * nC) FVI_ZERO(I.Ub, R, 3 * nB) FVI_ZERO(I.Ub0, R, 3 * nB)
    FVI_ZERO(I.Ubs, R, 3 * nB) FVI_ZERO(I.phi, R, nI) FVI_ZERO(I.phiB, R, nB) FVI_ZERO(I.p, R, nC)
    FVI_ZERO(I.G, R, I.wantG() ? 9 * nC : 1) FVI_ZERO(I.Gp, R, 3 * nC) FVI_ZERO(I.GL, R, s.orthogonal ? 1 : 3 * nC)
    FVI_ZERO(I.nut, R, nC)
    FVI_ZERO(I.F, R, 3 * nI) FVI_ZERO(I.Fb, R, 3 * nB) FVI_ZERO(I.phiS, R, nI) FVI_ZERO(I.phiSB, R, nB)
    FVI_ZERO(I.fp, R, nI) FVI_ZERO(I.fpB, R, nB) FVI_ZERO(I.kg, R, s.orthogonal ? 1 : nI) FVI_ZERO(I.rhs, R, nC)
    FVI_ZERO(I.gross, R, nC) FVI_ZERO(I.w0, R, nC) FVI_ZERO(I.w1, R, nC) FVI_ZERO(I.w2, R, nC)
    const std::size_t nMax = std::max({nC, nI, nB, std::size_t(1)});
    FVI_ZERO(I.wf, R, nMax) FVI_ZERO(I.wb, R, 6 * nB) FVI_ZERO(I.wU, R, 3 * nC) FVI_ZERO(I.ones, R, nMax)
    if (I.thermal) {
        FVI_UPR(I.Tc, T0) FVI_UPR(I.bT, s.bT) FVI_UP(I.bTfix, s.bTfix)
        FVI_ZERO(I.Ta, R, nC) FVI_ZERO(I.Ts, R, nC) FVI_ZERO(I.Tb, R, nB) FVI_ZERO(I.Tba, R, nB) FVI_ZERO(I.Tbs, R, nB)
        FVI_ZERO(I.GT, R, s.orthogonal ? 1 : 3 * nC) FVI_ZERO(I.FT, R, nI) FVI_ZERO(I.FTb, R, nB) FVI_ZERO(I.Bf, R, nI)
        FVI_ZERO(I.dpB, R, nB)
    }
    FVI_ZERO(I.part, double, std::size_t(la::internal::kMaxDotBatch) * std::size_t(la::internal::dotParts(nMax)))
    FVI_ZERO(I.outv, double, std::size_t(la::internal::kMaxDotBatch))
#undef FVI_UP
#undef FVI_UPR
#undef FVI_ZERO
    kFillR<R><<<grid1(nMax), kT>>>(nMax, R(1.0), I.ones.data());

    IMesh<R>& m = I.m;
    m.nC = s.nC;
    m.nI = s.nI;
    m.nB = s.nB;
    m.own = I.own.data();
    m.nei = I.nei.data();
    m.bOwn = I.bOwn.data();
    m.cStart = I.cStart.data();
    m.cEnt = I.cEnt.data();
    m.S = I.S.data();
    m.d = I.d.data();
    m.rO = I.rO.data();
    m.id2 = I.id2.data();
    m.cS = I.cS.data();
    m.wO = I.wO.data();
    m.a = I.a.data();
    m.k = I.k.data();
    m.bS = I.bS.data();
    m.bR = I.bR.data();
    m.bN = I.bN.data();
    m.bCs = I.bCs.data();
    m.bA = I.bA.data();
    m.bDn = I.bDn.data();
    m.bMagS = I.bMagS.data();
    m.bU = I.bU.data();
    m.bP = I.bP.data();
    m.bUc = I.bUc.data();
    m.V = I.V.data();
    m.gO = I.gO.data();
    m.gN = I.gN.data();
    m.bCode = I.bCode.data();
    m.bForce = I.bForce.data();
    m.bTfix = I.bTfix.data();
    m.bT = I.bT.data();

    // The pressure matrix: geometric, so the hierarchy is built and factored once.
    auto shape = dns::la::denseShape(1);
    if (!shape) return std::unexpected(shape.error());
    auto hm = dns::la::BlockMatrix::zeros(g, *shape);
    hm.diag = s.diagA;
    hm.off = s.slotA;
    const auto pat = dns::la::ellPattern(g);
    auto ap = la::BlockEll<R>::make(pat, *shape);
    if (!ap) return std::unexpected(ap.error());
    I.A.emplace(std::move(*ap));
    if (auto r = I.A->upload(hm); !r) return std::unexpected("fv incomp: pressure matrix: " + r.error());
    auto h = dns::la::amgHierarchy(g, dns::la::amgStrength(hm), 1, dns::la::AmgSetup{});
    if (!h) return std::unexpected("fv incomp: AMG setup: " + h.error());
    la::AmgOptions ao;
    ao.smoother = la::AmgSmoother::Jacobi;
    ao.sweeps = 2;
    ao.inner = 1;
    ao.constantNullSpace = s.pureNeumann;
    auto amg = la::Amg<R>::make(*I.A, *h, ao);
    if (!amg) return std::unexpected("fv incomp: AMG: " + amg.error());
    I.amg.emplace(std::move(*amg));
    if (auto r = I.amg->refresh(); !r) return std::unexpected("fv incomp: AMG refresh: " + r.error());
    auto kr = la::Krylov<R>::make(nC, 0);
    if (!kr) return std::unexpected(kr.error());
    I.kr.emplace(std::move(*kr));

    if (c.lowMach) return I.check("init");
    // The initial state: boundary values (phi = 0 reads as outflow), then one
    // projection of the interpolated flux at c = 1; p is not a pressure yet.
    kEvalB<R><<<grid1(nB), kT>>>(m, I.U.data(), I.phiB.data(), I.Ub.data());
    (void)hipMemcpy(I.Ub0.data(), I.Ub.data(), 3 * nB * sizeof(R), hipMemcpyDeviceToDevice);
    (void)hipMemcpy(I.Ubs.data(), I.Ub.data(), 3 * nB * sizeof(R), hipMemcpyDeviceToDevice);
    kFlux<R><<<grid1(nI), kT>>>(m, I.U.data(), I.phiS.data());
    kFluxB<R><<<grid1(nB), kT>>>(m, I.U.data(), I.Ub.data(), I.phiSB.data());
    if (s.hasConvective)
        if (auto r = I.closeConvective(I.Ub.data()); !r) return r;
    if (auto r = I.project(I.U.data(), R(1.0)); !r) return r;
    kEvalB<R><<<grid1(nB), kT>>>(m, I.U.data(), I.phiB.data(), I.Ub.data());
    (void)hipMemset(I.p.data(), 0, nC * sizeof(R));
    (void)hipMemset(I.Gp.data(), 0, 3 * nC * sizeof(R));
    if (I.thermal) kEvalT<R><<<grid1(nB), kT>>>(m, I.Tc.data(), I.Tb.data());
    return I.check("init");
}

} // namespace

DeviceFvIncomp::DeviceFvIncomp() = default;
DeviceFvIncomp::DeviceFvIncomp(DeviceFvIncomp&&) noexcept = default;
DeviceFvIncomp& DeviceFvIncomp::operator=(DeviceFvIncomp&&) noexcept = default;
DeviceFvIncomp::~DeviceFvIncomp() = default;

std::expected<DeviceFvIncomp, std::string> DeviceFvIncomp::make(const FvIncompSetup& s, const dns::la::CellGraph& g,
                                                                const FvIncompControls& c,
                                                                const std::vector<double>& U0,
                                                                const std::vector<double>& T0)
{
    if (auto ok = fviModulesCheck(fviModulesOf(c, !T0.empty())); !ok)
        return std::unexpected("fv incomp: " + ok.error() + " (docs/fvincomp.md \"Incompressible combinations\")");
    DeviceFvIncomp d;
    if (c.precision == FvPrecision::Fp32) {
        auto I = std::make_unique<IncompImpl<float>>();
        if (auto r = makeImpl(*I, s, g, c, U0, T0); !r) return std::unexpected(r.error());
        d.p_ = std::move(I);
    } else {
        auto I = std::make_unique<IncompImpl<double>>();
        if (auto r = makeImpl(*I, s, g, c, U0, T0); !r) return std::unexpected(r.error());
        d.p_ = std::move(I);
    }
    return d;
}

std::expected<double, std::string> DeviceFvIncomp::stableDt(double cfl) { return p_->stableDt(cfl); }
std::expected<FvIncompStepLog, std::string> DeviceFvIncomp::step(double dt) { return p_->step(dt); }
std::expected<FvIncompDiag, std::string> DeviceFvIncomp::diagnostics() { return p_->diagnostics(); }
std::expected<FvIncompBudget, std::string> DeviceFvIncomp::budget() { return p_->budget(); }
std::expected<FvIncompForces, std::string> DeviceFvIncomp::forces() { return p_->forces(); }
std::expected<FvIncompState, std::string> DeviceFvIncomp::state() const { return p_->state(); }
std::expected<std::vector<double>, std::string> DeviceFvIncomp::wallHeat() { return p_->wallHeat(); }
std::expected<std::vector<double>, std::string> DeviceFvIncomp::gradU() { return p_->gradU(); }

// --- Low-Mach (gpu/fv_incomp.hpp, docs/fvincomp.md "Low-Mach") ---------------------
namespace {

// The model's constants of a kernel. first: FvLowMachTables::first.
template <class R>
struct LmfK
{
    Transport tr;
    R invPr = 1, prGas = 1, lnA = 0, Ta = 0, qcp = 0;
    int nY = 1;
    const u8* first = nullptr;
    R stoich = 1; // the oxidizer's s (two reactants)
    // Open faces (FvLowMachTables::bOpen; null: none): the ambient a face drawing
    // gas in takes, rho_a = prGas / ambT; bfHalf = beta / 2 of the inflow face p.
    const u8* open = nullptr;
    R ambT = 1, ambRho = 1, ambY0 = 0, ambY1 = 0, bfHalf = 0.5;
    R gx = 0, gy = 0, gz = 0; // gravity (the Grav passes only)
    R invSct = 1;             // M4: 1 / Sc_t (the Eddy, Thick passes only)
};

// M4 (docs/fvincomp.md "Low-Mach LES"): the eddy fields of a stage, gpu/lowmach's
// LmEddy: mu_t per cell (null: no LES), TFLES F E and Omega per cell (null: not
// thickened). Diff: gpu/lowmach's kLmMolecular | kLmEddy | kLmThick.
template <class R>
struct LmfE
{
    const R* mut = nullptr;
    const R *FE = nullptr, *Om = nullptr;
};

enum : int
{
    kLmfMolecular = 0,
    kLmfEddy = 1,
    kLmfThick = 2
};

// gpu/lowmach lmEffD on an internal face o | n: the operands of species.hip's
// effRhoD (face values arithmetic means).
template <class R, int Diff>
__device__ inline R lmfEffD(const LmfK<R>& K, const LmfE<R>& E, std::size_t o, std::size_t n, R rd)
{
    if constexpr (Diff == kLmfMolecular) {
        return rd;
    } else if constexpr (Diff == kLmfEddy) {
        return rd + R(0.5) * (E.mut[o] + E.mut[n]) * K.invSct;
    } else {
        R r = R(0.5) * (E.FE[o] + E.FE[n]) * rd;
        if (E.mut) {
            const R om = R(0.5) * (E.Om[o] + E.Om[n]);
            r += (R(1) - om) * (R(0.5) * (E.mut[o] + E.mut[n])) * K.invSct;
        }
        return r;
    }
}

// The limiter's voters: Y_k, then T; their gradients. v[s] is species s.
template <class R>
struct LmfV
{
    const R* v[kFvLmMaxY + 1];
    const R* g[kFvLmMaxY + 1];
    int n;
};

// The compensated words of a stage: [0] rho, [1 + s] rho Y_s; lo unused plain.
template <class R>
struct LmfW
{
    const R *ah[kFvLmMaxY + 1], *al[kFvLmMaxY + 1], *bh[kFvLmMaxY + 1], *bl[kFvLmMaxY + 1];
    R *oh[kFvLmMaxY + 1], *ol[kFvLmMaxY + 1];
};

// a + (cb (b - a) + cr r): gpu/lowmach's lmPut, so a field uniform in a and b with r
// = 0 stays bitwise uniform.
template <class R>
__device__ inline R lmfPut(R a, R b, R cb, R cr, R r)
{
    return a + (cb * (b - a) + cr * r);
}

// rho D = lambda/cp = mu(T)/Pr (Le = 1) at face temperature t.
template <class R>
__device__ inline R lmfRhoD(const LmfK<R>& K, R t)
{
    R mu, kap;
    fvTransportAt(K.tr, t, mu, kap);
    return mu * K.invPr;
}

// The shared limiter of internal face f carrying mdot.
template <class R>
__device__ inline R lmfTheta(const IMesh<R>& m, const LmfK<R>& K, const LmfV<R>& V, std::size_t f, std::size_t o,
                             std::size_t n, R mdot, const R d[3])
{
    if (K.first[f] & (mdot >= R(0.0) ? 1 : 2)) return R(1.0);
    const std::size_t nC = std::size_t(m.nC);
    R th = R(2.0);
    for (int v = 0; v < V.n; ++v) {
        R gP[3], gN[3];
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            gP[q] = V.g[v][std::size_t(q) * nC + o];
            gN[q] = V.g[v][std::size_t(q) * nC + n];
        }
        th = fvThetaVote(th, mdot, V.v[v][o], V.v[v][n], gP, gN, d);
    }
    return th;
}

// An open face's sign-of-flux rule (gpu/lowmach's open faces, docs/fvincomp.md "Open
// faces") on the boundary values the evaluations just wrote: a face whose flux of
// the stage's input points INTO the mesh (phi_b < 0) carries the quiescent ambient,
// T_a, Y_a, and its own velocity phi_b / (|S| rho_a) along n (tangential 0: air at
// rest, the structured inflow ghost's antimirror); outflow keeps zero gradient.
template <class R>
__global__ void kLmfOpenB(IMesh<R> m, LmfK<R> K, const R* phiB, R* Ub, R* Tb, R* Yb0, R* Yb1)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB || !K.open[b] || !(phiB[b] < R(0.0))) return;
    const std::size_t nB = std::size_t(m.nB), bb = std::size_t(b);
    Tb[b] = K.ambT;
    Yb0[b] = K.ambY0;
    if (Yb1) Yb1[b] = K.ambY1;
    const R un = phiB[b] / (m.bMagS[b] * K.ambRho);
#pragma unroll
    for (int q = 0; q < 3; ++q) Ub[std::size_t(q) * nB + bb] = un * m.bN[std::size_t(q) * nB + bb];
}

// An open face's pressure (gpu/lowmach openFaceP): -beta rho_a u_n^2 / 2 while it
// draws gas in (the total-pressure inflow, Moghadam's stabilization), 0 out.
template <class R>
__global__ void kLmfOpenP(IMesh<R> m, LmfK<R> K, const R* phiB, R* bP)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB || !K.open[b]) return;
    const R mOut = phiB[b] / m.bMagS[b];
    bP[b] = mOut < R(0.0) ? -K.bfHalf * mOut * mOut / K.ambRho : R(0.0);
}

// gpu/lowmach's ONE buoyancy term addBuoyancyLm on faces: the face force (rho_f -
// rho_a) g . S, rho_f the face's linear density (kFlux's interpolation; the
// two-cell mean on a uniform mesh), into phi* with c and kept for the
// reconstruction (fp - Bf: a hydrostatic state cancels face by face).
template <class R>
__global__ void kLmfBuoyF(IMesh<R> m, LmfK<R> K, R c, const R* rho, R* phiS, R* Bf)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nI = std::size_t(m.nI), ff = std::size_t(f);
    const R w = m.wO[f], ro = rho[m.own[f]], rn = rho[m.nei[f]];
    const R gS = K.gx * m.S[ff] + K.gy * m.S[nI + ff] + K.gz * m.S[2 * nI + ff];
    const R B = (w * (ro - rn) + rn - K.ambRho) * gS;
    Bf[f] = B;
    phiS[f] += c * B;
}

// The boundary faces of the same term. A p-fixed face (outflow, open) carries the
// force of its upwind density (the cell's; the ambient's on an open face's inflow:
// 0) as gpu/lowmach's x-hi face; every other face is Neumann and carries none, its
// hydrostatic image dpB = (rho_b - rho_a) g . r (kGradP).
template <class R>
__global__ void kLmfBuoyB(IMesh<R> m, LmfK<R> K, R c, const R* rho, const R* Tb, const R* phiB, R* phiSB, R* BfB,
                          R* dpB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), bb = std::size_t(b);
    const R rP = rho[m.bOwn[b]];
    const u8 code = m.bCode[b];
    if (code == kOutflow) {
        const R rb = K.open && K.open[b] && phiB[b] < R(0.0) ? K.ambRho : rP;
        const R gS = K.gx * m.bS[bb] + K.gy * m.bS[nB + bb] + K.gz * m.bS[2 * nB + bb];
        const R B = (rb - K.ambRho) * gS;
        BfB[b] = B;
        phiSB[b] += c * B;
        dpB[b] = 0;
        return;
    }
    const R rb = code == kInflow ? K.prGas / Tb[b] : rP;
    BfB[b] = 0;
    dpB[b] = (rb - K.ambRho) * (K.gx * m.bR[bb] + K.gy * m.bR[nB + bb] + K.gz * m.bR[2 * nB + bb]);
}

// Cell pass: T, Y_k, omega, u of a state. TwoR: fuel 0, oxidizer 1, gpu/lowmach's
// kLmCellPass<true, Phi> rate; else the one-reactant rate, species 1 passive.
// Eof (M4 TFLES, one reactant): omega *= E/F (gpu/lowmach kLmCellPass<.., true>).
template <class R, bool TwoR, bool Phi, bool Eof = false>
__global__ void __launch_bounds__(256) kLmfCell(IMesh<R> m, LmfK<R> K, PhiRateLaw<R> law, const R* rho, const R* ry0,
                                                const R* ry1, const R* ru, R* T, R* Y0, R* Y1, R* om, R* u,
                                                const R* eof)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    const R r = rho[c], t = K.prGas / r, y = ry0[c] / r;
    T[c] = t;
    Y0[c] = y;
    if constexpr (TwoR) {
        const R y1 = ry1[c] / r;
        Y1[c] = y1;
        if constexpr (Phi)
            om[c] = arrheniusRate2(K.lnA + phiRateLn(law, y, y1), K.Ta, r, y, y1, t);
        else
            om[c] = arrheniusRate2(K.lnA, K.Ta, r, y, y1, t);
    } else {
        if (K.nY > 1) Y1[c] = ry1[c] / r;
        if constexpr (Eof)
            om[c] = arrheniusRate(K.lnA, K.Ta, r, y, t) * eof[c];
        else
            om[c] = arrheniusRate(K.lnA, K.Ta, r, y, t);
    }
#pragma unroll
    for (int q = 0; q < 3; ++q) u[std::size_t(q) * nC + cc] = ru[std::size_t(q) * nC + cc] / r;
}

// Internal face: T_f on the shared limiter, Df = rho D grad T . S.
// Diff (M4): lmfEffD on rho D; kLmfMolecular = the M3 kernel bit for bit.
template <class R, bool Corr, int Diff = kLmfMolecular>
__global__ void __launch_bounds__(256) kLmfFaceT(IMesh<R> m, LmfK<R> K, LmfV<R> V, const R* T, const R* GT, const R* phi,
                                                 R* Tf, R* Df, LmfE<R> E)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    R d[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) d[q] = m.d[std::size_t(q) * nI + ff];
    const R mdot = phi[f], to = T[o], tn = T[n], cs = m.cS[f];
    Tf[f] = fvFaceYShared(mdot, to, tn, lmfTheta(m, K, V, ff, o, n, mdot, d));
    R diff = cs * (tn - to);
    if constexpr (Corr) {
        R gS = 0, gd = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const R gb = R(0.5) * (GT[std::size_t(i) * nC + o] + GT[std::size_t(i) * nC + n]);
            gS += gb * m.S[std::size_t(i) * nI + ff];
            gd += gb * d[i];
        }
        diff = gS + (tn - to - gd) * cs;
    }
    Df[f] = lmfEffD<R, Diff>(K, E, o, n, lmfRhoD(K, R(0.5) * (to + tn))) * diff;
}

// Boundary conduction rho D(T_b) grad T . S on a fixed-T face (a zero-flux face is
// EXACTLY 0), and Eb = phi_b T_b - that: the face's enthalpy flux out (cp units).
template <class R, bool Corr>
__global__ void kLmfBndT(IMesh<R> m, LmfK<R> K, const R* T, const R* Tb, const R* GT, const R* phiB, R* DfB, R* Eb)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const std::size_t P = std::size_t(m.bOwn[b]);
    R v = 0;
    if (m.bTfix[b]) {
        const R dT = Tb[b] - T[P], cs = m.bCs[b];
        R x = dT * cs;
        if constexpr (Corr) {
            R gS = 0, gr = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const R g = GT[std::size_t(i) * nC + P];
                gS += g * m.bS[std::size_t(i) * nB + bb];
                gr += g * m.bR[std::size_t(i) * nB + bb];
            }
            x = gS + (dT - gr) * cs;
        }
        v = lmfRhoD(K, Tb[b]) * x;
    }
    DfB[b] = v;
    Eb[b] = phiB[b] * Tb[b] - v;
}

// The temperature predictor and the divergence target of gpu/lowmach's
// kLmTemperature on faces: R_T = [-sum_f m (T_f - T_c) + sum_f Df] / V + (q/cp) om) /
// rho_b, T* = lmfPut, V D the cancellation-free target. Ns = V rho_b R_T.
template <class R>
__global__ void __launch_bounds__(256) kLmfGatherT(IMesh<R> m, LmfK<R> K, R ca, R cb, R cr, const R* rhoA, const R* rhoB,
                                                   const R* T, const R* Tb, const R* om, const R* phi, const R* phiB,
                                                   const R* Tf, const R* Df, const R* DfB, R* Ts, R* VD, R* Ns)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const R tc = T[c];
    R conv = 0, dif = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            const R x = phi[f] * (Tf[f] - tc);
            if (code & 1) {
                conv += x;
                dif -= Df[f];
            } else {
                conv -= x;
                dif += Df[f];
            }
        } else {
            const std::size_t b = std::size_t(-code - 1);
            conv -= phiB[b] * (Tb[b] - tc);
            dif += DfB[b];
        }
    }
    const R V = m.V[c], ra = rhoA[c], rb = rhoB[c];
    const R N = (conv + dif) / V + K.qcp * om[c];
    const R RT = N / rb, ts = lmfPut(K.prGas / ra, tc, cb, cr, RT);
    Ts[c] = ts;
    const R r0 = lmfPut(ra, rb, cb, R(0.0), R(0.0)), dr = ra - rb;
    VD[c] = V * (r0 * RT / ts + (K.prGas * ca * cb) * (dr * dr) / (ra * rb * cr * ts));
    Ns[c] = V * N;
}

// Internal face: mdot u_mean - tau . S, the conservative central momentum flux
// (kinetic-energy consistent with the continuity on the same mdot). mu at the face's
// mean T; tau the full Newtonian stress of the AlongD face gradient.
// Les (M4): mu + mu_t,f (face mean) in the stress; false = the M3 kernel bit for bit.
template <class R, bool Les = false>
__global__ void __launch_bounds__(256) kLmfFaceM(IMesh<R> m, LmfK<R> K, const R* T, const R* u, const R* G, const R* phi,
                                                 R* F, const R* mut)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    R S[3], d[3], um[3], gu[3][3];
    const R id2 = m.id2[f];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        S[i] = m.S[std::size_t(i) * nI + ff];
        d[i] = m.d[std::size_t(i) * nI + ff];
    }
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        const R uo = u[std::size_t(q) * nC + o], un = u[std::size_t(q) * nC + n];
        um[q] = R(0.5) * (uo + un);
        R gb[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const std::size_t at = std::size_t(3 * i + q) * nC;
            gb[i] = R(0.5) * (G[at + o] + G[at + n]);
        }
        const R def = (un - uo - (gb[0] * d[0] + gb[1] * d[1] + gb[2] * d[2])) * id2;
#pragma unroll
        for (int i = 0; i < 3; ++i) gu[q][i] = gb[i] + def * d[i];
    }
    R mu, kap;
    fvTransportAt(K.tr, R(0.5) * (T[o] + T[n]), mu, kap);
    if constexpr (Les) mu = mu + R(0.5) * (mut[o] + mut[n]);
    R ts[3];
    fvTraction(gu, mu, S, ts);
    const R ph = phi[f];
#pragma unroll
    for (int j = 0; j < 3; ++j) F[std::size_t(j) * nI + ff] = ph * um[j] - ts[j];
}

// Boundary momentum flux: phi_b u_b - tau . S. Wall and inflow: the full stress of
// the cell gradient with its one-sided r part; slip: the normal stress; outflow none.
// An open face (K.open): gpu/lowmach's open face. Normal: p_b and the cell's normal
// viscous stress (zero normal-stress gradient); tangential: the full stress toward its
// u_b (zero gradient out, still air in), the structured edge stresses.
template <class R>
__global__ void __launch_bounds__(256) kLmfBndM(IMesh<R> m, LmfK<R> K, const R* Tb, const R* u, const R* Ub, const R* G,
                                                const R* phiB, R* Fb)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const std::size_t P = std::size_t(m.bOwn[b]);
    const u8 code = m.bCode[b];
    const R ph = phiB[b];
    R ts[3] = {0, 0, 0};
    const bool openF = K.open && K.open[b];
    if (code != kOutflow || openF) {
        R mu, kap;
        fvTransportAt(K.tr, Tb[b], mu, kap);
        if (code == kSlip) {
#pragma unroll
            for (int j = 0; j < 3; ++j)
                ts[j] = mu * m.bCs[b] * (Ub[std::size_t(j) * nB + bb] - u[std::size_t(j) * nC + P]);
        } else {
            R S[3], r[3], gu[3][3];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                S[i] = m.bS[std::size_t(i) * nB + bb];
                r[i] = m.bR[std::size_t(i) * nB + bb];
            }
            const R ir2 = R(1.0) / (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                R g[3];
#pragma unroll
                for (int i = 0; i < 3; ++i) g[i] = G[std::size_t(3 * i + q) * nC + P];
                const R du = Ub[std::size_t(q) * nB + bb] - u[std::size_t(q) * nC + P];
                const R def = (du - (g[0] * r[0] + g[1] * r[1] + g[2] * r[2])) * ir2;
#pragma unroll
                for (int i = 0; i < 3; ++i) gu[q][i] = g[i] + def * r[i];
            }
            fvTraction(gu, mu, S, ts);
            if (openF) {
                // The normal part is the CELL's (the cell gradient: gpu/lowmach's zero
                // normal-stress gradient at an open face), the tangential the one-sided.
                R n[3], gc[3][3], tc[3];
#pragma unroll
                for (int q = 0; q < 3; ++q)
#pragma unroll
                    for (int i = 0; i < 3; ++i) gc[q][i] = G[std::size_t(3 * i + q) * nC + P];
                fvTraction(gc, mu, S, tc);
#pragma unroll
                for (int j = 0; j < 3; ++j) n[j] = m.bN[std::size_t(j) * nB + bb];
                const R tn = ts[0] * n[0] + ts[1] * n[1] + ts[2] * n[2];
                const R cn = tc[0] * n[0] + tc[1] * n[1] + tc[2] * n[2];
#pragma unroll
                for (int j = 0; j < 3; ++j) ts[j] += (cn - tn) * n[j];
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) Fb[std::size_t(j) * nB + bb] = ph * Ub[std::size_t(j) * nB + bb] - ts[j];
}

// (rho u)* = a + (cb (b - a) + cr (-sum +-F / V)).
template <class R>
__global__ void __launch_bounds__(256) kLmfGatherM(IMesh<R> m, R cb, R cr, const R* a, const R* b, const R* F,
                                                   const R* Fb, R* out)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB), cc = std::size_t(c);
    R sum[3] = {0, 0, 0};
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                if (code & 1) sum[j] -= F[std::size_t(j) * nI + f];
                else sum[j] += F[std::size_t(j) * nI + f];
            }
        } else {
            const std::size_t bf = std::size_t(-code - 1);
#pragma unroll
            for (int j = 0; j < 3; ++j) sum[j] += Fb[std::size_t(j) * nB + bf];
        }
    }
    const R V = m.V[c];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const std::size_t i = std::size_t(j) * nC + cc;
        out[i] = lmfPut(a[i], b[i], cb, cr, -sum[j] / V);
    }
}

// Species flux of an internal face on the PROJECTED mdot: mdot Y_f - rho D grad Y . S,
// Y_f on the one theta. Fy[s nI + f].
// Diff (M4): lmfEffD on rho D; kLmfMolecular = the M3 kernel bit for bit.
template <class R, bool Corr, int Diff = kLmfMolecular>
__global__ void __launch_bounds__(256) kLmfFaceY(IMesh<R> m, LmfK<R> K, LmfV<R> V, const R* T, const R* phi, R* Fy,
                                                 LmfE<R> E)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    R d[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) d[q] = m.d[std::size_t(q) * nI + ff];
    const R mdot = phi[f], cs = m.cS[f];
    const R th = lmfTheta(m, K, V, ff, o, n, mdot, d);
    const R kd = lmfEffD<R, Diff>(K, E, o, n, lmfRhoD(K, R(0.5) * (T[o] + T[n])));
    for (int s = 0; s < K.nY; ++s) {
        const R yo = V.v[s][o], yn = V.v[s][n];
        R diff = cs * (yn - yo);
        if constexpr (Corr) {
            R gS = 0, gd = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const R gb = R(0.5) * (V.g[s][std::size_t(i) * nC + o] + V.g[s][std::size_t(i) * nC + n]);
                gS += gb * m.S[std::size_t(i) * nI + ff];
                gd += gb * d[i];
            }
            diff = gS + (yn - yo - gd) * cs;
        }
        Fy[std::size_t(s) * nI + ff] = mdot * fvFaceYShared(mdot, yo, yn, th) - kd * diff;
    }
}

// Boundary species flux, Fyb[s nB + b]: phi_b Y_b, Fick on an inflow face; a wall
// and a slip face carry EXACTLY 0.
template <class R, bool Corr>
__global__ void kLmfBndY(IMesh<R> m, LmfK<R> K, LmfV<R> V, const u8* fix, const R* Tb, const R* Yb0, const R* Yb1,
                         const R* phiB, R* Fyb)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const std::size_t P = std::size_t(m.bOwn[b]);
    const u8 code = m.bCode[b];
    for (int s = 0; s < K.nY; ++s) {
        const std::size_t at = std::size_t(s) * nB + bb;
        if (code == kWall || code == kSlip) {
            Fyb[at] = 0;
            continue;
        }
        R yb = s == 0 ? Yb0[b] : Yb1[b];
        // An open face: upwind by the sign of the PROJECTED flux this kernel carries
        // (gpu/lowmach kLmScalars: ambient | own), not by the stage input's: a face
        // whose flux turned outward would carry the ambient's Y out (Y_O < 0, measured).
        if (K.open && K.open[b]) yb = phiB[b] < R(0.0) ? (s == 0 ? K.ambY0 : K.ambY1) : V.v[s][P];
        const R adv = phiB[b] * yb;
        if (!fix[b]) {
            Fyb[at] = adv;
            continue;
        }
        const R dy = yb - V.v[s][P], cs = m.bCs[b];
        R x = dy * cs;
        if constexpr (Corr) {
            R gS = 0, gr = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const R g = V.g[s][std::size_t(i) * nC + P];
                gS += g * m.bS[std::size_t(i) * nB + bb];
                gr += g * m.bR[std::size_t(i) * nB + bb];
            }
            x = gS + (dy - gr) * cs;
        }
        Fyb[at] = adv - lmfRhoD(K, Tb[b]) * x;
    }
}

// Continuity and species in ONE source: rho (row 0) and rho Y_s (rows 1 + s) take
// the same arithmetic, so a uniform power-of-two Y scales rho's bits exactly. Fuel
// sink -omega on row 1. Comp: gpu/lowmach's lmPutWord. Contraction off: the fuel row
// fused its write differently, 2e-8 off rho's bits (L1).
template <class R, bool Comp, bool TwoR>
__global__ void __launch_bounds__(256) kLmfGatherS(IMesh<R> m, int nY, R cb, R cr, R stoich, LmfW<R> W, const R* om,
                                                   const R* phi, const R* phiB, const R* Fy, const R* Fyb)
{
#pragma clang fp contract(off)
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nI = std::size_t(m.nI), nB = std::size_t(m.nB), cc = std::size_t(c);
    R sum[kFvLmMaxY + 1] = {0, 0, 0};
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            if (code & 1) {
                sum[0] -= phi[f];
                for (int s = 0; s < nY; ++s) sum[1 + s] -= Fy[std::size_t(s) * nI + f];
            } else {
                sum[0] += phi[f];
                for (int s = 0; s < nY; ++s) sum[1 + s] += Fy[std::size_t(s) * nI + f];
            }
        } else {
            const std::size_t bf = std::size_t(-code - 1);
            sum[0] += phiB[bf];
            for (int s = 0; s < nY; ++s) sum[1 + s] += Fyb[std::size_t(s) * nB + bf];
        }
    }
    const R V = m.V[c];
    for (int v = 0; v < 1 + nY; ++v) {
        const R rate = -sum[v] / V;
        R src = rate;
        if (v == 1) src = rate - om[c];
        if constexpr (TwoR)
            if (v == 2) src = rate - stoich * om[c]; // gpu/lowmach: div - s omega
        const R inc = cr * src;
        if constexpr (Comp) {
            const Word2 w = internal::combineWord2(cb, W.ah[v][cc], W.al[v][cc], W.bh[v][cc], W.bl[v][cc], inc);
            W.oh[v][cc] = w.hi;
            W.ol[v][cc] = w.lo;
        } else {
            const R a = W.ah[v][cc];
            W.oh[v][cc] = a + (cb * (W.bh[v][cc] - a) + inc);
        }
    }
}

// The dt rates of a cell. diffF = max(4/3, 1/Pr). The diffusive sum is |S|^2/(S.d)
// (the pressure matrix's a), not corr.S: equal on an orthogonal mesh, the larger past
// it, where the explicit gradient correction adds to the stiffness. corr.S diverged in
// ~10 steps at 67 deg non-orthogonality, dt 2x above stable (issues.md LF2).
// TwoR: gpu/lowmach's LmChemJ<true, Phi>, k rho (Y_O + s Y_F + q/cp Y_F Y_O Ta/T^2), Y
// floored at 0, the unfloored Y feeding ln C.
// Eddy (M4): diff = (mu max(4/3, FE_c/Pr) + mu_t,c max(4/3, 1/Sc_t)) sum / (rho V), E.FE
// null = 1, E.mut null = 0 (invPr = 1/Pr, fT = max(4/3, 1/Sc_t)); false = the M3 rate.
template <class R, bool TwoR, bool Phi, bool Eddy = false>
__global__ void __launch_bounds__(256) kLmfRate(IMesh<R> m, LmfK<R> K, PhiRateLaw<R> law, R diffF, const R* rho,
                                                const R* T, const R* Y0, const R* Y1, const R* Tb, const R* phi,
                                                const R* phiB, R* conv, R* diff, R* chem, LmfE<R> E, R fT)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const R r = rho[c], t = T[c];
    R sc = 0, sd = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = faceOf(code);
            const R other = rho[(code & 1) ? m.own[f] : m.nei[f]];
            sc += fabs(phi[f]) / fmin(R(0.5) * (r + other), r);
            sd += m.a[f];
        } else {
            const std::size_t b = std::size_t(-code - 1);
            const u8 bc = m.bCode[b];
            const R rb = bc == kInflow                                      ? K.prGas / Tb[b]
                         : K.open && K.open[b] && phiB[b] < R(0.0) ? K.ambRho
                                                                     : r;
            sc += fabs(phiB[b]) / fmin(rb, r);
            if (bc != kOutflow) sd += m.bA[b];
        }
    }
    R mu, kap;
    fvTransportAt(K.tr, t, mu, kap);
    const R V = m.V[c], y = fmax(Y0[c], R(0.0));
    conv[c] = R(0.5) * sc / V;
    if constexpr (Eddy) {
        const R fe = E.FE ? E.FE[c] : R(1.0), mt = E.mut ? E.mut[c] : R(0.0);
        diff[c] = (mu * fmax(R(4.0 / 3.0), fe * K.invPr) + mt * fT) * sd / (r * V);
    } else {
        diff[c] = mu * diffF * sd / (r * V);
    }
    if constexpr (TwoR) {
        const R yo = fmax(Y1[c], R(0.0));
        R la = K.lnA;
        if constexpr (Phi) la += phiRateLn(law, Y0[c], Y1[c]);
        const R J = exp(la - K.Ta / t) * r * (yo + K.stoich * y + K.qcp * y * yo * K.Ta / (t * t));
        chem[c] = isfinite(J) ? J : R(HUGE_VAL);
    } else {
        chem[c] = exp(K.lnA - K.Ta / t) * (R(1.0) + K.qcp * y * K.Ta / (t * t));
    }
}

// The residuals of the last stage per cell: |div m - D|, |rho T* / P - 1|.
template <class R>
__global__ void __launch_bounds__(256) kLmfErr(IMesh<R> m, LmfK<R> K, const R* rho, const R* Ts, const R* VD, const R* phi,
                                               const R* phiB, R* eDiv, R* eEos)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R s = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) s += signOf<R>(code) * phi[faceOf(code)];
        else s += phiB[-code - 1];
    }
    eDiv[c] = fabs((s - VD[c]) / m.V[c]);
    eEos[c] = fabs(rho[c] * Ts[c] / K.prGas - R(1.0));
}

// out[c] = Ns[c] - T[c] sum_f +-m: V (rho R_T + T R_rho) of the semi-discrete balance;
// gross[c]: the larger of its two terms.
template <class R>
__global__ void __launch_bounds__(256) kLmfBudget(IMesh<R> m, const R* T, const R* Ns, const R* phi, const R* phiB, R* out,
                                                  R* gross)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R s = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) s += signOf<R>(code) * phi[faceOf(code)];
        else s += phiB[-code - 1];
    }
    const R x = T[c] * s;
    out[c] = Ns[c] - x;
    gross[c] = fmax(fabs(Ns[c]), fabs(x));
}

// --dt-rule cell: the per-cell positivity rate 2 C_c + D_c of kLmfRate's two rates
// (gpu/lowmach's LmPosRate: the MC face value reaches twice the upwind slope).
template <class R>
__global__ void __launch_bounds__(256) kLmfPos(int n, const R* conv, const R* diff, R* out)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= n) return;
    const R s = R(2.0) * conv[c] + diff[c];
    out[c] = isfinite(s) ? s : R(HUGE_VAL);
}

// The flame index of two gradients (gpu/derived kFlameIndex's expression).
template <class R>
__global__ void __launch_bounds__(256) kLmfIndex(int n, const R* gF, const R* gO, R floor, R* out)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= n) return;
    const std::size_t N = std::size_t(n), cc = std::size_t(c);
    const R fx = gF[cc], fy = gF[N + cc], fz = gF[2 * N + cc];
    const R ox = gO[cc], oy = gO[N + cc], oz = gO[2 * N + cc];
    const R nf = sqrt(fx * fx + fy * fy + fz * fz), no = sqrt(ox * ox + oy * oy + oz * oz);
    out[c] = nf < floor || no < floor ? R(0) : (fx * ox + fy * oy + fz * oz) / (nf * no);
}

// --- M4: LES and TFLES on faces (docs/fvincomp.md "Low-Mach LES") -----------------
// mu_t = rho (cs Delta)^2 rate(g) (gpu/lowmach kLmNut's expression; cd = (cs Delta)^2 per
// cell), g[q][i] = d_i u_q the least-squares gradient of the cell velocity (I.G).
template <class R>
__global__ void __launch_bounds__(256) kLmfNut(int nC, int sigma, const R* cd, const R* G, const R* rho, R* mut)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    const std::size_t n = std::size_t(nC), cc = std::size_t(c);
    R gd[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int q = 0; q < 3; ++q) gd[q][i] = G[std::size_t(3 * i + q) * n + cc];
    mut[c] = rho[c] * (cd[c] * sgsRate<R>(sigma, gd));
}

// The TFLES kernels' fixed arguments (fv_comp's TfView).
template <class R>
struct LmfTf
{
    int rad = 0, uniform = 0, charlette = 0;
    R lnAs = 0, TaS = 0, beta = 0, Fmax = 1, deltaL = 1, nuU = 1, effBeta = 0.5;
};

// gpu/tfles kTflesSensorRaw: tanh(beta w_s), w_s the rate at TaS of the fuel; curl u off
// I.G for the efficiency.
template <class R>
__global__ void kLmfTflesRaw(int nC, LmfTf<R> t, const R* rho, const R* ry0, const R* T, const R* G, R* raw, R* curl)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    const std::size_t n = std::size_t(nC), cc = std::size_t(c);
    if (!t.uniform) {
        const R r = rho[c];
        raw[c] = tanh(t.beta * arrheniusRate(t.lnAs, t.TaS, r, ry0[c] / r, T[c]));
    }
    if (t.charlette) {
        auto g = [&](int i, int q) { return G[std::size_t(3 * i + q) * n + cc]; }; // d_i u_q
        curl[cc] = g(1, 2) - g(2, 1);
        curl[n + cc] = g(2, 0) - g(0, 2);
        curl[2 * n + cc] = g(0, 1) - g(1, 0);
    }
}

// One pass of the T-gated max filter along a walked axis (gpu/tfles kTflesMaxDilate on
// fvTflesWalk's face walk): out = max of in over the cells reached in rad steps each way
// with T >= T_c (c included). next: the slot's two direction tables, -1 stops.
template <class R>
__global__ void kLmfTflesDilate(int nC, int rad, const int* next, const R* T, const R* in, R* out)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= nC) return;
    const R Tc = T[c];
    R mx = in[c];
    for (int dir = 0; dir < 2; ++dir) {
        const int* nx = next + std::size_t(dir) * std::size_t(nC);
        int q = c;
        for (int k = 0; k < rad; ++k) {
            q = nx[q];
            if (q < 0) break;
            if (T[q] >= Tc) mx = fmax(mx, in[q]);
        }
    }
    out[c] = mx;
}

// Colin's grid u' = c2 h^3 |lap(curl u)| (coef per cell), the compact Laplacian (1/V)
// sum_f cS_f (f_N - f_P) over the internal faces (gpu/tfles kTflesColinU's 7 points on a box).
template <class R>
__global__ void __launch_bounds__(256) kLmfTflesColin(IMesh<R> m, const R* coef, const R* curl, R* uh)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC);
    R s2 = R(0.0);
    for (int k = 0; k < 3; ++k) {
        const R* f = curl + std::size_t(k) * nC;
        R l = R(0.0);
        for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
            const int code = m.cEnt[e];
            if (code < 0) continue;
            const std::size_t fc = faceOf(code);
            const std::size_t o = std::size_t((code & 1) ? m.own[fc] : m.nei[fc]);
            l += m.cS[fc] * (f[o] - f[c]);
        }
        l /= m.V[c];
        s2 += l * l;
    }
    uh[c] = coef[c] * sqrt(s2);
}

// gpu/tfles kTflesCombine: fe = F E, eof = E / F, om = Omega; W null = uniform sensor.
template <class R>
__global__ void kLmfTflesCombine(int nC, LmfTf<R> t, const R* W, const R* uh, const R* hc, const u8* edge, R* fe, R* eof,
                                 R* om)
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

// One RK buffer set of the low-Mach state. lo: FP32 only.
template <class R>
struct LmfWords
{
    DeviceBuffer<R> rho, rlo, ru;
    DeviceBuffer<R> ry[kFvLmMaxY], ylo[kFvLmMaxY];
};

constexpr int kLmfTally = 2 + kFvLmMaxY + 1; // mass, y.., burn, enthalpy
using LmfTally = std::array<double, kLmfTally>;

} // namespace

struct DeviceFvLowMach::Impl
{
    virtual ~Impl() = default;
    virtual std::expected<FvLowMachDt, std::string> stableDt(double cfl, double K, double posK) = 0;
    virtual std::expected<FvLowMachFaces, std::string> faces() = 0;
    virtual std::expected<std::vector<double>, std::string> flameIndex(double floor) = 0;
    virtual std::expected<FvIncompStepLog, std::string> step(double dt) = 0;
    virtual std::expected<FvLowMachTotals, std::string> totals() = 0;
    virtual std::expected<FvLowMachBudget, std::string> budget() = 0;
    virtual std::expected<FvLowMachCells, std::string> cells() = 0;
    virtual std::expected<FvLowMachEddy, std::string> eddyStats() = 0;
    virtual std::expected<void, RunError> save(Checkpoint&) const = 0;
    virtual std::expected<void, RunError> load(const Checkpoint&) = 0;
    virtual std::expected<void, std::string> setInflow(const std::vector<double>& bU, const std::vector<double>& rUb) = 0;
};

namespace {

template <class R>
struct LmfImpl final : DeviceFvLowMach::Impl
{
    static constexpr bool Comp = std::is_same_v<R, float>;
    IncompImpl<R> I; // mesh view, reductions, the projection
    LmfK<R> K;
    PhiRateLaw<R> law; // C2b's ln C table (two reactants, law.n > 0)
    FvLowMachModel model;
    IMesh<R> mY; // bT, bTfix read Y's tables (kEvalT)
    DeviceBuffer<R> bT, bY[kFvLmMaxY], rUb;
    DeviceBuffer<u8> bTfix, bYfix, first, bOpen;
    bool open = false, grav = false;
    LmfWords<R> st, q1, q2;
    LmfTally tSt{}, tQ1{}, tQ2{};
    double enthalpy0 = 0; // uploaded Sum V rho T
    // cell work: T, Y, omega, u, gradients, T*, V D, V N
    DeviceBuffer<R> T, Y[kFvLmMaxY], om, u, GT, GY[kFvLmMaxY], Ts, VD, Ns;
    // face work
    DeviceBuffer<R> Tb, Yb[kFvLmMaxY], Tf, Df, DfB, Eb, Fy, Fyb;
    // M4: LES (mu_t rides I.nut; cd = (cs Delta)^2) and TFLES (F E, E/F, Omega; the
    // sensor's ping-pong, curl, u', h, c2 h^3, the walk, the edge layers).
    bool les = false, thick = false;
    int sigma = 0, nAx = 0;
    LmfTf<R> tf;
    DeviceBuffer<R> cd, fe, eof, omT, tfA, tfB, uh, curl, hc, coef;
    DeviceBuffer<int> walk;
    DeviceBuffer<u8> edge;

    int diffKind() const { return thick ? kLmfThick : les ? kLmfEddy : kLmfMolecular; }
    LmfE<R> eddyView() const
    {
        return {.mut = les ? I.nut.data() : nullptr,
                .FE = thick ? fe.data() : nullptr,
                .Om = thick ? omT.data() : nullptr};
    }

    // mu_t of the state whose gradients boundaryAndGradients just took.
    void eddy(const LmfWords<R>& b)
    {
        kLmfNut<R><<<grid1(std::size_t(I.m.nC)), kT>>>(I.m.nC, sigma, cd.data(), I.G.data(), b.rho.data(), I.nut.data());
    }

    // F E, E/F, Omega of state b (gpu/tfles update): T and u, the gradients for the
    // efficiency's curl, the raw sensor, a dilation pass per walked axis, u', combine.
    void tflesUpdate(const LmfWords<R>& b)
    {
        if (tf.uniform && !tf.charlette) return; // constant: formed at make
        const std::size_t nC = std::size_t(I.m.nC);
        const unsigned gC = grid1(nC);
        cellPass(b);
        if (tf.charlette) boundaryAndGradients();
        kLmfTflesRaw<R><<<gC, kT>>>(I.m.nC, tf, b.rho.data(), b.ry[0].data(), T.data(), I.G.data(), tfA.data(),
                                     curl.data());
        const R* W = nullptr;
        if (!tf.uniform) {
            R* buf[2] = {tfA.data(), tfB.data()};
            for (int p = 0; p < nAx; ++p)
                kLmfTflesDilate<R><<<gC, kT>>>(I.m.nC, tf.rad, walk.data() + std::size_t(2 * p) * nC, T.data(),
                                               buf[p & 1], buf[(p + 1) & 1]);
            W = buf[nAx & 1];
        }
        if (tf.charlette) kLmfTflesColin<R><<<gC, kT>>>(I.m, coef.data(), curl.data(), uh.data());
        kLmfTflesCombine<R><<<gC, kT>>>(I.m.nC, tf, W, uh.data(), hc.data(), edge.data(), fe.data(), eof.data(),
                                        omT.data());
    }

    // The eddy fields of the current state (dt, stats).
    void refreshEddy()
    {
        if (thick) tflesUpdate(st);
        if (!les) return;
        cellPass(st);
        boundaryAndGradients();
        eddy(st);
    }

    LmfV<R> voters() const
    {
        LmfV<R> v{};
        const int n = model.nY;
        for (int s = 0; s < n; ++s) {
            v.v[s] = Y[s].data();
            v.g[s] = GY[s].data();
        }
        v.v[n] = T.data();
        v.g[n] = GT.data();
        v.n = n + 1;
        return v;
    }

    bool twoR() const { return model.ox == 1; }
    bool phiLaw() const { return model.ox == 1 && model.law.n > 0; }

    void cellPass(const LmfWords<R>& b)
    {
        const auto k = thick     ? kLmfCell<R, false, false, true>
                       : !twoR() ? kLmfCell<R, false, false>
                       : phiLaw() ? kLmfCell<R, true, true>
                                  : kLmfCell<R, true, false>;
        k<<<grid1(std::size_t(I.m.nC)), kT>>>(I.m, K, law, b.rho.data(), b.ry[0].data(),
                                              model.nY > 1 ? b.ry[1].data() : nullptr, b.ru.data(), T.data(), Y[0].data(),
                                              Y[1].data(), om.data(), u.data(), thick ? eof.data() : nullptr);
    }

    // T, Y, u on the boundary, and every cell gradient, from the cell pass.
    void boundaryAndGradients()
    {
        const IMesh<R>& m = I.m;
        const unsigned gC = grid1(std::size_t(m.nC)), gB = grid1(std::size_t(m.nB));
        if (open) { // every boundary value, the open faces' rule, then the gradients
            kEvalB<R><<<gB, kT>>>(m, u.data(), I.phiB.data(), I.Ub.data());
            kEvalT<R><<<gB, kT>>>(m, T.data(), Tb.data());
            for (int s = 0; s < model.nY; ++s) kEvalT<R><<<gB, kT>>>(meshY(s), Y[s].data(), Yb[s].data());
            kLmfOpenB<R><<<gB, kT>>>(m, K, I.phiB.data(), I.Ub.data(), Tb.data(), Yb[0].data(),
                                     model.nY > 1 ? Yb[1].data() : nullptr);
            kGradT<R><<<gC, kT>>>(m, T.data(), Tb.data(), GT.data());
            for (int s = 0; s < model.nY; ++s) kGradT<R><<<gC, kT>>>(m, Y[s].data(), Yb[s].data(), GY[s].data());
            I.gradient(u.data(), I.Ub.data());
            return;
        }
        kEvalB<R><<<gB, kT>>>(m, u.data(), I.phiB.data(), I.Ub.data());
        kEvalT<R><<<gB, kT>>>(m, T.data(), Tb.data());
        kGradT<R><<<gC, kT>>>(m, T.data(), Tb.data(), GT.data());
        for (int s = 0; s < model.nY; ++s) {
            kEvalT<R><<<gB, kT>>>(meshY(s), Y[s].data(), Yb[s].data());
            kGradT<R><<<gC, kT>>>(m, Y[s].data(), Yb[s].data(), GY[s].data());
        }
        I.gradient(u.data(), I.Ub.data());
    }

    IMesh<R> meshY(int s) const
    {
        IMesh<R> o = mY;
        o.bT = bY[s].data();
        return o;
    }

    // T* and V D against (a, b), on the flux of b.
    void temperature(const LmfWords<R>& a, const LmfWords<R>& b, R cb, R cr)
    {
        const IMesh<R>& m = I.m;
        const std::size_t nI = std::size_t(m.nI);
        const unsigned gC = grid1(std::size_t(m.nC)), gI = grid1(nI), gB = grid1(std::size_t(m.nB));
        const LmfV<R> V = voters();
        const bool o = I.orthogonal;
        const int dk = diffKind();
        const auto kf = dk == kLmfThick ? (o ? kLmfFaceT<R, false, kLmfThick> : kLmfFaceT<R, true, kLmfThick>)
                        : dk == kLmfEddy ? (o ? kLmfFaceT<R, false, kLmfEddy> : kLmfFaceT<R, true, kLmfEddy>)
                                         : (o ? kLmfFaceT<R, false> : kLmfFaceT<R, true>);
        kf<<<gI, kT>>>(m, K, V, T.data(), GT.data(), I.phi.data(), Tf.data(), Df.data(), eddyView());
        (o ? kLmfBndT<R, false> : kLmfBndT<R, true>)<<<gB, kT>>>(m, K, T.data(), Tb.data(), GT.data(), I.phiB.data(),
                                                                 DfB.data(), Eb.data());
        kLmfGatherT<R><<<gC, kT>>>(m, K, R(1) - cb, cb, cr, a.rho.data(), b.rho.data(), T.data(), Tb.data(), om.data(),
                                   I.phi.data(), I.phiB.data(), Tf.data(), Df.data(), DfB.data(), Ts.data(), VD.data(),
                                   Ns.data());
    }

    // One stage out of (a, b). cb, cr: gpu/lowmach's weights.
    std::expected<void, std::string> stage(const LmfWords<R>& a, const LmfWords<R>& b, LmfWords<R>& o, const LmfTally& ta,
                                           const LmfTally& tb, LmfTally& to, R cb, R cr)
    {
        const IMesh<R>& m = I.m;
        const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
        const unsigned gC = grid1(nC), gI = grid1(nI), gB = grid1(nB);
        const int nY = model.nY;
        cellPass(b);
        if (open) kLmfOpenP<R><<<gB, kT>>>(m, K, I.phiB.data(), I.bP.data());
        boundaryAndGradients();
        if (les) eddy(b);
        temperature(a, b, cb, cr);
        (les ? kLmfFaceM<R, true> : kLmfFaceM<R>)<<<gI, kT>>>(m, K, T.data(), u.data(), I.G.data(), I.phi.data(),
                                                              I.F.data(), I.nut.data());
        kLmfBndM<R><<<gB, kT>>>(m, K, Tb.data(), u.data(), I.Ub.data(), I.G.data(), I.phiB.data(), I.Fb.data());
        kLmfGatherM<R><<<gC, kT>>>(m, cb, cr, a.ru.data(), b.ru.data(), I.F.data(), I.Fb.data(), o.ru.data());
        kFlux<R><<<gI, kT>>>(m, o.ru.data(), I.phiS.data());
        kFluxB<R><<<gB, kT>>>(m, o.ru.data(), rUb.data(), I.phiSB.data());
        if (grav) {
            kLmfBuoyF<R><<<gI, kT>>>(m, K, cr, b.rho.data(), I.phiS.data(), I.Bf.data());
            kLmfBuoyB<R><<<gB, kT>>>(m, K, cr, b.rho.data(), Tb.data(), I.phiB.data(), I.phiSB.data(),
                                     I.BfB.data(), I.dpB.data());
        }
        if (auto r = I.project(o.ru.data(), cr); !r) return r;
        const LmfV<R> V = voters();
        const bool orth = I.orthogonal;
        const int dk = diffKind();
        const auto ky = dk == kLmfThick ? (orth ? kLmfFaceY<R, false, kLmfThick> : kLmfFaceY<R, true, kLmfThick>)
                        : dk == kLmfEddy ? (orth ? kLmfFaceY<R, false, kLmfEddy> : kLmfFaceY<R, true, kLmfEddy>)
                                         : (orth ? kLmfFaceY<R, false> : kLmfFaceY<R, true>);
        ky<<<gI, kT>>>(m, K, V, T.data(), I.phi.data(), Fy.data(), eddyView());
        (orth ? kLmfBndY<R, false> : kLmfBndY<R, true>)<<<gB, kT>>>(m, K, V, bYfix.data(), Tb.data(), Yb[0].data(),
                                                                    Yb[nY > 1 ? 1 : 0].data(), I.phiB.data(), Fyb.data());
        LmfW<R> W{};
        for (int v = 0; v < 1 + nY; ++v) {
            W.ah[v] = v == 0 ? a.rho.data() : a.ry[v - 1].data();
            W.al[v] = v == 0 ? a.rlo.data() : a.ylo[v - 1].data();
            W.bh[v] = v == 0 ? b.rho.data() : b.ry[v - 1].data();
            W.bl[v] = v == 0 ? b.rlo.data() : b.ylo[v - 1].data();
            W.oh[v] = v == 0 ? o.rho.data() : o.ry[v - 1].data();
            W.ol[v] = v == 0 ? o.rlo.data() : o.ylo[v - 1].data();
        }
        (twoR() ? kLmfGatherS<R, Comp, true> : kLmfGatherS<R, Comp, false>)<<<gC, kT>>>(
            m, nY, cb, cr, K.stoich, W, om.data(), I.phi.data(), I.phiB.data(), Fy.data(), Fyb.data());
        if (!model.tally) return {};
        // Net flux in: boundary sums, FP64, fixed order.
        const R* xs[2 + kFvLmMaxY] = {I.phiB.data(), Eb.data(), Fyb.data(), Fyb.data() + (nY > 1 ? nB : 0)};
        const R* ys[2 + kFvLmMaxY] = {I.ones.data(), I.ones.data(), I.ones.data(), I.ones.data()};
        auto bs = I.dots(nB, 2 + nY, xs, ys);
        if (!bs) return std::unexpected(bs.error());
        const R* vo[1] = {om.data()};
        const R* vv[1] = {m.V};
        auto bu = I.dots(nC, 1, vo, vv);
        if (!bu) return std::unexpected(bu.error());
        const double qc = double(cb), rc = double(cr), burn = (*bu)[0];
        LmfTally flux{};
        flux[0] = -(*bs)[0];
        for (int s = 0; s < nY; ++s) flux[std::size_t(1 + s)] = -(*bs)[std::size_t(2 + s)];
        flux[1 + kFvLmMaxY] = burn;
        flux[2 + kFvLmMaxY] = -(*bs)[1] + model.qcp * burn;
        for (std::size_t k = 0; k < kLmfTally; ++k) to[k] = ta[k] + (qc * (tb[k] - ta[k]) + rc * flux[k]);
        return {};
    }

    std::expected<FvLowMachFaces, std::string> faces() override
    {
        const IMesh<R>& m = I.m;
        const std::size_t nB = std::size_t(m.nB);
        const int nY = model.nY;
        cellPass(st);
        boundaryAndGradients();
        const LmfV<R> V = voters();
        (I.orthogonal ? kLmfBndY<R, false> : kLmfBndY<R, true>)<<<grid1(nB), kT>>>(
            m, K, V, bYfix.data(), Tb.data(), Yb[0].data(), Yb[nY > 1 ? 1 : 0].data(), I.phiB.data(), Fyb.data());
        (I.orthogonal ? kLmfBndT<R, false> : kLmfBndT<R, true>)<<<grid1(nB), kT>>>(
            m, K, T.data(), Tb.data(), GT.data(), I.phiB.data(), DfB.data(), Eb.data());
        if (auto r = I.check("low-Mach faces"); !r) return std::unexpected(r.error());
        FvLowMachFaces o;
        auto ph = I.fetchR(I.phiB, nB);
        if (!ph) return std::unexpected(ph.error());
        o.phiB = std::move(*ph);
        auto fy = I.fetchR(Fyb, std::size_t(nY) * nB);
        if (!fy) return std::unexpected(fy.error());
        for (std::size_t s = 0; s < std::size_t(nY); ++s)
            o.fy[s].assign(fy->begin() + std::ptrdiff_t(s * nB), fy->begin() + std::ptrdiff_t((s + 1) * nB));
        auto ht = I.fetchR(DfB, nB);
        if (!ht) return std::unexpected(ht.error());
        o.heat = std::move(*ht);
        return o;
    }

    std::expected<std::vector<double>, std::string> flameIndex(double floor) override
    {
        if (model.nY != 2) return std::unexpected("fv low-Mach: the flame index needs two species");
        const std::size_t nC = std::size_t(I.m.nC);
        cellPass(st);
        boundaryAndGradients();
        kLmfIndex<R><<<grid1(nC), kT>>>(I.m.nC, GY[0].data(), GY[1].data(), R(floor), I.w0.data());
        if (auto r = I.check("low-Mach flame index"); !r) return std::unexpected(r.error());
        return I.fetchR(I.w0, nC);
    }

    std::expected<FvLowMachDt, std::string> stableDt(double cfl, double chemK, double posK) override
    {
        const IMesh<R>& m = I.m;
        const std::size_t nC = std::size_t(m.nC);
        const bool eddyOn = les || thick;
        if (eddyOn) refreshEddy(); // M4: mu_t, F E of the current state
        cellPass(st);
        kEvalT<R><<<grid1(std::size_t(m.nB)), kT>>>(m, T.data(), Tb.data());
        const double F = fvDiffusivityFactor(1.0, model.tr.Pr);
        const R fT = R(std::max(4.0 / 3.0, 1.0 / model.prt));
        const auto k = eddyOn    ? (!twoR()    ? kLmfRate<R, false, false, true>
                                    : phiLaw() ? kLmfRate<R, true, true, true>
                                               : kLmfRate<R, true, false, true>)
                       : !twoR() ? kLmfRate<R, false, false>
                       : phiLaw() ? kLmfRate<R, true, true>
                                  : kLmfRate<R, true, false>;
        k<<<grid1(nC), kT>>>(m, K, law, R(F), st.rho.data(), T.data(), Y[0].data(), Y[1].data(), Tb.data(), I.phi.data(),
                             I.phiB.data(), I.w0.data(), I.w1.data(), I.w2.data(), eddyView(), fT);
        auto c = I.maxOf(nC, I.w0.data());
        if (!c) return std::unexpected(c.error());
        auto d = I.maxOf(nC, I.w1.data());
        if (!d) return std::unexpected(d.error());
        auto j = I.maxOf(nC, I.w2.data());
        if (!j) return std::unexpected(j.error());
        double pos = HUGE_VAL;
        if (posK > 0) {
            kLmfPos<R><<<grid1(nC), kT>>>(m.nC, I.w0.data(), I.w1.data(), I.w0.data());
            auto p = I.maxOf(nC, I.w0.data());
            if (!p) return std::unexpected(p.error());
            pos = *p > 0 ? posK / *p : HUGE_VAL;
        }
        if (auto r = I.check("low-Mach dt"); !r) return std::unexpected(r.error());
        return FvLowMachDt{.conv = cfl / *c, .diff = 1.0 / *d, .chem = chemK / *j, .pos = pos};
    }

    std::expected<FvIncompStepLog, std::string> step(double dt) override
    {
        I.log = {};
        const R d = R(dt), cb2 = R(0.25), cb3 = R(2.0 / 3.0);
        if (thick) tflesUpdate(st); // M4: F, E, Omega frozen over the step
        if (auto r = stage(st, st, q1, tSt, tSt, tQ1, R(0.0), d); !r) return std::unexpected(r.error());
        if (auto r = stage(st, q1, q2, tSt, tQ1, tQ2, cb2, cb2 * d); !r) return std::unexpected(r.error());
        if (auto r = stage(st, q2, q1, tSt, tQ2, tQ1, cb3, cb3 * d); !r) return std::unexpected(r.error());
        std::swap(st, q1);
        tSt = tQ1;
        if (auto r = I.check("low-Mach step"); !r) return std::unexpected(r.error());
        return I.log;
    }

    std::expected<FvLowMachTotals, std::string> totals() override
    {
        const IMesh<R>& m = I.m;
        const std::size_t nC = std::size_t(m.nC);
        const int nY = model.nY;
        kLmfErr<R><<<grid1(nC), kT>>>(m, K, st.rho.data(), Ts.data(), VD.data(), I.phi.data(), I.phiB.data(), I.w0.data(),
                                      I.w1.data());
        auto de = I.maxOf(nC, I.w0.data());
        if (!de) return std::unexpected(de.error());
        auto ee = I.maxOf(nC, I.w1.data());
        if (!ee) return std::unexpected(ee.error());
        auto h = cells();
        if (!h) return std::unexpected(h.error());
        FvLowMachTotals o;
        const double* V = vols.data();
        for (std::size_t c = 0; c < nC; ++c) {
            o.mass += V[c] * h->rho[c];
            o.enthalpy += V[c] * h->rho[c] * h->T[c];
            for (std::size_t s = 0; s < std::size_t(nY); ++s) o.y[s] += V[c] * h->ry[s][c];
        }
        o.massTally = tSt[0];
        for (std::size_t s = 0; s < std::size_t(nY); ++s) o.yTally[s] = tSt[1 + s];
        o.burn = tSt[1 + kFvLmMaxY];
        o.enthalpyTally = enthalpy0 + tSt[2 + kFvLmMaxY];
        o.divErr = *de;
        o.eosErr = *ee;
        return o;
    }

    std::expected<FvLowMachBudget, std::string> budget() override
    {
        const IMesh<R>& m = I.m;
        const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
        if (thick) tflesUpdate(st);
        cellPass(st);
        boundaryAndGradients();
        if (les) eddy(st);
        temperature(st, st, R(0.0), R(1.0));
        kLmfBudget<R><<<grid1(nC), kT>>>(m, T.data(), Ns.data(), I.phi.data(), I.phiB.data(), I.w0.data(), I.w1.data());
        const R* xs[3] = {I.w0.data(), I.w1.data(), om.data()};
        const R* ys[3] = {I.ones.data(), I.ones.data(), m.V};
        auto s = I.dots(nC, 3, xs, ys);
        if (!s) return std::unexpected(s.error());
        const R* xb[1] = {Eb.data()};
        const R* yb[1] = {I.ones.data()};
        auto e = I.dots(nB, 1, xb, yb);
        if (!e) return std::unexpected(e.error());
        if (auto r = I.check("low-Mach budget"); !r) return std::unexpected(r.error());
        return FvLowMachBudget{.defect = (*s)[0] - (-(*e)[0] + model.qcp * (*s)[2]), .gross = (*s)[1]};
    }

    std::expected<std::vector<double>, std::string> fetch(const DeviceBuffer<R>& b, std::size_t n, const DeviceBuffer<R>* lo)
    {
        auto h = I.fetchR(b, n);
        if (!h) return h;
        if constexpr (Comp) {
            if (lo) {
                auto l = I.fetchR(*lo, n);
                if (!l) return l;
                for (std::size_t c = 0; c < n; ++c) (*h)[c] += (*l)[c];
            }
        }
        return h;
    }

    std::expected<void, std::string> setInflow(const std::vector<double>& bU, const std::vector<double>& ru) override
    {
        const std::size_t n = 3 * std::size_t(I.m.nB);
        if (bU.size() != n || ru.size() != n) return std::unexpected("fv low-Mach: setInflow needs [3 nB] tables");
        std::vector<R> a(bU.begin(), bU.end()), b(ru.begin(), ru.end());
        (void)hipDeviceSynchronize();
        if (n > 0 && (I.bU.upload(a.data()) != hipSuccess || rUb.upload(b.data()) != hipSuccess))
            return std::unexpected("fv low-Mach: inflow table upload failed");
        return {};
    }

    std::expected<FvLowMachEddy, std::string> eddyStats() override
    {
        refreshEddy();
        FvLowMachEddy e;
        const std::size_t nC = std::size_t(I.m.nC);
        if (les) {
            auto x = I.maxOf(nC, I.nut.data());
            if (!x) return std::unexpected(x.error());
            e.mutMax = *x;
        }
        if (thick) {
            auto x = I.maxOf(nC, fe.data());
            if (!x) return std::unexpected(x.error());
            e.feMax = *x;
            auto f = I.fetchR(fe, nC), q = I.fetchR(eof, nC), w = I.fetchR(omT, nC);
            if (!f || !q || !w) return std::unexpected("fv low-Mach: TFLES fields download");
            double so = 0, soe = 0;
            for (std::size_t c = 0; c < nC; ++c) {
                so += (*w)[c];
                soe += (*w)[c] * std::sqrt((*f)[c] * (*q)[c]);
            }
            e.eBar = so > 0 ? soe / so : NAN;
        }
        if (auto r = I.check("low-Mach eddy stats"); !r) return std::unexpected(r.error());
        return e;
    }

    std::expected<FvLowMachCells, std::string> cells() override
    {
        const IMesh<R>& m = I.m;
        const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
        cellPass(st);
        FvLowMachCells o;
        auto get = [&](std::vector<double>& h, const DeviceBuffer<R>& b, std::size_t n,
                       const DeviceBuffer<R>* lo = nullptr) -> std::expected<void, std::string> {
            auto r = fetch(b, n, lo);
            if (!r) return std::unexpected(r.error());
            h = std::move(*r);
            return {};
        };
        if (auto r = get(o.rho, st.rho, nC, &st.rlo); !r) return std::unexpected(r.error());
        if (auto r = get(o.ru, st.ru, 3 * nC); !r) return std::unexpected(r.error());
        if (auto r = get(o.p, I.p, nC); !r) return std::unexpected(r.error());
        if (auto r = get(o.T, T, nC); !r) return std::unexpected(r.error());
        if (auto r = get(o.omega, om, nC); !r) return std::unexpected(r.error());
        if (auto r = get(o.phi, I.phi, nI); !r) return std::unexpected(r.error());
        if (auto r = get(o.phiB, I.phiB, nB); !r) return std::unexpected(r.error());
        for (std::size_t s = 0; s < std::size_t(model.nY); ++s)
            if (auto r = get(o.ry[s], st.ry[s], nC, &st.ylo[s]); !r) return std::unexpected(r.error());
        if (les) {
            refreshEddy();
            if (auto r = get(o.mut, I.nut, nC); !r) return std::unexpected(r.error());
        }
        return o;
    }

    // The carried words (DeviceFvLowMach::save), one fixed record set per model:
    // f(record name, buffer) over self's (const for save, mutable for load).
    template <class Self, class F>
    static std::expected<void, RunError> eachWord(Self& self, F&& f)
    {
        auto& w = self.st;
        auto one = [&](const std::string& n, auto& b) { return f("state.fvlm." + n, b); };
        for (auto [n, b] : {std::pair{"rho", &w.rho}, {"rlo", &w.rlo}, {"ru", &w.ru}, {"p", &self.I.p},
                            {"phi", &self.I.phi}, {"phiB", &self.I.phiB}})
            if (auto e = one(n, *b); !e) return e;
        for (int s = 0; s < self.model.nY; ++s) {
            if (auto e = one(std::format("y{}", s), w.ry[s]); !e) return e;
            if (auto e = one(std::format("ylo{}", s), w.ylo[s]); !e) return e;
        }
        return {};
    }

    std::expected<void, RunError> save(Checkpoint& ck) const override
    {
        if (auto e = eachWord(*this, [&](const std::string& n, const DeviceBuffer<R>& b) {
                return internal::putWords(ck, n, b);
            });
            !e)
            return e;
        ck.putF64("state.fvlm.tally", tSt);
        ck.putF64("state.fvlm.enthalpy0", enthalpy0);
        return {};
    }

    std::expected<void, RunError> load(const Checkpoint& ck) override
    {
        if (auto e = eachWord(*this, [&](const std::string& n, DeviceBuffer<R>& b) {
                return internal::getWords(ck, n, b);
            });
            !e)
            return e;
        auto t = ck.f64("state.fvlm.tally");
        if (!t) return std::unexpected(RunError{3, "--restart: " + t.error()});
        if (t->size() != tSt.size())
            return std::unexpected(RunError{3, "--restart: state.fvlm.tally has the wrong length"});
        std::ranges::copy(*t, tSt.begin());
        auto h = ck.f64Scalar("state.fvlm.enthalpy0");
        if (!h) return std::unexpected(RunError{3, "--restart: " + h.error()});
        enthalpy0 = *h;
        return {};
    }

    std::vector<double> vols;
};

template <class R>
std::expected<void, std::string> allocWords(LmfWords<R>& w, std::size_t nC, int nY)
{
    auto one = [](DeviceBuffer<R>& b, std::size_t n) -> std::expected<void, std::string> {
        auto t = zeros<R>(n, "low-Mach state");
        if (!t) return std::unexpected(t.error());
        b = std::move(*t);
        return {};
    };
    const std::size_t nL = std::is_same_v<R, float> ? nC : 1;
    if (auto r = one(w.rho, nC); !r) return r;
    if (auto r = one(w.rlo, nL); !r) return r;
    if (auto r = one(w.ru, 3 * nC); !r) return r;
    for (int s = 0; s < nY; ++s) {
        if (auto r = one(w.ry[s], nC); !r) return r;
        if (auto r = one(w.ylo[s], nL); !r) return r;
    }
    return {};
}

template <class R>
std::expected<void, std::string> makeLowMach(LmfImpl<R>& L, const FvIncompSetup& s, const FvLowMachTables& t,
                                             const dns::la::CellGraph& g, const FvLowMachModel& model,
                                             const FvIncompControls& c, const FvLowMachIC& ic)
{
    if (s.pureNeumann) return std::unexpected("fv low-Mach: no face fixes p: a closed box needs a time-dependent P0");
    if (s.hasConvective) return std::unexpected("fv low-Mach: a convective outlet is not built");
    if (!t.bOpen.empty() && t.bOpen.size() != std::size_t(s.nB)) return std::unexpected("fv low-Mach: bOpen size");
    if (!(t.ambT > 0)) return std::unexpected("fv low-Mach: the ambient T must be > 0");
    if (model.nY < 1 || model.nY > kFvLmMaxY) return std::unexpected("fv low-Mach: 1 or 2 species");
    if (!(model.ox < 0 || (model.ox == 1 && model.nY == 2)) || (model.law.n > 0 && model.ox != 1))
        return std::unexpected("fv low-Mach: two reactants are fuel 0, oxidizer 1 (nY 2); the C2b law needs them");
    const std::size_t nC = std::size_t(s.nC), nI = std::size_t(s.nI), nB = std::size_t(s.nB);
    L.model = model;
    L.vols = s.V;
    FvIncompControls ctl = c;
    ctl.lowMach = true;
    if (auto r = makeImpl(L.I, s, g, ctl, std::vector<double>(3 * nC, 0.0), {}); !r) return r;
    auto upR = [](DeviceBuffer<R>& d, const std::vector<double>& v) -> std::expected<void, std::string> {
        auto b = uploadAs<R>(v, "low-Mach table");
        if (!b) return std::unexpected(b.error());
        d = std::move(*b);
        return {};
    };
    auto up8 = [](DeviceBuffer<u8>& d, const std::vector<u8>& v) -> std::expected<void, std::string> {
        auto b = upload(v.empty() ? std::vector<u8>{0} : v, "low-Mach table");
        if (!b) return std::unexpected(b.error());
        d = std::move(*b);
        return {};
    };
    auto zero = [](DeviceBuffer<R>& d, std::size_t n) -> std::expected<void, std::string> {
        auto b = zeros<R>(std::max(n, std::size_t(1)), "low-Mach work");
        if (!b) return std::unexpected(b.error());
        d = std::move(*b);
        return {};
    };
    if (auto r = upR(L.bT, s.bT); !r) return r;
    if (auto r = upR(L.rUb, t.rUb); !r) return r;
    for (int k = 0; k < model.nY; ++k)
        if (auto r = upR(L.bY[k], t.bY[std::size_t(k)]); !r) return r;
    if (auto r = up8(L.bTfix, s.bTfix); !r) return r;
    if (auto r = up8(L.bYfix, t.bYfix); !r) return r;
    if (auto r = up8(L.first, t.first); !r) return r;
    L.open = !t.bOpen.empty();
    if (L.open)
        if (auto r = up8(L.bOpen, t.bOpen); !r) return r;
    L.grav = model.gravity != std::array<double, 3>{};
    if (L.grav) {
        L.I.buoyant = true;
        if (auto r = zero(L.I.Bf, nI); !r) return r;
        if (auto r = zero(L.I.BfB, nB); !r) return r;
        if (auto r = zero(L.I.dpB, nB); !r) return r;
    }
    L.I.m.bT = L.bT.data();
    L.I.m.bTfix = L.bTfix.data();
    L.mY = L.I.m;
    L.mY.bTfix = L.bYfix.data();
    for (auto* w : {&L.st, &L.q1, &L.q2})
        if (auto r = allocWords(*w, nC, model.nY); !r) return r;
    for (auto [b, n] : {std::pair{&L.T, nC}, {&L.om, nC}, {&L.u, 3 * nC}, {&L.GT, 3 * nC}, {&L.Ts, nC}, {&L.VD, nC},
                        {&L.Ns, nC}, {&L.Tb, nB}, {&L.Tf, nI}, {&L.Df, nI}, {&L.DfB, nB}, {&L.Eb, nB},
                        {&L.Fy, std::size_t(model.nY) * nI}, {&L.Fyb, std::size_t(model.nY) * nB}})
        if (auto r = zero(*b, n); !r) return r;
    for (int k = 0; k < kFvLmMaxY; ++k) {
        if (auto r = zero(L.Y[k], nC); !r) return r;
        if (k >= model.nY) continue;
        if (auto r = zero(L.GY[k], 3 * nC); !r) return r;
        if (auto r = zero(L.Yb[k], nB); !r) return r;
    }
    L.K = LmfK<R>{.tr = model.tr,
                  .invPr = R(1.0 / model.tr.Pr),
                  .prGas = R(model.prGas),
                  .lnA = R(model.lnA),
                  .Ta = R(model.Ta),
                  .qcp = R(model.qcp),
                  .nY = model.nY,
                  .first = L.first.data(),
                  .stoich = R(model.stoich),
                  .open = L.open ? L.bOpen.data() : nullptr,
                  .ambT = R(t.ambT),
                  .ambRho = R(model.prGas / t.ambT),
                  .ambY0 = R(t.ambY[0]),
                  .ambY1 = R(t.ambY[1]),
                  .bfHalf = R(0.5 * model.backflow),
                  .gx = R(model.gravity[0]),
                  .gy = R(model.gravity[1]),
                  .gz = R(model.gravity[2])};
    L.law = model.law.as<R>();
    L.I.target = L.VD.data();
    // M4: LES and TFLES (dns/lmmodules' pairs; the CLI refuses by name first).
    if (!(std::isfinite(model.cs) && model.cs >= 0 && std::isfinite(model.prt) && model.prt > 0))
        return std::unexpected("fv low-Mach LES: Cs >= 0 and Sc_t = Pr_t > 0");
    if (model.cs > 0 && model.sgs == SgsModel::Sigma && s.emptyAxis >= 0)
        return std::unexpected("fv low-Mach LES: sigma: D_sigma is identically 0 on a 2D mesh");
    if (model.tfles && model.ox >= 0)
        return std::unexpected("fv low-Mach TFLES: premixed C0 only (two reactants refused, dns/lmmodules)");
    if (model.tfles && model.tfles->charlette && !(model.cs > 0 && s.emptyAxis < 0))
        return std::unexpected("fv low-Mach TFLES: the efficiency needs LES on a 3D mesh");
    if (model.tfles && !t.bOpen.empty())
        return std::unexpected("fv low-Mach TFLES with open faces: not built (dns/lmmodules)");
    if (model.tfles && t.walk.next.size() != std::size_t(2 * t.walk.nAxes) * nC)
        return std::unexpected("fv low-Mach TFLES: the sensor walk is not built (fvLowMachTfles)");
    L.K.invSct = R(1.0 / model.prt);
    L.les = model.cs > 0;
    L.sigma = model.sgs == SgsModel::Sigma ? 1 : 0;
    if (L.les) {
        std::vector<double> c2(nC);
        for (std::size_t q = 0; q < nC; ++q) c2[q] = model.cs * model.cs * s.delta2[q];
        if (auto r = upR(L.cd, c2); !r) return r;
    }
    L.thick = model.tfles.has_value();
    if (L.thick) {
        const TflesSetup& ts = *model.tfles;
        L.tf = {.rad = ts.dilate,
                .uniform = ts.uniform ? 1 : 0,
                .charlette = ts.charlette ? 1 : 0,
                .lnAs = R(ts.lnAs),
                .TaS = R(ts.TaS),
                .beta = R(ts.sensorBeta),
                .Fmax = R(ts.F),
                .deltaL = R(ts.deltaL),
                .nuU = R(ts.nuU),
                .effBeta = R(ts.effBeta)};
        L.nAx = t.walk.nAxes;
        for (auto* b : {&L.fe, &L.eof, &L.omT, &L.tfA, &L.tfB, &L.uh})
            if (auto r = zero(*b, nC); !r) return r;
        if (auto r = zero(L.curl, ts.charlette ? 3 * nC : 1); !r) return r;
        std::vector<double> hv(nC), cv(nC);
        for (std::size_t q = 0; q < nC; ++q) {
            hv[q] = std::sqrt(s.delta2[q]);
            cv[q] = kColinC2 * hv[q] * hv[q] * hv[q];
        }
        if (auto r = upR(L.hc, hv); !r) return r;
        if (auto r = upR(L.coef, cv); !r) return r;
        if (auto r = up8(L.edge, t.walk.edge); !r) return r;
        auto wk = upload(t.walk.next, "TFLES walk");
        if (!wk) return std::unexpected(wk.error());
        L.walk = std::move(*wk);
        // A uniform sensor without the efficiency is constant: formed here, once.
        if (ts.uniform && !ts.charlette)
            kLmfTflesCombine<R><<<grid1(nC), kT>>>(L.I.m.nC, L.tf, nullptr, L.uh.data(), L.hc.data(), L.edge.data(),
                                                    L.fe.data(), L.eof.data(), L.omT.data());
    }
    // The state: hi the rounding of the FP64 IC, lo its remainder, gpu/lowmach's upload.
    auto putWord = [&](DeviceBuffer<R>& hi, DeviceBuffer<R>& lo, const std::vector<double>& v) -> std::expected<void, std::string> {
        if (auto r = upR(hi, v); !r) return r;
        if constexpr (std::is_same_v<R, float>) {
            std::vector<double> l(v.size());
            for (std::size_t q = 0; q < v.size(); ++q) l[q] = double(float(v[q] - double(float(v[q]))));
            if (auto r = upR(lo, l); !r) return r;
        }
        return {};
    };
    if (auto r = putWord(L.st.rho, L.st.rlo, ic.rho); !r) return r;
    if (auto r = upR(L.st.ru, ic.ru); !r) return r;
    if (auto r = upR(L.I.p, ic.p); !r) return r;
    if (auto r = putWord(L.st.ry[0], L.st.ylo[0], ic.ry0); !r) return r;
    if (model.nY > 1)
        if (auto r = putWord(L.st.ry[1], L.st.ylo[1], ic.ry1); !r) return r;
    // The laid flux, then the tallies at the uploaded inventories.
    kFlux<R><<<grid1(nI), kT>>>(L.I.m, L.st.ru.data(), L.I.phi.data());
    kFluxB<R><<<grid1(nB), kT>>>(L.I.m, L.st.ru.data(), L.rUb.data(), L.I.phiB.data());
    if (auto r = L.I.check("low-Mach init"); !r) return r;
    auto h = L.cells();
    if (!h) return std::unexpected(h.error());
    for (std::size_t q = 0; q < nC; ++q) {
        L.tSt[0] += s.V[q] * h->rho[q];
        L.enthalpy0 += s.V[q] * h->rho[q] * h->T[q];
        for (std::size_t k = 0; k < std::size_t(model.nY); ++k) L.tSt[1 + k] += s.V[q] * h->ry[k][q];
    }
    return {};
}

} // namespace

DeviceFvLowMach::DeviceFvLowMach() = default;
DeviceFvLowMach::DeviceFvLowMach(DeviceFvLowMach&&) noexcept = default;
DeviceFvLowMach& DeviceFvLowMach::operator=(DeviceFvLowMach&&) noexcept = default;
DeviceFvLowMach::~DeviceFvLowMach() = default;

std::expected<DeviceFvLowMach, std::string> DeviceFvLowMach::make(const FvIncompSetup& s, const FvLowMachTables& t,
                                                                  const dns::la::CellGraph& g,
                                                                  const FvLowMachModel& model, FvIncompControls c,
                                                                  const FvLowMachIC& ic)
{
    DeviceFvLowMach d;
    if (c.precision == FvPrecision::Fp32) {
        auto I = std::make_unique<LmfImpl<float>>();
        if (auto r = makeLowMach(*I, s, t, g, model, c, ic); !r) return std::unexpected(r.error());
        d.p_ = std::move(I);
    } else {
        auto I = std::make_unique<LmfImpl<double>>();
        if (auto r = makeLowMach(*I, s, t, g, model, c, ic); !r) return std::unexpected(r.error());
        d.p_ = std::move(I);
    }
    return d;
}

std::expected<FvLowMachDt, std::string> DeviceFvLowMach::stableDt(double cfl, double K, double posK)
{
    return p_->stableDt(cfl, K, posK);
}
std::expected<FvLowMachFaces, std::string> DeviceFvLowMach::faces() { return p_->faces(); }
std::expected<std::vector<double>, std::string> DeviceFvLowMach::flameIndex(double floor)
{
    return p_->flameIndex(floor);
}
std::expected<FvIncompStepLog, std::string> DeviceFvLowMach::step(double dt) { return p_->step(dt); }
std::expected<FvLowMachTotals, std::string> DeviceFvLowMach::totals() { return p_->totals(); }
std::expected<FvLowMachBudget, std::string> DeviceFvLowMach::budget() { return p_->budget(); }
std::expected<FvLowMachCells, std::string> DeviceFvLowMach::cells() { return p_->cells(); }
std::expected<FvLowMachEddy, std::string> DeviceFvLowMach::eddyStats() { return p_->eddyStats(); }
std::expected<void, RunError> DeviceFvLowMach::save(Checkpoint& ck) const { return p_->save(ck); }
std::expected<void, RunError> DeviceFvLowMach::load(const Checkpoint& ck) { return p_->load(ck); }
std::expected<void, std::string> DeviceFvLowMach::setInflow(const std::vector<double>& bU, const std::vector<double>& rUb)
{
    return p_->setInflow(bU, rUb);
}

} // namespace dns::gpu
