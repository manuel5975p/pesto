#include "dns/gpu/fv_simple.hpp"

#include "dns/gpu/internal/fv_device.hpp"

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/la/amg.hpp"
#include "dns/gpu/la/blockell.hpp"
#include "dns/gpu/la/internal/dots.hpp"
#include "dns/gpu/la/krylov.hpp"
#include "dns/gpu/la/smooth.hpp"
#include "dns/la/graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <type_traits>

namespace dns::gpu {

using namespace fvd;

namespace {

// --- momentum ---------------------------------------------------------------

// MRF.correctBoundaryVelocity: U_b = Omega x (Cf - o) on included faces.
template <class R>
__global__ void kMrfBoundary(Mesh<R> m, R* Ub)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB || m.bMrf[b] != 1) return;
    const std::size_t nB = std::size_t(m.nB);
#pragma unroll
    for (int q = 0; q < 3; ++q) Ub[std::size_t(q) * nB + std::size_t(b)] = m.bUmrf[std::size_t(q) * nB + std::size_t(b)];
}

// Lim: limitedSnGrad laplacian (a separate instantiation: the corrected one
// is the P5b kernel bit for bit). Turb: nuEff per cell (nuC = nu + nut,
// interpolated linearly to the face, as fvm::laplacian(nuEff, U) and
// fvc::div(nuEff dev2(T(grad U)))); laminar keeps the scalar nu bit for bit.
template <class R, bool Lim, bool Turb>
__global__ void __launch_bounds__(256) kUFace(Mesh<R> m, Ctl<R> c, const R* U, const R* G, const R* phi,
                                              const R* nuC, R* up, R* lo, R* fx)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nI = std::size_t(m.nI), ff = std::size_t(f);
    const int o = m.own[f], n = m.nei[f];
    const R l = m.w[f], ph = phi[f];
    const R wc = c.div == kDivLinear ? l : (ph >= 0 ? R(1.0) : R(0.0));
    const R lowC = -wc * ph, upC = lowC + ph;
    const R gm = (Turb ? (l * (nuC[o] - nuC[n]) + nuC[n]) : c.nu) * m.magS[f];
    const R lapU = m.dc[f] * gm;
    up[f] = upC - lapU;
    lo[f] = lowC - lapU;
    R go[3][3], gn[3][3], Xo[3][3], Xn[3][3];
    loadG(G, m.nC, o, go);
    loadG(G, m.nC, n, gn);
    nuDev2T(go, Turb ? nuC[o] : c.nu, Xo);
    nuDev2T(gn, Turb ? nuC[n] : c.nu, Xn);
    R S[3], kk[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        S[i] = m.S[std::size_t(i) * nI + ff];
        kk[i] = m.k[std::size_t(i) * nI + ff];
    }
    R cl[3] = {0, 0, 0};
    if (c.div == kDivLUD) {
        const int cu = ph > 0 ? o : n;
        R gu[3][3], r[3];
        loadG(G, m.nC, cu, gu);
#pragma unroll
        for (int i = 0; i < 3; ++i)
            r[i] = (ph > 0 ? m.rO : m.rN)[std::size_t(i) * nI + ff];
#pragma unroll
        for (int j = 0; j < 3; ++j) cl[j] = r[0] * gu[0][j] + r[1] * gu[1][j] + r[2] * gu[2][j];
    }
    if constexpr (Lim) { // limited corrected: limiter * (k . interpolate(grad U)), one per face
        const std::size_t nC = std::size_t(m.nC);
        R kc[3], sn[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            kc[j] = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) kc[j] += kk[i] * (l * (go[i][j] - gn[i][j]) + gn[i][j]);
            sn[j] = m.dc[f] * (U[std::size_t(j) * nC + std::size_t(n)] - U[std::size_t(j) * nC + std::size_t(o)]);
        }
        const R lim = snLimiter(c.lapLim, sqrt(sn[0] * sn[0] + sn[1] * sn[1] + sn[2] * sn[2]),
                                     sqrt(kc[0] * kc[0] + kc[1] * kc[1] + kc[2] * kc[2]));
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            R fd = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) fd += S[i] * (l * (Xo[i][j] - Xn[i][j]) + Xn[i][j]);
            fx[std::size_t(j) * nI + ff] = fd + gm * (lim * kc[j]) - ph * cl[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            R fd = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) fd += S[i] * (l * (Xo[i][j] - Xn[i][j]) + Xn[i][j]);
            R corr = 0;
            if (c.lapCorr) {
                R kc = 0;
#pragma unroll
                for (int i = 0; i < 3; ++i) kc += kk[i] * (l * (go[i][j] - gn[i][j]) + gn[i][j]);
                corr = gm * kc;
            }
            fx[std::size_t(j) * nI + ff] = fd + corr - ph * cl[j];
        }
    }
}

// Boundary coefficients of the momentum matrix (convection + -laplacian),
// the explicit dev2 boundary flux, and the mixed patches' updateCoeffs.
// Turb: nuEff's boundary value nuB = nu + nut_b.
template <class R, bool Turb>
__global__ void __launch_bounds__(256) kUBnd(Mesh<R> m, Ctl<R> c, const R* U, const R* Ub, const R* G,
                                             const R* phiB, const R* nuB, R* vfU, R* ic,
                                             R* bc, R* bfx)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const int P = m.bOwn[b];
    const u8 code = m.bU[b];
    const R ph = phiB[b], dc = m.bDc[b], gm = (Turb ? nuB[b] : c.nu) * m.bMagS[b];
    R n[3], ub[3], up[3];
    bndN(m, b, n);
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        ub[q] = Ub[std::size_t(q) * nB + bb];
        up[q] = U[std::size_t(q) * nC + std::size_t(P)];
    }
    if (code == 2) vfU[b] = ph < 0 ? R(1.0) : R(0.0); // inletOutlet: neg(phi)
    if (code == 4) {                             // freestreamVelocity
        const R um = sqrt(ub[0] * ub[0] + ub[1] * ub[1] + ub[2] * ub[2]);
        vfU[b] = R(0.5) - R(0.5) * ((ub[0] * n[0] + ub[1] * n[1] + ub[2] * n[2]) / um);
    }
    R sn[3];
    uSnGrad(m, b, U, Ub, vfU, sn);
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        R vic, vbc, gic, gbc;
        const R ref = m.bRefU[std::size_t(q) * nB + bb];
        switch (code) {
        case 0: vic = 0; vbc = ub[q]; gic = -dc; gbc = dc * ub[q]; break;
        case 1: vic = 1; vbc = 0; gic = 0; gbc = 0; break;
        case 2:
        case 4: {
            const R vf = vfU[b];
            vic = R(1.0) * (R(1.0) - vf);
            vbc = (R(1.0) - vf) * (R(0.0) / dc) + vf * ref;
            gic = -R(1.0) * vf * dc;
            gbc = (R(1.0) - vf) * R(0.0) + vf * (dc * ref);
            break;
        }
        default: {
            const R d = fabs(n[q]);
            vic = R(1.0) - d;
            vbc = ub[q] - vic * up[q];
            gic = -dc * d;
            gbc = sn[q] - gic * up[q];
            break;
        }
        }
        ic[std::size_t(q) * nB + bb] = ph * vic - gm * gic;
        bc[std::size_t(q) * nB + bb] = -ph * vbc + gm * gbc;
    }
    R gP[3][3], gb[3][3], Xb[3][3];
    loadG(G, m.nC, P, gP);
    if (m.bSym[b]) symTensor(n, gP);
    bndGrad(gP, n, sn, gb);
    nuDev2T(gb, Turb ? nuB[b] : c.nu, Xb);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        R s = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) s += m.bS[std::size_t(i) * nB + bb] * Xb[i][j];
        bfx[std::size_t(j) * nB + bb] = s;
    }
}

// Rows of the momentum matrix: negSumDiag, the off entries (ELL), the
// explicit sources, bounded, MRF Coriolis, then fvMatrix::relax.
// Multi: more than one MRF zone (Omega by zone index); single-zone and no-MRF
// runs take the P5b instantiation.
template <class R, bool Multi>
__global__ void __launch_bounds__(256) kURow(Mesh<R> m, Ctl<R> c, const R* U, const R* up, const R* lo,
                                             const R* fx, const R* phi, const R* phiB,
                                             const R* ic, const R* bfx, R* off, R* dU,
                                             R* sU)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    const int e0 = m.cStart[cc], e1 = m.cStart[cc + 1];
    for (int e = e0; e < e1; ++e)
        if (m.cEll[e] >= 0) off[m.cEll[e]] = 0;
    R diag = 0, divphi = 0, sumOff = 0, icMax = 0, icMin = 0, s[3] = {0, 0, 0};
    for (int e = e0; e < e1; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            if (code & 1) {
                diag -= up[f];
                off[m.cEll[e]] += lo[f];
                sumOff += fabs(lo[f]);
                divphi -= phi[f];
#pragma unroll
                for (int j = 0; j < 3; ++j) s[j] -= fx[std::size_t(j) * nI + f];
            } else {
                diag -= lo[f];
                off[m.cEll[e]] += up[f];
                sumOff += fabs(up[f]);
                divphi += phi[f];
#pragma unroll
                for (int j = 0; j < 3; ++j) s[j] += fx[std::size_t(j) * nI + f];
            }
        } else {
            const std::size_t b = std::size_t(-code - 1);
            divphi += phiB[b];
#pragma unroll
            for (int j = 0; j < 3; ++j) s[j] += bfx[std::size_t(j) * nB + b];
            const R x = ic[b], y = ic[nB + b], z = ic[2 * nB + b];
            icMax += cmptMax3(fabs(x), fabs(y), fabs(z));
            icMin += cmptMin3(x, y, z);
        }
    }
    const R V = m.V[cc];
#pragma unroll
    for (int j = 0; j < 3; ++j) s[j] = V * (s[j] / V);
    if (c.bounded) diag -= V * (divphi / V);
    R u[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) u[j] = U[std::size_t(j) * nC + std::size_t(cc)];
    if (c.mrf && m.zone[cc]) {
        const R* om = Multi ? c.om[m.zone[cc] - 1] : c.om[0];
        const R cr[3] = {om[1] * u[2] - om[2] * u[1], om[2] * u[0] - om[0] * u[2],
                              om[0] * u[1] - om[1] * u[0]};
#pragma unroll
        for (int j = 0; j < 3; ++j) s[j] -= V * cr[j];
    }
    if (c.relaxU > 0) {
        const R D0 = diag;
        R D = diag + icMax;
        D = fmax(fabs(D), sumOff);
        D /= c.relaxU;
        D -= icMin;
#pragma unroll
        for (int j = 0; j < 3; ++j) s[j] += (D - D0) * u[j];
        diag = D;
    }
    dU[cc] = diag;
#pragma unroll
    for (int j = 0; j < 3; ++j) sU[std::size_t(j) * nC + std::size_t(cc)] = s[j];
}

// Component j's system for every j in mask: diag + boundary internalCoeffs,
// source - V grad p + boundaryCoeffs (component j: Ad + j nC, rhs + j nC).
template <class R>
__global__ void kUSys3(Mesh<R> m, unsigned mask, const R* dU, const R* ic, const R* bc, const R* sU,
                       const R* gP, R* Ad, R* rhs)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
    for (int j = 0; j < 3; ++j) {
        if (!((mask >> j) & 1u)) continue;
        const std::size_t J = std::size_t(j);
        R d = dU[cc];
        R s = sU[J * nC + std::size_t(cc)] + m.V[cc] * (-gP[J * nC + std::size_t(cc)]);
        for (int e = m.cStart[cc]; e < m.cStart[cc + 1]; ++e) {
            const int code = m.cEnt[e];
            if (code >= 0) continue;
            const std::size_t b = std::size_t(-code - 1);
            d += ic[J * nB + b];
            s += bc[J * nB + b];
        }
        Ad[J * nC + std::size_t(cc)] = d;
        rhs[J * nC + std::size_t(cc)] = s;
    }
}

// U.correctBoundaryConditions(). update: the mixed patches' updateCoeffs
// (vf from the current phi / U_b) before the evaluate.
template <class R>
__global__ void kUEval(Mesh<R> m, const R* U, R* Ub, const R* phiB, R* vfU, int update)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const int P = m.bOwn[b];
    R up[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) up[q] = U[std::size_t(q) * nC + std::size_t(P)];
    switch (m.bU[b]) {
    case 0: break;
    case 1:
#pragma unroll
        for (int q = 0; q < 3; ++q) Ub[std::size_t(q) * nB + bb] = up[q];
        break;
    case 2:
    case 4: {
        R n[3];
        bndN(m, b, n);
        if (update) {
            if (m.bU[b] == 2) vfU[b] = phiB[b] < 0 ? R(1.0) : R(0.0);
            else {
                R ub[3];
#pragma unroll
                for (int q = 0; q < 3; ++q) ub[q] = Ub[std::size_t(q) * nB + bb];
                const R um = sqrt(ub[0] * ub[0] + ub[1] * ub[1] + ub[2] * ub[2]);
                vfU[b] = R(0.5) - R(0.5) * ((ub[0] * n[0] + ub[1] * n[1] + ub[2] * n[2]) / um);
            }
        }
        const R vf = vfU[b], dc = m.bDc[b];
#pragma unroll
        for (int q = 0; q < 3; ++q)
            Ub[std::size_t(q) * nB + bb] =
                (R(1.0) - vf) * (up[q] + R(0.0) / dc) + vf * m.bRefU[std::size_t(q) * nB + bb];
        break;
    }
    default: {
        R n[3], t[3];
        bndN(m, b, n);
        reflect(n, up, t);
#pragma unroll
        for (int q = 0; q < 3; ++q) Ub[std::size_t(q) * nB + bb] = R(0.5) * (up[q] + t[q]);
        break;
    }
    }
}

// --- pressure ---------------------------------------------------------------

// rAU = 1/A, H = UEqn.H() / V (2D: the empty component zeroed), H1, rAtU, HbyA.
template <class R>
__global__ void __launch_bounds__(256) kRAU(Mesh<R> m, Ctl<R> c, const R* U, const R* dU, const R* up,
                                            const R* lo, const R* sU, const R* ic,
                                            const R* bc, R* rAU, R* rAtU, R* Hc, R* HbyA)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
    const int e0 = m.cStart[cc], e1 = m.cStart[cc + 1];
    R D = dU[cc], bd[3] = {0, 0, 0}, av = 0, bsum[3] = {0, 0, 0}, lduH[3] = {0, 0, 0}, H1 = 0;
    for (int e = e0; e < e1; ++e) {
        const int code = m.cEnt[e];
        if (code < 0) {
            const std::size_t b = std::size_t(-code - 1);
            const R x = ic[b], y = ic[nB + b], z = ic[2 * nB + b];
            const R ca = (x + y + z) / R(3.0);
            D += ca;
            av += ca;
            bd[0] += x;
            bd[1] += y;
            bd[2] += z;
#pragma unroll
            for (int j = 0; j < 3; ++j) bsum[j] += bc[std::size_t(j) * nB + b];
            continue;
        }
        const std::size_t f = std::size_t(code >> 1);
        const R a = (code & 1) ? lo[f] : up[f];
        const std::size_t other = std::size_t((code & 1) ? m.own[f] : m.nei[f]);
#pragma unroll
        for (int j = 0; j < 3; ++j) lduH[j] -= a * U[std::size_t(j) * nC + other];
        H1 -= a;
    }
    const R V = m.V[cc];
    const R A = D / V;
    const R ra = R(1.0) / A;
    const R h1 = H1 / V;
    const R rat = c.consistent ? R(1.0) / (R(1.0) / ra - h1) : ra;
    rAU[cc] = ra;
    rAtU[cc] = rat;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const R bdc = -bd[j] + av;
        R h = bdc * U[std::size_t(j) * nC + std::size_t(cc)];
        h += lduH[j] + sU[std::size_t(j) * nC + std::size_t(cc)];
        h += bsum[j];
        h /= V;
        if (j == c.emptyAxis) h = 0;
        Hc[std::size_t(j) * nC + std::size_t(cc)] = h;
        HbyA[std::size_t(j) * nC + std::size_t(cc)] = ra * h;
    }
}

// phiHbyA = flux(HbyA) (constrainHbyA on the boundary), MRF.makeRelative.
template <class R, bool Two>
__global__ void kPhiH(Mesh<R> m, Ctl<R> c, const R* HbyA, R* phiH)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nI = std::size_t(m.nI), nC = std::size_t(m.nC), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    const R l = m.w[f];
    R s = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const R ho = HbyA[std::size_t(i) * nC + o], hn = HbyA[std::size_t(i) * nC + n];
        s += m.S[std::size_t(i) * nI + ff] * (l * (ho - hn) + hn);
    }
    if (c.mrf) s -= m.frame[f];
    if constexpr (Two) s -= m.frame2[f]; // the second zone's makeRelative on a zone/zone face
    phiH[f] = s;
}

// Boundary phiHbyA: constrainHbyA (non-assignable U patches take U_b), the
// constraint patches' reflected H, MRF.makeRelative. init: createPhi (U_b).
template <class R>
__global__ void kPhiHB(Mesh<R> m, Ctl<R> c, const R* HbyA, const R* H, const R* rAU, const R* Ub,
                       int init, R* phiHB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const std::size_t P = std::size_t(m.bOwn[b]);
    R h[3];
    if (init || !m.bAssign[b]) {
#pragma unroll
        for (int i = 0; i < 3; ++i) h[i] = Ub[std::size_t(i) * nB + bb];
    } else if (m.bSym[b]) {
        R n[3], v[3], t[3];
        bndN(m, b, n);
#pragma unroll
        for (int i = 0; i < 3; ++i) v[i] = H[std::size_t(i) * nC + P];
        reflect(n, v, t);
        const R ra = rAU[P];
#pragma unroll
        for (int i = 0; i < 3; ++i) h[i] = ra * (R(0.5) * (v[i] + t[i]));
    } else {
#pragma unroll
        for (int i = 0; i < 3; ++i) h[i] = HbyA[std::size_t(i) * nC + P];
    }
    R s = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) s += m.bS[std::size_t(i) * nB + bb] * h[i];
    if (!init && c.mrf) {
        if (m.bMrf[b] == 1) s = R(0.0);
        else if (m.bMrf[b] == 2) s -= m.bFrame[b];
    }
    phiHB[b] = s;
}

// adjustPhi's three boundary sums.
template <class R>
__global__ void kAdjustTerms(Mesh<R> m, const R* phiHB, R* in, R* fixedOut, R* adjOut)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const R p = phiHB[b];
    in[b] = p < 0 ? -p : R(0.0);
    fixedOut[b] = (p >= 0 && !m.bAdjust[b]) ? p : R(0.0);
    adjOut[b] = (p >= 0 && m.bAdjust[b]) ? p : R(0.0);
}

template <class R>
__global__ void kAdjustScale(Mesh<R> m, R corr, R* phiHB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    if (m.bAdjust[b] && phiHB[b] > R(0.0)) phiHB[b] *= corr;
}

// The p patch snGrad.
template <class R>
__device__ inline R pSnGrad(const Mesh<R>& m, int b, const R* p, const R* pb, const R* vfP)
{
    const R dc = m.bDc[b], pP = p[m.bOwn[b]];
    switch (m.bP[b]) {
    case 0: return dc * (pb[b] - pP);
    case 1: return R(0.0);
    default: {
        const R vf = vfP[b];
        return (R(1.0) - vf) * R(0.0) + vf * ((m.bRefP[b] - pP) * dc);
    }
    }
}

// SIMPLEC: phiHbyA += interpolate(rAtU - rAU) snGrad(p) |S|.
template <class R, bool Lim>
__global__ void kConsistentF(Mesh<R> m, Ctl<R> c, const R* rAU, const R* rAtU, const R* p,
                             const R* gP, R* phiH)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nI = std::size_t(m.nI), nC = std::size_t(m.nC), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    const R l = m.w[f];
    const R dro = rAtU[o] - rAU[o], drn = rAtU[n] - rAU[n];
    R sg = m.dc[f] * (p[n] - p[o]);
    if (c.snCorr) {
        R kc = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const R go = gP[std::size_t(i) * nC + o], gn = gP[std::size_t(i) * nC + n];
            kc += m.k[std::size_t(i) * nI + ff] * (l * (go - gn) + gn);
        }
        if constexpr (Lim) sg += snLimiter(c.snLim, fabs(sg), fabs(kc)) * kc;
        else sg += kc;
    }
    phiH[f] += ((l * (dro - drn) + drn) * sg) * m.magS[f];
}

template <class R>
__global__ void kConsistentB(Mesh<R> m, const R* rAU, const R* rAtU, const R* p, const R* pb,
                             const R* vfP, R* phiHB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const int P = m.bOwn[b];
    phiHB[b] += ((rAtU[P] - rAU[P]) * pSnGrad(m, b, p, pb, vfP)) * m.bMagS[b];
}

template <class R>
__global__ void kConsistentC(Mesh<R> m, const R* rAU, const R* rAtU, const R* gP, R* HbyA)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC);
    const R d = rAU[cc] - rAtU[cc];
#pragma unroll
    for (int j = 0; j < 3; ++j) HbyA[std::size_t(j) * nC + std::size_t(cc)] -= d * gP[std::size_t(j) * nC + std::size_t(cc)];
}

// fvm::laplacian(rAtU, p) faces: upper = dc gamma|S|, faceFluxCorrection.
// Up false (a later non-orthogonal corrector): ffc only -- upper depends on
// rAtU alone, so the first corrector's pUp stands.
template <class R, bool Lim, bool Up = true>
__global__ void kPFace(Mesh<R> m, Ctl<R> c, const R* rAtU, const R* p, const R* gP, R* pUp,
                       R* ffc)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nI = std::size_t(m.nI), nC = std::size_t(m.nC), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    const R l = m.w[f];
    const R gMagS = (l * (rAtU[o] - rAtU[n]) + rAtU[n]) * m.magS[f];
    if constexpr (Up) pUp[f] = m.dc[f] * gMagS;
    R corr = 0;
    if (c.lapCorr) {
        R kc = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const R go = gP[std::size_t(i) * nC + o], gn = gP[std::size_t(i) * nC + n];
            kc += m.k[std::size_t(i) * nI + ff] * (l * (go - gn) + gn);
        }
        if constexpr (Lim) corr = gMagS * (snLimiter(c.lapLim, fabs(m.dc[f] * (p[n] - p[o])), fabs(kc)) * kc);
        else corr = gMagS * kc;
    }
    ffc[f] = corr;
}

// p boundary coefficients (updateCoeffs of freestreamPressure first).
template <class R>
__global__ void kPBnd(Mesh<R> m, const R* rAtU, const R* Ub, R* vfP, R* pIc, R* pBc)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), bb = std::size_t(b);
    const R gMagS = rAtU[m.bOwn[b]] * m.bMagS[b];
    const R dc = m.bDc[b];
    R gic = 0, gbc = 0;
    switch (m.bP[b]) {
    case 0: gic = -dc; gbc = dc * m.bRefP[b]; break;
    case 1: break;
    default: {
        R n[3], u[3];
        bndN(m, b, n);
#pragma unroll
        for (int q = 0; q < 3; ++q) u[q] = Ub[std::size_t(q) * nB + bb];
        const R um = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        const R vf = R(0.5) + R(0.5) * ((u[0] * n[0] + u[1] * n[1] + u[2] * n[2]) / um);
        vfP[b] = vf;
        gic = -R(1.0) * vf * dc;
        gbc = (R(1.0) - vf) * R(0.0) + vf * (dc * m.bRefP[b]);
        break;
    }
    }
    pIc[b] = gMagS * gic;
    pBc[b] = -gMagS * gbc;
}

// p rows: negSumDiag, off (ELL, NEGATED), source = V div(phiHbyA) - V div(ffc),
// setReference, then the SPD system: Ad = -(diag + sum pIc), rhs = -(src + sum pBc).
// Mat false (a later non-orthogonal corrector): rhs only -- off and Ad depend
// on pUp and pIc alone, which the first corrector wrote.
template <class R, bool Mat = true>
__global__ void kPRow(Mesh<R> m, const R* pUp, const R* ffc, const R* phiH, const R* phiHB,
                      const R* pIc, const R* pBc, int refCell, R refValue, R* off, R* Ad,
                      R* rhs)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    const int e0 = m.cStart[cc], e1 = m.cStart[cc + 1];
    if constexpr (Mat)
        for (int e = e0; e < e1; ++e)
            if (m.cEll[e] >= 0) off[m.cEll[e]] = 0;
    R diag = 0, dphi = 0, dffc = 0, bi = 0, bs = 0;
    for (int e = e0; e < e1; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            diag -= pUp[f];
            if constexpr (Mat) off[m.cEll[e]] -= pUp[f];
            if (code & 1) {
                dphi -= phiH[f];
                dffc -= ffc[f];
            } else {
                dphi += phiH[f];
                dffc += ffc[f];
            }
        } else {
            const std::size_t b = std::size_t(-code - 1);
            dphi += phiHB[b];
            bi += pIc[b];
            bs += pBc[b];
        }
    }
    const R V = m.V[cc];
    R src = -(V * (dffc / V));
    src += V * (dphi / V);
    if (cc == refCell) {
        src += diag * refValue;
        diag += diag;
    }
    if constexpr (Mat) Ad[cc] = -(diag + bi);
    rhs[cc] = -(src + bs);
}

template <class R>
__global__ void kPEval(Mesh<R> m, const R* p, R* pb, const R* vfP)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const R pP = p[m.bOwn[b]];
    switch (m.bP[b]) {
    case 0: break;
    case 1: pb[b] = pP; break;
    default: {
        const R vf = vfP[b];
        pb[b] = (R(1.0) - vf) * (pP + R(0.0) / m.bDc[b]) + vf * m.bRefP[b];
        break;
    }
    }
}

// phi = phiHbyA - pEqn.flux().
template <class R>
__global__ void kPhi(Mesh<R> m, const R* phiH, const R* pUp, const R* ffc, const R* p, R* phi)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const R u = pUp[f];
    phi[f] = phiH[f] - ((u * p[m.nei[f]] - u * p[m.own[f]]) + ffc[f]);
}

template <class R>
__global__ void kPhiB(Mesh<R> m, const R* phiHB, const R* pIc, const R* pBc, const R* p,
                      R* phiB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    phiB[b] = phiHB[b] - (pIc[b] * p[m.bOwn[b]] - pBc[b]);
}

// continuityErrs: V |div phi| and V div phi per cell.
template <class R>
__global__ void kContErr(Mesh<R> m, const R* phi, const R* phiB, R* a, R* g)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    R s = 0;
    for (int e = m.cStart[cc]; e < m.cStart[cc + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            s += (code & 1) ? -phi[f] : phi[f];
        } else s += phiB[-code - 1];
    }
    const R V = m.V[cc];
    const R d = s / V;
    a[cc] = V * fabs(d);
    g[cc] = V * d;
}

template <class R>
__global__ void kRelax(int n, R alpha, const R* prev, R* x)
{
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i >= n) return;
    x[i] = prev[i] + alpha * (x[i] - prev[i]);
}

template <class R>
__global__ void kUCorr(Mesh<R> m, const R* HbyA, const R* rAtU, const R* gP, R* U)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const std::size_t i = std::size_t(j) * nC + std::size_t(cc);
        U[i] = HbyA[i] - rAtU[cc] * gP[i];
    }
}

// OpenFOAM's normFactor terms |A x - sumA xRef| + |b - sumA xRef|, xRef =
// xSum[0] / n (gAverage(x): the host's division, now on the device; xSum the
// FP64 dot, xRef rounded once to R).
template <class R>
__global__ void kNorm(int n, const R* Ax, const R* sumA, const double* xSum, const R* b, R* out)
{
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i >= n) return;
    const R xRef = R(xSum[0] / double(n));
    const R t = sumA[i] * xRef;
    out[i] = fabs(Ax[i] - t) + fabs(b[i] - t);
}

// kNorm per component j in mask (blockIdx.y; vectors at j n, xSum[j]).
template <class R>
__global__ void kNorm3(int n, unsigned mask, const R* Ax, const R* sumA, const double* xSum, const R* b,
                       R* out)
{
    const int j = int(blockIdx.y);
    if (!((mask >> j) & 1u)) return;
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i >= n) return;
    const std::size_t k = std::size_t(j) * std::size_t(n) + std::size_t(i);
    const R xRef = R(xSum[j] / double(n));
    const R t = sumA[k] * xRef;
    out[k] = fabs(Ax[k] - t) + fabs(b[k] - t);
}

template <class R>
__global__ void kFill(std::size_t n, R v, R* x)
{
    const std::size_t i = std::size_t(blockIdx.x) * kT + threadIdx.x;
    if (i < n) x[i] = v;
}

// forces: per face fP, fV (Sf & -rho nu devTwoSymm(grad U_b)), moments; 12 x nB.
// Turb: devRhoReff's boundary value with nuEff_b = nuB[b].
template <class R, bool Turb>
__global__ void kForces(Mesh<R> m, R nu, R rho, R pRef, const R* cofr, const R* U,
                        const R* Ub, const R* G, const R* pb, const R* vfU, const R* nuB,
                        R* out)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), bb = std::size_t(b);
    if (!m.bForce[b]) {
#pragma unroll
        for (int k = 0; k < 12; ++k) out[std::size_t(k) * nB + bb] = 0;
        return;
    }
    R n[3], sn[3], gP[3][3], gb[3][3], S[3], md[3];
    bndN(m, b, n);
    uSnGrad(m, b, U, Ub, vfU, sn);
    loadG(G, m.nC, m.bOwn[b], gP);
    if (m.bSym[b]) symTensor(n, gP);
    bndGrad(gP, n, sn, gb);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        S[i] = m.bS[std::size_t(i) * nB + bb];
        md[i] = m.bCf[std::size_t(i) * nB + bb] - cofr[i];
    }
    const R sph = R(kTwoThirds) * (gb[0][0] + gb[1][1] + gb[2][2]);
    R T[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) T[i][j] = (i == j) ? 2 * gb[i][i] - sph : gb[i][j] + gb[j][i];
    const R coef = -rho * (Turb ? nuB[b] : nu);
    R fP[3], fV[3];
    const R dp = pb[b] - pRef;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        fP[j] = rho * S[j] * dp;
        R s = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) s += S[i] * (coef * T[i][j]);
        fV[j] = s;
    }
    const R mP[3] = {md[1] * fP[2] - md[2] * fP[1], md[2] * fP[0] - md[0] * fP[2], md[0] * fP[1] - md[1] * fP[0]};
    const R mV[3] = {md[1] * fV[2] - md[2] * fV[1], md[2] * fV[0] - md[0] * fV[2], md[0] * fV[1] - md[1] * fV[0]};
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        out[std::size_t(j) * nB + bb] = fP[j];
        out[std::size_t(3 + j) * nB + bb] = fV[j];
        out[std::size_t(6 + j) * nB + bb] = mP[j];
        out[std::size_t(9 + j) * nB + bb] = mV[j];
    }
}

} // namespace

// --- Impl ---------------------------------------------------------------------

template <class R>
std::expected<std::vector<double>, std::string> SimpleImpl<R>::sums(std::size_t N, std::span<const R* const> a,
                                                                    std::span<const bool> abs)
{
    la::internal::DotArgs<R> d{};
    const int K = int(a.size());
    for (int i = 0; i < K; ++i) {
        d.a[i] = a[std::size_t(i)];
        d.b[i] = abs[std::size_t(i)] ? nullptr : ones.data();
    }
    la::internal::deviceDots(N, d, K, part.data(), outv.data());
    std::vector<double> h(static_cast<std::size_t>(K));
    if (auto e = hipMemcpy(h.data(), outv.data(), std::size_t(K) * sizeof(double), hipMemcpyDeviceToHost);
        e != hipSuccess)
        return std::unexpected(std::string("fv: reduction: ") + hipGetErrorString(e));
    return h;
}

namespace {

// The stop only the floor made: converged, its residual above both the case's
// tolerance and relTol x initial (a floor of 0 never fires).
bool floorLimited(const la::KrylovResult& r, const FvSolverCtl& ctl, double floor)
{
    return floor > ctl.tolerance && r.converged && r.residual > std::max(ctl.relTol * r.r0, ctl.tolerance);
}

} // namespace

template <class R>
std::expected<FvSolveLog, std::string> SimpleImpl<R>::solve(la::BlockEll<R>& A, const la::LinOp<R>& M, bool pcg,
                                                            const FvSolverCtl& ctl, double floor, R* xv,
                                                            la::MultiColourGS<R>* factor, bool offChanged,
                                                            bool sumAValid)
{
    const std::size_t N = std::size_t(m.nC);
    if (factor) factor->factorDeferred(kr->preFlag(), offChanged);
    if (!sumAValid) A.spmv(ones.data(), sumA.data());
    A.spmv(xv, Ax.data());
    // xRef = sum(x) / N on the device (the sums() pair form), then the
    // normFactor terms; their sum rides the Krylov solve's first batch.
    la::internal::DotArgs<R> d{};
    d.a[0] = xv;
    d.b[0] = ones.data();
    la::internal::deviceDots(N, d, 1, part.data(), outv.data());
    kNorm<R><<<grid(N), kT>>>(int(N), Ax.data(), sumA.data(), outv.data(), rhs.data(), tmp.data());
    la::KrylovOptions o;
    o.maxIter = ctl.maxIter;
    o.minIter = ctl.minIter;
    o.relTol = ctl.relTol;
    o.absTol = std::max(ctl.tolerance, floor);
    o.flexible = pcg;
    o.r0First = r0First;
    const la::KrylovStart<R> st{Ax.data(), tmp.data()};
    const la::LinOp<R> Aop = [&A](const R* in, R* out) { A.spmv(in, out); };
    auto r = pcg ? kr->pcg(Aop, M, rhs.data(), xv, o, st) : kr->bicgstab(Aop, M, rhs.data(), xv, o, st);
    if (!r) return std::unexpected(r.error());
    if (r->flagged) return std::unexpected(la::BlockJacobi<R>::kFactorError);
    if (r->breakdown && !r->converged)
        return std::unexpected(std::format("fv: {} breakdown after {} iterations (residual {:.3e})",
                                           pcg ? "PCG" : "BiCGStab", r->iterations, r->residual));
    return FvSolveLog{r->r0, r->residual, r->iterations, floorLimited(*r, ctl, floor)};
}

template <class R>
std::expected<void, std::string> SimpleImpl<R>::solveU(unsigned mask, FvIterLog& log)
{
    const std::size_t N = std::size_t(m.nC);
    int* fl[3] = {kr3->preFlag(0), kr3->preFlag(1), kr3->preFlag(2)};
    // kURow wrote AU's off-diagonals this iteration: the private GS copy re-gathers
    if (auto f = gsU->factorMulti(3, mask, diag3.data(), fl, true); !f) return std::unexpected("fv: U: " + f.error());
    AU->spmvMulti(3, mask, diag3.data(), ones.data(), 0, sumA3.data());
    AU->spmvMulti(3, mask, diag3.data(), U.data(), N, Ax3.data());
    la::internal::DotArgs<R> d{};
    for (int j = 0; j < 3; ++j) {
        d.a[j] = U.data() + std::size_t(j) * N;
        d.b[j] = ones.data();
    }
    la::internal::deviceDots(N, d, 3, part.data(), outv.data());
    kNorm3<R><<<dim3(grid(N), 3), kT>>>(int(N), mask, Ax3.data(), sumA3.data(), outv.data(), rhs3.data(), tmp.data());
    const FvSolverCtl& ctl = fc.solverU;
    la::KrylovOptions o;
    o.maxIter = ctl.maxIter;
    o.minIter = ctl.minIter;
    o.relTol = ctl.relTol;
    o.absTol = std::max(ctl.tolerance, floor.U);
    const la::KrylovStart<R> st{Ax3.data(), tmp.data()};
    const la::LinOp<R> Aop = [&](const R* in, R* out) { AU->spmvMulti(3, mask, diag3.data(), in, N, out); };
    const la::LinOp<R> Mop = [&](const R* r, R* z) { gsU->applyMulti(3, mask, r, z); };
    auto rs = kr3->bicgstabBatch(3, mask, Aop, Mop, rhs3.data(), U.data(), o, st);
    for (int j = 0; j < 3; ++j) { // the first failing component, in solve order
        if (!((mask >> j) & 1u)) continue;
        const std::string who = "fv: U" + std::string("xyz").substr(std::size_t(j), 1) + ": ";
        if (!rs) return std::unexpected(who + rs.error());
        const auto& r = (*rs)[std::size_t(j)];
        if (!r) return std::unexpected(who + r.error());
        if (r->flagged) return std::unexpected(who + la::BlockJacobi<R>::kFactorError);
        if (r->breakdown && !r->converged)
            return std::unexpected(who + std::format("fv: BiCGStab breakdown after {} iterations (residual {:.3e})",
                                                     r->iterations, r->residual));
        log.U[std::size_t(j)] = FvSolveLog{r->r0, r->residual, r->iterations, floorLimited(*r, ctl, floor.U)};
    }
    return {};
}

DeviceSimple::DeviceSimple() = default;
DeviceSimple::DeviceSimple(DeviceSimple&&) noexcept = default;
DeviceSimple& DeviceSimple::operator=(DeviceSimple&&) noexcept = default;
DeviceSimple::~DeviceSimple() = default;

int DeviceSimple::iteration() const { return p_->iteration(); }
std::expected<FvIterLog, std::string> DeviceSimple::iterate() { return p_->iterate(); }
std::expected<FvIterLog, std::string> DeviceSimple::coupledIterate(const FvCoupledOpts& o)
{
    return p_->coupledIterate(o);
}
std::expected<FvForceSums, std::string> DeviceSimple::forces() { return p_->forces(); }
std::expected<FvState, std::string> DeviceSimple::state() const { return p_->state(); }

namespace {

template <class R>
std::expected<void, std::string> makeImpl(SimpleImpl<R>& I, const FvSimpleSetup& s, const FoamCase& fc,
                                          const dns::la::CellGraph& g, const FvFloors& floors)
{
    I.fc = fc;
    I.floor = floors;
    const std::size_t nC = std::size_t(s.nC), nI = std::size_t(s.nI), nB = std::size_t(s.nB);
#define FV_UP(dst, src)                                                                                             \
    {                                                                                                               \
        auto t_ = upload(src, #src);                                                                                \
        if (!t_) return std::unexpected(t_.error());                                                                \
        dst = std::move(*t_);                                                                                       \
    }
#define FV_UPR(dst, src)                                                                                            \
    {                                                                                                               \
        auto t_ = uploadAs<R>(src, #src);                                                                           \
        if (!t_) return std::unexpected(t_.error());                                                                \
        dst = std::move(*t_);                                                                                       \
    }
#define FV_ZERO(dst, T, n)                                                                                          \
    {                                                                                                               \
        auto t_ = zeros<T>(n, #dst);                                                                                \
        if (!t_) return std::unexpected(t_.error());                                                                \
        dst = std::move(*t_);                                                                                       \
    }
    FV_UP(I.own, s.own) FV_UP(I.nei, s.nei) FV_UP(I.bOwn, s.bOwn) FV_UP(I.cStart, s.cellStart)
    FV_UP(I.cEnt, s.cellEnt) FV_UP(I.cEll, s.cellEll)
    FV_UPR(I.S, s.S) FV_UPR(I.rO, s.rO) FV_UPR(I.rN, s.rN) FV_UPR(I.k, s.k) FV_UPR(I.magS, s.magS) FV_UPR(I.w, s.w)
    FV_UPR(I.dc, s.dc)
    FV_UPR(I.bS, s.bS) FV_UPR(I.bN, s.bN) FV_UPR(I.bCf, s.bCf) FV_UPR(I.bMagS, s.bMagS) FV_UPR(I.bDc, s.bDc)
    FV_UPR(I.bRefU, s.bRefU) FV_UPR(I.bRefP, s.bRefP) FV_UPR(I.V, s.V) FV_UPR(I.C, s.C)
    FV_UP(I.bU, s.bU) FV_UP(I.bP, s.bP) FV_UP(I.bAssign, s.bAssign) FV_UP(I.bSym, s.bSym) FV_UP(I.bAdjust, s.bAdjust)
    FV_UP(I.bForce, s.bForce)
    if (s.mrf) {
        FV_UPR(I.frame, s.frame) FV_UP(I.bMrf, s.bMrf) FV_UPR(I.bFrame, s.bFrame) FV_UPR(I.bUmrf, s.bUmrf)
        FV_UPR(I.frame2, s.frame2)
        FV_UP(I.zone, s.zone)
    } else {
        FV_ZERO(I.frame, R, 1) FV_ZERO(I.frame2, R, 1) FV_ZERO(I.bMrf, u8, nB) FV_ZERO(I.bFrame, R, 1)
        FV_ZERO(I.bUmrf, R, 1) FV_ZERO(I.zone, u8, 1)
    }
    FV_UPR(I.U, s.U0) FV_UPR(I.Ub, s.bU0) FV_UPR(I.p, s.p0) FV_UPR(I.pb, s.bP0)
    FV_ZERO(I.pPrev, R, nC) FV_ZERO(I.pbPrev, R, nB) FV_ZERO(I.vfU, R, nB)
    FV_ZERO(I.vfP, R, nB) FV_ZERO(I.phi, R, nI) FV_ZERO(I.phiB, R, nB)
    FV_ZERO(I.G, R, 9 * nC) FV_ZERO(I.gP, R, 3 * nC) FV_ZERO(I.up, R, nI) FV_ZERO(I.lo, R, nI)
    FV_ZERO(I.fx, R, 3 * nI) FV_ZERO(I.ic, R, 3 * nB) FV_ZERO(I.bc, R, 3 * nB)
    FV_ZERO(I.bfx, R, 3 * nB) FV_ZERO(I.dU, R, nC) FV_ZERO(I.sU, R, 3 * nC)
    FV_ZERO(I.rAU, R, nC) FV_ZERO(I.rAtU, R, nC) FV_ZERO(I.HbyA, R, 3 * nC) FV_ZERO(I.Hc, R, 3 * nC)
    FV_ZERO(I.phiH, R, nI) FV_ZERO(I.phiHB, R, nB) FV_ZERO(I.pUp, R, nI) FV_ZERO(I.ffc, R, nI)
    FV_ZERO(I.pIc, R, nB) FV_ZERO(I.pBc, R, nB) FV_ZERO(I.rhs, R, nC) FV_ZERO(I.x, R, nC)
    FV_ZERO(I.sumA, R, nC) FV_ZERO(I.Ax, R, nC)
    FV_ZERO(I.diag3, R, 3 * nC) FV_ZERO(I.rhs3, R, 3 * nC) FV_ZERO(I.Ax3, R, 3 * nC)
    FV_ZERO(I.sumA3, R, 3 * nC)
    const std::size_t nMax = std::max({nC, nI, nB, std::size_t(1)});
    FV_ZERO(I.tmp, R, 12 * nMax) FV_ZERO(I.ones, R, nMax)
    FV_ZERO(I.part, double, std::size_t(la::internal::kMaxDotBatch) * std::size_t(la::internal::dotParts(nMax)))
    FV_ZERO(I.outv, double, std::size_t(la::internal::kMaxDotBatch))
    FV_ZERO(I.cofrD, R, 3)
#undef FV_UP
#undef FV_UPR
#undef FV_ZERO
    kFill<R><<<grid(nMax), kT>>>(nMax, R(1.0), I.ones.data());

    Mesh<R>& m = I.m;
    m.nC = s.nC;
    m.nI = s.nI;
    m.nB = s.nB;
    m.own = I.own.data();
    m.nei = I.nei.data();
    m.S = I.S.data();
    m.rO = I.rO.data();
    m.rN = I.rN.data();
    m.k = I.k.data();
    m.magS = I.magS.data();
    m.w = I.w.data();
    m.dc = I.dc.data();
    m.bOwn = I.bOwn.data();
    m.bS = I.bS.data();
    m.bN = I.bN.data();
    m.bCf = I.bCf.data();
    m.bMagS = I.bMagS.data();
    m.bDc = I.bDc.data();
    m.bU = I.bU.data();
    m.bP = I.bP.data();
    m.bAssign = I.bAssign.data();
    m.bSym = I.bSym.data();
    m.bAdjust = I.bAdjust.data();
    m.bMrf = I.bMrf.data();
    m.bForce = I.bForce.data();
    m.bRefU = I.bRefU.data();
    m.bRefP = I.bRefP.data();
    m.bFrame = I.bFrame.data();
    m.bUmrf = I.bUmrf.data();
    m.frame = I.frame.data();
    m.frame2 = I.frame2.data();
    m.V = I.V.data();
    m.C = I.C.data();
    m.cStart = I.cStart.data();
    m.cEnt = I.cEnt.data();
    m.cEll = I.cEll.data();
    m.zone = I.zone.data();

    Ctl<R>& c = I.c;
    c.nu = R(fc.nu);
    c.div = fc.schemes.divU == FvDiv::Linear ? kDivLinear : fc.schemes.divU == FvDiv::LinearUpwind ? kDivLUD
                                                                                                    : kDivUpwind;
    c.bounded = fc.schemes.bounded;
    c.lapCorr = fc.schemes.lapCorrected;
    c.snCorr = fc.schemes.snGradCorrected;
    c.limU = fc.schemes.gradU.kind == FvGrad::CellLimited;
    c.kU = R(fc.schemes.gradU.k);
    c.limP = fc.schemes.gradP.kind == FvGrad::CellLimited;
    c.kP = R(fc.schemes.gradP.k);
    c.mrf = s.mrf;
    c.mrfTwo = s.mrfTwo;
    I.multiZone = s.omega.size() > 1;
    for (std::size_t z = 0; z < s.omega.size(); ++z)
        for (int q = 0; q < 3; ++q) c.om[z][q] = R(s.omega[z][std::size_t(q)]);
    c.lapLim = R(fc.schemes.lapLimit);
    c.snLim = R(fc.schemes.snGradLimit);
    c.consistent = fc.simple.consistent;
    c.relaxU = R(fc.relaxU ? *fc.relaxU : 0.0);
    c.emptyAxis = s.emptyAxis;
    I.needRef = s.needRef;
    I.refCell = s.refCell;
    I.refValue = s.refValue;
    I.relaxP = fc.relaxP;
    I.relaxPGiven = fc.relaxPGiven;
    for (std::size_t b = 0; s.mrf && b < s.bMrf.size(); ++b)
        if (s.bMrf[b] == 1 && s.bU[b] != 0) I.mrfStatic = false; // kMrfBoundary would overwrite a moving Ub
    I.nSolve = 3;
    for (double v : s.V) I.sumV += v;
    if (fc.forces) {
        I.cofr = fc.forces->CofR;
        const R cr[3] = {R(I.cofr[0]), R(I.cofr[1]), R(I.cofr[2])};
        if (auto e = hipMemcpy(I.cofrD.data(), cr, 3 * sizeof(R), hipMemcpyHostToDevice); e != hipSuccess)
            return std::unexpected(std::string("fv: upload CofR: ") + hipGetErrorString(e));
    }

    // Matrices, solvers, AMG (hierarchy on the geometric weights |S| dcNO).
    const auto pat = dns::la::ellPattern(g);
    const auto shape = dns::la::denseShape(1);
    if (!shape) return std::unexpected(shape.error());
    auto au = la::BlockEll<R>::make(pat, *shape);
    if (!au) return std::unexpected(au.error());
    I.AU.emplace(std::move(*au));
    auto ap = la::BlockEll<R>::make(pat, *shape);
    if (!ap) return std::unexpected(ap.error());
    I.AP.emplace(std::move(*ap));
    const auto colours = dns::la::greedyColour(g);
    auto gs = la::MultiColourGS<R>::make(*I.AU, colours);
    if (!gs) return std::unexpected(gs.error());
    I.gsU.emplace(std::move(*gs));
    std::vector<double> wts(g.col.size(), 0.0);
    for (std::size_t f = 0; f < nI; ++f) {
        const double a = s.magS[f] * s.dc[f];
        wts[std::size_t(g.ownSlot[f])] += a;
        wts[std::size_t(g.neiSlot[f])] += a;
    }
    auto h = dns::la::amgHierarchy(g, wts, 1, dns::la::AmgSetup{});
    if (!h) return std::unexpected("fv: AMG setup: " + h.error());
    // p: scaled-V cycle (OpenFOAM GAMG's scaleCorrection) with 2 damped
    // Jacobi sweeps per level. Measured on the 708k-cell F1 prop (50 SIMPLE
    // iterations, relTol 0.01): 34.9 ms/iteration vs 45.2 for the gpu/la
    // default (K-cycle, MC-GS level 0), 54.1 for a plain V-cycle.
    la::AmgOptions ao;
    ao.smoother = la::AmgSmoother::Jacobi;
    ao.sweeps = 2;
    ao.inner = 1;
    auto amg = la::Amg<R>::make(*I.AP, *h, ao);
    if (!amg) return std::unexpected("fv: AMG: " + amg.error());
    I.amg.emplace(std::move(*amg));
    auto kr = la::Krylov<R>::make(nC, 0);
    if (!kr) return std::unexpected(kr.error());
    I.kr.emplace(std::move(*kr));
    auto kr3 = la::Krylov<R>::make(nC, 0, 3);
    if (!kr3) return std::unexpected(kr3.error());
    I.kr3.emplace(std::move(*kr3));

    // createPhi: phi = flux(U) (linear interpolation; boundary U_b . S).
    kPhiH<R, false><<<grid(nI), kT>>>(m, Ctl<R>{}, I.U.data(), I.phi.data());
    kPhiHB<R><<<grid(nB), kT>>>(m, Ctl<R>{}, I.U.data(), I.Hc.data(), I.rAU.data(), I.Ub.data(), 1, I.phiB.data());
    if (auto e = hipDeviceSynchronize(); e != hipSuccess)
        return std::unexpected(std::string("fv: init: ") + hipGetErrorString(e));
    if (auto e = hipGetLastError(); e != hipSuccess)
        return std::unexpected(std::string("fv: init: ") + hipGetErrorString(e));
    if (s.ras.model != FvTurbModel::Laminar)
        if (auto r = rasMake(I, s, fc); !r) return std::unexpected(r.error());
    return {};
}

} // namespace

std::expected<DeviceSimple, std::string> DeviceSimple::make(const FvSimpleSetup& s, const FoamCase& fc,
                                                            const dns::la::CellGraph& g, const FvNumerics& num)
{
    DeviceSimple ds;
    if (num.precision == FvPrecision::Fp32) {
        auto I = std::make_unique<SimpleImpl<float>>();
        if (auto r = makeImpl(*I, s, fc, g, num.floors); !r) return std::unexpected(r.error());
        ds.p_ = std::move(I);
    } else {
        auto I = std::make_unique<SimpleImpl<double>>();
        if (auto r = makeImpl(*I, s, fc, g, num.floors); !r) return std::unexpected(r.error());
        ds.p_ = std::move(I);
    }
    return ds;
}

template <class R>
void SimpleImpl<R>::assembleU()
{
    SimpleImpl<R>& I = *this;
    const Mesh<R>& m = I.m;
    const Ctl<R>& c = I.c;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    const unsigned gC = grid(nC), gI = grid(nI), gB = grid(nB);
    if (c.mrf) kMrfBoundary<R><<<gB, kT>>>(m, I.Ub.data());
    // G and gP of unchanged (U, Ub) and (p, pb) from the previous iteration's
    // forces() / rasCorrect and final corrector: the same kernel on the same inputs.
    if (!(I.gValid && I.mrfStatic)) kGrad<R, 3><<<gC, kT>>>(m, I.U.data(), I.Ub.data(), c.limU, c.kU, I.G.data());
    if (!I.gPValid) kGrad<R, 1><<<gC, kT>>>(m, I.p.data(), I.pb.data(), c.limP, c.kP, I.gP.data());
    I.gValid = I.gPValid = false; // U, Ub, p, pb change below
    const R* nuC = I.ras ? I.ras->nuEff.data() : nullptr;
    if (I.ras) {
        if (c.lapLim >= 0) kUFace<R, true, true><<<gI, kT>>>(m, c, I.U.data(), I.G.data(), I.phi.data(), nuC, I.up.data(), I.lo.data(), I.fx.data());
        else kUFace<R, false, true><<<gI, kT>>>(m, c, I.U.data(), I.G.data(), I.phi.data(), nuC, I.up.data(), I.lo.data(), I.fx.data());
    } else {
        if (c.lapLim >= 0) kUFace<R, true, false><<<gI, kT>>>(m, c, I.U.data(), I.G.data(), I.phi.data(), nuC, I.up.data(), I.lo.data(), I.fx.data());
        else kUFace<R, false, false><<<gI, kT>>>(m, c, I.U.data(), I.G.data(), I.phi.data(), nuC, I.up.data(), I.lo.data(), I.fx.data());
    }
    I.boundaryU();
    I.rowsU();
}

template <class R>
void SimpleImpl<R>::boundaryU()
{
    const unsigned gB = grid(std::size_t(m.nB));
    if (ras)
        kUBnd<R, true><<<gB, kT>>>(m, c, U.data(), Ub.data(), G.data(), phiB.data(), ras->nuEffB.data(), vfU.data(),
                                   ic.data(), bc.data(), bfx.data());
    else
        kUBnd<R, false><<<gB, kT>>>(m, c, U.data(), Ub.data(), G.data(), phiB.data(), nullptr, vfU.data(), ic.data(),
                                    bc.data(), bfx.data());
}

template <class R>
void SimpleImpl<R>::rowsU()
{
    (multiZone ? kURow<R, true> : kURow<R, false>)<<<grid(std::size_t(m.nC)), kT>>>(
        m, c, U.data(), up.data(), lo.data(), fx.data(), phi.data(), phiB.data(), ic.data(), bfx.data(), AU->off(),
        dU.data(), sU.data());
}

template <class R>
void SimpleImpl<R>::systemU(unsigned mask)
{
    kUSys3<R><<<grid(std::size_t(m.nC)), kT>>>(m, mask, dU.data(), ic.data(), bc.data(), sU.data(), gP.data(),
                                                diag3.data(), rhs3.data());
}

template <class R>
void SimpleImpl<R>::evalU(int update)
{
    kUEval<R><<<grid(std::size_t(m.nB)), kT>>>(m, U.data(), Ub.data(), phiB.data(), vfU.data(), update);
}

template <class R>
void SimpleImpl<R>::evalP()
{
    kPEval<R><<<grid(std::size_t(m.nB)), kT>>>(m, p.data(), pb.data(), vfP.data());
}

template <class R>
std::expected<void, std::string> SimpleImpl<R>::prepP()
{
    SimpleImpl<R>& I = *this;
    const Mesh<R>& m = I.m;
    const Ctl<R>& c = I.c;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    const unsigned gC = grid(nC), gI = grid(nI), gB = grid(nB);
    kRAU<R><<<gC, kT>>>(m, c, I.U.data(), I.dU.data(), I.up.data(), I.lo.data(), I.sU.data(), I.ic.data(),
                        I.bc.data(), I.rAU.data(), I.rAtU.data(), I.Hc.data(), I.HbyA.data());
    (c.mrfTwo ? kPhiH<R, true> : kPhiH<R, false>)<<<gI, kT>>>(m, c, I.HbyA.data(), I.phiH.data());
    kPhiHB<R><<<gB, kT>>>(m, c, I.HbyA.data(), I.Hc.data(), I.rAU.data(), I.Ub.data(), 0, I.phiHB.data());
    if (I.needRef) { // adjustPhi
        R* in = I.tmp.data();
        R* fo = in + nB;
        R* ao = fo + nB;
        kAdjustTerms<R><<<gB, kT>>>(m, I.phiHB.data(), in, fo, ao);
        const R* a3[3] = {in, fo, ao};
        const bool n3[3] = {false, false, false};
        auto sb = I.sums(nB, a3, n3);
        if (!sb) return std::unexpected(sb.error());
        // totalFlux = VSMALL + sum(mag(phi)): GeometricField sum = INTERNAL faces only (adjustPhi.C).
        const R* ai[1] = {I.phiH.data()};
        const bool yes[1] = {true};
        auto ti = I.sums(nI, ai, yes);
        if (!ti) return std::unexpected(ti.error());
        const double massIn = (*sb)[0], fixedOut = (*sb)[1], adjOut = (*sb)[2];
        const double totalFlux = 1e-300 + (*ti)[0];
        double corr = 1.0;
        if (std::abs(adjOut) > 1e-300 && std::abs(adjOut) / totalFlux > 1e-15)
            corr = (massIn - fixedOut) / adjOut;
        else if (std::abs(fixedOut - massIn) / totalFlux > 1e-8)
            return std::unexpected(std::format("fv: adjustPhi: continuity error cannot be removed by adjusting the "
                                               "outflow (total flux {:g}, specified inflow {:g}, specified outflow "
                                               "{:g}, adjustable outflow {:g})",
                                               totalFlux, massIn, fixedOut, adjOut));
        if (corr != 1.0) kAdjustScale<R><<<gB, kT>>>(m, R(corr), I.phiHB.data());
    }
    if (c.consistent) {
        (c.snLim >= 0 ? kConsistentF<R, true> : kConsistentF<R, false>)<<<gI, kT>>>(m, c, I.rAU.data(), I.rAtU.data(), I.p.data(), I.gP.data(), I.phiH.data());
        kConsistentB<R><<<gB, kT>>>(m, I.rAU.data(), I.rAtU.data(), I.p.data(), I.pb.data(), I.vfP.data(),
                                    I.phiHB.data());
        kConsistentC<R><<<gC, kT>>>(m, I.rAU.data(), I.rAtU.data(), I.gP.data(), I.HbyA.data());
    }
    return {};
}

template <class R>
void SimpleImpl<R>::assembleP(int nc)
{
    SimpleImpl<R>& I = *this;
    const Mesh<R>& m = I.m;
    const Ctl<R>& c = I.c;
    const unsigned gC = grid(std::size_t(m.nC)), gI = grid(std::size_t(m.nI)), gB = grid(std::size_t(m.nB));
    // A later corrector re-forms ffc and the source only: the matrix (pUp,
    // pIc, off, Ad), vfP / pBc (rAtU, Ub) and sumA are the first's, bit for bit.
    if (nc > 0) {
        kGrad<R, 1><<<gC, kT>>>(m, I.p.data(), I.pb.data(), c.limP, c.kP, I.gP.data());
        (c.lapLim >= 0 ? kPFace<R, true, false> : kPFace<R, false, false>)<<<gI, kT>>>(m, c, I.rAtU.data(), I.p.data(), I.gP.data(), I.pUp.data(), I.ffc.data());
        kPRow<R, false><<<gC, kT>>>(m, I.pUp.data(), I.ffc.data(), I.phiH.data(), I.phiHB.data(), I.pIc.data(), I.pBc.data(),
                                    I.needRef ? I.refCell : -1, R(I.refValue), I.AP->off(), I.AP->diag(), I.rhs.data());
    } else {
        (c.lapLim >= 0 ? kPFace<R, true> : kPFace<R, false>)<<<gI, kT>>>(m, c, I.rAtU.data(), I.p.data(), I.gP.data(), I.pUp.data(), I.ffc.data());
        kPBnd<R><<<gB, kT>>>(m, I.rAtU.data(), I.Ub.data(), I.vfP.data(), I.pIc.data(), I.pBc.data());
        kPRow<R><<<gC, kT>>>(m, I.pUp.data(), I.ffc.data(), I.phiH.data(), I.phiHB.data(), I.pIc.data(), I.pBc.data(),
                             I.needRef ? I.refCell : -1, R(I.refValue), I.AP->off(), I.AP->diag(), I.rhs.data());
    }
}

template <class R>
std::expected<void, std::string> SimpleImpl<R>::fluxes(FvIterLog& log)
{
    SimpleImpl<R>& I = *this;
    const Mesh<R>& m = I.m;
    const std::size_t nC = std::size_t(m.nC);
    kPhi<R><<<grid(std::size_t(m.nI)), kT>>>(m, I.phiH.data(), I.pUp.data(), I.ffc.data(), I.p.data(), I.phi.data());
    kPhiB<R><<<grid(std::size_t(m.nB)), kT>>>(m, I.phiHB.data(), I.pIc.data(), I.pBc.data(), I.p.data(), I.phiB.data());
    R* a = I.tmp.data();
    R* g = a + nC;
    kContErr<R><<<grid(nC), kT>>>(m, I.phi.data(), I.phiB.data(), a, g);
    const R* ag[2] = {a, g};
    const bool nn[2] = {false, false};
    auto ce = I.sums(nC, ag, nn);
    if (!ce) return std::unexpected(ce.error());
    log.contLocal = (*ce)[0] / I.sumV;
    log.contGlobal = (*ce)[1] / I.sumV;
    return {};
}

template <class R>
void SimpleImpl<R>::correctU()
{
    SimpleImpl<R>& I = *this;
    const Mesh<R>& m = I.m;
    const Ctl<R>& c = I.c;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
    const unsigned gC = grid(nC), gB = grid(nB);
    if (I.relaxPGiven) {
        kRelax<R><<<gC, kT>>>(int(nC), R(I.relaxP), I.pPrev.data(), I.p.data());
        kRelax<R><<<gB, kT>>>(int(nB), R(I.relaxP), I.pbPrev.data(), I.pb.data());
    }
    kGrad<R, 1><<<gC, kT>>>(m, I.p.data(), I.pb.data(), c.limP, c.kP, I.gP.data());
    kUCorr<R><<<gC, kT>>>(m, I.HbyA.data(), I.rAtU.data(), I.gP.data(), I.U.data());
    kUEval<R><<<gB, kT>>>(m, I.U.data(), I.Ub.data(), I.phiB.data(), I.vfU.data(), 1);
    I.gPValid = true; // p, pb final for this iteration
}

template <class R>
std::expected<FvIterLog, std::string> SimpleImpl<R>::iterate()
{
    SimpleImpl<R>& I = *this;
    const std::size_t nC = std::size_t(I.m.nC), nB = std::size_t(I.m.nB);
    FvIterLog log;
    log.iter = ++I.iter;

    // storePrevIter (fields with a relaxation factor)
    if (I.relaxPGiven) {
        (void)hipMemcpyAsync(I.pPrev.data(), I.p.data(), nC * sizeof(R), hipMemcpyDeviceToDevice, 0);
        (void)hipMemcpyAsync(I.pbPrev.data(), I.pb.data(), nB * sizeof(R), hipMemcpyDeviceToDevice, 0);
    }
    // --- UEqn.H ---
    I.assembleU();
    if (I.fc.simple.momentumPredictor) { // the components' systems share AU's off-diagonals: one batched solve
        const unsigned mask = I.uMask();
        I.systemU(mask);
        if (auto r = I.solveU(mask, log); !r) return std::unexpected(r.error());
        I.evalU(0);
    }
    // --- pEqn.H ---
    if (auto r = I.prepP(); !r) return std::unexpected(r.error());
    // The AMG is refreshed from AP at its FIRST use in this iteration (the same
    // AP values as right after assembly: AP is not written again before the
    // next iteration's kPRow), so an iteration whose p solves all start
    // converged (the SIMPLE tail) builds no hierarchy. A refresh failure is
    // reported after that solve returns, with refresh()'s text.
    bool amgFresh = false;
    std::string amgError;
    const la::LinOp<R> Mp = [&](const R* r, R* z) {
        if (!amgFresh) {
            amgFresh = true;
            if (auto f = I.amg->refresh(); !f) amgError = f.error();
        }
        I.amg->apply(r, z);
    };
    for (int nc = 0; nc <= I.fc.simple.nNonOrthCorr; ++nc) {
        I.assembleP(nc);
        if (I.pLastIters.size() <= std::size_t(nc)) I.pLastIters.resize(std::size_t(nc) + 1, -1);
        I.r0First = I.pLastIters[std::size_t(nc)] == 0; // converged at r0 last time: test r0 before the V-cycle
        auto r = I.solve(*I.AP, Mp, true, I.fc.solverP, I.floor.p, I.p.data(), nullptr, true, nc > 0);
        if (!amgError.empty()) return std::unexpected("fv: p AMG: " + amgError); // before any p result is used
        if (!r) return std::unexpected("fv: p: " + r.error());
        I.pLastIters[std::size_t(nc)] = r->iterations;
        log.p.push_back(*r);
        I.evalP();
    }
    if (auto r = I.fluxes(log); !r) return std::unexpected(r.error());
    I.correctU();
    if (I.ras)
        if (auto r = rasCorrect(I, log); !r) return std::unexpected(r.error()); // turbulence->correct(): sets gValid
    if (auto e = hipGetLastError(); e != hipSuccess)
        return std::unexpected(std::string("fv: iteration: ") + hipGetErrorString(e));
    for (const auto& u : log.U)
        if (!std::isfinite(u.initial) || !std::isfinite(u.final)) return std::unexpected("fv: non-finite U residual");
    for (const auto& q : log.p)
        if (!std::isfinite(q.initial) || !std::isfinite(q.final)) return std::unexpected("fv: non-finite p residual");
    if (!std::isfinite(log.contLocal)) return std::unexpected("fv: non-finite continuity error");
    return log;
}

template <class R>
std::expected<FvForceSums, std::string> SimpleImpl<R>::forces()
{
    SimpleImpl<R>& I = *this;
    const Mesh<R>& m = I.m;
    const std::size_t nB = std::size_t(m.nB);
    FvForceSums out;
    if (!I.fc.forces || nB == 0) return out;
    // rasCorrect's grad(U) of this iteration's final (U, Ub) is this one
    if (!I.gValid) kGrad<R, 3><<<grid(std::size_t(m.nC)), kT>>>(m, I.U.data(), I.Ub.data(), I.c.limU, I.c.kU, I.G.data());
    I.gValid = true;
    const double rho = I.fc.forces->rhoInf;
    if (I.ras)
        kForces<R, true><<<grid(nB), kT>>>(m, I.c.nu, R(rho), R(I.fc.forces->pRef / rho), I.cofrD.data(), I.U.data(),
                                           I.Ub.data(), I.G.data(), I.pb.data(), I.vfU.data(), I.ras->nuEffB.data(),
                                           I.tmp.data());
    else
        kForces<R, false><<<grid(nB), kT>>>(m, I.c.nu, R(rho), R(I.fc.forces->pRef / rho), I.cofrD.data(), I.U.data(),
                                            I.Ub.data(), I.G.data(), I.pb.data(), I.vfU.data(), nullptr, I.tmp.data());
    const R* a[12];
    bool nab[12];
    for (int k = 0; k < 12; ++k) {
        a[k] = I.tmp.data() + std::size_t(k) * nB;
        nab[k] = false;
    }
    auto s = I.sums(nB, a, nab);
    if (!s) return std::unexpected(s.error());
    for (int j = 0; j < 3; ++j) {
        out.fp[std::size_t(j)] = (*s)[std::size_t(j)];
        out.fv[std::size_t(j)] = (*s)[std::size_t(3 + j)];
        out.mp[std::size_t(j)] = (*s)[std::size_t(6 + j)];
        out.mv[std::size_t(j)] = (*s)[std::size_t(9 + j)];
    }
    return out;
}

namespace {

// A device vector of R into a host FP64 vector of h.size() (R = double: one copy).
template <class R>
hipError_t download(std::vector<double>& h, const DeviceBuffer<R>& d)
{
    if (h.empty()) return hipSuccess;
    if constexpr (std::is_same_v<R, double>)
        return hipMemcpy(h.data(), d.data(), h.size() * sizeof(double), hipMemcpyDeviceToHost);
    else {
        std::vector<R> t(h.size());
        const hipError_t e = hipMemcpy(t.data(), d.data(), t.size() * sizeof(R), hipMemcpyDeviceToHost);
        for (std::size_t i = 0; i < t.size(); ++i) h[i] = double(t[i]);
        return e;
    }
}

} // namespace

template <class R>
std::expected<FvState, std::string> SimpleImpl<R>::state() const
{
    const SimpleImpl<R>& I = *this;
    const std::size_t nC = std::size_t(I.m.nC), nI = std::size_t(I.m.nI), nB = std::size_t(I.m.nB);
    FvState s;
    s.U.resize(3 * nC);
    s.Ub.resize(3 * nB);
    s.p.resize(nC);
    s.pb.resize(nB);
    s.phi.resize(nI);
    s.phiB.resize(nB);
    for (auto [h, d] : {std::pair{&s.U, &I.U}, {&s.Ub, &I.Ub}, {&s.p, &I.p}, {&s.pb, &I.pb}, {&s.phi, &I.phi},
                        {&s.phiB, &I.phiB}})
        if (auto e = download(*h, *d); e != hipSuccess)
            return std::unexpected(std::string("fv: download: ") + hipGetErrorString(e));
    if (const RasState<R>* Rs = I.ras.get()) {
        s.nut.resize(nC);
        s.nutB.resize(nB);
        std::vector<std::pair<std::vector<double>*, const DeviceBuffer<R>*>> l{{&s.nut, &Rs->nut},
                                                                               {&s.nutB, &Rs->nutB}};
        for (int q = 0; q < Rs->nS; ++q) {
            s.s[std::size_t(q)].resize(nC);
            s.sb[std::size_t(q)].resize(nB);
            l.push_back({&s.s[std::size_t(q)], &Rs->s[std::size_t(q)]});
            l.push_back({&s.sb[std::size_t(q)], &Rs->sb[std::size_t(q)]});
        }
        for (auto [h, d] : l)
            if (auto e = download(*h, *d); e != hipSuccess)
                return std::unexpected(std::string("fv: download: ") + hipGetErrorString(e));
    }
    return s;
}

template struct SimpleImpl<double>;
template struct SimpleImpl<float>;

} // namespace dns::gpu
