// EXPERIMENTAL coupled pressure-velocity prototype on the FV path
// (plans/plan_perf_5_coupled.md P0b, docs/fv.md "Coupled prototype").
//
// Unknown x = (Ux, Uy, Uz, p), component-major (4 nC). One outer iteration:
//  1. R* from SIMPLE's own assembly at the current (U, p, phi), no solve:
//     R_U = rhs3 - A_rel U (= the UNRELAXED momentum residual, U_old = U),
//     R_c = rhs - AP p (= -div of simpleFoam's flux at (U, p)), then
//     R_c* = R_c + div(interp(rAU R_U / V) . S), a nonsingular row transform
//     (same zero set) that removes H(U) - A U from the continuity rows.
//  2. probe: simpleFoam's normalised initial residuals at this state.
//  3. K dx = R*, K = -dR*/dx with phi, rAU, rAtU, the value fractions and the
//     U-side deferred corrections frozen (Picard), composed, never assembled:
//     K_UU = AU off + (D0 + tau Dm + ic) + V Omega x + the slip/symmetry bc
//     coupling, K_Up = V grad, K_cU = div(interp(.) . S) (HbyA's boundary
//     rule), K_pp = the linearised Rhie-Chow terms (+ setReference).
//     FGMRES(30), right preconditioner = one linearised SIMPLEC step
//     (MC-GS on the case's relaxed U system, one AMG V-cycle on AP,
//     u = u* - rAtU grad dp).
//  4. x += dx, then phi = F(U, p) with this iteration's matrices (kRAU ...
//     kPhi), so the convection lag is simpleFoam's.
// The zero set is simpleFoam's fixed point (Rhie-Chow coefficient = the
// case's alpha_U-relaxed rAU); pseudo-time and the Jacobian approximations
// only change the path. Deterministic: one thread per row or face, CSR face
// order, no atomics, FP64 reductions only through the batched dots.
#include "dns/gpu/fv_simple.hpp"
#include "dns/gpu/internal/fv_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>

namespace dns::gpu {

using namespace fvd;

namespace {

// kURow's diagonal before fvMatrix::relax (D0, in kURow's operand order), its
// dominance-fixed max(|D0 + sum cmptMax|ic||, sum|off|) (Dm), and the
// Jacobian's U diagonals D0 + tau Dm + sum_b ic_j (kUSys3's order) for j in mask.
template <class R>
__global__ void kCplDiag(Mesh<R> m, Ctl<R> c, unsigned mask, R tau, const R* up, const R* lo, const R* phi,
                         const R* phiB, const R* ic, R* D0o, R* Dmo, R* diagJ)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
    const int e0 = m.cStart[cc], e1 = m.cStart[cc + 1];
    R diag = 0, divphi = 0, sumOff = 0, icMax = 0;
    for (int e = e0; e < e1; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            if (code & 1) {
                diag -= up[f];
                sumOff += fabs(lo[f]);
                divphi -= phi[f];
            } else {
                diag -= lo[f];
                sumOff += fabs(up[f]);
                divphi += phi[f];
            }
        } else {
            const std::size_t b = std::size_t(-code - 1);
            divphi += phiB[b];
            icMax += cmptMax3(fabs(ic[b]), fabs(ic[nB + b]), fabs(ic[2 * nB + b]));
        }
    }
    const R V = m.V[cc];
    if (c.bounded) diag -= V * (divphi / V);
    const R Dm = fmax(fabs(diag + icMax), sumOff);
    D0o[cc] = diag;
    Dmo[cc] = Dm;
    for (int j = 0; j < 3; ++j) {
        if (!((mask >> j) & 1u)) continue;
        R d = diag + tau * Dm;
        for (int e = e0; e < e1; ++e) {
            const int code = m.cEnt[e];
            if (code < 0) d += ic[std::size_t(j) * nB + std::size_t(-code - 1)];
        }
        diagJ[std::size_t(j) * nC + std::size_t(cc)] = d;
    }
}

// R_U = rhs3 - Ax3 on the components in mask, 0 on the others (blockIdx.y = j).
template <class R>
__global__ void kCplResU(int n, unsigned mask, const R* rhs3, const R* Ax3, R* res)
{
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i >= n) return;
    const unsigned j = blockIdx.y;
    const std::size_t k = std::size_t(j) * std::size_t(n) + std::size_t(i);
    res[k] = ((mask >> j) & 1u) ? rhs3[k] - Ax3[k] : R(0.0);
}

// HbyA's boundary flux of a cell vector a at face b (kPhiHB's rule): not
// assignable -> S . ab (ab: the patch value; null: 0), symmetry -> S . the
// reflected average, else S . a_P; MRF-included faces carry none.
template <class R>
__device__ inline R hbyaFlux(const Mesh<R>& m, int b, const R (&aP)[3], const R* ab)
{
    if (m.bMrf[b] == 1) return R(0.0);
    const std::size_t nB = std::size_t(m.nB), bb = std::size_t(b);
    R h[3];
    if (!m.bAssign[b]) {
        if (!ab) return R(0.0);
#pragma unroll
        for (int i = 0; i < 3; ++i) h[i] = ab[std::size_t(i) * nB + bb];
    } else if (m.bSym[b]) {
        R n[3], t[3];
        bndN(m, b, n);
        reflect(n, aP, t);
#pragma unroll
        for (int i = 0; i < 3; ++i) h[i] = R(0.5) * (aP[i] + t[i]);
    } else {
#pragma unroll
        for (int i = 0; i < 3; ++i) h[i] = aP[i];
    }
    R s = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) s += m.bS[std::size_t(i) * nB + bb] * h[i];
    return s;
}

// R_c* = R_c + outflow(S . interp(q)), q = rAU R_U / V (res = [R_U | R_c]).
template <class R>
__global__ void kCplRcStar(Mesh<R> m, const R* rAU, R* res)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI);
    auto q = [&](std::size_t x, int i) { return rAU[x] * res[std::size_t(i) * nC + x] / m.V[x]; };
    R s = 0;
    for (int e = m.cStart[cc]; e < m.cStart[cc + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
            const R l = m.w[f];
            R fl = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const R qo = q(o, i), qn = q(n, i);
                fl += m.S[std::size_t(i) * nI + f] * (l * (qo - qn) + qn);
            }
            s += (code & 1) ? -fl : fl;
        } else {
            const R aP[3] = {q(std::size_t(cc), 0), q(std::size_t(cc), 1), q(std::size_t(cc), 2)};
            s += hbyaFlux(m, -code - 1, aP, static_cast<const R*>(nullptr));
        }
    }
    res[3 * nC + std::size_t(cc)] += s;
}

// The homogeneous U patch values of a correction dU (kUEval with zero
// reference values and the value fractions frozen).
template <class R>
__global__ void kCplHomU(Mesh<R> m, const R* dU, const R* vfU, R* dUb)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC), bb = std::size_t(b);
    const std::size_t P = std::size_t(m.bOwn[b]);
    R up[3], v[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) up[q] = dU[std::size_t(q) * nC + P];
    switch (m.bU[b]) {
    case 0:
#pragma unroll
        for (int q = 0; q < 3; ++q) v[q] = 0;
        break;
    case 1:
#pragma unroll
        for (int q = 0; q < 3; ++q) v[q] = up[q];
        break;
    case 2:
    case 4: {
        const R vf = vfU[b];
#pragma unroll
        for (int q = 0; q < 3; ++q) v[q] = (R(1.0) - vf) * up[q];
        break;
    }
    default: {
        R n[3], t[3];
        bndN(m, b, n);
        reflect(n, up, t);
#pragma unroll
        for (int q = 0; q < 3; ++q) v[q] = R(0.5) * (up[q] + t[q]);
        break;
    }
    }
#pragma unroll
    for (int q = 0; q < 3; ++q) dUb[std::size_t(q) * nB + bb] = v[q];
}

// The homogeneous p patch values of a correction dp (kPEval, vfP frozen).
template <class R>
__global__ void kCplHomP(Mesh<R> m, const R* dp, const R* vfP, R* dpb)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const R pP = dp[m.bOwn[b]];
    switch (m.bP[b]) {
    case 0: dpb[b] = 0; break;
    case 1: dpb[b] = pP; break;
    default: dpb[b] = (R(1.0) - vfP[b]) * pP; break;
    }
}

// K's U rows: y (= diagJ dU + Off dU, spmvMulti) += V Omega x dU + V g - the
// slip / symmetry patches' explicit bc coupling d bc / d U_P; components not
// in mask 0.
template <class R>
__global__ void kCplJU(Mesh<R> m, Ctl<R> c, unsigned mask, int multi, const R* dU, const R* g, const R* phiB,
                       R* y)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC);
    const R V = m.V[cc];
    R u[3], add[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        u[j] = dU[std::size_t(j) * nC + std::size_t(cc)];
        add[j] = V * g[std::size_t(j) * nC + std::size_t(cc)];
    }
    if (c.mrf && m.zone[cc]) {
        const R* om = multi ? c.om[m.zone[cc] - 1] : c.om[0];
        const R cr[3] = {om[1] * u[2] - om[2] * u[1], om[2] * u[0] - om[0] * u[2], om[0] * u[1] - om[1] * u[0]};
#pragma unroll
        for (int j = 0; j < 3; ++j) add[j] += V * cr[j];
    }
    for (int e = m.cStart[cc]; e < m.cStart[cc + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) continue;
        const int b = -code - 1;
        if (m.bU[b] != 3) continue;
        // kUBnd basicSymmetry: bc_q = (gm dc - ph)(|n_q| u_q - n_q (n . u))
        R n[3];
        bndN(m, b, n);
        const R coef = c.nu * m.bMagS[b] * m.bDc[b] - phiB[b];
        const R nu = n[0] * u[0] + n[1] * u[1] + n[2] * u[2];
#pragma unroll
        for (int q = 0; q < 3; ++q) add[q] -= coef * (fabs(n[q]) * u[q] - n[q] * nu);
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const std::size_t k = std::size_t(j) * nC + std::size_t(cc);
        y[k] = ((mask >> j) & 1u) ? y[k] + add[j] : R(0.0);
    }
}

// K's continuity rows: the outflow of the linearised flux of simpleFoam's
// phi = phiHbyA - pEqn.flux() at a correction (dU, dp):
//   S . interp(dU + rAU g)  [P: + the SIMPLEC term interp(rAtU - rAU) |S| sn(dp)
//   - pUp (dp_N - dp_P) - interp(rAtU) |S| k . interp(g)], HbyA's boundary rule,
//   setReference's + sum pUp dp on refCell (P). P false: dU terms only.
// Sub: out = rin - outflow (the preconditioner's r_c - K_cU u*).
template <class R, bool P, bool Sub>
__global__ void __launch_bounds__(256) kCplJc(Mesh<R> m, Ctl<R> c, const R* dU, const R* dUb, const R* g,
                                              const R* dp, const R* vfP, const R* rAU, const R* rAtU,
                                              const R* pUp, const R* pIc, int refCell, const R* rin, R* out)
{
    const int cc = int(blockIdx.x) * kT + int(threadIdx.x);
    if (cc >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI);
    auto a = [&](std::size_t x, int i) {
        const std::size_t k = std::size_t(i) * nC + x;
        if constexpr (P) return dU[k] + rAU[x] * g[k];
        else return dU[k];
    };
    R s = 0, sumUp = 0;
    for (int e = m.cStart[cc]; e < m.cStart[cc + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
            const R l = m.w[f];
            R fl = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const R ao = a(o, i), an = a(n, i);
                fl += m.S[std::size_t(i) * nI + f] * (l * (ao - an) + an);
            }
            if constexpr (P) {
                R kc = 0;
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    const R go = g[std::size_t(i) * nC + o], gn = g[std::size_t(i) * nC + n];
                    kc += m.k[std::size_t(i) * nI + f] * (l * (go - gn) + gn);
                }
                const R dpn = dp[n] - dp[o];
                if (c.consistent) {
                    const R dro = rAtU[o] - rAU[o], drn = rAtU[n] - rAU[n];
                    R sg = m.dc[f] * dpn;
                    if (c.snCorr) sg += kc;
                    fl += ((l * (dro - drn) + drn) * sg) * m.magS[f];
                }
                const R gMagS = (l * (rAtU[o] - rAtU[n]) + rAtU[n]) * m.magS[f];
                fl -= pUp[f] * dpn + (c.lapCorr ? gMagS * kc : R(0.0));
                sumUp += pUp[f];
            }
            s += (code & 1) ? -fl : fl;
        } else {
            const int b = -code - 1;
            const R aP[3] = {a(std::size_t(cc), 0), a(std::size_t(cc), 1), a(std::size_t(cc), 2)};
            R fl = hbyaFlux(m, b, aP, dUb);
            if constexpr (P) {
                const R pP = dp[cc];
                if (c.consistent) {
                    R sn = 0;
                    switch (m.bP[b]) {
                    case 0: sn = -m.bDc[b] * pP; break;
                    case 1: break;
                    default: sn = vfP[b] * ((R(0.0) - pP) * m.bDc[b]); break;
                    }
                    fl += ((rAtU[cc] - rAU[cc]) * sn) * m.bMagS[b];
                }
                fl -= pIc[b] * pP;
            }
            s += fl;
        }
    }
    if constexpr (P)
        if (cc == refCell) s += sumUp * dp[cc];
    out[cc] = Sub ? rin[cc] - s : s;
}

// The preconditioner's velocity correction z_u -= rAtU grad dp on mask, 0 off it.
template <class R>
__global__ void kCplPrecCorr(int n, unsigned mask, const R* rAtU, const R* g, R* zu)
{
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i >= n) return;
    const unsigned j = blockIdx.y;
    const std::size_t k = std::size_t(j) * std::size_t(n) + std::size_t(i);
    zu[k] = ((mask >> j) & 1u) ? zu[k] - rAtU[i] * g[k] : R(0.0);
}

// x += dx: U on mask (blockIdx.y < 3), p (blockIdx.y == 3).
template <class R>
__global__ void kCplUpdate(int n, unsigned mask, R scale, const R* dx, R* U, R* p)
{
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i >= n) return;
    const unsigned j = blockIdx.y;
    const std::size_t k = std::size_t(j) * std::size_t(n) + std::size_t(i);
    if (j < 3) {
        if ((mask >> j) & 1u) U[k] += scale * dx[k];
    } else
        p[i] += scale * dx[k];
}

// OpenFOAM's residual terms per row: |b - A x| and the normFactor terms
// |A x - sumA xRef| + |b - sumA xRef|, xRef = xSum[j] / n (component j =
// blockIdx.y, offset j n; components off mask 0).
template <class R>
__global__ void kCplProbe(int n, unsigned mask, const R* Ax, const R* sumA, const double* xSum, const R* b, R* absr,
                          R* nrm)
{
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i >= n) return;
    const unsigned j = blockIdx.y;
    const std::size_t k = std::size_t(j) * std::size_t(n) + std::size_t(i);
    if (!((mask >> j) & 1u)) {
        absr[k] = nrm[k] = 0;
        return;
    }
    const R xRef = R(xSum[j] / double(n));
    const R t = sumA[k] * xRef;
    absr[k] = fabs(b[k] - Ax[k]);
    nrm[k] = fabs(Ax[k] - t) + fabs(b[k] - t);
}

// G-JAC direction: smooth deterministic v (U on mask, p), times eps.
template <class R>
__global__ void kCplDir(int n, unsigned mask, R eps, R* v)
{
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i >= n) return;
    const unsigned j = blockIdx.y;
    const std::size_t k = std::size_t(j) * std::size_t(n) + std::size_t(i);
    const double x = double(i);
    const double s = j < 3 ? (((mask >> j) & 1u) ? sin(0.37 * x + 1.3 * double(j)) : 0.0) : 0.3 * cos(0.61 * x);
    v[k] = R(double(eps) * s);
}

// d = (a - b) + c over 4 n.
template <class R>
__global__ void kCplDiff(std::size_t N, const R* a, const R* b, const R* c, R* d)
{
    const std::size_t i = std::size_t(blockIdx.x) * kT + threadIdx.x;
    if (i < N) d[i] = (a[i] - b[i]) + c[i];
}

std::unexpected<std::string> hipFail(const char* what, hipError_t e)
{
    return std::unexpected(std::format("fv coupled: {}: {}", what, hipGetErrorString(e)));
}

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// R* at the current state into cp.res (and SIMPLE's U / p systems: diag3,
// rhs3, Ax3 = A_rel U, AP, rhs). frozen (G-JAC): phi, grad U, the face
// coefficients and the dev2 boundary flux bfx are the base point's (sBfx);
// the boundary coefficients, rows and the p side are re-formed. vfU is put
// back to sVfU only AFTER kUBnd used it: a freestreamVelocity patch's value
// fraction (from Ub) is therefore NOT frozen, nor is freestreamPressure's vfP
// (kPBnd) -- G-JAC is exact only without those patches (duct: 1e-4).
template <class R>
std::expected<void, std::string> cplResidual(SimpleImpl<R>& I, bool frozen, const R* sBfx = nullptr,
                                             const R* sVfU = nullptr)
{
    CoupledState<R>& S = *I.cp;
    const Mesh<R>& m = I.m;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
    const unsigned mask = I.uMask();
    if (!frozen) I.assembleU();
    else {
        kGrad<R, 1><<<grid(nC), kT>>>(m, I.p.data(), I.pb.data(), I.c.limP, I.c.kP, I.gP.data());
        I.boundaryU();
        (void)hipMemcpyAsync(I.bfx.data(), sBfx, 3 * nB * sizeof(R), hipMemcpyDeviceToDevice, 0);
        (void)hipMemcpyAsync(I.vfU.data(), sVfU, nB * sizeof(R), hipMemcpyDeviceToDevice, 0);
        I.rowsU();
    }
    I.systemU(mask);
    I.AU->spmvMulti(3, mask, I.diag3.data(), I.U.data(), nC, I.Ax3.data());
    kCplResU<R><<<dim3(grid(nC), 3), kT>>>(int(nC), mask, I.rhs3.data(), I.Ax3.data(), S.res.data());
    if (auto r = I.prepP(); !r) return std::unexpected(r.error());
    I.assembleP(0);
    I.AP->residual(I.rhs.data(), I.p.data(), S.res.data() + 3 * nC);
    kCplRcStar<R><<<grid(nC), kT>>>(m, I.rAU.data(), S.res.data());
    return {};
}

// The probe: simpleFoam's normalised initial residuals of the U components
// (relaxed system diag3 / rhs3, Ax3 from cplResidual) and of p (AP, rhs).
template <class R>
std::expected<std::array<double, 4>, std::string> cplProbe(SimpleImpl<R>& I)
{
    const std::size_t nC = std::size_t(I.m.nC);
    const unsigned mask = I.uMask();
    I.AU->spmvMulti(3, mask, I.diag3.data(), I.ones.data(), 0, I.sumA3.data());
    I.AP->spmv(I.ones.data(), I.sumA.data());
    I.AP->spmv(I.p.data(), I.Ax.data());
    la::internal::DotArgs<R> d{};
    for (int j = 0; j < 3; ++j) {
        d.a[j] = I.U.data() + std::size_t(j) * nC;
        d.b[j] = I.ones.data();
    }
    d.a[3] = I.p.data();
    d.b[3] = I.ones.data();
    la::internal::deviceDots(nC, d, 4, I.part.data(), I.outv.data());
    R* absr = I.tmp.data();
    R* nrm = absr + 4 * nC;
    kCplProbe<R><<<dim3(grid(nC), 3), kT>>>(int(nC), mask, I.Ax3.data(), I.sumA3.data(), I.outv.data(),
                                             I.rhs3.data(), absr, nrm);
    kCplProbe<R><<<dim3(grid(nC), 1), kT>>>(int(nC), 1u, I.Ax.data(), I.sumA.data(), I.outv.data() + 3,
                                             I.rhs.data(), absr + 3 * nC, nrm + 3 * nC);
    const R* a8[8];
    bool n8[8];
    for (int k = 0; k < 4; ++k) {
        a8[k] = absr + std::size_t(k) * nC;
        a8[4 + k] = nrm + std::size_t(k) * nC;
        n8[k] = n8[4 + k] = false;
    }
    auto s = I.sums(nC, a8, n8);
    if (!s) return std::unexpected(s.error());
    std::array<double, 4> out{};
    for (int k = 0; k < 4; ++k) out[std::size_t(k)] = (*s)[std::size_t(k)] / ((*s)[std::size_t(4 + k)] + 1e-20);
    return out;
}

} // namespace

template <class R>
std::expected<FvIterLog, std::string> SimpleImpl<R>::coupledIterate(const FvCoupledOpts& o)
{
    SimpleImpl<R>& I = *this;
    const Mesh<R>& m = I.m;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB), N4 = 4 * nC;
    const unsigned gC = grid(nC), gB = grid(nB);
    const unsigned mask = I.uMask();
    if (I.ras) return std::unexpected("fv coupled: the P0b prototype is laminar only (RAS: plan P2)");
    if (!I.cp) {
        auto S = std::make_unique<CoupledState<R>>();
#define CPL_ZERO(dst, T, n)                                                                                         \
    {                                                                                                               \
        auto t_ = zeros<T>(n, #dst);                                                                                \
        if (!t_) return std::unexpected(t_.error());                                                                \
        S->dst = std::move(*t_);                                                                                    \
    }
        CPL_ZERO(res, R, N4) CPL_ZERO(dx, R, N4) CPL_ZERO(diagJ, R, 3 * nC) CPL_ZERO(D0, R, nC) CPL_ZERO(Dm, R, nC)
        CPL_ZERO(gd, R, 3 * nC) CPL_ZERO(dUb, R, 3 * nB) CPL_ZERO(dpb, R, nB) CPL_ZERO(rc, R, nC)
        CPL_ZERO(save, R, 3 * nC + nC + 3 * nB + nB + nB + nB + 3 * nB + N4 + N4 + N4) CPL_ZERO(flags, int, 3)
#undef CPL_ZERO
        auto kr = la::Krylov<R>::make(N4, 30);
        if (!kr) return std::unexpected("fv coupled: " + kr.error());
        S->kr.emplace(std::move(*kr));
        I.cp = std::move(S);
    }
    CoupledState<R>& S = *I.cp;
    FvIterLog log;
    log.iter = ++I.iter;
    log.kIt = 0;

    // --- 1. R* and the probe ---
    auto t0 = Clock::now();
    if (auto r = cplResidual(I, false); !r) return std::unexpected(r.error());
    auto pr = cplProbe(I);
    if (!pr) return std::unexpected(pr.error());
    for (int j = 0; j < 3; ++j) log.U[std::size_t(j)].initial = ((mask >> j) & 1u) ? (*pr)[std::size_t(j)] : 0.0;
    log.p.push_back(FvSolveLog{(*pr)[3], 0.0, 0, false});
    double rmax = (*pr)[3];
    bool below = (*pr)[3] <= I.fc.solverP.tolerance;
    for (int j = 0; j < 3; ++j)
        if ((mask >> j) & 1u) {
            rmax = std::max(rmax, (*pr)[std::size_t(j)]);
            below = below && (*pr)[std::size_t(j)] <= I.fc.solverU.tolerance;
        }
    if (!std::isfinite(rmax)) return std::unexpected("fv coupled: non-finite probe residual");
    if (S.r0 < 0) S.r0 = rmax;
    // SER: tau = (1/alpha - 1) min(1, r / r0), alpha_c = 1 / (1 + tau)
    const double tau = (1.0 / o.alpha - 1.0) * std::min(1.0, rmax / S.r0);
    log.alphaC = 1.0 / (1.0 + tau);

    // --- G-JAC (first outer iteration): K v vs -(R*(x + v) - R*(x)), frozen ---
    if (o.check && S.outer == 0) {
        kCplDiag<R><<<gC, kT>>>(m, I.c, mask, R(0.0), I.up.data(), I.lo.data(), I.phi.data(), I.phiB.data(),
                                I.ic.data(), S.D0.data(), S.Dm.data(), S.diagJ.data());
        R* sv = S.save.data();
        R *sU = sv, *sp = sU + 3 * nC, *sUb = sp + nC, *spb = sUb + 3 * nB, *sVfU = spb + nB, *sVfP = sVfU + nB,
          *sBfx = sVfP + nB, *sRes = sBfx + 3 * nB, *v = sRes + N4, *Kv = v + N4;
        auto cp = [](R* d, const R* s, std::size_t n) {
            (void)hipMemcpyAsync(d, s, n * sizeof(R), hipMemcpyDeviceToDevice, 0);
        };
        cp(sU, I.U.data(), 3 * nC);
        cp(sp, I.p.data(), nC);
        cp(sUb, I.Ub.data(), 3 * nB);
        cp(spb, I.pb.data(), nB);
        cp(sVfU, I.vfU.data(), nB);
        cp(sVfP, I.vfP.data(), nB);
        cp(sBfx, I.bfx.data(), 3 * nB);
        cp(sRes, S.res.data(), N4);
        kCplDir<R><<<dim3(gC, 4), kT>>>(int(nC), mask, R(1e-2), v);
        // K v (the operator below, without the pseudo-time)
        {
            const R* xp = v + 3 * nC;
            kCplHomP<R><<<gB, kT>>>(m, xp, I.vfP.data(), S.dpb.data());
            kGrad<R, 1><<<gC, kT>>>(m, xp, S.dpb.data(), 0, R(1.0), S.gd.data());
            kCplHomU<R><<<gB, kT>>>(m, v, I.vfU.data(), S.dUb.data());
            I.AU->spmvMulti(3, mask, S.diagJ.data(), v, nC, Kv);
            kCplJU<R><<<gC, kT>>>(m, I.c, mask, int(I.multiZone), v, S.gd.data(), I.phiB.data(), Kv);
            kCplJc<R, true, false><<<gC, kT>>>(m, I.c, v, S.dUb.data(), S.gd.data(), xp, I.vfP.data(), I.rAU.data(),
                                               I.rAtU.data(), I.pUp.data(), I.pIc.data(),
                                               I.needRef ? I.refCell : -1, nullptr, Kv + 3 * nC);
        }
        kCplUpdate<R><<<dim3(gC, 4), kT>>>(int(nC), mask, R(1.0), v, I.U.data(), I.p.data());
        I.evalU(0);
        I.evalP();
        if (auto r = cplResidual(I, true, sBfx, sVfU); !r) return std::unexpected(r.error());
        // d = (R*(x + v) - R*(x)) + K v = K v - FD  (FD = -(R*(x + v) - R*(x)))
        R* d = I.tmp.data(); // 12 nMax >= 4 nC
        kCplDiff<R><<<grid(N4), kT>>>(N4, S.res.data(), sRes, Kv, d);
        la::internal::DotArgs<R> da{};
        for (int k = 0; k < 4; ++k) {
            da.a[k] = da.b[k] = d + std::size_t(k) * nC;
            da.a[4 + k] = da.b[4 + k] = Kv + std::size_t(k) * nC;
        }
        la::internal::deviceDots(nC, da, 8, I.part.data(), I.outv.data());
        double h[8];
        if (auto e = hipMemcpy(h, I.outv.data(), sizeof h, hipMemcpyDeviceToHost); e != hipSuccess)
            return hipFail("G-JAC", e);
        log.jac[0] = std::sqrt((h[0] + h[1] + h[2]) / (h[4] + h[5] + h[6]));
        log.jac[1] = std::sqrt(h[3] / h[7]);
        // restore the base state and its coefficients
        cp(I.U.data(), sU, 3 * nC);
        cp(I.p.data(), sp, nC);
        cp(I.Ub.data(), sUb, 3 * nB);
        cp(I.pb.data(), spb, nB);
        cp(I.vfU.data(), sVfU, nB);
        cp(I.vfP.data(), sVfP, nB);
        I.gValid = I.gPValid = false;
        if (auto r = cplResidual(I, false); !r) return std::unexpected(r.error());
    }
    kCplDiag<R><<<gC, kT>>>(m, I.c, mask, R(tau), I.up.data(), I.lo.data(), I.phi.data(), I.phiB.data(),
                            I.ic.data(), S.D0.data(), S.Dm.data(), S.diagJ.data());
    if (auto e = hipDeviceSynchronize(); e != hipSuccess) return hipFail("residual", e);
    log.msRes = msSince(t0);
    if (below) { // simpleFoam would solve nothing here: the probe is at tolerance
        S.outer++;
        return log;
    }

    // --- 2. preconditioner setup: MC-GS on the relaxed U system, AMG on AP ---
    t0 = Clock::now();
    int* fl[3] = {S.flags.data(), S.flags.data() + 1, S.flags.data() + 2};
    if (auto f = I.gsU->factorMulti(3, mask, I.diag3.data(), fl, true); !f)
        return std::unexpected("fv coupled: U smoother: " + f.error());
    if (auto f = I.amg->refresh(); !f) return std::unexpected("fv coupled: p AMG: " + f.error());
    if (auto e = hipDeviceSynchronize(); e != hipSuccess) return hipFail("setup", e);
    log.msSetup = msSince(t0);

    // --- 3. FGMRES on K dx = R* ---
    t0 = Clock::now();
    const int ref = I.needRef ? I.refCell : -1;
    const la::LinOp<R> Kop = [&](const R* x, R* y) {
        const R* xp = x + 3 * nC;
        kCplHomP<R><<<gB, kT>>>(m, xp, I.vfP.data(), S.dpb.data());
        kGrad<R, 1><<<gC, kT>>>(m, xp, S.dpb.data(), 0, R(1.0), S.gd.data());
        kCplHomU<R><<<gB, kT>>>(m, x, I.vfU.data(), S.dUb.data());
        I.AU->spmvMulti(3, mask, S.diagJ.data(), x, nC, y);
        kCplJU<R><<<gC, kT>>>(m, I.c, mask, int(I.multiZone), x, S.gd.data(), I.phiB.data(), y);
        kCplJc<R, true, false><<<gC, kT>>>(m, I.c, x, S.dUb.data(), S.gd.data(), xp, I.vfP.data(), I.rAU.data(),
                                           I.rAtU.data(), I.pUp.data(), I.pIc.data(), ref, nullptr, y + 3 * nC);
    };
    const la::LinOp<R> Mop = [&](const R* r, R* z) {
        if (I.c.emptyAxis >= 0)
            (void)hipMemsetAsync(z + std::size_t(I.c.emptyAxis) * nC, 0, nC * sizeof(R), 0);
        I.gsU->applyMulti(3, mask, r, z);
        kCplHomU<R><<<gB, kT>>>(m, z, I.vfU.data(), S.dUb.data());
        kCplJc<R, false, true><<<gC, kT>>>(m, I.c, z, S.dUb.data(), nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, -1, r + 3 * nC, S.rc.data());
        I.amg->apply(S.rc.data(), z + 3 * nC);
        kCplHomP<R><<<gB, kT>>>(m, z + 3 * nC, I.vfP.data(), S.dpb.data());
        kGrad<R, 1><<<gC, kT>>>(m, z + 3 * nC, S.dpb.data(), 0, R(1.0), S.gd.data());
        kCplPrecCorr<R><<<dim3(gC, 3), kT>>>(int(nC), mask, I.rAtU.data(), S.gd.data(), z);
    };
    (void)hipMemsetAsync(S.dx.data(), 0, N4 * sizeof(R), 0);
    la::KrylovOptions ko;
    ko.maxIter = o.maxKrylov;
    ko.restart = 30;
    ko.relTol = o.eta;
    ko.absTol = 0;
    auto kres = S.kr->fgmres(Kop, Mop, S.res.data(), S.dx.data(), ko);
    if (!kres) return std::unexpected("fv coupled: FGMRES: " + kres.error());
    int hf[3] = {0, 0, 0};
    if (auto e = hipMemcpy(hf, S.flags.data(), sizeof hf, hipMemcpyDeviceToHost); e != hipSuccess)
        return hipFail("flags", e);
    if (hf[0] || hf[1] || hf[2]) return std::unexpected("fv coupled: U smoother: " + std::string(la::BlockJacobi<R>::kFactorError));
    log.kIt = kres->iterations;
    log.kRel = kres->relResidual;
    log.p[0].iterations = kres->iterations;
    log.p[0].final = kres->relResidual;
    if (auto e = hipDeviceSynchronize(); e != hipSuccess) return hipFail("FGMRES", e);
    log.msKrylov = msSince(t0);

    // --- 4. update, phi = F(U, p) with this iteration's matrices ---
    t0 = Clock::now();
    kCplUpdate<R><<<dim3(gC, 4), kT>>>(int(nC), mask, R(1.0), S.dx.data(), I.U.data(), I.p.data());
    I.evalU(0);
    I.evalP();
    kGrad<R, 1><<<gC, kT>>>(m, I.p.data(), I.pb.data(), I.c.limP, I.c.kP, I.gP.data());
    if (auto r = I.prepP(); !r) return std::unexpected(r.error());
    I.assembleP(0);
    if (auto r = I.fluxes(log); !r) return std::unexpected(r.error());
    I.evalU(1);
    I.gValid = false;  // U, Ub moved
    I.gPValid = true;  // gP = grad(p, pb) of the final p, pb
    if (auto e = hipGetLastError(); e != hipSuccess) return hipFail("iteration", e);
    if (auto e = hipDeviceSynchronize(); e != hipSuccess) return hipFail("update", e);
    log.msUpdate = msSince(t0);
    if (!std::isfinite(log.contLocal)) return std::unexpected("fv coupled: non-finite continuity error");
    S.outer++;
    return log;
}

template std::expected<FvIterLog, std::string> SimpleImpl<double>::coupledIterate(const FvCoupledOpts&);
template std::expected<FvIterLog, std::string> SimpleImpl<float>::coupledIterate(const FvCoupledOpts&);

} // namespace dns::gpu
