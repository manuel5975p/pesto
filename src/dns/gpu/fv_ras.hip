// RAS turbulence of the incompressible FV path (phase BL, docs/fv.md):
// OpenFOAM v2606's kOmegaSST (kOmegaSSTBase::correct) and SpalartAllmaras
// (SpalartAllmarasBase::correct) for simpleFoam, with their wall functions
// (omegaWallFunction + its blenders, nutLowRe / nutUSpalding / nutk), the
// bounded scalar transport (fvm::div [bounded] Gauss linear | upwind |
// linearUpwind | limitedLinear, fvm::laplacian(Gamma) with linear Gamma
// interpolation), fvMatrix::relax, setValues (the omega wall cells), bound()
// and correctNut, verbatim. One thread per face / cell / boundary face, CSR
// face order, no atomics: deterministic.
#include "dns/gpu/internal/fv_device.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <type_traits>

namespace dns::gpu {

using namespace fvd;

namespace {

// OpenFOAM ROOTVSMALL of the arithmetic: doubleScalar 1e-150, floatScalar 1e-18.
template <class R>
__host__ __device__ constexpr R rootVSmall()
{
    if constexpr (std::is_same_v<R, double>) return 1e-150;
    else return R(1e-18);
}

enum : int { kSDivLinear = 0, kSDivUpwind = 1, kSDivLUD = 2, kSDivLL = 3 };
// FvSCode + Slip (a scalar's slip / symmetry / symmetryPlane: zeroGradient
// values, but bound() re-evaluates it instead of max-assigning it).
enum : u8 { kSFixed = 0, kSZeroGrad = 1, kSInletOutlet = 2, kSOmegaWall = 3, kSSlip = 4 };
enum : u8 { kNutCalc = 0, kNutLowRe = 1, kNutSpalding = 2, kNutK = 3, kNutCell = 4, kNutFixed = 5 };

template <class R>
__device__ inline R sqr(R x) { return x * x; }
template <class R>
__device__ inline R pow4(R x) { return sqr(sqr(x)); }
template <class R>
__device__ inline R pos0(R x) { return x >= 0 ? R(1.0) : R(0.0); }
template <class R>
__device__ inline R signF(R x) { return x >= 0 ? R(1.0) : -R(1.0); }

// A transported scalar's device view.
template <class R>
struct SEq
{
    R* x = nullptr;   // [nC]
    R* xb = nullptr;  // [nB]
    const u8* code = nullptr;
    const R* ref = nullptr;
    R* vf = nullptr;            // mixed value fraction
    const R* gC = nullptr;      // diffusivity, cells
    const R* gB = nullptr;      // diffusivity, boundary faces
    const R* gx = nullptr;      // grad(x) [3 nC] (its gradSchemes entry)
    const R* dlt = nullptr;     // C_n - C_o [3 nI]
    int div = kSDivUpwind;
    bool bounded = false;
    R twoByk = 2;               // limitedLinear 2 / max(k, SMALL)
};

// A scalar's patch snGrad (OpenFOAM per-type snGrad()).
template <class R>
__device__ inline R sSnGrad(const Mesh<R>& m, const SEq<R>& e, int b)
{
    const R xP = e.x[m.bOwn[b]], dc = m.bDc[b];
    switch (e.code[b]) {
    case kSFixed:
    case kSOmegaWall: return dc * (e.xb[b] - xP);
    case kSInletOutlet: {
        const R vf = e.vf[b];
        return (R(1.0) - vf) * R(0.0) + vf * ((e.ref[b] - xP) * dc);
    }
    default: return R(0.0);
    }
}

// fvc::grad's boundary value of a scalar: the extrapolated cell gradient
// (constraint patches: the reflected average), + n (snGrad - n.g).
template <class R>
__device__ inline void sBndGrad(const Mesh<R>& m, const SEq<R>& e, int b, R (&gb)[3])
{
    const std::size_t nC = std::size_t(m.nC);
    const int P = m.bOwn[b];
    R n[3], g[3];
    bndN(m, b, n);
#pragma unroll
    for (int i = 0; i < 3; ++i) g[i] = e.gx[std::size_t(i) * nC + std::size_t(P)];
    if (m.bSym[b]) {
        R t[3];
        reflect(n, g, t);
#pragma unroll
        for (int i = 0; i < 3; ++i) g[i] = R(0.5) * (g[i] + t[i]);
    }
    const R sn = sSnGrad(m, e, b);
    const R ng = n[0] * g[0] + n[1] * g[1] + n[2] * g[2];
#pragma unroll
    for (int i = 0; i < 3; ++i) gb[i] = g[i] + n[i] * (sn - ng);
}

// fvc::grad(U)'s boundary tensor at b (the momentum path's bndGrad).
template <class R>
__device__ inline void uBndGradAt(const Mesh<R>& m, int b, const R* U, const R* Ub, const R* vfU,
                                  const R* G, R (&gb)[3][3])
{
    R n[3], sn[3], gP[3][3];
    bndN(m, b, n);
    uSnGrad(m, b, U, Ub, vfU, sn);
    loadG(G, m.nC, m.bOwn[b], gP);
    if (m.bSym[b]) symTensor(n, gP);
    bndGrad(gP, n, sn, gb);
}

// 2 magSqr(symm(g)) (SymmTensor magSqr: off-diagonals counted twice).
template <class R>
__device__ inline R S2of(const R (&g)[3][3])
{
    const R xx = g[0][0], yy = g[1][1], zz = g[2][2];
    const R xy = R(0.5) * (g[0][1] + g[1][0]), xz = R(0.5) * (g[0][2] + g[2][0]), yz = R(0.5) * (g[1][2] + g[2][1]);
    return 2 * (sqr(xx) + 2 * sqr(xy) + 2 * sqr(xz) + sqr(yy) + 2 * sqr(yz) + sqr(zz));
}

// g && devTwoSymm(g)
template <class R>
__device__ inline R GbyNuOf(const R (&g)[3][3])
{
    const R tr = (R(2.0) / R(3.0)) * (g[0][0] + g[1][1] + g[2][2]);
    const R sxx = 2 * g[0][0] - tr, syy = 2 * g[1][1] - tr, szz = 2 * g[2][2] - tr;
    const R sxy = g[0][1] + g[1][0], sxz = g[0][2] + g[2][0], syz = g[1][2] + g[2][1];
    return g[0][0] * sxx + g[0][1] * sxy + g[0][2] * sxz + g[1][0] * sxy + g[1][1] * syy + g[1][2] * syz +
           g[2][0] * sxz + g[2][1] * syz + g[2][2] * szz;
}

// sqrt(2) mag(skew(g))
template <class R>
__device__ inline R OmegaOf(const R (&g)[3][3])
{
    const R xy = R(0.5) * (g[0][1] - g[1][0]), xz = R(0.5) * (g[0][2] - g[2][0]), yz = R(0.5) * (g[1][2] - g[2][1]);
    // skew tensor magSqr: all nine components (zero diagonal)
    return sqrt(R(2.0)) * sqrt(2 * sqr(xy) + 2 * sqr(xz) + 2 * sqr(yz));
}

// --- SST model functions (kOmegaSSTBase, on one point) ----------------------
template <class R>
struct SstK
{
    R alphaK1, alphaK2, alphaOmega1, alphaOmega2, gamma1, gamma2, beta1, beta2, betaStar, a1, b1, c1;
    int F3;
    R nu;
};

template <class R>
__device__ inline R sstF1(const SstK<R>& c, R k, R om, R y, R CDk)
{
    const R CDkPlus = fmax(CDk, R(1.0e-10));
    const R arg1 = fmin(fmin(fmax((R(1.0) / c.betaStar) * sqrt(k) / (om * y), R(500.0) * c.nu / (sqr(y) * om)),
                                  (4 * c.alphaOmega2) * k / (CDkPlus * sqr(y))),
                             R(10.0));
    return tanh(pow4(arg1));
}

template <class R>
__device__ inline R sstF23(const SstK<R>& c, R k, R om, R y)
{
    const R arg2 = fmin(fmax((R(2.0) / c.betaStar) * sqrt(k) / (om * y), R(500.0) * c.nu / (sqr(y) * om)), R(100.0));
    R f = tanh(sqr(arg2));
    if (c.F3) {
        const R arg3 = fmin(R(150.0) * c.nu / (om * sqr(y)), R(10.0));
        f *= 1 - tanh(pow4(arg3));
    }
    return f;
}

template <class R>
__device__ inline R sstNut(const SstK<R>& c, R k, R om, R F23, R S2)
{
    return c.a1 * k / fmax(c.a1 * om, c.b1 * F23 * sqrt(S2));
}

// --- SA model functions -------------------------------------------------------
template <class R>
struct SaK
{
    R sigmaNut, kappa, Cb1, Cb2, Cw1, Cw2, Cw3, Cv1, Cs, Ct3, Ct4;
    int ft2;
    R nu;
};

template <class R>
__device__ inline R saFv1(const SaK<R>& c, R chi)
{
    const R chi3 = chi * chi * chi; // pow3
    return chi3 / (chi3 + c.Cv1 * c.Cv1 * c.Cv1);
}

// --- kernels: model cells ---------------------------------------------------------

// S2, GbyNu0, G = nut GbyNu0, divU (SST); the explicit gradU products.
template <class R>
__global__ void kSstCell1(Mesh<R> m, const R* G, const R* phi, const R* phiB, const R* nut, R* S2,
                          R* GbyNu, R* Gk, R* divU)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R g[3][3];
    loadG(G, m.nC, c, g);
    S2[c] = S2of(g);
    const R gb = GbyNuOf(g);
    GbyNu[c] = gb;
    Gk[c] = nut[c] * gb;
    R s = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            s += (code & 1) ? -phi[f] : phi[f];
        } else s += phiB[-code - 1];
    }
    divU[c] = s / m.V[c];
}

// omegaWallFunction (every omegaWallFunction patch, cornerWeights 1/n): G and
// omega of the wall cells; the cells' flag for setValues.
template <class R>
__global__ void kOmegaWall(Mesh<R> m, SstK<R> K, const u8* bOm, const R* yW, const int* bPatch,
                           const FvWallFnPatch* wf, const R* k, const R* nutB, const R* U,
                           const R* Ub, const R* vfU, R* om, R* Gk, u8* fix)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const int e0 = m.cStart[c], e1 = m.cStart[c + 1];
    int nw = 0;
    for (int e = e0; e < e1; ++e) {
        const int code = m.cEnt[e];
        if (code < 0 && bOm[-code - 1] == kSOmegaWall) ++nw;
    }
    fix[c] = nw > 0;
    if (!nw) return;
    const R w = R(1.0) / R(nw);
    R om0 = 0, G0 = 0;
    const R kc = k[c], nuw = K.nu;
    for (int e = e0; e < e1; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) continue;
        const int b = -code - 1;
        if (bOm[b] != kSOmegaWall) continue;
        const FvWallFnPatch& P = wf[bPatch[b]];
        // the patch constants in R (R = double: themselves)
        const R pCmu = R(P.Cmu), pBeta1 = R(P.beta1), pKappa = R(P.kappa), pYPlusLam = R(P.yPlusLam), pN = R(P.n);
        const R Cmu25 = sqrt(sqrt(pCmu));
        const R y = yW[b];
        const R yPlus = Cmu25 * y * sqrt(kc) / nuw;
        const R omegaVis = R(6.0) * nuw / (pBeta1 * sqr(y));
        const R omegaLog = sqrt(kc) / (Cmu25 * pKappa * y);
        R v = 0;
        switch (P.blend) {
        case 0: v = yPlus > pYPlusLam ? omegaLog : omegaVis; break;                       // stepwise
        case 1: v = fmax(omegaVis, omegaLog); break;                                     // max
        case 2: v = pow(pow(omegaVis, pN) + pow(omegaLog, pN), R(1.0) / pN); break;        // binomial
        case 3: {                                                                         // exponential
            const R Gamma = R(0.01) * pow4(yPlus) / (1 + 5 * yPlus);
            const R invGamma = R(1.0) / (Gamma + rootVSmall<R>());
            v = omegaVis * exp(-Gamma) + omegaLog * exp(-invGamma);
            break;
        }
        default: { // tanh
            const R b1 = omegaVis + omegaLog;
            const R b2 = pow(pow(omegaVis, R(1.2)) + pow(omegaLog, R(1.2)), R(1.0) / R(1.2));
            const R phiTanh = tanh(pow4(R(0.1) * yPlus));
            v = phiTanh * b1 + (1 - phiTanh) * b2;
            break;
        }
        }
        om0 += w * v;
        if (P.blend != 0 || yPlus > pYPlusLam) {
            R sn[3];
            uSnGrad(m, b, U, Ub, vfU, sn);
            const R magGradUw = sqrt(sn[0] * sn[0] + sn[1] * sn[1] + sn[2] * sn[2]);
            G0 += w * (nutB[b] + nuw) * magGradUw * Cmu25 * sqrt(kc) / (pKappa * y);
        }
    }
    Gk[c] = G0;
    om[c] = om0;
}

// omega's wall-function patch values (opf == omega0 of the face cell).
template <class R>
__global__ void kOmegaWallB(Mesh<R> m, const u8* bOm, const R* om, R* omB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB || bOm[b] != kSOmegaWall) return;
    omB[b] = om[m.bOwn[b]];
}

// CDkOmega, F1, F23, gamma, beta, GbyNu (limited), DkEff, DomegaEff, and the
// omega equation's implicit / explicit sources (SST).
template <class R>
__global__ void kSstCell2(Mesh<R> m, SstK<R> K, const R* y, const R* k, const R* om, const R* gk,
                          const R* gom, const R* nut, const R* S2, const R* divU,
                          R* GbyNu, R* F1o, R* Dk, R* Dom, R* dAdd, R* sAdd)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    const R kc = k[c], oc = om[c], yc = y[c];
    const R dot = gk[cc] * gom[cc] + gk[nC + cc] * gom[nC + cc] + gk[2 * nC + cc] * gom[2 * nC + cc];
    const R CDk = (2 * K.alphaOmega2) * dot / oc;
    const R F1 = sstF1(K, kc, oc, yc, CDk);
    const R F23 = sstF23(K, kc, oc, yc);
    const R gamma = F1 * (K.gamma1 - K.gamma2) + K.gamma2;
    const R beta = F1 * (K.beta1 - K.beta2) + K.beta2;
    const R gbn = fmin(GbyNu[c], (K.c1 / K.a1) * K.betaStar * oc * fmax(K.a1 * oc, K.b1 * F23 * sqrt(S2[c])));
    GbyNu[c] = gbn;
    F1o[c] = F1;
    Dk[c] = (F1 * (K.alphaK1 - K.alphaK2) + K.alphaK2) * nut[c] + K.nu;
    Dom[c] = (F1 * (K.alphaOmega1 - K.alphaOmega2) + K.alphaOmega2) * nut[c] + K.nu;
    const R V = m.V[c];
    const R susp1 = (R(2.0) / R(3.0)) * gamma * divU[c];
    const R sp = beta * oc;
    const R susp2 = (F1 - R(1.0)) * CDk / oc;
    R d = V * fmax(susp1, R(0.0));
    d += V * sp;
    d += V * fmax(susp2, R(0.0));
    R s = V * (gamma * gbn);
    s -= V * fmin(susp1, R(0.0)) * oc;
    s -= V * fmin(susp2, R(0.0)) * oc;
    dAdd[c] = d;
    sAdd[c] = s;
}

// DkEff / DomegaEff on the boundary faces (F1 of the boundary values).
template <class R>
__global__ void kSstBnd(Mesh<R> m, SstK<R> K, SEq<R> ek, SEq<R> eo, const u8* bWall, const R* y, const R* nutB,
                        R* DkB, R* DomB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    R gk[3], go[3];
    sBndGrad(m, ek, b, gk);
    sBndGrad(m, eo, b, go);
    const R kb = ek.xb[b], ob = eo.xb[b];
    const R yb = bWall[b] ? R(kSmall) : y[m.bOwn[b]];
    const R CDk = (2 * K.alphaOmega2) * (gk[0] * go[0] + gk[1] * go[1] + gk[2] * go[2]) / ob;
    const R F1 = sstF1(K, kb, ob, yb, CDk);
    DkB[b] = (F1 * (K.alphaK1 - K.alphaK2) + K.alphaK2) * nutB[b] + K.nu;
    DomB[b] = (F1 * (K.alphaOmega1 - K.alphaOmega2) + K.alphaOmega2) * nutB[b] + K.nu;
}

// The k equation's sources (with the new omega): Pk(G), -SuSp(2/3 divU), -Sp(betaStar omega).
template <class R>
__global__ void kSstKSrc(Mesh<R> m, SstK<R> K, const R* k, const R* om, const R* Gk, const R* divU,
                         R* dAdd, R* sAdd)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const R V = m.V[c], kc = k[c], oc = om[c];
    const R Pk = fmin(Gk[c], (K.c1 * K.betaStar) * kc * oc);
    const R susp = (R(2.0) / R(3.0)) * divU[c];
    R d = V * fmax(susp, R(0.0));
    d += V * (K.betaStar * oc);
    R s = V * Pk;
    s -= V * fmin(susp, R(0.0)) * kc;
    dAdd[c] = d;
    sAdd[c] = s;
}

// SA: chi, fv1, ft2, Omega, Stilda, fw; the nuTilda equation's sources and
// DnuTildaEff (cells).
template <class R>
__global__ void kSaCell(Mesh<R> m, SaK<R> K, const R* G, const R* y, const R* nt, const R* gnt,
                        R* D, R* dAdd, R* sAdd)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), cc = std::size_t(c);
    R g[3][3];
    loadG(G, m.nC, c, g);
    const R n = nt[c], d = y[c];
    const R chi = n / K.nu;
    const R fv1 = saFv1(K, chi);
    const R ft2 = K.ft2 ? K.Ct3 * exp(-K.Ct4 * sqr(chi)) : R(0.0);
    const R Om = OmegaOf(g);
    const R fv2 = R(1.0) - chi / (R(1.0) + chi * fv1);
    const R Stilda = fmax(Om + fv2 * n / sqr(K.kappa * d), K.Cs * Om);
    const R r = fmin(n / (fmax(Stilda, R(kSmall)) * sqr(K.kappa * d)), R(10.0));
    const R gg = r + K.Cw2 * (pow(r, R(6.0)) - r);
    const R c6 = pow(K.Cw3, R(6.0));
    const R fw = gg * pow((1 + c6) / (pow(gg, R(6.0)) + c6), R(1.0) / R(6.0));
    const R gm2 = sqr(gnt[cc]) + sqr(gnt[nC + cc]) + sqr(gnt[2 * nC + cc]);
    const R V = m.V[c];
    R s = V * ((K.Cb2 / K.sigmaNut) * gm2);
    s += V * (K.Cb1 * Stilda * n * (R(1.0) - ft2));
    const R sp = (K.Cw1 * fw - K.Cb1 / sqr(K.kappa) * ft2) * n / sqr(d);
    dAdd[c] = V * sp;
    sAdd[c] = s;
    D[c] = (n + K.nu) / K.sigmaNut;
}

template <class R>
__global__ void kSaBnd(Mesh<R> m, SaK<R> K, const R* ntB, R* DB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    DB[b] = (ntB[b] + K.nu) / K.sigmaNut;
}

// --- kernels: the scalar transport matrix -------------------------------------------

// Faces: convection weights (linear | upwind | linearUpwind | limitedLinear) and
// -laplacian(Gamma), the explicit corrections (laplacian non-orthogonal, LUD).
template <class R, bool LimLap>
__global__ void __launch_bounds__(256) kSFace(Mesh<R> m, Ctl<R> c, SEq<R> e, const R* phi, R* up, R* lo,
                                              R* fx)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nI = std::size_t(m.nI), nC = std::size_t(m.nC), ff = std::size_t(f);
    const int o = m.own[f], n = m.nei[f];
    const R l = m.w[f], ph = phi[f];
    R wc;
    switch (e.div) {
    case kSDivLinear: wc = l; break;
    case kSDivLL: {
        const R gradf = e.x[n] - e.x[o];
        const int cu = ph > 0 ? o : n;
        R gradcf = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) gradcf += e.dlt[std::size_t(i) * nI + ff] * e.gx[std::size_t(i) * nC + std::size_t(cu)];
        R r;
        if (fabs(gradcf) >= 1000 * fabs(gradf)) r = 2 * 1000 * signF(gradcf) * signF(gradf) - 1;
        else r = 2 * (gradcf / gradf) - 1;
        const R lim = fmin(fmax(e.twoByk * r, R(0.0)), R(1.0));
        wc = lim * l + (R(1.0) - lim) * pos0(ph);
        break;
    }
    default: wc = pos0(ph); break;
    }
    const R lowC = -wc * ph, upC = lowC + ph;
    const R gm = (l * (e.gC[o] - e.gC[n]) + e.gC[n]) * m.magS[f];
    const R lap = m.dc[f] * gm;
    up[f] = upC - lap;
    lo[f] = lowC - lap;
    R corr = 0;
    if (c.lapCorr) {
        R kc = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const R go = e.gx[std::size_t(i) * nC + std::size_t(o)], gn = e.gx[std::size_t(i) * nC + std::size_t(n)];
            kc += m.k[std::size_t(i) * nI + ff] * (l * (go - gn) + gn);
        }
        if constexpr (LimLap) corr = gm * (snLimiter(c.lapLim, fabs(m.dc[f] * (e.x[n] - e.x[o])), fabs(kc)) * kc);
        else corr = gm * kc;
    }
    R cl = 0;
    if (e.div == kSDivLUD) {
        const int cu = ph > 0 ? o : n;
#pragma unroll
        for (int i = 0; i < 3; ++i)
            cl += (ph > 0 ? m.rO : m.rN)[std::size_t(i) * nI + ff] * e.gx[std::size_t(i) * nC + std::size_t(cu)];
    }
    fx[f] = corr - ph * cl;
}

// Boundary coefficients (convection + -laplacian) and the mixed updateCoeffs.
template <class R>
__global__ void kSBnd(Mesh<R> m, SEq<R> e, const R* phiB, R* ic, R* bc)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const R ph = phiB[b], dc = m.bDc[b], gm = e.gB[b] * m.bMagS[b], xb = e.xb[b];
    R vic, vbc, gic, gbc;
    switch (e.code[b]) {
    case kSFixed:
    case kSOmegaWall: vic = 0; vbc = xb; gic = -dc; gbc = dc * xb; break;
    case kSInletOutlet: {
        const R vf = ph < 0 ? R(1.0) : R(0.0);
        e.vf[b] = vf;
        const R ref = e.ref[b];
        vic = R(1.0) * (R(1.0) - vf);
        vbc = (R(1.0) - vf) * (R(0.0) / dc) + vf * ref;
        gic = -R(1.0) * vf * dc;
        gbc = (R(1.0) - vf) * R(0.0) + vf * (dc * ref);
        break;
    }
    default: vic = 1; vbc = 0; gic = 0; gbc = 0; break;
    }
    ic[b] = ph * vic - gm * gic;
    bc[b] = -ph * vbc + gm * gbc;
}

// Rows: negSumDiag, off entries, explicit face sources, bounded, the model's
// sources, fvMatrix::relax(alpha) (alpha <= 0: none), setValues on the fixed
// cells (fix != null), then the solved system (addBoundaryDiag/Source).
template <class R>
__global__ void __launch_bounds__(256) kSRow(Mesh<R> m, SEq<R> e, const R* up, const R* lo, const R* fx,
                                             const R* phi, const R* phiB, const R* ic,
                                             const R* bc, const R* dAdd, const R* sAdd,
                                             R alpha, const u8* fix, R* off, R* Ad, R* rhs)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const int e0 = m.cStart[c], e1 = m.cStart[c + 1];
    R diag = 0, divphi = 0, sumOff = 0, icMag = 0, icMin = 0, s = 0;
    for (int k = e0; k < e1; ++k) {
        const int code = m.cEnt[k];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            if (code & 1) {
                diag -= up[f];
                sumOff += fabs(lo[f]);
                divphi -= phi[f];
                s -= fx[f];
            } else {
                diag -= lo[f];
                sumOff += fabs(up[f]);
                divphi += phi[f];
                s += fx[f];
            }
        } else {
            const std::size_t b = std::size_t(-code - 1);
            divphi += phiB[b];
            icMag += fabs(ic[b]);
            icMin += ic[b];
        }
    }
    const R V = m.V[c], x = e.x[c];
    s = V * (s / V);
    if (e.bounded) diag -= V * (divphi / V);
    diag += dAdd[c];
    s += sAdd[c];
    if (alpha > 0) {
        const R D0 = diag;
        R D = diag + icMag;
        D = fmax(fabs(D), sumOff);
        D /= alpha;
        D -= icMin;
        s += (D - D0) * x;
        diag = D;
    }
    const bool fixed = fix && fix[c];
    R bd = 0, bs = 0;
    for (int k = e0; k < e1; ++k) {
        const int code = m.cEnt[k];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            const int other = (code & 1) ? m.own[f] : m.nei[f];
            R a = (code & 1) ? lo[f] : up[f];
            if (fixed) a = 0;
            else if (fix && fix[other]) {
                s -= a * e.x[other];
                a = 0;
            }
            off[m.cEll[k]] = a;
        } else if (!fixed) {
            const std::size_t b = std::size_t(-code - 1);
            bd += ic[b];
            bs += bc[b];
        }
    }
    if (fixed) {
        Ad[c] = diag;
        rhs[c] = x * diag;
        return;
    }
    Ad[c] = diag + bd;
    rhs[c] = s + bs;
}

// The mixed patches' updateCoeffs alone (omega_.boundaryFieldRef().updateCoeffs()
// before CDkOmega: grad(omega)'s boundary snGrad sees the new valueFraction).
template <class R>
__global__ void kSUpdate(Mesh<R> m, SEq<R> e, const R* phiB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB || e.code[b] != kSInletOutlet) return;
    e.vf[b] = phiB[b] < 0 ? R(1.0) : R(0.0);
}

// correctBoundaryConditions of a transported scalar.
template <class R>
__global__ void kSEval(Mesh<R> m, SEq<R> e)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const R xP = e.x[m.bOwn[b]];
    switch (e.code[b]) {
    case kSZeroGrad:
    case kSSlip: e.xb[b] = xP; break;
    case kSInletOutlet: {
        const R vf = e.vf[b];
        e.xb[b] = (R(1.0) - vf) * (xP + R(0.0) / m.bDc[b]) + vf * e.ref[b];
        break;
    }
    default: break;
    }
}

// --- bound() ---------------------------------------------------------------------------

template <class R>
__global__ void kMinPart(int n, const R* x, R* part)
{
    __shared__ R sh[kT];
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    sh[threadIdx.x] = i < n ? x[i] : R(1e300);
    __syncthreads();
    for (int s = kT / 2; s > 0; s >>= 1) {
        if (int(threadIdx.x) < s) sh[threadIdx.x] = fmin(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}

// max(max(x, average(max(x, lb)) pos0(-x)), lb) into xn.
template <class R>
__global__ void kBoundCells(Mesh<R> m, const R* x, const R* xb, R lb, R* xn)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R num = 0, den = 0;
    for (int k = m.cStart[c]; k < m.cStart[c + 1]; ++k) {
        const int code = m.cEnt[k];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            const R l = m.w[f];
            const R vo = fmax(x[m.own[f]], lb), vn = fmax(x[m.nei[f]], lb);
            num += m.magS[f] * (l * (vo - vn) + vn);
            den += m.magS[f];
        } else {
            const std::size_t b = std::size_t(-code - 1);
            num += m.bMagS[b] * fmax(xb[b], lb);
            den += m.bMagS[b];
        }
    }
    const R avg = num / den, v = x[c];
    xn[c] = fmax(fmax(v, avg * pos0(-v)), lb);
}

// Boundary after bound(): zeroGradient-type patches max-assigned, slip /
// symmetry re-evaluated, fixedValue / mixed untouched (operator= no-ops).
template <class R>
__global__ void kBoundB(Mesh<R> m, SEq<R> e, R lb)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    if (e.code[b] == kSZeroGrad) e.xb[b] = fmax(e.xb[b], lb);
    else if (e.code[b] == kSSlip) e.xb[b] = e.x[m.bOwn[b]];
}

// --- correctNut -------------------------------------------------------------------------------

template <class R>
__global__ void kSstNut(Mesh<R> m, SstK<R> K, const R* y, const R* k, const R* om, const R* S2,
                        R* nut, R* nuEff)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const R F23 = sstF23(K, k[c], om[c], y[c]);
    const R v = sstNut(K, k[c], om[c], F23, S2[c]);
    nut[c] = v;
    nuEff[c] = v + K.nu;
}

template <class R>
__global__ void kSaNut(Mesh<R> m, SaK<R> K, const R* nt, R* nut, R* nuEff)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const R v = nt[c] * saFv1(K, nt[c] / K.nu);
    nut[c] = v;
    nuEff[c] = v + K.nu;
}

// nut's boundary values: calculated = the model expression of the boundary
// values; wall functions (fixedValue-derived: the assignment is a no-op, the
// previous value is Spalding's first guess); cell / fixed.
template <class R, bool Sst>
__global__ void kNutB(Mesh<R> m, SstK<R> Ks, SaK<R> Ka, const u8* bNut, const u8* bWall, const R* y, const R* yW,
                      const int* bPatch, const FvWallFnPatch* wf, const R* s0, const R* s0B,
                      const R* s1B, const R* U, const R* Ub, const R* vfU, const R* G,
                      const R* nut, R* nutB, R* nuEffB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const int P = m.bOwn[b];
    const R nuw = Sst ? Ks.nu : Ka.nu;
    R v = nutB[b];
    switch (bNut[b]) {
    case kNutCalc: {
        if constexpr (Sst) {
            R gb[3][3];
            uBndGradAt(m, b, U, Ub, vfU, G, gb);
            const R yb = bWall[b] ? R(kSmall) : y[P];
            const R kb = s0B[b], ob = s1B[b];
            v = sstNut(Ks, kb, ob, sstF23(Ks, kb, ob, yb), S2of(gb));
        } else {
            v = s0B[b] * saFv1(Ka, s0B[b] / Ka.nu);
        }
        break;
    }
    case kNutCell: v = nut[P]; break;
    case kNutLowRe: v = R(0.0); break;
    case kNutSpalding: {
        const FvWallFnPatch& W = wf[bPatch[b]];
        const R wKappa = R(W.kappa), wE = R(W.E);
        const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC);
        R sn[3];
        uSnGrad(m, b, U, Ub, vfU, sn);
        const R magGradU = sqrt(sn[0] * sn[0] + sn[1] * sn[1] + sn[2] * sn[2]);
        R du[3];
#pragma unroll
        for (int q = 0; q < 3; ++q) du[q] = U[std::size_t(q) * nC + std::size_t(P)] - Ub[std::size_t(q) * nB + std::size_t(b)];
        const R magUp = sqrt(du[0] * du[0] + du[1] * du[1] + du[2] * du[2]);
        const R yy = yW[b];
        R uTau = 0;
        R ut = sqrt((v + nuw) * magGradU);
        if (rootVSmall<R>() < ut) {
            int iter = 0;
            R err = 0;
            do {
                const R kUu = fmin(wKappa * magUp / ut, R(50.0));
                const R fkUu = exp(kUu) - 1 - kUu * (1 + R(0.5) * kUu);
                const R f = -ut * yy / nuw + magUp / ut + R(1.0) / wE * (fkUu - R(1.0) / R(6.0) * kUu * sqr(kUu));
                const R df = yy / nuw + magUp / sqr(ut) + R(1.0) / wE * kUu * fkUu / ut;
                const R uTauNew = ut + f / df;
                err = fabs((ut - uTauNew) / ut);
                ut = uTauNew;
            } while (ut > rootVSmall<R>() && err > R(0.01) && ++iter < W.maxIter);
            uTau = fmax(R(0.0), ut);
        }
        v = fmax(R(0.0), sqr(uTau) / (magGradU + rootVSmall<R>()) - nuw);
        break;
    }
    case kNutK: {
        if constexpr (Sst) {
            const FvWallFnPatch& W = wf[bPatch[b]];
            const R wCmu = R(W.Cmu), wKappa = R(W.kappa), wE = R(W.E), wYPlusLam = R(W.yPlusLam), wN = R(W.nutN);
            const R Cmu25 = sqrt(sqrt(wCmu));
            const R yPlus = Cmu25 * yW[b] * sqrt(s0[P]) / nuw;
            const R nutLog = nuw * yPlus * wKappa / log(fmax(wE * yPlus, 1 + R(1e-4)));
            R nw;
            switch (W.nutBlend) {
            case 0: nw = yPlus > wYPlusLam ? nutLog : nuw; break;
            case 1: nw = fmax(nuw, nutLog); break;
            case 2: nw = pow(pow(nuw, wN) + pow(nutLog, wN), R(1.0) / wN); break;
            case 3: {
                const R Gamma = R(0.01) * pow4(yPlus) / (1 + 5 * yPlus);
                const R invGamma = R(1.0) / (Gamma + rootVSmall<R>());
                nw = nuw * exp(-Gamma) + nutLog * exp(-invGamma);
                break;
            }
            default: {
                const R b1 = nuw + nutLog;
                const R b2 = pow(pow(nuw, R(1.2)) + pow(nutLog, R(1.2)), R(1.0) / R(1.2));
                const R phiTanh = tanh(pow4(R(0.1) * yPlus));
                nw = phiTanh * b1 + (1 - phiTanh) * b2;
                break;
            }
            }
            v = nw - nuw;
        }
        break;
    }
    default: break; // fixed
    }
    nutB[b] = v;
    nuEffB[b] = v + nuw;
}

template <class T>
std::expected<void, std::string> put(DeviceBuffer<T>& d, const std::vector<T>& v, const char* what)
{
    auto x = upload(v, what);
    if (!x) return std::unexpected(x.error());
    d = std::move(*x);
    return {};
}

// A host FP64 vector as R (uploadAs).
template <class R>
std::expected<void, std::string> putR(DeviceBuffer<R>& d, const std::vector<double>& v, const char* what)
{
    auto x = uploadAs<R>(v, what);
    if (!x) return std::unexpected(x.error());
    d = std::move(*x);
    return {};
}

template <class T>
std::expected<void, std::string> zero(DeviceBuffer<T>& d, std::size_t n, const char* what)
{
    auto x = zeros<T>(n, what);
    if (!x) return std::unexpected(x.error());
    d = std::move(*x);
    return {};
}

template <class R>
SstK<R> sstK(const RasState<R>& Rs)
{
    const FvRas& c = Rs.co;
    return {R(c.alphaK1), R(c.alphaK2), R(c.alphaOmega1), R(c.alphaOmega2), R(c.gamma1), R(c.gamma2), R(c.beta1),
            R(c.beta2), R(c.betaStar), R(c.a1), R(c.b1), R(c.c1), c.F3 ? 1 : 0, R(Rs.nu)};
}

template <class R>
SaK<R> saK(const RasState<R>& Rs)
{
    const FvRas& c = Rs.co;
    const double Cw1 = c.Cb1 / (c.kappa * c.kappa) + (1.0 + c.Cb2) / c.sigmaNut;
    return {R(c.sigmaNut), R(c.kappa), R(c.Cb1), R(c.Cb2), R(Cw1), R(c.Cw2), R(c.Cw3), R(c.Cv1), R(c.Cs),
            R(c.Ct3), R(c.Ct4), c.ft2 ? 1 : 0, R(Rs.nu)};
}

int divCode(const FvConvScheme& s)
{
    switch (s.kind) {
    case FvSDiv::Linear: return kSDivLinear;
    case FvSDiv::Upwind: return kSDivUpwind;
    case FvSDiv::LinearUpwind: return kSDivLUD;
    case FvSDiv::LimitedLinear: return kSDivLL;
    }
    return kSDivUpwind;
}

// Scalar q's equation view (cell / boundary diffusivity D[q] / DB[q]).
template <class R>
SEq<R> seq(SimpleImpl<R>& I, int q)
{
    RasState<R>& Rs = *I.ras;
    const FvRas& c = Rs.co;
    const FvConvScheme& cs = c.model == FvTurbModel::KOmegaSST ? (q == 0 ? c.divK : c.divOmega) : c.divNuTilda;
    SEq<R> e;
    e.x = Rs.s[std::size_t(q)].data();
    e.xb = Rs.sb[std::size_t(q)].data();
    e.code = Rs.bS[std::size_t(q)].data();
    e.ref = Rs.bRef[std::size_t(q)].data();
    e.vf = Rs.vf[std::size_t(q)].data();
    e.gC = Rs.D[std::size_t(q)].data();
    e.gB = Rs.DB[std::size_t(q)].data();
    e.gx = Rs.gs[std::size_t(q)].data();
    e.dlt = Rs.dlt.data();
    e.div = divCode(cs);
    e.bounded = cs.bounded;
    e.twoByk = R(2.0 / std::max(cs.k, 1e-15));
    return e;
}

const FvGradScheme& gradOf(const FvRas& c, int q)
{
    if (c.model == FvTurbModel::KOmegaSST) return q == 0 ? c.gradK : c.gradOmega;
    return c.gradNuTilda;
}

template <class R>
void gradS(SimpleImpl<R>& I, int q)
{
    RasState<R>& Rs = *I.ras;
    const FvGradScheme& g = gradOf(Rs.co, q);
    kGrad<R, 1><<<grid(std::size_t(I.m.nC)), kT>>>(I.m, Rs.s[std::size_t(q)].data(), Rs.sb[std::size_t(q)].data(),
                                                    g.kind == FvGrad::CellLimited, R(g.k), Rs.gs[std::size_t(q)].data());
}

// bound(x, lb): true when it acted (the comparison and the bound in R).
template <class R>
std::expected<bool, std::string> boundS(SimpleImpl<R>& I, int q, double lbIn)
{
    RasState<R>& Rs = *I.ras;
    const Mesh<R>& m = I.m;
    const R lb = R(lbIn);
    const std::size_t nC = std::size_t(m.nC);
    const unsigned nb = grid(nC);
    kMinPart<R><<<nb, kT>>>(int(nC), Rs.s[std::size_t(q)].data(), Rs.minPart.data());
    std::vector<R> h(nb);
    if (auto e = hipMemcpy(h.data(), Rs.minPart.data(), nb * sizeof(R), hipMemcpyDeviceToHost); e != hipSuccess)
        return std::unexpected(std::string("fv: bound: ") + hipGetErrorString(e));
    const R mn = *std::ranges::min_element(h);
    if (!std::isfinite(mn)) return std::unexpected("fv: non-finite turbulence field");
    if (!(mn < lb)) return false;
    R* tmp = I.tmp.data();
    kBoundCells<R><<<nb, kT>>>(m, Rs.s[std::size_t(q)].data(), Rs.sb[std::size_t(q)].data(), lb, tmp);
    (void)hipMemcpyAsync(Rs.s[std::size_t(q)].data(), tmp, nC * sizeof(R), hipMemcpyDeviceToDevice, 0);
    kBoundB<R><<<grid(std::size_t(m.nB)), kT>>>(m, seq(I, q), lb);
    return true;
}

// Assemble, relax, (fix), solve scalar q.
template <class R>
std::expected<FvSolveLog, std::string> solveS(SimpleImpl<R>& I, int q, const FvSolverCtl& ctl,
                                               const std::optional<double>& relax, bool fixWall)
{
    RasState<R>& Rs = *I.ras;
    const Mesh<R>& m = I.m;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    const SEq<R> e = seq(I, q);
    if (I.c.lapLim >= 0)
        kSFace<R, true><<<grid(nI), kT>>>(m, I.c, e, I.phi.data(), Rs.sup.data(), Rs.slo.data(), Rs.sfx.data());
    else kSFace<R, false><<<grid(nI), kT>>>(m, I.c, e, I.phi.data(), Rs.sup.data(), Rs.slo.data(), Rs.sfx.data());
    kSBnd<R><<<grid(nB), kT>>>(m, e, I.phiB.data(), Rs.sic.data(), Rs.sbc.data());
    kSRow<R><<<grid(nC), kT>>>(m, e, Rs.sup.data(), Rs.slo.data(), Rs.sfx.data(), I.phi.data(), I.phiB.data(),
                               Rs.sic.data(), Rs.sbc.data(), Rs.dAdd.data(), Rs.sAdd.data(), R(relax ? *relax : 0.0),
                               fixWall ? Rs.fixC.data() : nullptr, I.AU->off(), I.AU->diag(), I.rhs.data());
    const la::LinOp<R> M = [&I](const R* r, R* z) { I.gsU->apply(r, z); };
    auto r = I.solve(*I.AU, M, false, ctl, I.floor.s, Rs.s[std::size_t(q)].data(), &*I.gsU); // refactors gsU first
    if (!r) return std::unexpected(r.error());
    kSEval<R><<<grid(nB), kT>>>(m, e);
    return *r;
}

// correctNut + nuEff (cells, boundary).
template <class R>
void correctNut(SimpleImpl<R>& I)
{
    RasState<R>& Rs = *I.ras;
    const Mesh<R>& m = I.m;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
    if (Rs.co.model == FvTurbModel::KOmegaSST) {
        kSstNut<R><<<grid(nC), kT>>>(m, sstK(Rs), Rs.y.data(), Rs.s[0].data(), Rs.s[1].data(), Rs.S2.data(),
                                     Rs.nut.data(), Rs.nuEff.data());
        kNutB<R, true><<<grid(nB), kT>>>(m, sstK(Rs), SaK<R>{}, Rs.bNut.data(), Rs.bWall.data(), Rs.y.data(),
                                         Rs.yWall.data(), Rs.bPatch.data(), Rs.wf.data(), Rs.s[0].data(),
                                         Rs.sb[0].data(), Rs.sb[1].data(), I.U.data(), I.Ub.data(), I.vfU.data(),
                                         I.G.data(), Rs.nut.data(), Rs.nutB.data(), Rs.nuEffB.data());
    } else {
        kSaNut<R><<<grid(nC), kT>>>(m, saK(Rs), Rs.s[0].data(), Rs.nut.data(), Rs.nuEff.data());
        kNutB<R, false><<<grid(nB), kT>>>(m, SstK<R>{}, saK(Rs), Rs.bNut.data(), Rs.bWall.data(), Rs.y.data(),
                                          Rs.yWall.data(), Rs.bPatch.data(), Rs.wf.data(), Rs.s[0].data(),
                                          Rs.sb[0].data(), Rs.sb[0].data(), I.U.data(), I.Ub.data(), I.vfU.data(),
                                          I.G.data(), Rs.nut.data(), Rs.nutB.data(), Rs.nuEffB.data());
    }
}

} // namespace

template <class R>
std::expected<void, std::string> rasMake(SimpleImpl<R>& I, const FvSimpleSetup& s, const FoamCase& fc)
{
    I.ras = std::make_unique<RasState<R>>();
    RasState<R>& Rs = *I.ras;
    const FvRasSetup& S = s.ras;
    Rs.co = fc.ras;
    Rs.nS = S.nS;
    Rs.nu = fc.nu;
    Rs.wfHost = S.wf;
    const std::size_t nC = std::size_t(s.nC), nI = std::size_t(s.nI), nB = std::size_t(s.nB);
#define FV_TRY(x)                                                                                                   \
    if (auto r_ = (x); !r_) return std::unexpected(r_.error());
    FV_TRY(putR(Rs.y, S.y, "y")) FV_TRY(putR(Rs.yWall, S.yWall, "yWall")) FV_TRY(put(Rs.bWall, S.bWall, "bWall"))
    FV_TRY(put(Rs.bNut, S.bNut, "bNut")) FV_TRY(put(Rs.bPatch, s.bPatch, "bPatch")) FV_TRY(put(Rs.wf, S.wf, "wf"))
    FV_TRY(putR(Rs.nutB, S.bNut0, "nutB")) FV_TRY(putR(Rs.nut, S.nut0, "nut")) FV_TRY(putR(Rs.dlt, S.dlt, "delta"))
    FV_TRY(zero(Rs.nuEff, nC, "nuEff")) FV_TRY(zero(Rs.nuEffB, nB, "nuEffB")) FV_TRY(zero(Rs.fixC, nC, "fixC"))
    for (int q = 0; q < 2; ++q) {
        const std::size_t Q = std::size_t(q);
        if (q < Rs.nS) {
            std::vector<std::uint8_t> code = S.bS[Q];
            // a scalar's slip / symmetry: kSSlip (bound() re-evaluates it)
            for (std::size_t b = 0; b < nB; ++b) {
                const FvPatchBc& pb = fc.patches[std::size_t(s.bPatch[b])];
                const auto& bcs = fc.ras.model == FvTurbModel::KOmegaSST ? (q == 0 ? fc.ras.k : fc.ras.omega)
                                                                         : fc.ras.nuTilda;
                const std::string& t = bcs[std::size_t(s.bPatch[b])].type;
                if (code[b] == std::uint8_t(FvSCode::ZeroGrad) &&
                    (t == "slip" || t == "symmetry" || t == "symmetryPlane" || pb.meshType == FoamPatchType::Symmetry ||
                     pb.meshType == FoamPatchType::SymmetryPlane))
                    code[b] = kSSlip;
            }
            FV_TRY(put(Rs.bS[Q], code, "bS")) FV_TRY(putR(Rs.bRef[Q], S.bRef[Q], "bRef"))
            FV_TRY(putR(Rs.s[Q], S.s0[Q], "s")) FV_TRY(putR(Rs.sb[Q], S.bS0[Q], "sb"))
            FV_TRY(zero(Rs.vf[Q], nB, "vf")) FV_TRY(zero(Rs.gs[Q], 3 * nC, "grad"))
            FV_TRY(zero(Rs.D[Q], nC, "D")) FV_TRY(zero(Rs.DB[Q], nB, "DB"))
        }
    }
    FV_TRY(zero(Rs.Gk, nC, "G")) FV_TRY(zero(Rs.GbyNu, nC, "GbyNu")) FV_TRY(zero(Rs.S2, nC, "S2"))
    FV_TRY(zero(Rs.divU, nC, "divU")) FV_TRY(zero(Rs.F1, nC, "F1")) FV_TRY(zero(Rs.dAdd, nC, "dAdd"))
    FV_TRY(zero(Rs.sAdd, nC, "sAdd")) FV_TRY(zero(Rs.sup, nI, "sup")) FV_TRY(zero(Rs.slo, nI, "slo"))
    FV_TRY(zero(Rs.sfx, nI, "sfx")) FV_TRY(zero(Rs.sic, nB, "sic")) FV_TRY(zero(Rs.sbc, nB, "sbc"))
    FV_TRY(zero(Rs.minPart, grid(nC) + 1, "minPart"))
#undef FV_TRY
    const Mesh<R>& m = I.m;
    // The model constructor: bound(k, kMin), bound(omega, omegaMin).
    if (Rs.co.model == FvTurbModel::KOmegaSST) {
        if (auto b = boundS(I, 0, Rs.co.kMin); !b) return std::unexpected(b.error());
        if (auto b = boundS(I, 1, Rs.co.omegaMin); !b) return std::unexpected(b.error());
    }
    // validate(): correctNut on the initial fields (gradU of the initial U).
    kGrad<R, 3><<<grid(nC), kT>>>(m, I.U.data(), I.Ub.data(), I.c.limU, I.c.kU, I.G.data());
    if (Rs.co.model == FvTurbModel::KOmegaSST)
        kSstCell1<R><<<grid(nC), kT>>>(m, I.G.data(), I.phi.data(), I.phiB.data(), Rs.nut.data(), Rs.S2.data(),
                                       Rs.GbyNu.data(), Rs.Gk.data(), Rs.divU.data());
    correctNut(I);
    if (auto e = hipDeviceSynchronize(); e != hipSuccess)
        return std::unexpected(std::string("fv: RAS init: ") + hipGetErrorString(e));
    if (auto e = hipGetLastError(); e != hipSuccess)
        return std::unexpected(std::string("fv: RAS init: ") + hipGetErrorString(e));
    return {};
}

template <class R>
std::expected<void, std::string> rasCorrect(SimpleImpl<R>& I, FvIterLog& lg)
{
    RasState<R>& Rs = *I.ras;
    const Mesh<R>& m = I.m;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
    const unsigned gC = grid(nC), gB = grid(nB);
    kGrad<R, 3><<<gC, kT>>>(m, I.U.data(), I.Ub.data(), I.c.limU, I.c.kU, I.G.data()); // fvc::grad(U)
    I.gValid = true; // of the final (U, Ub): the RAS correct writes neither
    lg.nT = 0;
    if (Rs.co.model == FvTurbModel::KOmegaSST) {
        const SstK<R> K = sstK(Rs);
        kSstCell1<R><<<gC, kT>>>(m, I.G.data(), I.phi.data(), I.phiB.data(), Rs.nut.data(), Rs.S2.data(),
                                 Rs.GbyNu.data(), Rs.Gk.data(), Rs.divU.data());
        // omega_.boundaryFieldRef().updateCoeffs(): the wall function's G and omega
        kOmegaWall<R><<<gC, kT>>>(m, K, Rs.bS[1].data(), Rs.yWall.data(), Rs.bPatch.data(), Rs.wf.data(),
                                  Rs.s[0].data(), Rs.nutB.data(), I.U.data(), I.Ub.data(), I.vfU.data(),
                                  Rs.s[1].data(), Rs.Gk.data(), Rs.fixC.data());
        kOmegaWallB<R><<<gB, kT>>>(m, Rs.bS[1].data(), Rs.s[1].data(), Rs.sb[1].data());
        kSUpdate<R><<<gB, kT>>>(m, seq(I, 1), I.phiB.data());
        gradS(I, 0);
        gradS(I, 1);
        kSstCell2<R><<<gC, kT>>>(m, K, Rs.y.data(), Rs.s[0].data(), Rs.s[1].data(), Rs.gs[0].data(), Rs.gs[1].data(),
                                 Rs.nut.data(), Rs.S2.data(), Rs.divU.data(), Rs.GbyNu.data(), Rs.F1.data(),
                                 Rs.D[0].data(), Rs.D[1].data(), Rs.dAdd.data(), Rs.sAdd.data());
        kSstBnd<R><<<gB, kT>>>(m, K, seq(I, 0), seq(I, 1), Rs.bWall.data(), Rs.y.data(), Rs.nutB.data(),
                               Rs.DB[0].data(), Rs.DB[1].data());
        auto ro = solveS(I, 1, Rs.co.solverOmega, Rs.co.relaxOmega, true);
        if (!ro) return std::unexpected("fv: omega: " + ro.error());
        lg.T[std::size_t(lg.nT++)] = *ro;
        auto bo = boundS(I, 1, Rs.co.omegaMin);
        if (!bo) return std::unexpected(bo.error());
        lg.bounded[1] = *bo;
        kSstKSrc<R><<<gC, kT>>>(m, K, Rs.s[0].data(), Rs.s[1].data(), Rs.Gk.data(), Rs.divU.data(), Rs.dAdd.data(),
                                Rs.sAdd.data());
        auto rk = solveS(I, 0, Rs.co.solverK, Rs.co.relaxK, false);
        if (!rk) return std::unexpected("fv: k: " + rk.error());
        lg.T[std::size_t(lg.nT++)] = *rk;
        auto bk = boundS(I, 0, Rs.co.kMin);
        if (!bk) return std::unexpected(bk.error());
        lg.bounded[0] = *bk;
    } else {
        const SaK<R> K = saK(Rs);
        gradS(I, 0);
        kSaCell<R><<<gC, kT>>>(m, K, I.G.data(), Rs.y.data(), Rs.s[0].data(), Rs.gs[0].data(), Rs.D[0].data(),
                               Rs.dAdd.data(), Rs.sAdd.data());
        kSaBnd<R><<<gB, kT>>>(m, K, Rs.sb[0].data(), Rs.DB[0].data());
        auto rn = solveS(I, 0, Rs.co.solverNuTilda, Rs.co.relaxNuTilda, false);
        if (!rn) return std::unexpected("fv: nuTilda: " + rn.error());
        lg.T[std::size_t(lg.nT++)] = *rn;
        auto bn = boundS(I, 0, 0.0);
        if (!bn) return std::unexpected(bn.error());
        lg.bounded[0] = *bn;
        kSEval<R><<<gB, kT>>>(m, seq(I, 0)); // nuTilda_.correctBoundaryConditions()
    }
    correctNut(I);
    for (int q = 0; q < lg.nT; ++q)
        if (!std::isfinite(lg.T[std::size_t(q)].initial) || !std::isfinite(lg.T[std::size_t(q)].final))
            return std::unexpected("fv: non-finite turbulence residual");
    return {};
}

template std::expected<void, std::string> rasMake(SimpleImpl<double>&, const FvSimpleSetup&, const FoamCase&);
template std::expected<void, std::string> rasMake(SimpleImpl<float>&, const FvSimpleSetup&, const FoamCase&);
template std::expected<void, std::string> rasCorrect(SimpleImpl<double>&, FvIterLog&);
template std::expected<void, std::string> rasCorrect(SimpleImpl<float>&, FvIterLog&);

} // namespace dns::gpu
