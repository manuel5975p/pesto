// cyclicAMI on the incompressible FV path (docs/fvpimple.md "cyclicAMI"):
// OpenFOAM v2606's coupled patch (coupledFvPatchField / cyclicAMIFvPatchField,
// no transform) in every operator pimpleFoam uses, on the faces of
// FvSimpleSetup::ami. The patch kernels of gpu/fv_simple treat such a face as
// zeroGradient; the kernels here run after them and overwrite what they wrote
// with the coupled expression:
//   face value            lerp(pnf, pif, w) = (1 - w) pnf + w pif
//   snGrad                dcNO (pnf - pif)
//   convection            internalCoeffs phi wc, boundaryCoeffs -phi (1 - wc), wc the scheme's weight
//   laplacian(gamma)      internalCoeffs -gamma_f |S| dcNO, boundaryCoeffs -gamma_f |S| dcNO,
//                         faceFluxCorrection gamma_f |S| limiter (k & lerp(grad))
//   matrix                row P gains -boundaryCoeffs w_j at column N_j (the
//                         interface update of lduMatrix::Amul, assembled)
// pnf = sum_j w_j x[N_j] (fvd::amiNbr), w / dcNO / k / delta the patch's
// (FvAmiSetup). One thread per coupled face or per row: deterministic.
#include "dns/gpu/internal/fv_device.hpp"

#include <algorithm>

namespace dns::gpu {

using namespace fvd;

namespace {

// evaluate(): xb = lerp(pnf, pif, w), NC components.
template <class R>
__global__ void kAmiEval(Mesh<R> m, int NC, const R* x, R* xb)
{
    const int a = int(blockIdx.x) * kT + int(threadIdx.x);
    if (a >= m.nA) return;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
    const int b = m.aB[a], P = m.bOwn[b];
    const R w = m.aWt[a];
    for (int q = 0; q < NC; ++q) {
        const R* xq = x + std::size_t(q) * nC;
        xb[std::size_t(q) * nB + std::size_t(b)] = amiLerp(w, xq[P], amiNbr(m, a, xq));
    }
}

// The neighbour gradient tensor sum_j w_j G(N_j), layout loadG's.
template <class R>
__device__ inline void amiNbrG(const Mesh<R>& m, int a, const R* G, R (&g)[3][3])
{
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int q = 0; q < 3; ++q) g[i][q] = amiNbr(m, a, G + std::size_t(3 * i + q) * std::size_t(m.nC));
}

// UEqn on the coupled faces (kUBnd's outputs): fvm::div(phi, U) -
// fvm::laplacian(nuEff, U) coefficients, and the explicit boundary flux
// Sf & interpolate(nuEff dev2(T(grad U))) + laplacian correction - phi x the
// linearUpwind correction.
template <class R, bool Lim, bool Turb>
__global__ void __launch_bounds__(256) kAmiUBnd(Mesh<R> m, Ctl<R> c, const R* U, const R* G, const R* phiB,
                                                const R* nuC, R* ic, R* acU, R* bc, R* bfx)
{
#pragma clang fp contract(off)
    const int a = int(blockIdx.x) * kT + int(threadIdx.x);
    if (a >= m.nA) return;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB), nA = std::size_t(m.nA);
    const int b = m.aB[a], P = m.bOwn[b];
    const std::size_t bb = std::size_t(b);
    const R w = m.aWt[a], ph = phiB[b], dc = m.bDc[b];
    // gaussConvectionScheme: the scheme's weight on the patch (linear: w; upwind, linearUpwind: pos0(phi))
    const R wc = c.div == kDivLinear ? w : (ph >= 0 ? R(1.0) : R(0.0));
    // interpolate(nuEff) of the CELL field (laminar: the uniform volScalarField nu): pnf = sum_j w_j nu from zero,
    // 0 on a face without overlap -- never the constant itself there.
    R nuN = 0;
    if constexpr (Turb) nuN = amiNbr(m, a, nuC);
    else
        for (int j = m.aStart[a]; j < m.aStart[a + 1]; ++j) nuN += m.aW[j] * c.nu;
    const R nuf = amiLerp(w, Turb ? nuC[P] : c.nu, nuN);
    const R gm = nuf * m.bMagS[b];
    const R icv = ph * wc - gm * (-dc);
    const R bcv = -ph * (R(1.0) - wc) + gm * dc;
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        ic[std::size_t(q) * nB + bb] = icv;
        bc[std::size_t(q) * nB + bb] = R(0.0);
    }
    acU[b] = bcv;
    R gP[3][3], gN[3][3], Xp[3][3], Xn[3][3], up[3], un[3], S[3], kk[3];
    loadG(G, m.nC, P, gP);
    amiNbrG(m, a, G, gN);
    nuDev2T(gP, Turb ? nuC[P] : c.nu, Xp);
    // patchNeighbourField of the cell field nuEff dev2(T(grad U))
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) Xn[i][j] = 0;
    for (int e = m.aStart[a]; e < m.aStart[a + 1]; ++e) {
        const int N = m.aCell[e];
        R g[3][3], X[3][3];
        loadG(G, m.nC, N, g);
        nuDev2T(g, Turb ? nuC[N] : c.nu, X);
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int j = 0; j < 3; ++j) Xn[i][j] += m.aW[e] * X[i][j];
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        up[i] = U[std::size_t(i) * nC + std::size_t(P)];
        un[i] = amiNbr(m, a, U + std::size_t(i) * nC);
        S[i] = m.bS[std::size_t(i) * nB + bb];
        kk[i] = m.aK[std::size_t(i) * nA + std::size_t(a)];
    }
    R cl[3] = {0, 0, 0};
    if (c.div == kDivLUD) { // linearUpwind::correction, coupled patch
        R r[3];
        if (ph > 0) {
#pragma unroll
            for (int i = 0; i < 3; ++i) r[i] = m.bCf[std::size_t(i) * nB + bb] - m.C[std::size_t(i) * nC + std::size_t(P)];
#pragma unroll
            for (int j = 0; j < 3; ++j) cl[j] = r[0] * gP[0][j] + r[1] * gP[1][j] + r[2] * gP[2][j];
        } else {
#pragma unroll
            for (int i = 0; i < 3; ++i)
                r[i] = m.bCf[std::size_t(i) * nB + bb] - m.aD[std::size_t(i) * nA + std::size_t(a)] -
                       m.C[std::size_t(i) * nC + std::size_t(P)];
#pragma unroll
            for (int j = 0; j < 3; ++j) cl[j] = r[0] * gN[0][j] + r[1] * gN[1][j] + r[2] * gN[2][j];
        }
    }
    R kc[3], sn[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        kc[j] = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) kc[j] += kk[i] * amiLerp(w, gP[i][j], gN[i][j]);
        sn[j] = dc * (un[j] - up[j]);
    }
    R lim = R(1.0);
    if constexpr (Lim)
        lim = snLimiter(c.lapLim, sqrt(sn[0] * sn[0] + sn[1] * sn[1] + sn[2] * sn[2]),
                        sqrt(kc[0] * kc[0] + kc[1] * kc[1] + kc[2] * kc[2]));
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        R fd = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) fd += S[i] * amiLerp(w, Xp[i][j], Xn[i][j]);
        R corr = 0;
        if constexpr (Lim) corr = gm * (lim * kc[j]);
        else if (c.lapCorr) corr = gm * kc[j];
        bfx[std::size_t(j) * nB + bb] = fd + corr - ph * cl[j];
    }
}

// The couplings of one matrix: row P's entry at column N_j gains -+ coef[b] w_j
// (ELL positions aEll; a pair coupled through several faces accumulates in
// ascending face order). One thread per row with coupled faces.
template <class R>
__global__ void kAmiOff(Mesh<R> m, const R* coef, int negate, const u8* fix, R* off)
{
#pragma clang fp contract(off)
    const int r = int(blockIdx.x) * kT + int(threadIdx.x);
    if (r >= m.nR) return;
    for (int t = m.rStart[r]; t < m.rStart[r + 1]; ++t) {
        const int a = m.rA[t];
        for (int e = m.aStart[a]; e < m.aStart[a + 1]; ++e) off[m.aEll[e]] = R(0.0);
    }
    // fvMatrix::setValues zeroes a fixed cell's internalCoeffs / boundaryCoeffs: no coupling in its row
    if (fix && fix[m.rCell[r]]) return;
    for (int t = m.rStart[r]; t < m.rStart[r + 1]; ++t) {
        const int a = m.rA[t];
        const R cf = coef[m.aB[a]];
        for (int e = m.aStart[a]; e < m.aStart[a + 1]; ++e) {
            const R v = cf * m.aW[e];
            off[m.aEll[e]] += negate ? -v : v;
        }
    }
}

// lduMatrix::sumA (the normFactor's row sum) subtracts a coupled face's
// boundaryCoeffs WHOLE; the assembled couplings of a face sum to coef sum_j w_j
// (= coef to rounding with normalised weights). A face without overlap has no
// address, so no assembled entry: its coefficient goes in here, with kAmiOff's
// sign and its fixed-row rule. Rows without such a face are not written. One
// thread per row with coupled faces; nComp component-major row sums (mask).
template <class R>
__global__ void kAmiSumA(Mesh<R> m, const R* coef, int negate, const u8* fix, int nComp, unsigned mask, R* sumA)
{
#pragma clang fp contract(off)
    const int r = int(blockIdx.x) * kT + int(threadIdx.x);
    if (r >= m.nR) return;
    const int cell = m.rCell[r];
    if (fix && fix[cell]) return;
    R s = 0;
    bool any = false;
    for (int t = m.rStart[r]; t < m.rStart[r + 1]; ++t) {
        const int a = m.rA[t];
        if (m.aStart[a] != m.aStart[a + 1]) continue;
        const R v = coef[m.aB[a]];
        s += negate ? -v : v;
        any = true;
    }
    if (!any) return;
    for (int q = 0; q < nComp; ++q)
        if ((mask >> q) & 1u) sumA[std::size_t(q) * std::size_t(m.nC) + std::size_t(cell)] += s;
}

// fvMatrix::addBoundarySource's coupled branch: bc = cmptMultiply(boundaryCoeffs, pnf).
template <class R>
__global__ void kAmiHsrc(Mesh<R> m, const R* U, const R* acU, R* bc)
{
#pragma clang fp contract(off)
    const int a = int(blockIdx.x) * kT + int(threadIdx.x);
    if (a >= m.nA) return;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
    const int b = m.aB[a];
#pragma unroll
    for (int q = 0; q < 3; ++q)
        bc[std::size_t(q) * nB + std::size_t(b)] = acU[b] * amiNbr(m, a, U + std::size_t(q) * nC);
}

// fvc::flux(HbyA) on a coupled face: Sf & lerp(pnf, pif, w).
template <class R>
__global__ void kAmiPhiH(Mesh<R> m, const R* HbyA, R* phiHB)
{
#pragma clang fp contract(off)
    const int a = int(blockIdx.x) * kT + int(threadIdx.x);
    if (a >= m.nA) return;
    const std::size_t nC = std::size_t(m.nC), nB = std::size_t(m.nB);
    const int b = m.aB[a], P = m.bOwn[b];
    const R w = m.aWt[a];
    R s = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const R* h = HbyA + std::size_t(i) * nC;
        s += m.bS[std::size_t(i) * nB + std::size_t(b)] * amiLerp(w, h[P], amiNbr(m, a, h));
    }
    phiHB[b] = s;
}

// fvm::laplacian(rA, p) on the coupled faces: internalCoeffs pIc, boundaryCoeffs
// acP (the source pBc 0) when first, and the faceFluxCorrection ffcB. rA: the
// CELL field (interpolated: lerp(pnf, pif, w), 0 on a face without overlap), or
// null = CorrectPhi's dimensioned rAUf 1, a SURFACE field: 1 on every face,
// a face without overlap included (pcorr's neighbour value is 0 there: it
// carries flux).
template <class R, bool Lim>
__global__ void kAmiPBnd(Mesh<R> m, Ctl<R> c, const R* rA, const R* p, const R* gp, int first, R* pIc, R* acP,
                         R* pBc, R* ffcB)
{
#pragma clang fp contract(off)
    const int a = int(blockIdx.x) * kT + int(threadIdx.x);
    if (a >= m.nA) return;
    const std::size_t nC = std::size_t(m.nC), nA = std::size_t(m.nA);
    const int b = m.aB[a], P = m.bOwn[b];
    const R w = m.aWt[a], dc = m.bDc[b];
    const R gMagS = (rA ? amiLerp(w, rA[P], amiNbr(m, a, rA)) : R(1.0)) * m.bMagS[b];
    if (first) {
        pIc[b] = gMagS * (-dc);
        acP[b] = -gMagS * dc;
        pBc[b] = R(0.0);
    }
    R corr = 0;
    if (c.lapCorr) {
        R kc = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const R* g = gp + std::size_t(i) * nC;
            kc += m.aK[std::size_t(i) * nA + std::size_t(a)] * amiLerp(w, g[P], amiNbr(m, a, g));
        }
        if constexpr (Lim) corr = gMagS * (snLimiter(c.lapLim, fabs(dc * (amiNbr(m, a, p) - p[P])), fabs(kc)) * kc);
        else corr = gMagS * kc;
    }
    ffcB[b] = corr;
}

// phi_b = phiHbyA_b - pEqn.flux()_b: internalCoeffs p_P - boundaryCoeffs pnf + faceFluxCorrection.
template <class R>
__global__ void kAmiPhiB(Mesh<R> m, const R* fHB, const R* pIc, const R* acP, const R* ffcB, const R* p, R* out)
{
#pragma clang fp contract(off)
    const int a = int(blockIdx.x) * kT + int(threadIdx.x);
    if (a >= m.nA) return;
    const int b = m.aB[a], P = m.bOwn[b];
    out[b] = fHB[b] - ((pIc[b] * p[P] - acP[b] * amiNbr(m, a, p)) + ffcB[b]);
}

} // namespace

template <class R>
std::expected<void, std::string> SimpleImpl<R>::amiMake(const FvSimpleSetup& s)
{
    // Called again after a mesh motion: the address lists change length, so
    // every buffer is uploaded anew (small: the interface's faces only).
    const FvAmiSetup& A = s.ami;
    const bool fresh = !amiD;
    if (fresh) amiD = std::make_unique<AmiDev>();
    AmiDev& D = *amiD;
    const std::size_t nB = std::size_t(s.nB);
    (void)hipDeviceSynchronize(); // no kernel may still read the buffers replaced below
#define FV_PUT(dst, src)                                                                                            \
    {                                                                                                               \
        auto t_ = upload(src, "ami " #src);                                                                         \
        if (!t_) return std::unexpected(t_.error());                                                                \
        dst = std::move(*t_);                                                                                       \
    }
#define FV_PUTR(dst, src)                                                                                           \
    {                                                                                                               \
        auto t_ = uploadAs<R>(src, "ami " #src);                                                                    \
        if (!t_) return std::unexpected(t_.error());                                                                \
        dst = std::move(*t_);                                                                                       \
    }
    FV_PUT(D.aB, A.aB) FV_PUT(D.bA, A.bA) FV_PUT(D.aStart, A.aStart) FV_PUT(D.aCell, A.aCell) FV_PUT(D.aEll, A.aEll)
    FV_PUT(D.rCell, A.rCell) FV_PUT(D.rStart, A.rStart) FV_PUT(D.rA, A.rA)
    FV_PUTR(D.aW, A.aW) FV_PUTR(D.aWt, A.wt) FV_PUTR(D.aK, A.k) FV_PUTR(D.aD, A.d)
#undef FV_PUT
#undef FV_PUTR
    if (fresh) {
        for (DeviceBuffer<R>* z : {&D.acU, &D.acP, &D.ffcB, &D.acS, &D.sfxB}) {
            auto t = zeros<R>(nB, "ami coefficients");
            if (!t) return std::unexpected(t.error());
            *z = std::move(*t);
        }
    }
    m.nA = A.nA;
    m.nR = int(A.rCell.size());
    m.aB = D.aB.data();
    m.bA = D.bA.data();
    m.aStart = D.aStart.data();
    m.aCell = D.aCell.data();
    m.aEll = D.aEll.data();
    m.aW = D.aW.data();
    m.aWt = D.aWt.data();
    m.aK = D.aK.data();
    m.aD = D.aD.data();
    m.rCell = D.rCell.data();
    m.rStart = D.rStart.data();
    m.rA = D.rA.data();
    return {};
}

template <class R>
void SimpleImpl<R>::amiEval(int nc, const R* x, R* xb)
{
    kAmiEval<R><<<grid(std::size_t(m.nA)), kT>>>(m, nc, x, xb);
}

template <class R>
void SimpleImpl<R>::amiUBnd()
{
    const unsigned gA = grid(std::size_t(m.nA));
    AmiDev& D = *amiD;
    const bool lim = c.lapLim >= 0;
    if (ras) {
        const R* nuC = ras->nuEff.data();
        if (lim) kAmiUBnd<R, true, true><<<gA, kT>>>(m, c, U.data(), G.data(), phiB.data(), nuC, ic.data(), D.acU.data(), bc.data(), bfx.data());
        else kAmiUBnd<R, false, true><<<gA, kT>>>(m, c, U.data(), G.data(), phiB.data(), nuC, ic.data(), D.acU.data(), bc.data(), bfx.data());
    } else {
        if (lim) kAmiUBnd<R, true, false><<<gA, kT>>>(m, c, U.data(), G.data(), phiB.data(), nullptr, ic.data(), D.acU.data(), bc.data(), bfx.data());
        else kAmiUBnd<R, false, false><<<gA, kT>>>(m, c, U.data(), G.data(), phiB.data(), nullptr, ic.data(), D.acU.data(), bc.data(), bfx.data());
    }
}

template <class R>
void SimpleImpl<R>::amiOff(la::BlockEll<R>& A, const R* coef, bool negate, const u8* fix)
{
    kAmiOff<R><<<grid(std::size_t(m.nR)), kT>>>(m, coef, negate ? 1 : 0, fix, A.off());
    (AU && &A == &*AU ? cplU : cplP) = AmiCpl{coef, negate, fix};
}

template <class R>
void SimpleImpl<R>::amiSumA(const la::BlockEll<R>& A, int nComp, unsigned mask, R* rowSum)
{
    const AmiCpl& k = AU && &A == &*AU ? cplU : cplP;
    if (!k.coef) return;
    kAmiSumA<R><<<grid(std::size_t(m.nR)), kT>>>(m, k.coef, k.negate ? 1 : 0, k.fix, nComp, mask, rowSum);
}

template <class R>
void SimpleImpl<R>::amiHsrc()
{
    kAmiHsrc<R><<<grid(std::size_t(m.nA)), kT>>>(m, U.data(), amiD->acU.data(), bc.data());
}

template <class R>
void SimpleImpl<R>::amiPhiH()
{
    kAmiPhiH<R><<<grid(std::size_t(m.nA)), kT>>>(m, HbyA.data(), phiHB.data());
}

template <class R>
void SimpleImpl<R>::amiPBnd(const R* rA, const R* pv, const R* gp, bool first)
{
    AmiDev& D = *amiD;
    const unsigned gA = grid(std::size_t(m.nA));
    if (c.lapLim >= 0)
        kAmiPBnd<R, true><<<gA, kT>>>(m, c, rA, pv, gp, first ? 1 : 0, pIc.data(), D.acP.data(), pBc.data(), D.ffcB.data());
    else
        kAmiPBnd<R, false><<<gA, kT>>>(m, c, rA, pv, gp, first ? 1 : 0, pIc.data(), D.acP.data(), pBc.data(), D.ffcB.data());
}

template <class R>
void SimpleImpl<R>::amiPhiB(const R* fHB, const R* pv, R* out)
{
    const AmiDev& D = *amiD;
    kAmiPhiB<R><<<grid(std::size_t(m.nA)), kT>>>(m, fHB, pIc.data(), D.acP.data(), D.ffcB.data(), pv, out);
}

#define FV_AMI_INST(R)                                                                                              \
    template std::expected<void, std::string> SimpleImpl<R>::amiMake(const FvSimpleSetup&);                        \
    template void SimpleImpl<R>::amiEval(int, const R*, R*);                                                        \
    template void SimpleImpl<R>::amiUBnd();                                                                         \
    template void SimpleImpl<R>::amiOff(la::BlockEll<R>&, const R*, bool, const u8*);                               \
    template void SimpleImpl<R>::amiSumA(const la::BlockEll<R>&, int, unsigned, R*);                                \
    template void SimpleImpl<R>::amiHsrc();                                                                         \
    template void SimpleImpl<R>::amiPhiH();                                                                         \
    template void SimpleImpl<R>::amiPBnd(const R*, const R*, const R*, bool);                                       \
    template void SimpleImpl<R>::amiPhiB(const R*, const R*, R*);
FV_AMI_INST(double)
FV_AMI_INST(float)
#undef FV_AMI_INST

} // namespace dns::gpu
