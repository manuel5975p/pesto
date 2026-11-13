// pimpleFoam on the FV path (docs/fvpimple.md): OpenFOAM v2606 pimpleFoam.C's
// time step on a STATIC mesh as a composition of the simpleFoam path's stage
// methods (gpu/fv_simple.hip: assembleU, systemU, solveU, prepP, assembleP,
// solve, fluxes, correctU; gpu/fv_ras.hip: rasCorrect) plus what a transient
// adds: fvm::ddt (Euler; kURow's Ddt instantiation, rasDdt), fvc::ddtCorr in
// phiHbyA (ddtCorr), the Final-iteration solver / relaxation entries, the
// pressure corrector loop, CourantNo.H. A simpleFoam case allocates nothing
// here and launches none of these kernels.
#include "dns/gpu/fv_simple.hpp"

#include "dns/gpu/internal/fv_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>

namespace dns::gpu {

using namespace fvd;

namespace {

// A pimpleFoam BiCGStab + multicolour-GS solve stopped by relTol > 0 iterates at
// least twice once it starts: its first iterate (the half-step exit = one MC
// symmetric sweep, alpha ~ 1) is not OpenFOAM's one SEQUENTIAL symGaussSeidel
// sweep. M17 (issues.md 65 (p)), FP64, user tolerances, vs dns relTol 0 over 20
// steps: aft thrust 3.1e-3 with one iteration, 4.9e-5 with two (pimpleFoam's
// own loose run: 1.2e-4).
constexpr int kPimpleLooseMinIter = 2;

// phiHbyA += interpolate(rAU) ddtCorr(U, phi), internal faces.
// EulerDdtScheme::fvcDdtPhiCorr: phiCorr = phi.oldTime() - (Sf & interpolate(
// U.oldTime())); ddtCorr = (ddtCouplingCoeff rDeltaT) phiCorr with
// ddtScheme::fvcDdtPhiCoeff (ddtPhiCoeff < 0): 1 - min(|phiCorr| /
// (|phi.oldTime()| + SMALL), 1).
template <class R>
__global__ void kDdtCorrF(Mesh<R> m, R rDeltaT, const R* Uold, const R* phiOld, const R* rAU, R* phiH)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nI = std::size_t(m.nI), nC = std::size_t(m.nC), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    const R l = m.w[f];
    R su = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const R uo = Uold[std::size_t(i) * nC + o], un = Uold[std::size_t(i) * nC + n];
        su += m.S[std::size_t(i) * nI + ff] * (l * (uo - un) + un);
    }
    const R ph = phiOld[f];
    const R phiCorr = ph - su;
    const R coef = R(1.0) - fmin(fabs(phiCorr) / (fabs(ph) + R(kSmall)), R(1.0));
    phiH[f] += (l * (rAU[o] - rAU[n]) + rAU[n]) * ((coef * rDeltaT) * phiCorr);
}

// Boundary faces: the coupling coefficient is 0 where U fixesValue()
// (fixedValue and the mixed types); elsewhere (zeroGradient, slip, symmetry)
// the same expression on the patch values, rAU's patch value the cell's
// (extrapolatedCalculated).
template <class R>
__global__ void kDdtCorrB(Mesh<R> m, R rDeltaT, const R* UbOld, const R* phiBOld, const R* rAU, R* phiHB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const u8 code = m.bU[b];
    if (code == 0 || code == 2 || code == 4) return;
    const std::size_t nB = std::size_t(m.nB), bb = std::size_t(b);
    R su = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) su += m.bS[std::size_t(i) * nB + bb] * UbOld[std::size_t(i) * nB + bb];
    const R ph = phiBOld[b];
    const R phiCorr = ph - su;
    const R coef = R(1.0) - fmin(fabs(phiCorr) / (fabs(ph) + R(kSmall)), R(1.0));
    phiHB[b] += rAU[m.bOwn[b]] * ((coef * rDeltaT) * phiCorr);
}

// Dynamic mesh, EulerDdtScheme::fvcDdtUfCorr: phiUf0 = Sf & Uf.oldTime() on the
// CURRENT Sf, phiCorr = phiUf0 - dotInterpolate(Sf, U.oldTime()), the coupling
// coefficient fvcDdtPhiCoeff(U.oldTime(), phiUf0, phiCorr).
template <class R>
__global__ void kDdtCorrUfF(Mesh<R> m, R rDeltaT, const R* Uold, const R* UfOld, const R* rAU, R* phiH)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nI = std::size_t(m.nI), nC = std::size_t(m.nC), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    const R l = m.w[f];
    R su = 0, ph = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const R uo = Uold[std::size_t(i) * nC + o], un = Uold[std::size_t(i) * nC + n];
        const R S = m.S[std::size_t(i) * nI + ff];
        su += S * (l * (uo - un) + un);
        ph += S * UfOld[std::size_t(i) * nI + ff];
    }
    const R phiCorr = ph - su;
    const R coef = R(1.0) - fmin(fabs(phiCorr) / (fabs(ph) + R(kSmall)), R(1.0));
    phiH[f] += (l * (rAU[o] - rAU[n]) + rAU[n]) * ((coef * rDeltaT) * phiCorr);
}

template <class R>
__global__ void kDdtCorrUfB(Mesh<R> m, R rDeltaT, const R* UbOld, const R* UfBOld, const R* rAU, R* phiHB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const u8 code = m.bU[b];
    if (code == 0 || code == 2 || code == 4) return; // fixesValue(): ddtCouplingCoeff 0
    const std::size_t nB = std::size_t(m.nB), bb = std::size_t(b);
    R su = 0, ph = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const R S = m.bS[std::size_t(i) * nB + bb];
        su += S * UbOld[std::size_t(i) * nB + bb];
        ph += S * UfBOld[std::size_t(i) * nB + bb];
    }
    const R phiCorr = ph - su;
    const R coef = R(1.0) - fmin(fabs(phiCorr) / (fabs(ph) + R(kSmall)), R(1.0));
    phiHB[b] += rAU[m.bOwn[b]] * ((coef * rDeltaT) * phiCorr);
}

// Uf = fvc::interpolate(U) [+ fvc::correctUf: n (phi / |S| - n & Uf), n = Sf / |S|,
// phi the ABSOLUTE flux]. Internal faces.
template <class R, bool Correct>
__global__ void kUfF(Mesh<R> m, const R* U, const R* phi, R* Uf)
{
    const int f = int(blockIdx.x) * kT + int(threadIdx.x);
    if (f >= m.nI) return;
    const std::size_t nI = std::size_t(m.nI), nC = std::size_t(m.nC), ff = std::size_t(f);
    const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
    const R l = m.w[f];
    R uf[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const R uo = U[std::size_t(i) * nC + o], un = U[std::size_t(i) * nC + n];
        uf[i] = l * (uo - un) + un;
    }
    if constexpr (Correct) {
        const R magS = m.magS[f];
        R nv[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) nv[i] = m.S[std::size_t(i) * nI + ff] / magS;
        const R d = phi[f] / magS - (nv[0] * uf[0] + nv[1] * uf[1] + nv[2] * uf[2]);
#pragma unroll
        for (int i = 0; i < 3; ++i) uf[i] += nv[i] * d;
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) Uf[std::size_t(i) * nI + ff] = uf[i];
}

template <class R, bool Correct>
__global__ void kUfB(Mesh<R> m, const R* Ub, const R* phiB, R* UfB)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= m.nB) return;
    const std::size_t nB = std::size_t(m.nB), bb = std::size_t(b);
    R uf[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) uf[i] = Ub[std::size_t(i) * nB + bb];
    if constexpr (Correct) {
        R nv[3];
        bndN(m, b, nv); // Sf / magSf
        const R d = phiB[b] / m.bMagS[b] - (nv[0] * uf[0] + nv[1] * uf[1] + nv[2] * uf[2]);
#pragma unroll
        for (int i = 0; i < 3; ++i) uf[i] += nv[i] * d;
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) UfB[std::size_t(i) * nB + bb] = uf[i];
}

// out = a - b | a + b (fvc::makeRelative / makeAbsolute / absolute; out may be a).
template <class R, bool Minus>
__global__ void kFluxShift(int n, const R* a, const R* b, R* out)
{
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    if (i >= n) return;
    out[i] = Minus ? a[i] - b[i] : a[i] + b[i];
}

// movingWallVelocity: the patch value is this step's Uwall().
template <class R>
__global__ void kSetWall(int nB, const u8* bMove, const R* uWall, R* Ub)
{
    const int b = int(blockIdx.x) * kT + int(threadIdx.x);
    if (b >= nB || !bMove[b]) return;
    const std::size_t n = std::size_t(nB), bb = std::size_t(b);
#pragma unroll
    for (int i = 0; i < 3; ++i) Ub[std::size_t(i) * n + bb] = uWall[std::size_t(i) * n + bb];
}

// fvm::ddt(x), Euler, onto a scalar equation's diagonal / source additions
// (V0: mesh.Vsc0(), m.V itself on a static mesh).
template <class R>
__global__ void kSDdt(Mesh<R> m, R rDeltaT, const R* old, const R* V0, R* dAdd, R* sAdd)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const R V = m.V[c];
    dAdd[c] += rDeltaT * V;
    sAdd[c] += (rDeltaT * old[c]) * V0[c];
}

// CourantNo.H: sumPhi = fvc::surfaceSum(mag(phi)) (a cell's internal faces,
// then its boundary faces) and sumPhi / V.
template <class R>
__global__ void kCoCell(Mesh<R> m, const R* phi, const R* phiB, R* sumPhi, R* ratio)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    R s = 0;
    for (int e = m.cStart[c]; e < m.cStart[c + 1]; ++e) {
        const int code = m.cEnt[e];
        s += code >= 0 ? fabs(phi[code >> 1]) : fabs(phiB[-code - 1]);
    }
    sumPhi[c] = s;
    ratio[c] = s / m.V[c];
}

// Block maxima of x >= 0 (gMax: exact in any order).
template <class R>
__global__ void kMaxPart(int n, const R* x, R* part)
{
    __shared__ R sh[kT];
    const int i = int(blockIdx.x) * kT + int(threadIdx.x);
    sh[threadIdx.x] = i < n ? x[i] : R(0.0);
    __syncthreads();
    for (int s = kT / 2; s > 0; s >>= 1) {
        if (int(threadIdx.x) < s) sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}

template <class R>
hipError_t copyDev(DeviceBuffer<R>& dst, const DeviceBuffer<R>& src, std::size_t n)
{
    return hipMemcpyAsync(dst.data(), src.data(), n * sizeof(R), hipMemcpyDeviceToDevice, 0);
}

// A host FP64 vector into an existing device buffer of R (R = float: each value rounded once).
template <class R>
std::expected<void, std::string> reup(DeviceBuffer<R>& d, const std::vector<double>& v, const char* what)
{
    if (v.empty()) return {};
    hipError_t e;
    if constexpr (std::is_same_v<R, double>) e = hipMemcpy(d.data(), v.data(), v.size() * sizeof(double), hipMemcpyHostToDevice);
    else {
        std::vector<R> r(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) r[i] = R(v[i]);
        e = hipMemcpy(d.data(), r.data(), r.size() * sizeof(R), hipMemcpyHostToDevice);
    }
    if (e != hipSuccess) return std::unexpected(std::format("fv: mesh update: upload ({}): {}", what, hipGetErrorString(e)));
    return {};
}

} // namespace

template <class R>
std::expected<void, std::string> SimpleImpl<R>::pimpleMake(const FvSimpleSetup& s)
{
    pim = std::make_unique<PimpleState<R>>();
    PimpleState<R>& P = *pim;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
#define FV_Z(dst, n)                                                                                                \
    {                                                                                                               \
        auto t_ = zeros<R>(n, #dst);                                                                                \
        if (!t_) return std::unexpected(t_.error());                                                                \
        dst = std::move(*t_);                                                                                       \
    }
    FV_Z(P.Uold, 3 * nC) FV_Z(P.UbOld, 3 * nB) FV_Z(P.phiOld, nI) FV_Z(P.phiBOld, nB) FV_Z(P.coPart, grid(nC) + 1)
    if (ras)
        for (int q = 0; q < ras->nS; ++q) FV_Z(P.sOld[std::size_t(q)], nC)
    if (fc.motion.on) { // mesh.dynamic()
        P.mov = std::make_unique<MotionState<R>>();
        MotionState<R>& M = *P.mov;
        FV_Z(M.meshPhi, nI) FV_Z(M.meshPhiB, nB) FV_Z(M.Uf, 3 * nI) FV_Z(M.UfB, 3 * nB) FV_Z(M.UfOld, 3 * nI)
        FV_Z(M.UfBOld, 3 * nB) FV_Z(M.V0, nC) FV_Z(M.phiAbs, nI) FV_Z(M.phiAbsB, nB) FV_Z(M.pc, nC) FV_Z(M.pcB, nB)
        FV_Z(M.zeroB, nB) FV_Z(M.uWall, 3 * nB)
        // pcorrTypes: fixedValue where p fixesValue() (fixedValue and the mixed freestreamPressure), else zeroGradient
        std::vector<u8> bMove(nB, 0), bPc(nB, 1);
        for (std::size_t b = 0; b < nB; ++b) {
            bMove[b] = fc.patches[std::size_t(s.bPatch[b])].movingWall;
            bPc[b] = s.bP[b] == u8(FvPCode::ZeroGrad) ? 1 : 0;
        }
        auto um = upload(bMove, "bMove");
        if (!um) return std::unexpected(um.error());
        M.bMove = std::move(*um);
        auto up2 = upload(bPc, "bPc");
        if (!up2) return std::unexpected(up2.error());
        M.bPc = std::move(*up2);
        for (std::size_t cc = 0; cc < nC; ++cc)
            if (s.srcCell[cc] == 0) M.pcRefCell = int(cc);
        M.limPc = fc.schemes.gradPcorr.kind == FvGrad::CellLimited;
        M.kPc = R(fc.schemes.gradPcorr.k);
        M.correctPhi = fc.pimple.correctPhi;
        M.checkMeshCo = fc.pimple.checkMeshCourantNo;
        M.solverPcorr = fc.pimple.solverPcorr;
        M.solverPcorrFinal = fc.pimple.solverPcorrFinal;
        (void)copyDev(M.V0, V, nC);
        // createUfIfPresent.H: Uf = fvc::interpolate(U)
        kUfF<R, false><<<grid(nI), kT>>>(m, U.data(), nullptr, M.Uf.data());
        kUfB<R, false><<<grid(nB), kT>>>(m, Ub.data(), nullptr, M.UfB.data());
    }
#undef FV_Z
    const FvPimpleCtl& pc = fc.pimple;
    P.solverU = fc.solverU;
    P.solverUFinal = pc.solverUFinal;
    P.solverP = fc.solverP;
    P.solverPFinal = pc.solverPFinal;
    P.relaxU = fc.relaxU;
    P.relaxUFinal = pc.relaxUFinal;
    if (fc.relaxPGiven) P.relaxP = fc.relaxP;
    P.relaxPFinal = pc.relaxPFinal;
    if (fc.ras.model == FvTurbModel::KOmegaSST) {
        P.solverS = {fc.ras.solverK, fc.ras.solverOmega};
        P.solverSFinal = {pc.solverKFinal, pc.solverOmegaFinal};
        P.relaxS = {fc.ras.relaxK, fc.ras.relaxOmega};
        P.relaxSFinal = {pc.relaxKFinal, pc.relaxOmegaFinal};
    } else if (fc.ras.model == FvTurbModel::SpalartAllmaras) {
        P.solverS[0] = fc.ras.solverNuTilda;
        P.solverSFinal[0] = pc.solverNuTildaFinal;
        P.relaxS[0] = fc.ras.relaxNuTilda;
        P.relaxSFinal[0] = pc.relaxNuTildaFinal;
    }
    return {};
}

template <class R>
void SimpleImpl<R>::ddtCorr()
{
    const PimpleState<R>& P = *pim;
    if (const MotionState<R>* M = P.mov.get()) { // mesh.dynamic(): fvc::ddtCorr(U, Uf)
        kDdtCorrUfF<R><<<grid(std::size_t(m.nI)), kT>>>(m, c.rDeltaT, P.Uold.data(), M->UfOld.data(), rAU.data(),
                                                         phiH.data());
        kDdtCorrUfB<R><<<grid(std::size_t(m.nB)), kT>>>(m, c.rDeltaT, P.UbOld.data(), M->UfBOld.data(), rAU.data(),
                                                         phiHB.data());
        return;
    }
    kDdtCorrF<R><<<grid(std::size_t(m.nI)), kT>>>(m, c.rDeltaT, P.Uold.data(), P.phiOld.data(), rAU.data(), phiH.data());
    kDdtCorrB<R><<<grid(std::size_t(m.nB)), kT>>>(m, c.rDeltaT, P.UbOld.data(), P.phiBOld.data(), rAU.data(),
                                                   phiHB.data());
}

template <class R>
void SimpleImpl<R>::rasDdt(int q)
{
    kSDdt<R><<<grid(std::size_t(m.nC)), kT>>>(m, c.rDeltaT, pim->sOld[std::size_t(q)].data(),
                                               pim->mov ? pim->mov->V0.data() : m.V, ras->dAdd.data(),
                                               ras->sAdd.data());
}

template <class R>
void SimpleImpl<R>::relFlux(R* fI, R* fB, bool relative)
{
    const MotionState<R>& M = *pim->mov;
    const std::size_t nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    if (relative) {
        kFluxShift<R, true><<<grid(nI), kT>>>(int(nI), fI, M.meshPhi.data(), fI);
        kFluxShift<R, true><<<grid(nB), kT>>>(int(nB), fB, M.meshPhiB.data(), fB);
    } else {
        kFluxShift<R, false><<<grid(nI), kT>>>(int(nI), fI, M.meshPhi.data(), fI);
        kFluxShift<R, false><<<grid(nB), kT>>>(int(nB), fB, M.meshPhiB.data(), fB);
    }
}

template <class R>
void SimpleImpl<R>::correctUf()
{
    MotionState<R>& M = *pim->mov;
    kUfF<R, true><<<grid(std::size_t(m.nI)), kT>>>(m, U.data(), phi.data(), M.Uf.data());
    kUfB<R, true><<<grid(std::size_t(m.nB)), kT>>>(m, Ub.data(), phiB.data(), M.UfB.data());
}

template <class R>
std::expected<void, std::string> SimpleImpl<R>::meshUpdate(const FvMovedMesh& mv)
{
    MotionState<R>& M = *pim->mov;
    const FvSimpleSetup& s = *mv.s;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    if (std::size_t(s.nC) != nC || std::size_t(s.nI) != nI || std::size_t(s.nB) != nB || mv.mv->meshPhi.size() != nI ||
        mv.mv->meshPhiB.size() != nB || mv.mv->uWall.size() != 3 * nB)
        return std::unexpected("fv: mesh update: the moved setup is not this mesh's");
    // fvMesh::movePoints: storeOldVol(V()), then the geometry on the new points
    (void)copyDev(M.V0, V, nC);
    (void)hipDeviceSynchronize();
#define FV_RE(dst, src)                                                                                             \
    if (auto r_ = reup(dst, src, #src); !r_) return r_;
    FV_RE(S, s.S) FV_RE(rO, s.rO) FV_RE(rN, s.rN) FV_RE(k, s.k) FV_RE(magS, s.magS) FV_RE(w, s.w) FV_RE(dc, s.dc)
    FV_RE(bS, s.bS) FV_RE(bN, s.bN) FV_RE(bCf, s.bCf) FV_RE(bMagS, s.bMagS) FV_RE(bDc, s.bDc) FV_RE(V, s.V) FV_RE(C, s.C)
    if (ras) { // wallDist::movePoints (updateInterval 1), nearWallDist::correct
        FV_RE(ras->y, s.ras.y) FV_RE(ras->yWall, s.ras.yWall) FV_RE(ras->dlt, s.ras.dlt)
    }
    FV_RE(M.meshPhi, mv.mv->meshPhi) FV_RE(M.meshPhiB, mv.mv->meshPhiB) FV_RE(M.uWall, mv.mv->uWall)
#undef FV_RE
    if (mv.g) { // cyclicAMI: the AMI was recomputed on the moved points -- new weights, new couplings, new pattern
        if (s.cellEll.size() != s.cellEnt.size()) return std::unexpected("fv: mesh update: the moved setup's ELL map");
        if (auto e = hipMemcpy(cEll.data(), s.cellEll.data(), s.cellEll.size() * sizeof(int), hipMemcpyHostToDevice);
            e != hipSuccess)
            return std::unexpected(std::string("fv: mesh update: upload (cellEll): ") + hipGetErrorString(e));
        if (auto r = amiMake(s); !r) return r;
        if (auto r = makeMatrices(s, *mv.g); !r) return r;
    }
    sumV = 0;
    for (const double v : s.V) sumV += v;
    gValid = gPValid = false;
    // dynamic*FvMesh::update(): U.correctBoundaryConditions() -- movingWallVelocity::updateCoeffs
    kSetWall<R><<<grid(nB), kT>>>(int(nB), M.bMove.data(), M.uWall.data(), Ub.data());
    evalU(1);
    return {};
}

template <class R>
std::expected<FvCourant, std::string> SimpleImpl<R>::courant(double deltaT)
{
    if (!pim) return std::unexpected("fv: courant(): not a pimpleFoam case");
    return courantOf(phi.data(), phiB.data(), deltaT);
}

template <class R>
std::expected<FvCourant, std::string> SimpleImpl<R>::courantOf(const R* f, const R* fB, double deltaT)
{
    const std::size_t nC = std::size_t(m.nC);
    const unsigned nb = grid(nC);
    R* sumPhi = tmp.data();
    R* ratio = sumPhi + nC;
    kCoCell<R><<<nb, kT>>>(m, f, fB, sumPhi, ratio);
    kMaxPart<R><<<nb, kT>>>(int(nC), ratio, pim->coPart.data());
    std::vector<R> h(nb);
    if (auto e = hipMemcpy(h.data(), pim->coPart.data(), nb * sizeof(R), hipMemcpyDeviceToHost); e != hipSuccess)
        return std::unexpected(std::string("fv: Courant number: ") + hipGetErrorString(e));
    const R* a[1] = {sumPhi};
    const bool sgn[1] = {false};
    auto s = sums(nC, a, sgn);
    if (!s) return std::unexpected(s.error());
    const double mx = double(*std::ranges::max_element(h));
    if (!std::isfinite(mx) || !std::isfinite((*s)[0])) return std::unexpected("fv: non-finite Courant number");
    return FvCourant{0.5 * ((*s)[0] / sumV) * deltaT, 0.5 * mx * deltaT};
}

template <class R>
std::expected<FvStepLog, std::string> SimpleImpl<R>::timeStep(double deltaT, const FvMovedMesh* moved)
{
    if (!pim) return std::unexpected("fv: timeStep(): not a pimpleFoam case");
    if (!(deltaT > 0)) return std::unexpected("fv: timeStep(): deltaT must be > 0");
    SimpleImpl<R>& I = *this;
    PimpleState<R>& P = *pim;
    MotionState<R>* const M = P.mov.get();
    if (M && !(moved && moved->s && moved->mv))
        return std::unexpected("fv: timeStep(): a moving-mesh case needs the step's mesh update");
    if (!M && moved) return std::unexpected("fv: timeStep(): a mesh update on a static-mesh case");
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    ++iter;
    // ++runTime: the old-time fields are the state now. Moving mesh, FIRST step:
    // U.oldTime() is first asked for by fvm::ddt(U), after mesh.update() rewrote
    // the patch values (GeometricField::oldTime() copies the CURRENT field when
    // no old-time field exists yet); from the second step on storeOldTimes()
    // runs at the head of mesh.update()'s U.correctBoundaryConditions().
    const bool oldAfterUpdate = M && M->steps == 0;
    if (!oldAfterUpdate) {
        (void)copyDev(P.Uold, U, 3 * nC);
        (void)copyDev(P.UbOld, Ub, 3 * nB);
    }
    if (M) { // Uf.oldTime() (fvc::ddtCorr(U, Uf)); phi.oldTime() is not read
        (void)copyDev(M->UfOld, M->Uf, 3 * nI);
        (void)copyDev(M->UfBOld, M->UfB, 3 * nB);
        ++M->steps;
    } else {
        (void)copyDev(P.phiOld, phi, nI);
        (void)copyDev(P.phiBOld, phiB, nB);
    }
    if (ras)
        for (int q = 0; q < ras->nS; ++q) (void)copyDev(P.sOld[std::size_t(q)], ras->s[std::size_t(q)], nC);
    c.rDeltaT = R(1.0 / deltaT);
    const FvPimpleCtl& pc = fc.pimple;
    const int nOuter = pc.nOuter, nCorr = pc.nCorr, nNonOrth = fc.simple.nNonOrthCorr;
    const bool sst = fc.ras.model == FvTurbModel::KOmegaSST;
    r0First = false;
    looseMinIter = kPimpleLooseMinIter;
    FvStepLog out;
    for (int corr = 1; corr <= nOuter; ++corr) { // while (pimple.loop())
        const bool fin = corr == nOuter;         // mesh.data().isFinalIteration()
        FvIterLog log;
        log.iter = iter;
        // solutionControl::storePrevIterFields: p when relaxField("p") is found
        if (P.relaxP) {
            (void)copyDev(pPrev, p, nC);
            (void)copyDev(pbPrev, pb, nB);
        }
        // this corrector's entries: U | UFinal, p.relax()'s p | pFinal, k | kFinal ...
        fc.solverU = fin ? P.solverUFinal : P.solverU;
        const std::optional<double>& ru = fin ? P.relaxUFinal : P.relaxU;
        c.relaxU = R(ru ? *ru : 0.0);
        const std::optional<double>& rp = fin ? P.relaxPFinal : P.relaxP;
        relaxPGiven = rp.has_value();
        relaxP = rp ? *rp : 1.0;
        if (ras) {
            FvRas& co = ras->co;
            const auto& sv = fin ? P.solverSFinal : P.solverS;
            const auto& rl = fin ? P.relaxSFinal : P.relaxS;
            if (sst) {
                co.solverK = sv[0];
                co.solverOmega = sv[1];
                co.relaxK = rl[0];
                co.relaxOmega = rl[1];
            } else {
                co.solverNuTilda = sv[0];
                co.relaxNuTilda = rl[0];
            }
        }
        if (M && corr == 1) { // pimple.firstIter(): mesh.controlledUpdate()
            const auto tm = std::chrono::steady_clock::now();
            if (auto r = I.meshUpdate(*moved); !r) return std::unexpected(r.error());
            (void)hipDeviceSynchronize(); // timing only: the device's share of the mesh update (uploads, pattern, AMG)
            log.msSetup = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tm).count();
            if (oldAfterUpdate) {
                (void)copyDev(P.Uold, U, 3 * nC);
                (void)copyDev(P.UbOld, Ub, 3 * nB);
            }
            if (M->correctPhi) { // mesh.changing(): phi = Sf & Uf, correctPhi.H, fvc::makeRelative(phi, U)
                if (auto r = I.correctPhi(deltaT, log); !r) return std::unexpected(r.error());
                I.relFlux(I.phi.data(), I.phiB.data(), true);
            }
            if (M->checkMeshCo) { // meshCourantNo.H
                auto co = I.courantOf(M->meshPhi.data(), M->meshPhiB.data(), deltaT);
                if (!co) return std::unexpected(co.error());
                log.meshCo = std::array<double, 2>{co->mean, co->max};
            }
        }
        // --- UEqn.H ---
        I.assembleU();
        const unsigned mask = I.uMask();
        I.systemU(mask);
        if (auto r = I.solveU(mask, log); !r) return std::unexpected(r.error());
        I.evalU(0);
        // --- while (pimple.correct()): pEqn.H ---
        for (int piso = 1; piso <= nCorr; ++piso) {
            if (auto r = I.prepP(); !r) return std::unexpected(r.error());
            bool amgFresh = false;
            std::string amgError;
            const la::LinOp<R> Mp = [&](const R* r, R* z) {
                if (!amgFresh) {
                    amgFresh = true;
                    if (auto f = I.amg->refresh(); !f) amgError = f.error();
                }
                I.amg->apply(r, z);
            };
            for (int nc = 0; nc <= nNonOrth; ++nc) {
                I.assembleP(nc);
                const bool finalInner = piso == nCorr && nc == nNonOrth; // pimple.finalInnerIter()
                // cyclicAMI: the matrix is not symmetric (each side's own weights and coefficients). The
                // FLEXIBLE PCG still converges on it (measured to 1e-13, 86k cells: 70-86 iterations);
                // BiCGStab around the scaled V-cycle (a variable preconditioner) diverged there.
                auto r = I.solve(*I.AP, Mp, true, finalInner ? P.solverPFinal : P.solverP, I.floor.p, I.p.data(),
                                 nullptr, true, nc > 0);
                if (!amgError.empty()) return std::unexpected("fv: p AMG: " + amgError);
                if (!r) return std::unexpected("fv: p: " + r.error());
                log.p.push_back(*r);
                I.evalP();
            }
            if (auto r = I.fluxes(log); !r) return std::unexpected(r.error());
            log.contLocal *= deltaT; // continuityErrs.H: runTime.deltaTValue() x the volume-weighted averages
            log.contGlobal *= deltaT;
            log.cont.push_back({log.contLocal, log.contGlobal});
            I.correctU();
            if (M) { // fvc::correctUf(Uf, U, phi) on the absolute flux, then fvc::makeRelative(phi, U)
                I.correctUf();
                I.relFlux(I.phi.data(), I.phiB.data(), true);
            }
        }
        if (I.ras && (fin || !pc.turbOnFinalIterOnly)) { // pimple.turbCorr()
            if (M) { // kOmegaSST: divU = fvc::div(fvc::absolute(phi, U))
                kFluxShift<R, false><<<grid(nI), kT>>>(int(nI), phi.data(), M->meshPhi.data(), M->phiAbs.data());
                kFluxShift<R, false><<<grid(nB), kT>>>(int(nB), phiB.data(), M->meshPhiB.data(), M->phiAbsB.data());
            }
            if (auto r = rasCorrect(I, log); !r) return std::unexpected(r.error());
        }
        if (auto e = hipGetLastError(); e != hipSuccess)
            return std::unexpected(std::string("fv: time step: ") + hipGetErrorString(e));
        for (const auto& u : log.U)
            if (!std::isfinite(u.initial) || !std::isfinite(u.final)) return std::unexpected("fv: non-finite U residual");
        for (const auto& q : log.p)
            if (!std::isfinite(q.initial) || !std::isfinite(q.final)) return std::unexpected("fv: non-finite p residual");
        if (!std::isfinite(log.contLocal)) return std::unexpected("fv: non-finite continuity error");
        out.outer.push_back(std::move(log));
    }
    return out;
}

template std::expected<void, std::string> SimpleImpl<double>::pimpleMake(const FvSimpleSetup&);
template std::expected<void, std::string> SimpleImpl<float>::pimpleMake(const FvSimpleSetup&);
template void SimpleImpl<double>::relFlux(double*, double*, bool);
template void SimpleImpl<float>::relFlux(float*, float*, bool);
template void SimpleImpl<double>::correctUf();
template void SimpleImpl<float>::correctUf();
template std::expected<void, std::string> SimpleImpl<double>::meshUpdate(const FvMovedMesh&);
template std::expected<void, std::string> SimpleImpl<float>::meshUpdate(const FvMovedMesh&);
template std::expected<FvCourant, std::string> SimpleImpl<double>::courantOf(const double*, const double*, double);
template std::expected<FvCourant, std::string> SimpleImpl<float>::courantOf(const float*, const float*, double);
template void SimpleImpl<double>::ddtCorr();
template void SimpleImpl<float>::ddtCorr();
template void SimpleImpl<double>::rasDdt(int);
template void SimpleImpl<float>::rasDdt(int);
template std::expected<FvCourant, std::string> SimpleImpl<double>::courant(double);
template std::expected<FvCourant, std::string> SimpleImpl<float>::courant(double);
template std::expected<FvStepLog, std::string> SimpleImpl<double>::timeStep(double, const FvMovedMesh*);
template std::expected<FvStepLog, std::string> SimpleImpl<float>::timeStep(double, const FvMovedMesh*);

} // namespace dns::gpu
