// Internal to the incompressible FV solver (gpu/fv_simple.hip, gpu/fv_ras.hip):
// the device mesh view, the shared device helpers and gradient kernel, the
// DeviceSimple state (SimpleImpl<R>) and the RAS state (phase BL), all on the
// arithmetic type R in {double, float} (docs/fv.md "Precision"; R = double is
// the FP64 solver bit for bit). GPU-only header.
#pragma once

#include "dns/fvcase.hpp"
#include "dns/fvsimple.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/fv_simple.hpp"
#include "dns/gpu/la/amg.hpp"
#include "dns/gpu/la/blockell.hpp"
#include "dns/gpu/la/internal/dots.hpp"
#include "dns/gpu/la/krylov.hpp"
#include "dns/gpu/la/smooth.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace dns::gpu {

namespace fvd {

using u8 = std::uint8_t;
constexpr int kT = 256;
inline unsigned grid(std::size_t n) { return unsigned((n + kT - 1) / kT); }
constexpr double kSmall = 1e-15;  // OpenFOAM SMALL (cellLimited)
constexpr double kTwoThirds = 2.0 / 3.0;

enum : int { kDivLinear = 0, kDivLUD = 1, kDivUpwind = 2 };

// Mesh view (device pointers).
template <class R>
struct Mesh
{
    int nC = 0, nI = 0, nB = 0;
    const int *own = nullptr, *nei = nullptr;
    const R *S = nullptr, *rO = nullptr, *rN = nullptr, *k = nullptr, *magS = nullptr, *w = nullptr,
                 *dc = nullptr;
    const int* bOwn = nullptr;
    const R *bS = nullptr, *bN = nullptr, *bCf = nullptr, *bMagS = nullptr, *bDc = nullptr;
    const u8 *bU = nullptr, *bP = nullptr, *bAssign = nullptr, *bSym = nullptr, *bAdjust = nullptr, *bMrf = nullptr,
             *bForce = nullptr;
    const R *bRefU = nullptr, *bRefP = nullptr, *bFrame = nullptr, *bUmrf = nullptr, *frame = nullptr,
                 *frame2 = nullptr;
    const R *V = nullptr, *C = nullptr;
    const int *cStart = nullptr, *cEnt = nullptr, *cEll = nullptr;
    const u8* zone = nullptr;
    // cyclicAMI (FvAmiSetup; nA == 0: none, the pointers null and unread)
    int nA = 0, nR = 0;
    const int *aB = nullptr, *bA = nullptr, *aStart = nullptr, *aCell = nullptr, *aEll = nullptr;
    const R *aW = nullptr, *aWt = nullptr, *aK = nullptr, *aD = nullptr;
    const int *rCell = nullptr, *rStart = nullptr, *rA = nullptr;
};

// cyclicAMI: patchNeighbourField() of the cell field x at AMI face a -- sum_j
// w_j x[cell_j] from zero in OpenFOAM's address order, each product its own
// rounding (AMIInterpolation::interpolateToSource / ToTarget).
template <class R>
__device__ inline R amiNbr(const Mesh<R>& m, int a, const R* x)
{
#pragma clang fp contract(off)
    R s = 0;
    for (int j = m.aStart[a]; j < m.aStart[a + 1]; ++j) s += m.aW[j] * x[m.aCell[j]];
    return s;
}

// A coupled patch's face value: lerp(patchNeighbourField, patchInternalField, weights).
template <class R>
__device__ inline R amiLerp(R w, R pif, R pnf)
{
#pragma clang fp contract(off)
    return (R(1.0) - w) * pnf + w * pif;
}

template <class R>
struct Ctl
{
    R nu = 0;
    int div = kDivLinear;
    bool bounded = false, lapCorr = true, snCorr = true;
    bool limU = false, limP = false;
    R kU = 1, kP = 1;
    // limitedSnGrad coefficients (< 0: not limited): the laplacians' and fvc::snGrad's.
    R lapLim = -1, snLim = -1;
    bool mrf = false, mrfTwo = false;
    R om[FvSimpleSetup::kMaxMrfZones][3] = {}; // Omega per zone (zone[c] - 1)
    bool consistent = false;
    R relaxU = 0; // 0: no equation relaxation
    int emptyAxis = -1;
    bool force = false; // fvOptions: f per unit mass on the U source
    R f[3] = {0, 0, 0};
    R rDeltaT = 0;      // pimpleFoam: 1 / deltaT of the current time step (Euler ddt); simpleFoam: unused
};

// --- small device helpers ---------------------------------------------------

template <class R>
__device__ inline R cmptMax3(R a, R b, R c) { return fmax(fmax(a, b), c); }
template <class R>
__device__ inline R cmptMin3(R a, R b, R c) { return fmin(fmin(a, b), c); }

// G at cell c: G[i][q] = d_i u_q, layout (3 i + q) nC + c.
template <class R>
__device__ inline void loadG(const R* G, int nC, int c, R (&g)[3][3])
{
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int q = 0; q < 3; ++q) g[i][q] = G[std::size_t(3 * i + q) * std::size_t(nC) + std::size_t(c)];
}

// X = nu dev2(T(g)): X_ij = nu (g_ji - (2/3 tr g) delta_ij).
template <class R>
__device__ inline void nuDev2T(const R (&g)[3][3], R nu, R (&X)[3][3])
{
    const R sph = R(kTwoThirds) * (g[0][0] + g[1][1] + g[2][2]);
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) X[i][j] = nu * (g[j][i] - (i == j ? sph : R(0.0)));
}

// limitedSnGrad's face limiter min(psi |sn| / ((1 - psi) |corr| + SMALL), 1):
// magSn = |dc (x_N - x_P)|, magCorr = |the corrected scheme's correction|.
template <class R>
__device__ inline R snLimiter(R psi, R magSn, R magCorr)
{
    return fmin((psi * magSn) / ((R(1.0) - psi) * magCorr + R(kSmall)), R(1.0));
}

// (I - 2 n n) . v
template <class R>
__device__ inline void reflect(const R (&n)[3], const R (&v)[3], R (&o)[3])
{
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        R s = 0;
#pragma unroll
        for (int j = 0; j < 3; ++j) s += ((i == j ? R(1.0) : R(0.0)) - R(2.0) * (n[i] * n[j])) * v[j];
        o[i] = s;
    }
}

template <class R>
__device__ inline void bndN(const Mesh<R>& m, int b, R (&n)[3])
{
    const std::size_t nB = std::size_t(m.nB);
#pragma unroll
    for (int i = 0; i < 3; ++i) n[i] = m.bN[std::size_t(i) * nB + std::size_t(b)];
}

// The U patch snGrad (OpenFOAM's per-type snGrad()) at boundary face b.
template <class R>
__device__ inline void uSnGrad(const Mesh<R>& m, int b, const R* U, const R* Ub, const R* vfU,
                               R (&sn)[3])
{
    const std::size_t nB = std::size_t(m.nB), nC = std::size_t(m.nC);
    const int P = m.bOwn[b];
    const R dc = m.bDc[b];
    R up[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) up[q] = U[std::size_t(q) * nC + std::size_t(P)];
    switch (m.bU[b]) {
    case 0: // fixedValue: dc (U_b - U_P)
#pragma unroll
        for (int q = 0; q < 3; ++q) sn[q] = dc * (Ub[std::size_t(q) * nB + std::size_t(b)] - up[q]);
        break;
    case 1:
#pragma unroll
        for (int q = 0; q < 3; ++q) sn[q] = 0;
        break;
    case 2:
    case 4: { // mixed: lerp(refGrad = 0, (ref - iF) dc, vf)
        const R vf = vfU[b];
#pragma unroll
        for (int q = 0; q < 3; ++q)
            sn[q] = (R(1.0) - vf) * R(0.0) + vf * ((m.bRefU[std::size_t(q) * nB + std::size_t(b)] - up[q]) * dc);
        break;
    }
    default: { // basicSymmetry: (0.5 dc) ((I - 2nn).iF - iF)
        R n[3], t[3];
        bndN(m, b, n);
        reflect(n, up, t);
#pragma unroll
        for (int q = 0; q < 3; ++q) sn[q] = (R(0.5) * dc) * (t[q] - up[q]);
        break;
    }
    }
}

// gaussGrad::correctBoundaryConditions: G_b = G_P + n (sn - n.G_P).
template <class R>
__device__ inline void bndGrad(const R (&gP)[3][3], const R (&n)[3], const R (&sn)[3],
                               R (&gb)[3][3])
{
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const R nG = n[0] * gP[0][j] + n[1] * gP[1][j] + n[2] * gP[2][j];
#pragma unroll
        for (int i = 0; i < 3; ++i) gb[i][j] = gP[i][j] + n[i] * (sn[j] - nG);
    }
}

// A constraint (symmetry / symmetryPlane) patch's value of a derived tensor
// field: 0.5 (g + R g R^T), R = I - 2 n n (symmetryPlaneFvPatchField::evaluate).
template <class R>
__device__ inline void symTensor(const R (&n)[3], R (&g)[3][3])
{
    R Rm[3][3], t[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) Rm[i][j] = (i == j ? R(1.0) : R(0.0)) - R(2.0) * (n[i] * n[j]);
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            R s = 0;
#pragma unroll
            for (int k = 0; k < 3; ++k)
#pragma unroll
                for (int l = 0; l < 3; ++l) s += Rm[i][k] * g[k][l] * Rm[j][l];
            t[i][j] = s;
        }
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) g[i][j] = R(0.5) * (g[i][j] + t[i][j]);
}

// --- gradient: Gauss linear [+ cellLimited minmod], NC components ---------
// Cpl (a mesh with cyclicAMI faces; false is the kernel bit for bit): a coupled
// face's value is lerp(pnf, pif, w) of the cell values, and cellLimited takes
// the patch NEIGHBOUR value (psf.coupled(): patchNeighbourField) for min / max.
template <class R, int NC, bool Cpl = false>
__global__ void __launch_bounds__(256) kGrad(Mesh<R> m, const R* x, const R* xb, int lim, R kLim,
                                             R* G)
{
    const int c = int(blockIdx.x) * kT + int(threadIdx.x);
    if (c >= m.nC) return;
    const std::size_t nC = std::size_t(m.nC), nI = std::size_t(m.nI), nB = std::size_t(m.nB);
    R g[3][NC];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int q = 0; q < NC; ++q) g[i][q] = 0;
    const int e0 = m.cStart[c], e1 = m.cStart[c + 1];
    for (int e = e0; e < e1; ++e) {
        const int code = m.cEnt[e];
        if (code >= 0) {
            const std::size_t f = std::size_t(code >> 1);
            const std::size_t o = std::size_t(m.own[f]), n = std::size_t(m.nei[f]);
            const R l = m.w[f];
#pragma unroll
            for (int q = 0; q < NC; ++q) {
                const R vo = x[std::size_t(q) * nC + o], vn = x[std::size_t(q) * nC + n];
                const R vf = l * (vo - vn) + vn;
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    const R Sv = m.S[std::size_t(i) * nI + f] * vf;
                    if (code & 1) g[i][q] -= Sv;
                    else g[i][q] += Sv;
                }
            }
        } else {
            const std::size_t b = std::size_t(-code - 1);
            if constexpr (Cpl) {
                const int a = m.bA[b];
                if (a >= 0) {
                    const R w = m.aWt[a];
                    for (int q = 0; q < NC; ++q) {
                        const R* xq = x + std::size_t(q) * nC;
                        const R vf = amiLerp(w, xq[c], amiNbr(m, a, xq));
#pragma unroll
                        for (int i = 0; i < 3; ++i) g[i][q] += m.bS[std::size_t(i) * nB + b] * vf;
                    }
                    continue;
                }
            }
#pragma unroll
            for (int q = 0; q < NC; ++q)
#pragma unroll
                for (int i = 0; i < 3; ++i) g[i][q] += m.bS[std::size_t(i) * nB + b] * xb[std::size_t(q) * nB + b];
        }
    }
    const R V = m.V[c];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int q = 0; q < NC; ++q) g[i][q] /= V;
    if (lim && kLim >= R(kSmall)) {
        R xc[NC], mx[NC], mn[NC], lm[NC];
#pragma unroll
        for (int q = 0; q < NC; ++q) {
            xc[q] = x[std::size_t(q) * nC + std::size_t(c)];
            mx[q] = mn[q] = xc[q];
            lm[q] = R(1.0);
        }
        for (int e = e0; e < e1; ++e) {
            const int code = m.cEnt[e];
#pragma unroll
            for (int q = 0; q < NC; ++q) {
                R v;
                if (code >= 0) {
                    const std::size_t f = std::size_t(code >> 1);
                    const std::size_t other = std::size_t((code & 1) ? m.own[f] : m.nei[f]);
                    v = x[std::size_t(q) * nC + other];
                } else {
                    v = xb[std::size_t(q) * nB + std::size_t(-code - 1)];
                    if constexpr (Cpl) {
                        const int a = m.bA[-code - 1];
                        if (a >= 0) v = amiNbr(m, a, x + std::size_t(q) * nC);
                    }
                }
                mx[q] = fmax(mx[q], v);
                mn[q] = fmin(mn[q], v);
            }
        }
#pragma unroll
        for (int q = 0; q < NC; ++q) {
            mx[q] -= xc[q];
            mn[q] -= xc[q];
            if (kLim < R(1.0)) {
                const R mm = (R(1.0) / kLim - R(1.0)) * (mx[q] - mn[q]);
                mx[q] += mm;
                mn[q] -= mm;
            }
        }
        for (int e = e0; e < e1; ++e) {
            const int code = m.cEnt[e];
            R r[3];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                if (code >= 0) {
                    const std::size_t f = std::size_t(code >> 1);
                    r[i] = ((code & 1) ? m.rN : m.rO)[std::size_t(i) * nI + f];
                } else
                    r[i] = m.bCf[std::size_t(i) * nB + std::size_t(-code - 1)] - m.C[std::size_t(i) * nC + std::size_t(c)];
            }
#pragma unroll
            for (int q = 0; q < NC; ++q) {
                const R ex = r[0] * g[0][q] + r[1] * g[1][q] + r[2] * g[2][q];
                R rr;
                if (ex > R(kSmall)) rr = mx[q] / ex;
                else if (ex < -R(kSmall)) rr = mn[q] / ex;
                else continue;
                lm[q] = fmin(lm[q], fmin(rr, R(1.0)));
            }
        }
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int q = 0; q < NC; ++q) g[i][q] *= lm[q];
    }
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int q = 0; q < NC; ++q) G[std::size_t(i * NC + q) * nC + std::size_t(c)] = g[i][q];
}

// kGrad launched for mesh m (the Cpl instantiation when it has cyclicAMI faces).
template <class R, int NC>
inline void gradLaunch(const Mesh<R>& m, const R* x, const R* xb, int lim, R kLim, R* G)
{
    const unsigned gC = grid(std::size_t(m.nC));
    if (m.nA) kGrad<R, NC, true><<<gC, kT>>>(m, x, xb, lim, kLim, G);
    else kGrad<R, NC, false><<<gC, kT>>>(m, x, xb, lim, kLim, G);
}

template <class T>
std::expected<DeviceBuffer<T>, std::string> upload(const std::vector<T>& v, const char* what)
{
    auto d = DeviceBuffer<T>::make(std::max<std::size_t>(1, v.size()));
    if (!d) return std::unexpected(std::format("fv: device allocation ({}): {}", what, hipGetErrorString(d.error())));
    if (!v.empty())
        if (auto e = hipMemcpy(d->data(), v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice); e != hipSuccess)
            return std::unexpected(std::format("fv: upload ({}): {}", what, hipGetErrorString(e)));
    return std::move(*d);
}

// A host FP64 vector on the device as R (R = double: upload() verbatim; float:
// each value rounded once).
template <class R>
std::expected<DeviceBuffer<R>, std::string> uploadAs(const std::vector<double>& v, const char* what)
{
    if constexpr (std::is_same_v<R, double>) return upload(v, what);
    else {
        std::vector<R> r(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) r[i] = R(v[i]);
        return upload(r, what);
    }
}

template <class T>
std::expected<DeviceBuffer<T>, std::string> zeros(std::size_t n, const char* what)
{
    auto d = DeviceBuffer<T>::make(std::max<std::size_t>(1, n));
    if (!d) return std::unexpected(std::format("fv: device allocation ({}): {}", what, hipGetErrorString(d.error())));
    (void)hipMemset(d->data(), 0, std::max<std::size_t>(1, n) * sizeof(T));
    return std::move(*d);
}

} // namespace fvd

// The RAS model's device state (phase BL; gpu/fv_ras.hip). FaceMesh numbering.
// SST: s[0] = k, s[1] = omega; SA: s[0] = nuTilda.
template <class R>
struct RasState
{
    FvRas co;                   // model and coefficients
    int nS = 0;
    double nu = 0;
    std::vector<FvWallFnPatch> wfHost;
    DeviceBuffer<R> y, yWall;
    DeviceBuffer<fvd::u8> bWall, bNut, fixC;
    DeviceBuffer<int> bPatch;
    DeviceBuffer<FvWallFnPatch> wf;
    std::array<DeviceBuffer<fvd::u8>, 2> bS;
    std::array<DeviceBuffer<R>, 2> bRef, s, sb, vf, gs, D, DB;
    DeviceBuffer<R> nut, nutB, nuEff, nuEffB;
    // work: G, GbyNu, S2, divU, F1, F23, CDkOmega, (SA: fv1-like scratch), model sources
    DeviceBuffer<R> Gk, GbyNu, S2, S2B, divU, F1, F23, CDk, dAdd, sAdd, fixV;
    DeviceBuffer<R> sup, slo, sfx, sic, sbc, minPart, dlt;
};

// The EXPERIMENTAL coupled prototype's work state (gpu/fv_coupled.hip),
// allocated at the first coupled iteration. Vectors x = (Ux, Uy, Uz, p),
// component-major, 4 nC.
template <class R>
struct CoupledState
{
    DeviceBuffer<R> res, dx;     // [4 nC] R* and the Newton correction
    DeviceBuffer<R> diagJ;       // [3 nC] the Jacobian's U diagonals (D0 + ic + tau)
    DeviceBuffer<R> D0, Dm;      // [nC] kURow's diagonal before relax, its dominance-fixed max
    DeviceBuffer<R> gd, dUb, dpb; // [3 nC] grad dp, [3 nB] dU_b, [nB] dp_b (homogeneous BCs)
    DeviceBuffer<R> rc;          // [nC] preconditioner continuity rhs
    DeviceBuffer<R> save;        // G-JAC: the base state
    DeviceBuffer<int> flags;     // [3] gsU factor flags
    std::optional<la::Krylov<R>> kr; // FGMRES on 4 nC
    int outer = 0;
    double r0 = -1;              // SER: the first probe
};

// pimpleFoam's state beside the SIMPLE one (gpu/fv_pimple.hip; null: a
// simpleFoam case, no ddt kernel instantiation launches). The old-time fields
// are the state at the start of the time step (GeometricField::oldTime()).
// The Impl's FoamCase copy and RasState::co hold the CURRENT corrector's
// solver / relaxation entries (the stage methods read them there); the case's
// own two sets live here.
// A moving mesh's state beside PimpleState (FoamCase::motion.on, mesh.dynamic();
// null: a static mesh, none of its kernels launches). docs/fvpimple.md "Moving
// mesh". phi is the RELATIVE flux between pEqn.H's end and the next
// correctPhi / pEqn.H, absolute inside them.
template <class R>
struct MotionState
{
    DeviceBuffer<R> meshPhi, meshPhiB; // [nI] [nB] fvc::meshPhi of the current step
    DeviceBuffer<R> Uf, UfB, UfOld, UfBOld; // [3 nI] [3 nB] the face velocity and its old-time value
    DeviceBuffer<R> V0;                // [nC] mesh.V0(): the volumes before this step's motion
    DeviceBuffer<R> phiAbs, phiAbsB;   // [nI] [nB] fvc::absolute(phi, U) for kOmegaSST's divU
    DeviceBuffer<R> pc, pcB, zeroB;    // [nC] [nB] [nB] CorrectPhi's pcorr, its patch values, zeros (its fixed value)
    DeviceBuffer<R> uWall;             // [3 nB] this step's Uwall() (host)
    DeviceBuffer<fvd::u8> bMove, bPc;  // [nB] movingWallVelocity face; pcorr's patch type (0 fixedValue, 1 zeroGradient)
    int pcRefCell = -1;                // pcorrEqn.setReference(0, 0): OpenFOAM cell 0 when pcorr needs a reference
    bool limPc = false;                // grad(pcorr)
    R kPc = 1;
    long steps = 0;                    // time steps taken (the first one copies U.oldTime() AFTER the mesh update)
    bool correctPhi = true, checkMeshCo = false;
    FvSolverCtl solverPcorr, solverPcorrFinal;
};

template <class R>
struct PimpleState
{
    std::unique_ptr<MotionState<R>> mov;          // null: static mesh
    DeviceBuffer<R> Uold, UbOld, phiOld, phiBOld; // [3 nC] [3 nB] [nI] [nB]
    std::array<DeviceBuffer<R>, 2> sOld;          // RAS scalars [nC]
    DeviceBuffer<R> coPart;                       // [blocks] CourantNo.H's block maxima
    FvSolverCtl solverU, solverUFinal, solverP, solverPFinal;
    std::array<FvSolverCtl, 2> solverS, solverSFinal;         // RasState::s order
    std::optional<double> relaxU, relaxUFinal, relaxP, relaxPFinal;
    std::array<std::optional<double>, 2> relaxS, relaxSFinal;
};

// The precision-independent face of the solver state (DeviceSimple's).
struct DeviceSimple::Impl
{
    virtual ~Impl() = default;
    virtual std::expected<FvCourant, std::string> courant(double deltaT) = 0;
    virtual std::expected<FvStepLog, std::string> timeStep(double deltaT, const FvMovedMesh* moved) = 0;
    virtual std::expected<FvIterLog, std::string> iterate() = 0;
    virtual std::expected<FvIterLog, std::string> coupledIterate(const FvCoupledOpts& o) = 0;
    virtual std::expected<FvForceSums, std::string> forces(int k) = 0;
    virtual std::expected<FvState, std::string> state() const = 0;
    virtual int iteration() const = 0;
    virtual void prepareMatrices(const FvSimpleSetup& s, const dns::la::CellGraph& g) = 0;
};

// The solver state in arithmetic R (double: the parity mode, bit for bit the
// FP64-only solver; float: FP32 fields, matrices, Krylov vectors, AMG, with
// every reduction in FP64 -- docs/fv.md "Precision").
template <class R>
struct SimpleImpl final : DeviceSimple::Impl
{
    fvd::Mesh<R> m;
    fvd::Ctl<R> c;
    FoamCase fc;
    FvFloors floor;    // solve floors (FP64: none): absTol = max(tolerance, floor)
    int iter = 0, nSolve = 3, refCell = -1;
    bool needRef = false, multiZone = false;
    double refValue = 0, relaxP = 1, sumV = 0;
    bool relaxPGiven = false;
    std::array<double, 3> cofr{};
    // Gradient reuse (masterplan_performance step 4), bit-identical by
    // construction: G = grad(U, Ub) is valid when the last writer of U, Ub ran
    // before the last kGrad<3> (forces(), rasCorrect); gP = grad(p, pb) after
    // the corrector's final kGrad<1>. mrfStatic: every MRF-included face is
    // fixedValue (bU 0), so kMrfBoundary rewrites the Ub it wrote last time
    // and G stays valid across it; otherwise G is always recomputed.
    bool gValid = false, gPValid = false, mrfStatic = true;
    // p corrector nc's last PCG iteration count (-1: none yet): 0 = the tail,
    // where r0 is read before the V-cycle (KrylovOptions::r0First, solve()).
    std::vector<int> pLastIters;
    bool r0First = false;

    // mesh storage
    DeviceBuffer<int> own, nei, bOwn, cStart, cEnt, cEll;
    DeviceBuffer<R> S, rO, rN, k, magS, w, dc, bS, bN, bCf, bMagS, bDc, bRefU, bRefP, bFrame, bUmrf, frame, frame2, V, C;
    DeviceBuffer<fvd::u8> bU, bP, bAssign, bSym, bAdjust, bMrf, bForce, zone;
    // state
    DeviceBuffer<R> U, Ub, p, pb, phi, phiB, pPrev, pbPrev, vfU, vfP;
    // work
    DeviceBuffer<R> G, gP, Hc, up, lo, fx, ic, bc, bfx, dU, sU, rAU, rAtU, HbyA, phiH, phiHB, pUp, ffc, pIc, pBc,
        rhs, x, sumA, Ax, tmp, ones, cofrD;
    DeviceBuffer<double> part, outv; // FP64 dot partials and results
    // the batched U solve: 3 component-major systems (diagonal, rhs, A x0, A 1)
    DeviceBuffer<R> diag3, rhs3, Ax3, sumA3;
    std::optional<la::BlockEll<R>> AU, AP;
    std::optional<la::MultiColourGS<R>> gsU;
    std::optional<la::Amg<R>> amg;
    std::optional<la::Krylov<R>> kr, kr3; // kr3: bicgstabBatch of the U components
    std::unique_ptr<RasState<R>> ras; // null: laminar
    std::unique_ptr<CoupledState<R>> cp; // null: no coupled iteration yet
    std::unique_ptr<PimpleState<R>> pim; // null: simpleFoam

    // --- cyclicAMI (gpu/fv_ami.hip, docs/fvpimple.md "cyclicAMI"; m.nA == 0:
    // null, none of it launches, the solver bit for bit). The patch kernels
    // treat a coupled face as zeroGradient; each ami*() call below overwrites
    // what they wrote there with OpenFOAM's coupled-patch expression.
    struct AmiDev
    {
        DeviceBuffer<int> aB, bA, aStart, aCell, aEll, rCell, rStart, rA;
        DeviceBuffer<R> aW, aWt, aK, aD;
        // acU / acP [nB]: boundaryCoeffs of the U / p matrix on coupled faces
        // (interfaceBouCoeffs; 0 elsewhere). ffcB [nB]: the p laplacian's
        // faceFluxCorrection on coupled faces.
        DeviceBuffer<R> acU, acP, ffcB;
        // RAS scalars (gpu/fv_ras.hip): boundaryCoeffs and the explicit flux of
        // the scalar being assembled, on coupled faces.
        DeviceBuffer<R> acS, sfxB;
    };
    std::unique_ptr<AmiDev> amiD;
    // (Re)uploads s.ami and points m at it. Error: allocation / upload.
    std::expected<void, std::string> amiMake(const FvSimpleSetup& s);
    // The matrices, GS, AMG and Krylov workspaces on graph g (make(); again
    // after a mesh motion changed the AMI pattern). Error: theirs.
    std::expected<void, std::string> makeMatrices(const FvSimpleSetup& s, const dns::la::CellGraph& g);
    // makeMatrices' HOST part: the ELL pattern, the colours, the AMG hierarchy on
    // the geometric weights. Pure (no device call, reads s and g only), so
    // prepareMatrices may run it on another thread while the mesh update's wall
    // distance is computed; makeMatrices takes a prepared one made on the SAME
    // graph object and computes it itself otherwise -- one definition, the same
    // values either way.
    struct MatPrep
    {
        const dns::la::CellGraph* g = nullptr;
        dns::la::EllPattern pat;
        std::vector<std::uint8_t> colours;
        std::expected<dns::la::AmgHierarchy, std::string> h;
    };
    std::optional<MatPrep> prep;
    static MatPrep matPrepOf(const FvSimpleSetup& s, const dns::la::CellGraph& g);
    void prepareMatrices(const FvSimpleSetup& s, const dns::la::CellGraph& g) override { prep.emplace(matPrepOf(s, g)); }
    // evaluate(): xb = lerp(pnf, pif, w) on coupled faces, NC components. Async.
    void amiEval(int nc, const R* x, R* xb);
    // UEqn: internalCoeffs (ic), boundaryCoeffs (acU; bc = 0), the explicit
    // boundary flux bfx on coupled faces; then the couplings into AU. Async.
    void amiUBnd();
    // fix (setValues: the omega wall cells): a fixed row's couplings are 0.
    void amiOff(la::BlockEll<R>& A, const R* coef, bool negate, const fvd::u8* fix = nullptr);
    // The couplings amiOff last assembled into AU / AP, and the row sum's share
    // of the faces WITHOUT overlap (lduMatrix::sumA subtracts boundaryCoeffs
    // whole; such a face has no assembled entry): added to rowSum (nComp
    // component-major, mask) after A 1. No such face: nothing is written. Async.
    struct AmiCpl
    {
        const R* coef = nullptr;
        bool negate = false;
        const fvd::u8* fix = nullptr;
    };
    AmiCpl cplU, cplP;
    void amiSumA(const la::BlockEll<R>& A, int nComp, unsigned mask, R* rowSum);
    // H(): bc = acU pnf(U) (addBoundarySource's coupled branch). phiHbyA on
    // coupled faces = Sf & lerp(pnf, pif, w)(HbyA). Async.
    void amiHsrc();
    void amiPhiH();
    // p | pcorr equation on coupled faces: pIc, acP, pBc = 0 (first) and ffcB
    // from gamma rA and the field pv with gradient gp; phi_b = fHB - flux. Async.
    void amiPBnd(const R* rA, const R* pv, const R* gp, bool first);
    void amiPhiB(const R* fHB, const R* pv, R* out);

    // gpu/fv_pimple.hip
    std::expected<FvCourant, std::string> courant(double deltaT) override;
    std::expected<FvStepLog, std::string> timeStep(double deltaT, const FvMovedMesh* moved) override;
    // Allocates pim and takes the case's two control sets; with fc.motion.on the
    // MotionState too (Uf = interpolate(U): createUfIfPresent.H). s: the setup
    // make() uploaded. Error: allocation.
    std::expected<void, std::string> pimpleMake(const FvSimpleSetup& s);
    // pEqn.H: phiHbyA += interpolate(rAU) fvc::ddtCorr(U, phi, Uf): static mesh
    // EulerDdtScheme::fvcDdtPhiCorr (phi.oldTime()), dynamic mesh fvcDdtUfCorr
    // (Sf & Uf.oldTime() on the CURRENT Sf), both with ddtScheme::fvcDdtPhiCoeff,
    // ddtPhiCoeff -1; internal and boundary faces. Called by prepP() right
    // after flux(HbyA). Async.
    void ddtCorr();
    // --- moving mesh (pim->mov; gpu/fv_pimple.hip) ---
    // mesh.update(): V0 = V, the moved geometry (and RAS wall distances), the
    // mesh flux, then U.correctBoundaryConditions() with movingWallVelocity's
    // Uwall(). Error: an upload's.
    std::expected<void, std::string> meshUpdate(const FvMovedMesh& mv);
    // pimpleFoam.C's correctPhi branch: phi = Sf & Uf, CorrectPhi (correctUphiBCs,
    // the pcorr equation with its adjustPhi and non-orthogonal loop, phi -=
    // pcorrEqn.flux()), continuityErrs.H into log.contCorr; phi is left ABSOLUTE.
    // Error: adjustPhi's, a solver's.
    std::expected<void, std::string> correctPhi(double deltaT, FvIterLog& log);
    // fvc::makeRelative / makeAbsolute(f, U): f -+= meshPhi, internal and boundary. Async.
    void relFlux(R* fI, R* fB, bool relative);
    // fvc::correctUf(Uf, U, phi) with the ABSOLUTE phi. Async.
    void correctUf();
    // adjustPhi(f, U, p) on the internal / boundary flux f (p needs a reference).
    // Error: "continuity error cannot be removed", a reduction's.
    std::expected<void, std::string> adjustPhi(R* fI, R* fB);
    // CourantNo.H / meshCourantNo.H of flux (f, fB) at deltaT. Error: a download's, non-finite.
    std::expected<FvCourant, std::string> courantOf(const R* f, const R* fB, double deltaT);
    // fvc::absolute(phi, U) of the current (relative) phi: phi itself on a static mesh.
    const R* phiAbs() const { return pim && pim->mov ? pim->mov->phiAbs.data() : phi.data(); }
    const R* phiAbsB() const { return pim && pim->mov ? pim->mov->phiAbsB.data() : phiB.data(); }
    // fvm::ddt(X) of RAS scalar q onto the model's dAdd / sAdd (called by
    // rasCorrect before each scalar's solve). Async.
    void rasDdt(int q);

    std::expected<FvIterLog, std::string> iterate() override;
    // gpu/fv_coupled.hip
    std::expected<FvIterLog, std::string> coupledIterate(const FvCoupledOpts& o) override;
    std::expected<FvForceSums, std::string> forces(int k) override;
    std::expected<FvState, std::string> state() const override;
    int iteration() const override { return iter; }

    // fvOptions actuationDiskSource (docs/fv.md "fvOptions"): per disk the
    // zone's cells and volume fractions on the device, the monitor cells, and
    // this iteration's thrust T (evalDisks; rowsU subtracts it). Empty: no
    // kernel launches, the solver bit for bit.
    struct DiskDev
    {
        DeviceBuffer<int> cells;
        DeviceBuffer<R> frac;
        int n = 0;
        std::vector<int> monitor;
        double T = 0;
    };
    std::vector<DiskDev> disks;
    // calcFroudeMethod's head: Uref (monitor mean of the current U), a, T per
    // disk into disks[k].T and log.disks. Syncs (3 scalars per monitor cell).
    // Error: a download's, a non-finite Uref.
    std::expected<void, std::string> evalDisks(FvIterLog& log);

    // --- the stages of one SIMPLE iteration (iterate() is their composition,
    // in this order, bit for bit; the coupled prototype reuses them) ---
    // UEqn assembly: MRF.correctBoundaryVelocity, grad U / grad p (reused when
    // valid), face / boundary coefficients, rows (AU off, dU, sU). Async.
    void assembleU();
    // assembleU's last two launches: the boundary coefficients (kUBnd: vfU, ic,
    // bc, bfx) and the rows (kURow: AU off, dU, sU) from the face coefficients.
    void boundaryU();
    void rowsU();
    // The solved momentum components: all but the empty axis (bit j).
    unsigned uMask() const { return 7u & ~(c.emptyAxis >= 0 ? 1u << c.emptyAxis : 0u); }
    // The predictor systems diag3 / rhs3 of the components in mask (-V grad p
    // with the current gP). Async.
    void systemU(unsigned mask);
    // U.correctBoundaryConditions() (update: the mixed patches re-read phi / Ub).
    void evalU(int update);
    // rAU, rAtU, H, HbyA, phiHbyA (constrainHbyA, makeRelative), adjustPhi,
    // the SIMPLEC terms, from the current U and UEqn coefficients.
    // Error: adjustPhi's, a reduction's.
    std::expected<void, std::string> prepP();
    // Corrector nc's p system in AP / rhs (nc > 0: grad p first, ffc and rhs
    // only). Async.
    void assembleP(int nc);
    // p.correctBoundaryConditions(). Async.
    void evalP();
    // phi = phiHbyA - pEqn.flux() with the current p, and the continuity
    // errors into log. Error: a reduction's.
    std::expected<void, std::string> fluxes(FvIterLog& log);
    // p.relax, grad p, U = HbyA - rAtU grad p, U.correctBoundaryConditions().
    void correctU();

    // sum_i a_k[i] (abs: |a_k[i]|) in FP64, K <= 66 arrays of length N, fixed order.
    std::expected<std::vector<double>, std::string> sums(std::size_t N, std::span<const R* const> a,
                                                         std::span<const bool> abs);
    // One OpenFOAM-semantics solve of A x = rhs (x holds the initial guess).
    // factor: refactored first, its failure flag read with the solve's first
    // batch (no sync of its own); error text MultiColourGS::factor()'s.
    // Host syncs: the Krylov method's alone (normFactor rides its first batch).
    // The momentum predictor's component solves (mask: the non-empty ones) as
    // one bicgstabBatch on AU's off-diagonals with diag3 / rhs3: each component
    // exactly solve(AU, gsU, BiCGStab) of it, the first failing component's
    // error ("fv: U<c>: ..."), as the component-by-component loop reported.
    std::expected<void, std::string> solveU(unsigned mask, FvIterLog& log);
    // offChanged false: A's off-diagonal values are those of factor's last
    // refactor (only the diagonal changed). sumAValid: A's values are those of
    // the previous solve (sumA = A 1 kept). floor: the field's FvFloors entry;
    // absTol = max(ctl.tolerance, floor), a stop only the floor made is
    // FvSolveLog::floorLimited.
    std::expected<FvSolveLog, std::string> solve(la::BlockEll<R>& A, const la::LinOp<R>& M, bool pcg,
                                                 const FvSolverCtl& ctl, double floor, R* xv,
                                                 la::MultiColourGS<R>* factor = nullptr, bool offChanged = true,
                                                 bool sumAValid = false);
};

// RAS (gpu/fv_ras.hip). rasMake: uploads setup s.ras, bounds k / omega as the
// model constructor, then validate() (correctNut on the initial fields).
// rasCorrect: turbulence->correct() after the SIMPLE corrector; appends the
// scalar solves to lg.T. Errors: device allocation, a solver failure.
template <class R>
std::expected<void, std::string> rasMake(SimpleImpl<R>& I, const FvSimpleSetup& s, const FoamCase& fc);
template <class R>
std::expected<void, std::string> rasCorrect(SimpleImpl<R>& I, FvIterLog& lg);

} // namespace dns::gpu
