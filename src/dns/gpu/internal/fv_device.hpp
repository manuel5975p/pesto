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
};

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
template <class R, int NC>
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
                } else v = xb[std::size_t(q) * nB + std::size_t(-code - 1)];
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

// The precision-independent face of the solver state (DeviceSimple's).
struct DeviceSimple::Impl
{
    virtual ~Impl() = default;
    virtual std::expected<FvIterLog, std::string> iterate() = 0;
    virtual std::expected<FvIterLog, std::string> coupledIterate(const FvCoupledOpts& o) = 0;
    virtual std::expected<FvForceSums, std::string> forces() = 0;
    virtual std::expected<FvState, std::string> state() const = 0;
    virtual int iteration() const = 0;
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

    std::expected<FvIterLog, std::string> iterate() override;
    // gpu/fv_coupled.hip
    std::expected<FvIterLog, std::string> coupledIterate(const FvCoupledOpts& o) override;
    std::expected<FvForceSums, std::string> forces() override;
    std::expected<FvState, std::string> state() const override;
    int iteration() const override { return iter; }

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
