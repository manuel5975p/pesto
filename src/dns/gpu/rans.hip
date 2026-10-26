#include "dns/gpu/rans.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/stage_write.hpp"
#include "dns/gpu/internal/wall_ghost.hpp"
#include "dns/rans.hpp"
#include "dns/sgs.hpp"

#include <cassert>

namespace dns::gpu::rans {

namespace {

// Molecular mu at temperature T (the face or cell value the caller chose).
__device__ inline Real muAt(const ViscView& vv, Real T)
{
    return vv.sutherland ? vv.mu * vv.ratio(T) : vv.mu;
}

__global__ void kRansMutCurv(GpuGrid g, ViscView vv, const Real* rho,
                             const Real* T, const Real* t0, Real* nut)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    const Real r = rho[c];
    nut[c] = saMut(r, t0[c] / r, muAt(vv, T[c]) / r);
}

__global__ void kMutMirrorY(GpuGrid g, int lo, int hi, Real* nut, Real* f1)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i, k;
    bool top;
    std::size_t c, s;
    if (!internal::yGhostPair(g, t, i, k, top, c, s)) return;
    if (!(top ? hi : lo)) return;
    nut[c] = -nut[s];
    if (f1) f1[c] = f1[s];
}

__device__ inline Real massFlux(PrimView q, const Real Sv[3], std::size_t a,
                                std::size_t b, const Real* wf, Real wS);

// The open-face scalar flux of the immersed path (ScalarBoundary), along +axis
// through the face area vector Sv of boundary cell c: mdot = rho_c (u_c . S)
// from the cell alone, carrying phi_c out and phi_inf in, no diffusion. `hi`:
// the face is the cell's +axis domain face (else its -axis one).
__device__ inline Real openFaceFlux(PrimView q, const Real Sv[3], std::size_t c,
                                    Real phiC, Real phiInf, bool hi)
{
    const Real mdot =
        q.rho[c] * (q.u[c] * Sv[0] + q.v[c] * Sv[1] + q.w[c] * Sv[2]);
    const bool out = hi ? mdot > Real(0) : mdot < Real(0);
    return mdot * (out ? phiC : phiInf);
}

// Is the +axis (hi) / -axis (lo) face of cell (i, j, k) an open domain face?
__device__ inline void openFaces(const GpuGrid& g, const ScalarBoundary& sb,
                                 int i, int j, int k, int a, bool& hi, bool& lo)
{
    const int ijk[3] = {i, j, k};
    const int nn[3] = {g.nx, g.ny, g.nz};
    hi = sb.open[a] && ijk[a] == nn[a] - 1;
    lo = sb.open[a] && ijk[a] == 0;
}

// The rate the SST production and mu_t use: S = sqrt(2 S:S) (2003m) or the
// vorticity magnitude Omega (Vm), from the cell-centre velocity gradient.
__device__ inline Real sstRate(RansModel m, const Real gd[3][3])
{
    if (m == RansModel::SstVm) {
        const Real w0 = gd[2][1] - gd[1][2], w1 = gd[0][2] - gd[2][0],
                   w2 = gd[1][0] - gd[0][1];
        return sqrtf(w0 * w0 + w1 * w1 + w2 * w2);
    }
    return sgsRate(0, gd); // |S| = sqrt(2 S:S), the Smagorinsky operator
}

// Cell-centre gradient pieces every SST kernel needs at cell c: the rate R
// and grad k . grad omega (specific scalars through the live density).
__device__ inline void sstCellGrads(RansModel model, PrimView q,
                                    const CurvView& m, const Offsets& o,
                                    std::size_t c, const Real* t0,
                                    const Real* t1, Real& R, Real& gkw)
{
    const Real* vc[3] = {q.u, q.v, q.w};
    Real gd[3][3];
    curvVelGradCentre(m, o, c,
                      [&](std::size_t x, int comp) { return vc[comp][x]; }, gd);
    R = sstRate(model, gd);
    Real gk[3], gw[3];
    curvScalarGradCentre(m, o, c, [&](std::size_t x) { return t0[x] / q.rho[x]; },
                         gk);
    curvScalarGradCentre(m, o, c, [&](std::size_t x) { return t1[x] / q.rho[x]; },
                         gw);
    gkw = gk[0] * gw[0] + gk[1] * gw[1] + gk[2] * gw[2];
}

__global__ void kSstMutCurv(GpuGrid g, ViscView vv, PrimView q, CurvView m,
                            RansModel model, const Real* t0, const Real* t1,
                            const Real* d, Real* nut, Real* f1)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    Real R, gkw;
    sstCellGrads(model, q, m, o, c, t0, t1, R, gkw);
    const Real r = q.rho[c];
    const Real kk = t0[c] / r, om = t1[c] / r;
    const Real nu = muAt(vv, q.T[c]) / r;
    const Real cd = sstCd(r, gkw, om, Real(sstCdFloor(model)));
    const SstBlend<Real> b = sstBlend(kk, om, d[c], nu, r, cd);
    nut[c] = sstMut(r, kk, om, R, b.F2);
    f1[c] = b.F1;
}

// Total flux (convective minus diffusive) of rho phi, phi = t/rho, through the
// face a-b: upwind on the mean mass flux, diffusion (mu_f + sig mu_t,f)
// (grad phi . S) with mu_t,f the face average of the (mirrored) cell mu_t.
__device__ inline Real sstFaceFlux(ViscView vv, PrimView q, const CurvView& m,
                                   const Offsets& o, std::size_t a,
                                   std::size_t b, int axis, const Real* t,
                                   Real mdot, Real sig, Real mutf)
{
    const Real Sv[3] = {m.S[axis][0][a], m.S[axis][1][a], m.S[axis][2][a]};
    auto ph = [&](std::size_t x) { return t[x] / q.rho[x]; };
    const Real F = fmaxf(mdot, Real(0)) * ph(a) + fminf(mdot, Real(0)) * ph(b);
    const Real muf = muAt(vv, Real(0.5) * (q.T[a] + q.T[b]));
    Real gr[3];
    curvScalarFaceGrad(m, o, a, b, axis, ph, gr);
    const Real gS = gr[0] * Sv[0] + gr[1] * Sv[1] + gr[2] * Sv[2];
    return F - (muf + sig * mutf) * gS;
}

// Comp: the compensated write (gpu/internal/stage_write.hpp). Src: the TEST
// ONLY sources are live (a compile-time switch, so the production
// instantiation is the source-free code verbatim).
// Bnd: the open domain faces of the immersed path are live (ScalarBoundary;
// compile-time, so the instantiation without them is the stage verbatim).
template <bool Comp, bool Src, bool Bnd>
__global__ void kSstStageCurv(GpuGrid g, ViscView vv, PrimView q, CurvView m,
                              MrfView mz, StageCombine cmb, SstStageArgs A,
                              int residual, Real* out0, Real* out1,
                              ScalarBoundary sb)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    Real divK = 0, divW = 0;
    for (int a = 0; a < 3; ++a) {
        Real whi[3], wlo[3];
        const Real *pwhi = nullptr, *pwlo = nullptr;
        Real qhi = 0, qlo = 0;
        if (mz.active) { // the same face correction kCompStageCurv applies
            const int il = a == 0 ? (i ? i - 1 : g.nx - 1) : i;
            const int jl = a == 1 ? (j ? j - 1 : g.ny - 1) : j;
            const int kl = a == 2 ? (k ? k - 1 : g.nz - 1) : k;
            if (mz.psi) {
                qhi = mrfFaceQNode(mz, g, i, j, a);
                qlo = mrfFaceQNode(mz, g, il, jl, a);
            } else {
                mrfFaceVel(mz, i, j, k, a, whi);
                mrfFaceVel(mz, il, jl, kl, a, wlo);
                pwhi = whi;
                pwlo = wlo;
            }
        }
        bool openHi = false, openLo = false;
        if constexpr (Bnd) openFaces(g, sb, i, j, k, a, openHi, openLo);
        for (int side = 0; side < 2; ++side) {
            const std::size_t fa = side ? c + o.m[a] : c;
            const std::size_t fb = side ? c : c + o.p[a];
            const Real Sv[3] = {m.S[a][0][fa], m.S[a][1][fa], m.S[a][2][fa]};
            if constexpr (Bnd) {
                if (side ? openLo : openHi) {
                    const Real r = q.rho[c];
                    const Real fk = openFaceFlux(q, Sv, c, A.t0[c] / r,
                                                 Real(sb.phiInf[0]), !side);
                    const Real fw = openFaceFlux(q, Sv, c, A.t1[c] / r,
                                                 Real(sb.phiInf[1]), !side);
                    divK -= side ? -fk : fk;
                    divW -= side ? -fw : fw;
                    continue;
                }
            }
            const Real mdot = massFlux(q, Sv, fa, fb, side ? pwlo : pwhi,
                                       side ? qlo : qhi);
            const Real F1f = Real(0.5) * (A.f1[fa] + A.f1[fb]);
            const Real mutf = Real(0.5) * (A.nut[fa] + A.nut[fb]);
            const Real sk = sstBlendC(F1f, sst::sigmaK1, sst::sigmaK2);
            const Real sw = sstBlendC(F1f, sst::sigmaW1, sst::sigmaW2);
            const Real fk = sstFaceFlux(vv, q, m, o, fa, fb, a, A.t0, mdot, sk, mutf);
            const Real fw = sstFaceFlux(vv, q, m, o, fa, fb, a, A.t1, mdot, sw, mutf);
            divK -= side ? -fk : fk;
            divW -= side ? -fw : fw;
        }
    }
    const Real iv = m.invVol[c];
    Real Rk, gkw;
    sstCellGrads(A.model, q, m, o, c, A.t0, A.t1, Rk, gkw);
    const Real r = q.rho[c];
    const SstPoint<Real> P = sstSource(A.model, r, A.t0[c] / r, A.t1[c] / r, Rk,
                                       A.nut[c], A.f1[c], gkw);
    Real rk = divK * iv + P.Rk, rw = divW * iv + P.Rw;
    if constexpr (Src) {
        rk += A.src0[c];
        rw += A.src1[c];
    }
    if (residual) {
        out0[c] = rk;
        out1[c] = rw;
        return;
    }
    const Real cr = cmb.crAt(c);
    const Real ck = cmb.ca * cmb.ta[0][c] + cmb.cb * cmb.tb[0][c];
    const Real cw = cmb.ca * cmb.ta[1][c] + cmb.cb * cmb.tb[1][c];
    Word2 ok, ow;
    if constexpr (Comp) {
        ok = internal::stageCombineT(
            cmb, 0, c, pointImplicitDelta(ck, cr, rk, P.Jk, A.t0[c]));
        ow = internal::stageCombineT(
            cmb, 1, c, pointImplicitDelta(cw, cr, rw, P.Jw, A.t1[c]));
    } else {
        ok = {pointImplicit(ck, cr, rk, P.Jk, A.t0[c]), Real(0)};
        ow = {pointImplicit(cw, cr, rw, P.Jw, A.t1[c]), Real(0)};
    }
    // The COUNTED positivity guard (docs/rans.md): never silent -- every
    // pinned test requires zero hits. Ghost cells are their mesh family's to
    // overwrite. On the pair it tests hi: a renormalised hi of 0 has lo == 0.
    const bool counted = !A.counted || A.counted[c];
    const Real wmin = A.omegaFloor * A.outRho[c];
    if (!(ok.hi >= Real(0))) {
        ok = {Real(0), Real(0)};
        if (counted) atomicAdd(A.hits, 1ull);
    }
    if (!(ow.hi >= wmin)) {
        ow = {wmin, Real(0)};
        if (counted) atomicAdd(A.hits, 1ull);
    }
    out0[c] = ok.hi;
    out1[c] = ow.hi;
    if constexpr (Comp) {
        cmb.otl[0][c] = ok.lo;
        cmb.otl[1][c] = ow.lo;
    }
}

// The advecting area flux through the +`axis` face of the a-b pair: the
// VERBATIM expression faceFluxCurv puts in f.rho (rm * Ua, MRF correction
// included) -- it must match faceFluxCurv, or a uniform specific scalar stops
// being transported exactly with the density (gpu_rans G3 pins it).
__device__ inline Real massFlux(PrimView q, const Real Sv[3], std::size_t a,
                                std::size_t b, const Real* wf, Real wS)
{
    const Real rm = Real(0.5) * (q.rho[a] + q.rho[b]);
    const Real um[3] = {Real(0.5) * (q.u[a] + q.u[b]),
                        Real(0.5) * (q.v[a] + q.v[b]),
                        Real(0.5) * (q.w[a] + q.w[b])};
    const Real Un = um[0] * Sv[0] + um[1] * Sv[1] + um[2] * Sv[2];
    const Real Ua =
        wf ? Un - (wf[0] * Sv[0] + wf[1] * Sv[1] + wf[2] * Sv[2]) : Un - wS;
    return rm * Ua;
}

// Total scalar flux (convective minus diffusive) of rho nu~ through the face
// between a and b. Convection: first-order upwind of the SPECIFIC scalar on
// the mean mass flux. Diffusion: (1/sigma) rho_f (nu_f + nu~_f fn(chi_f))
// (grad nu~ . S)_f with every coefficient formed ON THE FACE from face-averaged
// primitives -- never the average of per-cell coefficients (docs/rans.md).
__device__ inline Real saFaceFlux(ViscView vv, PrimView q, const CurvView& m,
                                  const Offsets& o, std::size_t a,
                                  std::size_t b, int axis, const Real* t0,
                                  const Real* wf, Real wS, bool transportOnly)
{
    const Real Sv[3] = {m.S[axis][0][a], m.S[axis][1][a], m.S[axis][2][a]};
    auto nt = [&](std::size_t x) { return t0[x] / q.rho[x]; };
    const Real mdot = massFlux(q, Sv, a, b, wf, wS);
    Real F = fmaxf(mdot, Real(0)) * nt(a) + fminf(mdot, Real(0)) * nt(b);
    if (transportOnly) return F;
    const Real rf = Real(0.5) * (q.rho[a] + q.rho[b]);
    const Real nuf = muAt(vv, Real(0.5) * (q.T[a] + q.T[b])) / rf;
    const Real ntf = Real(0.5) * (nt(a) + nt(b));
    Real gr[3];
    curvScalarFaceGrad(m, o, a, b, axis, nt, gr);
    const Real gS = gr[0] * Sv[0] + gr[1] * Sv[1] + gr[2] * Sv[2];
    return F - Real(1.0 / sa::sigma) * rf * saDiffCoef(nuf, ntf) * gS;
}

// Src: the TEST ONLY source is live; Bnd: the open domain faces (both
// compile-time, as in kSstStageCurv).
template <bool Comp, bool Src, bool Bnd>
__global__ void kSaStageCurv(GpuGrid g, ViscView vv, PrimView q, CurvView m,
                             MrfView mz, StageCombine cmb, const Real* t0,
                             const Real* d, int residual, int transportOnly,
                             const Real* src, Real* out0, ScalarBoundary sb)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    Real div = 0;
    for (int a = 0; a < 3; ++a) {
        Real whi[3], wlo[3];
        const Real *pwhi = nullptr, *pwlo = nullptr;
        Real qhi = 0, qlo = 0;
        if (mz.active) { // the same face correction kCompStageCurv applies
            const int il = a == 0 ? (i ? i - 1 : g.nx - 1) : i;
            const int jl = a == 1 ? (j ? j - 1 : g.ny - 1) : j;
            const int kl = a == 2 ? (k ? k - 1 : g.nz - 1) : k;
            if (mz.psi) {
                qhi = mrfFaceQNode(mz, g, i, j, a);
                qlo = mrfFaceQNode(mz, g, il, jl, a);
            } else {
                mrfFaceVel(mz, i, j, k, a, whi);
                mrfFaceVel(mz, il, jl, kl, a, wlo);
                pwhi = whi;
                pwlo = wlo;
            }
        }
        if constexpr (Bnd) {
            bool openHi = false, openLo = false;
            openFaces(g, sb, i, j, k, a, openHi, openLo);
            const Real phiC = t0[c] / q.rho[c], phiI = Real(sb.phiInf[0]);
            Real Sv[3];
            const std::size_t fl = c + o.m[a];
            for (int x = 0; x < 3; ++x) Sv[x] = m.S[a][x][c];
            const Real hi = openHi ? openFaceFlux(q, Sv, c, phiC, phiI, true)
                                   : saFaceFlux(vv, q, m, o, c, c + o.p[a], a,
                                                t0, pwhi, qhi, transportOnly);
            for (int x = 0; x < 3; ++x) Sv[x] = m.S[a][x][fl];
            const Real lo = openLo ? openFaceFlux(q, Sv, c, phiC, phiI, false)
                                   : saFaceFlux(vv, q, m, o, fl, c, a, t0,
                                                pwlo, qlo, transportOnly);
            div -= hi - lo;
        } else {
            const Real hi = saFaceFlux(vv, q, m, o, c, c + o.p[a], a, t0, pwhi,
                                       qhi, transportOnly);
            const Real lo = saFaceFlux(vv, q, m, o, c + o.m[a], c, a, t0, pwlo,
                                       qlo, transportOnly);
            div -= hi - lo;
        }
    }
    Real rhs = div * m.invVol[c];
    Real J = 0;
    if (!transportOnly) {
        const Real r = q.rho[c];
        const Real nu = muAt(vv, q.T[c]) / r;
        const Real ntc = t0[c] / r;
        const Real* vc[3] = {q.u, q.v, q.w};
        Real gd[3][3];
        curvVelGradCentre(m, o, c,
                          [&](std::size_t x, int comp) { return vc[comp][x]; },
                          gd);
        const Real w0 = gd[2][1] - gd[1][2], w1 = gd[0][2] - gd[2][0],
                   w2 = gd[1][0] - gd[0][1];
        const Real Omega = sqrtf(w0 * w0 + w1 * w1 + w2 * w2);
        Real gn[3], grho[3];
        curvScalarGradCentre(m, o, c,
                             [&](std::size_t x) { return t0[x] / q.rho[x]; },
                             gn);
        curvScalarGradCentre(m, o, c, [&](std::size_t x) { return q.rho[x]; },
                             grho);
        const RansPoint<Real> P = saSource(ntc, nu, Omega, d[c]);
        const Real gg = gn[0] * gn[0] + gn[1] * gn[1] + gn[2] * gn[2];
        const Real rg = grho[0] * gn[0] + grho[1] * gn[1] + grho[2] * gn[2];
        // TMR's conservative form: the cb2 term carries rho, and moving rho
        // inside the divergence leaves -(K/sigma) grad rho . grad nu~ with K
        // the SAME coefficient the diffusion uses (nu + nu~ fn).
        rhs += r * P.R + Real(sa::cb2 / sa::sigma) * r * gg -
               Real(1.0 / sa::sigma) * saDiffCoef(nu, ntc) * rg;
        J = P.J;
    }
    if constexpr (Src) rhs += src[c];
    if (residual) {
        out0[c] = rhs;
        return;
    }
    const Real comb = cmb.ca * cmb.ta[0][c] + cmb.cb * cmb.tb[0][c];
    if constexpr (!Comp) {
        out0[c] = pointImplicit(comb, cmb.crAt(c), rhs, J, t0[c]);
    } else {
        const Word2 w = internal::stageCombineT(
            cmb, 0, c, pointImplicitDelta(comb, cmb.crAt(c), rhs, J, t0[c]));
        out0[c] = w.hi;
        cmb.otl[0][c] = w.lo;
    }
}

} // namespace

void mutCurv(const GpuGrid& g, ViscView vv, const Real* rho, const Real* T,
             const Real* t0, Real* nut)
{
    kRansMutCurv<<<launchBlocks(g.size()), blockSize>>>(g, vv, rho, T, t0, nut);
    checkLastKernel("kRansMutCurv");
}

void mutMirrorY(const GpuGrid& g, bool lo, bool hi, Real* nut, Real* f1)
{
    if (!lo && !hi) return;
    kMutMirrorY<<<launchBlocks(2 * std::size_t(g.nx) * g.nz), blockSize>>>(
        g, int(lo), int(hi), nut, f1);
    checkLastKernel("kMutMirrorY");
}

void sstMutCurv(const GpuGrid& g, ViscView vv, PrimView q, CurvView m,
                RansModel model, const Real* t0, const Real* t1,
                const Real* d, Real* nut, Real* f1)
{
    kSstMutCurv<<<launchBlocks(g.size()), blockSize>>>(g, vv, q, m, model, t0,
                                                       t1, d, nut, f1);
    checkLastKernel("kSstMutCurv");
}

void sstStageCurv(const GpuGrid& g, ViscView vv, PrimView q, CurvView m,
                  MrfView mz, const StageCombine& cmb, const SstStageArgs& a,
                  bool residual, Real* out0, Real* out1,
                  const ScalarBoundary& sb)
{
    assert(!(sb.any() && a.src0)); // test sources and open faces never meet
    internal::withCompensation(cmb, [&](auto comp) {
        constexpr bool C = decltype(comp)::value;
        const unsigned nb = launchBlocks(g.size());
        if (sb.any())
            kSstStageCurv<C, false, true><<<nb, blockSize>>>(
                g, vv, q, m, mz, cmb, a, int(residual), out0, out1, sb);
        else if (a.src0)
            kSstStageCurv<C, true, false><<<nb, blockSize>>>(
                g, vv, q, m, mz, cmb, a, int(residual), out0, out1, sb);
        else
            kSstStageCurv<C, false, false><<<nb, blockSize>>>(
                g, vv, q, m, mz, cmb, a, int(residual), out0, out1, sb);
    });
    checkLastKernel("kSstStageCurv");
}

void saStageCurv(const GpuGrid& g, ViscView vv, PrimView q, CurvView m,
                 MrfView mz, const StageCombine& cmb, const Real* t0,
                 const Real* d, bool residual, bool transportOnly,
                 const Real* src, Real* out0, const ScalarBoundary& sb)
{
    assert(!(sb.any() && src)); // test sources and open faces never meet
    internal::withCompensation(cmb, [&](auto comp) {
        constexpr bool C = decltype(comp)::value;
        const unsigned nb = launchBlocks(g.size());
        if (sb.any())
            kSaStageCurv<C, false, true><<<nb, blockSize>>>(
                g, vv, q, m, mz, cmb, t0, d, int(residual), int(transportOnly),
                src, out0, sb);
        else if (src)
            kSaStageCurv<C, true, false><<<nb, blockSize>>>(
                g, vv, q, m, mz, cmb, t0, d, int(residual), int(transportOnly),
                src, out0, sb);
        else
            kSaStageCurv<C, false, false><<<nb, blockSize>>>(
                g, vv, q, m, mz, cmb, t0, d, int(residual), int(transportOnly),
                src, out0, sb);
    });
    checkLastKernel("kSaStageCurv");
}

} // namespace dns::gpu::rans
