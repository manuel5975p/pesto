// RANS kernels of the compressible curvilinear solver (docs/rans.md): eddy
// viscosity, its odd wall mirror, the turbulence-scalar stage, and the
// per-cell local dt. Driven by DeviceComp (gpu/comp.hip); the model algebra is
// dns/rans.hpp, shared with the FP64 oracle (dns/rans_channel1d).
//
// GPU-internal header (HIP TUs only: it speaks the comp_flux.hpp views).
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/gpu/internal/comp_flux.hpp"

namespace dns::gpu::rans {

// Per-cell values the mu_t kernel computes anyway and the scalar stage reads
// back instead of recomputing per face and neighbour (docs/rans.md, "Stage
// cell cache"). The SAME expressions, so the stage is bit-identical:
// phi[e] = t_e / rho (phi[1] SST only), and SST's rate R and grad k . grad
// omega of sstCellGrads. Valid only for the state the mu_t kernel last read.
struct RansCellCache
{
    Real* phi[2] = {nullptr, nullptr};
    Real *rate = nullptr, *gkw = nullptr; // SST only
};

// The primitives u..T of the conserved rho..E (kPrim's expression), then
// nut[c] = saMut(rho, nu~, nu(T)) and phi0[c] = nu~ from them and t0 = rho
// nu~, on every cell in ONE pointwise launch (each cell reads only its own
// primitives, so it is bit for bit the two passes in sequence).
void primMutCurv(const GpuGrid& g, ViscView vv, Real gamma, Real Rgas,
                 const Real* rho, const Real* mx, const Real* my,
                 const Real* mz, const Real* E, const Real* t0, Real* u,
                 Real* v, Real* w, Real* e, Real* p, Real* T, Real* nut,
                 Real* phi0);

// nut[ghost] = -nut[src] on the y ghost rows named: the wall face average is
// then exactly 0.0, so the wall tally and the u_tau fit stay the full wall
// stress. Never computed from ghost scalars. f1 (SST, may be null) mirrors
// EVEN: the face blend at the wall is the first cell's own.
void mutMirrorY(const GpuGrid& g, bool lo, bool hi, Real* nut, Real* f1);

// SST: per cell mu_t = sstMut(...) and F1 into nut / f1, from the live
// primitives, t0 = rho k, t1 = rho omega and the cell-centre gradients
// (grad k . grad omega for CDkw, S or Omega for mu_t), and the whole cache.
// Every cell. Pre: every cache pointer nonnull.
void sstMutCurv(const GpuGrid& g, ViscView vv, PrimView q, CurvView m,
                RansModel model, const Real* t0, const Real* t1,
                const Real* d, Real* nut, Real* f1, const RansCellCache& cache);

// The SST scalar stage (both scalars): upwind convection on the mean mass
// flux, face diffusion (mu + sigma(F1_f) mu_t,f) grad . S with face-averaged
// F1 and mu_t, pointwise sources, point-implicit combination (or the pure
// rhs when residual). The counted positivity guard (k >= 0, omega >= floor
// rho) runs only outside residual mode and counts into *hits the cells the
// fluid mask marks (ghost cells are their mesh family's to overwrite).
struct SstStageArgs
{
    RansModel model;
    const Real *t0, *t1, *nut, *f1, *d;
    const Real* outRho;   // the stage's mean-flow output density (guard)
    Real omegaFloor;      // specific omega floor of the counted guard
    const std::uint8_t* counted; // fluid mask; null = every cell counts
    unsigned long long* hits;
    // TEST ONLY: added to R_k / R_omega before the quotient; null = none.
    const Real *src0, *src1;
    RansCellCache cache; // sstMutCurv's, of the same t0 / t1 / primitives
};
// cells: the cells written (DeviceComp::setStageRange). sb: the open domain
// faces (ScalarBoundary; none = the kernel verbatim).
void sstStageCurv(const GpuGrid& g, ViscView vv, PrimView q, CurvView m,
                  MrfView mz, const StageCombine& cmb, const SstStageArgs& a,
                  bool residual, Real* out0, Real* out1, CellRange cells,
                  const ScalarBoundary& sb = {});

// The SA-neg scalar stage: first-order upwind convection on the mean mass
// flux, face diffusion with face coefficients, pointwise sources and the
// cb2 / grad rho cell terms, combined point-implicitly through cmb (or the
// pure rhs when residual). Writes out0 on the cells of `cells`. src (TEST
// ONLY, may be null) is added to R before the quotient. phi0: primMutCurv's
// nu~ of this t0.
void saStageCurv(const GpuGrid& g, ViscView vv, PrimView q, CurvView m,
                 MrfView mz, const StageCombine& cmb, const Real* t0,
                 const Real* phi0, const Real* d, bool residual,
                 bool transportOnly, const Real* src, Real* out0,
                 CellRange cells, const ScalarBoundary& sb = {});

// After the mean flow's boundary rhs hook (NSCBC) on the immersed path: the
// scalars of every open-face cell are re-carried by the density the mean flow
// actually produced (phi R_rho form; rans.hip kOpenScalarFix). A uniform
// specific scalar then stays exact through NSCBC faces. Pre: sb.any(),
// s.nT >= 1, out the stage output of s under cmb (residual: the pure rhs).
void openScalarConsistency(const GpuGrid& g, const ScalarBoundary& sb,
                           PrimView q, CurvView m, const StageCombine& cmb,
                           bool residual, const DeviceCompState& s,
                           DeviceCompState& out);

// Per-cell explicit diffusive dt bound under RANS, FP64 straight off the
// conserved state (pointwise -- no gradient):
//   dt_c = 0.25 / (D_c amax_c^2),  amax_c = max_a aov_a[c],
//   D_c = max((mu + mu_t)/rho, gamma (mu/Pr + mu_t/Pr_t)/rho,
//             SA: max(nu, nu + nu~ fn(nu~/nu))/sigma).
// SA's mu_t is saMut; SST's needs F2 (a stencil), so its upper bound rho k /
// omega stands in (mu_t = rho a1 k / max(a1 omega, R F2) <= rho k / omega),
// and sigma_k, sigma_w <= 1 leave the k/omega diffusivities under (mu+mu_t)/rho.
// A non-finite cell returns 0, so a min-reduction reads a diverged state as a
// collapsed dt instead of dropping it. RANS-only: the DNS/LES bound in
// stableDt is untouched.
struct RansDiffDt
{
    const Real *rho, *mx, *my, *mz, *E, *t0, *t1; // t1 null for SA
    const Real *ax, *ay, *az; // az == nullptr when nz == 1
    Transport tr;
    double prt;
    __host__ __device__ double operator()(int c) const
    {
        const double r = rho[c];
        const double u = mx[c] / r, v = my[c] / r, w = mz[c] / r;
        const double p =
            (tr.gamma - 1.0) * (double(E[c]) - 0.5 * (u * u + v * v + w * w) * r);
        const double mu = tr.mu(p / (r * tr.Rgas));
        const double nu = mu / r, s0 = double(t0[c]) / r;
        const double mut = t1 ? (s0 > 0.0 ? r * s0 / (double(t1[c]) / r) : 0.0)
                              : saMut(r, s0, nu);
        double D = (mu + mut) / r;
        const double th = tr.gamma * (mu / tr.Pr + mut / prt) / r;
        if (th > D) D = th;
        if (!t1) {
            const double sa = saDtDiffusivity(nu, s0) / sa::sigma;
            if (sa > D) D = sa;
        }
        if (!(D > 0.0)) return isfinite(D) ? HUGE_VAL : 0.0; // inviscid | NaN
        double a = ax[c] > ay[c] ? ax[c] : ay[c];
        if (az && az[c] > a) a = az[c];
        const double dt = 0.25 / (D * a * a);
        return isfinite(dt) && dt > 0.0 ? dt : 0.0;
    }
};

// The same per-cell bound without RANS (mu_t = 0): the laminar --steady
// march. A separate functor, not a null t0 inside RansDiffDt, so the RANS
// path's code (and its FMA contraction) is untouched.
struct LaminarDiffDt
{
    const Real *rho, *mx, *my, *mz, *E;
    const Real *ax, *ay, *az; // az == nullptr when nz == 1
    Transport tr;
    __host__ __device__ double operator()(int c) const
    {
        const double r = rho[c];
        const double u = mx[c] / r, v = my[c] / r, w = mz[c] / r;
        const double p =
            (tr.gamma - 1.0) * (double(E[c]) - 0.5 * (u * u + v * v + w * w) * r);
        const double mu = tr.mu(p / (r * tr.Rgas));
        double D = mu / r;
        const double th = tr.gamma * (mu / tr.Pr) / r;
        if (th > D) D = th;
        if (!(D > 0.0)) return isfinite(D) ? HUGE_VAL : 0.0; // inviscid | NaN
        double a = ax[c] > ay[c] ? ax[c] : ay[c];
        if (az && az[c] > a) a = az[c];
        const double dt = 0.25 / (D * a * a);
        return isfinite(dt) && dt > 0.0 ? dt : 0.0;
    }
};

} // namespace dns::gpu::rans
