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

// nut[c] = saMut(rho, nu~, nu(T)) on every cell, from the live primitives and
// the conserved t0 = rho nu~.
void mutCurv(const GpuGrid& g, ViscView vv, const Real* rho, const Real* T,
             const Real* t0, Real* nut);

// nut[ghost] = -nut[src] on the y ghost rows named: the wall face average is
// then exactly 0.0, so the wall tally and the u_tau fit stay the full wall
// stress. Never computed from ghost scalars. f1 (SST, may be null) mirrors
// EVEN: the face blend at the wall is the first cell's own.
void mutMirrorY(const GpuGrid& g, bool lo, bool hi, Real* nut, Real* f1);

// SST: per cell mu_t = sstMut(...) and F1 into nut / f1, from the live
// primitives, t0 = rho k, t1 = rho omega and the cell-centre gradients
// (grad k . grad omega for CDkw, S or Omega for mu_t). Every cell.
void sstMutCurv(const GpuGrid& g, ViscView vv, PrimView q, CurvView m,
                RansModel model, const Real* t0, const Real* t1,
                const Real* d, Real* nut, Real* f1);

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
};
// sb: the open domain faces (ScalarBoundary; none = the kernel verbatim).
void sstStageCurv(const GpuGrid& g, ViscView vv, PrimView q, CurvView m,
                  MrfView mz, const StageCombine& cmb, const SstStageArgs& a,
                  bool residual, Real* out0, Real* out1,
                  const ScalarBoundary& sb = {});

// The SA-neg scalar stage: first-order upwind convection on the mean mass
// flux, face diffusion with face coefficients, pointwise sources and the
// cb2 / grad rho cell terms, combined point-implicitly through cmb (or the
// pure rhs when residual). Writes out0 on every cell. src (TEST ONLY, may be
// null) is added to R before the quotient.
void saStageCurv(const GpuGrid& g, ViscView vv, PrimView q, CurvView m,
                 MrfView mz, const StageCombine& cmb, const Real* t0,
                 const Real* d, bool residual, bool transportOnly,
                 const Real* src, Real* out0, const ScalarBoundary& sb = {});

// Per-cell explicit diffusive dt bound under RANS, FP64 straight off the
// conserved state (pointwise -- no gradient):
//   dt_c = 0.25 / (D_c amax_c^2),  amax_c = max_a aov_a[c],
//   D_c = max((mu + mu_t)/rho, gamma (mu/Pr + mu_t/Pr_t)/rho,
//             SA: (nu + max(nu~, 0))/sigma).
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
            const double sa = (nu + (s0 > 0.0 ? s0 : 0.0)) / sa::sigma;
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
