#include "dns/gpu/cutflux.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/kernels.hpp"

#include <hip/hip_runtime.h>

namespace dns::gpu {

namespace {

constexpr Real openTol = Real(CutGeometry::openTol);

__device__ inline Real avg(Real a, Real b) { return Real(0.5) * (a + b); }

// Small-cell floor on the control volume ONLY: face fluxes stay exact-
// apertured, but the flux sum divides by max(lambda_face, lamFloor) so an
// explicit RK stage over a nearly-empty cell cannot diverge. At steady state
// the divisor cancels (sum of fluxes = 0 independent of it), so the
// quasi-steady upstream ripple this scheme removes cannot re-enter through
// it; the error is a transient time-lag confined to cells with lambda < 1/2.
// Measured dt penalty on the cylinder/sphere probes: none observed (near-wall
// face velocities are small); hard bound 2x by construction.
constexpr Real lamFloor = Real(0.5);

struct CutView
{
    const Real *ax, *ay, *az, *lam;
};

// Fp*up - Fm*um, written so that identical-argument pairs (the nz==1 wrap:
// Fp==Fm, up==um bitwise) cancel EXACTLY under any FMA contraction. The plain
// form fma(Fp, up, -(Fm*um)) leaves the product-rounding residual, which is
// precisely how a compiler/arch change can break the "w stays exactly 0"
// invariant. Exact-arithmetic-identical to the natural form.
__device__ inline Real fluxDiff(Real Fp, Real up, Real Fm, Real um)
{
    return Fp * (up - um) + (Fp - Fm) * um;
}

// Apertured cell divergence at (a,b,q): the kDivergenceCut expression, the
// quantity the projection drives to zero. The skew correction must use THIS
// divergence in the cut band -- the plain cellDiv is nonzero there by design.
__device__ inline Real dcut(const GpuGrid& g, const CutView& cv, const Real* u,
                            const Real* v, const Real* w, int a, int b, int q)
{
    const std::size_t c = gidx(g, a, b, q);
    const std::size_t xp = gidx(g, a + 1, b, q), yp = gidx(g, a, b + 1, q),
                      zp = gidx(g, a, b, q + 1);
    return (cv.ax[xp] * u[xp] - cv.ax[c] * u[c]) / g.hx
         + (cv.ay[yp] * v[yp] - cv.ay[c] * v[c]) / g.hy
         + (cv.az[zp] * w[zp] - cv.az[c] * w[c]) / g.hz;
}

// Apertured skew-symmetric convection at the u-face (i,j,k). Advecting fluxes
// are AVERAGES OF THE STORED FACE MASS FLUXES ax*u / ay*v / az*w -- the same
// quantities kDivergenceCut differences -- never "averaged aperture times
// interpolated velocity"; advected values stay the plain arithmetic means.
// With that pairing the CV's net advecting mass flux telescopes to the mean
// of two projected cell divergences, which the 1/2*u*dcut correction absorbs:
// the operator is exactly skew-symmetric in the lambda~-weighted inner
// product (symmetry-preserving apertured MAC). No hidden dissipation.
__device__ Real convXCut(const GpuGrid& g, const CutView& cv, const Real* u,
                         const Real* v, const Real* w, std::size_t c,
                         int i, int j, int k)
{
    auto fx = [&](int a, int b, int q) {
        const std::size_t m = gidx(g, a, b, q);
        return cv.ax[m] * u[m];
    };
    auto fy = [&](int a, int b, int q) {
        const std::size_t m = gidx(g, a, b, q);
        return cv.ay[m] * v[m];
    };
    auto fz = [&](int a, int b, int q) {
        const std::size_t m = gidx(g, a, b, q);
        return cv.az[m] * w[m];
    };
    const Real Fxp = avg(fx(i, j, k), fx(i + 1, j, k));
    const Real Fxm = avg(fx(i - 1, j, k), fx(i, j, k));
    const Real uc  = avg(u[c], u[gidx(g, i + 1, j, k)]);
    const Real ucm = avg(u[gidx(g, i - 1, j, k)], u[c]);
    const Real Fyp = avg(fy(i - 1, j + 1, k), fy(i, j + 1, k));
    const Real Fym = avg(fy(i - 1, j, k), fy(i, j, k));
    const Real ujp = avg(u[c], u[gidx(g, i, j + 1, k)]);
    const Real ujm = avg(u[gidx(g, i, j - 1, k)], u[c]);
    const Real Fzp = avg(fz(i - 1, j, k + 1), fz(i, j, k + 1));
    const Real Fzm = avg(fz(i - 1, j, k), fz(i, j, k));
    const Real ukp = avg(u[c], u[gidx(g, i, j, k + 1)]);
    const Real ukm = avg(u[gidx(g, i, j, k - 1)], u[c]);
    const Real flux = fluxDiff(Fxp, uc, Fxm, ucm) / g.hx
                    + fluxDiff(Fyp, ujp, Fym, ujm) / g.hy
                    + fluxDiff(Fzp, ukp, Fzm, ukm) / g.hz;
    const Real corr = Real(0.5) * u[c]
                    * avg(dcut(g, cv, u, v, w, i - 1, j, k),
                          dcut(g, cv, u, v, w, i, j, k));
    const Real lamu = fmax(avg(cv.lam[gidx(g, i - 1, j, k)], cv.lam[c]), lamFloor);
    return (-flux + corr) / lamu;
}

__device__ Real convYCut(const GpuGrid& g, const CutView& cv, const Real* u,
                         const Real* v, const Real* w, std::size_t c,
                         int i, int j, int k)
{
    auto fx = [&](int a, int b, int q) {
        const std::size_t m = gidx(g, a, b, q);
        return cv.ax[m] * u[m];
    };
    auto fy = [&](int a, int b, int q) {
        const std::size_t m = gidx(g, a, b, q);
        return cv.ay[m] * v[m];
    };
    auto fz = [&](int a, int b, int q) {
        const std::size_t m = gidx(g, a, b, q);
        return cv.az[m] * w[m];
    };
    const Real Fxp = avg(fx(i + 1, j, k), fx(i + 1, j - 1, k));
    const Real Fxm = avg(fx(i, j, k), fx(i, j - 1, k));
    const Real vip = avg(v[c], v[gidx(g, i + 1, j, k)]);
    const Real vim = avg(v[gidx(g, i - 1, j, k)], v[c]);
    const Real Fyp = avg(fy(i, j, k), fy(i, j + 1, k));
    const Real Fym = avg(fy(i, j - 1, k), fy(i, j, k));
    const Real vc  = avg(v[c], v[gidx(g, i, j + 1, k)]);
    const Real vcm = avg(v[gidx(g, i, j - 1, k)], v[c]);
    const Real Fzp = avg(fz(i, j - 1, k + 1), fz(i, j, k + 1));
    const Real Fzm = avg(fz(i, j - 1, k), fz(i, j, k));
    const Real vkp = avg(v[c], v[gidx(g, i, j, k + 1)]);
    const Real vkm = avg(v[gidx(g, i, j, k - 1)], v[c]);
    const Real flux = fluxDiff(Fxp, vip, Fxm, vim) / g.hx
                    + fluxDiff(Fyp, vc, Fym, vcm) / g.hy
                    + fluxDiff(Fzp, vkp, Fzm, vkm) / g.hz;
    const Real corr = Real(0.5) * v[c]
                    * avg(dcut(g, cv, u, v, w, i, j - 1, k),
                          dcut(g, cv, u, v, w, i, j, k));
    const Real lamv = fmax(avg(cv.lam[gidx(g, i, j - 1, k)], cv.lam[c]), lamFloor);
    return (-flux + corr) / lamv;
}

__device__ Real convZCut(const GpuGrid& g, const CutView& cv, const Real* u,
                         const Real* v, const Real* w, std::size_t c,
                         int i, int j, int k)
{
    auto fx = [&](int a, int b, int q) {
        const std::size_t m = gidx(g, a, b, q);
        return cv.ax[m] * u[m];
    };
    auto fy = [&](int a, int b, int q) {
        const std::size_t m = gidx(g, a, b, q);
        return cv.ay[m] * v[m];
    };
    auto fz = [&](int a, int b, int q) {
        const std::size_t m = gidx(g, a, b, q);
        return cv.az[m] * w[m];
    };
    const Real Fxp = avg(fx(i + 1, j, k), fx(i + 1, j, k - 1));
    const Real Fxm = avg(fx(i, j, k), fx(i, j, k - 1));
    const Real wip = avg(w[c], w[gidx(g, i + 1, j, k)]);
    const Real wim = avg(w[gidx(g, i - 1, j, k)], w[c]);
    const Real Fyp = avg(fy(i, j + 1, k), fy(i, j + 1, k - 1));
    const Real Fym = avg(fy(i, j, k), fy(i, j, k - 1));
    const Real wjp = avg(w[c], w[gidx(g, i, j + 1, k)]);
    const Real wjm = avg(w[gidx(g, i, j - 1, k)], w[c]);
    const Real Fzp = avg(fz(i, j, k), fz(i, j, k + 1));
    const Real Fzm = avg(fz(i, j, k - 1), fz(i, j, k));
    const Real wc  = avg(w[c], w[gidx(g, i, j, k + 1)]);
    const Real wcm = avg(w[gidx(g, i, j, k - 1)], w[c]);
    const Real flux = fluxDiff(Fxp, wip, Fxm, wim) / g.hx
                    + fluxDiff(Fyp, wjp, Fym, wjm) / g.hy
                    + fluxDiff(Fzp, wc, Fzm, wcm) / g.hz;
    const Real corr = Real(0.5) * w[c]
                    * avg(dcut(g, cv, u, v, w, i, j, k - 1),
                          dcut(g, cv, u, v, w, i, j, k));
    const Real lamw = fmax(avg(cv.lam[gidx(g, i, j, k - 1)], cv.lam[c]), lamFloor);
    return (-flux + corr) / lamw;
}

// Fluid<->fluid diffusive links scale by the averaged aperture of the two
// native faces the link crosses; a link whose NEIGHBOUR face is closed is a
// fluid->ghost link and keeps FULL conductance -- the ghost mirror already
// places the wall at its sub-cell position along that link, and the
// kBodyFriction tally sums exactly those full-conductance links. The two
// cases partition the links, so the wall cannot be double-counted. What is
// omitted is the tangential wall shear on the solid fraction of OPEN faces:
// smooth along the surface (it cannot ring), deferred with its cdf term.
__device__ inline Real cond(Real nbrFace, Real aA, Real aB)
{
    return nbrFace <= openTol ? Real(1) : avg(aA, aB);
}

__device__ Real lapCutU(const GpuGrid& g, const CutView& cv, const Real* q,
                        std::size_t c, int i, int j, int k)
{
    const Real ix = Real(1) / (g.hx * g.hx), iy = Real(1) / (g.hy * g.hy),
               iz = Real(1) / (g.hz * g.hz);
    const std::size_t xp = gidx(g, i + 1, j, k), xm = gidx(g, i - 1, j, k),
                      yp = gidx(g, i, j + 1, k), ym = gidx(g, i, j - 1, k),
                      zp = gidx(g, i, j, k + 1), zm = gidx(g, i, j, k - 1);
    const Real cxp = cond(cv.ax[xp], cv.ax[c], cv.ax[xp]);
    const Real cxm = cond(cv.ax[xm], cv.ax[xm], cv.ax[c]);
    const Real cyp = cond(cv.ax[yp], cv.ay[gidx(g, i - 1, j + 1, k)],
                          cv.ay[gidx(g, i, j + 1, k)]);
    const Real cym = cond(cv.ax[ym], cv.ay[gidx(g, i - 1, j, k)], cv.ay[c]);
    const Real czp = cond(cv.ax[zp], cv.az[gidx(g, i - 1, j, k + 1)],
                          cv.az[gidx(g, i, j, k + 1)]);
    const Real czm = cond(cv.ax[zm], cv.az[gidx(g, i - 1, j, k)], cv.az[c]);
    const Real lamu = fmax(avg(cv.lam[xm], cv.lam[c]), lamFloor);
    return (ix * (cxp * (q[xp] - q[c]) - cxm * (q[c] - q[xm]))
          + iy * (cyp * (q[yp] - q[c]) - cym * (q[c] - q[ym]))
          + iz * (czp * (q[zp] - q[c]) - czm * (q[c] - q[zm]))) / lamu;
}

__device__ Real lapCutV(const GpuGrid& g, const CutView& cv, const Real* q,
                        std::size_t c, int i, int j, int k)
{
    const Real ix = Real(1) / (g.hx * g.hx), iy = Real(1) / (g.hy * g.hy),
               iz = Real(1) / (g.hz * g.hz);
    const std::size_t xp = gidx(g, i + 1, j, k), xm = gidx(g, i - 1, j, k),
                      yp = gidx(g, i, j + 1, k), ym = gidx(g, i, j - 1, k),
                      zp = gidx(g, i, j, k + 1), zm = gidx(g, i, j, k - 1);
    const Real cxp = cond(cv.ay[xp], cv.ax[gidx(g, i + 1, j - 1, k)],
                          cv.ax[gidx(g, i + 1, j, k)]);
    const Real cxm = cond(cv.ay[xm], cv.ax[gidx(g, i, j - 1, k)], cv.ax[c]);
    const Real cyp = cond(cv.ay[yp], cv.ay[c], cv.ay[yp]);
    const Real cym = cond(cv.ay[ym], cv.ay[ym], cv.ay[c]);
    const Real czp = cond(cv.ay[zp], cv.az[gidx(g, i, j - 1, k + 1)],
                          cv.az[gidx(g, i, j, k + 1)]);
    const Real czm = cond(cv.ay[zm], cv.az[gidx(g, i, j - 1, k)], cv.az[c]);
    const Real lamv = fmax(avg(cv.lam[ym], cv.lam[c]), lamFloor);
    return (ix * (cxp * (q[xp] - q[c]) - cxm * (q[c] - q[xm]))
          + iy * (cyp * (q[yp] - q[c]) - cym * (q[c] - q[ym]))
          + iz * (czp * (q[zp] - q[c]) - czm * (q[c] - q[zm]))) / lamv;
}

__device__ Real lapCutW(const GpuGrid& g, const CutView& cv, const Real* q,
                        std::size_t c, int i, int j, int k)
{
    const Real ix = Real(1) / (g.hx * g.hx), iy = Real(1) / (g.hy * g.hy),
               iz = Real(1) / (g.hz * g.hz);
    const std::size_t xp = gidx(g, i + 1, j, k), xm = gidx(g, i - 1, j, k),
                      yp = gidx(g, i, j + 1, k), ym = gidx(g, i, j - 1, k),
                      zp = gidx(g, i, j, k + 1), zm = gidx(g, i, j, k - 1);
    const Real cxp = cond(cv.az[xp], cv.ax[gidx(g, i + 1, j, k - 1)],
                          cv.ax[gidx(g, i + 1, j, k)]);
    const Real cxm = cond(cv.az[xm], cv.ax[gidx(g, i, j, k - 1)], cv.ax[c]);
    const Real cyp = cond(cv.az[yp], cv.ay[gidx(g, i, j + 1, k - 1)],
                          cv.ay[gidx(g, i, j + 1, k)]);
    const Real cym = cond(cv.az[ym], cv.ay[gidx(g, i, j, k - 1)], cv.ay[c]);
    const Real czp = cond(cv.az[zp], cv.az[c], cv.az[zp]);
    const Real czm = cond(cv.az[zm], cv.az[zm], cv.az[c]);
    const Real lamw = fmax(avg(cv.lam[zm], cv.lam[c]), lamFloor);
    return (ix * (cxp * (q[xp] - q[c]) - cxm * (q[c] - q[xm]))
          + iy * (cyp * (q[yp] - q[c]) - cym * (q[c] - q[ym]))
          + iz * (czp * (q[zp] - q[c]) - czm * (q[c] - q[zm]))) / lamw;
}

// Band overwrite: recompute out = a*x + b*(in + dt*rhs_cut) ONLY where the
// flag is set; every other cell keeps whatever the plain kFusedRhsStage wrote.
// The host wrapper launches the ORIGINAL kernel first over the whole grid and
// this one second -- interior cells therefore carry the plain kernel's exact
// bits BY CONSTRUCTION (same kernel binary), not by hoping two TUs contract
// the same expressions identically (they don't -- measured). Convection reads
// the raw stage field (masked zeros at walls, the energy-stable contract);
// diffusion reads the ghost-filled view. The fringe relaxation A*(in - mean)
// is pointwise and takes no aperture weighting.
__global__ void kCutFluxOverwrite(GpuGrid g, Real nu, Real A, Real mx, Real my,
                                  Real mz, Real a, const Real* xu, const Real* xv,
                                  const Real* xw, Real b, const Real* u,
                                  const Real* v, const Real* w, const Real* du,
                                  const Real* dv, const Real* dw, CutView cv,
                                  const unsigned char* flag, Real dt, Real* ou,
                                  Real* ov, Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size() || !flag[c]) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Real ru = convXCut(g, cv, u, v, w, c, i, j, k)
                  + nu * lapCutU(g, cv, du, c, i, j, k) + A * (u[c] - mx);
    const Real rv = convYCut(g, cv, u, v, w, c, i, j, k)
                  + nu * lapCutV(g, cv, dv, c, i, j, k) + A * (v[c] - my);
    const Real rw = convZCut(g, cv, u, v, w, c, i, j, k)
                  + nu * lapCutW(g, cv, dw, c, i, j, k) + A * (w[c] - mz);
    ou[c] = a * xu[c] + b * (u[c] + dt * ru);
    ov[c] = a * xv[c] + b * (v[c] + dt * rv);
    ow[c] = a * xw[c] + b * (w[c] + dt * rw);
}

} // namespace

std::expected<DeviceCutFlag, hipError_t>
DeviceCutFlag::make(const std::vector<unsigned char>& hostFlag)
{
    auto buf = DeviceBuffer<unsigned char>::make(hostFlag.size());
    if (!buf) return std::unexpected(buf.error());
    if (auto e = buf->upload(hostFlag.data()); e != hipSuccess)
        return std::unexpected(e);
    return DeviceCutFlag{std::move(*buf)};
}

void fusedRhsStageCut(const GpuGrid& g, Real nu, Real A, Real mx, Real my,
                      Real mz, Real a, const DeviceField& x, Real b,
                      const DeviceField& in, const DeviceField& diff,
                      const DeviceCutGeometry& cut, const DeviceCutFlag& flag,
                      Real dt, DeviceField& out)
{
    // Plain kernel first over everything (interior bits BY CONSTRUCTION),
    // then the apertured overwrite on the flag band.
    fusedRhsStage(g, nu, A, mx, my, mz, a, x, b, in, diff, dt, out);
    const CutView cv{cut.ax.data(), cut.ay.data(), cut.az.data(),
                     cut.lambda.data()};
    kCutFluxOverwrite<<<launchBlocks(g.size()), blockSize>>>(
        g, nu, A, mx, my, mz, a, x.u.data(), x.v.data(), x.w.data(), b,
        in.u.data(), in.v.data(), in.w.data(), diff.u.data(), diff.v.data(),
        diff.w.data(), cv, flag.flag.data(), dt, out.u.data(), out.v.data(),
        out.w.data());
    checkLastKernel("cutFluxOverwrite");
}

} // namespace dns::gpu
