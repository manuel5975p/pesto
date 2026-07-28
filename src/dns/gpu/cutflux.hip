#include "dns/gpu/cutflux.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/stencil.hpp"
#include "dns/gpu/kernels.hpp"

#include <hip/hip_runtime.h>

namespace dns::gpu {

namespace {

using internal::applyStageTail;
using internal::avg;
using internal::Inv;
using internal::invOf;
using internal::openTol;

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

// The stored face mass fluxes ax*u / ay*v / az*w -- the quantities
// kDivergenceCut differences -- read at an already-resolved linear index.
struct FluxView
{
    const CutView& cv;
    const Real *u, *v, *w;

    __device__ Real x(std::size_t m) const { return cv.ax[m] * u[m]; }
    __device__ Real y(std::size_t m) const { return cv.ay[m] * v[m]; }
    __device__ Real z(std::size_t m) const { return cv.az[m] * w[m]; }
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

// The four apertured cell divergences one stage needs: at the centre cell and
// at each of its three minus-neighbours. This is the kDivergenceCut
// expression, the quantity the projection drives to zero -- the skew
// correction must use THIS divergence in the cut band, since the plain cellDiv
// is nonzero there by design. Each convective correction averages the centre
// with one minus-neighbour, so the centre used to be evaluated three times.
//
// Pre: `o` are the offsets at c. The minus cells differ from c along ONE axis,
// so o's other two offsets stay valid there (index.hpp), and the missing plus
// neighbour along that axis is c itself.
struct DivQuadCut
{
    Real c, xm, ym, zm;
};

__device__ inline DivQuadCut dcutQuad(const FluxView& f, const Offsets& o,
                                      const Inv& iv, std::size_t c)
{
    const std::size_t xm = c + o.m[0], ym = c + o.m[1], zm = c + o.m[2];
    const std::size_t xp = c + o.p[0], yp = c + o.p[1], zp = c + o.p[2];
    return {.c = (f.x(xp) - f.x(c)) * iv.x + (f.y(yp) - f.y(c)) * iv.y
               + (f.z(zp) - f.z(c)) * iv.z,
            .xm = (f.x(c) - f.x(xm)) * iv.x + (f.y(xm + o.p[1]) - f.y(xm)) * iv.y
                + (f.z(xm + o.p[2]) - f.z(xm)) * iv.z,
            .ym = (f.x(ym + o.p[0]) - f.x(ym)) * iv.x + (f.y(c) - f.y(ym)) * iv.y
                + (f.z(ym + o.p[2]) - f.z(ym)) * iv.z,
            .zm = (f.x(zm + o.p[0]) - f.x(zm)) * iv.x
                + (f.y(zm + o.p[1]) - f.y(zm)) * iv.y + (f.z(c) - f.z(zm)) * iv.z};
}

// Apertured skew-symmetric convection at the u-face (i,j,k). Advecting fluxes
// are AVERAGES OF THE STORED FACE MASS FLUXES ax*u / ay*v / az*w -- the same
// quantities kDivergenceCut differences -- never "averaged aperture times
// interpolated velocity"; advected values stay the plain arithmetic means.
// With that pairing the CV's net advecting mass flux telescopes to the mean
// of two projected cell divergences, which the 1/2*u*dcut correction absorbs:
// the operator is exactly skew-symmetric in the lambda~-weighted inner
// product (symmetry-preserving apertured MAC). No hidden dissipation.
__device__ Real convXCut(const FluxView& f, const Offsets& o, const Inv& iv,
                         std::size_t c, const DivQuadCut& d)
{
    const Real* u = f.u;
    const std::size_t xm = c + o.m[0], xp = c + o.p[0];
    const Real Fxp = avg(f.x(c), f.x(xp));
    const Real Fxm = avg(f.x(xm), f.x(c));
    const Real uc  = avg(u[c], u[xp]);
    const Real ucm = avg(u[xm], u[c]);
    const Real Fyp = avg(f.y(xm + o.p[1]), f.y(c + o.p[1]));
    const Real Fym = avg(f.y(xm), f.y(c));
    const Real ujp = avg(u[c], u[c + o.p[1]]);
    const Real ujm = avg(u[c + o.m[1]], u[c]);
    const Real Fzp = avg(f.z(xm + o.p[2]), f.z(c + o.p[2]));
    const Real Fzm = avg(f.z(xm), f.z(c));
    const Real ukp = avg(u[c], u[c + o.p[2]]);
    const Real ukm = avg(u[c + o.m[2]], u[c]);
    const Real flux = fluxDiff(Fxp, uc, Fxm, ucm) * iv.x
                    + fluxDiff(Fyp, ujp, Fym, ujm) * iv.y
                    + fluxDiff(Fzp, ukp, Fzm, ukm) * iv.z;
    const Real corr = Real(0.5) * u[c] * avg(d.xm, d.c);
    const Real lamu = fmax(avg(f.cv.lam[xm], f.cv.lam[c]), lamFloor);
    return (-flux + corr) / lamu;
}

__device__ Real convYCut(const FluxView& f, const Offsets& o, const Inv& iv,
                         std::size_t c, const DivQuadCut& d)
{
    const Real* v = f.v;
    const std::size_t ym = c + o.m[1], yp = c + o.p[1];
    const Real Fxp = avg(f.x(c + o.p[0]), f.x(ym + o.p[0]));
    const Real Fxm = avg(f.x(c), f.x(ym));
    const Real vip = avg(v[c], v[c + o.p[0]]);
    const Real vim = avg(v[c + o.m[0]], v[c]);
    const Real Fyp = avg(f.y(c), f.y(yp));
    const Real Fym = avg(f.y(ym), f.y(c));
    const Real vc  = avg(v[c], v[yp]);
    const Real vcm = avg(v[ym], v[c]);
    const Real Fzp = avg(f.z(ym + o.p[2]), f.z(c + o.p[2]));
    const Real Fzm = avg(f.z(ym), f.z(c));
    const Real vkp = avg(v[c], v[c + o.p[2]]);
    const Real vkm = avg(v[c + o.m[2]], v[c]);
    const Real flux = fluxDiff(Fxp, vip, Fxm, vim) * iv.x
                    + fluxDiff(Fyp, vc, Fym, vcm) * iv.y
                    + fluxDiff(Fzp, vkp, Fzm, vkm) * iv.z;
    const Real corr = Real(0.5) * v[c] * avg(d.ym, d.c);
    const Real lamv = fmax(avg(f.cv.lam[ym], f.cv.lam[c]), lamFloor);
    return (-flux + corr) / lamv;
}

__device__ Real convZCut(const FluxView& f, const Offsets& o, const Inv& iv,
                         std::size_t c, const DivQuadCut& d)
{
    const Real* w = f.w;
    const std::size_t zm = c + o.m[2], zp = c + o.p[2];
    const Real Fxp = avg(f.x(c + o.p[0]), f.x(zm + o.p[0]));
    const Real Fxm = avg(f.x(c), f.x(zm));
    const Real wip = avg(w[c], w[c + o.p[0]]);
    const Real wim = avg(w[c + o.m[0]], w[c]);
    const Real Fyp = avg(f.y(c + o.p[1]), f.y(zm + o.p[1]));
    const Real Fym = avg(f.y(c), f.y(zm));
    const Real wjp = avg(w[c], w[c + o.p[1]]);
    const Real wjm = avg(w[c + o.m[1]], w[c]);
    const Real Fzp = avg(f.z(c), f.z(zp));
    const Real Fzm = avg(f.z(zm), f.z(c));
    const Real wc  = avg(w[c], w[zp]);
    const Real wcm = avg(w[zm], w[c]);
    const Real flux = fluxDiff(Fxp, wip, Fxm, wim) * iv.x
                    + fluxDiff(Fyp, wjp, Fym, wjm) * iv.y
                    + fluxDiff(Fzp, wc, Fzm, wcm) * iv.z;
    const Real corr = Real(0.5) * w[c] * avg(d.zm, d.c);
    const Real lamw = fmax(avg(f.cv.lam[zm], f.cv.lam[c]), lamFloor);
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

// The three apertured Laplacians differ only in which conductances they feed
// in; this is the common body, with the six link conductances and the floored
// control volume already resolved by the caller.
__device__ inline Real lapCutBody(Real cxp, Real cxm, Real cyp, Real cym, Real czp,
                                  Real czm, const Real* q, const Offsets& o,
                                  const Inv& iv, std::size_t c, Real lam)
{
    const std::size_t xp = c + o.p[0], xm = c + o.m[0];
    const std::size_t yp = c + o.p[1], ym = c + o.m[1];
    const std::size_t zp = c + o.p[2], zm = c + o.m[2];
    return (iv.xx * (cxp * (q[xp] - q[c]) - cxm * (q[c] - q[xm]))
          + iv.yy * (cyp * (q[yp] - q[c]) - cym * (q[c] - q[ym]))
          + iv.zz * (czp * (q[zp] - q[c]) - czm * (q[c] - q[zm]))) / lam;
}

__device__ Real lapCutU(const CutView& cv, const Real* q, const Offsets& o,
                        const Inv& iv, std::size_t c)
{
    const std::size_t xp = c + o.p[0], xm = c + o.m[0];
    const std::size_t yp = c + o.p[1], ym = c + o.m[1];
    const std::size_t zp = c + o.p[2], zm = c + o.m[2];
    return lapCutBody(cond(cv.ax[xp], cv.ax[c], cv.ax[xp]),
                      cond(cv.ax[xm], cv.ax[xm], cv.ax[c]),
                      cond(cv.ax[yp], cv.ay[xm + o.p[1]], cv.ay[yp]),
                      cond(cv.ax[ym], cv.ay[xm], cv.ay[c]),
                      cond(cv.ax[zp], cv.az[xm + o.p[2]], cv.az[zp]),
                      cond(cv.ax[zm], cv.az[xm], cv.az[c]), q, o, iv, c,
                      fmax(avg(cv.lam[xm], cv.lam[c]), lamFloor));
}

__device__ Real lapCutV(const CutView& cv, const Real* q, const Offsets& o,
                        const Inv& iv, std::size_t c)
{
    const std::size_t xp = c + o.p[0], xm = c + o.m[0];
    const std::size_t yp = c + o.p[1], ym = c + o.m[1];
    const std::size_t zp = c + o.p[2], zm = c + o.m[2];
    return lapCutBody(cond(cv.ay[xp], cv.ax[ym + o.p[0]], cv.ax[xp]),
                      cond(cv.ay[xm], cv.ax[ym], cv.ax[c]),
                      cond(cv.ay[yp], cv.ay[c], cv.ay[yp]),
                      cond(cv.ay[ym], cv.ay[ym], cv.ay[c]),
                      cond(cv.ay[zp], cv.az[ym + o.p[2]], cv.az[zp]),
                      cond(cv.ay[zm], cv.az[ym], cv.az[c]), q, o, iv, c,
                      fmax(avg(cv.lam[ym], cv.lam[c]), lamFloor));
}

__device__ Real lapCutW(const CutView& cv, const Real* q, const Offsets& o,
                        const Inv& iv, std::size_t c)
{
    const std::size_t xp = c + o.p[0], xm = c + o.m[0];
    const std::size_t yp = c + o.p[1], ym = c + o.m[1];
    const std::size_t zp = c + o.p[2], zm = c + o.m[2];
    return lapCutBody(cond(cv.az[xp], cv.ax[zm + o.p[0]], cv.ax[xp]),
                      cond(cv.az[xm], cv.ax[zm], cv.ax[c]),
                      cond(cv.az[yp], cv.ay[zm + o.p[1]], cv.ay[yp]),
                      cond(cv.az[ym], cv.ay[zm], cv.ay[c]),
                      cond(cv.az[zp], cv.az[c], cv.az[zp]),
                      cond(cv.az[zm], cv.az[zm], cv.az[c]), q, o, iv, c,
                      fmax(avg(cv.lam[zm], cv.lam[c]), lamFloor));
}

// Band overwrite: recompute out = a*x + b*(in + dt*rhs_cut) ONLY where the
// flag is set; every other cell keeps whatever the plain kFusedRhsStage wrote.
// The host wrapper launches the ORIGINAL kernel first over the whole grid and
// this one second -- interior cells therefore carry the plain kernel's exact
// bits BY CONSTRUCTION (same kernel binary), not by hoping two TUs contract
// the same expressions identically (they don't -- measured). Convection reads
// the raw stage field (masked zeros at walls, the energy-stable contract);
// diffusion reads the ghost-filled view. The fringe relaxation A*(in - mean)
// is pointwise and takes no aperture weighting. The StageTail is applied here
// too, so a flagged cell gets exactly the tail the plain kernel gave the rest.
__global__ void kCutFluxOverwrite(GpuGrid g, Real nu, Real A, Real mx, Real my,
                                  Real mz, Real a, const Real* xu, const Real* xv,
                                  const Real* xw, Real b, const Real* __restrict__ u,
                                  const Real* __restrict__ v,
                                  const Real* __restrict__ w,
                                  const Real* __restrict__ du,
                                  const Real* __restrict__ dv,
                                  const Real* __restrict__ dw, CutView cv,
                                  const unsigned char* __restrict__ flag, Real dt,
                                  StageTail tail, Real* ou, Real* ov, Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size() || !flag[c]) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Inv iv = invOf(g);
    const FluxView f{cv, u, v, w};
    const DivQuadCut d = dcutQuad(f, o, iv, c);
    const Real ru = convXCut(f, o, iv, c, d) + nu * lapCutU(cv, du, o, iv, c)
                  + A * (u[c] - mx);
    const Real rv = convYCut(f, o, iv, c, d) + nu * lapCutV(cv, dv, o, iv, c)
                  + A * (v[c] - my);
    const Real rw = convZCut(f, o, iv, c, d) + nu * lapCutW(cv, dw, o, iv, c)
                  + A * (w[c] - mz);
    Real qu = a * xu[c] + b * (u[c] + dt * ru);
    Real qv = a * xv[c] + b * (v[c] + dt * rv);
    Real qw = a * xw[c] + b * (w[c] + dt * rw);
    applyStageTail(tail, o, c, qu, qv, qw);
    ou[c] = qu;
    ov[c] = qv;
    ow[c] = qw;
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
                      Real dt, DeviceField& out, const StageTail& tail)
{
    // Plain kernel first over everything (interior bits BY CONSTRUCTION),
    // then the apertured overwrite on the flag band.
    fusedRhsStage(g, nu, A, mx, my, mz, a, x, b, in, diff, dt, out, tail);
    const CutView cv{cut.ax.data(), cut.ay.data(), cut.az.data(),
                     cut.lambda.data()};
    kCutFluxOverwrite<<<launchBlocks(g.size()), blockSize>>>(
        g, nu, A, mx, my, mz, a, x.u.data(), x.v.data(), x.w.data(), b,
        in.u.data(), in.v.data(), in.w.data(), diff.u.data(), diff.v.data(),
        diff.w.data(), cv, flag.flag.data(), dt, tail, out.u.data(), out.v.data(),
        out.w.data());
    checkLastKernel("cutFluxOverwrite");
}

} // namespace dns::gpu
