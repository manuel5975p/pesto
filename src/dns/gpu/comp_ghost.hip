#include "dns/gpu/comp_ghost.hpp"

#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/internal/reduce.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/rans.hpp"

#include <algorithm> // std::max(initializer_list): NOT in <bits/stl_algobase.h>
#include <cassert>
#include <cmath>
#include <vector>

namespace dns::gpu {
namespace {

// Fill ghost cells from a sample point ds beyond the wall along the normal:
// velocity linearly extrapolated through the no-slip surface (u_g =
// -(dw/ds) u_sample; exact mirror when dw >= ds). Thermodynamics: adiabatic
// (iso == 0) copies pressure and density, a zero wall-normal dT/dn; isothermal
// images the temperature about the wall face (see below) and rebuilds rho from
// the EOS. rho and p come from the THERMO stencil (tnbr/tw), which is the
// velocity stencil wherever that one is all fluid and an on-normal sample a
// little further out where it is not (classifyCompGhost); where the two
// coincide, every ghost value is bit-identical to the single-stencil fill.
//
// uw != nullptr makes the wall MOVE at that per-ghost velocity. The image is
// the same one the isothermal temperature uses, about u_wall instead of about
// rest: with s the outward wall-normal coordinate, u(s) = Uw + (u_s - Uw)
// s/dsmp, so u(-dw) = (1 + a) Uw - a u_s. Uw = 0 recovers the resting mirror
// and a == 1 makes it the familiar 2 Uw - u_s. The resting branch is kept
// VERBATIM rather than folded into the general form: a*a*(ui*ui) and
// (a*ui)*(a*ui) are not the same float, and the static path must not move.
//
// WALL PRESSURE. The pressure is not a boundary condition of a no-slip wall --
// only u (and T or q) are -- so the ghost pressure is a closure, and the right
// closure is the one the solution itself satisfies at the wall: the momentum
// equation there, where the fluid moves with the wall (Du/Dt = a_wall),
//   grad p |wall = -rho a_wall + div tau.
// The inviscid part is imaged exactly: p_ghost = p_sample + rho_sample *
// a_wall . (x_sample - x_ghost) (`as`, per ghost). At rest a_wall = 0 and that
// is the historical copy, so the resting branch is the same condition, not an
// approximation of it. The viscous part is NOT imaged, deliberately: it is the
// normal half of the wall's viscous compatibility, whose tangential half
// (mu d2u_t/dn2 = dp/ds) the linear velocity image omits at the same order --
// imaging one half does not raise the order of the force, and imaging both
// needs a curvature-carrying (two-sample) image. Such an image has non-convex
// weights, and for the pressure the dissipation-free central flux makes it
// unstable outright: in the 1D semi-discrete acoustic operator with this
// image, swept over the sub-cell wall offset, a wall-bound mode grows at
// Re(lambda) = 0.58 c/h with one cell between the two samples and 0.009 c/h
// with two, while the copy -- from any distance -- stays at Re(lambda) = 0 to
// round-off. A pressure-only extrapolation is therefore not offered.
// The adiabatic image keeps T, not rho: rho_ghost = rho_sample p_g / p_s, so
// dT/dn stays zero and the gradient lives in the density, as in the gas.
__global__ void kGhostFill(Real gamma, int iso, Real Twall, Real Rgas,
                           const int* gcell, const int* gnbr, const Real* gw,
                           const int* tnbr, const Real* tw, const Real* gscale,
                           const Real* uw, const Real* as, int m, Real* rho,
                           Real* mx, Real* my, Real* mz, Real* E)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
    // Velocity sample (and its rho, p: the isothermal T image's sample).
    Real ri = 0, ui = 0, vi = 0, wi = 0, pi = 0;
    for (int s = 0; s < 8; ++s) {
        const Real w = gw[8 * q + s];
        if (w == Real(0)) continue;
        const int c = gnbr[8 * q + s];
        const Real r = rho[c];
        ri += w * r;
        ui += w * mx[c] / r;
        vi += w * my[c] / r;
        wi += w * mz[c] / r;
        pi += w * (gamma - Real(1)) *
              (E[c] -
               Real(0.5) * (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / r);
    }
    // Thermo sample: rho and p on the normal. Same arithmetic as above, so a
    // coinciding stencil gives rt == ri and pt == pi bit for bit.
    Real rt = 0, pt = 0;
    for (int s = 0; s < 8; ++s) {
        const Real w = tw[8 * q + s];
        if (w == Real(0)) continue;
        const int c = tnbr[8 * q + s];
        const Real r = rho[c];
        rt += w * r;
        pt += w * (gamma - Real(1)) *
              (E[c] -
               Real(0.5) * (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / r);
    }
    const Real a = gscale[q];
    const int c = gcell[q];
    if (uw) {
        const Real pg = pt + rt * as[q];
        Real rg = rt * (pg / pt);
        if (iso) {
            // The resting branch's floored temperature image, below.
            const Real Ts = pi / (Rgas * ri);
            const Real Tg = fmaxf(Twall + a * (Twall - Ts), Real(0.05) * Twall);
            rg = pg / (Rgas * Tg);
        }
        rho[c] = rg;
        const Real b = Real(1) + a;
        const Real ug = b * uw[3 * q + 0] - a * ui;
        const Real vg = b * uw[3 * q + 1] - a * vi;
        const Real wg = b * uw[3 * q + 2] - a * wi;
        mx[c] = rg * ug;
        my[c] = rg * vg;
        mz[c] = rg * wg;
        E[c] = pg / (gamma - Real(1)) +
               Real(0.5) * rg * (ug * ug + vg * vg + wg * wg);
        return;
    }
    // Isothermal image of the temperature. Let s be the outward wall-normal
    // coordinate with s = 0 on the wall face: the ghost centre sits at
    // s = -dw and the sample point at s = +dsmp, and a = gscale = dw/dsmp is
    // the SAME ratio the velocity mirror above uses. The linear profile through
    // the two known values T(0) = Twall and T(dsmp) = T_sample is
    //   T(s) = Twall + (T_sample - Twall) * s / dsmp,
    // so at the ghost centre
    //   T_ghost = T(-dw) = Twall - (dw/dsmp) (T_sample - Twall)
    //                    = Twall + a (Twall - T_sample).
    // a == 1 -- ghost and sample equidistant from the face -- is the channel
    // wall's 2 Twall - T_sample, i.e. that mirror is this image's special case.
    // Pressure is still copied (zero wall-normal dp/dn, from the thermo sample)
    // and density follows from p = rho R T, so the ghost state is
    // thermodynamically consistent.
    Real rg = rt;
    if (iso) {
        const Real Ts = pi / (Rgas * ri);
        // Floor: a transient with T_sample > (1 + 1/a) Twall makes the exact
        // image non-positive and rho = p/(R T) would come out negative -- an
        // unrecoverable state. Clipping at 0.05 Twall (the channel wall's
        // constant) caps the wall-normal gradient instead of imposing it: while
        // it is active the wall UNDER-cools, the face temperature the image
        // realizes being Twall + (T_floor - (Twall + a (Twall - T_sample)))
        // / (1 + a) instead of Twall. It is a transient guard, not a boundary
        // condition; a run that keeps hitting it is not resolving the thermal
        // layer and its wall heat flux is not trustworthy.
        const Real Tg = fmaxf(Twall + a * (Twall - Ts), Real(0.05) * Twall);
        rg = pt / (Rgas * Tg);
    }
    rho[c] = rg;
    mx[c] = -a * rg * ui;
    my[c] = -a * rg * vi;
    mz[c] = -a * rg * wi;
    E[c] = pt / (gamma - Real(1)) +
           Real(0.5) * a * a * rg * (ui * ui + vi * vi + wi * wi);
}

// RANS scalar images of the ghosts (DeviceCompGhost::applyTurb): the specific
// scalars sampled through the velocity stencil, imaged like the velocity
// about a wall value (0 for nu~ and k, omega_w for omega), stored with the
// ghost density kGhostFill wrote. omega_w = 60 nu_w / (beta1 d_s^2), d_s =
// dw / a the sample's wall distance, nu_w = mu(T_s) / rho_s of the thermo
// sample.
__global__ void kGhostTurb(int sst, ViscView vv, Real gamma, Real Rgas,
                           const int* gcell, const int* gnbr, const Real* gw,
                           const int* tnbr, const Real* tw, const Real* gscale,
                           const Real* gdw, int m, const Real* rho,
                           const Real* mx, const Real* my, const Real* mz,
                           const Real* E, Real* t0, Real* t1)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
    Real s0 = 0, s1 = 0;
    for (int s = 0; s < 8; ++s) {
        const Real w = gw[8 * q + s];
        if (w == Real(0)) continue;
        const int c = gnbr[8 * q + s];
        s0 += w * t0[c] / rho[c];
        if (sst) s1 += w * t1[c] / rho[c];
    }
    const Real a = gscale[q];
    const int c = gcell[q];
    const Real rg = rho[c];
    t0[c] = rg * (-a * s0);
    if (!sst) return;
    Real rt = 0, pt = 0;
    for (int s = 0; s < 8; ++s) {
        const Real w = tw[8 * q + s];
        if (w == Real(0)) continue;
        const int x = tnbr[8 * q + s];
        const Real r = rho[x];
        rt += w * r;
        pt += w * (gamma - Real(1)) *
              (E[x] -
               Real(0.5) * (mx[x] * mx[x] + my[x] * my[x] + mz[x] * mz[x]) / r);
    }
    const Real Ts = pt / (Rgas * rt);
    const Real muw = vv.sutherland ? vv.mu * vv.ratio(Ts) : vv.mu;
    const Real ds = gdw[q] / a;
    const Real omw = Real(60) * (muw / rt) / (Real(sst::beta1) * ds * ds);
    t1[c] = rg * ((Real(1) + a) * omw - a * s1);
}

// Deep cells under RANS: nu~ = k = 0, omega = omega_inf (density as filled).
__global__ void kDeepTurb(const int* cells, int m, int sst, Real omegaInf,
                          const Real* rho, Real* t0, Real* t1)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
    const int c = cells[q];
    t0[c] = Real(0);
    if (sst) t1[c] = rho[c] * omegaInf;
}

// Zero the low words of the listed cells, every field (compensated state).
struct LowWords
{
    Real* lo[7];
    int n;
};
__global__ void kZeroLow(const int* cells, int m, LowWords L)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
    const int c = cells[q];
    for (int f = 0; f < L.n; ++f) L.lo[f][c] = Real(0);
}

// mu_t odd and F1 even through the ghost image (mirrorMut); deep: 0 and 1.
__global__ void kGhostMut(const int* gcell, const int* gnbr, const Real* gw,
                          const Real* gscale, int m, Real* nut, Real* f1)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
    Real sm = 0, sf = 0;
    for (int s = 0; s < 8; ++s) {
        const Real w = gw[8 * q + s];
        if (w == Real(0)) continue;
        const int c = gnbr[8 * q + s];
        sm += w * nut[c];
        if (f1) sf += w * f1[c];
    }
    const int c = gcell[q];
    nut[c] = -gscale[q] * sm;
    if (f1) f1[c] = sf;
}
__global__ void kDeepMut(const int* cells, int m, Real* nut, Real* f1)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
    nut[cells[q]] = Real(0);
    if (f1) f1[cells[q]] = Real(1);
}

// First-cell y+ of every interface face (DeviceCompGhost::wallYplus).
__global__ void kWallYplus(ViscView vv, PrimView q, const Real* d,
                           const int* fcell, int m, double* out)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m) return;
    const int c = fcell[t];
    const double u = q.u[c], v = q.v[c], w = q.w[c];
    const double mu =
        vv.sutherland ? double(vv.mu) * double(vv.ratio(q.T[c])) : double(vv.mu);
    const double nu = mu / double(q.rho[c]);
    out[t] = sqrt(sqrt(u * u + v * v + w * w) * double(d[c]) / nu);
}

struct SpanMax
{
    const double* v;
    __host__ __device__ double operator()(int i) const { return v[i]; }
};

__global__ void kQuiescent(const int* cells, int m, Real rhoInf, Real EInf,
                           Real* rho, Real* mx, Real* my, Real* mz, Real* E)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
    const int c = cells[q];
    rho[c] = rhoInf;
    mx[c] = Real(0);
    my[c] = Real(0);
    mz[c] = Real(0);
    E[c] = EInf;
}

// Fourier (conduction) part of the face energy flux, in the sign convention
// faceFlux uses -- it enters f.E as -kappaFace dT/dn, with kappaFace the SAME
// face-averaged, Sutherland-scaled value the flux evaluates. Recomputed here
// instead of read off f.E because the wall tally must report conduction ALONE:
// f.E also carries the convective and viscous-work terms, which on this
// staircase wall are the mirror's own imperfection rather than a heat flux, and
// they do not vanish on an adiabatic wall (measured; they made the adiabatic
// column read a spurious nonzero rate). mu == 0 makes kappa == 0, so the
// inviscid early-out of faceFlux is reproduced without a branch.
__device__ inline Real faceConduction(const GpuGrid& g, ViscView vv, PrimView q,
                                      int i, int j, int k, int axis)
{
    const Offsets o = offsetsAt(g, i, j, k);
    const std::size_t a = gidx(g, i, j, k), b = a + o.p[axis];
    const Real h = axis == 0 ? g.hx : axis == 1 ? g.hy : g.hz;
    Real kapf = vv.kappa;
    if (vv.sutherland) kapf *= vv.ratio(Real(0.5) * (q.T[a] + q.T[b]));
    return -kapf * (q.T[b] - q.T[a]) / h;
}

// Curvilinear twin: the temperature gradient in index space mapped through the
// face inverse Jacobian and contracted with the face area vector, exactly as
// faceFluxCurv builds its Fourier term (so the result already carries area).
__device__ inline Real faceConductionCurv(const GpuGrid& g, ViscView vv,
                                          PrimView q, CurvView m, int i, int j,
                                          int k, int axis)
{
    const Offsets o = offsetsAt(g, i, j, k);
    const std::size_t a = gidx(g, i, j, k), b = a + o.p[axis];
    const Real Sv[3] = {m.S[axis][0][a], m.S[axis][1][a], m.S[axis][2][a]};
    Real dxi[3];
    dxi[axis] = q.T[b] - q.T[a];
    for (int d = 0; d < 3; ++d) {
        if (d == axis) continue;
        dxi[d] = Real(0.25) * (q.T[a + o.p[d]] - q.T[a + o.m[d]] +
                               q.T[b + o.p[d]] - q.T[b + o.m[d]]);
    }
    Real gT[3];
    for (int c = 0; c < 3; ++c)
        gT[c] = dxi[0] * m.Ji[axis][0 + c][a] + dxi[1] * m.Ji[axis][3 + c][a] +
                dxi[2] * m.Ji[axis][6 + c][a];
    Real kapf = vv.kappa;
    if (vv.sutherland) kapf *= vv.ratio(Real(0.5) * (q.T[a] + q.T[b]));
    return -kapf * (gT[0] * Sv[0] + gT[1] * Sv[1] + gT[2] * Sv[2]);
}

// Exact discrete momentum flux fluid -> body: the same faceFlux the rhs uses,
// summed over fluid/ghost interface faces, oriented into the body. The fourth
// staged span is the wall heat rate: the conduction through the same faces on
// an isothermal wall, and EXACTLY zero on an adiabatic one.
//
// The adiabatic gate is not a shortcut. That wall's image sets the wall-normal
// dT/dn to zero by construction, so its wall heat flux is zero by definition of
// the boundary condition; what the interface faces still carry there is the
// mirror band's own second-order offset (the ghost holds the SAMPLE point's
// temperature, one cell out, not the fluid neighbour's) flowing into cells that
// the next stage overwrites -- an artifact, not a flux. Measured on the Re100
// M0.2 cylinder at t = 2: 2.8e-2, against 6.8 for a Twall = 1.5 Tinf wall on
// the same rig. That same artifact does ride inside the isothermal number at
// roughly the half-percent level, which is this tally's accuracy -- far inside
// the staircase wall's own error, and the price of keeping the reported column
// readable: nonzero means flux, never offset.
// mz (--rotor-frame mrf): the SAME zone view the rhs used, so the tallied
// momentum flux is exactly the solver's -- the interface faces sit deep in
// the f = 1 core, where the advecting normal velocity is the wall-relative
// one and vanishes on the resolved surface.
__global__ void kBodyFlux(GpuGrid g, ViscView vv, int iso, PrimView q,
                          const int* fcell, const int* fdir, int m, double* acc,
                          MrfView mz, const Real* nut, Real kapScale)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m) return;
    int i, j, k;
    decompose(g, std::size_t(fcell[t]), i, j, k);
    const int d = fdir[t];
    const int axis = d / 2;
    const bool minus = d & 1;
    // Owning cell of the face: the fluid cell itself for a +face, its back
    // neighbour for a -face.
    const int fi = i - (minus ? (axis == 0) : 0),
              fj = j - (minus ? (axis == 1) : 0),
              fk = k - (minus ? (axis == 2) : 0);
    const Flux f = faceFlux(g, vv, q, fi, fj, fk, axis, nut, kapScale,
                            mz.active ? &mz : nullptr);
    const Real area = axis == 0 ? g.hy * g.hz
                    : axis == 1 ? g.hx * g.hz
                                : g.hx * g.hy;
    const Real sgn = minus ? Real(-1) : Real(1);
    // Staged per face, not atomically accumulated: the reduction below is
    // deterministic and FP64 (see DeviceCompGhost::facc_).
    acc[t] = double(sgn * f.mx * area);
    acc[m + t] = double(sgn * f.my * area);
    acc[2 * m + t] = double(sgn * f.mz * area);
    acc[3 * m + t] =
        iso ? double(sgn * faceConduction(g, vv, q, fi, fj, fk, axis) * area)
            : 0.0;
}

// Curvilinear counterpart: faceFluxCurv already integrates over the exact
// face area vector S, so there is no separate area multiply here.
__global__ void kBodyFluxCurv(GpuGrid g, ViscView vv, int iso, PrimView q,
                              CurvView m, const int* fcell, const int* fdir,
                              int n, double* acc, MrfView mz, const Real* nut,
                              Real kapScale)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int i, j, k;
    decompose(g, std::size_t(fcell[t]), i, j, k);
    const int d = fdir[t];
    const int axis = d / 2;
    const bool minus = d & 1;
    const int fi = i - (minus ? (axis == 0) : 0),
              fj = j - (minus ? (axis == 1) : 0),
              fk = k - (minus ? (axis == 2) : 0);
    const Flux f = faceFluxCurv(g, vv, q, m, fi, fj, fk, axis, nut, kapScale,
                                mz.active ? &mz : nullptr);
    const Real sgn = minus ? Real(-1) : Real(1);
    acc[t] = double(sgn * f.mx);
    acc[n + t] = double(sgn * f.my);
    acc[2 * n + t] = double(sgn * f.mz);
    acc[3 * n + t] =
        iso ? double(sgn * faceConductionCurv(g, vv, q, m, fi, fj, fk, axis))
            : 0.0;
}

// Moment of the force already staged by kBodyFlux/kBodyFluxCurv, about `pivot`:
// spans 4..6 of the same buffer, from spans 0..2 and the face positions. A
// separate pass on purpose -- it reads the STAGED values, so the reported
// torque is the moment of exactly the reported drag, face for face, on either
// mesh and with no second flux evaluation to drift from the first.
__global__ void kBodyMoment(const Real* fpos, double px, double py, double pz,
                            int m, double* acc)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m) return;
    const double rx = double(fpos[3 * t + 0]) - px;
    const double ry = double(fpos[3 * t + 1]) - py;
    const double rz = double(fpos[3 * t + 2]) - pz;
    const double fx = acc[t], fy = acc[m + t], fz = acc[2 * m + t];
    acc[4 * m + t] = ry * fz - rz * fy;
    acc[5 * m + t] = rz * fx - rx * fz;
    acc[6 * m + t] = rx * fy - ry * fx;
}

} // namespace

std::expected<DeviceCompGhost, hipError_t> DeviceCompGhost::make(
    const Grid& g, const Sdf& sdf, double gamma, double rhoInf, double pInf,
    const double* bbox, WallThermal wall, const WallMotion* motion)
{
    const double L[3] = {g.nx * g.hx, g.ny * g.hy, g.nz * g.hz};
    return make(g, sdf, gamma, rhoInf, pInf, bbox,
                AxisRemap3{Focal1D::identity(L[0]), Focal1D::identity(L[1]),
                           Focal1D::identity(L[2])},
                wall, motion);
}

void CompGhostCapacity::cover(const CompGhostTables& t)
{
    nGhost = std::max(nGhost, t.nGhost());
    nDeep = std::max(nDeep, t.nDeep());
    nFace = std::max(nFace, t.nFace());
}

CompGhostTables classifyCompGhost(const Grid& g, const GhostSdf& sdf,
                                  const double* bbox, const AxisRemap3& R,
                                  const GhostWallMotion* motion)
{
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const double hx = g.hx, hy = g.hy, hz = g.hz;
    const Focal1D* ax[3] = {&R.fx, &R.fy, &R.fz};
    const double href[3] = {hx, hy, hz};

    // Physical cell centre of the reference centre (i+0.5) h; the identity
    // remap returns the reference position unchanged, so the Cartesian path
    // below is arithmetically untouched.
    auto cx = [&](int i) { return R.fx.map((i + 0.5) * hx); };
    auto cy = [&](int j) { return R.fy.map((j + 0.5) * hy); };
    auto cz = [&](int k) { return R.fz.map((k + 0.5) * hz); };

    // Local physical spacing per axis at a cell: the map's slope times the
    // reference spacing. Degenerate axes report 0 and are skipped by every
    // consumer. These are physical lengths -- the near-wall band, the
    // deep-solid threshold and the sample distance are all physical.
    auto localSpacing = [&](int i, int j, int k, double d[3]) {
        const int idx[3] = {i, j, k};
        const int n[3] = {nx, ny, nz};
        for (int a = 0; a < 3; ++a)
            d[a] = n[a] == 1
                       ? 0.0
                       : ax[a]->deriv((idx[a] + 0.5) * href[a]) * href[a];
    };
    auto minSpacing = [](const double d[3]) {
        double m = 1e300;
        for (int a = 0; a < 3; ++a)
            if (d[a] > 0.0) m = std::min(m, d[a]);
        return m;
    };
    auto maxSpacing = [](const double d[3]) {
        double m = 0.0;
        for (int a = 0; a < 3; ++a) m = std::max(m, d[a]);
        return m;
    };
    // Coarsest spacing anywhere on the mesh. The bbox short-circuit below
    // needs an UPPER bound on any cell's near-wall reach, so it is the
    // coarse spacing -- not the fine one -- that sets the band.
    double hCoarse = 0.0;
    for (int a = 0; a < 3; ++a) {
        const int n[3] = {nx, ny, nz};
        if (n[a] == 1) continue;
        hCoarse = std::max(hCoarse, ax[a]->active ? ax[a]->c * href[a]
                                                  : href[a]);
    }
    const double Lx = nx * hx, Ly = ny * hy, Lz = nz * hz;
    auto pmod = [](double x, double L) { return x - L * std::floor(x / L); };

    // A cell farther than this from the body's bounding box is provably farther
    // than that from the surface (the box contains the surface), hence phi > 2h:
    // plain fluid, never a ghost / near-wall cell / interface neighbour, so its
    // exact sdf value is irrelevant. Margin covers the -2h deep band + a cell.
    // On a focal mesh the coarse spacing bounds every cell's near-wall reach,
    // so it -- not the fine one -- keeps this an upper bound.
    const double band = 2.0 * hCoarse + std::max({hx, hy, hz, hCoarse});
    const double outer = band + 1.0; // any value > band marks a skipped fluid cell
    auto farFromBox = [&](double x, double y, double z) {
        double d2 = 0.0;
        const double p[3] = {x, y, z};
        for (int a = 0; a < 3; ++a) {
            const double lo = bbox[a], hi = bbox[a + 3];
            const double e = p[a] < lo ? lo - p[a] : p[a] > hi ? p[a] - hi : 0.0;
            d2 += e * e;
        }
        return d2 > band * band;
    };

    std::vector<double> phi(g.size());
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double x = cx(i), y = cy(j), z = cz(k);
                phi[g.idx(i, j, k)] =
                    (bbox && farFromBox(x, y, z)) ? outer : sdf(x, y, z);
            }

    // The flux stencil of a fluid cell: its face neighbours and its EDGE
    // neighbours (faceFlux reads the normal pair plus the tangential averages
    // of both; no corner). Every solid cell in it is read by the fluid's own
    // update, so it must carry the wall image -- a deep cell there is a
    // resting wall at pInf the fluid feels and, on a face, a piece of the
    // surface the tally never reads (an OPEN surface: measured on a 3-blade
    // prop mesh, 16% of the wetted faces and a spurious CT of +0.025 from pInf
    // alone). At nz == 1 the z steps wrap onto the cell itself and are skipped.
    std::vector<std::array<int, 3>> reach;
    for (int dk = (nz == 1 ? 0 : -1); dk <= (nz == 1 ? 0 : 1); ++dk)
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di) {
                const int nnz = (di != 0) + (dj != 0) + (dk != 0);
                if (nnz == 1 || nnz == 2) reach.push_back({di, dj, dk});
            }
    auto inFluidReach = [&](int i, int j, int k) {
        return std::ranges::any_of(reach, [&](const std::array<int, 3>& s) {
            return phi[g.idx(i + s[0], j + s[1], k + s[2])] >= 0.0;
        });
    };

    CompGhostTables T;
    auto& gcell = T.gcell;
    auto& gnbr = T.gnbr;
    auto& deep = T.deep;
    auto& fcell = T.fcell;
    auto& fdir = T.fdir;
    auto& solid = T.solid;
    auto& gw = T.gw;
    auto& tnbr = T.tnbr;
    auto& tw = T.tw;
    auto& gscale = T.gscale;
    auto& gdw = T.gdw;
    auto& uw = T.uwall;
    auto& aspan = T.aspan;
    auto& fpos = T.fpos;
    std::vector<char> isGhost(g.size(), 0);

    // One trilinear stencil: 8 source cells, their weights (fluid cells only;
    // wsum their sum, normalized at push), and the physical point the
    // renormalized weights actually sample, as an offset from the ghost.
    struct Stencil
    {
        int nb[8];
        double w[8];
        double wsum = 0.0, wphi = 0.0;
        double span[3];
        bool dropped = false; // a positive-weight corner was solid
    };
    // One ghost's image: the velocity stencil, the extrapolation ratio
    // a = dw/dsmp, the wall point where the surface kinematics are sampled,
    // and the THERMODYNAMIC stencil (rho, p) -- see normalImage.
    struct Image
    {
        Stencil v, t;
        double a;
        double wall[3];
    };
    const double Lbox[3] = {Lx, Ly, Lz};
    const int nbox[3] = {nx, ny, nz};
    // Physical centre of cell `i` on axis `ax`, unwrapped to lie within half a
    // box of `near` (the sample point the stencil was built around).
    auto centreNear = [&](int ax, int i, double near) {
        const double c = ax == 0 ? cx(i) : ax == 1 ? cy(i) : cz(i);
        const double d = c - near;
        return nbox[ax] == 1 ? near
               : d > 0.5 * Lbox[ax] ? c - Lbox[ax]
               : d < -0.5 * Lbox[ax] ? c + Lbox[ax] : c;
    };

    // Trilinear stencil at the point `dist` from the ghost centre (x, y, z)
    // along the unit normal nv. The point is a PHYSICAL offset along the
    // physical normal; the per-axis inverse takes it back to reference space,
    // where the cell-centre lattice is uniform and the trilinear weights are
    // the Cartesian ones. The map is box-preserving, so the physical wrap is
    // the reference wrap. Solid corners get weight 0 (renormalized at push).
    auto stencilAt = [&](double x, double y, double z, const double nv[3],
                         double dist, Stencil& st) {
        const double xs[3] = {x + dist * nv[0], y + dist * nv[1],
                              z + dist * nv[2]};
        const double xi = R.fx.inverse(pmod(xs[0], Lx));
        const double yi = R.fy.inverse(pmod(xs[1], Ly));
        const double zi = R.fz.inverse(pmod(xs[2], Lz));
        const double fx = xi / hx - 0.5, fy = yi / hy - 0.5, fz = zi / hz - 0.5;
        const int i0 = int(std::floor(fx)), j0 = int(std::floor(fy)),
                  k0 = int(std::floor(fz));
        const double ax = fx - i0, ay = fy - j0, az = fz - k0;
        st.wsum = st.wphi = 0.0;
        st.dropped = false;
        double wx[3] = {0.0, 0.0, 0.0};
        for (int s = 0; s < 8; ++s) {
            const int sx = s & 1, sy = (s >> 1) & 1, sz = s >> 2;
            const int ii = (i0 + sx + nx) % nx, jj = (j0 + sy + ny) % ny,
                      kk = ((k0 + sz) % nz + nz) % nz;
            const std::size_t cc = g.idx(ii, jj, kk);
            const double ww =
                (sx ? ax : 1 - ax) * (sy ? ay : 1 - ay) * (sz ? az : 1 - az);
            st.nb[s] = int(cc);
            st.w[s] = phi[cc] > 0.0 ? ww : 0.0;
            st.dropped = st.dropped || (ww > 0.0 && phi[cc] <= 0.0);
            st.wsum += st.w[s];
            st.wphi += st.w[s] * phi[cc];
            wx[0] += st.w[s] * centreNear(0, ii, xs[0]);
            wx[1] += st.w[s] * centreNear(1, jj, xs[1]);
            wx[2] += st.w[s] * centreNear(2, kk, xs[2]);
        }
        // The point the renormalized weights sample: the requested point for a
        // full stencil (up to the map's curvature on a focal mesh), the fluid
        // cells' weighted centre -- OFF the normal -- when corners dropped.
        if (st.wsum > 0.0) {
            st.span[0] = wx[0] / st.wsum - x;
            st.span[1] = wx[1] / st.wsum - y;
            st.span[2] = wx[2] / st.wsum - z;
        }
    };

    // The normal-sampled image: a sample point ds beyond the wall along the
    // sdf normal, trilinear over the fluid cells around it. False when it is
    // unusable -- deeper than the mirror band, a degenerate normal, or a
    // sample landing among solid cells (thin bodies, concave junctions, an
    // sdf that is not a distance).
    //
    // The thermodynamic state (rho, p) is read from its own stencil, ON the
    // normal. Where the velocity stencil dropped a solid corner, its
    // renormalized point sits off the normal (up to 0.2 h measured on a focal
    // cylinder), and that is harmless for u, which vanishes at the wall and
    // varies along it only at O(h), but not for p, which varies along the
    // wall at O(1): a copy of p there carries an O(h) tangential error, a
    // first-order wall-pressure term (measured: the Froude-Krylov closure
    // error on the focal mesh fell 2.6e-4 -> 1.4e-4 over R/h 16 -> 32). So
    // the thermo sample moves OUT along the normal, in steps of h/8 up to two
    // cells, to the nearest point whose stencil is all fluid; when there is
    // none (a concave pocket), it keeps the velocity stencil. Where the
    // velocity stencil is already full the two coincide, and so do the images,
    // bit for bit.
    auto normalImage = [&](std::size_t c, int i, int j, int k,
                           const double dloc[3], Image& im) {
        const double hl = minSpacing(dloc);
        // The mirror band must be at least two cells deep along EVERY axis:
        // the interior stencil reaches two cells, and a quiescent deep-solid
        // cell inside its reach is a hard discontinuity against the fluid.
        // The threshold therefore follows the WIDEST local spacing -- with
        // the smallest, an anisotropic fine region leaves the band under two
        // cells thick along the wide axis (measured on the Re100 cylinder at
        // aspect 1.7: cd 1.49 -> -7.6, a spurious thrust). Isotropic cells
        // make the two identical, so the Cartesian construction is untouched.
        if (phi[c] < -2.0 * maxSpacing(dloc)) return false;
        const double x = cx(i), y = cy(j), z = cz(k);
        const double d = 1e-3 * hl;
        double nv[3] = {(sdf(x + d, y, z) - sdf(x - d, y, z)) / (2 * d),
                        (sdf(x, y + d, z) - sdf(x, y - d, z)) / (2 * d),
                        (sdf(x, y, z + d) - sdf(x, y, z - d)) / (2 * d)};
        if (nz == 1) nv[2] = 0.0;
        const double nn =
            std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
        if (nn < 1e-12) return false;
        for (double& v : nv) v /= nn;
        // Sample at ds beyond the wall (>= 0.75 of the finest local spacing so
        // the stencil always reaches fluid, even for cells barely inside).
        const double dw = -phi[c];
        const double dsmp = std::max(dw, 0.75 * hl);
        stencilAt(x, y, z, nv, dw + dsmp, im.v);
        if (im.v.wsum < 0.05) return false;
        // A dropped solid corner moves the renormalized sample off the point
        // at dsmp: it now reads the fluid cells' weighted wall distance, and
        // a must use THAT distance or the image is wrong by a grid-independent
        // factor (measured: +2.08% wall shear at dw = h/8, a non-converging
        // torque/heat plateau). With it, any wall-linear profile is exact.
        // Full stencils keep dw/dsmp, so their images are bit-identical.
        im.a = im.v.dropped ? std::min(1.0, dw / (im.v.wphi / im.v.wsum))
                            : dw / dsmp;
        im.t = im.v;
        for (int s = 1; im.v.dropped && s <= 16; ++s) {
            Stencil st;
            stencilAt(x, y, z, nv, dw + dsmp + s * 0.125 * hl, st);
            if (!st.dropped && st.wsum > 0.0) { im.t = st; break; }
        }
        // Wall velocity at the FOOT OF THE NORMAL, not at the ghost centre:
        // that point is on the surface, so a rotor's omega x r there is the
        // actual surface speed the fluid sees.
        im.wall[0] = x + dw * nv[0];
        im.wall[1] = y + dw * nv[1];
        im.wall[2] = z + dw * nv[2];
        return true;
    };

    // The neighbour image, for a cell the fluid's flux READS but whose normal
    // sample is unusable: the cell's own fluid face neighbours (edge
    // neighbours when it has none -- it has one of the two by construction),
    // equally weighted, at most 8 in stencil order. The wall sits where the
    // sdf, linear along the step from the ghost to the sample centroid,
    // crosses zero, so a = dw/dsmp = -phi_g / phi_s -- the normal image's own
    // ratio wherever the sdf is a distance -- capped at 1 like the normal
    // image's (dsmp >= dw there). The cap also bounds it where the sdf is not
    // a distance (the very cells that land here): a in [0, 1] is always a
    // damped mirror. The velocity is sampled at that wall point.
    auto neighbourImage = [&](std::size_t c, int i, int j, int k,
                              const double dloc[3], Image& im) {
        int m = 0;
        double phiS = 0.0, disp[3] = {0.0, 0.0, 0.0};
        for (int order = 1; order <= 2 && m == 0; ++order)
            for (const auto& s : reach) {
                if (m == 8) break;
                if (std::abs(s[0]) + std::abs(s[1]) + std::abs(s[2]) != order)
                    continue;
                const std::size_t cc = g.idx(i + s[0], j + s[1], k + s[2]);
                if (phi[cc] < 0.0) continue;
                im.v.nb[m] = int(cc);
                phiS += phi[cc];
                for (int q = 0; q < 3; ++q) disp[q] += s[q] * dloc[q];
                ++m;
            }
        assert(m > 0 && "neighbourImage: caller checked inFluidReach");
        for (int s = 0; s < 8; ++s) {
            im.v.w[s] = s < m ? 1.0 : 0.0;
            if (s >= m) im.v.nb[s] = im.v.nb[0];
        }
        im.v.wsum = double(m);
        phiS /= m;
        const double dw = -phi[c];
        im.a = phiS > 0.0 ? std::min(1.0, dw / phiS) : 1.0;
        const double t = im.a / (1.0 + im.a);
        const double p[3] = {cx(i), cy(j), cz(k)};
        for (int q = 0; q < 3; ++q) {
            im.wall[q] = p[q] + t * disp[q] / m;
            im.v.span[q] = disp[q] / m;
        }
        // No usable normal here, so no on-normal thermo sample either.
        im.t = im.v;
    };

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                if (phi[c] >= 0.0) continue;
                solid.push_back(int(c));
                // Local physical spacing: on a focal mesh the near-wall
                // thresholds must follow the cells that are actually there.
                double dloc[3];
                localSpacing(i, j, k, dloc);
                Image im;
                if (!normalImage(c, i, j, k, dloc, im)) {
                    // Deep only where no fluid flux reads it: then the cell is
                    // inert and the fluid/solid interface stays closed.
                    if (!inFluidReach(i, j, k)) {
                        deep.push_back(int(c));
                        continue;
                    }
                    neighbourImage(c, i, j, k, dloc, im);
                    ++T.fallback;
                }
                gcell.push_back(int(c));
                isGhost[c] = 1;
                gscale.push_back(Real(im.a));
                gdw.push_back(Real(-phi[c]));
                for (int s = 0; s < 8; ++s) {
                    gnbr.push_back(im.v.nb[s]);
                    gw.push_back(Real(im.v.w[s] / im.v.wsum));
                }
                for (int s = 0; s < 8; ++s) {
                    tnbr.push_back(im.t.nb[s]);
                    tw.push_back(Real(im.t.w[s] / im.t.wsum));
                }
                if (motion) {
                    const GhostWallKinematics kin =
                        (*motion)(im.wall[0], im.wall[1], im.wall[2]);
                    uw.push_back(Real(kin.u[0]));
                    uw.push_back(Real(kin.u[1]));
                    uw.push_back(Real(kin.u[2]));
                    // Span of the PRESSURE image: ghost -> thermo sample.
                    aspan.push_back(Real(kin.a[0] * im.t.span[0]
                                         + kin.a[1] * im.t.span[1]
                                         + kin.a[2] * im.t.span[2]));
                }
            }

    // Interface faces: fluid cell next to a ghost cell.
    const int dijk[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                            {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    const int ndir = nz == 1 ? 4 : 6;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                if (phi[c] < 0.0) continue;
                for (int d = 0; d < ndir; ++d) {
                    const std::size_t nc = g.idx(i + dijk[d][0], j + dijk[d][1],
                                                 k + dijk[d][2]);
                    if (isGhost[nc]) {
                        fcell.push_back(int(c));
                        fdir.push_back(d);
                        // Physical centre of the shared face: the cell centre
                        // stepped half a REFERENCE cell along the axis, then
                        // mapped. On the identity remap this is the Cartesian
                        // face centre exactly.
                        const int a = d / 2;
                        const double s = (d & 1) ? -0.5 : 0.5;
                        const int ijk[3] = {i, j, k};
                        double ref[3];
                        for (int q = 0; q < 3; ++q)
                            ref[q] = (ijk[q] + 0.5 + (q == a ? s : 0.0))
                                     * href[q];
                        fpos.push_back(Real(R.fx.map(ref[0])));
                        fpos.push_back(Real(R.fy.map(ref[1])));
                        fpos.push_back(Real(R.fz.map(ref[2])));
                    }
                }
            }

    return T;
}

std::expected<DeviceCompGhost, hipError_t> DeviceCompGhost::make(
    const Grid& g, const Sdf& sdf, double gamma, double rhoInf, double pInf,
    const double* bbox, const AxisRemap3& R, WallThermal wall,
    const WallMotion* motion)
{
    CompGhostTables T = classifyCompGhost(g, sdf, bbox, R, motion);
    CompGhostCapacity cap;
    cap.cover(T);
    const bool curv = R.fx.active || R.fy.active || R.fz.active;
    return fromTables(g, T, cap, gamma, rhoInf, pInf, curv, wall);
}

std::expected<DeviceCompGhost, hipError_t> DeviceCompGhost::fromTables(
    const Grid& g, const CompGhostTables& t, CompGhostCapacity cap,
    double gamma, double rhoInf, double pInf, bool curv, WallThermal wall)
{
    const bool iso = wall.mode == WallThermal::Mode::Isothermal;
    if (iso && !(wall.Twall > 0.0 && wall.Rgas > 0.0))
        return std::unexpected(hipErrorInvalidValue);
    if (cap.nGhost < t.nGhost() || cap.nDeep < t.nDeep()
        || cap.nFace < t.nFace())
        return std::unexpected(hipErrorInvalidValue);

    DeviceCompGhost out;
    out.g_ = GpuGrid::from(g);
    out.gamma_ = Real(gamma);
    out.rhoInf_ = Real(rhoInf);
    out.pInf_ = Real(pInf);
    out.iso_ = iso ? 1 : 0;
    out.Twall_ = Real(wall.Twall);
    out.Rgas_ = Real(wall.Rgas);
    out.curv_ = curv;
    out.moving_ = !t.uwall.empty();
    out.cap_ = cap;
    // Buffers are sized for the CAPACITY, so replaying another azimuth is a
    // copy and never a reallocation. A capacity axis of 0 allocates nothing and
    // the corresponding data() stays null, which the kernels already read as
    // "resting wall" / "nothing to do".
    auto alloc = [](auto& dst, std::size_t n) -> hipError_t {
        if (n == 0) return hipSuccess;
        auto b = DeviceBuffer<std::decay_t<decltype(*dst.data())>>::make(n);
        if (!b) return b.error();
        dst = std::move(*b);
        return hipSuccess;
    };
    const std::size_t ng = std::size_t(cap.nGhost);
    const std::size_t nf = std::size_t(cap.nFace);
    if (auto e = alloc(out.gcell_, ng); e != hipSuccess) return std::unexpected(e);
    if (auto e = alloc(out.gnbr_, 8 * ng); e != hipSuccess) return std::unexpected(e);
    if (auto e = alloc(out.gw_, 8 * ng); e != hipSuccess) return std::unexpected(e);
    if (auto e = alloc(out.tnbr_, 8 * ng); e != hipSuccess) return std::unexpected(e);
    if (auto e = alloc(out.tw_, 8 * ng); e != hipSuccess) return std::unexpected(e);
    if (auto e = alloc(out.gscale_, ng); e != hipSuccess) return std::unexpected(e);
    if (auto e = alloc(out.gdw_, ng); e != hipSuccess) return std::unexpected(e);
    if (auto e = alloc(out.uwall_, t.uwall.empty() ? 0 : 3 * ng);
        e != hipSuccess)
        return std::unexpected(e);
    if (auto e = alloc(out.aspan_, t.uwall.empty() ? 0 : ng); e != hipSuccess)
        return std::unexpected(e);
    if (auto e = alloc(out.deep_, std::size_t(cap.nDeep)); e != hipSuccess)
        return std::unexpected(e);
    if (auto e = alloc(out.fcell_, nf); e != hipSuccess) return std::unexpected(e);
    if (auto e = alloc(out.fdir_, nf); e != hipSuccess) return std::unexpected(e);
    if (auto e = alloc(out.fpos_, 3 * nf); e != hipSuccess) return std::unexpected(e);
    auto acc = DeviceBuffer<double>::make(std::max<std::size_t>(nf, 1) * 7);
    if (!acc) return std::unexpected(acc.error());
    out.facc_ = std::move(*acc);
    auto red = ReduceScratch::make();
    if (!red) return std::unexpected(red.error());
    out.red_ = std::move(*red);
    if (auto e = out.load(t); e != hipSuccess) return std::unexpected(e);
    return out;
}

hipError_t DeviceCompGhost::load(const CompGhostTables& t)
{
    if (cap_.nGhost < t.nGhost() || cap_.nDeep < t.nDeep()
        || cap_.nFace < t.nFace())
        return hipErrorInvalidValue;
    // A wall that moves must keep moving and one at rest must stay at rest:
    // the kernel selects its branch on whether the buffer exists, so a bin
    // whose velocities were dropped would silently freeze the surface.
    if (t.uwall.empty() != (uwall_.data() == nullptr)
        || t.aspan.size() * 3 != t.uwall.size()
        || t.tnbr.size() != t.gnbr.size() || t.tw.size() != t.gw.size())
        return hipErrorInvalidValue;
    auto put = [](auto& dst, const auto& src) -> hipError_t {
        if (src.empty()) return hipSuccess;
        return hipMemcpy(dst.data(), src.data(),
                         src.size() * sizeof(src[0]), hipMemcpyHostToDevice);
    };
    if (auto e = put(gcell_, t.gcell); e != hipSuccess) return e;
    if (auto e = put(gnbr_, t.gnbr); e != hipSuccess) return e;
    if (auto e = put(gw_, t.gw); e != hipSuccess) return e;
    if (auto e = put(tnbr_, t.tnbr); e != hipSuccess) return e;
    if (auto e = put(tw_, t.tw); e != hipSuccess) return e;
    if (auto e = put(gscale_, t.gscale); e != hipSuccess) return e;
    if (auto e = put(gdw_, t.gdw); e != hipSuccess) return e;
    if (auto e = put(uwall_, t.uwall); e != hipSuccess) return e;
    if (auto e = put(aspan_, t.aspan); e != hipSuccess) return e;
    if (auto e = put(deep_, t.deep); e != hipSuccess) return e;
    if (auto e = put(fcell_, t.fcell); e != hipSuccess) return e;
    if (auto e = put(fdir_, t.fdir); e != hipSuccess) return e;
    if (auto e = put(fpos_, t.fpos); e != hipSuccess) return e;
    nGhost_ = t.nGhost();
    nDeep_ = t.nDeep();
    nFace_ = t.nFace();
    fallback_ = t.fallback;
    solid_ = t.solid;
    deepHost_ = t.deep;
    return hipSuccess;
}

void DeviceCompGhost::apply(DeviceCompState& s)
{
    const Real EInf = pInf_ / (gamma_ - Real(1));
    if (nDeep_ > 0) {
        // The deep interior sits at (pInf, Twall): the mirror band is only two
        // cells thick, so the interior stencil reaches these cells, and a body
        // core left at the FREESTREAM temperature would put a step of
        // Twall - Tinf one cell behind an isothermal wall and conduct straight
        // through it. Density carries the change; E = pInf/(gamma-1) is the
        // same internal energy either way, and rhoInf_ is used verbatim on the
        // adiabatic path.
        const Real rhoQ = iso_ ? pInf_ / (Rgas_ * Twall_) : rhoInf_;
        kQuiescent<<<launchBlocks(nDeep_), blockSize>>>(
            deep_.data(), nDeep_, rhoQ, EInf, s.rho.data(), s.mx.data(),
            s.my.data(), s.mz.data(), s.E.data());
        checkLastKernel("kQuiescent");
    }
    if (nGhost_ > 0) {
        kGhostFill<<<launchBlocks(nGhost_), blockSize>>>(
            gamma_, iso_, Twall_, Rgas_, gcell_.data(), gnbr_.data(),
            gw_.data(), tnbr_.data(), tw_.data(), gscale_.data(),
            uwall_.data(), aspan_.data(), nGhost_, s.rho.data(),
            s.mx.data(), s.my.data(), s.mz.data(), s.E.data());
        checkLastKernel("kGhostFill");
    }
}

void DeviceCompGhost::applyTurb(DeviceCompState& s, RansModel model,
                                const Transport& tr, double omegaInf) const
{
    const ViscView vv = viscView(tr);
    assert(s.nT == ransScalars(model));
    const int sst = ransSst(model) ? 1 : 0;
    Real* t1 = sst ? s.t[1].data() : nullptr;
    if (nGhost_ > 0) {
        kGhostTurb<<<launchBlocks(nGhost_), blockSize>>>(
            sst, vv, gamma_, Rgas_, gcell_.data(), gnbr_.data(), gw_.data(),
            tnbr_.data(), tw_.data(), gscale_.data(), gdw_.data(), nGhost_,
            s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data(),
            s.t[0].data(), t1);
        checkLastKernel("kGhostTurb");
    }
    if (nDeep_ > 0) {
        kDeepTurb<<<launchBlocks(nDeep_), blockSize>>>(
            deep_.data(), nDeep_, sst, Real(omegaInf), s.rho.data(),
            s.t[0].data(), t1);
        checkLastKernel("kDeepTurb");
    }
    if (!s.compensated()) return;
    LowWords L{{}, 5 + s.nT};
    for (int f = 0; f < 5; ++f) L.lo[f] = s.lo[f].data();
    for (int i = 0; i < s.nT; ++i) L.lo[5 + i] = s.tlo[i].data();
    if (nGhost_ > 0)
        kZeroLow<<<launchBlocks(nGhost_), blockSize>>>(gcell_.data(), nGhost_,
                                                       L);
    if (nDeep_ > 0)
        kZeroLow<<<launchBlocks(nDeep_), blockSize>>>(deep_.data(), nDeep_, L);
    checkLastKernel("kZeroLow");
}

void DeviceCompGhost::mirrorMut(Real* nut, Real* f1) const
{
    if (nGhost_ > 0) {
        kGhostMut<<<launchBlocks(nGhost_), blockSize>>>(
            gcell_.data(), gnbr_.data(), gw_.data(), gscale_.data(), nGhost_,
            nut, f1);
        checkLastKernel("kGhostMut");
    }
    if (nDeep_ > 0) {
        kDeepMut<<<launchBlocks(nDeep_), blockSize>>>(deep_.data(), nDeep_,
                                                      nut, f1);
        checkLastKernel("kDeepMut");
    }
}

DeviceCompGhost::YPlus DeviceCompGhost::wallYplus(DeviceComp& solver,
                                                  const DeviceCompState& s)
{
    assert(solver.rans());
    if (nFace_ == 0) return {0.0, 0.0};
    solver.primitives(s);
    const PrimView q{s.rho.data(),      solver.u().data(), solver.v().data(),
                     solver.w().data(), solver.e().data(), solver.p().data(),
                     solver.T().data()};
    kWallYplus<<<launchBlocks(nFace_), blockSize>>>(
        viscView(solver.transport()), q, solver.wallDistance().data(),
        fcell_.data(), nFace_, facc_.data());
    checkLastKernel("kWallYplus");
    const std::size_t m = std::size_t(nFace_);
    const double mx = internal::reduce(red_, m, SpanMax{facc_.data()},
                                       rocprim::maximum<double>(), 0.0);
    return {mx, sumDouble(facc_.data(), m, red_) / double(m)};
}

DeviceCompGhost::Force DeviceCompGhost::force(DeviceComp& solver,
                                              const DeviceCompState& s,
                                              const std::array<double, 3>& pivot)
{
    if (nFace_ == 0) return {};
    solver.primitives(s);
    const PrimView q{s.rho.data(),      solver.u().data(), solver.v().data(),
                     solver.w().data(), solver.e().data(), solver.p().data(),
                     solver.T().data()};
    const MrfView mz = solver.mrfActive() ? solver.mrfView() : MrfView{};
    // RANS: mu_t of the state (updateMut refreshes primitives too) on the
    // solver's own curvilinear flux; otherwise null, today's tally.
    const Real* nut = nullptr;
    Real kap = Real(0);
    if (solver.rans()) {
        solver.updateMut(s);
        nut = solver.mut().data();
        kap = solver.eddyKapScale();
    }
    if (curv_ || solver.rans()) {
        const DeviceCurv* t = solver.curvTables();
        assert(t && "focal ghost force needs the curvilinear DeviceComp");
        if (!t) return {};
        const CurvView cv = t->view();
        kBodyFluxCurv<<<launchBlocks(nFace_), blockSize>>>(
            g_, viscView(solver.transport()), iso_, q, cv, fcell_.data(),
            fdir_.data(), nFace_, facc_.data(), mz, nut, kap);
        checkLastKernel("kBodyFluxCurv");
    } else {
        kBodyFlux<<<launchBlocks(nFace_), blockSize>>>(
            g_, viscView(solver.transport()), iso_, q, fcell_.data(),
            fdir_.data(), nFace_, facc_.data(), mz, nut, kap);
        checkLastKernel("kBodyFlux");
    }
    kBodyMoment<<<launchBlocks(nFace_), blockSize>>>(
        fpos_.data(), pivot[0], pivot[1], pivot[2], nFace_, facc_.data());
    checkLastKernel("kBodyMoment");
    // Deterministic FP64 reductions over disjoint spans of facc_; span 3 is
    // the wall heat and is skipped by the moment, hence the 4..6 offset.
    const std::size_t m = std::size_t(nFace_);
    double f[7];
    for (int c = 0; c < 7; ++c)
        f[c] = sumDouble(facc_.data() + c * m, m, red_);
    return {.fx = f[0], .fy = f[1], .fz = f[2], .heat = f[3],
            .qx = f[4], .qy = f[5], .qz = f[6]};
}

} // namespace dns::gpu
