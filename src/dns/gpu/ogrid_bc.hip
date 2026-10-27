#include "dns/gpu/ogrid_bc.hpp"

#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/internal/rans_wall.hpp"
#include "dns/gpu/internal/wall_ghost.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <vector>

namespace dns::gpu {

namespace {

// --- inner wall row (j = 0) --------------------------------------------------
//
// The mesh puts the ghost row at the exact geometric reflection of the first
// fluid row about the wall face, so a value mirror is second order ON that face
// -- the argument ChannelStretchY documents, unchanged. No-slip antimirrors ALL
// THREE momentum components, which is orientation-free and therefore correct on
// a curved wall: the face velocity is exactly zero whatever the local normal
// is. (A FREE-SLIP wall would not be: it needs m - 2 (m.n) n and the local
// normal, so it is deliberately not offered here.)
//
// Both ghost states are the shared no-slip wall (internal/wall_ghost.hpp) --
// the same one the channel and convection plates get. Only the row differs:
// there are two plates on a box and one wall here, at j = 0.
// uwx/uwy/as (all null or all set, nx entries): wall velocity for a rigidly
// rotating body (--rotor-frame mrf). The moving image is u0 = 2 u_wall - u1
// -- the resting mirror's general form at reflection factor exactly 1. The
// pressure is the wall-normal momentum balance, grad p = -rho a_wall (the
// fluid at the wall moves with it; in the rotating frame the same balance is
// the Coriolis term at u_rel = 0): p0 = p1 + rho1 a_wall . (x1 - x0), `as`.
// The temperature is built exactly as the resting branch builds it
// (adiabatic: T0 == T1, so rho0 = rho1 p0/p1 and the Fourier flux vanishes;
// isothermal: the same floored T0, rho0 = p0/(R T0)). The viscous part of the
// balance is not imaged, for the reason comp_ghost.hip's kGhostFill gives.
// The resting branch (a_wall = 0: the copy, the same condition) is kept
// VERBATIM under the null check rather than folded into the general form:
// -m and 2*rho*uw - m are not the same float at uw == 0.
//
// Every row pass below is one (i, k) column's work, run by kOGridRows in the
// order wall, far, scalars, low words. A pass reads only real cells (rows 1
// and ny-2, which no pass writes) and the ghost cells of its OWN column that
// an earlier pass wrote -- never another thread's -- so one launch is the four
// former launches bit for bit.
__device__ inline void wallRowCell(GpuGrid g, int i, int k, int isothermal,
                                   Real gam, Real Rgas, Real Twall,
                                   const Real* uwx, const Real* uwy,
                                   const Real* as, Real* rho, Real* mx,
                                   Real* my, Real* mz, Real* E)
{
    const std::size_t c = gidx(g, i, 0, k);
    const std::size_t s = gidx(g, i, 1, k);
    if (!uwx) {
        if (!isothermal) internal::adiabaticGhost(c, s, rho, mx, my, mz, E);
        else internal::isothermalGhost(c, s, gam, Rgas, Twall, Real(0), rho,
                                       mx, my, mz, E);
        return;
    }
    const Real r1 = rho[s];
    const Real u1 = mx[s] / r1, v1 = my[s] / r1, w1 = mz[s] / r1;
    const Real ke1 = Real(0.5) * (u1 * u1 + v1 * v1 + w1 * w1);
    const Real p1 = (gam - Real(1)) * (E[s] - r1 * ke1);
    // The wall spins in-plane about the z shaft; w antimirrors as at rest.
    const Real u0 = Real(2) * uwx[i] - u1, v0 = Real(2) * uwy[i] - v1,
               w0 = -w1;
    const Real ke0 = Real(0.5) * (u0 * u0 + v0 * v0 + w0 * w0);
    const Real p0 = p1 + r1 * as[i];
    Real r0 = r1 * (p0 / p1);
    if (isothermal) {
        const Real T0 =
            fmaxf(Real(2) * Twall - p1 / (Rgas * r1), Real(0.05) * Twall);
        r0 = p0 / (Rgas * T0);
    }
    rho[c] = r0;
    mx[c] = r0 * u0;
    my[c] = r0 * v0;
    mz[c] = r0 * w0;
    E[c] = p0 / (gam - Real(1)) + r0 * ke0;
}

// --- outer far-field row (j = ny-1) ------------------------------------------
//
// LODI characteristic condition in the LOCAL RADIAL normal n = (cos th, sin th,
// 0) of the exact far circle, with tangent t = (-sin th, cos th, 0). The wave
// algebra is comp_nscbc's L1..L5 block re-expressed in that frame, in
// DEVIATION form: every amplitude is built from the interior state MINUS the
// freestream, before any rotation. That is what keeps the uniform freestream a
// fixed point independently of how fast the freestream is -- rotating the TOTAL
// velocity would leave a residual of order eps |uInf| in every component, a
// spurious far-field source that grows with uInf. What remains is the FP32
// primitive recovery alone (p out of E, relative eps E / p), and nothing below
// amplifies it:
//
//   d(.)   = interior value minus freestream value, interior = cell j = ny-2
//   dun    = d(u) . n,  dut = d(u) . t,  dw = d(u) . z
//   zc     = rho1 c1                      (interior acoustic impedance)
//   L+     = dp + zc dun                  outgoing acoustic, speed un + c > 0
//   L-     = (1 - sigma) (dp - zc dun)    incoming acoustic, speed un - c < 0
//   dpb    = (L+ + L-) / 2                boundary pressure
//   dunb   = (L+ - L-) / (2 zc)           boundary normal velocity -- FREE, it
//                                         is solved for, never clamped to uInf
//
// L- is the only modeled amplitude, and sigma is the Rudy-Strikwerda anchor
// (ogrid_bc.hpp). It vanishes identically for a purely outgoing acoustic wave
// (dp == zc dun), so that wave leaves unchanged for ANY sigma.
//
// The entropy and shear waves travel at un: they leave over the downstream half
// (un > 0) and enter over the upstream half (un < 0). Downstream they are taken
// one-sided from the interior; upstream the freestream prescribes them, which
// is the subsonic-inflow branch -- the outgoing acoustic still leaves through
// L+. The two forms are blended with a tanh dead band of width `band` * c, and
// both vanish at the freestream, so the fixed point survives any blend weight.
//
// Rotating back and reflecting (ghost = 2 boundary - interior) puts the
// boundary state exactly on the far face, which is the +1 face of cell ny-2 and
// the exact circle of radius rFar.
__device__ inline void farRowCell(GpuGrid g, int i, int k, Real gam,
                                  Real rhoI, Real uI, Real pI, Real cI,
                                  Real sigma, Real band, const Real* cosT,
                                  const Real* sinT, Real* rho, Real* mx,
                                  Real* my, Real* mz, Real* E)
{
    const std::size_t c = gidx(g, i, g.ny - 1, k);
    const std::size_t s = gidx(g, i, g.ny - 2, k);

    const Real nrx = cosT[i], nry = sinT[i]; // outward radial normal
    const Real tgx = -nry, tgy = nrx;        // tangent

    const Real r1 = rho[s];
    const Real u1 = mx[s] / r1, v1 = my[s] / r1, w1 = mz[s] / r1;
    const Real ke1 = Real(0.5) * (u1 * u1 + v1 * v1 + w1 * w1);
    const Real p1 = (gam - Real(1)) * (E[s] - r1 * ke1);
    const Real c1 = sqrt(gam * p1 / r1);
    const Real zc = r1 * c1;

    // Deviation from the freestream, rotated into (n, t, z).
    const Real du = u1 - uI, dv = v1, dw = w1;
    const Real dun = du * nrx + dv * nry;
    const Real dut = du * tgx + dv * tgy;
    const Real dp = p1 - pI, dr = r1 - rhoI;

    const Real Lp = dp + zc * dun;
    const Real Lm = (Real(1) - sigma) * (dp - zc * dun);
    const Real dpb = Real(0.5) * (Lp + Lm);
    const Real dunb = (Lp - Lm) / (Real(2) * zc);

    // Total normal velocity decides which way entropy and shear travel; the
    // tanh weight is 1 on the outflow half and 0 on the inflow half.
    const Real un1 = uI * nrx + dun;
    const Real wgt = Real(0.5) * (Real(1) + tanh(un1 / (band * c1)));
    const Real drb = wgt * (dr + (dpb - dp) / (c1 * c1)) +
                     (Real(1) - wgt) * (dpb / (cI * cI));
    const Real dutb = wgt * dut;
    const Real dwb = wgt * dw;

    // Rotate back; the boundary state sits ON the far face.
    const Real ub = uI + dunb * nrx + dutb * tgx;
    const Real vb = dunb * nry + dutb * tgy;
    const Real wb = dwb;
    const Real rb = rhoI + drb, pb = pI + dpb;

    // Reflect through the face. The floors are the far-field twin of the wall's
    // temperature floor: a transient must not be able to build a ghost with a
    // non-positive density or pressure, which would make the face flux NaN.
    const Real r0 = fmaxf(Real(2) * rb - r1, Real(0.05) * r1);
    const Real p0 = fmaxf(Real(2) * pb - p1, Real(0.05) * p1);
    const Real u0 = Real(2) * ub - u1, v0 = Real(2) * vb - v1,
               w0 = Real(2) * wb - w1;
    rho[c] = r0;
    mx[c] = r0 * u0;
    my[c] = r0 * v0;
    mz[c] = r0 * w0;
    E[c] = p0 / (gam - Real(1)) +
           Real(0.5) * r0 * (u0 * u0 + v0 * v0 + w0 * w0);
}

// The same far row on a REVOLVED mesh: the LODI algebra of farRowCell in
// vector form about the far face's own unit normal n (3 per face, from the
// metrics), the tangential deviation du - (du . n) n blended by the same
// tanh weight. Both forms vanish at the freestream, so it is a fixed point.
// t: the face index i + nx k of the normal table.
__device__ inline void farRowCell3(GpuGrid g, std::size_t t, int i, int k,
                                   Real gam, Real rhoI, Real uI, Real pI,
                                   Real cI, Real sigma, Real band,
                                   const Real* nrm, Real* rho, Real* mx,
                                   Real* my, Real* mz, Real* E)
{
    const std::size_t c = gidx(g, i, g.ny - 1, k);
    const std::size_t s = gidx(g, i, g.ny - 2, k);
    const Real n[3] = {nrm[3 * t], nrm[3 * t + 1], nrm[3 * t + 2]};
    const Real r1 = rho[s];
    const Real u1[3] = {mx[s] / r1, my[s] / r1, mz[s] / r1};
    const Real ke1 =
        Real(0.5) * (u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
    const Real p1 = (gam - Real(1)) * (E[s] - r1 * ke1);
    const Real c1 = sqrt(gam * p1 / r1);
    const Real zc = r1 * c1;
    const Real du[3] = {u1[0] - uI, u1[1], u1[2]};
    const Real dun = du[0] * n[0] + du[1] * n[1] + du[2] * n[2];
    const Real dp = p1 - pI, dr = r1 - rhoI;
    const Real Lp = dp + zc * dun;
    const Real Lm = (Real(1) - sigma) * (dp - zc * dun);
    const Real dpb = Real(0.5) * (Lp + Lm);
    const Real dunb = (Lp - Lm) / (Real(2) * zc);
    const Real un1 = uI * n[0] + dun;
    const Real wgt = Real(0.5) * (Real(1) + tanh(un1 / (band * c1)));
    const Real drb = wgt * (dr + (dpb - dp) / (c1 * c1)) +
                     (Real(1) - wgt) * (dpb / (cI * cI));
    Real ub[3];
    for (int a = 0; a < 3; ++a)
        ub[a] = (a == 0 ? uI : Real(0)) + dunb * n[a] +
                wgt * (du[a] - dun * n[a]);
    const Real rb = rhoI + drb, pb = pI + dpb;
    const Real r0 = fmaxf(Real(2) * rb - r1, Real(0.05) * r1);
    const Real p0 = fmaxf(Real(2) * pb - p1, Real(0.05) * p1);
    const Real u0 = Real(2) * ub[0] - u1[0], v0 = Real(2) * ub[1] - u1[1],
               w0 = Real(2) * ub[2] - u1[2];
    rho[c] = r0;
    mx[c] = r0 * u0;
    my[c] = r0 * v0;
    mz[c] = r0 * w0;
    E[c] = p0 / (gam - Real(1)) +
           Real(0.5) * r0 * (u0 * u0 + v0 * v0 + w0 * w0);
}

// Zero the tally's per-face entries outside the real faces (revolved
// meshes: ghost columns, wedge ghost planes). spans of n = nx*nz each.
__global__ void kTallyMask(GpuGrid g, int spans, int iLo, int iHi, int kLo,
                           int kHi, double* acc)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t n = std::size_t(g.nx) * g.nz;
    if (t >= n) return;
    const int i = int(t % g.nx), k = int(t / g.nx);
    if (i >= iLo && i <= iHi && k >= kLo && k <= kHi) return;
    for (int c = 0; c < spans; ++c) acc[std::size_t(c) * n + t] = 0.0;
}

// --- RANS scalars on both rows ------------------------------------------------
//
// Runs after the wall and far passes (it reads their ghost densities). The
// wall row takes the shared no-slip images (gpu/internal/rans_wall.hpp): SA
// nu~ and SST k antimirrored on the specific value, SST omega the Dirichlet
// image of omega_w = 60 mu(T_f)/(beta1 d1^2 rho_f) per wall face -- wcoef
// holds 60 mu_ref/(beta1 d1^2), and Sutherland scales it by mu(T_f)/mu_ref
// on the face temperature, so the coefficient is the face's own viscosity.
// The far row takes phi_b = w phi_1 + (1 - w) phi_inf, w the far kernel's
// tanh weight recomputed from the same interior cell: positive, and the
// upwind flux of an inflow face then carries phi_inf exactly.
// t: the face index i + nx k (wcoef, the revolved normal table).
__device__ inline void turbRowsCell(GpuGrid g, std::size_t t, int i, int k,
                                    ViscView vv, int sst, Real gam, Real Rgas,
                                    Real uI, Real band, Real phi0, Real phi1,
                                    const Real* cosT, const Real* sinT,
                                    const Real* nrm, const Real* wcoef,
                                    const Real* rho, const Real* mx,
                                    const Real* my, const Real* mz,
                                    const Real* E, Real* t0, Real* t1)
{
    // Wall row.
    {
        const std::size_t c = gidx(g, i, 0, k), s = gidx(g, i, 1, k);
        t0[c] = internal::saGhost(rho[c], rho[s], t0[s]);
        if (sst) {
            Real coef = wcoef[t];
            if (vv.sutherland) {
                auto temp = [&](std::size_t x) {
                    const Real r = rho[x];
                    const Real ke = Real(0.5) *
                        (mx[x] * mx[x] + my[x] * my[x] + mz[x] * mz[x]) / r;
                    return (gam - Real(1)) * (E[x] - ke) / (r * Rgas);
                };
                coef *= vv.ratio(Real(0.5) * (temp(c) + temp(s)));
            }
            t1[c] = internal::sstOmegaGhost(coef, rho[c], rho[s], t1[s]);
        }
    }
    // Far row.
    const std::size_t c = gidx(g, i, g.ny - 1, k), s = gidx(g, i, g.ny - 2, k);
    // The far normal: the analytic in-plane one, or the revolved table's.
    const Real nrx = nrm ? nrm[3 * t] : cosT[i];
    const Real nry = nrm ? nrm[3 * t + 1] : sinT[i];
    const Real nrz = nrm ? nrm[3 * t + 2] : Real(0);
    const Real r1 = rho[s];
    const Real u1 = mx[s] / r1, v1 = my[s] / r1, w1 = mz[s] / r1;
    const Real ke1 = Real(0.5) * (u1 * u1 + v1 * v1 + w1 * w1);
    const Real p1 = (gam - Real(1)) * (E[s] - r1 * ke1);
    const Real c1 = sqrt(gam * p1 / r1);
    const Real dun = (u1 - uI) * nrx + v1 * nry + w1 * nrz;
    const Real un1 = uI * nrx + dun;
    const Real wgt = Real(0.5) * (Real(1) + tanh(un1 / (band * c1)));
    const Real rc = rho[c];
    t0[c] = rc * (wgt * (t0[s] / r1) + (Real(1) - wgt) * phi0);
    if (sst) t1[c] = rc * (wgt * (t1[s] / r1) + (Real(1) - wgt) * phi1);
}

// mu_t / F1 images: row 0 odd mu_t (wall face exactly 0.0) and even F1; row
// ny-1 an even copy of row ny-2 for both. f1 null for SA.
__global__ void kOGridMut(GpuGrid g, Real* nut, Real* f1)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxz = std::size_t(g.nx) * g.nz;
    if (t >= nxz) return;
    const int i = int(t % g.nx), k = int(t / g.nx);
    const std::size_t w0 = gidx(g, i, 0, k), w1 = gidx(g, i, 1, k);
    const std::size_t f0 = gidx(g, i, g.ny - 1, k), f1s = gidx(g, i, g.ny - 2, k);
    nut[w0] = -nut[w1];
    nut[f0] = nut[f1s];
    if (!f1) return;
    f1[w0] = f1[w1];
    f1[f0] = f1[f1s];
}

// Both boundary rows are SET, not accumulated: on a compensated state their
// low words are zeroed so each pair holds exactly the value the row wrote
// (the channel wall's rule, comp_channel.hip).
struct RowLo
{
    Real* p[7];
    int n;
};

// Everything one stage-hook call of the rows needs (DeviceOGridBc::apply).
struct RowArgs
{
    int isothermal;
    Real gam, Rgas, Twall;
    const Real *uwx, *uwy, *as;          // moving wall; null = at rest
    Real rhoI, uI, pI, cI, sigma, band;  // far row
    const Real *cosT, *sinT, *nrm;       // nrm: revolved far normals
    ViscView vv;                         // scalar rows (t0 null = none)
    int sst;
    Real phi0, phi1;
    const Real* wcoef;
    Real *rho, *mx, *my, *mz, *E, *t0, *t1;
    RowLo lo;                            // lo.n == 0: a plain state
};

// Both rows of one (i, k) column: wall, far (the revolved form when Far3),
// scalars, low words -- the former four launches' order (see wallRowCell).
template <bool Far3> __global__ void kOGridRows(GpuGrid g, RowArgs a)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxz = std::size_t(g.nx) * g.nz;
    if (t >= nxz) return;
    const int i = int(t % g.nx), k = int(t / g.nx);
    wallRowCell(g, i, k, a.isothermal, a.gam, a.Rgas, a.Twall, a.uwx, a.uwy,
                a.as, a.rho, a.mx, a.my, a.mz, a.E);
    if constexpr (Far3)
        farRowCell3(g, t, i, k, a.gam, a.rhoI, a.uI, a.pI, a.cI, a.sigma,
                    a.band, a.nrm, a.rho, a.mx, a.my, a.mz, a.E);
    else
        farRowCell(g, i, k, a.gam, a.rhoI, a.uI, a.pI, a.cI, a.sigma, a.band,
                   a.cosT, a.sinT, a.rho, a.mx, a.my, a.mz, a.E);
    if (a.t0)
        turbRowsCell(g, t, i, k, a.vv, a.sst, a.gam, a.Rgas, a.uI, a.band,
                     a.phi0, a.phi1, a.cosT, a.sinT, a.nrm, a.wcoef, a.rho,
                     a.mx, a.my, a.mz, a.E, a.t0, a.t1);
    const std::size_t w = gidx(g, i, 0, k), f = gidx(g, i, g.ny - 1, k);
    for (int v = 0; v < a.lo.n; ++v) a.lo.p[v][w] = a.lo.p[v][f] = Real(0);
}

// --- wall surface tally ------------------------------------------------------
//
// One thread per wall face. The wall face is the +1 face of the inner ghost row
// (the lo face of the first fluid cell), so faceFluxCurv already integrates
// over the exact face area vector S[1] -- there is no area multiply, no
// aperture and no sdf. The stored sign is the flux oriented INTO the body,
// which is minus the +1-direction flux, matching DeviceCompGhost::force's
// convention for a solid lying at -axis.
//
// Staged per face rather than atomically accumulated: the reduction is
// deterministic and FP64 (the ground rules require that of anything reported).
// mz: the rotating-frame zone when one is installed -- the tally passes the
// SAME view as the rhs, so the reported force is the budget of the solved
// equations (at f = 1 over the wall the transport correction cancels the
// moving wall's own normal flux exactly, which is what keeps the face
// watertight). wpx/wpy (nx, null when the wall rests): wall-face midpoints,
// the lever arms of the moment about the z shaft through (cx, cy).
__global__ void kOGridWallFlux(GpuGrid g, ViscView vv, PrimView q, CurvView m,
                               MrfView mz, const Real* wpx, const Real* wpy,
                               Real cx, Real cy, double* acc)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t n = std::size_t(g.nx) * g.nz;
    if (t >= n) return;
    const int i = int(t % g.nx), k = int(t / g.nx);
    const Flux f = faceFluxCurv(g, vv, q, m, i, 0, k, 1, nullptr, Real(0),
                                mz.active ? &mz : nullptr);
    acc[t] = -double(f.mx);
    acc[n + t] = -double(f.my);
    acc[2 * n + t] = -double(f.mz);
    acc[3 * n + t] = -double(f.E);
    acc[4 * n + t] = wpx ? double(wpx[i] - cx) * -double(f.my) -
                           double(wpy[i] - cy) * -double(f.mx)
                         : 0.0;
}

// Per wall face, 14 spans of nx*nz: [0,3) pressure force on the body, [3,6)
// viscous force on the body, [6,9) unit normal into the fluid, 9 energy flux
// into the body, 10 area, 11 face pressure, 12 y+, 13 spare. The face flux is
// kOGridWallFlux's; at the resting no-slip face its convective part is
// exactly zero (antimirrored velocity), so momentum = p_f S - tau.S and the
// viscous force on the body is the total minus the pressure part. d1 null:
// y+ 0.
__global__ void kOGridFaces(GpuGrid g, ViscView vv, PrimView q, CurvView m,
                            MrfView mz, const double* d1, double* out)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t n = std::size_t(g.nx) * g.nz;
    if (t >= n) return;
    const int i = int(t % g.nx), k = int(t / g.nx);
    const std::size_t c0 = gidx(g, i, 0, k), c1 = gidx(g, i, 1, k);
    const Flux f = faceFluxCurv(g, vv, q, m, i, 0, k, 1, nullptr, Real(0),
                                mz.active ? &mz : nullptr);
    const double S[3] = {m.S[1][0][c0], m.S[1][1][c0], m.S[1][2][c0]};
    const double A = sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]);
    // The flux's own face pressure (FP32, faceFluxCurv's pm): the pressure
    // part it subtracts is then the one the flux added.
    const double pf = double(Real(0.5) * (q.p[c0] + q.p[c1]));
    const double fm[3] = {-double(f.mx), -double(f.my), -double(f.mz)};
    double fv[3], nn[3];
    for (int a = 0; a < 3; ++a) {
        out[a * n + t] = -pf * S[a];
        fv[a] = fm[a] + pf * S[a];
        out[(3 + a) * n + t] = fv[a];
        nn[a] = A > 0 ? S[a] / A : 0.0;
        out[(6 + a) * n + t] = nn[a];
    }
    out[9 * n + t] = -double(f.E);
    out[10 * n + t] = A;
    out[11 * n + t] = pf;
    double yp = 0.0;
    if (d1 && A > 0) {
        const double tn = (fv[0] * nn[0] + fv[1] * nn[1] + fv[2] * nn[2]) / A;
        double tt2 = 0;
        for (int a = 0; a < 3; ++a) {
            const double x = fv[a] / A - tn * nn[a];
            tt2 += x * x;
        }
        const double rf = 0.5 * (double(q.rho[c0]) + double(q.rho[c1]));
        const Real Tf = Real(0.5) * (q.T[c0] + q.T[c1]);
        const double mu = vv.sutherland ? double(vv.mu * vv.ratio(Tf))
                                        : double(vv.mu);
        yp = d1[t] * sqrt(sqrt(tt2) / rf) * rf / mu;
    }
    out[12 * n + t] = yp;
    out[13 * n + t] = 0.0;
}

} // namespace

std::expected<DeviceOGridBc, hipError_t> DeviceOGridBc::make(const Grid& g,
                                                             Config c)
{
    const double cInf = c.rhoInf > 0 && c.pInf > 0
                            ? std::sqrt(c.gamma * c.pInf / c.rhoInf)
                            : 0.0;
    const bool ok = g.nx >= 3 && g.ny >= 4 && g.nz >= 1 && c.gamma > 1 &&
                    c.Rgas > 0 && c.rhoInf > 0 && c.pInf > 0 &&
                    std::abs(c.uInf) < cInf && c.sigma > 0 && c.sigma <= 1 &&
                    c.band > 0 &&
                    (c.thermal == OGridThermal::Adiabatic || c.Twall > 0) &&
                    (c.omega == 0 ||
                     (c.wallPos.size() == std::size_t(g.nx) &&
                      c.wallSpan.size() == std::size_t(g.nx))) &&
                    (c.rans == RansModel::None ||
                     (c.wallD1.size() == std::size_t(g.nx) * g.nz &&
                      std::ranges::all_of(c.wallD1,
                                          [](double d) { return d > 0; })));
    if (!ok) return std::unexpected(hipErrorInvalidValue);

    // Cell-centre radial normal, one per angular cell: the bisector of the far
    // face's two nodes, which is exactly the direction of that face's area
    // vector. Computed in double on the host so the angular table is the same
    // every run.
    std::vector<Real> ct(g.nx), st(g.nx);
    for (int i = 0; i < g.nx; ++i)
    {
        const double th =
            -2.0 * std::numbers::pi * (double(i) + 0.5) / double(g.nx);
        ct[std::size_t(i)] = Real(std::cos(th));
        st[std::size_t(i)] = Real(std::sin(th));
    }

    DeviceOGridBc out;
    out.g_ = GpuGrid::from(g);
    out.c_ = c;
    out.u_ = c.uInf;
    out.cInf_ = cInf;
    auto cb = DeviceScalar::make(std::size_t(g.nx));
    auto sb = DeviceScalar::make(std::size_t(g.nx));
    if (!cb) return std::unexpected(cb.error());
    if (!sb) return std::unexpected(sb.error());
    if (auto e = cb->upload(ct.data()); e != hipSuccess)
        return std::unexpected(e);
    if (auto e = sb->upload(st.data()); e != hipSuccess)
        return std::unexpected(e);
    out.cosT_ = std::move(*cb);
    out.sinT_ = std::move(*sb);
    if (c.omega != 0)
    {
        // Wall-face midpoints and the rigid wall velocity omega x (p - c)
        // there, both in double on the host so the tables are the same every
        // run; the kernels read them as Real like every field.
        std::vector<Real> px(g.nx), py(g.nx), ux(g.nx), uy(g.nx), as(g.nx);
        const double w2 = c.omega * c.omega;
        for (int i = 0; i < g.nx; ++i)
        {
            const auto& p = c.wallPos[std::size_t(i)];
            const auto& d = c.wallSpan[std::size_t(i)];
            px[std::size_t(i)] = Real(p[0]);
            py[std::size_t(i)] = Real(p[1]);
            ux[std::size_t(i)] = Real(-c.omega * (p[1] - c.cy));
            uy[std::size_t(i)] = Real(c.omega * (p[0] - c.cx));
            // Centripetal a = -omega^2 (p - c), dotted with the image span.
            as[std::size_t(i)] =
                Real(-w2 * ((p[0] - c.cx) * d[0] + (p[1] - c.cy) * d[1]));
        }
        for (auto [dst, src] : {std::pair{&out.wpx_, &px}, {&out.wpy_, &py},
                                {&out.uwx_, &ux}, {&out.uwy_, &uy},
                                {&out.aspan_, &as}})
        {
            auto b = DeviceScalar::make(std::size_t(g.nx));
            if (!b) return std::unexpected(b.error());
            if (auto e = b->upload(src->data()); e != hipSuccess)
                return std::unexpected(e);
            *dst = std::move(*b);
        }
    }
    auto acc = DeviceBuffer<double>::make(5 * std::size_t(g.nx) * g.nz);
    if (!acc) return std::unexpected(acc.error());
    out.acc_ = std::move(*acc);
    auto fc = DeviceBuffer<double>::make(14 * std::size_t(g.nx) * g.nz);
    if (!fc) return std::unexpected(fc.error());
    out.faces_ = std::move(*fc);
    if (out.c_.faceIHi < 0) out.c_.faceIHi = g.nx - 1;
    if (out.c_.faceKHi < 0) out.c_.faceKHi = g.nz - 1;
    if (!c.farNormal.empty())
    {
        if (c.farNormal.size() != std::size_t(g.nx) * g.nz)
            return std::unexpected(hipErrorInvalidValue);
        std::vector<Real> fn(3 * c.farNormal.size());
        for (std::size_t q = 0; q < c.farNormal.size(); ++q)
            for (int a = 0; a < 3; ++a) fn[3 * q + a] = Real(c.farNormal[q][a]);
        auto b = DeviceScalar::make(fn.size());
        if (!b) return std::unexpected(b.error());
        if (auto e = b->upload(fn.data()); e != hipSuccess)
            return std::unexpected(e);
        out.farN_ = std::move(*b);
    }
    if (!c.wallD1.empty())
    {
        // omega_w coefficient per wall face, rounded once from double.
        std::vector<Real> wc(c.wallD1.size());
        for (std::size_t q = 0; q < wc.size(); ++q)
            wc[q] = Real(sst::omegaWallFactor * c.tr.muRef /
                         (c.wallD1[q] * c.wallD1[q]));
        auto w = DeviceScalar::make(wc.size());
        auto d = DeviceBuffer<double>::make(c.wallD1.size());
        if (!w) return std::unexpected(w.error());
        if (!d) return std::unexpected(d.error());
        if (auto e = w->upload(wc.data()); e != hipSuccess)
            return std::unexpected(e);
        if (auto e = d->upload(c.wallD1.data()); e != hipSuccess)
            return std::unexpected(e);
        out.wcoef_ = std::move(*w);
        out.d1_ = std::move(*d);
    }
    auto red = ReduceScratch::make();
    if (!red) return std::unexpected(red.error());
    out.red_ = std::move(*red);
    return out;
}

void DeviceOGridBc::apply(DeviceCompState& s) const
{
    assert(s.nT == 0 ||
           (c_.rans != RansModel::None && s.nT == ransScalars(c_.rans)));
    RowArgs a{};
    a.isothermal = c_.thermal == OGridThermal::Isothermal ? 1 : 0;
    a.gam = Real(c_.gamma);
    a.Rgas = Real(c_.Rgas);
    a.Twall = Real(c_.Twall);
    a.uwx = uwx_.data();
    a.uwy = uwy_.data();
    a.as = aspan_.data();
    a.rhoI = Real(c_.rhoInf);
    a.uI = Real(u_);
    a.pI = Real(c_.pInf);
    a.cI = Real(cInf_);
    a.sigma = Real(c_.sigma);
    a.band = Real(c_.band);
    a.cosT = cosT_.data();
    a.sinT = sinT_.data();
    a.nrm = farN_.data();
    a.vv = viscView(c_.tr);
    a.sst = ransSst(c_.rans) ? 1 : 0;
    a.phi0 = Real(c_.phiInf[0]);
    a.phi1 = Real(c_.phiInf[1]);
    a.wcoef = wcoef_.data();
    a.rho = s.rho.data();
    a.mx = s.mx.data();
    a.my = s.my.data();
    a.mz = s.mz.data();
    a.E = s.E.data();
    a.t0 = s.nT > 0 ? s.t[0].data() : nullptr;
    a.t1 = s.nT > 1 ? s.t[1].data() : nullptr;
    if (s.compensated())
    {
        a.lo.n = 5 + s.nT;
        for (int v = 0; v < 5; ++v) a.lo.p[v] = s.lo[v].data();
        for (int i = 0; i < s.nT; ++i) a.lo.p[5 + i] = s.tlo[i].data();
    }
    const unsigned nb = launchBlocks(std::size_t(g_.nx) * g_.nz);
    if (farN_.size() == 0) kOGridRows<false><<<nb, blockSize>>>(g_, a);
    else kOGridRows<true><<<nb, blockSize>>>(g_, a);
    checkLastKernel("kOGridRows");
}

DeviceOGridBc::Surface DeviceOGridBc::surface(DeviceComp& solver,
                                              const DeviceCompState& s)
{
    const DeviceCurv* tab = solver.curvTables();
    // The runner pairs this only with the curvilinear DeviceComp built from the
    // O-grid's own metrics, so a Cartesian solver here is a programmer error.
    assert(tab && "the O-grid surface tally needs the curvilinear DeviceComp");
    if (!tab) return {0.0, 0.0, 0.0, 0.0, 0.0};
    solver.primitives(s);
    const PrimView q{s.rho.data(),      solver.u().data(), solver.v().data(),
                     solver.w().data(), solver.e().data(), solver.p().data(),
                     solver.T().data()};
    const std::size_t n = std::size_t(g_.nx) * g_.nz;
    // Same MrfView as the rhs (invariant: the tally is the budget of the
    // solved equations); inactive when no zone is installed.
    const MrfView mz = solver.mrfActive() ? solver.mrfView() : MrfView{};
    kOGridWallFlux<<<launchBlocks(n), blockSize>>>(
        g_, viscView(solver.transport()), q, tab->view(), mz, wpx_.data(),
        wpy_.data(), Real(c_.cx), Real(c_.cy), acc_.data());
    checkLastKernel("kOGridWallFlux");
    const bool ranged = c_.faceILo > 0 || c_.faceIHi < g_.nx - 1 ||
                        c_.faceKLo > 0 || c_.faceKHi < g_.nz - 1;
    if (ranged)
    {
        kTallyMask<<<launchBlocks(n), blockSize>>>(g_, 5, c_.faceILo, c_.faceIHi,
                                                   c_.faceKLo, c_.faceKHi,
                                                   acc_.data());
        checkLastKernel("kTallyMask");
    }
    double f[5];
    for (int c = 0; c < 5; ++c)
        f[c] = sumDouble(acc_.data() + std::size_t(c) * n, n, red_) *
               c_.tallyScale;
    return {f[0], f[1], f[2], f[3], f[4]};
}

MutHook DeviceOGridBc::mutHook() const
{
    return [g = g_](Real* nut, Real* f1) {
        const std::size_t nxz = std::size_t(g.nx) * g.nz;
        kOGridMut<<<launchBlocks(nxz), blockSize>>>(g, nut, f1);
        checkLastKernel("kOGridMut");
    };
}

DeviceOGridBc::Faces DeviceOGridBc::surfaceFaces(DeviceComp& solver,
                                                 const DeviceCompState& s)
{
    const DeviceCurv* tab = solver.curvTables();
    assert(tab && "the O-grid surface needs the curvilinear DeviceComp");
    Faces out;
    if (!tab) return out;
    solver.primitives(s);
    const PrimView q{s.rho.data(),      solver.u().data(), solver.v().data(),
                     solver.w().data(), solver.e().data(), solver.p().data(),
                     solver.T().data()};
    const std::size_t n = std::size_t(g_.nx) * g_.nz;
    const MrfView mz = solver.mrfActive() ? solver.mrfView() : MrfView{};
    kOGridFaces<<<launchBlocks(n), blockSize>>>(
        g_, viscView(solver.transport()), q, tab->view(), mz, d1_.data(),
        faces_.data());
    checkLastKernel("kOGridFaces");
    std::vector<double> h(14 * n);
    if (faces_.download(h.data()) != hipSuccess) return out;
    auto span = [&](int c) {
        return std::vector<double>(h.begin() + std::ptrdiff_t(c * n),
                                   h.begin() + std::ptrdiff_t((c + 1) * n));
    };
    for (int a = 0; a < 3; ++a) {
        out.fp[a] = span(a);
        out.fv[a] = span(3 + a);
        out.n[a] = span(6 + a);
    }
    out.heat = span(9);
    out.area = span(10);
    out.p = span(11);
    out.yplus = span(12);
    return out;
}

} // namespace dns::gpu
