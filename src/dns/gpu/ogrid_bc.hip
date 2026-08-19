#include "dns/gpu/ogrid_bc.hpp"

#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/internal/wall_ghost.hpp"

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
// uwx/uwy (both null or both set, nx entries): wall velocity for a rigidly
// rotating body (--rotor-frame mrf). The moving image is u0 = 2 u_wall - u1
// -- the resting mirror's general form at reflection factor exactly 1 -- with
// the thermodynamic state built exactly as the resting branch builds it
// (adiabatic: rho and p copied, so T0 == T1 and the Fourier flux vanishes;
// isothermal: the same floored T0 and rho0). The resting branch is kept
// VERBATIM under the null check rather than folded into the general form:
// -m and 2*rho*uw - m are not the same float at uw == 0.
__global__ void kOGridWall(GpuGrid g, int isothermal, Real gam, Real Rgas,
                           Real Twall, const Real* uwx, const Real* uwy,
                           Real* rho, Real* mx, Real* my, Real* mz, Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxz = std::size_t(g.nx) * g.nz;
    if (t >= nxz) return;
    const int i = int(t % g.nx), k = int(t / g.nx);
    const std::size_t c = gidx(g, i, 0, k);
    const std::size_t s = gidx(g, i, 1, k);
    if (!uwx) {
        if (!isothermal) internal::adiabaticGhost(c, s, rho, mx, my, mz, E);
        else internal::isothermalGhost(c, s, gam, Rgas, Twall, rho, mx, my,
                                       mz, E);
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
    Real r0 = r1;
    if (isothermal) {
        const Real T0 =
            fmaxf(Real(2) * Twall - p1 / (Rgas * r1), Real(0.05) * Twall);
        r0 = p1 / (Rgas * T0);
    }
    rho[c] = r0;
    mx[c] = r0 * u0;
    my[c] = r0 * v0;
    mz[c] = r0 * w0;
    E[c] = p1 / (gam - Real(1)) + r0 * ke0;
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
__global__ void kOGridFar(GpuGrid g, Real gam, Real rhoI, Real uI, Real pI,
                          Real cI, Real sigma, Real band, const Real* cosT,
                          const Real* sinT, Real* rho, Real* mx, Real* my,
                          Real* mz, Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxz = std::size_t(g.nx) * g.nz;
    if (t >= nxz) return;
    const int i = int(t % g.nx), k = int(t / g.nx);
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
                     c.wallPos.size() == std::size_t(g.nx));
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
        std::vector<Real> px(g.nx), py(g.nx), ux(g.nx), uy(g.nx);
        for (int i = 0; i < g.nx; ++i)
        {
            const auto& p = c.wallPos[std::size_t(i)];
            px[std::size_t(i)] = Real(p[0]);
            py[std::size_t(i)] = Real(p[1]);
            ux[std::size_t(i)] = Real(-c.omega * (p[1] - c.cy));
            uy[std::size_t(i)] = Real(c.omega * (p[0] - c.cx));
        }
        for (auto [dst, src] : {std::pair{&out.wpx_, &px}, {&out.wpy_, &py},
                                {&out.uwx_, &ux}, {&out.uwy_, &uy}})
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
    auto red = ReduceScratch::make();
    if (!red) return std::unexpected(red.error());
    out.red_ = std::move(*red);
    return out;
}

void DeviceOGridBc::apply(DeviceCompState& s) const
{
    const std::size_t nxz = std::size_t(g_.nx) * g_.nz;
    kOGridWall<<<launchBlocks(nxz), blockSize>>>(
        g_, c_.thermal == OGridThermal::Isothermal ? 1 : 0, Real(c_.gamma),
        Real(c_.Rgas), Real(c_.Twall), uwx_.data(), uwy_.data(), s.rho.data(),
        s.mx.data(), s.my.data(), s.mz.data(), s.E.data());
    checkLastKernel("kOGridWall");
    kOGridFar<<<launchBlocks(nxz), blockSize>>>(
        g_, Real(c_.gamma), Real(c_.rhoInf), Real(u_), Real(c_.pInf),
        Real(cInf_), Real(c_.sigma), Real(c_.band), cosT_.data(), sinT_.data(),
        s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data());
    checkLastKernel("kOGridFar");
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
    double f[5];
    for (int c = 0; c < 5; ++c)
        f[c] = sumDouble(acc_.data() + std::size_t(c) * n, n, red_);
    return {f[0], f[1], f[2], f[3], f[4]};
}

} // namespace dns::gpu
