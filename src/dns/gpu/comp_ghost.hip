#include "dns/gpu/comp_ghost.hpp"

#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/device_util.hpp"

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
// the EOS. iso == 0 executes exactly the arithmetic it always did.
//
// uw != nullptr makes the wall MOVE at that per-ghost velocity. The image is
// the same one the isothermal temperature uses, about u_wall instead of about
// rest: with s the outward wall-normal coordinate, u(s) = Uw + (u_s - Uw)
// s/dsmp, so u(-dw) = (1 + a) Uw - a u_s. Uw = 0 recovers the resting mirror
// and a == 1 makes it the familiar 2 Uw - u_s. The resting branch is kept
// VERBATIM rather than folded into the general form: a*a*(ui*ui) and
// (a*ui)*(a*ui) are not the same float, and the static path must not move.
__global__ void kGhostFill(Real gamma, int iso, Real Twall, Real Rgas,
                           const int* gcell, const int* gnbr, const Real* gw,
                           const Real* gscale, const Real* uw, int m, Real* rho,
                           Real* mx, Real* my, Real* mz, Real* E)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
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
    const Real a = gscale[q];
    const int c = gcell[q];
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
    // Pressure is still copied (zero wall-normal dp/dn) and density follows
    // from p = rho R T, so the ghost state is thermodynamically consistent.
    Real rg = ri;
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
        rg = pi / (Rgas * Tg);
    }
    rho[c] = rg;
    if (uw) {
        const Real b = Real(1) + a;
        const Real ug = b * uw[3 * q + 0] - a * ui;
        const Real vg = b * uw[3 * q + 1] - a * vi;
        const Real wg = b * uw[3 * q + 2] - a * wi;
        mx[c] = rg * ug;
        my[c] = rg * vg;
        mz[c] = rg * wg;
        E[c] = pi / (gamma - Real(1)) +
               Real(0.5) * rg * (ug * ug + vg * vg + wg * wg);
        return;
    }
    mx[c] = -a * rg * ui;
    my[c] = -a * rg * vi;
    mz[c] = -a * rg * wi;
    E[c] = pi / (gamma - Real(1)) +
           Real(0.5) * a * a * rg * (ui * ui + vi * vi + wi * wi);
}

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
                          MrfView mz)
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
    const Flux f = faceFlux(g, vv, q, fi, fj, fk, axis, nullptr, Real(0),
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
                              int n, double* acc, MrfView mz)
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
    const Flux f = faceFluxCurv(g, vv, q, m, fi, fj, fk, axis, nullptr,
                                Real(0), mz.active ? &mz : nullptr);
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
    const double* bbox, WallThermal wall, const WallVelocity* uwall)
{
    const double L[3] = {g.nx * g.hx, g.ny * g.hy, g.nz * g.hz};
    return make(g, sdf, gamma, rhoInf, pInf, bbox,
                AxisRemap3{Focal1D::identity(L[0]), Focal1D::identity(L[1]),
                           Focal1D::identity(L[2])},
                wall, uwall);
}

void CompGhostCapacity::cover(const CompGhostTables& t)
{
    nGhost = std::max(nGhost, t.nGhost());
    nDeep = std::max(nDeep, t.nDeep());
    nFace = std::max(nFace, t.nFace());
}

CompGhostTables classifyCompGhost(const Grid& g, const GhostSdf& sdf,
                                  const double* bbox, const AxisRemap3& R,
                                  const GhostWallVelocity* uwall)
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

    CompGhostTables T;
    auto& gcell = T.gcell;
    auto& gnbr = T.gnbr;
    auto& deep = T.deep;
    auto& fcell = T.fcell;
    auto& fdir = T.fdir;
    auto& solid = T.solid;
    auto& gw = T.gw;
    auto& gscale = T.gscale;
    auto& uw = T.uwall;
    auto& fpos = T.fpos;
    std::vector<char> isGhost(g.size(), 0);

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
                const double hl = minSpacing(dloc);
                // The mirror band must be at least two cells deep along
                // EVERY axis: the interior stencil reaches two cells, and a
                // quiescent deep-solid cell inside its reach is a hard
                // discontinuity against the fluid. The threshold therefore
                // follows the WIDEST local spacing -- with the smallest, an
                // anisotropic fine region leaves the band under two cells
                // thick along the wide axis (measured on the Re100 cylinder
                // at aspect 1.7: cd 1.49 -> -7.6, a spurious thrust).
                // Isotropic cells make the two identical, so the Cartesian
                // construction is untouched.
                if (phi[c] < -2.0 * maxSpacing(dloc)) {
                    deep.push_back(int(c));
                    continue;
                }
                const double x = cx(i), y = cy(j), z = cz(k);
                const double d = 1e-3 * hl;
                double nv[3] = {(sdf(x + d, y, z) - sdf(x - d, y, z)) / (2 * d),
                                (sdf(x, y + d, z) - sdf(x, y - d, z)) / (2 * d),
                                (sdf(x, y, z + d) - sdf(x, y, z - d)) / (2 * d)};
                if (nz == 1) nv[2] = 0.0;
                const double nn =
                    std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
                if (nn < 1e-12) {
                    deep.push_back(int(c));
                    continue;
                }
                for (double& v : nv) v /= nn;
                // Sample at ds beyond the wall (>= 0.75 of the finest local
                // spacing so the stencil always reaches fluid, even for cells
                // barely inside).
                const double dw = -phi[c];
                const double dsmp = std::max(dw, 0.75 * hl);
                // The sample point is a PHYSICAL offset along the physical
                // normal; the per-axis inverse takes it back to reference
                // space, where the cell-centre lattice is uniform and the
                // trilinear weights are the Cartesian ones. The map is
                // box-preserving, so the physical wrap is the reference wrap.
                const double xi = R.fx.inverse(pmod(x + (dw + dsmp) * nv[0], Lx));
                const double yi = R.fy.inverse(pmod(y + (dw + dsmp) * nv[1], Ly));
                const double zi = R.fz.inverse(pmod(z + (dw + dsmp) * nv[2], Lz));
                // Trilinear over cell centres, fluid cells only, renormalized.
                const double fx = xi / hx - 0.5, fy = yi / hy - 0.5,
                             fz = zi / hz - 0.5;
                const int i0 = int(std::floor(fx)), j0 = int(std::floor(fy)),
                          k0 = int(std::floor(fz));
                const double ax = fx - i0, ay = fy - j0, az = fz - k0;
                int nb[8];
                double w[8];
                double wsum = 0.0;
                for (int s = 0; s < 8; ++s) {
                    const int sx = s & 1, sy = (s >> 1) & 1, sz = s >> 2;
                    const std::size_t cc =
                        g.idx((i0 + sx + nx) % nx, (j0 + sy + ny) % ny,
                              ((k0 + sz) % nz + nz) % nz);
                    const double ww = (sx ? ax : 1 - ax) * (sy ? ay : 1 - ay) *
                                      (sz ? az : 1 - az);
                    nb[s] = int(cc);
                    w[s] = phi[cc] > 0.0 ? ww : 0.0;
                    wsum += w[s];
                }
                if (wsum < 0.05) {
                    deep.push_back(int(c));
                    continue;
                }
                gcell.push_back(int(c));
                isGhost[c] = 1;
                gscale.push_back(Real(dw / dsmp));
                for (int s = 0; s < 8; ++s) {
                    gnbr.push_back(nb[s]);
                    gw.push_back(Real(w[s] / wsum));
                }
                // Wall velocity at the FOOT OF THE NORMAL, not at the ghost
                // centre: that point is on the surface, so a rotor's omega x r
                // there is the actual surface speed the fluid sees.
                if (uwall) {
                    const auto v = (*uwall)(x + dw * nv[0], y + dw * nv[1],
                                            z + dw * nv[2]);
                    uw.push_back(Real(v[0]));
                    uw.push_back(Real(v[1]));
                    uw.push_back(Real(v[2]));
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
    const WallVelocity* uwall)
{
    CompGhostTables T = classifyCompGhost(g, sdf, bbox, R, uwall);
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
    if (auto e = alloc(out.gscale_, ng); e != hipSuccess) return std::unexpected(e);
    if (auto e = alloc(out.uwall_, t.uwall.empty() ? 0 : 3 * ng);
        e != hipSuccess)
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
    if (t.uwall.empty() != (uwall_.data() == nullptr))
        return hipErrorInvalidValue;
    auto put = [](auto& dst, const auto& src) -> hipError_t {
        if (src.empty()) return hipSuccess;
        return hipMemcpy(dst.data(), src.data(),
                         src.size() * sizeof(src[0]), hipMemcpyHostToDevice);
    };
    if (auto e = put(gcell_, t.gcell); e != hipSuccess) return e;
    if (auto e = put(gnbr_, t.gnbr); e != hipSuccess) return e;
    if (auto e = put(gw_, t.gw); e != hipSuccess) return e;
    if (auto e = put(gscale_, t.gscale); e != hipSuccess) return e;
    if (auto e = put(uwall_, t.uwall); e != hipSuccess) return e;
    if (auto e = put(deep_, t.deep); e != hipSuccess) return e;
    if (auto e = put(fcell_, t.fcell); e != hipSuccess) return e;
    if (auto e = put(fdir_, t.fdir); e != hipSuccess) return e;
    if (auto e = put(fpos_, t.fpos); e != hipSuccess) return e;
    nGhost_ = t.nGhost();
    nDeep_ = t.nDeep();
    nFace_ = t.nFace();
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
            gw_.data(), gscale_.data(), uwall_.data(), nGhost_, s.rho.data(),
            s.mx.data(), s.my.data(), s.mz.data(), s.E.data());
        checkLastKernel("kGhostFill");
    }
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
    if (curv_) {
        const DeviceCurv* t = solver.curvTables();
        assert(t && "focal ghost force needs the curvilinear DeviceComp");
        if (!t) return {};
        const CurvView cv = t->view();
        kBodyFluxCurv<<<launchBlocks(nFace_), blockSize>>>(
            g_, viscView(solver.transport()), iso_, q, cv, fcell_.data(),
            fdir_.data(), nFace_, facc_.data(), mz);
        checkLastKernel("kBodyFluxCurv");
    } else {
        kBodyFlux<<<launchBlocks(nFace_), blockSize>>>(
            g_, viscView(solver.transport()), iso_, q, fcell_.data(),
            fdir_.data(), nFace_, facc_.data(), mz);
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
