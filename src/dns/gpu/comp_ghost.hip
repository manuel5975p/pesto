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
// -(dw/ds) u_sample; exact mirror when dw >= ds), pressure/density copied
// (adiabatic Neumann wall).
__global__ void kGhostFill(Real gamma, const int* gcell, const int* gnbr,
                           const Real* gw, const Real* gscale, int m, Real* rho,
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
    rho[c] = ri;
    mx[c] = -a * ri * ui;
    my[c] = -a * ri * vi;
    mz[c] = -a * ri * wi;
    E[c] = pi / (gamma - Real(1)) +
           Real(0.5) * a * a * ri * (ui * ui + vi * vi + wi * wi);
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

// Exact discrete momentum flux fluid -> body: the same faceFlux the rhs uses,
// summed over fluid/ghost interface faces, oriented into the body.
__global__ void kBodyFlux(GpuGrid g, Real mu, Real kappa, PrimView q,
                          const int* fcell, const int* fdir, int m, double* acc)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m) return;
    int i, j, k;
    decompose(g, std::size_t(fcell[t]), i, j, k);
    const int d = fdir[t];
    const int axis = d / 2;
    const bool minus = d & 1;
    const int ox = axis == 0, oy = axis == 1, oz = axis == 2;
    const Flux f = minus
        ? faceFlux(g, mu, kappa, q, i - ox, j - oy, k - oz, axis)
        : faceFlux(g, mu, kappa, q, i, j, k, axis);
    const Real area = axis == 0 ? g.hy * g.hz
                    : axis == 1 ? g.hx * g.hz
                                : g.hx * g.hy;
    const Real sgn = minus ? Real(-1) : Real(1);
    // Staged per face, not atomically accumulated: the reduction below is
    // deterministic and FP64 (see DeviceCompGhost::facc_).
    acc[t] = double(sgn * f.mx * area);
    acc[m + t] = double(sgn * f.my * area);
    acc[2 * m + t] = double(sgn * f.mz * area);
}

// Curvilinear counterpart: faceFluxCurv already integrates over the exact
// face area vector S, so there is no separate area multiply here.
__global__ void kBodyFluxCurv(GpuGrid g, Real mu, Real kappa, PrimView q,
                              CurvView m, const int* fcell, const int* fdir,
                              int n, double* acc)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int i, j, k;
    decompose(g, std::size_t(fcell[t]), i, j, k);
    const int d = fdir[t];
    const int axis = d / 2;
    const bool minus = d & 1;
    const int ox = axis == 0, oy = axis == 1, oz = axis == 2;
    const Flux f = minus
        ? faceFluxCurv(g, mu, kappa, q, m, i - ox, j - oy, k - oz, axis)
        : faceFluxCurv(g, mu, kappa, q, m, i, j, k, axis);
    const Real sgn = minus ? Real(-1) : Real(1);
    acc[t] = double(sgn * f.mx);
    acc[n + t] = double(sgn * f.my);
    acc[2 * n + t] = double(sgn * f.mz);
}

} // namespace

std::expected<DeviceCompGhost, hipError_t> DeviceCompGhost::make(
    const Grid& g, const Sdf& sdf, double gamma, double rhoInf, double pInf,
    const double* bbox)
{
    const double L[3] = {g.nx * g.hx, g.ny * g.hy, g.nz * g.hz};
    return make(g, sdf, gamma, rhoInf, pInf, bbox,
                AxisRemap3{Focal1D::identity(L[0]), Focal1D::identity(L[1]),
                           Focal1D::identity(L[2])});
}

std::expected<DeviceCompGhost, hipError_t> DeviceCompGhost::make(
    const Grid& g, const Sdf& sdf, double gamma, double rhoInf, double pInf,
    const double* bbox, const AxisRemap3& R)
{
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const double hx = g.hx, hy = g.hy, hz = g.hz;
    const bool curv = R.fx.active || R.fy.active || R.fz.active;
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

    std::vector<int> gcell, gnbr, deep, fcell, fdir, solid;
    std::vector<Real> gw, gscale;
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
                    }
                }
            }

    DeviceCompGhost out;
    out.g_ = GpuGrid::from(g);
    out.gamma_ = Real(gamma);
    out.rhoInf_ = Real(rhoInf);
    out.pInf_ = Real(pInf);
    out.nGhost_ = int(gcell.size());
    out.nDeep_ = int(deep.size());
    out.nFace_ = int(fcell.size());
    out.curv_ = curv;
    out.solid_ = std::move(solid);
    out.deepHost_ = deep;
    auto up = [](auto& dst, const auto& src) -> hipError_t {
        if (src.empty()) return hipSuccess;
        auto b = DeviceBuffer<std::decay_t<decltype(src[0])>>::make(src.size());
        if (!b) return b.error();
        if (auto e = b->upload(src.data()); e != hipSuccess) return e;
        dst = std::move(*b);
        return hipSuccess;
    };
    if (auto e = up(out.gcell_, gcell); e != hipSuccess) return std::unexpected(e);
    if (auto e = up(out.gnbr_, gnbr); e != hipSuccess) return std::unexpected(e);
    if (auto e = up(out.gw_, gw); e != hipSuccess) return std::unexpected(e);
    if (auto e = up(out.gscale_, gscale); e != hipSuccess) return std::unexpected(e);
    if (auto e = up(out.deep_, deep); e != hipSuccess) return std::unexpected(e);
    if (auto e = up(out.fcell_, fcell); e != hipSuccess) return std::unexpected(e);
    if (auto e = up(out.fdir_, fdir); e != hipSuccess) return std::unexpected(e);
    auto acc = DeviceBuffer<double>::make(std::size_t(std::max(out.nFace_, 1)) * 3);
    if (!acc) return std::unexpected(acc.error());
    out.facc_ = std::move(*acc);
    auto red = ReduceScratch::make();
    if (!red) return std::unexpected(red.error());
    out.red_ = std::move(*red);
    return out;
}

void DeviceCompGhost::apply(DeviceCompState& s)
{
    const Real EInf = pInf_ / (gamma_ - Real(1));
    if (nDeep_ > 0) {
        kQuiescent<<<launchBlocks(nDeep_), blockSize>>>(
            deep_.data(), nDeep_, rhoInf_, EInf, s.rho.data(), s.mx.data(),
            s.my.data(), s.mz.data(), s.E.data());
        checkLastKernel("kQuiescent");
    }
    if (nGhost_ > 0) {
        kGhostFill<<<launchBlocks(nGhost_), blockSize>>>(
            gamma_, gcell_.data(), gnbr_.data(), gw_.data(), gscale_.data(),
            nGhost_, s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
            s.E.data());
        checkLastKernel("kGhostFill");
    }
}

DeviceCompGhost::Force DeviceCompGhost::force(DeviceComp& solver,
                                              const DeviceCompState& s)
{
    if (nFace_ == 0) return {0.0, 0.0, 0.0};
    solver.primitives(s);
    const PrimView q{s.rho.data(),      solver.u().data(), solver.v().data(),
                     solver.w().data(), solver.e().data(), solver.p().data(),
                     solver.T().data()};
    if (curv_) {
        const DeviceCurv* t = solver.curvTables();
        assert(t && "focal ghost force needs the curvilinear DeviceComp");
        if (!t) return {0.0, 0.0, 0.0};
        const CurvView cv = t->view();
        kBodyFluxCurv<<<launchBlocks(nFace_), blockSize>>>(
            g_, solver.mu(), solver.kappaCond(), q, cv, fcell_.data(),
            fdir_.data(), nFace_, facc_.data());
        checkLastKernel("kBodyFluxCurv");
    } else {
        kBodyFlux<<<launchBlocks(nFace_), blockSize>>>(
            g_, solver.mu(), solver.kappaCond(), q, fcell_.data(), fdir_.data(),
            nFace_, facc_.data());
        checkLastKernel("kBodyFlux");
    }
    // Three deterministic FP64 reductions over disjoint spans of facc_.
    const std::size_t m = std::size_t(nFace_);
    double f[3];
    for (int c = 0; c < 3; ++c)
        f[c] = sumDouble(facc_.data() + c * m, m, red_);
    return {f[0], f[1], f[2]};
}

} // namespace dns::gpu
