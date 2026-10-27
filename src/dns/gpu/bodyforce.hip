#include "dns/gpu/bodyforce.hpp"

#include "dns/gpu/device_util.hpp"

#include <algorithm>
#include <cmath>

namespace dns::gpu {
namespace {

// Pressure force the projection exerts on the body inside the index box:
// F = -sum phi_c * (a_+ - a_c) * A_face (discrete aperture identity, the exact
// surface integral of phi over the embedded boundary).
__global__ void kBodyPressure(GpuGrid g, const Real* lam, const Real* ax,
                              const Real* ay, const Real* az, const Real* phi,
                              BodyBox b, Real* fx, Real* fy, Real* fz)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    fx[c] = 0;
    fy[c] = 0;
    fz[c] = 0;
    if (lam[c] <= 0 || i < b.i0 || i > b.i1 || j < b.j0 || j > b.j1 || k < b.k0
        || k > b.k1)
        return;
    fx[c] = -phi[c] * (ax[gidx(g, i + 1, j, k)] - ax[c]) * g.hy * g.hz;
    fy[c] = -phi[c] * (ay[gidx(g, i, j + 1, k)] - ay[c]) * g.hx * g.hz;
    // At nz == 1 the k+1 neighbour wraps to the cell itself, so the difference
    // is identically zero and fz stays exactly 0 -- the 2D restriction, free.
    fz[c] = -phi[c] * (az[gidx(g, i, j, k + 1)] - az[c]) * g.hx * g.hy;
}

// Friction force the masked Laplacian exerts on the body. Momentum leaves the
// fluid only through Laplacian links from open faces to closed (masked-zero)
// faces; the closed neighbour holds the ghost-fill mirror value, so each link
// drains nu*(q - q_ghost)*A/d -- exactly the viscous flux the momentum update
// applied, with the wall at its true sub-cell position.
__global__ void kBodyFriction(GpuGrid g, Real nu, const Real* ax, const Real* ay,
                              const Real* az, const Real* u, const Real* v,
                              const Real* w, BodyBox b, Real* fx, Real* fy,
                              Real* fz)
{
    constexpr Real openTol = Real(CutGeometry::openTol);
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    fx[c] = 0;
    fy[c] = 0;
    fz[c] = 0;
    if (i < b.i0 || i > b.i1 || j < b.j0 || j > b.j1 || k < b.k0 || k > b.k1) return;
    const Real lx = g.hy * g.hz / g.hx; // link conductance A/d per direction
    const Real ly = g.hx * g.hz / g.hy;
    const Real lz = g.hx * g.hy / g.hz;
    auto closedLinks = [&](const Real* a, const Real* q) {
        Real s = 0;
        auto link = [&](std::size_t nb, Real l) {
            if (a[nb] <= openTol) s += l * (q[c] - q[nb]);
        };
        link(gidx(g, i - 1, j, k), lx);
        link(gidx(g, i + 1, j, k), lx);
        link(gidx(g, i, j - 1, k), ly);
        link(gidx(g, i, j + 1, k), ly);
        // nz == 1: both z neighbours ARE this cell, whose aperture passed the
        // > openTol guard below, so neither link is taken and the term is
        // exactly zero. No 2D/3D branch is needed or wanted.
        link(gidx(g, i, j, k - 1), lz);
        link(gidx(g, i, j, k + 1), lz);
        return s;
    };
    if (ax[c] > openTol) fx[c] = nu * closedLinks(ax, u);
    if (ay[c] > openTol) fy[c] = nu * closedLinks(ay, v);
    if (az[c] > openTol) fz[c] = nu * closedLinks(az, w);
}

// SGS counterpart of kBodyFriction: the eddy-diffusion substep drains momentum
// through the same closed links, but with the local nut and its exact per-link
// weights -- term by term those of kEddyDiffuse (les.hip): the normal link
// takes a cell-centre nut, a tangential link the edge average (nxy, nxz, nyz).
// Same A/d normalisation (nu -> nut), so this folds straight into cd/cl/cs.
// At nz == 1 both z neighbours ARE this cell, whose aperture passed the guard,
// so no z link is taken: the u/v sums keep their 2D bits and w (exactly 0
// there) drains an exact zero.
__global__ void kBodySgsFriction(GpuGrid g, const Real* nut, const Real* ax,
                                 const Real* ay, const Real* az, const Real* u,
                                 const Real* v, const Real* w, BodyBox b,
                                 Real* fx, Real* fy, Real* fz)
{
    constexpr Real openTol = Real(CutGeometry::openTol);
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    fx[c] = 0;
    fy[c] = 0;
    fz[c] = 0;
    if (i < b.i0 || i > b.i1 || j < b.j0 || j > b.j1 || k < b.k0 || k > b.k1) return;
    const Real lx = g.hy * g.hz / g.hx;
    const Real ly = g.hx * g.hz / g.hy;
    const Real lz = g.hx * g.hy / g.hz;
    auto n4 = [&](int a, int bb) { // nut at the (a,bb) cell corner
        return Real(0.25)
             * (nut[gidx(g, a, bb, k)] + nut[gidx(g, a - 1, bb, k)]
                + nut[gidx(g, a, bb - 1, k)] + nut[gidx(g, a - 1, bb - 1, k)]);
    };
    auto nxz = [&](int a, int q) { // nut on the (a,q) cell edge in the xz plane
        return Real(0.25)
             * (nut[gidx(g, a, j, q)] + nut[gidx(g, a - 1, j, q)]
                + nut[gidx(g, a, j, q - 1)] + nut[gidx(g, a - 1, j, q - 1)]);
    };
    auto nyz = [&](int bb, int q) { // nut on the (bb,q) cell edge in the yz plane
        return Real(0.25)
             * (nut[gidx(g, i, bb, q)] + nut[gidx(g, i, bb - 1, q)]
                + nut[gidx(g, i, bb, q - 1)] + nut[gidx(g, i, bb - 1, q - 1)]);
    };
    if (ax[c] > openTol)
    {
        Real s = 0;
        auto link = [&](std::size_t nb, Real nl) {
            if (ax[nb] <= openTol) s += nl * (u[c] - u[nb]);
        };
        link(gidx(g, i - 1, j, k), nut[gidx(g, i - 1, j, k)] * lx);
        link(gidx(g, i + 1, j, k), nut[c] * lx);
        link(gidx(g, i, j - 1, k), n4(i, j) * ly);
        link(gidx(g, i, j + 1, k), n4(i, j + 1) * ly);
        link(gidx(g, i, j, k - 1), nxz(i, k) * lz);
        link(gidx(g, i, j, k + 1), nxz(i, k + 1) * lz);
        fx[c] = s;
    }
    if (ay[c] > openTol)
    {
        Real s = 0;
        auto link = [&](std::size_t nb, Real nl) {
            if (ay[nb] <= openTol) s += nl * (v[c] - v[nb]);
        };
        link(gidx(g, i, j - 1, k), nut[gidx(g, i, j - 1, k)] * ly);
        link(gidx(g, i, j + 1, k), nut[c] * ly);
        link(gidx(g, i - 1, j, k), n4(i, j) * lx);
        link(gidx(g, i + 1, j, k), n4(i + 1, j) * lx);
        link(gidx(g, i, j, k - 1), nyz(j, k) * lz);
        link(gidx(g, i, j, k + 1), nyz(j, k + 1) * lz);
        fy[c] = s;
    }
    if (az[c] > openTol)
    {
        Real s = 0;
        auto link = [&](std::size_t nb, Real nl) {
            if (az[nb] <= openTol) s += nl * (w[c] - w[nb]);
        };
        link(gidx(g, i, j, k - 1), nut[gidx(g, i, j, k - 1)] * lz);
        link(gidx(g, i, j, k + 1), nut[c] * lz);
        link(gidx(g, i - 1, j, k), nxz(i, k) * lx);
        link(gidx(g, i + 1, j, k), nxz(i + 1, k) * lx);
        link(gidx(g, i, j - 1, k), nyz(j, k) * ly);
        link(gidx(g, i, j + 1, k), nyz(j + 1, k) * ly);
        fz[c] = s;
    }
}

} // namespace

// The sphere overload IS the AABB overload on the sphere's own AABB --
// arithmetically identical, so it forwards rather than repeating the spans.
BodyBox bodyBox(const GpuGrid& g, double cx, double cy, double cz, double r,
                int margin)
{
    return bodyBox(g, {cx - r, cy - r, cz - r, cx + r, cy + r, cz + r}, margin);
}

BodyBox bodyBox(const GpuGrid& g, const std::array<double, 6>& w, int margin)
{
    auto span = [margin](double lo, double hi, double h, int n) {
        return std::pair{std::max(2, int(lo / h) - margin),
                         std::min(n - 3, int(hi / h) + margin + 1)};
    };
    const auto [i0, i1] = span(w[0], w[3], g.hx, g.nx);
    const auto [j0, j1] = span(w[1], w[4], g.hy, g.ny);
    // Native 2D has no z ghost ring: the single plane k = 0 is the interior.
    if (g.nz == 1) return {i0, i1, j0, j1, 0, 0};
    const auto [k0, k1] = span(w[2], w[5], g.hz, g.nz);
    return {i0, i1, j0, j1, k0, k1};
}

bool overlaps(const BodyBox& a, const BodyBox& b)
{
    return a.i0 <= b.i1 && b.i0 <= a.i1 && a.j0 <= b.j1 && b.j0 <= a.j1
           && a.k0 <= b.k1 && b.k0 <= a.k1;
}

std::expected<DeviceBodyForce, hipError_t>
DeviceBodyForce::make(const GpuGrid& g, BodyBox b)
{
    if (b.i0 > b.i1 || b.j0 > b.j1 || b.k0 > b.k1)
        return std::unexpected(hipErrorInvalidValue);
    if (b.i0 < 0 || b.i1 >= g.nx || b.j0 < 0 || b.j1 >= g.ny || b.k0 < 0
        || b.k1 >= g.nz)
        return std::unexpected(hipErrorInvalidValue);
    auto fx = DeviceScalar::make(g.size());
    if (!fx) return std::unexpected(fx.error());
    auto fy = DeviceScalar::make(g.size());
    if (!fy) return std::unexpected(fy.error());
    auto fz = DeviceScalar::make(g.size());
    if (!fz) return std::unexpected(fz.error());
    return DeviceBodyForce(g, b, std::move(*fx), std::move(*fy), std::move(*fz));
}

DeviceBodyForce::Coeffs
DeviceBodyForce::measure(const DeviceCutGeometry& cut, const DeviceScalar& phi,
                         const DeviceField& ghosted, double nu, double qRef,
                         const DeviceScalar* nut, ReduceScratch& red)
{
    const auto blocks = launchBlocks(g_.size());
    Coeffs c{0, 0, 0, 0};

    kBodyPressure<<<blocks, blockSize>>>(g_, cut.lambda.data(), cut.ax.data(),
                                         cut.ay.data(), cut.az.data(), phi.data(),
                                         box_, fx_.data(), fy_.data(), fz_.data());
    checkLastKernel("bodyPressure");
    c.cd = sumScalar(fx_, red) / qRef;
    c.cl = sumScalar(fy_, red) / qRef;
    c.cs = sumScalar(fz_, red) / qRef;

    kBodyFriction<<<blocks, blockSize>>>(
        g_, Real(nu), cut.ax.data(), cut.ay.data(), cut.az.data(), ghosted.u.data(),
        ghosted.v.data(), ghosted.w.data(), box_, fx_.data(), fy_.data(),
        fz_.data());
    checkLastKernel("bodyFriction");
    c.cdf = sumScalar(fx_, red) / qRef;
    c.cd += c.cdf;
    c.cl += sumScalar(fy_, red) / qRef;
    c.cs += sumScalar(fz_, red) / qRef;

    if (nut)
    {   // Folds into cd/cl/cs; cdf stays nu-only by project convention.
        kBodySgsFriction<<<blocks, blockSize>>>(
            g_, nut->data(), cut.ax.data(), cut.ay.data(), cut.az.data(),
            ghosted.u.data(), ghosted.v.data(), ghosted.w.data(), box_,
            fx_.data(), fy_.data(), fz_.data());
        checkLastKernel("bodySgsFriction");
        c.cd += sumScalar(fx_, red) / qRef;
        c.cl += sumScalar(fy_, red) / qRef;
        // An exact-zero sum (every 2D run) is not added: -0.0 + 0.0 would
        // flip a 2D cs of -0.0 to +0.0, and 2D stays bit for bit.
        const double sz = sumScalar(fz_, red);
        if (sz != 0.0) c.cs += sz / qRef;
    }
    return c;
}

} // namespace dns::gpu
