#include "dns/gpu/ghostfill.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"

#include <hip/hip_runtime.h>

#include <algorithm> // std::min(initializer_list): NOT in <bits/stl_algobase.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace dns::gpu {

namespace {

// The mirror sum reads only OPEN faces (make() drops every closed sample) and
// writes only closed ones, so the samples may come from a SEPARATE source
// field: the destination is write-only here, and src == dst stays legal.
__global__ void kGhostFill(std::size_t count, const int* __restrict__ face,
                           const int* __restrict__ comp,
                           const int* __restrict__ nbr, const Real* __restrict__ w,
                           const Real* su, const Real* sv, const Real* sw, Real* u,
                           Real* v, Real* wz)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int cp = comp[t];
    const Real* s = cp == 0 ? su : cp == 1 ? sv : sw;
    Real* q = cp == 0 ? u : cp == 1 ? v : wz;
    Real acc = 0;
    for (int m = 0; m < 8; ++m)
        acc += w[8 * t + m] * s[nbr[8 * t + m]];
    q[face[t]] = acc;
}

int pmod(int i, int n) { return ((i % n) + n) % n; }

// Host-side accumulator for one component's ghost entries.
struct Build
{
    std::vector<int> face, comp, nbr;
    std::vector<Real> w;

    void add(int f, int c, const std::array<int, 8>& n,
             const std::array<double, 8>& wt)
    {
        face.push_back(f);
        comp.push_back(c);
        for (int m = 0; m < 8; ++m)
        {
            nbr.push_back(n[m]);
            w.push_back(Real(wt[m]));
        }
    }
};

} // namespace

std::expected<DeviceGhostFill, hipError_t>
DeviceGhostFill::make(const Grid& g, const CutGeometry& cut, const Sdf& sdf)
{
    const double tol = CutGeometry::openTol;
    const double hmin = std::min({g.hx, g.hy, g.hz});
    const double dlt = 1e-3 * hmin; // FD step for the sdf gradient
    // Image search: a planar wall leaves every corner of the interpolation
    // cell in the fluid once the image is one cell DIAGONAL out, so the
    // search spans that distance beyond the mirror point, in 1/16 cell steps.
    // A degenerate z (nz == 1) is not an axis: its hz is arbitrary.
    const int dim = g.nz == 1 ? 2 : 3;
    const double hmax = g.nz == 1 ? std::max(g.hx, g.hy)
                                  : std::max({g.hx, g.hy, g.hz});
    const double step = hmax / 16.0, reach = std::sqrt(double(dim)) * hmax;

    // Face-center offsets per component in lattice units (x/h space).
    const std::array<std::array<double, 3>, 3> off{
        {{0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}}};
    const std::array<const Scalar*, 3> ap{&cut.ax, &cut.ay, &cut.az};

    // Trilinear stencil of component `comp` at physical (x, y, z): the 8
    // sample indices and weights. clean = every sample carrying weight is an
    // open face, i.e. the interpolant is the true trilinear one (exact for
    // linear fields). At nz == 1 the dk = 1 weights are exactly 0.
    struct Stencil
    {
        std::array<int, 8> nbr{};
        std::array<double, 8> w{};
        std::array<std::array<double, 3>, 8> pos{}; // sample face centres
        bool clean = true;
    };
    auto stencil = [&](int comp, double x, double y, double z) {
        const Scalar& a = *ap[comp];
        const double s0 = x / g.hx - off[comp][0];
        const double s1 = y / g.hy - off[comp][1];
        const double s2 = z / g.hz - off[comp][2];
        const int i0 = int(std::floor(s0));
        const int j0 = int(std::floor(s1));
        const int k0 = int(std::floor(s2));
        const double fx = s0 - i0, fy = s1 - j0, fz = s2 - k0;
        Stencil st;
        for (int m = 0; m < 8; ++m)
        {
            const int di = m & 1, dj = (m >> 1) & 1, dk = (m >> 2) & 1;
            const int ii = pmod(i0 + di, g.nx), jj = pmod(j0 + dj, g.ny),
                      kk = pmod(k0 + dk, g.nz);
            const auto id = g.idx(ii, jj, kk);
            st.nbr[m] = int(id);
            st.pos[m] = {(ii + off[comp][0]) * g.hx, (jj + off[comp][1]) * g.hy,
                         (kk + off[comp][2]) * g.hz};
            st.w[m] = (di ? fx : 1.0 - fx) * (dj ? fy : 1.0 - fy)
                    * (dk ? fz : 1.0 - fz);
            if (st.w[m] > 0.0 && a[id] <= tol) st.clean = false;
        }
        return st;
    };

    Build b;
    std::size_t fallback = 0;
    for (int comp = 0; comp < 3; ++comp)
    {
        const Scalar& a = *ap[comp];
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                {
                    const auto c = g.idx(i, j, k);
                    if (a[c] > tol) continue; // open: a real DOF

                    const double px = (i + off[comp][0]) * g.hx;
                    const double py = (j + off[comp][1]) * g.hy;
                    const double pz = (k + off[comp][2]) * g.hz;
                    const double phi = sdf(px, py, pz);
                    if (phi >= 0.0) continue; // box ring / outside body

                    // Reachable by a stencil? Need an open same-component face
                    // in the 3x3x3 index neighborhood.
                    bool nearFluid = false;
                    for (int dk = -1; dk <= 1 && !nearFluid; ++dk)
                        for (int dj = -1; dj <= 1 && !nearFluid; ++dj)
                            for (int di = -1; di <= 1 && !nearFluid; ++di)
                                nearFluid = a[g.idx(i + di, j + dj, k + dk)] > tol;
                    if (!nearFluid) continue; // deep solid stays masked zero

                    // Wall normal from the sdf gradient; degenerate -> stay zero.
                    double nx = sdf(px + dlt, py, pz) - sdf(px - dlt, py, pz);
                    double ny = sdf(px, py + dlt, pz) - sdf(px, py - dlt, pz);
                    double nz = sdf(px, py, pz + dlt) - sdf(px, py, pz - dlt);
                    const double gn = std::sqrt(nx * nx + ny * ny + nz * nz);
                    if (gn < 1e-12) continue;
                    nx /= gn; ny /= gn; nz /= gn;

                    // The image: the foot of the normal (on the wall) plus
                    // dist along n, sampled by the TRUE trilinear interpolant,
                    // and q_ghost = (phi / dist) q(image) -- the profile
                    // linear along the normal through q = 0 at the wall.
                    // dist = -phi is the mirror (factor -1). Where its cell
                    // has a closed corner the image moves out until the cell
                    // is clean: dropping the closed corners and renormalizing
                    // instead reads the field at the open corners' centroid,
                    // an O(h) displacement, i.e. an O(1) error in the wall
                    // gradient the viscous links difference (measured: the
                    // Couette friction tally plateaued at +6% from R/h 13 to
                    // 64). |phi / dist| <= 1 and the weights are convex, so
                    // |q_ghost| <= max |q_nbr| still holds.
                    auto at = [&](double dist) {
                        return stencil(comp, px + (dist - phi) * nx,
                                       py + (dist - phi) * ny,
                                       pz + (dist - phi) * nz);
                    };
                    bool found = false;
                    for (int s = 0; !found && s * step <= reach; ++s)
                    {
                        const double dist = -phi + s * step;
                        const Stencil st = at(dist);
                        if (!st.clean) continue;
                        std::array<double, 8> wt{};
                        for (int m = 0; m < 8; ++m) wt[m] = st.w[m] * (phi / dist);
                        b.add(int(c), comp, st.nbr, wt);
                        found = true;
                    }
                    if (found) continue;

                    // No clean cell within a diagonal: the geometry is not
                    // resolved here (a gap or concave corner under ~2 cells).
                    // Mirror over the open corners, renormalized, with the
                    // factor taken from THEIR weighted wall distance (the
                    // compressible ghost wall's rule): still exact for a
                    // wall-linear profile, capped at the mirror so the image
                    // stays bounded. Below a quarter of open weight, or no
                    // open corner in the fluid, the face stays masked zero.
                    const Stencil st = at(-phi);
                    std::array<double, 8> wt{};
                    double wsum = 0.0, wphi = 0.0;
                    for (int m = 0; m < 8; ++m)
                    {
                        if (st.w[m] <= 0.0 || a[st.nbr[m]] <= tol) continue;
                        const auto& q = st.pos[m];
                        wt[m] = st.w[m];
                        wsum += wt[m];
                        wphi += wt[m] * sdf(q[0], q[1], q[2]);
                    }
                    if (wsum < 0.25 || wphi <= 0.0) continue;
                    // -phi / (wphi / wsum), capped at 1 (the mirror).
                    const double f = std::min(1.0, -phi * wsum / wphi);
                    for (double& x : wt) x = -f * x / wsum;
                    b.add(int(c), comp, st.nbr, wt);
                    ++fallback;
                }
    }

    const std::size_t n = b.face.size();
    auto face = DeviceBuffer<int>::make(std::max<std::size_t>(n, 1));
    if (!face) return std::unexpected(face.error());
    auto comp = DeviceBuffer<int>::make(std::max<std::size_t>(n, 1));
    if (!comp) return std::unexpected(comp.error());
    auto nbr = DeviceBuffer<int>::make(std::max<std::size_t>(8 * n, 1));
    if (!nbr) return std::unexpected(nbr.error());
    auto w = DeviceBuffer<Real>::make(std::max<std::size_t>(8 * n, 1));
    if (!w) return std::unexpected(w.error());
    if (n > 0)
    {
        if (auto e = face->upload(b.face.data()); e != hipSuccess)
            return std::unexpected(e);
        if (auto e = comp->upload(b.comp.data()); e != hipSuccess)
            return std::unexpected(e);
        if (auto e = nbr->upload(b.nbr.data()); e != hipSuccess)
            return std::unexpected(e);
        if (auto e = w->upload(b.w.data()); e != hipSuccess)
            return std::unexpected(e);
    }
    return DeviceGhostFill(std::move(*face), std::move(*comp), std::move(*nbr),
                           std::move(*w), n, fallback);
}

void DeviceGhostFill::fillFrom(const DeviceField& in, DeviceField& diff) const
{
    // One launch, not three blocking D2D copies: at three RK stages that was
    // nine pipeline stalls per step. Same-size fields are the caller's
    // invariant, so a failure here is a programmer error.
    [[maybe_unused]] const hipError_t e = copyField(diff, in);
    assert(e == hipSuccess && "ghostCopy failed");
    fill(in, diff);
}

void DeviceGhostFill::apply(DeviceField& U) const
{
    fill(U, U);
}

void DeviceGhostFill::fill(const DeviceField& src, DeviceField& dst) const
{
    if (count_ == 0) return;
    kGhostFill<<<launchBlocks(count_), blockSize>>>(
        count_, face_.data(), comp_.data(), nbr_.data(), w_.data(), src.u.data(),
        src.v.data(), src.w.data(), dst.u.data(), dst.v.data(), dst.w.data());
    checkLastKernel("ghostFill");
}

} // namespace dns::gpu
