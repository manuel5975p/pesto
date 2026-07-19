#include "dns/gpu/ghostfill.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace dns::gpu {

namespace {

__global__ void kGhostFill(std::size_t count, const int* face, const int* comp,
                           const int* nbr, const Real* w, Real* u, Real* v,
                           Real* wz)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) return;
    Real* q = comp[t] == 0 ? u : comp[t] == 1 ? v : wz;
    Real s = 0;
    for (int m = 0; m < 8; ++m)
        s += w[8 * t + m] * q[nbr[8 * t + m]];
    q[face[t]] = s;
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

    // Face-center offsets per component in lattice units (x/h space).
    const std::array<std::array<double, 3>, 3> off{
        {{0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}}};
    const std::array<const Scalar*, 3> ap{&cut.ax, &cut.ay, &cut.az};

    Build b;
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

                    // Image point mirrored across the wall (phi < 0).
                    const double xi = px - 2.0 * phi * nx;
                    const double yi = py - 2.0 * phi * ny;
                    const double zi = pz - 2.0 * phi * nz;

                    // Trilinear on the component's face lattice, open faces only.
                    const double s0 = xi / g.hx - off[comp][0];
                    const double s1 = yi / g.hy - off[comp][1];
                    const double s2 = zi / g.hz - off[comp][2];
                    const int i0 = int(std::floor(s0));
                    const int j0 = int(std::floor(s1));
                    const int k0 = int(std::floor(s2));
                    const double fx = s0 - i0, fy = s1 - j0, fz = s2 - k0;

                    std::array<int, 8> nbrs{};
                    std::array<double, 8> wt{};
                    double wsum = 0.0;
                    for (int m = 0; m < 8; ++m)
                    {
                        const int di = m & 1, dj = (m >> 1) & 1, dk = (m >> 2) & 1;
                        const auto id = g.idx(pmod(i0 + di, g.nx),
                                              pmod(j0 + dj, g.ny),
                                              pmod(k0 + dk, g.nz));
                        nbrs[m] = int(id);
                        if (a[id] <= tol) continue; // ghost/solid sample: skip
                        wt[m] = (di ? fx : 1.0 - fx) * (dj ? fy : 1.0 - fy)
                              * (dk ? fz : 1.0 - fz);
                        wsum += wt[m];
                    }
                    if (wsum < 0.25) continue; // image buried: stay masked zero

                    for (double& x : wt) x = -x / wsum; // mirror + renormalize
                    b.add(int(c), comp, nbrs, wt);
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
                           std::move(*w), n);
}

hipError_t DeviceGhostFill::fillFrom(const DeviceField& in, DeviceField& diff) const
{
    const std::size_t bytes = in.u.size() * sizeof(Real);
    if (auto e = hipMemcpy(diff.u.data(), in.u.data(), bytes,
                           hipMemcpyDeviceToDevice); e != hipSuccess)
        return e;
    if (auto e = hipMemcpy(diff.v.data(), in.v.data(), bytes,
                           hipMemcpyDeviceToDevice); e != hipSuccess)
        return e;
    if (auto e = hipMemcpy(diff.w.data(), in.w.data(), bytes,
                           hipMemcpyDeviceToDevice); e != hipSuccess)
        return e;
    apply(diff);
    return hipSuccess;
}

void DeviceGhostFill::apply(DeviceField& U) const
{
    if (count_ == 0) return;
    kGhostFill<<<launchBlocks(count_), blockSize>>>(
        count_, face_.data(), comp_.data(), nbr_.data(), w_.data(), U.u.data(),
        U.v.data(), U.w.data());
    checkLastKernel("ghostFill");
}

} // namespace dns::gpu
