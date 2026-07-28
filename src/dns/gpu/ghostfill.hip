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

// Whole-field copy of the three components in ONE launch. It replaces three
// hipMemcpy D2D calls, each of which is a blocking host synchronization: at
// three RK stages that was nine pipeline stalls per step, for a transfer the
// solver never has to wait on.
__global__ void kCopyField(std::size_t n, const Real* __restrict__ su,
                           const Real* __restrict__ sv, const Real* __restrict__ sw,
                           Real* __restrict__ du, Real* __restrict__ dv,
                           Real* __restrict__ dw)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    du[c] = su[c];
    dv[c] = sv[c];
    dw[c] = sw[c];
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

void DeviceGhostFill::fillFrom(const DeviceField& in, DeviceField& diff) const
{
    const std::size_t n = in.u.size();
    kCopyField<<<launchBlocks(n), blockSize>>>(n, in.u.data(), in.v.data(),
                                               in.w.data(), diff.u.data(),
                                               diff.v.data(), diff.w.data());
    checkLastKernel("ghostCopy");
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
