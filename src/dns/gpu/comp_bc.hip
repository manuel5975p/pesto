#include "dns/gpu/comp_bc.hpp"

#include "dns/gpu/device_util.hpp"

namespace dns::gpu {
namespace {

// Inflow (i <= 1): rho/u fixed, p copied from i = 2 (Neumann).
// Outflow (i = nx-1): state copied from i = nx-2, p relaxed toward pInf by
// alpha per stage (delta form: exact extrapolation at alpha = 0, so a uniform
// freestream stays a fixed point for any alpha).
// Reads only i in {2, nx-2}: disjoint from the written rows.
__global__ void kBcX(GpuGrid g, Real gamma, Real rhoInf, Real mxInf, Real pInf,
                     Real alpha, Real* rho, Real* mx, Real* my, Real* mz,
                     Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nyz = std::size_t(g.ny) * g.nz;
    if (t >= 3 * nyz) return;
    const int row = int(t / nyz); // 0,1: inflow columns; 2: outflow column
    const int j = int(t % g.ny), k = int((t / g.ny) % g.nz);
    if (row < 2) {
        const std::size_t c = gidx(g, row, j, k);
        const std::size_t s = gidx(g, 2, j, k);
        // Delta form copies the interior pressure exactly (no gamma
        // round-trip): E = E_s - ke_s + ke_inf.
        const Real ke = Real(0.5) *
            (mx[s] * mx[s] + my[s] * my[s] + mz[s] * mz[s]) / rho[s];
        const Real keInf = Real(0.5) * mxInf * mxInf / rhoInf;
        rho[c] = rhoInf;
        mx[c] = mxInf;
        my[c] = Real(0);
        mz[c] = Real(0);
        E[c] = E[s] - ke + keInf;
    } else {
        const std::size_t c = gidx(g, g.nx - 1, j, k);
        const std::size_t s = gidx(g, g.nx - 2, j, k);
        const Real ke = Real(0.5) *
            (mx[s] * mx[s] + my[s] * my[s] + mz[s] * mz[s]) / rho[s];
        const Real ps = (gamma - Real(1)) * (E[s] - ke);
        rho[c] = rho[s];
        mx[c] = mx[s];
        my[c] = my[s];
        mz[c] = mz[s];
        E[c] = E[s] + alpha * (pInf - ps) / (gamma - Real(1));
    }
}

// Free-slip mirror on the y extremes: j=0 := mirror(j=1), j=ny-1 :=
// mirror(j=ny-2); tangential components and thermodynamics symmetric, normal
// momentum antisymmetric. Writes j in {0, ny-1}, reads {1, ny-2}: disjoint.
__global__ void kBcWallY(GpuGrid g, Real* rho, Real* mx, Real* my, Real* mz,
                         Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxz = std::size_t(g.nx) * g.nz;
    if (t >= 2 * nxz) return;
    const bool hi = t >= nxz;
    const int i = int(t % g.nx), k = int((t / g.nx) % g.nz);
    const std::size_t c = gidx(g, i, hi ? g.ny - 1 : 0, k);
    const std::size_t s = gidx(g, i, hi ? g.ny - 2 : 1, k);
    rho[c] = rho[s];
    mx[c] = mx[s];
    my[c] = -my[s];
    mz[c] = mz[s];
    E[c] = E[s];
}

// Same for the z extremes (only launched when nz > 1).
__global__ void kBcWallZ(GpuGrid g, Real* rho, Real* mx, Real* my, Real* mz,
                         Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxy = std::size_t(g.nx) * g.ny;
    if (t >= 2 * nxy) return;
    const bool hi = t >= nxy;
    const int i = int(t % g.nx), j = int((t / g.nx) % g.ny);
    const std::size_t c = gidx(g, i, j, hi ? g.nz - 1 : 0);
    const std::size_t s = gidx(g, i, j, hi ? g.nz - 2 : 1);
    rho[c] = rho[s];
    mx[c] = mx[s];
    my[c] = my[s];
    mz[c] = -mz[s];
    E[c] = E[s];
}

} // namespace

std::expected<DeviceCompBc, hipError_t> DeviceCompBc::make(const Grid& g,
                                                           double gamma,
                                                           const Opts& o)
{
    DeviceCompBc bc;
    bc.g_ = GpuGrid::from(g);
    bc.gamma_ = Real(gamma);
    bc.o_ = o;
    return bc;
}

void DeviceCompBc::apply(DeviceCompState& s)
{
    const std::size_t nyz = std::size_t(g_.ny) * g_.nz;
    kBcX<<<launchBlocks(3 * nyz), blockSize>>>(
        g_, gamma_, Real(o_.rhoInf), Real(o_.rhoInf * o_.uInf), Real(o_.pInf),
        Real(o_.pRelax), s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
        s.E.data());
    checkLastKernel("kBcX");
    if (o_.wallY) {
        const std::size_t nxz = std::size_t(g_.nx) * g_.nz;
        kBcWallY<<<launchBlocks(2 * nxz), blockSize>>>(
            g_, s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data());
        checkLastKernel("kBcWallY");
    }
    if (o_.wallZ && g_.nz > 1) {
        const std::size_t nxy = std::size_t(g_.nx) * g_.ny;
        kBcWallZ<<<launchBlocks(2 * nxy), blockSize>>>(
            g_, s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data());
        checkLastKernel("kBcWallZ");
    }
}

} // namespace dns::gpu
