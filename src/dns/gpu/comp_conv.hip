#include "dns/gpu/comp_conv.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/hotpatch.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/internal/plane_sums.hpp"
#include "dns/gpu/internal/wall_ghost.hpp"

namespace dns::gpu {

namespace {

// The two plates. The ghost state is the shared isothermal no-slip wall
// (internal/wall_ghost.hpp) -- the same one the channel plates and the O-grid
// body surface get. What is specific here is only WHICH temperature each
// column sees: Ttop on the whole lid, Tbot on the floor inside the hot patch
// and Ttop outside it.
__global__ void kConvWallY(GpuGrid g, Real gam, Real Rgas, Real Tbot,
                           Real Ttop, Real patch, int ox, int oz, Real* rho,
                           Real* mx, Real* my, Real* mz, Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i, k;
    bool hi;
    std::size_t c, s;
    if (!internal::yGhostPair(g, t, i, k, hi, c, s)) return;
    const Real Twall =
        hi ? Ttop : (inHotPatch(g, i, k, patch, ox, oz) ? Tbot : Ttop);
    internal::isothermalGhost(c, s, gam, Rgas, Twall, rho, mx, my, mz, E);
}

// Lateral no-slip ADIABATIC mirror (--sides closed). T and p are COPIED --
// hence rho copied by the EOS, and E copied too, the mirrored velocity having
// the same magnitude -- and all three momentum components antimirrored. So the
// face carries exactly zero mass flux (the normal momentum reverses) and
// exactly zero heat flux (no temperature difference across it), and no wall
// temperature enters at all. That last part is the difference from the plates:
// the drop across THOSE is the experiment.
//
// `axis` is 0 for x and 2 for z; the two are one kernel because the state edit
// is identical and only the pair of indices moves (internal::sideGhostPair).
// The edit IS the shared adiabatic no-slip ghost.
__global__ void kConvWallSide(GpuGrid g, int axis, Real* rho, Real* mx, Real* my,
                              Real* mz, Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    bool hi;
    std::size_t c, s;
    if (!internal::sideGhostPair(g, axis, t, hi, c, s)) return;
    internal::adiabaticGhost(c, s, rho, mx, my, mz, E);
}

// Additive rhs contribution: the stage kernel already wrote the interior
// combination, so the source enters scaled by the same cr (comp.hpp,
// StageCombine) that multiplied the interior rhs. Momentum takes -g rho, the
// energy equation the work rate -g rho v = -g my of the SAME stage state.
__global__ void kGravityY(GpuGrid g, int ox, int oz, Real gy, Real cr,
                          const Real* rho, const Real* my, Real* omy, Real* oE)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    // Ghost cells: the wall mirror owns them (the rows always, the lateral
    // columns once that side is a wall).
    if (!inFluid(g, i, j, k, ox, oz)) return;
    omy[c] -= cr * gy * rho[c];
    oE[c] -= cr * gy * my[c];
}

// One thread per y row; each sums its whole (i,k) plane in FP64 in a fixed
// order. ~ny threads is a tiny launch, but the plane sums are exact-order
// deterministic, which the ground rules require of anything reported.
__global__ void kConvPlaneStats(GpuGrid g, int ox, int oz, const Real* rho,
                                const Real* u, const Real* v, const Real* w,
                                const Real* T, double* out)
{
    const int j = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (j >= g.ny) return;
    double s[6] = {0, 0, 0, 0, 0, 0};
    for (int k = oz; k < g.nz - oz; ++k)
        for (int i = ox; i < g.nx - ox; ++i)
        {
            const std::size_t c = gidx(g, i, j, k);
            const double r = rho[c], TT = T[c];
            const double uu = u[c], vv = v[c], ww = w[c];
            s[0] += r;
            s[1] += TT;
            s[2] += vv;
            s[3] += vv * TT;
            s[4] += r * (uu * uu + vv * vv + ww * ww);
            s[5] += r * vv * TT;
        }
    for (int q = 0; q < 6; ++q) out[std::size_t(j) * 6 + q] = s[q];
}

} // namespace

std::expected<DeviceConvWall, hipError_t> DeviceConvWall::make(const Grid& g,
                                                               Config c)
{
    if (g.ny < 4 || !(c.gamma > 1) || !(c.Rgas > 0) || !(c.Tbot > 0)
        || !(c.Ttop > 0) || !(c.patch > 0 && c.patch <= 1))
        return std::unexpected(hipErrorInvalidValue);
    // A walled axis needs its two ghost columns plus at least two fluid cells,
    // or the mirror would read a cell that is itself a ghost.
    if (c.closedSides && (g.nx < 4 || (g.nz != 1 && g.nz < 4)))
        return std::unexpected(hipErrorInvalidValue);
    return DeviceConvWall(GpuGrid::from(g), c);
}

void DeviceConvWall::apply(DeviceCompState& s) const
{
    const int ox = sideGhostX(c_.closedSides);
    const int oz = sideGhostZ(g_, c_.closedSides);
    kConvWallY<<<launchBlocks(2 * std::size_t(g_.nx) * g_.nz), blockSize>>>(
        g_, Real(c_.gamma), Real(c_.Rgas), Real(c_.Tbot), Real(c_.Ttop),
        Real(c_.patch), ox, oz, s.rho.data(), s.mx.data(), s.my.data(),
        s.mz.data(), s.E.data());
    checkLastKernel("kConvWallY");
    if (ox == 0) return;
    kConvWallSide<<<launchBlocks(2 * std::size_t(g_.ny) * g_.nz), blockSize>>>(
        g_, 0, s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data());
    checkLastKernel("kConvWallSide x");
    if (oz == 0) return;
    kConvWallSide<<<launchBlocks(2 * std::size_t(g_.nx) * g_.ny), blockSize>>>(
        g_, 2, s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data());
    checkLastKernel("kConvWallSide z");
}

void addGravityY(const GpuGrid& g, double gy, Real cr, bool closed,
                 const DeviceCompState& s, DeviceCompState& out)
{
    kGravityY<<<launchBlocks(g.size()), blockSize>>>(
        g, sideGhostX(closed), sideGhostZ(g, closed), Real(gy), cr,
        s.rho.data(), s.my.data(), out.my.data(), out.E.data());
    checkLastKernel("kGravityY");
}

hipError_t convPlaneStats(const GpuGrid& g, bool closed, const DeviceScalar& rho,
                          const DeviceScalar& u, const DeviceScalar& v,
                          const DeviceScalar& w, const DeviceScalar& T,
                          DeviceBuffer<double>& scratch,
                          std::vector<double>& host)
{
    const std::size_t n = std::size_t(g.ny) * 6;
    if (auto e = internal::ensurePlaneScratch(scratch, n); e != hipSuccess)
        return e;
    kConvPlaneStats<<<launchBlocks(std::size_t(g.ny)), blockSize>>>(
        g, sideGhostX(closed), sideGhostZ(g, closed), rho.data(), u.data(),
        v.data(), w.data(), T.data(), scratch.data());
    checkLastKernel("kConvPlaneStats");
    return internal::fetchPlaneScratch(scratch, n, host);
}

} // namespace dns::gpu
