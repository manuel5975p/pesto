#include "dns/gpu/comp_conv.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"

namespace dns::gpu {

namespace {

// Ghost rows {0, ny-1} mirror rows {1, ny-2}; writes and reads are disjoint.
// The thermodynamic ghost state is built so the wall face holds T = Twall and
// dp/dn = 0 with the mirrored velocity: T0 = 2 Twall - T1 (floored at
// Twall/20 so a transient cannot construct a negative-temperature ghost --
// the same guard, and the same 0.05 factor, as the channel wall), p0 = p1,
// rho0 = p0/(R T0), m0 = -rho0 u1, E0 from (p0, rho0, |u1|).
//
// The two walls carry their own temperature: Tbot on j = 0, Ttop on j = ny-1,
// and the floor's is Tbot only inside the hot patch.

// Centred hot patch on the floor; see the Boussinesq twin in
// gpu/boussinesq.hip. Fractional in the FLUID index range, and vacuous at
// nz == 1.
__host__ __device__ inline bool inHotPatchComp(const GpuGrid& g, int i, int k,
                                               Real patch, int ox, int oz)
{
    const Real fx = (Real(i - ox) + Real(0.5)) / Real(g.nx - 2 * ox)
                  - Real(0.5);
    if (fabsf(fx) > Real(0.5) * patch) return false;
    if (g.nz == 1) return true;
    const Real fz = (Real(k - oz) + Real(0.5)) / Real(g.nz - 2 * oz)
                  - Real(0.5);
    return fabsf(fz) <= Real(0.5) * patch;
}

__global__ void kConvWallY(GpuGrid g, Real gam, Real Rgas, Real Tbot,
                           Real Ttop, Real patch, int ox, int oz, Real* rho,
                           Real* mx, Real* my, Real* mz, Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxz = std::size_t(g.nx) * g.nz;
    if (t >= 2 * nxz) return;
    const bool hi = t >= nxz;
    const int i = int(t % g.nx), k = int((t / g.nx) % g.nz);
    const std::size_t c = gidx(g, i, hi ? g.ny - 1 : 0, k);
    const std::size_t s = gidx(g, i, hi ? g.ny - 2 : 1, k);
    const Real Twall =
        hi ? Ttop : (inHotPatchComp(g, i, k, patch, ox, oz) ? Tbot : Ttop);

    const Real r1 = rho[s];
    const Real u1 = mx[s] / r1, v1 = my[s] / r1, w1 = mz[s] / r1;
    const Real ke1 = Real(0.5) * (u1 * u1 + v1 * v1 + w1 * w1);
    const Real p1 = (gam - Real(1)) * (E[s] - r1 * ke1);
    const Real T1 = p1 / (Rgas * r1);
    const Real T0 = fmaxf(Real(2) * Twall - T1, Real(0.05) * Twall);
    const Real r0 = p1 / (Rgas * T0);
    rho[c] = r0;
    mx[c] = -r0 * u1;
    my[c] = -r0 * v1;
    mz[c] = -r0 * w1;
    E[c] = p1 / (gam - Real(1)) + r0 * ke1;
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
// is identical and only the pair of indices moves. Runs over every index of the
// other two axes, so a corner ghost is this axis's mirror of a cell an earlier
// pass already mirrored: the diagonal reflection.
__global__ void kConvWallSide(GpuGrid g, int axis, Real* rho, Real* mx, Real* my,
                              Real* mz, Real* E)
{
    const int nA = axis == 0 ? g.nx : g.nz;
    const int nB = axis == 0 ? g.ny : g.nx;   // the two swept axes, B then C
    const int nC = axis == 0 ? g.nz : g.ny;
    const std::size_t face = std::size_t(nB) * nC;
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= 2 * face) return;
    const bool hi = t >= face;
    const int b = int(t % nB), cc = int((t / nB) % nC);
    const int a0 = hi ? nA - 1 : 0, a1 = hi ? nA - 2 : 1;
    // axis 0: (b, cc) = (j, k) beside the walled i. axis 2: (b, cc) = (i, j).
    const std::size_t c = axis == 0 ? gidx(g, a0, b, cc) : gidx(g, b, cc, a0);
    const std::size_t s = axis == 0 ? gidx(g, a1, b, cc) : gidx(g, b, cc, a1);

    rho[c] = rho[s];
    mx[c] = -mx[s];
    my[c] = -my[s];
    mz[c] = -mz[s];
    E[c] = E[s];
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
    if (j < 1 || j > g.ny - 2) return;
    if (i < ox || i >= g.nx - ox) return;
    if (k < oz || k >= g.nz - oz) return;
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

double convPatchAreaFraction(const GpuGrid& g, double patch, bool closed)
{
    const int ox = sideGhostX(closed), oz = sideGhostZ(g, closed);
    std::size_t inside = 0;
    for (int k = oz; k < g.nz - oz; ++k)
        for (int i = ox; i < g.nx - ox; ++i)
            if (inHotPatchComp(g, i, k, Real(patch), ox, oz)) ++inside;
    return double(inside) / (double(g.nx - 2 * ox) * (g.nz - 2 * oz));
}

hipError_t convPlaneStats(const GpuGrid& g, bool closed, const DeviceScalar& rho,
                          const DeviceScalar& u, const DeviceScalar& v,
                          const DeviceScalar& w, const DeviceScalar& T,
                          DeviceBuffer<double>& scratch,
                          std::vector<double>& host)
{
    const std::size_t n = std::size_t(g.ny) * 6;
    if (scratch.size() < n)
    {
        auto b = DeviceBuffer<double>::make(n);
        if (!b) return b.error();
        scratch = std::move(*b);
    }
    kConvPlaneStats<<<launchBlocks(std::size_t(g.ny)), blockSize>>>(
        g, sideGhostX(closed), sideGhostZ(g, closed), rho.data(), u.data(),
        v.data(), w.data(), T.data(), scratch.data());
    checkLastKernel("kConvPlaneStats");
    host.resize(n);
    return hipMemcpy(host.data(), scratch.data(), n * sizeof(double),
                     hipMemcpyDeviceToHost);
}

} // namespace dns::gpu
