#include "dns/gpu/rotortable.hpp"

#include "dns/gpu/device_util.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <format>
#include <iterator>
#include <thread>

namespace dns::gpu {

std::size_t RotorTable::bytes() const
{
    std::size_t n = 0;
    for (const auto& t : bins)
        n += t.gcell.size() * sizeof(int) + t.gnbr.size() * sizeof(int)
             + t.gw.size() * sizeof(Real) + t.gscale.size() * sizeof(Real)
             + t.tnbr.size() * sizeof(int) + t.tw.size() * sizeof(Real)
             + t.gfnb.size() * sizeof(int)
             + t.uwall.size() * sizeof(Real) + t.aspan.size() * sizeof(Real)
             + t.deep.size() * sizeof(int)
             + t.fcell.size() * sizeof(int) + t.fdir.size() * sizeof(int)
             + (t.fpos.size() + t.fuwall.size()) * sizeof(Real)
             + t.solid.size() * sizeof(int)
             + (t.sideFace.size() + t.sideNbr.size() + t.sideTnbr.size()
                + t.sideCell.size() + t.sideStart.size()) * sizeof(int)
             + (t.sideW.size() + t.sideTw.size() + t.sideScale.size()
                + t.sideUwall.size() + t.sideAspan.size()) * sizeof(Real);
    return n;
}

GhostWallMotion rotorWallMotion(const Rotor& rot)
{
    return [rot](double x, double y, double z) {
        return GhostWallKinematics{rotorWallVelocity(rot, {x, y, z}),
                                   rotorWallAcceleration(rot, {x, y, z})};
    };
}

std::expected<RotorTable, std::string>
buildRotorTable(const Grid& g, const Rotor& rot,
                const std::function<GhostSdf()>& sdfFactory,
                const double* bbox, const AxisRemap3& remap, int threads)
{
    if (!rot.active()) return std::unexpected("buildRotorTable: rotor inactive");
    if (!sdfFactory) return std::unexpected("buildRotorTable: no sdf factory");

    RotorTable out;
    out.bins.resize(std::size_t(rot.bins));
    // No more workers than bins: an idle worker still builds its own sdf.
    const int hw = int(std::max(1u, std::thread::hardware_concurrency()));
    const int nw = std::min(rot.bins, threads > 0 ? threads : hw);

    // Bins are handed out atomically rather than sliced, because their cost is
    // not uniform: a blade edge-on to the grid touches far fewer cells than one
    // at 45 degrees, and a static slice would leave workers idle at the end.
    std::atomic<int> next{0};
    auto work = [&] {
        // One sdf object per worker: the classification is read-only, but
        // the sdf it reads need not be, so no worker shares one.
        const GhostSdf sdf = sdfFactory();
        for (int m = next++; m < rot.bins; m = next++)
        {
            const double theta = rotorBinAngle(rot, m);
            const GhostSdf turned = rotorSdf(rot, sdf, theta);
            const GhostWallMotion motion = rotorWallMotion(rot);
            out.bins[std::size_t(m)] =
                classifyCompGhost(g, turned, bbox, remap, &motion);
        }
    };
    {   // jthread joins at scope exit, before `out` is read below.
        std::vector<std::jthread> pool;
        pool.reserve(std::size_t(nw));
        for (int w = 0; w < nw; ++w) pool.emplace_back(work);
    }

    const int s = rotorAxisIndex(rot.axis);
    const int pb = (s + 1) % 3, pc = (s + 2) % 3;
    for (const auto& t : out.bins)
    {
        out.capacity.cover(t);
        for (std::size_t f = 0; f + 2 < t.fpos.size(); f += 3)
        {
            const double db = double(t.fpos[f + pb]) - rot.shaft[pb];
            const double dc = double(t.fpos[f + pc]) - rot.shaft[pc];
            out.tipRadius = std::max(out.tipRadius, std::hypot(db, dc));
        }
    }
    if (out.capacity.nGhost == 0)
        return std::unexpected(std::format(
            "buildRotorTable: the body has no mirrored ghost cells at any of "
            "{} azimuths -- it is not resolved by this grid at all", rot.bins));
    return out;
}

namespace {

// val[4 q + {0..3}] = (rho, mx, my, mz) of cell idx[q].
__global__ void kGatherMom(const int* __restrict__ idx, int n,
                           const Real* __restrict__ rho,
                           const Real* __restrict__ mx,
                           const Real* __restrict__ my,
                           const Real* __restrict__ mz, Real* __restrict__ val)
{
    const int q = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= n) return;
    const int c = idx[q];
    val[4 * q + 0] = rho[c];
    val[4 * q + 1] = mx[c];
    val[4 * q + 2] = my[c];
    val[4 * q + 3] = mz[c];
}

// Cell idx[q] holds a ghost image about wall velocity uw[3q..]: reflect its
// velocity about uw (m' = 2 rho uw - m), internal energy kept.
__global__ void kReflectAboutWall(const int* __restrict__ idx, int n,
                                  const Real* __restrict__ uw, Real* rho,
                                  Real* mx, Real* my, Real* mz, Real* E,
                                  Real* lo1, Real* lo2, Real* lo3, Real* lo4)
{
    const int q = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= n) return;
    const int c = idx[q];
    const Real r = rho[c];
    const Real m0[3] = {mx[c], my[c], mz[c]};
    Real m1[3];
    for (int d = 0; d < 3; ++d) m1[d] = Real(2) * r * uw[3 * q + d] - m0[d];
    const Real k0 = Real(0.5) * (m0[0] * m0[0] + m0[1] * m0[1] + m0[2] * m0[2]) / r;
    const Real k1 = Real(0.5) * (m1[0] * m1[0] + m1[1] * m1[1] + m1[2] * m1[2]) / r;
    mx[c] = m1[0];
    my[c] = m1[1];
    mz[c] = m1[2];
    E[c] = E[c] - k0 + k1;
    if (lo1) {
        lo1[c] = Real(0);
        lo2[c] = Real(0);
        lo3[c] = Real(0);
        lo4[c] = Real(0);
    }
}

} // namespace

std::expected<int, hipError_t>
SwapLedgerScratch::reseedCreated(DeviceCompState& s, const CompGhostTables& from,
                                 const CompGhostTables& to)
{
    assert(!from.uwall.empty());
    std::vector<int> created;
    std::ranges::set_difference(from.solid, to.solid, std::back_inserter(created));
    if (created.empty()) return 0;
    // Old ghost index of each created cell (gcell is in cell order when
    // classifyCompGhost built it; anything else is looked up through a sort).
    std::vector<std::pair<int, int>> byCell;
    const bool sorted = std::ranges::is_sorted(from.gcell);
    if (!sorted) {
        byCell.reserve(from.gcell.size());
        for (int q = 0; q < from.nGhost(); ++q) byCell.emplace_back(from.gcell[std::size_t(q)], q);
        std::ranges::sort(byCell);
    }
    std::vector<int> cells;
    std::vector<Real> uw;
    for (int c : created) {
        int q = -1;
        if (sorted) {
            auto it = std::ranges::lower_bound(from.gcell, c);
            if (it != from.gcell.end() && *it == c) q = int(it - from.gcell.begin());
        } else {
            auto it = std::ranges::lower_bound(byCell, std::pair{c, -1});
            if (it != byCell.end() && it->first == c) q = it->second;
        }
        if (q < 0) continue; // a deep cell: no image to reflect
        cells.push_back(c);
        for (int d = 0; d < 3; ++d) uw.push_back(from.uwall[3 * std::size_t(q) + d]);
    }
    if (cells.empty()) return 0;
    if (idx_.size() < cells.size()) {
        auto a = DeviceBuffer<int>::make(2 * cells.size());
        auto b = DeviceBuffer<Real>::make(8 * cells.size());
        if (!a) return std::unexpected(a.error());
        if (!b) return std::unexpected(b.error());
        idx_ = std::move(*a);
        val_ = std::move(*b);
    }
    if (uw_.size() < uw.size()) {
        auto u = DeviceBuffer<Real>::make(2 * uw.size());
        if (!u) return std::unexpected(u.error());
        uw_ = std::move(*u);
    }
    if (auto e = hipMemcpy(idx_.data(), cells.data(), cells.size() * sizeof(int),
                           hipMemcpyHostToDevice);
        e != hipSuccess)
        return std::unexpected(e);
    if (auto e = hipMemcpy(uw_.data(), uw.data(), uw.size() * sizeof(Real),
                           hipMemcpyHostToDevice);
        e != hipSuccess)
        return std::unexpected(e);
    const bool comp = s.compensated();
    kReflectAboutWall<<<launchBlocks(cells.size()), blockSize>>>(
        idx_.data(), int(cells.size()), uw_.data(), s.rho.data(), s.mx.data(),
        s.my.data(), s.mz.data(), s.E.data(), comp ? s.lo[1].data() : nullptr,
        comp ? s.lo[2].data() : nullptr, comp ? s.lo[3].data() : nullptr,
        comp ? s.lo[4].data() : nullptr);
    if (auto e = hipGetLastError(); e != hipSuccess) return std::unexpected(e);
    return int(cells.size());
}

std::expected<SwapLedger, hipError_t>
SwapLedgerScratch::measure(const Grid& g, const AxisRemap3& remap,
                           std::span<const double> vol,
                           const DeviceCompState& s, const CompGhostTables& from,
                           const CompGhostTables& to,
                           const std::array<double, 3>& pivot)
{
    assert(std::ranges::is_sorted(from.solid) && std::ranges::is_sorted(to.solid));
    assert(vol.empty() || vol.size() == g.size());
    // created first, then deleted: one gather, the sign from the position.
    std::vector<int> cells;
    std::ranges::set_difference(from.solid, to.solid, std::back_inserter(cells));
    const std::size_t nCreated = cells.size();
    std::ranges::set_difference(to.solid, from.solid, std::back_inserter(cells));
    SwapLedger L;
    L.created = int(nCreated);
    L.deleted = int(cells.size() - nCreated);
    if (cells.empty()) return L;
    if (idx_.size() < cells.size()) {
        auto a = DeviceBuffer<int>::make(2 * cells.size());
        auto b = DeviceBuffer<Real>::make(8 * cells.size());
        if (!a) return std::unexpected(a.error());
        if (!b) return std::unexpected(b.error());
        idx_ = std::move(*a);
        val_ = std::move(*b);
    }
    const int n = int(cells.size());
    if (auto e = hipMemcpy(idx_.data(), cells.data(), cells.size() * sizeof(int),
                           hipMemcpyHostToDevice);
        e != hipSuccess)
        return std::unexpected(e);
    kGatherMom<<<launchBlocks(cells.size()), blockSize>>>(
        idx_.data(), n, s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
        val_.data());
    std::vector<Real> v(4 * cells.size());
    if (auto e = hipMemcpy(v.data(), val_.data(), v.size() * sizeof(Real),
                           hipMemcpyDeviceToHost);
        e != hipSuccess)
        return std::unexpected(e);
    const double hv = g.cellVolume();
    for (std::size_t q = 0; q < cells.size(); ++q) {
        const std::size_t c = std::size_t(cells[q]);
        const double V = (vol.empty() ? hv : vol[c]) * (q < nCreated ? 1.0 : -1.0);
        const int i = int(c % std::size_t(g.nx));
        const int j = int((c / std::size_t(g.nx)) % std::size_t(g.ny));
        const int k = int(c / (std::size_t(g.nx) * std::size_t(g.ny)));
        const double X[3] = {(i + 0.5) * g.hx, (j + 0.5) * g.hy, (k + 0.5) * g.hz};
        double x[3];
        remap.map(X, x);
        const double r[3] = {x[0] - pivot[0], x[1] - pivot[1], x[2] - pivot[2]};
        const double m[3] = {double(v[4 * q + 1]) * V, double(v[4 * q + 2]) * V,
                             double(v[4 * q + 3]) * V};
        L.mass += double(v[4 * q]) * V;
        for (int d = 0; d < 3; ++d) L.mom[d] += m[d];
        L.moment[0] += r[1] * m[2] - r[2] * m[1];
        L.moment[1] += r[2] * m[0] - r[0] * m[2];
        L.moment[2] += r[0] * m[1] - r[1] * m[0];
    }
    return L;
}

} // namespace dns::gpu
