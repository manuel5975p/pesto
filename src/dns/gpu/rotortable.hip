#include "dns/gpu/rotortable.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <format>
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

} // namespace dns::gpu
