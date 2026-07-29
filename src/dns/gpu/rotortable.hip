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
             + t.uwall.size() * sizeof(Real) + t.deep.size() * sizeof(int)
             + t.fcell.size() * sizeof(int) + t.fdir.size() * sizeof(int)
             + t.fpos.size() * sizeof(Real) + t.solid.size() * sizeof(int);
    return n;
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
    const int nw =
        threads > 0 ? threads
                    : std::max(1u, std::thread::hardware_concurrency());

    // Bins are handed out atomically rather than sliced, because their cost is
    // not uniform: a blade edge-on to the grid touches far fewer cells than one
    // at 45 degrees, and a static slice would leave workers idle at the end.
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    pool.reserve(std::size_t(nw));
    for (int w = 0; w < nw; ++w)
        pool.emplace_back([&] {
            // One sdf object per worker: the classification is read-only, but
            // the sdf it reads need not be, so no worker shares one.
            const GhostSdf sdf = sdfFactory();
            for (int m = next++; m < rot.bins; m = next++)
            {
                const double theta = rotorBinAngle(rot, m);
                const GhostSdf turned = rotorSdf(rot, sdf, theta);
                // The wall velocity is a property of the SHAFT, not of the
                // azimuth: omega x r at a world point is the same whatever
                // angle the blade has reached, so this closure is bin-free.
                const GhostWallVelocity uw = [&rot](double x, double y,
                                                    double z) {
                    return rotorWallVelocity(rot, {x, y, z});
                };
                out.bins[std::size_t(m)] =
                    classifyCompGhost(g, turned, bbox, remap, &uw);
            }
        });
    for (auto& t : pool) t.join();

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
