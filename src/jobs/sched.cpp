#include "jobs/sched.hpp"

namespace jobs {

std::vector<std::int64_t> admit(std::span<const Pending> queue, Occupancy busy, bool gpuFree)
{
    std::vector<std::int64_t> out;
    if (busy.benchmark) return out;
    bool waitingCpu = false, waitingGpu = false;
    for (const Pending& p : queue) {
        if (p.benchmark) {
            // Exclusive, and nothing behind it may overtake it.
            if (!busy.any() && out.empty() && !waitingCpu && !waitingGpu && gpuFree)
                out.push_back(p.id);
            return out;
        }
        const bool gpu = p.resource == Resource::gpu;
        bool& occupied = gpu ? busy.gpu : busy.cpu;
        bool& waiting = gpu ? waitingGpu : waitingCpu;
        if (occupied || waiting || (gpu && !gpuFree)) {
            waiting = true;
            continue;
        }
        occupied = true;
        out.push_back(p.id);
    }
    return out;
}

} // namespace jobs
