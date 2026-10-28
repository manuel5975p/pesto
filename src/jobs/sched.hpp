// The admission rule, pure so it is testable without processes.
//   - at most one gpu job and one cpu job run at a time (they may overlap);
//   - FIFO per resource: a waiting job blocks later jobs of its resource;
//   - a benchmark runs ALONE: it starts only when nothing runs and nothing
//     submitted before it still waits, and while it waits or runs nothing
//     submitted after it starts.
#pragma once

#include "jobs/spec.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace jobs {

struct Pending
{
    std::int64_t id = 0;
    Resource resource = Resource::cpu;
    bool benchmark = false;
};

struct Occupancy
{
    bool cpu = false, gpu = false, benchmark = false;
    bool any() const { return cpu || gpu || benchmark; }
};

// Ids from `queue` (submission order) that may start now. gpuFree is false
// while an external holder owns the GPU lock (gates gpu and benchmark jobs).
std::vector<std::int64_t> admit(std::span<const Pending> queue, Occupancy busy, bool gpuFree);

} // namespace jobs
