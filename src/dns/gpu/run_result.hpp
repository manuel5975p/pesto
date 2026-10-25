// What every time-loop runner returns: its run statistics, or an error that
// carries the process exit code.
//
// GPU-only header.
#pragma once

#include <string>

namespace dns::gpu {

struct RunStats
{
    long steps = 0;
    double msPerStep = 0;
    double cd = 0, cl = 0, cs = 0, cdf = 0;   // last measured
    double resid = 0;         // last steady residual, uInf^2/D units
    double tStop = 0;         // time actually reached
    bool converged = false;   // steady mode only: the stop rule was met
};

// Errors carry the process exit code the masterplan assigns (6 resource,
// 7 numerical, 8 solver).
struct RunError
{
    int code;
    std::string message;
};

} // namespace dns::gpu
