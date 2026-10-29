// `dns flow --polymesh CASE`: the incompressible FV path's run loop (docs/fv.md).
// Acquire (phase 3): the output directory, the forces .dat files and the
// stats CSV are created BEFORE the first device allocation; Run: SIMPLE
// iterations (controlDict endTime, --max-iters, residualControl), the
// OpenFOAM-style log, forces every iteration, OpenFOAM time directories at
// writeInterval and at the end. GPU-only header.
#pragma once

#include "dns/fvprecision.hpp"
#include "dns/fvsimple.hpp"
#include "dns/gpu/fv_simple.hpp"
#include "dns/gpu/run_result.hpp"

#include <expected>
#include <string>

namespace dns::gpu {

struct FvRunSpec
{
    std::string outPrefix;  // PREFIX/ is the output case directory
    int maxIters = 0;       // > 0 overrides endTime / deltaT
    bool quiet = false;
    bool stats = true;      // PREFIX_stats.csv
    FvNumerics num = fvNumerics(FvPrecision::Fp32); // arithmetic and solve floors
    // EXPERIMENTAL coupled prototype (P0b): >= 0 = coupled outer iterations
    // after this many SIMPLE iterations; -1 = SIMPLE only (the default).
    int coupledAfter = -1;
    FvCoupledOpts coupled;
};

// Pre: L from loadFvCase. Errors: 3 setup (case/mesh binding), 4 output,
// 6 device resources, 7 numerical failure.
std::expected<RunStats, RunError> runFvSimple(const FvRunSpec& spec, FvLoaded& L);

} // namespace dns::gpu
