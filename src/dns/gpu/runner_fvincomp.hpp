// The time loop of the incompressible FV solver (docs/fvincomp.md): `dns flow
// --solver fv`, `dns flow --polymesh DIR --bc FILE.toml`, `dns tgv --solver
// fv`. Stats CSV created before the first device allocation. GPU-only header.
#pragma once

#include "dns/fvincomp.hpp"
#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/polymesh.hpp"

#include <expected>
#include <string>
#include <vector>

namespace dns::gpu {

struct FvIncompRun
{
    std::string outPrefix;     // PREFIX_stats.csv; PREFIX/<t>/{U,p} when pm is given
    std::string meshName;      // the label's mesh text
    FvIncompControls ctl;
    double cfl = 0.8, tEnd = 0;
    int statsEvery = 20;       // steps between stats rows
    bool quiet = false, stats = true;
    // > 0: the `dns tgv` series t,K,dKdt,epsVisc on stdout every this much t.
    double seriesDt = 0;
    double seriesVolume = 1;   // K per this volume
    const PolyMesh* pm = nullptr; // non-null: fm from it, write PREFIX/<tEnd>/{U,p}
};

// Runs fm under b from U0 ([3 nC]). final: non-null receives the last state.
// Errors: 3 setup (mesh binding), 4 output, 6 device resources, 7 numerical
// failure (a non-finite state, a collapsed dt).
std::expected<RunStats, RunError> runFvIncomp(const FvIncompRun& r, const FaceMesh& fm, const FvIncompBcs& b,
                                              const std::vector<double>& U0, FvIncompState* final = nullptr);

} // namespace dns::gpu
