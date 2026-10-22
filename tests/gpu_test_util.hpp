// Shared helpers for GPU parity tests: skip handling, random fields, tolerances.
#pragma once

#include "dns/field.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/grid.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace dns::test {

inline constexpr int skipExitCode = dnstest::kSkipExitCode;

// Call first in every GPU test's main().
inline void requireDeviceOrSkip()
{
    if (!gpu::deviceAvailable()) dnstest::skip("no HIP device");
}

inline VectorField randomField(const Grid& g, unsigned seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto U = VectorField::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        U.u[c] = dist(rng);
        U.v[c] = dist(rng);
        U.w[c] = dist(rng);
    }
    return U;
}

// Max |a-b| over both fields, and the max |b| scale for relative bounds.
struct FieldDiff { double maxErr, scale; };

inline FieldDiff fieldDiff(const VectorField& a, const VectorField& b)
{
    FieldDiff d{0.0, 0.0};
    for (std::size_t c = 0; c < a.u.size(); ++c)
    {
        d.maxErr = std::max({d.maxErr, std::abs(a.u[c] - b.u[c]),
                             std::abs(a.v[c] - b.v[c]), std::abs(a.w[c] - b.w[c])});
        d.scale = std::max({d.scale, std::abs(b.u[c]), std::abs(b.v[c]),
                            std::abs(b.w[c])});
    }
    return d;
}

inline bool checkClose(const char* what, double err, double scale, double rtol)
{
    const bool ok = err <= rtol * scale;
    std::printf("%-18s maxErr %.3e scale %.3e rel %.3e %s\n", what, err, scale,
                scale > 0 ? err / scale : 0.0, ok ? "ok" : "FAIL");
    return ok;
}

} // namespace dns::test
