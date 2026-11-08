// DNS_TEST_SHOCK=1: an existing ghost-wall test run with --shock installed
// (masterplan_shockflux gate 24): `dev` gets enableShock on the wall's solid cells.
// Unset: nothing, the test as written. The rig must be Cartesian with no MRF.
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>

namespace dns::test {

inline void shockIfAsked(gpu::DeviceComp& dev, const gpu::DeviceCompGhost& gh, const Grid& g, bool allowed = true)
{
    if (!allowed || !std::getenv("DNS_TEST_SHOCK")) return;
    std::vector<std::uint8_t> mask(g.size(), 0);
    for (int c : gh.solidCells()) mask[std::size_t(c)] = 1;
    // The mask outlives the solver: never freed, a test process.
    static auto* keep = new std::vector<gpu::DeviceBuffer<std::uint8_t>>;
    auto m = gpu::DeviceBuffer<std::uint8_t>::make(g.size());
    if (!m || m->upload(mask.data()) != hipSuccess || !dev.enableShock({.solid = m->data()})) {
        std::printf("shockIfAsked: device failure\n");
        std::exit(EXIT_FAILURE);
    }
    keep->push_back(std::move(*m));
}

} // namespace dns::test
