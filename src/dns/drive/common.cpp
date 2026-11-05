// The drivers' shared tails: die, the sink config, the GPU probe.
#include "dns/drive/internal.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/device_util.hpp"
#endif

namespace dns::drive {

int die(int code, const std::string& msg)
{
    std::fprintf(stderr, "dns: %s\n", msg.c_str());
    return code;
}

RunOutput::Config outputConfig(const Case& c, const std::string& simJson, bool suppress)
{
    RunOutput::Config oc;
    oc.prefix = c.outPrefix;
    oc.kinds = suppress ? std::set<Emit>{Emit::None} : c.emit;
    oc.fdt = c.fdt;
    oc.pdt = c.pdt;
    oc.wdt = c.wdt;
    oc.planesPerAxis = c.planes;
    oc.pngDown = c.pngDown;
    oc.frameStart = c.frameStart;
    oc.simJson = simJson;
    return oc;
}

bool haveGpu(const Case& c)
{
#if DNS_WITH_GPU
    if (c.backend == Backend::Cpu) return false;
    return gpu::deviceAvailable();
#else
    (void)c;
    return false;
#endif
}

std::vector<float> frameMask(std::size_t n, std::span<const std::uint8_t> map, std::uint8_t solid)
{
    std::vector<float> m(n, 1.0f);
    for (std::size_t q = 0; q < map.size() && q < n; ++q)
        if (map[q] == solid) m[q] = 0.0f;
    return m;
}

} // namespace dns::drive
