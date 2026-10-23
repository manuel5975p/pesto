#include "dns/prop/spec.hpp"

namespace dns::prop {

std::span<const char* const> methodNames()
{
    static constexpr const char* kNames[] = {"bemt", "liftline"};
    return kNames;
}

std::span<const char* const> hubModelNames()
{
    static constexpr const char* kNames[] = {"image", "none"};
    return kNames;
}

std::span<const char* const> wakePitchNames()
{
    static constexpr const char* kNames[] = {"mean", "local"};
    return kNames;
}

} // namespace dns::prop
