// Signed distance to a solid in world coordinates: negative inside, and
// approximately a true distance near the wall (|grad sdf| ~ 1 there).
// Host-side, build time only -- never called inside a stepping loop.
#pragma once

#include <functional>

namespace dns {

using Sdf = std::function<double(double x, double y, double z)>;

} // namespace dns
