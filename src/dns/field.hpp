// Flat scalar and staggered vector fields on a Grid; trivially GPU-mappable storage.
#pragma once

#include "dns/grid.hpp"

#include <vector>

namespace dns {

using Scalar = std::vector<double>; // one value per grid.idx()

struct VectorField
{
    Scalar u, v, w; // MAC face components, see Grid staggering

    static VectorField zeros(const Grid& g)
    {
        return {Scalar(g.size(), 0.0), Scalar(g.size(), 0.0), Scalar(g.size(), 0.0)};
    }
};

inline Scalar zeros(const Grid& g) { return Scalar(g.size(), 0.0); }

} // namespace dns
