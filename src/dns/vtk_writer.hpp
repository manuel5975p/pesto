// Legacy-VTK output of cell-centred velocity for ParaView inspection.
#pragma once

#include "dns/field.hpp"
#include "dns/grid.hpp"

#include <expected>
#include <string>

namespace dns {

// Writes STRUCTURED_POINTS with velocity interpolated to cell centres.
// Returns error text on I/O failure.
std::expected<void, std::string>
writeVtk(const Grid& g, const VectorField& U, const std::string& path);

} // namespace dns
