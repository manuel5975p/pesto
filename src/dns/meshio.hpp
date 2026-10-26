// The O-grid mesh and final-state export of `--emit mesh` (docs/viewer-
// format.md): byte encodings only, written by RunOutput. The OpenFOAM parity
// tools (tools/parity/dnsmesh.py) read them back to build an identical
// polyMesh. Host-side, little-endian.
#pragma once

#include "dns/ogrid.hpp"

#include <string>
#include <vector>

namespace dns {

// <prefix>_mesh.bin: "DNSMESH1", u32 version 1, u32 nx ny nz, u32 wrap[3]
// (AxisWrap: 0 Translate, 1 Cyclic, 2 Extend, 3 MirrorZ), u32 flags (bit 0
// ghost rows j = 0 and ny-1; bit 1 ghost columns i = 0 and nx-1; bit 2
// wedge; bit 3 revolved: a revolved mesh without bit 2 is the FULL
// revolution, wrap[2] Cyclic), u32 nTheta, f64 dphi (the wedge angle, or
// 2 pi / nz on the full revolution), f64 centre[3], f64 mirrorZ0,
// then f64 x[n], y[n], z[n] of the STORED nodes (Grid::idx order; the reader
// applies the wrap rules for node(n)), then f64 d1[nx*nz] (i fastest).
// Pre: d1 empty or nx*nz entries (empty writes zeros).
std::vector<char> encodeMesh(const OGrid& og, const double centre[3],
                             const std::vector<double>& d1);

// <prefix>_final.bin: "DNSFINL1", u32 version 1, u32 nx ny nz, u32 nfields,
// nfields 16-byte zero-padded names, then f64 field[nfields][n] in
// Grid::idx order. Pre: every field has g.size() entries, names < 16 chars.
std::vector<char> encodeFinal(const Grid& g,
                              const std::vector<std::string>& names,
                              const std::vector<std::vector<double>>& fields);

} // namespace dns
