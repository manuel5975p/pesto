// The O-grid mesh and final-state export of `--emit mesh` (docs/viewer-
// format.md): byte encodings only, written by RunOutput. The OpenFOAM parity
// tools (tools/parity/dnsmesh.py) read them back to build an identical
// polyMesh. Host-side, little-endian.
#pragma once

#include "dns/ogrid.hpp"

#include <expected>
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

// A full revolution's initial state from a converged WEDGE run (--init-from):
// primitives per cell of the target grid and the specific RANS scalars.
struct WedgeStart
{
    std::vector<double> rho, u, v, w, p;
    std::vector<std::vector<double>> phi; // scalars, in `scalars` order
};

// Revolve the wedge's real plane (k = 1, centred on phi = 0) over the target
// full revolution: cell (i, j, k) takes (i, j, 1) with (v, w) rotated by
// R_x(2 pi (k + 1/2) / N) -- the cell's centroid azimuth -- and rho, u, p and
// the scalars copied; ghost rows/columns too (the hooks overwrite them).
// mesh / fin: the wedge's _mesh.bin / _final.bin bytes. Errors, by name: not
// a revolved-wedge file, another (nx, ny, nTheta), meridional nodes that
// differ from og's beyond 1e-9 of the domain size (another --ogrid-beta,
// --ogrid-far or body), a closure-scalar set that is not `scalars`, a
// far-row mean pressure more than 2% off pInf (another --mach), non-finite
// or non-positive rho / p. Pre: og is a full revolution (og.rev.wedge false).
std::expected<WedgeStart, std::string>
revolveWedgeState(const std::vector<char>& mesh, const std::vector<char>& fin,
                  const OGrid& og, const double centre[3],
                  const std::vector<std::string>& scalars, double pInf);

} // namespace dns
