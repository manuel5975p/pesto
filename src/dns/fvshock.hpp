// The pressure sensor of FV --shock (docs/nozzle.md): raw stride 2 along mesh lines. A line
// is a cell and its neighbours two steps out through opposite faces; only a line whose 5
// cells exist counts, so a wall row reads no first difference (a blocked read as the centre
// fired 19 false cells on a smooth nozzle wall, 0 with whole lines). Host, serial at make.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvgrad.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace dns {

// Mesh lines of a FaceMesh. lineStart [nC + 1]: CSR into line; line [2 nL]: c-2, c+2 of each
// whole line through c; faceLine [2 nI]: P-, N+ of each internal face, -1 at a stop.
struct FvShockLines
{
    std::vector<int> lineStart, line, faceLine;
};

// The opposite of a CSR entry is the active entry of the cell with the most antiparallel
// outward normal, cos <= -0.5, ties to the lowest srcFace, else none. Bitwise the same lines
// under every CellOrder. Pre: geom = fvFaceGeom of fm. Post: a uniform periodic box gives
// one line per axis per cell; an axis prism has no radial line.
FvShockLines fvShockLines(const FaceMesh& fm, const FvFaceGeom& geom);

// Sensor bytes of p: shockByte of max over the cell's lines of shockJump (a NaN sticks);
// 0 on a cell with no whole line. Pre: p, b sized nC.
void fvShockSense(const FvShockLines& l, std::span<const double> p, std::span<std::uint8_t> b);

} // namespace dns
