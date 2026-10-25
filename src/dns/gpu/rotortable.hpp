// The azimuth table: a rotor's ghost-wall tables at every bin, built once and
// replayed by the time loop.
//
// Why a table exists at all. Classifying the ghost wall is a signed-distance
// query per cell in the body's band -- for a mesh rotor that is a BVH descent,
// order a second per azimuth -- while a compressible run takes thousands of
// steps per revolution. Rebuilding per step is three orders of magnitude too
// slow. But the geometry of a rigid body spinning at a constant rate is a pure
// function of azimuth and nothing else, so it is precomputed once at `bins`
// angles and replayed.
//
// It lives in HOST memory. A bin is ~1 MB of index and weight tables and there
// are hundreds of them; that does not belong in VRAM beside the fields.
// Playback copies one bin into the device buffers DeviceCompGhost already owns,
// which are sized for the largest bin, so a bin change is a copy and never an
// allocation.
//
// Quantization is the price: the body only moves when the bin index changes,
// which injects an impulse at bins * revolutions/s. `rotorNaturalBins` is the
// point past which extra bins buy nothing (the tip cannot cross a cell faster
// than that), and the validator warns on either side of it.
//
// GPU-only header: the tables are pure host data, but they are DeviceCompGhost's
// tables and their layout is that class's business.
#pragma once

#include "dns/gpu/comp_ghost.hpp"
#include "dns/rotor.hpp"

#include <expected>
#include <string>
#include <vector>

namespace dns::gpu {

// One classified body per azimuth bin, plus the capacity that covers them all.
struct RotorTable
{
    std::vector<CompGhostTables> bins;
    CompGhostCapacity capacity;
    // Largest distance from the shaft to any point of the wetted surface, over
    // every bin. MEASURED from the classified interface faces, so it is the
    // radius the solver actually sees -- on whatever mesh, for whatever body
    // kind, and without needing the body's own AABB. --rotor-mach is quoted on
    // the NOMINAL tip radius D/2; a body reaching further than that has a
    // faster tip than the flag says, and the run has to know.
    double tipRadius = 0.0;

    // Total host bytes, for the run to report before it commits to them.
    std::size_t bytes() const;
};

// The ghost wall's view of a rigid rotor's surface: rotorWallVelocity and
// rotorWallAcceleration at a world point. A property of the SHAFT, not of the
// azimuth -- omega x r at a world point is the same whatever angle the blade
// has reached -- so one closure serves every bin, the frozen-rotor (mrf)
// classification included.
GhostWallMotion rotorWallMotion(const Rotor& rot);

// Classify `base` at every bin of `rot`, with the wall kinematics that azimuth
// implies. `base` is the body at azimuth 0 IN ITS OWN FRAME -- rotorSdf turns
// it, so it must NOT already carry a rotation. `threads` <= 0 uses the hardware
// concurrency; the work is embarrassingly parallel because each bin classifies
// against its own sdf and shares nothing.
//
// `sdfFactory` exists because an Sdf is not required to be thread-safe: a
// MeshSdf's BVH query is const and reentrant, but a caller composing one out of
// mutable state is not, so each worker asks for its own. Return the SAME body
// every time -- the only thing that may differ is which object evaluates it.
//
// Pre: rot.active(); bbox null or 6 doubles, in the AZIMUTH-0 frame inflated to
// cover the swept volume (rotorSweptBox).
std::expected<RotorTable, std::string>
buildRotorTable(const Grid& g, const Rotor& rot,
                const std::function<GhostSdf()>& sdfFactory,
                const double* bbox, const AxisRemap3& remap, int threads = 0);

} // namespace dns::gpu
