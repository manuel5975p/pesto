// Rigid rotor motion: a body spinning about a fixed shaft at a constant rate.
//
// This is the whole of "time-dependent geometry" in the solver, and it is pure
// host arithmetic: an azimuth in, an Sdf and a wall velocity out. Nothing here
// touches the GPU, the filesystem or a Case -- gpu/comp_ghost consumes the Sdf
// exactly as it consumes a static body's, and dns/rotortable replays the
// per-azimuth tables built from it.
//
// Two rules the rest of the code depends on:
//   - The azimuth is COMPUTED from the step's own time (rotorAzimuth), never
//     accumulated. An integrated angle drifts off the bin grid over a long run.
//   - The bin index is integer arithmetic mod bins, so a revolution closes
//     EXACTLY: bin 0 after a full turn is the same table, bit for bit.
#pragma once

#include "dns/sdf.hpp"

#include <array>

namespace dns {

// Shaft axis of a rotor: the axis it spins about, and the axis thrust acts
// along. Rotation is right-handed about it.
enum class RotorAxis { X, Y, Z };

// Index of the shaft axis (0/1/2), for the generic (a, b, c) cyclic arithmetic
// every function here uses instead of three hand-written per-axis branches.
inline int rotorAxisIndex(RotorAxis a)
{
    return a == RotorAxis::X ? 0 : a == RotorAxis::Y ? 1 : 2;
}

// Constant-rate rigid rotation about a fixed shaft.
//
// `bins` quantizes the azimuth: geometry is precomputed at `bins` evenly
// spaced angles and the run replays them, so the solver never rebuilds
// geometry in its time loop. `thicken` is a uniform outward offset of the
// body's surface in WORLD units -- exactly a Minkowski sum with a ball, see
// rotorSdf -- and is a DECLARED modification of the geometry, not a fix.
struct Rotor
{
    RotorAxis axis = RotorAxis::X;
    std::array<double, 3> shaft{}; // a point on the shaft, world units
    double omega = 0;              // signed rate, radians per unit time
    double radius = 0;             // tip radius; sets the swept extent
    int bins = 0;                  // azimuth bins; > 0 iff active
    double thicken = 0;            // sdf dilation, world units, >= 0

    bool active() const { return omega != 0 && bins > 0; }
};

// Azimuth at time t: exactly omega * t. Callers must pass the step's absolute
// time, not a delta -- that is what keeps a long run on the bin grid.
double rotorAzimuth(const Rotor& r, double t);

// Bin holding `theta`, always in [0, bins). Negative and multi-revolution
// angles normalize. Pre: r.bins > 0.
int rotorBin(const Rotor& r, double theta);

// Centre angle of bin m -- the angle its geometry is built at, so the
// quantization error is half a bin either side rather than a whole one.
// Pre: r.bins > 0 and 0 <= m < r.bins.
double rotorBinAngle(const Rotor& r, int m);

// Bins beyond which extra angular resolution buys nothing: occupancy can only
// change when something crosses a cell boundary and the tip is fastest, so
// 2*pi*radius / hFine bin boundaries already resolve every change there.
// Pre: hFine > 0, r.radius > 0. Post: >= 1.
int rotorNaturalBins(const Rotor& r, double hFine);

// `base` rotated by `theta` about the shaft and dilated by r.thicken.
//
// The rotation is applied to the QUERY POINT by the inverse angle, which is
// exact and costs nothing -- no mesh is transformed and no BVH rebuilt, so a
// rotor's per-azimuth geometry is the same MeshSdf read from a different
// direction. Dilation is `sdf - thicken`: for a true distance field that is
// exactly the Minkowski sum with a ball of that radius, so it stays a valid
// distance field (which sampleCut's Lipschitz prefilter and the ghost wall's
// normal both require) and it rounds convex edges rather than distorting them.
// Pre: base non-empty, r.thicken >= 0.
Sdf rotorSdf(const Rotor& r, const Sdf& base, double theta);

// Surface velocity omega x (p - shaft) at a world point: what the wall
// condition mirrors about. Zero on the shaft axis, |omega|*radius at the tip.
std::array<double, 3> rotorWallVelocity(const Rotor& r,
                                        const std::array<double, 3>& p);

// Surface acceleration omega x (omega x (p - shaft)) = -omega^2 times the
// in-plane offset from the shaft: the centripetal acceleration of the wall
// material point. The ghost wall's pressure image needs it (wall-normal
// momentum: grad p = -rho a at a no-slip wall). Constant rate, so no
// tangential part.
std::array<double, 3> rotorWallAcceleration(const Rotor& r,
                                            const std::array<double, 3>& p);

// Tip Mach -> rate. The tip travels at mach * cInf, so omega = mach*cInf/radius.
// Pre: radius > 0.
double rotorOmegaFromTipMach(double tipMach, double cInf, double radius);

// World box the body can occupy at ANY azimuth, FROM FLAGS ALONE: the
// shaft-aligned cylinder of radius `radius` and half-height `halfSpan` about
// the shaft point, inflated by `pad`. This is the validator's version -- it is
// why a rotor can place a focal plateau without opening the mesh file, which
// validate() may not do. Returns {xlo, ylo, zlo, xhi, yhi, zhi}.
// Pre: radius > 0, halfSpan >= 0, pad >= 0.
std::array<double, 6> rotorSweptBox(const Rotor& r, double halfSpan, double pad);

// The same box MEASURED from the body's own azimuth-0 bounds, once they are
// known: the shaft-axis extent is unchanged by the turn, and the in-plane
// extent becomes the largest distance from the shaft to any corner. Exact, and
// tighter than the flags-only form whenever the body is smaller than its
// nominal radius. Run time only -- the loader has already read the mesh.
// Pre: box is {lo xyz, hi xyz} with lo <= hi.
std::array<double, 6> rotorSweepBounds(const Rotor& r,
                                       const std::array<double, 6>& box);

} // namespace dns
