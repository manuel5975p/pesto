// cyclicAMI weights of the pimpleFoam path (docs/fvpimple.md, stage 3c):
// OpenFOAM v2606's cyclicAMIPolyPatch::resetAMI -> faceAreaWeightAMI::calculate
// in SERIAL, operand for operand, host, FP64, pure, single-threaded:
//  * patches as resetAMI builds them (patch0 / nbrPatch0: localFaces on the
//    patch's own points, `transform noOrdering`: no transformPosition);
//  * advancingFrontAMI::calculate: face::triangles (`triMode mesh`:
//    face::split SPLITTRIANGLE from the most concave angle) and srcMagSf_ /
//    tgtMagSf_ = sum of the triangles' area normals & the patch faceNormals
//    (`areaNormalisationMode project`);
//  * the walk starts on the pair (source 0, target 0) as
//    faceAreaWeightAMI::calculate passes it (initialiseWalk searches nothing);
//  * faceAreaWeightAMI::calcAddressing (processSourceFace: target faces LIFO
//    through faceFaces, a pair kept when area / srcMagSf > 1e-6; setNextFaces),
//    restartUncoveredSourceFace (coverage < 0.95: a seed per face vertex),
//    faceAreaIntersect::calc / overlaps (triangleIntersect, triSliceWithPlane,
//    tolerance 1e-6 sqrt(source triangle area));
//  * findTargetFace (a stalled front, the restart): the target indexedOctree
//    (bb inflated 0.01, maxLevel 8, leaf 10, duplicity 3) and its findNearest
//    with findNearestMaskedOp, restated node for node: a tie between target
//    faces is decided by the tree's traversal order;
//  * AMIInterpolation::normaliseWeights(conformal = requireMatch = true).
// Defaults only: requireMatch true, reverseTarget false, lowWeightCorrection
// off, maxDistance2 / minCosAngle off, no AMI cache, no surface projection.
// Compiled with -ffp-contract=off (OpenFOAM's build contracts nothing).
// Gate: tests/test_fvami.cpp, addressing and every value BIT FOR BIT against
// tools/fv/foamAmiDump (serial). libm enters through acos (which diagonal a
// quad is split on) and cos(89 deg): same glibc as OpenFOAM's = same bits.
// A face without overlap has no entry and weightsSum 0: no error, exactly 0
// interpolated (OpenFOAM's behaviour under requireMatch without
// lowWeightCorrection).
#pragma once

#include "dns/polymesh.hpp"

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns {

struct FoamAmiSide
{
    std::vector<int> start, addr; // CSR over this side's patch-local faces; addr = the OTHER side's
                                  // patch-local face, in srcAddress_ / tgtAddress_ order
    std::vector<double> w;        // srcWeights_ / tgtWeights_ after normaliseWeights
    std::vector<double> wSum;     // srcWeightsSum_ / tgtWeightsSum_: sum(areas) / magSf, 0 without overlap
    std::vector<double> magSf;    // srcMagSf_ / tgtMagSf_ (projected triangle areas, not |Sf|)
    bool operator==(const FoamAmiSide&) const = default;
};

struct FoamAmi
{
    FoamAmiSide src, tgt;
    std::vector<std::string> warnings; // OpenFOAM's WarningInFunction texts (bounding boxes, invalid normal)
    bool operator==(const FoamAmi&) const = default;
};

// The AMI of the owner patch srcPatch (the cyclicAMI with the lower patch
// index) against its neighbourPatch tgtPatch on the points px, py, pz.
// Pre: px, py, pz sized pm.man.nPoints, connectivity from pm; both indices
// name CyclicAMI patches. Errors: OpenFOAM's fatal conditions, its wording.
std::expected<FoamAmi, std::string> foamCyclicAmi(const PolyMesh& pm, std::span<const double> px,
                                                  std::span<const double> py, std::span<const double> pz,
                                                  int srcPatch, int tgtPatch);

// AMIInterpolation::interpolateToSource (s = ami.src, fld on the target
// faces) / interpolateToTarget (s = ami.tgt, fld on the source faces) with
// the default multiplyWeightedOp<plusEqOp>, no lowWeightCorrection:
// result[f] = Zero; for i in the face's slots, in order: result[f] += w[i] * fld[addr[i]].
// Pre: fld holds nComp values per face of the other side, face-major. A face
// without overlap gets exactly 0.
std::vector<double> foamAmiInterpolate(const FoamAmiSide& s, std::span<const double> fld, int nComp = 1);

// What `AMI: Patch source|target sum(weights) min:.. max:.. average:..` prints.
struct FoamAmiSum
{
    double min = 0, max = 0, average = 0; // gMinMax, gAverage (sequential sum / size) of wSum
    std::size_t nZero = 0;                // faces with no overlap (wSum == 0)
};
// Pre: s.wSum non-empty.
FoamAmiSum foamAmiSum(const FoamAmiSide& s);
// The log line itself at Info's precision: controlDict writePrecision when present (Time::readDict), else 6.
std::string foamAmiSumLine(const FoamAmiSide& s, const char* sourceOrTarget, int precision = 6);

} // namespace dns
