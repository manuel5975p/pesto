// Host side of the incompressible FV path (P5b, docs/fv.md): OpenFOAM v2606's
// geometric factors on a FaceMesh and an OpenFOAM case bound to its faces --
// everything the GPU SIMPLE solver (gpu/fv_simple) uploads. FP64, FaceMesh
// numbering, pure (no GPU, no filesystem), deterministic.
//
// Factors (OpenFOAM basicFvGeometryScheme, verbatim):
//  w      = |S.(Cf - C_o)| / (|S.(Cf - C_o)| + |S.(C_n - Cf)|)   internal
//  dcNO   = 1 / max(n.delta, 0.05 |delta|), delta = C_n - C_o       internal
//  k      = n - delta dcNO  (non-orthogonal correction vector)       internal
//  dcB    = 1 / |n (n.(Cf - C_o))|  (patch deltaCoeffs)              boundary
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvcase.hpp"
#include "dns/la/ell.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace dns {

// Device BC codes (one per boundary face).
enum class FvUCode : std::uint8_t { Fixed = 0, ZeroGrad = 1, InletOutlet = 2, Symmetry = 3, FreestreamV = 4 };
enum class FvPCode : std::uint8_t { Fixed = 0, ZeroGrad = 1, FreestreamP = 2 };
// Turbulence scalars (k, omega, nuTilda): 3 = omegaWallFunction (fixed value
// from the wall cell; the cell is fixed in the matrix).
enum class FvSCode : std::uint8_t { Fixed = 0, ZeroGrad = 1, InletOutlet = 2, OmegaWall = 3 };
enum class FvNutCode : std::uint8_t { Calculated = 0, LowRe = 1, Spalding = 2, KWall = 3, Cell = 4, Fixed = 5 };

// One wall-function patch's constants (omegaWallFunction / nut*WallFunction).
struct FvWallFnPatch
{
    double Cmu = 0.09, kappa = 0.41, E = 9.8, yPlusLam = 0, beta1 = 0.075;
    double n = 2, nutN = 4;              // blending exponents: omegaWallFunction, nutkWallFunction
    int blend = 0, nutBlend = 0;         // FvBlend: omegaWallFunction, nutkWallFunction
    int maxIter = 10;                    // nutUSpaldingWallFunction
};

// RAS part of the setup (FvSimpleSetup::ras), FaceMesh numbering.
struct FvRasSetup
{
    FvTurbModel model = FvTurbModel::Laminar;
    std::vector<double> y;           // [nC] the model's wall distance (wallDist)
    std::vector<double> yWall;       // [nB] nearWallDist on wall faces, 0 elsewhere
    std::vector<std::uint8_t> bWall; // [nB] wall-type patch (the model's y_b = SMALL there, y_P elsewhere)
    // transported scalars: SST k (0), omega (1); SA nuTilda (0)
    int nS = 0;
    std::array<std::vector<std::uint8_t>, 2> bS;  // FvSCode per boundary face
    std::array<std::vector<double>, 2> bRef, bS0; // ref value (fixed / inletValue), initial boundary value
    std::array<std::vector<double>, 2> s0;        // initial internal field [nC]
    std::vector<std::uint8_t> bNut;               // FvNutCode
    std::vector<double> nut0, bNut0;              // initial nut [nC], [nB]
    std::vector<FvWallFnPatch> wf;                // per case patch (omega / nut wall functions)
    std::vector<double> dlt;                      // [3 nI] C_n - C_o (FaceMesh d; limitedLinear's d)
};

struct FvSimpleSetup
{
    int nC = 0, nI = 0, nB = 0; // nB: NON-empty boundary faces only
    int emptyAxis = -1;         // 2D: the component OpenFOAM does not solve
    // internal faces (FaceMesh order)
    std::vector<int> own, nei;
    std::vector<double> S, Cf, k;   // [3 nI] component-major (Cf in the owner's frame)
    std::vector<double> rO, rN;     // [3 nI] Cf - C_own, Cf - C_nei (each in its cell's frame)
    std::vector<double> magS, w, dc; // [nI]
    // boundary faces: b in [0, nB) is FaceMesh face bFace[b]
    std::vector<int> bOwn, bFace, bPatch, bLocal; // bLocal: index within its OpenFOAM patch
    std::vector<double> bS, bN, bCf;    // [3 nB]
    std::vector<double> bMagS, bDc;     // [nB]
    std::vector<std::uint8_t> bU, bP;   // FvUCode / FvPCode
    std::vector<std::uint8_t> bAssign;  // U patch assignable() (constrainHbyA)
    // 1 on symmetry / symmetryPlane (polyMesh CONSTRAINT) patches: OpenFOAM
    // gives every derived "calculated" field the constraint type there, so H
    // and grad(U) take the reflected average 0.5 (x_P + R x_P), R = I - 2nn.
    std::vector<std::uint8_t> bSym;
    std::vector<std::uint8_t> bAdjust;  // adjustPhi: !fixesValue || inletOutlet
    std::vector<double> bRefU, bRefP;   // [3 nB] / [nB]: fixed value, inletValue, freestreamValue
    std::vector<double> bU0, bP0;       // initial boundary values (0/ files)
    // cells
    std::vector<double> V, C;           // [nC], [3 nC]
    std::vector<int> cellStart;         // nC + 1
    // CSR entries: internal face f seen from its owner 2f, from its neighbour
    // 2f + 1; boundary face b: -(b + 1). Ascending FaceMesh face order.
    std::vector<int> cellEnt;
    std::vector<int> cellEll;           // ELL position E of the off entry (internal), -1 boundary
    std::vector<double> U0, p0;         // initial internal fields [3 nC] / [nC]
    // MRF (MRFZoneList, the ACTIVE zones in MRFProperties order, at most
    // kMaxMrfZones, cellZones disjoint): zone[c] = 1 + the cell's zone (0:
    // none), its Omega = omega_z axis_z and origin. Internal faces: OpenFOAM's
    // makeRelative subtracts zone z's frame flux (Omega_z x (Cf - o_z)).S on
    // every face with a cell in z, zone by zone -- frame[f] is the lower
    // zone's (0 if none), frame2[f] the higher zone's on a face between two
    // zones (only when mrfTwo: phi - frame - frame2, OpenFOAM's double
    // subtraction). Boundary (the owner's zone): 0 none, 1 included (phi = 0,
    // U_b = Omega x (Cf - o)), 2 excluded / nonRotatingPatches (phi -= frame).
    static constexpr int kMaxMrfZones = 8;
    bool mrf = false, mrfTwo = false;
    std::vector<std::uint8_t> zone;     // [nC]
    std::vector<std::array<double, 3>> omega, origin; // per active zone
    std::vector<double> frame, frame2;  // [nI] ([nI] when mrfTwo)
    std::vector<std::uint8_t> bMrf;     // [nB]
    std::vector<double> bFrame, bUmrf;  // [nB], [3 nB]
    // pressure reference (needReference: no p patch fixes its value)
    bool needRef = false;
    int refCell = -1;
    double refValue = 0;
    // forces patches: per boundary face 1 when on a forces patch
    std::vector<std::uint8_t> bForce;
    // OpenFOAM order maps: FaceMesh cell -> OpenFOAM cell; per OpenFOAM patch
    // its boundary-face list in local order (-1: an empty patch face)
    std::vector<std::int64_t> srcCell;
    std::vector<std::vector<int>> patchFaces; // [patch][local] -> b
    std::vector<int> internalSrc;             // internal face -> OpenFOAM face index
    std::int64_t nFoamInternal = 0;
    FvRasSetup ras;                           // model Laminar: empty
};

// An OpenFOAM case read for the FV path: the polyMesh, its FaceMesh (Source
// order: OpenFOAM's own face and cell order, so per-cell sums run in
// OpenFOAM's order), the resolved case.
struct FvLoaded
{
    PolyMesh pm;
    FaceMesh fm;
    FoamCase c;
};

// Phase 1.5 (--dry-run): the polyMesh manifest and the resolved case, no bulk
// mesh data. Error (exit 3): a missing file, a rejected entry.
std::expected<FoamCase, std::string> loadFoamCaseManifest(const std::filesystem::path& caseDir);
// Phase 3: the whole case. Error (exit 3) as above plus every mesh rejection.
std::expected<FvLoaded, std::string> loadFvCase(const std::filesystem::path& caseDir);

// Binds case c to fm. pat: ellPattern(graphOf(fm)). Errors (exit 3) by name:
// more than kMaxMrfZones active MRF zones, two active zones on one cellZone,
// an MRF zone cell on a cyclic face, a pressure reference needed but pRefCell
// absent.
std::expected<FvSimpleSetup, std::string> buildSimpleSetup(const FaceMesh& fm, const FoamCase& c,
                                                           const la::CellGraph& g, const la::EllPattern& pat);

// RAS: the wall distances (fvwalldist) and the turbulence fields' BC codes,
// reference and initial values into s.ras. Pre: s = buildSimpleSetup(fm, c,
// ...), c.ras.model != Laminar, fm from pm in Source order. Error (exit 3): the
// wall distance's (no wall patch, meshWave with cyclics, unreached cells).
std::expected<void, std::string> buildRasSetup(const PolyMesh& pm, const FaceMesh& fm, const FoamCase& c,
                                               FvSimpleSetup& s);

} // namespace dns
