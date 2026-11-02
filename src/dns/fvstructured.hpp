// A structured `dns flow --solver simple` case as a generated OpenFOAM
// simpleFoam case (docs/fvstructured.md): mesh = FaceMesh::polyOf of the
// CurvMesh the flags name, dictionaries = texts that load through the SAME
// strict parseFoamCase as a hand-written case. No second solver, no second
// geometry. Host only; everything but writeFoamCase is free of the filesystem.
#pragma once

#include "dns/fvcase.hpp"
#include "dns/grid.hpp"
#include "dns/ogrid.hpp"
#include "dns/polymesh.hpp"
#include "dns/sdf.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>

namespace dns {

// The mesh family. Box: inlet x-, outlet x+, one lateral kind on y, z empty |
// cyclic. Channel: cyclic x, noSlip y, body force 1, delta 1. OGrid: wall
// `body`, characteristic far circle `far`, z empty | cyclic | revolved.
enum class FvStructMesh : std::uint8_t { Box, Channel, OGrid };
// Box y faces: free slip (symmetryPlane), noSlip wall, open (freestream*).
enum class FvStructLateral : std::uint8_t { Symmetry, Wall, Open };

// The resolved case; validate() fills it, every field in solver units.
struct FvStructured
{
    bool active = false;
    FvStructMesh mesh = FvStructMesh::Box;
    Grid grid{1, 1, 1, 1, 1, 1};
    double uInf = 1, nu = 0;
    FvStructLateral lateral = FvStructLateral::Symmetry;
    double stretchBeta = 0;              // Box: TanhStretchY; Channel: ChannelStretchY
    bool focal = false;
    std::array<double, 3> focalLo{}, focalHi{}, focalW{};
    double focalRatio = 1;
    double reTau = 0;                    // Channel: nu = 1/reTau, u_tau = 1
    OGridSpec ogrid{};
    bool revolved = false;
    RevolvedSpec rev{};
    double refArea = 0;                  // cd = Fx / (rho U^2 A / 2); 0: no body
    FvTurbModel model = FvTurbModel::Laminar;
    double nutRatio = 0, kInf = 0, mutRatio = 0; // RAS freestream: nu~/nu, k/U^2, mu_t/mu
    double omega = 0, mrfRadius = 0;     // MRF: z shaft through ogrid (cx, cy), zone r < mrfRadius
    int iters = 0;                       // SIMPLE iterations (endTime)
    std::optional<double> residual;      // residualControl, every solved field
};

// The label of a mesh family with its map ("O-grid wedge 5 deg", ...).
std::string fvStructuredLabel(const FvStructured& s);

// polyOf's manifest from the flags alone (cellZones included). Phase 1.5.
std::expected<PolyManifest, std::string> fvStructuredManifest(const FvStructured& s);

// The whole case: 0/, system/, constant/ (no mesh). Pure, deterministic.
FoamCaseTexts fvStructuredTexts(const FvStructured& s);

// parseFoamCase(fvStructuredTexts(s), manifest): what a run reads, for
// --dry-run. Errors are the strict reader's, exit 3.
std::expected<FoamCase, std::string> fvStructuredResolved(const FvStructured& s);

// The polyMesh. sdf: the body (O-grid; unused otherwise). Errors (exit 3):
// buildOGrid's, a mesh the layout rejects. Post: man == fvStructuredManifest(s).
std::expected<PolyMesh, std::string> fvStructuredMesh(const FvStructured& s, const Sdf& sdf);

// Writes dir/constant/polyMesh (binary) and every text. Post: loadFvCase(dir)
// reads exactly this case. Errors (exit 4) name the file.
std::expected<void, std::string> writeFoamCase(const std::filesystem::path& dir, const PolyMesh& pm,
                                               const FoamCaseTexts& t);

} // namespace dns
