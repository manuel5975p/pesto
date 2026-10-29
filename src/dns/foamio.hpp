// OpenFOAM field files (ASCII) for the FV path (docs/fv.md): written in
// OpenFOAM's own cell / face / patch-face order, so a foam tool (or
// tools/fv/foamdiff.py) reads and diffs them against a simpleFoam time
// directory directly. Host only.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace dns {

// One patch of a written field: "calculated" + values (n x nc), or a
// constraint type without values ("empty", "cyclic").
struct FoamPatchOut
{
    std::string name;
    std::string constraint;     // non-empty: written as `type <constraint>;`
    std::vector<double> values; // n x nc, component-interleaved, patch-local order
};

// Writes <dir>/<object> as a vol{Scalar,Vector}Field (nc 1 / 3) or a
// surfaceScalarField (surface = true: internal = the internal faces).
// internal: n x nc interleaved. dims: OpenFOAM dimension set text.
// precision: significant digits (controlDict writePrecision). Creates dir.
std::expected<void, std::string> writeFoamField(const std::filesystem::path& dir, const std::string& object,
                                                int nc, bool surface, const std::string& dims,
                                                const std::vector<double>& internal,
                                                const std::vector<FoamPatchOut>& patches, int precision);

// The internalField of an OpenFOAM ASCII field file (uniform expands to
// nExpect entries): n x nc interleaved. Error: unreadable, binary, malformed,
// or a length other than nExpect.
std::expected<std::vector<double>, std::string> readFoamInternalField(const std::filesystem::path& file, int nc,
                                                                      std::int64_t nExpect);

} // namespace dns
