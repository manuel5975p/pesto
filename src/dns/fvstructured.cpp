#include "dns/fvstructured.hpp"

#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/remap.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <ranges>
#include <string_view>
#include <system_error>
#include <vector>

namespace dns {
namespace fs = std::filesystem;
using std::string;

namespace {

// What a patch IS to the dictionaries.
enum class Role : std::uint8_t { Inlet, Outlet, Wall, Far, SymmetryPlane, Symmetry, Empty, Cyclic };

struct Patch
{
    string name;
    Role role;
};

// The polyOf spec of this case; pure.
StructuredSpec specOf(const FvStructured& s)
{
    const Grid& g = s.grid;
    const FoamPatchType zType = g.nz == 1 ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    StructuredSpec sp;
    sp.lo = {0, 0, 0};
    sp.hi = {g.nx - 1, g.ny - 1, g.nz - 1};
    switch (s.mesh) {
    case FvStructMesh::Box: {
        const FoamPatchType lat = s.lateral == FvStructLateral::Symmetry ? FoamPatchType::SymmetryPlane
                                  : s.lateral == FvStructLateral::Wall   ? FoamPatchType::Wall
                                                                         : FoamPatchType::Patch;
        sp.name = {"inlet", "outlet", "lower", "upper", "front", "back"};
        sp.type = {FoamPatchType::Patch, FoamPatchType::Patch, lat, lat, zType, zType};
        break;
    }
    case FvStructMesh::Channel: {
        if (!s.focal) { // rows 0, ny - 1 are channelY's ghost rows
            sp.lo[1] = 1;
            sp.hi[1] = g.ny - 2;
        }
        sp.name = {"xlo", "xhi", "lower", "upper", "front", "back"};
        sp.type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Wall, FoamPatchType::Wall, zType, zType};
        break;
    }
    case FvStructMesh::OGrid: {
        sp.lo[1] = 1;
        sp.hi[1] = g.ny - 2;
        sp.name = {"ilo", "ihi", "body", "far", "front", "back"};
        sp.type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Wall, FoamPatchType::Patch, zType, zType};
        if (s.revolved) {
            sp.lo[0] = 1;
            sp.hi[0] = g.nx - 2;
            sp.type[0] = sp.type[1] = FoamPatchType::Patch; // the axis: no patch
            if (s.rev.wedge) {
                sp.lo[2] = sp.hi[2] = 1;
                sp.type[4] = sp.type[5] = FoamPatchType::Symmetry;
            } else {
                sp.type[4] = sp.type[5] = FoamPatchType::Cyclic;
            }
        }
        break;
    }
    }
    return sp;
}

// Node continuation per axis, as the builders set it.
std::array<AxisWrap, 3> wrapOf(const FvStructured& s)
{
    if (s.mesh != FvStructMesh::OGrid)
        return {AxisWrap::Translate, AxisWrap::Translate, AxisWrap::Translate};
    if (!s.revolved) return {AxisWrap::Cyclic, AxisWrap::Extend, AxisWrap::Translate};
    return {AxisWrap::Extend, AxisWrap::Extend, s.rev.wedge ? AxisWrap::MirrorZ : AxisWrap::Cyclic};
}

std::array<int, 2> axisOf(const FvStructured& s)
{
    return s.mesh == FvStructMesh::OGrid && s.revolved ? std::array<int, 2>{0, s.grid.nx - 2} : std::array<int, 2>{-1, -1};
}

// The case's patches in polyMesh order, as the dictionaries see them.
std::vector<Patch> patchesOf(const FvStructured& s)
{
    const Role zRole = s.grid.nz == 1 ? Role::Empty : Role::Cyclic;
    switch (s.mesh) {
    case FvStructMesh::Box: {
        const Role lat = s.lateral == FvStructLateral::Symmetry ? Role::SymmetryPlane
                         : s.lateral == FvStructLateral::Wall   ? Role::Wall
                                                                : Role::Far;
        return {{"inlet", Role::Inlet}, {"outlet", Role::Outlet}, {"lower", lat},
                {"upper", lat},         {"front", zRole},         {"back", zRole}};
    }
    case FvStructMesh::Channel:
        return {{"xlo", Role::Cyclic}, {"xhi", Role::Cyclic}, {"lower", Role::Wall},
                {"upper", Role::Wall}, {"front", zRole},      {"back", zRole}};
    case FvStructMesh::OGrid:
        if (s.revolved && !s.rev.wedge) return {{"body", Role::Wall}, {"far", Role::Far}};
        if (s.revolved) return {{"body", Role::Wall}, {"far", Role::Far}, {"front", Role::Symmetry}, {"back", Role::Symmetry}};
        return {{"body", Role::Wall}, {"far", Role::Far}, {"front", zRole}, {"back", zRole}};
    }
    return {};
}

bool hasCyclicPatch(const FvStructured& s)
{
    return std::ranges::any_of(patchesOf(s), [](const Patch& p) { return p.role == Role::Cyclic; });
}

string header(const char* cls, const char* location, const char* object)
{
    return std::format("FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {};\n"
                       "    location    \"{}\";\n    object      {};\n}}\n\n",
                       cls, location, object);
}

string vec(double x, double y, double z) { return std::format("({} {} {})", x, y, z); }

// The constraint entries every field shares; empty string: not a constraint.
string constraintBc(const Patch& p)
{
    switch (p.role) {
    case Role::SymmetryPlane: return "type symmetryPlane;";
    case Role::Symmetry: return "type symmetry;";
    case Role::Empty: return "type empty;";
    case Role::Cyclic: return "type cyclic;";
    default: return {};
    }
}

template <class Bc>
string boundaryField(const FvStructured& s, Bc&& bc)
{
    string o = "boundaryField\n{\n";
    for (const Patch& p : patchesOf(s)) {
        const string c = constraintBc(p);
        o += std::format("    {}\n    {{\n        {}\n    }}\n", p.name, c.empty() ? bc(p) : c);
    }
    return o + "}\n";
}

string fieldU(const FvStructured& s, double uStart)
{
    const string U = vec(s.uInf, 0, 0);
    string o = header("volVectorField", "0", "U");
    o += std::format("dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   uniform {};\n\n", vec(uStart, 0, 0));
    o += boundaryField(s, [&](const Patch& p) -> string {
        switch (p.role) {
        case Role::Inlet: return std::format("type fixedValue; value uniform {};", U);
        case Role::Outlet: return std::format("type inletOutlet; inletValue uniform {}; value uniform {};", vec(0, 0, 0), U);
        case Role::Wall: return "type noSlip;";
        default: return std::format("type freestreamVelocity; freestreamValue uniform {};", U);
        }
    });
    return o;
}

string fieldP(const FvStructured& s)
{
    string o = header("volScalarField", "0", "p");
    o += "dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform 0;\n\n";
    o += boundaryField(s, [](const Patch& p) -> string {
        switch (p.role) {
        case Role::Inlet:
        case Role::Wall: return "type zeroGradient;";
        case Role::Outlet: return "type fixedValue; value uniform 0;";
        default: return "type freestreamPressure; freestreamValue uniform 0;";
        }
    });
    return o;
}

// A transported scalar with freestream value inf: inflow, outflow and far by
// mixed characteristics; wall as given.
string fieldScalar(const FvStructured& s, const char* name, const char* dims, double start, double inf, const string& wall)
{
    string o = header("volScalarField", "0", name);
    o += std::format("dimensions      {};\n\ninternalField   uniform {};\n\n", dims, start);
    o += boundaryField(s, [&](const Patch& p) -> string {
        switch (p.role) {
        case Role::Inlet: return std::format("type fixedValue; value uniform {};", inf);
        case Role::Outlet: return std::format("type inletOutlet; inletValue uniform {}; value uniform {};", inf, inf);
        case Role::Wall: return wall;
        default: return std::format("type freestream; freestreamValue uniform {}; value uniform {};", inf, inf);
        }
    });
    return o;
}

string fieldNut(const FvStructured& s, double start, double inf)
{
    string o = header("volScalarField", "0", "nut");
    o += std::format("dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform {};\n\n", start);
    o += boundaryField(s, [&](const Patch& p) -> string {
        if (p.role == Role::Wall) return "type nutLowReWallFunction; value uniform 0;";
        return std::format("type calculated; value uniform {};", inf);
    });
    return o;
}

string controlDict(const FvStructured& s)
{
    string o = header("dictionary", "system", "controlDict");
    o += std::format("application     simpleFoam;\nstartFrom       startTime;\nstartTime       0;\nstopAt          endTime;\n"
                     "endTime         {};\ndeltaT          1;\nwriteControl    timeStep;\nwriteInterval   {};\n"
                     "writeFormat     ascii;\nwritePrecision  17;\nwriteCompression off;\ntimeFormat      general;\n"
                     "timePrecision   8;\nrunTimeModifiable false;\n",
                     s.iters, s.iters);
    const bool ras = s.model != FvTurbModel::Laminar;
    if (ras && hasCyclicPatch(s)) o += "libs            (\"libdistributed.so\");\n"; // exactDistance
    string walls;
    for (const Patch& p : patchesOf(s))
        if (p.role == Role::Wall) walls += (walls.empty() ? "" : " ") + p.name;
    if (!walls.empty()) {
        const double cx = s.mesh == FvStructMesh::OGrid ? s.ogrid.cx : 0, cy = s.mesh == FvStructMesh::OGrid ? s.ogrid.cy : 0,
                     cz = s.mesh == FvStructMesh::OGrid ? s.ogrid.cz : 0;
        o += std::format("\nfunctions\n{{\n    forces\n    {{\n        type        forces;\n        libs        (forces);\n"
                         "        writeControl timeStep;\n        writeInterval 1;\n        patches     ({});\n"
                         "        rho         rhoInf;\n        rhoInf      1;\n        CofR        {};\n    }}\n}}\n",
                         walls, vec(cx, cy, cz));
    }
    return o;
}

string fvSchemes(const FvStructured& s)
{
    const bool ras = s.model != FvTurbModel::Laminar;
    string o = header("dictionary", "system", "fvSchemes");
    o += "ddtSchemes\n{\n    default         steadyState;\n}\n\ngradSchemes\n{\n    default         Gauss linear;\n}\n\n"
         "divSchemes\n{\n    default         none;\n    div(phi,U)      bounded Gauss linearUpwind grad(U);\n";
    if (s.model == FvTurbModel::KOmegaSST)
        o += "    div(phi,k)      bounded Gauss upwind;\n    div(phi,omega)  bounded Gauss upwind;\n";
    if (s.model == FvTurbModel::SpalartAllmaras) o += "    div(phi,nuTilda) bounded Gauss upwind;\n";
    o += "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n\n"
         "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
         "interpolationSchemes\n{\n    default         linear;\n}\n\n"
         "snGradSchemes\n{\n    default         corrected;\n}\n";
    if (ras)
        o += std::format("\nwallDist\n{{\n    method          {};\n}}\n", hasCyclicPatch(s) ? "exactDistance" : "meshWave");
    return o;
}

string fvSolution(const FvStructured& s)
{
    string o = header("dictionary", "system", "fvSolution");
    o += "solvers\n{\n    p\n    {\n        solver          GAMG;\n        smoother        GaussSeidel;\n"
         "        tolerance       1e-10;\n        relTol          0.01;\n    }\n";
    std::vector<const char*> smoothed{"U"};
    if (s.model == FvTurbModel::KOmegaSST) smoothed.insert(smoothed.end(), {"k", "omega"});
    if (s.model == FvTurbModel::SpalartAllmaras) smoothed.push_back("nuTilda");
    for (const char* f : smoothed)
        o += std::format("    {}\n    {{\n        solver          smoothSolver;\n        smoother        GaussSeidel;\n"
                         "        tolerance       1e-12;\n        relTol          0.01;\n    }}\n",
                         f);
    o += "}\n\nSIMPLE\n{\n";
    const bool bodyWarped = s.mesh == FvStructMesh::OGrid;
    o += std::format("    nNonOrthogonalCorrectors {};\n    consistent      yes;\n    pRefCell        0;\n"
                     "    pRefValue       0;\n",
                     bodyWarped ? 1 : 0);
    if (s.residual) {
        o += "    residualControl\n    {\n";
        o += std::format("        p               {};\n        U               {};\n", *s.residual, *s.residual);
        for (const char* f : smoothed)
            if (std::string_view(f) != "U") o += std::format("        {:<15} {};\n", f, *s.residual);
        o += "    }\n";
    }
    // The laminar channel is linear in U: 0.9 only slows it. Not 1: SIMPLEC's 1/(A - H1) is singular on an unrelaxed
    // row and plain SIMPLE at 1 blew up by iteration 20 (measured).
    const bool linear = s.mesh == FvStructMesh::Channel && s.model == FvTurbModel::Laminar;
    o += std::format("}}\n\nrelaxationFactors\n{{\n    equations\n    {{\n        U               {};\n", linear ? 0.99 : 0.9);
    for (const char* f : smoothed)
        if (std::string_view(f) != "U") o += std::format("        {:<15} 0.7;\n", f);
    o += "    }\n}\n";
    return o;
}

string turbulenceProperties(const FvStructured& s)
{
    string o = header("dictionary", "constant", "turbulenceProperties");
    if (s.model == FvTurbModel::Laminar) return o + "simulationType  laminar;\n";
    o += "simulationType  RAS;\n\nRAS\n{\n";
    if (s.model == FvTurbModel::KOmegaSST) o += "    RASModel        kOmegaSST;\n    turbulence      on;\n";
    else o += "    RASModel        SpalartAllmaras;\n    turbulence      on;\n    SpalartAllmarasCoeffs\n    {\n        ft2             true;\n    }\n";
    return o + "}\n";
}

} // namespace

std::string fvStructuredLabel(const FvStructured& s)
{
    const Grid& g = s.grid;
    if (s.mesh == FvStructMesh::OGrid) {
        if (s.revolved && s.rev.wedge)
            return std::format("O-grid wedge {:g} deg, {} x {} (`symmetry` sides, not OpenFOAM `wedge`), axis prisms",
                               s.rev.dphi * 180.0 / std::numbers::pi, g.nx - 2, g.ny - 2);
        if (s.revolved) return std::format("O-grid full revolution, {} x {} x {}, axis prisms", g.nx - 2, g.ny - 2, g.nz);
        return std::format("O-grid {} x {} x {}", g.nx, g.ny - 2, g.nz);
    }
    const string map = s.focal         ? std::format("--focal ratio {:g}", s.focalRatio)
                       : s.stretchBeta > 0 ? std::format("--stretch {:g}", s.stretchBeta)
                                           : string("uniform");
    if (s.mesh == FvStructMesh::Channel)
        return std::format("plane channel Re_tau {:g}, {} x {} x {}, {}", s.reTau, g.nx, s.focal ? g.ny : g.ny - 2, g.nz, map);
    return std::format("box {} x {} x {}, {}", g.nx, g.ny, g.nz, map);
}

std::expected<PolyManifest, std::string> fvStructuredManifest(const FvStructured& s)
{
    auto m = FaceMesh::polyManifest(s.grid, wrapOf(s), axisOf(s), specOf(s));
    if (!m) return m;
    if (s.omega != 0) m->cellZones = {"rotor"};
    return m;
}

FoamCaseTexts fvStructuredTexts(const FvStructured& s)
{
    FoamCaseTexts t;
    const bool channel = s.mesh == FvStructMesh::Channel;
    // Channel start: the log-law bulk velocity under RAS; laminar from rest.
    const double uStart = !channel ? s.uInf
                          : s.model == FvTurbModel::Laminar ? 0.0
                                                            : 2.5 * std::log(std::max(s.reTau, 3.0)) + 2.6;
    t["0/U"] = fieldU(s, uStart);
    t["0/p"] = fieldP(s);
    if (s.model == FvTurbModel::KOmegaSST) {
        const double kI = s.kInf * s.uInf * s.uInf, wI = kI / (s.mutRatio * s.nu);
        const double k0 = channel ? 1.0 : kI, w0 = channel ? 20.0 : wI; // channel: u_tau^2, nut 0.05
        t["0/k"] = fieldScalar(s, "k", "[0 2 -2 0 0 0 0]", k0, kI, "type fixedValue; value uniform 0;");
        t["0/omega"] = fieldScalar(s, "omega", "[0 0 -1 0 0 0 0]", w0, wI,
                                   std::format("type omegaWallFunction; value uniform {};", w0));
        t["0/nut"] = fieldNut(s, k0 / w0, kI / wI);
    }
    if (s.model == FvTurbModel::SpalartAllmaras) {
        const double n = s.nutRatio * s.nu, n0 = channel ? 0.05 : n;
        const double chi = s.nutRatio, c3 = chi * chi * chi, cv1 = 7.1;
        const double nutI = n * c3 / (c3 + cv1 * cv1 * cv1);
        t["0/nuTilda"] = fieldScalar(s, "nuTilda", "[0 2 -1 0 0 0 0]", n0, n, "type fixedValue; value uniform 0;");
        t["0/nut"] = fieldNut(s, channel ? n0 : nutI, nutI);
    }
    t["system/controlDict"] = controlDict(s);
    t["system/fvSchemes"] = fvSchemes(s);
    t["system/fvSolution"] = fvSolution(s);
    t["constant/transportProperties"] = header("dictionary", "constant", "transportProperties") +
                                        std::format("transportModel  Newtonian;\nnu              [0 2 -1 0 0 0 0] {};\n", s.nu);
    t["constant/turbulenceProperties"] = turbulenceProperties(s);
    if (channel)
        t["constant/fvOptions"] = header("dictionary", "constant", "fvOptions") +
                                  std::format("momentumSource\n{{\n    type            vectorSemiImplicitSource;\n"
                                              "    selectionMode   all;\n    volumeMode      specific;\n    sources\n"
                                              "    {{\n        U               ({} 0);\n    }}\n}}\n",
                                              vec(1, 0, 0));
    if (s.omega != 0)
        t["constant/MRFProperties"] = header("dictionary", "constant", "MRFProperties") +
                                      std::format("rotor\n{{\n    cellZone        rotor;\n    active          yes;\n"
                                                  "    origin          {};\n    axis            (0 0 1);\n"
                                                  "    omega           {};\n}}\n",
                                                  vec(s.ogrid.cx, s.ogrid.cy, s.ogrid.cz), s.omega);
    return t;
}

std::expected<FoamCase, std::string> fvStructuredResolved(const FvStructured& s)
{
    auto man = fvStructuredManifest(s);
    if (!man) return std::unexpected(man.error());
    return parseFoamCase(fvStructuredTexts(s), *man);
}

std::expected<PolyMesh, std::string> fvStructuredMesh(const FvStructured& s, const Sdf& sdf)
{
    const Grid& g = s.grid;
    CurvMesh cm;
    switch (s.mesh) {
    case FvStructMesh::Box:
    case FvStructMesh::Channel:
        if (s.focal)
            cm = CurvMesh::fromRemap(g, focalRemap(g, true, s.focalLo.data(), s.focalHi.data(), s.focalW.data(), s.focalRatio));
        else if (s.mesh == FvStructMesh::Channel) cm = CurvMesh::channelY(g, s.stretchBeta, 1.0);
        else if (s.stretchBeta > 0) cm = CurvMesh::stretchedY(g, s.stretchBeta);
        else cm = CurvMesh::uniform(g);
        break;
    case FvStructMesh::OGrid: {
        auto og = s.revolved ? buildRevolvedOGrid(g, sdf, s.ogrid, s.rev) : buildOGrid(g, sdf, s.ogrid);
        if (!og) return std::unexpected("o-grid: " + og.error());
        cm = std::move(og->mesh);
        break;
    }
    }
    const StructuredSpec sp = specOf(s);
    auto pm = FaceMesh::polyOf(cm, sp);
    if (!pm) return pm;
    if (s.omega != 0) {
        // The zone: cells whose eight-node centroid lies inside the shaft radius.
        const int ni = sp.hi[0] - sp.lo[0] + 1, nj = sp.hi[1] - sp.lo[1] + 1, nk = sp.hi[2] - sp.lo[2] + 1;
        std::vector<std::int32_t> zone;
        for (int k = 0; k < nk; ++k)
            for (int j = 0; j < nj; ++j)
                for (int i = 0; i < ni; ++i) {
                    double cx = 0, cy = 0;
                    for (int d = 0; d < 8; ++d) {
                        const auto p = cm.node(sp.lo[0] + i + (d & 1), sp.lo[1] + j + ((d >> 1) & 1), sp.lo[2] + k + ((d >> 2) & 1));
                        cx += 0.125 * p[0];
                        cy += 0.125 * p[1];
                    }
                    if (std::hypot(cx - s.ogrid.cx, cy - s.ogrid.cy) < s.mrfRadius)
                        zone.push_back(std::int32_t(i + ni * (j + nj * k)));
                }
        pm->man.cellZones = {"rotor"};
        pm->zoneCells = {std::move(zone)};
    }
    return pm;
}

std::expected<void, std::string> writeFoamCase(const fs::path& dir, const PolyMesh& pm, const FoamCaseTexts& t)
{
    if (auto r = writePolyMesh(dir / "constant" / "polyMesh", pm, true); !r) return r;
    for (const auto& [rel, text] : t) {
        const fs::path p = dir / rel;
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
        if (ec) return std::unexpected(p.parent_path().string() + ": cannot create (" + ec.message() + ")");
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        f << text;
        f.close();
        if (!f) return std::unexpected(p.string() + ": write failed");
    }
    return {};
}

} // namespace dns
