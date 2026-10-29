// An OpenFOAM v2606 incompressible case as the FV path reads it (docs/fv.md,
// masterplan_meshsolver P5b): the boundary conditions of 0/U and 0/p, the
// schemes of system/fvSchemes, the solver and SIMPLE controls of
// system/fvSolution, the run length and the forces function object of
// system/controlDict, nu, laminar, and the MRF zones. Every entry the solver
// does not reproduce EXACTLY is rejected BY NAME (exit 3) -- there is no
// "close enough" spelling: the point is identical-case parity with simpleFoam.
// Host only. parseFoamCase is pure (texts in); loadFoamCase reads the files.
#pragma once

#include "dns/foamdict.hpp"
#include "dns/polymesh.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace dns {

using FvVec3 = std::array<double, 3>;

// Velocity boundary conditions (OpenFOAM fvPatchField semantics, verbatim).
//  Fixed      fixedValue / noSlip / movingWallVelocity (static mesh)
//  ZeroGrad   zeroGradient
//  InletOutlet mixed, valueFraction neg(phi), refValue inletValue
//  Symmetry   slip / symmetry / symmetryPlane (basicSymmetry, per-face normal)
//  Freestream freestreamVelocity: mixed, valueFraction 0.5 - 0.5 (U_b.n)/|U_b|
//  Empty      empty (no fvPatch faces)
enum class FvUBc : std::uint8_t { Fixed, ZeroGrad, InletOutlet, Symmetry, Freestream, Empty };
// Pressure: Fixed (fixedValue), ZeroGrad (zeroGradient, and the scalar side of
// slip/symmetry), Freestream (freestreamPressure: mixed, vf 0.5 + 0.5 (U_b.n)/|U_b|), Empty.
enum class FvPBc : std::uint8_t { Fixed, ZeroGrad, Freestream, Empty };

const char* fvUBcName(FvUBc);
const char* fvPBcName(FvPBc);

// One field's value spec: uniform, or one value per patch face / cell in
// OpenFOAM order (nonuniform List).
struct FvValue
{
    std::vector<double> v; // 1 or 3 values (uniform), or n * (1|3) (nonuniform)
    bool uniform = true;
};

struct FvPatchBc
{
    std::string name;          // the polyMesh patch
    FoamPatchType meshType{};
    FvUBc u{};
    FvPBc p{};
    std::string uType, pType;  // the OpenFOAM spellings (reports)
    FvValue uValue;            // Fixed: value; InletOutlet: inletValue; Freestream: freestreamValue
    FvValue uInit;             // InletOutlet/Freestream `value` (initial boundary value), may be empty
    FvValue pValue;            // Fixed / Freestream value
    FvValue pInit;
};

enum class FvGrad : std::uint8_t { GaussLinear, CellLimited };
struct FvGradScheme
{
    FvGrad kind = FvGrad::GaussLinear;
    double k = 1; // cellLimited coefficient (minmod limiter), in [0, 1]
};

enum class FvDiv : std::uint8_t { Linear, LinearUpwind, Upwind };

struct FvSchemes
{
    FvGradScheme gradU, gradP;   // grad(U), grad(p) (default when not named)
    FvDiv divU = FvDiv::Linear;  // div(phi,U)
    bool bounded = false;        // `bounded Gauss ...`
    bool lapCorrected = true;    // laplacian Gauss linear corrected | uncorrected
    bool snGradCorrected = true; // snGradSchemes default corrected | uncorrected
    // `limited [corrected] psi` (limitedSnGrad, psi in [0, 1]): the corrected
    // scheme's correction times min(psi |dc (x_N - x_P)| / ((1 - psi) |corr|
    // + SMALL), 1), one limiter per face (|.| of the vector for U). < 0: not
    // limited. Only with lapCorrected / snGradCorrected.
    double lapLimit = -1, snGradLimit = -1;
};

// One linear solve's controls (OpenFOAM's lduMatrix::solver): stop when the
// normalised residual sum|b - Ax| / normFactor < tolerance, or < relTol x the
// initial one; between minIter and maxIter iterations.
struct FvSolverCtl
{
    std::string solver; // the OpenFOAM name (GAMG, PCG, smoothSolver, PBiCGStab, ...)
    double tolerance = 1e-6, relTol = 0;
    int maxIter = 1000, minIter = 0;
};

struct FvSimpleCtl
{
    int nNonOrthCorr = 0;
    bool consistent = false;
    bool momentumPredictor = true;
    std::int64_t pRefCell = 0; // OpenFOAM cell index (source numbering)
    bool pRefGiven = false;    // pRefCell present (required when p needs a reference)
    double pRefValue = 0;
    // residualControl: stop when every listed initial residual is below it
    // (U: the max over the solved components). Absent = run to endTime.
    std::optional<double> resP, resU;
};

// One MRFProperties zone (OpenFOAM MRFZone). The list order is MRFZoneList's:
// every zone operation (makeRelative, DDt, correctBoundaryVelocity) runs zone
// by zone in this order.
struct FvMrfZone
{
    std::string name, cellZone;
    bool active = true;
    // nonRotatingPatches resolved to mesh patch names (exact names and
    // quoted regular expressions), in mesh patch order.
    std::vector<std::string> nonRotatingPatches;
    FvVec3 origin{}, axis{}; // axis normalised
    double omega = 0;      // rad/s (Function1 constant)
};

// --- RAS (phase BL: OpenFOAM v2606 kOmegaSST / SpalartAllmaras for simpleFoam) ---

enum class FvTurbModel : std::uint8_t { Laminar, KOmegaSST, SpalartAllmaras };

// A transported turbulence scalar's BC (k, omega, nuTilda), OpenFOAM semantics:
//  Fixed       fixedValue
//  ZeroGrad    zeroGradient, kqRWallFunction (a zeroGradient), slip / symmetry /
//              symmetryPlane (a scalar's basicSymmetry: zero normal gradient)
//  InletOutlet inletOutlet / freestream (mixed, valueFraction neg(phi))
//  OmegaWall   omegaWallFunction (omega only, wall patches): the near-wall cell
//              value set by the wall function, the cell fixed by setValues
//  Empty       empty, cyclic (no fvPatch faces)
enum class FvSBc : std::uint8_t { Fixed, ZeroGrad, InletOutlet, OmegaWall, Empty };
// nut: Calculated (the model's expression), LowRe (nutLowReWallFunction: 0),
// Spalding (nutUSpaldingWallFunction), KWall (nutkWallFunction), Cell
// (zeroGradient / symmetry types: the cell value), Fixed (fixedValue: kept),
// Empty (empty, cyclic).
enum class FvNutBc : std::uint8_t { Calculated, LowRe, Spalding, KWall, Cell, Fixed, Empty };
// wallFunctionBlenders (omegaWallFunction default binomial n 2, nutk stepwise).
enum class FvBlend : std::uint8_t { Stepwise, Max, Binomial, Exponential, Tanh };
const char* fvBlendName(FvBlend);

// wallFunctionCoefficients: Cmu, kappa, E; yPlusLam = 10 fixed-point
// iterations of log(max(E y, 1)) / kappa from 11 (calcYPlusLam verbatim).
struct FvWallCoeffs
{
    double Cmu = 0.09, kappa = 0.41, E = 9.8;
    double yPlusLam() const;
};

struct FvScalarBc
{
    FvSBc kind{};
    std::string type;
    FvValue value; // Fixed: value; InletOutlet: inletValue / freestreamValue
    FvValue init;  // `value` (initial boundary value) where it is not the BC itself
    // omegaWallFunction
    FvBlend blend = FvBlend::Binomial;
    double n = 2, beta1 = 0.075;
    FvWallCoeffs wc;
};

struct FvNutPatch
{
    FvNutBc kind{};
    std::string type;
    FvValue value; // `value`: the initial patch value (wall functions keep it as the first guess)
    FvWallCoeffs wc;
    FvBlend blend = FvBlend::Stepwise; // nutkWallFunction
    double n = 4;
    int maxIter = 10;       // nutUSpaldingWallFunction
    double tolerance = 0.01;
};

// A turbulence scalar's convection scheme: [bounded] Gauss linear | upwind |
// linearUpwind grad(X) | limitedLinear k (k in [0, 1]).
enum class FvSDiv : std::uint8_t { Linear, Upwind, LinearUpwind, LimitedLinear };
struct FvConvScheme
{
    FvSDiv kind = FvSDiv::Upwind;
    bool bounded = false;
    double k = 1;
};

struct FvRas
{
    FvTurbModel model = FvTurbModel::Laminar;
    // kOmegaSST coefficients (kOmegaSSTBase defaults; kOmegaSSTCoeffs overrides)
    double alphaK1 = 0.85, alphaK2 = 1.0, alphaOmega1 = 0.5, alphaOmega2 = 0.856, gamma1 = 5.0 / 9.0,
           gamma2 = 0.44, beta1 = 0.075, beta2 = 0.0828, betaStar = 0.09, a1 = 0.31, b1 = 1.0, c1 = 10.0;
    bool F3 = false;
    // SpalartAllmaras coefficients (SpalartAllmarasBase defaults)
    double sigmaNut = 0.66666, kappa = 0.41, Cb1 = 0.1355, Cb2 = 0.622, Cw2 = 0.3, Cw3 = 2.0, Cv1 = 7.1,
           Cs = 0.3, Ct3 = 1.2, Ct4 = 0.5;
    bool ft2 = false;
    double kMin = 1e-15, omegaMin = 1e-15; // RASModel kMin / omegaMin (SMALL)
    // wallDist: meshWave (correctWalls) | exactDistance (U9: the knob is stated
    // in every report)
    bool exactDist = false, correctWalls = true;
    FvValue kInt, omegaInt, nutInt, nuTildaInt;
    std::vector<FvScalarBc> k, omega, nuTilda; // per patch (polyMesh order), the model's fields only
    std::vector<FvNutPatch> nut;
    FvConvScheme divK, divOmega, divNuTilda;
    FvGradScheme gradK, gradOmega, gradNuTilda;
    FvSolverCtl solverK, solverOmega, solverNuTilda;
    std::optional<double> relaxK, relaxOmega, relaxNuTilda; // equations { }
    std::optional<double> resK, resOmega, resNuTilda;       // residualControl
};

const char* fvTurbModelName(FvTurbModel);

// The forces function object (rho rhoInf, kinematic p).
struct FvForces
{
    std::string name = "forces";
    std::vector<std::string> patches;
    double rhoInf = 1;
    FvVec3 CofR{};
    double pRef = 0;
};

struct FoamCase
{
    std::string dir;
    double nu = 0;
    int endIter = 0;        // endTime / deltaT
    double deltaT = 1;      // time per iteration (time directory names)
    int writeInterval = 0;  // in iterations (0: only the end)
    int writePrecision = 6;
    std::vector<FvPatchBc> patches;         // polyMesh boundary order (every patch)
    FvValue uInternal, pInternal;
    FvSchemes schemes;
    FvSolverCtl solverU, solverP;
    FvSimpleCtl simple;
    double relaxP = 1;                      // fields { p }: absent = 1 (no relax)
    bool relaxPGiven = false;
    std::optional<double> relaxU;           // equations { U }: absent = no relax
    std::vector<FvMrfZone> mrf;
    FvRas ras;                              // model Laminar: the P5b laminar solver
    std::optional<FvForces> forces;
    std::vector<std::string> notes;         // accepted-but-not-reproduced entries (reported)
};

// The texts of one case (relative path -> file contents). Required:
// 0/U, 0/p, system/{controlDict,fvSchemes,fvSolution},
// constant/transportProperties; optional: constant/MRFProperties,
// constant/momentumTransport | constant/turbulenceProperties; with RAS:
// 0/nut and 0/k + 0/omega (kOmegaSST) or 0/nuTilda (SpalartAllmaras).
// Present only to be rejected: constant/fvOptions, system/fvOptions.
using FoamCaseTexts = std::map<std::string, std::string>;

// Pure: resolves the case against the mesh manifest (every non-cyclic patch
// must get a BC on U and p; constraint patches take their constraint type).
// Errors (exit 3) name the file, line and entry.
std::expected<FoamCase, std::string> parseFoamCase(const FoamCaseTexts& t, const PolyManifest& man);

// Reads the case files under dir (dir/constant/polyMesh is the mesh).
// Error: a missing required file, an unreadable one.
std::expected<FoamCaseTexts, std::string> readFoamCaseTexts(const std::filesystem::path& dir);

// The --dry-run / log description of a resolved case.
std::string describeFoamCase(const FoamCase&);

} // namespace dns
