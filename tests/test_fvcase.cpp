// The OpenFOAM case layer of the incompressible FV path (docs/fv.md, P5b):
//   D1  dns/foamdict: OpenFOAM tokens (balanced-paren words, regex keys,
//       directives, valueless entries, lists), last-wins keys, errors by line
//   C1  the checked-in IF1 cases resolve to the documented BCs / schemes /
//       controls (cavity: closed, needs a reference; channel: linearUpwind,
//       cellLimited, inletOutlet, slip; skew: SIMPLEC, 2 correctors)
//   C2  every unsupported entry is rejected BY NAME (the error names it),
//       fvOptions files and forces e1/e3 without CofR included; CofR wins
//       over origin (forces::setCoordinateSystem)
//   S1  OpenFOAM's geometric factors on the uniform cavity: w 1/2, dcNO 1/h,
//       k 0, patch dc 2/h; the skewed cavity: k != 0, n.k == 0 (k is
//       tangential), w in (0, 1); CSR / ELL consistent; pRefCell mapped
//   M1  MRFZoneList (mrf2): zones in file order, an inactive zone, omega
//       `constant`, axis normalised, nonRotatingPatches literal + regex,
//       per-zone Omega, both frames on zone/zone faces; rejections by name
//       (unmatched selectors, time-dependent omega, shared cellZone, zero
//       axis, more than 8 active zones)
//   O1  dns/foamio: write -> read round trip of an ASCII field (17 digits)
//   D2  dns/foamdict macros: $name / ${name} in a value, `$name;` splices,
//       the scoped lookup, every other form rejected by name
//   C3  the M17 case's forms: `format binary` boundary + cellZones, macros in
//       0/ and controlDict, an equations factor matching p, several forces /
//       forceCoeffs objects, yPlus skipped; fvForceCoeffRow vs simpleFoam
//   K1  fvOptions actuationDiskSource: spellings, rejections by name,
//       OpenFOAM's findCell with its ties, the zone fractions
//   P1-P3 pimpleFoam (docs/fvpimple.md): the PIMPLE dictionary and the Final
//       entries, rejections by name (ddt schemes, loop counts, dynamicMeshDict,
//       MRF, fvOptions sources), FoamClock == pimpleFoam's printed deltaT
#include "dns/foamdict.hpp"
#include "dns/foamio.hpp"
#include "dns/fvcase.hpp"
#include "dns/fvmotion.hpp"
#include "dns/fvpimple.hpp"
#include "dns/fvsimple.hpp"
#include "dns/la/graph.hpp"

#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

#ifndef DNS_FV_DATA
#error "DNS_FV_DATA must point at tests/data/fv"
#endif

using namespace dns;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

fs::path data(const char* c) { return fs::path(DNS_FV_DATA) / c; }

void testDict()
{
    std::printf("D1 foamdict\n");
    auto d = foam::parseDict(R"X(
FoamFile { version 2.0; format ascii; }
a 1;               // comment
b /* block
     comment */ 2.5e-3;
div((nuEff*dev2(T(grad(U))))) Gauss linear;
"(U|e)" { solver smoothSolver; }
cache { grad(U); }
v (1 -2 3e1);
#includeEtc "caseDicts/setConstraintTypes"
a 7;
)X",
                             "t");
    check(bool(d), "parses");
    if (!d) {
        std::printf("    %s\n", d.error().c_str());
        return;
    }
    const auto* a = d->find("a");
    check(a && foam::scalarOf(*d, *a).value_or(0) == 7, "a repeated key: the last wins (7)");
    const auto* b = d->find("b");
    check(b && foam::scalarOf(*d, *b).value_or(0) == 2.5e-3, "comments skipped, 2.5e-3");
    const auto* t = d->find("div((nuEff*dev2(T(grad(U)))))");
    check(t && foam::joined(*t) == "Gauss linear", "a word carries balanced parentheses");
    bool regex = false, dir = false;
    for (const auto& e : d->entries) {
        regex = regex || (e.quotedKey && e.key == "(U|e)" && e.dict);
        dir = dir || (e.directive && e.key == "#includeEtc" && e.value.size() == 1);
    }
    check(regex, "a quoted key is kept as a regex entry");
    check(dir, "#includeEtc kept as a directive with its argument");
    const auto* c = d->find("cache");
    check(c && c->dict && c->dict->find("grad(U)") && c->dict->find("grad(U)")->value.empty(),
          "a valueless entry `grad(U);` is accepted");
    const auto* v = d->find("v");
    std::size_t pos = 0;
    auto vv = v ? foam::vectorOf(*d, v->value, pos, v->line, "v") : std::unexpected(std::string("no v"));
    check(vv && (*vv)[0] == 1 && (*vv)[1] == -2 && (*vv)[2] == 30, "a vector (1 -2 3e1)");
    auto e1 = foam::parseDict("a { b 1; ", "t1");
    check(!e1 && e1.error().find("unclosed") != std::string::npos, "unclosed '{' is an error: " + (e1 ? "" : e1.error()));
    auto e2 = foam::parseDict("a (1 2;\n", "t2");
    check(!e2, "unbalanced list is an error");
    auto e3 = foam::parseDict("a 1\nc { }", "t3");
    check(!e3 && e3.error().find("t3:") == 0, "a missing ';' is an error naming the file: " + (e3 ? "" : e3.error()));
}

std::expected<FoamCase, std::string> load(const fs::path& dir, const FoamCaseTexts* texts = nullptr)
{
    auto man = readPolyManifest(dir / "constant" / "polyMesh");
    if (!man) return std::unexpected(man.error());
    if (texts) return parseFoamCase(*texts, *man);
    auto t = readFoamCaseTexts(dir);
    if (!t) return std::unexpected(t.error());
    return parseFoamCase(*t, *man);
}

void testCases()
{
    std::printf("C1 the IF1 cases resolve\n");
    auto cav = load(data("cavity"));
    check(bool(cav), "cavity parses" + (cav ? std::string() : ": " + cav.error()));
    if (cav) {
        check(cav->nu == 0.001 && cav->endIter == 400 && cav->writeInterval == 100, "cavity nu 1e-3, 400 it, write 100");
        check(cav->patches.size() == 3 && cav->patches[0].u == FvUBc::Fixed && cav->patches[1].u == FvUBc::Fixed &&
                  cav->patches[2].u == FvUBc::Empty && cav->patches[0].p == FvPBc::ZeroGrad,
              "cavity: movingWall fixedValue, fixedWalls noSlip -> Fixed, empty; p zeroGradient");
        check(cav->patches[0].uValue.v == std::vector<double>{1, 0, 0}, "movingWall value (1 0 0)");
        check(cav->schemes.bounded && cav->schemes.divU == FvDiv::Linear && cav->schemes.lapCorrected,
              "bounded Gauss linear, corrected");
        check(!cav->simple.consistent && cav->simple.nNonOrthCorr == 0 && cav->simple.pRefGiven, "SIMPLE controls");
        check(cav->relaxPGiven && cav->relaxP == 0.3 && cav->relaxU && *cav->relaxU == 0.7, "relax p 0.3, U 0.7");
        check(cav->solverP.solver == "PCG" && cav->solverP.tolerance == 1e-13 && cav->solverP.relTol == 0 &&
                  cav->solverP.maxIter == 5000,
              "p solver controls");
    }
    auto ch = load(data("channel"));
    check(bool(ch), "channel parses" + (ch ? std::string() : ": " + ch.error()));
    if (ch) {
        check(ch->schemes.divU == FvDiv::LinearUpwind && ch->schemes.gradU.kind == FvGrad::CellLimited &&
                  ch->schemes.gradU.k == 1 && ch->schemes.gradP.kind == FvGrad::GaussLinear,
              "linearUpwind grad(U), cellLimited Gauss linear 1 on grad(U) only");
        check(ch->patches[1].u == FvUBc::InletOutlet && ch->patches[1].uValue.v == std::vector<double>{0, 0, 0} &&
                  ch->patches[1].p == FvPBc::Fixed,
              "outlet inletOutlet + fixedValue p");
        check(ch->patches[3].u == FvUBc::Symmetry && ch->patches[3].uType == "slip" && ch->patches[3].p == FvPBc::ZeroGrad,
              "lid slip (scalar side zeroGradient)");
        check(ch->patches[4].u == FvUBc::Empty, "frontAndBack empty via setConstraintTypes");
    }
    auto sk = load(data("cavity_skew"));
    check(sk && sk->simple.consistent && sk->simple.nNonOrthCorr == 2 && sk->relaxP == 1 && *sk->relaxU == 0.9,
          "skew: SIMPLEC, 2 non-orthogonal correctors, p 1, U 0.9");
    check(sk && sk->schemes.lapLimit < 0 && sk->schemes.snGradLimit < 0, "skew: corrected, not limited");
    auto lim = load(data("cavity_lim"));
    check(lim && lim->schemes.lapCorrected && lim->schemes.lapLimit == 0.33 && lim->schemes.snGradCorrected &&
              lim->schemes.snGradLimit == 0.33,
          "cavity_lim: laplacian and snGrad `limited corrected 0.33`" + (lim ? std::string() : ": " + lim.error()));
    if (auto t = readFoamCaseTexts(data("cavity_lim")); t) {
        auto& s = (*t)["system/fvSchemes"];
        for (auto [from, to] : {std::pair{std::string("Gauss linear limited corrected 0.33;"),
                                          std::string("Gauss linear limited 0.5;")},
                                {std::string("default limited corrected 0.33;"), std::string("default limited 0.25;")}})
            if (const auto at = s.find(from); at != std::string::npos) s.replace(at, from.size(), to);
        auto r = load(data("cavity_lim"), &*t);
        check(r && r->schemes.lapLimit == 0.5 && r->schemes.snGradLimit == 0.25,
              "the backward-compatible `limited psi` spelling = limited corrected psi" +
                  (r ? std::string() : ": " + r.error()));
        const std::string d = r ? describeFoamCase(*r) : std::string();
        check(d.find("laplacian Gauss linear limited corrected 0.5") != std::string::npos &&
                  d.find("snGrad limited corrected 0.25") != std::string::npos,
              "describe names the limited schemes");
    }

    std::printf("C2 rejections by name\n");
    auto base = readFoamCaseTexts(data("cavity"));
    if (!base) {
        check(false, "cavity texts: " + base.error());
        return;
    }
    struct Bad
    {
        const char* file;
        std::string from, to;
        const char* token;
        const char* what;
    };
    const Bad bad[] = {
        {"system/fvSchemes", "bounded Gauss linear;", "bounded Gauss limitedLinear 1;", "limitedLinear",
         "a div scheme not reproduced"},
        {"system/fvSchemes", "default steadyState;", "default Euler;", "steadyState", "a transient ddt scheme"},
        {"system/fvSchemes", "default Gauss linear corrected;", "default Gauss linear limited uncorrected 0.33;",
         "limited uncorrected", "laplacian limited uncorrected (no correction to limit)"},
        {"system/fvSchemes", "default Gauss linear corrected;", "default Gauss linear limited corrected 1.5;",
         "limited corrected 1.5", "a limiter coefficient outside [0, 1]"},
        {"system/fvSchemes", "default Gauss linear corrected;", "default Gauss linear faceCorrected;",
         "faceCorrected", "an snGrad scheme not reproduced"},
        {"system/fvSchemes", "gradSchemes     { default Gauss linear; }",
         "gradSchemes     { default leastSquares; }", "leastSquares", "a gradient scheme"},
        {"0/U", "type noSlip;", "type totalPressure;", "totalPressure", "an unknown velocity BC"},
        {"0/p", "movingWall { type zeroGradient; }", "movingWall { type fixedFluxPressure; }", "fixedFluxPressure",
         "fixedFluxPressure"},
        {"0/p", "[0 2 -2 0 0 0 0]", "[1 -1 -2 0 0 0 0]", "KINEMATIC", "a compressible (dynamic) p"},
        {"0/U", "internalField uniform (0 0 0);", "internalField uniform (0 0 0);\n#include \"x\"", "#include",
         "#include"},
        {"system/controlDict", "application simpleFoam;", "application pimpleFoam;", "pimpleFoam",
         "pimpleFoam on simpleFoam's steadyState schemes"},
        {"system/controlDict", "application simpleFoam;", "application pisoFoam;", "pisoFoam", "another application"},
        {"system/controlDict", "runTimeModifiable false;", "runTimeModifiable false;\nmaxCo 2;", "maxCo",
         "a transient time control on simpleFoam"},
        {"system/controlDict", "runTimeModifiable false;",
         "runTimeModifiable false;\nfunctions { s { type surfaces; } }", "surfaces", "a function object type"},
        {"constant/turbulenceProperties", "simulationType laminar;", "simulationType RAS;", "RAS", "RAS"},
        {"system/fvSolution", "relaxationFactors { fields { p 0.3; } equations { U 0.7; } }",
         "relaxationFactors { p 0.3; U 0.7; }", "'p'", "old-style relaxationFactors"},
        {"system/fvSolution", "solver PCG;", "solver diagonal;", "diagonal", "a p solver"},
        {"system/fvSolution", "pRefValue 0;", "pRefValue 0;\n    pRefPoint (0 0 0);", "pRefPoint", "pRefPoint"},
        {"system/fvSolution", "nNonOrthogonalCorrectors 0;", "nNonOrthogonalCorrectors 0;\n    transonic yes;",
         "transonic", "transonic"},
        {"system/fvSolution", "relTol 0; maxIter 5000; }\n    U", "relTol 0; maxIter 5000; nonsense 1; }\n    U",
         "nonsense", "an unknown solver key"},
    };
    for (const Bad& b : bad) {
        FoamCaseTexts t = *base;
        std::string& s = t[b.file];
        const auto at = s.find(b.from);
        if (at == std::string::npos) {
            check(false, std::format("{}: fixture text '{}' not found", b.what, b.from));
            continue;
        }
        s.replace(at, b.from.size(), b.to);
        auto r = load(data("cavity"), &t);
        const bool named = !r && r.error().find(b.token) != std::string::npos;
        check(named, std::format("{} rejected by name: {}", b.what, r ? std::string("ACCEPTED") : r.error()));
    }
    {
        FoamCaseTexts t = *base;
        t["constant/MRFProperties"] = "a { cellZone nope; origin (0 0 0); axis (1 0 0); omega 1; }";
        auto r = load(data("cavity"), &t);
        check(!r && r.error().find("nope") != std::string::npos, "an MRF cellZone not in the mesh: " +
                                                                     (r ? std::string("ACCEPTED") : r.error()));
    }
    for (const char* f : {"constant/fvOptions", "system/fvOptions"}) {
        FoamCaseTexts t = *base;
        t[f] = "src { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; "
               "sources { U ((1 2 3) 0); } }";
        auto r = load(data("cavity"), &t);
        check(r && r->uSource == FvVec3{1, 2, 3}, std::format("{}: the one built source is read", f));
        // fv::optionList::reset (v2606): no dictionary entry = an empty list, source-free.
        for (const char* txt : {"", "FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\n"
                                    "// No actuator disks.\n"}) {
            t[f] = txt;
            auto e = load(data("cavity"), &t);
            const bool noted = e && std::ranges::any_of(e->notes, [&](const std::string& n) {
                return n.find(f) != std::string::npos && n.find("no source in the file") != std::string::npos;
            });
            check(e && !e->uSource && e->disks.empty() && noted,
                  std::format("{}: a file with no source runs source-free, noted: {}", f, e ? "ok" : e.error()));
        }
        const char* rejected[] = {
            "src { type meanVelocityForce; selectionMode all; Ubar (1 0 0); }",
            "src { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; sources { U ((1 0 0) 2); } }",
            "src { type vectorSemiImplicitSource; selectionMode all; volumeMode absolute; sources { U ((1 0 0) 0); } }",
            "src { type vectorSemiImplicitSource; selectionMode cellZone; volumeMode specific; sources { U ((1 0 0) 0); } }",
            "src { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; sources { k (1 0); } }",
            "src { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; timeStart 1; sources { U ((1 0 0) 0); } }",
            "a { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; sources { U ((1 0 0) 0); } } "
            "b { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; sources { U ((1 0 0) 0); } }",
        };
        for (const char* txt : rejected) {
            t[f] = txt;
            auto x = load(data("cavity"), &t);
            check(!x && x.error().find(f) != std::string::npos,
                  std::format("{} '{:.40}': {}", f, txt, x ? "ACCEPTED" : x.error()));
        }
    }
    {
        // fvOptions.C createIOobject: constant/fvOptions wins, system/fvOptions is then never read.
        FoamCaseTexts t = *base;
        t["constant/fvOptions"] = "FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\n";
        t["system/fvOptions"] = "src { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; "
                                "sources { U ((1 2 3) 0); } }";
        auto r = load(data("cavity"), &t);
        check(!r && r.error().find("both exist") != std::string::npos,
              "constant/fvOptions + system/fvOptions rejected by name: " + (r ? std::string("ACCEPTED") : r.error()));
    }
    {
        // forces::setCoordinateSystem: without CofR, origin + e1/e3 rotate the output frame (rejected);
        // with CofR OpenFOAM reads neither origin nor e1/e3 (CofR is the moment centre).
        const auto withForces = [&](const char* extra) {
            FoamCaseTexts t = *base;
            std::string& s = t["system/controlDict"];
            s.replace(s.find("runTimeModifiable false;"), 24,
                      std::string("runTimeModifiable false;\nfunctions { forces { type forces; libs (forces); "
                                  "patches (movingWall); rho rhoInf; rhoInf 1; ") +
                          extra + " } }");
            return load(data("cavity"), &t);
        };
        auto r1 = withForces("origin (0 0 0); e1 (0 1 0); e3 (0 0 1);");
        check(!r1 && r1.error().find("'e1'") != std::string::npos && r1.error().find("CofR") != std::string::npos,
              "forces e1 without CofR (rotated frame) rejected by name: " + (r1 ? std::string("ACCEPTED") : r1.error()));
        auto r2 = withForces("origin (0 0 0); e3 (1 0 0);");
        check(!r2 && r2.error().find("'e3'") != std::string::npos,
              "forces e3 without CofR rejected by name: " + (r2 ? std::string("ACCEPTED") : r2.error()));
        auto r3 = withForces("CofR (1 2 3); origin (4 5 6); e1 (0 1 0); e3 (0 0 1);");
        check(r3 && r3->forces.size() == 1 && r3->forces[0].CofR == FvVec3{1, 2, 3},
              "forces with CofR: CofR wins over origin, e1/e3 unread (as OpenFOAM)" +
                  (r3 ? std::string() : ": " + r3.error()));
        // simpleFoam v2606 stops on both: origin only = "No entries of the type (e1, e2) or (e2, e3) or
        // (e3, e1) found"; nothing = "Entry 'origin' not found".
        for (const char* extra : {"origin (4 5 6);", ""}) {
            auto r4 = withForces(extra);
            check(!r4 && r4.error().find("needs 'CofR'") != std::string::npos,
                  std::format("forces without CofR ('{}', an OpenFOAM error) rejected by name: {}", extra,
                              r4 ? "ACCEPTED" : r4.error()));
        }
    }
    {
        FoamCaseTexts t = *base;
        t.erase("0/p");
        auto r = load(data("cavity"), &t);
        check(!r && r.error().find("0/p") != std::string::npos, "a missing 0/p");
    }
}

// R1: the RAS case layer (phase BL).
void testRas()
{
    std::printf("R1 RAS cases resolve\n");
    auto p = load(data("sst_plate"));
    check(bool(p), "sst_plate parses" + (p ? std::string() : ": " + p.error()));
    if (p) {
        const FvRas& r = p->ras;
        check(r.model == FvTurbModel::KOmegaSST && !r.exactDist && r.correctWalls, "kOmegaSST, meshWave, correctWalls");
        check(r.alphaK1 == 0.85 && r.betaStar == 0.09 && r.a1 == 0.31 && r.c1 == 10 && !r.F3 && r.gamma1 == 5.0 / 9.0,
              "kOmegaSSTBase default coefficients");
        // patches: inlet outlet top plate symmetry frontAndBack
        check(r.k[0].kind == FvSBc::Fixed && r.k[1].kind == FvSBc::InletOutlet && r.k[3].kind == FvSBc::ZeroGrad &&
                  r.k[3].type == "kqRWallFunction" && r.k[4].kind == FvSBc::ZeroGrad && r.k[5].kind == FvSBc::Empty,
              "k: fixedValue, inletOutlet, kqRWallFunction (a zeroGradient), symmetryPlane, empty");
        check(r.omega[3].kind == FvSBc::OmegaWall && r.omega[3].blend == FvBlend::Binomial && r.omega[3].n == 2 &&
                  r.omega[3].beta1 == 0.075,
              "omegaWallFunction: default blending binomial n 2, beta1 0.075");
        check(r.nut[3].kind == FvNutBc::Spalding && r.nut[3].maxIter == 10 && r.nut[0].kind == FvNutBc::Calculated &&
                  r.nut[4].kind == FvNutBc::Cell,
              "nut: nutUSpaldingWallFunction (maxIter 10), calculated, symmetryPlane = the cell value");
        check(r.divK.kind == FvSDiv::LinearUpwind && r.divK.bounded && r.divOmega.kind == FvSDiv::LinearUpwind,
              "bounded Gauss linearUpwind grad(k|omega)");
        check(r.relaxK && *r.relaxK == 0.8 && r.relaxOmega && *r.relaxOmega == 0.8 && r.solverK.solver == "PBiCGStab",
              "k/omega relaxation 0.8, PBiCGStab");
        check(std::abs(r.omega[3].wc.yPlusLam() - 11.53) < 0.01, std::format("yPlusLam(kappa 0.41, E 9.8) = {:.4f}",
                                                                              r.omega[3].wc.yPlusLam()));
        const std::string d = describeFoamCase(*p);
        check(d.find("RAS         kOmegaSST") != std::string::npos && d.find("wallDist    meshWave") != std::string::npos,
              "describe names the model and the wall-distance knob");
    }
    auto s = load(data("sst_skew"));
    check(s && s->ras.exactDist && s->ras.omega[2].blend == FvBlend::Stepwise && s->ras.omega[3].blend == FvBlend::Tanh &&
              s->ras.nut[2].kind == FvNutBc::KWall && s->ras.nut[3].kind == FvNutBc::LowRe &&
              s->ras.gradK.kind == FvGrad::CellLimited && s->ras.gradOmega.kind == FvGrad::GaussLinear,
          "sst_skew: exactDistance, stepwise / tanh omega wall functions, nutk / nutLowRe, cellLimited grad(k)" +
              (s ? std::string() : ": " + s.error()));
    auto a = load(data("sa_skew"));
    check(a && a->ras.model == FvTurbModel::SpalartAllmaras && a->ras.ft2 && a->ras.sigmaNut == 0.66666 &&
              a->ras.nuTilda[2].kind == FvSBc::Fixed && a->ras.divNuTilda.kind == FvSDiv::LinearUpwind,
          "sa_skew: SpalartAllmaras ft2 on, sigmaNut 0.66666, nuTilda fixedValue walls" +
              (a ? std::string() : ": " + a.error()));

    std::printf("R2 RAS rejections by name\n");
    auto base = readFoamCaseTexts(data("sst_plate"));
    if (!base) {
        check(false, "sst_plate texts: " + base.error());
        return;
    }
    struct Bad
    {
        const char* file;
        std::string from, to;
        const char* token;
        const char* what;
    };
    const Bad bad[] = {
        {"constant/turbulenceProperties", "RASModel kOmegaSST;", "RASModel kEpsilon;", "kEpsilon", "another model"},
        {"constant/turbulenceProperties", "turbulence on;", "turbulence off;", "turbulence off", "turbulence off"},
        {"0/nut", "type nutUSpaldingWallFunction;", "type nutkRoughWallFunction;", "nutkRoughWallFunction",
         "a nut BC not reproduced"},
        {"0/nut", "type nutUSpaldingWallFunction;", "type nutUSpaldingWallFunction; tolerance 0.02;", "tolerance",
         "Spalding's second-pass tolerance"},
        {"0/omega", "inlet { type fixedValue; value uniform 5; }", "inlet { type omegaWallFunction; value uniform 5; }",
         "wall patch", "omegaWallFunction on a non-wall patch"},
        {"0/k", "type kqRWallFunction;", "type kLowReWallFunction;", "kLowReWallFunction", "a k BC not reproduced"},
        {"system/fvSchemes", "div(phi,k) bounded Gauss linearUpwind grad(k);", "div(phi,k) bounded Gauss QUICK;",
         "QUICK", "a turbulence div scheme"},
        {"system/fvSchemes", "wallDist { method meshWave; }", "wallDist { method Poisson; }", "Poisson",
         "a wall-distance method"},
        {"system/fvSchemes", "wallDist { method meshWave; }", "", "wallDist", "a missing wallDist"},
        {"system/fvSolution", "equations { U 0.9; k 0.8; omega 0.8; }", "fields { k 0.5; } equations { U 0.9; }",
         "fields/k", "field relaxation of k"},
        {"0/omega", "type omegaWallFunction;", "type omegaWallFunction; blending fancy;", "fancy", "a blending name"},
    };
    for (const Bad& b : bad) {
        FoamCaseTexts t = *base;
        std::string& x = t[b.file];
        const auto at = x.find(b.from);
        if (at == std::string::npos) {
            check(false, std::format("{}: fixture text '{}' not found", b.what, b.from));
            continue;
        }
        x.replace(at, b.from.size(), b.to);
        auto r = load(data("sst_plate"), &t);
        const bool named = !r && r.error().find(b.token) != std::string::npos;
        check(named, std::format("{} rejected by name: {}", b.what, r ? std::string("ACCEPTED") : r.error()));
    }
    {
        FoamCaseTexts t = *base;
        t.erase("0/nut");
        auto r = load(data("sst_plate"), &t);
        check(!r && r.error().find("0/nut") != std::string::npos, "a missing 0/nut: " + (r ? std::string("ACCEPTED") : r.error()));
    }
    std::printf("R3 RAS setup (sst_plate)\n");
    auto L = loadFvCase(data("sst_plate"));
    if (!L) {
        check(false, L.error());
        return;
    }
    const auto g = la::graphOf(L->fm);
    const auto pat = la::ellPattern(g);
    auto su = buildSimpleSetup(L->fm, L->c, g, pat);
    if (!su) {
        check(false, su.error());
        return;
    }
    auto rs = buildRasSetup(L->pm, L->fm, L->c, *su);
    check(bool(rs), "buildRasSetup" + (rs ? std::string() : ": " + rs.error()));
    if (!rs) return;
    const FvRasSetup& R = su->ras;
    int walls = 0, omegaWall = 0, spal = 0;
    bool yw = true;
    for (int b = 0; b < su->nB; ++b) {
        walls += R.bWall[std::size_t(b)];
        omegaWall += R.bS[1][std::size_t(b)] == std::uint8_t(FvSCode::OmegaWall);
        spal += R.bNut[std::size_t(b)] == std::uint8_t(FvNutCode::Spalding);
        if (R.bWall[std::size_t(b)]) yw = yw && R.yWall[std::size_t(b)] > 0;
    }
    check(walls == 32 + 40 && omegaWall == 32 && spal == 32 && yw,
          std::format("wall faces {} (plate 32 + top 40), omegaWallFunction {}, Spalding {}, nearWallDist > 0 on walls",
                      walls, omegaWall, spal));
    check(R.nS == 2 && R.s0[0].size() == std::size_t(su->nC) && R.y.size() == std::size_t(su->nC),
          "two transported scalars, initial fields and y per cell");
}

void testSetup()
{
    std::printf("S1 OpenFOAM geometric factors\n");
    for (const char* name : {"cavity", "cavity_skew"}) {
        auto L = loadFvCase(data(name));
        if (!L) {
            check(false, std::string(name) + ": " + L.error());
            continue;
        }
        const auto g = la::graphOf(L->fm);
        const auto pat = la::ellPattern(g);
        auto s = buildSimpleSetup(L->fm, L->c, g, pat);
        if (!s) {
            check(false, std::string(name) + " setup: " + s.error());
            continue;
        }
        const bool skew = std::string(name) == "cavity_skew";
        const std::size_t nI = std::size_t(s->nI);
        double wMin = 1, wMax = 0, kMax = 0, nkMax = 0, dcErr = 0;
        const double h = 0.1 / 24;
        for (std::size_t f = 0; f < nI; ++f) {
            wMin = std::min(wMin, s->w[f]);
            wMax = std::max(wMax, s->w[f]);
            double kk = 0, nk = 0;
            for (int q = 0; q < 3; ++q) {
                const double k = s->k[std::size_t(q) * nI + f];
                kk += k * k;
                nk += k * s->S[std::size_t(q) * nI + f] / s->magS[f];
            }
            kMax = std::max(kMax, std::sqrt(kk));
            nkMax = std::max(nkMax, std::abs(nk));
            dcErr = std::max(dcErr, std::abs(s->dc[f] * h - 1));
        }
        if (!skew) {
            check(std::abs(wMin - 0.5) < 1e-14 && std::abs(wMax - 0.5) < 1e-14, "uniform cavity: w = 1/2");
            check(kMax < 1e-12, std::format("uniform cavity: k = 0 ({:.2e})", kMax));
            check(dcErr < 1e-12, std::format("uniform cavity: dcNO = 1/h ({:.2e})", dcErr));
            double bErr = 0;
            for (double d : s->bDc) bErr = std::max(bErr, std::abs(d * h / 2 - 1));
            check(bErr < 1e-12, std::format("patch deltaCoeffs = 2/h ({:.2e})", bErr));
            check(s->needRef && s->refCell >= 0 && s->srcCell[std::size_t(s->refCell)] == 0,
                  "closed cavity needs a reference; pRefCell 0 mapped");
            check(s->nB == 96 && s->emptyAxis == 2, "96 non-empty boundary faces, 2D (z empty)");
        } else {
            check(kMax > 0.1, std::format("skewed cavity: |k| up to {:.3f}", kMax));
            check(nkMax < 1e-12, std::format("k is tangential: max |n.k| {:.2e}", nkMax));
            check(wMin > 0 && wMax < 1, "weights in (0, 1)");
        }
        bool csr = true;
        for (int c = 0; c < s->nC; ++c)
            for (int e = s->cellStart[std::size_t(c)]; e < s->cellStart[std::size_t(c) + 1]; ++e) {
                const int code = s->cellEnt[std::size_t(e)];
                if (code >= 0) {
                    const int f = code >> 1;
                    const int other = (code & 1) ? s->own[std::size_t(f)] : s->nei[std::size_t(f)];
                    const int self = (code & 1) ? s->nei[std::size_t(f)] : s->own[std::size_t(f)];
                    const int E = s->cellEll[std::size_t(e)];
                    csr = csr && self == c && E >= 0 && pat.col[std::size_t(E)] == other;
                } else csr = csr && s->bOwn[std::size_t(-code - 1)] == c && s->cellEll[std::size_t(e)] == -1;
            }
        check(csr, std::string(name) + ": CSR entries name their cell, ELL positions their neighbour");
    }
}

void testMrf()
{
    std::printf("M1 MRFZoneList (mrf2)\n");
    auto c = load(data("mrf2"));
    check(bool(c), "mrf2 parses" + (c ? std::string() : ": " + c.error()));
    if (!c) return;
    check(c->mrf.size() == 3 && c->mrf[0].name == "rotA" && c->mrf[1].name == "rotB" && c->mrf[2].name == "ghost" &&
              c->mrf[0].active && c->mrf[1].active && !c->mrf[2].active,
          "three zones in file order, the third inactive (and needs no other key)");
    const double r2 = 1 / std::sqrt(2.0);
    check(c->mrf[1].omega == -1.5 && c->mrf[1].axis[0] == 0 && std::abs(c->mrf[1].axis[1] - r2) < 1e-16 &&
              std::abs(c->mrf[1].axis[2] - r2) < 1e-16,
          "rotB: `omega constant -1.5`, axis (0 1 1) normalised");
    check(c->mrf[0].nonRotatingPatches == std::vector<std::string>{"inlet"} &&
              c->mrf[1].nonRotatingPatches == std::vector<std::string>{"floor"},
          "nonRotatingPatches: literal (inlet), regex \"fl.*\" -> floor");
    auto L = loadFvCase(data("mrf2"));
    if (!L) {
        check(false, "mrf2 load: " + L.error());
        return;
    }
    const auto g = la::graphOf(L->fm);
    const auto pat = la::ellPattern(g);
    auto s = buildSimpleSetup(L->fm, L->c, g, pat);
    check(bool(s), "mrf2 setup" + (s ? std::string() : ": " + s.error()));
    if (s) {
        const auto nA = std::ranges::count(s->zone, std::uint8_t(1)), nB = std::ranges::count(s->zone, std::uint8_t(2));
        check(nA == 384 && nB == 384 && s->omega.size() == 2, std::format("zone cells {} + {} (384 each)", nA, nB));
        check(s->omega[0] == std::array<double, 3>{2, 0, 0} && s->omega[1][0] == 0 && s->omega[1][1] == -1.5 * r2,
              "Omega = omega axis per zone");
        // Every zone/zone face carries both frames: the lower zone's in frame,
        // the higher's in frame2; a face of one zone only, frame2 == 0.
        int both = 0, bad = 0;
        for (int f = 0; f < s->nI; ++f) {
            const int a = s->zone[std::size_t(s->own[std::size_t(f)])], b = s->zone[std::size_t(s->nei[std::size_t(f)])];
            const bool two = a && b && a != b;
            both += two;
            bad += !two && s->frame2[std::size_t(f)] != 0.0;
            bad += !a && !b && s->frame[std::size_t(f)] != 0.0;
        }
        check(both == 48 && bad == 0, std::format("{} zone/zone faces (48), {} misplaced frame values", both, bad));
    }
    auto base = readFoamCaseTexts(data("mrf2"));
    if (!base) return;
    struct Bad
    {
        std::string from, to;
        const char* token;
        const char* what;
        bool setup; // rejected by buildSimpleSetup (else by parseFoamCase)
    };
    const Bad bad[] = {
        {"nonRotatingPatches (inlet);", "nonRotatingPatches (inlett);", "inlett", "a literal matching no patch", false},
        {"nonRotatingPatches (\"fl.*\");", "nonRotatingPatches (\"x.*\");", "x.*", "a regex matching no patch", false},
        {"omega       2;", "omega       table ((0 1) (1 2));", "omega", "a time-dependent omega", false},
        {"cellZone    rotB;", "cellZone    rotA;", "share cellZone", "two active zones on one cellZone", true},
        {"axis        (1 0 0);", "axis        (0 0 0);", "axis", "a zero axis", false},
    };
    for (const Bad& b : bad) {
        FoamCaseTexts t = *base;
        std::string& x = t["constant/MRFProperties"];
        const auto at = x.find(b.from);
        if (at == std::string::npos) {
            check(false, std::format("{}: fixture text '{}' not found", b.what, b.from));
            continue;
        }
        x.replace(at, b.from.size(), b.to);
        std::string err;
        auto r = load(data("mrf2"), &t);
        if (!r) err = r.error();
        else if (b.setup) {
            L->c = *r;
            auto s2 = buildSimpleSetup(L->fm, L->c, g, pat);
            if (!s2) err = s2.error();
        }
        check(!err.empty() && err.find(b.token) != std::string::npos,
              std::format("{} rejected by name: {}", b.what, err.empty() ? std::string("ACCEPTED") : err));
    }
    {
        FoamCaseTexts t = *base;
        std::string zones;
        for (int i = 0; i < 9; ++i)
            zones += std::format("z{} {{ cellZone rotA; origin (0 0 0); axis (1 0 0); omega 1; }}\n", i);
        t["constant/MRFProperties"] = zones;
        auto r = load(data("mrf2"), &t);
        std::string err = r ? std::string() : r.error();
        if (r) {
            L->c = *r;
            auto s2 = buildSimpleSetup(L->fm, L->c, g, pat);
            if (!s2) err = s2.error();
        }
        check(err.find("at most 8") != std::string::npos, "9 active zones rejected: " + (err.empty() ? "ACCEPTED" : err));
    }
}

void testFoamIo()
{
    std::printf("O1 foamio round trip\n");
    const auto dir = fs::temp_directory_path() / "dns_test_fvcase_io";
    fs::remove_all(dir);
    std::vector<double> v;
    for (int i = 0; i < 30; ++i) v.push_back(std::sin(0.37 * i) * std::pow(10.0, i % 7 - 3));
    auto w = writeFoamField(dir / "5", "U", 3, false, "[0 1 -1 0 0 0 0]", v, {{"wall", {}, {1, 2, 3}}, {"e", "empty", {}}},
                            17);
    check(bool(w), "write");
    auto r = readFoamInternalField(dir / "5" / "U", 3, 10);
    check(r && *r == v, "17 significant digits round-trip every double bit for bit");
    auto bad = readFoamInternalField(dir / "5" / "U", 3, 11);
    check(!bad, "a wrong expected length is an error");
    fs::remove_all(dir);
}

// D2: `$name` macros as OpenFOAM expands them (the M17 case's two forms).
void testMacros()
{
    std::printf("D2 foamdict macros\n");
    auto d = foam::parseDict(R"X(
internalField uniform 0.184;
".*Field" 99;
_coeffs { type forceCoeffs; rhoInf 1.225; liftDir (0 0 1); sub { a 1; } }
boundaryField
{
    internalField uniform 7;
    inlet { type inletOutlet; inletValue $internalField; value ${internalField}; }
    "(a|b)" { type fixedValue; value $internalField; }
}
functions
{
    c1 { $_coeffs; patches (a b); rhoInf 2; }
    c2 { patches (b); $_coeffs; }
}
later $internalField;
internalField uniform 3;
)X",
                             "m");
    check(bool(d), "parses" + (d ? std::string() : ": " + d.error()));
    if (d) {
        const auto& bf = *d->find("boundaryField")->dict;
        const auto& in = *bf.find("inlet")->dict;
        check(foam::joined(*in.find("inletValue")) == "uniform 7" && foam::joined(*in.find("value")) == "uniform 7",
              "$name / ${name} in a value: the INNERMOST enclosing dictionary's entry (boundaryField's, not the top's)");
        check(foam::joined(*d->find("later")) == "uniform 0.184",
              "the entry as read SO FAR (a later redefinition is not seen), a pattern key never matches a value macro");
        const auto& c1 = *d->find("functions")->dict->find("c1")->dict;
        check(c1.find("type") && foam::joined(*c1.find("type")) == "forceCoeffs" && c1.find("sub") &&
                  c1.find("sub")->dict && foam::scalarOf(c1, *c1.find("rhoInf")).value_or(0) == 2 &&
                  foam::joined(*c1.find("patches")) == "( a b )",
              "`$_coeffs;` splices the dictionary's entries; a later key replaces a spliced one");
        const auto& c2 = *d->find("functions")->dict->find("c2")->dict;
        check(foam::scalarOf(c2, *c2.find("rhoInf")).value_or(0) == 1.225 && foam::joined(*c2.find("patches")) == "( b )",
              "a splice after an entry keeps both");
        check(d->find("_coeffs")->referenced && !d->find("functions")->referenced,
              "a macro's definition is marked referenced");
    }
    struct Bad
    {
        const char* text;
        const char* token;
    };
    const Bad bad[] = {
        {"a 1; b $:a;", "$:a"},
        {"a { x 1; } b { y $..a.x; }", "$..a.x"},
        {"a { x 1; } y $a.x;", "$a.x"},
        {"a 1; b ${a:-2};", "${a:-2}"},
        {"b $FOAM_CASE;", "FOAM_CASE"},
        {"b $a; a 1;", "no entry 'a'"},
        {"a { x 1; } b $a;", "names a dictionary"},
        {"a 1; b { $a; }", "is not a dictionary"},
        {"a { s { x 1; } } b { s { y 2; } $a; }", "already defined"},
        {"a { x 1; } b { $a 3; }", "must be '$a;'"},
        {"\"in.*\" { x 1; } b { y $inlet; }", "no entry 'inlet'"},
    };
    for (const Bad& b : bad) {
        auto r = foam::parseDict(b.text, "m");
        check(!r && r.error().find(b.token) != std::string::npos,
              std::format("'{}' rejected by name: {}", b.text, r ? "ACCEPTED" : r.error()));
    }
    auto pat = foam::parseDict("\"co.*\" { x 1; } b { $coeffs; }", "m");
    check(pat && pat->find("b")->dict->find("x"), "a `$name;` splice does match a pattern key (REGEX_RECURSIVE)");
}

// C3: the reader gaps of the M17 case: a `format binary` boundary / cellZones,
// macros in 0/ and controlDict, an equations factor matching p, several
// forces / forceCoeffs objects, yPlus; fvForceCoeffRow against simpleFoam.
void testM17Forms()
{
    std::printf("C3 the M17 case's forms\n");
    // (a) boundary and cellZones with a binary header (reconstructParMesh)
    const fs::path src = data("disc") / "constant" / "polyMesh";
    const fs::path tmp = fs::temp_directory_path() / "dns_test_fvcase_bin";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto ref = readPolyMesh(src);
    check(bool(ref), "disc polyMesh reads" + (ref ? std::string() : ": " + ref.error()));
    if (ref) {
        for (const char* f : {"points", "faces", "owner", "neighbour"}) fs::copy_file(src / f, tmp / f);
        const auto slurp = [](const fs::path& p) {
            std::ifstream in(p, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(in), {});
        };
        std::string b = slurp(src / "boundary");
        const std::string asc = "format      ascii;";
        const auto at = b.find(asc);
        check(at != std::string::npos, "fixture: the boundary header is ascii");
        if (at != std::string::npos) b.replace(at, asc.size(), "format      binary;");
        std::ofstream(tmp / "boundary", std::ios::binary) << b;
        // cellZones as OpenFOAM writes it in binary: text entries, raw int32 cellLabels
        std::string z = "FoamFile\n{\n    version     2.0;\n    format      binary;\n    arch        "
                        "\"LSB;label=32;scalar=64\";\n    class       regIOobject;\n    location    "
                        "\"constant/polyMesh\";\n    object      cellZones;\n    meta\n    {\n        names           ( "
                        "diskA diskB );\n    }\n}\n\n2\n(\n";
        for (std::size_t k = 0; k < ref->man.cellZones.size(); ++k) {
            const auto& cells = ref->zoneCells[k];
            z += ref->man.cellZones[k] + "\n{\n    type            cellZone;\n    cellLabels      List<label> \n" +
                 std::to_string(cells.size()) + "\n(";
            z.append(reinterpret_cast<const char*>(cells.data()), cells.size() * sizeof(std::int32_t));
            z += ");\n}\n";
        }
        z += ")\n";
        std::ofstream(tmp / "cellZones", std::ios::binary) << z;
        auto man = readPolyManifest(tmp);
        check(man && *man == ref->man, "a `format binary` boundary + cellZones: the same manifest" +
                                           (man ? std::string() : ": " + man.error()));
        auto bin = readPolyMesh(tmp);
        check(bin && *bin == *ref, "... and the same mesh, zones included" + (bin ? std::string() : ": " + bin.error()));
    }
    fs::remove_all(tmp);

    // (b) - (d) on the disc case's dictionaries
    auto c = load(data("disc"));
    check(bool(c), "disc parses" + (c ? std::string() : ": " + c.error()));
    if (!c) return;
    check(c->patches[0].uValue.v == std::vector<double>{1, 0, 0} && c->patches[1].uInit.v == std::vector<double>{1, 0, 0} &&
              c->patches[1].pValue.v == std::vector<double>{0},
          "(b) $internalField in 0/U and 0/p boundary values");
    check(c->relaxU && *c->relaxU == 0.9 && !c->relaxPGiven &&
              std::ranges::any_of(c->notes, [](const std::string& n) { return n.find("matching p is inert") != std::string::npos; }),
          "(c) equations { U 0.9; \".*\" 0.7; }: U 0.9, p unrelaxed, the inert p factor noted");
    check(c->forces.size() == 4 && c->forces[0].name == "coeffsWalls" && c->forces[0].coeffs &&
              c->forces[0].patches == std::vector<std::string>{"floor", "top"} && c->forces[1].coeffs &&
              c->forces[1].patches == std::vector<std::string>{"floor"} && c->forces[2].name == "forcesFloor" &&
              !c->forces[2].coeffs && c->forces[2].rhoInf == 1.2 && c->forces[3].name == "forcesWalls" &&
              c->forces[3].patches == std::vector<std::string>{"floor", "top"} && c->forces[3].CofR == FvVec3{0, 0, 0},
          "(d) four function objects in file order: two forceCoeffs through `$_coeffs;`, two forces");
    {
        const FvForceCoeffs& k = *c->forces[0].coeffs;
        const double dots = std::abs(k.e1[0] * k.e3[0] + k.e1[1] * k.e3[1] + k.e1[2] * k.e3[2]);
        check(dots < 1e-16 && std::abs(k.e3[0] - 0.1 / std::sqrt(1.01)) < 1e-16 && k.magUInf == 1 && k.lRef == 3 &&
                  k.Aref == 3,
              std::format("forceCoeffs axes: liftDir normalised, dragDir made orthogonal to it (e1.e3 = {:.1e})", dots));
    }
    auto base = readFoamCaseTexts(data("disc"));
    if (!base) return;
    const auto with = [&](const char* file, const std::string& from, const std::string& to) {
        FoamCaseTexts t = *base;
        std::string& s = t[file];
        const auto at = s.find(from);
        if (at == std::string::npos) return std::expected<FoamCase, std::string>(std::unexpected("fixture text '" + from + "' not found"));
        s.replace(at, from.size(), to);
        return load(data("disc"), &t);
    };
    {
        auto y = with("system/controlDict", "functions\n{", "functions\n{\n    yPlus { type yPlus; libs (fieldFunctionObjects); writeControl writeTime; }");
        check(y && y->forces.size() == 4 &&
                  std::ranges::any_of(y->notes, [](const std::string& n) { return n.find("yPlus") != std::string::npos && n.find("SKIPPED") != std::string::npos; }),
              "(d) yPlus accepted and skipped with a note" + (y ? std::string() : ": " + y.error()));
        auto f = with("system/fvSolution", "equations { U 0.9;", "fields { U 0.5; } equations { U 0.9;");
        check(!f && f.error().find("fields/U") != std::string::npos, "field relaxation of U stays rejected: " + (f ? "ACCEPTED" : f.error()));
        auto g = with("system/controlDict", "magUInf         1;", "magUInf         1;\n    coefficients (Cd Cl);");
        check(!g && g.error().find("coefficients") != std::string::npos, "forceCoeffs `coefficients` rejected by name: " + (g ? "ACCEPTED" : g.error()));
        auto h = with("system/controlDict", "dragDir         (1 0 0.05);", "dragDir         (0.2 2 0);");
        check(!h && h.error().find("co-linear") != std::string::npos, "co-linear liftDir / dragDir rejected: " + (h ? "ACCEPTED" : h.error()));
        auto u = with("system/controlDict", "runTimeModifiable false;", "runTimeModifiable false;\nunused { a 1; }");
        check(!u && u.error().find("'unused'") != std::string::npos,
              "an unknown controlDict entry no macro reads stays rejected: " + (u ? "ACCEPTED" : u.error()));
    }
    {
        // simpleFoam (np 12) on the M17 airframe, iteration 500: forcesTotal's force.dat / moment.dat rows
        // -> coeffsTotal's coefficient.dat row (same patches), its printed 9 digits.
        FvForces f;
        f.rhoInf = 1.225;
        f.CofR = {0.777, 0, 0};
        FoamCaseTexts t = *base;
        std::string& s = t["system/controlDict"];
        s.replace(s.find("liftDir         (0.1 1 0);"), 26, "liftDir         (0.052336 0 0.998630);");
        s.replace(s.find("dragDir         (1 0 0.05);"), 27, "dragDir         (-0.998630 0 0.052336);");
        for (auto [from, to] : {std::pair{std::string("rhoInf          1.2;"), std::string("rhoInf          1.225;")},
                                {std::string("magUInf         1;"), std::string("magUInf         35;")},
                                {std::string("lRef            3;"), std::string("lRef            0.754;")},
                                {std::string("Aref            3;"), std::string("Aref            0.06989;")}})
            s.replace(s.find(from), from.size(), to);
        auto m = load(data("disc"), &t);
        check(bool(m), "the M17 forceCoeffs dictionary parses" + (m ? std::string() : ": " + m.error()));
        if (m) {
            const FvVec3 fp{-6.89742922e-01, 1.37135327e-02, 8.41140140e+00}, fv{-1.29577346e+00, -2.05437249e-04, 4.20109263e-02};
            const FvVec3 mp{3.80109381e-05, 3.40374390e-02, -3.42455282e-03}, mv{-6.55354937e-05, -7.16212254e-04, 8.80229646e-05};
            const double want[12] = {4.62479668e-02,  2.31202622e-02,  2.31277046e-02,  1.59001108e-01,
                                     7.86578164e-02,  8.03432918e-02,  -8.42737682e-04, -3.72120521e-06,
                                     -8.43060175e-05, -2.57594686e-04, -2.13103360e-04, -4.44913255e-05};
            const auto row = fvForceCoeffRow(m->forces[0], fp, fv, mp, mv);
            double worst = 0;
            for (int i = 0; i < 12; ++i) worst = std::max(worst, std::abs(row[std::size_t(i)] - want[i]) / 0.159);
            check(worst < 5e-8, std::format("fvForceCoeffRow == simpleFoam's coefficient.dat row from its own force / "
                                            "moment rows: worst {:.1e} of Cl (9 printed digits)", worst));
        }
    }
}

// K1: fvOptions actuationDiskSource: the accepted spellings, rejections by
// name, OpenFOAM's findCell (ties: the lowest cell index), the zone fractions.
void testDisk()
{
    std::printf("K1 actuationDiskSource\n");
    auto c = load(data("disc"));
    if (!c) {
        check(false, "disc: " + c.error());
        return;
    }
    check(c->disks.size() == 2 && c->disks[0].name == "diskA" && c->disks[0].cellZone == "diskA" &&
              c->disks[0].Cp == 0.45 && c->disks[0].Ct == 0.5 && c->disks[0].dir == FvVec3{1, 0, 0} &&
              c->disks[0].diskArea == 0.2124 && c->disks[0].monitor == std::vector<FvVec3>{{0.3625, 0.5, 0.5}},
          "diskA: sink false flips the negative Cp / Ct, diskDir `constant (1 0 0)`, one monitor point");
    const double n = std::sqrt(4.04);
    check(c->disks[1].Cp == 0.4 && c->disks[1].Ct == 0.5 && c->disks[1].dir == FvVec3{2 / n, 0 / n, 0.2 / n} &&
              c->disks[1].monitor.size() == 2 && c->disks[1].writeToFile,
          "diskB: defaults (Froude, points, sink true), Cp `constant`, diskDir (2 0 0.2) normalised, two points");
    auto base = readFoamCaseTexts(data("disc"));
    if (!base) return;
    struct Bad
    {
        std::string from, to;
        const char* token;
    };
    const Bad bad[] = {
        {"variant         Froude;", "variant         variableScaling;", "variableScaling"},
        {"sink            false;", "sink            true;", "Cp and Ct must be greater than zero"},
        {"selectionMode   cellZone;\n    cellZone        diskA;", "selectionMode   all;", "selectionMode 'all'"},
        {"selectionMode   cellZone;\n    cellZone        diskA;", "selectionMode   cellZone;\n    cellZones (diskA);", "cellZones"},
        {"cellZone        diskA;", "cellZone        nope;", "nope"},
        {"cellZone        diskB;", "cellZone        diskA;", "second actuationDiskSource on cellZone"},
        {"monitorMethod   points;", "monitorMethod   cellSet;", "cellSet"},
        {"diskDir         constant (1 0 0);", "diskDir         table ((0 (1 0 0)) (1 (0 1 0)));", "diskDir"},
        {"diskDir         constant (1 0 0);", "diskDir         (0 0 0);", "surface-normal vector is zero"},
        {"Ct              0.5;", "Ct              table ((0 0.5) (10 0.4));", "'Ct' must be a constant"},
        {"diskArea        0.1521;", "diskArea        0;", "diskArea"},
        {"points ((0.3625 0.5 0.5));", "", "needs 'points'"},
        {"writeToFile     true;", "writeToFile     true;\n    timeStart 10; duration 5;", "timeStart"},
        {"writeToFile     true;", "writeToFile     true;\n    active no;", "inactive"},
        {"diskB\n{", "f { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; sources { U ((1 0 0) 0); } }\ndiskB\n{",
         "together with actuationDiskSource"},
    };
    for (const Bad& b : bad) {
        FoamCaseTexts t = *base;
        std::string& s = t["system/fvOptions"];
        const auto at = s.find(b.from);
        if (at == std::string::npos) {
            check(false, std::format("fixture text '{}' not found", b.from));
            continue;
        }
        s.replace(at, b.from.size(), b.to);
        auto r = load(data("disc"), &t);
        check(!r && r.error().find(b.token) != std::string::npos,
              std::format("'{:.44}' rejected by name: {}", b.to, r ? "ACCEPTED" : r.error()));
    }
    {
        FoamCaseTexts t = *base;
        std::string& s = t["system/fvOptions"];
        const std::string from = "monitorCoeffs\n    {\n        points ((0.3625 0.5 0.5));\n    }";
        const auto at = s.find(from);
        check(at != std::string::npos, "fixture: diskA's monitorCoeffs");
        if (at != std::string::npos) {
            s.replace(at, from.size(), "upstreamPoint (0.3625 0.5 0.5);");
            auto r = load(data("disc"), &t);
            check(r && r->disks[0].monitor == c->disks[0].monitor, "upstreamPoint = a one-point monitorCoeffs" +
                                                                       (r ? std::string() : ": " + r.error()));
        }
    }

    auto L = loadFvCase(data("disc"));
    if (!L) {
        check(false, "disc load: " + L.error());
        return;
    }
    // blockMesh (24 8 8): cell i + 24 (j + 8 k). The mesh is sheared in x by 0.1 y.
    const auto cell = [](int i, int j, int k) { return std::int64_t(i + 24 * (j + 8 * k)); };
    const auto find = [&](double x, double y, double z) { return foamFindCell(L->pm, L->fm, {x, y, z}).value_or(-99); };
    check(find(0.33, 0.27, 0.61) == cell(2, 2, 4) && find(0.41, 0.73, 0.33) == cell(2, 5, 2),
          std::format("interior points: cells {} {} (2,2,4) (2,5,2)", find(0.33, 0.27, 0.61), find(0.41, 0.73, 0.33)));
    check(find(0.3625, 0.5, 0.5) == cell(2, 3, 3),
          std::format("a point on the edge of cells (2, 3|4, 3|4): the lowest index {} (got {})", cell(2, 3, 3),
                      find(0.3625, 0.5, 0.5)));
    check(find(0.3625, 0.5, 0.61) == cell(2, 3, 4) && find(0.3, 0.5, 0.5) == cell(1, 3, 3),
          std::format("a point on a face: its lower cell {}; on a vertex shared by 8 cells: the lowest {}",
                      find(0.3625, 0.5, 0.61), find(0.3, 0.5, 0.5)));
    check(find(1.5, 0.5, 1.0) == cell(11, 3, 7) && find(1.5, 0.5, 1.2) == -1 && find(-0.2, 0.5, 0.5) == -1,
          "a point on the boundary is in its cell; outside the mesh: -1");
    const auto g = la::graphOf(L->fm);
    const auto pat = la::ellPattern(g);
    auto s = buildSimpleSetup(L->fm, L->c, g, pat);
    if (!s) {
        check(false, "disc setup: " + s.error());
        return;
    }
    auto ds = buildDiskSetup(L->pm, L->fm, L->c, *s);
    check(bool(ds) && s->disks.size() == 2, "buildDiskSetup" + (ds ? std::string() : ": " + ds.error()));
    if (ds && s->disks.size() == 2) {
        double sum = 0;
        for (const double f : s->disks[0].frac) sum += f;
        check(s->disks[0].cells.size() == 28 && s->disks[1].cells.size() == 18 && std::abs(sum - 1) < 1e-15 &&
                  s->disks[0].monitorSrc == std::vector<std::int64_t>{cell(2, 3, 3)} &&
                  s->disks[1].monitorSrc == std::vector<std::int64_t>{cell(2, 5, 2), cell(2, 2, 4)},
              std::format("28 + 18 zone cells, fractions sum to 1 ({:.17g}), monitor cells ascending", sum));
        check(s->disks[0].monitorHolders == std::vector<int>{4} && s->disks[1].monitorHolders == std::vector<int>{1, 1},
              "diskA's monitor point is held by 4 cells (a tie, reported), diskB's two by one each");
        FoamCase far = L->c;
        far.disks[0].monitor = {{9, 9, 9}};
        FvSimpleSetup s2 = *s;
        auto e = buildDiskSetup(L->pm, L->fm, far, s2);
        check(!e && e.error().find("No owner cell found for point (9 9 9)") != std::string::npos,
              "a monitor point outside the mesh: " + (e ? std::string("ACCEPTED") : e.error()));
    }
}

// pimpleFoam (docs/fvpimple.md): the reader's PIMPLE / Final entries, its
// rejections by name, and FoamClock against pimpleFoam's own log.
void testPimple()
{
    std::printf("P1 pimpleFoam cases resolve\n");
    auto pc = load(data("pcavity"));
    check(bool(pc), "pcavity parses" + (pc ? std::string() : ": " + pc.error()));
    if (pc) {
        const FvPimpleCtl& p = pc->pimple;
        check(p.on && p.nOuter == 1 && p.nCorr == 2 && pc->simple.nNonOrthCorr == 0 && pc->simple.pRefGiven &&
                  !pc->simple.consistent,
              "PISO mode: nOuterCorrectors 1, nCorrectors 2, pRefCell from PIMPLE");
        check(p.time.endTime == 0.2 && p.time.deltaT == 0.002 && !p.time.adjust &&
                  p.time.writeControl == FvWriteControl::TimeStep && p.time.writeInterval == 20 &&
                  p.time.timePrecision == 10,
              "time controls: endTime 0.2, deltaT 0.002, fixed, timeStep 20, timePrecision 10");
        check(!pc->relaxPGiven && !p.relaxPFinal && !pc->relaxU && !p.relaxUFinal, "no relaxation entries");
        check(p.solverUFinal.solver == "smoothSolver" && p.solverPFinal.solver == "PCG" && pc->solverP.solver == "PCG",
              "UFinal, pFinal and p (nCorrectors 2: a non-final inner iteration) read; U never looked up");
    }
    auto ps = load(data("pcavity_skew"));
    check(bool(ps), "pcavity_skew (the M17 fvSolution shape) parses" + (ps ? std::string() : ": " + ps.error()));
    if (ps) {
        const FvPimpleCtl& p = ps->pimple;
        check(p.nOuter == 3 && p.nCorr == 1 && ps->simple.nNonOrthCorr == 1 && p.turbOnFinalIterOnly,
              "nOuterCorrectors 3, nCorrectors 1, nNonOrthogonalCorrectors 1; turbOnFinalIterOnly stays yes");
        check(ps->relaxPGiven && ps->relaxP == 0.7 && p.relaxPFinal && *p.relaxPFinal == 1 && ps->relaxU &&
                  *ps->relaxU == 0.7 && p.relaxUFinal && *p.relaxUFinal == 0.7,
              "p 0.7, pFinal 1; \"U.*\" 0.7 matches U and UFinal");
        check(p.solverUFinal.solver == "smoothSolver" && p.solverUFinal.tolerance == 1e-14 &&
                  p.solverPFinal.solver == "PCG" && p.solverPFinal.maxIter == 5000 && p.solverPFinal.relTol == 0,
              "\"(U|k|omega)Final\" { $U; } and pFinal { $p; } splice the non-final entries");
        const std::string d = describeFoamCase(*ps);
        check(d.find("turbulenceOnFinalIterOnly is read by NOTHING") != std::string::npos &&
                  d.find("correctPhi is read by createDyMControls") != std::string::npos,
              "notes: turbulenceOnFinalIterOnly is not v2606's key; correctPhi is inert on a static mesh");
        check(d.find("PIMPLE      nOuterCorrectors 3") != std::string::npos && d.find("pimpleFoam: endTime 0.2") != std::string::npos,
              "describe prints the PIMPLE and time lines");
        check(ps->forces.size() == 1 && ps->forces[0].rhoInf == 1.225, "forces object");
    }
    auto ch = load(data("pchannel"));
    check(ch && ch->pimple.time.adjust && ch->pimple.time.maxCo == 2 && ch->pimple.time.maxDeltaT == 0.03 &&
              ch->pimple.time.writeControl == FvWriteControl::AdjustableRunTime && ch->pimple.time.writeInterval == 0.5 &&
              ch->pimple.nOuter == 2 && ch->pimple.nCorr == 2 && *ch->relaxU == 0.7 && *ch->pimple.relaxUFinal == 1,
          "pchannel: adjustTimeStep maxCo 2 maxDeltaT 0.03, adjustableRunTime 0.5, U 0.7 / UFinal 1" +
              (ch ? std::string() : ": " + ch.error()));
    auto sk = load(data("psst_skew"));
    check(sk && !sk->pimple.turbOnFinalIterOnly && sk->ras.relaxK && *sk->ras.relaxK == 0.7 && sk->pimple.relaxKFinal &&
              *sk->pimple.relaxKFinal == 1 && *sk->pimple.relaxOmegaFinal == 1 &&
              sk->pimple.solverOmegaFinal.solver == "PBiCGStab",
          "psst_skew: turbOnFinalIterOnly no, k 0.7 / kFinal 1, omegaFinal's solver" +
              (sk ? std::string() : ": " + sk.error()));

    std::printf("P2 pimpleFoam rejections by name\n");
    auto base = readFoamCaseTexts(data("pcavity_skew"));
    if (!base) {
        check(false, "pcavity_skew texts: " + base.error());
        return;
    }
    struct Bad
    {
        const char* file;
        std::string from, to;
        const char* token;
        const char* what;
    };
    const Bad bad[] = {
        {"system/fvSchemes", "default Euler;", "default backward;", "backward", "a ddt scheme other than Euler"},
        {"system/fvSchemes", "default Euler;", "default Euler 1;", "Euler 1", "an explicit ddtPhiCoeff"},
        {"system/fvSchemes", "default Euler;", "default Euler; ddt(U) Euler;", "ddt(U)", "a per-field ddt scheme"},
        {"system/fvSchemes", "default Euler;", "default steadyState;", "steadyState", "steadyState with pimpleFoam"},
        {"system/fvSchemes", "div(phi,U)      Gauss", "div(phi,U)      bounded Gauss", "bounded", "bounded with pimpleFoam"},
        {"system/fvSolution", "nOuterCorrectors            3;", "nOuterCorrectors            4;", "nOuterCorrectors 4",
         "an untested outer corrector count"},
        {"system/fvSolution", "nCorrectors                 1;", "nCorrectors                 3;", "nCorrectors 3",
         "an untested corrector count"},
        {"system/fvSolution", "nNonOrthogonalCorrectors    1;", "nNonOrthogonalCorrectors    3;",
         "nNonOrthogonalCorrectors 3", "an untested non-orthogonal corrector count"},
        {"system/fvSolution", "correctPhi                  yes;", "momentumPredictor no;", "momentumPredictor no",
         "momentumPredictor no"},
        {"system/fvSolution", "correctPhi                  yes;", "consistent yes;", "PIMPLEC", "consistent yes"},
        {"system/fvSolution", "correctPhi                  yes;", "ddtCorr no;", "ddtCorr no", "ddtCorr no"},
        {"system/fvSolution", "correctPhi                  yes;", "finalOnLastPimpleIterOnly yes;",
         "finalOnLastPimpleIterOnly yes", "finalOnLastPimpleIterOnly"},
        {"system/fvSolution", "correctPhi                  yes;", "residualControl { p 1e-3; }", "residualControl",
         "PIMPLE residualControl"},
        {"system/fvSolution", "correctPhi                  yes;", "moveMeshOuterCorrectors yes;", "moveMeshOuterCorrectors",
         "a dynamic-mesh control"},
        {"system/fvSolution", "correctPhi                  yes;", "SIMPLErho yes;", "SIMPLErho", "an unknown PIMPLE key"},
        {"system/fvSolution", "        p       0.7;\n", "", "pFinal", "a pFinal factor without a p factor (OpenFOAM stops)"},
        {"system/fvSolution", "        p       0.7;", "        p       0.7;\n        U 0.9;", "fields/U", "field relaxation of U"},
        {"system/fvSolution", "        p       0.7;", "        p       0.7;\n        default 0.9;", "default",
         "a default relaxation factor"},
        {"system/fvSolution", "PIMPLE\n", "PIMPLEX\n", "PIMPLE", "no PIMPLE dictionary"},
        {"system/fvSolution", "    pFinal\n    {\n        $p;\n        relTol          0;\n    }\n", "", "pFinal",
         "no pFinal solver entry"},
        {"system/controlDict", "writeControl timeStep;", "writeControl clockTime;", "clockTime", "a writeControl"},
        {"system/controlDict", "timeFormat general;", "timeFormat fixed;", "fixed", "a timeFormat"},
        {"system/controlDict", "startTime 0;", "startTime 1;", "start from time 0", "a restart"},
    };
    for (const Bad& b : bad) {
        FoamCaseTexts t = *base;
        std::string& s = t[b.file];
        const auto at = s.find(b.from);
        if (at == std::string::npos) {
            check(false, std::string(b.what) + ": pattern not in " + b.file);
            continue;
        }
        s.replace(at, b.from.size(), b.to);
        auto r = load(data("pcavity_skew"), &t);
        check(!r && r.error().find(b.token) != std::string::npos,
              std::string(b.what) + " -> " + (r ? std::string("ACCEPTED") : r.error()));
    }
    {
        FoamCaseTexts t = *base;
        t["constant/dynamicMeshDict"] = "FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }\n"
                                        "dynamicFvMesh dynamicMultiMotionSolverFvMesh;\n"
                                        "dynamicMultiMotionSolverFvMeshCoeffs { fwd { solver solidBody; cellZone r; } }\n";
        auto r = load(data("pcavity_skew"), &t);
        check(!r && r.error().find("needs 'cellZone'") == std::string::npos &&
                  r.error().find("No matching cellZones: r") != std::string::npos,
              "a multi-motion entry on a cellZone the mesh lacks -> " + (r ? std::string("ACCEPTED") : r.error()));
        t["constant/dynamicMeshDict"] = "FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }\n"
                                        "dynamicFvMesh staticFvMesh;\n";
        auto ok = load(data("pcavity_skew"), &t);
        check(bool(ok), "dynamicFvMesh staticFvMesh is the static mesh" + (ok ? std::string() : ": " + ok.error()));
        // simpleFoam reads no dynamicMeshDict (a static fvMesh): unchanged
        auto st = readFoamCaseTexts(data("cavity"));
        if (st) {
            (*st)["constant/dynamicMeshDict"] = "dynamicFvMesh dynamicMotionSolverFvMesh;\n";
            check(bool(load(data("cavity"), &*st)), "simpleFoam ignores constant/dynamicMeshDict (as OpenFOAM)");
        }
    }
    // an active MRF zone / an fvOptions source with pimpleFoam: the simpleFoam cases' files under pduct's system/
    auto sys = readFoamCaseTexts(data("pduct"));
    for (auto [name, token] : {std::pair{"mrf2", "MRF"}, {"disc", "momentum source"}}) {
        auto t = readFoamCaseTexts(data(name));
        if (!t || !sys) {
            check(false, std::string(name) + " texts");
            continue;
        }
        for (const char* f : {"system/controlDict", "system/fvSchemes", "system/fvSolution"}) (*t)[f] = (*sys)[f];
        auto r = load(data(name), &*t);
        check(!r && r.error().find(token) != std::string::npos && r.error().find("pimpleFoam") != std::string::npos,
              std::string(name) + " under pimpleFoam -> " + (r ? std::string("ACCEPTED") : r.error()));
    }

    std::printf("P3 FoamClock = OpenFOAM's Time\n");
    if (pc) {
        FoamClock k(pc->pimple.time);
        k.setInitialDeltaT(0.3);
        int n = 0, writes = 0;
        std::string firstWrite;
        while (k.running() && n < 1000) {
            k.setDeltaT(0.3);
            const bool w = k.advance();
            ++n;
            if (w && writes++ == 0) firstWrite = k.timeName();
        }
        check(n == 100 && writes == 5 && firstWrite == "0.04" && k.timeName() == "0.2" && k.deltaT() == 0.002,
              std::format("fixed deltaT: {} steps to endTime 0.2, {} write times, the first '{}', the last '{}'", n, writes,
                          firstWrite, k.timeName()));
    }
    if (ch) {
        // pimpleFoam's own log (ref/log.csv: time, the Courant number deltaT was set from, deltaT): fed its
        // Courant numbers, the clock must print its deltaT digits and stop on its last step.
        std::ifstream f(data("pchannel") / "ref" / "log.csv");
        std::string line;
        FoamClock k(ch->pimple.time);
        k.setInitialDeltaT(0.050000000000000829);
        k.setInitialDeltaT(0.050000000000000829);
        int n = 0, same = 0, writes = 0;
        double worstT = 0;
        bool ran = true;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string a, b, c;
            std::getline(ss, a, ',');
            std::getline(ss, b, ',');
            std::getline(ss, c, ',');
            ran = ran && k.running();
            k.setDeltaT(std::stod(b));
            same += k.deltaT() == std::stod(c);
            writes += k.advance();
            worstT = std::max(worstT, std::abs(k.value() - std::stod(a)) / std::stod(a));
            ++n;
        }
        check(n == 89 && same == n && ran && !k.running() && writes == 4 && worstT < 1e-9 && k.timeName() == "2",
              std::format("adjustTimeStep + adjustableRunTime: {} of {} deltaT equal pimpleFoam's printed 17 digits, {} write "
                          "times, time {:.1e}, ends at '{}'",
                          same, n, writes, worstT, k.timeName()));
    }
}

} // namespace

// Stage 3b (docs/fvpimple.md "Moving mesh"): constant/dynamicMeshDict in its
// three spellings, the PIMPLE dynamic-mesh controls and pcorr solver entries,
// the rejections by name; dns/fvmotion's point transform and swept-volume flux
// against pimpleFoam's own points / meshPhi files; FvMeshMotion's refusals.
void testMotion()
{
    std::printf("moving mesh (stage 3b)\n");
    auto a = load(data("mspin"));
    check(a && a->motion.on && !a->motion.multi && a->motion.bodies.size() == 1 && a->motion.bodies[0].cellZone.empty() &&
              a->motion.bodies[0].omega == 2 && a->motion.bodies[0].axis == FvVec3{0, 0, 1} && a->pimple.correctPhi &&
              a->pimple.checkMeshCourantNo && a->pimple.solverPcorrFinal.tolerance == 1e-13 &&
              a->patches[0].movingWall,
          "mspin: dynamicMotionSolverFvMesh, flat spelling (motionSolver solidBody, origin / axis / omega at the top), "
          "correctPhi by default, checkMeshCourantNo, pcorrFinal from \"pcorr.*\"" + (a ? std::string() : ": " + a.error()));
    auto b = load(data("mduct"));
    check(b && b->motion.on && !b->motion.multi && b->motion.bodies[0].cellZone == "rotor" &&
              b->motion.bodies[0].origin == FvVec3{0, 0.02, 0.01} && b->motion.bodies[0].omega == 4 &&
              b->schemes.gradPcorr.kind == FvGrad::CellLimited,
          "mduct: `solver solidBody; solidBodyCoeffs { cellZone rotor; ... rotatingMotionCoeffs { } }`, grad(pcorr) = the "
          "default scheme" + (b ? std::string() : ": " + b.error()));
    auto c = load(data("msst"));
    check(c && c->motion.on && c->motion.multi && c->motion.bodies.size() == 1 && c->motion.bodies[0].name == "fwd" &&
              c->motion.bodies[0].cellZone == "rotor" && c->motion.bodies[0].omega == -4,
          "msst: dynamicMultiMotionSolverFvMesh in the M17 dictionary's spelling" + (c ? std::string() : ": " + c.error()));
    if (c) {
        const std::string d = describeFoamCase(*c);
        check(d.find("dynamicMultiMotionSolverFvMesh, correctPhi yes") != std::string::npos &&
                  d.find("fwd: solidBody rotatingMotion on cellZone rotor") != std::string::npos,
              "the dry-run text states the motion and correctPhi");
    }
    // The user's M17 dynamicMeshDict, verbatim, against a manifest holding its zones.
    {
        auto t = readFoamCaseTexts(data("msst"));
        auto man = readPolyManifest(data("msst") / "constant" / "polyMesh");
        if (t && man) {
            man->cellZones = {"rotorFwd", "rotorAft"};
            (*t)["constant/dynamicMeshDict"] =
                "FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }\n"
                "dynamicFvMesh   dynamicMultiMotionSolverFvMesh;\n"
                "dynamicMultiMotionSolverFvMeshCoeffs\n{\n"
                "    fwd { solver solidBody; cellZone rotorFwd; solidBodyCoeffs { solidBodyMotionFunction rotatingMotion; "
                "rotatingMotionCoeffs { origin (0 0 0); axis (1 0 0); omega 837.758; } } }\n"
                "    aft { solver solidBody; cellZone rotorAft; solidBodyCoeffs { solidBodyMotionFunction rotatingMotion; "
                "rotatingMotionCoeffs { origin (0 0 0); axis (1 0 0); omega -837.758; } } }\n}\n";
            auto m = parseFoamCase(*t, *man);
            check(m && m->motion.bodies.size() == 2 && m->motion.bodies[0].cellZone == "rotorFwd" &&
                      m->motion.bodies[1].name == "aft" && m->motion.bodies[1].omega == -837.758,
                  "the M17 dynamicMeshDict (two solidBody / rotatingMotion entries) parses" +
                      (m ? std::string() : ": " + m.error()));
        } else check(false, "msst texts");
    }
    // rejections by name
    auto base = readFoamCaseTexts(data("mduct"));
    if (!base) {
        check(false, base.error());
        return;
    }
    struct Bad
    {
        const char* file;
        std::string from, to;
        const char* token;
        const char* what;
    };
    const Bad bad[] = {
        {"constant/dynamicMeshDict", "dynamicMotionSolverFvMesh", "dynamicRefineFvMesh", "dynamicRefineFvMesh",
         "another dynamicFvMesh"},
        {"constant/dynamicMeshDict", "solver          solidBody;", "solver          displacementLaplacian;",
         "displacementLaplacian", "another motion solver"},
        {"constant/dynamicMeshDict", "solidBodyMotionFunction rotatingMotion;", "solidBodyMotionFunction linearMotion;",
         "linearMotion", "another solidBodyMotionFunction"},
        {"constant/dynamicMeshDict", "omega       4;", "omega       table ((0 0) (1 4));", "Function1", "a Function1 omega"},
        {"constant/dynamicMeshDict", "cellZone        rotor;", "cellSet         rotor;", "cellSet", "a cellSet selection"},
        {"constant/dynamicMeshDict", "cellZone        rotor;", "cellZone        stator;", "No matching cellZones: stator",
         "a cellZone the mesh lacks"},
        {"constant/dynamicMeshDict", "solver          solidBody;", "solver          solidBody;\nupdateControl timeStep;",
         "updateControl", "a mesh update control"},
        {"constant/dynamicMeshDict", "axis        (1 0 0);", "axis        (0 0 0);", "axis must be non-zero", "a zero axis"},
        {"constant/dynamicMeshDict", "omega       4;", "omega       4;\n        amplitude 1;", "amplitude",
         "an unknown rotatingMotionCoeffs key"},
        {"system/fvSolution", "    \"pcorr.*\"\n", "    \"pcorX.*\"\n", "pcorrFinal", "no pcorr solver entry (correctPhi)"},
        {"system/fvSolution", "correctPhi                  yes;", "moveMeshOuterCorrectors yes;", "moveMeshOuterCorrectors",
         "moveMeshOuterCorrectors yes"},
    };
    for (const Bad& x : bad) {
        FoamCaseTexts t = *base;
        std::string& s = t[x.file];
        const auto at = s.find(x.from);
        if (at == std::string::npos) {
            check(false, std::string(x.what) + ": pattern not in " + x.file);
            continue;
        }
        s.replace(at, x.from.size(), x.to);
        auto r = load(data("mduct"), &t);
        check(!r && r.error().find(x.token) != std::string::npos,
              std::string(x.what) + " -> " + (r ? std::string("ACCEPTED") : r.error()));
    }
    {
        FoamCaseTexts t = *base; // correctPhi no: no pcorr entry needed
        std::string& s = t["system/fvSolution"];
        s.replace(s.find("correctPhi                  yes;"), 31, "correctPhi                  no;");
        s.replace(s.find("    \"pcorr.*\"\n"), 14, "    \"pcorX.*\"\n");
        auto r = load(data("mduct"), &t);
        check(r && !r->pimple.correctPhi, "correctPhi no reads no pcorr solver entry" + (r ? std::string() : ": " + r.error()));
    }
    {
        auto t = readFoamCaseTexts(data("pduct")); // cyclic patches + motion
        auto flat = readFoamCaseTexts(data("mspin"));
        if (t && flat) {
            (*t)["constant/dynamicMeshDict"] = (*flat)["constant/dynamicMeshDict"];
            auto r = load(data("pduct"), &*t);
            check(!r && r.error().find("cyclic patches") != std::string::npos,
                  "mesh motion on a mesh with cyclic patches -> " + (r ? std::string("ACCEPTED") : r.error()));
        }
    }
    // dns/fvmotion: the point transform and meshPhi against pimpleFoam's files at t = 1 (step 200 of mspin)
    {
        auto L = loadFvCase(data("mspin"));
        if (!L) {
            check(false, L.error());
            return;
        }
        auto M = FvMeshMotion::make(*L);
        check(bool(M) && M->movedPoints() == std::int64_t(L->pm.px.size()),
              "mspin: every point moves" + (M ? std::string() : ": " + M.error()));
        if (!M) return;
        FvGraph G{la::graphOf(L->fm), {}, {}};
        G.pat = la::ellPattern(G.g);
        auto s = buildSimpleSetup(L->fm, L->c, G.g, G.pat);
        if (!s) {
            check(false, s.error());
            return;
        }
        const std::vector<double> V0 = s->V;
        // OpenFOAM's time is the running sum of deltaT: reproduce the last two steps' times
        double t = 0;
        for (int i = 0; i < 199; ++i) t += 0.005;
        auto m1 = M->update(t, 0.005, *L, G, *s);
        t += 0.005;
        auto m2 = m1 ? M->update(t, 0.005, *L, G, *s) : m1;
        if (!m2) {
            check(false, m2.error());
            return;
        }
        // points: bitwise OpenFOAM's (17 significant digits round-trip)
        std::ifstream f(data("mspin") / "ref" / "1" / "polyMesh" / "points");
        std::stringstream ss;
        ss << f.rdbuf();
        std::string txt = ss.str();
        const std::size_t open = txt.find('(', txt.find('}'));
        for (char& ch : txt)
            if (ch == '(' || ch == ')') ch = ' ';
        std::stringstream v(open == std::string::npos ? std::string() : txt.substr(open));
        std::size_t n = 0, differ = 0;
        double x = 0, y = 0, z = 0;
        while (n < L->pm.px.size() && (v >> x >> y >> z)) {
            differ += x != L->pm.px[n] || y != L->pm.py[n] || z != L->pm.pz[n];
            ++n;
        }
        check(n == L->pm.px.size() && differ == 0,
              std::format("mspin t = 1: {} of {} points differ from pimpleFoam's by a bit (transformPoints of the "
                          "septernion, from the initial points)", differ, n));
        auto ref = readFoamInternalField(data("mspin") / "ref" / "1" / "meshPhi", 1, L->pm.man.nInternalFaces);
        if (ref) {
            double w = 0, sc = 0;
            for (std::size_t q = 0; q < std::size_t(s->nI); ++q) {
                const double sg = s->srcCell[std::size_t(s->own[q])] < s->srcCell[std::size_t(s->nei[q])] ? 1.0 : -1.0;
                const double th = (*ref)[std::size_t(s->internalSrc[q])];
                w = std::max(w, std::abs(sg * m2->meshPhi[q] - th));
                sc = std::max(sc, std::abs(th));
            }
            check(w == 0.0, std::format("mspin t = 1: meshPhi (face::sweptVol / deltaT) vs pimpleFoam's: max {:.2e} of {:.2e} "
                                        "(bitwise)", w, sc));
        } else check(false, ref.error());
        // geometric conservation: sum of a cell's mesh fluxes x deltaT = V - V0, to rounding
        double gcl = 0, vmax = 0;
        std::vector<double> sum(std::size_t(s->nC), 0.0);
        for (std::size_t q = 0; q < std::size_t(s->nI); ++q) {
            sum[std::size_t(s->own[q])] += m2->meshPhi[q];
            sum[std::size_t(s->nei[q])] -= m2->meshPhi[q];
        }
        for (std::size_t q = 0; q < std::size_t(s->nB); ++q) sum[std::size_t(s->bOwn[q])] += m2->meshPhiB[q];
        // (2D: the empty faces sweep nothing in a rotation about the empty axis)
        for (std::size_t cc = 0; cc < std::size_t(s->nC); ++cc) {
            gcl = std::max(gcl, std::abs(sum[cc] * 0.005 - (s->V[cc] - V0[cc])));
            vmax = std::max(vmax, s->V[cc]);
        }
        check(gcl < 1e-13 * vmax, std::format("geometric conservation: |sum meshPhi dt - (V - V0)| <= {:.2e} of V", gcl / vmax));
        // the wall velocity: Uwall . n = meshPhi_b / |S| and |Uwall| ~ omega r
        double wn = 0, wr = 0;
        const std::size_t nB = std::size_t(s->nB);
        for (std::size_t q = 0; q < nB; ++q) {
            const double un = m2->uWall[q] * s->bN[q] + m2->uWall[nB + q] * s->bN[nB + q] + m2->uWall[2 * nB + q] * s->bN[2 * nB + q];
            wn = std::max(wn, std::abs(un - m2->meshPhiB[q] / s->bMagS[q]));
            const double r = std::hypot(s->bCf[q], s->bCf[nB + q]);
            const double um = std::sqrt(m2->uWall[q] * m2->uWall[q] + m2->uWall[nB + q] * m2->uWall[nB + q]);
            wr = std::max(wr, std::abs(um / (2.0 * r) - 1.0));
        }
        // Uwall is first order in omega dt: the tangential part is the face centre's chord (direction at
        // mid-step), the normal part the swept volume on the NEW normal -- OpenFOAM's expression, kept.
        check(wn < 1e-15 && wr < 0.5 * 2.0 * 0.005,
              std::format("movingWallVelocity: Uwall.n - meshPhi_b/|S| {:.2e}; |Uwall| / (omega r) - 1 {:.2e} <= omega dt / 2 "
                          "= 5e-3", wn, wr));
    }
    // FvMeshMotion's refusals: a moving zone bordering other cells through ordinary faces
    {
        auto L = loadFvCase(data("mrf2")); // blocks rotA | rotB | (none), adjacent
        if (!L) {
            check(false, L.error());
            return;
        }
        L->c.motion.on = true;
        L->c.motion.bodies = {FvSolidBody{"", "rotA", {0, 0, 0}, {1, 0, 0}, 1.0}};
        auto M = FvMeshMotion::make(*L);
        check(!M && M.error().find("cellZone 'rotA'") != std::string::npos && M.error().find("cyclicAMI") != std::string::npos &&
                  M.error().find("non-moving") != std::string::npos,
              "a moving cellZone next to static cells -> " + (M ? std::string("ACCEPTED") : M.error()));
        L->c.motion.multi = true;
        L->c.motion.bodies = {FvSolidBody{"a", "rotA", {0, 0, 0}, {1, 0, 0}, 1.0}, FvSolidBody{"b", "rotB", {0, 0, 0}, {1, 0, 0}, -1.0}};
        auto M2 = FvMeshMotion::make(*L);
        check(!M2 && M2.error().find("cyclicAMI") != std::string::npos,
              "two moving cellZones sharing faces -> " + (M2 ? std::string("ACCEPTED") : M2.error()));
        L->c.motion.bodies = {FvSolidBody{"", "", {0, 0, 0}, {1, 0, 0}, 1.0}};
        check(bool(FvMeshMotion::make(*L)), "the whole mesh as one body is accepted");
    }
    // cyclicAMI (stage 3c): the reader, the graph with the couplings, the setup, the rejections
    {
        auto L = loadFvCase(data("aplanar"));
        if (!L) {
            check(false, L.error());
            return;
        }
        check(L->c.ami && L->c.patches[3].u == FvUBc::Ami && L->c.patches[4].p == FvPBc::Ami,
              "aplanar: AMI1 / AMI2 read as cyclicAMI on U and p");
        auto G = buildFvGraph(L->pm, L->fm);
        check(bool(G) && G->ami.size() == 1 && G->ami[0].src == 3 && G->ami[0].tgt == 4,
              "one cyclicAMI pair, owner AMI1" + (G ? std::string() : ": " + G.error()));
        if (!G) return;
        const la::CellGraph plain = la::graphOf(L->fm);
        auto s = buildSimpleSetup(L->fm, L->c, G->g, G->pat, &G->ami);
        check(bool(s) && s->ami.nA == 20 && s->ami.rCell.size() == 20, "8 + 12 coupled faces, one per cell");
        if (!s) return;
        bool sym = true, sums = true, wts = true;
        for (int a = 0; a < s->ami.nA; ++a) {
            const int P = s->bOwn[std::size_t(s->ami.aB[std::size_t(a)])];
            double w = 0;
            for (int j = s->ami.aStart[std::size_t(a)]; j < s->ami.aStart[std::size_t(a) + 1]; ++j) {
                const int N = s->ami.aCell[std::size_t(j)];
                sym = sym && la::graphSlot(G->g, N, P) >= 0 && la::graphSlot(plain, P, N) < 0;
                w += s->ami.aW[std::size_t(j)];
            }
            sums = sums && std::abs(w - 1) < 1e-14;
            wts = wts && s->ami.wt[std::size_t(a)] > 0.3 && s->ami.wt[std::size_t(a)] < 0.7;
        }
        check(sym, "every coupling is in the graph both ways and not in the mesh's own graph");
        check(sums && wts, "the normalised weights of every face sum to 1; patch weights near 1/2");
        check(G->g.col.size() == plain.col.size() + s->ami.aCell.size(), "the graph gains one entry per address");
        check(!buildSimpleSetup(L->fm, L->c, G->g, G->pat), "a cyclicAMI mesh without the AMI is refused");
        auto t = readFoamCaseTexts(data("aplanar"));
        if (!t) {
            check(false, t.error());
            return;
        }
        auto swap = [&](const char* file, const std::string& a, const std::string& b, const char* token, const char* what) {
            FoamCaseTexts x = *t;
            std::string& txt = x[file];
            const auto at = txt.find(a);
            if (at == std::string::npos) {
                check(false, std::string("(fixture) ") + what);
                return;
            }
            txt.replace(at, a.size(), b);
            auto c = parseFoamCase(x, L->pm.man);
            check(!c && c.error().find(token) != std::string::npos, std::string(what) + " -> " + (c ? "ACCEPTED" : c.error()));
        };
        swap("system/controlDict", "application pimpleFoam;", "application simpleFoam;", "cyclicAMI patches with simpleFoam",
             "cyclicAMI under simpleFoam");
        swap("system/fvSchemes", "Gauss linear limited corrected 0.33", "Gauss linear uncorrected", "cyclicAMI needs corrected",
             "cyclicAMI with an uncorrected laplacian");
        swap("0/U", "#includeEtc \"caseDicts/setConstraintTypes\"", "AMI1 { type zeroGradient; }", "cyclicAMI",
             "a non-cyclicAMI field type on a cyclicAMI patch");
    }
    {
        auto L = loadFvCase(data("amix2d"));
        if (!L) {
            check(false, L.error());
            return;
        }
        auto M = FvMeshMotion::make(*L);
        check(bool(M) && M->movedPoints() > 0 && M->movedPoints() < std::int64_t(L->pm.px.size()),
              "amix2d: the rotor zone behind cyclicAMI moves, the stator does not" + (M ? std::string() : ": " + M.error()));
        auto G = buildFvGraph(L->pm, L->fm);
        if (!M || !G) return;
        auto s = buildSimpleSetup(L->fm, L->c, G->g, G->pat, &G->ami);
        if (!s) {
            check(false, s.error());
            return;
        }
        const std::size_t e0 = s->ami.aCell.size();
        const std::vector<int> c0 = s->ami.aCell;
        auto mv = M->update(0.1, 0.1, *L, *G, *s);
        check(bool(mv) && (s->ami.aCell != c0 || s->ami.aCell.size() != e0),
              "after 0.4 rad the AMI addresses (the matrix pattern) changed" + (mv ? std::string() : ": " + mv.error()));
        if (!mv) return;
        double stat = 0, rot = 0;
        for (int a = 0; a < s->ami.nA; ++a) {
            const std::size_t b = std::size_t(s->ami.aB[std::size_t(a)]);
            (L->c.patches[std::size_t(s->bPatch[b])].name == "AMI1" ? rot : stat) =
                std::max(L->c.patches[std::size_t(s->bPatch[b])].name == "AMI1" ? rot : stat, std::abs(mv->meshPhiB[b]));
        }
        check(stat == 0.0 && rot > 0, std::format("meshPhi on the interface: rotor side up to {:.3e}, stator side exactly 0", rot));
    }
    {
        auto L = loadFvCase(data("mspin")); // 2D: the axis must be the empty one
        if (L) {
            L->c.motion.bodies[0].axis = {1, 0, 0};
            auto M = FvMeshMotion::make(*L);
            check(!M && M.error().find("empty axis") != std::string::npos,
                  "a 2D mesh rotating out of its plane -> " + (M ? std::string("ACCEPTED") : M.error()));
        }
    }
}

int main()
{
    testMotion();
    testPimple();
    testDict();
    testMacros();
    testM17Forms();
    testDisk();
    testCases();
    testSetup();
    testMrf();
    testRas();
    testFoamIo();
    return dnstest::finish();
}
