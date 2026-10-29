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
#include "dns/foamdict.hpp"
#include "dns/foamio.hpp"
#include "dns/fvcase.hpp"
#include "dns/fvsimple.hpp"
#include "dns/la/graph.hpp"

#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
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
        {"system/controlDict", "application simpleFoam;", "application pimpleFoam;", "pimpleFoam", "pimpleFoam"},
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
               "sources { U ((1 0 0) 0); } }";
        auto r = load(data("cavity"), &t);
        check(!r && r.error().find(f) != std::string::npos,
              std::format("{} (simpleFoam applies it) rejected by name: {}", f, r ? "ACCEPTED" : r.error()));
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
        check(r3 && r3->forces && r3->forces->CofR == FvVec3{1, 2, 3},
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

} // namespace

int main()
{
    testDict();
    testCases();
    testSetup();
    testMrf();
    testRas();
    testFoamIo();
    return dnstest::finish();
}
