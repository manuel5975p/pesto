// Species LES and the thickened flame on the compressible FV family (docs/fvcomp.md
// "Species LES and TFLES"), against the FP64 oracle and the structured kernels.
//   L1 double vs FvRef: LES (eddy species flux), TFLES (uniform; dynamic sensor;
//      dynamic + LES + Charlette): rhs, mean and species fluxes, mu_t, omega, F E,
//      E/F, Omega, 3 steps, stableDt <= 1e-13 on box_cyclic, box_refined,
//      cyl2d_snappy, a 3D periodic box; float vs double
//   L2 off = bit for bit: --tfles 1 (uniform, dynamic) == the DNS kernels; a passive
//      species under LES leaves the LES mean flow bit for bit; uniform power-of-two Y
//      exact under the eddy and thickened fluxes (FP32, compensated, FP64)
//   L3 conservation under a thickened burning march: mass, E + q rho Y, passive
//   L4 rerun byte-identical
//   L5 the face walk IS the structured filter on a uniform box: F E, Omega of the
//      FV path vs DeviceTfles on one state (open x, periodic y z)
//   X1 FV vs the structured LES / TFLES flame on a uniform periodic 3D box, same dt:
//      rounding, <= 3 x the FP32 run's own distance from FP64
//   tfles1d (`test_gpu_fv_flame_les tfles1d`): the binary route, F 2 laminar flame
#include "fv_comp_test_util.hpp"

#include "dns/fvcomp_case.hpp"
#include "dns/chem.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/tfles.hpp"
#include "gpu_test_util.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <unistd.h>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dnstest::fvc;
using dnstest::check;

namespace {

struct Case
{
    const char* name;
    FaceMesh fm;
    std::vector<FvBc> bc;
    Transport tr;
};

// A hot, non-stiff premixed step (test_gpu_fv_flame's kChem): omega ~ 0.6 rho Y.
constexpr FvChem kChem{.nY = 2, .fuel = 0, .lnA = 1.6, .Ta = 2.0, .q = 0.3, .invPr = 1.0 / 0.72};
constexpr double kNoRate = -std::numeric_limits<double>::infinity();

// A sensor that varies across the smooth states (rate at TaS 1, tanh 2), radius 3.
TflesSetup testTfles(bool uniform, bool charlette, double F = 3.0)
{
    return {.F = F,
            .uniform = uniform,
            .charlette = charlette,
            .deltaL = 0.2,
            .nuU = 0.05,
            .lnAs = 0.5,
            .TaS = 1.0,
            .sensorBeta = 2.0,
            .dilate = 3};
}

std::vector<Case> cases()
{
    std::vector<Case> v;
    {
        FaceMesh fm = loadMesh("box_cyclic");
        v.push_back({"box_cyclic (adiabatic walls), mu 0.05", std::move(fm), {}, Transport::constant(kGamma, 0.05)});
        v.back().bc = bcFor(v.back().fm, [](const MeshPatch&) { return FvBc{}; });
    }
    v.push_back({"box_refined (cyclic, split faces), Sutherland", loadMesh("box_refined"), {},
                 Transport::air(0.02, 0.9, 288.15, 0.72, kGamma, 1.0, true)});
    v.back().bc = bcFor(v.back().fm, [](const MeshPatch&) { return FvBc{}; });
    {
        FaceMesh fm = loadMesh("cyl2d_snappy");
        auto pick = [&](const MeshPatch& p) {
            FvBc b;
            if (p.name == "cylinder") {
                b.kind = FvBcKind::Wall;
            } else if (p.name == "inlet") {
                b.kind = FvBcKind::Inflow;
                b.uIn[0] = 0.5;
                b.tIn = 1.0;
            } else if (p.name == "outlet") {
                b.kind = FvBcKind::Outflow;
                b.pOut = 1.0;
                b.tBack = 1.0;
            } else if (p.name == "lower") {
                b.kind = FvBcKind::Slip;
            } else {
                b.kind = FvBcKind::Far;
                b.qFar[0] = 1.0;
                b.qFar[1] = 0.5;
                b.qFar[4] = 1.0 / (kGamma - 1.0) + 0.125;
            }
            b.y[0] = 0.7;
            b.y[1] = 0.3;
            return b;
        };
        v.push_back({"cyl2d_snappy (wall, inflow, outflow, slip, far), mu 0.01", std::move(fm), {},
                     Transport::constant(kGamma, 0.01)});
        v.back().bc = bcFor(v.back().fm, pick);
    }
    {
        const Grid g = Grid::make(12, 10, 8, 1.2, 1.0, 0.8);
        FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "box3d");
        v.push_back({"3D periodic box 12 x 10 x 8, mu 0.02", std::move(fm), {}, Transport::constant(kGamma, 0.02)});
        v.back().bc = bcFor(v.back().fm, [](const MeshPatch&) { return FvBc{}; });
    }
    return v;
}

// smoothState + a tanh front in Y0 (the limiter's clipped branches, the sensor's
// gradient) and a smooth Y1, a T rising across the front (the gate's order).
FvState speciesState(const FaceMesh& fm, bool uz)
{
    const FvState base = smoothState(fm, uz, 0.6);
    FvState s = FvState::zeros(fm.nC, kFvMeanVars + 2);
    std::copy(base.q.begin(), base.q.end(), s.q.begin());
    const auto [lo, hi] = std::ranges::minmax(fm.C[0]);
    const auto [ly, hy] = std::ranges::minmax(fm.C[1]);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double X = (fm.C[0][c] - lo) / (hi - lo), Y = (fm.C[1][c] - ly) / (hy - ly);
        const double y0 = 0.5 * (1.0 - std::tanh((X - 0.5 - 0.1 * std::sin(6 * Y)) / 0.05));
        const double y1 = 0.3 + 0.2 * std::cos(5 * X + 2 * Y);
        s.var(5)[c] = s.var(0)[c] * y0;
        s.var(6)[c] = s.var(0)[c] * y1;
        s.var(4)[c] += 0.8 * (1.0 - y0) / (kGamma - 1.0); // hot where burnt
    }
    return s;
}

double vecDiff(const std::vector<double>& a, const std::vector<double>& b)
{
    double e = 0.0, s = 0.0;
    for (std::size_t c = 0; c < b.size(); ++c) {
        e = std::max(e, std::abs(a[c] - b[c]));
        s = std::max(s, std::abs(b[c]));
    }
    return s == 0.0 ? e : e / s;
}

double vecsDiff(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b)
{
    double w = 0.0;
    for (std::size_t k = 0; k < b.size(); ++k) w = std::max(w, vecDiff(a[k], b[k]));
    return w;
}

struct Variant
{
    const char* tag;
    double cs;
    SgsModel sgs;
    std::optional<TflesSetup> tf;
};

std::vector<Variant> variants(bool threeD)
{
    const SgsModel les = threeD ? SgsModel::Sigma : SgsModel::Smagorinsky;
    const double cs = threeD ? 1.35 : 0.3;
    return {{"LES (eddy species flux)", cs, les, std::nullopt},
            {"TFLES uniform", 0.0, les, testTfles(true, false)},
            {"TFLES dynamic", 0.0, les, testTfles(false, false)},
            {"LES + TFLES dynamic + Charlette", cs, les, testTfles(false, true)}};
}

void gateOracle()
{
    std::printf("== L1 double vs FvRef, float vs double\n");
    for (Case& m : cases()) {
        const bool threeD = m.fm.dim == 3;
        for (const Variant& va : variants(threeD)) {
            const FvRef ref = orDie(FvRef::make(m.fm, {.tr = m.tr,
                                                       .bc = m.bc,
                                                       .chem = kChem,
                                                       .les = {.cs = va.cs, .sgs = va.sgs, .prt = 0.9},
                                                       .tfles = va.tf}),
                                    m.name);
            auto cfg = [&](FvPrecision pr) {
                return FvCompConfig{.tr = m.tr, .bc = m.bc, .precision = pr, .csLes = va.cs, .sgs = va.sgs, .chem = kChem,
                                    .tfles = va.tf};
            };
            DeviceFvComp dev = orDie(DeviceFvComp::make(m.fm, cfg(FvPrecision::Fp64)), m.name);
            DeviceFvComp flt = orDie(DeviceFvComp::make(m.fm, cfg(FvPrecision::Fp32)), m.name);
            FvState s = speciesState(m.fm, threeD);
            orDie(dev.upload(s), "upload");
            orDie(flt.upload(s), "upload");
            FvEval eR, eD;
            FvState rR, rD, rF;
            ref.evaluate(s, eR, rR);
            orDie(dev.evaluate(eD, rD), "evaluate");
            orDie(flt.rhs(rF), "rhs");
            const int nV = ref.nV(), nC = m.fm.nC;
            const double dR = relDiff(rD.q, rR.q, nV, nC), dO = vecDiff(eD.omega, eR.omega),
                         dFy = vecsDiff(eD.Fy, eR.Fy),
                         dF = vecsDiff({eD.F[0], eD.F[1], eD.F[2], eD.F[3], eD.F[4]}, {eR.F[0], eR.F[1], eR.F[2], eR.F[3], eR.F[4]});
            double dNut = 0.0, dFe = 0.0, dEof = 0.0, dOm = 0.0;
            if (va.cs > 0) dNut = vecDiff(eD.nut, eR.nut);
            if (va.tf) {
                dFe = vecDiff(eD.fe, eR.fe);
                dEof = vecDiff(eD.eof, eR.eof);
                dOm = vecDiff(eD.om, eR.om);
            }
            const double dtR = orDie(ref.stableDt(s), "dt"), dtD = orDie(dev.stableDt(), "dt");
            const double dt = 0.25 * dtR;
            FvState q = s, qD, qF;
            ref.march(q, dt, 3);
            for (int n = 0; n < 3; ++n) {
                dev.step(dt);
                flt.step(dt);
            }
            orDie(dev.download(qD), "download");
            orDie(flt.download(qF), "download");
            const double dS = relDiff(qD.q, q.q, nV, nC), fR = relDiff(rF.q, rD.q, nV, nC), fS = relDiff(qF.q, qD.q, nV, nC);
            std::printf("  %s, %s\n", m.name, va.tag);
            double feMin = HUGE_VAL, feMax = 0.0, omMin = HUGE_VAL, omMax = 0.0;
            for (std::size_t c = 0; va.tf && c < eR.fe.size(); ++c) {
                feMin = std::min(feMin, eR.fe[c]);
                feMax = std::max(feMax, eR.fe[c]);
                omMin = std::min(omMin, eR.om[c]);
                omMax = std::max(omMax, eR.om[c]);
            }
            double eMax = 0.0; // E = sqrt(F E * E/F)
            for (std::size_t c = 0; va.tf && c < eR.fe.size(); ++c) eMax = std::max(eMax, std::sqrt(eR.fe[c] * eR.eof[c]));
            if (va.tf)
                std::printf("    F E in [%.4f, %.4f], Omega in [%.4f, %.4f], max E %.6f\n", feMin, feMax, omMin, omMax, eMax);
            check(dR <= 1e-13, std::format("rhs {:.2e} <= 1e-13", dR));
            check(dF <= 1e-13, std::format("mean fluxes (thickened energy included) {:.2e} <= 1e-13", dF));
            check(dFy <= 1e-13, std::format("species fluxes {:.2e} <= 1e-13", dFy));
            check(dO <= 1e-13, std::format("omega {:.2e} <= 1e-13", dO));
            if (va.cs > 0) check(dNut <= 1e-13, std::format("mu_t {:.2e} <= 1e-13", dNut));
            if (va.tf) {
                check(dFe <= 1e-13 && dEof <= 1e-13 && dOm <= 1e-13,
                      std::format("F E {:.2e}, E/F {:.2e}, Omega {:.2e} <= 1e-13", dFe, dEof, dOm));
                if (!va.tf->uniform) check(feMax > feMin, "F E varies (the sensor is not trivial)");
            }
            check(std::abs(dtD / dtR - 1.0) <= 1e-13, std::format("stable dt {:.2e} <= 1e-13", std::abs(dtD / dtR - 1.0)));
            check(dS <= 1e-13, std::format("3 steps {:.2e} <= 1e-13", dS));
            std::printf("    float vs double: rhs %.2e, 3 steps %.2e\n", fR, fS);
            check(fR <= 2e-5, std::format("float rhs {:.2e} <= 2e-5", fR));
            check(fS <= 2e-6, std::format("float 3 steps {:.2e} <= 2e-6", fS));
        }
    }
}

template <class R>
bool exactRatio(const FvState& s, int row, double y)
{
    for (std::size_t c = 0; c < std::size_t(s.nC); ++c)
        if (R(s.var(row)[c]) != R(y) * R(s.var(0)[c])) return false;
    return true;
}

bool sameBytes(const FvState& a, const FvState& b)
{
    return a.q.size() == b.q.size() && std::memcmp(a.q.data(), b.q.data(), a.q.size() * sizeof(double)) == 0;
}

void gateBits()
{
    std::printf("== L2 off = bit for bit; uniform Y exact under the eddy and thickened fluxes\n");
    struct Mode
    {
        const char* tag;
        FvPrecision pr;
        bool comp;
    };
    const Mode modes[] = {{"FP32", FvPrecision::Fp32, false}, {"FP32 compensated", FvPrecision::Fp32, true},
                          {"FP64", FvPrecision::Fp64, false}};
    for (Case& m : cases()) {
        const bool threeD = m.fm.dim == 3;
        const FvState s = speciesState(m.fm, threeD);
        const double dt = 0.25 * orDie(orDie(FvRef::make(m.fm, {.tr = m.tr, .bc = m.bc, .chem = kChem}), m.name).stableDt(s), "dt");
        auto run = [&](const FvCompConfig& c, const FvState& ic, int steps) {
            DeviceFvComp d = orDie(DeviceFvComp::make(m.fm, c), m.name);
            orDie(d.upload(ic), "upload");
            for (int n = 0; n < steps; ++n) d.step(dt);
            FvState o;
            orDie(d.download(o), "download");
            return o;
        };
        for (const Mode md : modes) {
            const FvCompConfig dns{.tr = m.tr, .bc = m.bc, .precision = md.pr, .compensated = md.comp, .chem = kChem};
            const FvState base = run(dns, s, 40);
            for (const bool uni : {true, false}) {
                FvCompConfig one = dns;
                one.tfles = testTfles(uni, false, 1.0);
                check(sameBytes(run(one, s, 40), base),
                      std::format("{}, {}: --tfles 1 ({}) == the DNS kernels byte for byte, 40 steps", m.name, md.tag,
                                  uni ? "uniform" : "dynamic"));
            }
            // A passive species under LES: the LES mean flow bit for bit.
            const double cs = threeD ? 1.35 : 0.3;
            const SgsModel sg = threeD ? SgsModel::Sigma : SgsModel::Smagorinsky;
            FvChem passive = kChem;
            passive.lnA = kNoRate;
            passive.q = 0.0;
            FvState mean = FvState::zeros(m.fm.nC);
            std::copy_n(s.q.begin(), mean.q.size(), mean.q.begin());
            const FvState a = run({.tr = m.tr, .bc = m.bc, .precision = md.pr, .csLes = cs, .sgs = sg, .compensated = md.comp,
                                   .chem = passive},
                                  s, 40);
            const FvState b = run({.tr = m.tr, .bc = m.bc, .precision = md.pr, .csLes = cs, .sgs = sg, .compensated = md.comp},
                                  mean, 40);
            check(std::memcmp(a.q.data(), b.q.data(), b.q.size() * sizeof(double)) == 0,
                  std::format("{}, {}: passive species under LES: mean rows == nY 0 bitwise", m.name, md.tag));
            // Uniform power-of-two Y (open faces carry the same Y in): rho Y == Y rho
            // through the eddy and thickened fluxes.
            FvState u = s;
            for (std::size_t c = 0; c < std::size_t(m.fm.nC); ++c) {
                u.var(5)[c] = 0.25 * u.var(0)[c];
                u.var(6)[c] = 0.5 * u.var(0)[c];
            }
            std::vector<FvBc> ubc = m.bc;
            for (FvBc& b : ubc) {
                b.y[0] = 0.25;
                b.y[1] = 0.5;
            }
            const TflesSetup thick = testTfles(false, true);
            for (const Variant& va : {Variant{"LES", cs, sg, std::nullopt}, Variant{"LES + TFLES", cs, sg, thick}}) {
                const FvState o = run({.tr = m.tr, .bc = ubc, .precision = md.pr, .csLes = va.cs, .sgs = va.sgs,
                                       .compensated = md.comp, .chem = passive, .tfles = va.tf},
                                      u, 40);
                bool ok;
                if (md.comp) {
                    ok = true;
                    for (std::size_t c = 0; c < std::size_t(m.fm.nC); ++c)
                        ok = ok && o.var(5)[c] == 0.25 * o.var(0)[c] && o.var(6)[c] == 0.5 * o.var(0)[c];
                } else {
                    ok = md.pr == FvPrecision::Fp32 ? exactRatio<float>(o, 5, 0.25) && exactRatio<float>(o, 6, 0.5)
                                                    : exactRatio<double>(o, 5, 0.25) && exactRatio<double>(o, 6, 0.5);
                }
                check(ok, std::format("{}, {}, {}: uniform Y bitwise, 40 steps", m.name, md.tag, va.tag));
            }
        }
    }
}

void gateConservation()
{
    std::printf("== L3 conservation under a thickened burning march, FP64\n");
    for (Case& m : cases()) {
        if (m.bc.size() != m.fm.patches.size() || std::ranges::any_of(m.bc, [](const FvBc& b) {
                return b.kind != FvBcKind::Wall && b.kind != FvBcKind::Empty;
            }))
            continue; // closed meshes only
        const bool threeD = m.fm.dim == 3;
        const double cs = threeD ? 1.35 : 0.3;
        const SgsModel sg = threeD ? SgsModel::Sigma : SgsModel::Smagorinsky;
        const FvChem chem = kChem;
        DeviceFvComp d = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp64, .csLes = cs,
                                                         .sgs = sg, .chem = chem, .tfles = testTfles(false, true)}),
                               m.name);
        FvState s = speciesState(m.fm, threeD);
        orDie(d.upload(s), "upload");
        auto sums = [&](const FvState& q) {
            std::array<double, 4> o{};
            for (std::size_t c = 0; c < std::size_t(q.nC); ++c) {
                const double V = m.fm.V[c];
                o[0] += V * q.var(0)[c];
                o[1] += V * (q.var(4)[c] + chem.q * q.var(5)[c]);
                o[2] += V * q.var(6)[c];
                o[3] += V * q.var(5)[c];
            }
            return o;
        };
        const auto a = sums(s);
        const double dt = 0.25 * orDie(d.stableDt(), "dt");
        for (int n = 0; n < 200; ++n) d.step(dt);
        FvState o;
        orDie(d.download(o), "download");
        const auto b = sums(o);
        const double eM = std::abs(b[0] / a[0] - 1.0), eE = std::abs(b[1] / a[1] - 1.0), eP = std::abs(b[2] / a[2] - 1.0);
        std::printf("  %s: fuel burnt %.3f%% over 200 steps\n", m.name, 100.0 * (1.0 - b[3] / a[3]));
        check(eM <= 1e-13, std::format("mass {:.2e} <= 1e-13", eM));
        check(eE <= 1e-13, std::format("E + q rho Y {:.2e} <= 1e-13", eE));
        check(eP <= 1e-13, std::format("passive species {:.2e} <= 1e-13", eP));
        check(b[3] < a[3], "fuel falls");
    }
}

void gateRerun()
{
    std::printf("== L4 rerun byte-identical\n");
    for (Case& m : cases()) {
        const bool threeD = m.fm.dim == 3;
        for (const FvPrecision pr : {FvPrecision::Fp32, FvPrecision::Fp64}) {
            FvState a, b;
            for (FvState* o : {&a, &b}) {
                DeviceFvComp d = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = pr,
                                                                 .csLes = threeD ? 1.35 : 0.3,
                                                                 .sgs = threeD ? SgsModel::Sigma : SgsModel::Smagorinsky,
                                                                 .chem = kChem, .tfles = testTfles(false, true)}),
                                       m.name);
                orDie(d.upload(speciesState(m.fm, threeD)), "upload");
                const double dt = 0.25 * orDie(d.stableDt(), "dt");
                for (int n = 0; n < 40; ++n) d.step(dt);
                orDie(d.download(*o), "download");
            }
            check(sameBytes(a, b), std::format("{}, {}: reruns byte-identical", m.name, fvPrecisionName(pr)));
        }
    }
}

// --- the structured comparisons: the flame model, a 3D box. -------------------

struct Flame3d
{
    Grid g;
    Transport tr;
    gpu::SpeciesSetup ss;
    CompState ic;
    std::vector<double> ry;
    double dth;
};

// A burning slab across x in [0.3, 0.7] Lx with a cos(2 pi y/Ly) wrinkle, a
// vortical velocity A env(x) localized in the middle (the curl, mu_t and E away
// from the x ends: the structured edge rows and clamp sit in quiet fresh gas).
Flame3d flame3d(int nx, int n, double A)
{
    const double M = 0.05, sigma = 5.0;
    PremixedModel md;
    md.sigma = sigma;
    const PremixedEigen eig = orDie(premixedEigen(md), "oracle");
    const double Tu = 1.0 / (kGamma * M * M), h = eig.dthOverDD / 6.0;
    Flame3d f{.g = Grid::make(nx, n, n, nx * h, n * h, n * h),
              .tr = Transport::air(md.Pr, Tu, 288.15, md.Pr, kGamma, 1.0, false),
              .ss = {},
              .ic = {},
              .ry = {},
              .dth = eig.dthOverDD};
    const Grid& g = f.g;
    f.ss.lnA = std::log(eig.Lambda);
    f.ss.Ta = md.theta() * Tu;
    f.ss.q = f.tr.cp() * (sigma - 1.0) * Tu;
    f.ss.invPr = 1.0 / md.Pr;
    f.ic = CompState::zeros(g);
    f.ry.assign(g.size(), 0.0);
    const double Lx = g.nx * g.hx, Ly = g.ny * g.hy, Lz = g.nz * g.hz, w = 1.5 * eig.dthOverDD, tau = 2 * std::numbers::pi;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy, z = (k + 0.5) * g.hz;
                const double fr = 0.3 * Lx + 2 * h * std::cos(tau * y / Ly);
                const double t = 1.0 + 0.5 * (sigma - 1.0) * (std::tanh((x - fr) / w) - std::tanh((x - 0.7 * Lx) / w));
                const double rho = 1.0 / t;
                const double env = A * std::exp(-std::pow((x - 0.4 * Lx) / (0.12 * Lx), 2));
                const double u[3] = {env * std::sin(tau * y / Ly) * std::cos(tau * z / Lz), env * std::sin(tau * z / Lz),
                                     env * std::cos(tau * y / Ly + 0.3)};
                f.ic.rho[c] = rho;
                f.ic.mx[c] = rho * u[0];
                f.ic.my[c] = rho * u[1];
                f.ic.mz[c] = rho * u[2];
                f.ic.E[c] = Tu / (kGamma - 1.0) + 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                f.ry[c] = rho * (sigma - t) / (sigma - 1.0);
            }
    return f;
}

TflesSetup flameTfles(const Flame3d& f, double F, bool uniform, bool charlette, int dilate)
{
    PremixedModel md;
    md.sigma = 5.0;
    const double Tu = f.tr.Tref, thetaS = kTflesTaFraction * md.theta();
    return {.F = F,
            .uniform = uniform,
            .charlette = charlette,
            .deltaL = f.dth,
            .nuU = md.Pr,
            .lnAs = -tflesSensorLogMax(md.sigma, thetaS),
            .TaS = thetaS * Tu,
            .dilate = dilate};
}

std::vector<double> downloadField(const gpu::DeviceScalar& d)
{
    std::vector<gpu::Real> t(d.size());
    (void)hipDeviceSynchronize();
    if (d.download(t.data()) != hipSuccess) std::exit(EXIT_FAILURE);
    return {t.begin(), t.end()};
}

// L5: on a uniform box with open x (Far, Outflow) and periodic y, z, the FV sensor,
// F and E on one state equal DeviceTfles' (the walk stops where the structured x
// pass clamps, wraps where it wraps; the edge layers are its edge rows).
void gateWalk()
{
    std::printf("== L5 the face walk is the structured filter: F E, Omega on one state\n");
    const Flame3d f = flame3d(64, 16, 2.0);
    const Grid& g = f.g;
    const StructuredSpec sp{.lo = {0, 0, 0},
                            .hi = {g.nx - 1, g.ny - 1, g.nz - 1},
                            .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                            .type = {FoamPatchType::Patch, FoamPatchType::Patch, FoamPatchType::Cyclic, FoamPatchType::Cyclic,
                                     FoamPatchType::Cyclic, FoamPatchType::Cyclic}};
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::Morton), "open box");
    const double Tu = f.tr.Tref;
    const std::vector<FvBc> bc = bcFor(fm, [&](const MeshPatch& p) {
        FvBc b;
        if (p.name == "xlo") {
            b.kind = FvBcKind::Far;
            b.qFar[0] = 1.0;
            b.qFar[4] = Tu / (kGamma - 1.0);
            b.y[0] = 1.0;
        } else {
            b.kind = FvBcKind::Outflow;
            b.pOut = Tu;
            b.tBack = 5.0 * Tu;
        }
        return b;
    });
    const FvChem chem{.nY = 1, .fuel = 0, .lnA = f.ss.lnA, .Ta = f.ss.Ta, .q = f.ss.q, .invPr = f.ss.invPr};
    for (const bool charlette : {false, true}) {
        const TflesSetup ts = flameTfles(f, 2.0, false, charlette, 12);
        auto dev = gpu::DeviceComp::make(g, f.tr, 1.35, SgsModel::Sigma);
        auto ds = gpu::DeviceCompState::make(g.size(), 0, false, 1);
        auto tf = gpu::DeviceTfles::make(g, ts);
        if (!dev || !ds || !tf || ds->upload(f.ic) != hipSuccess || ds->uploadSpecies(0, f.ry) != hipSuccess)
            std::exit(EXIT_FAILURE);
        dev->primitives(*ds);
        tf->update(*ds, dev->u(), dev->v(), dev->w(), dev->T());
        const std::vector<double> sfe = downloadField(tf->feField()), som = downloadField(tf->omField());
        DeviceFvComp d = orDie(DeviceFvComp::make(fm, {.tr = f.tr, .bc = bc, .precision = FvPrecision::Fp32, .csLes = 1.35,
                                                       .sgs = SgsModel::Sigma, .chem = chem, .tfles = ts}),
                               "fv");
        orDie(d.upload(fvStateOf(fm, f.ic, std::span(&f.ry, 1))), "upload");
        FvEval ev;
        FvState r;
        orDie(d.evaluate(ev, r), "evaluate");
        double dFe = 0.0, dOm = 0.0, eMax = 0.0, omHi = 0.0;
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            const std::size_t s = std::size_t(fm.srcCell[c]);
            dFe = std::max(dFe, std::abs(ev.fe[c] - sfe[s]) / sfe[s]);
            dOm = std::max(dOm, std::abs(ev.om[c] - som[s]));
            eMax = std::max(eMax, sfe[s] / (1.0 + (2.0 - 1.0) * som[s]));
            omHi = std::max(omHi, som[s]);
        }
        std::printf("  %s: max|F E - structured|/F E %.2e, max|Omega - structured| %.2e (max E %.4f, max Omega %.4f)\n",
                    charlette ? "dynamic + Charlette" : "dynamic", dFe, dOm, eMax, omHi);
        check(omHi > 0.99, "the sensor reaches 1 at the flame");
        if (charlette) check(eMax > 1.001, "E > 1 somewhere (the efficiency is not trivial)");
        check(dOm <= 1e-6, std::format("Omega {:.2e} <= 1e-6 (FP32 rounding of the raw sensor)", dOm));
        check(dFe <= 1e-5, std::format("F E {:.2e} <= 1e-5 (FP32 rounding: LSQ vs central curl)", dFe));
    }
}

// X1: FV vs the structured LES / TFLES flame on one uniform periodic box, same dt.
void gateStructured()
{
    std::printf("== X1 FV vs the structured LES / TFLES flame, uniform periodic 3D box, same dt\n");
    const Flame3d f = flame3d(96, 16, 2.0);
    const Grid& g = f.g;
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "box");
    const std::vector<FvBc> bc = bcFor(fm, [](const MeshPatch&) { return FvBc{}; });
    const FvChem chem{.nY = 1, .fuel = 0, .lnA = f.ss.lnA, .Ta = f.ss.Ta, .q = f.ss.q, .invPr = f.ss.invPr};
    const FvState s0 = fvStateOf(fm, f.ic, std::span(&f.ry, 1));
    struct Row
    {
        const char* tag;
        double cs;
        SgsModel sgs;
        std::optional<TflesSetup> tf;
    };
    // The model constants, then TEST-ONLY strong ones (mu_t ~ mu: the eddy terms carry weight).
    const Row rows[] = {{"LES Smagorinsky 0.17", 0.17, SgsModel::Smagorinsky, std::nullopt},
                        {"LES sigma 1.35", 1.35, SgsModel::Sigma, std::nullopt},
                        {"LES Smagorinsky 1.5 (test-only, strong)", 1.5, SgsModel::Smagorinsky, std::nullopt},
                        {"TFLES F 2 uniform, no LES", 0.0, SgsModel::Sigma, flameTfles(f, 2.0, true, false, 12)},
                        {"TFLES F 2 dynamic, no LES", 0.0, SgsModel::Sigma, flameTfles(f, 2.0, false, false, 12)},
                        {"LES sigma 1.35 + TFLES F 2 dynamic + Charlette", 1.35, SgsModel::Sigma, flameTfles(f, 2.0, false, true, 12)},
                        {"LES Smagorinsky 1.5 + TFLES F 2 dynamic + Charlette (test-only)", 1.5, SgsModel::Smagorinsky,
                         flameTfles(f, 2.0, false, true, 12)}};
    constexpr int steps = 300;
    for (const Row& rw : rows) {
        auto dev = gpu::DeviceComp::make(g, f.tr, rw.cs, rw.sgs);
        auto red = gpu::ReduceScratch::make();
        if (!dev || !red) std::exit(EXIT_FAILURE);
        std::optional<gpu::DeviceTfles> tf;
        gpu::SpeciesSetup ss = f.ss;
        if (rw.tf) {
            auto t = gpu::DeviceTfles::make(g, *rw.tf);
            if (!t) std::exit(EXIT_FAILURE);
            tf = std::move(*t);
            ss.tflesFE = tf->fe();
            ss.tflesEoF = tf->eof();
            ss.tflesOm = tf->om();
        }
        if (!dev->enableSpecies(ss)) std::exit(EXIT_FAILURE);
        auto ds = gpu::DeviceCompState::make(g.size(), 0, false, 1);
        if (!ds || ds->upload(f.ic) != hipSuccess || ds->uploadSpecies(0, f.ry) != hipSuccess) std::exit(EXIT_FAILURE);
        auto fvCfg = [&](FvPrecision pr) {
            return FvCompConfig{.tr = f.tr, .bc = bc, .precision = pr, .csLes = rw.cs, .sgs = rw.sgs, .chem = chem, .tfles = rw.tf};
        };
        DeviceFvComp d32 = orDie(DeviceFvComp::make(fm, fvCfg(FvPrecision::Fp32)), "fv32");
        DeviceFvComp d64 = orDie(DeviceFvComp::make(fm, fvCfg(FvPrecision::Fp64)), "fv64");
        orDie(d32.upload(s0), "upload");
        orDie(d64.upload(s0), "upload");
        const double dtS = dev->stableDt(*ds, *red, 0.4), dtF = orDie(d64.stableDt(), "dt");
        const float dt = 0.5f * float(std::min(dtS, dtF));
        for (int n = 0; n < steps; ++n) {
            if (tf) {
                dev->primitives(*ds);
                tf->update(*ds, dev->u(), dev->v(), dev->w(), dev->T());
            }
            dev->step(*ds, gpu::Real(dt));
            d32.step(double(dt));
            d64.step(double(dt));
        }
        auto ho = CompState::zeros(g);
        std::vector<double> yo;
        if (ds->download(ho) != hipSuccess || ds->downloadSpecies(0, yo) != hipSuccess) std::exit(EXIT_FAILURE);
        const FvState st = fvStateOf(fm, ho, std::span(&yo, 1));
        FvState o32, o64;
        orDie(d32.download(o32), "download");
        orDie(d64.download(o64), "download");
        double burnt = 0.0, b0 = 0.0, nutMax = 0.0;
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            burnt += o64.var(5)[c];
            b0 += s0.var(5)[c];
        }
        if (rw.cs > 0) nutMax = maxAbs(orDie(d64.eddyViscosity(), "nut"));
        const double a = relDiff(o32.q, st.q, 6, fm.nC), own = relDiff(o32.q, o64.q, 6, fm.nC),
                     st64 = relDiff(st.q, o64.q, 6, fm.nC), dev64 = relDiff(o64.q, s0.q, 6, fm.nC);
        std::printf("  %s: %d x %d x %d, %d steps, fuel burnt %.3f%%, max mu_t/mu %.3g, state moved %.2e\n", rw.tag, g.nx, g.ny,
                    g.nz, steps, 100.0 * (1.0 - burnt / b0), nutMax / f.tr.muRef, dev64);
        std::printf("    FV FP32 vs structured %.3e, structured vs FV FP64 %.3e, FV FP32 vs FP64 %.3e\n", a, st64, own);
        check(a <= 3.0 * own, std::format("FV FP32 vs structured {:.2e} <= 3 x own {:.2e}", a, own));
        check(st64 <= 3.0 * own, std::format("structured vs FV FP64 {:.2e} <= 3 x own {:.2e}", st64, own));
    }
}

// The binary route: a laminar F 2 thickened flame at 3.75 cells per delta_th (7.5
// across the thickened flame), dynamic sensor, FP64.
int runTfles1d()
{
#ifndef DNS_BIN
    dnstest::skip("no dns binary");
#else
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / std::format("dns_fvtfles_{}", ::getpid());
    fs::create_directories(dir);
    const std::string out = (dir / "f").string();
    const std::string cmd = std::string(DNS_BIN) +
                            " flame --solver fv --fv-precision fp64 --tfles 2 --cells-per-delta 3.75 --upstream 16 "
                            "--downstream 12 --inflow-speed 0.98 --tt 20 --emit stats --quiet --out " + out + " > /dev/null";
    if (std::system(cmd.c_str()) != 0) {
        check(false, "dns flame --solver fv --tfles 2 ran");
        return dnstest::finish();
    }
    std::ifstream in(out + "_stats.csv");
    std::string line;
    std::vector<std::string> head;
    std::vector<std::vector<double>> rows;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream s(line);
        std::string f;
        std::vector<std::string> row;
        while (std::getline(s, f, ',')) row.push_back(f);
        if (head.empty()) {
            head = row;
            continue;
        }
        std::vector<double> v;
        for (const std::string& c : row) v.push_back(std::stod(c));
        rows.push_back(std::move(v));
    }
    fs::remove_all(dir);
    auto col = [&](const char* n) { return std::size_t(std::ranges::find(head, n) - head.begin()); };
    if (col("FEmax") >= head.size() || rows.empty()) {
        check(false, "the TFLES columns Ebar, FEmax are written");
        return dnstest::finish();
    }
    double sl = 0, dth = 0, hits = 0, fe = 0;
    int m = 0;
    for (const auto& r : rows) {
        hits = std::max(hits, r[col("yHits")]);
        if (r[col("t")] < 10.0) continue;
        sl += r[col("SL")];
        dth += r[col("dth")];
        fe = std::max(fe, r[col("FEmax")]);
        ++m;
    }
    sl /= m;
    dth /= m;
    std::printf("  F 2 at 3.75 cells/delta_th: SL %.6f, dth %.5f (thickened), FEmax %.4f, hits %g\n", sl, dth, fe, hits);
    check(std::abs(sl - 1.0) <= 0.02, std::format("S_L {:.5f} within 2% of the oracle (7.5 thickened cells)", sl));
    check(std::abs(dth / (2.0 * 1.60165) - 1.0) <= 0.05, std::format("delta_th {:.5f} within 5% of 2 x the oracle's", dth));
    check(std::abs(fe - 2.0) <= 1e-12, std::format("FEmax {:.6f} == F (no efficiency)", fe));
    check(hits == 0, "no Y out of bounds");
    return dnstest::finish();
#endif
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    if (argc > 1 && std::string(argv[1]) == "tfles1d") return runTfles1d();
    gateOracle();
    gateBits();
    gateConservation();
    gateRerun();
    gateWalk();
    gateStructured();
    return dnstest::finish();
}
