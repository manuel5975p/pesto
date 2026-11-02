// RANS on the compressible FV family (dns/fvrans.hpp, gpu/fv_comp, docs/fvcomp.md
// "RANS") vs its oracles. Agreement proves the port, not the physics.
//   R1 double instantiation vs FvRef: every RANS pass, rhs, dt, 3 SSP-RK3 steps
//      <= 1e-13, SA-neg / SST-2003m / SST-Vm, walls of both thermal kinds, every
//      boundary kind, Sutherland, the negative SA branch
//   R2 float vs double: rhs, 3 steps
//   R3 nu~ == 0 / k == 0 = the laminar solver, memcmp of the mean rows, both
//      precisions, plain and compensated
//   R4 uniform specific scalar 2^-13, transport only: exact, conserved
//   R5 mu_t on a wall face: momentum and energy fluxes = the laminar ones, bitwise
//   R6 rerun byte-identical, hits 0
//   R7 compensated FP32 tracks FP64
//   R8 `channel`: the converged channel vs rans_channel1d, order >= 1.5
//   `channel MODEL RETAU NY BETA`: one rung (tools/fv/rans_ladder.sh)
#include "fv_comp_test_util.hpp"

#include "dns/fvcomp_case.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/rans_channel1d.hpp"
#include "dns/transport.hpp"
#include "gpu_test_util.hpp"

#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dnstest::fvc;
using dnstest::check;

namespace {

constexpr RansModel kModels[] = {RansModel::SaNeg, RansModel::Sst2003m, RansModel::SstVm};

struct MeshCase
{
    const char* name;
    FaceMesh fm;
    std::vector<FvBc> bc;
    Transport tr;
    std::vector<double> d;
};

// Distance to the nearest Wall-kind face centre: a test's positive d field, not
// a wall distance (the oracle and the device read the same one).
std::vector<double> centreDistance(const FaceMesh& fm, const std::vector<FvBc>& bc)
{
    std::vector<double> d(std::size_t(fm.nC), 1e30);
    for (std::size_t p = 0; p < fm.patches.size(); ++p) {
        if (bc[p].kind != FvBcKind::Wall) continue;
        const MeshPatch& mp = fm.patches[p];
        for (int i = 0; i < mp.count; ++i) {
            const std::size_t f = std::size_t(fm.nI + mp.first + i);
            for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
                const double r = std::hypot(fm.C[0][c] - fm.Cf[0][f], fm.C[1][c] - fm.Cf[1][f],
                                            fm.C[2][c] - fm.Cf[2][f]);
                d[c] = std::min(d[c], r);
            }
        }
    }
    return d;
}

std::vector<MeshCase> cases()
{
    std::vector<MeshCase> v;
    {
        FaceMesh fm = loadMesh("box_cyclic");
        auto pick = [](const MeshPatch& p) {
            FvBc b;
            b.kind = FvBcKind::Wall;
            if (p.name == "yhi") {
                b.isothermal = true;
                b.tWall = 0.9;
                b.uWall[0] = 0.3;
            }
            return b;
        };
        v.push_back({"box_cyclic (walls: adiabatic; isothermal moving)", std::move(fm), {},
                     Transport::constant(kGamma, 0.05), {}});
        v.back().bc = bcFor(v.back().fm, pick);
    }
    {
        FaceMesh fm = loadMesh("cyl2d_snappy");
        FvBc far;
        far.kind = FvBcKind::Far;
        far.qFar[0] = 1.0;
        far.qFar[1] = 0.5;
        far.qFar[4] = 1.0 / (kGamma - 1.0) + 0.125;
        auto pick = [&](const MeshPatch& p) {
            FvBc b;
            if (p.name == "cylinder") {
                b.kind = FvBcKind::Wall;
                b.isothermal = true;
                b.tWall = 1.1;
                b.omega[2] = 0.4;
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
                b = far;
            }
            return b;
        };
        v.push_back({"cyl2d_snappy (spinning isothermal wall, inflow, outflow, slip, far), Sutherland", std::move(fm),
                     {}, Transport::air(0.01, 1.0, 288.15, 0.72, kGamma, 1.0, true), {}});
        v.back().bc = bcFor(v.back().fm, pick);
    }
    for (MeshCase& m : v) m.d = centreDistance(m.fm, m.bc);
    return v;
}

FvRansConfig closure(const MeshCase& m, RansModel model, bool only = false)
{
    const double mu = m.tr.muRef;
    return {.model = model,
            .wallDist = m.d,
            .phiInf = ransSst(model) ? std::array<double, 2>{2.25e-3, 80.0} : std::array<double, 2>{3.0 * mu, 0.0},
            .transportOnly = only};
}

// Smooth scalars on smoothState; SA spans both signs.
FvState withScalars(const FaceMesh& fm, const FvState& s0, RansModel model, double mu)
{
    FvState s = FvState::zeros(fm.nC, 5 + ransScalars(model));
    std::ranges::copy(s0.q, s.q.begin());
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double x = fm.C[0][c], y = fm.C[1][c], z = fm.C[2][c];
        const double rho = s.var(0)[c];
        const double wave = std::sin(3.1 * x + 1.0) * std::cos(2.3 * y) * std::cos(1.7 * z + 0.3);
        if (ransSst(model)) {
            s.var(5)[c] = rho * 2e-3 * (1.2 + wave);
            // Small enough for the R F2 branch of mu_t and the production limiter.
            s.var(6)[c] = rho * 2.0 * (1.05 + std::cos(1.9 * x + 2.1 * y));
        } else {
            s.var(5)[c] = rho * mu * (1.0 + 8.0 * wave); // chi in about [-7, 9]
        }
    }
    return s;
}

double vecDiff(const std::vector<double>& a, const std::vector<double>& b)
{
    double e = 0.0, sc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
        sc = std::max(sc, std::abs(b[i]));
    }
    return sc == 0.0 ? e : e / sc;
}

template <std::size_t N>
double rows(const std::array<std::vector<double>, N>& a, const std::array<std::vector<double>, N>& b, int n = int(N))
{
    double w = 0.0;
    for (std::size_t k = 0; k < std::size_t(n); ++k) w = std::max(w, vecDiff(a[k], b[k]));
    return w;
}

// R1, R2.
void oracle()
{
    std::printf("R1 double vs FvRef, R2 float vs double\n");
    for (const MeshCase& m : cases())
        for (const RansModel model : kModels) {
            const std::string nm = std::format("{} {}", m.name, ransName(model));
            const FvState s0 = withScalars(m.fm, smoothState(m.fm, m.fm.emptyAxis < 0), model, m.tr.muRef);
            const FvRansConfig rc = closure(m, model);
            const std::array<double, 3> force{0.7, -0.2, 0.1}; // the drive through both paths
            const FvRef ref = orDie(FvRef::make(m.fm, {.tr = m.tr, .bc = m.bc, .rans = rc, .bodyForce = force}), nm);
            FvEval ev, dv;
            FvState rR, rD, rF;
            ref.evaluate(s0, ev, rR);
            auto dd = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp64, .rans = rc,
                                                      .bodyForce = force}),
                            nm);
            orDie(dd.upload(s0), nm);
            orDie(dd.evaluate(dv, rD), nm);
            const int nT = ransScalars(model);
            const double eR = relDiff(rD.q, rR.q, 5 + nT, m.fm.nC);
            const double eP = rows(dv.tfld, ev.tfld), eB = rows(dv.tbnd, ev.tbnd), eG = rows(dv.tgrad, ev.tgrad);
            const double eS = rows(dv.tsrc, ev.tsrc), eN = vecDiff(dv.nut, ev.nut), eF1 = vecDiff(dv.f1, ev.f1);
            const double eFt = rows(dv.Ft, ev.Ft, nT), eF = rows(dv.F, ev.F);
            check(eR <= 1e-13, std::format("{}: rhs {:.2e} <= 1e-13", nm, eR));
            check(eP <= 1e-13 && eB <= 1e-13, std::format("  fields {:.2e}, boundary values {:.2e} <= 1e-13", eP, eB));
            check(eG <= 1e-13, std::format("  scalar gradients {:.2e} <= 1e-13", eG));
            check(eN <= 1e-13 && eF1 <= 1e-13, std::format("  mu_t {:.2e}, F1 {:.2e} <= 1e-13", eN, eF1));
            check(eS <= 1e-13, std::format("  cell terms {:.2e} <= 1e-13", eS));
            check(eFt <= 1e-13 && eF <= 1e-13, std::format("  scalar fluxes {:.2e}, mean fluxes {:.2e} <= 1e-13", eFt, eF));
            if (ransSst(model)) { // the gate must reach mu_t's rate branch
                long limited = 0;
                for (std::size_t c = 0; c < std::size_t(m.fm.nC); ++c)
                    if (ev.nut[c] < 0.99 * ev.prim[0][c] * ev.tfld[0][c] / ev.tfld[1][c]) ++limited;
                check(limited > 0, std::format("  mu_t limited by R F2 in {} cells", limited));
            }

            const double dtR = orDie(ref.stableDt(s0, {.cfl = 0.4}), "dt");
            orDie(dd.upload(s0), nm);
            const double dtD = orDie(dd.stableDt({.cfl = 0.4}), "dt");
            check(std::abs(dtD / dtR - 1.0) <= 1e-13,
                  std::format("  dt {:.6e}, off {:.2e} <= 1e-13", dtD, std::abs(dtD / dtR - 1.0)));
            FvState sR = s0, sD;
            ref.march(sR, 0.25 * dtR, 3);
            for (int i = 0; i < 3; ++i) dd.step(0.25 * dtR);
            orDie(dd.download(sD), nm);
            const double eM = relDiff(sD.q, sR.q, 5 + nT, m.fm.nC);
            check(eM <= 1e-13, std::format("  3 SSP-RK3 steps {:.2e} <= 1e-13", eM));

            auto df = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp32, .rans = rc,
                                                      .bodyForce = force}),
                            nm);
            orDie(df.upload(s0), nm);
            orDie(df.rhs(rF), nm);
            const double fR = relDiff(rF.q, rD.q, 5 + nT, m.fm.nC);
            for (int i = 0; i < 3; ++i) df.step(0.25 * dtR);
            FvState sF;
            orDie(df.download(sF), nm);
            const double fM = relDiff(sF.q, sD.q, 5 + nT, m.fm.nC);
            check(fR <= 1e-3 && fM <= 2e-5, std::format("  FP32: rhs {:.2e} <= 1e-3, 3 steps {:.2e} <= 2e-5", fR, fM));
        }
}

// R3: the mean rows equal the laminar solver's, byte for byte.
void zeroLimit()
{
    std::printf("R3 nu~ 0 / k 0 = laminar\n");
    for (const MeshCase& m : cases())
        for (const RansModel model : kModels)
            for (const FvPrecision pr : {FvPrecision::Fp32, FvPrecision::Fp64})
                for (const bool comp : {false, true}) {
                    if (comp && pr == FvPrecision::Fp64) continue;
                    const std::string nm = std::format("{} {} {}{}", m.name, ransName(model),
                                                       pr == FvPrecision::Fp64 ? "fp64" : "fp32", comp ? " comp" : "");
                    const FvState s0 = smoothState(m.fm, m.fm.emptyAxis < 0);
                    FvState sT = FvState::zeros(m.fm.nC, 5 + ransScalars(model));
                    std::ranges::copy(s0.q, sT.q.begin());
                    FvRansConfig rc = closure(m, model);
                    rc.phiInf[0] = 0.0;
                    if (ransSst(model)) std::fill_n(sT.var(6), m.fm.nC, 40.0);
                    auto lam = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = pr, .compensated = comp}),
                                     nm);
                    auto tur = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = pr, .compensated = comp,
                                                               .rans = rc}),
                                     nm);
                    orDie(lam.upload(s0), nm);
                    orDie(tur.upload(sT), nm);
                    const double dt = 0.25 * orDie(lam.stableDt({.cfl = 0.4}), nm);
                    const double dtT = 0.25 * orDie(tur.stableDt({.cfl = 0.4}), nm);
                    for (int i = 0; i < 20; ++i) {
                        lam.step(dt);
                        tur.step(dt);
                    }
                    FvState a, b;
                    orDie(lam.download(a), nm);
                    orDie(tur.download(b), nm);
                    const std::size_t bytes = 5 * std::size_t(m.fm.nC) * sizeof(double);
                    check(std::memcmp(a.q.data(), b.q.data(), bytes) == 0 && dt == dtT,
                          std::format("{}: 20 steps equal the laminar bytes, dt equal", nm));
                }
}

// R4: phi = 2^-13 uniform stays so, exactly; the scalar's total is the mass's.
void transport()
{
    std::printf("R4 uniform specific scalar, transport only\n");
    const double phi = std::ldexp(1.0, -13);
    for (const MeshCase& m : cases())
        for (const FvPrecision pr : {FvPrecision::Fp32, FvPrecision::Fp64}) {
            const std::string nm = std::format("{} {}", m.name, pr == FvPrecision::Fp64 ? "fp64" : "fp32");
            FvRansConfig rc = closure(m, RansModel::SaNeg, true);
            rc.phiInf[0] = phi;
            const FvState s0 = smoothState(m.fm, m.fm.emptyAxis < 0);
            // Exact in FP32: rho narrowed first.
            FvState s = FvState::zeros(m.fm.nC, 6);
            std::ranges::copy(s0.q, s.q.begin());
            for (std::size_t c = 0; c < std::size_t(m.fm.nC); ++c) {
                if (pr == FvPrecision::Fp32) s.var(0)[c] = double(float(s.var(0)[c]));
                s.var(5)[c] = s.var(0)[c] * phi;
            }
            auto d = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = pr, .rans = rc}), nm);
            orDie(d.upload(s), nm);
            const double dt = 0.25 * orDie(d.stableDt({.cfl = 0.4}), nm);
            for (int i = 0; i < 40; ++i) d.step(dt);
            FvState o;
            orDie(d.download(o), nm);
            double worst = 0.0;
            for (std::size_t c = 0; c < std::size_t(m.fm.nC); ++c)
                worst = std::max(worst, std::abs(o.var(5)[c] / (o.var(0)[c] * phi) - 1.0));
            check(worst == 0.0, std::format("{}: phi / 2^-13 - 1 after 40 steps {:.2e} == 0", nm, worst));
        }
}

// R5: a wall face carries the molecular stress and heat, whatever mu_t.
void wallFace()
{
    std::printf("R5 wall face fluxes\n");
    for (const MeshCase& m : cases())
        for (const RansModel model : kModels) {
            const std::string nm = std::format("{} {}", m.name, ransName(model));
            const FvState s0 = smoothState(m.fm, m.fm.emptyAxis < 0);
            const FvState sT = withScalars(m.fm, s0, model, m.tr.muRef);
            auto lam = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp64}), nm);
            auto tur = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp64,
                                                       .rans = closure(m, model)}),
                             nm);
            FvEval a, b;
            FvState ra, rb;
            orDie(lam.upload(s0), nm);
            orDie(tur.upload(sT), nm);
            orDie(lam.evaluate(a, ra), nm);
            orDie(tur.evaluate(b, rb), nm);
            long faces = 0, bad = 0;
            double massMax = 0.0;
            for (std::size_t p = 0; p < m.fm.patches.size(); ++p) {
                if (m.bc[p].kind != FvBcKind::Wall) continue;
                for (int i = 0; i < m.fm.patches[p].count; ++i) {
                    const std::size_t f = std::size_t(m.fm.nI + m.fm.patches[p].first + i);
                    ++faces;
                    massMax = std::max(massMax, std::abs(b.F[0][f]));
                    for (std::size_t k = 0; k < 5; ++k) bad += a.F[k][f] != b.F[k][f] ? 1 : 0;
                }
            }
            check(faces > 0 && bad == 0 && massMax == 0.0,
                  std::format("{}: {} wall faces, flux == laminar's, mass 0", nm, faces));
            check(maxAbs(b.nut) > 0.0, std::format("  mu_t max {:.3e} is live", maxAbs(b.nut)));
        }
}

// R6.
void rerun()
{
    std::printf("R6 rerun, floor hits\n");
    for (const MeshCase& m : cases())
        for (const RansModel model : kModels)
            for (const FvPrecision pr : {FvPrecision::Fp32, FvPrecision::Fp64}) {
                const std::string nm = std::format("{} {} {}", m.name, ransName(model),
                                                   pr == FvPrecision::Fp64 ? "fp64" : "fp32");
                const FvState s0 = withScalars(m.fm, smoothState(m.fm, m.fm.emptyAxis < 0), model, m.tr.muRef);
                FvState o[2];
                unsigned long long h = 0;
                for (int r = 0; r < 2; ++r) {
                    auto d = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = pr,
                                                             .rans = closure(m, model)}),
                                   nm);
                    orDie(d.upload(s0), nm);
                    const double dt = 0.25 * orDie(d.stableDt({.cfl = 0.4}), nm);
                    for (int i = 0; i < 20; ++i) d.step(dt);
                    orDie(d.download(o[r]), nm);
                    h = orDie(d.floorHits(), nm);
                }
                check(std::memcmp(o[0].q.data(), o[1].q.data(), o[0].q.size() * sizeof(double)) == 0 && h == 0,
                      std::format("{}: 20 steps rerun identical, hits {}", nm, h));
            }
}

// R7.
void compensated()
{
    std::printf("R7 compensated FP32 vs FP64\n");
    const auto all = cases();
    const MeshCase& m = all[1];
    for (const RansModel model : kModels) {
        const std::string nm(ransName(model));
        const FvState s0 = withScalars(m.fm, smoothState(m.fm, false), model, m.tr.muRef);
        const FvRansConfig rc = closure(m, model);
        auto f64 = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp64, .rans = rc}), nm);
        auto plain = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .rans = rc}), nm);
        auto comp = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .compensated = true, .rans = rc}), nm);
        orDie(f64.upload(s0), nm);
        orDie(plain.upload(s0), nm);
        orDie(comp.upload(s0), nm);
        const double dt = 0.02 * orDie(f64.stableDt({.cfl = 0.4}), nm);
        for (int i = 0; i < 2000; ++i) {
            f64.step(dt);
            plain.step(dt);
            comp.step(dt);
        }
        FvState a, b, c;
        orDie(f64.download(a), nm);
        orDie(plain.download(b), nm);
        orDie(comp.download(c), nm);
        const int nV = 5 + ransScalars(model);
        const double eP = relDiff(b.q, a.q, nV, m.fm.nC), eC = relDiff(c.q, a.q, nV, m.fm.nC);
        check(eC <= 0.2 * eP, std::format("{}: dt 2% of stable, 2000 steps: compensated {:.2e} <= 0.2 x plain {:.2e}", nm,
                                          eC, eP));
    }
}

RansModel modelOf(std::string_view s)
{
    if (s == "sa-neg") return RansModel::SaNeg;
    if (s == "sst-2003m") return RansModel::Sst2003m;
    if (s == "sst-vm") return RansModel::SstVm;
    std::printf("FATAL: model %s\n", std::string(s).c_str());
    std::exit(1);
}

// One rung: the channel on a FaceMesh (nx cells of x, one of z), walls isothermal,
// the oracle's force, started from the oracle's discrete solution, FP64, local
// time stepping to a flat ub, read against the oracle at the march's own mass.
struct ChannelRung
{
    bool flat = false;
    double ubRel = 0;
    unsigned long long hits = 0;
};

ChannelRung channel(RansModel model, double reTau, int ny, double beta)
{
    const Channel1dCase cc{.ny = ny, .beta = beta, .reTau = reTau, .mach = 0.2, .model = model};
    const auto sol = orDie(solveChannel1d(cc), "oracle");
    const int nx = 4;
    const Grid g = channel1dGrid(cc, nx, 1, 1.0, 1.0);
    const CurvMesh mesh = CurvMesh::channelY(g, beta, 1.0);
    const StructuredSpec ss{.lo = {0, 1, 0},
                            .hi = {nx - 1, g.ny - 2, 0},
                            .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                            .type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Wall,
                                     FoamPatchType::Wall, FoamPatchType::Empty, FoamPatchType::Empty}};
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(mesh, ss, CellOrder::Morton), "mesh");
    const FvBcSpec spec{.patches = {{.name = "ylo", .kind = FvPatchKind::Wall}, {.name = "yhi", .kind = FvPatchKind::Wall}}};
    auto dist = orDie(fvRansWallDistance(orDie(FaceMesh::polyOf(mesh, ss), "poly"), fm, spec), "wall distance");

    const CompState q0 = toCompState(sol, g);
    FvState s = FvState::zeros(fm.nC, 5 + ransScalars(model));
    const std::vector<double>* mean[5] = {&q0.rho, &q0.mx, &q0.my, &q0.mz, &q0.E};
    std::vector<std::vector<double>> turb;
    for (int e = 0; e < ransScalars(model); ++e) turb.push_back(toTurb(sol, g, e));
    std::vector<int> row(static_cast<std::size_t>(fm.nC));
    double dErr = 0.0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const std::size_t src = std::size_t(fm.srcCell[c]);
        row[c] = int(src / std::size_t(g.nx)) % g.ny;
        for (int v = 0; v < 5; ++v) s.var(v)[c] = (*mean[v])[src];
        for (std::size_t e = 0; e < turb.size(); ++e) s.var(5 + int(e))[c] = turb[e][src];
        dErr = std::max(dErr, std::abs(dist[c] / sol.d[std::size_t(row[c] - 1)] - 1.0));
    }
    FvBc wall;
    wall.kind = FvBcKind::Wall;
    wall.isothermal = true;
    wall.tWall = sol.Twall;
    const Transport tr = Transport::constant(kGammaAir, 1.0 / reTau);
    const auto bc = bcFor(fm, [&](const MeshPatch&) { return wall; });
    const FvRansConfig rc{.model = model, .wallDist = dist};
    const std::array<double, 3> force{1.0, 0.0, 0.0};
    FvEval ev;
    FvState rr;
    auto dev = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = bc, .precision = FvPrecision::Fp64, .rans = rc,
                                             .bodyForce = force}),
                     "device");
    orDie(dev.upload(s), "upload");
    const FvRef ref = orDie(FvRef::make(fm, {.tr = tr, .bc = bc, .rans = rc, .bodyForce = force}), "ref");
    FvState rRef;
    ref.rhs(s, rRef);
    orDie(dev.rhs(rr), "rhs");
    double eRhs = 0.0; // absolute, in force units: x/z-homogeneous rows hold only rounding
    for (std::size_t i = 0; i < rr.q.size(); ++i) eRhs = std::max(eRhs, std::abs(rr.q[i] - rRef.q[i]));
    std::printf("channel %s Re_tau %g ny %d beta %g: %d cells, y1+ %.3f, wall distance vs oracle %.1e, |rhs - FvRef| %.2e\n",
                std::string(ransName(model)).c_str(), reTau, ny, beta, fm.nC, sol.d[0] * reTau, dErr, eRhs);

    auto ubOf = [&](const FvState& q) {
        double m = 0.0, mu = 0.0;
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            m += q.var(0)[c] * fm.V[c];
            mu += q.var(1)[c] * fm.V[c];
        }
        return mu / m;
    };
    const auto wall0 = std::chrono::steady_clock::now();
    const double wallCap = 280.0;
    double ubPrev = ubOf(s), dPrev = 0.0, dt = 0.0;
    long steps = 0;
    bool flat = false;
    FvState o;
    // Pseudo-time from the oracle: stop when the geometric tail of ub is < 2e-8.
    while (!flat) {
        for (int i = 0; i < 2000; ++i) {
            if (i % 16 == 0) dt = 0.95 * orDie(dev.localDt({.cfl = 1.0}), "local dt");
            dev.step(dt);
        }
        steps += 2000;
        orDie(dev.download(o), "download");
        const double ub = ubOf(o), dub = ub / ubPrev - 1.0;
        const double r = dPrev != 0.0 ? std::abs(dub / dPrev) : 1.0;
        const double tail = r < 1.0 ? std::abs(dub) * r / (1.0 - r) : 1.0;
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
        if (steps % 100000 == 0 || tail < 2e-8)
            std::printf("  it %ld dtRef %.3e ub %.8f d %.2e tail %.2e wall %.0f s\n", steps, dt, ub, dub, tail, wall);
        std::fflush(stdout);
        flat = tail < 2e-8 && std::abs(dub) < 1e-8;
        ubPrev = ub;
        dPrev = dub;
        if (wall > wallCap) break;
    }
    // The march's mass: local time stepping conserves sum V rho / lts, and the
    // steady equations are a one-parameter family in mass (docs/rans.md).
    double massFv = 0.0, vol = 0.0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        massFv += o.var(0)[c] * fm.V[c];
        vol += fm.V[c];
    }
    Channel1dCase cm = cc;
    cm.totalMass = 2.0 * massFv / vol;
    const auto om = orDie(solveChannel1d(cm), "oracle at the march's mass");
    // Profiles: x mean per row against the oracle.
    std::vector<double> u(std::size_t(ny), 0.0), n(std::size_t(ny), 0.0), T(std::size_t(ny), 0.0);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const std::size_t j = std::size_t(row[c] - 1);
        const double rho = o.var(0)[c], ux = o.var(1)[c] / rho;
        u[j] += ux / nx;
        n[j] += o.var(5)[c] / rho / nx;
        T[j] += (kGammaAir - 1.0) * (o.var(4)[c] - 0.5 * rho * ux * ux) / rho / nx;
    }
    double eU = 0.0, eN = 0.0, eT = 0.0;
    const std::vector<double>& orc = model == RansModel::SaNeg ? om.nut : om.k;
    for (std::size_t j = 0; j < std::size_t(ny); ++j) {
        eU = std::max(eU, std::abs(u[j] - om.u[j]) / om.u[std::size_t(ny / 2)]);
        eN = std::max(eN, std::abs(n[j] - orc[j]) / maxAbs(orc));
        eT = std::max(eT, std::abs(T[j] - om.T[j]) / std::abs(om.T[std::size_t(ny / 2)] - om.Twall));
    }
    orDie(dev.rhs(rr), "rhs");
    double resM = 0.0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) resM = std::max(resM, std::abs(rr.var(1)[c]));
    const gpu::FvSurface sf = orDie(dev.surface(), "surface");
    const double tau = std::abs(sf.fv[0]) / sf.area;
    const unsigned long long h = orDie(dev.floorHits(), "hits");
    std::printf("RESULT %s Re_tau %g ny %d beta %g: %s, mass %.8f, ub %.8f oracle %.8f rel %+.4e, u max %.3e, T max %.3e, "
                "scalar max %.3e, tau_w %.8f oracle %.8f, res_m/f %.2e, hits %llu, %ld it, %.0f s\n",
                std::string(ransName(model)).c_str(), reTau, ny, beta, flat ? "flat" : "NOT FLAT (wall cap)", cm.totalMass,
                ubPrev, om.ub, ubPrev / om.ub - 1.0, eU, eT, eN, tau, om.tauWall, resM, h, steps,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count());
    return {.flat = flat, .ubRel = ubPrev / om.ub - 1.0, .hits = h};
}

// Two rungs per closure: FV - oracle shrinks with h. The schemes differ legitimately
// (LSQ vs central cell gradients on a stretched mesh; the wall's face vs its ghost).
void channelGate()
{
    std::printf("R8 channel Re_tau 180, FV vs rans_channel1d: converging order\n");
    for (const RansModel model : kModels) {
        const ChannelRung a = channel(model, 180.0, 24, 2.0), b = channel(model, 180.0, 48, 2.0);
        const double order = std::log2(a.ubRel / b.ubRel);
        check(a.flat && b.flat && a.hits == 0 && b.hits == 0, std::format("{}: flat, no floor hits", ransName(model)));
        check(order >= 1.5, std::format("  ub {:+.3e}, {:+.3e}: order {:.2f} >= 1.5", a.ubRel, b.ubRel, order));
    }
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    if (argc == 6 && std::string_view(argv[1]) == "channel")
        return channel(modelOf(argv[2]), std::atof(argv[3]), std::atoi(argv[4]), std::atof(argv[5])).flat ? 0 : 1;
    if (argc == 2 && std::string_view(argv[1]) == "channel") {
        channelGate();
        return dnstest::finish();
    }
    oracle();
    zeroLimit();
    transport();
    wallFace();
    rerun();
    compensated();
    return dnstest::finish();
}
