// GPU compressible FV family (gpu/fv_comp, docs/fvcomp.md) vs its FP64 oracle.
//   O1 double instantiation vs FvRef: every pass and rhs <= 1e-13 relative on
//      box_cyclic, box_refined, cyl2d_snappy (Wall/Slip/Inflow/Outflow/Far,
//      Sutherland); 3 SSP-RK3 steps; stableDt
//   O2 float vs double: rhs and a 3-step state
//   O3 uniform periodic box vs Comp::rhs (double instantiation)
//   O4 freestream on cyl2d_snappy, both precisions (number stated)
//   O5 rerun byte-identical, both precisions
//   O6 adiabatic wall: mass and heat flux exactly 0.0
//   O7 2D: rho w and every z flux exactly 0.0, after 20 steps
//   O8 inviscid periodic: KE budget = pressure work; totals conserved over a march
//   O9 LES: double instantiation vs a host restatement on FvRef's gradients
//   O10 tally: closed-body uniform pressure force 0, Euler force = momentum flux
//   O11 compensated FP32 tracks FP64 where plain FP32 drops updates below ulp/2
#include "fv_comp_test_util.hpp"

#include "dns/gpu/hip_runtime.hpp"
#include "dns/sgs.hpp"
#include "gpu_test_util.hpp"

#include <cstring>
#include <random>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dnstest::fvc;
using dnstest::check;

namespace {

struct MeshCase
{
    const char* name;
    FaceMesh fm;
    std::vector<FvBc> bc;
    Transport tr;
};

FvBc farBc(const FvState& s)
{
    FvBc b;
    b.kind = FvBcKind::Far;
    for (int v = 0; v < 5; ++v) b.qFar[v] = s.var(v)[0];
    return b;
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
        v.push_back({"box_cyclic (walls: adiabatic; isothermal moving), mu 0.05", std::move(fm), {},
                     Transport::constant(kGamma, 0.05)});
        v.back().bc = bcFor(v.back().fm, pick);
    }
    {
        FaceMesh fm = loadMesh("box_refined");
        v.push_back({"box_refined (all cyclic, split faces), Sutherland", std::move(fm), {},
                     Transport::air(0.02, 0.9, 288.15, 0.72, kGamma, 1.0, true)});
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
        v.push_back({"cyl2d_snappy (wall spinning isothermal, inflow, outflow, slip, far), mu 0.01", std::move(fm), {},
                     Transport::constant(kGamma, 0.01)});
        v.back().bc = bcFor(v.back().fm, pick);
    }
    return v;
}

FvConfig refCfg(const MeshCase& m) { return {.tr = m.tr, .bc = m.bc}; }

FvCompConfig devCfg(const MeshCase& m, FvPrecision pr)
{
    return {.tr = m.tr, .bc = m.bc, .precision = pr};
}

double rowsDiff(const std::array<std::vector<double>, 12>& a, const std::array<std::vector<double>, 12>& b)
{
    double worst = 0.0;
    for (std::size_t k = 0; k < 12; ++k) {
        double e = 0.0, s = 0.0;
        for (std::size_t c = 0; c < a[k].size(); ++c) {
            e = std::max(e, std::abs(a[k][c] - b[k][c]));
            s = std::max(s, std::abs(b[k][c]));
        }
        worst = std::max(worst, s == 0.0 ? e : e / s);
    }
    return worst;
}

// O1 + O2.
void oracle()
{
    std::printf("O1 double instantiation vs FvRef; O2 float vs double\n");
    for (const MeshCase& m : cases()) {
        const bool uz = m.fm.emptyAxis < 0;
        const FvState s0 = smoothState(m.fm, uz);
        const FvRef ref = orDie(FvRef::make(m.fm, refCfg(m)), m.name);
        FvEval ev, dv;
        FvState rR, rD, rF;
        ref.evaluate(s0, ev, rR);
        auto dd = orDie(DeviceFvComp::make(m.fm, devCfg(m, FvPrecision::Fp64)), m.name);
        orDie(dd.upload(s0), m.name);
        orDie(dd.evaluate(dv, rD), m.name);
        const double eR = relDiff(rD.q, rR.q, 5, m.fm.nC);
        const double eG = rowsDiff(dv.grad, ev.grad);
        double eF = 0.0;
        for (std::size_t k = 0; k < 5; ++k) {
            double e = 0.0, sc = 0.0;
            for (std::size_t f = 0; f < ev.F[k].size(); ++f) {
                e = std::max(e, std::abs(ev.F[k][f] - dv.F[k][f]));
                sc = std::max(sc, std::abs(ev.F[k][f]));
            }
            eF = std::max(eF, sc == 0 ? e : e / sc);
        }
        check(eR <= 1e-13, std::format("{}: rhs vs FvRef {:.2e} <= 1e-13", m.name, eR));
        check(eG <= 1e-13, std::format("  gradients vs FvRef {:.2e} <= 1e-13", eG));
        check(eF <= 1e-13, std::format("  face fluxes vs FvRef {:.2e} <= 1e-13", eF));

        const double dtRef = orDie(ref.stableDt(s0, {.cfl = 0.4}), "dt");
        orDie(dd.upload(s0), m.name);
        const double dtDev = orDie(dd.stableDt({.cfl = 0.4}), "dt");
        const double eDt = std::abs(dtDev / dtRef - 1.0);
        check(eDt <= 1e-13, std::format("  stableDt {:.6e} vs FvRef {:.2e} <= 1e-13", dtDev, eDt));

        FvState sR = s0, sD;
        ref.march(sR, dtRef, 3);
        for (int i = 0; i < 3; ++i) dd.step(dtRef);
        orDie(dd.download(sD), m.name);
        const double eM = relDiff(sD.q, sR.q, 5, m.fm.nC);
        check(eM <= 1e-13, std::format("  3 SSP-RK3 steps vs FvRef {:.2e} <= 1e-13", eM));

        auto df = orDie(DeviceFvComp::make(m.fm, devCfg(m, FvPrecision::Fp32)), m.name);
        orDie(df.upload(s0), m.name);
        orDie(df.rhs(rF), m.name);
        const double fR = relDiff(rF.q, rD.q, 5, m.fm.nC);
        for (int i = 0; i < 3; ++i) df.step(dtRef);
        FvState sF;
        orDie(df.download(sF), m.name);
        const double fS = relDiff(sF.q, sD.q, 5, m.fm.nC);
        check(fR <= 2e-5, std::format("  float rhs vs double {:.2e} <= 2e-5 (FP32 cancellation in the flux sum)", fR));
        check(fS <= 2e-6, std::format("  float 3-step state vs double {:.2e} <= 2e-6", fS));
    }
}

// O3.
void structured()
{
    std::printf("O3 uniform periodic box vs Comp::rhs\n");
    const Grid g = Grid::make(8, 6, 5, 8 * 0.3, 6 * 0.45, 5 * 0.2);
    const Transport tr = Transport::constant(kGamma, 0.05);
    Comp comp = Comp::make(g, kGamma, tr.muRef);
    comp.tr = tr;
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "box");
    const FvState s = smoothState(fm, true);
    CompState q = CompState::zeros(g), r = CompState::zeros(g);
    for (std::size_t n = 0; n < std::size_t(fm.nC); ++n) {
        const std::size_t c = std::size_t(fm.srcCell[n]);
        q.rho[c] = s.var(0)[n];
        q.mx[c] = s.var(1)[n];
        q.my[c] = s.var(2)[n];
        q.mz[c] = s.var(3)[n];
        q.E[c] = s.var(4)[n];
    }
    comp.rhs(q, r);
    FvState expect = FvState::zeros(fm.nC), out;
    const std::vector<double>* rv[5] = {&r.rho, &r.mx, &r.my, &r.mz, &r.E};
    for (int v = 0; v < 5; ++v)
        for (std::size_t n = 0; n < std::size_t(fm.nC); ++n) expect.var(v)[n] = (*rv[v])[std::size_t(fm.srcCell[n])];
    auto d = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = {}, .precision = FvPrecision::Fp64}), "box");
    orDie(d.upload(s), "box");
    orDie(d.rhs(out), "box");
    const double e = relDiff(out.q, expect.q, 5, fm.nC);
    check(e <= 1e-13, std::format("device FV (double) vs Comp::rhs {:.2e} <= 1e-13", e));
}

// O4.
void freestream()
{
    std::printf("O4 freestream on cyl2d_snappy (Far on every open patch, adiabatic slip-free interior)\n");
    FaceMesh fm = loadMesh("cyl2d_snappy");
    const double u[3] = {0.6, 0.2, 0.0}, rho = 1.0, p = 1.0 / kGamma;
    FvState s = FvState::zeros(fm.nC);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) setPrim(s, c, rho, u, p);
    const FvBc far = farBc(s);
    const auto bc = bcFor(fm, [&](const MeshPatch&) { return far; });
    const Transport tr = Transport::constant(kGamma, 0.01);
    for (const FvPrecision pr : {FvPrecision::Fp64, FvPrecision::Fp32}) {
        auto d = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = bc, .precision = pr}), "fs");
        orDie(d.upload(s), "fs");
        FvState out;
        orDie(d.rhs(out), "fs");
        const double e = maxAbs(out.q);
        const double dt = orDie(d.stableDt({.cfl = 0.4}), "fs");
        for (int i = 0; i < 100; ++i) d.step(dt);
        FvState st;
        orDie(d.download(st), "fs");
        const double drift = relDiff(st.q, s.q, 5, fm.nC);
        const double gate = pr == FvPrecision::Fp64 ? 1e-12 : 5e-5;
        const double gateD = pr == FvPrecision::Fp64 ? 1e-12 : 2e-6;
        check(e <= gate, std::format("{}: max|rhs| {:.2e} <= {:.0e} (FP32: Sum S rounds, p/V amplifies)",
                                     fvPrecisionName(pr), e, gate));
        check(drift <= gateD,
              std::format("{}: 100 steps, state drift {:.2e} <= {:.0e}", fvPrecisionName(pr), drift, gateD));
    }
}

// O5.
void rerun()
{
    std::printf("O5 rerun byte-identical\n");
    for (const MeshCase& m : cases())
        for (const FvPrecision pr : {FvPrecision::Fp64, FvPrecision::Fp32}) {
            const FvState s0 = smoothState(m.fm, m.fm.emptyAxis < 0);
            std::vector<double> runs[2];
            for (auto& q : runs) {
                auto d = orDie(DeviceFvComp::make(m.fm, devCfg(m, pr)), m.name);
                orDie(d.upload(s0), m.name);
                const double dt = orDie(d.stableDt({.cfl = 0.3}), "dt");
                for (int i = 0; i < 40; ++i) d.step(dt);
                FvState s;
                orDie(d.download(s), m.name);
                q = s.q;
            }
            check(std::memcmp(runs[0].data(), runs[1].data(), runs[0].size() * sizeof(double)) == 0,
                  std::format("{} {}: 40 steps, two runs memcmp equal", m.name, fvPrecisionName(pr)));
        }
}

// O6.
void adiabatic()
{
    std::printf("O6 adiabatic wall flux\n");
    FaceMesh fm = loadMesh("cyl2d_snappy");
    FvBc far;
    far.kind = FvBcKind::Far;
    far.qFar[0] = 1.0;
    far.qFar[1] = 0.5;
    far.qFar[4] = 1.0 / (kGamma - 1.0) + 0.125;
    const auto bc = bcFor(fm, [&](const MeshPatch& p) {
        FvBc b = far;
        if (p.name == "cylinder") b = FvBc{};
        return b;
    });
    const Transport tr = Transport::constant(kGamma, 0.01);
    for (const FvPrecision pr : {FvPrecision::Fp64, FvPrecision::Fp32}) {
        auto d = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = bc, .precision = pr}), "o6");
        orDie(d.upload(smoothState(fm, false)), "o6");
        FvEval ev;
        FvState out;
        orDie(d.evaluate(ev, out), "o6");
        int n = 0;
        double mass = 0.0, heat = 0.0;
        for (const MeshPatch& p : fm.patches) {
            if (p.name != "cylinder") continue;
            for (int b = 0; b < p.count; ++b, ++n) {
                const std::size_t f = std::size_t(fm.nI + p.first + b);
                mass = std::max(mass, std::abs(ev.F[0][f]));
                heat = std::max(heat, std::abs(ev.F[4][f]));
            }
        }
        check(n == 32 && mass == 0.0 && heat == 0.0,
              std::format("{}: {} wall faces, max |mass flux| {}, max |energy flux| {} (both exactly 0)",
                          fvPrecisionName(pr), n, mass, heat));
    }
}

// O7.
void twoD()
{
    std::printf("O7 2D restriction on cyl2d_snappy\n");
    auto cs = cases();
    const MeshCase& m = cs[2];
    for (const FvPrecision pr : {FvPrecision::Fp64, FvPrecision::Fp32}) {
        auto d = orDie(DeviceFvComp::make(m.fm, devCfg(m, pr)), m.name);
        const FvState s0 = smoothState(m.fm, false);
        orDie(d.upload(s0), m.name);
        FvEval ev;
        FvState r;
        orDie(d.evaluate(ev, r), m.name);
        const double fz = maxAbs(ev.F[3]), rz = maxAbs({r.var(3), r.var(3) + m.fm.nC});
        double gz = 0.0;
        for (std::size_t k = 0; k < 4; ++k) gz = std::max(gz, maxAbs(ev.grad[3 * k + 2]));
        const double dt = orDie(d.stableDt({.cfl = 0.3}), "dt");
        for (int i = 0; i < 20; ++i) d.step(dt);
        FvState s;
        orDie(d.download(s), m.name);
        const double w = maxAbs({s.var(3), s.var(3) + m.fm.nC});
        check(fz == 0.0 && rz == 0.0 && gz == 0.0 && w == 0.0,
              std::format("{}: z flux {}, rhs {}, d/dz {}, rho w after 20 steps {} (all exactly 0)",
                          fvPrecisionName(pr), fz, rz, gz, w));
    }
}

// O8.
void keConservation()
{
    std::printf("O8 inviscid KE: box_refined\n");
    const FaceMesh fm = loadMesh("box_refined");
    const Transport tr = Transport::constant(kGamma, 0.0);
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (const bool uniformP : {true, false}) {
        FvState s = FvState::zeros(fm.nC), out;
        std::vector<double> pc(std::size_t(fm.nC));
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            const double u[3] = {0.5 * U(rng), 0.5 * U(rng), 0.5 * U(rng)};
            pc[c] = uniformP ? 1.0 : 1.0 + 0.2 * U(rng);
            setPrim(s, c, 1.0 + 0.2 * U(rng), u, pc[c]);
        }
        auto d = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = {}, .precision = FvPrecision::Fp64}), "o8");
        orDie(d.upload(s), "o8");
        orDie(d.rhs(out), "o8");
        auto vel = [&](std::size_t c, int k) { return s.var(1 + k)[c] / s.var(0)[c]; };
        double ke = 0.0, scale = 0.0, pw = 0.0;
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            double uu = 0.0, um = 0.0, am = 0.0;
            for (int k = 0; k < 3; ++k) {
                uu += vel(c, k) * vel(c, k);
                um += vel(c, k) * out.var(1 + k)[c];
                am += std::abs(vel(c, k) * out.var(1 + k)[c]);
            }
            ke += fm.V[c] * (um - 0.5 * uu * out.var(0)[c]);
            scale += fm.V[c] * (am + 0.5 * uu * std::abs(out.var(0)[c]));
        }
        for (std::size_t f = 0; f < std::size_t(fm.nI); ++f) {
            const std::size_t o = std::size_t(fm.own[f]), n = std::size_t(fm.nei[f]);
            double du = 0.0;
            for (int k = 0; k < 3; ++k) du += fm.S[std::size_t(k)][f] * (vel(n, k) - vel(o, k));
            pw += 0.5 * (pc[o] + pc[n]) * du;
        }
        const double e = std::abs(ke - pw) / scale;
        check(e <= 1e-13, std::format("{}: |KE rate - pressure work| / scale {:.2e} <= 1e-13 (pw {:.3e})",
                                      uniformP ? "uniform p" : "random p", e, pw));
    }
    // Totals and KE over a smooth inviscid march, FP64.
    FvState s = smoothState(fm, true, 0.5);
    auto d = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = {}, .precision = FvPrecision::Fp64}), "o8");
    orDie(d.upload(s), "o8");
    const auto t0 = orDie(d.totals(), "o8");
    const double dt = orDie(d.stableDt({.cfl = 0.2}), "o8");
    for (int i = 0; i < 20; ++i) d.step(dt);
    const auto t1 = orDie(d.totals(), "o8");
    const double em = std::abs(t1.mass / t0.mass - 1.0), eE = std::abs(t1.energy / t0.energy - 1.0);
    double eP = 0.0;
    for (std::size_t k = 0; k < 3; ++k) eP = std::max(eP, std::abs(t1.mom[k] - t0.mom[k]));
    check(em <= 1e-14 && eE <= 1e-14 && eP <= 1e-13,
          std::format("20 steps: mass {:.2e}, energy {:.2e}, momentum {:.2e} (rounding)", em, eE, eP));
}

// O9: mu_t and rhs against a host restatement on FvRef's own gradients.
void les()
{
    std::printf("O9 LES, double instantiation vs host restatement\n");
    const Grid g = Grid::make(10, 9, 8, 1.0, 0.9, 0.8);
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::sinusoidal(g, 0.04), periodicSpec(g), CellOrder::Morton), "les");
    const Transport tr = Transport::constant(kGamma, 0.01);
    const double cs = 0.17, prt = 0.9;
    const FvRef ref = orDie(FvRef::make(fm, {.tr = tr, .bc = {}}), "les");
    const FvState s = smoothState(fm, true);
    FvEval ev;
    FvState rr;
    ref.evaluate(s, ev, rr);
    for (const SgsModel sm : {SgsModel::Smagorinsky, SgsModel::Sigma}) {
        const int sigma = sm == SgsModel::Sigma;
        const double C = sigma ? 1.35 : cs;
        const std::size_t nC = std::size_t(fm.nC), nI = std::size_t(fm.nI);
        std::vector<double> nut(nC);
        for (std::size_t c = 0; c < nC; ++c) {
            double gd[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) gd[i][j] = ev.grad[std::size_t(3 * i + j)][c];
            const double d = std::cbrt(fm.V[c]);
            nut[c] = ev.prim[0][c] * (C * C * d * d) * sgsRate(sigma, gd);
        }
        auto d = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = {}, .precision = FvPrecision::Fp64, .csLes = C, .sgs = sm, .prt = prt}), "les");
        orDie(d.upload(s), "les");
        const auto dn = orDie(d.eddyViscosity(), "les");
        double eN = 0.0, sN = 0.0;
        for (std::size_t c = 0; c < nC; ++c) {
            eN = std::max(eN, std::abs(dn[c] - nut[c]));
            sN = std::max(sN, std::abs(nut[c]));
        }
        std::vector<double> R(5 * nC, 0.0);
        const FvFaceGeom& fg = ref.faceGeom();
        for (std::size_t f = 0; f < nI; ++f) {
            const std::size_t o = std::size_t(fm.own[f]), n = std::size_t(fm.nei[f]);
            auto prim = [&](std::size_t c) {
                return FvPrim<double>{ev.prim[0][c], {ev.prim[1][c], ev.prim[2][c], ev.prim[3][c]}, ev.prim[4][c],
                                      ev.prim[5][c], ev.prim[6][c]};
            };
            const FvPrim<double> A = prim(o), B = prim(n);
            const double S[3] = {fm.S[0][f], fm.S[1][f], fm.S[2][f]}, dv[3] = {fm.d[0][f], fm.d[1][f], fm.d[2][f]},
                         co[3] = {fg.corr[0][f], fg.corr[1][f], fg.corr[2][f]};
            double gu[3][3], gT[3], ga[3], gb[3];
            auto cg = [&](std::size_t c, int k, double* gq) {
                for (int a = 0; a < 3; ++a) gq[a] = ev.grad[std::size_t(3 * k + a)][c];
            };
            for (int k = 0; k < 3; ++k) {
                cg(o, k, ga);
                cg(n, k, gb);
                fvFaceGrad(ga, gb, A.u[k], B.u[k], dv, co, gu[k]);
            }
            cg(o, 3, ga);
            cg(n, 3, gb);
            fvFaceGrad(ga, gb, A.temp, B.temp, dv, co, gT);
            double mu, kappa;
            fvTransportAt(tr, 0.5 * (A.temp + B.temp), mu, kappa);
            const double nf = 0.5 * (nut[o] + nut[n]);
            mu += nf;
            kappa += nf * tr.cp() / prt;
            const FvFlux<double> F = fvMeanFlux(A, B, S, gu, gT, mu, kappa);
            const double fl[5] = {F.rho, F.m[0], F.m[1], F.m[2], F.E};
            for (std::size_t v = 0; v < 5; ++v) {
                R[v * nC + o] -= fl[v] / fm.V[o];
                R[v * nC + n] += fl[v] / fm.V[n];
            }
        }
        FvState out;
        orDie(d.rhs(out), "les");
        const double eR = relDiff(out.q, R, 5, fm.nC);
        // The bound is the operator's own conditioning, measured here: the host
        // operator under a relative 2^-52 perturbation of each gradient entry.
        std::mt19937_64 rng(5);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        double cond = 0.0;
        for (std::size_t c = 0; c < nC; ++c) {
            double gd[3][3], gp[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    gd[i][j] = ev.grad[std::size_t(3 * i + j)][c];
                    gp[i][j] = gd[i][j] * (1.0 + 2.2e-16 * U(rng));
                }
            const double d = std::cbrt(fm.V[c]);
            const double a = ev.prim[0][c] * (C * C * d * d);
            cond = std::max(cond, a * std::abs(sgsRate(sigma, gd) - sgsRate(sigma, gp)));
        }
        check(eN <= 3.0 * cond + 1e-15 * sN && eR <= 1e-12,
              std::format("{}: mu_t {:.2e} (operator sensitivity to 1 ulp of grad u {:.2e}), rhs {:.2e} (mu_t max "
                          "{:.3e})",
                          sigma ? "sigma" : "smag", eN / sN, cond / sN, eR, sN));
    }
}

// O10.
void tally()
{
    std::printf("O10 tally\n");
    FaceMesh fm = loadMesh("cyl2d_snappy");
    const double u[3] = {0.5, 0.0, 0.0}, p = 1.0;
    FvState s = FvState::zeros(fm.nC);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) setPrim(s, c, 1.0, u, p);
    FvBc far = farBc(s);
    const auto bc = bcFor(fm, [&](const MeshPatch& pt) { return pt.name == "cylinder" ? FvBc{} : far; });
    auto d = orDie(DeviceFvComp::make(fm, {.tr = Transport::constant(kGamma, 0.0), .bc = bc,
                                           .precision = FvPrecision::Fp64, .pRef = 0.0}), "o10");
    orDie(d.upload(s), "o10");
    const auto f = orDie(d.surface(), "o10");
    double area = 0.0;
    for (const MeshPatch& pt : fm.patches)
        if (pt.name == "cylinder")
            for (int b = 0; b < pt.count; ++b) {
                const std::size_t i = std::size_t(fm.nI + pt.first + b);
                area += std::hypot(fm.S[0][i], fm.S[1][i], fm.S[2][i]);
            }
    check(std::abs(f.fp[0]) <= 1e-12 && std::abs(f.fp[1]) <= 1e-12 && f.fv[0] == 0.0 &&
              std::abs(f.area / area - 1) < 1e-14,
          std::format("uniform p on the closed wall: force ({:.2e}, {:.2e}), viscous {}, area {:.6f}", f.fp[0],
                      f.fp[1], f.fv[0], f.area));
}

// O11.
void compensated()
{
    std::printf("O11 compensated state, cyl2d_snappy, dt 2%% of stable, 4000 steps\n");
    auto cs = cases();
    const MeshCase& m = cs[2];
    const FvState s0 = smoothState(m.fm, false, 0.2);
    auto ref = orDie(DeviceFvComp::make(m.fm, devCfg(m, FvPrecision::Fp64)), m.name);
    orDie(ref.upload(s0), m.name);
    const double dt = 0.02 * orDie(ref.stableDt({.cfl = 0.4}), "dt");
    FvState sR, sP, sC, sC2;
    auto runIt = [&](bool comp, FvState& out) {
        FvCompConfig c = devCfg(m, FvPrecision::Fp32);
        c.compensated = comp;
        auto d = orDie(DeviceFvComp::make(m.fm, c), m.name);
        orDie(d.upload(s0), m.name);
        for (int i = 0; i < 4000; ++i) d.step(dt);
        orDie(d.download(out), m.name);
    };
    for (int i = 0; i < 4000; ++i) ref.step(dt);
    orDie(ref.download(sR), m.name);
    runIt(false, sP);
    runIt(true, sC);
    runIt(true, sC2);
    const double eP = relDiff(sP.q, sR.q, 5, m.fm.nC), eC = relDiff(sC.q, sR.q, 5, m.fm.nC);
    check(eC <= 0.2 * eP, std::format("vs FP64: plain {:.2e}, compensated {:.2e} <= 0.2 x plain", eP, eC));
    check(sC.q == sC2.q, "compensated rerun identical");
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    oracle();
    structured();
    freestream();
    rerun();
    adiabatic();
    twoD();
    keConservation();
    les();
    tally();
    compensated();
    return dnstest::finish();
}
