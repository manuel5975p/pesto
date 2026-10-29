#include "dns/gpu/runner_fv.hpp"

#include "dns/foamio.hpp"
#include "dns/gpu/fv_simple.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/interrupt.hpp"
#include "dns/la/graph.hpp"

#include <algorithm>
#include <chrono>
#include <array>
#include <cstdio>
#include <format>
#include <fstream>
#include <string>
#include <system_error>

namespace dns::gpu {
namespace fs = std::filesystem;

namespace {

std::unexpected<RunError> fail(int code, const std::string& m) { return std::unexpected(RunError{code, m}); }

std::string timeName(int iter, double dt)
{
    const double t = iter * dt;
    return std::format("{:g}", t);
}

// The OpenFOAM forces .dat header (v2606 forces::createIntegratedDataFiles).
std::string forceHeader(const char* what, const std::array<double, 3>& cofr)
{
    return std::format("# {:<18}\n# CofR              : ({:.12e} {:.12e} {:.12e})\n#\n# Time              \t"
                       "total_x total_y total_z\tpressure_x pressure_y pressure_z\tviscous_x viscous_y viscous_z\n",
                       what, cofr[0], cofr[1], cofr[2]);
}

std::string forceRow(double t, const std::array<double, 3>& p, const std::array<double, 3>& v)
{
    std::string s = std::format("{:<21}", std::format("{:g}", t));
    for (int j = 0; j < 3; ++j) s += std::format("{}{:.12e}", j ? " " : "", p[std::size_t(j)] + v[std::size_t(j)]);
    for (int j = 0; j < 3; ++j) s += std::format(" {:.12e}", p[std::size_t(j)]);
    for (int j = 0; j < 3; ++j) s += std::format(" {:.12e}", v[std::size_t(j)]);
    return s + "\n";
}

// Fields of the current state in OpenFOAM order under out/<time>/.
std::expected<void, std::string> writeTime(const fs::path& out, const std::string& tn, const FvLoaded& L,
                                           const FvSimpleSetup& s, const FvState& st)
{
    const std::size_t nC = std::size_t(s.nC), nB = std::size_t(s.nB), nI = std::size_t(s.nI);
    std::vector<int> foamCell(nC, -1);
    for (std::size_t c = 0; c < nC; ++c) foamCell[std::size_t(s.srcCell[c])] = int(c);
    std::vector<double> U(3 * nC), p(nC);
    for (std::size_t i = 0; i < nC; ++i) {
        const std::size_t c = std::size_t(foamCell[i]);
        for (int q = 0; q < 3; ++q) U[3 * i + std::size_t(q)] = st.U[std::size_t(q) * nC + c];
        p[i] = st.p[c];
    }
    std::vector<FoamPatchOut> pu, pp, pf;
    bool cyclic = false;
    for (std::size_t k = 0; k < L.c.patches.size(); ++k) {
        const FvPatchBc& bc = L.c.patches[k];
        FoamPatchOut a{bc.name, {}, {}}, b{bc.name, {}, {}}, f{bc.name, {}, {}};
        if (bc.meshType == FoamPatchType::Empty || bc.meshType == FoamPatchType::Cyclic) {
            const char* t = bc.meshType == FoamPatchType::Empty ? "empty" : "cyclic";
            cyclic = cyclic || bc.meshType == FoamPatchType::Cyclic;
            a.constraint = b.constraint = f.constraint = t;
        } else {
            for (const int bf : s.patchFaces[k]) {
                const std::size_t bb = std::size_t(bf);
                for (int q = 0; q < 3; ++q) a.values.push_back(st.Ub[std::size_t(q) * nB + bb]);
                b.values.push_back(st.pb[bb]);
                f.values.push_back(st.phiB[bb]);
            }
        }
        pu.push_back(std::move(a));
        pp.push_back(std::move(b));
        pf.push_back(std::move(f));
    }
    const fs::path dir = out / tn;
    const int prec = L.c.writePrecision;
    if (auto r = writeFoamField(dir, "U", 3, false, "[0 1 -1 0 0 0 0]", U, pu, prec); !r) return r;
    if (auto r = writeFoamField(dir, "p", 1, false, "[0 2 -2 0 0 0 0]", p, pp, prec); !r) return r;
    if (L.c.ras.model != FvTurbModel::Laminar) {
        // RAS fields: k, omega (SST) / nuTilda (SA), nut; boundary `calculated`.
        auto scalar = [&](const std::string& name, const char* dims, const std::vector<double>& in,
                          const std::vector<double>& bnd) -> std::expected<void, std::string> {
            std::vector<double> x(nC);
            for (std::size_t i = 0; i < nC; ++i) x[i] = in[std::size_t(foamCell[i])];
            std::vector<FoamPatchOut> po;
            for (std::size_t k = 0; k < L.c.patches.size(); ++k) {
                const FvPatchBc& bc = L.c.patches[k];
                FoamPatchOut a{bc.name, {}, {}};
                if (bc.meshType == FoamPatchType::Empty || bc.meshType == FoamPatchType::Cyclic)
                    a.constraint = bc.meshType == FoamPatchType::Empty ? "empty" : "cyclic";
                else
                    for (const int bf : s.patchFaces[k]) a.values.push_back(bnd[std::size_t(bf)]);
                po.push_back(std::move(a));
            }
            return writeFoamField(dir, name, 1, false, dims, x, po, prec);
        };
        const bool sst = L.c.ras.model == FvTurbModel::KOmegaSST;
        if (sst) {
            if (auto r = scalar("k", "[0 2 -2 0 0 0 0]", st.s[0], st.sb[0]); !r) return r;
            if (auto r = scalar("omega", "[0 0 -1 0 0 0 0]", st.s[1], st.sb[1]); !r) return r;
        } else if (auto r = scalar("nuTilda", "[0 2 -1 0 0 0 0]", st.s[0], st.sb[0]); !r)
            return r;
        if (auto r = scalar("nut", "[0 2 -1 0 0 0 0]", st.nut, st.nutB); !r) return r;
    }
    if (cyclic) return {}; // phi in OpenFOAM order needs the cyclic patch faces: not written
    std::vector<double> phi(std::size_t(L.pm.man.nInternalFaces), 0.0);
    for (std::size_t f = 0; f < nI; ++f) {
        const std::size_t src = std::size_t(s.internalSrc[f]);
        const double sg = s.srcCell[std::size_t(s.own[f])] < s.srcCell[std::size_t(s.nei[f])] ? 1.0 : -1.0;
        phi[src] = sg * st.phi[f];
    }
    return writeFoamField(dir, "phi", 1, true, "[0 3 -1 0 0 0 0]", phi, pf, prec);
}

} // namespace

std::expected<RunStats, RunError> runFvSimple(const FvRunSpec& sp, FvLoaded& L)
{
    const FoamCase& fc = L.c;
    const auto g = la::graphOf(L.fm);
    const auto pat = la::ellPattern(g);
    auto setup = buildSimpleSetup(L.fm, fc, g, pat);
    if (!setup) return fail(3, fc.dir + ": " + setup.error());
    const FvRas& ras = fc.ras;
    const bool turb = ras.model != FvTurbModel::Laminar, sst = ras.model == FvTurbModel::KOmegaSST;
    if (turb)
        if (auto r = buildRasSetup(L.pm, L.fm, fc, *setup); !r) return fail(3, fc.dir + ": " + r.error());
    const FvSimpleSetup& s = *setup;

    const bool mrf = std::ranges::any_of(fc.mrf, [](const FvMrfZone& z) { return z.active; });
    const std::string physics =
        !turb ? std::string("laminar")
        : sst ? std::string("RANS kOmegaSST (OpenFOAM v2606), steady (SIMPLE pseudo-time: not time-accurate)")
              : std::format("RANS SpalartAllmaras (OpenFOAM v2606, ft2 {}), steady (SIMPLE pseudo-time: not "
                            "time-accurate)",
                            ras.ft2 ? "on" : "off");
    const std::string wallDist = !turb ? std::string()
                                 : ras.exactDist ? std::string(", wallDist exactDistance")
                                                 : std::string(", wallDist meshWave");
    const FvNumerics& num = sp.num;
    const bool coupled = sp.coupledAfter >= 0;
    if (coupled && turb) return fail(2, "--fv-coupled: the P0b prototype is laminar only (RAS: plan P2)");
    const std::string label =
        std::format("[FV path, body-fitted, {}, {}] incompressible SIMPLE{} (OpenFOAM v2606 simpleFoam algorithm), "
                    "nu {:g}{}{}{}",
                    physics, fvPrecisionName(num.precision), fc.simple.consistent ? "C" : "", fc.nu, wallDist,
                    mrf ? " [MRF frozen rotor: APPROXIMATION, steady rotating frame]" : "",
                    coupled ? std::format(" [coupled p-U PROTOTYPE, EXPERIMENTAL (plan_perf_5 P0b): simpleFoam fixed "
                                          "point, Rhie-Chow alpha_U {:g}; after {} SIMPLE iterations; FGMRES(30) "
                                          "relTol {:g}, alpha_c {:g} SER]",
                                          fc.relaxU ? *fc.relaxU : 1.0, sp.coupledAfter, sp.coupled.eta,
                                          sp.coupled.alpha)
                            : std::string());
    // The effective stop of every solve, stated once: max(tolerance, floor), relTol.
    auto tolOf = [&](const char* name, const FvSolverCtl& ctl, double floor) {
        const double t = std::max(ctl.tolerance, floor);
        return std::format(" {} {:g} ({}, relTol {:g})", name, t,
                           floor > ctl.tolerance ? std::format("FP32 floor; fvSolution {:g}", ctl.tolerance)
                                                 : std::string("fvSolution"),
                           ctl.relTol);
    };
    std::string tols = num.precision == FvPrecision::Fp32
                           ? std::string("FP32 (FP64 reductions): absolute tolerance = max(fvSolution, FP32 floor):")
                           : std::string("FP64 (parity mode): absolute tolerance = fvSolution:");
    tols += tolOf("U", fc.solverU, num.floors.U) + "," + tolOf("p", fc.solverP, num.floors.p);
    if (sst) tols += "," + tolOf("omega", ras.solverOmega, num.floors.s) + "," + tolOf("k", ras.solverK, num.floors.s);
    else if (turb) tols += "," + tolOf("nuTilda", ras.solverNuTilda, num.floors.s);

    // --- phase 3c: every output before the first device allocation ---
    const fs::path out(sp.outPrefix);
    std::error_code ec;
    fs::create_directories(out, ec);
    if (ec) return fail(4, out.string() + ": cannot create (" + ec.message() + ")");
    std::ofstream fForce, fMoment, fStats;
    if (fc.forces) {
        const fs::path pd = out / "postProcessing" / fc.forces->name / "0";
        fs::create_directories(pd, ec);
        if (ec) return fail(4, pd.string() + ": cannot create (" + ec.message() + ")");
        fForce.open(pd / "force.dat", std::ios::trunc);
        fMoment.open(pd / "moment.dat", std::ios::trunc);
        if (!fForce || !fMoment) return fail(4, pd.string() + ": cannot create force.dat / moment.dat");
        fForce << forceHeader("Force", fc.forces->CofR);
        fMoment << forceHeader("Moment", fc.forces->CofR);
    }
    if (sp.stats) {
        const std::string path = sp.outPrefix + "_stats.csv";
        fStats.open(path, std::ios::trunc);
        if (!fStats) return fail(4, path + ": cannot create");
        fStats << "# " << label << "\n# " << tols << "\n";
        fStats << "iter,Ux0,Uy0,Uz0,UxIt,UyIt,UzIt,p0,pFinal,pIt,contLocal,contGlobal,Fx,Fy,Fz,Mx,My,Mz,ms";
        if (turb) fStats << (sst ? ",omega0,omegaIt,k0,kIt" : ",nuTilda0,nuTildaIt");
        // coupled rows: U*0 / p0 = the probe, pFinal / pIt = FGMRES relative residual / iterations
        if (coupled) fStats << ",kIt,alphaC,msRes,msSetup,msKrylov,msUpdate,jacU,jacC";
        fStats << "\n";
    }

    // --- device ---
    auto t0 = std::chrono::steady_clock::now();
    auto ds = DeviceSimple::make(s, fc, g, num);
    if (!ds) return fail(6, ds.error());
    const double tSetup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const int N = sp.maxIters > 0 ? sp.maxIters : fc.endIter;
    if (!sp.quiet)
        std::printf("# %s\n# %s\n# FV path: %d cells, %d internal faces, %d boundary faces; device setup %.3f s; %d "
                    "iterations\n",
                    label.c_str(), tols.c_str(), s.nC, s.nI, s.nB, tSetup, N);
    RunStats rs;
    double cumCont = 0, msLoop = 0;
    const char* uSolver = "BiCGStab";
    const char* pSolver = "AMG-PCG";
    // floor-limited / all solves: U components, p, turbulence scalars
    std::array<long, 3> nFloor{}, nSolves{};
    auto mark = [](const FvSolveLog& l) { return l.floorLimited ? ", floor-limited" : ""; };
    for (int it = 1; it <= N; ++it) {
        if (stopRequested()) {
            std::fprintf(stderr, "dns: %s at iteration %d\n", stopReason(), it);
            break;
        }
        const auto ti = std::chrono::steady_clock::now();
        const bool cIt = coupled && it > sp.coupledAfter;
        auto lg = cIt ? ds->coupledIterate(sp.coupled) : ds->iterate();
        if (!lg) return fail(7, std::format("iteration {}: {}", it, lg.error()));
        FvForceSums fo{};
        if (fc.forces) {
            auto f = ds->forces();
            if (!f) return fail(7, f.error());
            fo = *f;
        }
        (void)hipDeviceSynchronize();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti).count();
        msLoop += ms;
        cumCont += lg->contGlobal;
        const bool uSolved = fc.simple.momentumPredictor && !cIt;
        for (int j = 0; j < 3; ++j)
            if (j != s.emptyAxis && uSolved) {
                ++nSolves[0];
                nFloor[0] += lg->U[std::size_t(j)].floorLimited;
            }
        for (const auto& q : lg->p) {
            ++nSolves[1];
            nFloor[1] += q.floorLimited;
        }
        for (int q = 0; q < lg->nT; ++q) {
            ++nSolves[2];
            nFloor[2] += lg->T[std::size_t(q)].floorLimited;
        }
        if (!sp.quiet && cIt) {
            std::string o = std::format("Time = {}\n\nCoupled (EXPERIMENTAL):  probe", timeName(it, fc.deltaT));
            for (int j = 0; j < 3; ++j)
                if (j != s.emptyAxis) o += std::format(" U{} {:.6g}", "xyz"[j], lg->U[std::size_t(j)].initial);
            o += std::format(" p {:.6g}, FGMRES {} iterations (relative {:.3g}), alpha_c {:.4g}, ms residual {:.2f} "
                             "setup {:.2f} Krylov {:.2f} update {:.2f}\n",
                             lg->p.front().initial, lg->kIt, lg->kRel, lg->alphaC, lg->msRes, lg->msSetup,
                             lg->msKrylov, lg->msUpdate);
            if (lg->jac[0] >= 0)
                o += std::format("G-JAC: |K v - FD| / |K v| U rows {:.3e}, continuity rows {:.3e}\n", lg->jac[0],
                                 lg->jac[1]);
            o += std::format("time step continuity errors : sum local = {:.6g}, global = {:.6g}\nExecutionTime = "
                             "{:.4f} s\n\n",
                             lg->contLocal, lg->contGlobal, msLoop / 1000.0);
            std::fputs(o.c_str(), stdout);
        } else if (!sp.quiet) {
            std::string o = std::format("Time = {}\n\n", timeName(it, fc.deltaT));
            for (int j = 0; j < 3; ++j) {
                if (j == s.emptyAxis || !uSolved) continue;
                const auto& u = lg->U[std::size_t(j)];
                o += std::format("{}:  Solving for U{}, Initial residual = {:.6g}, Final residual = {:.6g}, No "
                                 "Iterations {}{}\n",
                                 uSolver, "xyz"[j], u.initial, u.final, u.iterations, mark(u));
            }
            for (const auto& q : lg->p)
                o += std::format("{}:  Solving for p, Initial residual = {:.6g}, Final residual = {:.6g}, No "
                                 "Iterations {}{}\n",
                                 pSolver, q.initial, q.final, q.iterations, mark(q));
            if (turb) {
                const char* names[2] = {sst ? "omega" : "nuTilda", "k"};
                const int boundIdx[2] = {sst ? 1 : 0, 0};
                for (int q = 0; q < lg->nT; ++q) {
                    const auto& t = lg->T[std::size_t(q)];
                    o += std::format("{}:  Solving for {}, Initial residual = {:.6g}, Final residual = {:.6g}, No "
                                     "Iterations {}{}\n",
                                     uSolver, names[q], t.initial, t.final, t.iterations, mark(t));
                    if (lg->bounded[std::size_t(boundIdx[q])]) o += std::format("bounding {}\n", names[q]);
                }
            }
            o += std::format("time step continuity errors : sum local = {:.6g}, global = {:.6g}, cumulative = "
                             "{:.6g}\nExecutionTime = {:.4f} s\n\n",
                             lg->contLocal, lg->contGlobal, cumCont, msLoop / 1000.0);
            std::fputs(o.c_str(), stdout);
        }
        if (fc.forces) {
            const double t = it * fc.deltaT;
            fForce << forceRow(t, fo.fp, fo.fv);
            fMoment << forceRow(t, fo.mp, fo.mv);
        }
        if (sp.stats) {
            fStats << std::format("{},{:.9e},{:.9e},{:.9e},{},{},{},{:.9e},{:.9e},{},{:.9e},{:.9e}", it,
                                  lg->U[0].initial, lg->U[1].initial, lg->U[2].initial, lg->U[0].iterations,
                                  lg->U[1].iterations, lg->U[2].iterations, lg->p.front().initial,
                                  lg->p.back().final, lg->p.back().iterations, lg->contLocal, lg->contGlobal);
            for (int j = 0; j < 3; ++j) fStats << std::format(",{:.12e}", fo.fp[std::size_t(j)] + fo.fv[std::size_t(j)]);
            for (int j = 0; j < 3; ++j) fStats << std::format(",{:.12e}", fo.mp[std::size_t(j)] + fo.mv[std::size_t(j)]);
            fStats << std::format(",{:.3f}", ms);
            for (int q = 0; q < lg->nT; ++q)
                fStats << std::format(",{:.9e},{}", lg->T[std::size_t(q)].initial, lg->T[std::size_t(q)].iterations);
            if (coupled)
                fStats << std::format(",{},{:.6g},{:.3f},{:.3f},{:.3f},{:.3f},{:.3e},{:.3e}", lg->kIt, lg->alphaC,
                                      lg->msRes, lg->msSetup, lg->msKrylov, lg->msUpdate, lg->jac[0], lg->jac[1]);
            fStats << "\n";
        }
        rs.steps = it;
        // residualControl: every listed field's initial residual below its bound.
        bool conv = fc.simple.resP || fc.simple.resU || ras.resK || ras.resOmega || ras.resNuTilda;
        // coupled: the probe at the fvSolution tolerances (no FGMRES ran) is the stop
        if (cIt) conv = lg->kIt == 0;
        if (fc.simple.resP && !(lg->p.front().initial < *fc.simple.resP)) conv = false;
        if (turb) {
            // lg->T in solve order: SST omega, k; SA nuTilda
            const std::optional<double> lim[2] = {sst ? ras.resOmega : ras.resNuTilda, sst ? ras.resK : std::nullopt};
            for (int q = 0; q < lg->nT; ++q)
                if (lim[q] && !(lg->T[std::size_t(q)].initial < *lim[q])) conv = false;
        }
        if (fc.simple.resU) {
            double mx = 0;
            for (int j = 0; j < 3; ++j)
                if (j != s.emptyAxis) mx = std::max(mx, lg->U[std::size_t(j)].initial);
            if (!(mx < *fc.simple.resU)) conv = false;
        }
        const bool last = it == N || conv;
        if ((fc.writeInterval > 0 && it % fc.writeInterval == 0) || last) {
            auto st = ds->state();
            if (!st) return fail(7, st.error());
            if (auto w = writeTime(out, timeName(it, fc.deltaT), L, s, *st); !w) return fail(4, w.error());
        }
        if (conv) {
            rs.converged = true;
            if (!sp.quiet)
                std::printf("\n%s solution converged in %s iterations\n\n", cIt ? "coupled" : "SIMPLE",
                            timeName(it, fc.deltaT).c_str());
            break;
        }
    }
    if (!sp.quiet && num.precision == FvPrecision::Fp32)
        std::printf("FP32 floor-limited solves: U %ld of %ld, p %ld of %ld, turbulence %ld of %ld\n", nFloor[0],
                    nSolves[0], nFloor[1], nSolves[1], nFloor[2], nSolves[2]);
    if (fc.forces) {
        fForce.close();
        fMoment.close();
        if (!fForce || !fMoment) return fail(4, "write failed on the forces .dat files");
    }
    if (sp.stats) {
        fStats.close();
        if (!fStats) return fail(4, sp.outPrefix + "_stats.csv: write failed");
    }
    rs.msPerStep = rs.steps ? msLoop / double(rs.steps) : 0.0;
    return rs;
}

} // namespace dns::gpu
