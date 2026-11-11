#include "dns/gpu/runner_fv.hpp"

#include "dns/foamio.hpp"
#include "dns/fvpimple.hpp"
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

// OpenFOAM functionObjects::writeFile text: scientific at precision p, columns
// charWidth = p + 8 wide, left-justified.
std::string sci(double v, int p) { return std::format("{:.{}e}", v, p); }
std::string sciVec(const std::array<double, 3>& v, int p)
{
    return "(" + sci(v[0], p) + " " + sci(v[1], p) + " " + sci(v[2], p) + ")";
}
std::string commented(const std::string& s, int p) { return std::format("# {:<{}}", s, p + 8 - 2); }
std::string headerValue(const std::string& k, const std::string& v, int p)
{
    return std::format("# {:<{}}: {}\n", k, p + 8 - 2, v);
}

// forceCoeffs::writeIntegratedDataFileHeader (coefficient.dat), every coefficient.
std::string coeffHeader(const FvForces& f, int p)
{
    const FvForceCoeffs& k = *f.coeffs;
    std::string s = commented("Force and moment coefficients", p) + "\n";
    s += headerValue("dragDir", sciVec(k.e1, p), p) + headerValue("sideDir", sciVec(k.e2, p), p) +
         headerValue("liftDir", sciVec(k.e3, p), p) + headerValue("rollAxis", sciVec(k.e1, p), p) +
         headerValue("pitchAxis", sciVec(k.e2, p), p) + headerValue("yawAxis", sciVec(k.e3, p), p) +
         headerValue("magUInf", sci(k.magUInf, p), p) + headerValue("lRef", sci(k.lRef, p), p) +
         headerValue("Aref", sci(k.Aref, p), p) + headerValue("CofR", sciVec(f.CofR, p), p) + "#\n";
    s += commented("Time", p);
    for (const char* n : {"Cd", "Cd(f)", "Cd(r)", "Cl", "Cl(f)", "Cl(r)", "CmPitch", "CmRoll", "CmYaw", "Cs", "Cs(f)",
                          "Cs(r)"})
        s += std::format("\t{:<{}}", n, p + 8);
    return s + "\n";
}

std::string coeffRow(double t, const std::array<double, 12>& c, int p)
{
    std::string s = std::format("{:<{}}", std::format("{:g}", t), p + 8);
    for (const double x : c) s += "\t" + sci(x, p);
    return s + "\n";
}

// actuationDiskSource::writeFileHeader (Froude) and calcFroudeMethod's row.
std::string diskHeader(int p)
{
    std::string s = commented("Actuation disk source", p) + "\n";
    for (const char* n : {"Time", "Uref", "Cp", "Ct", "a", "T", "diskDir"}) s += commented(n, p);
    return s + "\n";
}

std::string diskRow(double t, const FvDiskLog& d, const std::array<double, 3>& dir, int p)
{
    return std::format("{:<{}}", std::format("{:g}", t), p + 8) + sciVec(d.Uref, p) + "\t" + sci(d.Cp, p) + "\t" +
           sci(d.Ct, p) + "\t" + sci(d.a, p) + "\t" + sci(d.T, p) + "\t" + sciVec(dir, p) + "\n";
}

// A function object's files: per forces object force.dat + moment.dat, per
// forceCoeffs object coefficient.dat (a: the first file, b: moment.dat), per
// disk its actuationDiskSource.dat.
struct Files
{
    std::ofstream a, b;
};

// out/postProcessing/<name>/0/<file>, truncated.
std::expected<void, RunError> openPost(const fs::path& out, const std::string& name, const char* file, std::ofstream& f)
{
    std::error_code ec;
    const fs::path pd = out / "postProcessing" / name / "0";
    fs::create_directories(pd, ec);
    if (ec) return fail(4, pd.string() + ": cannot create (" + ec.message() + ")");
    f.open(pd / file, std::ios::trunc);
    if (!f) return fail(4, (pd / file).string() + ": cannot create");
    return {};
}

// Every forces / forceCoeffs object's files with their headers.
std::expected<void, RunError> openForceFiles(const fs::path& out, const FoamCase& fc, std::vector<Files>& ff)
{
    ff = std::vector<Files>(fc.forces.size());
    for (std::size_t k = 0; k < fc.forces.size(); ++k) {
        const FvForces& f = fc.forces[k];
        if (f.coeffs) {
            if (auto r = openPost(out, f.name, "coefficient.dat", ff[k].a); !r) return r;
            ff[k].a << coeffHeader(f, fc.writePrecision);
            continue;
        }
        if (auto r = openPost(out, f.name, "force.dat", ff[k].a); !r) return r;
        if (auto r = openPost(out, f.name, "moment.dat", ff[k].b); !r) return r;
        ff[k].a << forceHeader("Force", f.CofR);
        ff[k].b << forceHeader("Moment", f.CofR);
    }
    return {};
}

// One row per object at time t (fos: DeviceSimple::forces of each).
void writeForceRows(const FoamCase& fc, double t, const std::vector<FvForceSums>& fos, std::vector<Files>& ff)
{
    for (std::size_t k = 0; k < fc.forces.size(); ++k) {
        const FvForceSums& x = fos[k];
        if (fc.forces[k].coeffs) {
            ff[k].a << coeffRow(t, fvForceCoeffRow(fc.forces[k], x.fp, x.fv, x.mp, x.mv), fc.writePrecision);
            continue;
        }
        ff[k].a << forceRow(t, x.fp, x.fv);
        ff[k].b << forceRow(t, x.mp, x.mv);
    }
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
    if (auto r = writeFoamField(dir, "phi", 1, true, "[0 3 -1 0 0 0 0]", phi, pf, prec); !r) return r;
    if (st.meshPhi.empty()) return {};
    // Moving mesh: what pimpleFoam writes beside the fields -- meshPhi, Uf, polyMesh/points.
    // (phi above is the flux RELATIVE to the mesh motion.)
    std::vector<double> mphi(phi.size(), 0.0), Uf(3 * phi.size(), 0.0);
    for (std::size_t f = 0; f < nI; ++f) {
        const std::size_t src = std::size_t(s.internalSrc[f]);
        const double sg = s.srcCell[std::size_t(s.own[f])] < s.srcCell[std::size_t(s.nei[f])] ? 1.0 : -1.0;
        mphi[src] = sg * st.meshPhi[f];
        for (std::size_t q = 0; q < 3; ++q) Uf[3 * src + q] = st.Uf[q * nI + f];
    }
    std::vector<FoamPatchOut> pm, pv;
    for (std::size_t k = 0; k < L.c.patches.size(); ++k) {
        const FvPatchBc& bc = L.c.patches[k];
        FoamPatchOut a{bc.name, {}, {}}, b{bc.name, {}, {}};
        if (bc.meshType == FoamPatchType::Empty) a.constraint = b.constraint = "empty";
        else
            for (const int bf : s.patchFaces[k]) {
                const std::size_t bb = std::size_t(bf);
                a.values.push_back(st.meshPhiB[bb]);
                for (std::size_t q = 0; q < 3; ++q) b.values.push_back(st.UfB[q * nB + bb]);
            }
        pm.push_back(std::move(a));
        pv.push_back(std::move(b));
    }
    if (auto r = writeFoamField(dir, "meshPhi", 1, true, "[0 3 -1 0 0 0 0]", mphi, pm, prec); !r) return r;
    if (auto r = writeFoamField(dir, "Uf", 3, true, "[0 1 -1 0 0 0 0]", Uf, pv, prec); !r) return r;
    return writeFoamPoints(dir, L.pm.px, L.pm.py, L.pm.pz, prec);
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
    if (auto r = buildDiskSetup(L.pm, L.fm, fc, *setup); !r) return fail(3, fc.dir + ": " + r.error());
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
    if (coupled && !fc.disks.empty())
        return fail(2, "--fv-coupled: fvOptions actuationDiskSource is not built in the coupled prototype");
    const std::string label =
        std::format("[FV path, body-fitted, {}, {}] incompressible SIMPLE{} (OpenFOAM v2606 simpleFoam algorithm), "
                    "nu {:g}{}{}{}",
                    physics, fvPrecisionName(num.precision), fc.simple.consistent ? "C" : "", fc.nu, wallDist,
                    (mrf ? std::string(" [MRF frozen rotor: APPROXIMATION, steady rotating frame]") : std::string()) +
                        (fc.disks.empty() ? std::string()
                                          : std::format(" [{} actuator disk{}: fvOptions actuationDiskSource Froude, a "
                                                        "MODEL momentum source, no blades, no swirl]",
                                                        fc.disks.size(), fc.disks.size() > 1 ? "s" : "")),
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
    std::ofstream fStats;
    std::vector<Files> fForces, fDisks(fc.disks.size());
    const auto openIn = [&](const std::string& name, const char* file, std::ofstream& f) {
        return openPost(out, name, file, f);
    };
    if (auto r = openForceFiles(out, fc, fForces); !r) return std::unexpected(r.error());
    const auto diskPrec = [&](const FvDisk& d) { return d.writePrecision >= 0 ? d.writePrecision : fc.writePrecision; };
    for (std::size_t k = 0; k < fc.disks.size(); ++k) {
        if (!fc.disks[k].writeToFile) continue;
        if (auto r = openIn(fc.disks[k].name, "actuationDiskSource.dat", fDisks[k].a); !r)
            return std::unexpected(r.error());
        fDisks[k].a << diskHeader(diskPrec(fc.disks[k]));
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
    if (!sp.quiet)
        for (std::size_t k = 0; k < s.disks.size(); ++k) { // OpenFOAM: "- selected N cell(s) with volume V"
            std::string mon;
            for (const std::int64_t c : s.disks[k].monitorSrc) mon += std::format(" {}", c);
            for (const int h : s.disks[k].monitorHolders)
                if (h > 1)
                    mon += std::format(" (a monitor point lies on a face / edge / vertex of {} cells: the lowest index is "
                                       "taken, as serial simpleFoam; a decomposed run takes one cell per processor)", h);
            std::printf("# actuationDiskSource %s: selected %zu cell(s) with volume %.10g; monitor cell(s)%s\n",
                        fc.disks[k].name.c_str(), s.disks[k].cells.size(), s.disks[k].V, mon.c_str());
        }
    RunStats rs;
    double cumCont = 0, msLoop = 0;
    const char* uSolver = "BiCGStab";
    const char* pSolver = "AMG-PCG";
    // floor-limited / all solves: U components, p, turbulence scalars
    std::array<long, 3> nFloor{}, nSolves{};
    auto mark = [](const FvSolveLog& l) { return l.floorLimited ? ", floor-limited" : ""; };
    FvForceSums lastForce{};
    for (int it = 1; it <= N; ++it) {
        if (stopRequested()) {
            std::fprintf(stderr, "dns: %s at iteration %d\n", stopReason(), it);
            break;
        }
        const auto ti = std::chrono::steady_clock::now();
        const bool cIt = coupled && it > sp.coupledAfter;
        auto lg = cIt ? ds->coupledIterate(sp.coupled) : ds->iterate();
        if (!lg) return fail(7, std::format("iteration {}: {}", it, lg.error()));
        // fo: the first forces-type object's sums (stats columns, the cd line)
        FvForceSums fo{};
        std::vector<FvForceSums> fos(fc.forces.size());
        for (std::size_t k = 0; k < fc.forces.size(); ++k) {
            auto f = ds->forces(int(k));
            if (!f) return fail(7, f.error());
            fos[k] = *f;
        }
        if (!fos.empty()) lastForce = fo = fos[0];
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
        const double tNow = it * fc.deltaT;
        writeForceRows(fc, tNow, fos, fForces);
        for (std::size_t k = 0; k < fc.disks.size() && k < lg->disks.size(); ++k) {
            const FvDisk& d = fc.disks[k];
            if (d.writeToFile && tNow >= d.writeStart && tNow <= d.writeEnd)
                fDisks[k].a << diskRow(tNow, lg->disks[k], d.dir, diskPrec(d));
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
    for (auto* fl : {&fForces, &fDisks})
        for (Files& f : *fl) {
            if (f.a.is_open()) f.a.close();
            if (f.b.is_open()) f.b.close();
            if (!f.a || !f.b) return fail(4, "write failed on the postProcessing .dat files");
        }
    if (sp.stats) {
        fStats.close();
        if (!fStats) return fail(4, sp.outPrefix + "_stats.csv: write failed");
    }
    rs.msPerStep = rs.steps ? msLoop / double(rs.steps) : 0.0;
    if (!fc.forces.empty() && sp.refArea > 0) {
        const double q = 0.5 * sp.refArea;
        std::printf("cd %.8g cdp %.8g cdf %.8g cl %.8g (rho 1, U 1, A %.8g)\n", (lastForce.fp[0] + lastForce.fv[0]) / q,
                    lastForce.fp[0] / q, lastForce.fv[0] / q, (lastForce.fp[1] + lastForce.fv[1]) / q, sp.refArea);
    }
    return rs;
}

std::expected<RunStats, RunError> runFvPimple(const FvRunSpec& sp, FvLoaded& L)
{
    const FoamCase& fc = L.c;
    if (!fc.pimple.on) return fail(3, fc.dir + ": not a pimpleFoam case");
    if (sp.coupledAfter >= 0) return fail(2, "--fv-coupled: the coupled prototype is simpleFoam's (pimpleFoam case)");
    // The matrix graph: the mesh's, plus the cyclicAMI couplings (rebuilt with the AMI at every mesh motion).
    auto graph = buildFvGraph(L.pm, L.fm);
    if (!graph) return fail(3, fc.dir + ": " + graph.error());
    FvGraph& G = *graph;
    auto setup = buildSimpleSetup(L.fm, fc, G.g, G.pat, &G.ami);
    if (!setup) return fail(3, fc.dir + ": " + setup.error());
    const FvRas& ras = fc.ras;
    const bool turb = ras.model != FvTurbModel::Laminar, sst = ras.model == FvTurbModel::KOmegaSST;
    if (turb)
        if (auto r = buildRasSetup(L.pm, L.fm, fc, *setup, &G.ami); !r) return fail(3, fc.dir + ": " + r.error());
    // The setup is rebuilt in place on every mesh update (moving mesh); its topology never changes.
    const FvSimpleSetup& s = *setup;
    const FvPimpleCtl& pc = fc.pimple;
    const FvNumerics& num = sp.num;
    std::optional<FvMeshMotion> motion;
    if (fc.motion.on) {
        auto mm = FvMeshMotion::make(L);
        if (!mm) return fail(3, fc.dir + ": " + mm.error());
        motion.emplace(std::move(*mm));
    }
    const std::string meshLabel =
        !motion ? std::string("static mesh")
                : std::format("MOVING mesh: solid-body rotation of {}, {} points, omega {:g} rad/s, correctPhi "
                              "{}; phi written RELATIVE to the mesh",
                              G.ami.empty() ? std::string("the whole mesh")
                                            : std::format("{} cellZone(s) behind cyclicAMI (faceAreaWeightAMI, recomputed "
                                                          "at every mesh update)",
                                                          fc.motion.bodies.size()),
                              motion->movedPoints(), fc.motion.bodies.front().omega, pc.correctPhi ? "yes" : "no");
    const std::string label =
        std::format("[FV path, body-fitted, {}, {}] incompressible PIMPLE (OpenFOAM v2606 pimpleFoam algorithm, {}, "
                    "Euler; nOuterCorrectors {}, nCorrectors {}, nNonOrthogonalCorrectors {}), nu {:g}{}",
                    !turb ? std::string("laminar")
                          : std::format("RANS {} (OpenFOAM v2606), URANS (time-accurate)", fvTurbModelName(ras.model)),
                    fvPrecisionName(num.precision), meshLabel, pc.nOuter, pc.nCorr, fc.simple.nNonOrthCorr, fc.nu,
                    !turb ? "" : ras.exactDist ? ", wallDist exactDistance" : ", wallDist meshWave");

    // --- phase 3c: every output before the first device allocation ---
    const fs::path out(sp.outPrefix);
    std::error_code ec;
    fs::create_directories(out, ec);
    if (ec) return fail(4, out.string() + ": cannot create (" + ec.message() + ")");
    std::vector<Files> fForces;
    if (auto r = openForceFiles(out, fc, fForces); !r) return std::unexpected(r.error());
    std::ofstream fStats;
    if (sp.stats) {
        const std::string path = sp.outPrefix + "_stats.csv";
        fStats.open(path, std::ios::trunc);
        if (!fStats) return fail(4, path + ": cannot create");
        // U*0, p0: the step's first outer corrector (its first p solve); pFinal: the last p solve's final
        // residual; turbulence: the last corrector that ran it
        fStats << "# " << label << "\nstep,time,deltaT,coMean,coMax,Ux0,Uy0,Uz0,p0,pFinal,contLocal,contGlobal,Fx,Fy,Fz,"
                                   "Mx,My,Mz,ms";
        if (turb) fStats << (sst ? ",omega0,k0" : ",nuTilda0");
        fStats << "\n";
    }

    // --- device ---
    std::size_t memFree0 = 0, memTotal = 0; // device memory this process takes = the free-memory delta (bench.cpp)
    const bool memOk = hipMemGetInfo(&memFree0, &memTotal) == hipSuccess;
    const auto t0 = std::chrono::steady_clock::now();
    auto ds = DeviceSimple::make(s, fc, G.g, num);
    if (!ds) return fail(6, ds.error());
    const double tSetup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!sp.quiet)
        std::printf("# %s\n# FV path: %d cells, %d internal faces, %d boundary faces; device setup %.3f s\n", label.c_str(),
                    s.nC, s.nI, s.nB, tSetup);
    // A linear solve that ran into its maxIter is OpenFOAM's rule (not an error) but never silent: stderr, by
    // name, with the entry that applied (U | UFinal, p | pFinal, pcorr | pcorrFinal, k | kFinal, omega | omegaFinal).
    long maxIterStops = 0;
    auto atMaxIter = [&](const FvSolveLog& x, const FvSolverCtl& ctl, const char* name, long step, std::size_t outer) {
        if (ctl.maxIter <= 0 || x.iterations < ctl.maxIter) return;
        ++maxIterStops;
        std::fprintf(stderr, "dns: time step %ld, PIMPLE iteration %zu: %s reached maxIter %d (initial residual %.6g, "
                             "final %.6g, tolerance %.6g, relTol %.6g)\n",
                     step, outer + 1, name, ctl.maxIter, x.initial, x.final, ctl.tolerance, ctl.relTol);
    };
    double msHostMesh = 0, msDevMesh = 0;
    // OpenFOAM's AMI report (AMIInterpolation::normaliseWeights), printed at every AMI calculation.
    auto amiReport = [&] {
        if (sp.quiet) return;
        for (const FvAmiSetup::Report& r : s.ami.report) {
            std::printf("AMI: Patch source sum(weights) min:%.8g max:%.8g average:%.8g\n", r.srcSum[0], r.srcSum[1],
                        r.srcSum[2]);
            std::printf("AMI: Patch target sum(weights) min:%.8g max:%.8g average:%.8g\n", r.tgtSum[0], r.tgtSum[1],
                        r.tgtSum[2]);
            if (r.srcLow || r.tgtLow)
                std::printf("# cyclicAMI %s / %s: %d of %d source and %d of %d target faces have a weight sum below "
                            "0.2 (no lowWeightCorrection: an uncovered face's neighbour value is 0)\n",
                            r.src.c_str(), r.tgt.c_str(), r.srcLow, r.srcFaces, r.tgtLow, r.tgtFaces);
        }
    };
    amiReport();
    FoamClock clock(pc.time);
    // pimpleFoam.C: CourantNo.H + setInitialDeltaT.H before and after turbulence->validate() (make did it).
    for (int k = 0; k < 2; ++k) {
        auto co = ds->courant(clock.deltaT());
        if (!co) return fail(7, co.error());
        if (!sp.quiet) std::printf("Courant Number mean: %.9g max: %.9g\n", co->mean, co->max);
        clock.setInitialDeltaT(co->max);
    }
    if (!sp.quiet) std::printf("\nStarting time loop\n\n");
    RunStats rs;
    double cumCont = 0, msLoop = 0;
    bool wroteLast = false;
    while (clock.running()) {
        if (sp.maxIters > 0 && clock.index() >= sp.maxIters) break;
        if (stopRequested()) {
            std::fprintf(stderr, "dns: %s at time step %ld\n", stopReason(), clock.index() + 1);
            break;
        }
        const auto ti = std::chrono::steady_clock::now();
        auto co = ds->courant(clock.deltaT());
        if (!co) return fail(7, co.error());
        clock.setDeltaT(co->max);
        const double dt = clock.deltaT();
        const bool isWrite = clock.advance();
        // Moving mesh: mesh.update() to the new time -- the points, the FaceMesh and the setup on them (host),
        // applied by the device at the step's first outer corrector.
        std::optional<FvMeshMove> mv;
        double msHost = 0;
        if (motion) {
            const auto th = std::chrono::steady_clock::now();
            // cyclicAMI: the host part of the device's matrix rebuild runs beside the wall distance
            const FvMeshMotion::Beside beside = [&](const FvSimpleSetup& ns, const FvGraph& ng) {
                ds->prepareMatrices(ns, ng.g);
            };
            auto x = motion->update(clock.value(), dt, L, G, *setup, G.ami.empty() ? FvMeshMotion::Beside{} : beside);
            if (!x) return fail(7, std::format("time step {} (t = {}): {}", clock.index(), clock.timeName(), x.error()));
            mv.emplace(std::move(*x));
            msHost = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th).count();
            if (!G.ami.empty()) amiReport();
        }
        const FvMovedMesh moved{&*setup, mv ? &*mv : nullptr, motion && !G.ami.empty() ? &G.g : nullptr};
        auto lg = ds->timeStep(dt, motion ? &moved : nullptr);
        if (!lg) return fail(7, std::format("time step {} (t = {}): {}", clock.index(), clock.timeName(), lg.error()));
        std::vector<FvForceSums> fos(fc.forces.size());
        for (std::size_t k = 0; k < fc.forces.size(); ++k) {
            auto f = ds->forces(int(k));
            if (!f) return fail(7, f.error());
            fos[k] = *f;
        }
        (void)hipDeviceSynchronize();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti).count();
        msLoop += ms;
        msHostMesh += msHost;
        msDevMesh += lg->outer.empty() ? 0.0 : lg->outer.front().msSetup;
        for (std::size_t k = 0; k < lg->outer.size(); ++k) {
            const FvIterLog& q = lg->outer[k];
            const bool fin = k + 1 == lg->outer.size();
            const long n = clock.index();
            for (std::size_t i = 0; i < q.pcorr.size(); ++i)
                atMaxIter(q.pcorr[i], i + 1 == q.pcorr.size() ? pc.solverPcorrFinal : pc.solverPcorr, "pcorr", n, k);
            for (int j = 0; j < 3; ++j)
                if (j != s.emptyAxis)
                    atMaxIter(q.U[std::size_t(j)], fin ? pc.solverUFinal : fc.solverU, j == 0 ? "Ux" : j == 1 ? "Uy" : "Uz", n,
                              k);
            for (std::size_t i = 0; i < q.p.size(); ++i)
                atMaxIter(q.p[i], i + 1 == q.p.size() ? pc.solverPFinal : fc.solverP, "p", n, k);
            if (sst && q.nT == 2) {
                atMaxIter(q.T[0], fin ? pc.solverOmegaFinal : ras.solverOmega, "omega", n, k);
                atMaxIter(q.T[1], fin ? pc.solverKFinal : ras.solverK, "k", n, k);
            }
        }
        if (!sp.quiet) {
            std::string o = std::format("Courant Number mean: {:.9g} max: {:.9g}\n", co->mean, co->max);
            if (pc.time.adjust) o += std::format("deltaT = {:.9g}\n", dt);
            o += std::format("Time = {}\n\n", clock.timeName());
            for (std::size_t k = 0; k < lg->outer.size(); ++k) {
                const FvIterLog& q = lg->outer[k];
                o += std::format("PIMPLE: iteration {}\n", k + 1);
                for (const auto& x : q.pcorr) // correctPhi.H (moving mesh)
                    o += std::format("{}:  Solving for pcorr, Initial residual = {:.6g}, Final residual = {:.6g}, No "
                                     "Iterations {}{}\n",
                                     "AMG-PCG", x.initial, x.final, x.iterations,
                                     x.floorLimited ? ", floor-limited" : "");
                if (q.contCorr) {
                    cumCont += (*q.contCorr)[1];
                    o += std::format("time step continuity errors : sum local = {:.6g}, global = {:.6g}, cumulative = "
                                     "{:.6g}\n",
                                     (*q.contCorr)[0], (*q.contCorr)[1], cumCont);
                }
                if (q.meshCo)
                    o += std::format("Mesh Courant Number mean: {:.9g} max: {:.9g}\n", (*q.meshCo)[0], (*q.meshCo)[1]);
                for (int j = 0; j < 3; ++j) {
                    if (j == s.emptyAxis) continue;
                    const auto& u = q.U[std::size_t(j)];
                    o += std::format("BiCGStab:  Solving for U{}, Initial residual = {:.6g}, Final residual = {:.6g}, No "
                                     "Iterations {}{}\n",
                                     "xyz"[j], u.initial, u.final, u.iterations, u.floorLimited ? ", floor-limited" : "");
                }
                const std::size_t perCorr = std::size_t(fc.simple.nNonOrthCorr) + 1;
                for (std::size_t i = 0; i < q.p.size(); ++i) {
                    o += std::format("{}:  Solving for p, Initial residual = {:.6g}, Final residual = {:.6g}, No "
                                     "Iterations {}{}\n",
                                     "AMG-PCG", q.p[i].initial, q.p[i].final, q.p[i].iterations,
                                     q.p[i].floorLimited ? ", floor-limited" : "");
                    if ((i + 1) % perCorr != 0) continue;
                    const auto& ce = q.cont[i / perCorr];
                    cumCont += ce[1];
                    o += std::format("time step continuity errors : sum local = {:.6g}, global = {:.6g}, cumulative = "
                                     "{:.6g}\n",
                                     ce[0], ce[1], cumCont);
                }
                const char* names[2] = {sst ? "omega" : "nuTilda", "k"};
                const int boundIdx[2] = {sst ? 1 : 0, 0};
                for (int t = 0; t < q.nT; ++t) {
                    const auto& x = q.T[std::size_t(t)];
                    o += std::format("BiCGStab:  Solving for {}, Initial residual = {:.6g}, Final residual = {:.6g}, No "
                                     "Iterations {}{}\n",
                                     names[t], x.initial, x.final, x.iterations, x.floorLimited ? ", floor-limited" : "");
                    if (q.bounded[std::size_t(boundIdx[t])]) o += std::format("bounding {}\n", names[t]);
                }
            }
            if (motion)
                o += std::format("# mesh update: host {:.1f} ms, device {:.1f} ms; time step {:.1f} ms\n", msHost,
                                 lg->outer.empty() ? 0.0 : lg->outer.front().msSetup, ms);
            o += std::format("ExecutionTime = {:.4f} s\n\n", msLoop / 1000.0);
            std::fputs(o.c_str(), stdout);
        }
        const double tNow = clock.value();
        writeForceRows(fc, tNow, fos, fForces);
        if (sp.stats) {
            const FvIterLog& a = lg->outer.front();
            const FvIterLog& z = lg->outer.back();
            const FvForceSums fo = fos.empty() ? FvForceSums{} : fos[0];
            fStats << std::format("{},{:.17g},{:.17g},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
                                  clock.index(), tNow, dt, co->mean, co->max, a.U[0].initial, a.U[1].initial,
                                  a.U[2].initial, a.p.front().initial, z.p.back().final, z.contLocal, z.contGlobal);
            for (int j = 0; j < 3; ++j) fStats << std::format(",{:.12e}", fo.fp[std::size_t(j)] + fo.fv[std::size_t(j)]);
            for (int j = 0; j < 3; ++j) fStats << std::format(",{:.12e}", fo.mp[std::size_t(j)] + fo.mv[std::size_t(j)]);
            fStats << std::format(",{:.3f}", ms);
            for (int t = 0; t < z.nT; ++t) fStats << std::format(",{:.9e}", z.T[std::size_t(t)].initial);
            fStats << "\n";
        }
        rs.steps = clock.index();
        wroteLast = isWrite;
        if (isWrite) {
            auto st = ds->state();
            if (!st) return fail(7, st.error());
            if (auto w = writeTime(out, clock.timeName(), L, s, *st); !w) return fail(4, w.error());
        }
    }
    if (rs.steps > 0 && !wroteLast) { // the end state, as the simpleFoam path writes it
        auto st = ds->state();
        if (!st) return fail(7, st.error());
        if (auto w = writeTime(out, clock.timeName(), L, s, *st); !w) return fail(4, w.error());
    }
    for (Files& f : fForces) {
        if (f.a.is_open()) f.a.close();
        if (f.b.is_open()) f.b.close();
        if (!f.a || !f.b) return fail(4, "write failed on the postProcessing .dat files");
    }
    if (sp.stats) {
        fStats.close();
        if (!fStats) return fail(4, sp.outPrefix + "_stats.csv: write failed");
    }
    rs.msPerStep = rs.steps ? msLoop / double(rs.steps) : 0.0;
    if (!sp.quiet && rs.steps > 0) {
        if (motion)
            std::printf("# per time step: %.1f ms = mesh update host %.1f + device %.1f + the rest (solves) %.1f\n",
                        rs.msPerStep, msHostMesh / double(rs.steps), msDevMesh / double(rs.steps),
                        (msLoop - msHostMesh - msDevMesh) / double(rs.steps));
        std::size_t memFree1 = 0;
        if (memOk && hipMemGetInfo(&memFree1, &memTotal) == hipSuccess)
            std::printf("# device memory: %.0f MiB (free-memory delta of this process)\n",
                        memFree0 > memFree1 ? double(memFree0 - memFree1) / 1048576.0 : 0.0);
        std::ifstream status("/proc/self/status"); // host peak RSS (Linux; absent: no line)
        for (std::string line; std::getline(status, line);)
            if (line.starts_with("VmHWM:")) std::printf("# host peak RSS: %ld MiB\n", std::atol(line.c_str() + 6) / 1024);
        std::printf("# linear solves stopped at maxIter: %ld\n", maxIterStops);
    }
    return rs;
}

} // namespace dns::gpu
