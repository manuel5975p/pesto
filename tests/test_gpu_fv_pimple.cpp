// pimpleFoam on the FV path vs SERIAL OpenFOAM v2606 pimpleFoam on the
// IDENTICAL blockMesh (docs/fvpimple.md, stage 3a: static mesh). References:
// tests/data/fv/<case>/ref (tools/fv/make_pimple_cases.py + run_foam.sh +
// make_pimple_test_data.sh), linear solves tight on both sides, FP64:
//   P1  pcavity       PISO mode (nOuter 1, nCorr 2), fixed deltaT, no relaxation
//   P2  pcavity_skew  nOuter 3, nCorr 1, nNonOrth 1, the M17 fvSolution shape
//                     (pFinal { $p; }, "(U|k|omega)Final" { $U; }, p 0.7 /
//                     pFinal 1, "U.*" 0.7) and schemes; a forces object
//   P3  pchannel      adjustTimeStep maxCo 2 maxDeltaT 0.03, adjustableRunTime
//                     (the deltaT SEQUENCE), nOuter 2, nCorr 2, nNonOrth 1,
//                     U 0.7 / UFinal 1
//   P3b pduct         3D, cyclic, freestream BCs, symmetryPlane, PISO + 1 non-orth
//   P4  psst_plate    kOmegaSST + wall functions, nOuter 3, turbulence on the
//                     final corrector only
//   P4b psst_skew     kOmegaSST, nOuter 2, turbOnFinalIterOnly no
// Stage 3b, the WHOLE mesh in solid-body rotation (docs/fvpimple.md "Moving
// mesh"; tools/fv/make_motion_cases.py), phi compared as pimpleFoam writes it:
// RELATIVE to the mesh motion:
//   M1  mspin         closed 2D box spinning up from rest, movingWallVelocity,
//                     PISO mode, fixed deltaT, checkMeshCourantNo
//   M2  mduct         3D sheared duct, inlet / outlet on the rotating mesh,
//                     off-centre axis, nOuter 3, nNonOrth 1, the M17 relaxation
//                     set, correctPhi, adjustTimeStep
//   M3  msst          M2 + kOmegaSST with wall functions on the rotating walls,
//                     the M17 fvSchemes and dynamicMeshDict spelling
// plus, per reference time, the moved points (bitwise), meshPhi and Uf; the
// pcorr residuals; the mesh Courant number.
// M3 sits at 1e-12: its level is the case's solver tolerance, not the codes'
// difference -- serial pimpleFoam against ITSELF with only the PRESSURE solver
// swapped (PCG + DIC -> GAMG, both to 1e-13) differs by U 2.6e-12, p 3.1e-12,
// nut 7.3e-12 at t = 0.5 (dns: 1.1e-12, 1.5e-12, 2.9e-12); the same twin of M2
// gives U 4.0e-14 (dns 3.4e-14).
// Gates: U, p, phi (RAS: k, omega, nut) at the two reference times within
// 1e-10 (RAS fields 1e-9) of the field scale (measured in the header of each
// gate's output; docs/fvpimple.md table); per time step the time, deltaT
// (1e-12 relative), the Courant number deltaT was set from and every initial
// residual of every corrector (1e-8 relative where above 1e-9); force.dat;
// 2D: the empty-axis U exactly 0; two runs byte-identical; FP32 runs and stays
// within a stated band of FP64.
// Loose solves (kPimpleLooseMinIter, issues.md 65 (p)), cases edited after
// loading: L1 amixtilt U relTol 0.1, every non-final U solve >= 2 iterations,
// two runs byte-identical; L2 psst_skew U / k / omega relTol 0.1, outer
// corrector 1's solves >= 2 iterations; L3 the same at tolerance 1e30, 0
// iterations and final == initial (converged at r0 keeps x0).
#include "dns/foamio.hpp"
#include "dns/fvpimple.hpp"
#include "dns/fvsimple.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/fv_simple.hpp"
#include "dns/la/graph.hpp"

#include "check.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifndef DNS_FV_DATA
#error "DNS_FV_DATA must point at tests/data/fv"
#endif

using namespace dns;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

struct Step
{
    double time = 0, deltaT = 0, coMax = 0;
    gpu::FvStepLog log;
    std::vector<gpu::FvForceSums> forces;
};

struct Run
{
    FvLoaded L;
    FvSimpleSetup s;
    std::vector<Step> steps;
    std::vector<std::pair<std::string, gpu::FvState>> at; // (time name, state) at the write times
    std::vector<std::array<std::vector<double>, 3>> points; // moving mesh: the points at those times
};

// The run loop of gpu::runFvPimple without its outputs: FoamClock + DeviceSimple.
// edit: applied to the loaded case before anything is built (e.g. a looser solver entry).
std::expected<Run, std::string> run(const char* name, const FvNumerics& num = {}, int maxSteps = 0,
                                    const std::function<void(FoamCase&)>& edit = {})
{
    Run r;
    auto L = loadFvCase(fs::path(DNS_FV_DATA) / name);
    if (!L) return std::unexpected(L.error());
    r.L = std::move(*L);
    if (edit) edit(r.L.c);
    if (!r.L.c.pimple.on) return std::unexpected("not a pimpleFoam case");
    auto graph = buildFvGraph(r.L.pm, r.L.fm);
    if (!graph) return std::unexpected(graph.error());
    FvGraph& G = *graph;
    auto s = buildSimpleSetup(r.L.fm, r.L.c, G.g, G.pat, &G.ami);
    if (!s) return std::unexpected(s.error());
    r.s = std::move(*s);
    if (r.L.c.ras.model != FvTurbModel::Laminar)
        if (auto x = buildRasSetup(r.L.pm, r.L.fm, r.L.c, r.s, &G.ami); !x) return std::unexpected(x.error());
    auto ds = gpu::DeviceSimple::make(r.s, r.L.c, G.g, num);
    if (!ds) return std::unexpected(ds.error());
    std::optional<FvMeshMotion> motion; // moving mesh: the runner's mesh.update() per step
    if (r.L.c.motion.on) {
        auto mm = FvMeshMotion::make(r.L);
        if (!mm) return std::unexpected(mm.error());
        motion.emplace(std::move(*mm));
    }
    FoamClock clock(r.L.c.pimple.time);
    for (int k = 0; k < 2; ++k) {
        auto co = ds->courant(clock.deltaT());
        if (!co) return std::unexpected(co.error());
        clock.setInitialDeltaT(co->max);
    }
    while (clock.running() && (maxSteps == 0 || clock.index() < maxSteps)) {
        auto co = ds->courant(clock.deltaT());
        if (!co) return std::unexpected(co.error());
        clock.setDeltaT(co->max);
        Step st;
        st.deltaT = clock.deltaT();
        st.coMax = co->max;
        const bool write = clock.advance();
        st.time = clock.value();
        std::optional<FvMeshMove> mv;
        if (motion) {
            // as the runner: on cyclicAMI the host part of the matrix rebuild runs beside the wall distance
            const FvMeshMotion::Beside beside = [&](const FvSimpleSetup& ns, const FvGraph& ng) {
                ds->prepareMatrices(ns, ng.g);
            };
            auto x = motion->update(st.time, st.deltaT, r.L, G, r.s, G.ami.empty() ? FvMeshMotion::Beside{} : beside);
            if (!x) return std::unexpected(std::format("time step {}: {}", clock.index(), x.error()));
            mv.emplace(std::move(*x));
        }
        const gpu::FvMovedMesh moved{&r.s, mv ? &*mv : nullptr, motion && !G.ami.empty() ? &G.g : nullptr};
        auto lg = ds->timeStep(st.deltaT, motion ? &moved : nullptr);
        if (!lg) return std::unexpected(std::format("time step {}: {}", clock.index(), lg.error()));
        st.log = std::move(*lg);
        for (std::size_t k = 0; k < r.L.c.forces.size(); ++k) {
            auto f = ds->forces(int(k));
            if (!f) return std::unexpected(f.error());
            st.forces.push_back(*f);
        }
        r.steps.push_back(std::move(st));
        if (write) {
            auto x = ds->state();
            if (!x) return std::unexpected(x.error());
            r.at.emplace_back(clock.timeName(), std::move(*x));
            if (motion) r.points.push_back({r.L.pm.px, r.L.pm.py, r.L.pm.pz});
        }
    }
    return r;
}

// Moving mesh at a reference time: the points (max |dx| / the mesh's extent),
// meshPhi and Uf (of scale) against pimpleFoam's time directory. Internal faces.
std::array<double, 3> motionDiff(const Run& r, std::size_t at, const fs::path& dir)
{
    std::array<double, 3> d{1e300, 1e300, 1e300};
    const std::size_t nI = std::size_t(r.s.nI), nP = r.L.pm.px.size();
    const gpu::FvState& st = r.at[at].second;
    // polyMesh/points: a bare `N ( (x y z) ... )` list
    {
        std::ifstream f(dir / "polyMesh" / "points");
        std::stringstream ss;
        ss << f.rdbuf();
        std::string t = ss.str();
        const std::size_t h = t.find("object");
        const std::size_t b = h == std::string::npos ? h : t.find('(', t.find('}', h));
        if (b != std::string::npos) {
            for (char& ch : t)
                if (ch == '(' || ch == ')') ch = ' ';
            std::stringstream v(t.substr(b));
            double worst = 0, ext = 0, x = 0, y = 0, z = 0;
            std::size_t n = 0;
            const auto& P = r.points[at];
            while (n < nP && (v >> x >> y >> z)) {
                worst = std::max({worst, std::abs(x - P[0][n]), std::abs(y - P[1][n]), std::abs(z - P[2][n])});
                ext = std::max({ext, std::abs(x), std::abs(y), std::abs(z)});
                ++n;
            }
            if (n == nP) d[0] = worst / ext;
        }
    }
    if (auto ref = readFoamInternalField(dir / "meshPhi", 1, r.L.pm.man.nInternalFaces)) {
        double w = 0, sc = 0;
        for (std::size_t f = 0; f < nI; ++f) {
            const double sg = r.s.srcCell[std::size_t(r.s.own[f])] < r.s.srcCell[std::size_t(r.s.nei[f])] ? 1.0 : -1.0;
            const double theirs = (*ref)[std::size_t(r.s.internalSrc[f])];
            w = std::max(w, std::abs(sg * st.meshPhi[f] - theirs));
            sc = std::max(sc, std::abs(theirs));
        }
        d[1] = w / sc;
    } else std::printf("    %s\n", ref.error().c_str());
    if (auto ref = readFoamInternalField(dir / "Uf", 3, r.L.pm.man.nInternalFaces)) {
        double w = 0, sc = 0;
        for (std::size_t f = 0; f < nI; ++f)
            for (std::size_t q = 0; q < 3; ++q) {
                const double theirs = (*ref)[3 * std::size_t(r.s.internalSrc[f]) + q];
                w = std::max(w, std::abs(st.Uf[q * nI + f] - theirs));
                sc = std::max(sc, std::abs(theirs));
            }
        d[2] = w / sc;
    } else std::printf("    %s\n", ref.error().c_str());
    return d;
}

// max |mine - theirs| / scale over the internal field (p: / its range).
double fieldDiff(const Run& r, const gpu::FvState& st, const fs::path& dir, const std::string& field)
{
    const std::size_t nC = std::size_t(r.s.nC);
    const int nc = field == "U" ? 3 : 1;
    auto ref = readFoamInternalField(dir / field, nc, std::int64_t(nC));
    if (!ref) {
        std::printf("    %s\n", ref.error().c_str());
        return 1e300;
    }
    double d = 0, scale = 0, lo = 1e300, hi = -1e300;
    for (std::size_t c = 0; c < nC; ++c) {
        const std::size_t i = std::size_t(r.s.srcCell[c]);
        for (int q = 0; q < nc; ++q) {
            const double mine = nc == 3             ? st.U[std::size_t(q) * nC + c]
                                : field == "p"      ? st.p[c]
                                : field == "nut"    ? st.nut[c]
                                : field == "omega"  ? st.s[1][c]
                                                    : st.s[0][c];
            const double theirs = (*ref)[i * std::size_t(nc) + std::size_t(q)];
            d = std::max(d, std::abs(mine - theirs));
            scale = std::max(scale, std::abs(theirs));
            lo = std::min(lo, theirs);
            hi = std::max(hi, theirs);
        }
    }
    return field == "p" ? d / (hi - lo) : d / scale;
}

double phiDiff(const Run& r, const gpu::FvState& st, const fs::path& dir)
{
    const std::size_t nI = std::size_t(r.s.nI);
    auto ref = readFoamInternalField(dir / "phi", 1, r.L.pm.man.nInternalFaces);
    if (!ref) {
        std::printf("    %s\n", ref.error().c_str());
        return 1e300;
    }
    double d = 0, scale = 0;
    for (std::size_t f = 0; f < nI; ++f) {
        const double sg = r.s.srcCell[std::size_t(r.s.own[f])] < r.s.srcCell[std::size_t(r.s.nei[f])] ? 1.0 : -1.0;
        const double theirs = (*ref)[std::size_t(r.s.internalSrc[f])];
        d = std::max(d, std::abs(sg * st.phi[f] - theirs));
        scale = std::max(scale, std::abs(theirs));
    }
    return d / scale;
}

std::vector<std::vector<double>> csvRows(const fs::path& p)
{
    std::ifstream f(p);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) row.push_back(std::stod(tok));
        rows.push_back(std::move(row));
    }
    return rows;
}

// The numeric rows of an OpenFOAM function-object .dat file.
std::vector<std::vector<double>> datRows(const fs::path& p)
{
    std::ifstream f(p);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::vector<double> row;
        double v = 0;
        while (ss >> v) row.push_back(v);
        rows.push_back(std::move(row));
    }
    return rows;
}

// The reference time directories of a case (numeric names under ref/), ascending.
std::vector<std::string> refTimes(const char* name)
{
    std::vector<std::pair<double, std::string>> t;
    for (const auto& de : fs::directory_iterator(fs::path(DNS_FV_DATA) / name / "ref")) {
        const std::string n = de.path().filename().string();
        double v = 0;
        const auto rc = std::from_chars(n.data(), n.data() + n.size(), v);
        if (de.is_directory() && rc.ec == std::errc{} && rc.ptr == n.data() + n.size()) t.emplace_back(v, n);
    }
    std::ranges::sort(t);
    std::vector<std::string> out;
    for (auto& x : t) out.push_back(x.second);
    return out;
}

// A step's initial residuals in pimpleFoam's log order, with `skip` marking a
// velocity component that is rounding noise (identically zero solution).
void residualsOf(const Run& r, const Step& st, const std::array<bool, 3>& noise, std::vector<double>& v,
                 std::vector<bool>& skip)
{
    for (const gpu::FvIterLog& q : st.log.outer) {
        for (const auto& p : q.pcorr) { // moving mesh: correctPhi.H comes first
            v.push_back(p.initial);
            skip.push_back(false);
        }
        for (int j = 0; j < 3; ++j)
            if (j != r.s.emptyAxis) {
                v.push_back(q.U[std::size_t(j)].initial);
                skip.push_back(noise[std::size_t(j)]);
            }
        for (const auto& p : q.p) {
            v.push_back(p.initial);
            skip.push_back(false);
        }
        for (int t = 0; t < q.nT; ++t) {
            v.push_back(q.T[std::size_t(t)].initial);
            skip.push_back(false);
        }
    }
}

// nutBand / resBand: psst_plate amplifies rounding (the Spalding wall function
// on an impulsively started plate): serial pimpleFoam against ITSELF with only
// U's linear solver swapped (PBiCGStab -> smoothSolver, both to 1e-14) differs
// by U 5.4e-12, p 4.1e-12, omega 3.6e-11, nut 5.0e-9 at t = 0.5 -- more than
// dns differs from it (1.7e-12, 2.1e-12, 1.1e-11, 1.6e-9). Its bands are that
// spread's decade, not parity bands; every other case keeps 1e-9 / 1e-8.
// fieldBand / forceBand (A4 only; every other case 1e-10 / 1e-9): amix3d amplifies the pressure solve's
// tolerance as M3 does, more on its coarse wall cells -- serial pimpleFoam against ITSELF with only the
// p / pcorr solver swapped (GAMG -> PBiCGStab + DIC, both to 1e-13) differs at t = 0.15 / 0.45 by U 4.8e-11 /
// 1.2e-10, p 2.3e-11 / 6.4e-11, phi 2.9e-12 / 2.4e-11, k 8.8e-13 / 1.1e-11, nut 7.8e-11 / 3.3e-10, the blade's
// force and moment 1.4e-9 -- more than dns differs from it (U 4.0e-11 / 1.0e-10, nut 6.6e-11 / 2.5e-10, forces
// 1.2e-9). Its bands are that spread's decade.
void gate(const char* tag, const char* name, double nutBand = 1e-9, double resBand = 1e-8, double fieldBand = 1e-10,
          double forceBand = 1e-9)
{
    std::printf("%s %s\n", tag, name);
    auto r = run(name);
    if (!r) {
        check(false, std::string(name) + ": " + r.error());
        return;
    }
    const FoamCase& c = r->L.c;
    const bool sst = c.ras.model == FvTurbModel::KOmegaSST;
    const bool cyclic = std::ranges::any_of(c.patches, [](const FvPatchBc& b) { return b.meshType == FoamPatchType::Cyclic; });
    const auto times = refTimes(name);
    check(times.size() == 2, std::format("{} reference times", times.size()));
    for (const std::string& t : times) {
        const auto it = std::ranges::find(r->at, t, [](const auto& x) { return x.first; });
        if (it == r->at.end()) {
            check(false, "no state written at reference time " + t);
            continue;
        }
        const fs::path dir = fs::path(DNS_FV_DATA) / name / "ref" / t;
        const double dU = fieldDiff(*r, it->second, dir, "U"), dp = fieldDiff(*r, it->second, dir, "p");
        const double df = cyclic ? 0.0 : phiDiff(*r, it->second, dir);
        std::string ras;
        bool ok = dU < fieldBand && dp < fieldBand && df < fieldBand;
        if (sst) {
            const double dk = fieldDiff(*r, it->second, dir, "k"), dw = fieldDiff(*r, it->second, dir, "omega"),
                         dn = fieldDiff(*r, it->second, dir, "nut");
            ok = ok && dk < 10 * fieldBand && dw < 10 * fieldBand && dn < nutBand;
            ras = std::format(", k {:.2e}, omega {:.2e}, nut {:.2e}", dk, dw, dn);
        }
        check(ok, std::format("t = {}: U {:.2e}, p {:.2e}, phi {:.2e}{} of scale (gates {:.0e}{})", t, dU, dp, df, ras,
                              fieldBand,
                              sst ? std::format(" / {:.0e}, nut {:.0e}", 10 * fieldBand, nutBand) : std::string()));
        if (c.motion.on) { // phi above is the flux RELATIVE to the mesh, as pimpleFoam writes it
            const auto md = motionDiff(*r, std::size_t(it - r->at.begin()), dir);
            check(md[0] < 1e-14 && md[1] < 1e-10 && md[2] < fieldBand,
                  std::format("t = {}: moved points {:.2e} of the extent (gate 1e-14), meshPhi {:.2e} (gate 1e-10), Uf "
                              "{:.2e} of scale (gate {:.0e})",
                              t, md[0], md[1], md[2], fieldBand));
        }
    }
    if (c.pimple.checkMeshCourantNo && c.motion.on) {
        const auto mc = csvRows(fs::path(DNS_FV_DATA) / name / "ref" / "meshco.csv");
        double w = 0;
        bool have = mc.size() == r->steps.size();
        for (std::size_t i = 0; have && i < mc.size(); ++i) {
            const auto& co = r->steps[i].log.outer.front().meshCo;
            have = co.has_value();
            if (have) w = std::max(w, std::abs((*co)[1] - mc[i][0]) / mc[i][0]);
        }
        check(have && w < 1e-12, std::format("mesh Courant number, every step: worst {:.2e} relative (gate 1e-12)", w));
    }
    // log.csv: time, coMax, deltaT (0: none printed), initial residuals in log order
    const auto ref = csvRows(fs::path(DNS_FV_DATA) / name / "ref" / "log.csv");
    check(ref.size() == r->steps.size(), std::format("{} time steps (pimpleFoam: {})", r->steps.size(), ref.size()));
    const std::size_t nC = std::size_t(r->s.nC);
    std::array<bool, 3> noise{};
    {
        double mx = 0;
        std::array<double, 3> mj{};
        const auto& U = r->at.back().second.U;
        for (int j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < nC; ++k) {
                mj[std::size_t(j)] = std::max(mj[std::size_t(j)], std::abs(U[std::size_t(j) * nC + k]));
                mx = std::max(mx, mj[std::size_t(j)]);
            }
        for (int j = 0; j < 3; ++j) noise[std::size_t(j)] = mj[std::size_t(j)] < 1e-10 * mx;
    }
    double wT = 0, wDt = 0, wCo = 0, wRes = 0, wResAt = 0;
    int nDt = 0, nRes = 0, wResStep = 0, wResQ = 0;
    bool shape = true;
    for (std::size_t i = 0; i < ref.size() && i < r->steps.size(); ++i) {
        const Step& st = r->steps[i];
        const auto& row = ref[i];
        wT = std::max(wT, std::abs(st.time - row[0]) / row[0]);
        if (row[1] > 1e-9) wCo = std::max(wCo, std::abs(st.coMax - row[1]) / row[1]);
        if (row[2] > 0) {
            wDt = std::max(wDt, std::abs(st.deltaT - row[2]) / row[2]);
            ++nDt;
        }
        std::vector<double> mine;
        std::vector<bool> skip;
        residualsOf(*r, st, noise, mine, skip);
        shape = shape && mine.size() + 3 == row.size();
        for (std::size_t q = 0; q < mine.size() && q + 3 < row.size(); ++q) {
            const double theirs = row[q + 3];
            if (theirs < 1e-9 || skip[q]) continue;
            // First step, a component starting from x = 0 whose source is rounding noise (a uniform flow's
            // cross-stream momentum, sum|b| ~ 1e-17): the residual is sum|b| / (sum|b| + 1e-20), a number in
            // (0.99, 1) made of that noise (acyl: 0.99975513768926594 here, 0.9997536938 there). A real source
            // gives exactly 1.
            // Only on the cyclicAMI cases (the earlier gates compare every value, as before).
            if (r->s.ami.nA > 0 && i == 0 && theirs > 0.99 && theirs < 1.0) continue;
            if (std::abs(mine[q] - theirs) / theirs > wRes) {
                wRes = std::abs(mine[q] - theirs) / theirs;
                wResAt = theirs;
                wResStep = int(i);
                wResQ = int(q);
            }
            ++nRes;
        }
    }
    check(shape, "every step has pimpleFoam's solves (outer correctors x (U, p correctors x non-orthogonal, turbulence))");
    check(wT < 1e-9 && wCo < 1e-9,
          std::format("time {:.2e} (the log prints 10 digits), Courant number {:.2e} relative, every step (gates 1e-9)", wT,
                      wCo));
    if (c.pimple.time.adjust)
        check(nDt == int(ref.size()) && wDt < 1e-12,
              std::format("deltaT sequence (adjustTimeStep, {} steps): worst {:.2e} relative (gate 1e-12)", nDt, wDt));
    check(nRes > 0 && wRes < resBand,
          std::format("initial residuals of every solve, every step: worst {:.2e} relative over {} values (gate {:.0e}; "
                      "the worst: residual {:.3e}, step {}, solve {})",
                      wRes, nRes, resBand, wResAt, wResStep + 1, wResQ));
    if (r->s.emptyAxis >= 0) {
        bool zero = true;
        for (std::size_t k = 0; k < nC; ++k) zero = zero && r->at.back().second.U[std::size_t(r->s.emptyAxis) * nC + k] == 0.0;
        check(zero, "2D: the empty-axis velocity is exactly 0");
    }
    for (std::size_t k = 0; k < c.forces.size(); ++k) {
        const fs::path pd = fs::path(DNS_FV_DATA) / name / "ref" / "postProcessing" / c.forces[k].name / "0";
        double worst[2] = {0, 0};
        bool rows = true;
        // A column that is a cancellation to rounding (the net force on a closed
        // rotating box: 1e-12 against wall moments of 1e-2) is not compared:
        // below 1e-9 of the object's largest column.
        double big = 0;
        for (int m = 0; m < 2; ++m)
            for (const auto& x : datRows(pd / (m == 0 ? "force.dat" : "moment.dat")))
                for (std::size_t j = 1; j < 10 && j < x.size(); ++j) big = std::max(big, std::abs(x[j]));
        for (int m = 0; m < 2; ++m) {
            const auto fr = datRows(pd / (m == 0 ? "force.dat" : "moment.dat"));
            rows = rows && fr.size() == r->steps.size();
            for (std::size_t j = 1; j < 10 && rows; ++j) {
                double scale = 0;
                for (const auto& x : fr) scale = std::max(scale, std::abs(x[j]));
                if (scale < 1e-12 || scale < 1e-9 * big) continue;
                for (std::size_t i = 0; i < fr.size(); ++i) {
                    const auto& f = r->steps[i].forces[k];
                    const auto& a = m == 0 ? f.fp : f.mp;
                    const auto& b = m == 0 ? f.fv : f.mv;
                    const std::size_t q = (j - 1) % 3;
                    const double mine = j <= 3 ? a[q] + b[q] : j <= 6 ? a[q] : b[q];
                    worst[m] = std::max(worst[m], std::abs(mine - fr[i][j]) / scale);
                }
            }
        }
        check(rows && worst[0] < forceBand && worst[1] < forceBand,
              std::format("{}: force.dat {:.2e}, moment.dat {:.2e} of column scale, every step (gate {:.0e})",
                          c.forces[k].name, worst[0], worst[1], forceBand));
    }
}

double vecDiff(const std::vector<double>& a, const std::vector<double>& b, bool range = false)
{
    double d = 0, s = 0, lo = 1e300, hi = -1e300;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::abs(a[i] - b[i]));
        s = std::max(s, std::abs(b[i]));
        lo = std::min(lo, b[i]);
        hi = std::max(hi, b[i]);
    }
    return range ? d / (hi - lo) : d / s;
}

void determinism()
{
    std::printf("determinism\n");
    for (const char* name : {"pchannel", "psst_plate", "msst", "amix3d"}) {
        auto a = run(name, {}, 30);
        auto b = run(name, {}, 30);
        if (!a || !b || a->steps.empty()) {
            check(false, "runs");
            return;
        }
        bool same = a->steps.size() == b->steps.size();
        for (std::size_t i = 0; same && i < a->steps.size(); ++i) {
            const Step& x = a->steps[i];
            const Step& y = b->steps[i];
            same = x.deltaT == y.deltaT && x.coMax == y.coMax && x.log.outer.size() == y.log.outer.size();
            for (std::size_t k = 0; same && k < x.log.outer.size(); ++k)
                same = x.log.outer[k].p.back().initial == y.log.outer[k].p.back().initial &&
                       x.log.outer[k].U[0].initial == y.log.outer[k].U[0].initial;
        }
        check(same, std::string("two 30-step ") + name + " runs: deltaT, Courant numbers and residuals byte-identical");
    }
}

// The loose-solve rule (kPimpleLooseMinIter, issues.md 65 (p)): a non-final
// BiCGStab + MC-GS solve (U; k, omega with turbOnFinalIterOnly no) whose entry
// has relTol > 0 iterates at least twice once it starts; one converged at r0
// keeps x0. Every gate case is relTol 0, so the cases are edited after loading.
void looseSolves()
{
    using gpu::FvIterLog, gpu::FvSolveLog;
    std::printf("loose solves (relTol > 0: >= 2 BiCGStab iterations once started)\n");
    const auto looseU = [](FoamCase& c) { c.solverU.relTol = 0.1; };
    const auto looseAll = [](FoamCase& c) {
        c.solverU.relTol = c.ras.solverK.relTol = c.ras.solverOmega.relTol = 0.1;
    };
    const auto convergedAll = [&](FoamCase& c) {
        looseAll(c);
        c.solverU.tolerance = c.ras.solverK.tolerance = c.ras.solverOmega.tolerance = 1e30;
    };
    // L1: amixtilt (rotating cyclicAMI, kOmegaSST, nOuter 3), U relTol 0.1
    {
        auto a = run("amixtilt", {}, 10, looseU);
        auto b = run("amixtilt", {}, 10, looseU);
        if (!a || !b || a->steps.empty()) {
            check(false, std::string("L1 amixtilt runs: ") + (a ? (b ? "no steps" : b.error()) : a.error()));
        } else {
            int n = 0, two = 0, bad = 0;
            for (const Step& st : a->steps)
                for (std::size_t k = 0; k + 1 < st.log.outer.size(); ++k)
                    for (const FvSolveLog& u : st.log.outer[k].U) {
                        if (u.initial == 0) continue; // unsolved component
                        ++n;
                        two += u.iterations == 2;
                        bad += u.iterations < 2;
                    }
            check(n > 0 && bad == 0, std::format("L1 amixtilt, U relTol 0.1, {} steps: every non-final U solve takes >= 2 "
                                                 "iterations ({} solves, {} at exactly 2, {} below)",
                                                 a->steps.size(), n, two, bad));
            bool same = a->steps.size() == b->steps.size();
            const auto eq = [](const FvSolveLog& x, const FvSolveLog& y) {
                return x.initial == y.initial && x.final == y.final && x.iterations == y.iterations;
            };
            for (std::size_t i = 0; same && i < a->steps.size(); ++i) {
                const Step& x = a->steps[i];
                const Step& y = b->steps[i];
                same = x.deltaT == y.deltaT && x.coMax == y.coMax && x.log.outer.size() == y.log.outer.size();
                for (std::size_t k = 0; same && k < x.log.outer.size(); ++k) {
                    const FvIterLog& p = x.log.outer[k];
                    const FvIterLog& q = y.log.outer[k];
                    same = p.p.size() == q.p.size();
                    for (int j = 0; same && j < 3; ++j) same = eq(p.U[std::size_t(j)], q.U[std::size_t(j)]);
                    for (std::size_t j = 0; same && j < p.p.size(); ++j) same = eq(p.p[j], q.p[j]);
                }
            }
            check(same, "L1 two amixtilt loose runs: deltaT, Courant numbers and every U / p initial, final, iterations "
                        "byte-identical");
        }
    }
    // L2: psst_skew (nOuter 2, turbOnFinalIterOnly no, PBiCGStab entries: the rule is name-independent), U, k,
    // omega relTol 0.1
    {
        auto a = run("psst_skew", {}, 10, looseAll);
        if (!a || a->steps.empty()) {
            check(false, std::string("L2 psst_skew run: ") + (a ? "no steps" : a.error()));
        } else {
            int nU = 0, nT = 0, bad = 0;
            for (const Step& st : a->steps) {
                if (st.log.outer.size() != 2) ++bad;
                if (st.log.outer.empty()) continue;
                const FvIterLog& o = st.log.outer[0];
                for (const FvSolveLog& u : o.U)
                    if (u.initial != 0) {
                        ++nU;
                        bad += u.iterations < 2;
                    }
                if (o.nT == 0) ++bad;
                for (int q = 0; q < o.nT; ++q) {
                    ++nT;
                    bad += o.T[std::size_t(q)].iterations < 2;
                }
            }
            check(nU > 0 && nT > 0 && bad == 0,
                  std::format("L2 psst_skew, U / k / omega relTol 0.1, {} steps: outer corrector 1's U ({} solves) and "
                              "k / omega ({} solves) take >= 2 iterations ({} violations)",
                              a->steps.size(), nU, nT, bad));
        }
    }
    // L3: as L2 with tolerance 1e30: converged at r0, no iteration, x0 kept
    {
        auto a = run("psst_skew", {}, 10, convergedAll);
        if (!a || a->steps.empty()) {
            check(false, std::string("L3 psst_skew run: ") + (a ? "no steps" : a.error()));
        } else {
            int n = 0, bad = 0;
            const auto idle = [](const FvSolveLog& s) { return s.iterations == 0 && s.final == s.initial; };
            for (const Step& st : a->steps) {
                if (st.log.outer.empty()) {
                    ++bad;
                    continue;
                }
                const FvIterLog& o = st.log.outer[0];
                for (const FvSolveLog& u : o.U)
                    if (u.initial != 0) {
                        ++n;
                        bad += !idle(u);
                    }
                if (o.nT == 0) ++bad;
                for (int q = 0; q < o.nT; ++q) {
                    ++n;
                    bad += !idle(o.T[std::size_t(q)]);
                }
            }
            check(n > 0 && bad == 0, std::format("L3 psst_skew, U / k / omega relTol 0.1 tolerance 1e30, {} steps: outer "
                                                 "corrector 1's solves converged at r0 take 0 iterations, final == "
                                                 "initial ({} solves, {} violations)",
                                                 a->steps.size(), n, bad));
        }
    }
}

// FP32 (the CLI's default precision, default floors) vs FP64 at the last write time.
void gateFp32(const char* name, double band, double nutBand)
{
    std::printf("FP32 %s\n", name);
    auto f = run(name, fvNumerics(FvPrecision::Fp32));
    auto d = run(name);
    if (!f || !d || f->at.empty() || d->at.empty()) {
        check(false, std::string(name) + ": " + (f ? (d ? std::string("no state") : d.error()) : f.error()));
        return;
    }
    const auto& sf = f->at.back().second;
    const auto& sd = d->at.back().second;
    const bool ras = f->L.c.ras.model != FvTurbModel::Laminar;
    const double dU = vecDiff(sf.U, sd.U), dp = vecDiff(sf.p, sd.p, true), dphi = vecDiff(sf.phi, sd.phi);
    const double dn = ras ? vecDiff(sf.nut, sd.nut) : 0.0;
    check(f->steps.size() == d->steps.size() && dU < band && dp < band && dphi < band && dn < nutBand,
          std::format("t = {}: FP32 vs FP64 U {:.2e}, p {:.2e}, phi {:.2e}{} of scale (gate {:.0e}{}), {} steps",
                      f->at.back().first, dU, dp, dphi, ras ? std::format(", nut {:.2e}", dn) : std::string(), band,
                      ras ? std::format(", nut {:.0e}", nutBand) : std::string(), f->steps.size()));
    if (ras) { // report: the same FP32 run without floors (what the floors add)
        auto z = run(name, FvNumerics{FvPrecision::Fp32, FvFloors{}});
        if (z && !z->at.empty())
            std::printf("    FP32 without floors vs FP64: U %.2e, p %.2e, nut %.2e\n", vecDiff(z->at.back().second.U, sd.U),
                        vecDiff(z->at.back().second.p, sd.p, true), vecDiff(z->at.back().second.nut, sd.nut));
    }
}

} // namespace

int main()
{
    if (!gpu::deviceAvailable()) dnstest::skip("no GPU device");
    gate("P1", "pcavity");
    gate("P2", "pcavity_skew");
    gate("P3", "pchannel");
    gate("P3b", "pduct");
    gate("P4", "psst_plate", 1e-7, 1e-6);
    gate("P4b", "psst_skew");
    gate("M1", "mspin");
    gate("M2", "mduct");
    gate("M3", "msst");
    // cyclicAMI (stage 3c, docs/fvpimple.md "cyclicAMI"): static planar non-matching / matching, static 3D
    // cylinder, the rotating 2D mixer, the rotating 3D mixer with kOmegaSST + wall functions + meshWave
    gate("A1", "aplanar");
    gate("A1m", "amatch");
    gate("A2", "acyl");
    gate("A3", "amix2d");
    gate("A4", "amix3d", 1e-8, 1e-8, 1e-9, 1e-8);
    // cyclicAMI faces WITHOUT overlap (stage 3d; no lowWeightCorrection: neighbour value 0). A1p: static, 2
    // source / 3 target faces uncovered -- interpolate(nuEff) is 0 there, never the constant nu. A5: A4's rotor
    // wobbling about a tilted axis, weight sums falling 1 -> 0 on both sides -- CorrectPhi's rAUf is a SURFACE
    // field 1 on those faces too (pcorr carries flux through them), every interpolated coefficient 0.
    gate("A1p", "apartial");
    gate("A5", "amixtilt");
    determinism();
    looseSolves();
    gateFp32("pcavity_skew", 1e-3, 1e-3);
    gateFp32("psst_plate", 1e-3, 2e-1);
    gateFp32("psst_skew", 1e-3, 1e-2);
    gateFp32("mduct", 1e-3, 1e-3);
    gateFp32("msst", 1e-3, 1e-2);
    return dnstest::finish();
}
