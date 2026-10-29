// The incompressible FV solver vs OpenFOAM v2606 simpleFoam on the IDENTICAL
// blockMesh (docs/fv.md, gate IF1). References: tests/data/fv/<case>/ref
// (tools/fv/make_cases.py + run_foam.sh + make_test_data.sh), linear solves
// converged tight on both sides, so every SIMPLE iteration is the algorithm's:
//   IF1a cavity (orthogonal, SIMPLE, p relax 0.3, closed: setReference)
//   IF1b cavity_skew (26.6 deg, SIMPLEC, 2 non-orthogonal correctors)
//   IF1c channel (inletOutlet, fixedValue p, slip, linearUpwind, cellLimited)
//   IF1d duct (3D, non-orthogonal, cyclic z, freestreamVelocity/Pressure,
//        symmetryPlane: the constraint patch's reflected H and grad(U))
//   IF1e cavity_lim (cavity_skew with `limited corrected 0.33` laplacians
//        and snGrad: limitedSnGrad, 4% off the unlimited solution)
//   IF1f mrf2 (MRFZoneList: two ADJACENT zones -- zone/zone faces take both
//        frame fluxes -- + an inactive one; nonRotatingPatches literal and
//        regex; a rotating wall; MRF-included open, slip and symmetryPlane
//        faces; omega `constant`, a non-normalised axis)
// Gates: U, p, phi at iterations 100 and 400 within 1e-10 of the field scale
// (measured 3.6e-14 .. 7.4e-13); initial residuals of the first 50 iterations
// within 1e-8 relative where above 1e-9 (measured 2e-12 .. 4.1e-10) (the history is the algorithm's);
// 2D: Uz exactly 0; two runs byte-identical (determinism).
// FP32 (--fv-precision fp32, docs/fv.md "Precision", default floors): U, p,
// phi (RAS: nut) vs the FP64 run and vs simpleFoam within 1e-4 of scale
// (measured 4.6e-7 .. 3.2e-5; the 400-iteration sst_plate nut 1.03e-2 with
// and without floors, gate 3e-2: the unconverged plate transient amplifies
// rounding -- cause not isolated; the converged rotor2 SST without floors is
// 4.8e-5), no solve near maxIter, every
// floor-limited solve at or below its
// floor; floors: FP64 and floor-free FP32 never floor-limited, the floors stop
// the converged cavity tail (p iterations 69 vs 824 in the last 50); FP32
// runs byte-identical (channel, sst_plate, mrf2).
#include "dns/foamio.hpp"
#include "dns/fvsimple.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/fv_simple.hpp"
#include "dns/la/graph.hpp"

#include "check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
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

struct Run
{
    FvLoaded L;
    FvSimpleSetup s;
    std::vector<gpu::FvIterLog> logs;
    std::vector<gpu::FvState> at; // states at the requested iterations
};

std::expected<Run, std::string> run(const char* name, std::vector<int> stops, const FvNumerics& num = {})
{
    Run r;
    auto L = loadFvCase(fs::path(DNS_FV_DATA) / name);
    if (!L) return std::unexpected(L.error());
    r.L = std::move(*L);
    const auto g = la::graphOf(r.L.fm);
    const auto pat = la::ellPattern(g);
    auto s = buildSimpleSetup(r.L.fm, r.L.c, g, pat);
    if (!s) return std::unexpected(s.error());
    r.s = std::move(*s);
    if (r.L.c.ras.model != FvTurbModel::Laminar)
        if (auto x = buildRasSetup(r.L.pm, r.L.fm, r.L.c, r.s); !x) return std::unexpected(x.error());
    auto ds = gpu::DeviceSimple::make(r.s, r.L.c, g, num);
    if (!ds) return std::unexpected(ds.error());
    const int N = stops.back();
    for (int it = 1; it <= N; ++it) {
        auto lg = ds->iterate();
        if (!lg) return std::unexpected(std::format("iteration {}: {}", it, lg.error()));
        r.logs.push_back(*lg);
        if (std::ranges::find(stops, it) != stops.end()) {
            auto st = ds->state();
            if (!st) return std::unexpected(st.error());
            r.at.push_back(std::move(*st));
        }
    }
    return r;
}

// max |a - b| / scale over the internal field (OpenFOAM order on both sides).
double fieldDiff(const Run& r, const gpu::FvState& st, const char* name, const char* field, int t)
{
    const std::size_t nC = std::size_t(r.s.nC);
    const int nc = std::string(field) == "U" ? 3 : 1;
    auto ref = readFoamInternalField(fs::path(DNS_FV_DATA) / name / "ref" / std::to_string(t) / field, nc,
                                     std::int64_t(nC));
    if (!ref) {
        std::printf("    %s\n", ref.error().c_str());
        return 1e300;
    }
    double d = 0, scale = 0, lo = 1e300, hi = -1e300;
    for (std::size_t c = 0; c < nC; ++c) {
        const std::size_t i = std::size_t(r.s.srcCell[c]);
        for (int q = 0; q < nc; ++q) {
            const std::string fs(field);
            const double mine = nc == 3            ? st.U[std::size_t(q) * nC + c]
                                : fs == "p"        ? st.p[c]
                                : fs == "nut"      ? st.nut[c]
                                : fs == "omega"    ? st.s[1][c]
                                                   : st.s[0][c]; // k, nuTilda
            const double theirs = (*ref)[i * std::size_t(nc) + std::size_t(q)];
            d = std::max(d, std::abs(mine - theirs));
            scale = std::max(scale, std::abs(theirs));
            lo = std::min(lo, theirs);
            hi = std::max(hi, theirs);
        }
    }
    return std::string(field) == "p" ? d / (hi - lo) : d / scale;
}

double phiDiff(const Run& r, const gpu::FvState& st, const char* name, int t)
{
    const std::size_t nI = std::size_t(r.s.nI);
    auto ref = readFoamInternalField(fs::path(DNS_FV_DATA) / name / "ref" / std::to_string(t) / "phi", 1,
                                     r.L.pm.man.nInternalFaces);
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

// ref/residuals.csv: iter, U components (solved), first p.
std::vector<std::vector<double>> residuals(const char* name)
{
    std::ifstream f(fs::path(DNS_FV_DATA) / name / "ref" / "residuals.csv");
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) row.push_back(std::stod(tok));
        rows.push_back(row);
    }
    return rows;
}

void gate(const char* name, int last = 400)
{
    std::printf("IF1 %s\n", name);
    auto r = run(name, {100, last});
    if (!r) {
        check(false, std::string(name) + ": " + r.error());
        return;
    }
    const int stops[2] = {100, last};
    if (std::string(name) == "mrf2") {
        const auto& s = r->s;
        const auto n1 = std::ranges::count(s.bMrf, std::uint8_t(1)), n2 = std::ranges::count(s.bMrf, std::uint8_t(2));
        const auto two = std::ranges::count_if(s.frame2, [](double v) { return v != 0.0; });
        check(s.mrf && s.omega.size() == 2 && s.mrfTwo && two > 0 && n1 > 0 && n2 > 0,
              std::format("MRF setup: {} active zones, {} zone/zone faces, {} included + {} excluded boundary faces",
                          s.omega.size(), two, n1, n2));
    }
    // phi is compared in OpenFOAM face order, which a merged cyclic pair
    // (FaceMesh: internal faces) does not have.
    const bool cyclic = std::ranges::any_of(r->L.c.patches, [](const FvPatchBc& b) {
        return b.meshType == FoamPatchType::Cyclic;
    });
    for (int k = 0; k < 2; ++k) {
        const double dU = fieldDiff(*r, r->at[std::size_t(k)], name, "U", stops[k]);
        const double dp = fieldDiff(*r, r->at[std::size_t(k)], name, "p", stops[k]);
        const double df = cyclic ? 0.0 : phiDiff(*r, r->at[std::size_t(k)], name, stops[k]);
        check(dU < 1e-10 && dp < 1e-10 && df < 1e-10,
              std::format("iteration {}: U {:.2e}, p {:.2e}, phi {:.2e} of scale (gate 1e-10)", stops[k], dU, dp, df));
    }
    const auto ref = residuals(name);
    const int ea = r->s.emptyAxis;
    // A component whose solution is identically zero (the duct's Uz: a
    // z-invariant flow) has a residual of pure rounding noise on both sides,
    // normalised to O(1): not part of the algorithm's history.
    const std::size_t nC = std::size_t(r->s.nC);
    std::array<bool, 3> noise{};
    {
        double mx = 0;
        std::array<double, 3> mj{};
        for (int j = 0; j < 3; ++j)
            for (std::size_t c = 0; c < nC; ++c) {
                mj[std::size_t(j)] = std::max(mj[std::size_t(j)], std::abs(r->at[0].U[std::size_t(j) * nC + c]));
                mx = std::max(mx, mj[std::size_t(j)]);
            }
        for (int j = 0; j < 3; ++j) noise[std::size_t(j)] = mj[std::size_t(j)] < 1e-10 * mx;
    }
    double worst = 0;
    int n = 0;
    for (std::size_t it = 0; it < 50 && it < ref.size(); ++it) {
        std::vector<double> mine;
        std::vector<bool> skip;
        for (int j = 0; j < 3; ++j)
            if (j != ea) {
                mine.push_back(r->logs[it].U[std::size_t(j)].initial);
                skip.push_back(noise[std::size_t(j)]);
            }
        mine.push_back(r->logs[it].p.front().initial);
        skip.push_back(false);
        for (std::size_t q = 0; q < mine.size() && q + 1 < ref[it].size(); ++q) {
            const double theirs = ref[it][q + 1];
            if (theirs < 1e-9 || skip[q]) continue;
            worst = std::max(worst, std::abs(mine[q] - theirs) / theirs);
            ++n;
        }
    }
    check(n > 0 && worst < 1e-8,
          std::format("initial residuals, first 50 iterations: worst {:.2e} relative over {} values (gate 1e-8)", worst, n));
    if (ea >= 0) {
        bool zero = true;
        const std::size_t nC = std::size_t(r->s.nC);
        for (std::size_t c = 0; c < nC; ++c) zero = zero && r->at[1].U[std::size_t(ea) * nC + c] == 0.0;
        check(zero, "2D: the empty-axis velocity is exactly 0 after 400 iterations");
    }
}

// IF-T: the RAS cases (tools/fv/make_ras_cases.py, phase BL). U, p, phi
// within 1e-10 of scale and the model's fields (k, omega, nut | nuTilda, nut)
// within 1e-9 at iterations 100 and 400; initial residuals of every solve
// (U, first p, then omega, k | nuTilda) of the first 50 iterations within
// 1e-8 relative where above 1e-9.
void gateRas(const char* name, int last = 400)
{
    std::printf("IF-T %s\n", name);
    auto r = run(name, {100, last});
    if (!r) {
        check(false, std::string(name) + ": " + r.error());
        return;
    }
    const bool sst = r->L.c.ras.model == FvTurbModel::KOmegaSST;
    const int stops[2] = {100, last};
    for (int k = 0; k < 2; ++k) {
        const auto& st = r->at[std::size_t(k)];
        const double dU = fieldDiff(*r, st, name, "U", stops[k]);
        const double dp = fieldDiff(*r, st, name, "p", stops[k]);
        const double df = phiDiff(*r, st, name, stops[k]);
        const double d0 = fieldDiff(*r, st, name, sst ? "k" : "nuTilda", stops[k]);
        const double d1 = sst ? fieldDiff(*r, st, name, "omega", stops[k]) : 0.0;
        const double dn = fieldDiff(*r, st, name, "nut", stops[k]);
        check(dU < 1e-10 && dp < 1e-10 && df < 1e-10 && d0 < 1e-9 && d1 < 1e-9 && dn < 1e-9,
              std::format("iteration {}: U {:.2e}, p {:.2e}, phi {:.2e}, {} {:.2e}{}, nut {:.2e} of scale (gates "
                          "1e-10 / 1e-9)",
                          stops[k], dU, dp, df, sst ? "k" : "nuTilda", d0,
                          sst ? std::format(", omega {:.2e}", d1) : std::string(), dn));
    }
    const auto ref = residuals(name);
    const int ea = r->s.emptyAxis;
    double worst = 0;
    int n = 0;
    for (std::size_t it = 0; it < 50 && it < ref.size(); ++it) {
        const auto& lg = r->logs[it];
        std::vector<double> mine;
        for (int j = 0; j < 3; ++j)
            if (j != ea) mine.push_back(lg.U[std::size_t(j)].initial);
        mine.push_back(lg.p.front().initial);
        for (int q = 0; q < lg.nT; ++q) mine.push_back(lg.T[std::size_t(q)].initial);
        for (std::size_t q = 0; q < mine.size() && q + 1 < ref[it].size(); ++q) {
            const double theirs = ref[it][q + 1];
            if (theirs < 1e-9) continue;
            worst = std::max(worst, std::abs(mine[q] - theirs) / theirs);
            ++n;
        }
    }
    check(n > 0 && worst < 1e-8,
          std::format("initial residuals (U, p, turbulence), first 50 iterations: worst {:.2e} relative over {} values "
                      "(gate 1e-8)",
                      worst, n));
    if (ea >= 0) {
        bool zero = true;
        const std::size_t nC = std::size_t(r->s.nC);
        for (std::size_t c = 0; c < nC; ++c) zero = zero && r->at[1].U[std::size_t(ea) * nC + c] == 0.0;
        check(zero, "2D: the empty-axis velocity is exactly 0");
    }
}

void determinismRas()
{
    std::printf("determinism (RAS)\n");
    auto a = run("sst_plate", {30});
    auto b = run("sst_plate", {30});
    if (!a || !b) {
        check(false, "runs");
        return;
    }
    const auto& x = a->at[0];
    const auto& y = b->at[0];
    check(x.U == y.U && x.p == y.p && x.phi == y.phi && x.s == y.s && x.sb == y.sb && x.nut == y.nut &&
              x.nutB == y.nutB,
          "two 30-iteration sst_plate runs are byte-identical (U, p, phi, k, omega, nut)");
}

void determinism()
{
    std::printf("determinism\n");
    auto a = run("channel", {30});
    auto b = run("channel", {30});
    if (!a || !b) {
        check(false, "runs");
        return;
    }
    const auto& x = a->at[0];
    const auto& y = b->at[0];
    check(x.U == y.U && x.p == y.p && x.phi == y.phi && x.Ub == y.Ub && x.pb == y.pb,
          "two 30-iteration channel runs are byte-identical");
}

// --- FP32 (docs/fv.md "Precision") --------------------------------------------

const FvNumerics kFp32 = fvNumerics(FvPrecision::Fp32);

// max |a - b| / max |b| (p: / (max b - min b)), host vectors.
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

// The worst solve of a run: max iterations of any U / p / turbulence solve.
int maxIters(const Run& r)
{
    int m = 0;
    for (const auto& lg : r.logs) {
        for (const auto& u : lg.U) m = std::max(m, u.iterations);
        for (const auto& q : lg.p) m = std::max(m, q.iterations);
        for (int q = 0; q < lg.nT; ++q) m = std::max(m, lg.T[std::size_t(q)].iterations);
    }
    return m;
}

// FP32 (default floors) vs FP64 and vs simpleFoam at iteration `last`: U, p,
// phi (and the RAS fields) within `band` of scale; no solve near maxIter;
// every floor-limited solve ended at or below its floor; 2D: the empty-axis U
// exactly 0.
void gateFp32(const char* name, int last, double band, double nutBand = 1e-4)
{
    std::printf("FP32 %s\n", name);
    auto f = run(name, {last}, kFp32);
    auto d = run(name, {last});
    if (!f || !d) {
        check(false, std::string(name) + ": " + (f ? d.error() : f.error()));
        return;
    }
    const auto& sf = f->at[0];
    const auto& sd = d->at[0];
    const bool ras = f->L.c.ras.model != FvTurbModel::Laminar;
    const double fU = fieldDiff(*f, sf, name, "U", last), fp = fieldDiff(*f, sf, name, "p", last);
    const double dU = vecDiff(sf.U, sd.U), dp = vecDiff(sf.p, sd.p, true), dphi = vecDiff(sf.phi, sd.phi);
    const double dnut = ras ? vecDiff(sf.nut, sd.nut) : 0.0;
    check(dU < band && dp < band && dphi < band && dnut < nutBand,
          std::format("iteration {}: FP32 vs FP64 U {:.2e}, p {:.2e}, phi {:.2e}{} of scale (gate {:.0e}{})", last, dU,
                      dp, dphi, ras ? std::format(", nut {:.2e}", dnut) : std::string(), band,
                      ras ? std::format(", nut {:.0e}", nutBand) : std::string()));
    if (ras) { // report: the same FP32 run without floors (what the floors add to the FP32 offset)
        auto z = run(name, {last}, FvNumerics{FvPrecision::Fp32, FvFloors{}});
        if (z)
            std::printf("    FP32 without floors vs FP64: U %.2e, p %.2e, nut %.2e\n", vecDiff(z->at[0].U, sd.U),
                        vecDiff(z->at[0].p, sd.p, true), vecDiff(z->at[0].nut, sd.nut));
    }
    check(fU < band && fp < band,
          std::format("iteration {}: FP32 vs simpleFoam U {:.2e}, p {:.2e} of scale (gate {:.0e})", last, fU, fp, band));
    const int mi = maxIters(*f);
    check(mi < 100, std::format("FP32: no solve near maxIter (max {} iterations)", mi));
    int nfl = 0;
    bool below = true;
    for (const auto& lg : f->logs) {
        auto see = [&](const gpu::FvSolveLog& l, double fl) {
            if (!l.floorLimited) return;
            ++nfl;
            below = below && l.final <= fl;
        };
        for (const auto& u : lg.U) see(u, kFp32.floors.U);
        for (const auto& q : lg.p) see(q, kFp32.floors.p);
        for (int q = 0; q < lg.nT; ++q) see(lg.T[std::size_t(q)], kFp32.floors.s);
    }
    check(below, std::format("FP32: every floor-limited solve ({}) ends at or below its floor", nfl));
    const int ea = f->s.emptyAxis;
    if (ea >= 0) {
        const std::size_t nC = std::size_t(f->s.nC);
        bool zero = true;
        for (std::size_t c = 0; c < nC; ++c) zero = zero && sf.U[std::size_t(ea) * nC + c] == 0.0;
        check(zero, "FP32 2D: the empty-axis velocity is exactly 0");
    }
}

// Floors: FP64 never floor-limited; FP32 with no floors never floor-limited
// and its converged tail keeps iterating on noise; FP32 with the floors
// stops the converged tail at r0 (0 iterations, floor-limited).
void floorsFp32()
{
    std::printf("FP32 floors (cavity, 400 iterations)\n");
    const FvNumerics none{FvPrecision::Fp32, FvFloors{}};
    auto a = run("cavity", {400});
    auto b = run("cavity", {400}, none);
    auto c = run("cavity", {400}, kFp32);
    if (!a || !b || !c) {
        check(false, "runs");
        return;
    }
    auto count = [](const Run& r, int from) {
        std::array<int, 3> n{}; // floor-limited, p solves in [from, end), their iterations
        for (std::size_t it = std::size_t(from); it < r.logs.size(); ++it) {
            for (const auto& u : r.logs[it].U) n[0] += u.floorLimited;
            for (const auto& q : r.logs[it].p) {
                n[0] += q.floorLimited;
                ++n[1];
                n[2] += q.iterations;
            }
        }
        return n;
    };
    const auto na = count(*a, 0), nb = count(*b, 0), nbT = count(*b, 350), ncT = count(*c, 350);
    std::printf("  last 50 it: p iterations FP32 no floor %d / %d solves, FP32 floors %d / %d (floor-limited %d)\n",
                nbT[2], nbT[1], ncT[2], ncT[1], ncT[0]);
    check(na[0] == 0, "FP64: no floor-limited solve");
    check(nb[0] == 0, "FP32 without floors: no floor-limited solve");
    check(nbT[2] > 0 && ncT[2] < nbT[2] && ncT[0] > 0,
          std::format("FP32 floors: the converged tail stops on the floor (p iterations {} vs {} without)", ncT[2],
                      nbT[2]));
}

void determinismFp32()
{
    std::printf("determinism (FP32)\n");
    for (const char* name : {"channel", "sst_plate", "mrf2"}) {
        auto a = run(name, {30}, kFp32);
        auto b = run(name, {30}, kFp32);
        if (!a || !b) {
            check(false, "runs");
            return;
        }
        const auto& x = a->at[0];
        const auto& y = b->at[0];
        check(x.U == y.U && x.p == y.p && x.phi == y.phi && x.Ub == y.Ub && x.pb == y.pb && x.s == y.s &&
                  x.nut == y.nut,
              std::string("two 30-iteration FP32 ") + name + " runs are byte-identical");
    }
}

} // namespace

int main()
{
    if (!gpu::deviceAvailable()) dnstest::skip("no GPU device");
    gate("cavity");
    gate("cavity_skew");
    gate("channel");
    gate("duct", 300);
    gate("cavity_lim");
    gate("mrf2", 300);
    determinism();
    gateRas("sst_plate");
    gateRas("sst_skew");
    gateRas("sa_skew");
    determinismRas();
    gateFp32("cavity", 400, 1e-4);
    gateFp32("channel", 400, 1e-4);
    gateFp32("duct", 300, 1e-4);
    gateFp32("mrf2", 300, 1e-4);
    gateFp32("sst_plate", 400, 1e-4, 3e-2);
    gateFp32("sa_skew", 400, 1e-4);
    floorsFp32();
    determinismFp32();
    return dnstest::finish();
}
