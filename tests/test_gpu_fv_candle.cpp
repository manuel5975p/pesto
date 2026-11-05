// The candle on the compressible FV path (docs/fvcomp.md "Hydrostatic ambient", masterplan
// M5): gravity with open patches through the hydrostatic ambient's per-face rows.
//
//   rest    (a) the closed + open box (Wall floor, Inflow at rest, Outflow lid, Far side;
//           fvAmbientRows) at rest under gravity on the uniform, oblique-g (the rotated
//           mesh's frame) and random meshes: the double instantiation vs FvRef <= 1e-13 (rhs,
//           3 steps), FP32 20000 steps max |u| flat in time (second half <= 3 x first half) at
//           the FP32 floor. (b) the binary: dns flame --mode candle --quiescent --solver fv
//           --tt 10 (the structured gpu_candle_rest's case and gate: max |u| < 5e-3 U,
//           box-mean p/p_a within 1e-4, no out-of-bounds cell; structured measured 1.21e-3),
//           and FP64 tt 2: max |u| < 1e-9 U (the scheme's rest is exact: the FP32 number is
//           p's rounding, p = T0 = 1/(gamma M^2)).
//   g0      --mode candle --gravity 0 --solver fv against --mode diffusion --solver fv with
//           the same flags (tt 3): the jet's 21 stats columns byte-identical (the label differs).
//   height  the candle on the same mesh, FV vs structured --nscbc-etain 1000 (the inflow
//           model nearest the FV fixed face; defaults W 1, g 14, M 0.01, 6 cells/l, tt 30,
//           window t in [20, 30]): FV steady (halves within 2%), tip within 15% (measured
//           -9.5%: the inflow carries 5.9% less fuel, Roper's Q^(4/3) explains -7.4%), TmaxIn
//           within 1% (measured +0.2%). Ladder 4, 6, 8 cells/l: docs/fvcomp.md.
//
// Usage: test_gpu_fv_candle rest|g0|height. No HIP device -> skip (77).
#include "fv_comp_test_util.hpp"

#include "dns/fvconv.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <fstream>
#include <sstream>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

using namespace dnstest::fvc;
using dnstest::check;

namespace {

const std::string kBin = "\"" DNS_BIN "\" flame ";

// The first n comma-separated fields of every line of a CSV, as TEXT.
std::vector<std::string> csvFields(const std::string& path, int n)
{
    std::ifstream f(path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.starts_with('#')) continue; // the run label names the mode
        std::stringstream ss(line);
        std::string cell, row;
        for (int k = 0; k < n && std::getline(ss, cell, ','); ++k) row += cell + ",";
        out.push_back(row);
    }
    return out;
}

double maxU(const FvState& s)
{
    double m = 0.0;
    for (std::size_t c = 0; c < std::size_t(s.nC); ++c)
        m = std::max(m, std::hypot(s.var(1)[c], s.var(2)[c], s.var(3)[c]) / s.var(0)[c]);
    return m;
}

void restBox()
{
    std::printf("== rest (a): closed + open box at rest under gravity, 48 x 28\n");
    const double gam = kGamma, gs = 0.3, th = 30.0 * std::numbers::pi / 180.0;
    const Grid g = Grid::make(48, 28, 1, 2.0, 1.0 + 2.0 / 24.0, 1.0);
    const Transport tr = Transport::constant(gam, 0.02);
    const double T0 = 1.0 / tr.Rgas, c0 = std::sqrt(gam * tr.Rgas * T0);
    const FvAmbient amb{.p0 = 1.0, .T0 = T0};
    struct Case
    {
        const char* name;
        FvPerturb kind;
        std::array<double, 3> gv;
    };
    const Case cases[] = {{"uniform", FvPerturb::None, {0.0, -gs, 0.0}},
                          {"oblique g 30 deg", FvPerturb::None, {gs * std::sin(th), -gs * std::cos(th), 0.0}},
                          {"random 0.3h", FvPerturb::Random, {0.0, -gs, 0.0}}};
    for (const Case& k : cases) {
        const FaceMesh fm = orDie(fvConvBox(g, true, k.kind, k.kind == FvPerturb::None ? 0.0 : 0.3), "box");
        const std::vector<FvBc> bc = bcFor(fm, [&](const MeshPatch& p) {
            FvBc b;
            b.kind = p.type == FoamPatchType::Empty ? FvBcKind::Empty : FvBcKind::Wall;
            if (p.name == "xlo") {
                b.kind = FvBcKind::Inflow;
                b.tIn = T0;
            } else if (p.name == "lid") {
                b.kind = FvBcKind::Outflow;
                b.characteristic = true;
                b.pOut = 1.0;
                b.tBack = T0;
            } else if (p.name == "xhi") {
                b.kind = FvBcKind::Far;
                b.qFar[4] = 1.0 / (gam - 1.0);
            }
            return b;
        });
        const FvBcTable t = fvAmbientRows(fm, {.bc = bc, .faceRow = {}}, amb, k.gv, gam, tr.Rgas);
        const FvConfig rc{.tr = tr, .bc = t.bc, .bcRow = t.faceRow, .gravity = k.gv, .ambient = amb};
        const FvRef ref = orDie(FvRef::make(fm, rc), "FvRef");
        const FvState s0 = fvAmbientState(fm, amb, k.gv, gam, tr.Rgas, kFvMeanVars);
        auto dev = [&](FvPrecision pr) {
            return orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = t.bc, .bcRow = t.faceRow, .precision = pr, .gravity = k.gv,
                                                 .ambient = amb}),
                         "device");
        };
        // The oracle on a moving state (the rest is trivially equal): + a smooth velocity.
        FvState sm = s0;
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            const double rho = sm.var(0)[c], u = 0.05 * std::sin(3 * fm.C[0][c]) * std::cos(2 * fm.C[1][c]);
            const double v = 0.04 * std::cos(2 * fm.C[0][c] + fm.C[1][c]);
            sm.var(1)[c] = rho * u;
            sm.var(2)[c] = rho * v;
            sm.var(4)[c] += 0.5 * rho * (u * u + v * v);
        }
        auto d64 = dev(FvPrecision::Fp64);
        orDie(d64.upload(sm), "upload");
        FvState rR, rD;
        ref.rhs(sm, rR);
        orDie(d64.rhs(rD), "rhs");
        double e = relDiff(rD.q, rR.q, kFvMeanVars, fm.nC);
        const double dt = 0.4 * orDie(ref.stableDt(sm), "dt");
        FvState q = sm, qD;
        ref.march(q, dt, 3);
        for (int n = 0; n < 3; ++n) d64.step(dt);
        orDie(d64.download(qD), "download");
        e = std::max(e, relDiff(qD.q, q.q, kFvMeanVars, fm.nC));
        // FP32 at rest: max |u| sampled every 400 steps.
        auto d32 = dev(FvPrecision::Fp32);
        orDie(d32.upload(s0), "upload");
        const double dr = 0.4 * orDie(ref.stableDt(s0), "dt");
        double first = 0.0, second = 0.0;
        FvState s;
        for (int n = 1; n <= 20000; ++n) {
            d32.step(dr);
            if (n % 1000) continue;
            orDie(d32.download(s), "download");
            (n <= 10000 ? first : second) = std::max(n <= 10000 ? first : second, maxU(s));
        }
        std::printf("  %s: FP64 device vs FvRef %.2e; FP32 rest max|u|/c first half %.3e, second half %.3e (%.0f "
                    "acoustic crossings)\n",
                    k.name, e, first / c0, second / c0, 20000 * dr * c0 / 2.0);
        check(e <= 1e-13, std::format("{}: the double instantiation == FvRef to 1e-13 with the ambient rows", k.name));
        check(second <= 3.0 * first && second / c0 < 1e-5, std::format("{}: FP32 rest flat in time at the floor", k.name));
    }
}

void restBinary()
{
    std::printf("== rest (b): dns flame --mode candle --quiescent --solver fv\n");
    const std::string prefix = "/tmp/dns_test_fv_candle_rest";
    struct Run
    {
        const char* extra;
        const char* tt;
        double bound;
    };
    for (const Run r : {Run{"", "10", 5e-3}, Run{" --fv-precision fp64", "2", 1e-9}}) {
        const int rc = valtest::runBinary(kBin + "--mode candle --quiescent --solver fv --tt " + r.tt + r.extra +
                                          " --emit stats --quiet --out " + prefix);
        check(rc == 0, std::format("quiescent FV candle{} runs (exit 0)", r.extra));
        const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
        check(tbl.has_value(), "stats readable");
        if (rc != 0 || !tbl) return;
        const auto& t = tbl->col("t");
        double first = 0, second = 0, pdev = 0, hits = 0;
        for (std::size_t i = 0; i < t.size(); ++i) {
            (2 * i < t.size() ? first : second) = std::max(2 * i < t.size() ? first : second, tbl->col("uMax")[i]);
            pdev = std::max(pdev, std::abs(tbl->col("pMean")[i] - 1));
            hits = std::max(hits, tbl->col("hits")[i]);
        }
        std::printf("  %s tt %s: max |u| %.4g U (halves %.4g / %.4g) over %zu rows, max |p/p_a - 1| %.3g, hits %g "
                    "(structured gpu_candle_rest 1.21e-3)\n",
                    *r.extra ? "FP64" : "FP32", r.tt, std::max(first, second), first, second, t.size(), pdev, hits);
        check(std::max(first, second) < r.bound, std::format("hydrostatic box at rest: max |u| < {:g} U", r.bound));
        check(second <= 3.0 * first, "max |u| flat in time (second half <= 3 x first)");
        check(pdev < 1e-4, "box-mean p on the ambient column within 1e-4");
        check(hits == 0, "no out-of-bounds cells");
    }
}

void g0()
{
    std::printf("== g0: --mode candle --gravity 0 vs --mode diffusion, --solver fv\n");
    const std::string prefix = "/tmp/dns_test_fv_candle_g0";
    const std::string shared = " --solver fv --slot-width 4 --cells-per-unit 4 --slot-edge 1 --length 16 --coflow-width 4 "
                               "--ignition 4 --stoich 3.4 --ox-y 0.233 --damkohler 5000 --mach 0.05 --tt 3 --emit stats "
                               "--quiet --out ";
    const int ra = valtest::runBinary(kBin + "--mode candle --gravity 0" + shared + prefix + "_c");
    const int rb = valtest::runBinary(kBin + "--mode diffusion" + shared + prefix + "_d");
    check(ra == 0 && rb == 0, "both runs exit 0");
    const auto a = csvFields(prefix + "_c_stats.csv", 21);
    const auto b = csvFields(prefix + "_d_stats.csv", 21);
    std::size_t diff = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) diff += a[i] != b[i];
    std::printf("  g0: %zu / %zu rows, %zu differ\n", a.size(), b.size(), diff);
    check(a.size() == b.size() && a.size() > 10 && diff == 0, "g = 0: the jet's 21 stats columns byte-identical");
}

// Window means over t in [20, 30]: tip, TmaxIn / T_ad.
struct Window
{
    double tip = 0, tmax = 0, a = 0, b = 0;
    bool closed = true;
};

constexpr double kTipBand = 0.15, kTmaxBand = 0.01;

Window window(const std::string& path)
{
    Window w;
    const auto tbl = valtest::loadCsv(path);
    if (!tbl) {
        w.closed = false;
        return w;
    }
    const auto& t = tbl->col("t");
    double s[2] = {0, 0}, n[2] = {0, 0}, tm = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i] < 20) continue;
        const double x = tbl->col("xTip")[i];
        w.closed = w.closed && std::isfinite(x);
        const int k = t[i] < 25 ? 0 : 1;
        s[k] += x;
        n[k] += 1;
        tm += tbl->col("TmaxIn")[i];
    }
    w.a = s[0] / n[0];
    w.b = s[1] / n[1];
    w.tip = (s[0] + s[1]) / (n[0] + n[1]);
    w.tmax = tm / (n[0] + n[1]);
    return w;
}

void height()
{
    std::printf("== height: the candle, FV vs structured etaIn 1000, same mesh (6 cells/l)\n");
    const std::string prefix = "/tmp/dns_test_fv_candle_h";
    const std::string flags = "--mode candle --mach 0.01 --tt 30 --cells-per-unit 6 --emit stats --quiet --out ";
    const int rs = valtest::runBinary(kBin + "--nscbc-etain 1000 " + flags + prefix + "_s");
    const int rf = valtest::runBinary(kBin + "--solver fv " + flags + prefix + "_f");
    check(rs == 0 && rf == 0, "both candles run to tt 30 (exit 0)");
    if (rs != 0 || rf != 0) return;
    const Window s = window(prefix + "_s_stats.csv"), f = window(prefix + "_f_stats.csv");
    const double T0 = 1.0 / (kGamma * 0.01 * 0.01), Tad = 5.0 * T0;
    std::printf("  structured: tip %.4f l (halves %.4f / %.4f), TmaxIn/T_ad %.4f\n", s.tip, s.a, s.b, s.tmax / Tad);
    std::printf("  FV:         tip %.4f l (halves %.4f / %.4f), TmaxIn/T_ad %.4f\n", f.tip, f.a, f.b, f.tmax / Tad);
    std::printf("  FV / structured: tip %+.4f, TmaxIn %+.4f\n", f.tip / s.tip - 1, f.tmax / s.tmax - 1);
    check(s.closed && f.closed, "both tips closed in the window");
    check(std::abs(f.a / f.b - 1) < 0.02, "FV candle steady (window halves within 2%)");
    check(std::abs(f.tip / s.tip - 1) < kTipBand, std::format("FV tip within {:g} of the structured candle's", kTipBand));
    check(std::abs(f.tmax / s.tmax - 1) < kTmaxBand,
          std::format("FV TmaxIn within {:g} of the structured candle's", kTmaxBand));
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "rest";
    if (mode == "rest") {
        restBox();
        restBinary();
    } else if (mode == "g0") {
        g0();
    } else if (mode == "height") {
        height();
    } else {
        std::printf("usage: test_gpu_fv_candle rest|g0|height\n");
        return 1;
    }
    return dnstest::finish();
}
