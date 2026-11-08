// Cross-verification of the compressible FV family against the structured
// solver (docs/fvcomp.md "Cross-verification"): OpenFOAM is absent here, so
// the structured solver is the oracle.
//   X1 uniform periodic 3D TGV-like state, same dt: FV(FP64) vs dns::Comp and
//      FV(FP32) vs DeviceComp over >= 100 steps. The scheme is pinned equal:
//      expected relation = rounding.
//   X2 same with Smagorinsky and sigma LES on (FP32 both; no FP64 structured
//      LES exists): mu_t field and state.
//   X3 smooth curvilinear periodic mesh, grid ladder, FP64: the schemes differ
//      legitimately (LSQ gradients vs metric gradients; pyramids vs diagonal
//      volumes), so the FV - structured difference must SHRINK at order ~ 2.
//   X5 (`test_gpu_fv_cross far`) settled cd, FV - structured: shrinks with h.
//   X4 (`test_gpu_fv_cross ladder`) O-grid cylinder end to end through the dns
//      binary, compensated state, 3 levels, cd at a fixed time: walls and far
//      boundaries differ between the schemes, the difference must shrink. Plus a
//      smoke run of every --solver fv route (exit 0, finite, stats rows).
#include "fv_comp_test_util.hpp"

#include "dns/gpu/comp.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/reductions.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <cstring>
#include <numbers>

using namespace dnstest::fvc;
using dnstest::check;
using std::numbers::pi;

namespace {

CompState tgvState(const Grid& g, double M)
{
    auto s = CompState::zeros(g);
    const double p0 = 1.0 / (kGamma * M * M);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy, z = (k + 0.5) * g.hz;
                const double u = std::sin(x) * std::cos(y) * std::cos(z), v = -std::cos(x) * std::sin(y) * std::cos(z);
                const double p = p0 + (std::cos(2 * x) + std::cos(2 * y)) * (std::cos(2 * z) + 2.0) / 16.0;
                const std::size_t c = g.idx(i, j, k);
                s.rho[c] = 1.0;
                s.mx[c] = u;
                s.my[c] = v;
                s.E[c] = p / (kGamma - 1.0) + 0.5 * (u * u + v * v);
            }
    return s;
}

FvState toFv(const FaceMesh& fm, const CompState& q)
{
    FvState s = FvState::zeros(fm.nC);
    const std::vector<double>* v[5] = {&q.rho, &q.mx, &q.my, &q.mz, &q.E};
    for (int r = 0; r < 5; ++r)
        for (std::size_t n = 0; n < std::size_t(fm.nC); ++n) s.var(r)[n] = (*v[r])[std::size_t(fm.srcCell[n])];
    return s;
}

gpu::DeviceComp orDieH(std::expected<gpu::DeviceComp, hipError_t> d)
{
    if (!d) {
        std::printf("FATAL: DeviceComp\n");
        std::exit(1);
    }
    return std::move(*d);
}

void x1()
{
    std::printf("X1 TGV 32^3 M 0.3 Re 200, FV vs structured, same dt\n");
    const double L = 2 * pi;
    const Grid g = Grid::make(32, 32, 32, L, L, L);
    const double mu = 1.0 / 200.0, M = 0.3;
    const Transport tr = Transport::constant(kGamma, mu);
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "x1");
    const CompState ic = tgvState(g, M);
    const FvState s0 = toFv(fm, ic);

    Comp host = Comp::make(g, kGamma, mu);
    CompState h = ic;
    const double dt = 0.4 * orDie(host.stableDt(ic, 1.0), "dt");
    auto d64 = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = {}, .precision = FvPrecision::Fp64}), "x1");
    auto d32 = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = {}, .precision = FvPrecision::Fp32}), "x1");
    orDie(d64.upload(s0), "x1");
    orDie(d32.upload(s0), "x1");
    auto str = orDieH(gpu::DeviceComp::make(g, tr));
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!ds || ds->upload(ic) != hipSuccess) std::exit(1);

    constexpr int kN = 400;
    double fp64 = 0.0, fp32 = 0.0, own = 0.0;
    FvState a, b;
    CompState sh = CompState::zeros(g);
    std::printf("  step   FV64-Comp    FV32-DevComp  DevComp-Comp (structured FP32 own rounding)\n");
    for (int n = 1; n <= kN; ++n) {
        host.step(h, dt);
        d64.step(dt);
        d32.step(dt);
        str.step(*ds, gpu::Real(dt));
        if (n != 1 && n != 10 && n != 50 && n != 100 && n != 200 && n != kN) continue;
        if (ds->download(sh) != hipSuccess) std::exit(1);
        orDie(d64.download(a), "x1");
        orDie(d32.download(b), "x1");
        const FvState hf = toFv(fm, h), sf = toFv(fm, sh);
        fp64 = relDiff(a.q, hf.q, 5, fm.nC);
        fp32 = relDiff(b.q, sf.q, 5, fm.nC);
        own = relDiff(sf.q, hf.q, 5, fm.nC);
        std::printf("  %4d   %.3e    %.3e     %.3e\n", n, fp64, fp32, own);
    }
    check(fp64 <= 1e-12, std::format("FV(FP64) vs Comp after {} steps {:.2e} <= 1e-12 (rounding)", kN, fp64));
    check(fp32 <= 3.0 * own,
          std::format("FV(FP32) vs DeviceComp after {} steps {:.2e} <= 3 x structured FP32 vs FP64 {:.2e}", kN, fp32,
                      own));
}

// X6 (G12): the periodic double shock on a uniform box, eps 0. FP64 == FvRef; FP32 vs the
// structured --shock hll: inside 2 x the FV FP32 run's own distance from FP64; bytes within 1.
void x6()
{
    std::printf("X6 uniform periodic double shock, 400 x 4, FV vs FvRef vs structured --shock hll\n");
    const int n = 400, ny = 4;
    const Grid g = Grid::make(n, ny, 1, 1.0, double(ny) / n, 1.0 / n);
    const Transport tr = Transport::constant(kGamma, 1e-4);
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "x6");
    CompState ic = CompState::zeros(g);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < n; ++i) {
            const double x = (i + 0.5) / n, p = x > 0.25 && x < 0.75 ? 5.0 : 1.0, rho = x > 0.25 && x < 0.75 ? 2.0 : 1.0;
            const std::size_t c = g.idx(i, j, 0);
            ic.rho[c] = rho;
            ic.E[c] = p / (kGamma - 1.0);
        }
    const FvState s0 = toFv(fm, ic);
    const FvShock sh{.mode = ShockMode::Hll, .eps4 = 0.0};
    FvConfig rc{.tr = tr, .bc = {}};
    rc.shock = sh;
    for (const MeshPatch& p : fm.patches) {
        FvBc e;
        e.kind = FvBcKind::Empty;
        (void)p;
        rc.bc.push_back(e);
    }
    const FvRef ref = orDie(FvRef::make(fm, rc), "x6");
    const double dt = 0.5 * orDie(ref.stableDt(s0, {.cfl = 0.4}), "dt");
    FvState sR = s0;
    auto d64 = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = rc.bc, .precision = FvPrecision::Fp64, .shock = sh}), "x6");
    auto d32 = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = rc.bc, .precision = FvPrecision::Fp32, .shock = sh}), "x6");
    orDie(d64.upload(s0), "x6");
    orDie(d32.upload(s0), "x6");
    auto str = orDieH(gpu::DeviceComp::make(g, tr));
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!ds || ds->upload(ic) != hipSuccess || !str.enableShock({})) std::exit(1);
    constexpr int kN = 150;
    ref.march(sR, dt, kN);
    for (int i = 0; i < kN; ++i) {
        d64.step(dt);
        d32.step(dt);
        str.step(*ds, gpu::Real(dt));
    }
    FvState a, b;
    CompState sh2 = CompState::zeros(g);
    orDie(d64.download(a), "x6");
    orDie(d32.download(b), "x6");
    if (ds->download(sh2) != hipSuccess) std::exit(1);
    const FvState sf = toFv(fm, sh2);
    // The flow is 1D: rows my, mz are 0 to rounding, a relative row metric would read that noise.
    const auto rows = [&](const FvState& x, const FvState& y) {
        double w = 0.0;
        for (int v : {0, 1, 4}) {
            double e = 0.0, sc = 0.0;
            for (int c = 0; c < fm.nC; ++c) {
                e = std::max(e, std::abs(x.var(v)[c] - y.var(v)[c]));
                sc = std::max(sc, std::abs(y.var(v)[c]));
            }
            w = std::max(w, e / sc);
        }
        return w;
    };
    const double e64 = rows(a, sR), e32 = rows(b, sf), own = rows(b, a);
    std::vector<std::uint8_t> sb;
    if (str.shockBytes(*ds, sb) != hipSuccess) std::exit(1);
    FvEval ev;
    FvState o;
    orDie(d32.evaluate(ev, o), "x6");
    long worst = 0, sensed = 0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const std::size_t q = std::size_t(fm.srcCell[c]);
        worst = std::max(worst, std::labs(long(ev.sigma[c]) - long(sb[q])));
        sensed += ev.sigma[c] != 0;
    }
    std::printf("  %d steps, %ld sensed: FV64 - FvRef %.2e; FV32 - structured %.2e, own FP32 - FP64 %.2e; bytes off by %ld\n", kN, sensed, e64, e32, own, worst);
    check(sensed > 0, "sensed > 0");
    check(e64 <= 1e-13, "FP64 vs the oracle <= 1e-13");
    check(e32 <= 2.0 * own, "FP32 vs structured <= 2 x own FP32 vs FP64");
    check(worst <= 1, "bytes within 1");
}

void x2()
{
    std::printf("X2 LES, TGV 32^3 M 0.3 Re 2000, FV(FP32) vs DeviceComp, same dt\n");
    const double L = 2 * pi;
    const Grid g = Grid::make(32, 32, 32, L, L, L);
    const Transport tr = Transport::constant(kGamma, 1.0 / 2000.0);
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "x2");
    const CompState ic = tgvState(g, 0.3);
    const FvState s0 = toFv(fm, ic);
    Comp host = Comp::make(g, kGamma, tr.muRef);
    const double dt = 0.4 * orDie(host.stableDt(ic, 1.0), "dt");
    struct M
    {
        const char* name;
        SgsModel sgs;
        double C;
    };
    for (const M m : {M{"smag Cs 0.17", SgsModel::Smagorinsky, 0.17}, M{"sigma Csigma 1.35", SgsModel::Sigma, 1.35}}) {
        auto f32 = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = {}, .precision = FvPrecision::Fp32, .csLes = m.C,
                                                 .sgs = m.sgs, .prt = 0.9}), m.name);
        auto f64 = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = {}, .precision = FvPrecision::Fp64, .csLes = m.C,
                                                 .sgs = m.sgs, .prt = 0.9}), m.name);
        orDie(f32.upload(s0), m.name);
        orDie(f64.upload(s0), m.name);
        auto str = orDieH(gpu::DeviceComp::make(g, tr, m.C, m.sgs));
        auto ds = gpu::DeviceCompState::make(g.size());
        if (!ds || ds->upload(ic) != hipSuccess) std::exit(1);
        std::vector<double> nutS(g.size());
        double eS = 0.0, eO = 0.0, eN = 0.0, nutMax = 0.0;
        FvState a, b;
        CompState sh = CompState::zeros(g);
        constexpr int kN = 400;
        std::printf("  %s\n  step   FV32-DevComp  FV32-FV64 (FP32 own rounding)  mu_t FV32-DevComp / max   mu_t max\n",
                    m.name);
        for (int n = 1; n <= kN; ++n) {
            f32.step(dt);
            f64.step(dt);
            str.step(*ds, gpu::Real(dt));
            if (n != 1 && n != 10 && n != 100 && n != 200 && n != kN) continue;
            str.updateSgsNut(*ds);
            if (gpu::downloadWiden(str.mut(), nutS) != hipSuccess || ds->download(sh) != hipSuccess) std::exit(1);
            const auto nutF = orDie(f32.eddyViscosity(), m.name);
            orDie(f32.download(a), m.name);
            orDie(f64.download(b), m.name);
            const FvState sf = toFv(fm, sh);
            eS = relDiff(a.q, sf.q, 5, fm.nC);
            eO = relDiff(a.q, b.q, 5, fm.nC);
            double e = 0.0;
            nutMax = 0.0;
            for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
                const double ns = nutS[std::size_t(fm.srcCell[c])];
                e = std::max(e, std::abs(nutF[c] - ns));
                nutMax = std::max(nutMax, ns);
            }
            eN = e / nutMax;
            std::printf("  %4d   %.3e     %.3e                    %.3e                 %.3e\n", n, eS, eO, eN, nutMax);
        }
        check(eS <= 3.0 * eO, std::format("{}: state after {} steps {:.2e} <= 3 x FP32 own rounding {:.2e}", m.name,
                                          kN, eS, eO));
        const double gN = m.sgs == SgsModel::Sigma ? 5e-4 : 5e-5;
        check(eN <= gN, std::format("{}: mu_t field {:.2e} of max <= {:.0e} (FP32 rounding of the operator)", m.name,
                                    eN, gN));
    }
}

void x3()
{
    std::printf("X3 sinusoidal periodic 2D mesh, ladder, FP64: FV - structured, t = 0.1\n");
    double prevR = 0.0, prevS = 0.0;
    double ordR = 0.0, ordS = 0.0;
    std::printf("     N   |rhs diff|/|rhs|  order    |state diff|/|q|  order\n");
    for (const int N : {32, 64, 128}) {
        const Grid g = Grid::make(N, N, 1, 2 * pi, 2 * pi, 0.3);
        const CurvMesh cm = CurvMesh::sinusoidal(g, 0.04);
        const Transport tr = Transport::constant(kGamma, 0.02);
        auto host = orDie(Comp::make(cm, kGamma, tr.muRef), "x3");
        const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(cm, periodicSpec(g), CellOrder::Morton), "x3");
        CompState q = CompState::zeros(g);
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                // The node's own reference coordinate: the same smooth state on any N.
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
                const double rho = 1.0 + 0.1 * std::sin(x) * std::cos(y);
                const double u = 0.3 + 0.2 * std::sin(x) * std::cos(y), v = -0.2 * std::cos(x) * std::sin(y);
                const double p = 1.0 + 0.1 * std::cos(x + y);
                const std::size_t c = g.idx(i, j, 0);
                q.rho[c] = rho;
                q.mx[c] = rho * u;
                q.my[c] = rho * v;
                q.E[c] = p / (kGamma - 1.0) + 0.5 * rho * (u * u + v * v);
            }
        const FvState s0 = toFv(fm, q);
        CompState r = CompState::zeros(g);
        host.rhs(q, r);
        const FvState rs = toFv(fm, r);
        auto d = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = bcFor(fm, {}), .precision = FvPrecision::Fp64}), "x3");
        orDie(d.upload(s0), "x3");
        FvState rf;
        orDie(d.rhs(rf), "x3");
        const double eR = relDiff(rf.q, rs.q, 5, fm.nC);
        const double dt0 = 0.3 * orDie(host.stableDt(q, 1.0), "x3");
        const int steps = int(std::ceil(0.1 / dt0));
        const double dt = 0.1 / steps;
        orDie(d.upload(s0), "x3");
        CompState h = q;
        for (int n = 0; n < steps; ++n) {
            host.step(h, dt);
            d.step(dt);
        }
        FvState a;
        orDie(d.download(a), "x3");
        const double eS = relDiff(a.q, toFv(fm, h).q, 5, fm.nC);
        ordR = prevR > 0 ? std::log2(prevR / eR) : 0.0;
        ordS = prevS > 0 ? std::log2(prevS / eS) : 0.0;
        std::printf("  %4d   %.3e         %.2f     %.3e         %.2f\n", N, eR, ordR, eS, ordS);
        prevR = eR;
        prevS = eS;
    }
    check(ordS >= 1.7, std::format("state difference shrinks at order {:.2f} >= 1.7 (expected 2)", ordS));
}

#ifdef DNS_BIN
int runDns(const std::string& args) { return valtest::runBinary("\"" DNS_BIN "\" " + args + " > /dev/null"); }

double lastOf(const std::string& prefix, const char* col)
{
    const auto t = valtest::loadCsv(prefix + "_stats.csv");
    return t && !t->col(col).empty() ? t->col(col).back() : NAN;
}

// X5: the SETTLED gap. Far's outgoing invariant on c_P made the wake's entropy
// a pressure source: cd +4.8e-3, +5.8e-3, +6.0e-3 flat under h (docs/fvcomp.md
// "Far gap"). t 62.8: the wake is through the far row, cd flat to 2e-5.
void x5()
{
    std::printf("X5 O-grid cylinder Re 40 M 0.2, compensated, settled cd: FV - structured\n");
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "dns_fv_far";
    fs::create_directories(dir);
    const std::string base = "flow --physics compressible --ogrid --body cylinder --re 40 --mach 0.2 --ft 2 "
                             "--compensated --emit stats --quiet";
    double prev = 0.0, order = 0.0, d = 0.0;
    std::printf("     n   ny   cd structured   cd FV        difference   order\n");
    for (const auto [n, ny] : {std::pair{96, 50}, {192, 98}}) {
        const std::string a = (dir / std::format("s{}", n)).string(), b = (dir / std::format("f{}", n)).string();
        const std::string grid = std::format(" --n {} --ny {} --out ", n, ny);
        const int ra = runDns(base + grid + a), rb = runDns(base + " --solver fv" + grid + b);
        check(ra == 0 && rb == 0, std::format("level {}: both exit 0 ({}, {})", n, ra, rb));
        const double cs = lastOf(a, "cd"), cf = lastOf(b, "cd");
        d = std::abs(cf - cs);
        order = prev > 0 ? std::log2(prev / d) : 0.0;
        std::printf("  %4d  %3d   %.6f      %.6f     %.3e    %.2f\n", n, ny, cs, cf, cf - cs, order);
        prev = d;
    }
    check(d <= 6e-4, std::format("settled gap {:.2e} <= 6e-4 (the plateau was 5.8e-3)", d));
    check(order >= 1.5, std::format("settled gap shrinks at order {:.2f} >= 1.5", order));
    fs::remove_all(dir);
}

void x4()
{
    std::printf("X4 O-grid cylinder Re 40 M 0.2, compensated, cd at t = 12.6: FV - structured\n");
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "dns_fv_cross";
    fs::create_directories(dir);
    const std::string base = "flow --physics compressible --ogrid --body cylinder --re 40 --mach 0.2 --ft 0.4 "
                             "--compensated --emit stats --quiet";
    double prev = 0.0, order = 0.0;
    std::printf("     n   ny   cd structured   cd FV        difference   order\n");
    for (const auto [n, ny] : {std::pair{64, 34}, {128, 66}, {256, 130}}) {
        const std::string a = (dir / std::format("s{}", n)).string(), b = (dir / std::format("f{}", n)).string();
        const std::string grid = std::format(" --n {} --ny {} --out ", n, ny);
        const int ra = runDns(base + grid + a), rb = runDns(base + " --solver fv" + grid + b);
        check(ra == 0 && rb == 0, std::format("level {}: both exit 0 ({}, {})", n, ra, rb));
        const double cs = lastOf(a, "cd"), cf = lastOf(b, "cd"), d = std::abs(cf - cs);
        order = prev > 0 ? std::log2(prev / d) : 0.0;
        std::printf("  %4d  %3d   %.6f      %.6f     %.3e    %.2f\n", n, ny, cs, cf, d, order);
        prev = d;
    }
    check(order >= X4_ORDER, std::format("difference shrinks at order {:.2f} >= {}", order, X4_ORDER));
    std::printf("X4 smoke: every --solver fv route\n");
    const std::string fv = "flow --physics compressible --solver fv --emit stats --quiet --re 200 --mach 0.2 ";
    const std::string poly = std::string("flow --physics compressible --polymesh " DNS_MESH_DATA "/cyl2d_snappy --bc ") +
                             DNS_EXAMPLES "/fvcomp_cyl2d_snappy.toml --re 40 --mach 0.2 --tt 1 --ramp 0.2 --emit stats --quiet";
    const std::pair<const char*, std::string> routes[] = {
        {"periodic box", fv + "--body none --n 32 --ny 32 --far periodic --kick 0.2 --ft 0.05"},
        {"Far box", fv + "--body none --n 32 --ny 32 --ft 0.05"},
        {"stretched channel", fv + "--body none --n 32 --ny 32 --stretch 2 --noslip --ft 0.05"},
        {"focal box", fv + "--body none --n 64 --ny 64 --focal 2,4,2,4,0,0 --focal-ratio 2 --ft 0.02"},
        {"3D O-grid LES sigma", fv + "--ogrid --body cylinder --n 48 --ny 18 --nz 8 --les 1.35 --sgs sigma --ft 0.01"},
        {"3D periodic LES smag, fp64", fv + "--body none --n 32 --ny 32 --nz 16 --far periodic --kick 0.2 --les 0.17 "
                                            "--fv-precision fp64 --ft 0.05"},
        {"snappy polyMesh", poly},
        {"O-grid URANS SA-neg", fv + "--ogrid --body cylinder --n 48 --ny 24 --ogrid-beta 3 --rans sa-neg --ft 0.01"},
        {"O-grid URANS SST-2003m compensated",
         fv + "--ogrid --body cylinder --n 48 --ny 24 --ogrid-beta 3 --rans sst-2003m --compensated --ft 0.01"},
        {"stretched channel URANS SST-Vm", fv + "--body none --n 32 --ny 32 --stretch 2 --noslip --rans sst-vm --ft 0.02"},
        {"snappy polyMesh URANS SST-2003m", poly + " --rans sst-2003m"},
        {"snappy polyMesh URANS SA-neg, fp64", poly + " --rans sa-neg --fv-precision fp64"}};
    int i = 0;
    for (const auto& [name, args] : routes) {
        const std::string out = (dir / std::format("r{}", i++)).string();
        const int rc = runDns(args + " --out " + out);
        const double ke = lastOf(out, "kinetic");
        check(rc == 0 && std::isfinite(ke) && ke >= 0.0, std::format("{}: exit {}, kinetic {:.4g}", name, rc, ke));
    }
    // --steady: a real stop here costs minutes; the cap is exit 8 by contract.
    const std::string st = (dir / "steady").string();
    const int rc = runDns(fv + "--ogrid --body cylinder --n 48 --ny 24 --ogrid-beta 3 --rans sa-neg --steady 1e-3 "
                               "--steady-steps 400 --out " + st);
    const double ke = lastOf(st, "kinetic");
    check(rc == 8 && std::isfinite(ke) && ke >= 0.0, std::format("O-grid --steady at its cap: exit {}, kinetic {:.4g}", rc, ke));
    fs::remove_all(dir);
}
#endif

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
#ifdef DNS_BIN
    if (argc > 1 && std::strcmp(argv[1], "ladder") == 0) {
        x4();
        return dnstest::finish();
    }
    if (argc > 1 && std::strcmp(argv[1], "far") == 0) {
        x5();
        return dnstest::finish();
    }
#else
    (void)argc;
    (void)argv;
#endif
    x1();
    x2();
    x3();
    x6();
    return dnstest::finish();
}
