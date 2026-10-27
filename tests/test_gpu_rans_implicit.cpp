// The implicit steady RANS march (--steady-scheme implicit, docs/rans.md
// "Implicit pseudo-time") on the device, against the FP64 oracle:
//   I1  drift: the GPU FP32 residual vs the host line operator (FP64) on a
//       non-trivial state (v != 0, p non-uniform), per equation
//   I2  the line Jacobian against a central FD of the GPU residual
//   I3  the oracle is a fixed point: one iteration moves ub < 1e-7, mass pinned
//   I4  cold starts Re_tau 180 ny 64, all three closures, through the binary;
//       in process: the device state stays column-uniform bit for bit, R_mz
//       and w stay exactly 0 at every iteration
//   I5  cold starts Re_tau 1e4 ny 128 (SA, SST-2003m) through the binary
//   I5b SA Re_tau 1e5 ny 2048 stretch 4 through the binary == the oracle
//   I6  the residual is bitwise column-uniform at the IC on nx 6 x nz 2, and
//       that grid's iterate equals the nx 4 x nz 1 one
//   I7  TOL 1e-6 certifies (Re_tau 180 SA)
// `test_gpu_rans_implicit measure` prints the masterplan's P2 first
// measurement instead: the jitter of the stop's integrals over 30 iterations
// started AT the oracle. Seconds.
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_channel.hpp"
#include "dns/gpu/reductions.hpp"
#include "channel_line_test_util.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

using namespace dns;
using dnstest::check;
using gpu::Real;

namespace {

Channel1dCase caseOf(RansModel model, double reTau, int ny, double beta)
{
    Channel1dCase cc;
    cc.model = model;
    cc.reTau = reTau;
    cc.ny = ny;
    cc.beta = beta;
    cc.tol = 1e-8;
    return cc;
}

std::string nameOf(const Channel1dCase& cc)
{
    return std::format("{} Re_tau {:g} ny {}", ransName(cc.model), cc.reTau, cc.ny);
}

// The runner's device wiring (gpu/runner_channel.hip, implicit branch) around
// the host line of the same grid: plain state, wall stage hook, body force.
struct GpuRig
{
    explicit GpuRig(linetest::LineRig l) : L(std::move(l)) {}

    linetest::LineRig L;
    gpu::GpuGrid gg{};
    std::unique_ptr<gpu::DeviceComp> dev;
    std::unique_ptr<gpu::DeviceCompState> ds;
    std::unique_ptr<gpu::DeviceChannelWall> wall;
    std::unique_ptr<gpu::ReduceScratch> red;
    gpu::DeviceBuffer<Real> xfer;
    gpu::DeviceBuffer<double> sp, s0, s1;
    std::vector<Real> packed, Rf;
    std::vector<double> ps, p0, p1;
    gpu::ResidualNorms norms;
    // Invariants observed at every residual evaluation (I4).
    bool mzZero = true, wZero = true, uniform = true;
    bool watch = false; // download the whole state each evaluation
};

std::unique_ptr<GpuRig> makeGpuRig(const Channel1dCase& cc, int nx, int nz,
                                   double lx = 4 * std::numbers::pi,
                                   double lz = 4 * std::numbers::pi / 3)
{
    auto r = std::make_unique<GpuRig>(linetest::makeLine(cc, nx, nz, lx, lz));
    const auto& L = r->L;
    r->gg = gpu::GpuGrid::from(L.g);
    auto dev = gpu::DeviceComp::make(L.m, kGammaAir, L.mu);
    auto ds = gpu::DeviceCompState::make(L.g.size(), ransScalars(cc.model), false);
    auto wall = gpu::DeviceChannelWall::make(
        L.g, {kGammaAir, kRgasBox, L.Twall, true, L.omegaCoef});
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !wall || !red) return nullptr;
    r->dev = std::make_unique<gpu::DeviceComp>(std::move(*dev));
    r->ds = std::make_unique<gpu::DeviceCompState>(std::move(*ds));
    r->wall = std::make_unique<gpu::DeviceChannelWall>(std::move(*wall));
    r->red = std::make_unique<gpu::ReduceScratch>(std::move(*red));
    if (!r->dev->enableRans({.model = cc.model, .wallDist = L.d,
                             .mutHook = gpu::mutMirrorYHook(r->gg, true, true),
                             .fluidMask = gpu::rowMask(L.g, 1, L.g.ny - 2)}))
        return nullptr;
    GpuRig* p = r.get();
    p->dev->setStageHook([p](gpu::DeviceCompState& s) { p->wall->apply(s); });
    p->dev->setRhsHook([p](const gpu::DeviceCompState&, gpu::DeviceCompState& o,
                           const gpu::StageCombine& cmb) {
        gpu::addBodyForceX(p->gg, 1.0, cmb, p->dev->u(), o);
    });
    return r;
}

// float(q) into every column's fluid rows, then the wall's ghosts.
bool uploadColumn(GpuRig& r, const std::vector<double>& q)
{
    const int F = r.L.line.fields(), ny = r.L.g.ny;
    r.packed.assign(std::size_t(F) * ny, 0.0f);
    for (int j = 0; j < ny; ++j)
        for (int v = 0; v < F; ++v)
            r.packed[std::size_t(v) * ny + j] = Real(q[std::size_t(j) * F + v]);
    if (gpu::channelSetRows(r.gg, r.packed, r.xfer, *r.ds) != hipSuccess) return false;
    r.wall->apply(*r.ds);
    return true;
}

// Is every (i, k) column of the fluid rows bitwise column (0, 0)?
bool columnUniform(const GpuRig& r, const std::vector<std::vector<double>>& f)
{
    const Grid& g = r.L.g;
    for (const auto& a : f)
        for (int k = 0; k < g.nz; ++k)
            for (int j = 1; j <= g.ny - 2; ++j)
                for (int i = 0; i < g.nx; ++i)
                    if (std::bit_cast<std::uint64_t>(a[g.idx(i, j, k)]) !=
                        std::bit_cast<std::uint64_t>(a[g.idx(0, j, 0)]))
                        return false;
    return true;
}

// Every field of a device state, downloaded (plain: the exact floats).
std::vector<std::vector<double>> fieldsOf(const GpuRig& r,
                                          const gpu::DeviceCompState& s)
{
    auto h = CompState::zeros(r.L.g);
    (void)s.download(h);
    std::vector<std::vector<double>> f = {h.rho, h.mx, h.my, h.mz, h.E};
    for (int i = 0; i < s.nT; ++i) {
        std::vector<double> t;
        (void)s.downloadTurb(i, t);
        f.push_back(std::move(t));
    }
    return f;
}

// The GPU residual of q (column 0) in the line's layout; false when not
// finite on a fluid row.
bool gpuResidual(GpuRig& r, const std::vector<double>& q, std::vector<double>& R)
{
    const int F = r.L.line.fields(), ny = r.L.g.ny;
    if (!uploadColumn(r, q)) return false;
    r.norms = r.dev->residual(*r.ds, 1, ny - 2, *r.red);
    if (gpu::channelGatherColumn(r.gg, r.dev->residualField(), r.xfer, r.Rf) !=
        hipSuccess)
        return false;
    std::ranges::fill(R, 0.0);
    bool finite = true;
    for (int j = 1; j <= ny - 2; ++j)
        for (int v = 0; v < F; ++v) {
            const double x = r.Rf[std::size_t(v) * ny + j];
            R[std::size_t(j) * F + v] = x;
            finite = finite && std::isfinite(x);
            if (v == 3) r.mzZero = r.mzZero && x == 0.0;
        }
    if (r.watch) {
        const auto s = fieldsOf(r, *r.ds);
        r.wZero = r.wZero && std::ranges::all_of(s[3], [](double x) { return x == 0.0; });
        r.uniform = r.uniform && columnUniform(r, s);
    }
    return finite;
}

double residNorm(const GpuRig& r)
{
    const bool sst = ransSst(r.L.cc.model);
    const double ubEst = channelUbEstimate(r.L.cc.reTau);
    return std::max({r.norms.mom, r.norms.E / ubEst,
                     sst ? std::max(r.norms.t[0], r.norms.t[1]) : r.norms.t[0]});
}

// The stop's integrals measured on the DEVICE state, as the runner does
// (plane sums of the FP32 fields in FP64). Pre: the state is uploaded.
linetest::Integrals gpuIntegrals(GpuRig& r)
{
    const Grid& g = r.L.g;
    r.dev->primitives(*r.ds);
    (void)gpu::channelPlaneStats(r.gg, r.ds->rho, r.dev->u(), r.dev->v(),
                                 r.dev->w(), r.dev->T(), r.sp, r.ps);
    (void)gpu::channelPlaneScalarSums(r.gg, r.ds->t[0], r.s0, r.p0);
    if (r.ds->nT > 1) (void)gpu::channelPlaneScalarSums(r.gg, r.ds->t[1], r.s1, r.p1);
    linetest::Integrals I;
    double mass = 0, mom = 0, heat = 0;
    for (int j = 1; j <= g.ny - 2; ++j) {
        const double dy = r.L.dy[j];
        mass += dy * r.ps[j * 8];
        mom += dy * r.ps[j * 8 + 1];
        heat += dy * r.ps[j * 8 + 7];
        I.t[0] += dy * r.p0[j * 2];
        if (r.ds->nT > 1) I.t[1] += dy * r.p1[j * 2];
    }
    I.ub = mom / mass;
    I.Tb = heat / mass;
    return I;
}

linetest::Drive gpuDrive(GpuRig& r, std::vector<double> q, double tol, int maxIt)
{
    return linetest::driveWith(
        r.L, std::move(q), tol, maxIt,
        [&](const std::vector<double>& x, std::vector<double>& R) {
            return gpuResidual(r, x, R);
        },
        [&](const std::vector<double>&) { return gpuIntegrals(r); });
}

// The non-trivial state of I1/I2: the oracle x (1 + 1% sine modes), v != 0
// of one sign (no face on the upwind kink), p non-uniform.
std::vector<double> roughState(const linetest::LineRig& L, const Channel1dSolution& s)
{
    auto q = linetest::oracleColumn(L, s);
    const int F = L.line.fields(), N = L.g.ny;
    for (int j = 1; j <= N - 2; ++j) {
        double* x = q.data() + std::size_t(j) * F;
        const double y = double(j) / (N - 1);
        const double a = 1 + 0.01 * std::sin(7.3 * y + 0.4);
        x[0] *= a;
        x[1] *= a * (1 + 0.01 * std::cos(5.1 * y));
        x[2] = x[0] * (0.02 + 0.01 * std::sin(3 * y));
        x[4] *= a * (1 + 0.005 * std::sin(4.7 * y));
        for (int t = 5; t < F; ++t) x[t] *= a * (1 + 0.01 * std::sin(2.3 * y + t));
    }
    L.line.wall(q);
    return q;
}

// float-rounded fluid rows, FP64 ghosts: the state the device actually holds.
std::vector<double> rounded(const linetest::LineRig& L, std::vector<double> q)
{
    for (double& x : q) x = double(float(x));
    L.line.wall(q);
    return q;
}

// --- I1 ---------------------------------------------------------------------
// Per-equation thresholds: 10x the measured drift (see main()).
void drift(const Channel1dCase& cc, const std::vector<double>& bound)
{
    const auto s = solveChannel1d(cc);
    auto r = makeGpuRig(cc, 4, 1);
    if (!s || !r) { check(false, "I1 setup " + nameOf(cc)); return; }
    const auto q = rounded(r->L, roughState(r->L, *s));
    const int F = r->L.line.fields(), N = r->L.g.ny;
    std::vector<double> Rg(q.size()), Rh(q.size());
    gpuResidual(*r, q, Rg);
    r->L.line.residual(q, Rh);
    std::string line = "  I1 " + nameOf(cc) + ": max|dR|/max|R_line|";
    bool ok = true;
    for (int v = 0; v < F; ++v) {
        if (v == 3) continue; // rho w: both exactly 0 at w = 0
        double d = 0, m = 0;
        for (int j = 1; j <= N - 2; ++j) {
            const std::size_t k = std::size_t(j) * F + v;
            d = std::max(d, std::abs(Rg[k] - Rh[k]));
            m = std::max(m, std::abs(Rh[k]));
        }
        const double rel = d / m;
        line += std::format(" {:.2e}", rel);
        ok = ok && rel <= bound[v];
    }
    std::printf("%s\n", line.c_str());
    check(ok, "I1 " + nameOf(cc) + ": GPU vs line residual within 10x the measured drift");
}

// --- I2 ---------------------------------------------------------------------
void jacobianGpu(const Channel1dCase& cc)
{
    const auto s = solveChannel1d(cc);
    auto r = makeGpuRig(cc, 4, 1);
    if (!s || !r) { check(false, "I2 setup " + nameOf(cc)); return; }
    const auto q = rounded(r->L, roughState(r->L, *s));
    const auto& L = r->L.line;
    const int F = L.fields(), B = L.unknowns(), N = r->L.g.ny, K = L.band();
    // q +- 2^-12 delta, rounded to float; the effective step is what the
    // device sees, so the FD is the exact directional derivative along it.
    std::vector<double> qp = q, qm = q, d((N - 2) * std::size_t(B));
    for (int j = 1; j <= N - 2; ++j)
        for (int u = 0; u < B; ++u) {
            const std::size_t k = std::size_t(j) * F + ChannelLine::field(u);
            const double x = q[k], rho = q[std::size_t(j) * F];
            const double sc = u == 0 || u == 3 ? std::abs(x)
                                               : std::max(std::abs(x), 1e-3 * rho);
            const double e = 0x1p-12 * sc * std::sin(3.1 * j / N + 0.7 * u);
            qp[k] = double(float(x + e));
            qm[k] = double(float(x - e));
            d[std::size_t(j - 1) * B + u] = 0.5 * (qp[k] - qm[k]);
        }
    L.wall(qp);
    L.wall(qm);
    std::vector<double> Rp(q.size()), Rm(q.size());
    gpuResidual(*r, qp, Rp);
    gpuResidual(*r, qm, Rm);
    const BlockBand J = L.jacobian(q);
    double worst = 0;
    std::string line = "  I2 " + nameOf(cc) + ": max|J d - FD32|/max|FD32|";
    for (int u = 0; u < B; ++u) {
        double dm = 0, fm = 0;
        for (int i = 1; i <= N - 2; ++i) {
            double jd = 0;
            for (int o = -K; o <= K; ++o) {
                const int jj = i + o;
                if (jj < 1 || jj > N - 2) continue;
                for (int c = 0; c < B; ++c)
                    jd += J.at(i - 1, o, u, c) * d[std::size_t(jj - 1) * B + c];
            }
            const std::size_t k = std::size_t(i) * F + ChannelLine::field(u);
            const double fd = 0.5 * (Rp[k] - Rm[k]);
            dm = std::max(dm, std::abs(jd - fd));
            fm = std::max(fm, std::abs(fd));
        }
        line += std::format(" {:.2e}", dm / fm);
        worst = std::max(worst, dm / fm);
    }
    std::printf("%s\n", line.c_str());
    check(worst <= 2e-2, "I2 " + nameOf(cc) + ": line J vs GPU FP32 central FD <= 2e-2");
}

// --- I3 ---------------------------------------------------------------------
void fixedPoint(const Channel1dCase& cc)
{
    const auto s = solveChannel1d(cc);
    auto r = makeGpuRig(cc, 4, 1);
    if (!s || !r) { check(false, "I3 setup " + nameOf(cc)); return; }
    const auto q0 = linetest::oracleColumn(r->L, *s);
    const auto& L = r->L.line;
    const double M0 = L.mass(q0);
    std::vector<double> R(q0.size());
    gpuResidual(*r, q0, R);
    const double resid = residNorm(*r);
    auto dq = pseudoUpdate(L, q0, R, kImplicitCfl0);
    if (!dq) { check(false, "I3 solve"); return; }
    auto q1 = q0;
    L.advance(q1, *dq, stepLength(q0, *dq, L), M0);
    const double ub0 = linetest::integrals(r->L, q0).ub,
                 ub1 = linetest::integrals(r->L, q1).ub;
    std::printf("  I3 %s: resid at the oracle %.3f, one iteration dub/ub %.2e, "
                "mass %.2e\n",
                nameOf(cc).c_str(), resid, ub1 / ub0 - 1, L.mass(q1) / M0 - 1);
    check(std::abs(ub1 / ub0 - 1) <= 1e-7, "I3 " + nameOf(cc) + ": |dub/ub| <= 1e-7");
    check(std::abs(L.mass(q1) / M0 - 1) <= 1e-12, "I3 " + nameOf(cc) + ": mass = M0");
    check(resid <= 0.1, "I3 " + nameOf(cc) + ": resid <= 0.1 (FP32 static floor)");
}

// --- the binary --------------------------------------------------------------
struct BinRun
{
    int rc = -1;
    double iter = 0, ub = 0, remaining = 0, floorHits = -1, rejects = -1, secs = 0;
};

BinRun runImplicit(const Channel1dCase& cc, double tol, const std::string& prefix)
{
    const std::string cli = ransCliNames()[std::size_t(cc.model)];
    const std::string cmd = std::format(
        "\"{}\" channel --re-tau {:g} --ny {} --stretch {:g} --n 4 --nz 1 "
        "--rans {} --steady {:g} --steady-scheme implicit --out {}",
        DNS_BIN, cc.reTau, cc.ny, cc.beta, cli, tol, prefix);
    BinRun b;
    const std::string log = prefix + ".log";
    const auto t0 = std::chrono::steady_clock::now();
    b.rc = valtest::runBinaryTo(cmd, log);
    b.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (const auto tbl = valtest::loadCsv(prefix + "_stats.csv")) {
        b.iter = tbl->col("iter").back();
        b.remaining = tbl->col("remaining").back();
        b.rejects = tbl->col("rejects").back();
        if (ransSst(cc.model)) b.floorHits = tbl->col("floorHits").back();
    }
    // ub from the last progress line (7 decimals; the CSV carries 6 digits).
    std::ifstream f(log);
    for (std::string l; std::getline(f, l);)
        if (l.starts_with("iter=")) {
            const auto p = l.find(" ub=");
            if (p != std::string::npos) b.ub = std::strtod(l.c_str() + p + 4, nullptr);
        }
    std::error_code ec;
    for (const char* suf : {".json", "_stats.csv", "_profiles.csv", "_mask.bin", ".log"})
        std::filesystem::remove(prefix + suf, ec);
    return b;
}

void coldStart(const Channel1dCase& cc, double tol, double ubBand, int maxIter,
               double maxSecs, const std::string& id)
{
    const auto s = solveChannel1d(cc);
    if (!s) { check(false, id + " oracle"); return; }
    const auto b = runImplicit(cc, tol, "gpu_rans_implicit_" + id);
    const double err = b.ub / s->ub - 1;
    std::printf("  %s %s TOL %g: rc %d, %.0f iterations, ub %.7f oracle %.7f "
                "(%+.2e), remaining %.2e, rejects %.0f, floorHits %.0f, %.2f s\n",
                id.c_str(), nameOf(cc).c_str(), tol, b.rc, b.iter, b.ub, s->ub,
                err, b.remaining, b.rejects, b.floorHits, b.secs);
    const std::string w = id + " " + nameOf(cc);
    check(b.rc == 0, w + ": exit 0");
    check(b.iter <= maxIter, std::format("{}: <= {} iterations", w, maxIter));
    check(std::abs(err) <= ubBand, std::format("{}: |ub/ub_o - 1| <= {:g}", w, ubBand));
    check(b.rejects == 0, w + ": no rejections");
    if (ransSst(cc.model)) check(b.floorHits == 0, w + ": 0 SST guard hits");
    if (maxSecs > 0) check(b.secs < maxSecs, std::format("{}: < {:g} s", w, maxSecs));
}

// --- I4 (in process) --------------------------------------------------------
void invariants(const Channel1dCase& cc)
{
    auto r = makeGpuRig(cc, 4, 1);
    if (!r) { check(false, "I4 setup"); return; }
    r->watch = true;
    const auto d = gpuDrive(*r, linetest::icColumn(r->L), 1e-4, 40);
    std::printf("  I4 %s in process: %d iterations, certified %d\n",
                nameOf(cc).c_str(), d.iters, int(d.certified));
    check(r->mzZero, "I4 " + nameOf(cc) + ": gathered R_mz == 0 at every iteration");
    check(r->wZero, "I4 " + nameOf(cc) + ": device w == 0 at every iteration");
    check(r->uniform, "I4 " + nameOf(cc) + ": device state column-uniform, bitwise, "
                      "at every iteration");
}

// --- I6 -----------------------------------------------------------------------
// Both boxes have the SAME cells (hx = pi, hz = 1): a different cell would
// round the FP32 per-volume residual differently (S_y = hx hz, 1/vol), a
// 1e-8 difference the march then carries -- not a homogeneity question.
void homogeneity()
{
    using std::numbers::pi;
    const auto cc = caseOf(RansModel::SaNeg, 180, 64, 2);
    auto a = makeGpuRig(cc, 4, 1, 4 * pi, 1.0), b = makeGpuRig(cc, 6, 2, 6 * pi, 2.0);
    if (!a || !b) { check(false, "I6 setup"); return; }
    const auto q0 = linetest::icColumn(b->L);
    std::vector<double> R(q0.size());
    gpuResidual(*b, q0, R);
    check(columnUniform(*b, fieldsOf(*b, b->dev->residualField())),
          "I6: IC residual on nx 6 x nz 2 is bitwise column-uniform");
    // A tolerance no stop can reach: exactly 30 iterations each.
    const auto da = gpuDrive(*a, linetest::icColumn(a->L), 1e-300, 30);
    const auto db = gpuDrive(*b, q0, 1e-300, 30);
    // Per field, relative to the field's largest magnitude.
    const int F = a->L.line.fields();
    double worst = 0;
    for (int v = 0; v < F; ++v) {
        double d = 0, m = 0;
        for (int j = 0; j < a->L.g.ny; ++j) {
            const std::size_t k = std::size_t(j) * F + v;
            d = std::max(d, std::abs(da.q[k] - db.q[k]));
            m = std::max(m, std::abs(da.q[k]));
        }
        if (m > 0) worst = std::max(worst, d / m);
    }
    std::printf("  I6: host columns after 30 iterations, nx 4 nz 1 vs nx 6 nz 2 "
                "(same cells): max |diff| / max |field| %.2e\n", worst);
    check(da.iters == 30 && db.iters == 30 && worst <= 1e-12,
          "I6: the nx 6 x nz 2 iterate equals the nx 4 x nz 1 one to 1e-12");
}

// --- the masterplan's P2 first measurement -----------------------------------
// 30 iterations from the uploaded oracle state; jitter (max - min)/|mean| of
// each stop integral over all 30 samples and over the armed ones.
void firstMeasurement(const Channel1dCase& cc)
{
    const auto s = solveChannel1d(cc);
    auto r = makeGpuRig(cc, 4, 1);
    if (!s || !r) { check(false, "measurement setup"); return; }
    double residMax = 0, residMin = HUGE_VAL;
    const auto d = linetest::driveWith(
        r->L, linetest::oracleColumn(r->L, *s), 1e-300, 30,
        [&](const std::vector<double>& x, std::vector<double>& R) {
            const bool ok = gpuResidual(*r, x, R);
            residMax = std::max(residMax, residNorm(*r));
            residMin = std::min(residMin, residNorm(*r));
            return ok;
        },
        [&](const std::vector<double>&) { return gpuIntegrals(*r); });
    const int nS = ransSst(cc.model) ? 4 : 3;
    auto jitter = [&](int k, bool armedOnly) {
        double lo = HUGE_VAL, hi = -HUGE_VAL, sum = 0;
        int n = 0;
        for (std::size_t i = 0; i < d.hist.size(); ++i) {
            if (armedOnly && !d.armed[i]) continue;
            const auto& I = d.hist[i];
            const double v = k == 0 ? I.ub : k == 1 ? I.Tb : I.t[k - 2];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            sum += v;
            ++n;
        }
        return n ? (hi - lo) / std::abs(sum / n) : 0.0;
    };
    const char* names[4] = {"ub", "Tbulk", "turb0", "turb1"};
    std::printf("  P2 measurement %s: %d iterations from the oracle, resid "
                "%.4f-%.4f\n",
                nameOf(cc).c_str(), d.iters, residMin, residMax);
    double worst = 0;
    for (int k = 0; k < nS; ++k) {
        std::printf("    %-6s jitter all %.3e  armed %.3e\n", names[k], jitter(k, false),
                    jitter(k, true));
        worst = std::max(worst, jitter(k, false));
    }
    std::printf("    worst %.3e vs 2^-21 = %.3e: %s\n", worst, 0x1p-21,
                worst <= 0x1p-21 ? "below" : "ABOVE -- stop and report");
}

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();
    if (argc > 1 && std::string(argv[1]) == "measure") {
        firstMeasurement(caseOf(RansModel::SaNeg, 180, 64, 2));
        firstMeasurement(caseOf(RansModel::Sst2003m, 180, 64, 2));
        return 0;
    }
    // I1 bounds per field (rho, mx, my, mz, E, t0, t1): 10x the measured
    // max|R_gpu - R_line| / max|R_line| (gfx1100, 2026-10-25), rounded up.
    // Measured SA: 1.26e-7 2.71e-5 6.48e-5 - 1.61e-5 2.63e-5; SST-2003m:
    // 1.34e-7 2.23e-5 9.52e-5 - 6.38e-6 4.01e-5 6.91e-6 -- every one inside
    // the masterplan's 1e-4 expectation.
    const std::vector<double> driftBound = {2e-6, 3e-4, 1e-3, 0, 2e-4, 5e-4, 7e-5};
    drift(caseOf(RansModel::SaNeg, 180, 64, 2), driftBound);
    drift(caseOf(RansModel::Sst2003m, 180, 64, 2), driftBound);
    jacobianGpu(caseOf(RansModel::SaNeg, 180, 64, 2));
    jacobianGpu(caseOf(RansModel::Sst2003m, 180, 64, 2));
    fixedPoint(caseOf(RansModel::SaNeg, 180, 64, 2));
    fixedPoint(caseOf(RansModel::Sst2003m, 180, 64, 2));
    for (RansModel m : {RansModel::SaNeg, RansModel::Sst2003m, RansModel::SstVm}) {
        coldStart(caseOf(m, 180, 64, 2), 1e-4, 2e-6, 40, 0, "I4");
        invariants(caseOf(m, 180, 64, 2));
    }
    for (RansModel m : {RansModel::SaNeg, RansModel::Sst2003m})
        coldStart(caseOf(m, 1e4, 128, 4), 1e-4, 5e-6, 40, 2.0, "I5");
    // I5b: the CFL3D-comparison regime (rans_channel1d O5) through the binary,
    // Re_tau 1e5 ny 2048 stretch 4: the binary IS the oracle on its grid.
    // Measured (gfx1100, 2026-10-27, TOL 1e-6): 27 iterations, ub -2.5e-8
    // off, 0.23 s.
    coldStart(caseOf(RansModel::SaNeg, 1e5, 2048, 4), 1e-6, 1e-6, 40, 2.0, "I5b");
    homogeneity();
    coldStart(caseOf(RansModel::SaNeg, 180, 64, 2), 1e-6, 2e-6, 40, 0, "I7");
    return dnstest::finish();
}
