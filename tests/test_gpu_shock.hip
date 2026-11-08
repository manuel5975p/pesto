// --shock on the device (masterplan_shockflux section 8), driven through the BUILT
// binary. `test_gpu_shock <case>`:
//   sense   gate 11: --shock sense vs none, flame + chamber + the comp row: every
//           artefact byte equal after the appended shock columns are cut.
//   sensor  gate 14 (amendment A2), device, whole run at stats cadence, past the IC
//           (t >= 1): planar sigma 5, 10 at 4 cells, chamber sigma 5, 10 defaults.
//           (a) v_max in burning cells <= 0.0133, (b) 0 sensed cavity cells off
//           the Exit column, at most the 2 lip cells. Sigma 10's default diverges
//           at the outflow (issues.md S2): rc 7 there is the run's end, not the
//           gate's; it reads the rows it wrote.
// The device rigs (gates 15+) are on the direct API: rows held and sources by the
// kernels below. 1D rigs are nx x 4 x 1. FP32. A HIP TU.
#include "check.hpp"
#include "cli_test_util.hpp"
#include "gpu_test_util.hpp"
#include "restart_test_util.hpp"
#include "shock_ref.hpp"

#include "dns/comp.hpp"
#include "dns/flame1d.hpp"
#include "dns/gpu/chem.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/comp_sink.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/internal/stage_write.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must point at the dns binary"
#endif
#ifndef DNS_TEST_DIR
#error "DNS_TEST_DIR must name a scratch directory"
#endif

using dnstest::check;
namespace fs = std::filesystem;
namespace G = dns::gpu;
using G::GpuGrid;
using G::Real;

namespace {

// Rows i < rx, i >= nx - rxHi, j < ry, j >= ny - ry of q := h.
__global__ void kHoldRows(GpuGrid g, int rx, int rxHi, int ry, const Real* h0, const Real* h1, const Real* h2,
                          const Real* h3, const Real* h4, Real* q0, Real* q1, Real* q2, Real* q3, Real* q4)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (!(i < rx || i >= g.nx - rxHi || j < ry || j >= g.ny - ry)) return;
    q0[c] = h0[c];
    q1[c] = h1[c];
    q2[c] = h2[c];
    q3[c] = h3[c];
    q4[c] = h4[c];
}

// One solver, one state, a held copy of the IC.
struct Rig
{
    dns::Grid g;
    GpuGrid gg;
    std::optional<G::DeviceComp> dev;
    std::optional<G::DeviceCompState> s, held;
    std::optional<G::ReduceScratch> red;
    int rx = 0, ry = 0;
    bool shockOn = false;
    int rxHi = -1; // x-hi rows held; < 0 = rx

    void hold(G::DeviceCompState& q)
    {
        const int hi = rxHi < 0 ? rx : rxHi;
        if (!rx && !hi && !ry) return;
        kHoldRows<<<G::launchBlocks(g.size()), G::blockSize>>>(gg, rx, hi, ry, held->rho.data(), held->mx.data(),
                                                               held->my.data(), held->mz.data(), held->E.data(),
                                                               q.rho.data(), q.mx.data(), q.my.data(), q.mz.data(),
                                                               q.E.data());
        G::checkLastKernel("kHoldRows");
    }
};

std::unique_ptr<Rig> makeRig(const dns::Grid& g, const dns::CompState& ic, int rx, int ry,
                             const std::optional<G::ShockSetup>& sh, double mu = 0, double cs = 0, bool comp = false)
{
    auto r = std::make_unique<Rig>(Rig{g, GpuGrid::from(g), {}, {}, {}, {}, rx, ry, sh.has_value()});
    auto d = G::DeviceComp::make(g, shockref::kGamma, mu, cs);
    auto s = G::DeviceCompState::make(g.size(), 0, comp);
    auto h = G::DeviceCompState::make(g.size());
    auto q = G::ReduceScratch::make();
    if (!d || !s || !h || !q) dnstest::skip("device allocation failed");
    r->dev = std::move(*d);
    r->s = std::move(*s);
    r->held = std::move(*h);
    r->red = std::move(*q);
    if (sh && !r->dev->enableShock(*sh)) dnstest::skip("device allocation failed");
    if (comp && !r->dev->enableCompensation()) dnstest::skip("device allocation failed");
    if (r->s->upload(ic) != hipSuccess || r->held->upload(ic) != hipSuccess) dnstest::skip("upload failed");
    Rig* p = r.get();
    r->dev->setStageHook([p](G::DeviceCompState& q2) { p->hold(q2); });
    return r;
}

struct Marched
{
    bool ok = true;
    std::string why;
    long steps = 0;
    double peakCells = 0, rhoMin = HUGE_VAL, pMin = HUGE_VAL;
};

// March to tEnd at cfl, dt every step. mon(t, steps) after each step. Under shock
// each step's input is read for positivity and sensed cells, once per `every`.
template <class Mon>
Marched march(Rig& r, double tEnd, double cfl, Mon&& mon, int every = 1)
{
    Marched m;
    double t = 0;
    while (t < tEnd - 1e-14) {
        r.hold(*r.s);
        const double dt = std::min(r.dev->stableDt(*r.s, *r.red, cfl), tEnd - t);
        if (!(dt > 0 && std::isfinite(dt))) {
            m.ok = false;
            m.why = std::format("dt {} at t={:.4g} step {}", dt, t, m.steps);
            return m;
        }
        if (r.shockOn && m.steps % every == 0) {
            const G::ShockRow row = r.dev->shockRow(*r.s, *r.red);
            m.peakCells = std::max(m.peakCells, row.cells);
            m.rhoMin = std::min(m.rhoMin, row.rhoMin);
            m.pMin = std::min(m.pMin, row.pMin);
            if (!(row.rhoMin > 0 && row.pMin > 0)) {
                m.ok = false;
                m.why = std::format("rho {:.3g}, p {:.3g} at t={:.4g} step {}", row.rhoMin, row.pMin, t, m.steps);
                return m;
            }
        }
        r.dev->step(*r.s, Real(dt));
        t += dt;
        ++m.steps;
        mon(t, m.steps);
    }
    r.hold(*r.s);
    return m;
}
Marched march(Rig& r, double tEnd, double cfl = 0.4)
{
    return march(r, tEnd, cfl, [](double, long) {});
}

dns::CompState download(Rig& r)
{
    auto h = dns::CompState::zeros(r.g);
    if (r.s->download(h) != hipSuccess) dnstest::skip("download failed");
    return h;
}

dns::Grid grid1D(int N, double L) { return dns::Grid::make(N, 4, 1, L, 4 * L / N, 1.0); }

// A 1D FP32 state on g's rows (y-uniform), bits preserved through the FP64 upload.
dns::CompState rowsOf(const dns::Grid& g, const shockref::State<float>& U)
{
    auto h = dns::CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k), I = std::size_t(i);
                h.rho[c] = U.q[0][I];
                h.mx[c] = U.q[1][I];
                h.E[c] = U.q[2][I];
            }
    return h;
}

G::ShockSetup openX()
{
    G::ShockSetup s;
    s.open[0] = true;
    return s;
}

// mean |rho - host| / mean |host| over row j = 0, and y uniform bitwise.
double twinError(const dns::Grid& g, const dns::CompState& d, const shockref::State<float>& h, bool& uniform)
{
    double e = 0, sc = 0;
    uniform = true;
    for (int i = 0; i < g.nx; ++i) {
        const double a = d.rho[g.idx(i, 0, 0)], b = h.q[0][std::size_t(i)];
        e += std::abs(a - b);
        sc += std::abs(b);
        for (int j = 1; j < g.ny; ++j) uniform = uniform && d.rho[g.idx(i, j, 0)] == a;
    }
    return e / sc;
}

int run(const std::string& args, const std::string& prefix)
{
    const std::string cmd = std::format("{} {} --out {} > {}.log 2>&1", DNS_BIN, args, prefix, prefix);
    const int st = std::system(cmd.c_str());
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

std::string slurp(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

std::vector<std::string> linesOf(const std::string& s)
{
    std::vector<std::string> out;
    std::stringstream in(s);
    for (std::string l; std::getline(in, l);) out.push_back(l);
    return out;
}

// Every line of `on` is the `off` line + the appended columns.
bool statsEqualAfterCut(const fs::path& off, const fs::path& on)
{
    const auto a = linesOf(slurp(off)), b = linesOf(slurp(on));
    if (a.empty() || a.size() != b.size()) return false;
    for (std::size_t q = 0; q < a.size(); ++q)
        if (!(b[q].starts_with(a[q]) && b[q].size() > a[q].size() && b[q][a[q].size()] == ',')) {
            std::printf("    line %zu differs\n    off %s\n    on  %s\n", q, a[q].c_str(), b[q].c_str());
            return false;
        }
    return true;
}

// The run's .json without wall time and the mode key.
std::string jsonOf(const fs::path& p)
{
    static const std::regex wall("\"msPerStep\":[0-9.eE+-]+"), mode(",\"shock\":\"[a-z]+\"");
    return std::regex_replace(std::regex_replace(slurp(p), wall, ""), mode, "");
}

void gateBytes(const std::string& tag, const std::string& args, const fs::path& dir,
               const std::string& mode = "--shock sense")
{
    const std::string off = (dir / (tag + "_none")).string(), on = (dir / (tag + "_sense")).string();
    check(run(args, off) == 0, tag + ": --shock none runs");
    check(run(args + " " + mode, on) == 0, tag + ": " + mode + " runs");
    int files = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        const std::string head = tag + "_none";
        if (!name.starts_with(head) || name.ends_with(".log")) continue;
        const fs::path twin = dir / (tag + "_sense" + name.substr(head.size()));
        ++files;
        if (name.ends_with("_stats.csv")) check(statsEqualAfterCut(e.path(), twin), tag + ": " + name + " after the cut");
        else if (name.ends_with(".json")) check(jsonOf(e.path()) == jsonOf(twin), tag + ": " + name);
        else check(slurp(e.path()) == slurp(twin), tag + ": " + name + " bytes");
    }
    check(files >= 2, tag + ": artefacts were compared");
}

int sense()
{
    const fs::path dir = fs::path(DNS_TEST_DIR) / "sense";
    fs::remove_all(dir);
    fs::create_directories(dir);
    gateBytes("flame", "flame --expansion 5 --cells-per-delta 4 --mach 0.02 --tt 1 --emit stats --emit bin", dir);
    gateBytes("chamber", "flame --mode chamber --tt 0.5 --emit stats --emit bin", dir);
    gateBytes("comp", "flow --physics compressible --body cylinder --n 192 --ny 128 --re 100 --mach 0.2 --ft 0.3 "
                      "--emit stats",
              dir);
    return dnstest::finish();
}

// Column `name` of a stats file, rows with t >= tMin: its maximum.
double columnMax(const fs::path& p, const std::string& name, double tMin)
{
    const auto lines = linesOf(slurp(p));
    std::size_t head = 0;
    while (head < lines.size() && lines[head].starts_with("#")) ++head;
    auto split = [](const std::string& l) {
        std::vector<std::string> c;
        std::stringstream s(l);
        for (std::string x; std::getline(s, x, ',');) c.push_back(x);
        return c;
    };
    const auto cols = split(lines.at(head));
    const std::size_t at = std::size_t(std::ranges::find(cols, name) - cols.begin());
    double m = -HUGE_VAL;
    for (std::size_t q = head + 1; q < lines.size(); ++q) {
        const auto r = split(lines[q]);
        if (std::stod(r.at(0)) >= tMin) m = std::max(m, std::stod(r.at(at)));
    }
    return m;
}

int sensor()
{
    const fs::path dir = fs::path(DNS_TEST_DIR) / "sensor";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const double band = 0.0133; // the flame's stop rule; prototype 3.12e-3
    for (const char* sig : {"5", "10"}) {
        const std::string p = (dir / std::format("planar{}", sig)).string();
        check(run(std::format("flame --expansion {} --cells-per-delta 4 --mach 0.02 --tt 20 --shock sense --emit stats", sig),
                  p) == 0,
              std::format("planar sigma {} runs", sig));
        const double v = columnMax(p + "_stats.csv", "shockBurnMax", 1.0);
        const double n = columnMax(p + "_stats.csv", "shockCells", 1.0);
        std::printf("  planar sigma %s burning v_max %.4e all v_max %.4e cells %.0f\n", sig, v,
                    columnMax(p + "_stats.csv", "shockMax", 1.0), n);
        check(v <= band, std::format("planar sigma {}: burning v_max {:.4e} <= {:g}", sig, v, band));
        check(n == 0, std::format("planar sigma {}: 0 cells sensed", sig));
    }
    for (const auto& [sig, rcOk] : {std::pair{"5", 0}, std::pair{"10", 7}}) {
        const std::string p = (dir / std::format("chamber{}", sig)).string();
        const int rc = run(std::format("flame --mode chamber --expansion {} --tt 60 --shock sense --emit stats", sig), p);
        check(rc == rcOk, std::format("chamber sigma {} rc {} (expected {})", sig, rc, rcOk));
        const double v = columnMax(p + "_stats.csv", "shockBurnMax", 1.0);
        const double rest = columnMax(p + "_stats.csv", "shockCavRest", 1.0);
        const double lip = columnMax(p + "_stats.csv", "shockLip", 1.0);
        const double all = columnMax(p + "_stats.csv", "shockCells", 1.0);
        std::printf("  chamber sigma %s burning v_max %.4e cavity v_max %.4e cavity off Exit %.0f lip column %.0f "
                    "all %.0f (%.4f%%)\n",
                    sig, v, columnMax(p + "_stats.csv", "shockCavMax", 1.0), rest, lip, all, 100 * all / 55328);
        std::system(std::format("grep -E '^shock: (cavity|at most|peak)' {}.log | tail -8", p).c_str());
        check(v <= band, std::format("chamber sigma {}: burning v_max {:.4e} <= {:g}", sig, v, band));
        check(rest == 0, std::format("chamber sigma {}: {:.0f} cavity cells sensed off the Exit column", sig, rest));
        check(lip <= 2, std::format("chamber sigma {}: {:.0f} lip cells, at most 2", sig, lip));
    }
    return dnstest::finish();
}

// Gate 15: the Riemann rungs at N 200, device vs host twin.
int riemann()
{
    for (const shockref::Tube& b : shockref::kTubes) {
        const int N = 200;
        const dns::Grid g = grid1D(N, 1.0);
        shockref::State<float> H;
        const shockref::Report hr = shockref::tubeRun<float>(b, N, H, nullptr, true);
        auto r = makeRig(g, rowsOf(g, shockref::tubeIc<float>(b, N)), 4, 0, openX());
        const Marched m = march(*r, b.t);
        bool uniform = false;
        const double e = twinError(g, download(*r), H, uniform);
        // The rounding scale of the rung: the host's own FP32 against its FP64.
        shockref::State<double> Hd;
        (void)shockref::tubeRun<double>(b, N, Hd, nullptr, true);
        double fp = 0, sc = 0;
        for (int i = 0; i < N; ++i) {
            fp += std::abs(double(H.q[0][std::size_t(i)]) - Hd.q[0][std::size_t(i)]);
            sc += std::abs(Hd.q[0][std::size_t(i)]);
        }
        std::printf("  %-6s device vs host %.3e (host FP32 vs FP64 %.3e) steps %ld/%ld max sensed %.4f%% rhoMin %.3g pMin "
                    "%.3g %s\n",
                    b.name, e, fp / sc, m.steps, hr.steps, 100 * m.peakCells / double(g.size()), m.rhoMin, m.pMin,
                    m.why.c_str());
        check(m.ok && hr.ok, std::format("{}: marches, rho, p > 0", b.name));
        // A5: the near-vacuum rungs are bounded by the rung's own FP32 rounding scale.
        const std::string_view nm = b.name;
        const double band = nm == "u5" || nm == "u8" || nm == "u16" ? 2 * fp / sc : 1e-5;
        check(e <= band, std::format("{}: device vs host {:.3e} <= {:.3e}", b.name, e, band));
        check(uniform, std::format("{}: y uniform bitwise", b.name));
    }
    return dnstest::finish();
}

// Gate 16: Shu-Osher N 400.
int shuosher()
{
    const int N = 400;
    const dns::Grid g = grid1D(N, 10.0);
    shockref::State<float> H = shockref::shuOsherIc<float>(N);
    const shockref::Report hr = shockref::heldRun(H, 10.0 / N, 1.8, true);
    auto r = makeRig(g, rowsOf(g, shockref::shuOsherIc<float>(N)), 4, 0, openX());
    const Marched m = march(*r, 1.8);
    bool uniform = false;
    const double e = twinError(g, download(*r), H, uniform);
    std::printf("  Shu-Osher N 400 device vs host %.3e max sensed %.3f%% %s\n", e, 100 * m.peakCells / double(g.size()),
                m.why.c_str());
    check(m.ok && hr.ok, "marches");
    check(e <= 1e-5, std::format("device vs host {:.3e} <= 1e-5", e));
    return dnstest::finish();
}

// A 2D IC from a point function on cell centres, rounded as FP32 conserved.
template <class F>
dns::CompState field2D(const dns::Grid& g, F&& prim)
{
    auto h = dns::CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                double r, u, v, p;
                prim(i, j, (i + .5) * g.hx, (j + .5) * g.hy, r, u, v, p);
                const std::size_t c = g.idx(i, j, k);
                h.rho[c] = r;
                h.mx[c] = r * u;
                h.my[c] = r * v;
                h.E[c] = p / (shockref::kGamma - 1) + .5 * r * (u * u + v * v);
            }
    return h;
}

double maxAbsV(const dns::CompState& s)
{
    double m = 0;
    for (std::size_t c = 0; c < s.rho.size(); ++c) m = std::max(m, std::abs(s.my[c] / s.rho[c]));
    return m;
}

// Gate 19: Quirk Ms 6, 320 x 24; standing M 5, 80 x 40, on a face.
int aligned()
{
    {
        const shockref::RH a = shockref::rankine(6.0);
        const dns::Grid g = dns::Grid::make(320, 24, 1, 4.0, 1.0, 1.0);
        const auto ic = field2D(g, [&](int i, int j, double x, double, double& r, double& u, double& v, double& p) {
            const bool m = x < .25;
            r = (m ? a.r2 : 1.0) * (j == 12 ? 1 + 1e-3 * (i % 2 ? -1 : 1) : 1.0);
            u = m ? a.u2 : 0.0;
            v = 0;
            p = m ? a.p2 : 1.0;
        });
        auto r = makeRig(g, ic, 4, 0, openX());
        const Marched m = march(*r, 3.0 / (6 * std::sqrt(1.4)));
        const auto s = download(*r);
        const double pMid = .5 * (1 + a.p2);
        int lo = 1 << 30, hi = -1;
        for (int j = 0; j < g.ny; ++j) {
            int front = 0;
            for (int i = 0; i < g.nx - 4; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                if (shockref::pressureOf(s.rho[c], s.mx[c], s.E[c] - .5 * s.my[c] * s.my[c] / s.rho[c]) > pMid) front = i;
            }
            lo = std::min(lo, front);
            hi = std::max(hi, front);
        }
        const double vm = maxAbsV(s) / a.S;
        std::printf("  Quirk Ms 6 vmax/S %.3e spread %d max sensed %.3f%% %s\n", vm, hi - lo,
                    100 * m.peakCells / double(g.size()), m.why.c_str());
        check(m.ok, "Quirk marches");
        check(hi == lo, "Quirk: spread 0");
        check(vm <= 1e-4, std::format("Quirk vmax/S {:.3e} <= 1e-4", vm));
    }
    {
        const double M = 5, u1 = M * std::sqrt(1.4);
        const double r2 = 2.4 * M * M / (.4 * M * M + 2), p2 = 1 + 2 * 1.4 / 2.4 * (M * M - 1), u2 = u1 / r2;
        const dns::Grid g = dns::Grid::make(80, 40, 1, 2.0, 1.0, 1.0);
        const auto ic = field2D(g, [&](int i, int j, double x, double, double& r, double& u, double& v, double& p) {
            const bool m = x < 1;
            r = (m ? 1.0 : r2) * (j == 20 ? 1 + 1e-3 * (i % 2 ? -1 : 1) : 1.0);
            u = m ? u1 : u2;
            v = 0;
            p = m ? 1.0 : p2;
        });
        auto r = makeRig(g, ic, 4, 0, openX());
        const Marched m = march(*r, 4.0 / u1);
        const double vm = maxAbsV(download(*r)) / u1;
        std::printf("  standing M 5 vmax/u1 %.3e max sensed %.3f%% steps %ld %s\n", vm, 100 * m.peakCells / double(g.size()),
                    m.steps, m.why.c_str());
        check(m.ok, "standing marches");
        check(vm <= 1e-3, std::format("standing vmax/u1 {:.3e} <= 1e-3", vm));
    }
    return dnstest::finish();
}

struct VortexOut
{
    double l1 = 0, sensed = 0;
    dns::CompState s;
};

VortexOut vortex(int N, bool shock)
{
    const double L = 10, h = L / N, beta = 5, Minf = .84, gam = 1.4;
    const dns::Grid g = dns::Grid::make(N, N, 1, L, L, 1.0);
    const auto ic = field2D(g, [&](int, int, double xx, double yy, double& r, double& u, double& v, double& p) {
        const double X = xx - 5, Y = yy - 5, r2 = X * X + Y * Y;
        const double f = beta / (2 * std::numbers::pi) * std::exp(.5 * (1 - r2));
        const double T = 1 - (gam - 1) * beta * beta / (8 * gam * std::numbers::pi * std::numbers::pi) * std::exp(1 - r2);
        r = std::pow(T, 1 / (gam - 1));
        p = r * T;
        u = -Y * f + Minf * std::sqrt(gam);
        v = X * f;
    });
    (void)h;
    std::optional<G::ShockSetup> sh;
    if (shock) sh = G::ShockSetup{};
    auto r = makeRig(g, ic, 0, 0, sh);
    const Marched m = march(*r, L / (Minf * std::sqrt(gam)));
    VortexOut o;
    o.s = download(*r);
    o.sensed = m.peakCells;
    for (std::size_t c = 0; c < g.size(); ++c) o.l1 += std::abs(o.s.rho[c] - ic.rho[c]);
    o.l1 /= double(g.size());
    return o;
}

// Gate 20: the isentropic vortex.
int vortexGate()
{
    double e[3] = {};
    const int Ns[] = {80, 160, 320};
    for (int q = 0; q < 3; ++q) {
        const VortexOut on = vortex(Ns[q], true), off = vortex(Ns[q], false);
        e[q] = on.l1;
        std::printf("  N %d L1 rho %.4e (KG %.4e, ratio %.3f) max sensed %.0f cells %.3f%%\n", Ns[q], on.l1, off.l1,
                    on.l1 / off.l1, on.sensed, 100 * on.sensed / (double(Ns[q]) * Ns[q]));
        if (Ns[q] == 80) check(on.l1 <= 1.3 * off.l1, std::format("N 80: {:.3f} <= 1.3 x KG", on.l1 / off.l1));
        if (Ns[q] >= 160) {
            check(on.sensed == 0, std::format("N {}: 0 sensed", Ns[q]));
            check(on.s.rho == off.s.rho && on.s.E == off.s.E && on.s.mx == off.s.mx && on.s.my == off.s.my,
                  std::format("N {}: bytes equal to off", Ns[q]));
        }
    }
    const double order = std::log2(e[1] / e[2]);
    std::printf("  order 160-320 %.3f\n", order);
    check(order >= 1.9, std::format("order {:.3f} >= 1.9", order));
    return dnstest::finish();
}

// Gate 21: the contact rung of gate 6.
int contactGate()
{
    auto ic = [](double M, double width, int N) {
        std::vector<double> r(static_cast<std::size_t>(N)), u(r), p(r);
        for (int i = 0; i < N; ++i) {
            const double x = (i + .5) / N;
            const double d = width > 0 ? .5 * (1 - std::tanh((std::abs(x - .5) - .25) / (width / N)))
                                       : (std::abs(x - .5) < .25 ? 1.0 : 0.0);
            r[std::size_t(i)] = .1 + .9 * d;
            u[std::size_t(i)] = M * std::sqrt(1.4);
            p[std::size_t(i)] = 1;
        }
        return shockref::make<float>(r, u, p);
    };
    auto go = [&](double M, double w, bool shock, double& peak, dns::CompState& out) {
        const int N = 200;
        const dns::Grid g = grid1D(N, 1.0);
        std::optional<G::ShockSetup> sh;
        if (shock) sh = G::ShockSetup{};
        auto r = makeRig(g, rowsOf(g, ic(M, w, N)), 0, 0, sh);
        peak = 0;
        const double uu = M * std::sqrt(1.4);
        const Marched m = march(*r, .5 / std::max(uu, 1.0), .4, [&](double, long) {
            const auto s = download(*r);
            for (int i = 0; i < N; ++i) {
                const std::size_t c = g.idx(i, 0, 0);
                peak = std::max(peak, std::abs(double(shockref::pressureOf(float(s.rho[c]), float(s.mx[c]), float(s.E[c]))) - 1));
            }
        });
        out = download(*r);
        return m;
    };
    for (double w : {0.0, 2.0}) {
        double p1, p2;
        dns::CompState a, b;
        const Marched m = go(0.02, w, true, p1, a);
        go(0.02, w, false, p2, b);
        std::printf("  M 0.02 width %g peak %.3e sensed %.0f\n", w, p1, m.peakCells);
        check(m.ok && m.peakCells == 0, std::format("M 0.02 width {}: 0 sensed", w));
        check(a.rho == b.rho && a.E == b.E && a.mx == b.mx, std::format("M 0.02 width {}: equal to KG bitwise", w));
    }
    for (const auto& [M, band] : {std::pair{1.0, 0.26}, std::pair{3.0, 0.66}}) {
        double p1;
        dns::CompState a;
        const Marched m = go(M, 0, true, p1, a);
        std::printf("  M %g sharp peak %.3f max sensed %.2f%% %s\n", M, p1, 100 * m.peakCells / 800, m.why.c_str());
        check(m.ok, std::format("M {} sharp: no failure", M));
        check(p1 <= band, std::format("M {} sharp {:.3f} <= {:g}", M, p1, band));
    }
    return dnstest::finish();
}

// Gate 22: a periodic double shock, FP64 sums; LES, compensated.
int conservation()
{
    const int N = 400;
    const dns::Grid g = grid1D(N, 1.0);
    std::vector<double> rr(static_cast<std::size_t>(N), 1.0), uu(rr), pp(rr);
    for (int i = 0; i < N; ++i) uu[std::size_t(i)] = (i + .5) / N < .5 ? 4 : -4;
    const auto ic = rowsOf(g, shockref::make<float>(rr, uu, pp));
    // |sum/sum0 - 1| of mass and E, FP64 sums.
    auto driftOf = [&](const dns::CompState& s, const dns::CompState& s0, double d[2]) {
        const std::vector<double>* f[2] = {&s.rho, &s.E};
        const std::vector<double>* f0[2] = {&s0.rho, &s0.E};
        for (int q = 0; q < 2; ++q) {
            double a = 0, b = 0;
            for (std::size_t c = 0; c < g.size(); ++c) {
                a += (*f0[q])[c];
                b += (*f[q])[c];
            }
            d[q] = std::abs(b / a - 1);
        }
    };
    // A5's control: the same model, no --shock, a smooth wave (rho and u vary, so
    // the LES model is live), driven at the shock run's OWN dt sequence.
    std::vector<double> r2(rr), u2(rr), p2(rr);
    for (int i = 0; i < N; ++i) {
        const double w = 2 * std::numbers::pi * (i + .5) / N;
        r2[std::size_t(i)] = 1 + .3 * std::sin(w);
        u2[std::size_t(i)] = 4 + .5 * std::sin(w);
    }
    const auto icCtl = rowsOf(g, shockref::make<float>(r2, u2, p2));
    auto drift = [&](const char* tag, double cs, bool comp, bool relative) {
        auto r = makeRig(g, ic, 0, 0, G::ShockSetup{}, 0, cs, comp);
        std::vector<double> dts;
        double tLast = 0;
        const Marched m = march(*r, .1, .4, [&](double t, long) {
            dts.push_back(t - tLast);
            tLast = t;
        });
        double d[2], ctl[2];
        driftOf(download(*r), ic, d);
        auto c = makeRig(g, icCtl, 0, 0, std::nullopt, 0, cs, comp);
        for (const double dt : dts) c->dev->step(*c->s, Real(dt));
        driftOf(download(*c), icCtl, ctl);
        std::printf("  %-12s mass %.2e E %.2e | control mass %.2e E %.2e | ratio %.2f %.2f | %zu steps both, max sensed "
                    "%.3f%% %s\n",
                    tag, d[0], d[1], ctl[0], ctl[1], d[0] / ctl[0], d[1] / ctl[1], dts.size(),
                    100 * m.peakCells / double(g.size()), m.why.c_str());
        check(m.ok, std::format("{} marches", tag));
        if (relative) {
            check(std::isfinite(ctl[0]) && std::isfinite(ctl[1]), std::format("{}: the control marches", tag));
            check(d[0] <= 2 * ctl[0] && d[1] <= 2 * ctl[1], std::format("{}: drift <= 2 x its control", tag));
        } else {
            check(d[0] <= 1e-6 && d[1] <= 1e-6, std::format("{}: drift <= 1e-6", tag));
        }
    };
    {
        // The host twin, FP32, no FMA: the prototype's arithmetic on this IC.
        shockref::State<float> U = shockref::make<float>(rr, uu, pp);
        const shockref::State<float> U0 = U;
        shockref::Cfg<float> c;
        c.h = 1.0 / N;
        const shockref::Report rep = shockref::run(U, .1, c);
        double a[2] = {}, b[2] = {};
        for (int i = 0; i < N; ++i) {
            const std::size_t I = std::size_t(i);
            a[0] += U0.q[0][I];
            b[0] += U.q[0][I];
            a[1] += U0.q[2][I];
            b[1] += U.q[2][I];
        }
        std::printf("  host twin    mass %.2e E %.2e steps %ld\n", std::abs(b[0] / a[0] - 1), std::abs(b[1] / a[1] - 1),
                    rep.steps);
    }
    // A5: LES first, its control was the unknown.
    drift("LES", 0.17, false, true);
    drift("plain", 0, false, true);
    drift("compensated", 0, true, false);
    return dnstest::finish();
}

// Gate 24: a shock reflecting off an immersed plane, normal or at 30 deg.
struct GhostOut
{
    double plateau = 0, fcMax = 0, peakFrac = 0, fluidDrift = 0;
    int faces = 0, missed = 0;
    bool ok = true;
    std::string why;
};

GhostOut ghostPlane(double Ms, double deg)
{
    const shockref::RH a = shockref::rankine(Ms);
    const shockref::Reflect e = shockref::reflected(Ms);
    const double th = deg * std::numbers::pi / 180, nx = std::cos(th), ny = std::sin(th);
    const bool flat = deg == 0;
    const int Nx = flat ? 400 : 256, Ny = flat ? 8 : 256;
    const double h = 1.0 / Nx, off = .37 * h; // the sub-cell offset
    const double xw = (flat ? .875 : .72) + off, yw = .5 * Ny * h;
    const dns::Grid g = dns::Grid::make(Nx, Ny, 1, 1.0, Ny * h, 1.0);
    auto dist = [=](double x, double y) { return (x - xw) * nx + (y - yw) * ny; }; // < 0 in the fluid
    const double d0 = -.2;
    const auto ic = field2D(g, [&](int, int, double x, double y, double& r, double& u, double& v, double& p) {
        const bool m = dist(x, y) < d0;
        r = m ? a.r2 : 1.0;
        u = m ? a.u2 * nx : 0;
        v = m ? a.u2 * ny : 0;
        p = m ? a.p2 : 1.0;
    });
    auto gm = G::DeviceCompGhost::make(g, [=](double x, double y, double) { return -dist(x, y); }, shockref::kGamma, 1.0, 1.0);
    if (!gm) dnstest::skip("ghost init failed");
    G::DeviceCompGhost ghost = std::move(*gm);
    std::vector<std::uint8_t> solid(g.size(), 0);
    for (int c : ghost.solidCells()) solid[std::size_t(c)] = 1;
    auto dmask = G::DeviceBuffer<std::uint8_t>::make(g.size());
    if (!dmask || dmask->upload(solid.data()) != hipSuccess) dnstest::skip("mask upload failed");
    G::ShockSetup ss;
    ss.solid = dmask->data();
    ss.open[0] = true;
    ss.open[1] = !flat;
    auto r = makeRig(g, ic, flat ? 4 : 6, flat ? 0 : 6, ss);
    Rig* p = r.get();
    r->dev->setStageHook([p, &ghost](G::DeviceCompState& s) {
        p->hold(s);
        ghost.apply(s);
    });
    r->dev->setRhsHook([p, &ghost](const G::DeviceCompState& s, G::DeviceCompState& o, const G::StageCombine& cmb) {
        ghost.applyRhs(*p->dev, s, o, cmb);
    });
    // Each near-wall face (L, R fluid, a solid within 2 of it on the axis): within
    // +-4 cells of travel of the exact incident and reflected passage times, a step
    // with a jump > 10% needs a nonzero window byte. Ripple outside them is not a
    // crossing. tI, tR: those times, from the exact speeds.
    struct Near
    {
        std::size_t LL, L, R, RR;
        double tI = 0, tR = 0;
        bool crossed = false, missed = false;
        double top = 0, t = 0;              // the first unsensed step's jump, its time
        std::array<std::size_t, 7> cells{}; // at(-3 .. +3) along the face's axis
        std::array<double, 7> p7{};         // p and bytes there, at that step
        std::array<int, 7> b7{};
    };
    std::vector<Near> near;
    const double tWall = -d0 / a.S, T = tWall + .12 / e.w;
    const int dims[2] = {g.nx, g.ny};
    const int rim = flat ? 8 : 10;
    for (int j = flat ? 0 : rim; j < (flat ? g.ny : g.ny - rim); ++j)
        for (int i = rim; i < g.nx - rim; ++i)
            for (int ax = 0; ax < (flat ? 1 : 2); ++ax) {
                auto at = [&](int d) {
                    return ax == 0 ? g.idx(((i + d) % dims[0] + dims[0]) % dims[0], j, 0)
                                   : g.idx(i, ((j + d) % dims[1] + dims[1]) % dims[1], 0);
                };
                if (solid[at(0)] || solid[at(1)]) continue;
                if (!(solid[at(-1)] || solid[at(-2)] || solid[at(2)] || solid[at(3)])) continue;
                Near f{at(-1), at(0), at(1), at(2)};
                for (int d = -3; d <= 3; ++d) f.cells[std::size_t(d + 3)] = at(d);
                const double df = dist((i + .5 + (ax == 0 ? .5 : 0)) * g.hx, (j + .5 + (ax == 1 ? .5 : 0)) * g.hy);
                f.tI = (df - d0) / a.S;
                f.tR = tWall - df / e.w;
                near.push_back(f);
            }
    std::vector<std::uint8_t> by;
    GhostOut o;
    const double winI = 4 * h / a.S, winR = 4 * h / e.w;
    const Marched m = march(*r, T, .4, [&](double tNow, long) {
        ghost.apply(*r->s);
        const auto s = download(*r);
        if (r->dev->shockBytes(*r->s, by) != hipSuccess) dnstest::skip("byte download failed");
        auto press = [&](std::size_t c) {
            return .4 * (s.E[c] - .5 * (s.mx[c] * s.mx[c] + s.my[c] * s.my[c]) / s.rho[c]);
        };
        for (Near& f : near) {
            if (std::abs(tNow - f.tI) > winI && std::abs(tNow - f.tR) > winR) continue;
            const double pL = press(f.L), pR = press(f.R);
            const double jump = std::abs(pR - pL) / std::min(pL, pR);
            if (jump <= .1) continue;
            f.crossed = true;
            if (by[f.LL] || by[f.L] || by[f.R] || by[f.RR] || f.missed) continue;
            f.missed = true;
            f.top = jump;
            f.t = tNow;
            for (std::size_t q = 0; q < 7; ++q) {
                f.p7[q] = press(f.cells[q]);
                f.b7[q] = by[f.cells[q]];
            }
        }
    });
    o.ok = m.ok;
    o.why = m.why;
    o.peakFrac = m.peakCells / double(g.size());
    int shown = 0;
    for (const Near& f : near) {
        if (!f.crossed) continue;
        ++o.faces;
        if (!f.missed) continue;
        ++o.missed;
        if (shown++ < 6) {
            const std::size_t c = f.L;
            const double x = (double(c % g.nx) + .5) * g.hx, y = (double(c / g.nx) + .5) * g.hy;
            std::printf("    unsensed face L (%zu, %zu) R-L %zd jump %.3f at t %.4g of %.4g, d %.4f; window blocked "
                        "x,y:",
                        c % g.nx, c / g.nx, std::ptrdiff_t(f.R) - std::ptrdiff_t(f.L), f.top, f.t, T, dist(x, y));
            for (const std::size_t q : {f.LL, f.L, f.R, f.RR}) {
                const int i = int(q % g.nx), j = int(q / g.nx);
                const auto sol = [&](int a, int b) { return solid[g.idx(a, b, 0)] != 0; };
                const bool bx = sol(i - 2, j) || sol(i - 1, j) || sol(i + 1, j) || sol(i + 2, j);
                const bool by2 = sol(i, j - 2) || sol(i, j - 1) || sol(i, j + 1) || sol(i, j + 2);
                std::printf(" %s%d%d", sol(i, j) ? "S" : "", int(bx), int(by2));
            }
            std::printf("\n      at(-3..3) p:");
            for (const double v : f.p7) std::printf(" %.4g", v);
            std::printf("\n      solid:");
            for (const std::size_t q : f.cells) std::printf(" %d", int(solid[q]));
            std::printf("  bytes:");
            for (const int v : f.b7) std::printf(" %d", v);
            std::printf("\n      passage times: incident %.4g, reflected %.4g\n", f.tI, f.tR);
        }
    }
    ghost.apply(*r->s);
    const auto s = download(*r);
    double sum = 0, cnt = 0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const double x = (i + .5) * g.hx, y = (j + .5) * g.hy;
            const double d = dist(x, y), t = (y - yw) * nx - (x - xw) * ny;
            if (!(d > -.09 && d < -.03) || std::abs(t) > .2) continue;
            const std::size_t c = g.idx(i, j, 0);
            sum += .4 * (s.E[c] - .5 * (s.mx[c] * s.mx[c] + s.my[c] * s.my[c]) / s.rho[c]);
            cnt += 1;
        }
    o.plateau = sum / cnt / e.p5 - 1;
    auto fs = ghost.surfaceFaces(*r->dev, *r->s, 0.0);
    if (!fs) dnstest::skip("surface tally failed");
    for (const auto& f : *fs) o.fcMax = std::max({o.fcMax, std::abs(f.fc[0]), std::abs(f.fc[1]), std::abs(f.fc[2])});
    return o;
}

int ghostGate()
{
    for (double Ms : {1.5, 3.0, 6.0})
        for (double deg : {0.0, 30.0}) {
            const GhostOut o = ghostPlane(Ms, deg);
            std::printf("  Ms %g, %g deg: plateau %+.3e, |fc| max %.3g, near-wall faces crossed %d, unsensed %d, max "
                        "sensed %.2f%% %s\n",
                        Ms, deg, o.plateau, o.fcMax, o.faces, o.missed, 100 * o.peakFrac, o.why.c_str());
            check(o.ok, std::format("Ms {} {} deg: marches", Ms, deg));
            check(std::abs(o.plateau) <= .02, std::format("Ms {} {} deg: plateau p {:+.3e} <= 2%", Ms, deg, o.plateau));
            check(o.fcMax == 0, std::format("Ms {} {} deg: through-wall flux exactly 0", Ms, deg));
            check(o.faces > 0 && o.missed == 0, std::format("Ms {} {} deg: sensed on every near-wall face", Ms, deg));
        }
    return dnstest::finish();
}

// The open box of gates 25, 23: a p-ratio 3 slab, NSCBC x, sink, y periodic.
struct OpenRig
{
    dns::Grid g;
    std::unique_ptr<Rig> r;
    std::optional<G::DeviceCompNscbc> nsc;
    std::optional<G::DeviceCompSink> snk;
    std::optional<dns::CompState> stage1; // the first stage's output, hooks done
};

constexpr double kOpenP = 1.0 / 1.4, kOpenU = .2;

dns::CompState openIc(const dns::Grid& g, double ratio = 3)
{
    return field2D(g, [&](int, int, double x, double, double& r, double& u, double& v, double& p) {
        const bool slab = x > .45 && x < .55;
        r = 1;
        u = kOpenU;
        v = 0;
        p = kOpenP * (slab ? ratio : 1.0);
    });
}

std::unique_ptr<OpenRig> openRig(bool shock, bool force = false, const dns::CompState* ic = nullptr)
{
    auto o = std::make_unique<OpenRig>(OpenRig{dns::Grid::make(320, 8, 1, 1.0, 8.0 / 320, 1.0), {}, {}, {}, {}});
    G::ShockSetup ss = openX();
    ss.nscbc[0] = true;
    ss.testForce = force;
    o->r = makeRig(o->g, ic ? *ic : openIc(o->g), 0, 0, shock ? std::optional<G::ShockSetup>(ss) : std::nullopt);
    G::DeviceCompNscbc::Config nc;
    nc.rhoInf = 1;
    nc.uInf = kOpenU;
    nc.pInf = kOpenP;
    nc.gamma = shockref::kGamma;
    auto n = G::DeviceCompNscbc::make(o->g, nc);
    auto s = G::DeviceCompSink::make(o->g, {.nb = 16, .alpha = 1.0, .cInf = 1.0});
    if (!n || !s) dnstest::skip("NSCBC init failed");
    o->nsc.emplace(std::move(*n));
    o->snk.emplace(std::move(*s));
    o->nsc->setFreestream(kOpenU);
    OpenRig* p = o.get();
    o->r->dev->setRhsHook([p](const G::DeviceCompState& s2, G::DeviceCompState& q, const G::StageCombine& cmb) {
        p->nsc->apply(*p->r->dev, s2, q, cmb);
        p->snk->apply(s2, q, cmb);
        if (p->stage1) return;
        auto h = dns::CompState::zeros(p->g);
        if (q.download(h) != hipSuccess) dnstest::skip("download failed");
        p->stage1 = std::move(h);
    });
    return o;
}

// Gate 25 (A6): the mask's own statement, one stage from one state with strong
// jumps at the open rows, testForce against off: rows 0 and n-1 of the stage
// output bitwise equal; row 2 not (the teeth). Then a blast through x-hi.
int openGate()
{
    {
        const dns::Grid gj = dns::Grid::make(320, 8, 1, 1.0, 8.0 / 320, 1.0);
        const auto jumps = field2D(gj, [&](int i, int j, double, double, double& r, double& u, double& v, double& p) {
            const int e = std::min(i, gj.nx - 1 - i);
            r = e < 2 ? 2 : e < 5 ? .5 : 1;
            u = kOpenU * (e < 3 ? -1 : 1);
            v = j % 2 ? .1 : -.1;
            p = kOpenP * (e < 2 ? 4 : e < 5 ? .5 : 1);
        });
        auto a = openRig(true, true, &jumps), b = openRig(false, false, &jumps);
        const Real dt = Real(1e-4);
        a->r->dev->step(*a->r->s, dt);
        b->r->dev->step(*b->r->s, dt);
        const dns::CompState& sa = *a->stage1;
        const dns::CompState& sb = *b->stage1;
        auto rowEqual = [&](int i) {
            bool same = true;
            for (int j = 0; j < gj.ny; ++j) {
                const std::size_t c = gj.idx(i, j, 0);
                same = same && sa.rho[c] == sb.rho[c] && sa.mx[c] == sb.mx[c] && sa.my[c] == sb.my[c] &&
                       sa.mz[c] == sb.mz[c] && sa.E[c] == sb.E[c];
            }
            return same;
        };
        check(rowEqual(0), "testForce vs off, one stage: row 0 bitwise equal, all fields");
        check(rowEqual(gj.nx - 1), "testForce vs off, one stage: row n-1 bitwise equal, all fields");
        check(!rowEqual(2) && !rowEqual(gj.nx - 3), "teeth: rows 2 and n-3 differ");
    }
    auto on = openRig(true);
    const dns::Grid& g = on->g;
    double t = 0, maxMach = 0, peak = 0;
    const double tEnd = .8;
    long steps = 0;
    bool ok = true;
    while (t < tEnd) {
        const double dt = std::min(on->r->dev->stableDt(*on->r->s, *on->r->red, .4), tEnd - t);
        if (!(dt > 0 && std::isfinite(dt))) {
            ok = false;
            break;
        }
        const G::ShockRow row = on->r->dev->shockRow(*on->r->s, *on->r->red);
        ok = ok && row.rhoMin > 0 && row.pMin > 0;
        maxMach = std::max(maxMach, row.openMach);
        peak = std::max(peak, row.cells);
        on->r->dev->step(*on->r->s, Real(dt));
        t += dt;
        ++steps;
    }
    const auto end = download(*on->r);
    const bool finite = std::ranges::all_of(end.rho, [](double x) { return std::isfinite(x); }) &&
                        std::ranges::all_of(end.E, [](double x) { return std::isfinite(x); });
    std::printf("  blast exit: %ld steps, open-row Mach max %.3f, max sensed %.2f%%, ok %d, finite %d\n", steps, maxMach,
                100 * peak / double(g.size()), int(ok), int(finite));
    check(ok && finite, "finite after the exit, rho, p > 0");
    check(maxMach < 1, "the open rows stay subsonic");
    return dnstest::finish();
}

// Gate 23: uniform power-of-two Y through a double shock, NSCBC rows, the sink,
// the species sink and a solid block: rho Y == Y rho bitwise, every fluid cell,
// every step. The variants without the block or without --shock (a weaker slab,
// which KG survives) attribute a mismatch.
struct UniOut
{
    long steps = 0, bad = 0;
    double peak = 0, errMax = 0;
    bool ok = true;
    std::string first, worst;
    std::vector<double> dts;
};

// dts given: drive at that sequence (a control at the same steps), else cfl 0.4.
UniOut uniformRun(bool wall, bool shock, double pr, const std::vector<double>* dts = nullptr)
{
    const dns::Grid g = dns::Grid::make(320, 32, 1, 1.0, 0.1, 1.0);
    const double Y[2] = {.5, .125};
    const dns::Transport tr = dns::Transport::constant(shockref::kGamma, 0.0);
    auto dev = G::DeviceComp::make(g, tr);
    auto red = G::ReduceScratch::make();
    if (!dev || !red) dnstest::skip("device allocation failed");
    const auto sdf = [](double x, double y, double) {
        return std::max(std::abs(x - .72) - .04, std::abs(y - .05) - .02); // a block
    };
    auto gm = G::DeviceCompGhost::make(g, sdf, shockref::kGamma, 1.0, kOpenP);
    if (!gm) dnstest::skip("ghost init failed");
    G::DeviceCompGhost ghost = std::move(*gm);
    std::vector<std::uint8_t> solid(g.size(), 0);
    if (wall)
        for (int c : ghost.solidCells()) solid[std::size_t(c)] = 1;
    auto dmask = G::DeviceBuffer<std::uint8_t>::make(g.size());
    if (!dmask || dmask->upload(solid.data()) != hipSuccess) dnstest::skip("mask upload failed");
    G::SpeciesSetup sp;
    sp.nY = 2;
    sp.lnA = -std::numeric_limits<double>::infinity();
    sp.openX = true;
    sp.yIn[0] = Y[0];
    sp.yIn[1] = Y[1];
    sp.solid = wall ? dmask->data() : nullptr;
    if (!dev->enableSpecies(sp)) dnstest::skip("species allocation failed");
    G::ShockSetup ss = openX();
    ss.nscbc[0] = true;
    ss.solid = wall ? dmask->data() : nullptr;
    if (shock && !dev->enableShock(ss)) dnstest::skip("shock allocation failed");
    auto ds = G::DeviceCompState::make(g.size(), 0, false, 2);
    if (!ds) dnstest::skip("device allocation failed");
    G::DeviceCompNscbc::Config nc;
    nc.rhoInf = 1;
    nc.uInf = kOpenU;
    nc.pInf = kOpenP;
    nc.gamma = shockref::kGamma;
    auto nm = G::DeviceCompNscbc::make(g, nc);
    auto sm = G::DeviceCompSink::make(g, {.nb = 16, .alpha = 1.0, .cInf = 1.0});
    if (!nm || !sm) dnstest::skip("NSCBC init failed");
    G::DeviceCompNscbc nsc = std::move(*nm);
    G::DeviceCompSink snk = std::move(*sm);
    nsc.setFreestream(kOpenU);
    auto* d = &*dev;
    if (wall) dev->setStageHook([&](G::DeviceCompState& s) { ghost.apply(s); });
    dev->setRhsHook([&, d](const G::DeviceCompState& s, G::DeviceCompState& o, const G::StageCombine& cmb) {
        nsc.apply(*d, s, o, cmb);
        d->speciesOpenConsistency(s, o, cmb);
        snk.apply(s, o, cmb);
        snk.applySpecies(s, o, cmb);
        if (wall) ghost.applyRhs(*d, s, o, cmb);
    });
    const auto ic = openIc(g, pr);
    if (ds->upload(ic) != hipSuccess) dnstest::skip("upload failed");
    for (int k = 0; k < 2; ++k) {
        std::vector<double> ry(g.size());
        for (std::size_t c = 0; c < g.size(); ++c) ry[c] = solid[c] ? 0.0 : double(float(ic.rho[c]) * float(Y[k]));
        if (ds->uploadSpecies(k, ry) != hipSuccess) dnstest::skip("upload failed");
    }
    if (wall) ghost.apply(*ds);
    UniOut o;
    double t = 0;
    auto h = dns::CompState::zeros(g);
    std::vector<double> y0, y1;
    while ((dts ? o.steps < long(dts->size()) : t < .5) && o.ok) {
        const double dt = dts ? (*dts)[std::size_t(o.steps)] : dev->stableDt(*ds, *red, .4);
        if (!(dt > 0 && std::isfinite(dt))) {
            o.ok = false;
            break;
        }
        if (shock) o.peak = std::max(o.peak, dev->shockRow(*ds, *red).cells);
        dev->step(*ds, Real(dt));
        o.dts.push_back(dt);
        t += dt;
        ++o.steps;
        if (ds->download(h) != hipSuccess || ds->downloadSpecies(0, y0) != hipSuccess ||
            ds->downloadSpecies(1, y1) != hipSuccess)
            dnstest::skip("download failed");
        for (std::size_t c = 0; c < g.size(); ++c) {
            if (solid[c]) continue;
            const float r = float(h.rho[c]);
            const double err = std::max(std::abs(double(y0[c]) / r - Y[0]), std::abs(double(y1[c]) / r - Y[1]));
            if (err > o.errMax) {
                o.errMax = err;
                o.worst = std::format("worst at step {} i {} j {}", o.steps, c % g.nx, c / g.nx);
            }
            if (float(y0[c]) == r * float(Y[0]) && float(y1[c]) == r * float(Y[1])) continue;
            if (!o.bad++)
                o.first = std::format("first at step {} i {} j {}: rho {} rhoY0 {} vs {}", o.steps, c % g.nx, c / g.nx, r,
                                      float(y0[c]), r * float(Y[0]));
        }
    }
    return o;
}

int uniformY()
{
    // No solid: bitwise. With a solid: the no-shock control at the same dt sequence.
    const UniOut b = uniformRun(false, true, 3), a = uniformRun(true, true, 3);
    const UniOut c = uniformRun(true, false, 1.2, &a.dts), d = uniformRun(false, false, 1.2, &a.dts);
    auto show = [](const char* tag, const UniOut& o) {
        std::printf("  %-34s %ld steps, max sensed %.2f%%, mismatched cell-steps %ld, max |rhoY - Y rho|/rho %.3e, ok %d "
                    "%s; %s\n",
                    tag, o.steps, 100 * o.peak / (320.0 * 32), o.bad, o.errMax, int(o.ok), o.first.c_str(),
                    o.worst.c_str());
    };
    show("hll, no block, slab p ratio 3", b);
    show("hll, block", a);
    show("no --shock, block, p ratio 1.2", c);
    show("no --shock, no block", d);
    check(b.ok && a.ok, "marches through the rows, the sink and the block");
    check(a.peak > 0 && b.peak > 0, "the sensor fired");
    check(b.bad == 0, "no solid: rhoY == Y rho bitwise, every cell, every step");
    check(c.steps == a.steps && c.ok, "the control marches the same steps");
    check(a.errMax <= 2 * c.errMax, std::format("with a solid: {:.3e} <= 2 x control {:.3e}", a.errMax, c.errMax));
    return dnstest::finish();
}

// The burn integral of a planar flame on the direct API, past t 3 of t 5: S_c.
// force: testForce, sigma 1 on every allowed face (gate 13's teeth).
double directFlame(bool shock, bool force)
{
    auto c = dnstest::runCli({"flame", "--expansion", "10", "--cells-per-delta", "4", "--mach", "0.02", "--ny", "4"});
    if (!c) dnstest::skip("the flame case does not validate");
    const dns::Grid& g = c->grid;
    const double Tu = c->flameTu, sig = c->flameModel.sigma;
    const dns::Transport tr = dns::flameGas(*c, Tu);
    auto dev = G::DeviceComp::make(g, tr);
    auto red = G::ReduceScratch::make();
    if (!dev || !red) dnstest::skip("device allocation failed");
    G::SpeciesSetup sp;
    sp.nY = 1;
    sp.lnA = std::log(c->flameEig.Lambda);
    sp.Ta = c->flameModel.theta() * Tu;
    sp.q = tr.cp() * (sig - 1) * Tu;
    sp.invPr = 1.0 / c->prandtl;
    sp.openX = true;
    sp.yIn[0] = 1;
    if (!dev->enableSpecies(sp)) dnstest::skip("species allocation failed");
    G::ShockSetup ss = openX();
    ss.nscbc[0] = true;
    ss.testForce = force;
    if (shock && !dev->enableShock(ss)) dnstest::skip("shock allocation failed");
    auto ds = G::DeviceCompState::make(g.size(), 0, false, 1);
    if (!ds) dnstest::skip("device allocation failed");
    G::DeviceCompNscbc::Config nc;
    nc.rhoInf = 1;
    nc.uInf = c->flameUIn;
    nc.pInf = Tu - c->flameUIn * c->flameUIn * (sig - 1);
    nc.gamma = tr.gamma;
    nc.etaIn = c->nscbcEtaIn;
    nc.Rgas = tr.Rgas;
    nc.TInf = Tu;
    auto nm = G::DeviceCompNscbc::make(g, nc);
    auto sm = G::DeviceCompSink::make(g, {.nb = c->nscbcSink, .alpha = c->nscbcSinkAmp,
                                          .cInf = std::sqrt(tr.gamma * tr.Rgas * Tu)});
    if (!nm || !sm) dnstest::skip("NSCBC init failed");
    G::DeviceCompNscbc nsc = std::move(*nm);
    G::DeviceCompSink snk = std::move(*sm);
    nsc.setFreestream(c->flameUIn);
    auto* d = &*dev;
    dev->setRhsHook([&, d](const G::DeviceCompState& s, G::DeviceCompState& o, const G::StageCombine& cmb) {
        nsc.apply(*d, s, o, cmb);
        d->speciesOpenConsistency(s, o, cmb);
        snk.apply(s, o, cmb);
        snk.applySpecies(s, o, cmb);
    });
    auto host = dns::CompState::zeros(g);
    std::vector<double> rhoY;
    dns::flameIC(*c, host, rhoY);
    if (ds->upload(host) != hipSuccess || ds->uploadSpecies(0, rhoY) != hipSuccess) dnstest::skip("upload failed");
    double t = 0, sum = 0, cnt = 0;
    long steps = 0;
    double dtS = 0;
    std::vector<float> w(g.size());
    while (t < 5) {
        if (steps % 16 == 0) dtS = .95 * std::min(dev->stableDt(*ds, *red, .4), G::chemStableDt(*dev, *ds, *red));
        if (!(dtS > 0 && std::isfinite(dtS))) return NAN;
        dev->step(*ds, Real(dtS));
        t += dtS;
        ++steps;
        if (t < 3 || steps % 50) continue;
        dev->updateChem(*ds);
        if (dev->omega().download(w.data()) != hipSuccess) dnstest::skip("download failed");
        double b = 0;
        for (const float x : w) b += x;
        sum += b * g.hx / g.ny;
        cnt += 1;
    }
    return sum / cnt;
}

// Gate 13: planar sliding flames, S_L byte equal, 0 sensed; testForce moves S_L.
int flameGate()
{
    const fs::path dir = fs::path(DNS_TEST_DIR) / "flame";
    fs::remove_all(dir);
    fs::create_directories(dir);
    for (const char* cells : {"4", "6"}) {
        const std::string args = std::format("flame --expansion 10 --cells-per-delta {} --mach 0.02 --inflow-speed 0.98 --tt 20 "
                                             "--emit stats",
                                             cells);
        const std::string off = (dir / std::format("off{}", cells)).string(), on = (dir / std::format("on{}", cells)).string();
        check(run(args, off) == 0, std::format("{} cells: none runs", cells));
        check(run(args + " --shock hll", on) == 0, std::format("{} cells: hll runs", cells));
        check(statsEqualAfterCut(off + "_stats.csv", on + "_stats.csv"), std::format("{} cells: stats, S_L included, byte equal", cells));
        const double n = columnMax(on + "_stats.csv", "shockCells", 0.0);
        std::printf("  %s cells: max sensed cells %.0f\n", cells, n);
        check(n == 0, std::format("{} cells: shockCells 0", cells));
    }
    const double a = directFlame(false, false), b = directFlame(true, true), c = directFlame(true, false);
    std::printf("  direct API S_c past t 3: off %.5f, testForce %.5f (%+.1f%%), hll %.5f\n", a, b, 100 * (b / a - 1), c);
    check(!(std::abs(b / a - 1) <= .5), "teeth: testForce moves S_L > 50%");
    check(c == a, "direct API: hll equals off bitwise");
    return dnstest::finish();
}

// Gate 12: --shock hll vs none, default flags: bytes after the cut, 0 sensed.
int quiet()
{
    const fs::path dir = fs::path(DNS_TEST_DIR) / "quiet";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::pair<const char*, const char*> cases[] = {
        {"flame10", "flame --expansion 10 --cells-per-delta 4 --mach 0.02 --tt 2 --emit stats --emit bin"},
        {"flame5", "flame --expansion 5 --cells-per-delta 4 --mach 0.02 --tt 2 --emit stats --emit bin"},
        {"comp", "flow --physics compressible --body cylinder --n 192 --ny 128 --re 100 --mach 0.2 --ft 0.3 --emit stats"}};
    for (const auto& [tag, args] : cases) {
        gateBytes(tag, args, dir, "--shock hll");
        const double n = columnMax(dir / std::format("{}_sense_stats.csv", tag), "shockCells", -1.0);
        std::printf("  %s: max sensed cells %.0f\n", tag, n);
        check(n == 0, std::string(tag) + ": shockCells 0 on every row");
    }
    return dnstest::finish();
}

struct Column
{
    std::vector<double> t, v;
};

// Columns of a stats file, past tMin.
std::map<std::string, Column> readStats(const fs::path& p, double tMin)
{
    const auto lines = linesOf(slurp(p));
    std::size_t head = 0;
    while (head < lines.size() && lines[head].starts_with("#")) ++head;
    auto split = [](const std::string& l) {
        std::vector<std::string> c;
        std::stringstream s(l);
        for (std::string x; std::getline(s, x, ',');) c.push_back(x);
        return c;
    };
    const auto cols = split(lines.at(head));
    std::map<std::string, Column> o;
    for (std::size_t q = head + 1; q < lines.size(); ++q) {
        const auto r = split(lines[q]);
        const double t = std::stod(r.at(0));
        if (t < tMin) continue;
        for (std::size_t c = 0; c < cols.size(); ++c) {
            o[cols[c]].t.push_back(t);
            o[cols[c]].v.push_back(std::stod(r.at(c)));
        }
    }
    return o;
}

double timeMean(const Column& c)
{
    double s = 0, w = 0;
    for (std::size_t q = 1; q < c.t.size(); ++q) {
        const double dt = c.t[q] - c.t[q - 1];
        s += .5 * (c.v[q] + c.v[q - 1]) * dt;
        w += dt;
    }
    return s / w;
}

double timeRms(const Column& c)
{
    const double m = timeMean(c);
    double s = 0, w = 0;
    for (std::size_t q = 1; q < c.t.size(); ++q) {
        const double dt = c.t[q] - c.t[q - 1];
        const double a = c.v[q] - m, b = c.v[q - 1] - m;
        s += .5 * (a * a + b * b) * dt;
        w += dt;
    }
    return std::sqrt(s / w);
}

// The limit cycle's frequency (issues.md C1: 2.95 per flame time): the peak of the
// time-weighted Fourier sum of c with its linear trend removed, f in [1, 10],
// scanned finely (rows are not equally spaced). The slow drift of the window sits
// below 1: scanned from 0 it picks one period of the window, not the cycle.
double dominantFrequency(const Column& c)
{
    const double m = timeMean(c), T = c.t.back() - c.t.front(), t0 = .5 * (c.t.back() + c.t.front());
    double sxy = 0, sxx = 0;
    for (std::size_t r = 0; r < c.t.size(); ++r) {
        sxy += (c.t[r] - t0) * (c.v[r] - m);
        sxx += (c.t[r] - t0) * (c.t[r] - t0);
    }
    const double slope = sxy / sxx;
    double best = 0, fBest = 0;
    for (double f = 1; f <= 10; f += .05 / T) {
        const double w = 2 * std::numbers::pi * f;
        double a = 0, b = 0;
        for (std::size_t r = 1; r < c.t.size(); ++r) {
            const double dt = c.t[r] - c.t[r - 1], tm = .5 * (c.t[r] + c.t[r - 1]);
            const double v = .5 * (c.v[r] + c.v[r - 1]) - m - slope * (tm - t0);
            a += v * std::cos(w * tm) * dt;
            b += v * std::sin(w * tm) * dt;
        }
        const double e = a * a + b * b;
        if (e > best) {
            best = e;
            fBest = f;
        }
    }
    return fBest;
}

// Gate 12b: the lips, hll vs none on the default chamber, t > 20. Bounds fixed
// before any hll run (amendment A4).
int lip()
{
    const fs::path dir = fs::path(DNS_TEST_DIR) / "lip";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string args = "flame --mode chamber --tt 60 --emit stats";
    const std::string off = (dir / "none").string(), on = (dir / "hll").string();
    check(run(args, off) == 0, "none runs");
    const int rc = run(args + " --shock hll", on);
    check(rc == 0, std::format("hll runs (rc {})", rc));
    std::system(std::format("grep -E '^(shock:|dns:)' {}.log | head -8", on).c_str());
    const auto a = readStats(off + "_stats.csv", 20), b = readStats(on + "_stats.csv", 20);
    auto ratio = [](const Column& x, const Column& y) {
        Column r = x;
        for (std::size_t q = 0; q < r.v.size(); ++q) r.v[q] = x.v[q] / y.v[q];
        return r;
    };
    auto within = [&](const char* tag, double x, double y, double band) {
        const double e = std::abs(x / y - 1);
        std::printf("  %-12s none %.6g hll %.6g  off by %.3f%%\n", tag, y, x, 100 * e);
        check(e <= band, std::format("{} within {:g}%", tag, 100 * band));
    };
    within("Me", timeMean(b.at("Me")), timeMean(a.at("Me")), .01);
    within("pCh/pAmb", timeMean(ratio(b.at("pCh"), b.at("pAmb"))), timeMean(ratio(a.at("pCh"), a.at("pAmb"))), .01);
    within("burn/inj", timeMean(ratio(b.at("burn"), b.at("inj"))), timeMean(ratio(a.at("burn"), a.at("inj"))), .01);
    within("pCh rms", timeRms(b.at("pCh")), timeRms(a.at("pCh")), .10);
    within("frequency", dominantFrequency(b.at("pCh")), dominantFrequency(a.at("pCh")), .02);
    const auto past = readStats(on + "_stats.csv", 1);
    double cells = 0, sum = 0, off0 = 0, rest = 0;
    const std::size_t n = past.at("shockCells").v.size();
    for (std::size_t q = 0; q < n; ++q) {
        cells = std::max(cells, past.at("shockCells").v[q]);
        off0 = std::max(off0, past.at("shockCells").v[q] - past.at("shockLip").v[q]);
        rest = std::max(rest, past.at("shockCavRest").v[q]);
        sum += past.at("shockMax").v[q];
    }
    std::printf("  past t 1: max sensed cells %.0f (%.4f%%), off the lip column %.0f, cavity off Exit %.0f, lip max sigma mean "
                "%.3f; t > 20 sigma %.3f, pCh rms detrended (none, hll) %.3f %.3f\n",
                cells, 100 * cells / 55328, off0, rest, sum / double(n), timeMean(b.at("shockMax")), timeRms(a.at("pCh")),
                timeRms(b.at("pCh")));
    check(off0 == 0, "sensed cells past t 1 only in the lip column");
    check(cells <= 8, "at most 8 sensed cells");
    return dnstest::finish();
}

// March with stableDt every `recompute` steps, margin 0.95 (the runners').
template <class Mon>
Marched marchEvery(Rig& r, double tEnd, double cfl, int recompute, Mon&& mon)
{
    Marched m;
    double t = 0, dtS = 0;
    while (t < tEnd - 1e-14) {
        r.hold(*r.s);
        if (m.steps % recompute == 0) dtS = .95 * r.dev->stableDt(*r.s, *r.red, cfl);
        if (!(dtS > 0 && std::isfinite(dtS))) {
            m.ok = false;
            m.why = std::format("dt {} at t={:.4g} step {}", dtS, t, m.steps);
            return m;
        }
        const double dt = std::min(dtS, tEnd - t);
        const G::ShockRow row = r.dev->shockRow(*r.s, *r.red);
        m.peakCells = std::max(m.peakCells, row.cells);
        m.rhoMin = std::min(m.rhoMin, row.rhoMin);
        m.pMin = std::min(m.pMin, row.pMin);
        if (!(row.rhoMin > 0 && row.pMin > 0)) {
            m.ok = false;
            m.why = std::format("rho {:.3g}, p {:.3g} at t={:.4g} step {}", row.rhoMin, row.pMin, t, m.steps);
            return m;
        }
        r.dev->step(*r.s, Real(dt));
        t += dt;
        ++m.steps;
        mon(t, m.steps);
    }
    return m;
}

// A 3D box from rest: hi-p ball or diagonal Sod, periodic.
dns::CompState box3D(const dns::Grid& g, bool blast)
{
    auto h = dns::CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double x = (i + .5) * g.hx - .5, y = (j + .5) * g.hy - .5, z = (k + .5) * g.hz - .5;
                const std::size_t c = g.idx(i, j, k);
                const bool in = blast ? x * x + y * y + z * z < .01 : x + y + z < 0;
                h.rho[c] = blast ? 1.0 : in ? 1.0 : .125;
                h.E[c] = (blast ? (in ? 1000.0 : .01) : (in ? 1.0 : .1)) / .4;
            }
    return h;
}

// Gate 27: the hot cadence runs 3D Sod from rest; a dt held 16 steps diverges.
int cadence()
{
    const dns::Grid g = dns::Grid::make(48, 48, 48, 1, 1, 1);
    auto a = makeRig(g, box3D(g, false), 0, 0, G::ShockSetup{});
    const Marched h = marchEvery(*a, .1, .4, 1, [](double, long) {});
    auto b = makeRig(g, box3D(g, false), 0, 0, G::ShockSetup{});
    const Marched l = marchEvery(*b, .1, .4, 16, [](double, long) {});
    std::printf("  48^3 diagonal Sod, cfl 0.4: every step: %ld steps, ok %d, max sensed %.2f%%; every 16: %ld steps, ok %d %s\n",
                h.steps, int(h.ok), 100 * h.peakCells / double(g.size()), l.steps, int(l.ok), l.why.c_str());
    check(h.ok, "hot from step 0: runs, rho, p > 0");
    check(!l.ok, "teeth: --dt-recompute 16 diverges");
    bool hot = false;
    int calls = 0;
    const int e0 = dns::gpu::internal::shockDtEvery(true, 16, 1 << 30, hot, 0.0, [&] { ++calls; return 5.0; });
    const int e1 = dns::gpu::internal::shockDtEvery(true, 16, 3, hot, 1.0, [&] { ++calls; return 0.0; });
    bool quiet = false;
    const int e2 = dns::gpu::internal::shockDtEvery(true, 16, 3, quiet, 0.0, [&] { ++calls; return 0.0; });
    bool off = false;
    const int e3 = dns::gpu::internal::shockDtEvery(false, 16, 1 << 30, off, 0.0, [&] { ++calls; return 5.0; });
    check(e0 == 1 && e1 == 1 && hot, "hot is sticky");
    check(e2 == 16 && !quiet, "quiet stays at the route's cadence, unsensed off cadence");
    check(e3 == 16 && calls == 1, "an explicit --dt-recompute never senses");
    return dnstest::finish();
}

// Gate 28: split vs straight under --shock hll; --shock mismatch exit 3.
int restartGate()
{
    const fs::path dir = fs::path(DNS_TEST_DIR) / "restart";
    fs::remove_all(dir);
    fs::create_directories(dir / "a");
    fs::create_directories(dir / "b");
    fs::create_directories(dir / "c");
    const std::string pre = "flame --tt 2 --profile-dt 0.5 --emit stats --emit bin --fdt 0.25 --quiet";
    check(dnstest::restart::runDns(pre + " --shock hll --out " + (dir / "a/r").string(), (dir / "a/r.log").string()) == 0,
          "straight runs");
    const std::string pb = (dir / "b/r").string();
    bool done = false;
    int jobs = 0;
    for (; jobs < 8 && !done; ++jobs) {
        const std::string extra = jobs ? " --restart " + pb : "";
        const int rc = dnstest::restart::runDns(pre + " --shock hll --checkpoint-dt 0.5 --wall-limit 0 --out " + pb + extra,
                                                (dir / std::format("b/j{}.log", jobs)).string());
        if (rc != 0) std::printf("  job %d rc %d\n", jobs, rc);
        done = slurp(pb + ".json").find("\"complete\":true") != std::string::npos;
    }
    std::printf("  chained in %d jobs\n", jobs);
    check(done && jobs > 1, "the split run completed over several jobs");
    check(dnstest::restart::sameOutputs((dir / "a").string(), (dir / "b").string(), "hll split"), "split == straight, bytes");
    const std::string pc = (dir / "c/r").string();
    check(dnstest::restart::runDns(pre + " --shock hll --checkpoint-dt 0.5 --wall-limit 0 --out " + pc,
                                   (dir / "c/j.log").string()) == 0,
          "a checkpoint exists");
    const std::string ck = slurp(pc + "_ckpt.bin");
    check(ck.find("shockHot") != std::string::npos, "shockHot carried");
    const int rc = dnstest::restart::runDns(pre + " --shock sense --checkpoint-dt 0.5 --out " + pc + " --restart " + pc,
                                            (dir / "c/m.log").string());
    const std::string msg = slurp((dir / "c/m.log").string());
    std::printf("  mismatch rc %d: %s\n", rc, msg.substr(0, 160).c_str());
    check(rc == 3 && msg.find("--shock") != std::string::npos, "--shock mismatch exit 3 by name");
    const std::string po = (dir / "o").string();
    check(dnstest::restart::runDns(pre + " --checkpoint-dt 0.5 --wall-limit 0 --out " + po, (dir / "o.log").string()) == 0,
          "an off checkpoint exists");
    check(slurp(po + "_ckpt.bin").find("shockHot") == std::string::npos, "off checkpoint records unchanged");
    return dnstest::finish();
}

// Ghost rows of the quasi-1D nozzle: inlet (p0, T0) from the first interior p,
// outlet p_b with the last interior rho, u. shock 0: extrapolated; 2: no outlet.
__global__ void kNozzleBc(GpuGrid g, int ng, Real pb, int shock, Real* rho, Real* mx, Real* E)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (i >= ng && (i < g.nx - ng || shock == 2)) return;
    if (i < ng) {
        const std::size_t q = gidx(g, ng, j, k);
        const Real r = rho[q], m = mx[q], u = m / r, p = Real(.4) * (E[q] - Real(.5) * m * u);
        const Real pp = p < Real(.9999) ? p : Real(.9999);
        const Real Ti = powf(pp, Real(.4 / 1.4)), ri = pp / Ti;
        const Real w = Real(2 * 1.4 / .4) * (Real(1) - Ti), ui = sqrtf(w > Real(0) ? w : Real(0));
        rho[c] = ri;
        mx[c] = ri * ui;
        E[c] = pp / Real(.4) + Real(.5) * ri * ui * ui;
        return;
    }
    const std::size_t q = gidx(g, g.nx - ng - 1, j, k);
    rho[c] = rho[q];
    mx[c] = mx[q];
    E[c] = shock ? pb / Real(.4) + Real(.5) * mx[q] * (mx[q] / rho[q]) : E[q];
}

// The area source -(A'/A) {m, m u, u (E + p)} through the stage combination.
__global__ void kNozzleSrc(GpuGrid g, const Real* ar, G::StageCombine cmb, const Real* rho, const Real* mx, const Real* E,
                           Real* orho, Real* omx, Real* oE)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Real r = rho[c], m = mx[c], e = E[c], u = m / r, p = Real(.4) * (e - Real(.5) * m * u);
    const Real s = ar[i], cr = cmb.crAt(c);
    G::internal::stageAdd<false>(cmb, 0, c, cr, -s * m, orho);
    G::internal::stageAdd<false>(cmb, 1, c, cr, -s * (m * u), omx);
    G::internal::stageAdd<false>(cmb, 4, c, cr, -s * (u * (e + p)), oE);
}

struct DevNozzle
{
    double dx = NAN, jitter = NAN, l1p = NAN, mpk = 0, peak = 0;
    bool ok = true;
    std::string why;
};

// Gate 17: the host rig of gate 4 on the device.
DevNozzle devNozzle(int N, double tEnd, bool rest)
{
    const int ng = 3, nt = N + 2 * ng;
    const double h = 1.0 / N;
    const shockref::Nozzle nz;
    const double pb = nz.pBack();
    const dns::Grid g = dns::Grid::make(nt, 4, 1, nt * h, 4 * h, 1.0);
    std::vector<double> xg(static_cast<std::size_t>(nt));
    std::vector<Real> ar(static_cast<std::size_t>(nt));
    std::vector<double> rr(xg), uu(xg), pp(xg);
    for (int i = 0; i < nt; ++i) {
        const std::size_t I = std::size_t(i);
        xg[I] = (i - ng + .5) * h;
        ar[I] = Real(shockref::Nozzle::dArea(xg[I]) / shockref::Nozzle::area(xg[I]));
        const shockref::Prim q = nz.exact(xg[I], true);
        rr[I] = rest ? (xg[I] < shockref::Nozzle::xt ? 1.0 : pb) : q.r;
        uu[I] = rest ? 0.0 : q.u;
        pp[I] = rest ? rr[I] : q.p;
    }
    G::ShockSetup ss = openX();
    auto r = makeRig(g, rowsOf(g, shockref::make<float>(rr, uu, pp)), 0, 0, ss);
    auto da = G::DeviceBuffer<Real>::make(ar.size());
    if (!da || da->upload(ar.data()) != hipSuccess) dnstest::skip("upload failed");
    const GpuGrid gg = r->gg;
    Rig* p = r.get();
    auto bc = [gg, ng, pb](G::DeviceCompState& s) {
        kNozzleBc<<<G::launchBlocks(gg.size()), G::blockSize>>>(gg, ng, Real(pb), 1, s.rho.data(), s.mx.data(), s.E.data());
        G::checkLastKernel("kNozzleBc");
    };
    p->dev->setStageHook(bc);
    p->dev->setRhsHook([gg, &da](const G::DeviceCompState& s, G::DeviceCompState& o, const G::StageCombine& cmb) {
        kNozzleSrc<<<G::launchBlocks(gg.size()), G::blockSize>>>(gg, da->data(), cmb, s.rho.data(), s.mx.data(), s.E.data(),
                                                                 o.rho.data(), o.mx.data(), o.E.data());
        G::checkLastKernel("kNozzleSrc");
    });
    bc(*r->s);
    std::vector<double> xsh;
    const double tMon = (rest ? .8 : .6) * tEnd;
    DevNozzle o;
    auto mach = [&](const dns::CompState& s, int i) {
        const std::size_t c = g.idx(i, 0, 0);
        const double u = s.mx[c] / s.rho[c], pr = .4 * (s.E[c] - .5 * s.mx[c] * u);
        return u / std::sqrt(1.4 * pr / s.rho[c]);
    };
    const Marched m = march(*r, tEnd, .4, [&](double t, long) {
        if (!(t > tMon)) return;
        const auto s = download(*r);
        double Mj = 0, Mn = 0, xj = NAN;
        for (int i = ng; i + 1 < nt - ng; ++i) {
            const double a = mach(s, i), b = mach(s, i + 1);
            if (xg[std::size_t(i)] > shockref::Nozzle::xt && a > 1 && b <= 1) {
                xj = xg[std::size_t(i)];
                Mj = a;
                Mn = b;
            }
        }
        if (std::isfinite(xj)) xsh.push_back(xj + h * (Mj - 1) / (Mj - Mn));
    });
    bc(*r->s);
    o.ok = m.ok;
    o.why = m.why;
    o.peak = m.peakCells / double(g.size());
    double mean = 0, var = 0;
    for (const double x : xsh) mean += x;
    if (!xsh.empty()) {
        mean /= double(xsh.size());
        for (const double x : xsh) var += (x - mean) * (x - mean);
        o.dx = (mean - nz.xs) / h;
        o.jitter = std::sqrt(var / double(xsh.size())) / h;
    }
    const auto s = download(*r);
    double e = 0, sc = 0;
    for (int i = ng; i < nt - ng; ++i) {
        const double x = xg[std::size_t(i)];
        const std::size_t c = g.idx(i, 0, 0);
        o.mpk = std::max(o.mpk, mach(s, i));
        if (std::abs(x - nz.xs) <= .05) continue;
        const double u = s.mx[c] / s.rho[c], pr = .4 * (s.E[c] - .5 * s.mx[c] * u), px = nz.exact(x, true).p;
        e += std::abs(pr - px);
        sc += px;
    }
    o.l1p = e / sc;
    return o;
}

int nozzle()
{
    const std::pair<int, double> rows[] = {{200, 2.8e-3}, {400, 1.7e-3}};
    for (const auto& [N, band] : rows) {
        const DevNozzle f = devNozzle(N, 6.0, false), r = devNozzle(N, 8.0, true);
        const auto d = shockref::nozzleRun<double>(N, 6.0, true, false);
        std::printf("  N %d shock %+.2f jitter %.3f L1p %.3e Mpk %.3f max sensed %.2f%% %s\n", N, f.dx, f.jitter, f.l1p, f.mpk,
                    100 * f.peak, f.why.c_str());
        std::printf("  N %d rest shock %+.2f jitter %.3f max sensed %.2f%% %s; host FP64 shock %+.2f\n", N, r.dx, r.jitter,
                    100 * r.peak, r.why.c_str(), d.dx);
        check(f.ok && r.ok, std::format("N {} marches", N));
        check(std::abs(f.dx) <= 1, std::format("N {} |shock| {:.2f} <= 1 cell", N, std::abs(f.dx)));
        check(std::abs(r.dx) <= 1, std::format("N {} from rest |shock| {:.2f} <= 1 cell", N, std::abs(r.dx)));
        check(f.jitter <= .35, std::format("N {} jitter {:.3f} <= 0.35", N, f.jitter));
        check(f.l1p <= band, std::format("N {} L1 p {:.3e} <= {:.3g}", N, f.l1p, band));
        check(f.mpk >= 2.97, std::format("N {} Mpk {:.3f} >= 2.97", N, f.mpk));
        check(std::abs(f.dx - d.dx) <= .5, std::format("N {} FP32 device vs FP64 host {:.2f} <= 0.5 cell", N, std::abs(f.dx - d.dx)));
    }
    return dnstest::finish();
}

// Gate 18: oblique two-state, exact theta-beta-M, held rim, 100^2.
int oblique()
{
    const struct
    {
        double M, th, pBand, ovBand;
    } rows[] = {{3, 15, 1.2e-3, .05}, {5, 25, 1.0e-2, .05}};
    for (const auto& a : rows) {
        const double th = a.th * std::numbers::pi / 180;
        auto f = [&](double b) {
            return std::tan(th) - 2 / std::tan(b) * (a.M * a.M * std::sin(b) * std::sin(b) - 1) / (a.M * a.M * (1.4 + std::cos(2 * b)) + 2);
        };
        double lo = std::asin(1 / a.M) + 1e-3, hi = 1.1;
        for (int q = 0; q < 200; ++q) {
            const double m = .5 * (lo + hi);
            if (f(lo) * f(m) <= 0) hi = m;
            else lo = m;
        }
        const double beta = .5 * (lo + hi);
        const double Mn = a.M * std::sin(beta), r2 = 2.4 * Mn * Mn / (.4 * Mn * Mn + 2), p2 = 1 + 2 * 1.4 / 2.4 * (Mn * Mn - 1);
        const double Mn2 = std::sqrt((1 + .2 * Mn * Mn) / (1.4 * Mn * Mn - .2)), M2 = Mn2 / std::sin(beta - th);
        const double c2 = std::sqrt(1.4 * p2 / r2), u1 = a.M * std::sqrt(1.4);
        const int N = 100;
        const dns::Grid g = dns::Grid::make(N, N, 1, 1.0, 1.0, 1.0);
        auto dist = [&](double x, double y) { return (x - .5) * std::sin(beta) - (y - .5) * std::cos(beta); };
        const auto ic = field2D(g, [&](int, int, double x, double y, double& r, double& u, double& v, double& p) {
            const bool m = dist(x, y) < 0;
            r = m ? 1.0 : r2;
            p = m ? 1.0 : p2;
            u = m ? u1 : M2 * c2 * std::cos(th);
            v = m ? 0.0 : M2 * c2 * std::sin(th);
        });
        G::ShockSetup ss = openX();
        ss.open[1] = true;
        auto r = makeRig(g, ic, 6, 6, ss);
        const Marched m = march(*r, 1.5 / u1);
        const auto s = download(*r);
        double sum = 0, cnt = 0, top = 0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const double x = (i + .5) / N, y = (j + .5) / N;
                if (std::abs(x - .5) >= .33 || std::abs(y - .5) >= .33) continue;
                const std::size_t c = g.idx(i, j, 0);
                const double pr = .4 * (s.E[c] - .5 * (s.mx[c] * s.mx[c] + s.my[c] * s.my[c]) / s.rho[c]);
                top = std::max(top, pr);
                if (dist(x, y) > .06) {
                    sum += pr;
                    cnt += 1;
                }
            }
        const double err = sum / cnt / p2 - 1, ov = top / p2 - 1;
        std::printf("  M %g theta %g: p2 %+.3e overshoot %.3f max sensed %.2f%% %s\n", a.M, a.th, err, ov,
                    100 * m.peakCells / double(g.size()), m.why.c_str());
        check(m.ok, "marches");
        check(std::abs(err) <= a.pBand, std::format("p2 {:.3e} <= {:g}", err, a.pBand));
        check(ov <= a.ovBand, std::format("overshoot {:.3f} <= {:g}", ov, a.ovBand));
    }
    return dnstest::finish();
}

// Gate 26 (3D): blast and diagonal Sod from rest, 64^3, cfl 0.4, the hot cadence;
// the 0.5 outcome is the bracket.
int threeD()
{
    const dns::Grid g = dns::Grid::make(64, 64, 64, 1, 1, 1);
    for (const auto& [name, blast, tEnd] : {std::tuple{"blast", true, .012}, std::tuple{"diagonal Sod", false, .1}}) {
        for (const double cfl : {.4, .5}) {
            auto r = makeRig(g, box3D(g, blast), 0, 0, G::ShockSetup{});
            const Marched m = marchEvery(*r, tEnd, cfl, 1, [](double, long) {});
            std::printf("  %s cfl %g: %ld steps, ok %d, rhoMin %.3g pMin %.3g, max sensed %.2f%% %s\n", name, cfl, m.steps,
                        int(m.ok), m.rhoMin, m.pMin, 100 * m.peakCells / double(g.size()), m.why.c_str());
            if (cfl == .4) check(m.ok, std::format("{}: rho, p > 0 at cfl 0.4", name));
        }
    }
    return dnstest::finish();
}

// Gate 29: the choked chamber under --shock hll.
// What gates 29, 29b read off one chamber run's stats. budget: domain mass change
// against the integrals of inflow, injection and outflow, of the injected mass (A14).
// choked: the longest span with p0Noz/pAmb > 1.893 and MexitMax >= 0.98.
struct ChamberRead
{
    int rc = -1;
    double tEnd = 0, cells = 0, burning = 0, far = 0, rest = 0, budget = NAN, injected = 0, chokedFor = 0, chokedFrom = NAN;
    double MeMax = 0, MexMax = 0, MthMax = 0, p0Max = 0, MjetMax = 0;
};

ChamberRead chamberRead(const std::string& p, int rc, double fluidCells)
{
    ChamberRead o;
    o.rc = rc;
    const auto s = readStats(p + "_stats.csv", -1);
    const Column& t = s.at("t");
    const std::size_t n = t.t.size();
    auto top = [&](const char* name) {
        double m = 0;
        for (const double x : s.at(name).v)
            if (std::isfinite(x)) m = std::max(m, x);
        return m;
    };
    o.tEnd = t.t.back();
    o.cells = top("shockCells");
    o.burning = top("shockBurnCells");
    o.far = top("shockCavFar");
    o.rest = top("shockCavRest");
    o.MeMax = top("Me");
    o.MexMax = top("MexitMax");
    o.MthMax = top("MThroat");
    const auto& lo = s.at("mdotLo").v;
    const auto& hi = s.at("mdotHi").v;
    const auto& pn = s.at("p0Noz").v;
    const auto& pa = s.at("pAmb").v;
    const auto& mx = s.at("MexitMax").v;
    double through = 0, since = NAN;
    for (std::size_t q = 0; q < n; ++q) {
        o.p0Max = std::max(o.p0Max, pn[q] / pa[q]);
        if (q) through += .5 * (t.t[q] - t.t[q - 1]) * (lo[q] + lo[q - 1] - hi[q] - hi[q - 1]);
        const bool on = pn[q] / pa[q] > 1.893 && mx[q] >= .98;
        if (!on) {
            since = NAN;
            continue;
        }
        if (std::isnan(since)) since = t.t[q];
        if (t.t[q] - since > o.chokedFor) {
            o.chokedFor = t.t[q] - since;
            o.chokedFrom = since;
        }
    }
    o.injected = s.at("injCum").v.back();
    const double dM = s.at("mass").v.back() - s.at("mass").v.front();
    o.budget = (dM - through - o.injected) / o.injected;
    std::printf("  t Me MexitMax MThroat MJet p0Noz/pa p0Thr/pa pThr/pa pCh/pa cells far burn openMach\n");
    for (std::size_t q = 0; q < n; q += std::max<std::size_t>(1, n / 14)) {
        const double a = pa[q];
        std::printf("  %6.2f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %4.0f %3.0f %3.0f %.3f\n", t.t[q], s.at("Me").v[q], mx[q], s.at("MThroat").v[q],
                    s.at("MJet").v[q], pn[q] / a, s.at("p0Throat").v[q] / a, s.at("pThroat").v[q] / a, s.at("pCh").v[q] / a,
                    s.at("shockCells").v[q], s.at("shockCavFar").v[q], s.at("shockBurnCells").v[q], s.at("openMach").v[q]);
    }
    o.MjetMax = top("MJet");
    std::printf("  MJet max %.3f\n", o.MjetMax);
    std::printf("  rc %d t end %.4g, max sensed %.0f (%.3f%% of %.0f), cavity off Exit %.0f, past its 3 columns %.0f, burning %.0f\n", rc, o.tEnd,
                o.cells, 100 * o.cells / fluidCells, fluidCells, o.rest, o.far, o.burning);
    std::printf("  Me max %.3f MexitMax %.3f MThroat %.3f p0Noz/pAmb max %.3f; choked for %.2f from t %.2f; mass budget %+.3e of %.4g injected\n",
                o.MeMax, o.MexMax, o.MthMax, o.p0Max, o.chokedFor, o.chokedFrom, o.budget, o.injected);
    return o;
}

ChamberRead chamberRun(const std::string& name, double inject, double tt, const std::string& shock = "hll")
{
    const fs::path dir = fs::path(DNS_TEST_DIR) / "chamber";
    fs::create_directories(dir);
    const std::string p = (dir / name).string();
    fs::remove(p + "_stats.csv");
    const int rc = run(std::format("flame --mode chamber --expansion 10 --inject-speed {} --downstream 80 --inject-ramp 4 --coflow 6 "
                                   "--shock {} --tt {} --emit stats",
                                   inject, shock, tt),
                       p);
    std::system(std::format("grep -E '^(shock:|dns:)' {}.log | head -8", p).c_str());
    return chamberRead(p, rc, 524.0 * 152);
}

// Gate 29. Its burning and window clauses are UNMET, a real limit of D on stairs
// (A14, S17); the counts are reported on every chamber run and failed here as they
// stand.
int chamber()
{
    const ChamberRead o = chamberRun("c", 8, 20);
    const std::string log = slurp((fs::path(DNS_TEST_DIR) / "chamber" / "c").string() + ".log");
    const bool guard = o.rc == 7 && log.find("the NSCBC boundary is subsonic only (docs/shock.md)") != std::string::npos;
    check(o.rc == 0 || guard, "exit 0, or exit 7 with the guard's exact text");
    check(o.burning == 0, "no cell sensed in burning cells");
    check(o.far == 0, "cavity cells sensed only within 3 columns of Exit");
    check(std::abs(o.budget) <= 1e-3, std::format("mass budget {:.3e} <= 0.1% of the injected mass", o.budget));
    return dnstest::finish();
}

// Gate 29b: inject 12, 16, then 20 (A14; 24 is past --inject-speed's cap of 20), the
// first to choke over >= 2 flame times ends the ladder. tt 34: inject 16's longest
// span opens at t 26.9, and the blow-off guard ends that run at t 54; at tt 20 no
// inject holds 2 (inject 20: 1.93).
int chokedGate()
{
    for (const double inject : {12.0, 16.0, 20.0}) {
        std::printf("  inject %.0f\n", inject);
        const ChamberRead o = chamberRun(std::format("i{}", int(inject)), inject, 34);
        check(std::abs(o.budget) <= 1e-3, std::format("inject {}: mass budget {:.3e} <= 0.1%", inject, o.budget));
        if (o.chokedFor < 2) continue;
        check(o.rc == 0, std::format("inject {} exit 0", inject));
        check(o.burning == 0, std::format("inject {}: no cell sensed in burning cells", inject));
        return dnstest::finish();
    }
    check(false, "a throat chokes for >= 2 flame times");
    return dnstest::finish();
}

// Gate 30 (A9): the guard's number and text on one state. Supersonic outflow on
// x-hi passes; inward on x-hi, either way on x-lo, and a lateral row: exit 7, the
// exact string.
int guardGate()
{
    const dns::Grid g = grid1D(64, 1.0);
    enum Row { None, HiOut, HiIn, LoOut, LoIn, Lateral };
    auto guard = [&](Row which) {
        G::ShockSetup ss = openX();
        ss.nscbc[0] = true;
        ss.open[1] = ss.nscbc[1] = which == Lateral;
        const auto ic = field2D(g, [&](int i, int j, double, double, double& r, double& u, double& v, double& p) {
            r = 1;
            u = .5;
            v = 0;
            p = 1.0 / 1.4;
            if (which == HiOut && i > g.nx / 2) u = 2;
            if (which == HiIn && i == g.nx - 1) u = -2;
            if (which == LoOut && i == 0) u = -2;
            if (which == LoIn && i == 0) u = 2;
            if (which == Lateral && j == g.ny - 1) v = 2;
        });
        auto rig = makeRig(g, ic, 0, 0, ss);
        return rig->dev->shockRowGuard(*rig->s, *rig->red);
    };
    const double sub = guard(None), out = guard(HiOut);
    std::printf("  subsonic %.3f, x-hi flowing out at M 2 %.3f\n", sub, out);
    check(std::abs(sub - .5) <= 1e-6, "subsonic rows read their |u_n|/c");
    check(std::abs(out - .5) <= 1e-6, "supersonic outflow on x-hi is not the guard's");
    check(G::internal::shockOpenRow(out, 1.5).has_value(), "supersonic outflow on x-hi passes");
    const std::pair<Row, const char*> stops[] = {{HiIn, "inward on x-hi"}, {LoOut, "out of x-lo"}, {LoIn, "into x-lo"}, {Lateral, "a lateral row"}};
    for (const auto& [which, name] : stops) {
        const double m = guard(which);
        const auto e = G::internal::shockOpenRow(m, 1.5);
        std::printf("  %s: %.3f: %s\n", name, m, e ? "no error" : e.error().message.c_str());
        check(!e && e.error().code == 7, std::format("{}: exit 7", name));
        const std::string want = std::format("supersonic flow on the open row at t={:.4g} (|u_n|/c = {:.3g}): the NSCBC "
                                             "boundary is subsonic only (docs/shock.md)",
                                             1.5, m);
        check(!e && e.error().message == want, std::format("{}: the exact string", name));
    }
    check(G::internal::shockOpenRow(.99, 1.5).has_value(), "a subsonic row passes");
    return dnstest::finish();
}

// Step 9b's rig: the x-hi NSCBC row, Super or shipped, the sink on request, under
// --shock hll; rows i < 4 held at the IC, as the plan's "inlet held by the test".
struct SuperRig
{
    dns::Grid g;
    std::unique_ptr<Rig> r;
    std::optional<G::DeviceCompNscbc> nsc;
    std::optional<G::DeviceCompSink> snk;
};

std::unique_ptr<SuperRig> superRig(const dns::Grid& g, const dns::CompState& ic, bool super, bool sink, double pInf, double mu = 0,
                                   double cInf = 1)
{
    auto o = std::make_unique<SuperRig>(SuperRig{g, {}, {}, {}});
    G::ShockSetup ss = openX();
    ss.nscbc[0] = true;
    o->r = makeRig(g, ic, 0, 0, ss, mu);
    o->r->rx = 4;
    o->r->rxHi = 0;
    G::DeviceCompNscbc::Config nc;
    nc.rhoInf = 1;
    nc.uInf = 0;
    nc.pInf = pInf;
    nc.gamma = shockref::kGamma;
    nc.superOutflow = super;
    auto n = G::DeviceCompNscbc::make(g, nc);
    if (!n) dnstest::skip("NSCBC init failed");
    o->nsc.emplace(std::move(*n));
    o->nsc->setFreestream(0);
    if (sink) {
        auto s = G::DeviceCompSink::make(g, {.nb = 16, .alpha = 1.0, .cInf = cInf});
        if (!s) dnstest::skip("sink init failed");
        o->snk.emplace(std::move(*s));
    }
    SuperRig* p = o.get();
    o->r->dev->setRhsHook([p](const G::DeviceCompState& s2, G::DeviceCompState& q, const G::StageCombine& cmb) {
        p->nsc->apply(*p->r->dev, s2, q, cmb);
        if (p->snk) p->snk->apply(s2, q, cmb);
    });
    return o;
}

struct PassOut
{
    double before = 0, after = 0;
    bool ok = false;
    std::string why;
};

// Gate 33: one pulse, entropy (kind 0) or a right-going acoustic wave (kind 1, c = 1,
// rho = 1), at M. before: largest p, rho error against the exact advected pulse
// mid-box. after: the peak of max p, rho error against the base state from the
// pulse's exit until the slowest wave has crossed the box twice.
PassOut passRun(double M, int kind, bool super, double pFac = 1)
{
    const dns::Grid g = dns::Grid::make(320, 8, 1, 1.0, 8.0 / 320, 1.0);
    const double p0 = 1.0 / 1.4, x0 = .3, w = .03, eps = kind == 0 ? .1 : .02;
    const double speed = kind == 0 ? M : M + 1;
    auto prim = [&](double x, double t, double& r, double& u, double& p) {
        const double s = x - x0 - speed * t, b = eps * std::exp(-s * s / (w * w));
        r = 1;
        u = M;
        p = p0;
        if (kind == 0) {
            r += b;
            return;
        }
        const double dp = b * p0;
        p += dp;
        r += dp;
        u += dp;
    };
    const auto ic = field2D(g, [&](int, int, double x, double, double& r, double& u, double& v, double& p) {
        prim(x, 0, r, u, p);
        v = 0;
    });
    auto o = superRig(g, ic, super, false, p0 * pFac);
    const double tBefore = (.6 - x0) / speed, tExit = (1 + 4 * w - x0) / speed, tEnd = tExit + 2 / (M - 1);
    PassOut out;
    bool beforeDone = false;
    const Marched m = march(*o->r, tEnd, .4, [&](double t, long step) {
        const bool first = !beforeDone && t >= tBefore;
        if (!first && !(t >= tExit && step % 8 == 0)) return;
        beforeDone = true;
        const auto s = download(*o->r);
        double e = 0;
        for (int i = 4; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, 0, 0);
            double re = 1, ue = M, pe = p0;
            if (first) prim((i + .5) * g.hx, t, re, ue, pe);
            const double u = s.mx[c] / s.rho[c], pr = .4 * (s.E[c] - .5 * s.mx[c] * u);
            e = std::max({e, std::abs(pr - pe) / p0, std::abs(s.rho[c] - re)});
        }
        if (first) out.before = e;
        else out.after = std::max(out.after, e);
    });
    out.ok = m.ok;
    out.why = m.why;
    return out;
}

int outflowPass()
{
    const char* names[] = {"entropy", "acoustic"};
    for (const double M : {2.0, 3.0})
        for (const int kind : {0, 1}) {
            const PassOut a = passRun(M, kind, true);
            std::printf("  M %.0f %s: before %.3e; after exit %.3e %s\n", M, names[kind], a.before, a.after, a.why.c_str());
            check(a.ok, std::format("M {} {} marches", M, names[kind]));
            check(a.after <= 1e-3, std::format("M {} {} after exit {:.3e} <= 1e-3", M, names[kind], a.after));
            // The teeth: a supersonic row cannot feel the back pressure.
            for (const double fac : {.8, 1.25}) {
                const PassOut f = passRun(M, kind, true, fac), b = passRun(M, kind, false, fac);
                std::printf("    pInf/p %.2f: Super %.3e, shipped row %.3e (%.0fx) %s\n", fac, f.after, b.after, b.after / f.after,
                            b.why.c_str());
                check(f.ok, std::format("M {} {} pInf/p {} marches", M, names[kind], fac));
                check(f.after <= 2 * a.after, std::format("M {} {} pInf/p {}: Super {:.3e} within 2x of pInf = p0", M, names[kind], fac, f.after));
                check(!b.ok || b.after >= 10 * f.after, std::format("M {} {} pInf/p {}: shipped row >= 10x worse", M, names[kind], fac));
            }
        }
    return dnstest::finish();
}

struct ExitErr
{
    double M = NAN, p = NAN;
    bool ok = false;
    std::string why;
};

struct ContactOut
{
    double rowT = HUGE_VAL, rowP = HUGE_VAL, boxT = HUGE_VAL, boxP = HUGE_VAL, tCold = NAN, pCold = NAN, mnCold = NAN;
    double massDrift = 0, energyDrift = 0;
    double tSensed = NAN, topPeak = 0, topCold = NAN, tBoxCold = NAN;
    int iBoxCold = -1;
    bool ok = false;
    std::string why;
};

constexpr double kChamberH = 0.392589, kChamberP = 1785.71; // the failing long box

// Gate 36(a): a slab T/Tu = ratio, uniform p, sliding at V along y on the chamber's
// mesh spacing, gas and sink, on the SHIPPED row. gap: cells between the slab and the
// x-hi row (0: its contacts slide along the row). rowT, rowP: lowest on the row;
// boxT, boxP: anywhere.
ContactOut contactRun(double V, double ratio, int gap, double transits)
{
    const int nx = 96, ny = 152;
    const dns::Grid g = dns::Grid::make(nx, ny, 1, nx * kChamberH, ny * kChamberH, 1.0);
    const auto ic = field2D(g, [&](int i, int j, double, double, double& r, double& u, double& v, double& p) {
        const bool hot = j >= 38 && j < 114 && i >= nx - 60 - gap && i < nx - gap;
        r = hot ? 1 / ratio : 1;
        u = 0;
        v = V;
        p = kChamberP;
    });
    auto o = superRig(g, ic, false, true, kChamberP, .72, std::sqrt(1.4 * kChamberP));
    ContactOut out;
    auto sums = [&](const dns::CompState& s, double& m, double& e) {
        m = e = 0;
        for (std::size_t c = 0; c < g.size(); ++c) {
            m += s.rho[c];
            e += s.E[c];
        }
    };
    double m0, e0, m1, e1;
    sums(ic, m0, e0);
    const double tEnd = transits * ny * kChamberH / std::max(V, 1e-9);
    const Marched m = march(*o->r, tEnd, .4, [&](double t, long step) {
        if (step % 16) return;
        const auto s = download(*o->r);
        auto press = [&](int j) {
            const std::size_t c = g.idx(nx - 1, (j + ny) % ny, 0);
            return .4 * (s.E[c] - .5 * (s.mx[c] * s.mx[c] + s.my[c] * s.my[c]) / s.rho[c]);
        };
        double top = 0;
        for (int j = 0; j < ny; ++j) {
            const double a = press(j + 2), b = press(j), d = press(j - 2);
            top = std::max(top, std::abs(a - 2 * b + d) / (a + 2 * b + d));
        }
        if (std::isnan(out.tSensed) && top >= dns::kShockTh1) out.tSensed = t;
        if (!std::isnan(out.tCold) && std::isnan(out.topCold)) out.topCold = out.topPeak;
        out.topPeak = std::max(out.topPeak, top);
        for (int j = 0; j < ny; ++j)
            for (int i = 4; i < nx; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const double u = s.mx[c] / s.rho[c], w = s.my[c] / s.rho[c];
                const double pr = .4 * (s.E[c] - .5 * s.rho[c] * (u * u + w * w)), T = pr / s.rho[c] / kChamberP;
                if (std::isnan(out.tBoxCold) && T < .5) {
                    out.tBoxCold = t;
                    out.iBoxCold = i;
                }
                out.boxT = std::min(out.boxT, T);
                out.boxP = std::min(out.boxP, pr / kChamberP);
                if (i != nx - 1) continue;
                out.rowT = std::min(out.rowT, T);
                out.rowP = std::min(out.rowP, pr / kChamberP);
                if (std::isnan(out.tCold) && T < .5) {
                    out.tCold = t;
                    out.pCold = pr / kChamberP;
                    out.mnCold = u / std::sqrt(1.4 * pr / s.rho[c]);
                }
            }
    }, 16);
    out.ok = m.ok;
    out.why = m.why;
    const auto s = download(*o->r);
    sums(s, m1, e1);
    out.massDrift = std::abs(m1 / m0 - 1);
    out.energyDrift = std::abs(e1 / e0 - 1);
    return out;
}

// Rows i < rx of y := 0.5 rho, the held rows' uniform species.
__global__ void kHoldY(GpuGrid g, int rx, const Real* rho, Real* y)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (i < rx) y[c] = Real(.5) * rho[c];
}

struct BackflowOut
{
    double rowT = HUGE_VAL, rowP = HUGE_VAL, rowRho = HUGE_VAL, rowUn = 0, minRho = HUGE_VAL, minP = HUGE_VAL;
    double in = 0; // the mass taken in through x-hi, of the box's
    double yErr = 0; // max |Y - 1/2| on the row
    long bad = 0, checked = 0, backflow = 0;
    bool ok = true;
    std::string why, first;
};

// Gate 36(b): gate 36(a)'s rig, V 20, on the Super row with uniform Y = 1/2 under the
// chamber's hook order. rhoY == Y rho read bitwise on the row.
BackflowOut contactBlend(double V, double ratio, double transits, bool superRow = true)
{
    const int nx = 96, ny = 152;
    const dns::Grid g = dns::Grid::make(nx, ny, 1, nx * kChamberH, ny * kChamberH, 1.0);
    const GpuGrid gg = GpuGrid::from(g);
    auto dev = G::DeviceComp::make(g, dns::Transport::constant(shockref::kGamma, .72));
    auto red = G::ReduceScratch::make();
    if (!dev || !red) dnstest::skip("device allocation failed");
    G::SpeciesSetup sp;
    sp.nY = 1;
    sp.lnA = -std::numeric_limits<double>::infinity();
    sp.openX = true;
    sp.yIn[0] = .5;
    if (!dev->enableSpecies(sp)) dnstest::skip("species allocation failed");
    G::ShockSetup ss = openX();
    ss.nscbc[0] = true;
    if (!dev->enableShock(ss)) dnstest::skip("shock allocation failed");
    auto ds = G::DeviceCompState::make(g.size(), 0, false, 1);
    auto hs = G::DeviceCompState::make(g.size());
    if (!ds || !hs) dnstest::skip("device allocation failed");
    G::DeviceCompNscbc::Config nc;
    nc.rhoInf = 1;
    nc.uInf = 0;
    nc.pInf = kChamberP;
    nc.gamma = shockref::kGamma;
    nc.superOutflow = superRow;
    auto nm = G::DeviceCompNscbc::make(g, nc);
    auto sm = G::DeviceCompSink::make(g, {.nb = 16, .alpha = 1.0, .cInf = std::sqrt(1.4 * kChamberP)});
    if (!nm || !sm) dnstest::skip("NSCBC init failed");
    G::DeviceCompNscbc nsc = std::move(*nm);
    G::DeviceCompSink snk = std::move(*sm);
    nsc.setFreestream(0);
    auto* d = &*dev;
    const G::DeviceCompState* held = &*hs;
    auto hold = [gg, held](G::DeviceCompState& q) {
        kHoldRows<<<G::launchBlocks(gg.size()), G::blockSize>>>(gg, 4, 0, 0, held->rho.data(), held->mx.data(), held->my.data(),
                                                                held->mz.data(), held->E.data(), q.rho.data(), q.mx.data(),
                                                                q.my.data(), q.mz.data(), q.E.data());
        kHoldY<<<G::launchBlocks(gg.size()), G::blockSize>>>(gg, 4, q.rho.data(), q.y[0].data());
        G::checkLastKernel("kHoldY");
    };
    dev->setStageHook(hold);
    dev->setRhsHook([&, d](const G::DeviceCompState& s, G::DeviceCompState& o, const G::StageCombine& cmb) {
        nsc.apply(*d, s, o, cmb);
        d->speciesOpenConsistency(s, o, cmb);
        snk.apply(s, o, cmb);
        snk.applySpecies(s, o, cmb);
    });
    const auto ic = field2D(g, [&](int i, int j, double, double, double& r, double& u, double& v, double& p) {
        r = j >= 38 && j < 114 && i >= nx - 60 ? 1 / ratio : 1;
        u = 0;
        v = V;
        p = kChamberP;
    });
    if (ds->upload(ic) != hipSuccess || hs->upload(ic) != hipSuccess) dnstest::skip("upload failed");
    std::vector<double> ry(g.size());
    for (std::size_t c = 0; c < g.size(); ++c) ry[c] = double(float(ic.rho[c]) * .5f);
    if (ds->uploadSpecies(0, ry) != hipSuccess) dnstest::skip("upload failed");
    BackflowOut o;
    const double tEnd = transits * ny * kChamberH / V;
    double t = 0;
    long steps = 0;
    auto h = dns::CompState::zeros(g);
    std::vector<double> y0;
    while (t < tEnd && o.ok) {
        const double dt = dev->stableDt(*ds, *red, .4);
        if (!(dt > 0 && std::isfinite(dt))) {
            o.ok = false;
            o.why = std::format("dt {} at t={:.4g} step {}", dt, t, steps);
            break;
        }
        dev->step(*ds, Real(std::min(dt, tEnd - t)));
        t += dt;
        ++steps;
        hold(*ds);
        double in = 0;
        if (steps % 16) continue;
        if (ds->download(h) != hipSuccess || ds->downloadSpecies(0, y0) != hipSuccess) dnstest::skip("download failed");
        for (int j = 0; j < ny; ++j) {
            const std::size_t c = g.idx(nx - 1, j, 0);
            const double r = h.rho[c], u = h.mx[c] / r, w = h.my[c] / r;
            const double p = .4 * (h.E[c] - .5 * r * (u * u + w * w)), mn = u / std::sqrt(1.4 * p / r);
            o.rowT = std::min(o.rowT, p / r / kChamberP);
            o.rowP = std::min(o.rowP, p / kChamberP);
            o.rowRho = std::min(o.rowRho, r);
            o.rowUn = std::min(o.rowUn, mn);
            o.backflow += u < 0;
            in += r * std::min(u, 0.0);
            ++o.checked;
            o.yErr = std::max(o.yErr, std::abs(y0[c] / h.rho[c] - .5));
            if (float(y0[c]) == float(h.rho[c]) * .5f) continue;
            if (!o.bad++)
                o.first = std::format("first t {:.4g} j {}: rho {} rhoY {}", t, j, float(h.rho[c]), float(y0[c]));
        }
        o.in += -in * 16 * dt * g.hy;
        for (std::size_t c = 0; c < g.size(); ++c) {
            const double r = h.rho[c], u = h.mx[c] / r, w = h.my[c] / r, p = .4 * (h.E[c] - .5 * r * (u * u + w * w));
            o.minRho = std::min(o.minRho, r);
            o.minP = std::min(o.minP, p);
        }
        o.ok = o.ok && o.minRho > 0 && o.minP > 0;
        if (!o.ok) o.why = std::format("rho {:.3g}, p {:.3g} at t={:.4g}", o.minRho, o.minP, t);
    }
    return o;
}

// Gate 36.
int outflowContact()
{
    const ContactOut a = contactRun(20, 10, 0, 5), b = contactRun(20, 10, 20, 5), c = contactRun(10, 10, 0, 5);
    auto line = [](const char* name, const ContactOut& x) {
        std::printf("  %s: row T/Tu min %.3f p/p0 min %.3f; box T/Tu min %.3f p/p0 min %.3f; first T/Tu < 0.5 on the row t %.3f p/p0 %.3f "
                    "Mn %.3f, in the box t %.3f i %d; ok %d %s\n",
                    name, x.rowT, x.rowP, x.boxT, x.boxP, x.tCold, x.pCold, x.mnCold, x.tBoxCold, x.iBoxCold, int(x.ok), x.why.c_str());
    };
    line("(a) shipped row, V 20", a);
    line("(a) control, 20 cells off the row", b);
    line("(a) shipped row, V 10, report only", c);
    check(a.rowT < .5 && a.rowP < .9, "(a) the shipped row collapses: T/Tu < 0.5, p falling");
    check(a.iBoxCold == 95, "(a) it starts on the row");
    check(b.boxT >= .5 && b.ok, "(a) the control does not");
    const BackflowOut d = contactBlend(20, 10, 10);
    const double box = 96 * 152 * kChamberH * kChamberH;
    std::printf("  (b) Super, V 20, 10 transits: row T/Tu min %.3f p/p0 min %.3f rho min %.3f, min Mn %.3f; box rho min %.3f p/p0 min %.3f; "
                "row cells in backflow %.1f%%; taken in through x-hi %.4g, %.2f of a box at rho 1; rhoY off Y rho on the row %ld of %ld, |Y - 1/2| "
                "max %.3e %s %s\n",
                d.rowT, d.rowP, d.rowRho, d.rowUn, d.minRho, d.minP / kChamberP, 100.0 * double(d.backflow) / double(d.checked), d.in,
                d.in / box, d.bad, d.checked, d.yErr, d.first.c_str(), d.why.c_str());
    check(d.ok, "(b) rho, p > 0 for 10 transits");
    check(d.rowT >= .5, std::format("(b) row T/Tu {:.3f} >= 0.5", d.rowT));
    // A5's principle, S16: the shipped row is not bitwise on this rig either. It
    // collapses in transit 4, so the two are compared where both still live.
    const BackflowOut e = contactBlend(20, 10, 3), f = contactBlend(20, 10, 3, false);
    std::printf("  (b) 3 transits: row |Y - 1/2| max Super %.3e, shipped %.3e; shipped row T/Tu min %.3f\n", e.yErr, f.yErr, f.rowT);
    check(f.ok, "(b) the shipped control lives 3 transits");
    check(e.yErr <= 2 * f.yErr, std::format("(b) row |Y - 1/2| {:.3e} <= 2 x shipped {:.3e}", e.yErr, f.yErr));
    return dnstest::finish();
}

// Gate 34: the fully supersonic nozzle of gate 17's rig to steady state, from the
// exact solution. Exit-plane error, cell x = 1 - h/2, on one exit treatment.
enum class NozzleExit { Super, Shipped, Held, Extrapolated };
ExitErr superNozzle(int N, double tEnd, NozzleExit how)
{
    const bool row = how == NozzleExit::Super || how == NozzleExit::Shipped;
    const int ng = 3, nt = N + ng + (row ? 0 : ng), last = row ? nt - 1 : nt - ng - 1;
    const double h = 1.0 / N;
    const shockref::Nozzle nz;
    const dns::Grid g = dns::Grid::make(nt, 4, 1, nt * h, 4 * h, 1.0);
    std::vector<double> xg(static_cast<std::size_t>(nt));
    std::vector<Real> ar(static_cast<std::size_t>(nt));
    std::vector<double> rr(xg), uu(xg), pp(xg);
    for (int i = 0; i < nt; ++i) {
        const std::size_t I = std::size_t(i);
        xg[I] = (i - ng + .5) * h;
        ar[I] = Real(shockref::Nozzle::dArea(xg[I]) / shockref::Nozzle::area(xg[I]));
        const shockref::Prim q = nz.exact(xg[I], false);
        rr[I] = q.r;
        uu[I] = q.u;
        pp[I] = q.p;
    }
    double Mx = 0;
    const shockref::Prim qx = nz.exact(xg[std::size_t(last)], false, &Mx);
    G::ShockSetup ss = openX();
    ss.nscbc[0] = row;
    auto r = makeRig(g, rowsOf(g, shockref::make<float>(rr, uu, pp)), 0, 0, ss);
    auto da = G::DeviceBuffer<Real>::make(ar.size());
    if (!da || da->upload(ar.data()) != hipSuccess) dnstest::skip("upload failed");
    std::optional<G::DeviceCompNscbc> nsc;
    if (row) {
        G::DeviceCompNscbc::Config nc;
        nc.rhoInf = 1;
        nc.uInf = 0;
        nc.pInf = qx.p;
        nc.gamma = shockref::kGamma;
        nc.superOutflow = how == NozzleExit::Super;
        auto n = G::DeviceCompNscbc::make(g, nc);
        if (!n) dnstest::skip("NSCBC init failed");
        nsc.emplace(std::move(*n));
        nsc->setFreestream(0);
    }
    const GpuGrid gg = r->gg;
    Rig* p = r.get();
    r->rxHi = how == NozzleExit::Held ? ng : 0;
    const int mode = how == NozzleExit::Extrapolated ? 0 : 2;
    auto bc = [gg, ng, mode, p](G::DeviceCompState& s) {
        kNozzleBc<<<G::launchBlocks(gg.size()), G::blockSize>>>(gg, ng, Real(0), mode, s.rho.data(), s.mx.data(), s.E.data());
        G::checkLastKernel("kNozzleBc");
        p->hold(s);
    };
    p->dev->setStageHook(bc);
    p->dev->setRhsHook([gg, &da, &nsc, p](const G::DeviceCompState& s, G::DeviceCompState& o, const G::StageCombine& cmb) {
        if (nsc) nsc->apply(*p->dev, s, o, cmb);
        kNozzleSrc<<<G::launchBlocks(gg.size()), G::blockSize>>>(gg, da->data(), cmb, s.rho.data(), s.mx.data(), s.E.data(),
                                                                 o.rho.data(), o.mx.data(), o.E.data());
        G::checkLastKernel("kNozzleSrc");
    });
    bc(*r->s);
    // The hook runs on a stage's input: the state stableDt reads gets its rows here,
    // else rows 0 and nt - 1 keep the wrap's flux (the extrapolated row: E < 0).
    const Marched m = march(*r, tEnd, .4, [&](double, long) { bc(*r->s); });
    ExitErr o;
    o.ok = m.ok;
    o.why = m.why;
    const auto s = download(*r);
    const std::size_t c = g.idx(last, 0, 0);
    const double u = s.mx[c] / s.rho[c], pr = .4 * (s.E[c] - .5 * s.mx[c] * u);
    o.M = std::abs(u / std::sqrt(1.4 * pr / s.rho[c]) / Mx - 1);
    o.p = std::abs(pr / qx.p - 1);
    return o;
}

int outflowNozzle()
{
    double lowM = NAN, lowP = NAN, highM = NAN, highP = NAN;
    for (const int N : {200, 400, 800}) {
        const ExitErr a = superNozzle(N, 6, NozzleExit::Super), c = superNozzle(N, 6, NozzleExit::Extrapolated);
        const ExitErr b = superNozzle(N, 6, NozzleExit::Held), d = superNozzle(N, 6, NozzleExit::Shipped);
        std::printf("  N %d exit-plane M, p error: Super %.3e %.3e; extrapolated %.3e %.3e; held %.3e %.3e; shipped row %.3e %.3e %s\n",
                    N, a.M, a.p, c.M, c.p, b.M, b.p, d.M, d.p, (a.why + " | " + c.why).c_str());
        check(a.ok && c.ok, std::format("N {} marches", N));
        check(a.M <= c.M, std::format("N {} M error {:.3e} <= extrapolated {:.3e}", N, a.M, c.M));
        check(a.p <= c.p, std::format("N {} p error {:.3e} <= extrapolated {:.3e}", N, a.p, c.p));
        if (N == 200) {
            lowM = a.M;
            lowP = a.p;
        }
        highM = a.M;
        highP = a.p;
    }
    const double orderM = std::log2(lowM / highM) / 2, orderP = std::log2(lowP / highP) / 2;
    std::printf("  order N 200 to 800: M %.2f, p %.2f\n", orderM, orderP);
    check(orderM >= 1, std::format("M order {:.2f} >= 1", orderM));
    check(orderP >= 1, std::format("p order {:.2f} >= 1", orderP));
    return dnstest::finish();
}

struct ExitShock
{
    double left = 0, rhoMin = HUGE_VAL, pMin = HUGE_VAL;
    bool ok = false;
    std::string why;
};

// Gate 35: a shock of Mach Ms running into M 2 flow leaves through the row. left:
// the peak of max |p - p2|/(p2 - p1) once the start-up's slowest wave has left,
// over one more crossing of it: everything the exit left in the box.
ExitShock shockExit(double Ms, bool sink, bool super)
{
    const dns::Grid g = dns::Grid::make(320, 8, 1, 1.0, 8.0 / 320, 1.0);
    const double gm = 1.4, p1 = 1 / gm, u1 = 2, x0 = .4;
    const double p2 = p1 * (1 + 2 * gm / (gm + 1) * (Ms * Ms - 1));
    const double r2 = (gm + 1) * Ms * Ms / ((gm - 1) * Ms * Ms + 2);
    const double Vs = u1 + Ms, u2 = Vs - (Vs - u1) / r2, slow = u2 - std::sqrt(gm * p2 / r2);
    const auto ic = field2D(g, [&](int, int, double x, double, double& r, double& u, double& v, double& p) {
        const bool behind = x < x0;
        r = behind ? r2 : 1;
        u = behind ? u2 : u1;
        v = 0;
        p = behind ? p2 : p1;
    });
    auto o = superRig(g, ic, super, sink, p1);
    const double tWash = std::max((1 - x0) / Vs, (1 - x0) / slow) + 8 * g.hx / slow, tEnd = tWash + 1 / slow;
    ExitShock out;
    const Marched m = march(*o->r, tEnd, .4, [&](double t, long step) {
        if (t < tWash || step % 4) return;
        const auto s = download(*o->r);
        for (int i = 4; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, 0, 0);
            const double u = s.mx[c] / s.rho[c], pr = .4 * (s.E[c] - .5 * s.mx[c] * u);
            out.left = std::max(out.left, std::abs(pr - p2) / (p2 - p1));
        }
    });
    out.ok = m.ok;
    out.why = m.why;
    out.rhoMin = m.rhoMin;
    out.pMin = m.pMin;
    return out;
}

int outflowShock()
{
    for (const double Ms : {2.0, 3.0}) {
        const ExitShock a = shockExit(Ms, true, true), b = shockExit(Ms, false, true), c = shockExit(Ms, true, false);
        std::printf("  Ms %.0f left behind, sink on %.3e, sink off %.3e; shipped row, sink on %.3e; rhoMin %.3f pMin %.3f %s %s\n", Ms,
                    a.left, b.left, c.left, a.rhoMin, a.pMin, a.why.c_str(), c.why.c_str());
        check(a.ok, std::format("Ms {}: rho, p > 0 through the exit", Ms));
        check(a.left <= .05, std::format("Ms {}: left behind {:.3e} <= 5% of the jump", Ms, a.left));
    }
    return dnstest::finish();
}

// Gate 31's rig: 10% of the cells sensed, 256^3, slabs of pressure along x; run
// under `simjob --bench` only.
int bench10()
{
    const int n = 256;
    const dns::Grid g = dns::Grid::make(n, n, n, 1, 1, 1);
    auto ic = dns::CompState::zeros(g);
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t c = g.idx(i, j, k);
                ic.rho[c] = 1;
                ic.E[c] = (i / 48 % 2 ? 1.5 : 1.0) / .4;
            }
    double ms[4] = {}, frac = 0;
    for (int q = 0; q < 4; ++q) {
        std::optional<G::ShockSetup> sh;
        if (q % 2) sh = G::ShockSetup{};
        auto r = makeRig(g, ic, 0, 0, sh, q < 2 ? 0.0 : 1e-3); // the viscous pair: report only
        for (int s = 0; s < 3; ++s) r->dev->step(*r->s, Real(1e-5));
        (void)hipDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < 20; ++s) r->dev->step(*r->s, Real(1e-5));
        (void)hipDeviceSynchronize();
        ms[q] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20;
        if (q == 1) frac = r->dev->shockSense(*r->s, *r->red) / double(g.size());
    }
    std::printf("  256^3: off %.2f ms/step, hll %.2f ms/step (%+.1f%%), sensed %.2f%%\n", ms[0], ms[1], 100 * (ms[1] / ms[0] - 1),
                100 * frac);
    std::printf("  viscous: off %.2f ms/step, hll %.2f ms/step (%+.1f%%)\n", ms[2], ms[3], 100 * (ms[3] / ms[2] - 1));
    check(frac >= .05, "the rig senses a sizeable fraction");
    check(ms[1] <= 1.4 * ms[0], "sensed rig <= +40%");
    return dnstest::finish();
}

// Gate 26 (extrusion, permutation): fixed dt, the same run on two grids.
int extrude()
{
    const int nx = 96, ny = 16, steps = 120;
    const double dt = 2e-4;
    auto sod = [&](const dns::Grid& g, bool oblique) {
        return field2D(g, [&](int i, int j, double, double, double& r, double& u, double& v, double& p) {
            const bool m = oblique ? i + 2 * j < 60 : i < nx / 2;
            r = m ? 1.0 : .125;
            u = v = 0;
            p = m ? 1.0 : .1;
        });
    };
    for (const bool oblique : {false, true}) {
        const dns::Grid g2 = dns::Grid::make(nx, ny, 1, 1.0, 1.0 * ny / nx, 1.0);
        const dns::Grid g3 = dns::Grid::make(nx, ny, 8, 1.0, 1.0 * ny / nx, 8.0 / nx);
        G::ShockSetup a = openX();
        a.open[1] = oblique;
        auto r2 = makeRig(g2, sod(g2, oblique), 4, oblique ? 4 : 0, a);
        auto r3 = makeRig(g3, sod(g3, oblique), 4, oblique ? 4 : 0, a);
        for (int q = 0; q < steps; ++q) {
            r2->dev->step(*r2->s, Real(dt));
            r3->dev->step(*r3->s, Real(dt));
        }
        r2->hold(*r2->s);
        r3->hold(*r3->s);
        const auto s2 = download(*r2), s3 = download(*r3);
        bool same = true, wzero = true;
        for (int k = 0; k < g3.nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const std::size_t c2 = g2.idx(i, j, 0), c3 = g3.idx(i, j, k);
                    same = same && s2.rho[c2] == s3.rho[c3] && s2.mx[c2] == s3.mx[c3] && s2.my[c2] == s3.my[c3] &&
                           s2.E[c2] == s3.E[c3];
                    wzero = wzero && s3.mz[c3] == 0;
                }
        const double sensed = r2->dev->shockSense(*r2->s, *r2->red);
        std::printf("  extrusion %s: sensed %.0f cells of %d (%.2f%%)\n", oblique ? "oblique" : "Sod", sensed, nx * ny,
                    100 * sensed / (nx * ny));
        check(sensed > 0, std::format("extrusion {}: the sensor fired", oblique ? "oblique" : "Sod"));
        check(same, std::format("extrusion {}: nz 8 == 2D per plane bitwise", oblique ? "oblique" : "Sod"));
        check(wzero, std::format("extrusion {}: w exactly 0", oblique ? "oblique" : "Sod"));
    }
    // Permutation: a double shock along x, y, z of a cube box, periodic.
    {
        const int n = 48, t = 8;
        auto rig = [&](int axis) {
            const int dims[3] = {axis == 0 ? n : t, axis == 1 ? n : t, axis == 2 ? n : t};
            const double h = 1.0 / n;
            const dns::Grid g = dns::Grid::make(dims[0], dims[1], dims[2], dims[0] * h, dims[1] * h, dims[2] * h);
            auto ic = dns::CompState::zeros(g);
            for (int k = 0; k < g.nz; ++k)
                for (int j = 0; j < g.ny; ++j)
                    for (int i = 0; i < g.nx; ++i) {
                        const int q[3] = {i, j, k};
                        const std::size_t c = g.idx(i, j, k);
                        const double u = q[axis] < n / 2 ? 2 : -2;
                        std::vector<double>* mom[3] = {&ic.mx, &ic.my, &ic.mz};
                        ic.rho[c] = 1;
                        (*mom[axis])[c] = u;
                        ic.E[c] = 1 / .4 + .5 * u * u;
                    }
            return makeRig(g, ic, 0, 0, G::ShockSetup{});
        };
        auto rx = rig(0), ry = rig(1), rz = rig(2);
        for (int q = 0; q < 100; ++q) {
            rx->dev->step(*rx->s, Real(2e-4));
            ry->dev->step(*ry->s, Real(2e-4));
            rz->dev->step(*rz->s, Real(2e-4));
        }
        const auto sx = download(*rx), sy = download(*ry), sz = download(*rz);
        bool same = true;
        for (int a = 0; a < n; ++a) {
            const std::size_t cx = rx->g.idx(a, 0, 0), cy = ry->g.idx(0, a, 0), cz = rz->g.idx(0, 0, a);
            same = same && sx.rho[cx] == sy.rho[cy] && sx.rho[cx] == sz.rho[cz] && sx.mx[cx] == sy.my[cy] &&
                   sx.mx[cx] == sz.mz[cz] && sx.E[cx] == sy.E[cy] && sx.E[cx] == sz.E[cz];
        }
        const double sensed = rx->dev->shockSense(*rx->s, *rx->red);
        std::printf("  permutation: sensed %.0f cells of %zu\n", sensed, rx->g.size());
        check(sensed > 0, "permutation: the sensor fired");
        check(same, "bitwise under axis permutation (double shock along x, y, z)");
    }
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    const std::string_view m = argc > 1 ? argv[1] : "";
    if (m == "sense") return sense();
    if (m == "sensor") return sensor();
    if (m == "riemann") return riemann();
    if (m == "shuosher") return shuosher();
    if (m == "aligned") return aligned();
    if (m == "vortex") return vortexGate();
    if (m == "contact") return contactGate();
    if (m == "conservation") return conservation();
    if (m == "extrude") return extrude();
    if (m == "ghost") return ghostGate();
    if (m == "open") return openGate();
    if (m == "uniformy") return uniformY();
    if (m == "flame") return flameGate();
    if (m == "quiet") return quiet();
    if (m == "lip") return lip();
    if (m == "cadence") return cadence();
    if (m == "restart") return restartGate();
    if (m == "nozzle") return nozzle();
    if (m == "oblique") return oblique();
    if (m == "3d") return threeD();
    if (m == "chamber") return chamber();
    if (m == "guard") return guardGate();
    if (m == "choked") return chokedGate();
    if (m == "outflow_bytes") {
        const int a = sense();
        const int b = quiet();
        return a ? a : b;
    }
    if (m == "outflow_pass") return outflowPass();
    if (m == "outflow_nozzle") return outflowNozzle();
    if (m == "outflow_shock") return outflowShock();
    if (m == "outflow_contact") return outflowContact();
    if (m == "bench10") return bench10();
    std::printf("unknown case '%s'\n", argc > 1 ? argv[1] : "");
    return 2;
}
