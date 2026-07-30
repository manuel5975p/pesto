// What the open boundary CANNOT absorb, measured: the small-grid-scale
// acceptance spectrum of the characteristic far field, as a table.
//
// gpu_acoustics pins the interior facts in a periodic box: the central
// skew-symmetric flux eats nothing anywhere, and its semi-discrete group
// velocity is c cos(kh) -- zero at 4 cells per wavelength, NEGATIVE below.
// This test asks what those facts mean for a BOUNDARY: a wavepacket at kh is
// released mid-box against a characteristic outflow, and the fraction of its
// acoustic energy still inside after a full crossing time is the fraction the
// face never got the chance to absorb. No face-local closure -- LODI, Yoo-Im,
// adaptive, modal -- can act on content that does not arrive; whatever this
// table keeps is the specification for an explicit in-band sink
// (DeviceCompSink), not a defect of the characteristic rows.
//
// The vortical rows are the same question for the other Kovasznay family:
// a compact fine-scale solenoidal patch advected through the outflow at the
// mean flow, residual vorticity by a NON-WRAPPING scan (the x extremes are
// not periodic neighbours; a wrapped difference there once fabricated a
// vorticity source, docs/nscbc.md). Peak position is reported next to every
// number, and the packet rows are sampled twice in time -- a residual that is
// still moving is a wave in transit, not a boundary verdict.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_far.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/comp_sink.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

using namespace dns;

namespace {

using dnstest::check;

constexpr double gam = 1.4, rhoInf = 1.0, cInf = 1.0;
constexpr double pInf = rhoInf * cInf * cInf / gam;
constexpr double Lbox = 2.56; // fixed physical box: n sets h alone
constexpr double twoPi = 2.0 * std::numbers::pi;

// Peak of a field with its position: every boundary number states where it
// sits (a peak ON an extreme row is a different claim than one mid-box).
struct Peak { double v = 0; int i = 0, j = 0; };

// x-characteristic faces, free-slip y walls, no sponge anywhere: the pure
// characteristic boundary the acceptance spectrum is quoted against.
struct Rig
{
    Grid g;
    std::expected<gpu::DeviceComp, hipError_t> dev;
    std::expected<gpu::DeviceCompFar, hipError_t> far;
    std::expected<gpu::DeviceCompNscbc, hipError_t> nsc;
    std::expected<gpu::DeviceCompState, hipError_t> ds;
    std::expected<gpu::ReduceScratch, hipError_t> red;
    std::optional<gpu::DeviceCompSink> snk;

    // sinkNb > 0 layers the scale-selective band under the characteristic
    // rows, exactly as runner_comp composes the two: nscbc overwrites its
    // rows' rhs, then the sink ADDS its term on top.
    Rig(int nx, int ny, double uMean, int sinkNb = 0, double sinkAlpha = 1.0)
        : g(Grid::make(nx, ny, 1, Lbox, ny * (Lbox / nx), 1.0)),
          dev(gpu::DeviceComp::make(g, gam, 0.0)),
          far(gpu::DeviceCompFar::make(
              g, {rhoInf, uMean, pInf, gam, 8, 0.0, false, false})),
          nsc(gpu::DeviceCompNscbc::make(
              g, gpu::DeviceCompNscbc::Config{rhoInf, uMean, pInf, gam})),
          ds(gpu::DeviceCompState::make(g.size())),
          red(gpu::ReduceScratch::make())
    {
        if (!dev || !far || !nsc || !ds || !red) std::exit(EXIT_FAILURE);
        if (sinkNb > 0) {
            auto s = gpu::DeviceCompSink::make(
                g, {.nb = sinkNb, .alpha = sinkAlpha, .cInf = cInf});
            if (!s) std::exit(EXIT_FAILURE);
            snk.emplace(std::move(*s));
        }
        dev->setStageHook([this](gpu::DeviceCompState& st) { far->apply(st); });
        dev->setRhsHook([this](const gpu::DeviceCompState& st,
                               gpu::DeviceCompState& out,
                               const gpu::StageCombine& cmb) {
            nsc->apply(*dev, st, out, cmb);
            if (snk) snk->apply(st, out, cmb);
        });
    }

    void run(double tEnd)
    {
        double t = 0.0;
        while (t < tEnd) {
            const double dt = std::min(tEnd - t, dev->stableDt(*ds, *red));
            dev->step(*ds, gpu::Real(dt));
            t += dt;
        }
    }

    void set(auto prim) // prim(x, y) -> {rho, u, v, p}
    {
        auto h = CompState::zeros(g);
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const auto [rho, u, v, p] =
                    prim((i + 0.5) * g.hx, (j + 0.5) * g.hy);
                h.rho[c] = rho;
                h.mx[c] = rho * u;
                h.my[c] = rho * v;
                h.E[c] = p / (gam - 1.0) + 0.5 * rho * (u * u + v * v);
            }
        (void)ds->upload(h);
    }

    // Acoustic energy in the box: p'^2/(2 rho c^2) + rho |u'|^2 / 2 summed.
    // Position-blind on purpose -- a packet the scheme has torn into a wave
    // train still counts in full until it actually leaves.
    double acousticEnergy(double uMean)
    {
        auto h = CompState::zeros(g);
        (void)ds->download(h);
        double e = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c) {
            const double rho = h.rho[c];
            if (!std::isfinite(rho) || rho <= 0.0) return 1e18;
            const double u = h.mx[c] / rho - uMean, v = h.my[c] / rho;
            const double ke = 0.5 * (u * u + v * v);
            const double p =
                (gam - 1.0) *
                (h.E[c] - h.rho[c] * (ke + u * uMean + 0.5 * uMean * uMean));
            const double dp = p - pInf;
            e += dp * dp / (2.0 * rhoInf * cInf * cInf) + rho * ke;
        }
        return e;
    }

    Peak pressurePeak()
    {
        auto h = CompState::zeros(g);
        (void)ds->download(h);
        Peak pk{};
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const double rho = h.rho[c];
                const double u = h.mx[c] / rho, v = h.my[c] / rho;
                const double p = (gam - 1.0) *
                                 (h.E[c] - 0.5 * rho * (u * u + v * v));
                if (std::abs(p - pInf) > pk.v)
                    pk = {std::abs(p - pInf), i, j};
            }
        return pk;
    }

    // Peak |dv/dx - du/dy| by a NON-WRAPPING scan: one-sided differences at
    // every extreme (index clamped), margin 0. Peeling cells off instead
    // would peel off exactly where a boundary-laid sheet would live.
    Peak vorticityPeak()
    {
        auto h = CompState::zeros(g);
        (void)ds->download(h);
        Peak pk{};
        auto vel = [&](int i, int j, int comp) {
            const std::size_t c = g.idx(i, j, 0);
            return (comp ? h.my[c] : h.mx[c]) / h.rho[c];
        };
        for (std::size_t c = 0; c < g.size(); ++c)
            if (!std::isfinite(h.rho[c]) || h.rho[c] <= 0.0)
                return {1e18, -1, -1};
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const int ip = std::min(i + 1, g.nx - 1);
                const int im = std::max(i - 1, 0);
                const int jp = std::min(j + 1, g.ny - 1);
                const int jm = std::max(j - 1, 0);
                const double dvdx =
                    (vel(ip, j, 1) - vel(im, j, 1)) / ((ip - im) * g.hx);
                const double dudy =
                    (vel(i, jp, 0) - vel(i, jm, 0)) / ((jp - jm) * g.hy);
                if (std::abs(dvdx - dudy) > pk.v)
                    pk = {std::abs(dvdx - dudy), i, j};
            }
        return pk;
    }
};

// Acoustic wavepacket at kh: a +x travelling linear wave sin(k(x - x0))
// under a Gaussian envelope, released mid-box. Returns the fraction of its
// acoustic energy still inside after 1.3 crossing times -- the fraction the
// outflow never received. Quiescent: the group speed c cos(kh) is then the
// only transport, which is the property under test.
struct PacketLeft { double frac; Peak pk; };

PacketLeft packetRemainder(int nx, double khOverPi, double amp, int sinkNb = 0,
                           double x0frac = 0.5, double sigEnv = 0.32)
{
    Rig r(nx, 8, 0.0, sinkNb);
    const double h = r.g.hx;
    const double k = khOverPi * std::numbers::pi / h;
    const double x0 = x0frac * Lbox, sig = sigEnv;
    r.set([&](double x, double) {
        const double pp = amp * pInf * std::sin(k * (x - x0)) *
                          std::exp(-(x - x0) * (x - x0) / (sig * sig));
        return std::array{rhoInf + pp / (cInf * cInf), pp / (rhoInf * cInf),
                          0.0, pInf + pp};
    });
    const double e0 = r.acousticEnergy(0.0);
    r.run(1.3 * ((Lbox - x0) + 3 * sig) / cInf);
    const double e1 = r.acousticEnergy(0.0);
    return {e1 / e0, r.pressurePeak()};
}

// Fine-scale solenoidal patch: psi = A exp(-r^2/sig^2) cos(kx) cos(ky),
// u = dpsi/dy, v = -dpsi/dx (analytic; exactly divergence-free). Advected at
// uMean through the outflow; residual = peak vorticity after the crossing
// window over the initial peak, non-wrapping scan, sampled TWICE in time.
struct VortLeft { double frac, fracLate; Peak pk; };

VortLeft patchResidual(int nx, int ny, double khOverPi, double mach)
{
    const double uMean = mach * cInf;
    Rig r(nx, ny, uMean);
    const double h = r.g.hx;
    const double k = khOverPi * std::numbers::pi / h;
    const double sig = 16 * h;
    const double x0 = 0.6 * Lbox, y0 = 0.5 * r.g.ny * r.g.hy;
    // |u'| ~ A k at the fine scale: fix the VELOCITY at 5% of c so every row
    // is equally linear (an amplitude growing with k steepens and shocks).
    const double A = 0.05 * cInf / k;
    r.set([&](double x, double y) {
        const double dx = x - x0, dy = y - y0;
        const double env = std::exp(-(dx * dx + dy * dy) / (sig * sig));
        const double u =
            A * env * (std::cos(k * dx) * (-k * std::sin(k * dy)) +
                       std::cos(k * dx) * std::cos(k * dy) * (-2 * dy / (sig * sig)));
        const double v =
            -A * env * ((-k * std::sin(k * dx)) * std::cos(k * dy) +
                        std::cos(k * dx) * std::cos(k * dy) * (-2 * dx / (sig * sig)));
        return std::array{rhoInf, uMean + u, v, pInf};
    });
    const double w0 = r.vorticityPeak().v;
    const double tExit = (Lbox - x0 + 3 * sig) / uMean;
    r.run(tExit);
    const Peak p1 = r.vorticityPeak();
    // Second sample a full window later: what is merely slow (advective group
    // speed u cos(kh) < u) keeps draining, what cannot travel at all is flat.
    r.run(tExit);
    const Peak p2 = r.vorticityPeak();
    return {p1.v / w0, p2.v / w0, p2};
}

// ---- operator in isolation: one apply() on a crafted state ---------------

// out - in per field after a single apply with cr = 1: the raw rhs.
struct SinkRhs { double maxAbs[5]; double sum[5]; double sumAbs[5]; };

SinkRhs sinkRhs(const Grid& g, gpu::DeviceCompSink& snk, const CompState& h)
{
    auto s = gpu::DeviceCompState::make(g.size());
    auto o = gpu::DeviceCompState::make(g.size());
    if (!s || !o) std::exit(EXIT_FAILURE);
    (void)s->upload(h);
    (void)o->upload(h);
    gpu::StageCombine cmb{};
    cmb.cr = gpu::Real(1);
    snk.apply(*s, *o, cmb);
    // Baseline is the ROUNDTRIPPED state (double -> FP32 -> double), so the
    // comparison sees only what apply() wrote, not the upload quantization.
    auto hin = CompState::zeros(g);
    (void)s->download(hin);
    auto ho = CompState::zeros(g);
    (void)o->download(ho);
    SinkRhs r{};
    const double* in[5] = {hin.rho.data(), hin.mx.data(), hin.my.data(),
                           hin.mz.data(), hin.E.data()};
    const double* out[5] = {ho.rho.data(), ho.mx.data(), ho.my.data(),
                            ho.mz.data(), ho.E.data()};
    for (int v = 0; v < 5; ++v)
        for (std::size_t c = 0; c < g.size(); ++c) {
            const double d = out[v][c] - in[v][c];
            r.maxAbs[v] = std::max(r.maxAbs[v], std::abs(d));
            r.sum[v] += d;
            r.sumAbs[v] += std::abs(d);
        }
    return r;
}

CompState fillState(const Grid& g, auto prim) // prim(x, y) -> {rho, u, v, p}
{
    auto h = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const auto [rho, u, v, p] = prim((i + 0.5) * g.hx, (j + 0.5) * g.hy);
            h.rho[c] = rho;
            h.mx[c] = rho * u;
            h.my[c] = rho * v;
            h.E[c] = p / (gam - 1.0) + 0.5 * rho * (u * u + v * v);
        }
    return h;
}

// Host mirror of the band profile: the per-cell damping rate of a 2dx mode,
// sigma_eff(i) = (faceSig(i) + faceSig(i-1)) / 2 with zero-flux domain faces.
double sinkRate2dx(int i, int n, int nb, double sigMax)
{
    auto ramp = [&](int q) {
        double d;
        if (q < nb) d = double(nb - q) / nb;
        else if (q >= n - nb) d = double(q - (n - nb) + 1) / nb;
        else return 0.0;
        return 0.5 * (1.0 - std::cos(std::numbers::pi * d));
    };
    auto faceSig = [&](int m) { // face m+1/2
        if (m < 0 || m >= n - 1) return 0.0;
        return 0.5 * (ramp(m) + ramp(m + 1)) * sigMax;
    };
    return 0.5 * (faceSig(i) + faceSig(i - 1));
}

// A 2dx pressure ripple parked INSIDE the band, against the analytic decay:
// each cell damps at its own sigma_eff (sin^4 symbol = 1 at 2dx), and the
// scheme itself neither damps nor transports the mode (gpu_acoustics), so
// E(T)/E(0) = sum w_i exp(-2 sigma_i T) / sum w_i is the whole story bar the
// RK3-vs-exp integration error (<1% at sigma dt = 0.4).
struct Decay2dx { double measured, predicted; };

Decay2dx inBandDecay(int nx, int nb, double alpha)
{
    Rig r(nx, 8, 0.0, nb, alpha);
    const double h = r.g.hx;
    const double xc = Lbox - 0.5 * nb * h, sigE = 0.25 * nb * h;
    const double amp = 1e-3;
    r.set([&](double x, double) {
        const int i = int(std::lround(x / h - 0.5));
        const double pp = amp * pInf * (i % 2 ? -1.0 : 1.0) *
                          std::exp(-(x - xc) * (x - xc) / (sigE * sigE));
        return std::array{rhoInf + pp / (cInf * cInf), 0.0, 0.0, pInf + pp};
    });
    const double e0 = r.acousticEnergy(0.0);
    const double sigMax = alpha * cInf / h;
    const double T = 1.0 / sigMax;
    r.run(T);
    const double e1 = r.acousticEnergy(0.0);
    double num = 0.0, den = 0.0;
    for (int i = 0; i < nx; ++i) {
        const double x = (i + 0.5) * h;
        const double w = std::exp(-2.0 * (x - xc) * (x - xc) / (sigE * sigE));
        num += w * std::exp(-2.0 * T * sinkRate2dx(i, nx, nb, sigMax));
        den += w;
    }
    return {e1 / e0, num / den};
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();

    // ---- acceptance spectrum of the characteristic outflow --------------
    std::printf("acoustic wavepacket vs characteristic outflow, quiescent,\n"
                "n = 256, amp 1e-3, envelope sigma = 0.32; fraction of the\n"
                "packet's acoustic energy STILL IN THE BOX after 1.3 crossings\n");
    std::printf("   kh      ppw    E_left/E_0    peak |p'| at (i,j)\n");
    const double khs[4] = {0.25, 0.5, 0.75, 1.0};
    double left[4] = {};
    for (int i = 0; i < 4; ++i) {
        const auto [frac, pk] = packetRemainder(256, khs[i], 1e-3);
        left[i] = frac;
        std::printf("  %4.2fpi  %5.2f    %9.4f      (%d,%d)\n", khs[i],
                    2.0 / khs[i], frac, pk.i, pk.j);
    }

    // ---- the vortical family through the outflow ------------------------
    std::printf("\nfine-scale solenoidal patch through the outflow, M 0.2,\n"
                "256x128, sigma = 16h; peak |omega| over initial peak,\n"
                "non-wrapping scan, sampled at t_exit and 2 t_exit\n");
    std::printf("   kh      ppw    resid    resid(late)   peak at (i,j)\n");
    const double vkhs[2] = {0.25, 0.5};
    double vleft[2] = {}, vlate[2] = {};
    Peak vpk[2] = {};
    for (int i = 0; i < 2; ++i) {
        const auto [f1, f2, pk] = patchResidual(256, 128, vkhs[i], 0.2);
        vleft[i] = f1;
        vlate[i] = f2;
        vpk[i] = pk;
        std::printf("  %4.2fpi  %5.2f   %6.4f     %6.4f       (%d,%d)\n",
                    vkhs[i], 2.0 / vkhs[i], f1, f2, pk.i, pk.j);
    }

    // ---- the sink operator, isolated --------------------------------------
    std::printf("\nDeviceCompSink operator, one apply() at cr = 1, 128x64,\n"
                "nb = 16, alpha = 1, x and y bands\n");
    Grid og = Grid::make(128, 64, 1, Lbox, 64 * (Lbox / 128), 1.0);
    auto osnk = gpu::DeviceCompSink::make(
        og, {.nb = 16, .alpha = 1.0, .cInf = cInf, .openY = true});
    if (!osnk) std::exit(EXIT_FAILURE);

    auto hConst = fillState(og, [](double, double) {
        return std::array{rhoInf, 0.25, -0.125, pInf};
    });
    const SinkRhs rc = sinkRhs(og, *osnk, hConst);
    double rcMax = 0;
    for (double m : rc.maxAbs) rcMax = std::max(rcMax, m);
    std::printf("  constant state:   max |rhs| = %.3e (must be 0.0 exactly)\n",
                rcMax);

    auto hLin = CompState::zeros(og); // conserved fields LINEAR in x and y
    for (int j = 0; j < og.ny; ++j)
        for (int i = 0; i < og.nx; ++i) {
            const std::size_t c = og.idx(i, j, 0);
            const double x = (i + 0.5) * og.hx, y = (j + 0.5) * og.hy;
            hLin.rho[c] = 1.0 + 0.02 * x + 0.01 * y;
            hLin.mx[c] = 0.05 * x - 0.02 * y;
            hLin.my[c] = -0.03 * y + 0.01 * x;
            hLin.E[c] = pInf / (gam - 1.0) + 0.1 * x + 0.05 * y;
        }
    const SinkRhs rl = sinkRhs(og, *osnk, hLin);
    double rlMax = 0;
    for (double m : rl.maxAbs) rlMax = std::max(rlMax, m);
    std::printf("  linear ramps:     max |rhs| = %.3e (FP32 roundoff only)\n",
                rlMax);

    auto hMix = fillState(og, [&](double x, double y) {
        const int i = int(std::lround(x / og.hx - 0.5));
        const int j = int(std::lround(y / og.hy - 0.5));
        const double chk = 0.02 * ((i + j) % 2 ? -1.0 : 1.0);
        const double s = 0.05 * std::sin(twoPi * 7 * x / Lbox) *
                             std::sin(twoPi * 3 * y / (64 * og.hy)) + chk;
        return std::array{rhoInf * (1 + s), 0.3 * s, -0.2 * s,
                          pInf * (1 + 1.4 * s)};
    });
    const SinkRhs rm = sinkRhs(og, *osnk, hMix);
    double consWorst = 0;
    for (int v = 0; v < 5; ++v)
        if (rm.sumAbs[v] > 0)
            consWorst = std::max(consWorst, std::abs(rm.sum[v]) / rm.sumAbs[v]);
    std::printf("  mixed waves:      |sum rhs| / sum |rhs| = %.3e per field "
                "worst\n", consWorst);

    // ---- 2dx decay against the analytic band profile ----------------------
    const Decay2dx d1 = inBandDecay(256, 16, 1.0);
    const Decay2dx d2 = inBandDecay(256, 16, 2.0);
    std::printf("\n2dx ripple parked in the band, E(T)/E(0) at T = 1/sigma_max\n"
                "  alpha 1: measured %.4f  predicted %.4f\n"
                "  alpha 2: measured %.4f  predicted %.4f\n",
                d1.measured, d1.predicted, d2.measured, d2.predicted);

    // ---- selectivity ------------------------------------------------------
    const double off16 = packetRemainder(256, 0.125, 1e-3).frac;
    const double on16 = packetRemainder(256, 0.125, 1e-3, 16).frac;
    const double onMid4 = packetRemainder(256, 0.5, 1e-3, 16).frac;
    const double offBand4 =
        packetRemainder(256, 0.5, 1e-3, 0, 1.0 - 8.0 / 256, 0.04).frac;
    const double onBand4 =
        packetRemainder(256, 0.5, 1e-3, 16, 1.0 - 8.0 / 256, 0.04).frac;
    std::printf("\nselectivity, E_left/E_0: 16dx packet through the band\n"
                "  sink off %.4f  sink 16 %.4f  (added loss/reflection)\n"
                "4dx packet mid-box (band cannot reach it, by design):\n"
                "  sink off %.4f  sink 16 %.4f\n"
                "4dx packet parked IN the band:\n"
                "  sink off %.4f  sink 16 %.4f\n",
                off16, on16, left[1], onMid4, offBand4, onBand4);

    std::printf("\n");
    // The facts this file exists to pin (the sink's specification), measured
    // 2026-07-30 at n = 256: E_left/E_0 = 0.0215 / 0.2319 / 0.6869 / 0.9783
    // over kh = pi/4 .. pi. Resolved content leaves; the un-transportable
    // fraction grows smoothly toward 1 at the mesh scale -- the envelope has
    // spectral width, so even the 4-cell packet is not all at cos(kh) = 0.
    // None of this is the face's to absorb: it is the sink's specification.
    check(left[0] < 0.1, "an 8-cell packet leaves through the face (~2% stays)");
    check(left[1] > 5.0 * left[0],
          "a 4-cell packet strands an order more energy than an 8-cell one");
    check(left[2] > 0.5, "a 2.7-cell packet mostly stays");
    check(left[3] > 0.9, "a 2-cell packet nearly all stays (group velocity < 0)");
    // Vortical family, measured 2026-07-30: the 4-cell patch's peak has not
    // left the release point after two windows (advective group speed
    // u cos(kh) ~ 0) -- HALF the incident vorticity is simply parked. The
    // 8-cell patch drains, but through a slow tail at the face.
    check(vleft[1] > 0.3 && vlate[1] > 0.3,
          "a 4-cell vortical patch cannot advect out (residual parked)");
    check(vpk[1].i > 8 && vpk[1].i < 256 - 9,
          "...and its residual sits mid-box, not on the face rows");
    check(vlate[0] < vleft[0] + 0.02,
          "the 8-cell patch residual is draining or flat, not growing");
    // Operator invariants. The nested-difference flux is EXACTLY zero on a
    // constant state -- bitwise, not approximately -- and roundoff-only on
    // linear ramps (the extrapolated ghost closure is what keeps that true
    // through the extremes). The flux form telescopes, so per field the rhs
    // sums to zero up to per-cell rounding.
    check(rcMax == 0.0, "constant state: rhs is 0.0 to the bit");
    check(rlMax < 1e-4, "linear conserved ramps: rhs is FP32 roundoff only");
    check(consWorst < 1e-4, "flux form conserves every field to roundoff");
    // The band does what the profile says, quantitatively: a 2dx mode decays
    // at its cell's own sigma (sin^4 = 1), scheme transport is nil, so the
    // analytic prediction must land within the RK3-vs-exp and envelope-drift
    // error. alpha = 2 is the validator cap: still stable, still on theory.
    check(std::abs(d1.measured - d1.predicted) < 0.2 * d1.predicted,
          "2dx decay matches the band profile within 20% (alpha 1)");
    check(std::abs(d2.measured - d2.predicted) < 0.2 * d2.predicted &&
              d2.measured < 1.0,
          "...and at the alpha = 2 stability cap");
    // Selectivity: resolved content must not notice the band. 16dx pays under
    // 2% of its energy crossing it; the mid-box 4dx figure is UNCHANGED (the
    // band is a boundary treatment, not a global filter -- content stranded
    // mid-box is out of its reach and stays the M0 figure); in the band, 4dx
    // dies.
    check(std::abs(on16 - off16) < 0.02,
          "a 16dx packet crosses the band losing < 2%");
    check(std::abs(onMid4 - left[1]) < 0.03,
          "the band does not touch a mid-box 4dx packet (locality)");
    check(onBand4 < 0.1 && onBand4 < 0.5 * offBand4,
          "a 4dx packet parked in the band is absorbed");
    return dnstest::finish();
}
