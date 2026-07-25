// DeviceCompFar: uniform freestream is a fixed point (walled and open), the
// Galilean boost round-trips exactly, and the quiet start (rest + ramped
// boosts) radiates far less start-up acoustics than an impulsive freestream.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_far.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;



constexpr double gam = 1.4, rhoInf = 1.0, uInf = 1.0;
constexpr double mach = 0.2;
const double pInf = rhoInf * uInf * uInf / (gam * mach * mach);
const double EInf = pInf / (gam - 1.0) + 0.5 * rhoInf * uInf * uInf;

gpu::DeviceCompFar::Config config(int nx, bool open)
{
    gpu::DeviceCompFar::Config c;
    c.rhoInf = rhoInf;
    c.uInf = uInf;
    c.pInf = pInf;
    c.gamma = gam;
    c.ns = std::clamp(int(0.1 * nx), 8, nx / 2 - 1);
    c.openY = open;
    c.openZ = open;
    return c;
}

// Max deviation from freestream after nSteps in an empty domain.
double freestreamDrift(int nx, int ny, int nz, bool open)
{
    const auto g = Grid::make(nx, ny, nz, nx * 0.01, ny * 0.01,
                              nz > 1 ? nz * 0.01 : 1.0);
    auto dev = gpu::DeviceComp::make(g, gam, 1e-3);
    auto far = gpu::DeviceCompFar::make(g, config(nx, open));
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !far || !ds || !red) std::exit(EXIT_FAILURE);
    dev->setStageHook([&](gpu::DeviceCompState& st) { far->apply(st); });

    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.mx[c] = rhoInf * uInf;
        host.E[c] = EInf;
    }
    (void)ds->upload(host);
    for (int it = 0; it < 100; ++it)
        dev->step(*ds, gpu::Real(dev->stableDt(*ds, *red)));
    (void)ds->download(host);
    double drift = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c) {
        drift = std::max(drift, std::abs(host.rho[c] - rhoInf));
        drift = std::max(drift, std::abs(host.mx[c] - rhoInf * uInf));
        drift = std::max(drift, std::abs(host.my[c]));
        drift = std::max(drift, std::abs(host.mz[c]));
        drift = std::max(drift, std::abs(host.E[c] - EInf) / EInf);
    }
    return drift;
}

// Boost by +du then -du must round-trip to FP32 rounding.
double boostRoundTrip()
{
    const int nx = 64, ny = 32;
    const auto g = Grid::make(nx, ny, 1, 2 * pi, pi, 1.0);
    auto far = gpu::DeviceCompFar::make(g, config(nx, false));
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!far || !ds) std::exit(EXIT_FAILURE);
    auto h0 = CompState::zeros(g);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
            const std::size_t c = g.idx(i, j, 0);
            h0.rho[c] = 1.0 + 0.05 * std::sin(x);
            h0.mx[c] = 0.3 * std::cos(x) * std::sin(2 * y);
            h0.my[c] = 0.2 * std::sin(x + y);
            h0.E[c] = pInf / (gam - 1.0) + 0.1 * std::cos(y);
        }
    (void)ds->upload(h0);
    far->boost(*ds, 0.7);
    far->boost(*ds, -0.7);
    auto h1 = CompState::zeros(g);
    (void)ds->download(h1);
    double err = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c) {
        err = std::max(err, std::abs(h1.rho[c] - h0.rho[c]));
        err = std::max(err, std::abs(h1.mx[c] - h0.mx[c]));
        err = std::max(err, std::abs(h1.E[c] - h0.E[c]) / EInf);
    }
    return err;
}

// Cylinder start-up in a walled tunnel: max far-field pressure disturbance up
// to tProbe (outside 3R of the body and the sponge bands). quiet: rest IC +
// smoothstep boosts over tRamp; impulsive: freestream IC with a 3h taper.
double startupNoise(bool quiet)
{
    const int nx = 192, ny = 96;
    const auto g = Grid::make(nx, ny, 1, 2 * pi, pi, 1.0);
    const double h = g.hx, R = 6 * h;
    const double cx = pi, cy = pi / 2 + 0.013;
    auto sdf = [&](double x, double y, double) {
        return std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) - R;
    };
    auto dev = gpu::DeviceComp::make(g, gam, rhoInf * uInf * (2 * R) / 100.0);
    auto far = gpu::DeviceCompFar::make(g, config(nx, false));
    auto ghost = gpu::DeviceCompGhost::make(g, sdf, gam, rhoInf, pInf);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !far || !ghost || !ds || !red) std::exit(EXIT_FAILURE);
    auto applyBc = [&](gpu::DeviceCompState& st) {
        far->apply(st);
        ghost->apply(st);
    };
    dev->setStageHook(applyBc);

    const double tRamp = quiet ? 2.0 : 0.0;
    const double u0 = quiet ? 0.0 : uInf;
    auto host = CompState::zeros(g);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double d = sdf((i + 0.5) * h, (j + 0.5) * g.hy, 0.0);
            const double f = std::clamp(d / (3.0 * h), 0.0, 1.0);
            const double u = u0 * f * f * (3 - 2 * f);
            host.rho[c] = rhoInf;
            host.mx[c] = rhoInf * u;
            host.E[c] = pInf / (gam - 1.0) + 0.5 * rhoInf * u * u;
        }
    (void)ds->upload(host);
    far->setFreestream(u0);
    applyBc(*ds);

    const int ns = config(nx, false).ns;
    const double tProbe = 1.0;
    double t = 0.0, noise = 0.0;
    std::vector<gpu::Real> hp(g.size());
    int it = 0;
    while (t < tProbe) {
        const double dt = std::min(tProbe - t, dev->stableDt(*ds, *red));
        dev->step(*ds, gpu::Real(dt));
        if (t < tRamp) {
            const double s = std::min(1.0, (t + dt) / tRamp);
            const double uNew = uInf * s * s * (3 - 2 * s);
            far->boost(*ds, uNew - far->freestream());
            far->setFreestream(uNew);
        }
        applyBc(*ds);
        t += dt;
        if (++it % 5 != 0) continue;
        dev->primitives(*ds);
        (void)dev->p().download(hp.data());
        for (int j = 0; j < ny; ++j)
            for (int i = ns; i < nx - ns; ++i) {
                const double x = (i + 0.5) * h, y = (j + 0.5) * g.hy;
                if (sdf(x, y, 0.0) < 3 * R) continue;
                noise = std::max(noise,
                                 std::abs(double(hp[g.idx(i, j, 0)]) - pInf));
            }
    }
    return noise / pInf;
}

// No-slip mirror walls: after one apply, the ghost rows carry the FULL
// antimirror (every momentum component negated, rho/E copied), so all
// velocity components vanish at the wall faces by symmetry; and a short
// stepped run decelerates the near-wall flow (a boundary layer starts).
void noslipWall()
{
    const int nx = 64, ny = 32;
    const auto g = Grid::make(nx, ny, 1, 2 * pi, pi, 1.0);
    auto cfg = config(nx, false);
    cfg.noslipY = true;
    auto far = gpu::DeviceCompFar::make(g, cfg);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!far || !ds) std::exit(EXIT_FAILURE);
    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.mx[c] = rhoInf * uInf;
        host.my[c] = 0.1;
        host.E[c] = EInf;
    }
    (void)ds->upload(host);
    far->apply(*ds);
    auto h1 = CompState::zeros(g);
    (void)ds->download(h1);
    double err = 0.0;
    for (int i = 0; i < nx; ++i)
        for (const auto [jg, js] : {std::pair{0, 1}, {ny - 1, ny - 2}}) {
            const std::size_t cg = g.idx(i, jg, 0), cs = g.idx(i, js, 0);
            err = std::max({err, std::abs(h1.mx[cg] + h1.mx[cs]),
                            std::abs(h1.my[cg] + h1.my[cs]),
                            std::abs(h1.rho[cg] - h1.rho[cs]),
                            std::abs(h1.E[cg] - h1.E[cs]) / EInf});
        }
    std::printf("no-slip mirror max defect: %.2e\n", err);
    check(err < 1e-12, "no-slip walls antimirror all momentum");

    auto dev = gpu::DeviceComp::make(g, gam, 5e-3);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !red) std::exit(EXIT_FAILURE);
    dev->setStageHook([&](gpu::DeviceCompState& st) { far->apply(st); });
    for (int it = 0; it < 200; ++it)
        dev->step(*ds, gpu::Real(dev->stableDt(*ds, *red)));
    (void)ds->download(h1);
    double uWall = 0.0, uMid = 0.0;
    for (int i = 0; i < nx; ++i) {
        uWall += h1.mx[g.idx(i, 1, 0)] / h1.rho[g.idx(i, 1, 0)];
        uMid += h1.mx[g.idx(i, ny / 2, 0)] / h1.rho[g.idx(i, ny / 2, 0)];
    }
    uWall /= nx;
    uMid /= nx;
    std::printf("after 200 steps: <u> wall row %.4f, mid %.4f\n", uWall, uMid);
    check(std::isfinite(uWall) && uWall < 0.9 * uMid,
          "boundary layer decelerates the wall row");
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    const double dWall = freestreamDrift(96, 48, 1, false);
    std::printf("freestream drift walled 2D: %.2e\n", dWall);
    check(dWall < 5e-6, "walled: uniform freestream fixed to FP32 rounding");

    const double dOpen2 = freestreamDrift(96, 96, 1, true);
    std::printf("freestream drift open 2D:   %.2e\n", dOpen2);
    check(dOpen2 < 5e-6, "open 2D: uniform freestream fixed point");

    const double dOpen3 = freestreamDrift(48, 48, 48, true);
    std::printf("freestream drift open 3D:   %.2e\n", dOpen3);
    check(dOpen3 < 5e-6, "open 3D: uniform freestream fixed point");

    const double rt = boostRoundTrip();
    std::printf("boost round-trip err: %.2e\n", rt);
    check(rt < 1e-5, "Galilean boost round-trips to FP32 rounding");

    const double loud = startupNoise(false);
    const double still = startupNoise(true);
    std::printf("startup noise |p'|/pInf: impulsive %.3e quiet %.3e\n", loud,
                still);
    check(still < 0.5 * loud, "quiet start radiates < half the impulsive pulse");
    check(still < 2e-2, "quiet start far field stays near pInf");

    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
