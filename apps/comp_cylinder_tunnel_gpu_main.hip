// Compressible cylinder tunnel (2D, or 3D with --nz): streamwise freestream
// sponge + free-slip lateral walls (DeviceCompFar, the validated far field;
// --open swaps the walls for lateral sponge bands = completely open), and a
// sharp ghost-fill wall on the exact circle (mirror no-slip, adiabatic).
// --ramp T > 0 (default) starts from rest and raises the freestream by exact
// Galilean boosts -- no start-up acoustic pulse.
// Kennedy-Gruber central fluxes: smooth (shock-free) Mach only.
#include "dns/comp.hpp"
#include "dns/fieldpng.hpp"
#include "dns/webframes.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_far.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/device_util.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numbers>
#include <algorithm>
#include <string>
#include <vector>

using namespace dns;
using gpu::Real;
using std::numbers::pi;

namespace {

double arg(int argc, char** argv, const char* name, double dflt)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], name)) return std::atof(argv[i + 1]);
    return dflt;
}

std::string argS(int argc, char** argv, const char* name, const char* dflt)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

bool argB(int argc, char** argv, const char* name)
{
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], name)) return true;
    return false;
}

} // namespace

int main(int argc, char** argv)
{
    if (!gpu::deviceAvailable()) {
        std::fprintf(stderr, "no HIP device\n");
        return 1;
    }
    const int n = argc > 1 ? std::atoi(argv[1]) : 512;
    const int ny = int(arg(argc, argv, "--ny", n));
    const int nz = int(arg(argc, argv, "--nz", 1));
    const double reD = arg(argc, argv, "--re", 800.0);
    const double mach = arg(argc, argv, "--mach", 0.2);
    const double flowThroughs = arg(argc, argv, "--ft", 15.0);
    const double frameDt = arg(argc, argv, "--fdt", 0.3);
    // PNG cadence, decoupled from the (33 MB/frame at 2048^2) .bin cadence.
    const double pngDt = arg(argc, argv, "--pdt", frameDt);
    const double cfl = arg(argc, argv, "--cfl", 0.4);
    const bool open = argB(argc, argv, "--open");
    // NSCBC characteristic x-boundaries instead of the streamwise sponge
    // (lateral stays walled; sponge remains the validated default).
    const bool nscbc = argB(argc, argv, "--nscbc");
    // Thin absorbing strip (cells) inside the NSCBC rows: mops up residual
    // reflection that pumps box modes at high Re (production-standard hybrid).
    const int nsStrip = int(arg(argc, argv, "--nsstrip", 0.0));
    const double tRamp = arg(argc, argv, "--ramp", 2.0);
    const double kick = arg(argc, argv, "--kick", 0.05);
    const std::string prefix = argS(argc, argv, "--out", "comp_cyl");

    // Resolution scales with ny (domain height fixed at 8 D): D = 64 cells
    // at ny = 512, 256 at ny = 2048. nx stretches the tunnel relative to ny.
    const double D = pi / 4.0, R = D / 2.0, h = D / 64.0 * (512.0 / ny);
    const double L = n * h, Ly = ny * h;
    // Centre placed like the incompressible 512^2 run (slight y offset trips
    // the shedding instability).
    const double cxx = 2.1107, cyy = Ly / 2.0 + 0.0146;
    const double uInf = 1.0, rhoInf = 1.0, gamma = 1.4;
    const double pInf = rhoInf * uInf * uInf / (gamma * mach * mach);
    const double mu = rhoInf * uInf * D / reD;

    const auto g = Grid::make(n, ny, nz, L, Ly, nz > 1 ? nz * h : 1.0);
    auto dev = gpu::DeviceComp::make(g, gamma, mu);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    gpu::DeviceCompFar::Config fc;
    fc.rhoInf = rhoInf;
    fc.uInf = uInf;
    fc.pInf = pInf;
    fc.gamma = gamma;
    fc.ns = std::clamp(int(0.1 * n), 8, n / 2 - 1);
    fc.openY = open;
    fc.openZ = open;
    if (nscbc) { // x handled by NSCBC; optional thin strip absorbs residuals
        fc.sigMax = nsStrip > 0 ? 0.5 : 0.0;
        if (nsStrip > 0) fc.ns = nsStrip;
    }
    auto far = gpu::DeviceCompFar::make(g, fc);
    gpu::DeviceCompNscbc::Config nc;
    nc.rhoInf = rhoInf;
    nc.uInf = uInf;
    nc.pInf = pInf;
    nc.gamma = gamma;
    nc.betaTrans = arg(argc, argv, "--nsbeta", -1.0); // -1 = auto (Mach)
    // --nscbc --open: lateral extremes become characteristic far-field faces
    // (DeviceCompFar then applies no walls; sponge only if --nsstrip).
    nc.openY = open;
    nc.openZ = open && nz > 1;
    auto nsc = nscbc ? gpu::DeviceCompNscbc::make(g, nc)
                     : std::expected<gpu::DeviceCompNscbc, hipError_t>(
                           std::unexpected(hipErrorInvalidValue));
    if (!dev || !ds || !red || !far || (nscbc && !nsc)) {
        std::fprintf(stderr, "GPU alloc failed\n");
        return 1;
    }

    // Sharp ghost-fill wall on the exact circle (z-invariant when nz > 1).
    auto sdf = [&](double x, double y, double) {
        const double dx = x - cxx, dy = y - cyy;
        return std::sqrt(dx * dx + dy * dy) - R;
    };
    auto ghost = gpu::DeviceCompGhost::make(g, sdf, gamma, rhoInf, pInf);
    if (!ghost) {
        std::fprintf(stderr, "ghost wall init failed\n");
        return 1;
    }
    auto applyBc = [&](gpu::DeviceCompState& st) {
        far->apply(st);
        ghost->apply(st);
    };
    dev->setStageHook(applyBc);
    if (nscbc)
        dev->setRhsHook([&](const gpu::DeviceCompState& st,
                            gpu::DeviceCompState& k) { nsc->apply(*dev, st, k); });

    // IC. Quiet start (tRamp > 0): rest, exactly consistent with the no-slip
    // body -- the freestream is then raised by silent Galilean boosts.
    // tRamp == 0: impulsive freestream, velocity tapered to rest across ~3
    // cells outside the surface.
    const double u0 = tRamp > 0 ? 0.0 : uInf;
    // Deterministic solenoidal trip (quiet start only): a weak Gaussian
    // vortex one D behind the body seeds the wake instability -- a quiet
    // start alone is too symmetric to shed. Divergence-free, so silent.
    const double kA = tRamp > 0 ? kick * uInf : 0.0, kw = 0.25 * D;
    const double kx = cxx + D, ky = cyy + 0.25 * D;
    auto trip = [&](double x, double y, double& u, double& v) {
        const double dx = x - kx, dy = y - ky;
        const double e = kA * std::exp(-(dx * dx + dy * dy) / (kw * kw));
        u += 2 * dy / kw * e;
        v -= 2 * dx / kw * e;
    };
    auto host = CompState::zeros(g);
    std::vector<float> mask(std::size_t(n) * ny, 1.0f);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double x = (i + 0.5) * h, y = (j + 0.5) * h;
                const double d = sdf(x, y, 0.0);
                const double f = std::clamp(d / (3.0 * h), 0.0, 1.0);
                double u = u0 * f * f * (3 - 2 * f), v = 0.0;
                if (kA != 0.0 && d > 0.0) trip(x, y, u, v);
                host.rho[c] = rhoInf;
                host.mx[c] = rhoInf * u;
                host.my[c] = rhoInf * v;
                host.E[c] = pInf / (gamma - 1.0) + 0.5 * rhoInf * (u * u + v * v);
                if (k == 0 && d < 0.0) mask[std::size_t(j) * n + i] = 0.0f;
            }
    (void)ds->upload(host);
    far->setFreestream(u0);
    if (nscbc) nsc->setFreestream(u0);
    applyBc(*ds);

    // Quiet-start freestream schedule: smoothstep 0 -> 1 over [0, tRamp].
    auto rampFrac = [&](double tt) {
        if (tRamp <= 0 || tt >= tRamp) return 1.0;
        const double s = tt / tRamp;
        return s * s * (3 - 2 * s);
    };

    const double tEnd = tRamp + flowThroughs * L / uInf;
    const double qRef = 0.5 * rhoInf * uInf * uInf * D;

    std::ofstream stats(prefix + "_stats.csv");
    stats << "t,dt,cd,cl,wmax\n";
    std::ofstream(prefix + ".json")
        << "{\"nx\":" << n << ",\"ny\":" << ny << ",\"frameDt\":" << frameDt
        << ",\"D\":" << D << ",\"reD\":" << reD << ",\"mach\":" << mach
        << ",\"nz\":" << nz << ",\"open\":" << (open ? 1 : 0)
        << ",\"nscbc\":" << (nscbc ? 1 : 0) << ",\"ramp\":" << tRamp
        << ",\"cx\":" << cxx / h << ",\"cy\":" << cyy / h << ",\"R\":" << R / h
        << "}\n";
    {
        std::ofstream mf(prefix + "_mask.bin", std::ios::binary);
        mf.write(reinterpret_cast<const char*>(mask.data()),
                 std::streamsize(mask.size() * sizeof(float)));
    }
    std::ofstream vortFile(prefix + "_vort.bin", std::ios::binary);
    std::ofstream speedFile(prefix + "_speed.bin", std::ios::binary);

    // Live PNGs of the 2D fields (vorticity diverging, |u| viridis).
    const std::string pngDir = "png_" + prefix;
    const bool pngs = nz == 1;
    if (pngs) std::filesystem::create_directories(pngDir);
    const float vortScale = float(8.0 * uInf / D), speedMax = float(1.6 * uInf);
    // Live browser frames (docs/viewer-format.md), same cadence as the PNGs.
    std::unique_ptr<dns::WebFrames> web;
    if (pngs) {
        std::string sim = "{\"app\":\"comp_cylinder\",\"re\":" +
                          std::to_string(reD) + ",\"mach\":" +
                          std::to_string(mach) + ",\"nscbc\":" +
                          std::to_string(nscbc ? 1 : 0) + "}";
        web = std::make_unique<dns::WebFrames>("web_" + prefix, n, ny, sim);
        web->addField("vort", "diverging", -vortScale, vortScale);
        web->addField("speed", "sequential", 0.0, speedMax);
        web->writeStatic("mask", mask.data(), 0.0, 1.0);
    }

    const int kc = nz / 2;
    std::vector<Real> hr(g.size()), hmx(g.size()), hmy(g.size());
    std::vector<float> frame(std::size_t(n) * ny);
    std::vector<double> uu(g.size()), vv(g.size());
    int frames = 0, pngIdx = 0;
    std::vector<float> vortKeep(web ? std::size_t(n) * ny : 0);
    auto writeFrame = [&](bool bins, bool png, double tNow) {
        (void)ds->rho.download(hr.data());
        (void)ds->mx.download(hmx.data());
        (void)ds->my.download(hmy.data());
        for (std::size_t c = 0; c < g.size(); ++c) {
            uu[c] = double(hmx[c]) / hr[c];
            vv[c] = double(hmy[c]) / hr[c];
        }
        auto at = [&](const std::vector<double>& q, int i, int j) {
            return q[g.idx(i, j, kc)];
        };
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < n; ++i) {
                const double wz = (at(vv, i + 1, j) - at(vv, i - 1, j)) / (2 * h) -
                                  (at(uu, i, j + 1) - at(uu, i, j - 1)) / (2 * h);
                frame[std::size_t(j) * n + i] = float(wz);
            }
        if (bins)
            vortFile.write(reinterpret_cast<const char*>(frame.data()),
                           std::streamsize(frame.size() * sizeof(float)));
        char pth[4096];
        if (png && pngs) {
            std::snprintf(pth, sizeof pth, "%s/vort_%05d.png", pngDir.c_str(),
                          pngIdx);
            dns::writeDivergingPng(pth, n, ny, frame.data(), vortScale,
                                   mask.data());
            if (web) vortKeep = frame; // held until speed is computed
        }
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t c = g.idx(i, j, kc);
                frame[std::size_t(j) * n + i] =
                    float(std::sqrt(uu[c] * uu[c] + vv[c] * vv[c]));
            }
        if (bins)
            speedFile.write(reinterpret_cast<const char*>(frame.data()),
                            std::streamsize(frame.size() * sizeof(float)));
        if (png && pngs) {
            std::snprintf(pth, sizeof pth, "%s/speed_%05d.png", pngDir.c_str(),
                          pngIdx);
            dns::writeSequentialPng(pth, n, ny, frame.data(), 0.0f, speedMax,
                                    mask.data());
            if (web) web->writeFrame(tNow, {vortKeep.data(), frame.data()});
        }
        if (bins) ++frames;
        if (png) ++pngIdx;
    };

    double t = 0.0;
    long steps = 0, lastLog = 0;
    double nextFrame = 0.0, nextPng = 0.0;
    const auto wall0 = std::chrono::steady_clock::now();

    // stableDt costs two full-grid reductions + host syncs; the acoustic CFL
    // moves slowly, so recompute every 16 steps and take a 5% margin.
    double dtStab = 0.0;
    int sinceDt = 1 << 30;
    while (t < tEnd) {
        if (++sinceDt >= 16) {
            dtStab = 0.95 * dev->stableDt(*ds, *red, cfl);
            sinceDt = 0;
        }
        const double dt = std::min(tEnd - t, dtStab);
        if (dtStab < 1e-6) {
            std::fprintf(stderr, "dt collapsed at t=%.3f -- blowup\n", t);
            return 1;
        }
        dev->step(*ds, Real(dt));
        if (t < tRamp) { // quiet start: boost the frame, retarget the sponge
            const double uNew = uInf * rampFrac(t + dt);
            far->boost(*ds, uNew - far->freestream());
            far->setFreestream(uNew);
            if (nscbc) nsc->setFreestream(uNew);
        }
        applyBc(*ds); // leave the state BC/ghost-consistent between steps
        t += dt;
        ++steps;

        if (steps - lastLog >= 200 || t >= tEnd) {
            const auto F = ghost->force(*dev, *ds);
            const double span = nz > 1 ? nz * g.hz : 1.0;
            const double cd = F.fx / (qRef * span);
            const double cl = F.fy / (qRef * span);
            const double wmax = cfl / dtStab * std::min(g.hx, g.hy);
            stats << t << ',' << dt << ',' << cd << ',' << cl << ',' << wmax << '\n';
            if (steps - lastLog >= 4000 || t >= tEnd)
                std::printf("t=%.2f/%.2f cd=%.3f cl=%+.3f wmax=%.2f dt=%.2e\n", t,
                            tEnd, cd, cl, wmax, dt);
            lastLog = steps;
        }
        if (t >= nextFrame || t >= nextPng) {
            const bool bins = t >= nextFrame, png = t >= nextPng;
            writeFrame(bins, png, t);
            if (bins) nextFrame += frameDt;
            if (png) nextPng += pngDt;
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - wall0)
                          .count() /
                      double(steps);
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", steps, frames, ms);
    std::ofstream(prefix + ".json")
        << "{\"nx\":" << n << ",\"ny\":" << ny << ",\"frames\":" << frames
        << ",\"frameDt\":" << frameDt << ",\"D\":" << D << ",\"reD\":" << reD
        << ",\"mach\":" << mach << ",\"cx\":" << cxx / h << ",\"cy\":" << cyy / h
        << ",\"R\":" << R / h << ",\"msPerStep\":" << ms << "}\n";
    return 0;
}
