// Compressible cylinder tunnel (2D, or 3D with --nz): streamwise freestream
// sponge + free-slip lateral walls (DeviceCompFar, the validated far field;
// --open swaps the walls for lateral sponge bands = completely open), and a
// sharp ghost-fill wall on the exact circle (mirror no-slip, adiabatic).
// --ramp T > 0 (default) starts from rest and raises the freestream by exact
// Galilean boosts -- no start-up acoustic pulse.
// Kennedy-Gruber central fluxes: smooth (shock-free) Mach only.
#include "dns/comp.hpp"
#include "dns/output.hpp"
#include "dns/gpu/runner_comp.hpp"
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
    const double nsBeta = arg(argc, argv, "--nsbeta", -1.0); // -1 = auto (Mach)
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

    // Sharp ghost-fill wall on the exact circle (z-invariant when nz > 1).
    auto sdf = [&](double x, double y, double) {
        const double dx = x - cxx, dy = y - cyy;
        return std::sqrt(dx * dx + dy * dy) - R;
    };

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
    const double tEnd = tRamp + flowThroughs * L / uInf;
    const double qRef = 0.5 * rhoInf * uInf * uInf * D;

    // Every run artefact goes through the one sink.
    RunMeta rm;
    rm.nx = n; rm.ny = ny; rm.nz = nz;
    rm.hx = h; rm.hy = h; rm.hz = g.hz;
    rm.D = D; rm.re = reD; rm.mach = mach; rm.frameDt = frameDt;
    rm.body = "cylinder";
    rm.farField = nscbc ? (open ? "nscbc-open" : "nscbc") : (open ? "sponge-open" : "sponge");
    rm.physics = "compressible";
    rm.kick = kick;

    const auto gg = gpu::GpuGrid::from(g);
    const int kc = nz / 2;
    FrameGeometry fg;
    fg.mask.assign(g.size(), 1.0f);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < n; ++i)
                fg.mask[g.idx(i, j, k)] = mask[std::size_t(j) * n + i];
    fg.ic = n / 2; fg.jc = ny / 2; fg.kc = kc; fg.D = D;

    RunOutput::Config oc;
    oc.prefix = prefix;
    oc.kinds = {Emit::Stats, Emit::Bin};
    if (nz == 1) { oc.kinds.insert(Emit::Png); oc.kinds.insert(Emit::Web); }
    oc.fdt = frameDt;
    oc.pdt = pngDt;
    oc.wdt = pngDt;
    oc.simJson = "{\"app\":\"comp_cylinder\",\"re\":" + std::to_string(reD)
               + ",\"mach\":" + std::to_string(mach)
               + ",\"nscbc\":" + std::to_string(nscbc ? 1 : 0);
    auto out = RunOutput::make(oc, rm, g, fg);
    if (!out) {
        std::fprintf(stderr, "output init failed: %s\n", out.error().c_str());
        return 4;
    }
    // The runner declares the stats series.

    gpu::CompRun rr;
    rr.gamma = gamma;
    rr.mu = mu;
    rr.uInf = uInf;
    rr.rhoInf = rhoInf;
    rr.pInf = pInf;
    rr.mach = mach;
    rr.D = D;
    rr.qRef = qRef;
    rr.cfl = cfl;
    rr.tEnd = tEnd;
    rr.far = nscbc ? FarField::Nscbc : FarField::Sponge;
    rr.open = open;
    rr.nscbcBeta = nsBeta;
    rr.nscbcStrip = nsStrip;
    rr.ramp = tRamp;

    auto res = gpu::runCompressible(g, host, sdf, nullptr, rr, *out);
    if (!res) {
        std::fprintf(stderr, "%s\n", res.error().message.c_str());
        return res.error().code;
    }
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", res->steps,
                out->frames(), res->msPerStep);
    return 0;
}
