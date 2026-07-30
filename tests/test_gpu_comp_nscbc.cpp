// DeviceCompNscbc: uniform freestream is a fixed point, a right-going
// acoustic pulse exits through the outflow with small reflection (vs full
// wrap-around under pure periodic), a vortex advects out cleanly, and a
// uniform velocity offset is pulled back toward the inflow target. Open
// lateral faces (openY/openZ): freestream fixed point in 2D and true 3D, a
// radial acoustic pulse exits through all faces, and a vortex skimming the
// lateral boundary leaves without exciting the edge cells. Stretched-y
// (TanhStretchY PointRemap + curvilinear tables): freestream fixed point,
// beta = 0 consistency with the uniform path, pulse and vortex exit.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_far.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/remap.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dns;

namespace {

using dnstest::check;



constexpr double gam = 1.4, rhoInf = 1.0, uInf = 1.0, mach = 0.2;
const double pInf = rhoInf * uInf * uInf / (gam * mach * mach);
const double cInf = std::sqrt(gam * pInf / rhoInf);
const double EInf = pInf / (gam - 1.0) + 0.5 * rhoInf * uInf * uInf;

struct Rig
{
    Grid g;
    std::expected<gpu::DeviceComp, hipError_t> dev;
    std::expected<gpu::DeviceCompFar, hipError_t> far;
    std::expected<gpu::DeviceCompNscbc, hipError_t> nsc;
    std::expected<gpu::DeviceCompState, hipError_t> ds;
    std::expected<gpu::ReduceScratch, hipError_t> red;

    // NSCBC x-boundaries + free-slip y walls (sigMax = 0 far = walls only).
    Rig(int nx, int ny, double mu)
        : g(Grid::make(nx, ny, 1, nx * 0.01, ny * 0.01, 1.0)),
          dev(gpu::DeviceComp::make(g, gam, mu)),
          far(gpu::DeviceCompFar::make(
              g, {rhoInf, uInf, pInf, gam, 8, 0.0, false, false})),
          nsc(gpu::DeviceCompNscbc::make(g, {rhoInf, uInf, pInf, gam})),
          ds(gpu::DeviceCompState::make(g.size())),
          red(gpu::ReduceScratch::make())
    {
        if (!dev || !far || !nsc || !ds || !red) std::exit(EXIT_FAILURE);
        dev->setStageHook([this](gpu::DeviceCompState& st) { far->apply(st); });
        dev->setRhsHook([this](const gpu::DeviceCompState& st,
                               gpu::DeviceCompState& out,
                               const gpu::StageCombine& cmb) {
            nsc->apply(*dev, st, out, cmb);
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
};

void setState(Rig& r, auto prim) // prim(x, y) -> {rho, u, v, p}
{
    auto h = CompState::zeros(r.g);
    for (int j = 0; j < r.g.ny; ++j)
        for (int i = 0; i < r.g.nx; ++i) {
            const std::size_t c = r.g.idx(i, j, 0);
            const auto [rho, u, v, p] =
                prim((i + 0.5) * r.g.hx, (j + 0.5) * r.g.hy);
            h.rho[c] = rho;
            h.mx[c] = rho * u;
            h.my[c] = rho * v;
            h.E[c] = p / (gam - 1.0) + 0.5 * rho * (u * u + v * v);
        }
    (void)r.ds->upload(h);
}

double freestreamDrift()
{
    Rig r(96, 48, 1e-3);
    setState(r, [](double, double) {
        return std::array{rhoInf, uInf, 0.0, pInf};
    });
    for (int it = 0; it < 100; ++it)
        r.dev->step(*r.ds, gpu::Real(r.dev->stableDt(*r.ds, *r.red)));
    auto h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    double d = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        d = std::max(d, std::abs(h.rho[c] - rhoInf));
        d = std::max(d, std::abs(h.mx[c] - rhoInf * uInf));
        d = std::max(d, std::abs(h.my[c]));
        d = std::max(d, std::abs(h.E[c] - EInf) / EInf);
    }
    return d;
}

// Right-going linear acoustic pulse released mid-domain; after it has left,
// the residual |p - pInf| is the reflected amplitude. amp = pulse |p'|/pInf.
double pulseReflection(double amp)
{
    Rig r(256, 8, 0.0);
    const double Lx = r.g.nx * r.g.hx, x0 = 0.5 * Lx, sig = 8 * r.g.hx;
    setState(r, [&](double x, double) {
        const double pp = amp * pInf * std::exp(-(x - x0) * (x - x0) / (sig * sig));
        return std::array{rhoInf + pp / (cInf * cInf), uInf + pp / (rhoInf * cInf),
                          0.0, pInf + pp};
    });
    // Pulse head reaches the outflow in (Lx/2)/(u+c); run 2.5x that so the
    // tail is out while the (slower) reflected wave is still inside.
    r.run(2.5 * 0.5 * Lx / (uInf + cInf));
    r.dev->primitives(*r.ds);
    std::vector<gpu::Real> hp(r.g.size());
    (void)r.dev->p().download(hp.data());
    double resid = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c)
        resid = std::max(resid, std::abs(double(hp[c]) - pInf));
    return resid / (amp * pInf);
}

// Gaussian vortex advecting through the outflow; residual max |v| after exit
// relative to the initial vortex peak velocity.
double vortexResidual()
{
    Rig r(256, 64, 1e-4);
    const double Lx = r.g.nx * r.g.hx, Ly = r.g.ny * r.g.hy;
    const double x0 = 0.75 * Lx, y0 = 0.5 * Ly, kw = 8 * r.g.hx, A = 0.1 * uInf;
    double v0 = 0.0;
    setState(r, [&](double x, double y) {
        const double dx = x - x0, dy = y - y0;
        const double e = A * std::exp(-(dx * dx + dy * dy) / (kw * kw));
        const double u = uInf + 2 * dy / kw * e, v = -2 * dx / kw * e;
        v0 = std::max(v0, std::abs(v));
        return std::array{rhoInf, u, v, pInf};
    });
    r.run((Lx - x0 + 6 * kw) / uInf + 4 * kw / cInf);
    auto h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    double resid = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        if (!std::isfinite(h.my[c])) return 1e9;
        resid = std::max(resid, std::abs(h.my[c] / h.rho[c]));
    }
    return resid / v0;
}

// Uniform +5% velocity offset: the inflow relaxation must pull u back toward
// uInf (returns |<u at inflow row> - uInf| / offset after a while).
double inflowPull()
{
    Rig r(128, 16, 0.0);
    const double du = 0.05 * uInf;
    setState(r, [&](double, double) {
        return std::array{rhoInf, uInf + du, 0.0, pInf};
    });
    r.run(2.0 * r.g.nx * r.g.hx / cInf);
    auto h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    double um = 0.0;
    for (int j = 0; j < r.g.ny; ++j) {
        const std::size_t c = r.g.idx(0, j, 0);
        um += h.mx[c] / h.rho[c];
    }
    um /= r.g.ny;
    return std::abs(um - uInf) / du;
}

// Fully open rig: NSCBC on all faces (openY, openZ at nz > 1), no walls, no
// sponge. uT/etaIn configurable so the acoustic tests can soften the inflow.
struct OpenRig
{
    Grid g;
    std::expected<gpu::DeviceComp, hipError_t> dev;
    std::expected<gpu::DeviceCompNscbc, hipError_t> nsc;
    std::expected<gpu::DeviceCompState, hipError_t> ds;
    std::expected<gpu::ReduceScratch, hipError_t> red;

    // openZ < 0 = default (nz > 1); pass explicit 0/1 to gate a single lateral
    // axis (openZOnly* tests need openY = false, openZ = true). etaLat is the
    // incoming-relax rate of the lateral shear/entropy blend (Config default
    // 0.25); exposed for the transparency scan.
    OpenRig(int nx, int ny, int nz, double mu, double uT, double etaIn,
            bool openY = true, int openZ = -1, double etaLat = 0.25)
        : g(Grid::make(nx, ny, nz, nx * 0.01, ny * 0.01,
                       nz > 1 ? nz * 0.01 : 1.0)),
          dev(gpu::DeviceComp::make(g, gam, mu)),
          nsc(gpu::DeviceCompNscbc::make(g, [&] {
              gpu::DeviceCompNscbc::Config c{rhoInf, uT, pInf, gam};
              c.etaIn = etaIn;
              c.openY = openY;
              c.openZ = openZ < 0 ? (nz > 1) : (openZ != 0);
              c.etaLat = etaLat;
              return c;
          }())),
          ds(gpu::DeviceCompState::make(g.size())),
          red(gpu::ReduceScratch::make())
    {
        if (!dev || !nsc || !ds || !red) std::exit(EXIT_FAILURE);
        dev->setRhsHook([this](const gpu::DeviceCompState& st,
                               gpu::DeviceCompState& out,
                               const gpu::StageCombine& cmb) {
            nsc->apply(*dev, st, out, cmb);
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

    void set(auto prim) // prim(x, y, z) -> {rho, u, v, w, p}
    {
        auto h = CompState::zeros(g);
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    const std::size_t c = g.idx(i, j, k);
                    const auto [rho, u, v, w, p] =
                        prim((i + 0.5) * g.hx, (j + 0.5) * g.hy,
                             (k + 0.5) * g.hz);
                    h.rho[c] = rho;
                    h.mx[c] = rho * u;
                    h.my[c] = rho * v;
                    h.mz[c] = rho * w;
                    h.E[c] = p / (gam - 1.0) +
                             0.5 * rho * (u * u + v * v + w * w);
                }
        (void)ds->upload(h);
    }
};

double openFreestreamDrift(int nz, int steps)
{
    OpenRig r(48, 48, nz, 1e-3, uInf, 100.0);
    r.set([](double, double, double) {
        return std::array{rhoInf, uInf, 0.0, 0.0, pInf};
    });
    for (int it = 0; it < steps; ++it)
        r.dev->step(*r.ds, gpu::Real(r.dev->stableDt(*r.ds, *r.red)));
    auto h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    double d = 0.0;
    const double EI = pInf / (gam - 1.0) + 0.5 * rhoInf * uInf * uInf;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        d = std::max(d, std::abs(h.rho[c] - rhoInf));
        d = std::max(d, std::abs(h.mx[c] - rhoInf * uInf));
        d = std::max(d, std::abs(h.my[c]));
        d = std::max(d, std::abs(h.mz[c]));
        d = std::max(d, std::abs(h.E[c] - EI) / EI);
    }
    return d;
}

// Radial acoustic pulse in a quiescent box, all faces characteristic (soft
// inflow eta so x-lo is non-reflecting too): after 1.6 half-domain crossing
// times the residual |p - pInf| is the total all-face reflection.
double radialPulseResidual()
{
    OpenRig r(128, 128, 1, 0.0, 0.0, 2.0);
    const double amp = 0.01;
    const double Lx = r.g.nx * r.g.hx, Ly = r.g.ny * r.g.hy;
    const double x0 = 0.5 * Lx, y0 = 0.5 * Ly, sig = 6 * r.g.hx;
    r.set([&](double x, double y, double) {
        const double d2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
        const double pp = amp * pInf * std::exp(-d2 / (sig * sig));
        return std::array{rhoInf + pp / (cInf * cInf), 0.0, 0.0, 0.0, pInf + pp};
    });
    r.run(1.6 * 0.5 * Lx / cInf);
    r.dev->primitives(*r.ds);
    std::vector<gpu::Real> hp(r.g.size());
    (void)r.dev->p().download(hp.data());
    double resid = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        if (!std::isfinite(double(hp[c]))) return 1e9;
        resid = std::max(resid, std::abs(double(hp[c]) - pInf));
    }
    return resid / (amp * pInf);
}

// Vortex advecting with the freestream, centered 1.5 core widths off the y-hi
// face: its induced field crosses the lateral boundary and the outflow edge
// cells; residual |v| after exit relative to the initial peak.
double lateralVortexResidual()
{
    OpenRig r(192, 96, 1, 1e-4, uInf, 100.0);
    const double Lx = r.g.nx * r.g.hx, Ly = r.g.ny * r.g.hy;
    const double x0 = 0.6 * Lx, kw = 6 * r.g.hx, y0 = Ly - 1.5 * kw;
    const double A = 0.1 * uInf;
    double v0 = 0.0;
    r.set([&](double x, double y, double) {
        const double dx = x - x0, dy = y - y0;
        const double e = A * std::exp(-(dx * dx + dy * dy) / (kw * kw));
        const double u = uInf + 2 * dy / kw * e, v = -2 * dx / kw * e;
        v0 = std::max(v0, std::abs(v));
        return std::array{rhoInf, u, v, 0.0, pInf};
    });
    r.run((Lx - x0 + 6 * kw) / uInf + 4 * kw / cInf);
    auto h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    double resid = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        if (!std::isfinite(h.my[c])) return 1e9;
        resid = std::max(resid, std::abs(h.my[c] / h.rho[c]));
    }
    return resid / v0;
}

// openY = false, openZ = true (single lateral axis): the z-face j-edge cells
// (i interior, j in {0, ny-1}) must be enumerated by the NSCBC kernel; without
// Fix A they keep the wrap-contaminated periodic rhs and freestream drifts.
double openZOnlyFreestreamDrift(int steps)
{
    OpenRig r(48, 40, 8, 1e-3, uInf, 100.0, /*openY=*/false, /*openZ=*/1);
    r.set([](double, double, double) {
        return std::array{rhoInf, uInf, 0.0, 0.0, pInf};
    });
    for (int it = 0; it < steps; ++it)
        r.dev->step(*r.ds, gpu::Real(r.dev->stableDt(*r.ds, *r.red)));
    auto h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    double d = 0.0;
    const double EI = pInf / (gam - 1.0) + 0.5 * rhoInf * uInf * uInf;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        d = std::max(d, std::abs(h.rho[c] - rhoInf));
        d = std::max(d, std::abs(h.mx[c] - rhoInf * uInf));
        d = std::max(d, std::abs(h.my[c]));
        d = std::max(d, std::abs(h.mz[c]));
        d = std::max(d, std::abs(h.E[c] - EI) / EI);
    }
    return d;
}

// z-normal Gaussian acoustic pulse in a quiescent box with only the z faces
// characteristic (y walled, x soft inflow): after ~1.6 z-crossing times the
// residual |p - pInf| exercises the newly-enumerated z-face edge lines.
double openZOnlyPulseExit(double amp)
{
    OpenRig r(8, 8, 128, 0.0, 0.0, 2.0, /*openY=*/false, /*openZ=*/1);
    const double Lz = r.g.nz * r.g.hz, z0 = 0.5 * Lz, sig = 6 * r.g.hz;
    r.set([&](double, double, double z) {
        const double pp = amp * pInf * std::exp(-(z - z0) * (z - z0) / (sig * sig));
        return std::array{rhoInf + pp / (cInf * cInf), 0.0, 0.0,
                          pp / (rhoInf * cInf), pInf + pp};
    });
    r.run(1.6 * 0.5 * Lz / cInf);
    r.dev->primitives(*r.ds);
    std::vector<gpu::Real> hp(r.g.size());
    (void)r.dev->p().download(hp.data());
    double resid = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        if (!std::isfinite(double(hp[c]))) return 1e9;
        resid = std::max(resid, std::abs(double(hp[c]) - pInf));
    }
    return resid / (amp * pInf);
}

// Vortex core ON the y-hi face, advecting in x with the freestream: its
// induced normal velocity sweeps through 0 along the face over many steps, so
// the discontinuous entropy/shear switch chattered here. Fix B blends it
// smoothly. Finite everywhere; residual |v|/v0 at end.
double lateralVortexAlongFace(double etaLat = 0.25)
{
    OpenRig r(192, 96, 1, 1e-4, uInf, 100.0, true, -1, etaLat);
    const double Lx = r.g.nx * r.g.hx, Ly = r.g.ny * r.g.hy;
    const double x0 = 0.5 * Lx, kw = 6 * r.g.hx, y0 = Ly, A = 0.1 * uInf;
    double v0 = 0.0;
    r.set([&](double x, double y, double) {
        const double dx = x - x0, dy = y - y0;
        const double e = A * std::exp(-(dx * dx + dy * dy) / (kw * kw));
        const double u = uInf + 2 * dy / kw * e, v = -2 * dx / kw * e;
        v0 = std::max(v0, std::abs(v));
        return std::array{rhoInf, u, v, 0.0, pInf};
    });
    r.run((Lx - x0 + 6 * kw) / uInf + 4 * kw / cInf);
    auto h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    double resid = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        if (!std::isfinite(h.my[c])) return 1e9;
        resid = std::max(resid, std::abs(h.my[c] / h.rho[c]));
    }
    return resid / v0;
}

// Stretched-y rig: curvilinear DeviceComp on the TanhStretchY mesh, NSCBC x
// boundaries evaluating the SAME PointRemap (physical distances in the
// one-sided and transverse terms, transverse flux divergence from the metric
// tables), free-slip y mirror walls. 2D and small by design.
std::expected<gpu::DeviceComp, hipError_t> makeStretchedDev(const Grid& g,
                                                            double beta,
                                                            double mu)
{
    auto m = CurvMetrics::build(CurvMesh::stretchedY(g, beta));
    if (!m) return std::unexpected(hipErrorInvalidValue);
    return gpu::DeviceComp::make(*m, gam, mu);
}

struct StretchRig
{
    Grid g;
    TanhStretchY remap;
    std::expected<gpu::DeviceComp, hipError_t> dev;
    std::expected<gpu::DeviceCompFar, hipError_t> far;
    std::expected<gpu::DeviceCompNscbc, hipError_t> nsc;
    std::expected<gpu::DeviceCompState, hipError_t> ds;
    std::expected<gpu::ReduceScratch, hipError_t> red;

    StretchRig(int nx, int ny, double beta, double mu)
        : g(Grid::make(nx, ny, 1, nx * 0.01, ny * 0.01, 1.0)),
          remap{ny * 0.01, beta},
          dev(makeStretchedDev(g, beta, mu)),
          far(gpu::DeviceCompFar::make(
              g, {rhoInf, uInf, pInf, gam, 8, 0.0, false, false})),
          nsc(gpu::DeviceCompNscbc::make(g, {rhoInf, uInf, pInf, gam})),
          ds(gpu::DeviceCompState::make(g.size())),
          red(gpu::ReduceScratch::make())
    {
        if (!dev || !far || !nsc || !ds || !red) std::exit(EXIT_FAILURE);
        dev->setStageHook([this](gpu::DeviceCompState& st) { far->apply(st); });
        dev->setRhsHook([this](const gpu::DeviceCompState& st,
                               gpu::DeviceCompState& out,
                               const gpu::StageCombine& cmb) {
            nsc->apply(*dev, st, out, cmb, remap);
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

    void set(auto prim) // prim(xPhys, yPhys) -> {rho, u, v, p}
    {
        auto h = CompState::zeros(g);
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const auto [rho, u, v, p] =
                    prim((i + 0.5) * g.hx, remap.mapY((j + 0.5) * g.hy));
                h.rho[c] = rho;
                h.mx[c] = rho * u;
                h.my[c] = rho * v;
                h.E[c] = p / (gam - 1.0) + 0.5 * rho * (u * u + v * v);
            }
        (void)ds->upload(h);
    }
};

// Uniform freestream on the stretched mesh: the per-axis face closure of the
// metric tables and the remapped characteristic terms both vanish, so the
// state must stay at freestream to FP32 metric rounding.
double stretchedFreestreamDrift()
{
    StretchRig r(96, 48, 2.0, 1e-3);
    r.set([](double, double) {
        return std::array{rhoInf, uInf, 0.0, pInf};
    });
    for (int it = 0; it < 100; ++it)
        r.dev->step(*r.ds, gpu::Real(r.dev->stableDt(*r.ds, *r.red)));
    auto h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    double d = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        d = std::max(d, std::abs(h.rho[c] - rhoInf));
        d = std::max(d, std::abs(h.mx[c] - rhoInf * uInf));
        d = std::max(d, std::abs(h.my[c]));
        d = std::max(d, std::abs(h.E[c] - EInf) / EInf);
    }
    return d;
}

// beta = 0 is the exact identity remap on exactly-uniform metric tables: the
// stretched NSCBC path must reproduce the uniform path to FP32 arithmetic
// reordering (curv flux + remap divisors round differently, so not bitwise).
double stretchedBeta0VsUniform()
{
    Rig a(96, 48, 1e-3);
    StretchRig b(96, 48, 0.0, 1e-3);
    auto init = [](auto& r, auto set) {
        const double Lx = r.g.nx * r.g.hx, Ly = r.g.ny * r.g.hy;
        set([&](double x, double y) {
            const double dx = x - 0.6 * Lx, dy = y - 0.5 * Ly, kw = 6 * r.g.hx;
            const double e =
                0.05 * uInf * std::exp(-(dx * dx + dy * dy) / (kw * kw));
            return std::array{rhoInf, uInf + 2 * dy / kw * e, -2 * dx / kw * e,
                              pInf};
        });
    };
    init(a, [&](auto f) { setState(a, f); });
    init(b, [&](auto f) { b.set(f); });
    for (int it = 0; it < 50; ++it) {
        const double dt = a.dev->stableDt(*a.ds, *a.red);
        a.dev->step(*a.ds, gpu::Real(dt));
        b.dev->step(*b.ds, gpu::Real(dt));
    }
    auto ha = CompState::zeros(a.g), hb = CompState::zeros(b.g);
    (void)a.ds->download(ha);
    (void)b.ds->download(hb);
    double d = 0.0;
    for (std::size_t c = 0; c < a.g.size(); ++c)
        d = std::max({d, std::abs(ha.rho[c] - hb.rho[c]),
                      std::abs(ha.mx[c] - hb.mx[c]),
                      std::abs(ha.my[c] - hb.my[c]),
                      std::abs(ha.E[c] - hb.E[c]) / EInf});
    return d;
}

// Right-going acoustic pulse on the stretched mesh (y-uniform pulse, so the
// clustering is exercised only through the remapped boundary terms): it must
// leave through the outflow with the same small reflection as uniform.
double stretchedPulseReflection(double amp)
{
    StretchRig r(256, 48, 2.0, 0.0);
    const double Lx = r.g.nx * r.g.hx, x0 = 0.5 * Lx, sig = 8 * r.g.hx;
    r.set([&](double x, double) {
        const double pp =
            amp * pInf * std::exp(-(x - x0) * (x - x0) / (sig * sig));
        return std::array{rhoInf + pp / (cInf * cInf),
                          uInf + pp / (rhoInf * cInf), 0.0, pInf + pp};
    });
    r.run(2.5 * 0.5 * Lx / (uInf + cInf));
    r.dev->primitives(*r.ds);
    std::vector<gpu::Real> hp(r.g.size());
    (void)r.dev->p().download(hp.data());
    double resid = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        if (!std::isfinite(double(hp[c]))) return 1e9;
        resid = std::max(resid, std::abs(double(hp[c]) - pInf));
    }
    return resid / (amp * pInf);
}

// Gaussian vortex advecting through the outflow of the stretched mesh -- the
// end-to-end case --stretch + --far nscbc exists for. Placed mid-box where
// the clustering is mildest, but its induced field reaches the outflow rows
// at every y, so the remapped transverse terms are fully exercised.
double stretchedVortexResidual()
{
    StretchRig r(192, 64, 1.5, 1e-4);
    const double Lx = r.g.nx * r.g.hx, Ly = r.g.ny * r.g.hy;
    const double x0 = 0.75 * Lx, y0 = 0.5 * Ly, kw = 8 * r.g.hx, A = 0.1 * uInf;
    double v0 = 0.0;
    r.set([&](double x, double y) {
        const double dx = x - x0, dy = y - y0;
        const double e = A * std::exp(-(dx * dx + dy * dy) / (kw * kw));
        const double u = uInf + 2 * dy / kw * e, v = -2 * dx / kw * e;
        v0 = std::max(v0, std::abs(v));
        return std::array{rhoInf, u, v, pInf};
    });
    r.run((Lx - x0 + 6 * kw) / uInf + 4 * kw / cInf);
    auto h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    double resid = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        if (!std::isfinite(h.my[c])) return 1e9;
        resid = std::max(resid, std::abs(h.my[c] / h.rho[c]));
    }
    return resid / v0;
}

// Potential correction, neutral parameters (Q = 0, aBody = 0): the targets
// reduce to uniform freestream and the state after N steps must be BITWISE
// identical to the uncorrected boundary — no 2D run moves unless the flag is
// armed with a real source.
bool potNeutralBitIdentical()
{
    auto init = [](Rig& r) {
        const double Lx = r.g.nx * r.g.hx, Ly = r.g.ny * r.g.hy;
        setState(r, [&](double x, double y) {
            const double dx = x - 0.5 * Lx, dy = y - 0.5 * Ly, kw = 6 * r.g.hx;
            const double e = 0.05 * uInf * std::exp(-(dx * dx + dy * dy) / (kw * kw));
            return std::array{rhoInf, uInf + 2 * dy / kw * e, -2 * dx / kw * e,
                              pInf};
        });
    };
    Rig a(96, 48, 1e-3), b(96, 48, 1e-3);
    gpu::DeviceCompNscbc::Config c{rhoInf, uInf, pInf, gam};
    c.potCorrect = true; // Q defaults to 0, aBody = 0
    c.xc = 0.3 * a.g.nx * a.g.hx;
    c.yc = 0.5 * a.g.ny * a.g.hy;
    b.nsc = gpu::DeviceCompNscbc::make(b.g, c);
    if (!b.nsc) return false;
    init(a);
    init(b);
    for (int it = 0; it < 50; ++it) {
        const double dt = a.dev->stableDt(*a.ds, *a.red);
        a.dev->step(*a.ds, gpu::Real(dt));
        b.dev->step(*b.ds, gpu::Real(dt));
    }
    auto ha = CompState::zeros(a.g), hb = CompState::zeros(b.g);
    (void)a.ds->download(ha);
    (void)b.ds->download(hb);
    for (std::size_t q = 0; q < a.g.size(); ++q)
        if (ha.rho[q] != hb.rho[q] || ha.mx[q] != hb.mx[q] ||
            ha.my[q] != hb.my[q] || ha.E[q] != hb.E[q])
            return false;
    return true;
}

// Armed correction (Q > 0, aBody > 0) about a centre mid-domain: the stiff
// inflow now relaxes toward the source/doublet DEFICIT, so from uniform
// freestream the inflow-row u must settle below uInf, onto the potential
// target, and the state must stay finite.
double potInflowDeficit()
{
    Rig r(96, 48, 1e-3);
    const double Lx = r.g.nx * r.g.hx, Ly = r.g.ny * r.g.hy;
    gpu::DeviceCompNscbc::Config c{rhoInf, uInf, pInf, gam};
    c.potCorrect = true;
    c.xc = 0.5 * Lx;
    c.yc = 0.5 * Ly;
    c.aBody = 0.06 * Lx;
    r.nsc = gpu::DeviceCompNscbc::make(r.g, c);
    if (!r.nsc) return 1e9;
    r.nsc->setWakeSource(0.1 * uInf * Lx); // strong, well-defined source
    setState(r, [](double, double) {
        return std::array{rhoInf, uInf, 0.0, pInf};
    });
    r.run(2.0 * Lx / uInf);
    auto h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    // Expected target on the inflow centreline.
    const double dx = 0.5 * r.g.hx - c.xc;
    const double uPot = uInf + 0.1 * uInf * Lx / (2 * M_PI) * dx / (dx * dx) -
                        uInf * c.aBody * c.aBody / (dx * dx);
    const std::size_t cIn = r.g.idx(0, r.g.ny / 2, 0);
    if (!std::isfinite(h.mx[cIn]) || !std::isfinite(h.rho[cIn])) return 1e9;
    const double uIn = h.mx[cIn] / h.rho[cIn];
    std::printf("pot inflow u: %.4f (target %.4f)\n", uIn, uPot);
    return std::abs(uIn - uPot) / (uInf - uPot); // relative to the deficit
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    const double drift = freestreamDrift();
    std::printf("freestream drift: %.2e\n", drift);
    check(drift < 5e-6, "uniform freestream fixed to FP32 rounding");

    const double refl = pulseReflection(0.01);
    std::printf("acoustic pulse reflection: %.3f\n", refl);
    check(refl < 0.15, "acoustic pulse exits, reflection < 15%");

    const double vres = vortexResidual();
    std::printf("vortex residual |v|/v0: %.3f\n", vres);
    check(vres < 0.15, "vortex exits, residual < 15% of peak");

    const double pull = inflowPull();
    std::printf("inflow offset remaining: %.3f\n", pull);
    check(pull < 0.5, "inflow relaxation pulls u toward target");

    const double drift2 = openFreestreamDrift(1, 100);
    std::printf("open-lateral freestream drift (2D): %.2e\n", drift2);
    check(drift2 < 5e-6, "open 2D freestream fixed to FP32 rounding");

    const double drift3 = openFreestreamDrift(48, 40);
    std::printf("open freestream drift (3D): %.2e\n", drift3);
    check(drift3 < 5e-6, "open 3D freestream fixed to FP32 rounding");

    const double rres = radialPulseResidual();
    std::printf("radial pulse residual (all faces): %.3f\n", rres);
    check(rres < 0.2, "radial pulse exits all faces, residual < 20%");

    const double lres = lateralVortexResidual();
    std::printf("lateral vortex residual |v|/v0: %.3f\n", lres);
    check(lres < 0.2, "vortex skims open lateral face, residual < 20%");

    const double zdrift = openZOnlyFreestreamDrift(40);
    std::printf("openZ-only freestream drift (3D): %.2e\n", zdrift);
    check(zdrift < 5e-6, "openZ-only freestream fixed to FP32 rounding");

    const double zres = openZOnlyPulseExit(0.01);
    std::printf("openZ-only pulse residual: %.3f\n", zres);
    check(zres < 0.2, "z pulse exits, openZ-only edge lines, residual < 20%");

    const double fres = lateralVortexAlongFace();
    std::printf("lateral vortex-on-face residual |v|/v0: %.3f\n", fres);
    check(fres < 0.2, "vortex on open lateral face, residual < 20%");
    // Vortical transparency vs the lateral relax rate: is the default 0.25
    // near the optimum, and does stiffening reproduce the wall-in-disguise?
    const double fSoft = lateralVortexAlongFace(0.05);
    const double fStiff = lateralVortexAlongFace(1.0);
    std::printf("  etaLat scan: 0.05 -> %.3f, 0.25 -> %.3f, 1.0 -> %.3f\n",
                fSoft, fres, fStiff);

    const double sdrift = stretchedFreestreamDrift();
    std::printf("stretched freestream drift: %.2e\n", sdrift);
    check(sdrift < 1e-4, "stretched freestream fixed to FP32 metric rounding");

    const double b0 = stretchedBeta0VsUniform();
    std::printf("stretched beta=0 vs uniform max diff: %.2e\n", b0);
    check(b0 < 1e-3, "beta = 0 stretched path matches the uniform path");

    const double srefl = stretchedPulseReflection(0.01);
    std::printf("stretched pulse reflection: %.3f\n", srefl);
    check(srefl < 0.15, "stretched acoustic pulse exits, reflection < 15%");

    const double svres = stretchedVortexResidual();
    std::printf("stretched vortex residual |v|/v0: %.3f\n", svres);
    check(svres < 0.15, "stretched vortex exits, residual < 15% of peak");

    check(potNeutralBitIdentical(),
          "potCorrect with Q = 0, aBody = 0 is bit-identical");

    const double pres = potInflowDeficit();
    check(pres < 0.5, "armed potential targets pull the inflow row onto the "
                      "source/doublet deficit");

    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
