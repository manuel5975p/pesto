// Channel building blocks: the channelY mesh (exact ghost reflection, wall
// faces on the box edges, valid FV metrics), the isothermal wall mirror
// (quiescent fixed point, adiabatic parity with the hand mirror), and the
// laminar oracle -- body force + walls + stretch + plane stats must
// reproduce the analytic Poiseuille parabola without any turbulence in play.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_channel.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/remap.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace dns;

namespace {

using dnstest::check;



Grid channelGrid(int nx, int nyFluid, int nz, double beta, double lx, double lz)
{
    const int nyT = nyFluid + 2;
    const double h1 = ChannelStretchY::firstCell(nyT, beta, 1.0);
    return Grid::make(nx, nyT, nz, lx, 2.0 + 2.0 * h1, lz);
}

void meshGeometry()
{
    const double beta = 2.0;
    const Grid g = channelGrid(8, 32, 1, beta, 2.0, 2.0 / 8);
    const auto m = CurvMesh::channelY(g, beta, 1.0);
    const double h1 = ChannelStretchY::firstCell(g.ny, beta, 1.0);

    // Wall faces exactly on nodes 1 and ny-1; ghost nodes exact reflections.
    const double y0 = m.y[g.idx(0, 0, 0)], y1 = m.y[g.idx(0, 1, 0)];
    const double y2 = m.y[g.idx(0, 2, 0)];
    const double yTop = m.y[g.idx(0, g.ny - 1, 0)];
    const double yTop1 = m.y[g.idx(0, g.ny - 2, 0)];
    check(std::abs(y1) < 1e-14, "bottom wall face at y = 0");
    check(std::abs(yTop - 2.0) < 1e-12, "top wall face at y = 2 delta");
    check(std::abs(y0 - (2 * y1 - y2)) < 1e-12,
          "bottom ghost node is the exact reflection");
    check(std::abs(y2 - h1) < 1e-12, "first fluid cell height = firstCell()");
    // Periodic image of node 0 closes the top ghost cell as a reflection.
    const double period = g.ny * g.hy;
    check(std::abs((y0 + period) - (2 * yTop - yTop1)) < 1e-12,
          "top ghost cell is the exact reflection (via the period)");

    auto mt = CurvMetrics::build(m);
    check(mt.has_value(), "CurvMetrics::build accepts the channel mesh");
    if (mt)
    {
        double vol = 0;
        for (double v : mt->vol) vol += v;
        const double box = (g.nx * g.hx) * period * (g.nz == 1 ? g.hz : g.nz * g.hz);
        check(std::abs(vol - box) < 1e-9 * box, "cells tile the box");
    }
}

void wallMirror()
{
    const double beta = 1.0, gamma = 1.4, p0 = 286.0;
    const Grid g = channelGrid(8, 16, 1, beta, 2.0, 2.0 / 8);

    // Quiescent isothermal state at T = Twall is a bit-exact fixed point.
    auto s = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        s.rho[c] = 1.0;
        s.E[c] = p0 / (gamma - 1.0);
    }
    auto ds = gpu::DeviceCompState::make(g.size());
    auto wall = gpu::DeviceChannelWall::make(g, {gamma, 1.0, p0, true});
    if (!ds || !wall) { check(false, "alloc"); return; }
    (void)ds->upload(s);
    wall->apply(*ds);
    auto r = CompState::zeros(g);
    (void)ds->download(r);
    // FP32 fields: the ghost reconstruction roundtrips rho/E through float,
    // so the fixed point holds to float precision, not bitwise.
    double worstFp = 0;
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        worstFp = std::max(worstFp, std::abs(r.rho[c] - s.rho[c]) / s.rho[c]);
        worstFp = std::max(worstFp, std::abs(r.mx[c] - s.mx[c]) / p0);
        worstFp = std::max(worstFp, std::abs(r.E[c] - s.E[c]) / s.E[c]);
    }
    check(worstFp < 1e-6, "quiescent isothermal state is a fixed point");

    // Adiabatic mode reproduces the plain mirror arithmetic bit-for-bit.
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        s.mx[c] = 0.1f * float(c % 7) - 0.3f;
        s.my[c] = 0.05f * float(c % 5) - 0.1f;
        s.E[c] = p0 / (gamma - 1.0) + 0.01f * float(c % 11);
    }
    auto wallA = gpu::DeviceChannelWall::make(g, {gamma, 1.0, p0, false});
    if (!wallA) { check(false, "alloc"); return; }
    (void)ds->upload(s);
    wallA->apply(*ds);
    (void)ds->download(r);
    // Host doubles roundtrip through the FP32 device fields; compare against
    // the float-cast source, where the mirror IS exact.
    bool par = true;
    for (int k = 0; k < g.nz && par; ++k)
        for (int i = 0; i < g.nx && par; ++i)
            for (int jj : {0, g.ny - 1})
            {
                const std::size_t c = g.idx(i, jj, k);
                const std::size_t m = g.idx(i, jj == 0 ? 1 : g.ny - 2, k);
                par = par && float(r.rho[c]) == float(s.rho[m])
                      && float(r.mx[c]) == -float(s.mx[m])
                      && float(r.my[c]) == -float(s.my[m])
                      && float(r.mz[c]) == -float(s.mz[m])
                      && float(r.E[c]) == float(s.E[m]);
            }
    check(par, "adiabatic mode is the plain no-slip mirror");
}

// Laminar Poiseuille: f = 1, mu = 0.5 (reTau = 2 in wall units), no trip.
// Steady solution u(y) = (f / 2 mu) y (2 delta - y); u_max = 1. This
// certifies force + isothermal walls + stretched mesh + plane stats at once.
void laminarPoiseuille()
{
    const double beta = 1.0, gamma = 1.4, mu = 0.5, f = 1.0, p0 = 286.0;
    const Grid g = channelGrid(8, 32, 1, beta, 2.0, 2.0 / 8);
    const auto gg = gpu::GpuGrid::from(g);
    const ChannelStretchY remap{g.ny * g.hy, g.ny, beta, 1.0};

    auto mt = CurvMetrics::build(CurvMesh::channelY(g, beta, 1.0));
    if (!mt) { check(false, "metrics"); return; }
    auto dev = gpu::DeviceComp::make(*mt, gamma, mu);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto wall = gpu::DeviceChannelWall::make(g, {gamma, 1.0, p0, true});
    if (!dev || !ds || !red || !wall) { check(false, "alloc"); return; }

    dev->setStageHook([&](gpu::DeviceCompState& s) { wall->apply(s); });
    dev->setRhsHook([&](const gpu::DeviceCompState&, gpu::DeviceCompState& k) {
        gpu::addBodyForceX(gg, f, dev->u(), k);
    });

    auto s = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        s.rho[c] = 1.0;
        s.E[c] = p0 / (gamma - 1.0);
    }
    (void)ds->upload(s);
    wall->apply(*ds);

    // March to steady: viscous time delta^2 (mu/rho) = 2; integrate t = 12.
    double t = 0, dtStab = 0;
    int since = 1 << 30;
    long steps = 0;
    while (t < 12.0)
    {
        if (++since >= 16)
        {
            dtStab = 0.95 * dev->stableDt(*ds, *red, 0.4);
            if (!(dtStab > 0)) { check(false, "dt collapsed"); return; }
            since = 0;
        }
        dev->step(*ds, gpu::Real(dtStab));
        wall->apply(*ds);
        t += dtStab;
        ++steps;
    }

    dev->primitives(*ds);
    gpu::DeviceBuffer<double> scr;
    std::vector<double> ps;
    if (gpu::channelPlaneStats(gg, ds->rho, dev->u(), dev->v(), dev->w(),
                               dev->T(), scr, ps) != hipSuccess)
    { check(false, "plane stats"); return; }

    double worst = 0, umax = 0;
    for (int j = 1; j <= g.ny - 2; ++j)
    {
        const double y = remap.mapY((j + 0.5) * g.hy);
        const double uF = ps[j * 8 + 1] / ps[j * 8 + 0];
        const double uRef = f / (2 * mu) * y * (2.0 - y);
        worst = std::max(worst, std::abs(uF - uRef));
        umax = std::max(umax, uF);
    }
    std::printf("laminar: %ld steps, u_max %.4f (ref 1), Linf err %.2e\n",
                steps, umax, worst);
    check(worst < 0.015, "profile matches the Poiseuille parabola");

    // Wall shear from the measured profile: tau_w = f delta = 1 identically.
    auto uF = [&](int j) { return ps[j * 8 + 1] / ps[j * 8 + 0]; };
    const double y1 = remap.mapY(1.5 * g.hy), y2 = remap.mapY(2.5 * g.hy);
    const double dudy = (uF(1) * y2 * y2 - uF(2) * y1 * y1)
                      / (y1 * y2 * (y2 - y1));
    std::printf("laminar: tau_wall %.4f (ref 1)\n", mu * dudy);
    // The extracted gradient leans on the wall-adjacent cell, where the
    // mirror closure's truncation concentrates: measured 8.5% at ny=16,
    // 4.3% at ny=32 (halving with dy). At production resolution (y+ < 1)
    // the bias is well under 1%; this guards against regressions, not
    // absolute accuracy at ny=32.
    check(std::abs(mu * dudy - 1.0) < 0.05, "measured tau_wall = f delta = 1");
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    meshGeometry();
    wallMirror();
    laminarPoiseuille();
    if (dnstest::gFailures) { std::printf("%d failures\n", dnstest::gFailures); return 1; }
    std::printf("all ok\n");
    return 0;
}
