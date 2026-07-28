// GPU compressible solver vs CPU reference: step parity on a viscous
// isentropic vortex, and FP32 conservation drift bound.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/transport.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;



CompState vortexState(const Comp& c, double beta)
{
    auto s = CompState::zeros(c.g);
    const double L = c.g.nx * c.g.hx, x0 = L / 2, y0 = L / 2;
    for (int j = 0; j < c.g.ny; ++j)
        for (int i = 0; i < c.g.nx; ++i) {
            double dx = (i + 0.5) * c.g.hx - x0, dy = (j + 0.5) * c.g.hy - y0;
            const double r2 = dx * dx + dy * dy;
            const double f = beta / (2 * pi) * std::exp(0.5 * (1.0 - r2));
            const double u = 1.0 - f * dy, v = 1.0 + f * dx;
            const double T =
                1.0 - (c.gamma() - 1) * beta * beta / (8 * c.gamma() * pi * pi) * std::exp(1 - r2);
            const double rho = std::pow(T, 1.0 / (c.gamma() - 1)), p = rho * T;
            const std::size_t k = c.g.idx(i, j, 0);
            s.rho[k] = rho;
            s.mx[k] = rho * u;
            s.my[k] = rho * v;
            s.E[k] = p / (c.gamma() - 1) + 0.5 * rho * (u * u + v * v);
        }
    return s;
}

double relLinf(const std::vector<double>& a, const std::vector<double>& b)
{
    double e = 0.0, m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
        m = std::max(m, std::abs(b[i]));
    }
    return e / m;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    const int n = 64;
    const double L = 10.0;
    const auto g = Grid::make(n, n, 1, L, L, L / n);
    const auto cpu = Comp::make(g, 1.4, 2e-3);
    auto sc = vortexState(cpu, 5.0);
    auto sg = sc;

    auto dev = gpu::DeviceComp::make(g, 1.4, 2e-3);
    if (!dev) return test::skipExitCode;
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!ds) return EXIT_FAILURE;
    (void)ds->upload(sg);
    auto scratch = gpu::ReduceScratch::make();
    if (!scratch) return EXIT_FAILURE;

    const double dtc = *cpu.stableDt(sc);
    const double dtg = dev->stableDt(*ds, *scratch);
    std::printf("dt cpu %.6e gpu %.6e\n", dtc, dtg);
    check(std::abs(dtc - dtg) < 1e-5 * dtc, "stableDt parity");

    const double dt = 0.9 * dtc;
    for (int it = 0; it < 20; ++it) {
        cpu.step(sc, dt);
        dev->step(*ds, gpu::Real(dt));
    }
    (void)ds->download(sg);
    const double er = relLinf(sg.rho, sc.rho), em = relLinf(sg.mx, sc.mx),
                 ee = relLinf(sg.E, sc.E);
    std::printf("20-step rel Linf: rho %.2e mx %.2e E %.2e\n", er, em, ee);
    check(er < 1e-4 && em < 1e-4 && ee < 1e-4, "step parity within FP32");

    // FP32 conservation drift over 100 steps.
    double s0[4] = {0, 0, 0, 0}, s1[4] = {0, 0, 0, 0};
    for (std::size_t i = 0; i < g.size(); ++i) {
        s0[0] += sg.rho[i]; s0[1] += sg.mx[i]; s0[2] += sg.my[i]; s0[3] += sg.E[i];
    }
    for (int it = 0; it < 100; ++it) dev->step(*ds, gpu::Real(dt));
    (void)ds->download(sg);
    for (std::size_t i = 0; i < g.size(); ++i) {
        s1[0] += sg.rho[i]; s1[1] += sg.mx[i]; s1[2] += sg.my[i]; s1[3] += sg.E[i];
    }
    bool ok = true;
    for (int q = 0; q < 4; ++q) {
        const double scale = std::max(std::abs(s0[q]), double(g.size()));
        if (std::abs(s1[q] - s0[q]) > 1e-4 * scale) ok = false;
    }
    std::printf("FP32 drift: rho %.2e mx %.2e my %.2e E %.2e\n", s1[0] - s0[0],
                s1[1] - s0[1], s1[2] - s0[2], s1[3] - s0[3]);
    check(ok, "FP32 conservation drift bounded");

    // Sutherland mu(T), kappa(T): the device evaluates the law in FP32 from
    // the face-average temperature and the CPU in FP64 from the same average,
    // so this is the parity check that pins ViscView against dns::Transport.
    // The vortex carries a real temperature dip, so mu genuinely varies.
    {
        const double Tref = 1.0; // the isentropic vortex is built around T = 1
        const auto gas =
            Transport::air(2e-3, Tref, 288.15, 0.72, 1.4, 1.0, true);
        auto cpuS = Comp::make(g, 1.4, 2e-3);
        cpuS.tr = gas;
        auto ss = vortexState(cpuS, 5.0);
        auto sgs2 = ss;
        auto devS = gpu::DeviceComp::make(g, gas);
        auto dss = gpu::DeviceCompState::make(g.size());
        if (!devS || !dss) return EXIT_FAILURE;
        (void)dss->upload(sgs2);

        const double dc = *cpuS.stableDt(ss), dg = devS->stableDt(*dss, *scratch);
        std::printf("sutherland dt cpu %.6e gpu %.6e\n", dc, dg);
        check(std::abs(dc - dg) < 1e-4 * dc, "sutherland stableDt parity");

        const double dts = 0.9 * dc;
        for (int it = 0; it < 20; ++it) {
            cpuS.step(ss, dts);
            devS->step(*dss, gpu::Real(dts));
        }
        (void)dss->download(sgs2);
        const double sr = relLinf(sgs2.rho, ss.rho), sm = relLinf(sgs2.mx, ss.mx),
                     se = relLinf(sgs2.E, ss.E);
        std::printf("sutherland 20-step rel Linf: rho %.2e mx %.2e E %.2e\n",
                    sr, sm, se);
        check(sr < 1e-4 && sm < 1e-4 && se < 1e-4,
              "sutherland step parity within FP32");

        // And the law is engaged: the same state under constant viscosity must
        // take a measurably different trajectory.
        auto cpuC = Comp::make(g, 1.4, 2e-3);
        auto sConst = vortexState(cpuC, 5.0);
        for (int it = 0; it < 20; ++it) cpuC.step(sConst, dts);
        check(relLinf(sConst.E, ss.E) > 1e-8,
              "mu(T) changes the trajectory (the law is really on)");
    }

    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
