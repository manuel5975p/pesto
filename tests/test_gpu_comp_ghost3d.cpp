// Compressible sharp ghost wall in TRUE 3D (nz>1): quiescent invariance (no
// spurious currents or force), impulsive freestream stability + positive drag
// on a sphere, near-symmetry of the transverse force, and the thermal wall on a
// closed 3D surface. Exercises the 3D paths of DeviceCompGhost the 2D cylinder
// test cannot: z-normals, +z/-z interface faces (ndir=6), mz momentum in the
// mirror fill and force tally, and the z faces of the wall heat tally.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
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

// Temperature interpolated to the wall FACE, one interface pair at a time: the
// ghost centre and its axis neighbour in the fluid lie on the same wall-normal
// profile and the sdf gives each its signed distance, so the zero level of a
// linear fit through the pair is the wall face. R = 1 in this rig, so
// T = (gamma-1) (E - ke) / rho. Reports max and mean |T_face - Twall| / Twall.
template <class Sdf>
void wallFaceT(const Grid& g, const CompState& h, double gamma,
               const std::vector<int>& solid, const std::vector<int>& deep,
               const Sdf& sdf, double Twall, double& maxRel, double& meanRel)
{
    std::vector<char> isGhost(g.size(), 0);
    for (int c : solid) isGhost[c] = 1;
    for (int c : deep) isGhost[c] = 0;
    auto phiAt = [&](int i, int j, int k) {
        return sdf((i + 0.5) * g.hx, (j + 0.5) * g.hy, (k + 0.5) * g.hz);
    };
    auto Tat = [&](std::size_t c) {
        const double ke = 0.5 *
            (h.mx[c] * h.mx[c] + h.my[c] * h.my[c] + h.mz[c] * h.mz[c]) /
            h.rho[c];
        return (gamma - 1.0) * (h.E[c] - ke) / h.rho[c];
    };
    const int dijk[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                            {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    maxRel = 0.0;
    double sum = 0.0;
    int n = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double pf = phiAt(i, j, k);
                if (pf <= 0.0) continue;
                for (int d = 0; d < 6; ++d) {
                    const int ii = i + dijk[d][0], jj = j + dijk[d][1],
                              kk = k + dijk[d][2];
                    if (!isGhost[g.idx(ii, jj, kk)]) continue;
                    const double pg = phiAt(ii, jj, kk);
                    const double w = pf / (pf - pg);
                    const double Tf = Tat(g.idx(i, j, k));
                    const double Tg = Tat(g.idx(ii, jj, kk));
                    const double rel =
                        std::abs(Tf + w * (Tg - Tf) - Twall) / Twall;
                    maxRel = std::max(maxRel, rel);
                    sum += rel;
                    ++n;
                }
            }
    meanRel = n > 0 ? sum / n : 0.0;
}

// Wall heat rate of a quiescent uniform gas against a wall at Twall, from a
// fresh ghost of its own: the field never evolves, so the tally is the pure
// conduction the image drives and nothing else.
double quiescentHeat(const Grid& g, const gpu::DeviceCompGhost::Sdf& sdf,
                     double gamma, double mu, double rhoInf, double pInf,
                     double Twall)
{
    using WT = gpu::GhostWallThermal;
    auto dev = gpu::DeviceComp::make(g, gamma, mu);
    auto gh = gpu::DeviceCompGhost::make(
        g, sdf, gamma, rhoInf, pInf, nullptr,
        WT{.mode = WT::Mode::Isothermal, .Twall = Twall, .Rgas = 1.0});
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !gh || !ds) { check(false, "quiescentHeat setup"); return 0.0; }
    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.E[c] = pInf / (gamma - 1.0);
    }
    (void)ds->upload(host);
    gh->apply(*ds);
    return gh->force(*dev, *ds).heat;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    const int n = 64;
    const double L = 2 * pi, R = L / 8;
    const double cx = L / 2 + 0.013, cy = L / 2 + 0.017, cz = L / 2 + 0.011;
    const double gamma = 1.4, rhoInf = 1.0, pInf = 1.0 / (gamma * 0.04); // M 0.2
    const auto g = Grid::make(n, n, n, L, L, L);
    auto sdf = [&](double x, double y, double z) {
        return std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) +
                         (z - cz) * (z - cz)) -
               R;
    };
    const double mu = rhoInf * 2 * R / 100.0; // Re 100 on uInf = 1

    auto dev = gpu::DeviceComp::make(g, gamma, mu);
    auto ghost = gpu::DeviceCompGhost::make(g, sdf, gamma, rhoInf, pInf);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ghost || !ds || !red) return EXIT_FAILURE;
    dev->setStageHook([&](gpu::DeviceCompState& st) { ghost->apply(st); });

    // Quiescent gas + wall must stay exactly quiescent in 3D.
    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.E[c] = pInf / (gamma - 1.0);
    }
    (void)ds->upload(host);
    for (int it = 0; it < 50; ++it) dev->step(*ds, gpu::Real(1e-3));
    ghost->apply(*ds);
    (void)ds->download(host);
    double vmax = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c)
        vmax = std::max({vmax, std::abs(host.mx[c]), std::abs(host.my[c]),
                         std::abs(host.mz[c])});
    const auto F0 = ghost->force(*dev, *ds);
    std::printf("quiescent: max|m| %.2e force (%.2e, %.2e, %.2e) heat %.2e\n",
                vmax, F0.fx, F0.fy, F0.fz, F0.heat);
    check(vmax < 1e-5, "3D quiescent gas stays at rest");
    check(std::abs(F0.fx) < 1e-4 && std::abs(F0.fy) < 1e-4 &&
              std::abs(F0.fz) < 1e-4,
          "3D no spurious force");

    // Tapered freestream in +x: stable, positive drag, near-symmetric transverse
    // force, bounded speed.
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double d =
                    sdf((i + 0.5) * g.hx, (j + 0.5) * g.hy, (k + 0.5) * g.hz);
                const double f = std::clamp(d / (3.0 * g.hx), 0.0, 1.0);
                const double u = f * f * (3 - 2 * f);
                host.rho[c] = rhoInf;
                host.mx[c] = rhoInf * u;
                host.my[c] = 0.0;
                host.mz[c] = 0.0;
                host.E[c] = pInf / (gamma - 1.0) + 0.5 * rhoInf * u * u;
            }
    (void)ds->upload(host);
    double t = 0.0;
    while (t < 1.0) {
        const double dt = dev->stableDt(*ds, *red);
        dev->step(*ds, gpu::Real(dt));
        t += dt;
    }
    ghost->apply(*ds);
    const auto F = ghost->force(*dev, *ds);
    (void)ds->download(host);
    double smax = 0.0;
    bool finite = true;
    for (std::size_t c = 0; c < g.size(); ++c) {
        const double u = host.mx[c] / host.rho[c], v = host.my[c] / host.rho[c],
                     w = host.mz[c] / host.rho[c];
        if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(w))
            finite = false;
        smax = std::max(smax, std::sqrt(u * u + v * v + w * w));
    }
    // Sphere reference area pi R^2.
    const double cd = F.fx / (0.5 * rhoInf * pi * R * R);
    std::printf("freestream t=1: cd %.3f (fy %.2e fz %.2e) max|u| %.3f\n", cd,
                F.fy, F.fz, smax);
    check(finite, "3D freestream evolution finite");
    check(cd > 0.3 && cd < 30.0, "3D drag positive and sane (early transient)");
    check(std::abs(F.fy) < 0.25 * std::abs(F.fx) &&
              std::abs(F.fz) < 0.25 * std::abs(F.fx),
          "3D transverse force small (symmetry)");
    check(smax < 3.0, "3D speed bounded");

    // --- thermal wall on a closed 3D surface --------------------------------
    using WT = gpu::GhostWallThermal;
    const double Tinf = pInf / rhoInf; // R = 1 in this rig

    // The adiabatic wall carries no heat by construction, on a developed
    // viscous field with a real wall thermal signature included.
    std::printf("3D adiabatic wall on the developed field: heat %.3e\n", F.heat);
    check(F.heat == 0.0, "3D adiabatic wall reports exactly zero heat");

    // Sign and scaling in a quiescent gas, where the tally is pure conduction.
    // Small differences are exactly linear (the face gradient is
    // (1 + dw/dsmp) (Twall - Tinf) / h); the large ones only have to keep the
    // sign, since the deepest ghosts hit the positivity floor at 0.5 Tinf.
    {
        const double qHot = quiescentHeat(g, sdf, gamma, mu, rhoInf, pInf,
                                          1.5 * Tinf);
        const double qCold = quiescentHeat(g, sdf, gamma, mu, rhoInf, pInf,
                                           0.5 * Tinf);
        const double q10 = quiescentHeat(g, sdf, gamma, mu, rhoInf, pInf,
                                         1.1 * Tinf);
        const double q20 = quiescentHeat(g, sdf, gamma, mu, rhoInf, pInf,
                                         1.2 * Tinf);
        std::printf("3D quiescent wall heat: hot(1.5) %.4e cold(0.5) %.4e "
                    "1.1 %.4e 1.2 %.4e ratio %.4f\n",
                    qHot, qCold, q10, q20, q20 / q10);
        check(std::isfinite(qHot) && std::isfinite(qCold),
              "3D wall heat finite");
        check(qHot < 0.0, "3D hot wall drives heat into the fluid");
        check(qCold > 0.0, "3D cold wall pulls heat out of the fluid");
        check(std::abs(qHot) > 1e-3 && std::abs(qCold) > 1e-3,
              "3D wall heat is not degenerately small");
        check(std::abs(q20 / q10 - 2.0) < 0.02,
              "3D wall heat is linear in Twall - Tinf for small differences");
    }

    // Twall == Tinf is a fixed point of a quiescent gas, and a hot wall must
    // hold its temperature at the wall face after relaxing.
    for (const double f : {1.0, 1.5}) {
        const double Twall = f * Tinf;
        auto gh = gpu::DeviceCompGhost::make(
            g, sdf, gamma, rhoInf, pInf, nullptr,
            WT{.mode = WT::Mode::Isothermal, .Twall = Twall, .Rgas = 1.0});
        auto ds2 = gpu::DeviceCompState::make(g.size());
        auto red2 = gpu::ReduceScratch::make();
        auto dev2 = gpu::DeviceComp::make(g, gamma, mu);
        if (!gh || !ds2 || !red2 || !dev2) return EXIT_FAILURE;
        dev2->setStageHook([&](gpu::DeviceCompState& st) { gh->apply(st); });
        auto q = CompState::zeros(g);
        for (std::size_t c = 0; c < g.size(); ++c) {
            q.rho[c] = rhoInf;
            q.E[c] = pInf / (gamma - 1.0);
        }
        (void)ds2->upload(q);
        double tt = 0.0;
        while (tt < 0.3) {
            const double dt = dev2->stableDt(*ds2, *red2);
            dev2->step(*ds2, gpu::Real(dt));
            tt += dt;
        }
        gh->apply(*ds2);
        (void)ds2->download(q);
        double mmax = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c)
            mmax = std::max({mmax, std::abs(q.mx[c]), std::abs(q.my[c]),
                             std::abs(q.mz[c])});
        double maxRel = 0.0, meanRel = 0.0;
        wallFaceT(g, q, gamma, gh->solidCells(), gh->deepCells(), sdf, Twall,
                  maxRel, meanRel);
        const auto Fw = gh->force(*dev2, *ds2);
        std::printf("3D isothermal Twall=%.2f Tinf, t=0.3: max|m| %.2e "
                    "T_face max %.3f%% mean %.3f%% heat %.4e\n",
                    f, mmax, 100 * maxRel, 100 * meanRel, Fw.heat);
        check(meanRel < 0.02, "3D wall face holds Twall in the mean (< 2%)");
        check(maxRel < 0.08, "3D wall face holds Twall everywhere (< 8%)");
        if (f == 1.0) {
            check(mmax < 1e-5,
                  "3D isothermal wall at Tinf keeps the gas at rest");
            check(std::abs(Fw.heat) < 1e-4,
                  "3D isothermal wall at Tinf makes no spurious heat");
        } else {
            check(Fw.heat < 0.0,
                  "3D hot wall still heats the fluid after relaxing");
        }
    }

    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
