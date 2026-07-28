// Compressible ghost wall: quiescent invariance (no spurious currents or
// force), impulsive freestream stability + positive drag on the circle, and the
// thermal wall -- adiabatic reports no heat at all, an isothermal wall at the
// freestream temperature is a fixed point, hot/cold walls drive heat with the
// right sign and magnitude, and the wall face actually holds Twall.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

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
// linear fit through the pair is the wall face. R = 1 in these rigs, so
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
    const int ndir = g.nz == 1 ? 4 : 6;
    maxRel = 0.0;
    double sum = 0.0;
    int n = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double pf = phiAt(i, j, k);
                if (pf <= 0.0) continue;
                for (int d = 0; d < ndir; ++d) {
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

    const int n = 128;
    const double L = 2 * pi, R = L / 16, cx = L / 2, cy = L / 2 + 0.013;
    const double gamma = 1.4, rhoInf = 1.0, pInf = 1.0 / (gamma * 0.04); // M 0.2
    const auto g = Grid::make(n, n, 1, L, L, 1.0); // hz = 1: unit depth
    auto sdf = [&](double x, double y, double) {
        return std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) - R;
    };
    const double mu = rhoInf * 2 * R / 100.0; // Re 100 on uInf = 1

    auto dev = gpu::DeviceComp::make(g, gamma, mu);
    auto ghost = gpu::DeviceCompGhost::make(g, sdf, gamma, rhoInf, pInf);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ghost || !ds || !red) return EXIT_FAILURE;
    dev->setStageHook([&](gpu::DeviceCompState& st) { ghost->apply(st); });

    // Quiescent gas + wall must stay exactly quiescent: mirror of zero is
    // zero, copied p/rho are uniform.
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
        vmax = std::max({vmax, std::abs(host.mx[c]), std::abs(host.my[c])});
    const auto F0 = ghost->force(*dev, *ds);
    std::printf("quiescent: max|m| %.2e force (%.2e, %.2e) heat %.2e\n", vmax,
                F0.fx, F0.fy, F0.heat);
    check(vmax < 1e-5, "quiescent gas stays at rest");
    check(std::abs(F0.fx) < 1e-4 && std::abs(F0.fy) < 1e-4, "no spurious force");
    check(std::abs(F0.heat) < 1e-4, "no spurious heat");

    // Tapered freestream: stable evolution, positive drag, bounded speed.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double d = sdf((i + 0.5) * g.hx, (j + 0.5) * g.hy, 0.0);
            const double f = std::clamp(d / (3.0 * g.hx), 0.0, 1.0);
            const double u = f * f * (3 - 2 * f);
            host.rho[c] = rhoInf;
            host.mx[c] = rhoInf * u;
            host.my[c] = 0.0;
            host.E[c] = pInf / (gamma - 1.0) + 0.5 * rhoInf * u * u;
        }
    (void)ds->upload(host);
    double t = 0.0;
    while (t < 2.0) {
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
        const double u = host.mx[c] / host.rho[c], v = host.my[c] / host.rho[c];
        if (!std::isfinite(u) || !std::isfinite(v)) finite = false;
        smax = std::max(smax, std::sqrt(u * u + v * v));
    }
    const double cd = F.fx / (0.5 * rhoInf * 2 * R);
    std::printf("freestream t=2: cd %.3f max|u| %.3f\n", cd, smax);
    check(finite, "freestream evolution finite");
    check(cd > 0.5 && cd < 30.0, "drag positive and sane (early transient)");
    check(smax < 3.0, "speed bounded");

    // The tally used to atomicAdd in FP32: cancelling +- contributions made the
    // reported cd irreproducible run to run, non-determinism in the primary
    // validated quantity. It now stages per face in FP64 and reduces
    // deterministically, so repeats must agree to the last bit.
    {
        bool same = true;
        for (int r = 0; r < 4; ++r) {
            const auto Fr = ghost->force(*dev, *ds);
            if (Fr.fx != F.fx || Fr.fy != F.fy || Fr.fz != F.fz) same = false;
        }
        check(same, "force tally is bit-reproducible across repeats");
        std::printf("cd (FP64 tally) = %.17g\n", cd);
    }

    // --- thermal wall -------------------------------------------------------
    using WT = gpu::GhostWallThermal;
    const double Tinf = pInf / rhoInf; // R = 1 in this rig

    // An adiabatic wall carries NO heat by construction: its ghost copies p and
    // rho from the sample point, so the wall-normal dT/dn the image encodes is
    // identically zero and the conduction tally is zero face by face -- even
    // here, on a developed viscous field that HAS a wall thermal signature.
    // This is what makes the reported qw readable: a nonzero value is a flux,
    // never an offset. (The convective and viscous-work parts of the face
    // energy flux are deliberately NOT in it; on this staircase they do not
    // vanish and would masquerade as wall heat.)
    std::printf("adiabatic wall on the developed field: heat %.3e\n", F.heat);
    check(F.heat == 0.0, "adiabatic wall reports exactly zero heat");

    // The thermal configuration must not touch the momentum path: an explicitly
    // adiabatic wall is the default wall, bit for bit, forces included. The
    // repo's cd/cl anchors ride on this.
    {
        auto gh = gpu::DeviceCompGhost::make(
            g, sdf, gamma, rhoInf, pInf, nullptr,
            WT{.mode = WT::Mode::Adiabatic, .Twall = Tinf, .Rgas = 1.0});
        auto db = gpu::DeviceCompState::make(g.size());
        if (!gh || !db) return EXIT_FAILURE;
        (void)db->upload(host); // the evolved freestream field
        auto da = gpu::DeviceCompState::make(g.size());
        if (!da) return EXIT_FAILURE;
        (void)da->upload(host);
        ghost->apply(*da);
        gh->apply(*db);
        auto ha = CompState::zeros(g), hb = CompState::zeros(g);
        (void)da->download(ha);
        (void)db->download(hb);
        bool same = true;
        for (std::size_t c = 0; c < g.size(); ++c)
            if (ha.rho[c] != hb.rho[c] || ha.mx[c] != hb.mx[c] ||
                ha.my[c] != hb.my[c] || ha.mz[c] != hb.mz[c] ||
                ha.E[c] != hb.E[c])
                same = false;
        check(same, "explicit adiabatic wall fill is bit-identical to default");
        const auto Fa = ghost->force(*dev, *da);
        const auto Fb = gh->force(*dev, *db);
        check(Fa.fx == Fb.fx && Fa.fy == Fb.fy && Fa.fz == Fb.fz &&
                  Fa.heat == Fb.heat,
              "explicit adiabatic wall tally is bit-identical to default");
    }

    // Twall == Tinf on a quiescent uniform gas is a fixed point: the image
    // returns the freestream temperature, so there is neither momentum nor heat
    // to transfer.
    {
        auto gh = gpu::DeviceCompGhost::make(
            g, sdf, gamma, rhoInf, pInf, nullptr,
            WT{.mode = WT::Mode::Isothermal, .Twall = Tinf, .Rgas = 1.0});
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
        for (int it = 0; it < 50; ++it) dev2->step(*ds2, gpu::Real(1e-3));
        gh->apply(*ds2);
        (void)ds2->download(q);
        double mmax = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c)
            mmax = std::max({mmax, std::abs(q.mx[c]), std::abs(q.my[c])});
        const auto Fi = gh->force(*dev2, *ds2);
        std::printf("isothermal at Tinf: max|m| %.2e force (%.2e, %.2e) "
                    "heat %.2e\n", mmax, Fi.fx, Fi.fy, Fi.heat);
        check(mmax < 1e-5, "isothermal wall at Tinf keeps the gas at rest");
        check(std::abs(Fi.fx) < 1e-4 && std::abs(Fi.fy) < 1e-4,
              "isothermal wall at Tinf makes no spurious force");
        check(std::abs(Fi.heat) < 1e-4,
              "isothermal wall at Tinf makes no spurious heat");
    }

    // Sign and scaling in a quiescent gas, where the tally is pure conduction:
    // a hot wall pushes heat INTO the fluid (heat < 0, the flux is oriented
    // fluid -> body) and a cold wall pulls it out. The face gradient is
    // (1 + dw/dsmp) (Twall - Tinf) / h, i.e. exactly proportional to the
    // temperature difference, so small differences must scale linearly; the
    // large ones only have to keep the sign (the deepest ghosts, where
    // dw/dsmp == 1, hit the positivity floor at Twall = 0.5 Tinf).
    {
        const double qHot = quiescentHeat(g, sdf, gamma, mu, rhoInf, pInf,
                                          1.5 * Tinf);
        const double qCold = quiescentHeat(g, sdf, gamma, mu, rhoInf, pInf,
                                           0.5 * Tinf);
        const double q10 = quiescentHeat(g, sdf, gamma, mu, rhoInf, pInf,
                                         1.1 * Tinf);
        const double q20 = quiescentHeat(g, sdf, gamma, mu, rhoInf, pInf,
                                         1.2 * Tinf);
        std::printf("quiescent wall heat: hot(1.5) %.4e cold(0.5) %.4e "
                    "1.1 %.4e 1.2 %.4e ratio %.4f\n",
                    qHot, qCold, q10, q20, q20 / q10);
        check(std::isfinite(qHot) && std::isfinite(qCold), "wall heat finite");
        check(qHot < 0.0, "hot wall drives heat into the fluid");
        check(qCold > 0.0, "cold wall pulls heat out of the fluid");
        check(std::abs(qHot) > 1e-3 && std::abs(qCold) > 1e-3,
              "wall heat is not degenerately small");
        check(std::abs(q20 / q10 - 2.0) < 0.02,
              "wall heat is linear in Twall - Tinf for small differences");
    }

    // The isothermal wall must actually HOLD its temperature: after a short
    // relaxation from a uniform gas, the temperature interpolated to the wall
    // face has to sit at Twall.
    {
        const double Twall = 1.5 * Tinf;
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
        while (tt < 0.5) {
            const double dt = dev2->stableDt(*ds2, *red2);
            dev2->step(*ds2, gpu::Real(dt));
            tt += dt;
        }
        gh->apply(*ds2);
        (void)ds2->download(q);
        double maxRel = 0.0, meanRel = 0.0;
        wallFaceT(g, q, gamma, gh->solidCells(), gh->deepCells(), sdf, Twall,
                  maxRel, meanRel);
        const auto Fh = gh->force(*dev2, *ds2);
        std::printf("hot wall t=0.5: T_face vs Twall max %.3f%% mean %.3f%%, "
                    "heat %.4e\n", 100 * maxRel, 100 * meanRel, Fh.heat);
        check(meanRel < 0.02, "wall face holds Twall in the mean (< 2%)");
        check(maxRel < 0.08, "wall face holds Twall everywhere (< 8%)");
        check(Fh.heat < 0.0, "hot wall still heats the fluid after relaxing");
    }

    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
