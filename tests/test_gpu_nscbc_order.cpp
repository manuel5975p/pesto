// What an open face LEAVES BEHIND at oblique incidence, and why it is mostly
// vorticity.
//
// gpu_nscbc_absorb measures the reflected ACOUSTIC energy and finds the face
// also leaves spurious vorticity, larger than the acoustics, at every angle off
// normal. This test isolates the source of it and the fix.
//
// The mechanism. On a boundary row the LODI form replaces the normal-axis rhs
// by the characteristic amplitudes. The tangential-momentum and transverse
// terms it keeps are EXACT (L3 = un dv/dn is the true -u dv/dx, and the
// transverse flux divergence is the interior scheme), so the only error the
// face makes enters du_n/dt:
//
//     d(du_n/dt) = -(dL5 - dL1) / (2 rho c)
//
// with no companion tangential velocity. A perturbation in the NORMAL velocity
// alone, varying along the face, is a vortex sheet:
//
//     d(omega)/dt = -d/dy (d(du/dt)) = d/dy (dL5 - dL1) / (2 rho c)
//
// and in an inviscid barotropic interior vorticity is conserved, so what the
// row lays down at lambda = u stays there for good in a quiescent box.
//
// Which error dominates is the point. dL1 is the modeled incoming amplitude --
// the term --nscbc-beta corrects, and the Yoo-Im choice beta = 0.5 already
// cancels its leading angle error. dL5 is the OUTGOING amplitude, built from a
// one-sided normal difference, and at 30 deg incidence |L5| is ~14x |L1|, so
// its O(h) truncation is the larger of the two by an order of magnitude. It
// never shows up in the reflected acoustic energy -- it is injected with
// OUTGOING polarization (p' = +rho c u'), so its acoustic part leaves -- but
// its transverse gradient is vorticity, and that does not.
//
// So: --nscbc-order 2 (Config::normalOrder), a three-point one-sided normal
// difference. The columns below are the before/after.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_far.hpp"
#include "dns/gpu/comp_nscbc.hpp"
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
constexpr double Lbox = 2.56;

// What the face left in the box: acoustic energy, and the PEAK of each of the
// two lambda = u families, which a quiescent box cannot vent at all. peakI/peakJ
// locate the vorticity peak, because WHERE it sits is what says whether it is
// physics or the metric's own wrap.
struct Left
{
    double energy = 0, vorticity = 0, entropy = 0;
    int peakI = -1, peakJ = -1;
};

// x-characteristic faces, free-slip y walls (lossless, so every bit of the
// ridge has to leave through an x face), no sponge.
struct Rig
{
    Grid g;
    std::expected<gpu::DeviceComp, hipError_t> dev;
    std::expected<gpu::DeviceCompFar, hipError_t> far;
    std::expected<gpu::DeviceCompNscbc, hipError_t> nsc;
    std::expected<gpu::DeviceCompState, hipError_t> ds;
    std::expected<gpu::ReduceScratch, hipError_t> red;

    // abc: 0 = fixed betaTrans, 1 = per face cell, 2 = per DCT mode.
    Rig(int nx, int ny, int abc, double betaTrans = 0.5)
        : g(Grid::make(nx, ny, 1, Lbox, ny * (Lbox / nx), 1.0)),
          dev(gpu::DeviceComp::make(g, gam, 0.0)),
          far(gpu::DeviceCompFar::make(
              g, {rhoInf, 0.0, pInf, gam, 8, 0.0, false, false})),
          nsc(gpu::DeviceCompNscbc::make(g, [&] {
              gpu::DeviceCompNscbc::Config c{rhoInf, 0.0, pInf, gam};
              c.etaIn = 100.0;
              c.betaTrans = betaTrans;
              c.betaAuto = abc == 1;
              c.betaModal = abc == 2;
              return c;
          }())),
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

    void set(auto prim)
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

    // margin = cells skipped at every face before the vorticity scan. The x
    // extremes are NOT periodic neighbours of each other, so a wrapped central
    // difference there subtracts the outflow row from the inflow row: an O(amp)
    // difference over one cell, which reads as a vorticity of O(amp/h) and grows
    // as the grid refines. margin > 0 is the honest interior scan.
    Left measure(int margin = 0)
    {
        auto h = CompState::zeros(g);
        (void)ds->download(h);
        Left m{};
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const double rho = h.rho[c];
                if (!std::isfinite(rho) || rho <= 0.0)
                    return {1e9, 1e9, 1e9, -1, -1};
                const double u = h.mx[c] / rho, v = h.my[c] / rho;
                const double ke = 0.5 * (u * u + v * v);
                const double p = (gam - 1.0) * (h.E[c] - rho * ke);
                const double dp = p - pInf;
                m.energy += dp * dp / (2.0 * rhoInf * cInf * cInf) + rho * ke;
                if (i < margin || i >= g.nx - margin || j < margin ||
                    j >= g.ny - margin)
                    continue;
                m.entropy = std::max(
                    m.entropy, std::abs((rho - rhoInf) - dp / (cInf * cInf)));
                // NON-WRAPPING: one-sided at every extreme. Excluding the
                // boundary rows instead would exclude exactly where a vortex
                // sheet laid down by the face would live, so the sheet has to
                // be measured there, not peeled away.
                const int ipx = std::min(i + 1, g.nx - 1), imx = std::max(i - 1, 0);
                const int jpy = std::min(j + 1, g.ny - 1), jmy = std::max(j - 1, 0);
                const std::size_t jp = g.idx(i, jpy, 0), jm = g.idx(i, jmy, 0);
                const std::size_t ip = g.idx(ipx, j, 0), im = g.idx(imx, j, 0);
                const double dvdx = (h.my[ip] / h.rho[ip] - h.my[im] / h.rho[im]) /
                                    ((ipx - imx) * g.hx);
                const double dudy = (h.mx[jp] / h.rho[jp] - h.mx[jm] / h.rho[jm]) /
                                    ((jpy - jmy) * g.hy);
                const double w = std::abs(dvdx - dudy);
                if (w > m.vorticity) { m.vorticity = w; m.peakI = i; m.peakJ = j; }
            }
        return m;
    }
};

constexpr double sig = 0.03;   // ridge half-width
constexpr double amp = 0.01;   // p'/pInf

// The incident wave's own scale in each family, so the columns compare: a
// plane acoustic wave carries p' = amp pInf, no entropy at all, and no
// vorticity at all -- both of those are scaled to what a velocity
// perturbation u' = p'/(rho c) spread over the ridge width would have shown.
constexpr double pScale = amp * pInf;
constexpr double rScale = pScale / (cInf * cInf);
constexpr double wScale = (pScale / (rhoInf * cInf)) / sig;

// Measuring an oblique reflection has a WINDOW, and the published angle table
// missed it at both ends. An oblique plane ridge is a DIAGONAL band spanning
// x = xc +- (Ly/2) tan(theta), so
//   - it has not cleared until its most-upstream point has covered
//     (Lbox - xc + (Ly/2) tan(theta) + 3 sig) / (c cos theta), which the old
//     stop time of (Lbox/2 + 3 sig)/(c cos theta) does not reach past ~45 deg;
//   - and once the REFLECTION has covered Lbox it starts leaving the far face,
//     so running longer under-reports it.
// Sizing that makes the window EXIST at every angle needs
// xc > (Ly/2) tan(theta) + 3 sig -- at 75 deg a box as tall as it is wide has
// no valid measurement time at all, which is why the published table's large
// angles were mostly incident wave that had not left yet.
struct Window { double resid = 0, tClear = 0, tDrain = 0; Left left{}; };

Window residual(double thetaDeg, int abc, double betaTrans, double tMul,
                int nx = 1024, int ny = 128, double xc = 0.9 * Lbox)
{
    const double th = thetaDeg * std::numbers::pi / 180.0;
    const double ct = std::cos(th), st = std::sin(th);
    Rig r(nx, ny, abc, betaTrans);
    const double Ly = ny * r.g.hy;
    const double s0 = xc * ct + 0.5 * Ly * st;
    r.set([&](double x, double y) {
        const double s = x * ct + y * st;
        const double pp = amp * pInf * std::exp(-(s - s0) * (s - s0) / (sig * sig));
        const double q = pp / (rhoInf * cInf);
        return std::array{rhoInf + pp / (cInf * cInf), q * ct, q * st, pInf + pp};
    });
    const double xMin = std::max(0.0, xc - 0.5 * Ly * st / ct);
    Window w{};
    w.tClear = (Lbox - xMin + 3 * sig) / (cInf * ct);
    w.tDrain = Lbox / (cInf * ct);
    const double e0 = r.measure().energy;
    r.run(tMul * w.tClear);
    w.left = r.measure();
    w.resid = w.left.energy / e0;
    return w;
}

// The case a real run is in: a compact source in an open box, so EVERY face
// angle occurs at once and no single constant beta can be right across a face.
// This is the rotor's own geometry (all four faces characteristic, no
// freestream to convect anything out).
struct OpenRig
{
    Grid g;
    std::expected<gpu::DeviceComp, hipError_t> dev;
    std::expected<gpu::DeviceCompNscbc, hipError_t> nsc;
    std::expected<gpu::DeviceCompState, hipError_t> ds;
    std::expected<gpu::ReduceScratch, hipError_t> red;

    OpenRig(int n, int abc, double betaTrans, double etaIn)
        : g(Grid::make(n, n, 1, Lbox, Lbox, 1.0)),
          dev(gpu::DeviceComp::make(g, gam, 0.0)),
          nsc(gpu::DeviceCompNscbc::make(g, [&] {
              gpu::DeviceCompNscbc::Config c{rhoInf, 0.0, pInf, gam};
              c.etaIn = etaIn;
              c.openY = true;
              c.betaTrans = betaTrans;
              c.betaAuto = abc == 1;
              c.betaModal = abc == 2;
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
};

double radial(int n, int abc, double betaTrans, double etaIn = 100.0)
{
    OpenRig r(n, abc, betaTrans, etaIn);
    const double c0 = 0.5 * Lbox, sg = 0.12;
    auto h = CompState::zeros(r.g);
    double e0 = 0.0;
    for (int j = 0; j < r.g.ny; ++j)
        for (int i = 0; i < r.g.nx; ++i) {
            const std::size_t c = r.g.idx(i, j, 0);
            const double x = (i + 0.5) * r.g.hx - c0, y = (j + 0.5) * r.g.hy - c0;
            const double pp =
                amp * pInf * std::exp(-(x * x + y * y) / (sg * sg));
            h.rho[c] = rhoInf + pp / (cInf * cInf);
            h.E[c] = (pInf + pp) / (gam - 1.0);
            e0 += pp * pp / (2.0 * rhoInf * cInf * cInf);
        }
    (void)r.ds->upload(h);
    // Corner-directed rays travel sqrt(2)/2 further than face-normal ones.
    r.dev->setStageHook({});
    double t = 0.0;
    const double tEnd = 1.6 * std::sqrt(2.0) * 0.5 * Lbox / cInf;
    while (t < tEnd) {
        const double dt = std::min(tEnd - t, r.dev->stableDt(*r.ds, *r.red));
        r.dev->step(*r.ds, gpu::Real(dt));
        t += dt;
    }
    (void)r.ds->download(h);
    double e = 0.0;
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        const double rho = h.rho[c];
        if (!std::isfinite(rho) || rho <= 0.0) return 1e9;
        const double u = h.mx[c] / rho, v = h.my[c] / rho;
        const double ke = 0.5 * (u * u + v * v);
        const double dp = (gam - 1.0) * (h.E[c] - rho * ke) - pInf;
        e += dp * dp / (2.0 * rhoInf * cInf * cInf) + rho * ke;
    }
    return e / e0;
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();

    // Does the outflow row carry a REAL vortex sheet? In a quiescent box
    // lambda2..4 = u = 0, so L3 = u dv/dn vanishes and the row's tangential
    // velocity gets no normal-axis evolution at all -- the argument that it
    // must. Peeling the boundary rows off before the scan cannot answer this:
    // that is exactly where such a sheet would live. Measured instead with
    // NON-WRAPPING differences (one-sided at every extreme), margin 0.
    std::printf("peak |omega| after a 45 deg ridge has left, one-sided at the\n"
                "extremes, whole grid, vs the incident wave's vorticity scale\n");
    std::printf("  closure    E_reflected   peak |omega|   at (i,j) of (1024,128)\n");
    double vFix = 0, vAuto = 0;
    int peakI = -1;
    for (const int au : {0, 1}) {
        const Window w = residual(45.0, au, 0.5, 1.3);
        (au ? vAuto : vFix) = w.left.vorticity / wScale;
        if (!au) peakI = w.left.peakI;
        std::printf("  %-9s   %8.4f      %9.4f     (%4d,%4d)\n",
                    au ? "adaptive" : "beta 0.5", w.resid, w.left.vorticity / wScale,
                    w.left.peakI, w.left.peakJ);
    }

    // beta is a CONSTANT and the coefficient that cancels the LODI angle error
    // is not: beta_opt(theta) = cos t/(1 + cos t), of which 0.5 is only the
    // normal-incidence limit. beta = 0 must land on Engquist-Majda first order,
    // R1^2 = [(1-cos t)/(1+cos t)]^2, which is the check that the rig measures
    // a reflection and not a leftover.
    std::printf("\noblique reflection off the OUTFLOW, 1024x128, amp 1%%, quiescent,\n"
                "measured inside the window t_clear < t < t_drain (1.3 t_clear).\n"
                "FRACTION OF INCIDENT ACOUSTIC ENERGY REFLECTED.\n\n");
    std::printf("  theta  b_opt | b=0.0   b=0.2   b=0.3   b=0.4   b=0.5   b=0.6"
                " | ADAPTIVE   MODAL |  R1^2\n");
    const double betas[6] = {0.0, 0.2, 0.3, 0.4, 0.5, 0.6};
    double auto60 = 0, best60 = 0, auto0 = 0, fix0 = 0, modal0 = 0;
    double auto75 = 0, best75 = 0, modal75 = 0;
    for (const double th : {0.0, 15.0, 30.0, 45.0, 60.0, 75.0}) {
        const double c0 = std::cos(th * std::numbers::pi / 180.0);
        const double r1 = (1 - c0) / (1 + c0);
        double best = 1e9;
        std::printf("  %5.1f  %5.3f |", th, c0 / (1 + c0));
        for (double b : betas) {
            const double v = residual(th, 0, b, 1.3).resid;
            best = std::min(best, v);
            std::printf(" %7.4f", v);
        }
        const double au = residual(th, 1, 0.5, 1.3).resid;
        const double mo = residual(th, 2, 0.5, 1.3).resid;
        std::printf(" | %7.4f %7.4f | %7.4f  (adaptive %+5.1f%%, modal %+5.1f%%"
                    " vs best fixed)\n", au, mo, r1 * r1,
                    100.0 * (au - best) / best, 100.0 * (mo - best) / best);
        if (th == 60.0) { auto60 = au; best60 = best; }
        if (th == 75.0) { auto75 = au; best75 = best; modal75 = mo; }
        if (th == 0.0) { auto0 = au; fix0 = residual(th, 0, 0.5, 1.3).resid;
                         modal0 = mo; }
    }

    // A compact source in an open box: every face angle at once, which is what
    // a constant beta cannot serve and what an adaptive one is for. The stiff
    // inflow (etaIn 100) is an acoustic mirror on the x-lo face at ANY beta, so
    // the soft-inflow column is the one that isolates the transverse term.
    std::printf("\nradial pulse, all four faces characteristic, quiescent, n=256\n");
    std::printf("  etaIn |  b=0.2   b=0.3   b=0.4   b=0.5   b=0.6  |   AUTO\n");
    double autoSoft = 0, bestSoft = 1e9, fixSoft = 0;
    for (const double eta : {100.0, 0.5}) {
        std::printf("  %5.1f |", eta);
        double best = 1e9, at5 = 0;
        for (const double b : {0.2, 0.3, 0.4, 0.5, 0.6}) {
            const double v = radial(256, 0, b, eta);
            best = std::min(best, v);
            if (b == 0.5) at5 = v;
            std::printf(" %7.4f", v);
        }
        const double au = radial(256, 1, 0.5, eta);
        std::printf(" | %7.4f  (auto vs beta 0.5: %+5.1f%%)\n", au,
                    100.0 * (au - at5) / at5);
        if (eta < 1.0) { autoSoft = au; bestSoft = best; fixSoft = at5; }
    }

    std::printf("\n");
    // The outflow row is NOT the vorticity source. Some spurious vorticity is
    // real -- 0.22 of the incident scale, an order below the 1.51 the wrapped
    // metric once reported -- but its peak sits in the x-INTERIOR, hard against
    // the free-slip y wall of this rig, not on the characteristic face. So the
    // "L3 = u dv/dn vanishes at u = 0, hence the outflow row freezes a vortex
    // sheet" argument does not survive being measured: the row is clean and the
    // lateral wall closure is where the residual lives.
    check(vFix < 0.30 && vAuto < 0.30,
          "spurious vorticity is an order below the retracted wrapped-metric value");
    check(peakI > 8 && peakI < 1024 - 9,
          "and its peak is NOT on an x boundary row -- the outflow lays down no sheet");
    check(std::abs(auto0 - fix0) < 1e-12,
          "a normal-incidence face is unchanged by the adaptive coefficient");
    check(std::abs(modal0 - fix0) < 1e-6,
          "...and by the modal one (a constant survives DCT-II/III exactly)");
    // THE reason the modal form exists. A per-cell ratio is formed from a
    // field that at grazing incidence is a superposition -- the incident wave,
    // its wall images and a reflection that is no longer small -- so the angle
    // it reads is an average and the coefficient degrades. The DCT basis is
    // the exact transverse eigenbasis of a free-slip-walled face, so each mode
    // carries ONE kt and the ratio is formed inside it. At 75 deg that is the
    // whole difference between useless and as good as a hand-tuned constant.
    check(modal75 <= 1.05 * best75,
          "at 75 deg the modal coefficient still matches the best constant beta");
    check(modal75 < 0.85 * auto75,
          "...where the per-cell one does not: modal is the grazing-incidence form");
    check(auto60 <= 1.02 * best60,
          "at 60 deg the adaptive coefficient matches the best constant beta");
    check(autoSoft < fixSoft,
          "on an all-angle face the adaptive coefficient beats beta 0.5");
    check(autoSoft <= 1.05 * bestSoft,
          "...and comes within 5% of the best constant beta, without being told it");
    return dnstest::finish();
}
