// How ABSORBING the characteristic far field actually is, as a number: the
// reflection coefficient of a plane acoustic pulse leaving the OUTFLOW and the
// one leaving the stiff INFLOW, swept over mean Mach, grid spacing and
// amplitude, plus the all-face residual of a radial pulse in a QUIESCENT box.
//
// gpu_comp_nscbc pins that a pulse exits at all (bands of 15-20%); this test
// asks how well, and in particular what happens as the mean flow goes away.
// The near-static rotor is exactly that corner: at advance ratio ~0 the face
// normal velocity is ~0, so the outgoing wave is not convected out, and the
// inflow relaxation the production default keeps stiff (etaIn 100, sized
// against freestream sag, not against acoustics) has nothing to hide behind.
//
// Amplitudes are quoted as p'/pInf; the acoustic velocity ratio u'/c = amp/gam
// is then independent of the mean Mach, so a row is comparable across it.
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
#include <vector>

using namespace dns;

namespace {

using dnstest::check;

// Peak of each Kovasznay mode left in the box: pressure (acoustic), the
// density not explained by pressure (entropy), and the vorticity.
struct ModeSplit { double acoustic = 0, entropy = 0, vorticity = 0; };

constexpr double gam = 1.4, rhoInf = 1.0, cInf = 1.0;
constexpr double pInf = rhoInf * cInf * cInf / gam;
constexpr double Lbox = 2.56; // fixed physical box: refining n refines h alone

// x-characteristic boundaries, free-slip y walls (sigMax = 0 = no sponge).
// uMean = 0 is legal here: nothing in the LODI closure divides by it.
struct PlaneRig
{
    Grid g;
    std::expected<gpu::DeviceComp, hipError_t> dev;
    std::expected<gpu::DeviceCompFar, hipError_t> far;
    std::expected<gpu::DeviceCompNscbc, hipError_t> nsc;
    std::expected<gpu::DeviceCompState, hipError_t> ds;
    std::expected<gpu::ReduceScratch, hipError_t> red;

    std::optional<gpu::DeviceCompSink> snk;

    // strip = --nscbc-strip N: the production hybrid is a sponge band of N
    // cells at sigMax 0.5 INSIDE the characteristic rows (runner_comp.hip:124).
    // sink = --nscbc-sink N: the scale-selective band (gpu/comp_sink),
    // composed exactly as runner_comp does -- additive under the overwrite.
    PlaneRig(int nx, double uMean, double etaIn, int strip = 0, int ny = 8,
             double betaTrans = 0.5, int sink = 0, double etaInShear = -1)
        : g(Grid::make(nx, ny, 1, Lbox, ny * (Lbox / nx), 1.0)),
          dev(gpu::DeviceComp::make(g, gam, 0.0)),
          far(gpu::DeviceCompFar::make(
              g, {rhoInf, uMean, pInf, gam, strip > 0 ? strip : 8,
                  strip > 0 ? 0.5 : 0.0, false, false})),
          nsc(gpu::DeviceCompNscbc::make(g, [&] {
              gpu::DeviceCompNscbc::Config c{rhoInf, uMean, pInf, gam};
              c.etaIn = etaIn;
              c.betaTrans = betaTrans;
              c.etaInShear = etaInShear;
              return c;
          }())),
          ds(gpu::DeviceCompState::make(g.size())),
          red(gpu::ReduceScratch::make())
    {
        if (!dev || !far || !nsc || !ds || !red) std::exit(EXIT_FAILURE);
        if (sink > 0) {
            auto s = gpu::DeviceCompSink::make(
                g, {.nb = sink, .alpha = 1.0, .cInf = cInf});
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

    // Acoustic energy in the box, sum of p'^2/(2 rho c^2) + rho |u'|^2 / 2.
    // Unlike a peak-pressure residual this does not care where the wave is or
    // how many pieces it is in, so it survives the y-wall bounces an oblique
    // ridge makes on its way out.
    double acousticEnergy(double uMean)
    {
        auto h = CompState::zeros(g);
        (void)ds->download(h);
        double e = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c) {
            const double rho = h.rho[c];
            if (!std::isfinite(rho) || rho <= 0.0) return 1e9;
            const double u = h.mx[c] / rho - uMean, v = h.my[c] / rho;
            const double ke = 0.5 * (u * u + v * v);
            const double p = (gam - 1.0) * (h.E[c] - rho * (ke + u * uMean +
                                                            0.5 * uMean * uMean));
            const double dp = p - pInf;
            e += dp * dp / (2.0 * rhoInf * cInf * cInf) + rho * ke;
        }
        return e;
    }

    // Kovasznay split of what is left in the box. The three characteristic
    // families leave a boundary by completely different routes: acoustics at
    // u +- c, entropy and vorticity at u. In a QUIESCENT box the latter two do
    // not move at all, so anything spurious the face makes in them is frozen
    // there for good -- and no acoustic ABC, Mur's or otherwise, can reach it.
    // Worth knowing which one the residual is before building anything.
    // margin: cells peeled off every face before the scan. It must be >= 2.
    // The x extremes are NOT periodic neighbours of each other, so a wrapped
    // central difference there subtracts the outflow row from the inflow row --
    // an O(amp) difference over one cell, which reads as a vorticity of
    // O(amp/h) and GROWS as the grid refines. Measured 2026-07-29: at margin 0
    // a 45 deg ridge reported a peak of 1.51 sitting at i = nx-1; at margin 2
    // it is 0.006 sitting at random interior points, i.e. the noise floor. The
    // whole-grid form once made this test conclude the face was a vorticity
    // source, which it is not.
    ModeSplit modeSplit(int margin = 2)
    {
        auto h = CompState::zeros(g);
        (void)ds->download(h);
        ModeSplit m{};
        for (int j = margin; j < g.ny - margin; ++j)
            for (int i = margin; i < g.nx - margin; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const double rho = h.rho[c];
                const double u = h.mx[c] / rho, v = h.my[c] / rho;
                const double p = (gam - 1.0) *
                                 (h.E[c] - 0.5 * rho * (u * u + v * v));
                // Entropy mode: the density that is NOT explained by pressure.
                m.entropy = std::max(m.entropy,
                                     std::abs((rho - rhoInf) -
                                              (p - pInf) / (cInf * cInf)));
                m.acoustic = std::max(m.acoustic, std::abs(p - pInf));
                const std::size_t jp = g.idx(i, (j + 1) % g.ny, 0);
                const std::size_t jm = g.idx(i, (j + g.ny - 1) % g.ny, 0);
                const std::size_t ip = g.idx((i + 1) % g.nx, j, 0);
                const std::size_t im = g.idx((i + g.nx - 1) % g.nx, j, 0);
                const double dvdx =
                    (h.my[ip] / h.rho[ip] - h.my[im] / h.rho[im]) / (2 * g.hx);
                const double dudy =
                    (h.mx[jp] / h.rho[jp] - h.mx[jm] / h.rho[jm]) / (2 * g.hy);
                m.vorticity = std::max(m.vorticity, std::abs(dvdx - dudy));
            }
        return m;
    }

    // Peak |dv/dx - du/dy| over i in [iLo, iHi], NON-WRAPPING one-sided
    // differences at every extreme, margin 0 -- the face-row scan a boundary
    // claim needs (a wrapped difference across the x extremes fabricates
    // O(amp/h); see modeSplit).
    double vortPeak(int iLo, int iHi)
    {
        auto h = CompState::zeros(g);
        (void)ds->download(h);
        auto vel = [&](int i, int j, int comp) {
            const std::size_t c = g.idx(i, j, 0);
            return (comp ? h.my[c] : h.mx[c]) / h.rho[c];
        };
        double pk = 0.0;
        for (int j = 0; j < g.ny; ++j)
            for (int i = iLo; i <= iHi; ++i) {
                const int ip = std::min(i + 1, g.nx - 1), im = std::max(i - 1, 0);
                const int jp = std::min(j + 1, g.ny - 1), jm = std::max(j - 1, 0);
                const double dvdx =
                    (vel(ip, j, 1) - vel(im, j, 1)) / ((ip - im) * g.hx);
                const double dudy =
                    (vel(i, jp, 0) - vel(i, jm, 0)) / ((jp - jm) * g.hy);
                pk = std::max(pk, std::abs(dvdx - dudy));
            }
        return pk;
    }

    // max |p - pInf| over the domain, normalized by the incident amplitude.
    double pressureResidual(double amp)
    {
        dev->primitives(*ds);
        std::vector<gpu::Real> hp(g.size());
        (void)dev->p().download(hp.data());
        double r = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c) {
            if (!std::isfinite(double(hp[c]))) return 1e9;
            r = std::max(r, std::abs(double(hp[c]) - pInf));
        }
        return r / (amp * pInf);
    }
};

// A single travelling plane wave, exact for the linearized equations: dir = +1
// leaves through the outflow, dir = -1 through the stiff inflow. Integrate
// until the incident pulse plus three sigmas of its tail have cleared the face
// (1.3x that), which leaves the reflected wave -- slower by (c-u)/(c+u) -- well
// inside the box, so the residual it leaves is the reflection and not the
// pulse still on its way out.
double planeReflection(int nx, double mach, double amp, int dir,
                       double etaIn = 100.0, int strip = 0, int sink = 0,
                       double etaInShear = -1)
{
    const double uMean = mach * cInf;
    PlaneRig r(nx, uMean, etaIn, strip, 8, 0.5, sink, etaInShear);
    const double x0 = 0.5 * Lbox, sig = 0.08;
    r.set([&](double x, double) {
        const double pp =
            amp * pInf * std::exp(-(x - x0) * (x - x0) / (sig * sig));
        return std::array{rhoInf + pp / (cInf * cInf),
                          uMean + dir * pp / (rhoInf * cInf), 0.0, pInf + pp};
    });
    const double speed = cInf + dir * uMean;
    r.run(1.3 * (0.5 * Lbox + 3 * sig) / speed);
    return r.pressureResidual(amp);
}

// Angle of incidence is measured in gpu_nscbc_order, NOT here. It needs a rig
// this one cannot provide: an oblique plane ridge is a DIAGONAL band spanning
// x = xc +- (Ly/2) tan(theta), so it has not cleared the box until its
// most-upstream point has covered (Lbox - xc + (Ly/2) tan(theta) + 3 sig) /
// (c cos theta) -- while the reflection starts leaving the FAR face after
// Lbox / (c cos theta). In a box as tall as it is wide those two cross at
// ~45 deg and there is no valid measurement time past it. This file's rig is
// 256x8; every angle number it used to print past 45 deg was mostly incident
// wave that had not left yet.

// The inflow as a VORTICITY writer: a resolved vortex parked three core
// widths off the stiff x-lo face, quiescent. Its induced tangential velocity
// loads the face rows, and the lambda = u relax etaInS c/L (v - 0) writes
// vorticity onto them at a rate proportional to etaInS -- the inflow twin of
// the measured lateral-face source. Reported: peak |omega| on the face rows
// (i <= 2; the core's own tail there is e^-9 A) over the core's peak, after
// one box acoustic time.
double inflowVortexWrite(double etaInShear)
{
    PlaneRig r(256, 0.0, 100.0, 0, 64, 0.5, 0, etaInShear);
    const double h = r.g.hx, kw = 8 * h, Ly = r.g.ny * r.g.hy;
    const double x0 = 3 * kw, y0 = 0.5 * Ly, A = 0.05 * cInf;
    r.set([&](double x, double y) {
        const double dx = x - x0, dy = y - y0;
        const double e = A * std::exp(-(dx * dx + dy * dy) / (kw * kw));
        return std::array{rhoInf, 2 * dy / kw * e, -2 * dx / kw * e, pInf};
    });
    const double w0 = r.vortPeak(0, r.g.nx - 1);
    r.run(Lbox / cInf);
    return r.vortPeak(0, 2) / w0;
}

// All four faces characteristic, quiescent box: the rotor's own case. etaIn is
// the knob under test -- 100 is the production inflow, 2 is what the existing
// unit rig softens it to.
struct OpenRig
{
    Grid g;
    std::expected<gpu::DeviceComp, hipError_t> dev;
    std::expected<gpu::DeviceCompNscbc, hipError_t> nsc;
    std::expected<gpu::DeviceCompState, hipError_t> ds;
    std::expected<gpu::ReduceScratch, hipError_t> red;

    OpenRig(int n, double uMean, double etaIn)
        : g(Grid::make(n, n, 1, Lbox, Lbox, 1.0)),
          dev(gpu::DeviceComp::make(g, gam, 0.0)),
          nsc(gpu::DeviceCompNscbc::make(g, [&] {
              gpu::DeviceCompNscbc::Config c{rhoInf, uMean, pInf, gam};
              c.etaIn = etaIn;
              c.openY = true;
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
};

double radialResidual(int n, double mach, double amp, double etaIn)
{
    const double uMean = mach * cInf;
    OpenRig r(n, uMean, etaIn);
    const double c0 = 0.5 * Lbox, sig = 0.12;
    auto h = CompState::zeros(r.g);
    for (int j = 0; j < r.g.ny; ++j)
        for (int i = 0; i < r.g.nx; ++i) {
            const std::size_t c = r.g.idx(i, j, 0);
            const double x = (i + 0.5) * r.g.hx - c0, y = (j + 0.5) * r.g.hy - c0;
            const double pp =
                amp * pInf * std::exp(-(x * x + y * y) / (sig * sig));
            const double rho = rhoInf + pp / (cInf * cInf), p = pInf + pp;
            h.rho[c] = rho;
            h.mx[c] = rho * uMean;
            h.E[c] = p / (gam - 1.0) + 0.5 * rho * uMean * uMean;
        }
    (void)r.ds->upload(h);
    // Corner-directed rays travel sqrt(2)/2 further than face-normal ones.
    r.run(1.6 * std::sqrt(2.0) * 0.5 * Lbox / (cInf - std::abs(uMean)));
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

// log-log slope of R over a refinement pair: the observed order of absorption.
double order(double coarse, double fine) { return std::log2(coarse / fine); }

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();

    std::printf("plane pulse, amp 1%%, n = 256, etaIn 100 (production)\n");
    std::printf("   mach    R_outflow    R_inflow\n");
    double outAtM[4] = {}, inAtM[4] = {};
    const double machs[4] = {0.2, 0.05, 0.01, 0.0};
    for (int i = 0; i < 4; ++i) {
        outAtM[i] = planeReflection(256, machs[i], 0.01, +1);
        inAtM[i] = planeReflection(256, machs[i], 0.01, -1);
        std::printf("  %5.3f    %9.4f    %8.4f\n", machs[i], outAtM[i],
                    inAtM[i]);
    }

    std::printf("\ngrid convergence, amp 1%%, etaIn 100\n");
    std::printf("   mach     n     R_outflow    R_inflow\n");
    double outN[2][3] = {}, inN[2][3] = {};
    const double convMach[2] = {0.2, 0.01};
    const int ns[3] = {128, 256, 512};
    for (int m = 0; m < 2; ++m) {
        for (int k = 0; k < 3; ++k) {
            outN[m][k] = planeReflection(ns[k], convMach[m], 0.01, +1);
            inN[m][k] = planeReflection(ns[k], convMach[m], 0.01, -1);
            std::printf("  %5.3f  %4d     %9.4f    %8.4f\n", convMach[m], ns[k],
                        outN[m][k], inN[m][k]);
        }
        std::printf("         order      %5.2f       %5.2f   (n 128->256)\n",
                    order(outN[m][0], outN[m][1]), order(inN[m][0], inN[m][1]));
        std::printf("         order      %5.2f       %5.2f   (n 256->512)\n",
                    order(outN[m][1], outN[m][2]), order(inN[m][1], inN[m][2]));
    }

    // R is normalized by the incident amplitude, so a FIXED residual floor
    // (interior scheme noise, FP32) shows up as R growing at small amp. The
    // absolute column separates the two.
    std::printf("\namplitude, mach 0.2, n = 256, etaIn 100\n");
    std::printf("      amp    R_outflow   |dp|/pInf    R_inflow\n");
    double outA[4] = {}, inA[4] = {};
    const double amps[4] = {1e-4, 1e-3, 1e-2, 1e-1};
    for (int i = 0; i < 4; ++i) {
        outA[i] = planeReflection(256, 0.2, amps[i], +1);
        inA[i] = planeReflection(256, 0.2, amps[i], -1);
        std::printf("  %7.4f    %9.4f   %9.2e    %8.4f\n", amps[i], outA[i],
                    outA[i] * amps[i], inA[i]);
    }

    // The stiffness the inflow needs against a body's blockage sag (etaIn 100,
    // sized at the Re60 cylinder) against what it costs acoustically. A rotor
    // in still air has no freestream to sag.
    std::printf("\ninflow stiffness, amp 1%%, n = 256\n");
    std::printf("   etaIn   R_inflow(M 0)  R_inflow(M 0.2)  R_allface(M 0)\n");
    const double etas[5] = {0.5, 2.0, 10.0, 30.0, 100.0};
    double inEta[5] = {}, radEta[5] = {};
    for (int i = 0; i < 5; ++i) {
        inEta[i] = planeReflection(256, 0.0, 0.01, -1, etas[i]);
        const double inMoving = planeReflection(256, 0.2, 0.01, -1, etas[i]);
        radEta[i] = radialResidual(256, 0.0, 0.01, etas[i]);
        std::printf("  %6.1f      %9.4f        %9.4f       %9.4f\n", etas[i],
                    inEta[i], inMoving, radEta[i]);
    }

    // What the user can reach today: --nscbc-strip N, the sponge band inside
    // the characteristic rows. It is the only lever on the inflow mirror
    // while etaIn stays a compile-time 100.
    std::printf("\n--nscbc-strip against the inflow, amp 1%%, n = 256, etaIn 100\n");
    std::printf("   strip   R_inflow(M 0)  R_outflow(M 0)\n");
    const int strips[5] = {0, 4, 8, 16, 32};
    double inStrip[5] = {};
    for (int i = 0; i < 5; ++i) {
        inStrip[i] = planeReflection(256, 0.0, 0.01, -1, 100.0, strips[i]);
        const double out = planeReflection(256, 0.0, 0.01, +1, 100.0, strips[i]);
        std::printf("  %5d       %9.4f       %9.4f\n", strips[i], inStrip[i],
                    out);
    }

    // The scale-selective sink against the SAME faces. A resolved pulse
    // (sigma = 8h Gaussian: spectrally empty above kh ~ pi/4) must pass the
    // band untouched in BOTH directions -- if these rows move, the sink is
    // not selective and every reflection coefficient above is invalidated.
    // Conversely the sink must NOT fix the inflow mirror: that reflection is
    // resolved-scale, and it belongs to etaIn (or the strip), not to a
    // sub-6dx band. A sink that helped here would be eating the near field.
    std::printf("\n--nscbc-sink against the same faces, amp 1%%, n = 256, etaIn 100\n");
    std::printf("   sink   R_inflow(M 0)  R_outflow(M 0)\n");
    const int sinks[3] = {0, 8, 16};
    double inSink[3] = {}, outSink[3] = {};
    for (int i = 0; i < 3; ++i) {
        inSink[i] = planeReflection(256, 0.0, 0.01, -1, 100.0, 0, sinks[i]);
        outSink[i] = planeReflection(256, 0.0, 0.01, +1, 100.0, 0, sinks[i]);
        std::printf("  %5d       %9.4f       %9.4f\n", sinks[i], inSink[i],
                    outSink[i]);
    }

    // The inflow lambda = u family rate: what the face WRITES under a
    // vortical load, per rate. -1 = follow etaIn (100, the shipped inflow).
    std::printf("\ninflow vorticity written under a parked vortex, "
                "--nscbc-etain-shear scan\n");
    std::printf("   etaInShear   face-row peak|omega|/w0\n");
    const double shears[4] = {-1.0, 10.0, 1.0, 0.25};
    double wIn[4] = {};
    for (int i = 0; i < 4; ++i) {
        wIn[i] = inflowVortexWrite(shears[i]);
        std::printf("   %8.2f       %9.4f\n", shears[i], wIn[i]);
    }

    // Which family owns the acoustic mirror? A normal wave carries both a
    // density part (L2's to relax) and a normal-velocity part (L5's). Soften
    // the lambda = u family ALONE and read how much of the 0.81 follows it.
    std::printf("\nacoustic mirror split: R_inflow(M 0), etaIn 100 held\n");
    std::printf("   etaInShear   R_inflow\n");
    double rShear[3] = {};
    const double rsh[3] = {-1.0, 0.5, 0.25};
    for (int i = 0; i < 3; ++i) {
        rShear[i] = planeReflection(256, 0.0, 0.01, -1, 100.0, 0, 0, rsh[i]);
        std::printf("   %8.2f    %9.4f\n", rsh[i], rShear[i]);
    }

    std::printf("\nradial pulse, all four faces characteristic, amp 1%%\n");
    std::printf("   mach  etaIn     n    R_allface\n");
    double radStiff = 0, radSoft = 0;
    for (const double eta : {2.0, 100.0})
        for (const int n : {128, 256}) {
            const double R = radialResidual(n, 0.0, 0.01, eta);
            if (n == 256) (eta > 50 ? radStiff : radSoft) = R;
            std::printf("  %5.3f  %5.0f  %4d    %9.4f\n", 0.0, eta, n, R);
        }
    const double radMoving = radialResidual(256, 0.2, 0.01, 100.0);
    std::printf("  %5.3f  %5.0f  %4d    %9.4f\n", 0.2, 100.0, 256, radMoving);

    std::printf("\n");
    // The outflow is the face the scheme is built for: a normal plane wave is
    // exact in the linear limit, so what is left is the soft pressure anchor,
    // and it must stay a few percent at every Mach INCLUDING zero.
    check(outAtM[0] < 0.05, "outflow reflects < 5% at mach 0.2");
    check(outAtM[3] < 0.10, "outflow reflects < 10% in a quiescent box");
    // In the linear regime the coefficient must not depend on amplitude; at
    // 10% it legitimately does (a pulse that steep steepens further), and
    // below ~1e-3 the fixed residual floor takes over, so the band is on the
    // two decades in between.
    check(std::abs(outA[2] - outA[1]) < 1.0 * outA[1],
          "outflow reflection is amplitude-independent, amp 1e-3 to 1e-2");
    // Refinement must not make it worse (the BC error floor may dominate, so
    // this is a floor test, not an order test).
    check(outN[0][2] <= 1.2 * outN[0][0], "outflow reflection does not grow with n");
    check(radStiff < 0.25, "radial pulse leaves a quiescent box, all faces");
    check(radSoft <= radStiff, "soft inflow absorbs at least as well as stiff");
    // THE finding: the production inflow is an acoustic mirror, and unlike
    // every other face its reflection does not fall with h. Anything that
    // radiates toward x-lo in a near-static box gets ~80% of it back.
    check(inAtM[3] > 0.5, "stiff inflow reflects most of a normal wave (etaIn 100)");
    check(inEta[0] < 0.25, "a soft inflow (etaIn 0.5) absorbs");
    check(radEta[0] < 0.5 * radEta[4],
          "softening the inflow halves the all-face residual or better");
    // --nscbc-strip is the lever that exists today, and it is the fix: a band
    // of ny/64 (the Re > 800 minimum) already turns the mirror into a few
    // percent, and ny/16 into a fraction of one.
    check(inStrip[1] < 0.15, "--nscbc-strip 4 kills the inflow mirror");
    check(inStrip[3] < 0.01, "--nscbc-strip 16 absorbs the inflow to < 1%");
    // The sink is selective at the system level: a resolved pulse crosses its
    // band unchanged both ways, so it neither degrades the outflow nor
    // "fixes" the inflow mirror -- that reflection is resolved-scale and is
    // etaIn's (or the strip's) to absorb, which is exactly the division of
    // labour docs/nscbc.md prescribes.
    check(std::abs(outSink[2] - outSink[0]) < 0.01,
          "--nscbc-sink 16 leaves the resolved outflow reflection unchanged");
    check(std::abs(inSink[2] - inSink[0]) < 0.05,
          "--nscbc-sink does not (and must not) fix the resolved inflow mirror");
    // The per-family split, measured: softening the lambda = u family 400x
    // moves the acoustic mirror by 0.1% -- the mirror is the L5 sag guard
    // ITSELF, so no per-family softening can fix it while u stays pinned
    // (only --nscbc-etain or the strip can). What the family rate does own is
    // what the face WRITES under a vortical load, and that falls monotonely
    // as it softens. Its real customer is turbulent inflow, where L3/L4 at
    // 100 would clamp every injected eddy.
    check(rShear[2] > 0.75,
          "the acoustic mirror is L5's own: soft shear cannot fix it");
    check(wIn[3] <= wIn[0] && wIn[2] <= wIn[1],
          "the vorticity the inflow writes falls as the family rate softens");

    return dnstest::finish();
}
