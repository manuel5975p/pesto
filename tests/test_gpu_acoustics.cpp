// The acoustics of the interior scheme, with no boundary anywhere in the
// measurement: a linear acoustic wave in a PERIODIC box, carried k times
// around, compared with itself.
//
// A travelling plane wave of wavelength L in a periodic box returns to its
// exact initial state after each traversal, so after k of them the amplitude
// ratio IS the numerical dissipation and the phase shift IS the dispersion --
// no analytic solution to discretize, no boundary condition in the loop, no
// reference grid. Reported per wavelength travelled, so the rows compare.
//
// What each column answers:
//   c_num/c_exact  does sound travel at sqrt(gamma p / rho), and at u + c
//                  when the box drifts (the wave speed the rotor's tip Mach
//                  and the whole acoustic dt limit are quoted against)
//   amp ratio      how much of the wave the scheme eats per wavelength; a
//                  skew-symmetric central flux should eat ~nothing, so this
//                  isolates the RK3 time error
//   phase err      dispersion per wavelength: the error that turns a pulse
//                  into a wave train, which is what a badly resolved acoustic
//                  source looks like in a frame
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
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

constexpr double gam = 1.4, rhoInf = 1.0, cInf = 1.0, Lbox = 1.0;
constexpr double pInf = rhoInf * cInf * cInf / gam;
constexpr double twoPi = 2.0 * std::numbers::pi;

// Amplitude and phase of the first Fourier mode of p - pInf along x, averaged
// over the (uniform) y rows. Fitting the mode rather than reading a peak keeps
// the measurement free of where the sample points happen to fall.
struct Mode { double amp, phase; };

Mode fitMode(const Grid& g, const std::vector<gpu::Real>& p, int modes)
{
    double re = 0.0, im = 0.0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const double x = (i + 0.5) * g.hx;
            const double d = double(p[g.idx(i, j, 0)]) - pInf;
            re += d * std::cos(modes * twoPi * x / Lbox);
            im += d * std::sin(modes * twoPi * x / Lbox);
        }
    const double n = double(g.nx) * double(g.ny);
    re *= 2.0 / n;
    im *= 2.0 / n;
    return {std::hypot(re, im), std::atan2(im, re)};
}

struct Result
{
    double ampRatio;  // per wavelength travelled
    double phaseErr;  // radians per wavelength travelled, signed
    double speedRatio; // measured (u + c) / exact
};

// `modes` acoustic wavelengths across a periodic box of nx cells (so points
// per wavelength = nx / modes), carried `laps` wavelengths. laps < 1 keeps the
// accumulated phase error inside one branch of atan2, which is what makes the
// badly resolved modes measurable at all -- at 5 laps a 2-cell wave has
// wrapped and the number would be a lie.
Result travel(int nx, int modes, double mach, double amp, double laps,
              double cfl)
{
    const double uMean = mach * cInf;
    const double lam = Lbox / modes;
    Grid g = Grid::make(nx, 4, 1, Lbox, 4 * (Lbox / nx), 1.0);
    auto dev = gpu::DeviceComp::make(g, gam, 0.0);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !red) std::exit(EXIT_FAILURE);

    auto h = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double x = (i + 0.5) * g.hx;
            const double pp = amp * pInf * std::sin(twoPi * x / lam);
            const double rho = rhoInf + pp / (cInf * cInf);
            const double u = uMean + pp / (rhoInf * cInf);
            h.rho[c] = rho;
            h.mx[c] = rho * u;
            h.E[c] = (pInf + pp) / (gam - 1.0) + 0.5 * rho * u * u;
        }
    (void)ds->upload(h);

    std::vector<gpu::Real> hp(g.size());
    dev->primitives(*ds);
    (void)dev->p().download(hp.data());
    const Mode m0 = fitMode(g, hp, modes);

    // Exactly `laps` wavelengths at the linear wave speed: the wave should come
    // back to m0 in both amplitude and phase.
    const double tEnd = laps * lam / (cInf + uMean);
    double t = 0.0;
    while (t < tEnd) {
        const double dt = std::min(tEnd - t, dev->stableDt(*ds, *red, cfl));
        dev->step(*ds, gpu::Real(dt));
        t += dt;
    }
    dev->primitives(*ds);
    (void)dev->p().download(hp.data());
    const Mode m1 = fitMode(g, hp, modes);

    auto wrap = [](double a) {
        while (a > std::numbers::pi) a -= twoPi;
        while (a < -std::numbers::pi) a += twoPi;
        return a;
    };
    // A wave running late is a wave travelling slow: the phase it is short of
    // the exact -2 pi laps, spread over `laps` wavelengths, is the fractional
    // speed deficit. A mode the scheme does not transport at all reads the
    // whole expected shift as error, which is the point.
    // fitMode's atan2(sum d sin, sum d cos) advances by +omega t for a wave
    // running in +x, so the exact shift is +2 pi laps -- a sign that is
    // invisible at whole laps and decides the answer at a quarter of one.
    const double err = wrap(wrap(m1.phase - m0.phase) - wrap(twoPi * laps));
    return {std::pow(m1.amp / m0.amp, 1.0 / laps), err / laps,
            1.0 + err / (twoPi * laps)};
}

double order(double coarse, double fine) { return std::log2(coarse / fine); }

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    constexpr int laps = 5;
    constexpr double amp = 1e-3;

    constexpr int nx = 128;
    std::printf("linear acoustic wave, periodic box of %d cells, amp 1e-3,\n"
                "%d wavelengths travelled, quiescent, cfl 0.4 -- per wavelength\n",
                nx, laps);
    std::printf("   ppw   amp ratio    phase err     c_num/c_exact\n");
    // One grid, the wave scaled instead: the resolution a feature gets is set
    // by its own size, which is how a rotor's sources land on a fixed mesh.
    const int modes[5] = {1, 2, 4, 8, 16};
    double ph[5] = {}, ar[5] = {};
    for (int i = 0; i < 5; ++i) {
        const Result r = travel(nx, modes[i], 0.0, amp, laps, 0.4);
        ph[i] = std::abs(r.phaseErr);
        ar[i] = r.ampRatio;
        std::printf("  %5.1f    %8.6f    %+9.2e      %10.7f\n",
                    double(nx) / modes[i], r.ampRatio, r.phaseErr,
                    r.speedRatio);
    }
    std::printf("   dispersion order  %4.2f  %4.2f  %4.2f  %4.2f\n",
                order(ph[4], ph[3]), order(ph[3], ph[2]), order(ph[2], ph[1]),
                order(ph[1], ph[0]));

    // The mesh scale, where the phase error per wavelength is O(1) and would
    // wrap over a long ride: a quarter wavelength of travel instead.
    std::printf("\ndown at the mesh scale -- quarter wavelength travelled\n");
    std::printf("   ppw   amp ratio    phase err     c_num/c_exact\n");
    const int fine[6] = {8, 16, 21, 32, 43, 64};
    double arFine[6] = {}, spFine[6] = {};
    for (int i = 0; i < 6; ++i) {
        const Result r = travel(nx, fine[i], 0.0, amp, 0.25, 0.4);
        arFine[i] = r.ampRatio;
        spFine[i] = r.speedRatio;
        std::printf("  %5.1f    %8.6f    %+9.2e      %10.7f\n",
                    double(nx) / fine[i], r.ampRatio, r.phaseErr,
                    r.speedRatio);
    }

    std::printf("\nwith a mean flow -- the wave must ride it exactly (16 ppw)\n");
    std::printf("  mach   amp ratio    phase err     c_num/c_exact\n");
    for (const double m : {0.0, 0.2, 0.5}) {
        const Result r = travel(nx, 8, m, amp, laps, 0.4);
        std::printf("  %4.2f    %8.6f    %+9.2e      %10.7f\n", m, r.ampRatio,
                    r.phaseErr, r.speedRatio);
    }

    std::printf("\ntime error alone: same grid and wave, smaller cfl (16 ppw)\n");
    std::printf("   cfl   amp ratio    phase err\n");
    double phCfl[3] = {};
    const double cfls[3] = {0.4, 0.2, 0.1};
    for (int i = 0; i < 3; ++i) {
        const Result r = travel(nx, 8, 0.0, amp, laps, cfls[i]);
        phCfl[i] = std::abs(r.phaseErr);
        std::printf("  %4.2f    %8.6f    %+9.2e\n", cfls[i], r.ampRatio,
                    r.phaseErr);
    }
    std::printf("   order in dt       %4.2f  %4.2f\n", order(phCfl[0], phCfl[1]),
                order(phCfl[1], phCfl[2]));

    std::printf("\namplitude: linear response and steepening (16 ppw)\n");
    std::printf("      amp   amp ratio    phase err\n");
    for (const double a : {1e-4, 1e-3, 1e-2, 1e-1}) {
        const Result r = travel(nx, 8, 0.0, a, laps, 0.4);
        std::printf("  %7.4f    %8.6f    %+9.2e\n", a, r.ampRatio, r.phaseErr);
    }

    std::printf("\n");
    // Sound must travel at sqrt(gamma p / rho) once it is resolved at all.
    check(ph[2] < 0.05, "phase error < 0.05 rad per wavelength at 32 ppw");
    check(ph[2] < ph[3] && ph[3] < ph[4], "dispersion falls with resolution");
    // A skew-symmetric central flux carries no upwind dissipation: what a
    // resolved wave loses is time error, and that is near nothing here.
    check(ar[2] > 0.999, "less than 0.1% amplitude lost per wavelength at 32 ppw");
    check(phCfl[2] <= 1.05 * phCfl[0],
          "time error is negligible against the spatial one at cfl 0.4");
    // THE property that decides what a rotor's grid noise does. There is no
    // dissipation to lean on down at the mesh scale either: badly resolved
    // acoustic content does not decay, it stops travelling. The only sinks
    // are molecular nu and the SGS model.
    check(arFine[5] > 0.99, "a 2-cell wave keeps >99% of its amplitude");
    check(spFine[5] < 0.2, "...while barely propagating at all");
    check(arFine[3] > 0.98 && spFine[3] < 0.8,
          "a 4-cell wave is undamped and travels well under its own speed");
    return dnstest::finish();
}
