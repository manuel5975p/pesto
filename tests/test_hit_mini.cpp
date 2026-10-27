// Forced HIT at 64^3 in box units (L = 2pi, u'_target = 1, h = 1.5 eta):
// after spin-up the fixed-eps controller must give a stationary state with
// <eps_visc> ~ eps_f and u' near the linear-forcing prediction.
// Validation (literature): the time-averaged compensated spectrum
// C = <E(k) k^{5/3} eps^{-2/3}> over k in [4,10] must land in [0.7, 1.6].
// The asymptotic Kolmogorov constant is ~1.5 (Pope), but at 64^3 with
// h = 1.5 eta the band reaches k*eta = 0.26..0.65, i.e. into the dissipation
// rolloff: Pope's model spectrum (beta = 5.2, c_eta = 0.4) predicts a band
// mean of ~0.78 here; the bottleneck bump pushes the measured value above
// that (0.98 on 2026-07-17). Band = [0.9x, 2x] the model prediction -- a
// factor-2 normalization error still fails loudly.
// Window 3 + 3 eddy turnovers (was 6 + 6, 43.7 s): the same bands, half the
// steps. Measured 2026-10-27: 746 steps, eps/eps_f 0.963, u' 0.920, C 0.938
// (8 samples) against 1507 steps, 0.979 / 0.971 / 0.942 at 6 + 6. The GPU
// twin (gpu_hit_mini) runs the binary's FP32 path at the full 6 + 6.
#include "dns/diagnostics.hpp"
#include "dns/forcing.hpp"
#include "dns/hit_ic.hpp"
#include "dns/operators.hpp"
#include "dns/timestepper.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>

int main()
{
    using std::numbers::pi;
    const int n = 64;
    const double lBox = 2 * pi;
    const double uTarget = 1.0;
    const double epsF = uTarget * uTarget * uTarget / (0.19 * lBox); // 0.8377
    const double h = lBox / n;
    const double eta = h / 1.5;
    const double nu = std::cbrt(epsF * eta * eta * eta * eta); // eta=(nu^3/eps)^(1/4)
    const double tEddy = 1.5 * uTarget * uTarget / epsF;

    const auto g = dns::Grid::make(n, n, n, lBox, lBox, lBox);
    auto U = dns::VectorField::zeros(g);
    dns::hitInitialCondition(g, uTarget, 7, U);

    dns::TimeStepper ts(g, {nu, 0.9});
    const dns::LinearForcing forcing{epsF, 1.5 * uTarget * uTarget / 5.0};

    const double tSpin = 3.0 * tEddy, tEnd = 6.0 * tEddy;
    double t = 0.0;
    double epsSum = 0.0, kSum = 0.0, wSum = 0.0;
    long steps = 0;
    std::vector<double> specSum;
    long specCount = 0;

    while (t < tEnd)
    {
        const auto f = forcing.update(g, U);
        ts.setForcing(f);
        if (steps % 100 == 0) dns::subtractMean(g, f.mean, U);
        const double dt = ts.suggestDt(U, tEnd - t);
        ts.step(U, dt);
        t += dt;
        ++steps;
        if (t > tSpin)
        {
            epsSum += dt * nu * dns::velocityGradientSq(g, U);
            kSum += dt * dns::fluctuationEnergy(g, U);
            wSum += dt;
            if (steps % 50 == 0) // plans per call: sample sparingly
            {
                const auto E = dns::energySpectrum(g, U);
                if (specSum.empty()) specSum.assign(E.size(), 0.0);
                for (std::size_t s = 0; s < E.size(); ++s) specSum[s] += E[s];
                ++specCount;
            }
        }
    }

    const double epsMean = epsSum / wSum;
    const double uPrime = std::sqrt(2.0 * (kSum / wSum) / 3.0);
    const double maxDiv = dns::maxAbsDivergence(g, U);
    std::printf("steps=%ld  eps/epsF=%.3f  u'/target=%.3f  maxDiv=%.3e\n",
                steps, epsMean / epsF, uPrime / uTarget, maxDiv);

    bool ok = true;
    if (!(epsMean / epsF > 0.8 && epsMean / epsF < 1.2))
    {
        std::fprintf(stderr, "FAIL: eps_visc/eps_f = %.3f outside [0.8,1.2]\n",
                     epsMean / epsF);
        ok = false;
    }
    if (!(uPrime > 0.8 && uPrime < 1.2))
    {
        std::fprintf(stderr, "FAIL: u' = %.3f outside [0.8,1.2]\n", uPrime);
        ok = false;
    }
    if (!(maxDiv < 1e-8))
    {
        std::fprintf(stderr, "FAIL: maxDiv = %.3e >= 1e-8\n", maxDiv);
        ok = false;
    }

    // Compensated inertial-range spectrum (shell width k0 = 1 for L = 2pi).
    double comp = 0.0;
    int nk = 0;
    for (int k = 4; k <= 10 && std::size_t(k) < specSum.size(); ++k, ++nk)
        comp += specSum[std::size_t(k)] / double(specCount)
              * std::pow(double(k), 5.0 / 3.0) * std::pow(epsMean, -2.0 / 3.0);
    comp /= nk;
    std::printf("compensated spectrum C=%.3f (samples=%ld, k in [4,10])\n",
                comp, specCount);
    if (!(specCount > 0 && comp > 0.7 && comp < 1.6))
    {
        std::fprintf(stderr, "FAIL: C = %.3f outside [0.7, 1.6]\n", comp);
        ok = false;
    }
    return ok ? 0 : 1;
}
