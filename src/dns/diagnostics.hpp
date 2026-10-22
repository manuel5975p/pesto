// Runtime flow statistics and shell-binned energy spectra for HIT/TGV runs.
#pragma once

#include "dns/field.hpp"
#include "dns/grid.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace dns {

// The 3D TGV dissipation series -- `dns tgv`'s output, on stdout in the
// reference curves' format -- for both the CPU and the GPU stepper.
// dKdt is the backward slope -dK/dt; the first row (t == 0) reports 0.
class TgvSeries
{
public:
    static void header() { std::printf("t,K,dKdt,epsVisc\n"); }
    void row(double t, double k, double eps)
    {
        const double dkdt = t > tPrev_ ? -(k - kPrev_) / (t - tPrev_) : 0.0;
        std::printf("%.6f,%.10e,%.10e,%.10e\n", t, k, dkdt, eps);
        kPrev_ = k;
        tPrev_ = t;
    }

private:
    double kPrev_ = 0.0, tPrev_ = 0.0;
};

struct FlowStats
{
    double kPrime;       // specific fluctuation energy 0.5<|u'|^2>
    double uPrime;       // sqrt(2 kPrime / 3), one-component rms
    double epsVisc;      // nu <|grad u|^2>
    double eta;          // Kolmogorov length (nu^3/eps)^(1/4); 0 if eps == 0
    double reLambda;     // Taylor-scale Reynolds number; 0 if eps or nu == 0
    double hOverEta;     // resolution monitor, warn above 1.5
    double courantPerDt; // multiply by dt for the CFL number
    double maxDiv;       // max |div(U)|
};

#if DNS_WITH_FFTW
// Shell energies E_s (sum over shells = kPrime), shell width k0 = 2pi/lx,
// velocity interpolated to cell centres. Size min(n)/2. Plans per call: use sparingly.
std::vector<double> energySpectrum(const Grid& g, const VectorField& U);
#endif

} // namespace dns
