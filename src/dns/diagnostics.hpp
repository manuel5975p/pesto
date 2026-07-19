// Runtime flow statistics and shell-binned energy spectra for HIT/TGV runs.
#pragma once

#include "dns/field.hpp"
#include "dns/grid.hpp"

#include <string>
#include <vector>

namespace dns {

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

// One pass of FP64 reductions over U; nu >= 0.
FlowStats computeStats(const Grid& g, const VectorField& U, double nu);

// Appends one CSV row (writes the header on first use). Returns false on I/O error.
bool appendStatsCsv(const std::string& path, double t, double dt, double forcingA,
                    const FlowStats& s);

#if DNS_WITH_FFTW
// Shell energies E_s (sum over shells = kPrime), shell width k0 = 2pi/lx,
// velocity interpolated to cell centres. Size min(n)/2. Plans per call: use sparingly.
std::vector<double> energySpectrum(const Grid& g, const VectorField& U);
#endif

} // namespace dns
