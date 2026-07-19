#include "dns/diagnostics.hpp"

#include "dns/operators.hpp"

#if DNS_WITH_FFTW
#include <fftw3.h>
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>

namespace dns {

FlowStats computeStats(const Grid& g, const VectorField& U, double nu)
{
    FlowStats s{};
    s.kPrime = fluctuationEnergy(g, U);
    s.uPrime = std::sqrt(2.0 * s.kPrime / 3.0);
    s.epsVisc = nu * velocityGradientSq(g, U);
    if (s.epsVisc > 0 && nu > 0)
    {
        s.eta = std::pow(nu * nu * nu / s.epsVisc, 0.25);
        s.reLambda = s.uPrime * s.uPrime * std::sqrt(15.0 / (nu * s.epsVisc));
        s.hOverEta = std::max({g.hx, g.hy, g.hz}) / s.eta;
    }
    s.courantPerDt = maxCourantPerDt(g, U);
    s.maxDiv = maxAbsDivergence(g, U);
    return s;
}

bool appendStatsCsv(const std::string& path, double t, double dt, double forcingA,
                    const FlowStats& s)
{
    const bool fresh = !std::filesystem::exists(path);
    std::ofstream f(path, std::ios::app);
    if (!f) return false;
    if (fresh)
        f << "t,dt,A,kPrime,uPrime,epsVisc,epsInj,eta,reLambda,hOverEta,cfl,maxDiv\n";
    f << t << ',' << dt << ',' << forcingA << ',' << s.kPrime << ',' << s.uPrime
      << ',' << s.epsVisc << ',' << 2.0 * forcingA * s.kPrime << ',' << s.eta << ','
      << s.reLambda << ',' << s.hOverEta << ',' << s.courantPerDt * dt << ','
      << s.maxDiv << '\n';
    return bool(f);
}

#if DNS_WITH_FFTW

std::vector<double> energySpectrum(const Grid& g, const VectorField& U)
{
    using std::numbers::pi;
    const std::size_t n = g.size();
    const int nxh = g.nx / 2 + 1;
    const std::size_t nc = std::size_t(g.nz) * g.ny * nxh;
    const int nShells = std::min({g.nx, g.ny, g.nz}) / 2;
    const double lx = g.nx * g.hx, ly = g.ny * g.hy, lz = g.nz * g.hz;
    const double k0 = 2.0 * pi / lx;

    std::vector<double> shellE(nShells, 0.0);
    double* real = fftw_alloc_real(n);
    fftw_complex* cplx = fftw_alloc_complex(nc);
    fftw_plan fwd = fftw_plan_dft_r2c_3d(g.nz, g.ny, g.nx, real, cplx, FFTW_ESTIMATE);

    for (int comp = 0; comp < 3; ++comp)
    {
        // Interpolate the face component to cell centres.
        const Scalar& q = comp == 0 ? U.u : comp == 1 ? U.v : U.w;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                {
                    const auto c = g.idx(i, j, k);
                    const auto cp = comp == 0 ? g.idx(i + 1, j, k)
                                  : comp == 1 ? g.idx(i, j + 1, k)
                                              : g.idx(i, j, k + 1);
                    real[c] = 0.5 * (q[c] + q[cp]);
                }
        fftw_execute(fwd);

        const double norm = 1.0 / (double(n) * double(n));
        for (int kz = 0; kz < g.nz; ++kz)
            for (int ky = 0; ky < g.ny; ++ky)
                for (int kx = 0; kx < nxh; ++kx)
                {
                    const std::size_t c = std::size_t(kx)
                        + std::size_t(nxh) * (std::size_t(ky) + std::size_t(g.ny) * kz);
                    const int my = ky > g.ny / 2 ? ky - g.ny : ky;
                    const int mz = kz > g.nz / 2 ? kz - g.nz : kz;
                    const double kkx = 2.0 * pi * kx / lx;
                    const double kky = 2.0 * pi * my / ly;
                    const double kkz = 2.0 * pi * mz / lz;
                    const double kk = std::sqrt(kkx * kkx + kky * kky + kkz * kkz);
                    const int s = int(std::lround(kk / k0));
                    if (s >= nShells) continue;
                    const double w = (kx == 0 || 2 * kx == g.nx) ? 1.0 : 2.0;
                    const double e2 = cplx[c][0] * cplx[c][0] + cplx[c][1] * cplx[c][1];
                    shellE[s] += w * 0.5 * e2 * norm;
                }
    }

    fftw_destroy_plan(fwd);
    fftw_free(real);
    fftw_free(cplx);

    // Remove the mean-flow contribution so sum(shellE) tracks kPrime.
    shellE[0] = 0.0;
    return shellE;
}

#endif // DNS_WITH_FFTW

} // namespace dns
