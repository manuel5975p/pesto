#include "dns/hit_ic.hpp"

#include "dns/operators.hpp"
#include "dns/poisson.hpp"
#include "dns/projection.hpp"

#include <fftw3.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

namespace dns {

namespace {

// Signed mode index m in (-n/2, n/2] for FFT bin i in [0, n).
int signedMode(int i, int n) { return i > n / 2 ? i - n : i; }

} // namespace

void hitInitialCondition(const Grid& g, double uPrime, unsigned seed, VectorField& U)
{
    using std::numbers::pi;
    assert(uPrime > 0);

    const std::size_t n = g.size();
    const int nxh = g.nx / 2 + 1;
    const std::size_t nc = std::size_t(g.nz) * g.ny * nxh;
    const double lx = g.nx * g.hx, ly = g.ny * g.hy, lz = g.nz * g.hz;

    double* real = fftw_alloc_real(n);
    fftw_complex* cplx = fftw_alloc_complex(nc);
    fftw_plan fwd = fftw_plan_dft_r2c_3d(g.nz, g.ny, g.nx, real, cplx, FFTW_ESTIMATE);
    fftw_plan bwd = fftw_plan_dft_c2r_3d(g.nz, g.ny, g.nx, cplx, real, FFTW_ESTIMATE);

    // 1. White Gaussian noise per component, transformed to spectral space.
    std::vector<std::vector<std::complex<double>>> uh(3);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (auto& comp : uh)
    {
        for (std::size_t c = 0; c < n; ++c) real[c] = gauss(rng);
        fftw_execute(fwd);
        comp.assign(reinterpret_cast<std::complex<double>*>(cplx),
                    reinterpret_cast<std::complex<double>*>(cplx) + nc);
    }

    // 2. Per mode: zero Nyquist/DC, solenoidal projection (I - k k^T/|k|^2),
    //    accumulate shell energies (Hermitian half-plane weights).
    const double k0 = 2.0 * pi / lx; // shell width; assumes lx ~ ly ~ lz
    const int nShells = std::min({g.nx, g.ny, g.nz}) / 2;
    std::vector<double> shellE(nShells, 0.0);
    std::vector<int> shellOf(nc, -1);

    for (int kz = 0; kz < g.nz; ++kz)
        for (int ky = 0; ky < g.ny; ++ky)
            for (int kx = 0; kx < nxh; ++kx)
            {
                const std::size_t c = std::size_t(kx)
                    + std::size_t(nxh) * (std::size_t(ky) + std::size_t(g.ny) * kz);
                const int mx = kx, my = signedMode(ky, g.ny), mz = signedMode(kz, g.nz);
                const bool nyq = 2 * kx == g.nx || 2 * std::abs(my) == g.ny
                              || 2 * std::abs(mz) == g.nz;
                const double kkx = 2.0 * pi * mx / lx;
                const double kky = 2.0 * pi * my / ly;
                const double kkz = 2.0 * pi * mz / lz;
                const double k2 = kkx * kkx + kky * kky + kkz * kkz;
                const int s = int(std::lround(std::sqrt(k2) / k0));

                if (nyq || k2 == 0.0 || s < 1 || s >= nShells)
                {
                    uh[0][c] = uh[1][c] = uh[2][c] = 0.0;
                    continue;
                }
                const std::complex<double> kdotu =
                    (kkx * uh[0][c] + kky * uh[1][c] + kkz * uh[2][c]) / k2;
                uh[0][c] -= kkx * kdotu;
                uh[1][c] -= kky * kdotu;
                uh[2][c] -= kkz * kdotu;

                shellOf[c] = s;
                const double w = (kx == 0) ? 1.0 : 2.0; // Hermitian pair weight
                shellE[s] += w * 0.5 * (std::norm(uh[0][c]) + std::norm(uh[1][c])
                                      + std::norm(uh[2][c]));
            }

    // 3. Shell rescale to E(k) ~ (k/kp)^4 exp(-2 (k/kp)^2); absolute level is
    //    fixed by the final physical-space rescale, so the prefactor is 1.
    const double kp = 4.0 * k0;
    std::vector<double> scale(nShells, 0.0);
    for (int s = 1; s < nShells; ++s)
    {
        const double k = s * k0, r = k / kp;
        const double target = r * r * r * r * std::exp(-2.0 * r * r);
        scale[s] = shellE[s] > 0 ? std::sqrt(target / shellE[s]) : 0.0;
    }

    // 4. Apply shell scale + MAC half-cell phase shift per component, c2r.
    //    u lives at (i, j+.5, k+.5)h => shift by exp(i(ky hy + kz hz)/2), cyclic.
    Scalar* out[3] = {&U.u, &U.v, &U.w};
    for (int comp = 0; comp < 3; ++comp)
    {
        for (int kz = 0; kz < g.nz; ++kz)
            for (int ky = 0; ky < g.ny; ++ky)
                for (int kx = 0; kx < nxh; ++kx)
                {
                    const std::size_t c = std::size_t(kx)
                        + std::size_t(nxh) * (std::size_t(ky) + std::size_t(g.ny) * kz);
                    if (shellOf[c] < 0) { cplx[c][0] = cplx[c][1] = 0.0; continue; }
                    const double kkx = 2.0 * pi * kx / lx;
                    const double kky = 2.0 * pi * signedMode(ky, g.ny) / ly;
                    const double kkz = 2.0 * pi * signedMode(kz, g.nz) / lz;
                    double phase = 0.5 * (kkx * g.hx + kky * g.hy + kkz * g.hz);
                    if (comp == 0) phase -= 0.5 * kkx * g.hx;
                    if (comp == 1) phase -= 0.5 * kky * g.hy;
                    if (comp == 2) phase -= 0.5 * kkz * g.hz;
                    const std::complex<double> z = uh[comp][c] * scale[shellOf[c]]
                        * std::complex<double>(std::cos(phase), std::sin(phase));
                    cplx[c][0] = z.real();
                    cplx[c][1] = z.imag();
                }
        fftw_execute(bwd);
        const double norm = 1.0 / double(n);
        for (std::size_t c = 0; c < n; ++c) (*out[comp])[c] = real[c] * norm;
    }

    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(real);
    fftw_free(cplx);

    // 5. Discrete projection (spectral solenoidality != FD divergence-free),
    //    zero mean, exact rescale to k' = 1.5 uPrime^2.
    auto poisson = makeDefaultSolver(g);
    auto scratch = ProjectionScratch::make(g);
    project(g, *poisson, 1.0, U, scratch);
    subtractMean(g, meanVelocity(g, U), U);
    const double kPrime = fluctuationEnergy(g, U);
    assert(kPrime > 0);
    const double r = uPrime * std::sqrt(1.5 / kPrime);
    for (std::size_t c = 0; c < n; ++c)
    {
        U.u[c] *= r;
        U.v[c] *= r;
        U.w[c] *= r;
    }
}

} // namespace dns
