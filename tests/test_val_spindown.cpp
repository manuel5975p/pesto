// Validation A1: spin-down of solid-body rotation inside a stationary no-slip
// circle -- the analytic benchmark for the sharp ghost-fill viscous wall.
// Fluid INSIDE the circle (complemented CylinderZ cut), IC u_theta = Omega*r,
// wall at rest. The convective term is a pure radial gradient absorbed by the
// pressure, so the full NS solution is the axisymmetric diffusion series
//   u_theta(r,t) = sum_k c_k J1(lam_k r/R) exp(-nu lam_k^2 t / R^2),
//   lam_k = zeros of J1,  c_k = -2 Omega R / (lam_k J0(lam_k)),
// (Fourier-Bessel expansion of Omega*r; coefficients verified in-test by
// quadrature and the IC reproduced in L2 before the series is trusted).
// Exercises advection + projection + ghost-filled diffusion together.
#include "validation_util.hpp"

#include "dns/cutcell.hpp"
#include "dns/gpu/cutpoisson.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/ghostfill.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

using namespace dns;

namespace {

double besJ1(double x) { return std::cyl_bessel_j(1.0, x); }
double besJ0(double x) { return std::cyl_bessel_j(0.0, x); }

// First nk positive zeros of J1: McMahon estimate + Newton (J1' = J0 - J1/x).
std::vector<double> j1Zeros(int nk)
{
    using std::numbers::pi;
    std::vector<double> z(nk);
    for (int k = 1; k <= nk; ++k)
    {
        const double beta = (k + 0.25) * pi;
        double lam = beta - 3.0 / (8.0 * beta);
        for (int it = 0; it < 6; ++it)
            lam -= besJ1(lam) / (besJ0(lam) - besJ1(lam) / lam);
        z[k - 1] = lam;
    }
    return z;
}

// Truth series: u_theta(r,t) and the KE ratio KE(t)/KE(0) (KE(0) analytic:
// pi Omega^2 R^4 / 4 per unit depth; term k carries the fraction 8/lam_k^2).
struct SpinDownSeries
{
    std::vector<double> lam, c;
    double omega, radius, nu;

    double uTheta(double r, double t) const
    {
        double s = 0.0;
        for (std::size_t k = 0; k < lam.size(); ++k)
            s += c[k] * besJ1(lam[k] * r / radius)
               * std::exp(-nu * lam[k] * lam[k] * t / (radius * radius));
        return s;
    }
    double keRatio(double t) const
    {
        double s = 0.0;
        for (double l : lam)
            s += 8.0 / (l * l) * std::exp(-2.0 * nu * l * l * t / (radius * radius));
        return s;
    }

    static SpinDownSeries make(double omega, double radius, double nu, int nk)
    {
        SpinDownSeries s{j1Zeros(nk), {}, omega, radius, nu};
        s.c.resize(s.lam.size());
        for (std::size_t k = 0; k < s.lam.size(); ++k)
            s.c[k] = -2.0 * omega * radius / (s.lam[k] * besJ0(s.lam[k]));
        return s;
    }
};

// Simpson quadrature of f on [0,R] with n panels (n even).
template <class F> double simpson(F f, double hi, int n)
{
    const double h = hi / n;
    double s = f(0.0) + f(hi);
    for (int i = 1; i < n; ++i) s += (i % 2 ? 4.0 : 2.0) * f(i * h);
    return s * h / 3.0;
}

} // namespace

int main()
{
    using std::numbers::pi;
    if (!gpu::deviceAvailable()) valtest::skip("no HIP device");

    // --- truth series self-verification (before trusting it) ---------------
    const double omega = 1.0, radius = 2.4, nu = 0.05, tEnd = 4.0;
    const auto series = SpinDownSeries::make(omega, radius, nu, 60);

    // Closed-form c_k vs direct quadrature of the Fourier-Bessel projection.
    double worstCk = 0.0;
    for (int k = 0; k < 6; ++k)
    {
        const double l = series.lam[k];
        const double num = simpson(
            [&](double r) { return omega * r * besJ1(l * r / radius) * r; }, radius, 4000);
        const double den = 0.5 * radius * radius * besJ0(l) * besJ0(l); // = norm of J1 mode
        worstCk = std::max(worstCk, std::abs(num / den - series.c[k])
                                        / std::abs(series.c[k]));
    }
    valtest::check(worstCk < 1e-6, "series c_k matches quadrature (err "
                                       + std::to_string(worstCk) + ")");

    // Energy completeness: sum 8/lam_k^2 -> 1 (exact identity, slow 1/K tail).
    const double keSum = series.keRatio(0.0);
    valtest::check(keSum > 0.98 && keSum < 1.0,
                   "KE completeness sum(8/lam^2) = " + std::to_string(keSum));

    // IC reproduction in L2 over r in [0, 0.9R] (away from the Gibbs layer at
    // the wall, which the heat semigroup smooths instantly for t > 0).
    {
        const auto icSeries = SpinDownSeries::make(omega, radius, nu, 2000);
        double num = 0.0, den = 0.0;
        for (int i = 0; i < 200; ++i)
        {
            const double r = (i + 0.5) / 200.0 * 0.9 * radius;
            const double d = icSeries.uTheta(r, 0.0) - omega * r;
            num += d * d * r;
            den += omega * r * omega * r * r;
        }
        const double l2 = std::sqrt(num / den);
        valtest::check(l2 < 0.005, "series reproduces IC, L2 = " + std::to_string(l2));
    }

    // --- geometry: complement of the exact circle cut (fluid inside) -------
    const int n = 128;
    const double lBox = 2 * pi;
    const double h = lBox / n;
    const auto g = Grid::make(n, n, 1, lBox, lBox, h);
    const double cx = lBox / 2 + 0.31 * h, cy = lBox / 2 + 0.17 * h; // off-node
    auto cut = CutGeometry::make(g, CylinderZ{cx, cy, radius});
    for (std::size_t i = 0; i < g.size(); ++i)
    {
        cut.lambda[i] = 1.0 - cut.lambda[i];
        cut.ax[i] = 1.0 - cut.ax[i];
        cut.ay[i] = 1.0 - cut.ay[i];
        cut.az[i] = 1.0 - cut.az[i];
    }
    const auto st = sanitizeCut(g, cut);
    std::printf("sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed, st.minLambda,
                st.minAperture);

    // IC: solid-body rotation on the MAC faces (discretely divergence-free).
    auto U = VectorField::zeros(g);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
        {
            const auto c = g.idx(i, j, 0);
            U.u[c] = -omega * ((j + 0.5) * h - cy);
            U.v[c] = omega * ((i + 0.5) * h - cx);
        }
    maskSolidFaces(g, cut, U);

    // --- GPU setup: cylinder-app stage pattern minus the tunnel BCs ---------
    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    // sdf < 0 in the solid; solid is OUTSIDE the circle, so sdf = R - r.
    auto dGhost = gpu::DeviceGhostFill::make(g, cut, [&](double x, double y, double) {
        return radius - std::hypot(x - cx, y - cy);
    });
    auto dU = gpu::DeviceField::make(g.size());
    auto dU0 = gpu::DeviceField::make(g.size());
    auto dS = gpu::DeviceField::make(g.size());
    auto dG = gpu::DeviceField::make(g.size()); // ghost-filled diffusion view
    auto red = gpu::ReduceScratch::make();
    auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
    if (!dCut || !dGhost || !dU || !dU0 || !dS || !dG || !red || !scr)
    {
        std::printf("FAIL: GPU alloc\n");
        return 1;
    }
    auto poisson = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 500, 1e-5);
    if (!poisson)
    {
        std::printf("FAIL: Poisson init\n");
        return 1;
    }
    (void)gpu::uploadField(U, *dU);
    gpu::maskSolidFaces(gg, *dCut, *dU);
    (void)hipMemset(scr->phi.data(), 0, g.size() * sizeof(gpu::Real));
    gpu::projectCut(gg, *dCut, *poisson, gpu::Real(1), *dU, *scr);

    const double k0 = gpu::kineticEnergy(gg, *dU, *red);
    std::printf("K0 = %.6f (ghost faces: %zu)\n", k0, dGhost->count());

    // Diffusion reads the ghost-filled view; convection keeps the masked zeros.
    auto stage = [&](const gpu::DeviceField& in, gpu::DeviceField& out, double a,
                     double dt) {
        (void)dGhost->fillFrom(in, *dG);
        gpu::fusedRhsStage(gg, gpu::Real(nu), gpu::Real(0), 0, 0, 0, gpu::Real(a),
                           *dU0, gpu::Real(1.0 - a), in, *dG, gpu::Real(dt), out);
        gpu::maskSolidFaces(gg, *dCut, out);
        gpu::projectCut(gg, *dCut, *poisson, gpu::Real(dt), out, *scr);
    };

    double t = 0.0;
    long steps = 0;
    while (t < tEnd)
    {
        const double conv = gpu::maxCourantPerDt(gg, *dU, *red);
        const double diff = 6.0 * nu / (h * h);
        const double dt = std::min(tEnd - t, 0.8 / std::max(conv, diff));
        std::swap(*dU0, *dU);
        stage(*dU0, *dU, 0.0, dt);
        stage(*dU, *dS, 0.75, dt);
        stage(*dS, *dU, 1.0 / 3.0, dt);
        t += dt;
        ++steps;
        if (steps % 200 == 0)
            std::printf("t=%.3f K/K0=%.4f (truth %.4f)\n", t,
                        gpu::kineticEnergy(gg, *dU, *red) / k0, series.keRatio(t));
    }

    // --- KE decay vs series -------------------------------------------------
    const double keSim = gpu::kineticEnergy(gg, *dU, *red) / k0;
    const double keTruth = series.keRatio(tEnd);
    const double keErr = std::abs(keSim - keTruth) / keTruth;
    std::printf("steps=%ld  KE(t)/KE(0): sim=%.5f truth=%.5f relerr=%.4f\n", steps,
                keSim, keTruth, keErr);
    valtest::check(keErr < 0.03, "KE decay within 3% of series");

    // --- azimuthally binned u_theta profile vs series -----------------------
    (void)gpu::downloadField(*dU, U);
    const int nb = int(radius / h) - 4;
    std::vector<double> sumU(nb, 0.0), sumR(nb, 0.0);
    std::vector<int> cnt(nb, 0);
    for (int j = 1; j < n - 1; ++j)
        for (int i = 1; i < n - 1; ++i)
        {
            const auto c = g.idx(i, j, 0);
            if (cut.lambda[c] < 1.0) continue; // full fluid cells only
            const double x = (i + 0.5) * h - cx, y = (j + 0.5) * h - cy;
            const double r = std::hypot(x, y);
            const int b = int(r / h) - 2; // bins start at r = 2h
            if (b < 0 || b >= nb) continue; // skip axis + 3 cells at the wall
            const double uc = 0.5 * (U.u[c] + U.u[g.idx(i + 1, j, 0)]);
            const double vc = 0.5 * (U.v[c] + U.v[g.idx(i, j + 1, 0)]);
            sumU[b] += (-y * uc + x * vc) / r; // u_theta
            sumR[b] += r;
            ++cnt[b];
        }
    double num = 0.0, den = 0.0;
    for (int b = 0; b < nb; ++b)
    {
        if (cnt[b] == 0) continue;
        const double rBin = sumR[b] / cnt[b];
        const double sim = sumU[b] / cnt[b];
        const double truth = series.uTheta(rBin, tEnd);
        num += cnt[b] * (sim - truth) * (sim - truth);
        den += cnt[b] * truth * truth;
        if (b % 8 == 0)
            std::printf("r=%.3f u_theta sim=%.4f truth=%.4f\n", rBin, sim, truth);
    }
    const double l2 = std::sqrt(num / den);
    std::printf("profile L2 rel err = %.4f\n", l2);
    valtest::check(l2 < 0.04, "u_theta profile L2 within 4% of series");

    return valtest::finish();
}
