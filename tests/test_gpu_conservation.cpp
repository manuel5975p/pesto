// Structural invariants of the device operators, checked on RANDOM fields --
// properties that must hold for any input, not just smooth flows:
//  1) skew-symmetric convection is energy-neutral: <u, C(u)> = 0 even for
//     non-solenoidal u (the divergence form fails this at O(1) and blew up
//     under-resolved cut-cell shear layers before 2026-07-15).
//  2) apertured divergence/gradient are discrete negative adjoints:
//     <p, D_a u> = -sum_f a_f u_f (grad p)_f  (summation by parts). This
//     duality is what makes the cut Poisson operator SPD.
#include "gpu_test_util.hpp"

#include "dns/cutcell.hpp"
#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/kernels.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <vector>

using namespace dns;

namespace {

VectorField randomField(const Grid& g, unsigned seed, double amp)
{
    auto U = VectorField::zeros(g);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(-amp, amp);
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        U.u[c] = d(rng); U.v[c] = d(rng); U.w[c] = d(rng);
    }
    return U;
}

double dotF(const Scalar& a, const Scalar& b)
{
    double s = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c) s += a[c] * b[c];
    return s;
}

} // namespace

int main()
{
    using std::numbers::pi;
    test::requireDeviceOrSkip();

    const int n = 48;
    const double L = 2 * pi;
    const auto g = Grid::make(n, n, n, L, L, L);
    bool ok = true;

    // --- 1) convection energy neutrality on a random (non-solenoidal) field ---
    {
        const auto U = randomField(g, 7, 1.0);
        auto dU = gpu::DeviceField::make(g.size());
        auto dC = gpu::DeviceField::make(g.size());
        if (!dU || !dC) { std::fprintf(stderr, "FAIL: GPU alloc\n"); return 1; }
        (void)gpu::uploadField(U, *dU);
        gpu::convection(gpu::GpuGrid::from(g), *dU, *dC);
        auto C = VectorField::zeros(g);
        (void)gpu::downloadField(*dC, C);

        const double e = dotF(U.u, C.u) + dotF(U.v, C.v) + dotF(U.w, C.w);
        // Scale: |u|~1, |C|~|u|^2/h -> the raw dot is O(N/h) if conservation
        // breaks; the skew form cancels it to FP32 roundoff.
        const double scale = double(g.size()) / g.hx;
        std::printf("<u,C(u)>/scale = %.3e\n", e / scale);
        if (std::abs(e / scale) > 1e-4)
        {
            std::printf("FAIL: convection injects energy on non-solenoidal u\n");
            ok = false;
        }
    }

    // --- 2) apertured divergence/gradient duality on a sphere cut geometry ---
    {
        const double h = g.hx;
        const Sphere s{L / 2 + 0.137 * h, L / 2 - 0.293 * h, L / 2 + 0.051 * h,
                       L / 5};
        auto cut = CutGeometry::make(g, s);
        (void)sanitizeCut(g, cut);

        auto U = randomField(g, 11, 1.0);
        maskSolidFaces(g, cut, U);
        auto P = randomField(g, 13, 1.0).u; // random cell scalar

        auto dCut = gpu::DeviceCutGeometry::make(cut);
        auto dU = gpu::DeviceField::make(g.size());
        auto dG = gpu::DeviceField::make(g.size());
        auto dDiv = gpu::DeviceScalar::make(g.size());
        if (!dCut || !dU || !dG || !dDiv)
        {
            std::fprintf(stderr, "FAIL: GPU alloc\n");
            return 1;
        }
        const auto gg = gpu::GpuGrid::from(g);
        (void)gpu::uploadField(U, *dU);

        // lhs = <P, D_a U> via the device divergence.
        gpu::divergenceCut(gg, *dCut, *dU, *dDiv, gpu::Real(1));
        std::vector<gpu::Real> divH(g.size());
        (void)dDiv->download(divH.data());
        double lhs = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c) lhs += P[c] * double(divH[c]);

        // rhs = -sum_f a_f u_f (grad P)_f via the device gradient: G = grad P
        // on open faces obtained from subtractGradientCut(0 - (-1)*grad).
        {
            std::vector<gpu::Real> z(g.size(), gpu::Real(0));
            (void)dG->u.upload(z.data());
            (void)dG->v.upload(z.data());
            (void)dG->w.upload(z.data());
        }
        auto dP = gpu::DeviceScalar::make(g.size());
        if (!dP) { std::fprintf(stderr, "FAIL: GPU alloc\n"); return 1; }
        std::vector<gpu::Real> pH(P.begin(), P.end());
        (void)dP->upload(pH.data());
        gpu::subtractGradientCut(gg, *dCut, *dP, gpu::Real(-1), *dG);
        auto Gh = VectorField::zeros(g);
        (void)gpu::downloadField(*dG, Gh);

        double rhs = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c)
            rhs -= cut.ax[c] * U.u[c] * Gh.u[c] + cut.ay[c] * U.v[c] * Gh.v[c]
                 + cut.az[c] * U.w[c] * Gh.w[c];

        const double scale = std::max({std::abs(lhs), std::abs(rhs), 1.0});
        std::printf("duality lhs=%.6e rhs=%.6e rel=%.3e\n", lhs, rhs,
                    std::abs(lhs - rhs) / scale);
        if (std::abs(lhs - rhs) / scale > 1e-4)
        {
            std::printf("FAIL: D_a and G_a are not discrete adjoints\n");
            ok = false;
        }
    }

    return ok ? 0 : 1;
}
