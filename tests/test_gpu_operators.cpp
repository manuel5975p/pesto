// Per-kernel parity of the device stencils against the CPU double oracle.
#include "gpu_test_util.hpp"

#include "dns/gpu/kernels.hpp"
#include "dns/operators.hpp"

#include <numbers>
#include <vector>

namespace {

using namespace dns;

Scalar downloadScalar(const gpu::DeviceScalar& d)
{
    std::vector<gpu::Real> tmp(d.size());
    (void)d.download(tmp.data());
    return Scalar(tmp.begin(), tmp.end());
}

double maxAbs(const Scalar& a, const Scalar& b, double& scale)
{
    double e = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c)
    {
        e = std::max(e, std::abs(a[c] - b[c]));
        scale = std::max(scale, std::abs(b[c]));
    }
    return e;
}

} // namespace

int main()
{
    using std::numbers::pi;
    test::requireDeviceOrSkip();

    // Deliberately anisotropic to catch axis mix-ups.
    const auto g = Grid::make(32, 24, 20, 2 * pi, 1.5 * pi, 1.0 * pi);
    const auto U = test::randomField(g, 3);
    const auto gg = gpu::GpuGrid::from(g);

    auto D = *gpu::DeviceField::make(g.size());
    (void)gpu::uploadField(U, D);
    auto R = *gpu::DeviceField::make(g.size());
    auto S = *gpu::DeviceScalar::make(g.size());

    bool ok = true;

    { // divergence (with scale folding)
        auto ref = zeros(g);
        divergence(g, U, ref);
        for (auto& x : ref) x *= 2.5;
        gpu::divergence(gg, D, S, gpu::Real(2.5));
        double scale = 0.0;
        const double e = maxAbs(downloadScalar(S), ref, scale);
        ok &= test::checkClose("divergence", e, scale, 2e-6);
    }

    { // convection
        auto ref = VectorField::zeros(g);
        convection(g, U, ref);
        gpu::convection(gg, D, R);
        auto got = VectorField::zeros(g);
        (void)gpu::downloadField(R, got);
        const auto d = test::fieldDiff(got, ref);
        ok &= test::checkClose("convection", d.maxErr, d.scale, 1e-5);
    }

    { // laplacian via addScaledLaplacian on a zeroed accumulator
        auto ref = VectorField::zeros(g);
        laplacian(g, U, ref);
        const double nu = 0.37;
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            ref.u[c] *= nu;
            ref.v[c] *= nu;
            ref.w[c] *= nu;
        }
        auto Z = VectorField::zeros(g);
        (void)gpu::uploadField(Z, R);
        gpu::addScaledLaplacian(gg, gpu::Real(nu), D, R);
        auto got = VectorField::zeros(g);
        (void)gpu::downloadField(R, got);
        const auto d = test::fieldDiff(got, ref);
        ok &= test::checkClose("laplacian", d.maxErr, d.scale, 1e-5);
    }

    { // subtractGradient
        auto p = zeros(g);
        std::mt19937_64 rng(9);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& x : p) x = dist(rng);
        auto ref = U;
        subtractGradient(g, p, 0.01, ref);

        std::vector<gpu::Real> pf(p.begin(), p.end());
        (void)S.upload(pf.data());
        (void)gpu::uploadField(U, D);
        gpu::subtractGradient(gg, S, gpu::Real(0.01), D);
        auto got = VectorField::zeros(g);
        (void)gpu::downloadField(D, got);
        const auto d = test::fieldDiff(got, ref);
        ok &= test::checkClose("subtractGradient", d.maxErr, d.scale, 1e-5);
    }

    { // fusedRhsStage == convection + nu*lap + forcing + rkCombine
        (void)gpu::uploadField(U, D);
        const double nu = 0.02, A = 0.9, a = 0.75, b = 0.25, dt = 0.015;
        const Vec3 m{0.02, -0.01, 0.03};
        auto rhs = VectorField::zeros(g);
        convection(g, U, rhs);
        auto lap = VectorField::zeros(g);
        laplacian(g, U, lap);
        auto ref = VectorField::zeros(g);
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            ref.u[c] = a * U.u[c]
                     + b * (U.u[c] + dt * (rhs.u[c] + nu * lap.u[c] + A * (U.u[c] - m.x)));
            ref.v[c] = a * U.v[c]
                     + b * (U.v[c] + dt * (rhs.v[c] + nu * lap.v[c] + A * (U.v[c] - m.y)));
            ref.w[c] = a * U.w[c]
                     + b * (U.w[c] + dt * (rhs.w[c] + nu * lap.w[c] + A * (U.w[c] - m.z)));
        }
        gpu::fusedRhsStage(gg, gpu::Real(nu), gpu::Real(A), gpu::Real(m.x),
                           gpu::Real(m.y), gpu::Real(m.z), gpu::Real(a), D,
                           gpu::Real(b), D, gpu::Real(dt), R);
        auto got = VectorField::zeros(g);
        (void)gpu::downloadField(R, got);
        const auto d = test::fieldDiff(got, ref);
        ok &= test::checkClose("fusedRhsStage", d.maxErr, d.scale, 1e-5);
    }

    { // rkCombine + forcing
        (void)gpu::uploadField(U, D);
        auto ref = VectorField::zeros(g);
        const double a = 0.75, b = 0.25, dt = 0.02, A = 1.3;
        const Vec3 m{0.1, -0.2, 0.05};
        auto rhs = test::randomField(g, 11);
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            ref.u[c] = a * U.u[c] + b * (U.u[c] + dt * (rhs.u[c] + A * (U.u[c] - m.x)));
            ref.v[c] = a * U.v[c] + b * (U.v[c] + dt * (rhs.v[c] + A * (U.v[c] - m.y)));
            ref.w[c] = a * U.w[c] + b * (U.w[c] + dt * (rhs.w[c] + A * (U.w[c] - m.z)));
        }
        (void)gpu::uploadField(rhs, R);
        gpu::addForcing(gg, gpu::Real(A), gpu::Real(m.x), gpu::Real(m.y),
                        gpu::Real(m.z), D, R);
        auto out = *gpu::DeviceField::make(g.size());
        gpu::rkCombine(gg, gpu::Real(a), D, gpu::Real(b), D, gpu::Real(dt), R, out);
        auto got = VectorField::zeros(g);
        (void)gpu::downloadField(out, got);
        const auto d = test::fieldDiff(got, ref);
        ok &= test::checkClose("rkCombine+forcing", d.maxErr, d.scale, 1e-5);
    }

    return ok ? 0 : 1;
}
