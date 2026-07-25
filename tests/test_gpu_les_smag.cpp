// DeviceSmagorinsky (gpu/les.hpp), the incompressible LES model, tested
// against the library directly. This test used to shell out to a built
// cylinder_tunnel_gpu because the kernels lived in that app's anonymous
// namespace; linking dns_gpu instead is the proof the lift was faithful.
//
// Checks:
//   1. make() rejects cs <= 0 and accepts nz > 1 (full 3D since 2026-07-20).
//   2. nut >= 0 everywhere, is exactly 0 on the domain-boundary ring, and is
//      exactly 0 in solid cells (lambda == 0 scales it away).
//   3. nut scales as cs^2 for a fixed field.
//   4. A uniform field produces nut == 0 (|S| == 0) and no eddy flux.
//   5. The eddy-diffusion substep DISSIPATES: sum u . div(nut grad u) < 0 for a
//      sheared field, i.e. the sign of the model term is right.
//   6. 3D: a z-extruded field on cubic cells reproduces the 2D nut plane by
//      plane (the z terms vanish, Delta^2 matches), the 3D substep dissipates
//      a genuinely 3D field, and the conservative flux form conserves each
//      momentum component over the periodic box.
//   7. SgsModel::Sigma: nut == 0 identically for a two-component (z-extruded
//      2D) field where Smagorinsky is strongly nonzero -- the model's defining
//      property -- while a genuinely 3D field gives nut >= 0, > 0 somewhere,
//      zero ring/solid, and a dissipative substep.
//
// No HIP device -> skip (exit 77).

#include "dns/cutcell.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/les.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

namespace {

using dnstest::check;

constexpr int skipExitCode = 77;


} // namespace

int main()
{
    using namespace dns;
    using gpu::Real;

    if (!gpu::deviceAvailable())
    {
        std::printf("no HIP device, skipping\n");
        return skipExitCode;
    }

    const int n = 32;
    const auto g = Grid::make(n, n, 1, 1.0, 1.0, 1.0);
    const auto gg = gpu::GpuGrid::from(g);

    // --- (1) rejections ----------------------------------------------------
    check(!gpu::DeviceSmagorinsky::make(gg, 0.0).has_value(),
          "make() rejects cs == 0");
    check(!gpu::DeviceSmagorinsky::make(gg, -1.0).has_value(),
          "make() rejects cs < 0");
    {
        const auto g3 = Grid::make(8, 8, 8, 1.0, 1.0, 1.0);
        check(gpu::DeviceSmagorinsky::make(gpu::GpuGrid::from(g3), 0.2).has_value(),
              "make() accepts nz > 1 (the model is 3D)");
    }

    // Geometry: all fluid except a solid block, so lambda damping is exercised.
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        cut.lambda[c] = 1.0;
        cut.ax[c] = cut.ay[c] = cut.az[c] = 1.0;
    }
    for (int j = 12; j < 20; ++j)
        for (int i = 12; i < 20; ++i) cut.lambda[g.idx(i, j, 0)] = 0.0;
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    if (!dCut) { std::printf("geometry upload failed\n"); return 1; }

    auto dU = gpu::DeviceField::make(g.size());
    auto dOut = gpu::DeviceField::make(g.size());
    if (!dU || !dOut) { std::printf("alloc failed\n"); return 1; }

    auto upload = [&](const VectorField& U) { (void)gpu::uploadField(U, *dU); };
    auto nutHost = [&](gpu::DeviceSmagorinsky& les) {
        std::vector<Real> h(g.size());
        (void)les.nut().download(h.data());
        return h;
    };

    // --- (4) uniform field: |S| == 0 ---------------------------------------
    {
        auto U = VectorField::zeros(g);
        for (auto& x : U.u) x = 1.0;
        upload(U);
        auto les = gpu::DeviceSmagorinsky::make(gg, 0.2);
        if (!les) { std::printf("les make failed\n"); return 1; }
        les->computeNut(*dU, *dCut);
        const auto h = nutHost(*les);
        double m = 0;
        for (auto v : h) m = std::max(m, double(std::abs(v)));
        check(m < 1e-12, "uniform field gives nut == 0");
    }

    // Sheared + vortical field with real strain.
    auto U = VectorField::zeros(g);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
        {
            const double x = double(i) / n, y = double(j) / n;
            using std::numbers::pi;
            U.u[g.idx(i, j, 0)] = std::sin(2 * pi * y) + 0.3 * std::cos(4 * pi * x);
            U.v[g.idx(i, j, 0)] = 0.5 * std::sin(2 * pi * x);
        }
    upload(U);

    std::vector<Real> nutA, nutB;
    {
        auto les = gpu::DeviceSmagorinsky::make(gg, 0.1);
        if (!les) return 1;
        les->computeNut(*dU, *dCut);
        nutA = nutHost(*les);
    }
    {
        auto les = gpu::DeviceSmagorinsky::make(gg, 0.2);
        if (!les) return 1;
        les->computeNut(*dU, *dCut);
        nutB = nutHost(*les);
    }

    // --- (2) positivity, ring, solid ---------------------------------------
    bool nonneg = true, ringZero = true, solidZero = true, anyPositive = false;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
        {
            const double v = nutB[g.idx(i, j, 0)];
            if (v < 0) nonneg = false;
            if (v > 0) anyPositive = true;
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
                if (v != 0) ringZero = false;
            if (cut.lambda[g.idx(i, j, 0)] == 0.0 && v != 0) solidZero = false;
        }
    check(nonneg, "nut >= 0 everywhere");
    check(anyPositive, "nut > 0 somewhere (the field has strain)");
    check(ringZero, "nut == 0 on the domain-boundary ring");
    check(solidZero, "nut == 0 in solid cells (lambda damping)");

    // --- (3) cs^2 scaling ---------------------------------------------------
    {
        double worst = 0;
        for (std::size_t c = 0; c < g.size(); ++c)
            if (nutA[c] > 1e-6)
                worst = std::max(worst,
                                 std::abs(double(nutB[c]) / double(nutA[c]) - 4.0));
        check(worst < 1e-4, "nut scales as cs^2");
    }

    // --- (5) the eddy-diffusion substep dissipates -------------------------
    {
        auto les = gpu::DeviceSmagorinsky::make(gg, 0.2);
        if (!les) return 1;
        les->computeNut(*dU, *dCut);
        (void)hipMemset(dOut->u.data(), 0, g.size() * sizeof(Real));
        (void)hipMemset(dOut->v.data(), 0, g.size() * sizeof(Real));
        (void)hipMemset(dOut->w.data(), 0, g.size() * sizeof(Real));
        les->addEddyDiffusion(*dU, Real(1), *dOut);

        auto R = VectorField::zeros(g);
        (void)gpu::downloadField(*dOut, R);
        // sum u . div(nut grad u) over the interior: the model's power input.
        double power = 0;
        for (int j = 1; j < n - 1; ++j)
            for (int i = 1; i < n - 1; ++i)
            {
                const auto c = g.idx(i, j, 0);
                power += U.u[c] * R.u[c] + U.v[c] * R.v[c];
            }
        std::printf("eddy power = %.6g\n", power);
        check(std::isfinite(power) && power < 0,
              "eddy diffusion removes energy (power < 0)");
    }

    // --- (6) 3D --------------------------------------------------------------
    {
        const int nz = 8;
        // Cubic cells, h = 1/n: Delta^2 = (h^3)^(2/3) = h^2 = the 2D hx*hy.
        const auto g3 = Grid::make(n, n, nz, 1.0, 1.0, double(nz) / n);
        const auto gg3 = gpu::GpuGrid::from(g3);

        CutGeometry cut3{zeros(g3), zeros(g3), zeros(g3), zeros(g3)};
        for (std::size_t c = 0; c < g3.size(); ++c)
        {
            cut3.lambda[c] = 1.0;
            cut3.ax[c] = cut3.ay[c] = cut3.az[c] = 1.0;
        }
        for (int k = 0; k < nz; ++k)
            for (int j = 12; j < 20; ++j)
                for (int i = 12; i < 20; ++i) cut3.lambda[g3.idx(i, j, k)] = 0.0;
        auto dCut3 = gpu::DeviceCutGeometry::make(cut3);
        auto dU3 = gpu::DeviceField::make(g3.size());
        auto dOut3 = gpu::DeviceField::make(g3.size());
        if (!(dCut3 && dU3 && dOut3)) { std::printf("3d alloc failed\n"); return 1; }

        // z-extrusion of the 2D sheared field: every z term is zero, so nut
        // must reproduce the 2D plane wherever both grids are interior.
        auto U3 = VectorField::zeros(g3);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    U3.u[g3.idx(i, j, k)] = U.u[g.idx(i, j, 0)];
                    U3.v[g3.idx(i, j, k)] = U.v[g.idx(i, j, 0)];
                }
        (void)gpu::uploadField(U3, *dU3);
        auto les3 = gpu::DeviceSmagorinsky::make(gg3, 0.2);
        if (!les3) return 1;
        les3->computeNut(*dU3, *dCut3);
        std::vector<Real> nut3(g3.size());
        (void)les3->nut().download(nut3.data());
        double worst = 0;
        for (int j = 1; j < n - 1; ++j)
            for (int i = 1; i < n - 1; ++i)
            {
                const double a = nutB[g.idx(i, j, 0)];
                const double b = nut3[g3.idx(i, j, nz / 2)];
                worst = std::max(worst, std::abs(a - b));
            }
        check(worst < 1e-6, "extruded field reproduces the 2D nut plane");
        bool ringZ = true;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (nut3[g3.idx(i, j, 0)] != 0 || nut3[g3.idx(i, j, nz - 1)] != 0)
                    ringZ = false;
        check(ringZ, "3d nut == 0 on the z boundary ring");

        // Genuinely 3D field: dissipation and per-component conservation.
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    using std::numbers::pi;
                    const double x = double(i) / n, y = double(j) / n,
                                 z = double(k) / nz;
                    const auto c = g3.idx(i, j, k);
                    U3.u[c] += 0.2 * std::sin(2 * pi * z);
                    U3.w[c] = 0.4 * std::sin(2 * pi * x) * std::cos(2 * pi * y)
                            + 0.1 * std::cos(2 * pi * z);
                }
        (void)gpu::uploadField(U3, *dU3);
        les3->computeNut(*dU3, *dCut3);
        (void)hipMemset(dOut3->u.data(), 0, g3.size() * sizeof(Real));
        (void)hipMemset(dOut3->v.data(), 0, g3.size() * sizeof(Real));
        (void)hipMemset(dOut3->w.data(), 0, g3.size() * sizeof(Real));
        les3->addEddyDiffusion(*dU3, Real(1), *dOut3);
        auto R3 = VectorField::zeros(g3);
        (void)gpu::downloadField(*dOut3, R3);
        double power = 0, su = 0, sv = 0, sw = 0, mag = 0;
        for (std::size_t c = 0; c < g3.size(); ++c)
        {
            power += U3.u[c] * R3.u[c] + U3.v[c] * R3.v[c] + U3.w[c] * R3.w[c];
            su += R3.u[c];
            sv += R3.v[c];
            sw += R3.w[c];
            mag += std::abs(R3.u[c]) + std::abs(R3.v[c]) + std::abs(R3.w[c]);
        }
        std::printf("3d eddy power = %.6g\n", power);
        check(std::isfinite(power) && power < 0, "3d eddy diffusion dissipates");
        check(mag > 0
                  && std::abs(su) < 1e-3 * mag && std::abs(sv) < 1e-3 * mag
                  && std::abs(sw) < 1e-3 * mag,
              "3d flux form conserves momentum over the periodic box");
    }

    // --- (7) sigma model ----------------------------------------------------
    {
        const int nz = 8;
        const auto g3 = Grid::make(n, n, nz, 1.0, 1.0, double(nz) / n);
        const auto gg3 = gpu::GpuGrid::from(g3);
        CutGeometry cut3{zeros(g3), zeros(g3), zeros(g3), zeros(g3)};
        for (std::size_t c = 0; c < g3.size(); ++c)
        {
            cut3.lambda[c] = 1.0;
            cut3.ax[c] = cut3.ay[c] = cut3.az[c] = 1.0;
        }
        for (int k = 0; k < nz; ++k)
            for (int j = 12; j < 20; ++j)
                for (int i = 12; i < 20; ++i) cut3.lambda[g3.idx(i, j, k)] = 0.0;
        auto dCut3 = gpu::DeviceCutGeometry::make(cut3);
        auto dU3 = gpu::DeviceField::make(g3.size());
        auto dOut3 = gpu::DeviceField::make(g3.size());
        if (!(dCut3 && dU3 && dOut3)) { std::printf("3d alloc failed\n"); return 1; }
        auto sig = gpu::DeviceSmagorinsky::make(gg3, 1.35, SgsModel::Sigma);
        auto smag = gpu::DeviceSmagorinsky::make(gg3, 0.2);
        if (!(sig && smag)) return 1;
        std::vector<Real> h(g3.size());

        // Two-component field (the 2D shear extruded in z): Smagorinsky sees
        // strain everywhere, sigma must return EXACTLY zero.
        auto U3 = VectorField::zeros(g3);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    U3.u[g3.idx(i, j, k)] = U.u[g.idx(i, j, 0)];
                    U3.v[g3.idx(i, j, k)] = U.v[g.idx(i, j, 0)];
                }
        (void)gpu::uploadField(U3, *dU3);
        smag->computeNut(*dU3, *dCut3);
        (void)smag->nut().download(h.data());
        double smagMax = 0;
        for (auto v : h) smagMax = std::max(smagMax, double(v));
        sig->computeNut(*dU3, *dCut3);
        (void)sig->nut().download(h.data());
        double sigMax = 0;
        for (auto v : h) sigMax = std::max(sigMax, double(std::abs(v)));
        check(smagMax > 0 && sigMax == 0,
              "sigma nut == 0 on a two-component field (smag > 0)");

        // Genuinely 3D field: sigma switches on, stays a viscosity, and the
        // substep dissipates.
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    using std::numbers::pi;
                    const double x = double(i) / n, y = double(j) / n,
                                 z = double(k) / nz;
                    const auto c = g3.idx(i, j, k);
                    U3.u[c] += 0.2 * std::sin(2 * pi * z);
                    U3.v[c] += 0.15 * std::cos(2 * pi * z) * std::sin(2 * pi * x);
                    U3.w[c] = 0.4 * std::sin(2 * pi * x) * std::cos(2 * pi * y)
                            + 0.1 * std::cos(2 * pi * z);
                }
        (void)gpu::uploadField(U3, *dU3);
        sig->computeNut(*dU3, *dCut3);
        (void)sig->nut().download(h.data());
        bool nonneg3 = true, ring3 = true, solid3 = true, any3 = false;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    const auto c = g3.idx(i, j, k);
                    const double v = h[c];
                    if (v < 0) nonneg3 = false;
                    if (v > 0) any3 = true;
                    if (i == 0 || i == n - 1 || j == 0 || j == n - 1
                        || k == 0 || k == nz - 1)
                        if (v != 0) ring3 = false;
                    if (cut3.lambda[c] == 0.0 && v != 0) solid3 = false;
                }
        check(nonneg3, "sigma nut >= 0 everywhere (3D field)");
        check(any3, "sigma nut > 0 somewhere (3D field)");
        check(ring3, "sigma nut == 0 on the boundary ring");
        check(solid3, "sigma nut == 0 in solid cells");

        (void)hipMemset(dOut3->u.data(), 0, g3.size() * sizeof(Real));
        (void)hipMemset(dOut3->v.data(), 0, g3.size() * sizeof(Real));
        (void)hipMemset(dOut3->w.data(), 0, g3.size() * sizeof(Real));
        sig->addEddyDiffusion(*dU3, Real(1), *dOut3);
        auto R3 = VectorField::zeros(g3);
        (void)gpu::downloadField(*dOut3, R3);
        double power = 0;
        for (std::size_t c = 0; c < g3.size(); ++c)
            power += U3.u[c] * R3.u[c] + U3.v[c] * R3.v[c] + U3.w[c] * R3.w[c];
        std::printf("sigma eddy power = %.6g\n", power);
        check(std::isfinite(power) && power < 0, "sigma eddy diffusion dissipates");
    }

    std::printf("%s\n", dnstest::gFailures == 0 ? "ALL OK" : "FAILURES");
    return dnstest::finish();
}
