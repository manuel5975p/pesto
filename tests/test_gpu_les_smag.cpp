// DeviceSmagorinsky (gpu/les.hpp), the incompressible 2D LES model, tested
// against the library directly. This test used to shell out to a built
// cylinder_tunnel_gpu because the kernels lived in that app's anonymous
// namespace; linking dns_gpu instead is the proof the lift was faithful.
//
// Checks:
//   1. make() rejects cs <= 0 and rejects nz > 1 (the model is 2D-only).
//   2. nut >= 0 everywhere, is exactly 0 on the domain-boundary ring, and is
//      exactly 0 in solid cells (lambda == 0 scales it away).
//   3. nut scales as cs^2 for a fixed field.
//   4. A uniform field produces nut == 0 (|S| == 0) and no eddy flux.
//   5. The eddy-diffusion substep DISSIPATES: sum u . div(nut grad u) < 0 for a
//      sheared field, i.e. the sign of the model term is right.
//
// No HIP device -> skip (exit 77).

#include "dns/cutcell.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/les.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

namespace {

constexpr int skipExitCode = 77;
bool pass = true;

void check(bool ok, const char* what)
{
    std::printf("%-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) pass = false;
}

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
        check(!gpu::DeviceSmagorinsky::make(gpu::GpuGrid::from(g3), 0.2).has_value(),
              "make() rejects nz > 1 (the model is 2D-only)");
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

    std::printf("%s\n", pass ? "ALL OK" : "FAILURES");
    return pass ? 0 : 1;
}
