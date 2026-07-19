// gpu/derived.hpp: the device-side visualisation fields that replace the host
// slab loops the apps used to run per frame. The point of the test is the
// oracle: each device kernel is compared against the exact host loop it was
// lifted from, so "same field, less PCIe" is checked rather than assumed.
//
// No HIP device -> skip (exit 77).

#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"

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

double maxDiff(const std::vector<dns::gpu::Real>& a, const std::vector<double>& b)
{
    double m = 0;
    for (std::size_t c = 0; c < a.size(); ++c)
        m = std::max(m, std::abs(double(a[c]) - b[c]));
    return m;
}

} // namespace

int main()
{
    using namespace dns;
    using gpu::Real;
    using std::numbers::pi;

    if (!gpu::deviceAvailable())
    {
        std::printf("no HIP device, skipping\n");
        return skipExitCode;
    }

    for (int nz : {1, 8})
    {
        const int n = 16;
        const auto g = Grid::make(n, n, nz, 1.0, 1.0, nz == 1 ? 1.0 : 0.5);
        const auto gg = gpu::GpuGrid::from(g);

        auto U = VectorField::zeros(g);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    const auto c = g.idx(i, j, k);
                    const double x = double(i) / n, y = double(j) / n,
                                 z = double(k) / nz;
                    U.u[c] = std::sin(2 * pi * y) + 0.2 * std::cos(2 * pi * z);
                    U.v[c] = 0.7 * std::sin(2 * pi * x);
                    U.w[c] = nz == 1 ? 0.0 : 0.4 * std::cos(2 * pi * y);
                }

        auto dU = gpu::DeviceField::make(g.size());
        auto dOut = gpu::DeviceScalar::make(g.size());
        if (!dU || !dOut) { std::printf("alloc failed\n"); return 1; }
        (void)gpu::uploadField(U, *dU);

        std::vector<Real> got(g.size());

        // --- vorticityZ vs the 2D apps' host stencil -----------------------
        gpu::vorticityZ(gg, *dU, *dOut);
        (void)dOut->download(got.data());
        std::vector<double> want(g.size());
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    const auto c = g.idx(i, j, k);
                    want[c] = (U.v[c] - U.v[g.idx(i - 1, j, k)]) / g.hx
                            - (U.u[c] - U.u[g.idx(i, j - 1, k)]) / g.hy;
                }
        check(maxDiff(got, want) < 1e-3, "vorticityZ matches the host stencil");

        // --- speedMac vs the host face-average ------------------------------
        gpu::speedMac(gg, *dU, *dOut);
        (void)dOut->download(got.data());
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    const auto c = g.idx(i, j, k);
                    const double uc = 0.5 * (U.u[c] + U.u[g.idx(i + 1, j, k)]);
                    const double vc = 0.5 * (U.v[c] + U.v[g.idx(i, j + 1, k)]);
                    const double wc = 0.5 * (U.w[c] + U.w[g.idx(i, j, k + 1)]);
                    want[c] = std::sqrt(uc * uc + vc * vc + wc * wc);
                }
        check(maxDiff(got, want) < 1e-5, "speedMac matches the host stencil");

        // --- centred curl / speed vs the comp_mesh host stencil -------------
        auto du = gpu::DeviceScalar::make(g.size());
        auto dv = gpu::DeviceScalar::make(g.size());
        auto dw = gpu::DeviceScalar::make(g.size());
        auto drho = gpu::DeviceScalar::make(g.size());
        auto dmx = gpu::DeviceScalar::make(g.size());
        auto dmy = gpu::DeviceScalar::make(g.size());
        auto dmz = gpu::DeviceScalar::make(g.size());
        if (!du || !dv || !dw || !drho || !dmx || !dmy || !dmz) return 1;

        // rho = 2 everywhere: primitiveVelocity must divide it out exactly.
        std::vector<Real> hr(g.size(), Real(2)), hmx(g.size()), hmy(g.size()),
            hmz(g.size());
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            hmx[c] = Real(2 * U.u[c]);
            hmy[c] = Real(2 * U.v[c]);
            hmz[c] = Real(2 * U.w[c]);
        }
        (void)drho->upload(hr.data());
        (void)dmx->upload(hmx.data());
        (void)dmy->upload(hmy.data());
        (void)dmz->upload(hmz.data());
        gpu::primitiveVelocity(gg, *drho, *dmx, *dmy, *dmz, *du, *dv, *dw);
        (void)du->download(got.data());
        for (std::size_t c = 0; c < g.size(); ++c) want[c] = U.u[c];
        check(maxDiff(got, want) < 1e-5, "primitiveVelocity divides rho out");

        gpu::curlMagCentred(gg, *du, *dv, *dw, *dOut);
        (void)dOut->download(got.data());
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    const auto c = g.idx(i, j, k);
                    auto at = [&](const Scalar& q, int a, int b, int cc) {
                        return q[g.idx(a, b, cc)];
                    };
                    const double dwdy = (at(U.w, i, j + 1, k) - at(U.w, i, j - 1, k)) / (2 * g.hy);
                    const double dvdz = (at(U.v, i, j, k + 1) - at(U.v, i, j, k - 1)) / (2 * g.hz);
                    const double dudz = (at(U.u, i, j, k + 1) - at(U.u, i, j, k - 1)) / (2 * g.hz);
                    const double dwdx = (at(U.w, i + 1, j, k) - at(U.w, i - 1, j, k)) / (2 * g.hx);
                    const double dvdx = (at(U.v, i + 1, j, k) - at(U.v, i - 1, j, k)) / (2 * g.hx);
                    const double dudy = (at(U.u, i, j + 1, k) - at(U.u, i, j - 1, k)) / (2 * g.hy);
                    const double wx = dwdy - dvdz, wy = dudz - dwdx, wz = dvdx - dudy;
                    want[c] = std::sqrt(wx * wx + wy * wy + wz * wz);
                }
        check(maxDiff(got, want) < 5e-3, "curlMagCentred matches the host stencil");

        gpu::speedCentred(gg, *du, *dv, *dw, *dOut);
        (void)dOut->download(got.data());
        for (std::size_t c = 0; c < g.size(); ++c)
            want[c] = std::sqrt(U.u[c] * U.u[c] + U.v[c] * U.v[c] + U.w[c] * U.w[c]);
        check(maxDiff(got, want) < 1e-5, "speedCentred matches the host stencil");

        std::printf("  (nz = %d passed)\n", nz);
    }

    std::printf("%s\n", pass ? "ALL OK" : "FAILURES");
    return pass ? 0 : 1;
}
