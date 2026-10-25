// Linear viscous and thermal modes of the compressible solver against their
// EXACT decay rates, on four paths: CPU Cartesian, CPU curvilinear (smooth
// sinusoidal skew), GPU Cartesian, GPU focal (clustered along the mode).
//
//   shear wave   u_t = eps sin(k q) with q the mode axis and t a transverse
//                velocity component: uncoupled from p and T at linear order,
//                decays at exactly nu k^2. Run with q = y (u(y)), q = x
//                (v(x)) and, in 3D, q = z (u(z)).
//   entropy mode T' = eps cos(k q): the real root s of the linearized 1D
//                Navier-Stokes dispersion relation (rho0 = 1, R = 1)
//                  cv s^3 + (b + a cv) s^2 + (a b + k^2 T0 (cv + 1)) s
//                     + k^2 T0 b = 0,   a = 4/3 mu k^2, b = kappa k^2,
//                started on its exact eigenvector so no acoustic pair rides
//                along. For k chi << c (here 0.07) the root is the ISOBARIC
//                rate kappa k^2 / (rho cp) = nu k^2 / Pr to O((k chi/c)^2),
//                NOT the isochoric gamma nu k^2 / Pr, which is a conduction
//                rate at fixed density and the right number only for the
//                diffusive dt bound. Both are printed.
//
// The scheme's normal viscous and conductive gradients are compact two-point
// differences, whose symbol is k^2 (1 - (kh)^2/12 + ...): the rate is low by
// 0.32% at 32 cells per wavelength and the deficit is second order. So, per
// path and mode: |rate/exact - 1| < 1% at n = 32 and log2(d16/d32) in
// [1.8, 2.2]. Amplitudes are volume-weighted projections onto the mode shape
// at the physical cell centroids; eps = 1e-2 keeps the O(eps^2) nonlinear
// feedback (viscous heating, rho(T) in the energy equation) at ~1e-4 of the
// rate, under the 0.32% being measured.
// The CPU half runs without a device; the test then skips (77) unless it
// failed.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "curv_test_util.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using std::numbers::pi;

namespace {

constexpr double kGamma = 1.4, kMu = 0.05, kEps = 1e-2, kTEnd = 10.0;
constexpr double kL = 2 * pi, kK = 2 * pi / kL; // one wavelength per box
const double kT0 = 1.0 / kGamma;                // p0 = 1/gamma, c0 = 1

enum class Path { CpuCart, CpuCurv, GpuCart, GpuFocal };
enum class Mode { Shear, Entropy };

const char* pathName(Path p)
{
    switch (p) {
    case Path::CpuCart: return "CPU Cartesian";
    case Path::CpuCurv: return "CPU curvilinear";
    case Path::GpuCart: return "GPU Cartesian";
    case Path::GpuFocal: return "GPU focal";
    }
    return "?";
}

// The exact entropy-mode root (Newton from the isobaric estimate) and its
// eigenvector: T^ = 1, rho^ = (cv s + b)/(T0 s), longitudinal velocity
// amplitude c^ = (cv s + b)/(k T0) in quadrature (u_q = -eps c^ sin(k q)).
struct EntropyMode
{
    double s, rhoHat, uHat;
};

EntropyMode entropyMode()
{
    const Transport tr = Transport::constant(kGamma, kMu);
    const double cv = 1.0 / (kGamma - 1.0), k2 = kK * kK;
    const double a = 4.0 / 3.0 * kMu * k2, b = tr.kappa(kT0) * k2;
    const double c3 = cv, c2 = b + a * cv, c1 = a * b + k2 * kT0 * (cv + 1.0),
                 c0 = k2 * kT0 * b;
    double s = -b / (cv + 1.0);
    for (int it = 0; it < 50; ++it)
        s -= (((c3 * s + c2) * s + c1) * s + c0) / ((3 * c3 * s + 2 * c2) * s + c1);
    return {s, (cv * s + b) / (kT0 * s), (cv * s + b) / (kK * kT0)};
}

struct Setup
{
    Grid g;
    std::optional<CurvMesh> mesh;
    std::vector<std::array<double, 3>> pos; // physical centroids
    std::vector<double> vol;
};

// Grid of n cells along the mode axis q (3 for z) and 4 transverse; the
// curvilinear path uses an n x n sinusoidally skewed box instead (its
// distortion needs both axes resolved).
Setup makeSetup(Path p, int q, int n)
{
    const double h = kL / n;
    Setup s{};
    if (p == Path::CpuCurv) {
        s.g = Grid::make(n, n, 1, kL, kL, 1.0);
        s.mesh = CurvMesh::sinusoidal(s.g, 0.04);
    } else {
        const int nn[3] = {q == 0 ? n : 4, q == 1 ? n : 4, q == 2 ? n : 1};
        s.g = Grid::make(nn[0], nn[1], nn[2], nn[0] * h, nn[1] * h,
                         q == 2 ? kL : 1.0);
        if (p == Path::GpuFocal) {
            // Ratio-2 plateau on the mode axis, physical width ~ 1/4 box.
            const double L[3] = {s.g.nx * s.g.hx, s.g.ny * s.g.hy, s.g.nz * s.g.hz};
            Focal1D f[3] = {Focal1D::identity(L[0]), Focal1D::identity(L[1]),
                            Focal1D::identity(L[2])};
            f[q] = Focal1D::make(kL, 2.0, 3.5, 1.0, 2.0);
            s.mesh = CurvMesh::fromRemap(s.g, AxisRemap3{f[0], f[1], f[2]});
        }
    }
    const Grid& g = s.g;
    s.pos.resize(g.size());
    s.vol.assign(g.size(), g.cellVolume());
    if (s.mesh) {
        auto mm = CurvMetrics::build(*s.mesh);
        if (mm) s.vol = mm->vol;
    }
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                s.pos[g.idx(i, j, k)] =
                    s.mesh ? curvtest::centroid(*s.mesh, i, j, k)
                           : std::array<double, 3>{(i + 0.5) * g.hx, (j + 0.5) * g.hy,
                                                   (k + 0.5) * g.hz};
    return s;
}

// Transverse velocity component of the shear wave along axis q.
int shearComp(int q) { return q == 0 ? 1 : 0; }

CompState initial(const Setup& su, Mode mode, int q, const EntropyMode& em)
{
    const Grid& g = su.g;
    auto st = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        const double ph = kK * su.pos[c][q];
        double rho = 1.0, T = kT0, u[3] = {0, 0, 0};
        if (mode == Mode::Shear) {
            u[shearComp(q)] = kEps * std::sin(ph);
        } else {
            T = kT0 + kEps * std::cos(ph);
            rho = 1.0 + kEps * em.rhoHat * std::cos(ph);
            u[q] = -kEps * em.uHat * std::sin(ph);
        }
        st.rho[c] = rho;
        st.mx[c] = rho * u[0];
        st.my[c] = rho * u[1];
        st.mz[c] = rho * u[2];
        st.E[c] = rho * T / (kGamma - 1.0)
                  + 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    }
    return st;
}

// Volume-weighted projection of the mode observable onto its shape.
double amplitude(const Setup& su, const CompState& st, Mode mode, int q)
{
    double num = 0.0, den = 0.0;
    for (std::size_t c = 0; c < su.g.size(); ++c) {
        const double ph = kK * su.pos[c][q];
        double f, phi;
        if (mode == Mode::Shear) {
            const double m[3] = {st.mx[c], st.my[c], st.mz[c]};
            f = m[shearComp(q)] / st.rho[c];
            phi = std::sin(ph);
        } else {
            const double ke = 0.5 * (st.mx[c] * st.mx[c] + st.my[c] * st.my[c]
                                     + st.mz[c] * st.mz[c]) / st.rho[c];
            f = (kGamma - 1.0) * (st.E[c] - ke) / st.rho[c] - kT0;
            phi = std::cos(ph);
        }
        num += su.vol[c] * f * phi;
        den += su.vol[c] * phi * phi;
    }
    return num / den;
}

// Measured decay rate -ln(A(T)/A(0))/T; NaN on a setup failure.
double measure(Path p, Mode mode, int q, int n, const EntropyMode& em)
{
    const Setup su = makeSetup(p, q, n);
    const CompState s0 = initial(su, mode, q, em);
    const double a0 = amplitude(su, s0, mode, q);
    CompState s = s0;
    const bool gpuPath = p == Path::GpuCart || p == Path::GpuFocal;
    if (!gpuPath) {
        std::optional<Comp> c;
        if (su.mesh) {
            auto cc = Comp::make(*su.mesh, kGamma, kMu);
            if (!cc) return std::nan("");
            c.emplace(std::move(*cc));
        } else {
            c.emplace(Comp::make(su.g, kGamma, kMu));
        }
        const int steps = int(std::ceil(kTEnd / *c->stableDt(s)));
        for (int it = 0; it < steps; ++it) c->step(s, kTEnd / steps);
    } else {
        std::optional<gpu::DeviceComp> dev;
        if (su.mesh) {
            auto m = CurvMetrics::build(*su.mesh);
            if (!m) return std::nan("");
            auto d = gpu::DeviceComp::make(*m, kGamma, kMu);
            if (!d) return std::nan("");
            dev.emplace(std::move(*d));
        } else {
            auto d = gpu::DeviceComp::make(su.g, kGamma, kMu);
            if (!d) return std::nan("");
            dev.emplace(std::move(*d));
        }
        auto ds = gpu::DeviceCompState::make(su.g.size());
        auto red = gpu::ReduceScratch::make();
        if (!ds || !red) return std::nan("");
        (void)ds->upload(s);
        const int steps = int(std::ceil(kTEnd / dev->stableDt(*ds, *red)));
        for (int it = 0; it < steps; ++it)
            dev->step(*ds, gpu::Real(kTEnd / steps));
        (void)ds->download(s);
    }
    return -std::log(amplitude(su, s, mode, q) / a0) / kTEnd;
}

void modeCase(Path p, Mode mode, int q, double exact)
{
    const EntropyMode em = entropyMode();
    const double r16 = measure(p, mode, q, 16, em);
    const double r32 = measure(p, mode, q, 32, em);
    const double d16 = std::abs(r16 / exact - 1.0), d32 = std::abs(r32 / exact - 1.0);
    const double order = std::log2(d16 / d32);
    const char ax = "xyz"[q];
    const std::string tag = std::string(pathName(p))
                            + (mode == Mode::Shear ? " shear along " : " entropy along ")
                            + ax;
    std::printf("%-34s exact %.6e  rate n16 %.6e (%+.3f%%)  n32 %.6e (%+.3f%%)  "
                "order %.3f\n", tag.c_str(), exact, r16, 100 * (r16 / exact - 1),
                r32, 100 * (r32 / exact - 1), order);
    check(std::isfinite(r32) && d32 < 0.01, tag + ": rate within 1% at n = 32");
    check(order >= 1.8 && order <= 2.2, tag + ": error order in [1.8, 2.2]");
}

void allModes(Path p)
{
    const double shear = kMu * kK * kK;
    const EntropyMode em = entropyMode();
    modeCase(p, Mode::Shear, 1, shear);
    modeCase(p, Mode::Shear, 0, shear);
    if (p != Path::CpuCurv) modeCase(p, Mode::Shear, 2, shear);
    modeCase(p, Mode::Entropy, 1, -em.s);
}

} // namespace

int main()
{
    {
        const Transport tr = Transport::constant(kGamma, kMu);
        const EntropyMode em = entropyMode();
        const double k2 = kK * kK;
        std::printf("entropy root %.6e; isobaric nu k^2/Pr %.6e; isochoric "
                    "gamma nu k^2/Pr %.6e (Pr %.2f)\n", -em.s, kMu * k2 / tr.Pr,
                    kGamma * kMu * k2 / tr.Pr, tr.Pr);
    }
    allModes(Path::CpuCart);
    allModes(Path::CpuCurv);
    if (!gpu::deviceAvailable()) {
        if (dnstest::gFailures) return dnstest::finish();
        dnstest::skip("no HIP device (CPU half passed)");
    }
    allModes(Path::GpuCart);
    allModes(Path::GpuFocal);
    return dnstest::finish();
}
