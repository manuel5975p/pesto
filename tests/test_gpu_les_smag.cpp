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
//   8. Magnitude (W10): on linear velocity fields the discrete gradient is
//      EXACT in FP32, so nut is pinned to its closed form to 4 ulp. The code's
//      convention is |S| = sqrt(2 S_ij S_ij) (u = a y gives |S| = |a|, not
//      |a|/sqrt 2) and Delta^2 = (hx hy hz)^(2/3) (hx hy at nz == 1), the
//      cell's own vol^(2/3) on a curvilinear mesh. Incompressible (nut) and
//      compressible (mu_t = rho nut, checkerboard rho), uniform and focal
//      (compressible only: the incompressible model has no mesh map), and
//      sigma on a fixed rank-3 gradient against host FP64 sigmaDsigma (focal
//      sigma: on the FP64 discrete gradient, 3e-5 rel -- the mapped FP32
//      gradient's rounding, amplified by D_sigma, is not a 4-ulp quantity).
//
// `test_gpu_les_smag tally3d` (ctest gpu_les_tally3d) instead runs the 3D SGS
// wall tally: a 48^3 sphere, D = 16 cells, nu = 0, one eddy-diffusion substep
// on a compact field around it. The flux form telescopes, so the momentum the
// substep removes from the open faces, -sum(increment) vol / dt, is exactly
// the drain through the closed links -- which DeviceBodyForce::measure must
// report in cd/cl/cs (qRef = 1) to 1e-4 per component, z links included.
//
// No HIP device -> skip (exit 77).

#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/cutcell.hpp"
#include "dns/remap.hpp"
#include "dns/sgs.hpp"
#include "dns/gpu/bodyforce.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "dns/gpu/les.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numbers>
#include <vector>

namespace {

using dnstest::check;

int tally3d()
{
    using namespace dns;
    using gpu::Real;
    using std::numbers::pi;
    const int n = 48;
    const auto g = Grid::make(n, n, n, 1.0, 1.0, 1.0);
    const auto gg = gpu::GpuGrid::from(g);
    const double cx = 0.5 + 0.013, cy = 0.5 - 0.009, cz = 0.5 + 0.004,
                 r = 8.0 / n, Rb = 0.38;
    auto cut = CutGeometry::make(g, Sphere{cx, cy, cz, r});
    (void)sanitizeCut(g, cut);
    // Compact, genuinely 3D field: the bump f vanishes (with its gradient)
    // at Rb, well inside the box, so no eddy flux crosses the domain ring.
    auto U = VectorField::zeros(g);
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy,
                             z = (k + 0.5) * g.hz;
                const double q = std::sqrt((x - cx) * (x - cx) +
                                           (y - cy) * (y - cy) +
                                           (z - cz) * (z - cz)) / Rb;
                if (q >= 1.0) continue;
                const double f = (1 - q * q) * (1 - q * q);
                const std::size_t c = g.idx(i, j, k);
                U.u[c] = f * (1.0 + 0.4 * std::sin(2 * pi * y));
                U.v[c] = f * 0.5 * std::sin(2 * pi * (x + z));
                U.w[c] = f * 0.6 * std::cos(2 * pi * (x - y));
            }
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    auto dU = gpu::DeviceField::make(g.size());
    auto dOut = gpu::DeviceField::make(g.size());
    auto phi = gpu::DeviceScalar::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto les = gpu::DeviceSmagorinsky::make(gg, 0.17);
    auto bf = gpu::DeviceBodyForce::make(gg, gpu::bodyBox(gg, cx, cy, cz, r));
    if (!dCut || !dU || !dOut || !phi || !red || !les || !bf) {
        std::printf("tally3d setup failed\n");
        return 1;
    }
    (void)gpu::uploadField(U, *dU);
    gpu::maskSolidFaces(gg, *dCut, *dU); // the state's closed faces are zero
    (void)hipMemset(phi->data(), 0, g.size() * sizeof(Real));
    for (Real* p : {dOut->u.data(), dOut->v.data(), dOut->w.data()})
        (void)hipMemset(p, 0, g.size() * sizeof(Real));
    const double dt = 1e-2;
    les->computeNut(*dU, *dCut);
    les->addEddyDiffusion(*dU, Real(dt), *dOut);
    gpu::maskSolidFaces(gg, *dCut, *dOut); // as the runner's stage does
    auto R = VectorField::zeros(g);
    (void)gpu::downloadField(*dOut, R);
    const double vol = g.hx * g.hy * g.hz;
    double drain[3] = {0, 0, 0};
    for (std::size_t c = 0; c < g.size(); ++c) {
        drain[0] -= R.u[c] * vol / dt;
        drain[1] -= R.v[c] * vol / dt;
        drain[2] -= R.w[c] * vol / dt;
    }
    const auto co = bf->measure(*dCut, *phi, *dU, 0.0, 1.0, &les->nut(), *red);
    const double tal[3] = {co.cd, co.cl, co.cs};
    const double scale = std::max({std::abs(drain[0]), std::abs(drain[1]),
                                   std::abs(drain[2])});
    const char* name[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; ++a) {
        const double rel = std::abs(tal[a] - drain[a]) / scale;
        std::printf("tally3d %s: SGS drain %+.6e  tally %+.6e  rel %.2e\n",
                    name[a], drain[a], tal[a], rel);
        check(std::abs(drain[a]) > 1e-2 * scale,
              std::string("tally3d ") + name[a] +
                  ": the component is exercised");
        check(rel < 1e-4, std::string("tally3d ") + name[a] +
                              ": SGS tally == the substep's wall drain");
    }
    check(co.cdf == 0.0, "tally3d: cdf stays nu-only (nu = 0)");
    return dnstest::finish();
}

// --- (8) magnitude ------------------------------------------------------------

constexpr double kShear = 3.0;            // u = a y: |S| = sqrt(2 S:S) = a
constexpr double kCs = 0.17, kCsig = 1.35;
// Fixed rank-3 velocity gradient g[c][d] = du_c/dx_d with distinct singular
// values. Dyadic entries: G x is exact in FP32 at the dyadic cell positions of
// a power-of-two grid, so every central difference returns G bit-exactly.
constexpr double kG[3][3] = {{0.5, 1.25, -0.75},
                             {-0.5, 0.25, 1.0},
                             {0.75, -1.0, -0.75}};

// |dev - ref| in units of the FP32 spacing at |ref|.
double ulps(double dev, double ref)
{
    const float r = float(std::abs(ref));
    const double u = double(std::nextafter(r, INFINITY)) - double(r);
    return std::abs(dev - ref) / u;
}

// The filter-width convention under test, FP64: (hx hy hz)^(2/3), hx hy at
// nz == 1 (the unit depth carries no filter width).
double delta2(const dns::Grid& g)
{
    if (g.nz == 1) return g.hx * g.hy;
    const double c = std::cbrt(g.hx * g.hy * g.hz);
    return c * c;
}

// Cells whose central stencil wraps nowhere (a linear field is not periodic);
// for the incompressible kernel this is also exactly the non-ring set.
bool interior(const dns::Grid& g, int i, int j, int k)
{
    return i >= 1 && i <= g.nx - 2 && j >= 1 && j <= g.ny - 2
           && (g.nz == 1 || (k >= 1 && k <= g.nz - 2));
}

double rhoAt(int i, int j, int k) { return ((i + j + k) & 1) ? 2.0 : 1.0; }

// Worst FP32 ulp distance of DeviceSmagorinsky nut from expect(i, j, k) over
// the interior of an all-fluid box carrying the MAC field U.
template <class Expect>
double incompUlps(const dns::Grid& g, const dns::VectorField& U, double c,
                  dns::SgsModel model, Expect expect)
{
    using namespace dns;
    const auto gg = gpu::GpuGrid::from(g);
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    for (std::size_t q = 0; q < g.size(); ++q)
        cut.lambda[q] = cut.ax[q] = cut.ay[q] = cut.az[q] = 1.0;
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    auto dU = gpu::DeviceField::make(g.size());
    auto les = gpu::DeviceSmagorinsky::make(gg, c, model);
    if (!dCut || !dU || !les) return HUGE_VAL;
    (void)gpu::uploadField(U, *dU);
    les->computeNut(*dU, *dCut);
    std::vector<gpu::Real> h(g.size());
    (void)les->nut().download(h.data());
    double worst = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                if (interior(g, i, j, k))
                    worst = std::max(worst,
                                     ulps(h[g.idx(i, j, k)], expect(i, j, k)));
    return worst;
}

// Compressible state with velocity vel(i, j, k, u[3]) rounded to FP32 and
// m = rho u on a checkerboard rho in {1, 2}, so the device's u = m / rho is
// the same float. p = 1 (nut reads rho and u only).
template <class Vel>
dns::CompState linearState(const dns::Grid& g, Vel vel)
{
    auto s = dns::CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                double u[3];
                vel(i, j, k, u);
                for (double& x : u) x = double(float(x));
                const double r = rhoAt(i, j, k);
                s.rho[c] = r;
                s.mx[c] = r * u[0];
                s.my[c] = r * u[1];
                s.mz[c] = r * u[2];
                s.E[c] = 1.0 / 0.4
                         + 0.5 * r * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            }
    return s;
}

// mu_t of s as DeviceComp::updateSgsNut leaves it in mut(); empty on failure.
std::vector<dns::gpu::Real> compMut(dns::gpu::DeviceComp& dev,
                                    const dns::Grid& g, const dns::CompState& s)
{
    auto ds = dns::gpu::DeviceCompState::make(g.size());
    if (!ds) return {};
    (void)ds->upload(s);
    dev.updateSgsNut(*ds);
    std::vector<dns::gpu::Real> h(g.size());
    (void)dev.mut().download(h.data());
    return h;
}

template <class Expect>
double maxUlpsInterior(const dns::Grid& g, const std::vector<dns::gpu::Real>& h,
                       Expect expect)
{
    if (h.size() != g.size()) return HUGE_VAL;
    double worst = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                if (interior(g, i, j, k))
                    worst = std::max(worst,
                                     ulps(h[g.idx(i, j, k)], expect(i, j, k)));
    return worst;
}

void magnitude()
{
    using namespace dns;
    using gpu::Real;
    const double ulpTol = 4.0;
    // Uniform grids: 2D square, and 3D anisotropic so the (hx hy hz)^(2/3)
    // product is pinned against max/min(h) readings.
    const Grid g2 = Grid::make(32, 32, 1, 1.0, 1.0, 1.0);
    const Grid g3 = Grid::make(16, 32, 8, 1.0, 1.0, 1.0);

    // Incompressible Smagorinsky on u = a y (MAC u-faces at y = (j+1/2) hy).
    for (const Grid* g : {&g2, &g3}) {
        auto U = VectorField::zeros(*g);
        for (int k = 0; k < g->nz; ++k)
            for (int j = 0; j < g->ny; ++j)
                for (int i = 0; i < g->nx; ++i)
                    U.u[g->idx(i, j, k)] = kShear * (j + 0.5) * g->hy;
        const double ref = kCs * kCs * delta2(*g) * kShear;
        const double e = incompUlps(*g, U, kCs, SgsModel::Smagorinsky,
                                    [&](int, int, int) { return ref; });
        std::printf("magnitude incomp smag nz=%d: nut %.9g, worst %.2f ulp\n",
                    g->nz, ref, e);
        check(e <= ulpTol, "incomp smag nut == (Cs Delta)^2 |S| (nz=" +
                               std::to_string(g->nz) + ") to 4 ulp");
    }

    // Incompressible sigma on the rank-3 field u_c = G_cd x_d (each MAC
    // component at its own face position).
    {
        const Grid& g = g3;
        auto U = VectorField::zeros(g);
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    const double xc = (i + 0.5) * g.hx, yc = (j + 0.5) * g.hy,
                                 zc = (k + 0.5) * g.hz;
                    const double pos[3][3] = {{i * g.hx, yc, zc},
                                              {xc, j * g.hy, zc},
                                              {xc, yc, k * g.hz}};
                    double* comp[3] = {&U.u[g.idx(i, j, k)],
                                       &U.v[g.idx(i, j, k)],
                                       &U.w[g.idx(i, j, k)]};
                    for (int c = 0; c < 3; ++c)
                        *comp[c] = kG[c][0] * pos[c][0] + kG[c][1] * pos[c][1]
                                   + kG[c][2] * pos[c][2];
                }
        // The reference is the FP64 operator. A host FP32 replica is NOT one:
        // measured 13 ulp off the device (contraction and rounding order in
        // the singular-value algebra), while the device sits within 0.1 ulp
        // of FP64 -- so the FP64 value is the tighter, honest pin.
        const double ref64 = kCsig * kCsig * delta2(g) * sigmaDsigma(kG);
        const double e = incompUlps(g, U, kCsig, SgsModel::Sigma,
                                    [&](int, int, int) { return ref64; });
        std::printf("magnitude incomp sigma: D_sigma %.9g, nut %.9g, worst "
                    "%.2f ulp\n", sigmaDsigma(kG), ref64, e);
        check(ref64 > 0, "the fixed gradient is genuinely rank 3 (D_sigma > 0)");
        check(e <= ulpTol,
              "incomp sigma nut == (Csigma Delta)^2 sigmaDsigma to 4 ulp");
    }

    // Compressible, uniform: mu_t = rho (Cs Delta)^2 |S| on u = a y (cell
    // centres), rho on a checkerboard; and sigma on u_c = G_cd x_d.
    for (const Grid* g : {&g2, &g3}) {
        auto dev = gpu::DeviceComp::make(*g, 1.4, 1e-3, kCs);
        if (!dev) { check(false, "comp make"); return; }
        const auto s = linearState(*g, [&](int, int j, int, double u[3]) {
            u[0] = kShear * (j + 0.5) * g->hy;
            u[1] = u[2] = 0.0;
        });
        const double base = kCs * kCs * delta2(*g) * kShear;
        const double e = maxUlpsInterior(*g, compMut(*dev, *g, s),
                                         [&](int i, int j, int k) {
                                             return rhoAt(i, j, k) * base;
                                         });
        std::printf("magnitude comp smag nz=%d: mu_t/rho %.9g, worst %.2f ulp\n",
                    g->nz, base, e);
        check(e <= ulpTol, "comp smag mu_t == rho (Cs Delta)^2 |S| (nz=" +
                               std::to_string(g->nz) + ") to 4 ulp");
    }
    {
        const Grid& g = g3;
        auto dev = gpu::DeviceComp::make(g, 1.4, 1e-3, kCsig, SgsModel::Sigma);
        if (!dev) { check(false, "comp sigma make"); return; }
        const auto s = linearState(g, [&](int i, int j, int k, double u[3]) {
            const double x[3] = {(i + 0.5) * g.hx, (j + 0.5) * g.hy,
                                 (k + 0.5) * g.hz};
            for (int c = 0; c < 3; ++c)
                u[c] = kG[c][0] * x[0] + kG[c][1] * x[1] + kG[c][2] * x[2];
        });
        const auto h = compMut(*dev, g, s);
        const double e = maxUlpsInterior(g, h, [&](int i, int j, int k) {
            return rhoAt(i, j, k) * kCsig * kCsig * delta2(g) * sigmaDsigma(kG);
        });
        std::printf("magnitude comp sigma: worst %.2f ulp\n", e);
        check(e <= ulpTol, "comp sigma mu_t == rho (Csigma Delta)^2 "
                           "sigmaDsigma to 4 ulp");
    }

    // Compressible, focal: x (and z in 3D) clustered at ratio 3, y the exact
    // identity. u = a y then has an exact discrete gradient on the mapped mesh
    // (the map is separable, y is uniform), and the filter width is the cell's
    // own vol^(1/3) -- computed here from the 1D maps, independent of
    // CurvMetrics. 2D: Delta^2 = vol / hz.
    for (const int nz : {1, 32}) {
        const Grid g = Grid::make(32, 16, nz, 1.0, 1.0, 1.0);
        const double lo[3] = {0.45, 0.0, 0.45},
                     hi[3] = {0.55, 0.0, nz > 1 ? 0.55 : 0.0},
                     w[3] = {0.1, 0.1, 0.1};
        const AxisRemap3 fr = focalRemap(g, true, lo, hi, w, 3.0);
        check(fr.fx.active && !fr.fy.active && fr.fz.active == (nz > 1)
                  && fr.fx.clears(2 * g.hx),
              "focal remap: x" + std::string(nz > 1 ? "/z" : "")
                  + " clustered, y identity");
        auto met = CurvMetrics::build(CurvMesh::fromRemap(g, fr));
        if (!met) { check(false, "focal metrics: " + met.error()); return; }
        auto dev = gpu::DeviceComp::make(*met, 1.4, 1e-3, kCs);
        if (!dev) { check(false, "focal comp make"); return; }
        auto width = [&](const Focal1D& f, double h, int i) {
            return f.map((i + 1) * h) - f.map(i * h);
        };
        auto vol = [&](int i, int j, int k) {
            (void)j;
            return width(fr.fx, g.hx, i) * g.hy
                   * (nz > 1 ? width(fr.fz, g.hz, k) : g.hz);
        };
        auto d2 = [&](int i, int j, int k) {
            const double v = vol(i, j, k);
            if (nz == 1) return v / g.hz;
            const double c = std::cbrt(v);
            return c * c;
        };
        double volErr = 0, vmin = HUGE_VAL, vmax = 0;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    const double v = vol(i, j, k);
                    volErr = std::max(volErr,
                                      std::abs(met->vol[g.idx(i, j, k)] / v - 1));
                    vmin = std::min(vmin, v);
                    vmax = std::max(vmax, v);
                }
        check(volErr < 1e-12, "focal: CurvMetrics volume == product of the 1D "
                              "map widths");
        check(vmax / vmin > (nz > 1 ? 8.0 : 2.5),
              "focal: the filter width genuinely varies across the box");
        const auto s = linearState(g, [&](int, int j, int, double u[3]) {
            u[0] = kShear * (j + 0.5) * g.hy;
            u[1] = u[2] = 0.0;
        });
        const auto h = compMut(*dev, g, s);
        const double e = maxUlpsInterior(g, h, [&](int i, int j, int k) {
            return rhoAt(i, j, k) * kCs * kCs * d2(i, j, k) * kShear;
        });
        std::printf("magnitude comp focal smag nz=%d: vol ratio %.2f, worst "
                    "%.2f ulp\n", nz, vmax / vmin, e);
        check(e <= ulpTol, "focal comp smag mu_t == rho (Cs vol^(1/3))^2 |S| "
                           "(nz=" + std::to_string(nz) + ") to 4 ulp");
        if (nz == 1) continue;

        // Focal sigma: the discrete gradient of G x on a stretched axis is not
        // G, so the reference is curvVelGradCentre (index-space central
        // differences x the face-averaged inverse Jacobian) evaluated in FP64
        // on the device's FP32 inputs (velocities, Ji tables, 1/vol), then the
        // FP64 operator -- the same FP64-reference rule as above.
        auto devS = gpu::DeviceComp::make(*met, 1.4, 1e-3, kCsig,
                                          SgsModel::Sigma);
        if (!devS) { check(false, "focal sigma make"); return; }
        const auto sS = linearState(g, [&](int i, int j, int k, double u[3]) {
            const double x[3] = {fr.fx.map((i + 0.5) * g.hx), (j + 0.5) * g.hy,
                                 fr.fz.map((k + 0.5) * g.hz)};
            for (int c = 0; c < 3; ++c)
                u[c] = kG[c][0] * x[0] + kG[c][1] * x[1] + kG[c][2] * x[2];
        });
        const auto hS = compMut(*devS, g, sS);
        auto discrete = [&](int i, int j, int k) {
            const std::size_t c = g.idx(i, j, k);
            const std::size_t p[3] = {g.idx(i + 1, j, k), g.idx(i, j + 1, k),
                                      g.idx(i, j, k + 1)};
            const std::size_t m[3] = {g.idx(i - 1, j, k), g.idx(i, j - 1, k),
                                      g.idx(i, j, k - 1)};
            const std::vector<double>* mom[3] = {&sS.mx, &sS.my, &sS.mz};
            auto vel = [&](std::size_t q, int a) {
                return double(float((*mom[a])[q]) / float(sS.rho[q]));
            };
            double gd[3][3];
            for (int a = 0; a < 3; ++a)
                for (int e2 = 0; e2 < 3; ++e2) {
                    gd[a][e2] = 0;
                    for (int d = 0; d < 3; ++d)
                        gd[a][e2] += 0.5 * (vel(p[d], a) - vel(m[d], a)) * 0.5
                                     * (double(float(met->Ji[d][d * 3 + e2][c]))
                                        + double(float(met->Ji[d][d * 3 + e2][m[d]])));
                }
            const double v = 1.0 / double(float(1.0 / met->vol[c]));
            const double cb = std::cbrt(v);
            return sS.rho[c] * kCsig * kCsig * cb * cb * sigmaDsigma(gd);
        };
        const double eS = maxUlpsInterior(g, hS, discrete);
        double relS = 0, rel64 = 0;
        for (int k = 1; k < g.nz - 1; ++k)
            for (int j = 1; j < g.ny - 1; ++j)
                for (int i = 1; i < g.nx - 1; ++i) {
                    const double dv = hS[g.idx(i, j, k)];
                    relS = std::max(relS, std::abs(dv / discrete(i, j, k) - 1));
                    const double r64 = rhoAt(i, j, k) * kCsig * kCsig
                                       * d2(i, j, k) * sigmaDsigma(kG);
                    rel64 = std::max(rel64, std::abs(dv / r64 - 1));
                }
        std::printf("magnitude comp focal sigma: worst %.2f ulp = rel %.2e "
                    "(discrete gradient, FP64); vs the exact-G operator max "
                    "rel %.2e (the stretched-axis gradient error, not "
                    "pinned)\n", eS, relS, rel64);
        // Not 4 ulp: the device forms the mapped gradient in FP32 (rounded
        // products of Ji), and D_sigma amplifies that rounding where
        // sigma2 ~ sigma3 (measured 37 ulp = 2.7e-6 worst; uniform meshes,
        // whose gradient is exact, stay at 0.1 ulp). Band: ~10x the measured.
        check(relS < 3e-5, "focal comp sigma mu_t == rho (Csigma vol^(1/3))^2 "
                           "D_sigma(discrete grad) to 3e-5 rel");
    }
}

} // namespace

int main(int argc, char** argv)
{
    using namespace dns;
    using gpu::Real;

    dns::test::requireDeviceOrSkip();
    if (argc > 1 && std::strcmp(argv[1], "tally3d") == 0) return tally3d();
    magnitude();

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
