// LES wall-stress closure (masterplan_verify W1). Under LES the SGS mu_t takes
// the RANS wall image (DeviceComp::setMutHook) and every surface tally carries
// the SGS stress the face flux carries:
//   (a) O-grid 64 x 32 cylinder, smag 0.17, impulsive freestream (a wall
//       shear state): after a step, and after updateSgsNut, the wall face
//       0.5 (mu_t[j=0] + mu_t[j=1]) is exactly 0.0f on every face, and the far
//       ghost row is an even copy of row ny-2;
//   (b) momentum-budget closure in a fully periodic box with the body: one
//       residual-mode stage (ca = cb = 0, cr = 1), sum over the FLUID cells
//       of rhs(m) * vol in FP64 equals -(tally) per component. Immersed
//       Cartesian 64^2 cylinder, focal 64^2 cylinder, immersed 32^3 sphere.
//       The csLes = 0 run of the same rig is the FP32 rounding floor r0; the
//       LES run must close to max(1e-5, 3 r0) relative. Before W1 the LES
//       tally omitted the SGS stress and missed by its share (~10^-1; the
//       measured numbers are in plans/verify/impl_W1.md).
// Seconds.
#include "dns/bodies.hpp"
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/remap.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/ogrid_bc.hpp"

#include "check.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using std::numbers::pi;
using gpu::Real;

namespace {

constexpr double kGam = 1.4, kMach = 0.2, kRhoInf = 1.0, kUInf = 1.0;
const double kPInf = kRhoInf * kUInf * kUInf / (kGam * kMach * kMach);
constexpr double kCs = 0.17;

// Impulsive uniform freestream along x (plus a weak smooth swirl, so the
// tally has a nonzero lift and spanwise component to close as well).
CompState freestream(const Grid& g)
{
    auto h = CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double x = (i + 0.5) / g.nx, y = (j + 0.5) / g.ny,
                             z = (k + 0.5) / g.nz;
                const double u = kUInf + 0.1 * std::sin(2 * pi * y);
                const double v = 0.1 * std::sin(2 * pi * x);
                const double w = g.nz > 1 ? 0.1 * std::sin(2 * pi * (x + y))
                                                * std::cos(2 * pi * z)
                                          : 0.0;
                h.rho[c] = kRhoInf;
                h.mx[c] = kRhoInf * u;
                h.my[c] = kRhoInf * v;
                h.mz[c] = kRhoInf * w;
                h.E[c] = kPInf / (kGam - 1.0) +
                         0.5 * kRhoInf * (u * u + v * v + w * w);
            }
    return h;
}

// --- (a) ----------------------------------------------------------------------

void ogridWallFace()
{
    constexpr double kD = 0.785398163397448, kRFar = 20 * kD;
    const Grid g = ogridGrid(64, 32, 1, 0.5 * kD, kRFar, 1.0);
    const Sdf circle = [](double x, double y, double) {
        return std::hypot(x, y) - 0.5 * kD;
    };
    auto og = buildOGrid(g, circle, {.rFar = kRFar, .beta = 3.0, .span = 1.0});
    if (!og) { check(false, "(a) O-grid build"); return; }
    auto m = CurvMetrics::build(og->mesh);
    if (!m) { check(false, "(a) metrics"); return; }
    const double mu = kRhoInf * kUInf * kD / 100.0;
    auto dev = gpu::DeviceComp::make(*m, kGam, mu, kCs);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto bc = gpu::DeviceOGridBc::make(
        g, {.gamma = kGam, .rhoInf = kRhoInf, .uInf = kUInf, .pInf = kPInf});
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !bc || !red) { check(false, "(a) setup"); return; }
    dev->setMutHook(bc->mutHook());
    gpu::DeviceOGridBc* bp = &*bc;
    dev->setStageHook([bp](gpu::DeviceCompState& s) { bp->apply(s); });
    auto h = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        h.rho[c] = kRhoInf;
        h.mx[c] = kRhoInf * kUInf;
        h.E[c] = kPInf / (kGam - 1.0) + 0.5 * kRhoInf * kUInf * kUInf;
    }
    (void)ds->upload(h);
    bc->apply(*ds);
    auto rowCheck = [&](const std::string& when) {
        std::vector<float> nut(g.size());
        (void)dev->mut().download(nut.data());
        int bad = 0, badFar = 0;
        double wallMax = 0, row1Max = 0;
        for (int i = 0; i < g.nx; ++i) {
            const float n0 = nut[g.idx(i, 0, 0)], n1 = nut[g.idx(i, 1, 0)];
            if (0.5f * (n0 + n1) != 0.0f) ++bad;
            wallMax = std::max(wallMax, std::abs(0.5 * (double(n0) + n1)));
            row1Max = std::max(row1Max, double(n1));
            if (nut[g.idx(i, g.ny - 1, 0)] != nut[g.idx(i, g.ny - 2, 0)])
                ++badFar;
        }
        std::printf("(a) %s: max wall-face mu_t %.3e (first-cell max %.3e, "
                    "mu %.3e), %d nonzero faces, %d far rows not even\n",
                    when.c_str(), wallMax, row1Max, mu, bad, badFar);
        check(row1Max > 0.1 * mu, "(a) " + when +
                                      ": the first fluid row carries SGS mu_t "
                                      "(the check is not vacuous)");
        check(bad == 0, "(a) " + when + ": wall-face mu_t == 0.0f exactly");
        check(badFar == 0, "(a) " + when + ": far ghost row is an even copy");
    };
    dev->step(*ds, Real(1e-4));
    rowCheck("after a step");
    bc->apply(*ds);
    dev->updateSgsNut(*ds);
    rowCheck("after updateSgsNut");
}

// --- (b) ----------------------------------------------------------------------

struct Closure
{
    double rel = 0;   // |sum R vol + F| / |F|, worst component
    double F[3] = {0, 0, 0};
};

// One residual-mode stage on the ghost-filled state, summed over the fluid.
// vol: per-cell volume (FP64).
Closure closure(gpu::DeviceComp& dev, gpu::DeviceCompGhost& gh, const Grid& g,
                const std::vector<double>& vol)
{
    auto ds = gpu::DeviceCompState::make(g.size());
    auto R = gpu::DeviceCompState::make(g.size());
    Closure out;
    if (!ds || !R) { check(false, "closure alloc"); return out; }
    (void)ds->upload(freestream(g));
    gh.apply(*ds);
    dev.residualInto(*ds, *R);
    const auto F = gh.force(dev, *ds);
    auto hr = CompState::zeros(g);
    (void)R->download(hr);
    std::vector<char> solid(g.size(), 0);
    for (int c : gh.solidCells()) solid[std::size_t(c)] = 1;
    double S[3] = {0, 0, 0};
    for (std::size_t c = 0; c < g.size(); ++c) {
        if (solid[c]) continue;
        S[0] += hr.mx[c] * vol[c];
        S[1] += hr.my[c] * vol[c];
        S[2] += hr.mz[c] * vol[c];
    }
    out.F[0] = F.fx;
    out.F[1] = F.fy;
    out.F[2] = F.fz;
    const double Fn = std::sqrt(F.fx * F.fx + F.fy * F.fy + F.fz * F.fz);
    for (int a = 0; a < 3; ++a)
        out.rel = std::max(out.rel, std::abs(S[a] + out.F[a]) / Fn);
    return out;
}

struct Rig
{
    std::string name;
    Grid g;
    gpu::GhostSdf sdf;
    double bbox[6];
    std::optional<AxisRemap3> remap; // engaged = focal
};

// r0 (csLes = 0) and the LES closure of one rig; the LES one must sit at
// the rounding floor.
void budget(const Rig& r)
{
    const double mu = 2.5e-3;
    std::optional<CurvMetrics> m;
    std::vector<double> vol(r.g.size(), r.g.hx * r.g.hy * r.g.hz);
    if (r.remap) {
        auto mm = CurvMetrics::build(CurvMesh::fromRemap(r.g, *r.remap));
        if (!mm) { check(false, r.name + ": metrics"); return; }
        m = std::move(*mm);
        // The device divides by the FP32 1/vol it uploaded.
        for (std::size_t c = 0; c < r.g.size(); ++c)
            vol[c] = 1.0 / double(Real(1.0 / m->vol[c]));
    }
    auto mkGhost = [&] {
        return r.remap ? gpu::DeviceCompGhost::make(r.g, r.sdf, kGam, kRhoInf,
                                                    kPInf, r.bbox, *r.remap)
                       : gpu::DeviceCompGhost::make(r.g, r.sdf, kGam, kRhoInf,
                                                    kPInf, r.bbox);
    };
    double rel[2] = {0, 0};
    for (int les = 0; les < 2; ++les) {
        const double cs = les ? kCs : 0.0;
        auto dev = m ? gpu::DeviceComp::make(*m, kGam, mu, cs)
                     : gpu::DeviceComp::make(r.g, kGam, mu, cs);
        auto gh = mkGhost();
        if (!dev || !gh) { check(false, r.name + ": setup"); return; }
        gpu::DeviceCompGhost* gp = &*gh;
        gpu::DeviceComp* dp = &*dev;
        // The wall faces' rhs (impermeable faces): the tally reads them.
        dev->setRhsHook([gp, dp](const gpu::DeviceCompState& s,
                                 gpu::DeviceCompState& out,
                                 const gpu::StageCombine& cmb) {
            gp->applyRhs(*dp, s, out, cmb);
        });
        if (les)
            dev->setMutHook([gp](Real* nut, Real* f1) { gp->mirrorMut(nut, f1); });
        const Closure c = closure(*dev, *gh, r.g, vol);
        rel[les] = c.rel;
        std::printf("(b) %-22s %s: F = (%+.6e, %+.6e, %+.6e)  closure %.3e\n",
                    r.name.c_str(), les ? "LES smag" : "csLes 0 ", c.F[0],
                    c.F[1], c.F[2], c.rel);
    }
    const double band = std::max(1e-5, 3.0 * rel[0]);
    check(rel[0] < 1e-5, r.name + ": DNS closure floor r0 < 1e-5");
    check(rel[1] <= band, r.name + ": LES closure <= max(1e-5, 3 r0)");
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    ogridWallFace();

    // Periodic unit box, body off-centre so it cuts the lattice generically.
    {
        const Grid g = Grid::make(64, 64, 1, 1.0, 1.0, 1.0);
        const double cx = 0.5 + 0.013, cy = 0.5 + 0.021, R = 0.125;
        auto sdf = [=](double x, double y, double) {
            return std::hypot(x - cx, y - cy) - R;
        };
        budget({"immersed cylinder 64^2", g, sdf,
                {cx - R, cy - R, 0.0, cx + R, cy + R, 1.0}, std::nullopt});
        const double lo[3] = {cx - 2 * R, cy - 1.6 * R, 0.0},
                     hi[3] = {cx + 2 * R, cy + 1.6 * R, 1.0},
                     w[3] = {2 * R, 1.6 * R, 0.0};
        const AxisRemap3 fr = focalRemap(g, true, lo, hi, w, 3.0);
        check(fr.fx.active && fr.fy.active, "focal remap active");
        budget({"focal cylinder 64^2", g, sdf,
                {cx - R, cy - R, 0.0, cx + R, cy + R, 1.0}, fr});
    }
    {
        const Grid g = Grid::make(32, 32, 32, 1.0, 1.0, 1.0);
        const double cx = 0.5 + 0.011, cy = 0.5 - 0.017, cz = 0.5 + 0.007,
                     R = 0.19;
        auto sdf = [=](double x, double y, double z) {
            return std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) +
                             (z - cz) * (z - cz)) - R;
        };
        budget({"immersed sphere 32^3", g, sdf,
                {cx - R, cy - R, cz - R, cx + R, cy + R, cz + R}, std::nullopt});
    }
    return dnstest::finish();
}
