// Thin-section seal (comp_ghost.hpp, CompGhostTables::sealed; --thin-seal).
//
// The staircase lets flow through a section thinner than a cell: only a cell
// CENTRE in solid makes a ghost, so two fluid cells on opposite sides of a
// thin sheet share an ordinary interior face (analysis_propperf: the DTMB 4119
// tip at D/h 114 is 57-100% leaky). The seal turns every fluid/fluid face
// whose centre-to-centre segment crosses the body into a wall face of both
// cells, each imaged about its own surface.
//
// `host` (S1, no device): classification only.
//   - Plate t = 0.3 h at 20 deg in a 32^3 box (uniform and focal): the sealed
//     count equals an independent segment-box clipping count; every sealed
//     face is two table faces; ghosts/deep/solid are unchanged by the flag.
//   - Sphere D/h 20, cylinder D/h 20, a 3 h plate: no face sealed.
//   - The tables are identical for 1, 3 and 7 classification threads and for
//     repeated builds; the flag off reproduces the default call.
// `gpu` (S2): two wavy membranes t = 0.5 h (slope up to 10 deg) split a
// periodic 3D box into two gases at rest at different pressures. Sealed: no
// fluid face crosses a membrane, so each chamber's mass is constant to
// round-off over 500 steps and the jump is held at rest (every face reads its
// own gas). Unsealed: the gas leaks through the membranes.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/remap.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"

#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <string_view>
#include <vector>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;

// A box of half sizes b about centre c, rotated by rz about z: exact sdf, and
// the box frame of a world point.
struct Box
{
    double c[3], b[3], rz;

    void frame(const double x[3], double p[3]) const
    {
        const double d[3] = {x[0] - c[0], x[1] - c[1], x[2] - c[2]};
        p[0] = std::cos(rz) * d[0] + std::sin(rz) * d[1];
        p[1] = -std::sin(rz) * d[0] + std::cos(rz) * d[1];
        p[2] = d[2];
    }
    double sdf(double x, double y, double z) const
    {
        const double w[3] = {x, y, z};
        double p[3];
        frame(w, p);
        double out = 0.0, in = -1e300;
        for (int a = 0; a < 3; ++a) {
            const double q = std::abs(p[a]) - b[a];
            out += std::max(q, 0.0) * std::max(q, 0.0);
            in = std::max(in, q);
        }
        return out > 0.0 ? std::sqrt(out) : in;
    }
    // Does the segment x0 -> x1 pass through the box interior over a positive
    // length? Liang-Barsky clipping in the box frame: independent of the sdf.
    bool crosses(const double x0[3], const double x1[3]) const
    {
        double p0[3], p1[3];
        frame(x0, p0);
        frame(x1, p1);
        double lo = 0.0, hi = 1.0;
        for (int a = 0; a < 3; ++a) {
            const double d = p1[a] - p0[a];
            if (std::abs(d) < 1e-300) {
                if (std::abs(p0[a]) >= b[a]) return false;
                continue;
            }
            double s0 = (-b[a] - p0[a]) / d, s1 = (b[a] - p0[a]) / d;
            if (s0 > s1) std::swap(s0, s1);
            lo = std::max(lo, s0);
            hi = std::min(hi, s1);
        }
        return hi > lo;
    }
};

Grid box(int n, int nz)
{
    Grid g;
    g.nx = g.ny = n;
    g.nz = nz;
    g.hx = g.hy = g.hz = 1.0 / n;
    return g;
}

AxisRemap3 identity(const Grid& g)
{
    const double z0[3] = {0.0, 0.0, 0.0};
    return focalRemap(g, false, z0, z0, z0, 1.0);
}

// Every vector of the tables, compared exactly.
bool sameTables(const gpu::CompGhostTables& a, const gpu::CompGhostTables& b)
{
    return a.gcell == b.gcell && a.gnbr == b.gnbr && a.gw == b.gw
           && a.tnbr == b.tnbr && a.tw == b.tw && a.gscale == b.gscale
           && a.gdw == b.gdw && a.gfnb == b.gfnb && a.uwall == b.uwall
           && a.aspan == b.aspan && a.deep == b.deep && a.fcell == b.fcell
           && a.fdir == b.fdir && a.fpos == b.fpos && a.fuwall == b.fuwall
           && a.solid == b.solid && a.fallback == b.fallback
           && a.sideFace == b.sideFace && a.sideNbr == b.sideNbr
           && a.sideW == b.sideW && a.sideTnbr == b.sideTnbr
           && a.sideTw == b.sideTw && a.sideScale == b.sideScale
           && a.sideUwall == b.sideUwall && a.sideAspan == b.sideAspan
           && a.sideCell == b.sideCell && a.sideStart == b.sideStart
           && a.sideSkipped == b.sideSkipped && a.sealed == b.sealed;
}

// The analytic sealed count: fluid/fluid face pairs (no wrap) whose centre
// segment clips the box, centres from the remap.
int analyticCount(const Grid& g, const AxisRemap3& R, const Box& bx)
{
    auto ctr = [&](int i, int j, int k, double x[3]) {
        x[0] = R.fx.map((i + 0.5) * g.hx);
        x[1] = R.fy.map((j + 0.5) * g.hy);
        x[2] = R.fz.map((k + 0.5) * g.hz);
    };
    const int n[3] = {g.nx, g.ny, g.nz};
    int count = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                double xa[3];
                ctr(i, j, k, xa);
                if (bx.sdf(xa[0], xa[1], xa[2]) < 0.0) continue;
                const int ijk[3] = {i, j, k};
                for (int a = 0; a < (g.nz == 1 ? 2 : 3); ++a) {
                    if (ijk[a] + 1 >= n[a]) continue;
                    int nb[3] = {i, j, k};
                    ++nb[a];
                    double xb[3];
                    ctr(nb[0], nb[1], nb[2], xb);
                    if (bx.sdf(xb[0], xb[1], xb[2]) < 0.0) continue;
                    count += bx.crosses(xa, xb) ? 1 : 0;
                }
            }
    return count;
}

void plate(std::string_view name, bool focal)
{
    const Grid g = box(32, 32);
    const double h = 1.0 / 32;
    const double lo[3] = {0.3, 0.3, 0.3}, hi[3] = {0.7, 0.7, 0.7},
                 w[3] = {0.1, 0.1, 0.1};
    const AxisRemap3 R = focal ? focalRemap(g, true, lo, hi, w, 2.0)
                               : identity(g);
    // h_fine on the focal plateau: thickness 0.3 of the FINE cell there.
    const double hf = focal ? R.fx.c * R.fx.r * h : h;
    const Box bx{{0.5 + 0.17 * h, 0.5 + 0.29 * h, 0.5 + 0.11 * h},
                 {0.22, 0.15 * hf, 0.2},
                 20.0 * pi / 180.0};
    const gpu::GhostSdf sdf = [bx](double x, double y, double z) {
        return bx.sdf(x, y, z);
    };
    const auto off = gpu::classifyCompGhost(g, sdf, nullptr, R, nullptr);
    const auto deflt = gpu::classifyCompGhost(g, sdf, nullptr, R, nullptr, 1,
                                              false);
    const auto on1 = gpu::classifyCompGhost(g, sdf, nullptr, R, nullptr, 1,
                                            true);
    const auto on3 = gpu::classifyCompGhost(g, sdf, nullptr, R, nullptr, 3,
                                            true);
    const auto on7 = gpu::classifyCompGhost(g, sdf, nullptr, R, nullptr, 7,
                                            true);
    const auto again = gpu::classifyCompGhost(g, sdf, nullptr, R, nullptr, 3,
                                              true);
    const int want = analyticCount(g, R, bx);
    const std::string nm(name);
    std::printf("%s: sealed %d (analytic %d), faces %d -> %d, ghosts %d, deep "
                "%d, side faces %d -> %d, skipped %d -> %d\n",
                nm.c_str(), on1.sealed, want, off.nFace(), on1.nFace(),
                on1.nGhost(), on1.nDeep(), off.nSide(), on1.nSide(),
                off.sideSkipped, on1.sideSkipped);
    check(want > 100, nm + ": the plate leaks (premise: > 100 faces)");
    check(on1.sealed == want, nm + ": sealed count = analytic count");
    check(off.sealed == 0, nm + ": flag off seals nothing");
    check(sameTables(off, deflt), nm + ": flag off = the default call");
    check(on1.nFace() == off.nFace() + 2 * on1.sealed,
          nm + ": each sealed face is two table faces");
    check(on1.gcell == off.gcell && on1.deep == off.deep
              && on1.solid == off.solid && on1.gfnb == off.gfnb,
          nm + ": ghosts, deep, solid and gfnb unchanged");
    check(on1.nSide() - off.nSide() + (on1.sideSkipped - off.sideSkipped)
              == 2 * on1.sealed,
          nm + ": every sealed side imaged or counted skipped");
    check(on1.sideSkipped - off.sideSkipped <= on1.sealed / 10,
          nm + ": at most 10% of sealed sides skipped");
    check(sameTables(on1, on3) && sameTables(on1, on7)
              && sameTables(on3, again),
          nm + ": identical across thread counts and builds");
    // Contiguous ascending fluid-cell runs (the CSR load() builds).
    bool asc = true;
    for (int f = 1; f < on1.nFace(); ++f)
        asc = asc && on1.fcell[std::size_t(f)] >= on1.fcell[std::size_t(f - 1)];
    check(asc, nm + ": fcell ascending (contiguous per cell)");
}

void noLeaks()
{
    const double h = 1.0 / 64;
    struct Case
    {
        std::string name;
        int nz;
        gpu::GhostSdf sdf;
    };
    const double c[3] = {0.5 + 0.13 * h, 0.5 + 0.31 * h, 0.5 + 0.07 * h};
    const double r = 10.0 * h; // D/h 20
    const Case cases[] = {
        {"sphere D/h 20", 64,
         [=](double x, double y, double z) {
             return std::hypot(x - c[0], y - c[1], z - c[2]) - r;
         }},
        {"cylinder D/h 20 (3D)", 16,
         [=](double x, double y, double) {
             return std::hypot(x - c[0], y - c[1]) - r;
         }},
        {"cylinder D/h 20 (2D)", 1,
         [=](double x, double y, double) {
             return std::hypot(x - c[0], y - c[1]) - r;
         }},
        {"plate 3 h at 20 deg", 64,
         [=](double x, double y, double z) {
             const Box b{{c[0], c[1], c[2]}, {0.2, 1.5 * h, 0.2},
                         20.0 * pi / 180.0};
             return b.sdf(x, y, z);
         }},
    };
    for (const Case& k : cases) {
        Grid g = box(64, k.nz);
        g.hz = 1.0 / 64;
        const auto t = gpu::classifyCompGhost(g, k.sdf, nullptr, identity(g),
                                              nullptr, 4, true);
        std::printf("no leaks: %s: sealed %d, ghosts %d\n", k.name.c_str(),
                    t.sealed, t.nGhost());
        check(t.nGhost() > 0 && t.sealed == 0, k.name + ": nothing sealed");
    }
}

// ---- S2 -------------------------------------------------------------------

constexpr double gamma_ = 1.4, rhoInf = 1.0, mu = 0.01;
const double pInf = 1.0 / (gamma_ * 0.04); // M 0.2 against U = 1
const double pHi = 1.05 * pInf, pLo = 0.95 * pInf;

struct Leak
{
    double drift[2]; // relative chamber mass change after the run
    double umax;     // max |u| / (dp / rho c), fluid cells
    int sealed;
};

Leak membranes(bool seal, int steps)
{
    const int n = 32, nz = 8;
    Grid g = box(n, nz);
    const double h = 1.0 / n;
    // Membrane m: y = yc[m] + A sin(2 pi x + m), slope up to tan 10 deg,
    // thickness 0.5 h; sdf scaled by the slope (a lower bound, so the
    // sphere trace never steps across).
    const double yc[2] = {0.27 + 0.31 * h, 0.73 + 0.62 * h};
    const double A = std::tan(10.0 * pi / 180.0) / (2.0 * pi);
    const double t = 0.5 * h;
    auto centre = [&](int m, double x) {
        return yc[m] + A * std::sin(2 * pi * x + m);
    };
    const gpu::GhostSdf sdf = [=](double x, double y, double) {
        double best = 1e300;
        for (int m = 0; m < 2; ++m) {
            const double s = 2 * pi * A * std::cos(2 * pi * x + m);
            const double d = (std::abs(y - (yc[m] + A * std::sin(2 * pi * x + m)))
                              - 0.5 * t) / std::sqrt(1 + s * s);
            best = std::min(best, d);
        }
        return best;
    };
    const AxisRemap3 R = identity(g);
    const auto tb = gpu::classifyCompGhost(g, sdf, nullptr, R, nullptr, 2, seal);
    auto dev = gpu::DeviceComp::make(g, gamma_, mu);
    gpu::CompGhostCapacity cap;
    cap.cover(tb);
    auto gh = gpu::DeviceCompGhost::fromTables(g, tb, cap, gamma_, rhoInf, pInf,
                                               false);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !gh || !ds || !red) {
        check(false, "membranes: setup");
        return {};
    }
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });
    dev->setRhsHook([&](const gpu::DeviceCompState& s,
                        gpu::DeviceCompState& out,
                        const gpu::StageCombine& cmb) {
        gh->applyRhs(*dev, s, out, cmb);
    });
    // Chamber of a fluid cell: 1 between the membranes, 0 outside.
    std::vector<int> chamber(g.size(), -1);
    std::vector<char> solid(g.size(), 0);
    for (int c : tb.solid) solid[std::size_t(c)] = 1;
    auto h0 = CompState::zeros(g);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double x = (i + 0.5) * h, y = (j + 0.5) * h;
                const bool in = y > centre(0, x) && y < centre(1, x);
                if (!solid[c]) chamber[c] = in ? 1 : 0;
                const double p = in ? pHi : pLo;
                h0.rho[c] = in ? rhoInf : 1.2 * rhoInf;
                h0.E[c] = p / (gamma_ - 1.0);
            }
    (void)ds->upload(h0);
    gh->apply(*ds);
    auto mass = [&](const CompState& s, double m[2]) {
        m[0] = m[1] = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c)
            if (chamber[c] >= 0) m[chamber[c]] += s.rho[c];
    };
    double m0[2], m1[2];
    (void)ds->download(h0);
    mass(h0, m0);
    for (int s = 0; s < steps; ++s)
        dev->step(*ds, gpu::Real(dev->stableDt(*ds, *red)));
    gh->apply(*ds);
    auto h1 = CompState::zeros(g);
    (void)ds->download(h1);
    mass(h1, m1);
    double umax = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c) {
        if (chamber[c] < 0) continue;
        const double m2 = h1.mx[c] * h1.mx[c] + h1.my[c] * h1.my[c]
                          + h1.mz[c] * h1.mz[c];
        umax = std::max(umax, std::sqrt(m2) / h1.rho[c]);
    }
    const double cs = std::sqrt(gamma_ * pInf / rhoInf);
    Leak L;
    L.drift[0] = (m1[0] - m0[0]) / m0[0];
    L.drift[1] = (m1[1] - m0[1]) / m0[1];
    L.umax = umax / ((pHi - pLo) / (rhoInf * cs));
    L.sealed = tb.sealed;
    std::printf("membranes %s: %d sealed, %d ghosts, %d side faces (%d "
                "skipped); %d steps: chamber mass drift %.3e / %.3e, max|u| / "
                "(dp / rho c) %.3e\n",
                seal ? "sealed" : "unsealed", tb.sealed, tb.nGhost(),
                tb.nSide(), tb.sideSkipped, steps, L.drift[0], L.drift[1],
                L.umax);
    return L;
}

void gpuMembranes()
{
    const Leak open = membranes(false, 500);
    const Leak shut = membranes(true, 500);
    check(shut.sealed > 0, "membranes: t = 0.5 h leaks without the seal "
                           "(premise: sealed faces exist)");
    const double dOpen = std::max(std::abs(open.drift[0]),
                                  std::abs(open.drift[1]));
    const double dShut = std::max(std::abs(shut.drift[0]),
                                  std::abs(shut.drift[1]));
    check(std::isfinite(dShut) && dShut < 1e-5,
          "membranes: sealed chamber mass constant (1e-5)");
    check(dOpen > 1e-3, "membranes: unsealed chambers exchange mass (1e-3)");
    check(std::isfinite(shut.umax) && shut.umax < 1e-3,
          "membranes: sealed, the jump is held at rest (1e-3 dp / rho c)");
    check(open.umax > 1e-2, "membranes: unsealed, the jump drives a leak");
}

} // namespace

int main(int argc, char** argv)
{
    const std::string_view mode = argc > 1 ? argv[1] : "all";
    if (mode == "host" || mode == "all") {
        plate("plate 0.3 h at 20 deg", false);
        plate("plate 0.3 h_fine at 20 deg, focal", true);
        noLeaks();
    }
    if (mode == "gpu" || mode == "all") {
        test::requireDeviceOrSkip();
        gpuMembranes();
    }
    return dnstest::finish();
}
