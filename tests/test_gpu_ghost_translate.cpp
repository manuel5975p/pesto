// Translation equivariance of the immersed-body machinery on a periodic box.
//
// A cylinder moved by exactly (3h, 5h) is the same body on the same lattice,
// so everything built from it must be the original shifted by (3, 5) cells:
//   - the ghost classification: every index table (ghost, stencil, deep,
//     interface face, direction, solid) maps cell for cell, in the same order
//     (the body is clear of the periodic seam, so the scan order is kept), and
//     the real tables (weights, extrapolation ratio, face positions) are
//     bit-identical. Bit identity is EXPECTED here, not hoped for: centres,
//     offsets and shifts are dyadic and the body stays inside one binade
//     ([4, 8)) on both axes, so every coordinate difference the classifier
//     forms -- including the finite-difference normal -- rounds identically;
//   - 20 SSP-RK3 steps of the correspondingly shifted state: same state, same
//     force and moment (about the shifted pivot), to 1e-6;
//   - sampleCut (the cut-cell geometry): lambda, ax, ay bit-identical shifted.
#include "dns/comp.hpp"
#include "dns/cutcell.hpp"
#include "dns/geometry.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dns;

namespace {

using dnstest::check;

constexpr double gam = 1.4, rhoInf = 1.0;
const double pInf = 1.0 / (gam * 0.04);
constexpr int n = 128, sx = 3, sy = 5;
constexpr double L = 8.0, h = L / n, R = 0.5;
constexpr double cx0 = 5.5 + 0.3125 * h, cy0 = 5.25 + 0.1875 * h;

gpu::GhostSdf cylinder(double cx, double cy)
{
    return [=](double x, double y, double) { return std::hypot(x - cx, y - cy) - R; };
}

int shiftCell(const Grid& g, int c)
{
    const int i = c % g.nx, j = c / g.nx;
    return int(g.idx(i + sx, j + sy, 0));
}

AxisRemap3 identity(const Grid& g)
{
    return {Focal1D::identity(g.nx * g.hx), Focal1D::identity(g.ny * g.hy),
            Focal1D::identity(g.nz * g.hz)};
}

template <class T>
int countBitDiff(const std::vector<T>& a, const std::vector<T>& b)
{
    if (a.size() != b.size()) return -1;
    int d = 0;
    for (std::size_t q = 0; q < a.size(); ++q)
        if (a[q] != b[q]) ++d;
    return d;
}

bool sameShifted(const Grid& g, const std::vector<int>& a, const std::vector<int>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t q = 0; q < a.size(); ++q)
        if (shiftCell(g, a[q]) != b[q]) return false;
    return true;
}

void tables(const Grid& g)
{
    const auto A = gpu::classifyCompGhost(g, cylinder(cx0, cy0), nullptr,
                                          identity(g), nullptr);
    const auto B = gpu::classifyCompGhost(g, cylinder(cx0 + sx * h, cy0 + sy * h),
                                          nullptr, identity(g), nullptr);
    std::printf("tables: ghosts %d/%d deep %d/%d faces %d/%d solid %zu/%zu\n",
                A.nGhost(), B.nGhost(), A.nDeep(), B.nDeep(), A.nFace(),
                B.nFace(), A.solid.size(), B.solid.size());
    check(A.nGhost() > 0 && A.nDeep() > 0, "classification is non-trivial");
    check(sameShifted(g, A.gcell, B.gcell), "ghost cells map under the shift");
    check(sameShifted(g, A.gnbr, B.gnbr), "ghost stencils map under the shift");
    check(sameShifted(g, A.tnbr, B.tnbr), "thermo stencils map under the shift");
    check(countBitDiff(A.tw, B.tw) == 0, "thermo weights bit-identical");
    check(sameShifted(g, A.deep, B.deep), "deep cells map under the shift");
    check(sameShifted(g, A.fcell, B.fcell), "interface faces map under the shift");
    check(sameShifted(g, A.solid, B.solid), "solid cells map under the shift");
    check(A.fdir == B.fdir, "face directions identical");
    const int dw = countBitDiff(A.gw, B.gw), ds = countBitDiff(A.gscale, B.gscale);
    std::printf("  real tables: %d/%zu weights and %d/%zu ratios differ in bits\n",
                dw, A.gw.size(), ds, A.gscale.size());
    check(dw == 0 && ds == 0, "weights and extrapolation ratios bit-identical");
    bool posOk = A.fpos.size() == B.fpos.size();
    for (std::size_t q = 0; posOk && q < A.fpos.size(); q += 3)
        posOk = gpu::Real(A.fpos[q] + gpu::Real(sx * h)) == B.fpos[q]
                && gpu::Real(A.fpos[q + 1] + gpu::Real(sy * h)) == B.fpos[q + 1]
                && A.fpos[q + 2] == B.fpos[q + 2];
    check(posOk, "face positions shifted by exactly (3h, 5h)");
}

CompState shiftState(const Grid& g, const CompState& a)
{
    auto b = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t s = g.idx(i, j, 0), d = g.idx(i + sx, j + sy, 0);
            b.rho[d] = a.rho[s];
            b.mx[d] = a.mx[s];
            b.my[d] = a.my[s];
            b.mz[d] = a.mz[s];
            b.E[d] = a.E[s];
        }
    return b;
}

struct Run
{
    CompState s;
    gpu::DeviceCompGhost::Force F{};
    bool ok = false;
};

Run evolve(const Grid& g, double cx, double cy, const CompState& ic, double dt)
{
    Run r;
    auto dev = gpu::DeviceComp::make(g, gam, rhoInf * 2 * R / 100.0);
    auto gh = gpu::DeviceCompGhost::make(g, cylinder(cx, cy), gam, rhoInf, pInf);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !gh || !ds) return r;
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });
    (void)ds->upload(ic);
    for (int s = 0; s < 20; ++s) dev->step(*ds, gpu::Real(dt));
    gh->apply(*ds);
    r.F = gh->force(*dev, *ds, {cx, cy, 0.5});
    r.s = CompState::zeros(g);
    (void)ds->download(r.s);
    r.ok = true;
    return r;
}

void dynamics(const Grid& g)
{
    // Tapered freestream about the ORIGINAL body; the shifted run gets the
    // same field moved by (3, 5) cells.
    auto ic = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double d = cylinder(cx0, cy0)((i + 0.5) * h, (j + 0.5) * h, 0.0);
            const double f = std::clamp(d / (3.0 * h), 0.0, 1.0);
            const double u = f * f * (3 - 2 * f), v = 0.2 * f * std::sin((i + 0.5) * h);
            ic.rho[c] = rhoInf;
            ic.mx[c] = rhoInf * u;
            ic.my[c] = rhoInf * v;
            ic.E[c] = pInf / (gam - 1.0) + 0.5 * rhoInf * (u * u + v * v);
        }
    const double dt = 0.3 * h / (1.0 + std::sqrt(gam * pInf / rhoInf));
    const Run a = evolve(g, cx0, cy0, ic, dt);
    const Run b = evolve(g, cx0 + sx * h, cy0 + sy * h, shiftState(g, ic), dt);
    if (!a.ok || !b.ok) { check(false, "dynamics setup"); return; }
    const CompState as = shiftState(g, a.s);
    double err = 0.0, scale = 0.0;
    int bits = 0;
    const std::vector<double>* A[5] = {&as.rho, &as.mx, &as.my, &as.mz, &as.E};
    const std::vector<double>* B[5] = {&b.s.rho, &b.s.mx, &b.s.my, &b.s.mz, &b.s.E};
    for (int q = 0; q < 5; ++q)
        for (std::size_t c = 0; c < g.size(); ++c) {
            err = std::max(err, std::abs((*A[q])[c] - (*B[q])[c]));
            scale = std::max(scale, std::abs((*A[q])[c]));
            bits += (*A[q])[c] != (*B[q])[c];
        }
    const double fm = std::hypot(a.F.fx, a.F.fy);
    const double dF = std::hypot(a.F.fx - b.F.fx, a.F.fy - b.F.fy) / fm;
    const double dQ = std::abs(a.F.qz - b.F.qz) / std::max(std::abs(a.F.qz), 1e-300);
    std::printf("20 steps: state max diff %.2e (scale %.2e, %d values differ in "
                "bits); F (%.9e, %.9e) vs (%.9e, %.9e) rel %.2e; qz %.6e vs "
                "%.6e rel %.2e\n",
                err, scale, bits, a.F.fx, a.F.fy, b.F.fx, b.F.fy, dF, a.F.qz,
                b.F.qz, dQ);
    check(err <= 1e-6 * scale, "shifted run = shifted state (1e-6)");
    check(dF < 1e-6, "force translation-invariant (1e-6)");
    check(dQ < 1e-6, "moment about the shifted pivot invariant (1e-6)");
}

void cutSampling(const Grid& g)
{
    const auto A = sampleCut(g, cylinder(cx0, cy0));
    const auto B = sampleCut(g, cylinder(cx0 + sx * h, cy0 + sy * h));
    if (!A || !B) { check(false, "sampleCut"); return; }
    int d = 0, cut = 0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t s = g.idx(i, j, 0), t = g.idx(i + sx, j + sy, 0);
            d += A->lambda[s] != B->lambda[t];
            d += A->ax[s] != B->ax[t];
            d += A->ay[s] != B->ay[t];
            cut += A->lambda[s] > 0.0 && A->lambda[s] < 1.0;
        }
    std::printf("sampleCut: %d cut cells, %d lambda/ax/ay values differ\n", cut, d);
    check(cut > 0, "sampleCut has cut cells");
    check(d == 0, "sampleCut lambda/ax/ay bit-identical under the shift");
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    tables(g);
    dynamics(g);
    cutSampling(g);
    return dnstest::finish();
}
