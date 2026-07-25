// PhysicalRaster: the display resampling that undoes a curvilinear mesh map
// so frames are pictures of PHYSICAL space, not of the index space.
//
// The load-bearing property is the round trip: a field that is a known smooth
// function of the PHYSICAL position, stored at the reference (index-space)
// cells the solver actually uses, must come back as that same function of
// position on the uniform render raster. Everything else here (identity is a
// bit-exact copy, the raster pitch is the finest spacing, nz == 1 stays
// planar) pins the degenerate cases.
#include "dns/framemap.hpp"
#include "dns/remap.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

using namespace dns;

namespace {

using dnstest::check;



// A smooth periodic function of physical position; the round-trip probe.
double probe(double x, double y, double z, double L)
{
    using std::numbers::pi;
    return std::sin(2 * pi * x / L) + 0.5 * std::cos(4 * pi * y / L) +
           0.25 * std::sin(2 * pi * z / L);
}

AxisRemapView viewOf(const Focal1D& f, double L, double h, int n)
{
    AxisRemapView v;
    v.L = L;
    if (n == 1) return v;
    double hf = L;
    for (int i = 0; i < n; ++i)
        hf = std::min(hf, f.map((i + 1) * h) - f.map(i * h));
    v.hFine = hf;
    v.inverse = [f](double x) { return f.inverse(x); };
    return v;
}

void identityIsACopy()
{
    const auto g = Grid::make(16, 12, 8, 4.0, 4.0, 4.0);
    AxisRemapView v[3];
    for (int a = 0; a < 3; ++a) v[a].L = 4.0; // null inverse == identity
    const auto r = PhysicalRaster::make(g, v);
    check(!r.active(), "identity map: raster inactive");
    check(r.renderGrid().nx == g.nx && r.renderGrid().ny == g.ny &&
              r.renderGrid().nz == g.nz,
          "identity map: render grid IS the solver grid");
    std::vector<float> src(g.size()), dst(g.size());
    for (std::size_t c = 0; c < g.size(); ++c) src[c] = float(c) * 0.5f - 3.0f;
    r.apply(src.data(), dst.data());
    bool exact = true;
    for (std::size_t c = 0; c < g.size(); ++c)
        if (src[c] != dst[c]) exact = false;
    check(exact, "identity map: apply() is bit-exact");
}

// 3D focal mesh clustered on all three axes: the round trip.
void focalRoundTrip()
{
    const double L = 4.0;
    const auto g = Grid::make(64, 64, 64, L, L, L);
    const Focal1D fx = Focal1D::make(L, 1.2, 2.8, 0.8, 3.0);
    const Focal1D fy = Focal1D::make(L, 1.4, 2.6, 0.6, 3.0);
    const Focal1D fz = Focal1D::make(L, 1.4, 2.6, 0.6, 3.0);
    const AxisRemapView v[3] = {viewOf(fx, L, g.hx, g.nx),
                                viewOf(fy, L, g.hy, g.ny),
                                viewOf(fz, L, g.hz, g.nz)};
    const auto r = PhysicalRaster::make(g, v);
    check(r.active(), "focal map: raster active");

    // The raster pitch is the finest spacing, so clustering costs no
    // displayed detail: each axis grows by ~the focal ratio.
    const Grid& rg = r.renderGrid();
    std::printf("render grid %d x %d x %d (solver %d), pitch %.4e vs fine %.4e\n",
                rg.nx, rg.ny, rg.nz, g.nx, rg.hx, v[0].hFine);
    check(std::abs(rg.hx - v[0].hFine) < 0.02 * v[0].hFine &&
              std::abs(rg.hy - v[1].hFine) < 0.02 * v[1].hFine,
          "focal map: raster pitch is the finest spacing");
    check(rg.nx > g.nx && rg.ny > g.ny && rg.nz > g.nz,
          "focal map: render grid is finer than the solver grid");

    // Store probe(physical position) at the reference cells, resample, and
    // compare against probe evaluated on the uniform render raster.
    std::vector<float> src(g.size());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                src[g.idx(i, j, k)] =
                    float(probe(fx.map((i + 0.5) * g.hx),
                                fy.map((j + 0.5) * g.hy),
                                fz.map((k + 0.5) * g.hz), L));
    std::vector<float> dst(rg.size());
    r.apply(src.data(), dst.data());

    double worst = 0.0;
    for (int k = 0; k < rg.nz; ++k)
        for (int j = 0; j < rg.ny; ++j)
            for (int i = 0; i < rg.nx; ++i) {
                const double want = probe((i + 0.5) * rg.hx, (j + 0.5) * rg.hy,
                                          (k + 0.5) * rg.hz, L);
                worst = std::max(worst,
                                 std::abs(double(dst[rg.idx(i, j, k)]) - want));
            }
    std::printf("focal round-trip worst error %.3e\n", worst);
    // Floor = linear interpolation of the probe across the COARSE cells
    // (h_coarse = ratio * h_fine): h^2/8 * max|f''| ~ 0.104^2/8 * 7.4 ~ 1e-2.
    // The bound is that estimate, so a real indexing or wrap bug (which puts
    // the error at O(1), the amplitude of the probe) cannot hide under it.
    check(worst < 1.5e-2, "focal map: physical round trip is faithful");

    // A body edge must land at the right PLACE, which is the whole point.
    // A step at physical x = 2.0 must resample to a step at 2.0.
    for (std::size_t c = 0; c < src.size(); ++c) src[c] = 0.0f;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                if (fx.map((i + 0.5) * g.hx) > 2.0)
                    src[g.idx(i, j, k)] = 1.0f;
    r.apply(src.data(), dst.data());
    int edge = 0;
    for (int i = 0; i < rg.nx; ++i)
        if (dst[rg.idx(i, rg.ny / 2, rg.nz / 2)] >= 0.5f) { edge = i; break; }
    const double xEdge = (edge + 0.5) * rg.hx;
    std::printf("step at x = 2.0 resamples to x = %.4f (pitch %.4f)\n", xEdge,
                rg.hx);
    check(std::abs(xEdge - 2.0) < 2.0 * rg.hx,
          "focal map: a physical edge lands at its physical position");
}

// nz == 1 is the 2D restriction, here too: z is never resampled.
void planarStaysPlanar()
{
    const double L = 4.0;
    const auto g = Grid::make(64, 64, 1, L, L, L / 64);
    const Focal1D fx = Focal1D::make(L, 1.2, 2.8, 0.8, 4.0);
    AxisRemapView v[3];
    for (int a = 0; a < 3; ++a) v[a].L = L;
    v[0] = viewOf(fx, L, g.hx, g.nx);
    const auto r = PhysicalRaster::make(g, v);
    check(r.active() && r.renderGrid().nz == 1,
          "nz == 1: z axis is left alone");
    check(r.renderGrid().ny == g.ny, "an unmapped axis keeps its resolution");
    std::vector<float> dst(r.renderGrid().size());
    std::vector<float> src(g.size(), 2.5f);
    r.apply(src.data(), dst.data());
    bool flat = true;
    for (float q : dst)
        if (std::abs(q - 2.5f) > 1e-6f) flat = false;
    check(flat, "a constant field resamples to the same constant");
}

// The wall-clustered stretch map is the other curvilinear mesh; its inverse
// is analytic, so the round trip must be tight there too.
void stretchRoundTrip()
{
    const double L = 4.0;
    const auto g = Grid::make(32, 64, 1, L, L, L / 64);
    const TanhStretchY R{L, 2.0};
    double hf = L;
    for (int j = 0; j < g.ny; ++j)
        hf = std::min(hf, R.mapY((j + 1) * g.hy) - R.mapY(j * g.hy));
    AxisRemapView v[3];
    for (int a = 0; a < 3; ++a) v[a].L = L;
    v[1].hFine = hf;
    v[1].inverse = [R](double y) { return R.invMapY(y); };
    const auto r = PhysicalRaster::make(g, v);
    check(r.active(), "stretch map: raster active");

    double worstInv = 0.0;
    for (int j = 0; j < g.ny; ++j) {
        const double Y = (j + 0.5) * g.hy;
        worstInv = std::max(worstInv, std::abs(R.invMapY(R.mapY(Y)) - Y));
    }
    std::printf("TanhStretchY invMapY round trip worst %.3e\n", worstInv);
    check(worstInv < 1e-12, "TanhStretchY::invMapY inverts mapY");

    const Grid& rg = r.renderGrid();
    std::vector<float> src(g.size());
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            src[g.idx(i, j, 0)] =
                float(probe((i + 0.5) * g.hx, R.mapY((j + 0.5) * g.hy), 0.0, L));
    std::vector<float> dst(rg.size());
    r.apply(src.data(), dst.data());
    double worst = 0.0;
    for (int j = 0; j < rg.ny; ++j)
        for (int i = 0; i < rg.nx; ++i) {
            const double want =
                probe((i + 0.5) * rg.hx, (j + 0.5) * rg.hy, 0.0, L);
            worst = std::max(worst, std::abs(double(dst[rg.idx(i, j, 0)]) - want));
        }
    std::printf("stretch round-trip worst error %.3e\n", worst);
    check(worst < 2e-2, "stretch map: physical round trip is faithful");
}

} // namespace

int main()
{
    identityIsACopy();
    focalRoundTrip();
    planarStaysPlanar();
    stretchRoundTrip();
    std::printf("framemap: %s\n", dnstest::gFailures == 0 ? "ok" : "FAILED");
    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
