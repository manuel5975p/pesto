#include "dns/framemap.hpp"

#include <algorithm>
#include <cmath>

namespace dns {

namespace {

int axisCount(const Grid& g, int a)
{
    return a == 0 ? g.nx : a == 1 ? g.ny : g.nz;
}

double axisSpacing(const Grid& g, int a)
{
    return a == 0 ? g.hx : a == 1 ? g.hy : g.hz;
}

} // namespace

PhysicalRaster PhysicalRaster::identity(const Grid& g)
{
    PhysicalRaster r;
    r.src_ = g;
    r.rg_ = g;
    return r;
}

PhysicalRaster PhysicalRaster::make(const Grid& g, const AxisRemapView v[3],
                                    int growthCap, std::size_t cellCap)
{
    PhysicalRaster r = identity(g);
    int n[3];
    for (int a = 0; a < 3; ++a) {
        const int ns = axisCount(g, a);
        const double hs = axisSpacing(g, a);
        n[a] = ns;
        // A degenerate axis is the nz == 1 restriction: never resampled.
        if (ns == 1 || !v[a].inverse || !(v[a].hFine > 0) || !(v[a].hFine < hs))
            continue;
        n[a] = std::clamp(int(std::lround(v[a].L / v[a].hFine)), ns,
                          growthCap * ns);
        r.active_ = true;
    }
    if (!r.active_) return r;
    // Absolute size guard: the ideal pitch is the finest spacing on EVERY
    // axis, so a strongly clustered 3D mesh asks for ratio^3 times the cells.
    // Past the cap, shrink all grown axes by one common factor -- the plateau
    // is then displayed below its native resolution, but no run dies of a
    // display buffer. Frames stay geometrically true either way.
    {
        double cells = double(n[0]) * n[1] * n[2];
        if (cells > double(cellCap)) {
            const double s = std::cbrt(double(cellCap) / cells);
            for (int a = 0; a < 3; ++a) {
                const int ns = axisCount(g, a);
                if (n[a] > ns) n[a] = std::max(ns, int(n[a] * s));
            }
        }
    }

    r.rg_ = Grid::make(n[0], n[1], n[2], double(g.nx) * g.hx,
                       double(g.ny) * g.hy, double(g.nz) * g.hz);
    for (int a = 0; a < 3; ++a) {
        const int ns = axisCount(g, a);
        const double hs = axisSpacing(g, a);
        const double L = ns * hs;
        Axis& t = r.ax_[a];
        t.i0.resize(n[a]);
        t.i1.resize(n[a]);
        t.w.resize(n[a]);
        for (int p = 0; p < n[a]; ++p) {
            // Physical centre of the output pixel -> reference coordinate ->
            // fractional source CELL index (centres sit at (i + 1/2) h).
            const double x = (p + 0.5) * (L / n[a]);
            const double X = v[a].inverse ? v[a].inverse(x) : x;
            const double s = X / hs - 0.5;
            const double f = std::floor(s);
            int i0 = int(f) % ns;
            if (i0 < 0) i0 += ns;
            t.i0[p] = i0;
            t.i1[p] = (i0 + 1) % ns;
            t.w[p] = float(s - f);
        }
    }
    return r;
}

void PhysicalRaster::apply(const float* src, float* dst) const
{
    if (!active_) {
        std::copy_n(src, src_.size(), dst);
        return;
    }
    const std::size_t sx = 1, sy = std::size_t(src_.nx),
                      sz = std::size_t(src_.nx) * src_.ny;
    std::size_t q = 0;
    for (int k = 0; k < rg_.nz; ++k) {
        const int k0 = rg_.nz == 1 ? 0 : ax_[2].i0[k];
        const int k1 = rg_.nz == 1 ? 0 : ax_[2].i1[k];
        const float wk = rg_.nz == 1 ? 0.0f : ax_[2].w[k];
        for (int j = 0; j < rg_.ny; ++j) {
            const int j0 = ax_[1].i0[j], j1 = ax_[1].i1[j];
            const float wj = ax_[1].w[j];
            const std::size_t b00 = k0 * sz + j0 * sy, b01 = k0 * sz + j1 * sy;
            const std::size_t b10 = k1 * sz + j0 * sy, b11 = k1 * sz + j1 * sy;
            for (int i = 0; i < rg_.nx; ++i, ++q) {
                const std::size_t a0 = std::size_t(ax_[0].i0[i]) * sx;
                const std::size_t a1 = std::size_t(ax_[0].i1[i]) * sx;
                const float wi = ax_[0].w[i];
                auto lerpX = [&](std::size_t base) {
                    return src[base + a0] * (1.0f - wi) + src[base + a1] * wi;
                };
                const float y0 = lerpX(b00) * (1.0f - wj) + lerpX(b01) * wj;
                const float y1 = lerpX(b10) * (1.0f - wj) + lerpX(b11) * wj;
                dst[q] = y0 * (1.0f - wk) + y1 * wk;
            }
        }
    }
}

std::vector<float> PhysicalRaster::apply(const std::vector<float>& src) const
{
    std::vector<float> dst(rg_.size());
    if (src.empty()) return {};
    apply(src.data(), dst.data());
    return dst;
}

// The display raster for the two mesh maps the compressible path can have
// active, and the one place the per-axis dispatch lives for rendering.
//
// Frames are rasterized one pixel per CELL, so on a clustered mesh the picture
// is of the INDEX space: the plateau blown up, the far field squashed.
// Resampling onto a uniform PHYSICAL grid at the finest spacing makes the
// geometry true and costs no displayed detail. Both maps are separable and
// mutually exclusive; an inactive map leaves uniform-mesh runs bit-identical.
PhysicalRaster rasterFor(const Grid& g, const AxisRemap3& focal,
                         const TanhStretchY& stretch)
{
    AxisRemapView arv[3];
    {
        const double hs[3] = {g.hx, g.hy, g.nz > 1 ? g.hz : 1.0};
        const int n[3] = {g.nx, g.ny, g.nz};
        auto fwd = [&](int a, double X) {
            const Focal1D& f = a == 0 ? focal.fx : a == 1 ? focal.fy : focal.fz;
            return a == 1 ? f.map(stretch.mapY(X)) : f.map(X);
        };
        for (int a = 0; a < 3; ++a)
        {
            arv[a].L = double(n[a]) * hs[a];
            if (n[a] == 1) continue;
            // Finest physical spacing, straight off the node positions -- no
            // per-map special case, and it is exactly what the raster pitch
            // must be to lose nothing.
            double hf = arv[a].L;
            for (int i = 0; i < n[a]; ++i)
                hf = std::min(hf, fwd(a, (i + 1) * hs[a]) - fwd(a, i * hs[a]));
            arv[a].hFine = hf;
            arv[a].inverse = [&, a](double x) {
                const Focal1D& f =
                    a == 0 ? focal.fx : a == 1 ? focal.fy : focal.fz;
                const double X = f.inverse(x);
                return a == 1 ? stretch.invMapY(X) : X;
            };
        }
    }
    return PhysicalRaster::make(g, arv);
}

} // namespace dns
