#include "dns/wick.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace dns {
namespace {

// Exact signed distance to the axis-aligned box [x0, x1] x [y0, y1].
double boxSdf(double x0, double x1, double y0, double y1, double x, double y)
{
    const double qx = std::abs(x - 0.5 * (x0 + x1)) - 0.5 * (x1 - x0);
    const double qy = std::abs(y - 0.5 * (y0 + y1)) - 0.5 * (y1 - y0);
    const double ox = std::max(qx, 0.0), oy = std::max(qy, 0.0);
    return std::hypot(ox, oy) + std::min(std::max(qx, qy), 0.0);
}

} // namespace

double wickSdf(const WickSpec& s, double x, double y)
{
    const double body = boxSdf(s.xBase, s.xTop, s.yc - 0.5 * s.wCandle,
                               s.yc + 0.5 * s.wCandle, x, y);
    const double wick = boxSdf(s.xTop - s.wWick, s.xTip(), s.yc - 0.5 * s.wWick,
                               s.yc + 0.5 * s.wWick, x, y);
    return std::min(body, wick);
}

double wickWallT(const WickSpec& s, double x, double y)
{
    (void)y;
    return x > s.xTop + 1e-9 ? s.tWick : s.tCandle;
}

std::vector<std::uint8_t> wickRegions(const Grid& g, const WickSpec& s)
{
    assert(g.nz == 1);
    const int nx = g.nx, ny = g.ny;
    std::vector<std::uint8_t> r(g.size(), std::uint8_t(WickRegion::Fluid));
    auto cx = [&](int i) { return (i + 0.5) * g.hx; };
    auto cy = [&](int j) { return (j + 0.5) * g.hy; };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            if (wickSdf(s, cx(i), cy(j)) < 0)
                r[g.idx(i, j, 0)] = std::uint8_t(WickRegion::Solid);
    auto wickSolid = [&](int i, int j) {
        return r[g.idx(i, j, 0)] == std::uint8_t(WickRegion::Solid) &&
               cx(i) > s.xTop;
    };
    for (int j = 0; j < ny; ++j)
        for (int i = 1; i + 1 < nx; ++i) {
            if (r[g.idx(i, j, 0)] != std::uint8_t(WickRegion::Fluid)) continue;
            const int jm = (j + ny - 1) % ny, jp = (j + 1) % ny;
            if (wickSolid(i - 1, j) || wickSolid(i + 1, j) ||
                wickSolid(i, jm) || wickSolid(i, jp))
                r[g.idx(i, j, 0)] = std::uint8_t(WickRegion::Source);
        }
    return r;
}

double SootModel::lnAf(double sigma) const
{
    return std::log(Daf) + thetaF / sigma;
}

double SootModel::lnAox(double sigma) const
{
    return std::log(Daox) + thetaOx / sigma;
}

WickMeasure measureWick(const DiffusionModel& m, const WickSpec& s,
                        std::span<const float> YF, std::span<const float> YO,
                        std::span<const float> Ys,
                        std::span<const float> omega,
                        std::span<const std::uint8_t> solid, int nx, int ny,
                        double h)
{
    const std::size_t n = std::size_t(nx) * ny;
    assert(nx >= 2 && ny >= 4 && YF.size() == n && YO.size() == n &&
           omega.size() == n && solid.size() == n &&
           (Ys.empty() || Ys.size() == n));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double zs = m.zst();
    auto Z = [&](std::size_t c) { return mixtureFraction(m, YF[c], YO[c]); };
    WickMeasure r;
    r.zMin = HUGE_VAL;
    r.zMax = -HUGE_VAL;
    double wmax = 0;
    for (std::size_t c = 0; c < n; ++c) {
        if (solid[c]) continue;
        const double z = Z(c);
        r.zMin = std::min(r.zMin, z);
        r.zMax = std::max(r.zMax, z);
        bool out = z < -1e-6 || z > 1 + 1e-6 || YF[c] < -1e-6 ||
                   YF[c] > m.yF1 + 1e-6 || YO[c] < -1e-6 ||
                   YO[c] > m.yO2 + 1e-6;
        if (!Ys.empty()) {
            out = out || Ys[c] < -1e-6;
            r.sootMax = std::max(r.sootMax, double(Ys[c]));
        }
        r.hits += out ? 1.0 : 0.0;
        wmax = std::max(wmax, double(omega[c]));
    }
    const int jl = int(std::floor(s.yc / h - 0.5)), jr = jl + 1;
    assert(jl >= 0 && jr < ny);
    auto axis = [&](int i) {
        return 0.5 * (Z(std::size_t(jl) * nx + i) + Z(std::size_t(jr) * nx + i));
    };
    // First row whose centre lies above the wick tip.
    const int i0 = std::max(0, int(std::ceil(s.xTip() / h - 0.5)));
    r.xTip = nan;
    for (int i = i0; i + 1 < nx; ++i)
        if (axis(i) >= zs && axis(i + 1) < zs) {
            r.xTip = (i + 0.5 + (zs - axis(i)) / (axis(i + 1) - axis(i))) * h;
            break;
        }
    r.height = r.xTip - s.xTip();
    r.half = nan;
    if (std::isfinite(r.xTip)) {
        const int i = std::clamp(
            int(std::floor(0.5 * (s.xTip() + r.xTip) / h)), 0, nx - 1);
        auto Zc = [&](int j) { return Z(std::size_t(j) * nx + i); };
        double yR = nan, yL = nan;
        for (int j = jr; j + 1 < ny; ++j)
            if (Zc(j) >= zs && Zc(j + 1) < zs) {
                yR = (j + 0.5 + (zs - Zc(j)) / (Zc(j + 1) - Zc(j))) * h;
                break;
            }
        for (int j = jl; j - 1 >= 0; --j)
            if (Zc(j) >= zs && Zc(j - 1) < zs) {
                yL = (j + 0.5 - (zs - Zc(j)) / (Zc(j - 1) - Zc(j))) * h;
                break;
            }
        if (std::isfinite(yR) && std::isfinite(yL)) r.half = 0.5 * (yR - yL);
    }
    auto rowMax = [&](std::span<const float> f, int i) {
        double w = -HUGE_VAL;
        for (int j = 0; j < ny; ++j) {
            const std::size_t c = std::size_t(j) * nx + i;
            if (!solid[c]) w = std::max(w, double(f[c]));
        }
        return w;
    };
    r.xBase = nan;
    for (int i = 0; i < nx && wmax > 0; ++i)
        if (rowMax(omega, i) >= 0.1 * wmax) {
            r.xBase = i * h; // the row's lower face
            break;
        }
    r.xSoot = nan;
    if (!Ys.empty() && r.sootMax > 0)
        for (int i = nx - 1; i >= 0; --i)
            if (rowMax(Ys, i) >= 0.1 * r.sootMax) {
                r.xSoot = (i + 1) * h; // the row's upper face
                break;
            }
    return r;
}

} // namespace dns
