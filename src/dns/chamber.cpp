#include "dns/chamber.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace dns {

double polygonSdf(std::span<const Point2> poly, double x, double y)
{
    assert(poly.size() >= 3);
    double d2 = std::numeric_limits<double>::infinity();
    bool inside = false;
    const std::size_t n = poly.size();
    for (std::size_t a = 0, b = n - 1; a < n; b = a++) {
        const double ax = poly[a][0], ay = poly[a][1];
        const double bx = poly[b][0], by = poly[b][1];
        const double ex = bx - ax, ey = by - ay;
        const double wx = x - ax, wy = y - ay;
        const double ee = ex * ex + ey * ey;
        const double t = ee > 0 ? std::clamp((wx * ex + wy * ey) / ee, 0.0, 1.0)
                                : 0.0;
        const double dx = wx - t * ex, dy = wy - t * ey;
        d2 = std::min(d2, dx * dx + dy * dy);
        // Even-odd crossing of the ray +x from (x, y).
        if ((ay > y) != (by > y) && x < ax + (y - ay) * ex / ey) inside = !inside;
    }
    const double d = std::sqrt(d2);
    return inside ? -d : d;
}

std::vector<Point2> chamberWallPolygon(const ChamberSpec& s)
{
    const double yc = s.yc, ho = 0.5 * s.Hc + s.tw, he = 0.5 * s.He;
    const double hi = 0.5 * s.Hc;
    return {{s.xe(), yc + he + s.tw}, {s.xn(), yc + ho},
            {s.x0, yc + ho},          {s.x0, yc - ho},
            {s.xn(), yc - ho},        {s.xe(), yc - he - s.tw},
            {s.xe(), yc - he},        {s.xn(), yc - hi},
            {s.xb(), yc - hi},        {s.xb(), yc + hi},
            {s.xn(), yc + hi},        {s.xe(), yc + he}};
}

std::vector<Point2> holderPolygon(const ChamberSpec& s)
{
    return {{s.xg, s.yc},
            {s.xBase(), s.yc + 0.5 * s.Hg},
            {s.xBase(), s.yc - 0.5 * s.Hg}};
}

std::vector<Point2> cavityPolygon(const ChamberSpec& s)
{
    const double yc = s.yc, hi = 0.5 * s.Hc, he = 0.5 * s.He;
    return {{s.xb(), yc - hi}, {s.xn(), yc - hi}, {s.xe(), yc - he},
            {s.xe(), yc + he}, {s.xn(), yc + hi}, {s.xb(), yc + hi}};
}

double chamberSdf(const ChamberSpec& s, double x, double y)
{
    // Rebuilt per call: 15 vertices, and the ghost classification calls this
    // only inside its bbox band.
    const auto w = chamberWallPolygon(s);
    const auto h = holderPolygon(s);
    return std::min(polygonSdf(w, x, y), polygonSdf(h, x, y));
}

std::vector<std::uint8_t> chamberRegions(const Grid& g, const ChamberSpec& s)
{
    assert(g.nz == 1);
    const auto w = chamberWallPolygon(s);
    const auto hp = holderPolygon(s);
    const auto cav = cavityPolygon(s);
    const double h = g.hx;
    // The exit column: the last one whose centre lies inside x < xe.
    const int ie = int(std::ceil(s.xe() / h - 0.5)) - 1;
    std::vector<std::uint8_t> r(g.size());
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const double x = (i + 0.5) * h, y = (j + 0.5) * g.hy;
            const std::size_t c = g.idx(i, j, 0);
            const double sdf =
                std::min(polygonSdf(w, x, y), polygonSdf(hp, x, y));
            ChamberRegion k;
            if (sdf < 0) k = ChamberRegion::Solid;
            else if (polygonSdf(cav, x, y) >= 0)
                k = i == ie ? ChamberRegion::AmbientProbe : ChamberRegion::Ambient;
            else if (i == ie) k = ChamberRegion::Exit;
            else if (x < s.xb() + s.wInj) k = ChamberRegion::Injector;
            else if (x >= s.xb() + s.wInj + h && x <= s.xg - h)
                k = ChamberRegion::FreshProbe;
            else if (x >= s.xn() - 1.6 && x < s.xn())
                k = ChamberRegion::NozzleProbe;
            else k = ChamberRegion::Cavity;
            r[c] = std::uint8_t(k);
        }
    return r;
}

void chamberIC(const Grid& g, const std::vector<std::uint8_t>& regions,
               double gamma, double Tu, double sigma, double uCo,
               const std::function<double(double)>& theta, double xBase,
               CompState& q, std::vector<double>& rhoY)
{
    assert(regions.size() == g.size() && Tu > 0 && sigma > 1 && gamma > 1);
    q = CompState::zeros(g);
    rhoY.assign(g.size(), 0.0);
    const double p0 = Tu; // rho_u = R = 1
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const std::uint8_t r = regions[c];
            double T = Tu, Y = 0, u = 0;
            if (inCavity(r)) {
                const double tt = theta((i + 0.5) * g.hx - xBase);
                T = Tu * tt;
                Y = (sigma - tt) / (sigma - 1);
            } else if (r == std::uint8_t(ChamberRegion::Ambient) ||
                       r == std::uint8_t(ChamberRegion::AmbientProbe)) {
                u = uCo;
            }
            const double rho = p0 / T;
            q.rho[c] = rho;
            q.mx[c] = rho * u;
            q.E[c] = p0 / (gamma - 1) + 0.5 * rho * u * u;
            rhoY[c] = r == std::uint8_t(ChamberRegion::Solid) ? 0.0 : rho * Y;
        }
}

std::expected<NozzleEstimate, std::string>
isentropicNozzle(double mdot, double T0, double A, double pa, double gamma,
                 double R)
{
    assert(mdot > 0 && T0 > 0 && A > 0 && pa > 0 && gamma > 1 && R > 0);
    const double gm = 0.5 * (gamma - 1);
    auto flux = [&](double M) {
        return pa * M * std::sqrt(gamma * (1 + gm * M * M) / (R * T0));
    };
    const double target = mdot / A;
    if (target >= flux(1.0))
        return std::unexpected("mass flux at or above the sonic one (choked)");
    double lo = 0, hi = 1;
    for (int it = 0; it < 200 && hi - lo > 1e-15; ++it) {
        const double m = 0.5 * (lo + hi);
        (flux(m) < target ? lo : hi) = m;
    }
    const double Me = 0.5 * (lo + hi);
    return NozzleEstimate{Me, pa * std::pow(1 + gm * Me * Me, gamma / (gamma - 1))};
}

} // namespace dns
