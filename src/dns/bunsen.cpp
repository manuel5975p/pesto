#include "dns/bunsen.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace dns {

namespace {

// The burner's radial coordinate: |y - yc| for the slot (z-invariant),
// hypot(y - yc, z - zc) for the round nozzle.
double burnerR(const BunsenGeom& b, double y, double z)
{
    return b.round ? std::hypot(y - b.yc, z - b.zc) : std::abs(y - b.yc);
}

BunsenInflowRow inflowAtR(const BunsenGeom& b, double r)
{
    assert(b.edge > 0);
    const double jet = 0.5 * (1.0 - std::tanh((r - 0.5 * b.W) / b.edge));
    const double co =
        0.5 * (1.0 + std::tanh((r - 0.5 * b.W - b.pilot) / b.edge));
    const double p = b.pilot > 0 ? 1.0 - jet - co : 0.0;
    return {b.uJet * jet + b.uPilot * p + b.uCo * co,
            jet + b.tPilot * p + b.tCo * co, jet, jet, p, co};
}

double coneDistanceAtR(const BunsenGeom& b, double x, double r)
{
    assert(b.W > 0 && b.coneH > 0);
    // In the (x, r) half plane the flank is the segment P0 = (0, W/2) ->
    // P1 = (coneH, 0); inside = r below the flank line for 0 <= x <= coneH.
    const double ax = b.coneH, ar = -0.5 * b.W;
    const double L2 = ax * ax + ar * ar;
    const double s = std::clamp((x * ax + (r - 0.5 * b.W) * ar) / L2, 0.0, 1.0);
    const double dx = x - s * ax, dr = r - (0.5 * b.W + s * ar);
    const double d = std::hypot(dx, dr);
    const bool inside = x >= 0 && x <= b.coneH &&
                        r < 0.5 * b.W * (1.0 - x / b.coneH);
    return inside ? -d : d;
}

} // namespace

BunsenInflowRow bunsenInflow(const BunsenGeom& b, double y)
{
    assert(!b.round);
    return inflowAtR(b, std::abs(y - b.yc));
}

BunsenInflowRow bunsenInflowAt(const BunsenGeom& b, double y, double z)
{
    return inflowAtR(b, burnerR(b, y, z));
}

double bunsenConeDistance(const BunsenGeom& b, double x, double y)
{
    assert(!b.round);
    return coneDistanceAtR(b, x, std::abs(y - b.yc));
}

double bunsenConeDistanceAt(const BunsenGeom& b, double x, double y, double z)
{
    return coneDistanceAtR(b, x, burnerR(b, y, z));
}

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Linear fit y = a + b x (FP64, plain sums: the rows are few and O(1)).
struct Line
{
    double a = 0, b = 0, rms = 0;
};
Line fitLine(std::span<const double> x, std::span<const double> y)
{
    const double n = double(x.size());
    double sx = 0, sy = 0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        sx += x[k];
        sy += y[k];
    }
    const double mx = sx / n, my = sy / n;
    double sxx = 0, sxy = 0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        sxx += (x[k] - mx) * (x[k] - mx);
        sxy += (x[k] - mx) * (y[k] - my);
    }
    Line l;
    l.b = sxy / sxx;
    l.a = my - l.b * mx;
    double ss = 0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double e = y[k] - (l.a + l.b * x[k]);
        ss += e * e;
    }
    l.rms = std::sqrt(ss / n);
    return l;
}

} // namespace

ConeMeasure measureCone(std::span<const float> T, int nx, int ny, double h,
                        double Tmid, double yc, std::span<const float> omega,
                        int win)
{
    assert(nx >= 2 && ny >= 4 && T.size() == std::size_t(nx) * ny);
    const int jl = int(std::floor(yc / h - 0.5)), jr = jl + 1;
    assert(jl >= 0 && jr < ny);
    auto at = [&](int i, int j) { return double(T[std::size_t(j) * nx + i]); };
    auto axis = [&](int i) { return 0.5 * (at(i, jl) + at(i, jr)); };
    // Crossing position between centres q and q + 1 (fractional, 0..1).
    auto frac = [&](double a, double b) { return (Tmid - a) / (b - a); };

    ConeMeasure m;
    m.xTip = kNaN;
    m.xBase = kNaN;
    m.axisHot = !(axis(0) < Tmid);
    if (m.axisHot) return m; // the axis starts hot: flash-back
    for (int i = 0; i + 1 < nx; ++i)
        if (axis(i) < Tmid && axis(i + 1) >= Tmid) {
            m.xTip = (i + 0.5 + frac(axis(i), axis(i + 1))) * h;
            break;
        }
    if (!std::isfinite(m.xTip)) return m; // blow-through

    // A crossing between cells j and j + 1 of row i burns (or no omega).
    const double wcut =
        omega.empty() ? 0.0
                      : 0.1 * double(*std::max_element(omega.begin(), omega.end()));
    auto burns = [&](int i, int j) {
        if (omega.empty()) return true;
        double w = 0;
        for (int q = std::max(0, j - win); q <= std::min(ny - 1, j + 1 + win); ++q)
            w = std::max(w, double(omega[std::size_t(q) * nx + i]));
        return w >= wcut;
    };
    // Flank crossings of row i, walking outward from the axis.
    auto flank = [&](int i) -> std::optional<std::pair<double, double>> {
        if (!(axis(i) < Tmid)) return std::nullopt;
        double yR = kNaN, yL = kNaN;
        for (int j = jr; j + 1 < ny; ++j)
            if (at(i, j) < Tmid && at(i, j + 1) >= Tmid) {
                if (burns(i, j))
                    yR = (j + 0.5 + frac(at(i, j), at(i, j + 1))) * h;
                break;
            }
        for (int j = jl; j - 1 >= 0; --j)
            if (at(i, j) < Tmid && at(i, j - 1) >= Tmid) {
                if (burns(i, j - 1))
                    yL = (j + 0.5 - frac(at(i, j), at(i, j - 1))) * h;
                break;
            }
        if (!std::isfinite(yR) || !std::isfinite(yL)) return std::nullopt;
        return std::pair{yL, yR};
    };
    for (int i = 0; i < nx; ++i)
        if ((i + 0.5) * h < m.xTip && flank(i)) {
            m.xBase = (i + 0.5) * h;
            break;
        }
    if (!std::isfinite(m.xBase)) return m;

    const double lo = m.xBase + 0.25 * (m.xTip - m.xBase);
    const double hi = m.xBase + 0.75 * (m.xTip - m.xBase);
    std::vector<double> xs, hw, sr, sl;
    for (int i = 0; i < nx; ++i) {
        const double x = (i + 0.5) * h;
        if (x < lo || x > hi) continue;
        const auto f = flank(i);
        if (!f) continue;
        if (m.rows == 0) m.iLo = i;
        m.iHi = i;
        ++m.rows;
        m.fi.push_back(i);
        m.fyL.push_back(f->first);
        m.fyR.push_back(f->second);
        xs.push_back(x);
        hw.push_back(0.5 * (f->second - f->first));
        sr.push_back(f->second - yc);
        sl.push_back(yc - f->first);
    }
    if (m.rows < 4) return m;
    const Line a = fitLine(xs, hw), r = fitLine(xs, sr), l = fitLine(xs, sl);
    m.tanA = -a.b;
    m.tanR = -r.b;
    m.tanL = -l.b;
    m.rms = a.rms;
    m.ok = true;
    return m;
}

double flankMassFlux(const ConeMeasure& m, std::span<const float> rho,
                     std::span<const float> u, std::span<const float> v,
                     int nx, double h)
{
    if (!m.ok) return kNaN;
    const double nR = std::hypot(1.0, m.tanR), nL = std::hypot(1.0, m.tanL);
    // Outward normals: right flank y - yc = a - x tanR -> (tanR, 1)/|.|,
    // left flank yc - y = a - x tanL -> (tanL, -1)/|.|.
    const double nxR = m.tanR / nR, nyR = 1.0 / nR;
    const double nxL = m.tanL / nL, nyL = -1.0 / nL;
    auto lerp = [&](std::span<const float> f, int i, double y) {
        const double s = y / h - 0.5;
        const int j = int(std::floor(s));
        const double w = s - j;
        const auto at = [&](int jj) { return double(f[std::size_t(jj) * nx + i]); };
        return (1 - w) * at(j) + w * at(j + 1);
    };
    double sum = 0;
    for (std::size_t k = 0; k < m.fi.size(); ++k) {
        const int i = m.fi[k];
        for (const auto& [y, ex, ey] : {std::tuple{m.fyR[k], nxR, nyR},
                                        std::tuple{m.fyL[k], nxL, nyL}})
            sum += lerp(rho, i, y) * (lerp(u, i, y) * ex + lerp(v, i, y) * ey);
    }
    return sum / (2.0 * double(m.fi.size()));
}

} // namespace dns
