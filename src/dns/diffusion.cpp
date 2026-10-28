#include "dns/diffusion.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>

namespace dns {

double DiffusionModel::lnA() const
{
    return std::log(Da) + (sigma > 1 ? theta() / sigma : 0.0);
}

double mixtureFraction(const DiffusionModel& m, double YF, double YO)
{
    return (m.s * YF - YO + m.yO2) / (m.s * m.yF1 + m.yO2);
}

SheetState burkeSchumann(const DiffusionModel& m, double Z)
{
    const double zs = m.zst();
    SheetState st{};
    if (Z <= zs) {
        st.YF = 0;
        st.YO = m.yO2 * (1.0 - Z / zs);
    } else {
        st.YF = m.yF1 * (Z - zs) / (1.0 - zs);
        st.YO = 0;
    }
    st.t = 1.0 + m.qOverCpT0() * (m.yF1 * Z - st.YF);
    return st;
}

double erfcInv(double y)
{
    assert(y > 0 && y < 2);
    // erfc is decreasing: bisect to a bracket of 1e-3, then Newton.
    double lo = -6, hi = 6;
    while (hi - lo > 1e-3) {
        const double mid = 0.5 * (lo + hi);
        (std::erfc(mid) > y ? lo : hi) = mid;
    }
    double x = 0.5 * (lo + hi);
    for (int k = 0; k < 50; ++k) {
        const double f = std::erfc(x) - y;
        const double d = -2.0 / std::sqrt(std::numbers::pi) * std::exp(-x * x);
        const double dx = f / d;
        x -= dx;
        if (std::abs(dx) <= 1e-16 * std::max(1.0, std::abs(x))) break;
    }
    return x;
}

DiffusionInflowRow diffusionInflow(const DiffusionModel& m, const BunsenGeom& b,
                                   double y)
{
    const BunsenInflowRow w = bunsenInflow(b, y);
    const double Z = w.jet + m.zst() * w.pilot;
    const SheetState st = burkeSchumann(m, Z);
    return {w.jet + b.uPilot * w.pilot + b.uCo * w.co, Z, st.YF, st.YO, st.t,
            w.jet};
}

JetMeasure measureJet(const DiffusionModel& m, std::span<const float> YF,
                      std::span<const float> YO, std::span<const float> omega,
                      int nx, int ny, double h, double yc)
{
    const std::size_t n = std::size_t(nx) * ny;
    assert(nx >= 2 && ny >= 4 && YF.size() == n && YO.size() == n &&
           omega.size() == n);
    const int jl = int(std::floor(yc / h - 0.5)), jr = jl + 1;
    assert(jl >= 0 && jr < ny);
    const double zs = m.zst(), nan = std::numeric_limits<double>::quiet_NaN();
    auto Z = [&](int i, int j) {
        const std::size_t c = std::size_t(j) * nx + i;
        return mixtureFraction(m, YF[c], YO[c]);
    };
    JetMeasure r;
    r.zMin = HUGE_VAL;
    r.zMax = -HUGE_VAL;
    r.yfMin = HUGE_VAL;
    r.yoMin = HUGE_VAL;
    double wmax = 0;
    for (std::size_t c = 0; c < n; ++c) {
        const double z = mixtureFraction(m, YF[c], YO[c]);
        r.zMin = std::min(r.zMin, z);
        r.zMax = std::max(r.zMax, z);
        r.yfMin = std::min(r.yfMin, double(YF[c]));
        r.yoMin = std::min(r.yoMin, double(YO[c]));
        const bool out = z < -1e-6 || z > 1 + 1e-6 || YF[c] < -1e-6 ||
                         YF[c] > m.yF1 + 1e-6 || YO[c] < -1e-6 ||
                         YO[c] > m.yO2 + 1e-6;
        r.hits += out ? 1.0 : 0.0;
        wmax = std::max(wmax, double(omega[c]));
    }
    auto axis = [&](int i) { return 0.5 * (Z(i, jl) + Z(i, jr)); };
    r.xTip = nan;
    for (int i = 0; i + 1 < nx; ++i)
        if (axis(i) >= zs && axis(i + 1) < zs) {
            r.xTip = (i + 0.5 + (zs - axis(i)) / (axis(i + 1) - axis(i))) * h;
            break;
        }
    r.xBase = nan;
    for (int i = 0; i < nx && wmax > 0; ++i) {
        double w = 0;
        for (int j = 0; j < ny; ++j)
            w = std::max(w, double(omega[std::size_t(j) * nx + i]));
        if (w >= 0.1 * wmax) {
            r.xBase = i * h; // the row's lower face: 0 = burning in row 0
            break;
        }
    }
    r.half = nan;
    if (std::isfinite(r.xTip)) {
        const int i = std::clamp(int(std::floor(0.5 * r.xTip / h)), 0, nx - 1);
        double yR = nan, yL = nan;
        for (int j = jr; j + 1 < ny; ++j)
            if (Z(i, j) >= zs && Z(i, j + 1) < zs) {
                yR = (j + 0.5 + (zs - Z(i, j)) / (Z(i, j + 1) - Z(i, j))) * h;
                break;
            }
        for (int j = jl; j - 1 >= 0; --j)
            if (Z(i, j) >= zs && Z(i, j - 1) < zs) {
                yL = (j + 0.5 - (zs - Z(i, j)) / (Z(i, j - 1) - Z(i, j))) * h;
                break;
            }
        if (std::isfinite(yR) && std::isfinite(yL)) r.half = 0.5 * (yR - yL);
    }
    return r;
}

double candleAmbientP(double p0, double g, double RT0, double x)
{
    assert(p0 > 0 && g >= 0 && RT0 > 0);
    return g > 0 ? p0 * std::exp(-g * x / RT0) : p0;
}

CandleScales candleScales(const DiffusionModel& m, double g, double W,
                          double L, double mach)
{
    assert(g > 0 && W > 0 && L > 0 && mach > 0 && m.sigma > 1);
    constexpr double gam = 1.4;
    CandleScales s{};
    s.froude = 1.0 / (g * W);
    s.uFreeFall = std::sqrt(2.0 * g * L * (m.sigma - 1.0) / m.sigma);
    s.machBuoyant = mach * s.uFreeFall;
    s.adiabatic = g * L * (gam - 1.0) * mach * mach / (m.sigma - 1.0);
    s.hydroDrop = -std::expm1(-gam * mach * mach * g * L);
    s.flicker = 1.5 / std::sqrt(9.81) * std::sqrt(g / W);
    return s;
}

DiffusionInflowRow quiescentInflow(const DiffusionModel& m)
{
    return {0.0, 0.0, 0.0, m.yO2, 1.0, 0.0};
}

double sheetTipConstantDensity(double W, double zst)
{
    assert(W > 0 && zst > 0 && zst < 1);
    const double e = erfcInv(1.0 - zst); // erfinv(zst)
    return W * W / (16.0 * e * e);
}

} // namespace dns
