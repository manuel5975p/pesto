#include "dns/prop/polar.hpp"

#include "dns/prop/csv.hpp"
#include "dns/prop/numeric.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <numbers>

namespace dns::prop {
namespace {

using std::numbers::pi;

constexpr double kBackCl = 0.7;          // reversed-flow cl scale (AirfoilPrep)
constexpr double kThinStallDeg = 10.0;   // thin-airfoil linear range about alpha0
constexpr double kDegToRad = pi / 180.0;
constexpr double kRadToDeg = 180.0 / pi;
constexpr double kAlphaEndTolDeg = 1e-6; // a table must reach -180 and +180

// Viterna-Corrigan at al in [als, pi/2], matching (cls, cds) at als.
ClCd viterna(double al, double als, double cls, double cds, double cdm)
{
    const double sa = std::sin(als), ca = std::cos(als);
    const double a2 = (cls - cdm * sa * ca) * sa / (ca * ca);
    const double b2 = (cds - cdm * sa * sa) / ca;
    const double c = std::cos(al), s = std::sin(al);
    return {0.5 * cdm * std::sin(2 * al) + a2 * (c * c) / s, cdm * (s * s) + b2 * c};
}

// grid(lo, hi) of polars.extend: max(int(deg(hi - lo)) + 1, 2) points.
std::vector<double> extendGrid(double lo, double hi)
{
    const int n = std::max(int((hi - lo) * kRadToDeg) + 1, 2);
    return num::linspace(lo, hi, n);
}

// polars.extend: data on [alo, ahi] -> table on [-pi, pi]. Pre: alo < 0 < ahi
// < 90 deg, alpha in radians, increasing.
PolarTable extend(const std::vector<double>& al, const std::vector<double>& cl,
                  const std::vector<double>& cd, double cdm)
{
    const double ahi = al.back(), clh = cl.back(), cdh = cd.back();
    const double alo = -al.front(), cll = -cl.front(), cdl = cd.front();
    const double cd180 = 0.5 * (cdh + cdl);
    const double half = pi / 2;

    auto back = [&](double ap, double as, double cls, double cds) {
        const ClCd v = viterna(std::max(ap, as), as, cls, cds, cdm);
        const double t = std::clamp(ap / as, 0.0, 1.0);
        return ap >= as ? ClCd{-kBackCl * v.cl, v.cd}
                        : ClCd{-kBackCl * cls * t, cd180 + (cds - cd180) * t};
    };

    PolarTable out;
    auto push = [&](double a, ClCd v) {
        out.alpha.push_back(a);
        out.cl.push_back(v.cl);
        out.cd.push_back(v.cd);
    };
    // [-180, -90): reversed flow off the low side, mirrored.
    const auto gb = extendGrid(half, pi);
    for (std::size_t i = gb.size() - 1; i >= 1; --i)
    {
        const double b = -gb[i];
        const ClCd v = back(pi + b, alo, cll, cdl);
        push(b, {-v.cl, v.cd});
    }
    // [-90, alo): Viterna off the low side, mirrored.
    const auto gn = extendGrid(alo, half);
    for (std::size_t i = gn.size() - 1; i >= 1; --i)
    {
        const double a = -gn[i];
        const ClCd v = viterna(-a, alo, cll, cdl, cdm);
        push(a, {-v.cl, v.cd});
    }
    for (std::size_t i = 0; i < al.size(); ++i) push(al[i], {cl[i], cd[i]});
    const auto gp = extendGrid(ahi, half);
    for (std::size_t i = 1; i < gp.size(); ++i)
        push(gp[i], viterna(gp[i], ahi, clh, cdh, cdm));
    for (std::size_t i = 1; i < gb.size(); ++i)
        push(gb[i], back(pi - gb[i], ahi, clh, cdh));
    assert(std::ranges::adjacent_find(out.alpha, std::ranges::greater_equal{})
           == out.alpha.end());
    return out;
}

} // namespace

std::expected<Polar, std::string> Polar::fromCsv(std::string_view text)
{
    auto rows = csvRows(text, "re,alpha_deg,cl,cd", 4);
    if (!rows) return std::unexpected(rows.error());
    if (rows->empty()) return std::unexpected("no data rows");
    Polar p;
    for (const CsvRow& r : *rows)
    {
        const double re = r.v[0], adeg = r.v[1], cl = r.v[2], cd = r.v[3];
        if (!(re > 0))
            return std::unexpected(std::format("line {}: re must be > 0", r.line));
        if (!(cd > 0))
            return std::unexpected(std::format("line {}: cd must be > 0", r.line));
        if (!(adeg >= -180.0 - kAlphaEndTolDeg && adeg <= 180.0 + kAlphaEndTolDeg))
            return std::unexpected(
                std::format("line {}: alpha_deg {} outside [-180, 180]", r.line, adeg));
        if (p.tables_.empty() || re != p.tables_.back().re)
        {
            if (!p.tables_.empty() && re < p.tables_.back().re)
                return std::unexpected(std::format(
                    "line {}: re {} after {} -- groups must be ascending in re "
                    "(and each re one contiguous group)",
                    r.line, re, p.tables_.back().re));
            p.tables_.push_back({.re = re, .alpha = {}, .cl = {}, .cd = {}});
        }
        PolarTable& t = p.tables_.back();
        const double a = adeg * kDegToRad;
        if (!t.alpha.empty() && !(a > t.alpha.back()))
            return std::unexpected(std::format(
                "line {}: alpha_deg not strictly increasing within re {}", r.line, re));
        t.alpha.push_back(a);
        t.cl.push_back(cl);
        t.cd.push_back(cd);
    }
    for (const PolarTable& t : p.tables_)
    {
        const double lo = t.alpha.front() * kRadToDeg, hi = t.alpha.back() * kRadToDeg;
        if (t.alpha.size() < 2 || std::fabs(lo + 180.0) > kAlphaEndTolDeg
            || std::fabs(hi - 180.0) > kAlphaEndTolDeg)
            return std::unexpected(std::format(
                "re {}: alpha must span exactly -180..180 deg (got {}..{}); "
                "export the Viterna-extended table",
                t.re, lo, hi));
        p.logRe_.push_back(std::log(t.re));
    }
    return p;
}

Polar Polar::thinAirfoil(double alpha0Deg, double cd0, double cdMax)
{
    const auto al = num::linspace(alpha0Deg - kThinStallDeg,
                                  alpha0Deg + kThinStallDeg, 41);
    std::vector<double> rad(al.size()), cl(al.size()), cd(al.size(), cd0);
    for (std::size_t i = 0; i < al.size(); ++i)
    {
        rad[i] = al[i] * kDegToRad;
        cl[i] = 2.0 * pi * ((al[i] - alpha0Deg) * kDegToRad);
    }
    Polar p;
    p.tables_.push_back(extend(rad, cl, cd, cdMax));
    p.logRe_.push_back(0.0);
    p.thin_ = true;
    return p;
}

ClCd Polar::at(double alpha, double re) const
{
    assert(!tables_.empty());
    const double w = num::mod(alpha + pi, 2.0 * pi) - pi;
    auto one = [w](const PolarTable& t) {
        return ClCd{num::interp(w, t.alpha, t.cl), num::interp(w, t.alpha, t.cd)};
    };
    if (tables_.size() == 1) return one(tables_[0]);
    const double x = std::log(std::clamp(re, tables_.front().re, tables_.back().re));
    const auto pos = std::ranges::lower_bound(logRe_, x) - logRe_.begin();
    const std::size_t k = std::size_t(
        std::clamp<std::ptrdiff_t>(pos - 1, 0, std::ptrdiff_t(tables_.size()) - 2));
    const double f = (x - logRe_[k]) / (logRe_[k + 1] - logRe_[k]);
    const ClCd a = one(tables_[k]), b = one(tables_[k + 1]);
    return {(1 - f) * a.cl + f * b.cl, (1 - f) * a.cd + f * b.cd};
}

double viternaCdMax(double aspectRatio) { return 1.11 + 0.018 * std::min(aspectRatio, 50.0); }

} // namespace dns::prop
