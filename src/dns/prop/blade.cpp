#include "dns/prop/blade.hpp"

#include "dns/prop/csv.hpp"
#include "dns/prop/numeric.hpp"

#include <cassert>
#include <format>
#include <numbers>

namespace dns::prop {

std::expected<Planform, std::string> Planform::fromCsv(std::string_view text)
{
    auto rows = csvRows(text, "r_R,c_R,beta_deg", 3);
    if (!rows) return std::unexpected(rows.error());
    if (rows->size() < 2)
        return std::unexpected("need >= 2 rows of r_R,c_R,beta_deg");
    Planform p;
    for (const CsvRow& r : *rows)
    {
        const double x = r.v[0], c = r.v[1], b = r.v[2];
        if (!(x > 0.0 && x <= 1.0 + 1e-9) || !(c > 0.0) || !(b > -90.0 && b < 90.0))
            return std::unexpected(std::format(
                "line {}: need 0 < r_R <= 1, c_R > 0, |beta_deg| < 90, got {} {} {}",
                r.line, x, c, b));
        if (!p.x_.empty() && !(x > p.x_.back()))
            return std::unexpected(
                std::format("line {}: r_R not strictly increasing", r.line));
        p.x_.push_back(x);
        p.c_.push_back(c);
        // math.radians: x * (pi / 180).
        p.beta_.push_back(b * (std::numbers::pi / 180.0));
    }
    return p;
}

Section Planform::at(double r) const
{
    const double x = r / 0.5;
    double c, b;
    if (x <= x_.front()) { c = c_.front(); b = beta_.front(); }
    else if (x >= x_.back()) { c = c_.back(); b = beta_.back(); }
    else
    {
        std::size_t k = 1;
        while (x_[k] < x) ++k;
        const double f = (x - x_[k - 1]) / (x_[k] - x_[k - 1]);
        c = c_[k - 1] + f * (c_[k] - c_[k - 1]);
        b = beta_[k - 1] + f * (beta_[k] - beta_[k - 1]);
    }
    return {0.5 * c, b};   // c/R -> c/D: the tip radius is 0.5 D
}

double Planform::aspectRatio(double hubRR) const
{
    assert(hubRR > 0 && hubRR < 1);
    const double rh = hubRR * 0.5, R = 0.5;
    const auto rs = num::linspace(rh, R, 65);
    std::vector<double> chords;
    chords.reserve(rs.size());
    for (double r : rs) chords.push_back(at(r).chord);
    return (R - rh) / num::mean(chords);
}

} // namespace dns::prop
