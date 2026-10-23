// The blade planform, propeller.py's `--planform` table: chord and pitch angle
// linear in r/R between rows and held constant past either end.
#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace dns::prop {

// Section at one radius, unit-diameter blade (tip radius 0.5).
struct Section
{
    double chord;   // in diameters
    double beta;    // geometric pitch angle [rad]
};

class Planform
{
public:
    // The --planform CSV: header `r_R,c_R,beta_deg`, >= 2 rows hub -> tip,
    // 0 < r/R <= 1 strictly increasing, c/R > 0, |beta| < 90 deg.
    static std::expected<Planform, std::string> fromCsv(std::string_view text);

    // propeller._planform_at at radius r (unit diameter): x = r/0.5.
    Section at(double r) const;

    // The table's first r/R: where the measured blade starts.
    double rootRR() const { return x_.front(); }

    // polars.aspect_ratio: (R - rh) / mean chord over 65 stations, unit
    // diameter. Pre: 0 < hubRR < 1.
    double aspectRatio(double hubRR) const;

private:
    std::vector<double> x_, c_, beta_;   // r/R, c/R, beta [rad]
};

} // namespace dns::prop
