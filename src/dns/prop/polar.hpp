// Section polars cl(alpha, Re), cd(alpha, Re) over the whole alpha circle --
// the C++ side of tools/polars.py's Polar, which owns XFOIL and the Viterna
// extension and exports the finished tables as CSV.
#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace dns::prop {

// One Reynolds number's table: alpha [rad] strictly increasing over [-pi, pi].
struct PolarTable
{
    double re = 1.0;
    std::vector<double> alpha, cl, cd;
};

struct ClCd
{
    double cl, cd;
};

// Polar.cl_cd: linear in alpha (wrapped to [-pi, pi)), linear in log Re
// between tables, Re clamped to the table range (never extrapolated).
class Polar
{
public:
    // The --polar FILE CSV: header `re,alpha_deg,cl,cd`, grouped by Re
    // strictly ascending, alpha strictly ascending from -180 to 180 deg in
    // each group. Unexpected = a message naming the line (input error).
    static std::expected<Polar, std::string> fromCsv(std::string_view text);

    // Polar.from_thin_airfoil: 2 pi (alpha - alpha0) and constant cd0 on
    // alpha0 +- 10 deg, Viterna past it with flat-plate drag cdMax, one table.
    // Pre: 0 < cd0 < 1, cdMax > 0, |alpha0Deg| < 45.
    static Polar thinAirfoil(double alpha0Deg, double cd0, double cdMax);

    ClCd at(double alpha, double re) const;

    // The built-in thin-airfoil polar (BEMT then needs one Re pass, not three).
    bool thin() const { return thin_; }

private:
    std::vector<PolarTable> tables_;
    std::vector<double> logRe_;
    bool thin_ = false;
};

// Viterna's flat-plate drag at 90 deg, 1.11 + 0.018 AR with AR capped at 50.
double viternaCdMax(double aspectRatio);

} // namespace dns::prop
