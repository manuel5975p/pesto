#include "dns/prop/curve.hpp"

#include <cmath>
#include <format>
#include <limits>
#include <numbers>

namespace dns::prop {

CurvePoint curvePoint(double J, double T, double Q, double re75, bool converged,
                      const Operating& op)
{
    const double n = op.revPerSec(), D = op.diameter, rho = op.rho;
    CurvePoint c;
    c.J = J;
    c.T = T;
    c.Q = Q;
    c.P = 2.0 * std::numbers::pi * n * Q;
    c.CT = T / (rho * n * n * std::pow(D, 4));
    c.CQ = Q / (rho * n * n * std::pow(D, 5));
    c.CP = c.P / (rho * std::pow(n, 3) * std::pow(D, 5));
    c.eta = c.CP > 0 && J > 0 ? J * c.CT / c.CP
                              : std::numeric_limits<double>::quiet_NaN();
    c.re75 = re75;
    c.converged = converged;
    return c;
}

std::string curveCsv(std::span<const CurvePoint> rows, bool withConverged)
{
    std::string s = withConverged ? "J,CT,CP,CQ,eta,T_N,Q_Nm,P_W,Re75,converged\n"
                                  : "J,CT,CP,CQ,eta,T_N,Q_Nm,P_W,Re75\n";
    for (const CurvePoint& r : rows)
    {
        s += std::format("{:.10g},{:.10g},{:.10g},{:.10g},{:.10g},{:.10g},{:.10g},"
                         "{:.10g},{:.10g}",
                         r.J, r.CT, r.CP, r.CQ, r.eta, r.T, r.Q, r.P, r.re75);
        s += withConverged ? std::format(",{}\n", int(r.converged)) : "\n";
    }
    return s;
}

std::string curveSummary(std::span<const CurvePoint> rows)
{
    std::string s;
    if (rows.empty()) return s;
    if (rows.front().J == 0.0)
        s += std::format("  static: CT(0) = {:.5f}, CP(0) = {:.5f}\n", rows.front().CT,
                         rows.front().CP);
    bool crossed = false;
    for (std::size_t k = 0; k + 1 < rows.size(); ++k)
    {
        const double c0 = rows[k].CT, c1 = rows[k + 1].CT;
        if (c0 > 0 && 0 >= c1)
        {
            const double j0 = rows[k].J + (rows[k + 1].J - rows[k].J) * c0 / (c0 - c1);
            s += std::format("  zero thrust at J = {:.4f}\n", j0);
            crossed = true;
            break;
        }
    }
    if (!crossed) s += "  zero thrust: not crossed in the J range\n";
    const CurvePoint* best = nullptr;
    for (const CurvePoint& r : rows)
        if (std::isfinite(r.eta) && (!best || r.eta > best->eta)) best = &r;
    if (best)
        s += std::format("  peak eta = {:.4f} at J = {:g} (CT {:.5f})\n", best->eta,
                         best->J, best->CT);
    return s;
}

} // namespace dns::prop
