// One thrust-curve row and the conventions every prop method shares:
// PROPULSIVE signs (T > 0 pulls upstream, Q > 0 is motor torque, P = 2 pi n Q),
// CT = T/(rho n^2 D^4), CP = P/(rho n^3 D^5), CQ = CP/(2 pi), eta = J CT/CP
// (NaN where CP <= 0 or J = 0).
#pragma once

#include "dns/prop/spec.hpp"

#include <span>
#include <string>

namespace dns::prop {

struct CurvePoint
{
    double J = 0, CT = 0, CP = 0, CQ = 0, eta = 0;
    double T = 0, Q = 0, P = 0;   // N, N m, W
    double re75 = 0;              // section Re at r = 0.75 R
    bool converged = true;        // lifting line only; BEMT always true
};

// The row for thrust T and torque Q at advance ratio J.
CurvePoint curvePoint(double J, double T, double Q, double re75, bool converged,
                      const Operating& op);

// The CSV `J,CT,CP,CQ,eta,T_N,Q_Nm,P_W,Re75[,converged]`.
std::string curveCsv(std::span<const CurvePoint> rows, bool withConverged);

// The three curve anchors (static CT, zero-thrust J, peak eta), as text.
std::string curveSummary(std::span<const CurvePoint> rows);

} // namespace dns::prop
