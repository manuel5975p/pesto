// dns/pseudo_time: the implicit march's policy -- the CFL ramp and its cap,
// holding on a damped step, rejection, when the stop is ARMED (only after an
// undamped step taken at the cap: unarmed, the stop once certified ub 1.1% off
// after rejections, masterplan R-b), and the step-length limiter, bound by
// bound. Host only, milliseconds.
#include "dns/pseudo_time.hpp"
#include "channel_line_test_util.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

void ramp()
{
    PseudoTime pt;
    bool exact = true, quiet = true;
    int n = 0;
    for (; pt.cfl() < kImplicitCflMax && n < 64; ++n) {
        exact = exact && pt.cfl() == kImplicitCfl0 * std::ldexp(1.0, n);
        pt.accepted(1.0);
        quiet = quiet && !pt.armed();
    }
    check(exact, "CFL sequence is 100 * 2^n below the cap");
    check(n == 20 && pt.cfl() == kImplicitCflMax, "the cap 1e8 is reached at n = 20");
    check(quiet, "never armed while the steps were taken below the cap");
    pt.accepted(1.0);
    check(pt.armed() && pt.cfl() == kImplicitCflMax,
          "armed after an undamped step AT the cap; the cap holds");
    pt.accepted(0.5);
    check(!pt.armed() && pt.cfl() == kImplicitCflMax,
          "a damped step (alpha < 1) holds the CFL and disarms");
    pt.accepted(1.0);
    check(pt.armed(), "the next undamped step at the cap re-arms");
}

void hold()
{
    PseudoTime pt;
    pt.accepted(1.0);
    pt.accepted(0.9);
    check(pt.cfl() == 200 && !pt.armed(), "alpha < 1 below the cap holds the CFL");
}

void reject()
{
    PseudoTime pt;
    for (int i = 0; i < 25; ++i) pt.accepted(1.0);
    check(pt.armed(), "armed at the cap");
    const bool ok = pt.rejected();
    check(ok && !pt.armed() && pt.cfl() == kImplicitCflMax * kImplicitRejectFactor,
          "rejected(): x0.1 and disarmed");
    PseudoTime a(10.0);
    check(a.rejected() && a.cfl() == 1.0, "CFL 10 -> 1: still allowed (>= 1)");
    check(!a.rejected(), "CFL 1 -> 0.1: below kImplicitCflMin, false");
}

// R-b: ramp, reject, reject, regrow. The history must stay empty below the
// cap -- a1's unarmed run certified at iteration 11 with ub 1.1% off.
void neverArmedBelowCap()
{
    PseudoTime pt;
    bool armedBelow = false;
    for (int i = 0; i < 6; ++i) {
        const double c = pt.cfl();
        pt.accepted(1.0);
        armedBelow = armedBelow || (pt.armed() && c < kImplicitCflMax);
    }
    (void)pt.rejected();
    (void)pt.rejected();
    int steps = 0;
    while (!pt.armed() && steps < 100) {
        const double c = pt.cfl();
        pt.accepted(steps % 3 == 1 ? 0.7 : 1.0);
        armedBelow = armedBelow || (pt.armed() && c < kImplicitCflMax);
        ++steps;
    }
    check(!armedBelow, "R-b: [ramp, reject, reject, regrow] never arms below the cap");
    check(pt.armed(), "R-b: and arms once at the cap");
}

// One unknown of one fluid row moved at a time; alpha must equal that bound.
void stepBounds()
{
    for (RansModel model : {RansModel::SaNeg, RansModel::Sst2003m}) {
        Channel1dCase cc;
        cc.ny = 16;
        cc.model = model;
        const auto r = linetest::makeLine(cc);
        const auto q = linetest::icColumn(r);
        const int F = r.line.fields(), B = r.line.unknowns(), j = 5;
        const double* x = q.data() + std::size_t(j) * F;
        auto alphaOf = [&](int u, double d) {
            std::vector<double> dq(std::size_t(r.line.rows() - 2) * B, 0.0);
            dq[std::size_t(j - 1) * B + u] = d;
            return stepLength(q, dq, r.line);
        };
        auto near = [](double a, double b) { return std::abs(a - b) <= 1e-14; };
        const std::string m(ransName(model));
        check(alphaOf(0, 0.0) == 1.0, m + ": a zero update is undamped");
        check(near(alphaOf(0, -0.5 * x[0]), 0.4), m + ": rho decrease bound 20%");
        check(near(alphaOf(0, 1.0 * x[0]), 0.2), m + ": rho increase bound 20%");
        check(near(alphaOf(3, 0.4 * x[4]), 0.5), m + ": E bound 20%");
        check(alphaOf(1, 1e6 * x[0]) == 1.0 && alphaOf(2, -1e6) == 1.0,
              m + ": momentum carries no bound");
        if (model == RansModel::SaNeg) {
            check(alphaOf(4, -1e3 * std::abs(x[5]) - 1.0) == 1.0,
                  "SA-neg: nu~ has no bound (a negative nu~ is legal)");
            continue;
        }
        check(near(alphaOf(4, -2.0 * x[5]), 0.45), m + ": rho k decrease bound 90%");
        check(alphaOf(4, 100.0 * x[5]) == 1.0, m + ": rho k increase unbounded");
        check(near(alphaOf(5, -1.0 * x[6]), 0.5), m + ": rho omega decrease bound 50%");
        check(alphaOf(5, 100.0 * x[6]) == 1.0, m + ": rho omega increase unbounded");
    }
}

} // namespace

int main()
{
    ramp();
    hold();
    reject();
    neverArmedBelowCap();
    stepBounds();
    return dnstest::finish();
}
