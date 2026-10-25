// Shared helpers for the implicit steady RANS tests (host and GPU): the
// channel line of a Channel1dCase, its IC and oracle columns, the integrals
// the stop monitors, and the host Newton driver -- the runner's implicit loop
// with the line operator as BOTH residual and Jacobian.
#pragma once

#include "dns/channel_line.hpp"
#include "dns/config.hpp"
#include "dns/curvmesh.hpp"
#include "dns/pseudo_time.hpp"
#include "dns/rans_channel1d.hpp"
#include "dns/steady_stop.hpp"
#include "dns/transport.hpp"
#include "dns/walldist.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace linetest {

using namespace dns;

struct LineRig
{
    Channel1dCase cc;
    Grid g;
    CurvMetrics m;
    std::vector<double> d, dy; // per cell; per grid row (vol / S)
    double mu, Twall, omegaCoef;
    ChannelLine line;
};

// The channel of `cc` on an nx x (ny + 2) x nz grid (dns channel's box by
// default) and the line of its column (0, 0). Pre: cc valid.
inline LineRig makeLine(const Channel1dCase& cc, int nx = 4, int nz = 1,
                        double lx = 4 * std::numbers::pi,
                        double lz = 4 * std::numbers::pi / 3)
{
    const Grid g = channel1dGrid(cc, nx, nz, lx, lz);
    const CurvMesh mesh = CurvMesh::channelY(g, cc.beta, 1.0);
    CurvMetrics m = *CurvMetrics::build(mesh);
    std::vector<double> d = channelWallDistance(mesh, 1.0), dy;
    for (int j = 0; j < g.ny; ++j) {
        const std::size_t q = g.idx(0, j, 0);
        dy.push_back(m.vol[q] / m.S[1][1][q]);
    }
    const double mu = 1.0 / cc.reTau, Tw = channelPRef(cc.reTau, cc.mach);
    const double d1 = d[g.idx(0, 1, 0)];
    const double oc = ransSst(cc.model) ? sst::omegaWallFactor * mu / (d1 * d1)
                                        : 0.0;
    ChannelLine line = ChannelLine::make(
        m, g, d, {cc.model, kGammaAir, kRgasBox, mu, Tw, 1.0, oc});
    return {cc, g, std::move(m), std::move(d), std::move(dy), mu, Tw, oc,
            std::move(line)};
}

// dns channel --rans's IC column (flow_ic.cpp channelRansIC): Reichardt u,
// rho = 1, T = Twall, the model's starting scalars; ghosts by the wall.
inline std::vector<double> icColumn(const LineRig& r)
{
    const int F = r.line.fields();
    std::vector<double> q(std::size_t(r.g.ny) * F, 0.0);
    for (int j = 1; j <= r.g.ny - 2; ++j) {
        const double dw = r.d[r.g.idx(0, j, 0)];
        const double u = reichardtUplus(dw * r.cc.reTau);
        double* x = q.data() + std::size_t(j) * F;
        x[0] = 1.0;
        x[1] = u;
        x[4] = r.Twall / (kGammaAir - 1.0) + 0.5 * u * u;
        if (!ransSst(r.cc.model)) {
            x[5] = saInitialNut(dw, 1.0, 1.0);
            continue;
        }
        const KOmega kw = sstInitialKOmega(dw, r.mu, 1.0);
        x[5] = kw.k;
        x[6] = kw.omega;
    }
    r.line.wall(q);
    return q;
}

// The oracle's discrete solution as a line column (toCompState / toTurb).
inline std::vector<double> oracleColumn(const LineRig& r,
                                        const Channel1dSolution& s)
{
    const int F = r.line.fields();
    const CompState h = toCompState(s, r.g);
    std::vector<double> t[2];
    for (int i = 0; i < ransScalars(s.model); ++i) t[i] = toTurb(s, r.g, i);
    std::vector<double> q(std::size_t(r.g.ny) * F, 0.0);
    for (int j = 0; j < r.g.ny; ++j) {
        const std::size_t c = r.g.idx(0, j, 0);
        double* x = q.data() + std::size_t(j) * F;
        x[0] = h.rho[c];
        x[1] = h.mx[c];
        x[2] = h.my[c];
        x[3] = h.mz[c];
        x[4] = h.E[c];
        for (int i = 0; i < ransScalars(s.model); ++i) x[5 + i] = t[i][c];
    }
    r.line.wall(q);
    return q;
}

// The integrals the stop monitors (runner_channel measure()): Favre bulk u
// and T, and the y-integrated conserved scalars.
struct Integrals
{
    double ub = 0, Tb = 0, t[2] = {0, 0};
};
inline Integrals integrals(const LineRig& r, const std::vector<double>& q)
{
    const int F = r.line.fields();
    double mass = 0, mom = 0, heat = 0;
    Integrals I;
    for (int j = 1; j <= r.g.ny - 2; ++j) {
        const double* x = q.data() + std::size_t(j) * F;
        const double rho = x[0], u = x[1] / rho, v = x[2] / rho, w = x[3] / rho;
        const double p = (kGammaAir - 1) * (x[4] - 0.5 * rho * (u * u + v * v + w * w));
        mass += r.dy[j] * rho;
        mom += r.dy[j] * x[1];
        heat += r.dy[j] * p / kRgasBox;
        for (int i = 0; i < ransScalars(r.cc.model); ++i) I.t[i] += r.dy[j] * x[5 + i];
    }
    I.ub = mom / mass;
    I.Tb = heat / mass;
    return I;
}

struct Drive
{
    int iters = 0;          // residual evaluations (the runner's stats rows)
    bool certified = false;
    int rejects = 0;
    double remaining = HUGE_VAL;
    std::vector<double> q;
    std::vector<Integrals> hist; // per iteration, of the evaluated state
    std::vector<bool> armed;     // per iteration: the stop was listening
};

// The runner's implicit loop (gpu/runner_channel.hip): the stop armed only at
// the cap after an undamped step, mass projected to the IC's, a rejection
// restores the last accepted state. residualOf(q, R) -> false when R is not
// finite; integralsOf(q) -> Integrals.
// Test knobs: alphaMax < 1 forces every step damped (a synthetic stall);
// armedOnly = false lets the stop listen to every sample (the rule the
// masterplan rejects, to show what arming prevents).
template <class Residual, class Integrate>
Drive driveWith(const LineRig& r, std::vector<double> q, double tol, int maxIt,
                Residual&& residualOf, Integrate&& integralsOf,
                double cfl0 = kImplicitCfl0, double alphaMax = 1.0,
                bool armedOnly = true)
{
    const ChannelLine& L = r.line;
    const int nS = ransSst(r.cc.model) ? 4 : 3;
    const double M0 = L.mass(q);
    PseudoTime pt(cfl0);
    SteadyStop stop(nS);
    Drive out;
    std::vector<double> R(q.size()), qPrev = q;
    while (out.iters < maxIt) {
        const bool finite = residualOf(q, R);
        ++out.iters;
        if (!finite) {
            q = qPrev;
            stop = SteadyStop(nS);
            ++out.rejects;
            if (!pt.rejected()) break;
            continue;
        }
        const Integrals I = integralsOf(q);
        out.hist.push_back(I);
        out.armed.push_back(pt.armed());
        if (pt.armed() || !armedOnly) {
            const double s[4] = {I.ub, I.Tb, I.t[0], I.t[1]};
            stop.push(std::span(s, nS));
        } else {
            stop = SteadyStop(nS);
        }
        out.remaining = stop.remaining(tol);
        if (out.remaining < tol) {
            out.certified = true;
            break;
        }
        auto dq = pseudoUpdate(L, q, R, pt.cfl());
        if (!dq) break;
        const double alpha = std::min(stepLength(q, *dq, L), alphaMax);
        std::vector<double> qn = q;
        L.advance(qn, *dq, alpha, M0);
        if (!L.admissible(qn)) {
            stop = SteadyStop(nS);
            ++out.rejects;
            if (!pt.rejected()) break;
            continue;
        }
        qPrev = std::move(q);
        q = std::move(qn);
        pt.accepted(alpha);
    }
    out.q = std::move(q);
    return out;
}

// The host driver: the line as residual AND Jacobian, host integrals.
inline Drive drive(const LineRig& r, std::vector<double> q, double tol,
                   int maxIt, double cfl0 = kImplicitCfl0,
                   double alphaMax = 1.0, bool armedOnly = true)
{
    return driveWith(
        r, std::move(q), tol, maxIt,
        [&](const std::vector<double>& x, std::vector<double>& R) {
            r.line.residual(x, R);
            return true;
        },
        [&](const std::vector<double>& x) { return integrals(r, x); }, cfl0,
        alphaMax, armedOnly);
}

} // namespace linetest
