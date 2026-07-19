#include "dns/comp.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dns {
namespace {

struct Prim
{
    std::vector<double> u[3], e, p, T;
};

struct Flux
{
    double rho, m[3], E;
};

} // namespace

CompState CompState::zeros(const Grid& g)
{
    const std::size_t n = g.size();
    return {std::vector<double>(n, 0.0), std::vector<double>(n, 0.0),
            std::vector<double>(n, 0.0), std::vector<double>(n, 0.0),
            std::vector<double>(n, 0.0)};
}

Comp Comp::make(const Grid& g, double gamma, double mu)
{
    assert(gamma > 1.0 && mu >= 0.0);
    return {g, gamma, mu, 0.72, 1.0};
}

double Comp::pressure(double rho, double mx, double my, double mz, double E) const
{
    assert(rho > 0.0);
    return (gamma - 1.0) * (E - 0.5 * (mx * mx + my * my + mz * mz) / rho);
}

void Comp::rhs(const CompState& s, CompState& out) const
{
    const std::size_t n = g.size();
    const double h[3] = {g.hx, g.hy, g.hz};
    const double cp = gamma * Rgas / (gamma - 1.0);
    const double kappa = mu * cp / Pr;

    Prim q;
    for (int c = 0; c < 3; ++c) q.u[c].resize(n);
    q.e.resize(n);
    q.p.resize(n);
    q.T.resize(n);
    for (std::size_t c = 0; c < n; ++c) {
        assert(s.rho[c] > 0.0);
        q.u[0][c] = s.mx[c] / s.rho[c];
        q.u[1][c] = s.my[c] / s.rho[c];
        q.u[2][c] = s.mz[c] / s.rho[c];
        q.e[c] = s.E[c] / s.rho[c];
        q.p[c] = pressure(s.rho[c], s.mx[c], s.my[c], s.mz[c], s.E[c]);
        q.T[c] = q.p[c] / (s.rho[c] * Rgas);
    }

    auto at = [&](const std::vector<double>& f, int i, int j, int k) {
        return f[g.idx(i, j, k)];
    };
    // Central derivative of f along axis d at cell (i,j,k).
    auto cder = [&](const std::vector<double>& f, int i, int j, int k, int d) {
        const int o[3] = {d == 0, d == 1, d == 2};
        return (at(f, i + o[0], j + o[1], k + o[2]) -
                at(f, i - o[0], j - o[1], k - o[2])) / (2.0 * h[d]);
    };

    // Total flux (convective minus viscous) through the +axis face of cell.
    auto faceFlux = [&](int i, int j, int k, int a) -> Flux {
        const std::size_t ca = g.idx(i, j, k);
        const int o[3] = {a == 0, a == 1, a == 2};
        const int i2 = i + o[0], j2 = j + o[1], k2 = k + o[2];
        const std::size_t cb = g.idx(i2, j2, k2);

        const double rm = 0.5 * (s.rho[ca] + s.rho[cb]);
        double um[3];
        for (int c = 0; c < 3; ++c) um[c] = 0.5 * (q.u[c][ca] + q.u[c][cb]);
        const double em = 0.5 * (q.e[ca] + q.e[cb]);
        const double pm = 0.5 * (q.p[ca] + q.p[cb]);
        const double un = um[a];
        Flux f{rm * un, {rm * un * um[0], rm * un * um[1], rm * un * um[2]},
               rm * un * em + pm * un};
        f.m[a] += pm;
        if (mu == 0.0) return f;

        // Face gradients: normal derivatives compact, tangential averaged.
        double du[3][3]; // du[c][d] = d u_c / d x_d at the face
        for (int c = 0; c < 3; ++c)
            for (int d = 0; d < 3; ++d)
                du[c][d] = d == a
                    ? (q.u[c][cb] - q.u[c][ca]) / h[a]
                    : 0.5 * (cder(q.u[c], i, j, k, d) + cder(q.u[c], i2, j2, k2, d));
        const double divu = du[0][0] + du[1][1] + du[2][2];
        const double dTn = (q.T[cb] - q.T[ca]) / h[a];
        double eflux = kappa * dTn;
        for (int c = 0; c < 3; ++c) {
            const double tau = mu * (du[c][a] + du[a][c]) -
                               (c == a ? 2.0 / 3.0 * mu * divu : 0.0);
            f.m[c] -= tau;
            eflux += um[c] * tau;
        }
        f.E -= eflux;
        return f;
    };

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                double drho = 0.0, dm[3] = {0, 0, 0}, dE = 0.0;
                for (int a = 0; a < 3; ++a) {
                    const int o[3] = {a == 0, a == 1, a == 2};
                    const Flux hi = faceFlux(i, j, k, a);
                    const Flux lo = faceFlux(i - o[0], j - o[1], k - o[2], a);
                    drho -= (hi.rho - lo.rho) / h[a];
                    for (int cc = 0; cc < 3; ++cc)
                        dm[cc] -= (hi.m[cc] - lo.m[cc]) / h[a];
                    dE -= (hi.E - lo.E) / h[a];
                }
                out.rho[c] = drho;
                out.mx[c] = dm[0];
                out.my[c] = dm[1];
                out.mz[c] = dm[2];
                out.E[c] = dE;
            }
}

void Comp::step(CompState& s, double dt) const
{
    const std::size_t n = g.size();
    CompState k = CompState::zeros(g);
    CompState q1 = CompState::zeros(g), q2 = CompState::zeros(g);

    auto axpy = [n](const CompState& a, double ca, const CompState& b, double cb,
                    const CompState& r, double cr, CompState& out) {
        for (std::size_t i = 0; i < n; ++i) {
            out.rho[i] = ca * a.rho[i] + cb * b.rho[i] + cr * r.rho[i];
            out.mx[i] = ca * a.mx[i] + cb * b.mx[i] + cr * r.mx[i];
            out.my[i] = ca * a.my[i] + cb * b.my[i] + cr * r.my[i];
            out.mz[i] = ca * a.mz[i] + cb * b.mz[i] + cr * r.mz[i];
            out.E[i] = ca * a.E[i] + cb * b.E[i] + cr * r.E[i];
        }
    };

    rhs(s, k);
    axpy(s, 1.0, s, 0.0, k, dt, q1);
    rhs(q1, k);
    axpy(s, 0.75, q1, 0.25, k, 0.25 * dt, q2);
    rhs(q2, k);
    axpy(s, 1.0 / 3.0, q2, 2.0 / 3.0, k, 2.0 / 3.0 * dt, s);
}

double Comp::stableDt(const CompState& s, double cfl) const
{
    // Per-axis (|u| + c)/h; degenerate axes impose no constraint.
    double wmax = 0.0, rhomin = s.rho[0];
    for (std::size_t i = 0; i < g.size(); ++i) {
        assert(s.rho[i] > 0.0);
        const double p = pressure(s.rho[i], s.mx[i], s.my[i], s.mz[i], s.E[i]);
        assert(p > 0.0);
        const double r = s.rho[i];
        const double c = std::sqrt(gamma * p / r);
        double w = (std::abs(s.mx[i] / r) + c) / g.hx;
        w = std::max(w, (std::abs(s.my[i] / r) + c) / g.hy);
        if (g.nz > 1) w = std::max(w, (std::abs(s.mz[i] / r) + c) / g.hz);
        wmax = std::max(wmax, w);
        rhomin = std::min(rhomin, s.rho[i]);
    }
    double dt = cfl / wmax;
    if (mu > 0.0) {
        double h = std::min(g.hx, g.hy);
        if (g.nz > 1) h = std::min(h, g.hz);
        dt = std::min(dt, 0.25 * h * h * rhomin / mu);
    }
    return dt;
}

} // namespace dns
