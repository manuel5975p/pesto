#include "dns/comp.hpp"

#include "dns/curvmesh.hpp"

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

Prim primitives(const Comp& c, const CompState& s)
{
    const std::size_t n = c.g.size();
    Prim q;
    for (int d = 0; d < 3; ++d) q.u[d].resize(n);
    q.e.resize(n);
    q.p.resize(n);
    q.T.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        // rho > 0 is stableDt()'s checked precondition (see comp.hpp).
        q.u[0][i] = s.mx[i] / s.rho[i];
        q.u[1][i] = s.my[i] / s.rho[i];
        q.u[2][i] = s.mz[i] / s.rho[i];
        q.e[i] = s.E[i] / s.rho[i];
        q.p[i] = c.pressure(s.rho[i], s.mx[i], s.my[i], s.mz[i], s.E[i]);
        q.T[i] = q.p[i] / (s.rho[i] * c.tr.Rgas);
    }
    return q;
}

// Finite-volume curvilinear rhs: Kennedy-Gruber fluxes with the face area
// vector S in place of the axis normal (Un = u_avg . S), viscous gradients
// via the face inverse Jacobian, divergence scaled by the cell volume.
void rhsCurv(const Comp& c, const CurvMetrics& m, const CompState& s,
             CompState& out)
{
    const Grid& g = c.g;
    const Prim q = primitives(c, s);

    // Central index-space derivative (dxi = 1) of f along axis d.
    auto ider = [&](const std::vector<double>& f, int i, int j, int k, int d) {
        const int o[3] = {d == 0, d == 1, d == 2};
        return 0.5 * (f[g.idx(i + o[0], j + o[1], k + o[2])] -
                      f[g.idx(i - o[0], j - o[1], k - o[2])]);
    };

    // Total flux (convective minus viscous) through the +a face of cell.
    auto faceFlux = [&](int i, int j, int k, int a) -> Flux {
        const std::size_t ca = g.idx(i, j, k);
        const int o[3] = {a == 0, a == 1, a == 2};
        const int i2 = i + o[0], j2 = j + o[1], k2 = k + o[2];
        const std::size_t cb = g.idx(i2, j2, k2);
        const double Sv[3] = {m.S[a][0][ca], m.S[a][1][ca], m.S[a][2][ca]};

        const double rm = 0.5 * (s.rho[ca] + s.rho[cb]);
        double um[3];
        for (int d = 0; d < 3; ++d) um[d] = 0.5 * (q.u[d][ca] + q.u[d][cb]);
        const double em = 0.5 * (q.e[ca] + q.e[cb]);
        const double pm = 0.5 * (q.p[ca] + q.p[cb]);
        const double Un = um[0] * Sv[0] + um[1] * Sv[1] + um[2] * Sv[2];
        Flux f{rm * Un,
               {rm * Un * um[0] + pm * Sv[0], rm * Un * um[1] + pm * Sv[1],
                rm * Un * um[2] + pm * Sv[2]},
               rm * Un * em + pm * Un};
        if (c.tr.muRef == 0.0) return f;
        // One face temperature drives both coefficients (see the device twin
        // in gpu/internal/comp_flux.hpp).
        const double Tf = 0.5 * (q.T[ca] + q.T[cb]);
        const double mu = c.tr.mu(Tf), kappa = c.tr.kappa(Tf);

        // d{u,v,w,T}/dxi at the face: normal compact, tangential averaged.
        double dxi[4][3];
        for (int f4 = 0; f4 < 4; ++f4) {
            const std::vector<double>& fld = f4 < 3 ? q.u[f4] : q.T;
            for (int d = 0; d < 3; ++d)
                dxi[f4][d] = d == a
                    ? fld[cb] - fld[ca]
                    : 0.5 * (ider(fld, i, j, k, d) + ider(fld, i2, j2, k2, d));
        }
        double gr[4][3]; // d{u,v,w,T}/dx_c
        for (int f4 = 0; f4 < 4; ++f4)
            for (int cx = 0; cx < 3; ++cx) {
                double sum = 0.0;
                for (int d = 0; d < 3; ++d)
                    sum += dxi[f4][d] * m.Ji[a][d * 3 + cx][ca];
                gr[f4][cx] = sum;
            }
        const double divu = gr[0][0] + gr[1][1] + gr[2][2];
        double eflux =
            kappa * (gr[3][0] * Sv[0] + gr[3][1] * Sv[1] + gr[3][2] * Sv[2]);
        for (int cx = 0; cx < 3; ++cx) {
            double ts = 0.0; // (tau . S)_cx
            for (int d = 0; d < 3; ++d)
                ts += (mu * (gr[cx][d] + gr[d][cx]) -
                       (cx == d ? 2.0 / 3.0 * mu * divu : 0.0)) * Sv[d];
            f.m[cx] -= ts;
            eflux += um[cx] * ts;
        }
        f.E -= eflux;
        return f;
    };

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t cc = g.idx(i, j, k);
                double drho = 0.0, dm[3] = {0, 0, 0}, dE = 0.0;
                for (int a = 0; a < 3; ++a) {
                    const int o[3] = {a == 0, a == 1, a == 2};
                    const Flux hi = faceFlux(i, j, k, a);
                    const Flux lo = faceFlux(i - o[0], j - o[1], k - o[2], a);
                    drho -= hi.rho - lo.rho;
                    for (int d = 0; d < 3; ++d) dm[d] -= hi.m[d] - lo.m[d];
                    dE -= hi.E - lo.E;
                }
                const double iv = 1.0 / m.vol[cc];
                out.rho[cc] = drho * iv;
                out.mx[cc] = dm[0] * iv;
                out.my[cc] = dm[1] * iv;
                out.mz[cc] = dm[2] * iv;
                out.E[cc] = dE * iv;
            }
}

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
    return {g, Transport::constant(gamma, mu), nullptr};
}

std::expected<Comp, std::string> Comp::make(const CurvMesh& mesh, double gamma,
                                            double mu)
{
    assert(gamma > 1.0 && mu >= 0.0);
    auto m = CurvMetrics::build(mesh);
    if (!m) return std::unexpected(std::move(m.error()));
    Comp c = make(mesh.g, gamma, mu);
    c.curv = std::make_shared<const CurvMetrics>(std::move(*m));
    return c;
}

// Precondition rho > 0, CHECKED by stableDt() before every step (an assert
// here would be an input guard that vanishes in Release, and this is called
// once per cell in the inner loops).
double Comp::pressure(double rho, double mx, double my, double mz, double E) const
{
    return (tr.gamma - 1.0) * (E - 0.5 * (mx * mx + my * my + mz * mz) / rho);
}

void Comp::rhs(const CompState& s, CompState& out) const
{
    if (curv) return rhsCurv(*this, *curv, s, out);

    const double h[3] = {g.hx, g.hy, g.hz};
    const Prim q = primitives(*this, s);

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
        if (tr.muRef == 0.0) return f;
        const double Tf = 0.5 * (q.T[ca] + q.T[cb]);
        const double mu = tr.mu(Tf), kappa = tr.kappa(Tf);

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

std::expected<double, std::string> Comp::stableDt(const CompState& s,
                                                  double cfl) const
{
    // Per-axis (|u| + c)/h; degenerate axes impose no constraint. On a
    // curvilinear mesh the width is 1/aov and |u| bounds the face-normal
    // speed, so the curvilinear dt is conservative by construction.
    // rho <= 0 or p <= 0 means the field has already blown up. A USER can
    // cause that (any diverging run), so it is reported, not asserted: in
    // Release the assert vanished and sqrt(gamma*p/r) with p < 0 produced a
    // NaN dt, which the time loop then never recovered from -- silent
    // non-termination instead of exit 7.
    double wmax = 0.0, rhomin = s.rho[0], tmax = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        const double r = s.rho[i];
        if (!(r > 0.0) || !std::isfinite(r))
            return std::unexpected("compressible state diverged: density is "
                                   + std::to_string(r) + " at cell "
                                   + std::to_string(i));
        const double p = pressure(r, s.mx[i], s.my[i], s.mz[i], s.E[i]);
        if (!(p > 0.0) || !std::isfinite(p))
            return std::unexpected("compressible state diverged: pressure is "
                                   + std::to_string(p) + " at cell "
                                   + std::to_string(i));
        const double c = std::sqrt(tr.gamma * p / r);
        double w = 0.0;
        if (curv) {
            const double sp = std::sqrt(s.mx[i] * s.mx[i] + s.my[i] * s.my[i] +
                                        s.mz[i] * s.mz[i]) / r + c;
            w = sp * std::max(curv->aov[0][i], curv->aov[1][i]);
            if (g.nz > 1) w = std::max(w, sp * curv->aov[2][i]);
        } else {
            w = (std::abs(s.mx[i] / r) + c) / g.hx;
            w = std::max(w, (std::abs(s.my[i] / r) + c) / g.hy);
            if (g.nz > 1) w = std::max(w, (std::abs(s.mz[i] / r) + c) / g.hz);
        }
        wmax = std::max(wmax, w);
        rhomin = std::min(rhomin, s.rho[i]);
        tmax = std::max(tmax, p / (r * tr.Rgas));
    }
    double dt = cfl / wmax;
    if (tr.muRef > 0.0) {
        double h;
        if (curv) {
            h = curv->hmin;
        } else {
            h = std::min(g.hx, g.hy);
            if (g.nz > 1) h = std::min(h, g.hz);
        }
        // Thermal diffusivity gamma nu / Pr binds before the momentum one for
        // air; mu(T) is monotone, so its field max is mu(Tmax). Both mirror
        // gpu/comp.hip exactly -- test_gpu_comp pins the parity.
        dt = std::min(dt, 0.25 * h * h * rhomin /
                              (tr.mu(tmax) * tr.diffusivityFactor()));
    }
    if (!(dt > 0.0) || !std::isfinite(dt))
        return std::unexpected("compressible stable dt is not positive-finite");
    return dt;
}

} // namespace dns
