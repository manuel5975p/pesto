#include "dns/fvnozzle.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dns {

FvState fvRestState(const FaceMesh& fm, double p, double T, double gamma, double Rgas)
{
    assert(p > 0.0 && T > 0.0);
    FvState s = FvState::zeros(fm.nC);
    const double rho = p / (Rgas * T), E = p / (gamma - 1.0);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        s.var(0)[c] = rho;
        s.var(4)[c] = E;
    }
    return s;
}

double fvIdealMassFlow(double p0, double T0, double astar, double gamma, double Rgas)
{
    const double e = (gamma + 1.0) / (2.0 * (gamma - 1.0));
    return astar * p0 * std::sqrt(gamma / (Rgas * T0)) * std::pow(2.0 / (gamma + 1.0), e);
}

FvNozzleRow fvNozzleRow(std::span<const FvBc> bc, std::span<const double> mass, const std::array<std::vector<double>, 3>& mom,
                        const std::array<double, 3>& fw, const FvNozzleRun& run, double gamma, double Rgas)
{
    assert(run.astar > 0.0 && bc.size() == mass.size());
    const auto& a = run.axis;
    FvNozzleRow r;
    for (std::size_t p = 0; p < bc.size(); ++p) {
        const double along = mom[0][p] * a[0] + mom[1][p] * a[1] + mom[2][p] * a[2];
        if (bc[p].kind == FvBcKind::Total) {
            r.mdotIn -= mass[p];
            r.Fin += along;
        }
        if (bc[p].kind == FvBcKind::Outflow) {
            r.mdotOut += mass[p];
            r.Fexit += along;
        }
    }
    r.Fwall = fw[0] * a[0] + fw[1] * a[1] + fw[2] * a[2];
    r.Cd = r.mdotOut / fvIdealMassFlow(run.p0, run.t0, run.astar, gamma, Rgas);
    r.Cf = r.Fexit / (run.p0 * run.astar);
    return r;
}

namespace {

double dot3(const std::array<double, 3>& a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The role of boundary face b (FaceMesh boundary index).
FvRole roleOfFace(const FaceMesh& fm, std::span<const FvRole> roles, int b)
{
    for (std::size_t p = 0; p < fm.patches.size(); ++p)
        if (b >= fm.patches[p].first && b < fm.patches[p].first + fm.patches[p].count) return roles[p];
    assert(false && "boundary face outside every patch");
    return FvRole::None;
}

// Neumaier: s += v with the lost low part kept in c.
void neumaier(double& s, double& c, double v)
{
    const double t = s + v;
    c += std::abs(s) >= std::abs(v) ? (s - t) + v : (v - t) + s;
    s = t;
}

} // namespace

std::vector<FvRole> fvRoles(std::span<const FvBc> bc, const FaceMesh& fm)
{
    assert(bc.size() >= fm.patches.size());
    std::vector<FvRole> r(fm.patches.size(), FvRole::None);
    for (std::size_t p = 0; p < fm.patches.size(); ++p) {
        const FvBc& b = bc[p];
        const FoamPatchType ty = fm.patches[p].type;
        switch (b.kind) {
        case FvBcKind::Total:
        case FvBcKind::Inflow:
        case FvBcKind::Far: r[p] = FvRole::In; break;
        case FvBcKind::Outflow: r[p] = FvRole::Exit; break;
        case FvBcKind::Slip:
            r[p] = b.wedge || ty == FoamPatchType::SymmetryPlane || ty == FoamPatchType::Symmetry ? FvRole::Side : FvRole::Wall;
            break;
        case FvBcKind::Wall: r[p] = FvRole::Wall; break;
        case FvBcKind::Empty: r[p] = FvRole::None; break;
        }
    }
    return r;
}

FvRoleSums fvRoleSums(const FvLedger& l, std::span<const FvRole> roles, int fuelRow)
{
    assert(std::size_t(l.nP) == roles.size());
    FvRoleSums o;
    o.omega = l.omega;
    for (int p = 0; p < l.nP; ++p) {
        const FvRole r = roles[std::size_t(p)];
        if (r == FvRole::None) continue;
        FvRow6& d = o.I[std::size_t(r)];
        for (int v = 0; v < 5; ++v) d[std::size_t(v)] += l.at(p, v);
        if (fuelRow >= 0) d[5] += l.at(p, fuelRow);
    }
    return o;
}

FvRow6 fvTotals6(const FaceMesh& fm, const FvState& s, int fuelRow)
{
    FvRow6 sum{}, lo{};
    const int rows[6] = {0, 1, 2, 3, 4, fuelRow};
    for (std::size_t k = 0; k < 6; ++k) {
        if (rows[k] < 0) continue;
        const double* q = s.var(rows[k]);
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) neumaier(sum[k], lo[k], fm.V[c] * q[c]);
        sum[k] += lo[k];
    }
    return sum;
}

double fvExitArea(const FaceMesh& fm, std::span<const FvRole> roles, const std::array<double, 3>& axis)
{
    double a = 0.0;
    for (int b = 0; b < fm.nB; ++b) {
        if (roleOfFace(fm, roles, b) != FvRole::Exit) continue;
        const std::size_t f = std::size_t(fm.nI + b);
        const double S[3] = {fm.S[0][f], fm.S[1][f], fm.S[2][f]};
        a += dot3(axis, S);
    }
    return a;
}

FvExitSample fvExitSample(const FaceMesh& fm, std::span<const FvRole> roles, const FvEval& ev, const FvState& rhs,
                          const FvThrustCase& k)
{
    FvExitSample x;
    double rest = 0.0, area = 0.0, superArea = 0.0;
    for (int b = 0; b < fm.nB; ++b) {
        const FvRole r = roleOfFace(fm, roles, b);
        if (r == FvRole::None) continue;
        const std::size_t f = std::size_t(fm.nI + b);
        const double S[3] = {fm.S[0][f], fm.S[1][f], fm.S[2][f]};
        const double Fm[3] = {ev.F[1][f], ev.F[2][f], ev.F[3][f]};
        const double Sa = dot3(k.axis, S), along = dot3(k.axis, Fm) - k.pa * Sa;
        if (r != FvRole::Exit) {
            rest += along;
            continue;
        }
        x.Texit += along;
        const FvPrim<double>& B = ev.bnd[std::size_t(b)];
        x.Fmom += ev.F[0][f] * dot3(k.axis, B.u);
        x.Fpres += (B.p - k.pa) * Sa;
        const double mS = std::hypot(S[0], S[1], S[2]);
        const double un = (B.u[0] * S[0] + B.u[1] * S[1] + B.u[2] * S[2]) / mS;
        area += mS;
        if (un >= std::sqrt(k.gamma * B.p / B.rho)) superArea += mS;
    }
    double dP[3] = {0.0, 0.0, 0.0};
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c)
        for (int v = 0; v < 3; ++v) dP[v] += fm.V[c] * rhs.var(1 + v)[c];
    x.Twall = -rest - dot3(k.axis, dP);
    x.exitSuper = area > 0.0 ? superArea / area : 0.0;
    return x;
}

FvThrustWindow fvThrustWindow(const FvRoleSums& a, double t1, const FvRoleSums& b, double t2, const FvThrustCase& k)
{
    assert(t2 > t1 && k.p0 > 0.0 && k.astar > 0.0);
    const double dt = t2 - t1;
    const auto& ea = a.I[std::size_t(FvRole::Exit)];
    const auto& eb = b.I[std::size_t(FvRole::Exit)];
    const auto& ia = a.I[std::size_t(FvRole::In)];
    const auto& ib = b.I[std::size_t(FvRole::In)];
    FvThrustWindow w;
    w.mdotOut = (eb[0] - ea[0]) / dt;
    w.mdotIn = -(ib[0] - ia[0]) / dt;
    const double dI[3] = {eb[1] - ea[1], eb[2] - ea[2], eb[3] - ea[3]};
    w.Tvac = dot3(k.axis, dI) / dt;
    const double ref = k.p0 * k.astar;
    w.CfVac = w.Tvac / ref;
    w.Cf = (w.Tvac - k.pa * k.exitArea) / ref;
    w.cstar = ref / w.mdotOut;
    return w;
}

std::vector<std::string> fvThrustColumns()
{
    static const char* const rows[6] = {"mass", "mx", "my", "mz", "E", "ry"};
    static const char* const roles[kFvRoles] = {"in", "exit", "wall", "side"};
    std::vector<std::string> c = {"t", "step", "dt"};
    for (const char* r : rows) c.push_back(std::string("P_") + r);
    for (const char* o : roles)
        for (const char* r : rows) c.push_back(std::string("I_") + o + "_" + r);
    c.emplace_back("Omega");
    for (const char* r : rows) c.push_back(std::string("res_") + r);
    for (const char* s : {"Fmom", "Fpres", "Texit", "Twall", "exitSuper", "mdotIn", "mdotOut", "CfVac", "Cf", "cstar"})
        c.emplace_back(s);
    return c;
}

std::vector<double> fvThrustRow(double t, long step, double dt, const FvRow6& P, const FvRow6& P0, const FvRoleSums& now,
                                const FvThrustWindow& w, const FvExitSample& x, const FvThrustCase& k)
{
    std::vector<double> v = {t, double(step), dt};
    v.insert(v.end(), P.begin(), P.end());
    for (const FvRow6& r : now.I) v.insert(v.end(), r.begin(), r.end());
    v.push_back(now.omega);
    for (std::size_t r = 0; r < 6; ++r) {
        double s = P[r] - P0[r];
        for (const FvRow6& o : now.I) s += o[r];
        if (r == 4) s -= k.q * now.omega;
        if (r == 5) s += now.omega;
        v.push_back(s);
    }
    v.insert(v.end(), {x.Fmom, x.Fpres, x.Texit, x.Twall, x.exitSuper, w.mdotIn, w.mdotOut, w.CfVac, w.Cf, w.cstar});
    return v;
}

double fvExitMach(double f1, double f2, double f3, double gamma, bool supersonic)
{
    assert(f1 > 0.0);
    // p = f2 - f1 u, rho = f1/u, h = g/(g-1) p/rho, h0 = f3/f1 = h + u^2/2: a u^2 + b u + c = 0.
    const double gg = gamma / (gamma - 1.0), h0 = f3 / f1;
    const double a = 0.5 - gg, b = gg * f2 / f1, c = -h0;
    const double disc = b * b - 4.0 * a * c;
    if (!(disc >= 0.0)) return NAN;
    const double r1 = (-b + std::sqrt(disc)) / (2.0 * a), r2 = (-b - std::sqrt(disc)) / (2.0 * a);
    auto mach = [&](double u) {
        const double p = f2 - f1 * u, rho = f1 / u;
        return u / std::sqrt(gamma * p / rho);
    };
    const double m1 = mach(r1), m2 = mach(r2);
    return supersonic ? std::max(m1, m2) : std::min(m1, m2);
}

FvStation fvStation(const FaceMesh& fm, const FvState& s, const std::array<double, 3>& axis, double station, double gamma,
                    double Rgas)
{
    FvStation o;
    double w = 0.0, wp = 0.0, wt = 0.0;
    const double cp = gamma * Rgas / (gamma - 1.0);
    for (std::size_t f = 0; f < std::size_t(fm.nI); ++f) {
        const std::size_t a = std::size_t(fm.own[f]), b = std::size_t(fm.nei[f]);
        const double sa = axis[0] * fm.C[0][a] + axis[1] * fm.C[1][a] + axis[2] * fm.C[2][a];
        const double sb = axis[0] * fm.C[0][b] + axis[1] * fm.C[1][b] + axis[2] * fm.C[2][b];
        if (!((sa < station && station <= sb) || (sb < station && station <= sa))) continue;
        const FvPrim<double> P = fvPrimOf(s.var(0)[a], s.var(1)[a], s.var(2)[a], s.var(3)[a], s.var(4)[a], gamma, Rgas);
        const FvPrim<double> N = fvPrimOf(s.var(0)[b], s.var(1)[b], s.var(2)[b], s.var(3)[b], s.var(4)[b], gamma, Rgas);
        const double rho = 0.5 * (P.rho + N.rho), p = 0.5 * (P.p + N.p), T = 0.5 * (P.temp + N.temp);
        const double u[3] = {0.5 * (P.u[0] + N.u[0]), 0.5 * (P.u[1] + N.u[1]), 0.5 * (P.u[2] + N.u[2])};
        const double S[3] = {fm.S[0][f], fm.S[1][f], fm.S[2][f]};
        const double m = std::abs(rho * (u[0] * S[0] + u[1] * S[1] + u[2] * S[2]));
        const double T0 = T + (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) / (2.0 * cp);
        w += m;
        wt += m * T0;
        wp += m * p * std::pow(T0 / T, gamma / (gamma - 1.0));
        ++o.faces;
    }
    o.mdot = w;
    o.T0 = o.faces && w > 0.0 ? wt / w : NAN;
    o.p0 = o.faces && w > 0.0 ? wp / w : NAN;
    return o;
}

} // namespace dns
