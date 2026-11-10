#include "dns/fvnozzle.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <numbers>

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
    o.omega = l.omegaSum();
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

std::vector<std::string> fvThrustColumns(bool station)
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
    if (station)
        for (const char* s : {"p0b", "T0b", "CdB", "CfBvac", "CfB", "Mst", "Mth"}) c.emplace_back(s);
    return c;
}

FvStorage fvStorage(const FaceMesh& fm, const FvState& s, const std::array<double, 3>& axis, double station,
                    std::span<const std::uint8_t> pastThroat)
{
    assert(pastThroat.size() == std::size_t(fm.nC));
    double st = 0, stLo = 0, th = 0, thLo = 0;
    const double* rho = s.var(0);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double m = fm.V[c] * rho[c];
        if (axis[0] * fm.C[0][c] + axis[1] * fm.C[1][c] + axis[2] * fm.C[2][c] > station) neumaier(st, stLo, m);
        if (pastThroat[c]) neumaier(th, thLo, m);
    }
    return {.Mst = st + stLo, .Mth = th + thLo};
}

std::vector<double> fvThrustRow(double t, long step, double dt, const FvRow6& P, const FvRow6& P0, const FvRoleSums& now,
                                const FvThrustWindow& w, const FvExitSample& x, const FvThrustCase& k, const FvStation* b,
                                const FvStorage* m)
{
    assert((b == nullptr) == (m == nullptr));
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
    if (!b) return v;
    const double ref = b->p0 * k.astar;
    v.insert(v.end(), {b->p0, b->T0, w.mdotOut / fvIdealMassFlow(b->p0, b->T0, k.astar, k.gamma, k.Rgas), w.Tvac / ref,
                       (w.Tvac - k.pa * k.exitArea) / ref, m->Mst, m->Mth});
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

FvBurnt fvRayleighBurnt(double gamma, double Rgas, double cp, double q, double Tu, double pu, double uu)
{
    assert(Tu > 0.0 && pu > 0.0 && uu > 0.0);
    const double g = gamma, R = Rgas;
    const double rhou = pu / (R * Tu), m = rhou * uu, P = pu + m * uu, H = cp * Tu + uu * uu / 2 + q;
    const auto f = [&](double ub) { return cp * (P - m * ub) * ub / (m * R) + ub * ub / 2 - H; };
    double lo = uu, hi = (P / m) * g / (g + 1);
    for (int i = 0; i < 200; ++i) {
        const double mid = (lo + hi) / 2;
        if (f(mid) < 0) lo = mid;
        else hi = mid;
    }
    FvBurnt b;
    b.ub = (lo + hi) / 2;
    b.pb = P - m * b.ub;
    b.Tb = b.pb * b.ub / (m * R);
    b.Mb = b.ub / std::sqrt(g * R * b.Tb);
    b.T0b = b.Tb * (1 + (g - 1) / 2 * b.Mb * b.Mb);
    b.p0b = b.pb * std::pow(b.T0b / b.Tb, g / (g - 1));
    return b;
}

double fvAreaMach(double ar, double gamma, bool sup)
{
    const double g = gamma;
    const auto A = [&](double M) { return (1 / M) * std::pow((2 / (g + 1)) * (1 + (g - 1) / 2 * M * M), (g + 1) / (2 * (g - 1))); };
    if (ar <= 1.0) return 1.0;
    double lo = sup ? 1.0 : 1e-6, hi = sup ? 60.0 : 1.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = (lo + hi) / 2;
        // A(M) falls below M 1 and rises above it.
        if ((A(mid) < ar) == sup) lo = mid;
        else hi = mid;
    }
    return (lo + hi) / 2;
}

std::expected<FvReactingDesign, std::string> fvReactingNozzleDesign(double gamma, double Rgas, double cp, double q,
                                                                    double sigma, double epsC, double epsE, double p0u,
                                                                    double T0u)
{
    const double g = gamma, R = Rgas;
    if (!(epsE >= 1.0)) return std::unexpected(std::format("reacting nozzle: exit area ratio {} < 1", epsE));
    if (!(epsC > 1.0)) return std::unexpected(std::format("reacting nozzle: chamber area / A* = {:.6g}: no throat", epsC));
    FvReactingDesign d;
    d.g = g;
    d.R = R;
    d.cp = cp;
    d.q = q;
    d.Gamma = std::sqrt(g) * std::pow(2 / (g + 1), (g + 1) / (2 * (g - 1)));
    const auto at = [&](double Mu) {
        FvReactingDesign o = d;
        o.Mu = Mu;
        o.Tu = T0u / (1 + (g - 1) / 2 * Mu * Mu);
        o.pu = p0u * std::pow(o.Tu / T0u, g / (g - 1));
        o.rhou = o.pu / (R * o.Tu);
        o.uu = Mu * std::sqrt(g * R * o.Tu);
        o.m = o.rhou * o.uu;
        o.b = fvRayleighBurnt(g, R, cp, q, o.Tu, o.pu, o.uu);
        return o;
    };
    const auto excess = [&](const FvReactingDesign& o) { return o.m * epsC - d.Gamma * o.b.p0b / std::sqrt(R * o.b.T0b); };
    double lo = 1e-6, hi = 0.3;
    if (!(excess(at(lo)) < 0.0 && excess(at(hi)) >= 0.0))
        return std::unexpected(std::format("reacting nozzle: the throat does not choke the chamber for M_u in [1e-6, 0.3] "
                                           "(chamber area / A* = {:.6g})",
                                           epsC));
    for (int i = 0; i < 200; ++i) {
        const double Mu = (lo + hi) / 2;
        if (excess(at(Mu)) < 0.0) lo = Mu;
        else hi = Mu;
    }
    d = at((lo + hi) / 2);
    if (!(d.b.Mb < 1.0))
        return std::unexpected(std::format("reacting nozzle: the flame chokes the chamber thermally (M_b {:.6g})", d.b.Mb));
    d.Me = fvAreaMach(epsE, g, true);
    d.peOverP0b = std::pow(1 + (g - 1) / 2 * d.Me * d.Me, -g / (g - 1));
    d.CFvac = d.Gamma * std::sqrt(2 * g / (g - 1) * (1 - std::pow(d.peOverP0b, (g - 1) / g))) + d.peOverP0b * epsE;
    d.MuLowM = d.Gamma / (std::sqrt(g * sigma) * epsC);
    return d;
}

FvState fvReactingNozzleIC(const FaceMesh& fm, const FvFlameProbe& probe, const FvReactingDesign& d,
                           const std::function<double(double)>& t, double xf, double astar, const FvNozzleShape* shape)
{
    const FvFlameSpec& sp = probe.spec();
    assert(sp.chamber && astar > 0.0);
    assert(!shape || shape->Hc > 0.0);
    const int nCh = probe.chamberBins();
    const double g = sp.gamma, R = sp.Rgas;
    const std::vector<double> A = fvBinAreas(fm, probe);
    const int throat = fvThroatBin(A, nCh);
    // combscale S5: dA_b/dx at the bin centres (np.gradient over uniform bins: central inside, one-sided at the ends).
    std::vector<double> dA;
    if (shape) {
        const std::size_t n = A.size();
        const double h = probe.slab();
        dA.assign(n, 0.0);
        for (std::size_t b = 0; b < n && n >= 2; ++b)
            dA[b] = b == 0 ? (A[1] - A[0]) / h : b + 1 == n ? (A[n - 1] - A[n - 2]) / h : (A[b + 1] - A[b - 1]) / (2.0 * h);
    }
    // r = (C - origin).lift of cell i (shape only).
    const auto rOf = [&](std::size_t i) {
        return (fm.C[0][i] - shape->origin[0]) * shape->lift[0] + (fm.C[1][i] - shape->origin[1]) * shape->lift[1] +
               (fm.C[2][i] - shape->origin[2]) * shape->lift[2];
    };
    FvState q = FvState::zeros(fm.nC, kFvMeanVars + 1);
    for (int c = 0; c < fm.nC; ++c) {
        const std::size_t i = std::size_t(c);
        const int b = probe.binOf(c);
        double rho = 0, u = 0, E = 0, Y = 0, v = 0;
        if (b < nCh) {
            // The Rayleigh line of the design's (m, P, H) at the profile's Y (masterplan_combclose 2.1): never
            // premixedPoint's low-Mach T = sigma Tu (+1.71e-3 in T0, an entropy slug through the throat).
            const double s = fm.C[0][i] * sp.dir[0] + fm.C[1][i] * sp.dir[1] + fm.C[2][i] * sp.dir[2] - probe.lo();
            // combscale S5: the seeded front's lowest transverse mode (seed 0: s - xf - 0, the planar front's bits).
            const double tt = shape ? t(s - xf - shape->seed * std::cos(std::numbers::pi * rOf(i) / shape->Hc)) : t(s - xf);
            Y = (sp.sigma - tt) / (sp.sigma - 1);
            const FvBurnt r = fvRayleighBurnt(d.g, d.R, d.cp, d.q * (1 - Y), d.Tu, d.pu, d.uu);
            rho = r.pb / (R * r.Tb);
            u = r.ub;
            E = r.pb / (g - 1) + 0.5 * rho * u * u;
        } else {
            const double M = fvAreaMach(A[std::size_t(b)] / astar, g, b > throat);
            const double T = d.b.T0b / (1 + (g - 1) / 2 * M * M), p = d.b.p0b * std::pow(T / d.b.T0b, g / (g - 1));
            rho = p / (R * T);
            u = M * std::sqrt(g * R * T);
            E = p / (g - 1) + 0.5 * rho * u * u;
            // combscale S5: the generator's 2D v = u y_w' r / y_w (planar A proportional to y_w).
            if (shape) {
                v = u * rOf(i) * dA[std::size_t(b)] / A[std::size_t(b)];
                E += 0.5 * rho * v * v;
            }
        }
        q.var(0)[i] = rho;
        if (shape)
            for (int k = 0; k < 3; ++k)
                q.var(1 + k)[i] = rho * (u * sp.dir[std::size_t(k)] + v * shape->lift[std::size_t(k)]);
        else
            for (int k = 0; k < 3; ++k) q.var(1 + k)[i] = rho * u * sp.dir[std::size_t(k)];
        q.var(4)[i] = E;
        q.var(kFvMeanVars + sp.fuel)[i] = rho * Y;
    }
    return q;
}

std::optional<FvNozzleShape> fvNozzleShapeOf(const FaceMesh& fm, const FvFlameProbe& probe, std::span<const FvRole> roles,
                                             const std::array<double, 3>& origin, const std::array<double, 3>& lift)
{
    assert(probe.spec().chamber && roles.size() == fm.patches.size());
    const int nCh = probe.chamberBins();
    std::vector<int> per(std::size_t(nCh), 0);
    for (int c = 0; c < fm.nC; ++c)
        if (const int b = probe.binOf(c); b < nCh) ++per[std::size_t(b)];
    if (std::ranges::none_of(per, [](int n) { return n >= 2; })) return std::nullopt;
    FvNozzleShape s{.origin = origin, .lift = lift};
    for (std::size_t p = 0; p < fm.patches.size(); ++p) {
        if (roles[p] != FvRole::Wall) continue;
        const MeshPatch& m = fm.patches[p];
        for (int q = m.first; q < m.first + m.count; ++q) {
            const std::size_t f = std::size_t(fm.nI + q);
            if (probe.binOf(fm.own[f]) >= nCh) continue;
            const double r = (fm.Cf[0][f] - origin[0]) * lift[0] + (fm.Cf[1][f] - origin[1]) * lift[1] +
                             (fm.Cf[2][f] - origin[2]) * lift[2];
            s.Hc = std::max(s.Hc, r);
        }
    }
    if (!(s.Hc > 0.0)) return std::nullopt;
    return s;
}

std::array<double, 2> fvFrontAmplitudes(const FaceMesh& fm, const FvFlameProbe& probe, const FvNozzleShape& shape,
                                        const FvState& s)
{
    assert(shape.Hc > 0.0 && probe.spec().chamber);
    const FvFlameSpec& sp = probe.spec();
    const int nCh = probe.chamberBins();
    const double* rho = s.var(0);
    const double* ry = s.var(kFvMeanVars + sp.fuel);
    // The probe's reference slab (FvFlameProbe::finish: 2 delta_th inside the low end, within the chamber).
    const int ref = std::min(nCh - 1, int(std::lround(2.0 * sp.dth / probe.slab())));
    double vr = 0, vrLo = 0, vol = 0, volLo = 0;
    double a[2] = {0, 0}, lo[2] = {0, 0};
    for (int c = 0; c < fm.nC; ++c) {
        const std::size_t i = std::size_t(c);
        const int b = probe.binOf(c);
        if (b >= nCh) continue;
        const double V = fm.V[i], ryc = ry[i];
        if (b == ref) {
            neumaier(vr, vrLo, V * (ryc - sp.yBurnt * rho[i]));
            neumaier(vol, volLo, V);
        }
        const double r = (fm.C[0][i] - shape.origin[0]) * shape.lift[0] + (fm.C[1][i] - shape.origin[1]) * shape.lift[1] +
                         (fm.C[2][i] - shape.origin[2]) * shape.lift[2];
        for (int n = 0; n < 2; ++n) neumaier(a[n], lo[n], V * ryc * std::cos(double(n + 1) * std::numbers::pi * r / shape.Hc));
    }
    const double ryRef = (vr + vrLo) / (vol + volLo), den = ryRef * probe.area();
    return {2.0 * (a[0] + lo[0]) / den, 2.0 * (a[1] + lo[1]) / den};
}

std::vector<double> fvBinAreas(const FaceMesh& fm, const FvFlameProbe& probe)
{
    const double h = probe.slab();
    std::vector<double> A(std::size_t(probe.bins()), 0.0);
    for (int c = 0; c < fm.nC; ++c) A[std::size_t(probe.binOf(c))] += fm.V[std::size_t(c)] / h;
    return A;
}

int fvThroatBin(std::span<const double> A, int nCh)
{
    assert(nCh >= 0 && std::size_t(nCh) < A.size());
    int throat = nCh;
    for (int b = nCh; b < int(A.size()); ++b)
        if (A[std::size_t(b)] > 0.0 && (A[std::size_t(throat)] <= 0.0 || A[std::size_t(b)] < A[std::size_t(throat)])) throat = b;
    return throat;
}

std::vector<std::uint8_t> fvPastThroat(const FaceMesh& fm, const FvFlameProbe& probe)
{
    assert(probe.spec().chamber);
    const int throat = fvThroatBin(fvBinAreas(fm, probe), probe.chamberBins());
    std::vector<std::uint8_t> past(std::size_t(fm.nC), 0);
    for (int c = 0; c < fm.nC; ++c) past[std::size_t(c)] = probe.binOf(c) > throat ? 1 : 0;
    return past;
}

} // namespace dns
