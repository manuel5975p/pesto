// Host oracle of --shock (dns/shockflux.hpp, docs/shock.md): a 1D SSP-RK3 marcher
// on the header, T = float or double, in DeviceComp::step's stage order: one
// sensor per step, face sigma = max over the 4-cell axis window, sigma 0 on a
// face with a solid side, donor species correction. Rigs and exact solutions
// are ports of the masterplan's review harness (ha.py, riem.py, noz.py, spec.py);
// the device twin in tests/test_gpu_shock.hip reads the same ICs from here.
#pragma once

#include "dns/parallel.hpp"
#include "dns/shockflux.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace shockref {

inline constexpr double kGamma = 1.4;

// rho, m, E, rho Y (the last one empty without species).
template <class T>
struct State
{
    std::array<std::vector<T>, 4> q;
    int n() const { return int(q[0].size()); }
    bool species() const { return !q[3].empty(); }
};

enum class Base { Hybrid, Kg };
enum class Species { Donor, Reuse };

template <class T>
struct Cfg
{
    Base base = Base::Hybrid;
    Species species = Species::Donor;
    double h = 0, cfl = 0.4;
    std::function<void(State<T>&)> hook;                        // stage input, may overwrite rows
    std::function<void(const State<T>&, State<T>&)> src;        // adds to the rhs
    std::vector<std::uint8_t> solid, sensed0;                   // sensed0: rows whose s is forced 0
    bool testForce = false;                                     // sigma 1 everywhere
    bool openAxis = false;                                      // the device's ShockSetup::open[0]
    std::function<void(const State<T>&, double, long)> mon;     // after each step
};

struct Report
{
    bool ok = true;
    std::string why;
    long steps = 0, sensedCells = 0;
    double maxFrac = 0, rhoMin = HUGE_VAL, pMin = HUGE_VAL;
};

template <class T>
inline T pressureOf(T r, T m, T E)
{
    const T gam = T(kGamma);
    return (gam - T(1)) * (E - T(0.5) * (m * m) / r);
}

template <class T>
inline State<T> make(const std::vector<double>& r, const std::vector<double>& u, const std::vector<double>& p,
                     const std::vector<double>* Y = nullptr)
{
    const std::size_t n = r.size();
    State<T> s;
    for (int v = 0; v < 3; ++v) s.q[std::size_t(v)].resize(n);
    if (Y) s.q[3].resize(n);
    for (std::size_t c = 0; c < n; ++c) {
        const T rr = T(r[c]), uu = T(u[c]), pp = T(p[c]);
        s.q[0][c] = rr;
        s.q[1][c] = rr * uu;
        s.q[2][c] = pp / (T(kGamma) - T(1)) + T(0.5) * rr * uu * uu;
        if (Y) s.q[3][c] = rr * T((*Y)[c]);
    }
    return s;
}

template <class T>
inline T mcSlope(T a, T b)
{
    if (!(a * b > T(0))) return T(0);
    const T m = std::min({T(2) * std::abs(a), T(2) * std::abs(b), T(0.5) * std::abs(a + b)});
    return a > T(0) ? m : -m;
}

template <class T>
inline dns::ShockState<T> stateAt(const State<T>& s, const std::vector<T>& p, int c)
{
    const T r = s.q[0][std::size_t(c)];
    return {r, {s.q[1][std::size_t(c)] / r, T(0), T(0)}, p[std::size_t(c)], s.q[2][std::size_t(c)] / r, T(kGamma)};
}

// The stride-2 sensor of a 1D state: raw v, then bytes. A solid at +-1 or +-2
// reads the centre; solid and sensed0 rows are 0.
template <class T>
inline std::vector<std::uint8_t> senseBytes(const State<T>& s, const Cfg<T>& c, double* vmax = nullptr)
{
    const int n = s.n();
    std::vector<T> p(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        p[std::size_t(i)] = pressureOf(s.q[0][std::size_t(i)], s.q[1][std::size_t(i)], s.q[2][std::size_t(i)]);
    std::vector<std::uint8_t> by(static_cast<std::size_t>(n), 0);
    auto wrap = [n](int i) { return std::size_t(((i % n) + n) % n); };
    for (int i = 0; i < n; ++i) {
        const std::size_t I = std::size_t(i);
        if (!c.solid.empty() && c.solid[I]) continue;
        if (!c.sensed0.empty() && c.sensed0[I]) continue;
        if (c.openAxis && (i < 2 || i >= n - 2)) continue;
        const bool blockP = !c.solid.empty() && (c.solid[wrap(i + 1)] || c.solid[wrap(i + 2)]);
        const bool blockM = !c.solid.empty() && (c.solid[wrap(i - 1)] || c.solid[wrap(i - 2)]);
        const T pp = blockP ? p[I] : p[wrap(i + 2)], pm = blockM ? p[I] : p[wrap(i - 2)];
        const T v = dns::shockJump(pm, p[I], pp);
        by[I] = dns::shockByte(v);
        if (vmax) *vmax = std::max(*vmax, double(v));
    }
    return by;
}

// out = -div F (+ src): KG on every face, + D on the sigma > 0 faces.
template <class T>
inline void rhs(const State<T>& s, const std::vector<std::uint8_t>& by, const Cfg<T>& c, State<T>& out)
{
    const int n = s.n();
    const bool sp = s.species();
    std::vector<T> p(static_cast<std::size_t>(n)), Y(sp ? std::size_t(n) : std::size_t(0));
    for (int i = 0; i < n; ++i) {
        const std::size_t I = std::size_t(i);
        p[I] = pressureOf(s.q[0][I], s.q[1][I], s.q[2][I]);
        if (sp) Y[I] = s.q[3][I] / s.q[0][I];
    }
    auto wrap = [n](int i) { return std::size_t(((i % n) + n) % n); };
    const T inv = T(1) / T(c.h);
    std::vector<std::array<T, 4>> F(static_cast<std::size_t>(n));
    const T nrm[3] = {T(1), T(0), T(0)};
    for (int i = 0; i < n; ++i) {
        const std::size_t L = wrap(i), R = wrap(i + 1);
        const T rL = s.q[0][L], rR = s.q[0][R];
        const T uL = s.q[1][L] / rL, uR = s.q[1][R] / rR;
        const T rm = T(0.5) * (rL + rR), um = T(0.5) * (uL + uR);
        const T em = T(0.5) * (s.q[2][L] / rL + s.q[2][R] / rR), pm = T(0.5) * (p[L] + p[R]);
        const T mk = rm * um;
        std::array<T, 4> f{mk, mk * um + pm, mk * em + pm * um, T(0)};
        T yk = T(0);
        if (sp) {
            yk = mk >= T(0) ? Y[L] + T(0.5) * mcSlope(Y[L] - Y[wrap(i - 1)], Y[R] - Y[L])
                            : Y[R] + T(0.5) * mcSlope(Y[R] - Y[wrap(i + 2)], Y[L] - Y[R]);
            f[3] = mk * yk;
        }
        if (c.base == Base::Hybrid) {
            const bool wall = !c.solid.empty() && (c.solid[L] || c.solid[R]);
            const int b = std::max({int(by[wrap(i - 1)]), int(by[L]), int(by[R]), int(by[wrap(i + 2)])});
            const bool edge = c.openAxis && (i == 0 || i >= n - 2);
            if ((b || c.testForce) && !wall && !edge) {
                const T sig = c.testForce ? T(1) : T(b) / T(255);
                const auto D = dns::shockFaceDelta(sig, stateAt(s, p, int(L)), stateAt(s, p, int(R)), nrm);
                f[0] += D.v[0];
                f[1] += D.v[1];
                f[2] += D.v[4];
                if (sp)
                    f[3] += c.species == Species::Donor
                                ? std::max(D.v[0], T(0)) * Y[L] + std::min(D.v[0], T(0)) * Y[R]
                                : D.v[0] * yk;
            }
        }
        F[L] = f;
    }
    for (int i = 0; i < n; ++i) {
        const std::size_t I = std::size_t(i), Lo = wrap(i - 1);
        const bool sol = !c.solid.empty() && c.solid[I];
        for (std::size_t v = 0; v < (sp ? 4u : 3u); ++v)
            out.q[v][I] = sol ? T(0) : -(F[I][v] - F[Lo][v]) * inv;
    }
    if (c.src) c.src(s, out);
}

template <class T>
inline double waveMax(const State<T>& s)
{
    double w = 0;
    for (int i = 0; i < s.n(); ++i) {
        const std::size_t I = std::size_t(i);
        const double r = s.q[0][I], u = double(s.q[1][I]) / r;
        const double p = (kGamma - 1) * (double(s.q[2][I]) - 0.5 * u * u * r);
        const double sp = std::abs(u) + std::sqrt(std::max(kGamma * p / r, 0.0));
        if (!std::isfinite(sp)) return HUGE_VAL;
        w = std::max(w, sp);
    }
    return w;
}

// March to tend at --cfl, dt recomputed every step (the sticky hot cadence).
template <class T>
inline Report run(State<T>& U, double tend, const Cfg<T>& c, long maxSteps = 20000000)
{
    Report rep;
    const int n = U.n();
    const std::size_t nv = U.species() ? 4 : 3;
    State<T> U1 = U, U2 = U, R = U;
    double t = 0;
    auto hook = [&](State<T>& s) {
        if (c.hook) c.hook(s);
    };
    auto valid = [&](const State<T>& s) {
        for (int i = 0; i < n; ++i) {
            const std::size_t I = std::size_t(i);
            if (!c.solid.empty() && c.solid[I]) continue;
            const T r = s.q[0][I], p = pressureOf(r, s.q[1][I], s.q[2][I]);
            if (!(std::isfinite(double(r)) && std::isfinite(double(p)) && r > T(0) && p > T(0))) return false;
            rep.rhoMin = std::min(rep.rhoMin, double(r));
            rep.pMin = std::min(rep.pMin, double(p));
        }
        return true;
    };
    while (t < tend - 1e-14 && rep.steps < maxSteps) {
        hook(U);
        if (!valid(U)) {
            rep.ok = false;
            rep.why = "state not positive at t=" + std::to_string(t) + " step " + std::to_string(rep.steps);
            return rep;
        }
        const double w = waveMax(U);
        const double dtv = std::min(c.cfl * c.h / w, tend - t);
        const T dt = T(dtv);
        const auto by = c.base == Base::Hybrid ? senseBytes(U, c) : std::vector<std::uint8_t>(std::size_t(n), 0);
        const long a = std::ranges::count_if(by, [](std::uint8_t b) { return b != 0; });
        rep.sensedCells += a;
        rep.maxFrac = std::max(rep.maxFrac, double(a) / n);
        rhs(U, by, c, R);
        for (std::size_t v = 0; v < nv; ++v)
            for (std::size_t I = 0; I < std::size_t(n); ++I) U1.q[v][I] = U.q[v][I] + dt * R.q[v][I];
        hook(U1);
        rhs(U1, by, c, R);
        for (std::size_t v = 0; v < nv; ++v)
            for (std::size_t I = 0; I < std::size_t(n); ++I)
                U2.q[v][I] = T(0.75) * U.q[v][I] + T(0.25) * U1.q[v][I] + T(0.25) * dt * R.q[v][I];
        hook(U2);
        rhs(U2, by, c, R);
        const T cb = T(2.0 / 3.0), ca = T(1) - cb;
        for (std::size_t v = 0; v < nv; ++v)
            for (std::size_t I = 0; I < std::size_t(n); ++I)
                U.q[v][I] = ca * U.q[v][I] + cb * U2.q[v][I] + cb * dt * R.q[v][I];
        t += dtv;
        ++rep.steps;
        if (c.mon) c.mon(U, t, rep.steps);
    }
    hook(U);
    return rep;
}

// --- exact solutions -----------------------------------------------------------------

struct Prim
{
    double r, u, p;
};

// Toro's exact Riemann solver: (r, u, p) at x/t = S relative to x0.
struct Riemann
{
    double rl, ul, pl, rr, ur, pr, g = kGamma, cl = 0, cr = 0, ps = 0, us = 0;

    static void fk(double g, double ps, double rk, double pk, double ck, double& f, double& d)
    {
        if (ps > pk) {
            const double A = 2 / ((g + 1) * rk), B = (g - 1) / (g + 1) * pk, s = std::sqrt(A / (ps + B));
            f = (ps - pk) * s;
            d = s * (1 - (ps - pk) / (2 * (B + ps)));
            return;
        }
        f = 2 * ck / (g - 1) * (std::pow(ps / pk, (g - 1) / (2 * g)) - 1);
        d = 1 / (rk * ck) * std::pow(ps / pk, -(g + 1) / (2 * g));
    }
    void solve()
    {
        cl = std::sqrt(g * pl / rl);
        cr = std::sqrt(g * pr / rr);
        ps = std::max(1e-14, 0.5 * (pl + pr));
        double fl, dl, fr, dr;
        for (int it = 0; it < 200; ++it) {
            fk(g, ps, rl, pl, cl, fl, dl);
            fk(g, ps, rr, pr, cr, fr, dr);
            const double f = fl + fr + ur - ul, p2 = std::max(1e-14, ps - f / (dl + dr));
            const bool done = std::abs(p2 - ps) < 1e-13 * ps;
            ps = p2;
            if (done) break;
        }
        fk(g, ps, rl, pl, cl, fl, dl);
        fk(g, ps, rr, pr, cr, fr, dr);
        us = 0.5 * (ul + ur) + 0.5 * (fr - fl);
    }
    Prim at(double S) const
    {
        const double a = (g - 1) / (g + 1), e = (g - 1) / (2 * g);
        if (S <= us) {
            if (ps > pl) {
                const double SL = ul - cl * std::sqrt((g + 1) / (2 * g) * ps / pl + e);
                if (S <= SL) return {rl, ul, pl};
                return {rl * (ps / pl + a) / (a * ps / pl + 1), us, ps};
            }
            const double Hd = ul - cl, csl = cl * std::pow(ps / pl, e), Tl = us - csl;
            if (S <= Hd) return {rl, ul, pl};
            if (S >= Tl) return {rl * std::pow(ps / pl, 1 / g), us, ps};
            const double c = 2 / (g + 1) * (cl + (g - 1) / 2 * (ul - S)), uu = 2 / (g + 1) * (cl + (g - 1) / 2 * ul + S);
            return {rl * std::pow(c / cl, 2 / (g - 1)), uu, pl * std::pow(c / cl, 2 * g / (g - 1))};
        }
        if (ps > pr) {
            const double SR = ur + cr * std::sqrt((g + 1) / (2 * g) * ps / pr + e);
            if (S >= SR) return {rr, ur, pr};
            return {rr * (ps / pr + a) / (a * ps / pr + 1), us, ps};
        }
        const double Hd = ur + cr, csr = cr * std::pow(ps / pr, e), Tr = us + csr;
        if (S >= Hd) return {rr, ur, pr};
        if (S <= Tr) return {rr * std::pow(ps / pr, 1 / g), us, ps};
        const double c = 2 / (g + 1) * (cr - (g - 1) / 2 * (ur - S)), uu = 2 / (g + 1) * (-cr + (g - 1) / 2 * ur + S);
        return {rr * std::pow(c / cr, 2 / (g - 1)), uu, pr * std::pow(c / cr, 2 * g / (g - 1))};
    }
};

struct Tube
{
    const char* name;
    double rl, ul, pl, rr, ur, pr, x0, t;
};

// The Riemann cases of the ladder, held rows 4 cells, sensor masked 6 (final.py).
inline constexpr Tube kTubes[] = {
    {"sod", 1, 0, 1, .125, 0, .1, .5, .2},        {"lax", .445, .698, 3.528, .5, 0, .571, .5, .14},
    {"t123", 1, -2, .4, 1, 2, .4, .5, .15},       {"blast", 1, 0, 1000, 1, 0, .01, .5, .012},
    {"m3", 3.857, .92, 10.33, 1, 3.55, 1, .5, .09}, {"u5", 1, -5, .4, 1, 5, .4, .5, .05},
    {"hipr", 1, 0, 1e5, 1, 0, 1, .5, .0012},      {"u8", 1, -8, 1, 1, 8, 1, .5, .03},
    {"u16", 1, -16, 1, 1, 16, 1, .5, .015},       {"coll", 1, 8, 1, 1, -8, 1, .5, .03},
};

template <class T>
inline State<T> tubeIc(const Tube& b, int N)
{
    std::vector<double> r(static_cast<std::size_t>(N)), u(r), p(r);
    for (int i = 0; i < N; ++i) {
        const double x = (i + 0.5) / N;
        const bool m = x < b.x0;
        r[std::size_t(i)] = m ? b.rl : b.rr;
        u[std::size_t(i)] = m ? b.ul : b.ur;
        p[std::size_t(i)] = m ? b.pl : b.pr;
    }
    return make<T>(r, u, p);
}

// Shu-Osher on [-5, 5].
template <class T>
inline State<T> shuOsherIc(int N)
{
    std::vector<double> r(static_cast<std::size_t>(N)), u(r), p(r);
    for (int i = 0; i < N; ++i) {
        const std::size_t I = std::size_t(i);
        const double x = -5 + (i + .5) * 10.0 / N;
        const bool m = x < -4;
        r[I] = m ? 3.857143 : 1 + .2 * std::sin(5 * x);
        u[I] = m ? 2.629369 : 0;
        p[I] = m ? 10.33333 : 1;
    }
    return make<T>(r, u, p);
}

// March U in place with its first and last 4 rows held. deviceMask: the device
// rig's open axis (sensor 2 rows off the ends, no end face), else the review
// harness's 6-row mask.
template <class T>
inline Report heldRun(State<T>& U, double h, double tEnd, bool deviceMask)
{
    const int N = U.n();
    const State<T> U0 = U;
    Cfg<T> c;
    c.h = h;
    c.openAxis = deviceMask;
    if (!deviceMask) {
        c.sensed0.assign(std::size_t(N), 0);
        for (int i = 0; i < 6; ++i) c.sensed0[std::size_t(i)] = c.sensed0[std::size_t(N - 1 - i)] = 1;
    }
    c.hook = [&](State<T>& V) {
        for (std::size_t v = 0; v < 3; ++v)
            for (int i = 0; i < 4; ++i) {
                V.q[v][std::size_t(i)] = U0.q[v][std::size_t(i)];
                V.q[v][std::size_t(N - 1 - i)] = U0.q[v][std::size_t(N - 1 - i)];
            }
    };
    return run(U, tEnd, c);
}

// A tube on N cells: held rows, sensor masked. Writes L1 of rho against exact.
template <class T>
inline Report tubeRun(const Tube& b, int N, State<T>& out, double* l1rho = nullptr, bool deviceMask = false)
{
    State<T> U = tubeIc<T>(b, N);
    const Report rep = heldRun(U, 1.0 / N, b.t, deviceMask);
    out = U;
    if (l1rho) {
        Riemann ex{b.rl, b.ul, b.pl, b.rr, b.ur, b.pr};
        ex.solve();
        double s = 0;
        for (int i = 0; i < N; ++i) {
            const double x = (i + 0.5) / N;
            s += std::abs(double(U.q[0][std::size_t(i)]) - ex.at((x - b.x0) / b.t).r);
        }
        *l1rho = s / N;
    }
    return rep;
}

// --- nozzle (noz.py) -----------------------------------------------------------------

struct Nozzle
{
    static constexpr double g = kGamma, xt = .25, Ain = 2, Ae = 6, Ms = 3;
    static double area(double x)
    {
        const double a = (x - xt) / xt, b = (x - xt) / (1 - xt);
        return x < xt ? 1 + (Ain - 1) * a * a : 1 + (Ae - 1) * b * b;
    }
    static double dArea(double x)
    {
        return x < xt ? 2 * (Ain - 1) * (x - xt) / (xt * xt) : 2 * (Ae - 1) * (x - xt) / ((1 - xt) * (1 - xt));
    }
    static double ratio(double M)
    {
        return 1 / M * std::pow((2 / (g + 1)) * (1 + (g - 1) / 2 * M * M), (g + 1) / (2 * (g - 1)));
    }
    // M of area ratio a on the sub/supersonic branch, bisection.
    static double machOf(double a, bool sup)
    {
        double lo = sup ? 1 + 1e-12 : 1e-8, hi = sup ? 80.0 : 1 - 1e-12;
        if (a <= 1) return 1;
        for (int i = 0; i < 200; ++i) {
            const double m = 0.5 * (lo + hi);
            const bool above = ratio(m) > a;
            if (sup == above) hi = m;
            else lo = m;
        }
        return 0.5 * (lo + hi);
    }
    double As = ratio(Ms);
    double xs = xt + (1 - xt) * std::sqrt((As - 1) / (Ae - 1));
    double M2 = std::sqrt((1 + (g - 1) / 2 * Ms * Ms) / (g * Ms * Ms - (g - 1) / 2));
    double p02 = std::pow((g + 1) / 2 * Ms * Ms / (1 + (g - 1) / 2 * Ms * Ms), g / (g - 1)) *
                 std::pow(2 * g / (g + 1) * Ms * Ms - (g - 1) / (g + 1), -1 / (g - 1));
    double As2 = As / ratio(M2);
    Prim exact(double x, bool shock, double* M = nullptr) const
    {
        const double a = area(x);
        double m, pt = 1;
        if (x <= xt) m = machOf(a, false);
        else if (x <= xs || !shock) m = machOf(a, true);
        else {
            m = machOf(a / As2, false);
            pt = p02;
        }
        if (M) *M = m;
        const double T = 1 / (1 + (g - 1) / 2 * m * m), p = pt * std::pow(1 + (g - 1) / 2 * m * m, -g / (g - 1));
        return {p / T, m * std::sqrt(g * T), p};
    }
    double pBack() const { return exact(1.0, true).p; }
};

struct NozzleResult
{
    Report rep;
    double dx = NAN, jitter = NAN, l1p = NAN, mpk = 0;
    std::vector<double> rho;
};

// Quasi-1D nozzle: N cells + 3 ghosts per side, inlet (p0, T0) with interior p,
// outlet p_b with interior rho, u; area source. rest: a throat diaphragm from rest.
template <class T>
inline NozzleResult nozzleRun(int N, double tEnd, bool shock, bool rest, Base base = Base::Hybrid)
{
    const int ng = 3, nt = N + 2 * ng;
    const double h = 1.0 / N;
    const Nozzle nz;
    const double pb = nz.pBack();
    std::vector<double> xg(static_cast<std::size_t>(nt)), r(xg), u(xg), p(xg);
    std::vector<T> ar(static_cast<std::size_t>(nt));
    for (int i = 0; i < nt; ++i) {
        const std::size_t I = std::size_t(i);
        xg[I] = (i - ng + 0.5) * h;
        ar[I] = T(Nozzle::dArea(xg[I]) / Nozzle::area(xg[I]));
        const Prim q = nz.exact(xg[I], shock);
        r[I] = rest ? (xg[I] < Nozzle::xt ? 1.0 : pb) : q.r;
        u[I] = rest ? 0.0 : q.u;
        p[I] = rest ? r[I] : q.p;
    }
    State<T> U = make<T>(r, u, p);
    Cfg<T> c;
    c.h = h;
    c.base = base;
    c.sensed0.assign(std::size_t(nt), 0);
    for (int i = 0; i < ng + 2; ++i) c.sensed0[std::size_t(i)] = c.sensed0[std::size_t(nt - 1 - i)] = 1;
    c.src = [&ar](const State<T>& s, State<T>& R) {
        for (std::size_t I = 0; I < s.q[0].size(); ++I) {
            const T rr = s.q[0][I], m = s.q[1][I], E = s.q[2][I], uu = m / rr;
            const T pp = T(0.4) * (E - T(0.5) * m * uu);
            R.q[0][I] -= ar[I] * m;
            R.q[1][I] -= ar[I] * (m * uu);
            R.q[2][I] -= ar[I] * (uu * (E + pp));
        }
    };
    c.hook = [&](State<T>& V) {
        const std::size_t in = std::size_t(ng), out = std::size_t(nt - ng - 1);
        {
            const double rr = V.q[0][in], m = V.q[1][in], E = V.q[2][in], uu = m / rr;
            const double pp = std::min(0.4 * (E - 0.5 * m * uu), .9999);
            const double Ti = std::pow(pp, .4 / 1.4), ri = pp / Ti, ui = std::sqrt(std::max(2 * 1.4 / .4 * (1 - Ti), 0.0));
            for (int i = 0; i < ng; ++i) {
                V.q[0][std::size_t(i)] = T(ri);
                V.q[1][std::size_t(i)] = T(ri * ui);
                V.q[2][std::size_t(i)] = T(pp / .4 + .5 * ri * ui * ui);
            }
        }
        const T rr = V.q[0][out], m = V.q[1][out], E = V.q[2][out];
        const double uu = double(m) / double(rr);
        for (int i = 0; i < ng; ++i) {
            const std::size_t J = std::size_t(nt - 1 - i);
            V.q[0][J] = rr;
            V.q[1][J] = m;
            V.q[2][J] = shock ? T(pb / .4 + .5 * double(m) * uu) : E;
        }
    };
    std::vector<double> xsh;
    const double tMon = (rest ? 0.8 : 0.6) * tEnd;
    c.mon = [&](const State<T>& V, double t, long) {
        if (!(t > tMon)) return;
        double ms = NAN;
        double Mj = 0, Mn = 0, xj = 0;
        for (int i = ng; i + 1 < nt - ng; ++i) {
            const auto mach = [&](int q) {
                const double rr = V.q[0][std::size_t(q)], uu = double(V.q[1][std::size_t(q)]) / rr;
                const double pp = double(pressureOf(V.q[0][std::size_t(q)], V.q[1][std::size_t(q)], V.q[2][std::size_t(q)]));
                return uu / std::sqrt(kGamma * pp / rr);
            };
            const double a = mach(i), b = mach(i + 1);
            if (xg[std::size_t(i)] > Nozzle::xt && a > 1 && b <= 1) {
                ms = xg[std::size_t(i)];
                Mj = a;
                Mn = b;
                xj = ms;
            }
        }
        if (std::isfinite(ms)) xsh.push_back(xj + h * (Mj - 1) / (Mj - Mn));
    };
    NozzleResult o;
    o.rep = run(U, tEnd, c);
    if (!o.rep.ok) return o;
    if (!xsh.empty()) {
        double m = 0, v = 0;
        for (double x : xsh) m += x;
        m /= double(xsh.size());
        for (double x : xsh) v += (x - m) * (x - m);
        o.dx = (m - nz.xs) / h;
        o.jitter = std::sqrt(v / double(xsh.size())) / h;
    }
    double s = 0, sx = 0, cnt = 0;
    for (int i = ng; i < nt - ng; ++i) {
        const std::size_t I = std::size_t(i);
        const double x = xg[I];
        const double rr = U.q[0][I], uu = double(U.q[1][I]) / rr;
        const double pp = pressureOf(U.q[0][I], U.q[1][I], U.q[2][I]);
        o.mpk = std::max(o.mpk, uu / std::sqrt(kGamma * pp / rr));
        o.rho.push_back(rr);
        if (std::abs(x - nz.xs) <= .05) continue;
        const double px = nz.exact(x, shock).p;
        s += std::abs(pp - px);
        sx += px;
        cnt += 1;
    }
    o.l1p = s / sx;
    return o;
}

// --- Rankine-Hugoniot, the moving and reflected shocks (disc2.py, wall.py) ---------------

struct RH
{
    double r2, u2, p2, S;
};
inline RH rankine(double Ms)
{
    const double g = kGamma, r2 = (g + 1) * Ms * Ms / ((g - 1) * Ms * Ms + 2);
    const double p2 = 1 + 2 * g / (g + 1) * (Ms * Ms - 1), c1 = std::sqrt(g);
    return {r2, Ms * c1 * (1 - 1 / r2), p2, Ms * c1};
}
struct Reflect
{
    double r5, p5, w;
};
inline Reflect reflected(double Ms)
{
    const double g = kGamma;
    const RH a = rankine(Ms);
    const double c2 = std::sqrt(g * a.p2 / a.r2);
    double lo = 1.001, hi = 20;
    auto f = [&](double Mr) {
        const double rr = (g + 1) * Mr * Mr / ((g - 1) * Mr * Mr + 2);
        return (Mr * c2) * (1 - 1 / rr) - a.u2;
    };
    for (int i = 0; i < 200; ++i) {
        const double m = 0.5 * (lo + hi);
        if (f(m) > 0) hi = m;
        else lo = m;
    }
    const double Mr = 0.5 * (lo + hi);
    return {a.r2 * (g + 1) * Mr * Mr / ((g - 1) * Mr * Mr + 2), a.p2 * (1 + 2 * g / (g + 1) * (Mr * Mr - 1)), Mr * c2 - a.u2};
}

} // namespace shockref
