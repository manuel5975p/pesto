// --shock's header and host ladder (masterplan_shockflux section 8, gates 1-8):
// tests/shock_ref.hpp's marcher on dns/shockflux.hpp against exact solutions. One
// process per gate: `test_shockflux <unit|riemann|shuosher|nozzle|weak|contact|
// species|wall>`. Bands are the plan's (measured value in the comment).
#include "dns/fvflux.hpp"
#include "shock_ref.hpp"

#include "check.hpp"

#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <string_view>
#include <thread>

using namespace shockref;
using dnstest::check;

namespace {

template <class T>
dns::ShockState<T> st(double r, double u, double v, double w, double p)
{
    const double E = p / (kGamma - 1) + 0.5 * r * (u * u + v * v + w * w);
    return {T(r), {T(u), T(v), T(w)}, T(p), T(E / r), T(kGamma)};
}

template <class T>
void unitFor(const char* tag)
{
    const T ex[3] = {T(1), T(0), T(0)}, ey[3] = {T(0), T(1), T(0)}, mx[3] = {T(-1), T(0), T(0)};
    // HLL(U, U) = the Euler flux: the centred form makes it exact, D == 0.
    bool same = true;
    for (double u : {0.0, 0.3, -2.5, 7.0})
        for (double v : {0.0, 1.5})
            for (double r : {1.0, 0.125, 8.0})
                for (double p : {1.0, 1e-3, 1e4}) {
                    const auto s = st<T>(r, u, v, 0.0, p);
                    for (const T* n : {ex, ey}) {
                        const auto D = dns::shockFaceDelta(T(1), s, s, n);
                        for (T x : D.v) same = same && x == T(0);
                    }
                }
    check(same, std::format("{}: HLL(U,U) equals the Euler flux bit for bit (D == 0)", tag));
    // Mirror: (L, R, n) -> (R', L', -n) negates the mass and E fluxes, keeps tangentials.
    bool mir = true;
    for (double uL : {-1.0, 0.0, 2.0, 9.0})
        for (double uR : {-3.0, 0.5, 4.0})
            for (double pR : {0.1, 1.0, 50.0}) {
                const auto L = st<T>(1.0, uL, 0.25, 0.0, 1.0), R = st<T>(0.3, uR, -0.5, 0.0, pR);
                const auto Lm = st<T>(1.0, uL, 0.25, 0.0, 1.0), Rm = st<T>(0.3, uR, -0.5, 0.0, pR);
                const auto a = dns::shockFaceDelta(T(1), L, R, ex);
                const auto b = dns::shockFaceDelta(T(1), Rm, Lm, mx);
                for (int v = 0; v < 5; ++v) mir = mir && a.v[v] == -b.v[v];
            }
    check(mir, std::format("{}: the flux of a swapped face through -n is -F bit for bit", tag));
    const auto L = st<T>(1.0, 2.0, 0.0, 0.0, 3.0), R = st<T>(0.1, -1.0, 0.0, 0.0, 0.2);
    const auto Z = dns::shockFaceDelta(T(0), L, R, ex);
    bool zero = true;
    for (T x : Z.v) zero = zero && x == T(0);
    check(zero, std::format("{}: sigma 0 gives D exactly 0", tag));
    // The byte.
    const T th1 = T(dns::kShockTh1), th2 = T(dns::kShockTh2);
    check(dns::shockByte(T(0)) == 0 && dns::shockByte(th1) == 0, std::format("{}: byte 0 at or below th1", tag));
    check(dns::shockByte(std::nextafter(th1, T(1))) != 0, std::format("{}: byte nonzero just above th1", tag));
    check(dns::shockByte(th2) == 255 && dns::shockByte(T(1)) == 255, std::format("{}: byte 255 at th2 and above", tag));
    check(dns::shockByte(std::numeric_limits<T>::quiet_NaN()) == 255, std::format("{}: byte 255 on NaN", tag));
    check(dns::shockByte(std::nextafter(th2, T(0))) <= 255, std::format("{}: byte in range below th2", tag));
    int prev = 0;
    bool mono = true;
    for (int i = 0; i <= 2000; ++i) {
        const int b = dns::shockByte(T(0.01 + i * 6e-5));
        mono = mono && b >= prev;
        prev = b;
    }
    check(mono, std::format("{}: byte monotone in v", tag));
    // v exactly 0 on constant, linear, odd-even.
    check(dns::shockJump(T(3), T(3), T(3)) == T(0), std::format("{}: v 0 on a constant", tag));
    check(dns::shockJump(T(1), T(3), T(5)) == T(0), std::format("{}: v 0 on a linear field", tag));
    check(dns::shockJump(T(2), T(2), T(2)) == T(0) && dns::shockJump(T(7), T(7), T(7)) == T(0),
          std::format("{}: v 0 on an odd-even field (stride 2 reads its own colour)", tag));
}

int unit()
{
    unitFor<float>("FP32");
    unitFor<double>("FP64");
    // FP32 vs FP64 of one strong face.
    const auto a = dns::shockFaceDelta(float(1), st<float>(3.857, .92, 0, 0, 10.33), st<float>(1, 3.55, 0, 0, 1),
                                       (const float[3]){1, 0, 0});
    const auto b = dns::shockFaceDelta(1.0, st<double>(3.857, .92, 0, 0, 10.33), st<double>(1, 3.55, 0, 0, 1),
                                       (const double[3]){1, 0, 0});
    double e = 0, sc = 0;
    for (int v = 0; v < 5; ++v) {
        e = std::max(e, std::abs(double(a.v[v]) - b.v[v]));
        sc = std::max(sc, std::abs(b.v[v]));
    }
    std::printf("  FP32 vs FP64 D on a strong face: %.2e of %.3g\n", e, sc);
    check(e <= 1e-5 * sc, "FP32 vs FP64 D agree to 1e-5");
    return dnstest::finish();
}

int riemann()
{
    struct Gate
    {
        const char* name;
        double b200, b400;
    };
    // Bands: L1 rho at N 200, 400 (measured 6.79e-3 / 3.90e-3, ...).
    const Gate gates[] = {{"sod", 7.5e-3, 4.3e-3},
                          {"lax", 2.8e-2, 1.6e-2},
                          {"t123", 1.4e-2, 8.1e-3},
                          {"blast", 0.18, 0.102},
                          {"m3", 2.0e-2, 1.2e-2}};
    for (const Tube& b : kTubes) {
        for (int N : {200, 400}) {
            State<float> U;
            double l1 = 0;
            const Report rep = tubeRun<float>(b, N, U, &l1);
            std::printf("  %-6s N %d L1 rho %.4e steps %ld maxf %.3f rhoMin %.3g pMin %.3g %s\n", b.name, N, l1, rep.steps,
                        rep.maxFrac, rep.rhoMin, rep.pMin, rep.why.c_str());
            check(rep.ok, std::format("{} N {}: rho, p > 0", b.name, N));
            for (const Gate& g : gates)
                if (std::string_view(g.name) == b.name)
                    check(l1 <= (N == 200 ? g.b200 : g.b400),
                          std::format("{} N {}: L1 rho {:.4e} <= {:.3g}", b.name, N, l1, N == 200 ? g.b200 : g.b400));
        }
    }
    return dnstest::finish();
}

// One MUSCL-MC HLLC rhs on faces [NG-1, n-NG), FP64, threaded (own index only):
// the Shu-Osher reference, A3's t2_shuosher.py.
struct Hllc
{
    static double mc(double a, double b)
    {
        if (!(a * b > 0)) return 0;
        const double m = std::min({2 * std::abs(a), 2 * std::abs(b), 0.5 * std::abs(a + b)});
        return a > 0 ? m : -m;
    }
    static std::array<double, 3> flux(double rL, double uL, double pL, double rR, double uR, double pR)
    {
        const double g = kGamma;
        const double EL = pL / (g - 1) + .5 * rL * uL * uL, ER = pR / (g - 1) + .5 * rR * uR * uR;
        const double sL = std::sqrt(rL), sR = std::sqrt(rR), w = 1 / (sL + sR);
        const double ut = (sL * uL + sR * uR) * w, Ht = (sL * (EL + pL) / rL + sR * (ER + pR) / rR) * w;
        const double ct = std::sqrt(std::max((g - 1) * (Ht - .5 * ut * ut), 0.0));
        const double cL = std::sqrt(g * pL / rL), cR = std::sqrt(g * pR / rR);
        const double SL = std::min(uL - cL, ut - ct), SR = std::max(uR + cR, ut + ct);
        const double S = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR));
        const std::array<double, 3> FL{rL * uL, rL * uL * uL + pL, uL * (EL + pL)}, FR{rR * uR, rR * uR * uR + pR, uR * (ER + pR)};
        if (SL >= 0) return FL;
        if (SR <= 0) return FR;
        const auto star = [&](double r, double u, double p, double E, double Sk) {
            const double k = r * (Sk - u) / (Sk - S);
            return std::array<double, 3>{k, k * S, k * (E / r + (S - u) * (S + p / (r * (Sk - u))))};
        };
        const auto U = S >= 0 ? star(rL, uL, pL, EL, SL) : star(rR, uR, pR, ER, SR);
        const auto& F = S >= 0 ? FL : FR;
        const double Sk = S >= 0 ? SL : SR;
        const double r0 = S >= 0 ? rL : rR, u0 = S >= 0 ? uL : uR, E0 = S >= 0 ? EL : ER;
        return {F[0] + Sk * (U[0] - r0), F[1] + Sk * (U[1] - r0 * u0), F[2] + Sk * (U[2] - E0)};
    }
};

std::vector<double> shuOsherRef(int N, double tEnd, std::vector<double>& xc)
{
    const int NG = 3, nt = N + 2 * NG;
    const double h = 10.0 / N, g = kGamma;
    std::vector<double> xg(static_cast<std::size_t>(nt));
    std::array<std::vector<double>, 3> U;
    for (auto& a : U) a.resize(std::size_t(nt));
    for (int i = 0; i < nt; ++i) {
        const std::size_t I = std::size_t(i);
        xg[I] = -5 + (i - NG + .5) * h;
        const bool m = xg[I] < -4;
        const double r = m ? 3.857143 : 1 + .2 * std::sin(5 * xg[I]), u = m ? 2.629369 : 0, p = m ? 10.33333 : 1;
        U[0][I] = r;
        U[1][I] = r * u;
        U[2][I] = p / (g - 1) + .5 * r * u * u;
    }
    const int nw = dns::workerCount(0, std::size_t(nt));
    auto bc = [&](std::array<std::vector<double>, 3>& V) {
        for (auto& a : V)
            for (int k = 0; k < NG; ++k) {
                a[std::size_t(k)] = a[std::size_t(NG)];
                a[std::size_t(nt - 1 - k)] = a[std::size_t(nt - NG - 1)];
            }
    };
    std::vector<std::array<double, 3>> F(static_cast<std::size_t>(nt));
    std::array<std::vector<double>, 3> Q; // r u p
    for (auto& a : Q) a.resize(std::size_t(nt));
    auto rhs = [&](std::array<std::vector<double>, 3>& V, std::array<std::vector<double>, 3>& R) {
        bc(V);
        for (int i = 0; i < nt; ++i) {
            const std::size_t I = std::size_t(i);
            Q[0][I] = V[0][I];
            Q[1][I] = V[1][I] / V[0][I];
            Q[2][I] = (g - 1) * (V[2][I] - .5 * V[1][I] * V[1][I] / V[0][I]);
        }
        dns::forBlocks(nw, std::size_t(nt - 2 * NG + 1), [&](int, std::size_t lo, std::size_t hi) {
            for (std::size_t f = lo; f < hi; ++f) {
                const std::size_t i = std::size_t(NG - 1) + f, j = i + 1;
                double L[3], Rr[3];
                bool bad = false;
                for (int v = 0; v < 3; ++v) {
                    const auto& q = Q[std::size_t(v)];
                    L[v] = q[i] + .5 * Hllc::mc(q[i] - q[i - 1], q[j] - q[i]);
                    Rr[v] = q[j] - .5 * Hllc::mc(q[j] - q[i], q[j + 1] - q[j]);
                }
                bad = L[0] <= 0 || Rr[0] <= 0 || L[2] <= 0 || Rr[2] <= 0;
                F[i] = bad ? Hllc::flux(Q[0][i], Q[1][i], Q[2][i], Q[0][j], Q[1][j], Q[2][j])
                           : Hllc::flux(L[0], L[1], L[2], Rr[0], Rr[1], Rr[2]);
            }
        });
        for (int i = NG; i < nt - NG; ++i)
            for (std::size_t v = 0; v < 3; ++v)
                R[v][std::size_t(i)] = -(F[std::size_t(i)][v] - F[std::size_t(i - 1)][v]) / h;
    };
    auto U1 = U, U2 = U, R = U;
    double t = 0;
    while (t < tEnd - 1e-14) {
        bc(U);
        double w = 0;
        for (int i = NG; i < nt - NG; ++i) {
            const std::size_t I = std::size_t(i);
            const double r = U[0][I], u = U[1][I] / r, p = (g - 1) * (U[2][I] - .5 * U[1][I] * u);
            w = std::max(w, std::abs(u) + std::sqrt(std::max(g * p / r, 0.0)));
        }
        const double dt = std::min(0.4 * h / w, tEnd - t);
        rhs(U, R);
        for (std::size_t v = 0; v < 3; ++v)
            for (int i = NG; i < nt - NG; ++i) U1[v][std::size_t(i)] = U[v][std::size_t(i)] + dt * R[v][std::size_t(i)];
        rhs(U1, R);
        for (std::size_t v = 0; v < 3; ++v)
            for (int i = NG; i < nt - NG; ++i)
                U2[v][std::size_t(i)] = .75 * U[v][std::size_t(i)] + .25 * U1[v][std::size_t(i)] + .25 * dt * R[v][std::size_t(i)];
        rhs(U2, R);
        for (std::size_t v = 0; v < 3; ++v)
            for (int i = NG; i < nt - NG; ++i)
                U[v][std::size_t(i)] = (1.0 / 3) * U[v][std::size_t(i)] + (2.0 / 3) * U2[v][std::size_t(i)] + (2.0 / 3) * dt * R[v][std::size_t(i)];
        t += dt;
    }
    std::vector<double> rho(static_cast<std::size_t>(N));
    xc.resize(std::size_t(N));
    for (int i = 0; i < N; ++i) {
        rho[std::size_t(i)] = U[0][std::size_t(i + NG)];
        xc[std::size_t(i)] = xg[std::size_t(i + NG)];
    }
    return rho;
}

double interp(const std::vector<double>& x, const std::vector<double>& y, double q)
{
    const auto it = std::ranges::lower_bound(x, q);
    if (it == x.begin()) return y.front();
    if (it == x.end()) return y.back();
    const std::size_t j = std::size_t(it - x.begin());
    const double w = (q - x[j - 1]) / (x[j] - x[j - 1]);
    return y[j - 1] + w * (y[j] - y[j - 1]);
}

int shuosher()
{
    std::vector<double> xr;
    const auto ref = shuOsherRef(6400, 1.8, xr);
    const struct
    {
        int N;
        double band;
    } rows[] = {{200, 0.12}, {400, 0.106}, {800, 0.052}}; // measured 0.1092 / 0.0962 / 0.0469
    for (const auto& a : rows) {
        const int N = a.N;
        std::vector<double> r(static_cast<std::size_t>(N)), u(r), p(r), x(r);
        for (int i = 0; i < N; ++i) {
            const std::size_t I = std::size_t(i);
            x[I] = -5 + (i + .5) * 10.0 / N;
            const bool m = x[I] < -4;
            r[I] = m ? 3.857143 : 1 + .2 * std::sin(5 * x[I]);
            u[I] = m ? 2.629369 : 0;
            p[I] = m ? 10.33333 : 1;
        }
        State<float> U = make<float>(r, u, p);
        const State<float> U0 = U;
        Cfg<float> c;
        c.h = 10.0 / N;
        c.sensed0.assign(std::size_t(N), 0);
        for (int i = 0; i < 6; ++i) c.sensed0[std::size_t(i)] = c.sensed0[std::size_t(N - 1 - i)] = 1;
        c.hook = [&](State<float>& V) {
            for (std::size_t v = 0; v < 3; ++v)
                for (int i = 0; i < 4; ++i) {
                    V.q[v][std::size_t(i)] = U0.q[v][std::size_t(i)];
                    V.q[v][std::size_t(N - 1 - i)] = U0.q[v][std::size_t(N - 1 - i)];
                }
        };
        const Report rep = run(U, 1.8, c);
        double e = 0;
        for (int i = 0; i < N; ++i) e += std::abs(double(U.q[0][std::size_t(i)]) - interp(xr, ref, x[std::size_t(i)]));
        e /= N;
        std::printf("  Shu-Osher N %d L1 rho %.4f maxf %.3f\n", N, e, rep.maxFrac);
        check(rep.ok, std::format("N {} marches", N));
        check(e <= a.band, std::format("N {} L1 {:.4f} <= {:.3g}", N, e, a.band));
    }
    return dnstest::finish();
}

int nozzle()
{
    const Nozzle nz;
    std::printf("  exact xs %.6f pb %.6f p02 %.6f\n", nz.xs, nz.pBack(), nz.p02);
    check(std::abs(nz.xs - 0.853232) < 1e-5, "x_s 0.853232");
    check(std::abs(nz.pBack() - 0.307029) < 1e-5, "p_b/p0 0.307029");
    check(std::abs(nz.p02 - 0.328344) < 1e-5, "p02/p0 0.328344");
    const struct
    {
        int N;
        double l1;
    } rows[] = {{200, 2.8e-3}, {400, 1.7e-3}}; // measured 2.50e-3, 1.50e-3
    for (const auto& a : rows) {
        const auto f = nozzleRun<float>(a.N, 6.0, true, false);
        const auto d = nozzleRun<double>(a.N, 6.0, true, false);
        const auto r = nozzleRun<float>(a.N, 8.0, true, true);
        std::printf("  N %d FP32 shock %+.2f jitter %.3f L1p %.3e Mpk %.3f maxf %.3f\n", a.N, f.dx, f.jitter, f.l1p, f.mpk, f.rep.maxFrac);
        std::printf("  N %d FP64 shock %+.2f jitter %.3f L1p %.3e Mpk %.3f\n", a.N, d.dx, d.jitter, d.l1p, d.mpk);
        std::printf("  N %d rest FP32 shock %+.2f jitter %.3f %s\n", a.N, r.dx, r.jitter, r.rep.why.c_str());
        check(f.rep.ok && d.rep.ok && r.rep.ok, std::format("N {} marches", a.N));
        check(std::abs(f.dx) <= 1, std::format("N {} |shock| {:.2f} <= 1 cell", a.N, std::abs(f.dx)));
        check(std::abs(r.dx) <= 1, std::format("N {} from rest |shock| {:.2f} <= 1 cell", a.N, std::abs(r.dx)));
        check(f.jitter <= 0.35, std::format("N {} jitter {:.3f} <= 0.35", a.N, f.jitter));
        check(f.l1p <= a.l1, std::format("N {} L1 p {:.3e} <= {:.3g}", a.N, f.l1p, a.l1));
        check(f.mpk >= 2.97, std::format("N {} Mpk {:.3f} >= 2.97", a.N, f.mpk));
        check(std::abs(f.dx - d.dx) <= 0.5, std::format("N {} FP32 vs FP64 {:.2f} <= 0.5 cell", a.N, std::abs(f.dx - d.dx)));
    }
    // Fully supersonic twin: 0 sensed, equal to KG bitwise.
    for (int N : {100, 200}) {
        const auto h = nozzleRun<float>(N, 3.0, false, false);
        const auto k = nozzleRun<float>(N, 3.0, false, false, Base::Kg);
        std::printf("  N %d supersonic twin: sensed cell-steps %ld\n", N, h.rep.sensedCells);
        check(h.rep.ok && k.rep.ok, std::format("twin N {} marches", N));
        check(h.rep.sensedCells == 0, std::format("twin N {}: 0 sensed", N));
        check(h.rho == k.rho, std::format("twin N {}: equal to KG bitwise", N));
    }
    return dnstest::finish();
}

struct Moving
{
    double ring = 0;
    Report rep;
};

Moving moving(double Ms, int N)
{
    const RH a = rankine(Ms);
    std::vector<double> r(static_cast<std::size_t>(N)), u(r), p(r);
    for (int i = 0; i < N; ++i) {
        const double x = (i + .5) / N;
        const bool m = x < .1;
        r[std::size_t(i)] = m ? a.r2 : 1;
        u[std::size_t(i)] = m ? a.u2 : 0;
        p[std::size_t(i)] = m ? a.p2 : 1;
    }
    State<float> U = make<float>(r, u, p);
    const State<float> U0 = U;
    Cfg<float> c;
    c.h = 1.0 / N;
    c.sensed0.assign(std::size_t(N), 0);
    for (int i = 0; i < 6; ++i) c.sensed0[std::size_t(i)] = c.sensed0[std::size_t(N - 1 - i)] = 1;
    c.hook = [&](State<float>& V) {
        for (std::size_t v = 0; v < 3; ++v)
            for (int i = 0; i < 4; ++i) {
                V.q[v][std::size_t(i)] = U0.q[v][std::size_t(i)];
                V.q[v][std::size_t(N - 1 - i)] = U0.q[v][std::size_t(N - 1 - i)];
            }
    };
    Moving o;
    o.rep = run(U, .6 / a.S, c);
    const double xs = .7;
    for (int i = 0; i < N; ++i) {
        const double x = (i + .5) / N;
        if (!(x > xs - .3 && x < xs - 8.0 / N)) continue;
        const double pp = pressureOf(U.q[0][std::size_t(i)], U.q[1][std::size_t(i)], U.q[2][std::size_t(i)]);
        o.ring = std::max(o.ring, std::abs(pp - a.p2) / (a.p2 - 1));
    }
    return o;
}

int weak()
{
    const struct
    {
        double Ms, band;
    } rows[] = {{1.2, 0.07}, {1.5, 0.06}, {2, 0.04}, {3, 0.03}, {6, 0.02}}; // .0607 .0061 .0092 .0150 .0127
    for (const auto& a : rows) {
        const Moving m = moving(a.Ms, 400);
        std::printf("  Ms %g ringing %.4f maxf %.3f %s\n", a.Ms, m.ring, m.rep.maxFrac, m.rep.why.c_str());
        check(m.rep.ok, std::format("Ms {} marches", a.Ms));
        check(m.ring <= a.band, std::format("Ms {} ringing {:.4f} <= {:.3g}", a.Ms, m.ring, a.band));
    }
    const Moving m = moving(1.02, 400);
    std::printf("  Ms 1.02 sensed cell-steps %ld ringing %.4f\n", m.rep.sensedCells, m.ring);
    check(m.rep.ok && m.rep.sensedCells == 0, "Ms 1.02: 0 sensed");
    return dnstest::finish();
}

struct Contact
{
    double peak = 0;
    Report rep;
    State<float> U;
};

Contact contact(double M, double width, int N, Base base)
{
    const double ratio = 10;
    std::vector<double> r(static_cast<std::size_t>(N)), u(r), p(r);
    const double uu = M * std::sqrt(1.4);
    for (int i = 0; i < N; ++i) {
        const double x = (i + .5) / N;
        const double d = width > 0 ? .5 * (1 - std::tanh((std::abs(x - .5) - .25) / (width / N)))
                                   : (std::abs(x - .5) < .25 ? 1.0 : 0.0);
        r[std::size_t(i)] = 1 / ratio + (1 - 1 / ratio) * d;
        u[std::size_t(i)] = uu;
        p[std::size_t(i)] = 1;
    }
    Contact o;
    o.U = make<float>(r, u, p);
    Cfg<float> c;
    c.h = 1.0 / N;
    c.base = base;
    c.mon = [&](const State<float>& V, double, long) {
        for (int i = 0; i < N; ++i)
            o.peak = std::max(o.peak, std::abs(double(pressureOf(V.q[0][std::size_t(i)], V.q[1][std::size_t(i)], V.q[2][std::size_t(i)])) - 1));
    };
    o.rep = run(o.U, .5 / std::max(uu, 1.0), c);
    return o;
}

int contactGate()
{
    for (double w : {0.0, 2.0}) {
        const Contact h = contact(0.02, w, 200, Base::Hybrid), k = contact(0.02, w, 200, Base::Kg);
        std::printf("  M 0.02 width %g peak %.3e sensed cell-steps %ld\n", w, h.peak, h.rep.sensedCells);
        check(h.rep.ok && h.rep.sensedCells == 0, std::format("M 0.02 width {}: 0 sensed", w));
        bool same = true;
        for (std::size_t v = 0; v < 3; ++v) same = same && h.U.q[v] == k.U.q[v];
        check(same, std::format("M 0.02 width {}: equal to KG bitwise", w));
    }
    const struct
    {
        double M, band;
    } rows[] = {{1, 0.26}, {3, 0.66}}; // measured 0.225, 0.593
    for (const auto& a : rows) {
        const Contact h = contact(a.M, 0, 200, Base::Hybrid);
        std::printf("  M %g sharp peak %.3f %s\n", a.M, h.peak, h.rep.why.c_str());
        check(h.rep.ok, std::format("M {} sharp: no failure", a.M));
        check(h.peak <= a.band, std::format("M {} sharp {:.3f} <= {:.3g}", a.M, h.peak, a.band));
    }
    return dnstest::finish();
}

// spec.py's rig: rows held, or a mirror at the x-hi end (4 solid cells).
struct SpecIc
{
    const char* name;
    double T;
    bool wall;
    std::function<void(double, double&, double&, double&, double&)> ic;
};

std::vector<SpecIc> specCases()
{
    const RH a = rankine(3.0);
    const double c1 = std::sqrt(1.4);
    auto ms3 = [a](double x, double y, double& r, double& u, double& p, double& Y) {
        const bool m = x < .1;
        r = (m ? a.r2 : 1.0) * (1 + .5 * (x > .5));
        u = m ? a.u2 : 0;
        p = m ? a.p2 : 1;
        Y = y;
    };
    return {
        {"stepwall", 1.1 / (3 * c1), true, [=](double x, double& r, double& u, double& p, double& Y) { ms3(x, x < .5, r, u, p, Y); }},
        {"slab", .75 / (3 * c1), false, [=](double x, double& r, double& u, double& p, double& Y) { ms3(x, x > .4 && x < .6, r, u, p, Y); }},
        {"t123", .15, false, [](double x, double& r, double& u, double& p, double& Y) { r = 1; u = x < .5 ? -2 : 2; p = .4; Y = x < .5; }},
        {"u5", .05, false, [](double x, double& r, double& u, double& p, double& Y) { r = 1; u = x < .5 ? -5 : 5; p = .4; Y = x < .5; }},
        {"blast", .012, false, [](double x, double& r, double& u, double& p, double& Y) { r = 1; u = 0; p = x < .5 ? 1000 : .01; Y = x < .5; }},
        {"coll", .03, false, [](double x, double& r, double& u, double& p, double& Y) { r = 1; u = x < .5 ? 8 : -8; p = 1; Y = x < .5; }},
        {"pistonflip", .03, false, [](double x, double& r, double& u, double& p, double& Y) { r = x < .5 ? 1 : .2; u = x < .5 ? .2 : -3; p = x < .5 ? 1 : 12; Y = x < .5; }},
        {"thinslab", .75 / (3 * c1), false, [=](double x, double& r, double& u, double& p, double& Y) { ms3(x, x > .5 && x < .52, r, u, p, Y); }},
    };
}

struct SpecOut
{
    double lo = 1e9, hi = -1e9;
    Report rep;
    State<float> U;
};

SpecOut specRun(const SpecIc& s, int N, Species mode)
{
    std::vector<double> r(static_cast<std::size_t>(N)), u(r), p(r), Y(r);
    for (int i = 0; i < N; ++i) {
        double a, b, c2, d;
        s.ic((i + .5) / N, a, b, c2, d);
        r[std::size_t(i)] = a;
        u[std::size_t(i)] = b;
        p[std::size_t(i)] = c2;
        Y[std::size_t(i)] = d;
    }
    SpecOut o;
    o.U = make<float>(r, u, p, &Y);
    const State<float> U0 = o.U;
    Cfg<float> c;
    c.h = 1.0 / N;
    c.species = mode;
    c.sensed0.assign(std::size_t(N), 0);
    for (int i = 0; i < 6; ++i) c.sensed0[std::size_t(i)] = 1;
    if (s.wall) {
        c.solid.assign(std::size_t(N), 0);
        for (int i = 0; i < 4; ++i) c.solid[std::size_t(N - 1 - i)] = 1;
    } else
        for (int i = 0; i < 6; ++i) c.sensed0[std::size_t(N - 1 - i)] = 1;
    const bool wall = s.wall;
    c.hook = [&U0, N, wall](State<float>& V) {
        for (std::size_t v = 0; v < 4; ++v)
            for (int i = 0; i < 4; ++i) {
                V.q[v][std::size_t(i)] = U0.q[v][std::size_t(i)];
                if (!wall) V.q[v][std::size_t(N - 1 - i)] = U0.q[v][std::size_t(N - 1 - i)];
                else V.q[v][std::size_t(N - 4 + i)] = V.q[v][std::size_t(N - 5 - i)] * (v == 1 ? -1.0f : 1.0f);
            }
    };
    c.mon = [&](const State<float>& V, double, long) {
        for (int i = 6; i < N - 6; ++i) {
            const double y = V.q[3][std::size_t(i)] / V.q[0][std::size_t(i)];
            o.lo = std::min(o.lo, y);
            o.hi = std::max(o.hi, y);
        }
    };
    o.rep = run(o.U, s.T, c);
    return o;
}

int species()
{
    double worstHi = -1e9, worstLo = 1e9, reuseLo = 1e9, reuseHi = -1e9;
    for (const SpecIc& s : specCases()) {
        const SpecOut d = specRun(s, 400, Species::Donor);
        const SpecOut r = specRun(s, 400, Species::Reuse);
        std::printf("  %-10s donor [%.3e, 1%+.3e] reuse [%.3e, 1%+.3e] %s\n", s.name, d.lo, d.hi - 1, r.lo, r.hi - 1, d.rep.why.c_str());
        check(d.rep.ok, std::format("{} marches", s.name));
        check(d.lo >= -1e-6 && d.hi <= 1 + 1e-6, std::format("{}: donor Y in [-1e-6, 1 + 1e-6]", s.name));
        worstHi = std::max(worstHi, d.hi);
        worstLo = std::min(worstLo, d.lo);
        if (r.rep.ok) {
            reuseLo = std::min(reuseLo, r.lo);
            reuseHi = std::max(reuseHi, r.hi);
        }
    }
    std::printf("  worst donor Ymin %.3e Ymax-1 %+.3e; reuse Ymin %.3e Ymax-1 %+.3e\n", worstLo, worstHi - 1, reuseLo, reuseHi - 1);
    check(reuseLo < -1e-2 || reuseHi > 1.01, "teeth: D Y_f(sign m_KG) leaves [-1e-2, 1.01]");
    // Uniform Y = 2^-3 through a Mach 3 shock into a density step: 0 bit mismatches.
    const SpecIc uni{"uniform", .5 / (3 * std::sqrt(1.4)), false, [](double x, double& r, double& u, double& p, double& Y) {
                         const RH a = rankine(3.0);
                         const bool m = x < .1;
                         r = (m ? a.r2 : 1.0) * (1 + .5 * (x > .5));
                         u = m ? a.u2 : 0;
                         p = m ? a.p2 : 1;
                         Y = .125;
                     }};
    const SpecOut d = specRun(uni, 400, Species::Donor);
    long bad = 0;
    for (int i = 6; i < 394; ++i)
        if (d.U.q[3][std::size_t(i)] != 0.125f * d.U.q[0][std::size_t(i)]) ++bad;
    std::printf("  uniform Y 2^-3: %ld bit mismatches over %ld steps\n", bad, d.rep.steps);
    check(d.rep.ok && bad == 0, "uniform Y 2^-3: 0 mismatches");
    return dnstest::finish();
}

int wall()
{
    for (double Ms : {1.5, 3.0, 6.0}) {
        const int N = 400, nw = N - 4;
        const RH a = rankine(Ms);
        const Reflect e = reflected(Ms);
        std::vector<double> r(static_cast<std::size_t>(N)), u(r), p(r);
        for (int i = 0; i < N; ++i) {
            const bool m = (i + .5) / N < .5;
            r[std::size_t(i)] = m ? a.r2 : 1;
            u[std::size_t(i)] = m ? a.u2 : 0;
            p[std::size_t(i)] = m ? a.p2 : 1;
        }
        State<float> U = make<float>(r, u, p);
        const State<float> U0 = U;
        Cfg<float> c;
        c.h = 1.0 / N;
        c.solid.assign(std::size_t(N), 0);
        c.sensed0.assign(std::size_t(N), 0);
        for (int i = nw; i < N; ++i) c.solid[std::size_t(i)] = 1;
        for (int i = 0; i < 6; ++i) c.sensed0[std::size_t(i)] = 1;
        c.hook = [&](State<float>& V) {
            for (std::size_t v = 0; v < 3; ++v)
                for (int i = 0; i < 4; ++i) {
                    V.q[v][std::size_t(i)] = U0.q[v][std::size_t(i)];
                    V.q[v][std::size_t(nw + i)] = V.q[v][std::size_t(nw - 1 - i)] * (v == 1 ? -1.0f : 1.0f);
                }
        };
        const double T = (double(nw) / N - .5) / a.S + .25 / e.w;
        const Report rep = run(U, T, c);
        double sp = 0, cnt = 0;
        for (int i = 0; i < N; ++i) {
            const double x = (i + .5) / N;
            if (!(x > double(nw) / N - .12 && x < double(nw) / N - .03)) continue;
            sp += pressureOf(U.q[0][std::size_t(i)], U.q[1][std::size_t(i)], U.q[2][std::size_t(i)]);
            cnt += 1;
        }
        const double pl = sp / cnt / e.p5 - 1;
        const double pw = pressureOf(U.q[0][std::size_t(nw - 1)], U.q[1][std::size_t(nw - 1)], U.q[2][std::size_t(nw - 1)]);
        const double Tw = (pw / U.q[0][std::size_t(nw - 1)]) / (e.p5 / e.r5);
        std::printf("  Ms %g plateau %+.2e wall T/T5 %.3f wall rho/rho5 %.3f %s\n", Ms, pl, Tw, U.q[0][std::size_t(nw - 1)] / e.r5, rep.why.c_str());
        check(rep.ok, std::format("Ms {} marches", Ms));
        check(std::abs(pl) <= 1.2e-3, std::format("Ms {} plateau {:.2e} <= 1.2e-3", Ms, std::abs(pl)));
        check(Tw >= 0.96, std::format("Ms {} wall T/T5 {:.3f} >= 0.96", Ms, Tw));
    }
    return dnstest::finish();
}

} // namespace

// G8: the FV face terms, dns/fvflux.hpp fvShockFlux, fvD4Rate, fvD4Row.
template <class T>
void fvFor(const char* tag, double tol)
{
    using namespace dns;
    std::mt19937_64 gen(5);
    std::uniform_real_distribution<double> d(0.2, 3.0), v(-2.0, 2.0);
    const double g = 1.4;
    double worst = 0, zero = 0, mirror = 0;
    for (int i = 0; i < 20000; ++i) {
        const FvPrim<T> A = fvPrimOf<T>(T(d(gen)), T(v(gen)), T(v(gen)), T(v(gen)), T(5 + d(gen)), g, 1.0);
        const FvPrim<T> B = fvPrimOf<T>(T(d(gen)), T(v(gen)), T(v(gen)), T(v(gen)), T(5 + d(gen)), g, 1.0);
        const double S[3] = {v(gen), v(gen), v(gen)}, Sm[3] = {-S[0], -S[1], -S[2]};
        const double mag = std::sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]);
        T o[5], om[5];
        fvShockFlux(T(1), A, B, S, g, o);
        const T gu[3][3] = {}, gT[3] = {};
        const FvFlux<T> kg = fvMeanFlux(A, B, S, gu, gT, T(0), T(0));
        const double Sv[3] = {S[0], S[1], S[2]};
        // sigma 1: F_KG + D is HLL through S, written out here with Davis speeds.
        const auto un = [&](const FvPrim<T>& q) { return double(q.u[0]) * Sv[0] + double(q.u[1]) * Sv[1] + double(q.u[2]) * Sv[2]; };
        const double cL = std::sqrt(g * double(A.p) / double(A.rho)) * mag, cR = std::sqrt(g * double(B.p) / double(B.rho)) * mag;
        const double uL = un(A), uR = un(B);
        const double ap = std::max({uL + cL, uR + cR, 0.0}), am = std::min({uL - cL, uR - cR, 0.0});
        const auto hll = [&](double FL, double FR, double UL, double UR) { return (ap * FL - am * FR + ap * am * (UR - UL)) / (ap - am); };
        double e[5];
        e[0] = hll(double(A.rho) * uL, double(B.rho) * uR, double(A.rho), double(B.rho));
        for (int k = 0; k < 3; ++k)
            e[1 + k] = hll(double(A.rho) * uL * double(A.u[k]) + double(A.p) * Sv[k], double(B.rho) * uR * double(B.u[k]) + double(B.p) * Sv[k],
                           double(A.rho) * double(A.u[k]), double(B.rho) * double(B.u[k]));
        e[4] = hll(double(A.rho) * uL * double(A.e) + double(A.p) * uL, double(B.rho) * uR * double(B.e) + double(B.p) * uR,
                   double(A.rho) * double(A.e), double(B.rho) * double(B.e));
        const double got[5] = {double(kg.rho) + double(o[0]), double(kg.m[0]) + double(o[1]), double(kg.m[1]) + double(o[2]),
                               double(kg.m[2]) + double(o[3]), double(kg.E) + double(o[4])};
        for (int k = 0; k < 5; ++k) worst = std::max(worst, std::abs(got[k] - e[k]) / (mag * 20.0));
        T same[5];
        fvShockFlux(T(1), A, A, S, g, same);
        for (T x : same) zero = std::max(zero, double(std::abs(x)));
        fvShockFlux(T(1), B, A, Sm, g, om);
        for (int k = 0; k < 5; ++k) mirror = std::max(mirror, std::abs(double(o[k]) + double(om[k])) / (mag * 20.0));
    }
    std::printf("  %s: sigma 1 vs HLL %.2e, L == R %.2e, mirror %.2e\n", tag, worst, zero, mirror);
    check(worst <= tol, std::format("{}: sigma 1 sum == HLL", tag));
    check(zero == 0.0, std::format("{}: L == R writes exactly 0", tag));
    check(mirror <= tol, std::format("{}: a mirror negates", tag));
    // sigma 1 writes no D4; a uniform field's third difference is exactly 0.
    const FvPrim<T> A = fvPrimOf<T>(T(1.1), T(0.3), T(0.2), T(0), T(4.0), g, 1.0);
    const double S[3] = {0.5, 0.25, 0.0};
    check(fvD4Rate(T(0.0625), T(1), A, A, S) == T(0), std::format("{}: sigma 1 writes no D4", tag));
    check(fvD4Rate(T(0.0625), T(0), A, A, S) > T(0), std::format("{}: sigma 0 does", tag));
    long bad = 0;
    for (int i = 0; i < 20000; ++i) {
        const T x = T(d(gen) * std::pow(10.0, v(gen)));
        bad += fvD4Row(x, x, x, x) != T(0);
    }
    check(bad == 0, std::format("{}: a uniform field's D4 row is exactly 0", tag));
    long ung = 0;
    for (int i = 0; i < 20000; ++i) { // why the grouping: the plain alternating form leaves ulps
        const T x = T(d(gen) * std::pow(10.0, v(gen)));
        ung += (x - T(3) * x + T(3) * x - x) != T(0);
    }
    std::printf("  %s: ungrouped form leaves ulps on %ld of 20000 uniform fields\n", tag, ung);
    check(ung > 0, std::format("{}: control: the ungrouped form leaves ulps", tag));
}

int fv()
{
    fvFor<double>("FP64", 1e-14);
    fvFor<float>("FP32", 1e-5);
    return dnstest::finish();
}

int main(int argc, char** argv)
{
    const std::string_view m = argc > 1 ? argv[1] : "unit";
    if (m == "fv") return fv();
    if (m == "unit") return unit();
    if (m == "riemann") return riemann();
    if (m == "shuosher") return shuosher();
    if (m == "nozzle") return nozzle();
    if (m == "weak") return weak();
    if (m == "contact") return contactGate();
    if (m == "species") return species();
    if (m == "wall") return wall();
    std::printf("unknown mode %s\n", argv[1]);
    return 2;
}
