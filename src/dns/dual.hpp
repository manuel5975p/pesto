// Forward-mode dual numbers: a value and N directional derivatives carried
// through ordinary arithmetic. Host- and device-compilable (DNS_HD), FP64.
// The flow Newton-Krylov preconditioner differentiates its flux with these
// (docs/rans.md, "Newton-Krylov pseudo-time"): exact to rounding, never a
// finite-difference window (the channel's FP32-FD lesson).
#pragma once

#include "dns/hd.hpp"

#include <cmath>

namespace dns {

template <int N> struct Dual
{
    double v = 0;
    double d[N] = {};

    DNS_HD Dual() = default;
    // A constant: zero derivatives.
    DNS_HD Dual(double x) : v(x) {}
    // The k-th independent variable: d/dx_k = 1.
    static DNS_HD Dual var(double x, int k)
    {
        Dual r(x);
        r.d[k] = 1;
        return r;
    }
    // A hidden friend (ADL only), so it never hides ::sqrt(double) for
    // other code in namespace dns.
    friend DNS_HD Dual sqrt(const Dual& a)
    {
        Dual r(::sqrt(a.v));
        const double h = r.v > 0 ? 0.5 / r.v : 0.0;
        for (int k = 0; k < N; ++k) r.d[k] = a.d[k] * h;
        return r;
    }
    // a^b for a constant exponent (the FV characteristic far state's
    // isentropic ratios, dns/fvflux.hpp). Pre: a.v > 0.
    friend DNS_HD Dual pow(const Dual& a, double b)
    {
        Dual r(::pow(a.v, b));
        const double h = b * r.v / a.v;
        for (int k = 0; k < N; ++k) r.d[k] = a.d[k] * h;
        return r;
    }
};

template <int N> DNS_HD inline Dual<N> operator+(Dual<N> a, const Dual<N>& b)
{
    a.v += b.v;
    for (int k = 0; k < N; ++k) a.d[k] += b.d[k];
    return a;
}
template <int N> DNS_HD inline Dual<N> operator-(Dual<N> a, const Dual<N>& b)
{
    a.v -= b.v;
    for (int k = 0; k < N; ++k) a.d[k] -= b.d[k];
    return a;
}
template <int N> DNS_HD inline Dual<N> operator-(Dual<N> a)
{
    a.v = -a.v;
    for (int k = 0; k < N; ++k) a.d[k] = -a.d[k];
    return a;
}
template <int N>
DNS_HD inline Dual<N> operator*(const Dual<N>& a, const Dual<N>& b)
{
    Dual<N> r(a.v * b.v);
    for (int k = 0; k < N; ++k) r.d[k] = a.d[k] * b.v + a.v * b.d[k];
    return r;
}
template <int N>
DNS_HD inline Dual<N> operator/(const Dual<N>& a, const Dual<N>& b)
{
    const double ib = 1.0 / b.v;
    Dual<N> r(a.v * ib);
    for (int k = 0; k < N; ++k) r.d[k] = (a.d[k] - r.v * b.d[k]) * ib;
    return r;
}
template <int N> DNS_HD inline Dual<N> operator+(Dual<N> a, double b)
{
    a.v += b;
    return a;
}
template <int N> DNS_HD inline Dual<N> operator+(double b, Dual<N> a)
{
    a.v += b;
    return a;
}
template <int N> DNS_HD inline Dual<N> operator-(Dual<N> a, double b)
{
    a.v -= b;
    return a;
}
template <int N> DNS_HD inline Dual<N> operator-(double b, const Dual<N>& a)
{
    return Dual<N>(b) - a;
}
template <int N> DNS_HD inline Dual<N> operator*(Dual<N> a, double b)
{
    a.v *= b;
    for (int k = 0; k < N; ++k) a.d[k] *= b;
    return a;
}
template <int N> DNS_HD inline Dual<N> operator*(double b, Dual<N> a)
{
    return a * b;
}
template <int N> DNS_HD inline Dual<N> operator/(const Dual<N>& a, double b)
{
    return a * (1.0 / b);
}

// The value of a plain double or a dual (frozen coefficients read this).
DNS_HD inline double value(double x) { return x; }
template <int N> DNS_HD inline double value(const Dual<N>& x) { return x.v; }

} // namespace dns
