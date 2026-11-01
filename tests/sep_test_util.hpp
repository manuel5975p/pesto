// Host FP64 restatement of the separable Poisson problem (dns/septransform.hpp):
// the 7-point operator with per-axis boundary kinds, the fast line transforms
// emulated with a naive DFT, and the separable solve built from them. Shared by
// the host test `poisson_sep` and the device tests in test_gpu_poisson_sep.
#pragma once

#include "dns/septransform.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>

namespace septest {

using dns::SepKind;

struct Box
{
    int n[3];
    double h[3];
    SepKind kind[3];
    std::size_t size() const { return std::size_t(n[0]) * n[1] * n[2]; }
    std::size_t idx(int i, int j, int k) const
    {
        return std::size_t(i) + std::size_t(n[0]) * (std::size_t(j) + std::size_t(n[1]) * k);
    }
    bool singular() const
    {
        // nz == 1 contributes no z term (2D): only x and y decide.
        for (int a = 0; a < 3; ++a)
            if (n[a] > 1 && !dns::sepHasZeroMode(kind[a])) return false;
        return true;
    }
};

inline constexpr std::array<SepKind, 4> kAllKinds{SepKind::Periodic, SepKind::NN,
                                                  SepKind::DD, SepKind::ND};

// Neighbour value along axis a from cell q (index along a) with the kind's ghost
// rule; `line(i)` reads index i of the line. n == 1: no stencil (2D).
template <class Get>
double ghostAt(SepKind kind, int n, int q, Get line)
{
    if (q >= 0 && q < n) return line(q);
    if (kind == SepKind::Periodic) return line(q < 0 ? q + n : q - n);
    const bool lo = q < 0;
    const double s = lo ? (dns::sepLoDirichlet(kind) ? -1.0 : 1.0)
                        : (dns::sepHiDirichlet(kind) ? -1.0 : 1.0);
    return s * line(lo ? 0 : n - 1);
}

// out = L p (the FD Laplacian, negative semi-definite).
inline std::vector<double> laplacian(const Box& b, const std::vector<double>& p)
{
    std::vector<double> out(b.size(), 0.0);
    for (int k = 0; k < b.n[2]; ++k)
        for (int j = 0; j < b.n[1]; ++j)
            for (int i = 0; i < b.n[0]; ++i)
            {
                const int c3[3] = {i, j, k};
                const double pc = p[b.idx(i, j, k)];
                double s = 0.0;
                for (int a = 0; a < 3; ++a)
                {
                    if (b.n[a] == 1) continue;
                    auto line = [&](int q) {
                        int d[3] = {i, j, k};
                        d[a] = q;
                        return p[b.idx(d[0], d[1], d[2])];
                    };
                    const double lo = ghostAt(b.kind[a], b.n[a], c3[a] - 1, line);
                    const double hi = ghostAt(b.kind[a], b.n[a], c3[a] + 1, line);
                    s += (lo - 2.0 * pc + hi) / (b.h[a] * b.h[a]);
                }
                out[b.idx(i, j, k)] = s;
            }
    return out;
}

// Naive DFT, sign -1 (FORWARD) or +1 (BACKWARD), unnormalised.
inline std::vector<std::complex<double>> dft(const std::vector<std::complex<double>>& v, int sign)
{
    const int n = int(v.size());
    std::vector<std::complex<double>> o(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k)
    {
        std::complex<double> s{0.0, 0.0};
        for (int j = 0; j < n; ++j)
        {
            const double t = sign * 2.0 * std::numbers::pi * double((long(j) * k) % n) / n;
            s += v[std::size_t(j)] * std::complex<double>(std::cos(t), std::sin(t));
        }
        o[std::size_t(k)] = s;
    }
    return o;
}

using dns::SepC;

// Fast forward line transform (the device algorithm, FP64, naive DFT).
inline void put(std::vector<double>& out, const dns::SepOut<double>& o)
{
    out[std::size_t(o.d0)] = o.v0;
    if (o.d1 >= 0) out[std::size_t(o.d1)] = o.v1;
}

inline std::vector<double> fwdFast(SepKind kind, const std::vector<double>& x)
{
    const int n = int(x.size()), L = dns::sepLineLen(kind, n);
    auto get = [&](int i) { return x[std::size_t(i)]; };
    std::vector<std::complex<double>> v(static_cast<std::size_t>(L));
    for (int j = 0; j < L; ++j)
    {
        const SepC<double> r = dns::sepFwdPre<double>(kind, j, n, get, dns::sepTwPre(kind, n, j));
        v[std::size_t(j)] = {r.re, r.im};
    }
    const auto V = dft(v, -1);
    std::vector<double> c(static_cast<std::size_t>(n));
    for (int m = 0; m < L; ++m)
    {
        const SepC<double> Vm{V[std::size_t(m)].real(), V[std::size_t(m)].imag()};
        put(c, dns::sepFwdPost(kind, m, n, Vm, dns::sepTwPost(kind, n, m)));
    }
    return c;
}

// Fast inverse line transform: returns x / sepNorm.
inline std::vector<double> invFast(SepKind kind, const std::vector<double>& c)
{
    const int n = int(c.size()), L = dns::sepLineLen(kind, n);
    std::vector<std::complex<double>> v(static_cast<std::size_t>(L));
    auto get = [&](int i) { return c[std::size_t(i)]; };
    for (int k = 0; k < L; ++k)
    {
        const SepC<double> r = dns::sepInvPre<double>(kind, k, n, get, dns::sepTwPre(kind, n, k),
                                                      dns::sepTwPost(kind, n, k));
        v[std::size_t(k)] = {r.re, r.im};
    }
    const auto V = dft(v, dns::sepInvForward(kind) ? -1 : 1);
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int j = 0; j < L; ++j)
    {
        const SepC<double> Vj{V[std::size_t(j)].real(), V[std::size_t(j)].imag()};
        put(y, dns::sepInvPost(kind, j, n, Vj, dns::sepTwPost(kind, n, j)));
    }
    return y;
}

// Two lines through ONE complex FFT (dns::SepPack, the device packing):
// returns {pass(a), pass(b)} where pass = fwdFast or invFast. Single kinds run
// the plain pass twice.
inline std::array<std::vector<double>, 2> pairFast(SepKind kind, bool inverse, const std::vector<double>& a,
                                                   const std::vector<double>& b)
{
    const dns::SepPack pk = dns::sepPack(kind, inverse);
    if (pk == dns::SepPack::Single)
        return {inverse ? invFast(kind, a) : fwdFast(kind, a), inverse ? invFast(kind, b) : fwdFast(kind, b)};
    const int n = int(a.size()), L = dns::sepLineLen(kind, n);
    auto ga = [&](int i) { return a[std::size_t(i)]; };
    auto gb = [&](int i) { return b[std::size_t(i)]; };
    std::vector<std::complex<double>> v(static_cast<std::size_t>(L));
    for (int j = 0; j < L; ++j)
    {
        const auto w = dns::sepTwPre(kind, n, j), t = dns::sepTwPost(kind, n, j);
        SepC<double> pa, pb;
        if (inverse)
        {
            pa = dns::sepInvPre<double>(kind, j, n, ga, w, t);
            pb = dns::sepInvPre<double>(kind, j, n, gb, w, t);
        }
        else
        {
            pa = dns::sepFwdPre<double>(kind, j, n, ga, w);
            pb = dns::sepFwdPre<double>(kind, j, n, gb, w);
        }
        const SepC<double> z = pk == dns::SepPack::RealIn ? SepC<double>{pa.re, pb.re} : dns::sepJoin(pa, pb);
        v[std::size_t(j)] = {z.re, z.im};
    }
    const auto Z = dft(v, inverse && !dns::sepInvForward(kind) ? 1 : -1);
    std::array<std::vector<double>, 2> out{std::vector<double>(std::size_t(n)), std::vector<double>(std::size_t(n))};
    for (int m = 0; m < L; ++m)
    {
        const auto t = dns::sepTwPost(kind, n, m);
        const SepC<double> zk{Z[std::size_t(m)].real(), Z[std::size_t(m)].imag()};
        const auto& zr = Z[std::size_t(m == 0 ? 0 : L - m)];
        const SepC<double> zl{zr.real(), zr.imag()};
        for (int s = 0; s < 2; ++s)
        {
            const SepC<double> X = pk == dns::SepPack::RealIn ? dns::sepSplit(zk, zl, s == 1)
                                                              : SepC<double>{s ? zk.im : zk.re, 0.0};
            put(out[std::size_t(s)], inverse ? dns::sepInvPost(kind, m, n, X, t) : dns::sepFwdPost(kind, m, n, X, t));
        }
    }
    return out;
}

// Apply a line transform along axis a of a 3D array (n[a] == 1: identity).
template <class F>
void alongAxis(const Box& b, std::vector<double>& f, int a, F tr)
{
    if (b.n[a] == 1) return;
    const int o1 = (a + 1) % 3, o2 = (a + 2) % 3;
    std::vector<double> line(static_cast<std::size_t>(b.n[a]));
    for (int q2 = 0; q2 < b.n[o2]; ++q2)
        for (int q1 = 0; q1 < b.n[o1]; ++q1)
        {
            int d[3];
            d[o1] = q1;
            d[o2] = q2;
            for (int q = 0; q < b.n[a]; ++q)
            {
                d[a] = q;
                line[std::size_t(q)] = f[b.idx(d[0], d[1], d[2])];
            }
            const auto out = tr(b.kind[a], line);
            for (int q = 0; q < b.n[a]; ++q)
            {
                d[a] = q;
                f[b.idx(d[0], d[1], d[2])] = out[std::size_t(q)];
            }
        }
}

// Separable solve L p = rhs in FP64 through the fast transforms (zero mode
// dropped when singular: the zero-mean solution).
inline std::vector<double> sepSolve(const Box& b, const std::vector<double>& rhs)
{
    std::vector<double> f = rhs;
    for (int a = 0; a < 3; ++a) alongAxis(b, f, a, fwdFast);
    double norm = 1.0;
    for (int a = 0; a < 3; ++a)
        if (b.n[a] > 1) norm *= dns::sepNorm(b.kind[a], b.n[a]);
    for (int k = 0; k < b.n[2]; ++k)
        for (int j = 0; j < b.n[1]; ++j)
            for (int i = 0; i < b.n[0]; ++i)
            {
                const int q[3] = {i, j, k};
                double lam = 0.0;
                for (int a = 0; a < 3; ++a)
                    if (b.n[a] > 1) lam += dns::sepEigen(b.kind[a], b.n[a], b.h[a], q[a]);
                double& v = f[b.idx(i, j, k)];
                v = lam == 0.0 ? 0.0 : -v * norm / lam;
            }
    for (int a = 0; a < 3; ++a) alongAxis(b, f, a, invFast);
    return f;
}

} // namespace septest
