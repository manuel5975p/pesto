#include "dns/prop/numeric.hpp"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <limits>

namespace dns::prop::num {
namespace {

// numpy's pairwise_sum (loops_utils.h.src), PW_BLOCKSIZE 128.
double pairwise(const double* a, std::size_t n)
{
    if (n < 8)
    {
        double res = -0.0;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128)
    {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; ++k) r[k] += a[i + std::size_t(k)];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

// --- MINPACK (hybrd and its helpers), 0-based. R is the upper triangle packed
// by rows; element (i, j), i <= j, sits at packedIndex(n, i, j).

constexpr double kEpsMch = DBL_EPSILON;

std::size_t packedIndex(int n, int i, int j)
{
    return std::size_t(i) * std::size_t(n) - std::size_t(i) * std::size_t(i - 1) / 2
           + std::size_t(j - i);
}

double enorm(int n, const double* x)
{
    constexpr double rdwarf = 3.834e-20, rgiant = 1.304e19;
    double s1 = 0, s2 = 0, s3 = 0, x1max = 0, x3max = 0;
    const double agiant = rgiant / double(n);
    for (int i = 0; i < n; ++i)
    {
        const double xabs = std::fabs(x[i]);
        if (xabs > rdwarf && xabs < agiant) { s2 += xabs * xabs; continue; }
        if (xabs <= rdwarf)
        {
            if (xabs > x3max)
            {
                const double q = x3max / xabs;
                s3 = 1.0 + s3 * q * q;
                x3max = xabs;
            }
            else if (xabs != 0.0)
            {
                const double q = xabs / x3max;
                s3 += q * q;
            }
            continue;
        }
        if (xabs > x1max)
        {
            const double q = x1max / xabs;
            s1 = 1.0 + s1 * q * q;
            x1max = xabs;
        }
        else
        {
            const double q = xabs / x1max;
            s1 += q * q;
        }
    }
    if (s1 != 0.0) return x1max * std::sqrt(s1 + (s2 / x1max) / x1max);
    if (s2 != 0.0)
    {
        if (s2 >= x3max) return std::sqrt(s2 * (1.0 + (x3max / s2) * (x3max * s3)));
        return std::sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
    }
    return x3max * std::sqrt(s3);
}

// Column-major n x n.
struct ColMajor
{
    int n;
    std::vector<double> a;
    double& operator()(int i, int j) { return a[std::size_t(i) + std::size_t(j) * std::size_t(n)]; }
    double* col(int j) { return a.data() + std::size_t(j) * std::size_t(n); }
};

// qrfac without pivoting: Householder QR in place; rdiag = diag(R),
// acnorm = column norms of the input.
void qrfac(ColMajor& q, std::vector<double>& rdiag, std::vector<double>& acnorm)
{
    const int n = q.n;
    for (int j = 0; j < n; ++j)
    {
        acnorm[std::size_t(j)] = enorm(n, q.col(j));
        rdiag[std::size_t(j)] = acnorm[std::size_t(j)];
    }
    for (int j = 0; j < n; ++j)
    {
        double ajnorm = enorm(n - j, &q(j, j));
        if (ajnorm != 0.0)
        {
            if (q(j, j) < 0.0) ajnorm = -ajnorm;
            for (int i = j; i < n; ++i) q(i, j) /= ajnorm;
            q(j, j) += 1.0;
            for (int k = j + 1; k < n; ++k)
            {
                double sum = 0.0;
                for (int i = j; i < n; ++i) sum += q(i, j) * q(i, k);
                const double temp = sum / q(j, j);
                for (int i = j; i < n; ++i) q(i, k) -= temp * q(i, j);
            }
        }
        rdiag[std::size_t(j)] = -ajnorm;
    }
}

// qform: accumulate Q from qrfac's Householder vectors (square case).
void qform(ColMajor& q, std::vector<double>& wa)
{
    const int n = q.n;
    for (int j = 1; j < n; ++j)
        for (int i = 0; i < j; ++i) q(i, j) = 0.0;
    for (int l = 0; l < n; ++l)
    {
        const int k = n - 1 - l;
        for (int i = k; i < n; ++i)
        {
            wa[std::size_t(i)] = q(i, k);
            q(i, k) = 0.0;
        }
        q(k, k) = 1.0;
        if (wa[std::size_t(k)] == 0.0) continue;
        for (int j = k; j < n; ++j)
        {
            double sum = 0.0;
            for (int i = k; i < n; ++i) sum += q(i, j) * wa[std::size_t(i)];
            const double temp = sum / wa[std::size_t(k)];
            for (int i = k; i < n; ++i) q(i, j) -= temp * wa[std::size_t(i)];
        }
    }
}

void dogleg(int n, const std::vector<double>& r, const std::vector<double>& diag,
            const std::vector<double>& qtb, double delta, std::vector<double>& x,
            std::vector<double>& wa1, std::vector<double>& wa2)
{
    // Gauss-Newton direction.
    for (int j = n - 1; j >= 0; --j)
    {
        double sum = 0.0;
        for (int i = j + 1; i < n; ++i) sum += r[packedIndex(n, j, i)] * x[std::size_t(i)];
        double temp = r[packedIndex(n, j, j)];
        if (temp == 0.0)
        {
            for (int i = 0; i <= j; ++i)
                temp = std::max(temp, std::fabs(r[packedIndex(n, i, j)]));
            temp = kEpsMch * temp;
            if (temp == 0.0) temp = kEpsMch;
        }
        x[std::size_t(j)] = (qtb[std::size_t(j)] - sum) / temp;
    }
    for (int j = 0; j < n; ++j)
    {
        wa1[std::size_t(j)] = 0.0;
        wa2[std::size_t(j)] = diag[std::size_t(j)] * x[std::size_t(j)];
    }
    const double qnorm = enorm(n, wa2.data());
    if (qnorm <= delta) return;

    // Scaled gradient direction.
    for (int j = 0; j < n; ++j)
    {
        const double temp = qtb[std::size_t(j)];
        for (int i = j; i < n; ++i) wa1[std::size_t(i)] += r[packedIndex(n, j, i)] * temp;
        wa1[std::size_t(j)] /= diag[std::size_t(j)];
    }
    const double gnorm = enorm(n, wa1.data());
    double sgnorm = 0.0;
    double alpha = delta / qnorm;
    if (gnorm != 0.0)
    {
        for (int j = 0; j < n; ++j)
            wa1[std::size_t(j)] = (wa1[std::size_t(j)] / gnorm) / diag[std::size_t(j)];
        for (int j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (int i = j; i < n; ++i) sum += r[packedIndex(n, j, i)] * wa1[std::size_t(i)];
            wa2[std::size_t(j)] = sum;
        }
        const double temp = enorm(n, wa2.data());
        sgnorm = (gnorm / temp) / temp;
        alpha = 0.0;
        if (sgnorm < delta)
        {
            const double bnorm = enorm(n, qtb.data());
            const double dq = delta / qnorm, sd = sgnorm / delta;
            double t = (bnorm / gnorm) * (bnorm / qnorm) * sd;
            t = t - dq * (sd * sd)
                + std::sqrt((t - dq) * (t - dq) + (1.0 - dq * dq) * (1.0 - sd * sd));
            alpha = (dq * (1.0 - sd * sd)) / t;
        }
    }
    const double temp = (1.0 - alpha) * std::min(sgnorm, delta);
    for (int j = 0; j < n; ++j)
        x[std::size_t(j)] = temp * wa1[std::size_t(j)] + alpha * x[std::size_t(j)];
}

// Givens rotation (cos, sin, tau) that zeroes `b` against `a`, MINPACK's form.
void givens(double a, double b, double& cs, double& sn, double& tau)
{
    if (std::fabs(a) < std::fabs(b))
    {
        const double cotan = a / b;
        sn = 0.5 / std::sqrt(0.25 + 0.25 * cotan * cotan);
        cs = sn * cotan;
        tau = 1.0;
        if (std::fabs(cs) * DBL_MAX > 1.0) tau = 1.0 / cs;
    }
    else
    {
        const double tn = b / a;
        cs = 0.5 / std::sqrt(0.25 + 0.25 * tn * tn);
        sn = cs * tn;
        tau = sn;
    }
}

// r1updt, square case: R + u v^T re-triangularized by Givens rotations.
void r1updt(int n, std::vector<double>& s, const std::vector<double>& u,
            std::vector<double>& v, std::vector<double>& w)
{
    w[std::size_t(n - 1)] = s[packedIndex(n, n - 1, n - 1)];
    for (int j = n - 2; j >= 0; --j)
    {
        w[std::size_t(j)] = 0.0;
        if (v[std::size_t(j)] == 0.0) continue;
        double cs, sn, tau;
        givens(v[std::size_t(n - 1)], v[std::size_t(j)], cs, sn, tau);
        v[std::size_t(n - 1)] = sn * v[std::size_t(j)] + cs * v[std::size_t(n - 1)];
        v[std::size_t(j)] = tau;
        for (int i = j; i < n; ++i)
        {
            double& sl = s[packedIndex(n, j, i)];
            const double temp = cs * sl - sn * w[std::size_t(i)];
            w[std::size_t(i)] = sn * sl + cs * w[std::size_t(i)];
            sl = temp;
        }
    }
    for (int i = 0; i < n; ++i) w[std::size_t(i)] += v[std::size_t(n - 1)] * u[std::size_t(i)];
    for (int j = 0; j < n - 1; ++j)
    {
        if (w[std::size_t(j)] != 0.0)
        {
            double cs, sn, tau;
            givens(s[packedIndex(n, j, j)], w[std::size_t(j)], cs, sn, tau);
            for (int i = j; i < n; ++i)
            {
                double& sl = s[packedIndex(n, j, i)];
                const double temp = cs * sl + sn * w[std::size_t(i)];
                w[std::size_t(i)] = -sn * sl + cs * w[std::size_t(i)];
                sl = temp;
            }
            w[std::size_t(j)] = tau;
        }
    }
    s[packedIndex(n, n - 1, n - 1)] = w[std::size_t(n - 1)];
}

// Recover (cos, sin) from a stored tau.
void untau(double t, double& cs, double& sn)
{
    if (std::fabs(t) > 1.0)
    {
        cs = 1.0 / t;
        sn = std::sqrt(1.0 - cs * cs);
    }
    else
    {
        sn = t;
        cs = std::sqrt(1.0 - sn * sn);
    }
}

// r1mpyq on an m x n column-major block (m rows; m = 1 is a row vector).
void r1mpyq(int m, int n, double* a, int lda, const std::vector<double>& v,
            const std::vector<double>& w)
{
    auto at = [&](int i, int j) -> double& {
        return a[std::size_t(i) + std::size_t(j) * std::size_t(lda)];
    };
    for (int j = n - 2; j >= 0; --j)
    {
        double cs, sn;
        untau(v[std::size_t(j)], cs, sn);
        for (int i = 0; i < m; ++i)
        {
            const double temp = cs * at(i, j) - sn * at(i, n - 1);
            at(i, n - 1) = sn * at(i, j) + cs * at(i, n - 1);
            at(i, j) = temp;
        }
    }
    for (int j = 0; j < n - 1; ++j)
    {
        double cs, sn;
        untau(w[std::size_t(j)], cs, sn);
        for (int i = 0; i < m; ++i)
        {
            const double temp = cs * at(i, j) + sn * at(i, n - 1);
            at(i, n - 1) = -sn * at(i, j) + cs * at(i, n - 1);
            at(i, j) = temp;
        }
    }
}

} // namespace

double interp(double x, std::span<const double> xp, std::span<const double> fp)
{
    assert(!xp.empty() && xp.size() == fp.size());
    if (std::isnan(x)) return x;
    const std::size_t n = xp.size();
    if (x < xp[0]) return fp[0];
    if (x > xp[n - 1]) return fp[n - 1];
    const std::size_t j =
        std::size_t(std::ranges::upper_bound(xp, x) - xp.begin()) - 1;
    if (j == n - 1 || xp[j] == x) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double res = slope * (x - xp[j]) + fp[j];
    if (std::isnan(res))
    {
        res = slope * (x - xp[j + 1]) + fp[j + 1];
        if (std::isnan(res) && fp[j] == fp[j + 1]) res = fp[j];
    }
    return res;
}

double mod(double a, double b)
{
    double m = std::fmod(a, b);
    if (m != 0.0)
    {
        if ((b < 0) != (m < 0)) m += b;
    }
    else
        m = std::copysign(0.0, b);
    return m;
}

std::vector<double> linspace(double a, double b, int n)
{
    assert(n >= 1);
    std::vector<double> y(static_cast<std::size_t>(n));
    if (n == 1) { y[0] = a; return y; }
    const double step = (b - a) / double(n - 1);
    for (int i = 0; i < n; ++i) y[std::size_t(i)] = double(i) * step + a;
    y.back() = b;
    return y;
}

std::vector<double> arange(double a, double b, double step)
{
    assert(step > 0);
    const double len = std::ceil((b - a) / step);
    if (!(len > 0)) return {};
    std::vector<double> y(static_cast<std::size_t>(len));
    y[0] = a;
    if (y.size() > 1) y[1] = a + step;
    const double delta = y.size() > 1 ? y[1] - a : step;
    for (std::size_t i = 2; i < y.size(); ++i) y[i] = a + double(i) * delta;
    return y;
}

double sum(std::span<const double> a) { return 0.0 + pairwise(a.data(), a.size()); }

double mean(std::span<const double> a)
{
    assert(!a.empty());
    return sum(a) / double(a.size());
}

double brentq(const std::function<double(double)>& f, double xa, double xb,
              double xtol, double rtol, int maxiter)
{
    double xpre = xa, xcur = xb, xblk = 0.0, fblk = 0.0, spre = 0.0, scur = 0.0;
    double fpre = f(xpre), fcur = f(xcur);
    if (fpre == 0.0) return xpre;
    if (fcur == 0.0) return xcur;
    assert(std::signbit(fpre) != std::signbit(fcur));
    for (int i = 0; i < maxiter; ++i)
    {
        if (fpre != 0.0 && fcur != 0.0 && std::signbit(fpre) != std::signbit(fcur))
        {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (std::fabs(fblk) < std::fabs(fcur))
        {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        const double delta = (xtol + rtol * std::fabs(xcur)) / 2;
        const double sbis = (xblk - xcur) / 2;
        if (fcur == 0.0 || std::fabs(sbis) < delta) return xcur;
        if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre))
        {
            double stry;
            if (xpre == xblk)
                stry = -fcur * (xcur - xpre) / (fcur - fpre);   // interpolate
            else
            {   // extrapolate
                const double dpre = (fpre - fcur) / (xpre - xcur);
                const double dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            if (2 * std::fabs(stry) < std::min(std::fabs(spre), 3 * std::fabs(sbis) - delta))
            {
                spre = scur;
                scur = stry;
            }
            else
                spre = scur = sbis;
        }
        else
            spre = scur = sbis;
        xpre = xcur;
        fpre = fcur;
        if (std::fabs(scur) > delta) xcur += scur;
        else xcur += (sbis > 0 ? delta : -delta);
        fcur = f(xcur);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

std::vector<double> matvec(const Matrix& m, std::span<const double> x)
{
    assert(int(x.size()) == m.n);
    std::vector<double> y(std::size_t(m.n), 0.0);
    for (int i = 0; i < m.n; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < m.n; ++j) s += m(i, j) * x[std::size_t(j)];
        y[std::size_t(i)] = s;
    }
    return y;
}

std::expected<std::vector<double>, int> solve(Matrix m, std::vector<double> b)
{
    const int n = m.n;
    assert(int(b.size()) == n);
    for (int k = 0; k < n; ++k)
    {
        int p = k;
        double best = std::fabs(m(k, k));
        for (int i = k + 1; i < n; ++i)
            if (std::fabs(m(i, k)) > best) { best = std::fabs(m(i, k)); p = i; }
        if (m(p, k) == 0.0) return std::unexpected(k);
        if (p != k)
        {
            for (int j = 0; j < n; ++j) std::swap(m(k, j), m(p, j));
            std::swap(b[std::size_t(k)], b[std::size_t(p)]);
        }
        const double piv = m(k, k);
        for (int i = k + 1; i < n; ++i)
        {
            const double l = m(i, k) / piv;
            if (l == 0.0) continue;
            for (int j = k + 1; j < n; ++j) m(i, j) -= l * m(k, j);
            b[std::size_t(i)] -= l * b[std::size_t(k)];
        }
    }
    for (int i = n - 1; i >= 0; --i)
    {
        double s = b[std::size_t(i)];
        for (int j = i + 1; j < n; ++j) s -= m(i, j) * b[std::size_t(j)];
        b[std::size_t(i)] = s / m(i, i);
    }
    return b;
}

double norm(std::span<const double> x)
{
    double s = 0.0;
    for (double v : x) s += v * v;
    return std::sqrt(s);
}

std::vector<double> hybrd(const std::function<void(std::span<const double>,
                                                    std::span<double>)>& fcn,
                          std::vector<double> x, double xtol)
{
    const int n = int(x.size());
    if (n <= 0 || xtol < 0) return x;
    const std::size_t N = std::size_t(n);
    const int maxfev = 200 * (n + 1);
    const double factor = 100.0;
    const double eps = std::sqrt(kEpsMch);   // sqrt(max(epsfcn, epsmch))
    std::vector<double> fvec(N), diag(N), qtf(N), wa1(N), wa2(N), wa3(N), wa4(N);
    std::vector<double> r(N * (N + 1) / 2);
    ColMajor fjac{n, std::vector<double>(N * N)};

    fcn(x, fvec);
    int nfev = 1;
    double fnorm = enorm(n, fvec.data());
    int iter = 1, ncsuc = 0, ncfail = 0, nslow1 = 0, nslow2 = 0;
    double xnorm = 0.0, delta = 0.0;

    for (;;)   // outer loop: fresh Jacobian
    {
        bool jeval = true;
        // fdjac1, full forward differences.
        for (int j = 0; j < n; ++j)
        {
            const double temp = x[std::size_t(j)];
            double h = eps * std::fabs(temp);
            if (h == 0.0) h = eps;
            x[std::size_t(j)] = temp + h;
            fcn(x, wa1);
            x[std::size_t(j)] = temp;
            for (int i = 0; i < n; ++i)
                fjac(i, j) = (wa1[std::size_t(i)] - fvec[std::size_t(i)]) / h;
        }
        nfev += n;
        qrfac(fjac, wa1, wa2);
        if (iter == 1)
        {
            for (int j = 0; j < n; ++j)
            {
                diag[std::size_t(j)] = wa2[std::size_t(j)];
                if (wa2[std::size_t(j)] == 0.0) diag[std::size_t(j)] = 1.0;
            }
            for (int j = 0; j < n; ++j)
                wa3[std::size_t(j)] = diag[std::size_t(j)] * x[std::size_t(j)];
            xnorm = enorm(n, wa3.data());
            delta = factor * xnorm;
            if (delta == 0.0) delta = factor;
        }
        qtf = fvec;
        for (int j = 0; j < n; ++j)
        {
            if (fjac(j, j) == 0.0) continue;
            double sum = 0.0;
            for (int i = j; i < n; ++i) sum += fjac(i, j) * qtf[std::size_t(i)];
            const double temp = -sum / fjac(j, j);
            for (int i = j; i < n; ++i) qtf[std::size_t(i)] += fjac(i, j) * temp;
        }
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < j; ++i) r[packedIndex(n, i, j)] = fjac(i, j);
            r[packedIndex(n, j, j)] = wa1[std::size_t(j)];
        }
        qform(fjac, wa1);
        for (int j = 0; j < n; ++j)
            diag[std::size_t(j)] = std::max(diag[std::size_t(j)], wa2[std::size_t(j)]);

        for (;;)   // inner loop: Broyden updates
        {
            dogleg(n, r, diag, qtf, delta, wa1, wa2, wa3);
            for (int j = 0; j < n; ++j)
            {
                wa1[std::size_t(j)] = -wa1[std::size_t(j)];
                wa2[std::size_t(j)] = x[std::size_t(j)] + wa1[std::size_t(j)];
                wa3[std::size_t(j)] = diag[std::size_t(j)] * wa1[std::size_t(j)];
            }
            const double pnorm = enorm(n, wa3.data());
            if (iter == 1) delta = std::min(delta, pnorm);
            fcn(wa2, wa4);
            ++nfev;
            const double fnorm1 = enorm(n, wa4.data());
            double actred = -1.0;
            if (fnorm1 < fnorm)
            {
                const double q = fnorm1 / fnorm;
                actred = 1.0 - q * q;
            }
            for (int i = 0; i < n; ++i)
            {
                double sum = 0.0;
                for (int j = i; j < n; ++j) sum += r[packedIndex(n, i, j)] * wa1[std::size_t(j)];
                wa3[std::size_t(i)] = qtf[std::size_t(i)] + sum;
            }
            const double temp = enorm(n, wa3.data());
            double prered = 0.0;
            if (temp < fnorm)
            {
                const double q = temp / fnorm;
                prered = 1.0 - q * q;
            }
            double ratio = 0.0;
            if (prered > 0.0) ratio = actred / prered;
            if (ratio < 0.1)
            {
                ncsuc = 0;
                ++ncfail;
                delta = 0.5 * delta;
            }
            else
            {
                ncfail = 0;
                ++ncsuc;
                if (ratio >= 0.5 || ncsuc > 1) delta = std::max(delta, pnorm / 0.5);
                if (std::fabs(ratio - 1.0) <= 0.1) delta = pnorm / 0.5;
            }
            if (ratio >= 1e-4)
            {
                for (int j = 0; j < n; ++j)
                {
                    x[std::size_t(j)] = wa2[std::size_t(j)];
                    wa2[std::size_t(j)] = diag[std::size_t(j)] * x[std::size_t(j)];
                    fvec[std::size_t(j)] = wa4[std::size_t(j)];
                }
                xnorm = enorm(n, wa2.data());
                fnorm = fnorm1;
                ++iter;
            }
            ++nslow1;
            if (actred >= 0.001) nslow1 = 0;
            if (jeval) ++nslow2;
            if (actred >= 0.1) nslow2 = 0;
            if (delta <= xtol * xnorm || fnorm == 0.0) return x;
            if (nfev >= maxfev) return x;
            if (0.1 * std::max(0.1 * delta, pnorm) <= kEpsMch * xnorm) return x;
            if (nslow2 == 5 || nslow1 == 10) return x;
            if (ncfail == 2) break;   // recompute the Jacobian
            for (int j = 0; j < n; ++j)
            {
                double sum = 0.0;
                for (int i = 0; i < n; ++i) sum += fjac(i, j) * wa4[std::size_t(i)];
                wa2[std::size_t(j)] = (sum - wa3[std::size_t(j)]) / pnorm;
                wa1[std::size_t(j)] =
                    diag[std::size_t(j)] * ((diag[std::size_t(j)] * wa1[std::size_t(j)]) / pnorm);
                if (ratio >= 1e-4) qtf[std::size_t(j)] = sum;
            }
            r1updt(n, r, wa1, wa2, wa3);
            r1mpyq(n, n, fjac.a.data(), n, wa2, wa3);
            r1mpyq(1, n, qtf.data(), 1, wa2, wa3);
            jeval = false;
        }
    }
}

} // namespace dns::prop::num
