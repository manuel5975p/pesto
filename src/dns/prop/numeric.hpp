// The numpy / scipy primitives the propeller codes are written in, with THEIR
// semantics, so the C++ port reproduces the retired Python bemt.py/liftline.py to
// rounding: np.interp's clamping and exact-node rules, np.mod's sign, np.arange
// and np.linspace's element formulas, numpy's pairwise sum, scipy's brentq and
// MINPACK's hybrd (scipy.optimize.root(method="hybr")). Internal to dns/prop.
#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <span>
#include <vector>

namespace dns::prop::num {

// np.interp(x, xp, fp): clamped to fp[0]/fp.back() outside, exact at nodes.
// Pre: xp.size() == fp.size() >= 1, xp non-decreasing.
double interp(double x, std::span<const double> xp, std::span<const double> fp);

// np.mod(a, b) for b > 0: result in [0, b), +0 for an exact multiple.
double mod(double a, double b);

// np.linspace(a, b, n) (endpoint included, last element exactly b). Pre: n >= 1.
std::vector<double> linspace(double a, double b, int n);

// np.arange(a, b, step) with numpy's fill rule a + i*((a + step) - a).
// Pre: step > 0. Empty when b <= a.
std::vector<double> arange(double a, double b, double step);

// np.sum: a[0] + numpy's pairwise sum of the rest.
double sum(std::span<const double> a);

// np.mean = np.sum / n. Pre: non-empty.
double mean(std::span<const double> a);

// scipy.optimize.brentq (scipy/optimize/Zeros/brentq.c). Pre: f(a), f(b)
// finite with opposite signs (or one zero). nullopt-like NaN when the
// iteration cap is reached -- scipy would raise there.
double brentq(const std::function<double(double)>& f, double a, double b,
              double xtol, double rtol, int maxiter);

// Square dense matrix, row-major, n x n.
struct Matrix
{
    int n = 0;
    std::vector<double> a;
    explicit Matrix(int n_) : n(n_), a(std::size_t(n_) * std::size_t(n_), 0.0) {}
    double& operator()(int i, int j) { return a[std::size_t(i) * std::size_t(n) + std::size_t(j)]; }
    double operator()(int i, int j) const { return a[std::size_t(i) * std::size_t(n) + std::size_t(j)]; }
};

// y = M x. Pre: x.size() == M.n.
std::vector<double> matvec(const Matrix& m, std::span<const double> x);

// np.linalg.solve(M, b) by LU with partial pivoting. Unexpected on an exact
// zero pivot (numpy's LinAlgError). Pre: b.size() == M.n.
std::expected<std::vector<double>, int> solve(Matrix m, std::vector<double> b);

// ||x||_2 (np.linalg.norm of a vector).
double norm(std::span<const double> x);

// MINPACK hybrd as scipy.optimize.root(f, x0, method="hybr", xtol) calls it:
// forward-difference Jacobian (ml = mu = n-1, epsfcn = machine eps),
// factor 100, mode 1, maxfev 200 (n + 1). Returns the final iterate.
// f(x, out) fills out (size n).
std::vector<double> hybrd(const std::function<void(std::span<const double>,
                                                    std::span<double>)>& f,
                          std::vector<double> x0, double xtol);

} // namespace dns::prop::num
