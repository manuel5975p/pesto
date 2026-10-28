// dns/precond.hpp (`--precond K`, masterplan_levers M2 U1), on the CPU in FP64:
//   (1) the correction scales the pressure rate: b^T R' = eps b^T R, and
//       leaves the velocity and entropy rates exactly as they were;
//   (2) eps == 1 is the identity BIT FOR BIT (the off rows of the march);
//   (3) the eigenvalues of P A (A the x-flux Jacobian, transported by the
//       RELATIVE velocity u - w as in the frozen-rotor frame) are u_rel
//       (triple) and u_rel (1+eps)/2 +- sqrt(((1-eps)/2)^2 u_rel^2 + eps c^2):
//       det(P A - lambda I) vanishes at each, and the traces of P A and
//       (P A)^2 equal the sums of the predicted eigenvalues and their squares;
//   (4) precondWave is the largest |lambda|, and precondEps clamps as stated.

#include "dns/precond.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

namespace {

using dnstest::check;
using Mat = double[5][5];

struct State
{
    double rho, u[3], p, gamma;
    double c2() const { return gamma * p / rho; }
    double H() const
    {
        const double q2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        return gamma / (gamma - 1.0) * p / rho + 0.5 * q2;
    }
};

// x-flux Jacobian dF/dQ of the Euler equations, transported by u - wx.
void jacobian(const State& s, double wx, Mat A)
{
    const double g1 = s.gamma - 1.0, u = s.u[0], v = s.u[1], w = s.u[2];
    const double q2 = u * u + v * v + w * w, H = s.H();
    const double rows[5][5] = {
        {0, 1, 0, 0, 0},
        {0.5 * g1 * q2 - u * u, (3.0 - s.gamma) * u, -g1 * v, -g1 * w, g1},
        {-u * v, v, u, 0, 0},
        {-u * w, w, 0, u, 0},
        {u * (0.5 * g1 * q2 - H), H - g1 * u * u, -g1 * u * v, -g1 * u * w,
         s.gamma * u}};
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) A[i][j] = rows[i][j] - (i == j ? wx : 0.0);
}

// P A, column by column: P applied to each column of A (precondApply is P).
void precondTimes(const State& s, double eps, const Mat A, Mat PA)
{
    for (int j = 0; j < 5; ++j) {
        double col[5];
        for (int i = 0; i < 5; ++i) col[i] = A[i][j];
        dns::precondApply(col, s.u, s.H(), s.gamma, s.c2(), eps);
        for (int i = 0; i < 5; ++i) PA[i][j] = col[i];
    }
}

// |det(B)| / prod_i |row_i| (Hadamard-normalised), partial pivoting.
double relDet(const Mat Bin)
{
    double B[5][5];
    double scale = 1.0;
    for (int i = 0; i < 5; ++i) {
        double n = 0;
        for (int j = 0; j < 5; ++j) {
            B[i][j] = Bin[i][j];
            n += B[i][j] * B[i][j];
        }
        scale *= std::sqrt(n);
    }
    double det = 1.0;
    for (int k = 0; k < 5; ++k) {
        int p = k;
        for (int i = k + 1; i < 5; ++i)
            if (std::abs(B[i][k]) > std::abs(B[p][k])) p = i;
        if (B[p][k] == 0.0) return 0.0;
        if (p != k)
            for (int j = 0; j < 5; ++j) std::swap(B[k][j], B[p][j]);
        det *= B[k][k];
        for (int i = k + 1; i < 5; ++i) {
            const double f = B[i][k] / B[k][k];
            for (int j = k; j < 5; ++j) B[i][j] -= f * B[k][j];
        }
    }
    return std::abs(det) / scale;
}

double bR(const State& s, const double R[5])
{
    const double q2 = s.u[0] * s.u[0] + s.u[1] * s.u[1] + s.u[2] * s.u[2];
    return (s.gamma - 1.0) *
           (0.5 * q2 * R[0] - s.u[0] * R[1] - s.u[1] * R[2] - s.u[2] * R[3] + R[4]);
}

void algebra()
{
    std::mt19937 rng(20261028);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    double worstP = 0, worstV = 0, worstS = 0, worstDet = 0, worstTr = 0,
           worstWave = 0, worstOff = 0, bestWrong = 1e300;
    bool identity = true;
    for (int trial = 0; trial < 400; ++trial) {
        // Sound-speed units (c = 1), so every matrix entry is O(1) at any
        // Mach and the determinant test reads rounding, not scaling.
        State s{1.0 + 0.5 * U(rng), {U(rng), U(rng), U(rng)}, 0.0, 1.4};
        const double mach = 0.02 + 0.5 * (0.5 + 0.5 * U(rng));
        const double q = std::sqrt(s.u[0] * s.u[0] + s.u[1] * s.u[1] +
                                   s.u[2] * s.u[2]);
        for (double& x : s.u) x *= mach / q; // |u| / c = mach
        s.p = s.rho / s.gamma;
        const double c2 = s.c2();
        const double epsList[5] = {1.0, 0.5, 1e-2, 1e-4, mach * mach};
        for (double eps : epsList) {
            double R[5], R0[5];
            for (int v = 0; v < 5; ++v) R[v] = R0[v] = U(rng) * (v == 4 ? c2 : 1.0);
            dns::precondApply(R, s.u, s.H(), s.gamma, c2, eps);
            if (eps == 1.0) {
                for (int v = 0; v < 5; ++v) identity = identity && R[v] == R0[v];
                continue;
            }
            const double p0 = bR(s, R0), p1 = bR(s, R);
            worstP = std::max(worstP, std::abs(p1 - eps * p0) /
                                          (std::abs(p0) + 1e-300));
            // velocity rate (R_m - u R_rho) / rho and entropy rate
            // (R_p - c^2 R_rho) scaled: velocity unchanged, entropy unchanged.
            for (int d = 0; d < 3; ++d) {
                const double dv0 = (R0[1 + d] - s.u[d] * R0[0]) / s.rho;
                const double dv1 = (R[1 + d] - s.u[d] * R[0]) / s.rho;
                worstV = std::max(worstV, std::abs(dv1 - dv0) /
                                              (std::abs(R0[1 + d]) + std::abs(R0[0])));
            }
            const double s0 = p0 - c2 * R0[0], s1 = p1 - c2 * R[0];
            worstS = std::max(worstS, std::abs(s1 - s0) /
                                          (std::abs(p0) + c2 * std::abs(R0[0])));
        }
        // Eigenvalues, with and without a frame velocity.
        for (double wx : {0.0, 0.7 * U(rng)}) {
            Mat A, PA;
            jacobian(s, wx, A);
            for (double eps : epsList) {
                precondTimes(s, eps, A, PA);
                const double ur = s.u[0] - wx;
                const double h = 0.5 * (1.0 - eps);
                const double rt = std::sqrt(h * h * ur * ur + eps * c2);
                const double lam[5] = {ur, ur, ur, 0.5 * ur * (1.0 + eps) + rt,
                                       0.5 * ur * (1.0 + eps) - rt};
                for (double l : {lam[0], lam[3], lam[4]}) {
                    Mat B;
                    for (int i = 0; i < 5; ++i)
                        for (int j = 0; j < 5; ++j)
                            B[i][j] = PA[i][j] - (i == j ? l : 0.0);
                    worstDet = std::max(worstDet, relDet(B));
                    // The control: 0.05 c off an eigenvalue is visibly not one.
                    for (int i = 0; i < 5; ++i) B[i][i] -= 0.05;
                    bestWrong = std::min(bestWrong, relDet(B));
                }
                double tr = 0, tr2 = 0, el = 0, el2 = 0, sc = 0;
                for (int i = 0; i < 5; ++i) {
                    tr += PA[i][i];
                    for (int k = 0; k < 5; ++k) tr2 += PA[i][k] * PA[k][i];
                    el += lam[i];
                    el2 += lam[i] * lam[i];
                }
                sc = c2 + ur * ur;
                worstTr = std::max({worstTr, std::abs(tr - el) / std::sqrt(sc),
                                    std::abs(tr2 - el2) / sc});
                // |u'| + c' is the largest |lambda| (urel >= 0 by symmetry).
                const double wv = dns::precondWave(std::abs(ur), c2, eps);
                const double lmax = std::max(std::abs(lam[3]), std::abs(lam[4]));
                worstWave = std::max(worstWave, std::abs(wv - lmax) / lmax);
            }
            // eps == 1: the physical u - c, u, u + c (the off rows).
            const double ur = s.u[0] - wx, c = std::sqrt(c2);
            worstOff = std::max(
                worstOff, std::abs(dns::precondWave(std::abs(ur), c2, 1.0) -
                                   (std::abs(ur) + c)) / c);
        }
    }
    std::printf("precond: worst |b^T R' - eps b^T R| %.2e, velocity %.2e, "
                "entropy %.2e, det %.2e (0.05 c off: >= %.2e), traces %.2e, "
                "wave %.2e, eps=1 wave %.2e\n",
                worstP, worstV, worstS, worstDet, bestWrong, worstTr, worstWave,
                worstOff);
    check(identity, "eps == 1: R' == R bit for bit");
    check(worstP < 1e-12, "b^T R' = eps b^T R (pressure rate scaled)");
    check(worstV < 1e-12, "velocity rates unchanged");
    check(worstS < 1e-12, "entropy rate unchanged");
    check(worstDet < 1e-12, "det(P A - lambda I) = 0 at u_rel, u' +- c'");
    check(bestWrong > 1e3 * worstDet, "control: det 0.05 c off an eigenvalue "
                                      "is >= 1e3 x the on-eigenvalue det");
    check(worstTr < 1e-12, "tr(P A), tr((P A)^2) = predicted eigenvalue sums");
    check(worstWave < 1e-12, "precondWave = max |lambda|");
    check(worstOff < 1e-12, "precondWave(eps = 1) = |u| + c");
}

void clamps()
{
    using dns::precondEps;
    const double c2 = 100.0;
    check(precondEps(c2, 1.0, 4.0, 0.0) == 0.04, "K U_inf floor binds");
    check(precondEps(c2, 9.0, 4.0, 0.0) == 0.09, "local speed binds");
    check(precondEps(c2, 1.0, 4.0, 16.0) == 0.16, "viscous floor binds");
    check(precondEps(c2, 1.0, 400.0, 0.0) == 1.0, "K U_inf >= c: eps = 1 (off)");
    check(precondEps(c2, 200.0, 0.0, 0.0) == 1.0, "supersonic: eps = 1");
    check(std::isnan(precondEps(std::nan(""), 1.0, 4.0, 0.0)),
          "NaN c^2: NaN eps, never a clamp to a finite value");
    const float e = precondEps(100.0f, 1.0f, 4.0f, 0.0f);
    check(e == 0.04f, "float instantiation");
}

} // namespace

int main()
{
    algebra();
    clamps();
    return dnstest::finish();
}
