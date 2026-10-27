#include "dns/rans_channel1d.hpp"

#include "dns/config.hpp"
#include "dns/curvmesh.hpp"
#include "dns/remap.hpp"
#include "dns/transport.hpp"
#include "dns/walldist.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <format>

namespace dns {
namespace {

constexpr int kMaxV = 4; // unknowns per row: u, T, up to two scalars

// The column of the channel mesh the operator reads, per GRID row j: the +y
// face area S and inverse Jacobian d eta/dy, 1/vol, the row height, the cell
// centroid and its wall distance. Every entry comes from CurvMetrics in
// double, the same tables the GPU narrows to float.
struct Ctx
{
    int nf = 0, nyT = 0, nS = 0, nv = 0;
    RansModel model = RansModel::None;
    std::vector<double> S, Jy, iv, dy, yc, d;
    double mu = 0, kap = 0, cp = 0, Tw = 0, pRef = 0, Ue = 0, reTau = 0,
           mass = 0;
    double omegaCoef = 0; // SST: 60 mu / (beta1 d1^2) (gpu/comp_channel)
};

// Grid-row fields, ghost rows included. Td = T - Tw is carried on its own:
// T ~ Tw ~ 1e3-1e4 in these units, so differencing T itself would floor the
// energy residual at eps Tw / dy^2 (measured 1.4e-10 laminar), above the
// oracle tolerance. SST: k, om (omega) and the blend F1.
struct Rows
{
    std::vector<double> rho, u, T, Td, nt, mut, k, om, f1;
};

// Cell-centre gradient of a row field at fluid row j (curvScalarGradCentre's
// 1D restriction: index central difference times the mean of the two y-face
// inverse Jacobians).
double gradC(const Ctx& c, const std::vector<double>& f, int j)
{
    return 0.5 * (f[j + 1] - f[j - 1]) * (0.5 * (c.Jy[j] + c.Jy[j - 1]));
}

double& xu(std::vector<double>& x, const Ctx& c, int j, int v)
{
    return x[std::size_t(j) * c.nv + v];
}
double xg(const std::vector<double>& x, const Ctx& c, int j, int v)
{
    return x[std::size_t(j) * c.nv + v];
}

// Unknowns per fluid row f = j-1: (u, T/Tw - 1, then SA nu~ reTau | SST ln k,
// ln omega -- positive by construction); last entry p/pRef. The isothermal
// ghost is wall_ghost.hpp's algebra in double: T0 = max(2 Tw - T1, Tw/20),
// p0 = p1, rho0 from the EOS, u0 = -u1; SA nu~ and SST k mirror their
// specific values odd, SST omega is the image of the TMR wall value; mu_t
// mirrors odd and F1 even (never computed from ghost scalars).
void fill(const Ctx& c, const std::vector<double>& x, Rows& r)
{
    const int n = c.nyT;
    for (auto* v : {&r.rho, &r.u, &r.T, &r.Td, &r.nt, &r.mut, &r.k, &r.om,
                    &r.f1})
        v->assign(n, 0.0);
    const bool sst = ransSst(c.model);
    const double p = x.back() * c.pRef;
    for (int j = 1; j <= c.nf; ++j) {
        r.u[j] = xg(x, c, j - 1, 0);
        r.Td[j] = xg(x, c, j - 1, 1) * c.Tw;
        r.T[j] = c.Tw + r.Td[j];
        r.rho[j] = p / (kRgasBox * r.T[j]);
        if (sst) {
            r.k[j] = std::exp(xg(x, c, j - 1, 2));
            r.om[j] = std::exp(xg(x, c, j - 1, 3));
        } else if (c.nS) {
            r.nt[j] = xg(x, c, j - 1, 2) / c.reTau;
            r.mut[j] = saMut(r.rho[j], r.nt[j], c.mu / r.rho[j]);
        }
    }
    const std::pair<int, int> ghosts[] = {{0, 1}, {n - 1, n - 2}};
    for (auto [g, s] : ghosts) {
        r.Td[g] = std::max(-r.Td[s], -0.95 * c.Tw);
        r.T[g] = c.Tw + r.Td[g];
        r.rho[g] = p / (kRgasBox * r.T[g]);
        r.u[g] = -r.u[s];
        r.nt[g] = -r.nt[s];
        r.mut[g] = -r.mut[s];
        r.k[g] = -r.k[s];
        r.om[g] = 2.0 * c.omegaCoef / (0.5 * (r.rho[g] + r.rho[s])) - r.om[s];
    }
    if (!sst) return;
    // mu_t and F1 need the cell-centre gradients, so they come after the
    // ghosts: sstMutCurv's 1D restriction, then the odd/even mirror.
    for (int j = 1; j <= c.nf; ++j) {
        const double R = std::abs(gradC(c, r.u, j)); // S == Omega in 1D shear
        const double gkw = gradC(c, r.k, j) * gradC(c, r.om, j);
        const double cd = sstCd(r.rho[j], gkw, r.om[j], sstCdFloor(c.model));
        const auto b = sstBlend(r.k[j], r.om[j], c.d[j], c.mu / r.rho[j],
                                r.rho[j], cd);
        r.mut[j] = sstMut(r.rho[j], r.k[j], r.om[j], R, b.F2);
        r.f1[j] = b.F1;
    }
    for (auto [g, s] : ghosts) {
        r.mut[g] = -r.mut[s];
        r.f1[g] = r.f1[s];
    }
}

struct Norms
{
    double resU = 0, resE = 0, resT = 0, resid = 0;
};

// e: per fluid row (R_mx/rho, R_E/(rho Ue), R_t/rho) and the mass constraint
// (sum rho dy - M)/M. The per-volume residuals are the GPU stage rhs
// restricted to v = w = 0 and x/z homogeneity, where every x- and z-face flux
// cancels against its twin and the y-momentum balance is p uniform.
void eval(const Ctx& c, const std::vector<double>& x, std::vector<double>& e,
          Norms* nm = nullptr)
{
    Rows r;
    fill(c, x, r);
    const int n = c.nyT;
    const bool sst = ransSst(c.model);
    std::vector<double> Fmx(n, 0.0), FE(n, 0.0), Ft(n, 0.0), Fw(n, 0.0);
    for (int j = 0; j + 1 < n; ++j) { // the +y face of row j
        const int a = j, b = j + 1;
        const double rf = 0.5 * (r.rho[a] + r.rho[b]);
        const double uf = 0.5 * (r.u[a] + r.u[b]);
        const double nf = 0.5 * (r.mut[a] + r.mut[b]);
        const double muf = c.mu + nf;
        const double kapf = c.kap + nf * (c.cp / kPrandtlTurb);
        const double ts = muf * ((r.u[b] - r.u[a]) * c.Jy[a]) * c.S[a];
        Fmx[j] = -ts;
        FE[j] = -(kapf * ((r.Td[b] - r.Td[a]) * c.Jy[a]) * c.S[a] + uf * ts);
        if (sst) {
            const double F1f = 0.5 * (r.f1[a] + r.f1[b]);
            const double sk = sstBlendC(F1f, sst::sigmaK1, sst::sigmaK2);
            const double sw = sstBlendC(F1f, sst::sigmaW1, sst::sigmaW2);
            Ft[j] = -(c.mu + sk * nf) * (r.k[b] - r.k[a]) * c.Jy[a] * c.S[a];
            Fw[j] = -(c.mu + sw * nf) * (r.om[b] - r.om[a]) * c.Jy[a] * c.S[a];
        } else if (c.nS) {
            const double nuf = c.mu / rf;
            const double ntf = 0.5 * (r.nt[a] + r.nt[b]);
            const double gS = (r.nt[b] - r.nt[a]) * c.Jy[a] * c.S[a];
            Ft[j] = -(1.0 / sa::sigma) * rf * saDiffCoef(nuf, ntf) * gS;
        }
    }
    e.assign(std::size_t(c.nv) * c.nf + 1, 0.0);
    double su = 0, sE = 0, st = 0, sw = 0, mass = 0;
    for (int j = 1; j <= c.nf; ++j) {
        const int f = j - 1;
        const double Rmx = -(Fmx[j] - Fmx[j - 1]) * c.iv[j] + 1.0;
        const double RE = -(FE[j] - FE[j - 1]) * c.iv[j] + r.u[j];
        e[std::size_t(f) * c.nv + 0] = Rmx / r.rho[j];
        e[std::size_t(f) * c.nv + 1] = RE / (r.rho[j] * c.Ue);
        su += (Rmx / r.rho[j]) * (Rmx / r.rho[j]);
        sE += (RE / r.rho[j] / c.Ue) * (RE / r.rho[j] / c.Ue);
        mass += r.rho[j] * c.dy[j];
        if (sst) {
            const double R = std::abs(gradC(c, r.u, j));
            const double gkw = gradC(c, r.k, j) * gradC(c, r.om, j);
            const auto P = sstSource(c.model, r.rho[j], r.k[j], r.om[j], R,
                                     r.mut[j], r.f1[j], gkw);
            const double Rk = -(Ft[j] - Ft[j - 1]) * c.iv[j] + P.Rk;
            const double Rw = -(Fw[j] - Fw[j - 1]) * c.iv[j] + P.Rw;
            // k as R/rho, omega as the rate of ln omega (GPU ResidualNorms).
            e[std::size_t(f) * c.nv + 2] = Rk / r.rho[j];
            e[std::size_t(f) * c.nv + 3] = Rw / (r.rho[j] * r.om[j]);
            st += (Rk / r.rho[j]) * (Rk / r.rho[j]);
            sw += (Rw / (r.rho[j] * r.om[j])) * (Rw / (r.rho[j] * r.om[j]));
            continue;
        }
        if (!c.nS) continue;
        // Cell-centre gradients: index central difference times the mean of
        // the two y-face inverse Jacobians (curvScalarGradCentre).
        const double Jc = 0.5 * (c.Jy[j] + c.Jy[j - 1]);
        const double gn = 0.5 * (r.nt[j + 1] - r.nt[j - 1]) * Jc;
        const double grho = 0.5 * (r.rho[j + 1] - r.rho[j - 1]) * Jc;
        const double Om = std::abs(0.5 * (r.u[j + 1] - r.u[j - 1]) * Jc);
        const double nu = c.mu / r.rho[j];
        const auto P = saSource(r.nt[j], nu, Om, c.d[j]);
        const double Rt = -(Ft[j] - Ft[j - 1]) * c.iv[j] + r.rho[j] * P.R +
                          (sa::cb2 / sa::sigma) * r.rho[j] * gn * gn -
                          (1.0 / sa::sigma) * saDiffCoef(nu, r.nt[j]) * grho * gn;
        e[std::size_t(f) * c.nv + 2] = Rt / r.rho[j];
        st += (Rt / r.rho[j]) * (Rt / r.rho[j]);
    }
    e.back() = (mass - c.mass) / c.mass;
    if (!nm) return;
    nm->resU = std::sqrt(su / c.nf);
    nm->resE = std::sqrt(sE / c.nf);
    nm->resT = std::max(std::sqrt(st / c.nf), std::sqrt(sw / c.nf));
    nm->resid = std::max({nm->resU, nm->resE, nm->resT});
}

using Blk = std::array<double, kMaxV * kMaxV>;
using Vec = std::array<double, kMaxV>;

// Small dense solve of A X = B (nv x nv, nrhs columns), partial pivoting.
// Returns false on a singular pivot.
bool solveSmall(int nv, Blk A, double* B, int nrhs)
{
    for (int k = 0; k < nv; ++k) {
        int piv = k;
        for (int i = k + 1; i < nv; ++i)
            if (std::abs(A[i * nv + k]) > std::abs(A[piv * nv + k])) piv = i;
        if (!(std::abs(A[piv * nv + k]) > 0.0)) return false;
        if (piv != k) {
            for (int q = 0; q < nv; ++q) std::swap(A[k * nv + q], A[piv * nv + q]);
            for (int q = 0; q < nrhs; ++q)
                std::swap(B[k * nrhs + q], B[piv * nrhs + q]);
        }
        for (int i = k + 1; i < nv; ++i) {
            const double l = A[i * nv + k] / A[k * nv + k];
            for (int q = k; q < nv; ++q) A[i * nv + q] -= l * A[k * nv + q];
            for (int q = 0; q < nrhs; ++q) B[i * nrhs + q] -= l * B[k * nrhs + q];
        }
    }
    for (int k = nv - 1; k >= 0; --k)
        for (int q = 0; q < nrhs; ++q) {
            double s = B[k * nrhs + q];
            for (int i = k + 1; i < nv; ++i) s -= A[k * nv + i] * B[i * nrhs + q];
            B[k * nrhs + q] = s / A[k * nv + k];
        }
    return true;
}

// The Newton system: block-tridiagonal rows (L, D, U), the global-p column C,
// and the mass row (M, dm). Block Thomas with two right-hand sides (r and C),
// then the bordered unknown by its Schur complement.
struct Jac
{
    std::vector<Blk> L, D, U;
    std::vector<Vec> C, M;
    double dm = 0;
};

bool solveBordered(const Ctx& c, Jac J, const std::vector<double>& rhs,
                   std::vector<double>& dx)
{
    const int nf = c.nf, nv = c.nv;
    // Two columns per row: [r | C].
    std::vector<std::array<double, 2 * kMaxV>> R(nf);
    for (int j = 0; j < nf; ++j)
        for (int v = 0; v < nv; ++v) {
            R[j][v * 2 + 0] = rhs[std::size_t(j) * nv + v];
            R[j][v * 2 + 1] = J.C[j][v];
        }
    // Forward elimination: D_j -= L_j D_{j-1}^-1 U_{j-1}.
    for (int j = 1; j < nf; ++j) {
        // X = D_{j-1}^-1 [U_{j-1} | R_{j-1}]  (nv x (nv + 2))
        std::array<double, kMaxV * (kMaxV + 2)> X{};
        const int w = nv + 2;
        for (int i = 0; i < nv; ++i) {
            for (int q = 0; q < nv; ++q) X[i * w + q] = J.U[j - 1][i * nv + q];
            X[i * w + nv] = R[j - 1][i * 2 + 0];
            X[i * w + nv + 1] = R[j - 1][i * 2 + 1];
        }
        if (!solveSmall(nv, J.D[j - 1], X.data(), w)) return false;
        for (int i = 0; i < nv; ++i)
            for (int k = 0; k < nv; ++k) {
                const double l = J.L[j][i * nv + k];
                for (int q = 0; q < nv; ++q) J.D[j][i * nv + q] -= l * X[k * w + q];
                R[j][i * 2 + 0] -= l * X[k * w + nv];
                R[j][i * 2 + 1] -= l * X[k * w + nv + 1];
            }
        // Keep D_{j-1}^-1 R_{j-1} for the back substitution.
        for (int i = 0; i < nv; ++i) {
            R[j - 1][i * 2 + 0] = X[i * w + nv];
            R[j - 1][i * 2 + 1] = X[i * w + nv + 1];
        }
        // ... and D_{j-1}^-1 U_{j-1} in U.
        for (int i = 0; i < nv; ++i)
            for (int q = 0; q < nv; ++q) J.U[j - 1][i * nv + q] = X[i * w + q];
    }
    if (!solveSmall(nv, J.D[nf - 1], R[nf - 1].data(), 2)) return false;
    // Back substitution: y_j = D^-1 R_j - (D^-1 U_j) y_{j+1}, both columns.
    for (int j = nf - 2; j >= 0; --j)
        for (int i = 0; i < nv; ++i)
            for (int k = 0; k < nv; ++k)
                for (int q = 0; q < 2; ++q)
                    R[j][i * 2 + q] -= J.U[j][i * nv + k] * R[j + 1][k * 2 + q];
    // Border: dp = (s - M.y1) / (dm - M.y2).
    double my1 = 0, my2 = 0;
    for (int j = 0; j < nf; ++j)
        for (int v = 0; v < nv; ++v) {
            my1 += J.M[j][v] * R[j][v * 2 + 0];
            my2 += J.M[j][v] * R[j][v * 2 + 1];
        }
    const double den = J.dm - my2;
    if (!(std::abs(den) > 0.0)) return false;
    const double dp = (rhs.back() - my1) / den;
    dx.assign(rhs.size(), 0.0);
    for (int j = 0; j < nf; ++j)
        for (int v = 0; v < nv; ++v)
            dx[std::size_t(j) * nv + v] = R[j][v * 2 + 0] - R[j][v * 2 + 1] * dp;
    dx.back() = dp;
    return true;
}

// Forward-difference Jacobian with a 3-colouring of the rows (the SA / laminar
// stencil reaches j-1..j+1 only), the p column by its own difference, and the
// mass row analytically (rho = p / (R T)).
Jac jacobian(const Ctx& c, const std::vector<double>& x,
             const std::vector<double>& e0)
{
    const int nf = c.nf, nv = c.nv;
    const int ncol = 3;
    Jac J;
    J.L.assign(nf, Blk{});
    J.D.assign(nf, Blk{});
    J.U.assign(nf, Blk{});
    J.C.assign(nf, Vec{});
    J.M.assign(nf, Vec{});
    std::vector<double> xp, ep;
    for (int col = 0; col < ncol; ++col)
        for (int v = 0; v < nv; ++v) {
            xp = x;
            std::vector<double> h(nf, 0.0);
            for (int j = col; j < nf; j += ncol) {
                h[j] = 1e-7 * std::max(std::abs(xg(x, c, j, v)), 1.0);
                xu(xp, c, j, v) += h[j];
            }
            eval(c, xp, ep);
            for (int i = 0; i < nf; ++i)
                for (int jj = std::max(0, i - 1); jj <= std::min(nf - 1, i + 1);
                     ++jj) {
                    if (jj % ncol != col) continue;
                    Blk& B = jj == i ? J.D[i] : jj < i ? J.L[i] : J.U[i];
                    for (int q = 0; q < nv; ++q)
                        B[q * nv + v] = (ep[std::size_t(i) * nv + q] -
                                         e0[std::size_t(i) * nv + q]) / h[jj];
                }
        }
    xp = x;
    const double hp = 1e-7 * std::max(std::abs(x.back()), 1.0);
    xp.back() += hp;
    eval(c, xp, ep);
    for (int i = 0; i < nf; ++i)
        for (int q = 0; q < nv; ++q)
            J.C[i][q] = (ep[std::size_t(i) * nv + q] - e0[std::size_t(i) * nv + q]) / hp;
    // mass = sum_j (pi pRef / (R theta_j Tw)) dy_j, theta = 1 + x_T
    double m = 0;
    for (int j = 0; j < nf; ++j) {
        const double th = 1.0 + xg(x, c, j, 1);
        const double rho = x.back() * c.pRef / (kRgasBox * th * c.Tw);
        J.M[j][1] = -rho * c.dy[j + 1] / th / c.mass;
        m += rho * c.dy[j + 1];
    }
    J.dm = m / x.back() / c.mass;
    return J;
}

// The DENSE Jacobian for SST, whose stencil reaches j+-2 (mu_t and F1 carry
// their own gradients): 5-colour forward differences keep every block of
// distance <= 2, the p column by its own difference, the mass row
// analytically. Row-major N x N, N = nv nf + 1 (a few hundred: the dense LU
// is milliseconds).
std::vector<double> denseJacobian(const Ctx& c, const std::vector<double>& x,
                                  const std::vector<double>& e0)
{
    const int nf = c.nf, nv = c.nv;
    const std::size_t N = x.size();
    std::vector<double> A(N * N, 0.0);
    std::vector<double> xp, ep;
    for (int col = 0; col < 5; ++col)
        for (int v = 0; v < nv; ++v) {
            xp = x;
            std::vector<double> h(nf, 0.0);
            for (int j = col; j < nf; j += 5) {
                h[j] = 1e-7 * std::max(std::abs(xg(x, c, j, v)), 1.0);
                xu(xp, c, j, v) += h[j];
            }
            eval(c, xp, ep);
            for (int i = 0; i < nf; ++i)
                for (int jj = std::max(0, i - 2); jj <= std::min(nf - 1, i + 2);
                     ++jj) {
                    if (jj % 5 != col) continue;
                    for (int q = 0; q < nv; ++q) {
                        const std::size_t r = std::size_t(i) * nv + q;
                        A[r * N + std::size_t(jj) * nv + v] = (ep[r] - e0[r]) / h[jj];
                    }
                }
        }
    xp = x;
    const double hp = 1e-7 * std::max(std::abs(x.back()), 1.0);
    xp.back() += hp;
    eval(c, xp, ep);
    for (std::size_t r = 0; r + 1 < N; ++r) A[r * N + N - 1] = (ep[r] - e0[r]) / hp;
    double m = 0;
    for (int j = 0; j < nf; ++j) {
        const double th = 1.0 + xg(x, c, j, 1);
        const double rho = x.back() * c.pRef / (kRgasBox * th * c.Tw);
        A[(N - 1) * N + std::size_t(j) * nv + 1] = -rho * c.dy[j + 1] / th / c.mass;
        m += rho * c.dy[j + 1];
    }
    A[N * N - 1] = m / x.back() / c.mass;
    return A;
}

// Dense LU solve of A x = b in place (partial pivoting); false if singular.
bool solveDense(std::vector<double> A, std::vector<double>& b)
{
    const std::size_t N = b.size();
    for (std::size_t k = 0; k < N; ++k) {
        std::size_t piv = k;
        for (std::size_t i = k + 1; i < N; ++i)
            if (std::abs(A[i * N + k]) > std::abs(A[piv * N + k])) piv = i;
        if (!(std::abs(A[piv * N + k]) > 0.0)) return false;
        if (piv != k) {
            for (std::size_t q = 0; q < N; ++q) std::swap(A[k * N + q], A[piv * N + q]);
            std::swap(b[k], b[piv]);
        }
        const double inv = 1.0 / A[k * N + k];
        for (std::size_t i = k + 1; i < N; ++i) {
            const double l = A[i * N + k] * inv;
            if (l == 0.0) continue;
            for (std::size_t q = k; q < N; ++q) A[i * N + q] -= l * A[k * N + q];
            b[i] -= l * b[k];
        }
    }
    for (std::size_t k = N; k-- > 0;) {
        double s = b[k];
        for (std::size_t q = k + 1; q < N; ++q) s -= A[k * N + q] * b[q];
        b[k] = s / A[k * N + k];
    }
    return true;
}

} // namespace

Grid channel1dGrid(const Channel1dCase& c, int nx, int nz, double lx, double lz)
{
    const int nyT = c.ny + 2;
    const double h1 = ChannelStretchY::firstCell(nyT, c.beta, 1.0);
    return Grid::make(nx, nyT, nz, lx, 2.0 + 2.0 * h1, nz == 1 ? 1.0 : lz);
}

std::expected<Channel1dSolution, std::string> solveChannel1d(const Channel1dCase& cs)
{
    assert(cs.ny >= 8 && cs.reTau > 0 && cs.mach > 0);
    const bool sst = ransSst(cs.model);
    const Grid g = channel1dGrid(cs, 2, 1, 1.0, 1.0);
    const CurvMesh mesh = CurvMesh::channelY(g, cs.beta, 1.0);
    auto m = CurvMetrics::build(mesh);
    if (!m) return std::unexpected("oracle mesh: " + m.error());
    const auto wd = channelWallDistance(mesh, 1.0);

    Ctx c;
    c.nf = cs.ny;
    c.nyT = g.ny;
    c.model = cs.model;
    c.nS = ransScalars(cs.model);
    c.nv = 2 + c.nS;
    c.reTau = cs.reTau;
    c.mu = 1.0 / cs.reTau;
    const Transport tr = Transport::constant(kGammaAir, c.mu);
    c.cp = tr.cp();
    c.kap = tr.kappa(1.0);
    c.Ue = channelUbEstimate(cs.reTau);
    c.pRef = channelPRef(cs.reTau, cs.mach);
    c.Tw = c.pRef; // rho = R = 1
    c.mass = cs.totalMass;
    for (int j = 0; j < g.ny; ++j) {
        const std::size_t q = g.idx(0, j, 0);
        c.S.push_back(m->S[1][1][q]);
        c.Jy.push_back(m->Ji[1][4][q]);
        c.iv.push_back(1.0 / m->vol[q]);
        c.dy.push_back(m->vol[q] / m->S[1][1][q]);
        c.yc.push_back(0.5 * (mesh.node(0, j, 0)[1] + mesh.node(0, j + 1, 0)[1]));
        c.d.push_back(wd[q]);
    }
    c.omegaCoef = sst::omegaWallFactor * c.mu / (c.d[1] * c.d[1]);

    // Start: Reichardt + the model's starting scalars (turbulent), or the
    // exact laminar parabola u = y (2 - y) / (2 mu) (laminar: nothing else is
    // close).
    std::vector<double> x(std::size_t(c.nv) * c.nf + 1, 0.0);
    double sdy = 0;
    for (int j = 1; j <= c.nf; ++j) {
        const double dw = c.d[j];
        xu(x, c, j - 1, 0) = c.nS ? reichardtUplus(dw * cs.reTau)
                                  : c.yc[j] * (2.0 - c.yc[j]) / (2.0 * c.mu);
        xu(x, c, j - 1, 1) = 0.0; // T = Tw
        if (sst) {
            const KOmega kw = sstInitialKOmega(dw, c.mu, 1.0);
            xu(x, c, j - 1, 2) = std::log(kw.k);
            xu(x, c, j - 1, 3) = std::log(kw.omega);
        } else if (c.nS) {
            xu(x, c, j - 1, 2) = saInitialNut(dw, 1.0, 1.0) * cs.reTau;
        }
        sdy += c.dy[j];
    }
    x.back() = c.mass / sdy; // rho = pi at T = Tw

    // Pseudo-transient continuation: (Dm / dtau - J) dx = e, dtau grown by
    // switched evolution relaxation, backtracked on a failed step. Dm is the
    // pseudo-time mass of each unknown against its residual row (SST: d k/dt
    // = k d ln k/dt; the omega row is already the rate of ln omega).
    Vec Dm{};
    Dm[0] = 1.0;
    Dm[1] = (kRgasBox / (kGammaAir - 1.0)) * c.Tw / c.Ue;
    if (!sst)
        for (int s = 0; s < c.nS; ++s) Dm[2 + s] = 1.0 / cs.reTau;
    if (sst) Dm[3] = 1.0;
    std::vector<double> e, dx, xt, et;
    Norms nm;
    eval(c, x, e, &nm);
    const double res0 = nm.resid;
    double dtau = 1e-2;
    int it = 0;
    for (; it < 400 && !(nm.resid < cs.tol); ++it) {
        if (!std::isfinite(nm.resid))
            return std::unexpected("oracle: residual is not finite");
        Jac J;
        std::vector<double> A;
        if (sst) A = denseJacobian(c, x, e);
        else J = jacobian(c, x, e);
        bool accepted = false;
        for (int bt = 0; bt < 40 && !accepted; ++bt) {
            // e is the TIME DERIVATIVE (dq/dt = R), so implicit Euler in
            // pseudo-time is (Dm/dtau - J) dx = e; the pure Newton step is
            // its dtau -> inf limit. The mass row has no time derivative.
            bool solved = false;
            if (sst) {
                const std::size_t N = x.size();
                std::vector<double> M(A.size());
                for (std::size_t q = 0; q < A.size(); ++q) M[q] = -A[q];
                for (int j = 0; j < c.nf; ++j) {
                    Vec dmj = Dm;
                    dmj[2] = std::exp(xg(x, c, j, 2)); // d k = k d ln k
                    for (int v = 0; v < c.nv; ++v) {
                        const std::size_t r = std::size_t(j) * c.nv + v;
                        M[r * N + r] += dmj[v] / dtau;
                    }
                }
                dx = e;
                solved = solveDense(std::move(M), dx);
            } else {
                Jac Jt = J;
                for (int j = 0; j < c.nf; ++j) {
                    for (auto* B : {&Jt.L[j], &Jt.D[j], &Jt.U[j]})
                        for (double& a : *B) a = -a;
                    for (double& a : Jt.C[j]) a = -a;
                    for (double& a : Jt.M[j]) a = -a;
                    for (int v = 0; v < c.nv; ++v)
                        Jt.D[j][v * c.nv + v] += Dm[v] / dtau;
                }
                Jt.dm = -Jt.dm;
                solved = solveBordered(c, Jt, e, dx);
            }
            solved = solved && std::ranges::all_of(dx, [](double v) {
                         return std::isfinite(v);
                     });
            if (!solved) {
                dtau *= 0.25;
                continue;
            }
            // SST's unknowns are logarithms: cap their change at 0.5 per
            // step (a factor e^0.5), the only damping the exponentials need.
            double scale = 1.0;
            if (sst)
                for (int j = 0; j < c.nf; ++j)
                    for (int v = 2; v < 4; ++v)
                        scale = std::min(
                            scale, 0.5 / std::max(std::abs(
                                                      dx[std::size_t(j) * c.nv + v]),
                                                  1e-300));
            xt = x;
            for (std::size_t i = 0; i < x.size(); ++i) xt[i] += scale * dx[i];
            Norms nt;
            eval(c, xt, et, &nt);
            bool ok = std::isfinite(nt.resid) && nt.resid < 4.0 * nm.resid;            for (int j = 0; ok && j < c.nf; ++j) ok = xg(xt, c, j, 1) > -0.95;
            if (!ok) {
                dtau *= 0.25;
                continue;
            }
            x = xt;
            e = et;
            nm = nt;
            accepted = true;
        }
        if (!accepted)
            return std::unexpected(std::format(
                "oracle: no acceptable Newton step at iteration {} (resid {:.3e})",
                it, nm.resid));
        // Switched evolution relaxation off the start residual -- capped at
        // a doubling per step, so a lucky step cannot jump to pure Newton.
        dtau = std::min({1e14, 2.0 * dtau,
                         1e-2 * res0 / std::max(nm.resid, 1e-300)});
    }
    if (!(nm.resid < cs.tol))
        return std::unexpected(std::format(
            "oracle: not converged in {} iterations (resid {:.3e})", it,
            nm.resid));

    Rows r;
    fill(c, x, r);
    Channel1dSolution s;
    double mom = 0, mass = 0;
    for (int j = 1; j <= c.nf; ++j) {
        s.y.push_back(c.yc[j]);
        s.d.push_back(c.d[j]);
        s.rho.push_back(r.rho[j]);
        s.u.push_back(r.u[j]);
        s.T.push_back(r.T[j]);
        if (sst) {
            s.k.push_back(r.k[j]);
            s.omega.push_back(r.om[j]);
            s.mut.push_back(r.mut[j]);
        } else if (c.nS) {
            s.nut.push_back(r.nt[j]);
        }
        mom += r.rho[j] * r.u[j] * c.dy[j];
        mass += r.rho[j] * c.dy[j];
    }
    s.p = x.back() * c.pRef;
    s.ub = mom / mass;
    s.tauWall = c.mu * ((r.u[1] - r.u[0]) * c.Jy[0]); // mu_t is 0 on the face
    s.resU = nm.resU;
    s.resE = nm.resE;
    s.resT = nm.resT;
    s.resid = nm.resid;
    s.iters = it;
    s.ubEst = c.Ue;
    s.Twall = c.Tw;
    s.model = c.model;
    s.omegaCoef = c.omegaCoef;
    return s;
}

CompState toCompState(const Channel1dSolution& s, const Grid& g)
{
    const int nf = int(s.u.size());
    assert(g.ny == nf + 2);
    auto h = CompState::zeros(g);
    const double gam = kGammaAir;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const int src = std::clamp(j, 1, nf) - 1;
                const bool ghost = j == 0 || j == g.ny - 1;
                const double T = ghost ? std::max(2.0 * s.Twall - s.T[src],
                                                  0.05 * s.Twall)
                                       : s.T[src];
                const double rho = s.p / (kRgasBox * T);
                const double u = ghost ? -s.u[src] : s.u[src];
                const std::size_t q = g.idx(i, j, k);
                h.rho[q] = rho;
                h.mx[q] = rho * u;
                h.E[q] = s.p / (gam - 1.0) + 0.5 * rho * u * u;
            }
    return h;
}

std::vector<double> toTurb(const Channel1dSolution& s, const Grid& g, int i)
{
    const int nf = int(s.u.size());
    assert(g.ny == nf + 2 && i < ransScalars(s.model));
    if (s.model == RansModel::None) return {};
    const bool sst = ransSst(s.model);
    const std::vector<double>& f = sst ? (i == 0 ? s.k : s.omega) : s.nut;
    std::vector<double> t(g.size());
    const auto h = toCompState(s, g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int ix = 0; ix < g.nx; ++ix) {
                const int src = std::clamp(j, 1, nf) - 1;
                const bool ghost = j == 0 || j == g.ny - 1;
                const std::size_t q = g.idx(ix, j, k);
                double v = f[src];
                if (ghost && sst && i == 1) {
                    const std::size_t qs = g.idx(ix, src + 1, k);
                    const double rw = 0.5 * (h.rho[q] + h.rho[qs]);
                    v = 2.0 * s.omegaCoef / rw - f[src];
                } else if (ghost) {
                    v = -f[src];
                }
                t[q] = h.rho[q] * v;
            }
    return t;
}

} // namespace dns
