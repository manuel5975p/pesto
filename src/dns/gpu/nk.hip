#include "dns/gpu/nk.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/internal/reduce.hpp"
#include "dns/nk_precond.hpp"
#include "dns/rans.hpp"

#include <rocprim/device/device_reduce.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace dns::gpu {

namespace {

constexpr int kBlocks = 7; // diag, i-, i+, j-, j+, k-, k+

// The 5 + nT field pointers of a state.
struct StatePtrs
{
    Real* q[7];
    const Real* lo[7];
    int F;
};

StatePtrs ptrs(DeviceCompState& s)
{
    StatePtrs p{{s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data()},
                {},
                5 + s.nT};
    for (int i = 0; i < s.nT; ++i) p.q[5 + i] = s.t[i].data();
    if (s.compensated()) {
        for (int v = 0; v < 5; ++v) p.lo[v] = s.lo[v].data();
        for (int i = 0; i < s.nT; ++i) p.lo[5 + i] = s.tlo[i].data();
    }
    return p;
}

__global__ void kLoad(std::size_t n, StatePtrs s, double* q)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    for (int f = 0; f < s.F; ++f)
        q[f * n + c] = double(s.q[f][c]) + (s.lo[0] ? double(s.lo[f][c]) : 0.0);
}

// s = round(q + alpha D x) on the real cells (x null: q alone).
__global__ void kStore(std::size_t n, const std::uint8_t* mask, StatePtrs s,
                       const double* q, const double* ds, const Real* x,
                       double alpha)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n || !mask[c]) return;
    for (int f = 0; f < s.F; ++f) {
        double v = q[f * n + c];
        if (x) v += alpha * ds[f * n + c] * double(x[f * n + c]);
        s.q[f][c] = Real(v);
    }
}

__global__ void kCommit(std::size_t m, const double* src, double* dst)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c < m) dst[c] = src[c];
}

// q_t = q + alpha D x (FP64), every real cell.
__global__ void kTrial(std::size_t n, int F, const std::uint8_t* mask,
                       const double* q, const double* ds, const Real* x,
                       double alpha, double* qt)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    for (int f = 0; f < F; ++f)
        qt[f * n + c] = mask[c] ? q[f * n + c] +
                                      alpha * ds[f * n + c] * double(x[f * n + c])
                                : q[f * n + c];
}

// Per-cell scales (frozen per Newton step) and dtau = cfl dt_c.
__global__ void kScales(std::size_t n, int F, NkScales sc,
                        const double* q, const double* dtc, double cfl,
                        double* ds, double* dtau)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    ds[c] = sc.rho;
    ds[n + c] = ds[2 * n + c] = ds[3 * n + c] = sc.mom;
    ds[4 * n + c] = sc.E;
    if (F > 5) ds[5 * n + c] = sc.t[0];
    if (F > 6) {
        const double rw = q[6 * n + c];
        const double fl = sc.rho * sc.omegaInf;
        ds[6 * n + c] = rw > fl ? rw : fl;
    }
    dtau[c] = cfl * dtc[c];
}

// b = D^-1 R on the real cells, 0 elsewhere.
__global__ void kScaledRhs(std::size_t n, int F, const std::uint8_t* mask,
                           StatePtrs R, const double* ds, Real* b)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    for (int f = 0; f < F; ++f)
        b[f * n + c] = mask[c] ? Real(double(R.q[f][c]) / ds[f * n + c]) : Real(0);
}

// y = x / dtau - D^-1 (Rp - Rm) / (eps * w), w = 1 (forward: Rm = R0) or 2.
__global__ void kJacVec(std::size_t n, int F, const std::uint8_t* mask,
                        const Real* x, const double* dtau, StatePtrs Rp,
                        StatePtrs Rm, const double* ds, double eps, double w,
                        int withDiag, Real* y)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    for (int f = 0; f < F; ++f) {
        if (!mask[c]) {
            y[f * n + c] = Real(0);
            continue;
        }
        const double jv = (double(Rp.q[f][c]) - double(Rm.q[f][c])) /
                          (eps * w * ds[f * n + c]);
        const double d = withDiag ? double(x[f * n + c]) / dtau[c] : 0.0;
        y[f * n + c] = Real(d - jv);
    }
}

struct MaskedDot
{
    std::size_t n;
    const std::uint8_t* mask;
    const Real *a, *b;
    __host__ __device__ double operator()(int i) const
    {
        const std::size_t c = std::size_t(i) % n;
        return mask[c] ? double(a[i]) * double(b[i]) : 0.0;
    }
};

__global__ void kAxpy(std::size_t m, Real a, const Real* x, Real* y)
{
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < m) y[i] += a * x[i];
}
__global__ void kScale(std::size_t m, Real a, const Real* x, Real* y)
{
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < m) y[i] = a * x[i];
}

// --- assembly -----------------------------------------------------------------

struct AsmArgs
{
    GpuGrid g;
    CurvView m;
    ViscView vv;
    double kapScale;                     // cp / Pr_t (eddy conductivity)
    const Real *rho, *u, *v, *w, *p, *T;
    const Real *nut, *f1, *t0, *t1;      // nut null without RANS
    const double *ds, *dtau;
    const std::uint8_t* mask;
    int nT, sst;
    double gamma, Rgas;
    NkLayout lay;
    double cosd, sind;                   // the wedge fold rotation
    const Real* d;                       // wall distance (RANS)
    Real *Am, *At;
};

__device__ inline void consOf(const AsmArgs& A, std::size_t c, double q[5])
{
    const double r = A.rho[c];
    q[0] = r;
    q[1] = r * A.u[c];
    q[2] = r * A.v[c];
    q[3] = r * A.w[c];
    const double ke = 0.5 * r * (double(A.u[c]) * A.u[c] +
                                 double(A.v[c]) * A.v[c] +
                                 double(A.w[c]) * A.w[c]);
    q[4] = double(A.p[c]) / (A.gamma - 1.0) + ke;
}

__device__ inline double muAtT(const ViscView& vv, double T)
{
    return vv.sutherland ? double(vv.mu) * double(vv.ratio(Real(T)))
                         : double(vv.mu);
}

// Folding a ghost neighbour q_g = M q_c into the diagonal: B M, M the wall
// mirror diag(1, -1, -1, -1, 1) or the wedge rotation of (rho v, rho w).
__device__ inline void foldWall(double B[5][5])
{
    for (int r = 0; r < 5; ++r)
        for (int s = 1; s <= 3; ++s) B[r][s] = -B[r][s];
}
__device__ inline void foldRot(double B[5][5], double c, double s)
{
    // q_g[2] = c q2 - s q3, q_g[3] = s q2 + c q3: B M has columns 2, 3
    // B[:,2] c + B[:,3] s and -B[:,2] s + B[:,3] c.
    for (int r = 0; r < 5; ++r) {
        const double b2 = B[r][2], b3 = B[r][3];
        B[r][2] = b2 * c + b3 * s;
        B[r][3] = -b2 * s + b3 * c;
    }
}

__global__ void kAssemble(AsmArgs A)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const GpuGrid& g = A.g;
    const std::size_t n = g.size();
    if (c >= n || !A.mask[c]) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const double iv = A.m.invVol[c];
    double Jd[5][5] = {};          // dR_c / dq_c
    double Jo[6][5][5] = {};       // dR_c / dq_nb per direction slot
    double sd[2] = {0, 0};         // scalar diagonal (dR/dt)
    double so[2][6] = {};          // scalar off-diagonal
    for (int a = 0; a < 3; ++a) {
        if (a == 2 && A.lay.skipK) continue;
        for (int side = 0; side < 2; ++side) { // 0 lo, 1 hi
            const std::size_t nb = side ? c + o.p[a] : c + o.m[a];
            const std::size_t fa = side ? c : nb, fb = side ? nb : c;
            nk::FaceFrozen fr;
            for (int x = 0; x < 3; ++x) {
                fr.S[x] = A.m.S[a][x][fa];
                fr.Ji[x] = A.m.Ji[a][a * 3 + x][fa];
            }
            const double Sn = sqrt(fr.S[0] * fr.S[0] + fr.S[1] * fr.S[1] +
                                   fr.S[2] * fr.S[2]);
            double qa[5], qb[5];
            consOf(A, fa, qa);
            consOf(A, fb, qb);
            auto lam = [&](std::size_t x) {
                const double U = double(A.u[x]) * fr.S[0] +
                                 double(A.v[x]) * fr.S[1] +
                                 double(A.w[x]) * fr.S[2];
                const double cs = sqrt(A.gamma * double(A.p[x]) / double(A.rho[x]));
                return fabs(U) + cs * Sn;
            };
            const double la = lam(fa), lb = lam(fb);
            fr.lambda = la > lb ? la : lb;
            const double Tf = 0.5 * (double(A.T[fa]) + double(A.T[fb]));
            const double mut = A.nut ? 0.5 * (double(A.nut[fa]) + double(A.nut[fb]))
                                     : 0.0;
            const double mu0 = muAtT(A.vv, Tf);
            fr.mu = mu0 + mut;
            fr.kappa = mu0 * double(A.vv.kappa) / double(A.vv.mu) +
                       mut * A.kapScale;
            if (A.vv.mu == Real(0)) fr.kappa = mut * A.kapScale;
            fr.gamma = A.gamma;
            fr.Rgas = A.Rgas;
            double AL[5][5], AR[5][5];
            nk::precondBlocks(qa, qb, fr, AL, AR);
            // dR_c/dq: side hi (c = fa): -AL, -AR; side lo (c = fb): +AR, +AL.
            double Bd[5][5], Bo[5][5];
            for (int r = 0; r < 5; ++r)
                for (int s = 0; s < 5; ++s) {
                    Bd[r][s] = side ? -AL[r][s] * iv : AR[r][s] * iv;
                    Bo[r][s] = side ? -AR[r][s] * iv : AL[r][s] * iv;
                }
            // Scalars: upwind on the frozen KG mass flux, compact diffusion.
            const double rm = 0.5 * (qa[0] + qb[0]);
            const double Un = 0.5 * ((double(A.u[fa]) + A.u[fb]) * fr.S[0] +
                                     (double(A.v[fa]) + A.v[fb]) * fr.S[1] +
                                     (double(A.w[fa]) + A.w[fb]) * fr.S[2]);
            const double md = rm * Un;
            const double gS = fr.Ji[0] * fr.S[0] + fr.Ji[1] * fr.S[1] +
                              fr.Ji[2] * fr.S[2];
            double tsd[2] = {0, 0}, tso[2] = {0, 0};
            for (int e = 0; e < A.nT; ++e) {
                double K;
                if (!A.sst) {
                    const double nuf = mu0 / rm;
                    const double nta = double(A.t0[fa]) / qa[0],
                                 ntb = double(A.t0[fb]) / qb[0];
                    K = (1.0 / sa::sigma) * rm * saDiffCoef(nuf, 0.5 * (nta + ntb));
                } else {
                    const double F1f = 0.5 * (double(A.f1[fa]) + A.f1[fb]);
                    const double sg = e == 0
                        ? F1f * sst::sigmaK1 + (1 - F1f) * sst::sigmaK2
                        : F1f * sst::sigmaW1 + (1 - F1f) * sst::sigmaW2;
                    K = mu0 + sg * mut;
                }
                // F = max(md,0) phi_a + min(md,0) phi_b - K (phi_b - phi_a) gS
                const double dFa = (fmax(md, 0.0) + K * gS) / qa[0];
                const double dFb = (fmin(md, 0.0) - K * gS) / qb[0];
                tsd[e] = side ? -dFa * iv : dFb * iv;
                tso[e] = side ? -dFb * iv : dFa * iv;
            }
            // The neighbour: real, a wall ghost (fold the mirror), a wedge
            // ghost plane (fold the rotation), the far ghost or an axis ghost
            // (frozen / S = 0: nothing).
            const int slot = a * 2 + side;
            if (A.mask[nb] && nb != c) {
                for (int r = 0; r < 5; ++r)
                    for (int s = 0; s < 5; ++s) {
                        Jd[r][s] += Bd[r][s];
                        Jo[slot][r][s] += Bo[r][s];
                    }
                for (int e = 0; e < A.nT; ++e) {
                    sd[e] += tsd[e];
                    so[e][slot] += tso[e];
                }
                continue;
            }
            const bool wall = a == 1 && side == 0 && j == 1;
            const bool plane = a == 2 && A.lay.wedge;
            if (wall) foldWall(Bo);
            if (plane) foldRot(Bo, A.cosd, side ? A.sind : -A.sind);
            const bool fold = wall || plane;
            for (int r = 0; r < 5; ++r)
                for (int s = 0; s < 5; ++s)
                    Jd[r][s] += Bd[r][s] + (fold ? Bo[r][s] : 0.0);
            for (int e = 0; e < A.nT; ++e)
                sd[e] += tsd[e] + (wall ? -tso[e] : plane ? tso[e] : 0.0);
        }
    }
    // Point-implicit source rates (SA J; SST beta* omega, 2 beta omega).
    if (A.nT > 0) {
        const double r = A.rho[c];
        if (!A.sst) {
            const Real* vc[3] = {A.u, A.v, A.w};
            Real gd[3][3];
            curvVelGradCentre(A.m, o, c,
                              [&](std::size_t x, int comp) { return vc[comp][x]; },
                              gd);
            const double w0 = gd[2][1] - gd[1][2], w1 = gd[0][2] - gd[2][0],
                         w2 = gd[1][0] - gd[0][1];
            const double Om = sqrt(w0 * w0 + w1 * w1 + w2 * w2);
            const double nu = muAtT(A.vv, A.T[c]) / r;
            const auto P = saSource(double(A.t0[c]) / r, nu, Om, double(A.d[c]));
            sd[0] -= P.J;
        } else {
            const double om = double(A.t1[c]) / r;
            const double F1 = A.f1[c];
            const double beta = F1 * sst::beta1 + (1 - F1) * sst::beta2;
            sd[0] -= sst::betaStar * om;
            sd[1] -= 2.0 * beta * om;
        }
    }
    // P = I/dtau - J, scaled: Pt[r][s] = P[r][s] ds_s(nb) / ds_r(c).
    const double* ds = A.ds;
    auto dsAt = [&](int f, std::size_t x) { return ds[f * n + x]; };
    const double idt = 1.0 / A.dtau[c];
    for (int r = 0; r < 5; ++r)
        for (int s = 0; s < 5; ++s) {
            const double P = -Jd[r][s] + (r == s ? idt : 0.0);
            A.Am[std::size_t(r * 5 + s) * n + c] =
                Real(P * dsAt(s, c) / dsAt(r, c));
        }
    for (int e = 0; e < A.nT; ++e)
        A.At[std::size_t(e * kBlocks) * n + c] = Real(-sd[e] + idt);
    for (int slot = 0; slot < 6; ++slot) {
        const int a = slot / 2, side = slot % 2;
        const std::size_t nb = side ? c + o.p[a] : c + o.m[a];
        const bool real = A.mask[nb] && nb != c && !(a == 2 && A.lay.skipK);
        for (int r = 0; r < 5; ++r)
            for (int s = 0; s < 5; ++s) {
                const double P = real ? -Jo[slot][r][s] * dsAt(s, nb) / dsAt(r, c)
                                      : 0.0;
                A.Am[std::size_t((1 + slot) * 25 + r * 5 + s) * n + c] = Real(P);
            }
        for (int e = 0; e < A.nT; ++e)
            A.At[std::size_t(e * kBlocks + 1 + slot) * n + c] =
                Real(real ? -so[e][slot] * dsAt(5 + e, nb) / dsAt(5 + e, c) : 0.0);
    }
}

// --- lines --------------------------------------------------------------------

// Gauss-Jordan inverse of a 5x5 with partial pivoting; false when singular.
__device__ inline bool inv5(double M[5][5], double I[5][5])
{
    for (int r = 0; r < 5; ++r)
        for (int s = 0; s < 5; ++s) I[r][s] = r == s ? 1.0 : 0.0;
    for (int col = 0; col < 5; ++col) {
        int pr = col;
        for (int r = col + 1; r < 5; ++r)
            if (fabs(M[r][col]) > fabs(M[pr][col])) pr = r;
        const double piv = M[pr][col];
        if (!(fabs(piv) > 1e-300) || !isfinite(piv)) return false;
        if (pr != col)
            for (int s = 0; s < 5; ++s) {
                double t = M[col][s]; M[col][s] = M[pr][s]; M[pr][s] = t;
                t = I[col][s]; I[col][s] = I[pr][s]; I[pr][s] = t;
            }
        const double ip = 1.0 / piv;
        for (int s = 0; s < 5; ++s) {
            M[col][s] *= ip;
            I[col][s] *= ip;
        }
        for (int r = 0; r < 5; ++r) {
            if (r == col) continue;
            const double f = M[r][col];
            if (f == 0.0) continue;
            for (int s = 0; s < 5; ++s) {
                M[r][s] -= f * M[col][s];
                I[r][s] -= f * I[col][s];
            }
        }
    }
    for (int r = 0; r < 5; ++r)
        for (int s = 0; s < 5; ++s)
            if (!isfinite(I[r][s])) return false;
    return true;
}

struct LineArgs
{
    GpuGrid g;
    NkLayout lay;
    int nT;
    const Real *Am, *At;
    Real *Fm, *Lm, *Gm; // Dinv'_j, L'_j and G_j = Dinv'_j U_j per cell
    Real *Ft, *Lt, *Gt; // their scalar twins
    int* err;
};

// One thread per line (i, k): block Thomas along j. The substitution then
// needs y_j = rhs_j - L'_j y_{j-1} and x_j = Dinv'_j y_j - G_j x_{j+1}: one
// 5x5 product per row on each serial chain.
__global__ void kFactor(LineArgs A)
{
    const int ni = A.lay.iHi - A.lay.iLo + 1;
    const int t = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (t >= ni * (A.lay.kHi - A.lay.kLo + 1)) return;
    const int i = A.lay.iLo + t % ni, k = A.lay.kLo + t / ni;
    const GpuGrid& g = A.g;
    const std::size_t n = g.size();
    auto blk = [&](int b, std::size_t c, double B[5][5]) {
        for (int r = 0; r < 5; ++r)
            for (int s = 0; s < 5; ++s)
                B[r][s] = A.Am[std::size_t(b * 25 + r * 5 + s) * n + c];
    };
    double Dinv[5][5], Uprev[5][5];
    double tdinv[2] = {0, 0}, tuprev[2] = {0, 0};
    for (int j = 1; j <= g.ny - 2; ++j) {
        const std::size_t c = gidx(g, i, j, k);
        double D[5][5], L[5][5] = {};
        blk(0, c, D);
        if (j > 1) {
            double Lo[5][5];
            blk(3, c, Lo); // j- block
            for (int r = 0; r < 5; ++r)
                for (int s = 0; s < 5; ++s) {
                    double a = 0;
                    for (int q = 0; q < 5; ++q) a += Lo[r][q] * Dinv[q][s];
                    L[r][s] = a;
                }
            for (int r = 0; r < 5; ++r)
                for (int s = 0; s < 5; ++s) {
                    double a = 0;
                    for (int q = 0; q < 5; ++q) a += L[r][q] * Uprev[q][s];
                    D[r][s] -= a;
                }
        }
        if (!inv5(D, Dinv)) {
            atomicExch(A.err, 1);
            return;
        }
        blk(4, c, Uprev); // j+ block of this row
        for (int r = 0; r < 5; ++r)
            for (int s = 0; s < 5; ++s) {
                double gq = 0;
                for (int q = 0; q < 5; ++q) gq += Dinv[r][q] * Uprev[q][s];
                A.Fm[std::size_t(r * 5 + s) * n + c] = Real(Dinv[r][s]);
                A.Lm[std::size_t(r * 5 + s) * n + c] = Real(L[r][s]);
                A.Gm[std::size_t(r * 5 + s) * n + c] = Real(gq);
            }
        for (int e = 0; e < A.nT; ++e) {
            const double a = A.At[std::size_t(e * kBlocks) * n + c];
            double l = 0, d = a;
            if (j > 1) {
                l = A.At[std::size_t(e * kBlocks + 3) * n + c] * tdinv[e];
                d -= l * tuprev[e];
            }
            if (!(fabs(d) > 1e-300) || !isfinite(d)) {
                atomicExch(A.err, 1);
                return;
            }
            tdinv[e] = 1.0 / d;
            tuprev[e] = A.At[std::size_t(e * kBlocks + 4) * n + c];
            A.Ft[std::size_t(e) * n + c] = Real(tdinv[e]);
            A.Lt[std::size_t(e) * n + c] = Real(l);
            A.Gt[std::size_t(e) * n + c] = Real(tdinv[e] * tuprev[e]);
        }
    }
}

struct SweepArgs
{
    LineArgs L;
    const std::uint8_t* mask;
    const Real* r;  // F x n
    Real* z;        // F x n
    Real* y;        // F x n: the forward substitution of the line being solved
    int F;
};

// Line i, one 32-lane workgroup per plane k (k couples nothing: nz == 1 skips
// it, the wedge's k neighbours are folded ghosts). Phase 1, all rows in
// parallel: rhs_j = r_j - (i+-, k+- couplings) z. Phase 2, the serial
// chain y_j = rhs_j - L'_j y_{j-1}: lane l < 25 forms one product of the 5x5,
// the five row sums are a shared-memory reduction. Phase 3, all rows:
// w_j = Dinv'_j y_j. Phase 4, the chain x_j = w_j - G_j x_{j+1} into z.
constexpr int kLineLanes = 32;
__global__ void __launch_bounds__(kLineLanes)
kSweepLine(SweepArgs A, int i)
{
    const GpuGrid& g = A.L.g;
    const std::size_t n = g.size();
    const int k = A.L.lay.kLo + int(blockIdx.x);
    const int nf = g.ny - 2, F = A.F, nT = A.L.nT;
    const int lane = threadIdx.x;
    auto cell = [&](int jj) { return gidx(g, i, 1 + jj, k); };
    // Phase 1.
    for (int jj = lane; jj < nf; jj += kLineLanes) {
        const int j = 1 + jj;
        const std::size_t c = cell(jj);
        const Offsets o = offsetsAt(g, i, j, k);
        double y[7];
        for (int f = 0; f < F; ++f) y[f] = A.r[f * n + c];
        for (int slot = 0; slot < 6; ++slot) {
            if (slot == 2 || slot == 3) continue; // j: in-line
            const int a = slot / 2, side = slot % 2;
            const std::size_t nb = side ? c + o.p[a] : c + o.m[a];
            if (!A.mask[nb] || nb == c) continue;
            for (int r = 0; r < 5; ++r) {
                double sum = 0;
                for (int q = 0; q < 5; ++q)
                    sum += double(A.L.Am[std::size_t((1 + slot) * 25 + r * 5 + q) *
                                             n + c]) *
                           double(A.z[q * n + nb]);
                y[r] -= sum;
            }
            for (int e = 0; e < nT; ++e)
                y[5 + e] -=
                    double(A.L.At[std::size_t(e * kBlocks + 1 + slot) * n + c]) *
                    double(A.z[(5 + e) * n + nb]);
        }
        for (int f = 0; f < F; ++f) A.y[f * n + c] = Real(y[f]);
    }
    __syncthreads();
    // Phases 2 and 4: one serial chain v_j = b_j - M_j v_{j -+ 1}.
    __shared__ double prod[kLineLanes];
    __shared__ double vprev[7];
    const int r = lane / 5, q = lane % 5;
    auto chain = [&](const Real* M, const Real* Ms, const Real* b, Real* out,
                     bool down) {
        if (lane < 7) vprev[lane] = 0.0;
        __syncthreads();
        for (int step = 0; step < nf; ++step) {
            const int jj = down ? nf - 1 - step : step;
            const std::size_t c = cell(jj);
            if (lane < 25)
                prod[lane] = step ? double(M[std::size_t(lane) * n + c]) * vprev[q]
                                  : 0.0;
            else if (lane < 25 + nT)
                prod[lane] = step ? double(Ms[std::size_t(lane - 25) * n + c]) *
                                        vprev[5 + lane - 25]
                                  : 0.0;
            __syncthreads();
            double v = 0;
            if (lane < 5) {
                v = double(b[std::size_t(lane) * n + c]) -
                    (prod[lane * 5] + prod[lane * 5 + 1] + prod[lane * 5 + 2] +
                     prod[lane * 5 + 3] + prod[lane * 5 + 4]);
            } else if (lane < 5 + nT) {
                v = double(b[std::size_t(lane) * n + c]) - prod[25 + lane - 5];
            }
            __syncthreads();
            if (lane < 5 + nT) {
                vprev[lane] = v;
                out[std::size_t(lane) * n + c] = Real(v);
            }
            __syncthreads();
        }
        (void)r;
    };
    chain(A.L.Lm, A.L.Lt, A.y, A.y, false);
    // Phase 3: w_j = Dinv'_j y_j, in place.
    for (int jj = lane; jj < nf; jj += kLineLanes) {
        const std::size_t c = cell(jj);
        double y[7], w[7];
        for (int f = 0; f < F; ++f) y[f] = A.y[f * n + c];
        for (int rr = 0; rr < 5; ++rr) {
            double sum = 0;
            for (int qq = 0; qq < 5; ++qq)
                sum += double(A.L.Fm[std::size_t(rr * 5 + qq) * n + c]) * y[qq];
            w[rr] = sum;
        }
        for (int e = 0; e < nT; ++e)
            w[5 + e] = double(A.L.Ft[std::size_t(e) * n + c]) * y[5 + e];
        for (int f = 0; f < F; ++f) A.y[f * n + c] = Real(w[f]);
    }
    __syncthreads();
    chain(A.L.Gm, A.L.Gt, A.y, A.z, true);
}

// Step bounds per real cell: the largest alpha in (0, 1] keeping rho and E
// within 20%, rho omega above -50%, rho k above -90% (SST).
struct StepBound
{
    std::size_t n;
    int F;
    const std::uint8_t* mask;
    const double *q, *ds;
    const Real* x;
    __host__ __device__ double operator()(int c) const
    {
        if (!mask[c]) return 1.0;
        double a = 1.0;
        auto change = [&](int f, double frac) {
            const double d = ds[f * n + c] * double(x[f * n + c]);
            const double v = q[f * n + c];
            if (d != 0.0) {
                const double b = frac * fabs(v) / fabs(d);
                if (b < a) a = b;
            }
        };
        auto decrease = [&](int f, double frac) {
            const double d = ds[f * n + c] * double(x[f * n + c]);
            const double v = q[f * n + c];
            if (d < 0.0 && v > 0.0) {
                const double b = frac * v / -d;
                if (b < a) a = b;
            }
        };
        change(0, 0.2);
        change(4, 0.2);
        if (F > 6) {
            decrease(5, 0.9);
            decrease(6, 0.5);
        }
        return isfinite(a) ? a : 0.0;
    }
};

} // namespace

std::expected<DeviceNk, hipError_t>
DeviceNk::make(DeviceComp& dev, const std::vector<std::uint8_t>& mask,
               const Config& c, Hook hook)
{
    assert(dev.curvilinear() && dev.nTurb() == ransScalars(c.model));
    // The colour sweep needs lines that couple only in i (precond()).
    assert(c.lay.skipK || c.lay.wedge);
    DeviceNk k;
    k.dev_ = &dev;
    k.g_ = dev.grid();
    k.c_ = c;
    k.hook_ = std::move(hook);
    k.nT_ = ransScalars(c.model);
    k.F_ = 5 + k.nT_;
    k.n_ = k.g_.size();
    k.count_ = double(std::ranges::count(mask, std::uint8_t(1)));
    const std::size_t n = k.n_, F = std::size_t(k.F_);
    auto mk = [&](auto& b, std::size_t sz) -> hipError_t {
        auto e = std::remove_reference_t<decltype(b)>::make(sz);
        if (!e) return e.error();
        b = std::move(*e);
        return hipSuccess;
    };
    hipError_t e = hipSuccess;
    for (auto [b, sz] : {std::pair{&k.q64_, F * n}, {&k.qt64_, F * n},
                         {&k.ds_, F * n}, {&k.dtau_, n}})
        if ((e = mk(*b, sz)) != hipSuccess) return std::unexpected(e);
    for (auto [b, sz] : {std::pair{&k.V_, F * n * std::size_t(c.restart + 1)},
                         {&k.w_, F * n}, {&k.z_, F * n}, {&k.x_, F * n},
                         {&k.Am_, std::size_t(kBlocks) * 25 * n},
                         {&k.At_, std::size_t(kBlocks) * std::max(1, k.nT_) * n},
                         {&k.Lm_, 25 * n}, {&k.Fm_, 25 * n}, {&k.Gm_, 25 * n},
                         {&k.Lt_, std::size_t(std::max(1, k.nT_)) * n},
                         {&k.Ft_, std::size_t(std::max(1, k.nT_)) * n},
                         {&k.Gt_, std::size_t(std::max(1, k.nT_)) * n},
                         {&k.rhs_, F * n}})
        if ((e = mk(*b, sz)) != hipSuccess) return std::unexpected(e);
    if ((e = mk(k.mask_, n)) != hipSuccess) return std::unexpected(e);
    if ((e = k.mask_.upload(mask.data())) != hipSuccess) return std::unexpected(e);
    if ((e = mk(k.err_, 1)) != hipSuccess) return std::unexpected(e);
    for (auto* s : {&k.R0_, &k.Rp_, &k.Rm_, &k.sp_}) {
        auto st = DeviceCompState::make(n, k.nT_);
        if (!st) return std::unexpected(st.error());
        *s = std::move(*st);
    }
    return k;
}

hipError_t DeviceNk::load(const DeviceCompState& s)
{
    assert(s.nT == nT_);
    auto p = ptrs(const_cast<DeviceCompState&>(s));
    kLoad<<<launchBlocks(n_), blockSize>>>(n_, p, q64_.data());
    checkLastKernel("kLoad");
    return hipGetLastError();
}

void DeviceNk::store(DeviceCompState& s) const
{
    assert(!s.compensated() && s.nT == nT_);
    kStore<<<launchBlocks(n_), blockSize>>>(n_, mask_.data(), ptrs(s),
                                            q64_.data(), nullptr, nullptr, 0.0);
    checkLastKernel("kStore");
    hook_(s);
}

double DeviceNk::dot(const Real* a, const Real* b, ReduceScratch& r) const
{
    return internal::reduce(r, n_ * F_, MaskedDot{n_, mask_.data(), a, b},
                            rocprim::plus<double>(), 0.0);
}

double DeviceNk::residual(const DeviceCompState& s, ReduceScratch& r)
{
    dev_->residualInto(s, R0_);
    // Scales may not be set yet at the first call: use the frozen ones of
    // the current (or first) Newton step.
    kScales<<<launchBlocks(n_), blockSize>>>(
        n_, F_, c_.sc, q64_.data(),
        dev_->localDtCells().data(), cfl_, ds_.data(), dtau_.data());
    checkLastKernel("kScales");
    kScaledRhs<<<launchBlocks(n_), blockSize>>>(n_, F_, mask_.data(), ptrs(R0_),
                                                ds_.data(), w_.data());
    checkLastKernel("kScaledRhs");
    return std::sqrt(dot(w_.data(), w_.data(), r) / (count_ * F_));
}

void DeviceNk::jacVec(const DeviceBuffer<Real>& v, bool central, double eps,
                      DeviceBuffer<Real>& out, ReduceScratch& r)
{
    const double vn = std::sqrt(dot(v.data(), v.data(), r) / (count_ * F_));
    const double e = eps / (vn > 0 ? vn : 1.0);
    // q + e D v
    kStore<<<launchBlocks(n_), blockSize>>>(n_, mask_.data(), ptrs(sp_),
                                            q64_.data(), ds_.data(), v.data(), e);
    checkLastKernel("kStore");
    hook_(sp_);
    dev_->residualInto(sp_, Rp_);
    if (central) {
        kStore<<<launchBlocks(n_), blockSize>>>(n_, mask_.data(), ptrs(sp_),
                                                q64_.data(), ds_.data(), v.data(),
                                                -e);
        checkLastKernel("kStore");
        hook_(sp_);
        dev_->residualInto(sp_, Rm_);
    }
    kJacVec<<<launchBlocks(n_), blockSize>>>(
        n_, F_, mask_.data(), v.data(), dtau_.data(), ptrs(Rp_),
        ptrs(central ? Rm_ : R0_), ds_.data(), e, central ? 2.0 : 1.0, 0,
        out.data());
    checkLastKernel("kJacVec");
}

void DeviceNk::matvec(const Real* x, Real* y, ReduceScratch& r)
{
    const double vn = std::sqrt(dot(x, x, r) / (count_ * F_));
    const double e = c_.fdStep / (vn > 0 ? vn : 1.0);
    kStore<<<launchBlocks(n_), blockSize>>>(n_, mask_.data(), ptrs(sp_),
                                            q64_.data(), ds_.data(), x, e);
    checkLastKernel("kStore");
    hook_(sp_);
    dev_->residualInto(sp_, Rp_);
    kJacVec<<<launchBlocks(n_), blockSize>>>(n_, F_, mask_.data(), x,
                                             dtau_.data(), ptrs(Rp_), ptrs(R0_),
                                             ds_.data(), e, 1.0, 1, y);
    checkLastKernel("kJacVec");
}

std::expected<void, std::string> DeviceNk::assemble(DeviceCompState& s,
                                                    double cfl,
                                                    ReduceScratch& r)
{
    cfl_ = cfl;
    // dt_c at cfl 1 (the localDt bound), then scales and dtau.
    (void)dev_->localDt(s, r, 1.0);
    kScales<<<launchBlocks(n_), blockSize>>>(
        n_, F_, c_.sc, q64_.data(),
        dev_->localDtCells().data(), cfl_, ds_.data(), dtau_.data());
    checkLastKernel("kScales");
    if (dev_->rans()) dev_->updateMut(s);
    else dev_->primitives(s);
    const DeviceCurv* cv = dev_->curvTables();
    const Transport& tr = dev_->transport();
    AsmArgs A{};
    A.g = g_;
    A.m = cv->view();
    A.vv = viscView(tr);
    A.kapScale = tr.cp() / c_.prt;
    A.rho = s.rho.data();
    A.u = dev_->u().data();
    A.v = dev_->v().data();
    A.w = dev_->w().data();
    A.p = dev_->p().data();
    A.T = dev_->T().data();
    A.nut = dev_->rans() ? dev_->mut().data() : nullptr;
    A.f1 = ransSst(c_.model) ? dev_->f1().data() : nullptr;
    A.t0 = nT_ > 0 ? s.t[0].data() : nullptr;
    A.t1 = nT_ > 1 ? s.t[1].data() : nullptr;
    A.ds = ds_.data();
    A.dtau = dtau_.data();
    A.mask = mask_.data();
    A.nT = nT_;
    A.sst = ransSst(c_.model) ? 1 : 0;
    A.gamma = c_.gamma;
    A.Rgas = c_.Rgas;
    A.lay = c_.lay;
    A.cosd = std::cos(c_.lay.dphi);
    A.sind = std::sin(c_.lay.dphi);
    A.d = dev_->rans() ? dev_->wallDistance().data() : nullptr;
    A.Am = Am_.data();
    A.At = At_.data();
    kAssemble<<<unsigned((n_ + 127) / 128), 128>>>(A);
    checkLastKernel("kAssemble");
    const int zero = 0;
    (void)err_.upload(&zero);
    LineArgs L{g_,         c_.lay,      nT_,         Am_.data(), At_.data(),
               Fm_.data(), Lm_.data(),  Gm_.data(),  Ft_.data(), Lt_.data(),
               Gt_.data(), err_.data()};
    const int lines = (c_.lay.iHi - c_.lay.iLo + 1) * (c_.lay.kHi - c_.lay.kLo + 1);
    kFactor<<<launchBlocks(std::size_t(lines)), blockSize>>>(L);
    checkLastKernel("kFactor");
    int bad = 0;
    (void)err_.download(&bad);
    if (bad) return std::unexpected("singular or non-finite line pivot in the "
                                    "Newton-Krylov preconditioner");
    return {};
}

void DeviceNk::precond(const DeviceBuffer<Real>& rr, DeviceBuffer<Real>& z)
{
    (void)hipMemset(z.data(), 0, z.size() * sizeof(Real));
    const LineArgs L{g_,         c_.lay,     nT_,        Am_.data(),
                     At_.data(), Fm_.data(), Lm_.data(), Gm_.data(),
                     Ft_.data(), Lt_.data(), Gt_.data(), err_.data()};
    const SweepArgs S{L, mask_.data(), rr.data(), z.data(), rhs_.data(), F_};
    // Symmetric LEXICOGRAPHIC line Gauss-Seidel: i ascending, then
    // descending (the last line not repeated); a cyclic i's wrap neighbour
    // is simply the latest value. Measured (gfx1100, cylinder Re 40 256 x
    // 96): 5-8 Krylov per Newton step at the CFL cap, against 200 (the cap,
    // stalled) for red-black lines -- which the plan forbids anyway.
    const unsigned nk = unsigned(c_.lay.kHi - c_.lay.kLo + 1);
    for (int i = c_.lay.iLo; i <= c_.lay.iHi; ++i)
        kSweepLine<<<nk, kLineLanes>>>(S, i);
    for (int i = c_.lay.iHi - 1; i >= c_.lay.iLo; --i)
        kSweepLine<<<nk, kLineLanes>>>(S, i);
    checkLastKernel("kSweepLine");
}

std::expected<NkLinear, std::string> DeviceNk::solve(DeviceCompState& s,
                                                     double cfl,
                                                     ReduceScratch& r)
{
    if (auto a = assemble(s, cfl, r); !a) return std::unexpected(a.error());
    const std::size_t m = n_ * F_;
    const int M = c_.restart;
    // b = D^-1 R0 (w_ from residual() was with the previous scales: redo).
    kScaledRhs<<<launchBlocks(n_), blockSize>>>(n_, F_, mask_.data(), ptrs(R0_),
                                                ds_.data(), w_.data());
    checkLastKernel("kScaledRhs");
    (void)hipMemset(x_.data(), 0, m * sizeof(Real));
    const double bn = std::sqrt(dot(w_.data(), w_.data(), r));
    NkLinear out;
    if (!(bn > 0)) {
        out.converged = true;
        return out;
    }
    double rn = bn;
    std::vector<double> H(std::size_t(M + 1) * M), cs(M), sn(M), gv(M + 1);
    auto Vk = [&](int j) { return V_.data() + std::size_t(j) * m; };
    // r0 = b - A x (x = 0 first cycle: r0 = b).
    (void)hipMemcpy(Vk(0), w_.data(), m * sizeof(Real), hipMemcpyDeviceToDevice);
    while (out.krylov < c_.maxKrylov) {
        const double beta = std::sqrt(dot(Vk(0), Vk(0), r));
        if (!std::isfinite(beta)) return std::unexpected("GMRES breakdown");
        kScale<<<launchBlocks(m), blockSize>>>(m, Real(1.0 / beta), Vk(0), Vk(0));
        std::fill(gv.begin(), gv.end(), 0.0);
        gv[0] = beta;
        int jUsed = 0;
        for (int j = 0; j < M && out.krylov < c_.maxKrylov; ++j) {
            // w = A P^-1 v_j
            (void)hipMemcpy(w_.data(), Vk(j), m * sizeof(Real),
                            hipMemcpyDeviceToDevice);
            precond(w_, z_);
            matvec(z_.data(), w_.data(), r);
            // CGS2
            for (int pass = 0; pass < 2; ++pass)
                for (int q = 0; q <= j; ++q) {
                    const double h = dot(Vk(q), w_.data(), r);
                    H[std::size_t(q) * M + j] += h;
                    kAxpy<<<launchBlocks(m), blockSize>>>(m, Real(-h), Vk(q),
                                                          w_.data());
                }
            const double hn = std::sqrt(dot(w_.data(), w_.data(), r));
            H[std::size_t(j + 1) * M + j] = hn;
            ++out.krylov;
            jUsed = j + 1;
            if (!std::isfinite(hn)) return std::unexpected("GMRES breakdown");
            if (hn > 0)
                kScale<<<launchBlocks(m), blockSize>>>(m, Real(1.0 / hn), w_.data(),
                                                       Vk(j + 1));
            // Givens
            for (int q = 0; q < j; ++q) {
                const double a = H[std::size_t(q) * M + j],
                             b = H[std::size_t(q + 1) * M + j];
                H[std::size_t(q) * M + j] = cs[q] * a + sn[q] * b;
                H[std::size_t(q + 1) * M + j] = -sn[q] * a + cs[q] * b;
            }
            const double a = H[std::size_t(j) * M + j], b = hn;
            const double d = std::hypot(a, b);
            cs[j] = d > 0 ? a / d : 1.0;
            sn[j] = d > 0 ? b / d : 0.0;
            H[std::size_t(j) * M + j] = d;
            H[std::size_t(j + 1) * M + j] = 0.0;
            gv[j + 1] = -sn[j] * gv[j];
            gv[j] = cs[j] * gv[j];
            rn = std::abs(gv[j + 1]);
            if (rn <= c_.forcing * bn || hn == 0) break;
        }
        // y = H^-1 g; x += P^-1 (V y)
        std::vector<double> y(jUsed);
        for (int q = jUsed - 1; q >= 0; --q) {
            double v = gv[q];
            for (int t = q + 1; t < jUsed; ++t) v -= H[std::size_t(q) * M + t] * y[t];
            y[q] = v / H[std::size_t(q) * M + q];
        }
        (void)hipMemset(w_.data(), 0, m * sizeof(Real));
        for (int q = 0; q < jUsed; ++q)
            kAxpy<<<launchBlocks(m), blockSize>>>(m, Real(y[q]), Vk(q), w_.data());
        precond(w_, z_);
        kAxpy<<<launchBlocks(m), blockSize>>>(m, Real(1), z_.data(), x_.data());
        std::fill(H.begin(), H.end(), 0.0);
        if (rn <= c_.forcing * bn) break;
        // restart: r = b - A x into V_0
        matvec(x_.data(), w_.data(), r);
        kScaledRhs<<<launchBlocks(n_), blockSize>>>(n_, F_, mask_.data(),
                                                    ptrs(R0_), ds_.data(), Vk(0));
        kAxpy<<<launchBlocks(m), blockSize>>>(m, Real(-1), w_.data(), Vk(0));
        rn = std::sqrt(dot(Vk(0), Vk(0), r));
    }
    out.relResidual = rn / bn;
    out.converged = rn <= c_.forcing * bn;
    return out;
}

double DeviceNk::stepLength(ReduceScratch& r)
{
    return internal::reduce(r, n_,
                            StepBound{n_, F_, mask_.data(), q64_.data(),
                                      ds_.data(), x_.data()},
                            rocprim::minimum<double>(), 1.0);
}

void DeviceNk::trial(double alpha, DeviceCompState& s)
{
    kTrial<<<launchBlocks(n_), blockSize>>>(n_, F_, mask_.data(), q64_.data(),
                                            ds_.data(), x_.data(), alpha,
                                            qt64_.data());
    checkLastKernel("kTrial");
    kStore<<<launchBlocks(n_), blockSize>>>(n_, mask_.data(), ptrs(s),
                                            qt64_.data(), nullptr, nullptr, 0.0);
    checkLastKernel("kStore");
    hook_(s);
}

void DeviceNk::accept()
{
    const std::size_t m = n_ * F_;
    kCommit<<<launchBlocks(m), blockSize>>>(m, qt64_.data(), q64_.data());
    checkLastKernel("kCommit");
}

} // namespace dns::gpu
