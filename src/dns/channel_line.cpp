#include "dns/channel_line.hpp"

#include "dns/curvmesh.hpp"
#include "dns/pseudo_time.hpp"
#include "dns/sgs.hpp"
#include "dns/transport.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace dns {
namespace {

constexpr int kMaxF = 7;
using Row = std::array<double, kMaxF>;

// Primitives of every grid row, plus mu_t and F1 (fluid rows computed,
// ghosts mirrored odd / even).
struct Prims
{
    std::vector<double> r, u, v, w, e, p, T, t[2], mut, f1;
};

// Cell-centre gradient at fluid row j (curvScalarGradCentre's 1D
// restriction): index central difference times the mean of the two y-face
// inverse Jacobians.
double gradC(const std::vector<double>& Jy, const std::vector<double>& f, int j)
{
    return 0.5 * (f[j + 1] - f[j - 1]) * (0.5 * (Jy[j] + Jy[j - 1]));
}

// The rate SST's mu_t and production use, from the cell-centre velocity
// gradient (gpu/rans.hip sstRate): |S| = sqrt(2 S:S) for 2003m, |curl u| for
// Vm. Only the y column of the gradient is live on the line.
double sstRateLine(RansModel m, double dudy, double dvdy, double dwdy)
{
    double gd[3][3] = {};
    gd[0][1] = dudy;
    gd[1][1] = dvdy;
    gd[2][1] = dwdy;
    if (m != RansModel::SstVm) return sgsRate(0, gd);
    const double w0 = gd[2][1] - gd[1][2], w1 = gd[0][2] - gd[2][0],
                 w2 = gd[1][0] - gd[0][1];
    return std::sqrt(w0 * w0 + w1 * w1 + w2 * w2);
}

} // namespace

ChannelLine ChannelLine::make(const CurvMetrics& m, const Grid& g,
                              std::span<const double> wallDist,
                              const ChannelLineCase& c)
{
    assert(c.model != RansModel::None && c.mu > 0 && c.Twall > 0 &&
           wallDist.size() == g.size() && m.vol.size() == g.size());
    ChannelLine L;
    L.c_ = c;
    L.N_ = g.ny;
    L.nT_ = ransScalars(c.model);
    // The GPU's gas: Transport::constant, eddy conductivity mu_t cp / Pr_t.
    const Transport tr = Transport::constant(c.gamma, c.mu);
    L.kap_ = tr.kappa(1.0);
    L.cpPrt_ = tr.cp() / kPrandtlTurb;
    for (int j = 0; j < g.ny; ++j) {
        const std::size_t q = g.idx(0, j, 0);
        // channelY: the y faces are normal to y and carry no x/z metric.
        assert(m.S[1][0][q] == 0.0 && m.S[1][2][q] == 0.0 &&
               m.Ji[1][3][q] == 0.0 && m.Ji[1][5][q] == 0.0);
        L.S_.push_back(m.S[1][1][q]);
        L.Jy_.push_back(m.Ji[1][4][q]);
        L.iv_.push_back(1.0 / m.vol[q]);
        L.dy_.push_back(m.vol[q] / m.S[1][1][q]);
        L.d_.push_back(wallDist[q]);
        double a = std::max(m.aov[0][q], m.aov[1][q]);
        if (g.nz > 1) a = std::max(a, m.aov[2][q]);
        L.aov_.push_back(a);
    }
    return L;
}

void ChannelLine::wall(std::span<double> q) const
{
    const int F = fields();
    const std::pair<int, int> ghosts[] = {{0, 1}, {N_ - 1, N_ - 2}};
    const double gm1 = c_.gamma - 1.0, R = c_.Rgas, Tw = c_.Twall;
    for (auto [gr, sr] : ghosts) {
        double* x0 = q.data() + std::size_t(gr) * F;
        const double* x1 = q.data() + std::size_t(sr) * F;
        const double r1 = x1[0];
        const double u1 = x1[1] / r1, v1 = x1[2] / r1, w1 = x1[3] / r1;
        const double ke1 = 0.5 * (u1 * u1 + v1 * v1 + w1 * w1);
        const double p1 = gm1 * (x1[4] - r1 * ke1);
        const double T0 = std::max(2.0 * Tw - p1 / (R * r1), 0.05 * Tw);
        const double r0 = p1 / (R * T0);
        x0[0] = r0;
        x0[1] = -r0 * u1;
        x0[2] = -r0 * v1;
        x0[3] = -r0 * w1;
        x0[4] = p1 / gm1 + r0 * ke1;
        if (nT_ >= 1) x0[5] = -r0 * x1[5] / r1;
        if (nT_ == 2) {
            const double ww = c_.omegaCoef / (0.5 * (r0 + r1));
            x0[6] = r0 * (2.0 * ww - x1[6] / r1);
        }
    }
}

void ChannelLine::residual(std::span<const double> q, std::span<double> Rout) const
{
    const int N = N_, F = fields();
    assert(q.size() == std::size_t(N) * F && Rout.size() == q.size());
    const double gm1 = c_.gamma - 1.0, mu = c_.mu;
    Prims P;
    for (auto* a : {&P.r, &P.u, &P.v, &P.w, &P.e, &P.p, &P.T, &P.t[0], &P.t[1],
                    &P.mut, &P.f1})
        a->assign(N, 0.0);
    for (int j = 0; j < N; ++j) {
        const double* x = q.data() + std::size_t(j) * F;
        const double r = x[0];
        P.r[j] = r;
        P.u[j] = x[1] / r;
        P.v[j] = x[2] / r;
        P.w[j] = x[3] / r;
        P.e[j] = x[4] / r;
        P.p[j] = gm1 * (x[4] - 0.5 * (x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / r);
        P.T[j] = P.p[j] / (r * c_.Rgas);
        for (int s = 0; s < nT_; ++s) P.t[s][j] = x[5 + s];
    }
    // The specific scalars: nu~ (SA) | k, omega (SST).
    std::vector<double> sk(N, 0.0), sw(N, 0.0);
    for (int j = 0; j < N; ++j) {
        if (nT_ >= 1) sk[j] = P.t[0][j] / P.r[j];
        if (nT_ == 2) sw[j] = P.t[1][j] / P.r[j];
    }
    auto spec = [&](int s, int j) { return s == 0 ? sk[j] : sw[j]; };

    // mu_t (and SST's F1) on the fluid rows, mirrored odd (F1 even) onto the
    // ghosts: never computed from ghost scalars (kMutMirrorY).
    for (int j = 1; j <= N - 2; ++j) {
        const double r = P.r[j];
        if (nT_ == 1) {
            P.mut[j] = saMut(r, spec(0, j), mu / r);
            continue;
        }
        const double Rr = sstRateLine(c_.model, gradC(Jy_, P.u, j),
                                      gradC(Jy_, P.v, j), gradC(Jy_, P.w, j));
        const double gkw = gradC(Jy_, sk, j) * gradC(Jy_, sw, j);
        const double cd = sstCd(r, gkw, sw[j], sstCdFloor(c_.model));
        const SstBlend<double> b = sstBlend(sk[j], sw[j], d_[j], mu / r, r, cd);
        P.mut[j] = sstMut(r, sk[j], sw[j], Rr, b.F2);
        P.f1[j] = b.F1;
    }
    P.mut[0] = -P.mut[1];
    P.mut[N - 1] = -P.mut[N - 2];
    P.f1[0] = P.f1[1];
    P.f1[N - 1] = P.f1[N - 2];

    // Total flux through the +y face of every row j (between j and j + 1).
    std::vector<Row> Fl(N);
    for (int j = 0; j + 1 < N; ++j) {
        const int a = j, b = j + 1;
        const double S = S_[j], Jy = Jy_[j];
        const double rm = 0.5 * (P.r[a] + P.r[b]);
        const double um = 0.5 * (P.u[a] + P.u[b]), vm = 0.5 * (P.v[a] + P.v[b]),
                     wm = 0.5 * (P.w[a] + P.w[b]);
        const double em = 0.5 * (P.e[a] + P.e[b]), pm = 0.5 * (P.p[a] + P.p[b]);
        const double Un = vm * S;
        const double mdot = rm * Un;
        Row f{};
        f[0] = mdot;
        f[1] = mdot * um;
        f[2] = mdot * vm + pm * S;
        f[3] = mdot * wm;
        f[4] = mdot * em + pm * Un;
        const double nf = 0.5 * (P.mut[a] + P.mut[b]);
        const double muf = mu + nf, kapf = kap_ + nf * cpPrt_;
        const double gu = (P.u[b] - P.u[a]) * Jy, gv = (P.v[b] - P.v[a]) * Jy,
                     gw = (P.w[b] - P.w[a]) * Jy, gT = (P.T[b] - P.T[a]) * Jy;
        // (tau . S): only S_y is live; div u = dv/dy on the line.
        const double tsx = muf * gu * S;
        const double tsy = (muf * (gv + gv) - (2.0 / 3.0) * muf * gv) * S;
        const double tsz = muf * gw * S;
        f[1] -= tsx;
        f[2] -= tsy;
        f[3] -= tsz;
        f[4] -= kapf * gT * S + um * tsx + vm * tsy + wm * tsz;
        // Scalars: upwind of the specific value on the verbatim mass flux,
        // face diffusion with every coefficient formed ON the face.
        for (int s = 0; s < nT_; ++s) {
            const double pa = spec(s, a), pb = spec(s, b);
            double coef;
            if (nT_ == 1) {
                const double ntf = 0.5 * (pa + pb);
                coef = (1.0 / sa::sigma) * rm * saDiffCoef(mu / rm, ntf);
            } else {
                const double F1f = 0.5 * (P.f1[a] + P.f1[b]);
                const double sg = s == 0 ? sstBlendC(F1f, sst::sigmaK1, sst::sigmaK2)
                                         : sstBlendC(F1f, sst::sigmaW1, sst::sigmaW2);
                coef = mu + sg * nf;
            }
            f[5 + s] = std::max(mdot, 0.0) * pa + std::min(mdot, 0.0) * pb -
                       coef * (pb - pa) * Jy * S;
        }
        Fl[j] = f;
    }

    std::ranges::fill(Rout, 0.0);
    for (int j = 1; j <= N - 2; ++j) {
        double* R = Rout.data() + std::size_t(j) * F;
        for (int v = 0; v < F; ++v) R[v] = -(Fl[j][v] - Fl[j - 1][v]) * iv_[j];
        R[1] += c_.fBody;
        R[4] += c_.fBody * P.u[j];
        const double rho = P.r[j];
        const double dudy = gradC(Jy_, P.u, j), dwdy = gradC(Jy_, P.w, j);
        if (nT_ == 1) {
            // Omega = |curl u|; the cb2 and conservative grad rho terms with
            // the SAME coefficient the face diffusion uses (gpu/rans.hip).
            const double Om = std::sqrt(dwdy * dwdy + dudy * dudy);
            const double gn = gradC(Jy_, sk, j), grho = gradC(Jy_, P.r, j);
            const double nu = mu / rho;
            R[5] += rho * saSource(sk[j], nu, Om, d_[j]).R +
                    (sa::cb2 / sa::sigma) * rho * gn * gn -
                    (1.0 / sa::sigma) * saDiffCoef(nu, sk[j]) * grho * gn;
        } else if (nT_ == 2) {
            const double Rr = sstRateLine(c_.model, dudy, gradC(Jy_, P.v, j), dwdy);
            const double gkw = gradC(Jy_, sk, j) * gradC(Jy_, sw, j);
            const SstPoint<double> Ps =
                sstSource(c_.model, rho, sk[j], sw[j], Rr, P.mut[j], P.f1[j], gkw);
            R[5] += Ps.Rk;
            R[6] += Ps.Rw;
        }
    }
}

BlockBand ChannelLine::jacobian(std::span<const double> q) const
{
    const int N = N_, F = fields(), B = unknowns(), K = band(), nc = 2 * K + 1;
    BlockBand A = BlockBand::make(N - 2, B, K);
    std::vector<double> Rp(q.size()), Rm(q.size()), qp(q.begin(), q.end()),
        h(N, 0.0);
    // q + s h on the rows of colour `col`, unknown field v, wall reapplied.
    auto shifted = [&](int col, int v, double s, std::vector<double>& R) {
        std::ranges::copy(q, qp.begin());
        for (int j = 1 + col; j <= N - 2; j += nc)
            qp[std::size_t(j) * F + v] += s * h[j];
        wall(qp);
        residual(qp, R);
    };
    for (int col = 0; col < nc; ++col)
        for (int u = 0; u < B; ++u) {
            const int v = field(u);
            for (int j = 1 + col; j <= N - 2; j += nc) {
                const double x = q[std::size_t(j) * F + v];
                const double rho = q[std::size_t(j) * F];
                const double sc = v == 0 || v == 4 ? std::abs(x)
                                                   : std::max(std::abs(x), 1e-3 * rho);
                h[j] = kImplicitFdStep * std::max(sc, 1e-12);
            }
            shifted(col, v, 1.0, Rp);
            shifted(col, v, -1.0, Rm);
            // Each row sees at most one perturbed row of this colour within
            // its band: the colours are 2K + 1 apart.
            for (int i = 1; i <= N - 2; ++i)
                for (int jj = std::max(1, i - K); jj <= std::min(N - 2, i + K); ++jj) {
                    if ((jj - 1) % nc != col) continue;
                    for (int r = 0; r < B; ++r) {
                        const std::size_t c = std::size_t(i) * F + field(r);
                        A.at(i - 1, jj - i, r, u) = (Rp[c] - Rm[c]) / (2.0 * h[jj]);
                    }
                }
        }
    return A;
}

double ChannelLine::mass(std::span<const double> q) const
{
    const int F = fields();
    double m = 0;
    for (int j = 1; j <= N_ - 2; ++j) m += q[std::size_t(j) * F] * dy_[j];
    return m;
}

void ChannelLine::pseudoStep(std::span<const double> q, double cfl,
                             std::span<double> dtau) const
{
    const int F = fields();
    assert(dtau.size() == std::size_t(N_));
    std::ranges::fill(dtau, 0.0);
    for (int j = 1; j <= N_ - 2; ++j) {
        const double* x = q.data() + std::size_t(j) * F;
        const double r = x[0], u = x[1] / r, v = x[2] / r, w = x[3] / r;
        const double p = (c_.gamma - 1.0) * (x[4] - 0.5 * r * (u * u + v * v + w * w));
        const double wave = (std::sqrt(u * u + v * v + w * w) +
                             std::sqrt(c_.gamma * p / r)) * aov_[j];
        dtau[j] = cfl / wave;
    }
}

void ChannelLine::advance(std::span<double> q, std::span<const double> dq,
                          double alpha, double M0) const
{
    const int F = fields(), B = unknowns();
    assert(dq.size() == std::size_t(N_ - 2) * B);
    for (int j = 1; j <= N_ - 2; ++j)
        for (int u = 0; u < B; ++u)
            q[std::size_t(j) * F + field(u)] += alpha * dq[std::size_t(j - 1) * B + u];
    project(q, M0);
}

void ChannelLine::project(std::span<double> q, double M0) const
{
    assert(M0 > 0);
    const int F = fields();
    const double s = M0 / mass(q);
    for (int j = 1; j <= N_ - 2; ++j)
        for (int v = 0; v < F; ++v) q[std::size_t(j) * F + v] *= s;
    wall(q);
}

bool ChannelLine::admissible(std::span<const double> q) const
{
    const int F = fields();
    if (!std::ranges::all_of(q, [](double x) { return std::isfinite(x); }))
        return false;
    for (int j = 0; j < N_; ++j) {
        const double* x = q.data() + std::size_t(j) * F;
        const double ke = 0.5 * (x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / x[0];
        if (!(x[0] > 0.0) || !(x[4] - ke > 0.0)) return false;
        const bool fluid = j >= 1 && j <= N_ - 2;
        if (fluid && nT_ == 2 && !(x[5] >= 0.0 && x[6] > 0.0)) return false;
    }
    return true;
}

} // namespace dns
