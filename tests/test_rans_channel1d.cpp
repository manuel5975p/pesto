// The FP64 RANS channel oracle (dns/rans_channel1d): the laminar limit
// against the exact compressible Poiseuille solution (order of the discrete
// wall + interior scheme), SA-neg against the MKM DNS bulk velocities at
// Re_tau 178.12 and 392.24 (a MODEL vs DNS comparison, banded provisionally
// and the measured value pinned), the log-layer slope, the grid order of ub,
// the SST wall-omega ladder (first order, Richardson limit), SA-neg against
// TMR's CFL3D 2DFDC log layer (code to code), and Newton's convergence.
// Host-only, seconds.
#include "dns/rans_channel1d.hpp"

#include "check.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

std::expected<Channel1dSolution, std::string> solve(int ny, double reTau,
                                                    RansModel m,
                                                    double beta = 2.0,
                                                    double tol = 1e-11)
{
    Channel1dCase c;
    c.ny = ny;
    c.reTau = reTau;
    c.model = m;
    c.beta = beta;
    c.tol = tol;
    return solveChannel1d(c);
}

// O1: with constant mu and v = 0 the x-momentum is mu u'' = -f whatever rho
// does, so u = y (2 - y) / (2 mu) exactly, and kappa T'' = -mu u'^2 gives
// T = Tw + f^2 / (12 mu kappa) (1 - (1 - y)^4).
void laminarLimit()
{
    const double reTau = 2.0; // mu = 0.5, u_max = 1: a low-Mach laminar flow
    double eu[3], eT[3];
    const int nys[3] = {32, 64, 128};
    for (int q = 0; q < 3; ++q) {
        auto s = solve(nys[q], reTau, RansModel::None);
        check(s.has_value(), std::format("laminar ny {} converges", nys[q]));
        if (!s) {
            std::printf("  %s\n", s.error().c_str());
            return;
        }
        const double mu = 1.0 / reTau, kap = mu * 3.5 / 0.72;
        eu[q] = eT[q] = 0;
        for (std::size_t j = 0; j < s->u.size(); ++j) {
            const double y = s->y[j];
            const double ue = y * (2 - y) / (2 * mu);
            const double Te = s->Twall + 1.0 / (12 * mu * kap) *
                                             (1 - std::pow(1 - y, 4));
            eu[q] = std::max(eu[q], std::abs(s->u[j] - ue));
            eT[q] = std::max(eT[q], std::abs(s->T[j] - Te));
        }
        std::printf("laminar ny %3d: |u - u_exact| %.3e  |T - T_exact| %.3e  "
                    "(%d Newton its, resid %.1e)\n",
                    nys[q], eu[q], eT[q], s->iters, s->resid);
    }
    for (int q = 0; q < 2; ++q) {
        const double ou = std::log2(eu[q] / eu[q + 1]);
        const double oT = std::log2(eT[q] / eT[q + 1]);
        std::printf("laminar order ny %d->%d: u %.3f  T %.3f\n", nys[q],
                    nys[q + 1], ou, oT);
        check(ou > 1.8 && ou < 2.2, "laminar u is 2nd order (1.8-2.2)");
        check(oT > 1.8 && oT < 2.2, "laminar T is 2nd order (1.8-2.2)");
    }
}

// O2: SA-neg vs MKM. Reference ub+ from the trapezoid integral of the
// published chan180.means / chan395.means mean profiles (65 / 129 points):
// 15.679 at Re_tau 178.12, 17.545 at 392.24. The 5% band is a provisional
// CEILING on model error; the measured values are pinned (regression) below.
void mkm()
{
    struct Ref
    {
        double reTau, ubMkm, ubPinned;
        int ny;
    };
    // ubPinned: this oracle's own converged value (regression pin, 1e-6 rel).
    struct Model
    {
        RansModel m;
        double pin180, pin395;
    };
    const Model models[] = {{RansModel::SaNeg, 15.806911444, 17.559550211},
                            {RansModel::Sst2003m, 15.633241440, 17.888101490},
                            {RansModel::SstVm, 15.598635910, 17.845497090}};
    for (const auto& md : models) {
        const Ref refs[] = {{178.12, 15.679, md.pin180, 64},
                            {392.24, 17.545, md.pin395, 96}};
        const std::string name(ransName(md.m));
        for (const auto& r : refs) {
            auto s = solve(r.ny, r.reTau, md.m);
            check(s.has_value(),
                  std::format("{} Re_tau {} converges", name, r.reTau));
            if (!s) {
                std::printf("  %s\n", s.error().c_str());
                continue;
            }
            const double dev = (s->ub - r.ubMkm) / r.ubMkm;
            std::printf("%s Re_tau %.2f ny %d: ub+ %.9f  MKM %.3f  (%+.2f%%)  "
                        "tau_w %.9f  Uc+ %.4f  %d its\n",
                        name.c_str(), r.reTau, r.ny, s->ub, r.ubMkm, 100 * dev,
                        s->tauWall, s->u[s->u.size() / 2], s->iters);
            check(std::abs(dev) <= 0.05,
                  name + ": ub+ within 5% of MKM (provisional)");
            check(std::abs(s->ub - r.ubPinned) <= 1e-6 * r.ubPinned,
                  name + ": ub+ regression pin (1e-6 rel)");
        }
    }
}

// SST's grid convergence is FIRST order, and that is the TMR wall value's:
// omega_w = 60 nu / (beta1 d1^2) changes with the first cell, an O(d1)
// boundary error (measured with d1 = the first centre AND with d1 = the first
// cell height: both first order, to the same limit ~15.13). The ladder is
// pinned so a change in the wall treatment shows up as a change in order or
// limit. Measured (2026-10-27): ub 15.776441287 / 15.463959275 / 15.292007660
// / 15.206956290 at ny 48/96/192/384, order 0.86 / 1.02, Richardson limit
// 15.124 (ny 768 adds 15.166241752: order 1.06, limit 15.129). The limit is
// -3.5% against MKM's 15.679 -- the ny-64 -0.29% of mkm() is wall-BC error.
void sstOrder()
{
    struct Rung
    {
        int ny;
        double ub;
    };
    const Rung rungs[4] = {{48, 15.776441287}, {96, 15.463959275},
                           {192, 15.292007660}, {384, 15.206956290}};
    double ub[4];
    for (int q = 0; q < 4; ++q) {
        auto s = solve(rungs[q].ny, 178.12, RansModel::Sst2003m, 2.0, 1e-9);
        check(s.has_value(), std::format("SST-2003m ny {} converges", rungs[q].ny));
        if (!s) {
            std::printf("  %s\n", s.error().c_str());
            return;
        }
        ub[q] = s->ub;
        std::printf("SST-2003m ub ny %3d: %.9f (pinned %.9f)\n", rungs[q].ny, ub[q],
                    rungs[q].ub);
        check(std::abs(ub[q] / rungs[q].ub - 1) <= 1e-6,
              std::format("SST-2003m ny {}: ub ladder pin (1e-6 rel)", rungs[q].ny));
    }
    const double coarse = std::log2((ub[0] - ub[1]) / (ub[1] - ub[2]));
    const double order = std::log2((ub[1] - ub[2]) / (ub[2] - ub[3]));
    const double limit = ub[3] + (ub[3] - ub[2]) / (std::pow(2.0, order) - 1);
    std::printf("SST-2003m ub apparent order %.3f (48/96/192) %.3f (96/192/384), "
                "Richardson limit %.4f (%+.2f%% vs MKM 15.679)\n",
                coarse, order, limit, 100 * (limit / 15.679 - 1));
    check(order >= 0.8 && order <= 1.3, "SST ub converges at first order (the "
                                        "wall-omega BC), 96/192/384 in [0.8, 1.3]");
    check(std::abs(limit - 15.13) <= 0.02, "SST-2003m Richardson limit 15.13 +- 0.02");
}

// O5: SA-neg code to code against NASA TMR's CFL3D SA solution of the 2D
// fully-developed channel (2DFDC), file sa_u+y+KM_cfl3d.dat ("CFL3D high Re
// channel, SA"). u+ at y+ 30/100/300/1000, linear in y+ between the file's
// points: 13.2582 / 16.1767 / 18.8002 / 21.6989. The file's own sublayer
// slope u+/y+ is 0.988 (0.9877-0.9883 over y+ 0.6-1.7, where it must be 1):
// its wall units carry a ~0.6% u_tau offset, so the comparison is made both
// raw and sublayer-normalized (u+ * 0.988). Measured (Re_tau 1e5, ny 1024,
// stretch 5; 2026-10-27): raw +0.73..+0.77%, normalized -0.44..-0.48%,
// spread 0.04%; ny 512 / 2048 move it by < 0.06% (grid-independent).
void cfl3dSa()
{
    const double reTau = 1e5;
    // ny 1024's FP64 residual floors at ~1.3e-9 (eps / dy^2): tol 1e-8.
    auto s = solve(1024, reTau, RansModel::SaNeg, 5.0, 1e-8);
    check(s.has_value(), "SA-neg Re_tau 1e5 ny 1024 converges");
    if (!s) {
        std::printf("  %s\n", s.error().c_str());
        return;
    }
    const double ref[4][2] = {{30, 13.2582}, {100, 16.1767}, {300, 18.8002}, {1000, 21.6989}};
    const double sublayer = 0.988;
    double rawMax = 0, normMax = 0, lo = HUGE_VAL, hi = -HUGE_VAL;
    int found = 0;
    const std::size_t h = s->u.size() / 2;
    for (const auto& r : ref)
        for (std::size_t j = 0; j + 1 < h; ++j) {
            const double a = s->d[j] * reTau, b = s->d[j + 1] * reTau;
            if (!(a <= r[0] && r[0] <= b)) continue;
            const double u = s->u[j] + (r[0] - a) / (b - a) * (s->u[j + 1] - s->u[j]);
            const double raw = u / r[1] - 1, norm = u * sublayer / r[1] - 1;
            std::printf("CFL3D SA y+ %5g: u+ %.5f CFL3D %.4f raw %+.3f%% "
                        "sublayer-normalized %+.3f%%\n",
                        r[0], u, r[1], 100 * raw, 100 * norm);
            rawMax = std::max(rawMax, std::abs(raw));
            normMax = std::max(normMax, std::abs(norm));
            lo = std::min(lo, raw);
            hi = std::max(hi, raw);
            ++found;
            break;
        }
    check(found == 4, "CFL3D SA: all four y+ stations inside the half channel");
    std::printf("CFL3D SA: |raw| <= %.3f%%, |normalized| <= %.3f%%, spread %.3f%% "
                "(ub+ %.6f, y+_1 %.3f, %d its)\n",
                100 * rawMax, 100 * normMax, 100 * (hi - lo), s->ub, s->d[0] * reTau,
                s->iters);
    check(normMax <= 0.0075, "CFL3D SA: sublayer-normalized u+ within 0.75%");
    check(rawMax <= 0.012, "CFL3D SA: raw u+ within 1.2%");
    check(hi - lo <= 0.001, "CFL3D SA: deviation flat over the log layer (spread <= 0.1%)");
}

// O3a: the log-layer slope du+/d ln y+ = 1/kappa, least squares over the
// cells of 30 < y+ < 0.1 Re_tau. The masterplan's band reached 0.3 Re_tau;
// there the converged slope is +3.2% (ny 384..768, also at ny 96: +3.0%)
// because the total shear stress has dropped by y/delta -- an outer-layer
// effect, not a log-layer one, Re-independent (Re_tau 2000). The log layer
// proper (y/delta < 0.1) is what the cw1 identity governs.
void logSlope()
{
    const double reTau = 2000;
    auto s = solve(384, reTau, RansModel::SaNeg, 3.0);
    check(s.has_value(), "SA-neg Re_tau 2000 converges");
    if (!s) {
        std::printf("  %s\n", s.error().c_str());
        return;
    }
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    for (std::size_t j = 0; j < s->u.size() / 2; ++j) {
        const double yp = s->d[j] * reTau;
        if (yp <= 30 || yp >= 0.1 * reTau) continue;
        const double X = std::log(yp), Y = s->u[j];
        sx += X;
        sy += Y;
        sxx += X * X;
        sxy += X * Y;
        ++n;
    }
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    std::printf("log slope over %d cells: %.4f (1/kappa = %.4f, %+.2f%%), "
                "y+_1 %.3f\n",
                n, slope, 1 / 0.41, 100 * (slope * 0.41 - 1), s->d[0] * reTau);
    check(n >= 8, "log band holds >= 8 cells");
    // Measured +0.32% (2026-10-26); banded >= 3x, a truncation-level pin.
    check(std::abs(slope * 0.41 - 1) < 0.01, "log slope is 1/kappa within 1%");
}

// O3b: grid convergence of ub (apparent order). ny 48/96/192 is still
// pre-asymptotic (measured 1.23: y+_1 0.59 at 48); 192/384/768 is not
// (1.94, and 2.17 one level up).
void ubOrder()
{
    double ub[3];
    const int nys[3] = {192, 384, 768};
    for (int q = 0; q < 3; ++q) {
        // The FP64 floor of the residual grows as eps / dy^2 (resid ~2e-11 at
        // ny 768); 1e-9 still leaves ub converged far below its grid change.
        auto s = solve(nys[q], 178.12, RansModel::SaNeg, 2.0, 1e-9);
        check(s.has_value(), std::format("ny {} converges", nys[q]));
        if (!s) {
            std::printf("  %s\n", s.error().c_str());
            return;
        }
        ub[q] = s->ub;
        std::printf("ub ny %3d: %.8f\n", nys[q], ub[q]);
    }
    const double order = std::log2((ub[0] - ub[1]) / (ub[1] - ub[2]));
    std::printf("ub apparent order %.3f\n", order);
    check(order > 1.7 && order < 2.3, "ub apparent order in 1.7-2.3");
}

// O4: Newton converges to the oracle tolerance fast.
void newton()
{
    const auto t0 = std::chrono::steady_clock::now();
    auto s = solve(64, 180, RansModel::SaNeg);
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    check(s.has_value(), "Re_tau 180 ny 64 converges");
    if (!s) return;
    std::printf("newton: %d its, resid %.2e (U %.1e E %.1e T %.1e), %.3f s\n",
                s->iters, s->resid, s->resU, s->resE, s->resT, sec);
    check(s->resid < 1e-11, "resid < 1e-11");
    check(sec < 2.0, "converges in < 2 s");
    // The force balance pins tau_w = f delta = 1 discretely.
    check(std::abs(s->tauWall - 1.0) < 1e-9, "discrete wall stress is 1");
}

} // namespace

int main()
{
    laminarLimit();
    mkm();
    logSlope();
    ubOrder();
    sstOrder();
    cfl3dSa();
    newton();
    return dnstest::finish();
}
