// dns/channel_line: the FP64 host y-line operator the implicit steady RANS
// march preconditions with (docs/rans.md, "Implicit pseudo-time"):
//   (a) its residual vanishes at the FP64 oracle's discrete solution (the
//       same operator, solved independently in other variables)
//   (b) its structural band is EXACTLY K (SA 1, SST 2), ghost chain included
//   (c) the FD Jacobian against a central FD of the residual
//   (d) the host Newton driver (line as residual AND Jacobian) from the
//       Reichardt IC certifies in <= 35 iterations onto the oracle
//   (e) R-a: a damped stall is never certified. The masterplan's case (SST
//       Re_tau 1e4 ny 256 from CFL 1e8) stalled only under a FORWARD-FD
//       Jacobian; with the central one it converges (9 iterations, recorded
//       here), so the stall is forced: every step damped to alpha 0.5. The
//       armed stop never certifies it; an unarmed one does.
//   (f) the mass projection keeps every specific quantity and pins the mass
// plus the host cost of one Jacobian + solve at ny 128/256/512. Seconds.
#include "dns/channel_line.hpp"
#include "dns/pseudo_time.hpp"
#include "channel_line_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <random>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

Channel1dCase caseOf(RansModel model, double reTau, int ny, double beta)
{
    Channel1dCase cc;
    cc.model = model;
    cc.reTau = reTau;
    cc.ny = ny;
    cc.beta = beta;
    cc.tol = 1e-8;
    return cc;
}

std::string nameOf(const Channel1dCase& cc)
{
    return std::format("{} Re_tau {:g} ny {}", ransName(cc.model), cc.reTau, cc.ny);
}

// (a) Per-volume residual of the oracle state, solved to the oracle's own
// default tolerance (1e-11): at 1e-8 the oracle's unconverged remainder
// (mx 1e-9, E 2e-7) is what this would measure, not the operator. Measured
// mx ~2e-13, E ~5e-10 / 1e-9 (SA / SST); thresholds 100-1000x above. SST's
// omega row is relative to its destruction scale beta1 rho omega^2 (~1e9 at
// the wall, so FP64 rounding alone is ~1e-7 absolute).
void oracleResidual(Channel1dCase cc)
{
    cc.tol = Channel1dCase{}.tol;
    const auto s = solveChannel1d(cc);
    if (!s) { check(false, "oracle " + nameOf(cc) + ": " + s.error()); return; }
    const auto r = linetest::makeLine(cc);
    const auto q = linetest::oracleColumn(r, *s);
    std::vector<double> R(q.size());
    r.line.residual(q, R);
    const int F = r.line.fields();
    double m[7] = {}, wScale = 0;
    for (int j = 1; j <= r.g.ny - 2; ++j) {
        for (int v = 0; v < F; ++v)
            m[v] = std::max(m[v], std::abs(R[std::size_t(j) * F + v]));
        if (F == 7) {
            const double* x = q.data() + std::size_t(j) * F;
            wScale = std::max(wScale, sst::beta1 * x[6] * x[6] / x[0]);
        }
    }
    const double wRel = wScale > 0 ? m[6] / wScale : 0.0;
    // The energy row's FP64 rounding floor: E carries T ~ Twall (~1e4 at
    // Re_tau 1e4, M 0.2), differenced over a first cell of 1e-5, so 1 ulp of
    // the state moves R_E by ~1e-6 there (the oracle avoids it by carrying
    // T - Tw). Measured as the largest change under 1-ulp perturbations.
    std::mt19937 rng(1);
    double floorE = 0;
    std::vector<double> R2(q.size());
    for (int t = 0; t < 10; ++t) {
        auto qq = q;
        for (double& x : qq) x = std::nextafter(x, (rng() & 1) ? HUGE_VAL : -HUGE_VAL);
        r.line.wall(qq);
        r.line.residual(qq, R2);
        for (int j = 1; j <= r.g.ny - 2; ++j)
            floorE = std::max(floorE, std::abs(R2[std::size_t(j) * F + 4] -
                                               R[std::size_t(j) * F + 4]));
    }
    std::printf("  (a) %s: max|R| rho %.1e mx %.1e my %.1e mz %.1e E %.1e (1-ulp "
                "floor %.1e) t %.1e %.1e (omega / beta1 rho omega^2: %.1e)\n",
                nameOf(cc).c_str(), m[0], m[1], m[2], m[3], m[4], floorE, m[5],
                m[6], wRel);
    const bool mom = m[0] <= 1e-10 && m[1] <= 1e-10 && m[2] <= 1e-10 && m[3] == 0.0;
    const bool sc = m[5] <= 1e-10 && wRel <= 1e-13;
    check(mom && sc, "(a) " + nameOf(cc) + ": continuity, momentum, scalar rows <= 1e-10");
    check(m[4] <= std::max(1e-7, floorE),
          "(a) " + nameOf(cc) + ": energy row <= max(1e-7, its 1-ulp rounding floor)");
}

// (b) Perturb one fluid row: R changes in rows j-K..j+K only, EXACTLY.
void band(RansModel model)
{
    const auto cc = caseOf(model, 180, 32, 2);
    const auto r = linetest::makeLine(cc);
    const auto q0 = linetest::icColumn(r);
    const int F = r.line.fields(), B = r.line.unknowns(), K = r.line.band(),
              N = r.g.ny;
    std::vector<double> R0(q0.size()), R(q0.size());
    r.line.residual(q0, R0);
    bool outside = true, reachesK = false;
    for (int j : {1, 2, 3, N / 2, N - 4, N - 3, N - 2})
        for (int u = 0; u < B; ++u) {
            auto q = q0;
            double& x = q[std::size_t(j) * F + ChannelLine::field(u)];
            x += 1e-3 * std::max(std::abs(x), 1e-3);
            r.line.wall(q);
            r.line.residual(q, R);
            for (int i = 0; i < N; ++i) {
                bool same = true;
                for (int v = 0; v < F; ++v)
                    same = same && R[std::size_t(i) * F + v] == R0[std::size_t(i) * F + v];
                if (std::abs(i - j) > K) outside = outside && same;
                if (std::abs(i - j) == K && i >= 1 && i <= N - 2 && !same)
                    reachesK = true;
            }
        }
    const std::string m(ransName(model));
    check(outside, std::format("(b) {}: rows beyond j +- {} bitwise unchanged", m, K));
    check(reachesK, std::format("(b) {}: some row at distance {} does change (band is K)", m, K));
}

// (c) J delta against (R(q + e delta) - R(q - e delta)) / 2e, per equation.
void jacobianFd(const Channel1dCase& cc)
{
    const auto s = solveChannel1d(cc);
    if (!s) { check(false, "oracle " + nameOf(cc)); return; }
    const auto r = linetest::makeLine(cc);
    const auto q = linetest::oracleColumn(r, *s);
    const int F = r.line.fields(), B = r.line.unknowns(), N = r.g.ny,
              K = r.line.band();
    std::vector<double> dl(std::size_t(N - 2) * B);
    for (int j = 1; j <= N - 2; ++j)
        for (int u = 0; u < B; ++u) {
            const double x = q[std::size_t(j) * F + ChannelLine::field(u)];
            const double rho = q[std::size_t(j) * F];
            const double sc = u == 0 || u == 3 ? std::abs(x)
                                               : std::max(std::abs(x), 1e-3 * rho);
            dl[std::size_t(j - 1) * B + u] =
                sc * std::sin(3.1 * j / N + 0.7 * u) * (1.0 + 0.5 * std::cos(0.3 * j));
        }
    const BlockBand J = r.line.jacobian(q);
    const double eps = 1e-6;
    auto shifted = [&](double e) {
        auto qs = q;
        for (int j = 1; j <= N - 2; ++j)
            for (int u = 0; u < B; ++u)
                qs[std::size_t(j) * F + ChannelLine::field(u)] +=
                    e * dl[std::size_t(j - 1) * B + u];
        r.line.wall(qs);
        std::vector<double> R(qs.size());
        r.line.residual(qs, R);
        return R;
    };
    const auto Rp = shifted(eps), Rm = shifted(-eps);
    double worst = 0;
    for (int u = 0; u < B; ++u) {
        double dmax = 0, fmax = 0;
        for (int i = 1; i <= N - 2; ++i) {
            double jd = 0;
            for (int o = -K; o <= K; ++o) {
                const int jj = i + o;
                if (jj < 1 || jj > N - 2) continue;
                for (int c = 0; c < B; ++c)
                    jd += J.at(i - 1, o, u, c) * dl[std::size_t(jj - 1) * B + c];
            }
            const std::size_t k = std::size_t(i) * F + ChannelLine::field(u);
            const double fd = (Rp[k] - Rm[k]) / (2 * eps);
            dmax = std::max(dmax, std::abs(jd - fd));
            fmax = std::max(fmax, std::abs(fd));
        }
        worst = std::max(worst, fmax > 0 ? dmax / fmax : dmax);
    }
    std::printf("  (c) %s: max relative |J d - FD| per equation %.2e\n",
                nameOf(cc).c_str(), worst);
    check(worst <= 1e-6, "(c) " + nameOf(cc) + ": J delta vs central FD <= 1e-6");
}

// (d) The host Newton driver from the Reichardt IC.
void hostNewton(const Channel1dCase& cc)
{
    const auto s = solveChannel1d(cc);
    if (!s) { check(false, "oracle " + nameOf(cc)); return; }
    const auto r = linetest::makeLine(cc);
    const auto q0 = linetest::icColumn(r);
    const double M0 = r.line.mass(q0);
    const auto t0 = std::chrono::steady_clock::now();
    const auto d = linetest::drive(r, q0, 1e-4, 200);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    const double ub = linetest::integrals(r, d.q).ub;
    const double err = ub / s->ub - 1;
    std::printf("  (d) %s: %d iterations, certified %d, ub %.9f oracle %.9f "
                "(%+.2e), mass %.15f, %.0f ms\n",
                nameOf(cc).c_str(), d.iters, int(d.certified), ub, s->ub, err,
                r.line.mass(d.q), ms);
    check(std::abs(M0 - cc.totalMass) <= 1e-12 * cc.totalMass,
          "(d) " + nameOf(cc) + ": the IC carries the oracle's mass");
    check(d.certified && d.iters <= 35,
          "(d) " + nameOf(cc) + ": certified in <= 35 iterations");
    check(std::abs(err) <= 1e-7, "(d) " + nameOf(cc) + ": ub within 1e-7 of the oracle");
    check(std::abs(r.line.mass(d.q) / M0 - 1) <= 1e-12,
          "(d) " + nameOf(cc) + ": mass pinned to 1e-12");
    check(d.rejects == 0, "(d) " + nameOf(cc) + ": no rejections");
}

// (e) R-a: a damped stall is never certified.
void stall()
{
    const auto cc = caseOf(RansModel::Sst2003m, 1e4, 256, 4);
    const auto r = linetest::makeLine(cc);
    const auto s = solveChannel1d(cc);
    const auto q0 = linetest::icColumn(r);
    auto err = [&](const linetest::Drive& d) {
        return s ? linetest::integrals(r, d.q).ub / s->ub - 1 : NAN;
    };
    // The masterplan's start, pure Newton: with the central-FD Jacobian it
    // converges rather than stalls. A certificate here must be a real one.
    const auto newton = linetest::drive(r, q0, 1e-4, 100, kImplicitCflMax);
    std::printf("  (e) SST-2003m Re_tau 1e4 ny 256 from CFL 1e8: %d iterations, "
                "certified %d, ub %+.3e vs oracle\n",
                newton.iters, int(newton.certified), err(newton));
    check(!newton.certified || std::abs(err(newton)) <= 1e-7,
          "(e) pure Newton from CFL 1e8: certified only on the fixed point");
    // Forced stall at the cap: every step damped (alpha <= 0.5), so never
    // armed -- while the damped march still contracts, which is exactly
    // what an unarmed stop mistakes for convergence.
    const auto damped = linetest::drive(r, q0, 1e-4, 100, kImplicitCflMax, 0.5);
    const auto unarmed =
        linetest::drive(r, q0, 1e-4, 100, kImplicitCflMax, 0.5, false);
    std::printf("  (e) every step damped to 0.5: armed stop certified %d after "
                "%d (ub %+.3e); unarmed stop certified %d at %d (ub %+.3e)\n",
                int(damped.certified), damped.iters, err(damped),
                int(unarmed.certified), unarmed.iters, err(unarmed));
    check(!damped.certified, "(e) R-a: a damped march is never certified");
    check(unarmed.certified,
          "(e) R-a: the same march under an unarmed stop does certify (what "
          "arming prevents)");
}

// (f) Projection: specific quantities kept, mass pinned.
void projection()
{
    const auto cc = caseOf(RansModel::Sst2003m, 180, 64, 2);
    const auto s = solveChannel1d(cc);
    if (!s) { check(false, "oracle (f)"); return; }
    const auto r = linetest::makeLine(cc);
    const auto q = linetest::oracleColumn(r, *s);
    auto p = q;
    r.line.project(p, 2.0 * 1.0037);
    const int F = r.line.fields();
    auto specific = [&](const std::vector<double>& x, int j) {
        const double* a = x.data() + std::size_t(j) * F;
        const double rho = a[0], u = a[1] / rho, v = a[2] / rho;
        const double T = (kGammaAir - 1) * (a[4] - 0.5 * rho * (u * u + v * v)) /
                         (rho * kRgasBox);
        return std::array<double, 5>{u, v, T, a[5] / rho, a[6] / rho};
    };
    // Fluid rows: the ghost omega is the image of omega_w = coef / rho_w,
    // which a density scaling legitimately moves (TMR's wall value).
    double worst = 0;
    for (int j = 1; j <= r.g.ny - 2; ++j) {
        const auto a = specific(q, j), b = specific(p, j);
        for (int k = 0; k < 5; ++k)
            if (a[k] != 0.0) worst = std::max(worst, std::abs(b[k] / a[k] - 1));
    }
    const double dm = std::abs(r.line.mass(p) / (2.0 * 1.0037) - 1);
    std::printf("  (f) projection x1.0037: specific values max rel change %.2e "
                "(%.1f ulp), mass error %.2e\n",
                worst, worst / 0x1p-52, dm);
    check(worst <= 8 * 0x1p-52,
          "(f) u, v, T, k, omega unchanged to rounding (<= 8 ulp)");
    check(dm <= 1e-14, "(f) mass pinned to 1e-14");
}

// Host cost of one Jacobian + block solve (pseudoUpdate) at the IC.
void timing()
{
    for (RansModel model : {RansModel::SaNeg, RansModel::Sst2003m})
        for (int ny : {128, 256, 512}) {
            const auto cc = caseOf(model, 1e4, ny, 4);
            const auto r = linetest::makeLine(cc);
            const auto q = linetest::icColumn(r);
            std::vector<double> R(q.size());
            r.line.residual(q, R);
            const int reps = 5;
            const auto t0 = std::chrono::steady_clock::now();
            bool ok = true;
            for (int i = 0; i < reps; ++i)
                ok = ok && pseudoUpdate(r.line, q, R, 1e4).has_value();
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count() / reps;
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < 20; ++i) r.line.residual(q, R);
            const double us = std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - t1).count() / 20;
            std::printf("  timing %s ny %d: Jacobian + solve %.2f ms (one line "
                        "residual %.0f us)\n",
                        std::string(ransName(model)).c_str(), ny, ms, us);
            check(ok, std::format("timing {} ny {}: solves", ransName(model), ny));
        }
}

} // namespace

int main()
{
    for (const auto& cc : {caseOf(RansModel::SaNeg, 180, 64, 2),
                           caseOf(RansModel::Sst2003m, 180, 64, 2),
                           caseOf(RansModel::SstVm, 180, 64, 2),
                           caseOf(RansModel::SaNeg, 1e4, 128, 4)})
        oracleResidual(cc);
    band(RansModel::SaNeg);
    band(RansModel::Sst2003m);
    band(RansModel::SstVm);
    for (const auto& cc : {caseOf(RansModel::SaNeg, 180, 64, 2),
                           caseOf(RansModel::Sst2003m, 180, 64, 2),
                           caseOf(RansModel::SstVm, 180, 64, 2)})
        jacobianFd(cc);
    for (const auto& cc : {caseOf(RansModel::SaNeg, 180, 64, 2),
                           caseOf(RansModel::Sst2003m, 180, 64, 2),
                           caseOf(RansModel::SstVm, 180, 64, 2),
                           caseOf(RansModel::SaNeg, 1e4, 128, 4),
                           caseOf(RansModel::Sst2003m, 1e4, 128, 4)})
        hostNewton(cc);
    stall();
    projection();
    timing();
    return dnstest::finish();
}
