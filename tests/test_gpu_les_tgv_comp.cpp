// gpu_les_tgv_comp (W10): the compressible SGS models on a known answer. A
// periodic Taylor-Green vortex at M 0.1, Re 1600, 32^3 (deliberately
// under-resolved), marched to t = 12 as DNS, Smagorinsky Cs 0.17 and sigma
// Csigma 1.35 (DeviceComp at library level, cfl 0.4), and read against the
// incompressible 512^3 DNS reference data/validation/tgv3d_512.csv. Pins:
//   1. sigma mu_t == 0.0 EXACTLY at t = 0: the TGV IC has w == 0, so the third
//      row of grad u is zero, det(g) == 0 in FP32 and sigma3 == 0
//      (dns/sgs.hpp) -- the model's defining zero, on a 3D field.
//   2. Smagorinsky eps_sgs / eps_visc at t = 0, device mu_t against the host
//      FP64 evaluation of the closed-form IC (central differences at the cell
//      centres, |S| = sqrt(2 S:S), Delta = h) to 1e-3 relative. The ratio
//      itself is not banded: a fixed band would encode one |S| convention.
//      eps = sum m |S_dev|^2, |S_dev|^2 = 2 (S:S - (div u)^2 / 3), m = mu or
//      mu_t (the dissipation of the Stokes stress and of the SGS stress).
//   3. Every run reaches t = 12 finite.
//   4. Peak total -dK/dt (from K at dt_sample 0.1) relative to the 512^3 peak,
//      per model, within +- 3 percentage points of the first measurement
//      (DNS -27.8%, smag -9.1%, sigma -10.5%).
//   5. The LES dt bound does not bind: on every sampled LES state the LES
//      stableDt equals a DNS DeviceComp's stableDt of the SAME state exactly
//      -- on the uniform box and on a ratio-3 focal (clustered) box, where the
//      bound pairs the largest mu_t (coarse cell) with the smallest h.
// Measured and deliberately NOT pinned (false; see the note above the bands):
// sigma quiet for t < 3, and sigma's K closer to 512^3 than DNS's.
// No HIP device -> skip (exit 77).
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/remap.hpp"
#include "dns/gpu/comp.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;

constexpr double kGamma = 1.4, kMach = 0.1, kMu = 1.0 / 1600.0;
constexpr double kCfl = 0.4, kTEnd = 12.0, kDtSample = 0.1;
constexpr double kCs = 0.17, kCsig = 1.35;

// Closed-form compressible TGV (rho0 = V0 = 1) at physical point x.
void tgvAt(const double x[3], double& u, double& v, double& p)
{
    const double p0 = 1.0 / (kGamma * kMach * kMach);
    u = std::sin(x[0]) * std::cos(x[1]) * std::cos(x[2]);
    v = -std::cos(x[0]) * std::sin(x[1]) * std::cos(x[2]);
    p = p0 + (std::cos(2 * x[0]) + std::cos(2 * x[1]))
                 * (std::cos(2 * x[2]) + 2.0) / 16.0;
}

// Cell-centre position: the 1D maps of a focal box (identity when none).
std::array<double, 3> centre(const Grid& g, const AxisRemap3* fr, int i, int j,
                             int k)
{
    const double X[3] = {(i + 0.5) * g.hx, (j + 0.5) * g.hy, (k + 0.5) * g.hz};
    if (!fr) return {X[0], X[1], X[2]};
    return {fr->fx.map(X[0]), fr->fy.map(X[1]), fr->fz.map(X[2])};
}

CompState tgvState(const Grid& g, const AxisRemap3* fr)
{
    auto s = CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const auto x = centre(g, fr, i, j, k);
                double u, v, p;
                tgvAt(x.data(), u, v, p);
                const std::size_t c = g.idx(i, j, k);
                s.rho[c] = 1.0;
                s.mx[c] = u;
                s.my[c] = v;
                s.E[c] = p / (kGamma - 1) + 0.5 * (u * u + v * v);
            }
    return s;
}

double kinetic(const CompState& s)
{
    double k = 0;
    for (std::size_t c = 0; c < s.rho.size(); ++c)
        k += 0.5 * (s.mx[c] * s.mx[c] + s.my[c] * s.my[c] + s.mz[c] * s.mz[c])
             / s.rho[c];
    return k / double(s.rho.size());
}

// Central-difference velocity gradient at cell centres of the uniform
// periodic box, from vel(cell, comp); the SGS kernels' stencil.
template <class Vel>
void gradAt(const Grid& g, int i, int j, int k, Vel vel, double gd[3][3])
{
    const int ip = (i + 1) % g.nx, im = (i + g.nx - 1) % g.nx;
    const int jp = (j + 1) % g.ny, jm = (j + g.ny - 1) % g.ny;
    const int kp = (k + 1) % g.nz, km = (k + g.nz - 1) % g.nz;
    const std::size_t p[3] = {g.idx(ip, j, k), g.idx(i, jp, k), g.idx(i, j, kp)};
    const std::size_t m[3] = {g.idx(im, j, k), g.idx(i, jm, k), g.idx(i, j, km)};
    const double h[3] = {g.hx, g.hy, g.hz};
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
            gd[c][d] = (vel(p[d], c) - vel(m[d], c)) / (2 * h[d]);
}

// 2 S:S and the deviatoric |S_dev|^2 = 2 (S:S - (div u)^2 / 3).
void strain(const double gd[3][3], double& s2, double& s2dev)
{
    double ss = 0;
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d) {
            const double sij = 0.5 * (gd[c][d] + gd[d][c]);
            ss += sij * sij;
        }
    const double dv = gd[0][0] + gd[1][1] + gd[2][2];
    s2 = 2 * ss;
    s2dev = 2 * (ss - dv * dv / 3);
}

// eps_sgs / eps_visc of state s with per-cell mu_t (uniform box).
double epsRatio(const Grid& g, const CompState& s, const std::vector<gpu::Real>& mut)
{
    double num = 0, den = 0;
    auto vel = [&](std::size_t c, int comp) {
        const auto& m = comp == 0 ? s.mx : comp == 1 ? s.my : s.mz;
        return m[c] / s.rho[c];
    };
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                double gd[3][3], s2, s2d;
                gradAt(g, i, j, k, vel, gd);
                strain(gd, s2, s2d);
                num += double(mut[g.idx(i, j, k)]) * s2d;
                den += kMu * s2d;
            }
    return num / den;
}

struct Run
{
    std::string name;
    std::vector<double> t, K;
    std::vector<double> ratioT, ratio; // eps_sgs / eps_visc samples
    bool finite = true;
    int steps = 0;
    int dtSamples = 0, dtBound = 0;   // LES stableDt < DNS stableDt count
    double dtMinRatio = 1.0;          // min dtLes / dtDns over samples
    double dtMargin = HUGE_VAL;       // min LES diffusive bound / dt taken
    double seconds = 0;
};

// March the TGV to kTEnd; samples land exactly on multiples of kDtSample.
// ratioUntil: sample eps_sgs / eps_visc (uniform box only) up to that t.
Run march(const std::string& name, const Grid& g, const CurvMetrics* met,
          const AxisRemap3* fr, double cs, SgsModel model, double ratioUntil)
{
    Run r;
    r.name = name;
    const auto t0 = std::chrono::steady_clock::now();
    auto mk = [&](double c) {
        return met ? gpu::DeviceComp::make(*met, kGamma, kMu, c, model)
                   : gpu::DeviceComp::make(g, kGamma, kMu, c, model);
    };
    auto dev = mk(cs);
    auto probe = mk(0.0); // DNS twin: the same state's dt without the SGS bound
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !probe || !ds || !red) {
        r.finite = false;
        return r;
    }
    CompState h = tgvState(g, fr);
    (void)ds->upload(h);
    std::vector<gpu::Real> mut(g.size());
    double t = 0;
    int next = 1;
    auto sample = [&] {
        (void)ds->download(h);
        const double K = kinetic(h);
        if (!std::isfinite(K)) r.finite = false;
        r.t.push_back(t);
        r.K.push_back(K);
        if (!dev->les()) return;
        const double dl = dev->stableDt(*ds, *red, kCfl);
        const double dd = probe->stableDt(*ds, *red, kCfl);
        ++r.dtSamples;
        if (dl < dd) ++r.dtBound;
        r.dtMinRatio = std::min(r.dtMinRatio, dl / dd);
        dev->updateSgsNut(*ds);
        (void)dev->mut().download(mut.data());
        // Margin of the LES diffusive bound over the dt taken, rebuilt from
        // stableDt's formula with the FP32 mu_t max (exactly what stableDt
        // reads since W15): 0.25 hmin^2 rho_min / ((mu + mu_t,max) g/Pr).
        const double mutMax = double(*std::ranges::max_element(mut));
        const double rhoMin = *std::ranges::min_element(h.rho);
        const double hmin = met ? met->hmin : std::min({g.hx, g.hy, g.hz});
        const double dtv = 0.25 * hmin * hmin * rhoMin
                           / ((kMu + mutMax) * (kGamma / 0.72));
        r.dtMargin = std::min(r.dtMargin, dtv / dl);
        if (t <= ratioUntil + 1e-9) {
            r.ratioT.push_back(t);
            r.ratio.push_back(epsRatio(g, h, mut));
        }
    };
    sample();
    while (t < kTEnd - 1e-9 && r.finite) {
        double dt = dev->stableDt(*ds, *red, kCfl);
        const double tNext = next * kDtSample;
        const bool hit = t + dt >= tNext - 1e-12;
        if (hit) dt = tNext - t;
        dev->step(*ds, gpu::Real(dt));
        ++r.steps;
        t = hit ? tNext : t + dt;
        if (hit) {
            ++next;
            sample();
        }
    }
    (void)hipDeviceSynchronize();
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()
                                              - t0).count();
    return r;
}

// -dK/dt at the interior samples (central differences).
void dissipation(const Run& r, std::vector<double>& t, std::vector<double>& d)
{
    for (std::size_t i = 1; i + 1 < r.t.size(); ++i) {
        t.push_back(r.t[i]);
        d.push_back(-(r.K[i + 1] - r.K[i - 1]) / (r.t[i + 1] - r.t[i - 1]));
    }
}

// max |K - K_ref| over the run's samples, K_ref interpolated linearly.
double maxKErr(const Run& r, const std::vector<double>& tr,
               const std::vector<double>& kr)
{
    double worst = 0;
    std::size_t j = 1;
    for (std::size_t i = 0; i < r.t.size(); ++i) {
        while (j + 1 < tr.size() && tr[j] < r.t[i]) ++j;
        const double w = (r.t[i] - tr[j - 1]) / (tr[j] - tr[j - 1]);
        const double kref = kr[j - 1] + w * (kr[j] - kr[j - 1]);
        worst = std::max(worst, std::abs(r.K[i] - kref));
    }
    return worst;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const auto ref = valtest::loadCsv(std::string(DNS_VALIDATION_DIR)
                                      + "/tgv3d_512.csv");
    if (!ref) {
        std::printf("cannot read tgv3d_512.csv\n");
        return EXIT_FAILURE;
    }
    const auto& tr = ref->col("t");
    const auto& kr = ref->col("K");
    const auto peakRef = valtest::curvePeak(tr, ref->col("dKdt"));
    std::printf("512^3 reference: peak -dK/dt %.6f at t %.3f\n", peakRef.value,
                peakRef.t);

    const int n = 32;
    const Grid g = Grid::make(n, n, n, 2 * pi, 2 * pi, 2 * pi);

    // --- (1) sigma mu_t == 0.0 at t = 0, (2) smag ratio vs host FP64 ---------
    {
        const CompState ic = tgvState(g, nullptr);
        auto ds = gpu::DeviceCompState::make(g.size());
        auto sig = gpu::DeviceComp::make(g, kGamma, kMu, kCsig, SgsModel::Sigma);
        auto smag = gpu::DeviceComp::make(g, kGamma, kMu, kCs);
        if (!ds || !sig || !smag) return EXIT_FAILURE;
        (void)ds->upload(ic);
        std::vector<gpu::Real> mut(g.size());
        sig->updateSgsNut(*ds);
        (void)sig->mut().download(mut.data());
        const bool allZero = std::ranges::all_of(
            mut, [](gpu::Real v) { return v == gpu::Real(0); });
        check(allZero, "sigma mu_t == 0.0 exactly on the w == 0 TGV IC");

        smag->updateSgsNut(*ds);
        (void)smag->mut().download(mut.data());
        // Host FP64: the closed form sampled at x +- h (the same central
        // stencil), mu_t = rho (Cs h)^2 sqrt(2 S:S) with rho = 1.
        const double h = g.hx, d2 = h * h;
        double num = 0, numDev = 0, den = 0;
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const auto x = centre(g, nullptr, i, j, k);
                    double gd[3][3] = {};
                    for (int d = 0; d < 3; ++d) {
                        double xp[3] = {x[0], x[1], x[2]}, xm[3] = {x[0], x[1], x[2]};
                        xp[d] += h;
                        xm[d] -= h;
                        double up, vp, pp, um, vm, pm;
                        tgvAt(xp, up, vp, pp);
                        tgvAt(xm, um, vm, pm);
                        gd[0][d] = (up - um) / (2 * h);
                        gd[1][d] = (vp - vm) / (2 * h);
                    }
                    double s2, s2d;
                    strain(gd, s2, s2d);
                    num += kCs * kCs * d2 * std::sqrt(s2) * s2d;
                    numDev += double(mut[g.idx(i, j, k)]) * s2d;
                    den += kMu * s2d;
                }
        const double host = num / den, devR = numDev / den;
        std::printf("smag eps_sgs/eps_visc at t=0: device %.7f, host FP64 %.7f "
                    "(rel %.2e)\n", devR, host, std::abs(devR / host - 1));
        check(std::abs(devR / host - 1) < 1e-3,
              "smag eps_sgs/eps_visc(0) == host FP64 closed form to 1e-3");
    }

    // --- marches -------------------------------------------------------------
    std::vector<Run> runs;
    runs.push_back(march("DNS", g, nullptr, nullptr, 0.0, SgsModel::Smagorinsky, -1));
    runs.push_back(march("smag 0.17", g, nullptr, nullptr, kCs,
                         SgsModel::Smagorinsky, kTEnd));
    runs.push_back(march("sigma 1.35", g, nullptr, nullptr, kCsig,
                         SgsModel::Sigma, kTEnd));
    // Ratio-3 focal box (all three axes clustered on the centre), smag: the
    // clustered-mesh case of the dt question only.
    const double lo[3] = {pi - 0.6, pi - 0.6, pi - 0.6},
                 hi[3] = {pi + 0.6, pi + 0.6, pi + 0.6}, w[3] = {0.8, 0.8, 0.8};
    const AxisRemap3 fr = focalRemap(g, true, lo, hi, w, 3.0);
    auto met = CurvMetrics::build(CurvMesh::fromRemap(g, fr));
    check(met.has_value() && fr.fx.active && fr.fy.active && fr.fz.active,
          "focal ratio-3 box builds");
    if (met)
        runs.push_back(march("smag 0.17 focal", g, &*met, &fr, kCs,
                             SgsModel::Smagorinsky, -1));

    std::vector<double> peak(runs.size()), kerr(runs.size());
    for (std::size_t q = 0; q < runs.size(); ++q) {
        const Run& r = runs[q];
        std::vector<double> td, dd;
        dissipation(r, td, dd);
        const auto pk = valtest::curvePeak(td, dd);
        peak[q] = pk.value / peakRef.value - 1;
        kerr[q] = maxKErr(r, tr, kr);
        std::printf("%-16s steps %5d  %.2f s  finite %d  t_end %.2f  K %.6f  "
                    "peak -dK/dt %.6f at t %.2f (%+.1f%% vs 512^3)  "
                    "max|K-K512| %.5f\n",
                    r.name.c_str(), r.steps, r.seconds, int(r.finite),
                    r.t.back(), r.K.back(), pk.value, pk.t, 100 * peak[q],
                    kerr[q]);
        if (r.dtSamples)
            std::printf("%-16s dt bound: LES < DNS on %d/%d samples, min "
                        "dtLES/dtDNS %.6f, LES diffusive bound >= %.1f x dt\n",
                        r.name.c_str(), r.dtBound, r.dtSamples, r.dtMinRatio,
                        r.dtMargin);
        double rmax3 = 0, rmax = 0, tmax = 0;
        for (std::size_t i = 0; i < r.ratio.size(); ++i) {
            if (r.ratioT[i] <= 3.0 + 1e-9) rmax3 = std::max(rmax3, r.ratio[i]);
            if (r.ratio[i] > rmax) {
                rmax = r.ratio[i];
                tmax = r.ratioT[i];
            }
        }
        if (!r.ratio.empty())
            std::printf("%-16s eps_sgs/eps_visc: t=0 %.4f, max(t<=3) %.4f, max "
                        "%.4f at t %.1f\n",
                        r.name.c_str(), r.ratio.front(), rmax3, rmax, tmax);
        for (std::size_t i = 0; i < r.ratio.size(); i += 10)
            std::printf("  %s eps_sgs/eps_visc t %.0f: %.4f\n", r.name.c_str(),
                        r.ratioT[i], r.ratio[i]);
        check(r.finite && std::abs(r.t.back() - kTEnd) < 1e-9,
              r.name + ": reaches t = 12 finite");
        if (r.dtSamples)
            check(r.dtBound == 0, r.name + ": the LES dt bound never binds");
    }
    // NOT pinned, by measurement (2026-10-27, gfx1100): the planned "sigma
    // eps_sgs/eps_visc < 0.05 for t < 3" is false -- the TGV turns 3D at once
    // (w grows ~ t), and (Csigma h)^2 / nu = 112 here, so the ratio is 1.35 at
    // t 0.5 and 2.60 at t 3 (smag: 1.99, 2.91). Only the t = 0 zero (1) is
    // the model's defining property. Likewise "sigma's K closer to 512^3 than
    // DNS" is false: max|K - K512| DNS 0.0146, sigma 0.0229, smag 0.0250 --
    // both models over-dissipate early (peaks at t 6.4-6.9 vs 9.0).

    // --- (4) peak -dK/dt vs 512^3, first measurement +- 3 percentage points ---
    struct Band { const char* name; double first; };
    const Band bands[] = {{"DNS", -0.278}, {"smag 0.17", -0.091},
                          {"sigma 1.35", -0.105}};
    for (const Band& b : bands)
        for (std::size_t q = 0; q < runs.size(); ++q)
            if (runs[q].name == b.name)
                check(std::abs(peak[q] - b.first) < 0.03,
                      std::string(b.name) + ": peak -dK/dt vs 512^3 within "
                      "+-3 pp of the first measurement");
    return dnstest::finish();
}
