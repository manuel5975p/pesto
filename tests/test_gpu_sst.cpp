// SST on the GPU, small cases (docs/rans.md "SST: small-case verification"):
//   (a) freestream decay: a uniform periodic box (every gradient exactly 0)
//       is the ODE pair dk/dt = -beta* omega k, domega/dt = -beta omega^2
//       with the closed form omega = omega0/(1 + beta omega0 t), k = k0 (1 +
//       beta omega0 t)^(-beta*/beta); beta = beta2 (F1 = 0, far from a wall)
//       and beta1 (F1 = 1, a wall distance that pins arg1 > 10). Both
//       closures, the point-implicit stage at three dt: the error and its
//       ORDER in dt, the state stays exactly uniform, 0 guard hits.
//   (b) the smallest separated RANS case that certifies: the L1 sphere wedge
//       at Re 1e3 through the binary, pinned to its measured state and to
//       OpenFOAM kOmegaSST on the identical mesh.
// ~35 s.
#include "validation_util.hpp"
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/rans.hpp"
#include "dns/transport.hpp"
#include "dns/gpu/comp.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <memory>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

using namespace dns;
using dnstest::check;

namespace {

constexpr double kGam = 1.4, kRgas = 1.0, kMach = 0.2;
const double kPInf = 1.0 / (kGam * kMach * kMach);

struct Decay
{
    double errW, errK; // relative, at tEnd
    double spreadW;    // max - min over the box, relative
    unsigned long long hits;
};

// One decay run on a 4 x 4 x 1 periodic uniform box. Pre: dt > 0.
Decay decay(RansModel model, double d, double k0, double w0, double tEnd,
            int steps)
{
    const Grid g = Grid::make(4, 4, 1, 1.0, 1.0, 1.0);
    const CurvMesh mesh = CurvMesh::uniform(g);
    auto m = CurvMetrics::build(mesh);
    const double mu = 1e-4;
    const auto tr = Transport::air(mu, kPInf, 300.0, 0.72, kGam, kRgas, false);
    auto dev = gpu::DeviceComp::make(*m, tr);
    auto ds = gpu::DeviceCompState::make(g.size(), 2, false);
    if (!m || !dev || !ds) return {HUGE_VAL, HUGE_VAL, HUGE_VAL, ~0ull};
    if (!dev->enableRans({.model = model, .wallDist = std::vector<double>(g.size(), d)}))
        return {HUGE_VAL, HUGE_VAL, HUGE_VAL, ~0ull};
    auto h = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        h.rho[c] = 1.0;
        h.mx[c] = 1.0; // a uniform U = 1 rides along: convection must cancel
        h.E[c] = kPInf / (kGam - 1.0) + 0.5;
    }
    (void)ds->upload(h);
    (void)ds->uploadTurb(0, std::vector<double>(g.size(), k0));
    (void)ds->uploadTurb(1, std::vector<double>(g.size(), w0));
    const double dt = tEnd / steps;
    for (int n = 0; n < steps; ++n) dev->step(*ds, gpu::Real(dt));
    std::vector<double> k, w;
    (void)ds->downloadTurb(0, k);
    (void)ds->downloadTurb(1, w);
    // F1 decides beta: arg1 >= sqrt(k)/(beta* omega d) and 500 nu/(d^2 omega)
    // over the whole decay. The caller picks d so F1 is 0 or 1 throughout.
    const double beta = d < 1.0 ? sst::beta1 : sst::beta2;
    const double s = 1.0 + beta * w0 * tEnd;
    const double wEx = w0 / s, kEx = k0 * std::pow(s, -sst::betaStar / beta);
    const auto [wlo, whi] = std::ranges::minmax(w);
    return {std::abs(w[0] - wEx) / wEx, std::abs(k[0] - kEx) / kEx,
            (whi - wlo) / wEx, dev->floorHits()};
}

void freestreamDecay()
{
    // Two regimes: TMR's freestream (k 2.25e-7, omega 0.25, decaying over
    // t = 400: beta omega0 t ~ 8) and a stiff one (omega0 1e3, beta omega0 t
    // = 50, i.e. beta omega0 dt up to 0.5 at the coarsest dt).
    struct Case
    {
        double k0, w0, tEnd;
        int steps;
    } cases[] = {{2.25e-7, 0.25, 400.0, 400}, {1e-3, 1e3, 0.6, 100}};
    for (RansModel model : {RansModel::Sst2003m, RansModel::SstVm})
        for (double d : {1e3, 1e-4}) // F1 = 0 (beta2) | F1 = 1 (beta1)
            for (const Case& c : cases) {
                double eW[3], eK[3];
                bool ok = true;
                for (int r = 0; r < 3; ++r) {
                    const Decay e =
                        decay(model, d, c.k0, c.w0, c.tEnd, c.steps << r);
                    eW[r] = e.errW;
                    eK[r] = e.errK;
                    ok = ok && e.hits == 0 && e.spreadW == 0.0;
                }
                const double pW = std::log2(eW[1] / eW[2]),
                             pK = std::log2(eK[1] / eK[2]);
                std::printf("decay %s F1=%d w0 %.3g: err omega %.2e %.2e %.2e "
                            "(order %.2f) k %.2e %.2e %.2e (order %.2f)\n",
                            ransName(model).data(), d < 1 ? 1 : 0, c.w0, eW[0],
                            eW[1], eW[2], pW, eK[0], eK[1], eK[2], pK);
                check(ok, std::format("decay {} F1={} w0 {}: exactly uniform, 0 hits",
                                      ransName(model), d < 1 ? 1 : 0, c.w0));
                // MEASURED (gfx1100): FIRST order in dt, both scalars -- the
                // point-implicit quotient (D13) is first order in its stiff
                // source even inside SSP-RK3: omega 1.00-1.01, k 0.91-0.98.
                // The finest errors: freestream omega 1.03e-3 / k 7.9e-4,
                // stiff 8.0e-3 / 3.9e-2. Bands 1.25x those, order +-0.1.
                const bool stiff = c.w0 > 1.0;
                const double capW = stiff ? 1.0e-2 : 1.3e-3,
                             capK = stiff ? 4.9e-2 : 1.0e-3;
                check(eW[2] < capW && eK[2] < capK && pW > 0.9 && pW < 1.1 &&
                          pK > 0.8 && pK < 1.1,
                      std::format("decay {} F1={} w0 {}: converges to the closed "
                                  "form, first order (omega {:.2f}, k {:.2f})",
                                  ransName(model), d < 1 ? 1 : 0, c.w0, pW, pK));
            }
}

// (b) The small separated-flow gate: the revolved sphere wedge, L1 (64 x
// 48), Re 1e3, M 0.2, RANS SST-2003m, explicit LTS to TOL 1e-4 through the
// binary. MEASURED (gfx1100, 2026-10-26): exit 0 at 114600 iterations (28
// s), cd 0.49851, theta_s 100.30 deg, L_r 1.1656 D, mu_t/mu max 12.0, 0
// guard hits. The matched reference is OpenFOAM v2606 kOmegaSST on the
// IDENTICAL mesh with CENTRAL U convection (sph1e4.py --u-scheme linear):
// cd 0.48872, theta_s 100.77 deg, L_r 1.0674 D -- dns tier B (M 0.1 cd
// 0.49265 settled, exit 8 by the M 0.1 jitter floor) is +0.41%. With
// OpenFOAM's default linearUpwind the reference is 0.47129 / 100.02 deg /
// 1.4890 D: the mean-flow scheme, not the closure (docs/validation.md).
double sepAngle(const valtest::CsvTable& s)
{
    const auto& th = s.col("theta_deg");
    const auto& cf = s.col("cf");
    for (std::size_t i = 1; i < cf.size(); ++i)
        if (cf[i - 1] > 0 && cf[i] <= 0)
            return th[i - 1] + (th[i] - th[i - 1]) * cf[i - 1] / (cf[i - 1] - cf[i]);
    return std::nan("");
}

void wedgeGate()
{
    const std::string prefix = "gpu_sst_wedge";
    const int rc = valtest::runBinary(
        "\"" DNS_BIN "\" flow --physics compressible --body sphere --ogrid --nz 1 "
        "--n 64 --ny 48 --re 1e3 --mach 0.2 --ogrid-far 20 --ogrid-beta 4.5 "
        "--rans sst-2003m --steady 1e-4 --steady-steps 200000 --emit stats "
        "--emit surface --out " + prefix);
    check(rc == 0, "wedge L1 Re 1e3 SST-2003m: the steady march certifies (exit 0)");
    const auto st = valtest::loadCsv(prefix + "_stats.csv");
    const auto sf = valtest::loadCsv(prefix + "_surface.csv");
    if (!st || !sf || st->col("iter").empty()) {
        check(false, "wedge gate: stats and surface readable");
        return;
    }
    const double it = st->col("iter").back(), cd = st->col("cd").back();
    const double lr = st->col("lr").back(), ts = sepAngle(*sf);
    double hits = 0, mut = 0;
    for (double h : st->col("floorHits")) hits = std::max(hits, h);
    for (double m : st->col("mutMax")) mut = std::max(mut, m);
    std::printf("wedge L1 Re 1e3 SST-2003m: %.0f iterations, cd %.5f, theta_s "
                "%.2f, L_r %.4f, mutMax %.2f, hits %.0f\n",
                it, cd, ts, lr, mut, hits);
    check(it <= 1.5e5, "wedge gate: certified within 1.5e5 iterations (114600)");
    check(std::abs(cd / 0.49851 - 1) < 5e-4 && std::abs(lr / 1.1656 - 1) < 5e-3 &&
              std::abs(ts - 100.30) < 0.1,
          "wedge gate: cd, L_r, theta_s at their measured values");
    // Against OpenFOAM central: cd at M 0.2 carries ~+1.6% compressibility
    // (measured cd(M0.2)/cd0 = 1.0159), so the band is on cd/1.0159.
    check(std::abs(cd / 1.0159 / 0.48872 - 1) < 0.01 && std::abs(ts - 100.77) < 1.0,
          "wedge gate: cd0 within 1% and theta_s within 1 deg of OpenFOAM "
          "kOmegaSST (central U) on the identical mesh");
    check(hits == 0 && mut > 5 && mut < 20,
          "wedge gate: 0 guard hits, turbulent wake (mu_t/mu max in (5, 20))");
    std::error_code ec;
    for (const char* s : {"_stats.csv", "_surface.csv", ".json", "_mask.bin"})
        std::filesystem::remove(prefix + s, ec);
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    freestreamDecay();
    wedgeGate();
    return dnstest::finish();
}
