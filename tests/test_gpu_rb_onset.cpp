// Linear onset of Rayleigh-Benard convection, through the built binary.
//
// Rigid-rigid plates, unbounded layer: the critical Rayleigh number is
// Ra_c = 1707.76 at wavenumber k_c = 3.117, i.e. a roll pair of width
// 2 pi / k_c = 2.016 H. A periodic box of exactly that aspect holds the
// critical mode as its fundamental. A small kick then evolves linearly: past
// the transient the kinetic energy is one exponential, KE ~ exp(2 sigma t),
// and sigma changes sign at Ra_c. Measured at Ra 1550 (must decay) and 1900
// (must grow); the zero of the linear interpolation sigma(Ra) is the onset.
//
//   Boussinesq branch, 64 x 32 rows:   Ra_c to +-3%
//   compressible branch, M 0.02, on the superadiabatic drop (the definition
//   of --rayleigh): Ra_c to +-5%. The extra band is the model's own: a
//   10%-dtemp layer is stratified and non-Boussinesq at O(dtemp).
//
// Times are in free-fall units: the Boussinesq branch's t IS that, the
// compressible branch writes box time, so t_ff = tEnd / tt.
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <cmath>
#include <cstdio>
#include <string>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;

constexpr double kRaC = 1707.76, kRaLo = 1550.0, kRaHi = 1900.0;

struct Growth
{
    bool ok = false;
    double sigma = 0;
    int rows = 0;
};

// Least-squares slope of ln KE against t / t_ff over [w0, w1], halved.
Growth growth(const std::string& physics, double ra, int n, int ny, double tt,
              double w0, double w1, const std::string& extra)
{
    Growth g;
    const std::string prefix =
        "/tmp/dns_test_rb_onset_" + physics + std::to_string(int(ra));
    const std::string cmd = "\"" DNS_BIN "\" conv --physics " + physics
                          + " --nz 1 --n " + std::to_string(n) + " --ny "
                          + std::to_string(ny) + " --aspect 2.016 --rayleigh "
                          + std::to_string(ra) + " --tt " + std::to_string(tt)
                          + " " + extra + " --emit stats --quiet --out "
                          + prefix;
    const int rc = valtest::runBinary(cmd);
    check(rc == 0, physics + " Ra " + std::to_string(int(ra)) + ": exits 0");
    if (rc != 0) return g;
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    if (!tbl) { check(false, "stats CSV readable"); return g; }
    const auto& t = tbl->col("t");
    const auto& ke = tbl->col("KE");
    if (t.size() < 4) { check(false, "enough stats rows"); return g; }
    const double tff = t.back() / tt;
    double st = 0, sl = 0, stt = 0, stl = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double x = t[i] / tff;
        if (x < w0 || x > w1 || !(ke[i] > 0)) continue;
        const double y = std::log(ke[i]);
        st += x; sl += y; stt += x * x; stl += x * y;
        ++g.rows;
    }
    if (g.rows < 4) { check(false, "enough rows in the fit window"); return g; }
    const double m = g.rows;
    g.sigma = 0.5 * (m * stl - st * sl) / (m * stt - st * st);
    g.ok = std::isfinite(g.sigma);
    return g;
}

void onset(const std::string& physics, int n, int ny, double tt, double w0,
           double w1, const std::string& extra, double band)
{
    const Growth lo = growth(physics, kRaLo, n, ny, tt, w0, w1, extra);
    const Growth hi = growth(physics, kRaHi, n, ny, tt, w0, w1, extra);
    if (!lo.ok || !hi.ok) { check(false, physics + ": both fits"); return; }
    const double raC = kRaLo + (kRaHi - kRaLo) * lo.sigma / (lo.sigma - hi.sigma);
    std::printf("%s: sigma(Ra %.0f) = %+.5f, sigma(Ra %.0f) = %+.5f "
                "(%d/%d rows, t_ff in [%.0f, %.0f]) -> Ra_c %.1f "
                "(theory %.2f, %+.2f%%)\n",
                physics.c_str(), kRaLo, lo.sigma, kRaHi, hi.sigma, lo.rows,
                hi.rows, w0, w1, raC, kRaC, 100 * (raC / kRaC - 1));
    check(lo.sigma < 0, physics + ": Ra 1550 decays");
    check(hi.sigma > 0, physics + ": Ra 1900 grows");
    char msg[96];
    std::snprintf(msg, sizeof msg, ": interpolated Ra_c within %.0f%% of 1707.76",
                  100 * band);
    check(std::abs(raC / kRaC - 1) < band, physics + msg);
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    onset("incompressible", 64, 32, 25.0, 10.0, 25.0, "--kick 1e-3", 0.03);
    // The larger kick lifts the convective mode clear of the acoustic
    // residue of the start, which at kick 1e-3 dominated KE and hid the
    // mode (measured: growth at BOTH Ra values).
    onset("compressible", 32, 16, 20.0, 8.0, 20.0, "--mach 0.02 --kick 1e-2",
          0.05);
    return dnstest::finish();
}
