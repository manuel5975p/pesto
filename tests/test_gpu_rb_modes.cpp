// Exact linear modes of the Boussinesq Rayleigh-Benard runner (gpu::runRb),
// driven at library level with a hand-built initial condition.
//
// Free-fall scaling: H = dT = g alpha = 1, nu = sqrt(Pr/Ra), kappa =
// 1/sqrt(Ra Pr). Two horizontally uniform perturbations of the conduction
// state are EXACT solutions of the full nonlinear equations:
//   1. thermal mode  T = T_cond(y) + A sin(pi y/H), u = 0. The buoyancy it
//      drives is horizontally uniform, so the projection must cancel it
//      against a hydrostatic pressure: u stays 0 to round-off and the
//      amplitude decays as exp(-kappa pi^2 t);
//   2. shear mode    u = A sin(pi y/H), T = T_cond(y). No advection (d/dx = 0)
//      and no buoyancy change: the amplitude decays as exp(-nu pi^2 t), and
//      v and T must not move.
// Pr = 0.72, so nu != kappa and each mode proves its own diffusivity.
// sin(pi y) at the cell centres is an exact eigenvector of the 3-point
// Laplacian with the antimirror (T: 2 Twall - T) wall image, so the discrete
// rate is the continuous one times (2/(pi h))^2 sin^2(pi h/2) = 1 - O(h^2):
// within 1% at 16 rows and second order between 16 and 32.
#include "dns/field.hpp"
#include "dns/grid.hpp"
#include "dns/output.hpp"
#include "dns/gpu/runner_rb.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using std::numbers::pi;

namespace {

constexpr double kRa = 1000.0, kPr = 0.72, kA = 0.2, kTEnd = 2.0;
const double kNu = std::sqrt(kPr / kRa), kKappa = 1.0 / std::sqrt(kRa * kPr);
constexpr double kTbot = 0.5, kTtop = -0.5;

double tCond(double y) { return kTbot + (kTtop - kTbot) * y; }

struct Measured
{
    bool ok = false;
    double rate = 0;      // measured exponential decay rate of the mode
    double uMax = 0;      // thermal mode: max stats urms over the run
    double tDev = 0;      // shear mode: max |T_bar - T_cond| at tEnd
    double vMax = 0;      // shear mode: max |v_bar| at tEnd
};

// One run of `nInt` interior rows; thermal == true seeds mode 1, else mode 2.
Measured runMode(int nInt, bool thermal)
{
    Measured m;
    const double h = 1.0 / nInt;
    const int nx = 8;
    const auto g = Grid::make(nx, nInt + 2, 1, nx * h, (nInt + 2) * h, 1.0);
    auto U = VectorField::zeros(g);
    Scalar T(g.size(), 0.0);
    double sumS2 = 0.0;
    for (int j = 1; j <= nInt; ++j) {
        const double y = (j - 0.5) * h, s = std::sin(pi * y);
        sumS2 += s * s;
        for (int i = 0; i < nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            T[c] = tCond(y) + (thermal ? kA * s : 0.0);
            if (!thermal) U.u[c] = kA * s;
        }
    }

    const auto dir = std::filesystem::temp_directory_path() / "dns_test_rb_modes";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string prefix =
        (dir / ((thermal ? "T" : "u") + std::to_string(nInt))).string();
    RunOutput::Config oc;
    oc.prefix = prefix;
    oc.kinds = {Emit::Stats};
    auto out = RunOutput::make(oc, {}, g, {});
    if (!out) { check(false, "sink: " + out.error()); return m; }

    gpu::RbRun r;
    r.rayleigh = kRa;
    r.prandtl = kPr;
    r.nu = kNu;
    r.kappa = kKappa;
    r.Tbot = kTbot;
    r.Ttop = kTtop;
    r.tEnd = kTEnd;
    r.statsEvery = 10;
    r.profileDt = kTEnd; // one profile, at tEnd
    const auto st = gpu::runRb(g, U, T, r, *out);
    if (!st) { check(false, "runRb: " + st.error().message); return m; }
    out = std::unexpected(std::string{}); // flush and close the CSVs

    const auto stats = valtest::loadCsv(prefix + "_stats.csv");
    const auto prof = valtest::loadCsv(prefix + "_profiles.csv");
    if (!stats || !prof) { check(false, "read back CSVs"); return m; }
    const auto& ur = stats->col("urms");
    const auto& ts = stats->col("t");
    const auto& Ty = prof->col("y");
    const auto& Tb = prof->col("T");
    const auto& Vb = prof->col("v");
    if (ur.empty() || int(Tb.size()) != nInt || std::abs(ts.back() - kTEnd) > 1e-5) {
        check(false, "CSV shape");
        return m;
    }
    if (thermal) {
        double a = 0.0;
        for (int j = 0; j < nInt; ++j)
            a += (Tb[j] - tCond(Ty[j])) * std::sin(pi * Ty[j]);
        a /= sumS2;
        m.rate = -std::log(a / kA) / kTEnd;
        for (double u : ur) m.uMax = std::max(m.uMax, u);
    } else {
        const double u0 = kA * std::sqrt(sumS2 / nInt);
        m.rate = -std::log(ur.back() / u0) / kTEnd;
        for (int j = 0; j < nInt; ++j) {
            m.tDev = std::max(m.tDev, std::abs(Tb[j] - tCond(Ty[j])));
            m.vMax = std::max(m.vMax, std::abs(Vb[j]));
        }
    }
    m.ok = true;
    std::filesystem::remove(prefix + "_stats.csv", ec);
    std::filesystem::remove(prefix + "_profiles.csv", ec);
    std::filesystem::remove(prefix + ".json", ec);
    return m;
}

void mode(bool thermal)
{
    const char* tag = thermal ? "thermal mode" : "shear mode";
    const double exact = (thermal ? kKappa : kNu) * pi * pi;
    const Measured a = runMode(16, thermal), b = runMode(32, thermal);
    if (!a.ok || !b.ok) return;
    const double e16 = std::abs(a.rate / exact - 1.0);
    const double e32 = std::abs(b.rate / exact - 1.0);
    const double order = std::log2(e16 / e32);
    std::printf("%s: rate exact %.6f | ny16 %.6f (err %.3e) ny32 %.6f (err "
                "%.3e) order %.3f\n",
                tag, exact, a.rate, e16, b.rate, e32, order);
    check(e16 < 0.01, std::string(tag) + ": decay rate within 1% at 16 rows");
    check(e32 < 0.01, std::string(tag) + ": decay rate within 1% at 32 rows");
    check(order > 1.8 && order < 2.2,
          std::string(tag) + ": rate error is second order (16 -> 32)");
    if (thermal) {
        std::printf("  max urms over the run: %.3e / %.3e\n", a.uMax, b.uMax);
        check(a.uMax < 1e-6 && b.uMax < 1e-6,
              "thermal mode: the projection holds u at 0 to round-off");
    } else {
        std::printf("  max |T_bar - T_cond| %.3e / %.3e, max |v_bar| %.3e / "
                    "%.3e\n", a.tDev, b.tDev, a.vMax, b.vMax);
        check(a.tDev < 1e-5 && b.tDev < 1e-5,
              "shear mode: the conduction profile does not move");
        check(a.vMax < 1e-6 && b.vMax < 1e-6, "shear mode: v stays 0");
    }
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    mode(true);
    mode(false);
    return dnstest::finish();
}
