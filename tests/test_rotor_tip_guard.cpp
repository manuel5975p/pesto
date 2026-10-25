// The rotor tip-Mach guard: --rotor-mach is quoted on the NOMINAL tip radius
// D/2 (all validate() can know), so a body reaching further has a faster real
// tip and the run must measure it and STOP. CLAUDE.md's case:
// `--body lens --aspect 4 --rotor-mach 0.4` is a real tip Mach of 1.61.
//
// Exact expectation: the lens has height D and half-width w = aspect D/2 = 2 D
// (casegeom), so its tip sits at 2 D = 4 x the nominal radius and the real tip
// Mach is 0.4 x 4 = 1.6. The run measures the radius off the classified
// interface FACES, which lie within one cell of the surface, so the measured
// Mach is 1.6 to within 0.4 h / (D/2) = 0.8 h/D (0.05 at D = 16 h).
//
// Library level (host only): resolved Case -> casegeom sdf -> the azimuth table
// the runner builds -> internal::checkTipMach, the runner's own guard. A
// cylinder is the control: its faces sit at D/2, the guard passes. Then
// through the built binary: exit 2 with the corrected number in the message
// (needs the GPU, since the guard runs after acquisition; skipped without).
#include "dns/casegeom.hpp"
#include "dns/config.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/rotortable.hpp"
#include "dns/remap.hpp"
#include "dns/rotor.hpp"
#include "dns/transport.hpp"

#include "check.hpp"
#include "cli_test_util.hpp"
#include "gpu_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;

struct Tip
{
    bool ok = false;
    double tipR = 0, nominal = 0, h = 0, D = 0, machTip = 0;
    std::string guard; // checkTipMach's error message; empty = passed
};

// The runner's measurement and guard (runner_comp.hip, moving-geometry path),
// replayed on the host from the resolved case.
Tip measure(const std::vector<std::string>& args)
{
    Tip out;
    auto c = dnstest::runCli(args);
    if (!c)
    {
        std::printf("  -> %s\n", dns::format(c.error()).c_str());
        check(false, "case resolves");
        return out;
    }
    auto geo = dns::buildCaseGeometry(*c);
    if (!geo)
    {
        check(false, "case geometry: " + geo.error());
        return out;
    }
    const dns::Grid& g = c->grid;
    const dns::AxisRemap3 remap = dns::focalRemap(
        g, c->focalActive, c->focalLo, c->focalHi, c->focalW, c->focalRatio);
    dns::Rotor turn = c->rotor;
    turn.thicken = 0.0; // casegeom already applied it, as in the runner
    std::array<double, 6> swept{};
    if (geo->hasBbox) swept = dns::rotorSweepBounds(turn, geo->bbox);
    const dns::Sdf sdf = geo->sdf;
    auto tbl = dns::gpu::buildRotorTable(
        g, turn, [&sdf] { return dns::gpu::GhostSdf(sdf); },
        geo->hasBbox ? swept.data() : nullptr, remap);
    if (!tbl)
    {
        check(false, "rotor table: " + tbl.error());
        return out;
    }
    // Box units of the compressible flow (dns_main): rho = 1, p = uInf^2 /
    // (gamma M^2).
    const double gamma = dns::kGammaAir;
    const double pInf = c->uInf * c->uInf / (gamma * c->mach * c->mach);
    const double cInf = std::sqrt(gamma * pInf);
    auto e = dns::gpu::internal::checkTipMach(
        turn.omega, tbl->tipRadius, turn.radius, cInf, c->uInf / cInf,
        "measured tip radius", "body", "", "");
    out.ok = true;
    out.tipR = tbl->tipRadius;
    out.nominal = turn.radius;
    out.h = dns::hFine(*c);
    out.D = c->placement.D;
    out.machTip = std::abs(turn.omega) * tbl->tipRadius / cInf;
    if (!e) out.guard = e.error().message;
    return out;
}

std::string slurp(const std::string& path)
{
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

int main()
{
    const std::vector<std::string> base = {
        "flow", "--physics", "compressible", "--n", "128", "--nz", "1",
        "--rotor-mach", "0.4", "--rotor-axis", "z"};

    // --- the lens: tip at 4x the nominal radius ------------------------------
    auto lensArgs = base;
    lensArgs.insert(lensArgs.end(), {"--body", "lens", "--aspect", "4"});
    const Tip lens = measure(lensArgs);
    if (lens.ok)
    {
        const double tol = 0.8 * lens.h / lens.D;
        std::printf("lens aspect 4: tip %.5f nominal %.5f ratio %.4f (exact 4), "
                    "tip Mach %.4f (exact 1.6 +- %.3f), D/h %.1f\n",
                    lens.tipR, lens.nominal, lens.tipR / lens.nominal,
                    lens.machTip, tol, lens.D / lens.h);
        std::printf("  guard: %s\n", lens.guard.c_str());
        check(lens.tipR / lens.nominal > 1.0,
              "lens: measured tip radius exceeds the nominal D/2");
        check(std::abs(lens.machTip - 1.6) <= tol,
              "lens: real tip Mach = 0.4 x 4 = 1.6 to one cell");
        check(!lens.guard.empty(), "lens: checkTipMach refuses the run");
        char want[32];
        std::snprintf(want, sizeof want, "real tip Mach is %.3f",
                      lens.machTip);
        check(lens.guard.find(want) != std::string::npos,
              "lens: the refusal names the measured tip Mach");
        check(lens.guard.find("1.6") != std::string::npos,
              "lens: the named tip Mach is ~1.6");
    }

    // --- control: a cylinder's tip is its nominal radius ---------------------
    auto cylArgs = base;
    cylArgs.insert(cylArgs.end(), {"--body", "cylinder"});
    const Tip cyl = measure(cylArgs);
    if (cyl.ok)
    {
        std::printf("cylinder: tip %.5f nominal %.5f ratio %.4f, tip Mach "
                    "%.4f\n",
                    cyl.tipR, cyl.nominal, cyl.tipR / cyl.nominal, cyl.machTip);
        check(std::abs(cyl.tipR - cyl.nominal) <= cyl.h,
              "cylinder: measured tip = nominal D/2 to one cell");
        check(cyl.guard.empty(), "cylinder: the guard passes");
    }

    // --- through the binary: exit 2, the corrected number in the message ----
    if (!dns::gpu::deviceAvailable())
        dnstest::skip("no HIP device (library checks above ran)");
    const std::string log =
        (std::filesystem::temp_directory_path() / "dns_tip_guard.log").string();
    const std::string cmd =
        std::string("\"") + DNS_BIN + "\" flow --physics compressible --n 128 "
        "--nz 1 --rotor-mach 0.4 --rotor-axis z --body lens --aspect 4 "
        "--ft 0.01 --emit none --out dns_tip_guard > \"" + log + "\" 2>&1";
    const int rc = std::system(cmd.c_str());
    const int code = (rc != -1 && WIFEXITED(rc)) ? WEXITSTATUS(rc) : -1;
    const std::string text = slurp(log);
    std::printf("binary: exit %d\n%s", code, text.c_str());
    check(code == 2, "dns flow --body lens --aspect 4 --rotor-mach 0.4 exits 2");
    if (lens.ok)
    {
        char want[32];
        std::snprintf(want, sizeof want, "real tip Mach is %.3f", lens.machTip);
        check(text.find(want) != std::string::npos,
              "binary names the same measured tip Mach as the library");
    }
    std::error_code ec;
    std::filesystem::remove(log, ec);
    for (const char* suf : {".json", "_stats.csv"})
        std::filesystem::remove(std::string("dns_tip_guard") + suf, ec);
    return dnstest::finish();
}
