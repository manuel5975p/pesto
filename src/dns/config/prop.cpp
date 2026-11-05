// validate(): dns prop (+ its --dry-run text).
#include "dns/config.hpp"
#include "dns/config/internal.hpp"
#include "dns/numparse.hpp"



#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>

using std::numbers::pi;

namespace dns::cfg {
namespace {

// --- prop -------------------------------------------------------------------

// Python's round(x, 12): the correctly rounded 12-decimal value.
double round12(double x)
{
    double v = x;
    (void)parseDouble(std::format("{:.12f}", x), v);
    return v;
}

// --j: "a,b,c" or "start:stop:step" (stop inclusive) ->
// sorted, de-duplicated, every J >= 0. Python's parse_j, made strict: an
// empty list item is an error, not skipped.
std::expected<std::vector<double>, ConfigError> parseAdvance(std::string_view s)
{
    constexpr std::size_t kMaxPoints = 100000;
    auto fail = [&](std::string why) {
        return std::unexpected(ConfigError{2, "--j", std::move(why)});
    };
    std::vector<double> js;
    if (s.find(':') != std::string_view::npos)
    {
        double v[3];
        std::size_t start = 0;
        for (int k = 0; k < 3; ++k)
        {
            const std::size_t end = k < 2 ? s.find(':', start) : s.size();
            if (end == std::string_view::npos
                || !parseDouble(s.substr(start, end - start), v[k]))
                return fail("expected 'a,b,c' or 'start:stop:step', got '" + std::string(s) + "'");
            start = end + 1;
        }
        const double lo = v[0], hi = v[1], st = v[2];
        if (!(st > 0) || hi < lo)
            return fail("start:stop:step needs step > 0 and stop >= start");
        const double count = std::floor((hi - lo) / st + 1e-9);
        if (!(count < double(kMaxPoints)))
            return fail(std::format("more than {} advance ratios", kMaxPoints));
        for (int k = 0; k <= int(count); ++k) js.push_back(round12(lo + k * st));
    }
    else
    {
        std::size_t start = 0;
        while (start <= s.size())
        {
            const std::size_t end = std::min(s.find(',', start), s.size());
            double v = 0;
            if (!parseDouble(s.substr(start, end - start), v))
                return fail("expected 'a,b,c' or 'start:stop:step', got '" + std::string(s) + "'");
            js.push_back(v);
            start = end + 1;
        }
    }
    if (js.empty() || std::ranges::min(js) < 0.0)
        return fail("need J >= 0 (reverse flow through the disk is not modelled)");
    std::ranges::sort(js);
    js.erase(std::unique(js.begin(), js.end()), js.end());
    return js;
}

// The flags only one method reads, rejected (never ignored) with the other.
constexpr std::initializer_list<const char*> kBemtOnly = {"--stations-bemt"};

constexpr std::initializer_list<const char*> kLiftOnly = {
    "--panels",    "--wake-length", "--align-iters", "--align-tol",  "--seg-deg",
    "--core",      "--hub",         "--wake-pitch",  "--stall-visc", "--selftest"};

} // namespace
} // namespace dns::cfg

namespace dns::cfg {

std::expected<Case, ConfigError> validateProp(const RawConfig& r, Case c)
{
    prop::Case& p = c.prop;
    p.method = prop::Method(r.propMethod);
    p.quiet = r.quiet;
    const bool ll = p.method == prop::Method::Liftline;
    if (auto e = rejectGiven(r, ll ? kBemtOnly : kLiftOnly,
                             ll ? "is a --method bemt option"
                                : "is a --method liftline option");
        e)
        return std::unexpected(*e);

    if (r.selftest)
    {
        // Fixed analytic cases: every input flag would be silently unused.
        if (auto e = rejectGiven(
                r, {"--planform", "--polar", "--j", "--out", "--dump-span", "--blades",
                    "--hub-radius", "--diameter", "--rpm", "--rho", "--nu", "--sound",
                    "--thin-cd0", "--thin-alpha0", "--panels", "--wake-length",
                    "--align-iters", "--align-tol", "--seg-deg", "--core", "--hub",
                    "--wake-pitch", "--stall-visc"},
                "has no meaning with --selftest (it runs fixed analytic cases)");
            e)
            return std::unexpected(*e);
        p.selftest = true;
        return c;
    }

    if (r.planform.empty())
        return bad(2, "--planform", "is required: an r_R,c_R,beta_deg CSV, hub -> tip "
                                    "(tools/propeller.py exports it)");
    if (r.polar.empty())
        return bad(2, "--polar", "is required: a re,alpha_deg,cl,cd CSV over -180..180 "
                                 "deg (tools/polars.py exports it), or `thin`");
    p.planformPath = r.planform;
    const bool thin = r.polar == "thin";
    if (!thin) p.polarPath = r.polar;
    if (!thin)
        if (auto e = rejectGiven(r, {"--thin-cd0", "--thin-alpha0"},
                                 "shapes the built-in polar; it needs --polar thin");
            e)
            return std::unexpected(*e);
    if (!(r.thinCd0 > 0 && r.thinCd0 < 1))
        return bad(2, "--thin-cd0", "must be in (0, 1)");
    if (!(r.thinAlpha0 > -10.0 && r.thinAlpha0 < 10.0))
        return bad(2, "--thin-alpha0", "must be in (-10, 10) deg: the linear range "
                                       "alpha0 +- 10 deg has to contain alpha = 0");
    p.thinCd0 = r.thinCd0;
    p.thinAlpha0Deg = r.thinAlpha0;

    if (r.blades < 1) return bad(2, "--blades", "must be >= 1");
    p.blades = r.blades;
    if (r.given.contains("--hub-radius") && !(r.hubRR > 0 && r.hubRR < 1))
        return bad(2, "--hub-radius", "must be in (0, 1) (a fraction of the tip radius)");
    p.hubRR = r.given.contains("--hub-radius") ? r.hubRR : 0.0;

    for (auto [v, f] : {std::pair{r.diameter, "--diameter"}, {r.rpm, "--rpm"},
                        {r.rho, "--rho"}, {r.nu, "--nu"}, {r.sound, "--sound"}})
        if (!finitePos(v)) return bad(2, f, "must be finite and > 0");
    p.op = {.diameter = r.diameter, .rpm = r.rpm, .rho = r.rho, .nu = r.nu,
            .sound = r.sound};

    auto js = parseAdvance(r.given.contains("--j") ? std::string_view(r.advance)
                                                   : (ll ? "0:1.0:0.1" : "0:1:0.05"));
    if (!js) return std::unexpected(js.error());
    p.js = std::move(*js);
    if (r.given.contains("--dump-span"))
    {
        if (!(r.dumpSpan >= 0)) return bad(2, "--dump-span", "need J >= 0");
        p.dumpSpan = r.dumpSpan;
    }

    if (r.stationsBemt < 4) return bad(2, "--stations-bemt", "must be >= 4");
    p.stations = r.stationsBemt;
    if (r.panels < 4) return bad(2, "--panels", "must be >= 4");
    if (r.alignIters < 0) return bad(2, "--align-iters", "must be >= 0");
    if (!finitePos(r.wakeLength)) return bad(2, "--wake-length", "must be > 0");
    if (!(r.segDeg > 0 && r.segDeg <= 30)) return bad(2, "--seg-deg", "must be in (0, 30]");
    if (!finitePos(r.alignTol)) return bad(2, "--align-tol", "must be > 0");
    if (!(r.core >= 0)) return bad(2, "--core", "must be >= 0 (0 = automatic)");
    if (!(r.stallVisc >= 0)) return bad(2, "--stall-visc", "must be >= 0 (0 = off)");
    p.ll = {.panels = r.panels, .wakeLength = r.wakeLength, .alignIters = r.alignIters,
            .alignTol = r.alignTol, .segDeg = r.segDeg, .core = r.core,
            .hub = prop::HubModel(r.liftHub), .wakePitch = prop::WakePitch(r.wakePitch),
            .stallVisc = r.stallVisc};

    if (r.given.contains("--out") && r.out.empty())
        return bad(2, "--out", "must not be empty (omit it to write no CSV)");
    p.outCsv = r.out;
    return c;
}

std::string describeProp(const prop::Case& p)
{
    using prop::Method;
    if (p.selftest)
        return "case:\n"
               "  mode       lifting-line self-test (Biot-Savart closed forms, "
               "actuator-disk cylinder, Betz/Prandtl limits)\n"
               "  host-only  no GPU, no input files\n";
    const bool ll = p.method == Method::Liftline;
    const double n = p.op.revPerSec();
    const double tipMach = std::numbers::pi * n * p.op.diameter / p.op.sound;
    std::string s = std::format(
        "case:\n"
        "  mode       propeller thrust curve, {}\n"
        "  blade      {} blades, planform {}, hub r/R {}\n"
        "  polar      {}\n"
        "  operating  D={:.6g} m  {:.6g} rpm ({:.6g} rev/s)  rho={:.6g}  nu={:.6g}  "
        "c={:.6g}  tip Mach {:.3f}\n"
        "  J          {} points, {:.6g} .. {:.6g}{}\n",
        ll ? "LIFTING LINE (B blades, prescribed helical wake, Biot-Savart; no "
             "tip-loss factor)"
           : "BEMT (Ning residual, Prandtl tip + hub loss, Buhl branch)",
        p.blades, p.planformPath,
        p.hubRR > 0 ? std::format("{:.6g}", p.hubRR) : std::string("= the planform's first r/R"),
        p.polarPath.empty()
            ? std::format("built-in thin airfoil: 2 pi (alpha - {:.6g} deg), cd0 {:.6g}, "
                          "Viterna past +-10 deg",
                          p.thinAlpha0Deg, p.thinCd0)
            : "table " + p.polarPath,
        p.op.diameter, p.op.rpm, n, p.op.rho, p.op.nu, p.op.sound, tipMach, p.js.size(),
        p.js.front(), p.js.back(),
        p.dumpSpan ? std::format("  (span dump at J = {:g})", *p.dumpSpan) : std::string());
    if (ll)
        s += std::format(
            "  liftline   {} panels, wake {:.6g} D, {} realignments (tol {:.3g}), "
            "segment {:.6g} deg,\n"
            "             core {}, hub {}, wake pitch {}, stall viscosity {:.3g}\n",
            p.ll.panels, p.ll.wakeLength, p.ll.alignIters, p.ll.alignTol, p.ll.segDeg,
            p.ll.core > 0 ? std::format("{:.6g} tip radii", p.ll.core)
                          : std::string("auto (5% of the closest cp-node gap)"),
            prop::hubModelNames()[std::size_t(p.ll.hub)],
            prop::wakePitchNames()[std::size_t(p.ll.wakePitch)], p.ll.stallVisc);
    else
        s += std::format("  bemt       {} elements, cosine-clustered to root and tip\n",
                         p.stations);
    s += std::format("  out        {}\n  host-only  no GPU\n",
                     p.outCsv.empty() ? std::string("(table on stdout only)") : p.outCsv);
    return s;
}

} // namespace dns::cfg
