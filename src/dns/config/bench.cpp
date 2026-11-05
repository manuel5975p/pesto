// validate(): dns bench.
#include "dns/config.hpp"
#include "dns/config/internal.hpp"



#include <cassert>
#include <cmath>
#include <numbers>
#include <optional>
#include <string>

using std::numbers::pi;

namespace dns::cfg {

std::expected<Case, ConfigError> validateBench(const RawConfig& r, Case c)
{
    if (r.steps < 1) return bad(2, "--steps", "must be >= 1");
    c.steps = r.steps;
    // bench builds an n^3 box, so it needs the grid guards every other
    // branch applies and this one was missing: `--n 1300` overflowed the
    // int indexing the tree assumes, and `--n 1` produced nz == 1 with
    // hz == 2pi, violating the validator-enforced `nz == 1 => hz == 1`.
    // n >= 4 forecloses both the degenerate-z case (bench has no 2D mode:
    // the box is always n^3) and any n too small to time meaningfully.
    if (r.n < 4) return bad(2, "--n", "must be >= 4 (bench is an n^3 box)");
    auto box = makeBox(r.n, r.n, r.n, 2 * pi, 2 * pi, 2 * pi);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    c.nzInterior = r.n;
    // bench times the SOLVER the user names, so it needs the physics
    // parameters the solver reads: viscosity, Mach (compressible pressure
    // level) and the SGS constant. Before, --physics compressible parsed
    // and then ran the incompressible stepper anyway.
    c.nu = r.nu > 0 ? r.nu : 1.0 / r.re;
    if (!std::isfinite(c.nu) || c.nu < 0)
        return bad(2, "--nu", "must be finite and >= 0");
    c.mach = r.mach;
    if (r.physics == Physics::Compressible) {
        if (auto e = checkMach(c.mach); e) return std::unexpected(*e);
    } else if (r.given.contains("--mach")) {
        return bad(2, "--mach", "--mach is compressible-only");
    }
    c.cs = r.les;
    if (!(std::isfinite(c.cs) && c.cs >= 0))
        return bad(2, "--les", "must be finite and >= 0");
    if (c.cs > 0 && r.physics != Physics::Compressible)
        return bad(2, "--les", "bench --les times the compressible SGS path; "
                               "add --physics compressible (the incompressible "
                               "eddy viscosity is a runner-level substep, not "
                               "part of the periodic stepper this benchmarks)");
    c.sgsModel = r.sgs;
    if (r.given.contains("--sgs") && c.cs == 0)
        return bad(2, "--sgs", "selects the SGS model for --les; give --les CS "
                               "(> 0) or drop --sgs");
    c.lesLabel = c.cs > 0;
    c.meta = {.body = "none",
              .farField = "periodic",
              .physics = name(c.physics),
              .sgs = "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return c;
}

} // namespace dns::cfg
