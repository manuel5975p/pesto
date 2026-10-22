// dns conv --physics incompressible: the Boussinesq Rayleigh-Benard path.
//
// The check that matters most is the LAST one. The compressible and Boussinesq
// branches share nothing but the case derivation and the artefact sink -- one
// carries density as a state variable and runs on an acoustic time step, the
// other is divergence-free with a spectral pressure solve and no sound speed
// at all. At the same Rayleigh number they must transport the same heat. That
// agreement is a real cross-validation: a sign error, a mis-scaled buoyancy or
// a wrong wall condition in either one breaks it, and nothing else in the tree
// would notice.
//
// Checks (2D, small grids, a few seconds each):
//   1. Ra = 1000 < 1708: exactly quiescent, Nu = 1 -- also the test that the
//      conduction profile is an equilibrium of the discretisation.
//   2. The projection is healthy: interior max|div u| stays tiny. (Measured on
//      the interior on purpose: the antimirrored ghost rows have a legitimately
//      large divergence that would swamp a whole-grid max.)
//   3. Supercritical Nu in band, and the two plates plus the interior flux
//      agree -- the heat balance closing.
//   4. Boussinesq Nu == compressible Nu at Ra 5e4, to 10%.
//   5. --sides periodic is bit-for-bit the flagless run; --sides closed is a
//      sealed cavity that still projects cleanly (a DIFFERENT pressure problem:
//      the all-Neumann box), still closes its heat balance, and transports less
//      heat because the no-slip sides brake the rolls -- and the two physics
//      branches agree about it too. In 3D as well, where a second ghost ring
//      and a third DCT axis are in play.
//
// No HIP device -> skip (exit 77).

#include "dns/gpu/boussinesq.hpp"
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "dns/gpu/hotpatch.hpp"
#include "dns/gpu/runner_rb.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;


struct Tail
{
    bool ok = false;
    double nuBot = 0, nuTop = 0, nuFlux = 0, urms = 0, maxDiv = 0;
    // Largest INSTANTANEOUS gap between the two wall fluxes over the run.
    // Small time-mean gap = the heat balance closing. Small INSTANTANEOUS gap
    // everywhere = the flow is stuck in the up-down symmetric state the
    // Boussinesq equations are equivariant under, in which case the two are
    // equal by construction and prove nothing. See rbIC's cos(pi y/H) term.
    double maxWallGap = 0;
};

// Time-average over the last 60% of rows, past the growth transient. The
// instantaneous Nusselt number of a convecting layer swings by tens of
// percent, so a single row means nothing.
Tail tailMean(const std::string& prefix, bool wantDiv)
{
    Tail r;
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    if (!tbl) return r;
    const auto& nb = tbl->col("nuBot");
    const auto& nt = tbl->col("nuTop");
    const auto& nf = tbl->col("nuFlux");
    const auto& ur = tbl->col("urms");
    if (nb.size() < 8) return r;
    const std::size_t i0 = nb.size() * 2 / 5;
    const double n = double(nb.size() - i0);
    for (std::size_t i = i0; i < nb.size(); ++i)
    {
        r.nuBot += nb[i] / n;
        r.nuTop += nt[i] / n;
        r.nuFlux += nf[i] / n;
        r.urms += ur[i] / n;
    }
    if (wantDiv)
        for (double d : tbl->col("maxDiv")) r.maxDiv = std::max(r.maxDiv, d);
    for (std::size_t i = 0; i < nb.size(); ++i)
        r.maxWallGap = std::max(r.maxWallGap, std::abs(nb[i] - nt[i]));
    r.ok = std::isfinite(r.nuBot) && std::isfinite(r.nuTop)
        && std::isfinite(r.nuFlux);
    return r;
}

std::string casePrefix(const std::string& name)
{
    return "/tmp/dns_test_rb_" + name;
}

Tail runCase(const std::string& name, const std::string& physics, double ra,
             int nx, int ny, double tt, const std::string& extra = "",
             int nz = 1)
{
    const std::string prefix = casePrefix(name);
    const std::string cmd = "\"" DNS_BIN "\" conv --physics " + physics
                          + " --nz " + std::to_string(nz) + " --n "
                          + std::to_string(nx) + " --ny "
                          + std::to_string(ny) + " --rayleigh "
                          + std::to_string(ra) + " --tt " + std::to_string(tt)
                          + " " + extra + " --emit stats --quiet --out " + prefix;
    const int rc = valtest::runBinary(cmd);
    check(rc == 0, name + ": run exits 0");
    if (rc != 0) return {};
    return tailMean(prefix, physics == "incompressible");
}

// Byte-for-byte file comparison: the strongest form of "this flag changed
// nothing", and the only one that catches a stray reordering.
bool sameFile(const std::string& a, const std::string& b)
{
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    if (!fa || !fb) return false;
    return std::equal(std::istreambuf_iterator<char>(fa),
                      std::istreambuf_iterator<char>(),
                      std::istreambuf_iterator<char>(fb),
                      std::istreambuf_iterator<char>());
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();

    // 0. The one thing a stats file cannot show: that the hot patch is counted
    //    over the FLUID cells. Its predicate is a fraction of the index range,
    //    so a version that counted the lateral ghost columns would shift and
    //    widen the heated strip by a cell -- a few percent of area, which no
    //    Nusselt number would betray. Assert it where it is exact instead: the
    //    discrete area fraction of a closed grid must equal the periodic grid's
    //    with the same FLUID counts. Host-side, no kernel involved.
    {
        auto frac = [](int nx, int ny, int nz, double patch, bool closed) {
            const dns::gpu::GpuGrid g{nx, ny, nz, 1.0f, 1.0f, 1.0f};
            return dns::gpu::hotPatchAreaFraction(g, patch, closed);
        };
        // 32 fluid columns either way; closed carries two ghosts on top.
        const double p2 = frac(32, 18, 1, 0.5, false);
        const double c2 = frac(34, 18, 1, 0.5, true);
        check(p2 == c2 && p2 == 0.5,
              "hot patch: 2D closed counts the same fluid columns as periodic");
        // 3D closes z too, so BOTH offsets have to be right; the area is F^2.
        const double p3 = frac(24, 18, 24, 0.5, false);
        const double c3 = frac(26, 18, 26, 0.5, true);
        check(p3 == c3 && p3 == 0.25,
              "hot patch: 3D closed counts the same fluid columns as periodic");
        check(frac(26, 18, 26, 1.0, true) == 1.0,
              "hot patch: F = 1 covers the whole floor of a closed box");
    }

    // 1. Subcritical: the conduction state is an equilibrium of the scheme.
    {
        const Tail s = runCase("sub", "incompressible", 1000.0, 64, 32, 40.0);
        check(s.ok, "subcritical: stats are finite");
        if (s.ok)
        {
            check(std::abs(s.nuBot - 1.0) < 0.01
                      && std::abs(s.nuTop - 1.0) < 0.01,
                  "subcritical: Nu = 1 to 1% (pure conduction)");
            check(s.urms < 1e-3, "subcritical: the layer stays at rest");
        }
    }

    // 2-3. Supercritical: convecting, in band, heat balance closed, and the
    //      projection healthy throughout.
    const Tail b = runCase("ra5e4", "incompressible", 5e4, 96, 48, 40.0);
    check(b.ok, "supercritical: stats are finite");
    if (b.ok)
    {
        check(b.maxDiv < 1e-4,
              "projection healthy: interior max|div u| < 1e-4");
        check(b.nuBot > 3.0 && b.nuBot < 4.8, "Ra 5e4: Nu in [3.0, 4.8]");
        check(b.urms > 0.05, "Ra 5e4: the layer is actually moving");
        const double spread = std::max({std::abs(b.nuBot - b.nuTop),
                                        std::abs(b.nuBot - b.nuFlux),
                                        std::abs(b.nuTop - b.nuFlux)});
        check(spread / b.nuBot < 0.10,
              "wall and interior Nusselt agree to 10%");
        // The two together: the plates must agree in the MEAN (heat balance)
        // while differing INSTANTANEOUSLY (not symmetry-locked). Without the
        // second, the first is vacuous -- measured, a symmetric trip made them
        // identical to six digits at every step.
        check(b.maxWallGap > 0.05 * b.nuBot,
              "the up-down symmetry is broken (plates fluctuate independently)");
    }

    // 4. The cross-validation: two solvers that share no physics code agreeing
    //    on the transported heat.
    const Tail c = runCase("ra5e4c", "compressible", 5e4, 96, 48, 40.0);
    check(c.ok, "compressible reference: stats are finite");
    if (b.ok && c.ok)
    {
        const double rel = std::abs(b.nuBot - c.nuBot)
                         / (0.5 * (b.nuBot + c.nuBot));
        std::printf("  Nu: Boussinesq %.3f vs compressible %.3f (%.1f%%)\n",
                    b.nuBot, c.nuBot, 100 * rel);
        // 15%, not tighter: these are two different models sampled over a
        // finite window, and the compressible branch additionally normalises
        // Nu on the full drop while Ra is on the superadiabatic one, which
        // biases it low by that ratio. Tightening this band would be pinning
        // sampling noise.
        check(rel < 0.15,
              "Boussinesq and compressible Nu agree to 15% at Ra 5e4");
    }

    // 5. --hot-patch. F = 1 must be BIT-FOR-BIT the flagless run: the whole
    //    point of a default-off knob. F < 1 must then actually change the
    //    flow, and must still conserve heat (the patch heats, the surround
    //    absorbs, and the lid takes the balance).
    {
        const Tail one = runCase("patch1", "incompressible", 5e4, 64, 32, 20.0,
                                 "--hot-patch 1");
        check(one.ok, "--hot-patch 1 runs");
        const Tail none = runCase("patch0", "incompressible", 5e4, 64, 32, 20.0);
        check(none.ok, "flagless reference runs");
        check(sameFile(casePrefix("patch1") + "_stats.csv",
                       casePrefix("patch0") + "_stats.csv"),
              "--hot-patch 1 is bit-for-bit the uniformly heated plate");

        const Tail third = runCase("patch3", "incompressible", 5e4, 64, 32, 20.0,
                                   "--hot-patch 0.3333");
        check(third.ok, "--hot-patch 1/3 runs");
        if (third.ok && none.ok)
        {
            check(std::abs(third.nuBot - none.nuBot) > 0.1,
                  "a partially heated floor transports differently");
            check(third.maxDiv < 1e-4,
                  "patched floor: projection still healthy");
            // The floor's NET flux must equal what the lid removes: with a
            // cold surround the floor tally is heat in on the patch minus heat
            // out around it, and in a statistically steady state that net has
            // nowhere to go but the lid.
            check(std::abs(third.nuBot - third.nuTop)
                      < 0.25 * std::abs(third.nuTop),
                  "patched floor: net floor flux balances the lid to 25%");
        }
    }

    // 6. --sides. `periodic` spelled out must be BIT-FOR-BIT the flagless run
    //    -- the default-off proof -- and `closed` must then be a different,
    //    healthy flow: sealed, in balance, and braked.
    {
        const Tail per = runCase("sidesper", "incompressible", 5e4, 64, 32, 20.0,
                                 "--sides periodic");
        check(per.ok, "--sides periodic runs");
        check(sameFile(casePrefix("sidesper") + "_stats.csv",
                       casePrefix("patch0") + "_stats.csv"),
              "--sides periodic is bit-for-bit the flagless run");

        const Tail cl = runCase("sidescl", "incompressible", 5e4, 96, 48, 40.0,
                                "--sides closed");
        check(cl.ok, "--sides closed runs");
        if (cl.ok && b.ok)
        {
            // The projection has to stay healthy through a DIFFERENT pressure
            // problem: closing the sides swaps the FFT-in-x + tridiagonal-in-y
            // solver for the all-Neumann box. A wrong transform there shows up
            // as divergence, not as a plausible-looking Nusselt number.
            check(cl.maxDiv < 1e-4,
                  "closed: interior max|div u| < 1e-4 (all-Neumann solve is "
                  "consistent with the MAC divergence)");
            check(cl.urms > 0.05, "closed: the cavity is actually moving");
            // With INSULATING sides the floor's input can only equal the lid's
            // output if nothing crosses the lateral walls -- so the balance IS
            // the leak test. 12% rather than the periodic 10%: the braked
            // corners make the plate fluxes genuinely less symmetric.
            const double spread = std::max({std::abs(cl.nuBot - cl.nuTop),
                                            std::abs(cl.nuBot - cl.nuFlux),
                                            std::abs(cl.nuTop - cl.nuFlux)});
            check(spread / cl.nuBot < 0.12,
                  "closed: wall and interior Nusselt agree to 12%");
            // Sidewall braking, measured 2026-07-26 at Ra 5e4, aspect 2, 2D:
            // 3.97 periodic -> 3.65 closed. Deterministic runs, so this is not
            // a noise comparison.
            std::printf("  Nu: periodic %.3f vs closed %.3f\n", b.nuBot,
                        cl.nuBot);
            check(cl.nuBot < b.nuBot,
                  "closed: Nu is BELOW the periodic layer's");
        }

        // The two lateral knobs have to COMPOSE, and the patch is the one that
        // can silently disagree with them: its predicate is a fraction of the
        // index range, so a version counting the ghost columns would shift and
        // widen the heated strip. patch = 1 in a closed box must still be
        // bit-for-bit the un-patched closed box.
        const Tail cp1 = runCase("sidesclp1", "incompressible", 5e4, 64, 32,
                                 20.0, "--sides closed --hot-patch 1");
        const Tail cp0 = runCase("sidesclp0", "incompressible", 5e4, 64, 32,
                                 20.0, "--sides closed");
        check(cp1.ok && cp0.ok, "closed + hot patch runs");
        check(sameFile(casePrefix("sidesclp1") + "_stats.csv",
                       casePrefix("sidesclp0") + "_stats.csv"),
              "closed + --hot-patch 1 is bit-for-bit the closed box (the patch "
              "is centred on the FLUID extent)");

        // 3D, and this is the check the 2D ones cannot make. Closing the sides
        // in 3D walls off z as well, which means a SECOND ghost ring and a
        // three-axis DCT in the all-Neumann solve. That is exactly where this
        // tree has been bitten before: a plan that silently transforms the
        // wrong axis leaves every 2D case green and every 3D case wrong
        // (gpu_poisson_wally's own history). A leaking z wall, or a z pass that
        // never ran, shows up here as divergence and as a heat balance that
        // does not close -- the z ghost planes would otherwise be whatever the
        // stage kernels last wrote there, and the interior reads them.
        const Tail c3 = runCase("sides3d", "incompressible", 5e4, 24, 16, 60.0,
                                "--sides closed", 24);
        check(c3.ok, "3D closed sides: stats are finite");
        if (c3.ok)
        {
            check(c3.maxDiv < 1e-4,
                  "3D closed: interior max|div u| < 1e-4 (the three-axis "
                  "Neumann solve transforms the axes it claims to)");
            check(c3.urms > 0.05, "3D closed: the cavity is actually moving");
            const double spread = std::max({std::abs(c3.nuBot - c3.nuTop),
                                            std::abs(c3.nuBot - c3.nuFlux),
                                            std::abs(c3.nuTop - c3.nuFlux)});
            check(spread / c3.nuBot < 0.10,
                  "3D closed: wall and interior Nusselt agree to 10% (nothing "
                  "crosses the x or z walls)");
        }

        // The published MASK, which is what the viewer draws and the only place
        // the app layer's own ghost bookkeeping is visible. Ghost cells are not
        // fluid: the two plate rows always, and the lateral columns once those
        // faces are walls. (The compressible twin gets this from convIC and has
        // it pinned in conv_config; this is the Boussinesq branch's, built in
        // dns_main, and nothing checked it before -- not even for the rows.)
        {
            const int n = 48, nyF = 24;
            (void)runCase("sidesmask", "incompressible", 5e4, n, nyF, 2.0,
                          "--sides closed --emit bin");
            const int nx = n + 2, ny = nyF + 2;
            std::ifstream f(casePrefix("sidesmask") + "_mask.bin",
                            std::ios::binary);
            std::vector<float> m(std::size_t(nx) * ny);
            const bool read = bool(f)
                && bool(f.read(reinterpret_cast<char*>(m.data()),
                               std::streamsize(m.size() * sizeof(float))));
            check(read, "closed: the run publishes a mask of the whole grid");
            if (read)
            {
                bool ghostsClear = true, fluidSet = true;
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i)
                    {
                        const bool ghost = i == 0 || i == nx - 1 || j == 0
                            || j == ny - 1;
                        const float v = m[std::size_t(j) * nx + i];
                        if (ghost && v != 0.0f) ghostsClear = false;
                        if (!ghost && v != 1.0f) fluidSet = false;
                    }
                check(ghostsClear,
                      "closed: every ghost cell is masked out (plate rows AND "
                      "lateral columns)");
                check(fluidSet, "closed: every fluid cell is masked in");
            }
        }

        // And the cross-validation again, on the cavity this time: two models
        // sharing no physics code have to agree about a closed box as well,
        // which is what says the lateral wall condition means the same thing in
        // both -- an antimirrored momentum in one and a mirrored one in the
        // other would show up here and nowhere else.
        const Tail clc = runCase("sidesclc", "compressible", 5e4, 96, 48, 40.0,
                                 "--sides closed");
        check(clc.ok, "closed compressible reference runs");
        if (cl.ok && clc.ok)
        {
            const double rel = std::abs(cl.nuBot - clc.nuBot)
                             / (0.5 * (cl.nuBot + clc.nuBot));
            std::printf("  closed Nu: Boussinesq %.3f vs compressible %.3f "
                        "(%.1f%%)\n", cl.nuBot, clc.nuBot, 100 * rel);
            check(rel < 0.15,
                  "closed sides: Boussinesq and compressible Nu agree to 15%");
        }
    }

    // A NaN in the state is a diverged run (exit 7), not "a layer at rest":
    // the dt rule read `cpd > 0 ? cfl/cpd : 1e30`, and a NaN Courant number
    // fails `> 0` exactly like a quiescent one does.
    {
        const auto g = dns::Grid::make(16, 10, 1, 2.0, 1.25, 1.0);
        auto U = dns::VectorField::zeros(g);
        dns::Scalar T(g.size(), 0.0);
        U.u[g.idx(5, 4, 0)] = std::nan("");
        dns::gpu::RbRun rr;
        rr.nu = std::sqrt(0.72 / 1e4);
        rr.kappa = 1.0 / std::sqrt(1e4 * 0.72);
        rr.tEnd = 1.0;
        dns::RunOutput::Config oc;
        oc.prefix = (std::filesystem::temp_directory_path() / "dns_rbnan").string();
        oc.kinds = {dns::Emit::None};
        auto out = dns::RunOutput::make(oc, {}, g, {});
        check(out.has_value(), "NaN case: sink created");
        if (out)
        {
            const auto st = dns::gpu::runRb(g, U, T, rr, *out);
            std::printf("  NaN-seeded runRb: %s\n",
                        st ? "ran to tEnd" : st.error().message.c_str());
            check(!st && st.error().code == 7, "a NaN-seeded state exits 7");
        }
    }

    return dnstest::finish();
}
