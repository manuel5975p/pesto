// The azimuth table: precomputed geometry must be indistinguishable from
// geometry built on the spot, and replaying it must be indistinguishable from
// rebuilding the wall.
//
// The load-bearing check is the third one -- table against fresh, over a FULL
// revolution, as an exact equality rather than a tolerance. The first two exist
// because they fail differently: a rotationally symmetric body pins that the
// turn is applied about the right point in the right plane (every bin must come
// out identical), and a body with corners pins that the turn is applied at all.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/rotortable.hpp"
#include "dns/rotor.hpp"

#include "gpu_test_util.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>
#include <vector>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;

constexpr int n = 128;
const double L = 2 * pi, cx = L / 2, cy = L / 2, cz = 0.5;
constexpr double gamma_ = 1.4, rhoInf = 1.0;
const double pInf = 1.0 / (gamma_ * 0.04);

// A body with corners, so its classification genuinely depends on azimuth: the
// L1 ball is the same rotated square --body cube reduces to at nz == 1.
double sdfDiamond(double x, double y, double)
{
    return std::abs(x - cx) + std::abs(y - cy) - L / 16;
}

// A body that is its own rotation, so every bin must classify identically.
double sdfCircle(double x, double y, double)
{
    return std::hypot(x - cx, y - cy) - L / 16;
}

Rotor rotorFor(int bins)
{
    return {.axis = RotorAxis::Z,
            .shaft = {cx, cy, cz},
            .omega = 1.0,
            .radius = L / 16,
            .bins = bins,
            .thicken = 0.0};
}

bool sameTables(const gpu::CompGhostTables& a, const gpu::CompGhostTables& b)
{
    return a.gcell == b.gcell && a.gnbr == b.gnbr && a.gw == b.gw
           && a.tnbr == b.tnbr && a.tw == b.tw
           && a.gscale == b.gscale && a.uwall == b.uwall && a.aspan == b.aspan
           && a.deep == b.deep
           && a.fcell == b.fcell && a.fdir == b.fdir && a.fpos == b.fpos
           && a.solid == b.solid;
}

// The ghost fill a table produces, as a downloaded state, so playback can be
// compared against a wall built from scratch at the same angle.
CompState filledBy(const Grid& g, gpu::DeviceCompGhost& gh, const CompState& in)
{
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!ds) { check(false, "filledBy alloc"); return CompState::zeros(g); }
    (void)ds->upload(in);
    gh.apply(*ds);
    auto out = CompState::zeros(g);
    (void)ds->download(out);
    return out;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    const AxisRemap3 uniform{Focal1D::identity(L), Focal1D::identity(L),
                             Focal1D::identity(1.0)};
    const int bins = 24;
    const Rotor rot = rotorFor(bins);

    // --- a circle is its own rotation ---------------------------------------
    {
        auto tbl = gpu::buildRotorTable(g, rot, [] { return gpu::GhostSdf(sdfCircle); },
                                        nullptr, uniform);
        check(tbl.has_value(), "circle table builds");
        if (tbl)
        {
            bool allSame = true;
            for (int m = 1; m < bins; ++m)
                allSame = allSame && sameTables(tbl->bins[0], tbl->bins[std::size_t(m)]);
            std::printf("circle: %d bins, %d ghosts, all identical %d\n", bins,
                        tbl->capacity.nGhost, int(allSame));
            check(tbl->capacity.nGhost > 0, "circle has mirrored ghosts");
            check(allSame, "a rotationally symmetric body classifies identically "
                           "at every azimuth");
            // The wall velocity is NOT symmetric even when the geometry is, so
            // it must be present and non-trivial.
            bool spun = false;
            for (gpu::Real v : tbl->bins[0].uwall)
                if (v != gpu::Real(0)) spun = true;
            check(spun, "and its wall still moves");
        }
    }

    // --- a diamond does not --------------------------------------------------
    auto table = gpu::buildRotorTable(g, rot, [] { return gpu::GhostSdf(sdfDiamond); },
                                      nullptr, uniform);
    if (!table) { check(false, "diamond table builds"); return dnstest::finish(); }
    {
        // The diamond has FOUR-fold symmetry, and 24 bins divide a quarter turn
        // exactly, so bins 6, 12 and 18 must reproduce bin 0 and every other
        // bin must differ. That pattern is a sharper test than "they all
        // differ": it pins the bin -> angle map itself, since a wrong scale or
        // a half-bin offset would put the repeats somewhere else or nowhere.
        int wrong = 0;
        for (int m = 1; m < bins; ++m)
        {
            const bool same = sameTables(table->bins[0], table->bins[std::size_t(m)]);
            if (same != (m % (bins / 4) == 0)) ++wrong;
        }
        std::printf("diamond: quarter-turn repeats at %d bins, %d mismatches "
                    "(max %d ghosts, %d faces), tip radius %.4f\n",
                    bins / 4, wrong, table->capacity.nGhost,
                    table->capacity.nFace, table->tipRadius);
        check(wrong == 0, "the diamond repeats exactly every quarter turn and "
                          "differs at every other azimuth");
        check(table->tipRadius > 0.9 * rot.radius
                  && table->tipRadius < 1.2 * rot.radius,
              "the measured tip radius matches the body's own corner");
    }

    // --- the table IS the geometry ------------------------------------------
    //
    // For every bin, the stored tables must equal a fresh classification at
    // that bin's angle, array for array. This is the check the whole design
    // rests on: if it holds, a run replaying the table is a run rebuilding the
    // wall every step, exactly.
    {
        int bad = -1;
        for (int m = 0; m < bins; ++m)
        {
            const gpu::GhostSdf turned =
                rotorSdf(rot, gpu::GhostSdf(sdfDiamond), rotorBinAngle(rot, m));
            const gpu::GhostWallMotion uw = gpu::rotorWallMotion(rot);
            const auto fresh = gpu::classifyCompGhost(g, turned, nullptr,
                                                      uniform, &uw);
            if (!sameTables(fresh, table->bins[std::size_t(m)])) { bad = m; break; }
        }
        check(bad < 0, bad < 0 ? "every bin equals a fresh classification at its "
                                 "own angle"
                               : "bin " + std::to_string(bad)
                                     + " differs from a fresh classification");
    }

    // --- the bin index closes a revolution exactly --------------------------
    {
        bool wraps = true;
        for (int m = 0; m < bins; ++m)
        {
            const double th = rotorBinAngle(rot, m);
            for (int turn = 1; turn <= 50; ++turn)
                wraps = wraps && rotorBin(rot, th + turn * 2 * pi) == m;
        }
        check(wraps, "50 revolutions later the same angle is the same bin");
    }

    // --- playback equals a freshly built wall -------------------------------
    {
        auto dev = gpu::DeviceComp::make(g, gamma_, 0.01);
        auto gh = gpu::DeviceCompGhost::fromTables(g, table->bins[0],
                                                   table->capacity, gamma_,
                                                   rhoInf, pInf, false);
        if (!dev || !gh) return EXIT_FAILURE;
        check(gh->moving(), "a table-built wall reports that it moves");

        // A non-uniform field, so the mirror has something to image.
        auto in = CompState::zeros(g);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                const std::size_t c = g.idx(i, j, 0);
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
                in.rho[c] = rhoInf * (1.0 + 0.05 * std::sin(3 * x));
                in.mx[c] = in.rho[c] * 0.3 * std::cos(2 * y);
                in.my[c] = in.rho[c] * 0.2 * std::sin(x + y);
                in.E[c] = pInf / (gamma_ - 1.0)
                          + 0.5 * (in.mx[c] * in.mx[c] + in.my[c] * in.my[c])
                                / in.rho[c];
            }

        int bad = -1;
        double worst = 0.0;
        // Walk the bins in run order, so each load() lands on the buffers the
        // previous bin left behind -- a stale tail would show up here and not
        // in a fresh build per bin.
        for (int m = 0; m < bins; ++m)
        {
            if (gh->load(table->bins[std::size_t(m)]) != hipSuccess)
            {
                check(false, "load bin " + std::to_string(m));
                return dnstest::finish();
            }
            const auto played = filledBy(g, *gh, in);
            auto ref = gpu::DeviceCompGhost::fromTables(
                g, table->bins[std::size_t(m)], table->capacity, gamma_, rhoInf,
                pInf, false);
            if (!ref) return EXIT_FAILURE;
            const auto built = filledBy(g, *ref, in);
            for (std::size_t c = 0; c < g.size(); ++c)
            {
                const double d =
                    std::abs(played.rho[c] - built.rho[c])
                    + std::abs(played.mx[c] - built.mx[c])
                    + std::abs(played.my[c] - built.my[c])
                    + std::abs(played.mz[c] - built.mz[c])
                    + std::abs(played.E[c] - built.E[c]);
                worst = std::max(worst, d);
                if (d != 0.0 && bad < 0) bad = m;
            }
        }
        std::printf("playback: worst |state difference| over %d bins = %.3e\n",
                    bins, worst);
        check(bad < 0, bad < 0 ? "replaying a bin is bit-identical to building "
                                 "the wall at that bin"
                               : "bin " + std::to_string(bad)
                                     + " replayed differently");

        // Capacity is a contract, not a hint: a table that does not fit must be
        // refused rather than truncated into the buffers.
        gpu::CompGhostTables big = table->bins[0];
        big.gcell.push_back(0);
        big.gnbr.resize(big.gnbr.size() + 8, 0);
        big.gw.resize(big.gw.size() + 8, gpu::Real(0));
        big.tnbr.resize(big.tnbr.size() + 8, 0);
        big.tw.resize(big.tw.size() + 8, gpu::Real(0));
        big.gscale.push_back(gpu::Real(0));
        big.uwall.resize(big.uwall.size() + 3, gpu::Real(0));
        big.aspan.push_back(gpu::Real(0));
        check(gh->load(big) != hipSuccess,
              "loading a table larger than the capacity is refused");
    }

    return dnstest::finish();
}
