// The threaded host phase (dns/parallel.hpp) == serial, bit for bit, against
// threads = 1 at 2, an odd count, hardware concurrency and more threads than
// rows:
// - CurvMetrics::build: every table byte-compared on every mesh family the
//   solver builds (focal, channel, 2D sinusoidal, O-grid, revolved wedge with
//   axis faces, full revolution), plus the error path (the SAME first error
//   at every count).
// - sampleCut: lambda / ax / ay / az byte-compared on analytic bodies (3D,
//   z-invariant, nz == 1, explicit band) and on each IC case's body.
// - compressibleBodyIC: state and mask byte-compared on an STL rotor on a
//   focal mesh, an impulsive 2D cylinder and a quiet-start, tripped 3D
//   sphere.
#include "dns/bodies.hpp"
#include "dns/casegeom.hpp"
#include "dns/curvmesh.hpp"
#include "dns/flow_ic.hpp"
#include "dns/geometry.hpp"
#include "dns/ogrid.hpp"
#include "dns/remap.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <string>
#include <thread>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

bool sameBytes(const std::vector<double>& a, const std::vector<double>& b)
{
    return a.size() == b.size() &&
           (a.empty() ||
            std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// Every table of t equals r's byte for byte (hmin included).
bool sameMetrics(const CurvMetrics& t, const CurvMetrics& r)
{
    for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 3; ++c)
            if (!sameBytes(t.S[a][c], r.S[a][c])) return false;
        for (int e = 0; e < 9; ++e)
            if (!sameBytes(t.Ji[a][e], r.Ji[a][e])) return false;
        if (!sameBytes(t.aov[a], r.aov[a])) return false;
    }
    return sameBytes(t.vol, r.vol) && sameBytes(t.aovSteadyZ, r.aovSteadyZ) &&
           std::memcmp(&t.hmin, &r.hmin, sizeof(double)) == 0 &&
           t.wrap == r.wrap;
}

std::vector<int> threadCounts(const Grid& g)
{
    const int hw = int(std::max(1u, std::thread::hardware_concurrency()));
    const int rows = g.ny * g.nz;
    return {2, 3, 7, hw, 0 /* = hardware */, rows + 5 /* clamped to rows */};
}

void compareAll(const std::string& name, const CurvMesh& m)
{
    const auto ref = CurvMetrics::build(m, 1);
    if (!ref) {
        check(false, name + ": serial build failed: " + ref.error());
        return;
    }
    // A non-trivial mesh: the comparison must not pass on empty tables.
    check(ref->vol.size() == m.g.size() && ref->hmin > 0.0 &&
              ref->hmin < 1e300,
          name + ": serial build is populated");
    for (const int nt : threadCounts(m.g)) {
        const auto t = CurvMetrics::build(m, nt);
        check(t && sameMetrics(*t, *ref),
              std::format("{}: threads {} == serial, bit for bit", name, nt));
    }
}

void focal()
{
    // Plateau + blend on all three axes; odd sizes so blocks split rows
    // unevenly.
    const auto g = Grid::make(23, 19, 13, 4.0, 4.0, 2.0);
    const double lo[3] = {1.5, 1.4, 0.7}, hi[3] = {2.5, 2.4, 1.3};
    const double w[3] = {0.5, 0.5, 0.3};
    compareAll("focal 23x19x13", CurvMesh::fromRemap(g, focalRemap(g, true, lo,
                                                                   hi, w, 4.0)));
}

void channel()
{
    // Rows 0 and ny-1 are mirror ghost rows; ly = 2 delta + 2 firstCell.
    const int ny = 18;
    const double beta = 2.0, delta = 1.0;
    const double ly =
        2.0 * delta + 2.0 * ChannelStretchY::firstCell(ny, beta, delta);
    const auto g = Grid::make(11, ny, 9, 6.0, ly, 3.0);
    compareAll("channel 11x18x9", CurvMesh::channelY(g, beta, delta));
}

void planar()
{
    const auto g = Grid::make(17, 15, 1, 2.0, 1.7, 1.0);
    compareAll("sinusoidal nz=1", CurvMesh::sinusoidal(g, 0.05));
}

void ogrid()
{
    const double r = 0.5, rFar = 10.0;
    const auto g = ogridGrid(32, 20, 5, r, rFar, 1.0);
    auto og = buildOGrid(g, sdfOf(CylinderZ{0.0, 0.0, r}),
                         {.rFar = rFar, .beta = 2.5, .span = 1.0});
    if (!og) {
        check(false, "O-grid build: " + og.error());
        return;
    }
    compareAll("O-grid 32x20x5 (Cyclic x Extend)", og->mesh);
}

void revolved(bool wedge)
{
    const double r = 0.5, rFar = 10.0;
    const RevolvedSpec rv = wedge ? RevolvedSpec{.wedge = true, .dphi = 0.05}
                                  : RevolvedSpec{.wedge = false,
                                                 .dphi = 2 * M_PI / 8,
                                                 .nphi = 8};
    const auto g = ogridRevolvedGrid(16, 14, rv, r, rFar);
    const Sdf sphere = [r](double x, double y, double z) {
        return std::sqrt(x * x + y * y + z * z) - r;
    };
    auto og = buildRevolvedOGrid(g, sphere, {.rFar = rFar, .beta = 2.5}, rv);
    if (!og) {
        check(false, "revolved build: " + og.error());
        return;
    }
    compareAll(wedge ? "revolved wedge (MirrorZ, axis faces)"
                     : "full revolution nphi 8 (axis faces)",
               og->mesh);
}

void errorPath()
{
    // Two moved nodes far apart (k = 1 and k = 6): the serial loop reports a
    // cell next to the k = 1 one; so must every split, whichever block
    // finishes first.
    const auto g = Grid::make(10, 9, 8, 1.0, 1.0, 1.0);
    CurvMesh m = CurvMesh::uniform(g);
    m.x[g.idx(4, 3, 1)] += 3.0 * g.hx;
    m.x[g.idx(6, 5, 6)] += 3.0 * g.hx;
    const auto ref = CurvMetrics::build(m, 1);
    check(!ref, "a folded mesh is rejected");
    if (ref) return;
    std::printf("  serial error: %s\n", ref.error().c_str());
    const bool early = ref.error().find(", 0)") != std::string::npos ||
                       ref.error().find(", 1)") != std::string::npos;
    check(early, "the serial error names a cell at k <= 1, not the k = 6 fold");
    for (const int nt : threadCounts(g)) {
        const auto t = CurvMetrics::build(m, nt);
        check(!t && t.error() == ref.error(),
              std::format("threads {}: the serial loop's first error", nt));
    }
}

// sampleCut at o.threads = 1 vs every count: all four arrays byte-equal.
void cutCase(const std::string& name, const Grid& g, const Sdf& sdf,
             SampleOpts o)
{
    o.threads = 1;
    const auto ref = sampleCut(g, sdf, o);
    if (!ref) {
        check(false, name + ": serial sampleCut failed: " + ref.error());
        return;
    }
    // Non-trivial: some cut cells (0 < lambda < 1).
    const bool cutCells = std::ranges::any_of(
        ref->lambda, [](double v) { return v > 0.0 && v < 1.0; });
    check(cutCells, name + ": serial cut has partially fluid cells");
    for (const int nt : threadCounts(g)) {
        o.threads = nt;
        const auto t = sampleCut(g, sdf, o);
        check(t && sameBytes(t->lambda, ref->lambda) &&
                  sameBytes(t->ax, ref->ax) && sameBytes(t->ay, ref->ay) &&
                  sameBytes(t->az, ref->az),
              std::format("{}: threads {} == serial, bit for bit", name, nt));
    }
}

void cuts()
{
    // Block edges that do not divide the grid (B = 8), a z-invariant layer
    // replicated over k, the native 2D grid, and an explicit band.
    const auto g3 = Grid::make(37, 29, 21, 2.0, 2.0, 1.5);
    cutCase("sphere 37x29x21", g3, sdfOf(Sphere{1.0, 1.0, 0.75, 0.4}), {4, 4});
    cutCase("cylinder zInvariant", g3, sdfOf(CylinderZ{1.0, 1.0, 0.4}),
            {8, 8, true});
    const auto g2 = Grid::make(53, 41, 1, 2.0, 2.0, 1.0);
    cutCase("cylinder nz=1", g2, sdfOf(CylinderZ{1.0, 1.0, 0.4}), {8, 8});
    cutCase("sphere band 0.2", g3, sdfOf(Sphere{1.0, 1.0, 0.75, 0.4}),
            {4, 4, false, 0.2});
}

struct Ic
{
    CompState s;
    std::vector<float> mask;
};

Ic bodyIC(const Case& c, const CaseGeometry& geo, int threads)
{
    const auto& g = c.grid;
    const AxisRemap3 fr = focalRemap(g, c.focalActive, c.focalLo, c.focalHi,
                                     c.focalW, c.focalRatio);
    const TanhStretchY yr{double(g.ny) * g.hy, c.stretchBeta};
    Ic r{CompState::zeros(g), std::vector<float>(g.size(), 1.0f)};
    compressibleBodyIC(c, geo, fr, yr, 1.0, 1.0 / 1.4, 1.4, r.s, r.mask,
                       threads);
    return r;
}

bool sameIc(const Ic& a, const Ic& b)
{
    return sameBytes(a.s.rho, b.s.rho) && sameBytes(a.s.mx, b.s.mx) &&
           sameBytes(a.s.my, b.s.my) && sameBytes(a.s.mz, b.s.mz) &&
           sameBytes(a.s.E, b.s.E) && a.mask.size() == b.mask.size() &&
           std::memcmp(a.mask.data(), b.mask.data(),
                       a.mask.size() * sizeof(float)) == 0;
}

void icCase(const std::string& name, std::vector<std::string> args)
{
    const auto c = dnstest::runCli(std::move(args));
    if (!c) {
        check(false, name + ": case rejected: " + c.error().message);
        return;
    }
    const auto geo = buildCaseGeometry(*c);
    if (!geo) {
        check(false, name + ": geometry: " + geo.error());
        return;
    }
    cutCase(name + " cut {8,8}", c->grid, geo->sdf, {8, 8});
    const Ic ref = bodyIC(*c, *geo, 1);
    // Non-trivial: some solid cells and some moving fluid.
    const auto solid = std::ranges::count(ref.mask, 0.0f);
    const bool moving =
        std::ranges::any_of(ref.s.mx, [](double v) { return v != 0.0; });
    std::printf("  %s: %td solid of %zu\n", name.c_str(), solid,
                ref.mask.size());
    check(solid > 0 && std::size_t(solid) < ref.mask.size() && moving,
          name + ": serial IC has solid cells and a moving stream");
    for (const int nt : threadCounts(c->grid))
        check(sameIc(bodyIC(*c, *geo, nt), ref),
              std::format("{}: IC threads {} == serial, bit for bit", name, nt));
}

void ics()
{
    // The DTMB 4119 kind of case at toy size: STL rotor, focal mesh, MRF.
    // --ramp 0: the impulsive start, so the taper moves fluid at t = 0.
    icCase("rotor2 STL focal",
           {"flow", "--physics", "compressible", "--body", DNS_ROTOR2_STL,
            "--robust", "--n", "100", "--ny", "66", "--nz", "66", "--dr", "6",
            "--focal", "auto", "--focal-ratio", "3", "--focal-blend", "1",
            "--rotor-mach", "0.41", "--rotor-axis", "x", "--rotor-frame",
            "mrf", "--mach", "0.08", "--re", "220", "--ramp", "0", "--out",
            "unused"});
    icCase("cylinder 2D impulsive",
           {"flow", "--physics", "compressible", "--body", "cylinder", "--n",
            "96", "--ny", "64", "--re", "100", "--mach", "0.2", "--ramp", "0",
            "--out", "unused"});
    // Quiet start: the fluid moves only through the trip vortex.
    icCase("sphere 3D, trip",
           {"flow", "--physics", "compressible", "--body", "sphere", "--n",
            "48", "--ny", "32", "--nz", "32", "--re", "100", "--mach", "0.2",
            "--ramp", "0.5", "--kick", "0.1", "--out", "unused"});
}

} // namespace

int main()
{
    focal();
    channel();
    planar();
    ogrid();
    revolved(true);
    revolved(false);
    errorPath();
    cuts();
    ics();
    return dnstest::finish();
}
