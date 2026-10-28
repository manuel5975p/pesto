// The moving rotor's momentum budget closes on tally + swap ledger.
//
// A spinning 3:1 ellipse in a FULLY periodic quiescent box (no far-field
// module: nothing enters or leaves but through the body), 64 bins, two
// revolutions, the runner's own swap sequence (reseedCreated, measure, load,
// the step's entry hook filling the new ghosts). The fluid's linear momentum
// must change by exactly -sum dt F_tally + sum dP_swap, with F tallied on each
// RK stage input at the SSP-RK3 weights 1/6, 1/6, 2/3: the interior fluxes
// telescope, the interface faces are the tally (closed by construction), and
// the only other way content enters or leaves the fluid is a cell changing
// sides at a swap. The control drops the ledger and must miss by >= 10x the
// tolerance, so the test cannot pass on a ledger that is merely small.
//
// It also measures the moving wall's net sink -- the tally's convective part
// minus the swap ledger, i.e. the momentum the swept staircase removes from
// the fluid that no wall stress accounts for -- with and without the freed-
// cell fill, and pins that the fill cuts it (measured 2026-10-28, gfx1100:
// Lz sink -0.1504 image vs -0.0918 filled, of int Q 1.50 / 1.55).
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/rotortable.hpp"
#include "dns/rotor.hpp"

#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;

constexpr int n = 128;
const double L = 2 * pi, cx = L / 2 + 0.013, cy = L / 2 - 0.021;
constexpr double gam = 1.4, mu = 0.01;
const double pInf = 1.0 / (gam * 0.04); // c = 5: tip Mach 0.16 at omega 1
const double ea = L / 8, eb = L / 24;   // semi-axes, 3:1

// Ellipse sdf, the normalized-radius form scaled by the minor axis: a distance
// to first order at the surface, which is all the classification needs.
double sdfEllipse(double x, double y, double)
{
    const double X = (x - cx) / ea, Y = (y - cy) / eb;
    const double r = std::hypot(X, Y);
    const double grad = std::hypot(X / ea, Y / eb) / std::max(r, 1e-12);
    return (r - 1.0) / std::max(grad, 1e-12);
}

struct Budget
{
    double dP[3] = {}, tally[3] = {}, swap[3] = {}, absTally[3] = {};
    double conv[3] = {}; // sum dt * (convective part of the tally): x, y, moment z
    int swaps = 0, reseeded = 0;
};

// Fluid totals (x, y momentum, z angular momentum about the shaft) of `s`
// over the cells that are not solid in `t`.
void fluidTotals(const Grid& g, const gpu::DeviceCompState& s,
                 const gpu::CompGhostTables& t, double out[3])
{
    auto h = CompState::zeros(g);
    (void)s.download(h);
    std::vector<char> solid(g.size(), 0);
    for (int c : t.solid) solid[std::size_t(c)] = 1;
    const double V = g.cellVolume();
    out[0] = out[1] = out[2] = 0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            if (solid[c]) continue;
            const double x = (i + 0.5) * g.hx - cx, y = (j + 0.5) * g.hy - cy;
            out[0] += h.mx[c] * V;
            out[1] += h.my[c] * V;
            out[2] += (x * h.my[c] - y * h.mx[c]) * V;
        }
}

Budget run(bool reseed, bool& ok)
{
    ok = false;
    Budget B;
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    const AxisRemap3 R{Focal1D::identity(L), Focal1D::identity(L),
                       Focal1D::identity(1.0)};
    const Rotor rot{.axis = RotorAxis::Z, .shaft = {cx, cy, 0.5}, .omega = 1.0,
                    .radius = ea, .bins = 64, .thicken = 0.0};
    auto table = gpu::buildRotorTable(
        g, rot, [] { return gpu::GhostSdf(sdfEllipse); }, nullptr, R);
    auto dev = gpu::DeviceComp::make(g, gam, mu);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!table || !dev || !ds || !red) return B;
    auto gh = gpu::DeviceCompGhost::fromTables(g, table->bins[0], table->capacity,
                                               gam, 1.0, pInf, false);
    if (!gh) return B;
    gpu::DeviceCompGhost* G = &*gh;
    gpu::DeviceComp* D = &*dev;
    const std::array<double, 3> pivot{cx, cy, 0.5};
    // The tally of each RK stage INPUT, weighted as SSP-RK3 combines the stage
    // rhs (1/6, 1/6, 2/3): the step's exact fluid momentum change, so the
    // budget closes to round-off rather than to a quadrature of the tally.
    // The stage hook sees exactly those three inputs (Idempotent::no).
    int stageK = 0;
    double stageDt = 0.0;
    bool tallyFailed = false;
    auto stageTally = [&](gpu::DeviceCompState& s) {
        static constexpr double w[3] = {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};
        const double wk = w[stageK % 3] * stageDt;
        ++stageK;
        const auto F = G->force(*D, s, pivot);
        const double f[3] = {F.fx, F.fy, F.qz};
        auto sf = G->surfaceFaces(*D, s, 0.0);
        if (!sf) { tallyFailed = true; return; }
        double fc[3] = {0, 0, 0};
        for (const auto& q : *sf) {
            fc[0] += q.fc[0];
            fc[1] += q.fc[1];
            fc[2] += (q.x[0] - cx) * q.fc[1] - (q.x[1] - cy) * q.fc[0];
        }
        for (int d = 0; d < 3; ++d) {
            B.tally[d] += wk * f[d];
            B.absTally[d] += wk * std::abs(f[d]);
            B.conv[d] += wk * fc[d];
        }
    };
    dev->setStageHook([G, &stageTally](gpu::DeviceCompState& s) {
        G->apply(s);
        stageTally(s);
    });
    dev->setRhsHook([G, D](const gpu::DeviceCompState& s, gpu::DeviceCompState& o,
                           const gpu::StageCombine& c) { G->applyRhs(*D, s, o, c); });

    auto ic = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        ic.rho[c] = 1.0;
        ic.E[c] = pInf / (gam - 1.0);
    }
    if (ds->upload(ic) != hipSuccess) return B;
    G->apply(*ds);

    gpu::SwapLedgerScratch ledger;
    int bin = 0;
    double P0[3];
    fluidTotals(g, *ds, table->bins[0], P0);
    const double tEnd = 2 * 2 * pi / rot.omega;
    double t = 0.0;
    long steps = 0;
    while (t < tEnd) {
        const double dt = std::min(tEnd - t, dev->stableDt(*ds, *red, 0.4));
        const int b = rotorBin(rot, rotorAzimuth(rot, t));
        if (b != bin) {
            const auto& from = table->bins[std::size_t(bin)];
            const auto& to = table->bins[std::size_t(b)];
            if (reseed) {
                auto k = ledger.reseedCreated(*ds, from, to);
                if (!k) return B;
                B.reseeded += *k;
            }
            auto sw = ledger.measure(g, R, {}, *ds, from, to, pivot);
            if (!sw) return B;
            B.swap[0] += sw->mom[0];
            B.swap[1] += sw->mom[1];
            B.swap[2] += sw->moment[2];
            if (G->load(to) != hipSuccess) return B;
            bin = b; // the step's entry hook fills the new ghosts
            ++B.swaps;
        }
        stageK = 0;
        stageDt = dt;
        dev->step(*ds, gpu::Real(dt));
        if (tallyFailed || stageK != 3) return B;
        G->apply(*ds);
        t += dt;
        ++steps;
    }
    double P1[3];
    fluidTotals(g, *ds, table->bins[std::size_t(bin)], P1);
    for (int d = 0; d < 3; ++d) B.dP[d] = P1[d] - P0[d];
    std::printf("%s: %ld steps, %d swaps, %d cells reseeded\n",
                reseed ? "reseed" : "image ", steps, B.swaps, B.reseeded);
    ok = true;
    return B;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const char* name[3] = {"Px", "Py", "Lz"};
    double sink[2][3] = {};
    for (int mode = 0; mode < 2; ++mode) {
        bool ok = false;
        const Budget B = run(mode == 1, ok);
        check(ok, "the run completes");
        if (!ok) return dnstest::finish();
        for (int d = 0; d < 3; ++d) {
            const double pred = -B.tally[d] + B.swap[d];
            const double res = B.dP[d] - pred;
            const double ctl = B.dP[d] + B.tally[d];
            // Linear momentum telescopes exactly in the finite-volume scheme
            // (measured residual <= 2.9e-7, 1.5e-7 of int |F| dt); angular
            // momentum does not --
            // cell-centre arms against face fluxes, O(h^2) -- measured 7.5e-4
            // (image) / 5.1e-4 (reseeded) of dLz.
            const double tol = (d < 2 ? 1e-4 : 2e-3) * B.absTally[d];
            // Net momentum the swept wall removes beyond the wall stress: the
            // tally's convective part minus what the swaps hand back.
            sink[mode][d] = B.conv[d] - B.swap[d];
            std::printf("  %s: dP %+.6e  -int F %+.6e  swaps %+.6e  residual "
                        "%+.3e (tol %.3e)  without ledger %+.3e  conv %+.6e  "
                        "sink %+.6e\n",
                        name[d], B.dP[d], -B.tally[d], B.swap[d], res, tol, ctl,
                        B.conv[d], sink[mode][d]);
            check(std::abs(res) <= tol, std::string(name[d]) +
                                            ": fluid momentum = -int tally + swap ledger");
            check(std::abs(ctl) >= 10 * tol,
                  std::string(name[d]) + ": the ledger is load-bearing (control misses)");
        }
    }
    // The torque sink is the signal (the x/y forces of a spinning ellipse
    // average out over a revolution).
    std::printf("Lz sink: image %+.6e, reseeded %+.6e (ratio %.3f)\n", sink[0][2],
                sink[1][2], sink[1][2] / sink[0][2]);
    check(std::abs(sink[1][2]) < std::abs(sink[0][2]),
          "the freed-cell fill reduces the moving wall's angular-momentum sink");
    return dnstest::finish();
}
