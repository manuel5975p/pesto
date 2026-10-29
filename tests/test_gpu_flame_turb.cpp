// 3D premixed flames and the turbulent inflow of `dns flame --turb`
// (docs/chem.md, "3D and turbulence"), device API level, the runner's own
// hook set (NSCBC -> species re-carry -> sink -> species sink).
//
//   extrude: a 2D wrinkled premixed flame (nz 1) and the same IC extruded
//            to nz 4 (hz = hy), stepped with the SAME fixed dt: every k
//            plane of rho, m, E, rhoY equals the 2D field bitwise, w == 0.
//   inflow:  (a) the Turb instantiation with ZERO rows is the off kernel
//            bit for bit (uniform targets, and per-row Prof targets);
//            (b) DeviceTurbInflow's rows equal the host FP32 lerp bitwise
//            at several sweeps; (c) a nonzero frozen row set is what the
//            inflow row's u, v, w relax to.
#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/flame1d.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/comp_sink.hpp"
#include "dns/gpu/flame_turb.hpp"
#include "dns/transport.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"
#include "gpu_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

struct Fields
{
    CompState q = CompState::zeros(Grid::make(1, 1, 1, 1, 1, 1));
    std::vector<double> ry;
    std::vector<float> rowsV, rowsU, rowsW; // the inflow row's u, v, w
};

struct Opts
{
    int steps = 200;
    double dt = 0;
    bool prof = false;               // per-row targets at the uniform values
    const gpu::Real* tu = nullptr;   // turbulent rows (null = off)
    const gpu::Real *tv = nullptr, *tw = nullptr;
};

// The premixed runner's device setup on grid g with the IC of case c (whose
// grid must be g), stepped o.steps times at o.dt (o.dt == 0: return the
// stable dt in rowsU[0] without stepping).
Fields run(const Case& c, const Opts& o)
{
    const Grid& g = c.grid;
    const double Tu = c.flameTu, sig = c.flameModel.sigma;
    const auto tr = Transport::air(c.prandtl, Tu, c.tinfK, c.prandtl, kGammaAir,
                                   kRgasBox, c.sutherland);
    auto dev = gpu::DeviceComp::make(g, tr);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !red) std::exit(EXIT_FAILURE);
    gpu::SpeciesSetup sp;
    sp.nY = 1;
    sp.lnA = std::log(c.flameEig.Lambda);
    sp.Ta = c.flameModel.theta() * Tu;
    sp.q = tr.cp() * (sig - 1) * Tu;
    sp.invPr = 1.0 / c.prandtl;
    sp.openX = true;
    sp.yIn[0] = 1.0;
    if (!dev->enableSpecies(sp)) std::exit(EXIT_FAILURE);
    auto ds = gpu::DeviceCompState::make(g.size(), 0, false, 1);
    if (!ds) std::exit(EXIT_FAILURE);
    const double uIn = c.flameUIn;
    gpu::DeviceCompNscbc::Config nc;
    nc.rhoInf = 1.0;
    nc.uInf = uIn;
    nc.pInf = Tu - uIn * uIn * (sig - 1);
    nc.gamma = tr.gamma;
    nc.etaIn = c.nscbcEtaIn;
    nc.Rgas = tr.Rgas;
    nc.TInf = Tu;
    const std::size_t rows = std::size_t(g.ny) * g.nz;
    if (o.prof) {
        nc.inflowU.assign(rows, uIn);
        nc.inflowJet.assign(rows, 1.0);
        nc.inflowRho.assign(rows, 1.0);
    }
    auto nm = gpu::DeviceCompNscbc::make(g, nc);
    auto sm = gpu::DeviceCompSink::make(
        g, {.nb = c.nscbcSink, .alpha = c.nscbcSinkAmp,
            .cInf = std::sqrt(tr.gamma * tr.Rgas * Tu)});
    if (!nm || !sm) std::exit(EXIT_FAILURE);
    gpu::DeviceCompNscbc nsc = std::move(*nm);
    gpu::DeviceCompSink snk = std::move(*sm);
    nsc.setFreestream(uIn);
    if (o.tu) nsc.setTurbulentInflow(o.tu, o.tv, o.tw);
    auto* d = &*dev;
    dev->setRhsHook([&](const gpu::DeviceCompState& s, gpu::DeviceCompState& out,
                        const gpu::StageCombine& cmb) {
        nsc.apply(*d, s, out, cmb);
        d->speciesOpenConsistency(s, out, cmb);
        snk.apply(s, out, cmb);
        snk.applySpecies(s, out, cmb);
    });
    CompState h = CompState::zeros(g);
    std::vector<double> ry;
    flameIC(c, h, ry);
    if (ds->upload(h) != hipSuccess || ds->uploadSpecies(0, ry) != hipSuccess)
        std::exit(EXIT_FAILURE);
    Fields f;
    if (o.dt == 0) {
        f.rowsU.push_back(float(dev->stableDt(*ds, *red, c.cfl)));
        return f;
    }
    for (int n = 0; n < o.steps; ++n) dev->step(*ds, gpu::Real(o.dt));
    f.q = CompState::zeros(g);
    if (ds->download(f.q) != hipSuccess || ds->downloadSpecies(0, f.ry) != hipSuccess)
        std::exit(EXIT_FAILURE);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            const std::size_t cc = g.idx(0, j, k);
            const double r = f.q.rho[cc];
            f.rowsU.push_back(float(f.q.mx[cc] / r));
            f.rowsV.push_back(float(f.q.my[cc] / r));
            f.rowsW.push_back(float(f.q.mz[cc] / r));
        }
    return f;
}

bool sameAll(const Fields& a, const Fields& b)
{
    return a.q.rho == b.q.rho && a.q.mx == b.q.mx && a.q.my == b.q.my &&
           a.q.mz == b.q.mz && a.q.E == b.q.E && a.ry == b.ry;
}

Case caseOf(std::vector<std::string> args)
{
    auto c = dnstest::runCli(std::move(args));
    if (!c) {
        std::printf("case rejected: %s\n", c.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return *c;
}

void extrude()
{
    // 2D wrinkled flame: 6 + 4 delta_th at 8 cells + 2 x 16 band.
    const Case c2 = caseOf({"flame", "--ny", "32", "--cells-per-delta", "8",
                            "--upstream", "6", "--downstream", "4", "--mach",
                            "0.02", "--wrinkle", "0.5", "--wrinkle-modes", "2"});
    Case c3 = c2;
    const Grid& g2 = c2.grid;
    c3.grid = Grid::make(g2.nx, g2.ny, 4, g2.nx * g2.hx, g2.ny * g2.hy,
                         4 * g2.hy);
    const double dt2 = run(c2, {.dt = 0}).rowsU[0];
    const double dt3 = run(c3, {.dt = 0}).rowsU[0];
    const double dt = std::min(dt2, dt3);
    std::printf("grid %d x %d (x 4), dt 2D %.4e, 3D %.4e, fixed dt %.4e, 200 "
                "steps\n",
                g2.nx, g2.ny, dt2, dt3, dt);
    const Fields a = run(c2, {.steps = 200, .dt = dt});
    const Fields b = run(c3, {.steps = 200, .dt = dt});
    long bad = 0;
    double wmax = 0;
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < g2.ny; ++j)
            for (int i = 0; i < g2.nx; ++i) {
                const std::size_t p = g2.idx(i, j, 0);
                const std::size_t q = c3.grid.idx(i, j, k);
                bad += a.q.rho[p] != b.q.rho[q] || a.q.mx[p] != b.q.mx[q] ||
                       a.q.my[p] != b.q.my[q] || a.q.E[p] != b.q.E[q] ||
                       a.ry[p] != b.ry[q];
                wmax = std::max(wmax, std::abs(b.q.mz[q]));
            }
    // The run must have moved: the wrinkled front is not the IC any more.
    double moved = 0;
    {
        CompState h = CompState::zeros(g2);
        std::vector<double> ry;
        flameIC(c2, h, ry);
        for (std::size_t p = 0; p < h.my.size(); ++p)
            moved = std::max(moved, std::abs(a.q.my[p] - h.my[p]));
    }
    std::printf("  cells differing from the 2D field: %ld of %zu; max |rho w| "
                "%.3g; max |rho v - IC| %.3g\n",
                bad, c3.grid.size(), wmax, moved);
    check(moved > 0, "the 2D wrinkled run evolved (rho v != IC)");
    check(bad == 0, "every k plane of rho, m, E, rhoY == the 2D field bitwise");
    check(wmax == 0, "rho w == 0 exactly");
}

void inflow()
{
    const Case c = caseOf({"flame", "--ny", "8", "--nz", "8",
                           "--cells-per-delta", "8", "--upstream", "6",
                           "--downstream", "4", "--mach", "0.02",
                           "--wrinkle", "0.5", "--wrinkle-modes", "1"});
    const Grid& g = c.grid;
    const std::size_t rows = std::size_t(g.ny) * g.nz;
    const double dt = run(c, {.dt = 0}).rowsU[0];

    // (a) zero rows == off, uniform and Prof targets.
    auto zb = gpu::DeviceScalar::make(rows);
    if (!zb) std::exit(EXIT_FAILURE);
    const std::vector<gpu::Real> zeros(rows, 0.0f);
    if (zb->upload(zeros.data()) != hipSuccess) std::exit(EXIT_FAILURE);
    const gpu::Real* z = zb->data();
    {
        const Fields off = run(c, {.steps = 100, .dt = dt});
        const Fields on = run(c, {.steps = 100, .dt = dt, .tu = z, .tv = z, .tw = z});
        check(sameAll(off, on), "(a) Turb with zero rows == the off kernel "
                                "bitwise (uniform targets, 100 steps)");
        const Fields poff = run(c, {.steps = 100, .dt = dt, .prof = true});
        const Fields pon = run(c, {.steps = 100, .dt = dt, .prof = true,
                                   .tu = z, .tv = z, .tw = z});
        check(sameAll(poff, pon), "(a) Prof + Turb with zero rows == Prof "
                                  "bitwise (100 steps)");
        check(sameAll(off, poff), "(a) uniform Prof rows == uniform targets "
                                  "bitwise (gpu_flame_passive (f) in 3D)");
    }

    // (b) device rows == host FP32 lerp, bitwise.
    const int nbx = 8;
    const double h = g.hx;
    std::array<std::vector<double>, 3> box;
    for (int cc = 0; cc < 3; ++cc) {
        box[cc].resize(std::size_t(nbx) * rows);
        for (std::size_t r = 0; r < rows; ++r)
            for (int i = 0; i < nbx; ++i)
                box[cc][std::size_t(i) + nbx * r] =
                    0.3 * std::sin(2 * M_PI * i / nbx + 0.37 * double(r) + cc) +
                    0.05 * std::cos(1.3 * double(r) - cc * i);
    }
    auto ti = gpu::DeviceTurbInflow::make(nbx, g.ny, g.nz, h, box);
    if (!ti) std::exit(EXIT_FAILURE);
    bool bitwise = true;
    for (const double X : {0.0, 0.37 * h, 5.5 * h, 3.0 * nbx * h + 0.1 * h,
                           1234.567, -2.1 * h}) {
        ti->sample(X);
        const gpu::TurbSlice s = gpu::turbSliceOf(X, h, nbx);
        for (int cc = 0; cc < 3; ++cc) {
            std::vector<gpu::Real> dv(rows);
            if (ti->rows(cc).download(dv.data()) != hipSuccess) std::exit(EXIT_FAILURE);
            for (std::size_t r = 0; r < rows; ++r) {
                const float a = float(box[cc][std::size_t(s.i0) + nbx * r]);
                const float b = float(box[cc][std::size_t(s.i1) + nbx * r]);
                const float e = gpu::turbLerp(a, b, s.f);
                if (std::memcmp(&dv[r], &e, sizeof(float)) != 0) bitwise = false;
            }
        }
    }
    const gpu::TurbSlice s0 = gpu::turbSliceOf(0.0, h, nbx);
    check(s0.i0 == 0 && s0.f == 0.0f, "(b) X = 0 reads box column 0 (the IC's)");
    const gpu::TurbSlice s1 = gpu::turbSliceOf(0.25 * h, h, nbx);
    check(s1.i0 == nbx - 1 && s1.f == 0.75f,
          "(b) X = h/4 reads 3/4 of the way from column nbx-1 to 0 (upstream "
          "of the face)");
    check(bitwise, "(b) device rows == host fmaf lerp bitwise at 6 sweeps");

    // (c) a held nonzero row set is what the inflow row relaxes to: a
    // smooth, solenoidal row pattern (v(z), w(y), u(y, z); one wavelength per
    // box), every box column the same, at n = 16 and 32 cells per
    // wavelength. The LODI row keeps the transverse fluxes, which pull the
    // row off its target against the relax rate (etaIn c/L_x): the residual
    // must shrink with the wavelength. Measured 2026-10-29 at n 16: v, w 4.7%,
    // u 6.0% of the amplitude; n 32: 1.4%, 4.4% (the run: >= 50 cells per peak
    // wavelength).
    double prev = HUGE_VAL;
    for (const int n : {16, 32}) {
        const std::string ns = std::to_string(n);
        const Case cn = caseOf({"flame", "--ny", ns, "--nz", ns,
                                "--cells-per-delta", "8", "--upstream", "6",
                                "--downstream", "4", "--mach", "0.02"});
        const Grid& gn = cn.grid;
        const std::size_t rn = std::size_t(gn.ny) * gn.nz;
        std::array<std::vector<double>, 3> sm;
        for (auto& v : sm) v.resize(std::size_t(nbx) * rn);
        for (int k = 0; k < gn.nz; ++k)
            for (int j = 0; j < gn.ny; ++j) {
                const double y = 2 * M_PI * (j + 0.5) / gn.ny;
                const double zz = 2 * M_PI * (k + 0.5) / gn.nz;
                const std::size_t r = std::size_t(j) + std::size_t(gn.ny) * k;
                for (int i = 0; i < nbx; ++i) {
                    sm[0][std::size_t(i) + nbx * r] = 0.3 * std::cos(y) * std::cos(zz);
                    sm[1][std::size_t(i) + nbx * r] = 0.3 * std::sin(zz);
                    sm[2][std::size_t(i) + nbx * r] = 0.3 * std::sin(y);
                }
            }
        auto ts = gpu::DeviceTurbInflow::make(nbx, gn.ny, gn.nz, gn.hx, sm);
        if (!ts) std::exit(EXIT_FAILURE);
        ts->sample(0.37 * h);
        std::vector<gpu::Real> du(rn), dv(rn), dw(rn);
        if (ts->rows(0).download(du.data()) != hipSuccess ||
            ts->rows(1).download(dv.data()) != hipSuccess ||
            ts->rows(2).download(dw.data()) != hipSuccess)
            std::exit(EXIT_FAILURE);
        const double dtn = run(cn, {.dt = 0}).rowsU[0];
        const Fields t = run(cn, {.steps = 300, .dt = dtn, .tu = ts->du(),
                                  .tv = ts->dv(), .tw = ts->dw()});
        // u against the laminar run's own inflow row (the start transient's
        // sag is not the turbulence's): the fluctuation part only.
        const Fields lam = run(cn, {.steps = 300, .dt = dtn});
        double eu = 0, ev = 0, ew = 0, amp = 0;
        for (std::size_t r = 0; r < rn; ++r) {
            eu = std::max(eu, std::abs(double(t.rowsU[r]) - lam.rowsU[r] - du[r]));
            ev = std::max(ev, double(std::abs(t.rowsV[r] - dv[r])));
            ew = std::max(ew, double(std::abs(t.rowsW[r] - dw[r])));
            amp = std::max({amp, std::abs(double(du[r])), std::abs(double(dv[r])),
                            std::abs(double(dw[r]))});
        }
        const double worst = std::max({eu, ev, ew}) / amp;
        const double shear = std::max(ev, ew) / amp;
        std::printf("  (c) n %d, 300 steps: |row - target| / amplitude: u - "
                    "u_laminar %.4f, v %.4f, w %.4f\n",
                    n, eu / amp, ev / amp, ew / amp);
        check(worst <= 0.08, "(c) n " + ns + ": the inflow row u', v, w within "
                             "8% of the row amplitude of their targets");
        // The shear rows (v, w) relax at etaIn c/L_x against the transverse
        // fluxes: their residual falls ~k^2 (measured 3.3x per halving). u
        // rides the acoustic L5 relax (half that rate) and its residual
        // falls slowly (6.0% -> 4.4%): recorded in issues.md, not gated
        // beyond 8%.
        if (n == 32)
            check(shear < 0.5 * prev, "(c) the v, w residual at n 32 is under "
                                      "half of n 16's (transverse fluxes, k-scaled)");
        prev = shear;
    }
}

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "extrude") extrude();
    else if (mode == "inflow") inflow();
    else {
        std::printf("usage: test_gpu_flame_turb extrude|inflow\n");
        return EXIT_FAILURE;
    }
    return dnstest::finish();
}
