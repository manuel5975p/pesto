#include "dns/gpu/runner_ogrid.hpp"

#include "dns/interrupt.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/axis_bc.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/rans_stats.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/internal/nk_march.hpp"
#include "dns/gpu/internal/steady_march.hpp"
#include "dns/gpu/nk.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/meshio.hpp"
#include "dns/walldist.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;
using internal::rampFrac;

namespace {

// Curl of the cell-centred velocity through the O-grid's own metrics: index
// space central differences contracted with the cell-centre inverse Jacobian
// (gpu/internal/comp_flux.hpp), so the values are TRUE curl components on a
// body-fitted mesh -- the uniform-spacing kVorticityZCentred stencil would
// report the index-space derivative and be wrong by the local radial spacing.
// At nz == 1 the k stencil wraps onto itself and every z column is exactly
// zero, so the signed wz branch is the only one with content.
//
// The two ghost rows are written as zero: their j stencil reaches across the
// annulus (row 0's lo neighbour is row ny-1) and means nothing.
__global__ void kOGridCurl(GpuGrid g, CurvView m, int signedWz, const Real* u,
                           const Real* v, const Real* w, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (j == 0 || j == g.ny - 1)
    {
        out[c] = Real(0);
        return;
    }
    const Offsets o = offsetsAt(g, i, j, k);
    const Real* vc[3] = {u, v, w};
    Real gd[3][3];
    curvVelGradCentre(m, o, c,
                      [&](std::size_t cell, int comp) { return vc[comp][cell]; },
                      gd);
    const Real wz = gd[1][0] - gd[0][1];
    if (signedWz)
    {
        out[c] = wz;
        return;
    }
    const Real wx = gd[2][1] - gd[1][2], wy = gd[0][2] - gd[2][0];
    out[c] = sqrt(wx * wx + wy * wy + wz * wz);
}

// The real fluid cells of an O-grid: every cell but the two ghost rows, and on
// a revolved mesh also the two ghost columns past the axis and the wedge's
// two ghost planes.
std::vector<std::uint8_t> ogridFluidMask(const Grid& g, const OGrid& og)
{
    if (!og.revolved) return rowMask(g, 1, g.ny - 2);
    std::vector<std::uint8_t> m(g.size(), 0);
    const int kLo = og.rev.wedge ? 1 : 0, kHi = og.rev.wedge ? 1 : g.nz - 1;
    for (int k = kLo; k <= kHi; ++k)
        for (int j = 1; j <= g.ny - 2; ++j)
            for (int i = 1; i <= og.nTheta; ++i) m[g.idx(i, j, k)] = 1;
    return m;
}

// Is wall face (i, k) a real one (not a ghost column / ghost plane)?
bool realWallFace(const OGrid& og, int i, int k)
{
    if (!og.revolved) return true;
    return i >= 1 && i <= og.nTheta && (!og.rev.wedge || k == 1);
}

} // namespace

std::expected<RunStats, RunError>
runOGrid(const Grid& g, const OGrid& og, const CompState& ic, const OGridRun& r,
         RunOutput& out)
{
    const auto gg = GpuGrid::from(g);

    const bool rans = r.rans != RansModel::None;
    const bool sst = ransSst(r.rans);
    const bool steady = r.steadyTol > 0;
    auto mtr = CurvMetrics::build(og.mesh);
    if (!mtr) return fail(2, "O-grid metrics: " + mtr.error());
    auto dev = DeviceComp::make(*mtr, r.tr, r.les, r.sgs);
    // Explicit steady: the compensated state, so pseudo-time updates below
    // ulp(q)/2 still accumulate (docs/rans.md, "The FP32 update floor"); both
    // O-grid rows zero the low words they overwrite. The Newton-Krylov march
    // keeps its own FP64 iterate and hands the kernels a plain state.
    const bool implicit = steady && r.steadyImplicit;
    const bool compensated = (steady && !implicit) || r.compensated;
    auto ds = DeviceCompState::make(g.size(), ransScalars(r.rans), compensated);
    auto red = ReduceScratch::make();
    if (!dev || !ds || !red) return fail(6, "GPU allocation failed");
    if (implicit && og.revolved && !og.rev.wedge)
        return fail(2, "Newton-Krylov on the full-revolution O-grid is not "
                       "built (the validator enforces this)");

    if (r.rotor.active() && !r.rotorMrf)
        return fail(2, "the O-grid has no moving-geometry rotor: spinning the "
                       "body would spin the mesh. --rotor-frame mrf is the "
                       "supported form (the validator enforces this)");
    if (r.rotor.active())
    {
        // --rotor-frame mrf on the body-fitted mesh. The real tip radius is
        // EXACT here -- the wall is a node row -- so the tip-Mach guard uses
        // it directly; omega stays quoted on the nominal D/2 like the
        // Cartesian path (one definition, config.cpp).
        const double* cs = r.rotor.shaft.data();
        const double tipR = og.rWallMax;
        const double cInf = std::sqrt(r.tr.gamma * r.pInf / r.rhoInf);
        if (auto e = internal::checkTipMach(r.rotor.omega, tipR, r.rotor.radius,
                                            cInf, std::abs(r.uInf) / cInf,
                                            "wall tip radius", "wall", "", "");
            !e)
            return std::unexpected(e.error());

        // Zone: f = 1 over the whole wall (that is what cancels the moving
        // wall's normal flux against the transport correction exactly), the
        // blend inside the annulus, and psi CONSTANT before the far rows so
        // the characteristic far field never sees the frame. rFar and the
        // nominal fluid radial spacing come off the mesh itself.
        const auto pf = og.mesh.node(0, g.ny - 1, 0);
        const double rFar = std::hypot(pf[0] - cs[0], pf[1] - cs[1]);
        const double hr = (rFar - og.rWallMax) / std::max(1, g.ny - 3);
        MrfZone z;
        z.axis = 2;
        z.omega = r.rotor.omega;
        for (int d = 0; d < 3; ++d) z.center[d] = cs[d];
        z.r1 = tipR + std::max(0.05 * r.D, 2.0 * hr);
        z.r2 = z.r1 + std::max(0.15 * r.D, 4.0 * hr);
        // The O-grid is z-extruded with a periodic span: no axial boundary
        // exists, so the axial blend is degenerate exactly as the 2D
        // Cartesian zone's (setMrfCurv requires it).
        z.a1 = 1e30;
        z.a2 = 2e30;
        // Margin 4 nominal cells: with radial clustering the OUTER cells are
        // the widest, so the nominal spacing under-counts there.
        if (z.r2 >= rFar - 4.0 * hr)
            return fail(2, std::format(
                "MRF blend (r2 {:.3g}) reaches the far rows (rFar {:.3g}): "
                "the rotating frame must vanish before the characteristic "
                "boundary. Increase --ogrid-far",
                z.r2, rFar));
        std::printf("rotor: MRF zone about the shaft: f=1 to r %.4g, 0 past "
                    "%.4g (frozen rotor -- exact for a circular section, an "
                    "APPROXIMATION otherwise)\n",
                    z.r1, z.r2);
        std::fflush(stdout);
        if (dev->setMrfCurv(z, og.mesh) != hipSuccess)
            return fail(6, "MRF zone init failed");
    }

    DeviceOGridBc::Config bc;
    bc.gamma = r.tr.gamma;
    bc.Rgas = r.tr.Rgas;
    bc.thermal = r.thermal;
    bc.Twall = r.Twall;
    bc.rhoInf = r.rhoInf;
    bc.uInf = r.uInf;
    bc.pInf = r.pInf;
    bc.sigma = r.farSigma;
    bc.band = r.farBand;
    if (r.rotor.active())
    {
        bc.omega = r.rotor.omega;
        bc.cx = r.rotor.shaft[0];
        bc.cy = r.rotor.shaft[1];
        bc.wallPos = ogridWallMidpoints(og);
        bc.wallSpan = ogridWallSpans(og);
    }
    bc.tr = r.tr;
    if (!r.wallDist.empty()) bc.wallD1 = wallFirstCellDistance(g, r.wallDist);
    if (rans)
    {
        bc.rans = r.rans;
        bc.phiInf[0] = r.phiInf[0];
        bc.phiInf[1] = r.phiInf[1];
    }
    // A revolved mesh: the far row on each far face's own 3D normal, and the
    // tallies over the REAL wall faces (wedge: one plane times 2 pi/dphi).
    const bool rev = og.revolved;
    const bool wedge = rev && og.rev.wedge;
    if (rev)
    {
        bc.farNormal.resize(std::size_t(g.nx) * g.nz);
        for (int k = 0; k < g.nz; ++k)
            for (int i = 0; i < g.nx; ++i)
            {
                const std::size_t c = g.idx(i, g.ny - 2, k);
                const double s[3] = {mtr->S[1][0][c], mtr->S[1][1][c],
                                     mtr->S[1][2][c]};
                const double l = std::hypot(s[0], s[1], s[2]);
                bc.farNormal[std::size_t(k) * g.nx + i] = {s[0] / l, s[1] / l,
                                                            s[2] / l};
            }
        bc.faceILo = 1;
        bc.faceIHi = og.nTheta;
        bc.faceKLo = wedge ? 1 : 0;
        bc.faceKHi = wedge ? 1 : g.nz - 1;
        bc.tallyScale = wedge ? 2.0 * std::numbers::pi / og.rev.dphi : 1.0;
    }
    auto wall = DeviceOGridBc::make(g, bc);
    if (!wall) return fail(6, "O-grid boundary init failed");
    std::optional<DeviceAxisBc> axis;
    if (rev)
    {
        auto a = DeviceAxisBc::make(g, og.rev);
        if (!a) return fail(6, "axis boundary init failed");
        axis = *a;
    }

    // The real fluid cells: everything but the two ghost rows (and, on a
    // revolved mesh, the ghost columns and the wedge's ghost planes).
    const std::vector<std::uint8_t> fluid = ogridFluidMask(g, og);
    MutHook mutHook = wall->mutHook();
    if (axis)
        mutHook = [h = wall->mutHook(), a = *axis](Real* n, Real* f) {
            h(n, f);
            a.applyMut(n, f);
        };
    if (rans && !dev->enableRans({.model = r.rans,
                                  .wallDist = r.wallDist,
                                  .mutHook = mutHook,
                                  .fluidMask = fluid}))
        return fail(6, "GPU allocation failed (RANS)");
    // LES: the same rows image the SGS mu_t -- odd on the wall row, so the
    // wall face carries the molecular stress exactly (the wall tally then
    // needs no mu_t), and an even far row instead of row 0's centre gradient
    // differenced across the j-wrap.
    if (r.les > 0) dev->setMutHook(mutHook);
    if (steady && !rans && !dev->enableSteady(fluid))
        return fail(6, "GPU allocation failed (steady)");
    if (compensated && !dev->enableCompensation())
        return fail(6, "GPU allocation failed (compensated state)");

    // Both boundary rows ride the stage hook: they carry no solution, so the
    // rhs computed on them is discarded and re-derived every stage. The axis
    // ghosts follow the rows (they copy the rows' ghost cells too).
    // Idempotent: the rows read only real cells (rows 1 and ny-2) and their
    // own column's ghosts; the axis then rewrites its ghost columns (and the
    // wedge its ghost planes) from plane 1's cells alone -- so the hooked
    // state depends on no cell the hook writes. Every state this runner hands
    // step() comes straight from applyBc: after the upload below, after every
    // step (the time march's loop tail, SteadyMarchIo::afterStep), with only
    // reads (tallies, frames, residual, dt) in between. The NK march never
    // calls step().
    auto applyBc = [&](DeviceCompState& s) {
        wall->apply(s);
        if (axis) axis->apply(s);
    };
    dev->setStageHook(applyBc, Idempotent::yes);
    // The wedge's ghost planes k = 0, 2 are kWedgePlanes' rotated copies of
    // plane 1 -- every field and low word, rewritten by the hook before any
    // read -- so the stage kernels form plane 1 alone (2/3 of their cells
    // were discarded work).
    if (wedge)
        dev->setStageRange({std::size_t(g.nx) * g.ny,
                            2 * std::size_t(g.nx) * g.ny});

    bool upOk = ds->upload(ic) == hipSuccess;
    for (int i = 0; i < ds->nT; ++i)
        upOk = upOk && ds->uploadTurb(i, r.icTurb[i]) == hipSuccess;
    if (!upOk) return fail(6, "initial condition upload failed");
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());
    const double u0 = r.ramp > 0 ? 0.0 : r.uInf;
    wall->setFreestream(u0);
    applyBc(*ds);

    // heat: the wall energy flux INTO the body, raw (not a coefficient) --
    // a Nusselt number needs the caller's own kappa and temperature scale.
    // wmax: the acoustic wave speed the dt bound actually saw, put back on a
    // physical length (the mesh's narrowest cell width); the index spacings of
    // a curvilinear grid are not lengths.
    // A rotor adds thrust, shaft torque and their propeller coefficients
    // (rho n^2 D^4 / D^5, n = rev per unit time), exactly runner_comp's
    // columns; a static run keeps the columns it has always had.
    // RANS appends its columns and a steady march its own; a DNS/LES time
    // march keeps exactly the columns it always had.
    std::vector<std::string> cols =
        r.rotor.active()
            ? std::vector<std::string>{"t",    "dt",   "cd",     "cl",
                                       "cs",   "heat", "wmax",   "thrust",
                                       "torque", "ct", "cq"}
            : std::vector<std::string>{"t",  "dt",   "cd",  "cl",
                                       "cs", "heat", "wmax"};
    if (rans)
        cols.insert(cols.end(), {"cdp", "cdf", "mutMax", "yplusMax",
                                 "yplusMean", "floorHits"});
    if (steady)
        cols.insert(cols.end(),
                    {"iter", "resid", "resU", "resE", "resT", "remaining"});
    if (og.revolved) cols.emplace_back("lr");
    if (implicit) cols.insert(cols.end(), {"cfl", "alpha", "rejects", "krylov"});
    if (auto e = out.declareSeries("stats", cols); !e)
        return fail(4, e.error());
    const auto [ctDen, cqDen] = internal::propellerDen(r.rhoInf, r.rotor.omega, r.D);

    const float vortScale =
        float(r.curlMax > 0 ? r.curlMax
                            : (r.signedVorticity ? 8.0 : 15.0) * r.uInf / r.D);
    const float speedMax = float(1.6 * r.uInf);
    const char* vortName = r.signedVorticity ? "vort" : "curl";
    const bool signedWz = r.signedVorticity && g.nz == 1;
    // --temp: T - Tinf on the diverging map. Same convention as runner_comp,
    // and the same clip (internal/frame_scratch.hpp).
    const double TInf = r.pInf / (r.rhoInf * r.tr.Rgas);
    const double machSq = r.uInf * r.uInf * r.rhoInf / (r.tr.gamma * r.pInf);
    const float tempScale = internal::thermalFrameScale(
        TInf, machSq, r.tr.gamma, r.thermal == OGridThermal::Isothermal,
        r.Twall);
    internal::FrameScratch fs;
    auto writeFrame = [&](double t) {
        if (!fs.ensure(g.size(), r.tempFrames ? 3 : 2, true)) return;
        if (r.tempFrames) dev->primitives(*ds);
        fs.centreFromConserved(gg, *ds);
        kOGridCurl<<<launchBlocks(g.size()), blockSize>>>(
            gg, dev->curvTables()->view(), signedWz ? 1 : 0, fs.u().data(),
            fs.v().data(), fs.w().data(), fs.vort().data());
        checkLastKernel("kOGridCurl");
        speedCentred(gg, fs.u(), fs.v(), fs.w(), fs.speed());
        fs.take(0, fs.vort());
        fs.take(1, fs.speed(), r.speedDelta ? r.uInf : 0.0);
        if (r.tempFrames) fs.take(2, dev->T(), TInf);
        const FrameField ff[3]{
            {vortName, fs.channel(0), 0, vortScale, signedWz,
             r.vortAsinh && signedWz ? vortScale / 32.0f : 0.0f},
            internal::speedField(fs.channel(1), r.speedDelta, r.uInf, speedMax),
            {"temp", fs.channel(2), -tempScale, tempScale, true}};
        fs.emit(out, t, {ff, r.tempFrames ? 3u : 2u});
    };

    RunStats st;
    double heat = 0.0;
    double t = 0.0;
    internal::StatsCadence cad{r.statsEvery};
    double dtStab = 0.0;
    int sinceDt = 1 << 30;
    // Per unit span on a z-extruded mesh; a revolved mesh is a whole body.
    const double span = !rev && g.nz > 1 ? g.nz * g.hz : 1.0;
    const auto wall0 = std::chrono::steady_clock::now();

    // --- RANS / steady measurements (docs/rans.md, "dns flow") -------------
    // FP64 volumes of the fluid cells and the mask, for the stop integrals.
    DeviceBuffer<double> vol64;
    DeviceBuffer<std::uint8_t> fluidDev;
    if (rans || steady)
    {
        auto v = DeviceBuffer<double>::make(g.size());
        auto f = DeviceBuffer<std::uint8_t>::make(g.size());
        if (!v || !f || v->upload(mtr->vol.data()) != hipSuccess ||
            f->upload(fluid.data()) != hipSuccess)
            return fail(6, "GPU allocation failed (stop integrals)");
        vol64 = std::move(*v);
        fluidDev = std::move(*f);
    }
    struct Extra
    {
        double cdp = 0, cdf = 0, mutMax = 0, ypMax = 0, ypMean = 0;
        double hits = 0, wake = 0, turb[2] = {0, 0};
    };
    // The exact surface tally. On the axisymmetric wedge the side forces
    // and the torque of the revolved body vanish by symmetry: the scaled sum
    // of one wedge plane would report its lateral force times 2 pi/dphi.
    auto surfaceTally = [&] {
        auto F = wall->surface(*dev, *ds);
        if (wedge) F.fy = F.fz = F.tz = 0.0;
        return F;
    };
    // The real wall faces, derived per face from the per-face tally
    // (surfaceFaces): position, the angle from the UPSTREAM axis (front
    // stagnation 0; signed over the top/bottom on a z-extruded mesh, the
    // polar angle in [0, 180] on a revolved one), cp, cf along the
    // downstream surface tangent, the viscous traction, y+, d1, qw.
    struct WallFace
    {
        int i, k;
        double x[3], thetaDeg, area, cp, cf, tau[3], yplus, d1, qw, fpx;
    };
    const double tallyScale =
        wedge ? 2.0 * std::numbers::pi / og.rev.dphi : 1.0;
    const std::vector<double> d1All =
        r.wallDist.empty() ? std::vector<double>{}
                           : wallFirstCellDistance(g, r.wallDist);
    auto wallFaces = [&]() {
        const auto Fc = wall->surfaceFaces(*dev, *ds);
        const double qd = 0.5 * r.rhoInf * r.uInf * r.uInf;
        std::vector<WallFace> fs;
        for (int k = 0; k < g.nz; ++k)
            for (int i = 0; i < g.nx; ++i)
            {
                if (!realWallFace(og, i, k)) continue;
                const std::size_t q = std::size_t(k) * g.nx + i;
                WallFace w{};
                w.i = i;
                w.k = k;
                // Face midpoint: the mean of its four wall nodes.
                for (double& v : w.x) v = 0;
                for (int dk = 0; dk < 2; ++dk)
                    for (int di = 0; di < 2; ++di)
                    {
                        const auto p = og.mesh.node(i + di, 1, k + dk);
                        for (int a = 0; a < 3; ++a) w.x[a] += 0.25 * p[a];
                    }
                // rho^: the unit direction away from the axis through the
                // centre (z-extruded: +-y, the side of the face).
                double rv[3] = {0, w.x[1] - r.cy, rev ? w.x[2] - r.cz : 0.0};
                const double rho = std::hypot(rv[1], rv[2]);
                const double th =
                    std::atan2(rev ? rho : rv[1], r.cx - w.x[0]);
                for (double& v : rv) v = rho > 0 ? v / rho : 0.0;
                // Downstream tangent: d/d|theta| of (cx - r cos, r sin rho^).
                const double at = std::abs(th);
                double e[3] = {std::sin(at), std::cos(at) * rv[1],
                               std::cos(at) * rv[2]};
                const double en = e[0] * Fc.n[0][q] + e[1] * Fc.n[1][q] +
                                  e[2] * Fc.n[2][q];
                for (int a = 0; a < 3; ++a) e[a] -= en * Fc.n[a][q];
                const double el = std::hypot(e[0], e[1], e[2]);
                w.area = Fc.area[q];
                w.cf = 0;
                for (int a = 0; a < 3; ++a)
                {
                    w.tau[a] = w.area > 0 ? Fc.fv[a][q] / w.area : 0.0;
                    if (el > 0) w.cf += w.tau[a] * e[a] / el;
                }
                w.cf /= qd;
                w.thetaDeg = th * 180.0 / std::numbers::pi;
                w.cp = (Fc.p[q] - r.pInf) / qd;
                w.yplus = Fc.yplus[q];
                w.d1 = d1All.empty() ? 0.0 : d1All[q];
                w.qw = w.area > 0 ? Fc.heat[q] / w.area : 0.0;
                w.fpx = Fc.fp[0][q];
                fs.push_back(w);
            }
        return fs;
    };
    // cd from the exact tally, cdp from the per-face pressure force, the
    // wake deficit W = sum_fluid V (rho u - rho_inf U), the turbulence
    // integrals sum_fluid V t_i, max mu_t/mu_inf and the wall-face y+.
    auto measureExtra = [&](double cd) {
        Extra x;
        const auto fs = wallFaces();
        double fpx = 0, ys = 0;
        for (const auto& w : fs)
        {
            fpx += w.fpx;
            ys += w.yplus;
            x.ypMax = std::max(x.ypMax, w.yplus);
        }
        x.cdp = fpx * tallyScale / (r.qRef * span);
        x.cdf = cd - x.cdp;
        x.ypMean = fs.empty() ? 0.0 : ys / double(fs.size());
        x.wake = internal::reduce(*red, g.size(),
                                  internal::MaskedVolSum{
                                      fluidDev.data(), vol64.data(),
                                      ds->mx.data(), r.rhoInf * r.uInf},
                                  rocprim::plus<double>(), 0.0);
        for (int i = 0; i < ds->nT; ++i)
            x.turb[i] = internal::reduce(
                *red, g.size(),
                internal::MaskedVolSum{fluidDev.data(), vol64.data(),
                                       ds->t[i].data(), 0.0},
                rocprim::plus<double>(), 0.0);
        if (rans)
        {
            dev->updateMut(*ds);
            x.mutMax = internal::reduce(
                *red, g.size(),
                internal::MaskedMax{fluidDev.data(), dev->mut().data(),
                                    1.0 / r.tr.muRef},
                rocprim::maximum<double>(), -HUGE_VAL);
            x.hits = double(dev->floorHits());
        }
        return x;
    };

    // Revolved meshes: the recirculation length L_r on the downstream axis
    // column (i = nTheta; 3D: the k-mean), from the rear stagnation point to
    // the interpolated u = 0 crossing along j, in D; 0 without reversed flow.
    auto recircLength = [&]() {
        if (!rev) return 0.0;
        dev->primitives(*ds);
        std::vector<Real> u(g.size());
        if (dev->u().download(u.data()) != hipSuccess) return 0.0;
        const int i = og.nTheta;
        const int kLo = wedge ? 1 : 0, kHi = wedge ? 1 : g.nz - 1;
        auto at = [&](int j, double& xc) {
            double s = 0;
            xc = 0;
            for (int k = kLo; k <= kHi; ++k)
            {
                s += u[g.idx(i, j, k)];
                for (int dk = 0; dk < 2; ++dk)
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di)
                            xc += og.mesh.node(i + di, j + dj, k + dk)[0];
            }
            const double nk = kHi - kLo + 1;
            xc /= 8.0 * nk;
            return s / nk;
        };
        const double xRear = og.mesh.node(og.nTheta + 1, 1, kLo)[0];
        // The wedge demands reversed flow in the first cell (verbatim). The
        // full revolution starts at the first reversed cell instead: the
        // k-mean of the first cell sits within jitter of 0 at the rear
        // stagnation point and flipped L_r between its value and 0
        // (measured, SA-neg Re 1e3 64 x 48 x 64).
        int jStart = 1;
        if (!wedge)
            while (jStart < g.ny - 2)
            {
                double xs = 0;
                if (at(jStart, xs) < 0) break;
                ++jStart;
            }
        double x0 = 0, u0 = at(jStart, x0);
        if (!(u0 < 0)) return 0.0;
        for (int j = jStart + 1; j <= g.ny - 2; ++j)
        {
            double x1 = 0;
            const double u1 = at(j, x1);
            if (u1 >= 0)
                return (x0 + (x1 - x0) * (-u0) / (u1 - u0) - xRear) / r.D;
            x0 = x1;
            u0 = u1;
        }
        return 0.0;
    };
    // The separation angle: the first sign change of cf from the front
    // (revolved: along the k = kLo meridian; z-extruded: the upper side),
    // linearly interpolated; 0 when the flow never separates.
    auto separationAt = [&](const std::vector<WallFace>& fs, int kMer) {
        const WallFace* prev = nullptr;
        // Faces in increasing theta along one meridian / the upper side.
        std::vector<const WallFace*> line;
        for (const auto& w : fs)
            if ((rev && w.k == kMer) ||
                (!rev && w.k == 0 && w.thetaDeg >= 0))
                line.push_back(&w);
        std::ranges::sort(line, {}, [](const WallFace* w) { return w->thetaDeg; });
        for (const WallFace* w : line)
        {
            if (prev && prev->cf > 0 && w->cf <= 0)
                return prev->thetaDeg + (w->thetaDeg - prev->thetaDeg) *
                                            prev->cf / (prev->cf - w->cf);
            prev = w;
        }
        return 0.0;
    };

    // The surface CSV (--emit surface), once at the end of the run.
    auto writeSurface = [&]() -> std::expected<void, RunError> {
        const auto fs = wallFaces();
        if (auto h = internal::checkDevice("the surface tally", &*red); !h)
            return std::unexpected(h.error());
        if (wedge)
        {
            const double lr = recircLength();
            std::printf("# axisymmetric wall: separation theta_s %.2f deg "
                        "from the front, recirculation L_r %.4f D\n",
                        separationAt(fs, 1), lr);
            std::fflush(stdout);
        }
        else if (rev)
        {
            // The full revolution: theta_s on every meridian (mean, range),
            // L_r on the k-mean of the downstream axis ring, and how far the
            // state is from axisymmetric -- the largest azimuthal spread of
            // u over the real (i, j) rings, in U.
            double sMean = 0, sMin = HUGE_VAL, sMax = -HUGE_VAL;
            for (int k = 0; k < g.nz; ++k)
            {
                const double s = separationAt(fs, k);
                sMean += s / g.nz;
                sMin = std::min(sMin, s);
                sMax = std::max(sMax, s);
            }
            const double lr = recircLength(); // leaves u in dev->u()
            std::vector<Real> u(g.size());
            double spread = 0;
            if (dev->u().download(u.data()) == hipSuccess)
                for (int j = 1; j <= g.ny - 2; ++j)
                    for (int i = 1; i <= og.nTheta; ++i)
                    {
                        double lo = HUGE_VAL, hi = -HUGE_VAL;
                        for (int k = 0; k < g.nz; ++k)
                        {
                            lo = std::min(lo, double(u[g.idx(i, j, k)]));
                            hi = std::max(hi, double(u[g.idx(i, j, k)]));
                        }
                        spread = std::max(spread, hi - lo);
                    }
            std::printf("# revolved wall (full revolution, %d meridians): "
                        "separation theta_s %.2f deg (range %.2f..%.2f), "
                        "recirculation L_r %.4f D (k-mean), azimuthal spread "
                        "max |du| %.3e U\n",
                        g.nz, sMean, sMin, sMax, lr, spread / r.uInf);
            std::fflush(stdout);
        }
        if (!out.wantsSurface()) return {};
        std::vector<std::vector<double>> rows;
        rows.reserve(fs.size());
        for (const auto& w : fs)
            rows.push_back({double(w.i), double(w.k), w.x[0], w.x[1], w.x[2],
                            w.thetaDeg, w.area, w.cp, w.cf, w.tau[0],
                            w.tau[1], w.tau[2], w.yplus, w.d1, w.qw});
        const std::string cs[] = {"i",     "k",    "x",     "y",     "z",
                                  "theta_deg", "area", "cp", "cf", "tau_x",
                                  "tau_y", "tau_z", "yplus", "d1",   "qw"};
        if (auto e = out.writeSurface(r.label.empty() ? std::string("DNS")
                                                      : r.label,
                                      cs, rows);
            !e)
            return fail(4, e.error());
        return {};
    };
    // --emit mesh's end artefact: the final state in FP64 (the compensated
    // pair's value), primitives, the specific RANS scalars, mu_t and d.
    auto writeFinal = [&]() -> std::expected<void, RunError> {
        if (!out.wantsMesh()) return {};
        auto h = CompState::zeros(g);
        if (ds->download(h) != hipSuccess)
            return fail(6, "final state download failed");
        const std::size_t n = g.size();
        std::vector<std::vector<double>> f(6, std::vector<double>(n));
        for (std::size_t c = 0; c < n; ++c)
        {
            const double rho = h.rho[c], u = h.mx[c] / rho, v = h.my[c] / rho,
                         w = h.mz[c] / rho;
            const double p = (r.tr.gamma - 1.0) *
                             (h.E[c] - 0.5 * rho * (u * u + v * v + w * w));
            f[0][c] = rho;
            f[1][c] = u;
            f[2][c] = v;
            f[3][c] = w;
            f[4][c] = p;
            f[5][c] = p / (rho * r.tr.Rgas);
        }
        std::vector<std::string> names = {"rho", "u", "v", "w", "p", "T"};
        const char* tn[2][2] = {{"nutilde", ""}, {"k", "omega"}};
        for (int i = 0; i < ds->nT; ++i)
        {
            std::vector<double> t;
            if (ds->downloadTurb(i, t) != hipSuccess)
                return fail(6, "final scalar download failed");
            for (std::size_t c = 0; c < n; ++c) t[c] /= f[0][c];
            names.emplace_back(tn[sst ? 1 : 0][i]);
            f.push_back(std::move(t));
        }
        std::vector<double> mut(n, 0.0), dd(n, 0.0);
        if (rans)
        {
            dev->updateMut(*ds);
            if (downloadWiden(dev->mut(), mut) != hipSuccess)
                return fail(6, "final mu_t download failed");
        }
        if (!r.wallDist.empty()) dd = r.wallDist;
        names.emplace_back("mut");
        f.push_back(std::move(mut));
        names.emplace_back("d");
        f.push_back(std::move(dd));
        if (auto e = out.writeMeshArtefact("final", encodeFinal(g, names, f));
            !e)
            return fail(4, e.error());
        return {};
    };

    if (steady)
    {
        // The pseudo-time march. Residual norms in freestream units (rho_inf,
        // U_inf, D): mom D/U^2, energy D/U^3, SA nu~ 1/U^2, SST k D/U^3 and
        // ln omega D/U. Reported, never the stop: the stop is on cd, cdf, the
        // wake deficit and the turbulence integrals (decision 13).
        const double tU = r.D / r.uInf; // the time scale D/U
        // The full revolution's two lateral observables (cd + cl, cd + cs).
        const int lateral = rev && !wedge ? 2 : 0;
        double resid = HUGE_VAL, resU = 0, resE = 0, resT = 0;
        DeviceOGridBc::Surface F{};
        Extra x;
        internal::SteadyMarchIo io;
        io.afterStep = [&](DeviceCompState& s) { applyBc(s); };
        io.sample = [&](long) -> std::expected<std::vector<double>, RunError> {
            const ResidualNorms n = dev->residual(*ds, *red);
            F = surfaceTally();
            st.cd = F.fx / (r.qRef * span);
            st.cl = F.fy / (r.qRef * span);
            st.cs = F.fz / (r.qRef * span);
            x = measureExtra(st.cd);
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            resU = n.mom * tU / r.uInf;
            resE = n.E * tU / (r.uInf * r.uInf);
            resT = !rans ? 0.0
                   : sst ? std::max(n.t[0] * tU / (r.uInf * r.uInf),
                                    n.t[1] * tU)
                         : n.t[0] / (r.uInf * r.uInf);
            resid = std::max({resU, resE, resT});
            if (!std::isfinite(resid) || !std::isfinite(st.cd))
                return fail(7, "the residual is not finite (the state "
                               "diverged)");
            std::vector<double> q = {st.cd, x.cdf};
            // The full revolution also watches its lateral forces: cd is
            // even in the m = 1 mode, so a growing m = 1 oscillation leaves
            // cd flat while cl swings (measured: G1 SA-neg Re 1e4 from the
            // wedge certified at 46600 iterations with cl swinging +-1e-3
            // and growing to 1e-2 by 1.5e5). cl and cs have no scale of
            // their own (they are 0 at a symmetric state), so they ride on
            // cd's: the observables cd + cl and cd + cs.
            if (rev && !wedge)
                q.insert(q.end(), {st.cd + st.cl, st.cd + st.cs});
            q.push_back(x.wake);
            for (int i = 0; i < ds->nT; ++i) q.push_back(x.turb[i]);
            return q;
        };
        io.row = [&](const internal::SteadyRow& sr) {
            // wmax: the explicit march's dt bound; the implicit one has none.
            const double wmax =
                sr.dtRef > 0 ? r.cfl / sr.dtRef * mtr->hmin : 0.0;
            std::vector<double> row = {sr.t,  sr.dtRef, st.cd, st.cl,
                                       st.cs, F.heat,   wmax};
            if (rans)
                row.insert(row.end(), {x.cdp, x.cdf, x.mutMax, x.ypMax,
                                       x.ypMean, x.hits});
            row.insert(row.end(), {double(sr.iter), resid, resU, resE, resT,
                                   sr.remaining});
            if (rev) row.push_back(recircLength());
            if (implicit)
                row.insert(row.end(), {sr.cfl, sr.alpha, double(sr.rejects),
                                       double(sr.krylov)});
            out.row("stats", row);
            if (!sr.print) return;
            // The march's own step: dtRef (explicit) or the CFL, step length
            // and Krylov count (implicit).
            const std::string stepText =
                implicit ? std::format("cfl={:.2e} alpha={:.3f} krylov={}",
                                       sr.cfl, sr.alpha, sr.krylov)
                         : std::format("dtRef={:.2e}", sr.dtRef);
            std::printf("iter=%ld cd=%.5f cdf=%.5f cl=%+.2e%s remaining=%.2e "
                        "resid=%.3e (U %.2e E %.2e T %.2e) %s %s\n",
                        sr.iter, st.cd, x.cdf, st.cl,
                        rans ? std::format(" mutMax={:.3g} y+max={:.3g}",
                                           x.mutMax, x.ypMax)
                                   .c_str()
                             : "",
                        sr.remaining, resid, resU, resE, resT,
                        stepText.c_str(), r.label.c_str());
            std::fflush(stdout);
        };
        // The Newton-Krylov march (gpu/nk): lines along j for every real
        // (i, k); the freestream scales of decision 1.
        auto nkMarch = [&](const internal::SteadyMarchParams& p,
                           const internal::SteadyMarchIo& nio)
            -> std::expected<internal::SteadyResult, RunError> {
            DeviceNk::Config nc;
            if (og.revolved)
                nc.lay = {.iLo = 1, .iHi = og.nTheta, .kLo = 1, .kHi = 1,
                          .cyclicI = false, .skipK = false, .wedge = true,
                          .dphi = og.rev.dphi};
            else
                nc.lay = {.iLo = 0, .iHi = g.nx - 1, .kLo = 0, .kHi = g.nz - 1,
                          .cyclicI = true, .skipK = g.nz == 1, .wedge = false,
                          .dphi = 0};
            const double cInf = std::sqrt(r.tr.gamma * r.pInf / r.rhoInf);
            nc.sc.rho = r.rhoInf;
            nc.sc.mom = r.rhoInf * (r.uInf + cInf);
            nc.sc.E = r.rhoInf * cInf * cInf;
            if (sst) {
                nc.sc.t[0] = r.rhoInf * r.uInf * r.uInf;
                nc.sc.omegaInf = r.phiInf[1];
            } else if (rans) {
                nc.sc.t[0] =
                    r.rhoInf * std::max(r.phiInf[0], r.tr.muRef / r.rhoInf);
            }
            nc.model = r.rans;
            nc.gamma = r.tr.gamma;
            nc.Rgas = r.tr.Rgas;
            nc.restart = kNkRestart;
            nc.maxKrylov = kNkMaxKrylov;
            nc.forcing = kNkForcing;
            nc.fdStep = kNkFdStep;
            auto nk = DeviceNk::make(*dev, fluid, nc, applyBc);
            if (!nk) return fail(6, "GPU allocation failed (Newton-Krylov)");
            return internal::nkSteadyMarch(*nk, *ds, *red, p, nio);
        };
        // A revolved mesh's axis cells are legitimately below the absolute
        // 1e-6 guard at L3 (width rho dphi); it keeps the relative one.
        const internal::SteadyMarchParams sp{.cfl = r.cfl,
                                             .dtMargin = r.dtMargin,
                                             .collapse = rev ? 0.0 : 1e-6,
                                             .collapseRel = rev ? 1e-3 : 0.0,
                                             .tol = r.steadyTol,
                                             .dtRecompute = r.dtRecompute,
                                             .cap = r.steadySteps,
                                             .statsEvery = r.statsEvery,
                                             .series = 3 + ds->nT + lateral,
                                             .observables = 2 + lateral};
        auto res = implicit ? nkMarch(sp, io)
                            : internal::explicitSteadyMarch(*dev, *ds, *red,
                                                            sp, io);
        if (!res) return std::unexpected(res.error());
        {
            const char* tn = sst ? "int rho k" : "int rho nu~";
            const char* names3[5] = {"cd", "cdf", "W", tn, "int rho omega"};
            const char* names7[7] = {"cd",     "cdf", "cd+cl", "cd+cs",
                                     "W",      tn,    "int rho omega"};
            std::fputs(internal::steadyFloorReport(
                           res->floors,
                           std::span<const char* const>(
                               lateral ? names7 : names3,
                               3 + ds->nT + lateral))
                           .c_str(),
                       stdout);
            std::fflush(stdout);
        }
        st.steps = res->steps;
        st.tStop = res->t;
        if (auto h = internal::checkDevice("the end of the run", &*red); !h)
            return std::unexpected(h.error());
        if (auto e = writeSurface(); !e) return std::unexpected(e.error());
        if (auto e = writeFinal(); !e) return std::unexpected(e.error());
        finishTiming(st, wall0, out);
        if (!res->converged && !stopRequested())
            return fail(8, std::format("steady state not reached in {} "
                                       "iterations (remaining change {:.3e} "
                                       "> {:.3g})",
                                       res->steps, res->remaining,
                                       r.steadyTol));
        st.converged = res->converged;
        return st;
    }

    while (t < r.tEnd)
    {
        // A stop signal ends the run the SAME way reaching tEnd does: return
        // normally, so every destructor runs -- the stats file is flushed and
        // the shared-memory segment is unlinked. Terminating mid-step instead
        // leaves a segment that outlives its own run, flagged as live.
        if (stopRequested()) break;
        // 1e-6: the compressible collapse threshold, shared with runner_comp
        // because this is the same physics on a different mesh. NOT the
        // channel's 1e-9 -- that one lives in wall units.
        if (auto e = internal::recomputeDt(
                sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-6,
                [&] { return dev->stableDt(*ds, *red, r.cfl); });
            !e)
            return fail(7, e.error());
        const double dt = std::min(r.tEnd - t, dtStab);

        dev->step(*ds, Real(dt));
        if (t < r.ramp)
        {   // Quiet start: boost the frame, then retarget the far field. The
            // boost is an exact Galilean frame change -- no divergence, no
            // vorticity, no acoustic wave -- so the no-slip wall never sees an
            // impulsive start.
            const double uNew = r.uInf * rampFrac(t + dt, r.ramp);
            boostX(gg, *ds, uNew - wall->freestream());
            wall->setFreestream(uNew);
        }
        applyBc(*ds); // leave the state boundary-consistent between steps
        t += dt;
        ++st.steps;

        if (cad.due(st.steps, t, r.tEnd))
        {
            const auto F = surfaceTally();
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            st.cd = F.fx / (r.qRef * span);
            st.cl = F.fy / (r.qRef * span);
            st.cs = F.fz / (r.qRef * span);
            heat = F.heat;
            const double wmax = r.cfl / dtStab * mtr->hmin;
            if (r.rotor.active())
            {
                // z shaft: thrust is the force along it, torque the moment
                // about it, both signed like runner_comp's.
                const double row[] = {t,     dt,   st.cd, st.cl,  st.cs,
                                      heat,  wmax, F.fz,  F.tz,
                                      ctDen > 0 ? F.fz / ctDen : 0.0,
                                      cqDen > 0 ? F.tz / cqDen : 0.0};
                out.row("stats", row);
            }
            else if (rans || rev)
            {
                std::vector<double> row = {t,     dt,   st.cd, st.cl,
                                           st.cs, heat, wmax};
                if (rans)
                {
                    const Extra x = measureExtra(st.cd);
                    row.insert(row.end(), {x.cdp, x.cdf, x.mutMax, x.ypMax,
                                           x.ypMean, x.hits});
                }
                if (rev) row.push_back(recircLength());
                if (auto h = internal::checkDevice("a stats row", &*red); !h)
                    return std::unexpected(h.error());
                out.row("stats", row);
            }
            else
            {
                const double row[] = {t, dt, st.cd, st.cl, st.cs, heat, wmax};
                out.row("stats", row);
            }
            if (cad.printDue(st.steps, t, r.tEnd))
            {
                if (r.rotor.active())
                    std::printf("t=%.2f/%.2f torque=%+.4e cd=%.3f cl=%+.3f "
                                "wmax=%.2f dt=%.2e\n",
                                t, r.tEnd, F.tz, st.cd, st.cl, wmax, dt);
                else if (rans)
                    std::printf("t=%.2f/%.2f cd=%.3f cl=%+.3f cs=%+.3f "
                                "q=%+.3e wmax=%.2f dt=%.2e %s\n",
                                t, r.tEnd, st.cd, st.cl, st.cs, heat, wmax,
                                dt, r.label.c_str());
                else
                    std::printf("t=%.2f/%.2f cd=%.3f cl=%+.3f cs=%+.3f "
                                "q=%+.3e wmax=%.2f dt=%.2e\n",
                                t, r.tEnd, st.cd, st.cl, st.cs, heat, wmax,
                                dt);
                std::fflush(stdout);
                cad.printed(st.steps);
            }
            cad.logged(st.steps);
        }
        if (out.wantsAnyFrame(t)) writeFrame(t);
    }
    if (auto h = internal::checkDevice("the end of the run", &*red); !h)
        return std::unexpected(h.error());
    if (auto e = writeSurface(); !e) return std::unexpected(e.error());
        if (auto e = writeFinal(); !e) return std::unexpected(e.error());
    st.tStop = t;
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
