// dns -- the single executable. Four phases, and the phase order is the
// mechanism (masterplan §5):
//   1. parse    pure string -> RawConfig. No filesystem, no GPU.
//   2. validate pure RawConfig -> Case.   No filesystem, no GPU.
//   3. acquire  probe GPU, load and validate the mesh, then CREATE AND TRUNCATE
//               every output file, then allocate device memory.
//   4. run      the time loop, which by construction can only fail 7 or 8.
//
// This file contains no physics and no __global__: it maps a validated Case
// onto the library's geometry sources, initial conditions and runners.
#include "dns/cli.hpp"
#include "dns/casegeom.hpp"
#include "dns/config.hpp"
#include "dns/flow_ic.hpp"
#include "dns/framemap.hpp"
#include "dns/diagnostics.hpp"
#include "dns/geometry.hpp"
#include "dns/hit_ic.hpp"
#include "dns/interrupt.hpp"
#include "dns/meshsdf.hpp"
#include "dns/ogrid.hpp"
#include "dns/operators.hpp"
#include "dns/output.hpp"
#include "dns/remap.hpp"
#include "dns/scene.hpp"
#include "dns/taylor_green.hpp"
#include "dns/timestepper.hpp"
#include "dns/trimesh.hpp"
#include "dns/units.hpp"
#include "dns/vtk_writer.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/runner_channel.hpp"
#include "dns/gpu/runner_conv.hpp"
#include "dns/gpu/runner_rb.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/runner_comp.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/periodic_frames.hpp"
#include "dns/gpu/runner_incomp.hpp"
#include "dns/gpu/runner_ogrid.hpp"
#include "dns/gpu/runner_periodic.hpp"
#include "dns/gpu/timestepper.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace dns;
using std::numbers::pi;

// Exit codes: 0 ok, 1 CLI, 2 config, 3 input, 4 output, 5 no GPU, 6 resource,
// 7 numerical failure, 8 solver non-convergence.
constexpr int exitInput = 3, exitOutput = 4, exitNoGpu = 5, exitResource = 6;

int die(int code, const std::string& msg)
{
    std::fprintf(stderr, "dns: %s\n", msg.c_str());
    return code;
}

// Case -> sink config. Genuinely app-layer wiring (it maps the validated
// config onto the artefact sink's own config), but it was written out five
// times; the cadences and the PNG coarsening are identical every time and only
// the `sim` metadata differs.
RunOutput::Config outputConfig(const Case& c, const std::string& simJson,
                               bool suppress = false)
{
    RunOutput::Config oc;
    oc.prefix = c.outPrefix;
    oc.kinds = suppress ? std::set<Emit>{Emit::None} : c.emit;
    oc.fdt = c.fdt;
    oc.pdt = c.pdt;
    oc.wdt = c.wdt;
    oc.planesPerAxis = c.planes;
    oc.pngDown = c.pngDown;
    oc.simJson = simJson;
    return oc;
}

#if DNS_WITH_GPU
bool haveGpu(const Case& c)
{
    if (c.backend == Backend::Cpu) return false;
    return gpu::deviceAvailable();
}
#else
bool haveGpu(const Case&) { return false; }
#endif

#if DNS_WITH_GPU

// --- flow: incompressible --------------------------------------------------


// Trilinear prolongation of a MAC velocity field onto a finer grid, by world
// position. Both grids carry one ghost cell, so the interior origin is at
// x = h and coordinates are measured from there: a u-face at index i sits at
// X = (i-1)*h, a cell centre at X = (j-0.5)*h.
//
// Deliberately NOT conservative and NOT divergence-free: this only has to be
// close. The level that receives it masks, projects and re-converges, so the
// fixed point it reaches is that level's own -- prolongation accuracy buys
// transient length, never correctness.
int runFlowIncompressible(const Case& c, CaseGeometry& geo,
                          const VectorField* ic0 = nullptr,
                          VectorField* finalOut = nullptr,
                          bool suppressOutput = false)
{
    const auto& g = c.grid;
    const auto st = sanitizeCut(g, geo.cut);   // before the ring: body cells only
    const bool fringe = c.far == FarField::Fringe || c.far == FarField::Periodic;
    // --body floor: the slab crosses the inlet/outlet planes. Measure the
    // fully-solid interior row count and the cut row's open fraction from the
    // BODY cut now -- closeBoxBoundary is about to zero these very apertures
    // -- so the tunnel BC can zero its Dirichlet/convective values there and
    // keep the prescribed fluxes and global mass balance exact.
    // (measureFloorRows carries the sliver-promotion rule and its measurement;
    // see dns/geometry.hpp.)
    int floorRows = -1;
    double floorFrac = 1.0;
    if (c.bodyKind == BodyKind::Floor)
    {
        const FloorRows f = measureFloorRows(g, geo.cut);
        floorRows = f.rows;
        floorFrac = f.frac;
    }
    if (!fringe) closeBoxBoundary(g, geo.cut); // fringe: the ring stays periodic
    std::printf("# sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed, st.minLambda,
                st.minAperture);

    // Frontal area / reference dynamic pressure: one definition, in the
    // geometry layer (dns/geometry.hpp).
    const ReferenceQ rq = referenceQIncomp(g, geo.cut, 1.0, c.uInf);
    if (!rq.measured)
        return die(exitInput, "the body produced no solid cells (too small?)");
    const double qRef = rq.q;

    // IC: freestream + near-wake kick (dns/flow_ic.hpp). With grid sequencing
    // the coarse level's converged field replaces it.
    auto U = VectorField::zeros(g);
    if (ic0) U = *ic0;
    else incompressibleBodyIC(c, geo, U);

    // Phase 3c: everything the run will write is created and checked now.
    FrameGeometry fg;
    fg.mask.assign(g.size(), 1.0f);
    for (std::size_t q = 0; q < g.size(); ++q) fg.mask[q] = float(geo.cut.lambda[q]);
    fg.ic = g.nx / 2;
    fg.jc = g.ny / 2;
    fg.kc = g.nz == 1 ? 0 : g.nz / 2;
    fg.D = geo.D;
    fg.i0 = 1;
    fg.j0 = 1;
    fg.k0 = g.nz == 1 ? 0 : 1;
    fg.w = g.nx - 2;
    fg.h = g.ny - 2;
    fg.d = g.nz == 1 ? 1 : g.nz - 2;

    const auto oc = outputConfig(
        c, std::format("{{\"app\":\"dns\",\"re\":{}", c.re), suppressOutput);
    auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::IncompRun rr;
    rr.nu = c.nu;
    rr.uInf = c.uInf;
    rr.qRef = qRef;
    rr.D = geo.D;
    rr.cfl = c.cfl;
    rr.tEnd = c.ramp + c.tEnd;
    rr.tol = c.tol;
    rr.maxIters = c.maxIters;
    rr.les = c.cs;
    rr.sgs = c.sgsModel;
    rr.cutFlux = c.cutFlux;
    rr.penalize = c.penalize;
    rr.ramp = c.ramp;
    // The dt reduction is one mandatory host sync per step; --dt-recompute
    // amortizes it exactly as on the compressible path. Defaults to 1 here,
    // so no incompressible trajectory moves unless the user asks.
    rr.dtRecompute = c.dtRecompute;
    rr.vortMax = c.vortMax * c.uInf / geo.D;   // flag is in uInf/D units
    rr.vortAsinh = c.vortAsinh;
    rr.speedDelta = c.speedDelta;
    rr.fringe = fringe;
    rr.open = c.open;
    rr.pot = c.farPot;
    rr.steadyTol = c.steadyTol;
    rr.cx = c.placement.cx;
    rr.cy = c.placement.cy;
    rr.cz = c.placement.cz;
    rr.fringeCells = c.spongeCells;
    rr.fringeAxes = c.fringeAxes;
    rr.floorRows = floorRows;
    rr.floorFrac = floorFrac;

    const auto gg = gpu::GpuGrid::from(g);
    rr.box = c.bodyKind == BodyKind::Scene
                 ? gpu::bodyBox(gg, geo.bbox)
                 : gpu::bodyBox(gg, c.placement.cx, c.placement.cy,
                                c.placement.cz, 0.5 * geo.D);
    if (g.nz == 1 || c.bodyKind == BodyKind::Cylinder || c.bodyKind == BodyKind::Lens
        || c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor
        || (c.bodyKind == BodyKind::Scene && geo.zSpans))
    {   // z-invariant bodies span the whole interior in z.
        rr.box.k0 = gg.kInLo();
        rr.box.k1 = gg.kInHi();
    }
    if (c.bodyKind == BodyKind::Floor)
    {   // the slab spans every interior column (ghost ring is one cell).
        rr.box.i0 = 1;
        rr.box.i1 = gg.nx - 2;
    }

    // Per-object force integration: one tally box per scene object. Only
    // emitted when the boxes are pairwise disjoint -- overlapping boxes would
    // each count the shared surface, so those scenes get the total only.
    if (geo.objectBoxes.size() > 1)
    {
        for (const auto& [nm, bb] : geo.objectBoxes)
        {
            auto b = gpu::bodyBox(gg, bb);
            if (g.nz > 1 && bb[2] == 0.0 && bb[5] >= double(g.nz) * g.hz)
            {
                b.k0 = gg.kInLo();
                b.k1 = gg.kInHi();
            }
            rr.tallies.push_back({nm, b});
        }
        for (std::size_t a = 0; a < rr.tallies.size() && !rr.tallies.empty(); ++a)
            for (std::size_t b = a + 1; b < rr.tallies.size(); ++b)
                if (gpu::overlaps(rr.tallies[a].box, rr.tallies[b].box))
                {
                    std::fprintf(stderr,
                                 "dns: warning: force boxes of '%s' and '%s' "
                                 "overlap; per-object forces suppressed (total "
                                 "cd/cl only)\n",
                                 rr.tallies[a].name.c_str(),
                                 rr.tallies[b].name.c_str());
                    rr.tallies.clear();
                }
    }

    auto res = gpu::runIncompressible(g, geo.cut, geo.sdf, U, rr, *out, finalOut);
    if (!res) return die(res.error().code, res.error().message);
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", res->steps,
                out->frames(), res->msPerStep);
    return 0;
}

// --- flow: compressible ----------------------------------------------------

// --- flow: compressible on a body-fitted O-grid ------------------------------

// The body IS the mesh's inner boundary, so this shares nothing with the
// immersed-body path but the case: no cut geometry, no ghost wall, no far-field
// box. The mesh is built from the SAME sdf the immersed path would have used,
// so a body is described once.
int runFlowOGrid(const Case& c, CaseGeometry& geo)
{
    const auto& g = c.grid;
    const double gamma = 1.4, rhoInf = 1.0, Rgas = 1.0;
    const double pInf = rhoInf * c.uInf * c.uInf / (gamma * c.mach * c.mach);
    const double mu = rhoInf * c.uInf * geo.D / c.re;
    const double Tinf = pInf / (rhoInf * Rgas);
    const Transport gas = Transport::air(mu, Tinf, c.tinfK, c.prandtl, gamma,
                                         Rgas, c.sutherland);

    const OGridSpec spec{.cx = c.placement.cx,
                         .cy = c.placement.cy,
                         .cz = c.placement.cz,
                         .rFar = c.ogridRFar,
                         .beta = c.ogridBeta,
                         .span = c.ogridSpan};
    auto og = buildOGrid(g, geo.sdf, spec);
    if (!og) return die(exitInput, "o-grid: " + og.error());
    std::printf("# o-grid: rWall %.4f..%.4f  h1 %.5g (%.1f cells in D/sqrt(Re))"
                "  growth %.4f  wall-normal dev %.3g deg\n",
                og->rWallMin, og->rWallMax, og->h1,
                og->h1 > 0 ? geo.D / std::sqrt(c.re) / og->h1 : 0.0,
                og->growth, og->normalDevMax * 180.0 / pi);

    // Frames are the index-space raster of the annulus (runner_ogrid.hpp says
    // why), so the render grid IS the logical grid and there is no mask: every
    // cell is fluid.
    FrameGeometry fg;
    fg.mask.assign(g.size(), 1.0f);
    fg.ic = g.nx / 2;
    fg.jc = g.ny / 2;
    fg.kc = g.nz / 2;
    fg.D = geo.D;

    const auto oc = outputConfig(
        c, std::format("{{\"app\":\"dns\",\"re\":{},\"mach\":{}", c.re,
                       c.mach));
    auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::OGridRun rr;
    rr.tr = gas;
    rr.rhoInf = rhoInf;
    rr.uInf = c.uInf;
    rr.pInf = pInf;
    rr.thermal = c.bodyIsothermal ? gpu::OGridThermal::Isothermal
                                  : gpu::OGridThermal::Adiabatic;
    rr.Twall = c.twallRatio * Tinf;
    rr.D = geo.D;
    rr.qRef = 0.5 * rhoInf * c.uInf * c.uInf * geo.D;
    rr.les = c.cs;
    rr.sgs = c.sgsModel;
    rr.cfl = c.cfl;
    rr.tEnd = c.ramp + c.tEnd;
    rr.ramp = c.ramp;
    rr.dtRecompute = c.dtRecompute;
    rr.signedVorticity = c.vortSigned || g.nz == 1;
    rr.curlMax = c.vortMax * c.uInf / geo.D;
    rr.vortAsinh = c.vortAsinh;
    rr.speedDelta = c.speedDelta;
    rr.tempFrames = c.tempFrames;

    // Quiet start: the IC is at REST, exactly consistent with the no-slip wall
    // (--ramp 0 starts it in the freestream instead).
    auto ic = CompState::zeros(g);
    const double u0 = c.ramp > 0 ? 0.0 : c.uInf;
    for (std::size_t q = 0; q < g.size(); ++q)
    {
        ic.rho[q] = rhoInf;
        ic.mx[q] = rhoInf * u0;
        ic.E[q] = pInf / (gamma - 1.0) + 0.5 * rhoInf * u0 * u0;
    }

    const auto res = gpu::runOGrid(g, *og, ic, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    (void)out->setMsPerStep(res->msPerStep);
    std::printf("done: %ld steps, cd=%.4f cl=%.4f\n", res->steps, res->cd,
                res->cl);
    return 0;
}

int runFlowCompressible(const Case& c, CaseGeometry& geo)
{
    const auto& g = c.grid;
    const double gamma = 1.4, rhoInf = 1.0, Rgas = 1.0;
    const double pInf = rhoInf * c.uInf * c.uInf / (gamma * c.mach * c.mach);
    const double mu = rhoInf * c.uInf * geo.D / c.re;
    // Box-unit freestream temperature: T = p / (rho R) = 1 / (gamma M^2). It
    // is the reference the viscosity law and the wall ratio are measured
    // against, and the dimensional anchor --tinf maps onto (dns/transport.hpp).
    const double Tinf = pInf / (rhoInf * Rgas);
    const Transport gas = Transport::air(mu, Tinf, c.tinfK, c.prandtl, gamma,
                                         Rgas, c.sutherland);

    // Frontal area / reference dynamic pressure (used for qRef ONLY, never for
    // the solve -- the compressible body is the sharp ghost wall). One
    // definition, in the geometry layer; see geometry.hpp for why REFERENCE
    // spacings are the correct weights even under --focal (measured).
    const double axisL[3] = {double(g.nx) * g.hx, double(g.ny) * g.hy,
                             double(g.nz) * g.hz};
    AxisRemap3 fRemap;
    {
        Focal1D* f[3] = {&fRemap.fx, &fRemap.fy, &fRemap.fz};
        for (int a = 0; a < 3; ++a)
            *f[a] = c.focalActive && c.focalHi[a] > c.focalLo[a]
                        ? Focal1D::make(axisL[a], c.focalLo[a], c.focalHi[a],
                                        c.focalW[a], c.focalRatio)
                        : Focal1D::identity(axisL[a]);
    }
    ReferenceQ rq;
    if (c.bodyKind != BodyKind::None)
    {
        auto sc = geo.cut;
        (void)sanitizeCut(g, sc);
        rq = referenceQComp(g, sc, rhoInf, c.uInf, geo.D);
    }
    else
        rq.q = 0.5 * rhoInf * c.uInf * c.uInf * geo.D;
    const double qRef = rq.q;

    // IC: tapered freestream + solenoidal wake trip (dns/flow_ic.hpp). The
    // mesh maps go in because the taper and the trip must sit where the cells
    // physically are; both are the exact identity on a uniform grid.
    const TanhStretchY yRemap{double(g.ny) * g.hy, c.stretchBeta};
    auto host = CompState::zeros(g);
    std::vector<float> mask(g.size(), 1.0f);
    compressibleBodyIC(c, geo, fRemap, yRemap, rhoInf, pInf, gamma, host, mask);

    // Display raster for the active mesh maps (dns/framemap.hpp).
    const PhysicalRaster raster = rasterFor(g, fRemap, yRemap);
    const Grid& rg = raster.renderGrid();

    FrameGeometry fg;
    fg.mask = raster.apply(mask);
    fg.ic = rg.nx / 2;
    fg.jc = rg.ny / 2;
    fg.kc = rg.nz / 2;
    fg.D = geo.D;

    RunMeta rmeta = c.meta;
    if (raster.active())
    {
        rmeta.nx = rg.nx;
        rmeta.ny = rg.ny;
        rmeta.nz = rg.nz;
        rmeta.hx = rg.hx;
        rmeta.hy = rg.hy;
        rmeta.hz = rg.hz;
    }

    const auto oc = outputConfig(
        c, std::format("{{\"app\":\"dns\",\"re\":{},\"mach\":{}", c.re, c.mach));
    auto out = RunOutput::make(oc, rmeta, rg, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::CompRun rr;
    rr.raster = &raster;
    rr.gamma = gamma;
    rr.mu = mu;
    rr.tr = gas;
    rr.wallIsothermal = c.bodyIsothermal;
    rr.Twall = c.twallRatio * Tinf;
    rr.nscbcThermalInflow = c.nscbcThermalInflow;
    rr.uInf = c.uInf;
    rr.rhoInf = rhoInf;
    rr.pInf = pInf;
    rr.D = geo.D;
    rr.qRef = qRef;
    rr.cfl = c.cfl;
    rr.tEnd = c.ramp + c.tEnd;
    rr.les = c.cs;
    rr.sgs = c.sgsModel;
    // No compressible --cut-flux (validator-rejected; see gpu/comp.hpp), so
    // --penalize is the only consumer of the aperture geometry here.
    rr.cutGeom = c.penalize > 0 ? &geo.cut : nullptr;
    rr.penalize = c.penalize;
    rr.curlMax = c.vortMax * c.uInf / geo.D;   // flag is in uInf/D units
    rr.vortAsinh = c.vortAsinh;
    rr.speedDelta = c.speedDelta;
    rr.tempFrames = c.tempFrames;
    rr.far = c.far;
    rr.open = c.open;
    rr.spongeCells = c.spongeCells;
    rr.nscbcBeta = c.nscbcBeta;
    rr.nscbcStrip = c.nscbcStrip;
    rr.nscbcPot = c.farPot;
    rr.cx = c.placement.cx;
    rr.cy = c.placement.cy;
    rr.cz = c.placement.cz;
    rr.ramp = c.ramp;
    rr.dtRecompute = c.dtRecompute;
    rr.stretchBeta = c.stretchBeta;
    rr.noslipY = c.noslipY;
    rr.focalActive = c.focalActive;
    rr.focalRatio = c.focalRatio;
    for (int a = 0; a < 3; ++a)
    {
        rr.focalLo[a] = c.focalLo[a];
        rr.focalHi[a] = c.focalHi[a];
        rr.focalW[a] = c.focalW[a];
    }
    // --body floor: fully-solid row count at the x planes, same rule and same
    // implementation as the incompressible tunnel (dns/geometry.hpp).
    if (c.bodyKind == BodyKind::Floor)
        rr.floorRows = measureFloorRows(g, geo.cut).rows;
    // 3D runs report |curl| (sequential map) unless --vort-signed asks for
    // the centre-plane signed wz; native 2D always reports signed wz so the
    // diverging map keeps the sign of the shed vortices.
    rr.signedVorticity = g.nz == 1 || c.vortSigned;
    // The resolved rotor, verbatim: the validator owns the shaft, rate, bins
    // and dilation, and `active()` is false unless --rotor-mach was given.
    rr.rotor = c.rotor;

    auto res = gpu::runCompressible(g, host, geo.sdf,
                                    geo.hasBbox ? &geo.bbox : nullptr, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", res->steps,
                out->frames(), res->msPerStep);
    return 0;
}

#endif // DNS_WITH_GPU


// Grid sequencing: converge on n/2^(L-1), prolong, repeat, finish on n. Each
// level is a normally validated Case -- the ladder reuses validate() rather
// than hand-editing a Case, so no level can skip a check. Only the finest
// level writes output and only its result is reported.
int runFlowSequenced(const Case& c, const RawConfig& raw, const Scene* scene)
{
    const int L = c.seqLevels;
    VectorField carry = VectorField::zeros(c.grid);   // replaced at level 0
    const VectorField* ic = nullptr;
    Grid gPrev = c.grid;
    double hPrev = c.h;

    for (int l = 0; l < L; ++l)
    {
        const int shift = L - 1 - l;
        RawConfig lr = raw;
        lr.n = raw.n >> shift;
        if (raw.given.contains("--ny")) lr.ny = std::max(1, raw.ny >> shift);
        if (raw.given.contains("--nz") && raw.nz > 1)
            lr.nz = std::max(4, raw.nz >> shift);
        lr.seq = 1;   // the level itself is a plain steady solve

        auto lc = validate(lr);
        if (!lc) { std::fputs(format(lc.error()).c_str(), stderr); return lc.error().code; }

        auto geo = buildCaseGeometry(*lc, scene);
        if (!geo) return die(exitInput, geo.error());

        VectorField prolonged = VectorField::zeros(lc->grid);
        if (l > 0)
        {
            prolonged = prolongMac(gPrev, hPrev, carry, lc->grid, lc->h);
            ic = &prolonged;
        }

        const bool last = l == L - 1;
        std::printf("# seq level %d/%d: n=%d h=%.6g (%.1f cells/D)%s\n", l + 1, L,
                    lc->n, lc->h, lc->placement.D / lc->h,
                    last ? "  [final]" : "");
        std::fflush(stdout);

        VectorField fin = VectorField::zeros(lc->grid);
        const int rc = runFlowIncompressible(*lc, *geo, ic, last ? nullptr : &fin,
                                             !last);
        if (rc != 0) return rc;
        if (!last)
        {
            carry = std::move(fin);
            gPrev = lc->grid;
            hPrev = lc->h;
        }
    }
    return 0;
}

int runFlow(const Case& c, const RawConfig& raw, const Scene* scene)
{
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns flow is GPU-only)");
#if DNS_WITH_GPU
    if (c.seqLevels > 1) return runFlowSequenced(c, raw, scene);
    const auto t0 = std::chrono::steady_clock::now();
    auto geo = buildCaseGeometry(c, scene);
    if (!geo) return die(exitInput, geo.error());
    std::printf("# geometry built in %.2f s (D = %.4f, %.1f cells)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count(),
                geo->D, geo->D / c.h);
    if (c.ogrid) return runFlowOGrid(c, *geo);
    return c.physics == Physics::Compressible ? runFlowCompressible(c, *geo)
                                              : runFlowIncompressible(c, *geo);
#else
    return exitNoGpu;
#endif
}

// --- channel ---------------------------------------------------------------

int runChannel(const Case& c)
{
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns channel is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const double gamma = 1.4, delta = 1.0, rho0 = 1.0;
    const ChannelStretchY remap{double(g.ny) * g.hy, g.ny, c.stretchBeta,
                                delta};

    // IC: Poiseuille + solenoidal curl(psi) trip (dns/flow_ic.hpp).
    auto host = CompState::zeros(g);
    std::vector<float> mask(g.size(), 1.0f);
    channelIC(c, remap, delta, rho0, gamma, host, mask);

    FrameGeometry fg;
    fg.mask = std::move(mask);
    fg.ic = g.nx / 2;
    fg.jc = g.ny / 2;
    fg.kc = g.nz / 2;
    fg.D = 2 * delta;

    const auto oc = outputConfig(
        c, std::format("{{\"app\":\"dns\",\"retau\":{},\"mach\":{}",
                       c.reTau, c.mach));
    auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::ChannelRun rr;
    rr.mu = rho0 * c.nu;
    rr.reTau = c.reTau;
    rr.ubEst = c.ubEst;
    rr.delta = delta;
    rr.pRef = c.pRef;
    rr.Twall = c.pRef; // rho = R = 1
    rr.isothermal = c.wallIso;
    rr.stretchBeta = c.stretchBeta;
    rr.cfl = c.cfl;
    rr.tEnd = c.tEnd;
    rr.dtRecompute = c.dtRecompute;
    rr.profileDt = c.profileDt;

    auto res = gpu::runChannel(g, host, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", res->steps,
                out->frames(), res->msPerStep);
    return 0;
#else
    return exitNoGpu;
#endif
}


// --- conv ------------------------------------------------------------------

int runConv(const Case& c)
{
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns conv is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const double gamma = 1.4;

    // Boussinesq branch: a different model, not a cheaper solver for the same
    // one (gpu/boussinesq.hpp). It shares the case derivation and the artefact
    // sink and nothing else.
    if (c.physics == Physics::Incompressible)
    {
        auto U = VectorField::zeros(g);
        auto T = zeros(g);
        rbIC(c, U, T);

        // Ghost cells are not fluid and must not be drawn: the two wall rows
        // always, and the lateral columns once those faces are walls. (The
        // compressible twin gets the same mask from convIC.)
        FrameGeometry fg;
        fg.mask.assign(g.size(), 1.0f);
        const int ox = c.closedSides ? 1 : 0;
        const int oz = c.closedSides && g.nz > 1 ? 1 : 0;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                    if (j < 1 || j > g.ny - 2 || i < ox || i >= g.nx - ox
                        || k < oz || k >= g.nz - oz)
                        fg.mask[g.idx(i, j, k)] = 0.0f;
        fg.ic = g.nx / 2;
        fg.jc = g.ny / 2;
        fg.kc = g.nz / 2;
        fg.D = c.convH;

        const auto oc = outputConfig(
            c, std::format("{{\"app\":\"dns\",\"rayleigh\":{},\"prandtl\":{}",
                           c.rayleigh, c.prandtl));
        auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
        if (!out) return die(exitOutput, out.error());

        gpu::RbRun rr;
        rr.rayleigh = c.rayleigh;
        rr.prandtl = c.prandtl;
        rr.nu = c.nu;
        rr.kappa = c.kappaDiff;
        rr.gAlpha = c.gravity;
        rr.H = c.convH;
        rr.Tbot = c.tBot;
        rr.Ttop = c.tTop;
        rr.Tref = 0.0;
        rr.patch = c.hotPatch;
        rr.closedSides = c.closedSides;
        rr.cfl = c.cfl;
        rr.tEnd = c.tEnd;
        rr.profileDt = c.profileDt;

        auto res = gpu::runRb(g, U, T, rr, *out);
        if (!res) return die(res.error().code, res.error().message);
        std::printf("done: %ld steps, %d frames, %.2f ms/step\n", res->steps,
                    out->frames(), res->msPerStep);
        return 0;
    }

    auto host = CompState::zeros(g);
    std::vector<float> mask(g.size(), 1.0f);
    convIC(c, gamma, host, mask);

    FrameGeometry fg;
    fg.mask = std::move(mask);
    fg.ic = g.nx / 2;
    fg.jc = g.ny / 2;
    fg.kc = g.nz / 2;
    fg.D = c.convH;

    const auto oc = outputConfig(
        c, std::format("{{\"app\":\"dns\",\"rayleigh\":{},\"prandtl\":{}",
                       c.rayleigh, c.prandtl));
    auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::ConvRun rr;
    rr.gamma = gamma;
    rr.mu = c.nu;            // rho_ref = 1
    rr.Pr = c.prandtl;
    rr.tinfK = c.tinfK;
    rr.sutherland = c.sutherland;
    rr.gravity = c.gravity;
    rr.H = c.convH;
    rr.Tbot = c.tBot;
    rr.Ttop = c.tTop;
    rr.patch = c.hotPatch;
    rr.closedSides = c.closedSides;
    rr.rayleigh = c.rayleigh;
    rr.uFf = c.uFreeFall;
    rr.cfl = c.cfl;
    rr.tEnd = c.tEnd;
    rr.dtRecompute = c.dtRecompute;
    rr.profileDt = c.profileDt;

    auto res = gpu::runConv(g, host, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", res->steps,
                out->frames(), res->msPerStep);
    return 0;
#else
    return exitNoGpu;
#endif
}


// --- tgv -------------------------------------------------------------------

int runTgv(const Case& c)
{
    const auto& g = c.grid;
    const double nu = c.nu;

    if (g.nz == 1)
    {
        // The 2D analytic case: an exact solution exists, so this is a
        // convergence check rather than a benchmark. CPU reference stepper.
        auto U = VectorField::zeros(g);
        taylorGreen(g, nu, 0.0, U);
        TimeStepper ts(g, {.nu = nu, .cfl = 0.5});
        const double ke0 = kineticEnergy(g, U);
        double t = 0.0;
        long steps = 0;
        while (t < c.tEnd)
        {
            const double dt = ts.suggestDt(U, c.tEnd - t);
            ts.step(U, dt);
            t += dt;
            ++steps;
        }
        const double err = taylorGreenError(g, nu, t, U);
        const double ke = kineticEnergy(g, U);
        std::printf("n=%d nu=%g T=%g steps=%ld\n", g.nx, nu, t, steps);
        if (c.emit.contains(Emit::Error))
        {
            std::printf("max error vs analytic: %.3e\n", err);
            std::printf("KE ratio: %.6f (analytic %.6f)\n", ke / ke0,
                        std::exp(-4.0 * nu * t));
        }
        if (c.emit.contains(Emit::Vtk))
        {
            if (auto r = writeVtk(g, U, c.outPrefix + ".vtk"); !r)
                return die(exitOutput, r.error());
        }
        return 0;
    }

    // 3D Re1600 benchmark: a dissipation time series on stdout, the format the
    // spectral reference curves are compared against.
    const double vol = g.cellVolume() * double(g.size());
    const StepParams sp{nu, c.cfl};
    auto U = VectorField::zeros(g);
    taylorGreen3D(g, U);

    double kPrev = 0.0, tPrev = 0.0;
    auto emit = [&](double t, double k, double eps) {
        const double dkdt = t > tPrev ? -(k - kPrev) / (t - tPrev) : 0.0;
        std::printf("%.6f,%.10e,%.10e,%.10e\n", t, k, dkdt, eps);
        kPrev = k;
        tPrev = t;
    };
    const double outEvery = 0.1;

#if DNS_WITH_GPU
    if (haveGpu(c))
    {
        std::optional<RunOutput> slices;
        std::optional<gpu::PeriodicFrames> frames;
        if (c.emit.contains(Emit::Slices))
        {
            auto oc = outputConfig(c, "");
            oc.kinds = {Emit::Slices};
            auto o = RunOutput::make(oc, c.meta, g,
                                     gpu::periodicFrameGeometry(g));
            if (!o) return die(exitOutput, o.error());
            slices = std::move(*o);
            auto f = gpu::PeriodicFrames::make(g, *slices);
            if (!f) return die(exitOutput, f.error());
            frames = std::move(*f);
        }
        std::printf("# 3D TGV n=%d Re=%g cfl=%g backend=%s\n", g.nx, c.re, c.cfl,
                    gpu::backendName);
        gpu::TgvRun tr;
        tr.nu = nu;
        tr.cfl = sp.cfl;
        tr.tEnd = c.tEnd;
        tr.outEvery = outEvery;
        tr.volume = vol;
        auto st = gpu::runTgv3d(g, U, tr, frames ? &*frames : nullptr);
        if (!st) return die(st.error().code, st.error().message);
        std::fprintf(stderr, "done: %ld steps (%s)\n", st->steps, gpu::backendName);
        return 0;
    }
#endif
    TimeStepper ts(g, sp);
    std::printf("# 3D TGV n=%d Re=%g cfl=%g backend=cpu\n", g.nx, c.re, c.cfl);
    std::printf("t,K,dKdt,epsVisc\n");
    double t = 0.0, nextOut = 0.0;
    long steps = 0;
    auto probe = [&] {
        emit(t, kineticEnergy(g, U) / vol, nu * velocityGradientSq(g, U));
    };
    while (t < c.tEnd)
    {
        if (t >= nextOut) { probe(); nextOut += outEvery; }
        const double dt = ts.suggestDt(U, c.tEnd - t);
        ts.step(U, dt);
        t += dt;
        ++steps;
    }
    probe();
    std::fprintf(stderr, "done: %ld steps (cpu)\n", steps);
    return 0;
}

// --- hit -------------------------------------------------------------------

int runHit(const Case& c)
{
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns hit is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const auto d = HitDesign::make(g.nx, c.nu, c.urms);
    const double nu = d.nuBox(), epsF = d.epsBox(), tE = d.tEddyBox();
    std::printf("# HIT n=%d  SI: u'=%g m/s nu=%g  L=%.3e m  eps=%.3e W/kg  "
                "eta=%.3e m  Re_lambda=%.1f  T_e=%.3e s\n",
                g.nx, c.urms, c.nu, d.lBoxSI, d.epsSI, d.etaSI, d.reLambda,
                d.tEddySI);
    std::printf("# box units: nu=%.6e eps_f=%.6e T_e=%.4f  spin=%g Te stats=%g Te\n",
                nu, epsF, tE, c.spin, c.stats);

    auto U = VectorField::zeros(g);
    hitInitialCondition(g, 1.0, c.seed, U);

    const LinearForcing forcing{epsF, 1.5 / 5.0}; // kFloor = k_stationary/5
    const double tEnd = (c.spin + c.stats) * tE, tSpin = c.spin * tE;
    // Phase 3c: ONE sink for everything hit writes -- stats, slices, spectra
    // and snapshots. The spectra and snapshots used to go straight to
    // std::ofstream behind RunOutput's back, so a failed write was silent and
    // a full destination surfaced hours into a run instead of in the first
    // second.
    const auto oc = outputConfig(c, "");
    auto outOr = RunOutput::make(oc, c.meta, g, gpu::periodicFrameGeometry(g));
    if (!outOr) return die(exitOutput, outOr.error());
    RunOutput out = std::move(*outOr);
    const std::string statsCols[] = {"t",   "dt",       "A",        "kPrime",
                                     "uPrime", "epsVisc", "epsInj",  "eta",
                                     "reLambda", "hOverEta", "cfl", "maxDiv"};
    out.declareSeries("stats", statsCols);

    std::optional<gpu::PeriodicFrames> frames;
    if (c.emit.contains(Emit::Slices))
    {
        auto f = gpu::PeriodicFrames::make(g, out);
        if (!f) return die(exitOutput, f.error());
        frames = std::move(*f);
    }

    gpu::HitRun hr;
    hr.nu = nu;
    hr.epsF = epsF;
    hr.kFloor = forcing.kFloor;
    hr.tEnd = tEnd;
    hr.tSpin = tSpin;
    hr.tEddy = tE;
    hr.spectrum = c.emit.contains(Emit::Spectrum);
    hr.snapshot = c.emit.contains(Emit::Snapshot);
    hr.lBoxSI = d.lBoxSI;
    hr.timeToSI = [&d](double t) { return d.units().timeToSI(t); };
    gpu::RunStats st;
    auto avg = gpu::runHit(g, U, hr, out, frames ? &*frames : nullptr, st);
    if (!avg) return die(avg.error().code, avg.error().message);
    const long steps = st.steps;

    const double uPrime = std::sqrt(2.0 * avg->kPrime / 3.0);
    std::printf("# %ld steps.  Stationary averages over %.1f Te:\n", steps, c.stats);
    std::printf("#   eps_visc/eps_f = %.4f   u'/target = %.4f   Re_lambda = %.1f "
                "(design %.1f)\n",
                avg->epsVisc / epsF, uPrime, avg->reLambda, d.reLambda);
    std::printf("#   SI: u' = %.2f m/s  eps = %.4e W/kg\n",
                d.units().velocityToSI(uPrime), d.units().epsToSI(avg->epsVisc));
    return 0;
#else
    return exitNoGpu;
#endif
}

// --- bench -----------------------------------------------------------------

#if DNS_WITH_GPU
// Uniform-density Taylor-Green as a COMPRESSIBLE state: rho = 1, the same
// velocity field the incompressible benchmark uses, and a uniform pressure set
// by the Mach number (p = 1/(gamma M^2) with uInf = rho = 1). Not an exact
// solution of anything -- a benchmark needs a representative field, not a
// converged one -- but it is smooth, fills every cell and exercises the full
// convective + viscous flux.
CompState benchCompState(const Grid& g, const VectorField& U, double gamma,
                         double mach)
{
    auto s = CompState::zeros(g);
    const double p = 1.0 / (gamma * mach * mach);
    for (std::size_t i = 0; i < g.size(); ++i) {
        const double u = U.u[i], v = U.v[i], w = U.w[i];
        s.rho[i] = 1.0;
        s.mx[i] = u;
        s.my[i] = v;
        s.mz[i] = w;
        s.E[i] = p / (gamma - 1.0) + 0.5 * (u * u + v * v + w * w);
    }
    return s;
}

// Device memory this process holds, from the free-memory delta across the
// allocation. hipMemGetInfo's absolute "in use" also counts the display server
// and every other process on the card, which made the reported figure both
// wrong and machine-dependent.
double benchVramGiB(std::size_t freeBefore)
{
    std::size_t freeB = 0, totalB = 0;
    (void)hipMemGetInfo(&freeB, &totalB);
    return freeBefore > freeB ? double(freeBefore - freeB) / (1 << 30) : 0.0;
}

void benchReport(const Grid& g, const char* what, double ms, double vramGiB)
{
    std::printf("n=%d  %s  %.2f ms/step  (%.1f Mcells, %.1f Mcell-updates/s, "
                "%.2f GiB VRAM)\n",
                g.nx, what, ms, g.size() / 1e6, g.size() / (ms * 1e3), vramGiB);
}
#endif

int runBench(const Case& c)
{
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns bench is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    auto U = VectorField::zeros(g);
    taylorGreen3D(g, U);
    const double dt = 0.2 * (2 * pi / g.nx); // fixed benchmark dt, CFL-ish
    std::size_t freeBefore = 0, totalB = 0;
    (void)hipMemGetInfo(&freeBefore, &totalB);

    if (c.physics == Physics::Compressible)
    {
        constexpr double gamma = 1.4;
        auto D = gpu::DeviceComp::make(g, gamma, c.nu, c.cs, c.sgsModel);
        auto S = gpu::DeviceCompState::make(g.size());
        if (!D || !S) return die(exitResource, "GPU init failed");
        if (S->upload(benchCompState(g, U, gamma, c.mach)) != hipSuccess)
            return die(exitResource, "GPU upload failed");
        const double vram = benchVramGiB(freeBefore);
        for (int s = 0; s < 3; ++s) D->step(*S, gpu::Real(dt)); // warm-up
        (void)hipDeviceSynchronize();

        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < c.steps; ++s) D->step(*S, gpu::Real(dt));
        (void)hipDeviceSynchronize();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count()
                          / c.steps;
        benchReport(g, c.cs > 0 ? "compressible LES" : "compressible", ms, vram);
        return 0;
    }

    auto ts = gpu::TimeStepper::make(g, {c.nu, 0.9});
    auto D = gpu::DeviceField::make(g.size());
    if (!ts || !D) return die(exitResource, "GPU init failed");
    (void)gpu::uploadField(U, *D);
    ts->setForcing({0.1, {0, 0, 0}}); // exercise the forcing path too
    const double vram = benchVramGiB(freeBefore);

    for (int s = 0; s < 3; ++s) ts->step(*D, dt); // warm-up (incl. FFT tuning)
    (void)hipDeviceSynchronize();

    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < c.steps; ++s) ts->step(*D, dt);
    (void)hipDeviceSynchronize();
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count()
                      / c.steps;
    benchReport(g, "incompressible", ms, vram);
    return 0;
#else
    return exitNoGpu;
#endif
}

} // namespace

int main(int argc, char** argv)
{
    // Before anything allocates or opens a file: a stop signal must unwind
    // through the destructors, not terminate mid-step.
    dns::installInterruptHandler();
    // Phase 0.5: --config FILE.toml is an alternative SPELLING of the command
    // line -- it expands to the exact tokens parse() takes, spliced in before
    // the explicit flags so the command line wins where an option repeats.
    // Reading the file here keeps parse() itself pure.
    std::vector<std::string> store;
    std::vector<char*> ptrs;
    int cfg = -1;
    for (int i = 1; i < argc; ++i)
        if (std::string_view(argv[i]) == "--config") { cfg = i; break; }
    if (cfg >= 0)
    {
        if (cfg + 1 >= argc)
            return die(1, "--config requires a file path");
        const std::string path = argv[cfg + 1];
        std::ifstream f(path, std::ios::binary);
        if (!f) return die(exitInput, "cannot read config file '" + path + "'");
        std::ostringstream ss;
        ss << f.rdbuf();
        auto ct = cli::configTokens(ss.str());
        if (!ct) return die(exitInput, ct.error() + " ('" + path + "')");
        const bool subFromCli = cfg != 1;
        const std::string sub = subFromCli ? argv[1] : ct->subcommand;
        if (sub.empty())
            return die(1, "no subcommand: give one on the command line or set "
                          "`subcommand` in the config file");
        if (subFromCli && !ct->subcommand.empty() && ct->subcommand != sub)
            return die(1, "config file subcommand '" + ct->subcommand
                          + "' conflicts with '" + sub + "'");
        store.emplace_back(argv[0]);
        store.push_back(sub);
        for (auto& a : ct->args) store.push_back(std::move(a));
        for (int i = subFromCli ? 2 : 1; i < argc; ++i)
            if (i != cfg && i != cfg + 1) store.emplace_back(argv[i]);
        ptrs.reserve(store.size());
        for (auto& s : store) ptrs.push_back(s.data());
        argc = int(ptrs.size());
        argv = ptrs.data();
    }

    // Phase 1: parse.
    auto raw = cli::parse(argc, argv);
    if (!raw)
    {
        std::fputs(raw.error().message.c_str(),
                   raw.error().helpRequested ? stdout : stderr);
        return raw.error().helpRequested ? 0 : 1;
    }

    // Phase 1.5: --scene names a user INPUT file whose model block feeds the
    // validator, so it is loaded here -- after parse, before validate. A bad
    // scene file is an input error (3); a model conflict is a config error (2).
    std::optional<Scene> scene;
    if (!raw->scene.empty())
    {
        auto s = loadScene(raw->scene);
        if (!s) return die(exitInput, s.error());
        if (auto m = applySceneModel(*s, *raw); !m)
        {
            std::fputs(format(m.error()).c_str(), stderr);
            return m.error().code;
        }
        scene = std::move(*s);
    }

    // Phase 2: validate.
    auto c = validate(*raw);
    if (!c)
    {
        std::fputs(format(c.error()).c_str(), stderr);
        return c.error().code;
    }
    // validate() is pure: it collects advisories rather than printing them.
    for (const auto& w : c->warnings)
        std::fprintf(stderr, "dns: warning: %s\n", w.c_str());
    if (c->dryRun)
    {
        std::fputs(c->describe().c_str(), stdout);
        if (scene) std::fputs(scene->describe().c_str(), stdout);
        return 0;
    }

    // A stop that arrived during parse/validate is honoured HERE, before
    // phase 3 opens a file or touches the GPU. (A signal during the GPU
    // acquisition inside a runner is still only seen when its time loop
    // starts -- see docs/ui.md.)
    if (stopRequested())
    {
        std::fprintf(stderr, "dns: %s before the run started\n", stopReason());
        return 0;
    }

    // Phases 3 and 4.
    switch (c->sub)
    {
    case Subcommand::Flow: return runFlow(*c, *raw, scene ? &*scene : nullptr);
    case Subcommand::Tgv: return runTgv(*c);
    case Subcommand::Hit: return runHit(*c);
    case Subcommand::Bench: return runBench(*c);
    case Subcommand::Channel: return runChannel(*c);
    case Subcommand::Conv: return runConv(*c);
    }
    return 1;
}
