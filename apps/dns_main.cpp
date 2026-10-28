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
#include "dns/flame1d.hpp"
#include "dns/flow_ic.hpp"
#include "dns/framemap.hpp"
#include "dns/diagnostics.hpp"
#include "dns/facemesh.hpp"
#include "dns/geometry.hpp"
#include "dns/hit_ic.hpp"
#include "dns/interrupt.hpp"
#include "dns/meshio.hpp"
#include "dns/meshsdf.hpp"
#include "dns/ogrid.hpp"
#include "dns/operators.hpp"
#include "dns/output.hpp"
#include "dns/prop/bemt.hpp"
#include "dns/prop/liftline.hpp"
#include "dns/rans_label.hpp"
#include "dns/remap.hpp"
#include "dns/scene.hpp"
#include "dns/taylor_green.hpp"
#include "dns/timestepper.hpp"
#include "dns/transport.hpp"
#include "dns/trimesh.hpp"
#include "dns/units.hpp"
#include "dns/vtk_writer.hpp"
#include "dns/walldist.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/runner_channel.hpp"
#include "dns/gpu/runner_conv.hpp"
#include "dns/gpu/runner_flame.hpp"
#include "dns/gpu/runner_chamber.hpp"
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
#include <iterator>
#include <limits>
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
constexpr int exitInput = 3, exitOutput = 4, exitNoGpu = 5, exitResource = 6,
              exitNumerical = 7;

int die(int code, const std::string& msg)
{
    std::fprintf(stderr, "dns: %s\n", msg.c_str());
    return code;
}

// The end of every run: the sink's checked close (a write failure anywhere in
// the run is exit 4 here, never a silent 0), then the line every run ends with.
// Returns the process exit code, so a runner's tail is `return reportDone(...)`.
template <class Stats>
int reportDone(const Stats& st, RunOutput& out)
{
    if (auto e = out.finish(); !e) return die(exitOutput, e.error());
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", st.steps,
                out.frames(), st.msPerStep);
    return 0;
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

// --- flow: compressible freestream -----------------------------------------

// Box units of both compressible body runs (Cartesian and O-grid): rho = u =
// R = 1 and p from the Mach number.
constexpr double kRhoInf = 1.0;

struct Freestream
{
    double pInf, mu, Tinf;
    Transport gas;
};

// The freestream state and gas of a compressible body run on diameter D. Tinf
// is the box-unit temperature T = p / (rho R) = 1 / (gamma M^2): the reference
// the viscosity law and the wall ratio are measured against, and the
// dimensional anchor --tinf maps onto (dns/transport.hpp).
Freestream compFreestream(const Case& c, double D)
{
    const double gamma = kGammaAir, rhoInf = kRhoInf, Rgas = kRgasBox;
    const double pInf = rhoInf * c.uInf * c.uInf / (gamma * c.mach * c.mach);
    const double mu = rhoInf * c.uInf * D / c.re;
    const double Tinf = pInf / (rhoInf * Rgas);
    return {pInf, mu, Tinf,
            Transport::air(mu, Tinf, c.tinfK, c.prandtl, gamma, Rgas,
                           c.sutherland)};
}

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
    // The written window is the interior: the ghost ring never reaches a frame.
    const auto gg = gpu::GpuGrid::from(g);
    FrameGeometry fg = FrameGeometry::centred(g, geo.D);
    fg.mask.assign(g.size(), 1.0f);
    for (std::size_t q = 0; q < g.size(); ++q) fg.mask[q] = float(geo.cut.lambda[q]);
    fg.i0 = 1;
    fg.j0 = 1;
    fg.k0 = gg.kInLo();
    fg.w = g.nx - 2;
    fg.h = g.ny - 2;
    fg.d = gg.kInCount();

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

    rr.box = c.bodyKind == BodyKind::Scene
                 ? gpu::bodyBox(gg, geo.bbox)
                 : gpu::bodyBox(gg, c.placement.cx, c.placement.cy,
                                c.placement.cz, 0.5 * geo.D);
    if (g.nz == 1 || geo.zSpans)
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
    return reportDone(*res, *out);
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
    const double gamma = kGammaAir, rhoInf = kRhoInf;
    const Freestream fs = compFreestream(c, geo.D);
    const double pInf = fs.pInf, Tinf = fs.Tinf;
    const Transport& gas = fs.gas;

    const OGridSpec spec{.cx = c.placement.cx,
                         .cy = c.placement.cy,
                         .cz = c.placement.cz,
                         .rFar = c.ogridRFar,
                         .beta = c.ogridBeta,
                         .span = c.ogridSpan};
    const RevolvedSpec rev =
        c.ogridNphi > 0
            ? RevolvedSpec{.wedge = false, .dphi = 2.0 * pi / c.ogridNphi,
                           .nphi = c.ogridNphi}
            : RevolvedSpec{.wedge = true, .dphi = c.ogridWedgeDeg * pi / 180.0};
    auto og = c.ogridRevolved ? buildRevolvedOGrid(g, geo.sdf, spec, rev)
                              : buildOGrid(g, geo.sdf, spec);
    if (!og) return die(exitInput, "o-grid: " + og.error());
    std::printf("# o-grid: rWall %.4f..%.4f  h1 %.5g (%.1f cells in D/sqrt(Re))"
                "  growth %.4f  wall-normal dev %.3g deg\n",
                og->rWallMin, og->rWallMax, og->h1,
                og->h1 > 0 ? geo.D / std::sqrt(c.re) / og->h1 : 0.0,
                og->growth, og->normalDevMax * 180.0 / pi);

    // --init-from (full revolution): the converged wedge, read and checked
    // against this mesh before any output exists.
    std::optional<WedgeStart> wedgeStart;
    if (!c.ogridInitFrom.empty())
    {
        auto slurp = [](const std::string& p) -> std::optional<std::vector<char>> {
            std::ifstream f(p, std::ios::binary);
            if (!f) return std::nullopt;
            return std::vector<char>(std::istreambuf_iterator<char>(f), {});
        };
        const std::string mp = c.ogridInitFrom + "_mesh.bin",
                          fp = c.ogridInitFrom + "_final.bin";
        const auto mb = slurp(mp), fb = slurp(fp);
        if (!mb || !fb)
            return die(exitInput, std::format("--init-from: cannot read {} (a "
                                              "wedge run with --emit mesh "
                                              "writes it)",
                                              !mb ? mp : fp));
        std::vector<std::string> names;
        if (c.rans == RansModel::SaNeg) names = {"nutilde"};
        else if (ransSst(c.rans)) names = {"k", "omega"};
        const double centre[3] = {c.placement.cx, c.placement.cy,
                                  c.placement.cz};
        auto ws = revolveWedgeState(*mb, *fb, *og, centre, names, pInf);
        if (!ws)
            return die(exitInput, "--init-from " + c.ogridInitFrom + ": " +
                                      ws.error());
        wedgeStart = std::move(*ws);
        std::printf("# initial state: the wedge %s revolved over %d planes\n",
                    c.ogridInitFrom.c_str(), g.nz);
    }

    // Frames are the index-space raster of the annulus (runner_ogrid.hpp says
    // why), so the render grid IS the logical grid and there is no mask: every
    // cell is fluid.
    FrameGeometry fg = FrameGeometry::centred(g, geo.D);
    fg.mask.assign(g.size(), 1.0f);

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
    // The resolved rotor, verbatim (validator-owned); on the O-grid only
    // --rotor-frame mrf reaches here, and active() is false without
    // --rotor-mach.
    rr.rotor = c.rotor;
    rr.rotorMrf = c.rotorMrf;
    rr.D = geo.D;
    // Per unit span on a z-extruded mesh (the runner divides by the span);
    // the frontal disc pi D^2/4 on a revolved one (a whole body).
    rr.qRef = 0.5 * rhoInf * c.uInf * c.uInf *
              (c.ogridRevolved ? 0.25 * pi * geo.D * geo.D : geo.D);
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
    rr.cx = c.placement.cx;
    rr.cy = c.placement.cy;
    rr.cz = c.placement.cz;

    // RANS / steady (docs/rans.md, "dns flow"): the exact distance to the
    // DISCRETE wall polyline, the freestream scalars, the run label. The
    // wall distance also feeds d1 and y+ of a laminar steady or --emit
    // surface run.
    const bool rans = c.rans != RansModel::None;
    rr.rans = c.rans;
    rr.phiInf[0] = c.ransPhiInf[0];
    rr.phiInf[1] = c.ransPhiInf[1];
    rr.steadyTol = c.compSteady ? c.steadyTol : 0.0;
    rr.steadySteps = c.steadySteps;
    rr.steadyImplicit =
        c.compSteady && c.steadyScheme == SteadyScheme::Implicit;
    rr.compensated = c.compensated;
    // --steady-sample: the explicit march's stop/stats cadence (the validator
    // rejects it with the implicit one); a time march keeps its 200.
    if (c.compSteady)
    {
        rr.statsEvery = c.steadySample;
        rr.ltsMax = c.ltsMax;
        rr.precondK = c.precondK;
    }
    rr.label = flowRunLabel(c);
    if (rans || c.compSteady || c.emit.contains(Emit::Surface) ||
        c.emit.contains(Emit::Mesh))
        rr.wallDist = c.ogridRevolved
                          ? revolvedWallDistance(og->mesh, og->nTheta,
                                                 og->rev.wedge)
                          : ogridWallDistance(og->mesh);
    // --emit mesh: the mesh and its d1 now, before any device allocation.
    if (out->wantsMesh())
    {
        const double centre[3] = {c.placement.cx, c.placement.cy,
                                  c.placement.cz};
        const auto bytes =
            encodeMesh(*og, centre, wallFirstCellDistance(g, rr.wallDist));
        if (auto e = out->writeMeshArtefact("mesh", bytes); !e)
            return die(exitOutput, e.error());
    }
    if (!rr.label.empty()) std::printf("# %s\n", rr.label.c_str());

    // Quiet start: the IC is at REST, exactly consistent with the no-slip wall
    // (--ramp 0 starts it in the freestream instead; a steady march always
    // does, impulsively -- pseudo-time has no ramp). RANS scalars start at
    // their freestream value everywhere (the wall row images them).
    auto ic = CompState::zeros(g);
    const double u0 = c.ramp > 0 ? 0.0 : c.uInf;
    for (std::size_t q = 0; q < g.size(); ++q)
    {
        ic.rho[q] = rhoInf;
        ic.mx[q] = rhoInf * u0;
        ic.E[q] = pInf / (gamma - 1.0) + 0.5 * rhoInf * u0 * u0;
    }
    for (int i = 0; i < ransScalars(c.rans); ++i)
        rr.icTurb[i].assign(g.size(), rhoInf * c.ransPhiInf[i]);
    // The primitives the trip below adds to: the full revolution's wedge
    // start, or the uniform state above.
    std::vector<double> pr, pu, pv, pw, pp;
    if (c.ogridNphi > 0 || c.ogridTrip > 0)
    {
        if (wedgeStart)
        {
            pr = std::move(wedgeStart->rho);
            pu = std::move(wedgeStart->u);
            pv = std::move(wedgeStart->v);
            pw = std::move(wedgeStart->w);
            pp = std::move(wedgeStart->p);
            for (std::size_t q = 0; q < g.size(); ++q)
            {
                ic.rho[q] = pr[q];
                ic.mx[q] = pr[q] * pu[q];
                ic.my[q] = pr[q] * pv[q];
                ic.mz[q] = pr[q] * pw[q];
                ic.E[q] = pp[q] / (gamma - 1.0) +
                          0.5 * pr[q] *
                              (pu[q] * pu[q] + pv[q] * pv[q] + pw[q] * pw[q]);
            }
            for (int i = 0; i < ransScalars(c.rans); ++i)
                for (std::size_t q = 0; q < g.size(); ++q)
                    rr.icTurb[i][q] = pr[q] * wedgeStart->phi[i][q];
        }
        else
        {
            pr.assign(g.size(), rhoInf);
            pu.assign(g.size(), u0);
            pv.assign(g.size(), 0.0);
            pw.assign(g.size(), 0.0);
            pp.assign(g.size(), pInf);
        }
    }
    // --ogrid-trip (every O-grid but the wedge, which config rejects): a
    // solenoidal Gaussian vortex, axis along z, at (cx + D, cy + D/4, cz),
    // width w = D/4, strength A U -- u += 2 (y - y0)/w e,
    // v -= 2 (x - x0)/w e, e = A U exp(-|x - x0|^2/w^2): divergence-free
    // (w == 0, and e's z factor multiplies both), so acoustically silent. On
    // the full revolution it breaks the exact pi-rotation symmetry about the
    // axis that otherwise locks out the odd azimuthal modes; on the cylinder
    // the up-down symmetry that delays shedding. cz is the span's centre, and
    // the 2D cell centre sits on it. On the uniform start it is bit for bit
    // the aeb53b9 trip (adding v = w = 0 is exact).
    if (c.ogridTrip > 0)
    {
        const double A = c.ogridTrip * c.uInf, w = 0.25 * geo.D;
        const double x0 = c.placement.cx + geo.D,
                     y0 = c.placement.cy + 0.25 * geo.D, z0 = c.placement.cz;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                {
                    double x[3] = {0, 0, 0};
                    for (int dk = 0; dk < 2; ++dk)
                        for (int dj = 0; dj < 2; ++dj)
                            for (int di = 0; di < 2; ++di)
                            {
                                const auto p = og->mesh.node(i + di, j + dj,
                                                             k + dk);
                                for (int a = 0; a < 3; ++a) x[a] += 0.125 * p[a];
                            }
                    const double dx = x[0] - x0, dy = x[1] - y0, dz = x[2] - z0;
                    const double e =
                        A * std::exp(-(dx * dx + dy * dy + dz * dz) / (w * w));
                    const std::size_t q = g.idx(i, j, k);
                    const double uu = pu[q] + 2.0 * dy / w * e,
                                 vv = pv[q] + -2.0 * dx / w * e, ww = pw[q];
                    ic.mx[q] = pr[q] * uu;
                    ic.my[q] = pr[q] * vv;
                    ic.mz[q] = pr[q] * ww;
                    ic.E[q] = pp[q] / (gamma - 1.0) +
                              0.5 * pr[q] * (uu * uu + vv * vv + ww * ww);
                }
    }

    const auto res = gpu::runOGrid(g, *og, ic, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    if (auto e = out->finish(); !e) return die(exitOutput, e.error());
    std::printf("done: %ld steps, cd=%.4f cl=%.4f\n", res->steps, res->cd,
                res->cl);
    return 0;
}

int runFlowCompressible(const Case& c, CaseGeometry& geo)
{
    const auto& g = c.grid;
    const double gamma = kGammaAir, rhoInf = kRhoInf;
    const Freestream fs = compFreestream(c, geo.D);
    const double pInf = fs.pInf, Tinf = fs.Tinf;
    const Transport& gas = fs.gas;

    // Frontal area / reference dynamic pressure (used for qRef ONLY, never for
    // the solve -- the compressible body is the sharp ghost wall). One
    // definition, in the geometry layer; see geometry.hpp for why REFERENCE
    // spacings are the correct weights even under --focal (measured).
    const AxisRemap3 fRemap = focalRemap(g, c.focalActive, c.focalLo,
                                         c.focalHi, c.focalW, c.focalRatio);
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

    FrameGeometry fg = FrameGeometry::centred(rg, geo.D);
    fg.mask = raster.apply(mask);

    RunMeta rmeta = c.meta;
    if (raster.active()) rmeta.setGrid(rg);

    const auto oc = outputConfig(
        c, std::format("{{\"app\":\"dns\",\"re\":{},\"mach\":{}", c.re, c.mach));
    auto out = RunOutput::make(oc, rmeta, rg, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::CompRun rr;
    rr.raster = &raster;
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
    rr.thinSeal = c.thinSeal;
    rr.far = c.far;
    rr.open = c.open;
    rr.spongeCells = c.spongeCells;
    rr.nscbcBeta = c.nscbcBeta;
    rr.nscbcAbc = c.nscbcAbc;
    rr.nscbcEtaIn = c.nscbcEtaIn;
    rr.nscbcEtaInShear = c.nscbcEtaInShear;
    rr.nscbcEtaLat = c.nscbcEtaLat;
    rr.nscbcStrip = c.nscbcStrip;
    rr.nscbcSink = c.nscbcSink;
    rr.nscbcSinkAmp = c.nscbcSinkAmp;
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
    rr.rotorMrf = c.rotorMrf;
    rr.hFine = hFine(c);
    rr.compensated = c.compensated;
    // RANS on the immersed ghost wall (docs/rans.md, "Immersed-wall RANS"):
    // d = |sdf| at the physical centres, the freestream scalars, the label.
    if (c.rans != RansModel::None)
    {
        rr.rans = c.rans;
        rr.wallDist = immersedWallDistance(g, geo.sdf, fRemap);
        rr.phiInf[0] = c.ransPhiInf[0];
        rr.phiInf[1] = c.ransPhiInf[1];
        rr.steadyTol = c.compSteady ? c.steadyTol : 0.0;
        rr.steadySteps = c.steadySteps;
        rr.label = flowRunLabel(c);
        std::printf("# %s\n", rr.label.c_str());
    }
    // The frozen rotor's laminar steady march (--rotor-frame mrf --steady):
    // the same pseudo-time knobs, and its label.
    else if (c.compSteady)
    {
        rr.steadyTol = c.steadyTol;
        rr.steadySteps = c.steadySteps;
        rr.label = flowRunLabel(c);
        std::printf("# %s\n", rr.label.c_str());
    }
    // --steady-sample: the stop/stats cadence of the pseudo-time march only;
    // a time march keeps its 200.
    if (c.compSteady)
    {
        rr.statsEvery = c.steadySample;
        rr.ltsMax = c.ltsMax;
        rr.precondK = c.precondK;
    }

    auto res = gpu::runCompressible(g, host, geo.sdf,
                                    geo.hasBbox ? &geo.bbox : nullptr, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
}


// Grid sequencing: converge on n/2^(L-1), prolong, repeat, finish on n. Each
// level is a normally validated Case -- the ladder reuses validate() rather
// than hand-editing a Case, so no level can skip a check. Only the finest
// level writes output and only its result is reported. `fine` is the finest
// level's geometry, already built by runFlow; only the coarse ones are built here.
int runFlowSequenced(const Case& c, const RawConfig& raw, const Scene* scene,
                     CaseGeometry& fine)
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

        const bool last = l == L - 1;
        std::optional<CaseGeometry> coarse;
        if (!last)
        {
            auto g = buildCaseGeometry(*lc, scene);
            if (!g) return die(exitInput, g.error());
            coarse = std::move(*g);
        }
        CaseGeometry& geo = last ? fine : *coarse;

        VectorField prolonged = VectorField::zeros(lc->grid);
        if (l > 0)
        {
            prolonged = prolongMac(gPrev, hPrev, carry, lc->grid, lc->h);
            ic = &prolonged;
        }

        std::printf("# seq level %d/%d: n=%d h=%.6g (%.1f cells/D)%s\n", l + 1, L,
                    lc->n, lc->h, lc->placement.D / lc->h,
                    last ? "  [final]" : "");
        std::fflush(stdout);

        VectorField fin = VectorField::zeros(lc->grid);
        const int rc = runFlowIncompressible(*lc, geo, ic, last ? nullptr : &fin,
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

#endif // DNS_WITH_GPU

int runFlow(const Case& c, [[maybe_unused]] const RawConfig& raw, const Scene* scene)
{
    // Input before device: a bad mesh is exit 3 on every machine, GPU or not.
    [[maybe_unused]] const auto t0 = std::chrono::steady_clock::now();
    auto geo = buildCaseGeometry(c, scene);
    if (!geo) return die(exitInput, geo.error());
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns flow is GPU-only)");
#if DNS_WITH_GPU
    if (c.seqLevels > 1) return runFlowSequenced(c, raw, scene, *geo);
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
    const double gamma = kGammaAir, delta = 1.0, rho0 = 1.0;
    const ChannelStretchY remap{double(g.ny) * g.hy, g.ny, c.stretchBeta,
                                delta};

    // IC: Poiseuille + solenoidal curl(psi) trip (dns/flow_ic.hpp), or under
    // --rans the untripped mean start on the model's own wall distance.
    auto host = CompState::zeros(g);
    std::vector<float> mask(g.size(), 1.0f);
    std::vector<double> wallDist;
    std::array<std::vector<double>, 2> icTurb;
    if (c.rans == RansModel::None)
        channelIC(c, remap, delta, rho0, gamma, host, mask);
    else
    {
        wallDist = channelWallDistance(
            CurvMesh::channelY(g, c.stretchBeta, delta), delta);
        channelRansIC(c, wallDist, rho0, gamma, host, icTurb, mask);
    }

    FrameGeometry fg = FrameGeometry::centred(g, 2 * delta);
    fg.mask = std::move(mask);

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
    rr.rans = c.rans;
    rr.steadyTol = c.steadyTol;
    rr.steadySteps = c.steadySteps;
    rr.implicitSteady = c.steadyScheme == SteadyScheme::Implicit;
    rr.compensated = c.compensated;
    rr.les = c.cs;
    rr.sgs = c.sgsModel;
    rr.wallDist = std::move(wallDist);
    rr.icTurb = std::move(icTurb);

    auto res = gpu::runChannel(g, host, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
#else
    return exitNoGpu;
#endif
}


// --- conv ------------------------------------------------------------------

// The artefact sink of a convection run. The two branches are two MODELS and
// share no physics code, but they write the same artefacts with the same
// metadata and the same centre planes -- so the sink is built once and the
// only thing either branch supplies is its own ghost mask.
[[maybe_unused]] std::expected<RunOutput, std::string> convOutput(const Case& c,
                                                                  std::vector<float> mask)
{
    FrameGeometry fg = FrameGeometry::centred(c.grid, c.convH);
    fg.mask = std::move(mask);
    return RunOutput::make(
        outputConfig(c,
                     std::format("{{\"app\":\"dns\",\"rayleigh\":{},"
                                 "\"prandtl\":{}",
                                 c.rayleigh, c.prandtl)),
        c.meta, c.grid, std::move(fg));
}

int runConv(const Case& c)
{
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns conv is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const double gamma = kGammaAir;

    // Boussinesq branch: a different model, not a cheaper solver for the same
    // one (gpu/boussinesq.hpp). It shares the case derivation and the artefact
    // sink and nothing else.
    if (c.physics == Physics::Incompressible)
    {
        auto U = VectorField::zeros(g);
        auto T = zeros(g);
        std::vector<float> mask(g.size(), 1.0f);
        rbIC(c, U, T, mask);

        auto out = convOutput(c, std::move(mask));
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
        return reportDone(*res, *out);
    }

    auto host = CompState::zeros(g);
    std::vector<float> mask(g.size(), 1.0f);
    convIC(c, gamma, host, mask);

    auto out = convOutput(c, std::move(mask));
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
    return reportDone(*res, *out);
#else
    return exitNoGpu;
#endif
}


// --- flame -----------------------------------------------------------------

// --mode chamber (docs/chem.md, "Chamber"): the closed 2D combustion chamber.
int runChamberCase(const Case& c)
{
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns flame is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const ChamberSpec& s = c.chamber;
    const double Tu = c.flameTu, sig = c.flameModel.sigma;
    gpu::ChamberRun rr;
    rr.regions = chamberRegions(g, s);
    CompState host;
    std::vector<double> rhoY;
    const FlameEigen& tab = c.flameEig.tab;
    chamberIC(g, rr.regions, kGammaAir, Tu, sig, c.chamberCoflow,
              [&](double x) { return flameProfile(tab, x); }, s.xBase(), host,
              rhoY);

    FrameGeometry fg = FrameGeometry::centred(g, s.Hc);
    fg.mask.resize(g.size());
    for (std::size_t i = 0; i < g.size(); ++i)
        fg.mask[i] = rr.regions[i] == std::uint8_t(ChamberRegion::Solid) ? 0.0f
                                                                         : 1.0f;
    auto out = RunOutput::make(
        outputConfig(c, std::format("{{\"app\":\"dns\",\"flame\":\"chamber\","
                                    "\"sigma\":{},\"beta\":{},\"Lambda\":{},"
                                    "\"mach\":{},\"h\":{},\"xe\":{},"
                                    "\"yc\":{},\"He\":{},\"Hc\":{}",
                                    sig, c.flameModel.beta, c.flameEig.Lambda,
                                    c.mach, c.h, s.xe(), s.yc, s.He, s.Hc)),
        c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    rr.tr = Transport::air(c.prandtl, Tu, c.tinfK, c.prandtl, kGammaAir,
                           kRgasBox, c.sutherland);
    rr.sp.nY = 1;
    rr.sp.fuel = 0;
    rr.sp.lnA = c.chamberReaction ? std::log(c.flameEig.Lambda)
                                  : -std::numeric_limits<double>::infinity();
    rr.sp.Ta = c.flameModel.theta() * Tu;
    rr.sp.q = rr.tr.cp() * (sig - 1) * Tu;
    rr.sp.invPr = 1.0 / c.prandtl;
    rr.sp.openX = true;
    rr.sp.yIn[0] = 0.0; // the ambient is inert
    rr.Tu = Tu;
    rr.sigma = sig;
    rr.uCo = c.chamberCoflow;
    rr.etaIn = c.nscbcEtaIn;
    rr.thermalInflow = c.nscbcThermalInflow;
    rr.sinkNb = c.nscbcSink;
    rr.sinkAmp = c.nscbcSinkAmp;
    rr.spec = s;
    rr.mdot = c.chamberMdot;
    rr.ramp = c.chamberRamp;
    rr.reaction = c.chamberReaction;
    rr.cfl = c.cfl;
    rr.tEnd = c.tEnd;
    rr.dtRecompute = c.dtRecompute;
    rr.label = chamberRunLabel(c);

    auto res = gpu::runChamber(g, host, rhoY, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
#else
    return exitNoGpu;
#endif
}

int runFlame(const Case& c)
{
    if (c.chamberMode) return runChamberCase(c);
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns flame is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    CompState host = CompState::zeros(g);
    std::vector<double> rhoY[2];
    if (c.flameDiffusion) diffusionIC(c, host, rhoY);
    else if (c.flameBunsen) bunsenIC(c, host, rhoY[0]);
    else flameIC(c, host, rhoY[0]);

    FrameGeometry fg = FrameGeometry::centred(g, 1.0);
    fg.mask.assign(g.size(), 1.0f);
    auto out = RunOutput::make(
        outputConfig(
            c, c.flameCandle
                   ? std::format("{{\"app\":\"dns\",\"flame\":\"candle\","
                                 "\"stoich\":{},\"zst\":{},\"sigma\":{},"
                                 "\"beta\":{},\"Da\":{},\"mach\":{},"
                                 "\"gravity\":{},\"slot\":{},\"quiescent\":{}",
                                 c.diff.s, c.diff.zst(), c.diff.sigma,
                                 c.diff.beta, c.diff.Da, c.mach,
                                 c.flameGravity, c.bunsen.W,
                                 c.flameQuiescent)
               : c.flameDiffusion
                   ? std::format("{{\"app\":\"dns\",\"flame\":\"diffusion\","
                                 "\"stoich\":{},\"zst\":{},\"sigma\":{},"
                                 "\"beta\":{},\"Da\":{},\"mach\":{}",
                                 c.diff.s, c.diff.zst(), c.diff.sigma,
                                 c.diff.beta, c.diff.Da, c.mach)
                   : std::format("{{\"app\":\"dns\",\"flame\":\"premixed\","
                                 "\"sigma\":{},\"beta\":{},\"Lambda\":{},"
                                 "\"mach\":{}",
                                 c.flameModel.sigma, c.flameModel.beta,
                                 c.flameEig.Lambda, c.mach)),
        c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    if (c.flameDiffusion) {
        // Two reactants (dns/diffusion.hpp): box units rho0 = R = U_jet = 1,
        // l = D0/U_jet = 1 (mu = Pr), T0 = p0; the bunsen per-row inflow
        // carrying the Burke-Schumann sheet of the inflow mixture fraction.
        const DiffusionModel& m = c.diff;
        const double T0 = c.flameTu;
        gpu::FlameRun rr;
        rr.tr = Transport::air(c.prandtl, T0, c.tinfK, c.prandtl, kGammaAir,
                               kRgasBox, c.sutherland);
        rr.sp.nY = 2;
        rr.sp.fuel = 0;
        rr.sp.ox = 1;
        rr.sp.stoich = m.s;
        rr.sp.lnA = m.lnA();
        rr.sp.Ta = m.theta() * T0;
        rr.sp.q = rr.tr.cp() * T0 * m.qOverCpT0();
        rr.sp.invPr = 1.0 / c.prandtl;
        rr.sp.openX = true;
        rr.Tu = T0;
        rr.sigma = m.sigma;
        rr.uIn = 1.0;
        rr.pB = T0;
        rr.etaIn = c.nscbcEtaIn;
        rr.outSigma = kDiffOutflowSigma;
        rr.thermalInflow = c.nscbcThermalInflow;
        rr.sinkNb = c.nscbcSink;
        rr.sinkAmp = c.nscbcSinkAmp;
        rr.dth = 1.0; // frame omega scale: the length unit l
        rr.cfl = c.cfl;
        rr.tEnd = c.tEnd;
        rr.dtRecompute = c.dtRecompute;
        rr.label = flameRunLabel(c);
        rr.diffusion = true;
        rr.diff = m;
        rr.geom = c.bunsen;
        // Candle: gravity along -x over the isothermal ambient column
        // p_a(x) (R T0 = p0 = T0): the inflow row's density at p_a(h/2), the
        // outflow target p_a at the last row, and the column's hydrostatic
        // dp/dx = -g rho_a there for the NSCBC waves. g == 0: the jet's own
        // numbers, bit for bit.
        const double gx = c.flameCandle ? c.flameGravity : 0.0;
        double pIn = T0;
        if (gx > 0) {
            pIn = candleAmbientP(T0, gx, T0, 0.5 * g.hx);
            const double pOut = candleAmbientP(T0, gx, T0, (g.nx - 0.5) * g.hx);
            rr.pB = pOut;
            rr.gravity = gx;
            rr.hydroLo = -gx * pIn / T0;
            rr.hydroHi = -gx * pOut / T0;
            // Air drawn down through the top is the ambient's.
            rr.sp.ambientHi = true;
            rr.sp.yHi[0] = 0.0;
            rr.sp.yHi[1] = m.yO2;
        }
        rr.candle = c.flameCandle;
        rr.quiescent = c.flameQuiescent;
        for (int j = 0; j < g.ny; ++j) {
            const DiffusionInflowRow in =
                c.flameQuiescent ? quiescentInflow(m)
                                 : diffusionInflow(m, c.bunsen, (j + 0.5) * g.hy);
            const double T = T0 * in.t;
            const double rho = gx > 0 ? pIn / T : T0 / T; // p0 = T0, R = 1
            rr.rowU.push_back(in.u);
            rr.rowJet.push_back(in.jet);
            rr.rowTarget.push_back(c.nscbcThermalInflow ? 1.0 / (kRgasBox * T)
                                                        : rho);
            rr.rowRho.push_back(rho);
            rr.rowY.push_back(in.YF);
            rr.rowYO.push_back(in.YO);
        }
        rr.sp.yInRow[0] = rr.rowY;
        rr.sp.yInRow[1] = rr.rowYO;
        auto res = gpu::runFlame(g, host, rhoY, rr, *out);
        if (!res) return die(res.error().code, res.error().message);
        return reportDone(*res, *out);
    }

    const double Tu = c.flameTu, sig = c.flameModel.sigma;
    gpu::FlameRun rr;
    // mu = Pr at Tref = Tu: lambda_u = mu cp / Pr = cp (delta_D = 1).
    rr.tr = Transport::air(c.prandtl, Tu, c.tinfK, c.prandtl, kGammaAir,
                           kRgasBox, c.sutherland);
    rr.sp.nY = 1;
    rr.sp.fuel = 0;
    rr.sp.lnA = std::log(c.flameEig.Lambda);
    rr.sp.Ta = c.flameModel.theta() * Tu;
    rr.sp.q = rr.tr.cp() * (sig - 1) * Tu;
    rr.sp.invPr = 1.0 / c.prandtl;
    rr.sp.openX = true;
    rr.sp.yIn[0] = 1.0;
    rr.Tu = Tu;
    rr.sigma = sig;
    rr.uIn = c.flameUIn;
    rr.pB = Tu - c.flameUIn * c.flameUIn * (sig - 1); // the IC's burnt p
    rr.etaIn = c.nscbcEtaIn;
    rr.thermalInflow = c.nscbcThermalInflow;
    rr.sinkNb = c.nscbcSink;
    rr.sinkAmp = c.nscbcSinkAmp;
    rr.dth = c.flameEig.dthOverDD;
    rr.cfl = c.cfl;
    rr.tEnd = c.tEnd;
    rr.dtRecompute = c.dtRecompute;
    rr.profileDt = c.profileDt;
    rr.label = flameRunLabel(c);
    if (c.flameBunsen) {
        // The slot burner: per-row inflow (dns/bunsen.hpp) and p = p0 at the
        // outflow (the cone's pressure jump is O(M^2) of p0, not the 1D one).
        rr.bunsen = true;
        rr.geom = c.bunsen;
        rr.pB = Tu;
        for (int j = 0; j < g.ny; ++j) {
            const BunsenInflowRow in = bunsenInflow(c.bunsen, (j + 0.5) * g.hy);
            const double T = Tu * in.t, rho = Tu / T; // p0 = Tu, R = 1
            rr.rowU.push_back(in.u);
            rr.rowJet.push_back(in.jet);
            rr.rowTarget.push_back(c.nscbcThermalInflow ? 1.0 / (kRgasBox * T)
                                                        : rho);
            rr.rowRho.push_back(rho);
            rr.rowY.push_back(in.Y);
        }
        rr.sp.yInRow[0] = rr.rowY;
    }

    auto res = gpu::runFlame(g, host, std::span(rhoY, 1), rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
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
            if (stopRequested()) break;
            const double dt = ts.suggestDt(U, c.tEnd - t);
            if (!(dt > 0) || !std::isfinite(dt))
                return die(exitNumerical, "stable dt is not finite and positive "
                              "(the state diverged)");
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
                                     FrameGeometry::centred(g, 2 * pi));
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
        if (slices)
            if (auto e = slices->finish(); !e) return die(exitOutput, e.error());
        std::fprintf(stderr, "done: %ld steps (%s)\n", st->steps, gpu::backendName);
        return 0;
    }
#endif
    TimeStepper ts(g, sp);
    std::printf("# 3D TGV n=%d Re=%g cfl=%g backend=cpu\n", g.nx, c.re, c.cfl);
    TgvSeries::header();
    TgvSeries series;
    double t = 0.0, nextOut = 0.0;
    long steps = 0;
    auto probe = [&] {
        series.row(t, kineticEnergy(g, U) / vol, nu * velocityGradientSq(g, U));
    };
    while (t < c.tEnd)
    {
        if (stopRequested()) break;
        if (t >= nextOut) { probe(); nextOut += outEvery; }
        const double dt = ts.suggestDt(U, c.tEnd - t);
        if (!(dt > 0) || !std::isfinite(dt))
            return die(exitNumerical, "stable dt is not finite and positive "
                          "(the state diverged)");
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
    auto outOr = RunOutput::make(oc, c.meta, g, FrameGeometry::centred(g, 2 * pi));
    if (!outOr) return die(exitOutput, outOr.error());
    RunOutput out = std::move(*outOr);
    const std::string statsCols[] = {"t",   "dt",       "A",        "kPrime",
                                     "uPrime", "epsVisc", "epsInj",  "eta",
                                     "reLambda", "hOverEta", "cfl", "maxDiv"};
    if (auto e = out.declareSeries("stats", statsCols); !e)
        return die(exitOutput, e.error());

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
    if (auto e = out.finish(); !e) return die(exitOutput, e.error());
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
std::optional<double> benchVramGiB(std::size_t freeBefore)
{
    std::size_t freeB = 0, totalB = 0;
    if (hipMemGetInfo(&freeB, &totalB) != hipSuccess) return std::nullopt;
    return freeBefore > freeB ? double(freeBefore - freeB) / (1 << 30) : 0.0;
}

// ms per call of `step`, timed over `steps` calls after 3 warm-up calls, each
// phase closed by a checked synchronization; or the device failure that
// invalidated the measurement.
template <class Step>
std::expected<double, std::string> benchTime(int steps, Step&& step)
{
    for (int s = 0; s < 3; ++s) step();   // warm-up (incl. FFT tuning)
    if (auto e = hipDeviceSynchronize(); e != hipSuccess)
        return std::unexpected(std::string("warm-up: ") + hipGetErrorString(e));
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) step();
    if (auto e = hipDeviceSynchronize(); e != hipSuccess)
        return std::unexpected(std::string("timed steps: ") + hipGetErrorString(e));
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count()
                      / steps;
    if (auto e = hipGetLastError(); e != hipSuccess)
        return std::unexpected(std::string("launch: ") + hipGetErrorString(e));
    return ms;
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
    if (hipMemGetInfo(&freeBefore, &totalB) != hipSuccess)
        return die(exitResource, "GPU memory query failed");

    if (c.physics == Physics::Compressible)
    {
        constexpr double gamma = kGammaAir;
        auto D = gpu::DeviceComp::make(g, gamma, c.nu, c.cs, c.sgsModel);
        auto S = gpu::DeviceCompState::make(g.size());
        if (!D || !S) return die(exitResource, "GPU init failed");
        if (S->upload(benchCompState(g, U, gamma, c.mach)) != hipSuccess)
            return die(exitResource, "GPU upload failed");
        const auto vram = benchVramGiB(freeBefore);
        if (!vram) return die(exitResource, "GPU memory query failed");
        const auto ms =
            benchTime(c.steps, [&] { D->step(*S, gpu::Real(dt)); });
        if (!ms) return die(exitResource, "GPU failure in bench " + ms.error());
        benchReport(g, c.cs > 0 ? "compressible LES" : "compressible", *ms,
                    *vram);
        return 0;
    }

    auto ts = gpu::TimeStepper::make(g, {c.nu, 0.9});
    auto D = gpu::DeviceField::make(g.size());
    if (!ts || !D) return die(exitResource, "GPU init failed");
    if (gpu::uploadField(U, *D) != hipSuccess)
        return die(exitResource, "GPU upload failed");
    ts->setForcing({0.1, {0, 0, 0}}); // exercise the forcing path too
    const auto vram = benchVramGiB(freeBefore);
    if (!vram) return die(exitResource, "GPU memory query failed");
    const auto ms = benchTime(c.steps, [&] { ts->step(*D, dt); });
    if (!ms) return die(exitResource, "GPU failure in bench " + ms.error());
    benchReport(g, "incompressible", *ms, *vram);
    return 0;
#else
    return exitNoGpu;
#endif
}

// --- prop --------------------------------------------------------------------

std::expected<std::string, std::string> readText(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::unexpected("cannot read '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    if (f.bad()) return std::unexpected("error reading '" + path + "'");
    return ss.str();
}

// Actuator-disk (Froude) efficiency bound at this CT and J.
double etaIdeal(double J, double CT)
{
    if (J <= 0 || CT <= 0) return std::numeric_limits<double>::quiet_NaN();
    return 2.0 / (1.0 + std::sqrt(1.0 + 8.0 * CT / (pi * J * J)));
}

int runPropSelfTest()
{
    std::printf("liftline self-test\n");
    bool good = true;
    for (const auto& ck : prop::liftlineSelfTest())
    {
        std::printf("  %s  %s: %s\n", ck.pass ? "PASS" : "FAIL", ck.name.c_str(),
                    ck.message.c_str());
        good = good && ck.pass;
    }
    return good ? 0 : exitNumerical;
}

// Host-only: Acquire = read the two input CSVs (exit 3) and create the output
// CSV (exit 4), then Run. No GPU is ever probed.
int runProp(const Case& cs)
{
    const prop::Case& pc = cs.prop;
    if (pc.selftest) return runPropSelfTest();

    // Phase 3: inputs, then every output, before any work.
    auto pfText = readText(pc.planformPath);
    if (!pfText) return die(exitInput, "--planform: " + pfText.error());
    auto pf = prop::Planform::fromCsv(*pfText);
    if (!pf) return die(exitInput, "--planform " + pc.planformPath + ": " + pf.error());
    const double hubRR = pc.hubRR > 0 ? pc.hubRR : pf->rootRR();
    if (!(hubRR > 0 && hubRR < 1))
        return die(exitInput, "--planform " + pc.planformPath
                                  + ": its first r_R must be < 1 to serve as the hub "
                                    "radius (or pass --hub-radius)");
    std::optional<prop::Polar> polar;
    if (pc.polarPath.empty())
        polar = prop::Polar::thinAirfoil(pc.thinAlpha0Deg, pc.thinCd0,
                                         prop::viternaCdMax(pf->aspectRatio(hubRR)));
    else
    {
        auto pt = readText(pc.polarPath);
        if (!pt) return die(exitInput, "--polar: " + pt.error());
        auto p = prop::Polar::fromCsv(*pt);
        if (!p) return die(exitInput, "--polar " + pc.polarPath + ": " + p.error());
        polar = std::move(*p);
    }
    std::ofstream csv;
    if (!pc.outCsv.empty())
    {
        csv.open(pc.outCsv, std::ios::binary | std::ios::trunc);
        if (!csv) return die(exitOutput, "cannot create '" + pc.outCsv + "'");
    }
    prop::Case run = pc;
    run.hubRR = hubRR;

    // Phase 4.
    const bool ll = pc.method == prop::Method::Liftline;
    const double n = pc.op.revPerSec();
    std::vector<prop::CurvePoint> rows;
    std::string report, span;
    if (ll)
    {
        const auto geo = prop::liftGeometry(*pf, hubRR, pc.ll.panels, pc.ll.hub,
                                            pc.op.diameter);
        auto curve = prop::liftlineCurve(run, geo, *polar);
        rows = std::move(curve.rows);
        report = std::format("liftline: B={} D={:g} m {:.0f} rpm, {} panels, wake {:g} D, "
                             "tip Mach {:.3f}\n"
                             "     J       CT        CP       eta      T[N]     Q[Nm]   conv\n",
                             pc.blades, pc.op.diameter, pc.op.rpm, pc.ll.panels,
                             pc.ll.wakeLength, pi * n * pc.op.diameter / pc.op.sound);
        for (const auto& r : rows)
            report += std::format("  {:5.3f} {:9.5f} {:9.5f} {:8.4f} {:9.4f} {:9.5f}   {}\n",
                                  r.J, r.CT, r.CP, r.eta, r.T, r.Q, int(r.converged));
        report += prop::curveSummary(rows);
        if (curve.span) span = "\n" + prop::liftlineSpanTable(*pc.dumpSpan, geo, *curve.span);
    }
    else
    {
        const auto bl = prop::bemtBlade(*pf, pc.blades, hubRR, pc.stations, pc.op.diameter);
        auto curve = prop::bemtCurve(run, bl, *polar);
        rows = std::move(curve.rows);
        if (curve.machHi > 0.6)
            std::fprintf(stderr, "dns: warning: station Mach up to %.2f with "
                                 "incompressible polars\n", curve.machHi);
        report = std::format("bemt: B={} D={:g} m {:g} rpm, planform {}, polar {}, {} "
                             "elements\n"
                             "     J       CT       CP      eta  eta_ideal       T[N]     "
                             "Q[Nm]      P[W]     Re75\n",
                             pc.blades, pc.op.diameter, pc.op.rpm, pc.planformPath,
                             pc.polarPath.empty() ? std::string("thin") : pc.polarPath,
                             pc.stations);
        std::vector<double> over;
        for (const auto& r : rows)
        {
            const double ei = etaIdeal(r.J, r.CT);
            report += std::format("  {:5.3f} {:8.5f} {:8.5f} {:8.4f} {:9.4f} {:10.4f} "
                                  "{:9.5f} {:9.3f} {:8.3g}\n",
                                  r.J, r.CT, r.CP, r.eta, ei, r.T, r.Q, r.P, r.re75);
            if (std::isfinite(ei) && r.eta > ei + 1e-9) over.push_back(r.J);
        }
        report += prop::curveSummary(rows);
        report += std::format("  momentum check: max |dT_blade - dT_momentum| = {:.1e} of "
                              "peak dT/dr (momentum-region annuli)\n",
                              curve.worstMomErr);
        if (!over.empty())
        {
            std::string js;
            for (double j : over) js += std::format("{}{:g}", js.empty() ? "" : ", ", j);
            std::fprintf(stderr, "dns: warning: eta above the actuator-disk bound at J = %s\n",
                         js.c_str());
        }
        if (curve.unsolved)
            std::fprintf(stderr, "dns: warning: %d annulus solves found no bracket (nan, "
                                 "dropped from the integrals)\n", curve.unsolved);
        if (pc.dumpSpan)
            span = "\n" + prop::bemtSpanTable(*pc.dumpSpan, bl,
                                              prop::bemtSolve(*pc.dumpSpan, bl, pc.op, *polar));
    }
    if (csv.is_open())
    {
        csv << prop::curveCsv(rows, ll);
        csv.close();
        if (!csv) return die(exitOutput, "write failed on '" + pc.outCsv + "'");
        report += "wrote " + pc.outCsv + "\n";
    }
    if (!pc.quiet) std::fputs((report + span).c_str(), stdout);
    std::string bad;
    for (const auto& r : rows)
        if (!r.converged) bad += std::format("{}{:g}", bad.empty() ? "" : ", ", r.J);
    if (!bad.empty())
        std::fprintf(stderr, "dns: warning: lifting line NOT converged at J = %s "
                             "(flagged converged = 0)\n", bad.c_str());
    return 0;
}

// --- mesh --------------------------------------------------------------------

// --dry-run: the manifest only (owner note, boundary, zone names); no bulk
// data. A mesh that cannot be read is an input error (3).
int runMeshManifest(const Case& cs)
{
    auto m = readPolyManifest(cs.polyMeshDir);
    if (!m) return die(exitInput, m.error());
    std::string s = std::format("manifest:\n"
                                "  format     {}, label {}, scalar 64\n"
                                "  counts     {} points, {} cells, {} faces ({} internal)\n",
                                m->binary ? "binary" : "ascii", m->labelBits, m->nPoints, m->nCells,
                                m->nFaces, m->nInternalFaces);
    for (const auto& p : m->patches)
    {
        s += std::format("  patch      {:<24} {:<14} {:>9} faces", p.name, foamPatchTypeName(p.type),
                         p.nFaces);
        if (p.type == FoamPatchType::Cyclic)
            s += " <-> " + p.neighbourPatch
                 + (p.separation ? std::format(" (translational {} {} {})", (*p.separation)[0],
                                               (*p.separation)[1], (*p.separation)[2])
                                 : std::string(" (transform unknown)"));
        s += "\n";
    }
    for (const auto& z : m->cellZones) s += "  cellZone   " + z + "\n";
    std::fputs(s.c_str(), stdout);
    return 0;
}

// Host-only: Acquire = read the polyMesh and build the FaceMesh (exit 3 on
// any rejection), Run = the quality report. Never probes a GPU.
int runMesh(const Case& cs)
{
    auto pm = readPolyMesh(cs.polyMeshDir);
    if (!pm) return die(exitInput, pm.error());
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder(cs.meshOrder));
    if (!fm) return die(exitInput, cs.polyMeshDir + ": " + fm.error());
    std::fputs(meshReport(*fm, meshQuality(*fm, *pm)).c_str(), stdout);
    return 0;
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
        // Phase 1.5 for `dns mesh` (the --scene precedent): the manifest
        // names the input file's facts, read without bulk data.
        if (c->sub == Subcommand::Mesh) return runMeshManifest(*c);
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
    case Subcommand::Prop: return runProp(*c);
    case Subcommand::Flame: return runFlame(*c);
    case Subcommand::Mesh: return runMesh(*c);
    }
    return 1;
}
