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
#include "dns/config.hpp"
#include "dns/diagnostics.hpp"
#include "dns/geometry.hpp"
#include "dns/hit_ic.hpp"
#include "dns/meshsdf.hpp"
#include "dns/operators.hpp"
#include "dns/output.hpp"
#include "dns/taylor_green.hpp"
#include "dns/timestepper.hpp"
#include "dns/trimesh.hpp"
#include "dns/units.hpp"
#include "dns/vtk_writer.hpp"

#if DNS_WITH_HIP
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/runner_comp.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/runner_incomp.hpp"
#include "dns/gpu/timestepper.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <optional>
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

#if DNS_WITH_HIP
bool haveGpu(const Case& c)
{
    if (c.backend == Backend::Cpu) return false;
    return gpu::deviceAvailable();
}
#else
bool haveGpu(const Case&) { return false; }
#endif

// --- geometry -------------------------------------------------------------

struct Geometry
{
    CutGeometry cut;
    Sdf sdf;
    double D = 1.0;             // characteristic length actually realised
    double qRef = 1.0;
    std::array<double, 6> bbox{};
    bool hasBbox = false;
};

// Builds the body's sdf and cut geometry from a validated Case. The per-body
// sample budgets are carried through verbatim: changing one shifts the cut and
// with it the measured coefficients (masterplan §6.11).
std::expected<Geometry, std::string> buildGeometry(const Case& c)
{
    Geometry out;
    const auto& g = c.grid;
    const auto& p = c.placement;
    const double h = c.h;
    out.D = p.D;

    switch (c.bodyKind)
    {
    case BodyKind::Cylinder:
    {
        const CylinderZ cyl{p.cx, p.cy, p.D / 2.0};
        if (c.dimples > 0)
        {
            // Footprint 8h, depth 4h -- near real golf proportions when
            // D ~ 256h and still grid-resolved.
            const double depth = 4.0 * h, rd = pitRadius(8.0 * h, depth);
            const auto q = ringCentres(cyl, c.dimples, rd, depth);
            out.sdf = dimpled(sdfOf(cyl), q, rd, true);
            auto r = sampleCut(g, out.sdf, {8, 8, true});
            if (!r) return std::unexpected(r.error());
            out.cut = std::move(*r);
        }
        else
        {
            out.sdf = sdfOf(cyl);
            out.cut = CutGeometry::make(g, cyl);
        }
        break;
    }
    case BodyKind::Sphere:
    {
        const Sphere s{p.cx, p.cy, p.cz, p.D / 2.0};
        if (c.dimples > 0)
        {
            const double depth = 2.5 * h, rd = pitRadius(5.0 * h, depth);
            const auto q = fibonacciCentres(s, c.dimples, rd, depth);
            out.sdf = dimpled(sdfOf(s), q, rd);
            auto r = sampleCut(g, out.sdf, {4, 4});
            if (!r) return std::unexpected(r.error());
            out.cut = std::move(*r);
        }
        else
        {
            out.sdf = sdfOf(s);
            out.cut = CutGeometry::make(g, s);
        }
        break;
    }
    case BodyKind::Cube:
    {
        // 45-degree rotated square prism spanning the full z extent.
        const RotatedCube cube{p.cx, p.cy, p.D / 2.0, 0.0,
                               g.nz == 1 ? g.hz : double(g.nz) * g.hz};
        out.sdf = sdfOf(cube);
        out.cut = CutGeometry::make(g, cube);
        break;
    }
    case BodyKind::Lens:
    {
        // aspect = w / hh, with the height fixed at D.
        const double hh = p.D / 2.0, w = c.aspect * hh;
        const Lens lens{p.cx, p.cy, w, hh};
        out.sdf = sdfOf(lens);
        auto r = exactLensCut(g, lens, 64);
        if (!r) return std::unexpected(r.error());
        out.cut = std::move(*r);
        out.D = 2.0 * hh;
        break;
    }
    case BodyKind::Mesh:
    {
        auto meshOr = loadMesh(c.meshPath);
        if (!meshOr) return std::unexpected(meshOr.error());
        TriMesh mesh = std::move(*meshOr);
        if (mesh.tris.empty()) return std::unexpected("mesh has no triangles");
        const Aabb bb0 = bounds(mesh);
        const double ext0 = std::max({bb0.hi[0] - bb0.lo[0], bb0.hi[1] - bb0.lo[1],
                                      bb0.hi[2] - bb0.lo[2]});
        if (!(ext0 > 0)) return std::unexpected("degenerate mesh bounds");
        const double scale = c.scale > 0 ? c.scale : p.D / ext0;
        transform(mesh, scale, c.pitch * pi / 180.0, c.yaw * pi / 180.0,
                  {p.cx, p.cy, p.cz});
        const Aabb bb = bounds(mesh);
        out.D = std::max({bb.hi[0] - bb.lo[0], bb.hi[1] - bb.lo[1],
                          bb.hi[2] - bb.lo[2]});
        out.bbox = {bb.lo[0], bb.lo[1], bb.lo[2], bb.hi[0], bb.hi[1], bb.hi[2]};
        out.hasBbox = true;

        auto sdfp = std::make_shared<MeshSdf>(std::move(mesh), c.robust);
        if (!sdfp->watertight() && !c.robust)
            std::fprintf(stderr, "dns: warning: mesh is not watertight; "
                                 "consider --robust\n");
        out.sdf = [sdfp](double x, double y, double z) { return (*sdfp)(x, y, z); };
        auto r = sampleCut(g, out.sdf, {8, 8});
        if (!r) return std::unexpected(r.error());
        out.cut = std::move(*r);
        break;
    }
    case BodyKind::None:
        return std::unexpected("no body");
    }
    return out;
}

#if DNS_WITH_HIP

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
VectorField prolongMac(const Grid& gc, double hc, const VectorField& C,
                       const Grid& gf, double hf)
{
    auto F = VectorField::zeros(gf);
    const bool flat = gc.nz == 1 || gf.nz == 1;

    // Sample the coarse array at continuous coarse indices, clamped to range.
    auto sample = [&](const std::vector<double>& q, double ci, double cj, double ck) {
        const auto lerp1 = [](double a, double b, double t) { return a + (b - a) * t; };
        ci = std::clamp(ci, 0.0, double(gc.nx - 1));
        cj = std::clamp(cj, 0.0, double(gc.ny - 1));
        ck = flat ? 0.0 : std::clamp(ck, 0.0, double(gc.nz - 1));
        const int i0 = std::min(int(ci), gc.nx - 1), i1 = std::min(i0 + 1, gc.nx - 1);
        const int j0 = std::min(int(cj), gc.ny - 1), j1 = std::min(j0 + 1, gc.ny - 1);
        const int k0 = flat ? 0 : std::min(int(ck), gc.nz - 1);
        const int k1 = flat ? 0 : std::min(k0 + 1, gc.nz - 1);
        const double tx = ci - i0, ty = cj - j0, tz = flat ? 0.0 : ck - k0;
        const double c00 = lerp1(q[gc.idx(i0, j0, k0)], q[gc.idx(i1, j0, k0)], tx);
        const double c10 = lerp1(q[gc.idx(i0, j1, k0)], q[gc.idx(i1, j1, k0)], tx);
        const double c01 = lerp1(q[gc.idx(i0, j0, k1)], q[gc.idx(i1, j0, k1)], tx);
        const double c11 = lerp1(q[gc.idx(i0, j1, k1)], q[gc.idx(i1, j1, k1)], tx);
        return lerp1(lerp1(c00, c10, ty), lerp1(c01, c11, ty), tz);
    };

    for (int k = 0; k < gf.nz; ++k)
        for (int j = 0; j < gf.ny; ++j)
            for (int i = 0; i < gf.nx; ++i)
            {
                // Face-normal axis uses the face map, the other two the centre
                // map; that is what makes this a MAC interpolation and not a
                // collocated one.
                const double Xf = (i - 1) * hf, Yf = (j - 1) * hf, Zf = (k - 1) * hf;
                const double Xc = (i - 0.5) * hf, Yc = (j - 0.5) * hf,
                             Zc = (k - 0.5) * hf;
                const std::size_t q = gf.idx(i, j, k);
                F.u[q] = sample(C.u, Xf / hc + 1.0, Yc / hc + 0.5, Zc / hc + 0.5);
                F.v[q] = sample(C.v, Xc / hc + 0.5, Yf / hc + 1.0, Zc / hc + 0.5);
                F.w[q] = flat ? 0.0
                              : sample(C.w, Xc / hc + 0.5, Yc / hc + 0.5,
                                       Zf / hc + 1.0);
            }
    return F;
}

int runFlowIncompressible(const Case& c, Geometry& geo,
                          const VectorField* ic0 = nullptr,
                          VectorField* finalOut = nullptr,
                          bool suppressOutput = false)
{
    const auto& g = c.grid;
    const double h = c.h;
    const auto st = sanitizeCut(g, geo.cut);   // before the ring: body cells only
    const bool fringe = c.far == FarField::Fringe || c.far == FarField::Periodic;
    if (!fringe) closeBoxBoundary(g, geo.cut); // fringe: the ring stays periodic
    std::printf("# sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed, st.minLambda,
                st.minAperture);

    // Frontal area measured from the apertures: a (j,k) stream column is
    // blocked by 1 - min_i ax (exact for convex bodies, honest for the rest).
    double aFront = 0.0;
    for (int k = 1; k <= (g.nz == 1 ? 1 : g.nz - 2); ++k)
        for (int j = 1; j <= g.ny - 2; ++j)
        {
            double amin = 1.0;
            for (int i = 2; i <= g.nx - 2; ++i)
                amin = std::min(amin, geo.cut.ax[g.idx(i, j, k)]);
            aFront += (1.0 - amin) * g.hy * (g.nz == 1 ? g.hz : g.hz);
        }
    if (!(aFront > 0))
        return die(exitInput, "the body produced no solid cells (too small?)");
    const double qRef = 0.5 * c.uInf * c.uInf * aFront;

    // Clean laminar free stream plus a deterministic z-uniform trip one D
    // behind the body, to break the symmetric-wake transient early. With grid
    // sequencing the coarse level's converged field replaces both.
    auto U = VectorField::zeros(g);
    if (ic0) U = *ic0;
    else {
    for (auto& x : U.u) x = c.uInf;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const double x = i * h, y = (j + 0.5) * h;
                if (x > c.placement.cx + geo.D / 2 && x < c.placement.cx + 1.5 * geo.D
                    && std::abs(y - c.placement.cy) < 0.5 * geo.D)
                    U.v[g.idx(i, j, k)] = 0.05 * c.uInf;
            }
    }

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

    RunOutput::Config oc;
    oc.prefix = c.outPrefix;
    oc.kinds = suppressOutput ? std::set<Emit>{Emit::None} : c.emit;
    oc.fdt = c.fdt;
    oc.pdt = c.pdt;
    oc.wdt = c.wdt;
    oc.planesPerAxis = c.planes;
    oc.simJson = "{\"app\":\"dns\",\"re\":" + std::to_string(c.re);
    auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::IncompRun rr;
    rr.nu = c.nu;
    rr.uInf = c.uInf;
    rr.qRef = qRef;
    rr.D = geo.D;
    rr.cfl = c.cfl;
    rr.tEnd = c.tEnd;
    rr.tol = c.tol;
    rr.maxIters = c.maxIters;
    rr.les = c.cs;
    rr.fringe = fringe;
    rr.open = c.open;
    rr.pot = c.farPot;
    rr.steadyTol = c.steadyTol;
    rr.cx = c.placement.cx;
    rr.cy = c.placement.cy;
    rr.cz = c.placement.cz;
    rr.fringeCells = c.spongeCells;
    rr.fringeAxes = c.fringeAxes;

    const auto gg = gpu::GpuGrid::from(g);
    rr.box = gpu::bodyBox(gg, c.placement.cx, c.placement.cy, c.placement.cz,
                          0.5 * geo.D);
    if (g.nz == 1 || c.bodyKind == BodyKind::Cylinder || c.bodyKind == BodyKind::Lens)
    {   // z-invariant bodies span the whole interior in z.
        rr.box.k0 = gg.kInLo();
        rr.box.k1 = gg.kInHi();
    }

    auto res = gpu::runIncompressible(g, geo.cut, geo.sdf, U, rr, *out, finalOut);
    if (!res) return die(res.error().code, res.error().message);
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", res->steps,
                out->frames(), res->msPerStep);
    return 0;
}

// --- flow: compressible ----------------------------------------------------

int runFlowCompressible(const Case& c, Geometry& geo)
{
    const auto& g = c.grid;
    const double h = c.h;
    const double gamma = 1.4, rhoInf = 1.0;
    const double pInf = rhoInf * c.uInf * c.uInf / (gamma * c.mach * c.mach);
    const double mu = rhoInf * c.uInf * geo.D / c.re;

    // Frontal area from the aperture-measured cut (used for qRef ONLY, never
    // for the solve -- the compressible body is the sharp ghost wall).
    double aFront = 0.0;
    {
        auto sc = geo.cut;
        (void)sanitizeCut(g, sc);
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
            {
                double amin = 1.0;
                for (int i = 1; i < g.nx; ++i)
                    amin = std::min(amin, sc.ax[g.idx(i, j, k)]);
                aFront += (1.0 - amin) * g.hy * g.hz;
            }
    }
    const double span = g.nz > 1 ? g.nz * g.hz : 1.0;
    const double qRef =
        aFront > 0 ? 0.5 * rhoInf * c.uInf * c.uInf * (aFront / span)
                   : 0.5 * rhoInf * c.uInf * c.uInf * geo.D;

    // IC. Quiet start (ramp > 0): rest, exactly consistent with the no-slip
    // body; the freestream is then raised by silent Galilean boosts. ramp == 0:
    // impulsive freestream tapered to rest across ~3 cells outside the surface.
    const double u0 = c.ramp > 0 ? 0.0 : c.uInf;
    // Deterministic solenoidal trip, quiet start only: a weak Gaussian vortex
    // one D behind the body seeds the wake instability. Divergence-free, so
    // acoustically silent; the quiet start alone is too symmetric to shed.
    const double kA = c.ramp > 0 ? c.kick * c.uInf : 0.0, kw = 0.25 * geo.D;
    const double kx = c.placement.cx + geo.D, ky = c.placement.cy + 0.25 * geo.D;

    auto host = CompState::zeros(g);
    std::vector<float> mask(g.size(), 1.0f);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const std::size_t q = g.idx(i, j, k);
                const double x = (i + 0.5) * h, y = (j + 0.5) * h,
                             z = g.nz > 1 ? (k + 0.5) * g.hz : c.placement.cz;
                const double d = geo.sdf(x, y, z);
                const double f = std::clamp(d / (3.0 * h), 0.0, 1.0);
                double uu = u0 * f * f * (3 - 2 * f), vv = 0.0;
                if (kA != 0.0 && d > 0.0)
                {
                    const double dx = x - kx, dy = y - ky;
                    const double e = kA * std::exp(-(dx * dx + dy * dy) / (kw * kw));
                    uu += 2 * dy / kw * e;
                    vv -= 2 * dx / kw * e;
                }
                host.rho[q] = rhoInf;
                host.mx[q] = rhoInf * uu;
                host.my[q] = rhoInf * vv;
                host.E[q] = pInf / (gamma - 1.0)
                          + 0.5 * rhoInf * (uu * uu + vv * vv);
                if (d < 0.0) mask[q] = 0.0f;
            }

    FrameGeometry fg;
    fg.mask = std::move(mask);
    fg.ic = g.nx / 2;
    fg.jc = g.ny / 2;
    fg.kc = g.nz / 2;
    fg.D = geo.D;

    RunOutput::Config oc;
    oc.prefix = c.outPrefix;
    oc.kinds = c.emit;
    oc.fdt = c.fdt;
    oc.pdt = c.pdt;
    oc.wdt = c.wdt;
    oc.planesPerAxis = c.planes;
    oc.simJson = "{\"app\":\"dns\",\"re\":" + std::to_string(c.re)
               + ",\"mach\":" + std::to_string(c.mach);
    auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::CompRun rr;
    rr.gamma = gamma;
    rr.mu = mu;
    rr.uInf = c.uInf;
    rr.rhoInf = rhoInf;
    rr.pInf = pInf;
    rr.mach = c.mach;
    rr.D = geo.D;
    rr.qRef = qRef;
    rr.cfl = c.cfl;
    rr.tEnd = c.ramp + c.tEnd;
    rr.les = c.cs;
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
    // 3D runs report |curl| (sequential map); native 2D reports signed wz so
    // the diverging map keeps the sign of the shed vortices.
    rr.signedVorticity = g.nz == 1;

    auto res = gpu::runCompressible(g, host, geo.sdf,
                                    geo.hasBbox ? &geo.bbox : nullptr, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", res->steps,
                out->frames(), res->msPerStep);
    return 0;
}

#endif // DNS_WITH_HIP


// Grid sequencing: converge on n/2^(L-1), prolong, repeat, finish on n. Each
// level is a normally validated Case -- the ladder reuses validate() rather
// than hand-editing a Case, so no level can skip a check. Only the finest
// level writes output and only its result is reported.
int runFlowSequenced(const Case& c, const RawConfig& raw)
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

        auto geo = buildGeometry(*lc);
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

int runFlow(const Case& c, const RawConfig& raw)
{
    if (!haveGpu(c))
        return die(exitNoGpu, "no HIP device (dns flow is GPU-only)");
#if DNS_WITH_HIP
    if (c.seqLevels > 1) return runFlowSequenced(c, raw);
    const auto t0 = std::chrono::steady_clock::now();
    auto geo = buildGeometry(c);
    if (!geo) return die(exitInput, geo.error());
    std::printf("# geometry built in %.2f s (D = %.4f, %.1f cells)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count(),
                geo->D, geo->D / c.h);
    return c.physics == Physics::Compressible ? runFlowCompressible(c, *geo)
                                              : runFlowIncompressible(c, *geo);
#else
    return exitNoGpu;
#endif
}

#if DNS_WITH_HIP
// Mid-plane speed / |curl| frames for the periodic subcommands (the former
// `movie` app). NOTE: `movie` also wrote a pressure channel, obtained from a
// separate lap(p) = div(conv U) solve; that channel is not carried over.
class PeriodicFrames
{
public:
    static std::expected<PeriodicFrames, std::string>
    make(const Case& c, const Grid& g)
    {
        FrameGeometry fg;
        fg.ic = g.nx / 2;
        fg.jc = g.ny / 2;
        fg.kc = g.nz / 2;
        fg.D = 2 * pi;
        RunOutput::Config oc;
        oc.prefix = c.outPrefix;
        oc.kinds = {Emit::Slices};
        oc.fdt = c.fdt;
        auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
        if (!out) return std::unexpected(out.error());
        PeriodicFrames p;
        p.out_ = std::move(*out);
        auto a = gpu::DeviceScalar::make(g.size());
        auto b = gpu::DeviceScalar::make(g.size());
        if (!a || !b) return std::unexpected("GPU allocation failed");
        p.vort_ = std::move(*a);
        p.speed_ = std::move(*b);
        p.hv_.resize(g.size());
        p.hs_.resize(g.size());
        p.fv_.resize(g.size());
        p.fs_.resize(g.size());
        return p;
    }

    bool wants(double t) const { return out_->wantsFrame(t); }

    void write(const gpu::GpuGrid& gg, const gpu::DeviceField& U, double t,
               std::size_t n)
    {
        gpu::vorticityZ(gg, U, *vort_);
        gpu::speedMac(gg, U, *speed_);
        (void)vort_->download(hv_.data());
        (void)speed_->download(hs_.data());
        for (std::size_t c = 0; c < n; ++c)
        {
            fv_[c] = float(hv_[c]);
            fs_[c] = float(hs_[c]);
        }
        const FrameField ff[2]{{"vort", fv_.data(), 0, 8, true},
                               {"speed", fs_.data(), 0, 2, false}};
        (void)out_->frame(t, FrameViews{ff});
    }

private:
    std::optional<RunOutput> out_;
    std::optional<gpu::DeviceScalar> vort_, speed_;
    std::vector<gpu::Real> hv_, hs_;
    std::vector<float> fv_, fs_;
};
#endif

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

#if DNS_WITH_HIP
    if (haveGpu(c))
    {
        auto ts = gpu::TimeStepper::make(g, sp);
        auto D = gpu::DeviceField::make(g.size());
        if (!ts || !D) return die(exitResource, "GPU init failed");
        (void)gpu::uploadField(U, *D);
        const auto gg = gpu::GpuGrid::from(g);
        auto red = gpu::ReduceScratch::make();
        if (!red) return die(exitResource, "GPU init failed");

        std::optional<PeriodicFrames> frames;
        if (c.emit.contains(Emit::Slices))
        {
            auto f = PeriodicFrames::make(c, g);
            if (!f) return die(exitOutput, f.error());
            frames = std::move(*f);
        }
        std::printf("# 3D TGV n=%d Re=%g cfl=%g backend=hip\n", g.nx, c.re, c.cfl);
        std::printf("t,K,dKdt,epsVisc\n");
        double t = 0.0, nextOut = 0.0;
        long steps = 0;
        auto probe = [&] {
            emit(t, gpu::kineticEnergy(gg, *D, *red) / vol,
                 nu * gpu::velocityGradientSq(gg, *D, *red));
        };
        while (t < c.tEnd)
        {
            if (t >= nextOut) { probe(); nextOut += outEvery; }
            if (frames && frames->wants(t)) frames->write(gg, *D, t, g.size());
            const double dt = ts->suggestDt(*D, c.tEnd - t);
            ts->step(*D, dt);
            t += dt;
            ++steps;
        }
        probe();
        std::fprintf(stderr, "done: %ld steps (hip)\n", steps);
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
    if (!haveGpu(c)) return die(exitNoGpu, "no HIP device (dns hit is GPU-only)");
#if DNS_WITH_HIP
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

    auto ts = gpu::TimeStepper::make(g, {nu, 0.9});
    auto D = gpu::DeviceField::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!ts || !D || !red) return die(exitResource, "GPU init failed");
    (void)gpu::uploadField(U, *D);
    const auto gg = gpu::GpuGrid::from(g);

    const LinearForcing forcing{epsF, 1.5 / 5.0}; // kFloor = k_stationary/5
    const double tEnd = (c.spin + c.stats) * tE, tSpin = c.spin * tE;
    const std::string statsCsv = c.outPrefix + "_stats.csv";

    std::optional<PeriodicFrames> frames;
    if (c.emit.contains(Emit::Slices))
    {
        auto f = PeriodicFrames::make(c, g);
        if (!f) return die(exitOutput, f.error());
        frames = std::move(*f);
    }

    double t = 0.0, dt = 0.0;
    long steps = 0;
    int spectrumIdx = 0, snapIdx = 0;
    double nextSpectrum = tSpin, nextSnap = tSpin;
    double epsSum = 0.0, kSum = 0.0, reSum = 0.0, wSum = 0.0;

    while (t < tEnd)
    {
        const auto mean = gpu::meanVelocity(gg, *D, *red);
        const double kPrime = gpu::fluctuationEnergy(gg, *D, *red);
        const double A = epsF / (2.0 * std::max(kPrime, forcing.kFloor));
        ts->setForcing({A, {mean.x, mean.y, mean.z}});
        if (steps % 100 == 0)
            gpu::subtractMean(gg, gpu::Real(mean.x), gpu::Real(mean.y),
                              gpu::Real(mean.z), *D);

        if (frames && frames->wants(t)) frames->write(gg, *D, t, g.size());
        dt = ts->suggestDt(*D, tEnd - t);
        ts->step(*D, dt);
        t += dt;
        ++steps;

        if (steps % 10 == 0)
        {
            FlowStats fs{};
            fs.kPrime = gpu::fluctuationEnergy(gg, *D, *red);
            fs.uPrime = std::sqrt(2.0 * fs.kPrime / 3.0);
            fs.epsVisc = nu * gpu::velocityGradientSq(gg, *D, *red);
            if (fs.epsVisc > 0)
            {
                fs.eta = std::pow(nu * nu * nu / fs.epsVisc, 0.25);
                fs.reLambda =
                    fs.uPrime * fs.uPrime * std::sqrt(15.0 / (nu * fs.epsVisc));
                fs.hOverEta = double(g.hx) / fs.eta;
            }
            fs.courantPerDt = gpu::maxCourantPerDt(gg, *D, *red);
            fs.maxDiv = gpu::maxAbsDivergence(gg, *D, *red);
            (void)appendStatsCsv(statsCsv, t, dt, A, fs);
            if (fs.maxDiv * g.hx > 1e-3)
                std::fprintf(stderr, "WARN t=%.3f: maxDiv*h = %.3e\n", t,
                             fs.maxDiv * g.hx);
            if (t > tSpin)
            {
                epsSum += dt * 10 * fs.epsVisc;
                kSum += dt * 10 * fs.kPrime;
                reSum += dt * 10 * fs.reLambda;
                wSum += dt * 10;
            }
        }

#if DNS_WITH_FFTW
        if (c.emit.contains(Emit::Spectrum) && t >= nextSpectrum)
        {
            (void)gpu::downloadField(*D, U);
            std::ofstream f(c.outPrefix + "_spec_" + std::to_string(spectrumIdx++)
                            + ".csv");
            f << "k,E\n";
            const auto shellE = energySpectrum(g, U);
            for (std::size_t s = 1; s < shellE.size(); ++s)
                f << s << ',' << shellE[s] << '\n';
            nextSpectrum += 0.25 * tE;
        }
#endif
        if (c.emit.contains(Emit::Snapshot) && t >= nextSnap)
        {
            (void)gpu::downloadField(*D, U);
            const std::string p =
                c.outPrefix + "_snap_" + std::to_string(snapIdx++);
            std::ofstream raw(p + ".raw", std::ios::binary);
            auto dump = [&](const Scalar& q) {
                std::vector<float> tmp(q.begin(), q.end());
                raw.write(reinterpret_cast<const char*>(tmp.data()),
                          std::streamsize(tmp.size() * sizeof(float)));
            };
            dump(U.u);
            dump(U.v);
            dump(U.w);
            std::ofstream meta(p + ".json");
            meta << "{\"n\":" << g.nx << ",\"tBox\":" << t << ",\"tSI\":"
                 << d.units().timeToSI(t) << ",\"lBoxSI\":" << d.lBoxSI
                 << ",\"layout\":\"u,v,w float32 i-fastest MAC\"}\n";
            nextSnap += tE;
        }
    }

    const double epsMean = epsSum / wSum, kMean = kSum / wSum;
    const double uPrime = std::sqrt(2.0 * kMean / 3.0);
    std::printf("# %ld steps.  Stationary averages over %.1f Te:\n", steps, c.stats);
    std::printf("#   eps_visc/eps_f = %.4f   u'/target = %.4f   Re_lambda = %.1f "
                "(design %.1f)\n",
                epsMean / epsF, uPrime, reSum / wSum, d.reLambda);
    std::printf("#   SI: u' = %.2f m/s  eps = %.4e W/kg\n",
                d.units().velocityToSI(uPrime), d.units().epsToSI(epsMean));
    return 0;
#else
    return exitNoGpu;
#endif
}

// --- bench -----------------------------------------------------------------

int runBench(const Case& c)
{
    if (!haveGpu(c)) return die(exitNoGpu, "no HIP device (dns bench is GPU-only)");
#if DNS_WITH_HIP
    const auto& g = c.grid;
    auto U = VectorField::zeros(g);
    taylorGreen3D(g, U);

    auto ts = gpu::TimeStepper::make(g, {1e-3, 0.9});
    auto D = gpu::DeviceField::make(g.size());
    if (!ts || !D) return die(exitResource, "GPU init failed");
    (void)gpu::uploadField(U, *D);
    ts->setForcing({0.1, {0, 0, 0}}); // exercise the forcing path too

    const double dt = 0.2 * (2 * pi / g.nx); // fixed benchmark dt, CFL-ish
    for (int s = 0; s < 3; ++s) ts->step(*D, dt); // warm-up (incl. FFT tuning)
    (void)hipDeviceSynchronize();

    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < c.steps; ++s) ts->step(*D, dt);
    (void)hipDeviceSynchronize();
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count()
                      / c.steps;

    std::size_t freeB = 0, totalB = 0;
    (void)hipMemGetInfo(&freeB, &totalB);
    std::printf("n=%d  %.2f ms/step  (%.1f Mcells, %.1f Mcell-updates/s, "
                "%.2f GiB VRAM in use)\n",
                g.nx, ms, g.size() / 1e6, g.size() / (ms * 1e3),
                double(totalB - freeB) / (1 << 30));
    return 0;
#else
    return exitNoGpu;
#endif
}

} // namespace

int main(int argc, char** argv)
{
    // Phase 1: parse.
    auto raw = cli::parse(argc, argv);
    if (!raw)
    {
        std::fputs(raw.error().message.c_str(),
                   raw.error().helpRequested ? stdout : stderr);
        return raw.error().helpRequested ? 0 : 1;
    }

    // Phase 2: validate.
    auto c = validate(*raw);
    if (!c)
    {
        std::fputs(format(c.error()).c_str(), stderr);
        return c.error().code;
    }
    if (c->dryRun)
    {
        std::fputs(c->describe().c_str(), stdout);
        return 0;
    }

    // Phases 3 and 4.
    switch (c->sub)
    {
    case Subcommand::Flow: return runFlow(*c, *raw);
    case Subcommand::Tgv: return runTgv(*c);
    case Subcommand::Hit: return runHit(*c);
    case Subcommand::Bench: return runBench(*c);
    }
    return 1;
}
