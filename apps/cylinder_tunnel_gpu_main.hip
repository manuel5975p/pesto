// Flow around a z-aligned circular cylinder in a numerical flow channel (GPU),
// quasi-2D: only a few cells in z with FREE-SLIP z-walls (mirror u,v; w = 0),
// which are exact for a z-invariant flow -- the run is effectively a 2D
// circle simulation. --nz 1 (or 0) runs natively 2D instead: nz == 1, no z
// ghosts, the degenerate z axis stays periodic (exact for z-invariant flow).
// Laminar Dirichlet inflow, convective outflow with exact
// mass-flux correction, open free-stream lateral y-walls, sanitized cylinder
// cut-cell body. Pressure Poisson is PCG preconditioned by the exact DCT
// (NeumannBox) inverse Laplacian; drag/lift come from the discrete
// aperture-pressure surface integral plus the masked-link skin-friction tally
// (x/y links only: the free-slip z ghosts carry no viscous flux).
// The grid carries a one-cell ghost ring: interior is n x n x nzInt cells.
// --open replaces the channel with a fully periodic open far field (fringe-
// region method): no closeBoxBoundary, FFT-periodic pressure, and per stage a
// cosine-ramped sponge band along every x/y boundary relaxing u toward the
// free stream (z stays plain periodic -- the flow is z-invariant).
// Usage: cylinder_tunnel_gpu [n] [--nz nzInt | --nz 1 for native 2D]
//                            [--re ReD] [--dr DRatio] [--open]
//                            [--ft flowThroughs] [--fdt frameDt] [--tol rms] [--cfl c]
//                            [--les Cs (Smagorinsky, 2D only, 0 = off)]
//                            [--out prefix]
#include "dns/output.hpp"
#include "dns/gpu/runner_incomp.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/ghostfill.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/les.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/bodyforce.hpp"
#include "dns/gpu/tunnel.hpp"

#include <hip/hip_runtime.h>

#include <filesystem>

#include <algorithm>
#include <array>
#include <charconv>
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
using gpu::Real;

double parseOr(const char* s, double fallback)
{
    double v{};
    const std::string_view sv(s);
    auto [q, ec] = std::from_chars(sv.begin(), sv.end(), v);
    return ec == std::errc{} && q == sv.end() ? v : fallback;
}

} // namespace

// --- 2D golf ball: circle minus N dimple disks around the circumference -----
// Dimple footprint radius 8h, depth 4h -- near real golf proportions when
// D ~ 256h and still grid-resolved. sdf < 0 in the solid, z-invariant.
struct GolfDisk
{
    CylinderZ c;
    std::vector<std::array<double, 2>> q; // dimple-disk centers
    double rd, depth;

    static GolfDisk make(const CylinderZ& c, int nDimples, double h)
    {
        using std::numbers::pi;
        const double depth = 4.0 * h, a = 8.0 * h;
        GolfDisk b{c, {}, (a * a + depth * depth) / (2.0 * depth), depth};
        const double qr = c.r + b.rd - depth;
        for (int i = 0; i < nDimples; ++i)
        {   // deterministic irrational phase: no dimple sits axis-aligned
            const double th = 2.0 * pi * i / nDimples + 0.317;
            b.q.push_back({c.cx + qr * std::cos(th), c.cy + qr * std::sin(th)});
        }
        return b;
    }

    // Solid = disk \ union(dimple disks): sdf = max(disk, -dimples).
    double sdf(double x, double y) const
    {
        double phi = std::hypot(x - c.cx, y - c.cy) - c.r;
        for (const auto& p : q)
            phi = std::max(phi, rd - std::hypot(x - p[0], y - p[1]));
        return phi;
    }
};

// Cut geometry by supersampling the sdf on the dimpled shell band: lambda (and
// az) from 8x8 cell samples, ax/ay from 8 samples along the face segment; one
// x-y layer, replicated over k. Sampling noise ~1/8 of a face is absorbed by
// sanitizeCut and the first-order boundary scheme (cf. the eye app's lambda).
CutGeometry makeGolfCut(const Grid& g, const GolfDisk& b)
{
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    const double halfDiag = 0.5 * std::hypot(g.hx, g.hy);
    const double lo = b.c.r - b.depth - halfDiag, hi = b.c.r + halfDiag;
    auto dist = [&](double x, double y) {
        return std::hypot(x - b.c.cx, y - b.c.cy);
    };
    auto cellFrac = [&](double x0, double y0) {
        int m = 0;
        for (int bb = 0; bb < 8; ++bb)
            for (int a = 0; a < 8; ++a)
                if (b.sdf(x0 + (a + 0.5) * g.hx / 8,
                          y0 + (bb + 0.5) * g.hy / 8) > 0.0)
                    ++m;
        return m / 64.0;
    };
    auto segFrac = [&](double x0, double y0, double ex, double ey) {
        int m = 0;
        for (int a = 0; a < 8; ++a)
            if (b.sdf(x0 + (a + 0.5) * ex / 8, y0 + (a + 0.5) * ey / 8) > 0.0)
                ++m;
        return m / 8.0;
    };
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
        {
            const double x0 = i * g.hx, y0 = j * g.hy;
            auto classify = [&](double d, auto sample) {
                return d >= hi ? 1.0 : d <= lo ? 0.0 : sample();
            };
            const double lam =
                classify(dist(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy),
                         [&] { return cellFrac(x0, y0); });
            const double ax = classify(dist(x0, y0 + 0.5 * g.hy),
                                       [&] { return segFrac(x0, y0, 0, g.hy); });
            const double ay = classify(dist(x0 + 0.5 * g.hx, y0),
                                       [&] { return segFrac(x0, y0, g.hx, 0); });
            for (int k = 0; k < g.nz; ++k)
            {
                const auto id = g.idx(i, j, k);
                cut.lambda[id] = lam;
                cut.ax[id] = ax;
                cut.ay[id] = ay;
                cut.az[id] = lam;
            }
        }
    return cut;
}

int main(int argc, char** argv)
{
    using std::numbers::pi;

    int n = 1024, nzInt = 8;
    double reD = 60.0, flowThroughs = 20.0, dRatio = 8.0, frameDt = 0.15;
    double cfl = 0.8;
    double tol = 1e-4;
    double csLes = 0.0; // Smagorinsky constant; 0 = no LES model
    int golf = 0;       // number of dimples; 0 = smooth circle
    bool open = false;  // fully periodic open far field (fringe sponge)
    std::string prefix = "cylinder_tunnel_gpu";
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--re") reD = next(reD);
        else if (s == "--golf") golf = int(next(0));
        else if (s == "--nz") nzInt = int(next(nzInt));
        else if (s == "--dr") dRatio = next(dRatio);
        else if (s == "--ft") flowThroughs = next(flowThroughs);
        else if (s == "--cfl") cfl = next(cfl);
        else if (s == "--les") csLes = next(csLes);
        else if (s == "--open") open = true;
        else if (s == "--fdt") frameDt = next(frameDt);
        else if (s == "--tol") tol = next(tol);
        else if (s == "--out" && a + 1 < argc) prefix = argv[++a];
        else n = int(parseOr(argv[a], n));
    }
    if (!gpu::deviceAvailable())
    {
        std::fprintf(stderr, "no HIP device\n");
        return 1;
    }

    // Interior n x n x nzInt cells of size h; one ghost cell on every side.
    // nzInt <= 1 selects native 2D: nz == 1 with no z ghosts (periodic z).
    const double L = 2 * pi, uInf = 1.0;
    const double h = L / n;
    const int nx = n + 2, nz = nzInt <= 1 ? 1 : nzInt + 2;
    const int nzCells = nz == 1 ? 1 : nzInt; // interior z extent in cells
    const auto g = Grid::make(nx, nx, nz, nx * h, nx * h, nz * h);

    // Axis in grid coordinates: physical position + the ghost offset h.
    // Off-node on purpose (arbitrary cuts); sanitizeCut handles the tiny cells.
    const double D = L / dRatio;
    const CylinderZ cyl{h + L / 3.0 + 0.31 * h, h + L / 2.0 + 0.17 * h, D / 2.0};
    const GolfDisk ball = GolfDisk::make(cyl, std::max(golf, 1), h);
    auto cut = golf > 0 ? makeGolfCut(g, ball) : CutGeometry::make(g, cyl);
    if (golf > 0)
        std::printf("# golf: %d dimples, footprint 8h, depth 4h (%.3f D)\n",
                    golf, ball.depth / D);
    const auto st = sanitizeCut(g, cut); // before the ring: body cells only
    if (!open) closeBoxBoundary(g, cut); // open: ring cells stay periodic fluid

    const double nu = uInf * D / reD;
    std::printf("# cylinder_tunnel_gpu n=%d nz=%d%s Re_D=%g nu=%.4e D=%.4f (%.1f cells)"
                " les=%g open=%d\n",
                n, nzCells, nz == 1 ? " (native 2D)" : "", reD, nu, D, D / h, csLes,
                int(open));
    if (csLes > 0 && nz != 1)
    {
        std::fprintf(stderr, "--les is 2D only (nz == 1)\n");
        return 1;
    }
    std::printf("# sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed,
                st.minLambda, st.minAperture);

    // Clean laminar free stream; deterministic z-uniform trip one D behind the
    // cylinder to break the symmetric-wake transient early (Re 60 sheds anyway).
    auto U = VectorField::zeros(g);
    for (auto& x : U.u) x = uInf;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < nx; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const double x = i * h, y = (j + 0.5) * h;
                if (x > cyl.cx + cyl.r && x < cyl.cx + cyl.r + D
                    && std::abs(y - cyl.cy) < cyl.r)
                    U.v[g.idx(i, j, k)] = 0.05 * uInf;
            }

    const double tEnd = flowThroughs * L / uInf;
    const double qRef = 0.5 * uInf * uInf * D * (nzCells * h); // frontal rectangle

    // Every run artefact goes through the one sink; the ghost ring is cropped
    // out by the (i0, j0) = (1, 1) window so frames stay n x n interior cells.
    const int kc = nz == 1 ? 0 : nz / 2;
    RunMeta rm;
    rm.nx = n; rm.ny = n; rm.nz = nzCells;
    rm.hx = h; rm.hy = h; rm.hz = h;
    rm.D = D; rm.re = reD; rm.frameDt = frameDt; rm.cs = csLes;
    rm.les = csLes > 0;
    rm.body = golf > 0 ? "cylinder-golf" : "cylinder";
    rm.farField = open ? "fringe" : "tunnel";
    rm.physics = "incompressible";

    FrameGeometry fg;
    fg.mask.assign(g.size(), 1.0f);
    for (std::size_t c = 0; c < g.size(); ++c) fg.mask[c] = float(cut.lambda[c]);
    fg.ic = nx / 2; fg.jc = nx / 2; fg.kc = kc; fg.D = D;
    fg.i0 = 1; fg.j0 = 1; fg.k0 = nz == 1 ? 0 : 1;
    fg.w = n; fg.h = n; fg.d = nzCells;

    RunOutput::Config oc;
    oc.prefix = prefix;
    oc.kinds = {Emit::Stats, Emit::Bin};
    if (nz == 1) oc.kinds.insert(Emit::Png);
    oc.fdt = frameDt;
    auto out = RunOutput::make(oc, rm, g, fg);
    if (!out) {
        std::fprintf(stderr, "output init failed: %s\n", out.error().c_str());
        return 4;
    }
    // The runner declares the stats series; its column order is the contract
    // the val_* tests parse by name.
    gpu::IncompRun rr;
    rr.nu = nu;
    rr.uInf = uInf;
    rr.qRef = qRef;
    rr.D = D;
    rr.cfl = cfl;
    rr.tEnd = tEnd;
    rr.tol = tol;
    rr.les = csLes;
    rr.fringe = open;
    rr.fringeAxes = 2; // z-invariant wake: a z band would sponge the whole wake
    rr.box = gpu::bodyBox(gpu::GpuGrid::from(g), cyl.cx, cyl.cy, 0.0, cyl.r);
    rr.box.k0 = gpu::GpuGrid::from(g).kInLo();
    rr.box.k1 = gpu::GpuGrid::from(g).kInHi();

    auto sdf = [&](double x, double y, double) {
        if (golf > 0) return ball.sdf(x, y);
        return std::hypot(x - cyl.cx, y - cyl.cy) - cyl.r;
    };
    auto res = gpu::runIncompressible(g, cut, sdf, U, rr, *out);
    if (!res) {
        std::fprintf(stderr, "%s\n", res.error().message.c_str());
        return res.error().code;
    }
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", res->steps,
                out->frames(), res->msPerStep);
    return 0;
}
