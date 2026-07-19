// 3D compressible flow around an arbitrary triangle mesh (STL / glTF / GLB):
// the compressible analogue of mesh_tunnel_gpu. The already-3D compressible
// solver (Kennedy-Gruber central fluxes, SSP-RK3, optional 3D Smagorinsky LES)
// steps the state; the sharp ghost-fill wall (DeviceCompGhost, 3D-general)
// imposes sub-cell no-slip on the exact MeshSdf and tallies the momentum-flux
// drag/lift/side force. Far field: NSCBC/LODI characteristic boundaries on
// all six faces (DeviceCompNscbc, rhs overwrite inside RK3) -- stiff
// freestream inflow, subsonic outflow with the Yoo-Im transverse correction,
// and characteristic lateral far-field faces (soft pressure anchor, normal
// velocity free): no sponges, no walls. Quiet start: --ramp T > 0 starts from
// rest and raises the freestream by exact Galilean boosts (no start-up
// acoustic pulse). Smooth Mach only.
// Usage: comp_mesh_tunnel_gpu model.{stl,gltf,glb} [n] [--ny N] [--nz N]
//          [--re ReD] [--mach M] [--scale s (0 = fit max extent to L/dr)]
//          [--dr DRatio] [--pitch deg] [--yaw deg] [--robust] [--les Cs]
//          [--nsbeta B] [--ramp T] [--ft flowThroughs] [--fdt frameDt]
//          [--cfl C] [--out prefix]
#include "dns/comp.hpp"
#include "dns/cutcell.hpp"
#include "dns/geometry.hpp"
#include "dns/meshsdf.hpp"
#include "dns/trimesh.hpp"
#include "dns/webframes.hpp"

#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_far.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/device_util.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <numbers>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace dns;
using gpu::GpuGrid;
using gpu::Real;

double parseOr(const char* s, double fallback)
{
    double v{};
    const std::string_view sv(s);
    auto [q, ec] = std::from_chars(sv.begin(), sv.end(), v);
    return ec == std::errc{} && q == sv.end() ? v : fallback;
}

bool hasMeshExt(std::string_view s)
{
    auto ends = [&](std::string_view e) {
        return s.size() > e.size() && s.substr(s.size() - e.size()) == e;
    };
    return ends(".stl") || ends(".gltf") || ends(".glb");
}

// Quiet-start freestream schedule: smoothstep 0 -> 1 over [0, tRamp].
double rampFrac(double t, double tRamp)
{
    if (tRamp <= 0 || t >= tRamp) return 1.0;
    const double s = t / tRamp;
    return s * s * (3 - 2 * s);
}

} // namespace

int main(int argc, char** argv)
{
    using std::numbers::pi;
    const auto progT0 = std::chrono::steady_clock::now();

    int n = 128, nyArg = -1, nzArg = -1;
    double reD = 300.0, mach = 0.2, scaleArg = 0.0, dRatio = 6.0;
    double pitchDeg = 0.0, yawDeg = 0.0, csLes = 0.0;
    double flowThroughs = 6.0, frameDt = 0.5, cfl = 0.4, tRamp = 2.0, kick = 0.05;
    double webDt = 0.25; // live browser slice cadence (cheap slab download)
    double nsBeta = 0.5; // Yoo-Im transverse compensation (< 0 = Mach)
    int nsStrip = 0; // thin sponge strip inside NSCBC (0 = off, pure six-face)
    bool robust = false, sliceMode = false;
    std::string meshPath, prefix = "comp_mesh_tunnel_gpu";
    for (int a = 1; a < argc; ++a) {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--ny") nyArg = int(next(nyArg));
        else if (s == "--nz") nzArg = int(next(nzArg));
        else if (s == "--re") reD = next(reD);
        else if (s == "--mach") mach = next(mach);
        else if (s == "--scale") scaleArg = next(scaleArg);
        else if (s == "--dr") dRatio = next(dRatio);
        else if (s == "--pitch") pitchDeg = next(pitchDeg);
        else if (s == "--yaw") yawDeg = next(yawDeg);
        else if (s == "--robust") robust = true;
        else if (s == "--slices") sliceMode = true;
        else if (s == "--les") csLes = next(csLes);
        else if (s == "--nsbeta") nsBeta = next(nsBeta);
        else if (s == "--nsstrip") nsStrip = int(next(nsStrip));
        else if (s == "--ramp") tRamp = next(tRamp);
        else if (s == "--kick") kick = next(kick);
        else if (s == "--ft") flowThroughs = next(flowThroughs);
        else if (s == "--fdt") frameDt = next(frameDt);
        else if (s == "--wdt") webDt = next(webDt);
        else if (s == "--cfl") cfl = next(cfl);
        else if (s == "--out" && a + 1 < argc) prefix = argv[++a];
        else if (hasMeshExt(s)) meshPath = s;
        else n = int(parseOr(argv[a], n));
    }
    if (meshPath.empty()) {
        std::fprintf(stderr,
                     "usage: comp_mesh_tunnel_gpu model.{stl,gltf,glb} [n] "
                     "[--ny N] [--nz N] [--re ReD] [--mach M] [--scale s] "
                     "[--dr DRatio] [--pitch deg] [--yaw deg] [--robust] "
                     "[--slices] [--les Cs] [--nsbeta B] [--nsstrip N] "
                     "[--ramp T] [--ft T] "
                     "[--wdt dt] "
                     "[--fdt dt] [--cfl C] [--out prefix]\n");
        return 1;
    }
    if (!gpu::deviceAvailable()) {
        std::fprintf(stderr, "no HIP device\n");
        return 1;
    }
    const int nx = n;
    const int ny = nyArg > 0 ? nyArg : n;
    const int nz = nzArg > 0 ? nzArg : n; // TRUE 3D by default

    // Cubic-ish box; auto scale fits the mesh's largest extent to D0 = L/dRatio.
    const double L = 2 * pi, h = L / nx;
    const double uInf = 1.0, rhoInf = 1.0, gamma = 1.4;
    const double pInf = rhoInf * uInf * uInf / (gamma * mach * mach);
    const double D0 = L / dRatio;
    // Body ~1/3 downstream from inflow, centered laterally (off-node like the
    // incompressible mesh app so the shedding instability trips).
    const std::array<double, 3> C{L / 3.0 + 0.31 * h, ny * h / 2.0 + 0.17 * h,
                                  nz * h / 2.0 + 0.23 * h};

    auto meshOr = loadMesh(meshPath);
    if (!meshOr) {
        std::fprintf(stderr, "%s\n", meshOr.error().c_str());
        return 1;
    }
    TriMesh mesh = std::move(*meshOr);
    const Aabb bb0 = bounds(mesh);
    const double ext0 = std::max({bb0.hi[0] - bb0.lo[0], bb0.hi[1] - bb0.lo[1],
                                  bb0.hi[2] - bb0.lo[2]});
    if (!(ext0 > 0)) {
        std::fprintf(stderr, "degenerate mesh bounds\n");
        return 1;
    }
    const double scale = scaleArg > 0 ? scaleArg : D0 / ext0;
    transform(mesh, scale, pitchDeg * pi / 180.0, yawDeg * pi / 180.0, C);
    const Aabb bb = bounds(mesh);
    const double D = std::max({bb.hi[0] - bb.lo[0], bb.hi[1] - bb.lo[1],
                               bb.hi[2] - bb.lo[2]});
    const std::size_t nTris = mesh.tris.size();

    const auto sdfT0 = std::chrono::steady_clock::now();
    const MeshSdf sdf(std::move(mesh), robust);
    if (!sdf.watertight())
        std::printf("# WARNING: mesh not watertight%s\n",
                    robust ? "" : "; consider --robust");
    auto sdfFn = [&](double x, double y, double z) { return sdf(x, y, z); };

    const auto g = Grid::make(nx, ny, nz, nx * h, ny * h, nz * h);
    const auto gg = GpuGrid::from(g);

    // Exact frontal area from supersampled cut apertures (drag reference only;
    // the solver itself never uses the cut geometry, only the ghost wall).
    auto cutR = sampleCut(g, [&](double x, double y, double z) { return sdf(x, y, z); });
    if (!cutR)
    {
        std::fprintf(stderr, "cut geometry: %s\n", cutR.error().c_str());
        return 3;
    }
    auto cut = std::move(*cutR);
    const auto sst = sanitizeCut(g, cut);
    double aFront = 0.0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            double amin = 1.0;
            for (int i = 0; i < nx; ++i)
                amin = std::min(amin, cut.ax[g.idx(i, j, k)]);
            aFront += (1.0 - amin) * g.hy * g.hz;
        }
    const double geomSec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - sdfT0)
            .count();
    if (aFront <= 0) {
        std::fprintf(stderr, "mesh produced no solid cells (too small?)\n");
        return 1;
    }

    const double mu = rhoInf * uInf * D / reD;
    std::printf("# comp_mesh_tunnel_gpu %s nx=%d ny=%d nz=%d Re_D=%g M=%g "
                "mu=%.4e D=%.4f (%.1f cells) csLes=%g nscbc-open ramp=%g "
                "nsstrip=%d\n",
                meshPath.c_str(), nx, ny, nz, reD, mach, mu, D, D / h, csLes,
                tRamp, nsStrip);
    std::printf("# mesh: tris=%zu scale=%.4g pitch=%g yaw=%g robust=%d "
                "watertight=%d Afront=%.4f geomBuild=%.1fs\n",
                nTris, scale, pitchDeg, yawDeg, int(robust),
                int(sdf.watertight()), aFront, geomSec);
    std::printf("# sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                sst.cellsSolidified, sst.facesClosed, sst.pocketsClosed,
                sst.minLambda, sst.minAperture);

    auto dev = gpu::DeviceComp::make(g, gamma, mu, csLes);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    // Body bounds -> ghost skips the mesh sdf far from the airframe (O(bbox)
    // evals, not O(grid)): the ghost-wall precompute cost at high n.
    const double bbox[6] = {bb.lo[0], bb.lo[1], bb.lo[2],
                            bb.hi[0], bb.hi[1], bb.hi[2]};
    auto ghost = gpu::DeviceCompGhost::make(g, sdfFn, gamma, rhoInf, pInf, bbox);
    if (!dev || !ds || !red || !ghost) {
        std::fprintf(stderr, "GPU alloc failed\n");
        return 1;
    }

    // Far field: NSCBC characteristic boundaries on all six faces (rhs
    // overwrite inside RK3) -- no sponge, no walls. Ghost wall = stage hook.
    gpu::DeviceCompNscbc::Config nc;
    nc.rhoInf = rhoInf;
    nc.uInf = uInf;
    nc.pInf = pInf;
    nc.gamma = gamma;
    nc.betaTrans = nsBeta;
    nc.openY = true;
    nc.openZ = nz > 1;
    auto nsc = gpu::DeviceCompNscbc::make(g, nc);
    if (!nsc) {
        std::fprintf(stderr, "far-field init failed\n");
        return 1;
    }
    // Optional production hybrid: a thin sponge strip inside the NSCBC rows
    // absorbs residual reflections beyond Re ~ 800 (default off = pure six-face
    // NSCBC, byte-identical to the validated path). Applied before the ghost.
    std::expected<gpu::DeviceCompFar, hipError_t> far =
        std::unexpected(hipErrorInvalidValue);
    if (nsStrip > 0) {
        gpu::DeviceCompFar::Config fc;
        fc.rhoInf = rhoInf;
        fc.uInf = uInf;
        fc.pInf = pInf;
        fc.gamma = gamma;
        fc.ns = nsStrip;
        fc.sigMax = 0.5;
        fc.openY = true;
        fc.openZ = nz > 1;
        far = gpu::DeviceCompFar::make(g, fc);
        if (!far) {
            std::fprintf(stderr, "sponge strip init failed\n");
            return 1;
        }
    }
    auto applyBc = [&](gpu::DeviceCompState& s) {
        if (far) far->apply(s);
        ghost->apply(s);
    };
    dev->setStageHook(applyBc);
    dev->setRhsHook([&](const gpu::DeviceCompState& st, gpu::DeviceCompState& k) {
        nsc->apply(*dev, st, k);
    });

    // IC. Quiet start (tRamp > 0): everything at rest -- exactly consistent
    // with the no-slip body, so nothing radiates; the freestream is then
    // raised by uniform Galilean boosts (silent) over [0, tRamp].
    // tRamp == 0: impulsive freestream with the legacy 3h taper to rest at the
    // surface (bbox-culled sdf evals).
    auto host = CompState::zeros(g);
    const double u0 = tRamp > 0 ? 0.0 : uInf;
    const double taper = 3.0 * h;
    auto farFromBody = [&](double x, double y, double z) {
        double d2 = 0.0;
        const double p[3] = {x, y, z};
        for (int a = 0; a < 3; ++a) {
            const double e = p[a] < bb.lo[a] ? bb.lo[a] - p[a]
                           : p[a] > bb.hi[a] ? p[a] - bb.hi[a] : 0.0;
            d2 += e * e;
        }
        return d2 > taper * taper;
    };
    // Deterministic solenoidal trip (quiet start only): a weak Gaussian
    // vortex one D behind the body seeds the wake instability -- a quiet
    // start alone is too symmetric to shed. Divergence-free, so silent.
    const double kA = tRamp > 0 ? kick * uInf : 0.0, kw = 0.25 * D;
    const double kx = bb.hi[0] + 0.5 * D, ky = C[1] + 0.25 * D;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double x = (i + 0.5) * h, y = (j + 0.5) * h, z = (k + 0.5) * h;
                double u = u0, v = 0.0;
                if (u0 != 0.0 && !farFromBody(x, y, z)) {
                    const double f = std::clamp(sdf(x, y, z) / taper, 0.0, 1.0);
                    u = u0 * f * f * (3 - 2 * f);
                }
                if (kA != 0.0) {
                    const double dx = x - kx, dy = y - ky;
                    const double e =
                        kA * std::exp(-(dx * dx + dy * dy) / (kw * kw));
                    u += 2 * dy / kw * e;
                    v -= 2 * dx / kw * e;
                }
                host.rho[c] = rhoInf;
                host.mx[c] = rhoInf * u;
                host.my[c] = rhoInf * v;
                host.E[c] = pInf / (gamma - 1.0) + 0.5 * rhoInf * (u * u + v * v);
            }
    (void)ds->upload(host);
    nsc->setFreestream(u0);
    if (far) far->setFreestream(u0);
    applyBc(*ds);

    const double tEnd = tRamp + flowThroughs * L / uInf;
    const double qRef = 0.5 * rhoInf * uInf * uInf * aFront;

    std::ofstream stats(prefix + "_stats.csv");
    stats << "t,dt,cd,cl,cs,wmax\n";
    auto writeJson = [&](int frames, double ms) {
        std::ofstream(prefix + ".json")
            << "{\"nx\":" << nx << ",\"ny\":" << ny << ",\"nz\":" << nz
            << ",\"L\":" << L << ",\"D\":" << D << ",\"reD\":" << reD
            << ",\"mach\":" << mach << ",\"frameDt\":" << frameDt
            << ",\"frames\":" << frames << ",\"csLes\":" << csLes
            << ",\"cx\":" << C[0] / h << ",\"cy\":" << C[1] / h
            << ",\"cz\":" << C[2] / h << ",\"R\":" << 0.5 * D / h
            << ",\"open\":1,\"nscbc\":1,\"nsstrip\":" << nsStrip
            << ",\"ramp\":" << tRamp
            << ",\"slices\":" << (sliceMode ? 1 : 0) << ",\"kc\":" << nz / 2
            << ",\"jc\":" << ny / 2 << ",\"ic\":" << nx / 2
            << ",\"mesh\":\"" << meshPath << "\",\"msPerStep\":" << ms << "}\n";
    };
    writeJson(0, 0.0);

    // Center-plane indices for slice output.
    const int kc = nz / 2, jc = ny / 2, ic = nx / 2;

    // Solid mask: full-grid lambda for the 3D viewer, or the three center-plane
    // cross-sections (nx*ny, nx*nz, ny*nz) for the slice movies.
    auto writeLamPlanes = [&]() {
        auto dump = [&](const std::string& suf, auto sel, int a, int b) {
            std::ofstream f(prefix + suf, std::ios::binary);
            std::vector<float> buf(std::size_t(a) * b);
            std::size_t q = 0;
            for (int j = 0; j < b; ++j)
                for (int i = 0; i < a; ++i) buf[q++] = float(cut.lambda[sel(i, j)]);
            f.write(reinterpret_cast<const char*>(buf.data()),
                    std::streamsize(buf.size() * sizeof(float)));
        };
        dump("_xy_lam.bin", [&](int i, int j) { return g.idx(i, j, kc); }, nx, ny);
        dump("_xz_lam.bin", [&](int i, int k) { return g.idx(i, jc, k); }, nx, nz);
        dump("_yz_lam.bin", [&](int j, int k) { return g.idx(ic, j, k); }, ny, nz);
    };
    if (sliceMode) {
        writeLamPlanes();
    } else {
        std::ofstream lam(prefix + "_lam.bin", std::ios::binary);
        std::vector<float> lbuf(g.size());
        std::size_t q = 0;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    lbuf[q++] = float(cut.lambda[g.idx(i, j, k)]);
        lam.write(reinterpret_cast<const char*>(lbuf.data()),
                  std::streamsize(lbuf.size() * sizeof(float)));
    }

    // Per-cell |curl u| and |u| (3D central differences, periodic wrap).
    std::vector<Real> hr(g.size()), hmx(g.size()), hmy(g.size()), hmz(g.size());
    std::vector<double> uu(g.size()), vv(g.size()), ww(g.size());
    auto downloadPrims = [&]() {
        (void)ds->rho.download(hr.data());
        (void)ds->mx.download(hmx.data());
        (void)ds->my.download(hmy.data());
        (void)ds->mz.download(hmz.data());
        for (std::size_t c = 0; c < g.size(); ++c) {
            const double r = hr[c] > 0 ? hr[c] : 1.0;
            uu[c] = double(hmx[c]) / r;
            vv[c] = double(hmy[c]) / r;
            ww[c] = double(hmz[c]) / r;
        }
    };
    auto vortSpeed = [&](int i, int j, int k) -> std::pair<float, float> {
        const std::size_t c = g.idx(i, j, k);
        const double dwdy = (ww[g.idx(i, j + 1, k)] - ww[g.idx(i, j - 1, k)]) / (2 * h);
        const double dvdz = (vv[g.idx(i, j, k + 1)] - vv[g.idx(i, j, k - 1)]) / (2 * h);
        const double dudz = (uu[g.idx(i, j, k + 1)] - uu[g.idx(i, j, k - 1)]) / (2 * h);
        const double dwdx = (ww[g.idx(i + 1, j, k)] - ww[g.idx(i - 1, j, k)]) / (2 * h);
        const double dvdx = (vv[g.idx(i + 1, j, k)] - vv[g.idx(i - 1, j, k)]) / (2 * h);
        const double dudy = (uu[g.idx(i, j + 1, k)] - uu[g.idx(i, j - 1, k)]) / (2 * h);
        const double wx = dwdy - dvdz, wy = dudz - dwdx, wz = dvdx - dudy;
        return {float(std::sqrt(wx * wx + wy * wy + wz * wz)),
                float(std::sqrt(uu[c] * uu[c] + vv[c] * vv[c] + ww[c] * ww[c]))};
    };

    // Live browser frames: the three centre planes of |curl| and |u|, one
    // web_<prefix>[_xz|_yz]/ dir per plane (docs/viewer-format.md;
    // site/live.html discovers the _xz/_yz siblings and shows plane tabs).
    // Frame source = a full-volume primitive pull; at webDt cadence
    // (O(500) steps apart) the ~1 GB PCIe copy is noise.
    const std::string simJson =
        "{\"app\":\"comp_mesh\",\"re\":" + std::to_string(reD) +
        ",\"mach\":" + std::to_string(mach) + ",\"les\":" +
        std::to_string(csLes);
    struct WebPlane
    {
        std::unique_ptr<dns::WebFrames> wf;
        std::function<std::array<int, 3>(int, int)> sel; // (x, y) -> (i, j, k)
        int a = 0, b = 0;
    };
    std::vector<WebPlane> webs;
    std::string planesIdx; // planes.json entries (viewer plane discovery)
    auto addWeb = [&](const std::string& suf, const std::string& axis,
                      double offD, int a, int b, auto sel) {
        WebPlane wp;
        char offs[40];
        std::snprintf(offs, sizeof offs, "%+.3gD", offD);
        wp.wf = std::make_unique<dns::WebFrames>(
            "web_" + prefix + suf, a, b,
            simJson + ",\"plane\":\"" + axis + "\",\"off\":\"" +
                (offD == 0.0 ? "mid" : offs) + "\"}");
        wp.wf->addField("curl", "sequential", 0.0, 15.0 * uInf / D0);
        wp.wf->addField("speed", "sequential", 0.0, 1.6 * uInf);
        wp.sel = sel;
        wp.a = a;
        wp.b = b;
        std::vector<float> m(std::size_t(a) * b);
        std::size_t q = 0;
        for (int y = 0; y < b; ++y)
            for (int x = 0; x < a; ++x, ++q) {
                const auto [i, j, k] = sel(x, y);
                m[q] = float(cut.lambda[g.idx(i, j, k)]);
            }
        wp.wf->writeStatic("mask", m.data(), 0.0, 1.0);
        char e[128];
        std::snprintf(e, sizeof e, "%s{\"suf\":\"%s\",\"axis\":\"%s\",\"off\":%.4g}",
                      planesIdx.empty() ? "" : ",", suf.c_str(), axis.c_str(),
                      offD);
        planesIdx += e;
        webs.push_back(std::move(wp));
    };
    // Three planes per axis: centre plus +/- n/16 cells (offset labeled in
    // body diameters). Base dir stays the xy centre plane (back-compat).
    const int dI = std::max(1, nx / 16), dJ = std::max(1, ny / 16),
              dK = std::max(1, nz / 16);
    auto cellsD = [&](int cells) { return cells * h / D; };
    auto xyAt = [&](int k) {
        return [&, k](int i, int j) { return std::array{i, j, k}; };
    };
    auto xzAt = [&](int j) {
        return [&, j](int i, int k) { return std::array{i, j, k}; };
    };
    auto yzAt = [&](int i) {
        return [&, i](int j, int k) { return std::array{i, j, k}; };
    };
    addWeb("", "xy", 0.0, nx, ny, xyAt(kc));
    if (nz >= 16) {
        addWeb("_xy_m", "xy", -cellsD(dK), nx, ny, xyAt(kc - dK));
        addWeb("_xy_p", "xy", +cellsD(dK), nx, ny, xyAt(kc + dK));
    }
    addWeb("_xz", "xz", 0.0, nx, nz, xzAt(jc));
    if (ny >= 16) {
        addWeb("_xz_m", "xz", -cellsD(dJ), nx, nz, xzAt(jc - dJ));
        addWeb("_xz_p", "xz", +cellsD(dJ), nx, nz, xzAt(jc + dJ));
    }
    addWeb("_yz", "yz", 0.0, ny, nz, yzAt(ic));
    if (nx >= 16) {
        addWeb("_yz_m", "yz", -cellsD(dI), ny, nz, yzAt(ic - dI));
        addWeb("_yz_p", "yz", +cellsD(dI), ny, nz, yzAt(ic + dI));
    }
    std::ofstream("web_" + prefix + "/planes.json")
        << "{\"planes\":[" << planesIdx << "]}\n";
    auto webSlice = [&](double tNow) {
        downloadPrims();
        for (auto& wp : webs) {
            std::vector<float> cb(std::size_t(wp.a) * wp.b), sb(cb.size());
            std::size_t q = 0;
            for (int y = 0; y < wp.b; ++y)
                for (int x = 0; x < wp.a; ++x, ++q) {
                    const auto [i, j, k] = wp.sel(x, y);
                    std::tie(cb[q], sb[q]) = vortSpeed(i, j, k);
                }
            wp.wf->writeFrame(tNow, {cb.data(), sb.data()});
        }
    };

    int frames = 0;
    // 3D volume dump (non-slice): full |curl|/|u| grids appended per frame.
    std::ofstream curlFile, speedFile;
    std::vector<float> curlBuf, speedBuf;
    if (!sliceMode) {
        curlFile.open(prefix + "_curl.bin", std::ios::binary);
        speedFile.open(prefix + "_speed.bin", std::ios::binary);
        curlBuf.resize(g.size());
        speedBuf.resize(g.size());
    }
    auto writeVolume = [&]() {
        downloadPrims();
        std::size_t p = 0;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i, ++p)
                    std::tie(curlBuf[p], speedBuf[p]) = vortSpeed(i, j, k);
        curlFile.write(reinterpret_cast<const char*>(curlBuf.data()),
                       std::streamsize(curlBuf.size() * sizeof(float)));
        speedFile.write(reinterpret_cast<const char*>(speedBuf.data()),
                        std::streamsize(speedBuf.size() * sizeof(float)));
        ++frames;
    };

    // Slice dump: the three center planes of |curl| and |u|, appended per frame
    // (nx*ny, nx*nz, ny*nz floats each). Tiny vs the full volume.
    std::ofstream sxyC, sxyS, sxzC, sxzS, syzC, syzS;
    if (sliceMode) {
        sxyC.open(prefix + "_xy_curl.bin", std::ios::binary);
        sxyS.open(prefix + "_xy_speed.bin", std::ios::binary);
        sxzC.open(prefix + "_xz_curl.bin", std::ios::binary);
        sxzS.open(prefix + "_xz_speed.bin", std::ios::binary);
        syzC.open(prefix + "_yz_curl.bin", std::ios::binary);
        syzS.open(prefix + "_yz_speed.bin", std::ios::binary);
    }
    auto writeSlices = [&]() {
        downloadPrims();
        auto plane = [&](std::ofstream& fc, std::ofstream& fs, auto sel, int a, int b) {
            std::vector<float> cbuf(std::size_t(a) * b), sbuf(std::size_t(a) * b);
            std::size_t q = 0;
            for (int y = 0; y < b; ++y)
                for (int x = 0; x < a; ++x, ++q) {
                    const auto [i, j, k] = sel(x, y);
                    std::tie(cbuf[q], sbuf[q]) = vortSpeed(i, j, k);
                }
            fc.write(reinterpret_cast<const char*>(cbuf.data()),
                     std::streamsize(cbuf.size() * sizeof(float)));
            fs.write(reinterpret_cast<const char*>(sbuf.data()),
                     std::streamsize(sbuf.size() * sizeof(float)));
        };
        plane(sxyC, sxyS,
              [&](int i, int j) { return std::array{i, j, kc}; }, nx, ny);
        plane(sxzC, sxzS,
              [&](int i, int k) { return std::array{i, jc, k}; }, nx, nz);
        plane(syzC, syzS,
              [&](int j, int k) { return std::array{ic, j, k}; }, ny, nz);
        ++frames;
    };

    double t = 0.0;
    long steps = 0, lastLog = 0;
    double nextFrame = 0.0, nextWeb = 0.0;
    const auto wall0 = std::chrono::steady_clock::now();

    while (t < tEnd) {
        const double dtStab = dev->stableDt(*ds, *red, cfl);
        const double dt = std::min(tEnd - t, dtStab);
        if (dtStab < 1e-6) {
            std::fprintf(stderr, "dt collapsed at t=%.3f -- blowup\n", t);
            return 2;
        }
        dev->step(*ds, Real(dt));
        if (t < tRamp) { // quiet start: boost the frame, retarget the inflow
            const double uNew = uInf * rampFrac(t + dt, tRamp);
            gpu::boostX(gg, *ds, uNew - nsc->freestream());
            nsc->setFreestream(uNew);
            if (far) far->setFreestream(uNew);
        }
        applyBc(*ds); // keep the state BC/ghost-consistent between steps
        t += dt;
        ++steps;

        if (t >= nextFrame) {
            if (sliceMode) writeSlices();
            else writeVolume();
            nextFrame += frameDt;
        }
        if (t >= nextWeb) {
            webSlice(t);
            nextWeb += webDt;
        }
        if (steps - lastLog >= 200 || t >= tEnd) {
            const auto F = ghost->force(*dev, *ds);
            const double cd = F.fx / qRef;
            const double cl = F.fy / qRef;
            const double cs = F.fz / qRef;
            const double wmax = cfl / dtStab * std::min({g.hx, g.hy, g.hz});
            stats << t << ',' << dt << ',' << cd << ',' << cl << ',' << cs << ','
                  << wmax << '\n';
            stats.flush();
            if (steps - lastLog >= 2000 || t >= tEnd)
                std::printf("t=%.2f/%.2f cd=%.3f cl=%+.3f cs=%+.3f wmax=%.2f "
                            "dt=%.2e\n",
                            t, tEnd, cd, cl, cs, wmax, dt);
            lastLog = steps;
        }
    }
    (void)hipDeviceSynchronize();
    const auto wallNow = std::chrono::steady_clock::now();
    const double solveSec = std::chrono::duration<double>(wallNow - wall0).count();
    const double totalSec = std::chrono::duration<double>(wallNow - progT0).count();
    const double ms = 1e3 * solveSec / double(steps);
    const double cells = double(nx) * ny * nz;
    const double mcups = cells * double(steps) / solveSec / 1e6; // Mcell-upd/s
    writeJson(frames, ms);
    std::printf("done: %ld steps, %d frames\n", steps, frames);
    std::printf("# perf: wallTotal=%.2fs geomBuild=%.2fs solve=%.2fs "
                "msPerStep=%.3f steps=%ld cells=%.3gM throughput=%.1f "
                "Mcell-updates/s\n",
                totalSec, geomSec, solveSec, ms, steps, cells / 1e6, mcups);
    return 0;
}
