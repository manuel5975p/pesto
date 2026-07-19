// MeshSdf + mesh loaders + sampleCut against analytic oracles (CPU only).
//  1) cube mesh: MeshSdf == exact box sdf (vertex/edge/face pseudonormals).
//  2) icosphere: |MeshSdf - sphere sdf| < faceting sag; sign exact off-surface.
//  3) pseudonormal vs winding-number sign agreement at every cell center
//     (default and robust classification of solid cells must match).
//  4) sampleCut: solid-volume vs exact mesh volume, x-plane aperture deficit
//     vs cross-circle area (coarse: supersampling), sanitizeCut invariants.
//  5) STL round-trip: binary and ASCII through temp files.
#include "dns/geometry.hpp"
#include "dns/meshsdf.hpp"

#include "mesh_test_util.hpp"

#include "dns/cutcell.hpp"
#include "dns/trimesh.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numbers>

using namespace dns;

namespace {

using A3 = std::array<double, 3>;

bool g_ok = true;

void check(const char* what, double err, double tol)
{
    std::printf("%-34s %.3e (tol %.1e)\n", what, err, tol);
    if (!(err < tol))
    {
        std::printf("FAIL: %s\n", what);
        g_ok = false;
    }
}

void require(const char* what, bool cond)
{
    if (!cond)
    {
        std::printf("FAIL: %s\n", what);
        g_ok = false;
    }
}

// Axis-aligned cube [lo,hi]^3 as 12 outward-CCW triangles.
TriMesh cubeMesh(double lo, double hi)
{
    TriMesh m;
    for (int i = 0; i < 8; ++i)
        m.verts.push_back({i & 1 ? hi : lo, i & 2 ? hi : lo, i & 4 ? hi : lo});
    m.tris = {{0, 6, 2}, {0, 4, 6}, {1, 3, 7}, {1, 7, 5}, // -x, +x
              {0, 1, 5}, {0, 5, 4}, {2, 6, 7}, {2, 7, 3}, // -y, +y
              {0, 3, 1}, {0, 2, 3}, {4, 5, 7}, {4, 7, 6}}; // -z, +z
    return m;
}

// Exact box sdf: negative inside (distance to the nearest face).
double boxSdf(const A3& p, double lo, double hi)
{
    double q[3], inner = -1e300;
    for (int a = 0; a < 3; ++a)
    {
        q[a] = std::max(lo - p[a], p[a] - hi);
        inner = std::max(inner, q[a]);
    }
    if (inner <= 0) return inner;
    double s = 0;
    for (double v : q) s += std::max(v, 0.0) * std::max(v, 0.0);
    return std::sqrt(s);
}

using meshtest::icosphere; // shared with test_val_mesh_sphere (mesh_test_util)
using meshtest::writeBinaryStl;

// Signed volume of a closed mesh by the divergence theorem.
double meshVolume(const TriMesh& m)
{
    double v6 = 0;
    for (const auto& t : m.tris)
    {
        const A3 &a = m.verts[t[0]], &b = m.verts[t[1]], &c = m.verts[t[2]];
        v6 += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    return v6 / 6.0;
}

void writeAsciiStl(const std::filesystem::path& p, const TriMesh& m)
{
    std::ofstream f(p);
    f << "solid test\n";
    for (const auto& t : m.tris)
    {
        f << " facet normal 0 0 0\n  outer loop\n";
        for (int v = 0; v < 3; ++v)
            f << "   vertex " << m.verts[t[v]][0] << ' ' << m.verts[t[v]][1]
              << ' ' << m.verts[t[v]][2] << '\n';
        f << "  endloop\n endfacet\n";
    }
    f << "endsolid test\n";
}

} // namespace

int main()
{
    using std::numbers::pi;
    const int n = 32;
    const double L = 2 * pi;
    const auto g = Grid::make(n, n, n, L, L, L);
    const double h = g.hx;

    // --- 1) cube: mesh sdf must be the exact box sdf ---
    {
        const double lo = 2.0, hi = 4.0;
        const MeshSdf sdf(cubeMesh(lo, hi));
        require("cube watertight", sdf.watertight());
        double maxErr = 0.0;
        int badSign = 0;
        std::uint32_t rng = 12345;
        auto urand = [&] { // xorshift in [0,1)
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            return double(rng) / 4294967296.0;
        };
        for (int s = 0; s < 20000; ++s)
        {
            const A3 p{1.0 + 4.0 * urand(), 1.0 + 4.0 * urand(), 1.0 + 4.0 * urand()};
            const double want = boxSdf(p, lo, hi);
            const double got = sdf(p[0], p[1], p[2]);
            maxErr = std::max(maxErr, std::abs(got - want));
            if (std::abs(want) > 1e-9 && std::signbit(got) != std::signbit(want))
                ++badSign;
        }
        check("cube sdf max abs err", maxErr, 1e-9);
        require("cube sdf sign exact", badSign == 0);
    }

    // Centre off-node (arbitrary cuts), as test_cutcell_sphere.
    const double r = L / 5;
    const A3 c{L / 2 + 0.137 * h, L / 2 - 0.293 * h, L / 2 + 0.051 * h};
    const TriMesh sphMesh = icosphere(r, c, 3); // 1280 tris
    const double volExact = meshVolume(sphMesh);
    check("icosphere volume vs sphere",
          std::abs(volExact - 4.0 / 3.0 * pi * r * r * r) / (4.0 / 3.0 * pi * r * r * r),
          1e-2);
    const MeshSdf sdf{TriMesh(sphMesh)};
    require("icosphere watertight", sdf.watertight());
    require("icosphere tri count", sdf.triangleCount() == 1280);

    // --- 2) icosphere vs analytic sphere sdf ---
    {
        const double sag = 0.005 * r; // faceting bound, level-3 icosphere
        double maxErr = 0.0;
        int badSign = 0;
        for (int a = 0; a < 200; ++a)
        {
            const double th = pi * (a + 0.5) / 200, ph = 2.6180339 * a;
            const A3 d{std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph),
                       std::cos(th)};
            for (double rho : {0.3 * r, 0.7 * r, 0.98 * r, 1.02 * r, 1.5 * r, 3.0 * r})
            {
                const double got =
                    sdf(c[0] + rho * d[0], c[1] + rho * d[1], c[2] + rho * d[2]);
                maxErr = std::max(maxErr, std::abs(got - (rho - r)));
                if (std::abs(rho - r) > sag && std::signbit(got) != std::signbit(rho - r))
                    ++badSign;
            }
        }
        check("icosphere sdf max abs err", maxErr / r, 0.01);
        require("icosphere sdf sign exact", badSign == 0);
    }

    // --- 3) pseudonormal vs winding sign at every cell center ---
    {
        int solidPn = 0, solidWn = 0, disagree = 0;
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    const double x = (i + 0.5) * h, y = (j + 0.5) * h,
                                 z = (k + 0.5) * h;
                    const double phi = sdf(x, y, z);
                    const bool inPn = phi < 0;
                    const bool inWn = sdf.winding(x, y, z) > 0.5;
                    solidPn += inPn;
                    solidWn += inWn;
                    if (inPn != inWn && std::abs(phi) > 1e-9) ++disagree;
                }
        std::printf("solid cells: pseudonormal=%d winding=%d disagree=%d\n",
                    solidPn, solidWn, disagree);
        require("default vs robust solid count", solidPn == solidWn);
        require("sign paths agree", disagree == 0);
        require("solid cells found", solidPn > 0);
    }

    // Robust-mode MeshSdf takes the winding path inside operator().
    {
        const MeshSdf sdfR{TriMesh(sphMesh), true};
        double maxDiff = 0.0;
        for (double rho : {0.5 * r, 0.9 * r, 1.1 * r, 2.0 * r})
            maxDiff = std::max(maxDiff, std::abs(sdfR(c[0] + rho, c[1], c[2])
                                                 - sdf(c[0] + rho, c[1], c[2])));
        check("robust vs default sdf", maxDiff / r, 1e-9);
    }

    // --- 4) makeMeshCut geometry + sanitize invariants ---
    auto cutR = sampleCut(g, [&](double x, double y, double z) { return sdf(x, y, z); });
    if (!cutR) { std::printf("FAIL: sampleCut: %s\n", cutR.error().c_str()); return 1; }
    auto cut = std::move(*cutR);
    {
        double solid = 0.0;
        for (double l : cut.lambda) solid += 1.0 - l;
        solid *= g.cellVolume();
        check("cut volume rel err", std::abs(solid - volExact) / volExact, 1e-2);
    }
    {
        const int i = int(std::floor(c[0] / h)) + 2; // plane cutting the sphere
        const double d = i * h - c[0];
        const double exact = pi * (r * r - d * d);
        double solid = 0.0;
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                solid += (1.0 - cut.ax[g.idx(i, j, k)]) * g.hy * g.hz;
        check("cut aperture rel err", std::abs(solid - exact) / exact, 3e-2);
    }
    const double lambdaMin = 1e-2, apertureMin = 1e-2;
    const auto st = sanitizeCut(g, cut, lambdaMin, apertureMin);
    std::printf("sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed,
                st.minLambda, st.minAperture);
    require("sanitize thresholds",
            st.minLambda >= lambdaMin && st.minAperture >= apertureMin);
    {
        const double tol = CutGeometry::openTol;
        for (int k = 0; k < n && g_ok; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    const auto cc = g.idx(i, j, k);
                    const bool solidC = cut.lambda[cc] <= 0.0;
                    if ((cut.ax[cc] > tol && (solidC || cut.lambda[g.idx(i - 1, j, k)] <= 0.0))
                     || (cut.ay[cc] > tol && (solidC || cut.lambda[g.idx(i, j - 1, k)] <= 0.0))
                     || (cut.az[cc] > tol && (solidC || cut.lambda[g.idx(i, j, k - 1)] <= 0.0)))
                    {
                        std::printf("FAIL: open face on solid cell at %d %d %d\n", i, j, k);
                        g_ok = false;
                    }
                    if (!solidC
                        && !(cut.ax[cc] > tol || cut.ax[g.idx(i + 1, j, k)] > tol
                          || cut.ay[cc] > tol || cut.ay[g.idx(i, j + 1, k)] > tol
                          || cut.az[cc] > tol || cut.az[g.idx(i, j, k + 1)] > tol))
                    {
                        std::printf("FAIL: isolated fluid cell at %d %d %d\n", i, j, k);
                        g_ok = false;
                    }
                }
    }

    // --- 5) STL round-trip: binary and ASCII ---
    {
        const auto dir = std::filesystem::temp_directory_path();
        const auto binPath = dir / "dns_meshsdf_bin.stl";
        const auto ascPath = dir / "dns_meshsdf_asc.stl";
        writeBinaryStl(binPath, sphMesh);
        writeAsciiStl(ascPath, cubeMesh(2.0, 4.0));

        auto sb = loadStl(binPath.string());
        require("binary STL loads", bool(sb));
        if (sb)
        {
            require("binary STL tri count", sb->tris.size() == sphMesh.tris.size());
            const Aabb bb = bounds(*sb);
            double bbErr = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                bbErr = std::max(bbErr, std::abs(bb.lo[a] - (c[a] - r)));
                bbErr = std::max(bbErr, std::abs(bb.hi[a] - (c[a] + r)));
            }
            check("binary STL bbox err", bbErr, 1e-6);
            const MeshSdf s2(std::move(*sb));
            require("binary STL watertight", s2.watertight());
            double dErr = 0.0;
            for (double rho : {0.5 * r, 1.5 * r})
                dErr = std::max(dErr, std::abs(s2(c[0], c[1] + rho, c[2])
                                               - sdf(c[0], c[1] + rho, c[2])));
            check("binary STL sdf err", dErr, 1e-6);
        }
        auto sa = loadStl(ascPath.string());
        require("ASCII STL loads", bool(sa));
        if (sa)
        {
            require("ASCII STL tri count", sa->tris.size() == 12);
            require("ASCII STL vert dedup", sa->verts.size() == 8);
        }
        std::filesystem::remove(binPath);
        std::filesystem::remove(ascPath);
    }

    // loadMesh dispatch rejects unknown extensions with a message.
    {
        auto bad = loadMesh("nope.obj");
        require("loadMesh rejects .obj", !bad.has_value());
    }

    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
