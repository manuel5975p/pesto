// Flow around a 45-deg rotated cube via the cut-cell embedded boundary (CPU).
// Sharp no-through-flow on the cut faces + apertured pressure projection, as
// opposed to the staircase direct-forcing of cube_main.hip. Geometry is placed
// so every cut cell is halved (no small-cell stiffness). Uniform inflow held by
// an outflow fringe; body drag read from the fringe momentum injection.
// Usage: cube_cut [n] [--re ReD] [--ft flowThroughs] [--out prefix]
#include "dns/cutcell.hpp"
#include "dns/operators.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

using namespace dns;

namespace {

// Snap a world coordinate to the nearest grid node so diamond edges/caps stay
// node-aligned (the exact-halving condition).
double snapNode(double x, double h) { return std::round(x / h) * h; }

double parseOr(const char* s, double fallback)
{
    double v{};
    const std::string_view sv(s);
    auto [q, ec] = std::from_chars(sv.begin(), sv.end(), v);
    return ec == std::errc{} && q == sv.end() ? v : fallback;
}

// out = -(U.grad)U + nu*lap(U) on the plain MAC operators; the cut faces are
// then re-zeroed by the caller. (Apertured advection is out of scope here.)
struct Rhs
{
    const Grid& g;
    double nu;
    VectorField conv, lap;
    explicit Rhs(const Grid& g_, double nu_)
        : g(g_), nu(nu_), conv(VectorField::zeros(g_)), lap(VectorField::zeros(g_)) {}

    void operator()(const VectorField& U, VectorField& out)
    {
        convection(g, U, conv);
        laplacian(g, U, lap);
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            out.u[c] = conv.u[c] + nu * lap.u[c];
            out.v[c] = conv.v[c] + nu * lap.v[c];
            out.w[c] = conv.w[c] + nu * lap.w[c];
        }
    }
};

// Relax the outflow strip back to the free stream; accumulates the streamwise
// momentum it injects (== body drag at steady state). Returns that increment.
double fringe(const Grid& g, const CutGeometry& cut, double x0, double x1,
              double uInf, double blend, VectorField& U)
{
    double addMom = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                auto ramp = [&](double x) {
                    if (x < x0 || x > x1) return 0.0;
                    const double xi = (x - x0) / (x1 - x0);
                    return blend * xi * xi * (3 - 2 * xi);
                };
                const double su = ramp(i * g.hx), sc = ramp((i + 0.5) * g.hx);
                const double du = su * (uInf - U.u[c]);
                addMom += cut.ax[c] * du;
                U.u[c] += du;
                U.v[c] -= sc * U.v[c];
                U.w[c] -= sc * U.w[c];
            }
    return addMom * g.cellVolume();
}

} // namespace

int main(int argc, char** argv)
{
    using std::numbers::pi;

    int n = 96, nzFlat = 0;
    double reD = 300.0, flowThroughs = 6.0, dRatio = 8.0, frameDt = 0.15;
    std::string prefix = "cube_cut";
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--re") reD = next(reD);
        else if (s == "--dr") dRatio = next(dRatio);
        else if (s == "--ft") flowThroughs = next(flowThroughs);
        else if (s == "--flat") nzFlat = int(next(nzFlat)); // quasi-2D: thin z, prism
        else if (s == "--fdt") frameDt = next(frameDt);
        else if (s == "--out" && a + 1 < argc) prefix = argv[++a];
        else n = int(parseOr(argv[a], n));
    }

    const double L = 2 * pi, uInf = 1.0;
    const double h = L / n;
    const int nz = nzFlat > 0 ? nzFlat : n;
    const auto g = Grid::make(n, n, nz, L, L, nz * h); // cubic cells (hz == h)

    // Diamond half-diagonal R -> square side R*sqrt(2) = "D". Center on nodes so
    // every cut cell is halved. In flat mode the prism spans all z (2D cylinder-
    // like body); in 3D mode z-caps are node-aligned too.
    const double R = snapNode(L / dRatio / std::numbers::sqrt2, h);
    const double D = R * std::numbers::sqrt2;
    const double cx = snapNode(L / 3.0, h), cy = snapNode(L / 2.0, h);
    const double halfZ = snapNode(D / 2.0, h);
    const RotatedCube cube = nzFlat > 0
        ? RotatedCube{cx, cy, R, -1.0, nz * h + 1.0}
        : RotatedCube{cx, cy, R, L / 2 - halfZ, L / 2 + halfZ};
    const auto cut = CutGeometry::make(g, cube);

    const double nu = uInf * D / reD;
    std::printf("# cube_cut n=%d Re_D=%g nu=%.4e D=%.4f (%.1f cells) rotated 45deg\n",
                n, reD, nu, D, D / h);

    // Verify the halving invariant up front (cheap, and the whole point).
    int nCut = 0, offHalf = 0;
    for (double l : cut.lambda)
        if (l > 1e-12 && l < 1 - 1e-12) { ++nCut; if (std::abs(l - 0.5) > 1e-9) ++offHalf; }
    std::printf("# cut cells: %d, off-half: %d\n", nCut, offHalf);

    // Clean laminar free stream (wind-channel inflow). A small deterministic
    // trip in the strip one D behind the cube breaks the wake symmetry so
    // shedding starts without seeding the whole field with noise.
    auto U = VectorField::zeros(g);
    for (auto& x : U.u) x = uInf;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                const double x = i * h, y = (j + 0.5) * h;
                if (x > cx + R && x < cx + R + D && std::abs(y - cy) < R)
                    U.v[g.idx(i, j, k)] = 0.05 * uInf;
            }
    maskSolidFaces(g, cut, U);

    CutPoissonSolver poisson(g, cut, 2000, 1e-6);
    auto scr = CutProjectionScratch::make(g);
    Rhs rhs(g, nu);
    VectorField s1 = VectorField::zeros(g), k = VectorField::zeros(g);

    const double xf0 = 0.85 * L, sigmaFringe = 40.0;
    // Frontal area: D x span. In flat mode the prism spans the whole (thin)
    // z-extent, so the span is Lz, not D.
    const double span = nzFlat > 0 ? nz * h : D;
    const double vol = L * L * L, qRef = 0.5 * uInf * uInf * D * span;

    auto stage = [&](VectorField& Uin, VectorField& Uout, double a0,
                     const VectorField& U0, double dt) {
        rhs(Uin, k);
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            Uout.u[c] = a0 * U0.u[c] + (1 - a0) * (Uin.u[c] + dt * k.u[c]);
            Uout.v[c] = a0 * U0.v[c] + (1 - a0) * (Uin.v[c] + dt * k.v[c]);
            Uout.w[c] = a0 * U0.w[c] + (1 - a0) * (Uin.w[c] + dt * k.w[c]);
        }
        maskSolidFaces(g, cut, Uout);
        projectCut(g, cut, poisson, dt, Uout, scr);
    };

    const double tEnd = flowThroughs * L / uInf;
    std::ofstream stats(prefix + "_stats.csv");
    stats << "t,dt,K,cd,cfl,maxDiv\n";

    // Startup meta so a viewer can render before the run finishes (frame count
    // is derived from the growing .bin file size).
    std::ofstream(prefix + ".json")
        << "{\"nx\":" << n << ",\"ny\":" << n << ",\"frameDt\":" << frameDt
        << ",\"D\":" << D << ",\"reD\":" << reD << ",\"cx\":" << cx / h
        << ",\"cy\":" << cy / h << ",\"R\":" << R / h << "}\n";

    // Mid-plane slice output for the movie: signed z-vorticity at cell corners.
    const int kc = nz / 2;
    std::ofstream vortFile(prefix + "_vort.bin", std::ios::binary);
    int frames = 0;
    double nextFrame = 0.0;
    std::vector<float> frame(std::size_t(n) * n);
    auto writeFrame = [&]() {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                const auto c = g.idx(i, j, kc);
                const double wz = (U.v[c] - U.v[g.idx(i - 1, j, kc)]) / h
                                - (U.u[c] - U.u[g.idx(i, j - 1, kc)]) / h;
                frame[std::size_t(j) * n + i] = float(wz);
            }
        vortFile.write(reinterpret_cast<const char*>(frame.data()),
                       std::streamsize(frame.size() * sizeof(float)));
        ++frames;
    };
    // Solid mask (mid-plane fluid fraction) so the renderer can draw the body.
    {
        std::ofstream mf(prefix + "_mask.bin", std::ios::binary);
        std::vector<float> m(std::size_t(n) * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                m[std::size_t(j) * n + i] = float(cut.lambda[g.idx(i, j, kc)]);
        mf.write(reinterpret_cast<const char*>(m.data()),
                 std::streamsize(m.size() * sizeof(float)));
    }

    double t = 0.0, dt = 1e-3;
    long steps = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < tEnd)
    {
        const double conv = maxCourantPerDt(g, U);
        const double diff = nu > 0 ? 6.0 * nu / (h * h) : 0.0;
        dt = std::min(tEnd - t, 0.8 / std::max(conv, diff));

        // SSP-RK3, each stage masked + cut-projected.
        stage(U, s1, 0.0, U, dt);           // s1 = U + dt L
        stage(s1, s1, 0.75, U, dt);         // s1 = 3/4 U + 1/4(s1 + dt L)
        stage(s1, U, 1.0 / 3.0, U, dt);     // U  = 1/3 U + 2/3(s1 + dt L)

        const double dMom = fringe(g, cut, xf0, L, uInf,
                                   std::min(1.0, sigmaFringe * dt), U);
        projectCut(g, cut, poisson, dt, U, scr);

        if (t >= nextFrame) { writeFrame(); nextFrame += frameDt; }

        if (steps % 20 == 0)
        {
            const double K = kineticEnergy(g, U);
            const double cd = dMom / dt / qRef; // fringe injection == drag
            const double cfl = dt * conv;
            const double maxDiv = maxAbsDivergenceCut(g, cut, U);
            stats << t << ',' << dt << ',' << K << ',' << cd << ',' << cfl << ','
                  << maxDiv << '\n';
            stats.flush();
            if (steps % 200 == 0)
            {
                std::printf("t=%.2f/%.2f cd=%.3f K=%.1f maxDiv=%.2e\n",
                            t, tEnd, cd, K, maxDiv);
                std::fflush(stdout);
            }
        }
        t += dt;
        ++steps;
    }
    (void)vol;
    const double wallMs = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - wall0).count();

    std::ofstream(prefix + ".json")
        << "{\"nx\":" << n << ",\"ny\":" << n << ",\"frames\":" << frames
        << ",\"frameDt\":" << frameDt << ",\"D\":" << D << ",\"reD\":" << reD
        << ",\"cx\":" << cx / h << ",\"cy\":" << cy / h << ",\"R\":" << R / h << "}\n";
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n",
                steps, frames, wallMs / double(steps));
    return 0;
}
