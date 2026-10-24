#include "dns/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace dns {
namespace {

// Fluid fraction of an axis box by an sx*sy*sz sample lattice. A zero extent
// with one sample degenerates the box to a face (or a segment).
double frac(const Sdf& sdf, double x0, double y0, double z0, double ex, double ey,
            double ez, int sx, int sy, int sz)
{
    int m = 0;
    for (int c = 0; c < sz; ++c)
        for (int b = 0; b < sy; ++b)
            for (int a = 0; a < sx; ++a)
                if (sdf(x0 + (a + 0.5) * ex / sx, y0 + (b + 0.5) * ey / sy,
                        z0 + (c + 0.5) * ez / sz) > 0.0)
                    ++m;
    return double(m) / (sx * sy * sz);
}

} // namespace

std::expected<CutGeometry, std::string>
sampleCut(const Grid& g, const Sdf& sdf, SampleOpts o)
{
    if (g.nx < 1 || g.ny < 1 || g.nz < 1)
        return std::unexpected("sampleCut: grid dimensions must be >= 1");
    if (o.nCell < 1 || o.nFace < 1)
        return std::unexpected("sampleCut: sample counts must be >= 1");
    if (!(std::isfinite(o.band) && o.band >= 0))
        return std::unexpected("sampleCut: band must be finite and >= 0");

    const bool flat = o.zInvariant || g.nz == 1;
    const double hz = flat ? 0.0 : g.hz;
    // Half the cell diagonal: outside +-that of the surface a cell cannot be
    // cut, because the sdf is a distance.
    const double half = 0.5 * std::sqrt(g.hx * g.hx + g.hy * g.hy + hz * hz);
    const double band = o.band > 0 ? o.band : half;

    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    const int nc = o.nCell, nf = o.nFace;

    auto classify = [&](double phi, auto sample) {
        return phi >= band ? 1.0 : phi <= -band ? 0.0 : sample();
    };

    // Lipschitz block prefilter: |grad sdf| = 1, so one query at a B^3-cell
    // block centre bounds every sample point in the block. Blocks wholly
    // outside [-band - r, band + r] skip the per-cell queries entirely.
    constexpr int B = 8;
    const int kStep = flat ? std::max(g.nz, 1) : B;
    const double blockR = 0.5
        * std::sqrt(double(B * g.hx) * (B * g.hx) + double(B * g.hy) * (B * g.hy)
                    + double(flat ? 0.0 : double(B * g.hz) * (B * g.hz)));

    const int kEnd = flat ? 1 : g.nz;   // one layer when z-invariant
    for (int kb = 0; kb < kEnd; kb += kStep)
        for (int jb = 0; jb < g.ny; jb += B)
            for (int ib = 0; ib < g.nx; ib += B)
            {
                const int i1 = std::min(ib + B, g.nx), j1 = std::min(jb + B, g.ny);
                const int k1 = flat ? 1 : std::min(kb + B, g.nz);
                const double phiC = sdf((ib + 0.5 * (i1 - ib)) * g.hx,
                                        (jb + 0.5 * (j1 - jb)) * g.hy,
                                        flat ? 0.5 * g.hz
                                             : (kb + 0.5 * (k1 - kb)) * g.hz);
                const bool allFluid = phiC >= band + blockR;
                const bool allSolid = phiC <= -band - blockR;

                for (int k = kb; k < k1; ++k)
                    for (int j = jb; j < j1; ++j)
                        for (int i = ib; i < i1; ++i)
                        {
                            double lam, ax, ay, az;
                            if (allFluid || allSolid)
                                lam = ax = ay = az = allFluid ? 1.0 : 0.0;
                            else
                            {
                                const double x0 = i * g.hx, y0 = j * g.hy,
                                             z0 = flat ? 0.0 : k * g.hz;
                                const double zc = flat ? 0.5 * g.hz
                                                       : z0 + 0.5 * g.hz;
                                lam = classify(
                                    sdf(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy, zc),
                                    [&] {
                                        return frac(sdf, x0, y0, z0, g.hx, g.hy, hz,
                                                    nc, nc, flat ? 1 : nc);
                                    });
                                ax = classify(sdf(x0, y0 + 0.5 * g.hy, zc), [&] {
                                    return frac(sdf, x0, y0, z0, 0, g.hy, hz, 1, nf,
                                                flat ? 1 : nf);
                                });
                                ay = classify(sdf(x0 + 0.5 * g.hx, y0, zc), [&] {
                                    return frac(sdf, x0, y0, z0, g.hx, 0, hz, nf, 1,
                                                flat ? 1 : nf);
                                });
                                // A z-invariant body cuts no z face: the w face
                                // sees exactly the cell's own fluid fraction.
                                az = flat ? lam
                                          : classify(
                                                sdf(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy,
                                                    z0),
                                                [&] {
                                                    return frac(sdf, x0, y0, z0, g.hx,
                                                                g.hy, 0, nf, nf, 1);
                                                });
                            }
                            // Replicate the single layer over k when flat.
                            for (int kk = flat ? 0 : k; kk < (flat ? g.nz : k + 1);
                                 ++kk)
                            {
                                const auto id = g.idx(i, j, kk);
                                cut.lambda[id] = lam;
                                cut.ax[id] = ax;
                                cut.ay[id] = ay;
                                cut.az[id] = az;
                            }
                        }
            }
    return cut;
}

std::expected<CutGeometry, std::string>
sampleCutMapped(const Grid& g, const Sdf& sdf, const AxisRemap3& m, SampleOpts o)
{
    const Focal1D* ax[3] = {&m.fx, &m.fy, &m.fz};
    if (std::ranges::none_of(ax, [](const Focal1D* f) { return f->active; }))
        return sampleCut(g, sdf, o);
    // Focal1D's slope is in [c r, c]: c bounds it.
    double slope = 1.0;
    for (const Focal1D* f : ax)
        if (f->active) slope = std::max(slope, f->c);
    const Sdf pulled = [&sdf, &m, slope](double X, double Y, double Z) {
        return sdf(m.fx.map(X), m.fy.map(Y), m.fz.map(Z)) / slope;
    };
    return sampleCut(g, pulled, o);
}

std::expected<CutGeometry, std::string>
exactLensCut(const Grid& g, const Lens& b, int ns)
{
    if (!(b.w > b.hh)) return std::unexpected("exactLensCut: needs w > hh");
    if (ns < 1) return std::unexpected("exactLensCut: lambdaSamples must be >= 1");

    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    for (auto& v : cut.lambda) v = 1;
    for (auto& v : cut.ax) v = 1;
    for (auto& v : cut.ay) v = 1;
    for (auto& v : cut.az) v = 1;

    const double e = b.e(), ra = b.ra(), ra2 = ra * ra;
    auto inside = [&](double x, double y) {
        const double dx2 = (x - b.cx) * (x - b.cx);
        const double d1 = y - (b.cy - e), d2 = y - (b.cy + e);
        return dx2 + d1 * d1 <= ra2 && dx2 + d2 * d2 <= ra2;
    };
    // Solid y-interval at fixed x: both disks share s = sqrt(ra^2 - dx^2), the
    // intersection is [cy+e-s, cy-e+s] (empty when s <= e, i.e. |dx| >= w).
    auto chordY = [&](double x, double& lo, double& hi) {
        const double q = ra2 - (x - b.cx) * (x - b.cx);
        const double s = q > 0 ? std::sqrt(q) : 0;
        lo = b.cy + e - s;
        hi = b.cy - e + s;
    };
    // Solid x-interval at fixed y: cx +- min(s1, s2) over the two disks.
    auto chordX = [&](double y, double& lo, double& hi) {
        const double d1 = y - (b.cy - e), d2 = y - (b.cy + e);
        const double q = ra2 - std::max(d1 * d1, d2 * d2);
        const double s = q > 0 ? std::sqrt(q) : 0;
        lo = b.cx - s;
        hi = b.cx + s;
    };

    const int i0 = std::max(0, int((b.cx - b.w) / g.hx) - 2);
    const int i1 = std::min(g.nx - 1, int((b.cx + b.w) / g.hx) + 2);
    const int j0 = std::max(0, int((b.cy - b.hh) / g.hy) - 2);
    const int j1 = std::min(g.ny - 1, int((b.cy + b.hh) / g.hy) + 2);
    for (int j = j0; j <= j1; ++j)
        for (int i = i0; i <= i1; ++i)
        {
            const double x0 = i * g.hx, y0 = j * g.hy;
            double lo, hi;
            chordY(x0, lo, hi);
            const double sy =
                std::max(0.0, std::min(hi, y0 + g.hy) - std::max(lo, y0));
            chordX(y0, lo, hi);
            const double sx =
                std::max(0.0, std::min(hi, x0 + g.hx) - std::max(lo, x0));
            // All four corners inside a convex solid means fully solid;
            // otherwise supersample.
            double lam;
            if (inside(x0, y0) && inside(x0 + g.hx, y0) && inside(x0, y0 + g.hy)
                && inside(x0 + g.hx, y0 + g.hy))
                lam = 0;
            else
            {
                int hit = 0;
                for (int a = 0; a < ns; ++a)
                    for (int c = 0; c < ns; ++c)
                        hit += inside(x0 + (a + 0.5) * g.hx / ns,
                                      y0 + (c + 0.5) * g.hy / ns);
                lam = 1.0 - double(hit) / (double(ns) * ns);
            }
            for (int k = 0; k < g.nz; ++k)
            {
                const std::size_t c = g.idx(i, j, k);
                cut.ax[c] = 1.0 - sy / g.hy;
                cut.ay[c] = 1.0 - sx / g.hx;
                cut.lambda[c] = lam;
                cut.az[c] = lam;
            }
        }
    return cut;
}

// --- reference dynamic pressure ---------------------------------------------

namespace {

// A (j,k) stream column is blocked by 1 - min_i ax over the given i range,
// weighted by the REFERENCE cross-section. See geometry.hpp for the measured
// reason those are the right weights even on a clustered (--focal) mesh.
double frontalArea(const Grid& g, const CutGeometry& cut, int i0, int i1,
                   int j0, int j1, int k0, int k1)
{
    double a = 0.0;
    for (int k = k0; k <= k1; ++k)
        for (int j = j0; j <= j1; ++j)
        {
            double amin = 1.0;
            for (int i = i0; i <= i1; ++i)
                amin = std::min(amin, cut.ax[g.idx(i, j, k)]);
            // nz == 1 is a real thickness (hz == 1, validator-enforced), not a
            // degenerate axis to skip.
            a += (1.0 - amin) * g.hy * g.hz;
        }
    return a;
}

} // namespace

ReferenceQ referenceQIncomp(const Grid& g, const CutGeometry& cut,
                            double rhoInf, double uInf)
{
    // Interior only: these grids carry a one-cell ghost ring.
    const double a = frontalArea(g, cut, 2, g.nx - 2, 1, g.ny - 2, 1,
                                 g.nz == 1 ? 1 : g.nz - 2);
    ReferenceQ r;
    r.area = a;
    r.measured = a > 0.0;
    r.q = 0.5 * rhoInf * uInf * uInf * a;
    return r;
}

ReferenceQ referenceQComp(const Grid& g, const CutGeometry& cut, double rhoInf,
                          double uInf, double D)
{
    // No ghost ring; the full extent is swept.
    const double a = frontalArea(g, cut, 1, g.nx - 1, 0, g.ny - 1, 0, g.nz - 1);
    const double span = g.nz > 1 ? double(g.nz) * g.hz : 1.0;  // per unit depth
    ReferenceQ r;
    r.area = a;
    r.measured = a > 0.0;
    r.q = r.measured ? 0.5 * rhoInf * uInf * uInf * (a / span)
                     : 0.5 * rhoInf * uInf * uInf * D;
    return r;
}

FloorRows measureFloorRows(const Grid& g, const CutGeometry& cut)
{
    const int k0 = g.nz == 1 ? 0 : 1;
    int fr = 0;
    while (fr + 1 <= g.ny - 2 && cut.ax[g.idx(1, fr + 1, k0)] == 0.0) ++fr;
    if (fr + 1 <= g.ny - 2 && cut.ax[g.idx(1, fr + 1, k0)] < 0.5) ++fr;
    FloorRows f;
    f.rows = fr;
    if (fr + 1 <= g.ny - 2) f.frac = cut.ax[g.idx(1, fr + 1, k0)];
    return f;
}

} // namespace dns
