#include "dns/cutcell.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <vector>

namespace dns {

namespace {

// Overlap length of [a0,a1] with [b0,b1] (0 if disjoint).
double overlap(double a0, double a1, double b0, double b1)
{
    return std::max(0.0, std::min(a1, b1) - std::max(a0, b0));
}

// Solid (diamond) fraction of the axis-aligned cell square [x0,x0+hx] x
// [y0,y0+hy] by Sutherland-Hodgman clipping against the 4 half-planes
// s*(x-cx) + t*(y-cy) <= r, (s,t) in {+-1}^2. Returns clipped area / cell area.
double diamondAreaFrac(double x0, double y0, double hx, double hy,
                       const RotatedCube& c)
{
    struct P { double x, y; };
    std::vector<P> poly{{x0, y0}, {x0 + hx, y0}, {x0 + hx, y0 + hy}, {x0, y0 + hy}};
    const std::array<std::array<double, 2>, 4> st{{{1, 1}, {-1, 1}, {1, -1}, {-1, -1}}};
    for (const auto& e : st)
    {
        const double s = e[0], t = e[1];
        auto f = [&](const P& p) { return s * (p.x - c.cx) + t * (p.y - c.cy) - c.r; };
        std::vector<P> out;
        for (std::size_t i = 0; i < poly.size(); ++i)
        {
            const P& a = poly[i];
            const P& b = poly[(i + 1) % poly.size()];
            const double fa = f(a), fb = f(b);
            if (fa <= 0) out.push_back(a);
            if ((fa < 0) != (fb < 0))
            {
                const double u = fa / (fa - fb);
                out.push_back({a.x + u * (b.x - a.x), a.y + u * (b.y - a.y)});
            }
        }
        poly.swap(out);
        if (poly.empty()) return 0.0;
    }
    double area2 = 0.0;
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const P& a = poly[i];
        const P& b = poly[(i + 1) % poly.size()];
        area2 += a.x * b.y - b.x * a.y;
    }
    return std::abs(area2) * 0.5 / (hx * hy);
}

// Fluid fraction of an axis-parallel segment. The diamond cross-section at fixed
// coord p (distance dp = |p - cCoord|) spans [cOther - (r-dp), cOther + (r-dp)]
// in the other coordinate when dp < r. seg is [s0,s1] in the other coordinate.
double segFluidFrac(double dp, double cOther, double s0, double s1,
                    const RotatedCube& c)
{
    const double half = c.r - dp;
    if (half <= 0) return 1.0;
    const double solid = overlap(s0, s1, cOther - half, cOther + half);
    return 1.0 - solid / (s1 - s0);
}

// --- exact circle/rectangle overlap --------------------------------------
// Area of the disk X^2+Y^2 <= R^2 intersected with [x0,x1] x [y0,y1] (circle
// already shifted to the origin). Integrates max(0, min(y1,f) - max(y0,-f))
// with f(x) = sqrt(R^2-x^2) over x, splitting at every crossing of f with
// |y0|,|y1| so the min/max selections are constant per piece and each piece
// has a closed-form antiderivative.
double circleRectOverlap(double R, double x0, double x1, double y0, double y1)
{
    if (R <= 0.0 || x1 <= x0 || y1 <= y0) return 0.0;
    x0 = std::clamp(x0, -R, R);
    x1 = std::clamp(x1, -R, R);
    if (x1 <= x0) return 0.0;

    // F = int sqrt(R^2-x^2) dx
    auto F = [R](double x) {
        const double s = std::sqrt(std::max(0.0, R * R - x * x));
        return 0.5 * (x * s + R * R * std::asin(std::clamp(x / R, -1.0, 1.0)));
    };
    auto f = [R](double x) { return std::sqrt(std::max(0.0, R * R - x * x)); };

    std::array<double, 6> bp{x0, x1, 0, 0, 0, 0};
    std::size_t nb = 2;
    for (double y : {y0, y1})
        if (std::abs(y) < R)
        {
            const double xc = std::sqrt(R * R - y * y);
            bp[nb++] = std::clamp(-xc, x0, x1);
            bp[nb++] = std::clamp(xc, x0, x1);
        }
    std::sort(bp.begin(), bp.begin() + nb);

    double area = 0.0;
    for (std::size_t s = 0; s + 1 < nb; ++s)
    {
        const double a = bp[s], b = bp[s + 1];
        if (b - a <= 0.0) continue;
        const double fm = f(0.5 * (a + b));
        const double top = std::min(y1, fm), bot = std::max(y0, -fm);
        if (top <= bot) continue;
        area += fm < y1 ? F(b) - F(a) : y1 * (b - a);   // integral of the top
        area -= -fm > y0 ? -(F(b) - F(a)) : y0 * (b - a); // integral of the bottom
    }
    return area;
}

// Adaptive Simpson on [a,b] to absolute tolerance tol (integrand piecewise
// smooth: the sphere cross-section area as a function of the slice height).
template <class G>
double adaptSimpson(const G& fn, double a, double b, double fa, double fm,
                    double fb, double whole, double tol, int depth)
{
    const double m = 0.5 * (a + b);
    const double lm = 0.5 * (a + m), rm = 0.5 * (m + b);
    const double flm = fn(lm), frm = fn(rm);
    const double left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
    const double right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
    if (depth <= 0 || std::abs(left + right - whole) <= 15.0 * tol)
        return left + right + (left + right - whole) / 15.0;
    return adaptSimpson(fn, a, m, fa, flm, fm, left, 0.5 * tol, depth - 1)
         + adaptSimpson(fn, m, b, fm, frm, fb, right, 0.5 * tol, depth - 1);
}

template <class G>
double integrate(const G& fn, double a, double b, double tol)
{
    if (b <= a) return 0.0;
    const double m = 0.5 * (a + b);
    const double fa = fn(a), fm = fn(m), fb = fn(b);
    const double whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    return adaptSimpson(fn, a, b, fa, fm, fb, whole, tol, 24);
}

// Solid volume of the sphere within the axis box [x0,x0+hx]x[y0,..]x[z0,..],
// by integrating the (exact) circular cross-section overlap along z. The
// integrand kinks where the slice circle passes an edge or corner of the
// rectangle; splitting there first keeps every adaptive piece smooth (without
// the split, kink cells drive the recursion to its depth cap and the geometry
// build diverges at high resolution).
double sphereBoxSolidVolume(double x0, double y0, double z0, double hx,
                            double hy, double hz, const Sphere& s)
{
    const double zlo = std::max(z0, s.cz - s.r), zhi = std::min(z0 + hz, s.cz + s.r);
    if (zhi <= zlo) return 0.0;

    const double rx0 = x0 - s.cx, rx1 = x0 + hx - s.cx;
    const double ry0 = y0 - s.cy, ry1 = y0 + hy - s.cy;
    auto sliceArea = [&](double z) {
        const double rc2 = s.r * s.r - (z - s.cz) * (z - s.cz);
        if (rc2 <= 0.0) return 0.0;
        return circleRectOverlap(std::sqrt(rc2), rx0, rx1, ry0, ry1);
    };

    // Slice-circle radii at which the integrand kinks: the 4 edge lines and
    // 4 corners of the rectangle. rc(z) = d at z = cz +- sqrt(r^2 - d^2).
    std::array<double, 18> bp{zlo, zhi};
    std::size_t nb = 2;
    const std::array<double, 4> ex{std::abs(rx0), std::abs(rx1),
                                   std::abs(ry0), std::abs(ry1)};
    auto addBreak = [&](double d2) {
        const double h2 = s.r * s.r - d2;
        if (h2 <= 0.0) return;
        const double dz = std::sqrt(h2);
        for (double z : {s.cz - dz, s.cz + dz})
            if (z > zlo && z < zhi) bp[nb++] = z;
    };
    for (double d : ex) addBreak(d * d);
    for (double dx : {rx0, rx1})
        for (double dy : {ry0, ry1}) addBreak(dx * dx + dy * dy);
    std::sort(bp.begin(), bp.begin() + nb);

    const double tol = 1e-12 * hx * hy * (zhi - zlo);
    double vol = 0.0;
    for (std::size_t i = 0; i + 1 < nb; ++i)
        vol += integrate(sliceArea, bp[i], bp[i + 1], tol);
    return vol;
}

} // namespace

CutGeometry CutGeometry::make(const Grid& g, const RotatedCube& c)
{
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    for (int k = 0; k < g.nz; ++k)
    {
        const double zf = k * g.hz;
        // Fraction of THIS cell's z extent inside the slab. A centre-in/out
        // test was enough while the prism spanned every cell it touched; a
        // cube has two z faces that land mid-cell, and staircasing them would
        // leave z first-order against exact apertures in x and y. The slab is
        // axis-aligned, so the exact fraction is just the 1D overlap.
        const double zFrac = overlap(zf, zf + g.hz, c.z0, c.z1) / g.hz;
        const bool faceInSlab = zf >= c.z0 && zf <= c.z1; // w-face on/inside prism
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto id = g.idx(i, j, k);
                const double xf = i * g.hx, yf = j * g.hy;
                const double areaFrac = diamondAreaFrac(xf, yf, g.hx, g.hy, c);

                cut.lambda[id] = 1.0 - areaFrac * zFrac;
                // ax/ay are FACE apertures: rectangles spanning [zf, zf+hz] in
                // z, so the solid blocks zFrac of their height.
                cut.ax[id] =
                    1.0 - zFrac * (1.0 - segFluidFrac(std::abs(xf - c.cx), c.cy,
                                                      yf, yf + g.hy, c));
                cut.ay[id] =
                    1.0 - zFrac * (1.0 - segFluidFrac(std::abs(yf - c.cy), c.cx,
                                                      xf, xf + g.hx, c));
                // az lies ON the plane z = zf: a plane is in the slab or it is
                // not, so this one is genuinely binary.
                cut.az[id] = faceInSlab ? 1.0 - areaFrac : 1.0;
            }
    }
    return cut;
}

CutGeometry CutGeometry::make(const Grid& g, const CylinderZ& c)
{
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    // Chord the circle cuts from an x- or y-parallel face line at distance d
    // from the axis: fluid fraction of the [s0,s1] span (1 outside the circle).
    auto chordOpen = [&](double d, double cOther, double s0, double s1) {
        const double h2 = c.r * c.r - d * d;
        if (h2 <= 0.0) return 1.0;
        const double half = std::sqrt(h2);
        return 1.0 - overlap(s0, s1, cOther - half, cOther + half) / (s1 - s0);
    };
    // z-invariant: compute one x-y layer, replicate over k.
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
        {
            const double x0 = i * g.hx, y0 = j * g.hy;
            const double areaFrac = 1.0
                - circleRectOverlap(c.r, x0 - c.cx, x0 + g.hx - c.cx,
                                    y0 - c.cy, y0 + g.hy - c.cy)
                / (g.hx * g.hy);
            const double ax = chordOpen(x0 - c.cx, c.cy, y0, y0 + g.hy);
            const double ay = chordOpen(y0 - c.cy, c.cx, x0, x0 + g.hx);
            for (int k = 0; k < g.nz; ++k)
            {
                const auto id = g.idx(i, j, k);
                cut.lambda[id] = areaFrac;
                cut.ax[id] = ax;
                cut.ay[id] = ay;
                cut.az[id] = areaFrac;
            }
        }
    return cut;
}

CutGeometry CutGeometry::make(const Grid& g, const Sphere& s)
{
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    // Fluid fraction of a face rectangle cut by the sphere's cross-circle of
    // radius sqrt(r^2 - d^2) at plane distance d (aperture 1 when d >= r).
    auto faceOpen = [&](double d, double a0, double b0, double ha, double hb) {
        const double rc2 = s.r * s.r - d * d;
        if (rc2 <= 0.0) return 1.0;
        const double solid =
            circleRectOverlap(std::sqrt(rc2), a0, a0 + ha, b0, b0 + hb);
        return 1.0 - solid / (ha * hb);
    };
    const double halfDiag =
        0.5 * std::sqrt(g.hx * g.hx + g.hy * g.hy + g.hz * g.hz);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto id = g.idx(i, j, k);
                const double x0 = i * g.hx, y0 = j * g.hy, z0 = k * g.hz;

                // Classify by centre distance so only shell cells integrate.
                const double dx = x0 + 0.5 * g.hx - s.cx,
                             dy = y0 + 0.5 * g.hy - s.cy,
                             dz = z0 + 0.5 * g.hz - s.cz;
                const double dc = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dc >= s.r + halfDiag)
                    cut.lambda[id] = 1.0;
                else if (dc <= s.r - halfDiag)
                    cut.lambda[id] = 0.0;
                else
                    cut.lambda[id] = 1.0 - sphereBoxSolidVolume(
                        x0, y0, z0, g.hx, g.hy, g.hz, s) / g.cellVolume();

                cut.ax[id] = faceOpen(x0 - s.cx, y0 - s.cy, z0 - s.cz, g.hy, g.hz);
                cut.ay[id] = faceOpen(y0 - s.cy, x0 - s.cx, z0 - s.cz, g.hx, g.hz);
                cut.az[id] = faceOpen(z0 - s.cz, x0 - s.cx, y0 - s.cy, g.hx, g.hy);
            }
    return cut;
}

std::vector<unsigned char> buildCutFlag(const Grid& g, const CutGeometry& cut)
{
    // Seed: any cell whose volume or any INCIDENT face aperture is not exactly
    // 1. A cell's high faces are its +1 neighbours' low faces; checking both
    // sides keeps the seed symmetric so the dilation below is what sets the
    // reach, not the storage convention.
    const std::size_t n = g.size();
    std::vector<unsigned char> seed(n, 0), flag(n, 0);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const std::size_t c = g.idx(i, j, k);
                if (cut.lambda[c] < 1.0 || cut.ax[c] < 1.0 || cut.ay[c] < 1.0
                    || cut.az[c] < 1.0 || cut.ax[g.idx(i + 1, j, k)] < 1.0
                    || cut.ay[g.idx(i, j + 1, k)] < 1.0
                    || cut.az[g.idx(i, j, k + 1)] < 1.0)
                    seed[c] = 1;
            }
    // Dispatch radius 2 (see header): fixed-order box dilation, periodic wrap.
    constexpr int r = 2;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                if (!seed[g.idx(i, j, k)]) continue;
                const int dzr = g.nz == 1 ? 0 : r; // nz==1: z wraps to itself
                for (int dk = -dzr; dk <= dzr; ++dk)
                    for (int dj = -r; dj <= r; ++dj)
                        for (int di = -r; di <= r; ++di)
                            flag[g.idx(i + di, j + dj, k + dk)] = 1;
            }
    return flag;
}

void clearBoundaryFlag(const Grid& g, std::vector<unsigned char>& flag,
                       int width)
{
    assert(flag.size() == g.size());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const bool nearZ =
                    g.nz > 1 && (k < width || k >= g.nz - width);
                if (i < width || i >= g.nx - width || j < width
                    || j >= g.ny - width || nearZ)
                    flag[g.idx(i, j, k)] = 0;
            }
}

CutSanitizeStats sanitizeCut(const Grid& g, CutGeometry& cut, double lambdaMin,
                             double apertureMin)
{
    assert(lambdaMin >= 0.0 && lambdaMin < 1.0);
    assert(apertureMin >= 0.0 && apertureMin < 1.0);
    CutSanitizeStats st{0, 0, 0, 1.0, 1.0};

    for (auto& l : cut.lambda)
        if (l > 0.0 && l < lambdaMin) { l = 0.0; ++st.cellsSolidified; }

    // Close undersized and solid-adjacent faces, then solidify fluid cells
    // left with no open face; repeat until nothing changes. (One face pass
    // plus one pocket pass converges for any thresholded geometry, but the
    // fixed point is cheap and makes the postcondition unconditional.)
    const double tol = CutGeometry::openTol;
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                {
                    const auto c = g.idx(i, j, k);
                    auto close = [&](Scalar& a, std::size_t nbr) {
                        if (a[c] <= 0.0) return;
                        if (a[c] < apertureMin || cut.lambda[c] <= 0.0
                            || cut.lambda[nbr] <= 0.0)
                        {
                            a[c] = 0.0;
                            ++st.facesClosed;
                            changed = true;
                        }
                    };
                    close(cut.ax, g.idx(i - 1, j, k));
                    close(cut.ay, g.idx(i, j - 1, k));
                    close(cut.az, g.idx(i, j, k - 1));
                }
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                {
                    const auto c = g.idx(i, j, k);
                    if (cut.lambda[c] <= 0.0) continue;
                    const bool open =
                        cut.ax[c] > tol || cut.ax[g.idx(i + 1, j, k)] > tol
                     || cut.ay[c] > tol || cut.ay[g.idx(i, j + 1, k)] > tol
                     || cut.az[c] > tol || cut.az[g.idx(i, j, k + 1)] > tol;
                    if (!open)
                    {
                        cut.lambda[c] = 0.0;
                        ++st.pocketsClosed;
                        changed = true;
                    }
                }
    }

    for (double l : cut.lambda)
        if (l > 0.0) st.minLambda = std::min(st.minLambda, l);
    for (const Scalar* a : {&cut.ax, &cut.ay, &cut.az})
        for (double v : *a)
            if (v > tol) st.minAperture = std::min(st.minAperture, v);
    return st;
}

void closeBoxBoundary(const Grid& g, CutGeometry& cut)
{
    // A degenerate axis (n == 1, the 2D case) stays periodic: no ghosts, none
    // of its faces closed.
    const bool cz = g.nz > 1;
    assert(g.nx >= 4 && g.ny >= 4 && (!cz || g.nz >= 4));
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                const bool gi = i == 0 || i == g.nx - 1;
                const bool gj = j == 0 || j == g.ny - 1;
                const bool gk = cz && (k == 0 || k == g.nz - 1);
                if (gi || gj || gk) cut.lambda[c] = 0.0;
                if (i <= 1 || i == g.nx - 1 || gj || gk) cut.ax[c] = 0.0;
                if (j <= 1 || j == g.ny - 1 || gi || gk) cut.ay[c] = 0.0;
                if ((cz && (k <= 1 || k == g.nz - 1)) || gi || gj) cut.az[c] = 0.0;
            }
}

void divergenceCut(const Grid& g, const CutGeometry& cut, const VectorField& U,
                   Scalar& out)
{
    assert(out.size() == g.size());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                const auto xp = g.idx(i + 1, j, k), yp = g.idx(i, j + 1, k),
                           zp = g.idx(i, j, k + 1);
                out[c] =
                    (cut.ax[xp] * U.u[xp] - cut.ax[c] * U.u[c]) / g.hx
                  + (cut.ay[yp] * U.v[yp] - cut.ay[c] * U.v[c]) / g.hy
                  + (cut.az[zp] * U.w[zp] - cut.az[c] * U.w[c]) / g.hz;
            }
}

void subtractGradientCut(const Grid& g, const CutGeometry& cut, const Scalar& p,
                         double scale, VectorField& U)
{
    assert(p.size() == g.size());
    const double tol = CutGeometry::openTol;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                if (cut.ax[c] > tol)
                    U.u[c] -= scale * (p[c] - p[g.idx(i - 1, j, k)]) / g.hx;
                if (cut.ay[c] > tol)
                    U.v[c] -= scale * (p[c] - p[g.idx(i, j - 1, k)]) / g.hy;
                if (cut.az[c] > tol)
                    U.w[c] -= scale * (p[c] - p[g.idx(i, j, k - 1)]) / g.hz;
            }
}

void maskSolidFaces(const Grid& g, const CutGeometry& cut, VectorField& U)
{
    const double tol = CutGeometry::openTol;
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        if (cut.ax[c] <= tol) U.u[c] = 0.0;
        if (cut.ay[c] <= tol) U.v[c] = 0.0;
        if (cut.az[c] <= tol) U.w[c] = 0.0;
    }
}

double maxAbsDivergenceCut(const Grid& g, const CutGeometry& cut,
                           const VectorField& U)
{
    double m = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                if (cut.lambda[c] <= 0.0) continue;
                const auto xp = g.idx(i + 1, j, k), yp = g.idx(i, j + 1, k),
                           zp = g.idx(i, j, k + 1);
                const double d =
                    (cut.ax[xp] * U.u[xp] - cut.ax[c] * U.u[c]) / g.hx
                  + (cut.ay[yp] * U.v[yp] - cut.ay[c] * U.v[c]) / g.hy
                  + (cut.az[zp] * U.w[zp] - cut.az[c] * U.w[c]) / g.hz;
                m = std::max(m, std::abs(d));
            }
    return m;
}

CutPoissonSolver::CutPoissonSolver(const Grid& g, const CutGeometry& cut,
                                   int maxIters, double tol, std::size_t nFluid)
    : g_(g), cut_(cut), maxIters_(maxIters), tol_(tol), nFluid_(nFluid),
      r_(zeros(g)), d_(zeros(g)), q_(zeros(g))
{
}

std::expected<CutPoissonSolver, std::string>
CutPoissonSolver::make(const Grid& g, const CutGeometry& cut, int maxIters,
                       double tol)
{
    std::size_t nFluid = 0;
    for (double l : cut.lambda) if (l > 0.0) ++nFluid;
    if (nFluid == 0)
        return std::unexpected("cut geometry has no fluid cell: the body fills "
                               "the whole box, or the cut failed to sample");
    return CutPoissonSolver(g, cut, maxIters, tol, nFluid);
}

// y = -M(x): SPD on the fluid subspace modulo the constant mode.
void CutPoissonSolver::apply(const Scalar& x, Scalar& y) const
{
    const double ix = 1.0 / (g_.hx * g_.hx), iy = 1.0 / (g_.hy * g_.hy),
                 iz = 1.0 / (g_.hz * g_.hz);
    for (int k = 0; k < g_.nz; ++k)
        for (int j = 0; j < g_.ny; ++j)
            for (int i = 0; i < g_.nx; ++i)
            {
                const auto c = g_.idx(i, j, k);
                if (cut_.lambda[c] <= 0.0) { y[c] = 0.0; continue; }
                const auto xp = g_.idx(i + 1, j, k), yp = g_.idx(i, j + 1, k),
                           zp = g_.idx(i, j, k + 1);
                const double m =
                    ix * (cut_.ax[xp] * (x[xp] - x[c]) - cut_.ax[c] * (x[c] - x[g_.idx(i - 1, j, k)]))
                  + iy * (cut_.ay[yp] * (x[yp] - x[c]) - cut_.ay[c] * (x[c] - x[g_.idx(i, j - 1, k)]))
                  + iz * (cut_.az[zp] * (x[zp] - x[c]) - cut_.az[c] * (x[c] - x[g_.idx(i, j, k - 1)]));
                y[c] = -m;
            }
}

double CutPoissonSolver::fluidDot(const Scalar& a, const Scalar& b) const
{
    double s = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c)
        if (cut_.lambda[c] > 0.0) s += a[c] * b[c];
    return s;
}

void CutPoissonSolver::projectMean(Scalar& x) const
{
    double s = 0.0;
    for (std::size_t c = 0; c < x.size(); ++c)
        if (cut_.lambda[c] > 0.0) s += x[c];
    const double mean = s / double(nFluid_);
    for (std::size_t c = 0; c < x.size(); ++c)
        x[c] = cut_.lambda[c] > 0.0 ? x[c] - mean : 0.0;
}

int CutPoissonSolver::solve(const Scalar& rhs, Scalar& p, bool warmStart)
{
    assert(rhs.size() == g_.size());
    if (!warmStart) p.assign(g_.size(), 0.0);
    else projectMean(p);

    // b = -(rhs - mean) on fluid cells (system is A p = b with A = -M).
    // r = b - A p (A*0 = 0 when cold).
    if (warmStart) apply(p, q_);
    for (std::size_t c = 0; c < g_.size(); ++c)
        r_[c] = cut_.lambda[c] > 0.0 ? -rhs[c] - (warmStart ? q_[c] : 0.0) : 0.0;
    projectMean(r_);
    d_ = r_;
    double rr = fluidDot(r_, r_);
    const double target = tol_ * tol_ * double(nFluid_);

    // A blown-up field makes rr NaN, and `NaN > target` is FALSE -- without
    // this guard the loop exits at iteration 0 and solve() returns 0, which is
    // indistinguishable from a perfect warm-started solve. Signal divergence
    // the same way the GPU CG does (cutpoisson.hip): return maxIters_, which
    // the callers already read as non-convergence (exit 8).
    if (!std::isfinite(rr)) return maxIters_;

    int it = 0;
    for (; it < maxIters_ && std::isfinite(rr) && rr > target; ++it)
    {
        apply(d_, q_);
        projectMean(q_);
        const double alpha = rr / fluidDot(d_, q_);
        for (std::size_t c = 0; c < g_.size(); ++c)
        {
            p[c] += alpha * d_[c];
            r_[c] -= alpha * q_[c];
        }
        const double rrNew = fluidDot(r_, r_);
        const double beta = rrNew / rr;
        for (std::size_t c = 0; c < g_.size(); ++c)
            d_[c] = r_[c] + beta * d_[c];
        rr = rrNew;
        if (!std::isfinite(rr)) return maxIters_;   // diverged mid-solve
    }
    projectMean(p);
    return it;
}

void projectCut(const Grid& g, const CutGeometry& cut, CutPoissonSolver& poisson,
                double dt, VectorField& U, CutProjectionScratch& s)
{
    assert(dt > 0);
    divergenceCut(g, cut, U, s.div);
    for (auto& x : s.div) x /= dt;
    poisson.solve(s.div, s.phi, true); // warm-start from the previous solve
    subtractGradientCut(g, cut, s.phi, dt, U);
    maskSolidFaces(g, cut, U);
}

} // namespace dns
