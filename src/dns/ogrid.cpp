#include "dns/ogrid.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <numbers>

namespace dns {
namespace {

using std::numbers::pi;

// Radial samples per ray for the sign-change scan: the contour is resolved to
// rFar/kMarch, and a solid shell thinner than that can be missed.
constexpr int kMarch = 1024;
// z planes sampled for contour z-invariance; odd, so the middle one is cz.
constexpr int kZPlanes = 5;

// A point ON the surface counts as outside, so a grazing sample cannot be
// mistaken for a crossing pair.
int sgn(double f) { return f < 0.0 ? -1 : 1; }

struct Ray
{
    double cx, cy, z, ct, st;
};

double along(const Sdf& sdf, const Ray& ray, double r)
{
    return sdf(ray.cx + r * ray.ct, ray.cy + r * ray.st, ray.z);
}

// Wall radius along one ray: march for the bracket (exactly one sign change
// required, which IS the star-shapedness test), then a safeguarded Newton
// with bisection fallback. Pre: rFar > 0.
std::expected<double, std::string> rayRoot(const Sdf& sdf, const Ray& ray,
                                           double rFar, double theta)
{
    const double f0 = along(sdf, ray, 0.0);
    if (!(f0 < 0.0))
        return std::unexpected(std::format(
            "the O-grid centre ({:.6g}, {:.6g}) is not inside the body: sdf = "
            "{:.3e} there at z = {:.6g}",
            ray.cx, ray.cy, f0, ray.z));

    int crossings = 0;
    double lo = 0.0, hi = rFar, prev = f0;
    for (int m = 1; m <= kMarch; ++m) {
        const double r = rFar * m / kMarch;
        const double f = along(sdf, ray, r);
        if (sgn(f) != sgn(prev) && ++crossings == 1) {
            lo = rFar * (m - 1) / kMarch;
            hi = r;
        }
        prev = f;
    }
    if (crossings == 0)
        return std::unexpected(std::format(
            "the body still fills the ray at theta = {:.6f} rad out to rFar = "
            "{:.6g}: the far boundary must clear the body",
            theta, rFar));
    if (crossings > 1)
        return std::unexpected(std::format(
            "the ray at theta = {:.6f} rad crosses the surface {} times: the "
            "body is not star-shaped about the O-grid centre",
            theta, crossings));

    double r = 0.5 * (lo + hi);
    for (int it = 0; it < 100; ++it) {
        const double f = along(sdf, ray, r);
        if (f > 0.0) hi = r; else lo = r;
        if (std::abs(f) < 1e-15 * rFar) break;
        const double d = std::min(1e-6 * rFar, 0.5 * r);
        const double fp =
            (along(sdf, ray, r + d) - along(sdf, ray, r - d)) / (2.0 * d);
        double rn = fp > 0.0 ? r - f / fp : 0.5 * (lo + hi);
        if (!(rn > lo && rn < hi)) rn = 0.5 * (lo + hi);
        if (rn == r) break;
        r = rn;
    }
    return r;
}

// Unit outward wall normal, sdf central difference at the wall point taken in
// the local (e_r, e_theta) frame with step 1e-3 rw. Both differences are
// symmetric about the wall point, so a circular contour gives an exactly zero
// tangential component and the normal is e_r to roundoff -- the Cartesian
// (x, y) difference does not, and its ~eps^(2/3) error would show up in the
// node positions. Features finer than the step are smoothed (a sharp tip gets
// its bisector, which is the mesh-friendly answer).
std::expected<std::array<double, 3>, std::string>
wallNormal(const Sdf& sdf, const OGridSpec& sp, double theta, double ct,
           double st, double rw)
{
    const double e = 1e-3 * rw;
    const double px = sp.cx + rw * ct, py = sp.cy + rw * st;
    const double fr = (sdf(px + e * ct, py + e * st, sp.cz) -
                       sdf(px - e * ct, py - e * st, sp.cz)) /
                      (2.0 * e);
    const double ft = (sdf(px - e * st, py + e * ct, sp.cz) -
                       sdf(px + e * st, py - e * ct, sp.cz)) /
                      (2.0 * e);
    if (!(fr > 0.0))
        return std::unexpected(std::format(
            "the sdf does not increase outward at theta = {:.6f} rad (d sdf / "
            "dr = {:.3e}): the wall normal is undefined there",
            theta, fr));
    const double nx = fr * ct - ft * st, ny = fr * st + ft * ct;
    const double len = std::hypot(nx, ny);
    if (!(len > 0.0))
        return std::unexpected(
            std::format("degenerate wall normal at theta = {:.6f} rad", theta));
    return std::array<double, 3>{nx / len, ny / len, 0.0};
}

} // namespace

double ogridRadialDist(double t, double beta)
{
    assert(t >= 0.0 && t <= 1.0);
    if (beta <= 0.0) return t; // exact uniform limit, bit-for-bit
    return 1.0 + std::tanh(beta * (t - 1.0)) / std::tanh(beta);
}

double ogridNominalH1(int ny, double rWall, double rFar, double beta)
{
    assert(ny >= 4 && rWall > 0.0 && rFar > rWall);
    return (rFar - rWall) * ogridRadialDist(1.0 / (ny - 2), beta);
}

double ogridRadialCells(double len, int ny, double rWall, double rFar,
                        double beta)
{
    assert(ny >= 4 && rWall > 0.0 && rFar > rWall && len >= 0.0);
    const int nf = ny - 2;
    const double target = len / (rFar - rWall);
    if (target >= 1.0) return nf;
    double lo = 0.0, hi = 1.0;
    for (int it = 0; it < 200 && hi - lo > 1e-15; ++it) {
        const double mid = 0.5 * (lo + hi);
        (ogridRadialDist(mid, beta) < target ? lo : hi) = mid;
    }
    return nf * 0.5 * (lo + hi);
}

Grid ogridGrid(int nx, int ny, int nz, double rWall, double rFar, double span)
{
    assert(nx >= 3 && ny >= 4 && nz >= 1);
    assert(rWall > 0.0 && rFar > rWall && span > 0.0);
    return Grid::make(nx, ny, nz, 2.0 * pi, ny * (rFar - rWall) / (ny - 2),
                      span);
}

std::expected<OGrid, std::string> buildOGrid(const Grid& g, const Sdf& sdf,
                                             const OGridSpec& sp)
{
    if (g.nx < 3)
        return std::unexpected(std::format(
            "an O-grid needs nx >= 3 angular cells, got {}", g.nx));
    if (g.ny < 4)
        return std::unexpected(std::format(
            "an O-grid needs ny >= 4 radial cells (two ghost rows plus at "
            "least two fluid rows), got {}",
            g.ny));
    if (g.nz < 1)
        return std::unexpected("an O-grid needs nz >= 1 spanwise cells");
    if (!sdf) return std::unexpected("an O-grid needs a body sdf");
    if (!(sp.rFar > 0.0))
        return std::unexpected(
            std::format("rFar must be positive, got {:.6g}", sp.rFar));
    if (!(sp.span > 0.0))
        return std::unexpected(
            std::format("span must be positive, got {:.6g}", sp.span));
    if (!(sp.beta >= 0.0))
        return std::unexpected(std::format(
            "radial clustering beta must be >= 0, got {:.6g}", sp.beta));
    if (std::abs(g.nx * g.hx - 2.0 * pi) > 1e-12 * 2.0 * pi)
        return std::unexpected(std::format(
            "the angular axis must span 2 pi (nx hx = {:.15g}); size the grid "
            "with ogridGrid()",
            g.nx * g.hx));
    if (std::abs(g.nz * g.hz - sp.span) > 1e-12 * sp.span)
        return std::unexpected(std::format(
            "the spanwise axis must span span = {:.15g} (nz hz = {:.15g}); "
            "size the grid with ogridGrid()",
            sp.span, g.nz * g.hz));

    const int nx = g.nx, ny = g.ny, nz = g.nz, nf = ny - 2;
    const double zTol = 1e-9 * sp.rFar;

    // --- contour: wall radius, wall normal, and the normal-ray length that
    // lands on the far circle, one per angle.
    std::vector<double> cth(nx), sth(nx), rw(nx), ln(nx);
    std::vector<std::array<double, 3>> nh(nx);
    for (int i = 0; i < nx; ++i) {
        // Clockwise, so (i, j, k) stays right-handed; i == 0 kept off -0.0 so
        // the diagnostics read cleanly.
        const double th = i == 0 ? 0.0 : -2.0 * pi * i / nx;
        cth[i] = std::cos(th);
        sth[i] = std::sin(th);
        const Ray mid{sp.cx, sp.cy, sp.cz, cth[i], sth[i]};
        const auto r0 = rayRoot(sdf, mid, sp.rFar, th);
        if (!r0) return std::unexpected(r0.error());
        for (int p = 0; p < kZPlanes; ++p) {
            const double z =
                sp.cz + sp.span * (double(p) / (kZPlanes - 1) - 0.5);
            const auto rp =
                rayRoot(sdf, Ray{sp.cx, sp.cy, z, cth[i], sth[i]}, sp.rFar, th);
            if (!rp) return std::unexpected(rp.error());
            if (std::abs(*rp - *r0) > zTol)
                return std::unexpected(std::format(
                    "the contour is not z-invariant: at theta = {:.6f} rad the "
                    "wall radius is {:.12g} at z = {:.6g} but {:.12g} at the "
                    "centre plane z = {:.6g} (tolerance {:.1e}); an O-grid is "
                    "an extrusion",
                    th, *rp, z, *r0, sp.cz, zTol));
        }
        rw[i] = *r0;
        const auto n = wallNormal(sdf, sp, th, cth[i], sth[i], rw[i]);
        if (!n) return std::unexpected(n.error());
        nh[i] = *n;
        const double mu = nh[i][0] * cth[i] + nh[i][1] * sth[i];
        const double disc =
            sp.rFar * sp.rFar - rw[i] * rw[i] * (1.0 - mu * mu);
        ln[i] = disc > 0.0 ? -rw[i] * mu + std::sqrt(disc) : 0.0;
        if (!(ln[i] > 0.0))
            return std::unexpected(std::format(
                "the wall-normal ray at theta = {:.6f} rad cannot reach the "
                "far circle (wall radius {:.6g}, rFar {:.6g}): the far "
                "boundary must clear the body",
                th, rw[i], sp.rFar));
    }

    // --- nodes. Row 1 is the wall, row ny-1 the far circle, row 0 the point
    // reflection of row 2 through the wall node (so the wall face sits exactly
    // midway and a value-mirror ghost puts u = 0, T = Twall on it).
    std::vector<double> px(std::size_t(nx) * ny), py(std::size_t(nx) * ny);
    for (int j = 1; j < ny; ++j) {
        const double t = double(j - 1) / nf;
        const double d = ogridRadialDist(t, sp.beta);
        const double w = t * t * (3.0 - 2.0 * t);
        for (int i = 0; i < nx; ++i) {
            const double bx = (1.0 - w) * ln[i] * nh[i][0] +
                              w * (sp.rFar - rw[i]) * cth[i];
            const double by = (1.0 - w) * ln[i] * nh[i][1] +
                              w * (sp.rFar - rw[i]) * sth[i];
            px[std::size_t(j) * nx + i] = sp.cx + rw[i] * cth[i] + d * bx;
            py[std::size_t(j) * nx + i] = sp.cy + rw[i] * sth[i] + d * by;
        }
    }
    for (int i = 0; i < nx; ++i) {
        px[i] = 2.0 * px[std::size_t(nx) + i] - px[std::size_t(2) * nx + i];
        py[i] = 2.0 * py[std::size_t(nx) + i] - py[std::size_t(2) * nx + i];
    }

    OGrid out{.mesh = CurvMesh::uniform(g),
              .rWall = rw,
              .wallNormal = nh,
              .h1 = 1e300,
              .h1Max = 0.0,
              .growth = 1.0,
              .rWallMin = *std::ranges::min_element(rw),
              .rWallMax = *std::ranges::max_element(rw),
              .normalDevMax = 0.0};
    out.mesh.wrap = {AxisWrap::Cyclic, AxisWrap::Extend, AxisWrap::Translate};
    const double z0 = sp.cz - 0.5 * sp.span;
    for (int k = 0; k < nz; ++k) {
        const double zk = z0 + k * g.hz;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                out.mesh.x[c] = px[std::size_t(j) * nx + i];
                out.mesh.y[c] = py[std::size_t(j) * nx + i];
                out.mesh.z[c] = zk;
            }
    }

    // --- radial diagnostics over the FLUID rows only (the ghost rows mirror
    // their neighbours by construction).
    for (int i = 0; i < nx; ++i) {
        double prev = 0.0;
        for (int j = 1; j <= nf; ++j) {
            const std::size_t a = std::size_t(j) * nx + i,
                              b = std::size_t(j + 1) * nx + i;
            const double ex = px[b] - px[a], ey = py[b] - py[a];
            const double len = std::hypot(ex, ey);
            if (!(len > 0.0))
                return std::unexpected(std::format(
                    "radial edge {} collapsed at theta = {:.6f} rad: the "
                    "clustering or the contour folds the mesh",
                    j, -2.0 * pi * i / nx));
            if (j == 1) {
                out.h1 = std::min(out.h1, len);
                out.h1Max = std::max(out.h1Max, len);
                const double dot = (ex * nh[i][0] + ey * nh[i][1]) / len;
                out.normalDevMax = std::max(
                    out.normalDevMax, std::acos(std::clamp(dot, -1.0, 1.0)));
            } else {
                out.growth = std::max(out.growth, len / prev);
            }
            prev = len;
        }
    }
    return out;
}

Grid ogridRevolvedGrid(int nTheta, int ny, const RevolvedSpec& r, double rWall,
                       double rFar)
{
    assert(nTheta >= 4 && ny >= 4 && rWall > 0.0 && rFar > rWall);
    assert(r.wedge ? r.dphi > 0.0 : (r.nphi >= 4 && r.nphi % 2 == 0));
    const int nx = nTheta + 2, nz = r.wedge ? 3 : r.nphi;
    const double hz = r.wedge ? r.dphi : 2.0 * pi / r.nphi;
    return Grid::make(nx, ny, nz, nx * pi / nTheta, ny * (rFar - rWall) / (ny - 2),
                      nz * hz);
}

namespace {

// cos and sin of theta = pi m / n, evaluated on the half nearer 0 and
// reflected, so m == 0 gives (1, 0) and m == n gives (-1, 0) EXACTLY: the
// two axis node columns then sit at rho == 0 to the bit.
std::array<double, 2> polar(int m, int n)
{
    if (2 * m <= n) {
        const double t = pi * m / n;
        return {std::cos(t), std::sin(t)};
    }
    const double t = pi * (n - m) / n;
    return {-std::cos(t), std::sin(t)};
}

// The sdf along the meridional ray at polar angle theta (cos ct, sin st) in
// the half-plane phi: point (cx - r ct, cy + r st cos phi, cz + r st sin
// phi). Distance r from the centre.
double alongMeridian(const Sdf& sdf, const OGridSpec& sp, double ct, double st,
                     double cphi, double sphi, double r)
{
    return sdf(sp.cx - r * ct, sp.cy + r * st * cphi, sp.cz + r * st * sphi);
}

} // namespace

std::expected<OGrid, std::string> buildRevolvedOGrid(const Grid& g,
                                                     const Sdf& sdf,
                                                     const OGridSpec& sp,
                                                     const RevolvedSpec& rv)
{
    if (!sdf) return std::unexpected("a revolved O-grid needs a body sdf");
    if (!(sp.rFar > 0.0) || !(sp.beta >= 0.0))
        return std::unexpected("rFar must be positive and beta >= 0");
    if (rv.wedge ? !(rv.dphi > 0.0) : (rv.nphi < 4 || rv.nphi % 2))
        return std::unexpected("revolved layout: a wedge angle > 0, or an even "
                               "azimuthal count >= 4");
    const int nTheta = g.nx - 2, ny = g.ny, nz = g.nz, nf = ny - 2;
    if (nTheta < 4 || ny < 4 || nz != (rv.wedge ? 3 : rv.nphi))
        return std::unexpected(std::format(
            "a revolved O-grid needs nx = nTheta + 2 >= 6, ny >= 4 and nz = {}"
            "; got {} x {} x {} (size it with ogridRevolvedGrid())",
            rv.wedge ? 3 : rv.nphi, g.nx, ny, nz));

    // --- the meridional contour: wall radius, meridional wall normal and
    // the normal-ray length to the far circle, per node column i = 1..nT+1.
    const double zTol = 1e-9 * sp.rFar;
    std::vector<double> ct(g.nx), st(g.nx), rw(g.nx), ln(g.nx);
    std::vector<std::array<double, 2>> nh(g.nx); // (n_x', n_rho) in -e_r frame
    for (int i = 1; i <= nTheta + 1; ++i) {
        const auto cs = polar(i - 1, nTheta);
        ct[i] = cs[0];
        st[i] = cs[1];
        const double th = pi * (i - 1) / nTheta;
        // The wall radius along this meridian at 5 azimuths: one value, or
        // the body is not a body of revolution about x.
        double r0 = 0;
        for (int q = 0; q < 5; ++q) {
            const double phi = 2.0 * pi * q / 5.0;
            const double cp = std::cos(phi), spp = std::sin(phi);
            auto f = [&](double r) {
                return alongMeridian(sdf, sp, ct[i], st[i], cp, spp, r);
            };
            if (!(f(0.0) < 0.0))
                return std::unexpected(std::format(
                    "the O-grid centre ({:.6g}, {:.6g}, {:.6g}) is not inside "
                    "the body",
                    sp.cx, sp.cy, sp.cz));
            int crossings = 0;
            double lo = 0, hi = sp.rFar, prev = f(0.0);
            for (int m = 1; m <= 1024; ++m) {
                const double r = sp.rFar * m / 1024;
                const double v = f(r);
                if ((v < 0.0) != (prev < 0.0) && ++crossings == 1) {
                    lo = sp.rFar * (m - 1) / 1024;
                    hi = r;
                }
                prev = v;
            }
            if (crossings != 1)
                return std::unexpected(std::format(
                    "the meridian at theta = {:.6f} rad crosses the surface {} "
                    "times: the body is not star-shaped about the centre",
                    th, crossings));
            for (int it = 0; it < 200 && hi - lo > 1e-15 * sp.rFar; ++it) {
                const double mid = 0.5 * (lo + hi);
                (f(mid) > 0.0 ? hi : lo) = mid;
            }
            const double r = 0.5 * (lo + hi);
            if (q == 0) r0 = r;
            else if (std::abs(r - r0) > zTol)
                return std::unexpected(std::format(
                    "the body is not a body of revolution about x: at theta = "
                    "{:.6f} rad the wall radius is {:.12g} at phi = {:.4f} but "
                    "{:.12g} at phi = 0",
                    th, r, phi, r0));
        }
        rw[i] = r0;
        // Meridional wall normal (radial, tangential parts by symmetric
        // differences, as buildOGrid's wallNormal). ON the axis the normal
        // is along it by symmetry: taken exactly, so rho stays exactly 0.
        double nr = 1.0, nt = 0.0;
        if (i != 1 && i != nTheta + 1) {
            const double e = 1e-3 * rw[i];
            auto at = [&](double dr, double dt) {
                // (dr along e_r, dt along e_theta) from the wall point.
                const double X = -rw[i] * ct[i] - dr * ct[i] + dt * st[i];
                const double R = rw[i] * st[i] + dr * st[i] + dt * ct[i];
                return sdf(sp.cx + X, sp.cy + R, sp.cz);
            };
            const double fr = (at(e, 0) - at(-e, 0)) / (2 * e);
            const double ft = (at(0, e) - at(0, -e)) / (2 * e);
            if (!(fr > 0.0))
                return std::unexpected(std::format(
                    "the sdf does not increase outward at theta = {:.6f} rad",
                    th));
            const double l = std::hypot(fr, ft);
            nr = fr / l;
            nt = ft / l;
        }
        nh[i] = {nr, nt};
        const double disc = sp.rFar * sp.rFar - rw[i] * rw[i] * (1 - nr * nr);
        ln[i] = disc > 0 ? -rw[i] * nr + std::sqrt(disc) : 0.0;
        if (!(ln[i] > 0.0))
            return std::unexpected(std::format(
                "the wall-normal ray at theta = {:.6f} rad cannot reach the far "
                "circle",
                th));
    }

    // --- meridional nodes (X along x from the centre, R = rho >= 0), rows
    // 1..ny-1 on the real columns; e_r = (-cos, sin), e_theta = (sin, cos).
    std::vector<double> mx(std::size_t(g.nx) * ny), mr(std::size_t(g.nx) * ny);
    auto at2 = [&](int i, int j) { return std::size_t(j) * g.nx + i; };
    for (int j = 1; j < ny; ++j) {
        const double t = double(j - 1) / nf;
        const double d = ogridRadialDist(t, sp.beta);
        const double w = t * t * (3.0 - 2.0 * t);
        for (int i = 1; i <= nTheta + 1; ++i) {
            const double erx = -ct[i], err = st[i];
            const double nX = nh[i][0] * erx + nh[i][1] * st[i];
            const double nR = nh[i][0] * err + nh[i][1] * ct[i];
            const double bx = (1 - w) * ln[i] * nX + w * (sp.rFar - rw[i]) * erx;
            const double br = (1 - w) * ln[i] * nR + w * (sp.rFar - rw[i]) * err;
            mx[at2(i, j)] = rw[i] * erx + d * bx;
            mr[at2(i, j)] = rw[i] * err + d * br;
        }
    }
    // Axis columns: rho exactly 0 on every row (nR above is exactly 0 there
    // too -- st == 0, nh = (1, 0) -- this only states it).
    for (int j = 1; j < ny; ++j) {
        mr[at2(1, j)] = 0.0;
        mr[at2(nTheta + 1, j)] = 0.0;
    }
    // Ghost row 0: the point reflection of row 2 through the wall node.
    for (int i = 1; i <= nTheta + 1; ++i) {
        mx[at2(i, 0)] = 2 * mx[at2(i, 1)] - mx[at2(i, 2)];
        mr[at2(i, 0)] = 2 * mr[at2(i, 1)] - mr[at2(i, 2)];
    }

    // --- revolve. Azimuthal node planes and the exact-mirror tables.
    std::vector<double> cph(nz), sph(nz);
    if (rv.wedge) {
        const double c3 = std::cos(1.5 * rv.dphi), s3 = std::sin(1.5 * rv.dphi);
        const double c1 = std::cos(0.5 * rv.dphi), s1 = std::sin(0.5 * rv.dphi);
        cph = {c3, c1, c1};
        sph = {-s3, -s1, s1};
    } else {
        for (int k = 0; k < nz; ++k) {
            const int h = rv.nphi / 2;
            const double a = 2.0 * pi * (k % h) / rv.nphi;
            const double s = k < h ? 1.0 : -1.0; // phi + pi: (-cos, -sin)
            cph[k] = s * std::cos(a);
            sph[k] = s * std::sin(a);
        }
    }
    OGrid out{.mesh = CurvMesh::uniform(g)};
    out.revolved = true;
    out.rev = rv;
    out.nTheta = nTheta;
    out.mesh.wrap = {AxisWrap::Extend, AxisWrap::Extend,
                     rv.wedge ? AxisWrap::MirrorZ : AxisWrap::Cyclic};
    out.mesh.mirrorZ0 = sp.cz;
    out.mesh.axisFaces = {0, nTheta};
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 1; i <= nTheta + 1; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double R = mr[at2(i, j)];
                out.mesh.x[c] = sp.cx + mx[at2(i, j)];
                out.mesh.y[c] = sp.cy + R * cph[k];
                out.mesh.z[c] = sp.cz + R * sph[k];
            }
    // Ghost column 0: 2 node(1) - node(2), every row and plane.
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            const std::size_t a = g.idx(0, j, k), b = g.idx(1, j, k),
                              cc = g.idx(2, j, k);
            out.mesh.x[a] = 2 * out.mesh.x[b] - out.mesh.x[cc];
            out.mesh.y[a] = 2 * out.mesh.y[b] - out.mesh.y[cc];
            out.mesh.z[a] = 2 * out.mesh.z[b] - out.mesh.z[cc];
        }

    // --- diagnostics on the meridian (real columns, fluid rows).
    out.rWall.assign(std::size_t(g.nx), 0.0);
    out.wallNormal.assign(std::size_t(g.nx), {0.0, 0.0, 0.0});
    out.h1 = 1e300;
    out.h1Max = 0;
    out.growth = 1;
    out.rWallMin = 1e300;
    out.rWallMax = 0;
    for (int i = 1; i <= nTheta + 1; ++i) {
        out.rWall[i] = rw[i];
        out.rWallMin = std::min(out.rWallMin, rw[i]);
        out.rWallMax = std::max(out.rWallMax, rw[i]);
        out.wallNormal[i] = {nh[i][0] * -ct[i] + nh[i][1] * st[i],
                             nh[i][0] * st[i] + nh[i][1] * ct[i], 0.0};
        double prev = 0;
        for (int j = 1; j <= nf; ++j) {
            const double ex = mx[at2(i, j + 1)] - mx[at2(i, j)],
                         er = mr[at2(i, j + 1)] - mr[at2(i, j)];
            const double len = std::hypot(ex, er);
            if (!(len > 0.0))
                return std::unexpected(std::format(
                    "radial edge {} collapsed at theta column {}", j, i));
            if (j == 1) {
                out.h1 = std::min(out.h1, len);
                out.h1Max = std::max(out.h1Max, len);
            } else {
                out.growth = std::max(out.growth, len / prev);
            }
            prev = len;
        }
    }
    return out;
}

std::vector<std::array<double, 2>> ogridWallMidpoints(const OGrid& og)
{
    const int nx = og.mesh.g.nx;
    std::vector<std::array<double, 2>> mid(static_cast<std::size_t>(nx));
    for (int i = 0; i < nx; ++i) {
        const auto a = og.mesh.node(i, 1, 0);
        const auto b = og.mesh.node(i + 1, 1, 0);
        mid[std::size_t(i)] = {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])};
    }
    return mid;
}

std::vector<std::array<double, 2>> ogridWallSpans(const OGrid& og)
{
    const int nx = og.mesh.g.nx;
    auto centre = [&](int i, int j) {
        std::array<double, 2> c{0.0, 0.0};
        for (int dj = 0; dj < 2; ++dj)
            for (int di = 0; di < 2; ++di) {
                const auto p = og.mesh.node(i + di, j + dj, 0);
                c[0] += 0.25 * p[0];
                c[1] += 0.25 * p[1];
            }
        return c;
    };
    std::vector<std::array<double, 2>> span(static_cast<std::size_t>(nx));
    for (int i = 0; i < nx; ++i) {
        const auto c0 = centre(i, 0), c1 = centre(i, 1);
        span[std::size_t(i)] = {c1[0] - c0[0], c1[1] - c0[1]};
    }
    return span;
}

} // namespace dns
