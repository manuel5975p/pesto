// PointRemap: the mathematical object behind every curvilinear mesh. A remap
// is a smooth bijection x = phi(X) from the uniform reference box
// [0,lx] x [0,ly] x [0,lz] onto itself (identity plus a periodic-compatible
// distortion or a boundary-preserving stretch). It owns the analytic
// intricacies -- mapping, Jacobian, inverse Jacobian, metric tensor -- and is
// a trivially copyable value type with DNS_HD methods, so it can be passed by
// value as a TEMPLATE ARGUMENT to HIP kernels and used verbatim on the host.
//
// The finite-volume tables (CurvMetrics) are still DERIVED from node
// positions, never from the analytic Jacobian: the discrete face closure
// sum_faces S = 0 (freestream preservation) holds only for the bilinear-quad
// face areas of shared nodes. The remap is the source of truth for WHERE the
// nodes are and for any kernel that needs positions or derivatives at points
// that are not nodes (boundary conditions, derived fields, dt estimates).
//
// PointRemap concept (all methods const, DNS_HD):
//   void map(const double X[3], double x[3]) const;       // phi
//   void jacobian(const double X[3], double J[3][3]) const; // J[i][j] = dx_i/dX_j
#pragma once

#include "dns/hd.hpp"

#include <math.h>

namespace dns {

// x = X. The beta -> 0 / amp -> 0 limit of everything below.
struct IdentityRemap
{
    DNS_HD void map(const double X[3], double x[3]) const
    {
        x[0] = X[0];
        x[1] = X[1];
        x[2] = X[2];
    }
    DNS_HD void jacobian(const double[3], double J[3][3]) const
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) J[i][j] = i == j ? 1.0 : 0.0;
    }
};

// The fully periodic smooth distortion used by the curvilinear parity tests:
//   x += amp lx (sin(2 pi Y/ly) + sin(2 pi Z/lz))   (cyclically for y, z).
// planarZ freezes z (the nz == 1 exact-2D restriction). Arithmetic matches
// the historical CurvMesh::sinusoidal term for term.
struct SinusoidalRemap
{
    double lx, ly, lz;
    double amp;
    bool planarZ;

    DNS_HD void map(const double X[3], double x[3]) const
    {
        const double sx = sin(2.0 * M_PI * X[0] / lx);
        const double sy = sin(2.0 * M_PI * X[1] / ly);
        const double sz = planarZ ? 0.0 : sin(2.0 * M_PI * X[2] / lz);
        x[0] = X[0] + amp * lx * (sy + sz);
        x[1] = X[1] + amp * ly * (sz + sx);
        x[2] = planarZ ? X[2] : X[2] + amp * lz * (sx + sy);
    }
    DNS_HD void jacobian(const double X[3], double J[3][3]) const
    {
        const double cx = 2.0 * M_PI / lx * cos(2.0 * M_PI * X[0] / lx);
        const double cy = 2.0 * M_PI / ly * cos(2.0 * M_PI * X[1] / ly);
        const double cz = planarZ ? 0.0 : 2.0 * M_PI / lz * cos(2.0 * M_PI * X[2] / lz);
        J[0][0] = 1.0;             J[0][1] = amp * lx * cy;  J[0][2] = amp * lx * cz;
        J[1][0] = amp * ly * cx;   J[1][1] = 1.0;            J[1][2] = amp * ly * cz;
        if (planarZ) { J[2][0] = J[2][1] = 0.0; J[2][2] = 1.0; }
        else { J[2][0] = amp * lz * cx; J[2][1] = amp * lz * cy; J[2][2] = 1.0; }
    }
};

// Two-sided tanh (Vinokur-style) wall-normal clustering along y:
//   y(Y) = ly/2 (1 + tanh(beta (2Y/ly - 1)) / tanh(beta)),   x, z identity.
// Cells cluster toward BOTH y extremes with a bounded growth ratio; the
// first-cell compression is dy/dY|_wall = beta sech^2(beta) / tanh(beta).
// A power law (Y^2) is deliberately NOT used: it drives the first cell to
// zero thickness, which fails the positive-volume check or collapses dt.
// beta == 0 is the exact identity (bit-for-bit uniform nodes).
struct TanhStretchY
{
    double ly;
    double beta;

    DNS_HD double mapY(double Y) const
    {
        if (beta <= 0.0) return Y;
        const double s = 2.0 * Y / ly - 1.0;
        return 0.5 * ly * (1.0 + tanh(beta * s) / tanh(beta));
    }
    // dy/dY, the local compression factor (< 1 near the walls, > 1 mid-box).
    DNS_HD double dmapY(double Y) const
    {
        if (beta <= 0.0) return 1.0;
        const double t = tanh(beta * (2.0 * Y / ly - 1.0));
        return beta * (1.0 - t * t) / tanh(beta);
    }

    DNS_HD void map(const double X[3], double x[3]) const
    {
        x[0] = X[0];
        x[1] = mapY(X[1]);
        x[2] = X[2];
    }
    DNS_HD void jacobian(const double X[3], double J[3][3]) const
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) J[i][j] = i == j ? 1.0 : 0.0;
        J[1][1] = dmapY(X[1]);
    }
};

// Channel wall-normal map along y. The periodic array of ny cells holds two
// mirror GHOST rows (0 and ny-1) and ny-2 fluid rows spanning the physical
// channel [0, 2 delta]; the wall faces coincide with nodes 1 and ny-1. The
// fluid interior is a two-sided tanh (beta == 0: uniform), and beyond the
// walls the map is the exact REFLECTION about the wall, so the ghost cells
// mirror the first fluid cells geometrically and a value-mirror ghost places
// u = 0 (and T = Twall) exactly ON the wall face, second order.
//
// Periodicity fixes the box: the ghost rows each have the first-cell height
// h1, so the grid period MUST be ly = 2 delta + 2 h1 (firstCell() below).
// Like TanhStretchY, the map is only reflection-smooth one ghost row past
// each wall -- exactly the rows the mesh and the derived kernels see.
struct ChannelStretchY
{
    double ly;    // grid period; must equal 2*delta + 2*firstCell(...)
    int ny;       // total cells INCLUDING the two ghost rows
    double beta;  // tanh clustering strength; 0 = uniform fluid cells
    double delta; // channel half-height

    // First (== ghost) cell height from the tanh distribution alone; the
    // caller sizes the Grid with ly = 2*delta + 2*firstCell(...).
    DNS_HD static double firstCell(int nyTotal, double beta, double delta)
    {
        const int nf = nyTotal - 2; // fluid cells
        if (beta <= 0.0) return 2.0 * delta / nf;
        return delta * (1.0 - tanh(beta * (1.0 - 2.0 / nf)) / tanh(beta));
    }

    // Fluid map: reference [hy, (ny-1) hy] -> physical [0, 2 delta].
    DNS_HD double s(double Y) const
    {
        const double hy = ly / ny;
        const double t = (Y - hy) / ((ny - 2) * hy);
        if (beta <= 0.0) return 2.0 * delta * t;
        return delta * (1.0 + tanh(beta * (2.0 * t - 1.0)) / tanh(beta));
    }
    DNS_HD double ds(double Y) const
    {
        const double hy = ly / ny;
        const double span = (ny - 2) * hy;
        if (beta <= 0.0) return 2.0 * delta / span;
        const double t = (Y - hy) / span;
        const double th = tanh(beta * (2.0 * t - 1.0));
        return delta * beta * (1.0 - th * th) / tanh(beta) * 2.0 / span;
    }

    DNS_HD double mapY(double Y) const
    {
        const double hy = ly / ny, Y2 = (ny - 1) * hy;
        if (Y < hy) return -s(2.0 * hy - Y);
        if (Y > Y2) return 4.0 * delta - s(2.0 * Y2 - Y);
        return s(Y);
    }
    DNS_HD double dmapY(double Y) const
    {
        const double hy = ly / ny, Y2 = (ny - 1) * hy;
        if (Y < hy) return ds(2.0 * hy - Y);
        if (Y > Y2) return ds(2.0 * Y2 - Y);
        return ds(Y);
    }

    DNS_HD void map(const double X[3], double x[3]) const
    {
        x[0] = X[0];
        x[1] = mapY(X[1]);
        x[2] = X[2];
    }
    DNS_HD void jacobian(const double X[3], double J[3][3]) const
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) J[i][j] = i == j ? 1.0 : 0.0;
        J[1][1] = dmapY(X[1]);
    }
};

// Interior focal clustering along one axis: cells are FINE (spacing c r h) on
// a plateau [a, b], COARSE (spacing c h) outside, blended over a width w on
// each side by an integrated quintic smoothstep. The polynomial's compact
// support makes the slope exactly constant outside [a-w, b+w]: the far grid is
// exactly uniform, the periodic wrap is exactly C^2 (given the validator's
// identity margin at the domain faces), and the plateau spacing is exactly
// c r h — all pinnable. Derivative dx/dX in [c r, c], strictly positive, so
// the map cannot fold. Endpoints map to themselves; make() precomputes the
// normalization c = L / (L - (1-r) P), P = (b-a) + w, on the host, so the
// struct is pure baked coefficients (trivially copyable, kernel-by-value).
// Inactive (R <= 1 or empty interval) is the EXACT identity by branch, the
// TanhStretchY beta == 0 pattern.
struct Focal1D
{
    double L = 1.0;          // axis period
    double a = 0.0, b = 0.0; // fine plateau [a, b] (world coords)
    double w = 0.0;          // blend width each side
    double r = 1.0;          // fine/coarse slope ratio = 1/R
    double c = 1.0;          // normalization L / (L - (1-r) ((b-a)+w))
    bool active = false;

    static Focal1D identity(double L)
    {
        Focal1D f;
        f.L = L;
        return f;
    }
    // R = coarse/fine cell-size ratio. Invalid parameters (R <= 1, b <= a,
    // w <= 0) yield the inactive identity; the CLI validator rejects them
    // before this point with a message.
    static Focal1D make(double L, double a, double b, double w, double R)
    {
        Focal1D f = identity(L);
        if (!(R > 1.0) || !(b > a) || !(w > 0.0)) return f;
        f.a = a;
        f.b = b;
        f.w = w;
        f.r = 1.0 / R;
        f.c = L / (L - (1.0 - f.r) * ((b - a) + w));
        f.active = true;
        return f;
    }

    // Quintic smoothstep and its antiderivative on [0, 1].
    DNS_HD static double s5(double u)
    {
        return u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
    }
    DNS_HD static double S5(double u)
    {
        const double u2 = u * u;
        return u2 * u2 * (2.5 + u * (-3.0 + u));
    }

    // Blend weight (1 on the plateau, 0 outside the support) and its exact
    // antiderivative from 0; both piecewise over one period [0, L].
    DNS_HD double phi(double X) const
    {
        if (X <= a - w || X >= b + w) return 0.0;
        if (X < a) return s5((X - a + w) / w);
        if (X <= b) return 1.0;
        return 1.0 - s5((X - b) / w);
    }
    DNS_HD double PhiInt(double X) const
    {
        if (X <= a - w) return 0.0;
        if (X < a) return w * S5((X - a + w) / w);
        if (X <= b) return 0.5 * w + (X - a);
        if (X < b + w) {
            const double u = (X - b) / w;
            return 0.5 * w + (b - a) + w * (u - S5(u));
        }
        return (b - a) + w;
    }

    // Periodic extension: x(X + L) = x(X) + L exactly (endpoint-preserving
    // one-period map); callers may pass unwrapped coordinates.
    DNS_HD double map(double X) const
    {
        if (!active) return X;
        const double n = floor(X / L);
        const double Xl = X - n * L;
        return n * L + c * (Xl - (1.0 - r) * PhiInt(Xl));
    }
    DNS_HD double deriv(double X) const
    {
        if (!active) return 1.0;
        const double Xl = X - floor(X / L) * L;
        return c * (1.0 - (1.0 - r) * phi(Xl));
    }

    // Host-only (ghost-wall classification at make() time; nothing on device
    // inverts). Affine segments exact; blend segments by safeguarded Newton
    // (derivative >= c r > 0, bisection fallback).
    double inverse(double x) const
    {
        if (!active) return x;
        const double n = floor(x / L);
        const double xl = x - n * L;
        double X;
        if (xl <= map(a - w)) {
            X = xl / c;
        } else if (xl >= map(b + w)) {
            X = xl / c + (1.0 - r) * ((b - a) + w);
        } else if (xl >= map(a) && xl <= map(b)) {
            X = a + (xl - map(a)) / (c * r);
        } else {
            double lo = xl < map(a) ? a - w : b;
            double hi = xl < map(a) ? a : b + w;
            X = 0.5 * (lo + hi);
            for (int it = 0; it < 100; ++it) {
                const double f = map(X) - xl;
                if (f > 0.0) hi = X; else lo = X;
                const double step = f / deriv(X);
                double Xn = X - step;
                if (!(Xn > lo && Xn < hi)) Xn = 0.5 * (lo + hi);
                if (Xn == X || fabs(f) < 1e-15 * L) break;
                X = Xn;
            }
        }
        return n * L + X;
    }
};

// Separable per-axis remap: three independent Focal1D maps, diagonal
// Jacobian. ONE concrete type (no template combinatorics): NSCBC and the
// derived-field kernels each get exactly one extra instantiation, and the
// coefficients travel as data. All axes inactive is the exact identity, so
// the degenerate limit stays bit-pinnable through CurvMesh::fromRemap. An
// inactive fz keeps nz == 1 planar (z untouched).
struct AxisRemap3
{
    Focal1D fx, fy, fz;

    DNS_HD void map(const double X[3], double x[3]) const
    {
        x[0] = fx.map(X[0]);
        x[1] = fy.map(X[1]);
        x[2] = fz.map(X[2]);
    }
    DNS_HD void jacobian(const double X[3], double J[3][3]) const
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) J[i][j] = 0.0;
        J[0][0] = fx.deriv(X[0]);
        J[1][1] = fy.deriv(X[1]);
        J[2][2] = fz.deriv(X[2]);
    }
};

// --- derived quantities (valid for any PointRemap) ---------------------------

template <class Remap>
DNS_HD double remapDet(const Remap& r, const double X[3])
{
    double J[3][3];
    r.jacobian(X, J);
    return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
           J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
           J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
}

// Ji[i][j] = dX_i / dx_j. Returns false when the mapping folds (det <= 0).
template <class Remap>
DNS_HD bool remapInverseJacobian(const Remap& r, const double X[3],
                                 double Ji[3][3])
{
    double J[3][3];
    r.jacobian(X, J);
    Ji[0][0] = J[1][1] * J[2][2] - J[1][2] * J[2][1];
    Ji[0][1] = J[0][2] * J[2][1] - J[0][1] * J[2][2];
    Ji[0][2] = J[0][1] * J[1][2] - J[0][2] * J[1][1];
    Ji[1][0] = J[1][2] * J[2][0] - J[1][0] * J[2][2];
    Ji[1][1] = J[0][0] * J[2][2] - J[0][2] * J[2][0];
    Ji[1][2] = J[0][2] * J[1][0] - J[0][0] * J[1][2];
    Ji[2][0] = J[1][0] * J[2][1] - J[1][1] * J[2][0];
    Ji[2][1] = J[0][1] * J[2][0] - J[0][0] * J[2][1];
    Ji[2][2] = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    const double det = J[0][0] * Ji[0][0] + J[0][1] * Ji[1][0] + J[0][2] * Ji[2][0];
    if (!(det > 0.0)) return false;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) Ji[i][j] /= det;
    return true;
}

// Covariant metric tensor G = J^T J (G[i][j] = dx_k/dX_i dx_k/dX_j).
template <class Remap>
DNS_HD void remapMetric(const Remap& r, const double X[3], double G[3][3])
{
    double J[3][3];
    r.jacobian(X, J);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            G[i][j] = J[0][i] * J[0][j] + J[1][i] * J[1][j] + J[2][i] * J[2][j];
}

} // namespace dns
