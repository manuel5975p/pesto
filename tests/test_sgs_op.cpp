// sigmaDsigma (dns/sgs.hpp), the sigma-model differential operator, checked on
// the CPU against the states the model is DEFINED by (Nicoud et al. 2011):
// it must vanish identically for every two-component/2D flow, pure shear,
// solid rotation and axisymmetric strain -- these are the laminar/transitional
// states where Smagorinsky's |S| would inject eddy viscosity -- and equal the
// closed-form sigma3(sigma1-sigma2)(sigma2-sigma3)/sigma1^2 on tensors whose
// singular values are known. Rotation invariance and degree-1 homogeneity pin
// the invariant algebra; clamping keeps roundoff from ever producing a
// negative viscosity or NaN.
//
// The operator was re-conditioned on 2026-07-26 to evaluate in its template
// parameter (the nut kernels are float; FP64 runs at 1/32 rate on consumer
// GPUs), so this file also pins it against the previous all-double
// implementation, kept verbatim below as sigmaDsigmaRef, and measures the
// noise levels that justify detail::SigmaTune's per-precision floors.

#include "dns/sgs.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>

namespace {

using dnstest::check;

// The implementation replaced on 2026-07-26 (all-double invariants, acos + 3
// cosines, single 1e-6 sigma3 floor), verbatim, as the accuracy reference.
// Called on a float tensor it still evaluates in double, so it is also the
// accurate answer for the float instantiation's inputs.
template <class T>
double sigmaDsigmaRef(const T gd[3][3])
{
    double G[3][3]; // G = g^T g, symmetric PSD
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            double s = 0;
            for (int c = 0; c < 3; ++c)
                s += double(gd[c][a]) * double(gd[c][b]);
            G[a][b] = s;
        }
    const double i1 = G[0][0] + G[1][1] + G[2][2];
    if (!(i1 > 0)) return 0.0; // uniform flow (or non-finite input)
    double trG2 = 0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) trG2 += G[a][b] * G[b][a];
    const double i2 = 0.5 * (i1 * i1 - trG2);
    const double i3 = G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1])
                    - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
                    + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);
    // Nicoud et al. (2011), eqs. (18)-(20).
    const double a1 = fmax(i1 * i1 / 9.0 - i2 / 3.0, 0.0);
    const double a2 = i1 * i1 * i1 / 27.0 - i1 * i2 / 6.0 + i3 / 2.0;
    const double sq = sqrt(a1);
    const double cosArg =
        sq > 0 ? fmin(fmax(a2 / (a1 * sq), -1.0), 1.0) : 1.0;
    const double a3 = acos(cosArg) / 3.0;
    constexpr double pi3 = 1.0471975511965977; // pi/3
    const double s1 = sqrt(fmax(i1 / 3.0 + 2.0 * sq * cos(a3), 0.0));
    const double s2 = sqrt(fmax(i1 / 3.0 - 2.0 * sq * cos(pi3 + a3), 0.0));
    const double s3 = sqrt(fmax(i1 / 3.0 - 2.0 * sq * cos(pi3 - a3), 0.0));
    if (!(s1 > 0)) return 0.0;
    if (s3 <= 1e-6 * s1) return 0.0;
    return s3 * fmax(s1 - s2, 0.0) * fmax(s2 - s3, 0.0) / (s1 * s1);
}

double lcg(unsigned& s) // deterministic uniform in [-1, 1)
{
    s = 1664525u * s + 1013904223u;
    return double(s) / 2147483648.0 - 1.0;
}

// Rotation from three Euler angles, so the "defining zero" states can be taken
// off the coordinate axes -- where the operator's exactness stops being free.
void rotM(double a, double b, double c, double R[3][3])
{
    const double ca = std::cos(a), sa = std::sin(a);
    const double cb = std::cos(b), sb = std::sin(b);
    const double cc = std::cos(c), sc = std::sin(c);
    const double Rz[3][3] = {{ca, -sa, 0}, {sa, ca, 0}, {0, 0, 1}};
    const double Ry[3][3] = {{cb, 0, sb}, {0, 1, 0}, {-sb, 0, cb}};
    const double Rx[3][3] = {{1, 0, 0}, {0, cc, -sc}, {0, sc, cc}};
    double M[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += Rz[i][k] * Ry[k][j];
            M[i][j] = s;
        }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += M[i][k] * Rx[k][j];
            R[i][j] = s;
        }
}

// g = U diag(sv) V^T, i.e. a tensor with PRESCRIBED singular values.
void svdBuild(const double U[3][3], const double sv[3], const double V[3][3],
              double g[3][3])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += U[i][k] * sv[k] * V[j][k];
            g[i][j] = s;
        }
}

double amaxOf(const double g[3][3])
{
    double m = 0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) m = std::fmax(m, std::fabs(g[a][b]));
    return m;
}

void toFloat(const double g[3][3], float gf[3][3])
{
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) gf[a][b] = float(g[a][b]);
}

} // namespace

int main()
{
    using dns::sigmaDsigma;
    const double eps = 1e-9;

    { // uniform flow
        const double g[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        check(sigmaDsigma(g) == 0.0, "zero gradient -> exactly 0");
    }
    { // pure shear: singular values (gam, 0, 0)
        const double g[3][3] = {{0, 1.7, 0}, {0, 0, 0}, {0, 0, 0}};
        check(std::fabs(sigmaDsigma(g)) <= eps, "pure shear -> 0");
    }
    { // solid rotation: singular values (om, om, 0)
        const double om = 2.3;
        const double g[3][3] = {{0, -om, 0}, {om, 0, 0}, {0, 0, 0}};
        check(std::fabs(sigmaDsigma(g)) <= eps, "solid rotation -> 0");
    }
    { // ANY 2D flow (sigma3 == 0): random 2x2 blocks
        unsigned s = 7;
        bool ok = true;
        for (int t = 0; t < 100; ++t) {
            const double g[3][3] = {{lcg(s), lcg(s), 0},
                                    {lcg(s), lcg(s), 0},
                                    {0, 0, 0}};
            ok = ok && std::fabs(sigmaDsigma(g)) <= eps;
        }
        check(ok, "random 2D gradients -> 0 (100 draws)");
    }
    { // axisymmetric strain diag(a, a, -2a): two equal singular values
        const double a = 0.9;
        const double g[3][3] = {{a, 0, 0}, {0, a, 0}, {0, 0, -2 * a}};
        check(std::fabs(sigmaDsigma(g)) <= eps, "axisymmetric strain -> 0");
    }
    { // diag(3, 2, 1): D = 1*(3-2)*(2-1)/9 = 1/9 exactly
        const double g[3][3] = {{3, 0, 0}, {0, 2, 0}, {0, 0, 1}};
        check(std::fabs(sigmaDsigma(g) - 1.0 / 9.0) <= eps,
              "diag(3,2,1) -> 1/9 (closed form)");
    }
    { // left rotation leaves singular values (hence D) unchanged
        const double c = std::cos(0.6), sn = std::sin(0.6);
        const double g[3][3] = {{3 * c, -2 * sn, 0},
                                {3 * sn, 2 * c, 0},
                                {0, 0, 1}}; // Rz(0.6) * diag(3,2,1)
        check(std::fabs(sigmaDsigma(g) - 1.0 / 9.0) <= eps,
              "rotated tensor -> same D (invariance)");
    }
    { // homogeneous of degree 1 and never negative on generic tensors
        unsigned s = 42;
        bool hom = true, pos = true;
        for (int t = 0; t < 100; ++t) {
            double g[3][3], g2[3][3];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    g[a][b] = lcg(s);
                    g2[a][b] = 5.0 * g[a][b];
                }
            const double d = sigmaDsigma(g);
            pos = pos && d >= 0.0 && std::isfinite(d);
            hom = hom && std::fabs(sigmaDsigma(g2) - 5.0 * d)
                             <= 1e-7 * (1.0 + 5.0 * d);
        }
        check(pos, "generic tensors: D >= 0, finite (100 draws)");
        check(hom, "D(s*g) == s*D(g) (degree-1 homogeneity)");
    }
    { // float input (the kernels' Real) still returns the clamped zeros
        const float g[3][3] = {{0, 1.7f, 0}, {0, 0, 0}, {0, 0, 0}};
        check(std::fabs(sigmaDsigma(g)) <= 1e-6, "float pure shear -> 0");
    }

    // ---- the defining zeros, in FLOAT, exactly ------------------------------
    // A rank-deficient g whose missing direction is a coordinate axis has an
    // exactly-zero det(g), hence an exactly-zero sigma3, in either precision.
    { // 2D flow, pure shear, solid rotation: EXACT zeros in float
        unsigned s = 7;
        bool ok2d = true, okSh = true, okRot = true;
        for (int t = 0; t < 1000; ++t) {
            const double sc = std::pow(10.0, 6.0 * lcg(s));
            const float g2d[3][3] = {{float(lcg(s) * sc), float(lcg(s) * sc), 0},
                                     {float(lcg(s) * sc), float(lcg(s) * sc), 0},
                                     {0, 0, 0}};
            const float gam = float(lcg(s) * sc);
            const float gsh[3][3] = {{0, gam, 0}, {0, 0, 0}, {0, 0, 0}};
            const float grt[3][3] = {{0, -gam, 0}, {gam, 0, 0}, {0, 0, 0}};
            ok2d = ok2d && sigmaDsigma(g2d) == 0.0f;
            okSh = okSh && sigmaDsigma(gsh) == 0.0f;
            okRot = okRot && sigmaDsigma(grt) == 0.0f;
        }
        check(ok2d, "float 2D gradients -> EXACTLY 0 (1000 draws, 1e+-6)");
        check(okSh, "float pure shear -> EXACTLY 0 (1000 draws)");
        check(okRot, "float solid rotation -> EXACTLY 0 (1000 draws)");
    }
    { // and the same states in double
        unsigned s = 11;
        bool ok = true;
        for (int t = 0; t < 1000; ++t) {
            const double sc = std::pow(10.0, 12.0 * lcg(s));
            const double g[3][3] = {{lcg(s) * sc, lcg(s) * sc, 0},
                                    {lcg(s) * sc, lcg(s) * sc, 0},
                                    {0, 0, 0}};
            ok = ok && sigmaDsigma(g) == 0.0;
        }
        check(ok, "double 2D gradients -> EXACTLY 0 (1000 draws, 1e+-12)");
    }
    { // ROTATED rank-2 states: det(g) is no longer exactly 0, so this is the
      // noise the sigma3 floor has to cover. Measured, then asserted.
        unsigned s = 2027;
        double mxRd = 0, mxRf = 0, mxDd = 0, mxDf = 0;
        for (int t = 0; t < 4000; ++t) {
            double U[3][3], V[3][3], g[3][3];
            float gf[3][3];
            rotM(3.0 * lcg(s), 3.0 * lcg(s), 3.0 * lcg(s), U);
            rotM(3.0 * lcg(s), 3.0 * lcg(s), 3.0 * lcg(s), V);
            const double sv[3] = {std::pow(10.0, lcg(s)),
                                  std::pow(10.0, 3.0 * lcg(s) - 3.0), 0.0};
            svdBuild(U, sv, V, g);
            toFloat(g, gf);
            const double sc = amaxOf(g);
            mxDd = std::fmax(mxDd, sigmaDsigma(g) / sc);
            mxDf = std::fmax(mxDf, double(sigmaDsigma(gf)) / sc);
            mxRd = std::fmax(mxRd, sigmaDsigmaRef(g) / sc);
            mxRf = std::fmax(mxRf, sigmaDsigmaRef(gf) / sc);
        }
        std::printf("  rotated rank-2 (4000 draws, sigma2/sigma1 down to 1e-4):"
                    " max D/|g| new double %.3e float %.3e | ref %.3e %.3e\n",
                    mxDd, mxDf, mxRd, mxRf);
        check(mxDd <= 1e-12, "rotated rank-2 -> D <= 1e-12 |g| (double)");
        check(mxDf <= 1e-6, "rotated rank-2 -> D <= 1e-6 |g| (float)");
    }
    { // axisymmetric strain off the axes: sigma2 == sigma3 is a DOUBLE root of
      // the characteristic cubic, so (sigma2-sigma3) is intrinsically
      // sqrt(eps)-noisy once the arithmetic is FP32 -- that is the price of the
      // float instantiation, and it is bounded here.
        unsigned s = 909;
        double mxd = 0, mxf = 0;
        for (int t = 0; t < 2000; ++t) {
            double U[3][3], V[3][3], g[3][3];
            float gf[3][3];
            rotM(3.0 * lcg(s), 3.0 * lcg(s), 3.0 * lcg(s), U);
            rotM(3.0 * lcg(s), 3.0 * lcg(s), 3.0 * lcg(s), V);
            const double a = std::pow(10.0, 3.0 * lcg(s));
            const double sv[3] = {2 * a, a, a};
            svdBuild(U, sv, V, g);
            toFloat(g, gf);
            const double sc = amaxOf(g);
            mxd = std::fmax(mxd, sigmaDsigma(g) / sc);
            mxf = std::fmax(mxf, double(sigmaDsigma(gf)) / sc);
        }
        std::printf("  axisymmetric strain, rotated (2000 draws):"
                    " max D/|g| double %.3e float %.3e\n", mxd, mxf);
        check(mxd <= 1e-7, "rotated axisymmetric strain -> D <= 1e-7 |g| (double)");
        check(mxf <= 1e-3, "rotated axisymmetric strain -> D <= 1e-3 |g| (float)");
    }

    // ---- accuracy against the old all-double formula ------------------------
    { // generic tensors over 24 decades; the reference always runs in double,
      // so this isolates the new arithmetic's error in each instantiation.
        unsigned s = 31337;
        double mxAbsD = 0, mxAbsF = 0, mxRelD = 0, mxRelF = 0;
        long nAct = 0;
        for (int t = 0; t < 20000; ++t) {
            const double sc = std::pow(10.0, 12.0 * lcg(s));
            double g[3][3];
            float gf[3][3];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) g[a][b] = lcg(s) * sc;
            toFloat(g, gf);
            const double am = amaxOf(g);
            const double rd = sigmaDsigmaRef(g), nd = sigmaDsigma(g);
            mxAbsD = std::fmax(mxAbsD, std::fabs(nd - rd) / am);
            if (rd >= 1e-2 * am) {
                ++nAct;
                mxRelD = std::fmax(mxRelD, std::fabs(nd - rd) / rd);
            }
            if (sc > 1e-15 && sc < 1e15) { // float's own range, not the operator's
                const double rf = sigmaDsigmaRef(gf);
                const double nf = double(sigmaDsigma(gf));
                mxAbsF = std::fmax(mxAbsF, std::fabs(nf - rf) / am);
                if (rf >= 1e-2 * am)
                    mxRelF = std::fmax(mxRelF, std::fabs(nf - rf) / rf);
            }
        }
        std::printf("  vs old double formula (20000 draws, |g| over 1e+-12):\n"
                    "    max |dD|/|g|      double %.3e   float %.3e\n"
                    "    max rel err       double %.3e   float %.3e"
                    "   (%ld draws with D >= 1e-2|g|)\n",
                    mxAbsD, mxAbsF, mxRelD, mxRelF, nAct);
        check(mxAbsD <= 1e-9, "double instantiation: |dD| <= 1e-9 |g| vs old");
        check(mxRelD <= 1e-6, "double instantiation: rel err <= 1e-6 where D >= 1e-2|g|");
        check(mxAbsF <= 1e-3, "float instantiation: |dD| <= 1e-3 |g| vs old");
        check(mxRelF <= 1e-3, "float instantiation: rel err <= 1e-3 where D >= 1e-2|g|");
    }

    // ---- scale invariance ---------------------------------------------------
    { // a power-of-two rescale is EXACT in both instantiations: the operator
      // divides by amax up front, and that division scales exactly.
        unsigned s = 5150;
        bool okD = true, okF = true;
        for (int t = 0; t < 5000; ++t) {
            const double al = std::ldexp(1.0, int(20.0 * lcg(s)));
            double g[3][3], g2[3][3];
            float gf[3][3], gf2[3][3];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    g[a][b] = lcg(s);
                    g2[a][b] = g[a][b] * al;
                }
            toFloat(g, gf);
            toFloat(g2, gf2);
            okD = okD && sigmaDsigma(g2) == al * sigmaDsigma(g);
            okF = okF && sigmaDsigma(gf2) == float(al) * sigmaDsigma(gf);
        }
        check(okD, "D(2^k g) == 2^k D(g) EXACTLY (double, 5000 draws)");
        check(okF, "D(2^k g) == 2^k D(g) EXACTLY (float, 5000 draws)");
    }
    { // arbitrary rescale: exact up to the rounding of a*g itself
        unsigned s = 61616;
        double mxD = 0, mxF = 0;
        for (int t = 0; t < 5000; ++t) {
            const double al = std::pow(10.0, 6.0 * lcg(s));
            double g[3][3], g2[3][3];
            float gf[3][3], gf2[3][3];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    g[a][b] = lcg(s);
                    g2[a][b] = g[a][b] * al;
                }
            toFloat(g, gf);
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) gf2[a][b] = float(gf[a][b] * al);
            const double am = amaxOf(g);
            mxD = std::fmax(mxD, std::fabs(sigmaDsigma(g2) - al * sigmaDsigma(g))
                                     / (al * am));
            mxF = std::fmax(mxF, std::fabs(double(sigmaDsigma(gf2))
                                           - al * double(sigmaDsigma(gf)))
                                     / (al * am));
        }
        std::printf("  scale invariance, arbitrary factor over 1e+-6:"
                    " max |dD|/(a|g|) double %.3e float %.3e\n", mxD, mxF);
        check(mxD <= 1e-12, "D(a g) == a D(g) to 1e-12 |g| (double)");
        check(mxF <= 1e-3, "D(a g) == a D(g) to 1e-3 |g| (float)");
    }
    { // and the operator must not over/underflow anywhere in float's range
        bool ok = true;
        for (int e = -30; e <= 30; ++e) {
            const double sc = std::pow(10.0, e);
            const double g[3][3] = {{3 * sc, 0, 0}, {0, 2 * sc, 0}, {0, 0, sc}};
            float gf[3][3];
            toFloat(g, gf);
            const double d = double(sigmaDsigma(gf)); // exact answer: sc/9
            ok = ok && std::fabs(d - sc / 9.0) <= 1e-5 * sc / 9.0;
        }
        check(ok, "float diag(3,2,1)*1e^k -> 1e^k/9 for k in [-30, 30]");
    }

    return dnstest::finish();
}
