// sigmaDsigma (dns/sgs.hpp), the sigma-model differential operator, checked on
// the CPU against the states the model is DEFINED by (Nicoud et al. 2011):
// it must vanish identically for every two-component/2D flow, pure shear,
// solid rotation and axisymmetric strain -- these are the laminar/transitional
// states where Smagorinsky's |S| would inject eddy viscosity -- and equal the
// closed-form sigma3(sigma1-sigma2)(sigma2-sigma3)/sigma1^2 on tensors whose
// singular values are known. Rotation invariance and degree-1 homogeneity pin
// the invariant algebra; clamping keeps roundoff from ever producing a
// negative viscosity or NaN.

#include "dns/sgs.hpp"

#include <cmath>
#include <cstdio>

namespace {

bool pass = true;

void check(bool ok, const char* what)
{
    std::printf("%-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) pass = false;
}

double lcg(unsigned& s) // deterministic uniform in [-1, 1)
{
    s = 1664525u * s + 1013904223u;
    return double(s) / 2147483648.0 - 1.0;
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

    return pass ? 0 : 1;
}
