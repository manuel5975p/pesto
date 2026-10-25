// CPU compressible solver is axis-blind: the equations are invariant under
// rotations and reflections, so step(P s) == P step(s) for the cyclic
// rotation x -> y -> z (momenta permuted) and the mirror x -> -x (mx
// negated). An anisotropic box (lx, ly, lz all different) makes an operator
// that confused two axes' spacings fail here. Plus exact 3D conservation.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "curv_test_util.hpp"
#include "permute_test_util.hpp"
#include "check.hpp"

#include <array>
#include <cmath>
#include <cstdio>

using namespace dns;
using dnstest::check;

namespace {

constexpr double kGamma = 1.4, kMu = 0.01;
constexpr int kSteps = 10;

void advance(const Comp& c, CompState& s, double dt)
{
    for (int it = 0; it < kSteps; ++it) c.step(s, dt);
}

// step(P s) vs P step(s), relative to each field's max.
void permuteCase(const char* tag, const Grid& g, const CompState& s0,
                 const Grid& gp, CompState (*P)(const Grid&, const CompState&))
{
    const Comp c = Comp::make(g, kGamma, kMu);
    const Comp cp = Comp::make(gp, kGamma, kMu);
    const double dt = 0.5 * *c.stableDt(s0);
    auto a = s0;
    advance(c, a, dt);
    auto b = P(g, s0);
    advance(cp, b, dt);
    const double e = permtest::relState(b, P(g, a));
    std::printf("%-28s dt %.3e  max rel |step(P s) - P step(s)| %.3e\n", tag,
                dt, e);
    check(e < 1e-13, std::string(tag) + ": step(P s) == P step(s) to 1e-13");
}

void conservation3d(const Grid& g, const CompState& s0)
{
    const Comp c = Comp::make(g, kGamma, kMu);
    auto s = s0;
    auto sums = [&](const CompState& q) {
        std::array<double, 5> t{0, 0, 0, 0, 0};
        for (std::size_t k = 0; k < g.size(); ++k) {
            t[0] += q.rho[k];
            t[1] += q.mx[k];
            t[2] += q.my[k];
            t[3] += q.mz[k];
            t[4] += q.E[k];
        }
        return t;
    };
    const auto t0 = sums(s);
    advance(c, s, 0.5 * *c.stableDt(s));
    const auto t1 = sums(s);
    double worst = 0.0;
    for (int q = 0; q < 5; ++q) {
        const double scale = std::max(std::abs(t0[q]), double(g.size()));
        worst = std::max(worst, std::abs(t1[q] - t0[q]) / scale);
    }
    std::printf("3D conservation drift over %d steps: rho %.2e mx %.2e "
                "my %.2e mz %.2e E %.2e (worst rel %.2e)\n",
                kSteps, t1[0] - t0[0], t1[1] - t0[1], t1[2] - t0[2],
                t1[3] - t0[3], t1[4] - t0[4], worst);
    check(worst < 1e-11, "3D flux form conserves rho, m, E to 1e-11");
}

} // namespace

int main()
{
    const auto g = Grid::make(24, 24, 24, 2.0, 2.5, 3.0);
    const auto s0 = curvtest::smoothState(CurvMesh::uniform(g), kGamma);
    permuteCase("cyclic x->y->z", g, s0, permtest::cyclicGrid(g),
                permtest::cyclic);
    permuteCase("mirror x->-x", g, s0, g, permtest::mirrorX);
    conservation3d(g, s0);
    return dnstest::finish();
}
