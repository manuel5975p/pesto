// project() must return a discretely divergence-free field from noise.
#include "dns/operators.hpp"
#include "dns/projection.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <random>

using namespace dns;

int main()
{
    using std::numbers::pi;
    const auto g = Grid::make(24, 24, 24, 2 * pi, 2 * pi, 2 * pi);

    auto U = VectorField::zeros(g);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        U.u[c] = d(rng);
        U.v[c] = d(rng);
        U.w[c] = d(rng);
    }

    auto poisson = makeDefaultSolver(g);
    auto scratch = ProjectionScratch::make(g);
    project(g, *poisson, 0.01, U, scratch);

    Scalar div = zeros(g);
    divergence(g, U, div);
    double e = 0.0;
    for (double x : div) e = std::max(e, std::abs(x));

    std::printf("max |div| after projection: %.3e\n", e);
    return e < 1e-9 ? EXIT_SUCCESS : EXIT_FAILURE;
}
