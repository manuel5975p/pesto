// Taylor-Green vortex driver: integrates to t=T, reports error vs analytic
// solution and kinetic-energy decay. Usage: tgv [n] [nu] [T] [--vtk out.vtk]
#include "dns/operators.hpp"
#include "dns/taylor_green.hpp"
#include "dns/timestepper.hpp"
#include "dns/vtk_writer.hpp"

#include <charconv>
#include <cstdio>
#include <numbers>
#include <string_view>

int main(int argc, char** argv)
{
    using std::numbers::pi;

    int n = 64;
    double nu = 0.05, T = 1.0;
    std::string_view vtkPath;

    auto parse = [](std::string_view s, auto& out) {
        std::from_chars(s.data(), s.data() + s.size(), out);
    };
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s = argv[a];
        if (s == "--vtk" && a + 1 < argc) { vtkPath = argv[++a]; }
        else if (a == 1) parse(s, n);
        else if (a == 2) parse(s, nu);
        else if (a == 3) parse(s, T);
    }

    const auto g = dns::Grid::make(n, n, 1, 2 * pi, 2 * pi, 2 * pi / n);
    auto U = dns::VectorField::zeros(g);
    dns::taylorGreen(g, nu, 0.0, U);

    dns::TimeStepper ts(g, {.nu = nu, .cfl = 0.5});
    const double ke0 = dns::kineticEnergy(g, U);

    double t = 0.0;
    long steps = 0;
    while (t < T)
    {
        const double dt = ts.suggestDt(U, T - t);
        ts.step(U, dt);
        t += dt;
        ++steps;
    }

    const double err = dns::taylorGreenError(g, nu, t, U);
    const double ke = dns::kineticEnergy(g, U);
    std::printf("n=%d nu=%g T=%g steps=%ld\n", n, nu, t, steps);
    std::printf("max error vs analytic: %.3e\n", err);
    std::printf("KE ratio: %.6f (analytic %.6f)\n", ke / ke0, std::exp(-4.0 * nu * t));

    if (!vtkPath.empty())
    {
        if (auto r = dns::writeVtk(g, U, std::string(vtkPath)); !r)
        {
            std::fprintf(stderr, "vtk: %s\n", r.error().c_str());
            return 1;
        }
    }
    return 0;
}
