#include "dns/fvnozzle.hpp"

#include <cassert>
#include <cmath>

namespace dns {

FvState fvRestState(const FaceMesh& fm, double p, double T, double gamma, double Rgas)
{
    assert(p > 0.0 && T > 0.0);
    FvState s = FvState::zeros(fm.nC);
    const double rho = p / (Rgas * T), E = p / (gamma - 1.0);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        s.var(0)[c] = rho;
        s.var(4)[c] = E;
    }
    return s;
}

double fvIdealMassFlow(double p0, double T0, double astar, double gamma, double Rgas)
{
    const double e = (gamma + 1.0) / (2.0 * (gamma - 1.0));
    return astar * p0 * std::sqrt(gamma / (Rgas * T0)) * std::pow(2.0 / (gamma + 1.0), e);
}

FvNozzleRow fvNozzleRow(std::span<const FvBc> bc, std::span<const double> mass, const std::array<std::vector<double>, 3>& mom,
                        const std::array<double, 3>& fw, const FvNozzleRun& run, double gamma, double Rgas)
{
    assert(run.astar > 0.0 && bc.size() == mass.size());
    const auto& a = run.axis;
    FvNozzleRow r;
    for (std::size_t p = 0; p < bc.size(); ++p) {
        const double along = mom[0][p] * a[0] + mom[1][p] * a[1] + mom[2][p] * a[2];
        if (bc[p].kind == FvBcKind::Total) {
            r.mdotIn -= mass[p];
            r.Fin += along;
        }
        if (bc[p].kind == FvBcKind::Outflow) {
            r.mdotOut += mass[p];
            r.Fexit += along;
        }
    }
    r.Fwall = fw[0] * a[0] + fw[1] * a[1] + fw[2] * a[2];
    r.Cd = r.mdotOut / fvIdealMassFlow(run.p0, run.t0, run.astar, gamma, Rgas);
    r.Cf = r.Fexit / (run.p0 * run.astar);
    return r;
}

} // namespace dns
