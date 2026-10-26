#include "dns/rans_label.hpp"

#include <cassert>
#include <format>

namespace dns {

namespace {

// The mode tail of a label, spelled exactly as the channel always printed it.
std::string modeText(RansMode mode, double dualDt)
{
    switch (mode) {
    case RansMode::Urans: return "URANS (time-accurate)";
    case RansMode::UransDual:
        return std::format("URANS (time-accurate, dual time BDF2, dt={:g} D/U)",
                           dualDt);
    case RansMode::SteadyExplicit:
        return "steady (local time stepping: pseudo-time, not time-accurate)";
    case RansMode::SteadyImplicitLine:
        return "steady (implicit Newton pseudo-time: not time-accurate)";
    case RansMode::SteadyImplicitNk:
        return "steady (implicit Newton-Krylov pseudo-time: not "
               "time-accurate)";
    }
    return {};
}

bool steadyMode(RansMode m)
{
    return m != RansMode::Urans && m != RansMode::UransDual;
}

} // namespace

std::string ransModelLabel(RansModel m)
{
    assert(m != RansModel::None);
    return "RANS " + std::string(ransName(m));
}

std::string ransRunLabel(RansModel m, RansMode mode, double dualDt,
                         std::string_view geometry)
{
    assert((mode == RansMode::UransDual) == (dualDt > 0));
    if (m == RansModel::None && !steadyMode(mode)) return {};
    const std::string head =
        m == RansModel::None ? std::string("DNS") : ransModelLabel(m);
    return head + " -- " + modeText(mode, dualDt) + std::string(geometry);
}

std::string ransWedgeSuffix(double deg, bool ensemble)
{
    assert(deg > 0);
    return std::format(" [axisymmetric wedge {:g} deg: {}no swirl, no 3D wake "
                       "modes]",
                       deg, ensemble ? "ensemble mean, " : "");
}

std::span<const RansModel> ransBuilt(Subcommand s)
{
    static constexpr RansModel all[] = {RansModel::None, RansModel::SaNeg,
                                        RansModel::Sst2003m, RansModel::SstVm};
    if (s == Subcommand::Channel || s == Subcommand::Flow) return all;
    return {};
}

} // namespace dns
