// The ONE producer of the RANS run labels (docs/rans.md, "Labels"): the
// --dry-run `label` line, the progress lines, RunMeta and the surface CSV
// header all read these, so a RANS run cannot be described two ways.
#pragma once

#include "dns/config.hpp"
#include "dns/rans.hpp"

#include <format>
#include <span>
#include <string>
#include <string_view>

namespace dns {

// How a run advances its RANS (or laminar steady) state.
enum class RansMode
{
    Urans,              // explicit SSP-RK3 on the global dt
    UransDual,          // dual-time BDF2 (--urans-dt)
    SteadyExplicit,     // local-time-stepping pseudo-time
    SteadyImplicitLine, // the channel's implicit Newton line march
    SteadyImplicitNk,   // the flow Newton-Krylov pseudo-time march
};

// "RANS <name>": the closure alone, as every progress line prints it.
// Pre: m != RansModel::None.
std::string ransModelLabel(RansModel m);

// The run label: "RANS <name> -- <mode><geometry>". RANS off: "" for the
// time-accurate modes (callers keep their DNS/LES text) and "DNS -- <mode>"
// for a laminar steady solve. dualDt: UransDual's physical step in D/U.
// geometry: a suffix appended verbatim (ransWedgeSuffix, kImmersedSuffix).
// Pre: dualDt > 0 iff mode == UransDual.
std::string ransRunLabel(RansModel m, RansMode mode, double dualDt = 0,
                         std::string_view geometry = {});

// The axisymmetric wedge's geometry suffix: " [axisymmetric wedge DEG deg:
// ensemble mean, no swirl, no 3D wake modes]" (a DNS time march says
// "no swirl, no 3D wake modes" only: it is not an ensemble mean). Pre: deg > 0.
std::string ransWedgeSuffix(double deg, bool ensemble = true);
// The immersed path's geometry suffix (masterplan_flowrans decision 6): the
// wall sits at first-order position on a staircase of cells.
inline constexpr std::string_view kImmersedSuffix =
    " [immersed ghost wall: first-order wall position, NOT body-fitted]";
// The frozen rotor's steady march (--rotor-frame mrf --steady), ahead of
// kImmersedSuffix: its fixed point is the frozen rotor's, not a time mean.
inline constexpr std::string_view kMrfSteadySuffix =
    " [MRF frozen rotor: an APPROXIMATION, blade fixed at azimuth 0]";
// --thin-seal (classifyCompGhost thinSeal): sections under the staircase's
// resolution are walls of zero thickness, not the body's own thickness.
inline constexpr std::string_view kThinSealSuffix =
    " [thin sections sealed: sub-cell sheets imaged as zero-thickness walls]";
// --precond K, appended last: " [low-Mach preconditioned pseudo-time, K=k]".
// Pre: K > 0.
inline std::string precondSuffix(double K)
{
    return std::format(" [low-Mach preconditioned pseudo-time, K={:g}]", K);
}
// What a time-accurate run on the wedge appends to its label.
inline constexpr std::string_view kAxisymUnsteady =
    "; axisymmetric unsteadiness only -- NOT the sphere's physical shedding";

// The closures `dns <sub> --rans` accepts: only what is BUILT is registered,
// so --help and the CLI's "one of:" list stay truthful. Empty = no --rans.
std::span<const RansModel> ransBuilt(Subcommand s);

} // namespace dns
