// The FV flame routes' case assembly (dns flame --solver fv | --polymesh, compressible
// and low-Mach): a validated Case onto FaceMesh BC specs, gas and chemistry constants,
// initial states and inflow rows. The FaceMesh twin of dns/flame1d and dns/lowmach's
// ICs; host only, FP64, every value the structured route's expression (gpu_fv_flame,
// gpu_fv_lowmach, gpu_fv_turb pin the pairs).
#pragma once

#include "dns/config.hpp"
#include "dns/flame1d.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvcomp_case.hpp"
#include "dns/fvcomp_config.hpp"
#include "dns/fvflame.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvlowmach.hpp"
#include "dns/fvnozzle.hpp"
#include "dns/fvref.hpp"
#include "dns/fvturb.hpp"
#include "dns/tfles.hpp"
#include "dns/transport.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace dns {

// --tfles on a FaceMesh: cells per delta_th (the box's --cells-per-delta, a polyMesh's
// off the probe's slab), F cells >= 4 checked, the sensor radius from them on a
// polyMesh. Error: the message (exit 3). Pre: c.flameTfles > 0.
std::expected<TflesSetup, std::string> fvFlameTfles(const Case& c, double sigma, double slab);

// --turb on the FV flame routes: the structured box (flameTurbSpec) at the centroids,
// its laminar-weight window below the flame. Pre: c.flameTurb > 0.
FvTurb fvFlameTurb(const Case& c);

// --- compressible -----------------------------------------------------------------

// The flags' compressible FV flame box: the uniform grid's patches (x open, y cyclic, z
// cyclic | empty at nz 1) and their rows -- premixed: fresh Far, burnt Outflow; the jet:
// Inflow + Outflow; the candle under gravity: a Far top on the ambient column (the jet's
// fixed-p Outflow rang as an organ pipe). Pre: c validated, not --polymesh.
struct FvFlameBox
{
    StructuredSpec ss;
    FvBcSpec spec;
};
FvFlameBox fvFlameBox(const Case& c);

// The premixed compressible FV flame (C0, C2, C2b): gas (mu = Pr at Tu), chemistry, the
// fresh / burnt freestream along spec.dir and the probe's spec. C2: every Far / Inflow
// patch of spec carries the fresh premix, every Outflow the burnt one (validate refused a
// y in the file). Pre: c validated for the route, spec the route's.
struct FvFlameSetup
{
    FlameBurnt burnt;
    Transport gas;
    FvChem chem;
    FvFreestream fs;
    FvFlameSpec fl;
};
FvFlameSetup fvFlameSetup(const Case& c, FvBcSpec& spec);

// The premixed flame's initial state: on a polyMesh the 1D oracle along spec.dir from
// spec.origin (TFLES: thickened by F; C2: flameIC2's progress -> reactants rule); on the
// box the structured flameIC / flameIC2 cell for cell. Pre: fm the route's mesh.
FvState fvFlameIC(const Case& c, const FaceMesh& fm, const FvBcSpec& spec, const FvFlameSetup& s);

// The reacting nozzle's design and IC (a [nozzle] run, docs/nozzle.md "Reacting"): the composition oracle on
// the Total row's p0, t0 (box units), eps_c = A_ref / astar off the chamber probe, eps_e = sum_exit S.a / astar,
// the flame's q, cp, gamma; the IC fvReactingNozzleIC with the flame at --upstream from the inlet.
// Pre: spec.nozzle, probe a chamber probe of fm, bc fvBcsOf's table. Error: no Total patch, no choke (exit 3).
struct FvNozzleStart
{
    FvReactingDesign design;
    FvState ic;
};
std::expected<FvNozzleStart, std::string> fvFlameNozzleStart(const Case& c, const FaceMesh& fm, const FvBcSpec& spec,
                                                             const FvFlameSetup& s, const FvFlameProbe& probe,
                                                             const std::vector<FvBc>& bc);

// The diffusion jet and the candle (two reactants, model D): gas, chemistry, freestream,
// gravity along -spec.dir (candle) and the jet's spec at `origin`.
struct FvJetSetup
{
    Transport gas;
    FvChem chem;
    FvFreestream fs;
    std::array<double, 3> g{};
    FvJetSpec js;
};
FvJetSetup fvJetSetup(const Case& c, const FvBcSpec& spec, std::array<double, 3> origin);

// The jet's per-face rows (fvJetRows) over the patch rows of spec; a candle's Far rows
// move with the coflow (at rest with --quiescent), fvAmbientRows sets rho, E per owner.
FvJetRows fvJetCaseRows(const Case& c, const FaceMesh& fm, const FvBcSpec& spec, const FvJetSetup& s);

// --- low-Mach ---------------------------------------------------------------------

// The low-Mach route's --bc spec: c.fvBc on a polyMesh; the box's (fvLowMachBoxBc) with
// --laterals open at the room air (air with --equivalence, else Y = 0), C2's fresh premix
// on xlo and --rim's isothermal `rim` Wall patch.
FvBcSpec fvLowMachCaseSpec(const Case& c);

// The flags' low-Mach box (fvLowMachBox): --laterals open, --rim's staircase removed.
std::expected<FaceMesh, std::string> fvLowMachCaseBox(const Case& c);

// The slot burner's frame on a FaceMesh: x along spec.dir from the inflow plane, y =
// (C - origin).lift + yc. The box's origin is (0, yc, 0), lift +y, depth hz; a polyMesh's
// are its --bc reference (validate checked the axes), depth = V / (A_empty / 2).
struct FvBurnerFrame
{
    std::array<double, 3> dir{1, 0, 0}, origin{}, lift{0, 1, 0};
    double yc = 0, depth = 1;
    // (x, y) of a point in the burner's frame.
    std::array<double, 2> at(const std::array<double, 3>& p) const;
};
// Error (exit 3): a polyMesh not empty along z, no empty face to read the depth from.
std::expected<FvBurnerFrame, std::string> fvBurnerFrame(const Case& c, const FaceMesh& fm, const FvBcSpec& spec);

// The low-Mach FV initial state, both meshes: the burner lmBunsenPoint at the centroids
// (polyMesh) or lowmachIC cell for cell (box, rho u = the MAC x faces' m); the planar flame
// the 1D oracle along spec.dir (polyMesh; TFLES thickened by F; C2 flameIC2's reactants
// rule) or lowmachIC cell for cell (box, rho u = m). Pre: frame of fm when c.flameBunsen.
FvLowMachIC fvLowMachCaseIC(const Case& c, const FaceMesh& fm, const FvBcSpec& spec, const FvBurnerFrame& frame,
                            double sigma);

// The low-Mach FV gas and chemistry (C0; C2 / C2b: two reactants, q per unit fuel cp Tu
// (sigma - 1) s / Y_O1), gravity along -spec.dir, LES and TFLES unset (the caller's).
FvLowMachModel fvLowMachCaseModel(const Case& c, const FvBcSpec& spec, double sigma);

// The burner's inflow row of a face (lmInflowRow): on the box the owner's row j + ny k,
// exactly the structured row; on a polyMesh the face centre's.
std::function<FvLmRow(const std::array<double, 3>& cf, std::int64_t ownerSrc)>
fvBurnerRow(const Case& c, const FvBurnerFrame& frame);

} // namespace dns
