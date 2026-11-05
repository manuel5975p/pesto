// validate(): dns flow -- body, grid, physics, thermal, LES, NSCBC, bands, stretch, walls, output.
#include "dns/config.hpp"
#include "dns/config/internal.hpp"

#include "dns/ogrid.hpp"

#include "dns/remap.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>

using std::numbers::pi;

namespace dns::cfg {

std::optional<ConfigError> flowBody(const RawConfig& r, Case& c)
{
    // A scene file replaces --body: main loads it (phase 1.5) and hands the
    // parsed Scene to the geometry build; the validator only records the path
    // and applies the body-independent checks.
    if (!r.scene.empty())
    {
        if (!r.body.empty())
            return err(2, "--scene", "--scene and --body are mutually exclusive "
                                     "(the scene lists its own objects)");
        c.bodyKind = BodyKind::Scene;
        c.scenePath = r.scene;
    }
    else if (r.body.empty())
        return err(2, "--body", "required: one of " + bodyKeywords()
                                + ", a .stl/.gltf/.glb file, or "
                                  "--scene FILE.toml");
    // Mesh and Scene are kinds the validator ASSIGNS, never keywords the user
    // types: `--body mesh` falls through to the suffix test below and is then
    // rejected by name, as it always was.
    else if (const auto k = bodyKindFromName(r.body);
             k && *k != BodyKind::Mesh && *k != BodyKind::Scene)
        c.bodyKind = *k;
    else if (hasSuffix(r.body, ".stl") || hasSuffix(r.body, ".gltf")
             || hasSuffix(r.body, ".glb"))
    {
        c.bodyKind = BodyKind::Mesh;
        c.meshPath = r.body;
    }
    else
        return err(2, "--body", "unrecognised body '" + r.body + "'; expected "
                                + bodyKeywords()
                                + ", or a path ending in .stl/.gltf/.glb");

    // Body-free box (wall boundary layers, channel-like cases): the
    // incompressible runner requires solid cells, so this is compressible-only.
    if (c.bodyKind == BodyKind::None && c.physics != Physics::Compressible)
        return err(2, "--body", "--body none is compressible-only (the "
                                "incompressible tunnel needs a body)");

    // Far field defaults follow the physics when the user did not choose one.
    // Sponge is NEVER a default (any physics, any mode): it is an escape hatch that
    // eats the near field in a relaxation band. The compressible default is
    // the physical characteristic boundary; --far sponge must be asked for.
    if (!r.given.contains("--far")) c.far = defaultFar(c.physics);

    const bool comp = c.physics == Physics::Compressible;
    if (!farAllowed(c.physics, c.far))
        return err(2, "--far", std::string("--far ") + name(c.far)
                   + (comp ? " is an incompressible far field; compressible "
                             "runs use sponge, nscbc or periodic"
                           : " is a compressible far field; incompressible "
                             "runs use tunnel, fringe or periodic"));

    c.open = r.open;
    if (c.open && c.far != FarField::Sponge && c.far != FarField::Nscbc
        && c.far != FarField::Tunnel)
        return err(2, "--open", std::string("--open has no meaning with --far ")
                   + name(c.far)
                   + " (tunnel, sponge and nscbc have lateral faces)");

    // Floor-mounted bodies (fence blocker, raised floor): they sit ON the
    // tunnel floor (sealed through the ghost row), so they need the
    // incompressible tunnel's solid lateral wall. Compressible wall rows are
    // far-field mirror rows; a ghost body intersecting them is unvalidated --
    // rejected, not silently degraded.
    if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
    {
        const char* b = c.bodyKind == BodyKind::Fence ? "fence" : "floor";
        if (comp && c.bodyKind == BodyKind::Fence)
            return err(2, "--body", "fence is incompressible-only for now (a "
                                    "mid-plate blocker on the compressible "
                                    "mirror wall rows is unvalidated)");
        // Compressible floor: NSCBC/sponge x planes skip the fully-solid
        // rows (the ghost wall owns them); periodic has no walls at all.
        if (comp && c.far == FarField::Periodic)
            return err(2, "--far", "floor needs closed y walls; use --far "
                                   "nscbc (default) or an explicit sponge");
        if (!comp && c.far != FarField::Tunnel)
            return err(2, "--far", std::string(b) +
                                   " sits on the tunnel floor; incompressible "
                                   "runs need --far tunnel (periodic/fringe "
                                   "have no wall for it to stand on)");
        if (c.open)
            return err(2, "--open", std::string("--open removes the lateral "
                                    "walls; the ") + b + " needs the floor wall");
    }
    return std::nullopt;
}

std::optional<ConfigError> flowGrid(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Grid: two separately validated conventions, dispatched on --physics.
    const int ny = r.given.contains("--ny") ? r.ny : r.n;
    if (ny < 1) return err(2, "--ny", "must be >= 1");
    // --nz defaults to 1 (native 2D) for the z-invariant bodies and to --n for
    // the genuinely 3D ones: a sphere or a mesh in a one-cell-thick box has no
    // solid cells at all, which used to surface as a silent no-body run.
    const bool body3d = c.bodyKind == BodyKind::Sphere || c.bodyKind == BodyKind::Mesh;
    const int nzI = r.given.contains("--nz") ? r.nz : (body3d ? r.n : 1);
    if (auto e = checkNz(nzI, "must be >= 1 (interior cells; 1 = native 2D)"); e)
        return e;
    // The revolved O-grid reads --nz 1 as its axisymmetric wedge.
    const bool revolved = r.ogrid && c.bodyKind == BodyKind::Sphere;
    if (body3d && nzI == 1 && !revolved)
        return err(2, "--nz", std::string("--body ") + name(c.bodyKind)
                   + " is a 3D body; --nz 1 leaves it with no solid cells");
    if (!finitePos(r.dr)) return err(2, "--dr", "must be finite and > 0");
    c.dr = r.dr;
    c.nzInterior = nzI;

    c.uInf = 1.0;
    if (r.ogrid)
    {
        // Body-fitted O-grid (dns/ogrid.hpp): the domain is the DISC of radius
        // rFar around the body, not a box, and the wall is a mesh face -- so
        // every immersed-body and box knob is meaningless here and rejected by
        // name rather than silently ignored. --n is the angular cell count,
        // --ny the radial one INCLUDING the two ghost rows.
        if (!comp && r.solver != int(SolverKind::Fv))
            return err(2, "--ogrid", "the body-fitted O-grid is compressible-"
                                     "only (the curvilinear path is)");
        if (c.bodyKind == BodyKind::None)
            return err(2, "--ogrid", "wraps a body; there is none");
        if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
            return err(2, "--ogrid", "wall-mounted bodies are not star-shaped "
                                     "about a centre and have no annulus "
                                     "around them");
        if (auto e = rejectGiven(
                r, {"--focal", "--stretch", "--cut-flux", "--penalize", "--open",
                    "--far", "--dr", "--seq"},
                "belongs to the Cartesian box with an immersed body; --ogrid "
                "replaces both (the wall IS a mesh face, and the far field is "
                "the outer circle)");
            e)
            return e;
        if (r.n < 16)
            return err(2, "--n", "is the ANGULAR cell count on an O-grid; "
                                 "below 16 the contour is not resolved");
        if (ny < 6)
            return err(2, "--ny", "is the RADIAL cell count on an O-grid, two "
                                  "of which are ghost rows; >= 6 needed");
        if (!(std::isfinite(r.ogridFar) && r.ogridFar > 1.0))
            return err(2, "--ogrid-far", "far-boundary radius in DIAMETERS; "
                                         "must be finite and > 1");
        if (!(std::isfinite(r.ogridBeta) && r.ogridBeta >= 0))
            return err(2, "--ogrid-beta", "radial clustering; must be finite "
                                          "and >= 0 (0 = uniform)");
        const double D = pi / 4.0;
        c.ogrid = true;
        c.ogridRFar = r.ogridFar * D;
        c.ogridBeta = r.ogridBeta;
        if (!revolved && r.given.contains("--ogrid-wedge"))
            return err(2, "--ogrid-wedge", "is the revolved (--body sphere) "
                                           "O-grid's axisymmetric wedge angle");
        // The O-grid start is uniform (or at rest) and never reads --kick;
        // its trip is --ogrid-trip.
        if (r.given.contains("--kick"))
            return err(2, "--kick", "is the Cartesian box's wake trip; the "
                                    "O-grid start never applies it -- use "
                                    "--ogrid-trip A");
        // The axisymmetric wedge has no odd modes to seed, and the planar
        // vortex is not solenoidal in the meridional plane (v/r).
        if (revolved && nzI <= 1 && r.given.contains("--ogrid-trip"))
            return err(2, "--ogrid-trip", "seeds a solenoidal vortex behind "
                                          "the body; the axisymmetric wedge "
                                          "(--nz 1) has no mode it could "
                                          "seed and no planar vortex is "
                                          "solenoidal there");
        if (!(std::isfinite(r.ogridTrip) && r.ogridTrip >= 0 &&
              r.ogridTrip <= 0.5))
            return err(2, "--ogrid-trip", "the trip amplitude, / U: must "
                                          "be in [0, 0.5]");
        // Default 0 (bit for bit the untripped start) except on the full
        // revolution, whose pi-symmetry locks out the m = 1 mode.
        c.ogridTrip = r.given.contains("--ogrid-trip") ? r.ogridTrip
                      : revolved && nzI > 1            ? kOgridTripDefault
                                                       : 0.0;
        if (!(revolved && nzI > 1) && r.given.contains("--init-from"))
            return err(2, "--init-from", "starts the full-revolution sphere "
                                         "O-grid (--body sphere --nz N > 1) "
                                         "from a converged wedge only");
        if (revolved && nzI > 1)
        {
            // The FULL revolution: N azimuthal cells, Cyclic k, the axis
            // ghost columns copied from k + N/2 (gpu/axis_bc.hpp). N even so
            // every cell has its partner across the axis; N >= 4 so that
            // partner is not a k neighbour (buildRevolvedOGrid's own floor).
            if (r.given.contains("--ogrid-wedge"))
                return err(2, "--ogrid-wedge", "is the axisymmetric wedge's "
                                               "angle (--nz 1); --nz N > 1 is "
                                               "the full revolution");
            if (nzI % 2 != 0 || nzI < 4)
                return err(2, "--nz", std::format(
                                          "is the full revolution's azimuthal "
                                          "cell count: it must be even and "
                                          ">= 4 (every cell has its partner "
                                          "across the axis, k + N/2); got {}",
                                          nzI));
            c.ogridRevolved = true;
            c.ogridNphi = nzI;
            if (r.given.contains("--init-from")) {
                // The file is read by main (phase 3); here only its shape.
                if (r.initFrom.empty() ||
                    r.initFrom.find_first_of("\"\\\n\r\t") != std::string::npos)
                    return err(2, "--init-from", "a wedge run's --out prefix "
                                                 "(its _final.bin and "
                                                 "_mesh.bin); not empty, no "
                                                 "quotes or control "
                                                 "characters");
                if (r.given.contains("--ramp"))
                    return err(2, "--ramp", "with --init-from: the run "
                                            "starts from the wedge's state, "
                                            "not from rest");
                c.ogridInitFrom = r.initFrom; // flowRans zeroes the ramp
            }
            c.ogridWedgeDeg = 0;
            c.ogridSpan = 1.0;
            c.grid = ogridRevolvedGrid(
                r.n, ny,
                RevolvedSpec{.wedge = false, .dphi = 2.0 * pi / nzI,
                             .nphi = nzI},
                0.5 * D, c.ogridRFar);
            c.nzInterior = nzI;
            // The polygonal body: N planar facets about the axis, frontal
            // area sin(dphi)/dphi of the circle's (qRef stays pi D^2/4).
            const double dphi = 2.0 * pi / nzI;
            const double facet = 1.0 - std::sin(dphi) / dphi;
            if (facet > 1e-2)
                c.warnings.push_back(std::format(
                    "--nz {}: the revolved body is a {}-sided polygon about "
                    "the axis, its frontal area {:.2f}% below the circle's "
                    "(qRef is the circle's); --nz 32 brings it to 0.64%, 64 "
                    "to 0.16%",
                    nzI, nzI, 100.0 * facet));
        }
        else if (revolved)
        {
            // A body of revolution: the O-grid revolved about x. Axisymmetry
            // is a mesh plus a ghost hook (dns/ogrid.hpp, gpu/axis_bc.hpp).
            if (!(std::isfinite(r.ogridWedge) && r.ogridWedge > 0 &&
                  r.ogridWedge <= 10))
                return err(2, "--ogrid-wedge", "the wedge angle in degrees, "
                                               "in (0, 10]");
            c.ogridRevolved = true;
            c.ogridWedgeDeg = r.ogridWedge;
            c.ogridSpan = 1.0;
            c.grid = ogridRevolvedGrid(
                r.n, ny,
                RevolvedSpec{.wedge = true, .dphi = r.ogridWedge * pi / 180.0},
                0.5 * D, c.ogridRFar);
            c.nzInterior = 1;
        }
        else
        {
            // Spanwise extent: pi D, the usual cylinder-DNS span. nz == 1
            // keeps the unit depth the 2D convention requires.
            c.ogridSpan = nzI > 1 ? pi * D : 1.0;
            c.grid = ogridGrid(r.n, ny, nzI, 0.5 * D, c.ogridRFar, c.ogridSpan);
        }
        c.ogridH1 = ogridNominalH1(ny, 0.5 * D, c.ogridRFar, c.ogridBeta);
        if (auto e = checkCells(c.grid.size()); e) return e;
        c.L = 2.0 * c.ogridRFar;
        c.h = c.grid.hy; // nominal radial spacing; the mesh reports the real h1
        // The body sits at the origin: the mesh is built about its centre.
        c.placement = {0.0, 0.0, 0.0, D};
    }
    else if (r.given.contains("--ogrid-wedge"))
        return err(2, "--ogrid-wedge", "is the revolved (--body sphere) "
                                       "O-grid's axisymmetric wedge angle");
    else if (r.given.contains("--ogrid-trip"))
        return err(2, "--ogrid-trip", "seeds the odd azimuthal modes of the "
                                      "full-revolution sphere O-grid (--ogrid "
                                      "--body sphere --nz N > 1) only");
    else if (r.given.contains("--init-from"))
        return err(2, "--init-from", "starts the full-revolution sphere O-grid "
                                     "(--ogrid --body sphere --nz N > 1) from "
                                     "a converged wedge only");
    else if (comp)
    {
        // D = pi/4 fixed, h scales with ny AND --dr: box = dr*D in y
        // (dr*(n/ny)*D in x), D spans ny/dr cells. The dr = 8 default keeps
        // the historical 8D x 8D box BIT-IDENTICAL (the *(dr/8) factor is an
        // exact *1.0 and the 2.1107 placement literal is preserved). No ghost
        // ring: the compressible far fields act on interior rows.
        if (c.dr < 4)
            return err(2, "--dr", "compressible box is dr*D wide; below 4 the "
                                  "body sits under 1D from the inflow row");
        const double D = pi / 4.0;
        c.h = D / 64.0 * (512.0 / double(ny)) * (c.dr / 8.0);
        const double lx = r.n * c.h, ly = ny * c.h;
        const double lz = nzI > 1 ? nzI * c.h : 1.0;
        auto box = makeBox(r.n, ny, nzI, lx, ly, lz);
        if (!box) return box.error();
        c.grid = *box;
        c.L = lx;
        // The floor is pinned to the box bottom (no ghost ring here): the
        // slab spans [0, D], cy = centre of the blocked strip.
        c.placement = {c.dr == 8.0 ? 2.1107 : lx / 3.0 + 0.0163,
                       c.bodyKind == BodyKind::Floor ? D / 2.0
                                                     : ly / 2.0 + 0.0146,
                       nzI > 1 ? lz / 2.0 : 0.5, D};
    }
    else
    {
        // L = 2 pi, h = L/n, one ghost cell on every side; --nz is interior.
        c.L = 2 * pi;
        c.h = c.L / double(r.n);
        const int nxT = r.n + 2, nyT = ny + 2;
        const int nzT = nzI <= 1 ? 1 : nzI + 2;
        const double lz = nzT == 1 ? 1.0 : nzT * c.h;
        auto box = makeBox(nxT, nyT, nzT, nxT * c.h, nyT * c.h, lz);
        if (!box) return box.error();
        c.grid = *box;
        const double D = c.L / c.dr;
        // Per-body placement offsets: each deliberately puts the body off-node
        // so the shedding instability trips. Carried verbatim (masterplan §6).
        // The fence is pinned to the floor instead: the wall face is at y = h
        // (top of the ghost row) and the blocker spans [wall, wall + D], so
        // cy = centre of the blocked strip. Its top edge is off-node through
        // the 0.31h x-offset alone.
        // y (and a sphere's z) is centred on the INTERIOR, cells*h long, as
        // the compressible box centres on ly. The square-box form L/2 is kept
        // verbatim when the axis has n cells -- every ny == n run is
        // bit-identical -- and is wrong otherwise: --ny 32 at --n 64 put the
        // body centre outside the box.
        const auto mid = [&](int cells) {
            return cells == r.n ? c.L / 2.0 : 0.5 * double(cells) * c.h;
        };
        const double cx = c.h + c.L / 3.0 + 0.31 * c.h;
        const double cy = c.bodyKind == BodyKind::Fence
                                  || c.bodyKind == BodyKind::Floor
                              ? c.h + D / 2.0
                              : c.h + mid(ny) + 0.17 * c.h;
        const double cz = c.bodyKind == BodyKind::Sphere
                              ? c.h + mid(nzT == 1 ? r.n : nzI) + 0.23 * c.h
                              : (nzT == 1 ? 0.5 * c.h : c.h + nzI * c.h / 2.0);
        c.placement = {cx, cy, cz, D};
    }

    // (An O-grid's resolution is its angular --n >= 16, and its nominal
    // radial hy measures nothing about D.)
    if (!c.ogrid && c.placement.D < 4 * c.h)
        return err(2, "--dr", "body diameter is under 4 cells; refine --n or lower "
                              "--dr");
    // The fence is D/8 thick; below 32 cells of D the blocker itself is under
    // 4 cells and the sampled cut degenerates.
    if (c.bodyKind == BodyKind::Fence && c.placement.D < 32 * c.h)
        return err(2, "--dr", "fence thickness is D/8; D under 32 cells makes it "
                              "thinner than 4 cells -- refine --n or lower --dr");
    if (c.bodyKind == BodyKind::Floor && c.dr < 2)
        return err(2, "--dr", "floor solidifies the bottom 1/dr of the channel; "
                              "--dr under 2 blocks more than half of it");
    return std::nullopt;
}

std::optional<ConfigError> flowPhysics(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Physics.
    c.mach = r.mach;
    if (comp)
    {
        if (auto e = checkMach(c.mach); e) return e;
        if (r.given.contains("--tol"))
            return err(2, "--tol", "compressible runs are fully explicit; there is "
                                   "no pressure solve to converge");
    }
    else
    {
        if (auto e = rejectGiven(r, {"--mach", "--kick"},
                                 "is compressible-only");
            e)
            return e;
    }

    c.nu = c.uInf * c.placement.D / c.re;
    if (auto e = resolveCfl(r, c, comp ? 0.4 : 0.8); e) return e;
    return std::nullopt;
}

} // namespace dns::cfg

namespace dns::cfg {
namespace {

// Stage: the compressible thermal knobs.
std::optional<ConfigError> flowThermal(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Thermal model. Every knob here is compressible-only and OFF by default:
    // a constant-viscosity adiabatic-wall run is bit-for-bit what it was, so
    // no measured anchor in the repo moves until a run asks for temperature.
    c.tinfK = r.tinf;
    c.prandtl = r.prandtl;
    c.sutherland = r.visc == 1;
    c.twallRatio = r.twall;
    c.bodyIsothermal = r.given.contains("--wall") && r.wallIso == 1;
    c.nscbcThermalInflow = r.nscbcTinflow;
    if (!comp)
        if (auto e = rejectGiven(r,
                                 {"--tinf", "--prandtl", "--visc", "--wall",
                                  "--twall", "--nscbc-tinflow"},
                                 "is compressible-only (the incompressible "
                                 "solver carries no energy equation)");
            e)
            return e;
    if (!finitePos(c.tinfK))
        return err(2, "--tinf", "must be finite and > 0 (kelvin)");
    if (!finitePos(c.prandtl))
        return err(2, "--prandtl", "must be finite and > 0");
    if (!finitePos(c.twallRatio))
        return err(2, "--twall", "must be finite and > 0 (a ratio to Tinf)");
    if (r.given.contains("--twall") && !c.bodyIsothermal)
        return err(2, "--twall", "sets the temperature of an ISOTHERMAL wall; "
                                 "add --wall isothermal (the default body wall "
                                 "is adiabatic)");
    if (c.bodyIsothermal && c.bodyKind == BodyKind::None)
        return err(2, "--wall", "an isothermal wall needs a body; there is none");
    if (c.nscbcThermalInflow && c.far != FarField::Nscbc)
        return err(2, "--nscbc-tinflow", "is the NSCBC entropy inflow target; "
                                         "use --far nscbc");
    return std::nullopt;
}

// Stage: SGS model, the aperture/Brinkman options, the DNS/LES label.
std::optional<ConfigError> flowLes(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    c.cs = r.les;
    if (!(std::isfinite(c.cs) && c.cs >= 0))
        return err(2, "--les", "must be finite and >= 0");
    c.sgsModel = r.sgs;
    if (r.given.contains("--sgs") && c.cs == 0)
        return err(2, "--sgs", "selects the SGS model for --les; give --les CS "
                               "(> 0) or drop --sgs");
    c.cutFlux = r.cutFlux;
    if (c.cutFlux && c.bodyKind == BodyKind::None)
        return err(2, "--cut-flux", "apertures the body's momentum fluxes; "
                                    "there is no body");
    if (c.cutFlux && r.given.contains("--stretch"))
        return err(2, "--cut-flux", "apertures are uniform-mesh geometry; the "
                                    "curvilinear path has none. Drop --stretch");
    if (c.cutFlux && comp)
        return err(2, "--cut-flux", "compressible cut-flux DIVERGES near the "
                                    "body (measured twice, 2026-07-23: the KG "
                                    "split flux loses its KE property under "
                                    "per-face area weighting). Incompressible "
                                    "only until redesigned");
    c.penalize = r.penalize;
    if (!(std::isfinite(c.penalize) && c.penalize >= 0))
        return err(2, "--penalize", "must be finite and >= 0");
    if (c.penalize > 0 && c.bodyKind == BodyKind::None)
        return err(2, "--penalize", "decays the near-body band; there is no body");
    if (c.penalize > 0 && r.given.contains("--stretch"))
        return err(2, "--penalize", "needs the uniform-mesh fluid fractions; "
                                    "drop --stretch");
    if (c.sgsModel == SgsModel::Sigma && c.cs > 0 && c.grid.nz == 1)
        return err(2, "--sgs", "sigma: D_sigma is identically ZERO for any 2D "
                               "flow (that is the model's design); at nz == 1 "
                               "it is a silent no-op. Use nz > 1 or smag");
    // The wall face itself takes the image of nu_t (molecular stress only);
    // the first fluid cell does not, and |S| is the wall shear there.
    if (c.cs > 0 && c.sgsModel == SgsModel::Smagorinsky &&
        c.bodyKind != BodyKind::None && c.grid.nz > 1)
        c.warnings.push_back(std::format(
            "--les {:g} --sgs smag next to a wall: |S| does not vanish in "
            "laminar shear, so the first fluid cell carries nu_t/nu ~ "
            "Cs^2 Delta+^2 = {:.3g} Delta+^2 (no wall damping). --sgs sigma "
            "vanishes in pure shear and is the wall-consistent choice",
            c.cs, c.cs * c.cs));
    c.lesLabel = c.cs > 0;
    return std::nullopt;
}

// Stage: the far-field parameters: NSCBC closure, sink, strip, --pot.
std::optional<ConfigError> flowNscbc(const RawConfig& r, Case& c)
{
    // Far-field parameters.
    c.nscbcBeta = r.nscbcBeta;
    c.nscbcEtaIn = r.given.contains("--nscbc-etain") ? r.nscbcEtaIn : 100.0;
    c.nscbcStrip = r.nscbcStrip;
    c.nscbcEtaInShear = r.nscbcEtaInShear;
    c.nscbcEtaLat = r.nscbcEtaLat;
    c.nscbcSink = r.nscbcSink;
    c.nscbcSinkAmp = r.nscbcSinkAmp;
    // The O-grid's far boundary is its own characteristic row (gpu/ogrid_bc);
    // runFlowOGrid reads none of the NSCBC knobs, so each one is rejected by
    // name rather than accepted and ignored -- including the strip that used
    // to "satisfy" the Re-800 rule below without doing anything.
    if (r.ogrid)
    {
        if (auto e = rejectGiven(
                r, {"--nscbc-strip", "--nscbc-etain", "--nscbc-beta",
                    "--nscbc-abc", "--nscbc-etain-shear", "--nscbc-etalat",
                    "--nscbc-sink", "--nscbc-sink-amp", "--nscbc-tinflow",
                    "--pot"},
                "the O-grid's far boundary is its own characteristic row "
                "(gpu/ogrid_bc), not NSCBC; the flag would be ignored");
            e)
            return e;
        c.nscbcSink = 0;
        if (c.re > 800)
            c.warnings.emplace_back(
                "o-grid far row is unmeasured above Re 800 (the NSCBC Re-800 "
                "rule does not apply to it)");
        return std::nullopt;
    }
    // The sink is ON by default for --far nscbc wherever it is defined
    // (uniform mesh, whole faces, band fits twice on every open axis) --
    // measured free where it does not absorb: cd moved +0.003% on the Re60
    // nscbc cylinder, and it is what carries a run past Re 800 without the
    // strip's near-field cost. An explicit --nscbc-sink (any value, 0 = off)
    // always wins; incompatible cases fall back to 0 rather than rejecting,
    // because a DEFAULT that does not apply is absence, not an error.
    if (!r.given.contains("--nscbc-sink") && c.far == FarField::Nscbc)
    {
        const bool uniform = !remapsMesh(r) && !r.ogrid;
        const bool fits = 32 < c.grid.nx &&
                          (!c.open || (32 < c.grid.ny &&
                                       (c.grid.nz == 1 || 32 < c.grid.nz)));
        c.nscbcSink =
            uniform && fits && c.bodyKind != BodyKind::Floor ? 16 : 0;
    }
    if (c.far == FarField::Nscbc)
    {
        if (!finitePos(c.nscbcEtaIn))
            return err(2, "--nscbc-etain", "must be > 0");
        // Empirically bracketed stability window (masterplan §6.3): 0 pumps a
        // growing global mode, 1.0 diverges at t = 35.
        if (!(std::isfinite(c.nscbcBeta) && c.nscbcBeta >= 0 && c.nscbcBeta < 1))
            return err(2, "--nscbc-beta", "must be in [0,1); 1.0 is measured to diverge "
                                          "at t = 35 (full transverse cancellation "
                                          "is ill-posed at low Mach)");
        if (c.nscbcStrip < 0 || c.nscbcStrip >= c.grid.nx / 4)
            return err(2, "--nscbc-strip", "must be >= 0 and < nx/4");
        if (c.re > 800 && c.nscbcStrip == 0 && c.nscbcSink == 0)
            return err(2, "--nscbc-strip", "pure NSCBC is validated only up to "
                                           "Re 800 (it diverges at Re 2000, 2048^2); "
                                           "pass --nscbc-sink 16 (scale-selective, "
                                           "preferred) or --nscbc-strip ~ny/64 "
                                           "(freestream sponge; docs/nscbc.md)");
        if (c.nscbcSink < 0 || 2 * c.nscbcSink >= c.grid.nx ||
            (c.open && c.nscbcSink > 0 && 2 * c.nscbcSink >= c.grid.ny))
            return err(2, "--nscbc-sink", "band must be >= 0 and fit twice on "
                                          "every open axis");
        if (!(std::isfinite(c.nscbcSinkAmp) && c.nscbcSinkAmp > 0 &&
              c.nscbcSinkAmp <= 2.0))
            return err(2, "--nscbc-sink-amp",
                       "must be in (0, 2]; sigma dt <= 2 * cfl * axes is the "
                       "SSP-RK3 real-axis bound at cfl 0.4");
        // The sink's cutoff is quoted in CELLS on a uniform mesh; under a
        // remap the physical cutoff varies across the band, which needs its
        // own measurement before it is allowed. Rejected, never ignored.
        if (c.nscbcSink > 0 && remapsMesh(r))
            return err(2, "--nscbc-sink", "is measured on the uniform mesh; "
                                          "--stretch/--focal remap the band "
                                          "(unmeasured -- not yet allowed)");
        if (c.nscbcSink > 0 && c.bodyKind == BodyKind::Floor)
            return err(2, "--nscbc-sink", "band rows overlap the solid floor "
                                          "rows the ghost wall owns "
                                          "(--body floor)");
        if (c.nscbcSink > 0 && r.ogrid)
            return err(2, "--nscbc-sink", "is a Cartesian face band; the "
                                          "body-fitted O-grid has no such "
                                          "faces");
    }
    else if (r.given.contains("--nscbc-strip") && c.nscbcStrip > 0)
        return err(2, "--nscbc-strip", "requires --far nscbc");
    if (c.far != FarField::Nscbc && r.given.contains("--nscbc-sink") &&
        c.nscbcSink > 0)
        return err(2, "--nscbc-sink", "requires --far nscbc");
    if (r.given.contains("--nscbc-sink-amp") && c.nscbcSink == 0)
        return err(2, "--nscbc-sink-amp", "sets the strength of the "
                                          "--nscbc-sink band; pass a width");
    if (c.far != FarField::Nscbc && r.given.contains("--nscbc-etain"))
        return err(2, "--nscbc-etain", "requires --far nscbc");
    if (r.given.contains("--nscbc-etain-shear"))
    {
        if (c.far != FarField::Nscbc)
            return err(2, "--nscbc-etain-shear", "requires --far nscbc");
        if (!finitePos(c.nscbcEtaInShear))
            return err(2, "--nscbc-etain-shear", "must be > 0");
    }
    if (r.given.contains("--nscbc-etalat"))
    {
        if (c.far != FarField::Nscbc || !c.open)
            return err(2, "--nscbc-etalat", "is the OPEN lateral faces' relax "
                                            "rate; needs --far nscbc --open "
                                            "(walled laterals never read it)");
        if (!finitePos(c.nscbcEtaLat))
            return err(2, "--nscbc-etalat", "must be > 0");
        if (c.nscbcEtaLat > 0.5)
            c.warnings.emplace_back(
                "--nscbc-etalat > 0.5: the lateral relax writes vorticity onto "
                "the face in proportion to this rate (a stiff clamp is a wall "
                "in disguise; measured 2.16x the incident scale at 1.0)");
    }
    // The derived outflow coefficients. `adaptive` is formed inside the NSCBC
    // kernel from quantities it already has, so it constrains nothing beyond
    // needing the closure at all. `modal` owns ONE plane -- the uniform x-hi
    // face of a laterally walled box -- and each rejection below is a case
    // where that plane is not what the transform assumes. Rejected, never
    // silently ignored.
    c.nscbcAbc = r.nscbcAbc;
    if (c.nscbcAbc != NscbcAbc::YooIm && c.far != FarField::Nscbc)
        return err(2, "--nscbc-abc", "is the NSCBC outflow's transverse "
                                     "coefficient; use --far nscbc");
    if (c.nscbcAbc == NscbcAbc::Modal)
    {
        if (c.open)
            return err(2, "--nscbc-abc modal",
                       "needs walled lateral faces: with --open the far-field "
                       "problem is not separable in the transverse directions, "
                       "so the modal basis it transforms in does not exist "
                       "(--nscbc-abc adaptive works there)");
        if (remapsMesh(r))
            return err(2, "--nscbc-abc modal",
                       "is derived on the UNIFORM outflow plane; --stretch and "
                       "--focal remap it (--nscbc-abc adaptive works there)");
        if (r.ogrid)
            return err(2, "--nscbc-abc modal",
                       "is a Cartesian x-face closure; the body-fitted O-grid "
                       "has no such face");
        if (c.bodyKind == BodyKind::Floor)
            return err(2, "--nscbc-abc modal",
                       "needs a whole outflow plane; --body floor makes its "
                       "lower rows solid");
        if (c.grid.nz > 1)
            c.warnings.emplace_back(
                "--nscbc-abc modal transforms the outflow plane along y only; "
                "z is periodic, not mirror-symmetric, so a spanwise-oblique "
                "wave still sees a per-z-line coefficient");
    }
    c.farPot = r.farPot;
    if (c.farPot && c.bodyKind == BodyKind::None)
        return err(2, "--pot", "models the body's wake source and displacement "
                               "doublet; meaningless with --body none");
    if (c.farPot && c.far != FarField::Nscbc
        && !(c.far == FarField::Tunnel && c.open))
        return err(2, "--pot", "requires --far nscbc (compressible) or "
                               "--far tunnel --open (incompressible) -- the "
                               "correction retargets an open far field");
    return std::nullopt;
}

// Stage: sponge/fringe band widths.
std::optional<ConfigError> flowBands(const RawConfig& r, Case& c)
{
    // Sponge / fringe band width. clamp(0.1n, 8, n/2-1) is UB when n < 18.
    // Band width is measured along x for BOTH physics. (This was once a
    // `comp ? nx : nx` ternary -- a physics distinction someone intended and
    // abandoned. Restoring one needs a measurement, not a guess.)
    const int bandBase = c.grid.nx;
    if ((c.far == FarField::Fringe || c.far == FarField::Sponge
         || c.far == FarField::Nscbc)
        && bandBase < 18)
        return err(2, "--n", "sponge/fringe far fields need at least 18 cells along x");
    c.spongeCells = r.spongeCells > 0
                        ? r.spongeCells
                        : std::clamp(int(0.1 * bandBase), 8, bandBase / 2 - 1);
    if (c.spongeCells < 1 || c.spongeCells >= bandBase / 2)
        return err(2, "--sponge-cells", "must be >= 1 and < nx/2");

    // Auto: band z only when the box is thick enough to hold the band. A thin
    // quasi-2D run gets x,y only -- a z band there would sponge the whole wake.
    const bool zBandFits = c.grid.nz >= 2 * c.spongeCells + 2;
    c.fringeAxes = r.fringeAxes != 0 ? r.fringeAxes : (zBandFits ? 3 : 2);
    if (c.fringeAxes != 2 && c.fringeAxes != 3)
        return err(2, "--fringe-axes", "must be 2 or 3");
    if (c.far == FarField::Fringe && c.fringeAxes == 3 && !zBandFits)
        return err(2, "--fringe-axes", "a z fringe band needs nz >= 2*cells + 2; "
                                       "pass --fringe-axes 2 for a z-invariant run");
    return std::nullopt;
}

} // namespace
} // namespace dns::cfg

namespace dns::cfg {

std::optional<ConfigError> flowStretch(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Curvilinear y-stretch + wall type. Scope is deliberate: the interior
    // scheme (curv finite-volume path) is general and NSCBC evaluates the
    // PointRemap on device (physical distances in its one-sided and
    // transverse terms), but the immersed ghost wall still assumes uniform
    // spacing and the curvilinear solver has no LES -- each combination is
    // rejected, not silently degraded. Sponge stays explicit per the ground
    // rule: the validator never swaps --far for the user.
    c.stretchBeta = r.stretch;
    c.noslipY = r.noslip;
    if (!(std::isfinite(c.stretchBeta) && c.stretchBeta >= 0 && c.stretchBeta <= 4))
        return err(2, "--stretch", "must be in [0,4] (tanh clustering strength; "
                                   "4 already compresses the wall cell ~190x)");
    if (c.stretchBeta > 0)
    {
        if (!comp)
            return err(2, "--stretch", "the curvilinear solver is compressible-"
                                       "only; --stretch needs --physics "
                                       "compressible");
        if (c.bodyKind != BodyKind::None)
            return err(2, "--stretch", "the ghost wall samples the body SDF at "
                                       "uniform cell positions; stretched runs "
                                       "need --body none for now");
        if (c.open)
            return err(2, "--open", "--stretch clusters cells toward the closed "
                                    "y walls; open lateral faces defeat it");
        if (c.far != FarField::Nscbc && c.far != FarField::Sponge)
            return err(2, "--far", "stretched-y runs need --far nscbc (the "
                                   "characteristic terms evaluate the PointRemap) "
                                   "or an explicit --far sponge");
        // Resolved wall resolution: first INTERIOR cell (row 1; rows 0 and
        // ny-1 are the wall-mirror ghost rows) and max growth ratio. These
        // set the acoustic dt, so --dry-run must surface them.
        const TanhStretchY R{double(c.grid.ny) * c.grid.hy, c.stretchBeta};
        const double hy = c.grid.hy;
        c.stretchH1 = R.mapY(2 * hy) - R.mapY(hy);
        c.stretchGrowth = std::max(
            c.stretchGrowth,
            maxAdjacentGrowth([&](double y) { return R.mapY(y); }, hy, 1,
                              c.grid.ny, R.mapY(hy) - R.mapY(0.0)));
    }
    return std::nullopt;
}

} // namespace dns::cfg

namespace dns::cfg {
namespace {

// Stage: the no-slip y walls.
std::optional<ConfigError> flowWalls(const RawConfig&, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    if (c.noslipY)
    {
        if (!comp)
            return err(2, "--noslip", "the y mirror walls belong to the "
                                      "compressible far field; --noslip needs "
                                      "--physics compressible");
        if (c.open)
            return err(2, "--noslip", "--open removes the y walls; there is "
                                      "nothing to make no-slip");
        if (c.far != FarField::Sponge && c.far != FarField::Nscbc)
            return err(2, "--noslip", "requires a far field with closed y walls "
                                      "(--far sponge or --far nscbc)");
    }
    return std::nullopt;
}

// Stage: quiet start and trip.
std::optional<ConfigError> flowStartup(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Start-up. The raw default (2.0) is the COMPRESSIBLE default; the
    // incompressible ramp is opt-in so existing impulsive-start trajectories
    // and anchors do not silently move.
    c.ramp = comp ? r.ramp : (r.given.contains("--ramp") ? r.ramp : 0.0);
    if (!comp && c.ramp > 0 && (r.seq > 1 || r.steady > 0))
        return err(2, "--ramp", "the quiet start would re-ramp the developed "
                                "field each --seq level / fight --steady; "
                                "use an impulsive start there");
    // The O-grid never applies the kick (flowGrid rejects a given one): the
    // case carries the value the run applies.
    c.kick = c.ogrid ? 0.0 : r.kick;
    if (!(std::isfinite(c.ramp) && c.ramp >= 0)) return err(2, "--ramp", "must be >= 0");
    if (!std::isfinite(c.kick)) return err(2, "--kick", "must be finite");
    return std::nullopt;
}

// Stage: the pressure solve and the dt recompute.
std::optional<ConfigError> flowSolver(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Solver.
    c.tol = r.tol;
    if (!finitePos(c.tol)) return err(2, "--tol", "must be > 0");
    c.maxIters = r.maxIters;
    if (c.maxIters < 1) return err(2, "--max-iters", "must be >= 1");
    // Default per current app: 16 for the compressible cylinder (a validated
    // perf fix, masterplan §6.7), 1 everywhere else.
    if (auto e = resolveDtRecompute(
            r, c, comp && c.bodyKind == BodyKind::Cylinder ? 16 : 1);
        e)
        return e;
    return std::nullopt;
}

// Stage: the per-body options.
std::optional<ConfigError> flowGeometry(const RawConfig& r, Case& c)
{
    // Geometry options.
    // Which body takes which option is bodyOptions()'s, the table the editor
    // reads too.
    const BodyOptions bo = bodyOptions(c.bodyKind);
    c.aspect = r.aspect;
    if (bo.aspect && !(c.aspect > 1.0))
        return err(2, "--aspect", "must be > 1 (the lens needs w > hh)");
    c.dimples = r.dimples;
    if (c.dimples < 0) return err(2, "--dimples", "must be >= 0");
    if (c.dimples > 0 && !bo.dimples)
        return err(2, "--dimples", "only cylinder and sphere carry dimples");
    if (c.dimples > 0 && c.placement.D < 16 * c.h)
        return err(2, "--dimples", "dimple pits are 5h wide; D must span at least 16 "
                                   "cells or they consume the body");
    c.bumps = r.bumps;
    if (c.bumps < 0) return err(2, "--bumps", "must be >= 0");
    if (c.bumps > 0 && !bo.bumps)
        return err(2, "--bumps", "the surface ripple belongs to --body floor");
    c.bumpH = r.bumpH;
    if (c.bumps > 0)
    {
        if (!(std::isfinite(c.bumpH) && c.bumpH >= 4 && c.bumpH <= 48))
            return err(2, "--bump-h", "must be in [4,48] cells (base = height, "
                                      "63-degree flanks; under 4 the tip is "
                                      "sub-cell, over 48 it is a fence, not "
                                      "roughness)");
        if (c.bumps > int(0.65 * c.n / (2.0 * c.bumpH)))
            return err(2, "--bumps", "spike pitch is under twice the tooth "
                                     "base; the teeth merge -- fewer bumps, "
                                     "shorter --bump-h, or more cells");
        if (c.placement.D < 4 * c.bumpH * c.h)
            return err(2, "--bump-h", "spikes taller than D/4 are a fence "
                                      "field, not surface roughness -- lower "
                                      "--bump-h or --dr");
    }
    c.scale = r.scale;
    if (!(std::isfinite(c.scale) && c.scale >= 0))
        return err(2, "--scale", "must be finite and >= 0 (0 = fit D)");
    c.pitch = r.pitch;
    c.yaw = r.yaw;
    c.robust = r.robust;
    if (!bo.mesh)
        if (auto e = rejectGiven(r, {"--scale", "--pitch", "--yaw", "--robust"},
                                 "applies to mesh bodies only");
            e)
            return e;
    return std::nullopt;
}

// Stage: run length, stationary mode, sequencing, frames, metadata.
std::optional<ConfigError> flowOutput(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // A compressible --steady run marches in PSEUDO-time: it has no run
    // length (flowRans rejects --ft there) and stops on its integrals.
    c.compSteady = comp && r.steady > 0;
    // Run length + output cadences.
    c.ft = r.ft;
    if (!c.compSteady && !finitePos(c.ft))
        return err(2, "--ft", "must be finite and > 0");
    // One flow-through is L/uInf in both physics (c.L already carries each
    // physics' own box convention). Formerly a `comp ? X : X` ternary.
    c.tEnd = c.compSteady ? 0.0 : c.ft * c.L / c.uInf;

    // Stationary mode. Incompressible: tEnd stops being the run length and
    // becomes the give-up cap -- not converging by then is exit 8, never a
    // silent answer. Compressible: the pseudo-time march (flowRans).
    c.steadyTol = r.steady;
    if (c.steadyTol < 0 || !std::isfinite(c.steadyTol))
        return err(2, "--steady", "must be finite and >= 0 (0 = time-march)");
    if (c.steadyTol > 0 && c.cs > 0)
        return err(2, "--steady",
                   "with --les is meaningless: a filtered field has no "
                   "stationary state to converge to");

    // Grid sequencing. Only the initial guess comes from the coarse level, so
    // the fine-grid fixed point is unchanged; the ladder just skips paying the
    // startup transient at full resolution.
    c.seqLevels = r.seq;
    if (c.seqLevels < 1 || c.seqLevels > 6)
        return err(2, "--seq", "must be in [1,6] (1 = off)");
    if (c.seqLevels > 1)
    {
        if (c.steadyTol <= 0)
            return err(2, "--seq", "requires --steady: without a convergence "
                                   "test there is nothing to advance a level on");
        // The coarsest level still has to resolve the body: same D >= 4h rule.
        const int nCoarse = r.n >> (c.seqLevels - 1);
        if (nCoarse < 16 || double(nCoarse) < 4.0 * c.dr)
            return err(2, "--seq", "coarsest level would not resolve the body "
                                   "(needs n/2^(levels-1) >= max(16, 4*dr)); "
                                   "use fewer levels or a larger --n");
    }
    if (auto e = validateFrameOutputs(r, c); e) return e;
    c.vortMax = r.vortMax;
    if (!(std::isfinite(c.vortMax) && c.vortMax >= 0))
        return err(2, "--vort-max", "must be finite and >= 0 (0 = auto scale)");
    c.vortAsinh = r.vortMap == 1;
    if (c.vortAsinh && !(c.vortMax > 0))
        return err(2, "--vort-map", "asinh needs an explicit full-scale clip; "
                                    "set --vort-max (~ the peak |vorticity| in "
                                    "uInf/D, e.g. sqrt(Re))");
    c.speedDelta = r.speedMap == 1;
    c.tempFrames = r.tempFrames;
    if (c.tempFrames && !comp)
        return err(2, "--temp", "the incompressible solver carries no energy "
                                "equation and no temperature field; use "
                                "--physics compressible");
    c.vortSigned = r.vortSigned;
    if (c.vortSigned && !comp)
        return err(2, "--vort-signed", "incompressible frames are already "
                                       "signed z-vorticity; the flag selects "
                                       "it for 3D compressible runs");

    // Pseudo-time has no frame clock: a compressible steady run defaults to
    // the stats alone (flowRans rejects time-keyed kinds that are asked for).
    if (auto e = c.compSteady ? resolveEmit(c, {Emit::Stats})
                              : resolveEmit(c, {Emit::Stats, Emit::Bin});
        e)
        return e;
    // PNG at nz > 1 renders the centre z-plane (same machinery the channel
    // already uses); the old flow-only nz == 1 restriction was conservative,
    // not load-bearing.

    // NOTE: qRef is deliberately NOT resolved here. It is 0.5 rho uInf^2 A
    // with A the frontal area MEASURED from the cut apertures, and validate()
    // touches no filesystem and builds no geometry, so it cannot know A. The
    // field that used to live here held 0.5 uInf^2 D span -- a placeholder
    // that --dry-run printed as if it were the number the run would divide by.
    // The real value comes from dns/geometry.hpp's referenceQ*(), once.

    c.meta = {.D = c.placement.D,
              .re = c.re,
              .mach = comp ? c.mach : 0.0,
              .frameDt = c.fdt,
              .cs = c.cs,
              .les = c.lesLabel,
              .body = c.bodyKind == BodyKind::Mesh    ? c.meshPath
                      : c.bodyKind == BodyKind::Scene ? "scene:" + c.scenePath
                                                      : name(c.bodyKind),
              // The O-grid's far field is its own characteristic outer row;
              // c.far is the unused Cartesian default there.
              .farField = c.ogrid ? std::string("ogrid") : name(c.far),
              .physics = name(c.physics),
              .kick = comp ? c.kick : 0.0,
              .sgs = c.lesLabel ? name(c.sgsModel) : "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return std::nullopt;
}

} // namespace
} // namespace dns::cfg

namespace dns::cfg {

std::expected<Case, ConfigError> validateFlow(const RawConfig& r, Case c)
{
    using Stage = std::optional<ConfigError> (*)(const RawConfig&, Case&);
    static constexpr Stage kStages[] = {
        flowBody, flowGrid, flowPhysics, flowThermal,
        flowLes, flowNscbc, flowBands, flowStretch,
        flowFocal, flowRotor, flowWalls, flowStartup,
        flowSolver, flowGeometry, flowOutput, flowRans
    };
    for (const Stage s : kStages)
        if (auto e = s(r, c); e) return std::unexpected(*e);
    return c;
}

} // namespace dns::cfg
