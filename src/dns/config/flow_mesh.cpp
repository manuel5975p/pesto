// validate(): dns flow --focal clustering and the --rotor-* moving geometry.
#include "dns/config.hpp"
#include "dns/config/internal.hpp"
#include "dns/numparse.hpp"

#include "dns/ogrid.hpp"

#include "dns/remap.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>

using std::numbers::pi;

namespace dns::cfg {

std::optional<ConfigError> flowFocal(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Focal clustering: a fine plateau box (typically the body plus its wake)
    // with a coarse far field, one independent Focal1D per axis. Same scope
    // as --stretch (compressible curvilinear, NSCBC or explicit sponge; LES is
    // allowed, the SGS operators read the mesh metrics and the per-cell
    // Delta), but bodies ARE supported here: the ghost wall classifies
    // against the mapped cell centres. The blend support must stay clear of
    // the domain faces, so the map is the exact identity there and the
    // periodic wrap stays C^2.
    if (r.given.contains("--focal"))
    {
        if (!comp)
            return err(2, "--focal", "the curvilinear solver is compressible-"
                                     "only; --focal needs --physics "
                                     "compressible");
        if (c.stretchBeta > 0)
            return err(2, "--focal", "--stretch and --focal are different "
                                     "curvilinear meshes; pick one");
        // Bodies ARE supported (the ghost wall classifies against the mapped
        // cell centres), but only the analytic ones: their AABB follows from
        // the placement alone, and validate() may not touch the filesystem to
        // read a mesh.
        // A ROTOR is the exception, and for a reason that is not a loophole:
        // its body sweeps a shaft-aligned cylinder of radius D/2 and half-
        // height --rotor-span * D, so the extent that matters is the SWEPT one
        // and that follows from the placement and two flags. --rotor-span is
        // capped at 0.5 below, so the placement cube already contains it and
        // the plateau derivation needs no change at all.
        const bool sweptBox = r.rotorMach != 0;
        if ((c.bodyKind == BodyKind::Mesh || c.bodyKind == BodyKind::Scene)
            && !sweptBox)
            return err(2, "--focal", "a mesh/scene body's bounding box is only "
                                     "known after the file is loaded, and "
                                     "validation touches no filesystem; give "
                                     "the plateau explicitly for now (a rotor "
                                     "is exempt: its swept extent follows from "
                                     "--dr and --rotor-span)");
        if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
            return err(2, "--focal", "the wall-mounted bodies span the full "
                                     "domain width; there is no compact "
                                     "plateau to cluster on");
        if (c.cutFlux || c.penalize > 0)
            return err(2, "--focal", "apertures and the Brinkman band are "
                                     "uniform-mesh geometry; the curvilinear "
                                     "path has none");
        if (c.open)
            return err(2, "--open", "--focal needs the closed lateral faces "
                                    "its blend margin is measured against");
        if (c.far != FarField::Nscbc && c.far != FarField::Sponge)
            return err(2, "--far", "focal runs need --far nscbc (the "
                                   "characteristic terms evaluate the "
                                   "PointRemap) or an explicit --far sponge");

        c.focalRatio = r.focalRatio;
        if (!(std::isfinite(c.focalRatio) && c.focalRatio > 1
              && c.focalRatio <= 16))
            return err(2, "--focal-ratio", "must be in (1,16] (coarse/fine "
                                           "cell-size ratio; the fine spacing "
                                           "sets the acoustic dt)");
        const double blendFrac = r.focalBlend;
        if (!(finitePos(blendFrac) && blendFrac <= 2))
            return err(2, "--focal-blend", "must be in (0,2] (blend width per "
                                           "side as a fraction of the plateau)");

        // The body's axis-aligned bounds, when there is a body: the
        // containment rule below keeps the ghost wall's neighbourhood inside
        // the plateau, so it always sees a locally UNIFORM (anisotropic) grid
        // -- the geometry the Cartesian construction was validated on. The
        // cylinder is z-invariant, so its z extent is the whole box.
        const char* axName[3] = {"x", "y", "z"};
        const bool hasBody = c.bodyKind != BodyKind::None;
        const bool zSpanning = c.bodyKind == BodyKind::Cylinder;
        const double bLo[3] = {c.placement.cx - 0.5 * c.placement.D,
                               c.placement.cy - 0.5 * c.placement.D,
                               c.placement.cz - 0.5 * c.placement.D};
        const double bHi[3] = {c.placement.cx + 0.5 * c.placement.D,
                               c.placement.cy + 0.5 * c.placement.D,
                               c.placement.cz + 0.5 * c.placement.D};

        // Every per-axis rule below, auto and explicit alike, is evaluated on
        // the map actually built from the PHYSICAL box (Focal1D::
        // fromPhysical): the body is placed in physical coordinates, so the
        // plateau, the guard and the containment test must be too. The face
        // margin is a CELL count and is therefore tested in reference space,
        // where cells are uniform.
        const int nAx[3] = {c.grid.nx, c.grid.ny, c.grid.nz};
        const double hAx[3] = {c.grid.hx, c.grid.hy, c.grid.hz};
        auto axisMap = [&](int a, double lo, double hi) {
            return Focal1D::fromPhysical(nAx[a] * hAx[a], lo, hi,
                                         blendFrac * (hi - lo), c.focalRatio);
        };
        // Identity margin at both faces: the blend support must end at least
        // 4 cells short of each domain face (x also clears the NSCBC rows and
        // any absorbing strip), so the far grid is exactly uniform where the
        // boundary treatment reads it.
        auto marginCells = [&](int a) {
            return a == 0 ? std::max(4, c.nscbcStrip + 2) : 4;
        };
        auto clearsFaces = [&](int a, const Focal1D& f) {
            return f.active && f.clears(marginCells(a) * hAx[a]);
        };
        auto fineOf = [&](int a, const Focal1D& f) {
            return f.c * f.r * hAx[a];
        };
        auto holdsBody = [&](int a, double lo, double hi, double guard) {
            return lo <= bLo[a] - guard && hi >= bHi[a] + guard;
        };

        double box[6];
        if (r.focal == "auto")
        {
            if (!hasBody)
                return err(2, "--focal", "auto needs a body; --body none has "
                                         "no AABB to focus on");
            if (!(std::isfinite(r.focalWake) && r.focalWake >= 0))
                return err(2, "--focal-wake", "must be finite and >= 0");
            if (!(std::isfinite(r.focalPad) && r.focalPad >= 0))
                return err(2, "--focal-pad", "must be finite and >= 0");
            const double D = c.placement.D, pad = r.focalPad * D;
            for (int a = 0; a < 3; ++a)
            {
                // Axes the body does not localize stay uniform.
                if (a == 2 && (zSpanning || c.grid.nz == 1))
                {
                    box[4] = box[5] = 0.0;
                    continue;
                }
                // Smallest plateau: the body plus FOUR fine cells of that
                // plateau's own spacing -- one more than the containment rule
                // below asks, so rounding cannot reject auto's own box. The
                // fine spacing grows with the box (through c), hence the
                // fixed point; it is monotone from the c = 1 lower bound and
                // contracts by ~8 h/L per step.
                double guard = 4.0 * hAx[a] / c.focalRatio;
                for (int it = 0; it < 64; ++it)
                {
                    const double g2 = 4.0 * fineOf(a, axisMap(a, bLo[a] - guard,
                                                              bHi[a] + guard));
                    if (g2 == guard) break;
                    guard = g2;
                }
                const double need[2] = {bLo[a] - guard, bHi[a] + guard};
                // Desired plateau, never smaller than the smallest one. It
                // shrinks toward `need` until it fits; the resolved box is what
                // --dry-run prints, so the shrink is never silent.
                const double want[2] = {
                    std::min(need[0], bLo[a] - pad),
                    std::max(need[1],
                             bHi[a] + pad + (a == 0 ? r.focalWake * D : 0.0))};
                auto fits = [&](double lo, double hi) {
                    const Focal1D f = axisMap(a, lo, hi);
                    return clearsFaces(a, f)
                           && holdsBody(a, lo, hi, 3.0 * fineOf(a, f));
                };
                if (!fits(need[0], need[1]))
                    return err(2, "--focal", std::string(axName[a])
                                   + " has no plateau that both contains the "
                                     "body and clears the domain faces: the "
                                     "fine cells cost --focal-ratio x the "
                                     "plateau width out of the grid, which "
                                     "coarsens the far cells until the face "
                                     "margin no longer fits. Raise --n, or "
                                     "lower --focal-ratio or --focal-blend");
                double t = 1.0;
                if (!fits(want[0], want[1]))
                {
                    double lo = 0.0, hi = 1.0; // largest fitting t
                    for (int it = 0; it < 60; ++it)
                    {
                        const double mid = 0.5 * (lo + hi);
                        const double a0 = need[0] + mid * (want[0] - need[0]);
                        const double a1 = need[1] + mid * (want[1] - need[1]);
                        (fits(a0, a1) ? lo : hi) = mid;
                    }
                    t = lo;
                }
                box[2 * a] = need[0] + t * (want[0] - need[0]);
                box[2 * a + 1] = need[1] + t * (want[1] - need[1]);
            }
        }
        // "x0,x1,y0,y1,z0,z1"; a degenerate pair leaves that axis uniform.
        else {
            std::string_view s(r.focal);
            for (int q = 0; q < 6; ++q)
            {
                const std::size_t comma = s.find(',');
                if (!parseDouble(s.substr(0, comma), box[q]))
                    return err(2, "--focal", "expects six numbers "
                                             "x0,x1,y0,y1,z0,z1 (world units) "
                                             "or 'auto'");
                if (comma == std::string_view::npos)
                {
                    if (q != 5)
                        return err(2, "--focal", "expects six comma-separated "
                                                 "numbers x0,x1,y0,y1,z0,z1");
                    s = {};
                }
                else
                    s = s.substr(comma + 1);
            }
            if (!s.empty())
                return err(2, "--focal", "expects exactly six numbers "
                                         "x0,x1,y0,y1,z0,z1");
        }

        for (int a = 0; a < 3; ++a)
        {
            const double lo = box[2 * a], hi = box[2 * a + 1];
            if (hi <= lo)
            {
                // Degenerate pair = leave this axis uniform.
                if (hi < lo)
                    return err(2, "--focal", std::string(axName[a])
                                                 + " plateau has hi < lo");
                c.focalHFine[a] = c.focalHCoarse[a] = hAx[a];
                continue;
            }
            if (a == 2 && c.grid.nz == 1)
                return err(2, "--focal", "nz == 1 IS the exact 2D restriction; "
                                         "a z plateau there is meaningless -- "
                                         "pass z0,z1 equal");
            const Focal1D f = axisMap(a, lo, hi);
            if (!clearsFaces(a, f))
                return err(2, "--focal",
                           std::string(axName[a]) + " plateau plus its blend "
                           "must clear both domain faces by " +
                           std::to_string(marginCells(a)) + " (coarse) cells "
                           "(the map must be the exact identity there). The "
                           "fine cells cost --focal-ratio x the plateau width "
                           "out of the grid, which coarsens the far cells; "
                           "shrink the box, --focal-blend or --focal-ratio, "
                           "or raise --n");
            const double fine = fineOf(a, f);
            if (hi - lo < 4 * fine)
                return err(2, "--focal", std::string(axName[a])
                                             + " plateau spans under 4 fine "
                                               "cells; widen it");
            // Containment: the body plus the ghost wall's 3-cell mirror
            // neighbourhood must sit inside the PLATEAU, never in the blend.
            // The wall stencil is then built on locally uniform cells, which
            // is the geometry it was validated on.
            if (hasBody && !(a == 2 && zSpanning))
            {
                const double guard = 3.0 * fine;
                if (!holdsBody(a, lo, hi, guard))
                    return err(2, "--focal",
                               std::string(axName[a]) + " plateau must contain "
                               "the body plus a 3-fine-cell guard band "
                               "([" + std::to_string(bLo[a] - guard) + "," +
                               std::to_string(bHi[a] + guard) + "]); the ghost "
                               "wall's mirror stencil must not straddle the "
                               "coarse/fine blend");
            }
            // The case carries the PHYSICAL plateau and blend; focalRemap()
            // turns them into this same map for every consumer.
            c.focalActive = true;
            c.focalLo[a] = lo;
            c.focalHi[a] = hi;
            c.focalW[a] = blendFrac * (hi - lo);
            c.focalHFine[a] = fine;
            c.focalHCoarse[a] = f.c * hAx[a];
            // Discrete adjacent growth: the KG flux carries no dissipation,
            // so a fast coarse->fine transition is a dispersive reflector.
            c.focalGrowth = std::max(
                c.focalGrowth,
                maxAdjacentGrowth([&](double X) { return f.map(X); }, hAx[a], 1,
                                  nAx[a], f.map(hAx[a]) - f.map(0.0)));
        }
        if (!c.focalActive)
            return err(2, "--focal", "every axis pair is degenerate; the mesh "
                                     "would be uniform -- drop --focal");
        if (c.focalGrowth > 1.25)
            return err(2, "--focal-blend", "max adjacent cell growth " +
                       std::to_string(c.focalGrowth) + " exceeds 1.25; the "
                       "central flux has no dissipation and reflects off a "
                       "sharp transition. Widen --focal-blend or lower "
                       "--focal-ratio");
        c.focalDtFactor = hFine(c) / hMin(c.grid);
    }
    else
    {
        if (auto e = rejectGiven(r,
                                 {"--focal-ratio", "--focal-blend",
                                  "--focal-wake", "--focal-pad"},
                                 "requires --focal");
            e)
            return e;
    }
    // Honest labelling: the boundary layer scales as D/sqrt(Re), and it is
    // resolved by the spacing AT the body -- the plateau's under --focal, which
    // the containment rule above guarantees the body sits on. Hence here, once
    // the mesh is known, and not with the LES label.
    // On the O-grid the spacing at the wall is the CLUSTERED radial
    // distribution's, not the nominal hy: count the cells D/sqrt(Re) spans
    // through it (exact for a cylinder, the nominal circle otherwise).
    const double hBody = c.focalActive ? hFine(c) : c.h;
    const double blLen = c.placement.D / std::sqrt(c.re);
    const double blCells =
        c.ogrid ? ogridRadialCells(blLen, c.grid.ny, 0.5 * c.placement.D,
                                   c.ogridRFar, c.ogridBeta)
                : blLen / hBody;
    if (blCells < 3.0 && !c.lesLabel && c.bodyKind != BodyKind::None)
    {
        c.warnings.push_back(
            std::format("boundary layer is {:.1f} cells (D/sqrt(Re)/h); this "
                        "is under-resolved for DNS", blCells));
    }
    return std::nullopt;
}

std::optional<ConfigError> flowRotor(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // --- rotor: the body spins ------------------------------------------------
    //
    // rotorMach is the ONE switch. At 0 nothing below runs, c.rotor stays
    // inactive, and every geometry/wall/tally path downstream takes the static
    // arithmetic it always took -- that is what keeps the measured cd/cl
    // anchors valid.
    {
        if (r.rotorMach == 0)
        {
            if (auto e = rejectGiven(r,
                                     {"--rotor-axis", "--rotor-bins",
                                      "--rotor-span", "--blade-thicken",
                                      "--rotor-frame"},
                                     "configures a spinning body; give "
                                     "--rotor-mach to have one");
                e)
                return e;
        }
        else
        {
            if (!comp)
                return err(2, "--rotor-mach", "the moving wall is the "
                               "compressible ghost wall; --rotor-mach needs "
                               "--physics compressible");
            // The O-grid takes the frozen-rotor frame only: the wall is a
            // mesh face row, so MOVING geometry would spin the mesh itself.
            // --rotor-frame mrf keeps the mesh static (rotating-frame
            // equations + a wall moving in its own plane) and is exact for a
            // circular section.
            if (c.ogrid && r.rotorFrame != 1)
                return err(2, "--rotor-mach", "an O-grid IS the body surface: "
                               "spinning the body would spin the mesh. "
                               "--rotor-frame mrf solves the rotating frame "
                               "on the static O-grid instead; the moving-"
                               "geometry rotor needs the immersed box");
            if (c.ogrid && r.rotorAxis != 2)
                return err(2, "--rotor-axis", "the O-grid is z-extruded, so "
                               "only the z shaft keeps the wall motion in the "
                               "mesh plane");
            if (c.ogrid && r.given.contains("--blade-thicken"))
                return err(2, "--blade-thicken", "exists for the immersed "
                               "ghost wall's mirror band; the O-grid wall is "
                               "a mesh face row and needs no dilation");
            if (c.ogrid && r.given.contains("--rotor-span"))
                return err(2, "--rotor-span", "sizes the swept box of an "
                               "immersed rotor; the O-grid's extent is the "
                               "mesh itself");
            if (c.bodyKind == BodyKind::None)
                return err(2, "--rotor-mach", "there is no body to spin");
            if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
                return err(2, "--rotor-mach", "the wall-mounted bodies are "
                               "sealed through the domain floor; they have no "
                               "shaft and cannot turn");
            if (c.cutFlux || c.penalize > 0)
                return err(2, "--rotor-mach", "apertures and the Brinkman band "
                               "are built once from a static body; the moving "
                               "wall is the ghost wall alone");
            if (r.seq > 1)
                return err(2, "--rotor-mach", "a spinning body has no per-"
                               "level restart; drop --seq");
            // The frozen rotor (--rotor-frame mrf) HAS a fixed point -- the
            // steady-in-the-blade-frame flow -- so it takes --steady
            // (flowRans checks the rest); moving geometry does not.
            if (r.steady > 0 && r.rotorFrame != 1)
                return err(2, "--steady", "moving geometry has no steady "
                               "state; --rotor-frame mrf is the frozen rotor");
            if (!finitePos(r.rotorMach))
                return err(2, "--rotor-mach", "must be finite and > 0 (the TIP "
                                              "Mach number)");
            // The KG central flux carries no shock capturing, so the FASTEST
            // material point in the run has to stay subsonic -- that is the
            // advancing tip, not the freestream.
            if (r.rotorMach + c.mach >= 1.0)
                return err(2, "--rotor-mach", "tip Mach + freestream Mach must "
                               "stay under 1: the advancing tip is the fastest "
                               "point in the flow and the flux has no shock "
                               "capturing");
            if (!(finitePos(r.rotorSpan) && r.rotorSpan <= 0.5))
                return err(2, "--rotor-span", "blade half-extent along the "
                               "shaft in diameters, in (0, 0.5]: beyond 0.5 the "
                               "swept volume leaves the placement cube that "
                               "bounds every extent check");
            if (!(std::isfinite(r.bladeThicken) && r.bladeThicken >= 0))
                return err(2, "--blade-thicken", "must be finite and >= 0 "
                                                 "(fine cells of dilation)");
            // A rotor's mesh is placed and sized about its SHAFT, and its own
            // origin IS that shaft (casegeom). Baking an attitude into it would
            // tilt the model out of the axis it spins about, so the two
            // deprecated angles are rejected outright rather than silently
            // producing a wobbling body.
            if (auto e = rejectGiven(
                    r, {"--pitch", "--yaw"},
                    "tilts the body out of the shaft it spins about; a rotor's "
                    "mesh is placed about its own origin. Bake attitude into "
                    "the mesh, or drop --rotor-mach");
                e)
                return e;
            if (c.grid.nz == 1 && r.rotorAxis != 2)
                return err(2, "--rotor-axis", "nz == 1 is the exact 2D "
                               "restriction: only a z shaft keeps the motion in "
                               "the plane. An x or y shaft sweeps the blade out "
                               "of it, which 2D cannot represent");

            // Fine spacing: the plateau's under --focal, the grid's otherwise.
            // It sets both the natural bin count and what a "cell" of
            // --blade-thicken means, so the two always agree.
            const double hFine = dns::hFine(c);

            c.rotor.axis = r.rotorAxis == 0   ? RotorAxis::X
                           : r.rotorAxis == 1 ? RotorAxis::Y
                                              : RotorAxis::Z;
            c.rotor.shaft = {c.placement.cx, c.placement.cy, c.placement.cz};
            c.rotor.radius = 0.5 * c.placement.D;
            // c = uInf / mach: pInf is chosen as rho uInf^2/(gamma M^2), so
            // the freestream sound speed is exactly that. One definition, and
            // it must not be restated anywhere else.
            c.rotor.omega = rotorOmegaFromTipMach(r.rotorMach, c.uInf / c.mach,
                                                  c.rotor.radius);
            c.rotor.thicken = r.bladeThicken * hFine;

            // --rotor-frame mrf: static geometry, rotating-frame zone. The
            // azimuth table does not exist, so sizing it is a contradiction
            // and is rejected by name rather than ignored.
            c.rotorMrf = r.rotorFrame == 1;
            if (c.rotorMrf && r.given.contains("--rotor-bins"))
                return err(2, "--rotor-bins", "--rotor-frame mrf freezes the "
                               "geometry at azimuth 0; there is no azimuth "
                               "table to size");

            const int natural = rotorNaturalBins(c.rotor, hFine);
            if (c.rotorMrf)
                // active() needs bins > 0; one frozen azimuth is exactly what
                // MRF is.
                c.rotor.bins = 1;
            else if (r.given.contains("--rotor-bins"))
            {
                if (r.rotorBins < 8)
                    return err(2, "--rotor-bins", "under 8 bins the body jumps "
                                   "by more than a blade width per step change; "
                                   "0 selects the natural count");
                if (r.rotorBins > 1 << 16)
                    return err(2, "--rotor-bins", "over 65536 bins; the table "
                                   "grows linearly and buys nothing past the "
                                   "natural count");
                c.rotor.bins = r.rotorBins;
                if (r.rotorBins > natural)
                    c.warnings.push_back(std::format(
                        "--rotor-bins {} exceeds the natural count {} "
                        "(2 pi R / h_fine): occupancy cannot change faster "
                        "than that even at the tip, so the extra bins are "
                        "duplicate geometry",
                        r.rotorBins, natural));
                else if (2 * r.rotorBins < natural)
                    c.warnings.push_back(std::format(
                        "--rotor-bins {} is under half the natural count {}: "
                        "the tip crosses more than two cells per bin, so the "
                        "motion is visibly quantized and injects an impulse at "
                        "bins*rev/s",
                        r.rotorBins, natural));
            }
            else
                c.rotor.bins = natural;

            // The dilated body must still be a body in this box: a dilation
            // comparable to the radius turns a blade into a disc.
            if (c.rotor.thicken > 0.25 * c.rotor.radius)
                return err(2, "--blade-thicken", "dilation exceeds a quarter of "
                               "the tip radius; that is a different body, not a "
                               "resolvable one");

            c.rotorTipMach = r.rotorMach;
            c.rotorSpanD = r.rotorSpan;
            c.bladeThickenCells = r.bladeThicken;

            // The blade has to be thick enough for the ghost wall's mirror
            // band on BOTH sides. We cannot measure the mesh here (validation
            // touches no filesystem), so this is the advisory; the run reports
            // the measured deep-solid count, which is the real check. The
            // O-grid wall is a mesh face row -- no mirror band exists there.
            if (!c.ogrid)
            c.warnings.push_back(std::format(
                "rotor: the ghost wall needs a mirror band >= 2 cells thick "
                "per side, so the blade must be >= 4 fine cells thick "
                "(h_fine {:.4g}, D = {:.1f} fine cells). The run reports the "
                "measured deep-solid cell count; 0 means the two blade "
                "surfaces read through each other",
                hFine, c.placement.D / hFine));
        }
    }
    return std::nullopt;
}

} // namespace dns::cfg
