#include "dns/casegeom.hpp"

#include "dns/bodies.hpp"
#include "dns/geometry.hpp"
#include "dns/meshsdf.hpp"
#include "dns/scene.hpp"
#include "dns/trimesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>

namespace dns {

using std::numbers::pi;

std::expected<CaseGeometry, std::string>
buildCaseGeometry(const Case& c, const Scene* scene)
{
    CaseGeometry out;
    const auto& g = c.grid;
    const auto& p = c.placement;
    const double h = c.h;
    out.D = p.D;

    switch (c.bodyKind)
    {
    case BodyKind::Cylinder:
    {
        const CylinderZ cyl{p.cx, p.cy, p.D / 2.0};
        if (c.dimples > 0)
        {
            // Footprint 8h, depth 4h -- near real golf proportions when
            // D ~ 256h and still grid-resolved.
            const double depth = 4.0 * h, rd = pitRadius(8.0 * h, depth);
            const auto q = ringCentres(cyl, c.dimples, rd, depth);
            out.sdf = dimpled(sdfOf(cyl), q, rd, true);
            auto r = sampleCut(g, out.sdf, {8, 8, true});
            if (!r) return std::unexpected(r.error());
            out.cut = std::move(*r);
        }
        else
        {
            out.sdf = sdfOf(cyl);
            out.cut = CutGeometry::make(g, cyl);
        }
        out.zSpans = true;
        break;
    }
    case BodyKind::Sphere:
    {
        const Sphere s{p.cx, p.cy, p.cz, p.D / 2.0};
        if (c.dimples > 0)
        {
            const double depth = 2.5 * h, rd = pitRadius(5.0 * h, depth);
            const auto q = fibonacciCentres(s, c.dimples, rd, depth);
            out.sdf = dimpled(sdfOf(s), q, rd);
            auto r = sampleCut(g, out.sdf, {4, 4});
            if (!r) return std::unexpected(r.error());
            out.cut = std::move(*r);
        }
        else
        {
            out.sdf = sdfOf(s);
            out.cut = CutGeometry::make(g, s);
        }
        break;
    }
    case BodyKind::Cube:
    {
        // An actual cube: the 45-degree rotated square, extruded to its own
        // side length about cz. sceneCube is the one definition of that rule
        // (and of its 2D degeneration back to a prism) -- --scene's cube goes
        // through the same call.
        const RotatedCube cube = sceneCube(g, p.cx, p.cy, p.cz, p.D / 2.0);
        out.sdf = sdfOf(cube);
        out.cut = CutGeometry::make(g, cube);
        break;
    }
    case BodyKind::Lens:
    {
        // aspect = w / hh, with the height fixed at D.
        const double hh = p.D / 2.0, w = c.aspect * hh;
        const Lens lens{p.cx, p.cy, w, hh};
        out.sdf = sdfOf(lens);
        auto r = exactLensCut(g, lens, 64);
        if (!r) return std::unexpected(r.error());
        out.cut = std::move(*r);
        out.D = 2.0 * hh;
        out.zSpans = true;
        break;
    }
    case BodyKind::Fence:
    {
        // Validator pins cy = wallY + D/2, so the blocker spans exactly
        // [wall, wall + D]; thickness D/8. Same sampled-cut budget as the
        // golf disk (z-invariant layer, replicated).
        const Fence f{p.cx, p.D / 16.0, p.cy + p.D / 2.0};
        out.sdf = sdfOf(f);
        auto r = sampleCut(g, out.sdf, {8, 8, true});
        if (!r) return std::unexpected(r.error());
        out.cut = std::move(*r);
        out.zSpans = true;
        break;
    }
    case BodyKind::Floor:
    {
        // Raised floor: solid up to wallY + D (cy pinned at wallY + D/2),
        // optionally rippled (--bumps): amplitude 4h, nodes of the sine
        // pinned at the inlet/outlet planes x = hx and (nx-1) hx so the
        // boundary-crossing height -- and with it the floorRows/floorFrac
        // mass-balance measurement -- stays exactly the flat-plate one.
        Floor f{p.cy + p.D / 2.0};
        if (c.bumps > 0)
        {
            // Triangular trip spikes: 8h tall, 8h base (63-degree flanks),
            // evenly pitched over 10%..75% of the plate. Both flat margins
            // are load-bearing, in PHYSICAL length (fractions of the plate,
            // not cell counts): a spike near the inlet takes the raw uInf
            // with no boundary layer over it (measured 2048^2 Re 10000 tip
            // blowup at 12 cells in -- the 512^2 pilot survived only because
            // 12 of its cells are 4x the physical distance), and a spike
            // near the outlet sheds straight across the convective plane
            // (measured v -> 53 uInf). The margins also keep the boundary
            // cut rows pocket-free and the floorRows/floorFrac mass balance
            // exactly flat-plate.
            f.amp = c.bumpH * h;
            f.halfw = 0.5 * c.bumpH * h;
            f.x0 = g.hx + 0.10 * double(g.nx - 2) * g.hx;
            f.pitch = 0.65 * double(g.nx - 2) * g.hx / c.bumps;
            f.n = c.bumps;
        }
        out.sdf = sdfOf(f);
        auto r = sampleCut(g, out.sdf, {8, 8, true});
        if (!r) return std::unexpected(r.error());
        out.cut = std::move(*r);
        out.zSpans = true;
        break;
    }
    case BodyKind::Mesh:
    {
        auto meshOr = loadMesh(c.meshPath);
        if (!meshOr) return std::unexpected(meshOr.error());
        TriMesh mesh = std::move(*meshOr);
        if (mesh.tris.empty()) return std::unexpected("mesh has no triangles");
        const Aabb bb0 = bounds(mesh);
        const double ext0 = std::max({bb0.hi[0] - bb0.lo[0], bb0.hi[1] - bb0.lo[1],
                                      bb0.hi[2] - bb0.lo[2]});
        if (!(ext0 > 0)) return std::unexpected("degenerate mesh bounds");
        // A ROTOR is placed and sized by its SWEPT circle, not its bounding
        // box, and both halves of that matter:
        //   - the anchor is the mesh's own ORIGIN, which is the shaft. Three
        //     blades put the bounding-box centre 10% of the tip radius off the
        //     axis, and anchoring there would spin the model about a point that
        //     is not its axis (measured: tip radius 0.49 against a nominal
        //     0.3927).
        //   - auto-fit matches the SWEPT diameter to D, so the nominal tip
        //     radius D/2 the validator quotes --rotor-mach on and the real one
        //     agree by construction. Fitting the static extent instead leaves
        //     the tip 1/0.91 outside it.
        const bool spun = c.rotor.active();
        const int shaft = spun ? rotorAxisIndex(c.rotor.axis) : 0;
        const double fit = spun ? sweptDiameter(mesh, shaft) : ext0;
        if (!(fit > 0)) return std::unexpected("degenerate mesh extent");
        const double scale = c.scale > 0 ? c.scale : p.D / fit;
        transform(mesh, scale, c.pitch * pi / 180.0, c.yaw * pi / 180.0,
                  {p.cx, p.cy, p.cz},
                  spun ? MeshAnchor::Origin : MeshAnchor::BoxCenter);
        const Aabb bb = bounds(mesh);
        // A rotor's characteristic length is its swept diameter, which is what
        // CT/CQ are normalised on. It is `scale * fit` exactly: the placement
        // is a scale plus a translation, and the validator rejects
        // --pitch/--yaw with a rotor precisely so no rotation can tilt the
        // model out of the shaft it is meant to spin about.
        out.D = spun ? scale * fit
                     : std::max({bb.hi[0] - bb.lo[0], bb.hi[1] - bb.lo[1],
                                 bb.hi[2] - bb.lo[2]});
        out.bbox = {bb.lo[0], bb.lo[1], bb.lo[2], bb.hi[0], bb.hi[1], bb.hi[2]};
        out.hasBbox = true;

        if (!meshHasArea(mesh))
            return std::unexpected("mesh has no triangle with nonzero area");
        auto sdfp = std::make_shared<MeshSdf>(std::move(mesh), c.robust);
        if (!sdfp->watertight() && !c.robust)
            std::fprintf(stderr, "dns: warning: mesh is not watertight; "
                                 "consider --robust\n");
        out.sdf = [sdfp](double x, double y, double z) { return (*sdfp)(x, y, z); };
        auto r = sampleCut(g, out.sdf, {8, 8});
        if (!r) return std::unexpected(r.error());
        out.cut = std::move(*r);
        break;
    }
    case BodyKind::Scene:
    {
        // The generalized environment: main loaded and parsed the scene in
        // phase 1.5; here it is compiled to one union sdf + sampled cut.
        if (!scene) return std::unexpected("internal: scene not loaded");
        auto sg = buildSceneGeometry(g, p, *scene);
        if (!sg) return std::unexpected(sg.error());
        out.sdf = std::move(sg->sdf);
        out.cut = std::move(sg->cut);
        out.bbox = sg->bbox;
        out.hasBbox = true;
        out.zSpans = scene->zInvariant();
        for (auto& ob : sg->objects)
            out.objectBoxes.emplace_back(std::move(ob.label), ob.bbox);
        break;
    }
    case BodyKind::None:
        // Body-free box (compressible-only, validator-enforced): an SDF that
        // is fluid everywhere makes the ghost wall an exact no-op; the cut
        // geometry stays empty and must not be touched downstream.
        out.sdf = [](double, double, double) { return 1e9; };
        break;
    }

    // A rotor's geometry is the body AT AZIMUTH 0, and `rotorSdf` is the one
    // definition of both the rigid turn (the exact identity at theta = 0, so
    // nothing moves here) and the --blade-thicken dilation. Wrapping it here
    // rather than at the runner is what keeps the sdf the ghost wall reads and
    // the cut qRef measures the SAME body: a dilated sdf against an undilated
    // cut would report a frontal area the solver is not using.
    if (c.rotor.active() && c.bodyKind != BodyKind::None)
    {
        out.sdf = rotorSdf(c.rotor, out.sdf, 0.0);
        if (c.rotor.thicken > 0)
        {
            // Only a real dilation resamples: at thicken == 0 the wrap is
            // arithmetically the identity, so the exact closed-form cuts above
            // stay exactly what they were.
            auto r = sampleCut(g, out.sdf, {8, 8});
            if (!r) return std::unexpected(r.error());
            out.cut = std::move(*r);
            // The bbox is the ghost wall's "provably far from the surface"
            // short-circuit; a dilated body that outgrows it would be silently
            // clipped off its own boundary.
            for (int a = 0; a < 3; ++a)
            {
                out.bbox[a] -= c.rotor.thicken;
                out.bbox[a + 3] += c.rotor.thicken;
            }
        }
    }
    return out;
}

} // namespace dns
