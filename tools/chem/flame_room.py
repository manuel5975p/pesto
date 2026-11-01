#!/usr/bin/env python3
"""Path-traced render of a 3D `dns flame --mode bunsen --nz N` run as a burner
in a small room (Blender Cycles, run INSIDE Blender):

  blender -b --factory-startup --python tools/chem/flame_room.py -- PREFIX
      --out DIR [--first F --last F --stride S] [--res 1920x1080]
      [--samples 256] [--device gpu|cpu] [--tile 2] [--height 0.30] ...

Per frame: the --emit volume omega field (cropped in x: --bands cells at both
ends, or --crop-lo / --crop-hi) becomes sparse VDB grids, omega_ref (the
99.9th percentile over a sample of frames) FIXED so the light flickers with
the burn rate; x (the jet axis) is world up, y across, z the span.

ONE colour (compressible runs, no findex): grid `emission` = max(omega,
0)/omega_ref in turb_render.chemi_rgb's colour. TWO cones (a low-Mach
--equivalence burner: the run wrote `findex`): grids `em_pre` = max(omega,
0) w_p(xi)/ref and `em_dif` = max(omega, 0)(1 - w_p)/ref split by
dns::lmPremixedWeight (the ONE definition, ported in flame_room_lib), each
its own Emission node: the inner premixed cone CH* + rich C2* (blue-green),
the outer diffusion cone the CO2* continuum (violet-blue) at EQUAL radiant
power per unit omega times --dif-gain (a display choice; flame_room_lib's
docstring states the colour basis). Burnt gas is not drawn (CO2/H2O radiate
in the IR): the light is omega > 0 only; no soot (laminar phi 1.3).

Scale: --height (m) = the cropped box height; with --height 0 the run's .json
gives it (the "rim" record's h x the "si" record's deltaD, `dns ...
--si-delta-d MM --si-sl M_PER_S`): REAL scale. --crop-lo auto = the rim's top
face (the low-Mach run has no sink bands: --bands 0). --burner round draws a
steel Bunsen tube: inner/outer radius from the json "rim" (real ID/OD) when
present (else the jet width from the fuel field), its top at the rim, an air
collar with two air holes, a cast base and the gas inlet. The slot burner is
a steel block with the dark slot; z is periodic there, so the box is tiled
--tile times along the span (seamless).

The volume is an emission-only medium (density 0: a premixed flame's gas is
transparent). Cycles does not light-sample volume emission, so the room is
lit by PROXY point lights, hidden from camera and glossy rays: one per
(height band, span tile) at that band's power-weighted centroid, radius its
spread, power its share of the volume's emitted power, colour the
power-weighted mix of the two cones. The volume itself is hidden from diffuse
rays (no double count); camera, glossy and transmission rays see the field.

Writes DIR/vdb/f####.vdb (deleted after use unless --keep-vdb) and
DIR/f####.png; mp4 assembly is the caller's (ffmpeg)."""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from turb_render import chemi_rgb  # noqa: E402
from flame_room_lib import (Run, cone_colours, crop_lo, light_proxies,  # noqa: E402
                            omega_ref, real_scale, split, tube_geometry)

import bpy  # noqa: E402
import openvdb as vdb  # noqa: E402


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="flame_room.py")
    ap.add_argument("prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=-1, help="inclusive; -1 = last frame")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--bands", type=int, default=16, help="sink band cells cropped at both x ends")
    ap.add_argument("--crop-lo", default=None,
                    help="cells cropped at x-lo: an int, or 'auto' = the json rim's top "
                         "face (low-Mach --rim runs) [--bands]")
    ap.add_argument("--crop-hi", type=int, default=None, help="cells cropped at x-hi [--bands]")
    ap.add_argument("--tile", type=int, default=1,
                    help="span copies (z is periodic; > 1 shows the repeat at most angles)")
    ap.add_argument("--tile-mode", choices=("periodic", "mirror"), default="periodic",
                    help="periodic: exact copies (seamless); mirror: every other copy z-reversed "
                         "(continuous, kinked at the seam, hides the repeat)")
    ap.add_argument("--t0", type=float, default=0.0, help="time of frame 0 (the run's --frame-start)")
    ap.add_argument("--height", type=float, default=0.30,
                    help="world height of the cropped box, m; 0 = the run's real scale "
                         "(json rim h x si deltaD)")
    ap.add_argument("--delta-d", type=float, default=0.0,
                    help="with --height 0: the model's delta_D in mm when the json has no "
                         "si record [the json's]")
    ap.add_argument("--emit", type=float, default=60.0, help="volume emission strength at omega_ref")
    ap.add_argument("--cones", choices=("auto", "two", "one"), default="auto",
                    help="two: premixed / diffusion split by the flame index (needs "
                         "<prefix>_findex.bin); one: the single chemiluminescence colour "
                         "[auto: two when findex exists]")
    ap.add_argument("--dif-gain", type=float, default=1.0,
                    help="outer (diffusion) cone emission per unit omega relative to the "
                         "inner at equal radiant power (a DISPLAY choice)")
    ap.add_argument("--light-gain", type=float, default=40.0,
                    help="proxy light power / volume estimate (appearance-tuned, not radiometric)")
    ap.add_argument("--light-bands", type=int, default=6)
    ap.add_argument("--ambient", type=float, default=4.0, help="ceiling fill light, W (0 = flame only)")
    ap.add_argument("--azimuth", type=float, default=25.0, help="camera azimuth from the span axis, deg")
    ap.add_argument("--distance", type=float, default=1.25, help="camera distance to the flame centre, m")
    ap.add_argument("--elevation", type=float, default=8.0, help="camera elevation, deg")
    ap.add_argument("--orbit", type=float, default=0.0,
                    help="camera azimuth sweep over the frames, deg (eased); 0 = fixed")
    ap.add_argument("--orbit-span", type=lambda s: tuple(int(v) for v in s.split(":")),
                    default=None, help="F0:F1 frames the orbit spans [the rendered range]; "
                                       "pass it when resuming with --first")
    ap.add_argument("--target", type=float, default=-1.0,
                    help="camera aim height above the rim, fraction of --height "
                         "[round 0.15, slot 0.45]")
    ap.add_argument("--burner", choices=("slot", "round"), default="slot",
                    help="the run's nozzle: slot block or round Bunsen tube (dns flame "
                         "--nozzle); round needs --tile 1")
    ap.add_argument("--burner-height", type=float, default=0.14,
                    help="round: the tube's rim above the bench, m (a lab Bunsen ~0.14)")
    ap.add_argument("--props", choices=("on", "off"), default="on",
                    help="glassware and a ring stand on the bench")
    ap.add_argument("--lens", type=float, default=40.0, help="focal length, mm")
    ap.add_argument("--fstop", type=float, default=4.0, help="aperture f-number (depth of field)")
    ap.add_argument("--exposure", type=float, default=0.0, help="view transform exposure, stops")
    ap.add_argument("--keep-vdb", action="store_true")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- scene ----

def mat(name, base, rough, metal=0.0, spec=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*base, 1.0)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Specular IOR Level"].default_value = spec
    return m


def box(name, lo, hi, material):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(0.5 * (lo + hi)))
    o = bpy.context.active_object
    o.name = name
    o.scale = tuple(hi - lo)
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(material)
    return o


def cylinder(name, x, y, z0, r, h, material, fill=True, vertices=96):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=r, depth=h,
                                        end_fill_type="NGON" if fill else "NOTHING",
                                        location=(x, y, z0 + h / 2))
    o = bpy.context.active_object
    o.name = name
    bpy.ops.object.shade_smooth()
    o.data.materials.append(material)
    return o


def tube(name, z0, h, ri, ro, material):
    """A hollow tube (inner radius ri, outer ro) standing on z0, height h: an
    open cylinder of radius ro solidified inward by ro - ri."""
    o = cylinder(name, 0, 0, z0, ro, h, material, fill=False, vertices=128)
    sol = o.modifiers.new("wall", "SOLIDIFY")
    sol.thickness = ro - ri
    sol.offset = -1.0  # inward: the outer surface stays at ro
    sol.use_even_offset = True
    return o


def bunsen_tube(top, hb, ri, ro, steel, dark):
    """A lab Bunsen burner on the bench: a cast base (dark enamel), the gas
    inlet (brass barb + hose), the air collar with two air holes, and the
    steel tube of inner radius ri / outer radius ro (the run's real ID/OD)
    whose rim is at top + hb; the bore dark below the rim. Axis = world z."""
    brass = mat("brass", (0.78, 0.6, 0.32), 0.28, metal=1.0)
    cast = mat("enamel", (0.03, 0.035, 0.05), 0.35, spec=0.6)
    chrome = mat("tube_steel", (0.72, 0.72, 0.74), 0.18, metal=1.0)
    rb = max(0.04, 6.0 * ro)
    cylinder("base", 0, 0, top, rb, 0.010, cast)
    bpy.ops.mesh.primitive_cone_add(vertices=96, radius1=0.75 * rb, radius2=1.8 * ro,
                                    depth=0.018, location=(0, 0, top + 0.010 + 0.009))
    bpy.ops.object.shade_smooth()
    bpy.context.active_object.data.materials.append(cast)
    zc = top + 0.028  # the collar (air regulator) sits on the base's boss
    rc = ro + 0.0025
    cylinder("collar", 0, 0, zc, rc, 0.022, steel)
    # Two air holes through the collar (dark discs set into its surface).
    for s in (-1.0, 1.0):
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.0045, depth=0.002,
                                            location=(0.0, s * (rc - 0.0006), zc + 0.011),
                                            rotation=(math.pi / 2, 0, 0))
        bpy.context.active_object.data.materials.append(dark)
    tube("tube", zc, top + hb - zc, ri, ro, chrome)
    cylinder("bore", 0, 0, top + hb - 0.03, 0.999 * ri, 0.028, dark)
    # Gas inlet: a brass barb out of the base boss, the hose along the bench.
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.0045, depth=0.05,
                                        location=(0.035, 0.0, top + 0.016),
                                        rotation=(0, math.pi / 2, 0))
    bpy.ops.object.shade_smooth()
    bpy.context.active_object.data.materials.append(brass)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.0065, depth=0.5,
                                        location=(0.30, 0.0, top + 0.0075),
                                        rotation=(0, math.pi / 2, 0))
    bpy.ops.object.shade_smooth()
    bpy.context.active_object.data.materials.append(mat("hose", (0.5, 0.08, 0.04), 0.5))


def build_room(burner_top, slot_w, burner_y, burner_x, kind="slot", tube_r=None):
    """Room 4.0 x 3.4 x 2.6 m, lab bench, burner whose rim is at z =
    burner_top, centred at the origin: a slot burner (x = span, y = across
    the slot) or a round Bunsen tube (kind round; tube_r = (ri, ro), else
    the jet diameter slot_w with a 1.5 mm wall)."""
    wall = mat("plaster", (0.62, 0.60, 0.57), 0.9, spec=0.3)
    floor = mat("concrete", (0.18, 0.18, 0.19), 0.55)
    bench = mat("epoxy", (0.035, 0.037, 0.04), 0.22)
    steel = mat("steel", (0.62, 0.62, 0.64), 0.32, metal=1.0)
    dark = mat("slot", (0.01, 0.01, 0.01), 0.6)
    leg = mat("legs", (0.25, 0.26, 0.28), 0.4, metal=1.0)
    t = 0.05
    W, D, H = 4.0, 3.4, 2.6
    cy = 0.3  # the bench sits off-centre toward the back wall; the camera stays inside
    box("floor", (-W / 2, -D / 2 + cy, -t), (W / 2, D / 2 + cy, 0), floor)
    box("ceiling", (-W / 2, -D / 2 + cy, H), (W / 2, D / 2 + cy, H + t), wall)
    box("wall_back", (-W / 2, D / 2 + cy, 0), (W / 2, D / 2 + cy + t, H), wall)
    box("wall_front", (-W / 2, -D / 2 + cy - t, 0), (W / 2, -D / 2 + cy, H), wall)
    box("wall_left", (-W / 2 - t, -D / 2 + cy, 0), (-W / 2, D / 2 + cy, H), wall)
    box("wall_right", (W / 2, -D / 2 + cy, 0), (W / 2 + t, D / 2 + cy, H), wall)
    top = 0.90
    box("bench_top", (-0.9, -0.42, top - 0.04), (0.9, 0.42, top), bench)
    for sx in (-0.85, 0.85):
        for sy in (-0.37, 0.37):
            box("bench_leg", (sx - 0.025, sy - 0.025, 0), (sx + 0.025, sy + 0.025, top - 0.04), leg)
    hb = burner_top - top
    if kind == "round":
        ri, ro = tube_r if tube_r else (0.5 * slot_w, 0.5 * slot_w + 0.0015)
        bunsen_tube(top, hb, ri, ro, steel, dark)
        return top
    # Burner: steel body, a plenum below, the dark slot recessed in its top.
    box("burner_plenum", (-burner_x / 2 - 0.02, -burner_y / 2 - 0.02, top),
        (burner_x / 2 + 0.02, burner_y / 2 + 0.02, top + 0.35 * hb), steel)
    box("burner_body", (-burner_x / 2, -burner_y / 2, top + 0.35 * hb),
        (burner_x / 2, burner_y / 2, burner_top - 0.002), steel)
    box("burner_slot", (-burner_x / 2 + 0.002, -slot_w / 2, burner_top - 0.004),
        (burner_x / 2 - 0.002, slot_w / 2, burner_top), dark)
    # Gas feed: a hose run along the bench from the plenum.
    bpy.ops.mesh.primitive_cylinder_add(radius=0.008, depth=0.5,
                                        location=(burner_x / 2 + 0.25, 0.0, top + 0.01),
                                        rotation=(0, math.pi / 2, 0))
    bpy.context.active_object.data.materials.append(mat("hose", (0.5, 0.08, 0.04), 0.5))
    return top


def glass():
    m = mat("glass", (0.95, 0.97, 1.0), 0.02)
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Transmission Weight"].default_value = 1.0
    p.inputs["IOR"].default_value = 1.47
    return m


def beaker(name, x, y, z0, r, h, material):
    """An open glass cylinder (2 mm wall) with a bottom."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=r, depth=h, end_fill_type="NOTHING",
                                        location=(x, y, z0 + h / 2))
    o = bpy.context.active_object
    o.name = name
    sol = o.modifiers.new("wall", "SOLIDIFY")
    sol.thickness = 0.002
    bpy.ops.object.shade_smooth()
    o.data.materials.append(material)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=r, depth=0.003,
                                        location=(x, y, z0 + 0.0015))
    bpy.context.active_object.data.materials.append(material)


def build_props(top, burner_x, burner_y):
    """Bench clutter that catches the flame light: two beakers (refraction,
    reflections) and a ring stand behind the burner."""
    g = glass()
    steel = bpy.data.materials["steel"]
    beaker("beaker_a", burner_x / 2 + 0.14, -burner_y / 2 - 0.10, top, 0.035, 0.10, g)
    beaker("beaker_b", burner_x / 2 + 0.28, -burner_y / 2 + 0.02, top, 0.028, 0.075, g)
    bx, by = burner_x / 2 + 0.16, burner_y / 2 + 0.12
    box("stand_base", (bx - 0.07, by - 0.11, top), (bx + 0.07, by + 0.11, top + 0.012), steel)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.006, depth=0.55,
                                        location=(bx, by + 0.06, top + 0.012 + 0.275))
    bpy.ops.object.shade_smooth()
    bpy.context.active_object.data.materials.append(steel)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.045, minor_radius=0.004,
                                     location=(bx - 0.05, by + 0.06, top + 0.40))
    bpy.ops.object.shade_smooth()
    bpy.context.active_object.data.materials.append(steel)


def flame_material(channels):
    """An emission-only volume: one Emission node per (grid name, rgb,
    strength), summed."""
    m = bpy.data.materials.new("chemiluminescence")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    acc = None
    for name, rgb, strength in channels:
        em = nt.nodes.new("ShaderNodeEmission")
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_name = name
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = strength
        em.inputs["Color"].default_value = (*rgb, 1.0)
        nt.links.new(at.outputs["Fac"], mul.inputs[0])
        nt.links.new(mul.outputs[0], em.inputs["Strength"])
        if acc is None:
            acc = em.outputs["Emission"]
        else:
            add = nt.nodes.new("ShaderNodeAddShader")
            nt.links.new(acc, add.inputs[0])
            nt.links.new(em.outputs["Emission"], add.inputs[1])
            acc = add.outputs["Shader"]
    nt.links.new(acc, out.inputs["Volume"])
    return m


def setup_render(a, w, h):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    cy = sc.cycles
    cy.samples = a.samples
    cy.use_adaptive_sampling = True
    cy.use_denoising = True
    cy.max_bounces = 6
    cy.volume_bounces = 0
    cy.transparent_max_bounces = 4
    cy.volume_step_rate = 1.0
    cy.seed = 1  # deterministic noise; the denoiser removes it per frame
    cy.caustics_reflective = False  # point-light caustics through glass flicker frame to frame
    cy.caustics_refractive = False
    if a.device == "gpu":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "HIP"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "HIP"
        if not any(d.use for d in prefs.devices):
            sys.exit("flame_room: --device gpu but Cycles found no HIP device")
        cy.device = "GPU"
        cy.denoiser = "OPENIMAGEDENOISE"
        cy.denoising_use_gpu = True
    else:
        cy.device = "CPU"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = a.exposure
    sc.world = bpy.data.worlds.new("dark")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0


def camera(a):
    cam = bpy.data.cameras.new("cam")
    cam.lens = a.lens
    cam.dof.use_dof = True
    cam.dof.focus_distance = a.distance
    cam.dof.aperture_fstop = a.fstop
    o = bpy.data.objects.new("cam", cam)
    bpy.context.scene.collection.objects.link(o)
    bpy.context.scene.camera = o
    return o


def place_camera(o, a, target, azimuth):
    """The span runs along world x; azimuth 0 looks down the span (the cone
    profile), 90 faces the sheet. Camera on the -y side."""
    az, el = math.radians(azimuth), math.radians(a.elevation)
    d = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.asarray(target) + a.distance * d
    o.location = tuple(pos)
    o.rotation_euler = vec_to_euler(np.asarray(target) - pos)


def azimuth_at(a, f, first, last):
    """Camera azimuth of frame f: --azimuth, swept by --orbit over [first, last]
    with a smoothstep ease. A function of the frame index only, so a resumed
    render (--first) continues the same path when given the same --orbit-span."""
    lo, hi = (a.orbit_span or (first, last))
    q = 0.0 if hi <= lo else min(max((f - lo) / (hi - lo), 0.0), 1.0)
    return a.azimuth + a.orbit * q * q * (3 - 2 * q)


def vec_to_euler(fwd):
    from mathutils import Vector
    return Vector(tuple(fwd)).to_track_quat("-Z", "Y").to_euler()


# ---------------------------------------------------------------- frames ---

def write_vdb(path, grids, voxel):
    """grids {name: array indexed (x_world, y_world, z_world) = (span,
    across, up)}."""
    out = []
    for name, em in grids.items():
        g = vdb.FloatGrid()
        g.copyFromArray(em, tolerance=1e-4)
        g.name = name
        g.transform = vdb.createLinearTransform(voxelSize=voxel)
        g.background = 0.0
        out.append(g)
    vdb.write(path, grids=out)


def main():
    a = parse()
    run = Run(a.prefix)
    last = run.frames - 1 if a.last < 0 else min(a.last, run.frames - 1)
    frames = list(range(a.first, last + 1, a.stride))
    if not frames:
        sys.exit(f"flame_room: no frames in [{a.first}, {last}] ({run.frames} written)")
    two = a.cones == "two" or (a.cones == "auto" and run.has("findex"))
    if two and not run.has("findex"):
        sys.exit("flame_room: --cones two needs <prefix>_findex.bin (a low-Mach "
                 "--equivalence burner)")
    w, h = (int(v) for v in a.res.split("x"))
    x0 = crop_lo(run, a.crop_lo) if a.crop_lo is not None else a.bands
    x1 = run.nx - (a.crop_hi if a.crop_hi is not None else a.bands)
    X = x1 - x0
    voxel = real_scale(run, a.height, X, 1e-3 * a.delta_d)
    if voxel is None:
        sys.exit("flame_room: --height 0 needs the run's json rim record and its si "
                 "record (dns ... --rim ... --si-delta-d MM --si-sl M_PER_S) or "
                 "--delta-d MM")
    height = voxel * X
    span = a.tile * run.nz * voxel
    across = run.ny * voxel
    os.makedirs(os.path.join(a.out, "vdb"), exist_ok=True)
    print(f"flame_room: crop x [{x0}, {x1}) = {X} cells, voxel {voxel * 1e3:.4g} mm, "
          f"height {height * 1e3:.4g} mm, {'two cones' if two else 'one colour'}", flush=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_render(a, w, h)
    tube_r = tube_geometry(run, voxel) if a.burner == "round" else None
    # The slot width: the jet's fuel-fed inflow cells, unless the json's tube.
    slot_w = slot_width(run) * voxel
    burner_top = 0.90 + (a.burner_height if a.burner == "round" else 0.12)
    top = build_room(burner_top, slot_w, across, span, a.burner, tube_r)
    if a.props == "on":
        build_props(top, span, across)
    origin = np.array([-span / 2, -across / 2, burner_top])
    if two:
        ci, co = cone_colours()
        chans = [("em_pre", ci, a.emit), ("em_dif", co, a.emit * a.dif_gain)]
        print(f"flame_room: inner {np.round(ci, 3)}, outer {np.round(co, 3)} x dif-gain "
              f"{a.dif_gain:g}", flush=True)
    else:
        rgb = chemi_rgb()
        chans = [("emission", tuple(float(c) for c in rgb / rgb.max()), a.emit)]
    fm = flame_material(chans)

    # Emission strength S per unit volume radiates S * 4 pi per steradian-
    # integrated unit (Cycles volume emission is isotropic radiance density).
    power_per_unit = a.emit * 4.0 * math.pi * a.light_gain
    ref = omega_ref(run, frames, x0, x1)
    # Round: aim lower by default, the tube and its base belong in the frame.
    frac = a.target if a.target >= 0 else (0.15 if a.burner == "round" else 0.45)
    target = (0.0, 0.0, burner_top + frac * height)
    cam = camera(a)
    if a.ambient > 0:
        ld = bpy.data.lights.new("fill", "AREA")
        ld.energy = a.ambient
        ld.size = 0.8
        ld.color = (1.0, 0.92, 0.8)
        lo = bpy.data.objects.new("fill", ld)
        lo.location = (0.0, 0.9, 2.55)
        bpy.context.scene.collection.objects.link(lo)

    vol_obj, proxies = None, []
    for f in frames:
        om = run.omega(f)[:, :, x0:x1]  # (nz, ny, X)
        if two:
            pre, dif = split(om, run.field("findex", f)[:, :, x0:x1], ref)
            planes = {"em_pre": pre, "em_dif": dif}
        else:
            planes = {"emission": (np.maximum(om, 0) / ref).astype(np.float32)}
        grids = {}
        for name, em in planes.items():
            copies = [em if c % 2 == 0 or a.tile_mode == "periodic" else em[::-1]
                      for c in range(a.tile)]
            grids[name] = np.ascontiguousarray(np.concatenate(copies, 0))  # (span, across, up)
        path = os.path.join(a.out, "vdb", f"f{f:04d}.vdb")
        write_vdb(path, grids, voxel)
        vd = bpy.data.volumes.new(f"flame{f:04d}")  # a fresh datablock per frame: no grid cache
        vd.filepath = path
        vd.materials.append(fm)
        vd.render.space = "WORLD"
        vd.render.step_size = 0.5 * voxel
        if vol_obj is None:
            vol_obj = bpy.data.objects.new("flame", vd)
            bpy.context.scene.collection.objects.link(vol_obj)
            vol_obj.location = tuple(origin)
            vol_obj.visible_diffuse = False
            vol_obj.visible_shadow = False
        else:
            old = vol_obj.data
            vol_obj.data = vd
            bpy.data.volumes.remove(old)
        for o in proxies:
            bpy.data.objects.remove(o, do_unlink=True)
        proxies = []
        gains = [(s / a.emit, rgb) for _, rgb, s in chans]
        for i, (loc, rad, pw, col) in enumerate(
                light_proxies(list(grids.values()), gains, voxel, origin, a.light_bands,
                              a.tile, power_per_unit)):
            ld = bpy.data.lights.new(f"flame{i}", "POINT")
            ld.energy = pw
            ld.shadow_soft_size = rad
            ld.color = col
            lo = bpy.data.objects.new(f"flame{i}", ld)
            lo.location = tuple(loc)
            lo.visible_camera = False
            lo.visible_glossy = False
            lo.visible_transmission = False
            bpy.context.scene.collection.objects.link(lo)
            proxies.append(lo)
        place_camera(cam, a, target, azimuth_at(a, f, frames[0], frames[-1]))
        bpy.context.scene.render.filepath = os.path.join(a.out, f"f{f:04d}.png")
        bpy.ops.render.render(write_still=True)
        tot = sum(o.data.energy for o in proxies)
        print(f"flame_room: frame {f} t={a.t0 + f * run.fdt:.3f} lights {len(proxies)} "
              f"power {tot:.3g} W ref {ref:.4g}", flush=True)
        if not a.keep_vdb:
            os.remove(path)


def slot_width(run):
    """Slot width in cells: the fuel-fed run of the inflow column (mid-span),
    read off the first frame's fuel field at x = 0 (Y > 0.5 Y_max)."""
    if not run.has("fuel"):
        return 24 * 2.5
    a = run.field("fuel", 0)[run.nz // 2, :, 0]
    return float((a > 0.5 * a.max()).sum()) if a.max() > 0 else 24 * 2.5


if __name__ == "__main__":
    main()
