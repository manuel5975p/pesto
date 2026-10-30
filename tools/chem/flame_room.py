#!/usr/bin/env python3
"""Path-traced render of a 3D `dns flame --mode bunsen --nz N` run as a slot
burner in a small room (Blender Cycles, run INSIDE Blender):

  blender -b --factory-startup --python tools/chem/flame_room.py -- PREFIX
      --out DIR [--first F --last F --stride S] [--res 1920x1080]
      [--samples 256] [--device gpu|cpu] [--tile 2] [--height 0.30] ...

Per frame: the --emit volume omega field (sink bands cropped in x) becomes a
sparse VDB grid `emission` = max(omega, 0)/omega_ref (omega_ref: the 99.9th
percentile over a sample of frames, FIXED, so the light flickers with the
burn rate); x (the jet axis) is world up, y across the slot, z the span. z is
periodic, so the box is tiled --tile times along the span (seamless) and the
burner slot ends where the volume ends.

The volume is an emission-only medium (density 0: a premixed flame's gas is
transparent) in the chemiluminescence colour of turb_render.chemi_rgb (CH*,
C2*, CO2* continuum). Cycles does not light-sample volume emission, so the
room is lit by PROXY point lights, hidden from camera and glossy rays: one per
(height band, span tile) at that band's omega-weighted centroid, radius its
spread, power its share of the volume's emitted power. The volume itself is
hidden from diffuse rays (no double count); camera, glossy and transmission
rays see the real field. Burnt gas is not drawn (CO2/H2O emit in the IR).

Writes DIR/vdb/f####.vdb (deleted after use unless --keep-vdb) and
DIR/f####.png; mp4 assembly is the caller's (ffmpeg)."""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from turb_render import chemi_rgb  # noqa: E402

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
    ap.add_argument("--tile", type=int, default=1,
                    help="span copies (z is periodic; > 1 shows the repeat at most angles)")
    ap.add_argument("--tile-mode", choices=("periodic", "mirror"), default="periodic",
                    help="periodic: exact copies (seamless); mirror: every other copy z-reversed "
                         "(continuous, kinked at the seam, hides the repeat)")
    ap.add_argument("--t0", type=float, default=0.0, help="time of frame 0 (the run's --frame-start)")
    ap.add_argument("--height", type=float, default=0.30, help="world height of the cropped box, m")
    ap.add_argument("--emit", type=float, default=60.0, help="volume emission strength at omega_ref")
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
    ap.add_argument("--props", choices=("on", "off"), default="on",
                    help="glassware and a ring stand on the bench")
    ap.add_argument("--lens", type=float, default=40.0, help="focal length, mm")
    ap.add_argument("--exposure", type=float, default=0.0, help="view transform exposure, stops")
    ap.add_argument("--keep-vdb", action="store_true")
    return ap.parse_args(argv)


class Run:
    """The --emit volume files of one dns run (nz, ny, nx float32 per frame)."""

    def __init__(self, prefix):
        self.prefix = prefix
        m = json.load(open(prefix + ".json"))
        self.nx, self.ny, self.nz, self.frames = m["nx"], m["ny"], m["nz"], m["frames"]
        self.fdt = m.get("frameDt", 1.0)
        self.n3 = self.nx * self.ny * self.nz
        size = os.path.getsize(prefix + "_omega.bin")
        on_disk = size // (4 * self.n3)  # a run still writing has frames 0 in its json
        self.frames = min(self.frames, on_disk) if self.frames > 0 else on_disk

    def omega(self, f):
        a = np.fromfile(self.prefix + "_omega.bin", "<f4", self.n3, offset=4 * self.n3 * f)
        return a.reshape(self.nz, self.ny, self.nx)


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


def cylinder(name, x, y, z0, r, h, material, fill=True):
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=r, depth=h,
                                        end_fill_type="NGON" if fill else "NOTHING",
                                        location=(x, y, z0 + h / 2))
    o = bpy.context.active_object
    o.name = name
    bpy.ops.object.shade_smooth()
    o.data.materials.append(material)
    return o


def bunsen_tube(top, hb, d, steel, dark):
    """A Bunsen burner on the bench: base disc, gas inlet, tube of inner
    diameter d (the run's nozzle) whose rim is at top + hb, an air collar,
    the dark bore at the rim. The axis is the world z axis through 0."""
    brass = mat("brass", (0.78, 0.6, 0.32), 0.28, metal=1.0)
    rt = 0.5 * d + 0.0015
    cylinder("base", 0, 0, top, max(0.045, 2.2 * rt), 0.014, steel)
    cylinder("base_boss", 0, 0, top + 0.014, 1.5 * rt, 0.02, steel)
    tube = cylinder("tube", 0, 0, top + 0.034, rt, hb - 0.034, brass, fill=False)
    sol = tube.modifiers.new("wall", "SOLIDIFY")
    sol.thickness = 0.0015
    cylinder("collar", 0, 0, top + 0.040, rt + 0.0035, 0.018, steel)
    cylinder("bore", 0, 0, top + hb - 0.02, 0.5 * d, 0.019, dark)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.005, depth=0.06,
                                        location=(0.045, 0.0, top + 0.024),
                                        rotation=(0, math.pi / 2, 0))
    bpy.ops.object.shade_smooth()
    bpy.context.active_object.data.materials.append(brass)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.0065, depth=0.5,
                                        location=(0.32, 0.0, top + 0.0075),
                                        rotation=(0, math.pi / 2, 0))
    bpy.ops.object.shade_smooth()
    bpy.context.active_object.data.materials.append(mat("hose", (0.5, 0.08, 0.04), 0.5))


def build_room(burner_top, slot_w, burner_y, burner_x, kind="slot"):
    """Room 4.0 x 3.4 x 2.6 m, lab bench, burner whose rim is at z =
    burner_top, centred at the origin: a slot burner (x = span, y = across
    the slot) or a round Bunsen tube of diameter slot_w (kind round)."""
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
        bunsen_tube(top, hb, slot_w, steel, dark)
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


def flame_material(rgb, strength):
    m = bpy.data.materials.new("chemiluminescence")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "emission"
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = strength
    em.inputs["Color"].default_value = (*rgb, 1.0)
    nt.links.new(at.outputs["Fac"], mul.inputs[0])
    nt.links.new(mul.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs["Emission"], out.inputs["Volume"])
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
    cam.dof.aperture_fstop = 4.0
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

def omega_ref(run, frames, x0, x1):
    pick = frames[:: max(1, len(frames) // 8)]
    vals = [np.percentile(np.maximum(run.omega(f)[:, :, x0:x1], 0), 99.9) for f in pick]
    return float(np.median(vals))


def write_vdb(path, em, voxel, origin):
    """em indexed (x_world, y_world, z_world) = (span, across, up)."""
    g = vdb.FloatGrid()
    g.copyFromArray(em, tolerance=1e-4)
    g.name = "emission"
    g.transform = vdb.createLinearTransform(voxelSize=voxel)
    g.background = 0.0
    vdb.write(path, grids=[g])
    return origin


def light_proxies(em, voxel, origin, nband, tile_n, power_per_unit):
    """[(location, radius, power W)] per (height band, span tile)."""
    nzt, ny, nx = em.shape[0] // tile_n, em.shape[1], em.shape[2]
    out = []
    edges = np.linspace(0, nx, nband + 1).astype(int)
    xs = (np.arange(nzt) + 0.5) * voxel
    ys = (np.arange(ny) + 0.5) * voxel
    zs = (np.arange(nx) + 0.5) * voxel
    for t in range(tile_n):
        blk = em[t * nzt:(t + 1) * nzt]
        for b in range(nband):
            w = blk[:, :, edges[b]:edges[b + 1]]
            s = float(w.sum())
            if s <= 0:
                continue
            px = (w.sum((1, 2)) @ xs) / s
            py = (w.sum((0, 2)) @ ys) / s
            pz = (w.sum((0, 1)) @ zs[edges[b]:edges[b + 1]]) / s
            sy = math.sqrt(max((w.sum((0, 2)) @ ys ** 2) / s - py ** 2, 0.0))
            loc = origin + np.array([px + t * nzt * voxel, py, pz])
            out.append((loc, max(sy, 2 * voxel), power_per_unit * s * voxel ** 3))
    return out


def main():
    a = parse()
    run = Run(a.prefix)
    last = run.frames - 1 if a.last < 0 else min(a.last, run.frames - 1)
    frames = list(range(a.first, last + 1, a.stride))
    if not frames:
        sys.exit(f"flame_room: no frames in [{a.first}, {last}] ({run.frames} written)")
    w, h = (int(v) for v in a.res.split("x"))
    x0, x1 = a.bands, run.nx - a.bands
    X = x1 - x0
    voxel = a.height / X
    span = a.tile * run.nz * voxel
    across = run.ny * voxel
    os.makedirs(os.path.join(a.out, "vdb"), exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_render(a, w, h)
    # The slot width from the run's inflow: the burner geometry is W = 24 by
    # default; read the json if the runner recorded it, else the jet's |Y| row.
    slot_w = slot_width(run) * voxel
    burner_top = 0.90 + (0.14 if a.burner == "round" else 0.12)
    top = build_room(burner_top, slot_w, across, span, a.burner)
    if a.props == "on":
        build_props(top, span, across)
    origin = np.array([-span / 2, -across / 2, burner_top])
    rgb = chemi_rgb()
    rgb = tuple(float(c) for c in rgb / rgb.max())
    fm = flame_material(rgb, a.emit)

    # Emission strength S per unit volume radiates S * 4 pi per steradian-
    # integrated unit (Cycles volume emission is isotropic radiance density).
    power_per_unit = a.emit * 4.0 * math.pi * a.light_gain
    ref = omega_ref(run, frames, x0, x1)
    # Round: aim lower by default, the tube and its base belong in the frame.
    frac = a.target if a.target >= 0 else (0.15 if a.burner == "round" else 0.45)
    target = (0.0, 0.0, burner_top + frac * a.height)
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
        em = np.maximum(om, 0) / ref
        copies = [em if c % 2 == 0 or a.tile_mode == "periodic" else em[::-1] for c in range(a.tile)]
        em = np.ascontiguousarray(np.concatenate(copies, 0))  # (span, across, up)
        path = os.path.join(a.out, "vdb", f"f{f:04d}.vdb")
        write_vdb(path, em.astype(np.float32), voxel, origin)
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
        for i, (loc, rad, pw) in enumerate(light_proxies(em, voxel, origin, a.light_bands,
                                                           a.tile, power_per_unit)):
            ld = bpy.data.lights.new(f"flame{i}", "POINT")
            ld.energy = pw
            ld.shadow_soft_size = rad
            ld.color = rgb
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
    read off the first frame's fuel field at x = 0 (Y > 0.5)."""
    fn = run.prefix + "_fuel.bin"
    if not os.path.exists(fn):
        return 24 * 2.5
    a = np.fromfile(fn, "<f4", run.n3).reshape(run.nz, run.ny, run.nx)
    return float((a[run.nz // 2, :, 0] > 0.5).sum())


if __name__ == "__main__":
    main()
