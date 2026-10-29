#!/usr/bin/env python3
"""Volume-render a 3D `dns flame --mode bunsen --nz N` run into an mp4.

Inputs (docs/chem.md, "3D turbulent slot Bunsen"; docs/viewer-format.md):
<prefix>.json (nx, ny, nz), the --emit volume files <prefix>_omega.bin,
<prefix>_fuel.bin, <prefix>_vort.bin (raw little-endian float32, one nx*ny*nz
volume per frame, i fastest, k slowest) and the --emit slices mid-span planes
<prefix>_xy_vort.bin, <prefix>_xy_speed.bin (nx*ny per frame).

Left panel -- the rendered LIGHT, an emission-absorption ray march:
  * chemiluminescence j = omega (the resolved heat release; with TFLES the
    line-of-sight integral of (E/F) omega is rho_u S_c per unit area, so the
    picture integrates consistently), OPTICALLY THIN, coloured by a line
    spectrum through the CIE 1931 colour-matching functions (Wyman, Sloan &
    Shirley 2013 multi-lobe fit) -> XYZ -> linear sRGB: CH* 431.4 nm (1.0),
    C2* Swan 516.5 / 473.7 nm (0.6 / 0.3), a weak CO2* continuum 350-600 nm
    (0.2 in total);
  * a CONTEXT overlay that is NOT light: a faint steel-blue fog of density
    |curl u| Y_F (the cold jet's vorticity), with a small artificial
    absorption for depth (--context off removes it).
  Camera: orthographic, rays in (y, z) planes rotated about the jet axis x
  by theta(t) (a slow orbit --theta0 -> --theta1 over the clip); one ray
  covers ONE span period of z (periodic, bilinear), y is bounded (zero
  outside the box): a tile of an infinite slot burner. 2 samples per cell
  along the ray and across the image.
Middle / right: |curl u| (sqrt-mapped) and |u| on the mid-span xy plane
(inferno-like / viridis-like LUTs, fixed scales).

Fixed exposure per clip (the light's 99.5th percentile over the last 30
frames maps to --exposure), FFT Gaussian bloom, ACES filmic curve, sRGB OETF, 1920 x 1080,
libx264 yuv420p. numpy + ffmpeg only.

    turb_render.py PREFIX --out X.mp4 --png X.png [--workers 8] [--fps 30]
                   [--crf 20] [--bands 16] [--theta0 15 --theta1 40]
                   [--titles "A;B;C"] [--footer TEXT] [--exposure 0.6]
                   [--context vort|off]
    turb_render.py --selftest
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np

FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"

# --- colour --------------------------------------------------------------------


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_xyz(lam):
    """CIE 1931 2-degree CMFs, multi-lobe Gaussian fit (Wyman et al. 2013)."""
    x = (1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7)
         - 0.065 * _g(lam, 501.1, 20.4, 26.2))
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986],
                    [-0.9689, 1.8758, 0.0415],
                    [0.0557, -0.2040, 1.0570]])


def chemi_rgb():
    """Linear sRGB of the premixed-flame emission spectrum, unit luminance:
    CH* 431.4 nm (1.0), C2* 516.5 / 473.7 nm (0.6 / 0.3), CO2* continuum flat
    over 350-600 nm carrying 0.2 in total. The CH* line lies outside the
    sRGB gamut (negative red): mapped in by adding equal-energy white until
    the smallest channel is 0 (desaturation, hue kept), then renormalised to
    luminance 1."""
    lines = np.array([431.4, 516.5, 473.7])
    w = np.array([1.0, 0.6, 0.3])
    xyz = (cie_xyz(lines) * w[:, None]).sum(0)
    lam = np.arange(350.0, 600.0 + 1e-9, 1.0)
    xyz = xyz + 0.2 * cie_xyz(lam).mean(0)
    rgb = XYZ2RGB @ xyz
    rgb = rgb - min(0.0, rgb.min())
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    return rgb / lum


FOG_RGB = np.array([0.27, 0.42, 0.62])  # steel blue (context, not light)

# Colormap control points (position, r, g, b), perceptual-ish dark LUTs.
INFERNO = np.array([
    [0.00, 0.00, 0.00, 0.02], [0.13, 0.08, 0.04, 0.22], [0.26, 0.26, 0.04, 0.41],
    [0.39, 0.42, 0.09, 0.43], [0.52, 0.58, 0.15, 0.40], [0.65, 0.74, 0.21, 0.33],
    [0.78, 0.89, 0.35, 0.20], [0.90, 0.98, 0.58, 0.04], [1.00, 0.99, 0.95, 0.64]])
VIRIDIS = np.array([
    [0.00, 0.27, 0.00, 0.33], [0.13, 0.28, 0.14, 0.46], [0.25, 0.25, 0.27, 0.53],
    [0.38, 0.21, 0.37, 0.55], [0.50, 0.16, 0.47, 0.56], [0.63, 0.12, 0.57, 0.55],
    [0.75, 0.21, 0.67, 0.50], [0.88, 0.48, 0.82, 0.32], [1.00, 0.99, 0.91, 0.14]])


def lut(v, cp):
    v = np.clip(v, 0.0, 1.0)
    out = np.empty(v.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, cp[:, 0], cp[:, c + 1])
    return out


def aces(x):
    """ACES filmic tone curve (Narkowicz 2015 fit)."""
    x = np.maximum(x, 0)
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def gauss_fft(img, sigma):
    """Gaussian blur of (H, W, C) by FFT, zero padded (no wrap)."""
    H, W = img.shape[:2]
    P, Q = 2 * H, 2 * W
    fy = np.fft.fftfreq(P)[:, None]
    fx = np.fft.rfftfreq(Q)[None, :]
    G = np.exp(-2 * (np.pi * sigma) ** 2 * (fy ** 2 + fx ** 2))
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        p = np.zeros((P, Q), np.float32)
        p[:H, :W] = img[..., c]
        out[..., c] = np.fft.irfft2(np.fft.rfft2(p) * G, s=(P, Q))[:H, :W]
    return out


def resample(a, H, W):
    """Bilinear resample of (h, w[, C]) onto (H, W), cell centres kept."""
    h, w = a.shape[:2]
    yi = (np.arange(H) + 0.5) * h / H - 0.5
    xi = (np.arange(W) + 0.5) * w / W - 0.5
    y0 = np.clip(np.floor(yi).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xi).astype(int), 0, w - 1)
    y1, x1 = np.clip(y0 + 1, 0, h - 1), np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(yi - y0, 0, 1).astype(np.float32)
    fx = np.clip(xi - x0, 0, 1).astype(np.float32)
    if a.ndim == 3:
        fy, fx = fy[:, None, None], fx[None, :, None]
    else:
        fy, fx = fy[:, None], fx[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


# --- the ray march ---------------------------------------------------------------


def ray_tables(ny, nz, theta, uhalf, yc, zc, sub=2):
    """Bilinear (y, z) gather tables of an orthographic camera rotated by theta
    about x: image column u in [-uhalf, uhalf) cells, `sub` samples per cell
    on both u and the ray parameter s.

    The ray is windowed so that every point of the z-periodic field is seen
    with total weight exactly 1 at ANY theta, with no hard seam: along s the
    periodic images of a point are spaced P = nz cos(theta) apart, and the Hann
    window w(s) = cos^2(pi s / (2P)), |s| < P, is a partition of unity over
    that spacing (cos^2 + sin^2 = 1). Returns flat indices (4, Nu*Ns) into a
    (ny*nz) plane, bilinear weights (4, Nu*Ns) with the window folded in, Nu,
    Ns, ds."""
    du = 1.0 / sub
    u = -uhalf + (np.arange(int(round(2 * uhalf * sub))) + 0.5) * du
    ds = 1.0 / sub
    P = nz * np.cos(theta)
    ns = int(np.ceil(2 * P * sub))
    s = -0.5 * ns * ds + (np.arange(ns) + 0.5) * ds
    win = np.where(np.abs(s) < P, np.cos(0.5 * np.pi * s / P) ** 2, 0.0)
    U, S = np.meshgrid(u, s, indexing="ij")
    Wn = np.broadcast_to(win[None, :], U.shape)
    c, sn = np.cos(theta), np.sin(theta)
    # Cell-centre coordinates: y = j + 1/2 at index j.
    y = yc + U * c - S * sn - 0.5
    z = zc + U * sn + S * c - 0.5
    j0 = np.floor(y).astype(np.int64)
    k0 = np.floor(z).astype(np.int64)
    fy = (y - j0).astype(np.float32)
    fz = (z - k0).astype(np.float32)
    idx, wt = [], []
    for dj, wj in ((0, 1 - fy), (1, fy)):
        jj = j0 + dj
        ok = (jj >= 0) & (jj < ny)  # y bounded: zero outside the box
        for dk, wk in ((0, 1 - fz), (1, fz)):
            kk = np.mod(k0 + dk, nz)  # z periodic
            idx.append(np.where(ok, jj, 0) * nz + kk)
            wt.append(np.where(ok, wj * wk * Wn, 0).astype(np.float32))
    Nu, Ns = U.shape
    return (np.stack([i.ravel() for i in idx]), np.stack([w.ravel() for w in wt]),
            Nu, Ns, ds)


def sample(vol_xyz, tab):
    """vol (X, ny, nz) -> (X, Nu*Ns) bilinear samples on the ray tables."""
    idx, wt = tab[0], tab[1]
    flat = vol_xyz.reshape(vol_xyz.shape[0], -1)
    out = flat[:, idx[0]] * wt[0]
    for q in range(1, 4):
        out += flat[:, idx[q]] * wt[q]
    return out


def march(light, fog, tab, kfog, light_rgb, fog_rgb):
    """Front-to-back emission-absorption along s: L = sum T_i j_i ds,
    T_{i+1} = T_i exp(-kappa_i ds), kappa = kfog fog (the light itself is
    optically thin). light/fog: (X, Ns*Nu) samples -> (X, Nu, 3) radiance."""
    Nu, Ns, ds = tab[2], tab[3], tab[4]
    X = light.shape[0]
    jl = light.reshape(X, Nu, Ns)
    if fog is None:
        tot = jl.sum(2) * ds
        return tot[..., None] * light_rgb[None, None, :]
    jf = fog.reshape(X, Nu, Ns)
    tau = np.cumsum(jf, axis=2) * (kfog * ds)
    T = np.exp(-(tau - jf * (kfog * ds)))  # transmittance BEFORE sample i
    L = (T * jl).sum(2) * ds
    F = (T * jf).sum(2) * ds
    return L[..., None] * light_rgb[None, None, :] + F[..., None] * fog_rgb[None, None, :]


# --- I/O -------------------------------------------------------------------------


class Run:
    def __init__(self, prefix):
        m = json.load(open(prefix + ".json"))
        self.nx, self.ny, self.nz = m["nx"], m["ny"], m["nz"]
        self.prefix = prefix
        n3 = self.nx * self.ny * self.nz
        sz = os.path.getsize(prefix + "_omega.bin")
        self.frames = sz // (4 * n3)
        self.n3 = n3

    def vol(self, field, f):
        mm = np.memmap(self.prefix + "_" + field + ".bin", dtype="<f4", mode="r",
                       offset=4 * self.n3 * f, shape=(self.nz, self.ny, self.nx))
        return np.asarray(mm)

    def plane(self, field, f):
        n2 = self.nx * self.ny
        mm = np.memmap(self.prefix + "_xy_" + field + ".bin", dtype="<f4",
                       mode="r", offset=4 * n2 * f, shape=(self.ny, self.nx))
        return np.asarray(mm)


# --- the frame -------------------------------------------------------------------

G = {}  # per-worker state (set by init)


def init(cfg):
    G.update(cfg)
    G["run"] = Run(cfg["prefix"])
    G["tabs"] = {}


def theta_of(f):
    """The orbit: theta0 -> theta1 (degrees) over frames f0 .. f0 + nfr - 1,
    cosine ease in/out; radians."""
    q = (f - G["f0"]) / max(1, G["nfr"] - 1)
    q = 0.5 - 0.5 * np.cos(np.pi * q)
    return np.radians(G["theta0"] + (G["theta1"] - G["theta0"]) * q)


def light_image(f):
    """(X, Nu, 3) linear radiance of frame f (exposure not applied)."""
    r = G["run"]
    nb = G["bands"]
    x0, x1 = nb, r.nx - nb
    th = theta_of(f)
    key = round(th, 9)
    if key not in G["tabs"]:
        G["tabs"] = {key: ray_tables(r.ny, r.nz, th, G["uhalf"], 0.5 * r.ny,
                                     0.5 * r.nz)}
    tab = G["tabs"][key]
    # (nz, ny, nx) -> (X, ny, nz)
    om = np.ascontiguousarray(
        np.maximum(r.vol("omega", f)[:, :, x0:x1], 0).transpose(2, 1, 0))
    d = None
    if G["context"] == "vort":
        vo = r.vol("vort", f)[:, :, x0:x1].transpose(2, 1, 0)
        yf = r.vol("fuel", f)[:, :, x0:x1].transpose(2, 1, 0)
        d = np.ascontiguousarray(np.maximum(vo, 0) * np.clip(yf, 0, 1))
    # In x chunks (memory: the samples are X * Nu * Ns floats).
    out = []
    for a in range(0, om.shape[0], 24):
        b = min(a + 24, om.shape[0])
        light = sample(om[a:b], tab) * G["wscale"]
        fog = sample(d[a:b], tab) * G["vscale"] if d is not None else None
        out.append(march(light, fog, tab, G["kfog"], G["light_rgb"],
                         G["fog_rgb"]))
    return np.concatenate(out, 0)


PW, PH, TOP, BOT = 640, 1080, 56, 24  # panel size, title / footer margins


def panel_scale(nrows, ncols_cells):
    """Pixels per cell fitting rows = x cells and ncols_cells into a panel."""
    return min((PH - TOP - BOT) / nrows, (PW - 16) / ncols_cells)


def compose(f):
    r = G["run"]
    nb = G["bands"]
    X = r.nx - 2 * nb
    img = light_image(f) * G["exposure"]  # (X, Nu, 3), x up = row index
    img = img[::-1]  # jet axis up
    sc = panel_scale(X, 2 * G["uhalf"])
    ch, cw = int(round(X * sc)), int(round(2 * G["uhalf"] * sc))
    hdr = resample(img.astype(np.float32), ch, cw)
    bloom = 0.10 * gauss_fft(hdr, 5.0) + 0.06 * gauss_fft(hdr, 22.0)
    flame = srgb(aces(hdr + bloom))
    frame = np.zeros((PH, 3 * PW, 3), np.float32)
    ox = (PW - cw) // 2
    frame[TOP:TOP + ch, ox:ox + cw] = flame
    # Mid-span planes, the same crop and vertical scale.
    y0 = int(round(0.5 * r.ny - G["uhalf"]))
    y1 = y0 + int(round(2 * G["uhalf"]))
    for p, (field, cp, vs) in enumerate((("vort", INFERNO, G["vmax"]),
                                         ("speed", VIRIDIS, G["smax"]))):
        pl = r.plane(field, f)[:, nb:r.nx - nb]  # (ny, X)
        pl = pl[max(y0, 0):min(y1, r.ny)].T[::-1]  # (X, Y) x up
        v = resample(pl / vs, ch, int(round(pl.shape[1] * sc)))
        rgb = lut(np.sqrt(np.clip(v, 0, None)) if field == "vort" else v, cp)
        w = rgb.shape[1]
        o = (p + 1) * PW + (PW - w) // 2
        frame[TOP:TOP + ch, o:o + w] = rgb
    return (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)


def raw_light_lum(f):
    img = light_image(f)
    return np.percentile(img @ np.array([0.2126, 0.7152, 0.0722]), 99.5)


# --- self-test ---------------------------------------------------------------------


def selftest():
    ok = True
    nx, ny, nz = 6, 40, 32
    j, k = np.meshgrid(np.arange(ny) + 0.5, np.arange(nz) + 0.5, indexing="ij")
    rgb = np.array([1.0, 1.0, 1.0])
    # An emitting Gaussian blob centred off-centre in z (wraps): total light
    # = integral j dV / pixel area (optically thin), any theta.
    def blob(zc):
        dz = (k - zc + nz / 2) % nz - nz / 2
        g = np.exp(-((j - 20.0) ** 2 + dz ** 2) / (2 * 3.0 ** 2))
        return np.broadcast_to(g, (nx, ny, nz)).astype(np.float32)
    v = blob(30.0)
    total = v.sum()  # cell volume 1
    for th in (0.0, np.radians(25.0), np.radians(40.0)):
        tab = ray_tables(ny, nz, th, 19.0, 20.0, 16.0)
        img = march(sample(v, tab), None, tab, 0.0, rgb, rgb)
        tot = img[..., 0].sum() * 0.5  # pixel = 1 cell (x) x 0.5 cell (u)
        e = abs(tot - total) / total
        print(f"  blob theta {np.degrees(th):4.1f}: image total / integral = "
              f"{tot / total:.6f}")
        ok &= e < 0.01
    # Periodic-wrap continuity: a z-uniform slab renders z-shift invariant.
    v2 = blob(4.0)
    tab = ray_tables(ny, nz, np.radians(30.0), 19.0, 20.0, 16.0)
    a = march(sample(blob(30.0), tab), None, tab, 0.0, rgb, rgb)
    b = march(sample(v2, tab), None, tab, 0.0, rgb, rgb)
    sh = np.abs(a[..., 0].sum() - b[..., 0].sum()) / a[..., 0].sum()
    print(f"  wrap: blob across the z seam vs inside, total differs {sh:.2e}")
    ok &= sh < 0.01
    # Determinism: identical bytes twice.
    c = march(sample(v, tab), sample(v, tab), tab, 0.3, rgb, rgb)
    d = march(sample(v, tab), sample(v, tab), tab, 0.3, rgb, rgb)
    same = c.tobytes() == d.tobytes()
    print(f"  determinism: {'identical' if same else 'DIFFERENT'} bytes")
    ok &= same
    # Absorption lowers the light, never raises it.
    ok &= bool((c[..., 0] <= march(sample(v, tab), sample(v, tab), tab, 0.0,
                                   rgb, rgb)[..., 0] + 1e-6).all())
    col = chemi_rgb()
    print(f"  chemiluminescence linear sRGB (unit luminance): "
          f"{col[0]:.3f} {col[1]:.3f} {col[2]:.3f}")
    ok &= col[2] > col[1] > 0 and col[2] > col[0]
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --- main ------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prefix", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--png")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--bands", type=int, default=16, help="sink band cells")
    ap.add_argument("--uhalf", type=float, default=0,
                    help="image half-width in cells (0 = ny/2)")
    ap.add_argument("--theta0", type=float, default=15.0)
    ap.add_argument("--theta1", type=float, default=40.0)
    ap.add_argument("--context", choices=("vort", "off"), default="vort")
    ap.add_argument("--fog", type=float, default=0.25,
                    help="fog emission relative to the light's exposure")
    ap.add_argument("--kfog", type=float, default=0.02,
                    help="fog absorption per cell at unit fog density")
    ap.add_argument("--exposure", type=float, default=0.6,
                    help="the light's 99.5th percentile maps to this")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=-1)
    ap.add_argument("--titles", default="")
    ap.add_argument("--footer", default="")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.prefix or not a.out:
        sys.exit("PREFIX and --out are required")
    r = Run(a.prefix)
    last = r.frames if a.last < 0 else min(a.last, r.frames)
    frames = list(range(a.first, last))
    if not frames:
        sys.exit("no frames")
    nfr = len(frames)
    uhalf = a.uhalf if a.uhalf > 0 else 0.5 * r.ny

    # Scales from the last frames (fixed per clip).
    tail = frames[-min(30, nfr):]
    om = [np.percentile(r.vol("omega", f), 99.99) for f in tail[::6]]
    wscale = 1.0 / max(np.median(om), 1e-30)
    nb = a.bands
    vv = []
    for f in tail[::6]:
        vo = r.vol("vort", f)[:, :, nb:r.nx - nb]
        yf = r.vol("fuel", f)[:, :, nb:r.nx - nb]
        vv.append(np.percentile(vo * np.clip(yf, 0, 1), 99.9))
    vscale = 1.0 / max(np.median(vv), 1e-30)
    vmax = np.median([np.percentile(r.plane("vort", f), 99.5) for f in tail])
    smax = np.median([np.percentile(r.plane("speed", f), 99.9) for f in tail])
    cfg = dict(prefix=a.prefix, bands=nb, uhalf=uhalf, context=a.context,
               kfog=a.kfog, light_rgb=chemi_rgb().astype(np.float32),
               fog_rgb=(FOG_RGB * a.fog).astype(np.float32), wscale=wscale,
               vscale=vscale, vmax=vmax, smax=smax, exposure=1.0,
               f0=frames[0], nfr=nfr, theta0=a.theta0, theta1=a.theta1)
    init(cfg)
    # Exposure: the light's 99.5th percentile luminance over the last frames
    # maps to --exposure before the tone curve.
    with mp.Pool(a.workers, initializer=init, initargs=(cfg,)) as pool:
        lum = pool.map(raw_light_lum, tail)
    cfg["exposure"] = a.exposure / max(float(np.median(lum)), 1e-30)
    print(f"{nfr} frames, omega scale {1 / wscale:.4g}, fog scale "
          f"{1 / vscale:.4g}, |curl u| max {vmax:.4g}, |u| max {smax:.4g}, "
          f"exposure {cfg['exposure']:.4g}", flush=True)

    W, H = 3 * PW, PH
    # Text through textfile= (no filtergraph escaping), files beside --out.
    vf = []

    def textfile(tag, txt):
        path = f"{a.out}.{tag}.txt"
        with open(path, "w") as fh:
            fh.write(txt)
        return path

    if a.titles:
        for p, txt in enumerate(a.titles.split(";")[:3]):
            vf.append(f"drawtext=fontfile={FONT}:textfile={textfile(p, txt)}:"
                      f"fontcolor=0xd8d8d8:fontsize=22:"
                      f"x={p * PW}+({PW}-text_w)/2:y=18")
    if a.footer:
        vf.append(f"drawtext=fontfile={FONT}:textfile={textfile('f', a.footer)}:"
                  f"fontcolor=0x9a9a9a:fontsize=15:x=(w-text_w)/2:y=h-20")
    enc = ["/usr/bin/ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-"]
    filt = ["-vf", ",".join(vf)] if vf else []
    p = subprocess.Popen(enc + filt + ["-c:v", "libx264", "-preset", "slow",
                                       "-pix_fmt", "yuv420p", "-crf",
                                       str(a.crf), "-movflags", "+faststart",
                                       a.out], stdin=subprocess.PIPE)
    lastimg = None
    with mp.Pool(a.workers, initializer=init, initargs=(cfg,)) as pool:
        for n, img in enumerate(pool.imap(compose, frames, chunksize=2)):
            p.stdin.write(img.tobytes())
            lastimg = img
            if n % 50 == 0:
                print(f"  frame {n}/{nfr}", flush=True)
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    if a.png:
        q = subprocess.run(enc + filt + ["-frames:v", "1", a.png],
                           input=lastimg.tobytes())
        if q.returncode != 0:
            sys.exit("ffmpeg png failed")
    print(f"{a.out}: {nfr} frames {W}x{H} @ {a.fps} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
