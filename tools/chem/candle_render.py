#!/usr/bin/env python3
"""Physically coloured render of a `dns flame --mode candle --wick` run.

Reads <prefix>.json (nx, ny, frames, frameDt, mach), <prefix>_mask.bin (1
fluid, 0 solid: the candle body and wick) and the --emit bin frames
<prefix>_<field>.bin (raw little-endian float32, nx*ny per frame, i fastest):
temp (T - T0), soot (rho Y_s, box units), omega (the fuel reaction rate).
docs/chem.md, "Wick candle"; docs/viewer-format.md.

Colour is PHYSICS, not a colormap:
  * soot incandescence -- small (Rayleigh) particles: spectral absorption
    kappa_l = 6 pi E(m) f_v / lambda, optically thin, emitted spectral radiance
    I_l = kappa_l B_l(T) L for a slab of depth L (2D: one cell column, no line
    of sight). f_v = rho Y_s rho0 / rho_soot (rho_soot/rho0 = 1800/1.177),
    T = T0K (1 + temp/T0_box). Integrated against the CIE 1931 colour-matching
    functions (Wyman-Sloan-Shirley 2013 multi-lobe fit) -> XYZ -> linear sRGB
    (D65).
  * chemiluminescence -- CH* (431 nm) and C2* (Swan, 516 / 473 nm) stand-in:
    a line spectrum scaled by the local heat release omega (one-step
    chemistry has no CH*; the intensity is a stand-in, its colour is the
    lines').
  * exposure fixed per video from the last frame (no flicker from auto
    exposure), filmic tone curve (ACES fit), sRGB OETF, mild optical blur and
    a bloom halo.
  * the candle body (wax albedo) and wick (char) are drawn from the mask,
    lit by the flame's own emission (a 1/r falloff from the emitting cells,
    evaluated by an FFT convolution), with a translucent glow under the wax
    top. The wick is held at its vaporisation temperature in the model
    (~600 K): it does not incandesce, so it is drawn dark.

numpy + ffmpeg only.

    candle_render.py PREFIX --out candle.mp4 [--png last.png] [--first N]
                     [--scale 4] [--fps 30] [--t0-kelvin 300] [--exposure X]
                     [--chemi 1] [--width L] [--below L] [--above L]
"""

import argparse
import json
import subprocess
import sys

import numpy as np

# --- spectra ------------------------------------------------------------------
LAM = np.arange(380.0, 781.0, 5.0)  # nm
H_PLANCK, C_LIGHT, K_BOLTZ = 6.62607015e-34, 2.99792458e8, 1.380649e-23
RHO_SOOT_OVER_AIR = 1800.0 / 1.177  # dns::kSootDensityBox
E_M = 0.26  # soot absorption function E(m) (Dalzell-Sarofim range 0.2-0.4)


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


def planck(lam_nm, T):
    """Spectral radiance B_lambda(T), W / (m^2 sr m)."""
    l = lam_nm * 1e-9
    a = 2 * H_PLANCK * C_LIGHT ** 2 / l ** 5
    with np.errstate(over="ignore"):
        return a / np.expm1(H_PLANCK * C_LIGHT / (l * K_BOLTZ * T))


def soot_table(tmax=3500.0):
    """XYZ per unit (f_v * L / 1 m) of an optically thin Rayleigh soot slab
    at T: sum over lambda of kappa_l/f_v B_l(T) cmf(lambda) dlambda."""
    Ts = np.arange(0.0, tmax + 1.0, 1.0)
    cmf = cie_xyz(LAM)
    kap = 6 * np.pi * E_M / (LAM * 1e-9)  # per unit f_v, 1/m
    tab = np.zeros((Ts.size, 3))
    hot = Ts > 250
    B = planck(LAM[None, :], Ts[hot, None])  # (nT, nl)
    tab[hot] = (B * kap[None, :]) @ cmf * (5e-9)
    return Ts, tab


def chemi_rgb():
    """Linear sRGB colour (unit luminance) of the CH* 431 nm + C2* 516/473 nm
    line spectrum (relative line strengths 1 : 0.6 : 0.3)."""
    lines = np.array([431.0, 516.0, 473.0])
    w = np.array([1.0, 0.6, 0.3])
    xyz = (cie_xyz(lines) * w[:, None]).sum(0)
    rgb = XYZ2RGB @ (xyz / xyz[1])
    return np.clip(rgb, 0, None)


# --- image helpers ---------------------------------------------------------------
def blur(a, sigma):
    """Gaussian blur (separable, reflect padding) of a 2D or 3D (H, W, C)
    array, sigma in pixels."""
    if sigma <= 0:
        return a
    r = int(3 * sigma + 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(r, r), (0, 0)] + [(0, 0)] * (a.ndim - 2)
    p = np.pad(a, pad, mode="reflect")
    a = sum(k[i] * p[i:i + a.shape[0]] for i in range(2 * r + 1))
    pad = [(0, 0), (r, r)] + [(0, 0)] * (a.ndim - 2)
    p = np.pad(a, pad, mode="reflect")
    return sum(k[i] * p[:, i:i + a.shape[1]] for i in range(2 * r + 1))


def fft_conv(a, kern):
    """2D convolution of each channel of a (H, W, C) with a centred kernel of
    shape (2H, 2W) (zero padded: no wrap)."""
    H, W = a.shape[:2]
    K = np.fft.rfft2(np.fft.ifftshift(kern))
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        p = np.zeros((2 * H, 2 * W))
        p[:H, :W] = a[..., c]
        out[..., c] = np.fft.irfft2(np.fft.rfft2(p) * K, s=(2 * H, 2 * W))[:H, :W]
    return out


def aces(x):
    """ACES filmic tone curve (Narkowicz 2015 fit)."""
    x = np.maximum(x, 0)
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def upscale(a, s):
    """Bilinear upscale by integer s of (H, W[, C]) (cell centres kept)."""
    if s == 1:
        return a
    H, W = a.shape[:2]
    yi = (np.arange(H * s) + 0.5) / s - 0.5
    xi = (np.arange(W * s) + 0.5) / s - 0.5
    y0 = np.clip(np.floor(yi).astype(int), 0, H - 1)
    x0 = np.clip(np.floor(xi).astype(int), 0, W - 1)
    y1, x1 = np.clip(y0 + 1, 0, H - 1), np.clip(x0 + 1, 0, W - 1)
    fy = np.clip(yi - y0, 0, 1)
    fx = np.clip(xi - x0, 0, 1)
    if a.ndim == 3:
        fy, fx = fy[:, None, None], fx[None, :, None]
    else:
        fy, fx = fy[:, None], fx[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prefix")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--png", help="final frame .png")
    ap.add_argument("--first", type=int, default=0, help="first frame")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=4, help="pixels per cell")
    ap.add_argument("--t0-kelvin", type=float, default=300.0,
                    help="ambient T0 in kelvin (the box T scale's anchor)")
    ap.add_argument("--cell-mm", type=float, default=0.4,
                    help="cell size in mm: the soot slab depth L (one cell)")
    ap.add_argument("--exposure", type=float, default=0.0,
                    help="linear exposure; 0 = auto from the last frame")
    ap.add_argument("--chemi", type=float, default=1.0,
                    help="chemiluminescence gain (relative to auto)")
    ap.add_argument("--white-k", type=float, default=1900.0,
                    help="white balance: the blackbody T rendered neutral "
                         "(candlelight ~1900 K; 0 = none, raw D65 sRGB)")
    ap.add_argument("--revolve", action="store_true",
                    help="render the planar field AS IF axisymmetric (line-"
                         "of-sight integral of its body of revolution); a "
                         "visualization, the physics is 2D")
    ap.add_argument("--light", type=float, default=1.0,
                    help="flame light on the candle body (relative to auto)")
    ap.add_argument("--cells-per-unit", type=float, default=8.0,
                    help="the run's --cells-per-unit (cells per l)")
    ap.add_argument("--width", type=float, default=12.0,
                    help="image width in l, centred on the wick")
    ap.add_argument("--below", type=float, default=2.5,
                    help="candle shown below the wax top, l")
    ap.add_argument("--above", type=float, default=14.0,
                    help="image extent above the wax top, l")
    a = ap.parse_args()

    meta = json.load(open(a.prefix + ".json"))
    nx, ny, nf = meta["nx"], meta["ny"], meta["frames"]
    n = nx * ny
    T0box = 1.0 / (1.4 * meta["mach"] ** 2)
    mask = np.fromfile(a.prefix + "_mask.bin", dtype="<f4").reshape(ny, nx)
    temp = np.memmap(a.prefix + "_temp.bin", dtype="<f4", mode="r")
    soot = np.memmap(a.prefix + "_soot.bin", dtype="<f4", mode="r")
    omega = np.memmap(a.prefix + "_omega.bin", dtype="<f4", mode="r")
    if min(temp.size, soot.size, omega.size) < nf * n:
        sys.exit("frame files shorter than the json says")

    # Geometry from the mask: the wick = the solid columns' narrowest top
    # part; wax top = the highest row where the solid spans > 2x the wick.
    solid = mask < 0.5  # (ny, nx), x = i
    cols = solid.sum(0)  # solid cells per x row
    rows = np.nonzero(cols)[0]
    wickw = cols[rows[-1]]
    body = rows[cols[rows] > 2 * wickw]
    iTop = body[-1] + 1  # first row above the wax top
    js = np.nonzero(solid[:, rows[-1]])[0]
    jc = 0.5 * (js[0] + js[-1] + 1)  # axis (cell units)
    # The json carries no cell size: --cells-per-unit names it (the crop is
    # in l).
    cpl = a.cells_per_unit
    i0 = max(0, int(iTop - a.below * cpl))
    i1 = min(nx, int(iTop + a.above * cpl))
    j0 = max(0, int(jc - 0.5 * a.width * cpl))
    j1 = min(ny, int(jc + 0.5 * a.width * cpl))

    Ts, tab = soot_table()
    # Camera white balance (von Kries in linear sRGB): a blackbody at
    # --white-k renders neutral, as a photograph of a candle is balanced.
    wb = np.ones(3)
    if a.white_k > 0:
        ref = np.clip(XYZ2RGB @ ((planck(LAM[:, None], a.white_k)[:, 0]
                                  [:, None] * cie_xyz(LAM)).sum(0)), 1e-30,
                      None)
        wb = ref[1] / ref
    tab = (tab @ XYZ2RGB.T) * wb  # the table now holds balanced linear RGB
    crgb = chemi_rgb() * wb
    crgb /= crgb.max()  # unit peak channel: the display gain sets its level
    L = a.cell_mm * 1e-3

    def crop(f, arr):
        v = np.asarray(arr[f * n:(f + 1) * n]).reshape(ny, nx)
        return v[j0:j1, i0:i1].T[::-1, :].astype(np.float64)  # rows = x down

    sm = solid[j0:j1, i0:i1].T[::-1, :]
    xs = (np.arange(i1 - 1, i0 - 1, -1) + 0.5)  # row centres (cells), top first
    wax = sm & (xs[:, None] < iTop)
    wick = sm & ~wax

    def emission(f):
        """Linear sRGB radiance of the gas: soot incandescence + chemi."""
        T = a.t0_kelvin * (1.0 + crop(f, temp) / T0box)
        fv = np.maximum(crop(f, soot), 0.0) / RHO_SOOT_OVER_AIR
        idx = np.clip(np.rint(T), 0, Ts.size - 1).astype(int)
        rgb = np.clip(tab[idx] * (fv * L)[..., None], 0, None)
        w = np.maximum(crop(f, omega), 0.0)
        return rgb, w

    H, W = sm.shape
    # 1/r light falloff kernel (cells), softened at the core.
    yy, xx = np.mgrid[-H:H, -W:W]
    kern = 1.0 / (1.0 + (xx ** 2 + yy ** 2) / 4.0)
    depth = np.where(wax, iTop - xs[:, None], 0.0)  # cells below the wax top
    albedo_wax = np.array([0.95, 0.92, 0.85])
    albedo_wick = np.array([0.03, 0.026, 0.022])
    S = a.scale

    # --revolve: the planar emission read as the meridional slice of a body
    # of revolution about the wick axis and integrated along the line of
    # sight, I(x, Y) = int E(x, sqrt(Y^2 + z^2)) dz (the left/right halves
    # averaged through the radial profile). A VISUALIZATION of 2D data as if
    # it were axisymmetric -- the physics is planar.
    jAxis = jc - j0  # axis in cropped columns (cells)
    Wc = j1 - j0
    rgrid = np.abs(np.arange(Wc) + 0.5 - jAxis)
    order = np.argsort(rgrid, kind="stable")
    zz = np.arange(-0.5 * Wc, 0.5 * Wc + 0.25, 0.25)
    rr = np.sqrt((np.arange(Wc) + 0.5 - jAxis)[:, None] ** 2 + zz[None, :] ** 2)

    def revolve(e):
        """(H, Wc[, C]) planar -> line-of-sight integral (same shape)."""
        out = np.zeros_like(e)
        rs = rgrid[order]
        for r_ in range(e.shape[0]):
            prof = e[r_][order]
            if prof.ndim == 1:
                out[r_] = np.interp(rr, rs, prof, right=0.0).sum(1) * 0.25
            else:
                for c in range(prof.shape[1]):
                    out[r_, :, c] = (np.interp(rr, rs, prof[:, c], right=0.0)
                                     .sum(1) * 0.25)
        return out

    def fields(f):
        """(soot radiance, heat release) of frame f, masked, optionally
        revolved."""
        rgb, w = emission(f)
        w = blur(w, 0.7)
        rgb[sm] = 0.0
        w[sm] = 0.0
        if a.revolve:
            rgb, w = revolve(rgb), revolve(w)
            rgb[sm] = 0.0
            w[sm] = 0.0
        return rgb, w

    # Exposure and chemiluminescence scale from the last frame: the soot
    # zone's 99.5th percentile luminance maps near the shoulder of the tone
    # curve; the chemiluminescence peak channel is set to 0.5 after exposure
    # (a candle's blue base is faint beside its yellow zone, but visible).
    rgbL, wL = fields(nf - 1)
    lum = rgbL @ np.array([0.2126, 0.7152, 0.0722])
    lref = np.percentile(lum[lum > 0], 99.5) if np.any(lum > 0) else 1.0
    expo = a.exposure if a.exposure > 0 else 1.6 / lref
    wref = np.percentile(wL[wL > 0], 99.9) if np.any(wL > 0) else 1.0
    cgain = a.chemi * 0.5 / (expo * wref)

    def gasRadiance(f):
        """(total gas radiance, soot part): the soot part lights the body
        (a candle's chemiluminescent output is ~1e-3 of its soot
        radiation; the display gain above exaggerates it for visibility)."""
        rgb, w = fields(f)
        return rgb + cgain * w[..., None] * crgb, rgb

    # The body's illumination scale: the last frame's light on the wax top
    # maps to a mid grey after exposure (the flame is the only light).
    lightL = fft_conv(gasRadiance(nf - 1)[1], kern)
    lumW = lightL @ np.array([0.2126, 0.7152, 0.0722])
    ltop = lumW[wax].max() if np.any(wax) else 1.0
    lgain = a.light * 0.35 / (expo * ltop)
    Hs, Ws = H * S - (H * S) % 2, W * S - (W * S) % 2

    def render(f):
        gas, sootRgb = gasRadiance(f)
        # Light reaching the body: the emitting soot through the 1/r kernel;
        # the wax glows translucently just under its top.
        light = fft_conv(sootRgb, kern) * lgain
        img = gas.copy()
        lit = light * (0.55 + 0.45 * np.exp(-depth / 5.0))[..., None]
        amb = np.array([0.006, 0.006, 0.009]) / expo
        img[wax] = albedo_wax * (lit[wax] + amb)
        img[wick] = albedo_wick * (lit[wick] + amb)
        img = upscale(img, S)
        # Optics: mild blur + bloom halo of the bright gas.
        bright = upscale(gas, S)
        img = blur(img, 0.6 * S / 4) + 0.10 * blur(bright, 3.0 * S) \
            + 0.05 * blur(bright, 10.0 * S)
        out = srgb(aces(img * expo))
        return (out[:Hs, :Ws] * 255 + 0.5).astype(np.uint8)

    enc = ["/usr/bin/ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{Ws}x{Hs}", "-r", str(a.fps), "-i", "-"]
    p = subprocess.Popen(enc + ["-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-crf", "16", a.out], stdin=subprocess.PIPE)
    for f in range(a.first, nf):
        p.stdin.write(render(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    if a.png:
        q = subprocess.run(enc + ["-frames:v", "1", a.png],
                           input=render(nf - 1).tobytes())
        if q.returncode != 0:
            sys.exit("ffmpeg png failed")
    print(f"{a.out}: {nf - a.first} frames {Ws}x{Hs} @ {a.fps} fps "
          f"(exposure {expo:.4g}, soot luminance ref {lref:.4g}, "
          f"chemi gain {cgain:.4g})")


if __name__ == "__main__":
    main()
