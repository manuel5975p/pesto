# dns — incompressible 3D DNS (CPU + HIP GPU)

Grid-based DNS of incompressible turbulence in a triply periodic box.
Staggered MAC grid, 2nd-order central differences (discretely energy-conserving
divergence form), SSP-RK3 with Chorin projection per stage, FFT pressure solve
(exact inversion of the discrete Laplacian via modified wavenumbers).

Two backends:
- **CPU** (`dns_core`): serial, double precision — the correctness oracle.
- **GPU** (`dns_gpu`, optional): HIP/ROCm, FP32 fields with FP64 reductions,
  hipFFT Poisson, fused RHS+RK-stage kernel. ~86 ms/step at 512³ on an
  RX 7900 XTX (~7 GiB VRAM).

## Build

```sh
cmake -B build -G Ninja -DDNS_WITH_HIP=ON   # OFF for CPU-only
ninja -C build
ctest --test-dir build                       # GPU tests skip without a device
```

Options: `DNS_WITH_FFTW` (default ON), `DNS_WITH_HIP` (default OFF,
`CMAKE_HIP_ARCHITECTURES` defaults to gfx1100).

## Apps

- `hit [n] [--urms m/s] [--nu m2/s] [--spin Te] [--stats Te] [--seed s]
  [--out prefix] [--snap]` — forced homogeneous isotropic turbulence (GPU).
  Configured in SI; the box is sized so h = 1.5·η (honest DNS), which fixes
  Re_λ by N alone (256³ → 68, 512³ → 107, at any u′). Runs internally in box
  units (L = 2π, u′ = 1) for FP32 health. Writes `<out>_stats.csv`
  (K, u′, ε, Re_λ, CFL, divergence), `<out>_spec_*.csv` energy spectra, and
  optional raw snapshots. Linear forcing with fixed-ε controller
  (Carroll–Blanquart): stationary ⟨ε_visc⟩ = ε_f.
- `tgv3d [n] [Re] [T]` — 3D Taylor–Green benchmark, CSV dissipation series
  (GPU when available). At Re=1600 vs the HiOCFD spectral reference
  (peak dissipation 0.012858 @ t=8.97): 256³ peak −4.3%, 512³ peak +1.2%.
- `tgv` — 2D Taylor–Green (analytic validation), `bench [n] [steps]` — GPU
  step timing.

## Layout

`src/dns/` CPU core (grid, operators, poisson, projection, timestepper,
forcing, hit_ic, diagnostics, units). `src/dns/gpu/` HIP backend mirroring the
CPU API; no HIP header leaks into public headers. `tests/` parity- and
physics-tests (GPU vs CPU oracle; KE conservation; TGV convergence; forced-HIT
stationarity).

## Physics ceiling (single 24 GB GPU)

Resolving the Kolmogorov scale in air (ν=1.5e-5) at u′=60 m/s forces sub-mm
boxes: 512³ → L ≈ 1 mm, ε ≈ 1.1e9 W/kg, Re_λ ≈ 107, dt ≈ 7 ns. Lower u′
scales the box up but leaves Re_λ unchanged. 768³ (Re_λ ≈ 141) exceeds 24 GiB
with the current buffer set — needs scratch elimination first.
