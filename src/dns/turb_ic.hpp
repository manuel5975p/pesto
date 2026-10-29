// Synthetic homogeneous isotropic turbulence for `dns flame --turb`
// (docs/chem.md, "3D and turbulence"): a periodic, solenoidal, zero-mean
// velocity box built from its Fourier modes DIRECTLY, so the field is a
// function of the seed and the box alone -- never of the grid it is sampled
// on. Each lattice wavevector m draws its complex Gaussian amplitude from a
// hash of (seed, m) (SplitMix64 + Box-Muller); the canonical member of every
// +-m pair is drawn and the other is its conjugate (exactly Hermitian). A
// coarser grid holds a subset of exactly the same modes: the coarse field is
// the spectral truncation of the fine one (`dns hit`'s generator draws white
// noise in PHYSICAL space and is not).
//
// Spectrum: hitModelSpectrum(k/kp) (dns/hit_ic.hpp), kp = sqrt(2 pi)/lt,
// for which the longitudinal integral scale L11 = sqrt(2 pi)/kp = lt exactly
// (continuous spectrum). Shell energies are rescaled to the deterministic
// per-mode weights E(k) dk^3/(4 pi k^2) summed over the shell, then one
// global factor sets u'. Shells and normalisation are taken over the
// REFERENCE set |k| <= kTurbKmax kp (grid-independent), serial FP64 sums in
// lexicographic (mz, my, mx) order. Modes with my = mz = 0 are excluded: every
// x-plane mean of u, v, w is then exactly 0 (no inflow mass-flux pumping when
// the box is swept through an inflow face).
//
// Geometry layer: host only, no GPU. turbBox needs FFTW.
#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <vector>

namespace dns {

// Reference-set cutoff in units of kp. E(8 kp) ~ 4e-24 of the peak.
inline constexpr double kTurbKmax = 8.0;

struct TurbSpec
{
    double urms = 1;               // u' = sqrt(<|u|^2>/3), box units
    double lt = 1;                 // integral scale L11 (sets kp)
    double Lx = 1, Ly = 1, Lz = 1; // the periodic box
    int nx = 1, ny = 1, nz = 1;    // the grid it is sampled on (h_d = L_d/n_d)
    std::uint64_t seed = 1;
};

// One lattice mode of the field: integer wavevector m, its velocity
// amplitude (u(x) = sum_m uh_m exp(i k_m . x), k_d = 2 pi m_d / L_d).
struct TurbMode
{
    int m[3];
    std::complex<double> uh[3];
};

// kp of the model spectrum for integral scale lt: sqrt(2 pi)/lt.
// Pre: lt > 0.
double turbPeak(double lt);

// The modes of s on its grid: the reference set (|k| <= kTurbKmax kp, |k| >
// 0, (my, mz) != (0, 0)) restricted to |m_d| < n_d/2 (no Nyquist), both
// members of each +-m pair, (mz, my, mx) lexicographic. Amplitudes are the
// SAME for a mode on any grid that holds it.
// Pre: urms > 0, lt > 0, L_d > 0, n_d >= 1.
std::vector<TurbMode> turbModes(const TurbSpec& s);

// Direct FP64 evaluation of the field at a point (the oracle of turbBox).
std::array<double, 3> turbPoint(const std::vector<TurbMode>& modes,
                                const TurbSpec& s, double x, double y,
                                double z);

// Expected L11 of the lattice spectrum (the deterministic per-mode weights,
// not the random draw): (3 pi/4) sum w/|k| / sum w over the reference set.
// Converges to lt as L/lt grows.
double turbLatticeL11(const TurbSpec& s);

// Realized shell energies of `modes` (0.5 sum |uh|^2 per shell, shell =
// lround(|k|/k0), k0 = min_d 2 pi/L_d) and the deterministic targets the
// generator rescaled them to (same index; the targets scaled by the same
// global factor). Both FP64, serial.
struct TurbShells
{
    std::vector<double> realized, target;
};
TurbShells turbShells(const std::vector<TurbMode>& modes, const TurbSpec& s);

#if DNS_WITH_FFTW
// The field at the cell centres ((i + 1/2) Lx/nx, ...), i fastest
// (i + nx (j + ny k)): {u, v, w}. One FFTW c2r per component, serial.
std::array<std::vector<double>, 3> turbBox(const TurbSpec& s);
#endif

// Turbulence groups of a premixed flame, box units (S_L = 1): Re_t = u' lt /
// nu, Kolmogorov eta = lt Re_t^(-3/4), Ka = u'^1.5 (lt/dth)^(-0.5), Da =
// (lt/dth)/u', eddy turnover tau_t = lt/u'.
// Pre: urms, lt, nu, dth > 0.
struct TurbNumbers
{
    double ReT, eta, Ka, Da, tauT;
};
TurbNumbers turbNumbers(double urms, double lt, double nu, double dth);

// Peters (1999) S_T/S_L (u' = urms, l = lt, l_F = lF; a4 0.78, b1 2.0, b3 1.0):
// (S_T - 1) = -A l/l_F + sqrt((A l/l_F)^2 + a4 b3^2 u' l/l_F), A = a4 b3^2/(2
// b1). Report-only correlation.
double petersST(double urms, double lt, double lF);
// Damkoehler's large-scale limit S_T/S_L = 1 + u'/S_L. Report-only.
double damkohlerST(double urms);

} // namespace dns
