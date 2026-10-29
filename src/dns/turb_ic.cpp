#include "dns/turb_ic.hpp"

#include "dns/hit_ic.hpp"

#if DNS_WITH_FFTW
#include <fftw3.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>

namespace dns {

namespace {

using cplx = std::complex<double>;
constexpr double kTwoPi = 2.0 * std::numbers::pi;

std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// The mode's own random stream: the seed mixed with the packed wavevector
// (21 bits per component, offset 2^20).
std::uint64_t modeKey(std::uint64_t seed, int mx, int my, int mz)
{
    auto pk = [](int v) {
        return std::uint64_t(std::int64_t(v) + (std::int64_t(1) << 20)) &
               0x1fffffULL;
    };
    return splitmix64(seed ^ splitmix64(pk(mx) | (pk(my) << 21) | (pk(mz) << 42)));
}

// Uniform in [0, 1) from the n-th draw of a key (53 bits).
double unit(std::uint64_t key, int n)
{
    return double(splitmix64(key + std::uint64_t(n)) >> 11) * 0x1.0p-53;
}

// The canonical member of a +-m pair (modes with my = mz = 0 are excluded).
bool canonical(int my, int mz) { return mz > 0 || (mz == 0 && my > 0); }

struct Lattice
{
    double dk[3];  // 2 pi / L_d
    double kp, kmax, k0;
    int mMax[3];   // reference-set bound per axis
};

Lattice latticeOf(const TurbSpec& s)
{
    assert(s.urms > 0 && s.lt > 0 && s.Lx > 0 && s.Ly > 0 && s.Lz > 0);
    Lattice l;
    const double L[3] = {s.Lx, s.Ly, s.Lz};
    l.kp = turbPeak(s.lt);
    l.kmax = kTurbKmax * l.kp;
    l.k0 = HUGE_VAL;
    for (int d = 0; d < 3; ++d) {
        l.dk[d] = kTwoPi / L[d];
        l.mMax[d] = int(std::floor(l.kmax / l.dk[d]));
        l.k0 = std::min(l.k0, l.dk[d]);
    }
    return l;
}

// Visits the reference set in (mz, my, mx) lexicographic order:
// f(m, k[3], |k|^2).
template <class F>
void forReference(const Lattice& l, F&& f)
{
    const double kmax2 = l.kmax * l.kmax;
    for (int mz = -l.mMax[2]; mz <= l.mMax[2]; ++mz)
        for (int my = -l.mMax[1]; my <= l.mMax[1]; ++my) {
            if (my == 0 && mz == 0) continue;
            for (int mx = -l.mMax[0]; mx <= l.mMax[0]; ++mx) {
                const double k[3] = {mx * l.dk[0], my * l.dk[1], mz * l.dk[2]};
                const double kk = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                if (kk > kmax2) continue;
                const int m[3] = {mx, my, mz};
                f(m, k, kk);
            }
        }
}

// The projected (solenoidal) draw of mode m, before any scaling: three
// complex standard Gaussians (Box-Muller on the mode's key), minus k (k.a)/k^2.
// A non-canonical mode is the conjugate of its partner's draw.
void drawMode(std::uint64_t seed, const int m[3], const double k[3], double kk,
              cplx a[3])
{
    const bool can = canonical(m[1], m[2]);
    const int s = can ? 1 : -1;
    const std::uint64_t key = modeKey(seed, s * m[0], s * m[1], s * m[2]);
    for (int c = 0; c < 3; ++c) {
        const double u1 = 1.0 - unit(key, 2 * c), u2 = unit(key, 2 * c + 1);
        const double r = std::sqrt(-2.0 * std::log(u1));
        a[c] = cplx(r * std::cos(kTwoPi * u2), r * std::sin(kTwoPi * u2));
    }
    // Projection with the canonical k (sign-free: k k^T is even in k).
    const cplx kd = (k[0] * a[0] + k[1] * a[1] + k[2] * a[2]) / kk;
    for (int c = 0; c < 3; ++c) a[c] -= k[c] * kd;
    if (!can)
        for (int c = 0; c < 3; ++c) a[c] = std::conj(a[c]);
}

int shellOf(const Lattice& l, double kk)
{
    return int(std::lround(std::sqrt(kk) / l.k0));
}

// Deterministic per-mode energy weight E(|k|) dk^3 / (4 pi k^2).
double modeWeight(const Lattice& l, double kk)
{
    const double k = std::sqrt(kk);
    return hitModelSpectrum(k / l.kp) * l.dk[0] * l.dk[1] * l.dk[2] /
           (2.0 * kTwoPi * kk);
}

// Per-shell realized and target energies over the reference set.
void shellSums(const TurbSpec& s, const Lattice& l, std::vector<double>& R,
               std::vector<double>& T)
{
    const int nS = shellOf(l, l.kmax * l.kmax) + 2;
    R.assign(std::size_t(nS), 0.0);
    T.assign(std::size_t(nS), 0.0);
    forReference(l, [&](const int m[3], const double k[3], double kk) {
        cplx a[3];
        drawMode(s.seed, m, k, kk, a);
        const int sh = shellOf(l, kk);
        R[std::size_t(sh)] +=
            0.5 * (std::norm(a[0]) + std::norm(a[1]) + std::norm(a[2]));
        T[std::size_t(sh)] += modeWeight(l, kk);
    });
}

} // namespace

double turbPeak(double lt)
{
    assert(lt > 0);
    return std::sqrt(kTwoPi) / lt;
}

std::vector<TurbMode> turbModes(const TurbSpec& s)
{
    const Lattice l = latticeOf(s);
    std::vector<double> R, T;
    shellSums(s, l, R, T);
    double tot = 0;
    for (const double t : T) tot += t;
    assert(tot > 0);
    const double G = std::sqrt(1.5 * s.urms * s.urms / tot);
    const int n[3] = {s.nx, s.ny, s.nz};
    std::vector<TurbMode> out;
    forReference(l, [&](const int m[3], const double k[3], double kk) {
        for (int d = 0; d < 3; ++d)
            if (2 * std::abs(m[d]) >= n[d]) return;
        cplx a[3];
        drawMode(s.seed, m, k, kk, a);
        const int sh = shellOf(l, kk);
        const double f = std::sqrt(T[std::size_t(sh)] / R[std::size_t(sh)]) * G;
        TurbMode tm{{m[0], m[1], m[2]}, {a[0] * f, a[1] * f, a[2] * f}};
        out.push_back(tm);
    });
    return out;
}

std::array<double, 3> turbPoint(const std::vector<TurbMode>& modes,
                                const TurbSpec& s, double x, double y,
                                double z)
{
    const double dk[3] = {kTwoPi / s.Lx, kTwoPi / s.Ly, kTwoPi / s.Lz};
    std::array<double, 3> u{0, 0, 0};
    for (const TurbMode& m : modes) {
        const double ph = m.m[0] * dk[0] * x + m.m[1] * dk[1] * y +
                          m.m[2] * dk[2] * z;
        const cplx e(std::cos(ph), std::sin(ph));
        for (int c = 0; c < 3; ++c) u[std::size_t(c)] += (m.uh[c] * e).real();
    }
    return u;
}

double turbLatticeL11(const TurbSpec& s)
{
    const Lattice l = latticeOf(s);
    double a = 0, b = 0;
    forReference(l, [&](const int*, const double*, double kk) {
        const double w = modeWeight(l, kk);
        a += w / std::sqrt(kk);
        b += w;
    });
    return 0.75 * std::numbers::pi * a / b;
}

TurbShells turbShells(const std::vector<TurbMode>& modes, const TurbSpec& s)
{
    const Lattice l = latticeOf(s);
    std::vector<double> R, T;
    shellSums(s, l, R, T);
    double tot = 0;
    for (const double t : T) tot += t;
    const double G2 = 1.5 * s.urms * s.urms / tot;
    TurbShells out;
    out.realized.assign(T.size(), 0.0);
    out.target.resize(T.size());
    for (std::size_t i = 0; i < T.size(); ++i) out.target[i] = T[i] * G2;
    for (const TurbMode& m : modes) {
        const double k[3] = {m.m[0] * l.dk[0], m.m[1] * l.dk[1],
                             m.m[2] * l.dk[2]};
        const double kk = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
        out.realized[std::size_t(shellOf(l, kk))] +=
            0.5 * (std::norm(m.uh[0]) + std::norm(m.uh[1]) + std::norm(m.uh[2]));
    }
    return out;
}

#if DNS_WITH_FFTW
std::array<std::vector<double>, 3> turbBox(const TurbSpec& s)
{
    const std::vector<TurbMode> modes = turbModes(s);
    const std::size_t n = std::size_t(s.nx) * s.ny * s.nz;
    const int nxh = s.nx / 2 + 1;
    const std::size_t nc = std::size_t(nxh) * s.ny * s.nz;
    double* real = fftw_alloc_real(n);
    fftw_complex* cp = fftw_alloc_complex(nc);
    const fftw_plan bwd =
        fftw_plan_dft_c2r_3d(s.nz, s.ny, s.nx, cp, real, FFTW_ESTIMATE);
    std::array<std::vector<double>, 3> out;
    for (int c = 0; c < 3; ++c) {
        for (std::size_t q = 0; q < nc; ++q) cp[q][0] = cp[q][1] = 0.0;
        for (const TurbMode& m : modes) {
            if (m.m[0] < 0) continue; // its conjugate partner (mx > 0) is there
            const int jy = (m.m[1] % s.ny + s.ny) % s.ny;
            const int kz = (m.m[2] % s.nz + s.nz) % s.nz;
            // Cell centres: u((i + 1/2) h) = sum uh e^{i k.h/2} e^{i 2 pi m i/n}.
            const double ph = std::numbers::pi *
                              (double(m.m[0]) / s.nx + double(m.m[1]) / s.ny +
                               double(m.m[2]) / s.nz);
            const cplx z = m.uh[c] * cplx(std::cos(ph), std::sin(ph));
            const std::size_t q =
                std::size_t(m.m[0]) + std::size_t(nxh) * (std::size_t(jy) +
                                                          std::size_t(s.ny) * kz);
            cp[q][0] = z.real();
            cp[q][1] = z.imag();
        }
        fftw_execute(bwd);
        out[std::size_t(c)].assign(real, real + n);
    }
    fftw_destroy_plan(bwd);
    fftw_free(real);
    fftw_free(cp);
    return out;
}
#endif

TurbNumbers turbNumbers(double urms, double lt, double nu, double dth)
{
    assert(urms > 0 && lt > 0 && nu > 0 && dth > 0);
    TurbNumbers t;
    t.ReT = urms * lt / nu;
    t.eta = lt * std::pow(t.ReT, -0.75);
    t.Ka = std::pow(urms, 1.5) / std::sqrt(lt / dth);
    t.Da = (lt / dth) / urms;
    t.tauT = lt / urms;
    return t;
}

double petersST(double urms, double lt, double lF)
{
    constexpr double a4 = 0.78, b1 = 2.0, b3 = 1.0;
    const double A = a4 * b3 * b3 / (2.0 * b1);
    const double r = A * lt / lF;
    return 1.0 - r + std::sqrt(r * r + a4 * b3 * b3 * urms * lt / lF);
}

double damkohlerST(double urms) { return 1.0 + urms; }

} // namespace dns
