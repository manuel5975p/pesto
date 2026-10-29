// The flame's synthetic turbulence box (dns/turb_ic.hpp, docs/chem.md "3D and
// turbulence"): solenoidal and Hermitian per mode, shell energies exactly the
// deterministic targets, u' exact, every x-plane mean zero, the FFT box equal
// to the direct mode sum, a coarse grid the spectral truncation of the fine
// one (same amplitudes, same field), seed determinism, the lattice L11 against
// l_t, and the report-only correlations.
#include "dns/turb_ic.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <map>
#include <tuple>

using namespace dns;
using dnstest::check;

namespace {

TurbSpec spec(int n, std::uint64_t seed = 7)
{
    TurbSpec s;
    s.urms = 2.0;
    s.lt = 4.0;
    s.Lx = s.Ly = s.Lz = 24.0; // 6 l_t: the run's box ratio
    s.nx = s.ny = s.nz = n;
    s.seed = seed;
    return s;
}

using Key = std::tuple<int, int, int>;

std::map<Key, const TurbMode*> index(const std::vector<TurbMode>& m)
{
    std::map<Key, const TurbMode*> out;
    for (const auto& t : m) out[{t.m[0], t.m[1], t.m[2]}] = &t;
    return out;
}

} // namespace

int main()
{
    const TurbSpec s = spec(64);
    const auto modes = turbModes(s);
    std::printf("modes on 64^3: %zu\n", modes.size());
    check(!modes.empty(), "the 64^3 grid holds modes");

    // Solenoidal, Hermitian, no (my, mz) = (0, 0).
    double divMax = 0, hermMax = 0;
    bool partners = true, noPlane = true;
    const auto idx = index(modes);
    const double dk = 2 * M_PI / 24.0;
    for (const auto& t : modes) {
        const double k[3] = {t.m[0] * dk, t.m[1] * dk, t.m[2] * dk};
        const double kn = std::sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2]);
        const std::complex<double> kd =
            k[0] * t.uh[0] + k[1] * t.uh[1] + k[2] * t.uh[2];
        const double un = std::sqrt(std::norm(t.uh[0]) + std::norm(t.uh[1]) +
                                    std::norm(t.uh[2]));
        divMax = std::max(divMax, std::abs(kd) / (kn * un));
        if (t.m[1] == 0 && t.m[2] == 0) noPlane = false;
        auto it = idx.find({-t.m[0], -t.m[1], -t.m[2]});
        if (it == idx.end()) {
            partners = false;
            continue;
        }
        for (int c = 0; c < 3; ++c)
            hermMax = std::max(hermMax,
                               std::abs(it->second->uh[c] - std::conj(t.uh[c])));
    }
    std::printf("  max |k.u|/(|k||u|) = %.3g, max |u(-m) - conj u(m)| = %.3g\n",
                divMax, hermMax);
    check(divMax <= 1e-12, "|k.uh| <= 1e-12 |k||uh| on every mode");
    check(partners && hermMax == 0, "uh(-m) == conj(uh(m)) exactly, every mode");
    check(noPlane, "no mode with my = mz = 0 (plane means vanish)");

    // Shell energies: realized == the deterministic targets.
    const TurbShells sh = turbShells(modes, s);
    double shellErr = 0, tot = 0;
    int populated = 0;
    for (std::size_t i = 0; i < sh.target.size(); ++i) {
        tot += sh.realized[i];
        if (sh.target[i] <= 0) continue;
        ++populated;
        shellErr = std::max(shellErr, std::abs(sh.realized[i] / sh.target[i] - 1));
    }
    std::printf("  %d populated shells, max |E_s/T_s - 1| = %.3g, sum = %.15g "
                "(1.5 u'^2 = %.15g)\n",
                populated, shellErr, tot, 1.5 * s.urms * s.urms);
    check(shellErr <= 1e-12, "every populated shell's energy equals its target "
                             "to 1e-12");
    check(std::abs(tot / (1.5 * s.urms * s.urms) - 1) <= 1e-12,
          "spectral energy 0.5 sum |uh|^2 = 1.5 u'^2 to 1e-12");

#if DNS_WITH_FFTW
    const auto box = turbBox(s);
    const std::size_t n = std::size_t(s.nx) * s.ny * s.nz;
    // u' exact.
    double e2 = 0;
    for (int c = 0; c < 3; ++c)
        for (std::size_t q = 0; q < n; ++q) e2 += box[c][q] * box[c][q];
    const double up = std::sqrt(e2 / (3.0 * n));
    std::printf("  u' = %.15g (target %.15g)\n", up, s.urms);
    check(std::abs(up / s.urms - 1) <= 1e-12, "box u' exact to 1e-12");

    // Every x-plane mean of u, v, w is zero.
    double pm = 0;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < s.nx; ++i) {
            double a = 0;
            for (int k = 0; k < s.nz; ++k)
                for (int j = 0; j < s.ny; ++j)
                    a += box[c][std::size_t(i) + std::size_t(s.nx) * (j + std::size_t(s.ny) * k)];
            pm = std::max(pm, std::abs(a) / (double(s.ny) * s.nz));
        }
    std::printf("  max |x-plane mean| = %.3g\n", pm);
    check(pm <= 1e-14 * s.urms, "every x-plane mean of u, v, w <= 1e-14 u'");

    // FFT == the direct sum at cell centres.
    auto direct = [&](const TurbSpec& sp, const std::vector<TurbMode>& md,
                      const std::array<std::vector<double>, 3>& bx) {
        double e = 0;
        const int pts[][3] = {{0, 0, 0}, {5, 17, 3}, {sp.nx - 1, 1, sp.nz - 2},
                              {sp.nx / 2, sp.ny / 3, sp.nz / 5}};
        for (const auto& p : pts) {
            const double hx = sp.Lx / sp.nx, hy = sp.Ly / sp.ny, hz = sp.Lz / sp.nz;
            const auto u = turbPoint(md, sp, (p[0] + 0.5) * hx, (p[1] % sp.ny + 0.5) * hy,
                                     (p[2] + 0.5) * hz);
            const std::size_t q = std::size_t(p[0]) +
                                  std::size_t(sp.nx) * (p[1] % sp.ny + std::size_t(sp.ny) * p[2]);
            for (int c = 0; c < 3; ++c) e = std::max(e, std::abs(u[c] - bx[c][q]));
        }
        return e;
    };
    const double eF = direct(s, modes, box);
    std::printf("  FFT vs direct sum (64^3): %.3g\n", eF);
    check(eF <= 1e-12 * s.urms, "FFT box == direct mode sum to 1e-12 u'");

    // Coarse grids: the SAME amplitudes on every shared mode (bitwise), and
    // their FFT box the direct sum of that truncated set; the truncated
    // field read at the coarse centres against the fine field there.
    for (const int nc : {32, 16}) {
        const TurbSpec sc = spec(nc);
        const auto mc = turbModes(sc);
        bool same = true;
        for (const auto& t : mc) {
            auto it = idx.find({t.m[0], t.m[1], t.m[2]});
            if (it == idx.end()) { same = false; continue; }
            for (int c = 0; c < 3; ++c)
                if (it->second->uh[c] != t.uh[c]) same = false;
        }
        check(same, std::format("{}^3: every mode's amplitude bitwise the 64^3 "
                                "one (spectral truncation)", nc));
        const auto bc = turbBox(sc);
        const double ec = direct(sc, mc, bc);
        check(ec <= 1e-12 * s.urms,
              std::format("{}^3: FFT box == direct sum of its modes to 1e-12 u'",
                          nc));
        // Truncation error against the fine field at the coarse centres.
        double et = 0;
        for (int k = 0; k < nc; k += 3)
            for (int j = 0; j < nc; j += 5)
                for (int i = 0; i < nc; i += 7) {
                    const double h = 24.0 / nc;
                    const auto u = turbPoint(modes, s, (i + 0.5) * h,
                                             (j + 0.5) * h, (k + 0.5) * h);
                    const std::size_t q =
                        std::size_t(i) + std::size_t(nc) * (j + std::size_t(nc) * k);
                    for (int c = 0; c < 3; ++c)
                        et = std::max(et, std::abs(u[c] - bc[c][q]));
                }
        std::printf("  %d^3 vs the 64^3 field at its centres: max %.3g (truncated "
                    "modes: %zu of %zu)\n",
                    nc, et, modes.size() - mc.size(), modes.size());
        if (nc == 32)
            check(et <= 1e-12 * s.urms, "32^3 (= n/2) equals the fine field at "
                                        "its centres to 1e-12 u'");
    }

    // Seed determinism.
    const auto box2 = turbBox(s);
    check(box2 == box, "same seed: the same bytes");
    const auto box3 = turbBox(spec(64, 8));
    check(box3[0] != box[0], "a different seed: a different field");
#endif

    // Lattice L11 vs l_t: converges as L/l_t grows.
    for (const double L : {24.0, 48.0, 96.0}) {
        TurbSpec sl = s;
        sl.Lx = sl.Ly = sl.Lz = L;
        const double l11 = turbLatticeL11(sl);
        std::printf("  L/l_t = %g: lattice L11/l_t = %.5f\n", L / s.lt, l11 / s.lt);
        if (L == 24.0) // the run's box ratio (L_y = L_z = 6 l_t)
            check(std::abs(l11 / s.lt - 1) <= 0.03,
                  "lattice L11 within 3% of l_t at L = 6 l_t (the run's box)");
        if (L == 96.0)
            check(std::abs(l11 / s.lt - 1) <= 0.002,
                  "lattice L11 within 0.2% of l_t at L = 24 l_t (converging)");
    }

    // Correlations and groups (hand values).
    const double p = petersST(2.0, 4.0, 1.0);
    const double ref = 1 - 0.78 + std::sqrt(0.78 * 0.78 + 0.78 * 8.0);
    std::printf("  Peters(u' 2, l 4) = %.12g, Damkoehler %.6g\n", p, damkohlerST(2.0));
    check(std::abs(p - ref) <= 1e-14, "Peters 1999 S_T/S_L at u' 2, l/l_F 4");
    const TurbNumbers tn = turbNumbers(4.0, 4.0, 0.72, 1.6);
    check(std::abs(tn.ReT - 16.0 / 0.72) <= 1e-12 &&
              std::abs(tn.Ka - 8.0 / std::sqrt(2.5)) <= 1e-12 &&
              std::abs(tn.Da - 2.5 / 4.0) <= 1e-15 && tn.tauT == 1.0 &&
              std::abs(tn.eta - 4.0 * std::pow(16.0 / 0.72, -0.75)) <= 1e-12,
          "turbNumbers: Re_t, Ka, Da, tau_t, eta");
    return dnstest::finish();
}
