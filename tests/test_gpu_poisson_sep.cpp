// SeparablePoisson + LmPressure on the device (masterplan real_bunsen S3,
// docs/la.md "Separable Poisson"). Modes:
//   sep    every kind triple (64 in 3D, 16 in 2D) vs the host FP64 separable
//          solve of the SAME float rhs (tests/sep_test_util.hpp, itself pinned
//          to dense LU by `poisson_sep`); manufactured smooth solutions; every
//          eigenmode of a small box (device eigenvalue); nz == 1 vs a
//          z-extruded box (per plane, bitwise); reruns, a second instance and
//          in-place solves byte-identical; keep/scale semantics bitwise; the
//          make() rejections by name.
//   apert  a solid annulus tube standing on the x-lo face of a 64 x 96 x 96 box
//          (x ND, y/z periodic and DD): PCG cold/warm iterations, closed faces
//          (projected flux divergence-free on every fluid cell, solid phi 0.0),
//          device operator vs the host FP64 one, the FP32 floor vs an FP64 CG
//          solve, determinism, enclosed-pocket / no-Dirichlet rejection, 2D.
//   perf   ms/solve at the final-case grid 384 x 176 x 176 (and 180), half
//          scale 192 x 88 x 88, and the real tube's PCG (meaningful under
//          simjob --bench).
#include "gpu_test_util.hpp"
#include "sep_test_util.hpp"

#include "dns/gpu/la/krylov.hpp"
#include "dns/gpu/lm_poisson.hpp"
#include "dns/gpu/poisson.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using gpu::Real;
using gpu::SepBox;

namespace {

std::string kname(SepKind k) { return sepKindName(k); }
std::string combo(const SepBox& b)
{
    return std::to_string(b.nx) + "x" + std::to_string(b.ny) + "x" + std::to_string(b.nz) + " " +
           kname(b.kx) + "/" + kname(b.ky) + "/" + kname(b.kz);
}

septest::Box hostBox(const SepBox& b)
{
    return septest::Box{{b.nx, b.ny, b.nz}, {b.hx, b.hy, b.hz}, {b.kx, b.ky, b.kz}};
}

template <class T>
gpu::DeviceBuffer<T> dev(const std::vector<T>& h)
{
    auto d = *gpu::DeviceBuffer<T>::make(h.size());
    (void)d.upload(h.data());
    return d;
}

template <class T>
std::vector<T> host(const gpu::DeviceBuffer<T>& d)
{
    std::vector<T> h(d.size());
    (void)d.download(h.data());
    return h;
}

double maxAbs(const std::vector<double>& v)
{
    double m = 0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

// Device solve of a float rhs; returns p.
std::vector<float> devSolve(gpu::SeparablePoisson& s, const std::vector<float>& rhs)
{
    auto r = dev(rhs);
    auto p = *gpu::DeviceScalar::make(rhs.size());
    s.solve(r.data(), p.data());
    return host(p);
}

// ------------------------------------------------------------------ sep ---

void kindTriples(int nx, int ny, int nz, double& worst, double& worstMms)
{
    std::mt19937_64 rng(std::uint64_t(nx * 7919 + ny * 31 + nz));
    std::uniform_real_distribution<double> u(-1, 1);
    for (SepKind kx : septest::kAllKinds)
        for (SepKind ky : septest::kAllKinds)
            for (SepKind kz : septest::kAllKinds)
            {
                if (nz == 1 && kz != SepKind::Periodic) continue;
                // ND needs an even axis (rejected by name: sepDeterminism).
                if ((kx == SepKind::ND && nx % 2) || (ky == SepKind::ND && ny % 2) ||
                    (kz == SepKind::ND && nz % 2))
                    continue;
                const SepBox b{nx, ny, nz, 0.37, 0.29, 0.41, kx, ky, kz};
                auto s = gpu::SeparablePoisson::make(b);
                if (!s)
                {
                    check(false, combo(b) + ": make: " + s.error());
                    continue;
                }
                // (1) random float rhs vs the host FP64 solve of the same rhs.
                std::vector<float> rf(b.size());
                for (auto& v : rf) v = float(u(rng));
                const std::vector<double> rd(rf.begin(), rf.end());
                const auto ref = septest::sepSolve(hostBox(b), rd);
                const auto got = devSolve(*s, rf);
                double err = 0;
                for (std::size_t c = 0; c < b.size(); ++c) err = std::max(err, std::abs(got[c] - ref[c]));
                const double rel = err / maxAbs(ref);
                worst = std::max(worst, rel);
                // (2) manufactured: smooth p (not an eigenfunction), rhs = L p
                // in FP64, rounded once; p reproduced up to the zero mode.
                std::vector<double> p(b.size());
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i)
                        {
                            const double x = (i + 0.5) / nx, y = (j + 0.5) / ny, z = (k + 0.5) / nz;
                            p[std::size_t(i) + std::size_t(nx) * (j + std::size_t(ny) * k)] =
                                std::exp(-3 * (x - 0.4) * (x - 0.4)) * std::cos(2.1 * y + 0.3) *
                                    (1 + 0.5 * std::sin(1.7 * z)) +
                                0.2 * x * y;
                        }
                const auto lp = septest::laplacian(hostBox(b), p);
                const std::vector<float> lf(lp.begin(), lp.end());
                const auto pm = devSolve(*s, lf);
                double mg = 0, mr = 0;
                if (b.singular())
                {
                    for (std::size_t c = 0; c < b.size(); ++c)
                    {
                        mg += pm[c];
                        mr += p[c];
                    }
                    mg /= double(b.size());
                    mr /= double(b.size());
                }
                double e2 = 0;
                for (std::size_t c = 0; c < b.size(); ++c)
                    e2 = std::max(e2, std::abs((pm[c] - mg) - (p[c] - mr)));
                worstMms = std::max(worstMms, e2 / maxAbs(p));
                if (rel > 1e-6 || !(e2 / maxAbs(p) <= 1e-5))
                    check(false, combo(b) + " random rel " + std::to_string(rel) + " mms rel " +
                                     std::to_string(e2 / maxAbs(p)));
            }
}

void sepAllKinds()
{
    std::printf("(1) every kind triple: device FP32 vs host FP64 on the same rhs\n");
    struct G
    {
        int nx, ny, nz;
    };
    for (G g : {G{24, 20, 16}, G{18, 15, 9}, G{8, 6, 12}, G{24, 20, 1}, G{16, 9, 1}})
    {
        double worst = 0, mms = 0;
        kindTriples(g.nx, g.ny, g.nz, worst, mms);
        std::printf("  %dx%dx%d: random rhs max rel %.2e, manufactured max rel %.2e\n", g.nx, g.ny,
                    g.nz, worst, mms);
        check(worst <= 1e-6 && mms <= 1e-5,
              std::to_string(g.nx) + "x" + std::to_string(g.ny) + "x" + std::to_string(g.nz) +
                  ": every kind triple to 1e-6 (random, vs FP64) and 1e-5 (manufactured)");
    }
}

// Every eigenmode of a small box: the device eigenvalue -<f,f>/<p,f> vs sepEigen.
void sepEigenmodes()
{
    std::printf("(2) every eigenmode of 8x6x4 boxes (device eigenvalue)\n");
    const SepKind combos[][3] = {{SepKind::ND, SepKind::DD, SepKind::Periodic},
                                 {SepKind::ND, SepKind::Periodic, SepKind::DD},
                                 {SepKind::NN, SepKind::DD, SepKind::ND},
                                 {SepKind::Periodic, SepKind::NN, SepKind::Periodic}};
    for (const auto& kk : combos)
    {
        const SepBox b{8, 6, 4, 0.5, 0.7, 0.3, kk[0], kk[1], kk[2]};
        auto s = *gpu::SeparablePoisson::make(b);
        double worst = 0, worstShape = 0;
        for (int k3 = 0; k3 < b.nz; ++k3)
            for (int k2 = 0; k2 < b.ny; ++k2)
                for (int k1 = 0; k1 < b.nx; ++k1)
                {
                    const double lam = sepEigen(b.kx, b.nx, b.hx, k1) + sepEigen(b.ky, b.ny, b.hy, k2) +
                                       sepEigen(b.kz, b.nz, b.hz, k3);
                    if (lam == 0) continue;
                    std::vector<float> f(b.size());
                    std::vector<double> fd(b.size());
                    for (int k = 0; k < b.nz; ++k)
                        for (int j = 0; j < b.ny; ++j)
                            for (int i = 0; i < b.nx; ++i)
                            {
                                const std::size_t c = std::size_t(i) + 8 * (j + 6 * std::size_t(k));
                                fd[c] = sepBasis(b.kx, b.nx, k1, i) * sepBasis(b.ky, b.ny, k2, j) *
                                        sepBasis(b.kz, b.nz, k3, k);
                                f[c] = float(fd[c]);
                            }
                    const auto p = devSolve(s, f);
                    double pf = 0, ff = 0;
                    for (std::size_t c = 0; c < b.size(); ++c)
                    {
                        pf += p[c] * double(f[c]);
                        ff += double(f[c]) * f[c];
                    }
                    worst = std::max(worst, std::abs(-ff / pf / lam - 1));
                    double sh = 0;
                    for (std::size_t c = 0; c < b.size(); ++c)
                        sh = std::max(sh, std::abs(p[c] + fd[c] / lam));
                    worstShape = std::max(worstShape, sh * lam / maxAbs(fd));
                }
        std::printf("  %s: max |lambda_dev/lambda - 1| %.2e, mode shape %.2e\n", combo(b).c_str(), worst,
                    worstShape);
        check(worst <= 1e-6 && worstShape <= 1e-6,
              combo(b) + ": every eigenmode solved with its eigenvalue to 1e-6");
    }
}

// nz == 1 vs a z-extruded box (z periodic, z-invariant rhs): per plane.
void sep2dExtrude()
{
    std::printf("(3) 2D (nz 1) vs the z-extruded box (nz 8, periodic z)\n");
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> u(-1, 1);
    for (SepKind ky : {SepKind::Periodic, SepKind::DD})
    {
        const SepBox b2{48, 40, 1, 0.25, 0.25, 0.25, SepKind::ND, ky, SepKind::Periodic};
        SepBox b3 = b2;
        b3.nz = 8;
        auto s2 = *gpu::SeparablePoisson::make(b2);
        auto s3 = *gpu::SeparablePoisson::make(b3);
        std::vector<float> r2(b2.size()), r3(b3.size());
        for (auto& v : r2) v = float(u(rng));
        for (int k = 0; k < 8; ++k) std::copy(r2.begin(), r2.end(), r3.begin() + std::ptrdiff_t(k * b2.size()));
        const auto p2 = devSolve(s2, r2);
        const auto p3 = devSolve(s3, r3);
        bool bits = true;
        double d = 0, sc = 0;
        for (int k = 0; k < 8; ++k)
            for (std::size_t c = 0; c < b2.size(); ++c)
            {
                const float a = p3[k * b2.size() + c];
                bits &= std::memcmp(&a, &p2[c], sizeof a) == 0;
                d = std::max(d, double(std::abs(a - p2[c])));
                sc = std::max(sc, double(std::abs(p2[c])));
            }
        std::printf("  ND/%s: per plane %s (max diff %.2e of %.2e)\n", kname(ky).c_str(),
                    bits ? "BITWISE" : "not bitwise", d, sc);
        check(d <= 1e-6 * sc, "ND/" + kname(ky) + ": z-extruded == 2D per plane to 1e-6");
        check(bits, "ND/" + kname(ky) + ": z-extruded == 2D per plane bitwise");
    }
}

// Reruns, a second instance, in-place; keep/scale; rejections.
void sepDeterminism()
{
    std::printf("(4) determinism, aliasing, keep/scale, rejections\n");
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> u(-1, 1);
    const SepBox b{40, 36, 28, 0.1, 0.1, 0.1, SepKind::ND, SepKind::DD, SepKind::Periodic};
    auto s1 = *gpu::SeparablePoisson::make(b);
    auto s2 = *gpu::SeparablePoisson::make(b);
    std::vector<float> rf(b.size());
    for (auto& v : rf) v = float(u(rng));
    const auto a = devSolve(s1, rf);
    const auto a2 = devSolve(s1, rf);
    const auto a3 = devSolve(s2, rf);
    auto d = dev(rf);
    s1.solve(d.data(), d.data());
    const auto a4 = host(d);
    const std::size_t bytes = a.size() * sizeof(float);
    check(std::memcmp(a.data(), a2.data(), bytes) == 0, "rerun byte-identical");
    check(std::memcmp(a.data(), a3.data(), bytes) == 0, "second instance byte-identical");
    check(std::memcmp(a.data(), a4.data(), bytes) == 0, "in-place (rhs == p) byte-identical");

    std::vector<std::uint8_t> keep(b.size());
    std::vector<float> masked(b.size());
    for (std::size_t c = 0; c < b.size(); ++c)
    {
        keep[c] = (c * 2654435761u >> 7) % 5 != 0;
        masked[c] = keep[c] ? rf[c] : 0.0f;
    }
    auto dk = dev(keep);
    auto dr = dev(rf);
    auto dp = *gpu::DeviceScalar::make(b.size());
    s1.solve(dr.data(), dp.data(), Real(-1), dk.data());
    const auto km = host(dp);
    const auto ref = devSolve(s1, masked);
    bool ok = true;
    for (std::size_t c = 0; c < b.size(); ++c)
    {
        const float want = keep[c] ? -ref[c] : 0.0f;
        ok &= std::memcmp(&km[c], &want, sizeof want) == 0;
    }
    check(ok, "solve(rhs, p, -1, keep) == -(solve(rhs keep)) keep, bitwise");
    check(s1.status() == hipSuccess, "status hipSuccess");

    const SepBox singular{20, 16, 12, 0.2, 0.2, 0.2, SepKind::NN, SepKind::Periodic, SepKind::NN};
    auto ss = *gpu::SeparablePoisson::make(singular);
    std::vector<float> rs(singular.size());
    for (auto& v : rs) v = float(u(rng));
    const auto ps = devSolve(ss, rs);
    double mean = 0, mx = 0;
    for (float v : ps)
    {
        mean += v;
        mx = std::max(mx, double(std::abs(v)));
    }
    mean /= double(ps.size());
    std::printf("  singular NN/P/NN: mean %.2e of max %.2e\n", mean, mx);
    check(std::abs(mean) <= 1e-6 * mx, "singular box: zero-mean solution (rhs mean dropped)");

    auto rej = [](SepBox x, const char* needle, const char* what) {
        auto r = gpu::SeparablePoisson::make(x);
        check(!r && r.error().find(needle) != std::string::npos,
              std::string("rejected: ") + what + (r ? "" : " (" + r.error() + ")"));
    };
    rej({9, 8, 4, 1, 1, 1, SepKind::ND, SepKind::Periodic, SepKind::Periodic}, "even", "odd ND axis");
    rej({8, 8, 1, 1, 1, 1, SepKind::ND, SepKind::DD, SepKind::DD}, "2D", "nz 1 with DD z");
    rej({1, 8, 4, 1, 1, 1, SepKind::NN, SepKind::DD, SepKind::DD}, ">= 2", "nx 1");
    rej({8, 8, 4, 0, 1, 1, SepKind::NN, SepKind::DD, SepKind::DD}, "> 0", "h 0");
}

// ---------------------------------------------------------------- apert ---

struct Tube
{
    double ri, ro, xr; // radii and rim height, in cells
};

// Annulus ri <= r <= ro, x < xr, axis at the cross-section centre.
std::vector<std::uint8_t> annulus(const SepBox& b, const Tube& t)
{
    std::vector<std::uint8_t> s(b.size(), 0);
    const double cy = 0.5 * b.ny, cz = 0.5 * b.nz;
    for (int k = 0; k < b.nz; ++k)
        for (int j = 0; j < b.ny; ++j)
            for (int i = 0; i < b.nx; ++i)
            {
                const double y = j + 0.5 - cy, z = b.nz == 1 ? 0.0 : k + 0.5 - cz;
                const double r = b.nz == 1 ? std::abs(y) : std::sqrt(y * y + z * z);
                if (i + 0.5 < t.xr && r >= t.ri && r <= t.ro)
                    s[std::size_t(i) + std::size_t(b.nx) * (j + std::size_t(b.ny) * k)] = 1;
            }
    return s;
}

// Host face-flux model of the apertured operator: open(c, axis, side) and the
// neighbour index, from the device face codes.
struct HostAp
{
    SepBox b;
    std::vector<std::uint8_t> code;
    std::size_t st[3];
    int n[3];
    SepKind kind[3];
    double h[3];
    HostAp(const SepBox& bx, std::vector<std::uint8_t> c) : b(bx), code(std::move(c))
    {
        st[0] = 1;
        st[1] = std::size_t(b.nx);
        st[2] = std::size_t(b.nx) * b.ny;
        n[0] = b.nx, n[1] = b.ny, n[2] = b.nz;
        kind[0] = b.kx, kind[1] = b.ky, kind[2] = b.kz;
        h[0] = b.hx, h[1] = b.hy, h[2] = b.hz;
    }
    int q(std::size_t c, int a) const { return int((c / st[a]) % std::size_t(n[a])); }
    std::size_t hiNb(std::size_t c, int a) const
    {
        return q(c, a) < n[a] - 1 ? c + st[a] : c - st[a] * std::size_t(n[a] - 1);
    }
    // Hi-face gradient (phi_nb - phi_c)/h, Dirichlet hi boundary (0 - phi_c)/(h/2),
    // lo Dirichlet boundary: (phi_c - 0)/(h/2) (outward-lo sign handled by div).
    // div of a face field F (F_hi[c][a], F_lo boundary only).
    std::vector<double> apply(const std::vector<double>& x) const
    {
        std::vector<double> y(b.size());
        const std::uint8_t lo[3] = {gpu::kLmFaceXLo, gpu::kLmFaceYLo, gpu::kLmFaceZLo};
        const std::uint8_t hi[3] = {gpu::kLmFaceXHi, gpu::kLmFaceYHi, gpu::kLmFaceZHi};
        for (std::size_t c = 0; c < b.size(); ++c)
        {
            if (code[c] & gpu::kLmSolid)
            {
                y[c] = x[c];
                continue;
            }
            double s = 0;
            for (int a = 0; a < 3; ++a)
            {
                if (n[a] == 1) continue;
                const int qq = q(c, a);
                double t = 0;
                if (code[c] & lo[a]) t += x[c] - x[qq > 0 ? c - st[a] : c + st[a] * std::size_t(n[a] - 1)];
                else if (qq == 0 && sepLoDirichlet(kind[a])) t += 2 * x[c];
                if (code[c] & hi[a]) t += x[c] - x[hiNb(c, a)];
                else if (qq == n[a] - 1 && sepHiDirichlet(kind[a])) t += 2 * x[c];
                s += t / (h[a] * h[a]);
            }
            y[c] = s;
        }
        return y;
    }
};

// Face fluxes on the hi face of every cell per axis (+ the lo boundary face
// of a non-periodic axis); closed faces hold 0 and are never written.
struct Faces
{
    std::vector<double> hi[3], loB[3];
};

// m* random on open faces and on the x-hi outflow (Dirichlet) faces, 0 on
// closed / Neumann faces. div m* on fluid cells.
std::vector<double> divergence(const HostAp& ap, const Faces& f)
{
    std::vector<double> d(ap.b.size(), 0.0);
    for (std::size_t c = 0; c < ap.b.size(); ++c)
    {
        if (ap.code[c] & gpu::kLmSolid) continue;
        double s = 0;
        for (int a = 0; a < 3; ++a)
        {
            if (ap.n[a] == 1) continue;
            const int qq = ap.q(c, a);
            const std::size_t lo = qq > 0 ? c - ap.st[a] : c + ap.st[a] * std::size_t(ap.n[a] - 1);
            const bool per = ap.kind[a] == SepKind::Periodic;
            const double fLo = (qq > 0 || per) ? f.hi[a][lo] : f.loB[a][c];
            s += (f.hi[a][c] - fLo) / ap.h[a];
        }
        d[c] = s;
    }
    return d;
}

void apertCase(const SepBox& b, const Tube& t, bool full)
{
    const auto solid = annulus(b, t);
    auto lm = gpu::LmPressure::make(b, solid, {1e-6, 400});
    if (!lm)
    {
        check(false, combo(b) + ": make: " + lm.error());
        return;
    }
    std::vector<std::uint8_t> code(b.size());
    (void)hipMemcpy(code.data(), lm->faceCodes(), b.size(), hipMemcpyDeviceToHost);
    HostAp ap(b, code);
    std::printf("  %s, tube ri %.0f ro %.0f rim %.0f cells: %zu solid cells\n", combo(b).c_str(), t.ri,
                t.ro, t.xr, lm->solidCells());

    // (a) the device operator == the host FP64 one (face codes, BC rules).
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(-1, 1);
    {
        std::vector<float> x(b.size());
        for (auto& v : x) v = float(u(rng));
        auto dx = dev(x);
        auto dy = *gpu::DeviceScalar::make(b.size());
        lm->apply(dx.data(), dy.data());
        const auto y = host(dy);
        const auto yr = ap.apply(std::vector<double>(x.begin(), x.end()));
        double e = 0;
        for (std::size_t c = 0; c < b.size(); ++c) e = std::max(e, std::abs(y[c] - yr[c]));
        check(e <= 1e-6 * maxAbs(yr), "device apertured operator == host FP64 operator (1e-6)");
    }

    // (b) projection: m* on open faces + outflow faces, rhs = div m*, solve
    // L_ap phi = rhs, m = m* - G phi on the open / Dirichlet faces only.
    Faces f;
    for (int a = 0; a < 3; ++a)
    {
        f.hi[a].assign(b.size(), 0.0);
        f.loB[a].assign(b.size(), 0.0);
    }
    const std::uint8_t hiBit[3] = {gpu::kLmFaceXHi, gpu::kLmFaceYHi, gpu::kLmFaceZHi};
    for (std::size_t c = 0; c < b.size(); ++c)
    {
        if (code[c] & gpu::kLmSolid) continue;
        const double yy = (ap.q(c, 1) + 0.5) / b.ny - 0.5;
        for (int a = 0; a < 3; ++a)
        {
            if (ap.n[a] == 1) continue;
            const bool open = code[c] & hiBit[a];
            const bool dirHi = ap.q(c, a) == ap.n[a] - 1 && sepHiDirichlet(ap.kind[a]);
            if (open || dirHi) f.hi[a][c] = (a == 0 ? 1.0 + 0.5 * std::cos(6 * yy) : 0.0) + 0.3 * u(rng);
            if (ap.q(c, a) == 0 && sepLoDirichlet(ap.kind[a])) f.loB[a][c] = 0.3 * u(rng);
        }
    }
    const auto rhsD = divergence(ap, f);
    const std::vector<float> rhs(rhsD.begin(), rhsD.end());
    auto dr = dev(rhs);
    auto dphi = *gpu::DeviceScalar::make(b.size());
    auto t0 = std::chrono::steady_clock::now();
    auto cold = lm->solve(dr.data(), dphi.data(), false);
    (void)hipDeviceSynchronize();
    const double msCold = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!cold)
    {
        check(false, "solve: " + cold.error());
        return;
    }
    const auto phi = host(dphi);
    std::printf("    cold: %d iterations to |r| <= 1e-6 |b| (rel %.2e, floor est %.2e), %.1f ms\n",
                cold->iterations, cold->relResidual, cold->floorRel, msCold);
    check(cold->converged, "cold PCG converged to 1e-6");
    bool solidZero = true;
    for (std::size_t c = 0; c < b.size(); ++c)
        if (code[c] & gpu::kLmSolid) solidZero &= phi[c] == 0.0f;
    check(solidZero, "phi == 0.0 exactly on every solid cell");
    // Correct the face fluxes: open faces (phi_nb - phi_c)/h, Dirichlet hi
    // (0 - phi_c)/(h/2), Dirichlet lo (phi_c - 0)/(h/2); closed faces untouched.
    Faces m = f;
    std::size_t closedTouched = 0;
    for (std::size_t c = 0; c < b.size(); ++c)
    {
        if (code[c] & gpu::kLmSolid) continue;
        for (int a = 0; a < 3; ++a)
        {
            if (ap.n[a] == 1) continue;
            if (code[c] & hiBit[a]) m.hi[a][c] -= (phi[ap.hiNb(c, a)] - phi[c]) / ap.h[a];
            else if (ap.q(c, a) == ap.n[a] - 1 && sepHiDirichlet(ap.kind[a]))
                m.hi[a][c] -= (0.0 - phi[c]) / (0.5 * ap.h[a]);
            if (ap.q(c, a) == 0 && sepLoDirichlet(ap.kind[a])) m.loB[a][c] -= (phi[c] - 0.0) / (0.5 * ap.h[a]);
        }
    }
    for (int a = 0; a < 3; ++a)
        for (std::size_t c = 0; c < b.size(); ++c)
            if (f.hi[a][c] == 0.0 && m.hi[a][c] != 0.0) ++closedTouched;
    const auto dm = divergence(ap, m);
    const double divRel = maxAbs(dm) / maxAbs(rhsD);
    std::printf("    projected max|div m| / max|div m*| = %.2e; closed faces written: %zu\n", divRel,
                closedTouched);
    check(closedTouched == 0, "closed faces carry exactly 0 flux after the correction");
    check(divRel <= 1e-4, "projected flux divergence-free on fluid cells (1e-4 of div m*)");

    // (c) warm starts: rhs perturbed by eps (a smooth relative change).
    for (double eps : {1e-2, 1e-3})
    {
        std::vector<float> r2(b.size());
        for (std::size_t c = 0; c < b.size(); ++c)
            r2[c] = float(rhsD[c] * (1 + eps * std::sin(0.05 * double(ap.q(c, 0)) + 0.1 * ap.q(c, 1))));
        auto d2 = dev(r2);
        auto dw = dev(phi);
        t0 = std::chrono::steady_clock::now();
        auto warm = lm->solve(d2.data(), dw.data(), true);
        (void)hipDeviceSynchronize();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("    warm (rhs change %.0e): %d iterations, %.1f ms\n", eps, warm ? warm->iterations : -1, ms);
        check(warm && warm->converged, "warm PCG converged");
    }
    if (!full) return;

    // (d) determinism: a second cold solve byte-identical.
    {
        auto dphi2 = *gpu::DeviceScalar::make(b.size());
        (void)lm->solve(dr.data(), dphi2.data(), false);
        const auto phi2 = host(dphi2);
        check(std::memcmp(phi.data(), phi2.data(), phi.size() * sizeof(float)) == 0,
              "PCG rerun byte-identical");
    }

    // (e) FP32 floor vs FP64: plain CG in FP64 on the FP64 operator.
    {
        std::vector<double> bd(b.size());
        for (std::size_t c = 0; c < b.size(); ++c) bd[c] = (code[c] & gpu::kLmSolid) ? 0.0 : -double(rhs[c]);
        auto db = dev(bd);
        auto dx = *gpu::DeviceBuffer<double>::make(b.size());
        (void)hipMemset(dx.data(), 0, b.size() * sizeof(double));
        auto k64 = *gpu::la::Krylov<double>::make(b.size(), 0);
        const gpu::la::LinOp<double> A = [&](const double* x, double* y) { lm->apply(x, y); };
        gpu::la::KrylovOptions o;
        o.maxIter = 20000;
        o.relTol = 1e-12;
        auto r64 = k64.pcg(A, {}, db.data(), dx.data(), o);
        const auto x64 = host(dx);
        // phi solves L phi = rhs, x64 solves A x = -rhs: phi == x64.
        auto trueRes = [&](const std::vector<double>& x) {
            const auto ax = ap.apply(x);
            double r = 0, bn = 0;
            for (std::size_t c = 0; c < b.size(); ++c)
            {
                r += (ax[c] - bd[c]) * (ax[c] - bd[c]);
                bn += bd[c] * bd[c];
            }
            return std::sqrt(r / bn);
        };
        // The FP32 storage floor: the true residual of x64 ROUNDED to float.
        std::vector<double> x64r(b.size());
        for (std::size_t c = 0; c < b.size(); ++c) x64r[c] = double(float(x64[c]));
        const double storeFloor = trueRes(x64r);
        const double sc = maxAbs(x64);
        std::printf("    FP64 CG: %d iterations, rel %.1e; FP32 storage floor (true rel residual of "
                    "float(x64)) %.2e\n",
                    r64 ? r64->iterations : -1, r64 ? r64->relResidual : -1.0, storeFloor);
        check(r64 && r64->converged, "FP64 CG reference converged (1e-12)");
        for (double tol : {1e-3, 1e-4, 1e-5, 1e-6, 1e-8})
        {
            auto lmT = *gpu::LmPressure::make(b, solid, {tol, 200});
            auto dt = *gpu::DeviceScalar::make(b.size());
            auto rt = lmT.solve(dr.data(), dt.data(), false);
            const auto xt = host(dt);
            const std::vector<double> xd(xt.begin(), xt.end());
            double e = 0;
            for (std::size_t c = 0; c < b.size(); ++c) e = std::max(e, std::abs(xd[c] - x64[c]));
            const double tr = trueRes(xd);
            std::printf("    FP32 PCG tol %.0e: %3d it, recursive %.2e, TRUE %.2e, max|x - x64|/max "
                        "%.2e, floor est %.2e\n",
                        tol, rt ? rt->iterations : -1, rt ? rt->relResidual : -1.0, tr, e / sc,
                        rt ? rt->floorRel : -1.0);
            if (tol == 1e-6)
            {
                check(rt && rt->converged && e <= 1e-6 * sc, "FP32 PCG tol 1e-6: max|x - x64| <= 1e-6 of max");
                // Measured 3.3-3.8x: the iterate's own rounding on top of storage.
                check(tr <= 5 * storeFloor, "FP32 PCG tol 1e-6: true residual <= 5x the storage floor");
                check(rt && rt->floorRel >= tr, "floor estimate bounds the true residual");
            }
        }
    }
}

void apertRejections()
{
    // A closed solid shell around a fluid pocket (x ND, y/z periodic).
    const SepBox b{16, 16, 16, 1, 1, 1, SepKind::ND, SepKind::Periodic, SepKind::Periodic};
    std::vector<std::uint8_t> s(b.size(), 0);
    for (int k = 4; k <= 9; ++k)
        for (int j = 4; j <= 9; ++j)
            for (int i = 4; i <= 9; ++i)
                if (i == 4 || i == 9 || j == 4 || j == 9 || k == 4 || k == 9)
                    s[std::size_t(i) + 16 * (j + 16 * std::size_t(k))] = 1;
    auto r = gpu::LmPressure::make(b, s);
    check(!r && r.error().find("enclosed") != std::string::npos,
          "fluid pocket enclosed by solid rejected by name" + (r ? std::string() : " (" + r.error() + ")"));
    SepBox nb = b;
    nb.kx = SepKind::NN;
    auto r2 = gpu::LmPressure::make(nb, annulus(nb, {3, 5, 6}));
    check(!r2 && r2.error().find("Dirichlet") != std::string::npos,
          "apertured solve without a Dirichlet face rejected by name");
    auto r3 = gpu::LmPressure::make(b, std::vector<std::uint8_t>(7, 0));
    check(!r3, "solid map size mismatch rejected");
}

// No solid: the direct path equals SeparablePoisson bitwise.
void apertDirect()
{
    const SepBox b{32, 24, 24, 0.5, 0.5, 0.5, SepKind::ND, SepKind::DD, SepKind::DD};
    auto lm = *gpu::LmPressure::make(b, {});
    auto sp = *gpu::SeparablePoisson::make(b);
    std::vector<float> r(b.size());
    std::mt19937_64 rng(9);
    std::uniform_real_distribution<double> u(-1, 1);
    for (auto& v : r) v = float(u(rng));
    auto dr = dev(r);
    auto dp = *gpu::DeviceScalar::make(b.size());
    auto res = lm.solve(dr.data(), dp.data(), false);
    const auto a = host(dp);
    const auto ref = devSolve(sp, r);
    check(res && res->direct && res->iterations == 0 &&
              std::memcmp(a.data(), ref.data(), a.size() * sizeof(float)) == 0,
          "no solid: one direct spectral solve, == SeparablePoisson bitwise");
}

// --------------------------------------------------- cross-process hash ---

std::uint64_t fnv(const void* p, std::size_t n, std::uint64_t h = 1469598103934665603ull)
{
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// Prints one hash per solve: a fresh process must reproduce every byte (no
// uninitialised device memory reaches a result, hipFFT plans deterministic).
void hashes()
{
    std::mt19937_64 rng(21);
    std::uniform_real_distribution<double> u(-1, 1);
    const SepBox boxes[] = {{64, 48, 40, 0.1, 0.1, 0.1, SepKind::ND, SepKind::DD, SepKind::DD},
                            {64, 48, 40, 0.1, 0.1, 0.1, SepKind::ND, SepKind::Periodic, SepKind::Periodic},
                            {30, 28, 22, 0.1, 0.1, 0.1, SepKind::NN, SepKind::Periodic, SepKind::NN},
                            {96, 64, 1, 0.1, 0.1, 0.1, SepKind::ND, SepKind::DD, SepKind::Periodic}};
    for (const SepBox& b : boxes)
    {
        auto s = *gpu::SeparablePoisson::make(b);
        std::vector<float> r(b.size());
        for (auto& v : r) v = float(u(rng));
        const auto p = devSolve(s, r);
        std::printf("hash sep %s %016llx\n", combo(b).c_str(),
                    (unsigned long long)fnv(p.data(), p.size() * sizeof(float)));
    }
    const SepBox b{64, 96, 96, 1, 1, 1, SepKind::ND, SepKind::DD, SepKind::DD};
    auto lm = *gpu::LmPressure::make(b, annulus(b, {16, 20, 24}));
    std::vector<float> r(b.size());
    for (auto& v : r) v = float(u(rng));
    auto dr = dev(r);
    auto dp = *gpu::DeviceScalar::make(b.size());
    const auto c = lm.solve(dr.data(), dp.data(), false);
    const auto p = host(dp);
    for (auto& v : r) v *= 1.001f;
    auto dr2 = dev(r);
    const auto w = lm.solve(dr2.data(), dp.data(), true);
    const auto p2 = host(dp);
    std::printf("hash pcg cold %d it %016llx warm %d it %016llx\n", c ? c->iterations : -1,
                (unsigned long long)fnv(p.data(), p.size() * sizeof(float)), w ? w->iterations : -1,
                (unsigned long long)fnv(p2.data(), p2.size() * sizeof(float)));
}

// Runs this binary's `hash` mode in two fresh processes; equal, non-empty output.
void crossProcess()
{
    std::error_code ec;
    const std::string self = std::filesystem::read_symlink("/proc/self/exe", ec).string();
    auto run = [&] {
        std::string out;
        if (ec) return out;
        FILE* f = popen(("'" + self + "' hash").c_str(), "r");
        if (!f) return out;
        char buf[256];
        while (std::fgets(buf, sizeof buf, f)) out += buf;
        if (pclose(f) != 0) out += "rc != 0";
        return out;
    };
    const std::string a = run(), b = run();
    std::printf("%s", a.c_str());
    check(!a.empty() && a.find("rc != 0") == std::string::npos && a == b,
          "two fresh processes: every solve byte-identical (separable + PCG cold/warm)");
}

// ----------------------------------------------------------------- perf ---

double msPerSolve(gpu::SeparablePoisson& s, std::size_t n, int reps)
{
    auto r = *gpu::DeviceScalar::make(n);
    auto p = *gpu::DeviceScalar::make(n);
    (void)hipMemset(r.data(), 0, n * sizeof(Real));
    for (int i = 0; i < 3; ++i) s.solve(r.data(), p.data());
    (void)hipDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) s.solve(r.data(), p.data());
    (void)hipDeviceSynchronize();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

void perf()
{
    struct P
    {
        int nx, ny, nz;
        SepKind ky;
    };
    std::printf("(perf) SeparablePoisson ms/solve (x ND)\n");
    for (P p : {P{384, 176, 176, SepKind::Periodic}, P{384, 176, 176, SepKind::DD},
                P{384, 180, 180, SepKind::DD}, P{192, 88, 88, SepKind::Periodic},
                P{192, 88, 88, SepKind::DD}, P{192, 90, 90, SepKind::DD}})
    {
        const SepBox b{p.nx, p.ny, p.nz, 1, 1, 1, SepKind::ND, p.ky, p.ky};
        auto s = gpu::SeparablePoisson::make(b);
        if (!s)
        {
            check(false, s.error());
            continue;
        }
        const double ms = msPerSolve(*s, b.size(), 20);
        std::printf("  %s: %.3f ms/solve (%.2f ns/cell)\n", combo(b).c_str(), ms, ms * 1e6 / double(b.size()));
    }
    std::printf("(perf) LmPressure, the final-case tube (ID 57, lip 6, 30 cells below the rim)\n");
    for (P p : {P{384, 176, 176, SepKind::DD}, P{192, 88, 88, SepKind::DD}})
    {
        const double sc = p.nx / 384.0;
        const SepBox b{p.nx, p.ny, p.nz, 1, 1, 1, SepKind::ND, p.ky, p.ky};
        apertCase(b, {28.5 * sc, 34.5 * sc, 30 * sc}, false);
        // ms per operator apply
        auto lm = *gpu::LmPressure::make(b, annulus(b, {28.5 * sc, 34.5 * sc, 30 * sc}));
        auto x = *gpu::DeviceScalar::make(b.size());
        auto y = *gpu::DeviceScalar::make(b.size());
        (void)hipMemset(x.data(), 0, b.size() * sizeof(Real));
        lm.apply(x.data(), y.data());
        (void)hipDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) lm.apply(x.data(), y.data());
        (void)hipDeviceSynchronize();
        std::printf("    apply: %.3f ms\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20);
    }
}

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "sep";
    if (mode == "sep")
    {
        sepAllKinds();
        sepEigenmodes();
        sep2dExtrude();
        sepDeterminism();
    }
    else if (mode == "apert")
    {
        std::printf("(apert) annulus tube on the x-lo face\n");
        apertCase({64, 96, 96, 1, 1, 1, SepKind::ND, SepKind::Periodic, SepKind::Periodic}, {16, 20, 24}, true);
        apertCase({64, 96, 96, 1, 1, 1, SepKind::ND, SepKind::DD, SepKind::DD}, {16, 20, 24}, true);
        apertCase({128, 96, 1, 1, 1, 1, SepKind::ND, SepKind::DD, SepKind::Periodic}, {16, 20, 24}, true);
        apertRejections();
        apertDirect();
        crossProcess();
    }
    else if (mode == "hash")
    {
        hashes();
        return 0;
    }
    else if (mode == "perf")
        perf();
    else if (mode == "perfsep") // one grid, for a kernel profiler
    {
        const SepBox b{384, 176, 176, 1, 1, 1, SepKind::ND, SepKind::DD, SepKind::DD};
        auto s = *gpu::SeparablePoisson::make(b);
        std::printf("384x176x176 ND/DD/DD: %.3f ms/solve\n", msPerSolve(s, b.size(), 20));
    }
    else
    {
        std::printf("usage: test_gpu_poisson_sep sep|apert|perf\n");
        return 1;
    }
    return dnstest::finish();
}
