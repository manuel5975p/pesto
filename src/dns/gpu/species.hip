#include "dns/gpu/species.hpp"

#include "dns/chem.hpp"
#include "dns/gpu/device_util.hpp"

#include <cassert>

namespace dns::gpu::species {
namespace {

__global__ void kChemCellPass(GpuGrid g, int nY, const Real* rho,
                              SpeciesCombine ys, const Real* T, Real lnA,
                              Real Ta, int fuel, Real* Y0, Real* Y1, Real* Y2,
                              Real* Y3, Real* omega)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    Real* Y[kMaxY] = {Y0, Y1, Y2, Y3};
    const Real r = rho[c];
    for (int k = 0; k < nY; ++k) Y[k][c] = ys.ya[k][c] / r;
    omega[c] = arrheniusRate(lnA, Ta, r, Y[fuel][c], T[c]);
}

// Wrap-aware offsets from cell (i,j,k) to its SECOND neighbours along each
// axis (p = +2, m = -2); a degenerate axis (n == 1) gives 0, the self-wrap.
// Pre: n >= 3 on every non-degenerate axis.
__device__ inline Offsets offsets2At(const GpuGrid& g, int i, int j, int k)
{
    const int q[3] = {i, j, k}, n[3] = {g.nx, g.ny, g.nz};
    const std::ptrdiff_t s[3] = {1, g.nx, std::ptrdiff_t(g.nx) * g.ny};
    Offsets o;
    for (int a = 0; a < 3; ++a) {
        const int up = (q[a] + 2) % n[a], dn = ((q[a] - 2) % n[a] + n[a]) % n[a];
        o.p[a] = std::ptrdiff_t(up - q[a]) * s[a];
        o.m[a] = std::ptrdiff_t(dn - q[a]) * s[a];
    }
    return o;
}

// Monotonized-central slope: 0 unless d1, d2 share a sign, else
// sign * min(2|d1|, 2|d2|, |d1 + d2|/2). Exactly 0 on a uniform field.
__device__ inline Real mcSlope(Real d1, Real d2)
{
    if (!(d1 * d2 > Real(0))) return Real(0);
    const Real m = fminf(fminf(Real(2) * fabsf(d1), Real(2) * fabsf(d2)),
                         Real(0.5) * fabsf(d1 + d2));
    return d1 > Real(0) ? m : -m;
}

// One interior face L | R along `axis` (R = L + one step): the mass flux in
// faceFlux's own expression, rho D, and the stencil cells LL = L - 1,
// RR = R + 1. firstL/firstR: that upwind side has no second neighbour (an
// open face's boundary cell: its LL/RR would wrap across the domain), so the
// face goes first order when its upwind cell is that side.
struct Face
{
    std::size_t L, R, LL, RR;
    bool firstL, firstR;
};

__device__ inline Real faceMdot(PrimView q, const Real* un, const Face& f)
{
    const Real rm = Real(0.5) * (q.rho[f.L] + q.rho[f.R]);
    const Real um = Real(0.5) * (un[f.L] + un[f.R]);
    return rm * um;
}

__device__ inline Real faceRhoD(ViscView vv, PrimView q, Real invPr,
                                const Face& f)
{
    Real muf = vv.mu;
    if (vv.sutherland) muf *= vv.ratio(Real(0.5) * (q.T[f.L] + q.T[f.R]));
    return muf * invPr;
}

__device__ inline Real faceY(const Real* Y, const Face& f, Real mdot,
                             Real rhoD, Real h)
{
    Real yf;
    if (mdot >= Real(0)) {
        const Real d1 = f.firstL ? Real(0) : Y[f.L] - Y[f.LL];
        yf = Y[f.L] + Real(0.5) * mcSlope(d1, Y[f.R] - Y[f.L]);
    } else {
        const Real d1 = f.firstR ? Real(0) : Y[f.R] - Y[f.RR];
        yf = Y[f.R] + Real(0.5) * mcSlope(d1, Y[f.L] - Y[f.R]);
    }
    return mdot * yf - rhoD * (Y[f.R] - Y[f.L]) / h;
}

// The two faces of cell c along axis a, both as L | R pairs.
__device__ inline void cellFaces(const GpuGrid& g, int i, int a, int open,
                                 std::size_t c, const Offsets& o,
                                 const Offsets& o2, Face& hi, Face& lo)
{
    const bool op = open && a == 0;
    hi = {c, c + o.p[a], c + o.m[a], c + o2.p[a], op && i == 0,
          op && i + 1 == g.nx - 1};
    lo = {c + o.m[a], c, c + o2.m[a], c + o.p[a], op && i - 1 == 0,
          op && i == g.nx - 1};
}

// kSolid: the immersed-wall instantiation (SpeciesSetup::solid). false is
// the kernel as it was before walls existed, token for token.
template <bool kSolid>
__global__ void kSpeciesStage(GpuGrid g, ViscView vv, PrimView q,
                              StageCombine cmb, SpeciesCombine sc,
                              SpeciesView sv, Real* oy0, Real* oy1, Real* oy2,
                              Real* oy3, Real* oE)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    if constexpr (kSolid) {
        // A solid cell carries no species rhs and releases no heat (its rho
        // Y stays at its IC value, 0, so omega is 0 there anyway; this also
        // keeps a non-finite ghost omega out of E).
        if (sv.solid[c]) {
            Real* out[kMaxY] = {oy0, oy1, oy2, oy3};
            for (int s = 0; s < sv.nY; ++s)
                out[s][c] = cmb.ca * sc.ya[s][c] + cmb.cb * sc.yb[s][c];
            return;
        }
    }
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Offsets o2 = offsets2At(g, i, j, k);
    const Real* uc[3] = {q.u, q.v, q.w};
    Real div[kMaxY] = {Real(0), Real(0), Real(0), Real(0)};
    for (int a = 0; a < 3; ++a) {
        const Real h = a == 0 ? g.hx : a == 1 ? g.hy : g.hz;
        Face fh, fl;
        cellFaces(g, i, a, sv.openX, c, o, o2, fh, fl);
        // Walled faces: no flux; a solid second neighbour (the MUSCL
        // upwind-upwind cell) drops the face to first order.
        bool wallH = false, wallL = false;
        if constexpr (kSolid) {
            wallH = sv.solid[fh.R] != 0;
            wallL = sv.solid[fl.L] != 0;
            fh.firstL = fh.firstL || sv.solid[fh.LL];
            fh.firstR = fh.firstR || sv.solid[fh.RR];
            fl.firstL = fl.firstL || sv.solid[fl.LL];
            fl.firstR = fl.firstR || sv.solid[fl.RR];
        }
        // Open x domain faces: the boundary cell's own mdot, the inflow
        // value in and the cell's out, no diffusion (ScalarBoundary).
        const bool openHi = sv.openX && a == 0 && i == g.nx - 1;
        const bool openLo = sv.openX && a == 0 && i == 0;
        const Real mo = q.rho[c] * q.u[c];
        const Real mh = openHi ? mo : faceMdot(q, uc[a], fh);
        const Real ml = openLo ? mo : faceMdot(q, uc[a], fl);
        const Real dh = openHi ? Real(0) : faceRhoD(vv, q, sv.invPr, fh);
        const Real dl = openLo ? Real(0) : faceRhoD(vv, q, sv.invPr, fl);
        for (int s = 0; s < sv.nY; ++s) {
            const Real* Y = sv.Y[s];
            Real Fh = openHi ? mo * (mo > Real(0) ? Y[c] : sv.yIn[s])
                             : faceY(Y, fh, mh, dh, h);
            Real Fl = openLo ? mo * (mo < Real(0) ? Y[c] : sv.yIn[s])
                             : faceY(Y, fl, ml, dl, h);
            if constexpr (kSolid) {
                if (wallH) Fh = Real(0);
                if (wallL) Fl = Real(0);
            }
            div[s] -= (Fh - Fl) / h;
        }
    }
    Real* out[kMaxY] = {oy0, oy1, oy2, oy3};
    const Real cr = cmb.crAt(c);
    const Real om = sv.omega[c];
    for (int s = 0; s < sv.nY; ++s) {
        const Real r = s == sv.fuel ? div[s] - om : div[s];
        out[s][c] = cmb.ca * sc.ya[s][c] + cmb.cb * sc.yb[s][c] + cr * r;
    }
    oE[c] += cr * (sv.q * om);
}

// The species mass divergence of boundary cell c: the mdot of every face
// exactly as kSpeciesStage forms it, accumulated in the same order.
__device__ inline Real speciesMassDiv(const GpuGrid& g, PrimView q, int i,
                                      int j, int k, std::size_t c)
{
    const Offsets o = offsetsAt(g, i, j, k);
    const Offsets o2 = offsets2At(g, i, j, k);
    const Real* uc[3] = {q.u, q.v, q.w};
    Real div = 0;
    for (int a = 0; a < 3; ++a) {
        const Real h = a == 0 ? g.hx : a == 1 ? g.hy : g.hz;
        Face fh, fl;
        cellFaces(g, i, a, 1, c, o, o2, fh, fl);
        const bool openHi = a == 0 && i == g.nx - 1;
        const bool openLo = a == 0 && i == 0;
        const Real mo = q.rho[c] * q.u[c];
        const Real mh = openHi ? mo : faceMdot(q, uc[a], fh);
        const Real ml = openLo ? mo : faceMdot(q, uc[a], fl);
        div -= (mh - ml) / h;
    }
    return div;
}

__global__ void kSpeciesOpenFix(GpuGrid g, PrimView q, StageCombine cmb,
                                int nY, const Real* oRho, Real* y0, Real* y1,
                                Real* y2, Real* y3)
{
    const std::size_t nA = std::size_t(g.ny) * g.nz;
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= 2 * nA) return;
    const std::size_t r = t % nA;
    const int i = t < nA ? 0 : g.nx - 1;
    const int j = int(r % g.ny), k = int(r / g.ny);
    const std::size_t c = gidx(g, i, j, k);
    const Real M = speciesMassDiv(g, q, i, j, k, c);
    const Real comb = cmb.ca * cmb.a[0][c] + cmb.cb * cmb.b[0][c];
    const Real rs = comb + cmb.crAt(c) * M;
    Real* y[kMaxY] = {y0, y1, y2, y3};
    for (int s = 0; s < nY; ++s) y[s][c] = y[s][c] / rs * oRho[c];
}

} // namespace

void cellPass(const GpuGrid& g, const DeviceCompState& s, const Real* T,
              Real lnA, Real Ta, int fuel, Real* const Y[kMaxY], Real* omega)
{
    assert(s.nY >= 1 && fuel >= 0 && fuel < s.nY);
    SpeciesCombine ys;
    for (int k = 0; k < s.nY; ++k) ys.ya[k] = s.y[k].data();
    ys.nY = s.nY;
    kChemCellPass<<<launchBlocks(g.size()), blockSize>>>(
        g, s.nY, s.rho.data(), ys, T, lnA, Ta, fuel, Y[0], Y[1], Y[2], Y[3],
        omega);
    checkLastKernel("kChemCellPass");
}

void stage(const GpuGrid& g, ViscView vv, PrimView q, const StageCombine& cmb,
           const SpeciesCombine& sc, const SpeciesView& sv,
           DeviceCompState& out)
{
    assert(!cmb.ol[0] && !cmb.lts && sc.nY == sv.nY && out.nY == sv.nY);
    if (sv.solid)
        kSpeciesStage<true><<<launchBlocks(g.size()), blockSize>>>(
            g, vv, q, cmb, sc, sv, out.y[0].data(), out.y[1].data(),
            out.y[2].data(), out.y[3].data(), out.E.data());
    else
        kSpeciesStage<false><<<launchBlocks(g.size()), blockSize>>>(
            g, vv, q, cmb, sc, sv, out.y[0].data(), out.y[1].data(),
            out.y[2].data(), out.y[3].data(), out.E.data());
    checkLastKernel("kSpeciesStage");
}

void openConsistency(const GpuGrid& g, PrimView q, const StageCombine& cmb,
                     int nY, DeviceCompState& out)
{
    assert(nY >= 1 && out.nY == nY && !cmb.ol[0]);
    const std::size_t n = 2 * std::size_t(g.ny) * g.nz;
    kSpeciesOpenFix<<<launchBlocks(n), blockSize>>>(
        g, q, cmb, nY, out.rho.data(), out.y[0].data(), out.y[1].data(),
        out.y[2].data(), out.y[3].data());
    checkLastKernel("kSpeciesOpenFix");
}

} // namespace dns::gpu::species
