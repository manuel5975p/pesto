#include "dns/gpu/species.hpp"

#include "dns/chem.hpp"
#include "dns/gpu/device_util.hpp"

#include <cassert>
#include <type_traits>

namespace dns::gpu::species {
namespace {

// Thick: the thickened flame's rate (E/F) omega, eof the per-cell E/F of
// gpu/tfles (its own instantiation; false is the shipped kernel).
template <bool Thick = false>
__global__ void kChemCellPass(GpuGrid g, int nY, const Real* rho,
                              SpeciesCombine ys, const Real* T, Real lnA,
                              Real Ta, int fuel, Real* Y0, Real* Y1, Real* Y2,
                              Real* Y3, Real* omega, const Real* eof)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    Real* Y[kMaxY] = {Y0, Y1, Y2, Y3};
    const Real r = rho[c];
    for (int k = 0; k < nY; ++k) Y[k][c] = ys.ya[k][c] / r;
    if constexpr (Thick)
        omega[c] = arrheniusRate(lnA, Ta, r, Y[fuel][c], T[c]) * eof[c];
    else
        omega[c] = arrheniusRate(lnA, Ta, r, Y[fuel][c], T[c]);
}

// The two-reactant cell pass (SpeciesSetup::ox >= 0): arrheniusRate2.
__global__ void kChemCellPass2(GpuGrid g, int nY, const Real* rho,
                               SpeciesCombine ys, const Real* T, Real lnA,
                               Real Ta, int fuel, int ox, Real* Y0, Real* Y1,
                               Real* Y2, Real* Y3, Real* omega)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    Real* Y[kMaxY] = {Y0, Y1, Y2, Y3};
    const Real r = rho[c];
    for (int k = 0; k < nY; ++k) Y[k][c] = ys.ya[k][c] / r;
    omega[c] = arrheniusRate2(lnA, Ta, r, Y[fuel][c], Y[ox][c], T[c]);
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

// The effective species rho D of an interior face (Diff != Molecular):
// Eddy  rho D + nut_f / Sc_t (the LES sub-grid flux, nut_f = faceFlux's own
//       face mean of mu_t);
// Thick FE_f rho D + (1 - Om_f) nut_f / Sc_t (no SGS flux where the sensor
//       thickens; nut null = no LES), FE_f, Om_f arithmetic face means.
template <int Diff>
__device__ inline Real effRhoD(const SpeciesView& sv, const Face& f, Real rhoD)
{
    if constexpr (Diff == int(SpeciesDiff::Eddy)) {
        return rhoD + Real(0.5) * (sv.nut[f.L] + sv.nut[f.R]) * sv.invSct;
    } else {
        Real r = Real(0.5) * (sv.FE[f.L] + sv.FE[f.R]) * rhoD;
        if (sv.nut) {
            const Real om = Real(0.5) * (sv.Om[f.L] + sv.Om[f.R]);
            r += (Real(1) - om) * (Real(0.5) * (sv.nut[f.L] + sv.nut[f.R])) *
                 sv.invSct;
        }
        return r;
    }
}

// Thick: the energy flux faceFlux lacks on an interior face L | R, so the
// conductivity becomes FE kappa + (1 - Om) nut cp/Pr_t:
//   -(FE_f - 1) kappa_f dT/dn + Om_f nut_f (cp/Pr_t) dT/dn,
// kappa_f with faceFlux's own Sutherland face ratio. An additive flux, so the
// correction is conservative and faceFlux is untouched.
__device__ inline Real thickEnergyFlux(ViscView vv, PrimView q,
                                       const SpeciesView& sv, const Face& f,
                                       Real h)
{
    Real kapf = vv.kappa;
    if (vv.sutherland) kapf *= vv.ratio(Real(0.5) * (q.T[f.L] + q.T[f.R]));
    const Real dTn = (q.T[f.R] - q.T[f.L]) / h;
    Real fl = -(Real(0.5) * (sv.FE[f.L] + sv.FE[f.R]) - Real(1)) * kapf * dTn;
    if (sv.nut)
        fl += Real(0.5) * (sv.Om[f.L] + sv.Om[f.R]) *
              (Real(0.5) * (sv.nut[f.L] + sv.nut[f.R])) * sv.kapLes * dTn;
    return fl;
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

// Row: per-row inflow fractions (SpeciesView::yInRow); its own instantiation,
// so the uniform-inflow kernel is the shipped code. kSolid: the immersed-wall
// instantiation (SpeciesSetup::solid); false is the kernel as it was before
// walls existed. Row && kSolid is not instantiated. Diff (SpeciesDiff): the
// interior faces' rho D (effRhoD) and, Thick, the energy correction flux and
// its divergence added with the heat release; Molecular is the shipped
// kernel. kSolid && Diff != Molecular is not instantiated.
template <bool Row, bool kSolid, int Diff = 0>
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
    [[maybe_unused]] Real dEc = Real(0); // Thick: the energy correction
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
        Real dh = openHi ? Real(0) : faceRhoD(vv, q, sv.invPr, fh);
        Real dl = openLo ? Real(0) : faceRhoD(vv, q, sv.invPr, fl);
        if constexpr (Diff != int(SpeciesDiff::Molecular)) {
            if (!openHi) dh = effRhoD<Diff>(sv, fh, dh);
            if (!openLo) dl = effRhoD<Diff>(sv, fl, dl);
        }
        if constexpr (Diff == int(SpeciesDiff::Thick)) {
            const Real eh = openHi ? Real(0) : thickEnergyFlux(vv, q, sv, fh, h);
            const Real el = openLo ? Real(0) : thickEnergyFlux(vv, q, sv, fl, h);
            dEc -= (eh - el) / h;
        }
        for (int s = 0; s < sv.nY; ++s) {
            const Real* Y = sv.Y[s];
            Real yin = sv.yIn[s];
            if constexpr (Row)
                if (sv.yInRow[s])
                    yin = sv.yInRow[s][std::size_t(j) + std::size_t(g.ny) * k];
            Real Fh = openHi ? mo * (mo > Real(0) ? Y[c] : yin)
                             : faceY(Y, fh, mh, dh, h);
            Real Fl = openLo ? mo * (mo < Real(0) ? Y[c] : yin)
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
    if constexpr (Diff == int(SpeciesDiff::Thick))
        oE[c] += cr * (sv.q * om + dEc);
    else
        oE[c] += cr * (sv.q * om);
}

// The shared limiter of face f (two-reactant kernel): theta = min over the
// VOTING species of mcSlope(d1, d2) / d2, in [0, 2] (2 = none votes). The
// face value Y_up + theta/2 d2 is LINEAR in Y across species: an affine
// combination of the species (the coupling functions, the mixture fraction)
// is transported exactly by the same formula, whatever theta is. A voter
// keeps its own TVD bound (theta <= its own MC ratio; Sweby's region is
// star-shaped toward 0). A species votes iff |d2| >= kVoteJump = 2^-24 (the
// FP32 resolution of a unit mass fraction): a reactant consumed to ~1e-40
// on the far side of the sheet carries sign-flipping rounding noise whose
// "extrema" (theta = 0) forced first-order upwind on EVERY species across
// the whole far side -- measured on the advected layer (gpu_diffusion_mix)
// Z 5.0e-3 below the analytic erf at the slab centre with every species
// voting, max error 1.5e-3 with the threshold. A non-voter moves at most |d2| <
// 2^-24 past its own neighbours per face, and votes again once it varies.
inline constexpr float kVoteJump = 0x1p-24f;
__device__ inline Real sharedTheta(const SpeciesView& sv, const Face& f,
                                   Real mdot)
{
    // Upwind cell = an open-face boundary cell (no second neighbour): the
    // CENTRAL face value (theta 1, between the two cells, so bounded) -- the
    // premixed kernel's first-order upwind there is a convective-flux
    // mismatch mdot (Y_R - Y_L)/2 against the central energy flux.
    if (mdot >= Real(0) ? f.firstL : f.firstR) return Real(1);
    Real th = Real(2);
    for (int s = 0; s < sv.nY; ++s) {
        const Real* Y = sv.Y[s];
        Real d1, d2;
        if (mdot >= Real(0)) {
            d1 = f.firstL ? Real(0) : Y[f.L] - Y[f.LL];
            d2 = Y[f.R] - Y[f.L];
        } else {
            d1 = f.firstR ? Real(0) : Y[f.R] - Y[f.RR];
            d2 = Y[f.L] - Y[f.R];
        }
        if (fabsf(d2) >= kVoteJump) th = fminf(th, mcSlope(d1, d2) / d2);
    }
    return th;
}

__device__ inline Real faceY2(const Real* Y, const Face& f, Real mdot,
                              Real th, Real rhoD, Real h)
{
    const Real yf = mdot >= Real(0)
                        ? Y[f.L] + Real(0.5) * th * (Y[f.R] - Y[f.L])
                        : Y[f.R] + Real(0.5) * th * (Y[f.L] - Y[f.R]);
    return mdot * yf - rhoD * (Y[f.R] - Y[f.L]) / h;
}

// The two-reactant species stage (SpeciesSetup::ox >= 0): kSpeciesStage
// with the shared limiter and the oxidizer sink stoich * omega. Its own
// kernel, so the premixed one stays the shipped code. The open x-hi face
// carries the boundary cell's own Y BOTH ways: the inflow fractions belong
// to x-lo (the premixed kernel's backflow took the x-lo row's jet fuel in
// through the outflow). AmbHi (the candle's top): backflow carries sv.yHi.
template <bool Row, bool AmbHi = false>
__global__ void kSpeciesStage2(GpuGrid g, ViscView vv, PrimView q,
                               StageCombine cmb, SpeciesCombine sc,
                               SpeciesView sv, Real* oy0, Real* oy1, Real* oy2,
                               Real* oy3, Real* oE)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
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
        const bool openHi = sv.openX && a == 0 && i == g.nx - 1;
        const bool openLo = sv.openX && a == 0 && i == 0;
        const Real mo = q.rho[c] * q.u[c];
        const Real mh = openHi ? mo : faceMdot(q, uc[a], fh);
        const Real ml = openLo ? mo : faceMdot(q, uc[a], fl);
        const Real dh = openHi ? Real(0) : faceRhoD(vv, q, sv.invPr, fh);
        const Real dl = openLo ? Real(0) : faceRhoD(vv, q, sv.invPr, fl);
        const Real th = openHi ? Real(0) : sharedTheta(sv, fh, mh);
        const Real tl = openLo ? Real(0) : sharedTheta(sv, fl, ml);
        for (int s = 0; s < sv.nY; ++s) {
            const Real* Y = sv.Y[s];
            Real yin = sv.yIn[s];
            if constexpr (Row)
                if (sv.yInRow[s])
                    yin = sv.yInRow[s][std::size_t(j) + std::size_t(g.ny) * k];
            Real yb = Y[c]; // backflow: the cell's own Y
            if constexpr (AmbHi) yb = mo < Real(0) ? sv.yHi[s] : Y[c];
            const Real Fh = openHi ? mo * yb : faceY2(Y, fh, mh, th, dh, h);
            const Real Fl = openLo ? mo * (mo < Real(0) ? Y[c] : yin)
                                   : faceY2(Y, fl, ml, tl, dl, h);
            div[s] -= (Fh - Fl) / h;
        }
    }
    Real* out[kMaxY] = {oy0, oy1, oy2, oy3};
    const Real cr = cmb.crAt(c);
    const Real om = sv.omega[c];
    const Real omO = sv.stoich * om;
    for (int s = 0; s < sv.nY; ++s) {
        const Real r = s == sv.fuel ? div[s] - om
                       : s == sv.ox ? div[s] - omO
                                    : div[s];
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
              Real lnA, Real Ta, int fuel, int ox, Real* const Y[kMaxY],
              Real* omega, const Real* eof)
{
    assert(s.nY >= 1 && fuel >= 0 && fuel < s.nY && ox < s.nY && ox != fuel);
    assert(!eof || ox < 0);
    SpeciesCombine ys;
    for (int k = 0; k < s.nY; ++k) ys.ya[k] = s.y[k].data();
    ys.nY = s.nY;
    if (ox >= 0) {
        kChemCellPass2<<<launchBlocks(g.size()), blockSize>>>(
            g, s.nY, s.rho.data(), ys, T, lnA, Ta, fuel, ox, Y[0], Y[1], Y[2],
            Y[3], omega);
        checkLastKernel("kChemCellPass2");
        return;
    }
    if (eof)
        kChemCellPass<true><<<launchBlocks(g.size()), blockSize>>>(
            g, s.nY, s.rho.data(), ys, T, lnA, Ta, fuel, Y[0], Y[1], Y[2],
            Y[3], omega, eof);
    else
        kChemCellPass<false><<<launchBlocks(g.size()), blockSize>>>(
            g, s.nY, s.rho.data(), ys, T, lnA, Ta, fuel, Y[0], Y[1], Y[2],
            Y[3], omega, nullptr);
    checkLastKernel("kChemCellPass");
}

void stage(const GpuGrid& g, ViscView vv, PrimView q, const StageCombine& cmb,
           const SpeciesCombine& sc, const SpeciesView& sv,
           DeviceCompState& out)
{
    assert(!cmb.ol[0] && !cmb.lts && sc.nY == sv.nY && out.nY == sv.nY);
    bool row = false;
    for (int s = 0; s < sv.nY; ++s) row = row || sv.yInRow[s] != nullptr;
    // The walled kernel is premixed and uniform-inflow only (the chamber).
    assert(!sv.solid || (!row && sv.ox < 0));
    // The eddy / thickened instantiations: premixed, no wall.
    const SpeciesDiff diff = sv.FE    ? SpeciesDiff::Thick
                             : sv.nut ? SpeciesDiff::Eddy
                                      : SpeciesDiff::Molecular;
    assert(diff == SpeciesDiff::Molecular || (!sv.solid && sv.ox < 0));
    assert(diff != SpeciesDiff::Thick || (sv.Om && sv.FE));
    auto launch = [&](auto r, auto solid) {
        kSpeciesStage<decltype(r)::value, decltype(solid)::value>
            <<<launchBlocks(g.size()), blockSize>>>(
                g, vv, q, cmb, sc, sv, out.y[0].data(), out.y[1].data(),
                out.y[2].data(), out.y[3].data(), out.E.data());
    };
    auto launchD = [&](auto r, auto d) {
        kSpeciesStage<decltype(r)::value, false, decltype(d)::value>
            <<<launchBlocks(g.size()), blockSize>>>(
                g, vv, q, cmb, sc, sv, out.y[0].data(), out.y[1].data(),
                out.y[2].data(), out.y[3].data(), out.E.data());
    };
    using Eddy = std::integral_constant<int, int(SpeciesDiff::Eddy)>;
    using Thick = std::integral_constant<int, int(SpeciesDiff::Thick)>;
    if (diff == SpeciesDiff::Eddy) {
        if (row) launchD(std::true_type{}, Eddy{});
        else launchD(std::false_type{}, Eddy{});
        checkLastKernel("kSpeciesStage<Eddy>");
        return;
    }
    if (diff == SpeciesDiff::Thick) {
        if (row) launchD(std::true_type{}, Thick{});
        else launchD(std::false_type{}, Thick{});
        checkLastKernel("kSpeciesStage<Thick>");
        return;
    }
    auto launch2 = [&](auto r, auto amb) {
        kSpeciesStage2<decltype(r)::value, decltype(amb)::value>
            <<<launchBlocks(g.size()), blockSize>>>(
                g, vv, q, cmb, sc, sv, out.y[0].data(), out.y[1].data(),
                out.y[2].data(), out.y[3].data(), out.E.data());
    };
    if (sv.ox >= 0) {
        // The ambient top rides the per-row (candle) instantiation only.
        assert(!sv.ambientHi || (row && sv.openX));
        if (row && sv.ambientHi) launch2(std::true_type{}, std::true_type{});
        else if (row) launch2(std::true_type{}, std::false_type{});
        else launch2(std::false_type{}, std::false_type{});
        checkLastKernel("kSpeciesStage2");
        return;
    }
    if (sv.solid) launch(std::false_type{}, std::true_type{});
    else if (row) launch(std::true_type{}, std::false_type{});
    else launch(std::false_type{}, std::false_type{});
    checkLastKernel("kSpeciesStage");
}

namespace {

__global__ void kBoundaryHeat(GpuGrid g, StageCombine cmb, Real q,
                              const Real* omega, Real* oE)
{
    const std::size_t nA = std::size_t(g.ny) * g.nz;
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= 2 * nA) return;
    const std::size_t r = t % nA;
    const int i = t < nA ? 0 : g.nx - 1;
    const std::size_t c = gidx(g, i, int(r % g.ny), int(r / g.ny));
    oE[c] += cmb.crAt(c) * (q * omega[c]);
}

__global__ void kSpeciesInflowPin(GpuGrid g, int nY, SpeciesView sv,
                                  const Real* oRho, Real* y0, Real* y1,
                                  Real* y2, Real* y3)
{
    const std::size_t nA = std::size_t(g.ny) * g.nz;
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= nA) return;
    const int j = int(t % g.ny), k = int(t / g.ny);
    const std::size_t c = gidx(g, 0, j, k);
    Real* y[kMaxY] = {y0, y1, y2, y3};
    for (int s = 0; s < nY; ++s) {
        const Real yt = sv.yInRow[s] ? sv.yInRow[s][t] : sv.yIn[s];
        y[s][c] = yt * oRho[c];
    }
}

} // namespace

void inflowPin(const GpuGrid& g, const SpeciesView& sv, DeviceCompState& out)
{
    assert(sv.nY >= 1 && out.nY == sv.nY);
    const std::size_t n = std::size_t(g.ny) * g.nz;
    kSpeciesInflowPin<<<launchBlocks(n), blockSize>>>(
        g, sv.nY, sv, out.rho.data(), out.y[0].data(), out.y[1].data(),
        out.y[2].data(), out.y[3].data());
    checkLastKernel("kSpeciesInflowPin");
}

void boundaryHeat(const GpuGrid& g, const StageCombine& cmb, Real q,
                  const Real* omega, DeviceCompState& out)
{
    assert(!cmb.ol[0]);
    const std::size_t n = 2 * std::size_t(g.ny) * g.nz;
    kBoundaryHeat<<<launchBlocks(n), blockSize>>>(g, cmb, q, omega,
                                                  out.E.data());
    checkLastKernel("kBoundaryHeat");
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
