// DeviceLowMach (gpu/lowmach.hpp): the low-Mach flame solver's kernels and
// its SSP-RK3 step. Masterplan real_bunsen section 2 / S4; open laterals S6;
// gravity S7; the immersed tube (the Wall instantiations) S8.
#include "dns/gpu/lowmach.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/internal/plane_sums.hpp"
#include "dns/gpu/internal/reduce.hpp"
#include "dns/gpu/internal/species_face.hpp"
#include "dns/gpu/internal/stage_write.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <span>
#include <type_traits>
#include <utility>

namespace dns::gpu {
namespace {

using internal::Face;
using internal::faceValue2;
using internal::faceY2;
using internal::sharedThetaOf;

// The kernels' constants (FP32; the reductions keep FP64 copies).
// The inflow face is per ROW (j + ny k; S5): m, rho = P0/(R T), T, Y_k.
// S6: the ambient of the open faces (and, with any lateral open, of the x-hi
// backflow): T, rho = P0/(R T), Y_k; bfHalf = beta/2 of the stabilization.
struct LmK
{
    GpuGrid g;
    ViscView vv;
    Real invPr, prGas;
    const Real *inM, *inRho, *inT, *inY0, *inY1;
    int nY, fuel, ox;
    Real stoich, lnA, Ta, qcp;
    Real ambT, ambRho, ambY0, ambY1, bfHalf;
    Real grav; // S7: g along -x (read by the Grav instantiation only)
    // S8: LmPressure's per-cell face codes (kLmSolid = the tube) and the
    // tube's wall T (read by the Wall instantiations only).
    const std::uint8_t* code;
    Real wallT;
    // M5 --turb: the inflow's tangential rows (read by the Turb instantiation only).
    const Real *inV, *inW;
};

// The MAC mass fluxes of one state: x (nx + 1 per row), the y and z LO faces
// (indexed like the cell), the open laterals' hi faces (null when periodic).
struct LmM
{
    const Real *x, *y, *z, *yHi, *zHi;
};

// Modular wrap (any offset; n == 1 maps everything to 0: the 2D z axis).
__host__ __device__ inline int wrapN(int q, int n)
{
    const int r = q % n;
    return r < 0 ? r + n : r;
}

__host__ __device__ inline int clampN(int q, int n)
{
    return q < 0 ? 0 : q >= n ? n - 1 : q;
}

// The inflow row of (j, k), j and k periodic.
__host__ __device__ inline std::size_t rowAt(const GpuGrid& g, int j, int k)
{
    return std::size_t(wrapN(j, g.ny)) + std::size_t(g.ny) * wrapN(k, g.nz);
}

__device__ inline Real inY(const LmK& K, int s, std::size_t r)
{
    return s == 0 ? K.inY0[r] : K.inY1[r];
}

__device__ inline Real ambY(const LmK& K, int s)
{
    return s == 0 ? K.ambY0 : K.ambY1;
}

// Cell (i, j, k), j and k periodic; i must be in [0, nx).
__host__ __device__ inline std::size_t cellAt(const GpuGrid& g, int i, int j,
                                              int k)
{
    return std::size_t(i) +
           std::size_t(g.nx) *
               (std::size_t(wrapN(j, g.ny)) + std::size_t(g.ny) * wrapN(k, g.nz));
}

// x face i in [0, nx] of row (j, k), j and k periodic.
__host__ __device__ inline std::size_t faceX(const GpuGrid& g, int i, int j,
                                             int k)
{
    return std::size_t(i) +
           std::size_t(g.nx + 1) *
               (std::size_t(wrapN(j, g.ny)) + std::size_t(g.ny) * wrapN(k, g.nz));
}

// The lateral topology of a kernel instantiation (S6). A periodic axis
// WRAPS (Lat<false, false> is the S5 indexing exactly); an open axis CLAMPS:
// a cell read across an open face is the boundary cell (the zero-gradient
// ghost -- the inflow exceptions are explicit at each use), and a face read
// past it is the boundary face (lo: j <= 0 -> face 0, stored at cell (i, 0,
// k); hi: j >= ny -> yHi[i + nx k]).
template <bool OY, bool OZ>
struct Lat
{
    __host__ __device__ static int y(const GpuGrid& g, int j)
    {
        return OY ? clampN(j, g.ny) : wrapN(j, g.ny);
    }
    __host__ __device__ static int z(const GpuGrid& g, int k)
    {
        return OZ ? clampN(k, g.nz) : wrapN(k, g.nz);
    }
    __host__ __device__ static std::size_t cell(const GpuGrid& g, int i, int j, int k)
    {
        return std::size_t(i) +
               std::size_t(g.nx) * (std::size_t(y(g, j)) + std::size_t(g.ny) * z(g, k));
    }
    __host__ __device__ static std::size_t fx(const GpuGrid& g, int i, int j, int k)
    {
        return std::size_t(i) +
               std::size_t(g.nx + 1) * (std::size_t(y(g, j)) + std::size_t(g.ny) * z(g, k));
    }
    __host__ __device__ static std::size_t row(const GpuGrid& g, int j, int k)
    {
        return std::size_t(y(g, j)) + std::size_t(g.ny) * z(g, k);
    }
    // y face j of column (i, k); z face k of column (i, j).
    __host__ __device__ static Real my(const GpuGrid& g, const LmM& m, int i, int j,
                                       int k)
    {
        if (OY && j >= g.ny) return m.yHi[std::size_t(i) + std::size_t(g.nx) * z(g, k)];
        return m.y[cell(g, i, j, k)];
    }
    __host__ __device__ static Real mz(const GpuGrid& g, const LmM& m, int i, int j,
                                       int k)
    {
        if (OZ && k >= g.nz) return m.zHi[std::size_t(i) + std::size_t(g.nx) * y(g, j)];
        return m.z[cell(g, i, j, k)];
    }
};

// The density of x face i (m its mass flux): the inflow's at 0, the last
// cell's (upwind of the outflow) at nx -- with a lateral open, a BACKFLOW (m
// < 0) there enters as the ambient -- else the mean of the two cells.
template <bool OY, bool OZ>
__device__ inline Real rhoX(const LmK& K, const Real* rho, Real m, int i, int j,
                            int k)
{
    using L = Lat<OY, OZ>;
    if (i == 0) return K.inRho[L::row(K.g, j, k)];
    if (i == K.g.nx) {
        if constexpr (OY || OZ) {
            if (m < Real(0)) return K.ambRho;
        }
        return rho[L::cell(K.g, K.g.nx - 1, j, k)];
    }
    return Real(0.5) * (rho[L::cell(K.g, i - 1, j, k)] + rho[L::cell(K.g, i, j, k)]);
}

// The density of y face j of column (i, k) (m its flux): the mean of the two
// cells; an open boundary face (j <= 0, j >= ny) the upwind one -- the
// ambient's on inflow, the boundary cell's on outflow.
template <bool OY, bool OZ>
__device__ inline Real rhoY(const LmK& K, const Real* rho, Real m, int i, int j,
                            int k)
{
    using L = Lat<OY, OZ>;
    if constexpr (OY) {
        if (j <= 0) return m > Real(0) ? K.ambRho : rho[L::cell(K.g, i, 0, k)];
        if (j >= K.g.ny)
            return m < Real(0) ? K.ambRho : rho[L::cell(K.g, i, K.g.ny - 1, k)];
    }
    return Real(0.5) * (rho[L::cell(K.g, i, j - 1, k)] + rho[L::cell(K.g, i, j, k)]);
}

template <bool OY, bool OZ>
__device__ inline Real rhoZ(const LmK& K, const Real* rho, Real m, int i, int j,
                            int k)
{
    using L = Lat<OY, OZ>;
    if constexpr (OZ) {
        if (k <= 0) return m > Real(0) ? K.ambRho : rho[L::cell(K.g, i, j, 0)];
        if (k >= K.g.nz)
            return m < Real(0) ? K.ambRho : rho[L::cell(K.g, i, j, K.g.nz - 1)];
    }
    return Real(0.5) * (rho[L::cell(K.g, i, j, k - 1)] + rho[L::cell(K.g, i, j, k)]);
}

// The model's ONE buoyancy term (S7; gravity g along -x): x face i of row
// (j, k) carrying m adds -(rho_f - rho_a) g to its momentum rhs R, rho_f =
// rhoX (the two-cell mean; the outflow face the last cell's, a backflow the
// ambient's), rho_a = the ambient's. The quiescent ambient's hydrostatic
// pressure is thus p = 0 exactly (the open faces' p = 0 is its pressure).
template <bool OY, bool OZ>
__device__ inline Real addBuoyancyLm(const LmK& K, const Real* rho, Real m, int i,
                                     int j, int k, Real R)
{
    return R - K.grav * (rhoX<OY, OZ>(K, rho, m, i, j, k) - K.ambRho);
}

// The face pressure of an open face carrying m along its OUTWARD normal
// (mOut < 0: inflow): -beta rho_amb u_n^2 / 2 on inflow (the total-pressure
// inflow, Moghadam's stabilization), 0 on outflow.
__device__ inline Real openFaceP(const LmK& K, Real mOut)
{
    return mOut < Real(0) ? -K.bfHalf * mOut * mOut / K.ambRho : Real(0);
}

__device__ inline Real ratioAt(const ViscView& vv, Real T)
{
    return vv.sutherland ? vv.ratio(T) : Real(1);
}

// rho D = lambda/cp = mu(T)/Pr (Le = 1) at face temperature T: the species'
// and the temperature's diffusion coefficient.
__device__ inline Real rhoDAt(const LmK& K, Real T)
{
    return K.vv.mu * ratioAt(K.vv, T) * K.invPr;
}

// The x face between cells iL and iL + 1 of the row of cL (= cell iL).
__device__ inline Face faceAlongX(const GpuGrid& g, std::size_t cL, int iL)
{
    const bool fL = iL == 0, fR = iL + 1 == g.nx - 1;
    return {cL, cL + 1, fL ? cL : cL - 1, fR ? cL + 1 : cL + 2, fL, fR};
}

// The face between cells at coordinate q - 1 and q along lateral axis A (1 =
// y, 2 = z) of column (i, j, k) (q = j or k). Periodic: the wrapped stencil
// (S5's facePeriodic); open: the cells clamp and an end cell's missing
// second neighbour flags the face (central when that side is upwind).
template <bool OY, bool OZ, int A>
__device__ inline Face faceLat(const GpuGrid& g, int i, int j, int k)
{
    using L = Lat<OY, OZ>;
    auto at = [&](int d) {
        return A == 1 ? L::cell(g, i, j + d, k) : L::cell(g, i, j, k + d);
    };
    Face f{at(-1), at(0), at(-2), at(1), false, false};
    if constexpr (A == 1 ? OY : OZ) {
        const int q = A == 1 ? j : k, n = A == 1 ? g.ny : g.nz;
        f.firstL = q - 1 == 0;
        f.firstR = q == n - 1;
    }
    return f;
}

// The stage write a + (cb (b - a) + cr r): the increment form of ca a + cb b
// + cr r (ca == 1 - cb exactly), so a field uniform in a and b with r = 0
// stays bitwise uniform (the plain form drifted the uniform burnt density by
// an ulp per step, a coherent mass drift).
__device__ inline Real lmPut(Real a, Real b, Real cb, Real cr, Real r)
{
    return a + (cb * (b - a) + cr * r);
}

// -div m of cell (i, j, k) from the face mass fluxes, accumulated x, y, z as
// the continuity and the projection read it (one expression).
template <bool OY, bool OZ>
__device__ inline Real negDivM(const GpuGrid& g, const LmM& M, int i, int j, int k)
{
    using L = Lat<OY, OZ>;
    Real d = Real(0);
    d -= (M.x[L::fx(g, i + 1, j, k)] - M.x[L::fx(g, i, j, k)]) / g.hx;
    d -= (L::my(g, M, i, j + 1, k) - L::my(g, M, i, j, k)) / g.hy;
    d -= (L::mz(g, M, i, j, k + 1) - L::mz(g, M, i, j, k)) / g.hz;
    return d;
}

// --- The immersed tube (S8; the Wall instantiations) ------------------------
// Cell (i, j, k) is SOLID: LmPressure's face codes' kLmSolid bit (i outside
// [0, nx) is no cell: false; an open lateral clamps like every other read).
// A MAC face with a solid cell on either side is CLOSED -- for an interior
// face exactly the complement of the codes' face-open bit (that bit is set
// iff both cells are fluid): m = 0 there in every stage (the predictor writes
// 0, the correction skips it), so the face transports no mass, momentum,
// enthalpy or species. The wall's other faces: no-slip (a tangential
// neighbour face that is closed reads as the antimirror of the face itself:
// u = 0 on the wall), isothermal (conduction rho D(T_w) (T_c - T_w)/(h/2),
// the inflow face's form), species-tight (zero flux).
template <bool OY, bool OZ>
__device__ inline bool solidAt(const LmK& K, int i, int j, int k)
{
    if (i < 0 || i >= K.g.nx) return false;
    return (K.code[Lat<OY, OZ>::cell(K.g, i, j, k)] & kLmSolid) != 0;
}

// x face i of row (j, k); the lo face of cell (i, j, k) along y / z (j or k
// may be the hi boundary n: both reads clamp to the last cell).
template <bool OY, bool OZ>
__device__ inline bool xClosed(const LmK& K, int i, int j, int k)
{
    return solidAt<OY, OZ>(K, i - 1, j, k) || solidAt<OY, OZ>(K, i, j, k);
}

template <bool OY, bool OZ, int A>
__device__ inline bool latClosed(const LmK& K, int i, int j, int k)
{
    return A == 1 ? solidAt<OY, OZ>(K, i, j - 1, k) || solidAt<OY, OZ>(K, i, j, k)
                  : solidAt<OY, OZ>(K, i, j, k - 1) || solidAt<OY, OZ>(K, i, j, k);
}

// A face whose upwind side's SECOND neighbour is solid has no usable second
// neighbour: the shared limiter goes central there (sharedThetaOf's
// first-side rule; the walled compressible kernel's theta-1 rule).
__device__ inline void wallFace(const LmK& K, Face& f)
{
    f.firstL = f.firstL || (K.code[f.LL] & kLmSolid) != 0;
    f.firstR = f.firstR || (K.code[f.RR] & kLmSolid) != 0;
}

// The isothermal wall's conduction coefficient rho D(T_w) over h/2.
__device__ inline Real wallCond(const LmK& K, Real h)
{
    return rhoDAt(K, K.wallT) / (Real(0.5) * h);
}

// --- LES and TFLES (M4; header comment, docs/chem.md "Low-Mach LES") ----------
// The interior faces' diffusivity of T and the species, one instantiation each:
// Molecular rho D (the S8 kernels), Eddy rho D + mu_t,f / Sc_t, Thick F E_f rho D
// + (1 - Omega_f) mu_t,f / Sc_t (no SGS flux where the sensor thickens; mut null
// = no LES); face values arithmetic means (species.hip effRhoD's operands).
enum : int
{
    kLmMolecular = 0,
    kLmEddy = 1,
    kLmThick = 2
};

struct LmEddy
{
    const Real* mut; // mu_t per cell (null: no LES)
    Real invSct;
    const Real *FE, *Om; // TFLES (null: not thickened)
};

template <int Diff>
__device__ inline Real lmEffD(const LmEddy& E, std::size_t L, std::size_t R, Real rd)
{
    if constexpr (Diff == kLmMolecular) {
        return rd;
    } else if constexpr (Diff == kLmEddy) {
        return rd + Real(0.5) * (E.mut[L] + E.mut[R]) * E.invSct;
    } else {
        Real r = Real(0.5) * (E.FE[L] + E.FE[R]) * rd;
        if (E.mut) {
            const Real om = Real(0.5) * (E.Om[L] + E.Om[R]);
            r += (Real(1) - om) * (Real(0.5) * (E.mut[L] + E.mut[R])) * E.invSct;
        }
        return r;
    }
}

// ---------------------------------------------------------------------------
// 1. Cell pass: T, Y_k, omega, mu(T), div u.

// Eof (M4 TFLES, C0 only): omega *= E/F of the cell (gpu/tfles' eof); false
// = the S8 kernel bit for bit.
template <bool TwoR, bool Phi, bool OY, bool OZ, bool Eof = false>
__global__ void kLmCellPass(LmK K, PhiRateLaw<Real> law, const Real* rho,
                            const Real* ry0, const Real* ry1, LmM M, Real* T,
                            Real* Y0, Real* Y1, Real* om, Real* mu, Real* divu,
                            const Real* eof)
{
    using L = Lat<OY, OZ>;
    const GpuGrid& g = K.g;
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Real r = rho[c];
    const Real t = K.prGas / r;
    T[c] = t;
    const Real y0 = ry0[c] / r;
    Y0[c] = y0;
    if constexpr (TwoR) {
        const Real y1 = ry1[c] / r;
        Y1[c] = y1;
        // fuel 0, oxidizer 1 (make() asserts it).
        if constexpr (Phi)
            om[c] = arrheniusRate2(K.lnA + phiRateLn(law, y0, y1), K.Ta, r, y0,
                                   y1, t);
        else
            om[c] = arrheniusRate2(K.lnA, K.Ta, r, y0, y1, t);
    } else {
        if constexpr (Eof)
            om[c] = arrheniusRate(K.lnA, K.Ta, r, y0, t) * eof[c];
        else
            om[c] = arrheniusRate(K.lnA, K.Ta, r, y0, t);
    }
    mu[c] = K.vv.mu * ratioAt(K.vv, t);
    const Real mxl = M.x[L::fx(g, i, j, k)], mxh = M.x[L::fx(g, i + 1, j, k)];
    const Real ul = mxl / rhoX<OY, OZ>(K, rho, mxl, i, j, k);
    const Real uh = mxh / rhoX<OY, OZ>(K, rho, mxh, i + 1, j, k);
    const Real myl = L::my(g, M, i, j, k), myh = L::my(g, M, i, j + 1, k);
    const Real vl = myl / rhoY<OY, OZ>(K, rho, myl, i, j, k);
    const Real vh = myh / rhoY<OY, OZ>(K, rho, myh, i, j + 1, k);
    const Real mzl = L::mz(g, M, i, j, k), mzh = L::mz(g, M, i, j, k + 1);
    const Real wl = mzl / rhoZ<OY, OZ>(K, rho, mzl, i, j, k);
    const Real wh = mzh / rhoZ<OY, OZ>(K, rho, mzh, i, j, k + 1);
    divu[c] = (uh - ul) / g.hx + (vh - vl) / g.hy + (wh - wl) / g.hz;
}

// Cell-centred velocity (frames, the cone's flank flux): the mean of the two
// face velocities per axis, the cell pass's div u faces.
template <bool OY, bool OZ>
__global__ void kLmCellVelocity(LmK K, const Real* rho, LmM M, Real* u, Real* v,
                                Real* w)
{
    using L = Lat<OY, OZ>;
    const GpuGrid& g = K.g;
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Real mxl = M.x[L::fx(g, i, j, k)], mxh = M.x[L::fx(g, i + 1, j, k)];
    const Real ul = mxl / rhoX<OY, OZ>(K, rho, mxl, i, j, k);
    const Real uh = mxh / rhoX<OY, OZ>(K, rho, mxh, i + 1, j, k);
    const Real myl = L::my(g, M, i, j, k), myh = L::my(g, M, i, j + 1, k);
    const Real vl = myl / rhoY<OY, OZ>(K, rho, myl, i, j, k);
    const Real vh = myh / rhoY<OY, OZ>(K, rho, myh, i, j + 1, k);
    const Real mzl = L::mz(g, M, i, j, k), mzh = L::mz(g, M, i, j, k + 1);
    const Real wl = mzl / rhoZ<OY, OZ>(K, rho, mzl, i, j, k);
    const Real wh = mzh / rhoZ<OY, OZ>(K, rho, mzh, i, j, k + 1);
    u[c] = Real(0.5) * (ul + uh);
    v[c] = Real(0.5) * (vl + vh);
    w[c] = Real(0.5) * (wl + wh);
}

// M4 LES: mu_t = rho (cs Delta)^2 rate(g) of cell (i, j, k) (dns/sgs.hpp; cd2 =
// (cs Delta)^2), g[q][a] = d u_q / d x_a the central difference of the
// cell-centred velocity (kLmCellVelocity) over 2 h_a; a missing neighbour is
// the cell itself (x ends, open lateral ends: Lat clamps; nz == 1: the z row
// is 0). mu += mu_t: the momentum stress reads mu + mu_t through mu.
template <bool OY, bool OZ>
__global__ void kLmNut(LmK K, Real cd2, int sigma, const Real* rho, const Real* u,
                       const Real* v, const Real* w, Real* mut, Real* mu)
{
    using L = Lat<OY, OZ>;
    const GpuGrid& g = K.g;
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Real* U[3] = {u, v, w};
    const Real ih[3] = {Real(0.5) / g.hx, Real(0.5) / g.hy, Real(0.5) / g.hz};
    const std::size_t nb[3][2] = {
        {L::cell(g, clampN(i - 1, g.nx), j, k), L::cell(g, clampN(i + 1, g.nx), j, k)},
        {L::cell(g, i, j - 1, k), L::cell(g, i, j + 1, k)},
        {L::cell(g, i, j, k - 1), L::cell(g, i, j, k + 1)}};
    Real gd[3][3];
    for (int q = 0; q < 3; ++q)
        for (int a = 0; a < 3; ++a) gd[q][a] = (U[q][nb[a][1]] - U[q][nb[a][0]]) * ih[a];
    const Real mt = rho[c] * (cd2 * sgsRate<Real>(sigma, gd));
    mut[c] = mt;
    mu[c] = mu[c] + mt;
}

// ---------------------------------------------------------------------------
// 2. Temperature predictor and the divergence target.

// The lateral axis A's part of R_T for cell (i, j, k): conv += -(ch - cl)/h,
// dif += (dh - dl)/h. An open boundary face: upwind (ambient on inflow, the
// cell's own on outflow: 0), no diffusion.
template <bool OY, bool OZ, int A, bool Wall, int Diff = kLmMolecular>
__device__ inline void latT(const LmK& K, const Real* const* Vp, int nv,
                            const Real* T, const LmM& M, int i, int j, int k,
                            Real tc, Real& conv, Real& dif, const LmEddy& E = {})
{
    using L = Lat<OY, OZ>;
    const GpuGrid& g = K.g;
    const Real h = A == 1 ? g.hy : g.hz;
    const Real ml = A == 1 ? L::my(g, M, i, j, k) : L::mz(g, M, i, j, k);
    const Real mh = A == 1 ? L::my(g, M, i, j + 1, k) : L::mz(g, M, i, j, k + 1);
    Real cl, dl, ch, dh;
    bool loOpen = false, hiOpen = false;
    if constexpr (A == 1 ? OY : OZ) {
        const int q = A == 1 ? j : k, n = A == 1 ? g.ny : g.nz;
        loOpen = q == 0;
        hiOpen = q == n - 1;
    }
    // S8: a solid neighbour = the isothermal wall face.
    bool loWall = false, hiWall = false;
    if constexpr (Wall) {
        loWall = !loOpen && latClosed<OY, OZ, A>(K, i, j, k);
        hiWall = !hiOpen && (A == 1 ? latClosed<OY, OZ, A>(K, i, j + 1, k)
                                    : latClosed<OY, OZ, A>(K, i, j, k + 1));
    }
    if (loOpen) {
        cl = ml > Real(0) ? ml * (K.ambT - tc) : Real(0);
        dl = Real(0);
    } else if (loWall) {
        cl = Real(0);
        dl = wallCond(K, h) * (tc - K.wallT);
    } else {
        Face fl = faceLat<OY, OZ, A>(g, i, j, k);
        if constexpr (Wall) wallFace(K, fl);
        const Real tl = sharedThetaOf(Vp, nv, -1, fl, ml);
        cl = ml * (faceValue2(T, fl, ml, tl) - tc);
        dl = lmEffD<Diff>(E, fl.L, fl.R, rhoDAt(K, Real(0.5) * (T[fl.L] + T[fl.R]))) *
             (tc - T[fl.L]) / h;
    }
    if (hiOpen) {
        ch = mh < Real(0) ? mh * (K.ambT - tc) : Real(0);
        dh = Real(0);
    } else if (hiWall) {
        ch = Real(0);
        dh = wallCond(K, h) * (K.wallT - tc);
    } else {
        Face fh = A == 1 ? faceLat<OY, OZ, A>(g, i, j + 1, k)
                         : faceLat<OY, OZ, A>(g, i, j, k + 1);
        if constexpr (Wall) wallFace(K, fh);
        const Real tH = sharedThetaOf(Vp, nv, -1, fh, mh);
        ch = mh * (faceValue2(T, fh, mh, tH) - tc);
        dh = lmEffD<Diff>(E, fh.L, fh.R, rhoDAt(K, Real(0.5) * (T[fh.L] + T[fh.R]))) *
             (T[fh.R] - tc) / h;
    }
    conv -= (ch - cl) / h;
    dif += (dh - dl) / h;
}

// Diff (M4): the interior faces' conduction (lmEffD); kLmMolecular = the S8
// kernel bit for bit.
template <bool TwoR, bool OY, bool OZ, bool Wall, int Diff = kLmMolecular>
__global__ void kLmTemperature(LmK K, Real ca, Real cb, Real cr,
                               const Real* rhoA, const Real* rhoB, const Real* T,
                               const Real* Y0, const Real* Y1, const Real* om,
                               LmM M, Real* Tstar, Real* D, LmEddy E)
{
    using L = Lat<OY, OZ>;
    const GpuGrid& g = K.g;
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    // The limiter's voters: the species AND T (lmVoters; issue LM8).
    const Real* Vp[kLmMaxY + 1] = {Y0, TwoR ? Y1 : T, T};
    const int nY = TwoR ? 2 : 1;
    const Real tc = T[c];
    // conv = sum over faces of the outward m (T_f - T_c) / h; dif = div of
    // (lambda/cp) grad T.
    Real conv = Real(0), dif = Real(0);
    // S8: the tube carries nothing (R_T = 0: its density stays, D = the RK
    // combination's own term, 0 for a constant solid).
    bool solid = false;
    if constexpr (Wall) solid = (K.code[c] & kLmSolid) != 0;
    if (!solid) {
        const Real h = g.hx;
        bool loWall = false, hiWall = false;
        if constexpr (Wall) {
            loWall = solidAt<OY, OZ>(K, i - 1, j, k);
            hiWall = i + 1 < g.nx && solidAt<OY, OZ>(K, i + 1, j, k);
        }
        Real cl, dl;
        if (i == 0) {
            // The inflow FACE: T fixed, convective and diffusive flux.
            const Real TIn = K.inT[L::row(g, j, k)];
            cl = M.x[L::fx(g, 0, j, k)] * (TIn - tc);
            dl = rhoDAt(K, TIn) * (tc - TIn) / (Real(0.5) * h);
        } else if (loWall) {
            cl = Real(0);
            dl = wallCond(K, h) * (tc - K.wallT);
        } else {
            Face f = faceAlongX(g, c - 1, i - 1);
            if constexpr (Wall) wallFace(K, f);
            const Real m = M.x[L::fx(g, i, j, k)];
            const Real th = sharedThetaOf(Vp, nY + 1, -1, f, m);
            cl = m * (faceValue2(T, f, m, th) - tc);
            dl = lmEffD<Diff>(E, f.L, f.R, rhoDAt(K, Real(0.5) * (T[f.L] + T[f.R]))) *
                 (tc - T[f.L]) / h;
        }
        Real ch, dh;
        if (i == g.nx - 1) {
            // The outflow: the cell's own T both ways, no diffusion; with a
            // lateral open the backflow enters as the ambient.
            ch = Real(0);
            dh = Real(0);
            if constexpr (OY || OZ) {
                const Real m = M.x[L::fx(g, g.nx, j, k)];
                if (m < Real(0)) ch = m * (K.ambT - tc);
            }
        } else if (hiWall) {
            ch = Real(0);
            dh = wallCond(K, h) * (K.wallT - tc);
        } else {
            Face f = faceAlongX(g, c, i);
            if constexpr (Wall) wallFace(K, f);
            const Real m = M.x[L::fx(g, i + 1, j, k)];
            const Real th = sharedThetaOf(Vp, nY + 1, -1, f, m);
            ch = m * (faceValue2(T, f, m, th) - tc);
            dh = lmEffD<Diff>(E, f.L, f.R, rhoDAt(K, Real(0.5) * (T[f.L] + T[f.R]))) *
                 (T[f.R] - tc) / h;
        }
        conv -= (ch - cl) / h;
        dif += (dh - dl) / h;
        latT<OY, OZ, 1, Wall, Diff>(K, Vp, nY + 1, T, M, i, j, k, tc, conv, dif, E);
        latT<OY, OZ, 2, Wall, Diff>(K, Vp, nY + 1, T, M, i, j, k, tc, conv, dif, E);
    }
    const Real ra = rhoA[c], rb = rhoB[c];
    const Real RT = solid ? Real(0) : (conv + dif + K.qcp * om[c]) / rb;
    const Real TA = K.prGas / ra;
    const Real ts = lmPut(TA, tc, cb, cr, RT);
    Tstar[c] = ts;
    // D = (rho0 - P/T*)/cr, rho0 = the combined density, WITHOUT the
    // cancellation: rho0 T0 - P = P ca cb (rho_a - rho_b)^2/(rho_a rho_b)
    // exactly (T0 = the combined P/rho), so
    //   D = rho0 R_T / T* + P ca cb (rho_a - rho_b)^2 / (rho_a rho_b cr T*)
    // (rho_a - rho_b is exact: Sterbenz). The difference form read FP32
    // rounding / cr into the projection (p noise ~ 1/dt^2, measured).
    const Real r0 = lmPut(ra, rb, cb, Real(0), Real(0));
    const Real dr = ra - rb;
    D[c] = r0 * RT / ts + (K.prGas * ca * cb) * (dr * dr) / (ra * rb * cr * ts);
}

// ---------------------------------------------------------------------------
// 3. Momentum predictor.

// The x faces: one thread per face (nx + 1 per row). Face 0 is the inflow
// (fixed m = mIn); face nx the outflow (ghost cell = the last cell: zero
// gradient of u, v, w, mu, tau_xx; p = 0 on the face -- with a lateral open,
// a backflow face takes openFaceP). Open y (z) edges: outflow zero gradient,
// inflow the ambient at rest (u_t = 0 on the face: the ghost antimirrors).
// Grav (S7, addBuoyancyLm: the model's ONE buoyancy term): -(rho_f - rho_a)
// g, rho_f = rhoX (the density the face velocity reads); false = the S6
// kernel bit for bit.
template <bool OY, bool OZ, bool Grav, bool Wall>
__global__ void kLmMomX(LmK K, Real ca, Real cb, Real cr, const Real* mxA,
                        const Real* rho, LmM M, const Real* p, const Real* mu,
                        const Real* divu, Real* out)
{
    using L = Lat<OY, OZ>;
    const GpuGrid& g = K.g;
    const std::size_t f = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nf = std::size_t(g.nx + 1) * g.ny * g.nz;
    if (f >= nf) return;
    const int i = int(f % std::size_t(g.nx + 1));
    const int r = int(f / std::size_t(g.nx + 1));
    const int j = r % g.ny, k = r / g.ny;
    if (i == 0) {
        out[f] = K.inM[std::size_t(r)];
        return;
    }
    // S8: a face touching the tube is closed.
    if constexpr (Wall) {
        if (xClosed<OY, OZ>(K, i, j, k)) {
            out[f] = Real(0);
            return;
        }
    }
    const int iL = i - 1, iR = i < g.nx ? i : g.nx - 1;
    auto uX = [&](int ii, int jj, int kk) {
        const Real m = M.x[L::fx(g, ii, jj, kk)];
        return m / rhoX<OY, OZ>(K, rho, m, ii, jj, kk);
    };
    // x: the cell-centred flux and normal stress either side.
    const Real ui = uX(i, j, k);
    const Real FL = Real(0.25) * (M.x[L::fx(g, i - 1, j, k)] + M.x[f]) *
                    (uX(i - 1, j, k) + ui);
    Real FR, vis = Real(0);
    if (i < g.nx) {
        FR = Real(0.25) * (M.x[f] + M.x[L::fx(g, i + 1, j, k)]) *
             (ui + uX(i + 1, j, k));
        const std::size_t cl = L::cell(g, iL, j, k), cc = L::cell(g, i, j, k);
        const Real tR = mu[cc] * (Real(2) * (uX(i + 1, j, k) - ui) / g.hx -
                                  Real(2.0 / 3.0) * divu[cc]);
        const Real tL = mu[cl] * (Real(2) * (ui - uX(i - 1, j, k)) / g.hx -
                                  Real(2.0 / 3.0) * divu[cl]);
        vis = (tR - tL) / g.hx;
    } else {
        FR = M.x[f] * ui;
    }
    Real R = -(FR - FL) / g.hx + vis;
    // y edges (rows jj - 1 | jj at this x face), jj = j and j + 1.
    auto edgeY = [&](int jj, Real& F, Real& tau) {
        const Real mL = L::my(g, M, iL, jj, k), mR = L::my(g, M, iR, jj, k);
        const Real vL = mL / rhoY<OY, OZ>(K, rho, mL, iL, jj, k);
        const Real vR = mR / rhoY<OY, OZ>(K, rho, mR, iR, jj, k);
        Real u0 = uX(i, jj - 1, k), u1 = uX(i, jj, k);
        if constexpr (OY) {
            if (jj == 0 && mL + mR > Real(0)) u0 = -u1;
            if (jj == g.ny && mL + mR < Real(0)) u1 = -u0;
        }
        // S8: a closed neighbour face is the no-slip wall's antimirror.
        if constexpr (Wall) {
            if (jj == j) {
                if (xClosed<OY, OZ>(K, i, j - 1, k)) u0 = -u1;
            } else if (xClosed<OY, OZ>(K, i, j + 1, k)) {
                u1 = -u0;
            }
        }
        F = Real(0.25) * (mL + mR) * (u0 + u1);
        const Real me = Real(0.25) * (mu[L::cell(g, iL, jj - 1, k)] + mu[L::cell(g, iL, jj, k)] +
                                      mu[L::cell(g, iR, jj - 1, k)] + mu[L::cell(g, iR, jj, k)]);
        tau = me * ((u1 - u0) / g.hy + (vR - vL) / g.hx);
    };
    auto edgeZ = [&](int kk, Real& F, Real& tau) {
        const Real mL = L::mz(g, M, iL, j, kk), mR = L::mz(g, M, iR, j, kk);
        const Real wL = mL / rhoZ<OY, OZ>(K, rho, mL, iL, j, kk);
        const Real wR = mR / rhoZ<OY, OZ>(K, rho, mR, iR, j, kk);
        Real u0 = uX(i, j, kk - 1), u1 = uX(i, j, kk);
        if constexpr (OZ) {
            if (kk == 0 && mL + mR > Real(0)) u0 = -u1;
            if (kk == g.nz && mL + mR < Real(0)) u1 = -u0;
        }
        if constexpr (Wall) {
            if (kk == k) {
                if (xClosed<OY, OZ>(K, i, j, k - 1)) u0 = -u1;
            } else if (xClosed<OY, OZ>(K, i, j, k + 1)) {
                u1 = -u0;
            }
        }
        F = Real(0.25) * (mL + mR) * (u0 + u1);
        const Real me = Real(0.25) * (mu[L::cell(g, iL, j, kk - 1)] + mu[L::cell(g, iL, j, kk)] +
                                      mu[L::cell(g, iR, j, kk - 1)] + mu[L::cell(g, iR, j, kk)]);
        tau = me * ((u1 - u0) / g.hz + (wR - wL) / g.hx);
    };
    Real F0, F1, t0, t1;
    edgeY(j, F0, t0);
    edgeY(j + 1, F1, t1);
    R += -(F1 - F0) / g.hy + (t1 - t0) / g.hy;
    edgeZ(k, F0, t0);
    edgeZ(k + 1, F1, t1);
    R += -(F1 - F0) / g.hz + (t1 - t0) / g.hz;
    Real gp;
    if (i < g.nx) {
        gp = (p[L::cell(g, i, j, k)] - p[L::cell(g, iL, j, k)]) / g.hx;
    } else {
        Real pf = Real(0);
        if constexpr (OY || OZ) pf = openFaceP(K, M.x[f]);
        gp = (pf - p[L::cell(g, iL, j, k)]) / (Real(0.5) * g.hx);
    }
    R -= gp;
    if constexpr (Grav) R = addBuoyancyLm<OY, OZ>(K, rho, M.x[f], i, j, k, R);
    out[f] = lmPut(mxA[f], M.x[f], cb, cr, R);
}

// The tangential faces of axis A (1 = y, 2 = z; B the other one): one thread
// per cell, its lo face; an OPEN axis A adds one thread per hi boundary face
// (j = ny: out to outHi). x edges: the inflow face is a tangential no-slip
// face (antimirror ghost: u_t = 0 there), the outflow zero-gradient (with a
// lateral open, a backflow edge antimirrors: the ambient at rest). An open
// A face: zero normal-stress gradient (traction-free), face pressure
// openFaceP; open B edges: outflow zero gradient, inflow u_t = 0.
// Turb (M5): the inflow face's tangential velocity is the row's (K.inV / K.inW at
// row (j, k): the x edge of this A face), vl = 2 v_in - v_r; false: the antimirror.
template <int A, bool OY, bool OZ, bool Wall, bool Turb = false>
__global__ void kLmMomT(LmK K, Real ca, Real cb, Real cr, const Real* mAa,
                        const Real* mAaHi, const Real* rho, LmM M, const Real* p,
                        const Real* mu, const Real* divu, Real* out, Real* outHi)
{
    using L = Lat<OY, OZ>;
    constexpr bool OA = A == 1 ? OY : OZ;
    constexpr bool OB = A == 1 ? OZ : OY;
    constexpr bool Op = OY || OZ;
    const GpuGrid& g = K.g;
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i, j, k;
    std::size_t hiAt = 0;
    bool hi = false;
    if (t < g.size()) {
        decompose(g, t, i, j, k);
    } else {
        if constexpr (!OA) {
            return;
        } else {
            const std::size_t q = t - g.size();
            const std::size_t nh = std::size_t(g.nx) * (A == 1 ? g.nz : g.ny);
            if (q >= nh) return;
            i = int(q % std::size_t(g.nx));
            const int o = int(q / std::size_t(g.nx));
            j = A == 1 ? g.ny : o;
            k = A == 1 ? o : g.nz;
            hi = true;
            hiAt = q;
        }
    }
    const Real hA = A == 1 ? g.hy : g.hz, hB = A == 1 ? g.hz : g.hy;
    // Coordinates of the cell offset a along A, b along B (i given).
    auto jj = [&](int a, int b) { return A == 1 ? j + a : j + b; };
    auto kk = [&](int a, int b) { return A == 1 ? k + b : k + a; };
    // S8: the A face of x cell ii at B offset b is closed (touches the tube).
    [[maybe_unused]] auto aClosed = [&](int ii, int b) {
        return latClosed<OY, OZ, A>(K, ii, jj(0, b), kk(0, b));
    };
    if constexpr (Wall) {
        if (aClosed(i, 0)) {
            if (hi) outHi[hiAt] = Real(0);
            else out[t] = Real(0);
            return;
        }
    }
    auto cell = [&](int ii, int a, int b) { return L::cell(g, ii, jj(a, b), kk(a, b)); };
    // The A face at A offset a (B cell b); the B face at B offset b (A cell a).
    auto mAf = [&](int ii, int a, int b) {
        return A == 1 ? L::my(g, M, ii, jj(a, b), kk(a, b))
                      : L::mz(g, M, ii, jj(a, b), kk(a, b));
    };
    auto mBf = [&](int ii, int a, int b) {
        return A == 1 ? L::mz(g, M, ii, jj(a, b), kk(a, b))
                      : L::my(g, M, ii, jj(a, b), kk(a, b));
    };
    auto vA = [&](int ii, int a, int b) {
        const Real m = mAf(ii, a, b);
        return m / (A == 1 ? rhoY<OY, OZ>(K, rho, m, ii, jj(a, b), kk(a, b))
                           : rhoZ<OY, OZ>(K, rho, m, ii, jj(a, b), kk(a, b)));
    };
    auto vB = [&](int ii, int a, int b) {
        const Real m = mBf(ii, a, b);
        return m / (A == 1 ? rhoZ<OY, OZ>(K, rho, m, ii, jj(a, b), kk(a, b))
                           : rhoY<OY, OZ>(K, rho, m, ii, jj(a, b), kk(a, b)));
    };
    auto uX = [&](int ii, int a, int b) {
        const Real m = M.x[L::fx(g, ii, jj(a, b), kk(a, b))];
        return m / rhoX<OY, OZ>(K, rho, m, ii, jj(a, b), kk(a, b));
    };
    // x edges at x face ii (cells ii - 1 | ii).
    auto edgeX = [&](int ii, Real& F, Real& tau) {
        const bool in = ii == 0, outE = ii == g.nx;
        const int cl = in ? 0 : ii - 1, cR = outE ? g.nx - 1 : ii;
        const Real vl0 = vA(cl, 0, 0), vr0 = vA(cR, 0, 0);
        const Real mx0 = M.x[L::fx(g, ii, jj(-1, 0), kk(-1, 0))];
        const Real mx1 = M.x[L::fx(g, ii, j, k)];
        Real vl = in ? -vr0 : vl0;
        if constexpr (Turb) {
            if (in) vl = Real(2) * (A == 1 ? K.inV : K.inW)[rowAt(g, j, k)] - vr0;
        }
        Real vr = outE ? vl0 : vr0;
        if constexpr (Op) {
            if (outE && mx0 + mx1 < Real(0)) vr = -vl0;
        }
        // S8: a closed x neighbour's A face is the wall's antimirror.
        if constexpr (Wall) {
            if (!in && !outE) {
                if (ii == i) {
                    if (aClosed(i - 1, 0)) vl = -vr;
                } else if (aClosed(i + 1, 0)) {
                    vr = -vl;
                }
            }
        }
        F = Real(0.25) * (mx0 + mx1) * (vl + vr);
        const Real me = Real(0.25) * (mu[cell(cl, -1, 0)] + mu[cell(cl, 0, 0)] +
                                      mu[cell(cR, -1, 0)] + mu[cell(cR, 0, 0)]);
        tau = me * ((vr - vl) / g.hx + (uX(ii, 0, 0) - uX(ii, -1, 0)) / hA);
    };
    Real F0, F1, t0, t1;
    edgeX(i, F0, t0);
    edgeX(i + 1, F1, t1);
    Real R = -(F1 - F0) / g.hx + (t1 - t0) / g.hx;
    // A: the cell-centred flux and normal stress of the cells a = -1, 0.
    auto FA = [&](int a) {
        return Real(0.25) * (mAf(i, a, 0) + mAf(i, a + 1, 0)) *
               (vA(i, a, 0) + vA(i, a + 1, 0));
    };
    auto tAA = [&](int a) {
        const std::size_t q = cell(i, a, 0);
        return mu[q] * (Real(2) * (vA(i, a + 1, 0) - vA(i, a, 0)) / hA -
                        Real(2.0 / 3.0) * divu[q]);
    };
    bool bnd = false; // an open boundary face of A (lo q = 0 or hi q = n)
    if constexpr (OA) {
        const int q = A == 1 ? j : k, nA = A == 1 ? g.ny : g.nz;
        bnd = q == 0 || q == nA;
    }
    if (bnd)
        R += -(FA(0) - FA(-1)) / hA;
    else
        R += -(FA(0) - FA(-1)) / hA + (tAA(0) - tAA(-1)) / hA;
    // B edges (rows b - 1 | b at this A face), b = 0 and 1.
    auto edgeB = [&](int b, Real& F, Real& tau) {
        const Real mb = mBf(i, -1, b) + mBf(i, 0, b);
        Real va0 = vA(i, 0, b - 1), va1 = vA(i, 0, b);
        if constexpr (OB) {
            const int qb = (A == 1 ? k : j) + b, nB = A == 1 ? g.nz : g.ny;
            if (qb == 0 && mb > Real(0)) va0 = -va1;
            if (qb == nB && mb < Real(0)) va1 = -va0;
        }
        if constexpr (Wall) {
            if (b == 0) {
                if (aClosed(i, -1)) va0 = -va1;
            } else if (aClosed(i, 1)) {
                va1 = -va0;
            }
        }
        F = Real(0.25) * mb * (va0 + va1);
        const Real me = Real(0.25) * (mu[cell(i, -1, b - 1)] + mu[cell(i, -1, b)] +
                                      mu[cell(i, 0, b - 1)] + mu[cell(i, 0, b)]);
        tau = me * ((va1 - va0) / hB + (vB(i, 0, b) - vB(i, -1, b)) / hA);
    };
    edgeB(0, F0, t0);
    edgeB(1, F1, t1);
    R += -(F1 - F0) / hB + (t1 - t0) / hB;
    if constexpr (OA) {
        const int q = A == 1 ? j : k;
        if (q == 0) {
            const Real m = mAf(i, 0, 0);
            R -= (p[cell(i, 0, 0)] - openFaceP(K, -m)) / (Real(0.5) * hA);
        } else if (bnd) {
            const Real m = mAf(i, 0, 0);
            R -= (openFaceP(K, m) - p[cell(i, -1, 0)]) / (Real(0.5) * hA);
        } else {
            R -= (p[cell(i, 0, 0)] - p[cell(i, -1, 0)]) / hA;
        }
        if (hi) {
            outHi[hiAt] = lmPut(mAaHi[hiAt], mAf(i, 0, 0), cb, cr, R);
            return;
        }
    } else {
        R -= (p[t] - p[cell(i, -1, 0)]) / hA;
    }
    out[t] = lmPut(mAa[t], mAf(i, 0, 0), cb, cr, R);
}

// ---------------------------------------------------------------------------
// 4. Projection: rhs, correction.

template <bool OY, bool OZ>
__global__ void kLmDivRhs(GpuGrid g, Real cr, LmM M, const Real* D, Real* rhs)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    rhs[c] = (-negDivM<OY, OZ>(g, M, i, j, k) - D[c]) / cr;
}

// m_x -= cr G phi: interior (phi_R - phi_L)/h, the outflow face (0 -
// phi_last)/(h/2) (p = 0 on it), the inflow face untouched (Neumann). Wall
// (S8): a face the codes close (interior: no kLmFaceXLo bit; the outflow
// face: its cell solid) keeps its 0.
template <bool Wall>
__global__ void kLmCorrectX(GpuGrid g, Real cr, const Real* phi, Real* mx,
                            const std::uint8_t* code)
{
    const std::size_t f = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nf = std::size_t(g.nx + 1) * g.ny * g.nz;
    if (f >= nf) return;
    const int i = int(f % std::size_t(g.nx + 1));
    if (i == 0) return;
    const int r = int(f / std::size_t(g.nx + 1));
    const int j = r % g.ny, k = r / g.ny;
    if constexpr (Wall) {
        if (i < g.nx ? !(code[cellAt(g, i, j, k)] & kLmFaceXLo)
                     : (code[cellAt(g, g.nx - 1, j, k)] & kLmSolid) != 0)
            return;
    }
    const Real gp = i < g.nx ? (phi[cellAt(g, i, j, k)] - phi[cellAt(g, i - 1, j, k)]) / g.hx
                             : (Real(0) - phi[cellAt(g, i - 1, j, k)]) / (Real(0.5) * g.hx);
    mx[f] -= cr * gp;
}

// The y/z faces: the cell's lo face; an open axis' boundary faces (phi = 0 on
// the face, h/2 away): the lo face of the first cell and the hi face (myHi /
// mzHi) of the last, each written by its one adjacent cell.
template <bool OY, bool OZ, bool Wall>
__global__ void kLmCorrectC(GpuGrid g, Real cr, const Real* phi, const Real* pB,
                            Real* my, Real* mz, Real* myHi, Real* mzHi, Real* p,
                            const std::uint8_t* code)
{
    using L = Lat<OY, OZ>;
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Real pc = phi[c];
    // S8: which of the cell's faces the codes leave open (an open lateral's
    // boundary face: open unless its cell is solid; 2D z: no code bit, its
    // correction is the cell minus itself, 0, either way).
    bool yLo = true, zLo = true, hiOpen = true;
    if constexpr (Wall) {
        const std::uint8_t q = code[c];
        hiOpen = !(q & kLmSolid);
        yLo = OY && j == 0 ? hiOpen : (q & kLmFaceYLo) != 0;
        zLo = OZ && k == 0 ? hiOpen : (q & kLmFaceZLo) != 0;
    }
    if constexpr (OY) {
        if (!yLo)
            ;
        else if (j == 0)
            my[c] -= cr * (pc - Real(0)) / (Real(0.5) * g.hy);
        else
            my[c] -= cr * (pc - phi[L::cell(g, i, j - 1, k)]) / g.hy;
        if (j == g.ny - 1 && hiOpen)
            myHi[std::size_t(i) + std::size_t(g.nx) * k] -=
                cr * (Real(0) - pc) / (Real(0.5) * g.hy);
    } else {
        if (yLo) my[c] -= cr * (pc - phi[L::cell(g, i, j - 1, k)]) / g.hy;
    }
    if constexpr (OZ) {
        if (!zLo)
            ;
        else if (k == 0)
            mz[c] -= cr * (pc - Real(0)) / (Real(0.5) * g.hz);
        else
            mz[c] -= cr * (pc - phi[L::cell(g, i, j, k - 1)]) / g.hz;
        if (k == g.nz - 1 && hiOpen)
            mzHi[std::size_t(i) + std::size_t(g.nx) * j] -=
                cr * (Real(0) - pc) / (Real(0.5) * g.hz);
    } else {
        if (zLo) mz[c] -= cr * (pc - phi[L::cell(g, i, j, k - 1)]) / g.hz;
    }
    p[c] = pB[c] + pc;
}

// ---------------------------------------------------------------------------
// 5. Continuity and species on the projected flux.

// The COMPENSATED scalar words of a stage: [0] rho, [1] rho Y_0, [2] rho Y_1,
// hi and lo (Kahan low word) of a, b and the output. Measured without it
// (2026-10-31, 1D sigma 5, inflow 0.98): the preheat zone's per-step
// increments fall below half an ulp and are dropped, so the state LAGS --
// the Shvab-Zeldovich defect grew with the step count (4.8e-5 / 3.3e-4 /
// 7.7e-4 at 7.5 / 15 / 30 cells, 1.1e-3 at 5x smaller dt) and S_L fell with
// it (-1.2e-3 grid-independent, -7e-4 more at 5x smaller dt).
struct LmWords
{
    const Real *ah[3], *al[3], *bh[3], *bl[3];
    Real *oh[3], *ol[3];
};

__device__ inline void lmPutWord(const LmWords& w, int v, std::size_t c, Real cb,
                                 Real inc)
{
    const Word2 r = internal::combineWord2(cb, w.ah[v][c], w.al[v][c],
                                           w.bh[v][c], w.bl[v][c], inc);
    w.oh[v][c] = r.hi;
    w.ol[v][c] = r.lo;
}

// The stage's boundary face fluxes (DeviceLowMach::bflux_): species s at f
// + s nb; segments x-lo / x-hi (row j + ny k), y-lo / y-hi (i + nx k), z-lo /
// z-hi (i + nx j). Each written by its one adjacent cell.
struct LmBF
{
    Real* f;
    std::size_t nb, xlo, xhi, ylo, yhi, zlo, zhi;
};

// The lateral axis A's species fluxes of cell (i, j, k): div[s] -= (Fh -
// Fl)/h; an open boundary face convective upwind (ambient | own), no
// diffusion, recorded in bf.
template <bool OY, bool OZ, int A, int NY, bool Wall, int Diff = kLmMolecular>
__device__ inline void latY(const LmK& K, const Real* const* Yp,
                            const Real* const* Vp, const Real* T, const LmM& M,
                            const LmBF& bf, int i, int j, int k, std::size_t c,
                            Real* div, const LmEddy& E = {})
{
    using L = Lat<OY, OZ>;
    const GpuGrid& g = K.g;
    const Real h = A == 1 ? g.hy : g.hz;
    const Real ml = A == 1 ? L::my(g, M, i, j, k) : L::mz(g, M, i, j, k);
    const Real mh = A == 1 ? L::my(g, M, i, j + 1, k) : L::mz(g, M, i, j, k + 1);
    Real Fl[kLmMaxY], Fh[kLmMaxY];
    bool loOpen = false, hiOpen = false;
    if constexpr (A == 1 ? OY : OZ) {
        const int q = A == 1 ? j : k, n = A == 1 ? g.ny : g.nz;
        loOpen = q == 0;
        hiOpen = q == n - 1;
    }
    // S8: a face to the tube carries nothing (species-tight).
    bool loWall = false, hiWall = false;
    if constexpr (Wall) {
        loWall = !loOpen && latClosed<OY, OZ, A>(K, i, j, k);
        hiWall = !hiOpen && (A == 1 ? latClosed<OY, OZ, A>(K, i, j + 1, k)
                                    : latClosed<OY, OZ, A>(K, i, j, k + 1));
    }
    if (loOpen) {
        const std::size_t e = (A == 1 ? bf.ylo : bf.zlo) + std::size_t(i) +
                              std::size_t(g.nx) * (A == 1 ? k : j);
        for (int s = 0; s < NY; ++s) {
            Fl[s] = ml * (ml > Real(0) ? ambY(K, s) : Yp[s][c]);
            bf.f[e + std::size_t(s) * bf.nb] = Fl[s];
        }
    } else if (loWall) {
        for (int s = 0; s < NY; ++s) Fl[s] = Real(0);
    } else {
        Face fl = faceLat<OY, OZ, A>(g, i, j, k);
        if constexpr (Wall) wallFace(K, fl);
        const Real tl = sharedThetaOf(Vp, NY + 1, -1, fl, ml);
        const Real kl = lmEffD<Diff>(E, fl.L, fl.R, rhoDAt(K, Real(0.5) * (T[fl.L] + T[fl.R])));
        for (int s = 0; s < NY; ++s) Fl[s] = faceY2(Yp[s], fl, ml, tl, kl, h);
    }
    if (hiOpen) {
        const std::size_t e = (A == 1 ? bf.yhi : bf.zhi) + std::size_t(i) +
                              std::size_t(g.nx) * (A == 1 ? k : j);
        for (int s = 0; s < NY; ++s) {
            Fh[s] = mh * (mh < Real(0) ? ambY(K, s) : Yp[s][c]);
            bf.f[e + std::size_t(s) * bf.nb] = Fh[s];
        }
    } else if (hiWall) {
        for (int s = 0; s < NY; ++s) Fh[s] = Real(0);
    } else {
        Face fh = A == 1 ? faceLat<OY, OZ, A>(g, i, j + 1, k)
                         : faceLat<OY, OZ, A>(g, i, j, k + 1);
        if constexpr (Wall) wallFace(K, fh);
        const Real tH = sharedThetaOf(Vp, NY + 1, -1, fh, mh);
        const Real kh = lmEffD<Diff>(E, fh.L, fh.R, rhoDAt(K, Real(0.5) * (T[fh.L] + T[fh.R])));
        for (int s = 0; s < NY; ++s) Fh[s] = faceY2(Yp[s], fh, mh, tH, kh, h);
    }
    for (int s = 0; s < NY; ++s) div[s] -= (Fh[s] - Fl[s]) / h;
}

// Diff (M4): the interior faces' Fick coefficient (lmEffD); kLmMolecular =
// the S8 kernel bit for bit.
template <bool TwoR, bool OY, bool OZ, bool Wall, int Diff = kLmMolecular>
__global__ void kLmScalars(LmK K, Real cb, Real cr, LmWords W, const Real* T,
                           const Real* Y0, const Real* Y1, const Real* om, LmM M,
                           LmBF bf, LmEddy E)
{
    using L = Lat<OY, OZ>;
    const GpuGrid& g = K.g;
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    constexpr int nY = TwoR ? 2 : 1;
    if constexpr (Wall) {
        if (K.code[c] & kLmSolid) {
            // S8: the tube carries its words unchanged (no flux, no
            // reaction); the boundary segments it owns record 0.
            for (int s = 0; s < nY; ++s) {
                const std::size_t o = std::size_t(s) * bf.nb;
                const std::size_t r = L::row(g, j, k);
                if (i == 0) bf.f[bf.xlo + r + o] = Real(0);
                if (i == g.nx - 1) bf.f[bf.xhi + r + o] = Real(0);
                if (OY && j == 0) bf.f[bf.ylo + i + std::size_t(g.nx) * k + o] = Real(0);
                if (OY && j == g.ny - 1) bf.f[bf.yhi + i + std::size_t(g.nx) * k + o] = Real(0);
                if (OZ && k == 0) bf.f[bf.zlo + i + std::size_t(g.nx) * j + o] = Real(0);
                if (OZ && k == g.nz - 1) bf.f[bf.zhi + i + std::size_t(g.nx) * j + o] = Real(0);
            }
            for (int v = 0; v < 1 + nY; ++v) lmPutWord(W, v, c, cb, Real(0));
            return;
        }
    }
    const Real* Yp[kLmMaxY] = {Y0, Y1};
    // The limiter's voters: the species AND T (lmVoters; issue LM8).
    const Real* Vp[kLmMaxY + 1] = {Y0, TwoR ? Y1 : T, T};
    Real div[kLmMaxY] = {Real(0), Real(0)};
    {
        const Real h = g.hx;
        Real Fl[kLmMaxY], Fh[kLmMaxY];
        const std::size_t r = L::row(g, j, k);
        bool loWall = false, hiWall = false;
        if constexpr (Wall) {
            loWall = solidAt<OY, OZ>(K, i - 1, j, k);
            hiWall = solidAt<OY, OZ>(K, i + 1, j, k);
        }
        if (i == 0) {
            const Real m = M.x[L::fx(g, 0, j, k)];
            const Real kd = rhoDAt(K, K.inT[r]);
            for (int s = 0; s < nY; ++s) {
                const Real yi = inY(K, s, r);
                Fl[s] = m * yi - kd * (Yp[s][c] - yi) / (Real(0.5) * h);
                bf.f[bf.xlo + r + std::size_t(s) * bf.nb] = Fl[s];
            }
        } else if (loWall) {
            for (int s = 0; s < nY; ++s) Fl[s] = Real(0);
        } else {
            Face f = faceAlongX(g, c - 1, i - 1);
            if constexpr (Wall) wallFace(K, f);
            const Real m = M.x[L::fx(g, i, j, k)];
            const Real th = sharedThetaOf(Vp, nY + 1, -1, f, m);
            const Real kd = lmEffD<Diff>(E, f.L, f.R, rhoDAt(K, Real(0.5) * (T[f.L] + T[f.R])));
            for (int s = 0; s < nY; ++s) Fl[s] = faceY2(Yp[s], f, m, th, kd, h);
        }
        if (i == g.nx - 1) {
            const Real m = M.x[L::fx(g, g.nx, j, k)];
            for (int s = 0; s < nY; ++s) {
                if constexpr (OY || OZ)
                    Fh[s] = m * (m < Real(0) ? ambY(K, s) : Yp[s][c]);
                else
                    Fh[s] = m * Yp[s][c];
                bf.f[bf.xhi + r + std::size_t(s) * bf.nb] = Fh[s];
            }
        } else if (hiWall) {
            for (int s = 0; s < nY; ++s) Fh[s] = Real(0);
        } else {
            Face f = faceAlongX(g, c, i);
            if constexpr (Wall) wallFace(K, f);
            const Real m = M.x[L::fx(g, i + 1, j, k)];
            const Real th = sharedThetaOf(Vp, nY + 1, -1, f, m);
            const Real kd = lmEffD<Diff>(E, f.L, f.R, rhoDAt(K, Real(0.5) * (T[f.L] + T[f.R])));
            for (int s = 0; s < nY; ++s) Fh[s] = faceY2(Yp[s], f, m, th, kd, h);
        }
        for (int s = 0; s < nY; ++s) div[s] -= (Fh[s] - Fl[s]) / h;
    }
    latY<OY, OZ, 1, nY, Wall, Diff>(K, Yp, Vp, T, M, bf, i, j, k, c, div, E);
    latY<OY, OZ, 2, nY, Wall, Diff>(K, Yp, Vp, T, M, bf, i, j, k, c, div, E);
    const Real dm = negDivM<OY, OZ>(g, M, i, j, k);
    lmPutWord(W, 0, c, cb, cr * dm);
    const Real w = om[c];
    lmPutWord(W, 1, c, cb, cr * (div[0] - w));
    if constexpr (TwoR) lmPutWord(W, 2, c, cb, cr * (div[1] - K.stoich * w));
}

// The FP64 mass tally of the stage: out = ca a + cb b + cr (in - out) area,
// the face sums in a fixed order (one block, strided per thread, shared
// tree: deterministic, no atomics). Open laterals add their faces (a second
// fixed-order tree; periodic: the S5 expression).
constexpr int kTallyThreads = 256;

__device__ inline double blockSum(double* s, double v)
{
    const int t = int(threadIdx.x);
    s[t] = v;
    __syncthreads();
    for (int w = kTallyThreads / 2; w > 0; w >>= 1) {
        if (t < w) s[t] += s[t + w];
        __syncthreads();
    }
    const double r = s[0];
    __syncthreads();
    return r;
}

template <bool OY, bool OZ>
__global__ void kLmTally(GpuGrid g, double ca, double cb, double cr,
                         const double* ta, const double* tb, LmM M, double area,
                         double* to)
{
    __shared__ double s[kTallyThreads];
    const int t = int(threadIdx.x);
    const int nA = g.ny * g.nz;
    double acc = 0;
    for (int q = t; q < nA; q += kTallyThreads) {
        const int j = q % g.ny, k = q / g.ny;
        acc += double(M.x[faceX(g, 0, j, k)]) - double(M.x[faceX(g, g.nx, j, k)]);
    }
    s[t] = acc;
    __syncthreads();
    for (int w = kTallyThreads / 2; w > 0; w >>= 1) {
        if (t < w) s[t] += s[t + w];
        __syncthreads();
    }
    if constexpr (OY || OZ) {
        const double sx = s[0];
        __syncthreads();
        double ay = 0, az = 0;
        if constexpr (OY)
            for (int q = t; q < g.nx * g.nz; q += kTallyThreads) {
                const int i = q % g.nx, k = q / g.nx;
                ay += double(M.y[cellAt(g, i, 0, k)]) - double(M.yHi[q]);
            }
        if constexpr (OZ)
            for (int q = t; q < g.nx * g.ny; q += kTallyThreads) {
                const int i = q % g.nx, j = q / g.nx;
                az += double(M.z[cellAt(g, i, j, 0)]) - double(M.zHi[q]);
            }
        const double sy = blockSum(s, ay), sz = blockSum(s, az);
        const double flux = sx * area + sy * (double(g.hx) * g.hz) +
                            sz * (double(g.hx) * g.hy);
        if (t == 0) to[0] = ta[0] + (cb * (tb[0] - ta[0]) + cr * flux);
    } else {
        if (t == 0) to[0] = ta[0] + (cb * (tb[0] - ta[0]) + cr * (s[0] * area));
    }
}

// The species tallies (S6): the stage's boundary face fluxes (bf, written by
// kLmScalars) in, out, times the face areas, with the RK weights.
template <bool OY, bool OZ>
__global__ void kLmTallyY(GpuGrid g, int nY, double cb, double cr,
                          const double* ta, const double* tb, LmBF bf, double* to)
{
    __shared__ double s[kTallyThreads];
    const int t = int(threadIdx.x);
    const double aX = double(g.hy) * g.hz, aY = double(g.hx) * g.hz,
                 aZ = double(g.hx) * g.hy;
    for (int sp = 0; sp < nY; ++sp) {
        const Real* f = bf.f + std::size_t(sp) * bf.nb;
        double acc = 0;
        for (int q = t; q < g.ny * g.nz; q += kTallyThreads)
            acc += (double(f[bf.xlo + q]) - double(f[bf.xhi + q])) * aX;
        if constexpr (OY)
            for (int q = t; q < g.nx * g.nz; q += kTallyThreads)
                acc += (double(f[bf.ylo + q]) - double(f[bf.yhi + q])) * aY;
        if constexpr (OZ)
            for (int q = t; q < g.nx * g.ny; q += kTallyThreads)
                acc += (double(f[bf.zlo + q]) - double(f[bf.zhi + q])) * aZ;
        const double flux = blockSum(s, acc);
        if (t == 0) to[sp] = ta[sp] + (cb * (tb[sp] - ta[sp]) + cr * flux);
    }
}

// LmBoundarySums of a state, one block, fixed order: [0] mIn, [1] mOut, [2]
// mLat (net in), [3] uLatMin, [4] uLatMax, [5 + s] species out.
constexpr int kBSums = 5 + kLmMaxY;

template <bool OY, bool OZ>
__global__ void kLmBSums(LmK K, LmM M, const Real* rho, const Real* ry0,
                         const Real* ry1, double* out)
{
    using L = Lat<OY, OZ>;
    constexpr bool Op = OY || OZ;
    __shared__ double s[kTallyThreads];
    const GpuGrid& g = K.g;
    const int t = int(threadIdx.x);
    const double aX = double(g.hy) * g.hz, aY = double(g.hx) * g.hz,
                 aZ = double(g.hx) * g.hy;
    double a[kBSums] = {0, 0, 0, HUGE_VAL, -HUGE_VAL, 0, 0};
    auto yOf = [&](std::size_t c, int sp) {
        return double((sp == 0 ? ry0 : ry1)[c]) / double(rho[c]);
    };
    auto amb = [&](int sp) { return double(sp == 0 ? K.ambY0 : K.ambY1); };
    for (int q = t; q < g.ny * g.nz; q += kTallyThreads) {
        const int j = q % g.ny, k = q / g.ny;
        a[0] += double(M.x[L::fx(g, 0, j, k)]) * aX;
        const double mo = M.x[L::fx(g, g.nx, j, k)];
        a[1] += mo * aX;
        const std::size_t c = L::cell(g, g.nx - 1, j, k);
        for (int sp = 0; sp < K.nY; ++sp)
            a[5 + sp] += mo * ((Op && mo < 0) ? amb(sp) : yOf(c, sp)) * aX;
    }
    // An open face of cell c carrying mo along its outward normal.
    auto lat = [&](double mo, std::size_t c, double area) {
        a[2] -= mo * area;
        const double u = mo / (mo > 0 ? double(rho[c]) : double(K.ambRho));
        a[3] = fmin(a[3], u);
        a[4] = fmax(a[4], u);
        for (int sp = 0; sp < K.nY; ++sp)
            a[5 + sp] += mo * (mo > 0 ? yOf(c, sp) : amb(sp)) * area;
    };
    if constexpr (OY)
        for (int q = t; q < g.nx * g.nz; q += kTallyThreads) {
            const int i = q % g.nx, k = q / g.nx;
            const std::size_t c0 = L::cell(g, i, 0, k), c1 = L::cell(g, i, g.ny - 1, k);
            lat(-double(M.y[c0]), c0, aY);
            lat(double(M.yHi[q]), c1, aY);
        }
    if constexpr (OZ)
        for (int q = t; q < g.nx * g.ny; q += kTallyThreads) {
            const int i = q % g.nx, j = q / g.nx;
            const std::size_t c0 = L::cell(g, i, j, 0), c1 = L::cell(g, i, j, g.nz - 1);
            lat(-double(M.z[c0]), c0, aZ);
            lat(double(M.zHi[q]), c1, aZ);
        }
    for (int v = 0; v < kBSums; ++v) {
        s[t] = a[v];
        __syncthreads();
        for (int w = kTallyThreads / 2; w > 0; w >>= 1) {
            if (t < w)
                s[t] = v == 3   ? fmin(s[t], s[t + w])
                       : v == 4 ? fmax(s[t], s[t + w])
                                : s[t] + s[t + w];
            __syncthreads();
        }
        if (t == 0) out[v] = s[0];
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Reductions (FP64 functors) and the column sums.

template <bool OY, bool OZ>
struct LmConvRate
{
    GpuGrid g;
    LmM M;
    const Real *rho, *inRho;
    double ambRho;
    using L = Lat<OY, OZ>;
    __host__ __device__ double rhoXd(int i, int j, int k, double m) const
    {
        if (i == 0) return inRho[L::row(g, j, k)];
        if (i == g.nx) {
            if ((OY || OZ) && m < 0) return ambRho;
            return rho[L::cell(g, g.nx - 1, j, k)];
        }
        return 0.5 * (double(rho[L::cell(g, i - 1, j, k)]) + rho[L::cell(g, i, j, k)]);
    }
    // The density of lateral face q (axis A) of cell (i, j, k): q = the cell's
    // own coordinate (lo face) or + 1 (hi face).
    template <int A>
    __host__ __device__ double rhoLd(int i, int j, int k, int q, double m) const
    {
        constexpr bool O = A == 1 ? OY : OZ;
        const int n = A == 1 ? g.ny : g.nz;
        auto at = [&](int qq) {
            return double(A == 1 ? rho[L::cell(g, i, qq, k)] : rho[L::cell(g, i, j, qq)]);
        };
        if (O && q <= 0) return m > 0 ? ambRho : at(0);
        if (O && q >= n) return m < 0 ? ambRho : at(n - 1);
        return 0.5 * (at(q - 1) + at(q));
    }
    // Per axis the larger of the cell's two face speeds, each over the
    // SMALLER of the face's and the cell's density (S5): a hot cell beside a
    // cold one (rho 1/7 vs 1) exchanges m dt/(rho_c h) of its own mass per
    // step -- the face-mean density read 4x low there, and the 4-cell
    // partially premixed round nozzle over/undershot T and Y (T 8 vs
    // sigma_st 6.98, Y_O -5e-3) or diverged at cfl 0.8 (measured).
    __host__ __device__ double operator()(int c) const
    {
        const int i = c % g.nx, j = (c / g.nx) % g.ny, k = c / (g.nx * g.ny);
        const double r = rho[c];
        auto sp = [&](double m, double rf) { return fabs(m / fmin(rf, r)); };
        const double mxl = M.x[L::fx(g, i, j, k)], mxh = M.x[L::fx(g, i + 1, j, k)];
        const double ul = sp(mxl, rhoXd(i, j, k, mxl));
        const double uh = sp(mxh, rhoXd(i + 1, j, k, mxh));
        const double myl = L::my(g, M, i, j, k), myh = L::my(g, M, i, j + 1, k);
        const double mzl = L::mz(g, M, i, j, k), mzh = L::mz(g, M, i, j, k + 1);
        const double vl = sp(myl, rhoLd<1>(i, j, k, j, myl));
        const double vh = sp(myh, rhoLd<1>(i, j, k, j + 1, myh));
        const double wl = sp(mzl, rhoLd<2>(i, j, k, k, mzl));
        const double wh = sp(mzh, rhoLd<2>(i, j, k, k + 1, mzh));
        const double s = fmax(ul, uh) / g.hx + fmax(vl, vh) / g.hy +
                         (g.nz > 1 ? fmax(wl, wh) / g.hz : 0.0);
        return isfinite(s) ? s : HUGE_VAL;
    }
};

struct LmAbsField
{
    const Real* f;
    __host__ __device__ double operator()(int c) const
    {
        const double v = fabs(double(f[c]));
        return isfinite(v) ? v : HUGE_VAL;
    }
};

struct LmMinField
{
    const Real* f;
    __host__ __device__ double operator()(int c) const
    {
        const double v = f[c];
        return isfinite(v) ? v : -HUGE_VAL;
    }
};

// J of the cp form: C0 k (1 + q/cp Y Ta/T^2); two reactants k rho (Y_O + s
// Y_F + q/cp Y_F Y_O Ta/T^2), k = e^(lnA (+ ln C(xi)) - Ta/T), Y floored at 0
// as the rate floors them (the unfloored Y feed ln C, as the cell pass).
template <bool TwoR, bool Phi>
struct LmChemJ
{
    const Real *rho, *r0, *r1;
    double prGas, lnA, Ta, qcp, s;
    PhiRateLaw<double> law;
    __host__ __device__ double operator()(int c) const
    {
        const double r = rho[c];
        const double T = prGas / r;
        double J;
        if constexpr (TwoR) {
            const double YF = r0[c] / r, YO = r1[c] / r;
            const double yf = fmax(YF, 0.0), yo = fmax(YO, 0.0);
            double la = lnA;
            if constexpr (Phi) la += phiRateLn(law, YF, YO);
            const double k = exp(la - Ta / T);
            J = k * r * (yo + s * yf + qcp * yf * yo * Ta / (T * T));
        } else {
            const double Y = fmax(r0[c] / r, 0.0);
            const double k = exp(lnA - Ta / T);
            J = k * (1.0 + qcp * Y * Ta / (T * T));
        }
        return isfinite(J) ? J : HUGE_VAL;
    }
};

// The per-cell transport positivity rate of `--dt-rule cell` (masterplan
// real_bunsen S9, issues LM10): the explicit transport step keeps a cell's
// own T and Y_k coefficient non-negative when dt (2 C_c + D_c) <= 1 -- C_c
// the convective rate of LmConvRate (x2: the MC limiter's face value reaches
// twice the upwind slope), D_c = 2 sum_a 1/h_a^2 mu(T_c) max(4/3, 1/Pr)/rho_c
// (the explicit Laplacian of the cell's OWN diffusivity; at the hottest cell
// exactly 1/diffusiveDt). The separate bounds take each term's max over the
// box, each spending the whole budget: the hot, fast products hold both at
// once. Chemistry keeps its own K / J bound (adding J_c here, or only its
// consumption part, cost 1.5x in dt and bounded nothing more: measured).
template <bool OY, bool OZ>
struct LmPosRate
{
    LmConvRate<OY, OZ> conv;
    Transport tr;
    double prGas;
    double dcoef; // 2 sum_a 1/h_a^2 max(4/3, 1/Pr)
    __host__ __device__ double operator()(int c) const
    {
        const double r = conv.rho[c];
        const double s = 2.0 * conv(c) + dcoef * tr.mu(prGas / r) / r;
        return isfinite(s) ? s : HUGE_VAL;
    }
};

// M4: LmPosRate with the eddy fields: D_c = dcoef2 (mu(T_c) max(4/3, FE_c/Pr)
// + mu_t,c max(4/3, 1/Sc_t)) / rho_c, dcoef2 = 2 sum_a 1/h_a^2 (null FE: 1,
// null mut: 0). Off: LmPosRate.
template <bool OY, bool OZ>
struct LmPosRateEddy
{
    LmConvRate<OY, OZ> conv;
    Transport tr;
    double prGas;
    double dcoef2, invPr, fT; // fT = max(4/3, 1/Sc_t)
    const Real *mut, *FE;
    __host__ __device__ double operator()(int c) const
    {
        const double r = conv.rho[c];
        const double fe = FE ? double(FE[c]) : 1.0;
        const double mt = mut ? double(mut[c]) : 0.0;
        const double d = tr.mu(prGas / r) * fmax(4.0 / 3.0, fe * invPr) + mt * fT;
        const double s = 2.0 * conv(c) + dcoef2 * d / r;
        return isfinite(s) ? s : HUGE_VAL;
    }
};

template <bool OY, bool OZ>
struct LmDivErr
{
    GpuGrid g;
    LmM M;
    const Real* D;
    using L = Lat<OY, OZ>;
    __host__ __device__ double operator()(int c) const
    {
        const int i = c % g.nx, j = (c / g.nx) % g.ny, k = c / (g.nx * g.ny);
        double d = 0;
        d += (double(M.x[L::fx(g, i + 1, j, k)]) - M.x[L::fx(g, i, j, k)]) / g.hx;
        d += (double(L::my(g, M, i, j + 1, k)) - L::my(g, M, i, j, k)) / g.hy;
        d += (double(L::mz(g, M, i, j, k + 1)) - L::mz(g, M, i, j, k)) / g.hz;
        const double e = fabs(d - D[c]);
        return isfinite(e) ? e : HUGE_VAL;
    }
};

struct LmEosErr
{
    const Real *rho, *Ts;
    double prGas;
    __host__ __device__ double operator()(int c) const
    {
        const double e = fabs(double(rho[c]) * Ts[c] - prGas) / prGas;
        return isfinite(e) ? e : HUGE_VAL;
    }
};

struct LmColK
{
    GpuGrid g;
    const Real* inRho;
    double qcp, Hin, zS, zO2, zDen;
    int fuel, ox;
};

__global__ void kLmColumns(LmColK C, const Real* rho, const Real* rlo,
                           const Real* mx, const Real* T, const Real* ryF,
                           const Real* ryFlo, const Real* om,
                           const Real* p, const Real* YF, const Real* YO,
                           double* out)
{
    const GpuGrid& g = C.g;
    const int i = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (i >= g.nx) return;
    double s[kLmSums] = {};
    s[9] = HUGE_VAL;
    s[10] = -HUGE_VAL;
    s[11] = C.zDen > 0 ? HUGE_VAL : 0.0;
    s[12] = C.zDen > 0 ? -HUGE_VAL : 0.0;
    auto rx = [&](int ii, int j, int k) {
        if (ii == 0) return double(C.inRho[rowAt(g, j, k)]);
        if (ii == g.nx) return double(rho[cellAt(g, g.nx - 1, j, k)]);
        return 0.5 * (double(rho[cellAt(g, ii - 1, j, k)]) + rho[cellAt(g, ii, j, k)]);
    };
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            const std::size_t c = cellAt(g, i, j, k);
            const double ml = mx[faceX(g, i, j, k)], mh = mx[faceX(g, i + 1, j, k)];
            const double yf = YF[c];
            s[0] += double(rho[c]) + double(rlo[c]);
            s[1] += 0.5 * (ml + mh);
            s[2] += T[c];
            s[3] += double(ryF[c]) + double(ryFlo[c]);
            s[4] += om[c];
            s[5] = fmax(s[5], fabs(double(T[c]) + C.qcp * yf - C.Hin));
            s[6] += 0.5 * (ml / rx(i, j, k) + mh / rx(i + 1, j, k));
            s[7] += p[c];
            s[8] += (yf < -1e-6 || yf > 1 + 1e-6) ? 1.0 : 0.0;
            s[9] = fmin(s[9], yf);
            s[10] = fmax(s[10], yf);
            if (C.zDen > 0) {
                const double z = (C.zS * yf - double(YO[c]) + C.zO2) / C.zDen;
                s[11] = fmin(s[11], z);
                s[12] = fmax(s[12], z);
            }
        }
    for (int q = 0; q < kLmSums; ++q) out[std::size_t(i) * kLmSums + q] = s[q];
}

// in: the inflow rows (m, rho, T, Y_0, Y_1; Y_1 null with one species).
LmK kernelConsts(const GpuGrid& g, const LowMachSetup& s,
                 const std::array<const Real*, 5>& in, const std::uint8_t* code)
{
    LmK K{};
    K.code = code;
    K.wallT = Real(s.wallT);
    K.g = g;
    K.vv = viscView(s.tr);
    K.invPr = Real(1.0 / s.tr.Pr);
    K.prGas = Real(s.prGas);
    K.inM = in[0];
    K.inRho = in[1];
    K.inT = in[2];
    K.inY0 = in[3];
    K.inY1 = in[4];
    K.nY = s.nY;
    K.fuel = s.fuel;
    K.ox = s.ox;
    K.stoich = Real(s.stoich);
    K.lnA = Real(s.lnA);
    K.Ta = Real(s.Ta);
    K.qcp = Real(s.qcp);
    K.ambT = Real(s.ambT);
    K.ambRho = Real(s.prGas / s.ambT);
    K.ambY0 = Real(s.ambY[0]);
    K.ambY1 = Real(s.ambY[1]);
    K.bfHalf = Real(0.5 * s.backflow);
    K.grav = Real(s.gravity);
    return K;
}

hipError_t alloc(DeviceScalar& b, std::size_t n)
{
    auto m = DeviceScalar::make(n);
    if (!m) return m.error();
    b = std::move(*m);
    return hipSuccess;
}

// The lateral instantiation of a call: f(std::bool_constant<OY>,
// std::bool_constant<OZ>) for (false, false), (true, false), (true, true)
// (make() rejects openZ without openY).
template <class F>
void withLat(const LowMachSetup& s, F&& f)
{
    if (s.openY && s.openZ) f(std::true_type{}, std::true_type{});
    else if (s.openY) f(std::true_type{}, std::false_type{});
    else f(std::false_type{}, std::false_type{});
}

// The tube's instantiation (S8): f(std::bool_constant<immersed>).
template <class F>
void withWall(bool immersed, F&& f)
{
    if (immersed) f(std::true_type{});
    else f(std::false_type{});
}

LmM massFluxes(const DeviceLowMach::State& s)
{
    return {s.mx.data(), s.my.data(), s.mz.data(),
            s.myHi.size() ? s.myHi.data() : nullptr,
            s.mzHi.size() ? s.mzHi.data() : nullptr};
}

// The bflux_ layout of grid g (LmBF without the pointer).
LmBF bfLayout(const GpuGrid& g, Real* f)
{
    LmBF b{};
    b.f = f;
    const std::size_t nyz = std::size_t(g.ny) * g.nz, nxz = std::size_t(g.nx) * g.nz,
                      nxy = std::size_t(g.nx) * g.ny;
    b.xlo = 0;
    b.xhi = nyz;
    b.ylo = 2 * nyz;
    b.yhi = b.ylo + nxz;
    b.zlo = b.yhi + nxz;
    b.zhi = b.zlo + nxy;
    b.nb = b.zhi + nxy;
    return b;
}

} // namespace

std::expected<DeviceLowMach, std::string> DeviceLowMach::make(const Grid& g,
                                                              const LowMachSetup& s)
{
    if (g.nx % 2 != 0 || g.nx < 4)
        return std::unexpected(std::format("low-Mach: nx {} must be even and >= 4 "
                                           "(the x DCT-IV)",
                                           g.nx));
    if (s.nY < 1 || s.nY > kLmMaxY || s.fuel != 0 ||
        (s.nY == 2 && s.ox != 1) || (s.nY == 1 && s.ox >= 0))
        return std::unexpected("low-Mach: species layout must be fuel 0 "
                               "(and oxidizer 1 with two reactants)");
    if (g.ny < 4 || (g.nz != 1 && g.nz < 4))
        return std::unexpected("low-Mach: ny >= 4 and nz == 1 or >= 4");
    if (s.openZ && (!s.openY || g.nz == 1))
        return std::unexpected("low-Mach: open z laterals need open y and nz > 1 "
                               "(2D is nz == 1: z degenerate, periodic)");
    if ((s.openY || s.openZ) &&
        !(std::isfinite(s.ambT) && s.ambT > 0 && std::isfinite(s.backflow) &&
          s.backflow >= 0))
        return std::unexpected("low-Mach: open laterals need an ambient T > 0 and "
                               "a backflow beta >= 0");
    if (!(std::isfinite(s.gravity) && s.gravity >= 0 && std::isfinite(s.ambT) &&
          s.ambT > 0))
        return std::unexpected("low-Mach: gravity must be finite and >= 0 (along "
                               "-x) with an ambient T > 0 (rho_a = P0/(R T_amb))");
    if (s.turb && (!s.solid.empty() || s.openY || s.openZ))
        return std::unexpected("low-Mach: --turb is the planar box's (no tube, periodic laterals: the synthetic "
                               "box is periodic in y and z)");
    SepBox box;
    box.nx = g.nx;
    box.ny = g.ny;
    box.nz = g.nz;
    box.hx = g.hx;
    box.hy = g.hy;
    box.hz = g.hz;
    box.kx = SepKind::ND;
    // Open laterals: p = phi = 0 on the face (Dirichlet both ends, DD).
    box.ky = s.openY ? SepKind::DD : SepKind::Periodic;
    box.kz = s.openZ ? SepKind::DD : SepKind::Periodic;
    // S8: the tube's cells are the solid of the apertured solve (its face
    // codes are the kernels' face-open definition).
    if (!s.solid.empty() && s.solid.size() != g.size())
        return std::unexpected(std::format("low-Mach: the solid map has {} cells, "
                                           "the grid {}",
                                           s.solid.size(), g.size()));
    if (!s.solid.empty() && !(std::isfinite(s.wallT) && s.wallT > 0))
        return std::unexpected("low-Mach: the tube's wall T must be > 0");
    // M4: LES and TFLES (dns/lmmodules' pairs; the CLI refuses them by name
    // first, these are the API's guards).
    if (!(std::isfinite(s.cs) && s.cs >= 0 && std::isfinite(s.prt) && s.prt > 0))
        return std::unexpected("low-Mach LES: Cs >= 0 and Sc_t = Pr_t > 0");
    if (s.cs > 0 && s.sgs == SgsModel::Sigma && g.nz == 1)
        return std::unexpected("low-Mach LES: sigma: D_sigma is identically 0 for any 2D state");
    if ((s.cs > 0 || s.tfles) && !s.solid.empty())
        return std::unexpected("low-Mach LES / TFLES with the immersed tube: not built (no wall "
                               "model for mu_t at the staircase; the sensor reads the solid)");
    if (s.tfles && s.nY != 1)
        return std::unexpected("low-Mach TFLES: premixed C0 only (two reactants refused, "
                               "dns/lmmodules)");
    if (s.tfles && (s.openY || s.openZ))
        return std::unexpected("low-Mach TFLES with open laterals: the sensor's max filter "
                               "wraps y and z periodic");
    if (s.tfles && s.tfles->charlette && !(s.cs > 0 && g.nz > 1))
        return std::unexpected("low-Mach TFLES: the efficiency needs LES in 3D");
    LmPressureOptions po;
    po.tol = s.pcgTol;
    auto pr = LmPressure::make(box, s.solid, po);
    if (!pr) return std::unexpected(pr.error());
    DeviceLowMach d(GpuGrid::from(g), s, std::move(*pr));
    const std::size_t n = g.size();
    const std::size_t nfx = std::size_t(g.nx + 1) * g.ny * g.nz;
    hipError_t e = hipSuccess;
    for (State* st : {&d.s_, &d.q1_, &d.q2_}) {
        if (e == hipSuccess) e = alloc(st->rho, n);
        if (e == hipSuccess) e = alloc(st->p, n);
        if (e == hipSuccess) e = alloc(st->mx, nfx);
        if (e == hipSuccess) e = alloc(st->my, n);
        if (e == hipSuccess) e = alloc(st->mz, n);
        if (e == hipSuccess && s.openY) e = alloc(st->myHi, std::size_t(g.nx) * g.nz);
        if (e == hipSuccess && s.openZ) e = alloc(st->mzHi, std::size_t(g.nx) * g.ny);
        for (int k = 0; k < s.nY && e == hipSuccess; ++k) e = alloc(st->y[k], n);
        if (e == hipSuccess) e = alloc(st->rlo, n);
        for (int k = 0; k < s.nY && e == hipSuccess; ++k) e = alloc(st->ylo[k], n);
        if (e == hipSuccess) {
            auto t = DeviceBuffer<double>::make(1);
            if (!t) e = t.error();
            else st->tally = std::move(*t);
        }
        if (e == hipSuccess) {
            auto t = DeviceBuffer<double>::make(kLmMaxY);
            if (!t) e = t.error();
            else st->ytally = std::move(*t);
        }
    }
    for (DeviceScalar* b : {&d.T_, &d.om_, &d.mu_, &d.divu_, &d.Tstar_, &d.D_,
                            &d.rhs_, &d.phi_})
        if (e == hipSuccess) e = alloc(*b, n);
    // S8: the per-stage pressure increments (the PCG warm starts), from 0.
    if (d.pr_.immersed())
        for (DeviceScalar& b : d.phiW_) {
            if (e == hipSuccess) e = alloc(b, n);
            if (e == hipSuccess) e = hipMemset(b.data(), 0, n * sizeof(Real));
        }
    for (int k = 0; k < kLmMaxY && e == hipSuccess; ++k) e = alloc(d.Y_[k], n);
    // M4: mu_t, the cell velocity it (and the efficiency) differences, TFLES.
    {
        const bool vel = s.cs > 0 || (s.tfles && s.tfles->charlette);
        if (e == hipSuccess) e = alloc(d.mut_, s.cs > 0 ? n : 1);
        for (DeviceScalar* b : {&d.uc_, &d.vc_, &d.wc_})
            if (e == hipSuccess) e = alloc(*b, vel ? n : 1);
        if (e == hipSuccess) e = hipMemset(d.mut_.data(), 0, d.mut_.size() * sizeof(Real));
        const double c3 = std::cbrt(double(g.hx) * g.hy * g.hz);
        const double d2 = g.nz > 1 ? c3 * c3 : double(g.hx) * g.hy;
        d.cd2_ = Real(s.cs * s.cs * d2);
        if (s.tfles && e == hipSuccess) {
            auto t = DeviceTfles::make(g, *s.tfles);
            if (!t) e = t.error();
            else d.tf_ = std::move(*t);
        }
    }
    if (e == hipSuccess)
        e = alloc(d.bflux_, std::size_t(s.nY) * bfLayout(d.g_, nullptr).nb);
    if (e == hipSuccess) {
        auto t = DeviceBuffer<double>::make(kBSums);
        if (!t) e = t.error();
        else d.bsums_ = std::move(*t);
    }
    // The inflow rows: the profile, or every row the scalars.
    const std::size_t nr = std::size_t(g.ny) * g.nz;
    for (const auto* v : {&s.rowM, &s.rowT, &s.rowY[0], &s.rowY[1]})
        if (!v->empty() && v->size() != nr)
            return std::unexpected(std::format("low-Mach: an inflow row profile "
                                               "has {} entries, the face {}",
                                               v->size(), nr));
    if (s.rowM.empty() != s.rowT.empty() || s.rowM.empty() != s.rowY[0].empty() ||
        (s.nY == 2 && s.rowM.empty() != s.rowY[1].empty()))
        return std::unexpected("low-Mach: the inflow row profile needs m, T and "
                               "every Y_k");
    for (DeviceScalar* b : {&d.inM_, &d.inRho_, &d.inT_, &d.inY_[0], &d.inY_[1]})
        if (e == hipSuccess) e = alloc(*b, nr);
    if (s.turb)
        for (DeviceScalar* b : {&d.inV_, &d.inW_}) {
            if (e == hipSuccess) e = alloc(*b, nr);
            if (e == hipSuccess) e = hipMemset(b->data(), 0, nr * sizeof(Real));
        }
    if (e == hipSuccess) {
        const bool rows = !s.rowM.empty();
        std::vector<Real> f(nr);
        auto put = [&](DeviceScalar& b, auto&& at) {
            for (std::size_t q = 0; q < nr; ++q) f[q] = Real(at(q));
            if (e == hipSuccess) e = b.upload(f.data());
        };
        put(d.inM_, [&](std::size_t q) { return rows ? s.rowM[q] : s.mIn; });
        put(d.inT_, [&](std::size_t q) { return rows ? s.rowT[q] : s.TIn; });
        put(d.inRho_, [&](std::size_t q) {
            return s.prGas / (rows ? s.rowT[q] : s.TIn);
        });
        for (int k = 0; k < kLmMaxY; ++k)
            put(d.inY_[k], [&](std::size_t q) {
                return rows && k < s.nY ? s.rowY[k][q] : s.yIn[k];
            });
    }
    if (e != hipSuccess)
        return std::unexpected(std::string("low-Mach: GPU allocation failed: ") +
                               hipGetErrorString(e));
    // Before the first step: divErr / eosErr read 0; the boundary fluxes 0.
    if (hipMemset(d.D_.data(), 0, n * sizeof(Real)) != hipSuccess ||
        hipMemset(d.Tstar_.data(), 0, n * sizeof(Real)) != hipSuccess ||
        hipMemset(d.bflux_.data(), 0, d.bflux_.size() * sizeof(Real)) != hipSuccess)
        return std::unexpected("low-Mach: GPU memset failed");
    d.area_ = g.hy * g.hz;
    return d;
}

std::expected<void, std::string> DeviceLowMach::upload(const LowMachState& h)
{
    const std::size_t n = g_.size();
    const std::size_t nfx = std::size_t(g_.nx + 1) * g_.ny * g_.nz;
    const std::size_t nyh = sp_.openY ? std::size_t(g_.nx) * g_.nz : 0;
    const std::size_t nzh = sp_.openZ ? std::size_t(g_.nx) * g_.ny : 0;
    if (h.rho.size() != n || h.p.size() != n || h.my.size() != n ||
        h.mz.size() != n || h.mx.size() != nfx || h.nY != sp_.nY ||
        !(h.myHi.empty() || h.myHi.size() == nyh) ||
        !(h.mzHi.empty() || h.mzHi.size() == nzh))
        return std::unexpected("low-Mach: host state does not match the grid");
    std::vector<Real> f;
    auto put = [&](DeviceScalar& b, const std::vector<double>& v) {
        f.assign(v.begin(), v.end());
        return b.upload(f.data()) == hipSuccess;
    };
    // An open hi-face array: the host's, or 0.
    auto putHi = [&](DeviceScalar& b, const std::vector<double>& v, std::size_t m) {
        if (m == 0) return true;
        f.assign(m, Real(0));
        for (std::size_t q = 0; q < v.size(); ++q) f[q] = Real(v[q]);
        return b.upload(f.data()) == hipSuccess;
    };
    // The compensated words: hi = the FP32 rounding, lo = the FP64 IC's
    // remainder.
    auto putLo = [&](DeviceScalar& b, const std::vector<double>& v) {
        f.resize(v.size());
        for (std::size_t c = 0; c < v.size(); ++c) f[c] = Real(v[c] - double(Real(v[c])));
        return b.upload(f.data()) == hipSuccess;
    };
    bool ok = put(s_.rho, h.rho) && putLo(s_.rlo, h.rho) && put(s_.p, h.p) &&
              put(s_.mx, h.mx) && put(s_.my, h.my) && put(s_.mz, h.mz) &&
              putHi(s_.myHi, h.myHi, nyh) && putHi(s_.mzHi, h.mzHi, nzh);
    for (int k = 0; ok && k < sp_.nY; ++k)
        ok = put(s_.y[k], h.rhoY[std::size_t(k)]) &&
             putLo(s_.ylo[k], h.rhoY[std::size_t(k)]);
    if (!ok) return std::unexpected("low-Mach: initial condition upload failed");
    // The tallies start at the uploaded inventories (hi + lo words), summed
    // in FP64 in index order on the host.
    auto inventory = [&](const std::vector<double>& v) {
        double m = 0;
        for (std::size_t c = 0; c < n; ++c) {
            const Real hi = Real(v[c]);
            m += double(hi) + double(Real(v[c] - double(hi)));
        }
        return m * (g_.hx * double(g_.hy) * g_.hz);
    };
    const double m = inventory(h.rho);
    double yt[kLmMaxY] = {0, 0};
    for (int k = 0; k < sp_.nY; ++k) yt[k] = inventory(h.rhoY[std::size_t(k)]);
    if (s_.tally.upload(&m) != hipSuccess || s_.ytally.upload(yt) != hipSuccess)
        return std::unexpected("low-Mach: tally upload failed");
    return {};
}

std::expected<void, std::string> DeviceLowMach::setInflowTurb(const std::vector<double>& m,
                                                              const std::vector<double>& v,
                                                              const std::vector<double>& w)
{
    const std::size_t nr = std::size_t(g_.ny) * g_.nz;
    if (!sp_.turb) return std::unexpected("low-Mach: setInflowTurb needs setup.turb");
    if (m.size() != nr || v.size() != nr || w.size() != nr)
        return std::unexpected(std::format("low-Mach: --turb rows have {}, {}, {} entries, the face {}", m.size(),
                                           v.size(), w.size(), nr));
    std::vector<Real> f(nr);
    (void)hipDeviceSynchronize();
    for (auto [b, h] : {std::pair{&inM_, &m}, std::pair{&inV_, &v}, std::pair{&inW_, &w}}) {
        for (std::size_t q = 0; q < nr; ++q) f[q] = Real((*h)[q]);
        if (b->upload(f.data()) != hipSuccess) return std::unexpected("low-Mach: --turb row upload failed");
    }
    return {};
}

std::array<const Real*, 5> DeviceLowMach::inflowRows() const
{
    return {inM_.data(), inRho_.data(), inT_.data(), inY_[0].data(),
            sp_.nY == 2 ? inY_[1].data() : nullptr};
}

void DeviceLowMach::cellVelocity(DeviceScalar& u, DeviceScalar& v,
                                 DeviceScalar& w) const
{
    const LmK K = kernelConsts(g_, sp_, inflowRows(), pr_.faceCodes());
    const LmM M = massFluxes(s_);
    withLat(sp_, [&](auto oy, auto oz) {
        kLmCellVelocity<decltype(oy)::value, decltype(oz)::value>
            <<<launchBlocks(g_.size()), blockSize>>>(K, s_.rho.data(), M, u.data(),
                                                     v.data(), w.data());
    });
    checkLastKernel("kLmCellVelocity");
}

namespace {

// The cell pass on state (rho, ry0, ry1, M) into the derived fields.
// eof (M4 TFLES, C0): omega *= E/F (the Eof instantiation); null = the S8 pass.
template <bool OY, bool OZ>
void cellPass(const LmK& K, const LowMachSetup& sp, const Real* rho,
              const Real* ry0, const Real* ry1, const LmM& M, Real* T, Real* Y0,
              Real* Y1, Real* om, Real* mu, Real* divu, const Real* eof = nullptr)
{
    const std::size_t n = K.g.size();
    const bool two = sp.nY == 2;
    const PhiRateLaw<Real> law = sp.law.as<Real>();
    assert(!eof || !two);
    if (two && sp.law.n > 0)
        kLmCellPass<true, true, OY, OZ><<<launchBlocks(n), blockSize>>>(
            K, law, rho, ry0, ry1, M, T, Y0, Y1, om, mu, divu, nullptr);
    else if (two)
        kLmCellPass<true, false, OY, OZ><<<launchBlocks(n), blockSize>>>(
            K, law, rho, ry0, ry1, M, T, Y0, Y1, om, mu, divu, nullptr);
    else if (eof)
        kLmCellPass<false, false, OY, OZ, true><<<launchBlocks(n), blockSize>>>(
            K, law, rho, ry0, ry1, M, T, Y0, Y1, om, mu, divu, eof);
    else
        kLmCellPass<false, false, OY, OZ><<<launchBlocks(n), blockSize>>>(
            K, law, rho, ry0, ry1, M, T, Y0, Y1, om, mu, divu, nullptr);
}

// The diffusivity instantiation of a call (M4): f(std::integral_constant<int,
// kLmMolecular | kLmEddy | kLmThick>).
template <class F>
void withDiff(int d, F&& f)
{
    if (d == kLmThick) f(std::integral_constant<int, kLmThick>{});
    else if (d == kLmEddy) f(std::integral_constant<int, kLmEddy>{});
    else f(std::integral_constant<int, kLmMolecular>{});
}

} // namespace

void DeviceLowMach::stage(const State& a, const State& b, State& o, double ca,
                          double cb, double cr, int si, std::string& err)
{
    if (!err.empty()) return;
    const GpuGrid& g = g_;
    LmK K = kernelConsts(g, sp_, inflowRows(), pr_.faceCodes());
    if (sp_.turb) {
        K.inV = inV_.data();
        K.inW = inW_.data();
    }
    const std::size_t n = g.size();
    const std::size_t nfx = std::size_t(g.nx + 1) * g.ny * g.nz;
    const Real rca = Real(ca), rcb = Real(cb), rcr = Real(cr);
    const bool two = sp_.nY == 2;
    const Real* y1b = two ? b.y[1].data() : nullptr;
    Real* Y1 = Y_[1].data();
    const LmM Mb = massFluxes(b);
    const LmBF bf = bfLayout(g, bflux_.data());
    const bool imm = pr_.immersed();
    // The pressure increment: with the tube, this stage's own buffer -- the
    // PCG's warm start is the same stage's phi of the previous step
    // (checkpointed); without, the scratch the direct solve overwrites.
    Real* phi = imm ? phiW_[si].data() : phi_.data();
    // M4: the interior faces' diffusivity and its fields.
    const int diff = tf_ ? kLmThick : les() ? kLmEddy : kLmMolecular;
    const LmEddy E{.mut = les() ? mut_.data() : nullptr,
                   .invSct = Real(1.0 / sp_.prt),
                   .FE = tf_ ? tf_->fe() : nullptr,
                   .Om = tf_ ? tf_->om() : nullptr};
    withLat(sp_, [&](auto oyT, auto ozT) {
      withWall(imm, [&](auto wT) {
        constexpr bool OY = decltype(oyT)::value, OZ = decltype(ozT)::value;
        constexpr bool WL = decltype(wT)::value;
        // 1. Cell pass on b (TFLES: omega E/F); LES: mu_t, mu += mu_t.
        cellPass<OY, OZ>(K, sp_, b.rho.data(), b.y[0].data(), y1b, Mb, T_.data(),
                         Y_[0].data(), Y1, om_.data(), mu_.data(), divu_.data(),
                         tf_ ? tf_->eof() : nullptr);
        checkLastKernel("kLmCellPass");
        if (les()) eddyOf(b);
        // 2. Temperature predictor -> T*, D.
        withDiff(diff, [&](auto dT) {
            constexpr int DF = decltype(dT)::value;
            if constexpr (WL && DF != kLmMolecular) {
                assert(false); // make() refuses LES / TFLES with the tube
            } else {
                if (two) {
                    if constexpr (DF != kLmThick)
                        kLmTemperature<true, OY, OZ, WL, DF><<<launchBlocks(n), blockSize>>>(
                            K, rca, rcb, rcr, a.rho.data(), b.rho.data(), T_.data(),
                            Y_[0].data(), Y1, om_.data(), Mb, Tstar_.data(), D_.data(), E);
                    else
                        assert(false); // make() refuses TFLES with two reactants
                } else {
                    kLmTemperature<false, OY, OZ, WL, DF><<<launchBlocks(n), blockSize>>>(
                        K, rca, rcb, rcr, a.rho.data(), b.rho.data(), T_.data(), Y_[0].data(),
                        Y1, om_.data(), Mb, Tstar_.data(), D_.data(), E);
                }
            }
        });
        checkLastKernel("kLmTemperature");
        // 3. Momentum predictor -> o.m* (gravity: the Grav instantiation;
        // g == 0 the S6 kernel).
        if (sp_.gravity > 0)
            kLmMomX<OY, OZ, true, WL><<<launchBlocks(nfx), blockSize>>>(
                K, rca, rcb, rcr, a.mx.data(), b.rho.data(), Mb, b.p.data(),
                mu_.data(), divu_.data(), o.mx.data());
        else
            kLmMomX<OY, OZ, false, WL><<<launchBlocks(nfx), blockSize>>>(
                K, rca, rcb, rcr, a.mx.data(), b.rho.data(), Mb, b.p.data(),
                mu_.data(), divu_.data(), o.mx.data());
        const std::size_t nyh = OY ? std::size_t(g.nx) * g.nz : 0;
        const std::size_t nzh = OZ ? std::size_t(g.nx) * g.ny : 0;
        // M5 --turb: the Turb instantiation (make refuses it with a solid or open laterals).
        auto momT = [&](auto tb) {
            constexpr bool TB = decltype(tb)::value;
            kLmMomT<1, OY, OZ, WL, TB><<<launchBlocks(n + nyh), blockSize>>>(
                K, rca, rcb, rcr, a.my.data(), OY ? a.myHi.data() : nullptr, b.rho.data(),
                Mb, b.p.data(), mu_.data(), divu_.data(), o.my.data(),
                OY ? o.myHi.data() : nullptr);
            kLmMomT<2, OY, OZ, WL, TB><<<launchBlocks(n + nzh), blockSize>>>(
                K, rca, rcb, rcr, a.mz.data(), OZ ? a.mzHi.data() : nullptr, b.rho.data(),
                Mb, b.p.data(), mu_.data(), divu_.data(), o.mz.data(),
                OZ ? o.mzHi.data() : nullptr);
        };
        if constexpr (!WL && !OY && !OZ) {
            if (sp_.turb) momT(std::true_type{});
            else momT(std::false_type{});
        } else {
            momT(std::false_type{});
        }
        checkLastKernel("kLmMom");
        // 4. Projection on the mass flux.
        const LmM Mo = massFluxes(o);
        kLmDivRhs<OY, OZ><<<launchBlocks(n), blockSize>>>(g, rcr, Mo, D_.data(),
                                                          rhs_.data());
        checkLastKernel("kLmDivRhs");
        auto res = pr_.solve(rhs_.data(), phi, imm);
        if (!res) {
            err = res.error();
            return;
        }
        pcgIt_[si] = res->iterations;
        if (!res->converged) {
            pcgFailed_ = true;
            err = std::format("the apertured PCG did not converge in {} iterations "
                              "(|r|/|b| = {:.3g}, tol {:.3g}; FP32 floor estimate "
                              "{:.3g})",
                              res->iterations, res->relResidual, sp_.pcgTol,
                              res->floorRel);
            return;
        }
        kLmCorrectX<WL><<<launchBlocks(nfx), blockSize>>>(g, rcr, phi, o.mx.data(),
                                                          K.code);
        kLmCorrectC<OY, OZ, WL><<<launchBlocks(n), blockSize>>>(
            g, rcr, phi, b.p.data(), o.my.data(), o.mz.data(),
            OY ? o.myHi.data() : nullptr, OZ ? o.mzHi.data() : nullptr, o.p.data(),
            K.code);
        checkLastKernel("kLmCorrect");
        // 5. Continuity + species on the projected flux.
        LmWords W{};
        {
            const State* src[2] = {&a, &b};
            for (int v = 0; v < 1 + sp_.nY; ++v) {
                for (int q = 0; q < 2; ++q) {
                    const State& S = *src[q];
                    const Real* hi = v == 0 ? S.rho.data() : S.y[v - 1].data();
                    const Real* lo = v == 0 ? S.rlo.data() : S.ylo[v - 1].data();
                    (q == 0 ? W.ah : W.bh)[v] = hi;
                    (q == 0 ? W.al : W.bl)[v] = lo;
                }
                W.oh[v] = v == 0 ? o.rho.data() : o.y[v - 1].data();
                W.ol[v] = v == 0 ? o.rlo.data() : o.ylo[v - 1].data();
            }
        }
        withDiff(diff, [&](auto dT) {
            constexpr int DF = decltype(dT)::value;
            if constexpr (WL && DF != kLmMolecular) {
                assert(false);
            } else {
                if (two) {
                    if constexpr (DF != kLmThick)
                        kLmScalars<true, OY, OZ, WL, DF><<<launchBlocks(n), blockSize>>>(
                            K, rcb, rcr, W, T_.data(), Y_[0].data(), Y1, om_.data(), Mo, bf, E);
                    else
                        assert(false);
                } else {
                    kLmScalars<false, OY, OZ, WL, DF><<<launchBlocks(n), blockSize>>>(
                        K, rcb, rcr, W, T_.data(), Y_[0].data(), Y1, om_.data(), Mo, bf, E);
                }
            }
        });
        checkLastKernel("kLmScalars");
        kLmTally<OY, OZ><<<1, kTallyThreads>>>(g, double(rca), double(rcb),
                                               double(rcr), a.tally.data(),
                                               b.tally.data(), Mo, area_,
                                               o.tally.data());
        kLmTallyY<OY, OZ><<<1, kTallyThreads>>>(g, sp_.nY, double(rcb), double(rcr),
                                                a.ytally.data(), b.ytally.data(),
                                                bf, o.ytally.data());
        checkLastKernel("kLmTally");
      });
    });
}

void DeviceLowMach::eddyOf(const State& b)
{
    const LmK K = kernelConsts(g_, sp_, inflowRows(), pr_.faceCodes());
    const LmM M = massFluxes(b);
    const int sig = sp_.sgs == SgsModel::Sigma ? 1 : 0;
    withLat(sp_, [&](auto oy, auto oz) {
        constexpr bool OY = decltype(oy)::value, OZ = decltype(oz)::value;
        kLmCellVelocity<OY, OZ><<<launchBlocks(g_.size()), blockSize>>>(
            K, b.rho.data(), M, uc_.data(), vc_.data(), wc_.data());
        kLmNut<OY, OZ><<<launchBlocks(g_.size()), blockSize>>>(
            K, cd2_, sig, b.rho.data(), uc_.data(), vc_.data(), wc_.data(), mut_.data(),
            mu_.data());
    });
    checkLastKernel("kLmNut");
}

void DeviceLowMach::tflesOf(const State& b)
{
    // T (and the cell velocity for the efficiency) of b, then F E, E/F, Omega.
    const LmK K = kernelConsts(g_, sp_, inflowRows(), pr_.faceCodes());
    const LmM M = massFluxes(b);
    const Real* y1 = sp_.nY == 2 ? b.y[1].data() : nullptr;
    withLat(sp_, [&](auto oy, auto oz) {
        constexpr bool OY = decltype(oy)::value, OZ = decltype(oz)::value;
        cellPass<OY, OZ>(K, sp_, b.rho.data(), b.y[0].data(), y1, M, T_.data(),
                         Y_[0].data(), Y_[1].data(), om_.data(), mu_.data(),
                         divu_.data());
        if (sp_.tfles->charlette)
            kLmCellVelocity<OY, OZ><<<launchBlocks(g_.size()), blockSize>>>(
                K, b.rho.data(), M, uc_.data(), vc_.data(), wc_.data());
    });
    checkLastKernel("kLmCellPass");
    tf_->update(b.rho.data(), b.y[0].data(), uc_, vc_, wc_, T_);
}

void DeviceLowMach::refreshEddy()
{
    if (tf_) tflesOf(s_);
    if (!les()) return;
    // The cell pass of s_ (mu(T) before mu += mu_t), then mu_t.
    const LmK K = kernelConsts(g_, sp_, inflowRows(), pr_.faceCodes());
    const LmM M = massFluxes(s_);
    const Real* y1 = sp_.nY == 2 ? s_.y[1].data() : nullptr;
    withLat(sp_, [&](auto oy, auto oz) {
        cellPass<decltype(oy)::value, decltype(oz)::value>(
            K, sp_, s_.rho.data(), s_.y[0].data(), y1, M, T_.data(), Y_[0].data(),
            Y_[1].data(), om_.data(), mu_.data(), divu_.data(),
            tf_ ? tf_->eof() : nullptr);
    });
    checkLastKernel("kLmCellPass");
    eddyOf(s_);
}

LmEddyStats DeviceLowMach::eddyStats(ReduceScratch& r)
{
    refreshEddy();
    LmEddyStats e;
    if (les())
        e.mutMax = internal::reduce(r, g_.size(), LmAbsField{mut_.data()},
                                    rocprim::maximum<double>(), 0.0);
    if (tf_) {
        e.feMax = tf_->maxFE(r);
        DeviceBuffer<double> scratch;
        std::vector<double> h;
        if (tf_->columnStats(scratch, h) == hipSuccess) {
            double so = 0, soe = 0;
            for (int i = 0; i < g_.nx; ++i) {
                so += h[std::size_t(i) * DeviceTfles::kTflesSums];
                soe += h[std::size_t(i) * DeviceTfles::kTflesSums + 1];
            }
            e.eBar = so > 0 ? soe / so : NAN;
        }
    }
    return e;
}

std::expected<void, std::string> DeviceLowMach::step(double dt)
{
    // SSP-RK3 with StageCombine's weights (gpu/comp.hip DeviceComp::step):
    // ca = 1 - cb exactly (Sterbenz), cr = cb dt.
    const Real d = Real(dt);
    std::string err;
    // M4 TFLES: F, E, Omega frozen over the step, from its input state.
    if (tf_) tflesOf(s_);
    stage(s_, s_, q1_, 1.0, 0.0, double(d), 0, err);
    stage(s_, q1_, q2_, 0.75, 0.25, double(Real(0.25) * d), 1, err);
    const Real cb3 = Real(2.0 / 3.0), ca3 = Real(1) - cb3;
    stage(s_, q2_, q1_, double(ca3), double(cb3), double(cb3 * d), 2, err);
    if (!err.empty()) return std::unexpected("low-Mach pressure solve: " + err);
    std::swap(s_, q1_);
    return {};
}

int DeviceLowMach::pcgIterations() const
{
    return std::max({pcgIt_[0], pcgIt_[1], pcgIt_[2]});
}

double DeviceLowMach::convectiveDt(ReduceScratch& r, double cfl) const
{
    const LmM M = massFluxes(s_);
    const double amb = sp_.prGas / sp_.ambT;
    double w = 0;
    withLat(sp_, [&](auto oy, auto oz) {
        const LmConvRate<decltype(oy)::value, decltype(oz)::value> f{
            g_, M, s_.rho.data(), inRho_.data(), amb};
        w = internal::reduce(r, g_.size(), f, rocprim::maximum<double>(), 0.0);
    });
    return w > 0 ? cfl / w : HUGE_VAL;
}

double DeviceLowMach::diffusiveDt(ReduceScratch& r)
{
    const double rhoMin = internal::reduce(r, g_.size(), LmMinField{s_.rho.data()},
                                           rocprim::minimum<double>(), 1e300);
    const double Tmax = sp_.prGas / rhoMin;
    const double mu = sp_.tr.mu(Tmax);
    if (!(mu > 0)) return HUGE_VAL;
    double h = std::min<double>(g_.hx, g_.hy);
    if (g_.nz > 1) h = std::min<double>(h, g_.hz);
    // 0.5/ndim: the explicit Laplacian's positivity bound sum_a 2 D dt/h^2
    // <= 1 (2D: 0.25 as S4; 3D: 1/6, S5). At 0.25 in 3D (sum 1.5) the
    // 8-cell round partially premixed burner undershot Y_O and overshot T to
    // 8.6 (sigma_st 6.98) at the inner tip (measured).
    const double coef = g_.nz > 1 ? 1.0 / 6.0 : 0.25;
    if (!les() && !tf_)
        return coef * h * h * rhoMin / (mu * std::max(4.0 / 3.0, 1.0 / sp_.tr.Pr));
    // M4: the eddy fields of the current state, maxima over the box (the
    // compressible tflesStableDt's form).
    refreshEddy();
    const double fe = tf_ ? tf_->maxFE(r) : 1.0;
    const double mt = les() ? internal::reduce(r, g_.size(), LmAbsField{mut_.data()},
                                               rocprim::maximum<double>(), 0.0)
                            : 0.0;
    const double dd = mu * std::max(4.0 / 3.0, fe / sp_.tr.Pr) +
                      mt * std::max(4.0 / 3.0, 1.0 / sp_.prt);
    return coef * h * h * rhoMin / dd;
}

double DeviceLowMach::chemDt(ReduceScratch& r, double K) const
{
    if (!std::isfinite(sp_.lnA)) return HUGE_VAL; // no reaction
    const Real* r1 = sp_.nY == 2 ? s_.y[1].data() : nullptr;
    double J;
    if (sp_.nY == 2 && sp_.law.n > 0)
        J = internal::reduce(r, g_.size(),
                             LmChemJ<true, true>{s_.rho.data(), s_.y[0].data(), r1,
                                                 sp_.prGas, sp_.lnA, sp_.Ta, sp_.qcp,
                                                 sp_.stoich, sp_.law},
                             rocprim::maximum<double>(), 0.0);
    else if (sp_.nY == 2)
        J = internal::reduce(r, g_.size(),
                             LmChemJ<true, false>{s_.rho.data(), s_.y[0].data(), r1,
                                                  sp_.prGas, sp_.lnA, sp_.Ta, sp_.qcp,
                                                  sp_.stoich, sp_.law},
                             rocprim::maximum<double>(), 0.0);
    else
        J = internal::reduce(r, g_.size(),
                             LmChemJ<false, false>{s_.rho.data(), s_.y[0].data(), r1,
                                                   sp_.prGas, sp_.lnA, sp_.Ta, sp_.qcp,
                                                   sp_.stoich, sp_.law},
                             rocprim::maximum<double>(), 0.0);
    return J > 0 ? K / J : HUGE_VAL;
}

double DeviceLowMach::positivityDt(ReduceScratch& r, double K)
{
    const LmM M = massFluxes(s_);
    const double amb = sp_.prGas / sp_.ambT;
    double inv2 = 1.0 / (double(g_.hx) * g_.hx) + 1.0 / (double(g_.hy) * g_.hy);
    if (g_.nz > 1) inv2 += 1.0 / (double(g_.hz) * g_.hz);
    const double dcoef = 2.0 * inv2 * std::max(4.0 / 3.0, 1.0 / sp_.tr.Pr);
    const bool eddy = les() || tf_.has_value();
    if (eddy) refreshEddy();
    double w = 0;
    withLat(sp_, [&](auto oy, auto oz) {
        constexpr bool OY = decltype(oy)::value, OZ = decltype(oz)::value;
        const LmConvRate<OY, OZ> cr{g_, M, s_.rho.data(), inRho_.data(), amb};
        if (eddy) {
            const LmPosRateEddy<OY, OZ> f{cr,
                                          sp_.tr,
                                          sp_.prGas,
                                          2.0 * inv2,
                                          1.0 / sp_.tr.Pr,
                                          std::max(4.0 / 3.0, 1.0 / sp_.prt),
                                          les() ? mut_.data() : nullptr,
                                          tf_ ? tf_->fe() : nullptr};
            w = internal::reduce(r, g_.size(), f, rocprim::maximum<double>(), 0.0);
        } else {
            const LmPosRate<OY, OZ> f{cr, sp_.tr, sp_.prGas, dcoef};
            w = internal::reduce(r, g_.size(), f, rocprim::maximum<double>(), 0.0);
        }
    });
    return w > 0 ? K / w : HUGE_VAL;
}

void DeviceLowMach::updateCells()
{
    // M4 TFLES: omega is (E/F) omega of the current state's F, E.
    if (tf_) tflesOf(s_);
    const LmK K = kernelConsts(g_, sp_, inflowRows(), pr_.faceCodes());
    const LmM M = massFluxes(s_);
    const Real* y1 = sp_.nY == 2 ? s_.y[1].data() : nullptr;
    withLat(sp_, [&](auto oy, auto oz) {
        cellPass<decltype(oy)::value, decltype(oz)::value>(
            K, sp_, s_.rho.data(), s_.y[0].data(), y1, M, T_.data(), Y_[0].data(),
            Y_[1].data(), om_.data(), mu_.data(), divu_.data(),
            tf_ ? tf_->eof() : nullptr);
    });
    checkLastKernel("kLmCellPass");
}

hipError_t DeviceLowMach::columnStats(DeviceBuffer<double>& scratch,
                                      std::vector<double>& host) const
{
    const std::size_t n = std::size_t(g_.nx) * kLmSums;
    if (auto e = internal::ensurePlaneScratch(scratch, n); e != hipSuccess) return e;
    LmColK C;
    C.g = g_;
    C.inRho = inRho_.data();
    C.qcp = sp_.qcp;
    C.Hin = sp_.TIn + sp_.qcp * sp_.yIn[sp_.fuel];
    C.zS = sp_.zS;
    C.zO2 = sp_.zO2;
    C.zDen = sp_.nY == 2 ? sp_.zDen : 0.0;
    C.fuel = sp_.fuel;
    C.ox = sp_.ox;
    kLmColumns<<<launchBlocks(std::size_t(g_.nx)), blockSize>>>(
        C, s_.rho.data(), s_.rlo.data(), s_.mx.data(), T_.data(),
        s_.y[sp_.fuel].data(), s_.ylo[sp_.fuel].data(), om_.data(), s_.p.data(), Y_[0].data(), Y_[1].data(), scratch.data());
    checkLastKernel("kLmColumns");
    return internal::fetchPlaneScratch(scratch, n, host);
}

double DeviceLowMach::divErr(ReduceScratch& r) const
{
    const LmM M = massFluxes(s_);
    double e = 0;
    withLat(sp_, [&](auto oy, auto oz) {
        e = internal::reduce(r, g_.size(),
                             LmDivErr<decltype(oy)::value, decltype(oz)::value>{
                                 g_, M, D_.data()},
                             rocprim::maximum<double>(), 0.0);
    });
    return e;
}

double DeviceLowMach::divTargetMax(ReduceScratch& r) const
{
    return internal::reduce(r, g_.size(), LmAbsField{D_.data()},
                            rocprim::maximum<double>(), 0.0);
}

double DeviceLowMach::eosErr(ReduceScratch& r) const
{
    // Before the first step T* is 0: report 0, not 1.
    const double e = internal::reduce(r, g_.size(),
                                      LmEosErr{s_.rho.data(), Tstar_.data(), sp_.prGas},
                                      rocprim::maximum<double>(), 0.0);
    return e;
}

double DeviceLowMach::massTally() const
{
    double m = NAN;
    if (s_.tally.download(&m) != hipSuccess) return NAN;
    return m;
}

double DeviceLowMach::speciesTally(int k) const
{
    assert(k >= 0 && k < sp_.nY);
    double t[kLmMaxY] = {NAN, NAN};
    if (s_.ytally.download(t) != hipSuccess) return NAN;
    return t[k];
}

LmBoundarySums DeviceLowMach::boundarySums() const
{
    const LmK K = kernelConsts(g_, sp_, inflowRows(), pr_.faceCodes());
    const LmM M = massFluxes(s_);
    const Real* r1 = sp_.nY == 2 ? s_.y[1].data() : nullptr;
    withLat(sp_, [&](auto oy, auto oz) {
        kLmBSums<decltype(oy)::value, decltype(oz)::value><<<1, kTallyThreads>>>(
            K, M, s_.rho.data(), s_.y[0].data(), r1, bsums_.data());
    });
    checkLastKernel("kLmBSums");
    double h[kBSums];
    LmBoundarySums b;
    if (bsums_.download(h) != hipSuccess) {
        b.mIn = b.mOut = b.mLat = NAN;
        return b;
    }
    b.mIn = h[0];
    b.mOut = h[1];
    const bool open = sp_.openY || sp_.openZ;
    b.mLat = h[2];
    b.uLatMin = open ? h[3] : 0.0;
    b.uLatMax = open ? h[4] : 0.0;
    for (int k = 0; k < sp_.nY; ++k) b.yOut[k] = h[5 + k];
    return b;
}

std::expected<LowMachState, std::string> DeviceLowMach::download(bool hiOnly) const
{
    LowMachState h;
    h.nY = sp_.nY;
    std::vector<Real> f, fl;
    auto get = [&](const DeviceScalar& b, std::vector<double>& v) {
        f.resize(b.size());
        if (b.size() && b.download(f.data()) != hipSuccess) return false;
        v.assign(f.begin(), f.end());
        return true;
    };
    // hi + lo in FP64 (unless hiOnly).
    auto getW = [&](const DeviceScalar& hi, const DeviceScalar& lo,
                    std::vector<double>& v) {
        if (!get(hi, v)) return false;
        if (hiOnly) return true;
        fl.resize(lo.size());
        if (lo.download(fl.data()) != hipSuccess) return false;
        for (std::size_t c = 0; c < v.size(); ++c) v[c] += double(fl[c]);
        return true;
    };
    bool ok = getW(s_.rho, s_.rlo, h.rho) && get(s_.p, h.p) && get(s_.mx, h.mx) &&
              get(s_.my, h.my) && get(s_.mz, h.mz) && get(s_.myHi, h.myHi) &&
              get(s_.mzHi, h.mzHi);
    for (int k = 0; ok && k < sp_.nY; ++k)
        ok = getW(s_.y[k], s_.ylo[k], h.rhoY[std::size_t(k)]);
    if (!ok) return std::unexpected("low-Mach: state download failed");
    return h;
}

std::expected<void, RunError> DeviceLowMach::save(Checkpoint& ck) const
{
    std::vector<float> h;
    auto put = [&](const std::string& name, const DeviceScalar& b) -> bool {
        h.resize(b.size());
        if (b.download(h.data()) != hipSuccess) return false;
        ck.putF32(name, h);
        return true;
    };
    bool ok = put("state.rho", s_.rho) && put("state.mx", s_.mx) &&
              put("state.my", s_.my) && put("state.mz", s_.mz) &&
              put("state.p", s_.p) && put("state.rlo", s_.rlo);
    if (ok && sp_.openY) ok = put("state.myhi", s_.myHi);
    if (ok && sp_.openZ) ok = put("state.mzhi", s_.mzHi);
    for (int k = 0; ok && k < sp_.nY; ++k)
        ok = put(std::format("state.y{}", k), s_.y[k]) &&
             put(std::format("state.ylo{}", k), s_.ylo[k]);
    // S8: the PCG warm starts (the next step's initial guesses: K12).
    if (pr_.immersed())
        for (int q = 0; ok && q < 3; ++q) ok = put(std::format("state.phi{}", q), phiW_[q]);
    double t = 0, yt[kLmMaxY] = {0, 0};
    ok = ok && s_.tally.download(&t) == hipSuccess &&
         s_.ytally.download(yt) == hipSuccess;
    if (!ok) return std::unexpected(RunError{6, "checkpoint: state download failed"});
    ck.putF64("state.tally", t);
    ck.putF64("state.ytally", std::span<const double>(yt, std::size_t(sp_.nY)));
    return {};
}

std::expected<void, RunError> DeviceLowMach::load(const Checkpoint& ck)
{
    std::vector<float> h;
    std::expected<void, RunError> res;
    auto get = [&](const std::string& name, DeviceScalar& b) -> bool {
        h.resize(b.size());
        if (auto e = ck.f32Into(name, h.data(), h.size()); !e) {
            res = std::unexpected(RunError{3, "--restart: " + e.error()});
            return false;
        }
        if (b.upload(h.data()) != hipSuccess) {
            res = std::unexpected(RunError{6, "--restart: state upload failed"});
            return false;
        }
        return true;
    };
    bool ok = get("state.rho", s_.rho) && get("state.mx", s_.mx) &&
              get("state.my", s_.my) && get("state.mz", s_.mz) &&
              get("state.p", s_.p) && get("state.rlo", s_.rlo);
    if (ok && sp_.openY) ok = get("state.myhi", s_.myHi);
    if (ok && sp_.openZ) ok = get("state.mzhi", s_.mzHi);
    for (int k = 0; ok && k < sp_.nY; ++k)
        ok = get(std::format("state.y{}", k), s_.y[k]) &&
             get(std::format("state.ylo{}", k), s_.ylo[k]);
    if (pr_.immersed())
        for (int q = 0; ok && q < 3; ++q) ok = get(std::format("state.phi{}", q), phiW_[q]);
    if (!ok) return res;
    auto t = ck.f64Scalar("state.tally");
    if (!t) return std::unexpected(RunError{3, "--restart: " + t.error()});
    auto yt = ck.f64("state.ytally");
    if (!yt) return std::unexpected(RunError{3, "--restart: " + yt.error()});
    if (yt->size() != std::size_t(sp_.nY))
        return std::unexpected(RunError{3, "--restart: state.ytally has the wrong "
                                           "species count"});
    double ytv[kLmMaxY] = {0, 0};
    for (int k = 0; k < sp_.nY; ++k) ytv[k] = (*yt)[std::size_t(k)];
    const double tv = *t;
    if (s_.tally.upload(&tv) != hipSuccess || s_.ytally.upload(ytv) != hipSuccess)
        return std::unexpected(RunError{6, "--restart: state upload failed"});
    return {};
}

} // namespace dns::gpu
