// HIP mirror of the compressible solver (dns/comp.hpp): FP32 fields, FP64
// reductions, same Kennedy-Gruber + face-viscous + SSP-RK3 scheme; nz == 1 is
// the exact 2D restriction. GPU-only header.
#pragma once

#include "dns/comp.hpp"
#include "dns/cutcell.hpp"
#include "dns/transport.hpp"
#include "dns/rans.hpp"
#include "dns/sgs.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/reductions.hpp"

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <vector>

namespace dns {
struct CurvMetrics;
struct CurvMesh;
struct AxisRemap3;
}

namespace dns::gpu {

struct CurvView; // dns/gpu/internal/comp_flux.hpp (HIP TUs only)
struct MrfView;  // dns/gpu/internal/comp_flux.hpp (HIP TUs only)
namespace rans { struct RansDiffDt; } // dns/gpu/rans.hpp (HIP TUs only)

// Rotating-frame zone of --rotor-frame mrf, in world units: the frozen-rotor
// approximation on static geometry. f = 1 on the shaft-aligned cylinder
// (radius r1, axial half-extent a1 about `center`), 0 outside (r2, a2), C1
// blend between; the kernels see it through MrfView. See comp_flux.hpp for
// the formulation and why the blended transport field stays solenoidal.
// Pre: 0 < r1 < r2, 0 < a1 < a2, omega != 0.
struct MrfZone
{
    int axis = 0;               // shaft axis 0/1/2
    double omega = 0;           // signed rate, rad per unit time
    double center[3] = {};      // shaft point
    double r1 = 0, r2 = 0;      // radial blend bounds
    double a1 = 0, a2 = 0;      // axial |s| blend bounds
};

// Conservative state on device; each buffer has g.size() cells. The RANS
// scalars ride in the SAME struct so the stage choreography (no-alias rule,
// handle swap) covers them by construction; nT == 0 leaves t[] empty.
//
// Compensated (DeviceComp::enableCompensation, --steady): every field also
// carries a LOW word, value = hi + lo with |lo| <= ulp(hi)/2. The kernels read
// hi only; the stage combination accumulates into the pair, so an update
// below ulp(hi)/2 is kept instead of rounding away (docs/rans.md, "The FP32
// update floor"). Empty lo = the plain FP32 state.
struct DeviceCompState
{
    DeviceScalar rho, mx, my, mz, E;
    DeviceScalar t[2]; // conserved turbulence scalars (rho nu~ | rho k, rho omega)
    DeviceScalar lo[5], tlo[2]; // low words of the above; empty = plain
    int nT = 0;

    // Pre: 0 <= nT <= 2. compensated allocates the low words, zeroed.
    static std::expected<DeviceCompState, hipError_t>
    make(std::size_t n, int nT = 0, bool compensated = false);

    bool compensated() const { return lo[0].size() != 0; }

    // Uploads replace the value: they zero the matching low words.
    hipError_t upload(const CompState& h);         // the five mean-flow fields
    // hi + lo in FP64 (exact: both words widen without rounding).
    hipError_t download(CompState& h) const;
    // Turbulence scalar i from/to host doubles (download resizes h).
    // Pre: i < nT; upload: h.size() == n.
    hipError_t uploadTurb(int i, const std::vector<double>& h);
    hipError_t downloadTurb(int i, std::vector<double>& h) const;
};

// An unevaluated sum hi + lo of two floats (the compensated state's words).
struct Word2
{
    Real hi, lo;
};

// Knuth's TwoSum: s.hi + s.lo == a + b EXACTLY, for any magnitudes (no
// |a| >= |b| precondition, unlike Fast2Sum -- a momentum near zero takes an
// update larger than itself). Needs IEEE rounding; contraction cannot touch
// it (no products).
__host__ __device__ inline Word2 twoSum(Real a, Real b)
{
    const Real s = a + b;
    const Real bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

// (hi + lo) + d as a renormalised pair: the hi sum is error-free, the
// leftover is rounded once at the lo level (~ulp(hi) * eps).
__host__ __device__ inline Word2 addWord2(Real hi, Real lo, Real d)
{
    const Word2 s = twoSum(hi, d);
    return twoSum(s.hi, s.lo + lo);
}

// Exact uniform Galilean x-boost by du: mx += rho du, E += mx_old du +
// rho du^2/2. A frame change -- no divergence, vorticity or acoustics;
// quiet starts = repeated small boosts tracking the freestream ramp.
void boostX(const GpuGrid& g, DeviceCompState& s, double du);

// Device tables of CurvMetrics (FP32, like the fields); hmin stays host-side
// for the viscous dt limit.
struct DeviceCurv
{
    DeviceScalar S[3][3], Ji[3][9], invVol, aov[3];
    DeviceScalar aovSteadyZ; // CurvMetrics::aovSteadyZ (empty: not a wedge)
    double hmin = 0.0;

    // m must come from CurvMetrics::build (already validated).
    static std::expected<DeviceCurv, hipError_t> make(const CurvMetrics& m);

    // The metric tables as a by-value kernel view.
    // HIP TUs only (CurvView lives in gpu/internal/comp_flux.hpp).
    CurvView view() const;
};

// The RK combination a stage kernel applies on top of the rhs it just built:
//   out = ca * a + cb * b + cr * rhs,   a/b in (rho, mx, my, mz, E) order.
// Materializing the rhs into its own five arrays and combining it in a second
// pass cost 40 B/cell/stage of pure memory traffic plus a whole conserved
// state of VRAM; the combination rides the tail of the rhs kernel instead.
//
// A boundary hook that OVERWRITES rhs rows (NSCBC) still expresses an RHS and
// never a state overwrite -- it just writes that rhs THROUGH this same
// combination on its own rows (stagePut). An additive hook (the channel body
// force) scales its contribution by crAt(c) (stageAdd). Every write goes
// through those two (gpu/internal/stage_write.hpp), which is what makes the
// compensated mode reach the hooks too.
//
// lts (--steady, local time stepping) is a per-cell RATIO dt_c / dtRef >= 1
// on top of cr, which already carries the RK coefficient and dtRef. EVERY
// consumer of cr reads it through crAt(c): an additive term scaled by the bare
// cr while the interior uses crAt converges to lts R + f = 0, the WRONG steady
// state. Null lts is the uniform step, bit-for-bit.
//
// Compensated (al/bl/ol set, DeviceCompState::lo): stagePut writes the INCREMENT
// form a + cb (b - a) + cr R into the (hi, lo) pair of the output -- equal to
// ca a + cb b because ca == 1 - cb exactly (DeviceComp::step), and it keeps
// what plain FP32 rounds away. Null = the plain expression, bit-for-bit.
struct StageCombine
{
    const Real* a[5];
    const Real* b[5];
    Real ca, cb, cr;
    const Real* ta[2] = {nullptr, nullptr}; // turbulence a/b; null when nT == 0
    const Real* tb[2] = {nullptr, nullptr};
    const Real* lts = nullptr;              // dt_c/dtRef >= 1; null = uniform
    // Low words of a, b and the stage OUTPUT; null = plain FP32.
    const Real* al[5] = {};
    const Real* bl[5] = {};
    Real* ol[5] = {};
    const Real* tal[2] = {};
    const Real* tbl[2] = {};
    Real* otl[2] = {};

    __host__ __device__ Real crAt(std::size_t c) const
    {
        return lts ? cr * lts[c] : cr;
    }
};

// A contiguous run [first, end) of flattened cells (gidx order).
struct CellRange
{
    std::size_t first = 0, end = 0;
    std::size_t size() const { return end - first; }
};

// Whether DeviceComp::step may skip its entry call of the stage hook
// (setStageHook).
enum class Idempotent : bool
{
    no,
    yes
};

// The wall/far images of mu_t (and F1 for SST, null for SA) a mesh family
// writes after mu_t is formed on every cell and before any flux reads it.
// Eddy-viscosity agnostic: an LES/DES wall image can reuse the slot.
using MutHook = std::function<void(Real* nut, Real* f1)>;

// Open domain faces of the immersed RANS path (masterplan_flowrans 4.9): at
// an open face the scalar flux is mdot = rho_c (u_c . S) of the boundary cell
// alone, carrying the specific freestream value in and the cell's out, with
// no diffusion. A COMPILE-TIME kernel branch: none open is the scalar stage
// verbatim (the O-grid and the channel never open a face).
//
// solid (device, set by enableRans from RansSetup::solidMask; null = none):
// 1 on the immersed body's ghost and deep cells. A face between a fluid and a
// solid cell lies at the wall to within the staircase, so when its mass flux
// points INTO the fluid the zero-wall-value scalars (SA nu~, SST k) are
// carried at their wall value 0, not at the antimirrored ghost value
// -a sum w phi_s < 0 -- which the counted SST k guard measured as negative k
// upwinded into the fluid (docs/rans.md, "Immersed-wall RANS"). omega keeps
// its (positive) wall image. Diffusion still reads the ghost: that IS the
// Dirichlet image.
struct ScalarBoundary
{
    int open[3] = {0, 0, 0};       // x, y, z domain faces
    double phiInf[2] = {0.0, 0.0}; // specific freestream scalars
    const std::uint8_t* solid = nullptr;
    bool any() const { return open[0] || open[1] || open[2]; }
};

// The RANS closure installed by DeviceComp::enableRans. Host-side; validated
// upstream.
struct RansSetup
{
    RansModel model = RansModel::None;   // != None
    std::vector<double> wallDist{};      // g.size(), > 0 everywhere
    MutHook mutHook{};                   // null = no image rows
    // 1 = real fluid cell: the SST guard counts only these and residual()
    // norms only these. Empty = every cell.
    std::vector<std::uint8_t> fluidMask{};
    // TEST ONLY (gpu_rans G3): transport by the mean mass flux alone -- no
    // sources, no diffusion (SA). Never set by a runner.
    bool transportOnly = false;
    // TEST ONLY (MMS): per-cell sources (g.size() each, device) added to the
    // scalar rhs INSIDE R, before the point-implicit quotient. Null = none.
    const Real* testSource[2] = {nullptr, nullptr};
    // SST: the specific-omega floor of the COUNTED positivity guard (hits
    // are reported, never silent; docs/rans.md).
    double omegaFloor = 1e-10;
    // The open domain faces (immersed path); default none.
    ScalarBoundary bnd{};
    // The immersed body's ghost + deep cells (1), uploaded into bnd.solid;
    // empty = none. Pre: empty unless bnd.any().
    std::vector<std::uint8_t> solidMask{};
};

// The channel's mutHook: mu_t odd (exactly 0.0 on the wall face) and F1
// even across the y ghost rows named. Pre: g outlives the hook.
MutHook mutMirrorYHook(const GpuGrid& g, bool lo, bool hi);

// 1 on rows [jLo, jHi] of every (i, k), 0 elsewhere: the fluid mask of a
// mesh whose only ghosts are y rows. Pre: 0 <= jLo <= jHi < g.ny.
std::vector<std::uint8_t> rowMask(const Grid& g, int jLo, int jHi);

// Per-equation RMS of the pure rhs over rows [jLo, jHi] (FP64): continuity,
// |momentum| / rho, energy / rho, and the scalars: SA nu~ as R / rho; SST k
// as R_k / rho and omega as R_w / (rho omega), the rate of ln omega (omega
// spans six decades across the wall layer).
struct ResidualNorms
{
    double rho = 0, mom = 0, E = 0, t[2] = {0, 0};
};

// Owns RK3 / primitive scratch; periodic domain (BCs via the stage hook).
class DeviceComp
{
public:
    // gamma > 1, mu >= 0, csLes >= 0. csLes > 0 turns on an SGS stress
    // (3D-valid): per RK stage a dynamic eddy viscosity mu_t = rho (C*Delta)^2
    // op(grad u) is added to the molecular mu in the face-viscous flux, with
    // eddy conductivity mu_t cp / Pr_t. op is |S| (Smagorinsky, default) or
    // D_sigma (SgsModel::Sigma; dns/sgs.hpp). csLes == 0 is molecular DNS,
    // bit-for-bit unchanged. Delta^2 = (hx hy hz)^(2/3) in 3D, hx hy when
    // nz == 1 (the |S| z-shear terms cancel by periodic wrap). mu_t is formed
    // on every cell, ghosts included; a runner with walls installs their image
    // with setMutHook (no hook = the raw ghost values).
    static std::expected<DeviceComp, hipError_t>
    make(const Grid& g, const Transport& tr, double csLes = 0.0,
         SgsModel sgs = SgsModel::Smagorinsky);

    // Constant-viscosity shorthand: Transport::constant(gamma, mu), the gas
    // the solver had before transport was temperature-dependent, bit-for-bit.
    static std::expected<DeviceComp, hipError_t>
    make(const Grid& g, double gamma, double mu, double csLes = 0.0,
         SgsModel sgs = SgsModel::Smagorinsky);

    // Curvilinear variant on a distorted regular mesh (dns/curvmesh.hpp):
    // rhs/stableDt switch to the finite-volume metric form, everything else
    // (RK3, primitives, hooks) is shared. LES works here exactly as on the
    // uniform path -- same two operators, same eddy conductivity -- with the
    // velocity gradient taken through the mesh metrics and Delta^2 the
    // PER-CELL volume^(2/3), so a clustered mesh filters at the local
    // spacing. csLes == 0 is molecular DNS, bit-for-bit unchanged.
    static std::expected<DeviceComp, hipError_t>
    make(const CurvMetrics& m, const Transport& tr, double csLes = 0.0,
         SgsModel sgs = SgsModel::Smagorinsky);
    static std::expected<DeviceComp, hipError_t>
    make(const CurvMetrics& m, double gamma, double mu, double csLes = 0.0,
         SgsModel sgs = SgsModel::Smagorinsky);

    bool curvilinear() const { return curv_.has_value(); }

    // NOTE: there is deliberately no compressible --cut-flux. Aperture-
    // weighted face fluxes DIVERGE near the body here: the KG split flux
    // loses its kinetic-energy property under per-face area weighting
    // (measured twice, 2026-07-23). config.cpp rejects the combination, and
    // the implementation was deleted rather than left as a dead branch in the
    // hot RHS dispatch. `--cut-flux` remains incompressible-only
    // (gpu/cutflux.*) until the compressible variant is redesigned.

    // Device metric tables of the curvilinear path (null on the uniform
    // path); the stretched NSCBC variant reads its retained transverse flux
    // divergence from them so boundary cells use the interior scheme exactly.
    const DeviceCurv* curvTables() const { return curv_ ? &*curv_ : nullptr; }

    // One SSP-RK3 step in place; dt within stableDt (the reference dt under
    // setLts(true)). Pre: s.nT == nTurb().
    void step(DeviceCompState& s, Real dt);

    // Largest stable dt (acoustic CFL + viscous limit), FP64 reduction. With
    // RANS on, the diffusive bound is per cell and carries mu_t (RANS-only
    // branch); RANS off is the DNS/LES bound byte for byte. LES: the SGS part
    // is the max of the solver's own mu_t of s (updateSgsNut), so mut() and
    // the primitives then hold s's -- not const for that reason.
    double stableDt(const DeviceCompState& s, ReduceScratch& r, double cfl = 0.4);

    // --- RANS (dns/rans.hpp, gpu/rans.hpp; docs/rans.md) ---------------------
    // Install the closure: allocates the scalar stage buffers, mu_t, d and
    // the local-dt buffers, and installs s.fluidMask (enableSteady's).
    // Pre: curvilinear(), !les(), s.model != None, s.wallDist sized g.size(),
    // d > 0, fluidMask empty or g.size(). Never calling it keeps every path
    // bit-for-bit today's solver.
    std::expected<void, hipError_t> enableRans(const RansSetup& s);
    // The steady pseudo-time march without RANS (laminar --steady): the
    // local-dt buffers and the residual mask (1 = real fluid cell, empty =
    // every cell). Pre: curvilinear(), mask empty or g.size().
    std::expected<void, hipError_t>
    enableSteady(const std::vector<std::uint8_t>& fluidMask);
    bool rans() const { return ransModel_ != RansModel::None; }
    RansModel ransModel() const { return ransModel_; }
    int nTurb() const { return ransScalars(ransModel_); }
    // Per-cell mu_t of the last stage, updateMut, updateSgsNut or LES
    // stableDt (LES: the SGS mu_t), image rows included.
    const DeviceScalar& mut() const { return nut_; }
    // mu_t of s into mut() (primitives refreshed). Pre: rans().
    void updateMut(const DeviceCompState& s);
    // The LES twin of updateMut: the SGS mu_t of s into mut() (primitives
    // refreshed), image rows included. Pre: les().
    void updateSgsNut(const DeviceCompState& s);
    // Install the wall/far image of mu_t (LES or RANS): runs after mu_t is
    // formed on every cell and before any flux reads it, so a wall face can
    // carry the molecular stress only. Null = no image rows. Both closures put
    // mu_t (rho nu_t) in the one slot, and the hooks are odd/even maps anyway.
    void setMutHook(MutHook h) { mutHook_ = std::move(h); }
    // Pure rhs of every equation at s (hooks included, scalar kernel in
    // explicit mode) into scratch, then FP64 RMS per equation over the
    // installed fluid mask. Not state differencing. Pre: enableRans or
    // enableSteady ran, s stage-hook consistent, between steps only (it
    // reuses the stage scratch).
    ResidualNorms residual(const DeviceCompState& s, ReduceScratch& r);
    // The same over rows [jLo, jHi] instead of the mask (gpu_rans_mask pins
    // the two equal bitwise on the channel). Pre: as residual().
    ResidualNorms residual(const DeviceCompState& s, int jLo, int jHi,
                           ReduceScratch& r);
    // The pure rhs of every equation at s into `out` (the stage's residual
    // mode; ghost cells meaningless), on either mesh. Pre: s stage-hook
    // consistent, &out != &s, out sized like s with out.nT == nTurb() (low
    // words never written).
    void residualInto(const DeviceCompState& s, DeviceCompState& out);
    // The pure rhs the last residual() wrote, every equation and cell; ghost
    // rows meaningless. Valid until the next step() or residual().
    const DeviceCompState& residualField() const { return q2_; }
    // Per-cell local dt_c = min(cfl/wave_c, diffusive_c) into the lts ratios
    // dt_c/dtRef; returns dtRef = min_c dt_c (FP64). The diffusive bound is
    // RANS's per-cell one, or the molecular one (mu_t = 0) without RANS.
    // Pre: enableRans or enableSteady ran.
    double localDt(const DeviceCompState& s, ReduceScratch& r, double cfl);
    // The per-cell dt_c of the last localDt, FP64, g.size() entries.
    const DeviceBuffer<double>& localDtCells() const { return dtc_; }
    // step() passes the lts ratios in every StageCombine when on.
    // Pre (on): localDt or uploadLts ran.
    void setLts(bool on);
    // --lts-max: later localDt calls cap every ratio dt_c/dtRef at R (dtRef
    // itself unchanged). R == 0 = unlimited, the default. Pre: R == 0 or >= 1.
    void setLtsMax(double R) { ltsMax_ = R; }
    // Compensated accumulation (DeviceCompState::lo): the stage buffers gain
    // low words and every step() writes through them. Irreversible. Every
    // state-overwriting hook must zero its overwritten cells' low words (the
    // channel wall and the O-grid rows do).
    // Post: step() requires s.compensated().
    std::expected<void, hipError_t> enableCompensation();
    bool compensated() const { return compensated_; }
    // Diagnostic/test: the lts ratios from host, g.size() entries >= 1.
    hipError_t uploadLts(const std::vector<Real>& ratios);
    // --precond K (masterplan_levers M2, dns/precond.hpp): Weiss-Smith low-
    // Mach preconditioning of the pseudo-time march. Later localDt calls bound
    // each cell by its preconditioned wave u' + c', and stepPrecond applies
    // the rank-one correction to the TOTAL residual (stage kernel + every rhs
    // hook). offRows (1 = eps 1, the physical speeds): boundary/overwrite
    // rows, ghost and solid cells. Allocates one residual scratch state.
    // Never calling it keeps step() and localDt bit for bit.
    // Pre: curvilinear(), !rans(), K > 0, uInf > 0, offRows g.size().
    std::expected<void, hipError_t>
    setPrecond(double K, double uInf, const std::vector<std::uint8_t>& offRows);
    bool precond() const { return precKU2_ > 0.0; }
    // One preconditioned SSP-RK3 step in place: step()'s stage sequence and
    // hooks, but each stage materializes R = residualInto(stage input) and
    // writes out = ca a + cb b + crAt(c) P R. Same fixed point as step()
    // (P invertible); a different pseudo-time path. Pre: precond(),
    // s.nT == 0, dt from the preconditioned localDt.
    void stepPrecond(DeviceCompState& s, Real dt);
    // SST: positivity-guard hits since enableRans (0 for SA). Synchronizes.
    unsigned long long floorHits() const;
    // SST: the per-cell F1 blend of the last stage or updateMut.
    const DeviceScalar& f1() const { return f1_; }
    // The eddy conductivity scale cp / Pr_t the RANS/LES face flux uses.
    Real eddyKapScale() const { return Real(tr_.cp() / prt_); }
    // RANS: the wall distance enableRans uploaded (FP32, g.size()).
    const DeviceScalar& wallDistance() const { return wd_; }

    // Primitives of s into the internal scratch (valid until next step call);
    // exposed for app-side diagnostics/BC kernels.
    void primitives(const DeviceCompState& s);
    const DeviceScalar& u() const { return u_; }
    const DeviceScalar& v() const { return v_; }
    const DeviceScalar& w() const { return w_; }
    const DeviceScalar& e() const { return e_; }
    const DeviceScalar& p() const { return p_; }
    const DeviceScalar& T() const { return T_; }

    const GpuGrid& grid() const { return g_; }
    Real gamma() const { return Real(tr_.gamma); }
    Real mu() const { return Real(tr_.muRef); }
    bool les() const { return csLes_ > Real(0); }

    // The gas: viscosity law, Prandtl number, R and gamma. Everything that
    // needs mu or kappa goes through this, so a temperature-dependent law
    // reaches the surface tallies and the boundary kernels for free.
    const Transport& transport() const { return tr_; }

    // Install the rotating-frame zone (--rotor-frame mrf). Uploads the
    // physical per-axis coordinate tables (through `remap`, the identity on a
    // uniform mesh) that the kernels evaluate the frame velocity from. Never
    // calling this keeps every path bit-for-bit the inertial solver.
    // Pre: z as documented on MrfZone.
    hipError_t setMrf(const MrfZone& z, const AxisRemap3& remap);
    // The same zone on a GENERAL z-extruded curvilinear mesh (the O-grid):
    // uploads a per-node stream potential and per-cell centres built from the
    // mesh's own nodes, so the transport correction telescopes exactly where
    // no separable coordinate table exists. Pre: z as on MrfZone with
    // z.axis == 2, and `mesh` is the mesh this solver's metrics were built
    // from, z-invariant in the extrusion sense (the O-grid builder enforces
    // it).
    hipError_t setMrfCurv(const MrfZone& z, const CurvMesh& mesh);
    bool mrfActive() const { return mrf_.has_value(); }
    // The zone as a by-value kernel view (HIP TUs only); pre: mrfActive().
    MrfView mrfView() const;

    // Called on every RK stage input before its rhs (boundary conditions and
    // embedded-boundary fill); the hook may mutate the state in place.
    // Idempotent::yes skips the call on step()'s INPUT, which is then the
    // caller's job: legal only when (a) the hooked state depends on no cell
    // the hook writes, so hook(hook(s)) == hook(s) bit for bit -- never an
    // in-place relaxation such as comp_far's sponge -- and (b) every state the
    // caller passes to step() already went through the hook, untouched since.
    void setStageHook(std::function<void(DeviceCompState&)> h,
                      Idempotent idem = Idempotent::no)
    {
        hook_ = std::move(h);
        hookAtEntry_ = idem == Idempotent::no;
    }

    // Curvilinear only: the stage kernels (mean flow and RANS scalars) write
    // cells [r.first, r.end) only -- the wedge's real plane. Primitives, mu_t,
    // the hooks, localDt and stableDt stay whole-grid. Pre: curvilinear(),
    // r.first < r.end <= g.size(), and the stage hook rewrites every field and
    // low word of every cell outside before anything reads it (the stage
    // output there is never formed). Default: the whole grid.
    void setStageRange(CellRange r);

    // Called after the periodic rhs of every RK stage with the stage state s,
    // the stage OUTPUT state out, and the combination the interior cells were
    // written through; may overwrite rhs rows (NSCBC characteristic
    // boundaries) by writing its own rhs through `cmb`. Stage primitives are
    // live in the scratch during the call.
    void setRhsHook(
        std::function<void(const DeviceCompState&, DeviceCompState&,
                           const StageCombine&)> h)
    {
        rhsHook_ = std::move(h);
    }

private:
    DeviceComp(const GpuGrid& g, const Transport& tr, Real csLes, SgsModel sgs);

    // One RK stage: out = cmb.ca a + cmb.cb b + cmb.crAt(c) L(s), boundary
    // hooks included. `s` supplies the stencil. `out` must alias NEITHER the
    // stencil state NOR cmb.a: the rhs hook is a second kernel over the same
    // cells that re-reads a (see step()).
    // residualMode: the scalar kernel writes its pure rhs instead of the
    // point-implicit combination (the mean flow is pure rhs under ca = cb = 0,
    // cr = 1 already).
    void stage(const DeviceCompState& s, const StageCombine& cmb,
               DeviceCompState& out, bool residualMode = false);
    // stepPrecond's stage: R of s into precR_, then out = ca a + cb b +
    // crAt(c) P R over the stage range. Same aliasing contract as stage().
    void precondStage(const DeviceCompState& s, const StageCombine& cmb,
                      DeviceCompState& out);
    // The primitives of s, then mu_t from them into nut_, then the mesh
    // family's image rows (mutHook_); also the scalar stage's per-cell cache
    // of s. SA fuses the two pointwise kernels into one launch (same bits).
    // Pre: rans().
    void primitivesAndMut(const DeviceCompState& s);
    // The SGS mu_t of s into nut_ on either mesh, then mutHook_. Pre: les(),
    // primitives(s) current.
    void sgsNut(const DeviceCompState& s);
    // The per-cell RANS diffusive dt functor over s. Pre: rans().
    rans::RansDiffDt ransDiffDt(const DeviceCompState& s) const;
    // (Re)allocate both stage buffers for nT scalars, low words iff
    // compensated_.
    std::expected<void, hipError_t> allocStages(int nT);
    // The local-dt buffers (lts_, dtc_) and the residual mask. Pre: mask
    // empty or g.size().
    std::expected<void, hipError_t>
    allocSteady(const std::vector<std::uint8_t>& mask);
    // Device pointer table of curv_; pre: curvilinear().
    CurvView curvView() const;

    GpuGrid g_;
    Transport tr_;
    Real csLes_ = Real(0);  // SGS model constant (0 = molecular DNS)
    Real smagCd_ = Real(0); // C^2 * Delta^2 (filter width folded in)
    SgsModel sgs_ = SgsModel::Smagorinsky;
    ScalarBoundary bnd_{};  // RANS open domain faces (enableRans)
    Real prt_ = Real(0.9);  // turbulent Prandtl number
    // Two stage buffers, not three: SSP-RK3 needs the running state plus one
    // stage output at a time once the rhs is not materialized separately.
    DeviceCompState q1_, q2_;
    std::optional<DeviceCurv> curv_; // engaged = curvilinear rhs/stableDt
    // Rotating-frame zone (engaged by setMrf): parameters plus the uploaded
    // per-axis physical coordinate tables (cell centres and +faces).
    // setMrfCurv fills the node-psi and per-cell-centre tables instead.
    std::optional<MrfZone> mrf_;
    DeviceScalar mrfCc_[3], mrfFc_[3];
    DeviceScalar mrfPsi_, mrfPc_[3];
    DeviceScalar u_, v_, w_, e_, p_, T_;
    DeviceScalar nut_;      // per-cell dynamic eddy viscosity (LES or RANS)
    MutHook mutHook_; // the mu_t image rows (setMutHook; LES or RANS)
    // RANS (enableRans): model, wall distance, the lts ratios and the FP64
    // per-cell dt scratch localDt reduces.
    RansModel ransModel_ = RansModel::None;
    bool ransTransport_ = false;
    const Real* testSrc_[2] = {nullptr, nullptr};
    DeviceScalar wd_, lts_, f1_;
    // rans::RansCellCache storage: specific scalars (phi1 SST only) and SST's
    // rate / grad k . grad omega, written by primitivesAndMut, read by the
    // scalar stage. cacheOf_: the t0 they were computed from (stage's assert).
    DeviceScalar ransPhi_[2], ransRate_, ransGkw_;
    const Real* cacheOf_ = nullptr;
    DeviceBuffer<double> dtc_;
    // Residual / guard mask (1 = counted); empty = every cell. maskCount_ is
    // the number of counted cells, the residual's RMS denominator.
    DeviceBuffer<std::uint8_t> mask_;
    DeviceBuffer<std::uint8_t> solid_; // ScalarBoundary::solid (immersed)
    double maskCount_ = 0;
    DeviceBuffer<unsigned long long> hits_;
    double omegaFloor_ = 0;
    bool ltsOn_ = false;
    double ltsMax_ = 0;   // --lts-max; 0 = unlimited
    bool compensated_ = false;
    // --precond: K^2 uInf^2 (0 = off), the eps = 1 rows, the residual scratch.
    double precKU2_ = 0.0;
    DeviceBuffer<std::uint8_t> precOff_;
    DeviceCompState precR_;
    std::function<void(DeviceCompState&)> hook_;
    bool hookAtEntry_ = true; // setStageHook's Idempotent::no
    CellRange stageRange_{};  // setStageRange; the whole grid by default
    std::function<void(const DeviceCompState&, DeviceCompState&,
                       const StageCombine&)> rhsHook_;
};

// Brinkman-style solid-band penalization (--penalize eta): cell momentum
// decays by exp(-dt*(1-lambda)/eta); rho and E stay (the drained kinetic
// energy implicitly becomes internal energy -- conservative, dissipative).
// Smooth along the surface, unlike the ghost staircase; see the incompressible
// twin in gpu/cutcell.hpp. lam is the uploaded cell fluid fraction.
void penalizeMomentum(const GpuGrid& g, const DeviceScalar& lam, Real eta,
                      Real dt, DeviceCompState& s);

} // namespace dns::gpu
