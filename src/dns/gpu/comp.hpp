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

// The RANS closure installed by DeviceComp::enableRans. Host-side; validated
// upstream.
struct RansSetup
{
    RansModel model = RansModel::None;   // != None
    std::vector<double> wallDist;        // g.size(), > 0 everywhere
    bool mirrorYLo = false, mirrorYHi = false; // mu_t odd across y ghost rows
    // TEST ONLY (gpu_rans G3): transport by the mean mass flux alone -- no
    // sources, no diffusion (SA). Never set by a runner.
    bool transportOnly = false;
    // SST: the specific-omega floor of the COUNTED positivity guard (hits
    // are reported, never silent; docs/rans.md).
    double omegaFloor = 1e-10;
};

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
    // nz == 1 (the |S| z-shear terms cancel by periodic wrap).
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
    // branch); RANS off is the DNS/LES bound byte for byte.
    double stableDt(const DeviceCompState& s, ReduceScratch& r, double cfl = 0.4) const;

    // --- RANS (dns/rans.hpp, gpu/rans.hpp; docs/rans.md) ---------------------
    // Install the closure: allocates the scalar stage buffers, mu_t and d.
    // Pre: curvilinear(), !les(), s.model != None, s.wallDist sized g.size(),
    // d > 0. Never calling it keeps every path bit-for-bit today's solver.
    std::expected<void, hipError_t> enableRans(const RansSetup& s);
    bool rans() const { return ransModel_ != RansModel::None; }
    RansModel ransModel() const { return ransModel_; }
    int nTurb() const { return ransScalars(ransModel_); }
    // Per-cell mu_t of the last stage or updateMut (LES: the SGS mu_t).
    const DeviceScalar& mut() const { return nut_; }
    // mu_t of s into mut() (primitives refreshed). Pre: rans().
    void updateMut(const DeviceCompState& s);
    // Pure rhs of every equation at s (hooks included, scalar kernel in
    // explicit mode) into scratch, then FP64 RMS per equation over rows
    // [jLo, jHi]. Not state differencing. Pre: rans(), s stage-hook
    // consistent, between steps only (it reuses the stage scratch).
    ResidualNorms residual(const DeviceCompState& s, int jLo, int jHi,
                           ReduceScratch& r);    // Per-cell local dt_c = min(cfl/wave_c, diffusive_c) into the lts ratios
    // dt_c/dtRef; returns dtRef = min_c dt_c (FP64). Pre: rans().
    double localDt(const DeviceCompState& s, ReduceScratch& r, double cfl);
    // step() passes the lts ratios in every StageCombine when on.
    // Pre (on): localDt or uploadLts ran.
    void setLts(bool on);
    // Compensated accumulation (DeviceCompState::lo): the stage buffers gain
    // low words and every step() writes through them. Irreversible. Only the
    // channel wall keeps its overwritten cells' low words consistent; a runner
    // installing any other state-overwriting hook must not enable this.
    // Post: step() requires s.compensated().
    std::expected<void, hipError_t> enableCompensation();
    bool compensated() const { return compensated_; }
    // Diagnostic/test: the lts ratios from host, g.size() entries >= 1.
    hipError_t uploadLts(const std::vector<Real>& ratios);
    // SST: positivity-guard hits since enableRans (0 for SA). Synchronizes.
    unsigned long long floorHits() const;
    // SST: the per-cell F1 blend of the last stage or updateMut.
    const DeviceScalar& f1() const { return f1_; }

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
    void setStageHook(std::function<void(DeviceCompState&)> h) { hook_ = std::move(h); }

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
    // mu_t from the live primitives of s into nut_, mirrored odd across the
    // wall ghost rows. Pre: rans(), primitives(s) current.
    void ransMut(const DeviceCompState& s);
    // The per-cell RANS diffusive dt functor over s. Pre: rans().
    rans::RansDiffDt ransDiffDt(const DeviceCompState& s) const;
    // (Re)allocate both stage buffers for nT scalars, low words iff
    // compensated_.
    std::expected<void, hipError_t> allocStages(int nT);
    // Device pointer table of curv_; pre: curvilinear().
    CurvView curvView() const;

    GpuGrid g_;
    Transport tr_;
    Real csLes_ = Real(0);  // SGS model constant (0 = molecular DNS)
    Real smagCd_ = Real(0); // C^2 * Delta^2 (filter width folded in)
    SgsModel sgs_ = SgsModel::Smagorinsky;
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
    // RANS (enableRans): model, wall distance, mirror flags, the lts ratios
    // and the FP64 per-cell dt scratch localDt reduces.
    RansModel ransModel_ = RansModel::None;
    bool ransMirrorLo_ = false, ransMirrorHi_ = false, ransTransport_ = false;
    DeviceScalar wd_, lts_, f1_;
    DeviceBuffer<double> dtc_;
    DeviceBuffer<unsigned long long> hits_;
    double omegaFloor_ = 0;
    bool ltsOn_ = false;
    bool compensated_ = false;
    std::function<void(DeviceCompState&)> hook_;
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
