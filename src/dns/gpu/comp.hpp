// HIP mirror of the compressible solver (dns/comp.hpp): FP32 fields, FP64
// reductions, same Kennedy-Gruber + face-viscous + SSP-RK3 scheme; nz == 1 is
// the exact 2D restriction. GPU-only header.
#pragma once

#include "dns/comp.hpp"
#include "dns/cutcell.hpp"
#include "dns/transport.hpp"
#include "dns/sgs.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/reductions.hpp"

#include <expected>
#include <functional>
#include <optional>

namespace dns {
struct CurvMetrics;
struct AxisRemap3;
}

namespace dns::gpu {

struct CurvView; // dns/gpu/internal/comp_flux.hpp (HIP TUs only)
struct MrfView;  // dns/gpu/internal/comp_flux.hpp (HIP TUs only)

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

// Conservative state on device; each buffer has g.size() cells.
struct DeviceCompState
{
    DeviceScalar rho, mx, my, mz, E;

    static std::expected<DeviceCompState, hipError_t> make(std::size_t n);

    hipError_t upload(const CompState& h);
    hipError_t download(CompState& h) const;
};

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
// combination on its own rows. An additive hook (the channel body force) scales
// its contribution by cr.
struct StageCombine
{
    const Real* a[5];
    const Real* b[5];
    Real ca, cb, cr;
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

    // One SSP-RK3 step in place; dt within stableDt.
    void step(DeviceCompState& s, Real dt);

    // Largest stable dt (acoustic CFL + viscous limit), FP64 reduction.
    double stableDt(const DeviceCompState& s, ReduceScratch& r, double cfl = 0.4) const;

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

    // One RK stage: out = cmb.ca a + cmb.cb b + cmb.cr L(s), boundary hooks
    // included. `s` supplies the stencil, `out` may alias cmb.a (read at the
    // own cell only) but never the stencil state.
    void stage(const DeviceCompState& s, const StageCombine& cmb,
               DeviceCompState& out);
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
    std::optional<MrfZone> mrf_;
    DeviceScalar mrfCc_[3], mrfFc_[3];
    DeviceScalar u_, v_, w_, e_, p_, T_;
    DeviceScalar nut_;      // per-cell dynamic eddy viscosity (LES only)
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
