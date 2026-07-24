// HIP mirror of the compressible solver (dns/comp.hpp): FP32 fields, FP64
// reductions, same Kennedy-Gruber + face-viscous + SSP-RK3 scheme; nz == 1 is
// the exact 2D restriction. GPU-only header.
#pragma once

#include "dns/comp.hpp"
#include "dns/cutcell.hpp"
#include "dns/sgs.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/reductions.hpp"

#include <expected>
#include <functional>
#include <optional>

namespace dns {
struct CurvMetrics;
}

namespace dns::gpu {

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
    make(const Grid& g, double gamma, double mu, double csLes = 0.0,
         SgsModel sgs = SgsModel::Smagorinsky);

    // Curvilinear variant on a distorted regular mesh (dns/curvmesh.hpp):
    // rhs/stableDt switch to the finite-volume metric form, everything else
    // (RK3, primitives, hooks) is shared. No LES on curvilinear meshes.
    static std::expected<DeviceComp, hipError_t> make(const CurvMetrics& m,
                                                      double gamma, double mu);

    bool curvilinear() const { return curv_.has_value(); }

    // --cut-flux (uniform mesh only): after this, rhs() overwrites the cells
    // of `flag` (dns::buildCutFlag) with aperture-weighted face fluxes over
    // lambda-floored volumes. Faces joining a fluid cell to a SOLID-CENTRE
    // cell keep the FULL flux -- that is the existing ghost-mirror wall
    // closure, so kBodyFlux's fluid<->ghost tally stays the exact budget of
    // what the update applied. `solid` must be the same centre classification
    // the ghost fill uses (sdf < 0). Cells outside the flag keep kCompRhs's
    // exact bits (they are simply not overwritten). stableDt folds the
    // 1/lambda~ amplification into the acoustic bound.
    std::expected<void, hipError_t>
    attachCutFlux(const CutGeometry& cut, const std::vector<unsigned char>& flag,
                  const std::vector<unsigned char>& solid);

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
    Real gamma() const { return gamma_; }
    Real mu() const { return mu_; }
    bool les() const { return csLes_ > Real(0); }

    // Heat conductivity mu cp / Pr (surface-force tally needs it).
    Real kappaCond() const { return mu_ * gamma_ * Rgas_ / ((gamma_ - Real(1)) * Pr_); }

    // Called on every RK stage input before its rhs (boundary conditions and
    // embedded-boundary fill); the hook may mutate the state in place.
    void setStageHook(std::function<void(DeviceCompState&)> h) { hook_ = std::move(h); }

    // Called after the periodic rhs of every RK stage with the stage state s
    // and its rhs k; may overwrite rhs rows (NSCBC characteristic
    // boundaries). Stage primitives are live in the scratch during the call.
    void setRhsHook(std::function<void(const DeviceCompState&, DeviceCompState&)> h)
    {
        rhsHook_ = std::move(h);
    }

private:
    DeviceComp(const GpuGrid& g, Real gamma, Real mu, Real csLes, SgsModel sgs);

    void rhs(const DeviceCompState& s, DeviceCompState& out);

    GpuGrid g_;
    Real gamma_, mu_, Pr_ = Real(0.72), Rgas_ = Real(1);
    Real csLes_ = Real(0);  // SGS model constant (0 = molecular DNS)
    Real smagCd_ = Real(0); // C^2 * Delta^2 (filter width folded in)
    SgsModel sgs_ = SgsModel::Smagorinsky;
    Real prt_ = Real(0.9);  // turbulent Prandtl number
    DeviceCompState k_, q1_, q2_;
    std::optional<DeviceCurv> curv_; // engaged = curvilinear rhs/stableDt
    DeviceScalar u_, v_, w_, e_, p_, T_;
    DeviceScalar nut_;      // per-cell dynamic eddy viscosity (LES only)
    bool cutFlux_ = false;  // attachCutFlux engaged (uniform path only)
    DeviceScalar cutAx_, cutAy_, cutAz_, cutLam_;
    DeviceBuffer<unsigned char> cutFlag_, cutSolid_;
    DeviceBuffer<const Real*> ptrs_; // axpy pointer table
    std::function<void(DeviceCompState&)> hook_;
    std::function<void(const DeviceCompState&, DeviceCompState&)> rhsHook_;
};

// Brinkman-style solid-band penalization (--penalize eta): cell momentum
// decays by exp(-dt*(1-lambda)/eta); rho and E stay (the drained kinetic
// energy implicitly becomes internal energy -- conservative, dissipative).
// Smooth along the surface, unlike the ghost staircase; see the incompressible
// twin in gpu/cutcell.hpp. lam is the uploaded cell fluid fraction.
void penalizeMomentum(const GpuGrid& g, const DeviceScalar& lam, Real eta,
                      Real dt, DeviceCompState& s);

} // namespace dns::gpu
