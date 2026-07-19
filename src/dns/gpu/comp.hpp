// HIP mirror of the compressible solver (dns/comp.hpp): FP32 fields, FP64
// reductions, same Kennedy-Gruber + face-viscous + SSP-RK3 scheme; nz == 1 is
// the exact 2D restriction. GPU-only header.
#pragma once

#include "dns/comp.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/reductions.hpp"

#include <expected>
#include <functional>

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

// Owns RK3 / primitive scratch; periodic domain (BCs via the stage hook).
class DeviceComp
{
public:
    // gamma > 1, mu >= 0, csLes >= 0. csLes > 0 turns on a Smagorinsky SGS
    // stress (3D-valid): per RK stage a dynamic eddy viscosity mu_t = rho
    // (Cs*Delta)^2 |S| is added to the molecular mu in the face-viscous flux,
    // with eddy conductivity mu_t cp / Pr_t. csLes == 0 is molecular DNS,
    // bit-for-bit unchanged. Delta^2 = (hx hy hz)^(2/3) in 3D, hx hy when
    // nz == 1 (the |S| z-shear terms cancel by periodic wrap).
    static std::expected<DeviceComp, hipError_t> make(const Grid& g, double gamma,
                                                      double mu, double csLes = 0.0);

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
    DeviceComp(const GpuGrid& g, Real gamma, Real mu, Real csLes);

    void rhs(const DeviceCompState& s, DeviceCompState& out);

    GpuGrid g_;
    Real gamma_, mu_, Pr_ = Real(0.72), Rgas_ = Real(1);
    Real csLes_ = Real(0);  // Smagorinsky constant Cs (0 = molecular DNS)
    Real smagCd_ = Real(0); // Cs^2 * Delta^2 (filter width folded in)
    Real prt_ = Real(0.9);  // turbulent Prandtl number
    DeviceCompState k_, q1_, q2_;
    DeviceScalar u_, v_, w_, e_, p_, T_;
    DeviceScalar nut_;      // per-cell dynamic eddy viscosity (LES only)
    DeviceBuffer<const Real*> ptrs_; // axpy pointer table
    std::function<void(DeviceCompState&)> hook_;
    std::function<void(const DeviceCompState&, DeviceCompState&)> rhsHook_;
};

} // namespace dns::gpu
