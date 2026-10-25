// Channel-flow boundary + forcing + statistics pieces for the compressible
// curvilinear path: the y-wall mirror (isothermal or adiabatic no-slip), the
// constant streamwise body force, and the deterministic per-y-plane moment
// sums the profile statistics are built from. x and z stay periodic (the
// core's native behaviour) -- there is deliberately no x or z kernel here.
// GPU-only header.
#pragma once

#include "dns/gpu/comp.hpp"

#include <expected>
#include <vector>

namespace dns::gpu {

// Mirror walls on the y extremes for the channel: ghost rows j = 0 and ny-1
// mirror rows 1 and ny-2 (the mesh makes them exact geometric reflections, so
// the value mirror is 2nd order at the wall face). No-slip: all momentum
// antimirrored. Isothermal: T_ghost = 2 Twall - T_in, p copied (zero
// wall-normal pressure gradient), rho from the EOS -- the Fourier flux
// through the wall face is the heat sink that balances the body-force power.
// isothermal = false reproduces the adiabatic kFarWallY mirror bit-for-bit.
// A state carrying RANS scalars also gets their ghosts, rebuilt with the
// ghost's OWN density: SA (nT == 1) nu~ antimirrored; SST (nT == 2) k
// antimirrored and omega_g = 2 omega_w - omega_1 with TMR's wall value
// omega_w = 60 nu_w / (beta1 d1^2) = sstOmegaCoef / rho_w, rho_w the wall-face
// average density.
class DeviceChannelWall
{
public:
    struct Config
    {
        double gamma = 1.4, Rgas = 1.0, Twall = 1.0;
        bool isothermal = true;
        double sstOmegaCoef = 0; // 60 mu / (beta1 d1^2), d1 the first centre
    };

    static std::expected<DeviceChannelWall, hipError_t> make(const Grid& g,
                                                             Config c);
    void apply(DeviceCompState& s) const;

private:
    DeviceChannelWall(const GpuGrid& g, Config c) : g_(g), c_(c) {}

    GpuGrid g_;
    Config c_;
};

// Constant streamwise body force on the FLUID rows (1..ny-2): out.mx +=
// crAt(c) f and out.E += crAt(c) f u, with u the live stage primitive. Applied
// inside the RHS hook so it rides SSP-RK3 exactly (and local time stepping:
// the per-cell crAt is what keeps the steady state right); the ghost rows are
// excluded (the stage hook overwrites them anyway).
void addBodyForceX(const GpuGrid& g, double f, const StageCombine& cmb,
                   const DeviceScalar& u, DeviceCompState& out);

// The implicit steady march's column transfer (--steady-scheme implicit,
// docs/rans.md): the RANS channel mean is x/z-homogeneous, so ONE column
// carries the whole state. Packed layout field-major [v * ny + j], v in
// DeviceCompState order (rho, mx, my, mz, E, t0[, t1]), 5 + s.nT fields.
// Column (0, j, 0) of every field into `host` (resized), through `packed`
// (allocated on first use).
hipError_t channelGatherColumn(const GpuGrid& g, const DeviceCompState& s,
                               DeviceBuffer<Real>& packed,
                               std::vector<Real>& host);
// Every cell (i, j, k) of the FLUID rows 1..ny-2 takes host[v * ny + j]; the
// ghost rows are the wall's (apply it after). A plain-state edit, like
// boostX. Pre: host.size() == (5 + s.nT) * ny, !s.compensated().
hipError_t channelSetRows(const GpuGrid& g, const std::vector<Real>& host,
                          DeviceBuffer<Real>& packed, DeviceCompState& s);

// Deterministic FP64 per-y-plane sum and max of one field: out[j*2 + 0] =
// sum over (i,k), out[j*2 + 1] = max over (i,k). One thread per row, fixed
// order. `host` is resized to g.ny * 2.
hipError_t channelPlaneScalarSums(const GpuGrid& g, const DeviceScalar& field,
                                  DeviceBuffer<double>& scratch,
                                  std::vector<double>& host);

// Deterministic FP64 per-y-plane moment sums: for every row j,
//   out[j*8 + 0..7] = sum over (i,k) of {rho, rho u, rho v, rho u^2,
//                                        rho v^2, rho w^2, rho u v, rho T}.
// One thread per row, fixed (i,k) order -- bit-reproducible run to run.
// `host` is resized to g.ny * 8 and holds the downloaded sums on return.
hipError_t channelPlaneStats(const GpuGrid& g, const DeviceScalar& rho,
                             const DeviceScalar& u, const DeviceScalar& v,
                             const DeviceScalar& w, const DeviceScalar& T,
                             DeviceBuffer<double>& scratch,
                             std::vector<double>& host);

} // namespace dns::gpu
