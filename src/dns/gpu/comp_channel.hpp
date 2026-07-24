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
class DeviceChannelWall
{
public:
    struct Config
    {
        double gamma = 1.4, Rgas = 1.0, Twall = 1.0;
        bool isothermal = true;
    };

    static std::expected<DeviceChannelWall, hipError_t> make(const Grid& g,
                                                             Config c);
    void apply(DeviceCompState& s) const;

private:
    DeviceChannelWall(const GpuGrid& g, Config c) : g_(g), c_(c) {}

    GpuGrid g_;
    Config c_;
};

// Constant streamwise body force on the FLUID rows (1..ny-2): k.mx += f and
// k.E += f * u, with u the live stage primitive. Applied inside the RHS hook
// so it rides SSP-RK3 exactly; the ghost rows are excluded (the stage hook
// overwrites them anyway).
void addBodyForceX(const GpuGrid& g, double f, const DeviceScalar& u,
                   DeviceCompState& k);

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
