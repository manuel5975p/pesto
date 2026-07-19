// Tunnel boundary conditions for the compressible solver (2D/3D), applied by
// overwriting boundary cell rows per RK stage (solver core stays periodic):
//   inflow  (i <= 1)  rho/u fixed to freestream, pressure Neumann (copied
//                     from the interior -- outgoing acoustics pass),
//   outflow (i = nx-1) interior state extrapolated, pressure relaxed toward
//                     pInf (Rudy-Strikwerda partially-reflecting outlet),
//   walls   (j and, for nz > 1, k extremes) free-slip mirrors: tangential
//           symmetric (zero wall vorticity), normal antisymmetric.
// A uniform freestream is a machine-exact fixed point. GPU-only header.
#pragma once

#include "dns/gpu/comp.hpp"

#include <expected>

namespace dns::gpu {

class DeviceCompBc
{
public:
    struct Opts
    {
        double rhoInf = 1.0, uInf = 1.0, pInf = 1.0;
        bool wallY = true;    // free-slip walls on the y extremes
        bool wallZ = true;    // free-slip walls on the z extremes (nz > 1)
        double pRelax = 0.1;  // outflow pressure relaxation per stage
                              // (1 = hard pInf, reflective; ~0.1 transmits)
    };

    // gamma from the solver the state belongs to.
    static std::expected<DeviceCompBc, hipError_t> make(const Grid& g,
                                                        double gamma,
                                                        const Opts& o);

    // Overwrite boundary rows; install via DeviceComp::setStageHook (chain
    // before the body ghost fill).
    void apply(DeviceCompState& s);

private:
    DeviceCompBc() = default;

    GpuGrid g_{};
    Real gamma_ = Real(1.4);
    Opts o_{};
};

} // namespace dns::gpu
