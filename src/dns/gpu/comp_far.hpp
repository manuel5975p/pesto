// Validated far-field for the compressible tunnel apps: streamwise freestream
// sponge (soft inflow + acoustic absorber), lateral free-slip mirror walls or
// lateral sponge bands (open far field), and a quiet-start freestream ramp via
// exact uniform Galilean boosts (a boost radiates no acoustics). GPU-only.
#pragma once

#include "dns/gpu/comp.hpp"

namespace dns::gpu {

class DeviceCompFar
{
public:
    struct Config
    {
        double rhoInf = 1.0, uInf = 1.0, pInf = 1.0, gamma = 1.4;
        int ns = 8;              // sponge band width, cells
        double sigMax = 0.5;     // per-stage relax strength at the boundary
        // Open lateral band: relax rho/p/tangential velocity to freestream,
        // leave the boundary-normal momentum free (the displacement flow
        // vents instead of reflecting). Default = free-slip mirror wall.
        bool openY = false;
        bool openZ = false;      // ignored when nz == 1
        // Closed-y walls become adiabatic NO-SLIP mirrors (all momentum
        // antimirrored) instead of free-slip; ignored when openY.
        bool noslipY = false;
        // RANS: the specific freestream scalars the sponge relaxes toward
        // (read only when the state carries scalars).
        double phiInf[2] = {0.0, 0.0};
    };

    // Periodic grid; 1 <= ns < nx/2, ny/nz >= 3 on walled axes.
    static std::expected<DeviceCompFar, hipError_t> make(const Grid& g, Config c);

    // Sponge + walls toward the current freestream; call per RK stage. A
    // state with RANS scalars (s.nT > 0) also gets them relaxed to rho phi_inf
    // with the same strength and copied EVEN onto the mirror rows; a
    // compensated one gets the low words of every cell written here zeroed.
    // Neither exists without RANS, so that path is today's bit for bit.
    void apply(DeviceCompState& s) const;

    // mu_t and F1 EVEN onto the mirror rows (the RANS mutHook's far part);
    // f1 may be null. Nothing on an open axis.
    void applyMut(Real* nut, Real* f1) const;

    // Exact Galilean x-boost of the whole state by du (silent by construction);
    // quiet start = repeated small boosts paired with setFreestream(u + du).
    void boost(DeviceCompState& s, double du) const;

    void setFreestream(double u) { u_ = u; }
    double freestream() const { return u_; }

private:
    DeviceCompFar(const Grid& g, const Config& c);

    GpuGrid g_;
    Config c_;
    double u_;
};

} // namespace dns::gpu
