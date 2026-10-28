// Scale-selective absorbing band for the open compressible boundaries: an
// explicit sink for the content no face-local closure can act on. The central
// skew-symmetric flux transports nothing at the mesh scale (group velocity
// c cos kh, <= 0 from 4 cells per wavelength down; measured acceptance
// spectrum in docs/nscbc.md "Small scales and the strip hand-off"), so
// sub-6dx acoustics and vorticity never reach the characteristic rows -- this
// band removes them where they strand. Layered UNDER DeviceCompNscbc as an
// ADDITIVE rhs term riding SSP-RK3; unlike the --nscbc-strip freestream
// sponge it damps only wavelengths below the sin^4 knee (86% per crossing at
// 4dx, 1.2% at 16dx), so the resolved near field and every measured
// reflection coefficient pass through unchanged. GPU-only.
#pragma once

#include "dns/gpu/comp.hpp"

namespace dns::gpu {

class DeviceCompSink
{
public:
    struct Config
    {
        int nb = 16;         // band width per open face, cells
        double alpha = 1.0;  // sigma_max = alpha c/h; SSP-RK3 needs <= 2
        double cInf = 1.0;   // freestream sound speed (sets the rate scale)
        // Bands live on the x extremes always; on y/z only when that axis is
        // characteristic (matching DeviceCompNscbc::Config openY/openZ).
        bool openY = false;
        bool openZ = false;  // ignored when nz == 1
    };

    // Periodic uniform grid; 1 <= nb, 2 nb < n on every banded axis,
    // 0 < alpha <= 2 (the RK3 real-axis bound at cfl 0.4), cInf > 0.
    static std::expected<DeviceCompSink, hipError_t> make(const Grid& g,
                                                          Config c);

    // out += cmb.crAt(c) * rhs on the banded cells: the compounded-Laplacian
    // high-pass D4 (symbol sin^4(kh/2)) in conservative flux form, flux zero
    // through the domain faces, one-sided (clamped) closure -- exactly zero on
    // constant and linear fields, and sums to zero per axis, so the band
    // redistributes nothing and only drains fluctuation content.
    // Call AFTER the nscbc rhs overwrite; reads s (stage input), never out.
    void apply(const DeviceCompState& s, DeviceCompState& out,
               const StageCombine& cmb) const;

    // The species twin (dns flame), plain write: rho Y_k += crAt Y_k R_rho,
    // R_rho the band's own continuity term and Y_k the stage input's -- the
    // band moves species with the mass and never changes Y. Without it the
    // band damps rho but not rho Y and a uniform Y drifts; with D4 on rho Y
    // itself (measured) Y overshoots its bounds. Call right after apply().
    // Pre: s.nY >= 1, plain cmb.
    void applySpecies(const DeviceCompState& s, DeviceCompState& out,
                      const StageCombine& cmb) const;

private:
    DeviceCompSink(const Grid& g, const Config& c);

    GpuGrid g_;
    Config c_;
};

} // namespace dns::gpu
