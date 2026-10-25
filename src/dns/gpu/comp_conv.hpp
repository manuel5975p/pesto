// Rayleigh-Benard convection pieces for the compressible uniform-mesh path:
// the two isothermal no-slip y walls (hot floor, cold lid), the OPTIONAL
// no-slip adiabatic lateral walls, the gravity source and the deterministic
// per-y-plane moment sums the Nusselt numbers are built from. With periodic
// sides x and z keep the core's native behaviour and no lateral kernel runs at
// all, so that case is bit-for-bit what it was before `--sides` existed.
//
// The solver is FULLY COMPRESSIBLE, so density responds to temperature through
// the equation of state and buoyancy is just gravity acting on that density.
// There is no Boussinesq approximation here and no buoyancy term beyond
// addGravityY: the whole effect is rho(T) g.
//
// GPU-only header.
#pragma once

#include "dns/gpu/comp.hpp"

#include <expected>
#include <vector>

namespace dns::gpu {

// Mirror walls on the y extremes: ghost rows j = 0 and ny-1 mirror rows 1 and
// ny-2 (uniform mesh, so the mirror is an exact geometric reflection and the
// value mirror is 2nd order at the wall face). No-slip: all momentum
// antimirrored. Isothermal with a DIFFERENT temperature per wall -- that
// difference is the entire experiment: T_ghost = 2 Twall - T_in, p imaged
// HYDROSTATICALLY (dp/dy = -rho g at the plate, the wall-normal momentum
// balance under gravity: p_ghost = p_in +- rho_in g hy), rho from the EOS.
//
// Tbot == Ttop reproduces DeviceChannelWall's isothermal branch bit-for-bit.
class DeviceConvWall
{
public:
    struct Config
    {
        double gamma = 1.4, Rgas = 1.0;
        double Tbot = 1.0, Ttop = 1.0;   // both > 0; Tbot > Ttop drives convection
        // Side fraction of a centred hot patch on the FLOOR, in (0,1]. The
        // floor holds Tbot inside it and Ttop outside; 1 = the uniformly
        // heated plate, bit-for-bit unchanged. Vacuous in z at nz == 1, so
        // the 2D case is the middle strip in x. Centred on the FLUID extent,
        // so closing the sides does not move it.
        double patch = 1.0;
        // Lateral faces (--sides closed): x, and z when nz > 1, become no-slip
        // ADIABATIC mirrors on their own ghost column -- momentum antimirrored,
        // T and p COPIED (hence rho and E copied), which is the exact
        // zero-heat-flux, zero-mass-flux wall. Adiabatic on purpose: the drop
        // across the PLATES is the experiment, and a conducting side would be a
        // second uncontrolled source. false leaves x and z periodic.
        bool closedSides = false;
        // Gravity magnitude (along -y), the SAME gy addGravityY applies: the
        // plates' pressure image balances it. 0 is the zero-gradient copy.
        double gravity = 0.0;
    };

    // ny >= 4, gamma > 1, Rgas > 0, both wall temperatures > 0, gravity >= 0;
    // with
    // closedSides, nx >= 4 and (nz == 1 or nz >= 4).
    static std::expected<DeviceConvWall, hipError_t> make(const Grid& g,
                                                          Config c);
    // Plates first, then x, then z, so a corner ghost holds the diagonal
    // reflection of an interior cell.
    void apply(DeviceCompState& s) const;

private:
    DeviceConvWall(const GpuGrid& g, Config c) : g_(g), c_(c) {}

    GpuGrid g_;
    Config c_;
};

// Gravity of magnitude gy pointing in -y, on the FLUID cells (rows 1..ny-2,
// and with `closed` the lateral ghost columns excluded too):
// out.my += crAt * (-gy rho) and out.E += crAt * (-gy my), with rho and my the
// live stage state. Applied inside the RHS hook so it rides SSP-RK3 exactly
// and the FUSED stage kernel stays untouched (StageCombine, comp.hpp); the
// ghost rows are excluded because the wall mirror owns them.
//
// gy does not enter stableDt: the free-fall bound sqrt(h/gy) is orders above
// the acoustic one for every case this solver can run (the layer would have to
// be shorter than a cell for it to bind).
void addGravityY(const GpuGrid& g, double gy, const StageCombine& cmb,
                 bool closed, const DeviceCompState& s, DeviceCompState& out);

// Deterministic FP64 per-y-plane moment sums: for every row j,
//   out[j*6 + 0..5] = sum over (i,k) of
//                     {rho, T, v, v T, rho (u^2+v^2+w^2), rho v T}.
// One thread per row, fixed (i,k) order -- bit-reproducible run to run.
// `host` is resized to g.ny * 6 and holds the downloaded sums on return. The
// sums run over the FLUID (i,k) only: a lateral ghost column holds the wall
// mirror and would bias every plane mean.
hipError_t convPlaneStats(const GpuGrid& g, bool closed, const DeviceScalar& rho,
                          const DeviceScalar& u, const DeviceScalar& v,
                          const DeviceScalar& w, const DeviceScalar& T,
                          DeviceBuffer<double>& scratch,
                          std::vector<double>& host);

} // namespace dns::gpu
