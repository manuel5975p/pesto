// The thickened-flame model of `dns flame --tfles` on the device (docs/chem.md,
// "LES species and TFLES"; dns/tfles.hpp holds the formulas), and the flame
// LES's open-face eddy-viscosity image. GPU-only header.
//
// Per step (update, from the step's INPUT state; held through the three RK
// stages -- the coefficients are frozen within a step, the rotor-geometry
// precedent):
//   sensor   Omega_raw = tanh(beta w_s/w_s,max), spread toward colder gas by
//            a separable T-gated exact max filter (x clamped at the open
//            faces, y and z periodic; dns/tfles.hpp kTflesDilate);
//            --tfles-sensor uniform: Omega == 1, no sensor launch.
//   E        Charlette 2002 on u'_Delta = Colin's c2 h^3 |lap(curl u)|
//            (Delta/h)^(1/3), Delta = F delta_L; E = 1 on the kTflesEdgeRows
//            outermost x rows (the curl would difference across the wrap).
//   out      FE = F E, EoF = E / F, Om = Omega (DeviceComp reads them through
//            SpeciesSetup::tfles*).
// One thread per cell, each writing its own index; no atomics. A uniform
// sensor without the efficiency is constant and is formed once at make().
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/grid.hpp"
#include "dns/tfles.hpp"

#include <expected>
#include <vector>

namespace dns::gpu {

class DeviceTfles
{
public:
    // Allocates FE, EoF, Om (and the sensor / curl scratch the setup needs).
    // Pre: s as TflesSetup documents; !s.charlette || g.nz > 1.
    static std::expected<DeviceTfles, hipError_t> make(const Grid& g,
                                                       const TflesSetup& s);

    // FE, EoF, Om from state s; u, v, w, T its primitives (DeviceComp::
    // primitives(s) just ran). A no-op for a constant field (uniform sensor,
    // no efficiency).
    void update(const DeviceCompState& s, const DeviceScalar& u,
                const DeviceScalar& v, const DeviceScalar& w,
                const DeviceScalar& T);

    const Real* fe() const { return fe_.data(); }
    const Real* eof() const { return eof_.data(); }
    const Real* om() const { return om_.data(); }
    const DeviceScalar& feField() const { return fe_; }
    const DeviceScalar& eofField() const { return eof_; }
    const DeviceScalar& omField() const { return om_; }
    const TflesSetup& setup() const { return s_; }

    // The sensor's T-gated dilation of an arbitrary raw field into out (the
    // same three passes update() runs; test entry). Pre: raw, T, out sized
    // g.size(), out distinct from raw and T; !setup().uniform (the ping-pong
    // scratch exists).
    void dilate(const DeviceScalar& raw, const DeviceScalar& T,
                DeviceScalar& out);

    // max_c FE (FP64, exact max reduction).
    double maxFE(ReduceScratch& r) const;

    // Deterministic per-x-column FP64 sums over (j, k), (k, j) ascending, one
    // thread per column: out[i * kTflesSums + q], q = 0 sum Omega, 1 sum
    // Omega E (E = sqrt(FE EoF)), 2 max FE.
    static constexpr int kTflesSums = 3;
    hipError_t columnStats(DeviceBuffer<double>& scratch,
                           std::vector<double>& host) const;

private:
    void combine(const Real* omega, const Real* uh);

    GpuGrid g_{};
    TflesSetup s_{};
    DeviceScalar fe_, eof_, om_;
    DeviceScalar ra_, rb_;        // sensor raw / ping-pong (dynamic sensor)
    DeviceScalar cx_, cy_, cz_, uh_; // curl and grid-scale u' (Charlette)
};

// The flame LES's mu_t image on the open x faces (DeviceComp::setMutHook):
// nut on i = 0 and nx - 1 := nut on i = 1 and nx - 2. The SGS kernel's
// central gradient there differences the inflow row against the outflow row
// across the periodic wrap (issues.md §17) -- for a flame the fresh/burnt jump
// read as strain. Pre: g outlives the hook, g.nx >= 3.
MutHook mutCopyOpenXHook(const GpuGrid& g);

} // namespace dns::gpu
