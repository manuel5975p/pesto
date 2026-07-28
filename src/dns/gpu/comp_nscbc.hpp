// NSCBC/LODI characteristic open boundaries (Poinsot-Lele): outgoing wave
// amplitudes from one-sided interior differences, incoming ones modeled.
// x-lo = stiff freestream inflow, x-hi = subsonic outflow with a soft
// pressure anchor L1 = sigma (1-M^2) c/Lx (p - pInf) and the Yoo-Im
// transverse correction. Optional openY/openZ turn the lateral extremes into
// characteristic far-field faces (replace free-slip walls / sponge bands):
// the incoming acoustic wave is a soft pressure anchor -- the boundary-normal
// velocity stays free, so the body's displacement field vents -- and the
// entropy/shear waves upwind on the sign of the normal velocity (outgoing =
// computed, incoming = gentle freestream relax at etaLat c/L). Cells on two
// or three open faces sum the per-axis LODI contributions (Lodato-style
// edge/corner closure). The boundary cells evolve through these amplitudes
// inside RK3 (rhs overwrite via DeviceComp::setRhsHook, never a state
// overwrite), so waves leave without instantaneous-BC reflection. Transverse
// flux divergence on non-boundary axes is retained exactly (shared faceFlux);
// boundary-normal viscous terms are dropped (Poinsot-Lele viscous
// conditions). GPU-only.
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/remap.hpp"

namespace dns::gpu {

class DeviceCompNscbc
{
public:
    struct Config
    {
        double rhoInf = 1.0, uInf = 1.0, pInf = 1.0, gamma = 1.4;
        double sigma = 0.25; // outflow pressure-anchor strength (Rudy-Strikwerda)
        // Inflow relaxation strength (u, v, w, rho). Must be stiff: the body's
        // blockage pressure gradient decelerates the inflow row, equilibrium
        // sag ~ 1/etaIn (measured: etaIn 2 sagged u_in to 0.79 uInf at the
        // Re60 cylinder -- cd biased 30% low). 100 keeps the sag < 1% and
        // stays RK3-stable (rate dt << 1); the inflow is then acoustically
        // stiff at low frequency, which is what a tunnel inlet is.
        double etaIn = 100.0;
        // Outflow transverse compensation L1 -= betaTrans (Tp - rho c Tu)
        // (Yoo-Im). Measured at cyl Re800 M0.2 512: 0 pumps a growing global
        // mode (cl 2 -> 13), beta = M = 0.2 still drifts, 0.5 = clean limit
        // cycle matching the sponge's cd sigma and St, 1.0 diverges (full
        // cancellation is ill-posed at low M). < 0 = freestream Mach.
        double betaTrans = 0.5;
        // Characteristic far-field lateral faces (default = leave the lateral
        // axes to the caller: periodic, walls, ...). openZ ignored at nz == 1.
        bool openY = false;
        bool openZ = false;
        double sigLat = 0.25; // lateral pressure-anchor strength
        // Thermal inflow: the entropy wave relaxes toward the density the
        // TARGET TEMPERATURE implies at the LOCAL pressure, rho -> p/(R TInf),
        // instead of toward rhoInf. The two agree at the fixed point and
        // differ off it: the density form lets T_in drift with the incoming
        // pressure field, which is wrong the moment transport depends on T or
        // a wall exchanges heat. false is the historical arithmetic, bit-for-
        // bit. TInf is in solver units (pInf / (rhoInf R)).
        bool thermalInflow = false;
        double TInf = 0.0;
        double Rgas = 1.0;
        // Potential-corrected far-field targets: relax/anchor toward
        // freestream + wake source (strength Q set via setWakeSource, from the
        // measured drag) + displacement doublet (radius aBody about
        // xc,yc,zc), with the Bernoulli pressure. Removes the finite-domain
        // bias of uniform-freestream targets (pinning u = uInf ~2.7D upstream
        // ignores the body's ~7% upstream influence at an 8D box). 2D
        // (nz == 1) source+doublet or 3D forms. With potCorrect == false, or
        // Q == 0 and aBody == 0, behaviour is bit-identical to the uniform
        // targets; uniform freestream then stays an exact fixed point.
        bool potCorrect = false;
        double xc = 0, yc = 0, zc = 0; // body centre (world coords)
        double aBody = 0;              // displacement radius (D/2)
        // Lateral entropy/shear relax rate where flow enters the domain. Keep
        // it gentle: a stiff lateral velocity clamp is a wall in disguise
        // (measured with the sponge far field: lateral u-relax cd 1.98 ~=
        // walls 1.96, rho/p-only 1.91).
        double etaLat = 0.25;
        // Full-width solid floor crossing the x planes (--body floor): rows
        // j <= floorRows are fully solid there -- the ghost wall owns their
        // state, so the characteristic overwrite is skipped and they keep
        // the interior rhs. -1 = no floor (bit-identical). Incompatible with
        // openY/openZ (validator-enforced upstream).
        int floorRows = -1;
    };

    // Periodic grid, nx >= 4 (ny/nz >= 4 on open axes); subsonic freestream.
    static std::expected<DeviceCompNscbc, hipError_t> make(const Grid& g, Config c);

    // Overwrite the rhs of every open-boundary cell with the characteristic
    // form, written into the stage output through the same RK combination the
    // interior used (dns/gpu/comp.hpp, StageCombine); install via
    // DeviceComp::setRhsHook (stage primitives are live).
    void apply(const DeviceComp& d, const DeviceCompState& s,
               DeviceCompState& out, const StageCombine& cmb) const;

    // Stretched-y (curvilinear) variant: the PointRemap travels as a template
    // argument of the kernel, so every one-sided normal derivative, Yoo-Im
    // transverse derivative and potential-target position uses PHYSICAL
    // distances, and the retained transverse flux divergence comes from
    // faceFluxCurv on d's metric tables -- the interior scheme, exactly.
    // Pre: d is the curvilinear DeviceComp built from the same remap's nodes.
    void apply(const DeviceComp& d, const DeviceCompState& s,
               DeviceCompState& out, const StageCombine& cmb,
               const TanhStretchY& remap) const;

    // Focal (per-axis interior clustering) variant: same contract, one more
    // kernel instantiation. AxisRemap3 is ONE concrete type whatever the
    // focal box is -- the coefficients travel as data, so x/y/z clustering
    // in any combination costs no extra instantiations.
    void apply(const DeviceComp& d, const DeviceCompState& s,
               DeviceCompState& out, const StageCombine& cmb,
               const AxisRemap3& remap) const;

    // Inflow target u (quiet-start ramp tracking); must stay subsonic.
    void setFreestream(double u) { u_ = u; }
    double freestream() const { return u_; }

    // Wake-source strength Q = F_drag / (rhoInf uInf) for the potential
    // correction; caller smooths (EMA) so the shedding oscillation does not
    // feed back. Ignored unless potCorrect. Pre: Q >= 0, finite.
    void setWakeSource(double Q) { Q_ = Q; }

private:
    DeviceCompNscbc(const Grid& g, const Config& c);

    GpuGrid g_;
    Config c_;
    double u_, cInf_;
    double Q_ = 0;
};

} // namespace dns::gpu
