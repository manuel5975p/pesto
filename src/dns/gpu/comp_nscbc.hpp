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
#include "dns/gpu/nscbc_modal.hpp"
#include "dns/remap.hpp"

#include <memory>
#include <vector>

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
        // Angle-adaptive transverse coefficient. betaTrans is a CONSTANT, and
        // the constant that cancels the LODI angle error is not: for an
        // outgoing acoustic wave at incidence theta the exact incoming
        // amplitude is L1 = -i omega p cos t (1 - cos t) while the transverse
        // term carries Tp = i omega p sin^2 t, so the coefficient that
        // reproduces it is beta_opt = cos t / (1 + cos t) -- and 0.5 is only
        // its NORMAL-INCIDENCE limit. The face can recover its own theta with
        // no extra state, because the same two quantities give it away:
        //     Tp / Lout = (1 - cos t) / cos t
        // with Lout the outgoing acoustic amplitude (L5 at a hi face, L1 at a
        // lo one). Writing X = Tp/Lout, beta_opt = betaTrans * 2/(2 + X), and
        // substituting reproduces L1 EXACTLY at every angle rather than
        // through O(t^2) -- the Engquist-Majda symbol summed, not truncated,
        // and without the auxiliary time history an exact ABC normally costs.
        // X is clamped to [0, 18] (beta never exceeds betaTrans, so this is
        // never more aggressive than the constant form, and grazing incidence
        // out to ~87 deg is covered). X == 0 gives betaTrans back exactly, so
        // a normal-incidence face is bit-for-bit unchanged.
        bool betaAuto = false;
        // Same coefficient, formed per transverse DCT-II MODE of the outflow
        // plane instead of per cell (gpu/nscbc_modal.hpp). A per-cell ratio
        // averages a superposition of angles; a modal one separates it, because
        // the free-slip laterals make the cosine basis the exact transverse
        // eigenbasis. Costs one plane and two transforms per stage, applies to
        // the x-hi face ONLY, and is mutually exclusive with betaAuto.
        // Pre: !openY, !openZ, floorRows < 0, uniform mesh.
        bool betaModal = false;
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
        // Lateral entropy/shear relax rate where flow enters the domain.
        // GENTLE ON MEASUREMENT, twice over: a stiff lateral velocity clamp
        // is a wall in disguise (sponge-era: lateral u-relax cd 1.98 ~= walls
        // 1.96, rho/p-only 1.91), and the relax itself writes vorticity onto
        // the face in proportion to this rate under acoustic load
        // (0.20/0.80/2.16 of incident scale at 0.05/0.25/1.0 -- the 45-deg
        // ridge, gpu_nscbc_order 2026-07-30). 0.05 is the scan's optimum and
        // free on the transparency and cd rigs.
        double etaLat = 0.05;
        // Inflow lambda = u family rate (entropy L2, shear L3/L4). etaIn is
        // sized against freestream sag -- an L5 argument -- and clamping the
        // OTHER amplitudes at 100 makes the inflow a mirror for vorticity
        // and entropy too. < 0 = follow etaIn (same value, same arithmetic,
        // bit-identical).
        double etaInShear = -1;
        // There is deliberately no outflow backflow knob: an etaOut relax
        // blend on the x-hi L2/L3/L4 was prototyped and measured inert even
        // under 4x-freestream backflow (comp_nscbc.hip has the numbers).
        // Full-width solid floor crossing the x planes (--body floor): rows
        // j <= floorRows are fully solid there -- the ghost wall owns their
        // state, so the characteristic overwrite is skipped and they keep
        // the interior rhs. -1 = no floor (bit-identical). Incompatible with
        // openY/openZ (validator-enforced upstream).
        int floorRows = -1;
        // Per-row x-lo inflow targets (dns flame --mode bunsen), one entry per
        // (j, k) row at j + ny k: the u target inflowU + pulse * inflowJet
        // (setInflowPulse), and the entropy target inflowRho -- the density
        // itself, or with thermalInflow 1/(R T_row) times the LOCAL pressure.
        // Empty = the uniform uInf/rhoInf (or TInf) targets, in a separate
        // kernel instantiation: bit for bit. Uniform mesh, !potCorrect only.
        std::vector<double> inflowU = {}, inflowJet = {}, inflowRho = {};
        // Gravity along -x (dns flame --mode candle; the source itself is
        // addGravity, added to every cell AFTER this overwrite): the AMBIENT
        // column's hydrostatic dp/dx at the x-lo and x-hi rows (<= 0). The
        // incoming acoustic amplitude carries its hydrostatic value (source-
        // term LODI, Sutherland & Kennedy 2003): L5 += (u + c) hydroLo at
        // the inflow, L1 += (u - c) hydroHi at the outflow, so LODI + gravity
        // is exactly at rest in the ambient column. Without it both rows
        // accelerate at g/2 there (LODI hands half the hydrostatic gradient
        // to each wave). AMBIENT, not local, rho g: a plume crossing the
        // outflow sits at the ambient pressure, and a local-rho term would
        // offset its p by L (rho_a - rho) g / sigma. Both 0 = the shipped
        // kernel. Per-row targets, uniform mesh, !openY/!openZ only.
        double hydroLo = 0, hydroHi = 0;
    };

    // Periodic grid, nx >= 4 (ny/nz >= 4 on open axes); subsonic freestream.
    static std::expected<DeviceCompNscbc, hipError_t> make(const Grid& g, Config c);

    // Focal variant of make: also tabulates `remap`'s physical cell centres
    // (indices -1..n per axis, filled on the device by the same map), so the
    // AxisRemap3 apply below reads them instead of evaluating the FP64 map 14
    // times per boundary cell. NOT bit-identical (verify_propperf: ulp drift).
    // Pre: apply(..., AxisRemap3) is then always passed this same remap.
    static std::expected<DeviceCompNscbc, hipError_t>
    make(const Grid& g, Config c, const AxisRemap3& remap);

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

    // Jet modulation of a per-row inflow: the u target becomes inflowU +
    // du * inflowJet. 0 = the profile itself. Ignored without a profile.
    void setInflowPulse(double du) { pulse_ = du; }

    // Wake-source strength Q = F_drag / (rhoInf uInf) for the potential
    // correction; caller smooths (EMA) so the shedding oscillation does not
    // feed back. Ignored unless potCorrect. Pre: Q >= 0, finite.
    void setWakeSource(double Q) { Q_ = Q; }

private:
    DeviceCompNscbc(const Grid& g, const Config& c,
                    std::unique_ptr<NscbcModalAbc> abc);

    GpuGrid g_;
    Config c_;
    double u_, cInf_;
    double Q_ = 0;
    double pulse_ = 0;
    // The per-row inflow targets on the device (Config::inflowU/Jet/Rho);
    // empty = uniform targets.
    DeviceScalar inU_, inJ_, inR_;
    // Scratch, rewritten every stage: held by pointer so apply() stays const
    // (it is an rhs producer, not a state owner) and the class stays movable.
    std::unique_ptr<NscbcModalAbc> abc_;
    // Focal centre table (make with an AxisRemap3), x | y | z, n + 2 each;
    // empty = the AxisRemap3 apply evaluates the map itself.
    DeviceBuffer<double> ctab_;
};

} // namespace dns::gpu
