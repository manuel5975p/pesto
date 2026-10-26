// Policy of the implicit steady RANS march (--steady-scheme implicit,
// docs/rans.md "Implicit pseudo-time"): the CFL ramp, the step-length
// limiter, the rejection rule and when the integral stop may listen. No
// physics lives here; the line operator is dns/channel_line.
//
// Every constant is defined once, here, with the bracket it was measured in
// (Re_tau 180-1e4, ny 64-512, M 0.05-0.5, reviewer reruns of the FP64
// prototype; masterplan_implicit section 6).
#pragma once

#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns {

class ChannelLine; // dns/channel_line.hpp

// Starting CFL. Prototype (forward-FD Jacobian, FP32-emulated residual):
// Re_tau 1e4 ny 256 stops SA/SST at 29/29 iterations from 10, 25/26 from
// 100, 22/23 from 1000; from 1e8 (pure Newton) SA stops in 8 and SST stalls
// damped at ub -24.7%. With the central-FD Jacobian the host line certifies
// from 1e8 in 7-11 iterations in all 36 cases measured (FP64 residual); the
// ramp is kept as the plan's guard for the FP32 GPU residual.
inline constexpr double kImplicitCfl0 = 100;
// Growth per accepted undamped step: x1.5 needed 23-28 iterations to 1e-4
// against 15-18 at x2; x3 from 1e4 diverged SST.
inline constexpr double kImplicitCflGrowth = 2;
// The cap. With the exact Jacobian the stop certifies at 25-26 iterations,
// errors <= 6e-8; a 1e5 cap leaves a slow tail (SA Re_tau 1e4 stopped at
// +5.9e-5 / +7.7e-5, outside the 1e-5 target).
inline constexpr double kImplicitCflMax = 1e8;
// Below this after rejections the march has collapsed (exit 7). Rejection
// never fired in 24 measured cases; this is CFL0 / 100.
inline constexpr double kImplicitCflMin = 1;
inline constexpr double kImplicitRejectFactor = 0.1;
// Step-length bounds, fractions of the current conserved value: rho and E
// within 20%, SST rho omega down at most 50%, rho k down at most 90%. They
// fired only at iteration 0 on SST (alpha 0.952 at Re_tau 180, 0.873 at 1e4
// ny 256), never later.
inline constexpr double kImplicitStepRho = 0.2, kImplicitStepE = 0.2,
                        kImplicitStepOmega = 0.5, kImplicitStepK = 0.9;
// --steady-steps default under implicit: measured stops 25-26 (29 from CFL0
// 10), so ~20x margin. The explicit default (1000000) is unchanged.
inline constexpr long kImplicitSteadySteps = 500;
// CENTRAL-FD step of the line Jacobian, relative to each unknown's scale
// (dns/channel_line). Central, not forward: a forward step was off by up to
// 1.6e-5 relative per entry near the centreline (Omega = |du/dy| -> 0 is a
// curvature spike), and one-sided across the upwind kink max(mdot, 0) at
// v = 0; with it, pure Newton from CFL 1e8 stalled damped on SST Re_tau 1e4
// ny 256 (ub +19.9% after 100 iterations), where the central Jacobian
// certifies in 9 (36 host cases from CFL 1e8: 7-11 iterations). Twice the
// line evaluations: 0.70 ms per Jacobian + solve at ny 256 SST.
inline constexpr double kImplicitFdStep = 1e-7;

// The flow Newton-Krylov march (--steady-scheme implicit on dns flow,
// docs/rans.md "Newton-Krylov pseudo-time"), the one home of its constants.
// Bracket: the analyst-2 2D prototype (cylinder O-grid, Re 40, M 0.2, KG
// central residual, GMRES + lexicographic line-SGS on a first-order Rusanov
// block P): from CFL 10, x2 per undamped step, cap 1e8, it reached cd 1e-4 in
// 8 Newton / 532 Krylov iterations; forcing 1e-2 stalled at the Krylov cap,
// 0.1 converged (masterplan_flowrans decision 14).
inline constexpr double kNkCfl0 = 10, kNkCflGrowth = 2, kNkCflMax = 1e8,
                        kNkCflMin = 0.1;
// A trial whose scaled residual exceeds this multiple of the current one is
// rejected (restore, CFL x0.1).
inline constexpr double kNkRejectGrowth = 3;
// GMRES: restart length, Krylov cap per Newton step, constant forcing.
inline constexpr int kNkRestart = 40, kNkMaxKrylov = 200;
inline constexpr double kNkForcing = 0.1;
// At the CFL cap a GMRES that ran to kNkMaxKrylov without its forcing is at
// the FP32 residual floor (the FD J v is noise there): the NEXT armed step
// gets one restart cycle only. Measured (sphere wedge Re 1e4 SA L2 from the
// explicit plateau, gfx1100): every capped step ran the full 200 (1.5 s) for
// the same residual band 1.3-2.3e-3 the 40-cycle steps hold.
inline constexpr int kNkStallKrylov = kNkRestart;
// The JFNK differencing step, relative to rms of the scaled direction. The
// plan's 2^-10 failed gate N1 on SA (forward vs central J v of the nu~ field
// 2.3e-2 mid-transient, 7.2e-2 near the plateau, against 1e-2; the mean flow
// <= 4.6e-3); the N1 sweep's 2^-14 passes both (1.1e-3, 5.7e-3; mean flow
// <= 2.9e-3), 2^-12 only the first (3.0e-3, 2.8e-2). 16 x 12 cylinder, Re 40
// (gpu_nk, gfx1100).
inline constexpr double kNkFdStep = 0x1p-14;

// The CFL ramp's constants: the channel's line march (the default) or the
// flow Newton-Krylov march's.
struct PseudoTimePolicy
{
    double cfl0 = kImplicitCfl0, growth = kImplicitCflGrowth,
           cflMax = kImplicitCflMax, cflMin = kImplicitCflMin,
           rejectFactor = kImplicitRejectFactor;
};
inline constexpr PseudoTimePolicy kNkPolicy{kNkCfl0, kNkCflGrowth, kNkCflMax,
                                            kNkCflMin, kImplicitRejectFactor};

// The CFL ramp and its arming. accepted() after an undamped step grows the
// CFL x2 up to the cap; a damped one holds it. rejected() cuts it x0.1.
// armed(): the step that produced the current state was undamped AND taken AT
// the cap -- only then may the integral stop listen (unarmed it certified ub
// 1.1% off after rejections, and certified a damped stall).
class PseudoTime
{
public:
    // cfl0 other than kImplicitCfl0: tests only (the pure-Newton stall).
    // Pre: kImplicitCflMin <= cfl0 <= kImplicitCflMax.
    explicit PseudoTime(double cfl0 = kImplicitCfl0) : cfl_(cfl0) {}
    // Another march's constants (kNkPolicy). Pre: cflMin <= cfl0 <= cflMax.
    explicit PseudoTime(const PseudoTimePolicy& p) : p_(p), cfl_(p.cfl0) {}

    double cfl() const { return cfl_; }
    // Pre: 0 < alpha <= 1.
    void accepted(double alpha);
    // Disarms and cuts the CFL; false once it is below the policy's minimum.
    bool rejected();
    bool armed() const { return armed_; }

private:
    PseudoTimePolicy p_{};
    double cfl_ = kImplicitCfl0;
    bool armed_ = false;
};

// The largest alpha in (0, 1] keeping q + alpha dq inside the step bounds
// above: the whole update is scaled, never clipped per cell (a clip changes
// the Newton DIRECTION, not its length). SA's nu~ has no bound (negative nu~
// is SA-neg's legal branch). dq: fluid rows in the line's unknown layout.
// Pre: q wall-consistent with positive rho, E (SST: rho omega > 0).
double stepLength(std::span<const double> q, std::span<const double> dq,
                  const ChannelLine& line);

// One backward-Euler pseudo-time update of the line about q: solves
// (diag(1/dtau_j) - J(q)) dq = R over the fluid rows, J the line's FP64
// Jacobian, dtau_j at `cfl`. R is the caller's residual (the GPU's, or the
// line's own in host tests): only R decides the fixed point.
// Pre: R has the line's field layout (rows x fields), q wall-consistent.
// Post: dq in the unknown layout. Error on a singular or non-finite pivot.
std::expected<std::vector<double>, std::string>
pseudoUpdate(const ChannelLine& line, std::span<const double> q,
             std::span<const double> R, double cfl);

} // namespace dns
