// The flow Newton-Krylov pseudo-time solver (docs/rans.md, "Newton-Krylov
// pseudo-time"; masterplan_flowrans decisions 1, 14-16): per Newton step
//   (diag(1/dtau_c) - dR/dq) dq = R(q),   dtau_c = CFL dt_c,
// solved matrix-free (JFNK: J v by a forward difference of the GPU residual
// itself, R = DeviceComp::residualInto of the hooked FP32 rounding of an FP64
// iterate) by right-preconditioned GMRES(40), CGS2 with FP64 masked dot
// products. The preconditioner P = diag(1/dtau) - J1, J1 the derivative of a
// first-order Rusanov + compact-viscous flux (dns/nk_precond.hpp, FP64 duals)
// with the wall/wedge ghosts folded in, applied by symmetric LINE Gauss-Seidel
// along the wall-normal j lines in fixed tangential order (lexicographic in
// i). Mean flow and the
// turbulence scalars are block-diagonal in P (5x5 + 1x1 per scalar).
//
// The residual is ALWAYS the central KG stage: dissipation lives only in P,
// so the fixed point is the explicit march's. GPU-only header.
#pragma once

#include "dns/gpu/comp.hpp"

#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <vector>

namespace dns::gpu {

// The unknowns: lines along j (rows 1..ny-2) for every real (i, k).
struct NkLayout
{
    int iLo = 0, iHi = 0;   // real columns (lines), inclusive
    int kLo = 0, kHi = 0;   // real planes, inclusive
    bool cyclicI = false;   // the extruded O-grid's i wraps (lagged in GS)
    bool skipK = false;     // nz == 1: the k faces cancel exactly
    bool wedge = false;     // the k neighbours are the wedge's ghost planes
    double dphi = 0;        // wedge angle (rotation of the folded planes)
};

// Freestream scales of the scaled unknowns x = D^-1 dq (decision 1, 4.7).
struct NkScales
{
    double rho = 1, mom = 1, E = 1;
    double t[2] = {1, 1};   // SA: rho max(nu~inf, nu); SST k: rho U^2
    double omegaInf = 0;    // SST omega: per cell max(rho omega, rho omegaInf)
};

// One linear solve's account.
struct NkLinear
{
    int krylov = 0;          // Krylov iterations (all cycles)
    double relResidual = 0;  // final |b - A x| / |b| (GMRES estimate)
    bool converged = false;  // reached the forcing within the cap
};

class DeviceNk
{
public:
    // Applies the boundary rows (and axis ghosts) to a state.
    using Hook = std::function<void(DeviceCompState&)>;

    struct Config
    {
        NkLayout lay;
        NkScales sc;
        RansModel model = RansModel::None;
        double gamma = 1.4, Rgas = 1.0, prt = 0.9;
        int restart = 40, maxKrylov = 200;
        double forcing = 0.1, fdStep = 0x1p-14;
    };

    // Pre: dev curvilinear with enableRans or enableSteady (localDt), its
    // nTurb() == ransScalars(model); mask the real fluid cells; hook the
    // runner's boundary hook.
    static std::expected<DeviceNk, hipError_t>
    make(DeviceComp& dev, const std::vector<std::uint8_t>& mask,
         const Config& c, Hook hook);

    // The FP64 iterate from a state (hi + lo of a compensated one).
    hipError_t load(const DeviceCompState& s);
    // s = the FP32 rounding of the iterate on the real cells, then the hook.
    // Pre: s plain, nT matching.
    void store(DeviceCompState& s) const;

    // R(s) of the hooked state s into the internal residual; returns the
    // scaled RMS norm over the real cells (FP64). Pre: s == store()'d.
    double residual(const DeviceCompState& s, ReduceScratch& r);

    // One linear solve at the current iterate, cfl the pseudo-time CFL:
    // assembles P (frozen at s), factors its lines, and solves for the
    // update with GMRES. Pre: residual(s) ran at this iterate. An error
    // (exit 7) on a singular or non-finite line pivot.
    std::expected<NkLinear, std::string> solve(DeviceCompState& s, double cfl,
                                               ReduceScratch& r);

    // The largest alpha in (0, 1] keeping rho and E within 20% and SST rho
    // omega above -50%, rho k above -90% of the iterate (whole-step scaling,
    // never per cell). Pre: solve() ran.
    double stepLength(ReduceScratch& r);
    // The trial iterate q + alpha dq into s (rounded + hooked); the iterate
    // itself is unchanged until accept(). Pre: solve() ran.
    void trial(double alpha, DeviceCompState& s);
    // q = the last trial.
    void accept();

    // --- test access (gpu_nk) ---------------------------------------------
    // J v by the forward difference with step eps (relative to rms(v)) or the
    // central one, scaled: D^-1 J D v. Pre: residual(s) ran; v F x n FP32.
    void jacVec(const DeviceBuffer<Real>& v, bool central, double eps,
                DeviceBuffer<Real>& out, ReduceScratch& r);
    // P^-1 r by the line SGS (scaled). Pre: solve() or assemble() ran.
    void precond(const DeviceBuffer<Real>& r, DeviceBuffer<Real>& z);
    // P at s for cfl: its blocks (7 per cell: diag, i-, i+, j-, j+, k-, k+;
    // 25 mean + nT scalar entries each, scaled), and the line factors.
    std::expected<void, std::string> assemble(DeviceCompState& s, double cfl,
                                              ReduceScratch& r);
    const DeviceBuffer<Real>& meanBlocks() const { return Am_; }
    const DeviceBuffer<Real>& scalarBlocks() const { return At_; }
    const DeviceBuffer<double>& scales() const { return ds_; }
    int fields() const { return F_; }
    std::size_t cells() const { return n_; }

private:
    DeviceNk() = default;
    double dot(const Real* a, const Real* b, ReduceScratch& r) const;
    void matvec(const Real* x, Real* y, ReduceScratch& r);

    DeviceComp* dev_ = nullptr;
    GpuGrid g_{};
    Config c_{};
    Hook hook_;
    int F_ = 5, nT_ = 0;
    std::size_t n_ = 0;
    double cfl_ = 1;
    double count_ = 1;                       // real cells
    DeviceBuffer<std::uint8_t> mask_;
    DeviceBuffer<double> q64_, qt64_, ds_, dtau_;
    DeviceCompState R0_, Rp_, Rm_, sp_;      // residuals, perturbed state
    DeviceBuffer<Real> V_, w_, z_, x_;       // Krylov basis, work vectors
    DeviceBuffer<Real> Am_, At_;             // P blocks (scaled)
    DeviceBuffer<Real> Lm_, Fm_, Gm_, Lt_, Ft_, Gt_; // line factors
    DeviceBuffer<int> err_;                  // a singular pivot was met
    DeviceBuffer<Real> rhs_;                 // line-sweep scratch
};

} // namespace dns::gpu
