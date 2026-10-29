// Aggregation AMG of the solver library (docs/la.md, masterplan_meshsolver
// P5): the host hierarchy (dns/la/amg.hpp: pairwise aggregates, coarse
// patterns, gather maps) once per mesh, the coarse values re-gathered on the
// device (coarse = P^T A P, P piecewise constant: each coarse entry a FIXED-
// ORDER FP64 sum of fine entries, no SpGEMM, no atomics) whenever the fine
// values change, per-level smoothers of gpu/la/smooth, a dense FP64 inverse
// on the coarsest level, and a V-, scaled-V or K-cycle whose inner Krylov
// coefficients stay ON the device (no host sync inside a cycle).
// B = 1 (pressure) and every block shape. Deterministic: byte-identical reruns.
// GPU-only header.
#pragma once

#include "dns/gpu/la/blockell.hpp"
#include "dns/gpu/la/smooth.hpp"
#include "dns/la/amg.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace dns::gpu::la {

enum class AmgSmoother : std::uint8_t { Jacobi, MultiColourGS, Lines };

struct AmgOptions
{
    // Level 0, and levels >= 1. Coarse levels are launch-bound (a GS sweep
    // is 2 launches per colour, aggregated graphs have many colours): Jacobi
    // there was measured faster to the same tolerance (docs/la.md).
    AmgSmoother smoother = AmgSmoother::MultiColourGS;
    AmgSmoother coarseSmoother = AmgSmoother::Jacobi;
    int sweeps = 1;       // pre = post (GS: symmetric sweeps; Jacobi / lines: steps)
    double omega = 0.67;  // Jacobi and line-Jacobi damping (GS: none)
    // Coarse-correction Krylov steps per coarse level: 0 V-cycle, 1 scaled V
    // (one line search: OpenFOAM GAMG's scaleCorrection), 2 K-cycle (Notay;
    // the outer PCG must then be flexible).
    int inner = 2;
    // Inner steps on coarse levels 1 and 2 always; a deeper level with fewer
    // rows than this takes inner 0: the K-cycle doubles the visits of every
    // level below it, and below ~1e4 rows a visit costs its launches, not its
    // work (measured, docs/la.md). The V-cycled tail stays a bounded depth.
    int innerMinRows = 10000;
    // The operator annihilates the constant (per component): pure Neumann /
    // periodic pressure. apply() then works on the mean-free part (r and z
    // projected, so a PCG iterate never drifts along the null space: without
    // it FP64 PCG broke down at relTol 1e-12, measured) and the coarsest
    // inverse is of A + alpha 1 1^T. Pre: b mean-free per component.
    bool constantNullSpace = false;
};

template <class T>
class Amg
{
public:
    // Pre: A on the pattern ellPattern(g) of the graph h was built from,
    // h.B == A.shape().B; A outlives the Amg and stays at its address.
    // Error: option out of range, a smoother the hierarchy lacks (MC-GS
    // without colours, lines without line sets), device allocation.
    static std::expected<Amg, std::string> make(const BlockEll<T>& A, const dns::la::AmgHierarchy& h,
                                                const AmgOptions& o);

    // Re-gathers every coarse level from A's current values, refactors the
    // smoothers and inverts the coarsest matrix (FP64). Error: a singular
    // block, line or coarsest matrix (constantNullSpace not set?).
    std::expected<void, std::string> refresh();

    // z = one cycle on A z = r from z = 0 (the preconditioner). Pre:
    // refresh() succeeded; r, z device vectors of A.size(), z != r.
    // Deterministic: byte-identical for identical inputs.
    void apply(const T* r, T* z);

    int levels() const { return int(lv_.size()); }
    // Level l's matrix (0: the caller's A). Pre: l < levels().
    const BlockEll<T>& matrix(int l) const;

    Amg(Amg&&) noexcept;
    Amg& operator=(Amg&&) noexcept;
    ~Amg();

    struct Level; // opaque (amg.hip)

private:
    Amg();
    void cycle(int l, const T* b, T* x);
    // The coarse correction on level k into x; true = scaled V (inner 1): x
    // NOT written, the caller prolongs coef[0] c1 (kProlongAddScaled).
    bool correct(int k, const T* b, T* x);
    void smooth(int l, const T* b, T* x);
    void removeMean(const T* x, T* y); // constantNullSpace: y = x - its mean
    AmgOptions o_;
    std::vector<std::unique_ptr<Level>> lv_;
};

extern template class Amg<float>;
extern template class Amg<double>;

} // namespace dns::gpu::la
