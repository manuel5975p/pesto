// The stage writes every StageCombine consumer goes through: the fused stage
// kernels, the NSCBC overwrite, the additive hooks and the RANS scalars.
// HIP TUs only.
//
// Plain vs compensated (dns/gpu/comp.hpp) is a COMPILE-TIME choice: each such
// kernel is instantiated for both and the launch picks one (withCompensation).
// A runtime branch at the write splits the kernel's last basic block, and the
// backend then fuses the rhs arithmetic into FMAs differently -- measured, it
// moved the NSCBC rows off the pre-compensation bits. The plain instantiation
// is the pre-compensation expression in the pre-compensation block, so it is
// bit-for-bit by construction.
#pragma once

#include "dns/gpu/comp.hpp"

#include <type_traits>

namespace dns::gpu::internal {

// f(std::true_type{}) when k carries low words, f(std::false_type{}) else.
template <class F> void withCompensation(const StageCombine& k, F&& f)
{
    if (k.ol[0]) f(std::true_type{});
    else f(std::false_type{});
}

// (ah + al) + cb ((bh - ah) + (bl - al)) + inc: the increment form of
// ca a + cb b + inc, equal because ca == 1 - cb exactly. Near a steady state
// bh - ah is exact (Sterbenz); far from it the increment's own FP32 rounding
// is a time-accuracy error, not an accumulation one.
__device__ inline Word2 combineWord2(Real cb, Real ah, Real al, Real bh, Real bl,
                                     Real inc)
{
    return addWord2(ah, al, cb * ((bh - ah) + (bl - al)) + inc);
}

// out[c] = ca a + cb b + cr r for mean-flow variable v (rho mx my mz E).
// Pre (Comp): k.ol[v] set.
template <bool Comp>
__device__ inline void stagePut(const StageCombine& k, int v, std::size_t c,
                                Real cr, Real r, Real* out)
{
    if constexpr (!Comp) {
        out[c] = k.ca * k.a[v][c] + k.cb * k.b[v][c] + cr * r;
    } else {
        const Word2 w = combineWord2(k.cb, k.a[v][c], k.al[v][c], k.b[v][c],
                                     k.bl[v][c], cr * r);
        out[c] = w.hi;
        k.ol[v][c] = w.lo;
    }
}

// out[c] += cr x on top of an already-written stage value (additive hook).
// Pre (Comp): k.ol[v] set.
template <bool Comp>
__device__ inline void stageAdd(const StageCombine& k, int v, std::size_t c,
                                Real cr, Real x, Real* out)
{
    if constexpr (!Comp) {
        out[c] += cr * x;
    } else {
        const Word2 w = addWord2(out[c], k.ol[v][c], cr * x);
        out[c] = w.hi;
        k.ol[v][c] = w.lo;
    }
}

// Compensated turbulence scalar i: a + cb (b - a) + inc. Pre: k.otl[i].
__device__ inline Word2 stageCombineT(const StageCombine& k, int i,
                                      std::size_t c, Real inc)
{
    return combineWord2(k.cb, k.ta[i][c], k.tal[i][c], k.tb[i][c], k.tbl[i][c],
                        inc);
}

} // namespace dns::gpu::internal
