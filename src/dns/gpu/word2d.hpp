// The compensated FP64 state's arithmetic (masterplan_combclose R6, docs/fvcomp.md "Precision"):
// the double-double twins of comp.hpp's Word2 / twoSum / addWord2 and stage_write.hpp's
// combineWord2. Overloads, never templates: the float functions stay byte-verbatim, and an
// all-double call resolves here exactly. Host-includable (DNS_HD): test_stage_bias reads it.
#pragma once

#include "dns/hd.hpp"

namespace dns::gpu {

// An unevaluated sum hi + lo of two doubles (the compensated FP64 state's words).
struct Word2d
{
    double hi, lo;
};

// Knuth's TwoSum in FP64: s.hi + s.lo == a + b EXACTLY, any magnitudes. No products.
DNS_HD inline Word2d twoSum(double a, double b)
{
    const double s = a + b;
    const double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

// (hi + lo) + d renormalised: the hi sum error-free, the leftover rounded once at the lo level.
// Error vs hi + lo + d: <= u |s.lo + lo| <= ~2 u^2 |hi|.
DNS_HD inline Word2d addWord2(double hi, double lo, double d)
{
    const Word2d s = twoSum(hi, d);
    return twoSum(s.hi, s.lo + lo);
}

// The word type of a state precision: Word2 (float, comp.hpp), Word2d (double). A kernel
// declares its words through it, so a missing double overload is a compile error, never a
// silent float narrowing.
template <class R> struct WordOf;
template <> struct WordOf<double>
{
    using type = Word2d;
};
template <class R> using Word2Of = typename WordOf<R>::type;

} // namespace dns::gpu

namespace dns::gpu::internal {

// (ah + al) + cb ((bh - ah) + (bl - al)) + inc in FP64: the increment form of ca a + cb b + inc
// (ca == 1 - cb exactly). The plain FP64 stage-3 write rounds ca a + RN(cb b) on a lattice of
// spacing ulp/3 before inc is added: +0.11 ulp per write on mantissas [1, 1.5) (N16).
DNS_HD inline Word2d combineWord2(double cb, double ah, double al, double bh, double bl, double inc)
{
    return addWord2(ah, al, cb * ((bh - ah) + (bl - al)) + inc);
}

} // namespace dns::gpu::internal
