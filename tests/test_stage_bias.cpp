// N16 (masterplan_combclose K4, docs/fvcomp.md "Precision"): the plain FP64 stage-3 write
// `ca a + cb b + cr r` (cb = R(2/3), ca = 1 - cb exactly) is BIASED in the order the GPU
// compiles it (kFvGather<double, false>, gfx1100: t = RN(cb b); x = fma(ca, a, t); out =
// fma(cr, r, x)): x rounds ca a + t on a lattice of spacing ulp/3 before the increment is
// added. Emulated here with std::fma in a TU built with -ffp-contract=off, against exact
// binary128 values, b = a + k ulp(a), a's mantissa uniform:
//   (1) control: plain form, mean error over mantissas [1, 1.5) >= +0.05 ulp (the bias exists;
//       [1.5, 2) printed)
//   (2) the increment form a + (cb (b - a) + cr r), unfused and with the fma: |mean| <= 0.01 ulp
//   (3) the compensated FP64 write (dns/gpu/word2d.hpp combineWord2, the shipped function):
//       |hi + lo - (a + al + d)| <= 4 u^2 |a| on every sample (d the computed increment: the
//       accumulation is exact to the lo word's rounding), and its mean total error vs exact
//       <= 1e-3 ulp
//   (4) combscale S3 (issues.md N20): the same write at FP32 (cb = R_f32(2/3) rounds UP): plain mean
//       on [1, 1.5) <= -0.05 ulp_f32 (the negative sign; [1.5, 2) printed), increment |mean| <= 0.01
#include "dns/gpu/word2d.hpp"

#include <cmath>
#include <cstdio>
#include <format>
#include <random>
#include <string>

namespace {

using Q = __float128;

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct Stats
{
    double plain = 0, incr = 0, incrFma = 0, comp = 0, accWorst = 0;
};

// N samples, a = m 2^e with the mantissa m uniform in [lo, hi), e uniform in [-6, 6].
Stats sample(double lo, double hi, int n, std::uint64_t seed)
{
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> mant(lo, hi), unit(-1.0, 1.0), dtD(1e-4, 1e-3);
    std::uniform_int_distribution<int> ex(-6, 6), kk(-(1 << 20), 1 << 20);
    const double cb = 2.0 / 3.0, ca = 1.0 - cb, u = std::ldexp(1.0, -53);
    Q sp = 0, si = 0, sf = 0, sc = 0;
    double acc = 0;
    for (int i = 0; i < n; ++i) {
        const double a = std::ldexp(mant(g), ex(g));
        const double ulp = std::nextafter(a, 2 * a) - a;
        const double b = a + double(kk(g)) * ulp;
        const double cr = dtD(g);
        const double rr = unit(g) * 16.0 * ulp / cr; // cr rr ~ U(-16, 16) ulp(a)
        const Q exact = Q(ca) * Q(a) + Q(cb) * Q(b) + Q(cr) * Q(rr);
        // (1) the compiled plain order.
        const double plain = std::fma(cr, rr, std::fma(ca, a, cb * b));
        sp += (Q(plain) - exact) / Q(ulp);
        // (2) the increment form.
        const double incr = a + (cb * (b - a) + cr * rr);
        const double incrF = a + std::fma(cb, b - a, cr * rr);
        si += (Q(incr) - exact) / Q(ulp);
        sf += (Q(incrF) - exact) / Q(ulp);
        // (3) the compensated write on a two-word state: lo words a fraction of ulp.
        const double al = 0.5 * unit(g) * ulp, bl = 0.5 * unit(g) * ulp; // |lo| <= ulp(hi) / 2
        const double inc = cr * rr;
        const dns::gpu::Word2d w = dns::gpu::internal::combineWord2(cb, a, al, b, bl, inc);
        const double d = cb * ((b - a) + (bl - al)) + inc; // the increment combineWord2 forms
        const Q got = Q(w.hi) + Q(w.lo);
        const double e = double(got - (Q(a) + Q(al) + Q(d)));
        acc = std::max(acc, std::abs(e) / (u * u * std::abs(a)));
        const Q exactC = (Q(1) - Q(cb)) * (Q(a) + Q(al)) + Q(cb) * (Q(b) + Q(bl)) + Q(inc);
        sc += (got - exactC) / Q(ulp);
    }
    return {.plain = double(sp / n), .incr = double(si / n), .incrFma = double(sf / n), .comp = double(sc / n), .accWorst = acc};
}

// combscale S3 (issues.md N20): the same write at FP32 (kFvGather<float, false>'s order), cb = R_f32(2/3)
// (rounded UP, where the double's rounds down), b = a + k ulp_f32 (|k| <= 2^8); exact in binary128.
Stats sampleF32(float lo, float hi, int n, std::uint64_t seed)
{
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<float> mant(lo, hi), unit(-1.0f, 1.0f), dtD(1e-4f, 1e-3f);
    std::uniform_int_distribution<int> ex(-6, 6), kk(-(1 << 8), 1 << 8);
    const float cb = 2.0f / 3.0f, ca = 1.0f - cb;
    Q sp = 0, si = 0, sf = 0;
    for (int i = 0; i < n; ++i) {
        const float a = std::ldexp(mant(g), ex(g));
        const float ulp = std::nextafter(a, 2 * a) - a;
        const float b = a + float(kk(g)) * ulp;
        const float cr = dtD(g);
        const float rr = unit(g) * 16.0f * ulp / cr;
        const Q exact = Q(ca) * Q(a) + Q(cb) * Q(b) + Q(cr) * Q(rr);
        const float plain = std::fma(cr, rr, std::fma(ca, a, cb * b));
        sp += (Q(plain) - exact) / Q(ulp);
        const float incr = a + (cb * (b - a) + cr * rr);
        const float incrF = a + std::fma(cb, b - a, cr * rr);
        si += (Q(incr) - exact) / Q(ulp);
        sf += (Q(incrF) - exact) / Q(ulp);
    }
    return {.plain = double(sp / n), .incr = double(si / n), .incrFma = double(sf / n)};
}

} // namespace

int main()
{
    constexpr int kN = 400000;
    std::printf("stage-3 write bias, cb = R(2/3), b = a + k ulp (|k| <= 2^20), cr r ~ U(-16, 16) ulp, %d samples per range\n", kN);
    const Stats lo = sample(1.0, 1.5, kN, 20261109), hi = sample(1.5, 2.0, kN, 20261110);
    std::printf("  mantissa [1, 1.5): plain %+.4f ulp, increment %+.4f / fma %+.4f, compensated %+.2e (accumulation %.2f u^2|a|)\n",
                lo.plain, lo.incr, lo.incrFma, lo.comp, lo.accWorst);
    std::printf("  mantissa [1.5, 2): plain %+.4f ulp, increment %+.4f / fma %+.4f, compensated %+.2e (accumulation %.2f u^2|a|)\n",
                hi.plain, hi.incr, hi.incrFma, hi.comp, hi.accWorst);
    check(lo.plain >= 0.05, std::format("control: the plain compiled order is biased on [1, 1.5): {:+.4f} >= +0.05 ulp", lo.plain));
    for (const Stats* s : {&lo, &hi}) {
        const char* r = s == &lo ? "[1, 1.5)" : "[1.5, 2)";
        check(std::abs(s->incr) <= 0.01 && std::abs(s->incrFma) <= 0.01,
              std::format("increment form unbiased on {}: {:+.4f} / {:+.4f} <= 0.01 ulp", r, s->incr, s->incrFma));
        check(s->accWorst <= 4.0, std::format("compensated FP64 accumulation on {}: {:.2f} <= 4 u^2 |a|", r, s->accWorst));
        check(std::abs(s->comp) <= 1e-3, std::format("compensated FP64 mean error on {}: {:+.2e} <= 1e-3 ulp", r, s->comp));
    }
    // (4) FP32 (combscale S3): the plain write is biased the OTHER way (R_f32(2/3) rounds up).
    const Stats flo = sampleF32(1.0f, 1.5f, kN, 20261111), fhi = sampleF32(1.5f, 2.0f, kN, 20261112);
    std::printf("  FP32 mantissa [1, 1.5): plain %+.4f ulp_f32, increment %+.4f / fma %+.4f\n", flo.plain, flo.incr, flo.incrFma);
    std::printf("  FP32 mantissa [1.5, 2): plain %+.4f ulp_f32, increment %+.4f / fma %+.4f\n", fhi.plain, fhi.incr, fhi.incrFma);
    check(flo.plain <= -0.05, std::format("FP32 plain compiled order biased NEGATIVE on [1, 1.5): {:+.4f} <= -0.05 ulp_f32", flo.plain));
    for (const Stats* s : {&flo, &fhi})
        check(std::abs(s->incr) <= 0.01 && std::abs(s->incrFma) <= 0.01,
              std::format("FP32 increment form unbiased on {}: {:+.4f} / {:+.4f} <= 0.01 ulp_f32", s == &flo ? "[1, 1.5)" : "[1.5, 2)",
                          s->incr, s->incrFma));
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
