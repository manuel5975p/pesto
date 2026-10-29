// Krylov solvers of the solver library (docs/la.md): right-preconditioned
// flexible GMRES (CGS2), PCG and right-preconditioned BiCGStab on vectors of T,
// with every inner product a BATCHED FP64 dot: one launch of fixed-order block
// partials, one fixed-order reduce, one host copy per batch -- deterministic
// (byte-identical reruns), wave-size agnostic (shared-memory trees, no
// shuffles). Operators are callables; the matrix never reaches this layer.
// GPU-only header.
#pragma once

#include "dns/gpu/device_buffer.hpp"

#include <cstddef>
#include <array>
#include <expected>
#include <functional>
#include <span>
#include <string>

namespace dns::gpu::la {

// y = Op(x). Pre: x, y device vectors of the solver's length, not aliased.
template <class T>
using LinOp = std::function<void(const T* x, T* y)>;

// Stop when |r| <= max(relTol |r0|, absTol) (OpenFOAM's relTol / tolerance
// split, both on the 2-norm), or after maxIter iterations.
struct KrylovOptions
{
    int maxIter = 500;
    int restart = 30;  // FGMRES cycle length, <= Krylov::kMaxRestart
    double relTol = 1e-6, absTol = 0;
    // PCG only: flexible CG (Polak-Ribiere beta = (r', z' - z) / (r, z)) for
    // a VARIABLE preconditioner (the AMG K-cycle); false = classic beta.
    bool flexible = false;
    // PCG and BiCGStab: > 0 replaces the 2-norm by OpenFOAM's residual
    // measure sum|r_i| / normFactor (lduMatrix::solver::normFactor, computed
    // by the caller from the initial guess) in r0, residual and the stop;
    // absTol is then OpenFOAM's `tolerance`. 0 = the 2-norm, byte for byte.
    double normFactor = 0;
    // PCG and BiCGStab: iterate at least this often (OpenFOAM minIter).
    int minIter = 0;
    // PCG with minIter 0: read r0 BEFORE the first M apply (an already-
    // converged solve then costs no M, one extra sync otherwise); false: M
    // first, r0 and (r, z) in one batch. The same results either way -- a
    // caller's hint (e.g. "the last solve took 0 iterations").
    bool r0First = true;
};

// One solve's account. relResidual is the solver's RECURSIVE estimate
// |r_k| / |r0| (FGMRES: |g_{k+1}|; PCG/BiCGStab: the updated r).
struct KrylovResult
{
    int iterations = 0;
    int restarts = 0; // BiCGStab shadow-residual restarts (Lanczos breakdowns)
    double r0 = 0, residual = 0, relResidual = 0;
    bool converged = false, breakdown = false;
    // Krylov::preFlag() was nonzero at the solve's first readback: the solve
    // stopped there, x untouched (the caller's deferred failure, e.g. a factor).
    bool flagged = false;
};

// Optional device-side start of pcg / bicgstab, so the caller's own setup
// costs no host sync: Ax = A x (the solver skips its own A x0, bitwise the
// same product), normTerms = OpenFOAM's per-row normFactor terms (>= 0): the
// solver then uses normFactor = sum normTerms + 1e-20 (the L1 measure),
// reduced in its FIRST dot batch, in place of KrylovOptions::normFactor.
template <class T>
struct KrylovStart
{
    const T* Ax = nullptr;
    const T* normTerms = nullptr;
};

// Max inner products per batch (CGS2 over a full FGMRES basis, plus one).
inline constexpr int kMaxDots = 66;

template <class T>
struct DotPair
{
    const T* a;
    const T* b;
};

template <class T>
class Krylov
{
public:
    static constexpr int kMaxRestart = kMaxDots - 2;
    static constexpr int kMaxBatch = 3; // systems per bicgstabBatch

    // Workspace for vectors of length N; restart > 0 also allocates the
    // FGMRES basis (2 restart + 1 vectors); batch > 1 sizes the work vectors
    // for bicgstabBatch. Error: restart or batch out of range, device allocation.
    static std::expected<Krylov, std::string> make(std::size_t N, int restart, int batch = 1);

    // Each solves A x = b from the x given. M empty = identity.
    // Error: a device failure or a non-finite residual (x then unspecified).
    std::expected<KrylovResult, std::string> fgmres(const LinOp<T>& A, const LinOp<T>& M, const T* b, T* x,
                                                    const KrylovOptions& o);
    // Pre: A symmetric positive definite, M symmetric positive definite.
    // With minIter == 0 and r0First, r0 is tested BEFORE the first M apply:
    // an already-converged solve costs no M (same results, x untouched).
    // Host syncs: 1 per iteration (alpha stays on the device), start 1-2.
    std::expected<KrylovResult, std::string> pcg(const LinOp<T>& A, const LinOp<T>& M, const T* b, T* x,
                                                 const KrylovOptions& o, const KrylovStart<T>& start = {});
    // Host syncs: 2 per iteration (alpha stays on the device), start 1.
    std::expected<KrylovResult, std::string> bicgstab(const LinOp<T>& A, const LinOp<T>& M, const T* b, T* x,
                                                      const KrylovOptions& o, const KrylovStart<T>& start = {});

    // nb <= kMaxBatch independent systems (system j: offset j N of b, x and of
    // every operator vector) solved exactly as nb bicgstab calls would, bit for
    // bit, in LOCKSTEP: A and M are applied once per step to all nb systems,
    // every dot batch carries each active system's pairs, each system keeps its
    // own scalars, branches and stop. mask: the systems taken (bit j; the others
    // are never touched or reported). Per system: its KrylovResult (flagged:
    // preFlag(j) was set) or its error text (bicgstab's). Host syncs per step:
    // 2, for all systems together. Pre: make(N, ., >= nb); A, M on nb N vectors;
    // start vectors (if given) of nb N.
    using BatchResult = std::array<std::expected<KrylovResult, std::string>, kMaxBatch>;
    std::expected<BatchResult, std::string> bicgstabBatch(int nb, unsigned mask, const LinOp<T>& A, const LinOp<T>& M,
                                                          const T* b, T* x, const KrylovOptions& o,
                                                          const KrylovStart<T>& start = {});

    // A device int a caller's kernel may set nonzero (a deferred failure, e.g.
    // a smoother's factor flag) before the next pcg / bicgstab (system j of the
    // next bicgstabBatch): that solve reads it with its first dot batch (no
    // extra sync), stops with KrylovResult::flagged, and clears it. Stays 0
    // otherwise. Pre: j < kMaxBatch.
    int* preFlag(int j = 0);

    // out[k] = sum_i a_k[i] b_k[i] in FP64, all pairs in one batch
    // (b_k == null: sum_i |a_k[i]|, the L1 sum).
    // Pre: 1 <= pairs.size() <= kMaxDots, out.size() >= pairs.size(); vectors of length N.
    std::expected<void, std::string> dots(std::span<const DotPair<T>> pairs, std::span<double> out);

    std::size_t length() const { return N_; }

private:
    Krylov() = default;
    T* vec(int i) { return aux_.data() + std::size_t(i) * N_; }
    // Launches the batch into out_[0, K) and copies out_ WHOLE (batch, the
    // device scalar slot, the pre-flag) to host_ in one transfer.
    std::expected<void, std::string> batch(std::span<const DotPair<T>> pairs);
    // pre-flag j set at this readback: clears it, true.
    std::expected<bool, std::string> takePreFlag(int j = 0);
    std::size_t N_ = 0;
    int restart_ = 0, parts_ = 1, batch_ = 1;
    DeviceBuffer<T> aux_, basis_; // 8 work vectors (x batch); V (restart + 1), Z (restart)
    // out_: [0, kMaxDots) batch results, [kMaxDots + j] system j's device-
    // computed dot (read with the next batch), then kMaxBatch pre-flag ints
    // (zeroed at make). coef_: [alpha, -alpha, bad] per system.
    DeviceBuffer<double> partial_, out_, coef_;
    double host_[kMaxDots + kMaxBatch + 2] = {};
};

extern template class Krylov<float>;
extern template class Krylov<double>;

} // namespace dns::gpu::la
