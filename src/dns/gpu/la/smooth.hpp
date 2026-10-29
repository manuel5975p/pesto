// Smoothers / preconditioners of the solver library (docs/la.md): block
// Jacobi, multicolour block Gauss-Seidel, and batched block-tridiagonal line
// solves. Each holds a pointer to its matrix (which must outlive it) and its
// own factors; factor() refreshes them after the matrix values change.
// Factors are computed in FP64 per thread and stored in T. Deterministic: one
// thread per row (or line), fixed order, no atomics. GPU-only header.
#pragma once

#include "dns/gpu/la/blockell.hpp"
#include "dns/la/lines.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns::gpu::la {

// D^-1 per row (dense B x B inverse, partial pivoting).
template <class T>
class BlockJacobi
{
public:
    static std::expected<BlockJacobi, std::string> make(const BlockEll<T>& A);
    // Error (exit 7 class): a zero or non-finite pivot, or a non-finite
    // inverse in T, in any row; the factors are then unspecified.
    std::expected<void, std::string> factor();
    // factor() without its host sync: flag (device int) is set nonzero on
    // that failure, never cleared. Pre: the caller reads flag before any
    // result computed with these factors is used; failure text kFactorError.
    void factorDeferred(int* flag);
    static constexpr const char* kFactorError = "BlockJacobi factor: a singular or non-finite diagonal block";
    // z = D^-1 r. Pre: factor() succeeded; z != r.
    void apply(const T* r, T* z) const;
    // `sweeps` damped Jacobi steps x += omega D^-1 (b - A x). work: size(),
    // != x, b (a ping-pong buffer: each step is one fused launch).
    void smooth(const T* b, T* x, int sweeps, double omega, T* work) const;
    // The same steps from x = 0 (x's contents ignored): bit for bit a zeroed x
    // then smooth(), one launch fewer per call and no memset.
    void smoothFromZero(const T* b, T* x, int sweeps, double omega, T* work) const;
    const T* inverse() const { return inv_.data(); }

private:
    BlockJacobi() = default;
    void run(const T* b, T* x, bool fromZero, int count, double omega, T* work) const;
    const BlockEll<T>* A_ = nullptr;
    DeviceBuffer<T> inv_;
    DeviceBuffer<int> bad_;
};

// Multicolour block GS: rows of one colour never couple (distance-1 colours,
// dns::la::greedyColour), so each colour is one exact parallel GS pass
// x_c = (1 - w) x_c + w D_c^-1 (b_c - sum_{j != c} A_cj x_j). The smoother
// keeps a PRIVATE colour-major copy of A's off-diagonal ELL (rows of a colour
// contiguous, ascending cell order inside a colour, each row's entries in A's
// column order) and its permuted D^-1, so each colour pass reads coalesced
// slices whatever the cell order; vectors stay in cell order, A is untouched.
// Same per-row expressions as a cell-order walk: the same bits.
template <class T>
class MultiColourGS
{
public:
    // Pre: colours.size() == A.rows(), a valid distance-1 colouring of A's
    // pattern (checked: error names the first conflicting row pair).
    static std::expected<MultiColourGS, std::string> make(const BlockEll<T>& A,
                                                          std::span<const std::uint8_t> colours);
    // Re-reads A: the off-diagonal values into the private copy (offChanged
    // false: they are those of the last factor, the copy is kept), D^-1.
    // Error: kFactorError (a singular or non-finite diagonal block).
    std::expected<void, std::string> factor(bool offChanged = true);
    // factor() without its host sync (contract of BlockJacobi::factorDeferred).
    void factorDeferred(int* flag, bool offChanged = true);
    static constexpr const char* kFactorError = BlockJacobi<T>::kFactorError;
    // One sweep over the colours ascending (forward) or descending.
    void sweep(const T* b, T* x, bool forward, double omega = 1.0) const;
    // `sweeps` symmetric sweeps (forward then backward each). At omega 1 a
    // colour pass that would directly repeat the previous pass (the turn of a
    // symmetric sweep, the seam between two) is skipped: its inputs are
    // unchanged, so it would rewrite the same bits.
    void smooth(const T* b, T* x, int sweeps, double omega = 1.0) const;
    // z = M^-1 r: one symmetric sweep from z = 0 (a symmetric M for symmetric A).
    void apply(const T* r, T* z) const;
    int colours() const { return int(colourStart_.size()) - 1; }

    // nb <= 3 B = 1 systems sharing A's off-diagonals, system j with its own
    // diagonal diags + j rows() (j in mask, bit j): factorMulti is factor() of
    // each (flags[j] its deferred flag, contract of factorDeferred), applyMulti
    // apply() of each on r + j rows() -> z + j rows(), bit for bit, one launch
    // per colour pass for all. Pre: shape B == 1. Error: device allocation.
    std::expected<void, std::string> factorMulti(int nb, unsigned mask, const T* diags, int* const* flags,
                                                 bool offChanged = true);
    void applyMulti(int nb, unsigned mask, const T* r, T* z) const;

private:
    MultiColourGS() = default;
    // passes over colours `order` (skipping a direct repeat at omega 1)
    void passes(const T* b, T* x, std::span<const int> order, double omega) const;
    const BlockEll<T>* A_ = nullptr;
    std::vector<int> colourStart_; // rows_[colourStart_[k] .. colourStart_[k+1])
    int nE_ = 0;                   // private ELL entries
    DeviceBuffer<int> rows_, base_, width_, col_, src_, bad_;
    DeviceBuffer<T> off_, sc_, inv_; // private values (A's, by src_), D^-1 per slot
    DeviceBuffer<T> invMulti_;       // factorMulti: system j's D^-1 per slot at j rows()
    int nbMulti_ = 0;
    unsigned maskMulti_ = 0;
};

// Batched block-tridiagonal line solves: M = the matrix restricted to each
// line's tridiagonal (diagonal blocks + the couplings between consecutive line
// cells), one thread per line, lines sorted by length (descending) in groups
// of 64 padded to their longest. Off-line couplings are lagged by smooth().
template <class T>
class LineSmoother
{
public:
    // Pre: lines.n == A.rows() (checked), lines from dns::la::lineSet.
    static std::expected<LineSmoother, std::string> make(const BlockEll<T>& A, const dns::la::LineSet& lines);
    // Block Thomas in FP64. Error: a zero or non-finite pivot.
    std::expected<void, std::string> factor();
    // factor() without its host sync (contract of BlockJacobi::factorDeferred).
    void factorDeferred(int* flag);
    static constexpr const char* kFactorError = "LineSmoother factor: a singular or non-finite line pivot";
    // z = M^-1 r. Pre: factor() succeeded; z != r.
    void apply(const T* r, T* z) const;
    // `sweeps` line-Jacobi steps x += omega M^-1 (b - A x). work: 2 size().
    void smooth(const T* b, T* x, int sweeps, T* work, double omega = 1.0) const;
    int lines() const { return nLines_; }

private:
    LineSmoother() = default;
    const BlockEll<T>* A_ = nullptr;
    int nLines_ = 0, nP_ = 0;
    DeviceBuffer<int> len_, gBase_, cell_, eLo_, eHi_, bad_;
    DeviceBuffer<T> Dp_, G_, C_; // per position: T^-1, T^-1 L, T^-1 U (B x B each)
};

extern template class BlockJacobi<float>;
extern template class BlockJacobi<double>;
extern template class MultiColourGS<float>;
extern template class MultiColourGS<double>;
extern template class LineSmoother<float>;
extern template class LineSmoother<double>;

} // namespace dns::gpu::la
