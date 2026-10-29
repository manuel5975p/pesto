// Device block matrices of the solver library (docs/la.md, masterplan_meshsolver
// 2.4): sliced ELL on a CellGraph pattern, SoA component-major so a slice's 64
// rows read coalesced. Values T in {float, double}; every supported BlockShape
// (dns/la/ell.hpp) has compiled kernels. Mesh-agnostic, deterministic (one
// thread per row, entries in ascending column order, no atomics).
// GPU-only header (host-includable: no kernel declarations).
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/la/ell.hpp"

#include <expected>
#include <string>

namespace dns::gpu::la {

// Raw device view, what kernels (and future assembly kernels) take:
//  diag[(r B + q) n + c], off[(r Bd + q) nE + E], sc[e nE + E], col[E]
// with E = sliceBase[c / 64] + 64 k + c % 64 (dns::la::EllPattern).
template <class T>
struct EllView
{
    int n = 0, nE = 0;
    dns::la::BlockShape shape;
    const int* sliceBase = nullptr;
    const int* sliceWidth = nullptr;
    const int* col = nullptr;
    const T* diag = nullptr;
    const T* off = nullptr;
    const T* sc = nullptr;
};

// Vectors are flat, component-major: v[q n + c], length n B.
template <class T>
class BlockEll
{
public:
    // Allocates a zero matrix. Error: unsupported shape, device allocation.
    static std::expected<BlockEll, std::string> make(const dns::la::EllPattern& p, dns::la::BlockShape s);

    // Values from a host FP64 matrix, rounded once to T.
    // Error (nothing uploaded): shape, row or nnz mismatch; device copy.
    std::expected<void, std::string> upload(const dns::la::BlockMatrix& m);

    // y = A x. Pre: x, y device vectors of size(), not aliased.
    void spmv(const T* x, T* y) const;
    // r = b - A x. Pre: as spmv; r may alias b, never x.
    void residual(const T* b, const T* x, T* r) const;
    // nb <= 3 B = 1 systems sharing A's off-diagonals, each with its own
    // diagonal: y_j = D_j x_j + Off x_j for j in mask (bit j), D_j = diags + j
    // rows(), x_j = x + j xStride (xStride 0: one x for all), y_j = y + j rows().
    // Bit for bit spmv() with A's diagonal replaced by D_j; one launch, the
    // column and value loads shared. Pre: shape().B == 1.
    void spmvMulti(int nb, unsigned mask, const T* diags, const T* x, std::size_t xStride, T* y) const;

    EllView<T> view() const;
    const dns::la::EllPattern& pattern() const { return pat_; }
    dns::la::BlockShape shape() const { return shape_; }
    int rows() const { return pat_.n; }
    std::size_t size() const { return std::size_t(pat_.n) * std::size_t(shape_.B); }

    // Device values for in-place assembly (slot -> E through slotPos()).
    T* diag() { return diag_.data(); }
    T* off() { return off_.data(); }
    T* sc() { return sc_.data(); }
    const int* slotPos() const { return slot_.data(); }

private:
    BlockEll() = default;
    dns::la::EllPattern pat_;
    dns::la::BlockShape shape_;
    DeviceBuffer<int> base_, width_, col_, slot_;
    DeviceBuffer<T> diag_, off_, sc_;
};

extern template class BlockEll<float>;
extern template class BlockEll<double>;

} // namespace dns::gpu::la
