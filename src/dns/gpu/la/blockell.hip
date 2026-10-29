#include "dns/gpu/la/blockell.hpp"

#include "dns/gpu/la/internal/ellops.hpp"

#include <cassert>
#include <vector>

namespace dns::gpu::la {

namespace {

using internal::gridFor;
using internal::kThreads;

// y = A x, or y = b - A x when b is non-null. One thread per row; the
// diagonal block first, then the entries in ascending column order
// (internal::spmvRow: prefetched at B == 1, same bits).
template <int B, int Bd, class T>
__global__ void __launch_bounds__(256) kSpmv(EllView<T> A, const T* __restrict__ x, const T* b, T* y)
{
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (c >= A.n) return;
    const std::size_t n = std::size_t(A.n);
    T acc[B];
    internal::spmvRow<B, Bd>(A, c, x, acc);
#pragma unroll
    for (int r = 0; r < B; ++r) {
        const std::size_t i = std::size_t(r) * n + std::size_t(c);
        y[i] = b ? b[i] - acc[r] : acc[r];
    }
}

// kSpmv<1, 1> for nb systems with their own diagonals (BlockEll::spmvMulti).
template <class T>
__global__ void __launch_bounds__(256) kSpmvMulti(EllView<T> A, int nb, unsigned mask, const T* __restrict__ diags,
                                                  const T* __restrict__ x, std::size_t xs, T* __restrict__ y)
{
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (c >= A.n) return;
    const std::size_t n = std::size_t(A.n);
    T acc[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        acc[q] = T(0);
        if (q >= nb || !((mask >> q) & 1u)) continue;
        T s = T(0);
        s += diags[std::size_t(q) * n + std::size_t(c)] * x[std::size_t(q) * xs + std::size_t(c)];
        acc[q] = s;
    }
    internal::ellRowMulti<false>(A, c, nb, mask, x, xs, acc);
#pragma unroll
    for (int q = 0; q < 3; ++q)
        if (q < nb && ((mask >> q) & 1u)) y[std::size_t(q) * n + std::size_t(c)] = acc[q];
}

// SoA scatter of the host matrix: one host pass, deterministic.
template <class T>
void scatter(const dns::la::BlockMatrix& m, const dns::la::EllPattern& p, std::vector<T>& diag,
             std::vector<T>& off, std::vector<T>& sc)
{
    const auto& s = m.shape;
    const std::size_t n = std::size_t(m.n), nE = std::size_t(p.nE), B = std::size_t(s.B),
                      Bd = std::size_t(s.Bd), nS = std::size_t(s.nS);
    diag.assign(std::max<std::size_t>(1, n * B * B), T(0));
    off.assign(std::max<std::size_t>(1, nE * Bd * Bd), T(0));
    sc.assign(std::max<std::size_t>(1, nE * nS), T(0));
    for (std::size_t c = 0; c < n; ++c)
        for (std::size_t e = 0; e < B * B; ++e) diag[e * n + c] = T(m.diag[c * B * B + e]);
    for (std::size_t q = 0; q < m.nnz(); ++q) {
        const std::size_t E = std::size_t(p.slotPos[q]);
        for (std::size_t e = 0; e < Bd * Bd; ++e) off[e * nE + E] = T(m.off[q * Bd * Bd + e]);
        for (std::size_t e = 0; e < nS; ++e) sc[e * nE + E] = T(m.sc[q * nS + e]);
    }
}

std::string hipText(const char* what, hipError_t e) { return std::string(what) + ": " + hipGetErrorString(e); }

template <class U>
std::expected<DeviceBuffer<U>, std::string> devVec(const std::vector<U>& h, const char* what)
{
    auto b = DeviceBuffer<U>::make(std::max<std::size_t>(1, h.size()));
    if (!b) return std::unexpected(hipText(what, b.error()));
    if (!h.empty())
        if (const hipError_t e = hipMemcpy(b->data(), h.data(), h.size() * sizeof(U), hipMemcpyHostToDevice);
            e != hipSuccess)
            return std::unexpected(hipText(what, e));
    return std::move(*b);
}

} // namespace

template <class T>
std::expected<BlockEll<T>, std::string> BlockEll<T>::make(const dns::la::EllPattern& p, dns::la::BlockShape s)
{
    if (auto ok = dns::la::blockShape(s.B, s.Bd, s.nS == 1); !ok || *ok != s)
        return std::unexpected<std::string>(ok ? std::string("block shape: nS inconsistent with B, Bd") : ok.error());
    BlockEll a;
    a.pat_ = p;
    a.shape_ = s;
    auto base = devVec(p.sliceBase, "BlockEll slices");
    if (!base) return std::unexpected(base.error());
    a.base_ = std::move(*base);
    auto width = devVec(p.sliceWidth, "BlockEll slices");
    if (!width) return std::unexpected(width.error());
    a.width_ = std::move(*width);
    auto col = devVec(p.col, "BlockEll columns");
    if (!col) return std::unexpected(col.error());
    a.col_ = std::move(*col);
    auto slot = devVec(p.slotPos, "BlockEll slots");
    if (!slot) return std::unexpected(slot.error());
    a.slot_ = std::move(*slot);
    const std::size_t B = std::size_t(s.B), nE = std::size_t(p.nE);
    const std::vector<T> d(std::max<std::size_t>(1, std::size_t(p.n) * B * B), T(0)),
        o(std::max<std::size_t>(1, nE * std::size_t(s.offDense())), T(0)),
        c(std::max<std::size_t>(1, nE * std::size_t(s.nS)), T(0));
    auto dd = devVec(d, "BlockEll values");
    if (!dd) return std::unexpected(dd.error());
    a.diag_ = std::move(*dd);
    auto oo = devVec(o, "BlockEll values");
    if (!oo) return std::unexpected(oo.error());
    a.off_ = std::move(*oo);
    auto cc = devVec(c, "BlockEll values");
    if (!cc) return std::unexpected(cc.error());
    a.sc_ = std::move(*cc);
    return a;
}

template <class T>
std::expected<void, std::string> BlockEll<T>::upload(const dns::la::BlockMatrix& m)
{
    if (m.shape != shape_) return std::unexpected("BlockEll upload: block shape mismatch");
    if (m.n != pat_.n) return std::unexpected("BlockEll upload: row count mismatch");
    if (m.nnz() != pat_.slotPos.size()) return std::unexpected("BlockEll upload: pattern (nnz) mismatch");
    std::vector<T> d, o, c;
    scatter(m, pat_, d, o, c);
    for (auto [buf, h] : {std::pair{&diag_, &d}, std::pair{&off_, &o}, std::pair{&sc_, &c}})
        if (const hipError_t e = buf->upload(h->data()); e != hipSuccess)
            return std::unexpected(hipText("BlockEll upload", e));
    return {};
}

template <class T>
EllView<T> BlockEll<T>::view() const
{
    return {pat_.n, pat_.nE, shape_, base_.data(), width_.data(), col_.data(), diag_.data(), off_.data(), sc_.data()};
}

template <class T>
void BlockEll<T>::spmv(const T* x, T* y) const
{
    if (pat_.n == 0) return;
    const EllView<T> v = view();
    [[maybe_unused]] const bool ok = internal::withShape(shape_, [&](auto sh) {
        kSpmv<decltype(sh)::B, decltype(sh)::Bd, T><<<gridFor(std::size_t(pat_.n)), kThreads>>>(v, x, nullptr, y);
    });
    assert(ok);
}

template <class T>
void BlockEll<T>::residual(const T* b, const T* x, T* r) const
{
    if (pat_.n == 0) return;
    const EllView<T> v = view();
    [[maybe_unused]] const bool ok = internal::withShape(shape_, [&](auto sh) {
        kSpmv<decltype(sh)::B, decltype(sh)::Bd, T><<<gridFor(std::size_t(pat_.n)), kThreads>>>(v, x, b, r);
    });
    assert(ok);
}

template <class T>
void BlockEll<T>::spmvMulti(int nb, unsigned mask, const T* diags, const T* x, std::size_t xStride, T* y) const
{
    assert(shape_.B == 1 && nb >= 1 && nb <= 3);
    if (pat_.n == 0) return;
    kSpmvMulti<T><<<gridFor(std::size_t(pat_.n)), kThreads>>>(view(), nb, mask, diags, x, xStride, y);
}

template class BlockEll<float>;
template class BlockEll<double>;

} // namespace dns::gpu::la
