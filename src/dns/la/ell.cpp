#include "dns/la/ell.hpp"

#include <algorithm>
#include <array>
#include <cassert>

namespace dns::la {

namespace {
constexpr std::array<int, 7> kSizes{1, 3, 5, 6, 7, 9, 11};
constexpr int kDenseSplit = 5; // the mean-flow block of every split shape
} // namespace

std::span<const int> supportedBlockSizes() { return kSizes; }

std::expected<BlockShape, std::string> blockShape(int B, int Bd, bool sharedScalar)
{
    if (std::ranges::find(kSizes, B) == kSizes.end())
        return std::unexpected("unsupported block size B = " + std::to_string(B) +
                               " (compiled: 1 3 5 6 7 9 11)");
    if (Bd == B) return BlockShape{B, B, 0};
    if (Bd != kDenseSplit || B <= kDenseSplit)
        return std::unexpected("unsupported dense off-diagonal part Bd = " + std::to_string(Bd) +
                               " for B = " + std::to_string(B) + " (Bd == B, or Bd == 5 < B)");
    const int nS = sharedScalar ? 1 : B - Bd;
    return BlockShape{B, Bd, nS};
}

std::expected<BlockShape, std::string> denseShape(int B) { return blockShape(B, B, false); }

EllPattern ellPattern(const CellGraph& g)
{
    EllPattern p;
    p.n = g.n;
    p.nSlices = (g.n + kSliceRows - 1) / kSliceRows;
    p.sliceBase.assign(std::size_t(p.nSlices) + 1, 0);
    p.sliceWidth.assign(std::size_t(p.nSlices), 0);
    for (int c = 0; c < g.n; ++c) {
        const int w = g.rowPtr[std::size_t(c) + 1] - g.rowPtr[std::size_t(c)];
        int& sw = p.sliceWidth[std::size_t(c / kSliceRows)];
        sw = std::max(sw, w);
    }
    for (int s = 0; s < p.nSlices; ++s) {
        p.sliceBase[std::size_t(s) + 1] = p.sliceBase[std::size_t(s)] + kSliceRows * p.sliceWidth[std::size_t(s)];
        p.maxWidth = std::max(p.maxWidth, p.sliceWidth[std::size_t(s)]);
    }
    p.nE = p.sliceBase[std::size_t(p.nSlices)];
    p.col.assign(std::size_t(p.nE), -1);
    p.slotPos.assign(g.col.size(), -1);
    for (int c = 0; c < g.n; ++c) {
        const int base = p.sliceBase[std::size_t(c / kSliceRows)] + c % kSliceRows;
        for (int q = g.rowPtr[std::size_t(c)]; q < g.rowPtr[std::size_t(c) + 1]; ++q) {
            const int E = base + kSliceRows * (q - g.rowPtr[std::size_t(c)]);
            p.col[std::size_t(E)] = g.col[std::size_t(q)];
            p.slotPos[std::size_t(q)] = E;
        }
    }
    return p;
}

BlockMatrix BlockMatrix::zeros(const CellGraph& g, BlockShape s)
{
    BlockMatrix m;
    m.shape = s;
    m.n = g.n;
    m.rowPtr = g.rowPtr;
    m.col = g.col;
    const std::size_t B = std::size_t(s.B);
    m.diag.assign(std::size_t(g.n) * B * B, 0.0);
    m.off.assign(g.col.size() * std::size_t(s.offDense()), 0.0);
    m.sc.assign(g.col.size() * std::size_t(s.nS), 0.0);
    return m;
}

double BlockMatrix::offAt(int slot, int r, int q) const
{
    assert(r < shape.B && q < shape.B);
    const std::size_t sl = std::size_t(slot);
    if (r < shape.Bd && q < shape.Bd)
        return off[(sl * std::size_t(shape.Bd) + std::size_t(r)) * std::size_t(shape.Bd) + std::size_t(q)];
    if (r >= shape.Bd && r == q)
        return sc[sl * std::size_t(shape.nS) + std::size_t(shape.nS == 1 ? 0 : r - shape.Bd)];
    return 0.0;
}

} // namespace dns::la
