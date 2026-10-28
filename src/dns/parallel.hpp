// Deterministic host threading for per-cell sweeps (CurvMetrics::build,
// sampleCut, the compressible body IC): contiguous blocks (forBlocks) or an
// atomic item queue (forItems), std::jthread workers. A sweep whose cells
// write only their own index and read only shared const input is
// bit-identical for any worker count -- the split decides which thread
// computes a cell, never what it computes. Reductions stay the caller's: one
// partial per forBlocks block, merged in block order.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace dns {

// Workers for `threads` (<= 0: hardware concurrency) over `items` units of
// work; never more than items, never fewer than 1.
inline int workerCount(int threads, std::size_t items)
{
    const int hw = int(std::max(1u, std::thread::hardware_concurrency()));
    const std::size_t want = std::size_t(threads > 0 ? threads : hw);
    return int(std::max<std::size_t>(1, std::min(want, items)));
}

// body(w, lo, hi) over nw contiguous ascending blocks of [0, n): block w is
// [n w / nw, n (w + 1) / nw). nw == 1 runs inline on the caller's thread.
// Pre: nw >= 1. Post: every block has returned.
template <class F>
void forBlocks(int nw, std::size_t n, F&& body)
{
    const auto lo = [&](int w) {
        return n * std::size_t(w) / std::size_t(nw);
    };
    if (nw <= 1) {
        body(0, std::size_t(0), n);
        return;
    }
    std::vector<std::jthread> pool;
    pool.reserve(std::size_t(nw));
    for (int w = 0; w < nw; ++w)
        pool.emplace_back([&, w] { body(w, lo(w), lo(w + 1)); });
}

// body(w, item) for every item of [0, n) exactly once, handed out by an
// atomic counter (rotortable's schedule): for items of uneven cost (rows
// near a body cost many sdf samples, far rows one). Which worker runs an
// item varies run to run, so an item must write only its own outputs.
// nw == 1 runs inline in ascending order. Pre: nw >= 1.
template <class F>
void forItems(int nw, std::size_t n, F&& body)
{
    if (nw <= 1) {
        for (std::size_t i = 0; i < n; ++i) body(0, i);
        return;
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::jthread> pool;
    pool.reserve(std::size_t(nw));
    for (int w = 0; w < nw; ++w)
        pool.emplace_back([&, w] {
            for (std::size_t i = next++; i < n; i = next++) body(w, i);
        });
}

} // namespace dns
