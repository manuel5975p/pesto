// fieldpng: written PNGs are valid (signature, IHDR dims, IDAT inflates to
// filter-0 scanlines), the colormaps hit their anchor colours (diverging
// -scale/0/+scale, viridis lo/hi), masked cells render dark, row 0 lands at
// the image bottom, and bad paths fail cleanly.
#include "dns/fieldpng.hpp"

#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%-56s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

std::uint32_t be32(const unsigned char* p)
{
    return std::uint32_t(p[0]) << 24 | std::uint32_t(p[1]) << 16 |
           std::uint32_t(p[2]) << 8 | p[3];
}

// Decode: verify signature/IHDR, inflate the single IDAT, strip filter bytes.
// Returns top-down RGB (empty on any structural failure).
std::vector<unsigned char> decode(const char* path, int w, int h)
{
    std::ifstream f(path, std::ios::binary);
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (buf.size() < 45 || std::memcmp(buf.data(), sig, 8) != 0) return {};
    if (std::memcmp(buf.data() + 12, "IHDR", 4) != 0) return {};
    if (be32(buf.data() + 16) != std::uint32_t(w) ||
        be32(buf.data() + 20) != std::uint32_t(h))
        return {};
    if (buf[24] != 8 || buf[25] != 2) return {}; // 8-bit RGB
    std::size_t off = 8 + 8 + 13 + 4;            // sig + IHDR chunk
    while (off + 8 <= buf.size() &&
           std::memcmp(buf.data() + off + 4, "IDAT", 4) != 0)
        off += 12 + be32(buf.data() + off);
    if (off + 8 > buf.size()) return {};
    const std::uint32_t zlen = be32(buf.data() + off);
    const std::size_t stride = std::size_t(w) * 3 + 1;
    std::vector<unsigned char> raw(stride * h);
    uLongf rlen = uLongf(raw.size());
    if (uncompress(raw.data(), &rlen, buf.data() + off + 8, zlen) != Z_OK ||
        rlen != raw.size())
        return {};
    std::vector<unsigned char> rgb(std::size_t(w) * h * 3);
    for (int r = 0; r < h; ++r) {
        if (raw[r * stride] != 0) return {}; // filter 0 only
        std::memcpy(rgb.data() + std::size_t(r) * w * 3,
                    raw.data() + r * stride + 1, std::size_t(w) * 3);
    }
    return rgb;
}

bool px(const std::vector<unsigned char>& rgb, int w, int h, int i, int jBottom,
        int r, int g, int b, int tol = 1)
{
    const int row = h - 1 - jBottom; // writer flips: row 0 = top = j max
    const unsigned char* p = rgb.data() + (std::size_t(row) * w + i) * 3;
    return std::abs(p[0] - r) <= tol && std::abs(p[1] - g) <= tol &&
           std::abs(p[2] - b) <= tol;
}

} // namespace

int main()
{
    const int w = 8, h = 4;
    // Probe cells sit in the INTERIOR (i in [1,6], j in [1,2]): writeMapped
    // replicates the inner neighbour into the outermost 1px ring (cosmetic
    // trim of the periodic-stencil rim), so edge cells no longer show their
    // own value.
    std::vector<float> field(w * h, 0.0f), mask(w * h, 1.0f);
    field[1 * w + 2] = -2.0f;      // (2, 1) = -scale, clipped
    field[2 * w + 3] = 1.0f;       // (3, 2) = +scale
    field[2 * w + 6] = 0.25f;      // (6, 2) = mid-blend
    mask[1 * w + 5] = 0.0f;        // (5, 1) solid

    const char* dPath = "fieldpng_div.png";
    check(dns::writeDivergingPng(dPath, w, h, field.data(), 1.0f, mask.data()),
          "diverging write succeeds");
    auto d = decode(dPath, w, h);
    check(!d.empty(), "diverging PNG structurally valid");
    if (!d.empty()) {
        // Glow diverging map: dark-grey core, blue/red saturation knot at
        // 0.62, white-hot extremes (matches site/live.html).
        check(px(d, w, h, 1, 1, 20, 20, 24), "zero maps to dark core");
        check(px(d, w, h, 2, 1, 225, 240, 255), "-scale clips to white-hot blue");
        check(px(d, w, h, 3, 2, 255, 240, 225), "+scale maps to white-hot red");
        check(px(d, w, h, 5, 1, 32, 32, 32), "masked cell renders dark");
        check(px(d, w, h, 6, 2, 115, 30, 26, 2), "quarter-scale blends");
        check(px(d, w, h, 0, 0, 20, 20, 24),
              "edge ring replicates the inner neighbour");
    }

    const char* sPath = "fieldpng_seq.png";
    std::vector<float> sf(w * h, 0.0f);
    sf[0 * w + 0] = -5.0f; // clips to lo
    sf[1 * w + 2] = 2.0f;  // = hi
    check(dns::writeSequentialPng(sPath, w, h, sf.data(), 0.0f, 2.0f,
                                  mask.data()),
          "sequential write succeeds");
    auto s = decode(sPath, w, h);
    check(!s.empty(), "sequential PNG structurally valid");
    if (!s.empty()) {
        check(px(s, w, h, 0, 0, 68, 1, 84), "lo clips to viridis dark");
        check(px(s, w, h, 2, 1, 253, 231, 37), "hi maps to viridis bright");
        check(px(s, w, h, 5, 1, 32, 32, 32), "mask honoured in sequential");
    }

    check(!dns::writeDivergingPng("/nonexistent-dir/x.png", w, h, field.data(),
                                  1.0f, nullptr),
          "unwritable path returns false");

    // No mask pointer: whole image mapped.
    check(dns::writeSequentialPng(sPath, w, h, sf.data(), 0.0f, 2.0f, nullptr),
          "null mask accepted");

    std::remove(dPath);
    std::remove(sPath);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
