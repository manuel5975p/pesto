#include "dns/fieldpng.hpp"

#include <zlib.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace dns {

namespace {

void be32(unsigned char* p, std::uint32_t v)
{
    p[0] = (v >> 24) & 0xff;
    p[1] = (v >> 16) & 0xff;
    p[2] = (v >> 8) & 0xff;
    p[3] = v & 0xff;
}

void chunk(std::ofstream& f, const char* type, const unsigned char* data,
           std::uint32_t len)
{
    unsigned char hdr[8];
    be32(hdr, len);
    std::memcpy(hdr + 4, type, 4);
    f.write(reinterpret_cast<const char*>(hdr), 8);
    if (len) f.write(reinterpret_cast<const char*>(data), len);
    std::uint32_t crc = crc32(0, hdr + 4, 4);
    if (len) crc = crc32(crc, data, len);
    unsigned char tail[4];
    be32(tail, crc);
    f.write(reinterpret_cast<const char*>(tail), 4);
}

// rgb is top-down, 3 w h bytes.
bool writePngRgb(const std::string& path, int w, int h,
                 const std::vector<unsigned char>& rgb)
{
    const std::size_t stride = std::size_t(w) * 3;
    std::vector<unsigned char> raw((stride + 1) * h); // filter byte + row
    for (int r = 0; r < h; ++r) {
        raw[r * (stride + 1)] = 0;
        std::memcpy(raw.data() + r * (stride + 1) + 1, rgb.data() + r * stride,
                    stride);
    }
    uLongf zlen = compressBound(uLong(raw.size()));
    std::vector<unsigned char> z(zlen);
    if (compress2(z.data(), &zlen, raw.data(), uLong(raw.size()), 6) != Z_OK)
        return false;

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    f.write(reinterpret_cast<const char*>(sig), 8);
    unsigned char ihdr[13] = {};
    be32(ihdr, std::uint32_t(w));
    be32(ihdr + 4, std::uint32_t(h));
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // colour type RGB
    chunk(f, "IHDR", ihdr, 13);
    chunk(f, "IDAT", z.data(), std::uint32_t(zlen));
    chunk(f, "IEND", nullptr, 0);
    return bool(f);
}

// map(value at cell) -> rgb; solid cells dark grey; j = 0 bottom -> top-down.
template <class Map>
bool writeMapped(const std::string& path, int w, int h, const float* field,
                 const float* mask, Map map)
{
    assert(w > 0 && h > 0 && field);
    std::vector<unsigned char> rgb(std::size_t(w) * h * 3);
    for (int r = 0; r < h; ++r) {
        const int j = h - 1 - r;
        for (int i = 0; i < w; ++i) {
            const std::size_t c = std::size_t(j) * w + i;
            unsigned char* px = rgb.data() + (std::size_t(r) * w + i) * 3;
            if (mask && mask[c] < 0.5f) {
                px[0] = px[1] = px[2] = 32;
                continue;
            }
            const std::array<float, 3> col = map(field[c]);
            for (int q = 0; q < 3; ++q)
                px[q] = static_cast<unsigned char>(col[q] * 255.0f + 0.5f);
        }
    }
    return writePngRgb(path, w, h, rgb);
}

std::array<float, 3> lerp3(const float a[3], const float b[3], float t)
{
    return {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t};
}

} // namespace

bool writeDivergingPng(const std::string& path, int w, int h,
                       const float* field, float scale, const float* mask)
{
    assert(scale > 0.0f);
    // Moreland cool-warm endpoints, light midpoint.
    static const float lo[3] = {59 / 255.0f, 76 / 255.0f, 192 / 255.0f};
    static const float mid[3] = {221 / 255.0f, 221 / 255.0f, 221 / 255.0f};
    static const float hi[3] = {180 / 255.0f, 4 / 255.0f, 38 / 255.0f};
    return writeMapped(path, w, h, field, mask, [&](float v) {
        float t = v / scale;
        t = t < -1.0f ? -1.0f : (t > 1.0f ? 1.0f : t);
        return t < 0.0f ? lerp3(mid, lo, -t) : lerp3(mid, hi, t);
    });
}

bool writeSequentialPng(const std::string& path, int w, int h,
                        const float* field, float lo, float hi,
                        const float* mask)
{
    assert(hi > lo);
    static const float vir[10][3] = {
        {68, 1, 84},    {72, 40, 120},  {62, 74, 137},  {49, 104, 142},
        {38, 130, 142}, {31, 158, 137}, {53, 183, 121}, {109, 205, 89},
        {180, 222, 44}, {253, 231, 37}};
    return writeMapped(path, w, h, field, mask, [&](float v) {
        float t = (v - lo) / (hi - lo);
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        const float s = t * 9.0f;
        const int a = s >= 9.0f ? 8 : int(s);
        const float f = s - a;
        std::array<float, 3> c = lerp3(vir[a], vir[a + 1], f);
        return std::array<float, 3>{c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f};
    });
}

} // namespace dns
