#include "jobs/sha256.hpp"

#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

namespace jobs {

namespace {

constexpr std::array<std::uint32_t, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

std::string errnoText(const char* what)
{
    return std::string(what) + ": " + std::strerror(errno);
}

} // namespace

Sha256::Sha256()
    : h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}
{
}

void Sha256::block(const std::uint8_t* p)
{
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = std::uint32_t(p[4 * i]) << 24 | std::uint32_t(p[4 * i + 1]) << 16 |
               std::uint32_t(p[4 * i + 2]) << 8 | std::uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = h_;
    for (int i = 0; i < 64; ++i) {
        const std::uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) +
                                 ((e & f) ^ (~e & g)) + kK[std::size_t(i)] + w[i];
        const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) +
                                 ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    const std::uint32_t add[8] = {a, b, c, d, e, f, g, h};
    for (int i = 0; i < 8; ++i) h_[std::size_t(i)] += add[i];
}

void Sha256::update(std::span<const std::byte> data)
{
    bits_ += std::uint64_t(data.size()) * 8;
    for (std::byte x : data) {
        buf_[fill_++] = std::uint8_t(x);
        if (fill_ == 64) {
            block(buf_.data());
            fill_ = 0;
        }
    }
}

std::array<std::uint8_t, 32> Sha256::finish()
{
    const std::uint64_t bits = bits_;
    const std::byte one{0x80}, zero{0};
    update({&one, 1});
    while (fill_ != 56) update({&zero, 1});
    for (int i = 7; i >= 0; --i) {
        const std::byte b{std::uint8_t(bits >> (8 * i))};
        update({&b, 1});
    }
    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j)
            out[std::size_t(4 * i + j)] = std::uint8_t(h_[std::size_t(i)] >> (24 - 8 * j));
    return out;
}

std::string toHex(std::span<const std::uint8_t> d)
{
    static constexpr char kHex[] = "0123456789abcdef";
    std::string s;
    s.reserve(2 * d.size());
    for (std::uint8_t b : d) {
        s += kHex[b >> 4];
        s += kHex[b & 15];
    }
    return s;
}

std::expected<std::string, std::string> sha256Fd(int fd)
{
    if (::lseek(fd, 0, SEEK_SET) < 0) return std::unexpected(errnoText("lseek"));
    Sha256 h;
    std::array<std::byte, 1 << 16> buf;
    for (;;) {
        const ssize_t n = ::read(fd, buf.data(), buf.size());
        if (n < 0) {
            if (errno == EINTR) continue;
            return std::unexpected(errnoText("read"));
        }
        if (n == 0) break;
        h.update({buf.data(), std::size_t(n)});
    }
    if (::lseek(fd, 0, SEEK_SET) < 0) return std::unexpected(errnoText("lseek"));
    const auto d = h.finish();
    return toHex(d);
}

std::expected<std::string, std::string> sha256File(const std::string& path)
{
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return std::unexpected(errnoText(("open " + path).c_str()));
    auto r = sha256Fd(fd);
    ::close(fd);
    return r;
}

} // namespace jobs
