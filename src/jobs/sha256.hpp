// SHA-256 (FIPS 180-4), self-contained: the jobserver pins executables by
// content hash and must not pull a crypto library in for one digest.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>

namespace jobs {

class Sha256
{
public:
    Sha256();
    void update(std::span<const std::byte> data);
    // Final digest; the object is spent afterwards.
    std::array<std::uint8_t, 32> finish();

private:
    void block(const std::uint8_t* p);
    std::array<std::uint32_t, 8> h_;
    std::array<std::uint8_t, 64> buf_{};
    std::size_t fill_ = 0;
    std::uint64_t bits_ = 0;
};

// Lower-case hex of a digest.
std::string toHex(std::span<const std::uint8_t> d);

// Hex SHA-256 of everything readable from fd, from offset 0; the offset is
// left at 0 so the fd can be exec'd or re-read. Fails with errno text.
std::expected<std::string, std::string> sha256Fd(int fd);

// Hex SHA-256 of a file by path.
std::expected<std::string, std::string> sha256File(const std::string& path);

} // namespace jobs
