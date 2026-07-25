// WebFrames: manifest carries the declared fields/frames, u16 quantization
// round-trips within one LSB of the declared range, statics and per-frame
// minmax are recorded, and the manifest is valid enough JSON to parse keys.
#include "dns/webframes.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using dnstest::check;



std::string slurp(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::vector<std::uint16_t> loadU16(const std::string& p, std::size_t n)
{
    std::vector<std::uint16_t> v(n);
    std::ifstream f(p, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()),
           std::streamsize(n * sizeof(std::uint16_t)));
    return f ? v : std::vector<std::uint16_t>{};
}

} // namespace

int main()
{
    const std::string dir = "webframes_test_dir";
    std::filesystem::remove_all(dir);
    const int w = 16, h = 8;
    const std::size_t n = std::size_t(w) * h;

    dns::WebFrames wf(dir, w, h, "{\"re\":60}");
    wf.addField("vort", "diverging", -4.0, 4.0);
    wf.addField("speed", "sequential", 0.0, 2.0);

    std::vector<float> mask(n, 1.0f);
    mask[3] = 0.0f;
    check(wf.writeStatic("mask", mask.data(), 0.0, 1.0), "static mask written");

    std::vector<float> vort(n), speed(n);
    for (std::size_t c = 0; c < n; ++c) {
        vort[c] = float(std::sin(0.3 * double(c)) * 3.0);
        speed[c] = float(0.5 + 0.4 * std::cos(0.2 * double(c)));
    }
    vort[5] = 9.0f; // clips at hi
    check(wf.writeFrame(0.5, {vort.data(), speed.data()}), "frame 0 written");
    check(wf.writeFrame(1.0, {vort.data(), speed.data()}), "frame 1 written");
    check(wf.frames() == 2, "frame counter");

    const std::string m = slurp(dir + "/manifest.json");
    check(m.find("\"version\":1") != std::string::npos, "manifest version");
    check(m.find("\"re\":60") != std::string::npos, "sim passthrough");
    check(m.find("\"vort_%05d.bin\"") != std::string::npos, "field file pattern");
    check(m.find("\"kind\":\"diverging\"") != std::string::npos, "field kind");
    check(m.find("\"mask.bin\"") != std::string::npos, "static entry");
    check(m.find("\"t\":0.5") != std::string::npos &&
              m.find("\"t\":1") != std::string::npos,
          "frame times recorded");
    check(m.find("\"minmax\"") != std::string::npos, "per-frame minmax");
    check(!std::filesystem::exists(dir + "/manifest.json.tmp"),
          "tmp manifest renamed away");

    // u16 round-trip: |decoded - clip(v)| <= one quantization step.
    auto q = loadU16(dir + "/vort_00000.bin", n);
    check(!q.empty(), "frame file present and sized");
    double maxErr = 0.0;
    if (!q.empty())
        for (std::size_t c = 0; c < n; ++c) {
            const double v =
                std::min(4.0, std::max(-4.0, double(vort[c])));
            const double dec = -4.0 + 8.0 * q[c] / 65535.0;
            maxErr = std::max(maxErr, std::abs(dec - v));
        }
    std::printf("u16 round-trip max err: %.3e (step %.3e)\n", maxErr,
                8.0 / 65535.0);
    check(maxErr <= 8.0 / 65535.0, "quantization within one step");

    auto q1 = loadU16(dir + "/speed_00001.bin", n);
    check(!q1.empty(), "second frame second field present");

    std::filesystem::remove_all(dir);
    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
