// Run-to-run determinism through the built binary: every runner, run twice on
// the same arguments, must write a byte-identical stats CSV. The repo pins
// "deterministic iteration/order wherever output depends on it" and the FP64
// staged tallies exist so cd is reproducible; this is the end-to-end check
// that no atomic, no unordered container and no thread race reached a column.
//
// Cases: compressible nscbc cylinder, incompressible tunnel (the default),
// moving rotor with an azimuth table, MRF rotor, O-grid, channel, conv (both
// physics). Each run is a few dozen steps -- determinism, not physics.
#include "validation_util.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;

bool sameFile(const std::string& a, const std::string& b)
{
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    if (!fa || !fb) return false;
    return std::equal(std::istreambuf_iterator<char>(fa),
                      std::istreambuf_iterator<char>(),
                      std::istreambuf_iterator<char>(fb),
                      std::istreambuf_iterator<char>());
}

std::string scratch(const std::string& name)
{
    return (std::filesystem::temp_directory_path() / ("dns_det_" + name))
        .string();
}

void cleanup(const std::string& prefix)
{
    std::error_code ec;
    for (const char* suf : {".json", "_stats.csv", "_profiles.csv"})
        std::filesystem::remove(prefix + suf, ec);
}

// Run `sub args` twice and compare the two stats files byte for byte.
void twice(const std::string& name, const std::string& args)
{
    const std::string pa = scratch(name + "_a"), pb = scratch(name + "_b");
    cleanup(pa);
    cleanup(pb);
    const auto t0 = std::chrono::steady_clock::now();
    const int ra = valtest::runBinary("\"" DNS_BIN "\" " + args
                                      + " --emit stats --quiet --out " + pa);
    const int rb = valtest::runBinary("\"" DNS_BIN "\" " + args
                                      + " --emit stats --quiet --out " + pb);
    const double secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
    const auto tbl = valtest::loadCsv(pa + "_stats.csv");
    const std::size_t rows = tbl ? tbl->col("t").size() : 0;
    std::printf("%-10s exit %d/%d, %zu stats rows, %.2f s for both runs\n",
                name.c_str(), ra, rb, rows, secs);
    check(ra == 0 && rb == 0, name + ": both runs exit 0");
    check(rows >= 2, name + ": stats rows written");
    check(sameFile(pa + "_stats.csv", pb + "_stats.csv"),
          name + ": stats CSV byte-identical across two runs");
    cleanup(pa);
    cleanup(pb);
}

} // namespace

int main()
{
    if (valtest::runBinary("\"" DNS_BIN "\" flow --body cylinder --n 32 --nz 1 "
                           "--ft 0.001 --emit none --out "
                           + scratch("probe")) == 5)
        dnstest::skip("no HIP device");
    cleanup(scratch("probe"));

    const std::string comp =
        "flow --physics compressible --body cylinder --n 128 --nz 1 ";
    twice("comp", comp + "--ft 0.03");
    twice("incomp", "flow --body cylinder --n 128 --nz 1 --ft 0.1");
    twice("rotor", comp + "--rotor-mach 0.3 --rotor-axis z --ft 0.03");
    twice("mrf", comp + "--rotor-mach 0.3 --rotor-axis z --rotor-frame mrf "
                        "--ft 0.03");
    twice("ogrid", "flow --physics compressible --body cylinder --ogrid "
                   "--n 128 --ny 66 --ogrid-far 5 --ramp 0 --ft 0.003");
    twice("channel", "channel --n 32 --ny 32 --nz 32 --re-tau 180 --tt 0.02 "
                     "--profile-dt 0");
    twice("conv", "conv --n 64 --ny 34 --rayleigh 5e4 --tt 0.5 "
                  "--profile-dt 0");
    twice("rb", "conv --physics incompressible --n 64 --ny 34 --rayleigh 5e4 "
                "--tt 2 --profile-dt 0");
    return dnstest::finish();
}
