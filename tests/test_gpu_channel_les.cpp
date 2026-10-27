// Channel LES (`dns channel --les CS [--sgs smag|sigma]`, docs/channel.md
// "LES") through the built binary, 16 x 32 x 16, ~200 steps per leg:
//   L1  smag and sigma on the tripped IC: exit 0, every stats/profile value
//       finite, nutMax > 0, and the wall-face mean mu_t (nutWall) EXACTLY 0.0
//       on every row while the first fluid row's mean is > 0 -- the odd image
//       on both ghost rows (a raw ghost row is >= 0, so only the image sums
//       to zero)
//   L2  the label: "# LES <model> ..." opens both CSVs, the .json says les 1
//       and the model, the progress line carries it
//   L3  sigma on the UNTRIPPED laminar IC (--kick 0, one step): mu_t == 0
//       exactly everywhere (D_sigma vanishes in pure shear); smag on the same
//       state is not (the check discriminates)
//   L4  --les 0 is the DNS channel byte for byte (stats + profiles + json
//       minus the timing field)
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;

const std::string kGrid = "channel --n 16 --ny 32 --nz 16 --re-tau 180 ";

std::string scratch(const std::string& name)
{
    return (std::filesystem::temp_directory_path()
            / std::format("dns_chles_{}_{}", ::getpid(), name))
        .string();
}

void cleanup(const std::string& prefix)
{
    std::error_code ec;
    for (const char* suf :
         {".json", "_stats.csv", "_profiles.csv", "_mask.bin", ".log"})
        std::filesystem::remove(prefix + suf, ec);
}

std::string slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::string firstLine(const std::string& path)
{
    std::ifstream f(path);
    std::string l;
    std::getline(f, l);
    return l;
}

// The .json without its msPerStep field (wall time, never reproducible).
std::string jsonSansTiming(const std::string& path)
{
    std::string s = slurp(path);
    const auto p = s.find(",\"msPerStep\":");
    if (p == std::string::npos) return s;
    const auto q = s.find_first_of(",}", p + 1);
    return s.erase(p, q - p);
}

bool allFinite(const valtest::CsvTable& t)
{
    return std::ranges::all_of(t.cols, [](const auto& c) {
        return std::ranges::all_of(c, [](double v) { return std::isfinite(v); });
    });
}

int run(const std::string& args, const std::string& prefix)
{
    cleanup(prefix);
    return valtest::runBinaryTo("\"" DNS_BIN "\" " + kGrid + args
                                    + " --emit stats --out " + prefix,
                                prefix + ".log");
}

// L1 + L2 for one model on the tripped IC.
void tripped(const std::string& model, const std::string& flags,
             const std::string& label)
{
    const std::string p = scratch(model);
    // ~250-300 steps: a stats row at step 200 and one at tEnd.
    const int rc = run(flags + " --tt 0.015 --profile-dt 0.015", p);
    check(rc == 0, model + ": exit 0");
    const auto st = valtest::loadCsv(p + "_stats.csv");
    const auto pr = valtest::loadCsv(p + "_profiles.csv");
    check(st && st->rows() >= 2 && pr && pr->rows() >= 32,
          model + ": stats and profile rows written");
    if (!st || !pr || st->rows() == 0 || pr->rows() < 32) return;
    check(allFinite(*st) && allFinite(*pr), model + ": every value finite");
    const auto& nmax = st->col("nutMax");
    const auto& nwall = st->col("nutWall");
    const auto& nprof = pr->col("nutRatio");
    const auto& yp = pr->col("y");
    std::printf("  %s: %zu stats rows, nutMax %.4g, first-row mean mu_t/mu "
                "%.4g at y %.4g\n",
                model.c_str(), st->rows(), nmax.back(), nprof.front(),
                yp.front());
    check(std::ranges::all_of(nmax, [](double v) { return v > 0; }),
          model + ": nutMax > 0 on every row");
    check(std::ranges::all_of(nwall, [](double v) { return v == 0.0; }),
          model + ": wall-face mean mu_t == 0.0 exactly on every row");
    check(nprof.front() > 0 && nprof.back() > 0,
          model + ": first fluid rows carry mu_t > 0 (the zero is the image)");
    const std::string head = "# " + label;
    check(firstLine(p + "_stats.csv").starts_with(head),
          model + ": stats CSV opens with '" + head + "'");
    check(firstLine(p + "_profiles.csv").starts_with(head),
          model + ": profiles CSV opens with the label");
    const std::string js = slurp(p + ".json");
    check(js.contains("\"les\":1") && js.contains("\"sgs\":\"" + model + "\""),
          model + ": .json says les 1 and the model");
    check(slurp(p + ".log").contains(label), model + ": progress line labelled");
    cleanup(p);
}

// L3: one step on the untripped laminar state; returns max nutMax.
double laminarNutMax(const std::string& model, const std::string& flags)
{
    const std::string p = scratch(model + "_lam");
    const int rc = run(flags + " --kick 0 --tt 1e-5 --profile-dt 1e-5", p);
    check(rc == 0, model + " laminar: exit 0");
    const auto st = valtest::loadCsv(p + "_stats.csv");
    const auto pr = valtest::loadCsv(p + "_profiles.csv");
    if (!st || !pr || st->rows() == 0 || pr->rows() == 0) {
        check(false, model + " laminar: rows written");
        return NAN;
    }
    const auto& np = pr->col("nutRatio");
    const double m = std::max(st->col("nutMax").back(),
                              *std::ranges::max_element(np));
    std::printf("  %s laminar (kick 0, %zu row(s)): max mu_t/mu %.6g\n",
                model.c_str(), st->rows(), m);
    cleanup(p);
    return m;
}

} // namespace

int main()
{
    const std::string probe = scratch("probe");
    if (valtest::runBinary("\"" DNS_BIN "\" " + kGrid
                           + "--tt 1e-6 --emit none --out " + probe) == 5)
        dnstest::skip("no HIP device");
    cleanup(probe);

    std::printf("L1/L2 tripped IC:\n");
    tripped("smag", "--les 0.17",
            "LES smag Cs=0.17 (filtered: grid- and Cs-dependent)");
    tripped("sigma", "--les 1.35 --sgs sigma",
            "LES sigma Csigma=1.35 (filtered: grid- and Cs-dependent)");

    std::printf("L3 laminar parabolic IC:\n");
    const double sig = laminarNutMax("sigma", "--les 1.35 --sgs sigma");
    const double sma = laminarNutMax("smag", "--les 0.17");
    check(sig == 0.0, "sigma: mu_t == 0 exactly on the laminar (pure shear) "
                      "state");
    check(sma > 0.0, "smag: mu_t > 0 on the same state (L3 discriminates)");

    std::printf("L4 --les 0 vs no flag:\n");
    const std::string a = scratch("dns"), b = scratch("les0");
    const std::string args = "--tt 0.01 --profile-dt 0.01";
    const int ra = run(args, a), rb = run(args + " --les 0", b);
    check(ra == 0 && rb == 0, "both exit 0");
    const std::string sa = slurp(a + "_stats.csv"), pa = slurp(a + "_profiles.csv");
    check(!sa.empty() && sa == slurp(b + "_stats.csv"),
          "--les 0 stats byte-identical to the flagless run");
    check(!pa.empty() && pa == slurp(b + "_profiles.csv"),
          "--les 0 profiles byte-identical to the flagless run");
    check(jsonSansTiming(a + ".json") == jsonSansTiming(b + ".json"),
          "--les 0 .json identical but for msPerStep");
    check(!sa.starts_with("#"), "a DNS stats CSV carries no label line");
    cleanup(a);
    cleanup(b);
    return dnstest::finish();
}
