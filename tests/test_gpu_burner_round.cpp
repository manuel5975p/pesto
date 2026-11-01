// The real-burner integration smoke (masterplan real_bunsen S9): the low-Mach
// round Bunsen with EVERY piece of the final case at once -- the immersed
// isothermal tube rim (--rim), open y/z laterals, real gravity, the rich
// partially premixed jet (--equivalence 1.3, C2b), Sutherland, the burner's
// default dt rule (--dt-rule cell), the SI echo -- nz 64, 4 cells/delta_th,
// tt 2, volumes. Driven through the dns binary.
//
//   (a) exits 0; every stats column the renderer and the final case read is
//       finite in every row (NaN allowed only where the stats say "not
//       defined": SLflank on the round nozzle, xTipOut while the outer cone
//       is open); dtPos present, dt <= dtPos; pcgIt in [1, 15]; yHits 0, Z in
//       [-1e-5, 1 + 1e-5];
//   (b) volumes written for every channel (temp, fuel, oxid, omega, findex,
//       speed: 5 frames, t 0..2), findex in [-1, 1]; the .json carries "rim"
//       and "si" (the renderer's real scale);
//   (c) the premixed / diffusion split of the LAST frame (t = tEnd: the last
//       stats row's state) with dns::lmPremixedWeight reproduces the row's
//       burnPre and burnDif to the CSV's 6 digits (the renderer's port,
//       tools/chem/flame_room_lib.py, is checked against the same files by
//       ctest flame_room_split);
//   (d) the restart byte rule once (a checkpoint at t 1, one restart).
//
// Usage: test_gpu_burner_round. No HIP device -> 77. Output kept under
// /tmp/dns_test_burner_round/br/straight/r for flame_room_split.
#include "dns/gpu/device_util.hpp"
#include "dns/lowmach.hpp"
#include "gpu_test_util.hpp"
#include "restart_test_util.hpp"
#include "validation_util.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using dnstest::check;
using namespace dnstest::restart;

const std::string kRoot = "/tmp/dns_test_burner_round";

// D 8 delta_th, lip 1 (4 cells), clearance 3: ny = nz = 16 delta_th x 4 = 64;
// length 30 above the rim (the validator's ideal cone + 4) + tube 4: nx 140
// (136 rounded up: nx/2 = 70, 7-smooth).
const std::string kArgs =
    "flame --mode bunsen --nozzle round --physics lowmach --equivalence 1.3 "
    "--expansion 6.899 --visc sutherland --tinf 298 --prandtl 0.5245 "
    "--slot-width 8 --jet-speed 6.5 --rim 1 --tube-length 4 --laterals open "
    "--coflow-width 3 --gravity 0.02406 --cells-per-delta 4 --length 30 --nz 64 "
    "--tt 2 --emit stats --emit volume --fdt 0.5 --si-delta-d 0.1312 --si-sl 0.23124";

std::vector<float> floats(const std::string& path)
{
    const std::string b = slurp(path);
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

// Agreement to the CSV's 6 significant digits: the value prints as the row.
bool sameRow(double v, double row)
{
    if (std::format("{:.6g}", v) == std::format("{:.6g}", row)) return true;
    const double ulp6 = 0.5 * std::pow(10.0, std::floor(std::log10(std::abs(row))) - 5);
    return std::abs(v - row) <= ulp6 * (1 + 1e-9);
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    std::error_code ec;
    fs::remove_all(kRoot, ec);
    fs::create_directories(kRoot);

    // (d) first: chained() runs the straight case too; (a)-(c) read it.
    chained(kRoot, "br", kArgs, 1.0, 2);
    const std::string p = kRoot + "/br/straight/r";
    const auto t = valtest::loadCsv(p + "_stats.csv");
    check(t.has_value(), "stats readable");
    if (!t) return dnstest::finish();

    // (a)
    const std::vector<std::string> finite = {
        "t",     "dt",      "dtChem", "dtDiff", "xTip",    "xBase",  "burn",
        "fuelIn", "Tmax",   "mass",   "massErr", "divErr", "eosErr", "pcgIt",
        "uMax",  "burnPre", "burnDif", "preFrac", "zMin",  "zMax",   "mdotIn",
        "mdotOut", "mdotLat", "uLatMin", "uLatMax", "xRimBase", "rBase", "qRim",
        "dtPos"};
    bool fin = true;
    for (const auto& c : finite)
        for (const double v : t->col(c))
            if (!std::isfinite(v)) {
                std::printf("    non-finite %s\n", c.c_str());
                fin = false;
                break;
            }
    check(fin, std::format("every key column finite in all {} rows", t->rows()));
    check(t->names.back() == "dtPos", "dtPos is the last stats column (--dt-rule cell)");
    double hits = 0, zmin = 1, zmax = 0, tmax = 0;
    int pmin = 99, pmax = 0;
    bool dtOk = true;
    for (std::size_t i = 0; i < t->rows(); ++i) {
        hits = std::max(hits, t->col("yHits")[i]);
        zmin = std::min(zmin, t->col("zMin")[i]);
        zmax = std::max(zmax, t->col("zMax")[i]);
        tmax = std::max(tmax, t->col("Tmax")[i]);
        pmin = std::min(pmin, int(t->col("pcgIt")[i]));
        pmax = std::max(pmax, int(t->col("pcgIt")[i]));
        dtOk = dtOk && t->col("dt")[i] <= t->col("dtPos")[i] * (1 + 1e-5);
    }
    std::printf("  br: %zu rows, Tmax %.5f, yHits %g, Z [%.2e, 1%+.2e], pcgIt "
                "[%d, %d], last dt %.4e dtPos %.4e, xRimBase %.4f, preFrac %.4f\n",
                t->rows(), tmax, hits, zmin, zmax - 1, pmin, pmax, t->col("dt").back(),
                t->col("dtPos").back(), t->col("xRimBase").back(),
                t->col("preFrac").back());
    check(dtOk, "dt <= dtPos in every row (the positivity bound binds or not)");
    check(pmin >= 1 && pmax <= 15, "pcgIt in [1, 15] (the tube's apertured PCG)");
    check(hits == 0 && zmin >= -1e-5 && zmax <= 1 + 1e-5,
          "bounded: yHits 0, Z in [-1e-5, 1 + 1e-5]");

    // (b)
    const std::size_t n = std::size_t(140) * 64 * 64;
    for (const char* f : {"temp", "fuel", "oxid", "omega", "findex", "speed"}) {
        const std::size_t sz = slurp(p + "_" + f + ".bin").size();
        check(sz == 5 * n * 4,
              std::format("{} volumes written (5 frames, {} bytes)", f, sz));
    }
    const auto xi = floats(p + "_findex.bin");
    bool in = xi.size() == 5 * n;
    for (const float v : xi) in = in && std::abs(v) <= 1.0f + 1e-6f;
    check(in, "findex in [-1, 1]");
    check(jsonHas(p, "\"rim\":{\"x\":") && jsonHas(p, "\"si\":{\"deltaD\":0.0001312,"),
          ".json carries the rim and si records");

    // (c) the last frame = the last stats row's state (t = tEnd).
    const auto om = floats(p + "_omega.bin");
    if (om.size() == 5 * n && xi.size() == 5 * n) {
        // h (delta_D) from the json's rim record, full precision.
        const std::string js = slurp(p + ".json");
        const std::size_t at = js.find("\"h\":");
        const double h = at == std::string::npos ? NAN : std::strtod(js.c_str() + at + 4, nullptr);
        double pre = 0, dif = 0;
        for (std::size_t c = 0; c < n; ++c) {
            const std::size_t q = 4 * n + c;
            const double w = dns::lmPremixedWeight(xi[q]);
            pre += float(double(om[q]) * w);
            dif += float(double(om[q]) * (1.0 - w));
        }
        const double dV = h * h * h;
        std::printf("  split: burnPre %.9g vs row %.6g, burnDif %.9g vs row %.6g "
                    "(dV %.6g)\n",
                    pre * dV, t->col("burnPre").back(), dif * dV,
                    t->col("burnDif").back(), dV);
        check(sameRow(pre * dV, t->col("burnPre").back()) &&
                  sameRow(dif * dV, t->col("burnDif").back()),
              "the last frame's lmPremixedWeight split = the row's burnPre/burnDif "
              "(6 digits)");
    }
    return dnstest::finish();
}
