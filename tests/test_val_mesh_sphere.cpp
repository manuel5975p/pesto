// Validation (heavy, analytic): mesh-loader fusion test. A generated
// unit-sphere STL driven through mesh_tunnel_gpu must reproduce the analytic
// sphere_tunnel_gpu drag at matched Re 300 / n = 96 / dr = 8: windowed cd over
// the last 3 flow-throughs within 5%. Measured 0.7099 vs 0.6851 (3.6%),
// INDEPENDENT of faceting (subdiv 4 and subdiv 5 give identical cd to 1e-4):
// the gap is the supersampled-sdf-band cut vs the exact-overlap geometry at
// D = 12 cells, where the first-order cut wall is O(h/D) ~ 8% per pipeline;
// aperture-measured qRef contributes only -0.15%.
#include "mesh_test_util.hpp"
#include "validation_util.hpp"

#include "dns/gpu/device_util.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>

namespace {

// Windowed cd from a tunnel-app stats file; cleans up the run's outputs.
double windowedCd(const std::string& prefix, double t0, double t1)
{
    auto tab = valtest::loadCsv(prefix + "_stats.csv");
    if (!tab) { std::printf("FAIL: no stats csv (%s)\n", prefix.c_str()); std::exit(1); }
    const double cd = valtest::windowMean(tab->col("t"), tab->col("cd"), t0, t1);
    for (const char* s : {"_stats.csv", ".json", "_vort.bin", "_speed.bin", "_mask.bin"})
        std::remove((prefix + s).c_str());
    return cd;
}

} // namespace

int main()
{
    using std::numbers::pi;
    if (!dns::gpu::deviceAvailable()) valtest::skip("no HIP device");

    const std::string scratch = DNS_VAL_SCRATCH;
    const std::string stl = scratch + "/val_unit_sphere.stl";
    meshtest::writeBinaryStl(stl, meshtest::icosphere(1.0, {0.0, 0.0, 0.0}, 4));

    const double tEnd = 6.0 * 2 * pi; // ~37.7
    const double t0 = tEnd - 3.0 * 2 * pi;
    const std::string args = " 96 --re 300 --dr 8 --ft 6 --fdt 1e9 --out ";

    const int rcMesh = valtest::runBinary(std::string(MESH_BIN) + " " + stl + args
                                          + scratch + "/mesh_sph");
    valtest::check(rcMesh == 0, "mesh_tunnel_gpu exits 0");
    const int rcSph = valtest::runBinary(std::string(SPHERE_BIN) + args
                                         + scratch + "/ana_sph");
    valtest::check(rcSph == 0, "sphere_tunnel_gpu exits 0");

    const double cdMesh = windowedCd(scratch + "/mesh_sph", t0, tEnd);
    const double cdSph = windowedCd(scratch + "/ana_sph", t0, tEnd);
    const double rel = std::abs(cdMesh - cdSph) / cdSph;
    std::printf("cd mesh=%.4f analytic=%.4f rel diff=%.4f\n", cdMesh, cdSph, rel);
    valtest::check(std::isfinite(rel) && rel < 0.05,
                   "mesh-STL cd within 5% of analytic-sphere cd");

    std::remove(stl.c_str());
    return valtest::finish();
}
