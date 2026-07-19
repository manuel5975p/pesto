// Validation (analytic band): mesh insertion into the 3D compressible solver.
// A generated unit-sphere STL driven through comp_mesh_tunnel_gpu at Re 100 /
// M 0.2 must settle to a physical sphere drag. Windowed cd over the last third
// of the run lands in a blockage-consistent band around the Re 100 sphere value
// (literature cd ~ 1.09; the confined free-slip tunnel at dr = 6 raises it, as
// documented for the incompressible blockage bands), with near-zero transverse
// force (symmetry). Complements the library-level gpu_comp_mesh parity test.
#include "mesh_test_util.hpp"
#include "validation_util.hpp"

#include "dns/gpu/device_util.hpp"

#include <cmath>
#include <cstdio>
#include <string>

int main()
{
    if (!dns::gpu::deviceAvailable()) valtest::skip("no HIP device");

    const std::string scratch = DNS_VAL_SCRATCH;
    const std::string stl = scratch + "/val_comp_sphere.stl";
    meshtest::writeBinaryStl(stl, meshtest::icosphere(1.0, {0.0, 0.0, 0.0}, 3));

    const std::string prefix = scratch + "/comp_sph";
    const std::string cmd = std::string(COMP_MESH_BIN) + " " + stl +
                            " 64 --nz 64 --re 100 --mach 0.2 --dr 6 --ft 2 "
                            "--fdt 1e9 --out " + prefix;
    const int rc = valtest::runBinary(cmd);
    valtest::check(rc == 0, "comp_mesh_tunnel_gpu exits 0");

    auto tab = valtest::loadCsv(prefix + "_stats.csv");
    if (!tab) { std::printf("FAIL: no stats csv\n"); return 1; }
    const double tEnd = 2.0 * 2 * M_PI, t0 = tEnd - (2.0 / 3.0) * 2 * M_PI;
    const double cd = valtest::windowMean(tab->col("t"), tab->col("cd"), t0, tEnd);
    const double cl = valtest::windowMean(tab->col("t"), tab->col("cl"), t0, tEnd);
    const double cs = valtest::windowMean(tab->col("t"), tab->col("cs"), t0, tEnd);
    std::printf("windowed cd=%.4f cl=%+.4f cs=%+.4f\n", cd, cl, cs);

    valtest::check(std::isfinite(cd) && cd > 0.8 && cd < 1.8,
                   "compressible sphere cd in blockage-consistent band");
    valtest::check(std::abs(cl) < 0.1 && std::abs(cs) < 0.1,
                   "transverse force small (symmetry)");

    for (const char* s : {"_stats.csv", ".json", "_curl.bin", "_speed.bin", "_lam.bin"})
        std::remove((prefix + s).c_str());
    std::remove(stl.c_str());
    return valtest::finish();
}
