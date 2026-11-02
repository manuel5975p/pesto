// Shared by the compressible FV gates (test_gpu_fv_comp, test_gpu_fv_cross):
// meshes, boundary tables, smooth states and norms.
#pragma once

#include "check.hpp"
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvref.hpp"
#include "dns/gpu/fv_comp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <functional>
#include <numbers>
#include <string>
#include <vector>

namespace dnstest::fvc {

using namespace dns;
using gpu::DeviceFvComp;
using gpu::FvCompConfig;

inline constexpr double kGamma = 1.4;

template <class T>
T orDie(std::expected<T, std::string> e, const std::string& what)
{
    if (!e) {
        std::printf("FATAL: %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
    return std::move(*e);
}

inline void orDie(std::expected<void, std::string> e, const std::string& what)
{
    if (!e) {
        std::printf("FATAL: %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
}

#ifdef DNS_MESH_DATA
inline FaceMesh loadMesh(const char* name, CellOrder order = CellOrder::Morton)
{
    namespace fs = std::filesystem;
    const PolyMesh pm = orDie(readPolyMesh(fs::path(DNS_MESH_DATA) / name / "constant/polyMesh"), name);
    return orDie(FaceMesh::fromPolyMesh(pm, order), name);
}
#endif

// Cyclic on x and y, z cyclic (3D) or empty (nz == 1).
inline StructuredSpec periodicSpec(const Grid& g)
{
    const FoamPatchType cz = g.nz == 1 ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    return {.lo = {0, 0, 0},
            .hi = {g.nx - 1, g.ny - 1, g.nz - 1},
            .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
            .type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic, cz,
                     cz}};
}

inline std::vector<FvBc> bcFor(const FaceMesh& fm, const std::function<FvBc(const MeshPatch&)>& pick)
{
    std::vector<FvBc> v;
    for (const MeshPatch& p : fm.patches) {
        if (p.type == FoamPatchType::Empty) {
            FvBc e;
            e.kind = FvBcKind::Empty;
            v.push_back(e);
        } else
            v.push_back(pick(p));
    }
    return v;
}

inline void setPrim(FvState& s, std::size_t c, double rho, const double u[3], double p)
{
    s.var(0)[c] = rho;
    s.var(1)[c] = rho * u[0];
    s.var(2)[c] = rho * u[1];
    s.var(3)[c] = rho * u[2];
    s.var(4)[c] = p / (kGamma - 1.0) + 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
}

// A smooth positive state on any mesh; uz false: w = 0 (2D).
inline FvState smoothState(const FaceMesh& fm, bool uz, double m = 1.0)
{
    FvState s = FvState::zeros(fm.nC);
    double lo[3], hi[3];
    for (std::size_t a = 0; a < 3; ++a) {
        const auto [mn, mx] = std::ranges::minmax(fm.C[a]);
        lo[a] = mn;
        hi[a] = std::max(mx, mn + 1e-30);
    }
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double X = (fm.C[0][c] - lo[0]) / (hi[0] - lo[0]), Y = (fm.C[1][c] - lo[1]) / (hi[1] - lo[1]);
        const double Z = (fm.C[2][c] - lo[2]) / (hi[2] - lo[2]);
        const double rho = 1.0 + 0.2 * std::sin(5 * X + 1) * std::cos(4 * Y) * std::cos(3 * Z + 0.5);
        const double u[3] = {m * (0.3 + 0.2 * std::cos(4 * X) * std::sin(5 * Y + 0.3)),
                             m * 0.2 * std::sin(3 * X + 4 * Y),
                             uz ? m * 0.15 * std::sin(4 * Z + 1) * std::cos(3 * X) : 0.0};
        setPrim(s, c, rho, u, 1.0 + 0.1 * std::cos(6 * X) * std::cos(5 * Y + 1) * std::sin(2 * Z + 0.2));
    }
    return s;
}

// Max over rows of max|a - b| / max|b| (a zero row counts absolute).
inline double relDiff(const std::vector<double>& a, const std::vector<double>& b, int nV, int nC)
{
    double worst = 0.0;
    for (std::size_t v = 0; v < std::size_t(nV); ++v) {
        double e = 0.0, sc = 0.0;
        for (std::size_t c = 0; c < std::size_t(nC); ++c) {
            const std::size_t i = v * std::size_t(nC) + c;
            e = std::max(e, std::abs(a[i] - b[i]));
            sc = std::max(sc, std::abs(b[i]));
        }
        worst = std::max(worst, sc == 0.0 ? e : e / sc);
    }
    return worst;
}

inline double maxAbs(const std::vector<double>& a)
{
    double m = 0.0;
    for (const double v : a) m = std::max(m, std::abs(v));
    return m;
}

} // namespace dnstest::fvc
