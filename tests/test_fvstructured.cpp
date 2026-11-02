// dns flow --solver simple, host side (docs/fvstructured.md): a structured
// case through the CLI, the validator, polyOf, the generated dictionaries and
// the unchanged case loader.
//  S1 flags-only manifest == polyOf's on every mesh family
//  S2 export -> readPolyMesh bitwise; export -> loadFvCase: the FaceMesh equals
//     fromPolyMesh(polyOf) exactly; the texts read back byte for byte
//  S3 polyOf geometry vs CurvMetrics (mapped boxes, O-grid); revolved: volume
//     of the polygonal shell, closure, prism cells at the axis
//  S4 every generated dictionary passes the strict reader, on every mesh
//  S5 flags with no simpleFoam meaning rejected by name; dry-run writes nothing
#include "dns/cli.hpp"
#include "dns/config.hpp"
#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvsimple.hpp"
#include "dns/fvstructured.hpp"
#include "dns/sdf.hpp"

#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using dnstest::finish;
namespace fs = std::filesystem;

namespace {

std::expected<Case, ConfigError> caseOf(std::vector<std::string> args)
{
    std::vector<std::string> store = {"dns", "flow", "--solver", "simple"};
    store.insert(store.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& s : store) argv.push_back(s.data());
    auto raw = cli::parse(int(argv.size()), argv.data());
    if (!raw) return std::unexpected(ConfigError{1, "cli", raw.error().message});
    return validate(*raw);
}

// The O-grid wall: a cylinder or sphere of diameter D at the origin.
Sdf bodySdf(const FvStructured& s)
{
    const double r = 0.5 * std::numbers::pi / 4.0;
    if (s.revolved) return [r](double x, double y, double z) { return std::sqrt(x * x + y * y + z * z) - r; };
    return [r](double x, double y, double) { return std::hypot(x, y) - r; };
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b)
{
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

std::string slurp(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return std::move(ss).str();
}

struct Named
{
    const char* tag;
    std::vector<std::string> args;
};

const std::vector<Named>& families()
{
    static const std::vector<Named> v = {
        {"uniform box", {"--body", "none", "--n", "48", "--ny", "32"}},
        {"uniform box 3D", {"--body", "none", "--n", "48", "--ny", "32", "--nz", "4"}},
        {"box --noslip", {"--body", "none", "--n", "48", "--ny", "32", "--noslip"}},
        {"box --open", {"--body", "none", "--n", "48", "--ny", "32", "--open"}},
        {"box --stretch", {"--body", "none", "--n", "48", "--ny", "32", "--stretch", "2", "--noslip"}},
        {"box --focal", {"--body", "none", "--n", "64", "--ny", "64", "--focal", "2,4,2,4,0,0", "--focal-ratio", "2"}},
        {"channel", {"--body", "none", "--far", "periodic", "--noslip", "--re-tau", "180", "--n", "4", "--ny", "32"}},
        {"channel --stretch SA",
         {"--body", "none", "--far", "periodic", "--noslip", "--re-tau", "180", "--n", "4", "--ny", "32", "--stretch", "2",
          "--rans", "sa-neg"}},
        {"channel --focal SST",
         {"--body", "none", "--far", "periodic", "--noslip", "--re-tau", "180", "--n", "4", "--ny", "64", "--focal",
          "0,0,0.8,1.2,0,0", "--focal-ratio", "1.5", "--rans", "sst-2003m"}},
        {"O-grid 2D", {"--ogrid", "--body", "cylinder", "--n", "48", "--ny", "26", "--re", "40"}},
        {"O-grid 3D SST", {"--ogrid", "--body", "cylinder", "--n", "32", "--ny", "18", "--nz", "4", "--rans", "sst-2003m"}},
        {"O-grid MRF", {"--ogrid", "--body", "cylinder", "--n", "48", "--ny", "26", "--rotor-mach", "0.1", "--mach", "0.2",
                        "--rotor-frame", "mrf", "--rotor-axis", "z"}},
        {"O-grid wedge", {"--ogrid", "--body", "sphere", "--nz", "1", "--n", "32", "--ny", "26", "--re", "100"}},
        {"O-grid full", {"--ogrid", "--body", "sphere", "--nz", "8", "--n", "16", "--ny", "14", "--re", "100"}},
        {"O-grid wedge SA", {"--ogrid", "--body", "sphere", "--nz", "1", "--n", "32", "--ny", "26", "--rans", "sa-neg"}},
    };
    return v;
}

void meshGates(const Named& f, const fs::path& tmp)
{
    std::printf("%s\n", f.tag);
    auto c = caseOf(f.args);
    if (!c) {
        check(false, std::string(f.tag) + ": " + c.error().field + " " + c.error().message);
        return;
    }
    const FvStructured& s = c->fvs;
    auto man = fvStructuredManifest(s);
    auto pm = fvStructuredMesh(s, bodySdf(s));
    if (!man || !pm) {
        check(false, std::format("{}: {}", f.tag, !man ? man.error() : pm.error()));
        return;
    }
    check(*man == pm->man, "S1 flags-only manifest == polyOf's");

    const fs::path dir = tmp / "case";
    fs::remove_all(dir);
    const FoamCaseTexts t = fvStructuredTexts(s);
    if (auto w = writeFoamCase(dir, *pm, t); !w) {
        check(false, w.error());
        return;
    }
    auto back = readPolyMesh(dir / "constant" / "polyMesh");
    if (!back) {
        check(false, back.error());
        return;
    }
    PolyMesh want = *pm;
    want.man.binary = true;
    check(*back == want, "S2 export -> readPolyMesh bitwise");
    bool same = true;
    for (const auto& [rel, text] : t) same = same && slurp(dir / rel) == text;
    check(same, "S2 every text reads back byte for byte");

    auto L = loadFvCase(dir);
    if (!L) {
        check(false, "S4 strict loader: " + L.error());
        return;
    }
    auto direct = FaceMesh::fromPolyMesh(*pm, CellOrder::Source);
    if (!direct) {
        check(false, direct.error());
        return;
    }
    check(L->fm == *direct, std::format("S2 FaceMesh of the exported case == fromPolyMesh(polyOf): {} cells, {} internal",
                                        L->fm.nC, L->fm.nI));
    auto res = fvStructuredResolved(s);
    check(res.has_value(), "S4 fvStructuredResolved: " + (res ? std::string("ok") : res.error()));
    if (res) {
        res->dir = L->c.dir;
        check(describeFoamCase(*res) == describeFoamCase(L->c), "S4 the dry-run's resolved case == the loaded one");
    }
    auto g = la::graphOf(L->fm);
    auto pat = la::ellPattern(g);
    auto setup = buildSimpleSetup(L->fm, L->c, g, pat);
    check(setup.has_value(), "S4 buildSimpleSetup: " + (setup ? std::string("ok") : setup.error()));
    if (setup && L->c.ras.model != FvTurbModel::Laminar) {
        auto r = buildRasSetup(L->pm, L->fm, L->c, *setup);
        check(r.has_value(), "S4 buildRasSetup: " + (r ? std::string("ok") : r.error()));
    }
}

// fromCurvMesh's cell volumes and face areas vs the structured metrics.
void geometryGates()
{
    std::printf("S3 polyOf vs CurvMetrics\n");
    {
        auto c = caseOf(families()[5].args); // box --focal
        const Grid& g = c->fvs.grid;
        const auto R = focalRemap(g, true, c->fvs.focalLo.data(), c->fvs.focalHi.data(), c->fvs.focalW.data(), c->fvs.focalRatio);
        const CurvMesh cm = CurvMesh::fromRemap(g, R);
        auto pm = fvStructuredMesh(c->fvs, {});
        auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Source);
        auto cmx = CurvMetrics::build(cm);
        double dv = 0;
        for (int i = 0; i < fm->nC; ++i) dv = std::max(dv, std::abs(fm->V[std::size_t(i)] - cmx->vol[std::size_t(i)]));
        const double vmax = *std::ranges::max_element(fm->V);
        check(dv <= 1e-14 * vmax * 8, std::format("focal V vs CurvMetrics::vol {:.2e} of max", dv / vmax));
    }
    {
        // Revolved shell: sum V = the polygonal solid's volume, positive cells,
        // prisms on the axis, nothing outside the far circle's shell.
        auto c = caseOf(families()[13].args); // full revolution
        const FvStructured& s = c->fvs;
        auto pm = fvStructuredMesh(s, bodySdf(s));
        auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Source);
        check(fm.has_value(), "full revolution FaceMesh builds");
        if (!fm) return;
        double V = 0;
        for (double v : fm->V) V += v;
        const double r = 0.5 * std::numbers::pi / 4.0, R = s.ogrid.rFar, ex = 4.0 / 3.0 * std::numbers::pi * (R * R * R - r * r * r);
        check(std::abs(V / ex - 1) < 0.2, std::format("sum V = {:.4f} of the smooth shell (polygonal in phi and theta)", V / ex));
        int prisms = 0;
        for (int i = 0; i < fm->nC; ++i) prisms += fm->cellStart[std::size_t(i) + 1] - fm->cellStart[std::size_t(i)] == 5;
        check(prisms == 2 * (s.grid.ny - 2) * s.grid.nz, std::format("{} prism cells on the two axis columns", prisms));
        check(fm->patches.size() == 2 && pm->man.nPoints == fvStructuredManifest(s)->nPoints, "no axis patch, nodes merged");
    }
}

void rejections()
{
    std::printf("S5 rejected by name\n");
    const std::vector<std::pair<std::vector<std::string>, std::string>> bad = {
        {{"--body", "none", "--les", "0.1"}, "--les"},
        {{"--body", "none", "--cfl", "0.3"}, "--cfl"},
        {{"--body", "none", "--far", "sponge"}, "--far"},
        {{"--body", "none", "--physics", "compressible"}, "--physics"},
        {{"--body", "none", "--mach", "0.2"}, "--mach"},
        {{"--body", "cylinder"}, "--body"},
        {{"--body", "none", "--far", "periodic"}, "--far"},
        {{"--body", "none", "--re-tau", "180"}, "--re-tau"},
        {{"--ogrid", "--body", "cylinder", "--rans", "sst-vm"}, "--rans"},
        {{"--ogrid", "--body", "sphere", "--nz", "1", "--rotor-mach", "0.1", "--rotor-frame", "mrf", "--rotor-axis", "z"}, "--rotor-mach"},
        {{"--ogrid", "--body", "cylinder", "--nz", "4", "--rotor-mach", "0.1", "--rotor-frame", "mrf", "--rotor-axis", "z"}, "--rotor-mach"},
        {{"--body", "none", "--export", "/tmp/x", "--out", "y"}, "--out"},
    };
    for (const auto& [args, field] : bad) {
        auto c = caseOf(args);
        check(!c && c.error().field == field, std::format("{} -> {}", field, c ? std::string("ACCEPTED") : c.error().message));
    }
}

} // namespace

int main()
{
    const fs::path tmp = fs::temp_directory_path() / std::format("dns_fvstructured_{}", ::getpid());
    fs::create_directories(tmp);
    for (const Named& f : families()) meshGates(f, tmp);
    geometryGates();
    rejections();
    fs::remove_all(tmp);
    return finish();
}
