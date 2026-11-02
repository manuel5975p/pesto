// Host side of the time-accurate incompressible FV solver (docs/fvincomp.md):
// the --bc table, its binding to a mesh, the setup tables, the CLI rejections.
#include "dns/cli.hpp"
#include "dns/config.hpp"
#include "dns/fvincomp.hpp"
#include "dns/la/graph.hpp"
#include "dns/polymesh.hpp"

#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dns;
using dnstest::check;

namespace {

const char* const kBc = R"(
[reference]
u = 2.0
length = 0.5
area = 0.1
[initial]
u = [2, 0, 0]
[patch.inlet]
kind = "inflow"
u = [2, 0, 0]
[patch.outlet]
kind = "outflow"
p = 0.5
[patch.lower]
kind = "slip"
[patch.upper]
kind = "far"
u = [2, 0, 0]
[patch.cylinder]
kind = "wall"
omega = [0, 0, 1]
origin = [0, 0, 0]
force = true
)";

bool hasError(const std::string& text, std::string_view needle)
{
    auto r = parseFvIncompBcs(text);
    if (r) return false;
    const bool ok = r.error().find(needle) != std::string::npos;
    if (!ok) std::printf("    got: %s\n", r.error().c_str());
    return ok;
}

// argv through the real parser + validator: the exit code of the first failure, 0 if valid.
int codeOf(std::vector<const char*> a)
{
    a.insert(a.begin(), "dns");
    auto raw = dns::cli::parse(int(a.size()), const_cast<char**>(a.data()));
    if (!raw) return 1;
    auto c = validate(*raw);
    return c ? 0 : c.error().code;
}

void parsing()
{
    std::printf("--bc table\n");
    auto b = parseFvIncompBcs(kBc);
    check(b.has_value(), "the table parses");
    if (!b) return;
    check(b->uRef == 2.0 && b->length == 0.5 && b->area == 0.1 && b->u0[0] == 2.0, "reference and initial");
    check(b->patches.size() == 5, "five patches");
    check(hasError("[patch.a]\nkind = \"wal\"\n", "kind"), "unknown kind rejected by name");
    check(hasError("[patch.a]\nu = [1, 0, 0]\n", "needs kind"), "a missing kind rejected");
    check(hasError("[patch.a]\nkind = \"wall\"\nsplat = 1\n", "unknown key 'splat'"), "unknown key rejected");
    check(hasError("[patch.a]\nkind = \"slip\"\nomega = [0, 0, 1]\n", "omega"), "omega on a slip rejected");
    check(hasError("[patch.a]\nkind = \"slip\"\nu = [1, 0, 0]\n", "u belongs"), "u on a slip rejected");
    check(hasError("[patch.a]\nkind = \"wall\"\np = 1\n", "p belongs"), "p on a wall rejected");
    check(hasError("[patch.a]\nkind = \"convective\"\nuc = -1\n", "uc"), "a negative uc rejected");
    check(hasError("[reference]\nu = -1\n[patch.a]\nkind = \"wall\"\n", "reference.u"), "reference.u > 0");
    check(hasError("[wrong]\n[patch.a]\nkind = \"wall\"\n", "unknown key 'wrong'"), "an unknown table rejected");
    check(hasError("x = \n", "parse error"), "a TOML error rejected");
}

void binding()
{
    std::printf("setup on cyl2d_snappy\n");
    auto pm = readPolyMesh(std::string(DNS_MESH_DATA) + "/cyl2d_snappy/constant/polyMesh");
    check(pm.has_value(), "the mesh reads");
    if (!pm) return;
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Morton);
    check(fm.has_value(), "the FaceMesh builds");
    if (!fm) return;
    auto b = parseFvIncompBcs(kBc);
    if (!b) return;
    auto miss = *b;
    miss.patches.pop_back();
    check(!bindFvIncompBcs(miss, *fm).has_value(), "a mesh patch without an entry rejected");
    auto extra = *b;
    extra.patches.push_back({.name = "nothere", .kind = FvIncompBc::Wall});
    check(!bindFvIncompBcs(extra, *fm).has_value(), "an entry naming no patch rejected");
    auto bad = *b;
    std::ranges::find_if(bad.patches, [](const FvIncompPatch& q) { return q.name == "cylinder"; })->omega = {1, 0, 0};
    check(!bindFvIncompBcs(bad, *fm).has_value(), "a 2D wall rotating off the empty axis rejected");
    auto ok = bindFvIncompBcs(*b, *fm);
    check(ok.has_value(), "binding");
    if (!ok) return;
    auto s = buildFvIncompSetup(*fm, *b);
    check(s.has_value(), "the setup builds");
    if (!s) return;
    const std::size_t nI = std::size_t(s->nI), nB = std::size_t(s->nB), nC = std::size_t(s->nC);
    check(!s->orthogonal && !s->pureNeumann, "a snappy mesh with an outflow");
    // 2D: every offset and weight along the empty axis is an exact 0 (w stays 0).
    const std::size_t z = std::size_t(s->emptyAxis);
    bool exact0 = true;
    for (std::size_t f = 0; f < nI; ++f)
        exact0 = exact0 && s->rO[z * nI + f] == 0 && s->S[z * nI + f] == 0 && s->d[z * nI + f] == 0 &&
                 s->k[z * nI + f] == 0 && s->gN[z * nI + f] == 0;
    for (std::size_t b2 = 0; b2 < nB; ++b2) exact0 = exact0 && s->bR[z * nB + b2] == 0 && s->bS[z * nB + b2] == 0;
    check(exact0, "2D: every emptyAxis offset, area and weight is exactly 0");
    // The pressure matrix: symmetric, A 1 = fixed-p faces only.
    double rowMax = 0, sumFix = 0;
    std::vector<double> row(s->diagA);
    const auto g = la::graphOf(*fm);
    for (std::size_t c = 0; c < nC; ++c)
        for (int e = g.rowPtr[c]; e < g.rowPtr[c + 1]; ++e) row[c] += s->slotA[std::size_t(e)];
    for (std::size_t b2 = 0; b2 < nB; ++b2)
        if (s->bCode[b2] == std::uint8_t(FvIncompFaceBc::Outflow)) {
            row[std::size_t(s->bOwn[b2])] -= s->bA[b2];
            sumFix += s->bA[b2];
        }
    for (double v : row) rowMax = std::max(rowMax, std::abs(v));
    check(rowMax < 1e-12 * sumFix, "A 1 = the fixed-p faces' coefficient");
    // far = inflow on the side the velocity enters.
    int in = 0, out = 0;
    for (std::size_t b2 = 0; b2 < nB; ++b2) {
        if (s->bCode[b2] == std::uint8_t(FvIncompFaceBc::Inflow)) ++in;
        if (s->bCode[b2] == std::uint8_t(FvIncompFaceBc::Outflow)) ++out;
    }
    check(in > 0 && out > 0, "inflow and outflow faces exist");
    // Least-squares weights reproduce a linear field exactly in the interior.
    double worst = 0;
    const double slope[3] = {0.3, -0.7, 0.0};
    for (std::size_t c = 0; c < nC; ++c) {
        double gr[3] = {0, 0, 0};
        for (int e = s->cellStart[c]; e < s->cellStart[c + 1]; ++e) {
            const int code = s->cellEnt[std::size_t(e)];
            for (std::size_t i = 0; i < 3; ++i) {
                double dp = 0, w = 0;
                for (std::size_t q = 0; q < 3; ++q) {
                    if (code >= 0) {
                        const std::size_t f = std::size_t(code >> 1);
                        const double sg = (code & 1) ? -1.0 : 1.0;
                        dp += sg * s->d[q * nI + f] * slope[q];
                    } else {
                        dp += s->bR[q * nB + std::size_t(-code - 1)] * slope[q];
                    }
                }
                if (code >= 0) {
                    const std::size_t f = std::size_t(code >> 1);
                    w = (code & 1) ? s->gN[i * nI + f] : s->gO[i * (nI + nB) + f];
                } else
                    w = s->gO[i * (nI + nB) + nI + std::size_t(-code - 1)];
                gr[i] += w * dp;
            }
        }
        for (std::size_t i = 0; i < 3; ++i) worst = std::max(worst, std::abs(gr[i] - slope[i]));
    }
    std::printf("  linear-field gradient error %.2e\n", worst);
    check(worst < 1e-11, "least-squares weights exact for a linear field");
    // CSR is ascending: the deterministic gather order.
    bool asc = true;
    for (std::size_t c = 0; c < nC; ++c) {
        int last = -1;
        for (int e = s->cellStart[c]; e < s->cellStart[c + 1]; ++e) {
            const int code = s->cellEnt[std::size_t(e)];
            const int face = code >= 0 ? (code >> 1) : s->nI - code;
            asc = asc && face > last;
            last = face;
        }
    }
    check(asc, "CSR: ascending faces, internal before boundary");
}

void periodicBox()
{
    std::printf("periodic boxes\n");
    const auto g = Grid::make(8, 6, 4, 8.0, 6.0, 4.0);
    auto fm = fvPeriodicBox(g, FvPerturb::None, 0);
    check(fm.has_value() && fm->nB == 0, "the uniform box has no boundary");
    if (!fm) return;
    auto s = buildFvIncompSetup(*fm, FvIncompBcs{});
    check(s.has_value() && s->orthogonal && s->pureNeumann && s->nI == 3 * 192, "orthogonal, pure Neumann, 3 N faces");
    auto r1 = fvPeriodicBox(g, FvPerturb::Random, 0.2, 7);
    auto r2 = fvPeriodicBox(g, FvPerturb::Random, 0.2, 7);
    auto r3 = fvPeriodicBox(g, FvPerturb::Random, 0.2, 8);
    check(r1.has_value() && r2.has_value() && *r1 == *r2, "the hash perturbation is deterministic");
    check(r3.has_value() && !(*r1 == *r3), "the seed moves it");
    auto rs = r1 ? buildFvIncompSetup(*r1, FvIncompBcs{}) : std::unexpected(std::string("no mesh"));
    check(rs.has_value() && !rs->orthogonal, "a perturbed box is non-orthogonal");
    const auto g2 = Grid::make(8, 6, 1, 8.0, 6.0, 1.0);
    auto f2 = fvPeriodicBox(g2, FvPerturb::Sine, 0.04);
    check(f2.has_value() && f2->emptyAxis == 2, "nz 1 is an `empty` extrusion");
}

void cliFlags()
{
    std::printf("CLI\n");
    check(codeOf({"tgv", "--solver", "fv", "--n", "16", "--dry-run"}) == 0, "tgv --solver fv");
    check(codeOf({"tgv", "--solver", "fv", "--mesh-perturb", "0.1", "--mesh-perturb-kind", "random"}) == 0,
          "tgv on a random mesh");
    check(codeOf({"tgv", "--mesh-perturb", "0.01"}) == 2, "--mesh-perturb needs --solver fv");
    check(codeOf({"tgv", "--les", "0.1"}) == 2, "--les needs --solver fv");
    check(codeOf({"tgv", "--solver", "fv", "--mesh-perturb", "0.1"}) == 2, "sine amp > 0.06 rejected");
    check(codeOf({"tgv", "--solver", "fv", "--nz", "1", "--les", "1", "--sgs", "sigma"}) == 2, "2D sigma rejected");
    check(codeOf({"flow", "--solver", "fv", "--body", "cylinder"}) == 2, "--solver fv without a body-fitted mesh rejected");
    check(codeOf({"flow", "--solver", "fv", "--ogrid", "--body", "cylinder", "--n", "64", "--ny", "32", "--re", "40"}) == 0,
          "--solver fv --ogrid");
    check(codeOf({"flow", "--solver", "fv", "--ogrid", "--body", "cylinder", "--n", "64", "--ny", "32", "--mach", "0.1"}) == 2,
          "a compressible flag rejected");
    check(codeOf({"flow", "--ogrid", "--body", "cylinder", "--n", "64", "--ny", "32"}) == 2,
          "--ogrid without --solver fv stays compressible-only");
    check(codeOf({"flow", "--polymesh", "x", "--bc", "y.toml", "--re", "10", "--ft", "1"}) == 0, "--polymesh + --bc");
    check(codeOf({"flow", "--polymesh", "x", "--solver", "fv", "--re", "10", "--ft", "1"}) == 2, "--solver fv needs --bc");
    check(codeOf({"flow", "--bc", "y.toml", "--re", "10"}) == 2, "--bc needs --polymesh");
    check(codeOf({"flow", "--polymesh", "x", "--bc", "y.toml", "--far", "tunnel", "--re", "10", "--ft", "1"}) == 2,
          "a structured flag rejected");
    check(codeOf({"flow", "--polymesh", "x"}) == 0, "--polymesh CASE alone stays the simpleFoam path");
    check(codeOf({"flow", "--polymesh", "x", "--re", "10"}) == 2, "... with its own rejection of every other flag");
}

} // namespace

int main()
{
    parsing();
    binding();
    periodicBox();
    cliFlags();
    {
        auto t = parseFvIncompBcs("[patch.a]\nkind = \"wall\"\nt = 0.5\n[patch.b]\nkind = \"slip\"\n");
        check(t.has_value() && t->patches[0].hasT && t->patches[0].T == 0.5 && !t->patches[1].hasT, "wall t parses");
        check(!parseFvIncompBcs("[patch.a]\nkind = \"slip\"\nt = 0.5\n"), "t belongs to a wall");
    }
    return dnstest::finish();
}
