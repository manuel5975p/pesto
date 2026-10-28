// OpenFOAM polyMesh reader/writer (dns/polymesh, masterplan_meshsolver P1):
//   T1  the ASCII box equals its foamFormatConvert binary twin, bit for bit
//   T2  writePolyMesh -> readPolyMesh round trips (ascii/binary x label 32/64,
//       with cellZones), bit for bit, on every checked-in mesh
//   T3  the manifest needs no bulk data (points deleted: still reads)
//   T4  every rejection of the accepted-subset contract, each naming its token
#include "dns/polymesh.hpp"

#include "check.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dns;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

fs::path data(const char* name) { return fs::path(DNS_MESH_DATA) / name / "constant" / "polyMesh"; }

fs::path scratch(const std::string& tag)
{
    const auto d = fs::temp_directory_path() / ("dns_test_polymesh_" + tag);
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

std::string slurp(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

void spit(const fs::path& p, const std::string& s)
{
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << s;
}

// A fresh copy of box_cyclic (ASCII) to break one file of.
fs::path boxCopy(const std::string& tag)
{
    const auto d = scratch(tag);
    for (const auto& e : fs::directory_iterator(data("box_cyclic")))
        fs::copy_file(e.path(), d / e.path().filename());
    return d;
}

// Replace the first `from` in dir/file by `to`; false if absent.
bool edit(const fs::path& dir, const char* file, const std::string& from, const std::string& to)
{
    std::string s = slurp(dir / file);
    const auto at = s.find(from);
    if (at == std::string::npos) return false;
    s.replace(at, from.size(), to);
    spit(dir / file, s);
    return true;
}

// The case must fail and its message must contain `token`.
void rejects(const fs::path& dir, const std::string& token, const std::string& what)
{
    const auto m = readPolyMesh(dir);
    const bool ok = !m && m.error().find(token) != std::string::npos;
    check(ok, what + (m ? std::string(" [accepted!]") : " [" + m.error() + "]"));
}

bool sameExceptFormat(PolyMesh a, PolyMesh b)
{
    a.man.binary = b.man.binary;
    a.man.labelBits = b.man.labelBits;
    return a == b;
}

void t1AsciiBinary()
{
    std::printf("T1 ASCII vs binary box\n");
    const auto a = readPolyMesh(data("box_cyclic"));
    const auto b = readPolyMesh(data("box_cyclic_bin"));
    check(a.has_value(), "box_cyclic reads" + (a ? std::string() : ": " + a.error()));
    check(b.has_value(), "box_cyclic_bin reads" + (b ? std::string() : ": " + b.error()));
    if (!a || !b) return;
    check(!a->man.binary && b->man.binary, "formats detected (ascii, binary)");
    check(a->px == b->px && a->py == b->py && a->pz == b->pz, "points bitwise equal");
    check(a->faceStart == b->faceStart && a->faceNodes == b->faceNodes, "faces equal (faceList == faceCompactList)");
    check(a->owner == b->owner && a->neighbour == b->neighbour, "owner/neighbour equal");
    check(a->man.patches == b->man.patches, "patches equal, separations included");
    check(sameExceptFormat(*a, *b), "whole PolyMesh equal up to the format flags");
    check(a->man.nCells == 24 && a->man.nFaces == 98 && a->man.nInternalFaces == 46 && a->man.nPoints == 60,
          "counts 24 cells, 98 faces, 46 internal, 60 points");
    const auto& p = a->man.patches;
    check(p.size() == 6 && p[0].type == FoamPatchType::Cyclic && p[0].neighbourPatch == "xhi"
              && p[0].separation && (*p[0].separation)[0] == 1.0 && p[2].type == FoamPatchType::Wall,
          "xlo cyclic <-> xhi, separation (1 0 0); ylo wall");
}

void t2RoundTrip()
{
    std::printf("T2 write/read round trips\n");
    for (const char* name : {"box_cyclic", "box_cyclic_bin", "box_refined", "cyl2d_snappy"}) {
        auto m = readPolyMesh(data(name));
        if (!m) {
            check(false, std::string(name) + " reads: " + m.error());
            continue;
        }
        if (std::string(name) == "box_refined") {  // exercise cellZones too
            m->man.cellZones = {"left", "right"};
            m->zoneCells = {{0, 1, 2, 7, 9}, {20, 21, 30}};
        }
        for (const bool bin : {false, true})
            for (const int bits : {32, 64}) {
                const auto d = scratch(std::format("rt_{}_{}_{}", name, bin, bits));
                const auto w = writePolyMesh(d, *m, bin, bits);
                std::expected<PolyMesh, std::string> r = std::unexpected(w ? std::string() : w.error());
                if (w) r = readPolyMesh(d);
                const bool ok = r && r->man.binary == bin && r->man.labelBits == bits && sameExceptFormat(*r, *m);
                check(ok, std::format("{} {} label {}: bitwise round trip{}", name, bin ? "binary" : "ascii",
                                      bits, r ? "" : " (" + r.error() + ")"));
                fs::remove_all(d);
            }
    }
}

void t3Manifest()
{
    std::printf("T3 manifest without bulk data\n");
    const auto d = boxCopy("manifest");
    fs::remove(d / "points");
    const auto m = readPolyManifest(d);
    check(m.has_value(), "manifest reads with points deleted" + (m ? std::string() : ": " + m.error()));
    if (m) {
        check(m->nPoints == 60 && m->nCells == 24 && m->nFaces == 98 && m->nInternalFaces == 46,
              "manifest counts from the owner note");
        check(m->patches.size() == 6 && m->patches[0].name == "xlo" && m->patches[5].name == "zhi",
              "manifest patch names in boundary order");
        check(!m->binary && m->labelBits == 32, "manifest format ascii, label 32");
    }
    rejects(d, "points", "the full read names the missing points file");
    fs::remove_all(d);
}

void t4Rejections()
{
    std::printf("T4 rejections by name\n");
    {
        const auto d = boxCopy("gz");
        spit(d / "points.gz", "x");
        rejects(d, ".gz", "*.gz is rejected");
    }
    auto boundaryType = [](const char* tag, const std::string& type, const std::string& token,
                           const std::string& what) {
        const auto d = boxCopy(tag);
        const bool e = edit(d, "boundary", "type            wall;", "type            " + type + ";");
        check(e, std::string("(fixture edit ") + tag + ")");
        rejects(d, token, what);
    };
    boundaryType("proc", "processor", "reconstructParMesh", "processor patch: decomposed, run reconstructParMesh");
    boundaryType("ami", "cyclicAMI", "cyclicAMI", "cyclicAMI is rejected");
    boundaryType("acmi", "cyclicACMI", "cyclicACMI", "cyclicACMI is rejected");
    boundaryType("nc", "nonConformalCyclic", "nonConformal", "nonConformal* is rejected");
    boundaryType("mapped", "mappedWall", "mapped", "mapped* is rejected");
    boundaryType("unk", "fancyWall", "unsupported type 'fancyWall'", "an unknown patch type is rejected");
    {
        const auto d = boxCopy("rot");
        edit(d, "boundary", "transform       translational;", "transform       rotational;");
        rejects(d, "rotational", "cyclic transform rotational is rejected");
    }
    {
        const auto d = boxCopy("include");
        edit(d, "boundary", "    ylo\n", "    #include \"other\"\n    ylo\n");
        rejects(d, "#include", "#include is rejected");
    }
    {
        const auto d = boxCopy("calc");
        edit(d, "boundary", "nFaces          8;", "nFaces          #calc \"4*2\";");
        rejects(d, "#calc", "#calc is rejected");
    }
    {
        const auto d = boxCopy("ownnei");
        check(edit(d, "neighbour", "(\n1\n", "(\n0\n"), "(fixture edit ownnei)");
        rejects(d, "owner >= neighbour", "owner >= neighbour on an internal face is rejected");
    }
    {
        const auto d = boxCopy("range");
        check(edit(d, "faces", "4(1 6 26 21)", "4(1 6 26 9999)"), "(fixture edit range)");
        rejects(d, "out of range", "a point label out of range is rejected");
    }
    {
        const auto d = boxCopy("cellrange");
        check(edit(d, "owner", "(\n0\n", "(\n24\n"), "(fixture edit cellrange)");
        rejects(d, "out of range", "a cell label out of range is rejected");
    }
    {
        const auto d = boxCopy("small");
        check(edit(d, "faces", "4(1 6 26 21)", "2(1 6)"), "(fixture edit small)");
        rejects(d, "< 3", "a face with < 3 nodes is rejected");
    }
    {
        const auto d = boxCopy("contig");
        check(edit(d, "boundary", "startFace       66;", "startFace       67;"), "(fixture edit contig)");
        rejects(d, "starts at face", "patches not contiguous are rejected");
    }
    {
        const auto d = boxCopy("pair");
        edit(d, "boundary", "neighbourPatch  xlo;", "neighbourPatch  zlo;");
        rejects(d, "naming it back", "a cyclic whose neighbour does not name it back is rejected");
    }
    {
        const auto d = boxCopy("zones");
        spit(d / "cellZones",
             "FoamFile { version 2.0; format ascii; class regIOobject; object cellZones; }\n"
             "2 ( a { type cellZone; cellLabels List<label> 3(0 1 2); } "
             "b { type cellZone; cellLabels List<label> 2(2 3); } )\n");
        rejects(d, "overlapping", "overlapping cellZones are rejected");
    }
    {
        const auto d = boxCopy("trunc");
        std::string s = slurp(d / "owner");
        spit(d / "owner", s.substr(0, s.find("(\n") + 40));
        rejects(d, "truncated", "a truncated list is rejected");
    }
    {
        const auto d = boxCopy("scalar32");
        edit(d, "points", "scalar=64", "scalar=32");
        rejects(d, "only 64-bit", "a scalar=32 mesh is rejected");
    }
    {   // label 64 values past int32 are rejected, not wrapped.
        const auto m = readPolyMesh(data("box_cyclic"));
        const auto d = scratch("label64");
        if (m && writePolyMesh(d, *m, true, 64)) {
            std::string s = slurp(d / "owner");
            const auto at = s.find("98\n(");
            const std::int64_t big = std::int64_t(1) << 40;
            s.replace(at + 4, 8, reinterpret_cast<const char*>(&big), 8);
            spit(d / "owner", s);
            rejects(d, "out of range (int32", "a label64 value past int32 is rejected");
        } else
            check(false, "(fixture label64)");
    }
    {
        const auto d = scratch("trunc_bin");
        for (const auto& e : fs::directory_iterator(data("box_cyclic_bin")))
            fs::copy_file(e.path(), d / e.path().filename());
        std::string s = slurp(d / "points");
        spit(d / "points", s.substr(0, s.find("60\n(") + 100));
        rejects(d, "truncated", "truncated binary points are rejected");
    }
    {
        const auto d = scratch("missing");
        rejects(d, "missing polyMesh file", "an empty directory names the missing file");
    }
}

} // namespace

int main()
{
    t1AsciiBinary();
    t2RoundTrip();
    t3Manifest();
    t4Rejections();
    for (const auto& e : fs::directory_iterator(fs::temp_directory_path()))
        if (e.path().filename().string().starts_with("dns_test_polymesh_")) fs::remove_all(e.path());
    return dnstest::finish();
}
