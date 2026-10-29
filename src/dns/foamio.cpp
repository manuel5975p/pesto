#include "dns/foamio.hpp"

#include <charconv>
#include <cstdio>
#include <format>
#include <fstream>
#include <sstream>
#include <system_error>

namespace dns {
namespace fs = std::filesystem;

namespace {

void putNumber(std::string& o, double v, int prec)
{
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::general, prec);
    o.append(buf, r.ptr);
}

void putValue(std::string& o, const double* v, int nc, int prec)
{
    if (nc == 1) {
        putNumber(o, v[0], prec);
        return;
    }
    o += '(';
    for (int q = 0; q < nc; ++q) {
        if (q) o += ' ';
        putNumber(o, v[q], prec);
    }
    o += ')';
}

void putList(std::string& o, const std::vector<double>& v, int nc, int prec)
{
    const std::size_t n = v.size() / std::size_t(nc);
    o += std::format("nonuniform List<{}> \n{}\n(\n", nc == 1 ? "scalar" : "vector", n);
    for (std::size_t i = 0; i < n; ++i) {
        putValue(o, v.data() + i * std::size_t(nc), nc, prec);
        o += '\n';
    }
    o += ")\n";
}

} // namespace

std::expected<void, std::string> writeFoamField(const fs::path& dir, const std::string& object, int nc, bool surface,
                                                const std::string& dims, const std::vector<double>& internal,
                                                const std::vector<FoamPatchOut>& patches, int precision)
{
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return std::unexpected(dir.string() + ": cannot create (" + ec.message() + ")");
    const char* cls = surface ? "surfaceScalarField" : (nc == 1 ? "volScalarField" : "volVectorField");
    std::string o;
    o.reserve(internal.size() * 24 + 4096);
    o += std::format("FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    arch        \"LSB;label=32;scalar=64\";\n"
                     "    class       {};\n    location    \"{}\";\n    object      {};\n}}\n\n",
                     cls, dir.filename().string(), object);
    o += "dimensions      " + dims + ";\n\ninternalField   ";
    putList(o, internal, nc, precision);
    o += ";\n\nboundaryField\n{\n";
    for (const FoamPatchOut& p : patches) {
        o += "    " + p.name + "\n    {\n";
        if (!p.constraint.empty()) o += "        type            " + p.constraint + ";\n";
        else {
            o += "        type            calculated;\n        value           ";
            putList(o, p.values, nc, precision);
            o += ";\n";
        }
        o += "    }\n";
    }
    o += "}\n";
    const fs::path file = dir / object;
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    if (!f) return std::unexpected(file.string() + ": cannot create");
    f.write(o.data(), std::streamsize(o.size()));
    f.close();
    if (!f) return std::unexpected(file.string() + ": write failed");
    return {};
}

std::expected<std::vector<double>, std::string> readFoamInternalField(const fs::path& file, int nc,
                                                                      std::int64_t nExpect)
{
    std::ifstream f(file, std::ios::binary);
    if (!f) return std::unexpected(file.string() + ": cannot read");
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string t = std::move(ss).str();
    auto bad = [&](const std::string& w) { return std::unexpected(file.string() + ": " + w); };
    if (t.find("format      binary") != std::string::npos || t.find("format binary") != std::string::npos)
        return bad("binary field files are not supported (foamFormatConvert to ascii)");
    std::size_t p = t.find("internalField");
    if (p == std::string::npos) return bad("no internalField");
    p += 13;
    auto ws = [&] {
        while (p < t.size() && (t[p] == ' ' || t[p] == '\n' || t[p] == '\t' || t[p] == '\r')) ++p;
    };
    auto num = [&](double& v) -> bool {
        ws();
        const auto r = std::from_chars(t.data() + p, t.data() + t.size(), v);
        if (r.ec != std::errc{}) return false;
        p = std::size_t(r.ptr - t.data());
        return true;
    };
    auto expect = [&](char c) -> bool {
        ws();
        if (p >= t.size() || t[p] != c) return false;
        ++p;
        return true;
    };
    auto one = [&](std::vector<double>& out) -> bool {
        if (nc == 1) {
            double v{};
            if (!num(v)) return false;
            out.push_back(v);
            return true;
        }
        if (!expect('(')) return false;
        for (int q = 0; q < nc; ++q) {
            double v{};
            if (!num(v)) return false;
            out.push_back(v);
        }
        return expect(')');
    };
    ws();
    std::vector<double> out;
    if (t.compare(p, 7, "uniform") == 0) {
        p += 7;
        std::vector<double> v;
        if (!one(v)) return bad("malformed uniform internalField");
        for (std::int64_t i = 0; i < nExpect; ++i) out.insert(out.end(), v.begin(), v.end());
        return out;
    }
    if (t.compare(p, 10, "nonuniform") != 0) return bad("internalField is neither uniform nor nonuniform");
    p = t.find('>', p);
    if (p == std::string::npos) return bad("malformed nonuniform List");
    ++p;
    double n{};
    if (!num(n)) return bad("nonuniform list without a count");
    if (std::int64_t(n) != nExpect)
        return bad(std::format("internalField has {} values, expected {}", std::int64_t(n), nExpect));
    if (!expect('(')) return bad("nonuniform list without '('");
    out.reserve(std::size_t(nExpect) * std::size_t(nc));
    for (std::int64_t i = 0; i < nExpect; ++i)
        if (!one(out)) return bad(std::format("malformed value {}", i));
    if (!expect(')')) return bad("nonuniform list not closed");
    return out;
}

} // namespace dns
