#include "dns/checkpoint.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>

#include <fcntl.h>
#include <unistd.h>

namespace dns {

namespace {

static_assert(std::endian::native == std::endian::little,
              "the checkpoint format is little-endian");

constexpr char kMagic[8] = {'D', 'N', 'S', 'C', 'K', 'P', 'T', '\0'};
constexpr char kTrailer[8] = {'E', 'N', 'D', 'C', 'K', 'P', 'T', '\0'};

std::size_t elemSize(Checkpoint::Type t)
{
    switch (t) {
    case Checkpoint::Type::Text: return 1;
    case Checkpoint::Type::F64:
    case Checkpoint::Type::I64: return 8;
    case Checkpoint::Type::F32: return 4;
    }
    return 0;
}

const char* typeName(Checkpoint::Type t)
{
    switch (t) {
    case Checkpoint::Type::Text: return "text";
    case Checkpoint::Type::F64: return "f64";
    case Checkpoint::Type::I64: return "i64";
    case Checkpoint::Type::F32: return "f32";
    }
    return "?";
}

// Appends the raw bytes of a trivially copyable value.
template <class T>
void putRaw(std::vector<std::byte>& b, const T& v)
{
    const auto* p = reinterpret_cast<const std::byte*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}

std::string errnoText(const std::string& what)
{
    return what + ": " + std::strerror(errno);
}

// write(2) until done (EINTR-safe).
bool writeAll(int fd, const std::byte* p, std::size_t n)
{
    while (n > 0) {
        const ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += w;
        n -= std::size_t(w);
    }
    return true;
}

} // namespace

std::string checkpointPath(const std::string& prefix)
{
    return prefix + "_ckpt.bin";
}

std::string caseFlagsText(const CaseFlags& f)
{
    std::string s;
    for (const auto& [k, v] : f) {
        s += k;
        if (!v.empty()) s += " " + v;
        s += '\n';
    }
    return s;
}

CaseFlags parseCaseFlagsText(std::string_view text)
{
    CaseFlags f;
    while (!text.empty()) {
        const auto nl = text.find('\n');
        const std::string_view line = text.substr(0, nl);
        text.remove_prefix(nl == std::string_view::npos ? text.size() : nl + 1);
        if (line.empty()) continue;
        const auto sp = line.find(' ');
        if (sp == std::string_view::npos)
            f.emplace_back(std::string(line), std::string());
        else
            f.emplace_back(std::string(line.substr(0, sp)),
                           std::string(line.substr(sp + 1)));
    }
    return f;
}

std::optional<std::string> caseFlagsDiff(const CaseFlags& ckpt,
                                         const CaseFlags& run)
{
    // Both sorted by name: a merge walk, first difference by name.
    // A bare flag (no value) reads "given".
    auto show = [](const std::string& v) {
        return v.empty() ? std::string("given") : v;
    };
    std::size_t a = 0, b = 0;
    while (a < ckpt.size() || b < run.size()) {
        const bool takeA =
            b == run.size() || (a < ckpt.size() && ckpt[a].first < run[b].first);
        const bool takeB =
            a == ckpt.size() || (b < run.size() && run[b].first < ckpt[a].first);
        if (takeA)
            return std::format("{} {} vs (absent)", ckpt[a].first,
                               show(ckpt[a].second));
        if (takeB)
            return std::format("{} (absent) vs {}", run[b].first,
                               show(run[b].second));
        if (ckpt[a].second != run[b].second)
            return std::format("{} {} vs {}", ckpt[a].first,
                               show(ckpt[a].second), show(run[b].second));
        ++a;
        ++b;
    }
    return std::nullopt;
}

// --- Checkpoint ---------------------------------------------------------------

const Checkpoint::Record* Checkpoint::find(std::string_view name) const
{
    auto it = std::ranges::find_if(recs_, [&](const Record& r) { return r.name == name; });
    return it == recs_.end() ? nullptr : &*it;
}

Checkpoint::Record& Checkpoint::slot(std::string name, Type t)
{
    auto it = std::ranges::find_if(recs_, [&](const Record& r) { return r.name == name; });
    if (it == recs_.end()) {
        recs_.push_back(Record{std::move(name), t, 0, {}});
        return recs_.back();
    }
    it->type = t;
    it->bytes.clear();
    return *it;
}

void Checkpoint::putText(std::string name, std::string_view v)
{
    Record& r = slot(std::move(name), Type::Text);
    r.count = v.size();
    const auto* p = reinterpret_cast<const std::byte*>(v.data());
    r.bytes.assign(p, p + v.size());
}

void Checkpoint::putF64(std::string name, std::span<const double> v)
{
    Record& r = slot(std::move(name), Type::F64);
    r.count = v.size();
    const auto* p = reinterpret_cast<const std::byte*>(v.data());
    r.bytes.assign(p, p + v.size_bytes());
}

void Checkpoint::putI64(std::string name, std::span<const std::int64_t> v)
{
    Record& r = slot(std::move(name), Type::I64);
    r.count = v.size();
    const auto* p = reinterpret_cast<const std::byte*>(v.data());
    r.bytes.assign(p, p + v.size_bytes());
}

void Checkpoint::putF32(std::string name, std::span<const float> v)
{
    Record& r = slot(std::move(name), Type::F32);
    r.count = v.size();
    const auto* p = reinterpret_cast<const std::byte*>(v.data());
    r.bytes.assign(p, p + v.size_bytes());
}

bool Checkpoint::has(std::string_view name) const { return find(name) != nullptr; }

std::expected<std::string, std::string> Checkpoint::text(std::string_view name) const
{
    const Record* r = find(name);
    if (!r) return std::unexpected(std::format("checkpoint lacks '{}'", name));
    if (r->type != Type::Text)
        return std::unexpected(std::format("checkpoint '{}' is {}, not text",
                                           name, typeName(r->type)));
    return std::string(reinterpret_cast<const char*>(r->bytes.data()),
                       r->bytes.size());
}

std::expected<std::vector<double>, std::string>
Checkpoint::f64(std::string_view name) const
{
    const Record* r = find(name);
    if (!r) return std::unexpected(std::format("checkpoint lacks '{}'", name));
    if (r->type != Type::F64)
        return std::unexpected(std::format("checkpoint '{}' is {}, not f64",
                                           name, typeName(r->type)));
    std::vector<double> v(r->count);
    std::memcpy(v.data(), r->bytes.data(), r->bytes.size());
    return v;
}

std::expected<std::vector<std::int64_t>, std::string>
Checkpoint::i64(std::string_view name) const
{
    const Record* r = find(name);
    if (!r) return std::unexpected(std::format("checkpoint lacks '{}'", name));
    if (r->type != Type::I64)
        return std::unexpected(std::format("checkpoint '{}' is {}, not i64",
                                           name, typeName(r->type)));
    std::vector<std::int64_t> v(r->count);
    std::memcpy(v.data(), r->bytes.data(), r->bytes.size());
    return v;
}

std::expected<double, std::string> Checkpoint::f64Scalar(std::string_view name) const
{
    auto v = f64(name);
    if (!v) return std::unexpected(v.error());
    if (v->size() != 1)
        return std::unexpected(std::format("checkpoint '{}' holds {} values, not 1",
                                           name, v->size()));
    return (*v)[0];
}

std::expected<std::int64_t, std::string>
Checkpoint::i64Scalar(std::string_view name) const
{
    auto v = i64(name);
    if (!v) return std::unexpected(v.error());
    if (v->size() != 1)
        return std::unexpected(std::format("checkpoint '{}' holds {} values, not 1",
                                           name, v->size()));
    return (*v)[0];
}

std::expected<void, std::string>
Checkpoint::f32Into(std::string_view name, float* out, std::size_t n) const
{
    const Record* r = find(name);
    if (!r) return std::unexpected(std::format("checkpoint lacks '{}'", name));
    if (r->type != Type::F32)
        return std::unexpected(std::format("checkpoint '{}' is {}, not f32",
                                           name, typeName(r->type)));
    if (r->count != n)
        return std::unexpected(std::format(
            "checkpoint '{}' holds {} values, the run needs {}", name, r->count, n));
    if (r->bytes.size() != n * sizeof(float))
        return std::unexpected(std::format(
            "checkpoint '{}' was read header-only (no payload)", name));
    std::memcpy(out, r->bytes.data(), r->bytes.size());
    return {};
}

std::vector<std::string> Checkpoint::names(std::string_view prefix) const
{
    std::vector<std::string> v;
    for (const auto& r : recs_)
        if (r.name.starts_with(prefix)) v.push_back(r.name);
    std::ranges::sort(v);
    return v;
}

std::expected<void, std::string> Checkpoint::write(const std::string& path) const
{
    std::vector<std::byte> head;
    head.insert(head.end(), reinterpret_cast<const std::byte*>(kMagic),
                reinterpret_cast<const std::byte*>(kMagic) + 8);
    putRaw(head, kCheckpointVersion);
    putRaw(head, std::uint32_t(recs_.size()));

    const std::string tmp = path + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return std::unexpected(errnoText("cannot create '" + tmp + "'"));
    bool ok = writeAll(fd, head.data(), head.size());
    for (const auto& r : recs_) {
        if (!ok) break;
        std::vector<std::byte> h;
        putRaw(h, std::uint32_t(r.name.size()));
        const auto* np = reinterpret_cast<const std::byte*>(r.name.data());
        h.insert(h.end(), np, np + r.name.size());
        putRaw(h, std::uint32_t(r.type));
        putRaw(h, std::uint64_t(r.count));
        ok = writeAll(fd, h.data(), h.size()) &&
             writeAll(fd, r.bytes.data(), r.bytes.size());
    }
    if (ok)
        ok = writeAll(fd, reinterpret_cast<const std::byte*>(kTrailer), 8);
    if (ok) ok = ::fsync(fd) == 0;
    std::string werr = ok ? std::string() : errnoText("write failed on '" + tmp + "'");
    if (::close(fd) != 0 && ok) {
        ok = false;
        werr = errnoText("close failed on '" + tmp + "'");
    }
    if (!ok) {
        ::unlink(tmp.c_str());
        return std::unexpected(werr);
    }
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        const std::string e = errnoText("cannot rename '" + tmp + "' to '" + path + "'");
        ::unlink(tmp.c_str());
        return std::unexpected(e);
    }
    // The rename itself durable: fsync the directory.
    std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (dir.empty()) dir = ".";
    const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd >= 0) {
        (void)::fsync(dfd);
        ::close(dfd);
    }
    return {};
}

std::expected<Checkpoint, std::string> Checkpoint::read(const std::string& path,
                                                        bool headerOnly)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::unexpected(std::format("cannot read '{}'", path));
    auto bad = [&](const std::string& why) {
        return std::unexpected(std::format("'{}' is not a dns checkpoint: {}",
                                           path, why));
    };
    char magic[8];
    std::uint32_t version = 0, nrec = 0;
    if (!f.read(magic, 8) || std::memcmp(magic, kMagic, 8) != 0)
        return bad("bad magic");
    if (!f.read(reinterpret_cast<char*>(&version), 4) ||
        !f.read(reinterpret_cast<char*>(&nrec), 4))
        return bad("truncated header");
    if (version != kCheckpointVersion)
        return bad(std::format("format version {} (this binary reads {})",
                               version, kCheckpointVersion));
    Checkpoint ck;
    for (std::uint32_t i = 0; i < nrec; ++i) {
        std::uint32_t nl = 0, type = 0;
        std::uint64_t count = 0;
        if (!f.read(reinterpret_cast<char*>(&nl), 4) || nl > 4096)
            return bad(std::format("record {} truncated", i));
        std::string name(nl, '\0');
        if (!f.read(name.data(), nl) || !f.read(reinterpret_cast<char*>(&type), 4) ||
            !f.read(reinterpret_cast<char*>(&count), 8))
            return bad(std::format("record {} truncated", i));
        const auto t = Type(type);
        const std::size_t es = elemSize(t);
        if (es == 0) return bad(std::format("record '{}' has type {}", name, type));
        const std::uint64_t nbytes = count * es;
        Record r{name, t, count, {}};
        if (headerOnly && t == Type::F32) {
            f.seekg(std::streamoff(nbytes), std::ios::cur);
            if (!f) return bad(std::format("record '{}' truncated", name));
        } else {
            r.bytes.resize(nbytes);
            if (!f.read(reinterpret_cast<char*>(r.bytes.data()),
                        std::streamsize(nbytes)))
                return bad(std::format("record '{}' truncated", name));
        }
        ck.recs_.push_back(std::move(r));
    }
    char trailer[8];
    if (!f.read(trailer, 8) || std::memcmp(trailer, kTrailer, 8) != 0)
        return bad("missing end marker (a file cut short)");
    return ck;
}

// --- facts, output resume -----------------------------------------------------

std::expected<CheckpointFacts, std::string> checkpointFacts(const Checkpoint& ck)
{
    CheckpointFacts f;
    auto bin = ck.text("binary");
    auto sol = ck.text("solver");
    auto cs = ck.text("case");
    auto grid = ck.i64("grid");
    auto t = ck.f64Scalar("t");
    auto steps = ck.i64Scalar("steps");
    auto rs = ck.i64Scalar("restarts");
    if (!bin) return std::unexpected(bin.error());
    if (!sol) return std::unexpected(sol.error());
    if (!cs) return std::unexpected(cs.error());
    if (!grid) return std::unexpected(grid.error());
    if (!t) return std::unexpected(t.error());
    if (!steps) return std::unexpected(steps.error());
    if (!rs) return std::unexpected(rs.error());
    if (grid->size() != 4)
        return std::unexpected("checkpoint 'grid' must hold nx, ny, nz, nY");
    f.binary = *bin;
    f.solver = *sol;
    if (ck.has("mesh")) {
        auto m = ck.text("mesh");
        if (!m) return std::unexpected(m.error());
        f.mesh = *m;
    }
    f.caseFlags = parseCaseFlagsText(*cs);
    f.nx = int((*grid)[0]);
    f.ny = int((*grid)[1]);
    f.nz = int((*grid)[2]);
    f.nY = int((*grid)[3]);
    f.t = *t;
    f.steps = long(*steps);
    f.restarts = long(*rs);
    return f;
}

void putOutputResume(Checkpoint& ck, const OutputResume& at)
{
    for (const auto& [suffix, len] : at.files)
        ck.putI64("out.file:" + suffix, std::int64_t(len));
    ck.putI64("out.frames", at.frames);
    ck.putI64("out.pngIdx", at.pngIdx);
    ck.putF64("out.nextFrame", at.nextFrame);
    ck.putF64("out.nextPng", at.nextPng);
    ck.putF64("out.nextWeb", at.nextWeb);
}

std::expected<OutputResume, std::string> outputResume(const Checkpoint& ck)
{
    OutputResume at;
    const std::string key = "out.file:";
    for (const auto& n : ck.names(key)) {
        auto len = ck.i64Scalar(n);
        if (!len) return std::unexpected(len.error());
        at.files.emplace_back(n.substr(key.size()), std::uint64_t(*len));
    }
    auto fr = ck.i64Scalar("out.frames");
    auto pi = ck.i64Scalar("out.pngIdx");
    auto nf = ck.f64Scalar("out.nextFrame");
    auto np = ck.f64Scalar("out.nextPng");
    auto nw = ck.f64Scalar("out.nextWeb");
    for (const auto* e : {&fr, &pi})
        if (!*e) return std::unexpected(e->error());
    for (const auto* e : {&nf, &np, &nw})
        if (!*e) return std::unexpected(e->error());
    at.frames = int(*fr);
    at.pngIdx = int(*pi);
    at.nextFrame = *nf;
    at.nextPng = *np;
    at.nextWeb = *nw;
    return at;
}

} // namespace dns
