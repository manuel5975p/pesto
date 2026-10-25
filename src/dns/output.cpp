#include "dns/output.hpp"

#include "dns/fieldpng.hpp"
#include "dns/vtk_writer.hpp"
#include "dns/shmframes.hpp"
#include "dns/webframes.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>

namespace dns {

namespace {

bool has(const std::set<Emit>& s, Emit e) { return s.contains(e); }

// "png_"/"web_" tag onto the LAST path component of the prefix: an --out with
// directories ("/tmp/dns/chan512") must keep its outputs together
// ("/tmp/dns/png_chan512"), not grow a literal "png_/tmp/..." tree under the
// CWD (measured: a channel run quietly filed its frames inside build/).
std::string tagPrefix(const std::string& tag, const std::string& prefix)
{
    const std::filesystem::path p(prefix);
    return (p.parent_path() / (tag + p.filename().string())).string();
}

// Closes explicitly and only then reports success: a buffered ENOSPC surfaces
// at close, so testing the stream while it is still open reports success on a
// truncated file.
std::expected<void, std::string> closeChecked(std::ofstream& f,
                                              const std::string& path)
{
    f.close();
    if (!f) return std::unexpected(std::format("write failed on '{}'", path));
    return {};
}

// A web plane: which (i,j,k) the (x,y) pixel of the plane image maps to.
struct WebPlane
{
    std::unique_ptr<WebFrames> wf;
    std::function<std::array<int, 3>(int, int)> sel;
    int a = 0, b = 0;
};

} // namespace

struct RunOutput::Impl
{
    Config cfg;
    RunMeta meta;
    Grid g{1, 1, 1, 1, 1, 1};
    FrameGeometry geo;

    // <prefix>_<field>.bin (mid-plane or volume) and the slice triples.
    std::map<std::string, std::ofstream> bins;
    std::map<std::string, std::ofstream> series;
    std::vector<WebPlane> webs;
    // The live shared-memory bridge (--emit shm). Lazily created on the first
    // frame, when the field list is finally known.
    ShmFrames shm;
    bool shmReady = false;

    std::string pngDir;
    double nextFrame = 0, nextPng = 0, nextWeb = 0;
    int frames = 0, pngIdx = 0, vtkIdx = 0;
    double msPerStep = 0;
    bool moved = false;

    // The first write failure of the run, reported once on stderr when it
    // happens and returned by finish(). The run itself continues: a full disk
    // must not lose a run that is otherwise healthy, but it must not exit 0.
    std::optional<std::string> firstError;
    std::optional<std::expected<void, std::string>> finished;

    void record(const std::string& e)
    {
        if (firstError) return;
        firstError = e;
        std::fprintf(stderr, "dns: output error: %s (the run continues; it "
                             "will exit 4)\n", e.c_str());
    }

    // Writes one binary record to a frame stream and records a failure.
    void writeBin(const std::string& name, const std::vector<float>& data)
    {
        auto& out = binFor(name);
        out.write(reinterpret_cast<const char*>(data.data()),
                  std::streamsize(data.size() * sizeof(float)));
        if (!out)
            record(std::format("write failed on '{}_{}.bin'", cfg.prefix, name));
    }

    std::vector<float> buf;   // scratch plane / volume buffer

    std::string json() const
    {
        // "rans" is an appended field, present only on a RANS run, so every
        // other run's .json reads exactly as it did.
        return std::format(
            "{{\"nx\":{},\"ny\":{},\"nz\":{},\"frames\":{},\"frameDt\":{},"
            "\"D\":{},\"re\":{},\"mach\":{},\"cs\":{},\"les\":{},\"sgs\":\"{}\","
            "\"body\":\"{}\",\"far\":\"{}\",\"physics\":\"{}\",\"kick\":{},"
            "\"msPerStep\":{}{}}}\n",
            meta.nx, meta.ny, meta.nz, frames, meta.frameDt, meta.D, meta.re,
            meta.mach, meta.cs, meta.les ? 1 : 0, meta.sgs, meta.body,
            meta.farField, meta.physics, meta.kick, msPerStep,
            meta.rans.empty() ? std::string()
                              : ",\"rans\":\"" + meta.rans + "\"");
    }

    void writeJson() const
    {
        std::ofstream f(cfg.prefix + ".json");
        f << json();
    }

    std::expected<void, std::string> writeJsonChecked() const
    {
        const std::string path = cfg.prefix + ".json";
        std::ofstream f(path);
        f << json();
        return closeChecked(f, path);
    }

    // Frame streams open on first use -- their names come from the runner's
    // field list, which phase 3c does not have. What phase 3c DOES prove is
    // that the destination directory is writable (the .json probe), so the
    // only failure left here is a mid-run one (disk full, quota, unlink). Say
    // so once per stream instead of silently writing to a dead stream, which
    // is what this did before.
    std::ofstream& binFor(const std::string& name)
    {
        auto it = bins.find(name);
        if (it != bins.end()) return it->second;
        auto& f = bins[name];
        const std::string path = cfg.prefix + "_" + name + ".bin";
        f.open(path, std::ios::binary);
        if (!f)
            std::fprintf(stderr, "dns: warning: cannot open '%s'; those frames "
                                 "will not be written\n", path.c_str());
        return f;
    }

    // The three plane selectors: which (i,j,k) the (x,y) pixel of an xy, xz or
    // yz plane image reads, in the WRITTEN window's coordinates. Every caller
    // that cuts a plane -- .bin, slices, PNG, web -- asks for one of these;
    // they used to be written out inline at each of those sites.
    auto xySel(int kk) const
    {
        return [i0 = geo.i0, j0 = geo.j0, kk](int x, int y) {
            return std::array{i0 + x, j0 + y, kk};
        };
    }
    auto xzSel(int jj) const
    {
        return [i0 = geo.i0, k0 = geo.k0, jj](int x, int y) {
            return std::array{i0 + x, jj, k0 + y};
        };
    }
    auto yzSel(int ii) const
    {
        return [j0 = geo.j0, k0 = geo.k0, ii](int x, int y) {
            return std::array{ii, j0 + x, k0 + y};
        };
    }

    // Lazily built: the field list, and so the plane manifest, is only known
    // at the first frame. Its own method rather than an immediately-invoked
    // lambda inside frame(), which is what it was.
    std::expected<void, std::string>
    buildWebPlanes(std::span<const FrameField> fields, int nx, int ny, int nz)
    {
        Impl& p = *this;
        const int per = std::clamp(p.cfg.planesPerAxis, 1, 3);
        const int dI = std::max(1, nx / 16), dJ = std::max(1, ny / 16),
                  dK = std::max(1, nz / 16);
        std::string idx;
        bool failed = false;   // a static mask that did not write
        auto add = [&](const std::string& suf, const std::string& axis,
                       double offD, int a, int b, auto sel) {
            WebPlane wp;
            char offs[40];
            std::snprintf(offs, sizeof offs, "%+.3gD", offD);
            wp.wf = std::make_unique<WebFrames>(
                tagPrefix("web_", p.cfg.prefix) + suf, a, b,
                p.cfg.simJson + ",\"plane\":\"" + axis + "\",\"off\":\"" +
                    (offD == 0.0 ? "mid" : offs) + "\"}");
            for (const auto& f : fields)
                wp.wf->addField(f.name,
                                f.diverging ? "diverging" : "sequential",
                                f.diverging ? -f.hi : f.lo, f.hi);
            wp.sel = sel;
            wp.a = a;
            wp.b = b;
            if (!p.geo.mask.empty())
            {
                std::vector<float> m;
                p.gather(p.geo.mask.data(), a, b, wp.sel, m);
                if (!wp.wf->writeStatic("mask", m.data(), 0.0, 1.0))
                    failed = true;
            }
            char e[160];
            std::snprintf(e, sizeof e,
                          "%s{\"suf\":\"%s\",\"axis\":\"%s\",\"off\":%.4g}",
                          idx.empty() ? "" : ",", suf.c_str(),
                          axis.c_str(), offD);
            idx += e;
            p.webs.push_back(std::move(wp));
        };
        const double D = p.geo.D > 0 ? p.geo.D : 1.0;
        auto cellsD = [&](int c) { return c * p.g.hx / D; };
        const int ic = p.geo.ic, jc = p.geo.jc, kc = p.geo.kc;
        const std::string maskErr = std::format(
            "web static mask write failed under '{}'",
            tagPrefix("web_", p.cfg.prefix));
        add("", "xy", 0.0, nx, ny, xySel(kc));
        if (nz == 1)                        // native 2D: one plane only
            return failed ? std::unexpected(maskErr)
                          : std::expected<void, std::string>{};
        if (per == 3 && nz >= 16)
        {
            add("_xy_m", "xy", -cellsD(dK), nx, ny, xySel(kc - dK));
            add("_xy_p", "xy", +cellsD(dK), nx, ny, xySel(kc + dK));
        }
        add("_xz", "xz", 0.0, nx, nz, xzSel(jc));
        if (per == 3 && ny >= 16)
        {
            add("_xz_m", "xz", -cellsD(dJ), nx, nz, xzSel(jc - dJ));
            add("_xz_p", "xz", +cellsD(dJ), nx, nz, xzSel(jc + dJ));
        }
        add("_yz", "yz", 0.0, ny, nz, yzSel(ic));
        if (per == 3 && nx >= 16)
        {
            add("_yz_m", "yz", -cellsD(dI), ny, nz, yzSel(ic - dI));
            add("_yz_p", "yz", +cellsD(dI), ny, nz, yzSel(ic + dI));
        }
        if (failed) return std::unexpected(maskErr);
        const std::string path =
            tagPrefix("web_", p.cfg.prefix) + "/planes.json";
        std::ofstream pf(path);
        pf << "{\"planes\":[" << idx << "]}\n";
        return closeChecked(pf, path);
    }

    // Copies the plane selected by `sel` out of a full-grid field.
    void gather(const float* src, int a, int b,
                const std::function<std::array<int, 3>(int, int)>& sel,
                std::vector<float>& out) const
    {
        out.resize(std::size_t(a) * b);
        std::size_t q = 0;
        for (int y = 0; y < b; ++y)
            for (int x = 0; x < a; ++x, ++q)
            {
                const auto [i, j, k] = sel(x, y);
                out[q] = src[g.idx(i, j, k)];
            }
    }
};

RunOutput::RunOutput() : p_(std::make_unique<Impl>()) {}
RunOutput::RunOutput(RunOutput&&) noexcept = default;
RunOutput& RunOutput::operator=(RunOutput&&) noexcept = default;

RunOutput::~RunOutput()
{
    if (!p_ || p_->finished) return;      // moved from, or already finished
    for (auto& [n, f] : p_->bins) f.close();
    for (auto& [n, f] : p_->series) f.close();
    p_->writeJson();                      // final frame count + ms/step
}

int RunOutput::frames() const { return p_->frames; }
void RunOutput::setMsPerStep(double ms) { p_->msPerStep = ms; }

void RunOutput::updateMask(std::span<const float> mask)
{
    // A size mismatch would silently mis-index every plane gather, so it is a
    // programmer error the caller cannot reach by configuration: the mask is
    // always the full grid the run was made with.
    assert(mask.size() == p_->geo.mask.size());
    if (mask.size() != p_->geo.mask.size()) return;
    p_->geo.mask.assign(mask.begin(), mask.end());
}

std::expected<RunOutput, std::string>
RunOutput::make(Config cfg, RunMeta meta, const Grid& g, FrameGeometry geo)
{
    if (cfg.prefix.empty())
        return std::unexpected("output prefix is empty");
    // PNG at nz > 1 renders the centre z-plane (the gather below slices kc,
    // same as the bin path). Every subcommand allows it now; whether that
    // plane is the meaningful one is the caller's judgement (for a channel
    // the x-y mid-plane IS the picture).
    if (cfg.pdt <= 0) cfg.pdt = cfg.fdt;
    if (cfg.wdt <= 0) cfg.wdt = cfg.fdt;

    RunOutput o;
    auto& p = *o.p_;
    p.cfg = std::move(cfg);
    p.meta = std::move(meta);
    p.g = g;
    p.geo = std::move(geo);
    if (p.geo.w <= 0) p.geo.w = g.nx - 2 * p.geo.i0;
    if (p.geo.h <= 0) p.geo.h = g.ny - 2 * p.geo.j0;
    if (p.geo.d <= 0) p.geo.d = g.nz - 2 * p.geo.k0;

    // Phase 3c: create and truncate now, so a full or read-only destination is
    // discovered in the first second rather than after hours of GPU. The .json
    // probe is what establishes that the directory is writable at all; the
    // mask, the PNG directory and (via declareSeries) every stats CSV follow
    // here. Frame .bin streams are named by the runner's field list and open
    // on first use -- see binFor().
    {
        const auto dir = std::filesystem::path(p.cfg.prefix).parent_path();
        std::error_code ec;
        if (!dir.empty()) std::filesystem::create_directories(dir, ec);
        std::ofstream probe(p.cfg.prefix + ".json");
        if (!probe)
            return std::unexpected(
                std::format("cannot write '{}.json'", p.cfg.prefix));
    }
    p.writeJson();

    if (!p.geo.mask.empty())
    {
        const int W = p.geo.w, H = p.geo.h;
        std::vector<float> m(std::size_t(W) * H);
        for (int j = 0; j < H; ++j)
            for (int i = 0; i < W; ++i)
                m[std::size_t(j) * W + i] =
                    p.geo.mask[g.idx(p.geo.i0 + i, p.geo.j0 + j, p.geo.kc)];
        const std::string path = p.cfg.prefix + "_mask.bin";
        std::ofstream mf(path, std::ios::binary);
        mf.write(reinterpret_cast<const char*>(m.data()),
                 std::streamsize(m.size() * sizeof(float)));
        if (auto r = closeChecked(mf, path); !r) return std::unexpected(r.error());
    }

    if (has(p.cfg.kinds, Emit::Png))
    {
        p.pngDir = tagPrefix("png_", p.cfg.prefix);
        std::error_code ec;
        std::filesystem::create_directories(p.pngDir, ec);
        if (ec) return std::unexpected(
            std::format("cannot create '{}': {}", p.pngDir, ec.message()));
    }
    return o;
}

namespace {

std::string auxPath(const std::string& prefix, std::string_view kind, int index,
                    std::string_view ext)
{
    return prefix + "_" + std::string(kind) + "_" + std::to_string(index)
         + std::string(ext);
}

} // namespace

std::expected<void, std::string>
RunOutput::auxText(std::string_view kind, int index, std::string_view ext,
                   std::string_view content)
{
    const std::string path = auxPath(p_->cfg.prefix, kind, index, ext);
    std::ofstream f(path);
    f << content;
    return closeChecked(f, path);
}

std::expected<void, std::string>
RunOutput::auxBinary(std::string_view kind, int index, std::string_view ext,
                     std::span<const float> data)
{
    const std::string path = auxPath(p_->cfg.prefix, kind, index, ext);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),
            std::streamsize(data.size() * sizeof(float)));
    return closeChecked(f, path);
}

std::expected<void, std::string>
RunOutput::declareSeries(std::string_view name, std::span<const std::string> cols)
{
    if (!has(p_->cfg.kinds, Emit::Stats)) return {};
    const std::string key(name);
    const std::string path = p_->cfg.prefix + "_" + key + ".csv";
    auto& f = p_->series[key];
    // Re-declaring a series reopens it: opening an already-open stream would
    // set failbit and silently swallow every subsequent row.
    if (f.is_open()) f.close();
    f.clear();
    f.open(path);
    if (!f) return std::unexpected(std::format("cannot write '{}'", path));
    for (std::size_t i = 0; i < cols.size(); ++i)
        f << (i ? "," : "") << cols[i];
    f << '\n';
    return {};
}

bool RunOutput::row(std::string_view name, std::span<const double> v)
{
    auto it = p_->series.find(std::string(name));
    if (it == p_->series.end()) return true;    // series not enabled
    auto& f = it->second;
    for (std::size_t i = 0; i < v.size(); ++i)
        f << (i ? "," : "") << v[i];
    f << '\n';
    f.flush();
    if (f) return true;
    p_->record(std::format("write failed on series '{}'", name));
    return false;
}

std::expected<void, std::string> RunOutput::finish()
{
    auto& p = *p_;
    if (p.finished) return *p.finished;
    std::optional<std::string> err = p.firstError;
    for (auto& [n, f] : p.bins)
        if (auto r = closeChecked(f, p.cfg.prefix + "_" + n + ".bin"); !r && !err)
            err = r.error();
    for (auto& [n, f] : p.series)
        if (auto r = closeChecked(f, p.cfg.prefix + "_" + n + ".csv"); !r && !err)
            err = r.error();
    if (auto r = p.writeJsonChecked(); !r && !err) err = r.error();
    p.finished = err ? std::expected<void, std::string>(std::unexpected(*err))
                     : std::expected<void, std::string>{};
    return *p.finished;
}

bool RunOutput::wantsFrame(double t) const
{
    const auto& k = p_->cfg.kinds;
    return (has(k, Emit::Bin) || has(k, Emit::Volume) || has(k, Emit::Slices))
           && t >= p_->nextFrame;
}
bool RunOutput::wantsPng(double t) const
{
    return has(p_->cfg.kinds, Emit::Png) && t >= p_->nextPng;
}
bool RunOutput::wantsWeb(double t) const
{
    return has(p_->cfg.kinds, Emit::Web) && t >= p_->nextWeb;
}
bool RunOutput::wantsShm(double t) const
{
    return has(p_->cfg.kinds, Emit::Shm) && t >= p_->nextWeb;
}

bool RunOutput::wantsAnyFrame(double t) const
{
    return wantsFrame(t) || wantsPng(t) || wantsWeb(t) || wantsShm(t);
}

std::expected<void, std::string>
RunOutput::frame(double t, std::span<const FrameField> fields)
{
    auto r = frameKinds(t, fields);
    if (!r) p_->record(r.error());
    return r;
}

std::expected<void, std::string>
RunOutput::frameKinds(double t, std::span<const FrameField> fields)
{
    auto& p = *p_;
    const auto& k = p.cfg.kinds;
    const bool doBin = wantsFrame(t), doPng = wantsPng(t), doWeb = wantsWeb(t);
    const bool doShm = wantsShm(t);
    if (!doBin && !doPng && !doWeb && !doShm) return {};

    const int nx = p.geo.w, ny = p.geo.h, nz = p.geo.d;
    const int i0 = p.geo.i0, j0 = p.geo.j0, k0 = p.geo.k0;

    // Lazily build the web planes: the field list is only known here.
    if (doWeb && p.webs.empty())
        if (auto r = p.buildWebPlanes(fields, nx, ny, nz); !r) return r;

    if (doBin)
    {
        for (const auto& f : fields)
        {
            if (has(k, Emit::Volume))
            {
                p.buf.resize(std::size_t(nx) * ny * nz);
                std::size_t q = 0;
                for (int kk = 0; kk < nz; ++kk)
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i, ++q)
                            p.buf[q] = f.data[p.g.idx(i0 + i, j0 + j, k0 + kk)];
                p.writeBin(f.name, p.buf);
            }
            if (has(k, Emit::Bin))
            {
                p.gather(f.data, nx, ny, p.xySel(p.geo.kc), p.buf);
                p.writeBin(f.name, p.buf);
            }
            if (has(k, Emit::Slices))
            {
                const int ic = p.geo.ic, jc = p.geo.jc, kc = p.geo.kc;
                auto plane = [&](const char* suf, int a, int b, auto sel) {
                    p.gather(f.data, a, b, sel, p.buf);
                    p.writeBin(std::string(suf) + "_" + f.name, p.buf);
                };
                plane("xy", nx, ny, p.xySel(kc));
                plane("xz", nx, nz, p.xzSel(jc));
                plane("yz", ny, nz, p.yzSel(ic));
            }
        }
        ++p.frames;
        p.nextFrame += p.cfg.fdt;
    }

    if (doPng)
    {
        for (const auto& f : fields)
        {
            p.gather(f.data, nx, ny, p.xySel(p.geo.kc), p.buf);
            const auto path = std::format("{}/{}_{:05d}.png", p.pngDir, f.name,
                                          p.pngIdx);
            std::vector<float> m;
            if (!p.geo.mask.empty())
                p.gather(p.geo.mask.data(), nx, ny, p.xySel(p.geo.kc), m);
            // Optional box-average coarsening of the PNG frame (--png-down).
            // The field averages over each s*s block; a block is rendered solid
            // only if ALL its cells are solid, so the body outline never bleeds.
            int pw = nx, ph = ny;
            const float* pf = p.buf.data();
            const float* pm = m.empty() ? nullptr : m.data();
            std::vector<float> dbuf, dmask;
            const int s = p.cfg.pngDown;
            if (s > 1 && nx % s == 0 && ny % s == 0)
            {
                pw = nx / s; ph = ny / s;
                dbuf.assign(std::size_t(pw) * ph, 0.0f);
                if (pm) dmask.assign(std::size_t(pw) * ph, 1.0f);
                const float inv = 1.0f / float(s * s);
                for (int J = 0; J < ph; ++J)
                    for (int I = 0; I < pw; ++I)
                    {
                        float acc = 0.0f, msolid = 0.0f;
                        for (int dj = 0; dj < s; ++dj)
                            for (int di = 0; di < s; ++di)
                            {
                                const std::size_t c = std::size_t(J * s + dj) * nx
                                                    + (I * s + di);
                                acc += p.buf[c];
                                if (pm && pm[c] < 0.5f) msolid += 1.0f;
                            }
                        dbuf[std::size_t(J) * pw + I] = acc * inv;
                        if (pm)
                            dmask[std::size_t(J) * pw + I] =
                                msolid >= float(s * s) ? 0.0f : 1.0f;
                    }
                pf = dbuf.data();
                pm = dmask.empty() ? nullptr : dmask.data();
            }
            std::vector<float> abuf;
            if (f.diverging && f.asinhToe > 0)
            {   // display-only asinh compression; bin/web carry the raw field
                abuf.resize(std::size_t(pw) * ph);
                const float den = std::asinh(f.hi / f.asinhToe);
                for (std::size_t c = 0; c < abuf.size(); ++c)
                    abuf[c] = std::asinh(pf[c] / f.asinhToe) / den;
                pf = abuf.data();
            }
            const bool ok =
                f.diverging
                    ? writeDivergingPng(path, pw, ph, pf,
                                        f.asinhToe > 0 ? 1.0f : f.hi, pm)
                    : writeSequentialPng(path, pw, ph, pf, f.lo, f.hi, pm);
            if (!ok) return std::unexpected(std::format("PNG write failed: {}", path));
        }
        ++p.pngIdx;
        p.nextPng += p.cfg.pdt;
    }

    if (doWeb)
    {
        std::vector<std::vector<float>> planes(fields.size());
        for (auto& wp : p.webs)
        {
            std::vector<const float*> ptr;
            for (std::size_t i = 0; i < fields.size(); ++i)
            {
                p.gather(fields[i].data, wp.a, wp.b, wp.sel, planes[i]);
                ptr.push_back(planes[i].data());
            }
            if (!wp.wf->writeFrame(t, ptr))
                return std::unexpected("web frame write failed");
        }
        p.nextWeb += p.cfg.wdt;
    }

    if (doShm)
    {
        // Same planes as the web path, straight into shared memory: no
        // quantization, no filenames, no page cache. Created on the first
        // frame because the field list is only known here.
        const int ic = p.geo.ic, jc = p.geo.jc, kc = p.geo.kc;
        std::vector<ShmFrames::PlaneSpec> specs;
        specs.push_back({2, 0.0f, nx, ny});
        if (nz > 1)
        {
            specs.push_back({1, 0.0f, nx, nz});
            specs.push_back({0, 0.0f, ny, nz});
        }
        if (!p.shmReady)
        {
            std::vector<ShmField> fs;
            for (const auto& f : fields)
            {
                ShmField sf;
                std::snprintf(sf.name, sizeof sf.name, "%s", f.name.c_str());
                sf.lo = float(f.diverging ? -f.hi : f.lo);
                sf.hi = float(f.hi);
                sf.diverging = f.diverging ? 1 : 0;
                fs.push_back(sf);
            }
            // The volume rides along in 3D: it is what a volumetric render
            // needs, and it is the same data --emit volume would write.
            const int vw = nz > 1 ? nx : 0, vh = nz > 1 ? ny : 0,
                      vd = nz > 1 ? nz : 0;
            if (auto r = p.shm.create(ShmFrames::nameFor(p.cfg.prefix), fs, specs,
                                      vw, vh, vd);
                !r)
                return std::unexpected(r.error());
            // The body centre is only known here as the frame geometry's
            // centre indices; the viewer uses it to frame the plane, and it has
            // the exact placement from its own validated Case anyway.
            p.shm.setGrid(p.g.nx, p.g.ny, p.g.nz, i0, j0, k0, nx, ny, nz, p.g.hx,
                          p.g.hy, p.g.hz, (p.geo.ic + 0.5) * p.g.hx,
                          (p.geo.jc + 0.5) * p.g.hy, (p.geo.kc + 0.5) * p.g.hz,
                          p.geo.D);
            p.shmReady = true;
        }

        std::vector<std::vector<float>> store;
        std::vector<std::vector<const float*>> planeData;
        std::vector<const float*> planeMask;
        auto sel2 = [&](int axis) {
            return [axis, ic, jc, kc, i0, j0, k0](int a, int b) {
                return axis == 2 ? std::array{i0 + a, j0 + b, kc}
                     : axis == 1 ? std::array{i0 + a, jc, k0 + b}
                                 : std::array{ic, j0 + a, k0 + b};
            };
        };
        store.reserve(specs.size() * (fields.size() + 1));
        for (const auto& sp : specs)
        {
            std::vector<const float*> ptr;
            for (const auto& f : fields)
            {
                store.emplace_back();
                p.gather(f.data, sp.w, sp.h, sel2(sp.axis), store.back());
                ptr.push_back(store.back().data());
            }
            planeData.push_back(std::move(ptr));
            if (!p.geo.mask.empty())
            {
                store.emplace_back();
                p.gather(p.geo.mask.data(), sp.w, sp.h, sel2(sp.axis), store.back());
                planeMask.push_back(store.back().data());
            }
            else
            {
                planeMask.push_back(nullptr);
            }
        }

        // Every field's volume, so the viewer can slice or fog whichever one
        // is selected instead of being stuck with the first.
        std::vector<std::vector<float>> vols;
        std::vector<const float*> volPtr;
        if (nz > 1 && p.shm.hasVolume())
        {
            const std::size_t n = std::size_t(nx) * ny * nz;
            vols.resize(fields.size());
            for (std::size_t f = 0; f < fields.size(); ++f)
            {
                vols[f].resize(n);
                std::size_t q = 0;
                for (int kk = 0; kk < nz; ++kk)
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i, ++q)
                            vols[f][q] =
                                fields[f].data[p.g.idx(i0 + i, j0 + j, k0 + kk)];
                volPtr.push_back(vols[f].data());
            }
        }
        p.shm.publish(t, planeData, planeMask, volPtr);
        if (!doWeb) p.nextWeb += p.cfg.wdt;
    }
    return {};
}

std::expected<void, std::string> RunOutput::vtkFrame(const Grid& g,
                                                     const VectorField& U)
{
    if (!has(p_->cfg.kinds, Emit::Vtk)) return {};
    const auto path = std::format("{}_{:05d}.vtk", p_->cfg.prefix, p_->vtkIdx++);
    return writeVtk(g, U, path);
}

std::expected<void, std::string> RunOutput::flush()
{
    for (auto& [n, f] : p_->bins)
    {
        f.flush();
        if (!f) return std::unexpected(std::format("write failed on '{}'", n));
    }
    for (auto& [n, f] : p_->series) f.flush();
    p_->writeJson();
    return {};
}

void reportBlowup(const Grid& g, const VectorField& host, const Scalar& lambda,
                  double t)
{
    const Scalar* comp[3]{&host.u, &host.v, &host.w};
    const char* name[3]{"u", "v", "w"};
    for (int q = 0; q < 3; ++q)
    {
        std::size_t am = 0;
        double m = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c)
            if (std::abs((*comp[q])[c]) > m)
            {
                m = std::abs((*comp[q])[c]);
                am = c;
            }
        const int i = int(am % std::size_t(g.nx));
        const int j = int((am / std::size_t(g.nx)) % std::size_t(g.ny));
        const int k = int(am / (std::size_t(g.nx) * g.ny));
        std::printf("BLOWUP t=%.4f max|%s|=%.3e at i=%d j=%d k=%d (lam=%.3f)\n",
                    t, name[q], m, i, j, k,
                    lambda.empty() ? 1.0 : lambda[am]);
    }
    std::fflush(stdout);
}

} // namespace dns
