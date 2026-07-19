#include "dns/output.hpp"

#include "dns/fieldpng.hpp"
#include "dns/vtk_writer.hpp"
#include "dns/webframes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <memory>

namespace dns {

namespace {

bool has(const std::set<Emit>& s, Emit e) { return s.contains(e); }

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

    std::string pngDir;
    double nextFrame = 0, nextPng = 0, nextWeb = 0;
    int frames = 0, pngIdx = 0, vtkIdx = 0;
    double msPerStep = 0;
    bool moved = false;

    std::vector<float> buf;   // scratch plane / volume buffer

    std::string json() const
    {
        return std::format(
            "{{\"nx\":{},\"ny\":{},\"nz\":{},\"frames\":{},\"frameDt\":{},"
            "\"D\":{},\"re\":{},\"mach\":{},\"cs\":{},\"les\":{},"
            "\"body\":\"{}\",\"far\":\"{}\",\"physics\":\"{}\",\"kick\":{},"
            "\"msPerStep\":{}}}\n",
            meta.nx, meta.ny, meta.nz, frames, meta.frameDt, meta.D, meta.re,
            meta.mach, meta.cs, meta.les ? 1 : 0, meta.body, meta.farField,
            meta.physics, meta.kick, msPerStep);
    }

    void writeJson() const
    {
        std::ofstream f(cfg.prefix + ".json");
        f << json();
    }

    std::ofstream& binFor(const std::string& name)
    {
        auto it = bins.find(name);
        if (it != bins.end()) return it->second;
        auto& f = bins[name];
        f.open(cfg.prefix + "_" + name + ".bin", std::ios::binary);
        return f;
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
    if (!p_) return;                      // moved from
    for (auto& [n, f] : p_->bins) f.close();
    for (auto& [n, f] : p_->series) f.close();
    p_->writeJson();                      // final frame count + ms/step
}

int RunOutput::frames() const { return p_->frames; }
void RunOutput::setMsPerStep(double ms) { p_->msPerStep = ms; }

std::expected<RunOutput, std::string>
RunOutput::make(Config cfg, RunMeta meta, const Grid& g, FrameGeometry geo)
{
    if (cfg.prefix.empty())
        return std::unexpected("output prefix is empty");
    if (has(cfg.kinds, Emit::Png) && g.nz != 1)
        return std::unexpected("PNG output requires nz == 1");
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

    // Phase 3c: create and truncate everything now, so a full or read-only
    // destination is discovered in the first second, not after hours of GPU.
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
        p.pngDir = "png_" + p.cfg.prefix;
        std::error_code ec;
        std::filesystem::create_directories(p.pngDir, ec);
        if (ec) return std::unexpected(
            std::format("cannot create '{}': {}", p.pngDir, ec.message()));
    }
    return o;
}

void RunOutput::declareSeries(std::string_view name,
                              std::span<const std::string> cols)
{
    if (!has(p_->cfg.kinds, Emit::Stats)) return;
    const std::string key(name);
    auto& f = p_->series[key];
    // Re-declaring a series reopens it: opening an already-open stream would
    // set failbit and silently swallow every subsequent row.
    if (f.is_open()) f.close();
    f.clear();
    f.open(p_->cfg.prefix + "_" + key + ".csv");
    for (std::size_t i = 0; i < cols.size(); ++i)
        f << (i ? "," : "") << cols[i];
    f << '\n';
}

std::expected<void, std::string> RunOutput::row(std::string_view name,
                                                std::span<const double> v)
{
    auto it = p_->series.find(std::string(name));
    if (it == p_->series.end()) return {};      // series not enabled
    auto& f = it->second;
    for (std::size_t i = 0; i < v.size(); ++i)
        f << (i ? "," : "") << v[i];
    f << '\n';
    f.flush();
    if (!f) return std::unexpected(std::format("write failed on series '{}'", name));
    return {};
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

std::expected<void, std::string> RunOutput::frame(double t, const FrameViews& fv)
{
    auto& p = *p_;
    const auto& k = p.cfg.kinds;
    const bool doBin = wantsFrame(t), doPng = wantsPng(t), doWeb = wantsWeb(t);
    if (!doBin && !doPng && !doWeb) return {};

    const int nx = p.geo.w, ny = p.geo.h, nz = p.geo.d;
    const int i0 = p.geo.i0, j0 = p.geo.j0, k0 = p.geo.k0;

    // Lazily build the web planes: the field list is only known here.
    if (doWeb && p.webs.empty())
        if (auto r = [&]() -> std::expected<void, std::string> {
                const int per = std::clamp(p.cfg.planesPerAxis, 1, 3);
                const int dI = std::max(1, nx / 16), dJ = std::max(1, ny / 16),
                          dK = std::max(1, nz / 16);
                std::string idx;
                auto add = [&](const std::string& suf, const std::string& axis,
                               double offD, int a, int b, auto sel) {
                    WebPlane wp;
                    char offs[40];
                    std::snprintf(offs, sizeof offs, "%+.3gD", offD);
                    wp.wf = std::make_unique<WebFrames>(
                        "web_" + p.cfg.prefix + suf, a, b,
                        p.cfg.simJson + ",\"plane\":\"" + axis + "\",\"off\":\"" +
                            (offD == 0.0 ? "mid" : offs) + "\"}");
                    for (const auto& f : fv.fields)
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
                        wp.wf->writeStatic("mask", m.data(), 0.0, 1.0);
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
                auto xy = [&](int kk) {
                    return [kk, i0, j0](int i, int j) {
                        return std::array{i0 + i, j0 + j, kk};
                    };
                };
                auto xz = [&](int jj) {
                    return [jj, i0, k0](int i, int kk) {
                        return std::array{i0 + i, jj, k0 + kk};
                    };
                };
                auto yz = [&](int ii) {
                    return [ii, j0, k0](int j, int kk) {
                        return std::array{ii, j0 + j, k0 + kk};
                    };
                };
                add("", "xy", 0.0, nx, ny, xy(kc));
                if (nz == 1) return {};             // native 2D: one plane only
                if (per == 3 && nz >= 16)
                {
                    add("_xy_m", "xy", -cellsD(dK), nx, ny, xy(kc - dK));
                    add("_xy_p", "xy", +cellsD(dK), nx, ny, xy(kc + dK));
                }
                add("_xz", "xz", 0.0, nx, nz, xz(jc));
                if (per == 3 && ny >= 16)
                {
                    add("_xz_m", "xz", -cellsD(dJ), nx, nz, xz(jc - dJ));
                    add("_xz_p", "xz", +cellsD(dJ), nx, nz, xz(jc + dJ));
                }
                add("_yz", "yz", 0.0, ny, nz, yz(ic));
                if (per == 3 && nx >= 16)
                {
                    add("_yz_m", "yz", -cellsD(dI), ny, nz, yz(ic - dI));
                    add("_yz_p", "yz", +cellsD(dI), ny, nz, yz(ic + dI));
                }
                const std::string path = "web_" + p.cfg.prefix + "/planes.json";
                std::ofstream pf(path);
                pf << "{\"planes\":[" << idx << "]}\n";
                return closeChecked(pf, path);
            }();
            !r)
            return r;

    if (doBin)
    {
        for (const auto& f : fv.fields)
        {
            if (has(k, Emit::Volume))
            {
                auto& out = p.binFor(f.name);
                p.buf.resize(std::size_t(nx) * ny * nz);
                std::size_t q = 0;
                for (int kk = 0; kk < nz; ++kk)
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i, ++q)
                            p.buf[q] = f.data[p.g.idx(i0 + i, j0 + j, k0 + kk)];
                out.write(reinterpret_cast<const char*>(p.buf.data()),
                          std::streamsize(p.buf.size() * sizeof(float)));
            }
            if (has(k, Emit::Bin))
            {
                auto& out = p.binFor(f.name);
                p.gather(f.data, nx, ny,
                         [kc = p.geo.kc, i0, j0](int i, int j) {
                             return std::array{i0 + i, j0 + j, kc};
                         },
                         p.buf);
                out.write(reinterpret_cast<const char*>(p.buf.data()),
                          std::streamsize(p.buf.size() * sizeof(float)));
            }
            if (has(k, Emit::Slices))
            {
                const int ic = p.geo.ic, jc = p.geo.jc, kc = p.geo.kc;
                auto plane = [&](const char* suf, int a, int b, auto sel) {
                    p.gather(f.data, a, b, sel, p.buf);
                    auto& out = p.binFor(std::string(suf) + "_" + f.name);
                    out.write(reinterpret_cast<const char*>(p.buf.data()),
                              std::streamsize(p.buf.size() * sizeof(float)));
                };
                plane("xy", nx, ny, [kc, i0, j0](int i, int j) {
                    return std::array{i0 + i, j0 + j, kc};
                });
                plane("xz", nx, nz, [jc, i0, k0](int i, int kk) {
                    return std::array{i0 + i, jc, k0 + kk};
                });
                plane("yz", ny, nz, [ic, j0, k0](int j, int kk) {
                    return std::array{ic, j0 + j, k0 + kk};
                });
            }
        }
        ++p.frames;
        p.nextFrame += p.cfg.fdt;
    }

    if (doPng)
    {
        for (const auto& f : fv.fields)
        {
            p.gather(f.data, nx, ny,
                     [kc = p.geo.kc, i0, j0](int i, int j) {
                         return std::array{i0 + i, j0 + j, kc};
                     },
                     p.buf);
            const auto path = std::format("{}/{}_{:05d}.png", p.pngDir, f.name,
                                          p.pngIdx);
            std::vector<float> m;
            if (!p.geo.mask.empty())
                p.gather(p.geo.mask.data(), nx, ny,
                         [kc = p.geo.kc, i0, j0](int i, int j) {
                             return std::array{i0 + i, j0 + j, kc};
                         },
                         m);
            const bool ok =
                f.diverging
                    ? writeDivergingPng(path, nx, ny, p.buf.data(), f.hi,
                                        m.empty() ? nullptr : m.data())
                    : writeSequentialPng(path, nx, ny, p.buf.data(), f.lo, f.hi,
                                         m.empty() ? nullptr : m.data());
            if (!ok) return std::unexpected(std::format("PNG write failed: {}", path));
        }
        ++p.pngIdx;
        p.nextPng += p.cfg.pdt;
    }

    if (doWeb)
    {
        std::vector<std::vector<float>> planes(fv.fields.size());
        for (auto& wp : p.webs)
        {
            std::vector<const float*> ptr;
            for (std::size_t i = 0; i < fv.fields.size(); ++i)
            {
                p.gather(fv.fields[i].data, wp.a, wp.b, wp.sel, planes[i]);
                ptr.push_back(planes[i].data());
            }
            if (!wp.wf->writeFrame(t, ptr))
                return std::unexpected("web frame write failed");
        }
        p.nextWeb += p.cfg.wdt;
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
