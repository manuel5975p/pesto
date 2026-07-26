#include "ui/text.hpp"

#include <SDL3/SDL.h>
#include <hb.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace ui {
namespace {

constexpr uint32_t kAtlasW = 1024;
constexpr uint32_t kAtlasH = 1024;
constexpr int kSubRows = 4;   // vertical supersampling; horizontal is exact

// --- outline capture --------------------------------------------------------
//
// hb-draw hands us the glyph in FONT UNITS with y up. We scale to pixels here
// (one multiply per point) and flatten curves at a segment count set by the
// chord length, so a 12 px glyph is not paying for 32 segments per curve.

struct Pt
{
    float x, y;
};

struct Outline
{
    std::vector<std::vector<Pt>> contours;
    float scale = 1.0f;

    void moveTo(float x, float y)
    {
        contours.emplace_back();
        contours.back().push_back({x * scale, y * scale});
    }
    void lineTo(float x, float y)
    {
        if (contours.empty()) return;
        contours.back().push_back({x * scale, y * scale});
    }
    int segments(Pt a, Pt b) const
    {
        const float d = std::abs(b.x - a.x) + std::abs(b.y - a.y);
        return std::clamp(int(d / 1.5f) + 2, 3, 24);
    }
    void quadTo(float cx, float cy, float x, float y)
    {
        if (contours.empty()) return;
        const Pt p0 = contours.back().back();
        const Pt c{cx * scale, cy * scale}, p1{x * scale, y * scale};
        const int n = segments(p0, p1);
        for (int i = 1; i <= n; ++i) {
            const float t = float(i) / float(n), u = 1 - t;
            contours.back().push_back({u * u * p0.x + 2 * u * t * c.x + t * t * p1.x,
                                       u * u * p0.y + 2 * u * t * c.y + t * t * p1.y});
        }
    }
    void cubicTo(float c1x, float c1y, float c2x, float c2y, float x, float y)
    {
        if (contours.empty()) return;
        const Pt p0 = contours.back().back();
        const Pt a{c1x * scale, c1y * scale}, b{c2x * scale, c2y * scale},
            p1{x * scale, y * scale};
        const int n = segments(p0, p1);
        for (int i = 1; i <= n; ++i) {
            const float t = float(i) / float(n), u = 1 - t;
            const float w0 = u * u * u, w1 = 3 * u * u * t, w2 = 3 * u * t * t,
                        w3 = t * t * t;
            contours.back().push_back(
                {w0 * p0.x + w1 * a.x + w2 * b.x + w3 * p1.x,
                 w0 * p0.y + w1 * a.y + w2 * b.y + w3 * p1.y});
        }
    }
};

void cbMove(hb_draw_funcs_t*, void* d, hb_draw_state_t*, float x, float y, void*)
{
    static_cast<Outline*>(d)->moveTo(x, y);
}
void cbLine(hb_draw_funcs_t*, void* d, hb_draw_state_t*, float x, float y, void*)
{
    static_cast<Outline*>(d)->lineTo(x, y);
}
void cbQuad(hb_draw_funcs_t*, void* d, hb_draw_state_t*, float cx, float cy, float x,
            float y, void*)
{
    static_cast<Outline*>(d)->quadTo(cx, cy, x, y);
}
void cbCubic(hb_draw_funcs_t*, void* d, hb_draw_state_t*, float c1x, float c1y,
             float c2x, float c2y, float x, float y, void*)
{
    static_cast<Outline*>(d)->cubicTo(c1x, c1y, c2x, c2y, x, y);
}
void cbClose(hb_draw_funcs_t*, void*, hb_draw_state_t*, void*) {}

// --- scanline fill ----------------------------------------------------------
//
// Non-zero winding, kSubRows samples per pixel row, exact horizontal coverage
// at the span ends. That combination is what makes 11 px text legible: pure
// box supersampling at 4x4 leaves visible steps on near-vertical stems, and
// the horizontal term is nearly free.

struct Edge
{
    float x0, y0, x1, y1;
    int dir;
};

void addSpan(std::vector<float>& row, float xs, float xe, float weight, int w)
{
    if (xe <= xs) return;
    xs = std::max(xs, 0.0f);
    xe = std::min(xe, float(w));
    if (xe <= xs) return;
    const int i0 = int(std::floor(xs)), i1 = int(std::ceil(xe)) - 1;
    if (i0 == i1) {
        row[size_t(i0)] += (xe - xs) * weight;
        return;
    }
    row[size_t(i0)] += (float(i0 + 1) - xs) * weight;
    for (int i = i0 + 1; i < i1; ++i) row[size_t(i)] += weight;
    row[size_t(i1)] += (xe - float(i1)) * weight;
}

// Rasterizes `o` into an 8-bit coverage bitmap; returns its integer bbox in
// pixel space (y up), i.e. the bitmap covers [x0,x1) x [y0,y1).
struct Raster
{
    std::vector<uint8_t> bits;
    int x0 = 0, y0 = 0, w = 0, h = 0;
};

Raster rasterize(const Outline& o)
{
    Raster r;
    float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
    for (const auto& c : o.contours)
        for (const auto& p : c) {
            minX = std::min(minX, p.x);
            maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y);
            maxY = std::max(maxY, p.y);
        }
    if (minX > maxX) return r;   // empty glyph (space)

    r.x0 = int(std::floor(minX)) - 1;
    r.y0 = int(std::floor(minY)) - 1;
    r.w = int(std::ceil(maxX)) + 1 - r.x0;
    r.h = int(std::ceil(maxY)) + 1 - r.y0;
    if (r.w <= 0 || r.h <= 0 || r.w > 4096 || r.h > 4096) return Raster{};
    r.bits.assign(size_t(r.w) * size_t(r.h), 0);

    std::vector<Edge> edges;
    for (const auto& c : o.contours) {
        const size_t n = c.size();
        if (n < 2) continue;
        for (size_t i = 0; i < n; ++i) {
            const Pt a = c[i], b = c[(i + 1) % n];
            if (a.y == b.y) continue;
            edges.push_back(a.y < b.y ? Edge{a.x - float(r.x0), a.y - float(r.y0),
                                             b.x - float(r.x0), b.y - float(r.y0), 1}
                                      : Edge{b.x - float(r.x0), b.y - float(r.y0),
                                             a.x - float(r.x0), a.y - float(r.y0), -1});
        }
    }
    if (edges.empty()) return r;

    std::vector<float> row(size_t(r.w));
    std::vector<std::pair<float, int>> xs;
    const float weight = 1.0f / float(kSubRows);
    for (int py = 0; py < r.h; ++py) {
        std::fill(row.begin(), row.end(), 0.0f);
        for (int s = 0; s < kSubRows; ++s) {
            // Bitmap row py (top-down) covers y in [h-1-py, h-py] going up.
            const float y = float(r.h - 1 - py) + (float(s) + 0.5f) / float(kSubRows);
            xs.clear();
            for (const auto& e : edges) {
                if (y < e.y0 || y >= e.y1) continue;
                const float t = (y - e.y0) / (e.y1 - e.y0);
                xs.push_back({e.x0 + t * (e.x1 - e.x0), e.dir});
            }
            if (xs.size() < 2) continue;
            std::sort(xs.begin(), xs.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            int wind = 0;
            float start = 0;
            for (const auto& [x, d] : xs) {
                const int prev = wind;
                wind += d;
                if (prev == 0 && wind != 0) start = x;
                else if (prev != 0 && wind == 0) addSpan(row, start, x, weight, r.w);
            }
        }
        uint8_t* dst = &r.bits[size_t(py) * size_t(r.w)];
        for (int px = 0; px < r.w; ++px)
            dst[px] = uint8_t(std::clamp(row[size_t(px)], 0.0f, 1.0f) * 255.0f + 0.5f);
    }
    return r;
}

std::string findFont(const char* env, const char* const* candidates)
{
    if (const char* e = SDL_getenv(env); e && e[0]) return e;
    for (const char* const* p = candidates; *p; ++p) {
        SDL_PathInfo info{};
        if (SDL_GetPathInfo(*p, &info)) return *p;
    }
    return {};
}

std::expected<std::vector<char>, std::string> readAll(const std::string& path)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return std::unexpected("cannot open font " + path);
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0) {
        std::fclose(f);
        return std::unexpected("empty font " + path);
    }
    const size_t bytes = size_t(n);
    std::vector<char> data(bytes);
    const size_t got = std::fread(data.data(), 1, bytes, f);
    std::fclose(f);
    if (got != size_t(n)) return std::unexpected("short read on font " + path);
    return data;
}

} // namespace

std::expected<std::unique_ptr<TextEngine>, std::string> TextEngine::build(Vk* vk)
{
    static const char* sansCandidates[] = {
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        nullptr};
    static const char* monoCandidates[] = {
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        nullptr};

    const std::string sans = findFont("DNS_UI_FONT", sansCandidates);
    if (sans.empty())
        return std::unexpected(
            "no UI font found; set DNS_UI_FONT to a .ttf/.otf path");
    const std::string mono = [&] {
        const std::string m = findFont("DNS_UI_FONT_MONO", monoCandidates);
        return m.empty() ? sans : m;
    }();

    auto te = std::unique_ptr<TextEngine>(new TextEngine());
    te->vk_ = vk;

    const std::string paths[2] = {sans, mono};
    for (int i = 0; i < 2; ++i) {
        Face& f = te->faces_[i];
        auto data = readAll(paths[size_t(i)]);
        if (!data) return std::unexpected(data.error());
        f.data = std::move(*data);
        hb_blob_t* blob = hb_blob_create(f.data.data(), unsigned(f.data.size()),
                                         HB_MEMORY_MODE_READONLY, nullptr, nullptr);
        f.face = hb_face_create(blob, 0);
        hb_blob_destroy(blob);
        if (hb_face_get_glyph_count(f.face) == 0)
            return std::unexpected("font has no glyphs: " + paths[size_t(i)]);
        f.upem = float(hb_face_get_upem(f.face));
        f.font = hb_font_create(f.face);
        // Scale == upem: everything HarfBuzz reports is then in font units,
        // and the px conversion is one multiply we control.
        hb_font_set_scale(f.font, int(f.upem), int(f.upem));

        hb_font_extents_t ext{};
        if (hb_font_get_h_extents(f.font, &ext)) {
            f.ascent = float(ext.ascender);
            f.descent = float(-ext.descender);
            f.lineGap = float(ext.line_gap);
        } else {
            f.ascent = 0.8f * f.upem;
            f.descent = 0.2f * f.upem;
        }
    }

    te->buf_ = hb_buffer_create();
    te->draw_ = hb_draw_funcs_create();
    hb_draw_funcs_set_move_to_func(te->draw_, cbMove, nullptr, nullptr);
    hb_draw_funcs_set_line_to_func(te->draw_, cbLine, nullptr, nullptr);
    hb_draw_funcs_set_quadratic_to_func(te->draw_, cbQuad, nullptr, nullptr);
    hb_draw_funcs_set_cubic_to_func(te->draw_, cbCubic, nullptr, nullptr);
    hb_draw_funcs_set_close_path_func(te->draw_, cbClose, nullptr, nullptr);

    te->atlasW_ = kAtlasW;
    te->atlasH_ = kAtlasH;
    te->pixels_.assign(size_t(kAtlasW) * kAtlasH, 0);
    // Headless (--check): no device, so no atlas texture and no upload. Shaping,
    // metrics and the CPU rasterizer need nothing from Vulkan, which is what
    // makes the whole widget layer testable without a window.
    if (vk) {
        auto tex = vk->createCoverageTexture(kAtlasW, kAtlasH);
        if (!tex) return std::unexpected(tex.error());
        te->atlas_ = *tex;
    }
    te->dirty_ = true;
    if (auto r = te->flush(); !r) return std::unexpected(r.error());
    return te;
}

std::expected<std::unique_ptr<TextEngine>, std::string> TextEngine::create(Vk& vk)
{
    return build(&vk);
}

std::expected<std::unique_ptr<TextEngine>, std::string> TextEngine::createHeadless()
{
    return build(nullptr);
}

TextEngine::~TextEngine()
{
    if (draw_) hb_draw_funcs_destroy(draw_);
    if (buf_) hb_buffer_destroy(buf_);
    for (auto& f : faces_) {
        if (f.font) hb_font_destroy(f.font);
        if (f.face) hb_face_destroy(f.face);
    }
    if (vk_) vk_->destroy(atlas_);
}

float TextEngine::ascent(FontId id, float px) const
{
    const Face& f = faceOf(id);
    return f.ascent * px / f.upem;
}
float TextEngine::descent(FontId id, float px) const
{
    const Face& f = faceOf(id);
    return f.descent * px / f.upem;
}
float TextEngine::lineHeight(FontId id, float px) const
{
    const Face& f = faceOf(id);
    return (f.ascent + f.descent + f.lineGap) * px / f.upem;
}

const TextEngine::Glyph* TextEngine::glyph(FontId id, uint32_t gid, float px)
{
    const int ipx = int(px + 0.5f);
    const uint64_t key = (uint64_t(int(id)) << 56) | (uint64_t(ipx) << 32) | gid;
    if (auto it = cache_.find(key); it != cache_.end()) return &it->second;

    Face& f = faceOf(id);
    Outline o;
    o.scale = float(ipx) / f.upem;
    hb_font_draw_glyph_or_fail(f.font, gid, draw_, &o);
    const Raster r = rasterize(o);

    Glyph g{};
    if (r.w > 0 && r.h > 0) {
        // Shelf packing with a 1 px gutter, so linear filtering at the quad
        // edge cannot pull in a neighbour's coverage.
        if (penX_ + uint32_t(r.w) + 1 > atlasW_) {
            penX_ = 1;
            penY_ += rowH_ + 1;
            rowH_ = 0;
        }
        if (penY_ + uint32_t(r.h) + 1 <= atlasH_) {
            for (int y = 0; y < r.h; ++y)
                std::memcpy(&pixels_[size_t(penY_ + uint32_t(y)) * atlasW_ + penX_],
                            &r.bits[size_t(y) * size_t(r.w)], size_t(r.w));
            g.u0 = float(penX_) / float(atlasW_);
            g.v0 = float(penY_) / float(atlasH_);
            g.u1 = float(penX_ + uint32_t(r.w)) / float(atlasW_);
            g.v1 = float(penY_ + uint32_t(r.h)) / float(atlasH_);
            g.w = float(r.w);
            g.h = float(r.h);
            g.bx = float(r.x0);
            // Bitmap top in y-down pixels relative to the baseline.
            g.by = -float(r.y0 + r.h);
            penX_ += uint32_t(r.w) + 1;
            rowH_ = std::max(rowH_, uint32_t(r.h));
            dirty_ = true;
        }
        // Atlas full: g stays zero-sized. The text still advances correctly,
        // so the UI degrades to gaps rather than to garbage.
    }
    return &cache_.emplace(key, g).first->second;
}

float TextEngine::append(std::vector<Vtx2>& out, FontId id, std::string_view s,
                         float x, float baselineY, float px, Rgba col)
{
    if (s.empty()) return 0;
    Face& f = faceOf(id);
    const int ipx = int(px + 0.5f);
    const float toPx = float(ipx) / f.upem;

    hb_buffer_clear_contents(buf_);
    hb_buffer_add_utf8(buf_, s.data(), int(s.size()), 0, int(s.size()));
    hb_buffer_set_direction(buf_, HB_DIRECTION_LTR);
    hb_buffer_set_script(buf_, HB_SCRIPT_LATIN);
    hb_buffer_set_language(buf_, hb_language_from_string("en", -1));
    hb_shape(f.font, buf_, nullptr, 0);

    unsigned n = 0;
    const hb_glyph_info_t* info = hb_buffer_get_glyph_infos(buf_, &n);
    const hb_glyph_position_t* pos = hb_buffer_get_glyph_positions(buf_, &n);

    float penX = x, penY = baselineY;
    for (unsigned i = 0; i < n; ++i) {
        const Glyph* g = glyph(id, info[i].codepoint, float(ipx));
        if (g && g->w > 0) {
            const float gx = penX + float(pos[i].x_offset) * toPx + g->bx;
            const float gy = penY - float(pos[i].y_offset) * toPx + g->by;
            const Vtx2 a{{gx, gy}, {g->u0, g->v0}, col};
            const Vtx2 b{{gx + g->w, gy}, {g->u1, g->v0}, col};
            const Vtx2 c{{gx + g->w, gy + g->h}, {g->u1, g->v1}, col};
            const Vtx2 d{{gx, gy + g->h}, {g->u0, g->v1}, col};
            out.push_back(a);
            out.push_back(b);
            out.push_back(c);
            out.push_back(a);
            out.push_back(c);
            out.push_back(d);
        }
        penX += float(pos[i].x_advance) * toPx;
        penY -= float(pos[i].y_advance) * toPx;
    }
    return penX - x;
}

float TextEngine::measure(FontId id, std::string_view s, float px)
{
    if (s.empty()) return 0;
    Face& f = faceOf(id);
    const float toPx = float(int(px + 0.5f)) / f.upem;
    hb_buffer_clear_contents(buf_);
    hb_buffer_add_utf8(buf_, s.data(), int(s.size()), 0, int(s.size()));
    hb_buffer_set_direction(buf_, HB_DIRECTION_LTR);
    hb_buffer_set_script(buf_, HB_SCRIPT_LATIN);
    hb_buffer_set_language(buf_, hb_language_from_string("en", -1));
    hb_shape(f.font, buf_, nullptr, 0);
    unsigned n = 0;
    const hb_glyph_position_t* pos = hb_buffer_get_glyph_positions(buf_, &n);
    float w = 0;
    for (unsigned i = 0; i < n; ++i) w += float(pos[i].x_advance) * toPx;
    return w;
}

std::expected<void, std::string> TextEngine::flush()
{
    if (!dirty_) return {};
    dirty_ = false;
    if (!vk_) return {};   // headless: nothing to upload to
    return vk_->uploadCoverage(atlas_, pixels_.data());
}

} // namespace ui
