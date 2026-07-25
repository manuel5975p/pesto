// Text, with HarfBuzz as the ONLY dependency below it.
//
// HarfBuzz shapes -- it does not rasterize, and it has no Vulkan backend at
// any version. What it does have since the hb-draw API is
// hb_font_draw_glyph_or_fail(), which walks a glyph's outline as move/line/
// quadratic/cubic callbacks straight out of the font tables. That is enough to
// own the whole pipeline without FreeType:
//
//   shape (hb_shape)  ->  outline (hb-draw)  ->  flatten  ->  scanline fill
//   with 4x vertical supersampling and exact horizontal spans  ->  R8 atlas
//
// The atlas is a single coverage texture (see Renderer::setGlyphAtlas), so one
// texture and one pipeline serve every size and colour in the UI. Glyphs are
// rasterized on first use and the atlas re-uploaded then -- which is why the
// whole UI is built BEFORE the render pass opens (app.cpp), not inside it.
#pragma once

#include "ui/draw.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// HarfBuzz's opaque handles, forward declared at global scope so this header
// does not pull hb.h into everything that draws text.
struct hb_face_t;
struct hb_font_t;
struct hb_buffer_t;
struct hb_draw_funcs_t;

namespace ui {

// Index into TextEngine's face list.
enum class FontId : int
{
    Sans = 0,
    Mono = 1,
};

class TextEngine
{
public:
    // Loads a proportional and a monospace face. Paths are resolved from
    // DNS_UI_FONT / DNS_UI_FONT_MONO or a short list of standard locations;
    // "no usable font" is a startup error, not a silent blank UI.
    static std::expected<std::unique_ptr<TextEngine>, std::string> create(Vk&);
    ~TextEngine();

    TextEngine(const TextEngine&) = delete;
    TextEngine& operator=(const TextEngine&) = delete;

    // Metrics in pixels at size `px`.
    float ascent(FontId, float px) const;
    float descent(FontId, float px) const;
    float lineHeight(FontId, float px) const;
    float measure(FontId, std::string_view, float px);

    // Appends glyph quads with the pen at (x, baselineY). Returns the advance.
    float append(std::vector<Vtx2>& out, FontId, std::string_view, float x,
                 float baselineY, float px, Rgba);

    // Uploads the atlas if glyphs were added since the last call. Call OUTSIDE
    // the render pass.
    std::expected<void, std::string> flush();
    const Texture& atlas() const { return atlas_; }

private:
    TextEngine() = default;

    struct Face
    {
        std::vector<char> data;   // the mmap-free copy hb_blob borrows
        ::hb_face_t* face = nullptr;
        ::hb_font_t* font = nullptr;
        float upem = 1000;
        float ascent = 0, descent = 0, lineGap = 0;   // font units
    };

    struct Glyph
    {
        float u0 = 0, v0 = 0, u1 = 0, v1 = 0;
        float w = 0, h = 0;       // pixels
        float bx = 0, by = 0;     // offset from the pen, y down
    };

    const Glyph* glyph(FontId, uint32_t gid, float px);
    Face& faceOf(FontId f) { return faces_[int(f)]; }
    const Face& faceOf(FontId f) const { return faces_[int(f)]; }

    Vk* vk_ = nullptr;
    Face faces_[2];
    ::hb_buffer_t* buf_ = nullptr;
    ::hb_draw_funcs_t* draw_ = nullptr;

    Texture atlas_{};
    std::vector<uint8_t> pixels_;
    uint32_t atlasW_ = 0, atlasH_ = 0;
    uint32_t penX_ = 1, penY_ = 1, rowH_ = 0;   // shelf packer
    bool dirty_ = false;

    std::unordered_map<uint64_t, Glyph> cache_;
};

} // namespace ui
