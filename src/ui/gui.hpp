// A small immediate-mode widget set: panels, buttons, checkboxes, drag-number
// fields with keyboard entry, segmented choices, lists and a scroll region.
//
// Immediate mode because the thing being edited is a dns::Case: there is no
// widget state worth persisting except what the user is physically dragging,
// and every number shown must be the CURRENT resolved value, not a copy that
// can drift from it. The whole UI is rebuilt each frame from the document.
//
// Everything is laid out in pixels, top-left origin, and emitted into one
// Vtx2 stream with a UiBatch per scissor change (draw.hpp).
#pragma once

#include "ui/draw.hpp"
#include "ui/text.hpp"

#include <SDL3/SDL.h>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ui {

// One frame of input, assembled by app.cpp from the SDL event queue.
struct Input
{
    float mx = 0, my = 0;     // cursor, pixels
    float dx = 0, dy = 0;     // motion since last frame
    float wheel = 0;
    bool lDown = false, lPressed = false, lReleased = false;
    bool rDown = false, rPressed = false;
    bool mDown = false;
    bool doubleClick = false;
    bool shift = false, ctrl = false;
    std::string text;                 // characters typed this frame
    std::vector<SDL_Keycode> keys;    // keys pressed this frame

    bool key(SDL_Keycode k) const;
    void clearFrame();   // clears the per-frame edges, keeps held state
};

// Dark palette. Authored in the values that reach the screen (the swapchain is
// UNORM, see vk.cpp).
struct Theme
{
    Rgba panel = rgb(0x14181d, 235);
    Rgba panelHead = rgb(0x1c222a);
    Rgba border = rgb(0x2b333d);
    Rgba text = rgb(0xd7dee6);
    Rgba textDim = rgb(0x8b96a3);
    Rgba accent = rgb(0x4fa3ff);
    Rgba accentDim = rgb(0x2a5d94);
    Rgba widget = rgb(0x232b34);
    Rgba widgetHot = rgb(0x2e3945);
    Rgba widgetActive = rgb(0x3a4757);
    Rgba good = rgb(0x5fd08a);
    Rgba warn = rgb(0xf0b45a);
    Rgba bad = rgb(0xef6b6b);

    float pad = 8;
    float rowH = 22;
    float gap = 4;
    float fontPx = 14;
    float smallPx = 12;
    float radius = 4;
};

class Gui
{
public:
    Gui(TextEngine& text) : text_(&text) {}

    void begin(const Input&, float screenW, float screenH);
    void end();

    std::span<const Vtx2> vertices() const { return verts_; }
    std::span<const UiBatch> batches() const { return batches_; }

    // True when the pointer is over a panel, or a widget is being dragged --
    // the camera must not also act on that drag.
    bool capturingMouse() const { return hoverPanel_ || active_ != 0; }
    bool capturingKeyboard() const { return editing_ != 0; }

    Theme theme;

    // --- panels -------------------------------------------------------------
    // Panels are placed explicitly; height comes from last frame's content, so
    // the background can be emitted before the content that determines it.
    void beginPanel(std::string_view title, float x, float y, float w,
                    float maxH = 0);
    void endPanel();
    float panelHeight(std::string_view title) const;

    // --- widgets ------------------------------------------------------------
    void label(std::string_view, Rgba col);
    void label(std::string_view s) { label(s, theme.text); }
    // Key on the left, value right-aligned: the read-out form used by every
    // diagnostic row.
    void field(std::string_view key, std::string_view value, Rgba col);
    void field(std::string_view key, std::string_view value)
    {
        field(key, value, theme.text);
    }
    void separator();
    void spacing(float h = 4);
    bool sectionHeader(std::string_view, bool& open);

    bool button(std::string_view, float w = 0, Rgba tint = {});
    // N buttons sharing one row; returns the index clicked, or -1.
    int buttonRow(std::span<const char* const> labels);
    bool checkbox(std::string_view, bool& v);
    // Horizontal drag to change, double-click to type. `step` is per pixel.
    // Returns true on any change.
    bool dragDouble(std::string_view label, double& v, double step, double lo,
                    double hi, int decimals = 3);
    bool dragInt(std::string_view label, int& v, double step, int lo, int hi);
    // Segmented control; returns true when the selection changed.
    bool choice(std::string_view label, int& idx, std::span<const char* const>);
    bool selectable(std::string_view, bool selected);
    // A colour key row: swatch + text, for the cut-cell legend.
    void swatch(Rgba, std::string_view);
    // Filled proportion bar with a caption, e.g. "cut cells 3.1% of grid".
    void bar(std::string_view caption, double frac, Rgba);

    // --- raw drawing (used for the viewport overlay) -------------------------
    void rect(float x, float y, float w, float h, Rgba, float radius = 0);
    void rectOutline(float x, float y, float w, float h, Rgba, float t = 1);
    void textAt(std::string_view, float x, float baselineY, Rgba,
                FontId = FontId::Sans, float px = 0);
    float textWidth(std::string_view, FontId = FontId::Sans, float px = 0);

    float cursorY() const { return cur_.y; }
    float contentWidth() const { return cur_.w; }

private:
    uint64_t idFor(std::string_view) const;
    void pushBatch();
    void setScissor(float x, float y, float w, float h);
    void clearScissor();
    // Advances the layout cursor by one row and returns its rect.
    struct Rect { float x, y, w, h; };
    Rect row(float h = 0);
    bool hit(const Rect&) const;
    void tri(float x0, float y0, float x1, float y1, float x2, float y2, Rgba);

    TextEngine* text_ = nullptr;
    std::vector<Vtx2> verts_;
    std::vector<UiBatch> batches_;
    VkRect2D scissor_{};
    uint32_t batchStart_ = 0;

    Input in_{};
    float screenW_ = 0, screenH_ = 0;

    struct Cursor
    {
        float x = 0, y = 0, w = 0;
        float panelX = 0, panelY = 0, panelW = 0, panelTop = 0;
        bool inPanel = false;
        float clipY0 = 0, clipY1 = 0;
    } cur_;

    std::string panelName_;
    std::unordered_map<uint64_t, float> panelH_;
    std::unordered_map<uint64_t, float> scroll_;

    uint64_t hot_ = 0, active_ = 0, editing_ = 0;
    double dragAccum_ = 0;
    std::string editBuf_;
    bool hoverPanel_ = false;
    int rowCounter_ = 0;
};

} // namespace ui
