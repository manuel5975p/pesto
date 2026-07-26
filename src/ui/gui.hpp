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

    // UI scale. Every metric above is authored at s == 1 ("design pixels"),
    // and so is every baked-in offset in gui.cpp -- text baselines, checkbox
    // boxes, the scrollbar thumb. They all go through px(), so scaling is one
    // number rather than a hunt for literals, and glyphs are RASTERIZED at the
    // scaled size (fontPx feeds the atlas) instead of being magnified, which
    // is the difference between larger text and blurrier text.
    float s = 1.0f;
    float px(float designPx) const { return designPx * s; }

    void scale(float f)
    {
        s *= f;
        pad *= f;
        rowH *= f;
        gap *= f;
        fontPx *= f;
        smallPx *= f;
        radius *= f;
    }
};

// What a text field will accept. Token is not a UI opinion: an object name
// becomes a force-CSV column (cd_<name>) and dns::validSceneName rejects
// anything outside [A-Za-z0-9_-], so refusing the keystroke is the same rule
// enforced one second earlier.
enum class Charset { Any, Token, Path, Number };

struct TextOpts
{
    Charset charset = Charset::Any;
    std::size_t maxLen = 256;
    std::string_view hint;          // placeholder, drawn dim when empty
    bool selectAllOnFocus = true;
};

// Clipboard as injected function pointers rather than direct SDL calls: the
// headless check runs with no video subsystem, and SDL's clipboard needs one.
struct Clipboard
{
    std::string (*get)() = nullptr;
    void (*set)(std::string_view) = nullptr;
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
    // Latched: `active_` is retired in end(), so testing it after the frame
    // would report false on the very frame a click was consumed -- and the app
    // reads this AFTER buildUi() to decide whether the viewport also gets the
    // click. `gestureHeld_` records that a drag was in flight when the frame
    // began, which is the honest answer for the whole frame.
    bool capturingMouse() const { return hoverPanel_ || gestureHeld_; }
    bool capturingKeyboard() const { return keyboardCaptured_; }
    // True when a drag field under the pointer took this frame's Left/Right.
    // The app's timeline is on the same two keys, so without this, nudging a
    // number while hovering it would also scrub the run. Valid only AFTER the
    // panels are built -- which is the order buildUi() already runs in.
    bool consumedArrows() const { return arrowsConsumed_; }

    Theme theme;

    // --- panels -------------------------------------------------------------
    // Panels are placed explicitly; height comes from last frame's content, so
    // the background can be emitted before the content that determines it.
    void beginPanel(std::string_view title, float x, float y, float w,
                    float maxH = 0);
    // A DOCKED region: same widget machinery, but it fills the rect it is
    // given exactly (square corners, no drop shadow, scrolls when the content
    // overflows) instead of shrink-wrapping its content and floating over the
    // viewport. The layout owns the rect; the region never negotiates for it.
    void beginRegion(std::string_view title, float x, float y, float w, float h);
    void endPanel();
    float panelHeight(std::string_view title) const;

    // A draggable region divider. `pos` is the edge coordinate it controls and
    // is updated in place; returns true while being dragged. `vertical` means
    // the splitter itself is a vertical bar (it moves in x).
    //
    // The bar is hit-tested NOW and drawn LATER, by flushSplitters(). It has to
    // be both: the press must reach it before any panel widget underneath, and
    // the bar must be painted after the regions or the panels drawn over it
    // leave an invisible handle -- which is a handle nobody finds.
    bool splitter(std::string_view id, float x, float y, float w, float h,
                  float& pos, bool vertical, float lo, float hi);
    // Paints every splitter registered this frame, on top. Call after the
    // regions and before the status strip.
    void flushSplitters();

    // A full-width bar for the menu / status strips. Returns the rect's y
    // baseline for text so callers do not re-derive it.
    void strip(float x, float y, float w, float h, Rgba);

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
    // Same, with a unit printed after the value ("0.100 m", "30.0 m/s"). The
    // suffix is display only -- typing a value still types a bare number --
    // but a length field that does not say `m` is exactly the ambiguity the
    // physical-units work exists to remove, so it is not optional at the call
    // sites that have a unit.
    bool dragUnit(std::string_view label, double& v, double step, double lo,
                  double hi, std::string_view unit, int decimals = 3);
    bool dragInt(std::string_view label, int& v, double step, int lo, int hi);
    // The two drag-field rules, exposed because they are pure and because
    // guicheck pins them directly rather than inferring them from a gesture.
    //
    // dragRate: per-pixel increment at value `v`. `step` is the increment near
    // zero; a range that cannot go negative is a SCALE and gets a
    // magnitude-relative term on top, so a multi-decade field is reachable at
    // both ends. A signed range (a position) stays purely linear.
    static double dragRate(double v, double step, double lo);
    // dragFill: where `v` sits in [lo,hi] as 0..1 for the position track,
    // log-mapped when the range spans more than ~1.5 decades. NEGATIVE means
    // draw no track: the range is a sentinel (positions are declared
    // [-1e4, 1e4] m for want of a real bound), and a track against a sentinel
    // states a proportion that does not exist.
    static double dragFill(double v, double lo, double hi);
    // Click to focus, type, Enter or click-away commits, Esc reverts. Returns
    // true ONLY on commit -- never per keystroke, because every commit here
    // queues a background cut rebuild.
    bool textInput(std::string_view label, std::string& value,
                   const TextOpts& = {});
    // The same editor filling an explicit rect, with no label column. This is
    // what a list's rename-in-place uses.
    bool textInputAt(uint64_t id, float x, float y, float w, float h,
                     std::string& value, const TextOpts& = {});
    bool editingText() const { return edit_.id != 0; }
    int caretIndex() const { return edit_.caret; }

    Clipboard clipboard;
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

    // --- interaction edges, for undo transactions ---------------------------
    // True on the frame a drag/edit began and the frame it ended. One hook in
    // the app loop can therefore bracket EVERY slider and field in the UI into
    // a single undo entry per gesture.
    bool interactionStarted() const { return started_; }
    bool interactionEnded() const { return ended_; }

private:
    // Layout rect, in window pixels. Declared before behave(), which takes one.
    struct Rect { float x, y, w, h; };

    // The one press/drag/release state machine. Every clickable widget goes
    // through it, because eight hand-written copies is how three of them ended
    // up disagreeing and how `active_`'s lifetime came to contradict the input
    // the app actually produces.
    struct Response
    {
        bool hovered = false;       // pointer on it, and allowed to react
        bool pressed = false;       // press edge landed here; it now owns active_
        bool held = false;          // active_ == id: a drag is in progress
        bool released = false;      // it owned active_ and the release arrived
        bool clicked = false;       // released while STILL over it: the fire signal
        bool doubleClicked = false;
    };
    enum Behave : uint32_t
    {
        kNoCapture = 1,    // report the press without taking active_
        kFireOnPress = 2,  // toggles: act immediately, no release needed
    };
    Response behave(const Rect&, uint64_t id, uint32_t flags = 0);

    uint64_t idFor(std::string_view, uint64_t extra = 0) const;
    // The part of a label before "##", i.e. what is actually drawn.
    static std::string_view visibleLabel(std::string_view);
    void beginPanelImpl(std::string_view title, float x, float y, float w,
                        float maxH, bool docked);
    bool dragImpl(std::string_view label, double& v, double step, double lo,
                  double hi, int decimals, std::string_view unit,
                  double quantum = 0);
    // The shared editing core. Returns true when the edit was COMMITTED.
    bool editKeys(const TextOpts&);
    // Draws the box + text + selection + caret for the active or idle field.
    void drawTextBox(const Rect& box, std::string_view shown, bool active,
                     bool hovered, std::string_view hint, FontId,
                     float fill = -1);
    static bool acceptChar(char, Charset);
    void pushBatch();
    void setScissor(float x, float y, float w, float h);
    void clearScissor();
    // Advances the layout cursor by one row and returns its rect.
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

    uint64_t hot_ = 0, active_ = 0;
    // Splitter bars registered this frame, painted by flushSplitters().
    struct SplitBar
    {
        float x = 0, y = 0, w = 0, h = 0;
        bool vertical = false, over = false, dragging = false;
    };
    std::vector<SplitBar> splits_;
    // The CONTINUOUS value behind the field being dragged. The committed value
    // is a snap of this, never the accumulator of its own increments: a field
    // whose quantum is coarser than one pixel of travel (nz stepped 0.25 cells
    // per pixel) otherwise re-rounded to its old value every frame and could
    // not be dragged at all, and a field held against lo/hi built up an
    // invisible debt that had to be dragged back off before it would move.
    // Clamping the continuous value is what kills both.
    double dragCont_ = 0;
    double dragLast_ = 0;   // what we last wrote, to detect a caller re-clamp
    // One text editor, shared by textInput and the numeric drag fields: two
    // implementations is how the numeric one ended up with no caret, no
    // selection and no clipboard.
    struct TextEdit
    {
        uint64_t id = 0;
        std::string buf, original;
        int caret = 0, anchor = 0;   // byte offsets; equal = no selection
        float scrollX = 0;
    };
    TextEdit edit_;
    bool hoverPanel_ = false;
    int rowCounter_ = 0;
    // Latched at begin(): a frame that STARTED in a text edit must never leak
    // keys to the app, or Esc-cancelling an edit falls through to the global
    // Esc handler and quits the application.
    bool keyboardCaptured_ = false;
    bool arrowsConsumed_ = false;
    bool gestureHeld_ = false;
    bool started_ = false, ended_ = false;
    uint64_t prevInteract_ = 0;
};

} // namespace ui
