// Headless coverage for the widget layer.
//
// The widget set had no automated test at all, and the cost of that was
// concrete: `Gui::begin` retired `active_` on `!lDown` while SDL delivers
// `lDown=false` and `lReleased=true` in the SAME frame, so every release-gated
// widget in the application -- every button, every button row -- could only
// fire if press and release happened to land in one polled frame. The app
// looked alive because checkbox/choice/selectable fire on press.
//
// So the first test here is the press/hold/release sequence, and it is the one
// that must never be deleted.
//
// Everything is driven through the PUBLIC contract: return values, vertices(),
// batches(), capturingMouse(). No window, no Vulkan device, no GPU -- the text
// engine comes from TextEngine::createHeadless(), which shapes and measures
// with the real font but owns no atlas.
#include "ui/guicheck.hpp"

#include "ui/gui.hpp"
#include "ui/text.hpp"

#include <cstdio>
#include <string>

namespace ui {
namespace {

int failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("   %-52s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}

// Drives a Gui with synthesized input, one frame at a time.
struct Driver
{
    Input in;
    Gui& g;

    template <class F>
    void frame(F&& body)
    {
        g.begin(in, 1600, 950);
        body();
        g.end();
        in.clearFrame();
    }
    void moveTo(float x, float y)
    {
        in.dx = x - in.mx;
        in.dy = y - in.my;
        in.mx = x;
        in.my = y;
    }
    void press() { in.lDown = true; in.lPressed = true; }
    void hit(SDL_Keycode k) { in.keys.push_back(k); }
    void release() { in.lDown = false; in.lReleased = true; }
};

// A panel wide enough that a row's button spans a predictable rect. The panel
// is placed at the origin so row 0 sits just under the header.
constexpr float kPanelX = 0, kPanelY = 0, kPanelW = 300, kPanelH = 600;

// A point inside the first row of the panel, and one far outside it.
constexpr float kInX = 60, kInY = 44;
constexpr float kOutX = 900, kOutY = 700;

void checkClickSequence(Gui& g)
{
    Driver d{{}, g};
    int fired = 0;
    auto oneButton = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        if (g.button("go")) ++fired;
        g.endPanel();
    };

    // THE regression: press, hold, release, idle.
    d.moveTo(kInX, kInY);
    d.frame(oneButton);
    expect(fired == 0, "press frame does not fire");
    d.press();
    d.frame(oneButton);
    expect(fired == 0, "press edge does not fire");
    d.frame(oneButton);
    expect(fired == 0, "held does not fire");
    d.release();
    d.frame(oneButton);
    expect(fired == 1, "RELEASE fires exactly once");
    d.frame(oneButton);
    expect(fired == 1, "idle after release does not fire again");

    // Press on, release off: cancelled. Every real button has this affordance.
    fired = 0;
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(oneButton);
    d.moveTo(kOutX, kOutY);
    d.frame(oneButton);
    d.release();
    d.frame(oneButton);
    expect(fired == 0, "press on, release off is cancelled");

    // Press off, release on: never captured, must not fire.
    fired = 0;
    d.moveTo(kOutX, kOutY);
    d.press();
    d.frame(oneButton);
    d.moveTo(kInX, kInY);
    d.release();
    d.frame(oneButton);
    expect(fired == 0, "press off, release on does not fire");

    // A fast click coalesced into one polled frame still fires.
    fired = 0;
    d.moveTo(kInX, kInY);
    d.press();
    d.in.lDown = false;
    d.in.lReleased = true;
    d.frame(oneButton);
    expect(fired == 1, "press+release in one frame fires once");
}

// A caption that changes between press and release must not change identity --
// otherwise the click silently does nothing. That is what "##key" is for.
void checkToggleIdentity(Gui& g)
{
    Driver d{{}, g};
    bool state = false;
    int fired = 0;
    auto stable = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        if (g.button(state ? "hide me##t" : "show me##t")) {
            ++fired;
            state = !state;
        }
        g.endPanel();
    };
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(stable);
    state = !state;   // caption flips mid-interaction
    d.release();
    d.frame(stable);
    expect(fired == 1, "caption change between press/release still fires (##key)");

    // And the label before "##" is what gets drawn: no "##" in the vertex
    // stream is not directly observable, but the width is.
    const float w1 = g.textWidth("abc");
    const float w2 = g.textWidth("abc##ignored-suffix");
    expect(w1 > 0 && w2 > w1, "textWidth measures the raw string (caller strips)");
}

// One active widget at a time: pressing A and releasing over B must fire
// neither. Without this two overlapping widgets both take the same press.
void checkExclusive(Gui& g)
{
    Driver d{{}, g};
    int a = 0, b = 0;
    auto two = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        if (g.button("A")) ++a;
        if (g.button("B")) ++b;
        g.endPanel();
    };
    // Row 0 and row 1; rowH+gap apart. Probe the second row generously.
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(two);
    d.moveTo(kInX, kInY + 40);
    d.release();
    d.frame(two);
    expect(a == 0 && b == 0, "press A, release over B fires neither");
}

// capturingMouse() must be true for the whole gesture, press through release
// inclusive, or the viewport steals the click that a widget is consuming.
void checkCapture(Gui& g)
{
    Driver d{{}, g};
    bool capturedDuring = true, capturedAtRelease = false;
    auto one = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        (void)g.button("go");
        g.endPanel();
    };
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(one);
    capturedDuring = g.capturingMouse();
    d.moveTo(kOutX, kOutY);   // drag off the panel, still holding
    d.frame(one);
    capturedDuring = capturedDuring && g.capturingMouse();
    d.release();
    d.frame(one);
    capturedAtRelease = g.capturingMouse();
    expect(capturedDuring, "capturingMouse holds while dragging off the panel");
    expect(capturedAtRelease, "capturingMouse still true on the release frame");
}

// A press in EMPTY SPACE (the 3D viewport) must not be captured, or the app
// skips picking and dragging entirely. This is the regression for "moving by
// dragging does NOT work": capturingMouse() was latched on any lPressed rather
// than on a widget actually owning the gesture.
void checkViewportNotCaptured(Gui& g)
{
    Driver d{{}, g};
    auto panelOnly = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        (void)g.button("go");
        g.endPanel();
    };
    // Press far from any panel: the viewport owns this click.
    d.moveTo(kOutX, kOutY);
    d.press();
    d.frame(panelOnly);
    const bool freeOnPress = !g.capturingMouse();
    d.moveTo(kOutX + 20, kOutY + 20);
    d.frame(panelOnly);
    const bool freeOnDrag = !g.capturingMouse();
    d.release();
    d.frame(panelOnly);
    const bool freeOnRelease = !g.capturingMouse();
    expect(freeOnPress, "a press in empty space is NOT captured by the UI");
    expect(freeOnDrag, "dragging in empty space stays uncaptured");
    expect(freeOnRelease, "releasing in empty space stays uncaptured");

    // ...but a press that starts on a widget IS captured for the whole gesture.
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(panelOnly);
    expect(g.capturingMouse(), "a press on a widget IS captured");
    // Leave the Gui idle: a test that ends mid-press keeps active_ held, and
    // the NEXT test's press is then refused (one active widget at a time).
    d.release();
    d.frame(panelOnly);
    d.frame(panelOnly);
}

void checkButtonRow(Gui& g)
{
    Driver d{{}, g};
    int clicked = -1;
    static const char* const labels[4] = {"+cyl", "+sph", "+cube", "+lens"};
    auto rowUi = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        if (const int k = g.buttonRow({labels, 4}); k >= 0) clicked = k;
        g.endPanel();
    };
    // Third segment of four across a 300 px panel (16 px padding each side).
    const float segW = (kPanelW - 2 * 8) / 4.0f;
    d.moveTo(8 + segW * 2.5f, kInY);
    d.press();
    d.frame(rowUi);
    expect(clicked == -1, "buttonRow does not fire on press");
    d.release();
    d.frame(rowUi);
    expect(clicked == 2, "buttonRow fires the segment under the cursor");

    // Segments must be distinct widgets even though they share a row.
    clicked = -1;
    d.moveTo(8 + segW * 0.5f, kInY);
    d.press();
    d.frame(rowUi);
    d.release();
    d.frame(rowUi);
    expect(clicked == 0, "buttonRow segments have independent identity");
}

void checkDrag(Gui& g)
{
    Driver d{{}, g};
    double v = 10.0;
    auto slider = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        (void)g.dragDouble("v", v, 1.0, -1e6, 1e6, 3);
        g.endPanel();
    };
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(slider);
    for (int i = 0; i < 3; ++i) {
        d.moveTo(d.in.mx + 10, kInY);
        d.frame(slider);
    }
    expect(std::abs(v - 40.0) < 1e-9, "drag accumulates 3 x 10 px x step 1");

    // Shift is the fine modifier: a tenth per pixel.
    d.release();
    d.frame(slider);
    v = 0;
    d.in.shift = true;
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(slider);
    d.moveTo(kInX + 10, kInY);
    d.frame(slider);
    expect(std::abs(v - 1.0) < 1e-9, "shift drag is 0.1x per pixel");
    d.in.shift = false;
    d.release();
    d.frame(slider);

    // Clamping.
    v = 0;
    double lo = 0;
    auto clamped = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        (void)g.dragDouble("c", lo, 1.0, 0.0, 5.0, 3);
        g.endPanel();
    };
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(clamped);
    d.moveTo(kInX + 500, kInY);
    d.frame(clamped);
    expect(lo <= 5.0 + 1e-9, "drag clamps at the upper bound");
}

// The drag LAW, as opposed to the drag gesture checkDrag covers. Four separate
// complaints of "the sliders are weird" reduce to the three defects pinned
// here, plus the position track.
void checkDragLaw(Gui& g)
{
    // --- the two pure rules --------------------------------------------------
    // A signed range is linear everywhere: an object at x = 1000 must not drag
    // a thousand times faster than one at the origin.
    expect(Gui::dragRate(0, 1.0, -10) == 1.0, "signed range: linear at zero");
    expect(Gui::dragRate(1000, 1.0, -10) == 1.0, "signed range: linear far out");
    // A non-negative range is a scale: 1% of the value per pixel on top of step.
    expect(std::abs(Gui::dragRate(100, 1.0, 0) - 2.0) < 1e-12,
           "scale range: step + 1% of the value per pixel");

    expect(Gui::dragFill(5, -1e4, 1e4) < 0, "no track for a signed sentinel range");
    expect(std::abs(Gui::dragFill(0.1, 0, 1) - 0.1) < 1e-12,
           "a range starting at zero maps linearly (LES Cs is not a decade knob)");
    // [8, 4096] spans 2.7 decades, so it is log-mapped: a 128-cell grid reads
    // as mid-track rather than the 3% a linear map would show.
    expect(std::abs(Gui::dragFill(128, 8, 4096) - 0.4444) < 1e-3,
           "a multi-decade range is log-mapped");
    // The claim app.cpp's diameter field makes by choosing a D-relative range.
    expect(std::abs(Gui::dragFill(0.05, 0.05e-3, 0.05e3) - 0.5) < 1e-12,
           "an object at exactly D sits at mid-track");

    // --- no debt at a limit --------------------------------------------------
    // Held against hi, the old field accumulated the whole overshoot and then
    // refused to move until it had been dragged back off.
    Driver d{{}, g};
    double v = 0;
    auto knob = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        (void)g.dragDouble("k", v, 1.0, -100.0, 5.0, 3);
        g.endPanel();
    };
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(knob);
    d.moveTo(kInX + 400, kInY);       // slam into hi with 395 px to spare
    d.frame(knob);
    expect(v == 5.0, "drag pins at the upper bound");
    d.moveTo(kInX + 390, kInY);       // ten pixels back
    d.frame(knob);
    expect(v < 5.0, "dragging back off a limit moves the value at once");
    d.release();
    d.frame(knob);

    // --- sub-quantum travel survives -----------------------------------------
    // `cells z` steps 0.25 per pixel with a quantum of 1. Rounding to the same
    // int every frame and re-reading it discarded the fraction, so a slow drag
    // moved it not at all.
    d.in = {};
    int nz = 1;
    auto cells = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        (void)g.dragInt("nz", nz, 0.25, 1, 2048);
        g.endPanel();
    };
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(cells);
    for (int i = 0; i < 8; ++i) {     // eight frames of one pixel each
        d.moveTo(d.in.mx + 1, kInY);
        d.frame(cells);
    }
    expect(nz > 1, "a one-pixel-per-frame drag moves an integer field");
    d.release();
    d.frame(cells);

    // ...and the same field is traversable: 8 -> 4096 used to want 4088 px.
    d.in = {};
    int nx = 8;
    auto cellsX = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        (void)g.dragInt("nx", nx, 0.5, 8, 4096);
        g.endPanel();
    };
    d.moveTo(kInX, kInY);
    d.press();
    d.frame(cellsX);
    for (int i = 0; i < 500; ++i) {
        d.moveTo(d.in.mx + 1, kInY);
        d.frame(cellsX);
    }
    expect(nx == 4096, "500 px reaches the top of the cell-count range");
    d.release();
    d.frame(cellsX);

    // --- arrow nudge while hovered -------------------------------------------
    d.in = {};
    int copies = 8;
    auto arrows = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        (void)g.dragInt("copies", copies, 0.05, 2, 64);
        g.endPanel();
    };
    d.moveTo(kInX, kInY);
    d.frame(arrows);
    bool claimed = false;
    d.hit(SDLK_RIGHT);
    d.frame([&] { arrows(); claimed = g.consumedArrows(); });
    expect(copies == 9, "right arrow over a field nudges one quantum");
    expect(claimed, "the field CLAIMS the arrow, so the timeline does not scrub");
    d.hit(SDLK_LEFT);
    d.frame(arrows);
    expect(copies == 8, "left arrow nudges back");
    // Not when the pointer is elsewhere: the Live panel's rewind keys must
    // still reach it.
    d.moveTo(kOutX, kOutY);
    d.frame(arrows);
    claimed = true;
    d.hit(SDLK_RIGHT);
    d.frame([&] { arrows(); claimed = g.consumedArrows(); });
    expect(copies == 8, "arrows do not reach an unhovered field");
    expect(!claimed, "an unhovered field leaves the arrow for the timeline");
}

// Structural invariants: a lost pushBatch silently drops a draw, and a scissor
// outside the framebuffer is a Vulkan validation error rather than a visual bug.
// The text editor: caret, selection, clipboard, charset, commit/revert. The
// charset test is the load-bearing one -- it pins Charset::Token to the same
// rule dns::validSceneName enforces, instead of restating it in a comment.
void checkTextInput(Gui& g)
{
    static std::string clip;
    g.clipboard.get = [] { return clip; };
    g.clipboard.set = [](std::string_view v) { clip = std::string(v); };

    Driver d{{}, g};
    std::string value = "abc";
    int commits = 0;
    TextOpts opts;
    opts.charset = Charset::Token;
    opts.maxLen = 16;
    auto field = [&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        if (g.textInput("name", value, opts)) ++commits;
        g.endPanel();
    };

    // Focus by clicking the box (right half of the row).
    d.moveTo(kPanelW - 40, kInY);
    d.press();
    d.frame(field);
    d.release();
    d.frame(field);
    expect(g.editingText(), "clicking a text field focuses it");
    expect(g.caretIndex() == 3, "focus puts the caret at the end");

    // selectAllOnFocus: typing replaces.
    d.in.text = "xy";
    d.frame(field);
    expect(commits == 0, "typing does not commit");
    d.hit(SDLK_RETURN);
    d.frame(field);
    expect(commits == 1 && value == "xy", "Enter commits the typed value");

    // Esc reverts.
    d.moveTo(kPanelW - 40, kInY);
    d.press(); d.frame(field); d.release(); d.frame(field);
    d.in.text = "zzz";
    d.frame(field);
    d.hit(SDLK_ESCAPE);
    d.frame(field);
    expect(value == "xy", "Esc reverts the edit");
    expect(!g.editingText(), "Esc leaves the field");

    // Charset: illegal characters are refused at the keystroke.
    value = "";
    d.moveTo(kPanelW - 40, kInY);
    d.press(); d.frame(field); d.release(); d.frame(field);
    d.in.text = "a b/c.d-e_f";
    d.frame(field);
    d.hit(SDLK_RETURN);
    d.frame(field);
    expect(value == "abcd-e_f", "Charset::Token drops space, slash and dot");

    // Clipboard round trip, and the charset applies to a paste too.
    clip = "no spaces here";
    value = "";
    d.moveTo(kPanelW - 40, kInY);
    d.press(); d.frame(field); d.release(); d.frame(field);
    d.in.ctrl = true;
    d.hit(SDLK_V);
    d.frame(field);
    d.in.ctrl = false;
    d.hit(SDLK_RETURN);
    d.frame(field);
    expect(value == "nospaceshere", "paste is filtered by the charset");

    // maxLen.
    value = "";
    d.moveTo(kPanelW - 40, kInY);
    d.press(); d.frame(field); d.release(); d.frame(field);
    d.in.text = "0123456789abcdefghij";   // 20 > maxLen 16
    d.frame(field);
    d.hit(SDLK_RETURN);
    d.frame(field);
    expect(value.size() == 16, "maxLen is enforced");

    // Backspace at the start and Delete at the end are no-ops, not underflows.
    value = "ab";
    d.moveTo(kPanelW - 40, kInY);
    d.press(); d.frame(field); d.release(); d.frame(field);
    d.hit(SDLK_HOME);
    d.frame(field);
    d.hit(SDLK_BACKSPACE);
    d.hit(SDLK_BACKSPACE);
    d.frame(field);
    d.hit(SDLK_END);
    d.frame(field);
    d.hit(SDLK_DELETE);
    d.frame(field);
    d.hit(SDLK_RETURN);
    d.frame(field);
    expect(value == "ab", "Backspace at 0 and Delete at end are no-ops");

    // The app must not see keys while a field is being edited, or Esc quits.
    d.moveTo(kPanelW - 40, kInY);
    d.press(); d.frame(field); d.release(); d.frame(field);
    expect(g.capturingKeyboard(), "an open editor captures the keyboard");
    d.hit(SDLK_ESCAPE);
    d.frame(field);
    d.frame(field);
    expect(!g.capturingKeyboard(), "capture ends once the editor closes");
}

void checkBatches(Gui& g)
{
    Driver d{{}, g};
    bool sectionOpen = true;
    d.frame([&] {
        g.beginPanel("p", kPanelX, kPanelY, kPanelW, kPanelH);
        g.label("hello");
        if (g.sectionHeader("sec", sectionOpen)) {
            g.field("k", "v");
            g.separator();
            bool cb = false;
            (void)g.checkbox("cb", cb);
        }
        g.endPanel();
        g.beginRegion("r", 400, 0, 300, 400);
        g.label("world");
        g.endPanel();
    });
    std::size_t total = 0;
    bool empties = false, outside = false;
    for (const auto& b : g.batches()) {
        total += b.count;
        if (b.count == 0) empties = true;
        if (b.scissor.offset.x < 0 || b.scissor.offset.y < 0 ||
            b.scissor.offset.x + int32_t(b.scissor.extent.width) > 1600 ||
            b.scissor.offset.y + int32_t(b.scissor.extent.height) > 950)
            outside = true;
    }
    expect(total == g.vertices().size(), "every vertex belongs to exactly one batch");
    expect(!empties, "no empty batches are emitted");
    expect(!outside, "every scissor lies inside the framebuffer");
}

// Wheel scrolling must affect only the region under the pointer. When this was
// keyed on the frame-wide hoverPanel_ flag, scrolling anywhere scrolled every
// region -- which is what pushed the body-kind buttons out of view.
void checkScrollIsolation(Gui& g)
{
    Driver d{{}, g};
    float firstA = 0, firstB = 0;
    auto twoRegions = [&](float* outA, float* outB) {
        return [&, outA, outB] {
            g.beginRegion("A", 0, 0, 300, 120);
            *outA = g.cursorY();
            for (int i = 0; i < 20; ++i) g.label("row");
            g.endPanel();
            g.beginRegion("B", 400, 0, 300, 120);
            *outB = g.cursorY();
            for (int i = 0; i < 20; ++i) g.label("row");
            g.endPanel();
        };
    };
    d.moveTo(150, 60);            // over region A
    d.frame(twoRegions(&firstA, &firstB));
    const float baseA = firstA, baseB = firstB;
    d.in.wheel = -3;
    d.frame(twoRegions(&firstA, &firstB));
    d.in.wheel = 0;
    d.frame(twoRegions(&firstA, &firstB));
    expect(firstA < baseA - 1.0f, "wheel over region A scrolls A");
    expect(std::abs(firstB - baseB) < 0.01f, "wheel over region A leaves B alone");
}

// The region dividers -- and specifically the horizontal one, which is the
// bottom dock's only resize handle. It was drawn BEFORE the regions, so the
// panels painted over it and the handle was invisible: the dock read as fixed
// height. So this pins two things: that the drag moves the edge, and that the
// bar is still on the vertex list after a region has been drawn over its rect.
void checkSplitter(Gui& g)
{
    Driver d{{}, g};
    float dockTop = 700;
    auto layout = [&] {
        g.splitter("splitD", 0, dockTop - 6, 1600, 6, dockTop, false, 200, 900);
        g.beginRegion("Run", 0, dockTop, 1600, 220);   // painted over the bar
        g.label("plots");
        g.endPanel();
        g.flushSplitters();
    };

    d.moveTo(800, 697);
    d.frame(layout);
    const float before = dockTop;
    d.press();
    d.frame(layout);
    d.moveTo(800, 603);           // drag up 100 px
    d.frame(layout);
    d.release();
    d.frame(layout);
    expect(dockTop < before - 90.0f, "dragging the divider moves the edge");
    expect(dockTop > before - 110.0f, "the edge follows the pointer 1:1");

    // Clamped, not runaway: the dock can never eat the menu or the status bar.
    d.in = {};
    d.moveTo(800, dockTop - 3);
    d.frame(layout);
    d.press();
    d.frame(layout);
    d.moveTo(800, -4000);
    d.frame(layout);
    d.release();
    d.frame(layout);
    expect(dockTop == 200.0f, "the divider clamps to its limit");

    // ...and the bar survives the region drawn on top of it. The splitter's
    // pixels must be the LAST ones written in that band.
    d.in = {};
    d.moveTo(0, 0);
    std::size_t before2 = 0;
    d.frame([&] {
        g.splitter("splitD2", 0, 694, 1600, 6, dockTop, false, 200, 900);
        before2 = g.vertices().size();
        g.beginRegion("Run", 0, 700, 1600, 220);
        g.label("plots");
        g.endPanel();
        g.flushSplitters();
        expect(g.vertices().size() > before2 + 6,
               "the divider is painted after the regions");
    });
}

} // namespace

int guiCheck()
{
    failures = 0;
    std::printf("\n== widgets\n");
    auto text = TextEngine::createHeadless();
    if (!text) {
        std::printf("   SKIP: %s\n", text.error().c_str());
        return 0;   // no font on this machine; not a failure
    }
    Gui g(**text);
    // Between checks: two frames with nothing held, so active_/edit_ retire.
    // A check that ends mid-gesture would otherwise block the next one's press
    // -- which is exactly how four text-input assertions failed spuriously.
    Driver idleDrv{{}, g};
    auto idle = [&] {
        idleDrv.in = {};
        idleDrv.frame([] {});
        idleDrv.frame([] {});
    };
    checkClickSequence(g);
    idle();
    checkToggleIdentity(g);
    idle();
    checkExclusive(g);
    idle();
    checkCapture(g);
    idle();
    checkViewportNotCaptured(g);
    idle();
    checkButtonRow(g);
    idle();
    checkDrag(g);
    idle();
    checkDragLaw(g);
    idle();
    checkTextInput(g);
    idle();
    checkBatches(g);
    idle();
    checkScrollIsolation(g);
    idle();
    checkSplitter(g);
    return failures;
}

} // namespace ui
