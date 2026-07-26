#include "ui/gui.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ui {
namespace {

// FNV-1a over the label. Widget identity is its label within its panel, which
// is why every label in a repeated row carries the object index.
uint64_t fnv(std::string_view s, uint64_t h = 1469598103934665603ull)
{
    for (char c : s) {
        h ^= uint8_t(c);
        h *= 1099511628211ull;
    }
    return h;
}

std::string format(double v, int decimals)
{
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", decimals, v);
    // Trim trailing zeros so 512.000 reads as 512 -- a resolution panel full
    // of padded zeros is harder to scan, not more precise.
    std::string s(b);
    if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s;
}

Rgba mix(Rgba a, Rgba b, float t)
{
    auto f = [&](uint8_t x, uint8_t y) {
        return uint8_t(float(x) + (float(y) - float(x)) * t);
    };
    return {f(a.r, b.r), f(a.g, b.g), f(a.b, b.b), f(a.a, b.a)};
}

} // namespace

bool Input::key(SDL_Keycode k) const
{
    return std::find(keys.begin(), keys.end(), k) != keys.end();
}

void Input::clearFrame()
{
    dx = dy = wheel = 0;
    lPressed = lReleased = rPressed = false;
    doubleClick = false;
    text.clear();
    keys.clear();
}

// --- primitive emission -----------------------------------------------------

void Gui::tri(float x0, float y0, float x1, float y1, float x2, float y2, Rgba c)
{
    verts_.push_back({{x0, y0}, {-1, -1}, c});
    verts_.push_back({{x1, y1}, {-1, -1}, c});
    verts_.push_back({{x2, y2}, {-1, -1}, c});
}

void Gui::rect(float x, float y, float w, float h, Rgba c, float radius)
{
    if (w <= 0 || h <= 0) return;
    const float r = std::min(radius, std::min(w, h) * 0.5f);
    if (r <= 0.5f) {
        tri(x, y, x + w, y, x + w, y + h, c);
        tri(x, y, x + w, y + h, x, y + h, c);
        return;
    }
    // Middle slab plus two end slabs plus four corner fans. Corners are
    // tessellated rather than shaded because a coverage-based rounded rect
    // would need its own pipeline for one visual detail.
    tri(x, y + r, x + w, y + r, x + w, y + h - r, c);
    tri(x, y + r, x + w, y + h - r, x, y + h - r, c);
    tri(x + r, y, x + w - r, y, x + w - r, y + r, c);
    tri(x + r, y, x + w - r, y + r, x + r, y + r, c);
    tri(x + r, y + h - r, x + w - r, y + h - r, x + w - r, y + h, c);
    tri(x + r, y + h - r, x + w - r, y + h, x + r, y + h, c);

    const int seg = 5;
    const float cx[4] = {x + r, x + w - r, x + w - r, x + r};
    const float cy[4] = {y + r, y + r, y + h - r, y + h - r};
    const float a0[4] = {180.f, 270.f, 0.f, 90.f};
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < seg; ++i) {
            const float t0 = (a0[k] + 90.0f * float(i) / seg) * 3.14159265f / 180.0f;
            const float t1 =
                (a0[k] + 90.0f * float(i + 1) / seg) * 3.14159265f / 180.0f;
            tri(cx[k], cy[k], cx[k] + r * std::cos(t0), cy[k] + r * std::sin(t0),
                cx[k] + r * std::cos(t1), cy[k] + r * std::sin(t1), c);
        }
}

void Gui::rectOutline(float x, float y, float w, float h, Rgba c, float t)
{
    rect(x, y, w, t, c);
    rect(x, y + h - t, w, t, c);
    rect(x, y, t, h, c);
    rect(x + w - t, y, t, h, c);
}

void Gui::textAt(std::string_view s, float x, float baselineY, Rgba c, FontId f,
                 float px)
{
    text_->append(verts_, f, s, x, baselineY, px > 0 ? px : theme.fontPx, c);
}

float Gui::textWidth(std::string_view s, FontId f, float px)
{
    return text_->measure(f, s, px > 0 ? px : theme.fontPx);
}

// --- frame ------------------------------------------------------------------

void Gui::begin(const Input& in, float w, float h)
{
    in_ = in;
    screenW_ = w;
    screenH_ = h;
    verts_.clear();
    batches_.clear();
    batchStart_ = 0;
    scissor_ = {{0, 0}, {uint32_t(w), uint32_t(h)}};
    hot_ = 0;
    hoverPanel_ = false;
    rowCounter_ = 0;
    splits_.clear();

    // active_ MUST survive the frame in which the release edge arrives. SDL
    // delivers lDown=false together with lReleased=true, so retiring it on
    // !lDown alone kills the widget one instruction before it can observe its
    // own release -- which is why every button in this application was dead.
    // The identity is retired in end(), once every widget has been asked.
    if (!in_.lDown && !in_.lReleased) {
        active_ = 0;
        dragCont_ = dragLast_ = 0;
    }
    // Latched for the whole frame: see the member comments.
    keyboardCaptured_ = edit_.id != 0;
    arrowsConsumed_ = false;
    // A WIDGET owns the gesture -- not merely "a button is down". Including
    // in_.lPressed here made capturingMouse() true for a press in the VIEWPORT,
    // so the app skipped picking and dragging entirely.
    gestureHeld_ = active_ != 0;

    const uint64_t interact = active_ ? active_ : edit_.id;
    started_ = interact != 0 && prevInteract_ == 0;
    ended_ = interact == 0 && prevInteract_ != 0;
    prevInteract_ = interact;
}

void Gui::end()
{
    pushBatch();
    // A widget that took the press THIS frame must also report capture, since
    // the app reads capturingMouse() after the frame is built.
    gestureHeld_ = gestureHeld_ || active_ != 0;
    if (in_.lReleased) {
        active_ = 0;
        dragCont_ = dragLast_ = 0;
    }
}

// The one behaviour function. `clicked` requires the release to land ON the
// widget, so dragging off cancels -- the affordance every real button has and
// none of the hand-rolled copies did.
Gui::Response Gui::behave(const Rect& r, uint64_t id, uint32_t flags)
{
    Response o;
    // While something else owns the drag nothing else is hoverable; without
    // this two overlapping widgets both take the same press.
    const bool inside = hit(r);
    if (inside && (active_ == 0 || active_ == id)) {
        hot_ = id;
        o.hovered = true;
    }
    if (o.hovered && in_.lPressed && active_ == 0) {
        o.pressed = true;
        o.doubleClicked = in_.doubleClick;
        if (!(flags & kNoCapture)) active_ = id;
    }
    o.held = active_ == id;
    if (o.held && in_.lReleased) {
        o.released = true;
        o.clicked = inside;
    }
    if (flags & kFireOnPress) o.clicked = o.pressed;
    return o;
}

void Gui::pushBatch()
{
    const uint32_t n = uint32_t(verts_.size()) - batchStart_;
    if (n) batches_.push_back({batchStart_, n, scissor_});
    batchStart_ = uint32_t(verts_.size());
}

void Gui::setScissor(float x, float y, float w, float h)
{
    pushBatch();
    const float x0 = std::max(0.0f, x), y0 = std::max(0.0f, y);
    const float x1 = std::min(screenW_, x + w), y1 = std::min(screenH_, y + h);
    scissor_ = {{int32_t(x0), int32_t(y0)},
                {uint32_t(std::max(0.0f, x1 - x0)), uint32_t(std::max(0.0f, y1 - y0))}};
}

void Gui::clearScissor()
{
    pushBatch();
    scissor_ = {{0, 0}, {uint32_t(screenW_), uint32_t(screenH_)}};
}

// Text after "##" is IDENTITY ONLY and is never drawn. A button whose caption
// toggles ("show"/"hide resolved case") is ONE widget with two captions; keying
// it on the caption changes its identity between press and release, which is a
// click that silently does nothing. `extra` separates repeated rows.
uint64_t Gui::idFor(std::string_view s, uint64_t extra) const
{
    const size_t h = s.find("##");
    const std::string_view key = h == std::string_view::npos ? s : s.substr(h + 2);
    return fnv(key, fnv(panelName_)) ^ (extra * 1099511628211ull);
}

std::string_view Gui::visibleLabel(std::string_view s)
{
    const size_t h = s.find("##");
    return h == std::string_view::npos ? s : s.substr(0, h);
}

// --- panels -----------------------------------------------------------------

float Gui::panelHeight(std::string_view title) const
{
    auto it = panelH_.find(fnv(title));
    return it == panelH_.end() ? 0 : it->second;
}

void Gui::beginPanel(std::string_view title, float x, float y, float w, float maxH)
{
    beginPanelImpl(title, x, y, w, maxH, false);
}

void Gui::beginRegion(std::string_view title, float x, float y, float w, float h)
{
    beginPanelImpl(title, x, y, w, h, true);
}

void Gui::beginPanelImpl(std::string_view title, float x, float y, float w,
                         float maxH, bool docked)
{
    panelName_ = std::string(title);
    const uint64_t id = fnv(title);
    const float headH = theme.rowH + theme.px(4);
    float contentH = panelH_.count(id) ? panelH_[id] : 0;
    // Docked: the rect is the layout's decision, so take it verbatim.
    // Floating: shrink-wrap the content, capped by maxH.
    float h = docked ? maxH : headH + contentH + theme.pad;
    if (!docked && maxH > 0) h = std::min(h, maxH);
    const float rad = docked ? 0.0f : theme.radius;

    if (!docked)
        rect(x - theme.px(1), y - theme.px(1), w + theme.px(2), h + theme.px(2),
             theme.border, rad + theme.px(1));
    rect(x, y, w, h, theme.panel, rad);
    rect(x, y, w, headH, theme.panelHead, rad);
    if (rad > 0) rect(x, y + headH - rad, w, rad, theme.panelHead);
    textAt(title, x + theme.pad, y + headH - theme.px(8), theme.text);
    // A hairline in the accent colour under the title: with six panels open,
    // this is what separates them at a glance.
    rect(x, y + headH - theme.px(1), w, theme.px(1), theme.accentDim);

    // Hover for THIS panel, kept separate from the frame-wide hoverPanel_ flag.
    // They are not the same thing: hoverPanel_ is sticky across every panel
    // built this frame (it feeds capturingMouse()), so using it to gate the
    // wheel scrolled EVERY region whenever the pointer was over ANY of them --
    // which silently scrolled the body-kind buttons out of the Geometry region
    // and made them, in the user's words, impossible to see.
    const bool overThis =
        in_.mx >= x && in_.mx <= x + w && in_.my >= y && in_.my <= y + h;
    if (overThis) hoverPanel_ = true;

    cur_.panelX = x;
    cur_.panelY = y;
    cur_.panelW = w;
    cur_.panelTop = y + headH;
    cur_.inPanel = true;
    cur_.x = x + theme.pad;
    cur_.w = w - 2 * theme.pad;
    cur_.clipY0 = y + headH;
    cur_.clipY1 = y + h;

    float& scroll = scroll_[id];
    if (maxH > 0 && contentH > h - headH - theme.pad) {
        if (overThis && in_.wheel != 0) scroll -= in_.wheel * theme.px(40);
        scroll = std::clamp(scroll, 0.0f, contentH - (h - headH - theme.pad));
        // Scrollbar: a thin track on the right edge.
        const float trackH = h - headH - theme.pad;
        const float thumb = std::max(theme.px(20), trackH * trackH / contentH);
        const float t = scroll / (contentH - trackH);
        rect(x + w - theme.px(5), y + headH + t * (trackH - thumb), theme.px(4),
             thumb, theme.accentDim, theme.px(2));
    } else {
        scroll = 0;
    }
    cur_.y = y + headH + theme.gap - scroll;
    setScissor(x, cur_.clipY0, w, cur_.clipY1 - cur_.clipY0);
}

void Gui::endPanel()
{
    const uint64_t id = fnv(panelName_);
    panelH_[id] = cur_.y + scroll_[id] - cur_.panelTop - theme.gap + theme.gap;
    cur_.inPanel = false;
    panelName_.clear();
    clearScissor();
}

// A region divider. Deliberately NOT a panel widget: it lives in screen space
// between regions, so it takes an explicit rect and its own id rather than
// consuming a layout row.
bool Gui::splitter(std::string_view id, float x, float y, float w, float h,
                   float& pos, bool vertical, float lo, float hi)
{
    const uint64_t sid = fnv(id);
    const bool over = in_.mx >= x && in_.mx <= x + w && in_.my >= y &&
                      in_.my <= y + h;
    if (over) hot_ = sid;
    if (over && in_.lPressed) active_ = sid;

    const bool dragging = active_ == sid;
    if (dragging) {
        pos = std::clamp(pos + (vertical ? in_.dx : in_.dy), lo, hi);
        // A splitter drag must not also orbit the camera behind it.
        hoverPanel_ = true;
    }
    if (over) hoverPanel_ = true;

    splits_.push_back({x, y, w, h, vertical, over, dragging});
    return dragging;
}

void Gui::flushSplitters()
{
    // Drawn without a scissor: a splitter is chrome BETWEEN regions and must
    // never inherit the clip rect of whichever panel happened to be last.
    clearScissor();
    for (const SplitBar& s : splits_) {
        rect(s.x, s.y, s.w, s.h,
             s.dragging ? theme.accentDim
                        : (s.over ? theme.widgetHot : theme.panelHead));
        // A grip, so the bar reads as draggable rather than as a border. Three
        // short pips at the centre, along the bar: the same cue every docked
        // application uses, and the reason people find the handle at all.
        const Rgba pip = s.dragging || s.over ? theme.text : theme.textDim;
        const float t = std::max(1.0f, theme.px(1));
        const float len = theme.px(2);
        const float gap = theme.px(5);
        const float cx = s.x + s.w * 0.5f, cy = s.y + s.h * 0.5f;
        for (int i = -1; i <= 1; ++i) {
            if (s.vertical)
                rect(cx - t * 0.5f, cy + float(i) * gap - len * 0.5f, t, len, pip);
            else
                rect(cx + float(i) * gap - len * 0.5f, cy - t * 0.5f, len, t, pip);
        }
    }
    splits_.clear();
}

void Gui::strip(float x, float y, float w, float h, Rgba col)
{
    rect(x, y, w, h, col);
    if (in_.mx >= x && in_.mx <= x + w && in_.my >= y && in_.my <= y + h)
        hoverPanel_ = true;
}

Gui::Rect Gui::row(float h)
{
    const float rh = h > 0 ? h : theme.rowH;
    Rect r{cur_.x, cur_.y, cur_.w, rh};
    cur_.y += rh + theme.gap;
    ++rowCounter_;
    return r;
}

bool Gui::hit(const Rect& r) const
{
    // Rows scrolled out of the panel must not take the click.
    if (cur_.inPanel && (r.y + r.h < cur_.clipY0 || r.y > cur_.clipY1)) return false;
    return in_.mx >= r.x && in_.mx <= r.x + r.w && in_.my >= r.y &&
           in_.my <= r.y + r.h;
}

// --- widgets ----------------------------------------------------------------

void Gui::label(std::string_view s, Rgba col)
{
    const Rect r = row(theme.rowH - 4);
    textAt(s, r.x, r.y + r.h - theme.px(5), col);
}

void Gui::field(std::string_view key, std::string_view value, Rgba col)
{
    const Rect r = row(theme.rowH - 4);
    textAt(key, r.x, r.y + r.h - theme.px(5), theme.textDim, FontId::Sans, theme.smallPx);
    const float w = textWidth(value, FontId::Mono, theme.smallPx);
    textAt(value, r.x + r.w - w, r.y + r.h - theme.px(5), col, FontId::Mono, theme.smallPx);
}

void Gui::separator()
{
    const Rect r = row(7);
    rect(r.x, r.y + theme.px(3), r.w, theme.px(1), theme.border);
}

void Gui::spacing(float h)
{
    cur_.y += h;
}

bool Gui::sectionHeader(std::string_view title, bool& open)
{
    const Rect r = row(theme.rowH);
    const Response b = behave(r, idFor(title), kFireOnPress | kNoCapture);
    if (b.clicked) open = !open;
    rect(r.x, r.y, r.w, r.h, b.hovered ? theme.widgetHot : theme.widget,
         theme.radius);
    const char* mark = open ? "\xE2\x96\xBE" : "\xE2\x96\xB8";   // down / right
    textAt(mark, r.x + theme.px(6), r.y + r.h - theme.px(6), theme.accent);
    textAt(title, r.x + theme.px(22), r.y + r.h - theme.px(6), theme.text);
    return open;
}

bool Gui::button(std::string_view s, float w, Rgba tint)
{
    const float bw = w > 0 ? w : cur_.w;
    Rect r = row();
    r.w = bw;
    const Response b = behave(r, idFor(s));

    Rgba base = tint.a ? tint : theme.widget;
    if (b.held && b.hovered) base = theme.widgetActive;
    else if (b.hovered) base = mix(base, theme.widgetHot, 0.7f);
    rect(r.x, r.y, r.w, r.h, base, theme.radius);
    const std::string_view vis = visibleLabel(s);
    const float tw = textWidth(vis);
    textAt(vis, r.x + (r.w - tw) * 0.5f, r.y + r.h - theme.px(7), theme.text);
    return b.clicked;
}

int Gui::buttonRow(std::span<const char* const> labels)
{
    if (labels.empty()) return -1;
    const Rect r = row();
    const float seg = (r.w - theme.gap * float(labels.size() - 1)) /
                      float(labels.size());
    int clicked = -1;
    for (size_t i = 0; i < labels.size(); ++i) {
        Rect s{r.x + (seg + theme.gap) * float(i), r.y, seg, r.h};
        // The index is folded into the id: two rows sharing a label are two
        // widgets, not one.
        const Response b = behave(s, idFor(labels[i], i + 1));
        if (b.clicked) clicked = int(i);
        rect(s.x, s.y, s.w, s.h,
             (b.held && b.hovered) ? theme.widgetActive
             : b.hovered           ? theme.widgetHot
                                   : theme.widget,
             theme.radius);
        const std::string_view vis = visibleLabel(labels[i]);
        const float tw = textWidth(vis, FontId::Sans, theme.smallPx);
        textAt(vis, s.x + (s.w - tw) * 0.5f, s.y + s.h - theme.px(7), theme.text,
               FontId::Sans, theme.smallPx);
    }
    return clicked;
}

bool Gui::checkbox(std::string_view s, bool& v)
{
    const Rect r = row();
    // Toggles fire on PRESS: waiting for the release would make a checkbox feel
    // laggy, and there is no drag to cancel.
    const Response b = behave(r, idFor(s), kFireOnPress | kNoCapture);
    bool changed = false;
    if (b.clicked) {
        v = !v;
        changed = true;
    }
    const float box = theme.px(14);
    const float by = r.y + (r.h - box) * 0.5f;
    rect(r.x, by, box, box,
         v ? theme.accent : (b.hovered ? theme.widgetHot : theme.widget), 3);
    if (v) {
        // Check mark as two thick strokes; at 14 px a glyph would be mushy.
        rect(r.x + 3.5f, by + 6.5f, 4, 2, theme.panel);
        rect(r.x + 5.5f, by + 8.5f, 2, 2, theme.panel);
        rect(r.x + 6.5f, by + 4.5f, 5, 2, theme.panel);
    }
    textAt(visibleLabel(s), r.x + box + theme.px(8), r.y + r.h - theme.px(7),
           theme.text);
    return changed;
}

bool Gui::dragDouble(std::string_view lbl, double& v, double step, double lo,
                     double hi, int decimals)
{
    return dragImpl(lbl, v, step, lo, hi, decimals, {});
}

bool Gui::dragUnit(std::string_view lbl, double& v, double step, double lo,
                   double hi, std::string_view unit, int decimals)
{
    return dragImpl(lbl, v, step, lo, hi, decimals, unit);
}

// A pixel of drag is worth `step` near zero. On top of that, a quantity that
// cannot go negative (lo >= 0) is a SCALE, and a scale wants a constant
// FRACTION per pixel or it is unusable at one end of its range: diameter spans
// [1e-4, 1e4] x D, and no single absolute step both resolves the small end and
// reaches the large one. Signed quantities -- positions, array pitches -- stay
// purely linear, because dragging an object that sits at x = 50 D must not run
// fifty times faster than dragging one at the origin.
//
// 1% per pixel: ~70 px to double a value, ~430 px to take cells x from 8 to
// 4096. Before this, that sweep was 4088 px, and `cells z` (0.25 cells/px, a
// quantum of 1) could not be moved by a slow drag at all.
double Gui::dragRate(double v, double step, double lo)
{
    constexpr double kRelPerPx = 0.01;
    return step + (lo >= 0 ? kRelPerPx * std::abs(v) : 0.0);
}

double Gui::dragFill(double v, double lo, double hi)
{
    // A signed or unbounded range is a sentinel (positions are declared
    // [-1e4, 1e4] m because there is no real bound), and a track drawn against
    // a sentinel says nothing true. Same test as dragRate's, deliberately: the
    // fields that drag as a scale are exactly the ones with a scale to show.
    if (!(lo >= 0) || !(hi > lo) || !std::isfinite(hi)) return -1;
    const double c = std::clamp(v, lo, hi);
    // Log where the range spans more than ~1.5 decades, so a 128-cell grid in
    // [8, 4096] reads as 44% rather than 3%. lo == 0 forces linear: a range
    // that starts AT zero (LES Cs in [0,1]) is an ordinary linear knob, and
    // log-mapping it against an invented floor would park Cs = 0.1 at 83%.
    if (lo > 0 && hi / lo > 30.0)
        return (std::log(c) - std::log(lo)) / (std::log(hi) - std::log(lo));
    return (c - lo) / (hi - lo);
}

bool Gui::dragImpl(std::string_view lbl, double& v, double step, double lo,
                   double hi, int decimals, std::string_view unit,
                   double quantum)
{
    const Rect r = row();
    const uint64_t id = idFor(lbl);
    const float split = std::min(r.w * 0.52f, r.w - theme.px(70));
    const Rect box{r.x + split, r.y, r.w - split, r.h};

    // A drag field captures on press and follows the pointer anywhere, so it
    // wants `held`, not `clicked`. No press is taken while it is being typed
    // into, or clicking to place a caret would restart a drag.
    const Response b = edit_.id == id ? Response{} : behave(r, id);

    bool changed = false;
    if (edit_.id == id) {
        TextOpts o;
        o.charset = Charset::Number;
        o.maxLen = 32;
        const bool clickAway = in_.lPressed && !hit(box);
        if (editKeys(o) || clickAway) {
            try {
                v = std::clamp(std::stod(edit_.buf), lo, hi);
                changed = true;
            } catch (...) {
            }
            edit_ = {};
        }
    } else {
        // Snap to the caller's quantum: 0 means continuous (a length), 1 means
        // the value is really an int (dragInt) and the continuous shadow only
        // exists so sub-quantum travel is not thrown away every frame.
        auto snap = [quantum](double x) {
            return quantum > 0 ? std::round(x / quantum) * quantum : x;
        };
        if (b.pressed && b.doubleClicked) {
            edit_ = {id, format(v, decimals), format(v, decimals), 0, 0, 0};
            edit_.caret = int(edit_.buf.size());
            edit_.anchor = 0;
            active_ = 0;
        } else if (b.pressed) {
            dragCont_ = v;
            dragLast_ = v;
        }
        // NOT on the press frame: in_.dx there is the pointer's travel BEFORE
        // the button went down, and consuming it makes the value jump by
        // however far the cursor happened to move on its way to the field.
        if (b.held && !b.pressed && in_.dx != 0) {
            // A caller may re-clamp what we wrote (the diameter field floors
            // it, dragInt rounds it). Follow it rather than drag on from a
            // shadow value the model has already rejected.
            if (v != dragLast_) dragCont_ = v;
            // Shift = fine, ctrl = coarse. The resolution fields want a
            // one-cell nudge and a full-range sweep from the same gesture.
            const double mult = in_.shift ? 0.1 : (in_.ctrl ? 10.0 : 1.0);
            dragCont_ = std::clamp(
                dragCont_ + double(in_.dx) * dragRate(dragCont_, step, lo) * mult,
                lo, hi);
            const double nv = std::clamp(snap(dragCont_), lo, hi);
            if (nv != v) {
                v = nv;
                dragLast_ = nv;
                changed = true;
            }
        }
        // Arrow nudge while hovered: one quantum, or one `step` for a
        // continuous field. No focus concept is needed -- the pointer is the
        // focus, and being over the field means being over a panel, so the
        // viewport is not competing for the key.
        if (b.hovered && !in_.lDown) {
            const int dir = in_.key(SDLK_RIGHT) ? 1 : (in_.key(SDLK_LEFT) ? -1 : 0);
            if (dir != 0) {
                // Claim the key even if the value is already at its limit: the
                // timeline must not scrub just because the nudge had nowhere
                // to go.
                arrowsConsumed_ = true;
                const double q =
                    quantum > 0 ? quantum
                                : dragRate(v, step, lo) * (in_.shift ? 1.0 : 10.0);
                const double nv = std::clamp(snap(v + dir * q), lo, hi);
                if (nv != v) {
                    v = nv;
                    dragCont_ = nv;
                    dragLast_ = nv;
                    changed = true;
                }
            }
        }
    }

    textAt(visibleLabel(lbl), r.x, r.y + r.h - theme.px(7), theme.textDim);
    const bool active = edit_.id == id;
    // The unit rides along only in the display string: typing still types a
    // bare number, so an edit never has to reproduce " m/s".
    const std::string shown =
        active ? edit_.buf
               : format(v, decimals) +
                     (unit.empty() ? std::string() : " " + std::string(unit));
    drawTextBox(box, shown, active, b.hovered, {}, FontId::Mono,
                active ? -1.0f : float(dragFill(v, lo, hi)));
    return changed;
}

bool Gui::acceptChar(char c, Charset cs)
{
    const unsigned char u = static_cast<unsigned char>(c);
    switch (cs) {
    case Charset::Token:
        return std::isalnum(u) || c == '_' || c == '-';
    case Charset::Number:
        return (c >= '0' && c <= '9') || c == '.' || c == '-' || c == 'e' ||
               c == '+';
    case Charset::Path:
        return u >= 32 && u < 127;
    case Charset::Any:
    default:
        return u >= 32;
    }
}

// The shared editing core: every key that edits text goes through here, so the
// numeric drag fields get caret, selection and clipboard for free.
bool Gui::editKeys(const TextOpts& o)
{
    TextEdit& e = edit_;
    auto selRange = [&] {
        return std::pair<int, int>{std::min(e.caret, e.anchor),
                                   std::max(e.caret, e.anchor)};
    };
    auto eraseSel = [&] {
        auto [a, b] = selRange();
        if (a == b) return false;
        e.buf.erase(std::size_t(a), std::size_t(b - a));
        e.caret = e.anchor = a;
        return true;
    };

    for (char c : in_.text) {
        if (!acceptChar(c, o.charset)) continue;
        eraseSel();
        if (e.buf.size() >= o.maxLen) break;
        e.buf.insert(e.buf.begin() + e.caret, c);
        ++e.caret;
        e.anchor = e.caret;
    }

    for (SDL_Keycode k : in_.keys) {
        const bool sh = in_.shift, ct = in_.ctrl;
        switch (k) {
        case SDLK_LEFT:
            if (e.caret > 0) --e.caret;
            if (!sh) e.anchor = e.caret;
            break;
        case SDLK_RIGHT:
            if (e.caret < int(e.buf.size())) ++e.caret;
            if (!sh) e.anchor = e.caret;
            break;
        case SDLK_HOME:
            e.caret = 0;
            if (!sh) e.anchor = 0;
            break;
        case SDLK_END:
            e.caret = int(e.buf.size());
            if (!sh) e.anchor = e.caret;
            break;
        case SDLK_BACKSPACE:
            if (!eraseSel() && e.caret > 0) {
                e.buf.erase(std::size_t(e.caret - 1), 1);
                --e.caret;
                e.anchor = e.caret;
            }
            break;
        case SDLK_DELETE:
            if (!eraseSel() && e.caret < int(e.buf.size()))
                e.buf.erase(std::size_t(e.caret), 1);
            break;
        case SDLK_A:
            if (ct) {
                e.anchor = 0;
                e.caret = int(e.buf.size());
            }
            break;
        case SDLK_C:
        case SDLK_X:
            if (ct && clipboard.set) {
                auto [a, b] = selRange();
                if (a != b)
                    clipboard.set(std::string_view(e.buf).substr(
                        std::size_t(a), std::size_t(b - a)));
                if (k == SDLK_X) eraseSel();
            }
            break;
        case SDLK_V:
            if (ct && clipboard.get) {
                eraseSel();
                for (char c : clipboard.get()) {
                    if (!acceptChar(c, o.charset) || e.buf.size() >= o.maxLen)
                        continue;
                    e.buf.insert(e.buf.begin() + e.caret, c);
                    ++e.caret;
                }
                e.anchor = e.caret;
            }
            break;
        case SDLK_RETURN:
        case SDLK_KP_ENTER:
            return true;                 // commit
        case SDLK_ESCAPE:
            e.buf = e.original;          // revert, and stop editing
            e = {};
            return false;
        default:
            break;
        }
    }
    return false;
}

void Gui::drawTextBox(const Rect& box, std::string_view shown, bool active,
                      bool hovered, std::string_view hint, FontId font,
                      float fill)
{
    rect(box.x, box.y, box.w, box.h,
         active ? theme.widgetActive : (hovered ? theme.widgetHot : theme.widget),
         theme.radius);
    // The drag field's position-in-range track. Painted between the background
    // and the text so the value still reads over it; the caller passes < 0 for
    // a plain box (and while typing, when a half-lit box is just noise).
    if (fill >= 0) {
        const float w = std::clamp(fill, 0.0f, 1.0f) * box.w;
        if (w > 0.5f) rect(box.x, box.y, w, box.h, theme.accentDim, theme.radius);
    }
    if (active) rectOutline(box.x, box.y, box.w, box.h, theme.accent, theme.px(1));

    const float pad = theme.px(6);
    const float baseline = box.y + box.h - theme.px(7);
    if (!active) {
        // Idle: right-aligned, which is how every read-out in this UI reads.
        if (shown.empty() && !hint.empty()) {
            textAt(hint, box.x + pad, baseline, theme.textDim, font, theme.smallPx);
            return;
        }
        const float tw = textWidth(shown, font, theme.smallPx);
        textAt(shown, box.x + box.w - tw - pad, baseline, theme.text, font,
               theme.smallPx);
        return;
    }

    // Active: left-aligned with a caret, scrolled so the caret stays visible.
    const std::string_view buf = edit_.buf;
    const float caretX =
        textWidth(buf.substr(0, std::size_t(edit_.caret)), font, theme.smallPx);
    const float inner = box.w - 2 * pad;
    if (caretX - edit_.scrollX > inner) edit_.scrollX = caretX - inner;
    if (caretX < edit_.scrollX) edit_.scrollX = caretX;
    if (edit_.scrollX < 0) edit_.scrollX = 0;
    const float x0 = box.x + pad - edit_.scrollX;

    // Selection behind the glyphs.
    if (edit_.caret != edit_.anchor) {
        const int a = std::min(edit_.caret, edit_.anchor);
        const int b = std::max(edit_.caret, edit_.anchor);
        const float xa = textWidth(buf.substr(0, std::size_t(a)), font, theme.smallPx);
        const float xb = textWidth(buf.substr(0, std::size_t(b)), font, theme.smallPx);
        rect(x0 + xa, box.y + theme.px(3), xb - xa, box.h - theme.px(6),
             theme.accentDim, theme.px(2));
    }
    textAt(buf, x0, baseline, theme.text, font, theme.smallPx);
    rect(x0 + caretX, box.y + theme.px(3), theme.px(1.5f), box.h - theme.px(6),
         theme.accent);
}

bool Gui::textInput(std::string_view label, std::string& value, const TextOpts& o)
{
    const Rect r = row();
    const uint64_t id = idFor(label);
    const float split = std::min(r.w * 0.40f, r.w - theme.px(110));
    const Rect box{r.x + split, r.y, r.w - split, r.h};
    textAt(visibleLabel(label), r.x, r.y + r.h - theme.px(7), theme.textDim);
    return textInputAt(id, box.x, box.y, box.w, box.h, value, o);
}

bool Gui::textInputAt(uint64_t id, float x, float y, float w, float h,
                      std::string& value, const TextOpts& o)
{
    const Rect box{x, y, w, h};
    const Response b = edit_.id == id ? Response{}
                                      : behave(box, id, kFireOnPress | kNoCapture);
    bool committed = false;

    if (edit_.id == id) {
        const bool clickAway = in_.lPressed && !hit(box);
        if (editKeys(o) || clickAway) {
            if (edit_.id == id) {          // Esc already cleared it otherwise
                committed = edit_.buf != value;
                value = edit_.buf;
                edit_ = {};
            }
        }
    } else if (b.clicked) {
        edit_ = {id, value, value, 0, 0, 0};
        edit_.caret = int(value.size());
        edit_.anchor = o.selectAllOnFocus ? 0 : edit_.caret;
    }

    drawTextBox(box, edit_.id == id ? edit_.buf : value, edit_.id == id,
                b.hovered, o.hint, FontId::Mono);
    return committed;
}

bool Gui::dragInt(std::string_view lbl, int& v, double step, int lo, int hi)
{
    double d = v;
    // Quantum 1: dragImpl keeps the fractional travel in its own continuous
    // shadow. Rounding here and handing the rounded value back next frame is
    // what used to lose it -- `cells z` at 0.25/px re-rounded to the same int
    // every frame and never moved.
    dragImpl(lbl, d, step, lo, hi, 0, {}, 1.0);
    const int nv = int(std::lround(d));
    if (nv != v) {
        v = nv;
        return true;
    }
    return false;
}

bool Gui::choice(std::string_view lbl, int& idx, std::span<const char* const> items)
{
    if (items.empty()) return false;
    const Rect r = row();
    const float split = lbl.empty() ? 0 : std::min(r.w * 0.4f, r.w - theme.px(120));
    if (!lbl.empty()) textAt(lbl, r.x, r.y + r.h - theme.px(7), theme.textDim);
    const float x0 = r.x + split, w = r.w - split;
    const float seg = w / float(items.size());
    bool changed = false;
    for (size_t i = 0; i < items.size(); ++i) {
        Rect s{x0 + seg * float(i), r.y, seg - 1, r.h};
        const Response b = behave(s, idFor(items[i], i + 1),
                                  kFireOnPress | kNoCapture);
        const bool sel = int(i) == idx;
        if (b.clicked && !sel) {
            idx = int(i);
            changed = true;
        }
        rect(s.x, s.y, s.w, s.h,
             sel ? theme.accentDim : (b.hovered ? theme.widgetHot : theme.widget),
             i == 0 ? theme.radius : (i + 1 == items.size() ? theme.radius : 0));
        const float tw = textWidth(items[i], FontId::Sans, theme.smallPx);
        textAt(items[i], s.x + (s.w - tw) * 0.5f, s.y + s.h - theme.px(7),
               sel ? theme.text : theme.textDim, FontId::Sans, theme.smallPx);
    }
    return changed;
}

bool Gui::selectable(std::string_view s, bool selected)
{
    const Rect r = row(theme.rowH - 2);
    const Response b = behave(r, idFor(s), kFireOnPress | kNoCapture);
    rect(r.x, r.y, r.w, r.h,
         selected ? theme.accentDim : (b.hovered ? theme.widgetHot : theme.widget),
         theme.radius);
    textAt(visibleLabel(s), r.x + theme.px(8), r.y + r.h - theme.px(6),
           selected ? theme.text : theme.textDim);
    return b.clicked;
}

void Gui::swatch(Rgba c, std::string_view s)
{
    const Rect r = row(16);
    rect(r.x, r.y + theme.px(2), theme.px(12), theme.px(12), c, theme.px(2));
    textAt(s, r.x + theme.px(20), r.y + theme.px(12), theme.textDim, FontId::Sans, theme.smallPx);
}

void Gui::bar(std::string_view caption, double frac, Rgba c)
{
    const Rect r = row(theme.rowH - 2);
    textAt(caption, r.x, r.y + r.h - theme.px(8), theme.textDim, FontId::Sans, theme.smallPx);
    const float bh = theme.px(4);
    const float by = r.y + r.h - theme.px(3);
    rect(r.x, by, r.w, bh, theme.widget, 2);
    rect(r.x, by, r.w * float(std::clamp(frac, 0.0, 1.0)), bh, c, 2);
}

} // namespace ui
