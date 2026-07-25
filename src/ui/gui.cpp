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
    if (!in_.lDown) {
        active_ = 0;
        dragAccum_ = 0;
    }
}

void Gui::end()
{
    pushBatch();
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

uint64_t Gui::idFor(std::string_view s) const
{
    return fnv(s, fnv(panelName_));
}

// --- panels -----------------------------------------------------------------

float Gui::panelHeight(std::string_view title) const
{
    auto it = panelH_.find(fnv(title));
    return it == panelH_.end() ? 0 : it->second;
}

void Gui::beginPanel(std::string_view title, float x, float y, float w, float maxH)
{
    panelName_ = std::string(title);
    const uint64_t id = fnv(title);
    const float headH = theme.rowH + 4;
    float contentH = panelH_.count(id) ? panelH_[id] : 0;
    float h = headH + contentH + theme.pad;
    if (maxH > 0) h = std::min(h, maxH);

    rect(x - 1, y - 1, w + 2, h + 2, theme.border, theme.radius + 1);
    rect(x, y, w, h, theme.panel, theme.radius);
    rect(x, y, w, headH, theme.panelHead, theme.radius);
    rect(x, y + headH - theme.radius, w, theme.radius, theme.panelHead);
    textAt(title, x + theme.pad, y + headH - 8, theme.text);
    // A hairline in the accent colour under the title: with six panels open,
    // this is what separates them at a glance.
    rect(x, y + headH - 1, w, 1, theme.accentDim);

    if (in_.mx >= x && in_.mx <= x + w && in_.my >= y && in_.my <= y + h)
        hoverPanel_ = true;

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
        if (hoverPanel_ && in_.wheel != 0) scroll -= in_.wheel * 40.0f;
        scroll = std::clamp(scroll, 0.0f, contentH - (h - headH - theme.pad));
        // Scrollbar: a 3 px track on the right edge.
        const float trackH = h - headH - theme.pad;
        const float thumb = std::max(20.0f, trackH * trackH / contentH);
        const float t = scroll / (contentH - trackH);
        rect(x + w - 4, y + headH + t * (trackH - thumb), 3, thumb, theme.accentDim,
             1.5f);
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
    textAt(s, r.x, r.y + r.h - 5, col);
}

void Gui::field(std::string_view key, std::string_view value, Rgba col)
{
    const Rect r = row(theme.rowH - 4);
    textAt(key, r.x, r.y + r.h - 5, theme.textDim, FontId::Sans, theme.smallPx);
    const float w = textWidth(value, FontId::Mono, theme.smallPx);
    textAt(value, r.x + r.w - w, r.y + r.h - 5, col, FontId::Mono, theme.smallPx);
}

void Gui::separator()
{
    const Rect r = row(7);
    rect(r.x, r.y + 3, r.w, 1, theme.border);
}

void Gui::spacing(float h)
{
    cur_.y += h;
}

bool Gui::sectionHeader(std::string_view title, bool& open)
{
    const Rect r = row(theme.rowH);
    const uint64_t id = idFor(title);
    const bool over = hit(r);
    if (over) hot_ = id;
    if (over && in_.lPressed) open = !open;
    rect(r.x, r.y, r.w, r.h, over ? theme.widgetHot : theme.widget, theme.radius);
    const char* mark = open ? "\xE2\x96\xBE" : "\xE2\x96\xB8";   // down / right
    textAt(mark, r.x + 6, r.y + r.h - 6, theme.accent);
    textAt(title, r.x + 22, r.y + r.h - 6, theme.text);
    return open;
}

bool Gui::button(std::string_view s, float w, Rgba tint)
{
    const float bw = w > 0 ? w : cur_.w;
    Rect r = row();
    r.w = bw;
    const uint64_t id = idFor(s);
    const bool over = hit(r);
    if (over) hot_ = id;
    if (over && in_.lPressed) active_ = id;
    const bool fired = over && in_.lReleased && active_ == id;

    Rgba base = tint.a ? tint : theme.widget;
    if (active_ == id && over) base = theme.widgetActive;
    else if (over) base = mix(base, theme.widgetHot, 0.7f);
    rect(r.x, r.y, r.w, r.h, base, theme.radius);
    const float tw = textWidth(s);
    textAt(s, r.x + (r.w - tw) * 0.5f, r.y + r.h - 7, theme.text);
    return fired;
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
        const uint64_t id = idFor(labels[i]);
        const bool over = hit(s);
        if (over) hot_ = id;
        if (over && in_.lPressed) active_ = id;
        if (over && in_.lReleased && active_ == id) clicked = int(i);
        rect(s.x, s.y, s.w, s.h,
             (active_ == id && over) ? theme.widgetActive
             : over                  ? theme.widgetHot
                                     : theme.widget,
             theme.radius);
        const float tw = textWidth(labels[i], FontId::Sans, theme.smallPx);
        textAt(labels[i], s.x + (s.w - tw) * 0.5f, s.y + s.h - 7, theme.text,
               FontId::Sans, theme.smallPx);
    }
    return clicked;
}

bool Gui::checkbox(std::string_view s, bool& v)
{
    const Rect r = row();
    const uint64_t id = idFor(s);
    const bool over = hit(r);
    if (over) hot_ = id;
    bool changed = false;
    if (over && in_.lPressed) {
        v = !v;
        changed = true;
    }
    const float box = 14;
    const float by = r.y + (r.h - box) * 0.5f;
    rect(r.x, by, box, box, v ? theme.accent : (over ? theme.widgetHot : theme.widget),
         3);
    if (v) {
        // Check mark as two thick strokes; at 14 px a glyph would be mushy.
        rect(r.x + 3.5f, by + 6.5f, 4, 2, theme.panel);
        rect(r.x + 5.5f, by + 8.5f, 2, 2, theme.panel);
        rect(r.x + 6.5f, by + 4.5f, 5, 2, theme.panel);
    }
    textAt(s, r.x + box + 8, r.y + r.h - 7, theme.text);
    return changed;
}

bool Gui::dragDouble(std::string_view lbl, double& v, double step, double lo,
                     double hi, int decimals)
{
    const Rect r = row();
    const uint64_t id = idFor(lbl);
    const bool over = hit(r);
    if (over) hot_ = id;

    const float split = std::min(r.w * 0.52f, r.w - 70.0f);
    Rect box{r.x + split, r.y, r.w - split, r.h};

    bool changed = false;
    if (editing_ == id) {
        for (char c : in_.text)
            if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == 'e' || c == '+')
                editBuf_ += c;
        if (in_.key(SDLK_BACKSPACE) && !editBuf_.empty()) editBuf_.pop_back();
        if (in_.key(SDLK_RETURN) || in_.key(SDLK_KP_ENTER) ||
            (in_.lPressed && !hit(box))) {
            try {
                v = std::clamp(std::stod(editBuf_), lo, hi);
                changed = true;
            } catch (...) {
            }
            editing_ = 0;
        } else if (in_.key(SDLK_ESCAPE)) {
            editing_ = 0;
        }
    } else {
        if (over && in_.doubleClick) {
            editing_ = id;
            editBuf_ = format(v, decimals);
            active_ = 0;
        } else if (over && in_.lPressed) {
            active_ = id;
            dragAccum_ = 0;
        }
        if (active_ == id && in_.dx != 0) {
            // Fine drag with shift; the resolution fields need both a coarse
            // sweep and a one-cell nudge.
            const double s = step * (in_.shift ? 0.1 : 1.0);
            dragAccum_ += double(in_.dx) * s;
            const double nv = std::clamp(v + dragAccum_, lo, hi);
            if (nv != v) {
                v = nv;
                dragAccum_ = 0;
                changed = true;
            }
        }
    }

    textAt(lbl, r.x, r.y + r.h - 7, theme.textDim);
    const bool edit = editing_ == id;
    rect(box.x, box.y, box.w, box.h,
         edit ? theme.widgetActive : (over ? theme.widgetHot : theme.widget),
         theme.radius);
    if (edit) rectOutline(box.x, box.y, box.w, box.h, theme.accent, 1);
    const std::string shown = edit ? editBuf_ + "_" : format(v, decimals);
    const float tw = textWidth(shown, FontId::Mono, theme.smallPx);
    textAt(shown, box.x + box.w - tw - 6, box.y + box.h - 7, theme.text, FontId::Mono,
           theme.smallPx);
    return changed;
}

bool Gui::dragInt(std::string_view lbl, int& v, double step, int lo, int hi)
{
    double d = v;
    const bool c = dragDouble(lbl, d, step, lo, hi, 0);
    const int nv = int(std::lround(d));
    if (nv != v) {
        v = nv;
        return true;
    }
    return c && false;
}

bool Gui::choice(std::string_view lbl, int& idx, std::span<const char* const> items)
{
    if (items.empty()) return false;
    const Rect r = row();
    const float split = lbl.empty() ? 0 : std::min(r.w * 0.4f, r.w - 120.0f);
    if (!lbl.empty()) textAt(lbl, r.x, r.y + r.h - 7, theme.textDim);
    const float x0 = r.x + split, w = r.w - split;
    const float seg = w / float(items.size());
    bool changed = false;
    for (size_t i = 0; i < items.size(); ++i) {
        Rect s{x0 + seg * float(i), r.y, seg - 1, r.h};
        const bool over = hit(s);
        const bool sel = int(i) == idx;
        if (over && in_.lPressed && !sel) {
            idx = int(i);
            changed = true;
        }
        rect(s.x, s.y, s.w, s.h,
             sel ? theme.accentDim : (over ? theme.widgetHot : theme.widget),
             i == 0 ? theme.radius : (i + 1 == items.size() ? theme.radius : 0));
        const float tw = textWidth(items[i], FontId::Sans, theme.smallPx);
        textAt(items[i], s.x + (s.w - tw) * 0.5f, s.y + s.h - 7,
               sel ? theme.text : theme.textDim, FontId::Sans, theme.smallPx);
    }
    return changed;
}

bool Gui::selectable(std::string_view s, bool selected)
{
    const Rect r = row(theme.rowH - 2);
    const uint64_t id = idFor(s);
    const bool over = hit(r);
    if (over) hot_ = id;
    rect(r.x, r.y, r.w, r.h,
         selected ? theme.accentDim : (over ? theme.widgetHot : theme.widget),
         theme.radius);
    textAt(s, r.x + 8, r.y + r.h - 6, selected ? theme.text : theme.textDim);
    return over && in_.lPressed;
}

void Gui::swatch(Rgba c, std::string_view s)
{
    const Rect r = row(16);
    rect(r.x, r.y + 2, 12, 12, c, 2);
    textAt(s, r.x + 20, r.y + 12, theme.textDim, FontId::Sans, theme.smallPx);
}

void Gui::bar(std::string_view caption, double frac, Rgba c)
{
    const Rect r = row(theme.rowH - 2);
    textAt(caption, r.x, r.y + r.h - 8, theme.textDim, FontId::Sans, theme.smallPx);
    const float bh = 4;
    const float by = r.y + r.h - 3;
    rect(r.x, by, r.w, bh, theme.widget, 2);
    rect(r.x, by, r.w * float(std::clamp(frac, 0.0, 1.0)), bh, c, 2);
}

} // namespace ui
