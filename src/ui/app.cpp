#include "ui/app.hpp"

#include <filesystem>
#include <cstring>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

namespace ui {
namespace {

std::string num(double v, int dec = 4)
{
    char b[80];
    std::snprintf(b, sizeof b, "%.*g", dec, v);
    return b;
}

std::string bigInt(std::size_t v)
{
    std::string s = std::to_string(v);
    for (int i = int(s.size()) - 3; i > 0; i -= 3) s.insert(size_t(i), " ");
    return s;
}

const char* const kPhysicsNames[] = {"incomp", "comp"};
// `dns flow` and `dns conv` are two cases, not two settings of one: the panels
// below gate wholesale on this rather than greying knobs the other case has no
// meaning for.
const char* const kSubNames[] = {"flow", "convection"};
// The convection model. Boussinesq is FIRST and the default: it is the M -> 0
// limit taken analytically, so it needs no Mach number, carries no
// stratification, and runs the same case in a fortieth of the steps.
const char* const kConvModelNames[] = {"boussinesq", "compressible"};
// The lateral faces. Periodic is FIRST and the default: it is the classical
// Rayleigh-Benard setup, an unbounded layer sampled over one box. Closed is a
// sealed cavity -- a different experiment, not a detail (see the panel).
const char* const kConvSidesNames[] = {"periodic", "closed"};
const char* const kViscNames[] = {"constant", "sutherland"};
const char* const kWallNames[] = {"adiabatic", "isothermal"};
const char* const kSgsNames[] = {"smag", "sigma"};
const char* const kAxisNames[] = {"x", "y", "z"};

// A length in metres, printed at a readable magnitude. A 4 mm cell and a 12 m
// tunnel both show up in this tool, and "0.004" next to "12" reads as noise.
std::string metres(double m)
{
    char b[80];
    const double a = std::abs(m);
    if (a > 0 && a < 1e-3)
        std::snprintf(b, sizeof b, "%.3g um", m * 1e6);
    else if (a > 0 && a < 1.0)
        std::snprintf(b, sizeof b, "%.4g mm", m * 1e3);
    else if (a < 1e3)
        std::snprintf(b, sizeof b, "%.4g m", m);
    else
        std::snprintf(b, sizeof b, "%.4g km", m * 1e-3);
    return b;
}

std::string seconds(double s)
{
    char b[80];
    const double a = std::abs(s);
    if (a > 0 && a < 1e-3)
        std::snprintf(b, sizeof b, "%.3g us", s * 1e6);
    else if (a > 0 && a < 1.0)
        std::snprintf(b, sizeof b, "%.4g ms", s * 1e3);
    else
        std::snprintf(b, sizeof b, "%.4g s", s);
    return b;
}

} // namespace

// --- layout -----------------------------------------------------------------

void Layout::compute(float w, float h, bool dockOpen)
{
    // Clamp the columns before deriving the viewport: on a narrow window the
    // two columns can ask for more than the window has, and a negative
    // viewport width would reach Vulkan as a validation error rather than as a
    // squashed panel.
    const float maxCol = std::max(px(120), (w - px(240)) * 0.5f);
    leftW = std::clamp(leftW, px(200), maxCol);
    rightW = std::clamp(rightW, px(200), maxCol);

    dockTotalH = dockOpen ? dockHeadH + std::max(px(80), dockH) : dockHeadH;
    const float bodyTop = menuH;
    const float bodyBot = h - statusH - dockTotalH;

    dockY = bodyBot;
    vpX = leftW + split;
    vpY = bodyTop;
    vpW = std::max(1.0f, w - leftW - rightW - 2 * split);
    vpH = std::max(1.0f, bodyBot - bodyTop);

    // Split the left column: Case scrolls, Geometry is pinned to the bottom.
    geomH = std::clamp(geomH, px(140), std::max(px(140), vpH - px(140)));
    caseH = std::max(px(60), vpH - geomH - split);
    geomY = bodyTop + caseH + split;
}

// --- camera -----------------------------------------------------------------

Vec3 Camera::forward() const
{
    const float cp = std::cos(pitch);
    return normalize({-cp * std::cos(yaw), -cp * std::sin(yaw), -std::sin(pitch)});
}

Vec3 Camera::eye() const { return target - forward() * dist; }

Vec3 Camera::worldUp() const
{
    const Vec3 f = forward();
    return std::abs(f.z) > 0.999f ? Vec3{0, 1, 0} : Vec3{0, 0, 1};
}

void Camera::basis(Vec3& right, Vec3& up) const
{
    const Vec3 f = forward();
    right = normalize(cross(f, worldUp()));
    up = cross(right, f);
}

void Camera::setStandardView(int which)
{
    switch (which) {
    case 0: yaw = 0.0f; pitch = 0.0f; break;              // looking along -x
    case 1: yaw = 1.5707963f; pitch = 0.0f; break;        // side
    case 2: yaw = 0.0f; pitch = 1.5707963f; break;        // top
    default: yaw = -0.9f; pitch = 0.5f; break;            // iso (the default)
    }
}

Mat4 Camera::viewProj(float aspect) const
{
    const Vec3 e = eye();
    const Mat4 v = lookAt(e, target, worldUp());
    // Near/far bracket the target: a viewer that clips the domain box when you
    // zoom to a cut cell is unusable, so they track dist rather than being
    // fixed constants.
    const float zn = std::max(1e-4f, dist * 0.002f), zf = dist * 40.0f + 100.0f;
    const Mat4 p = ortho ? orthographic(dist * 0.5f * aspect, dist * 0.5f, -zf, zf)
                         : perspective(fovY, aspect, zn, zf);
    return p * v;
}

float Camera::worldPerPixel(float h) const
{
    return ortho ? dist / h : 2.0f * dist * std::tan(fovY * 0.5f) / h;
}

float Camera::worldPerPixelAt(const Vec3& p, float vpH) const
{
    if (ortho) return dist / vpH;   // depth-independent by construction
    // VIEW depth, not range: a point off to the side of the frustum is not
    // further away in the sense that matters for a screen-space size.
    const float zv = std::abs(dot(p - eye(), forward()));
    return 2.0f * zv * std::tan(fovY * 0.5f) / vpH;
}

void Camera::ray(float px, float py, float w, float h, Vec3& origin, Vec3& dir) const
{
    const float ndcX = 2.0f * px / w - 1.0f;
    const float ndcY = 2.0f * py / h - 1.0f;
    Vec3 right, up;
    basis(right, up);
    const Vec3 f = forward();
    if (ortho) {
        const float halfH = dist * 0.5f, halfW = halfH * (w / h);
        origin = eye() + right * (ndcX * halfW) + up * (-ndcY * halfH);
        dir = f;
    } else {
        const float t = 1.0f / std::tan(fovY * 0.5f);
        origin = eye();
        dir = normalize(f + right * (ndcX * (w / h) / t) + up * (-ndcY / t));
    }
}

// --- lifecycle --------------------------------------------------------------

std::expected<std::unique_ptr<App>, std::string> App::create()
{
    if (!SDL_Init(SDL_INIT_VIDEO))
        return std::unexpected(std::string("SDL_Init: ") + SDL_GetError());

    auto app = std::unique_ptr<App>(new App());
    app->window_ = SDL_CreateWindow("dns-ui  |  obstacle editor & grid viewer",
                                    int(app->w_), int(app->h_),
                                    SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                                        SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!app->window_)
        return std::unexpected(std::string("SDL_CreateWindow: ") + SDL_GetError());

    auto vk = Vk::create(app->window_);
    if (!vk) return std::unexpected(vk.error());
    app->vk_ = std::move(*vk);

    auto ren = Renderer::create(*app->vk_);
    if (!ren) return std::unexpected(ren.error());
    app->ren_ = std::move(*ren);

    auto text = TextEngine::create(*app->vk_);
    if (!text) return std::unexpected(text.error());
    app->text_ = std::move(*text);
    if (auto r = app->ren_->setGlyphAtlas(app->text_->atlas()); !r)
        return std::unexpected(r.error());

    app->gui_ = std::make_unique<Gui>(*app->text_);
    // One scale for the whole UI: Theme carries the widget metrics and the
    // font sizes the glyph atlas rasterizes at, Layout carries the chrome and
    // the column widths. Scaling only one of them clips text against rows that
    // did not grow. DNS_UI_SCALE overrides it.
    float uiScale = 1.6f;
    if (const char* e = SDL_getenv("DNS_UI_SCALE")) {
        if (const auto v = parseNumber(e); v && *v >= 0.5 && *v <= 4.0)
            uiScale = float(*v);
    }
    app->gui_->clipboard.get = [] {
        char* t = SDL_GetClipboardText();
        std::string s = t ? t : "";
        SDL_free(t);
        return s;
    };
    app->gui_->clipboard.set = [](std::string_view v) {
        SDL_SetClipboardText(std::string(v).c_str());
    };
    app->gui_->theme.scale(uiScale);
    app->lay_.scale(uiScale);
    SDL_StartTextInput(app->window_);
    return app;
}

App::~App()
{
    if (vk_) vk_->waitIdle();
    if (ren_) destroyView(*ren_, meshes_);
    gui_.reset();
    text_.reset();
    ren_.reset();
    vk_.reset();
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

// --- input ------------------------------------------------------------------

void App::handleEvents(bool& quit)
{
    in_.clearFrame();
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_EVENT_QUIT:
            quit = true;
            break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_RESIZED:
            vk_->invalidateSwapchain();
            break;
        case SDL_EVENT_MOUSE_MOTION: {
            // SDL reports window coordinates; the framebuffer may be larger on
            // a HiDPI display, and the UI is laid out in framebuffer pixels.
            int ww = 1, wh = 1;
            SDL_GetWindowSize(window_, &ww, &wh);
            const float sx = w_ / float(std::max(1, ww));
            const float sy = h_ / float(std::max(1, wh));
            in_.mx = e.motion.x * sx;
            in_.my = e.motion.y * sy;
            in_.dx += e.motion.xrel * sx;
            in_.dy += e.motion.yrel * sy;
            break;
        }
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (e.button.button == SDL_BUTTON_LEFT) {
                in_.lDown = true;
                in_.lPressed = true;
                if (e.button.clicks >= 2) in_.doubleClick = true;
            } else if (e.button.button == SDL_BUTTON_RIGHT) {
                in_.rDown = true;
                in_.rPressed = true;
            } else if (e.button.button == SDL_BUTTON_MIDDLE) {
                in_.mDown = true;
            }
            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (e.button.button == SDL_BUTTON_LEFT) {
                in_.lDown = false;
                in_.lReleased = true;
            } else if (e.button.button == SDL_BUTTON_RIGHT) {
                in_.rDown = false;
            } else if (e.button.button == SDL_BUTTON_MIDDLE) {
                in_.mDown = false;
            }
            break;
        case SDL_EVENT_DROP_FILE: {
            // The cheapest possible file affordance, and the only one that
            // needs no portal, no dialog and no extra thread.
            const std::string path = e.drop.data ? e.drop.data : "";
            if (path.size() > 5 && path.substr(path.size() - 5) == ".toml") {
                const std::string err = doc_.openScene(path);
                status_ = err.empty() ? "opened " + path : err;
                if (err.empty()) { scenePath_ = path; firstBuild_ = true; }
            } else if (!path.empty()) {
                if (SceneItem* it = doc_.find(doc_.primary());
                    it && it->obj.kind == dns::SceneObjectKind::Mesh) {
                    doc_.beginEdit("set mesh path");
                    it->obj.meshPath = path;
                    doc_.touch();
                    doc_.endEdit();
                    status_ = "mesh: " + path;
                } else {
                    doc_.st.meshPath = path;
                    doc_.st.bodyKind = int(dns::BodyKind::Mesh);
                    doc_.touch();
                    status_ = "mesh body: " + path;
                }
            }
            break;
        }
        case SDL_EVENT_MOUSE_WHEEL:
            in_.wheel += e.wheel.y;
            break;
        case SDL_EVENT_TEXT_INPUT:
            in_.text += e.text.text;
            break;
        case SDL_EVENT_KEY_DOWN:
            in_.keys.push_back(e.key.key);
            in_.shift = (e.key.mod & SDL_KMOD_SHIFT) != 0;
            in_.ctrl = (e.key.mod & SDL_KMOD_CTRL) != 0;
            break;
        case SDL_EVENT_KEY_UP:
            in_.shift = (e.key.mod & SDL_KMOD_SHIFT) != 0;
            in_.ctrl = (e.key.mod & SDL_KMOD_CTRL) != 0;
            break;
        default:
            break;
        }
    }
    const VkExtent2D ext = vk_->extent();
    if (ext.width && ext.height) {
        w_ = float(ext.width);
        h_ = float(ext.height);
    }
}

void App::cameraInput()
{
    setProjectionMode();
    const Build& b = doc_.current();
    const bool flat = b.ok && b.cas().grid.nz == 1;

    if (in_.wheel != 0) {
        cam_.dist *= std::pow(0.88f, in_.wheel);
        cam_.dist = std::clamp(cam_.dist, 1e-4f, 1e6f);
    }

    Vec3 right, up;
    cam_.basis(right, up);
    const float wpp = cam_.worldPerPixel(lay_.vpH);

    if (in_.rDown || in_.mDown || (in_.lDown && in_.shift)) {
        cam_.target = cam_.target - right * (in_.dx * wpp) + up * (in_.dy * wpp);
    } else if (in_.lDown && !dragObject_) {
        if (flat) {
            // A 2D case is a picture, not a scene: orbiting it would only ever
            // hide the plane you are trying to read. Left-drag pans instead.
            cam_.target = cam_.target - right * (in_.dx * wpp) + up * (in_.dy * wpp);
        } else {
            cam_.yaw -= in_.dx * 0.006f;
            cam_.pitch = std::clamp(cam_.pitch + in_.dy * 0.006f, -1.5f, 1.5f);
        }
    }
}

void App::frameCamera(bool onBody)
{
    const Build& b = doc_.current();
    if (!b.ok) return;
    setProjectionMode();
    Vec3 c;
    float r = 1;
    domainBounds(b, c, r);
    // With something selected, F frames THAT object rather than the whole
    // union of bodies -- framing the domain because you selected one cylinder
    // is not "focus".
    Vec3 selC;
    if (!onBody && doc_.sceneMode() && gizmoOrigin(selC)) {
        const SceneItem* it = doc_.find(doc_.primary());
        const double D = b.cas().placement.D;
        cam_.target = selC;
        cam_.dist = std::max(1e-4f, float((it ? it->obj.d : 1.0) * D) *
                                        (cam_.ortho ? 4.0f : 3.2f));
        return;
    }
    if (onBody && b.geo.hasBbox) {
        const auto& bb = b.geo.bbox;
        c = {float(0.5 * (bb[0] + bb[3])), float(0.5 * (bb[1] + bb[4])),
             float(b.cas().grid.nz == 1 ? 0.0 : 0.5 * (bb[2] + bb[5]))};
        r = 0.5f * float(std::max({bb[3] - bb[0], bb[4] - bb[1],
                                   b.cas().grid.nz == 1 ? 0.0 : bb[5] - bb[2]}));
        r = std::max(r, float(b.cas().placement.D) * 0.5f);
        r *= 3.0f;
    }
    cam_.target = c;
    cam_.dist = std::max(1e-4f, r * (cam_.ortho ? 2.4f : 2.2f));
}

// nz == 1 is a plane, and the only honest way to look at a plane is straight
// at it: pitch is pinned to +90 deg so forward is exactly -z, right is +x and
// up is +y. An oblique view of a 2D cut foreshortens the very apertures the
// tool exists to show.
void App::setProjectionMode()
{
    const Build& b = doc_.current();
    const bool flat = b.ok && b.cas().grid.nz == 1;
    // 2D is ALWAYS orthographic; in 3D the user's preference decides. Writing
    // `ortho` directly here is what used to clobber a toggle every frame.
    cam_.ortho = flat || cam_.orthoPref;
    if (flat) {
        cam_.yaw = 0.0f;
        cam_.pitch = 1.5707963f;
    }
}

// Ray/AABB against every object box, nearest wins. Scene mode only: with a
// single --body there is nothing to select.
// Analytic ray tests against the LIVE objects. The old version tested the
// published AABBs, and for a cylinder, cube or lens that box spans the entire
// domain depth -- so a click eight diameters away selected it, and an object
// genuinely behind another could never be reached.
bool App::pick()
{
    const Build& b = doc_.current();
    if (!doc_.sceneMode() || !b.ok) return false;
    const Ray r = viewportRay();
    const auto& p = b.cas().placement;
    const auto& g = b.cas().grid;
    const auto& boxes = b.geo.objectBoxes;

    struct Cand { float t; int idx; };
    std::vector<Cand> hits;
    for (std::size_t i = 0; i < doc_.st.items.size(); ++i) {
        const auto& o = doc_.st.items[i].obj;
        // A mesh's true shape needs the loaded file, so its published box is
        // the fallback -- and that box is tight, unlike a cylinder's.
        const std::array<double, 6>* mb = nullptr;
        if (o.kind == dns::SceneObjectKind::Mesh)
            for (const auto& bx : boxes)
                if (bx.first == o.label) mb = &bx.second;
        float t = 0;
        if (rayObject(o, p, g, mb, r, t)) hits.push_back({t, int(i)});
    }
    if (hits.empty()) return false;   // empty click keeps the selection
    std::sort(hits.begin(), hits.end(),
              [](const Cand& a, const Cand& c) { return a.t < c.t; });

    std::vector<int> order;
    for (const auto& h : hits) order.push_back(h.idx);
    const bool sameSpot = std::abs(in_.mx - lastClickX_) < gui_->theme.px(4) &&
                          std::abs(in_.my - lastClickY_) < gui_->theme.px(4);
    lastClickX_ = in_.mx;
    lastClickY_ = in_.my;
    const int cur = doc_.indexOf(doc_.primary());
    const int idx = cycleSelection(order, cur, sameSpot);
    if (idx < 0) return false;
    doc_.select(doc_.idAt(idx), in_.ctrl ? Doc::Sel::Toggle : Doc::Sel::Replace);
    return true;
}

Ray App::viewportRay() const
{
    Vec3 o, d;
    cam_.ray(in_.mx - lay_.vpX, in_.my - lay_.vpY, lay_.vpW, lay_.vpH, o, d);
    return {o, d};
}

// The gizmo sits at the primary selection's centre.
bool App::gizmoOrigin(Vec3& out) const
{
    const Build& b = doc_.current();
    if (!b.ok || !doc_.sceneMode()) return false;
    const SceneItem* it = doc_.find(doc_.primary());
    if (!it) return false;
    const auto& p = b.cas().placement;
    const bool flat = b.cas().grid.nz == 1;
    out = {float(p.cx + it->obj.x * p.D), float(p.cy + it->obj.y * p.D),
           float(flat ? 0.0 : p.cz + it->obj.z * p.D)};
    return true;
}

void App::updateGizmo()
{
    gizVerts_.clear();
    Vec3 origin;
    if (!gizmoOrigin(origin)) {
        gizHot_ = {};
        (void)ren_->upload(gizMesh_, std::span<const Vtx3>(gizVerts_));
        return;
    }
    const Build& b = doc_.current();
    const SceneItem* it = doc_.find(doc_.primary());

    // z is only read by the scene builder for spheres and meshes: a cylinder,
    // cube or lens spans the full box depth by construction, so a z handle for
    // them would move a number that changes nothing.
    const bool zMeans =
        b.ok && b.cas().grid.nz > 1 && it &&
        (it->obj.kind == dns::SceneObjectKind::Sphere ||
         it->obj.kind == dns::SceneObjectKind::Mesh);
    const bool enabled[3] = {true, true, zMeans};

    Vec3 right, up;
    cam_.basis(right, up);
    // Sized at the OBJECT's depth, not the orbit target's: under perspective a
    // gizmo sized at the target is the wrong size for anything nearer or
    // further, and its handles then do not line up with the cursor.
    const float wpp = cam_.worldPerPixelAt(origin, lay_.vpH);
    gizFrame_ = makeGizmoFrame(origin, cam_.eye(), cam_.forward(), right, up, wpp,
                               gui_->theme.s, enabled);

    if (!gizDrag_.active) {
        const Mat4 mvp = cam_.viewProj(lay_.vpW / std::max(1.0f, lay_.vpH));
        gizHot_ = lay_.inViewport(in_.mx, in_.my)
                      ? gizmoPick(gizFrame_, gizStyle_, mvp, lay_.vpW, lay_.vpH,
                                  Vec2{in_.mx - lay_.vpX, in_.my - lay_.vpY})
                      : GizmoHit{};
    }
    gizmoBuild(gizVerts_, gizFrame_, gizStyle_, gizHot_.part, gizDrag_.part);
    (void)ren_->upload(gizMesh_, std::span<const Vtx3>(gizVerts_));
}

// --- UI ---------------------------------------------------------------------

void App::buildUi()
{
    Gui& g = *gui_;
    g.begin(in_, w_, h_);
    lay_.compute(w_, h_, dockOpen_);
    const Build& b = doc_.current();
    const Theme& th = g.theme;

    // --- menu strip ---------------------------------------------------------
    // Informational, plus the one toggle that changes the FRAME rather than
    // the document. Drawn with raw primitives and hit-tested against in_
    // directly: strip widgets sit outside any panel, so the layout cursor the
    // Gui widgets consume does not exist here.
    g.strip(0, 0, w_, lay_.menuH, th.panelHead);
    g.rect(0, lay_.menuH - th.px(1), w_, th.px(1), th.border);
    g.textAt("dns-ui", th.px(12), lay_.menuH - th.px(9), th.accent);

    // Buttons are laid out by measuring, not by hardcoded x: at 1.6x the old
    // fixed 62 px slots would clip their own labels.
    float mx = g.textWidth("dns-ui", FontId::Sans, th.fontPx) + th.px(24);
    auto stripButton = [&](std::string_view text, bool on) {
        const float w = g.textWidth(text, FontId::Sans, th.smallPx) + th.px(20);
        const float x = mx;
        mx += w + th.px(6);
        const bool over = in_.mx >= x && in_.mx <= x + w && in_.my >= th.px(3) &&
                          in_.my <= lay_.menuH - th.px(4);
        g.rect(x, th.px(3), w, lay_.menuH - th.px(7),
               on ? th.accentDim : (over ? th.widgetHot : th.widget), th.radius);
        const float tw = g.textWidth(text, FontId::Sans, th.smallPx);
        g.textAt(text, x + (w - tw) * 0.5f, lay_.menuH - th.px(10), th.text,
                 FontId::Sans, th.smallPx);
        return over && in_.lPressed;
    };
    if (stripButton("frame", false)) frameCamera(false);
    if (stripButton("body", false)) frameCamera(true);
    {
        const Build& fb = doc_.current();
        const bool flat = fb.ok && fb.cas().grid.nz == 1;
        // Forced on and inert in 2D: an oblique view of a 2D cut foreshortens
        // the very apertures the tool exists to show.
        if (stripButton("ortho", cam_.ortho) && !flat)
            cam_.orthoPref = !cam_.orthoPref;
        if (stripButton("view", false)) {
            standardView_ = (standardView_ + 1) % 4;
            if (!flat) cam_.setStandardView(standardView_);
        }
    }
    if (stripButton("log", dockOpen_)) dockOpen_ = !dockOpen_;
    if (stripButton("undo", false)) doc_.undo();
    if (stripButton("redo", false)) doc_.redo();
    // The one control that leaves the editor and starts real work -- and it
    // has three states, not two. `run_.running()` only knows about a solver
    // THIS process spawned; while attached to someone else's live run, offering
    // RUN would start a second solver competing for the same GPU and the same
    // segment name.
    const bool ourRun = run_.running();
    const bool foreignRun = !ourRun && live_.attached() && live_.running();
    if (ourRun || foreignRun) {
        if (stripButton("stop", true)) {
            if (ourRun) {
                run_.stop();
                status_ = "run stopped";
            } else {
                // A run we did not spawn: signal the publisher directly. The
                // segment carries its pid for exactly this, and SIGTERM is the
                // same clean shutdown a Ctrl-C in its terminal would give.
                const int pid = live_.publisherPid();
                const std::string err = live_.stopPublisher();
                status_ = err.empty()
                              ? "stopped the live run (pid " + std::to_string(pid) + ")"
                              : err;
            }
        }
    } else if (stripButton("RUN", false)) {
        startRun();
    }

    // The label lives here now: it is the single most important word on the
    // screen and it belongs in chrome, not in a panel that can scroll away.
    if (b.ok) {
        const std::string msg =
            b.label + (doc_.building() ? "   |   rebuilding cut..."
                                       : "   |   cut in " + num(b.buildSeconds, 2) + " s");
        const float tw = g.textWidth(msg, FontId::Sans, th.smallPx);
        g.textAt(msg, w_ - tw - th.px(12), lay_.menuH - th.px(9),
                 b.cas().lesLabel ? th.warn : th.textDim, FontId::Sans, th.smallPx);
    }

    // --- region splitters ---------------------------------------------------
    const float minCol = lay_.px(200);
    const float maxCol = std::max(lay_.px(120), (w_ - lay_.px(240)) * 0.5f);
    g.splitter("splitL", lay_.leftW, lay_.menuH, lay_.split, lay_.vpH, lay_.leftW,
               true, minCol, maxCol);
    // The right splitter controls an EDGE, not a width: dragging it right must
    // shrink the column, so it drives w_ - rightW and the width is recovered.
    float rightEdge = w_ - lay_.rightW - lay_.split;
    g.splitter("splitR", rightEdge, lay_.menuH, lay_.split, lay_.vpH, rightEdge,
               true, w_ - maxCol, w_ - minCol);
    lay_.rightW = std::clamp(w_ - rightEdge - lay_.split, minCol, maxCol);
    {
        // The dock divider. It exists COLLAPSED too: dragging the closed
        // header's edge upward opens the dock at the height you dragged it to,
        // which is what every attempt to resize a collapsed dock means.
        float dockTop = lay_.dockY;
        const float was = dockTop;
        g.splitter("splitD", 0, dockTop - lay_.split, w_, lay_.split, dockTop,
                   false, lay_.menuH + lay_.px(120),
                   h_ - lay_.statusH - (dockOpen_ ? lay_.px(120) : 0.0f));
        if (!dockOpen_ && dockTop < was - 1.0f) dockOpen_ = true;
        if (dockOpen_)
            lay_.dockH =
                std::max(lay_.px(80), h_ - lay_.statusH - dockTop - lay_.dockHeadH);
    }
    // Splitter between the two left regions.
    {
        float geomTop = lay_.geomY;
        g.splitter("splitG", 0, geomTop - lay_.split, lay_.leftW, lay_.split,
                   geomTop, false, lay_.menuH + lay_.px(60),
                   lay_.dockY - lay_.px(140));
        lay_.geomH = std::max(lay_.px(140), lay_.dockY - geomTop);
    }
    lay_.compute(w_, h_, dockOpen_);

    leftRegion();
    geometryRegion();
    rightRegion();
    dockRegion();
    // On top of the regions: a divider painted before them is a divider the
    // panels paint over, and an invisible handle is not a handle.
    g.flushSplitters();

    // --- status strip -------------------------------------------------------
    // The validator's own message, verbatim, with the exit code the CLI would
    // return. A rejected edit is the one thing that must never be hidden by a
    // collapsed section, so it owns the full width.
    const float sy = h_ - lay_.statusH;
    if (!b.ok && !b.error.empty()) {
        g.strip(0, sy, w_, lay_.statusH, rgb(0x3c1719));
        g.textAt("exit " + std::to_string(b.errorCode) + ":  " + b.error, th.px(12),
                 sy + lay_.statusH - th.px(7), th.bad, FontId::Sans, th.smallPx);
    } else if (b.sceneEmpty) {
        // Usable, but it would NOT run: say both, in the same breath.
        g.strip(0, sy, w_, lay_.statusH, rgb(0x3a2f16));
        g.textAt("scene has no objects: add one with +cyl.  `dns --scene` would "
                 "exit 3 until you do.",
                 th.px(12), sy + lay_.statusH - th.px(7), th.warn, FontId::Sans,
                 th.smallPx);
    } else {
        g.strip(0, sy, w_, lay_.statusH, th.panelHead);
        std::string msg =
            "LMB orbit/pan  ·  RMB pan  ·  wheel zoom  ·  F frame  ·  B body  ·  "
            "G C M A layers  ·  Del remove";
        if (!doc_.notice().empty()) status_ = doc_.notice();
        if (!status_.empty()) msg = status_ + "     " + msg;
        g.textAt(msg, th.px(12), sy + lay_.statusH - th.px(7), th.textDim, FontId::Sans,
                 th.smallPx);
        if (live_.attached()) {
            // Whether frames are arriving is the question a live view raises,
            // so it is answered in the chrome rather than only in a panel.
            const std::string lv =
                "live " + std::to_string(live_.frame()) + " @ t " +
                num(live_.time(), 4) + (live_.running() ? "" : "  (ended)");
            const float tw = g.textWidth(lv, FontId::Sans, th.smallPx);
            g.textAt(lv, w_ * 0.5f - tw * 0.5f, sy + lay_.statusH - th.px(7),
                     live_.running() ? th.good : th.textDim, FontId::Sans,
                     th.smallPx);
        }
        if (b.ok) {
            // The one-line verdict, in whichever terms the case has: a body
            // boundary layer and a Reynolds number, or a thermal boundary
            // layer and a Rayleigh number.
            const std::string r =
                b.cas().sub == dns::Subcommand::Conv
                    ? "thermal BL " + num(b.blCells, 3) +
                          (b.bodyResolved ? " resolved" : " UNDER") + "   ·   " +
                          metres(doc_.st.convH) + " layer   ·   Ra " +
                          num(doc_.st.convRayleigh(), 4)
                    : "BL " + num(b.blCells, 3) +
                          (b.bodyResolved ? " resolved" : " UNDER") + "   ·   " +
                          metres(doc_.st.phys.dBody) + " body   ·   Re " +
                          num(doc_.st.re(), 4);
            const float tw = g.textWidth(r, FontId::Sans, th.smallPx);
            g.textAt(r, w_ - tw - th.px(12), sy + lay_.statusH - th.px(7),
                     b.bodyResolved ? th.good : th.warn, FontId::Sans, th.smallPx);
        }
    }
    g.rect(0, sy, w_, th.px(1), th.border);

    g.end();
}

// --- left region: material, flow, grid, geometry -----------------------------

void App::leftRegion()
{
    Gui& g = *gui_;
    const Build& b = doc_.current();
    const Theme& th = g.theme;
    auto& ph = doc_.st.phys;

    g.beginRegion("Case", 0, lay_.menuH, lay_.leftW, lay_.caseH);

    // Which case this is. Everything below gates on it.
    g.label("case", th.textDim);
    if (g.choice("", doc_.st.sub, {kSubNames, 2})) doc_.touch();
    g.separator();
    const bool conv = doc_.st.isConv();

    // --- material -----------------------------------------------------------
    if (g.sectionHeader("material", showMaterial_)) {
        const auto mats = materials();
        std::vector<const char*> names;
        for (const auto& m : mats) names.push_back(m.name);
        // Label on its own row, control full width: five segments sharing half
        // a 320 px column renders as overlapping letters.
        g.label("fluid", th.textDim);
        if (g.choice("", doc_.st.material, {names.data(), names.size()})) {
            if (doc_.st.material != customMaterialIndex()) {
                const auto& m = mats[size_t(doc_.st.material)];
                ph.nu = m.nu;
                ph.c = m.c;
                ph.rho = m.rho;
            }
            doc_.touch();
        }
        const bool custom = doc_.st.material == customMaterialIndex();
        if (custom) {
            if (g.dragUnit("nu", ph.nu, 1e-7, 1e-9, 1.0, "m2/s", 6)) doc_.touch();
            if (g.dragUnit("sound speed", ph.c, 2.0, 1.0, 1e5, "m/s", 5))
                doc_.touch();
            if (g.dragUnit("density", ph.rho, 2.0, 1e-3, 3e4, "kg/m3", 5))
                doc_.touch();
        } else {
            g.field("nu", num(ph.nu, 4) + " m2/s");
            g.field("sound speed", num(ph.c, 4) + " m/s");
            g.field("density", num(ph.rho, 4) + " kg/m3");
        }
    }

    // --- convection ---------------------------------------------------------
    // The SI face again: a layer height and a temperature drop are what a
    // person has, and the Rayleigh number is DERIVED from them and the
    // material -- exactly as Re is derived from a body size and a freestream.
    if (conv && g.sectionHeader("convection", showConv_)) {
        if (g.choice("model", doc_.st.convPhysics, {kConvModelNames, 2}))
            doc_.touch();
        if (g.dragUnit("layer height", doc_.st.convH, 0.001, 1e-5, 10.0, "m", 4))
            doc_.touch();
        if (g.dragUnit("temperature drop", doc_.st.convDT, 0.5, 1e-3, 2000.0,
                       "K", 4))
            doc_.touch();
        if (g.dragUnit("mean temperature", doc_.st.convTmean, 1.0, 1.0, 5000.0,
                       "K", 5))
            doc_.touch();
        if (g.dragDouble("Prandtl", doc_.st.prandtl, 0.005, 1e-3, 1e4, 4))
            doc_.touch();
        if (g.dragDouble("box aspect", doc_.st.convAspect, 0.05, 0.1, 32.0, 3))
            doc_.touch();
        // The lateral faces. Aspect and sides together are the container, which
        // is why they sit next to each other: with periodic sides the aspect is
        // only a sampling window on an unbounded layer, with closed sides it is
        // the cavity itself and fixes the roll count.
        if (g.choice("sides", doc_.st.convSides, {kConvSidesNames, 2}))
            doc_.touch();
        if (doc_.st.convSides == 1) {
            g.label("no-slip adiabatic side walls: a sealed cavity", th.textDim);
            // The two numbers people quote against a convection cell -- the
            // 1708 onset and the Ra-Nu scaling -- are the unbounded layer's,
            // and this box is not that. Measured 2026-07-26 at Ra 5e4, 2D:
            // Nu 3.97 -> 3.65 (Boussinesq), 3.79 -> 3.40 (compressible).
            g.label("Nu is an aspect-dependent cavity number, below the",
                    th.warn);
            g.label("periodic one; onset 1708 is the unbounded layer's",
                    th.warn);
        }
        g.separator();

        const double ra = doc_.st.convRayleigh();
        g.field("Rayleigh", num(ra, 5), th.accent);
        // 1708 is the onset for rigid isothermal plates; below it the layer
        // only conducts, which is a legitimate run and a surprising picture.
        if (ra < 1708.0 && doc_.st.hotPatch >= 1.0)
            g.label("below onset (~1708): the layer will just conduct", th.warn);
        const double uff = ph.freeFall(doc_.st.convDT, doc_.st.convH,
                                       doc_.st.convTmean);
        const double tff = ph.freeFallSeconds(doc_.st.convDT, doc_.st.convH,
                                              doc_.st.convTmean);
        g.field("free fall", num(uff, 4) + " m/s   (" + seconds(tff) + ")");
        g.field("Mach (physical)",
                num(ph.convMachPhysical(doc_.st.convDT, doc_.st.convH,
                                        doc_.st.convTmean), 3),
                th.textDim);
        g.separator();

        // The heated fraction of the floor. A SIDE fraction, so in 3D the
        // heated area is its square -- the read-out says which, because
        // "middle third" means two different things in 2D and 3D.
        if (g.dragDouble("hot patch", doc_.st.hotPatch, 0.005, 0.02, 1.0, 3))
            doc_.touch();
        if (doc_.st.hotPatch < 1.0) {
            const double area = doc_.st.nz == 1
                                    ? doc_.st.hotPatch
                                    : doc_.st.hotPatch * doc_.st.hotPatch;
            g.field("heated area", num(100.0 * area, 3) + " % of the floor",
                    th.textDim);
            g.label("a patch is a localised plume, not Rayleigh-Benard:",
                    th.warn);
            g.label("no 1D conduction state, onset 1708 does not apply",
                    th.warn);
        }
        g.separator();

        if (g.dragDouble("run length", doc_.st.convTt, 0.5, 0.1, 1e5, 4))
            doc_.touch();
        g.field("free-fall times", num(doc_.st.convTt, 4) + "   (" +
                                       seconds(doc_.st.convTt * tff) + ")",
                th.textDim);
        if (g.dragDouble("trip", doc_.st.convKick, 0.005, 0.0, 2.0, 3))
            doc_.touch();
        if (g.dragDouble("frame interval", doc_.st.wdt, 0.0005, 1e-4, 10.0, 4))
            doc_.touch();

        // Compressible-only. The Boussinesq branch REJECTS these by name, so
        // showing them there would be offering an error.
        if (doc_.st.convPhysics == 1) {
            g.separator();
            if (g.dragDouble("solver Mach", doc_.st.convMach, 0.002, 1e-3, 0.9,
                             4))
                doc_.touch();
            // This is NOT the physical Mach above: it is what sets gravity,
            // g = M^2 gamma / dtemp, and therefore how stratified the layer is
            // and how many steps a free-fall time costs. The adiabatic
            // fraction is the honest read-out of the compromise.
            const double eps = doc_.st.convDtemp();
            const double frac = eps > 0 ? doc_.st.convMach * doc_.st.convMach
                                              * 0.4 / (eps * eps)
                                        : 1.0;
            g.field("adiabatic fraction", num(100.0 * frac, 3) + " %",
                    frac >= 1.0 ? th.bad : frac > 0.25 ? th.warn : th.textDim);
            if (frac >= 1.0)
                g.label("subadiabatic: the layer cannot convect", th.bad);
            else if (frac > 0.25)
                g.label("gravity eats much of the drop; lower solver Mach",
                        th.warn);
            if (g.choice("viscosity", doc_.st.visc, {kViscNames, 2}))
                doc_.touch();
        }
    }

    // --- flow ---------------------------------------------------------------
    if (!conv && g.sectionHeader("flow", showFlow_)) {
        if (g.choice("model", doc_.st.physics, {kPhysicsNames, 2})) {
            // Switching model invalidates the far field half the time; move it
            // to a legal one rather than letting the case go red until the user
            // works out that `tunnel` is incompressible-only.
            doc_.st.far = remapFar(doc_.st.far, doc_.st.physics);
            doc_.touch();
        }
        if (g.dragUnit("body size", ph.dBody, 0.001, 1e-6, 1e4, "m", 4))
            doc_.touch();
        if (g.dragUnit("freestream", ph.uInf, 0.05, 1e-6, 1e5, "m/s", 4))
            doc_.touch();
        g.separator();
        // Re and Mach are DERIVED (units.hpp). Shown as read-outs, in the same
        // panel as the metres they come from, so the two can never disagree.
        g.field("Reynolds", num(doc_.st.re(), 5), th.accent);
        g.field("Mach", num(doc_.st.mach(), 4),
                doc_.st.mach() > 0.3 ? th.warn : th.text);
        if (doc_.st.physics == 0 && doc_.st.mach() > 0.3)
            g.label("Ma > 0.3 with incompressible model", th.warn);
        if (doc_.st.physics == 1 && doc_.st.mach() >= 0.99)
            g.label("Ma >= 1: validate() will reject", th.bad);
        g.separator();
        if (g.dragUnit("run length", doc_.st.ft, 0.02, 1e-6, 1e6, "FT", 3))
            doc_.touch();
        if (b.ok) {
            const double dBox = b.geo.D > 0 ? b.geo.D : b.cas().placement.D;
            g.field("run time",
                    seconds(doc_.st.ft * ph.flowThroughSeconds(b.cas().L, dBox)));
        }
        // The live frame cadence (--wdt). This is the output frame rate: every
        // plot, plane and fog frame the viewer draws is one of these, so a run
        // that publishes every 0.15 time units cannot be watched at 30 Hz no
        // matter what the viewer does.
        if (g.dragDouble("frame interval", doc_.st.wdt, 0.0005, 1e-4, 10.0, 4))
            doc_.touch();
        if (b.ok) {
            // One flow-through is L/uInf solver time units, so --wdt converts to
            // physical seconds through the same factor "run time" uses.
            const double perFT = b.cas().L / b.cas().uInf;
            const double tEnd = doc_.st.ft * perFT;
            const int nFrames = int(tEnd / std::max(1e-9, doc_.st.wdt)) + 1;
            const double dBox = b.geo.D > 0 ? b.geo.D : b.cas().placement.D;
            const double perFrame = doc_.st.wdt / perFT *
                                    ph.flowThroughSeconds(b.cas().L, dBox);
            g.field("frames", std::to_string(nFrames) + "   (" +
                                  seconds(perFrame) + " apart)",
                    nFrames > 20000 ? th.warn : th.textDim);
            if (nFrames > 20000)
                g.label("a frame every few steps costs the run real time",
                        th.warn);
        }
        // Only the far fields this physics accepts. The others are not "not
        // recommended", they are rejected by validate() -- offering them made
        // two of every five clicks a red status bar.
        g.label("far field", th.textDim);
        const auto valid = farForPhysics(doc_.st.physics);
        std::vector<const char*> farOpts;
        int sel = 0;
        for (size_t i = 0; i < valid.size(); ++i) {
            farOpts.push_back(farNames()[valid[i]]);
            if (valid[i] == doc_.st.far) sel = int(i);
        }
        if (g.choice("", sel, {farOpts.data(), farOpts.size()})) {
            doc_.st.far = valid[size_t(sel)];
            doc_.touch();
        }
        if (doc_.st.far == int(dns::FarField::Sponge))
            g.label("sponge eats the near field - explicit fallback only", th.warn);
    }

    // --- thermal (compressible body flow) -----------------------------------
    // Folded away by default, and every knob in it is off by default, because
    // a run that names none of them is bit-for-bit the constant-viscosity,
    // adiabatic-wall solver every measured cd/cl anchor was taken on. Turning
    // one on is a deliberate act, and the panel says what it costs.
    if (!conv && doc_.st.physics == 1
        && g.sectionHeader("thermal", showThermal_)) {
        if (g.choice("viscosity", doc_.st.visc, {kViscNames, 2})) doc_.touch();
        if (doc_.st.visc == 1) {
            // Sutherland's 110.4 K is dimensional, so a temperature-dependent
            // run needs one anchor. Nothing else reads it.
            if (g.dragUnit("Tinf", doc_.st.tinf, 1.0, 1.0, 5000.0, "K", 5))
                doc_.touch();
            g.label("mu(T) and kappa(T) = mu cp / Pr", th.textDim);
        }
        if (g.dragDouble("Prandtl", doc_.st.prandtl, 0.005, 1e-3, 1e4, 4))
            doc_.touch();
        g.separator();

        // The BODY surface condition -- not the box walls, which are always
        // adiabatic mirrors.
        int wall = doc_.st.wallIso ? 1 : 0;
        if (g.choice("body wall", wall, {kWallNames, 2})) {
            doc_.st.wallIso = wall == 1;
            doc_.touch();
        }
        if (doc_.st.wallIso) {
            if (g.dragDouble("Twall / Tinf", doc_.st.twall, 0.01, 0.05, 20.0, 4))
                doc_.touch();
            // validate() rejects an isothermal wall with no body, so say it
            // here rather than letting the case go red.
            if (doc_.st.bodyKind == int(dns::BodyKind::None))
                g.label("an isothermal wall needs a body", th.bad);
            g.label("wall heat flux appears as the qw stats column",
                    th.textDim);
        }
        g.separator();

        // NSCBC-only, and validate() says so.
        if (g.checkbox("--nscbc-tinflow (inflow holds T)",
                       doc_.st.nscbcTinflow))
            doc_.touch();
        if (doc_.st.nscbcTinflow && doc_.st.far != 3)
            g.label("needs --far nscbc", th.bad);
        if (g.checkbox("--temp (temperature frame channel)",
                       doc_.st.tempFrames))
            doc_.touch();
        if (doc_.st.tempFrames)
            g.label("adds a third published field: T - Tinf", th.textDim);
    }

    // --- numerics -----------------------------------------------------------
    // Folded away by default: these are the knobs you reach for after the
    // physical case is right, and two of them are documented foot-guns.
    if (!conv && g.sectionHeader("numerics", showNumerics_)) {
        if (g.dragDouble("LES Cs (0 = DNS)", doc_.st.les, 0.002, 0.0, 1.0, 3))
            doc_.touch();
        if (doc_.st.les > 0)
            if (g.choice("SGS", doc_.st.sgs, {kSgsNames, 2})) doc_.touch();
        if (g.checkbox("--cut-flux (apertured momentum)", doc_.st.cutFlux))
            doc_.touch();
        if (g.dragDouble("--penalize eta", doc_.st.penalize, 0.05, 0.0, 1e4, 2))
            doc_.touch();
        if (doc_.st.penalize > 0)
            g.label("penalize HALVES cdf - never for friction", th.warn);
    }

    // --- grid ---------------------------------------------------------------
    if (g.sectionHeader("grid", showGridPanel_)) {
        if (g.dragInt("cells x", doc_.st.n, 0.5, 8, 4096)) doc_.touch();
        if (g.dragInt("cells y (0 = follow x)", doc_.st.ny, 0.5, 0, 4096))
            doc_.touch();
        if (g.dragInt("cells z (1 = 2D)", doc_.st.nz, 0.25, 1, 2048)) doc_.touch();
        // --dr measures the box in body diameters; a convection cell has no
        // body and its box comes from the layer height and the aspect ratio
        // instead, so the row is not shown rather than shown and ignored.
        if (!conv) {
            if (g.dragDouble("domain (box in D)", doc_.st.dr, 0.02, 0.5, 128, 2))
                doc_.touch();
            // --dr is a length in D, so its SI value needs no grid factor.
            g.field("domain size", metres(doc_.st.dr * ph.dBody), th.textDim);
        } else {
            g.field("box", num(doc_.st.convAspect, 3) + " x 1 x " +
                               (doc_.st.nz == 1
                                    ? std::string("-")
                                    : num(doc_.st.convAspect, 3)) +
                               "  layer heights",
                    th.textDim);
            g.field("box size", metres(doc_.st.convAspect * doc_.st.convH) +
                                    " x " + metres(doc_.st.convH),
                    th.textDim);
            g.label("--ny is the FLUID rows; two ghost rows are added",
                    th.textDim);
        }
        if (!conv && doc_.st.physics == 1) {
            if (g.checkbox("--focal auto", doc_.st.focalAuto)) doc_.touch();
            if (doc_.st.focalAuto) {
                if (g.dragDouble("ratio R", doc_.st.focalRatio, 0.01, 1.01, 64, 2))
                    doc_.touch();
                if (g.dragDouble("blend", doc_.st.focalBlend, 0.005, 0.01, 4, 2))
                    doc_.touch();
                if (g.dragDouble("wake (D)", doc_.st.focalWake, 0.02, 0, 100, 2))
                    doc_.touch();
            }
        }
        g.separator();
        if (b.ok) {
            const dns::Grid& gr = b.cas().grid;
            // Which case these numbers came FROM, not which one is selected.
            // The build is asynchronous, so between flipping the case selector
            // and the rebuild landing, `doc_.st` and `b` disagree -- and every
            // row below reads `b`. Gating them on the live selector printed a
            // convection read-out over a body-flow case: `fluid cells` came out
            // 0 x n x 0 (a flow case resolves no fluid counts) under a grid row
            // that was the cylinder's.
            const bool convBuilt = b.cas().sub == dns::Subcommand::Conv;
            g.field("grid", std::to_string(gr.nx) + " x " + std::to_string(gr.ny) +
                                " x " + std::to_string(gr.nz));
            g.field("cells", bigInt(gr.size()));
            // Spacings in metres. These DO need the box factor: unlike --dr and
            // the object offsets, h is a box length, not a multiple of D.
            const double dBox = b.geo.D > 0 ? b.geo.D : b.cas().placement.D;
            // metresPerBox is dBody/dBox, and a convection cell has no body:
            // its physical length scale is the LAYER HEIGHT, which is 1 box
            // unit. Using the body size here would scale every spacing by a
            // millimetre that has nothing to do with the case.
            const double mpb = convBuilt ? (dBox > 0 ? doc_.st.convH / dBox : 0.0)
                                         : ph.metresPerBox(dBox);
            // In 2D, hz is 1 box unit by validator fiat (CLAUDE.md: nz == 1 is
            // the exact 2D restriction, hz = 1 enforced). Printing it as a
            // metre spacing would invent a third dimension the case does not
            // have, so z is dropped rather than converted.
            g.field("cell size",
                    gr.nz == 1 ? metres(gr.hx * mpb) + ", " + metres(gr.hy * mpb)
                               : metres(gr.hx * mpb) + ", " +
                                     metres(gr.hy * mpb) + ", " +
                                     metres(gr.hz * mpb));
            if (b.cas().focalActive)
                g.field("cell size (fine)",
                        metres(b.cas().focalHFine[0] * mpb) + ", " +
                            metres(b.cas().focalHFine[1] * mpb) + ", " +
                            metres(b.cas().focalHFine[2] * mpb));
            if (convBuilt) {
                // The load-bearing row for a convection cell is the THERMAL
                // boundary layer at the plates, delta = H / (2 Nu), not a body
                // boundary layer. Nu is not known before the run, so this uses
                // the a-priori Nu ~ 0.1 Ra^(1/3), which runs high against this
                // repo's own measurements and therefore errs toward calling the
                // grid coarser than it is.
                g.field("layer H", metres(dBox * mpb));
                // The grid row above is the ALLOCATION: two mirror rows for the
                // plates, and two more columns per closed lateral axis. The
                // FLUID counts are what was typed, so print both rather than
                // let the reader wonder which one they are looking at.
                g.field("fluid cells",
                        std::to_string(b.cas().nxInterior) + " x " +
                            std::to_string(gr.ny - 2) + " x " +
                            std::to_string(b.cas().nzInterior),
                        th.textDim);
                g.field("cells across H", num(b.dOverH, 4),
                        b.dOverH >= 64 ? th.good
                                       : (b.dOverH >= 32 ? th.warn : th.bad));
                g.field("thermal BL cells H/(2 Nu h)", num(b.blCells, 3),
                        b.bodyResolved ? th.good : th.bad);
                if (!b.bodyResolved)
                    g.label("under-resolved: Nu becomes a property of the grid",
                            th.warn);
            } else {
            g.field("body D", metres(dBox * mpb));
            g.field("D / h", num(b.dOverH, 4),
                    b.dOverH >= 16 ? th.good : (b.dOverH >= 8 ? th.warn : th.bad));
            // The DNS criterion from CLAUDE.md: the body BL is ~D/sqrt(Re) and
            // needs 3 cells across it. This row is the whole reason the tool
            // shows a grid at all.
            g.field("BL cells (D/h)/sqrt(Re)", num(b.blCells, 3),
                    b.bodyResolved ? th.good : th.bad);
            if (!b.bodyResolved && !b.cas().lesLabel)
                g.label("under-resolved: expect trapped ripples", th.warn);
            }
            g.field("label", b.label, b.cas().lesLabel ? th.warn : th.good);
            if (b.cas().focalActive)
                g.field("dt penalty", num(b.cas().focalDtFactor, 3));
        }
    }

    g.endPanel();
}

// --- geometry: the object list and the transform inspector -------------------
//
// Runs inside the left region's panel (it is a section, not its own region).
// Offsets and diameters are stored in units of D, which is exactly what
// scene.hpp writes to TOML; the metre values shown here are `value * dBody`
// with no grid or box factor, because D IS dBody by definition.
void App::geometryRegion()
{
    Gui& g = *gui_;
    const Theme& th = g.theme;
    const double D = doc_.st.phys.dBody;

    g.beginRegion("Geometry", 0, lay_.geomY, lay_.leftW,
                  lay_.dockY - lay_.geomY);

    // A convection cell has no body. The region stays (the docking layout is
    // fixed) but it says so instead of offering a body picker whose every
    // choice the conv validator would reject.
    if (doc_.st.isConv()) {
        g.label("no body: the case is a heated layer", th.textDim);
        g.label("floor and lid are no-slip isothermal plates", th.textDim);
        // Read-out, not a second edit point: `sides` is set in the convection
        // section, beside the aspect ratio it acts with.
        if (doc_.st.convSides == 1) {
            g.label(doc_.st.nz == 1
                        ? "x sides are no-slip adiabatic walls (2D: z wraps)"
                        : "x and z sides are no-slip adiabatic walls",
                    th.textDim);
            g.label("one ghost column per wall; fluid extent unchanged",
                    th.textDim);
        } else
            g.label("sides are periodic", th.textDim);
        g.endPanel();
        return;
    }

    // Two rows of the ONE table (doc.hpp), split for width only: the second
    // row takes whatever the first leaves, so a body kind added to the solver's
    // enum appears here rather than falling off the end.
    constexpr int kRow1 = 5;
    int bk = doc_.st.bodyKind;
    if (g.choice("", bk, {bodyNames(), kRow1})) {
        doc_.st.bodyKind = bk;
        doc_.touch();
    }
    int bk2 = doc_.st.bodyKind >= kRow1 ? doc_.st.bodyKind - kRow1 : -1;
    if (g.choice("", bk2,
                 {bodyNames() + kRow1, std::size_t(bodyNameCount() - kRow1)})) {
        doc_.st.bodyKind = bk2 + kRow1;
        doc_.touch();
    }

    // Per-body options, from the same capability table toRaw() emits against
    // (doc.hpp). A body that does not take an option does not show it, and the
    // stale value is discarded rather than passed along.
    const BodyOpts bo = bodyOptions(doc_.st.bodyKind);
    if (bo.aspect)
        if (g.dragDouble("aspect (w/h)", doc_.st.aspect, 0.01, 1.05, 20, 2))
            doc_.touch();
    if (bo.dimples)
        if (g.dragInt("dimples", doc_.st.dimples, 0.2, 0, 4096)) doc_.touch();
    if (bo.bumps) {
        if (g.dragInt("trip bumps", doc_.st.bumps, 0.2, 0, 512)) doc_.touch();
        if (doc_.st.bumps > 0)
            if (g.dragUnit("bump height", doc_.st.bumpH, 0.1, 4, 48, "cells", 1))
                doc_.touch();
    }
    if (bo.mesh) {
        g.field("mesh", doc_.st.meshPath.empty() ? "(none: pass a path)"
                                                 : doc_.st.meshPath,
                doc_.st.meshPath.empty() ? th.warn : th.text);
        if (g.dragDouble("scale (0 = fit to D)", doc_.st.meshScale, 0.002, 0, 1e4,
                         4))
            doc_.touch();
        if (g.checkbox("robust winding (leaky mesh)", doc_.st.robust)) doc_.touch();
    }

    if (!doc_.sceneMode()) {
        g.spacing(2);
        g.label("choose `scene` for multiple objects", th.textDim);
        g.endPanel();
        return;
    }

    // Snapping applies to the gizmo drag. Ctrl inverts it live, so the toggle
    // is the default and the modifier is the exception.
    g.separator();
    static const char* const kSnap[3] = {"free", "grid", "D/n"};
    g.label("snap  (ctrl inverts)", th.textDim);
    if (g.choice("", snapMode_, {kSnap, 3}))
        snap_.mode = SnapSettings::Mode(snapMode_);
    if (snap_.mode == SnapSettings::Mode::Fraction)
        g.dragDouble("fraction of D", snap_.fraction, 0.002, 0.01, 2.0, 3);

    // --- object list --------------------------------------------------------
    g.separator();
    auto& items = doc_.st.items;
    if (items.empty()) g.label("no objects - add one below", th.textDim);
    // Deferred: erasing inside the row loop invalidates the very vector being
    // iterated.
    ObjId pendingDelete = kNoObj;
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& o = items[i].obj;
        // Size in the row: with several objects the diameter is what tells them
        // apart, and reading it should not require selecting each one.
        const std::string nm = o.label + "   " + dns::name(o.kind) + "   " +
                               metres(o.d * D);
        if (g.selectable(nm + "##obj" + std::to_string(items[i].id),
                         doc_.isSelected(items[i].id))) {
            doc_.select(items[i].id, in_.ctrl    ? Doc::Sel::Toggle
                                     : in_.shift ? Doc::Sel::Range
                                                 : Doc::Sel::Replace);
        }
    }
    if (pendingDelete != kNoObj) doc_.removeObject(pendingDelete);

    g.spacing(2);
    static const dns::SceneObjectKind kinds[5] = {
        dns::SceneObjectKind::Cylinder, dns::SceneObjectKind::Sphere,
        dns::SceneObjectKind::Cube, dns::SceneObjectKind::Lens,
        dns::SceneObjectKind::Mesh};
    static const char* const addLabels[5] = {"+cyl", "+sph", "+cube", "+lens",
                                             "+mesh"};
    if (const int k = g.buttonRow({addLabels, 5}); k >= 0) {
        // Arm click-to-place: the next viewport click decides where it goes.
        // Falls back to a free spot near the selection if you click a panel.
        placeKind_ = int(kinds[k]);
        placing_ = true;
        status_ = "click in the viewport to place the "
                  + std::string(dns::name(kinds[k]));
    }
    if (placing_) {
        g.label("click in the viewport to place  (Esc cancels)", th.accent);
    }

    const std::size_t nSel = doc_.selection().size();
    SceneItem* sel = doc_.find(doc_.primary());
    if (!sel) {
        g.endPanel();
        return;
    }

    // --- transform inspector ------------------------------------------------
    dns::SceneObject& o = sel->obj;
    const ObjId selId = sel->id;
    g.separator();
    g.label(nSel > 1 ? std::to_string(nSel) + " objects selected  (deltas apply "
                       "to all)"
                     : "selected: " + o.label,
            th.accent);

    if (nSel == 1) {
        // The name is not decoration: it becomes the cd_<name> column in the
        // per-object force CSV and it is what validator errors quote.
        std::string nm = o.label;
        if (g.textInput("name", nm,
                        {.charset = Charset::Token, .maxLen = 48,
                         .hint = "cylinder0"})) {
            if (!doc_.renameObject(selId, nm))
                status_ = "name must be [A-Za-z0-9_-] and unique - kept as '" +
                          std::string(doc_.find(selId)->obj.label) + "'";
        }
        int k = int(o.kind);
        if (g.choice("kind", k, {objectKindNames(), size_t(objectKindCount())})) {
            o.kind = dns::SceneObjectKind(k);
            doc_.touch();
        }
    }

    // Position. Edited in metres and stored in D: the drag step is scaled by
    // 1/D so a pixel of drag moves the same fraction of a body whatever the
    // body's size, which is what makes the drag usable across the 1e-3..1e2 m
    // range these cases span. With several selected, an edit is a DELTA applied
    // to all of them, so the primary's value stays the one shown.
    const double inv = D > 0 ? 1.0 / D : 1.0;
    double px = o.x * D, py = o.y * D, pz = o.z * D, pd = o.d * D;
    const double posStep = std::max(1e-9, D * 0.004);
    if (g.dragUnit("x", px, posStep, -1e4, 1e4, "m", 4)) {
        if (nSel > 1) doc_.nudgeSelection(px * inv - o.x, 0, 0);
        else { o.x = px * inv; doc_.touch(); }
    }
    if (g.dragUnit("y", py, posStep, -1e4, 1e4, "m", 4)) {
        if (nSel > 1) doc_.nudgeSelection(0, py * inv - o.y, 0);
        else { o.y = py * inv; doc_.touch(); }
    }
    // z is only read by the scene builder for the objects that are FINITE in
    // z: a cylinder or lens spans the full box depth by construction
    // (scene.cpp), so offering them a z move would be a control that does
    // nothing. A cube is finite in z (dns::sceneCube) and does read it -- but
    // only in 3D, where nz > 1 already gates every entry here.
    const bool zMeaningful =
        doc_.st.nz > 1 && (o.kind == dns::SceneObjectKind::Sphere ||
                           o.kind == dns::SceneObjectKind::Cube ||
                           o.kind == dns::SceneObjectKind::Mesh);
    if (zMeaningful) {
        if (g.dragUnit("z", pz, posStep, -1e4, 1e4, "m", 4)) {
            if (nSel > 1) doc_.nudgeSelection(0, 0, pz * inv - o.z);
            else { o.z = pz * inv; doc_.touch(); }
        }
    } else {
        g.field("z", doc_.st.nz == 1 ? "- (2D: nz = 1)" : "- (spans the box)",
                th.textDim);
    }

    // Scale. The range is expressed RELATIVE to the reference diameter rather
    // than as the old absolute [1e-9, 1e4] m, for two reasons. It disagreed
    // with the model: 1e-9 m is far below the 1e-6 D floor applied just below,
    // so the field would accept a value the object then snapped away from. And
    // a thirteen-decade absolute range makes the position track meaningless --
    // in D-relative terms an object at exactly D sits at mid-track, which is
    // the one reading worth having here.
    const double dRef = D > 0 ? D : 1.0;
    if (g.dragUnit("diameter", pd, posStep, dRef * 1e-3, dRef * 1e3, "m", 4)) {
        const double want = std::max(1e-6, pd * inv);
        if (nSel > 1) doc_.scaleSelection(o.d > 0 ? want / o.d : 1.0);
        else { o.d = want; doc_.touch(); }
    }
    g.field("relative to D", num(o.d, 3) + " D", th.textDim);
    if (nSel == 1 && o.kind == dns::SceneObjectKind::Lens)
        if (g.dragDouble("aspect", o.aspect, 0.01, 1.05, 20, 2)) doc_.touch();
    if (nSel == 1 && o.kind == dns::SceneObjectKind::Mesh) {
        std::string path = o.meshPath;
        if (g.textInput("mesh file", path,
                        {.charset = Charset::Path, .maxLen = 512,
                         .hint = "drop a .stl here", .selectAllOnFocus = false})) {
            doc_.beginEdit("set mesh path");
            o.meshPath = path;
            doc_.touch();
            doc_.endEdit();
        }
        if (g.dragDouble("mesh scale (0 = fit)", o.scale, 0.002, 0, 1e4, 4))
            doc_.touch();
        if (g.checkbox("robust winding", o.robust)) doc_.touch();
    }

    g.spacing(2);
    static const char* const actLabels[3] = {"duplicate", "array", "remove"};
    const int a = g.buttonRow({actLabels, 3});
    if (a == 0) {
        doc_.duplicateObject(selId);
    } else if (a == 1) {
        showArray_ = !showArray_;
    } else if (a == 2) {
        if (nSel > 1) doc_.removeSelection();
        else doc_.removeObject(selId);
    }
    if (showArray_) {
        g.dragInt("copies", arrayCount_, 0.05, 2, 64);
        g.dragDouble("pitch x (D)", arrayDx_, 0.01, -20, 20, 2);
        g.dragDouble("pitch y (D)", arrayDy_, 0.01, -20, 20, 2);
        if (doc_.st.nz > 1) g.dragDouble("pitch z (D)", arrayDz_, 0.01, -20, 20, 2);
        if (g.button("make array")) {
            const int made = doc_.arrayObjects(selId, arrayCount_, arrayDx_,
                                               arrayDy_, arrayDz_);
            status_ = "arrayed " + std::to_string(made) + " copies";
            showArray_ = false;
        }
    }
    g.endPanel();
}

// --- right region: cut census, layers, export --------------------------------

void App::rightRegion()
{
    Gui& g = *gui_;
    const Build& b = doc_.current();
    const Theme& th = g.theme;
    g.beginRegion("Diagnostics", w_ - lay_.rightW, lay_.menuH, lay_.rightW,
                  lay_.vpH);

    // --- cut cells ----------------------------------------------------------
    if (g.sectionHeader("cut cells", showCutPanel_)) {
        if (b.ok) {
            g.field("cut cells", bigInt(b.nCut));
            g.field("solid cells", bigInt(b.nSolid));
            g.bar("cut fraction of grid",
                  b.nCells ? double(b.nCut) / double(b.nCells) : 0.0, th.accent);
            g.field("min lambda", num(b.minLambda, 3),
                    b.minLambda >= 1e-2 ? th.good : th.warn);
            g.field("min aperture", num(b.minAperture, 3),
                    b.minAperture >= 1e-2 ? th.good : th.warn);
            g.separator();
            g.label("sanitizeCut", th.textDim);
            g.field("solidified", bigInt(b.sanitize.cellsSolidified));
            g.field("faces closed", bigInt(b.sanitize.facesClosed));
            g.field("pockets closed", bigInt(b.sanitize.pocketsClosed),
                    b.sanitize.pocketsClosed ? th.warn : th.text);
        }
    }

    // --- layers -------------------------------------------------------------
    if (g.sectionHeader("layers", showLayers_)) {
        bool dirty = false;
        dirty |= g.checkbox("grid lines", opts_.showGrid);
        dirty |= g.checkbox("body surface", opts_.showSurface);
        dirty |= g.checkbox("cut cells (lambda)", opts_.showCut);
        dirty |= g.checkbox("solid mask (staircase)", opts_.showSolid);
        dirty |= g.checkbox("face apertures", opts_.showApertures);
        dirty |= g.checkbox("domain box", opts_.showBox);
        if (b.ok && b.cas().grid.nz > 1) {
            dirty |= g.choice("slice axis", opts_.sliceAxis, {kAxisNames, 3});
            const int nAx = opts_.sliceAxis == 0   ? b.cas().grid.nx
                            : opts_.sliceAxis == 1 ? b.cas().grid.ny
                                                   : b.cas().grid.nz;
            dirty |= g.dragInt("slice index", opts_.sliceIndex, 0.25, 0, nAx - 1);
            dirty |= g.dragInt("slab half-width", opts_.slabCells, 0.05, 0, 64);
            dirty |= g.checkbox("whole body (slow)", opts_.wholeBody);
        }
        if (dirty) buildViewNow_ = true;
        g.separator();
        g.swatch(rgb(0x4d8cf2), "lambda ~ 1  (barely cut)");
        g.swatch(rgb(0x40ccb3), "lambda ~ 0.6");
        g.swatch(rgb(0xf2bf40), "lambda ~ 0.3");
        g.swatch(rgb(0xeb4d47), "lambda -> 0  (sliver)");
        g.swatch(rgb(0x39434f), "solid");
        if (meshes_.stats.strideUsed > 1)
            g.label("grid lines strided 1:" + std::to_string(meshes_.stats.strideUsed),
                    th.textDim);
        if (meshes_.stats.capped)
            g.label("cell cap hit - showing a subset", th.warn);
    }

    liveSection();

    // --- export -------------------------------------------------------------
    if (g.sectionHeader("export", showExport_)) {
        if (b.sceneEmpty)
            g.label("NOT RUNNABLE: the scene has no objects", th.bad);
        const std::string cmd = doc_.commandLine();
        // Wrapped by hand: the command line is the artefact people copy out of
        // this tool, so it must be readable, not elided.
        std::string linebuf;
        for (size_t i = 0; i < cmd.size();) {
            size_t j = i;
            while (j < cmd.size() &&
                   g.textWidth(cmd.substr(i, j - i + 1), FontId::Mono, th.smallPx) <
                       g.contentWidth())
                ++j;
            if (j > i && j < cmd.size()) {
                const size_t sp = cmd.rfind(' ', j);
                if (sp != std::string::npos && sp > i) j = sp;
            }
            g.label(cmd.substr(i, j - i), th.textDim);
            i = j < cmd.size() && cmd[j] == ' ' ? j + 1 : j;
        }
        if (doc_.sceneMode()) {
            (void)g.textInput("scene file", scenePath_,
                              {.charset = Charset::Path, .maxLen = 512,
                               .selectAllOnFocus = false});
            static const char* const io[2] = {"write scene", "reload scene"};
            const int a = g.buttonRow({io, 2});
            if (a == 0) {
                // writeScene pre-flights the parser, so a failure here is the
                // parser's message and no file is written.
                const std::string err = doc_.writeScene(scenePath_);
                status_ = err.empty() ? "wrote " + scenePath_ : err;
            } else if (a == 1) {
                const std::string err = doc_.openScene(scenePath_);
                status_ = err.empty() ? "opened " + scenePath_ : err;
            }
            g.label("drop a .toml on the window to open it", th.textDim);
        }
        // "##describe" pins the identity: a caption that changes between press
        // and release would otherwise be a different widget by the time the
        // click completes.
        if (g.button(showDescribe_ ? "hide resolved case##describe"
                                   : "show resolved case##describe"))
            showDescribe_ = !showDescribe_;
        if (showDescribe_ && b.ok) {
            g.separator();
            std::string line;
            for (char c : b.describe) {
                if (c == '\n') {
                    g.label(line, th.textDim);
                    line.clear();
                } else {
                    line += c;
                }
            }
            if (!line.empty()) g.label(line, th.textDim);
        }
    }
    g.endPanel();
}

// Where to run. `dns` is invoked by absolute path so the run does not depend
// on PATH, and the cwd decides where its output lands.
std::string App::runDir() const
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path p = fs::current_path(ec);
    if (ec) return ".";
    // Walk up looking for the build tree the editor was built into. The editor
    // is normally started from the repo root, but not always.
    for (int i = 0; i < 4 && !p.empty(); ++i) {
        for (const char* b : {"build-cuda", "build"})
            if (fs::exists(p / b / "dns", ec)) return p.string();
        p = p.parent_path();
    }
    return fs::current_path(ec).string();
}

void App::startRun()
{
    namespace fs = std::filesystem;
    const Build& b = doc_.current();
    if (!b.ok) {
        status_ = "cannot run: fix the case first";
        dockOpen_ = true;
        return;
    }
    if (b.sceneEmpty) {
        status_ = "cannot run: the scene has no objects";
        dockOpen_ = true;
        return;
    }

    if (live_.attached() && live_.running() && !run_.running()) {
        status_ = "already watching a live run ('" + live_.name() +
                  "'): detach before starting another";
        dockOpen_ = true;
        return;
    }

    const std::string dir = runDir();
    std::error_code ec;
    std::string exe;
    for (const char* cand : {"build-cuda/dns", "build/dns"})
        if (fs::exists(fs::path(dir) / cand, ec)) {
            exe = (fs::path(dir) / cand).string();
            break;
        }
    if (exe.empty()) {
        status_ = "cannot run: no build/dns or build-cuda/dns under " + dir;
        dockOpen_ = true;
        return;
    }

    // In scene mode the solver reads the scene from a FILE, so the document has
    // to be on disk before it can run. writeScene pre-flights the parser, so a
    // scene the CLI would reject stops here with the parser's own message
    // rather than as a failed run.
    std::string scenePath = scenePath_;
    if (doc_.sceneMode()) {
        const std::string err = doc_.writeScene((fs::path(dir) / scenePath).string());
        if (!err.empty()) {
            status_ = "cannot run: " + err;
            dockOpen_ = true;
            return;
        }
    }

    auto argv = doc_.commandArgs(exe, scenePath);
    // The live bridge: publish frames into shared memory (no disk), and keep
    // the stats stream, which is what the dock plots. `--out` fixes the segment
    // name so the viewer knows what to attach to.
    argv.push_back("--out");
    argv.push_back("dnsui");
    argv.push_back("--emit");
    argv.push_back("shm");
    argv.push_back("--emit");
    argv.push_back("stats");
    liveName_ = "dns_dnsui";
    runCmd_.clear();
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (i) runCmd_ += ' ';
        runCmd_ += argv[i];
    }
    if (const std::string err = run_.start(argv, dir); !err.empty()) {
        status_ = "cannot run: " + err;
    } else {
        status_ = "running";
        runFollow_ = true;
        live_.detach();
        liveOptOut_ = false;
        scrub_ = -1;   // a new run has no past to be parked in
        series_ = {};
        seriesConsumed_ = 0;
    }
    dockOpen_ = true;   // you asked it to run; show what it is doing
}

// Rebuilds the live plane / fog geometry. Outside the render pass, like the
// selection mesh: growing a vertex buffer waits for the device, which must not
// happen inside an open pass.
void App::updateLiveMeshes()
{
    if (!liveDirty_) return;
    liveDirty_ = false;
    liveVerts_.clear();
    livePlaneStats_ = {};
    liveFogStats_ = {};

    const auto& fs = live_.fields();
    if (!fs.empty() && showLive_) {
        const auto& f = fs[std::size_t(std::clamp(live_.selected(), 0,
                                                  int(fs.size()) - 1))];
        const bool div = f.diverging != 0;
        // Rewound frames carry planes only, so the same builders run on a
        // volume-less view and the fog simply has nothing to draw.
        const LiveSnapshot* s = scrubbed();
        const LiveView v = s ? live_.viewOf(*s) : live_.view();
        buildLivePlane(v, liveOpts_, div, f.lo, f.hi, liveVerts_,
                       livePlaneStats_);
        if (liveOpts_.showFog) {
            buildLiveFog(v, liveOpts_, div, f.lo, f.hi, cam_.forward(),
                         liveVerts_, liveFogStats_);
        }
    }
    (void)ren_->upload(liveMesh_, std::span<const Vtx3>(liveVerts_));
}

// The frame the viewer is SHOWING: null while following the run, otherwise the
// snapshot the timeline is parked on. Clamped rather than validated, because
// the history evicts from the front under a live run -- an index that was valid
// when the user let go can be one frame stale by the time it is read.
const LiveSnapshot* App::scrubbed() const
{
    const auto& h = live_.history();
    if (scrub_ < 0 || h.empty()) return nullptr;
    return &h[std::size_t(std::min(std::size_t(scrub_), h.size() - 1))];
}

// Alpha compositing is only correct back-to-front, so the fog mesh belongs to
// the octant it was built in. Orbiting past a 45-degree diagonal flips the
// slice axis or its direction, and the stack must be rebuilt -- otherwise the
// fog quietly composites inside out, which looks like the data went wrong.
void App::checkFogOrder()
{
    if (!liveOpts_.showFog || !liveFogStats_.quads) return;
    bool asc = false;
    const int axis = fogOrder(cam_.forward(), asc);
    if (axis != liveFogStats_.fogAxis || asc != liveFogStats_.fogAscending)
        liveDirty_ = true;
}

// The solver's own stats lines, parsed straight out of the captured stdout:
//   t=8.21/12.57 cd=1.456 (cdf=0.506) cl=-0.173 K=22.9 maxDiv=1.16e-04 ...
// and, for a convection run,
//   t=12.00/40.00 Nu=10.559/10.548 (flux 10.486) urms=0.244 dt=4.13e-03
// No second channel and no file -- the run log IS the series.
void App::parseSeries()
{
    const auto lines = run_.lines();
    for (std::size_t i = seriesConsumed_; i < lines.size(); ++i) {
        const std::string& ln = lines[i];
        if (ln.rfind("t=", 0) != 0) continue;
        auto num = [&](const char* key, float& out) {
            const std::size_t p = ln.find(key);
            if (p == std::string::npos) return false;
            const auto v =
                leadingNumber(std::string_view(ln).substr(p + std::strlen(key)));
            if (!v) return false;
            out = float(*v);
            return true;
        };
        float t = 0, cd = 0, cl = 0, kk = 0, dv = 0;
        if (!num("t=", t)) continue;
        num("cd=", cd);
        num("cl=", cl);
        num("K=", kk);
        num("maxDiv=", dv);
        series_.t.push_back(t);
        series_.cd.push_back(cd);
        series_.cl.push_back(cl);
        series_.k.push_back(kk);
        series_.div.push_back(dv);

        // Convection: "Nu=10.559/10.548 (flux 10.486) urms=0.244". The two
        // wall values share one key, so the second is read past the slash.
        float nb = 0, nt = 0, nf = 0, ur = 0;
        if (num("Nu=", nb)) {
            const std::size_t p = ln.find("Nu=");
            const std::size_t sl = ln.find('/', p);
            if (sl != std::string::npos)
                if (const auto v = leadingNumber(std::string_view(ln).substr(sl + 1)))
                    nt = float(*v);
        }
        num("(flux ", nf);
        // The compressible runner prints urms as a ratio to the free-fall
        // velocity and the Boussinesq one in box units; both are the same
        // column, so the longer key is tried first.
        if (!num("urms/uff=", ur)) num("urms=", ur);
        series_.nuBot.push_back(nb);
        series_.nuTop.push_back(nt);
        series_.nuFlux.push_back(nf);
        series_.urms.push_back(ur);
    }
    seriesConsumed_ = lines.size();
}

// The live-results controls. Lives in the right column beside the cut-cell
// census, because it is the same question asked of a running solve rather than
// of the geometry.
void App::liveSection()
{
    Gui& g = *gui_;
    const Theme& th = g.theme;
    if (!g.sectionHeader("live results", showLiveSection_)) return;

    if (!live_.attached()) {
        if (liveOptOut_) {
            g.label("detached from live results", th.textDim);
            if (g.button("attach to a live run")) {
                liveOptOut_ = false;
                status_ = "looking for a live run...";
            }
            return;
        }
        g.label(run_.running() ? "waiting for the first frame..."
                               : "press RUN to see live fields here",
                th.textDim);
        g.label("frames arrive in shared memory; nothing is written to disk.",
                th.textDim);
        return;
    }

    {
        const LiveSnapshot* s = scrubbed();
        g.field("frame",
                std::to_string(s ? s->frame : live_.frame()) + "   t = " +
                    num(s ? s->t : live_.time(), 4),
                s ? th.warn : (live_.running() ? th.good : th.textDim));
    }
    g.field("source", live_.name() + (run_.running() ? "  (this window)"
                                                     : "  (external run)"),
            th.textDim);
    if (!live_.running()) g.label("run finished - showing the last frame", th.textDim);

    // --- the timeline -------------------------------------------------------
    // Every frame this viewer decoded is kept (planes only -- see
    // LiveSnapshot), so a run can be scrubbed back through while it is still
    // going, and a finished run stays scrubbable until the viewer is closed.
    {
        const auto& hist = live_.history();
        const int n = int(hist.size());
        g.separator();
        if (n <= 1) {
            g.field("timeline", n ? "1 frame" : "no frames yet", th.textDim);
        } else {
            int idx = scrub_ < 0 ? n - 1 : std::min(scrub_, n - 1);
            if (g.dragInt("timeline", idx, 0.25, 0, n - 1)) {
                scrub_ = idx >= n - 1 ? -1 : idx;   // the tail IS following
                liveDirty_ = true;
            }
            static const char* const kNav[4] = {"|<", "<", ">", "live"};
            const int nav = g.buttonRow({kNav, 4});
            if (nav >= 0) {
                const int cur = scrub_ < 0 ? n - 1 : std::min(scrub_, n - 1);
                int want = cur;
                if (nav == 0) want = 0;
                else if (nav == 1) want = cur - 1;
                else if (nav == 2) want = cur + 1;
                else want = n - 1;
                want = std::clamp(want, 0, n - 1);
                scrub_ = (nav == 3 || want >= n - 1) ? -1 : want;
                liveDirty_ = true;
            }
            g.label(scrub_ < 0 ? "following the run   (left/right arrows step)"
                               : "rewound " +
                                     std::to_string(n - 1 - std::min(scrub_, n - 1)) +
                                     " frames - press live to catch up",
                    scrub_ < 0 ? th.textDim : th.warn);
            if (scrub_ >= 0)
                g.label("rewind shows the published centre planes; fog and the "
                        "off-centre plane need the volume, which is not kept",
                        th.textDim);
        }
        g.field("history",
                std::to_string(n) + " frames, " +
                    num(double(live_.historyBytes()) / (1 << 20), 3) + " MB of " +
                    std::to_string(live_.historyBudget() >> 20) + " MB",
                th.textDim);
    }
    g.separator();

    // Which published field to colour by.
    const auto& fs = live_.fields();
    if (fs.size() > 1) {
        std::vector<const char*> names;
        for (const auto& f : fs) names.push_back(f.name);
        int sel = std::clamp(live_.selected(), 0, int(fs.size()) - 1);
        g.label("field", th.textDim);
        if (g.choice("", sel, {names.data(), names.size()})) {
            live_.select(sel);
            // The timeline restarts with the new field (see LiveField::poll),
            // so a rewound position would point into a series that no longer
            // exists. Catch back up to the live frame instead.
            scrub_ = -1;
            liveDirty_ = true;
        }
    }

    bool dirty = false;
    dirty |= g.checkbox("show live field", showLive_);
    dirty |= g.checkbox("cut plane", liveOpts_.showPlane);
    if (liveOpts_.showPlane && live_.planes().size() > 1) {
        static const char* const kPlanes[3] = {"yz", "xz", "xy"};
        int idx = liveOpts_.plane;
        g.label("plane", th.textDim);
        if (g.choice("", idx, {kPlanes, 3})) {
            liveOpts_.plane = idx;
            dirty = true;
        }
    }
    if (liveOpts_.showPlane) {
        if (live_.hasVolume()) {
            // Sliced out of the published volume by the viewer, so dragging
            // this is instant -- the solver is not asked for a new plane.
            double at = liveOpts_.planeAt;
            if (g.dragDouble("plane position", at, 0.002, 0.0, 1.0, 3)) {
                liveOpts_.planeAt = float(at);
                dirty = true;
            }
            if (livePlaneStats_.sliceIndex >= 0)
                g.field("slice", std::to_string(livePlaneStats_.sliceIndex),
                        th.textDim);
        } else {
            g.field("plane position", "centre only", th.textDim);
            g.label(live_.volD() > 0 || live_.planes().size() > 1
                        ? "no volume published: the run is 2D or over the "
                          "volume budget"
                        : "2D run: one plane",
                    th.textDim);
        }
    }
    {
        double a = liveOpts_.planeAlpha;
        if (g.dragDouble("plane opacity", a, 0.004, 0.05, 1.0, 2)) {
            liveOpts_.planeAlpha = float(a);
            dirty = true;
        }
    }

    // Volumetric fog: the whole 3D field as a stack of translucent slices.
    if (live_.hasVolume()) {
        // Volumetric rendering of whichever field is selected above -- vort or
        // speed -- because every field's volume is published, not just the
        // first.
        dirty |= g.checkbox("volumetric fog", liveOpts_.showFog);
        if (liveOpts_.showFog) {
            double a = liveOpts_.fogAlpha;
            if (g.dragDouble("fog density", a, 0.0005, 0.002, 0.5, 3)) {
                liveOpts_.fogAlpha = float(a);
                dirty = true;
            }
            double fl = liveOpts_.fogFloor;
            if (g.dragDouble("fog threshold", fl, 0.002, 0.0, 0.9, 3)) {
                liveOpts_.fogFloor = float(fl);
                dirty = true;
            }
            // How wide the threshold's ramp is. Drag it to zero and the fog
            // grows visible shells where the field crosses the threshold: the
            // softness is what makes it read as fog rather than as contours.
            double fsoft = liveOpts_.fogSoft;
            if (g.dragDouble("fog softness", fsoft, 0.002, 0.0, 0.6, 3)) {
                liveOpts_.fogSoft = float(fsoft);
                dirty = true;
            }
            int sl = liveOpts_.fogSlices;
            if (g.dragInt("fog slices", sl, 0.5, 8, 512)) {
                liveOpts_.fogSlices = sl;
                dirty = true;
            }
            g.field("slices drawn", std::to_string(liveFogStats_.slices), th.textDim);
            if (liveFogStats_.slices && liveFogStats_.fogStep > 1)
                g.label("one slice per " + std::to_string(liveFogStats_.fogStep) +
                            " cells: density is compensated, but raise the "
                            "slice count if it bands",
                        th.textDim);
        }
    }
    // Relative colouring: the red/blue/dark look, applied to a MAGNITUDE by
    // re-centring it on a reference speed. Slower than the tunnel reads blue,
    // faster reads red, at-tunnel-speed sinks into the background.
    g.separator();
    static const char* const kRef[3] = {"field", "vs mean", "vs tunnel"};
    int rm = int(liveOpts_.refMode);
    g.label("colour", th.textDim);
    if (g.choice("", rm, {kRef, 3})) {
        liveOpts_.refMode = RefMode(rm);
        dirty = true;
    }
    if (liveOpts_.refMode != RefMode::Off) {
        double ct = liveOpts_.contrast;
        if (g.dragDouble("contrast", ct, 0.01, 0.1, 20.0, 2)) {
            liveOpts_.contrast = float(ct);
            dirty = true;
        }
        // The tunnel speed is the case's own uInf -- the editor knows it from
        // the resolved Case, so it is never a guess.
        const Build& cb = doc_.current();
        if (cb.ok) liveOpts_.refValue = float(cb.cas().uInf);
        const double mps = doc_.st.phys.uInf;   // the same speed, in m/s
        g.field("reference",
                num(livePlaneStats_.refUsed, 4) + "  (" +
                    num(livePlaneStats_.refUsed * (liveOpts_.refValue > 0
                                                       ? mps / liveOpts_.refValue
                                                       : 0.0),
                        4) +
                    " m/s)",
                th.textDim);
        g.field("span +/-", num(livePlaneStats_.spanUsed, 4), th.textDim);
        g.swatch(rgb(0x4090ff), "slower than the reference");
        g.swatch(rgb(0xff5933), "faster than the reference");
    }

    dirty |= g.checkbox("hide solid cells", liveOpts_.hideSolid);
    if (dirty) liveDirty_ = true;

    g.spacing(2);
    static const char* const kLiveActs[2] = {"stop run", "detach"};
    const int la = g.buttonRow({kLiveActs, 2});
    if (la == 0) {
        if (run_.running()) {
            run_.stop();
            status_ = "run stopped";
        } else {
            const std::string err = live_.stopPublisher();
            status_ = err.empty() ? "stopped the live run" : err;
        }
    } else if (la == 1) {
        // Sticky: without the opt-out the auto-attach scan reattaches within
        // half a second and detaching looks broken.
        live_.detach();
        liveName_.clear();
        liveOptOut_ = true;
        scrub_ = -1;
        liveDirty_ = true;
        status_ = "detached - the run keeps going; press RUN or restart to reattach";
    }

    g.field("plane quads", bigInt(livePlaneStats_.quads), th.textDim);
    if (liveFogStats_.quads)
        g.field("fog quads", bigInt(liveFogStats_.quads), th.textDim);
    if (livePlaneStats_.stride > 1 || liveFogStats_.stride > 1)
        g.label("strided 1:" + std::to_string(std::max(livePlaneStats_.stride,
                                                       liveFogStats_.stride)) +
                    " to stay in budget",
                th.textDim);
}

// --- bottom dock -------------------------------------------------------------
//
// The frame for live cut-plane plots. It is built now and empty on purpose:
// the layout, the splitter and the viewport arithmetic are the part that has
// to be right before a field stream exists, and wiring them once means the
// plots arrive as content rather than as another reshuffle of every rect.
// Field data will come from a `dns` subprocess (docs/ui.md), which is why
// dns-ui still links dns_core only.
// One line plot, drawn with the raw 2D primitives Gui already exposes for the
// viewport overlay. No chart library, and no second data path: the series come
// from the solver's OWN stdout, which the runner is already capturing.
void App::plot(float x, float y, float w, float h, const char* title,
               const std::vector<float>& xs, const std::vector<float>& ys,
               Rgba col)
{
    Gui& g = *gui_;
    const Theme& th = g.theme;
    g.rect(x, y, w, h, rgb(0x10141a), th.px(3));
    g.rectOutline(x, y, w, h, th.border, th.px(1));

    if (ys.size() < 2) {
        g.textAt(std::string(title) + "  (waiting)", x + th.px(6),
                 y + th.px(14), th.textDim, FontId::Sans, th.smallPx);
        return;
    }
    float lo = ys[0], hi = ys[0];
    for (float v : ys) {
        if (!std::isfinite(v)) continue;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    if (!(hi > lo)) {
        hi = lo + 1.0f;
        lo -= 1.0f;
    }
    const float pad = (hi - lo) * 0.08f;
    lo -= pad;
    hi += pad;
    const float t0 = xs.front(), t1 = xs.back();
    const float span = t1 > t0 ? t1 - t0 : 1.0f;
    const float x0 = x + th.px(4), y0 = y + th.px(18);
    const float pw = w - th.px(8), ph = h - th.px(24);

    // A zero line, when zero is in range: cl swinging about zero is the whole
    // signal in a shedding case.
    if (lo < 0 && hi > 0) {
        const float zy = y0 + ph * (1.0f - (0.0f - lo) / (hi - lo));
        g.rect(x0, zy, pw, th.px(1), th.border);
    }
    // Polyline as thin quads: the line pipeline is 3D, and this is 2D chrome.
    float px = x0, py = y0 + ph * (1.0f - (ys[0] - lo) / (hi - lo));
    for (std::size_t i = 1; i < ys.size(); ++i) {
        const float fx = x0 + pw * ((xs[i] - t0) / span);
        const float fy = y0 + ph * (1.0f - (ys[i] - lo) / (hi - lo));
        const float dx = fx - px, dy = fy - py;
        const float len = std::sqrt(dx * dx + dy * dy);
        if (len > 0.01f) {
            const int steps = int(len / th.px(1.2f)) + 1;
            for (int k2 = 0; k2 <= steps; ++k2) {
                const float u = float(k2) / steps;
                g.rect(px + dx * u, py + dy * u, th.px(1.6f), th.px(1.6f), col);
            }
        }
        px = fx;
        py = fy;
    }
    char lab[96];
    std::snprintf(lab, sizeof lab, "%s   %.4g", title, double(ys.back()));
    g.textAt(lab, x + th.px(6), y + th.px(14), col, FontId::Sans, th.smallPx);
}

void App::dockRegion()
{
    Gui& g = *gui_;
    const Theme& th = g.theme;

    if (!dockOpen_) {
        g.strip(0, lay_.dockY, w_, lay_.dockHeadH, th.panelHead);
        g.rect(0, lay_.dockY, w_, th.px(1), th.border);
        const std::string s =
            run_.running()
                ? "run  (" + num(run_.seconds(), 3) + " s, " +
                      std::to_string(series_.t.size()) + " samples)"
                : (run_.lineCount() ? "run  (finished, exit " +
                                          std::to_string(run_.exitCode()) + ")"
                                    : "run  (collapsed - press RUN)");
        g.textAt(s, th.px(12), lay_.dockY + th.px(17),
                 run_.running() ? th.accent : th.textDim, FontId::Sans, th.smallPx);
        return;
    }

    g.beginRegion("Run", 0, lay_.dockY, w_, lay_.dockTotalH);
    if (run_.running())
        g.field("status", "running   " + num(run_.seconds(), 3) + " s   " +
                              std::to_string(series_.t.size()) + " samples",
                th.accent);
    else if (run_.lineCount())
        g.field("status", "exit " + std::to_string(run_.exitCode()) + "   " +
                              num(run_.seconds(), 3) + " s",
                run_.exitCode() == 0 ? th.good : th.bad);
    else
        g.field("status", "idle - press RUN", th.textDim);
    g.endPanel();

    // The plots occupy the dock body, beside the tail of the log. Drawn with
    // raw primitives rather than as panel rows because a chart is not a row.
    const float top = lay_.dockY + th.rowH * 2.2f;
    const float bot = lay_.dockY + lay_.dockTotalH - th.px(4);
    const float ph = std::max(th.px(40), bot - top);
    const float logW = std::min(w_ * 0.42f, th.px(560));
    const float plotsW = w_ - logW - th.px(12);
    const int nPlots = 4;
    const float pw = std::max(th.px(80), (plotsW - th.px(8) * (nPlots + 1)) / nPlots);
    float px = th.px(8);
    if (doc_.st.isConv()) {
        // Floor and lid Nusselt numbers side by side is the diagnostic: they
        // fluctuate independently and must agree in the MEAN, which is the
        // heat balance closing.
        plot(px, top, pw, ph, "Nu floor", series_.t, series_.nuBot, th.accent);
        px += pw + th.px(8);
        plot(px, top, pw, ph, "Nu lid", series_.t, series_.nuTop,
             rgb(0x5fd08a));
        px += pw + th.px(8);
        plot(px, top, pw, ph, "Nu flux", series_.t, series_.nuFlux,
             rgb(0xf0b45a));
        px += pw + th.px(8);
        plot(px, top, pw, ph, "u rms", series_.t, series_.urms, rgb(0xef6b6b));
    } else {
        plot(px, top, pw, ph, "cd", series_.t, series_.cd, th.accent);
        px += pw + th.px(8);
        plot(px, top, pw, ph, "cl", series_.t, series_.cl, rgb(0x5fd08a));
        px += pw + th.px(8);
        plot(px, top, pw, ph, "K", series_.t, series_.k, rgb(0xf0b45a));
        px += pw + th.px(8);
        plot(px, top, pw, ph, "maxDiv", series_.t, series_.div, rgb(0xef6b6b));
    }

    // The tail of the raw log on the right: the plots are a summary, the log is
    // the truth, and a warning only ever appears in the latter.
    const auto lines = run_.lines();
    const float lx = w_ - logW - th.px(4);
    g.rect(lx, top, logW, ph, rgb(0x10141a), th.px(3));
    g.rectOutline(lx, top, logW, ph, th.border, th.px(1));
    const std::size_t rows =
        std::size_t(std::max(1.0f, (ph - th.px(8)) / (th.smallPx + th.px(4))));
    const std::size_t first = lines.size() > rows ? lines.size() - rows : 0;
    float ly = top + th.px(14);
    if (lines.empty())
        g.textAt("the solver's output appears here", lx + th.px(6), ly, th.textDim,
                 FontId::Sans, th.smallPx);
    for (std::size_t i = first; i < lines.size(); ++i) {
        const std::string& ln = lines[i];
        const bool warn = ln.find("warning") != std::string::npos;
        const bool bad = !warn && ln.rfind("dns: ", 0) == 0;
        g.textAt(ln, lx + th.px(6), ly, bad ? th.bad : (warn ? th.warn : th.textDim),
                 FontId::Mono, th.smallPx);
        ly += th.smallPx + th.px(4);
    }
}

// --- 3D ---------------------------------------------------------------------

void App::drawScene()
{
    // The scene is projected into the VIEWPORT rect, not the window: aspect
    // and the Vulkan viewport must agree or the body comes out sheared.
    ren_->setViewport(lay_.vpX, lay_.vpY, lay_.vpW, lay_.vpH);
    const float aspect = lay_.vpW / std::max(1.0f, lay_.vpH);
    const Mat4 mvp = cam_.viewProj(aspect);
    const Vec3 light = normalize({0.4f, 0.7f, 0.6f});

    ren_->drawTriangles(meshes_.surface, mvp, light, 1.0f, true);
    ren_->drawLines(meshes_.lines, mvp, 1.0f);
    ren_->drawTriangles(meshes_.cells, mvp, light, 1.0f, false);
    // The live field, on the same translucent pass the cut cells use: depth
    // tested so the body occludes it, no depth write so slices composite.
    if (liveMesh_.count && showLive_)
        ren_->drawTriangles(liveMesh_, mvp, light, 1.0f, false);

    if (selMesh_.count) {
        // Twice: depth-tested at full alpha, then depth-disabled and faint --
        // the standard x-ray look, so a selection inside the body is still
        // visible without floating on top of everything.
        ren_->drawLines(selMesh_, mvp, Rgba{255, 255, 255, 255}, true);
        ren_->drawLines(selMesh_, mvp, Rgba{255, 255, 255, 80}, false);
    }
    // The gizmo last and on top: a handle hidden by the body cannot be grabbed.
    if (gizMesh_.count)
        ren_->drawTriangles(gizMesh_, mvp, light, Rgba{255, 255, 255, 255}, false,
                            false);
}

// The selection box is computed from the EDITED offsets, not from the last
// build, so dragging an object is responsive even while the cut rebuilds in
// the background. Uploaded here rather than in drawScene(): growing a buffer
// waits for the device, which must not happen inside an open render pass.
void App::updateSelectionMesh()
{
    selVerts_.clear();
    const Build& b = doc_.current();
    if (doc_.sceneMode() && b.ok) {
        for (ObjId id : doc_.selection()) {
            const SceneItem* it = doc_.find(id);
            if (!it) continue;
            // dns::sceneObjectBox is the SAME arithmetic buildSceneGeometry
            // uses, so the highlight is the volume the solver tallies -- not an
            // isotropic cube, which was wrong for every lens and mesh.
            const auto bb = dns::sceneObjectBox(b.cas().grid, b.cas().placement,
                                                it->obj);
            const bool flat = b.cas().grid.nz == 1;
            const Vec3 lo{float(bb[0]), float(bb[1]), flat ? 0.0f : float(bb[2])};
            const Vec3 hi{float(bb[3]), float(bb[4]), flat ? 0.0f : float(bb[5])};
            appendBoxLines(selVerts_, lo, hi,
                           id == doc_.primary() ? rgb(0x4fa3ff) : rgb(0x2a5d94));
        }
    }
    (void)ren_->upload(selMesh_, std::span<const VtxL>(selVerts_));
}

// --- loop -------------------------------------------------------------------

int App::run()
{
    bool quit = false;
    while (!quit) {
        handleEvents(quit);

        // Attach to a live segment. Not only to a run WE started: any `dns
        // ... --emit shm` on this machine is visible, so a run launched from a
        // terminal shows up in the viewer too. Rescanned occasionally, not
        // every frame -- it is a directory listing.
        if (!live_.attached() && !liveOptOut_ && (liveTick_++ % 30) == 0) {
            std::string want = liveName_;
            if (want.empty() || live_.attach(want) != "") {
                namespace fs = std::filesystem;
                std::error_code ec;
                std::string best;
                fs::file_time_type bestT{};
                for (const auto& e : fs::directory_iterator("/dev/shm", ec)) {
                    const std::string nm = e.path().filename().string();
                    if (nm.rfind("dns_", 0) != 0) continue;
                    // Skip the corpse of a run killed with SIGKILL: its pid is
                    // gone, and attaching would show a dead run as live.
                    if (LiveField::isStale(nm)) continue;
                    const auto t = e.last_write_time(ec);
                    if (best.empty() || t > bestT) {
                        best = nm;
                        bestT = t;
                    }
                }
                if (!best.empty() && live_.attach(best).empty())
                    status_ = "attached to live run '" + best + "'";
            }
        }
        // A finished run's segment is unlinked; drop the mapping once the run
        // is over so a later run can attach cleanly.
        if (live_.attached() && !live_.running() && !run_.running() &&
            (liveTick_ % 120) == 0)
            liveName_.clear();
        // The volume backs BOTH the fog and an off-centre cut plane, so it is
        // wanted whenever either is in play.
        live_.wantVolume(liveOpts_.showFog ||
                         (liveOpts_.showPlane && live_.hasVolume()));
        // A new frame only redraws what is on screen when the timeline is
        // following the run; while rewound it just extends the history.
        if (live_.poll() && scrub_ < 0) liveDirty_ = true;
        // The fog belongs to the camera octant it was built in.
        checkFogOrder();
        parseSeries();

        if (run_.poll()) {
            status_ = run_.exitCode() == 0
                          ? "run finished (exit 0, " + num(run_.seconds(), 3) + " s)"
                          : "run FAILED: exit " + std::to_string(run_.exitCode());
        }

        if (doc_.poll() || buildViewNow_) {
            buildViewNow_ = false;
            const Build& b = doc_.current();
            if (b.ok && b.cas().grid.nz > 1 && !sliceCentred_) {
                sliceCentred_ = true;
                opts_.sliceIndex = b.cas().grid.nz / 2;
            }
            setProjectionMode();
            buildView(*ren_, b, opts_, meshes_);
            if (firstBuild_ && b.ok) {
                firstBuild_ = false;
                frameCamera(false);
            }
        }

        // One hook brackets EVERY slider, field and segmented control in the
        // application into a single undo entry per gesture. The viewport drag
        // is the one interaction Gui does not own, so it opens its own below.
        // An arrow nudge is a whole gesture in one frame and owns neither
        // active_ nor the editor, so the hook above never sees it. Bracket the
        // frame whenever an arrow is down: if no field took it, the revision
        // does not move and commitEdit drops the empty entry.
        const bool arrowFrame = in_.key(SDLK_LEFT) || in_.key(SDLK_RIGHT);
        if (gui_->interactionStarted() || arrowFrame) doc_.beginEdit("nudge");
        buildUi();
        if (gui_->interactionEnded() || arrowFrame) doc_.endEdit();

        // The viewport is a REGION now, so the camera and picking only respond
        // inside it. A drag that started inside keeps following the pointer
        // even when it leaves -- otherwise an orbit dies the moment it crosses
        // a panel edge, which is the single most irritating way to get docking
        // wrong.
        const bool overVp = lay_.inViewport(in_.mx, in_.my);
        if (!gui_->capturingMouse()) {
            // The gizmo gets first refusal on a press: a click that lands on a
            // handle must never also re-pick whatever is behind it.
            if (in_.lPressed && overVp && !placing_ &&
                gizHot_.part != GizmoPart::None) {
                if (gizmoBeginDrag(gizDrag_, gizFrame_, gizHot_.part, viewportRay()))
                    doc_.beginEdit("move object");
            } else if (in_.lPressed && overVp) {
                if (placing_) {
                    // Click-to-place. The click ALWAYS adds something: if the
                    // current build has no usable case (an empty scene, or a
                    // rejected edit) there is no plane to cast onto, and a
                    // silent no-op would leave no way to add an object back.
                    const Build& pb = doc_.current();
                    std::optional<std::array<double, 3>> seed;
                    if (pb.ok) {
                        const Ray r = viewportRay();
                        const auto& pl = pb.cas().placement;
                        const float zPlane =
                            pb.cas().grid.nz == 1 ? 0.0f : float(pl.cz);
                        if (std::abs(r.d.z) > 1e-6f) {
                            const float t = (zPlane - r.o.z) / r.d.z;
                            if (t > 0) {
                                const Vec3 hit = r.o + r.d * t;
                                seed = std::array<double, 3>{
                                    (double(hit.x) - pl.cx) / pl.D,
                                    (double(hit.y) - pl.cy) / pl.D, 0.0};
                            }
                        }
                    }
                    doc_.addObject(dns::SceneObjectKind(placeKind_), seed);
                    placing_ = false;
                    status_.clear();
                } else {
                    (void)pick();
                }
            }

            if (gizDrag_.active && in_.lDown) {
                const Build& b = doc_.current();
                if (b.ok) {
                    Vec3 want = gizmoSolve(gizDrag_, viewportRay());
                    // Ctrl inverts the snap toggle, the way every DCC does it.
                    SnapSettings sn = snap_;
                    if (in_.ctrl)
                        sn.mode = sn.mode == SnapSettings::Mode::Free
                                      ? SnapSettings::Mode::Grid
                                      : SnapSettings::Mode::Free;
                    const auto& g = b.cas().grid;
                    sn.h[0] = g.hx;
                    sn.h[1] = g.hy;
                    sn.h[2] = g.hz;
                    sn.D = b.cas().placement.D;
                    // An axis the handle constrains is already exact; snapping
                    // it would fight the constraint.
                    bool axisFree[3] = {true, true, true};
                    if (gizDrag_.part == GizmoPart::AxisX) axisFree[1] = axisFree[2] = false;
                    if (gizDrag_.part == GizmoPart::AxisY) axisFree[0] = axisFree[2] = false;
                    if (gizDrag_.part == GizmoPart::AxisZ) axisFree[0] = axisFree[1] = false;
                    want = snapOrigin(sn, gizDrag_.originAtGrab, want, axisFree);

                    const auto& p = b.cas().placement;
                    // Apply as a DELTA from the grab, so every selected object
                    // moves together and the maths stays absolute.
                    const SceneItem* it = doc_.find(doc_.primary());
                    if (it) {
                        const double nx = (double(want.x) - p.cx) / p.D;
                        const double ny = (double(want.y) - p.cy) / p.D;
                        const double nz = (double(want.z) - p.cz) / p.D;
                        const double dx = nx - it->obj.x, dy = ny - it->obj.y;
                        const double dz = b.cas().grid.nz > 1 ? nz - it->obj.z : 0.0;
                        if (dx || dy || dz) doc_.nudgeSelection(dx, dy, dz);
                    }
                    status_ = "moved " + num(double(length(want - gizDrag_.originAtGrab)) /
                                                 (p.D > 0 ? p.D : 1.0), 3) + " D" +
                              (sn.mode == SnapSettings::Mode::Free ? "" : "  (snapped)");
                }
            }
            if (in_.lReleased && gizDrag_.active) {
                gizDrag_ = {};
                doc_.endEdit();   // one undo entry per drag, not per frame
            }

            if (overVp && (in_.lPressed || in_.rPressed || in_.mDown) &&
                !gizDrag_.active)
                camDrag_ = true;
            if (!in_.lDown && !in_.rDown && !in_.mDown) camDrag_ = false;
            if (!gizDrag_.active && (camDrag_ || (overVp && in_.wheel != 0)))
                cameraInput();
        }

        if (!gui_->capturingKeyboard()) {
            if (in_.key(SDLK_F)) frameCamera(false);
            if (in_.key(SDLK_B)) frameCamera(true);
            if (in_.key(SDLK_G)) { opts_.showGrid = !opts_.showGrid; buildViewNow_ = true; }
            if (in_.key(SDLK_C)) { opts_.showCut = !opts_.showCut; buildViewNow_ = true; }
            if (in_.key(SDLK_M)) { opts_.showSolid = !opts_.showSolid; buildViewNow_ = true; }
            if (in_.key(SDLK_A)) { opts_.showApertures = !opts_.showApertures; buildViewNow_ = true; }
            if (in_.key(SDLK_DELETE)) doc_.removeSelection();
            // The timeline, on the keys a video player uses. Stepping off the
            // end is how you get back to following the run, so there is no
            // state to get stuck in.
            // ...unless a drag field under the pointer already took the key.
            if (!gui_->consumedArrows() &&
                (in_.key(SDLK_LEFT) || in_.key(SDLK_RIGHT))) {
                const int n = int(live_.history().size());
                if (n > 1) {
                    const int cur = scrub_ < 0 ? n - 1 : std::min(scrub_, n - 1);
                    const int want = std::clamp(cur + (in_.key(SDLK_RIGHT) ? 1 : -1),
                                                0, n - 1);
                    scrub_ = want >= n - 1 ? -1 : want;
                    liveDirty_ = true;
                }
            }
            if (in_.ctrl && in_.key(SDLK_Z)) {
                if (in_.shift) doc_.redo();
                else doc_.undo();
            }
            if (in_.ctrl && in_.key(SDLK_Y)) doc_.redo();
            // Esc cancels the pending gesture, then clears the selection, and
            // only quits when there is nothing left to cancel: with unsaved
            // scene edits, a bare Esc-quits is a hazard of its own.
            if (in_.key(SDLK_ESCAPE)) {
                if (placing_) { placing_ = false; status_.clear(); }
                else if (doc_.primary() != kNoObj) doc_.clearSelection();
                else quit = true;
            }
        }

        updateSelectionMesh();
        updateLiveMeshes();
        updateGizmo();
        if (auto r = text_->flush(); !r) status_ = r.error();

        const float clear[4] = {0.055f, 0.065f, 0.078f, 1.0f};
        if (vk_->beginFrame(clear)) {
            drawScene();
            (void)ren_->drawUi(gui_->vertices(), gui_->batches());
            vk_->endFrame();
        }
        if (vk_->deviceLost()) {
            std::fprintf(stderr, "dns-ui: the Vulkan device was lost; "
                                 "quitting\n");
            return 1;
        }
    }
    return 0;
}

} // namespace ui
