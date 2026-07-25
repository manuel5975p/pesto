#include "ui/app.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

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
const char* const kSgsNames[] = {"smag", "sigma"};
const char* const kAxisNames[] = {"x", "y", "z"};

} // namespace

// --- camera -----------------------------------------------------------------

Vec3 Camera::forward() const
{
    const float cp = std::cos(pitch);
    return normalize({-cp * std::cos(yaw), -cp * std::sin(yaw), -std::sin(pitch)});
}

Vec3 Camera::eye() const { return target - forward() * dist; }

void Camera::basis(Vec3& right, Vec3& up) const
{
    const Vec3 f = forward();
    const Vec3 worldUp = ortho ? Vec3{0, 1, 0} : Vec3{0, 0, 1};
    right = normalize(cross(f, worldUp));
    up = cross(right, f);
}

Mat4 Camera::viewProj(float aspect) const
{
    const Vec3 e = eye();
    const Vec3 worldUp = ortho ? Vec3{0, 1, 0} : Vec3{0, 0, 1};
    const Mat4 v = lookAt(e, target, worldUp);
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
    const float wpp = cam_.worldPerPixel(h_);

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
    cam_.ortho = flat;
    if (flat) {
        cam_.yaw = 0.0f;
        cam_.pitch = 1.5707963f;
    }
}

// Ray/AABB against every object box, nearest wins. Scene mode only: with a
// single --body there is nothing to select.
void App::pick()
{
    const Build& b = doc_.current();
    if (!doc_.sceneMode() || !b.ok) return;
    Vec3 o, d;
    cam_.ray(in_.mx, in_.my, w_, h_, o, d);

    int best = -1;
    float bestT = 1e30f;
    const auto& boxes = b.geo.objectBoxes;
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& bb = boxes[i].second;
        float t0 = -1e30f, t1 = 1e30f;
        const float lo[3] = {float(bb[0]), float(bb[1]), float(bb[2])};
        const float hi[3] = {float(bb[3]), float(bb[4]), float(bb[5])};
        const float od[3] = {o.x, o.y, o.z};
        const float dd[3] = {d.x, d.y, d.z};
        bool miss = false;
        for (int a = 0; a < 3; ++a) {
            if (std::abs(dd[a]) < 1e-12f) {
                if (od[a] < lo[a] || od[a] > hi[a]) miss = true;
                continue;
            }
            float ta = (lo[a] - od[a]) / dd[a], tb = (hi[a] - od[a]) / dd[a];
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
        }
        if (miss || t1 < std::max(t0, 0.0f)) continue;
        const float t = t0 > 0 ? t0 : t1;
        if (t < bestT) {
            bestT = t;
            best = int(i);
        }
    }
    doc_.selected = best;
}

// --- UI ---------------------------------------------------------------------

void App::buildUi()
{
    Gui& g = *gui_;
    g.begin(in_, w_, h_);
    const Build& b = doc_.current();
    const Theme& th = g.theme;

    const float colW = 312;
    const float leftX = 12, rightX = w_ - colW - 12;
    float ly = 12, ry = 12;

    // --- status / error banner ---------------------------------------------
    {
        const float bh = 26;
        const float bx = leftX + colW + 16, bw = rightX - bx - 16;
        if (bw > 200) {
            if (!b.ok && !b.error.empty()) {
                g.rect(bx, 12, bw, bh, {60, 24, 26, 235}, 4);
                g.rectOutline(bx, 12, bw, bh, th.bad, 1);
                const std::string msg =
                    "exit " + std::to_string(b.errorCode) + ":  " + b.error;
                g.textAt(msg, bx + 10, 12 + 18, th.bad);
            } else {
                g.rect(bx, 12, bw, bh, {20, 26, 33, 200}, 4);
                std::string msg = b.label;
                if (b.ok)
                    msg += doc_.building() ? "   |   rebuilding cut..."
                                           : "   |   cut built in " +
                                                 num(b.buildSeconds, 2) + " s";
                if (!status_.empty()) msg += "   |   " + status_;
                g.textAt(msg, bx + 10, 12 + 18,
                         b.bodyResolved || b.cas().lesLabel ? th.good : th.warn);
            }
        }
    }

    // --- case ---------------------------------------------------------------
    g.beginPanel("Case", leftX, ly, colW);
    if (g.sectionHeader("physics", showCase_)) {
        if (g.choice("model", doc_.st.physics, {kPhysicsNames, 2})) doc_.touch();
        if (g.dragDouble("Re", doc_.st.re, 2.0, 0.1, 1e9, 1)) doc_.touch();
        if (g.dragDouble("--ft (run length)", doc_.st.ft, 0.02, 1e-6, 1e6, 3))
            doc_.touch();
        if (doc_.st.physics == 1)
            if (g.dragDouble("Mach", doc_.st.mach, 0.002, 0.01, 0.99, 3))
                doc_.touch();
        if (g.choice("far field", doc_.st.far, {farNames(), size_t(farNameCount())}))
            doc_.touch();
        if (doc_.st.far == 2)
            g.label("sponge eats the near field - explicit fallback only", th.warn);
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
    g.endPanel();
    ly += g.panelHeight("Case") + th.rowH + 4 + th.pad + 12;

    // --- grid ---------------------------------------------------------------
    g.beginPanel("Grid", leftX, ly, colW);
    if (g.sectionHeader("resolution", showGridPanel_)) {
        if (g.dragInt("--n", doc_.st.n, 0.5, 8, 4096)) doc_.touch();
        if (g.dragInt("--ny (0 = follow n)", doc_.st.ny, 0.5, 0, 4096)) doc_.touch();
        if (g.dragInt("--nz (1 = 2D)", doc_.st.nz, 0.25, 1, 2048)) doc_.touch();
        if (g.dragDouble("--dr (box in D)", doc_.st.dr, 0.02, 0.5, 128, 2))
            doc_.touch();
        if (doc_.st.physics == 1) {
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
            g.field("grid", std::to_string(gr.nx) + " x " + std::to_string(gr.ny) +
                                " x " + std::to_string(gr.nz));
            g.field("cells", bigInt(gr.size()));
            g.field("h (x,y,z)",
                    num(gr.hx, 3) + ", " + num(gr.hy, 3) + ", " + num(gr.hz, 3));
            if (b.cas().focalActive)
                g.field("h fine", num(b.cas().focalHFine[0], 3) + ", " +
                                      num(b.cas().focalHFine[1], 3) + ", " +
                                      num(b.cas().focalHFine[2], 3));
            g.field("D", num(b.geo.D > 0 ? b.geo.D : b.cas().placement.D, 4));
            g.field("D / h", num(b.dOverH, 4),
                    b.dOverH >= 16 ? th.good : (b.dOverH >= 8 ? th.warn : th.bad));
            // The DNS criterion from CLAUDE.md: the body BL is ~D/sqrt(Re) and
            // needs 3 cells across it. This row is the whole reason the tool
            // shows a grid at all.
            g.field("BL cells (D/h)/sqrt(Re)", num(b.blCells, 3),
                    b.bodyResolved ? th.good : th.bad);
            if (!b.bodyResolved && !b.cas().lesLabel)
                g.label("under-resolved: expect trapped ripples", th.warn);
            g.field("label", b.label, b.cas().lesLabel ? th.warn : th.good);
            if (b.cas().focalActive)
                g.field("dt penalty", num(b.cas().focalDtFactor, 3));
        }
    }
    g.endPanel();
    ly += g.panelHeight("Grid") + th.rowH + 4 + th.pad + 12;

    // --- body / objects -----------------------------------------------------
    g.beginPanel("Obstacles", leftX, ly, colW, h_ - ly - 12);
    if (g.sectionHeader("body", showBody_)) {
        int bk = doc_.st.bodyKind;
        if (g.choice("", bk, {bodyNames(), 5})) {
            doc_.st.bodyKind = bk;
            doc_.touch();
        }
        int bk2 = doc_.st.bodyKind >= 5 ? doc_.st.bodyKind - 5 : -1;
        if (g.choice("", bk2, {bodyNames() + 5, 4})) {
            doc_.st.bodyKind = bk2 + 5;
            doc_.touch();
        }
        if (doc_.st.bodyKind == 4)
            if (g.dragDouble("aspect", doc_.st.aspect, 0.01, 1.05, 20, 2))
                doc_.touch();
        if (doc_.st.bodyKind == 1 || doc_.st.bodyKind == 2)
            if (g.dragInt("dimples", doc_.st.dimples, 0.2, 0, 4096)) doc_.touch();

        if (doc_.sceneMode()) {
            g.separator();
            auto& objs = doc_.st.scene.objects;
            for (size_t i = 0; i < objs.size(); ++i) {
                const std::string nm =
                    std::to_string(i) + "  " + dns::name(objs[i].kind) +
                    (objs[i].label.empty() ? "" : "  (" + objs[i].label + ")");
                if (g.selectable(nm, int(i) == doc_.selected))
                    doc_.selected = int(i);
            }
            g.spacing(2);
            static const dns::SceneObjectKind kinds[4] = {
                dns::SceneObjectKind::Cylinder, dns::SceneObjectKind::Sphere,
                dns::SceneObjectKind::Cube, dns::SceneObjectKind::Lens};
            static const char* const labels[4] = {"+cyl", "+sph", "+cube", "+lens"};
            if (const int k = g.buttonRow({labels, 4}); k >= 0)
                doc_.selected = doc_.addObject(kinds[k]);
            if (doc_.selected >= 0 && doc_.selected < int(objs.size())) {
                auto& o = objs[size_t(doc_.selected)];
                g.separator();
                int k = int(o.kind);
                if (g.choice("kind", k,
                             {objectKindNames(), size_t(objectKindCount())})) {
                    o.kind = dns::SceneObjectKind(k);
                    doc_.touch();
                }
                if (g.dragDouble("x  (D)", o.x, 0.004, -1e4, 1e4, 3)) doc_.touch();
                if (g.dragDouble("y  (D)", o.y, 0.004, -1e4, 1e4, 3)) doc_.touch();
                if (g.dragDouble("z  (D)", o.z, 0.004, -1e4, 1e4, 3)) doc_.touch();
                if (g.dragDouble("d  (D)", o.d, 0.004, 1e-3, 1e3, 3)) doc_.touch();
                if (o.kind == dns::SceneObjectKind::Lens)
                    if (g.dragDouble("aspect", o.aspect, 0.01, 1.05, 20, 2))
                        doc_.touch();
                if (g.button("remove object")) {
                    doc_.removeObject(doc_.selected);
                }
            }
        }
    }
    g.endPanel();

    // --- cut cells ----------------------------------------------------------
    g.beginPanel("Cut cells", rightX, ry, colW);
    if (g.sectionHeader("geometry", showCutPanel_)) {
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
        g.separator();
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
    g.endPanel();
    ry += g.panelHeight("Cut cells") + th.rowH + 4 + th.pad + 12;

    // --- export -------------------------------------------------------------
    g.beginPanel("Export", rightX, ry, colW, h_ - ry - 12);
    if (g.sectionHeader("command / scene", showExport_)) {
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
            if (g.button("write scene.toml")) {
                const std::string err = doc_.writeScene("scene.toml");
                status_ = err.empty() ? "wrote scene.toml" : err;
            }
        }
        if (g.button(showDescribe_ ? "hide resolved case" : "show resolved case"))
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

    // --- viewport hints -----------------------------------------------------
    {
        const char* hint =
            "LMB orbit/pan  |  RMB or shift+LMB pan  |  wheel zoom  |  "
            "F frame  |  B frame body  |  click an object to select, drag to move";
        const float tw = g.textWidth(hint, FontId::Sans, th.smallPx);
        g.textAt(hint, (w_ - tw) * 0.5f, h_ - 12, th.textDim, FontId::Sans,
                 th.smallPx);
    }

    g.end();
}

// --- 3D ---------------------------------------------------------------------

void App::drawScene()
{
    const Build& b = doc_.current();
    const float aspect = w_ / std::max(1.0f, h_);
    const Mat4 mvp = cam_.viewProj(aspect);
    const Vec3 light = normalize({0.4f, 0.7f, 0.6f});

    ren_->drawTriangles(meshes_.surface, mvp, light, 1.0f, true);
    ren_->drawLines(meshes_.lines, mvp, 1.0f);
    ren_->drawTriangles(meshes_.cells, mvp, light, 1.0f, false);

    if (selMesh_.count) ren_->drawLines(selMesh_, mvp, 1.0f);
    (void)b;
}

// The selection box is computed from the EDITED offsets, not from the last
// build, so dragging an object is responsive even while the cut rebuilds in
// the background. Uploaded here rather than in drawScene(): growing a buffer
// waits for the device, which must not happen inside an open render pass.
void App::updateSelectionMesh()
{
    selVerts_.clear();
    const Build& b = doc_.current();
    if (doc_.sceneMode() && b.ok && doc_.selected >= 0 &&
        doc_.selected < int(doc_.st.scene.objects.size())) {
        const auto& o = doc_.st.scene.objects[size_t(doc_.selected)];
        const auto& p = b.cas().placement;
        const float r = float(o.d * p.D * 0.5);
        const bool flat = b.cas().grid.nz == 1;
        const Vec3 c{float(p.cx + o.x * p.D), float(p.cy + o.y * p.D),
                     float(flat ? 0.0 : p.cz + o.z * p.D)};
        const Vec3 half{r, r, flat ? 0.0f : r};
        appendBoxLines(selVerts_, c - half, c + half, rgb(0x4fa3ff));
    }
    (void)ren_->upload(selMesh_, std::span<const VtxL>(selVerts_));
}

// --- loop -------------------------------------------------------------------

int App::run()
{
    bool quit = false;
    while (!quit) {
        handleEvents(quit);

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

        buildUi();

        if (!gui_->capturingMouse()) {
            if (in_.lPressed) {
                pick();
                dragObject_ = doc_.selected >= 0;
            }
            if (in_.lReleased) dragObject_ = false;

            if (dragObject_ && in_.lDown && (in_.dx || in_.dy) && !in_.shift) {
                const Build& b = doc_.current();
                if (b.ok && doc_.selected < int(doc_.st.scene.objects.size())) {
                    Vec3 right, up;
                    cam_.basis(right, up);
                    const float wpp = cam_.worldPerPixel(h_);
                    const Vec3 dw = right * (in_.dx * wpp) - up * (in_.dy * wpp);
                    auto& o = doc_.st.scene.objects[size_t(doc_.selected)];
                    const double D = b.cas().placement.D;
                    o.x += dw.x / D;
                    o.y += dw.y / D;
                    if (b.cas().grid.nz > 1) o.z += dw.z / D;
                    doc_.touch();
                }
            } else {
                cameraInput();
            }
        }

        if (!gui_->capturingKeyboard()) {
            if (in_.key(SDLK_F)) frameCamera(false);
            if (in_.key(SDLK_B)) frameCamera(true);
            if (in_.key(SDLK_G)) { opts_.showGrid = !opts_.showGrid; buildViewNow_ = true; }
            if (in_.key(SDLK_C)) { opts_.showCut = !opts_.showCut; buildViewNow_ = true; }
            if (in_.key(SDLK_M)) { opts_.showSolid = !opts_.showSolid; buildViewNow_ = true; }
            if (in_.key(SDLK_A)) { opts_.showApertures = !opts_.showApertures; buildViewNow_ = true; }
            if (in_.key(SDLK_DELETE) && doc_.selected >= 0)
                doc_.removeObject(doc_.selected);
            if (in_.key(SDLK_ESCAPE)) quit = true;
        }

        updateSelectionMesh();
        if (auto r = text_->flush(); !r) status_ = r.error();

        const float clear[4] = {0.055f, 0.065f, 0.078f, 1.0f};
        if (vk_->beginFrame(clear)) {
            drawScene();
            (void)ren_->drawUi(gui_->vertices(), gui_->batches());
            vk_->endFrame();
        }
    }
    return 0;
}

} // namespace ui
