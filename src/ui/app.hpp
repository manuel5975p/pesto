// The application: window, camera, panels, picking, and the frame loop.
//
// Order within a frame matters and is not arbitrary:
//
//   1. drain SDL events into an Input
//   2. poll the document (publish a finished background build)
//   3. build the whole UI into vertex data
//   4. flush the glyph atlas            <- an upload, must be outside a pass
//   5. open the render pass, draw 3D then 2D, present
//
// Steps 3-4 come before the render pass because rasterizing a glyph uploads a
// texture, and a blocking upload inside an open render pass is illegal.
#pragma once

#include "ui/doc.hpp"
#include "ui/draw.hpp"
#include "ui/gui.hpp"
#include "ui/field.hpp"
#include "ui/gizmo.hpp"
#include "ui/pickobj.hpp"
#include "ui/livemesh.hpp"
#include "ui/runner.hpp"
#include "ui/text.hpp"
#include "ui/view.hpp"
#include "ui/vk.hpp"

#include <memory>
#include <string>

namespace ui {

// Orbit camera. up is +z (the solver's spanwise axis), matching the web
// viewer's camera.up.set(0,0,1) so a screenshot from either tool lines up.
struct Camera
{
    Vec3 target{0, 0, 0};
    float dist = 5;
    float yaw = -0.9f, pitch = 0.5f;
    // `ortho` is the EFFECTIVE projection, derived every frame; `orthoPref` is
    // what the user asked for. 2D is always orthographic, whatever the toggle.
    bool ortho = false;
    bool orthoPref = false;
    float fovY = 0.9f;

    Vec3 forward() const;
    Vec3 eye() const;
    Mat4 viewProj(float aspect) const;
    // World-space ray through a pixel.
    void ray(float px, float py, float w, float h, Vec3& origin, Vec3& dir) const;
    // World units per pixel at the ORBIT TARGET. Correct for panning, which
    // genuinely happens at the target plane.
    float worldPerPixel(float h) const;
    // World units per pixel at the depth of `p` -- which is what a gizmo needs.
    // Sizing handles with the target-plane value makes them the wrong size for
    // any object nearer or further under perspective, so they stop lining up
    // with the cursor.
    float worldPerPixelAt(const Vec3& p, float vpH) const;
    void basis(Vec3& right, Vec3& up) const;
    // The up axis, chosen by DEGENERACY rather than by projection: forward is
    // exactly -z in the pinned 2D view, where {0,0,1} would be parallel. Tying
    // it to `ortho` instead rolled the camera 90 degrees the first time a user
    // toggled orthographic in a 3D scene.
    Vec3 worldUp() const;
    void setStandardView(int which);   // 0 front, 1 side, 2 top, 3 iso
};

// The docked frame. Every region gets a rect from here and nothing floats:
//
//   +------------------------------------------------------+
//   | menu                                                  |
//   +----------+--------------------------------+-----------+
//   |  left    |            viewport            |   right   |
//   |  case /  |                                |  cut /    |
//   | geometry |                                |  layers / |
//   |          |                                |  export   |
//   +----------+--------------------------------+-----------+
//   |                     bottom dock (plots)               |
//   +------------------------------------------------------+
//   | status                                                |
//   +------------------------------------------------------+
//
// The widths and the dock height are draggable and live here rather than in
// Gui, because they are a property of THIS application's frame, not of the
// widget set. The viewport rect is derived, never stored independently: it is
// whatever the four edges leave over, so a resize cannot desynchronise the
// scene from the panels.
struct Layout
{
    // Authored in design pixels, like Theme; scale() multiplies them by the
    // same factor so the chrome grows with the text instead of clipping it.
    float menuH = 28;
    float statusH = 22;
    float leftW = 320;
    float rightW = 320;
    // Content height of the open dock, dragged by "splitD". Authored big
    // enough that four plots and the log tail are legible the moment RUN opens
    // it -- at the old minimum they were 128 px tall and read as smears.
    float dockH = 190;
    float split = 5;              // splitter thickness
    float dockHeadH = 26;         // the collapsed dock is its header
    // The left column is TWO regions, not one scrolling list. Geometry owns a
    // fixed share at the bottom so the add/remove buttons are always on screen:
    // in one tall scrolling column they sit below the fold and are, in
    // practice, invisible.
    float geomH = 300;            // height of the Geometry region
    float s = 1.0f;

    // Derived per frame by compute().
    float vpX = 0, vpY = 0, vpW = 1, vpH = 1;
    float dockY = 0, dockTotalH = 0;
    float caseH = 1, geomY = 0;   // the left column's two rects

    float px(float designPx) const { return designPx * s; }

    void scale(float f)
    {
        s *= f;
        menuH *= f;
        statusH *= f;
        leftW *= f;
        rightW *= f;
        split *= f;
        dockHeadH *= f;
        dockH *= f;
        geomH *= f;
    }

    void compute(float w, float h, bool dockOpen);
    bool inViewport(float x, float y) const
    {
        return x >= vpX && x < vpX + vpW && y >= vpY && y < vpY + vpH;
    }
};

class App
{
public:
    static std::expected<std::unique_ptr<App>, std::string> create();
    ~App();

    int run();
    // Opens a scene .toml before the loop starts (dns-ui --scene FILE).
    std::string openSceneFile(const std::string& path) { return doc_.openScene(path); }

private:
    App() = default;
    void handleEvents(bool& quit);
    void buildUi();
    void leftRegion();      // case: material, flow, grid
    void geometryRegion();  // the object list + inspector
    void rightRegion();     // cut census, layers, export
    void dockRegion();      // the run log + live series
    void liveSection();     // the Live panel in the right column
    void updateLiveMeshes();
    // The frame being shown: null while following the live run.
    const LiveSnapshot* scrubbed() const;
    void checkFogOrder();   // rebuild the fog when the camera changes octant
    void parseSeries();
    void plot(float x, float y, float w, float h, const char* title,
              const std::vector<float>& xs, const std::vector<float>& ys, Rgba);
    void startRun();
    // Where the run happens: the repo root if we can find build/dns from the
    // cwd, else the cwd. Reported in the dock so it is never a mystery.
    std::string runDir() const;
    void drawScene();
    void updateSelectionMesh();
    void cameraInput();
    void setProjectionMode();
    bool pick();   // true when the click actually hit an object
    void updateGizmo();          // rebuild the frame + geometry each frame
    Ray viewportRay() const;     // the ray under the cursor, viewport-relative
    // The gizmo's origin: the primary selection's centre in world coordinates.
    bool gizmoOrigin(Vec3& out) const;
    void frameCamera(bool onBody);

    SDL_Window* window_ = nullptr;
    std::unique_ptr<Vk> vk_;
    std::unique_ptr<Renderer> ren_;
    std::unique_ptr<TextEngine> text_;
    std::unique_ptr<Gui> gui_;
    Doc doc_;
    ViewOptions opts_;
    ViewMeshes meshes_;
    Camera cam_;
    Input in_;

    GpuMesh selMesh_;               // live selection box
    GpuMesh gizMesh_;               // the translate gizmo, drawn as an overlay
    std::vector<Vtx3> gizVerts_;
    GizmoFrame gizFrame_;
    GizmoStyle gizStyle_;
    GizmoHit gizHot_;
    GizmoDrag gizDrag_;
    SnapSettings snap_;
    int snapMode_ = 0;
    int standardView_ = 3;   // iso
    float lastClickX_ = -1e9f, lastClickY_ = -1e9f;
    std::vector<VtxL> selVerts_;    // built and uploaded outside the pass
    float w_ = 1600, h_ = 950;
    Layout lay_;
    bool dockOpen_ = false;
    bool firstBuild_ = true;
    bool sliceCentred_ = false;
    bool buildViewNow_ = false;
    bool dragObject_ = false;
    bool camDrag_ = false;   // a camera drag that STARTED in the viewport
    std::string status_;
    bool showDescribe_ = false;
    bool showMaterial_ = true, showFlow_ = true, showGridPanel_ = true;
    bool showThermal_ = false;   // compressible flow: mu(T), walls, T frames
    bool showConv_ = true;       // the convection case's own section
    bool showObjects_ = true, showNumerics_ = false;
    bool showCutPanel_ = true, showLayers_ = true, showExport_ = false;
    // Click-to-place: +cyl arms a kind, the next viewport click positions it.
    bool placing_ = false;
    int placeKind_ = 0;
    bool showArray_ = false;
    int arrayCount_ = 3;
    double arrayDx_ = 0, arrayDy_ = 1.5, arrayDz_ = 0;
    std::string scenePath_ = "scene.toml";
    Runner run_;
    // The live bridge: the run publishes into shared memory and this maps it.
    LiveField live_;
    LiveMeshOptions liveOpts_;
    LiveMeshStats livePlaneStats_, liveFogStats_;
    GpuMesh liveMesh_;
    std::vector<Vtx3> liveVerts_;
    bool liveDirty_ = false;
    // Two DIFFERENT things, and conflating them meant collapsing the panel
    // silently stopped the live field from rendering -- which reads exactly
    // like "the data stopped updating".
    bool showLiveSection_ = true;   // is the panel expanded
    bool showLive_ = true;          // is the live field drawn
    std::string liveName_;
    // The timeline. -1 FOLLOWS the run (the newest frame); anything else is an
    // index into LiveField::history(), which is what rewinding means here.
    int scrub_ = -1;
    unsigned liveTick_ = 0;
    bool liveOptOut_ = false;   // an explicit detach; must survive the scan
    // Series parsed out of the run's own stdout, for the plots in the dock.
    // Both cases' progress series. `dns flow` prints cd/cl/K/maxDiv on its
    // stdout line and `dns conv` prints Nu/urms, so the dock parses whichever
    // is there and plots the four that mean something for the case in flight.
    struct Series
    {
        std::vector<float> t, cd, cl, k, div;
        std::vector<float> nuBot, nuTop, nuFlux, urms;
    };
    Series series_;
    std::size_t seriesConsumed_ = 0;
    bool runFollow_ = true;   // stick to the tail of the log
    std::string runCmd_;      // what was actually executed, for the dock header
};

} // namespace ui
