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
    bool ortho = false;
    float fovY = 0.9f;

    Vec3 forward() const;
    Vec3 eye() const;
    Mat4 viewProj(float aspect) const;
    // World-space ray through a pixel.
    void ray(float px, float py, float w, float h, Vec3& origin, Vec3& dir) const;
    // World units per pixel at the target plane.
    float worldPerPixel(float h) const;
    void basis(Vec3& right, Vec3& up) const;
};

class App
{
public:
    static std::expected<std::unique_ptr<App>, std::string> create();
    ~App();

    int run();

private:
    App() = default;
    void handleEvents(bool& quit);
    void buildUi();
    void drawScene();
    void updateSelectionMesh();
    void cameraInput();
    void setProjectionMode();
    void pick();
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
    std::vector<VtxL> selVerts_;    // built and uploaded outside the pass
    float w_ = 1600, h_ = 950;
    bool firstBuild_ = true;
    bool sliceCentred_ = false;
    bool buildViewNow_ = false;
    bool dragObject_ = false;
    std::string status_;
    bool showDescribe_ = false;
    bool showCase_ = true, showGridPanel_ = true, showBody_ = true;
    bool showCutPanel_ = true, showExport_ = false;
};

} // namespace ui
