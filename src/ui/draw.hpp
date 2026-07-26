// Three pipelines and the buffers that feed them. Everything drawn in this
// application is one of:
//
//   Vtx3  lit triangles  -- obstacle surfaces, cut-cell boxes, slice quads
//   VtxL  unlit lines    -- the mesh, the domain box, axes, gizmo handles
//   Vtx2  2D pixels      -- panels and HarfBuzz glyph quads
//
// The 3D streams are STATIC meshes: the grid for a 512^3 case is millions of
// vertices and it changes only when the case does, so it is uploaded on edit,
// not per frame. The 2D stream is regenerated every frame (it is an
// immediate-mode UI) into a per-frame-in-flight ring so a frame still being
// presented never has its vertices overwritten.
#pragma once

#include "ui/math.hpp"
#include "ui/vk.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ui {

struct Rgba
{
    uint8_t r = 255, g = 255, b = 255, a = 255;
};

inline Rgba rgb(uint32_t hex, uint8_t a = 255)
{
    return {uint8_t(hex >> 16), uint8_t(hex >> 8), uint8_t(hex), a};
}

struct Vtx3
{
    float p[3];
    float n[3];
    Rgba c;
};

struct VtxL
{
    float p[3];
    Rgba c;
};

struct Vtx2
{
    float p[2];
    float uv[2];   // negative u = untextured
    Rgba c;
};

// A vertex buffer plus its count. Owned by the caller; released through
// Renderer::destroy().
struct GpuMesh
{
    Buffer buf;
    uint32_t count = 0;
};

// One scissored run of the 2D vertex stream.
struct UiBatch
{
    uint32_t first = 0, count = 0;
    VkRect2D scissor{};
};

class Renderer
{
public:
    static std::expected<std::unique_ptr<Renderer>, std::string> create(Vk&);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // --- static 3D meshes ---------------------------------------------------
    std::expected<void, std::string> upload(GpuMesh&, std::span<const Vtx3>);
    std::expected<void, std::string> upload(GpuMesh&, std::span<const VtxL>);
    void destroy(GpuMesh&);

    // depthWrite == false is the translucent pass (cut-cell volumes, slice
    // planes): they still depth-TEST against the opaque surfaces, they just do
    // not occlude each other, which for a stack of aligned boxes reads far
    // better than an arbitrary winner.
    // tint multiplies the vertex colour (its alpha scales the alpha), which is
    // what Push3 has always carried and both 3D shaders have always applied.
    //
    // depthTest == false is the OVERLAY: the gizmo and the selection outline
    // must be visible THROUGH the body. It also matters that they do not WRITE
    // depth -- the line pipeline does, which both occluded the old selection
    // box and punched it into the depth buffer for everything drawn after.
    void drawTriangles(const GpuMesh&, const Mat4& mvp, const Vec3& lightDir,
                       Rgba tint, bool depthTest, bool depthWrite);
    void drawLines(const GpuMesh&, const Mat4& mvp, Rgba tint, bool depthTest);

    // The pre-tint spellings, so existing call sites read unchanged.
    void drawTriangles(const GpuMesh& m, const Mat4& v, const Vec3& l, float a,
                       bool w)
    {
        drawTriangles(m, v, l, Rgba{255, 255, 255, uint8_t(a * 255.0f)}, true, w);
    }
    void drawLines(const GpuMesh& m, const Mat4& v, float a)
    {
        drawLines(m, v, Rgba{255, 255, 255, uint8_t(a * 255.0f)}, true);
    }

    // Restricts subsequent 3D draws to a sub-rect of the framebuffer, in
    // pixels. This is what makes the viewport a REGION rather than the whole
    // window with panels floating on top: the scene is projected into the rect
    // and clipped to it, so a docked panel can never be drawn over by 3D
    // geometry. drawUi() restores the full-window viewport itself, since the
    // 2D stream is laid out in window pixels and carries its own scissors.
    void setViewport(float x, float y, float w, float h);
    void resetViewport();

    // --- per-frame 2D -------------------------------------------------------
    // Binds the glyph atlas for subsequent 2D draws. Must be called once
    // before drawUi() in a frame that draws text.
    std::expected<void, std::string> setGlyphAtlas(const Texture&);
    std::expected<void, std::string> drawUi(std::span<const Vtx2>,
                                            std::span<const UiBatch>);

private:
    Renderer() = default;
    std::expected<void, std::string> build(Vk&);

    Vk* vk_ = nullptr;
    VkPipelineLayout layout3_ = VK_NULL_HANDLE;
    VkPipelineLayout layout2_ = VK_NULL_HANDLE;
    VkPipeline pipeTriOpaque_ = VK_NULL_HANDLE;
    VkPipeline pipeTriBlend_ = VK_NULL_HANDLE;
    VkPipeline pipeLine_ = VK_NULL_HANDLE;
    VkPipeline pipeTriOverlay_ = VK_NULL_HANDLE;
    VkPipeline pipeLineOverlay_ = VK_NULL_HANDLE;
    VkPipeline pipeUi_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
    VkDescriptorPool dpool_ = VK_NULL_HANDLE;
    VkDescriptorSet dset_ = VK_NULL_HANDLE;
    Buffer ui_[Vk::framesInFlight];   // pre-sized; see Renderer::build
};

} // namespace ui
