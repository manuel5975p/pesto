#include "ui/draw.hpp"

#include "ui/vkcheck.hpp"

#include <algorithm>
#include <cstring>

namespace {

// Compiled by CMake from src/ui/shaders/*, so a shader error is a BUILD error.
const uint32_t scene3d_vert[] =
#include "scene3d_vert.spv.h"
    ;
const uint32_t scene3d_frag[] =
#include "scene3d_frag.spv.h"
    ;
const uint32_t line3d_vert[] =
#include "line3d_vert.spv.h"
    ;
const uint32_t line3d_frag[] =
#include "line3d_frag.spv.h"
    ;
const uint32_t ui2d_vert[] =
#include "ui2d_vert.spv.h"
    ;
const uint32_t ui2d_frag[] =
#include "ui2d_frag.spv.h"
    ;

} // namespace

namespace ui {
namespace {

// The 96 bytes both 3D pipelines push: mvp, light direction, tint.
struct Push3
{
    Mat4 mvp;
    float light[4];
    float tint[4];
};

std::expected<VkShaderModule, std::string> makeModule(VkDevice dev,
                                                      const uint32_t* code,
                                                      size_t bytes)
{
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = bytes;
    ci.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    VKCHECK(vkCreateShaderModule(dev, &ci, nullptr, &m), "vkCreateShaderModule");
    return m;
}

} // namespace

std::expected<std::unique_ptr<Renderer>, std::string> Renderer::create(Vk& vk)
{
    auto r = std::unique_ptr<Renderer>(new Renderer());
    r->vk_ = &vk;
    if (auto b = r->build(vk); !b) return std::unexpected(b.error());
    return r;
}

std::expected<void, std::string> Renderer::build(Vk& vk)
{
    const VkDevice dev = vk.dev();

    // --- layouts ------------------------------------------------------------
    VkPushConstantRange pc3{};
    pc3.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pc3.size = sizeof(Push3);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc3;
    VKCHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &layout3_),
            "vkCreatePipelineLayout(3d)");

    VkDescriptorSetLayoutBinding bind{};
    bind.binding = 0;
    bind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bind.descriptorCount = 1;
    bind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 1;
    dlci.pBindings = &bind;
    VKCHECK(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &dsl_),
            "vkCreateDescriptorSetLayout");

    VkPushConstantRange pc2{};
    pc2.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pc2.size = sizeof(float) * 2;
    VkPipelineLayoutCreateInfo plci2{};
    plci2.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci2.setLayoutCount = 1;
    plci2.pSetLayouts = &dsl_;
    plci2.pushConstantRangeCount = 1;
    plci2.pPushConstantRanges = &pc2;
    VKCHECK(vkCreatePipelineLayout(dev, &plci2, nullptr, &layout2_),
            "vkCreatePipelineLayout(2d)");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VKCHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool_),
            "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = dpool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl_;
    VKCHECK(vkAllocateDescriptorSets(dev, &dsai, &dset_),
            "vkAllocateDescriptorSets");

    // --- shared pipeline state ---------------------------------------------
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = vp.scissorCount = 1;

    const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = dyn;

    VkPipelineMultisampleStateCreateInfo msaa{};
    msaa.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msaa.rasterizationSamples = vk.samples();

    // Straight alpha over the destination everywhere; the whole UI and the
    // translucent cut cells are authored in straight alpha.
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &blend;

    auto rasterizer = [](VkPolygonMode mode, VkCullModeFlags cull) {
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = mode;
        rs.cullMode = cull;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        return rs;
    };
    auto depth = [](bool test, bool write) {
        VkPipelineDepthStencilStateCreateInfo dsi{};
        dsi.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dsi.depthTestEnable = test;
        dsi.depthWriteEnable = write;
        dsi.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        dsi.maxDepthBounds = 1.0f;
        return dsi;
    };

    struct Modules { VkShaderModule vs, fs; };
    auto load = [&](const uint32_t* v, size_t vb, const uint32_t* f,
                    size_t fb) -> std::expected<Modules, std::string> {
        auto vs = makeModule(dev, v, vb);
        if (!vs) return std::unexpected(vs.error());
        auto fs = makeModule(dev, f, fb);
        if (!fs) return std::unexpected(fs.error());
        return Modules{*vs, *fs};
    };

    auto sceneM = load(scene3d_vert, sizeof(scene3d_vert), scene3d_frag,
                       sizeof(scene3d_frag));
    if (!sceneM) return std::unexpected(sceneM.error());
    auto lineM = load(line3d_vert, sizeof(line3d_vert), line3d_frag,
                      sizeof(line3d_frag));
    if (!lineM) return std::unexpected(lineM.error());
    auto uiM = load(ui2d_vert, sizeof(ui2d_vert), ui2d_frag, sizeof(ui2d_frag));
    if (!uiM) return std::unexpected(uiM.error());

    auto stages = [](const Modules& m, VkPipelineShaderStageCreateInfo out[2]) {
        out[0] = {};
        out[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        out[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        out[0].module = m.vs;
        out[0].pName = "main";
        out[1] = out[0];
        out[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        out[1].module = m.fs;
    };

    // --- vertex formats -----------------------------------------------------
    VkVertexInputBindingDescription bind3{0, sizeof(Vtx3), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription att3[3] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vtx3, p)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vtx3, n)},
        {2, 0, VK_FORMAT_R8G8B8A8_UINT, offsetof(Vtx3, c)},
    };
    VkPipelineVertexInputStateCreateInfo vi3{};
    vi3.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi3.vertexBindingDescriptionCount = 1;
    vi3.pVertexBindingDescriptions = &bind3;
    vi3.vertexAttributeDescriptionCount = 3;
    vi3.pVertexAttributeDescriptions = att3;

    VkVertexInputBindingDescription bindL{0, sizeof(VtxL), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attL[2] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VtxL, p)},
        {1, 0, VK_FORMAT_R8G8B8A8_UINT, offsetof(VtxL, c)},
    };
    VkPipelineVertexInputStateCreateInfo viL{};
    viL.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    viL.vertexBindingDescriptionCount = 1;
    viL.pVertexBindingDescriptions = &bindL;
    viL.vertexAttributeDescriptionCount = 2;
    viL.pVertexAttributeDescriptions = attL;

    VkVertexInputBindingDescription bind2{0, sizeof(Vtx2), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription att2[3] = {
        {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vtx2, p)},
        {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vtx2, uv)},
        {2, 0, VK_FORMAT_R8G8B8A8_UINT, offsetof(Vtx2, c)},
    };
    VkPipelineVertexInputStateCreateInfo vi2{};
    vi2.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi2.vertexBindingDescriptionCount = 1;
    vi2.pVertexBindingDescriptions = &bind2;
    vi2.vertexAttributeDescriptionCount = 3;
    vi2.pVertexAttributeDescriptions = att2;

    auto topology = [](VkPrimitiveTopology t) {
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = t;
        return ia;
    };

    VkPipelineShaderStageCreateInfo st[2];
    VkGraphicsPipelineCreateInfo gp{};
    gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.stageCount = 2;
    gp.pStages = st;
    gp.pViewportState = &vp;
    gp.pMultisampleState = &msaa;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &ds;
    gp.renderPass = vk.renderPass();

    // Opaque triangles. No back-face culling: cut-cell boxes and slice quads
    // are inspected from both sides, and a viewer that hides a wall when you
    // orbit past it is worse than useless for judging a cut.
    auto rsFill = rasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE);
    auto dsOn = depth(true, true);
    auto dsRead = depth(true, false);
    auto dsOff = depth(false, false);
    auto iaTri = topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    auto iaLine = topology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);

    stages(*sceneM, st);
    gp.pVertexInputState = &vi3;
    gp.pInputAssemblyState = &iaTri;
    gp.pRasterizationState = &rsFill;
    gp.pDepthStencilState = &dsOn;
    gp.layout = layout3_;
    VKCHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr,
                                      &pipeTriOpaque_),
            "vkCreateGraphicsPipelines(tri opaque)");
    gp.pDepthStencilState = &dsRead;
    VKCHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr,
                                      &pipeTriBlend_),
            "vkCreateGraphicsPipelines(tri blend)");

    // Overlay triangles: depth OFF. A gizmo handle you cannot see because the
    // body is in front of it is a handle you cannot grab.
    gp.pDepthStencilState = &dsOff;
    VKCHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr,
                                      &pipeTriOverlay_),
            "vkCreateGraphicsPipelines(tri overlay)");

    stages(*lineM, st);
    gp.pVertexInputState = &viL;
    gp.pInputAssemblyState = &iaLine;
    gp.pDepthStencilState = &dsOn;
    VKCHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeLine_),
            "vkCreateGraphicsPipelines(line)");
    gp.pDepthStencilState = &dsOff;
    VKCHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr,
                                      &pipeLineOverlay_),
            "vkCreateGraphicsPipelines(line overlay)");

    stages(*uiM, st);
    gp.pVertexInputState = &vi2;
    gp.pInputAssemblyState = &iaTri;
    gp.pDepthStencilState = &dsOff;
    gp.layout = layout2_;
    VKCHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeUi_),
            "vkCreateGraphicsPipelines(ui)");

    vkDestroyShaderModule(dev, sceneM->vs, nullptr);
    vkDestroyShaderModule(dev, sceneM->fs, nullptr);
    vkDestroyShaderModule(dev, lineM->vs, nullptr);
    vkDestroyShaderModule(dev, lineM->fs, nullptr);
    vkDestroyShaderModule(dev, uiM->vs, nullptr);
    vkDestroyShaderModule(dev, uiM->fs, nullptr);

    // Pre-size the per-frame UI streams generously. Growing one costs a
    // vkDeviceWaitIdle, and drawUi() runs INSIDE the render pass -- legal, but
    // a stall nobody should pay for a panel that got one row longer.
    for (auto& b : ui_) {
        auto nb = vk.createHostBuffer(1u << 20, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        if (!nb) return std::unexpected(nb.error());
        b = *nb;
    }
    return {};
}

namespace {

template <class V>
std::expected<void, std::string> uploadVerts(Vk& vk, GpuMesh& m,
                                             std::span<const V> verts)
{
    m.count = uint32_t(verts.size());
    if (verts.empty()) return {};
    const VkDeviceSize bytes = verts.size_bytes();
    if (auto r = vk.ensureCapacity(m.buf, bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT); !r)
        return r;
    std::memcpy(m.buf.mapped, verts.data(), size_t(bytes));
    return {};
}

} // namespace

std::expected<void, std::string> Renderer::upload(GpuMesh& m,
                                                  std::span<const Vtx3> v)
{
    return uploadVerts(*vk_, m, v);
}

std::expected<void, std::string> Renderer::upload(GpuMesh& m,
                                                  std::span<const VtxL> v)
{
    return uploadVerts(*vk_, m, v);
}

void Renderer::destroy(GpuMesh& m)
{
    vk_->destroy(m.buf);
    m.count = 0;
}

void Renderer::drawTriangles(const GpuMesh& m, const Mat4& mvp, const Vec3& light,
                             Rgba tint, bool depthTest, bool depthWrite)
{
    if (!m.count) return;
    const VkCommandBuffer cb = vk_->cmd();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      !depthTest  ? pipeTriOverlay_
                      : depthWrite ? pipeTriOpaque_
                                   : pipeTriBlend_);
    Push3 p{};
    p.mvp = mvp;
    p.light[0] = light.x;
    p.light[1] = light.y;
    p.light[2] = light.z;
    // The shaders have always multiplied by pc.tint; the old signature just
    // pinned rgb to white and exposed only the alpha.
    p.tint[0] = float(tint.r) / 255.0f;
    p.tint[1] = float(tint.g) / 255.0f;
    p.tint[2] = float(tint.b) / 255.0f;
    p.tint[3] = float(tint.a) / 255.0f;
    vkCmdPushConstants(cb, layout3_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(p), &p);
    const VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &m.buf.buf, &off);
    vkCmdDraw(cb, m.count, 1, 0, 0);
}

void Renderer::drawLines(const GpuMesh& m, const Mat4& mvp, Rgba tint,
                         bool depthTest)
{
    if (!m.count) return;
    const VkCommandBuffer cb = vk_->cmd();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      depthTest ? pipeLine_ : pipeLineOverlay_);
    Push3 p{};
    p.mvp = mvp;
    p.light[2] = 1.0f;
    p.tint[0] = float(tint.r) / 255.0f;
    p.tint[1] = float(tint.g) / 255.0f;
    p.tint[2] = float(tint.b) / 255.0f;
    p.tint[3] = float(tint.a) / 255.0f;
    vkCmdPushConstants(cb, layout3_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(p), &p);
    const VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &m.buf.buf, &off);
    vkCmdDraw(cb, m.count, 1, 0, 0);
}

std::expected<void, std::string> Renderer::setGlyphAtlas(const Texture& t)
{
    VkDescriptorImageInfo ii{};
    ii.sampler = t.samp;
    ii.imageView = t.view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = dset_;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(vk_->dev(), 1, &w, 0, nullptr);
    return {};
}

std::expected<void, std::string> Renderer::drawUi(std::span<const Vtx2> verts,
                                                  std::span<const UiBatch> batches)
{
    if (verts.empty() || batches.empty()) return {};
    Buffer& b = ui_[vk_->frameIndex()];
    if (auto r = vk_->ensureCapacity(b, verts.size_bytes(),
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        !r)
        return r;
    std::memcpy(b.mapped, verts.data(), verts.size_bytes());

    const VkCommandBuffer cb = vk_->cmd();
    // The 2D stream is in WINDOW pixels; a viewport left over from the 3D
    // region would shift and squash every panel. Restore it before drawing.
    resetViewport();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeUi_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout2_, 0, 1,
                            &dset_, 0, nullptr);
    const VkExtent2D e = vk_->extent();
    const float inv[2] = {2.0f / float(e.width), 2.0f / float(e.height)};
    vkCmdPushConstants(cb, layout2_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(inv), inv);
    const VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &b.buf, &off);
    for (const auto& ba : batches) {
        if (!ba.count) continue;
        vkCmdSetScissor(cb, 0, 1, &ba.scissor);
        vkCmdDraw(cb, ba.count, 1, ba.first, 0);
    }
    VkRect2D full{{0, 0}, e};
    vkCmdSetScissor(cb, 0, 1, &full);
    return {};
}

void Renderer::setViewport(float x, float y, float w, float h)
{
    const VkCommandBuffer cb = vk_->cmd();
    if (!cb) return;
    const VkExtent2D e = vk_->extent();
    // Clamp into the framebuffer: Vulkan rejects a scissor that leaves it, and
    // a window can be resized to less than the docked columns want.
    const float x0 = std::max(0.0f, x), y0 = std::max(0.0f, y);
    const float x1 = std::min(float(e.width), x + w);
    const float y1 = std::min(float(e.height), y + h);
    const float vw = std::max(1.0f, x1 - x0), vh = std::max(1.0f, y1 - y0);

    VkViewport vp{x0, y0, vw, vh, 0, 1};
    VkRect2D sc{{int32_t(x0), int32_t(y0)}, {uint32_t(vw), uint32_t(vh)}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
}

void Renderer::resetViewport()
{
    const VkCommandBuffer cb = vk_->cmd();
    if (!cb) return;
    const VkExtent2D e = vk_->extent();
    VkViewport vp{0, 0, float(e.width), float(e.height), 0, 1};
    VkRect2D sc{{0, 0}, e};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
}

Renderer::~Renderer()
{
    if (!vk_) return;
    const VkDevice dev = vk_->dev();
    vkDeviceWaitIdle(dev);
    for (auto& b : ui_) vk_->destroy(b);
    if (pipeTriOpaque_) vkDestroyPipeline(dev, pipeTriOpaque_, nullptr);
    if (pipeTriBlend_) vkDestroyPipeline(dev, pipeTriBlend_, nullptr);
    if (pipeLine_) vkDestroyPipeline(dev, pipeLine_, nullptr);
    if (pipeTriOverlay_) vkDestroyPipeline(dev, pipeTriOverlay_, nullptr);
    if (pipeLineOverlay_) vkDestroyPipeline(dev, pipeLineOverlay_, nullptr);
    if (pipeUi_) vkDestroyPipeline(dev, pipeUi_, nullptr);
    if (dpool_) vkDestroyDescriptorPool(dev, dpool_, nullptr);
    if (dsl_) vkDestroyDescriptorSetLayout(dev, dsl_, nullptr);
    if (layout3_) vkDestroyPipelineLayout(dev, layout3_, nullptr);
    if (layout2_) vkDestroyPipelineLayout(dev, layout2_, nullptr);
}

} // namespace ui
