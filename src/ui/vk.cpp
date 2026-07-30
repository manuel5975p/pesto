#include "ui/vk.hpp"

#include "ui/vkcheck.hpp"

#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <cstring>

namespace ui {
namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCb(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                       VkDebugUtilsMessageTypeFlagsEXT,
                                       const VkDebugUtilsMessengerCallbackDataEXT* d,
                                       void*)
{
    SDL_Log("[vulkan] %s", d->pMessage);
    return VK_FALSE;
}

bool wantValidation()
{
    const char* e = SDL_getenv("DNS_UI_VALIDATION");
    return e && e[0] && e[0] != '0';
}

} // namespace

std::expected<std::unique_ptr<Vk>, std::string> Vk::create(SDL_Window* window)
{
    auto vk = std::unique_ptr<Vk>(new Vk());
    vk->window_ = window;

    uint32_t nExt = 0;
    const char* const* sdlExt = SDL_Vulkan_GetInstanceExtensions(&nExt);
    if (!sdlExt)
        return std::unexpected(std::string("SDL_Vulkan_GetInstanceExtensions: ") +
                               SDL_GetError());
    std::vector<const char*> exts(sdlExt, sdlExt + nExt);

    std::vector<const char*> layers;
    if (wantValidation()) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dns-ui";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = uint32_t(exts.size());
    ici.ppEnabledExtensionNames = exts.data();
    ici.enabledLayerCount = uint32_t(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    VKCHECK(vkCreateInstance(&ici, nullptr, &vk->inst_), "vkCreateInstance");

    if (!layers.empty()) {
        auto mk = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            vk->inst_, "vkCreateDebugUtilsMessengerEXT");
        if (mk) {
            VkDebugUtilsMessengerCreateInfoEXT dci{};
            dci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = debugCb;
            mk(vk->inst_, &dci, nullptr, &vk->debug_);
        }
    }

    if (!SDL_Vulkan_CreateSurface(window, vk->inst_, nullptr, &vk->surface_))
        return std::unexpected(std::string("SDL_Vulkan_CreateSurface: ") +
                               SDL_GetError());

    uint32_t nDev = 0;
    vkEnumeratePhysicalDevices(vk->inst_, &nDev, nullptr);
    if (nDev == 0) return std::unexpected("no Vulkan physical device");
    std::vector<VkPhysicalDevice> devs(nDev);
    vkEnumeratePhysicalDevices(vk->inst_, &nDev, devs.data());

    // Prefer a discrete GPU that can present; fall back to anything that can.
    int best = -1, bestScore = -1;
    uint32_t bestFamily = 0;
    for (uint32_t d = 0; d < nDev; ++d) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, qf.data());
        for (uint32_t q = 0; q < nq; ++q) {
            if (!(qf[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(devs[d], q, vk->surface_, &present);
            if (!present) continue;
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(devs[d], &p);
            const int score =
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                                                                         : 1;
            if (score > bestScore) { bestScore = score; best = int(d); bestFamily = q; }
            break;
        }
    }
    if (best < 0) return std::unexpected("no Vulkan device can present to the window");
    vk->phys = devs[size_t(best)];
    vk->queueFamily_ = bestFamily;

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = vk->queueFamily_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    const char* devExts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkPhysicalDeviceFeatures have{};
    vkGetPhysicalDeviceFeatures(vk->phys, &have);
    VkPhysicalDeviceFeatures feats{};
    feats.fillModeNonSolid = have.fillModeNonSolid;   // wireframe cut cells
    feats.wideLines = have.wideLines;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = devExts;
    dci.pEnabledFeatures = &feats;
    // No device layers: they have not existed since Vulkan 1.0 and passing
    // them is a validation error in its own right. Instance layers cover the
    // device too.
    VKCHECK(vkCreateDevice(vk->phys, &dci, nullptr, &vk->dev_), "vkCreateDevice");
    vkGetDeviceQueue(vk->dev_, vk->queueFamily_, 0, &vk->queue_);

    // 4x MSAA where the device offers it on both attachments. Grid lines are
    // the entire point of this viewer and they alias badly at 1x.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(vk->phys, &props);
    const VkSampleCountFlags ok = props.limits.framebufferColorSampleCounts &
                                  props.limits.framebufferDepthSampleCounts;
    if (ok & VK_SAMPLE_COUNT_4_BIT) vk->samples_ = VK_SAMPLE_COUNT_4_BIT;
    else if (ok & VK_SAMPLE_COUNT_2_BIT) vk->samples_ = VK_SAMPLE_COUNT_2_BIT;

    for (auto& f : vk->frames_) {
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = vk->queueFamily_;
        VKCHECK(vkCreateCommandPool(vk->dev_, &pci, nullptr, &f.pool),
                "vkCreateCommandPool");
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = f.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VKCHECK(vkAllocateCommandBuffers(vk->dev_, &ai, &f.cmd),
                "vkAllocateCommandBuffers");
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VKCHECK(vkCreateSemaphore(vk->dev_, &si, nullptr, &f.acquired),
                "vkCreateSemaphore");
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VKCHECK(vkCreateFence(vk->dev_, &fi, nullptr, &f.done), "vkCreateFence");
    }
    {
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pci.queueFamilyIndex = vk->queueFamily_;
        VKCHECK(vkCreateCommandPool(vk->dev_, &pci, nullptr, &vk->uploadPool_),
                "vkCreateCommandPool(upload)");
    }

    if (auto r = vk->initSwapchain(); !r) return std::unexpected(r.error());
    return vk;
}

std::expected<void, std::string> Vk::initSwapchain()
{
    VkSurfaceCapabilitiesKHR caps{};
    VKCHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface_, &caps),
            "vkGetPhysicalDeviceSurfaceCapabilities");

    int pw = 0, ph = 0;
    SDL_GetWindowSizeInPixels(window_, &pw, &ph);
    extent_ = caps.currentExtent.width != UINT32_MAX
                  ? caps.currentExtent
                  : VkExtent2D{std::clamp(uint32_t(pw), caps.minImageExtent.width,
                                          caps.maxImageExtent.width),
                               std::clamp(uint32_t(ph), caps.minImageExtent.height,
                                          caps.maxImageExtent.height)};
    if (extent_.width == 0 || extent_.height == 0) return {};   // minimized

    uint32_t nFmt = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface_, &nFmt, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(nFmt);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface_, &nFmt, fmts.data());
    // UNORM, not SRGB: the palette is authored in the values that hit the
    // screen, so an implicit linear->sRGB encode would wash it out.
    VkSurfaceFormatKHR chosen = fmts[0];
    for (const auto& f : fmts)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM ||
             f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    colorFormat_ = chosen.format;

    uint32_t nImg = caps.minImageCount + 1;
    if (caps.maxImageCount && nImg > caps.maxImageCount) nImg = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface = surface_;
    sci.minImageCount = nImg;
    sci.imageFormat = colorFormat_;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = extent_;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;   // always supported; vsync
    sci.clipped = VK_TRUE;
    VKCHECK(vkCreateSwapchainKHR(dev_, &sci, nullptr, &swapchain_),
            "vkCreateSwapchainKHR");

    uint32_t got = 0;
    vkGetSwapchainImagesKHR(dev_, swapchain_, &got, nullptr);
    images_.resize(got);
    vkGetSwapchainImagesKHR(dev_, swapchain_, &got, images_.data());

    views_.resize(got);
    for (uint32_t i = 0; i < got; ++i) {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = images_[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = colorFormat_;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VKCHECK(vkCreateImageView(dev_, &vci, nullptr, &views_[i]),
                "vkCreateImageView");
    }

    renderDone_.resize(got);
    for (auto& s : renderDone_) {
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VKCHECK(vkCreateSemaphore(dev_, &si, nullptr, &s), "vkCreateSemaphore");
    }

    auto makeImage = [&](VkFormat fmt, VkImageUsageFlags usage,
                         VkImageAspectFlags aspect, VkImage& img,
                         VkDeviceMemory& mem,
                         VkImageView& view) -> std::expected<void, std::string> {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = {extent_.width, extent_.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = samples_;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VKCHECK(vkCreateImage(dev_, &ici, nullptr, &img), "vkCreateImage");
        VkMemoryRequirements mr{};
        vkGetImageMemoryRequirements(dev_, img, &mr);
        auto type = findMemoryType(mr.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!type) return std::unexpected(type.error());
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = *type;
        VKCHECK(vkAllocateMemory(dev_, &mai, nullptr, &mem), "vkAllocateMemory");
        vkBindImageMemory(dev_, img, mem, 0);
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt;
        vci.subresourceRange = {aspect, 0, 1, 0, 1};
        VKCHECK(vkCreateImageView(dev_, &vci, nullptr, &view), "vkCreateImageView");
        return {};
    };

    if (auto r = makeImage(depthFormat_,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT, depthImg_, depthMem_,
                           depthView_);
        !r)
        return r;
    if (samples_ != VK_SAMPLE_COUNT_1_BIT)
        if (auto r = makeImage(colorFormat_,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT, msaaImg_, msaaMem_,
                               msaaView_);
            !r)
            return r;

    // One subpass. With MSAA the colour attachment is the multisample image
    // and the swapchain image is its resolve target; without, they are one.
    const bool ms = samples_ != VK_SAMPLE_COUNT_1_BIT;
    VkAttachmentDescription atts[3]{};
    atts[0].format = colorFormat_;                  // the presented image
    atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
    atts[0].loadOp = ms ? VK_ATTACHMENT_LOAD_OP_DONT_CARE
                        : VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    atts[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    atts[1].format = depthFormat_;
    atts[1].samples = samples_;
    atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    atts[2].format = colorFormat_;                  // MSAA colour
    atts[2].samples = samples_;
    atts[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[2].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{ms ? 2u : 0u,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1,
                                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkAttachmentReference resolveRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;
    if (ms) sub.pResolveAttachments = &resolveRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = dep.srcStageMask;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = ms ? 3 : 2;
    rpci.pAttachments = atts;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;
    VKCHECK(vkCreateRenderPass(dev_, &rpci, nullptr, &renderPass_),
            "vkCreateRenderPass");

    fbs_.resize(got);
    for (uint32_t i = 0; i < got; ++i) {
        VkImageView fbv[3] = {views_[i], depthView_, msaaView_};
        VkFramebufferCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass = renderPass_;
        fci.attachmentCount = ms ? 3 : 2;
        fci.pAttachments = fbv;
        fci.width = extent_.width;
        fci.height = extent_.height;
        fci.layers = 1;
        VKCHECK(vkCreateFramebuffer(dev_, &fci, nullptr, &fbs_[i]),
                "vkCreateFramebuffer");
    }
    return {};
}

void Vk::destroySwapchain()
{
    for (auto f : fbs_) vkDestroyFramebuffer(dev_, f, nullptr);
    fbs_.clear();
    if (renderPass_) vkDestroyRenderPass(dev_, renderPass_, nullptr);
    renderPass_ = VK_NULL_HANDLE;
    for (auto s : renderDone_) vkDestroySemaphore(dev_, s, nullptr);
    renderDone_.clear();
    for (auto v : views_) vkDestroyImageView(dev_, v, nullptr);
    views_.clear();
    images_.clear();
    if (depthView_) vkDestroyImageView(dev_, depthView_, nullptr);
    if (depthImg_) vkDestroyImage(dev_, depthImg_, nullptr);
    if (depthMem_) vkFreeMemory(dev_, depthMem_, nullptr);
    depthView_ = VK_NULL_HANDLE;
    depthImg_ = VK_NULL_HANDLE;
    depthMem_ = VK_NULL_HANDLE;
    if (msaaView_) vkDestroyImageView(dev_, msaaView_, nullptr);
    if (msaaImg_) vkDestroyImage(dev_, msaaImg_, nullptr);
    if (msaaMem_) vkFreeMemory(dev_, msaaMem_, nullptr);
    msaaView_ = VK_NULL_HANDLE;
    msaaImg_ = VK_NULL_HANDLE;
    msaaMem_ = VK_NULL_HANDLE;
    if (swapchain_) vkDestroySwapchainKHR(dev_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

bool Vk::beginFrame(const float clear[4])
{
    Frame& f = frames_[frame_];
    vkWaitForFences(dev_, 1, &f.done, VK_TRUE, UINT64_MAX);

    if (swapchainDirty_ || !swapchain_) {
        vkDeviceWaitIdle(dev_);
        destroySwapchain();
        swapchainDirty_ = false;
        if (!initSwapchain()) return false;
        if (!swapchain_) return false;   // still minimized
    }

    const VkResult acq = vkAcquireNextImageKHR(dev_, swapchain_, UINT64_MAX,
                                               f.acquired, VK_NULL_HANDLE,
                                               &imageIndex_);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchainDirty_ = true;
        return false;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) return false;

    vkResetFences(dev_, 1, &f.done);
    vkResetCommandBuffer(f.cmd, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);

    VkClearValue cv[3]{};
    cv[0].color = {{clear[0], clear[1], clear[2], clear[3]}};
    cv[1].depthStencil = {1.0f, 0};
    cv[2].color = cv[0].color;

    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = renderPass_;
    rp.framebuffer = fbs_[imageIndex_];
    rp.renderArea.extent = extent_;
    rp.clearValueCount = samples_ != VK_SAMPLE_COUNT_1_BIT ? 3 : 2;
    rp.pClearValues = cv;
    vkCmdBeginRenderPass(f.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{0, 0, float(extent_.width), float(extent_.height), 0, 1};
    VkRect2D sc{{0, 0}, extent_};
    vkCmdSetViewport(f.cmd, 0, 1, &vp);
    vkCmdSetScissor(f.cmd, 0, 1, &sc);

    frameOpen_ = true;
    return true;
}

void Vk::endFrame()
{
    if (!frameOpen_) return;
    frameOpen_ = false;
    Frame& f = frames_[frame_];
    vkCmdEndRenderPass(f.cmd);
    vkEndCommandBuffer(f.cmd);

    const VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &f.acquired;
    si.pWaitDstStageMask = &wait;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &renderDone_[imageIndex_];
    vkQueueSubmit(queue_, 1, &si, f.done);

    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &renderDone_[imageIndex_];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &imageIndex_;
    const VkResult pr = vkQueuePresentKHR(queue_, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR)
        swapchainDirty_ = true;

    frame_ = (frame_ + 1) % framesInFlight;
}

std::expected<uint32_t, std::string> Vk::findMemoryType(uint32_t bits,
                                                        VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return std::unexpected("no suitable Vulkan memory type");
}

std::expected<Buffer, std::string> Vk::createHostBuffer(VkDeviceSize size,
                                                        VkBufferUsageFlags usage)
{
    Buffer b;
    b.size = std::max<VkDeviceSize>(size, 256);
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = b.size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKCHECK(vkCreateBuffer(dev_, &bci, nullptr, &b.buf), "vkCreateBuffer");
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(dev_, b.buf, &mr);
    auto type = findMemoryType(mr.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!type) return std::unexpected(type.error());
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = *type;
    VKCHECK(vkAllocateMemory(dev_, &mai, nullptr, &b.mem), "vkAllocateMemory");
    vkBindBufferMemory(dev_, b.buf, b.mem, 0);
    VKCHECK(vkMapMemory(dev_, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
    return b;
}

std::expected<void, std::string> Vk::ensureCapacity(Buffer& b, VkDeviceSize need,
                                                    VkBufferUsageFlags usage)
{
    if (b.buf && b.size >= need) return {};
    VkDeviceSize want = std::max<VkDeviceSize>(need, 4096);
    if (b.size) want = std::max(want, b.size + b.size / 2);
    // The old buffer may still be referenced by an in-flight frame.
    vkDeviceWaitIdle(dev_);
    destroy(b);
    auto nb = createHostBuffer(want, usage);
    if (!nb) return std::unexpected(nb.error());
    b = *nb;
    return {};
}

void Vk::destroy(Buffer& b)
{
    if (b.mem && b.mapped) vkUnmapMemory(dev_, b.mem);
    if (b.buf) vkDestroyBuffer(dev_, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev_, b.mem, nullptr);
    b = Buffer{};
}

std::expected<Texture, std::string> Vk::createCoverageTexture(uint32_t w, uint32_t h)
{
    Texture t;
    t.w = w;
    t.h = h;
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8_UNORM;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKCHECK(vkCreateImage(dev_, &ici, nullptr, &t.img), "vkCreateImage(atlas)");
    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(dev_, t.img, &mr);
    auto type = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!type) return std::unexpected(type.error());
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = *type;
    VKCHECK(vkAllocateMemory(dev_, &mai, nullptr, &t.mem), "vkAllocateMemory(atlas)");
    vkBindImageMemory(dev_, t.img, t.mem, 0);

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = t.img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VKCHECK(vkCreateImageView(dev_, &vci, nullptr, &t.view),
            "vkCreateImageView(atlas)");

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VKCHECK(vkCreateSampler(dev_, &sci, nullptr, &t.samp), "vkCreateSampler");
    return t;
}

std::expected<void, std::string> Vk::uploadCoverage(Texture& t, const uint8_t* pixels)
{
    const VkDeviceSize bytes = VkDeviceSize(t.w) * t.h;
    auto stage = createHostBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    if (!stage) return std::unexpected(stage.error());
    std::memcpy(stage->mapped, pixels, size_t(bytes));

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = uploadPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VKCHECK(vkAllocateCommandBuffers(dev_, &ai, &cb), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    auto barrier = [&](VkImageLayout from, VkImageLayout to, VkAccessFlags srcA,
                       VkAccessFlags dstA, VkPipelineStageFlags srcS,
                       VkPipelineStageFlags dstS) {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = t.img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.srcAccessMask = srcA;
        b.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cb, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy cp{};
    cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.imageExtent = {t.w, t.h, 1};
    vkCmdCopyBufferToImage(cb, stage->buf, t.img,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    vkEndCommandBuffer(cb);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);

    vkFreeCommandBuffers(dev_, uploadPool_, 1, &cb);
    destroy(*stage);
    return {};
}

void Vk::destroy(Texture& t)
{
    if (t.samp) vkDestroySampler(dev_, t.samp, nullptr);
    if (t.view) vkDestroyImageView(dev_, t.view, nullptr);
    if (t.img) vkDestroyImage(dev_, t.img, nullptr);
    if (t.mem) vkFreeMemory(dev_, t.mem, nullptr);
    t = Texture{};
}

void Vk::waitIdle()
{
    if (dev_) vkDeviceWaitIdle(dev_);
}

Vk::~Vk()
{
    if (dev_) {
        vkDeviceWaitIdle(dev_);
        destroySwapchain();
        for (auto& f : frames_) {
            if (f.done) vkDestroyFence(dev_, f.done, nullptr);
            if (f.acquired) vkDestroySemaphore(dev_, f.acquired, nullptr);
            if (f.pool) vkDestroyCommandPool(dev_, f.pool, nullptr);
        }
        if (uploadPool_) vkDestroyCommandPool(dev_, uploadPool_, nullptr);
        vkDestroyDevice(dev_, nullptr);
    }
    if (surface_) vkDestroySurfaceKHR(inst_, surface_, nullptr);
    if (debug_) {
        auto rm = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            inst_, "vkDestroyDebugUtilsMessengerEXT");
        if (rm) rm(inst_, debug_, nullptr);
    }
    if (inst_) vkDestroyInstance(inst_, nullptr);
}

} // namespace ui
