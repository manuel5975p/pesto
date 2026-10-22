// The Vulkan device this UI runs on: instance, device, swapchain, one render
// pass, and a small pile of resource helpers. Nothing above this file touches
// a VkPhysicalDevice or a swapchain image.
//
// Deliberately small: a single graphics+present queue, FIFO present (a viewer
// that spins the GPU at 400 fps to show a static grid is a bug), two frames in
// flight, and one render pass with colour + depth. Everything drawn goes
// through that pass -- 3D first, then the 2D UI with depth test off.
#pragma once

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace ui {

// Host-visible, persistently mapped, HOST_COHERENT: the vertex data is
// regenerated every frame from an immediate-mode UI, so a staging copy would
// buy nothing. Grows by reallocation (see Vk::ensureCapacity).
struct Buffer
{
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

// R8_UNORM coverage texture with its sampler; the glyph atlas is the only one.
struct Texture
{
    VkImage img = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler samp = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
};

class Vk
{
public:
    static constexpr int framesInFlight = 2;

    // Creates everything against an already-created SDL window that carries
    // SDL_WINDOW_VULKAN. Any failure is a user-visible string (no GPU, no
    // driver, no surface) -- never an assert: a missing Vulkan driver is a
    // user condition, not a programmer error.
    static std::expected<std::unique_ptr<Vk>, std::string> create(SDL_Window*);
    ~Vk();

    Vk(const Vk&) = delete;
    Vk& operator=(const Vk&) = delete;

    // Acquires an image and opens the render pass with `clear` as the colour.
    // Returns false when the frame must be skipped (window minimized, or the
    // swapchain went out of date and was rebuilt) -- the caller simply does
    // not draw and comes back next iteration.
    bool beginFrame(const float clear[4]);
    void endFrame();
    // A Vulkan call failed in a way no frame can recover from (device lost, a
    // command buffer that would not begin or submit). Sticky; the app quits.
    bool deviceLost() const { return deviceLost_; }
    // Call on SDL_EVENT_WINDOW_RESIZED / PIXEL_SIZE_CHANGED.
    void invalidateSwapchain() { swapchainDirty_ = true; }

    VkCommandBuffer cmd() const { return frames_[frame_].cmd; }
    VkExtent2D extent() const { return extent_; }
    VkRenderPass renderPass() const { return renderPass_; }
    int frameIndex() const { return frame_; }
    VkDevice dev() const { return dev_; }
    // MSAA sample count actually in use (1 = off). The 3D grid is nothing but
    // thin lines, so this is the difference between a readable mesh and a
    // shimmering moire.
    VkSampleCountFlagBits samples() const { return samples_; }

    // --- resources ----------------------------------------------------------
    std::expected<Buffer, std::string> createHostBuffer(VkDeviceSize,
                                                        VkBufferUsageFlags);
    // Reallocates `b` if it is smaller than `need` (rounded up, 1.5x growth).
    // Safe only between frames or on a buffer this frame has not recorded yet.
    std::expected<void, std::string> ensureCapacity(Buffer& b, VkDeviceSize need,
                                                    VkBufferUsageFlags);
    void destroy(Buffer&);

    std::expected<Texture, std::string> createCoverageTexture(uint32_t w,
                                                              uint32_t h);
    // Full-image upload through a staging buffer on a blocking immediate
    // submit. The atlas changes only when a new glyph is rasterized, so the
    // stall is a non-issue and it keeps the frame path free of barriers.
    std::expected<void, std::string> uploadCoverage(Texture&,
                                                    const uint8_t* pixels);
    void destroy(Texture&);

    // Blocks until the device is idle; call before tearing down pipelines.
    void waitIdle();

    VkPhysicalDevice phys = VK_NULL_HANDLE;

private:
    Vk() = default;
    std::expected<void, std::string> initSwapchain();
    void destroySwapchain();
    std::expected<uint32_t, std::string> findMemoryType(uint32_t bits,
                                                        VkMemoryPropertyFlags);

    SDL_Window* window_ = nullptr;
    VkInstance inst_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkQueue queue_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat colorFormat_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depthFormat_ = VK_FORMAT_D32_SFLOAT;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkExtent2D extent_{0, 0};
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    std::vector<VkFramebuffer> fbs_;
    // Per-swapchain-image signal semaphore: signalling the same semaphore from
    // two in-flight submits is the classic validation error here.
    std::vector<VkSemaphore> renderDone_;
    VkImage depthImg_ = VK_NULL_HANDLE;
    VkDeviceMemory depthMem_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;
    VkImage msaaImg_ = VK_NULL_HANDLE;
    VkDeviceMemory msaaMem_ = VK_NULL_HANDLE;
    VkImageView msaaView_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;

    struct Frame
    {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkSemaphore acquired = VK_NULL_HANDLE;
        VkFence done = VK_NULL_HANDLE;
    };
    Frame frames_[framesInFlight];
    int frame_ = 0;
    uint32_t imageIndex_ = 0;
    bool swapchainDirty_ = false;
    bool frameOpen_ = false;
    bool deviceLost_ = false;
    // Upper bound on a frame-fence wait; past it the frame is skipped.
    static constexpr uint64_t kFenceTimeoutNs = 5'000'000'000ull;

    VkCommandPool uploadPool_ = VK_NULL_HANDLE;
};

} // namespace ui
