// The one Vulkan error check. Every VkResult-returning call in the UI funnels
// through it so a driver failure becomes a message rather than a silently
// half-built device.
//
// Usable only inside a function returning std::expected<T, std::string> --
// which is every Vulkan setup path in vk.cpp and draw.cpp.
#pragma once

#include <string>

#define VKCHECK(expr, what)                                                    \
    do {                                                                       \
        const VkResult _r = (expr);                                            \
        if (_r != VK_SUCCESS)                                                  \
            return std::unexpected(std::string(what) + " failed (VkResult " +  \
                                   std::to_string(int(_r)) + ")");             \
    } while (0)
