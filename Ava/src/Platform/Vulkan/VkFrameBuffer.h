#pragma once
/// @file VkFrameBuffer.h
/// @brief file implementing FrameBuffer.h for Vulkan.

#include <Graphics/FrameBuffer.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <vulkan/vulkan.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    class VkFrameBuffer final : public FrameBuffer
    {
    public:
        explicit VkFrameBuffer(const FrameBufferDescription& _desc);
        ~VkFrameBuffer() override;

        void SetDebugName(const char* _name) override;

        VkExtent2D GetExtent() const { return { m_width, m_height }; }
        VkFramebuffer GetFramebuffer() const { return m_framebuffer; }
        VkRenderPass GetRenderPass() const { return m_renderPass; }
        bool IsValid() const { return m_framebuffer && m_renderPass; }

    private:
        VkRenderPass m_renderPass = VK_NULL_HANDLE;
        VkFramebuffer m_framebuffer = VK_NULL_HANDLE;
    };

}