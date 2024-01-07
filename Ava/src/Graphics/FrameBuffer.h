#pragma once
/// @file FrameBuffer.h
/// @brief interface for framebuffers.

#include <Graphics/Texture.h>
#include <Math/Types.h>

namespace Ava {

    static constexpr u32 kMaxFramebufferAttachments = 5;

    struct FramebufferAttachment
    {
        Texture* texture = nullptr;
        u16 layer = 0;
        u16 mip = 0;

        u32 GetWidth() const;
        u32 GetHeight() const;
        bool operator==(const FramebufferAttachment& _other) const;
    };

    struct FrameBufferDescription
    {
        FramebufferAttachment attachments[kMaxFramebufferAttachments]{};
        u32 attachmentCount = 0;

        void AddAttachment(Texture* _texture, u16 _layer = 0, u16 _mip = 0);
    };

    /// @brief Regroups a bunch of textures in which intermediate render results are stored.
    class FrameBuffer
    {
    public:
        explicit FrameBuffer(const FrameBufferDescription& _desc);
        virtual ~FrameBuffer() = default;

        virtual void SetDebugName(const char* _name) { }

        u32 GetWidth() const { return m_width; }
        u32 GetHeight() const { return m_height; }
        Vec2u GetSize() const { return { m_width, m_height }; }

        u32 GetColorAttachmentCount() const { return m_colorAttachmentCount; }
        const FramebufferAttachment* GetColorAttachment(u8 _index) const;

        bool HasDepthAttachment() const { return m_hasDepthAttachment; }
        const FramebufferAttachment* GetDepthAttachment() const;

    protected:
        u32 m_width = 0;
        u32 m_height = 0;
        u32 m_attachmentCount = 0;

        // color attachments
        static constexpr u8 kMaxColorAttachments = kMaxFramebufferAttachments - 1;
        FramebufferAttachment m_colorAttachments[kMaxColorAttachments];
        u32 m_colorAttachmentCount = 0;

        // depth stencil attachment
        FramebufferAttachment m_depthAttachment;
        bool m_hasDepthAttachment = false;
    };

}
