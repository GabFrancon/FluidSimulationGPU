#include <avapch.h>
#include "FrameBuffer.h"

#include <Graphics/GraphicsContext.h>
#include <Debug/Assert.h>
#include <Debug/Log.h>

namespace Ava {

    // ----- Framebuffer Attachment --------------------------------------------

    u32 FramebufferAttachment::GetWidth() const
    {
        return texture->GetMipWidth(mip);
    }

    u32 FramebufferAttachment::GetHeight() const
    {
        return texture->GetMipHeight(mip);
    }

    bool FramebufferAttachment::operator==(const FramebufferAttachment& _other) const
    {
        return
            texture == _other.texture
            && layer == _other.layer
            && mip == _other.mip;
    }


    // ----- Framebuffer Description -------------------------------------------

    void FrameBufferDescription::AddAttachment(Texture* _texture, const u16 _layer/*= 0*/, const u16 _mip/*= 0*/)
    {
        AVA_ASSERT(attachmentCount < kMaxFramebufferAttachments - 1, 
            "Reached max framebuffer attachment count (%d).", kMaxFramebufferAttachments);

        attachments[attachmentCount].texture = _texture;
        attachments[attachmentCount].layer = _layer;
        attachments[attachmentCount].mip = _mip;
        attachmentCount++;
    }


    // ----- Framebuffer  ------------------------------------------------------

    FrameBuffer::FrameBuffer(const FrameBufferDescription& _desc)
    {
        if (_desc.attachmentCount == 0) {
            return;
        }

        m_width = _desc.attachments[0].GetWidth();
        m_height = _desc.attachments[0].GetHeight();

        // Identifies the different attachments
        for (u8 i = 0; i < _desc.attachmentCount; i++)
        {
            AVA_ASSERT(_desc.attachments[i].texture, 
                "all render targets attached to a framebuffer must be valid.");

            AVA_ASSERT(_desc.attachments[i].texture->GetFlags() & AVA_TEXTURE_RENDER_TARGET,
                "all render targets attached to a framebuffer must have the AVA_RENDER_TARGET flag.");

            AVA_ASSERT(_desc.attachments[i].GetWidth() == m_width && _desc.attachments[i].GetHeight() == m_height,
                "all render targets attached to a framebuffer must have the same size.");

            if (IsDepth(_desc.attachments[i].texture->GetFormat()))
            {
                m_depthAttachment = _desc.attachments[i];
                m_hasDepthAttachment = true;
                m_attachmentCount++;
            }
            else if (m_colorAttachmentCount < kMaxColorAttachments)
            {
                m_colorAttachments[m_colorAttachmentCount] = _desc.attachments[i];
                m_colorAttachmentCount++;
                m_attachmentCount++;
            }
            else
            {
                AVA_CORE_ERROR(
                    "You reached the maximum number of attachments (%d) per framebuffer.",
                    kMaxColorAttachments);

                return;
            }
        }
    }

    const FramebufferAttachment* FrameBuffer::GetColorAttachment(const u8 _index) const
    {
        if (!AVA_VERIFY(_index < m_colorAttachmentCount, "Invalid color attachment index (%d)", _index))
        {
            return nullptr;
        }
        return m_colorAttachments[_index].texture ? &m_colorAttachments[_index] : nullptr;
    }

    const FramebufferAttachment* FrameBuffer::GetDepthAttachment() const
    {
        return m_depthAttachment.texture ? &m_depthAttachment : nullptr;
    }

}