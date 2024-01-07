#include <avapch.h>
#include "VkFrameBuffer.h"

#include <Debug/Assert.h>
#include <Platform/Vulkan/VkTexture.h>
#include <Platform/Vulkan/VkGraphicsContext.h>

namespace Ava {

    VkFrameBuffer::VkFrameBuffer(const FrameBufferDescription& _desc)
        : FrameBuffer(_desc)
    {
        std::vector<VkAttachmentDescription> attachments(m_attachmentCount);
        std::vector<VkImageView> attachmentViews(m_attachmentCount);

        std::vector<VkAttachmentReference> colorAttachmentsRef(m_colorAttachmentCount);
        VkAttachmentReference depthAttachmentRef{};

        u32 currentAttachment = 0;

        // ------Color attachments------------------------------------
        for (u32 i = 0; i < m_colorAttachmentCount; i++)
        {
            const FramebufferAttachment* colorAttachment = GetColorAttachment(i);
            auto* texture = static_cast<VkTexture*>(colorAttachment->texture);

            const VkAttachmentLoadOp loadOperation = texture->GetLayout() == VK_IMAGE_LAYOUT_UNDEFINED
                                                    ? VK_ATTACHMENT_LOAD_OP_DONT_CARE
                                                    : VK_ATTACHMENT_LOAD_OP_LOAD;

            VkAttachmentDescription& colorAttachmentInfo = attachments[currentAttachment];
            colorAttachmentInfo.format = texture->GetVkFormat();
            colorAttachmentInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachmentInfo.loadOp = loadOperation;
            colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachmentInfo.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachmentInfo.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachmentInfo.initialLayout = texture->GetLayout();
            colorAttachmentInfo.finalLayout = texture->GetDefaultLayout();

            VkAttachmentReference& colorAttachmentRef = colorAttachmentsRef[i];
            colorAttachmentRef.attachment = currentAttachment;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            attachmentViews[currentAttachment] = texture->GetMipLayerView(colorAttachment->layer, colorAttachment->mip);
            currentAttachment++;
        }

        // ------Depth attachment--------------------------------------
        if (HasDepthAttachment())
        {
            const FramebufferAttachment* depthAttachment = GetDepthAttachment();
            auto* texture = static_cast<VkTexture*>(depthAttachment->texture);

            const VkAttachmentLoadOp loadOperation = texture->GetLayout() == VK_IMAGE_LAYOUT_UNDEFINED
                                                    ? VK_ATTACHMENT_LOAD_OP_DONT_CARE
                                                    : VK_ATTACHMENT_LOAD_OP_LOAD;

            VkAttachmentDescription& depthAttachmentInfo = attachments[currentAttachment];
            depthAttachmentInfo.format = texture->GetVkFormat();
            depthAttachmentInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            depthAttachmentInfo.loadOp = loadOperation;
            depthAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAttachmentInfo.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachmentInfo.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachmentInfo.initialLayout = texture->GetLayout();
            depthAttachmentInfo.finalLayout = texture->GetDefaultLayout();

            depthAttachmentRef.attachment = currentAttachment;
            depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            attachmentViews[currentAttachment] = texture->GetMipLayerView(depthAttachment->layer, depthAttachment->mip);
            currentAttachment++;
        }

        // -------Sub-pass description-------------------------------------
        VkSubpassDescription subPass{};
        subPass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subPass.colorAttachmentCount = m_colorAttachmentCount;
        subPass.pColorAttachments = colorAttachmentsRef.data();
        subPass.pDepthStencilAttachment = HasDepthAttachment() ? &depthAttachmentRef : nullptr;

        std::array<VkSubpassDependency, 2> dependencies{};
        dependencies[0].dependencyFlags = 0;
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = 0;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        // Self dependency, required to make memory barrier during a render pass.
        VkAccessFlags anyAccessMask =
            //VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
            VK_ACCESS_INDEX_READ_BIT |
            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
            VK_ACCESS_UNIFORM_READ_BIT |
            VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        dependencies[1].dependencyFlags = 0;
        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = 0;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
        dependencies[1].srcAccessMask = anyAccessMask;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
        dependencies[1].dstAccessMask = anyAccessMask;

        // -------Render pass creation------------------------------------
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = m_attachmentCount;
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subPass;
        renderPassInfo.dependencyCount = (u32)dependencies.size();
        renderPassInfo.pDependencies = dependencies.data();

        m_renderPass = VkGraphicsContext::CreateRenderPass(&renderPassInfo);

        // -------Framebuffer creation------------------------------------
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = m_renderPass;
        framebufferInfo.width = m_width;
        framebufferInfo.height = m_height;
        framebufferInfo.layers = 1;
        framebufferInfo.pAttachments = attachmentViews.data();
        framebufferInfo.attachmentCount = m_attachmentCount;

        const VkResult result = vkCreateFramebuffer(VkGraphicsContext::GetDevice(), &framebufferInfo, nullptr, &m_framebuffer);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create framebuffer: %s", VkGraphicsContext::GetVkResultStr(result));
    }

    VkFrameBuffer::~VkFrameBuffer()
    {
        vkDestroyFramebuffer(VkGraphicsContext::GetDevice(), m_framebuffer, nullptr);
    }

    void VkFrameBuffer::SetDebugName(const char* _name)
    {
        VkGraphicsContext::SetDebugObjectName(m_framebuffer, VK_OBJECT_TYPE_FRAMEBUFFER, _name);
        VkGraphicsContext::SetDebugObjectName(m_renderPass, VK_OBJECT_TYPE_RENDER_PASS, _name);
    }

}