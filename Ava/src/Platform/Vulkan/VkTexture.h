#pragma once
/// @file VkTexture.h
/// @brief file implementing Texture.h for Vulkan.

#include <Graphics/Texture.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <vulkan/vulkan.h>
#include <VMA/vk_mem_alloc.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    class VkTexture final : public Texture
    {
    public:
        VkTexture() = default;
        ~VkTexture() override;

        /// @brief for render target textures.
        explicit VkTexture(const TextureDescription& _desc);
        /// @brief for path-based texture resources.
        explicit VkTexture(const char* _path, u32 _flags = 0);
        /// @brief for built-in texture data.
        explicit VkTexture(const TextureData& _data, u32 _flags = 0);
        /// @brief for swapchain images.
        explicit VkTexture(VkExtent2D _extent, VkImage _source, VkFormat _format);

        void SetDebugName(const char* _name) override;
        void Load(const TextureData& _data);

        VkFormat GetVkFormat() const { return m_vkFormat; }
        VkImage GetImage() const { return m_image; }
        VkImageView GetImageView() const { return m_view; }
        VkImageLayout GetLayout() const { return m_layout; }
        VkImageAspectFlagBits GetAspectFlag() const { return m_aspect; }

        /// @brief For storage texture views.
        VkImageView GetMipView(u16 _mip);
        /// @brief For framebuffer attachment views.
        VkImageView GetMipLayerView(u16 _layer, u16 _mip);

        VkSampler GetSampler();
        VkImageLayout GetDefaultLayout() const;

        void TransitionLayout(VkCommandBuffer _cmd, VkImageLayout _newLayout);
        void TransitionToDefaultLayout(const VkCommandBuffer _cmd) { TransitionLayout(_cmd, GetDefaultLayout()); }
        void CopyBufferToImage(VkCommandBuffer _cmd, VkBuffer _buffer, const u32* _mipSizes, u32 _layerCount, u32 _mipCount);

    private:
        void _CreateImage();
        void _CreateView();
        void _CreateSampler();
        void _UploadPixels(const TextureData& _data);

        VkImage m_image = VK_NULL_HANDLE;
        VmaAllocation m_alloc = VK_NULL_HANDLE;
        VkImageView m_view = VK_NULL_HANDLE;
        VkSampler m_sampler = VK_NULL_HANDLE;
        VkFormat m_vkFormat = VK_FORMAT_UNDEFINED;
        VkImageLayout m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageAspectFlagBits m_aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        u32 m_alignment = 4;

        // Single mip views, for storage textures.
        std::vector<VkImageView> m_mipViews;
        // Single mip layer views, for framebuffer attachments.
        std::map<std::pair<u16, u16>, VkImageView> m_mipLayerViews;
    };

}
