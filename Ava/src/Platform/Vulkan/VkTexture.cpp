#include <avapch.h>
#include "VkTexture.h"
#include "VkGraphicsContext.h"

#include <Debug/Log.h>
#include <Debug/Assert.h>
#include <Math/BitUtils.h>
#include <Resources/TextureData.h>
#include <Platform/Vulkan/VkGraphicsContext.h>

namespace Ava {

    VkTexture::VkTexture(const TextureDescription& _desc)
        : Texture(_desc)
    {
        m_vkFormat = ToVk(m_format);
        m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        m_aspect = IsDepth(m_format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

        _CreateImage();
        _CreateView();

        const VkCommandBuffer cmd = VkGraphicsContext::BeginTransferCommand();
        TransitionToDefaultLayout(cmd);
        VkGraphicsContext::EndTransferCommand(cmd);
    }

    VkTexture::VkTexture(const char* _path, const u32 _flags/*= 0*/)
        : Texture(_path, _flags)
    {
        const char* path = GetResourcePath();
        TextureData data;

        if (!TextureLoader::Load(path, data))
        {
            AVA_CORE_ERROR("[Texture] failed to load '%s'.", path);
            SetResourceState(ResourceState::LoadFailed);
            return;
        }

        Load(data);
        TextureLoader::Release(data);
        SetResourceState(ResourceState::Loaded);
    }

    VkTexture::VkTexture(const TextureData& _data, const u32 _flags/*= 0*/)
        : Texture(_data, _flags)
    {
        Load(_data);
        SetResourceState(ResourceState::Loaded);
    }

    VkTexture::VkTexture(const VkExtent2D _extent, const VkImage _source, const VkFormat _format)
    {
        m_width = _extent.width;
        m_height = _extent.height;
        m_type = TextureType::Texture2D;
        m_format = TextureFormat::Undefined;
        m_flags = AVA_TEXTURE_DISPLAYABLE | AVA_TEXTURE_RENDER_TARGET;

        m_image = _source;
        m_vkFormat = _format;
        m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        m_aspect = VK_IMAGE_ASPECT_COLOR_BIT;

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = m_vkFormat;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        const VkResult result = vkCreateImageView(VkGraphicsContext::GetDevice(), &viewInfo, nullptr, &m_view);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create swapchain image view: %s", VkGraphicsContext::GetVkResultStr(result));
    }

    VkTexture::~VkTexture()
    {
        for (const auto& imageView : m_mipViews)
        {
            if (imageView)
            {
                vkDestroyImageView(VkGraphicsContext::GetDevice(), imageView, nullptr);
            }
        }
        for (const auto& pair : m_mipLayerViews)
        {
            if (pair.second)
            {
                vkDestroyImageView(VkGraphicsContext::GetDevice(), pair.second, nullptr);
            }
        }
        if (m_view)
        {
            vkDestroyImageView(VkGraphicsContext::GetDevice(), m_view, nullptr);
        }
        if (!HasFlag(AVA_TEXTURE_DISPLAYABLE) && m_image)
        {
            vmaDestroyImage(VkGraphicsContext::GetAllocator(), m_image, m_alloc);
        }
    }

    void VkTexture::SetDebugName(const char* _name)
    {
        VkGraphicsContext::SetDebugObjectName(m_image, VK_OBJECT_TYPE_IMAGE, _name);
    }

    void VkTexture::Load(const TextureData& _data)
    {
        m_width = _data.width;
        m_height = _data.height;
        m_depth = _data.depth;
        m_format = _data.format;
        m_layerCount = _data.imageCount;

        // Deduce texture type
        const int dimension = m_depth > 1 ? 3 : (m_width > 1 && m_height > 1) ? 2 : 1;

        if (_data.isCubemap)
        {
            AVA_ASSERT(dimension == 2, "[Texture] only 2D cubemaps are supported.");
            m_type = TextureType::TextureCube;
        }
        else if (m_layerCount > 1)
        {
            AVA_ASSERT(dimension < 3, "[Texture] 3D texture arrays are not supported.");
            m_type = dimension == 2 ? TextureType::Texture2DArray : TextureType::Texture1DArray;
        }
        else
        {
            m_type = dimension == 3 ? TextureType::Texture3D : dimension == 2 ? TextureType::Texture2D : TextureType::Texture1D;
        }

        m_mipCount = _data.mipCount;
        SetMipRange(0, m_mipCount - 1u);

        // Vulkan init
        m_vkFormat = ToVk(m_format);
        m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        m_aspect = IsDepth(m_format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

        const u8 minAlignment = static_cast<u8>( (IsCompressed(m_format) ? 16.f : 1.f) * BytesPerPixel(m_format) );
        m_alignment = Math::max((u8)4, minAlignment);

        _CreateImage();
        _CreateView();
        _UploadPixels(_data);

        const VkCommandBuffer cmd = VkGraphicsContext::BeginTransferCommand();
        TransitionToDefaultLayout(cmd);
        VkGraphicsContext::EndTransferCommand(cmd);
    }

    void VkTexture::_CreateImage()
    {
        VkImageCreateFlags createFlags = 0;
        VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        if (m_type == TextureType::TextureCube)
        {
            createFlags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        }
        if (HasFlag(AVA_TEXTURE_RENDER_TARGET))
        {
            usageFlags |= IsDepth(m_format) 
                ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT 
                : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
        if (HasFlag(AVA_TEXTURE_SAMPLED))
        {
            usageFlags |= VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        if (HasFlag(AVA_TEXTURE_READ_WRITE))
        {
            usageFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
        }

        const VkImageType vkImageType =
            Is1D(m_type) ? VK_IMAGE_TYPE_1D :
            Is2D(m_type) ? VK_IMAGE_TYPE_2D :
            VK_IMAGE_TYPE_3D;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = vkImageType;
        imageInfo.extent = { m_width, m_height, m_depth };
        imageInfo.format = m_vkFormat;
        imageInfo.flags = createFlags;
        imageInfo.usage = usageFlags;
        imageInfo.mipLevels = m_mipCount;
        imageInfo.arrayLayers = m_layerCount;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        if (HasFlag(AVA_TEXTURE_RENDER_TARGET))
        {
            allocInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        }

        const VkResult result = vmaCreateImage(VkGraphicsContext::GetAllocator(), &imageInfo, &allocInfo, &m_image, &m_alloc, nullptr);
        AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate image: %s", VkGraphicsContext::GetVkResultStr(result));
    }

    void VkTexture::_CreateView()
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_image;
        viewInfo.viewType = ToVk(m_type);
        viewInfo.format = m_vkFormat;
        viewInfo.subresourceRange.aspectMask = m_aspect;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = m_mipCount;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = m_layerCount;

        const VkResult result = vkCreateImageView(VkGraphicsContext::GetDevice(), &viewInfo, nullptr, &m_view);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create texture image view: %s", VkGraphicsContext::GetVkResultStr(result));
    }

    void VkTexture::_CreateSampler()
    {
        // Only creates sampler for sampled textures
        if (!HasFlag(AVA_TEXTURE_SAMPLED))
        {
            return;
        }

        const VkBool32 anisotropyEnabled = HasFlag(AVA_TEXTURE_ANISOTROPIC) ? VK_TRUE : VK_FALSE;
        const VkFilter filterMode = HasFlag(AVA_TEXTURE_LINEAR) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        const VkSamplerMipmapMode mipMode = HasFlag(AVA_TEXTURE_LINEAR) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;

        const VkSamplerAddressMode wrapMode =
            HasFlag(AVA_TEXTURE_MIRROR) ? VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT :
            HasFlag(AVA_TEXTURE_BORDER) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER :
            HasFlag(AVA_TEXTURE_CLAMP) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE :
            VK_SAMPLER_ADDRESS_MODE_REPEAT;
        
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = filterMode;
        samplerInfo.minFilter = filterMode;
        samplerInfo.addressModeU = wrapMode;
        samplerInfo.addressModeV = wrapMode;
        samplerInfo.addressModeW = wrapMode;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = false;
        samplerInfo.compareEnable = false;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = mipMode;
        samplerInfo.minLod = m_baseMipLevel;
        samplerInfo.maxLod = m_maxMipLevel;
        samplerInfo.mipLodBias = 0.f;
        samplerInfo.anisotropyEnable = anisotropyEnabled;
        samplerInfo.maxAnisotropy = VkGraphicsContext::GetDeviceLimits().maxSamplerAnisotropy;

        m_sampler = VkGraphicsContext::CreateSampler(&samplerInfo);
    }

    void VkTexture::_UploadPixels(const TextureData& _data)
    {
        const u8 mipCount = _data.mipCount;
        AVA_ASSERT(mipCount <= m_mipCount, "[Texture] data contains too much mips for this texture.");

        const u8 layerCount = _data.imageCount;
        AVA_ASSERT(layerCount <= m_layerCount, "[Texture] data contains too much layers for this texture.");

        std::vector<u32> mipSizes;
        mipSizes.resize(mipCount);

        // Computes transfer buffer size
        u32 totalBufferSize = 0;

        for (u8 mip = 0; mip < mipCount; mip++)
        {
            const u32 currentMipSize = _data.GetMipDataSize(0, mip);

            mipSizes[mip] = currentMipSize;
            totalBufferSize += currentMipSize * layerCount;

            // All copy operations must be aligned correctly in memory.
            totalBufferSize = GetAligned(totalBufferSize, m_alignment);
        }

        // Creates transfer buffer
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = totalBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCreateInfo{};
        allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer transferBuffer;
        VmaAllocation transferAlloc;
        VmaAllocationInfo allocInfo;
        const VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &transferBuffer, &transferAlloc, &allocInfo);
        AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate transfer buffer: %s", VkGraphicsContext::GetVkResultStr(result));

        // Fills buffer with mips data
        void* mappedData = allocInfo.pMappedData;
        u32 offset = 0;

        for (u32 mip = 0; mip < mipCount; mip++)
        {
            for (u32 layer = 0; layer < layerCount; layer++)
            {
                if (const void* pixels = _data.GetMipData(layer, mip))
                {
                    memcpy(static_cast<u8*>(mappedData) + offset, pixels, mipSizes[mip]);
                }
                else
                {
                    // avoid garbage, fill with 0
                    AVA_CORE_ERROR("[Texture] %s has no valid data to copy for mip %d.", GetResourcePath(), mip);
                    memset(static_cast<u8*>(mappedData) + offset, 0, mipSizes[mip]);
                }
                offset += mipSizes[mip];

                // All copy operations must be aligned correctly in memory.
                offset = GetAligned(offset, m_alignment);
            }
        }

        // Transfers data to texture image
        const VkCommandBuffer cmd = VkGraphicsContext::BeginTransferCommand();
        CopyBufferToImage(cmd, transferBuffer, mipSizes.data(), layerCount, mipCount);
        VkGraphicsContext::EndTransferCommand(cmd);

        // Destroys transfer buffer
        vmaDestroyBuffer(VkGraphicsContext::GetAllocator(), transferBuffer, transferAlloc);
    }

    void VkTexture::CopyBufferToImage(const VkCommandBuffer _cmd, const VkBuffer _buffer, const u32* _mipSizes, const u32 _layerCount, const u32 _mipCount)
    {
        std::vector<VkBufferImageCopy> copies;
        copies.resize(_mipCount);

        u32 offset = 0;
        u32 width = m_width;
        u32 height = m_height;
        u32 depth = m_depth;

        for (u32 mip = 0; mip < _mipCount; mip++)
        {
            VkBufferImageCopy& copyInfo = copies[mip];
            copyInfo.bufferImageHeight = 0;
            copyInfo.bufferRowLength = 0;
            copyInfo.bufferOffset = offset;

            copyInfo.imageExtent = { width, height, depth };
            copyInfo.imageOffset = { 0, 0, 0 };

            copyInfo.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyInfo.imageSubresource.baseArrayLayer = 0;
            copyInfo.imageSubresource.layerCount = _layerCount;
            copyInfo.imageSubresource.mipLevel = mip;

            offset += _mipSizes[mip] * _layerCount;

            // All copy operations must be aligned correctly in memory.
            offset = GetAligned((int)offset, m_alignment);

            width = Math::max(width >> 1, 1u);
            height = Math::max(height >> 1, 1u);
            depth = Math::max(depth >> 1, 1u);
        }

        // Performs memory transfer
        TransitionLayout(_cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(_cmd, _buffer, m_image, m_layout, _mipCount, copies.data());
        TransitionToDefaultLayout(_cmd);
    }

    void VkTexture::TransitionLayout(const VkCommandBuffer _cmd, const VkImageLayout _newLayout)
    {
        if (_newLayout == m_layout) {
            return;
        }

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = m_layout;
        barrier.newLayout = _newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_image;
        barrier.subresourceRange.aspectMask = m_aspect;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = m_mipCount;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = m_layerCount;

        VkPipelineStageFlags srcStages;
        VkPipelineStageFlags dstStages;

        // Get the source access mask and stage from the current layout.
        switch (m_layout)
        {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            barrier.srcAccessMask = 0;
            srcStages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            srcStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            srcStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
            break;
        case VK_IMAGE_LAYOUT_GENERAL: // only used when binding a storage image
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            srcStages =
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            break;
        default:
            AVA_ASSERT(false, "[Texture] unsupported image layout transition.");
            return;
        }

        // Get the destination access mask and stage from the target layout.
        switch (_newLayout)
        {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            dstStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            dstStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dstStages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dstStages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dstStages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dstStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            barrier.dstAccessMask = 0;
            dstStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            break;
        case VK_IMAGE_LAYOUT_GENERAL: // only used when binding a storage image
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            dstStages =
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            break;
        default:
            AVA_ASSERT(false, "[Texture] unsupported image layout transition.");
            return;
        }

        vkCmdPipelineBarrier(
            _cmd,
            srcStages,
            dstStages,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        m_layout = _newLayout;
    }

    VkImageLayout VkTexture::GetDefaultLayout() const
    {
        // swapchain image
        if (HasFlag(AVA_TEXTURE_DISPLAYABLE))
        {
            return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        }
        // storage texture
        if (HasFlag(AVA_TEXTURE_READ_WRITE))
        {
            return VK_IMAGE_LAYOUT_GENERAL;
        }
        // sampled texture
        if (HasFlag(AVA_TEXTURE_SAMPLED))
        {
            if (IsDepth(m_format))
            {
                return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            }
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        // render target
        if (HasFlag(AVA_TEXTURE_RENDER_TARGET))
        {
            if (IsDepth(m_format))
            {
                return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
    
        return VK_IMAGE_LAYOUT_GENERAL;
    }

    VkImageView VkTexture::GetMipView(const u16 _mip)
    {
        AVA_ASSERT(_mip < m_mipCount, "Invalid mip level.");

        // No mips, just use generic view
        if (m_mipCount == 1)
        {
            return m_view;
        }

        // Initializes the mip view array
        if (m_mipViews.size() < m_mipCount)
        {
            m_mipViews.resize(m_mipCount);
        }

        // Creates image view the first time
        if (!m_mipViews[_mip])
        {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_image;
            viewInfo.viewType = ToVk(m_type);
            viewInfo.format = m_vkFormat;
            viewInfo.subresourceRange.aspectMask = m_aspect;
            viewInfo.subresourceRange.baseMipLevel = _mip;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = m_layerCount;

            VkImageView imageView;
            const VkResult result = vkCreateImageView(VkGraphicsContext::GetDevice(), &viewInfo, nullptr, &imageView);
            AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create texture image view: %s", VkGraphicsContext::GetVkResultStr(result));

            m_mipViews[_mip] = imageView;
        }

        return m_mipViews[_mip];
    }

    VkImageView VkTexture::GetMipLayerView(const u16 _layer, const u16 _mip)
    {
        AVA_ASSERT(_mip < m_mipCount, "Invalid mip level.");

        // No layers, just use mip views
        if (m_layerCount == 1)
        {
            return GetMipView(_mip);
        }

        const std::pair key(_layer, _mip);

        // Creates image view the first time
        if (!m_mipLayerViews[key])
        {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_image;
            viewInfo.viewType = ToVk(m_type);
            viewInfo.format = m_vkFormat;
            viewInfo.subresourceRange.aspectMask = m_aspect;
            viewInfo.subresourceRange.baseMipLevel = _mip;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = _layer;
            viewInfo.subresourceRange.layerCount = 1;

            VkImageView imageView;
            const VkResult result = vkCreateImageView(VkGraphicsContext::GetDevice(), &viewInfo, nullptr, &imageView);
            AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create texture image view: %s", VkGraphicsContext::GetVkResultStr(result));

            m_mipLayerViews[key] = imageView;
        }

        return m_mipLayerViews[key];
    }

    VkSampler VkTexture::GetSampler()
    {
        AVA_ASSERT(HasFlag(AVA_TEXTURE_SAMPLED), "Texture is missing AVA_TEXTURE_SAMPLED flag.");

        if (m_samplerDirty)
        {
            _CreateSampler();
            m_samplerDirty = false;
        }

        return m_sampler;
    }

}