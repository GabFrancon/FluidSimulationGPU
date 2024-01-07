#include <avapch.h>
#include "Texture.h"

#include <Graphics/GraphicsContext.h>
#include <Math/Math.h>
#include <Debug/Log.h>

namespace Ava {

    //-------------- Texture description implementation ---------------------------------------------------------------

    bool TextureDescription::operator==(const TextureDescription& _other) const
    {
        return width == _other.width
            && height == _other.height
            && depth == _other.depth
            && layerCount == _other.layerCount
            && mipCount == _other.mipCount
            && format == _other.format
            && flags == _other.flags;
    }

    bool TextureDescription::operator!=(const TextureDescription& _other) const
    {
        return !(*this == _other);
    }

    u32 TextureDescription::Hash() const
    {
        u32 hash = HashU32(width);
        hash = HashU32Combine(height, hash);
        hash = HashU32Combine(depth, hash);
        hash = HashU32Combine(layerCount, hash);
        hash = HashU32Combine(mipCount, hash);
        hash = HashU32Combine(flags, hash);
        hash = HashU32Combine(format, hash);
        return hash;
    }


    //-------------- Texture implementation ---------------------------------------------------------------------------

    Texture::Texture(const TextureDescription& _desc)
    {
        SetResourceName(_desc.debugName.c_str());

        m_width = _desc.width;
        m_height = _desc.height;
        m_depth = _desc.depth;
        m_format = _desc.format;
        m_flags = _desc.flags;
        m_layerCount = _desc.layerCount;

        // Deduce texture type
        const int dimension = _desc.depth > 1 ? 3 : (_desc.width > 1 && _desc.height > 1) ? 2 : 1;

        if (_desc.layerCount > 1)
        {
            AVA_ASSERT(dimension < 3, "[Texture] 3D texture arrays are not supported.");
            m_type = dimension == 2 ? TextureType::Texture2DArray : TextureType::Texture1DArray;
        }
        else
        {
            m_type = dimension == 3 ? TextureType::Texture3D : dimension == 2 ? TextureType::Texture2D : TextureType::Texture1D;
        }

        // Set texture mip range
        m_mipCount = (u8)_desc.mipCount;
        SetMipRange(0, m_mipCount - 1);
    }

    Texture::Texture(const char* _path, const u32 _flags/*= 0*/)
    {
        AVA_ASSERT(_path, "[Texture] trying to load an invalid texture resource.");
        SetResourceName(_path);
        m_flags = _flags;
    }

    Texture::Texture(const TextureData& _data, const u32 _flags/*= 0*/)
    {
        SetResourceName(_data.resourceName);
        SetResourceID(HashStr(_data.resourceName));
        m_flags = _flags;
    }

    u16 Texture::GetMipWidth(const u16 _mip) const
    {
        const u16 minWidth = IsCompressed(m_format) ? 4 : 1;
        return Math::max(static_cast<u16>(m_width >> _mip), minWidth);
    }

    u16 Texture::GetMipHeight(const u16 _mip) const
    {
        const u16 minHeight = IsCompressed(m_format) ? 4 : 1;
        return Math::max(static_cast<u16>(m_height >> _mip), minHeight);
    }

    void Texture::SetMipRange(const u8 _baseMipLevel, const u8 _maxMipLevel)
    {
        AVA_ASSERT(_maxMipLevel < m_mipCount, "Maximum mip level exceeds texture mip count.");

        m_baseMipLevel = _baseMipLevel;
        m_maxMipLevel = _maxMipLevel;
        m_samplerDirty = true;
    }

    void Texture::SetSamplerState(const u32 _samplerFlags)
    {
        // Don't try to set bits that are not in the mask
        AVA_ASSERT((_samplerFlags & ~AVA_TEXTURE_SAMPLER_MASK) == 0, "Invalid sampler flags");

        // Clear previous sampler flags
        m_flags &= ~AVA_TEXTURE_SAMPLER_MASK;

        // Set new sampler flags
        m_flags |= (_samplerFlags & AVA_TEXTURE_SAMPLER_MASK);
        m_samplerDirty = true;
    }

    int Texture::GetMemorySize() const
    {
        const float bytesPerPixel = BytesPerPixel(m_format);

        int width = m_width;
        int height = m_height;
        int pixelCount = 0;

        for (int i = 0; i < m_mipCount; ++i)
        {
            // Round up dimensions to a multiple of 4.
            // That's a good enough approximation of padding added in reality.
            const int w = (width + 3) / 4 * 4;
            const int h = (height + 3) / 4 * 4;

            pixelCount += w * h;

            // Calculate dimensions of next mipmap
            if (width > 1)
                width /= 2;

            if (height > 1)
                height /= 2;
        }

        pixelCount = pixelCount * m_layerCount * m_depth;

        const int size = (int)floor(pixelCount * bytesPerPixel);
        return size;
    }


    //-------------- TextureResourceMgr implementation ----------------------------------------------------------------

    Resource* TextureResourceMgr::CreateResource(const void* _constructionData)
    {
        const auto* texturePath = static_cast<const char*>(_constructionData);
        Texture* texture = GraphicsContext::CreateTexture(texturePath, AVA_TEXTURE_SAMPLED);
        return texture;
    }

    void TextureResourceMgr::DestroyResource(Resource* _resource)
    {
        auto* texture = static_cast<Texture*>(_resource);
        GraphicsContext::DestroyTexture(texture);
    }

    ResourceID TextureResourceMgr::BuildResourceID(const void* _constructionData)
    {
        const auto* texturePath = static_cast<const char*>(_constructionData);
        return HashStr(texturePath);
    }

    bool TextureResourceMgr::HotReloadResource(const ResourceID _resourceID)
    {
        // Checks if resource exists.
        auto* texture = static_cast<Texture*>(GetResource(_resourceID));
        if (!texture)
        {
            AVA_CORE_WARN("[Texture] No texture with ID '%u' found.", _resourceID);
            return false;
        }

        // Saves texture description.
        const std::string texturePath = texture->GetResourcePath();
        const u32 textureFlags = texture->GetFlags();

        // Waits to sync with graphics context.
        GraphicsContext::WaitIdle();
        
        // Reconstructs texture resource.
        ReleaseResource(texture);
        GraphicsContext::DestructTexture(texture);
        GraphicsContext::ConstructTexture(texture, texturePath.c_str(), textureFlags);
        texture->SetResourceID(_resourceID);
        LoadResource(texture);
        
        return true;
    }

}
