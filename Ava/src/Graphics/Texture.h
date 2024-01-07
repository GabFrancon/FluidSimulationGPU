#pragma once
/// @file Texture.h
/// @brief interface for GPU images.

#include <Resources/Resource.h>
#include <Resources/TextureData.h>

namespace Ava {

    enum TextureFlags
    {
        AVA_TEXTURE_NONE = 0,
        // texture usage
        AVA_TEXTURE_SAMPLED = AVA_BIT(0),
        AVA_TEXTURE_READ_WRITE = AVA_BIT(1),
        AVA_TEXTURE_RENDER_TARGET = AVA_BIT(2),
        AVA_TEXTURE_DISPLAYABLE = AVA_BIT(3),
        // sampling wrap mode
        AVA_TEXTURE_REPEAT = AVA_BIT(4),
        AVA_TEXTURE_MIRROR = AVA_BIT(5),
        AVA_TEXTURE_CLAMP = AVA_BIT(6),
        AVA_TEXTURE_BORDER = AVA_BIT(7),
        // sampling filter mode
        AVA_TEXTURE_NEAREST = AVA_BIT(8),
        AVA_TEXTURE_LINEAR = AVA_BIT(9),
        AVA_TEXTURE_ANISOTROPIC = AVA_BIT(10),
    };

    #define AVA_TEXTURE_SAMPLER_MASK ( \
            AVA_TEXTURE_REPEAT | AVA_TEXTURE_MIRROR | AVA_TEXTURE_CLAMP | AVA_TEXTURE_BORDER | \
            AVA_TEXTURE_NEAREST | AVA_TEXTURE_LINEAR | AVA_TEXTURE_ANISOTROPIC) \

    /// @brief Used to create a blank texture, e.g a render target.
    struct TextureDescription
    {
        u16 width = 0;
        u16 height = 0;
        u16 depth = 1;
        u16 layerCount = 1;
        u16 mipCount = 1;
        std::string debugName;
        u32 flags = AVA_TEXTURE_NONE;
        TextureFormat::Enum format = TextureFormat::Undefined;

        bool operator==(const TextureDescription& _other) const;
        bool operator!=(const TextureDescription& _other) const;
        u32 Hash() const;
    };

    /// @brief Resource or render target texture object.
    class Texture : public Resource
    {
        friend class TextureResourceMgr;

    public:
        Texture() = default;
        ~Texture() override = default;

        /// @brief for render target textures.
        explicit Texture(const TextureDescription& _desc);
        /// @brief for path-based texture resources.
        explicit Texture(const char* _path, u32 _flags = 0);
        /// @brief for built-in texture data.
        explicit Texture(const TextureData& _data, u32 _flags = 0);

        virtual void SetDebugName(const char* _name) { }
        const char* GetResourcePath() const { return GetResourceName(); }

        // Loading is handled by resource manager to stay sync with graphics context.
        void Load() override {}
        void Release() override {}

        // dimensions
        u16 GetWidth() const { return m_width; }
        u16 GetHeight() const { return m_height; }
        u16 GetDepth() const { return m_depth; }
        u16 GetLayerCount() const { return m_layerCount; }

        // mip maps
        u8 GetMipCount() const { return m_mipCount; }
        u8 GetBaseMipLevel() const { return m_baseMipLevel; }
        u8 GetMaxMipLevel() const { return m_maxMipLevel; }

        u16 GetMipWidth(u16 _mip) const;
        u16 GetMipHeight(u16 _mip) const;
        void SetMipRange(u8 _baseMipLevel, u8 _maxMipLevel);

        // flags
        u32 GetFlags() const { return m_flags; }
        bool HasFlag(const TextureFlags _flag) const { return m_flags & _flag; }
        void SetSamplerState(u32 _samplerFlags);

        // type and format
        TextureType::Enum GetType() const { return m_type; }
        TextureFormat::Enum GetFormat() const { return m_format; }
        int GetMemorySize() const;

    protected:
        u16 m_width = 0;
        u16 m_height = 0;
        u16 m_depth = 1;
        u16 m_layerCount = 1;

        u8 m_mipCount = 1;
        u8 m_baseMipLevel = 0;
        u8 m_maxMipLevel = 1;

        TextureType::Enum m_type = TextureType::Texture2D;
        TextureFormat::Enum m_format = TextureFormat::Undefined;
        u32 m_flags = AVA_TEXTURE_NONE;

        bool m_samplerDirty = true;
    };

    /// @brief Custom texture resource manager.
    class TextureResourceMgr final : public ResourceMgr<Texture, char>
    {
    public:
        Resource* CreateResource(const void* _constructionData) override;
        void DestroyResource(Resource* _resource) override;
        ResourceID BuildResourceID(const void* _constructionData) override;
        bool HotReloadResource(ResourceID _resourceID) override;
    };

    /// @brief Texture resource helper, to handle path based Texture resources.
    class TextureResH : public ResH<TextureResourceMgr>
    {
    };
}

