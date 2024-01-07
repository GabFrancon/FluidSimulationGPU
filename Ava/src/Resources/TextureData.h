#pragma once
/// @file TextureData.h
/// @brief

#include <Graphics/GraphicsEnums.h>

namespace Ava {

    // Corresponds to a full mip-mapped 4K texture.
    static constexpr u32 kMaxTextureMipLevels = 13;

    /// @brief Stores raw data to fill a Texture object.
    struct TextureData
    {
        const void* GetMipData(u8 _image, u8 _mip) const;
        u32 GetMipDataSize(u8 _image, u8 _mip) const;
        void* GetMipData(u8 _image, u8 _mip);

        char resourceName[MAX_PATH];
        TextureFormat::Enum format = TextureFormat::Undefined;

        u16 width = 0;
        u16 height = 0;
        u16 depth = 1;
        u8 mipCount = 1;
        u8 imageCount = 1;
        bool isCubemap = false;

        struct Image
        {
            // not allocated, just a pointer to the start
            // of each mip in the textureData shared memory
            void* mipPixels[kMaxTextureMipLevels];
            u32 mipSizes[kMaxTextureMipLevels];
        };
        std::vector<Image> images;

        // raw binaries
        void* binData = nullptr;
        u32 binSize = 0;
    };

    /// @brief Generic texture loader.
    class TextureLoader
    {
    public:
        static bool Load(const char* _path, TextureData& _data);
        static void Release(TextureData& _data);
    };


    // ----- .DDS texture files ------------------------------------

    /// @brief Static helper to deal with .DDS images.
    class DDSTextureLoader final : public TextureLoader
    {
    public:
        static bool Load(const char* _path, TextureData& _data);
    };


    // ----- Default loader using stb image ------------------------

    /// @brief Static helper to deal with default image formats.
    class STBTextureLoader final : public TextureLoader
    {
    public:
        static bool Load(const char* _path, TextureData& _data);
    };


}