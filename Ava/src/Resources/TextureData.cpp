#include <avapch.h>
#include "TextureData.h"

#include <Files/File.h>
#include <Files/FilePath.h>
#include <Strings/StringBuilder.h>
#include <Resources/DDS.h>
#include <Memory/Memory.h>
#include <Math/Math.h>
#include <Debug/Log.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <LZ4/lz4.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    // ---- Texture data -------------------------------------------------

    const void* TextureData::GetMipData(const u8 _image, const u8 _mip) const
    {
        if (_image < imageCount && _mip < mipCount)
        {
            return images[_image].mipPixels[_mip];
        }
        return nullptr;
    }

    u32 TextureData::GetMipDataSize(const u8 _image, const u8 _mip) const
    {
        if (_image < imageCount && _mip < mipCount)
        {
            return images[_image].mipSizes[_mip];
        }
        return 0u;
    }

    void* TextureData::GetMipData(const u8 _image, const u8 _mip)
    {
        if (_image < imageCount && _mip < mipCount)
        {
            return images[_image].mipPixels[_mip];
        }
        return nullptr;
    }


    // ---- Texture loader -----------------------------------------------

    bool TextureLoader::Load(const char* _path, TextureData& _data)
    {
        // redirects to specific format texture loader
        const auto* extension = FilePath::FindExtension(_path);

        if (strcmp(extension, ".dds") == 0)
        {
            return DDSTextureLoader::Load(_path, _data);
        }

        return STBTextureLoader::Load(_path, _data);
    }

    void TextureLoader::Release(TextureData& _data)
    {
        if (_data.binData)
        {
            // free the shader memory
            AVA_FREE(_data.binData);
            _data.binData = nullptr;
            _data.binSize = 0u;

            // nullify all image and mip pointers
            for (u8 i = 0; i < _data.imageCount; i++)
            {
                for (u8 j = 0; j < _data.mipCount; j++)
                {
                    _data.images[i].mipPixels[j] = nullptr;
                    _data.images[i].mipSizes[j] = 0u;
                }
            }
        }
    }


    // ---- Texture loader for .DDS --------------------------------------

    static TextureFormat::Enum GetDDSFormat(const DDS_HEADER& _ddsHeader)
    {
        // gets format from dds
        const DDS_PIXELFORMAT &pf = _ddsHeader.ddspf;

        // compressed format
        if (pf.dwFlags & DDS_FOURCC) 
        {
            switch (_ddsHeader.ddspf.dwFourCC) 
            {
                case MAKEFOURCC('D','X','T','1'):
                case MAKEFOURCC('D','X','T','2'):
                    return TextureFormat::BC1;

                case MAKEFOURCC('D','X','T','3'):
                case MAKEFOURCC('D','X','T','4'):
                    return TextureFormat::BC2;

                case MAKEFOURCC('D','X','T','5'):
                    return TextureFormat::BC3;

                case MAKEFOURCC('B','C','4','U'):
                    return TextureFormat::BC4;

                case MAKEFOURCC('B','C','5','U'):
                    return TextureFormat::BC5;

                case MAKEFOURCC('B','C','7','U'):
                    return TextureFormat::BC7;

            default:
                AVA_ASSERT(false, "[Texture] DDS compressed format not supported.");
                return TextureFormat::Undefined;
            };
        }

        // uncompressed format
        if (pf.dwFlags == DDS_RGBA && pf.dwRGBBitCount == 32)
            return TextureFormat::RGBA8;

        if (pf.dwFlags == DDS_RGB && pf.dwRGBBitCount == 32)
            return TextureFormat::RGBA8;

        if (pf.dwFlags == DDS_RGB && pf.dwRGBBitCount == 24)
            return TextureFormat::RGB8;

        if (pf.dwRGBBitCount == 16)
            return TextureFormat::RG8;

        if (pf.dwRGBBitCount == 8)
            return TextureFormat::R8;

        AVA_ASSERT(false, "[Texture] DDS format not supported.");
        return TextureFormat::Undefined;
    }

    static TextureFormat::Enum GetDXT10Format(const DDS_HEADER_DXT10& _dxt10Header)
    {
        // gets format from DX10 header
        switch (_dxt10Header.dxgiFormat) 
        {
            case DXGI_FORMAT_R8_UNORM:                 return TextureFormat::R8;
            case DXGI_FORMAT_R8G8_UNORM:               return TextureFormat::RG8;
            case DXGI_FORMAT_R8G8B8A8_UNORM:           return TextureFormat::RGBA8;

            case DXGI_FORMAT_R8_SINT:                  return TextureFormat::R8_SINT;
            case DXGI_FORMAT_R8G8_SINT:                return TextureFormat::RG8_SINT;
            case DXGI_FORMAT_R8G8B8A8_SINT:            return TextureFormat::RGBA8_SINT;

            case DXGI_FORMAT_R8_UINT:                  return TextureFormat::R8_UINT;
            case DXGI_FORMAT_R8G8_UINT:                return TextureFormat::RG8_UINT;
            case DXGI_FORMAT_R8G8B8A8_UINT:            return TextureFormat::RGBA8_UINT;

            case DXGI_FORMAT_R16_UNORM:                return TextureFormat::R16;
            case DXGI_FORMAT_R16G16_UNORM:             return TextureFormat::RG16;
            case DXGI_FORMAT_R16G16B16A16_UNORM:       return TextureFormat::RGBA16;

            case DXGI_FORMAT_R16_SINT:                 return TextureFormat::R16_SINT;
            case DXGI_FORMAT_R16G16_SINT:              return TextureFormat::RG16_SINT;
            case DXGI_FORMAT_R16G16B16A16_SINT:        return TextureFormat::RGBA16_SINT;

            case DXGI_FORMAT_R16_UINT:                 return TextureFormat::R16_UINT;
            case DXGI_FORMAT_R16G16_UINT:              return TextureFormat::RG16_UINT;
            case DXGI_FORMAT_R16G16B16A16_UINT:        return TextureFormat::RGBA16_UINT;

            case DXGI_FORMAT_R16_FLOAT:                return TextureFormat::R16F;
            case DXGI_FORMAT_R16G16_FLOAT:             return TextureFormat::RG16F;
            case DXGI_FORMAT_R16G16B16A16_FLOAT:       return TextureFormat::RGBA16F;

            case DXGI_FORMAT_R32_FLOAT:                return TextureFormat::R32F;
            case DXGI_FORMAT_R32G32_FLOAT:             return TextureFormat::RG32F;
            case DXGI_FORMAT_R32G32B32_FLOAT:          return TextureFormat::RGB32F;
            case DXGI_FORMAT_R32G32B32A32_FLOAT:       return TextureFormat::RGBA32F;

            case DXGI_FORMAT_BC1_UNORM:                return TextureFormat::BC1;
            case DXGI_FORMAT_BC2_UNORM:                return TextureFormat::BC2;
            case DXGI_FORMAT_BC3_UNORM:                return TextureFormat::BC3;
            case DXGI_FORMAT_BC4_UNORM:                return TextureFormat::BC4;
            case DXGI_FORMAT_BC5_UNORM:                return TextureFormat::BC5;
            case DXGI_FORMAT_BC6H_SF16:                return TextureFormat::BC6F;
            case DXGI_FORMAT_BC7_UNORM:                return TextureFormat::BC7;

        default:
            AVA_ASSERT(false, "[Texture] DXGI format not supported.");
            return TextureFormat::Undefined;
        };
    }

    static u32 GetImageSize(const u32 _width, const u32 _height, const u32 _depth, const TextureFormat::Enum _format)
    {
        switch (_format) 
        {
            // BC1-BC4 use 8 bits per 4x4 texel block
            case TextureFormat::BC1:
            case TextureFormat::BC4:
                return Math::max(1u, ((_width + 3) / 4) * ((_height + 3) / 4) * _depth * 8);

            // BC2-BC3-BC5-BC6F-BC7 use 16 bits per 4x4 texel block
            case TextureFormat::BC2:
            case TextureFormat::BC3:
            case TextureFormat::BC5:
            case TextureFormat::BC6F:
            case TextureFormat::BC7:
                return Math::max(1u, ((_width + 3) / 4) * ((_height + 3) / 4) * _depth * 16);

            default: 
                return static_cast<u32>(_width * _height * _depth * BytesPerPixel(_format));
        };
    }

    bool DDSTextureLoader::Load(const char* _path, TextureData& _data)
    {
        // loads binary file
        File binFile;
        if (!binFile.Open(_path, AVA_FILE_READ | AVA_FILE_BINARY))
        {
            AVA_ERROR(binFile.GetErrorStr());
            return false;
        }

        StrFormat(_data.resourceName, _path);

        unsigned long type;
        binFile.Read(type);

        if (type != DDS_MAGIC)
        {
            AVA_ERROR("[Texture] '%s' is not formatted as a DDS file.", _path);
            return false;
        }

        DDS_HEADER ddsHeader;
        binFile.Read(ddsHeader);

        _data.width = ddsHeader.dwWidth > 0 ? (u16)ddsHeader.dwWidth : 1;
        _data.height = ddsHeader.dwHeight > 0 ? (u16)ddsHeader.dwHeight : 1;
        _data.depth = ddsHeader.dwDepth  > 0 ? (u16)ddsHeader.dwDepth : 1;
        _data.mipCount = ddsHeader.dwMipMapCount > 0 ? (u8)ddsHeader.dwMipMapCount : 1;
        _data.isCubemap = ddsHeader.dwCubemapFlags & DDS_CUBEMAP_ALLFACES;
        _data.imageCount = _data.isCubemap ? 6 : 1;

        if ((ddsHeader.ddspf.dwFlags & DDS_FOURCC) && (MAKEFOURCC('D','X','1','0') == ddsHeader.ddspf.dwFourCC))
        {
            DDS_HEADER_DXT10 dxt10Header;
            binFile.Read(dxt10Header);

            if (!_data.isCubemap && dxt10Header.arraySize > 0)
            {
                _data.imageCount = (u8)dxt10Header.arraySize;
            }

            _data.format = GetDXT10Format(dxt10Header);
        }
        else
        {
            _data.format = GetDDSFormat(ddsHeader);
        }

        bool isBgr = false;
        if (_data.format == TextureFormat::RGB8 || _data.format == TextureFormat::RGBA8)
        {
            isBgr = ddsHeader.ddspf.dwBBitMask == 0x000000ff;
        }

        // gets raw pixel data
        _data.binSize = binFile.GetSize() - binFile.GetCursor();
        _data.binData = AVA_MALLOC(_data.binSize);
        binFile.Read(_data.binData, _data.binSize);

        // finds mips for each image
        _data.images.resize(_data.imageCount);
        auto* pixels = static_cast<u8*>(_data.binData);

        for (auto& image : _data.images)
        {
            u32 width = _data.width;
            u32 height = _data.height;
            u32 depth = _data.depth;

            for (u32 mip = 0; mip < _data.mipCount; ++mip) 
            {
                const u32 mipSize = GetImageSize(width, height, depth, _data.format);

                image.mipPixels[mip] = pixels;
                image.mipSizes[mip] = mipSize;

                // switches BGR to RGB
                if (isBgr) 
                {
                    const u32 inc = (u32)BytesPerPixel(_data.format);

                    for (u32 k = 0 ; k < mipSize; k += inc)
                    {
                        const u8 r = pixels[k + 0];
                        const u8 b = pixels[k + 2];
                        pixels[k + 0] = b;
                        pixels[k + 2] = r;
                    }
                }

                width = Math::max(width >> 1, 1u);
                height = Math::max(height >> 1, 1u);
                depth = Math::max(depth >> 1, 1u);
                pixels += mipSize;
            }
        }

        binFile.Close();
        return true;
    }


    // ---- Default texture loader ----------------------------------------

    bool STBTextureLoader::Load(const char* _path, TextureData& _data)
    {
        StrFormat(_data.resourceName, _path);

        int width, height, channels;
        _data.binData = stbi_load(_path, &width, &height, &channels, STBI_rgb_alpha);
        _data.binSize = static_cast<u32>(width * height * 4);

        _data.width = static_cast<u16>(width);
        _data.height = static_cast<u16>(height);
        _data.format = TextureFormat::RGBA8;

        _data.depth = 1u;
        _data.mipCount = 1u;
        _data.imageCount = 1u;

        auto& image = _data.images.emplace_back();
        image.mipSizes[0] = _data.binSize;
        image.mipPixels[0] = _data.binData;

        return true;
    }

}
