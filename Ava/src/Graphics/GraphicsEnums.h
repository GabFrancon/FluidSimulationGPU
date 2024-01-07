#pragma once
/// @file GraphicsEnums.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    namespace TextureType
    {
        enum Enum : u8 {
            Texture1D,
            Texture2D,
            Texture3D,
            Texture1DArray,
            Texture2DArray,
            TextureCube,

            Count
        };

        bool Is1D(TextureType::Enum _type);
        bool Is2D(TextureType::Enum _type);
        bool IsArray(TextureType::Enum _type);
        TextureType::Enum Parse(const char* _typeStr);
        const char* Stringify(TextureType::Enum _type);
    }

    namespace TextureFormat
    {
        enum Enum : u8 {
            // Undefined format
            Undefined,

            // 8-bit unsigned normalized format  [ 0 ... 1 ]
            R8,
            RG8,
            RGB8,
            RGBA8,

            // 8-bit signed integer format       [|0 ... n|]
            R8_SINT,
            RG8_SINT,
            RGB8_SINT,
            RGBA8_SINT,

            // 8-bit unsigned integer format     [|-n .. n|]
            R8_UINT,
            RG8_UINT,
            RGB8_UINT,
            RGBA8_UINT,

            // 16-bit unsigned normalized format [ 0 ... 1 ]
            R16,
            RG16,
            RGB16,
            RGBA16,

            // 16-bit signed integer formats     [|0 ... n|]
            R16_SINT,
            RG16_SINT,
            RGB16_SINT,
            RGBA16_SINT,

            // 16-bit unsigned integer format    [|-n .. n|]
            R16_UINT,
            RG16_UINT,
            RGB16_UINT,
            RGBA16_UINT,

            // 16-bit floating points formats    [ -x .. x ]
            R16F,
            RG16F,
            RGB16F,
            RGBA16F,

            // 32-bit floating points formats    [ -x .. x ]
            R32F,
            RG32F,
            RGB32F,
            RGBA32F,

            // block compressed formats
            BC1,  // RGB unorm (color maps)
            BC2,  // RGB unorm + 4-bit alpha (n/a)
            BC3,  // RGBA unorm (color alpha maps)
            BC4,  // R unorm (gray scale maps)
            BC5,  // RG unorm (normal maps)
            BC6F, // RGB floating points (HDR images)
            BC7,  // RGBA unorm (high-quality color maps)

            // depth formats
            D16,  // 16-bit unorm (mid-quality depth buffer)
            D32F, // 32-bit floating points (high-quality depth buffer)

            Count
        };

        bool IsDepth(TextureFormat::Enum _format);
        bool IsCompressed(TextureFormat::Enum _format);
        float BytesPerPixel(TextureFormat::Enum _format);
        u8 ChannelCount(TextureFormat::Enum _format);
        TextureFormat::Enum Parse(const char* _formatStr);
        const char* Stringify(TextureFormat::Enum _format);
    };

    namespace DataType
    {
        enum Enum : u8 {
            SINT8,
            SINT16,
            SINT32,

            UINT8,
            UINT16,
            UINT32,

            SNORM8,
            SNORM16,

            UNORM8,
            UNORM16,

            FLOAT16,
            FLOAT32,

            Count
        };

        u8 BytesCount(DataType::Enum _type);
        DataType::Enum Parse(const char* _typeStr);
        const char* Stringify(DataType::Enum _type);
    };

    namespace VertexSemantic
    {
        enum Enum : u8 {
            Position,
            Normal,
            Tangent,
            TexCoord,
            BoneIndices,
            BoneWeights,
            Color,

            Count
        };

        VertexSemantic::Enum Parse(const char* _semanticStr);
        const char* Stringify(VertexSemantic::Enum _semantic);
    };

    namespace ShaderStage
    {
        enum Enum : u8 {
            Vertex,
            Geometric,
            Fragment,
            Compute,

            DrawCount = Compute,
            Count
        };

        ShaderStage::Enum Parse(const char* _stageStr);
        const char* Stringify(ShaderStage::Enum _stage);
    };

    namespace PrimitiveType
    {
        enum Enum : u8 {
            Triangle,
            TriangleStrip,
            Line,
            LineStrip,
            Point,

            Count
        };
    };

    namespace CullMode
    {
        enum Enum : u8 {
            None,
            Front,
            Back,
            FrontBack,

            Count
        };
    };

    namespace CompareFunc
    {
        enum Enum : u8 {
            Never,
            NotEqual,
            Equal,
            Less,
            LessEqual,
            Greater,
            GreaterEqual,
            Always,

            Count
        };
    }

    namespace BlendFunc
    {
        enum Enum : u8 {
            Add,
            Subtract,
            Min,
            Max,

            Count
        };
    };

    namespace BlendFactor
    {
        enum Enum : u8 {
            Zero,
            One,

            SrcColor,
            OneMinusSrcColor,
            DstColor,
            OneMinusDstColor,

            SrcAlpha,
            OneMinusSrcAlpha,
            DstAlpha,
            OneMinusDstAlpha,

            Count
        };
    };

    namespace Access {

        enum Enum : u8 {
            Read,
            Write,
            ReadWrite,

            Count
        };
    }

    namespace CommandType {

        enum Enum : u8 {
            Draw,
            DrawIndexed,
            Dispatch,

            Count
        };
    }

    namespace BufferType {

        enum Enum : u8 {
            Constant,
            Indirect,
            Vertex,
            Index
        };

    }

}