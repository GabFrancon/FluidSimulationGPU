#include <avapch.h>
#include "GraphicsEnums.h"

namespace Ava {

    namespace TextureType
    {
        static constexpr char const* kTypeNames[Count] = {
            "Texture 1D", "Texture 2D", "Texture 3D",
            "Texture 1D array", "Texture 2D array",
            "Cube map"
        };

        bool TextureType::Is1D(const Enum _type)
        {
            return _type == Texture1D || _type == Texture1DArray;
        }

        bool TextureType::Is2D(const Enum _type)
        {
            return _type == Texture2D || _type == Texture2DArray || _type == TextureCube;
        }

        bool TextureType::IsArray(const Enum _type)
        {
            return _type == Texture1DArray || _type == Texture2DArray;
        }

        Enum TextureType::Parse(const char* _typeStr)
        {
            for (u8 i = 0; i < Count; i++)
            {
                if (strcmp(kTypeNames[i], _typeStr) == 0)
                {
                    return Enum(i);
                }
            }
            return Count;
        }

        const char* TextureType::Stringify(const Enum _type)
        {
            return kTypeNames[_type < Count ? _type : 0];
        }
    }

    namespace TextureFormat
    {
        static constexpr char const* kFormatNames[Count] = {
            "Undefined",

            "R8", "RG8", "RGB8", "RGBA8",
            "R8_SINT", "RG8_SINT", "RGB8_SINT", "RGBA8_SINT",
            "R8_UINT", "RG8_UINT", "RGB8_UINT", "RGBA8_UINT",

            "R16", "RG16", "RGB16", "RGBA16",
            "R16_SINT", "RG16_SINT", "RGB16_SINT", "RGBA16_SINT",
            "R16_UINT", "RG16_UINT", "RGB16_UINT", "RGBA16_UINT",

            "R16F", "RG16F", "RGB16F", "RGBA16F",
            "R32F", "RG32F", "RGB32F", "RGBA32F",

            "BC1", "BC2", "BC3", "BC4", "BC5", "BC6F", "BC7",

            "D16", "D32F"
        };

        static constexpr float kBytesCount[Count] = {
            0.f,

            1.f, 2.f, 3.f, 4.f,   // RGBA8
            1.f, 2.f, 3.f, 4.f,   // RGBA8_SINT
            1.f, 2.f, 3.f, 4.f,   // RGBA8_UINT

            2.f, 4.f, 6.f, 8.f,   // RGBA16
            2.f, 4.f, 6.f, 8.f,   // RGBA16_SINT
            2.f, 4.f, 6.f, 8.f,   // RGBA16_UINT

            2.f, 4.f, 6.f, 8.f,   // RGBA16F
            4.f, 8.f, 12.f, 16.f, // RGBA32F

            0.5f, 1.f, 1.f,       // BC 1-2-3
            0.5f, 1.f, 1.f, 1.f,  // BC 4-5-6F-7

            2.f, 4.f              // DEPTH
        };

        static constexpr u8 kChannelsCount[Count] = {
            0,

            1, 2, 3, 4,   // RGBA8
            1, 2, 3, 4,   // RGBA8_SINT
            1, 2, 3, 4,   // RGBA8_UINT

            1, 2, 3, 4,   // RGBA16
            1, 2, 3, 4,   // RGBA16_SINT
            1, 2, 3, 4,   // RGBA16_UINT

            1, 2, 3, 4,   // RGBA16F
            1, 2, 3, 4,   // RGBA32F

            4, 4, 4,      // BC 1-2-3
            1, 2, 3, 4,   // BC 4-5-6F-7

            1, 1          // DEPTH
        };

        bool TextureFormat::IsDepth(const Enum _format)
        {
            return _format == D16 || _format == D32F;
        }

        bool IsCompressed(const Enum _format)
        {
            return _format >= BC1 && _format <= BC7;
        }

        float TextureFormat::BytesPerPixel(const Enum _format)
        {
            return kBytesCount[_format < Count ? _format : Undefined];
        }

        u8 TextureFormat::ChannelCount(const Enum _format)
        {
            return kChannelsCount[_format < Count ? _format : Undefined];
        }

        Enum TextureFormat::Parse(const char* _formatStr)
        {
            for (u8 i = 0; i < Count; i++)
            {
                if (strcmp(kFormatNames[i], _formatStr) == 0)
                {
                    return Enum(i);
                }
            }
            return Undefined;
        }

        const char* TextureFormat::Stringify(const Enum _format)
        {
            return kFormatNames[_format < Count ? _format : Undefined];
        }
    };

    namespace DataType
    {
        static constexpr char const* kTypeNames[Count] = {
            "SINT8", "SINT16", "SINT32", "UINT8", "UINT16", "UINT32",
            "SNORM8", "SNORM16", "UNORM8", "UNORM16", "FLOAT16", "FLOAT32"
        };

        static constexpr u8 kBytesCount[Count] = {
            1, 2, 4,  // SINT 8-16-32
            1, 2, 4,  // UINT 8-16-32
            1, 2,     // SNORM 8-16
            1, 2,     // UNORM 8-16
            2, 4      // FLOAT 16-32
        };

        u8 DataType::BytesCount(const Enum _type)
        {
            return kBytesCount[_type < Count ? _type : 0];
        }

        DataType::Enum Parse(const char* _typeStr)
        {
            for (u8 i = 0; i < Count; i++)
            {
                if (strcmp(kTypeNames[i], _typeStr) == 0)
                {
                    return Enum(i);
                }
            }
            return UNORM8;
        }

        const char* Stringify(const Enum _type)
        {
            return kTypeNames[_type < Count ? _type : 0];
        }
    };

    namespace VertexSemantic
    {
        static constexpr char const* kSemanticNames[Count] = {
            "aPosition", "aNormal", "aTangent", "aTexCoord",
            "aBoneIndices", "aBoneWeights", "aColor"
        };

        Enum VertexSemantic::Parse(const char* _semanticStr)
        {
            for (u8 i = 0; i < Count; i++)
            {
                if (strcmp(kSemanticNames[i], _semanticStr) == 0)
                {
                    return Enum(i);
                }
            }
            return Count;
        }

        const char* VertexSemantic::Stringify(const Enum _semantic)
        {
            return kSemanticNames[_semantic < Count ? _semantic : 0];
        }
    };

    namespace ShaderStage
    {
        static constexpr char const* kStageNames[Count] = {
            "Vertex", "Geometric", "Fragment", "Compute"
        };

        Enum ShaderStage::Parse(const char* _stageStr)
        {
            for (u8 i = 0; i < Count; i++)
            {
                if (strcmp(kStageNames[i], _stageStr) == 0)
                {
                    return Enum(i);
                }
            }
            return Count;
        }

        const char* ShaderStage::Stringify(const Enum _stage)
        {
            return kStageNames[_stage < Count ? _stage : 0];
        }
    }

}
