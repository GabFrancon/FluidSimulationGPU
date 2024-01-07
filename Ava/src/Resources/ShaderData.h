#pragma once
/// @file ShaderData.h
/// @brief

#include <Graphics/GraphicsEnums.h>

namespace Ava {

    /// @brief Stores raw data to fill a Shader object.
    struct ShaderData
    {
        std::string resourceName;
        ShaderStage::Enum stage = ShaderStage::Count;

        u32 bindingsCount = 0;
        u32 attributesCount = 0;
        u32 colorOutputsCount = 0;

        // not allocated, just a pointer to the start of the
        // shader code in the shaderData shared memory
        u32* code = nullptr;
        u32 codeSize = 0;

        // not allocated, just a pointer to the start of the
        // reflection data in the shaderData shared memory
        char* reflectionData = nullptr;
        u32 reflectionDataSize = 0;

        // raw binaries
        void* binData = nullptr;
        u32 binSize = 0;

        SerializeError Serialize(Serializer& _serializer, const char* _tag);
        SerializeError SerializeHeaderOnly(Serializer& _serializer, const char* _tag);
    };

    /// @brief Static helper to deal with .SHD binary files.
    class ShaderLoader
    {
    public:
        static bool Load(const char* _path, ShaderData& _data);
        static bool LoadFromMemory(const void* _memory, u32 _size, ShaderData& _data);
        static bool Save(const char* _path, const ShaderData& _data);
        static void Release(ShaderData& _data);
    };

}