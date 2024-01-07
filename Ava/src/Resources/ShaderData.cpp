#include <avapch.h>
#include "ShaderData.h"

#include <Files/BinarySerializer.h>
#include <Memory/Memory.h>
#include <Debug/Log.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <LZ4/lz4.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    // ---- Shader loader ------------------------------------------------

    bool ShaderLoader::Load(const char* _path, ShaderData& _data)
    {
        BinarySerializer serializer(SerializeMode::Read, true);

        // load shader data from disk
        if (!serializer.Load(_path))
        {
            AVA_CORE_ERROR("[Shader] failed to load '%s'.",_path);
            return false;
        }

        // deserialize shader data
        const auto retCode = serializer.Serialize(_data);
        if (retCode != SerializeError::None)
        {
            AVA_CORE_ERROR("[Shader] failed to serialize %s: %s.", _path, Serializer::GetErrorStr(retCode));
            return false;
        }

        return true;
    }

    bool ShaderLoader::LoadFromMemory(const void* _memory, const u32 _size, ShaderData& _data)
    {
        if (!_memory || _size == 0u)
        {
            AVA_CORE_ERROR("[Shader] memory is invalid.");
            return false;
        }

        // deserialize shader data
        BinarySerializer serializer((u8*)_memory, _size, SerializeMode::Read, true);
        const auto retCode = serializer.Serialize(_data);

        if (retCode != SerializeError::None)
        {
            AVA_CORE_ERROR("[Shader] failed to serialize shader: %s.", Serializer::GetErrorStr(retCode));
            return false;
        }

        return true;
    }

    bool ShaderLoader::Save(const char* _path, const ShaderData& _data)
    {
        BinarySerializer serializer(SerializeMode::Write, true);

        // serialize shader data
        const auto retCode = serializer.Serialize(_data);
        if (retCode != SerializeError::None)
        {
            AVA_CORE_ERROR("[Shader] failed to serialize %s: %s.", _path, Serializer::GetErrorStr(retCode));
            return false;
        }

        // save shader data to disk
        if (!serializer.Save(_path))
        {
            AVA_CORE_ERROR("[Shader] failed to save '%s'.",_path);
            return false;
        }

        return true;
    }

    void ShaderLoader::Release(ShaderData& _data)
    {
        if (_data.binData)
        {
            // free the shared memory
            AVA_FREE(_data.binData);
            _data.binData = nullptr;
            _data.binSize = 0u;

            // nullify the code and reflection pointers
            _data.code = nullptr;
            _data.codeSize = 0u;
            _data.reflectionData = nullptr;
            _data.reflectionDataSize = 0u;
        }
    }


    // ---- Serialization ------------------------------------------------

    SerializeError ShaderData::Serialize(Serializer& _serializer, const char* _tag)
    {
        if (_serializer.OpenSection(_tag) == SerializeError::None)
        {
            _serializer.Serialize("resourceName", resourceName);
            _serializer.Serialize("shaderStage", stage);

            _serializer.Serialize("bindingsCount", bindingsCount);
            _serializer.Serialize("attributesCount", attributesCount);
            _serializer.Serialize("colorOutputsCount", colorOutputsCount);

            _serializer.Serialize("codeSize", codeSize);
            _serializer.Serialize("reflectionDataSize", reflectionDataSize);

            // compute shader size
            binSize = (u64)codeSize + (u64)reflectionDataSize;

            // binaryBlob contains a compressed version of shaderData
            std::vector<char> binaryBlob;

            // compress binaries
            if (_serializer.IsWriting())
            {
                const auto shaderSize = static_cast<int>(binSize);
                auto* shaderData = static_cast<char*>(binData);
                bool internallyAllocated = false;

                // shader code and reflection data were probably stored separately,
                // so let's allocate a continuous buffer and copy them both in a row
                if (!shaderData)
                {
                    shaderData = (char*)AVA_MALLOC(shaderSize);
                    internallyAllocated = true;

                    if (char* binaryCursor = shaderData)
                    {
                        if (AVA_VERIFY(code != nullptr), "[Serializer] code buffer was not allocated.")
                        {
                            memcpy(binaryCursor, code, codeSize);
                            binaryCursor += codeSize;
                        }

                        if (AVA_VERIFY(reflectionData != nullptr, "[Serializer] reflection buffer was not allocated."))
                        {
                            memcpy(binaryCursor, reflectionData, reflectionDataSize);
                            //binaryCursor += reflectionDataSize;
                        }
                    }
                }

                // compress shaderData to binaryBlob
                const int compressStaging = LZ4_compressBound(shaderSize);
                binaryBlob.resize(compressStaging);

                const int compressedSize = LZ4_compress_default(shaderData, binaryBlob.data(), shaderSize, compressStaging);
                binaryBlob.resize(compressedSize);

                // free the continuous buffer we internally allocated
                // to merge both the shader code and reflection data
                if (internallyAllocated)
                {
                    AVA_FREE(shaderData);
                    shaderData = nullptr;
                }
            }

            // quickly serialize binaries
            _serializer.SerializeBytes("binaryBlob", binaryBlob);

            // decompress binaries
            if (_serializer.IsReading())
            {
                // allocate shader data
                AVA_ASSERT(binData == nullptr, "[Serializer] shader data is already allocated.");
                binData = AVA_MALLOC(binSize);

                const auto shaderSize = static_cast<int>(binSize);
                auto* shaderData = static_cast<char*>(binData);
                const auto compressedSize = static_cast<int>(binaryBlob.size());
                const char* compressedData = binaryBlob.data();

                // decompress binaryBlob to shader data
                LZ4_decompress_safe(compressedData, shaderData, compressedSize, shaderSize);

                // find shader code
                code = reinterpret_cast<u32*>(shaderData);
                shaderData += codeSize;

                // find reflection data
                reflectionData = shaderData;
                //shaderData += reflectionDataSize;
            }
        }

        _serializer.CloseSection(_tag);
        return SerializeError::None;
    }

    SerializeError ShaderData::SerializeHeaderOnly(Serializer& _serializer, const char* _tag)
    {
        AVA_ASSERT(_serializer.IsReading(), "[Serializer] header only serialization should only be used in read mode.");

        if (_serializer.OpenSection(_tag) == SerializeError::None)
        {
            _serializer.Serialize("resourceName", resourceName);
            _serializer.Serialize("shaderStage", stage);

            _serializer.Serialize("bindingsCount", bindingsCount);
            _serializer.Serialize("attributesCount", attributesCount);
            _serializer.Serialize("colorOutputsCount", colorOutputsCount);

            _serializer.Serialize("codeSize", codeSize);
            _serializer.Serialize("reflectionDataSize", reflectionDataSize);

            // compute shader size
            binSize = (u64)codeSize + (u64)reflectionDataSize;
        }

        _serializer.CloseSection(_tag);
        return SerializeError::None;
    }

}
