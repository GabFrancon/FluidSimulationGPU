#include <avapch.h>
#include "GraphicsCommon.h"

#include <Graphics/GraphicsContext.h>
#include <Strings/StringBuilder.h>
#include <Files/Serializer.h>
#include <Files/FilePath.h>
#include <Debug/Log.h>

namespace Ava {

    // --------- Vertex attribute --------------------------------------

    bool VertexAttribute::operator==(const VertexAttribute& _other) const
    {
        return
            semantic == _other.semantic
            && dataType == _other.dataType
            && dataCount == _other.dataCount
            && offset == _other.offset;
    }

    bool VertexAttribute::operator!=(const VertexAttribute& _other) const
    {
        return !(*this == _other);
    }

    u16 VertexAttribute::GetBytesOffset() const
    {
        return offset;
    }

    u16 VertexAttribute::GetBytesCount() const
    {
        return dataCount * BytesCount(dataType);
    }

    SerializeError VertexAttribute::Serialize(Serializer& _serializer, const char* _tag)
    {
        if (_serializer.OpenSection(_tag) == SerializeError::None)
        {
            _serializer.Serialize("semantic", semantic);
            _serializer.Serialize("dataType", dataType);
            _serializer.Serialize("dataCount", dataCount);
            _serializer.Serialize("offset", offset);
        }

        _serializer.CloseSection(_tag);
        return SerializeError::None;
    }

    u32 VertexAttribute::Hash() const
    {
        u32 hash = HashU32(semantic);
        hash = HashU32Combine(dataType, hash);
        hash = HashU32Combine(dataCount, hash);
        hash = HashU32Combine(offset, hash);
        return hash;
    }


    // --------- Vertex layout -----------------------------------------

    VertexLayout VertexLayout::EmptyLayout = VertexLayout();

    VertexLayout& VertexLayout::AddAttribute(const VertexSemantic::Enum _semantic, const DataType::Enum _dataType, const u8 _dataCount)
    {
        if (AVA_VERIFY(attributesCount < VertexSemantic::Count, 
            "You have exceeded the max number of vertex semantic allowed."))
        {
            const VertexAttribute newElement(_semantic, _dataType, _dataCount, 0);
            attributes[attributesCount] = newElement;
            attributesCount++;
        }
        return *this;
    }

    VertexLayout& VertexLayout::Build()
    {
        // recompute vertex stride and attributes offset
        u32 currentOffset = 0;
        for (u8 i = 0; i < attributesCount; i++)
        {
            attributes[i].offset = currentOffset;
            currentOffset += attributes[i].dataCount * BytesCount(attributes[i].dataType);
        }
        stride = currentOffset;
        return *this;
    }

    bool VertexLayout::HasAttribute(VertexSemantic::Enum _semantic) const
    {
        for (u8 i = 0; i < attributesCount; i++)
        {
            if (attributes[i].semantic == _semantic)
            {
                return true;
            }
        }
        return false;
    }

    const VertexAttribute* VertexLayout::GetAttribute(VertexSemantic::Enum _semantic) const
    {
        for (u8 i = 0; i < attributesCount; i++)
        {
            if (attributes[i].semantic == _semantic)
            {
                return &attributes[i];
            }
        }
        return nullptr;
    }

    SerializeError VertexLayout::Serialize(Serializer& _serializer, const char* _tag)
    {
        if (_serializer.IsWriting() && stride == 0u)
        {
            // Recomputes attribute offsets and vertex stride
            // in case the layout was not properly built.
            Build();
        }

        if (_serializer.OpenSection(_tag) == SerializeError::None)
        {
            _serializer.SerializeArray("attributes", attributes, VertexSemantic::Count);
            _serializer.Serialize("attributesCount", attributesCount);
            _serializer.Serialize("stride", stride);
        }

        _serializer.CloseSection(_tag);
        return SerializeError::None;
    }

    u32 VertexLayout::Hash() const
    {
        u32 hash = HashU32(stride);
        for (u8 i = 0; i < attributesCount; i++)
        {
            const u32 attributeHash = attributes[i].Hash();
            hash = HashU32Combine(attributeHash, hash);
        }
        return hash;
    }


    // --------- Shader resource names ---------------------------------

    ShaderResourceNames::ShaderResourceNames()
    {
        const u32 maxSlotPerStage = GraphicsContext::GetSettings().nbBindingSlotPerStage;

        for (int stage = 0; stage < ShaderStage::Count; ++stage)
        {
            m_names[NameType::ConstantBuffer][(ShaderStage::Enum)stage].resize(maxSlotPerStage);
            m_names[NameType::SampledTexture][(ShaderStage::Enum)stage].resize(maxSlotPerStage);
            m_names[NameType::StorageBuffer][(ShaderStage::Enum)stage].resize(maxSlotPerStage);
            m_names[NameType::StorageTexture][(ShaderStage::Enum)stage].resize(maxSlotPerStage);
        }

        Reset();
    }

    ShaderResourceNames::~ShaderResourceNames()
    {
    }

    u32 ShaderResourceNames::Hash() const
    {
        u32 hash = HashU32Init();
        for (int type = 0; type < NameType::Count; ++type)
        {
            for (int stage = 0; stage < ShaderStage::Count; ++stage)
            {
                for (int slot = 0; slot < (int)m_names[type][stage].size(); ++slot)
                {
                    hash = HashU32Combine(m_names[type][stage][slot].GetValue(), hash);
                }
            }
        }
        return hash;
    }

    void ShaderResourceNames::Reset()
    {
        for (int type = 0; type < NameType::Count; ++type)
        {
            for (int stage = 0; stage < ShaderStage::Count; ++stage)
            {
                for (int slot = 0; slot < (int)m_names[type][stage].size(); ++slot)
                {
                    m_names[type][stage][slot] = StringHash::Invalid;
                }
            }
        }
    }

    void ShaderResourceNames::_SetName(const NameType::Enum _type, const ShaderStage::Enum _stage, const u8 _slot, const StringHash _name)
    {
        AVA_ASSERT(_slot < m_names[_type][_stage].size());
        m_names[_type][_stage][_slot] = _name;
    }

    StringHash ShaderResourceNames::_GetName(const NameType::Enum _type, const ShaderStage::Enum _stage, const u8 _slot) const
    {
        AVA_ASSERT(_slot < m_names[_type][_stage].size());
        return m_names[_type][_stage][_slot];
    }


    // --------- Shader description ------------------------------------

    const char* ShaderDescription::kSrcExtension = ".glsl";
    const char* ShaderDescription::kBinExtension = ".vks";
    const char* ShaderDescription::kStageSuffixes[ShaderStage::Count] = { "vs", "gs", "fs", "cs" };

    bool ShaderDescription::IsValid() const
    {
        return !path.empty();
    }

    void ShaderDescription::AddDefine(const std::string& _define)
    {
        defines.insert(_define);
    }

    void ShaderDescription::Reset()
    {
        path = "";
        defines.clear();
    }

    u32 ShaderDescription::BuildShaderId() const
    {
        u32 shaderId = HashStr(path.c_str());

        // Defines are stored in a std::set so that the insert order does not change the resulting shader ID.
        for (const auto& define : defines)
        {
            const u32 defineHash = HashStr(define.c_str());
            shaderId = HashU32Combine(defineHash, shaderId);
        }
        return shaderId;
    }

    std::string ShaderDescription::BuildResourcePath(const ShaderStage::Enum _stage) const
    {
        char pathClean[MAX_PATH];
        if (const char* last = strrchr(path.c_str(), '_'))
        {
            // Remove '_xx' to avoid duplication (we will have stage tag at the end of the file).
            StrFormat(pathClean, MAX_PATH, "%.*s", last - path.c_str(), path.c_str());
        }

        StringBuilder builder;
        builder.append(pathClean);

        // Adds shader ID.
        builder.appendf("_%08x", BuildShaderId());

        // Adds shader stage.
        builder.appendf("_%s", kStageSuffixes[_stage]);

        // Adds binary extension.
        builder.appendf("%s", kBinExtension);

        return builder.c_str();
    }

    bool ShaderDescription::ExtractStageAndIdFromPath(const char* _resourcePath, ShaderStage::Enum* _stage, u32* _resID)
    {
        const char* binExtension = FilePath::FindExtension(_resourcePath);
        AVA_ASSERT(binExtension[-3] == '_', "Incorrect shader binary path: '%s'", _resourcePath);

        if (_stage)
        {
            switch (binExtension[-2])
            {
                case 'v':
                    *_stage = ShaderStage::Vertex;
                    break;
                case 'g':
                    *_stage = ShaderStage::Geometric;
                    break;
                case 'f':
                    *_stage = ShaderStage::Fragment;
                    break;
                case 'c':
                    *_stage = ShaderStage::Compute;
                    break;
                default:
                    AVA_CORE_ERROR("[Shader] incorrect shader binary path: '%s'", _resourcePath);
                    return false;
            }
        }

        if (_resID)
        {
            // skip "_xx" + 8 hex chars
            const char* shaderIdBegin = binExtension - 11;

            if (shaderIdBegin - 1 < _resourcePath || shaderIdBegin[-1] != '_')
            {
                AVA_CORE_ERROR("[Shader] path '%s' is missing shader id.", _resourcePath);
                return false;
            }
            *_resID = strtoul(shaderIdBegin, nullptr, 16);
        }

        return true;
    }

    bool ShaderDescription::operator==(const ShaderDescription& _other) const
    {
        return BuildShaderId() == _other.BuildShaderId();
    }

    bool ShaderDescription::operator!=(const ShaderDescription& _other) const
    {
        return BuildShaderId() != _other.BuildShaderId();
    }

    SerializeError ShaderDescription::Serialize(Serializer& _serializer, const char* _tag)
    {
        if (_serializer.OpenSection(_tag) == SerializeError::None)
        {
            _serializer.Serialize("path", path);
            _serializer.Serialize("defines", defines);
        }
        _serializer.CloseSection(_tag);
        return SerializeError::None;
    }


    // --------- Shader program description ----------------------------

    u32 ShaderProgramDescription::BuildProgramID() const
    {
        u32 id = resourceNames.Hash();

        for (u8 stage = 0; stage < ShaderStage::Count; stage++)
        {
            if (shaders[stage].IsValid())
            {
                const u32 shaderID = shaders[stage].BuildShaderId();
                id = HashU32Combine(shaderID, id);
            }
        }
        return id;
    }

    std::string ShaderProgramDescription::BuildProgramName() const
    {
        for (u8 stage = 0; stage < ShaderStage::Count; stage++)
        {
            if (shaders[stage].IsValid())
            {
                char name[MAX_PATH];
                FilePath::RemoveExtension(name, shaders[stage].path.c_str());
                return name;
            }
        }
        return "";
    }


    // --------- Raster state ------------------------------------------

    RasterState RasterState::NoCulling = RasterState();
    RasterState RasterState::BackCulling = RasterState(CullMode::Back);
    RasterState RasterState::FrontCulling = RasterState(CullMode::Front);

    u32 RasterState::Hash() const
    {
        u32 hash = HashU32(cullMode);
        hash = HashU32Combine(depthBiasEnable, hash);
        hash = HashU32Combine(depthBiasConstantFactor, hash);
        hash = HashU32Combine(depthBiasSlopeFactor, hash);
        return hash;
    }


    // --------- Depth state -------------------------------------------

    DepthState DepthState::NoDepth = DepthState();
    DepthState DepthState::ReadOnly = DepthState(CompareFunc::Less, false);
    DepthState DepthState::ReadWrite = DepthState(CompareFunc::Less, true);
    DepthState DepthState::ReadOnlyReversed = DepthState(CompareFunc::Greater, false);
    DepthState DepthState::ReadWriteReversed = DepthState(CompareFunc::Greater, true);

    u32 DepthState::Hash() const
    {
        u32 hash = HashU32(testEnable);
        hash = HashU32Combine(writeEnable, hash);
        hash = HashU32Combine(compareFunc, hash);
        return hash;
    }


    // --------- Blend state -------------------------------------------

    BlendState BlendState::NoBlending = BlendState();
    BlendState BlendState::AlphaBlending = BlendState(BlendFunc::Add);

    u32 BlendState::Hash() const
    {
        u32 hash = HashU32(blendEnable);
        hash = HashU32Combine(rgbSrcFactor, hash);
        hash = HashU32Combine(rgbDstFactor, hash);
        hash = HashU32Combine(alphaSrcFactor, hash);
        hash = HashU32Combine(alphaDstFactor, hash);
        hash = HashU32Combine(redWrite, hash);
        hash = HashU32Combine(greenWrite, hash);
        hash = HashU32Combine(blueWrite, hash);
        hash = HashU32Combine(alphaWrite, hash);
        return hash;
    }

}
