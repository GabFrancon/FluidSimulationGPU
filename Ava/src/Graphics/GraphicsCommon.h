#pragma once
/// @file GraphicsCommon.h
/// @brief

#include <Math/Hash.h>
#include <Math/Types.h>
#include <Strings/StringHash.h>
#include <Graphics/GraphicsEnums.h>

namespace Ava {

    // ----- Vertices ------------------------------------------------

    /// @brief VertexAttribute is used as a building block for defining the
    /// layout of vertex data within a VertexLayout. It specifies the type,
    /// format, and byte offset of a particular vertex attribute, ensuring
    /// proper interpretation by the graphics pipeline and vertex shader.
    struct VertexAttribute
    {
        VertexSemantic::Enum semantic;
        DataType::Enum dataType;
        u8 dataCount = 0;
        u32 offset = 0;

        VertexAttribute() = default;

        explicit VertexAttribute(const VertexSemantic::Enum _semantic, const DataType::Enum _dataType, const u8 _dataCount, const u32 _offset)
            : semantic(_semantic), dataType(_dataType), dataCount(_dataCount), offset(_offset) {}

        bool operator==(const VertexAttribute& _other) const;
        bool operator!=(const VertexAttribute& _other) const;

        u16 GetBytesOffset() const;
        u16 GetBytesCount() const;

        SerializeError Serialize(Serializer& _serializer, const char* _tag);
        u32 Hash() const;
    };

    /// @brief Describes the layout of vertex attributes for
    /// configuring graphics pipeline input state and ensuring
    /// correct access to per-vertex data in the vertex shader.
    struct VertexLayout
    {
        VertexAttribute attributes[VertexSemantic::Count];
        u8 attributesCount = 0;
        u32 stride = 0;

        VertexLayout() = default;

        VertexLayout& AddAttribute(VertexSemantic::Enum _semantic, DataType::Enum _dataType, u8 _dataCount);
        VertexLayout& Build();

        bool HasAttribute(VertexSemantic::Enum _semantic) const;
        const VertexAttribute* GetAttribute(VertexSemantic::Enum _semantic) const;

        SerializeError Serialize(Serializer& _serializer, const char* _tag);
        u32 Hash() const;

        // empty vertex layout
        static VertexLayout EmptyLayout;
    };


    // ----- Shaders -------------------------------------------------

    /// @brief Stores the names of every resources declared in a shader.
    class ShaderResourceNames
    {
    public:
        ShaderResourceNames();
        ~ShaderResourceNames();

        void       SetConstantBufferName(const ShaderStage::Enum _stage, const u8 _slot, const StringHash _bufferName)  { _SetName(NameType::ConstantBuffer, _stage, _slot, _bufferName); }
        StringHash GetConstantBufferName(const ShaderStage::Enum _stage, const u8 _slot) const                          { return _GetName(NameType::ConstantBuffer, _stage, _slot); }

        void       SetSampledTextureName(const ShaderStage::Enum _stage, const u8 _slot, const StringHash _textureName) { _SetName(NameType::SampledTexture, _stage, _slot, _textureName); }
        StringHash GetSampledTextureName(const ShaderStage::Enum _stage, const u8 _slot) const                          { return _GetName(NameType::SampledTexture, _stage, _slot); }

        void       SetStorageBufferName(const ShaderStage::Enum _stage, const u8 _slot, const StringHash _bufferName)   { _SetName(NameType::StorageBuffer, _stage, _slot, _bufferName); }
        StringHash GetStorageBufferName(const ShaderStage::Enum _stage, const u8 _slot) const                           { return _GetName(NameType::StorageBuffer, _stage, _slot); }

        void       SetStorageTextureName(const ShaderStage::Enum _stage, const u8 _slot, const StringHash _textureName) { _SetName(NameType::StorageTexture, _stage, _slot, _textureName); }
        StringHash GetStorageTextureName(const ShaderStage::Enum _stage, const u8 _slot) const                          { return _GetName(NameType::StorageTexture, _stage, _slot); }

        u32 Hash() const;
        void Reset();

    private:
        struct NameType
        {
            enum Enum : u8
            {
                ConstantBuffer,
                SampledTexture,
                StorageBuffer,
                StorageTexture,

                Count,
            };
        };

        void       _SetName(NameType::Enum _type, ShaderStage::Enum _stage, u8 _slot, StringHash _name);
        StringHash _GetName(NameType::Enum _type, ShaderStage::Enum _stage, u8 _slot) const;

        std::vector<StringHash> m_names[NameType::Count][ShaderStage::Count];
    };

    /// @brief Shader resource construction data.
    struct ShaderDescription
    {
        std::string path = "";
        std::set<std::string> defines;

        bool IsValid() const;
        void AddDefine(const std::string& _define);
        void Reset();

        u32 BuildShaderId() const;
        std::string BuildResourcePath(ShaderStage::Enum _stage) const;
        static bool ExtractStageAndIdFromPath(const char* _resourcePath, ShaderStage::Enum* _stage, u32* _resID);

        bool operator==(const ShaderDescription& _other) const;
        bool operator!=(const ShaderDescription& _other) const;

        SerializeError Serialize(Serializer& _serializer, const char* _tag);

        static const char* kSrcExtension;
        static const char* kBinExtension;
        static const char* kStageSuffixes[ShaderStage::Count];
    };

    /// @brief Shader program resource construction data.
    struct ShaderProgramDescription
    {
        ShaderDescription shaders[ShaderStage::Count];
        ShaderResourceNames resourceNames;

        u32 BuildProgramID() const;
        std::string BuildProgramName() const;
    };


    // ----- Fixed pipeline states -----------------------------------

    /// @brief Describes how the rasterization stage behaves in the graphics pipeline.
    struct RasterState
    {
        CullMode::Enum cullMode = CullMode::None;
        bool depthBiasEnable = false;
        float depthBiasConstantFactor = 0.f;
        float depthBiasSlopeFactor = 0.f;

        RasterState() = default;
        explicit RasterState(const CullMode::Enum _cullMode) : cullMode(_cullMode) {}

        // predefined raster states
        static RasterState NoCulling;
        static RasterState BackCulling;
        static RasterState FrontCulling;

        u32 Hash() const;
    };

    /// @brief Describes how depth testing is performed in the graphics pipeline.
    struct DepthState
    {
        bool testEnable = false;
        bool writeEnable = false;
        CompareFunc::Enum compareFunc = CompareFunc::Less;

        DepthState() = default;
        explicit DepthState(const CompareFunc::Enum _func, const bool _write) : testEnable(true), writeEnable(_write), compareFunc(_func) {}

        // predefined depth states
        static DepthState NoDepth;
        static DepthState ReadOnly;
        static DepthState ReadWrite;
        static DepthState ReadOnlyReversed;
        static DepthState ReadWriteReversed;

        u32 Hash() const;
    };

    /// @brief Describes how color blending is performed in the graphics pipeline.
    struct BlendState
    {
        bool blendEnable = false;
        BlendFunc::Enum blendFunc = BlendFunc::Add;

        BlendFactor::Enum rgbSrcFactor = BlendFactor::SrcAlpha;
        BlendFactor::Enum rgbDstFactor = BlendFactor::OneMinusSrcAlpha;
        BlendFactor::Enum alphaSrcFactor = BlendFactor::One;
        BlendFactor::Enum alphaDstFactor = BlendFactor::Zero;

        bool redWrite = true;
        bool greenWrite = true;
        bool blueWrite = true;
        bool alphaWrite = true;

        BlendState() = default;
        explicit BlendState(const BlendFunc::Enum _func) : blendEnable(true), blendFunc(_func) {}

        // predefined blend states
        static BlendState NoBlending;
        static BlendState AlphaBlending;

        u32 Hash() const;
    };


    // ----- Fixed Pipeline state IDs --------------------------------

    static constexpr u8 kStateIdxInvalid = 0xFF;

    #define DEFINE_PIPELINE_STATE_ID(_name)                                       \
        struct _name                                                              \
        {                                                                         \
            u8 index = kStateIdxInvalid;                                          \
                                                                                  \
            bool operator!=(const _name _id) const                                \
            {                                                                     \
                return index != _id.index;                                        \
            }                                                                     \
        };                                                                        \
                                                                                  \
        inline bool IsValid(const _name _id)                                      \
        {                                                                         \
            return _id.index != kStateIdxInvalid;                                 \
        }

    DEFINE_PIPELINE_STATE_ID(VertexLayoutID);
    DEFINE_PIPELINE_STATE_ID(RasterStateID);
    DEFINE_PIPELINE_STATE_ID(DepthStateID);
    DEFINE_PIPELINE_STATE_ID(BlendStateID);

    #define AVA_DISABLE_STATE { kStateIdxInvalid }


    // ----- Indirect commands ---------------------------------------

    /// @brief Indirect draw command, based on most graphics APIs.
    struct IndirectDrawCmd
    {
        u32 vertexCount   = 0; ///< The number of vertices to draw.
        u32 instanceCount = 0; ///< The number of instances to draw.
        u32 firstVertex   = 0; ///< An offset added to each vertex index.
        u32 firstInstance = 0; ///< An offset added to each instance index.
    };

    /// @brief Indirect draw indexed command, based on most graphics APIs.
    struct IndirectDrawIndexedCmd
    {
        u32 indexCount     = 0; ///< The number of indices to draw.
        u32 instanceCount  = 0; ///< The number of instances to draw.
        u32 firstIndex     = 0; ///< The base index within the index buffer.
        u32 vertexOffset   = 0; ///< An offset added to each vertex index.
        u32 firstInstance  = 0; ///< An offset added to each instance index.
    };

    /// @brief Indirect dispatch command, based on most graphics APIs.
    struct IndirectDispatchCmd
    {
        u32 threadGroupCountX = 1; ///< The number of thread groups to dispatch in the X dimension.
        u32 threadGroupCountY = 1; ///< The number of thread groups to dispatch in the Y dimension.
        u32 threadGroupCountZ = 1; ///< The number of thread groups to dispatch in the Z dimension.
    };

}
