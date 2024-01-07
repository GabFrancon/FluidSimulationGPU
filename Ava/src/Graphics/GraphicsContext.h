#pragma once
/// @file GraphicsContext.h
/// @brief interface to emit render commands on the GPU.

#include <Graphics/GraphicsEnums.h>
#include <Graphics/GraphicsCommon.h>

namespace Ava {

    class ConstantBuffer;
    struct ConstantBufferRange;

    class IndirectBuffer;
    struct IndirectBufferRange;

    class VertexBuffer;
    struct VertexBufferRange;

    class IndexBuffer;
    struct IndexBufferRange;

    class Texture;
    struct TextureDescription;
    struct TextureData;

    class Shader;
    struct ShaderDescription;
    struct ShaderData;

    class ShaderProgram;
    struct ShaderProgramDescription;
    class ShaderResourceNames;

    class FrameBuffer;
    struct FrameBufferDescription;

    class GraphicsContextImpl;
    struct TransientPool;

    /// @brief Graphics settings, required to create a GraphicsContext.
    struct GraphicsSettings
    {
        u32 contextCount = 2;
        u32 maxDescriptorPerFrame = 100;
        u32 nbBindingSlotPerStage = 32;
        bool enableVSync = true;
    };

    /// @brief Main handler to interact with the graphics processing unit (GPU).
    class GraphicsContext
    {
    public:
        static void Init(const GraphicsSettings& _settings);
        static void Shutdown();
        void Reset() const;

        // Getters
        static GraphicsContext* GetCurrentContext();
        static const GraphicsSettings& GetSettings();
        static u32 GetCurrentContextId();
        static u32 GetGlobalFrameId();
        static u32 GetContextCount();

        GraphicsContextImpl* GetImpl() const;
        u32 GetContextId() const;
        u32 GetFrameId() const;

        // Time
        u64 GetGpuTimestamp() const;
        static void SetVSyncMode(bool _enable);
        static bool GetVSyncMode();
        static void WaitIdle();

        // Frame cycle
        void StartFrame();
        void EndFrameAndSubmit() const;

        // Draw commands
        void Draw(ShaderProgram* _program, u32 _instanceCount = 1) const;
        void Draw(ShaderProgram* _program, const IndirectBufferRange& _indirectRange) const;

        // Dispatch commands
        void Dispatch(ShaderProgram* _program, u32 _groupCountX, u32 _groupCountY, u32 _groupCountZ) const;
        void Dispatch(ShaderProgram* _program, const IndirectBufferRange& _indirectRange) const;

        // Constant buffers
        static u32 GetConstantBufferAlignment(u32 _flags = 0);
        static ConstantBuffer* CreateConstantBuffer(u32 _size, u32 _flags = 0);
        ConstantBufferRange CreateTransientConstantBuffer(u32 _size, u32 _flags = 0) const;
        void SetConstantBuffer(ShaderStage::Enum _stage, u8 _slot, const ConstantBufferRange& _range) const;
        static void DestroyBuffer(ConstantBuffer* _buffer);

        // Indirect buffers
        static u32 GetIndirectBufferAlignment(u32 _flags = 0);
        static IndirectBuffer* CreateIndirectBuffer(u32 _size, u32 _flags = 0);
        static IndirectBuffer* CreateIndirectBuffer(CommandType::Enum _commandType, u32 _commandCount, u32 _flags = 0);
        IndirectBufferRange CreateTransientIndirectBuffer(CommandType::Enum _commandType, u32 _commandCount, u32 _flags = 0) const;
        static void DestroyBuffer(IndirectBuffer* _buffer);

        // Vertex buffers
        static u32 GetVertexBufferAlignment(u32 _flags = 0);
        static VertexBuffer* CreateVertexBuffer(u32 _size, u32 _flags = 0);
        static VertexBuffer* CreateVertexBuffer(const VertexLayout& _vertexLayout, u32 _vertexCount, u32 _flags = 0);
        VertexBufferRange CreateTransientVertexBuffer(const VertexLayout& _vertexLayout, u32 _vertexCount, u32 _flags = 0) const;
        void SetVertexBuffer(const VertexBufferRange& _range) const;
        static void DestroyBuffer(VertexBuffer* _buffer);

        // Index buffers
        static u32 GetIndexBufferAlignment(u32 _flags = 0);
        static IndexBuffer* CreateIndexBuffer(u32 _indexCount, u32 _flags = 0);
        IndexBufferRange CreateTransientIndexBuffer(u32 _indexCount, u32 _flags = 0) const;
        void SetIndexBuffer(const IndexBufferRange& _range) const;
        static void DestroyBuffer(IndexBuffer* _buffer);

        // Textures
        static Texture* CreateTexture(const TextureDescription& _desc);
        static Texture* CreateTexture(const char* _path, u32 _flags = 0);
        static Texture* CreateTexture(const TextureData& _data, u32 _flags = 0);
        static Texture* ConstructTexture(void* _memory, const char* _path, u32 _flags = 0);
        void SetTexture(ShaderStage::Enum _stage, u8 _slot, Texture* _texture) const;
        static void DestructTexture(Texture* _texture);
        static void DestroyTexture(Texture* _texture);

        // Storage resources
        static u32 GetStorageBufferAlignment();
        void SetStorageBuffer(ShaderStage::Enum _stage, u8 _slot, const ConstantBufferRange& _range, Access::Enum _access) const;
        void SetStorageBuffer(ShaderStage::Enum _stage, u8 _slot, const IndirectBufferRange& _range, Access::Enum _access) const;
        void SetStorageBuffer(ShaderStage::Enum _stage, u8 _slot, const VertexBufferRange& _range, Access::Enum _access) const;
        void SetStorageBuffer(ShaderStage::Enum _stage, u8 _slot, const IndexBufferRange& _range, Access::Enum _access) const;
        void SetStorageTexture(ShaderStage::Enum _stage, u8 _slot, Texture* _texture, Access::Enum _access, u16 _mip = 0) const;

        // Framebuffers
        static FrameBuffer* GetMainFramebuffer();
        const FrameBuffer* GetCurrentFramebuffer() const;
        static FrameBuffer* CreateFrameBuffer(const FrameBufferDescription& _desc);
        void SetFramebuffer(FrameBuffer* _framebuffer = nullptr) const;
        void Clear(const Color* _color, const float* _depth) const;
        static void DestroyFrameBuffer(FrameBuffer* _framebuffer);

        // Shaders
        static Shader* CreateShader(ShaderStage::Enum _stage, const ShaderDescription& _desc);
        static Shader* CreateShader(const ShaderData& _data);
        static Shader* ConstructShader(void* _memory, ShaderStage::Enum _stage, const ShaderDescription& _desc);
        static void DestructShader(Shader* _shader);
        static void DestroyShader(Shader* _shader);

        // Shader programs
        static ShaderProgram* CreateProgram(const ShaderProgramDescription& _desc);
        static ShaderProgram* CreateProgram(Shader* _stages[ShaderStage::Count], const ShaderResourceNames& _varNames);
        static ShaderProgram* ConstructProgram(void* _memory, const ShaderProgramDescription& _desc);
        static void DestructProgram(ShaderProgram* _program);
        static void DestroyProgram(ShaderProgram* _program);

        // States
        static VertexLayoutID CreateVertexLayout(const VertexLayout& _vertexLayout);
        static RasterStateID CreateRasterState(const RasterState& _rasterState);
        static DepthStateID CreateDepthState(const DepthState& _depthState);
        static BlendStateID CreateBlendState(const BlendState& _blendState);
        void SetPrimitiveType(PrimitiveType::Enum _type) const;
        void SetRasterState(RasterStateID _rasterState) const;
        void SetDepthState(DepthStateID _depthState) const;
        void SetBlendState(BlendStateID _blendState) const;
        void SetWireframe(bool _enabled) const;
        void SetLineWidth(float _lineWidth) const;
        void SetViewport(u16 _width, u16 _height, u16 _x = 0, u16 _y = 0) const;
        void SetScissor(u16 _width, u16 _height, u16 _x = 0, u16 _y = 0) const;
        static const VertexLayout& GetVertexLayout(VertexLayoutID _id);
        static const RasterState& GetRasterState(RasterStateID _id);
        static const DepthState& GetDepthState(DepthStateID _id);
        static const BlendState& GetBlendState(BlendStateID _id);

        // Debug options
        static void EnableGraphicsDebug(bool _enable);
        static void ForceRecreateFramebuffer();
        static void TriggerFrameCapture();
        bool ValidateTransientBuffer(u32 _frameCreationId) const;
        void AddDebugMarker(const char* _label, const Color& _color) const;
        void BeginDebugMarkerRegion(const char* _label, const Color& _color) const;
        void EndDebugMarkerRegion() const;
        void DisplayDebug() const;

    private:
        explicit GraphicsContext(u32 _contextId);
        ~GraphicsContext();

        GraphicsContextImpl* m_impl = nullptr;
        TransientPool* m_transientPool = nullptr;

        u32 m_contextId = 0;
        u32 m_frameId = 0;
    };

}
