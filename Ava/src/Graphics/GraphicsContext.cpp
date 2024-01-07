#include <avapch.h>
#include "GraphicsContext.h"

#include <Graphics/Texture.h>
#include <Graphics/GpuBuffer.h>
#include <Graphics/Shader.h>
#include <Graphics/ShaderProgram.h>
#include <Graphics/FrameBuffer.h>
#include <Graphics/TransientPool.h>

#include <Time/Profiler.h>
#include <Debug/Capture.h>
#include <Containers/HashMap.h>
#include <UI/ImGuiTools.h>
#include <Debug/Assert.h>
#include <Debug/Log.h>

#if defined(AVA_GRAPHIC_API_VULKAN)
    #include <Platform/Vulkan/VkGraphicsContext.h>
#else
    #error Unkown graphics API
#endif

namespace Ava {

    static GraphicsSettings s_graphicsSettings{};       // holds the settings applied to all graphics contexts
    static std::vector<GraphicsContext*> s_instances{}; // holds all the graphics context instances
    static u32 s_globalFrameId = 0;                     // tracks the current frame ID
    static bool s_isInitialized = false;                // flag to check if initialization was done correctly

    // Graphics pipeline states
    static constexpr u32 kMaxPipelineStates = 16;
    static HashMap<VertexLayout> s_vertexLayoutCache (kMaxPipelineStates);
    static HashMap<RasterState>  s_rasterStateCache  (kMaxPipelineStates);
    static HashMap<DepthState>   s_depthStateCache   (kMaxPipelineStates);
    static HashMap<BlendState>   s_blendStateCache   (kMaxPipelineStates);


    void GraphicsContext::Init(const GraphicsSettings& _settings)
    {
        if (s_isInitialized)
        {
            AVA_CORE_WARN("[GraphicsContext] all graphics contexts have already been initialized.");
            return;
        }

        // Platform dependent initialization
        s_graphicsSettings = _settings;
        GraphicsContextImpl::Init();

        // Init each context instance
        s_instances.resize(s_graphicsSettings.contextCount);

        for (u32 contextId = 0; contextId < s_graphicsSettings.contextCount; contextId++)
        {
            s_instances[contextId] = new GraphicsContext(contextId);
        }

        s_globalFrameId = 0;
        s_isInitialized = true;
    }

    void GraphicsContext::Shutdown()
    {
        if (!s_isInitialized)
        {
            AVA_CORE_WARN("[GraphicsContext] no graphics context have been initialized.");
            return;
        }

        // Shutdown each instance
        for (size_t i = 0; i < GetContextCount(); i++)
        {
            delete s_instances[i];
        }

        // Static shutdown
        GraphicsContextImpl::Shutdown();
    }

    void GraphicsContext::Reset() const
    {
        m_impl->Reset();
    }

    GraphicsContext::GraphicsContext(const u32 _contextId)
    {
        m_impl = new GraphicsContextImpl(_contextId);
        m_transientPool = new TransientPool();
        m_contextId = _contextId;
    }

    GraphicsContext::~GraphicsContext()
    {
        delete m_impl;
        delete m_transientPool;
    }


    // ------- Getters ------------------------------------------------------------------------------------------------

    u32 GraphicsContext::GetCurrentContextId()
    {
        return s_globalFrameId % GetContextCount();
    }

    u32 GraphicsContext::GetGlobalFrameId()
    {
        return s_globalFrameId;
    }

    GraphicsContext* GraphicsContext::GetCurrentContext()
    {
        return s_instances[GetCurrentContextId()];
    }

    const GraphicsSettings& GraphicsContext::GetSettings()
    {
        return s_graphicsSettings;
    }

    u32 GraphicsContext::GetContextCount()
    {
        return s_graphicsSettings.contextCount;
    }

    GraphicsContextImpl* GraphicsContext::GetImpl() const
    {
        return m_impl;
    }

    u32 GraphicsContext::GetContextId() const
    {
        return m_contextId;
    }

    u32 GraphicsContext::GetFrameId() const
    {
        return m_frameId;
    }


    // ------- Time ---------------------------------------------------------------------------------------------------

    u64 GraphicsContext::GetGpuTimestamp() const
    {
        return m_impl->GetGpuTimestamp();
    }

    void GraphicsContext::SetVSyncMode(const bool _enable)
    {
        s_graphicsSettings.enableVSync = _enable;
        ForceRecreateFramebuffer();
    }

    bool GraphicsContext::GetVSyncMode()
    {
        return s_graphicsSettings.enableVSync;
    }

    void GraphicsContext::WaitIdle()
    {
        GraphicsContextImpl::WaitIdle();
    }


    // ------- Frame cycle --------------------------------------------------------------------------------------------

    void GraphicsContext::StartFrame()
    {
        AUTO_CPU_MARKER("Start GPU Frame");

        // Updates current frame ID
        m_frameId = s_globalFrameId;

        // Resets transient pool
        m_transientPool->Reset();

        // Starts platform dependent frame rendering
        m_impl->StartFrame();

        // Notifies GPU profiler
        if (const auto* profiler = Profiler::GetInstance())
        {
            profiler->StartGpuFrame(this);
        }
    }

    void GraphicsContext::EndFrameAndSubmit() const
    {
        AUTO_CPU_MARKER("End GPU Frame");

        // Ends platform dependent frame rendering
        m_impl->EndFrameAndSubmit();

        // Increments global frame ID
        s_globalFrameId++;
    }


    // ------- Draw commands ------------------------------------------------------------------------------------------

    void GraphicsContext::Draw(ShaderProgram* _program, const u32 _instanceCount) const
    {
        m_impl->Draw(_program, _instanceCount);
    }

    void GraphicsContext::Draw(ShaderProgram* _program, const IndirectBufferRange& _indirectRange) const
    {
        return m_impl->Draw(_program, _indirectRange);
    }


    // ------- Dispatch commands --------------------------------------------------------------------------------------

    void GraphicsContext::Dispatch(ShaderProgram* _program, const u32 _groupCountX, const u32 _groupCountY, const u32 _groupCountZ) const
    {
        return m_impl->Dispatch(_program, _groupCountX, _groupCountY, _groupCountZ);
    }

    void GraphicsContext::Dispatch(ShaderProgram* _program, const IndirectBufferRange& _indirectRange) const
    {
        return m_impl->Dispatch(_program, _indirectRange);
    }


    // ------- Constant buffers ---------------------------------------------------------------------------------------

    u32 GraphicsContext::GetConstantBufferAlignment(const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::GetConstantBufferAlignment(_flags);
    }

    ConstantBuffer* GraphicsContext::CreateConstantBuffer(const u32 _size, const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::CreateConstantBuffer(_size, _flags);
    }

    ConstantBufferRange GraphicsContext::CreateTransientConstantBuffer(const u32 _size, const u32 _flags/*= 0*/) const
    {
        ConstantBufferRange range = _flags & AVA_BUFFER_GPU_ONLY
                        ? m_transientPool->gpuOnlyBuffers.AllocateRange(_size, GetConstantBufferAlignment(_flags))
                        : m_transientPool->constantBuffers.AllocateRange(_size, GetConstantBufferAlignment(_flags));

        range.frameCreationId = s_globalFrameId;
        return range;
    }

    void GraphicsContext::SetConstantBuffer(const ShaderStage::Enum _stage, const u8 _slot, const ConstantBufferRange& _range) const
    {
        AVA_ASSERT(ValidateTransientBuffer(_range.frameCreationId), "[GraphicsContext] transient buffers can only be used for 1 frame.");
        m_impl->SetConstantBuffer(_stage, _slot, _range);
    }

    void GraphicsContext::DestroyBuffer(ConstantBuffer* _buffer)
    {
        GraphicsContextImpl::DestroyBuffer(_buffer);
    }


    // ------- Indirect buffers ---------------------------------------------------------------------------------------

    u32 GraphicsContext::GetIndirectBufferAlignment(const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::GetIndirectBufferAlignment(_flags);
    }

    IndirectBuffer* GraphicsContext::CreateIndirectBuffer(const u32 _size, const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::CreateIndirectBuffer(_size, _flags | AVA_BUFFER_READ_WRITE);
    }

    IndirectBuffer* GraphicsContext::CreateIndirectBuffer(const CommandType::Enum _commandType, const u32 _commandCount, const u32 _flags/*= 0*/)
    {
        u32 stride = 0;
        switch (_commandType)
        {
            case CommandType::Draw:
                stride = sizeof(IndirectDrawCmd);
                break;
            case CommandType::DrawIndexed:
                stride = sizeof(IndirectDrawIndexedCmd);
                break;
            case CommandType::Dispatch:
                stride = sizeof(IndirectDispatchCmd);
                break;
            default:
                break;
        }

        return CreateIndirectBuffer(stride * _commandCount, _flags);
    }

    IndirectBufferRange GraphicsContext::CreateTransientIndirectBuffer(const CommandType::Enum _commandType, const u32 _commandCount, const u32 _flags/*= 0*/) const
    {
        u32 stride = 0;
        switch (_commandType)
        {
            case CommandType::Draw:
                stride = sizeof(IndirectDrawCmd);
                break;
            case CommandType::DrawIndexed:
                stride = sizeof(IndirectDrawIndexedCmd);
                break;
            case CommandType::Dispatch:
                stride = sizeof(IndirectDispatchCmd);
                break;
            default:
                break;
        }

        IndirectBufferRange range = m_transientPool->indirectBuffers.AllocateRange(stride * _commandCount, GetIndirectBufferAlignment(_flags | AVA_BUFFER_READ_WRITE));
        range.frameCreationId = s_globalFrameId;
        return range;
    }

    void GraphicsContext::DestroyBuffer(IndirectBuffer* _buffer)
    {
        GraphicsContextImpl::DestroyBuffer(_buffer);
    }


    // ------- Vertex buffers -----------------------------------------------------------------------------------------

    u32 GraphicsContext::GetVertexBufferAlignment(const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::GetVertexBufferAlignment(_flags);
    }

    VertexBuffer* GraphicsContext::CreateVertexBuffer(const u32 _size, const u32 _flags)
    {
        return GraphicsContextImpl::CreateVertexBuffer(_size, _flags);
    }

    VertexBuffer* GraphicsContext::CreateVertexBuffer(const VertexLayout& _vertexLayout, const u32 _vertexCount, const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::CreateVertexBuffer(_vertexLayout, _vertexCount, _flags);
    }

    VertexBufferRange GraphicsContext::CreateTransientVertexBuffer(const VertexLayout& _vertexLayout, const u32 _vertexCount, const u32 _flags/*= 0*/) const
    {
        const u32 bufferSize = _vertexLayout.stride * _vertexCount;

        VertexBufferRange range = m_transientPool->vertexBuffers.AllocateRange(bufferSize, GetVertexBufferAlignment(_flags));
        range.vertexCount = _vertexCount;
        range.vertexLayout = CreateVertexLayout(_vertexLayout);
        range.frameCreationId = s_globalFrameId;

        return range;
    }

    void GraphicsContext::SetVertexBuffer(const VertexBufferRange& _range) const
    {
        AVA_ASSERT(ValidateTransientBuffer(_range.frameCreationId), "[GraphicsContext] transient buffers can only be used for 1 frame.");
        m_impl->SetVertexBuffer(_range);
    }

    void GraphicsContext::DestroyBuffer(VertexBuffer* _buffer)
    {
        GraphicsContextImpl::DestroyBuffer(_buffer);
    }


    // ------- Index buffers ------------------------------------------------------------------------------------------

    u32 GraphicsContext::GetIndexBufferAlignment(const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::GetIndexBufferAlignment(_flags);
    }

    IndexBuffer* GraphicsContext::CreateIndexBuffer(const u32 _indexCount, const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::CreateIndexBuffer(_indexCount, _flags);
    }

    IndexBufferRange GraphicsContext::CreateTransientIndexBuffer(const u32 _indexCount, const u32 _flags/*= 0*/) const
    {
        const u32 bufferSize = (_flags & AVA_BUFFER_INDEX_UINT32 ? 4 : 2) * _indexCount;

        IndexBufferRange range = m_transientPool->indexBuffers.AllocateRange(bufferSize, GetIndexBufferAlignment(_flags));
        range.indexCount = _indexCount;
        range.use32Bits = _flags & AVA_BUFFER_INDEX_UINT32;
        range.frameCreationId = s_globalFrameId;

        return range;
    }

    void GraphicsContext::SetIndexBuffer(const IndexBufferRange& _range) const
    {
        AVA_ASSERT(ValidateTransientBuffer(_range.frameCreationId), "[GraphicsContext] transient buffers can only be used for 1 frame.");
        m_impl->SetIndexBuffer(_range);
    }

    void GraphicsContext::DestroyBuffer(IndexBuffer* _buffer)
    {
        GraphicsContextImpl::DestroyBuffer(_buffer);
    }


    // ------- Textures -----------------------------------------------------------------------------------------------

    Texture* GraphicsContext::CreateTexture(const TextureDescription& _desc)
    {
        return GraphicsContextImpl::CreateTexture(_desc);
    }

    Texture* GraphicsContext::CreateTexture(const char* _path, const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::CreateTexture(_path, _flags);
    }

    Texture* GraphicsContext::CreateTexture(const TextureData& _data, const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::CreateTexture(_data, _flags);
    }

    Texture* GraphicsContext::ConstructTexture(void* _memory, const char* _path, const u32 _flags/*= 0*/)
    {
        return GraphicsContextImpl::ConstructTexture(_memory, _path, _flags);
    }

    void GraphicsContext::SetTexture(const ShaderStage::Enum _stage, const u8 _slot, Texture* _texture) const
    {
        AVA_ASSERT(_texture && _texture->HasFlag(AVA_TEXTURE_SAMPLED), "[GraphicsContext] texture is missing AVA_TEXTURE_SAMPLED flag.");
        m_impl->SetTexture(_stage, _slot, _texture);
    }

    void GraphicsContext::DestructTexture(Texture* _texture)
    {
        GraphicsContextImpl::DestructTexture(_texture);
    }

    void GraphicsContext::DestroyTexture(Texture* _texture)
    {
        GraphicsContextImpl::DestroyTexture(_texture);
    }


    // ------- Storage resources -------------------------------------------------------------------------------------

    u32 GraphicsContext::GetStorageBufferAlignment()
    {
        return GraphicsContextImpl::GetStorageBufferAlignment();
    }

    void GraphicsContext::SetStorageBuffer(const ShaderStage::Enum _stage, const u8 _slot, const ConstantBufferRange& _range, const Access::Enum _access) const
    {
        AVA_ASSERT(ValidateTransientBuffer(_range.frameCreationId), "[GraphicsContext] transient buffers can only be used for 1 frame.");
        AVA_ASSERT(_range.buffer && _range.buffer->HasFlag(AVA_BUFFER_READ_WRITE), "[GraphicsContext] storage buffer is missing AVA_BUFFER_READ_WRITE flag.");

        m_impl->SetStorageBuffer(_stage, _slot, _range, _access);
    }

    void GraphicsContext::SetStorageBuffer(const ShaderStage::Enum _stage, const u8 _slot, const IndirectBufferRange& _range, const Access::Enum _access) const
    {
        AVA_ASSERT(ValidateTransientBuffer(_range.frameCreationId), "[GraphicsContext] transient buffers can only be used for 1 frame.");
        AVA_ASSERT(_range.buffer && _range.buffer->HasFlag(AVA_BUFFER_READ_WRITE), "[GraphicsContext] storage buffer is missing AVA_BUFFER_READ_WRITE flag.");

        m_impl->SetStorageBuffer(_stage, _slot, _range, _access);
    }

    void GraphicsContext::SetStorageBuffer(const ShaderStage::Enum _stage, const u8 _slot, const VertexBufferRange& _range, const Access::Enum _access) const
    {
        AVA_ASSERT(ValidateTransientBuffer(_range.frameCreationId), "[GraphicsContext] transient buffers can only be used for 1 frame.");
        AVA_ASSERT(_range.buffer && _range.buffer->HasFlag(AVA_BUFFER_READ_WRITE), "[GraphicsContext] storage buffer is missing AVA_BUFFER_READ_WRITE flag.");

        m_impl->SetStorageBuffer(_stage, _slot, _range, _access);
    }

    void GraphicsContext::SetStorageBuffer(const ShaderStage::Enum _stage, const u8 _slot, const IndexBufferRange& _range, const Access::Enum _access) const
    {
        AVA_ASSERT(ValidateTransientBuffer(_range.frameCreationId), "[GraphicsContext] yransient buffers can only be used for 1 frame.");
        AVA_ASSERT(_range.buffer && _range.buffer->HasFlag(AVA_BUFFER_READ_WRITE), "[GraphicsContext] storage buffer is missing AVA_BUFFER_READ_WRITE flag.");

        m_impl->SetStorageBuffer(_stage, _slot, _range, _access);
    }

    void GraphicsContext::SetStorageTexture(const ShaderStage::Enum _stage, const u8 _slot, Texture* _texture, const Access::Enum _access, const u16 _mip/*= 0*/) const
    {
        AVA_ASSERT(_texture && _texture->HasFlag(AVA_TEXTURE_READ_WRITE), "[GraphicsContext] storage texture is missing AVA_TEXTURE_READ_WRITE flag.");
        m_impl->SetStorageTexture(_stage, _slot, _texture, _access, _mip);
    }


    // ------- Framebuffers -------------------------------------------------------------------------------------------

    FrameBuffer* GraphicsContext::GetMainFramebuffer()
    {
        return GraphicsContextImpl::GetMainFramebuffer();
    }

    const FrameBuffer* GraphicsContext::GetCurrentFramebuffer() const
    {
        return m_impl->GetCurrentFramebuffer();
    }

    FrameBuffer* GraphicsContext::CreateFrameBuffer(const FrameBufferDescription& _desc)
    {
        return GraphicsContextImpl::CreateFrameBuffer(_desc);
    }

    void GraphicsContext::SetFramebuffer(FrameBuffer* _framebuffer/*= nullptr*/) const
    {
        m_impl->SetFramebuffer(_framebuffer);
    }

    void GraphicsContext::Clear(const Color* _color, const float* _depth) const
    {
        m_impl->Clear(_color, _depth);
    }

    void GraphicsContext::DestroyFrameBuffer(FrameBuffer* _framebuffer)
    {
        GraphicsContextImpl::DestroyFrameBuffer(_framebuffer);
    }


    // ------- Shaders ------------------------------------------------------------------------------------------------

    Shader* GraphicsContext::CreateShader(const ShaderStage::Enum _stage, const ShaderDescription& _desc)
    {
        return GraphicsContextImpl::CreateShader(_stage, _desc);
    }

    Shader* GraphicsContext::CreateShader(const ShaderData& _data)
    {
        return GraphicsContextImpl::CreateShader(_data);
    }

    Shader* GraphicsContext::ConstructShader(void* _memory, const ShaderStage::Enum _stage, const ShaderDescription& _desc)
    {
        return GraphicsContextImpl::ConstructShader(_memory, _stage, _desc);
    }

    void GraphicsContext::DestructShader(Shader* _shader)
    {
        GraphicsContextImpl::DestructShader(_shader);
    }

    void GraphicsContext::DestroyShader(Shader* _shader)
    {
        GraphicsContextImpl::DestroyShader(_shader);
    }


    // ------- Shader programs ----------------------------------------------------------------------------------------

    ShaderProgram* GraphicsContext::CreateProgram(const ShaderProgramDescription& _desc)
    {
        return GraphicsContextImpl::CreateProgram(_desc);
    }

    ShaderProgram* GraphicsContext::CreateProgram(Shader* _stages[ShaderStage::Count], const ShaderResourceNames& _varNames)
    {
        return GraphicsContextImpl::CreateProgram(_stages, _varNames);
    }

    ShaderProgram* GraphicsContext::ConstructProgram(void* _memory, const ShaderProgramDescription& _desc)
    {
        return GraphicsContextImpl::ConstructProgram(_memory, _desc);
    }

    void GraphicsContext::DestructProgram(ShaderProgram* _program)
    {
        GraphicsContextImpl::DestructProgram(_program);
    }

    void GraphicsContext::DestroyProgram(ShaderProgram* _program)
    {
        GraphicsContextImpl::DestroyProgram(_program);
    }


    // ------- States -------------------------------------------------------------------------------------------------

    VertexLayoutID GraphicsContext::CreateVertexLayout(const VertexLayout& _vertexLayout)
    {
        return { s_vertexLayoutCache.insert(_vertexLayout, _vertexLayout.Hash()) };
    }

    RasterStateID GraphicsContext::CreateRasterState(const RasterState& _rasterState)
    {
        return { s_rasterStateCache.insert(_rasterState, _rasterState.Hash()) };
    }

    DepthStateID GraphicsContext::CreateDepthState(const DepthState& _depthState)
    {
        return { s_depthStateCache.insert(_depthState, _depthState.Hash())};
    }

    BlendStateID GraphicsContext::CreateBlendState(const BlendState& _blendState)
    {
        return { s_blendStateCache.insert(_blendState, _blendState.Hash()) };
    }

    void GraphicsContext::SetPrimitiveType(const PrimitiveType::Enum _type) const
    {
        m_impl->SetPrimitiveType(_type);
    }

    void GraphicsContext::SetRasterState(const RasterStateID _rasterState) const
    {
        m_impl->SetRasterState(_rasterState);
    }

    void GraphicsContext::SetDepthState(const DepthStateID _depthState) const
    {
        m_impl->SetDepthState(_depthState);
    }

    void GraphicsContext::SetBlendState(const BlendStateID _blendState) const
    {
        m_impl->SetBlendState(_blendState);
    }

    void GraphicsContext::SetWireframe(const bool _enabled) const
    {
        m_impl->SetWireframe(_enabled);
    }

    void GraphicsContext::SetLineWidth(const float _lineWidth) const
    {
        m_impl->SetLineWidth(_lineWidth);
    }

    void GraphicsContext::SetViewport(const u16 _width, const u16 _height, const u16 _x/*= 0*/, const u16 _y/*= 0*/) const
    {
        m_impl->SetViewport(_width, _height, _x, _y);
    }

    void GraphicsContext::SetScissor(const u16 _width, const u16 _height, const u16 _x/*= 0*/, const u16 _y/*= 0*/) const
    {
        m_impl->SetScissor(_width, _height, _x, _y);
    }

    const VertexLayout& GraphicsContext::GetVertexLayout(const VertexLayoutID _id)
    {
        return IsValid(_id) ? s_vertexLayoutCache[_id.index] : VertexLayout::EmptyLayout;
    }

    const RasterState& GraphicsContext::GetRasterState(const RasterStateID _id)
    {
        return IsValid(_id) ? s_rasterStateCache[_id.index] : RasterState::NoCulling;
    }

    const DepthState& GraphicsContext::GetDepthState(const DepthStateID _id)
    {
        return IsValid(_id) ? s_depthStateCache[_id.index] : DepthState::NoDepth;
    }

    const BlendState& GraphicsContext::GetBlendState(const BlendStateID _id)
    {
        return IsValid(_id) ? s_blendStateCache[_id.index] : BlendState::NoBlending;
    }


    // ------- Debug options ------------------------------------------------------------------------------------------

    void GraphicsContext::EnableGraphicsDebug(const bool _enable)
    {
        GraphicsContextImpl::EnableGraphicsDebug(_enable);
    }

    void GraphicsContext::ForceRecreateFramebuffer()
    {
        GraphicsContextImpl::ForceRecreateFramebuffer();
    }

    void GraphicsContext::TriggerFrameCapture()
    {
        GraphicsContextImpl::TriggerFrameCapture();
    }

    bool GraphicsContext::ValidateTransientBuffer(const u32 _frameCreationId) const
    {
        // Not a transient buffer
        if (!_frameCreationId)
        {
            return true;
        }
        // If the buffer was created at a frame earlier than the current frame,
        // the buffer might be overwritten while the GPU is using it.
        return _frameCreationId >= m_frameId;
    }

    void GraphicsContext::AddDebugMarker(const char* _label, const Color& _color) const
    {
        m_impl->AddDebugMarker(_label, _color);
    }

    void GraphicsContext::BeginDebugMarkerRegion(const char* _label, const Color& _color) const
    {
        m_impl->BeginDebugMarkerRegion(_label, _color);
    }

    void GraphicsContext::EndDebugMarkerRegion() const
    {
        m_impl->EndDebugMarkerRegion();
    }

    void GraphicsContext::DisplayDebug() const
    {
        if (!AVA_VERIFY(ImGuiTools::WithinFrameScope(), 
            "[GraphicsContext] DisplayDebug() can only be called within ImGui frame scope."))
        {
            return;
        }

        static bool enableVSync = GetVSyncMode();
        if (ImGui::Checkbox("VSync", &enableVSync))
        {
            SetVSyncMode(enableVSync);
        }

        ImGui::BeginDisabled(!CaptureMgr::IsEnabled());
        if (ImGui::Button("Trigger capture"))
        {
            TriggerFrameCapture();
        }
        ImGui::EndDisabled();

        if (ImGui::Button("Recreate framebuffer"))
        {
            ForceRecreateFramebuffer();
        }

        if (ImGui::TreeNodeEx("Transient pool", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("GPU only buffers : %.1f KiB (%u buffer(s) of capacity %.1f KiB)", 
                m_transientPool->gpuOnlyBufferLastAllocation / 1024.f, 
                m_transientPool->gpuOnlyBuffers.GetBufferCount(), 
                m_transientPool->gpuOnlyBuffers.GetCapacity() / 1024.f);

            ImGui::Text("Constant buffers : %.1f KiB (%u buffer(s) of capacity %.1f KiB)",
                m_transientPool->constantBufferLastAllocation / 1024.f,
                m_transientPool->constantBuffers.GetBufferCount(),
                m_transientPool->constantBuffers.GetCapacity() / 1024.f);

            ImGui::Text("Indirect buffers : %.1f KiB (%u buffer(s) of capacity %.1f KiB)",
                m_transientPool->indirectBufferLastAllocation / 1024.f,
                m_transientPool->indirectBuffers.GetBufferCount(),
                m_transientPool->indirectBuffers.GetCapacity() / 1024.f);

            ImGui::Text("Vertex buffers   : %.1f KiB (%u buffer(s) of capacity %.1f KiB)", 
                m_transientPool->vertexBufferLastAllocation / 1024.f,
                m_transientPool->vertexBuffers.GetBufferCount(),
                m_transientPool->vertexBuffers.GetCapacity() / 1024.f);

            ImGui::Text("Index buffers    : %.1f KiB (%u buffer(s) of capacity %.1f KiB)", 
                m_transientPool->indexBufferLastAllocation / 1024.f,
                m_transientPool->indexBuffers.GetBufferCount(),
                m_transientPool->indexBuffers.GetCapacity() / 1024.f);

            ImGui::TreePop();
        }

        ImGui::Separator();

        m_impl->DisplayDebug();
    }

}
