#include <avapch.h>
#include "DebugLayer.h"

#include <Application/GUIApplication.h>
#include <Resources/ShaderData.h>
#include <Time/Profiler.h>
#include <Math/Math.h>

#include <Graphics/Camera.h>
#include <Graphics/Texture.h>
#include <Graphics/GpuBuffer.h>
#include <Graphics/FrameBuffer.h>
#include <Graphics/ShaderProgram.h>
#include <Graphics/GraphicsContext.h>

// pre-compiled data
#include "Im3dShaders.embed"

namespace Ava {

    // ----- Debug layer lifecycle --------------------------------------------------------

    DebugLayer::DebugLayer()
        : Layer("Debug layer")
    {
        Im3d::CreateContext();
    }

    DebugLayer::~DebugLayer()
    {
        Im3d::DestroyContext();
    }

    void DebugLayer::OnAttach()
    {
        Shader* stages[ShaderStage::Count]{};
        stages[ShaderStage::Geometric] = nullptr;
        stages[ShaderStage::Compute] = nullptr;

        // Fragment shader is common to both lines and triangles shaders
        ShaderData fragmentShaderData;
        ShaderLoader::LoadFromMemory(im3d_fs_data,  im3d_fs_size, fragmentShaderData);
        stages[ShaderStage::Fragment] = GraphicsContext::CreateShader(fragmentShaderData);
        ShaderLoader::Release(fragmentShaderData);

        ShaderResourceNames varNames;
        // @warning Must match the resources declared in "shaders/Debug/Im3d_vs.glsl"
        varNames.SetConstantBufferName(ShaderStage::Vertex, 0, "cbParameters");
        // @warning Must match the resources declared in "shaders/Debug/Im3d_fs.glsl"
        varNames.SetSampledTextureName(ShaderStage::Fragment, 0, "txDepth");
        varNames.SetConstantBufferName(ShaderStage::Fragment, 1, "cbParameters");

        // Loads lines shader
        ShaderData linesVertexShaderData;
        ShaderLoader::LoadFromMemory(im3d_lines_vs_data, im3d_lines_vs_size, linesVertexShaderData);
        stages[ShaderStage::Vertex] = GraphicsContext::CreateShader(linesVertexShaderData);
        m_lineShaderProgram = GraphicsContext::CreateProgram(stages, varNames);
        ShaderLoader::Release(linesVertexShaderData);

        // Loads triangles shader
        ShaderData trianglesVertexShaderData;
        ShaderLoader::LoadFromMemory(im3d_triangles_vs_data, im3d_triangles_vs_size, trianglesVertexShaderData);
        stages[ShaderStage::Vertex] = GraphicsContext::CreateShader(trianglesVertexShaderData);
        m_triangleShaderProgram = GraphicsContext::CreateProgram(stages, varNames);
        ShaderLoader::Release(trianglesVertexShaderData);
    }

    void DebugLayer::OnUpdate(Timestep& _dt)
    {
        AUTO_CPU_MARKER("IM3D Update");

        const auto* window = GUIApplication::GetInstance().GetWindow();

        if (!window->IsMinimized())
        {
            Im3d::NewFrame();
        }
    }

    void DebugLayer::OnRender(GraphicsContext* _ctx)
    {
        Im3d::Render();

        const Im3dDrawData* drawCommands = Im3d::GetDrawData();
        AVA_ASSERT(drawCommands->valid, "[Im3d] invalid draw data, did you call Im3d::Render() ?");

        if (drawCommands->cmdListsCount > 0)
        {
            AUTO_CPU_GPU_MARKER("IM3D Render");
            _RenderDrawCommands(_ctx, drawCommands);
        }
    }

    void DebugLayer::OnEvent(Event& _event)
    {
    }

    void DebugLayer::OnDetach()
    {
        // Releases lines shader
        if (m_lineShaderProgram)
        {
            GraphicsContext::DestroyProgram(m_lineShaderProgram);
            m_lineShaderProgram = nullptr;
        }

        // Releases triangles shader
        if (m_triangleShaderProgram)
        {
            GraphicsContext::DestroyProgram(m_triangleShaderProgram);
            m_triangleShaderProgram = nullptr;
        }
    }


    // ----- Debug layer helpers ----------------------------------------------------------

    void DebugLayer::_RenderDrawCommands(const GraphicsContext* _ctx, const Im3dDrawData* _drawData) const
    {
        // Early discards when no commands to draw
        if (_drawData->cmdListsCount == 0)
        {
            return;
        }

        _ctx->Reset();

        // Disables back-face culling
        static const RasterStateID rasterState = GraphicsContext::CreateRasterState(RasterState::NoCulling);
        _ctx->SetRasterState(rasterState);

        // Enables alpha blending
        static const BlendStateID blendState = GraphicsContext::CreateBlendState(BlendState::AlphaBlending);
        _ctx->SetBlendState(blendState);

        // Setups framebuffer
        FrameBuffer* frameBuffer = m_targetFramebuffer ? m_targetFramebuffer : _ctx->GetMainFramebuffer();
        _ctx->SetFramebuffer(frameBuffer);

        // Setups viewport
        const Vec2f renderExtents = frameBuffer->GetSize();
        _ctx->SetViewport((u16)renderExtents.x, (u16)renderExtents.y);

        // Setups depth buffer
        AVA_ASSERT(m_depthTexture, "[Im3d] requires a valid depth buffer.");
        _ctx->SetTexture(ShaderStage::Fragment, 0, m_depthTexture);

        // @warning Must match the cbParameters buffer defined in "shaders/Debug/Im3d_vs.glsl"
        struct VertexParamGPU
        {
            Mat4 view;
            Mat4 proj;
            Vec2f screenSize;
        };

        // Setups vertex shader parameters
        AVA_ASSERT(m_sceneCamera != nullptr, "[Im3d] requires a valid scene camera.");
        const ConstantBufferRange vertexShaderParamBuffer = _ctx->CreateTransientConstantBuffer(sizeof VertexParamGPU);
        auto* vertexShaderParameters = static_cast<VertexParamGPU*>(vertexShaderParamBuffer.data);
        vertexShaderParameters->view = m_sceneCamera->GetView();
        vertexShaderParameters->proj = m_sceneCamera->GetProj();
        vertexShaderParameters->screenSize = renderExtents;
        _ctx->SetConstantBuffer(ShaderStage::Vertex, 0, vertexShaderParamBuffer);

        // Emits Im3d draw commands
        for (int i = 0; i < _drawData->cmdListsCount; i++)
        {
            const Im3dDrawList* cmdList = _drawData->cmdLists[i];

            for (int depthTest = 0; depthTest < Im3dDepthTest_Count; depthTest++)
            {
                const auto& lineVertexBuffer = cmdList->m_lineVtxBuffer[depthTest];
                const auto& triangleVertexBuffer = cmdList->m_triangleVtxBuffer[depthTest];

                // We need at least 3 points to draw a triangle / triangle strip
                const bool drawLines = lineVertexBuffer.size() >= 3;
                const bool drawTriangles = triangleVertexBuffer.size() >= 3;

                if (!drawLines && !drawTriangles)
                {
                    continue;
                }

                const float depthFailAlphaFactor =
                    depthTest == Im3dDepthTest_Enable ? 0.f :
                    depthTest == Im3dDepthTest_Disable ? 1.f :
                    depthTest == Im3dDepthTest_TransparentWhenBehind ? 0.1f :
                    1.f;

                // @warning Must match the cbParameters buffer defined in "shaders/Debug/Im3d_fs.glsl"
                struct FragmentParamGPU
                {
                    Vec2f screenSize;
                    float depthFailAlphaFactor;
                };

                // Setups fragment shader parameters
                const ConstantBufferRange fragmentShaderParamBuffer = _ctx->CreateTransientConstantBuffer(sizeof FragmentParamGPU);
                auto* fragmentShaderParameters = static_cast<FragmentParamGPU*>(fragmentShaderParamBuffer.data);
                fragmentShaderParameters->screenSize =  renderExtents;
                fragmentShaderParameters->depthFailAlphaFactor = depthFailAlphaFactor;
                _ctx->SetConstantBuffer(ShaderStage::Fragment, 1, fragmentShaderParamBuffer);

                if (drawLines)
                {
                    static VertexLayout vertexLayout = VertexLayout()
                            .AddAttribute(VertexSemantic::Position, DataType::FLOAT32, 3)
                            .AddAttribute(VertexSemantic::Tangent, DataType::FLOAT32, 3)
                            .AddAttribute(VertexSemantic::Color, DataType::UNORM8, 4)
                            .Build();

                    _ctx->SetPrimitiveType(PrimitiveType::TriangleStrip);

                    // 9K vertices, 28 bytes each -> ~252 KB
                    // (less than 256 KB and divisible by 3)
                    constexpr u32 kMaxVerticesPerDraw = 9 * 1024;
                    const Im3dLineVertex* currentVertices = lineVertexBuffer.data();
                    u32 nbRemainingVertices = (u32)lineVertexBuffer.size();

                    // Split the buffer into several draw calls if it's very big because
                    // we need to allocate a transient buffer and their size is limited.
                    while (nbRemainingVertices > 0)
                    {
                        const u32 vertexCount = Math::min(nbRemainingVertices, kMaxVerticesPerDraw);

                        VertexBufferRange vertexBuffer = _ctx->CreateTransientVertexBuffer(vertexLayout, vertexCount);
                        vertexBuffer.data = (char*)currentVertices;

                        _ctx->SetVertexBuffer(vertexBuffer);
                        _ctx->Draw(m_lineShaderProgram);

                        nbRemainingVertices -= vertexCount;
                        currentVertices += vertexCount;
                    }
                }

                if (drawTriangles)
                {
                    static VertexLayout vertexLayout = VertexLayout()
                            .AddAttribute(VertexSemantic::Position, DataType::FLOAT32, 3)
                            .AddAttribute(VertexSemantic::Color, DataType::UNORM8, 4)
                            .Build();

                    _ctx->SetPrimitiveType(PrimitiveType::Triangle);

                    // 12K vertices, 16 bytes each -> ~182 KB
                    // (less than 256 KB and divisible by 3)
                    constexpr u32 kMaxVerticesPerDraw = 12 * 1024;
                    const Im3dVertex* currentVertices = triangleVertexBuffer.data();
                    u32 nbRemainingVertices = (u32)triangleVertexBuffer.size();

                    // Split the buffer into several draw calls if it's very big because
                    // we need to allocate a transient buffer and their size is limited.
                    while (nbRemainingVertices > 0)
                    {
                        const u32 vertexCount = Math::min(nbRemainingVertices, kMaxVerticesPerDraw);

                        VertexBufferRange vertexBuffer = _ctx->CreateTransientVertexBuffer(vertexLayout, vertexCount);
                        vertexBuffer.data = (char*)currentVertices;

                        _ctx->SetVertexBuffer(vertexBuffer);
                        _ctx->Draw(m_triangleShaderProgram);

                        nbRemainingVertices -= vertexCount;
                        currentVertices += vertexCount;
                    }
                }
            }
        }
    }

}
