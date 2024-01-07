#pragma once
/// @file DebugLayer.h
/// @brief Template for an Im3D-based debug layer.

#include <Layers/Layer.h>
#include <Debug/Im3D.h>

namespace Ava {

    class Shader;
    class Texture;
    class FrameBuffer;
    class ShaderProgram;
    class GraphicsContext;
    class Camera;

    // Debug layer based on the Im3D library.
    class DebugLayer : public Layer
    {
    public:
        explicit DebugLayer();
        ~DebugLayer() override;

        void OnAttach() override;
        void OnUpdate(Timestep& _dt) override;
        void OnRender(GraphicsContext* _ctx) override;
        void OnEvent(Event& _event) override;
        void OnDetach() override;

        void SetDepthTexture(Texture* _depthTexture) { m_depthTexture = _depthTexture; }
        void SetSceneCamera(Camera* _sceneCamera) { m_sceneCamera = _sceneCamera; }
        void SetFramebuffer(FrameBuffer* _framebuffer) { m_targetFramebuffer = _framebuffer; }

    private:
        void _RenderDrawCommands(const GraphicsContext* _ctx, const Im3dDrawData* _drawData) const;

        // Im3D shaders
        ShaderProgram* m_lineShaderProgram = nullptr;
        ShaderProgram* m_triangleShaderProgram = nullptr;

        // user-defined render objects
        Texture* m_depthTexture = nullptr;
        Camera* m_sceneCamera = nullptr;
        FrameBuffer* m_targetFramebuffer = nullptr;
    };

}