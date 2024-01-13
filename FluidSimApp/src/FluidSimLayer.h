#pragma once

#include <Math/Types.h>
#include <Layers/Layer.h>
#include <Graphics/Color.h>

namespace Ava {
    class Camera;
    class Texture;
    class ProfilerViewer;
}

namespace sph {
    class SphSolver;
}

class FluidSimLayer final : public Ava::Layer
{
public:
    FluidSimLayer();
    ~FluidSimLayer() override;

    void OnUpdate(Ava::Timestep& _dt) override;
    void OnRender(Ava::GraphicsContext* _ctx) override;

private:
    // scene
    const float kSpacing = 0.125f;
    const float kRestDensity = 1e3f;
    const int kBoundaryThickness = 2;
    const Ava::Vec2f kFluidVolume = {  7.f, 10.f };
    const Ava::Vec2f kBoundVolume = { 25.f, 20.f };
    const Ava::Color kClearColor = {  10 / 255.0f,  10 / 255.0f,  10 / 255.0f };
    const Ava::Color kFluidColor = {   0 / 255.0f,  37 / 255.0f, 147 / 255.0f };
    const Ava::Color kBoundColor = { 195 / 255.0f,  50 / 255.0f,  30 / 255.0f };

    // simulation
    sph::SphSolver* m_solver = nullptr;
    bool m_simulationRunning = false;
    bool m_enableGpuSimulation = true;
    float m_simulationStepSize = 1.f / 180.f;
    int m_simulationStepCount = 3; // 60 FPS
    Ava::ProfilerViewer* m_profiler = nullptr;

    // rendering
    Ava::Camera* m_camera = nullptr;
    Ava::Texture* m_colorTexture = nullptr;
    Ava::Texture* m_depthTexture = nullptr;
    Ava::FrameBuffer* m_frameBuffer = nullptr;

    void _ResetScene() const;
    void _DrawParticles() const;
    void _DisplayUI();
};
