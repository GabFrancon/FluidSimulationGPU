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
    // visualization
    Ava::Camera* m_camera = nullptr;
    Ava::Texture* m_colorTexture = nullptr;
    Ava::Texture* m_depthTexture = nullptr;
    Ava::FrameBuffer* m_frameBuffer = nullptr;
    Ava::ProfilerViewer* m_profiler = nullptr;

    // simulation
    sph::SphSolver* m_solver = nullptr;
    bool m_simulationRunning = false;

    // Settings
    const float kSpacing = 0.125f;
    const float kRestDensity = 1e3f;

    const float kSimulationStepSize = 1.f / 180.f;
    const int kSimulationStepCount = 3; // 60 FPS

    const Ava::Vec2f kFluidVolume = {  6.f, 12.f };
    const Ava::Vec2f kBoundVolume = { 25.f, 14.f };

    const Ava::Color kClearColor = {  10 / 255.0f,  10 / 255.0f,  10 / 255.0f };
    const Ava::Color kLightColor = {  79 / 255.0f, 132 / 255.0f, 237 / 255.0f };
    const Ava::Color kDenseColor = {  10 / 255.0f,  47 / 255.0f, 119 / 255.0f };
    const Ava::Color kWallColor =  { 195 / 255.0f,  50 / 255.0f,  30 / 255.0f };


    void _ResetParticles() const;
    void _DrawParticles() const;
    void _DisplayUI();
};
