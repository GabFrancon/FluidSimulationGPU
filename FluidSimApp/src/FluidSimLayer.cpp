#include "FluidSimLayer.h"
#include "FluidSimApp.h"

// Ava
#include <UI/ImGuiTools.h>
#include <Math/BitUtils.h>
#include <Time/Timestep.h>
#include <Time/Profiler.h>
#include <Time/ProfilerViewer.h>
#include <Graphics/Color.h>
#include <Graphics/GraphicsContext.h>
#include <Graphics/Texture.h>
#include <Graphics/Camera.h>
#include <Layers/UILayer.h>
#include <Layers/DebugLayer.h>
#include <Debug/Im3D.h>
#include <Debug/Log.h>
#include <Graphics/FrameBuffer.h>
using namespace Ava;

// SPH
#include <SPH/SphSolver.h>

FluidSimLayer::FluidSimLayer()
{
    // Simulation setup
    {
        sph::SphSettings settings{};
        settings.spacing = kSpacing;
        settings.dimensions = kBoundVolume;
        settings.restDensity = kRestDensity;
        settings.nbFluidParticles = static_cast<int>(kFluidVolume.x / kSpacing) * static_cast<int>(kFluidVolume.y / kSpacing);
        settings.nbBoundaryParticles = 4 * static_cast<int>(kBoundVolume.x / kSpacing) + 4 * static_cast<int>((kBoundVolume.y - 4 * kSpacing) / kSpacing);

        m_solver = new sph::SphSolver(settings);
        _ResetParticles();
    }

    // Visualization setup
    {
        TextureDescription colorDesc;
        colorDesc.width = 1920;
        colorDesc.height = 1080;
        colorDesc.format = TextureFormat::RGBA8;
        colorDesc.flags = AVA_TEXTURE_RENDER_TARGET | AVA_TEXTURE_SAMPLED | AVA_TEXTURE_CLAMP;
        m_colorTexture = GraphicsContext::CreateTexture(colorDesc);

        TextureDescription depthDesc;
        depthDesc.width = 1920;
        depthDesc.height = 1080;
        depthDesc.format = TextureFormat::D16;
        depthDesc.flags = AVA_TEXTURE_RENDER_TARGET | AVA_TEXTURE_SAMPLED | AVA_TEXTURE_CLAMP;
        m_depthTexture = GraphicsContext::CreateTexture(depthDesc);

        FrameBufferDescription fbDesc;
        fbDesc.AddAttachment(m_colorTexture);
        m_frameBuffer = GraphicsContext::CreateFrameBuffer(fbDesc);

        m_camera = new Camera();
        m_camera->SetPosition(Math::Origin);
        m_camera->SetViewVector(Math::AxisZ);
        m_camera->SetOrtho(0.f, kBoundVolume.x, 0.f, kBoundVolume.y, -2.f, 2.f);

        m_profiler = new ProfilerViewer();
        m_profiler->SetPauseKey(ImGuiKey_F4, "F4");
    }
}

FluidSimLayer::~FluidSimLayer()
{
    delete m_solver;
    delete m_profiler;
    delete m_camera;

    GraphicsContext::DestroyTexture(m_colorTexture);
    GraphicsContext::DestroyTexture(m_depthTexture);
    GraphicsContext::DestroyFrameBuffer(m_frameBuffer);
}

void FluidSimLayer::OnUpdate(Timestep& _dt)
{
    if (m_simulationRunning)
    {
        AUTO_CPU_MARKER("SIMULATION");

        for (int i = 0; i < kSimulationStepCount; i++)
        {
            m_solver->Simulate(kSimulationStepSize);
        }
    }
}

void FluidSimLayer::OnRender(GraphicsContext* _ctx)
{
    _ctx->SetFramebuffer(m_frameBuffer);
    _ctx->Clear(&kClearColor, nullptr);

    _DrawParticles();
    _DisplayUI();
}

void FluidSimLayer::_ResetParticles() const
{
    const float offset25  = 0.25f * kSpacing * 2.f;
    const float offset75  = 0.75f * kSpacing * 2.f;
    const float offset100 = 1.00f * kSpacing * 2.f;

    // Boundary particles
    {
        int particleID = 0;

        constexpr float bottomX = 0.f;
        const float topX = kBoundVolume.x;

        constexpr float bottomY = 0.f;
        const float topY = kBoundVolume.y;

        for (float x = bottomX; x < topX; x += offset100)
        {
            // bottom boundary
            m_solver->SetBoundaryPosition(particleID++, Vec2f(x + offset25, bottomY + offset25));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(x + offset25, bottomY + offset75));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(x + offset75, bottomY + offset25));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(x + offset75, bottomY + offset75));

            // top boundary
            m_solver->SetBoundaryPosition(particleID++, Vec2f(x + offset25, topY - offset25));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(x + offset25, topY - offset75));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(x + offset75, topY - offset25));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(x + offset75, topY - offset75));
        }

        for (float y = bottomY + offset100; y < topY - offset100; y += offset100)
        {
            // left boundary
            m_solver->SetBoundaryPosition(particleID++, Vec2f(bottomX + offset25, y + offset25));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(bottomX + offset25, y + offset75));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(bottomX + offset75, y + offset25));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(bottomX + offset75, y + offset75));

            // right boundary
            m_solver->SetBoundaryPosition(particleID++, Vec2f(topX - offset25, y + offset25));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(topX - offset25, y + offset75));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(topX - offset75, y + offset25));
            m_solver->SetBoundaryPosition(particleID++, Vec2f(topX - offset75, y + offset75));
        }
    }

    // Fluid particles
    {
        int particleID = 0;

        for (float x = offset100; x < kFluidVolume.x + offset100; x += offset100)
        {
            for (float y = offset100; y < kFluidVolume.y + offset100; y += offset100)
            {
                m_solver->SetFluidVelocity(particleID, { 0.f, 0.f });
                m_solver->SetFluidPosition(particleID++, { x + offset25, y + offset25 });

                m_solver->SetFluidVelocity(particleID, { 0.f, 0.f });
                m_solver->SetFluidPosition(particleID++, { x + offset25, y + offset75 });

                m_solver->SetFluidVelocity(particleID, { 0.f, 0.f });
                m_solver->SetFluidPosition(particleID++, { x + offset75, y + offset25 });

                m_solver->SetFluidVelocity(particleID, { 0.f, 0.f });
                m_solver->SetFluidPosition(particleID++, { x + offset75, y + offset75 });
            }
        }
    }

    m_solver->Prepare();
}

void FluidSimLayer::_DrawParticles() const
{
    DebugLayer* debugLayer = FluidSimApp::GetApp().GetDebugLayer();
    debugLayer->SetFramebuffer(m_frameBuffer);
    debugLayer->SetDepthTexture(m_depthTexture);
    debugLayer->SetSceneCamera(m_camera);

    const Vec3f particleRadius = Vec3f(0.5f * kSpacing);

    for (int particleID = 0; particleID < m_solver->GetBoundaryCount(); particleID++)
    {
        const Vec2f position = m_solver->GetBoundaryPosition(particleID);

        const Vec3f boundPos = { position.x, position.y, 0.f };
        const Color boundColor = kWallColor;

        Im3d::AddRectFilled(boundPos, particleRadius, Math::Identity4, boundColor);
    }

    for (int particleID = 0; particleID < m_solver->GetFluidCount(); particleID++)
    {
        const Vec2f position = m_solver->GetFluidPosition(particleID);
        const float density = m_solver->GetFluidDensity(particleID) / kRestDensity;

        const Vec3f fluidPos = { position.x, position.y, 0.f };

        Color fluidColor;
        fluidColor.r = kLightColor.r + (kDenseColor.r - kLightColor.r) * density;
        fluidColor.g = kLightColor.g + (kDenseColor.g - kLightColor.g) * density;
        fluidColor.b = kLightColor.b + (kDenseColor.b - kLightColor.b) * density;
    
        Im3d::AddRectFilled(fluidPos, particleRadius, Math::Identity4, fluidColor);
    }
}

void FluidSimLayer::_DisplayUI()
{
    // DOCK SPACE BEGIN
    {
        constexpr ImGuiWindowFlags windowFlags = 
            ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | 
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | 
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGui::Begin("DockSpace", (bool*)true, windowFlags);
        ImGui::PopStyleVar(3);

        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DockingEnable)
        {
            ImGui::DockSpace(ImGui::GetID("DockSpace"));
        }
    }

    // SETTINGS
    {
        ImGui::Begin("Settings");
        const ImVec2 regionSize = ImGui::GetContentRegionAvail();
        const ImVec2 buttonSize = ImVec2(regionSize.x, 50.f);

        if (ImGui::Button(m_simulationRunning ? "Stop" : "Start", buttonSize))
        {
            m_simulationRunning = !m_simulationRunning;
        }

        if (ImGui::Button("Reset", buttonSize))
        {
            _ResetParticles();
            m_simulationRunning = false;
        }

        ImGui::End();
    }

    // VIEWPORT
    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
        ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollWithMouse);

        const ImVec2 regionSize = ImGui::GetContentRegionAvail();
        const Vec2f windowSize = GUIApplication::GetInstance().GetWindow()->GetExtents();

        ImVec2 viewportSize{};
        viewportSize.x =  windowSize.x > windowSize.y ? regionSize.x : windowSize.x * regionSize.y / windowSize.y;
        viewportSize.y = windowSize.x > windowSize.y ? windowSize.y * regionSize.x / windowSize.x : regionSize.y;

        static ShadedTexture shViewport;
        shViewport.texture = m_colorTexture;

        ImGui::Image(&shViewport, viewportSize);

        ImGui::End();
        ImGui::PopStyleVar();
    }

    // PROFILER
    m_profiler->Display(nullptr);

    // DOCK SPACE END
    ImGui::End();
}
