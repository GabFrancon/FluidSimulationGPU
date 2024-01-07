#include "FluidSimApp.h"
#include "FluidSimLayer.h"

// Ava
#include <Core/EntryPoint.h>
#include <Files/FileManager.h>
#include <Inputs/InputManager.h>
#include <Graphics/GraphicsContext.h>
#include <Time/Profiler.h>
#include <Layers/UILayer.h>
#include <Layers/DebugLayer.h>


// ----- Create Ava Application ---------------------------------------

Ava::Application* Ava::CreateApplication(const CmdLineParser& _args)
{
    WindowSettings windowSettings{};
    windowSettings.title = "Fluid Simulation";
    windowSettings.width = 1920;
    windowSettings.height = 1080;
    windowSettings.darkMode = true;

    InputSettings inputSettings{};
    inputSettings.trackMouse = true;                    // mouse position and scroll offsets are computed every frame
    inputSettings.useGamepad = true;                    // if connected, gamepad state is retrieved every frame
    inputSettings.cursorMode = Mouse::ModeNormal;       // mouse cursor behaves like in traditional UI interfaces

    GraphicsSettings graphicSettings{};
    graphicSettings.enableVSync = true;                 // v-sync on
    graphicSettings.contextCount = 2;                   // use double buffering
    graphicSettings.nbBindingSlotPerStage = 8;         // 8 resources available per shader stage
    graphicSettings.maxDescriptorPerFrame = 200;       // up to 200 shader resources bound per frame (for each type)

    ProfilerSettings profilerSettings{};
    profilerSettings.numFramesToRecord = 5;
    profilerSettings.maxCpuMarkerCount = 128;
    profilerSettings.maxGpuMarkerCount = 128;

    GUIAppSettings appSettings;
    appSettings.window = &windowSettings;
    appSettings.inputs = &inputSettings;
    appSettings.graphics = &graphicSettings;
    appSettings.profiler = &profilerSettings;

    return new FluidSimApp(appSettings);
}


// ----- FluidSim App --------------------------------------------------

FluidSimApp::FluidSimApp(const Ava::GUIAppSettings& _settings) : GUIApplication(_settings)
{
    // Instantiates custom editor layer
    PushLayer<FluidSimLayer>();

    // Instantiates built-in debug layer
    m_debugLayer = PushLayer<Ava::DebugLayer>();

    // Instantiates built-in UI layer
    Ava::ImGuiSettings UISettings{};
    UISettings.fontScale = 2.f;
    UISettings.theme = Ava::UI::ThemeDark;
    UISettings.configFlags = Ava::AVA_UI_DOCKING | Ava::AVA_UI_KEYBOARD_NAVIGATION;
    m_UILayer = PushOverlay<Ava::UILayer>(UISettings);
}

FluidSimApp::~FluidSimApp()
{
}