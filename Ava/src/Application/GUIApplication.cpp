#include <avapch.h>
#include "GUIApplication.h"

#include <Files/FileManager.h>
#include <Inputs/InputManager.h>
#include <Time/TimeManager.h>
#include <Time/Profiler.h>
#include <Time/PrecisionTimer.h>
#include <Graphics/GraphicsContext.h>
#include <Events/WindowEvent.h>

namespace Ava {

    GUIApplication* GUIApplication::s_instance = nullptr;

    GUIApplication::GUIApplication(const GUIAppSettings& _settings)
    {
        const PrecisionTimer timer;

        AVA_ASSERT(!s_instance, "GUI app already exists.");
        s_instance = this;

        // TIME MANAGER
        {
            TimeMgr::Init();
        }

        // FILE MANAGER
        {
            FileMgr::Init();
        }

        // WINDOW
        {
            AVA_ASSERT(_settings.window, "GUI app requires valid window settings.");
            const WindowSettings settings = _settings.window ? *_settings.window : WindowSettings();
            m_window = CreateScope<Window>(settings);
        }

        // INPUT MANAGER
        {
            AVA_ASSERT(_settings.inputs, "GUI app requires valid input settings.");
            const InputSettings settings = _settings.inputs ? *_settings.inputs : InputSettings();
            InputMgr::Init(settings);
        }

        // GRAPHICS CONTEXT
        {
            AVA_ASSERT(_settings.graphics, "GUI app requires valid graphics settings.");
            const GraphicsSettings settings = _settings.graphics ? *_settings.graphics : GraphicsSettings();
            GraphicsContext::Init(settings);
        }

    #if defined(AVA_ENABLE_PROFILER)
        // PROFILER
        if (_settings.profiler)
        {
            Profiler::Init(*_settings.profiler);
        }
    #endif

        // Set event callbacks
        {
            const EventCallbackFunc mainEventCallback = AVA_BIND_FN(GUIApplication::ProcessEvent);

            m_window->SetEventCallback(mainEventCallback);
            INPUT_MGR->SetEventCallback(mainEventCallback);
        }

        AVA_CORE_INFO("GUI app base initialization completed (%.6f s)", timer.Elapsed());
    }

    GUIApplication::~GUIApplication()
    {
        m_layerStack.Clear();
        Profiler::Close();

        GraphicsContext::Shutdown();
        InputMgr::Shutdown();
        FileMgr::Shutdown();
        TimeMgr::Shutdown();
    }

    void GUIApplication::Run()
    {
        // this is the main game loop
        while (m_appRunning)
        {
            // --- Starts frame profiling -------------------------
            if (const auto* profiler = Profiler::GetInstance())
            {
                profiler->StartFrame();
            }

            // --- Process events ---------------------------------
            {
                m_window->Process();
                INPUT_MGR->Process();
            }

            // --- Updates logic ----------------------------------
            {
                TIME_MGR->Process();
                Timestep dt = TIME_MGR->GetLastFrameDuration();

                for (const auto& layer : m_layerStack)
                {
                    layer->OnUpdate(dt);
                }
            }

            // --- Renders graphics -------------------------------
            if (m_appActive)
            {
                GraphicsContext* ctx = GraphicsContext::GetCurrentContext();
                ctx->StartFrame();

                for (const auto& layer : m_layerStack)
                {
                    layer->OnRender(ctx);
                }

                ctx->EndFrameAndSubmit();
            }

            // --- Ends frame profiling ---------------------------
            if (auto* profiler = Profiler::GetInstance())
            {
                profiler->EndFrame();
            }
        }
    }

    void GUIApplication::ProcessEvent(Event& _event)
    {
        EventDispatcher dispatcher(_event);
        dispatcher.Dispatch<WindowMinimizedEvent>(AVA_BIND_FN(_OnWindowMinimized));
        dispatcher.Dispatch<WindowRestoredEvent>(AVA_BIND_FN(_OnWindowRestored));
        dispatcher.Dispatch<WindowClosedEvent>(AVA_BIND_FN(_OnWindowClosed));

        // Dispatches the event on the different layers until it gets handled.
        for (auto it = m_layerStack.rbegin(); it != m_layerStack.rend(); ++it)
        {
            if (_event.m_handled) {
                return;
            }
            (*it)->OnEvent(_event);
        }
    }

    void GUIApplication::Close()
    {
        m_appRunning = false;
    }

    bool GUIApplication::_OnWindowMinimized(const WindowMinimizedEvent& _event)
    {
        m_appActive = false;
        return false;
    }

    bool GUIApplication::_OnWindowRestored(const WindowRestoredEvent& _event)
    {
        m_appActive = true;
        return false;
    }

    bool GUIApplication::_OnWindowClosed(const WindowClosedEvent& _event)
    {
        m_appRunning = false;
        return false;
    }

}
