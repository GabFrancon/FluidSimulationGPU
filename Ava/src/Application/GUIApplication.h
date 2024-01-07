#pragma once
/// @file GUIApplication.h
/// @brief

#include <Application/Application.h>
#include <Application/Window.h>
#include <Memory/Memory.h>
#include <Layers/Layer.h>

namespace Ava {

    struct WindowSettings;
    struct InputSettings;
    struct GraphicsSettings;
    struct ProfilerSettings;

    /// @brief app settings for GUIApplication instances.
    struct GUIAppSettings
    {
        WindowSettings* window = nullptr;
        InputSettings* inputs = nullptr;
        GraphicsSettings* graphics = nullptr;
        ProfilerSettings* profiler = nullptr;
    };

    /// @brief Subclass of Application tailored for traditional graphical user interface (GUI) applications. Extends the
    /// base Application class by initializing additional managers and components for handling user input, UI elements,
    /// and graphics. Ideal for creating applications that provide a visual and interactive experience for users, such as
    /// games, productivity software, or any application with a graphical front-end.
    class GUIApplication : public Application
    {
        static GUIApplication* s_instance;

    public:
        explicit GUIApplication(const GUIAppSettings& _settings);
        ~GUIApplication() override;

        static GUIApplication& GetInstance() { return *s_instance; }
        Window* GetWindow() const { return m_window.get(); }

        void Run() override;
        void ProcessEvent(Event& _event);
        void Close();

        template <typename LayerType, typename... Args>
        Ref<LayerType> PushLayer(Args&&... _args)
        {
            static_assert(std::is_base_of_v<Layer, LayerType>, "LayerType must inherit from Ava::Layer.");
            Ref<LayerType> layer = CreateRef<LayerType>(std::forward<Args>(_args)...);
            m_layerStack.PushLayer(layer);
            return layer;
        }

        template <typename LayerType, typename... Args>
        Ref<LayerType> PushOverlay(Args&&... _args)
        {
            static_assert(std::is_base_of_v<Layer, LayerType>, "LayerType must inherit from Ava::Layer.");
            Ref<LayerType> overlay = CreateRef<LayerType>(std::forward<Args>(_args)...);
            m_layerStack.PushOverlay(overlay);
            return overlay;
        }

        template <typename LayerType>
        void PopLayer(const Ref<LayerType>& _layer)
        {
            static_assert(std::is_base_of_v<Layer, LayerType>, "LayerType must inherit from Ava::Layer.");
            m_layerStack.PopLayer(_layer);
        }

        template <typename LayerType>
        void PopOverlay(const Ref<LayerType>& _overlay)
        {
            static_assert(std::is_base_of_v<Layer, LayerType>, "LayerType must inherit from Ava::Layer.");
            m_layerStack.PopOverlay(_overlay);
        }

    private:
        bool _OnWindowMinimized(const WindowMinimizedEvent& _event);
        bool _OnWindowRestored(const WindowRestoredEvent& _event);
        bool _OnWindowClosed(const WindowClosedEvent& _event);

        LayerStack m_layerStack;
        Scope<Window> m_window;

        bool m_appRunning = true;
        bool m_appActive = true;
    };

}

