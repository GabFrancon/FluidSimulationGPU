#pragma once
/// @file UILayer.h
/// @brief Template for an ImGui-based UI Layer.

#include <Layers/Layer.h>
#include <Events/Event.h>
#include <UI/UICommon.h>

// Forward declare ImGui structures
struct ImDrawData;

namespace Ava {

    class Texture;
    class FrameBuffer;
    class ShaderProgram;

    /// @brief ImGui layer settings, required to create a UILayer.
    struct ImGuiSettings
    {
        float fontScale = 2.f;
        UI::ThemePreset theme = UI::ThemeClassic;
        std::string configFilePath = "imgui.ini";
        u32 blockingEventFlags = AVA_EVENT_NONE;
        u32 configFlags = AVA_UI_NONE;
    };

    /// @brief UI layer based on the ImGui library.
    class UILayer : public Layer
    {
    public:
        explicit UILayer(const ImGuiSettings& _settings);
        ~UILayer() override;

        void OnAttach() override;
        void OnUpdate(Timestep& _dt) override;
        void OnRender(GraphicsContext* _ctx) override;
        void OnEvent(Event& _event) override;
        void OnDetach() override;

        void SetConfigFlag(UIConfigFlags _flag, bool _enable);
        void SetConfigFlags(u32 _flags);

        void SetBlockingEvent(EventFlags _flag, bool _enable);
        void SetBlockingEvents(u32 _flags);

        void SetFramebuffer(FrameBuffer* _framebuffer);
        void SetTheme(UI::ThemePreset _theme);

    private:
        bool _HasConfigFlag(UIConfigFlags _flag) const;
        bool _ShouldBlockEvent(EventFlags _flag) const;

        // Backend implementation
        void _UpdateInputs(const Timestep& _dt, const Window* _window) const;
        void _RenderDrawCommands(const GraphicsContext* _ctx, const ImDrawData* _drawData) const;

        // Event callbacks
        bool _OnWindowFocused(const WindowFocusedEvent& _event) const;
        bool _OnMouseButtonPressed(const MouseButtonPressedEvent& _event) const;
        bool _OnMouseButtonReleased(const MouseButtonReleasedEvent& _event) const;
        bool _OnMouseEnteredFrame(const MouseEnteredFrameEvent& _event) const;
        bool _OnMouseExitedFrame(const MouseExitedFrameEvent& _event) const;
        bool _OnMouseMoved(const MouseMovedEvent& _event) const;
        bool _OnMouseScrolled(const MouseScrolledEvent& _event) const;
        bool _OnKeyPressed(const KeyPressedEvent& _event) const;
        bool _OnKeyReleased(const KeyReleasedEvent& _event) const;
        bool _OnKeyTyped(const KeyTypedEvent& _event) const;

        // Clipboard inputs
        static void _SetClipboardText(void* _userData, const char* _text);
        static const char* _GetClipboardText(void* _userData);

        // UI settings
        ImGuiSettings m_settings;

        // font resources
        Texture* m_fontTexture = nullptr;
        ShaderProgram* m_fontShaderProgram = nullptr;
        ShadedTexture m_fontAtlas;

        // user-defined render objects
        FrameBuffer* m_targetFramebuffer = nullptr;
    };

}

