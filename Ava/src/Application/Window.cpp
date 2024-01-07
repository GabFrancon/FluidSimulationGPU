#include <avapch.h>
#include "Window.h"

#include <Application/GUIApplication.h>
#include <Resources/TextureData.h>
#include <Events/WindowEvent.h>
#include <Time/Profiler.h>
#include <UI/ImGuiTools.h>
#include <Debug/Assert.h>
#include <Debug/Log.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <GLFW/glfw3.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    // ---- Window lifecycle ----------------------------------------

    Window::Window(const WindowSettings& _settings)
    {
        m_title = _settings.title;
        m_windowFrame.width = _settings.width;
        m_windowFrame.height = _settings.height;
        AVA_CORE_INFO("Creating window '%s' (%dx%d)", m_title.c_str(), m_windowFrame.width, m_windowFrame.height);

        // GLFW context initialization
        const bool success = glfwInit();
        AVA_ASSERT(success, "Could not initialize GLFW!");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        if (_settings.customTitlebar)
        {
            glfwWindowHint(GLFW_TITLEBAR, GLFW_FALSE);
        }

        // GLFW error callback
        glfwSetErrorCallback([](const int _error, const char* _description)
            {
                AVA_CORE_ERROR("[GLFW] error %d: %s", _error, _description);
            });

        // GLFW window initialization
        m_windowHandle = glfwCreateWindow((int)m_windowFrame.width, (int)m_windowFrame.height, m_title.c_str(), nullptr, nullptr);

        // Window icon
        if (!_settings.iconPath.empty())
        {
            const char* iconPath = _settings.iconPath.c_str();
            TextureData iconData;

            if (TextureLoader::Load(iconPath, iconData))
            {
                GLFWimage icon{};
                icon.width = iconData.width;
                icon.height = iconData.height;
                icon.pixels = static_cast<unsigned char*>(iconData.GetMipData(0, 0));

                glfwSetWindowIcon(m_windowHandle, 1, &icon);
                TextureLoader::Release(iconData);
            }
            else
            {
                AVA_CORE_ERROR("[Window] failed to load app icon '%s'.", iconPath);
            }
        }

        // Window dark mode
        _SetDarkMode(m_windowHandle, _settings.darkMode);

        // Window cursor
        glfwSetInputMode(m_windowHandle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        // Window user pointer
        glfwSetWindowUserPointer(m_windowHandle, &m_callbackData);
    }

    Window::~Window()
    {
        glfwDestroyWindow(m_windowHandle);
        glfwTerminate();
    }

    void Window::Process()
    {
        AUTO_CPU_MARKER("Window Process");

        // Flushes pending updates.
        while (!m_pendingSyncUpdates.empty())
        {
            auto& func = m_pendingSyncUpdates.front();
            func();

            m_pendingSyncUpdates.pop();
        }

        // Triggers event callbacks.
        glfwPollEvents();
    }

    void Window::DrawTitlebar(Texture* _appIcon/*= nullptr*/)
    {
        // each titlebar style element is derived from one of these parameters
        constexpr float titlebarHeight = 65.f;
        constexpr ImU32 titlebarColor = IM_COL32(15, 15, 15, 255);

        // we add padding in fullscreen to get consistent titlebar size
        const float titlebarPadding = IsMaximized() ? 12.f : 0.f;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {titlebarPadding, titlebarPadding});

        ImGuiViewport* viewport = ImGui::GetMainViewport();
        const float windowWidth = viewport->Size.x;
        const float windowHeight = titlebarHeight + titlebarPadding;
        const float titlebarWidth = windowWidth - 2.f * titlebarPadding;
        const ImU32 textColor = ImColor(ImGui::GetStyle().Colors[ImGuiCol_Text]);

        // no rounding nor borders
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);

        // no usual window controls (resize, collapse, etc)
        constexpr ImGuiWindowFlags windowFlags = 
            ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | 
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::SetNextWindowSize(ImVec2(windowWidth, windowHeight));
        ImGui::Begin("Titlebar", (bool*)true, windowFlags);

        // draw titlebar opaque background
        ImGui::GetForegroundDrawList()->AddRectFilled(viewport->Pos, {windowWidth, windowHeight}, titlebarColor);

        constexpr int buttonsCount = 3; // iconify, maximize, close
        constexpr float buttonWidth = 1.5f * titlebarHeight; // reasonable proportions
        constexpr float buttonHeight = titlebarHeight;

        // add drag area (all width except the window buttons area)
        ImGui::SetCursorPos({titlebarPadding, titlebarPadding});
        ImGui::InvisibleButton("Drag area", {titlebarWidth - buttonsCount * buttonWidth, titlebarHeight});
        m_isTitlebarHovered = ImGui::IsItemHovered();

        // draw app logo at the left
        if (_appIcon)
        {
            constexpr float logoSize = 0.8f * titlebarHeight; // reasonable proportions
            constexpr float logoPadding = (titlebarHeight - logoSize) / 2.f;
            const float logoOffsetX = titlebarPadding + logoPadding;
            const float logoOffsetY = titlebarPadding + logoPadding;

            static ShadedTexture shAppIcon;
            shAppIcon.texture = _appIcon;
            ImGui::GetForegroundDrawList()->AddImage(&shAppIcon, {logoOffsetX, logoOffsetY}, {logoOffsetX + logoSize, logoOffsetY + logoSize});
        }

        // draw app title at the center
        ImGui::PushFont(IMGUI_FONT_BOLD);
        const char* title = m_title.c_str();
        const auto titleSize = ImGui::CalcTextSize(title);
        const float titleOffsetX = titlebarPadding + (titlebarWidth - titleSize.x) / 2.f;
        const float titleOffsetY = titlebarPadding + (titlebarHeight - titleSize.y) / 2.f;
        ImGui::GetForegroundDrawList()->AddText({titleOffsetX, titleOffsetY}, textColor, title);
        ImGui::PopFont();

        auto colorWithMultipliedValue = [](const ImU32 _color, const float _multiplier)
        {
            const ImVec4& colRow = ImColor(_color).Value;
            float hue, sat, val;
            ImGui::ColorConvertRGBtoHSV(colRow.x, colRow.y, colRow.z, hue, sat, val);
            return ImU32(ImColor::HSV(hue, sat, std::min(val * _multiplier, 1.0f)));
        };

        // setup buttons theme
        constexpr ImU32 buttonIdleColor = titlebarColor;
        const ImU32 buttonHoveredColor = colorWithMultipliedValue(buttonIdleColor, 1.7f);
        const ImU32 buttonPressedColor = colorWithMultipliedValue(buttonIdleColor, 1.2f);

        // draw window minimize button
        {
            const float buttonOffsetX = titlebarPadding + titlebarWidth - 3.f * buttonWidth;
            const float buttonOffsetY = titlebarPadding + titlebarHeight - buttonHeight;
            ImGui::SetCursorPos({buttonOffsetX, buttonOffsetY});

            if (ImGui::InvisibleButton("Minimize", {buttonWidth, buttonHeight}))
            {
                _QueueSyncUpdate([this]
                {
                    glfwIconifyWindow(m_windowHandle);
                });
            }

            const ImU32 buttonColor = ImGui::IsItemActive() ? buttonPressedColor
                                    : ImGui::IsItemHovered() ? buttonHoveredColor
                                    : buttonIdleColor;

            const auto* icon = ICON_FA_WINDOW_MINIMIZE;
            const auto iconSize = ImGui::CalcTextSize(icon);
            const float buttonIconOffsetX = buttonOffsetX + (buttonWidth - iconSize.x) / 2.f;
            const float buttonIconOffsetY = buttonOffsetY + (buttonHeight - 1.5f * iconSize.y) / 2.f;

            ImGui::GetForegroundDrawList()->AddRectFilled({buttonOffsetX, buttonOffsetY}, {buttonOffsetX + buttonWidth, buttonOffsetY + buttonHeight}, buttonColor);
            ImGui::GetForegroundDrawList()->AddText({buttonIconOffsetX, buttonIconOffsetY}, textColor, icon);
        }

        // draw window maximize button
        {
            const float buttonOffsetX = titlebarPadding + titlebarWidth - 2.f * buttonWidth;
            const float buttonOffsetY = titlebarPadding + titlebarHeight - buttonHeight;
            ImGui::SetCursorPos({buttonOffsetX, buttonOffsetY});

            if (ImGui::InvisibleButton("Maximize", {buttonWidth, buttonHeight}))
            {
                _QueueSyncUpdate([this]
                {
                    if (IsMaximized()) {
                        glfwRestoreWindow(m_windowHandle);
                    }
                    else {
                        glfwMaximizeWindow(m_windowHandle);
                    }
                });
            }

            const ImU32 buttonColor = ImGui::IsItemActive() ? buttonPressedColor
                                    : ImGui::IsItemHovered() ? buttonHoveredColor
                                    : buttonIdleColor;

            const char* icon = IsMaximized() ? ICON_FA_WINDOW_RESTORE : ICON_FA_SQUARE;
            const auto iconSize = ImGui::CalcTextSize(icon);
            const float buttonIconOffsetX = buttonOffsetX + (buttonWidth - iconSize.x) / 2.f;
            const float buttonIconOffsetY = buttonOffsetY + (buttonHeight - iconSize.y) / 2.f;

            ImGui::GetForegroundDrawList()->AddRectFilled({buttonOffsetX, buttonOffsetY}, {buttonOffsetX + buttonWidth, buttonOffsetY + buttonHeight}, buttonColor);
            ImGui::GetForegroundDrawList()->AddText({buttonIconOffsetX, buttonIconOffsetY}, textColor, icon);
        }

        // draw app close button
        {
            const float buttonOffsetX = titlebarPadding + titlebarWidth - buttonWidth;
            const float buttonOffsetY = titlebarPadding + titlebarHeight - buttonHeight;
            ImGui::SetCursorPos({buttonOffsetX, buttonOffsetY});

            if (ImGui::InvisibleButton("Close", {buttonWidth, buttonHeight}))
            {
                GUIApplication::GetInstance().Close();
            }

            const ImU32 buttonColor = ImGui::IsItemActive() ? buttonPressedColor
                                    : ImGui::IsItemHovered() ? buttonHoveredColor
                                    : buttonIdleColor;

            ImGui::PushFont(IMGUI_FONT_BOLD);

            const auto* icon = ICON_FA_XMARK;
            const auto iconSize = ImGui::CalcTextSize(icon);
            const float buttonIconOffsetX = buttonOffsetX + (buttonWidth - iconSize.x) / 2.f;
            const float buttonIconOffsetY = buttonOffsetY + (buttonHeight - iconSize.y) / 2.f;

            ImGui::GetForegroundDrawList()->AddRectFilled({buttonOffsetX, buttonOffsetY}, {buttonOffsetX + buttonWidth, buttonOffsetY + buttonHeight}, buttonColor);
            ImGui::GetForegroundDrawList()->AddText({buttonIconOffsetX, buttonIconOffsetY}, textColor, icon);

            ImGui::PopFont();
        }

        ImGui::PopStyleVar(3);
        ImGui::End();

        // set viewport working area for next UI
        viewport->WorkPos = {viewport->Pos.x, viewport->Pos.y + titlebarHeight};
        viewport->WorkSize = {viewport->Size.x, viewport->Size.y - titlebarHeight};
    }


    // ---- Window events -------------------------------------------

    void Window::SetEventCallback(const EventCallbackFunc& _callback)
    {
        // Update window event callback
         m_eventCallback = _callback;

        // Update window pointer accessible from GLFW callbacks
        m_callbackData.window = this;

        // Set titlebar hovered callback
        glfwSetTitlebarHitTestCallback(m_windowHandle, _TitlebarHitTestCallback);

        // Set window resized callback
        glfwSetWindowSizeCallback(m_windowHandle, _WindowSizeCallback);

        // Set window moved callback
        glfwSetWindowPosCallback(m_windowHandle, _WindowPosCallback);

        // Set window minimized / restored callback
        glfwSetWindowIconifyCallback(m_windowHandle, _WindowIconifyCallback);

        // Set window maximized / restored callback
        glfwSetWindowMaximizeCallback(m_windowHandle, _WindowMaximizeCallback);

        // Set window focused callback
        glfwSetWindowFocusCallback(m_windowHandle, _WindowFocusCallback);

        // Set window closed callback
        glfwSetWindowCloseCallback(m_windowHandle, _WindowCloseCallback);
    }

    void Window::_TitlebarHitTestCallback(GLFWwindow* _window, int _xPos, int _yPos, int* _hit)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);
        *_hit = data.window->m_isTitlebarHovered;
    }

    void Window::_WindowSizeCallback(GLFWwindow* _window, const int _width, const int _height)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);
        data.window->m_windowFrame.width = (u32)_width;
        data.window->m_windowFrame.height = (u32)_height;

        // Propagates window resized event.
        WindowResizedEvent event(_width, _height);
        data.window->m_eventCallback(event);
    }

    void Window::_WindowPosCallback(GLFWwindow* _window, const int _xPos, const int _yPos)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);
        data.window->m_windowFrame.x = (u32)_xPos;
        data.window->m_windowFrame.y = (u32)_yPos;

        // Propagates window moved event.
        WindowMovedEvent event(_xPos, _yPos);
        data.window->m_eventCallback(event);
    }

    void Window::_WindowIconifyCallback(GLFWwindow* _window, const int _minimized)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);

        if (_minimized)
        {
            data.window->m_currentState = WindowMinimized;

            // Propagates window minimized event.
            WindowMinimizedEvent event;
            data.window->m_eventCallback(event);
        }
        else
        {
            data.window->m_currentState = WindowRestored;

            // Propagates window restored event.
            WindowRestoredEvent event;
            data.window->m_eventCallback(event);
        }
    }

    void Window::_WindowMaximizeCallback(GLFWwindow* _window, const int _maximized)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);

        if (_maximized)
        {
            data.window->m_currentState = WindowMaximized;

            // Propagates window maximized event.
            WindowMaximizedEvent event;
            data.window->m_eventCallback(event);
        }
        else
        {
            data.window->m_currentState = WindowRestored;

            // Propagates window restored event.
            WindowRestoredEvent event;
            data.window->m_eventCallback(event);
        }
    }

    void Window::_WindowFocusCallback(GLFWwindow* _window, const int _focused)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);
        data.window->m_windowFocused = _focused > 0;

        if (_focused)
        {
            // Propagates window focused event.
            WindowFocusedEvent event;
            data.window->m_eventCallback(event);
        }
    }

    void Window::_WindowCloseCallback(GLFWwindow* _window)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);

        // Propagates window closed event.
        WindowClosedEvent event;
        data.window->m_eventCallback(event);
    }


    // ---- Window getters ------------------------------------------

    bool Window::IsMinimized() const
    {
        return m_currentState == WindowMinimized;
    }

    bool Window::IsMaximized() const
    {
        return m_currentState == WindowMaximized;
    }

    bool Window::IsFocused() const
    {
        return m_windowFocused;
    }

    Vec2f Window::GetExtents() const
    {
        return { static_cast<float>(m_windowFrame.width), static_cast<float>(m_windowFrame.height) };
    }

    Vec2f Window::GetOrigin() const
    {
        return { static_cast<float>(m_windowFrame.x), static_cast<float>(m_windowFrame.y) };
    }

    float Window::GetWidth() const
    {
        return static_cast<float>(m_windowFrame.width);
    }

    float Window::GetHeight() const
    {
        return static_cast<float>(m_windowFrame.height);
    }

    float Window::GetPositionX() const
    {
        return static_cast<float>(m_windowFrame.x);
    }

    float Window::GetPositionY() const
    {
        return static_cast<float>(m_windowFrame.y);
    }

    Rect Window::GetRelativeWindowFrame() const
    {
        Rect relativeFrame = m_windowFrame;
        relativeFrame.x = 0u;
        relativeFrame.y = 0u;
        return relativeFrame;
    }

    Rect Window::GetAbsoluteWindowFrame() const
    {
        return m_windowFrame;
    }

    GLFWwindow* Window::GetWindowHandle() const
    {
        return m_windowHandle;
    }

    void* Window::GetNativeWindowHandle() const
    {
        return _GetNativeHandle(m_windowHandle);
    }
}
