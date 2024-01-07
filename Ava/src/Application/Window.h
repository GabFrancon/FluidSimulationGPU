#pragma once
/// @file Window.h
/// @brief

#include <Math/Types.h>
#include <Math/Geometry.h>
#include <Events/Event.h>

// Forward declare GLFW structures
struct GLFWwindow;
struct GLFWcursor;

namespace Ava {

    /// @brief Window properties specification.
    struct WindowSettings
    {
        std::string title = "Ava";
        std::string iconPath = "";

        u32 width = 1920;
        u32 height = 1080;

        bool darkMode = true;
        bool customTitlebar = false;
    };

    /// @brief Interface representing a window.
    class Window
    {
        friend class InputMgr;

    public:
        Window(const WindowSettings& _settings);
        ~Window();

        /// @brief Poll events.
        void Process();
        /// @brief Emits ImGui draw commands  to render the custom titlebar.
        void DrawTitlebar(Texture* _appIcon = nullptr);
        /// @brief Sets the function to call when a new window event is generated.
        void SetEventCallback(const EventCallbackFunc& _callback);

        /// @brief Returns true if the window is currently minimized.
        bool IsMinimized() const;
        /// @brief Returns true if the window if currently maximized.
        bool IsMaximized() const;
        /// @brief Returns true if the window is currently focused.
        bool IsFocused() const;
        /// @brief Returns the window size, in pixels.
        Vec2f GetExtents() const;
        /// @brief Returns the window origin, in pixels.
        Vec2f GetOrigin() const;
        /// @brief Returns the window screen width, in pixels.
        float GetWidth() const;
        /// @brief Returns the window screen height, in pixels.
        float GetHeight() const;
        /// @brief Returns the window screen origin along horizontal axis, in pixels.
        float GetPositionX() const;
        /// @brief Returns the window screen origin along vertical axis, in pixels.
        float GetPositionY() const;
        /// @brief Returns the window frame relative to window origin.
        Rect GetRelativeWindowFrame() const;
        /// @brief Returns the window frame relative to screen origin.
        Rect GetAbsoluteWindowFrame() const;
        /// @brief Returns a pointer to the GLFW window handle.
        GLFWwindow* GetWindowHandle() const;
        /// @brief Returns a pointer to the native window handle.
        void* GetNativeWindowHandle() const;

    private:
        /// @brief Structure containing data to pass to GLFW static callbacks.
        struct CallbackData
        {
            /// used by window callbacks
            Window* window = nullptr;
            /// user by input callbacks
            InputMgr* inputMgr = nullptr;
        };

        GLFWwindow* m_windowHandle;
        CallbackData m_callbackData{};

        enum WindowState
        {
            WindowMinimized,
            WindowMaximized,
            WindowRestored
        };

        std::string m_title = "Ava";
        WindowState m_currentState = WindowRestored;
        EventCallbackFunc m_eventCallback;
        bool m_isTitlebarHovered = false;
        bool m_windowFocused = true;
        Rect m_windowFrame{};

        // pending updates
        std::queue<std::function<void()>> m_pendingSyncUpdates;

        // sync updates
        template <typename Func>
        void _QueueSyncUpdate(Func&& _func)
        {
            m_pendingSyncUpdates.push(_func);
        }

        // GLFW window event callbacks
        static void _TitlebarHitTestCallback(GLFWwindow* _window, int _xPos, int _yPos, int* _hit);
        static void _WindowSizeCallback(GLFWwindow* _window, int _width, int _height);
        static void _WindowPosCallback(GLFWwindow* _window, int _xPos, int _yPos);
        static void _WindowIconifyCallback(GLFWwindow* _window, int _minimized);
        static void _WindowMaximizeCallback(GLFWwindow* _window, int _maximized);
        static void _WindowFocusCallback(GLFWwindow* _window, int _focused);
        static void _WindowCloseCallback(GLFWwindow* _window);

        // Platform dependent static helpers
        static void* _GetNativeHandle(GLFWwindow* _windowHandle);
        static void _SetDarkMode(GLFWwindow* _windowHandle, bool _enable);

    };

}
