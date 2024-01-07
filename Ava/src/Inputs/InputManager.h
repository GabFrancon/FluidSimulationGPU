#pragma once
/// @file InputManager.h
/// @brief

#include <Math/Types.h>
#include <Events/Event.h>
#include <Application/Window.h>

#include <Inputs/KeyCodes.h>
#include <Inputs/MouseCodes.h>
#include <Inputs/GamepadCodes.h>

namespace Ava {

    /// @brief Input settings, required to create an InputMgr.
    struct InputSettings
    {
        bool trackMouse = false;
        bool useGamepad = false;
        Mouse::CursorMode cursorMode = Mouse::ModeNormal;
    };

    /// @brief Input manager, handles system inputs.
    class InputMgr
    {
        static InputMgr* s_instance;

    public:
        static void Init(const InputSettings& _settings);
        static void Shutdown();

        static InputMgr* GetInstance() { return s_instance; }
        const InputSettings& GetSettings() const { return m_settings; }

        /// @brief Sets the function to call when a new input event is generated.
        void SetEventCallback(const EventCallbackFunc& _callback);

        /// @brief Updates the current input states of all connected controllers.
        void Process();

        // --- Keyboard -------------------------------------------------------

        /// @brief Returns true if _key is being pressed on the detected keyboard.
        bool IsKeyPressed(Keyboard::Key _key) const;
        /// @brief Copies _text to the clipboard.
        static void CopyToClipBoard(const char* _text);
        /// @brief Returns the current text stored in the clipboard.
        static const char* GetClipBoard();

        // --- Mouse ----------------------------------------------------------

        /// @brief Returns true if _button is being pressed on the detected mouse.
        bool IsMouseButtonPressed(Mouse::Button _button) const;
        /// @brief Switches to the given mouse cursor mode.
        void SetMouseCursorMode(Mouse::CursorMode _mode);
        /// @brief Returns the current mouse cursor mode enabled.
        Mouse::CursorMode GetMouseCursorMode() const;
        /// @brief Display the given mouse cursor icon.
        void SetMouseCursorIcon(Mouse::CursorIcon _icon);
        /// @brief Returns the current mouse cursor icon displayed.
        Mouse::CursorIcon GetMouseCursorIcon() const;
        /// @brief Sets the mouse cursor position in pixels, relative to the upper-left corner of the window
        void SetMouseCursorPosition(const Vec2f& _position);
        /// @brief Returns the mouse cursor position in pixels, relative to the upper-left corner of the window.
        Vec2f GetMouseCursorPosition() const;
        /// @brief Returns true if the mouse cursor stands within the window frame.
        bool IsMouseCursorWithinFrame() const;
        /// @brief Returns the mouse wheel position in increments, relative to the initial position when engine started.
        Vec2f GetMouseWheelPosition() const;
        /// @brief Returns the number of pixels the mouse cursor moved since last frame.
        Vec2f GetLastFrameMouseCursorOffset() const;
        /// @brief Returns the number of increments the mouse wheel scrolled since last frame.
        Vec2f GetLastFrameMouseWheelOffset() const;

        // --- Gamepad --------------------------------------------------------

        /// @brief Checks if a gamepad is connected to the device.
        bool IsGamepadConnected() const;
        /// @brief Returns true if _button is being pressed on the detected gamepad.
        bool IsGamepadButtonPressed(Gamepad::Button _button) const;
        /// @brief Returns current position of specified joystick in [-1, 1] (-1 = upper-left corner / 0 = idle / 1 = bottom-right corner).
        Vec2f GetGamepadJoystick(Gamepad::Side _side) const;
        /// @brief Returns current position of specified trigger in [-1, 1]  (-1 = idle / 0 = pressed halfway / 1 = completely pressed).
        float GetGamepadTrigger(Gamepad::Side _side) const;

    private:
        explicit InputMgr(const InputSettings& _settings);
        ~InputMgr();

        // init / shutdown helpers
        void _CreateMouseCursorIcons();
        void _FreeMouseCursorIcons();

        // runtime helpers
        void _UpdateMouseTracking();
        void _UpdateGamepadState();

        static void _CharCallback(GLFWwindow* _window, unsigned int _keycode);
        static void _KeyCallback(GLFWwindow* _window, int _key, int _scanCode, int _action, int _mods);
        static void _MouseButtonCallback(GLFWwindow* _window, int _button, int _action, int _mods);
        static void _CursorEnterCallback(GLFWwindow* _window, int _entered);
        static void _CursorPosCallback(GLFWwindow* _window, double _xPos, double _yPos);
        static void _ScrollCallback(GLFWwindow* _window, double _xOffset, double _yOffset);
        InputSettings m_settings;
        EventCallbackFunc m_eventCallback;

        // Keyboard state
        struct
        {
            std::array<bool, Keyboard::KeyCount> buttons { false };
        } m_keyboard;

        // Mouse state
        struct
        {
            bool cursorWithinFrame = true;

            std::array<bool,        Mouse::ButtonCount> buttons { false };
            std::array<GLFWcursor*, Mouse::IconCount>   icons { nullptr };

            Mouse::CursorMode currentCursorMode = Mouse::ModeNormal;
            Mouse::CursorIcon currentCursorIcon = Mouse::IconArrow;

            Vec2f previousWheelPosition = { 0.f, 0.f };
            Vec2f currentWheelPosition  = { 0.f, 0.f };
            Vec2f frameWheelOffset      = { 0.f, 0.f };

            Vec2f previousCursorPosition = { 0.f, 0.f };
            Vec2f currentCursorPosition  = { 0.f, 0.f };
            Vec2f frameCursorOffset      = { 0.f, 0.f };
        } m_mouse;

        // Gamepad state
        struct
        {
            bool connected = false;
            std::array<bool,  Gamepad::ButtonCount> buttons   { false };
            std::array<float, Gamepad::SideCount>   triggers  { -1.f };
            std::array<Vec2f, Gamepad::SideCount>   joysticks { Vec2f(0.f) };
        } m_gamepad;
    };

}

#define INPUT_MGR Ava::InputMgr::GetInstance()