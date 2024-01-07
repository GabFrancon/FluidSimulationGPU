#include <avapch.h>
#include "InputManager.h"

#include <Application/GUIApplication.h>
#include <Events/GamepadEvent.h>
#include <Events/MouseEvent.h>
#include <Events/KeyEvent.h>
#include <Time/Profiler.h>
#include <Math/Math.h>
#include <Debug/Log.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <GLFW/glfw3.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    // ----- Input manager lifecycle ----------------------------------------------------------

    InputMgr* InputMgr::s_instance = nullptr;

    InputMgr::InputMgr(const InputSettings& _settings)
        : m_settings(_settings)
    {
        // creates default mouse cursor icons
        _CreateMouseCursorIcons();

        // updates mouse cursor icon
        SetMouseCursorIcon(Mouse::IconArrow);

        // updates mouse cursor mode
        SetMouseCursorMode(_settings.cursorMode);
    }

    InputMgr::~InputMgr()
    {
        _FreeMouseCursorIcons();
    }

    void InputMgr::Init(const InputSettings& _settings)
    {
        if (!s_instance)
        {
            s_instance = new InputMgr(_settings);
        }
    }

    void InputMgr::Shutdown()
    {
        if (s_instance)
        {
            delete s_instance;
        }
    }

    void InputMgr::Process()
    {
        AUTO_CPU_MARKER("INPUT_MGR Process");

        // fetch full mouse state
        if (m_settings.trackMouse)
        {
            _UpdateMouseTracking();
        }

        // fetch full gamepad state
        if (m_settings.useGamepad)
        {
            _UpdateGamepadState();
        }
    }


    // ----- Input events ---------------------------------------------------------------------

    void InputMgr::SetEventCallback(const EventCallbackFunc& _callback)
    {
        Window* window = GUIApplication::GetInstance().GetWindow();
        GLFWwindow* windowHandle = window->m_windowHandle;
        Window::CallbackData& callbackData = window->m_callbackData;

        // Update input event callback
         m_eventCallback = _callback;

        // Update window pointer accessible from GLFW callbacks
        callbackData.inputMgr = this;

        // Set key typed callback
        glfwSetCharCallback(windowHandle, _CharCallback);

        // Set key pressed/released callback
        glfwSetKeyCallback(windowHandle, _KeyCallback);

        // Set mouse button pressed/released callback
        glfwSetMouseButtonCallback(windowHandle, _MouseButtonCallback);

        // Set mouse moved callback
        glfwSetCursorPosCallback(windowHandle, _CursorPosCallback);

        // Set mouse entered callback
        glfwSetCursorEnterCallback(windowHandle, _CursorEnterCallback);

        // Set mouse scrolled callback
        glfwSetScrollCallback(windowHandle, _ScrollCallback);
    }

    void InputMgr::_CharCallback(GLFWwindow* _window, const unsigned _keycode)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);

        // Propagates key typed event.
        KeyTypedEvent event(_keycode);
        data.inputMgr->m_eventCallback(event);
    }

    void InputMgr::_KeyCallback(GLFWwindow* _window, const int _key, const int _scanCode, const int _action, const int _mods)
    {
        // GLFW sometimes returns an invalid key
        if (_key < Keyboard::FirstKey || _key >= Keyboard::KeyCount)
        {
            return;
        }

        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);
        const auto key = Keyboard::Key(_key);
        const auto mods = (u16)_mods;

        switch (_action)
        {
            case GLFW_PRESS:
            {
                data.inputMgr->m_keyboard.buttons[key] = true;

                // Propagates key pressed event.
                KeyPressedEvent event(key, mods);
                data.inputMgr->m_eventCallback(event);
                break;
            }
            case GLFW_RELEASE:
            {
                data.inputMgr->m_keyboard.buttons[key] = false;

                // Propagates key released event.
                KeyReleasedEvent event(key, mods);
                data.inputMgr->m_eventCallback(event);
                break;
            }
            default:
                break;
        }
    }

    void InputMgr::_MouseButtonCallback(GLFWwindow* _window, const int _button, const int _action, const int _mods)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);
        const auto button = Mouse::Button(_button);
        const auto mods = (u16)_mods;

        switch (_action)
        {
            case GLFW_PRESS:
            {
                data.inputMgr->m_mouse.buttons[button] = true;

                // Propagates button pressed event.
                MouseButtonPressedEvent event(button, mods);
                data.inputMgr->m_eventCallback(event);
                break;
            }
            case GLFW_RELEASE:
            {
                data.inputMgr->m_mouse.buttons[button] = false;

                // Propagates button released event.
                MouseButtonReleasedEvent event(button, mods);
                data.inputMgr->m_eventCallback(event);
                break;
            }
        default:
            break;
        }
    }

    void InputMgr::_CursorEnterCallback(GLFWwindow* _window, const int _entered)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);

        if (_entered)
        {
            data.inputMgr->m_mouse.cursorWithinFrame = true;

            // Propagates mouse entered frame event.
            MouseEnteredFrameEvent event;
            data.inputMgr->m_eventCallback(event);
        }
        else
        {
            data.inputMgr->m_mouse.cursorWithinFrame = false;

            // Propagates mouse exited frame event.
            MouseExitedFrameEvent event;
            data.inputMgr->m_eventCallback(event);
        }
    }

    void InputMgr::_CursorPosCallback(GLFWwindow* _window, const double _xPos, const double _yPos)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);

        // Mouse moved callback returns the current screen position.
        data.inputMgr->m_mouse.currentCursorPosition.x = (float)_xPos;
        data.inputMgr->m_mouse.currentCursorPosition.y = (float)_yPos;

        // Propagates mouse moved event. 
        MouseMovedEvent event((float)_xPos, (float)_yPos);
        data.inputMgr->m_eventCallback(event);
    }

    void InputMgr::_ScrollCallback(GLFWwindow* _window, const double _xOffset, const double _yOffset)
    {
        const auto& data = *(Window::CallbackData*)glfwGetWindowUserPointer(_window);

        // Mouse scrolled callback returns the last scroll offset.
        data.inputMgr->m_mouse.currentWheelPosition.x += (float)_xOffset;
        data.inputMgr->m_mouse.currentWheelPosition.y += (float)_yOffset;

        // Propagates mouse scrolled event.
        MouseScrolledEvent event((float)_xOffset, (float)_yOffset);
        data.inputMgr->m_eventCallback(event);
    }


    // ----- Keyboard -------------------------------------------------------------------------

    bool InputMgr::IsKeyPressed(const Keyboard::Key _key) const
    {
        return m_keyboard.buttons[_key];
    }

    void InputMgr::CopyToClipBoard(const char* _text)
    {
        const Window* window = GUIApplication::GetInstance().GetWindow();
        GLFWwindow* windowHandle = window->m_windowHandle;

        glfwSetClipboardString(windowHandle, _text);
    }

    const char* InputMgr::GetClipBoard()
    {
        const Window* window = GUIApplication::GetInstance().GetWindow();
        GLFWwindow* windowHandle = window->m_windowHandle;

        return glfwGetClipboardString(windowHandle);
    }


    // ----- Mouse ----------------------------------------------------------------------------

    bool InputMgr::IsMouseButtonPressed(const Mouse::Button _button) const
    {
        return m_mouse.buttons[_button];
    }

    void InputMgr::SetMouseCursorMode(const Mouse::CursorMode _mode)
    {
        if (_mode == m_mouse.currentCursorMode)
        {
            // already up-to-date
            return;
        }

        m_mouse.currentCursorMode = _mode;

        const Window* window = GUIApplication::GetInstance().GetWindow();
        GLFWwindow* windowHandle = window->m_windowHandle;

        const int value = _mode == Mouse::ModeDisabled ? GLFW_CURSOR_DISABLED
                        : _mode == Mouse::ModeHidden ? GLFW_CURSOR_HIDDEN
                        : _mode == Mouse::ModeNormal ? GLFW_CURSOR_NORMAL
                        : GLFW_CURSOR_NORMAL;

        glfwSetInputMode(windowHandle, GLFW_CURSOR, value);
    }

    Mouse::CursorMode InputMgr::GetMouseCursorMode() const
    {
        return m_mouse.currentCursorMode;
    }

    void InputMgr::SetMouseCursorIcon(const Mouse::CursorIcon _icon)
    {
        if (_icon == m_mouse.currentCursorIcon)
        {
            // already up-to-date
            return;
        }

        m_mouse.currentCursorIcon = _icon;

        const Window* window = GUIApplication::GetInstance().GetWindow();
        GLFWwindow* windowHandle = window->m_windowHandle;
        glfwSetCursor(windowHandle, m_mouse.icons[_icon]);
    }

    Mouse::CursorIcon InputMgr::GetMouseCursorIcon() const
    {
        return m_mouse.currentCursorIcon;
    }

    void InputMgr::SetMouseCursorPosition(const Vec2f& _position)
    {
        m_mouse.currentCursorPosition = _position;

        const Window* window = GUIApplication::GetInstance().GetWindow();
        GLFWwindow* windowHandle = window->m_windowHandle;
        glfwSetCursorPos(windowHandle, _position.x, _position.y);
    }

    Vec2f InputMgr::GetMouseCursorPosition() const
    {
        return m_mouse.currentCursorPosition;
    }

    bool InputMgr::IsMouseCursorWithinFrame() const
    {
        return m_mouse.cursorWithinFrame;
    }

    Vec2f InputMgr::GetMouseWheelPosition() const
    {
        return m_mouse.currentWheelPosition;
    }

    Vec2f InputMgr::GetLastFrameMouseCursorOffset() const
    {
        return m_mouse.frameCursorOffset;
    }

    Vec2f InputMgr::GetLastFrameMouseWheelOffset() const
    {
        return m_mouse.frameWheelOffset;
    }

    void InputMgr::_CreateMouseCursorIcons()
    {
        // On X11 cursors are user configurable and some cursors may be missing. When a cursor doesn't exist, GLFW
        // will emit an error which will often be printed by the app, so we temporarily disable error reporting.
        const GLFWerrorfun errorCallback = glfwSetErrorCallback(nullptr);

        m_mouse.icons[Mouse::IconArrow]                    = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
        m_mouse.icons[Mouse::IconTextInput]                = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
        m_mouse.icons[Mouse::IconResizeVertical]           = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);
        m_mouse.icons[Mouse::IconResizeHorizontal]         = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
        m_mouse.icons[Mouse::IconHand]                     = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
        m_mouse.icons[Mouse::IconResizeAll]                = glfwCreateStandardCursor(GLFW_RESIZE_ALL_CURSOR);
        m_mouse.icons[Mouse::IconResizeTopLeftBottomRight] = glfwCreateStandardCursor(GLFW_RESIZE_NWSE_CURSOR);
        m_mouse.icons[Mouse::IconResizeTopRightBottomLeft] = glfwCreateStandardCursor(GLFW_RESIZE_NESW_CURSOR);
        m_mouse.icons[Mouse::IconNotAllowed]               = glfwCreateStandardCursor(GLFW_NOT_ALLOWED_CURSOR);

        glfwSetErrorCallback(errorCallback);
    }

    void InputMgr::_FreeMouseCursorIcons()
    {
        for (int i = 0; i < Mouse::IconCount; i++)
        {
            glfwDestroyCursor(m_mouse.icons[i]);
            m_mouse.icons[i] = nullptr;
        }
    }

    void InputMgr::_UpdateMouseTracking()
    {
        // update mouse screen position
        m_mouse.frameCursorOffset.x = m_mouse.currentCursorPosition.x - m_mouse.previousCursorPosition.x;
        m_mouse.frameCursorOffset.y = m_mouse.currentCursorPosition.y - m_mouse.previousCursorPosition.y;
        m_mouse.previousCursorPosition = m_mouse.currentCursorPosition;

        // update mouse scroll position
        m_mouse.frameWheelOffset.x = m_mouse.currentWheelPosition.x - m_mouse.previousWheelPosition.x;
        m_mouse.frameWheelOffset.y = m_mouse.currentWheelPosition.y - m_mouse.previousWheelPosition.y;
        m_mouse.previousWheelPosition = m_mouse.currentWheelPosition;
    }


    // ----- Gamepad --------------------------------------------------------------------------

    bool InputMgr::IsGamepadConnected() const
    {
        return m_gamepad.connected;
    }

    bool InputMgr::IsGamepadButtonPressed(const Gamepad::Button _button) const
    {
        if (!IsGamepadConnected())
        {
            return false;
        }

        return m_gamepad.buttons[_button];
    }

    Vec2f InputMgr::GetGamepadJoystick(const Gamepad::Side _side) const
    {
        if (!IsGamepadConnected())
        {
            return Vec2f(0.f);
        }

        return m_gamepad.joysticks[_side];
    }

    float InputMgr::GetGamepadTrigger(const Gamepad::Side _side) const
    {
        if (!IsGamepadConnected())
        {
            return -1.f;
        }

        return m_gamepad.triggers[_side];
    }

    void InputMgr::_UpdateGamepadState()
    {
        if (!glfwJoystickIsGamepad(GLFW_JOYSTICK_1))
        {
            m_gamepad.connected = false;
            return;
        }

        GLFWgamepadstate state;
        if (!glfwGetGamepadState(GLFW_JOYSTICK_1, &state))
        {
            AVA_WARN("[InputMgr] failed to fetch gamepad state.");
            m_gamepad.connected = false;
            return;
        }

        m_gamepad.connected = true;

        // Buttons
        for (int i = 0u; i < Gamepad::ButtonCount; i++)
        {
            bool buttonPressed = false;
            const auto button = Gamepad::Button(i);

            // L2 and R2 are treated as analogic inputs (like the joysticks)
            // -1.f means the button is released, and 1.f it is fully pressed.
            if (button == Gamepad::L2)
            {
                buttonPressed = state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] > 0.f;
            }
            else if (button == Gamepad::R2)
            {
                buttonPressed = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] > 0.f;
            }
            else
            {
                buttonPressed = state.buttons[button] == GLFW_PRESS;
            }

            if (buttonPressed && !m_gamepad.buttons[button])
            {
                // Propagates button pressed event.
                GamepadButtonPressedEvent event(button);
                m_eventCallback(event);
            }
            else if (!buttonPressed && m_gamepad.buttons[button])
            {
                // Propagates button released event.
                GamepadButtonReleasedEvent event(button);
                m_eventCallback(event);
            }

            m_gamepad.buttons[button] = buttonPressed;
        }

        // Left trigger
        {
            float triggerPos = state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER];
            triggerPos *= Math::step(0.1f, Math::abs(triggerPos));

            if (abs(triggerPos - m_gamepad.triggers[Gamepad::SideLeft]) > FLT_EPSILON)
            {
                // Propagates trigger moved event.
                GamepadTriggerMovedEvent event(Gamepad::SideLeft, triggerPos);
                m_eventCallback(event);
            }

            m_gamepad.triggers[Gamepad::SideLeft] = triggerPos;
        }

        // Right trigger
        {
            float triggerPos = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER];
            triggerPos *= Math::step(0.1f, Math::abs(triggerPos));

            if (abs(triggerPos - m_gamepad.triggers[Gamepad::SideRight]) > FLT_EPSILON)
            {
                // Propagates trigger moved event.
                GamepadTriggerMovedEvent event(Gamepad::SideRight, triggerPos);
                m_eventCallback(event);
            }

            m_gamepad.triggers[Gamepad::SideRight] = triggerPos;
        }

        // Left joystick
        {
            Vec2f joystickPos = { state.axes[GLFW_GAMEPAD_AXIS_LEFT_X], state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y] };
            joystickPos *= Math::step(Vec2f(0.1f), Math::abs(joystickPos));

            if (Math::length(joystickPos - m_gamepad.joysticks[Gamepad::SideLeft]) > FLT_EPSILON)
            {
                // Propagates joystick moved event.
                GamepadJoystickMovedEvent event(Gamepad::SideLeft, joystickPos.x, joystickPos.y);
                m_eventCallback(event);
            }

            m_gamepad.joysticks[Gamepad::SideLeft] = joystickPos;
        }

        // Right joystick
        {
            Vec2f joystickPos = { state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y] };
            joystickPos *= Math::step(Vec2f(0.1f), Math::abs(joystickPos));

            if (Math::length(joystickPos - m_gamepad.joysticks[Gamepad::SideRight]) > FLT_EPSILON)
            {
                // Propagates joystick moved event.
                GamepadJoystickMovedEvent event(Gamepad::SideRight, joystickPos.x, joystickPos.y);
                m_eventCallback(event);
            }

            m_gamepad.joysticks[Gamepad::SideRight] = joystickPos;
        }
    }

}
