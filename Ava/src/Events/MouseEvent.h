#pragma once
/// @file MouseEvent.h
/// @brief

#include <Events/Event.h>
#include <Inputs/MouseCodes.h>

namespace Ava {

    /// @brief Abstract class for mouse button events.
    class MouseButtonEvent : public Event
    {
    public:
        EVENT_FLAGS(AVA_EVENT_INPUT | AVA_EVENT_MOUSE | AVA_EVENT_MOUSE_BUTTON)

        Mouse::Button GetMouseButton() const { return m_button; }
        u16 GetModifiers() const { return m_modifiers; }

    protected:
        explicit MouseButtonEvent(const Mouse::Button _button, const u16 _mods)
            : m_button(_button), m_modifiers(_mods) {}

        Mouse::Button m_button;
        u16 m_modifiers;
    };

    /// @brief Event emitted when a mouse button is pressed.
    class MouseButtonPressedEvent final : public MouseButtonEvent
    {
    public:
        EVENT_TYPE(MouseButtonPressed)

        explicit MouseButtonPressedEvent(const Mouse::Button _button, const u16 _mods)
            : MouseButtonEvent(_button, _mods) {}

        std::string ToString() const override
        {
            std::string ret("MouseButtonPressedEvent: ");
            ret += std::to_string(m_button);
            return ret;
        }
    };

    /// @brief Event emitted when a mouse button is released.
    class MouseButtonReleasedEvent final : public MouseButtonEvent
    {
    public:
        EVENT_TYPE(MouseButtonReleased)

        explicit MouseButtonReleasedEvent(const Mouse::Button _button, const u16 _mods)
            : MouseButtonEvent(_button, _mods) {}

        std::string ToString() const override
        {
            std::string ret("MouseButtonReleasedEvent: ");
            ret += std::to_string(m_button);
            return ret;
        }
    };

    /// @brief Abstract class for mouse cursor events.
    class MouseCursorEvent : public Event
    {
    public:
        EVENT_FLAGS(AVA_EVENT_INPUT | AVA_EVENT_MOUSE | AVA_EVENT_MOUSE_CURSOR)

    protected:
        explicit MouseCursorEvent() = default;
    };

    /// @brief Event emitted when the mouse cursors enters the window frame.
    class MouseEnteredFrameEvent final : public MouseCursorEvent
    {
    public:
        EVENT_TYPE(MouseEntered)
        MouseEnteredFrameEvent() = default;
    };

    /// @brief Event emitted when the mouse cursor exits the window frame.
    class MouseExitedFrameEvent final : public MouseCursorEvent
    {
    public:
        EVENT_TYPE(MouseExited)
        MouseExitedFrameEvent() = default;
    };

    /// @brief Event emitted when the mouse cursor is moved.
    class MouseMovedEvent final : public MouseCursorEvent
    {
    public:
        EVENT_TYPE(MouseMoved)

        explicit MouseMovedEvent(const float _x, const float _y)
            : MouseCursorEvent(), m_mouseX(_x), m_mouseY(_y) {}

        float GetX() const { return m_mouseX; }
        float GetY() const { return m_mouseY; }

        std::string ToString() const override
        {
            std::string ret("MouseMovedEvent: ");
            ret += std::to_string(m_mouseX) + std::string(", ") + std::to_string(m_mouseY);
            return ret;
        }

    private:
        float m_mouseX;
        float m_mouseY;
    };

    /// @brief Event emitted when the mouse wheel is scrolled.
    class MouseScrolledEvent final : public Event
    {
    public:
        EVENT_TYPE(MouseScrolled)
        EVENT_FLAGS(AVA_EVENT_INPUT | AVA_EVENT_MOUSE | AVA_EVENT_MOUSE_WHEEL)

        explicit MouseScrolledEvent(const float _xOffset, const float _yOffset)
            : m_offsetX(_xOffset), m_offsetY(_yOffset) {}

        float GetXOffset() const { return m_offsetX; }
        float GetYOffset() const { return m_offsetY; }

        std::string ToString() const override
        {
            std::string ret("MouseScrolledEvent: ");
            ret += std::to_string(m_offsetX) + std::string(", ") + std::to_string(m_offsetY);
            return ret;
        }

    private:
        float m_offsetX;
        float m_offsetY;
    };

}