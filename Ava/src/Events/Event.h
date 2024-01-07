#pragma once
/// @file Event.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    enum class EventType {
        // window
        WindowResized,
        WindowMoved,
        WindowMinimized,
        WindowMaximized,
        WindowRestored,
        WindowFocused,
        WindowClosed,
        // file
        FileChanged,
        // keyboard
        KeyPressed,
        KeyReleased,
        KeyTyped,
        // mouse
        MouseButtonPressed,
        MouseButtonReleased,
        MouseMoved,
        MouseScrolled,
        MouseEntered,
        MouseExited,
        // gamepad
        GamepadButtonPressed,
        GamepadButtonReleased,
        GamepadJoystickMoved,
        GamepadTriggerMoved
    };

#define EVENT_TYPE(type) static EventType StaticType() { return EventType::type; }        \
                         EventType GetEventType() const override { return StaticType(); } \
                         const char* GetName() const override { return #type; }

    enum EventFlags {
        AVA_EVENT_NONE = 0,
        AVA_EVENT_WINDOW = AVA_BIT(0),
        AVA_EVENT_FILE = AVA_BIT(1),
        AVA_EVENT_INPUT = AVA_BIT(2),
        AVA_EVENT_KEYBOARD = AVA_BIT(3),
        AVA_EVENT_MOUSE = AVA_BIT(4),
        AVA_EVENT_MOUSE_BUTTON = AVA_BIT(5),
        AVA_EVENT_MOUSE_CURSOR = AVA_BIT(6),
        AVA_EVENT_MOUSE_WHEEL = AVA_BIT(7),
        AVA_EVENT_GAMEPAD = AVA_BIT(8),
        AVA_EVENT_GAMEPAD_BUTTON = AVA_BIT(9),
        AVA_EVENT_GAMEPAD_ANALOGIC = AVA_BIT(10),
    };

#define EVENT_FLAGS(flags) int GetEventFlags() const override { return flags; }

    /// @brief Abstract class representing an event managed by the application.
    class Event
    {
    public:
        virtual ~Event() = default;
        virtual EventType GetEventType() const = 0;
        virtual const char* GetName() const = 0;
        virtual int GetEventFlags() const = 0;
        virtual std::string ToString() const { return GetName(); }
        bool HasFlag(const EventFlags _flag) const { return GetEventFlags() & _flag; }

        bool m_handled = false;
    };

    using EventCallbackFunc = std::function<void(Event&)>;


    /// @brief Helper providing a tool to dispatch events.
    class EventDispatcher
    {
    public:
        explicit EventDispatcher(Event& _event) : m_event(_event) {}

        template <typename T, typename F>
        bool Dispatch(const F& _function)
        {
            if (m_event.GetEventType() == T::StaticType())
            {
                m_event.m_handled |= _function(static_cast<T&>(m_event));
                return true;
            }
            return false;
        }

    private:
        Event& m_event;
    };

}
