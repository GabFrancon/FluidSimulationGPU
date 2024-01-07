#pragma once
/// @file WindowEvent.h
/// @brief

#include <Events/Event.h>

namespace Ava {

    /// @brief Event emitted when the window is resized.
    class WindowResizedEvent final : public Event
    {
    public:
        EVENT_TYPE(WindowResized)
        EVENT_FLAGS(AVA_EVENT_WINDOW)
        WindowResizedEvent(const u32 _width, const u32 _height) : m_width(_width), m_height(_height) {}

        u32 GetWidth() const { return m_width; }
        u32 GetHeight() const { return m_height; }

        std::string ToString() const override
        {
            std::string ret("WindowResizedEvent: ");
            ret += std::to_string(m_width) + ", " + std::to_string(m_height);
            return ret;
        }

    private:
        u32 m_width;
        u32 m_height;
    };

    /// @brief Event emitted when the window is moved.
    class WindowMovedEvent final : public Event
    {
    public:
        EVENT_TYPE(WindowMoved)
        EVENT_FLAGS(AVA_EVENT_WINDOW)
        WindowMovedEvent(const u32 _xPos, const u32 _yPos) : m_originX(_xPos), m_originY(_yPos) {}

        u32 GetX() const { return m_originX; }
        u32 GetY() const { return m_originY; }

        std::string ToString() const override
        {
            std::string ret("WindowMovedEvent: ");
            ret += std::to_string(m_originX) + ", " + std::to_string(m_originY);
            return ret;
        }

    private:
        u32 m_originX;
        u32 m_originY;
    };

    /// @brief Event emitted when the window is minimized.
    class WindowMinimizedEvent final : public Event
    {
    public:
        EVENT_TYPE(WindowMinimized)
        EVENT_FLAGS(AVA_EVENT_WINDOW)
        WindowMinimizedEvent() = default;
    };

    /// @brief Event emitted when the window is maximized.
    class WindowMaximizedEvent final : public Event
    {
    public:
        EVENT_TYPE(WindowMaximized)
        EVENT_FLAGS(AVA_EVENT_WINDOW)
        WindowMaximizedEvent() = default;
    };

    /// @brief Event emitted when the window is restored.
    class WindowRestoredEvent final : public Event
    {
    public:
        EVENT_TYPE(WindowRestored)
        EVENT_FLAGS(AVA_EVENT_WINDOW)
        WindowRestoredEvent() = default;
    };

    /// @brief Event emitted when the window if focused.
    class WindowFocusedEvent final : public Event
    {
    public:
        EVENT_TYPE(WindowFocused)
        EVENT_FLAGS(AVA_EVENT_WINDOW)
        WindowFocusedEvent() = default;
    };

    /// @brief Event emitted when the window is closed.
    class WindowClosedEvent final : public Event
    {
    public:
        EVENT_TYPE(WindowClosed)
        EVENT_FLAGS(AVA_EVENT_WINDOW)
        WindowClosedEvent() = default;
    };

}
