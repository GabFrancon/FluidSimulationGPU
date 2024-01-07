#pragma once
/// @file KeyEvent.h
/// @brief

#include <Events/Event.h>
#include <Inputs/KeyCodes.h>

namespace Ava {

    /// @brief Abstract class for keyboard events.
    class KeyEvent : public Event
    {
    public:
        EVENT_FLAGS(AVA_EVENT_INPUT | AVA_EVENT_KEYBOARD)
        Keyboard::Key GetKeyCode() const { return m_key; }
        u16 GetModifiers() const { return m_modifiers; }

    protected:
        explicit KeyEvent(const Keyboard::Key _key, const u16 _mods)
            : m_key(_key), m_modifiers(_mods) {}

        Keyboard::Key m_key;
        u16 m_modifiers;
    };

    /// @brief Event emitted when a key is pressed.
    class KeyPressedEvent final : public KeyEvent
    {
    public:
        EVENT_TYPE(KeyPressed)
        explicit KeyPressedEvent(const Keyboard::Key _key, const u16 _mods) : KeyEvent(_key, _mods) {}

        std::string ToString() const override
        {
            std::string ret("KeyPressedEvent: ");
            ret += std::to_string(m_key);
            return ret;
        }
    };

    /// @brief Event emitted when a key is released.
    class KeyReleasedEvent final: public KeyEvent
    {
    public:
        EVENT_TYPE(KeyReleased)
        explicit KeyReleasedEvent(const Keyboard::Key _key, const u16 _mods) : KeyEvent(_key, _mods) {}

        std::string ToString() const override
        {
            std::string ret("KeyReleasedEvent: ");
            ret += std::to_string(m_key);
            return ret;
        }
    };

    /// @brief Event emitted when a key is typed.
    class KeyTypedEvent final : public Event
    {
    public:
        EVENT_TYPE(KeyTyped)
        EVENT_FLAGS(AVA_EVENT_INPUT | AVA_EVENT_KEYBOARD)
        explicit KeyTypedEvent(const unsigned _char) : m_char(_char) {}

        unsigned char GetChar() const { return m_char; }

        std::string ToString() const override
        {
            std::string ret("KeyTypedEvent: ");
            ret += std::to_string(m_char);
            return ret;
        }

    private:
        unsigned char m_char;
    };

}

