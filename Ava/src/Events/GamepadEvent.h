#pragma once
/// @file GamepadEvent.h
/// @brief

#include <Events/Event.h>
#include <Inputs/GamepadCodes.h>

namespace Ava {

    /// @brief Abstract class for gamepad button events.
    class GamepadButtonEvent : public Event
    {
    public:
        EVENT_FLAGS(AVA_EVENT_INPUT | AVA_EVENT_GAMEPAD | AVA_EVENT_GAMEPAD_BUTTON)
        Gamepad::Button GetGamepadButton() const { return m_button; }

    protected:
        explicit GamepadButtonEvent(const Gamepad::Button _button) : m_button(_button) {}
        Gamepad::Button m_button;
    };

    /// @brief Event emitted when a gamepad button is pressed.
    class GamepadButtonPressedEvent final : public GamepadButtonEvent
    {
    public:
        EVENT_TYPE(GamepadButtonPressed)
        explicit GamepadButtonPressedEvent(const Gamepad::Button _button) : GamepadButtonEvent(_button) {}

        std::string ToString() const override
        {
            std::string ret("GamepadButtonPressedEvent: ");
            ret += std::to_string(m_button);
            return ret;
        }
    };

    /// @brief Event emitted when a gamepad button is released.
    class GamepadButtonReleasedEvent final : public GamepadButtonEvent
    {
    public:
        EVENT_TYPE(GamepadButtonReleased)
        explicit GamepadButtonReleasedEvent(const Gamepad::Button _button) : GamepadButtonEvent(_button) {}

        std::string ToString() const override
        {
            std::string ret("GamepadButtonReleasedEvent: ");
            ret += std::to_string(m_button);
            return ret;
        }
    };

    /// @brief Abstract class for gamepad analogic events.
    class GamepadAnalogicEvent : public Event
    {
    public:
        EVENT_FLAGS(AVA_EVENT_INPUT | AVA_EVENT_GAMEPAD | AVA_EVENT_GAMEPAD_ANALOGIC)
        Gamepad::Side GetSide() const { return m_side; }

    protected:
        explicit GamepadAnalogicEvent(const Gamepad::Side _side) : m_side(_side) {}
        Gamepad::Side m_side;
    };

    /// @brief Event emitted when a gamepad joystick is moved.
    class GamepadJoystickMovedEvent final : public GamepadAnalogicEvent
    {
    public:
        EVENT_TYPE(GamepadJoystickMoved)

        explicit GamepadJoystickMovedEvent(const Gamepad::Side _side, const float _x, const float _y)
            : GamepadAnalogicEvent(_side), m_joystickX(_x), m_joystickY(_y) {}

        float GetX() const { return m_joystickX; }
        float GetY() const { return m_joystickY; }

        std::string ToString() const override
        {
            std::string ret("GamepadJoystickMovedEvent: ");
            ret += std::string(m_side == Gamepad::SideLeft ? "[Left]" : "[Right]");
            ret += std::to_string(m_joystickX) + std::string(", ") + std::to_string(m_joystickY);
            return ret;
        }

    private:
        float m_joystickX;
        float m_joystickY;
    };

    /// @brief Event emitted when a gamepad trigger is moved.
    class GamepadTriggerMovedEvent final : public GamepadAnalogicEvent
    {
    public:
        EVENT_TYPE(GamepadTriggerMoved)

        explicit GamepadTriggerMovedEvent(const Gamepad::Side _side, const float _pos)
            : GamepadAnalogicEvent(_side), m_position(_pos) {}

        float GetPosition() const { return m_position; }

        std::string ToString() const override
        {
            std::string ret("GamepadTriggerMovedEvent: ");
            ret += std::string(m_side == Gamepad::SideLeft ? "[Left] " : "[Right] ");
            ret += std::to_string(m_position);
            return ret;
        }

    private:
        float m_position;
    };

}
