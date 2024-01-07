#pragma once
/// @file GamepadCodes.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    namespace Gamepad
    {
        /// @brief Codes to identify which button is triggered on the gamepad.
        /// @note based on GLFW gamepad codes (see glfw3.h).
        enum Button : u16
        {
            Cross = 0,
            Circle = 1,
            Square = 2,
            Triangle = 3,

            L1 = 4,
            L2 = 15,
            L3 = 9,

            R1 = 5,
            R2 = 16,
            R3 = 10,

            Share = 6,
            Options = 7,
            Start = 8,

            ButtonUp = 11,
            ButtonRight = 12,
            ButtonDown = 13,
            ButtonLeft = 14,

            // reserved
            ButtonCount = 17,
        };

        /// @brief Codes to identity to which side of the gamepad a trigger or  a joystick belongs.
        enum Side : u8
        {
            SideLeft = 0,
            SideRight = 1,

            // reserved
            SideCount = 2
        };
    }

}