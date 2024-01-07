#pragma once
/// @file MouseCodes.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    namespace Mouse
    {
        /// @brief Codes to identify which button is triggered on the mouse.
        /// @note based on GLFW mouse codes (see glfw3.h).
        enum Button : u16
        {
            ButtonLeft = 0,
            ButtonRight = 1,
            ButtonMiddle = 2,

            // reserved
            ButtonCount = 5
        };

        /// @brief Codes to identify which cursor icon is displayed on the screen.
        enum CursorIcon : u8
        {
            IconArrow,
            IconTextInput,
            IconResizeAll,
            IconResizeVertical,
            IconResizeHorizontal,
            IconResizeTopLeftBottomRight,
            IconResizeTopRightBottomLeft,
            IconHand,
            IconNotAllowed,

            // reserved
            IconCount
        };

        enum CursorMode : u8
        {
            ModeNormal,
            ModeHidden,
            ModeDisabled,

            // reserved
            ModeCount
        };

    }


}
