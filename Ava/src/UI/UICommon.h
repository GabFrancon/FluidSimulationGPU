#pragma once
/// @file UICommon.h
/// @brief

#include <Core/Base.h>
#include <UI/IconsFontAwesome6.h>

namespace Ava {

    class Texture;
    class ShaderProgram;

    struct ShadedTexture
    {
        Texture* texture = nullptr;
        ShaderProgram* shader = nullptr;
    };

    /// @brief UI config flags
    enum UIConfigFlags
    {
        AVA_UI_NONE = 0,
        AVA_UI_DOCKING = AVA_BIT(0),
        AVA_UI_MULTI_VIEWPORTS = AVA_BIT(1),
        AVA_UI_KEYBOARD_NAVIGATION = AVA_BIT(2),
        AVA_UI_GAMEPAD_NAVIGATION = AVA_BIT(3),
        AVA_UI_NO_MOUSE_CURSOR_CHANGE = AVA_BIT(4)
    };

    namespace UI
    {
        /// @brief Theme presets
        enum ThemePreset
        {
            ThemeClassic,
            ThemeLight,
            ThemeDark,

            ThemeCount
        };

        /// @brief Font types
        enum FontType
        {
            FontRegular,
            FontBold,

            FontCount
        };

    }

}