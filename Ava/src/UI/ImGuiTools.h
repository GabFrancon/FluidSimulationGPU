#pragma once
/// @file ImGuiTools.h
/// @brief

#include <UI/UICommon.h>
#include <Inputs/KeyCodes.h>
#include <Inputs/MouseCodes.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <ImGui/imgui.h>
#include <ImGui/ImGuizmo.h>
#include <ImGui/imgui_internal.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    namespace ImGuiTools
    {
        // ImGui frame scope
        bool WithinFrameScope();

        // Window resize border
        bool IsWindowResizeBorderHovered();
        bool IsWindowResizeBorderActive();

        // Window scroll bar
        bool IsWindowScrollbarHovered();
        bool IsWindowScrollbarActive();

        // Text fonts
        void SetFont(UI::FontType _fontType, ImFont* _font);
        ImFont* GetFont(UI::FontType _fontType);

        // Ava -> ImGui conversions
        ImGuiKey AvaToImGui(Keyboard::Key _key);
        ImGuiMouseButton AvaToImGui(Mouse::Button _button);
        ImGuiConfigFlags AvaToImGui(UIConfigFlags _flag);

        // ImGui -> Ava conversions
        Mouse::CursorIcon ImGuiToAva(ImGuiMouseCursor _icon);
    }

}

// useful macros
#define IMGUI_FONT_REGULAR Ava::ImGuiTools::GetFont(Ava::UI::FontRegular)
#define IMGUI_FONT_BOLD    Ava::ImGuiTools::GetFont(Ava::UI::FontBold)