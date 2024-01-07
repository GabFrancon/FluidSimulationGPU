#include <avapch.h>
#include "ImGuiTools.h"

#include <Debug/Assert.h>

namespace Ava {

    // --- ImGui vector operators (copy pasted from imgui_internal.h) ---------------

    static ImVec2 operator*(const ImVec2& _lhs, const float _rhs)    { return ImVec2(_lhs.x * _rhs, _lhs.y * _rhs); }
    static ImVec2 operator/(const ImVec2& _lhs, const float _rhs)    { return ImVec2(_lhs.x / _rhs, _lhs.y / _rhs); }
    static ImVec2 operator+(const ImVec2& _lhs, const ImVec2& _rhs)  { return ImVec2(_lhs.x + _rhs.x, _lhs.y + _rhs.y); }
    static ImVec2 operator-(const ImVec2& _lhs, const ImVec2& _rhs)  { return ImVec2(_lhs.x - _rhs.x, _lhs.y - _rhs.y); }
    static ImVec2 operator*(const ImVec2& _lhs, const ImVec2& _rhs)  { return ImVec2(_lhs.x * _rhs.x, _lhs.y * _rhs.y); }
    static ImVec2 operator/(const ImVec2& _lhs, const ImVec2& _rhs)  { return ImVec2(_lhs.x / _rhs.x, _lhs.y / _rhs.y); }
    static ImVec2& operator*=(ImVec2& _lhs, const float _rhs)        { _lhs.x *= _rhs; _lhs.y *= _rhs; return _lhs; }
    static ImVec2& operator/=(ImVec2& _lhs, const float _rhs)        { _lhs.x /= _rhs; _lhs.y /= _rhs; return _lhs; }
    static ImVec2& operator+=(ImVec2& _lhs, const ImVec2& _rhs)      { _lhs.x += _rhs.x; _lhs.y += _rhs.y; return _lhs; }
    static ImVec2& operator-=(ImVec2& _lhs, const ImVec2& _rhs)      { _lhs.x -= _rhs.x; _lhs.y -= _rhs.y; return _lhs; }
    static ImVec2& operator*=(ImVec2& _lhs, const ImVec2& _rhs)      { _lhs.x *= _rhs.x; _lhs.y *= _rhs.y; return _lhs; }
    static ImVec2& operator/=(ImVec2& _lhs, const ImVec2& _rhs)      { _lhs.x /= _rhs.x; _lhs.y /= _rhs.y; return _lhs; }
    static ImVec4 operator+(const ImVec4& _lhs, const ImVec4& _rhs)  { return ImVec4(_lhs.x + _rhs.x, _lhs.y + _rhs.y, _lhs.z + _rhs.z, _lhs.w + _rhs.w); }
    static ImVec4 operator-(const ImVec4& _lhs, const ImVec4& _rhs)  { return ImVec4(_lhs.x - _rhs.x, _lhs.y - _rhs.y, _lhs.z - _rhs.z, _lhs.w - _rhs.w); }
    static ImVec4 operator*(const ImVec4& _lhs, const ImVec4& _rhs)  { return ImVec4(_lhs.x * _rhs.x, _lhs.y * _rhs.y, _lhs.z * _rhs.z, _lhs.w * _rhs.w); }


    // --- ImGui frame scope --------------------------------------------------------

    bool ImGuiTools::WithinFrameScope()
    {
        const ImGuiContext* imguiCtx = ImGui::GetCurrentContext();
        return imguiCtx && imguiCtx->WithinFrameScope;
    }


    // --- Window resize border -----------------------------------------------------

    bool ImGuiTools::IsWindowResizeBorderHovered()
    {
        constexpr float windowHoverPadding = 4.f;
        const ImGuiWindow* window = ImGui::GetCurrentWindow();
        const float fontSize = window->CalcFontSize();
        const ImRect rect = window->Rect();

        const float gripDrawSize = IM_FLOOR(ImMax(fontSize * 1.35f, window->WindowRounding + 1.0f + fontSize * 0.2f));
        const float gripHoverInnerSize = IM_FLOOR(gripDrawSize * 0.75f);
        const ImVec2 mousePos = ImGui::GetMousePos();

        const ImRect leftBorder(rect.Min.x - windowHoverPadding, rect.Min.y + gripHoverInnerSize, rect.Min.x + windowHoverPadding, rect.Max.y - gripHoverInnerSize);
        if (leftBorder.Contains(mousePos))
        {
            return true;
        }

        const ImRect rightBorder(rect.Max.x - windowHoverPadding, rect.Min.y + gripHoverInnerSize, rect.Max.x + windowHoverPadding, rect.Max.y - gripHoverInnerSize);
        if (rightBorder.Contains(mousePos))
        {
            return true;
        }

        const ImRect upBorder(rect.Min.x + gripHoverInnerSize, rect.Min.y - windowHoverPadding, rect.Max.x - gripHoverInnerSize, rect.Min.y + windowHoverPadding);
        if (upBorder.Contains(mousePos))
        {
            return true;
        }

        const ImRect downBorder(rect.Min.x + gripHoverInnerSize, rect.Max.y - windowHoverPadding, rect.Max.x - gripHoverInnerSize, rect.Max.y + windowHoverPadding);
        if (downBorder.Contains(mousePos))
        {
            return true;
        }

        return false;
    }

    bool ImGuiTools::IsWindowResizeBorderActive()
    {
        ImGuiWindow* window = ImGui::GetCurrentWindow();
        const ImGuiID activeId = ImGui::GetActiveID();

        if (!activeId)
        {
            return false;
        }

        return activeId == ImGui::GetWindowResizeBorderID(window, ImGuiDir_Left)
            || activeId == ImGui::GetWindowResizeBorderID(window, ImGuiDir_Right)
            || activeId == ImGui::GetWindowResizeBorderID(window, ImGuiDir_Down)
            || activeId == ImGui::GetWindowResizeBorderID(window, ImGuiDir_Up);
    }


    // --- Window scroll bar --------------------------------------------------------

    bool ImGuiTools::IsWindowScrollbarHovered()
    {
        ImGuiWindow* window = ImGui::GetCurrentWindow();
        const ImVec2 mousePos = ImGui::GetMousePos();

        if (window->ScrollbarSizes[ImGuiAxis_X] > 0.f)
        {
            const ImRect verticalRect = ImGui::GetWindowScrollbarRect(window, ImGuiAxis_Y);
            if (verticalRect.Contains(mousePos))
            {
                return true;
            }
        }

        if (window->ScrollbarSizes[ImGuiAxis_Y] > 0.f)
        {
            const ImRect horizontalRect = ImGui::GetWindowScrollbarRect(window, ImGuiAxis_X);
            if (horizontalRect.Contains(mousePos))
            {
                return true;
            }
        }

        return false;
    }

    bool ImGuiTools::IsWindowScrollbarActive()
    {
        ImGuiWindow* window = ImGui::GetCurrentWindow();
        const ImGuiID activeId = ImGui::GetActiveID();

        if (!activeId)
        {
            return false;
        }

        return activeId == ImGui::GetWindowScrollbarID(window, ImGuiAxis_X)
            || activeId == ImGui::GetWindowScrollbarID(window, ImGuiAxis_Y);
    }


    // --- Text fonts ---------------------------------------------------------------

    static ImFont* s_fonts[UI::FontCount];

    void ImGuiTools::SetFont(const UI::FontType _fontType, ImFont* _font)
    {
        if (!AVA_VERIFY(_fontType < UI::FontCount, "[UI] invalid font type provided."))
        {
            return;
        }

        s_fonts[_fontType < UI::FontCount ? _fontType : 0] = _font;
    }

    ImFont* ImGuiTools::GetFont(const UI::FontType _fontType)
    {
        if (!AVA_VERIFY(_fontType < UI::FontCount, "[UI] invalid font type provided."))
        {
            return nullptr;
        }

        return s_fonts[_fontType < UI::FontCount ? _fontType : 0];
    }


    // --- Ava -> ImGui conversions -------------------------------------------------

    ImGuiKey ImGuiTools::AvaToImGui(const Keyboard::Key _key)
    {
        switch (_key)
        {
            case Keyboard::Tab: return ImGuiKey_Tab;
            case Keyboard::Left: return ImGuiKey_LeftArrow;
            case Keyboard::Right: return ImGuiKey_RightArrow;
            case Keyboard::Up: return ImGuiKey_UpArrow;
            case Keyboard::Down: return ImGuiKey_DownArrow;
            case Keyboard::PageUp: return ImGuiKey_PageUp;
            case Keyboard::PageDown: return ImGuiKey_PageDown;
            case Keyboard::Home: return ImGuiKey_Home;
            case Keyboard::End: return ImGuiKey_End;
            case Keyboard::Insert: return ImGuiKey_Insert;
            case Keyboard::Delete: return ImGuiKey_Delete;
            case Keyboard::Backspace: return ImGuiKey_Backspace;
            case Keyboard::Space: return ImGuiKey_Space;
            case Keyboard::Enter: return ImGuiKey_Enter;
            case Keyboard::Escape: return ImGuiKey_Escape;
            case Keyboard::Apostrophe: return ImGuiKey_Apostrophe;
            case Keyboard::Comma: return ImGuiKey_Comma;
            case Keyboard::Minus: return ImGuiKey_Minus;
            case Keyboard::Period: return ImGuiKey_Period;
            case Keyboard::Slash: return ImGuiKey_Slash;
            case Keyboard::Semicolon: return ImGuiKey_Semicolon;
            case Keyboard::Equal: return ImGuiKey_Equal;
            case Keyboard::LeftBracket: return ImGuiKey_LeftBracket;
            case Keyboard::BackSlash: return ImGuiKey_Backslash;
            case Keyboard::RightBracket: return ImGuiKey_RightBracket;
            case Keyboard::GraveAccent: return ImGuiKey_GraveAccent;
            case Keyboard::CapsLock: return ImGuiKey_CapsLock;
            case Keyboard::ScrollLock: return ImGuiKey_ScrollLock;
            case Keyboard::NumLock: return ImGuiKey_NumLock;
            case Keyboard::PrintScreen: return ImGuiKey_PrintScreen;
            case Keyboard::Pause: return ImGuiKey_Pause;
            case Keyboard::KP0: return ImGuiKey_Keypad0;
            case Keyboard::KP1: return ImGuiKey_Keypad1;
            case Keyboard::KP2: return ImGuiKey_Keypad2;
            case Keyboard::KP3: return ImGuiKey_Keypad3;
            case Keyboard::KP4: return ImGuiKey_Keypad4;
            case Keyboard::KP5: return ImGuiKey_Keypad5;
            case Keyboard::KP6: return ImGuiKey_Keypad6;
            case Keyboard::KP7: return ImGuiKey_Keypad7;
            case Keyboard::KP8: return ImGuiKey_Keypad8;
            case Keyboard::KP9: return ImGuiKey_Keypad9;
            case Keyboard::KPDecimal: return ImGuiKey_KeypadDecimal;
            case Keyboard::KPDivide: return ImGuiKey_KeypadDivide;
            case Keyboard::KPMultiply: return ImGuiKey_KeypadMultiply;
            case Keyboard::KPSubtract: return ImGuiKey_KeypadSubtract;
            case Keyboard::KPAdd: return ImGuiKey_KeypadAdd;
            case Keyboard::KPEnter: return ImGuiKey_KeypadEnter;
            case Keyboard::KPEqual: return ImGuiKey_KeypadEqual;
            case Keyboard::LeftShift: return ImGuiKey_LeftShift;
            case Keyboard::LeftControl: return ImGuiKey_LeftCtrl;
            case Keyboard::LeftAlt: return ImGuiKey_LeftAlt;
            case Keyboard::LeftSuper: return ImGuiKey_LeftSuper;
            case Keyboard::RightShift: return ImGuiKey_RightShift;
            case Keyboard::RightControl: return ImGuiKey_RightCtrl;
            case Keyboard::RightAlt: return ImGuiKey_RightAlt;
            case Keyboard::RightSuper: return ImGuiKey_RightSuper;
            case Keyboard::Menu: return ImGuiKey_Menu;
            case Keyboard::D0: return ImGuiKey_0;
            case Keyboard::D1: return ImGuiKey_1;
            case Keyboard::D2: return ImGuiKey_2;
            case Keyboard::D3: return ImGuiKey_3;
            case Keyboard::D4: return ImGuiKey_4;
            case Keyboard::D5: return ImGuiKey_5;
            case Keyboard::D6: return ImGuiKey_6;
            case Keyboard::D7: return ImGuiKey_7;
            case Keyboard::D8: return ImGuiKey_8;
            case Keyboard::D9: return ImGuiKey_9;
            case Keyboard::A: return ImGuiKey_A;
            case Keyboard::B: return ImGuiKey_B;
            case Keyboard::C: return ImGuiKey_C;
            case Keyboard::D: return ImGuiKey_D;
            case Keyboard::E: return ImGuiKey_E;
            case Keyboard::F: return ImGuiKey_F;
            case Keyboard::G: return ImGuiKey_G;
            case Keyboard::H: return ImGuiKey_H;
            case Keyboard::I: return ImGuiKey_I;
            case Keyboard::J: return ImGuiKey_J;
            case Keyboard::K: return ImGuiKey_K;
            case Keyboard::L: return ImGuiKey_L;
            case Keyboard::M: return ImGuiKey_M;
            case Keyboard::N: return ImGuiKey_N;
            case Keyboard::O: return ImGuiKey_O;
            case Keyboard::P: return ImGuiKey_P;
            case Keyboard::Q: return ImGuiKey_Q;
            case Keyboard::R: return ImGuiKey_R;
            case Keyboard::S: return ImGuiKey_S;
            case Keyboard::T: return ImGuiKey_T;
            case Keyboard::U: return ImGuiKey_U;
            case Keyboard::V: return ImGuiKey_V;
            case Keyboard::W: return ImGuiKey_W;
            case Keyboard::X: return ImGuiKey_X;
            case Keyboard::Y: return ImGuiKey_Y;
            case Keyboard::Z: return ImGuiKey_Z;
            case Keyboard::F1: return ImGuiKey_F1;
            case Keyboard::F2: return ImGuiKey_F2;
            case Keyboard::F3: return ImGuiKey_F3;
            case Keyboard::F4: return ImGuiKey_F4;
            case Keyboard::F5: return ImGuiKey_F5;
            case Keyboard::F6: return ImGuiKey_F6;
            case Keyboard::F7: return ImGuiKey_F7;
            case Keyboard::F8: return ImGuiKey_F8;
            case Keyboard::F9: return ImGuiKey_F9;
            case Keyboard::F10: return ImGuiKey_F10;
            case Keyboard::F11: return ImGuiKey_F11;
            case Keyboard::F12: return ImGuiKey_F12;
            default: return ImGuiKey_None;
        }
    }

    ImGuiMouseButton ImGuiTools::AvaToImGui(const Mouse::Button _button)
    {
        switch (_button)
        {
            case Mouse::ButtonLeft: return ImGuiMouseButton_Left;
            case Mouse::ButtonRight: return ImGuiMouseButton_Right;
            case Mouse::ButtonMiddle: return ImGuiMouseButton_Middle;
            default: return ImGuiMouseButton_Left;
        }
    }

    ImGuiConfigFlags ImGuiTools::AvaToImGui(const UIConfigFlags _flag)
    {
        switch (_flag)
        {
            case AVA_UI_DOCKING: return ImGuiConfigFlags_DockingEnable;
            case AVA_UI_MULTI_VIEWPORTS: return ImGuiConfigFlags_ViewportsEnable;
            case AVA_UI_KEYBOARD_NAVIGATION: return ImGuiConfigFlags_NavEnableKeyboard;
            case AVA_UI_GAMEPAD_NAVIGATION: return ImGuiConfigFlags_NavEnableGamepad;
            case AVA_UI_NO_MOUSE_CURSOR_CHANGE: return ImGuiConfigFlags_NoMouseCursorChange;
            case AVA_UI_NONE: return ImGuiConfigFlags_None;
            default: return ImGuiConfigFlags_None;
        }
    }

    Mouse::CursorIcon ImGuiTools::ImGuiToAva(const ImGuiMouseCursor _icon)
    {
        switch (_icon)
        {
            case ImGuiMouseCursor_Arrow: return Mouse::IconArrow;
            case ImGuiMouseCursor_TextInput: return Mouse::IconTextInput;
            case ImGuiMouseCursor_ResizeAll: return Mouse::IconResizeAll;
            case ImGuiMouseCursor_ResizeNS: return Mouse::IconResizeVertical;
            case ImGuiMouseCursor_ResizeEW: return Mouse::IconResizeHorizontal;
            case ImGuiMouseCursor_ResizeNWSE: return Mouse::IconResizeTopLeftBottomRight;
            case ImGuiMouseCursor_ResizeNESW: return Mouse::IconResizeTopRightBottomLeft;
            case ImGuiMouseCursor_Hand: return Mouse::IconHand;
            case ImGuiMouseCursor_NotAllowed: return Mouse::IconNotAllowed;
            default: return Mouse::IconArrow;
        }
    }

}
