#pragma once
/// @file KeyCodes.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    namespace Keyboard
    {
        /// @brief Codes to identify which key is triggered on the keyboard.
        /// @note based on GLFW key codes (see glfw3.h).
        enum Key : u16
        {
            Space = 32,         /*   */
            Apostrophe = 39,    /* ' */
            Comma = 44,         /* , */
            Minus = 45,         /* - */
            Period = 46,        /* . */
            Slash = 47,         /* / */
            Semicolon = 59,     /* ; */
            Equal = 61,         /* = */

            D0 = 48,            /* 0 */
            D1 = 49,            /* 1 */
            D2 = 50,            /* 2 */
            D3 = 51,            /* 3 */
            D4 = 52,            /* 4 */
            D5 = 53,            /* 5 */
            D6 = 54,            /* 6 */
            D7 = 55,            /* 7 */
            D8 = 56,            /* 8 */
            D9 = 57,            /* 9 */

            A = 65,             /* A */
            B = 66,             /* B */
            C = 67,             /* C */
            D = 68,             /* D */
            E = 69,             /* E */
            F = 70,             /* F */
            G = 71,             /* G */
            H = 72,             /* H */
            I = 73,             /* I */
            J = 74,             /* J */
            K = 75,             /* K */
            L = 76,             /* L */
            M = 77,             /* M */
            N = 78,             /* N */
            O = 79,             /* O */
            P = 80,             /* P */
            Q = 81,             /* Q */
            R = 82,             /* R */
            S = 83,             /* S */
            T = 84,             /* T */
            U = 85,             /* U */
            V = 86,             /* V */
            W = 87,             /* W */
            X = 88,             /* X */
            Y = 89,             /* Y */
            Z = 90,             /* Z */

            LeftBracket = 91,   /* [ */
            BackSlash = 92,     /* \ */
            RightBracket = 93,  /* ] */
            GraveAccent = 96,   /* ` */

            /* Non-US keys */
            World1 = 161,
            World2 = 162,

            /* Function keys */
            Escape = 256,
            Enter = 257,
            Tab = 258,
            Backspace = 259,
            Insert = 260,
            Delete = 261,
            Right = 262,
            Left = 263,
            Down = 264,
            Up = 265,
            PageUp = 266,
            PageDown = 267,
            Home = 268,
            End = 269,
            CapsLock = 280,
            ScrollLock = 281,
            NumLock = 282,
            PrintScreen = 283,
            Pause = 284,

            /* Generic functions keys */
            F1 = 290,
            F2 = 291,
            F3 = 292,
            F4 = 293,
            F5 = 294,
            F6 = 295,
            F7 = 296,
            F8 = 297,
            F9 = 298,
            F10 = 299,
            F11 = 300,
            F12 = 301,
            F13 = 302,
            F14 = 303,
            F15 = 304,
            F16 = 305,
            F17 = 306,
            F18 = 307,
            F19 = 308,
            F20 = 309,
            F21 = 310,
            F22 = 311,
            F23 = 312,
            F24 = 313,
            F25 = 314,

            /* Keypad */
            KP0 = 320,
            KP1 = 321,
            KP2 = 322,
            KP3 = 323,
            KP4 = 324,
            KP5 = 325,
            KP6 = 326,
            KP7 = 327,
            KP8 = 328,
            KP9 = 329,
            KPDecimal = 330,
            KPDivide = 331,
            KPMultiply = 332,
            KPSubtract = 333,
            KPAdd = 334,
            KPEnter = 335,
            KPEqual = 336,
            
            /* Modifier keys */
            LeftShift = 340,
            LeftControl = 341,
            LeftAlt = 342,
            LeftSuper = 343,
            RightShift = 344,
            RightControl = 345,
            RightAlt = 346,
            RightSuper = 347,
            Menu = 348,

            /* Reserved */
            FirstKey = 32,
            KeyCount = 349
        };

        /// @brief Codes to identify which modifier is triggered on the keyboard.
        /// @note based on GLFW modifier flags (see glfw3.h).
        enum Modifier : u16
        {
            AVA_MODIFIER_NONE = 0,
            AVA_MODIFIER_SHIFT = AVA_BIT(0),
            AVA_MODIFIER_CTRL = AVA_BIT(1),
            AVA_MODIFIER_ALT = AVA_BIT(2),
            AVA_MODIFIER_SUPER = AVA_BIT(3),
            AVA_MODIFIER_CAPS_LOCK = AVA_BIT(4),
            AVA_MODIFIER_NUM_LOCK = AVA_BIT(5)
        };
    }

}
