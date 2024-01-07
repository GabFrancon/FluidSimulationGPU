#pragma once
/// @file Color.h
/// @brief File defining the Color primitive.

#include <Math/Types.h>

namespace Ava {

    /// @brief Represents a color with red, green, blue, and alpha components.
    /// @details Provides methods for color manipulation and conversions.
    struct Color
    {
        float r = 0.f;
        float g = 0.f;
        float b = 0.f;
        float a = 1.f;

        constexpr Color() = default;

        constexpr Color(const float _r, const float _g, const float _b, const float _a = 1.f)
            : r(_r), g(_g), b(_b), a(_a) {}

        constexpr Color(const int _r, const int _g, const int _b, const int _a = 255)
            : r(_r / 255.f), g(_g / 255.f), b(_b / 255.f), a(_a / 255.f) {}

        Color ToSRGB() const;
        Color ToLinear() const;

        operator u32() const;
        operator Vec3f() const;
        operator Vec4f() const;

        Color operator*(float _factor) const;
        Color operator/(float _factor) const;
        // Color operator+(const Color& _other) const;
        // Color operator-(const Color& _other) const;

        bool operator==(const Color& _other) const;
        bool operator!=(const Color& _other) const;
        float* operator&();

        // predefined colors
        static Color White;
        static Color Red;
        static Color Green;
        static Color Blue;
        static Color Yellow;
        static Color Magenta;
        static Color Cyan;
        static Color Black;
        static Color Grey;
    };

}