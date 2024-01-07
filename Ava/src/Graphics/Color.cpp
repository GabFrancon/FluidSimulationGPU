#include <avapch.h>
#include "Color.h"

#include <Math/Math.h>

namespace Ava {

    // --------- Color ---------------------------------------------

    Color Color::White   = { 1.0f, 1.0f, 1.0f, 1.0f };
    Color Color::Red     = { 1.0f, 0.0f, 0.0f, 1.0f };
    Color Color::Green   = { 0.0f, 1.0f, 0.0f, 1.0f };
    Color Color::Blue    = { 0.0f, 0.0f, 1.0f, 1.0f };
    Color Color::Yellow  = { 1.0f, 1.0f, 0.0f, 1.0f };
    Color Color::Magenta = { 1.0f, 0.0f, 1.0f, 1.0f };
    Color Color::Cyan    = { 0.0f, 1.0f, 1.0f, 1.0f };
    Color Color::Black   = { 0.0f, 0.0f, 0.0f, 1.0f };
    Color Color::Grey    = { 0.5f, 0.5f, 0.5f, 1.0f };

    Color Color::ToSRGB() const
    {
        static constexpr float kInvGamma = 2.2f;

        Color srgb;
        srgb.r = pow(r, kInvGamma);
        srgb.g = pow(g, kInvGamma);
        srgb.b = pow(b, kInvGamma);
        srgb.a = a;

        return srgb;
    }

    Color Color::ToLinear() const
    {
        static constexpr float kGamma = 2.2f;

        Color linear;
        linear.r = pow(r, kGamma);
        linear.g = pow(g, kGamma);
        linear.b = pow(b, kGamma);
        linear.a = a;

        return linear;
    }

    Color::operator u32() const
    {
        const u32 value = 
              (u32)(Math::saturate(r) * 255.f)
            | (u32)(Math::saturate(g) * 255.f) << 8
            | (u32)(Math::saturate(b) * 255.f) << 16
            | (u32)(Math::saturate(a) * 255.f) << 24;
    
        return value;
    }
    
    Color::operator Vec3f() const
    {
        return { r, g, b };
    }
    
    Color::operator Vec4f() const
    {
        return { r, g, b, a };
    }

    Color Color::operator*(const float _factor) const
    {
        Color res;
        res.r = r * _factor;
        res.g = g * _factor;
        res.b = b * _factor;
        res.a = a * _factor;
        return res;
    }

    Color Color::operator/(const float _factor) const
    {
        Color res;
        res.r = r / _factor;
        res.g = g / _factor;
        res.b = b / _factor;
        res.a = a / _factor;
        return res;
    }

    // Color Color::operator+(const Color& _other) const
    // {
    //     Color res;
    //     res.r = r + _other.r;
    //     res.g = g + _other.g;
    //     res.b = b + _other.b;
    //     res.a = a + _other.a;
    //
    //     return res;
    // }
    //
    // Color Color::operator-(const Color& _other) const
    // {
    //     Color res;
    //     res.r = r - _other.r;
    //     res.g = g - _other.g;
    //     res.b = b - _other.b;
    //     res.a = a - _other.a;
    //
    //     return res;
    // }

    bool Color::operator==(const Color& _other) const
    {
        return r == _other.r && g == _other.g && b == _other.b && a == _other.a;
    }

    bool Color::operator!=(const Color& _other) const
    {
        return !(*this == _other);
    }

    float* Color::operator&()
    {
        return &r;
    }

}