#pragma once
/// @file BitUtils.h
/// @brief Set of useful bit manipulation helpers, including packing and bitfield operations.

#include <Math/Math.h>

namespace Ava {

    /// @brief Packs 32 bits float color value on 8 bits.
    inline u32 ColorToUNorm8(const float _r, const float _g, const float _b, const float _a)
    {
        const auto value = 
              (u32)(Math::clamp(_r, 0.f, 1.f) * 255.f)
            | (u32)(Math::clamp(_g, 0.f, 1.f) * 255.f) << 8
            | (u32)(Math::clamp(_b, 0.f, 1.f) * 255.f) << 16
            | (u32)(Math::clamp(_a, 0.f, 1.f) * 255.f) << 24;

        return value;
    }

    /// @brief Packs 32 bits int color value on 8 bits.
    inline u32 ColorToUNorm8(const int _r, const int _g, const int _b, const int _a)
    {

        const auto value = 
              (u32)Math::clamp(_r, 0, 255)
            | (u32)Math::clamp(_g, 0, 255) << 8
            | (u32)Math::clamp(_b, 0, 255) << 16
            | (u32)Math::clamp(_a, 0, 255) << 24;

        return value;
    }

    /// @brief Packs 32 bits unorm value on 8 bits.
    inline u8 Float32ToUNorm8(float _value)
    {
        _value = Math::clamp(_value, 0.f, 1.f);
        return static_cast<u8>(_value * std::numeric_limits<u8>::max());
    }

    /// @brief Unpacks 8 bits unorm value on 32 bits.
    inline float UNorm8ToFloat32(const u8 _value)
    {
        return static_cast<float>(_value) / std::numeric_limits<u8>::max();
    }

    /// @brief Packs 32 bits unorm value on 16 bits.
    inline u16 Float32ToUNorm16(float _value)
    {
        _value = Math::clamp(_value, 0.f, 1.f);
        return static_cast<u16>(_value * std::numeric_limits<u16>::max());
    }

    /// @brief Unpacks 16 bits unorm value on 32 bits.
    inline float UNorm16ToFloat32(const u16 _value)
    {
        return static_cast<float>(_value) / std::numeric_limits<u16>::max();
    }

    /// @brief Packs 32 bits snorm value on 8 bits.
    inline s8 Float32ToSNorm8(float _value)
    {
        _value = Math::clamp(_value, -1.f, 1.f);

        return _value < 0
          ? -static_cast<s8>(_value * std::numeric_limits<s8>::min())
          :  static_cast<s8>(_value * std::numeric_limits<s8>::max());
    }

    /// @brief Unpacks 8 bits snorm value on 32 bits.
    inline float SNorm8ToFloat32(const s8 _value)
    {
        return _value < 0
          ? -static_cast<float>(_value) / std::numeric_limits<s8>::min()
          :  static_cast<float>(_value) / std::numeric_limits<s8>::max();
    }

    /// @brief Packs 32 bits snorm value on 16 bits.
    inline s16 Float32ToSNorm16(float _value)
    {
        _value = Math::clamp(_value, -1.f, 1.f);

        return _value < 0
          ? -static_cast<s16>(_value * std::numeric_limits<s16>::min())
          :  static_cast<s16>(_value * std::numeric_limits<s16>::max());
    }

    /// @brief Unpacks 16 bits snorm value on 32 bits.
    inline float SNorm16ToFloat32(const s16 _value)
    {
        return _value < 0
          ? -static_cast<float>(_value) / std::numeric_limits<s16>::min()
          :  static_cast<float>(_value) / std::numeric_limits<s16>::max();
    }

    /// @brief Aligns the memory value on _alignment bytes.
    inline int GetAligned(const int _value, const int _alignment)
    {
        return _value + (_alignment - 1) & ~(_alignment - 1);
    }

    /// @brief Creates a bit mask covering _count bits.
    template <typename T>
    constexpr T BitfieldMask(const int _count)
    { 
        return (1 << _count) - 1;
    }

    /// @brief Creates a bit mask covering _count bits starting at _offset.
    template <typename T>
    T BitfieldMask(const int _offset, const int _count) 
    { 
        return BitfieldMask<T>(_count) << _offset;
    }

    /// @brief Inserts _count least significant bits from _insert into _base at _offset.
    template <typename T>
    T BitfieldInsert(T _base, T _insert, const int _offset, const int _count)
    { 
        T mask = BitfieldMask<T>(_count);
        return (_base & ~(mask << _offset)) | ((_insert & mask) << _offset);
    }

    /// @brief Extracts _count bits from _base starting at _offset into the _count least significant bits of the result.
    template <typename T>
    T BitfieldExtract(T _base, const int _offset, const int _count)
    { 
        T mask = BitfieldMask<T>(_offset, _count);
        return (_base & mask) >> _offset;
    }

    /// @brief Reverses the sequence of _bits.
    template <typename T>
    T BitfieldReverse(T _bits) 
    { 
        T ret = _bits;
        int s = sizeof(T) * 8 - 1;

        for (_bits >>= 1; _bits != 0; _bits >>= 1)
        {
            ret <<= 1;
            ret |= _bits & 1;
            --s;
        }

        ret <<= s;
        return ret;
    }

}
