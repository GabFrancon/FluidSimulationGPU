#pragma once
/// @file BitField.h
/// @brief

#include <Core/Base.h>
#include <Debug/Assert.h>

namespace Ava {

    template <class IntegerType, int BitCount = 0>
    class BitField
    {
    public:
        IntegerType bits() const { return m_bits; }

        void set() { m_bits = ~0; }
        void set(const u32 _index)
        {
            AVA_ASSERT(_index < kBitSize, "Invalid bit field index.");
            m_bits |= (IntegerType(1) << _index);
        }

        void reset() { m_bits = 0; }
        void reset(const u32 _index)
        {
            AVA_ASSERT(_index < kBitSize, "Invalid bit field index.");
            m_bits &= ~(IntegerType(1) << _index);
        }

        void flip() { m_bits = ~m_bits; }
        void flip(const u32 _index)
        {
            AVA_ASSERT(_index < kBitSize, "Invalid bit field index.");
            m_bits ^= (IntegerType(1) << _index);
        }

        bool any() const { return m_bits != 0; }
        bool none() const { return m_bits == 0; }
        bool test(const u32 _index) const
        {
            AVA_ASSERT(_index < kBitSize, "Invalid bit field index.");
            return m_bits & (IntegerType(1) << _index);
        }

        bool operator==(const BitField<IntegerType, BitCount>& _other) const { return m_bits == _other.m_bits; }
        bool operator!=(const BitField<IntegerType, BitCount>& _other) const { return m_bits != _other.m_bits; }

    private:
        IntegerType m_bits = 0;
        static const u32 kBitSize = BitCount ? BitCount : 8 * sizeof(IntegerType);
    };

}
