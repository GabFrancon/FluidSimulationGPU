#pragma once
/// @file HashMap.h
/// @brief

#include <Core/Base.h>
#include <Debug/Assert.h>

namespace Ava {

    static constexpr u8 kHashMapIdxInvalid = 0xFF;

    /// @brief Stores unique elements based on their hash.
    template <class T>
    class HashMap
    {
    public:
        explicit HashMap(const u8 _capacity) : m_capacity(_capacity), m_size(0)
        {
            m_data = new T[_capacity];
            m_dataHash = new u32[_capacity];
        }

        ~HashMap()
        {
            delete[] m_data;
            delete[] m_dataHash;
        }

        u8 insert(const T& _val, const u32 _hash)
        {
            u8 idx = 0;
            if ((idx = find(_hash)) != kHashMapIdxInvalid)
            {
                return idx;
            }

            u8 i = m_size++;
            if (!AVA_VERIFY(i < m_capacity, "Cache has reach its full capacity!"))
            {
                return kHashMapIdxInvalid;
            }

            m_data[i] = _val;
            m_dataHash[i] = _hash;
            return i;
        }

        u8 insert(T&& _val, const u32 _hash)
        {
            u8 idx = 0;
            if ((idx = find(_hash)) != kHashMapIdxInvalid)
            {
                return idx;
            }

            u8 i = m_size++;
            if (!AVA_VERIFY(i < m_capacity, "Cache has reach its full capacity!"))
            {
                return kHashMapIdxInvalid;
            }

            m_data[i] = std::move(_val);
            m_dataHash[i] = _hash;
            return i;
        }

        u8 find(const u32 _hash) const
        {
            for (u8 i = 0; i < m_size; i++)
            {
                if (m_dataHash[i] == _hash)
                {
                    return i;
                }
            }
            return kHashMapIdxInvalid;
        }

        const T& operator[](u8 _idx) const
        {
            AVA_ASSERT(_idx < m_size, "Cache out of range");
            return m_data[_idx];
        }

    private:
        const u8 m_capacity;
        u8 m_size;

        T* m_data;
        u32* m_dataHash;
    };

}
