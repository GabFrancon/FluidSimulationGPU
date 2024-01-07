#pragma once
/// @file Hash.h
/// @brief File defining useful hash functions, used throughout the engine.

#include <Core/Base.h>

namespace Ava {

    //--------- 32bit hash functions ------------------------------------------------------------

    /// @brief Hashes some bytes and produces a 32bit hash.
    inline u32 HashU32(const u8* _bytes, const size_t _count, const u32 _basis = 2166136261U)
    {
        // FNV-1a hash function
        const u32 FNV_offsetBasis = _basis;
        static constexpr u32 FNV_prime = 16777619U;

        u32 hash = FNV_offsetBasis;
        for (u32 i = 0; i < _count; ++i)
        {
            hash ^= (u32)_bytes[i];
            hash *= FNV_prime;
        }
        return hash;
    }

    /// @brief Hashes an instance of T and produces a 32bit hash value.
    template <class T>
    u32 HashU32(const T& _val)
    {
        return HashU32((const u8*)&_val, sizeof(_val));
    }

    /// @brief Hashes an instance of T and combines it to a previously produced hash.
    template <class T>
    u32 HashU32Combine(const T& _val, const u32 _baseHash)
    {
        return HashU32((const u8*)&_val, sizeof(_val), _baseHash);
    }

    /// @brief Returns a valid base hash to use with HashU32Combine().
    inline u32 HashU32Init()
    {
        return HashU32(nullptr, 0);
    }


    //--------- 64bit hash functions ------------------------------------------------------------

    /// @brief Hashes some bytes and produces a 64bit hash value.
    inline u64 HashU64(const u8* _bytes, const size_t _count, const u64 _basis = 14695981039346656037ULL)
    {
        // FNV-1a hash function
        const u64 FNV_offsetBasis = _basis;
        static constexpr u64 FNV_prime = 1099511628211ULL;

        u64 hash = FNV_offsetBasis;
        for (u64 i = 0; i < _count; ++i)
        {
            hash ^= (u64)_bytes[i];
            hash *= FNV_prime;
        }

        return hash;
    }

    /// @brief Hashes an instance of T and produces a 64bit hash value.
    template <class T>
    u64 HashU64(const T& _val)
    {
        return HashU64((const u8*)&_val, sizeof(_val));
    }

    /// @brief Hashes an instance of T and combines it to a previously produced hash.
    template <class T>
    u64 HashU64Combine(const T& _val, const u64 _baseHash)
    {
        return HashU64((const u8*)&_val, sizeof(_val), _baseHash);
    }

    /// @brief Returns a valid base hash to use with HashU64Combine().
    inline u64 HashU64Init()
    {
        return HashU64(nullptr, 0);
    }


    //--------- string hash functions -----------------------------------------------------------

    /// @brief Recursive string hasher, internally used by HashStr().
    inline u32 HashStrRecursive(const u32 _hash, const char* _str)
    {
        return (*_str > 0) ? HashStrRecursive((_hash << 5) + _hash + *_str, _str + 1) : _hash;
    }

    /// @brief Hashes a character string to produce a 32bit hash value.
    inline u32 HashStr(const char* _str)
    {
        return (*_str > 0) ? HashStrRecursive(HashU32Init(), _str) : 0;
    }

}
