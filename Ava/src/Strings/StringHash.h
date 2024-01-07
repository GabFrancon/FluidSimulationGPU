#pragma once
/// @file StringHash.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    /// @brief Fast 32bit string hash.
    class StringHash
    {
    public:
        StringHash() = default;
        StringHash(u32 _hash);
        StringHash(const char* _str);
        StringHash(const std::string& _str);

        u32 GetValue() const;
        const char* GetString() const;

        bool operator==(const StringHash& _other) const { return m_hash == _other.GetValue(); }
        bool operator!=(const StringHash& _other) const { return m_hash != _other.GetValue(); }
        bool operator< (const StringHash& _other) const { return m_hash <  _other.GetValue(); }
        bool operator> (const StringHash& _other) const { return m_hash >  _other.GetValue(); }
        bool operator<=(const StringHash& _other) const { return m_hash <= _other.GetValue(); }
        bool operator>=(const StringHash& _other) const { return m_hash >= _other.GetValue(); }

        /// Invalid string hash
        static StringHash Invalid;

    private:
        void _UpdateString(const char* _str) const;

        // The hash value computed from
        // the input character string.
        u32 m_hash = 0;

    #if defined(AVA_STORE_STRING_HASH)
    // String map is made public to be
    // accessible from the .NATVIS files.
    public:
        class StringMap;
        static StringMap* s_globalStringMap;
    #endif
    };

}

// Template specialization for StringHash hashing
template <>
struct std::hash<Ava::StringHash>
{
    size_t operator()(const Ava::StringHash& _stringHash) const noexcept
    {
        return _stringHash.GetValue();
    }
};