#pragma once
/// @file StringBuilder.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    /// @brief String formatting helper.
    class StringBuilder
    {
    public:
        StringBuilder() { m_buffer.push_back(0); }
        ~StringBuilder() { m_buffer.clear(); }

        const char* c_str() const { return &m_buffer.front(); }
        const char* begin() const { return &m_buffer.front(); }
        const char* end() const { return &m_buffer.back(); }
        size_t size() const { return m_buffer.size()-1; }
        bool empty() const { return m_buffer.size() <= 1; }
        void clear() { m_buffer.clear(); m_buffer.push_back(0); }
        void reserve(const size_t _capacity) { m_buffer.reserve(_capacity); }

        void append(char _c);
        void append(const char* _str, const char* _strEnd = nullptr);
        void append(const std::string& _str);
        void appendf(const char* _fmt, ...);
        void appendv(const char* _format, va_list _args);

        void operator+=(const char* _str) { append(_str); }
        void operator+=(const std::string& _str) { append(_str); }

    private:
        std::vector<char> m_buffer;
    };

    // Finds substring in string, exist in sensitive/non-sensitive to case.
    const char* StrFind(const char* _haystack, const char* _needle, const char* _needleEnd = nullptr);
    const char* StrFindNoCase(const char* _haystack, const char* _needle, const char* _needleEnd = nullptr);

    // 100000000 -> "100,000,000".
    std::string StrPrettyInt(int _value);

    // String formatting using va_args.
    size_t StrFormatV(char* _buffer, size_t _bufferSize, const char* _format, va_list _args);
    size_t StrFormatV(std::string& _strOut, const char* _format, va_list _args);

    size_t StrFormat(char* _buffer, size_t _bufferSize, const char* _format, ...);
    size_t StrFormat(std::string& _outStr, const char* _format, ...);

    template <size_t BufferSize> 
    size_t StrFormatV(char (&_buffer)[BufferSize], const char* _format, const va_list _args)
    {
        return StrFormatV(_buffer, BufferSize, _format, _args);
    }

    template <size_t BufferSize> 
    size_t StrFormat(char (&_buffer)[BufferSize], const char* _format, ...)
    {
        va_list args;
        va_start(args, _format);
        const size_t count = StrFormatV(_buffer, BufferSize, _format, args);
        va_end(args);
        return count;
    }
}