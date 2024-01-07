#include <avapch.h>
#include "StringBuilder.h"

#include <Debug/Assert.h>

namespace Ava {

    // ------ String builder append methods ----------------------------------------------------

    void StringBuilder::append(const char _c)
    {
        const size_t writeOff = m_buffer.size();
        const size_t neededSize = writeOff + 1;
        if (neededSize >= m_buffer.capacity())
        {
            const size_t doubleCapacity = m_buffer.capacity() * 2;
            m_buffer.reserve(neededSize > doubleCapacity ? neededSize : doubleCapacity);
        }

        m_buffer.resize(neededSize);
        m_buffer[writeOff-1] = _c;
        m_buffer[writeOff] = 0;
    }

    void StringBuilder::append(const char* _str, const char* _strEnd/*= nullptr*/)
    {
        if (!_strEnd)
            _strEnd = _str + strlen(_str);
        const size_t len = _strEnd - _str;
        if (len == 0)
            return;

        const size_t writeOff = m_buffer.size();
        const size_t neededSize = writeOff + len;
        if (neededSize >= m_buffer.capacity())
        {
            const size_t doubleCapacity = m_buffer.capacity() * 2;
            m_buffer.reserve(neededSize > doubleCapacity ? neededSize : doubleCapacity);
        }

        m_buffer.resize(neededSize);
        memcpy(&m_buffer[writeOff-1], _str, len);
        m_buffer[writeOff-1+len] = 0;
    }

    void StringBuilder::append(const std::string& _str)
    { 
        append(_str.c_str(), _str.c_str() + _str.length());
    }

    void StringBuilder::appendv(const char* _format, const va_list _args)
    {
        va_list argsCopy;
        va_copy(argsCopy, _args);

        const int len = vsnprintf(nullptr, 0, _format, _args);
        if (len <= 0)
        {
            return;
        }

        const size_t writeOff = m_buffer.size();
        const size_t neededSize = writeOff + (size_t)len;
        if (writeOff + (size_t)len >= m_buffer.capacity())
        {
            const size_t doubleCapacity = m_buffer.capacity() * 2;
            m_buffer.reserve(neededSize > doubleCapacity ? neededSize : doubleCapacity);
        }

        m_buffer.resize(neededSize);
        StrFormatV(&m_buffer[writeOff] - 1, (size_t)len+1, _format, argsCopy);
    }

    void StringBuilder::appendf(const char* _fmt, ...)
    {
        va_list args;
        va_start(args, _fmt);
        appendv(_fmt, args);
        va_end(args);
    }


    // ------ String formatting, search, etc. --------------------------------------------------

    size_t StrFormatV(char* _buffer, const size_t _bufferSize, const char* _format, const va_list _args)
    {
        const int w = vsnprintf(_buffer, _bufferSize, _format, _args);
        _buffer[_bufferSize-1] = 0;
        return (w == -1) ? _bufferSize-1 : (size_t)w;
    }

    size_t StrFormatV(std::string& _strOut, const char* _format, const va_list _args)
    {
        int w = vsnprintf(nullptr, 0, _format, _args);
        AVA_ASSERT(w >= 0);
        
        _strOut.resize((size_t)w);
        w = vsnprintf(&_strOut[0], (size_t)w+1, _format, _args);
        AVA_ASSERT(w == (int)_strOut.size());

        return w;
    }
        
    size_t StrFormat(char* _buffer, const size_t _bufferSize, const char* _format, ...)
    {
        va_list args;
        va_start(args, _format);
        const size_t count = StrFormatV(_buffer, _bufferSize, _format, args);
        va_end(args);
        return count;
    }

    size_t StrFormat(std::string& _outStr, const char* _format, ...)
    {
        va_list args;
        va_start(args, _format);
        const size_t count = StrFormatV(_outStr, _format, args);
        va_end(args);
        return count;
    }

    const char* StrFind(const char* _haystack, const char* _needle, const char* _needleEnd)
    {
        if (!_needleEnd)
        {
            _needleEnd = _needle + strlen(_needle);
        }

        const char un0 = *_needle;
        while (*_haystack)
        {
            if (*_haystack == un0)
            {
                const char* b = _needle + 1;
                for (const char* a = _haystack + 1; b < _needleEnd; a++, b++)
                {
                    if (*a != *b)
                    {
                        break;
                    }
                }
                if (b == _needleEnd)
                {
                    return _haystack;
                }
            }
            _haystack++;
        }
        return nullptr;
    }

    const char* StrFindNoCase(const char* _haystack, const char* _needle, const char* _needleEnd)
    {
        if (!_needleEnd)
        {
            _needleEnd = _needle + strlen(_needle);
        }

        const char un0 = (char)toupper(*_needle);
        while (*_haystack)
        {
            if (toupper(*_haystack) == un0)
            {
                const char* b = _needle + 1;
                for (const char* a = _haystack + 1; b < _needleEnd; a++, b++)
                {
                    if (toupper(*a) != toupper(*b))
                    {
                        break;
                    }
                }
                if (b == _needleEnd)
                {
                    return _haystack;
                }
            }
            _haystack++;
        }
        return nullptr;
    }

    std::string StrPrettyInt(const int _value)
    {
        const std::string s = std::to_string(_value);
        std::string tempResult = "";
        int j = 0;

        for (int i = (int)s.size()-1; i >= 0; i--)
        {
            if (j % 3 == 0)
            {
                tempResult.append(",");
            }
            tempResult.append(s, i, 1);
            j++;
        }

        tempResult = tempResult.substr(1, tempResult.size() - 1);
        std::string result = "";

        for (int i = (int)tempResult.size() - 1; i >= 0; i--) 
        {
            result.append(tempResult, i, 1);
        }
        
        return result;
    }

}
