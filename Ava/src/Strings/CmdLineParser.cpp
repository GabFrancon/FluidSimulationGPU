#include <avapch.h>
#include "CmdLineParser.h"

namespace Ava {

    CmdLineParser::CmdLineParser(const int _argc, char** _argv)
    {
        for (int i = 1; i < _argc; ++i)
        {
            m_tokens.push_back(_argv[i]);
        }
    }

    bool CmdLineParser::HasOption(const std::string& _option) const
    {
        return _GetArgument(_option);
    }

    template <typename T>
    bool CmdLineParser::GetOptionValue(const std::string& _option, T& _value) const
    {
        return false;
    }

    template <>
    bool CmdLineParser::GetOptionValue(const std::string& _option, std::string& _value) const
    {
        std::string strValue;

        if (_GetArgumentValue(_option, strValue))
        {
            _value = strValue;
            return true;
        }

        return false;
    }

    template <>
    bool CmdLineParser::GetOptionValue(const std::string& _option, int& _value) const
    {
        std::string strValue;

        if (_GetArgumentValue(_option, strValue))
        {
            _value = std::stoi(strValue);
            return true;
        }

        return false;
    }

    template <>
    bool CmdLineParser::GetOptionValue(const std::string& _option, float& _value) const
    {
        std::string strValue;

        if (_GetArgumentValue(_option, strValue))
        {
            _value = std::stof(strValue);
            return true;
        }
        return false;
    }


// PRIVATE
    bool CmdLineParser::_GetArgument(const std::string& _token) const
    {
        return std::find(m_tokens.begin(), m_tokens.end(), _token) != m_tokens.end();
    }

    bool CmdLineParser::_GetArgumentValue(const std::string& _token, std::string& _value) const
    {
        auto findIt = std::find(m_tokens.begin(), m_tokens.end(), _token);

        if (findIt != m_tokens.end() && ++findIt != m_tokens.end())
        {
            _value = *findIt;
            return true;
        }

        return false;
    }
}
