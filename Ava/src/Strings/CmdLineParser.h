#pragma once
/// @file CmdLineParser.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    class CmdLineParser
    {
    public:
        CmdLineParser(int _argc, char** _argv);

        bool HasOption(const std::string& _option) const;

        template <typename T>
        bool GetOptionValue(const std::string& _option, T& _value) const;

    private:
        bool _GetArgument(const std::string& _token) const;
        bool _GetArgumentValue(const std::string& _token, std::string& _value) const;

        std::vector<std::string> m_tokens;
    };

}
