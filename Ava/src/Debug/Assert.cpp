#include <avapch.h>
#include "Assert.h"

#include <Debug/CallStack.h>
#include <Debug/Log.h>
#include <Math/Hash.h>

namespace Ava {

    static bool s_assertEnabled = false;

    static std::string ComputeCallStack(const unsigned _extraDepthsToIgnore)
    {
        void* stack[100]{};
        u32 outHash{};

        // Skipping two stack levels to avoid:
        // - CallStack::GetCallStack
        // - this current function
        // - The caller could add an extra level to skip
        const auto collectedStackSize = CallStack::GetCallStack(stack, 20, 2 + _extraDepthsToIgnore, &outHash);
        StringBuilder writtenCallStack;
        CallStack::ToString(&stack[1], collectedStackSize, &writtenCallStack, true);
        return{ writtenCallStack.c_str() };
    }

    //--- Platform specific implementation of the assert function -------------------

    #if defined(AVA_PLATFORM_WINDOWS)

        static bool WinAssertHandler(AssertMgr::AssertAction& _action, const char* _cond, const char* _file, const int _line, const char* _msg)
        {
            if (_action == AssertMgr::AssertIgnore)
            {
                return false;
            }

            const std::string callStack = ComputeCallStack(1);

            AVA_CORE_TRACE("\n------------------------------------------------\n"
            "ASSERT(%s)\n"
            "%s\n"
            "File: %s\n"
            "Line: %d\n"
            "-- Stack --\n"
            "%s\n"
            "------------------------------------------------\n\n",
            _cond, _msg, _file, _line, callStack.c_str());

            if (_action == AssertMgr::AssertBreak)
            {
                return true;
            }

            if (_action == AssertMgr::AssertAsk)
            {
                const std::string title = "ASSERT";

                std::string finalMsg;
                StrFormat(finalMsg,
                    "ASSERT(%s)\n\n"
                    "%s\n\n"
                    "File: %s\n"
                    "Line: %d\n\n"
                    "-- Stack --\n"
                    "%s\n"
                    "Press Yes to break, No to ignore once, Cancel to ignore always.",
                    _cond, _msg, _file, _line, callStack.c_str());

                switch (MessageBoxA(nullptr, finalMsg.c_str(), title.c_str(), MB_YESNOCANCEL | MB_ICONEXCLAMATION))
                    {
                        case IDYES:
                            return true;
                        case IDNO:
                            return false;
                        case IDCANCEL:
                            _action = AssertMgr::AssertIgnore;
                            return false;
                        default:
                            return true;
                    }
            }
            return true;
        }

    #else

        static bool DefaultAssertHandler(const AssertMgr::AssertAction& _action, const char* _cond, const char* _file, const int _line, const char* _msg)
        {
            AVA_CORE_TRACE("\n------------------------------------------------\n"
            "ASSERT(%s)\n"
            "%s\n"
            "File: %s\n"
            "Line: %d\n"
            "------------------------------------------------\n\n",
            _cond, _msg, _file, _line);

            switch(_action)
            {
                case AssertMgr::AssertBreak:
                    return true;
                case AssertMgr::AssertIgnore:
                    return false;
                case AssertMgr::AssertAsk:
                    return true;
                default:
                    return true;
            }
            return true;
        }

    #endif

    //--- Assert manager implementation ---------------------------------------------

    void AssertMgr::EnableAsserts()
    {
        CallStack::LoadDebugSymbols();
        s_assertEnabled = true;
    }

    AssertMgr::AssertMap& AssertMgr::GetFailedAsserts()
    {
        // Deliberately leaks to have asserts during destruction of globals.
        static auto* s_failedAsserts = new AssertMap;
        return *s_failedAsserts;
    }

    AssertMgr::AssertHandler& AssertMgr::GetAssertHandler()
    {
    #if defined(AVA_PLATFORM_WINDOWS)
        static AssertHandler s_assertHandler = WinAssertHandler;
    #else
        static AssertHandler s_assertHandler = DefaultAssertHandler;
    #endif
        return s_assertHandler;
    }

    bool AssertMgr::Assert(const char* _cond, const char* _file, const int _line, const char* _msg, ...)
    {
        if (!s_assertEnabled)
        {
            return false;
        }

        AssertAction action = GetFailedAsserts().Get(_file, _line, AssertAsk);

        bool shouldBreak = action != AssertIgnore;
        const bool shouldFormatMsg = GetAssertHandler() != nullptr;

        if (shouldBreak && shouldFormatMsg)
        {
            static constexpr int kMsgBufferMaxSize = 1024;
            char formattedMsg[kMsgBufferMaxSize]{};

            if (_msg)
            {
                va_list args;
                va_start(args, _msg);
                StrFormatV(formattedMsg, kMsgBufferMaxSize, _msg, args);
                formattedMsg[sizeof(formattedMsg) - 1] = 0;
                va_end(args);
            }
            else
            {
                formattedMsg[0] = 0;
            }

            if (GetAssertHandler())
            {
                shouldBreak = GetAssertHandler()(action, _cond, _file, _line, formattedMsg);
            }
            GetFailedAsserts().Set(_file, _line, action);
        }
        return shouldBreak;
    }

    AssertMgr::AssertAction AssertMgr::AssertMap::Get(const char* _file, const int _line, const AssertAction _defaultAction)
    {
        const auto it = m_locationMap.find(AssertLocation(_file, _line));
        if (it != m_locationMap.end())
        {
            return it->second;
        }
        return _defaultAction;
    }

    void AssertMgr::AssertMap::Set(const char* _file, const int _line, const AssertAction _action)
    {
        m_locationMap[AssertLocation(_file, _line)] = _action;
    }

    size_t AssertMgr::AssertMap::AssertLocationHasher::operator()(const AssertLocation _location) const
    {
        u32 hash = HashStr(_location.file);
        hash = HashU32Combine(_location.line, hash);
        return hash;
    }

}
