#pragma once
/// @file CallStack.h
/// @brief

#include <Strings/StringBuilder.h>

namespace Ava {

    /// @brief Call stack static interface.
    class CallStack
    {
    public:

        /// @brief Update symbol list. Does nothing if already initialized.
        static void LoadDebugSymbols();

        /// @brief Retrieves current call stack. Returns the size of _backTrace.
        /// @details Skips this function call (or more, as requested by _skipNbStack).
        /// @returns the call stack identifier.
        static int GetCallStack(void** _backTrace, int _backTraceMaxSize, int _skipNbStack = 1, u32* _outHash = nullptr);

        /// @brief Writes a readable symbol name in _outStr.
        /// @returns the address of the beginning of the symbol (may be used to group call stacks that contain the same symbols even if they are at a different place in the symbol).
        static const void* GetSymbolName(const void* _ptr, StringBuilder* _outStr);

        /// @brief Stringifies the source file information at the given address. 
        /// @returns readable source file info related to given address (ex : "C:/YourProject/Main/Src/YourObj.cpp:23").
        static void GetSourceFileInformation(const void* _ptr, StringBuilder* _outStr);

        /// @brief Converts the call stack into a readable string.
        static void ToString(const void* const* _backTrace, const int _backTraceSize, StringBuilder* _outStr, const bool _withFileInfo = false)
        {
            assert(_outStr);

            for (auto i = 0; i < _backTraceSize; i++)
            {
                _outStr->append("-> ");

                if (_withFileInfo)
                {
                    StringBuilder str;
                    GetSourceFileInformation(_backTrace[i], &str);

                    _outStr->append(str.c_str());
                    _outStr->append(' ');
                }

                StringBuilder str;
                GetSymbolName(_backTrace[i], &str);
                
                _outStr->append(str.c_str());
                _outStr->append('\n');
            }
        }
    };

}
