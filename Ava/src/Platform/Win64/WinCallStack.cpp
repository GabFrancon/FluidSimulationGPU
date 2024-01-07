#include <avapch.h>

#include <Debug/Log.h>
#include <Debug/CallStack.h>
#include <Files/FileManager.h>

//--- Include Windows ------------------------
#include <dbghelp.h>
//--------------------------------------------

namespace Ava {

    static const char* kUnknownStackName = "Unknown";

    void CallStack::LoadDebugSymbols()
    {
        const HANDLE hProcess = GetCurrentProcess();
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);

        if (!SymInitialize(hProcess, nullptr, true))
        {
            assert(false);
        }

        if (!SymRefreshModuleList(hProcess))
        {
            const DWORD error = GetLastError();
            char* errorStr = FileMgr::GetFormattedErrorString(error);

            AVA_CORE_ERROR("[Windows] SymRefreshModuleList() failed : %s", errorStr);
            FileMgr::FreeFormattedErrorString(errorStr);
        }
    }

    const void* CallStack::GetSymbolName(const void* _ptr, StringBuilder* _outStr)
    {
        const HANDLE hProcess = GetCurrentProcess();
        DWORD64 dwDisplacement = 0;
        const DWORD64 dwAddress = DWORD64(_ptr);

        constexpr char pSymbolBuffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)]{};
        const auto pSymbol = (PSYMBOL_INFO)pSymbolBuffer;

        pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        pSymbol->MaxNameLen = MAX_SYM_NAME;

        if (!SymFromAddr(hProcess, dwAddress, &dwDisplacement, pSymbol))
        {
            _outStr->append(kUnknownStackName);
        }
        else
        {
            _outStr->append(pSymbol->Name);
        }

        return (u8*)_ptr - dwDisplacement;
    }

    void CallStack::GetSourceFileInformation(const void* _ptr, StringBuilder* _outStr)
    {
        const HANDLE hProcess = GetCurrentProcess();
        DWORD dwDisplacementLine;
        const DWORD64 dwAddress = DWORD64(_ptr);

        IMAGEHLP_LINE64 line{};
        line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

        SymSetOptions(SYMOPT_LOAD_LINES);

        if (SymGetLineFromAddr(hProcess, dwAddress, &dwDisplacementLine, &line))
        {
            _outStr->appendf("%s(%d):", line.FileName, line.LineNumber);
        }
    }

    int CallStack::GetCallStack(void** _backTrace, const int _backTraceMaxSize, const int _skipNbStack /*= 1*/, u32* _outHash /*= nullptr*/)
    {
        return CaptureStackBackTrace(_skipNbStack, _backTraceMaxSize, _backTrace, (PDWORD)_outHash);
    }

}