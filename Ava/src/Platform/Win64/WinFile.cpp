#include <avapch.h>

#include <Files/FileManager.h>
#include <Files/FilePath.h>
#include <Debug/Assert.h>
#include <Debug/Log.h>

//--- Include Windows ------------------------
#include <shlwapi.h>
#include <direct.h>
//--------------------------------------------

namespace Ava {

    bool FileMgr::IsPattern(const char* _path)
    {
        return strchr(_path, '*') || strchr(_path, '?');
    }

    bool FileMgr::PatternMatch(const char* _path, const char* _pattern)
    {
        return PathMatchSpecA(_path, _pattern) != FALSE;
    }

    bool FileMgr::IsPathAbsolute(const char* _path)
    {
        return !PathIsRelativeA(_path);
    }

    bool FileMgr::FileExists(const char* _absolutePath)
    {
        return PathFileExistsA(_absolutePath);
    }

    bool FileMgr::FolderExists(const char* _folderPath)
    {
        return PathIsDirectoryA(_folderPath);
    }

    bool FileMgr::GetLastWriteTime(const char* _absolutePath, void* _fileTime)
    {
        auto* time = static_cast<FILETIME*>(_fileTime);

        const HANDLE handle = CreateFileA(_absolutePath, 
            GENERIC_READ,                                       // access rights demanded
            FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE, // authorize other processes to do this
            nullptr,                                            // security attribute
            OPEN_EXISTING,                                      // only open if the file exists
            FILE_ATTRIBUTE_NORMAL,
            nullptr);

        if (handle != INVALID_HANDLE_VALUE)
        {
            const BOOL ret = GetFileTime(handle, nullptr, nullptr, time);
            CloseHandle(handle);
            return ret != 0;
        }
        return false;
    }

    bool FileMgr::SetLastWriteTime(const char* _absolutePath, const void* _fileTime)
    {
        auto* time = static_cast<const FILETIME*>(_fileTime);

        const HANDLE handle = CreateFileA(_absolutePath, 
            GENERIC_WRITE,                                      // access rights demanded
            FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE, // authorize other processes to do this
            nullptr,                                            // security attribute
            OPEN_ALWAYS,                                        // open or create if doesn't exist
            FILE_ATTRIBUTE_NORMAL,
            nullptr);

        if (handle != INVALID_HANDLE_VALUE)
        {
            const BOOL ret = SetFileTime(handle, nullptr, nullptr, time);
            CloseHandle(handle);
            return ret != 0;
        }
        return false;
    }

    void FileMgr::GetLocalTimeFromFileTime(void* _localTime, u64 _fileTime)
    {
        auto* systemTime = static_cast<SYSTEMTIME*>(_localTime);
        if (_fileTime)
        {
            FILETIME fileTime = *(FILETIME*)&_fileTime;
            FileTimeToLocalFileTime( &fileTime, &fileTime );
            FileTimeToSystemTime( &fileTime, systemTime );
        }
        else
        {
            memset(_localTime, 0, sizeof(SYSTEMTIME));
        }
    }

    void FileMgr::GetCurrentTimeAsFileTime(u64& _fileTime)
    {
        FILETIME time;
        GetSystemTimeAsFileTime(&time);
        _fileTime = time.dwLowDateTime | (u64)time.dwHighDateTime << 32;
    }

    void FileMgr::CreateDirectories(const char* _absolutePath)
    {
        char directoryPath[MAX_PATH];
        for (int n = 0; _absolutePath[n]; n++)
        {
            const char c = _absolutePath[n];
            if (c == '/' || c == '\\')
            {
                directoryPath[n] = 0;
                CreateDirectoryA(directoryPath, nullptr);
            }
            directoryPath[n] = c;
        }
    }

    void FileMgr::SetWorkingDirectory(const char* _absolutePath)
    {
        const int success = _chdir(_absolutePath);
        AVA_ASSERT(success == 0);
    }

    void FileMgr::GetWorkingDirectory(char _absolutePath[MAX_PATH])
    {
        char currentDirectory[MAX_PATH];
        GetCurrentDirectoryA(MAX_PATH, currentDirectory);
        FilePath::SanitizeSlashes(_absolutePath, MAX_PATH, currentDirectory);
    }

    void FileMgr::GetExecutablePath(char _absolutePath[MAX_PATH])
    {
        char exePath[MAX_PATH];
        GetModuleFileNameA(nullptr, exePath, MAX_PATH);
        FilePath::SanitizeSlashes(_absolutePath, MAX_PATH, exePath);
    }

    char* FileMgr::GetFormattedErrorString(const unsigned long _error)
    {
        LPVOID errorStr{};
        FormatMessageA( 
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        _error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&errorStr,
        0,
        nullptr 
        );

        return (char*) errorStr;
    }

    void FileMgr::FreeFormattedErrorString(char* _errorStr)
    {
        if (_errorStr)
        {
            LocalFree(_errorStr);
        }
    }

    void FileMgr::OsShellExecute(const char* _path)
    {
        ShellExecuteA(
            nullptr, 
            "open", 
            _path, 
            nullptr, 
            nullptr, 
            SW_SHOWDEFAULT
        );
    }

    bool FileMgr::ListFilesInDir(const char* _folderName, std::vector<FilePath>& _filePaths,const char* _searchExtension)
    {
        // Adds missing trailing slash
        std::string folderPath = _folderName;
        if (!folderPath.empty() && folderPath.back() != '/')
        {
            folderPath += '/';
        }

        // Converts folder path to Windows search query
        std::string searchQuery = folderPath;
        size_t slashID = searchQuery.find_first_of("/");

        while (slashID > -1)
        {
            searchQuery.replace(slashID,1,"\\");
            slashID = searchQuery.find_first_of("/");
        }
        searchQuery += "*";

        wchar_t wStr[512];
        mbstowcs(wStr,searchQuery.c_str(),512);

        WIN32_FIND_DATA findFileData;
        const HANDLE hFind = FindFirstFile(wStr, &findFileData);

        if (hFind == INVALID_HANDLE_VALUE)
        {
            const DWORD error = GetLastError();
            char* errorStr = GetFormattedErrorString(error);
            AVA_ERROR("[Files] could not enumerate files in '%s':\n-> %s", folderPath.c_str(), errorStr);
            FreeFormattedErrorString(errorStr);
            return false;
        }

        const bool hasExtension = _searchExtension && strlen(_searchExtension) > 0;

        do
        {
            char fileName[512];
            wcstombs(fileName,findFileData.cFileName,512);
            FilePath entryName(folderPath, fileName);

            if (strcmp(fileName, ".") && strcmp(fileName,"..") && !FolderExists(entryName.c_str()))
            {
                if (!hasExtension || strcmp(FilePath::FindExtension(fileName), _searchExtension) == 0)
                {
                    _filePaths.push_back(entryName);
                }
            }
        }
        while (FindNextFile(hFind, &findFileData) != NULL);
        FindClose(hFind);

        return true;
    }

}