#pragma once
/// @file FileManager.h
/// @brief

#include <Core/Base.h>
#include <Events/Event.h>

#if !defined(MAX_PATH)
    #define MAX_PATH 260
#endif

namespace Ava {

    class FilePath;


    /// @brief Interface representing file system.
    class FileMgr
    {
        static FileMgr* s_instance;

    public:
        static void Init();
        static void Shutdown();

        static FileMgr* GetInstance() { return s_instance; }

        /// @brief Iterates through directories and retrieve the path of all files with given extension.
        static bool ListFilesInDir(const char* _folderName,std::vector<FilePath>& _filePaths, const char* _searchExtension = nullptr);

        // Platform dependent static helpers
        static bool IsPattern(const char* _path);
        static bool PatternMatch(const char* _path, const char* _pattern);
        static bool IsPathAbsolute(const char* _path);
        static bool FileExists(const char* _absolutePath);
        static bool FolderExists(const char* _folderPath);
        static bool GetLastWriteTime(const char* _absolutePath, void* _fileTime);
        static bool SetLastWriteTime(const char* _absolutePath, const void* _fileTime);
        static void GetLocalTimeFromFileTime(void* _localTime, u64 _fileTime);
        static void GetCurrentTimeAsFileTime(u64& _fileTime);
        static void CreateDirectories(const char* _absolutePath);
        static void SetWorkingDirectory(const char* _absolutePath);
        static void GetWorkingDirectory(char _absolutePath[MAX_PATH]);
        static void GetExecutablePath(char _absolutePath[MAX_PATH]);
        static char* GetFormattedErrorString(unsigned long _error);
        static void FreeFormattedErrorString(char* _errorStr);
        static void OsShellExecute(const char* _path);

    protected:
        explicit FileMgr();
        ~FileMgr();
    };

}

#define FILE_MGR Ava::FileMgr::GetInstance()