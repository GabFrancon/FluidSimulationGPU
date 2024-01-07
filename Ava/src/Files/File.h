#pragma once
/// @file File.h
/// @brief

#include <Core/Base.h>
#include <Files/FilePath.h>

namespace Ava {

    /// @brief Enumeration of flags classifying file operations, formats, and actions.
    enum FileFlags
    {
        AVA_FILE_NONE = 0,
        // file operations
        AVA_FILE_READ = AVA_BIT(0),
        AVA_FILE_WRITE = AVA_BIT(1),
        AVA_FILE_APPEND = AVA_BIT(2),
        // file formats
        AVA_FILE_TEXT = AVA_BIT(3),
        AVA_FILE_BINARY = AVA_BIT(4),
        // file actions
        AVA_FILE_CREATED = AVA_BIT(5),
        AVA_FILE_MODIFIED = AVA_BIT(6),
        AVA_FILE_DELETED = AVA_BIT(7)
    };

    /// @brief Versatile file access interface, supporting text and binary read/write operations.
    class File
    {
    public:
        File();
        ~File();

        bool Open(const char* _absolutePath, u32 _fileFlags);
        bool Open(const char* _rootDir, const char* _path, u32 _fileFlags);
        bool Close();

        // bytes mode
        bool Read(void* _buffer, u32 _bytesCount);
        bool Write(const void* _buffer, u32 _bytesCount);

        template <typename T>
        bool Read(T& _value)
        {
            return Read(&_value, sizeof(T));
        }

        template <typename T>
        bool Write(const T& _value)
        {
            return Write(&_value, sizeof(T));
        }

        // string mode
        bool Appendv(const char* _format, va_list _args);
        bool Appendf(const char* _format, ...);
        bool Append(const char* _format);

        u32 GetSize() const;
        u32 GetCursor() const;

        bool IsOpen() const { return m_file != nullptr; }
        bool HasFlag(const FileFlags _flag) const { return m_flags & _flag; }

        const char* GetPath() const { return m_path.c_str(); }
        const char* GetErrorStr() const { return m_errorStr.c_str(); }

    private:
        FilePath m_path;
        FILE* m_file = nullptr;
        u32 m_flags = AVA_FILE_NONE;
        std::string m_errorStr;
    };

    /// @brief Stores file operation details: path, type (read/write), and timestamp.
    struct FileAccessEntry
    {
        std::string filePath;
        FileFlags accessType;
        u32 fileHash;
        u64 timestamp;

        FileAccessEntry();
        FileAccessEntry(const char* _filePath, FileFlags _accessType);
    };

    /// @brief Collects FileAccessEntry instances and generates a .DEP file to log file operations.
    struct FileAccessLogger
    {
        std::vector<FileAccessEntry> entries;
        u64 startTime;

        FileAccessLogger();
        void RegisterFileAccess(const char* _absolutePath, FileFlags _accessType);
        void RegisterFileAccess(const char* _rootDir, const char* _path, FileFlags _accessType);
        void ExportDependencies(const char* _depPath, bool _errored) const;
    };

}
