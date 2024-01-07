#include <avapch.h>
#include "File.h"

#include <Files/FileManager.h>
#include <Strings/StringBuilder.h>
#include <Debug/Log.h>
#include <Math/Hash.h>

namespace Ava {

    // ---- File ---------------------------------------------------------------------------------

    File::File()
    {
    }

    File::~File()
    {
        if (IsOpen())
        {
            Close();
        }
    }

    bool File::Open(const char* _absolutePath, const u32 _fileFlags)
    {
        if (m_file)
        {
            // file was already opened
            StrFormat(m_errorStr, "[File] '%s' is already opened.", _absolutePath);
            return false;
        }

        m_path = _absolutePath;
        m_flags = _fileFlags;
        std::string mode;

        // access type
        if (HasFlag(AVA_FILE_READ))
        {
            mode += "r";
        }
        else if (HasFlag(AVA_FILE_WRITE))
        {
            mode += "w";
        }
        else if (HasFlag(AVA_FILE_APPEND))
        {
            mode += "a";
        }

        // file format
        if (HasFlag(AVA_FILE_TEXT))
        {
            mode += "t";
        }
        else if (HasFlag(AVA_FILE_BINARY))
        {
            mode += "b";
        }

        if (mode.empty())
        {
            StrFormat(m_errorStr, "[File] invalid combination of flags for %s.", m_path.c_str());
            return false;
        }

        // creates file directories
        FileMgr::CreateDirectories(_absolutePath);

        // opens file
        m_file = fopen(_absolutePath, mode.c_str());

        if (!m_file)
        {
            // open operation failed
            StrFormat(m_errorStr, 
                "[File] failed to open %s file '%s' for %s : %s",
                HasFlag(AVA_FILE_BINARY) ? "binary" : "text",
                _absolutePath,
                HasFlag(AVA_FILE_READ) ? "read" : "write",
                strerror(errno));

            return false;
        }

        if (ferror(m_file))
        {
            // file opened with errors
            StrFormat(m_errorStr, "[File] %d", errno);
            fclose(m_file);
            return false;
        }

        return true;
    }

    bool File::Open(const char* _rootDir, const char* _path, const u32 _fileFlags)
    {
        const char lastChar = _rootDir[strlen(_rootDir) - 1];
        const bool missingSlash = lastChar != '/' && lastChar != '\\';

        char absolutePath[MAX_PATH];
        StrFormat(absolutePath, "%s%s%s", _rootDir, missingSlash ? "/" : "", _path);

        return Open(absolutePath, _fileFlags);
    }

    bool File::Close()
    {
        if (!m_file)
        {
            StrFormat(m_errorStr, "[File] '%s' is not opened.", m_path.c_str());
            return false;
        }

        fclose(m_file);
        m_file = nullptr;

        return true;
    }

    bool File::Read(void* _buffer, const u32 _bytesCount)
    {
        if (!m_file)
        {
            m_errorStr = "You must open the file before attempting to read it.";
            return false;
        }
        if (!HasFlag(AVA_FILE_READ))
        {
            m_errorStr = "File is missing AVA_FILE_READ initialization flag.";
            return false;
        }
        if (!_buffer)
        {
            m_errorStr = "Dst buffer was not allocated.";
            return false;
        }

        fread(_buffer, 1, _bytesCount, m_file);
        return true;
    }

    bool File::Write(const void* _buffer, const u32 _bytesCount)
    {
        if (!m_file)
        {
            m_errorStr = "You must open the file before attempting to write in it.";
            return false;
        }
        if (!HasFlag(AVA_FILE_WRITE))
        {
            m_errorStr = "File is missing AVA_FILE_WRITE initialization flag.";
            return false;
        }
        if (!_buffer)
        {
            m_errorStr = "Dst buffer was null.";
            return false;
        }

        fwrite(_buffer, 1, _bytesCount, m_file);
        return true;
    }

    bool File::Appendv(const char* _format, const va_list _args)
    {
        if (!m_file)
        {
            m_errorStr = "You must open a file before attempting to write in it.";
            return false;
        }
        if (!HasFlag(AVA_FILE_WRITE))
        {
            m_errorStr = "File is missing AVA_FILE_WRITE initialization flag.";
            return false;
        }
        if (!_format)
        {
            m_errorStr = "String format was null.";
            return false;
        }

        vfprintf(m_file, _format, _args);
        return true;
    }

    bool File::Appendf(const char* _format, ...)
    {
        va_list args;
        va_start(args, _format);
        const bool res = Appendv(_format, args);
        va_end(args);

        return res;
    }

    bool File::Append(const char* _format)
    {
        if (!m_file)
        {
            m_errorStr = "You must open a file before attempting to write in it.";
            return false;
        }
        if (!HasFlag(AVA_FILE_WRITE))
        {
            m_errorStr = "File is missing AVA_FILE_WRITE initialization flag.";
            return false;
        }
        if (!_format)
        {
            m_errorStr = "String format was null.";
            return false;
        }

        fprintf(m_file, _format);
        return true;
    }

    u32 File::GetSize() const
    {
        if (!m_file)
        {
            return 0u;
        }

        // save cursor position
        fpos_t currentPos;
        fgetpos(m_file, &currentPos);

        // search for the end of file
        fseek(m_file, 0, SEEK_END);
        const u32 fileSize = ftell(m_file);

        // reset cursor position
        fsetpos(m_file, &currentPos);

        return fileSize;
    }

    u32 File::GetCursor() const
    {
        if (!m_file)
        {
            return 0u;
        }

        return static_cast<u32>(ftell(m_file));
    }


    // ---- File access entry --------------------------------------------------------------------

    FileAccessEntry::FileAccessEntry()
    {
        fileHash = 0;
        timestamp = 0;
        accessType = AVA_FILE_NONE;
    }

    FileAccessEntry::FileAccessEntry(const char* _filePath, const FileFlags _accessType)
    {
        filePath = _filePath;
        fileHash = HashStr(_filePath);
        accessType = _accessType;
        timestamp = 0;
    }


    // ---- File access logger -------------------------------------------------------------------

    FileAccessLogger::FileAccessLogger()
    {
        FileMgr::GetCurrentTimeAsFileTime(startTime);
    }

    void FileAccessLogger::RegisterFileAccess(const char* _absolutePath, const FileFlags _accessType)
    {
        // Don't register file accesses to folders
        const char last = _absolutePath[strlen(_absolutePath) - 1];
        if (last == '/' || last == '\\')
        {
            return;
        }

        // Sanitize path
        char cleanPath[MAX_PATH];
        FilePath::SanitizeSlashes(cleanPath, MAX_PATH, _absolutePath);

        // Check existing
        const auto hash = HashStr(cleanPath);
        for (const auto& entry : entries)
        {
            if (entry.fileHash == hash)
            {
                return;
            }
        }

        // Add access entry
        auto& fileEntry = entries.emplace_back(cleanPath, _accessType);
        FileMgr::GetCurrentTimeAsFileTime(fileEntry.timestamp);
    }

    void FileAccessLogger::RegisterFileAccess(const char* _rootDir, const char* _path, const FileFlags _accessType)
    {
        const char last = _rootDir[strlen(_rootDir) - 1];
        const bool missingSlash = last != '/' && last != '\\';

        char absolutePath[MAX_PATH];
        StrFormat(absolutePath, "%s%s%s", _rootDir, missingSlash ? "/" : "", _path);
        RegisterFileAccess(absolutePath, _accessType);
    }

    void FileAccessLogger::ExportDependencies(const char* _depPath, const bool _errored) const
    {
        File depFile;
        if (!depFile.Open(_depPath, AVA_FILE_WRITE | AVA_FILE_TEXT))
        {
            AVA_CORE_ERROR(depFile.GetErrorStr());
            return;
        }

        // command status
        depFile.Appendf("STATUS: %s\n", _errored ? "ERROR": "SUCCESS");

        // command dependencies
        for (const FileAccessEntry& dep : entries)
        {
            depFile.Appendf("DEP: %c%c / TIME: %llx / FILE: %s\n", 
                dep.accessType & AVA_FILE_READ ? 'R' : '-', 
                dep.accessType & AVA_FILE_WRITE ? 'W' : '-', 
                dep.timestamp, dep.filePath.c_str());
        }

        depFile.Close();

        // Set .dep timestamp to the start of our build to avoid missing out
        // on files that were modified since we started logging/building
        FILETIME lastWrite{};
        lastWrite.dwLowDateTime  = (DWORD)startTime;
        lastWrite.dwHighDateTime = startTime >> 32;
        FileMgr::SetLastWriteTime(_depPath, &lastWrite);
    }

}
