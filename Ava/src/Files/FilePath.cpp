#include <avapch.h>
#include "FilePath.h"

#include <Strings/StringBuilder.h>
#include <Debug/Assert.h>

namespace Ava {

    FilePath::FilePath(const FilePath& _other)
    {
        m_path = _other.m_path;
        m_fileStart = _other.m_fileStart;
    }

    FilePath::FilePath(FilePath&& _other) noexcept
    {
        m_path = std::move(_other.m_path);
        m_fileStart = _other.m_fileStart;
    }

    FilePath::FilePath(const std::string& _path)
    {
        Set(_path);
    }

    FilePath::FilePath(const std::string& _directory, const std::string& _fileName)
    {
        SetFileName(_fileName);
        SetDirectory(_directory);
    }

    FilePath& FilePath::operator=(const FilePath& _other)
    {
        m_path = _other.m_path;
        m_fileStart = _other.m_fileStart;
        return *this;
    }

    FilePath& FilePath::operator=(FilePath&& _other) noexcept
    {
        m_path = std::move(_other.m_path);
        m_fileStart = _other.m_fileStart;
        return *this;
    }

    FilePath& FilePath::operator=(const std::string& _path)
    {
        Set(_path);
        return *this;
    }

    bool FilePath::operator<(const FilePath& _other) const
    {
        return m_path < _other.m_path;
    }

    bool FilePath::operator<(const std::string& _other) const
    {
        return m_path < _other;
    }

    bool FilePath::operator==(const FilePath& _other) const
    {
        return m_path == _other.m_path;
    }

    bool FilePath::operator==(const std::string& _other) const
    {
        return m_path == _other;
    }


    // --- Filepath --------------------------------------------------------

    void FilePath::Set(const std::string& _path)
    {
        m_path = SanitizeSlashes(_path);

        const size_t lastSlash = m_path.rfind('/');
        m_fileStart = lastSlash != std::string::npos ? (u32)lastSlash + 1 : 0;
    }

    void FilePath::Set(const char* _path, const char* _pathEnd)
    {
        m_path = _pathEnd ? std::string(_path, _pathEnd) : _path;
        m_path = SanitizeSlashes(m_path);

        const size_t lastSlash = m_path.rfind('/');
        m_fileStart = lastSlash != std::string::npos ? (u32)lastSlash + 1 : 0;
    }

    const std::string& FilePath::str() const
    {
        return m_path;
    }

    const char* FilePath::c_str() const
    {
        return m_path.c_str();
    }

    size_t FilePath::size() const
    {
        return m_path.size();
    }

    bool FilePath::empty() const
    {
        return m_path.empty();
    }

    std::string FilePath::SanitizeSlashes(const std::string& _src)
    {
        std::string result;
        result.reserve(_src.size());

        for (const char c : _src)
        {
            if (c == '\\')
            {
                result.push_back('/');
            }
            else
            {
                result.push_back(c);
            }
        }

        return result;
    }

    const char* FilePath::SanitizeSlashes(char* _dst, const size_t _dstSize, const char* _src)
    {
        const char* dstEnd = _dst + _dstSize-1;
        char c = *_src++;

        while (_dst < dstEnd && c)
        {
            if (c == '\\')
            {
                c = '/';
            }
            *_dst++ = c;
            c = *_src++;
        }
        *_dst = '\0';
        return _dst;
    }

    const char* FilePath::GetRelativePath(const char* _filePath, const char* _rootPath)
    {
        const char* rootDirStart = StrFind(_filePath, _rootPath);
        return rootDirStart + strlen(_rootPath);
    }


    // --- Filename --------------------------------------------------------

    void FilePath::SetFileName(const std::string& _fileName)
    {
        AVA_ASSERT(_fileName.find_first_of('/') == std::string::npos, "Invalid filename");
        m_path = GetDirectory() + _fileName;
    }

    void FilePath::SetFileName(const char* _fileName)
    {
        AVA_ASSERT(strchr(_fileName, '/') == nullptr, "Invalid filename");
        m_path = GetDirectory() + _fileName;
    }

    std::string FilePath::GetFileName() const
    {
        return m_path.substr(m_fileStart);
    }

    const char* FilePath::GetFileNameCStr() const
    {
        return m_path.c_str() + m_fileStart;
    }

    size_t FilePath::GetFileNameSize() const
    {
        return m_path.size() - m_fileStart;
    }

    std::string FilePath::GetFileNameWithoutExtension() const
    {
        const char* filename = GetFileNameCStr();
        if (const char* extension = strrchr(filename, '.'))
        {
            return m_path.substr(m_fileStart, extension - filename);
        }
        return GetFileName();
    }

    const char* FilePath::FindFileName(const char* _filePath)
    {
        const char* fileNameStart = _filePath;
        char c = *_filePath++;

        while (c)
        {
            if (c == '/' || c == '\\')
            {
                fileNameStart = _filePath;
            }
            c = *_filePath++;
        }
        return fileNameStart;
    }

    void FilePath::RemoveFileName(char _dst[MAX_PATH], const char* _src)
    {
        const char* fileName = FindFileName(_src);
        StrFormat(_dst, MAX_PATH, "%.*s", fileName - _src, _src);
    }

    void FilePath::ReplaceFileName(char _filePath[MAX_PATH], const char* _fileName)
    {
        char directory[MAX_PATH];
        RemoveFileName(directory, _filePath);

        AVA_ASSERT(!strchr(_fileName, '/'), "[Path] '%s' shoud not contain any slash characted.", _fileName);
        StrFormat(_filePath, MAX_PATH, "%s%s", directory, _fileName);
    }


    // --- Directory -------------------------------------------------------

    void FilePath::SetDirectory(const std::string& _directory)
    {
        const std::string fileName = GetFileName();
        if (_directory.empty() || _directory[_directory.size() - 1] == '/')
        {
            m_fileStart = (u32)_directory.size();
            m_path = _directory + fileName;
        }
        else
        {
            m_fileStart = (u32)_directory.size() + 1;
            m_path = _directory + "/" + fileName;
        }
    }

    std::string FilePath::GetDirectory() const
    {
        return m_path.substr(0, m_fileStart);
    }

    size_t FilePath::GetDirectorySize() const
    {
        return m_fileStart;
    }


    // --- Extensions ------------------------------------------------------

    void FilePath::AddExtension(const std::string& _extension)
    {
        if (_extension.rfind('.') == std::string::npos)
        {
            m_path += ".";
        }
        m_path += _extension;
    }

    void FilePath::SetLastExtension(const std::string& _extension)
    {
        // Early discard
        if (_extension.empty() || m_fileStart >= m_path.size())
        {
            return;
        }

        const size_t lastDot = m_path.rfind('.');
        if (lastDot != std::string::npos)
        {
            m_path.resize(lastDot);
        }
        AddExtension(_extension);
    }

    void FilePath::SetFullExtension(const std::string& _extension)
    {
        // Early discard
        if (_extension.empty() || m_fileStart >= m_path.size())
        {
            return;
        }

        const size_t firstDot = m_path.find('.');
        if (firstDot != std::string::npos)
        {
            m_path.resize(firstDot);
        }
        AddExtension(_extension);
    }

    void FilePath::RemoveLastExtension()
    {
        if (m_fileStart < m_path.size())
        {
            const size_t lastDot = m_path.rfind('.');
            if (lastDot != std::string::npos)
            {
                m_path.resize(lastDot);
            }
        }
    }

    void FilePath::RemoveFullExtension()
    {
        if (m_fileStart < m_path.size())
        {
            const size_t firstDot = m_path.find('.');
            if (firstDot != std::string::npos)
            {
                m_path.resize(firstDot);
            }
        }
    }

    std::string FilePath::GetLastExtension() const
    {
        if (m_fileStart < m_path.size())
        {
            const size_t lastDot = m_path.rfind('.');
            if (lastDot != std::string::npos)
            {
                return m_path.substr(lastDot + 1);
            }
        }
        return "";
    }

    std::string FilePath::GetFullExtension() const
    {
        if (m_fileStart < m_path.size())
        {
            const size_t firstDot = m_path.find('.');
            if (firstDot != std::string::npos)
            {
                return m_path.substr(firstDot + 1);
            }
        }
        return "";
    }

    const char* FilePath::GetLastExtensionCStr() const
    {
        if (m_fileStart < m_path.size())
        {
            if (const char* lastExt = strrchr(m_path.c_str() + m_fileStart, '.'))
            {
                return lastExt;
            }
        }
        // Point at the end to allow for pointer arithmetic
        return m_path.c_str() + m_path.size();
    }

    const char* FilePath::GetFullExtensionCStr() const
    {
        if (m_fileStart < m_path.size())
        {
            if (const char* fullExt = strchr(m_path.c_str() + m_fileStart, '.'))
            {
                return fullExt;
            }
        }
        // Point at the end to allow for pointer arithmetic
        return m_path.c_str() + m_path.size();
    }

    const char* FilePath::FindExtension(const char* _filePath)
    {
        const char* fileNameStart = FindFileName(_filePath);
        const char* extensionStart = strchr(fileNameStart, '.');
        if (!extensionStart)
        {
            extensionStart = fileNameStart + strlen(fileNameStart);
        }
        return extensionStart;
    }

    void FilePath::RemoveExtension(char _dst[MAX_PATH], const char* _src)
    {
        const char* extension = FindExtension(_src);
        StrFormat(_dst, MAX_PATH, "%.*s", extension - _src, _src);
    }

    void FilePath::ReplaceExtension(char _filePath[MAX_PATH], const char* _extension)
    {
        char pathWithoutExtension[MAX_PATH];
        RemoveExtension(pathWithoutExtension, _filePath);

        AVA_ASSERT(_extension[0] == '.', "Invalid extension: %s.", _extension);
        StrFormat(_filePath, MAX_PATH, "%s%s", pathWithoutExtension, _extension);
    }

}
