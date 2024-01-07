#pragma once
/// @file FilePath.h
/// @brief

#include <Core/Base.h>

#if !defined(MAX_PATH)
    #define MAX_PATH 260
#endif

namespace Ava {

    /// @brief Generic helper class to create and manipulate with path-based strings.
    class FilePath
    {
    public:
        FilePath() = default;
        FilePath(const FilePath& _other);
        FilePath(FilePath&& _other) noexcept;
        explicit FilePath(const std::string& _path);
        explicit FilePath(const std::string& _directory, const std::string& _fileName);

        FilePath& operator=(const FilePath& _other);
        FilePath& operator=(FilePath&& _other) noexcept;
        FilePath& operator=(const std::string& _path);

        bool operator<(const FilePath& _other) const;
        bool operator<(const std::string& _other) const;
        bool operator==(const FilePath& _other) const;
        bool operator==(const std::string& _other) const;


        // --- Filepath -----------------------------------------------------
        void Set(const std::string& _path);
        void Set(const char* _path, const char* _pathEnd = nullptr);

        const std::string& str() const;
        const char* c_str() const;
        size_t size() const;
        bool empty() const;

        static std::string SanitizeSlashes(const std::string& _path);
        static const char* SanitizeSlashes(char* _dst, size_t _dstSize, const char* _src);
        static const char* GetRelativePath(const char* _filePath, const char* _rootPath);

        // --- Filename -----------------------------------------------------
        void SetFileName(const std::string& _fileName);
        void SetFileName(const char* _fileName);

        std::string GetFileName() const;
        const char* GetFileNameCStr() const;
        size_t GetFileNameSize() const;
        std::string GetFileNameWithoutExtension() const;

        static const char* FindFileName(const char* _filePath);
        static void RemoveFileName(char _dst[MAX_PATH], const char* _src);
        static void ReplaceFileName(char _filePath[MAX_PATH], const char* _fileName);

        // --- Directory ----------------------------------------------------
        void SetDirectory(const std::string& _directory);
        std::string GetDirectory() const;
        size_t GetDirectorySize() const;

        // --- Extensions ---------------------------------------------------
        void AddExtension(const std::string& _extension);
        void SetLastExtension(const std::string& _extension);
        void SetFullExtension(const std::string& _extension);
        void RemoveLastExtension();
        void RemoveFullExtension();

        std::string GetLastExtension() const;
        std::string GetFullExtension() const;
        const char* GetLastExtensionCStr() const;
        const char* GetFullExtensionCStr() const;

        static const char* FindExtension(const char* _filePath);
        static void RemoveExtension(char _dst[MAX_PATH], const char* _src);
        static void ReplaceExtension(char _filePath[MAX_PATH], const char* _extension);

    private:
        std::string m_path = "";
        u32 m_fileStart = 0u;
    };

}
