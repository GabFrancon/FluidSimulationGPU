#pragma once
/// @file FileEvent.h
/// @brief

#include <Events/Event.h>

namespace Ava {

    /// @brief Event emitted when a new file from working directory is changed.
    class FileChangedEvent final : public Event
    {
    public:
        EVENT_TYPE(FileChanged)
        EVENT_FLAGS(AVA_EVENT_FILE)
        FileChangedEvent(const char* _filepath, const int _flags) : m_filePath(_filepath), m_actionFlags(_flags) {}

        const char* GetPath() const { return m_filePath; }
        int GetActionFlags() const { return m_actionFlags; }

        std::string ToString() const override
        {
            std::string ret("FileChangedEvent: ");
            ret += std::string(m_filePath);
            return ret;
        }

    protected:
        const char* m_filePath = nullptr;
        const int m_actionFlags = 0;
    };

}
