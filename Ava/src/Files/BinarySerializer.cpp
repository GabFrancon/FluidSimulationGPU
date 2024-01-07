#include <avapch.h>
#include "BinarySerializer.h"

#include <Files/File.h>
#include <Memory/Memory.h>
#include <Math/Math.h>

namespace Ava {

    static constexpr int kBufferAlignment = 128;

    BinarySerializer::BinarySerializer(const SerializeMode _mode, const bool _pureBinary/*= false*/)
    : Serializer(_mode), m_size(0), m_buffer(nullptr), m_currentAddress(nullptr), m_pureBinary(_pureBinary), m_internalBuffer(true)
    {
        m_sectionsStack.emplace_back();
        _ComputeRootMember();
    }

    BinarySerializer::BinarySerializer(u8* _buffer, const u32 _size, const SerializeMode _mode, const bool _pureBinary/*= false*/)
    : BinarySerializer(_mode, _pureBinary)
    {
        _SetBuffer(_buffer, _size);
        _ComputeRootMember();
    }

    BinarySerializer::~BinarySerializer()
    {
        _ReleaseBuffer();
    }


    // ----- File ------------------------------------------------------------------------------------

    bool BinarySerializer::Load(const char* _path)
    {
        File binFile;
        if (!binFile.Open(_path, AVA_FILE_READ | AVA_FILE_BINARY))
        {
            AVA_CORE_ERROR(binFile.GetErrorStr());
            return false;
        }

        const u32 fileSize = binFile.GetSize();

        std::vector<u8> fileData(fileSize);
        binFile.Read(fileData.data(), fileSize);

        _CopyBuffer(fileData.data(), fileSize);
        _ComputeRootMember();
        binFile.Close();
        return true;
    }

    bool BinarySerializer::Save(const char* _path)
    {
        File binFile;
        if (!binFile.Open(_path, AVA_FILE_WRITE | AVA_FILE_BINARY))
        {
            return false;
        }

        binFile.Write(m_buffer, _GetCurrentSize());
        binFile.Close();
        return true;
    }


    // ----- Sections --------------------------------------------------------------------------------

    SerializeError BinarySerializer::OpenSection(const char* _sectionName)
    {
        //'we force a root section, so if you open it, do nothing
        if (_sectionName == nullptr && m_sectionsStack.size() == 1)
        {
            return SerializeError::None;
        }

        u32 hashTag = _ComputeHashTag(_sectionName);

        Section& section = m_sectionsStack.emplace_back();
        section.member.hashTag = hashTag;
        section.member.pureBinary = m_pureBinary;
        section.member.startIndex = _GetCurrentSize();
        section.member.type = SerializeValueType::Object;

        // we push a new section member even if the open fail
        // because a close can be call on a failed opened section
        m_currentSectionMembers.emplace_back();
        if (IsWriting())
        {
            section.member.WriteHeader(*this);
        }
        else
        {
            if (!m_pureBinary)
            {
                // current section is the entry last - 1
                auto &currentSectionMembers = m_currentSectionMembers[m_currentSectionMembers.size() - 2];
                const auto memberIt = std::find_if(currentSectionMembers.begin(), currentSectionMembers.end(),
                    [hashTag](const Member& _member)->bool { return _member.hashTag == hashTag; });

                if (memberIt == currentSectionMembers.end())
                {
                    return SerializeError::EntryNotFound;
                }
                section.member = *memberIt;
            }
            // set current address to the beginning of the section
            m_currentAddress = m_buffer + section.member.startIndex + section.member.GetHeaderSize();
        }
        if (!_RecomputeMembers())
        {
            return SerializeError::SectionError;
        }
        return SerializeError::None;
    }

    SerializeError BinarySerializer::CloseSection(const char* _sectionName)
    {
        // we force a root section, so if you opened it, do nothing
        if (_sectionName == nullptr && m_sectionsStack.size() == 1)
        {
            return SerializeError::None;
        }

        const u32 hashTag = _ComputeHashTag(_sectionName, true);
        Section section = m_sectionsStack.back();

        AVA_ASSERT(!section.isArray, "[Serializer] Previous array was not properly closed.");
        AVA_ASSERT(section.member.hashTag == hashTag, "[Serializer] Attempting to close invalid '%s' section.", _sectionName);

        _UpdateCurrentSectionSize();

        //set current address to the end of section
        section = m_sectionsStack.back();
        m_currentAddress = m_buffer + section.member.startIndex + section.member.GetHeaderSize() + section.member.size;
        m_sectionsStack.pop_back();

        AVA_ASSERT(m_currentSectionMembers.size());
        m_currentSectionMembers.pop_back();

        if (!m_sectionsStack.empty())
        {
            _UpdateCurrentSectionSize();
        }
        return SerializeError::None;
    }

    SerializeError BinarySerializer::CancelSection(const char* _sectionName)
    {
        //'we force a root section, so if you opened it, do nothing
        if (_sectionName == nullptr && m_sectionsStack.size() == 1)
        {
            return SerializeError::None;
        }

        const u32 hashTag = _ComputeHashTag(_sectionName, true);
        const Section& section = m_sectionsStack.back();

        AVA_ASSERT(!section.isArray, "[Serializer] Previous array was not properly closed.");
        AVA_ASSERT(section.member.hashTag == hashTag, "[Serializer] Attempting to cancel invalid '%s' section.", _sectionName);

        // set current address to start of the cancelled section
        m_currentAddress = m_buffer + section.member.startIndex;
        m_sectionsStack.pop_back();

        if (!m_sectionsStack.empty())
        {
            _UpdateCurrentSectionSize();

            if (!_RecomputeMembers())
            {
                return SerializeError::SectionError;
            }
        }
        return SerializeError::None;
    }


    // ----- Arrays --------------------------------------------------------------------------------

    SerializeError BinarySerializer::BeginArray(const char* _arrayName)
    {
        const u32 hashTag = _ComputeHashTag(_arrayName);
        Section& section = m_sectionsStack.emplace_back();

        section.isArray = true;
        section.member.pureBinary = m_pureBinary;
        section.member.hashTag = hashTag;

        section.member.startIndex = _GetCurrentSize();
        section.member.type = SerializeValueType::Array;

        if (m_pureBinary)
        {
            if (IsReading())
            {
                // just read the array count
                return _Read(section.arrayCount);
            }

            if (IsWriting())
            {
                // skip 4 bytes so we can write the array size
                // at the beginning later when calling EndArray()
                m_currentAddress += sizeof section.arrayCount;
            }

            return SerializeError::None;
        }

        // we push a new section member even if the open fails
        // because a close can be call on a failed opened array
        m_currentSectionMembers.emplace_back();

        if (IsWriting())
        {
            section.member.WriteHeader(*this);
        }
        else
        {
            // current section is the last entry - 1
            auto &arrayMembers = m_currentSectionMembers[m_currentSectionMembers.size() - 2];

            const auto member = std::find_if(
                arrayMembers.rbegin(), arrayMembers.rend(),
                [hashTag](const Member& _member) { return _member.hashTag == hashTag; });

            if (member == arrayMembers.rend())
            {
                return SerializeError::EntryNotFound;
            }

            section.member = *member;

            // set current address to the beginning of the section
            m_currentAddress = m_buffer + section.member.startIndex + section.member.GetHeaderSize();
        }

        if (!_RecomputeMembers())
        {
            return SerializeError::SectionError;
        }

        const auto& currentSessionMembers = m_currentSectionMembers.back();
        section.arrayCount = (u32)currentSessionMembers.size();
        return SerializeError::None;
    }

    SerializeError BinarySerializer::EndArray(const char* _arrayName)
    {
        const u32 hashTag = _ComputeHashTag(_arrayName, true);
        Section& section = m_sectionsStack.back();

        AVA_ASSERT(section.isArray, "[Serializer] EndArray() should not be called outside array scope.");
        AVA_ASSERT(section.member.hashTag == hashTag, "[Serializer] Attempting to close invalid '%s' array.", _arrayName);

        if (m_pureBinary)
        {
            if (IsWriting())
            {
                // save current address
                u8* currentAddress = m_currentAddress;

                // write the array size at the beginning of the array
                m_currentAddress = m_buffer + section.member.startIndex;
                AVA_ASSERT(section.arrayCount > 0, "[Serializer] about to serialize an empty array.");
                _Write(section.arrayCount);

                // reset current address
                m_currentAddress = currentAddress;
            }

            m_sectionsStack.pop_back();
            return SerializeError::None;
        }

        _UpdateCurrentSectionSize();
        m_currentAddress = m_buffer + section.member.startIndex + section.member.GetHeaderSize() + section.member.size;
        m_sectionsStack.pop_back();

        AVA_ASSERT(m_currentSectionMembers.size());
        m_currentSectionMembers.pop_back();

        _UpdateCurrentSectionSize();
        return SerializeError::None;
    }

    SerializeError BinarySerializer::NextElementInArray()
    {
        Section& section = m_sectionsStack.back();
        AVA_ASSERT(section.isArray, "[Serializer] NextElementInArray() should not be called outside array scope.");

        const int arrayIndex = ++section.arrayIndex;

        if (m_pureBinary)
        {
            if (IsWriting())
            {
                section.arrayCount++;
            }

            return arrayIndex < (int)section.arrayCount ? SerializeError::None : SerializeError::NoMoreElementInList;
        }

        auto& arrayMembers = m_currentSectionMembers.back();
        if (arrayIndex >= (int)arrayMembers.size())
        {
            if (IsWriting())
            {
                Member& member = arrayMembers.emplace_back();
                AVA_ASSERT(arrayIndex == (int)arrayMembers.size() - 1);
                member.pureBinary = m_pureBinary;
                section.arrayCount++;
            }
            else
            {
                return SerializeError::NoMoreElementInList;
            }
        }

        if (IsReading())
        {
            AVA_ASSERT(arrayMembers[arrayIndex].startIndex <= m_size, "[Serializer] start index exceeds total buffer size.");
            m_currentAddress = m_buffer + arrayMembers[arrayIndex].startIndex;
        }

        return SerializeError::None;
    }

    u32 BinarySerializer::GetArraySize() const
    {
        if (m_sectionsStack.back().isArray)
        {
            return m_sectionsStack.back().arrayCount;
        }

        return 0;
    }

    u32 BinarySerializer::GetCurrentArrayIndex() const
    {
        const Section& section = m_sectionsStack.back();
        AVA_ASSERT(section.isArray, "[Serializer] GetCurrentArrayIndex() should not be called outside array scope.");
        return (u32)section.arrayIndex;
    }

    SerializeValueType BinarySerializer::GetArrayElementType(const u32 _index) const
    {
        if (m_pureBinary)
        {
            return SerializeValueType::Unknown;
        }

        const Section& section = m_sectionsStack.back();
        AVA_ASSERT(section.isArray, "[Serializer] GetArrayElementType() should not be called outside array scope.");

        auto& arrayMembers = m_currentSectionMembers.back();
        AVA_ASSERT(_index < (u32)arrayMembers.size(), "[Serializer] array index exceeds the total number of array members.");
        return arrayMembers[_index].type;
    }


    // ----- Children --------------------------------------------------------------------------------

    u32 BinarySerializer::GetChildCount() const
    {
        if (m_sectionsStack.empty())
        {
            return 0;
        }
        if (m_sectionsStack.back().isArray)
        {
            return 0;
        }

        const auto& sectionMembers = m_currentSectionMembers.back();
        return (u32)sectionMembers.size();
    }

    SerializeValueType BinarySerializer::GetChildType(const char* _tag) const
    {
        if (m_pureBinary)
        {
            return SerializeValueType::Unknown;
        }

        const u32 hashTag = StringHash(_tag).GetValue();
        const auto& sectionMembers = m_currentSectionMembers.back();

        const auto memberIt = std::find_if(
            sectionMembers.begin(), sectionMembers.end(),
            [hashTag](const Member& _member)->bool { return _member.hashTag == hashTag; }
        );

        if (memberIt == sectionMembers.end())
        {
            return SerializeValueType::Null;
        }

        return memberIt->type;
    }

    u32 BinarySerializer::GetChildTags(const char** _tags, const u32 _tagsCount) const
    {
        AVA_ASSERT(!m_pureBinary);

        u32 index = 0;
        auto &members = m_currentSectionMembers.back();

        for (auto memberIt = members.begin(); memberIt != members.end(); ++memberIt)
        {
            if (index == _tagsCount)
            {
                break;
            }

            const auto stringHash = StringHash(memberIt->hashTag);
            _tags[index] = stringHash.GetString();
            index++;
        }
        return index;
    }


    // ----- Optimized serialization -----------------------------------------------------------------

    SerializeError BinarySerializer::SerializeBytes(const char* _tag, std::vector<char>& _bytes)
    {
        if (!m_pureBinary)
        {
            return Serialize(_tag, _bytes);
        }

        // In pure binary mode, we can quickly serialize array items
        // using one single memory copy operation for all entries.
        SerializeError result = BeginArray(_tag);

        if (result == SerializeError::None)
        {
            if (IsReading())
            {
                const u32 bytesCount = GetArraySize();
                _bytes.resize(bytesCount);

                result = _ReadArray(_bytes.data(), bytesCount);
            }
            else if (IsWriting())
            {
                const u32 bytesCount = (u32)_bytes.size();
                result = _WriteArray(_bytes.data(), bytesCount);

                // update array size
                auto& arraySection = m_sectionsStack.back();
                arraySection.arrayCount = bytesCount;
            }
        }

        EndArray(_tag);
        return result;
    }


    // ----- Recycling -------------------------------------------------------------------------------

    void BinarySerializer::Restart()
    {
        m_currentAddress = m_buffer;

        if (IsWriting())
        {
            Section& root = m_sectionsStack.front();
            root.member.WriteHeader(*this);
        }

        m_sectionsStack.clear();
        m_currentSectionMembers.clear();
        Section& section = m_sectionsStack.emplace_back();
        section.member.pureBinary = m_pureBinary;

        if (IsReading())
        {
            Section& root = m_sectionsStack.back();
            root.member.ReadHeader(*this);
        }
    }

    void BinarySerializer::Reset()
    {
        if (m_buffer != nullptr && m_internalBuffer)
        {
            AVA_FREE_ALIGNED(m_buffer);
        }

        m_buffer = nullptr;
        m_currentAddress = nullptr;
        m_size = 0;

        m_sectionsStack.clear();
        m_currentSectionMembers.clear();
        m_sectionsStack.emplace_back();
    }


    // ----- Object or array sections ----------------------------------------------------------------

    u32 BinarySerializer::Member::GetHeaderSize() const
    {
        if (!pureBinary)
        {
            // bytes to skip from startIndex to point buffer on member data
            return sizeof(hashTag) + sizeof(size) + sizeof(type);
        }

        return 0;
    }

    SerializeError BinarySerializer::Member::ReadHeader(BinarySerializer& _s)
    {
        if (!pureBinary)
        {
            if (_s._Read(hashTag) != SerializeError::None)
            {
                return SerializeError::CannotRead;
            }
            if (_s._Read(size) != SerializeError::None)
            {
                return SerializeError::CannotRead;
            }
            if (_s._Read(type) != SerializeError::None)
            {
                return SerializeError::CannotRead;
            }
        }

        return SerializeError::None;
    }

    SerializeError BinarySerializer::Member::WriteHeader(BinarySerializer& _s)
    {
        if (!pureBinary)
        {
            if (_s._Write(hashTag) != SerializeError::None)
            {
                return SerializeError::CannotWrite;
            }
            if (_s._Write(size) != SerializeError::None)
            {
                return SerializeError::CannotWrite;
            }
            if (_s._Write(type) != SerializeError::None)
            {
                return SerializeError::CannotWrite;
            }
        }

        return SerializeError::None;
    }


    // ----- Generic helpers -------------------------------------------------------------------------

    void BinarySerializer::_ComputeRootMember()
    {
        u8* currentAddress = m_currentAddress;
        m_currentAddress = m_buffer;
        Section& root = m_sectionsStack.front();
        root.member.pureBinary = m_pureBinary;

        if (IsReading())
        {
            root.member.ReadHeader(*this);
        }
        else
        {
            Member member = root.member;
            member.WriteHeader(*this);
            root.member = member;
        }

        if (currentAddress && currentAddress != m_buffer)
        {
            m_currentAddress = currentAddress;
        }

        m_currentSectionMembers.emplace_back();
        _RecomputeMembers();
    }

    bool BinarySerializer::_RecomputeMembers()
    {
        if (!m_buffer)
        {
            return false;
        }

        // early discards when in pure binary mode
        if (m_pureBinary)
        {
            return true;
        }

        u8* currentAddress = m_currentAddress;
        const Section& currentSection = m_sectionsStack.back();
        auto &currentSectionMembers = m_currentSectionMembers.back();
        u32 index = 0;

        // parse all members
        while (index < currentSection.member.size)
        {
            const u32 previousIndex = index;
            const u32 bytesToSkip = currentSection.member.startIndex + currentSection.member.GetHeaderSize() + index;
            AVA_ASSERT(bytesToSkip <= m_size, "[Serializer] bytes to skip exceeds total buffer size.");
            m_currentAddress = m_buffer + bytesToSkip;

            Member& member = currentSectionMembers.emplace_back();
            member.pureBinary = m_pureBinary;
            member.startIndex =_GetCurrentSize();

            member.ReadHeader(*this);
            index += member.GetHeaderSize() + member.size;
            AVA_ASSERT(index != previousIndex);
        }

        // back to previous address
        m_currentAddress = currentAddress;
        return true;
    }

    bool BinarySerializer::_EnsureIsReadable(const u32 _size) const
    {
        if (!m_buffer)
        {
            return false;
        }

        const u8* end = m_buffer + m_size;
        if (!AVA_VERIFY(m_currentAddress + _size <= end, "[Serializer] can't read binary buffer."))
        {
            return false;
        }
        return true;
    }

    bool BinarySerializer::_EnsureIsWritable(const u32 _size)
    {
        const u8* end = m_buffer + m_size;
        if (m_currentAddress + _size > end)
        {
            if (!m_internalBuffer)
            {
                return false;
            }

            const u32 newBufferSize = static_cast<u32>(m_currentAddress + _size - m_buffer);
            if (!_ResizeBuffer(newBufferSize))
            {
                AVA_ASSERT(false, "[Serializer] failed to allocate new buffer memory.");
                return false;
            }

            AVA_ASSERT(m_currentAddress != nullptr);
        }
        return true;
    }

    void BinarySerializer::_SetBuffer(u8* _buffer, const u32 _size)
    {
        _ReleaseBuffer();

        m_size = _size;
        m_buffer = _buffer;

        if (IsReading())
        {
            m_currentAddress = m_buffer;
        }
        else if (IsWriting())
        {
            m_currentAddress = m_buffer + _size;
        }

        m_internalBuffer = false;
    }

    void BinarySerializer::_CopyBuffer(const u8* _buffer, const u32 _size)
    {
        _ReleaseBuffer();

        m_size = _size;
        m_buffer = (u8*)AVA_MALLOC_ALIGNED(m_size, kBufferAlignment);

        if (m_buffer != nullptr && m_size > 0)
        {
            memcpy(m_buffer, _buffer, m_size);
        }

        if (IsReading())
        {
            m_currentAddress = m_buffer;
        }
        else if (IsWriting())
        {
            m_currentAddress = m_buffer + m_size;
        }

        m_internalBuffer = true;
    }

    bool BinarySerializer::_ResizeBuffer(const u32 _size)
    {
        // no need for resizing
        if (m_size >= _size)
        {
            return true;
        }

        // buffer can't be less than 16 bytes long
        u32 newAllocatedSize = Math::max(m_size, 16u);
        while (newAllocatedSize < _size)
        {
            newAllocatedSize *= 2;
        }

        if (m_currentAddress + _size > m_buffer + m_size)
        {
            u8* oldBuffer = m_buffer;
            const int oldSize = _GetCurrentSize();

            m_size = newAllocatedSize;
            m_buffer = (u8*)AVA_MALLOC_ALIGNED(m_size, kBufferAlignment);

            if (m_buffer != nullptr && oldSize > 0)
            {
                memcpy(m_buffer, oldBuffer, oldSize);
                m_currentAddress = m_buffer + oldSize;
                AVA_FREE_ALIGNED(oldBuffer);
            }
            else
            {
                m_currentAddress = m_buffer;
            }
        }

        AVA_ASSERT(m_buffer != nullptr);
        return m_buffer != nullptr;
    }

    void BinarySerializer::_ReleaseBuffer()
    {
        if (m_buffer != nullptr && m_internalBuffer)
        {
            AVA_FREE_ALIGNED(m_buffer);
        }

        m_buffer = nullptr;
    }

    void BinarySerializer::UpdateGlobalSize()
    {
        Section& root = m_sectionsStack.front();

        // size should not take header
        u32 currentSize = _GetCurrentSize() - root.member.GetHeaderSize();
        u8* currentAddress = m_currentAddress;
        u32 size = 0;

        m_currentAddress = m_buffer + sizeof(u32);

        if (!m_pureBinary)
        {
            _Read(size);
        }

        if (currentSize >= size)
        {
            m_currentAddress = m_buffer + sizeof(u32);

            if (!m_pureBinary)
            {
                _Write(currentSize);
            }
            root.member.size = currentSize;
        }

        // reset current address
        m_currentAddress = currentAddress;
    }

    void BinarySerializer::_UpdateCurrentSectionSize()
    {
        if (IsReading() && !m_pureBinary)
        {
            return;
        }

        // in pure binary mode, update size since you wont get it from somewhere else
        u8* currentAddress = m_currentAddress;
        Section& section = m_sectionsStack.back();

        // overwrite size
        const u8* startSection = m_buffer + section.member.startIndex + section.member.GetHeaderSize();
        const u8* endSection = m_currentAddress;

        if (endSection >= m_buffer)
        {
            section.member.size = static_cast<u32>(endSection - startSection);
            m_currentAddress = m_buffer + section.member.startIndex + 4;

            if (!m_pureBinary)
            {
                _Write(section.member.size);
            }
        }

        // reset current address
        m_currentAddress = currentAddress;
    }

    u32 BinarySerializer::_GetCurrentSize() const
    {
        return static_cast<u32>(m_currentAddress - m_buffer);
    }

    u32 BinarySerializer::_GetRemainingSize() const
    {
        return m_size - _GetCurrentSize();
    }

    u32 BinarySerializer::_ComputeHashTag(const char* _tag, const bool _closingSection/*= false*/) const
    {
        //retrieve the last section
        const Section* section = nullptr;

        if (_closingSection)
        {
            if (m_sectionsStack.size() > 2)
            {
                section = &m_sectionsStack.at(m_sectionsStack.size() - 2);
            }
        }
        else
        {
            if (!m_sectionsStack.empty())
            {
                section = &m_sectionsStack.back();
            }
        }

        // for array, tag is equal to arrayIndex
        if (section && section->isArray)
        {
            return section->arrayIndex;
        }

        return _tag ? StringHash(_tag).GetValue() : 0;
    }


    // ----- Basic types serialization -------------------------------------------------

    template <typename T>
    u32 BinarySerializer::_SizeOf(const T& _data)
    {
        return static_cast<u32>(sizeof(T));
    }

    template <typename T>
    SerializeError BinarySerializer::_Read(T& _data)
    {
        const u32 dataSize = sizeof(T);
        if (_EnsureIsReadable(dataSize))
        {
            memcpy(&_data, m_currentAddress, dataSize);
            m_currentAddress += dataSize;
            return SerializeError::None;
        }

        return SerializeError::CannotRead;
    }

    template <typename T>
    SerializeError BinarySerializer::_Write(T& _data)
    {
        const u32 dataSize = sizeof(T);
        if (_EnsureIsWritable(dataSize))
        {
            memcpy(m_currentAddress, &_data, dataSize);
            m_currentAddress += dataSize;
            return SerializeError::None;
        }

        return SerializeError::CannotWrite;
    }

    template <typename T>
    SerializeError BinarySerializer::_ReadArray(T* _data, const u32 _count)
    {
        const u32 dataSize = _count * sizeof(T);
        if (_EnsureIsReadable(dataSize))
        {
            memcpy(_data, m_currentAddress, dataSize);
            m_currentAddress += dataSize;
            return SerializeError::None;
        }

        return SerializeError::CannotRead;
    }

    template <typename T>
    SerializeError BinarySerializer::_WriteArray(T* _data, const u32 _count)
    {
        const u32 dataSize = _count * sizeof(T);
        if (_EnsureIsWritable(dataSize))
        {
            memcpy(m_currentAddress, _data, dataSize);
            m_currentAddress += dataSize;
            return SerializeError::None;
        }

        return SerializeError::CannotWrite;
    }

    template <typename T>
    SerializeError BinarySerializer::_InternalSerialize(T& _data, const char* _tag)
    {
        if (IsReading())
        {
            if (m_pureBinary)
            {
                // In pure binary mode, simply read raw data.
                return _Read(_data);
            }

            const auto& currentSessionMembers = m_currentSectionMembers.back();
            const u32 hashTag =  _ComputeHashTag(_tag);

            auto memberIt = std::find_if(
                currentSessionMembers.begin(), currentSessionMembers.end(),
                [hashTag](const Member& _member)->bool { return _member.hashTag == hashTag; }
            );

            if (memberIt != currentSessionMembers.end())
            {
                // skip header
                m_currentAddress = m_buffer + memberIt->startIndex + memberIt->GetHeaderSize();
                return _Read(_data);
            }

            return SerializeError::EntryNotFound;

        }

        if (IsWriting())
        {
            if (m_pureBinary)
            {
                // In pure binary mode, simply write raw data.
                return _Write(_data);
            }

            Member* member = nullptr;
            auto &currentSessionMembers = m_currentSectionMembers.back();

            if (m_sectionsStack.back().isArray)
            {
                AVA_ASSERT(!currentSessionMembers.empty(), "[Serializer] current node should be allocated, call NextElement().");
                member = &currentSessionMembers.back();
            }
            else
            {
                member = &currentSessionMembers.emplace_back();
            }

            member->size = _SizeOf(_data);
            member->hashTag = _ComputeHashTag(_tag);
            member->pureBinary = m_pureBinary;
            member->type = _GetSerializeValueType<T>();
            member->startIndex = _GetCurrentSize();
            member->WriteHeader(*this);

            const SerializeError error = _Write(_data);
            UpdateGlobalSize();
            return error;
        }

        return SerializeError::NoSerializer;
    }

    template <typename T>
    SerializeValueType BinarySerializer::_GetSerializeValueType() const
    {
        return SerializeValueType::Null;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<bool>() const
    {
        return SerializeValueType::Bool;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<float>() const
    {
        return SerializeValueType::Float;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<double>() const
    {
        return SerializeValueType::Double;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<u8>() const
    {
        return SerializeValueType::U32;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<u16>() const
    {
        return SerializeValueType::U32;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<u32>() const
    {
        return SerializeValueType::U32;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<u64>() const
    {
        return SerializeValueType::U64;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<s8>() const
    {
        return SerializeValueType::S32;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<s16>() const
    {
        return SerializeValueType::S32;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<s32>() const
    {
        return SerializeValueType::S32;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<s64>() const
    {
        return SerializeValueType::S64;
    }

    template <>
    SerializeValueType BinarySerializer::_GetSerializeValueType<std::string>() const
    {
        return SerializeValueType::String;
    }


    // ------ Special case for std::string ---------------------------------------------

    template <>
    u32 BinarySerializer::_SizeOf(const std::string& _data)
    {
        return static_cast<u32>(_data.length() + sizeof(u32));
    }

    template <>
    SerializeError BinarySerializer::_Read(std::string& _data)
    {
        u32 length;
        if (_Read(length) == SerializeError::None)
        {
            _data.resize(length);
            return _ReadArray(_data.data(), length);
        }

        return SerializeError::CannotRead;
    }

    template <>
    SerializeError BinarySerializer::_Write(std::string& _data)
    {
        const u32 length = (u32)_data.length();
        if (_Write(length) == SerializeError::None)
        {
            return _WriteArray(_data.data(), length);
        }

        return SerializeError::CannotWrite;
    }

}
