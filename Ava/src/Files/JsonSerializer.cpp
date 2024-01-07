#include <avapch.h>
#include "JsonSerializer.h"

#include <Files/File.h>
#include <Debug/Log.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/error/en.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    JsonSerializer::JsonSerializer(RapidJsonValue* _value, RapidJsonDocument::AllocatorType* _allocator, const SerializeMode _mode)
        : Serializer(_mode), m_root(_value), m_allocator(_allocator), m_currentNode(_value)
    {
        AVA_ASSERT(_value == nullptr || _allocator != nullptr || _mode == SerializeMode::Read,
            "[Serializer] specify a rapidjson allocator to use JsonSerializer in write mode.");

        _EnsureRoot();
    }

    JsonSerializer::JsonSerializer(RapidJsonDocument* _value, const SerializeMode _mode)
        : JsonSerializer(_value, _value ? &_value->GetAllocator() : nullptr, _mode)
    {
    }

    JsonSerializer::JsonSerializer(const SerializeMode _mode)
        : JsonSerializer(nullptr, nullptr, _mode)
    {
    }

    JsonSerializer::~JsonSerializer()
    {
        Reset();
    }


    // ----- File ------------------------------------------------------------------------------------

    bool JsonSerializer::Load(const char* _path)
    {
        File jsonFile;
        if (!jsonFile.Open(_path, AVA_FILE_READ))
        {
            AVA_CORE_ERROR(jsonFile.GetErrorStr());
            return false;
        }

        const u32 fileSize = jsonFile.GetSize();

        std::vector<char> content(fileSize);
        jsonFile.Read(content.data(), fileSize);

        auto* doc = static_cast<RapidJsonDocument*>(m_root);
        static constexpr int kJsonParseFlags = rapidjson::kParseCommentsFlag | rapidjson::kParseTrailingCommasFlag | rapidjson::kParseStopWhenDoneFlag;
        const bool error = doc->Parse<kJsonParseFlags>(content.data(), fileSize).HasParseError();

        if (error)
        {
            const auto errorCode = doc->GetParseError();
            AVA_CORE_ERROR("[Serializer] JSON parse error : %s.", GetParseError_En(errorCode));
        }

        jsonFile.Close();
        return !error;
    }

    bool JsonSerializer::Save(const char* _path)
    {
        File jsonFile;
        if (!jsonFile.Open(_path, AVA_FILE_WRITE))
        {
            return false;
        }

        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter writer(buffer);
        const bool result = m_root->Accept(writer);

        if (result)
        {
            jsonFile.Write(buffer.GetString(), (u32)buffer.GetLength());
        }

        jsonFile.Close();
        return result;
    }


    // ----- Sections --------------------------------------------------------------------------------

    SerializeError JsonSerializer::OpenSection(const char* _sectionName)
    {
        m_sectionsStack.push_back(Section(_sectionName));

        if (!_sectionName)
        {
            // Rapidjson starts with an object as root value. If you try to open this object, do nothing.
            if (m_currentNode == m_root)
            {
                return SerializeError::None;
            }

            if (m_arrayRoot)
            {
                AVA_ASSERT(_sectionName == nullptr, "[Serialize] section added to an array should never have a tag.");

                if (IsWriting())
                {
                    AVA_ASSERT(m_currentNode->IsNull(), "[Serializer] current root should be a null value.");
                    m_currentNode->SetObject();
                }

                m_arrayRoot = nullptr;
                return SerializeError::None;
            }
        }

        if (m_currentNode->IsObject())
        {
            if (!m_currentNode->HasMember(_sectionName))
            {
                if (IsWriting())
                {
                    RapidJsonValue key(_sectionName, *m_allocator);
                    RapidJsonValue obj;
                    obj.SetObject();
                    m_currentNode->AddMember(key, obj, *m_allocator);
                }
                else
                {
                    return SerializeError::EntryNotFound;
                }
            }

            m_currentNode = &(*m_currentNode)[_sectionName];
            m_arrayRoot = nullptr;
            return SerializeError::None;
        }

        return SerializeError::SectionError;
    }

    SerializeError JsonSerializer::CloseSection(const char* _sectionName)
    {
        AVA_ASSERT(m_sectionsStack.back() == Section(_sectionName),
            "[Serializer] attempting to close invalid '%s' section.", _sectionName);

        m_sectionsStack.pop_back();
        _RecomputeCurrentRoot();
        return SerializeError::None;
    }

    SerializeError JsonSerializer::CancelSection(const char* _sectionName)
    {
        if (m_sectionsStack.back() == Section(_sectionName))
        {
            m_sectionsStack.pop_back();
        }

        // section is an array
        if (!_sectionName)
        {
            // can't remove root
            if (m_currentNode == m_root)
            {
                return SerializeError::SectionError;
            }
        }

        if (AVA_VERIFY(m_currentNode->IsObject()))
        {
            if (m_currentNode->HasMember(_sectionName))
            {
                m_currentNode->RemoveMember(_sectionName);
            }
            else
            {
                AVA_ASSERT(false, "[Serializer] attempting to cancel invalid '%s' section.", _sectionName);
                return SerializeError::EntryNotFound;
            }
        }

        _RecomputeCurrentRoot();
        return SerializeError::None;
    }


    // ----- Arrays --------------------------------------------------------------------------------

    SerializeError JsonSerializer::BeginArray(const char* _arrayName)
    {
        const int arrayIndex = m_sectionsStack.empty() ? 0 : m_sectionsStack.back().arrayIndex;
        m_sectionsStack.push_back(Section(_arrayName, true));

        if (m_currentNode->IsObject())
        {
            // tag should not be null when member of an object, except for root object
            AVA_ASSERT(_arrayName || m_sectionsStack.size() == 1, "[Serializer] arrays should always have a tag.");

            if (!m_currentNode->HasMember(_arrayName ? _arrayName : ""))
            {
                if (IsWriting())
                {
                    RapidJsonValue array;
                    array.SetArray();

                    RapidJsonValue key(_arrayName, *m_allocator);
                    m_currentNode->AddMember(key, array, *m_allocator);
                }
                else
                {
                    return SerializeError::EntryNotFound;
                }
            }

            const auto node = &(*m_currentNode)[_arrayName];

            if (!node->IsArray())
            {
                return SerializeError::EntryNotFound;
            }

            m_currentNode = node;
            m_arrayRoot = m_currentNode;
            return SerializeError::None;
        }

        if (m_arrayRoot)
        {
            AVA_ASSERT(_arrayName == nullptr, "[Serializer] nested arrays should never have a tag.");
            const auto node = &(m_arrayRoot->GetArray()[arrayIndex]);

            if (IsWriting())
            {
                AVA_ASSERT(node->IsNull(), "[Serializer] current node should be null.");
                node->SetArray();
            }
            if (!node->IsArray())
            {
                return SerializeError::EntryNotFound;
            }

            m_currentNode = node;
            m_arrayRoot = m_currentNode;
            return SerializeError::None;
        }

        return SerializeError::Corrupted;
    }

    SerializeError JsonSerializer::EndArray(const char* _arrayName)
    {
        AVA_ASSERT(m_sectionsStack.back() == Section(_arrayName, true),
            "[Serializer] attempting to close invalid '%s' array.", _arrayName);

        m_sectionsStack.pop_back();
        _RecomputeCurrentRoot();
        return SerializeError::None;
    }

    SerializeError JsonSerializer::NextElementInArray()
    {
        AVA_ASSERT(m_arrayRoot != nullptr && m_arrayRoot->IsArray() , 
            "[Serializer] NextElementInArray() should not be called outside array scope.");

        const int arrayIndex = ++(m_sectionsStack.back().arrayIndex);

        if (arrayIndex >= (int)m_arrayRoot->GetArray().Size())
        {
            if (IsWriting())
            {
                RapidJsonValue array;
                array.SetNull();

                m_arrayRoot->GetArray().PushBack(array, *m_allocator);
                AVA_ASSERT(arrayIndex == static_cast<int>(m_arrayRoot->GetArray().Size() - 1));
            }
            else
            {
                return SerializeError::NoMoreElementInList;
            }
        }

        m_currentNode = &(m_arrayRoot->GetArray()[arrayIndex]);
        return SerializeError::None;
    }

    u32 JsonSerializer::GetCurrentArrayIndex() const
    {
        AVA_ASSERT(m_arrayRoot != nullptr && m_arrayRoot->IsArray() , 
            "[Serializer] GetCurrentArrayIndex() should not be called outside array scope.");

        return m_sectionsStack.back().arrayIndex;
    }

    SerializeValueType JsonSerializer::GetArrayElementType(const u32 _index) const
    {
        AVA_ASSERT(m_arrayRoot != nullptr && m_arrayRoot->IsArray() , 
            "[Serializer] GetArrayElementType() should not be called outside array scope.");

        return _GetSerializeValueType(&m_arrayRoot->GetArray()[_index]);
    }

    u32 JsonSerializer::GetArraySize() const
    {
        AVA_ASSERT(m_arrayRoot != nullptr && m_arrayRoot->IsArray() , 
            "[Serializer] GetArraySize() should not be called outside array scope.");

        return m_arrayRoot->GetArray().Size();
    }


    // ----- Children --------------------------------------------------------------------------------

    SerializeValueType JsonSerializer::GetChildType(const char* _tag) const
    {
        if (m_currentNode && m_currentNode->IsObject()  && m_currentNode->HasMember(_tag))
        {
             return _GetSerializeValueType(&(*m_currentNode)[_tag]);
        }
        return SerializeValueType::Null;
    }

    u32 JsonSerializer::GetChildCount() const
    {
        if (m_currentNode && m_currentNode->IsObject())
        {
            return m_currentNode->MemberCount();
        }
        return 0;
    }

    u32 JsonSerializer::GetChildTags(const char** _tags, const u32 _tagsCount) const
    {
        u32 index = 0;
        if (m_currentNode && m_currentNode->IsObject())
        {
            for (auto member = m_currentNode->MemberBegin(); member != m_currentNode->MemberEnd(); ++member)
            {
                if (index == _tagsCount)
                {
                    break;
                }
                _tags[index] = member->name.GetString();
                index++;
            }
        }
        return index;
    }


    // ----- Recycling -------------------------------------------------------------------------------

    void JsonSerializer::Restart()
    {
        m_currentNode = m_root;
        m_arrayRoot = nullptr;
        m_sectionsStack.clear();
    }

    void JsonSerializer::Reset()
    {
        if (m_internalRoot)
        {
            delete m_root;
        }

        m_root = nullptr;
        m_allocator = nullptr;
        m_currentNode = nullptr;
        m_arrayRoot = nullptr;
        m_sectionsStack.clear();

        _EnsureRoot();
    }


    // ----- Generic helpers -------------------------------------------------------------------------

    SerializeValueType JsonSerializer::_GetSerializeValueType(const RapidJsonValue* _value)
    {
        if (!_value || _value->IsNull())
        {
            return SerializeValueType::Null;
        }

        switch (_value->GetType())
        {
            case rapidjson::Type::kStringType:
                return SerializeValueType::String;

            case rapidjson::Type::kObjectType:
                return SerializeValueType::Object;

            case rapidjson::Type::kArrayType:
            case rapidjson::Type::kNullType:
                return SerializeValueType::Array;

            case rapidjson::Type::kFalseType:
            case rapidjson::Type::kTrueType:
                return SerializeValueType::Bool;

            case rapidjson::Type::kNumberType:
            {
                if(_value->IsFloat())
                    return SerializeValueType::Float;
                if(_value->IsDouble())
                    return SerializeValueType::Double;
                if(_value->IsUint())
                    return SerializeValueType::U32;
                if(_value->IsUint64())
                    return SerializeValueType::U64;
                if(_value->IsInt())
                    return SerializeValueType::S32;
                if(_value->IsInt64())
                    return SerializeValueType::S64;
            }
        }
        return SerializeValueType::Null;
    }

    RapidJsonValue& JsonSerializer::_GetValue(const char* _tag, RapidJsonValue& _root)
    {
        if (!_tag)
        {
            return _root;
        }

        if (_root.IsNull() || !_root.IsObject() || !_root.HasMember(_tag))
        {
            static RapidJsonValue nullValue;
            return nullValue;
        }

        return _root[_tag];
    }

    void JsonSerializer::_RecomputeCurrentRoot()
    {
        m_currentNode = m_root;
        m_arrayRoot = nullptr;
        const u32 size = (u32)m_sectionsStack.size();

        for(u32 id = 0; id < size ; ++id)
        {
            m_arrayRoot = nullptr;
            Section& section = m_sectionsStack[id];

            if(section.name.length() > 0)
            {
                AVA_ASSERT(m_currentNode->HasMember(section.name.c_str()));
                m_currentNode = &(*m_currentNode)[section.name.c_str()];

            }

            if(section.isArray)
            {
                AVA_ASSERT(m_currentNode->IsArray(),
                    "[Serializer] can't retrieve valid array root, you may have started a member array with an invalid tag.");

                m_arrayRoot = m_currentNode;

                if( m_arrayRoot->GetArray().Size() > 0)
                {
                    const u16 index = section.arrayIndex > 0 ? section.arrayIndex : 0;
                    m_currentNode = & m_arrayRoot->GetArray()[index];
                }
            }
        }
    }

    void JsonSerializer::_EnsureRoot()
    {
        if (!m_root)
        {
            const auto document = new RapidJsonDocument();
            m_allocator = &document->GetAllocator();
            m_currentNode = document;
            m_internalRoot = true;
            m_root = document;
        }

        if (m_root->IsNull())
        {
            m_currentNode->SetObject();
        }
    }


    // ----- Basic types serialization ---------------------------------------------------------------

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, bool* _v)
    {
        if (_value->IsBool())
        {
            *_v = _value->GetBool();
            return true;
        }
        return false;
    }

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, float* _v)
    {
        if ( _value->IsLosslessFloat() || _value->IsFloat() )
        {
            *_v =  _value->GetFloat();
            return true;
        }
        return false;
    }

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, double* _v)
    {
        if (_value->IsLosslessDouble() || _value->IsDouble())
        {
            *_v = _value->GetDouble();
            return true;
        }
        return false;
    }

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, s8* _v)
    {
        if (_value->IsInt())
        {
            *_v =  _value->GetInt();
            return true;
        }
        return false;
    }

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, s16* _v)
    {
        if (_value->IsInt())
        {
            *_v = _value->GetInt();
            return true;
        }
        return false;
    }

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, s32* _v)
    {
        if (_value->IsInt())
        {
            *_v = _value->GetInt();
            return true;
        }
        return false;
    }

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, s64* _v)
    {
        if (_value->IsInt64())
        {
            *_v = _value->GetInt64();
            return true;
        }
        return false;
    }

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, u8* _v)
    {
        if (_value->IsUint())
        {
            *_v = _value->GetUint();
            return true;
        }
        return false;
    }

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, u16* _v)
    {
        if (_value->IsUint())
        {
            *_v = _value->GetUint();
            return true;
        }
        return false;
    }

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, u32* _v)
    {
        if (_value->IsUint())
        {
            *_v = _value->GetUint();
            return true;
        }
        return false;
    }

    template <> bool JsonSerializer::_GetValueAsType(const RapidJsonValue* _value, u64* _v)
    {
        if (_value->IsUint64())
        {
            *_v = _value->GetUint64();
            return true;
        }
        return false;
    }

    template <typename T>
    SerializeError JsonSerializer::_InternalSerialize(T& _data, const char* _tag)
    {
        RapidJsonValue& root = *m_currentNode;
        AVA_ASSERT(_tag == nullptr || std::strlen(_tag) > 0, "[Serializer] empty tags are not valid.");

        if (IsReading())
        {
            const RapidJsonValue& val = _GetValue(_tag, root);
            if (!val.IsNull())
            {
                if (!_GetValueAsType<T>(&val, &_data))
                {
                    return SerializeError::Corrupted;
                }
                return SerializeError::None;
            }

            return SerializeError::EntryNotFound;
        }
        if (IsWriting())
        {
            // if this is an object
            if (root.IsObject())
            {
                AVA_ASSERT(_tag != nullptr, "[Serializer] tags are mandatory outside arrays.");
                if (root.HasMember(_tag))
                {
                    root[_tag] = _data;
                }
                else
                {
                    RapidJsonValue key(_tag, *m_allocator);
                    root.AddMember(key, _data, *m_allocator);
                }
                return SerializeError::None;
            }

            // if we are in an array
            if (m_arrayRoot)
            {
                AVA_ASSERT(_tag == nullptr, "[Serializer] member added to an array should never have a tag.");
                AVA_ASSERT(root.IsNull(), "[Serializer] current root should be a null value.");

                // root is the current element in array
                root = _data;
                return SerializeError::None;
            }
            return SerializeError::EntryNotFound;
        }

        return SerializeError::NoSerializer;
    }

    template SerializeError JsonSerializer::_InternalSerialize(bool& _data, const char* _tag);
    template SerializeError JsonSerializer::_InternalSerialize(float& _data, const char* _tag);
    template SerializeError JsonSerializer::_InternalSerialize(double& _data, const char* _tag);
    template SerializeError JsonSerializer::_InternalSerialize(s8& _data, const char* _tag);
    template SerializeError JsonSerializer::_InternalSerialize(s16& _data, const char* _tag);
    template SerializeError JsonSerializer::_InternalSerialize(s32& _data, const char* _tag);
    template SerializeError JsonSerializer::_InternalSerialize(u8& _data, const char* _tag);
    template SerializeError JsonSerializer::_InternalSerialize(u16& _data, const char* _tag);
    template SerializeError JsonSerializer::_InternalSerialize(u32& _data, const char* _tag);

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, bool& _value)
    {
        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, char& _value)
    {
        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, float& _value)
    {
        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, double& _value)
    {
        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, u8& _value)
    {
        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, u16& _value)
    {
        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, u32& _value)
    {
        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, u64& _value)
    {
        RapidJsonValue& root = *m_currentNode;
        AVA_ASSERT(_tag == nullptr || std::strlen(_tag) > 0, "[Serializer] empty tags are not valid.");
        AVA_ASSERT(_tag != nullptr || m_arrayRoot != nullptr, "[Serializer] tags are mandatory outside arrays.");

        if (IsReading())
        {
            const RapidJsonValue& val = _GetValue(_tag, root);
            if (val.IsArray())
            {
                u64 lp = 0, rp = 0;
                lp = val.GetArray()[0].GetUint();
                rp = val.GetArray()[1].GetUint();

                _value = (lp << 32) + rp;
                return SerializeError::None;
            }
            if (val.IsNumber())
            {
                return _InternalSerialize(_value, _tag);
            }

            return SerializeError::EntryNotFound;
        }

        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, s8& _value)
    {
        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, s16& _value)
    {
        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, s32& _value)
    {
        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, s64& _value)
    {
        RapidJsonValue& root = *m_currentNode;
        AVA_ASSERT(_tag == nullptr || std::strlen(_tag) > 0, "[Serializer] empty tags are not valid.");
        AVA_ASSERT(_tag != nullptr || m_arrayRoot != nullptr, "[Serializer] tags are mandatory outside arrays.");

        if (IsReading())
        {
            const RapidJsonValue& val = _GetValue(_tag, root);
            if (val.IsArray())
            {
                u64 lp = 0, rp = 0;
                lp = val.GetArray()[0].GetUint();
                rp = val.GetArray()[1].GetUint();

                _value = (lp << 32) + rp;
                return SerializeError::None;
            }

            if (val.IsNumber())
            {
                return _InternalSerialize(_value, _tag);
            }

            return SerializeError::EntryNotFound;
        }

        return _InternalSerialize(_value, _tag);
    }

    SerializeError JsonSerializer::_SerializeBasicValue(const char* _tag, std::string& _value)
    {
        RapidJsonValue& root = *m_currentNode;
        AVA_ASSERT(_tag == nullptr || std::strlen(_tag) > 0, "[Serializer] empty tags are not valid.");
        AVA_ASSERT(_tag != nullptr || m_arrayRoot != nullptr, "[Serializer] tags are mandatory outside arrays.");

        if (IsReading())
        {
            const RapidJsonValue& val = _GetValue(_tag, root);

            if (val.IsString())
            {
                _value = val.GetString();
                return SerializeError::None;
            }

            if (val.IsObject())
            {
                rapidjson::StringBuffer sb;
                rapidjson::PrettyWriter writer(sb);
                val.Accept(writer);
                _value = sb.GetString();
                return SerializeError::None;
            }

            return SerializeError::EntryNotFound;
        }

        if (IsWriting())
        {
            if (!_tag)
            {
                root.SetString(_value.c_str(), (rapidjson::SizeType)_value.size(), *m_allocator);
                return SerializeError::None;
            }

            if (root.HasMember(_tag))
            {
                root[_tag].SetString(_value.c_str(), (rapidjson::SizeType)_value.size(), *m_allocator);
                return SerializeError::None;
            }

            RapidJsonValue key(_tag, *m_allocator);
            RapidJsonValue str(_value.c_str(), *m_allocator);
            root.AddMember(key, str, *m_allocator);
            return SerializeError::None;
        }

        return SerializeError::NoSerializer;
    }

}
