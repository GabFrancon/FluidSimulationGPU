#pragma once
/// @file Serializer.h
/// @brief

#include <Memory/Memory.h>
#include <Debug/Assert.h>

// Ava basic primitives
#include <Math/Types.h>
#include <Graphics/Color.h>
#include <Strings/StringHash.h>

namespace Ava {

    /// @brief Serialization modes enum.
    enum class SerializeMode
    {
        None,
        Read,
        Write
    };

    /// @brief Serialization value types enum.
    enum class SerializeValueType
    {
        Unknown,
        Null,
        Bool,
        U32,
        U64,
        S32,
        S64,
        Float,
        Double,
        String,
        Object,
        Array,
    };

    /// @brief Serialization error types enum.
    enum class SerializeError
    {
        None,
        Partial,
        EntryNotFound,
        CannotRead,
        CannotWrite,
        Corrupted,
        SectionError,
        ArrayError,
        NoMoreElementInList,
        NoSerializer,

        Count
    };

    /// @brief Serializer interface, to read from and write to files.
    class Serializer
    {
    public:
        Serializer(const SerializeMode _mode) : m_mode(_mode) {}
        virtual ~Serializer() = default;

        Serializer(Serializer&&) = delete;
        Serializer(const Serializer&) = delete;
        Serializer& operator=(Serializer&&) = delete;
        Serializer& operator=(const Serializer&) = delete;

        // ----- File ---------------------------------------------
        virtual bool Load(const char* _path) { return false; }
        virtual bool Save(const char* _path) { return false; }

        // ----- Core serialization functions ---------------------
        template <typename T>
        SerializeError Serialize(const char* _tag, T& _value);

        template <typename T>
        SerializeError SerializeArray(const char* _tag, T** _data, u32* _size, bool _allocate = false);

        // ----- Serialization template helpers -------------------
        template <typename T>
        SerializeError Serialize(const char* _tag, const T& _value) { return Serialize<T>(_tag, const_cast<T&>(_value)); }

        template <typename T>
        SerializeError Serialize(T& _value) { return Serialize<T>(nullptr, _value); }

        template <typename T>
        SerializeError Serialize(const T& _value) { return Serialize<T>(nullptr, _value); }

        template <typename T>
        SerializeError SerializeArray(const char* _tag, T* _data, u32 _size) { return SerializeArray<T>(_tag, &_data, &_size); }

        // ----- Sections -----------------------------------------
        virtual SerializeError OpenSection(const char* _sectionName = nullptr) = 0;
        virtual SerializeError CloseSection(const char* _sectionName = nullptr) = 0;
        virtual SerializeError CancelSection(const char* _sectionName = nullptr) = 0;

        // ----- Arrays -------------------------------------------
        virtual SerializeError BeginArray(const char* _arrayName = nullptr) = 0;
        virtual SerializeError EndArray(const char* _arrayName = nullptr) = 0;
        virtual SerializeError NextElementInArray() = 0;

        virtual u32 GetArraySize() const = 0;
        virtual u32 GetCurrentArrayIndex() const = 0;
        virtual SerializeValueType GetArrayElementType(u32 _index) const = 0;
        SerializeValueType GetCurrentArrayElementType() const { return GetArrayElementType(GetCurrentArrayIndex()); }

        // ----- Children -----------------------------------------
        virtual u32 GetChildCount() const = 0;
        virtual SerializeValueType GetChildType(const char* _tag) const = 0;
        virtual u32 GetChildTags(const char** _tags, u32 _tagsCount) const = 0;

        // ----- Mode ---------------------------------------------
        SerializeMode GetMode() const { return m_mode; }
        bool IsReading() const { return m_mode == SerializeMode::Read; }
        bool IsWriting() const { return m_mode == SerializeMode::Write; }

        // ----- Errors -------------------------------------------
        static const char* GetErrorStr(const SerializeError _error)
        {
            static const char* kSerializeErrorStr[(int)SerializeError::Count] = {
                "", "Partial error", "Entry no found", "Cannot read", "Cannot write", "Corrupted",
                "Section error", "Array error", "No more element in list", "No serializer"
                };
            return kSerializeErrorStr[_error < SerializeError::Count ? (int)_error : 0];
        }

        // ----- Optimized serialization --------------------------

        /// @brief Serializes an array of bytes. This common case can be optimized in the derived classes.
        virtual SerializeError SerializeBytes(const char* _tag, std::vector<char>& _bytes) { return Serialize(_tag, _bytes); }

        // ----- Recycling ----------------------------------------

        /// @brief Moves the cursor back to the json root.
        virtual void Restart() = 0;

        /// @brief Destroys the document if handled internally.
        virtual void Reset() = 0;

    protected:
        // ------ Basic types serialization -----------------------
        virtual SerializeError _SerializeBasicValue(const char* _tag, bool&        _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, char&        _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, float&       _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, double&      _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, u8&          _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, u16&         _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, u32&         _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, u64&         _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, s8&          _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, s16&         _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, s32&         _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, s64&         _value) = 0;
        virtual SerializeError _SerializeBasicValue(const char* _tag, std::string& _value) = 0;

    private:
        SerializeMode m_mode = SerializeMode::None;
    };

    /// @brief Custom serializer template to implement.
    template <typename T, typename Enable = void>
    struct custom_serializer
    {
        static SerializeError Serialize(Serializer& _serializer, const char* _tag, T& _value)
        {
            return _value.Serialize(_serializer, _tag);
        }
    };

    template <typename T>
    SerializeError Serializer::Serialize(const char* _tag, T& _value)
    {
        return custom_serializer<T>::Serialize(*this, _tag, _value);
    }

    template <typename T>
    SerializeError Serializer::SerializeArray(const char* _tag, T** _data, u32* _size, const bool _allocate)
    {
        AVA_ASSERT(_size && (*_data || *_size == 0));

        const SerializeError error = BeginArray(_tag);
        if (error != SerializeError::None)
        {
            EndArray(_tag);
            return error;
        }

        if (IsReading())
        {
            *_size = GetArraySize();
            if (_allocate)
            {
                AVA_ASSERT(!*_data, "[Serializer] buffer already allocated.");
                *_data = static_cast<T*>(AVA_MALLOC(sizeof(T) * *_size));
            }
        }

        if (*_data)
        {
            u32 index = 0;
            while (index < *_size && NextElementInArray() == SerializeError::None)
            {
                T& element = (*_data)[index];
                if (Serialize(element) != SerializeError::None)
                {
                    EndArray(_tag);
                    return SerializeError::ArrayError;
                }
                index++;
            }
        }

        return EndArray(_tag);
    }

    template <> inline SerializeError Serializer::Serialize(const char* _tag, bool& _value)        { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, char& _value)        { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, float& _value)       { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, double& _value)      { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, u8& _value)          { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, u16& _value)         { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, u32& _value)         { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, u64& _value)         { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, s8& _value)          { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, s16& _value)         { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, s32& _value)         { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, s64& _value)         { return _SerializeBasicValue(_tag, _value); }
    template <> inline SerializeError Serializer::Serialize(const char* _tag, std::string& _value) { return _SerializeBasicValue(_tag, _value); }

}


// ---- Ava Color serialization ---------------------------------------------------------------------

// Ava::Color
template <>
struct Ava::custom_serializer<Ava::Color>
{
    static SerializeError Serialize(Serializer& _serializer, const char* _tag, Color& _value)
    {
        return _serializer.SerializeArray(_tag, &_value, 4);
    }
};


// ---- Ava StringHash serialization ----------------------------------------------------------------

// Ava::StringHash
template <>
struct Ava::custom_serializer<Ava::StringHash>
{
    static SerializeError Serialize(Serializer& _serializer, const char* _tag, StringHash& _value)
    {
        auto error = SerializeError::None;

        if (_serializer.IsReading())
        {
            if (_tag && _serializer.GetChildType(_tag) == SerializeValueType::String
                || !_tag && _serializer.GetCurrentArrayElementType() == SerializeValueType::String)
            {
                std::string str;
                error = _serializer.Serialize(_tag, str);
                _value = StringHash(str);
                return error;
            }

            u32 hash;
            error = _serializer.Serialize(_tag, hash);
            _value = StringHash(hash);
            return error;
        }

        if (_serializer.IsWriting())
        {
        #if defined(AVA_STORE_STRING_HASH)
            std::string str = _value.GetString();
            error = _serializer.Serialize(_tag, str);
        #else
            u32 hash = _value.GetValue();
            error = _serializer.Serialize(_tag, hash);
        #endif

            return error;
        }

        return SerializeError::NoSerializer;
    }
};


// ---- GLM mathematical types serialization --------------------------------------------------------

// glm::vec<>
template <glm::length_t Length, typename Type, glm::qualifier Qualifier>
struct Ava::custom_serializer<glm::vec<Length, Type, Qualifier>>
{
    static SerializeError Serialize(Serializer& _serializer, const char* _tag, glm::vec<Length, Type, Qualifier>& _value)
    {
       return _serializer.SerializeArray(_tag, &_value, Length);
    }
};

// glm::mat<>
template <glm::length_t Column, glm::length_t Row, typename Type, glm::qualifier Qualifier>
struct Ava::custom_serializer<glm::mat<Column, Row, Type, Qualifier>>
{
    static SerializeError Serialize(Serializer& _serializer, const char* _tag, glm::mat<Column, Row, Type, Qualifier>& _value)
    {
        return _serializer.SerializeArray(_tag, &_value, Column * Row);
    }
};

// glm::quat<>
template <typename Type, glm::qualifier Qualifier>
struct Ava::custom_serializer<glm::qua<Type, Qualifier>>
{
    static SerializeError Serialize(Serializer& _serializer, const char* _tag, glm::qua<Type, Qualifier>& _value)
    {
        return _serializer.SerializeArray(_tag, &_value, 4);
    }
};


// ---- STL containers serialization ---------------------------------------------------------------------

// std::vector<>
template <typename T>
struct Ava::custom_serializer<std::vector<T>>
{
    static SerializeError Serialize(Serializer& _serializer, const char* _tag, std::vector<T>& _value)
    {
        SerializeError result = _serializer.BeginArray(_tag);

        if (result == SerializeError::None)
        {
            u32 size;
            if (_serializer.IsReading())
            {
                size = _serializer.GetArraySize();

                _value.clear();
                _value.resize(size);
            }
            else
            {
                size = (u32)_value.size();
            }

            u32 index = 0;
            while (index < size && _serializer.NextElementInArray() == SerializeError::None)
            {
                const SerializeError error = _serializer.Serialize(_value[index]);
                index++;

                if (error != SerializeError::None && result == SerializeError::None)
                {
                    result = error;
                }
            }
        }

        _serializer.EndArray(_tag);
        return result;
    }
};

// std::set<>
template <typename T>
struct Ava::custom_serializer<std::set<T>>
{
    static SerializeError Serialize(Serializer& _serializer, const char* _tag, std::set<T>& _value)
    {
        SerializeError result = _serializer.BeginArray(_tag);

        if (result == SerializeError::None)
        {
            u32 size;
            if (_serializer.IsReading())
            {
                size = _serializer.GetArraySize();
                _value.clear();
            }
            else
            {
                size = (u32)_value.size();
            }

            u32 index = 0;
            auto it = _value.begin();

            while (index < size && _serializer.NextElementInArray() == SerializeError::None)
            {
                auto error = SerializeError::None;

                if (_serializer.IsReading())
                {
                    T v{};
                    error = _serializer.Serialize(v);
                    _value.insert(v);
                }
                else
                {
                    error = _serializer.Serialize(*it);
                    ++it;
                }
                index++;

                if (error != SerializeError::None && result == SerializeError::None)
                {
                    result = error;
                }
            }
        }

        _serializer.EndArray(_tag);
        return result;
    }
};


// ---- Enum serialization ------------------------------------------------------------------------------

// enum
template <typename T>
struct Ava::custom_serializer<T, std::enable_if_t<std::is_enum_v<T>>>
{
    static SerializeError Serialize(Serializer& _serializer, const char* _tag, T& _value)
    {
        if (_serializer.IsReading())
        {
            int enumId;
            const auto error = _serializer.Serialize(_tag, enumId);
            _value = static_cast<T>(enumId);
            return error;
        }

        if (_serializer.IsWriting())
        {
            auto enumId = static_cast<int>(_value);
            const auto error = _serializer.Serialize(_tag, enumId);
            return error;
        }

        return SerializeError::NoSerializer;
    }
};