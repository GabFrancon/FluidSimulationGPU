#pragma once
/// @file JsonSerializer.h
/// @brief Serializer for JSON files, using the rapidjson library.

#include <Files/Serializer.h>
#include <Files/RapidJson.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <rapidjson/document.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    class JsonSerializer final : public Serializer
    {
    public:
        /// @brief Use JSON serializer with an existing value (not handled internally).
        JsonSerializer(RapidJsonValue* _value, RapidJsonDocument::AllocatorType* _allocator, SerializeMode _mode);
        /// @brief Use JSON serializer on an existing document (not handled internally).
        JsonSerializer(RapidJsonDocument* _value, SerializeMode _mode);
        /// @brief Use JSON serializer without existing document (created and handled internally).
        JsonSerializer(SerializeMode _mode);

        ~JsonSerializer() override;

        // ----- File ---------------------------------------------
        bool Load(const char* _path) override;
        bool Save(const char* _path) override;

        // ----- Sections -----------------------------------------
        SerializeError OpenSection(const char* _sectionName = nullptr) override;
        SerializeError CloseSection(const char* _sectionName = nullptr) override;
        SerializeError CancelSection(const char* _sectionName) override;

        // ----- Arrays -------------------------------------------
        SerializeError BeginArray(const char* _arrayName = nullptr) override;
        SerializeError EndArray(const char* _arrayName = nullptr) override;
        SerializeError NextElementInArray() override;

        u32 GetArraySize() const override;
        u32 GetCurrentArrayIndex() const override;
        SerializeValueType GetArrayElementType(u32 _index) const override;

        // ----- Children -----------------------------------------
        u32 GetChildCount() const override;
        SerializeValueType GetChildType(const char* _tag) const override;
        u32 GetChildTags(const char** _tags, u32 _tagsCount) const override;

        // ----- Recycling ----------------------------------------
        void Restart() override;
        void Reset() override;

    private:
        // ----- Rapidjson objects --------------------------------
        RapidJsonValue* m_root = nullptr;
        RapidJsonDocument::AllocatorType* m_allocator = nullptr;
        bool m_internalRoot = false;

        RapidJsonValue* m_currentNode = nullptr;
        RapidJsonValue* m_arrayRoot = nullptr;

        // ----- Object or array sections -------------------------
        struct Section
        {
            Section(const char* _tag, const bool _array = false) : name(""), isArray(_array), arrayIndex(-1)
            {
                if (_tag != nullptr)
                {
                    name = _tag;
                }
            }

            bool operator==(const Section& _other) const { return name == _other.name && isArray == _other.isArray; }

            std::string name;
            bool isArray;
            int arrayIndex;
        };

        std::vector<Section> m_sectionsStack;

        // ------ Basic types serialization -----------------------

        /// @brief Shared serialization routine to avoid copying code.
        template <typename T>
        SerializeError _InternalSerialize(T& _data, const char* _tag = nullptr);

        /// @brief Try casting rapidjson value as a specific basic type.
        template <typename T>
        static bool _GetValueAsType(const RapidJsonValue* _jsonVal, T* _val) { return false; }

        SerializeError _SerializeBasicValue(const char* _tag, bool&        _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, char&        _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, float&       _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, double&      _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, u8&          _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, u16&         _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, u32&         _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, u64&         _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, s8&          _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, s16&         _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, s32&         _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, s64&         _value) override;
        SerializeError _SerializeBasicValue(const char* _tag, std::string& _value) override;

        // ------ Generic helpers ---------------------------------

        /// @brief Converts rapidjson type to SerializeValueType.
        static SerializeValueType _GetSerializeValueType(const RapidJsonValue* _value);

        /// @brief Internal getter with some check.
        static RapidJsonValue& _GetValue(const char* _tag, RapidJsonValue& _root);

        /// @brief Recomputes current root (used when dereferencing current root).
        void _RecomputeCurrentRoot();

        /// @brief Creates a document if none is supplied.
        void _EnsureRoot();
    };

}