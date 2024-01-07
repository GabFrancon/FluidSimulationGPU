#pragma once
/// @file BinarySerializer.h
/// @brief Serializer for binary files, used either with a header or as a purely ordered serializer.


#include <Files/Serializer.h>

namespace Ava {

    class BinarySerializer final : public Serializer
    {
    public:
        /// @brief Use binary serializer without existing buffer (created and handled internally).
        BinarySerializer(SerializeMode _mode, bool _pureBinary = false);
        /// @brief Use binary serializer on an existing buffer (not handled internally).
        BinarySerializer(u8* _buffer, u32 _size, SerializeMode _mode, bool _pureBinary = false);

        ~BinarySerializer() override;

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

        // ----- Optimized serialization --------------------------
        SerializeError SerializeBytes(const char* _tag, std::vector<char>& _bytes) override;

        // ----- Reset --------------------------------------------
        void Restart() override;
        void Reset() override;

    private:
        u32 m_size;
        u8* m_buffer;
        u8* m_currentAddress;

        bool m_pureBinary;
        bool m_internalBuffer;

        // ----- Object or array sections -------------------------

        struct Member
        {
            u32 size = 0;
            u32 startIndex = 0;
            u32 hashTag = 0;
            SerializeValueType type = SerializeValueType::Null;
            bool pureBinary = false;

            u32 GetHeaderSize() const;
            SerializeError ReadHeader(BinarySerializer& _s);
            SerializeError WriteHeader(BinarySerializer& _s);
        };

        struct Section
        {
            bool isArray = false;
            int arrayIndex = -1;
            u32 arrayCount = 0;
            Member member;
        };

        std::vector<Section> m_sectionsStack;
        std::vector<std::vector<Member>> m_currentSectionMembers;

        // ------ Basic types serialization -----------------------

        /// @brief Returns the size in bytes of type T.
        template <typename T>
        u32 _SizeOf(const T& _data);

        /// @brief Reads the current member as an element of type T.
        template <typename T>
        SerializeError _Read(T& _data);

        /// @brief Writes the current member as an element of type T.
        template <typename T>
        SerializeError _Write(T& _data);

        /// @brief Reads the current member as an array of type T.
        template <typename T>
        SerializeError _ReadArray(T* _data, u32 _count);

        /// @brief Write the current member as an array of type T.
        template <typename T>
        SerializeError _WriteArray(T* _data, u32 _count);

        /// @brief Shared serialization routine to avoid copying code.
        template <typename T>
        SerializeError _InternalSerialize(T& _data, const char* _tag = nullptr);

        /// @brief Returns SerializeValueType enum corresponding to template type.
        template <typename T>
        SerializeValueType _GetSerializeValueType() const;

        SerializeError _SerializeBasicValue(const char* _tag, bool&        _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, char&        _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, float&       _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, double&      _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, u8&          _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, u16&         _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, u32&         _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, u64&         _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, s8&          _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, s16&         _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, s32&         _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, s64&         _value) override { return _InternalSerialize(_value, _tag); }
        SerializeError _SerializeBasicValue(const char* _tag, std::string& _value) override { return _InternalSerialize(_value, _tag); }

        // ------ Generic helpers ---------------------------------
        void _ComputeRootMember();
        bool _RecomputeMembers();

        bool _EnsureIsReadable(u32 _size) const;
        bool _EnsureIsWritable(u32 _size);

        void _SetBuffer(u8* _buffer, u32 _size);
        void _CopyBuffer(const u8* _buffer, u32 _size);
        bool _ResizeBuffer(u32 _size);
        void _ReleaseBuffer();

        void UpdateGlobalSize();
        void _UpdateCurrentSectionSize();

        u32 _GetCurrentSize() const;
        u32 _GetRemainingSize() const;

        u32 _ComputeHashTag(const char* _tag, bool _closingSection = false) const;
    };

}
