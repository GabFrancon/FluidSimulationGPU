#include <avapch.h>
#include "StringHash.h"

#include <Synchro/Synchro.h>
#include <Memory/Memory.h>
#include <Debug/Assert.h>
#include <Math/Hash.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <rigtorp/HashMap.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

#if defined(AVA_STORE_STRING_HASH)

    // ------ StringBuffer -----------------------------------------------------

    class StringBuffer
    {
        typedef std::vector<char*> ChunkList;

        const u32     m_chunkCapacity;
        u32           m_lastChunkSize;
        ChunkList     m_chunks;

        void _AddChunk()
        {
            char* newChunk = new char[m_chunkCapacity];

            m_chunks.push_back(newChunk);
            m_lastChunkSize = 0;
        }

        StringBuffer(StringBuffer& _other);

    public:
        StringBuffer(const u32 _chunkCapacity)
            : m_chunkCapacity(_chunkCapacity)
            , m_lastChunkSize(_chunkCapacity) // start full so next push will add a chunk
        {

        }

        ~StringBuffer()
        {
            for (ChunkList::const_iterator it = m_chunks.begin();
                it != m_chunks.end();
                ++it)
            {
                delete[] * it;
            }
        }

        const char* PushBack(const char* _str)
        {
             // +1 for null terminator
            const u32 strMemSize = (u32)strlen(_str) + 1;
            AVA_ASSERT(strMemSize <= m_chunkCapacity);

            if (m_lastChunkSize + strMemSize > m_chunkCapacity)
            {
                _AddChunk();
            }

            const u32 size = m_lastChunkSize;
            memcpy(m_chunks.back() + size, _str, strMemSize);

            m_lastChunkSize += strMemSize;

            return &m_chunks.back()[size];
        }
    };


    // ------ StringMap ---------------------------------------------------------

    class StringHash::StringMap
    {
        struct NoHash
        {
            size_t operator() (const u32 _hash) const
            {
                return _hash;
            }
        };

        // rigtorp hash map uses linear probing which gives
        // much better performance than std::unordered_map.
        typedef rigtorp::HashMap<u32, const char*, NoHash> Map;

        Map m_map;
        mutable Mutex m_mutex;
        StringBuffer  m_stringBuffer;

    public:
        struct InsertResult
        {
            const char* str;
            bool inserted;
        };

        StringMap()
            : m_map(1024, 0)
            , m_stringBuffer(1024 * 1024)
        {
        }

        InsertResult Insert(const u32 _hash, const char* _str)
        {
            auto lock = m_mutex.MakeAutoLock();

            bool inserted = false;
            const char*& str = m_map[_hash];
            if (str == nullptr)
            {
                inserted = true;
                str = m_stringBuffer.PushBack(_str);
            }
            return{ str, inserted };
        }

        const char* Find(const u32 _hash) const
        {
            auto lock = m_mutex.MakeAutoLock();

            const char* str;
            const auto it = m_map.find(_hash);

            if (it != m_map.end())
            {
                str = it->second;
            }
            else
            {
                str = nullptr;
            }

            return str;
        }

        int Size() const
        {
            auto lock = m_mutex.MakeAutoLock();
            return (u32)m_map.size();
        }
    };


    // -------- L1 / L2 Cache ---------------------------------------------------

    template <class K, class V, int Capacity, K KeyInvalidValue>
    class Cache
    {
        size_t m_maxAge;
        size_t m_ages[Capacity];
        K      m_keys[Capacity];
        V      m_values[Capacity];

    public:
        Cache()
        {
            Reset();
        }

        ~Cache()
        {
        }

        V Find(K _key)
        {
            for (int i = 0; i < Capacity; ++i)
            {
                if (m_keys[i] == _key)
                {
                    m_ages[i] = ++m_maxAge;
                    return m_values[i];
                }
            }
            return NULL;
        }

        void Add(K _key, V _value)
        {
            int index = 0;
            size_t minAge = ~0;
            for (int i = 0; i < Capacity; ++i)
            {
                if (m_ages[i] < minAge)
                {
                    index = i;
                    minAge = m_ages[i];
                }
            }
            m_ages[index] = ++m_maxAge;
            m_keys[index] = _key;
            m_values[index] = _value;
        }

        void Reset()
        {
            m_maxAge = 0;
            for (auto& age : m_ages) age = 0;
            for (auto& key : m_keys) key = KeyInvalidValue;
        }
    };


    // -------- Thread Map ------------------------------------------------------
    class ThreadStringMap
    {
        StringHash::StringMap* m_globalStringMap;
        Cache<u32, const char*, 16, 0> m_l1Cache;
        Cache<u32, const char*, 128, 0> m_l2Cache;

    public:
        ThreadStringMap(StringHash::StringMap* _globalStringMap)
            : m_globalStringMap(_globalStringMap)
        {
        }

        /// @param _hash The hash of _str
        /// @param _str The string to insert or NULL to just do a find
        /// @return The copy of the string stored inside the map or NULL if not found (only if _str was NULL)
        const char* FindOrInsert(const u32 _hash, const char* _str)
        {
            const char* storedStr = nullptr;

            // Search in L1 cache
            storedStr =  m_l1Cache.Find(_hash);
            if (storedStr)
            {
                return storedStr;
            }

            // Search in L2 cache
            storedStr = m_l2Cache.Find(_hash);
            if (storedStr)
            {
                m_l1Cache.Add(_hash, storedStr);
                return storedStr;
            }

            // Insert the string and update the caches
            if (_str)
            {
                const auto result = m_globalStringMap->Insert(_hash, _str);
                storedStr = result.str;

                m_l1Cache.Add(_hash, storedStr);
                m_l2Cache.Add(_hash, storedStr);
            }

            // Find the string and update the caches
            else
            {
                storedStr = m_globalStringMap->Find(_hash);
                if (storedStr)
                {
                    m_l1Cache.Add(_hash, storedStr);
                    m_l2Cache.Add(_hash, storedStr);
                }
            }

            return storedStr;
        }
    };


    // -------- Data -------------------------------------------------------------

    static StringHash::StringMap& GetGlobalStringMap()
    {
        static StringHash::StringMap s_globalStringMap;
        return s_globalStringMap;
    }

    static ThreadStringMap& GetThreadStringMap()
    {
        static AVA_THREAD_LOCAL ThreadStringMap s_ThreadStringMap(&GetGlobalStringMap());
        return s_ThreadStringMap;
    }

    StringHash::StringMap* StringHash::s_globalStringMap = &GetGlobalStringMap();

#endif

    // ------ StringHash ---------------------------------------------------------

    StringHash::StringHash(const u32 _hash)
    {
        m_hash = _hash;
    }

    StringHash::StringHash(const char* _str)
    {
        m_hash = HashStr(_str);
        _UpdateString(_str);
    }

    StringHash::StringHash(const std::string& _str)
    {
        const char* str = _str.c_str();

        m_hash = HashStr(str);
        _UpdateString(str);

    }

    u32 StringHash::GetValue() const
    {
        return m_hash;
    }

    const char* StringHash::GetString() const
    {
    #if defined(AVA_STORE_STRING_HASH)
        const char* str = GetThreadStringMap().FindOrInsert(m_hash, nullptr);
        return str ? str : "";
    #else
        return "";
    #endif
    }

    void StringHash::_UpdateString(const char* _str) const
    {
    #if defined(AVA_STORE_STRING_HASH)
        if (_str == nullptr)
        {
            // Nothing to do
            return;
        }

        if (m_hash == 0)
        {
            // StringMap cannot store a zero hash (zero is used to
            // mark empty buckets in the hash map and in the caches).
            return;
        }

        // Insert the string in the hash map
        const char* foundStr = GetThreadStringMap().FindOrInsert(m_hash, _str);

        // Check hash collisions
        AVA_ASSERT(strcmp(foundStr, _str) == 0, 
            "[StringHash] collision detected between '%s' and '%s' (hash = %u).", foundStr, _str, m_hash);

    #endif
    }

    StringHash StringHash::Invalid = StringHash(0u);

}