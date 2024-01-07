#pragma once
/// @file RapidJson.h
/// @brief Typedefs for RapidJson library.

#include <Memory/Memory.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <rapidjson/fwd.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

class RapidJsonAllocator
{
public:
    static void* Malloc(const size_t _size)
    {
        return AVA_MALLOC(_size);
    }

    static void* Realloc(void* _originalPtr, size_t _originalSize, const size_t _newSize)
    {
        return AVA_REALLOC(_originalPtr, _newSize);
    }

    static void Free(void* _ptr)
    {
        AVA_FREE(_ptr);
    }
};

class RapidJsonPoolAllocator
{
public:
    typedef RapidJsonAllocator BaseAllocator;

    /// Tell users that no need to call Free() with this allocator (concept Allocator).
    static constexpr bool kNeedFree = false;

    RapidJsonPoolAllocator(size_t _chunkSize = kDefaultChunkCapacity, BaseAllocator* _baseAllocator = nullptr);

    /// @brief Constructor with user-supplied buffer.
    /// @details The user buffer will be used firstly. When it is full, memory pool allocates new chunk with chunk size.
    /// The user buffer will not be deallocated when this allocator is destructed.
    /// @param _buffer User supplied buffer.
    /// @param _size Size of the buffer in bytes. It must at least larger than sizeof(ChunkHeader).
    /// @param _chunkSize The size of memory chunk. The default is kDefaultChunkSize.
    /// @param _baseAllocator The allocator for allocating memory chunks.
    RapidJsonPoolAllocator(void *_buffer, size_t _size, size_t _chunkSize = kDefaultChunkCapacity, BaseAllocator* _baseAllocator = nullptr);

    /// @brief This deallocates all memory chunks, excluding the user-supplied buffer.
    ~RapidJsonPoolAllocator();

    /// @brief Deallocates all memory chunks, excluding the user-supplied buffer.
    void Clear();

    /// @brief Computes the total capacity of allocated memory chunks.
    /// @return total capacity in bytes.
    size_t Capacity() const;

    /// @brief Computes the memory blocks allocated.
    /// @return total used bytes.
    size_t Size() const;

    /// @brief Allocates a memory block. (concept Allocator)
    void* Malloc(size_t _size);

    /// @brief Resizes a memory block (concept Allocator)
    void* Realloc(void* _originalPtr, size_t _originalSize, size_t _newSize);

    /// @brief Frees a memory block (concept Allocator)
    static void Free(const void* _ptr) { (void)_ptr; } // Do nothing

private:
    /// @brief Copy constructor is not permitted.
    RapidJsonPoolAllocator(const RapidJsonPoolAllocator& _other) = delete;

    /// @brief Copy assignment operator is not permitted.
    RapidJsonPoolAllocator& operator=(const RapidJsonPoolAllocator& _other) = delete;

    /// @brief Creates a new chunk.
    /// @param _capacity Capacity of the chunk in bytes.
    /// @return true if success.
    bool AddChunk(size_t _capacity);

    /// modified rapid json's chunk capacity (default is 64 * 1024).
    static constexpr int kDefaultChunkCapacity = 2 * 1024;

    struct ChunkHeader
    {
        /// Capacity of the chunk in bytes (excluding the header itself).
        size_t capacity;
        /// Current size of allocated memory in bytes.
        size_t size;
        /// Next chunk in the linked list.
        ChunkHeader *next;
    };

    /// Head of the chunk linked-list. Only the head chunk serves allocation.
    ChunkHeader* m_chunkHead;
    /// The minimum capacity of chunk when they are allocated.
    size_t m_chunkCapacity;
    /// User supplied buffer.
    void* m_userBuffer;
    /// base allocator for allocating memory chunks.
    BaseAllocator* m_baseAllocator;
    /// base allocator created by this object.
    BaseAllocator* m_ownBaseAllocator;
};

typedef rapidjson::GenericDocument<rapidjson::UTF8<char>, RapidJsonPoolAllocator, RapidJsonAllocator> RapidJsonDocument;
typedef rapidjson::GenericValue<rapidjson::UTF8<char>, RapidJsonPoolAllocator> RapidJsonValue;