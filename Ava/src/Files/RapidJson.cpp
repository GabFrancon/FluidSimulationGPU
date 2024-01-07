#include <avapch.h>
#include "RapidJson.h"

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <rapidjson/document.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

RapidJsonPoolAllocator::RapidJsonPoolAllocator(const size_t _chunkSize /*= kDefaultChunkCapacity*/, BaseAllocator* _baseAllocator /*= nullptr*/) :
    m_chunkHead(nullptr), m_chunkCapacity(_chunkSize), m_userBuffer(nullptr), m_baseAllocator(_baseAllocator), m_ownBaseAllocator(nullptr)
{

}

RapidJsonPoolAllocator::RapidJsonPoolAllocator(void *_buffer, const size_t _size, const size_t _chunkSize /*= kDefaultChunkCapacity*/, BaseAllocator* _baseAllocator /*= nullptr*/) :
    m_chunkHead(nullptr), m_chunkCapacity(_chunkSize), m_userBuffer(_buffer), m_baseAllocator(_baseAllocator), m_ownBaseAllocator(nullptr)
{
    RAPIDJSON_ASSERT(_buffer != nullptr);
    RAPIDJSON_ASSERT(_size > sizeof(ChunkHeader));
    m_chunkHead = static_cast<ChunkHeader*>(_buffer);
    m_chunkHead->capacity = _size - sizeof(ChunkHeader);
    m_chunkHead->size = 0;
    m_chunkHead->next = nullptr;
}

bool RapidJsonPoolAllocator::AddChunk(const size_t _capacity)
{
    if (!m_baseAllocator)
    {
         m_ownBaseAllocator = m_baseAllocator = RAPIDJSON_NEW(BaseAllocator());
    }

    if (auto* chunk = static_cast<ChunkHeader*>(m_baseAllocator->Malloc(RAPIDJSON_ALIGN(sizeof(ChunkHeader)) + _capacity))) 
    {
        chunk->capacity = _capacity;
        chunk->size = 0;
        chunk->next = m_chunkHead;
        m_chunkHead = chunk;
        return true;
    }
    return false;
}

RapidJsonPoolAllocator::~RapidJsonPoolAllocator()
{
    Clear();
    RAPIDJSON_DELETE(m_ownBaseAllocator);
}

void RapidJsonPoolAllocator::Clear()
{
    while (m_chunkHead && m_chunkHead != m_userBuffer)
    {
        ChunkHeader* next = m_chunkHead->next;
        m_baseAllocator->Free(m_chunkHead);
        m_chunkHead = next;
    }
    if (m_chunkHead && m_chunkHead == m_userBuffer)
    {
        m_chunkHead->size = 0; // Clear user buffer
    }
}

size_t RapidJsonPoolAllocator::Capacity() const
{
    size_t capacity = 0;
    for (const ChunkHeader* c = m_chunkHead; c != nullptr; c = c->next)
    {
        capacity += c->capacity;
    }
    return capacity;
}

size_t RapidJsonPoolAllocator::Size() const
{
    size_t size = 0;
    for (const ChunkHeader* c = m_chunkHead; c != nullptr; c = c->next)
    {
        size += c->size;
    }
    return size;
}

void* RapidJsonPoolAllocator::Malloc(size_t _size)
{
    if (!_size)
    {
        return nullptr;
    }
    _size = RAPIDJSON_ALIGN(_size);
    if (m_chunkHead == nullptr || m_chunkHead->size + _size > m_chunkHead->capacity)
    {
        if (!AddChunk(m_chunkCapacity > _size ? m_chunkCapacity : _size))
        {
            return nullptr;
        }
    }

    void *buffer = reinterpret_cast<char *>(m_chunkHead) + RAPIDJSON_ALIGN(sizeof(ChunkHeader)) + m_chunkHead->size;
    m_chunkHead->size += _size;
    return buffer;
}

void* RapidJsonPoolAllocator::Realloc(void* _originalPtr, size_t _originalSize, size_t _newSize)
{
    if (_originalPtr == nullptr)
    {
        return Malloc(_newSize);
    }

    if (_newSize == 0)
    {
        return nullptr;
    }

    _originalSize = RAPIDJSON_ALIGN(_originalSize);
    _newSize = RAPIDJSON_ALIGN(_newSize);

    // Do not shrink if new size is smaller than original
    if (_originalSize >= _newSize)
    {
        return _originalPtr;
    }

    // Simply expand it if it is the last allocation and there is sufficient space
    if (_originalPtr == reinterpret_cast<char *>(m_chunkHead) + RAPIDJSON_ALIGN(sizeof(ChunkHeader)) + m_chunkHead->size - _originalSize) 
    {
        const size_t increment = _newSize - _originalSize;
        if (m_chunkHead->size + increment <= m_chunkHead->capacity)
        {
            m_chunkHead->size += increment;
            return _originalPtr;
        }
    }

    // Realloc process: allocate and copy memory, do not free original buffer.
    if (void* newBuffer = Malloc(_newSize)) 
    {
        if (_originalSize)
        {
            std::memcpy(newBuffer, _originalPtr, _originalSize);
        }
        return newBuffer;
    }

    return nullptr;
}