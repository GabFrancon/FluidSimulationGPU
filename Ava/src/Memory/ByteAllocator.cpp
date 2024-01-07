#include <avapch.h>
#include "ByteAllocator.h"

#include <Debug/Assert.h>

namespace Ava {

    void ByteAllocatorBase::Init(const u32 _byteSize, const int _deallocWaitingFrames)
    {
        AVA_ASSERT(m_allocGranularity > 0);
        m_bytesCapacity = _byteSize;

        // Compute size in alloc units
        const u32 sizeAllocUnits = _byteSize / m_allocGranularity + (m_bytesCapacity % m_allocGranularity ? 1 : 0);

        AVA_ASSERT(m_freeByOffset.empty());
        AVA_ASSERT(m_freeBySize.empty());
        AVA_ASSERT(m_allocated.empty());

        m_freeByOffset.insert({ 0, sizeAllocUnits });
        m_freeBySize.insert({ 0, sizeAllocUnits });

        AVA_ASSERT(_deallocWaitingFrames >= 1);
        m_waitingLists.resize(_deallocWaitingFrames);
        m_waitingDealloc = &m_waitingLists[0];
    }

    void ByteAllocatorBase::Shutdown()
    {
        m_freeByOffset.clear();
        m_freeBySize.clear();
        m_allocated.clear();
        m_waitingLists.clear();
        m_waitingDealloc = nullptr;
    }

    void ByteAllocatorBase::NextFrame()
    {
        // Process the waiting de-allocations
        for (const auto offset : *m_waitingDealloc)
        {
            _ProcessDealloc(offset);
        }
        m_waitingDealloc->clear();

        // And update the current waiting list
        if (m_waitingDealloc == &m_waitingLists.back())
        {
            m_waitingDealloc = &m_waitingLists[0];
        }
        else
        {
            m_waitingDealloc++;
        }
    }

    u32 ByteAllocatorBase::Allocate(const u32 _byteSize)
    {
        AVA_ASSERT(_byteSize);

        // Compute size in alloc units
        const u32 sizeAllocUnits = _byteSize / m_allocGranularity + (_byteSize % m_allocGranularity ? 1 : 0);

        // Find a free node big enough
        const auto it = m_freeBySize.lower_bound({ 0, sizeAllocUnits });
        if (it == m_freeBySize.end())
        {
            AVA_ASSERT(false, "Not enough free space to allocate %d bytes.\n", _byteSize);
            return ~0;
        }

        const u32 allocatedOffset = it->offset;

        // Calculate what remains after allocating from the node
        Node remaining = *it;
        remaining.offset += sizeAllocUnits;
        remaining.size -= sizeAllocUnits;

        // Remove the node from both free lists
        m_freeByOffset.erase(*it);
        m_freeBySize.erase(*it);

        if (remaining.size > 0)
        {
            // The node was bigger than what we needed
            // Insert a free node back
            m_freeByOffset.insert(remaining).second;
            m_freeBySize.insert(remaining).second;
        }

        // Add a node in the allocated list
        m_allocated.insert({ allocatedOffset, sizeAllocUnits });

        return allocatedOffset;
    }

    void ByteAllocatorBase::Deallocate(const u32 _offset)
    {
        AVA_ASSERT(m_allocated.find({ _offset, 0 }) != m_allocated.end(),
            "[Memory] bad deallocation, offset was not allocated.");

        m_waitingDealloc->push_back(_offset);
    }

    u32 ByteAllocatorBase::Size() const
    {
        int byteSize = m_bytesCapacity;
        for (const auto& node : m_freeBySize)
        {
            byteSize -= node.size * m_allocGranularity;
        }

        return byteSize;
    }

    u32 ByteAllocatorBase::Capacity() const
    {
        return m_bytesCapacity;
    }

    void ByteAllocatorBase::_ProcessDealloc(const u32 _offset)
    {
        const auto itAlloc = m_allocated.find({ _offset, 0 });
        if (itAlloc == m_allocated.end())
        {
             // Deallocate() already raised an assert at this point.
            return;
        }

        Node alloc = *itAlloc;
        m_allocated.erase(itAlloc);

        auto it = m_freeByOffset.lower_bound({ _offset, 0 });

        // Try to merge with next free node
        if (it != m_freeByOffset.end() && it->offset == alloc.offset + alloc.size)
        {
            alloc.size += it->size;
            m_freeBySize.erase(*it);
            it = m_freeByOffset.erase(it);
        }

        // Try to merge with previous free node
        if (it != m_freeByOffset.begin())
        {
            --it;
            if (it->offset + it->size == alloc.offset)
            {
                alloc.offset -= it->size;
                alloc.size += it->size;
                m_freeBySize.erase(*it);
                m_freeByOffset.erase(it);
            }
        }

        m_freeByOffset.insert(alloc).second;
        m_freeBySize.insert(alloc).second;
    }

}