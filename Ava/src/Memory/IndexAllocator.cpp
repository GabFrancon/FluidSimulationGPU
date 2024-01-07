#include <avapch.h>
#include "IndexAllocator.h"

#include <Debug/Assert.h>

namespace Ava {

    void IndexAllocator::Init(const int _indexCount, const int _deallocWaitingFrames)
    {
        AVA_ASSERT(m_maxAllocatedIndex == 0);
        m_maxIndex = _indexCount;

        AVA_ASSERT(_deallocWaitingFrames >= 1);
        m_waitingLists.resize(_deallocWaitingFrames);
        m_waitingDealloc = &m_waitingLists.front();
        m_oldestWaitingDealloc = &m_waitingLists.back();
    }

    void IndexAllocator::Shutdown()
    {
        m_maxIndex = 0;
        m_maxAllocatedIndex = 0;
        m_freeList.clear();
        m_waitingLists.clear();
        m_waitingDealloc = nullptr;
        m_oldestWaitingDealloc = nullptr;
    }

    void IndexAllocator::NextFrame()
    {
        // Move the oldest waiting indices to the free list
        m_freeList.insert(m_freeList.end(), m_oldestWaitingDealloc->begin(), m_oldestWaitingDealloc->end());
        m_oldestWaitingDealloc->clear();

        // And update the current waiting list
        if (m_waitingDealloc == &m_waitingLists.back())
        {
            m_waitingDealloc = &m_waitingLists[0];
        }
        else
        {
            m_waitingDealloc++;
        }

        // And update the oldest waiting list
        if (m_oldestWaitingDealloc == &m_waitingLists.back())
        {
            m_oldestWaitingDealloc = &m_waitingLists[0];
        }
        else
        {
            m_oldestWaitingDealloc++;
        }
    }

    int IndexAllocator::Allocate()
    {
        int index;
        Allocate(1, &index);
        return index;
    }

    void IndexAllocator::Deallocate(const int _index)
    {
        m_waitingDealloc->push_back(_index);
    }

}
