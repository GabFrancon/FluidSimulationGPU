#pragma once
/// @file IndexAllocator.h
/// @brief

#include <Core/Base.h>
#include <Debug/Log.h>

namespace Ava {

    /// @brief Index (integer) allocator helper.
    class IndexAllocator
    {
    public:

        void Init(int _indexCount, int _deallocWaitingFrames);
        void Shutdown();
        void NextFrame();

        int Allocate();
        void Deallocate(int _index);

       // expressed in indices
        int MaxAllocatedIndex() const { return m_maxAllocatedIndex; }
        int Size() const { return m_maxAllocatedIndex - (int)m_freeList.size(); }
        int Capacity() const { return m_maxIndex; }

        template <class T>
        void Allocate(int _count, T& _container)
        {
            _container.resize(_count);
            Allocate(_count, _container.data());
        }

        template <class T>
        void Allocate(int _count, T* _array)
        {
            const int totalFree = (int)m_freeList.size() + (m_maxIndex - m_maxAllocatedIndex);

            if (!AVA_VERIFY(_count <= totalFree, 
                "[Memory] not enough free space to allocate %d indices (used %d/%d).", 
                _count, Size(), Capacity()))
            {
                // Not enough free space to allocate
                return;
            }

            // If the free list is not empty, take from it
            if (!m_freeList.empty())
            {
                const int countFromFreeList = std::min(_count, (int)m_freeList.size());
                const auto freeListStart = m_freeList.end() - countFromFreeList;
                const auto freeListEnd = m_freeList.end();

                for (auto it = freeListStart; it != freeListEnd; ++it)
                {
                    *_array++ = *it;
                }

                m_freeList.erase(freeListStart, freeListEnd);
                _count -= countFromFreeList;
            }

            // And "allocate" the rest
            for (int index = m_maxAllocatedIndex; index < (m_maxAllocatedIndex + _count); ++index)
            {
                *_array++ = index;
            }

            m_maxAllocatedIndex += _count;
        }

        template <class T>
        void Deallocate(const T& _container)
        {
            m_waitingDealloc->insert(m_waitingDealloc->end(), _container.begin(), _container.end());
        }

    private:
        int                               m_maxIndex = 0;
        int                               m_maxAllocatedIndex = 0;
        std::vector<int>                  m_freeList{};
        std::vector<int>*                 m_waitingDealloc = nullptr;
        std::vector<int>*                 m_oldestWaitingDealloc = nullptr;
        std::vector< std::vector<int> >   m_waitingLists{};
    };

}
