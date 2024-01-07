#include <avapch.h>
#include "RangeAllocator.h"

#include <Debug/Assert.h>

namespace Ava {

    RangeAllocator::RangeAllocator(const u32 _rangeCapacity) : m_rangeCapacity(_rangeCapacity)
    {
    }

    RangeAllocator::Range RangeAllocator::AllocateRange(const u32 _size, const u32 _alignment)
    {
        AVA_ASSERT(_size < m_rangeCapacity);
        AVA_ASSERT(_alignment % 2 == 0);

        // Align beginning of next range
        u32 offset = (m_lastRangeSize + _alignment - 1) & ~(_alignment - 1);

        if (offset + _size > m_rangeCapacity)
        {
            // Doesn't fit, go into the next buffer
            m_lastRangeIdx++;
            m_lastRangeSize = _size;
            offset = 0;
        }
        else
        {
            m_lastRangeSize = offset + _size;
        }

        Range range{};
        range.offset = offset;
        range.rangeIdx = m_lastRangeIdx;
        return range;
    }

    u32 RangeAllocator::GetCapacity() const
    {
        return m_rangeCapacity;
    }

    void RangeAllocator::Reset()
    {
        m_lastRangeIdx = 0;
        m_lastRangeSize = 0;
    }

}