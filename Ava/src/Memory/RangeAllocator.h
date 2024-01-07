#pragma once
/// @file RangeAllocator.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    /// @brief Linear range allocator helper.
    class RangeAllocator
    {
    public:
        RangeAllocator(u32 _rangeCapacity);
        ~RangeAllocator() = default;

        struct Range
        {
            u32 rangeIdx;
            u32 offset;
        };

        Range AllocateRange(u32 _size, u32 _alignment);
        u32 GetCapacity() const;
        void Reset();

    protected:
        const u32 m_rangeCapacity;
        u32 m_lastRangeIdx = 0u;
        u32 m_lastRangeSize = 0u;
    };

}
