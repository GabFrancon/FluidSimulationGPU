#pragma once
/// @file TransientPool.h
/// @brief Handles allocation inside a ring buffer, valid for the duration of one graphics context cycle.

#include <Graphics/GpuBuffer.h>
#include <Graphics/GraphicsContext.h>
#include <Memory/RangeAllocator.h>

namespace Ava {

    template <class Buffer>
    static Buffer* AllocateTransientBuffer(u32 _size, u32 _flags);

    template <>
    inline ConstantBuffer* AllocateTransientBuffer<ConstantBuffer>(const u32 _size, const u32 _flags)
    {
        ConstantBuffer* buffer = GraphicsContext::CreateConstantBuffer(_size, _flags);
        buffer->SetDebugName("Transient Constant Buffer");
        return buffer;
    }

    template <>
    inline IndirectBuffer* AllocateTransientBuffer<IndirectBuffer>(const u32 _size, const u32 _flags)
    {
        IndirectBuffer* buffer = GraphicsContext::CreateIndirectBuffer(_size, _flags);
        buffer->SetDebugName("Transient Indirect Buffer");
        return buffer;
    }

    template <>
    inline VertexBuffer* AllocateTransientBuffer<VertexBuffer>(const u32 _size, const u32 _flags)
    {
        VertexBuffer* buffer = GraphicsContext::CreateVertexBuffer(_size, _flags);
        buffer->SetDebugName("Transient Vertex Buffer");
        return buffer;
    }

    template <>
    inline IndexBuffer* AllocateTransientBuffer<IndexBuffer>(const u32 _size, const u32 _flags)
    {
        IndexBuffer* buffer = GraphicsContext::CreateIndexBuffer(_size / 2, _flags);
        buffer->SetDebugName("Transient Index Buffer");
        return buffer;
    }

    template <class Buffer, class BufferRange>
    class BufferRangeAllocator : public RangeAllocator
    {
    public:
        BufferRangeAllocator(const u32 _bufferCapacity, const u32 _flags)
            : RangeAllocator(_bufferCapacity)
            , m_flags(_flags)
        {
        }

        ~BufferRangeAllocator()
        {
            for (BufferData& bufferData : m_buffers)
            {
                GraphicsContext::DestroyBuffer(bufferData.buffer);

                if (bufferData.data)
                {
                    delete[] bufferData.data;
                }
            }
            m_buffers.clear();
        }

        void NextFrame()
        {
            Reset();
            m_allocatedBytes = 0;
        }

        BufferRange AllocateRange(const u32 _size, const u32 _alignment)
        {
            Range range = RangeAllocator::AllocateRange(_size, _alignment);

            if (range.rangeIdx >= m_buffers.size())
            {
                BufferData& bufferData = m_buffers.emplace_back();
                bufferData.buffer = AllocateTransientBuffer<Buffer>(m_rangeCapacity, m_flags);

                if ((m_flags & AVA_BUFFER_GPU_ONLY) == 0)
                {
                    bufferData.data = new u8[m_rangeCapacity];
                }
            }

            BufferRange bufferRange{};
            bufferRange.buffer = m_buffers[range.rangeIdx].buffer;
            bufferRange.offset = range.offset;
            bufferRange.size = _size;

            if ((m_flags & AVA_BUFFER_GPU_ONLY) == 0)
            {
                bufferRange.data = m_buffers[range.rangeIdx].data + range.offset;
                memset(bufferRange.data, 0, bufferRange.size);
            }

            m_allocatedBytes += _size;
            return bufferRange;
        }

        u32 GetAllocatedBytes() const
        {
            return m_allocatedBytes;
        }

        u32 GetBufferCount() const
        {
            return (u32)m_buffers.size();
        }

    private:
        struct BufferData
        {
            Buffer* buffer = nullptr;
            u8*     data   = nullptr;
        };

        std::vector<BufferData> m_buffers;
        u32 m_allocatedBytes = 0;
        u32 m_flags = 0;
    };

    /// @brief Holds a transient buffer range allocator for each buffer type.
    struct TransientPool
    {
        // Each transient buffer is allocated in sub-buffers of 256 KiB
        // -> a single transient allocation request cannot exceed this size.
        static constexpr u32 kBufferCapacity = 256 * 1024;

        // transient buffer allocators
        BufferRangeAllocator<ConstantBuffer, ConstantBufferRange> gpuOnlyBuffers { kBufferCapacity, AVA_BUFFER_READ_WRITE | AVA_BUFFER_GPU_ONLY };
        BufferRangeAllocator<ConstantBuffer, ConstantBufferRange> constantBuffers{ kBufferCapacity, AVA_BUFFER_READ_WRITE | AVA_BUFFER_DYNAMIC  };
        BufferRangeAllocator<IndirectBuffer, IndirectBufferRange> indirectBuffers{ kBufferCapacity, AVA_BUFFER_READ_WRITE | AVA_BUFFER_DYNAMIC  };
        BufferRangeAllocator<VertexBuffer,   VertexBufferRange>   vertexBuffers  { kBufferCapacity, AVA_BUFFER_READ_WRITE | AVA_BUFFER_DYNAMIC  };
        BufferRangeAllocator<IndexBuffer,    IndexBufferRange>    indexBuffers   { kBufferCapacity, AVA_BUFFER_READ_WRITE | AVA_BUFFER_DYNAMIC  };

        // bytes allocated per frame
        u32 gpuOnlyBufferLastAllocation = 0;
        u32 constantBufferLastAllocation = 0;
        u32 indirectBufferLastAllocation = 0;
        u32 vertexBufferLastAllocation = 0;
        u32 indexBufferLastAllocation = 0;

        void Reset()
        {
            // save allocation statistics
            gpuOnlyBufferLastAllocation = gpuOnlyBuffers.GetAllocatedBytes();
            constantBufferLastAllocation = constantBuffers.GetAllocatedBytes();
            indirectBufferLastAllocation = indirectBuffers.GetAllocatedBytes();
            vertexBufferLastAllocation = vertexBuffers.GetAllocatedBytes();
            indexBufferLastAllocation = indexBuffers.GetAllocatedBytes();

            // reset buffer allocators
            gpuOnlyBuffers.NextFrame();
            constantBuffers.NextFrame();
            indirectBuffers.NextFrame();
            vertexBuffers.NextFrame();
            indexBuffers.NextFrame();
        }
    };
}
