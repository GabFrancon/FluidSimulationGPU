#include <avapch.h>
#include "GpuBuffer.h"

#include <Graphics/GraphicsContext.h>
#include <Debug/Assert.h>

namespace Ava {

    // -------- GpuBuffer implementation --------------------------------------------------------

    void* GpuBuffer::Map(const u32 _offset/*= 0*/, const u32 _size/*= UINT32_MAX*/)
    {
        return nullptr;
    }

    void GpuBuffer::Unmap()
    {
        AVA_ASSERT(!HasFlag(AVA_BUFFER_GPU_ONLY), "The buffer can't be accessed from CPU side!");
    }

    GpuBufferRange::GpuBufferRange(GpuBuffer* _buffer)
    {
        if (_buffer)
        {
            buffer = _buffer;
            size = _buffer->GetSize();
        }
    }

    void GpuBufferRange::UploadData() const
    {
        memcpy(buffer->Map(offset, size), data, size);
        buffer->Unmap();
    }

    bool GpuBufferRange::operator==(const GpuBufferRange& _other) const
    {
        return
            data == _other.data &&
            offset == _other.offset &&
            size == _other.size &&
            buffer == _other.buffer;
    }

    bool GpuBufferRange::operator!=(const GpuBufferRange& _other) const
    {
        return !(*this == _other);
    }


    // -------- UniformBuffer implementation ----------------------------------------------------

    ConstantBuffer::ConstantBuffer(const u32 _size, const u32 _flags/*= 0*/)
    {
        m_size = _size;
        m_flags = _flags;
    }


    // -------- IndirectBuffer implementation ---------------------------------------------------

    IndirectBuffer::IndirectBuffer(const u32 _size, const u32 _flags/*= 0*/)
    {
        m_size = _size;
        m_flags = _flags;
    }


    // -------- VertexBuffer implementation------------------------------------------------------

    VertexBuffer::VertexBuffer(const u32 _size, const u32 _flags/*= 0*/)
    {
        m_stride = 1;
        m_size = _size;
        m_flags = _flags;

        m_vertexLayout = AVA_DISABLE_STATE;
    }

    VertexBuffer::VertexBuffer(const VertexLayout& _vertexLayout, const u32 _vertexCount, const u32 _flags/*= 0*/)
    {
        m_vertexCount = _vertexCount;
        m_stride = _vertexLayout.stride;
        m_size = _vertexCount * m_stride;
        m_flags = _flags;

        m_vertexLayout = GraphicsContext::CreateVertexLayout(_vertexLayout);
    }

    void VertexBufferRange::UploadData() const
    {
        memcpy(buffer->Map(offset), data, size);
        buffer->Unmap();
    }

    u32 VertexBufferRange::GetVertexCount() const
    {
        return vertexCount;
    }

    VertexLayoutID VertexBufferRange::GetVertexLayout() const
    {
        return vertexLayout;
    }

    bool VertexBufferRange::operator==(const VertexBufferRange& _other) const
    {
        return
            data == _other.data &&
            offset == _other.offset &&
            size == _other.size &&
            buffer == _other.buffer;
    }

    bool VertexBufferRange::operator!=(const VertexBufferRange& _other) const
    {
        return !(*this == _other);
    }


    // -------- IndexBuffer implementation ------------------------------------------------------

    IndexBuffer::IndexBuffer(const u32 _indexCount, const u32 _flags/*= 0*/)
    {
        m_indexCount = _indexCount;
        m_flags = _flags;

        if (HasFlag(AVA_BUFFER_INDEX_UINT32))
        {
            m_size = _indexCount * sizeof(u32);
        }
        else
        {
            m_size = _indexCount * sizeof(u16);
        }
    }

    void IndexBufferRange::UploadData() const
    {
        memcpy(buffer->Map(offset), data, size);
        buffer->Unmap();
    }

    u32 IndexBufferRange::GetIndexCount() const
    {
        return indexCount;
    }

    bool IndexBufferRange::Use32BitIndices() const
    {
        return use32Bits;
    }

    bool IndexBufferRange::operator==(const IndexBufferRange& _other) const
    {
        return
            data == _other.data &&
            offset == _other.offset &&
            size == _other.size &&
            buffer == _other.buffer;
    }

    bool IndexBufferRange::operator!=(const IndexBufferRange& _other) const
    {
        return !(*this == _other);
    }

}