#pragma once
/// @file GpuBuffer.h
/// @brief interface for GPU buffers.

#include <Graphics/GraphicsCommon.h>

namespace Ava {

    enum BufferFlags
    {
        AVA_BUFFER_NONE = 0,
        // buffer usage
        AVA_BUFFER_READ_WRITE = AVA_BIT(0),
        // memory access
        AVA_BUFFER_DYNAMIC = AVA_BIT(1),
        AVA_BUFFER_GPU_ONLY = AVA_BIT(2),
        // additional features
        AVA_BUFFER_INDEX_UINT32 = AVA_BIT(3)
    };

    /// @brief Interface representing GPU buffer objects.
    class GpuBuffer
    {
    public:
        virtual ~GpuBuffer() = default;
        virtual void SetDebugName(const char* _name) { }

        u32 GetFlags() const { return m_flags; }
        bool HasFlag(const BufferFlags _flag) const { return m_flags & _flag; }
        u32 GetSize() const { return m_size; }

        virtual void* Map(u32 _offset = 0, u32 _size = UINT32_MAX);
        virtual void Unmap();

    protected:
        u32 m_size = 0;
        u32 m_flags = AVA_BUFFER_NONE;
    };

    /// @brief Represents a subset of a GpuBuffer.
    struct GpuBufferRange
    {
        GpuBuffer* buffer = nullptr;
        u32 size = 0;
        u32 offset = 0;
        u32 frameCreationId = 0;
        void* data = nullptr;

        GpuBufferRange() = default;
        explicit GpuBufferRange(GpuBuffer* _buffer);

        void UploadData() const;

        bool operator==(const GpuBufferRange& _other) const;
        bool operator!=(const GpuBufferRange& _other) const;
    };

    /// @brief GpuBuffer used to store usual shader data.
    class ConstantBuffer : public GpuBuffer
    {
    public:
        ConstantBuffer(u32 _size, u32 _flags = 0);
    };

    /// @brief Represents a subset of a ConstantBuffer.
    struct ConstantBufferRange : public GpuBufferRange
    {
        ConstantBufferRange() = default;
        ConstantBufferRange(const ConstantBufferRange&) = default;
        ConstantBufferRange(ConstantBuffer* _buffer) : GpuBufferRange(_buffer) {}
        ConstantBufferRange(const GpuBufferRange& _range) { static_cast<GpuBufferRange&>(*this) = _range; }
    };

    /// @brief GpuBuffer used to store draw/dispatch commands.
    class IndirectBuffer : public GpuBuffer
    {
    public:
        IndirectBuffer(u32 _size, u32 _flags = 0);
    };

    /// @brief Represents a subset of an IndirectBuffer.
    struct IndirectBufferRange : public GpuBufferRange
    {
        IndirectBufferRange() = default;
        IndirectBufferRange(const IndirectBufferRange&) = default;
        IndirectBufferRange(IndirectBuffer* _buffer) : GpuBufferRange(_buffer) {}
        IndirectBufferRange(const GpuBufferRange& _range) { static_cast<GpuBufferRange&>(*this) = _range; }
    };

    /// @brief Vertex buffer object.
    class VertexBuffer : public GpuBuffer
    {
    public:
        /// @brief for giant vertex buffers divided in sub-allocations.
        explicit VertexBuffer(u32 _size, u32 _flags = 0);
        /// @brief for classic vertex buffers.
        explicit VertexBuffer(const VertexLayout& _vertexLayout, u32 _vertexCount, u32 _flags = 0);

        u32 GetStride() const { return m_stride; }
        u32 GetVertexCount() const { return m_vertexCount; }
        VertexLayoutID GetVertexLayout() const { return m_vertexLayout; }

    protected:
        VertexLayoutID m_vertexLayout;
        u32 m_vertexCount = 0;
        u32 m_stride = 0;
    };

    /// @brief Represents a subset of a VertexBuffer.
    struct VertexBufferRange
    {
        VertexBuffer* buffer = nullptr;

        u32 vertexCount = 0;
        VertexLayoutID vertexLayout;
        u32 frameCreationId = 0;

        u32 size = 0;
        u32 offset = 0;
        void* data = nullptr;

        VertexBufferRange() = default;
        VertexBufferRange(const VertexBufferRange&) = default;

        explicit VertexBufferRange(VertexBuffer* _buffer)
        : buffer(_buffer), vertexCount(_buffer->GetVertexCount()), vertexLayout(_buffer->GetVertexLayout()), size(_buffer->GetSize()) {}

        void UploadData() const;
        u32 GetVertexCount() const;
        VertexLayoutID GetVertexLayout() const;

        bool operator==(const VertexBufferRange& _other) const;
        bool operator!=(const VertexBufferRange& _other) const;
    };

    /// @brief Index buffer object.
    class IndexBuffer : public GpuBuffer
    {
    public:
        explicit IndexBuffer(u32 _indexCount, u32 _flags = 0);
        u32 GetIndexCount() const { return m_indexCount; }

    protected:
        u32 m_indexCount = 0;
    };

    /// @brief Represents a subset of a IndexBuffer.
    struct IndexBufferRange
    {
        IndexBuffer* buffer = nullptr;

        u32 indexCount = 0;
        bool use32Bits = false;
        u32 frameCreationId = 0;

        u32 size = 0;
        u32 offset = 0;
        void* data = nullptr;

        IndexBufferRange() = default;
        IndexBufferRange(const IndexBufferRange&) = default;

        explicit IndexBufferRange(IndexBuffer* _buffer)
        : buffer(_buffer), indexCount(_buffer->GetIndexCount()), use32Bits(_buffer->HasFlag(AVA_BUFFER_INDEX_UINT32)), size(_buffer->GetSize()) {}

        void UploadData() const;
        u32 GetIndexCount() const;
        bool Use32BitIndices() const;

        bool operator==(const IndexBufferRange& _other) const;
        bool operator!=(const IndexBufferRange& _other) const;
    };

}