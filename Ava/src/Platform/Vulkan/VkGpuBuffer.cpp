#include <avapch.h>
#include "VkGpuBuffer.h"

#include <Debug/Assert.h>
#include <Platform/Vulkan/VkGraphicsContext.h>

namespace Ava {

    //------------- VkConstantBuffer implementation ------------------------------------------------------

    VkConstantBuffer::VkConstantBuffer(const u32 _size, const u32 _flags/*= 0*/)
        : ConstantBuffer(_size, _flags)
    {
        _InitVkBuffers();
    }

    VkConstantBuffer::~VkConstantBuffer()
    {
        vmaDestroyBuffer(VkGraphicsContext::GetAllocator(), m_buffer, m_alloc);

        if (m_transferBuffer)
        {
            vmaDestroyBuffer(VkGraphicsContext::GetAllocator(), m_transferBuffer, m_transferAlloc);
        }
    }

    void VkConstantBuffer::SetDebugName(const char* _name)
    {
        VkGraphicsContext::SetDebugObjectName(m_buffer, VK_OBJECT_TYPE_BUFFER, _name);
    }

    void* VkConstantBuffer::Map(const u32 _offset/*= 0*/, const u32 _size/*= UINT32_MAX*/)
    {
        AVA_ASSERT(!HasFlag(AVA_BUFFER_GPU_ONLY), "The buffer can't be accessed from CPU side.");

        if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            void* data = m_mappedData;
            if (!data)
            {
                vmaMapMemory(VkGraphicsContext::GetAllocator(), m_alloc, &data);
            }
            return (u8*)data + _offset;
        }
        else
        {
            m_transferOffset = _offset;
            m_transferSize = (_size == UINT32_MAX) ? (m_size - _offset) : _size;

            void* data;
            vmaMapMemory(VkGraphicsContext::GetAllocator(), m_transferAlloc, &data);
            return (u8*)data + _offset;
        }
    }

    void VkConstantBuffer::Unmap()
    {
        AVA_ASSERT(!HasFlag(AVA_BUFFER_GPU_ONLY), "The buffer can't be accessed from CPU side.");

        if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            if (!m_mappedData)
            {
                vmaUnmapMemory(VkGraphicsContext::GetAllocator(), m_alloc);
            }
        }
        else
        {
            vmaUnmapMemory(VkGraphicsContext::GetAllocator(), m_transferAlloc);

            // Copies the data to GPU
            VkBufferCopy copyInfo{};
            copyInfo.srcOffset = m_transferOffset;
            copyInfo.dstOffset = m_transferOffset;
            copyInfo.size = m_transferSize;

            const VkCommandBuffer cmd = VkGraphicsContext::BeginTransferCommand();
            vkCmdCopyBuffer(cmd, m_transferBuffer, m_buffer, 1, &copyInfo);
            VkGraphicsContext::EndTransferCommand(cmd);
        }
    }

    void VkConstantBuffer::_InitVkBuffers()
    {
        VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if (HasFlag(AVA_BUFFER_READ_WRITE))
        {
            usageFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = m_size;
        bufferInfo.usage = usageFlags;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (HasFlag(AVA_BUFFER_GPU_ONLY))
        {
            // Main buffer is only accessible from GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            const VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));
        }
        else if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            // Main buffer can be accessed from both CPU and GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

            VmaAllocationInfo allocInfo;
            const VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, &allocInfo);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));

            m_mappedData = allocInfo.pMappedData;
            AVA_ASSERT(m_mappedData != nullptr);
        }
        else
        {
            // Main buffer is only accessible from GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));

            // Transfer buffer is accessible from CPU
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

            result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_transferBuffer, &m_transferAlloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate transfer buffer: %s", VkGraphicsContext::GetVkResultStr(result));
        }
    }


    //-------------- VkIndirectBuffer implementation -----------------------------------------------------

    VkIndirectBuffer::VkIndirectBuffer(const u32 _size, const u32 _flags/*= 0*/)
        : IndirectBuffer(_size, _flags)
    {
        _InitVkBuffers();
    }

    VkIndirectBuffer::~VkIndirectBuffer()
    {
        vmaDestroyBuffer(VkGraphicsContext::GetAllocator(), m_buffer, m_alloc);

        if (m_transferBuffer)
        {
            vmaDestroyBuffer(VkGraphicsContext::GetAllocator(), m_transferBuffer, m_transferAlloc);
        }
    }

    void VkIndirectBuffer::SetDebugName(const char* _name)
    {
        VkGraphicsContext::SetDebugObjectName(m_buffer, VK_OBJECT_TYPE_BUFFER, _name);
    }

    void* VkIndirectBuffer::Map(const u32 _offset/*= 0*/, const u32 _size/*= UINT32_MAX*/)
    {
        AVA_ASSERT(!HasFlag(AVA_BUFFER_GPU_ONLY), "The buffer can't be accessed from CPU side.");

        if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            void* data = m_mappedData;
            if (!data)
            {
                vmaMapMemory(VkGraphicsContext::GetAllocator(), m_alloc, &data);
            }
            return (u8*)data + _offset;
        }
        else
        {
            m_transferOffset = _offset;
            m_transferSize = (_size == UINT32_MAX) ? (m_size - _offset) : _size;

            void* data;
            vmaMapMemory(VkGraphicsContext::GetAllocator(), m_transferAlloc, &data);
            return (u8*)data + _offset;
        }
    }

    void VkIndirectBuffer::Unmap()
    {
        AVA_ASSERT(!HasFlag(AVA_BUFFER_GPU_ONLY), "The buffer can't be accessed from CPU side.");

        if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            if (!m_mappedData)
            {
                vmaUnmapMemory(VkGraphicsContext::GetAllocator(), m_alloc);
            }
        }
        else
        {
            vmaUnmapMemory(VkGraphicsContext::GetAllocator(), m_transferAlloc);

            // Copies the data to GPU
            VkBufferCopy copyInfo{};
            copyInfo.srcOffset = m_transferOffset;
            copyInfo.dstOffset = m_transferOffset;
            copyInfo.size = m_transferSize;

            const VkCommandBuffer cmd = VkGraphicsContext::BeginTransferCommand();
            vkCmdCopyBuffer(cmd, m_transferBuffer, m_buffer, 1, &copyInfo);
            VkGraphicsContext::EndTransferCommand(cmd);
        }
    }

    void VkIndirectBuffer::_InitVkBuffers()
    {
        VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if (HasFlag(AVA_BUFFER_READ_WRITE))
        {
            usageFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = m_size;
        bufferInfo.usage = usageFlags;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (HasFlag(AVA_BUFFER_GPU_ONLY))
        {
            // Main buffer is only accessible from GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            const VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));
        }
        else if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            // Main buffer can be accessed from both CPU and GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

            VmaAllocationInfo allocInfo;
            const VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, &allocInfo);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));

            m_mappedData = allocInfo.pMappedData;
            AVA_ASSERT(m_mappedData != nullptr);
        }
        else
        {
            // Main buffer is only accessible from GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));

            // Transfer buffer is accessible from CPU
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

            result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_transferBuffer, &m_transferAlloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate transfer buffer: %s", VkGraphicsContext::GetVkResultStr(result));
        }
    }


    //-------------- VkVertexBuffer implementation -------------------------------------------------------

    VkVertexBuffer::VkVertexBuffer(const u32 _size, const u32 _flags)
        : VertexBuffer(_size, _flags)
    {
        _InitVkBuffers();
    }

    VkVertexBuffer::VkVertexBuffer(const VertexLayout& _vertexLayout, const u32 _vertexCount, const u32 _flags/*= 0*/)
        : VertexBuffer(_vertexLayout, _vertexCount, _flags)
    {
        _InitVkBuffers();
    }

    VkVertexBuffer::~VkVertexBuffer()
    {
        vmaDestroyBuffer(VkGraphicsContext::GetAllocator(), m_buffer, m_alloc);

        if (m_transferBuffer)
        {
            vmaDestroyBuffer(VkGraphicsContext::GetAllocator(), m_transferBuffer, m_transferAlloc);
        }
    }

    void VkVertexBuffer::SetDebugName(const char* _name)
    {
        VkGraphicsContext::SetDebugObjectName(m_buffer, VK_OBJECT_TYPE_BUFFER, _name);
    }

    void* VkVertexBuffer::Map(const u32 _offset/*= 0*/, const u32 _size/*= UINT32_MAX*/)
    {
        AVA_ASSERT(!HasFlag(AVA_BUFFER_GPU_ONLY), "The buffer can't be accessed from CPU side.");

        if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            void* data = m_mappedData;
            if (!data)
            {
                vmaMapMemory(VkGraphicsContext::GetAllocator(), m_alloc, &data);
            }
            return (u8*)data + _offset;
        }
        else
        {
            m_transferOffset = _offset;
            m_transferSize = (_size == UINT32_MAX) ? (m_size - _offset) : _size;

            void* data;
            vmaMapMemory(VkGraphicsContext::GetAllocator(), m_transferAlloc, &data);
            return (u8*)data + _offset;
        }
    }

    void VkVertexBuffer::Unmap()
    {
        AVA_ASSERT(!HasFlag(AVA_BUFFER_GPU_ONLY), "The buffer can't be accessed from CPU side.");

        if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            if (!m_mappedData)
            {
                vmaUnmapMemory(VkGraphicsContext::GetAllocator(), m_alloc);
            }
        }
        else
        {
            vmaUnmapMemory(VkGraphicsContext::GetAllocator(), m_transferAlloc);

            // Copies the data to GPU
            VkBufferCopy copyInfo{};
            copyInfo.srcOffset = m_transferOffset;
            copyInfo.dstOffset = m_transferOffset;
            copyInfo.size = m_transferSize;

            const VkCommandBuffer cmd = VkGraphicsContext::BeginTransferCommand();
            vkCmdCopyBuffer(cmd, m_transferBuffer, m_buffer, 1, &copyInfo);
            VkGraphicsContext::EndTransferCommand(cmd);
        }
    }

    void VkVertexBuffer::_InitVkBuffers()
    {
        VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if (HasFlag(AVA_BUFFER_READ_WRITE))
        {
            usageFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = m_size;
        bufferInfo.usage = usageFlags;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (HasFlag(AVA_BUFFER_GPU_ONLY))
        {
            // Main buffer is only accessible from GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            const VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));
        }
        else if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            // Main buffer can be accessed from both CPU and GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

            VmaAllocationInfo allocInfo;
            const VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, &allocInfo);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));

            m_mappedData = allocInfo.pMappedData;
            AVA_ASSERT(m_mappedData != nullptr);
        }
        else
        {
            // Main buffer is only accessible from GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));

            // Transfer buffer is accessible from CPU
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

            result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_transferBuffer, &m_transferAlloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate transfer buffer: %s", VkGraphicsContext::GetVkResultStr(result));
        }
    }


    //-------------- VkIndexBuffer implementation --------------------------------------------------------

    VkIndexBuffer::VkIndexBuffer(const u32 _indexCount, const u32 _flags/*= 0*/)
        : IndexBuffer(_indexCount, _flags)
    {
        _InitVkBuffers();
    }

    VkIndexBuffer::~VkIndexBuffer()
    {
        vmaDestroyBuffer(VkGraphicsContext::GetAllocator(), m_buffer, m_alloc);

        if (m_transferBuffer)
        {
            vmaDestroyBuffer(VkGraphicsContext::GetAllocator(), m_transferBuffer, m_transferAlloc);
        }
    }

    void VkIndexBuffer::SetDebugName(const char* _name)
    {
        VkGraphicsContext::SetDebugObjectName(m_buffer, VK_OBJECT_TYPE_BUFFER, _name);
    }

    void* VkIndexBuffer::Map(const u32 _offset/*= 0*/, const u32 _size/*= UINT32_MAX*/)
    {
        AVA_ASSERT(!HasFlag(AVA_BUFFER_GPU_ONLY), "The buffer can't be accessed from CPU side.");

        if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            void* data = m_mappedData;
            if (!data)
            {
                vmaMapMemory(VkGraphicsContext::GetAllocator(), m_alloc, &data);
            }
            return (u8*)data + _offset;
        }
        else
        {
            m_transferOffset = _offset;
            m_transferSize = (_size == UINT32_MAX) ? (m_size - _offset) : _size;

            void* data;
            vmaMapMemory(VkGraphicsContext::GetAllocator(), m_transferAlloc, &data);
            return (u8*)data + _offset;
        }
    }

    void VkIndexBuffer::Unmap()
    {
        AVA_ASSERT(!HasFlag(AVA_BUFFER_GPU_ONLY), "The buffer can't be accessed from CPU side.");

        if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            if (!m_mappedData)
            {
                vmaUnmapMemory(VkGraphicsContext::GetAllocator(), m_alloc);
            }
        }
        else
        {
            vmaUnmapMemory(VkGraphicsContext::GetAllocator(), m_transferAlloc);

            // Copies the data to GPU
            VkBufferCopy copyInfo{};
            copyInfo.srcOffset = m_transferOffset;
            copyInfo.dstOffset = m_transferOffset;
            copyInfo.size = m_transferSize;

            const VkCommandBuffer cmd = VkGraphicsContext::BeginTransferCommand();
            vkCmdCopyBuffer(cmd, m_transferBuffer, m_buffer, 1, &copyInfo);
            VkGraphicsContext::EndTransferCommand(cmd);
        }
    }

    void VkIndexBuffer::_InitVkBuffers()
    {
        VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if (HasFlag(AVA_BUFFER_READ_WRITE))
        {
            usageFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = m_size;
        bufferInfo.usage = usageFlags;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (HasFlag(AVA_BUFFER_GPU_ONLY))
        {
            // Main buffer is only accessible from GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            const VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));
        }
        else if (HasFlag(AVA_BUFFER_DYNAMIC))
        {
            // Main buffer can be accessed from both CPU and GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

            VmaAllocationInfo allocInfo;
            const VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, &allocInfo);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));

            m_mappedData = allocInfo.pMappedData;
            AVA_ASSERT(m_mappedData != nullptr);
        }
        else
        {
            // Main buffer is only accessible from GPU
            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            VkResult result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_buffer, &m_alloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate GPU buffer: %s", VkGraphicsContext::GetVkResultStr(result));

            // Transfer buffer is accessible from CPU
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

            result = vmaCreateBuffer(VkGraphicsContext::GetAllocator(), &bufferInfo, &allocCreateInfo, &m_transferBuffer, &m_transferAlloc, nullptr);
            AVA_ASSERT(result == VK_SUCCESS, "[VMA] failed to allocate transfer buffer: %s", VkGraphicsContext::GetVkResultStr(result));
        }
    }

}