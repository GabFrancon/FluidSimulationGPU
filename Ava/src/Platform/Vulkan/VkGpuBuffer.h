#pragma once
/// @file VkGpuBuffer.h
/// @brief file implementing GpuBuffer.h for Vulkan.

#include <Graphics/GpuBuffer.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <vulkan/vulkan.h>
#include <VMA/vk_mem_alloc.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    class VkConstantBuffer final : public ConstantBuffer
    {
    public:
        explicit VkConstantBuffer(u32 _size, u32 _flags = 0);
        ~VkConstantBuffer() override;

        VkBuffer GetVkBuffer() const { return m_buffer; }
        VmaAllocation GetAlloc() const { return m_alloc; }

        void SetDebugName(const char* _name) override;
        void* Map(u32 _offset = 0, u32 _size = UINT32_MAX) override;
        void Unmap() override;

    private:
        void _InitVkBuffers();

        // Main buffer
        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_alloc = VK_NULL_HANDLE;
        void* m_mappedData = nullptr;

        // Transfer CPU->GPU buffer
        u32 m_transferSize = 0;
        u32 m_transferOffset = 0;
        VkBuffer m_transferBuffer = VK_NULL_HANDLE;
        VmaAllocation m_transferAlloc = VK_NULL_HANDLE;
    };

    class VkIndirectBuffer final: public IndirectBuffer
    {
    public:
        explicit VkIndirectBuffer(u32 _size, u32 _flags = 0);
        ~VkIndirectBuffer() override;

        VkBuffer GetVkBuffer() const { return m_buffer; }
        VmaAllocation GetAlloc() const { return m_alloc; }

        void SetDebugName(const char* _name) override;
        void* Map(u32 _offset = 0, u32 _size = UINT32_MAX) override;
        void Unmap() override;

    private:
        void _InitVkBuffers();

        // Main buffer
        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_alloc = VK_NULL_HANDLE;
        void* m_mappedData = nullptr;

        // Transfer CPU->GPU buffer
        u32 m_transferSize = 0;
        u32 m_transferOffset = 0;
        VkBuffer m_transferBuffer = VK_NULL_HANDLE;
        VmaAllocation m_transferAlloc = VK_NULL_HANDLE;
    };

    class VkVertexBuffer final : public VertexBuffer
    {
    public:
        /// @brief for giant vertex buffers divided in sub-allocations.
        explicit VkVertexBuffer(u32 _size, u32 _flags = 0);
        /// @brief for classic vertex buffers.
        explicit VkVertexBuffer(const VertexLayout& _vertexLayout, u32 _vertexCount, u32 _flags = 0);
        ~VkVertexBuffer() override;

        VkBuffer GetVkBuffer() const { return m_buffer; }
        VmaAllocation GetAlloc() const { return m_alloc; }

        void SetDebugName(const char* _name) override;
        void* Map(u32 _offset = 0, u32 _size = UINT32_MAX) override;
        void Unmap() override;

    private:
        void _InitVkBuffers();

        // Main buffer
        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_alloc = VK_NULL_HANDLE;
        void* m_mappedData = nullptr;

        // Transfer CPU->GPU buffer
        u32 m_transferSize = 0;
        u32 m_transferOffset = 0;
        VkBuffer m_transferBuffer = VK_NULL_HANDLE;
        VmaAllocation m_transferAlloc = VK_NULL_HANDLE;
    };

    class VkIndexBuffer final : public IndexBuffer
    {
    public:
        explicit VkIndexBuffer(u32 _indexCount, u32 _flags = 0);
        ~VkIndexBuffer() override;

        VkBuffer GetVkBuffer() const { return m_buffer; }
        VmaAllocation GetAlloc() const { return m_alloc; }

        void SetDebugName(const char* _name) override;
        void* Map(u32 _offset = 0, u32 _size = UINT32_MAX) override;
        void Unmap() override;

    private:
        void _InitVkBuffers();

        // Main buffer
        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_alloc = VK_NULL_HANDLE;
        void* m_mappedData = nullptr;

        // Transfer CPU->GPU buffer
        u32 m_transferSize = 0;
        u32 m_transferOffset = 0;
        VkBuffer m_transferBuffer = VK_NULL_HANDLE;
        VmaAllocation m_transferAlloc = VK_NULL_HANDLE;
    };

}

