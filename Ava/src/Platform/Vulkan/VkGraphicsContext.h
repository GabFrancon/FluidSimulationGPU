#pragma once
/// @file VkGraphicsContext.h
/// @brief file implementing GraphicsContext.h for Vulkan.

#include <Graphics/GraphicsContext.h>
#include <Math/Geometry.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <VMA/vk_mem_alloc.h>

AVA_DISABLE_WARNINGS_END
//--------------------------------------------

#include "VkShader.h"
#include "VkTexture.h"
#include "VkGpuBuffer.h"
#include "VkFrameBuffer.h"
#include "VkShaderProgram.h"
#include "VkGraphicsEnums.h"
#include "VkGraphicsCommon.h"

namespace Ava {

    class VkGraphicsContext
    {
    public:
        explicit VkGraphicsContext(u32 _contextId);
        ~VkGraphicsContext();

        static void Init();
        static void Shutdown();
        void Reset();

        // Getters
        static VkDevice GetDevice();
        static VkPhysicalDeviceLimits GetDeviceLimits();
        static VmaAllocator GetAllocator();
        VkCommandBuffer GetCommandBuffer() const;

        // Time
        u64 GetGpuTimestamp() const;
        static void WaitIdle();

        // Frame cycle commands
        void StartFrame();
        void EndFrameAndSubmit();

        // Transfer commands
        static VkCommandBuffer BeginTransferCommand();
        static void EndTransferCommand(VkCommandBuffer _cmd);

        // Draw commands
        void Draw(ShaderProgram* _program, u32 _instanceCount = 1);
        void Draw(ShaderProgram* _program, const IndirectBufferRange& _indirectRange);

        // Dispatch commands
        void Dispatch(ShaderProgram* _program, u32 _groupCountX, u32 _groupCountY, u32 _groupCountZ);
        void Dispatch(ShaderProgram* _program, const IndirectBufferRange& _indirectRange);

        // Constant buffers
        static u32 GetConstantBufferAlignment(u32 _flags = 0);
        static ConstantBuffer* CreateConstantBuffer(u32 _size, u32 _flags = 0);
        void SetConstantBuffer(ShaderStage::Enum _stage, u8 _slot, const ConstantBufferRange& _bfRange);
        static void DestroyBuffer(ConstantBuffer* _buffer);

        // Indirect buffers
        static u32 GetIndirectBufferAlignment(u32 _flags = 0);
        static IndirectBuffer* CreateIndirectBuffer(u32 _size, u32 _flags = 0);
        static void DestroyBuffer(IndirectBuffer* _buffer);

        // Vertex buffers
        static u32 GetVertexBufferAlignment(u32 _flags = 0);
        static VertexBuffer* CreateVertexBuffer(u32 _size, u32 _flags = 0);
        static VertexBuffer* CreateVertexBuffer(const VertexLayout& _vertexLayout, u32 _vertexCount, u32 _flags = 0);
        void SetVertexBuffer(const VertexBufferRange& _bfRange);
        static void DestroyBuffer(VertexBuffer* _buffer);

        // Index buffers
        static u32 GetIndexBufferAlignment(u32 _flags = 0);
        static IndexBuffer* CreateIndexBuffer(u32 _indexCount, u32 _flags = 0);
        void SetIndexBuffer(const IndexBufferRange& _bfRange);
        static void DestroyBuffer(IndexBuffer* _buffer);

        // Textures
        static Texture* CreateTexture(const TextureDescription& _desc);
        static Texture* CreateTexture(const char* _path, u32 _flags = 0);
        static Texture* CreateTexture(const TextureData& _data, u32 _flags = 0);
        static Texture* ConstructTexture(void* _memory, const char* _path, u32 _flags = 0);
        static VkSampler CreateSampler(const VkSamplerCreateInfo* _info);
        void SetTexture(ShaderStage::Enum _stage, u8 _slot, Texture* _texture);
        static void DestructTexture(Texture* _texture);
        static void DestroyTexture(Texture* _texture);

        // Storage resources
        static u32 GetStorageBufferAlignment();
        void SetStorageBuffer(ShaderStage::Enum _stage, u8 _slot, const ConstantBufferRange& _bfRange, Access::Enum _access);
        void SetStorageBuffer(ShaderStage::Enum _stage, u8 _slot, const IndirectBufferRange& _bfRange, Access::Enum _access);
        void SetStorageBuffer(ShaderStage::Enum _stage, u8 _slot, const VertexBufferRange& _bfRange, Access::Enum _access);
        void SetStorageBuffer(ShaderStage::Enum _stage, u8 _slot, const IndexBufferRange& _bfRange, Access::Enum _access);
        void SetStorageTexture(ShaderStage::Enum _stage, u8 _slot, Texture* _texture, Access::Enum _access, u16 _mip = 0);

        // Framebuffers
        static FrameBuffer* GetMainFramebuffer();
        const FrameBuffer* GetCurrentFramebuffer() const;
        static FrameBuffer* CreateFrameBuffer(const FrameBufferDescription& _desc);
        static VkRenderPass CreateRenderPass(const VkRenderPassCreateInfo* _info);
        void SetFramebuffer(FrameBuffer* _framebuffer = nullptr);
        void Clear(const Color* _color, const float* _depth);
        static void DestroyFrameBuffer(FrameBuffer* _framebuffer);

        // Shaders
        static Shader* CreateShader(ShaderStage::Enum _stage, const ShaderDescription& _desc);
        static Shader* CreateShader(const ShaderData& _data);
        static Shader* ConstructShader(void* _memory, ShaderStage::Enum _stage, const ShaderDescription& _desc);
        static VkDescriptorSetLayout CreateShaderLayout(const VkDescriptorSetLayoutCreateInfo* _info);
        static void DestructShader(Shader* _shader);
        static void DestroyShader(Shader* _shader);

        // Shader programs
        static ShaderProgram* CreateProgram(const ShaderProgramDescription& _desc);
        static ShaderProgram* CreateProgram(Shader* _stages[ShaderStage::Count], const ShaderResourceNames& _varNames);
        static ShaderProgram* ConstructProgram(void* _memory, const ShaderProgramDescription& _desc);
        static void DestructProgram(ShaderProgram* _program);
        static void DestroyProgram(ShaderProgram* _program);

        // States
        void SetPrimitiveType(PrimitiveType::Enum _type);
        void SetRasterState(RasterStateID _rasterState);
        void SetDepthState(DepthStateID _depthState);
        void SetBlendState(BlendStateID _blendState);
        void SetWireframe(bool _enabled);
        void SetLineWidth(float _lineWidth);
        void SetViewport(u16 _width, u16 _height, u16 _x = 0, u16 _y = 0);
        void SetScissor(u16 _width, u16 _height, u16 _x = 0, u16 _y = 0);

        // Debug options
        static void EnableGraphicsDebug(bool _enable);
        static void ForceRecreateFramebuffer();
        static void TriggerFrameCapture();
        static const char* GetVkResultStr(VkResult _result);
        static void SetDebugObjectName(void* _object, VkObjectType _objectType, const char* _name);
        void AddDebugMarker(const char* _label, const Color& _color) const;
        void BeginDebugMarkerRegion(const char* _label, const Color& _color) const;
        void EndDebugMarkerRegion() const;
        void DisplayDebug() const;

    private:
        /// @brief Describes the dynamic states we want to track.
        struct DynamicState {
            Rect viewport{};
            Rect scissor{};
            float lineWidth = 1.f;
            FrameBuffer* framebuffer{ nullptr };
        };

        /// @brief Describes a constant buffer binding.
        struct ConstantBufferBinding {
            ConstantBufferRange bfRange;
        };

        /// @brief Describes a sampled texture binding.
        struct SampledTextureBinding {
            Texture* texture;
        };

        /// @brief Describes a storage buffer binding.
        struct StorageBufferBinding {
            BufferType::Enum bufferType;
            Access::Enum access;
            void* buffer;
            void* data;
            u32 offset;
            u32 size;
        };

        /// @brief Describes a storage texture binding.
        struct StorageTextureBinding {
            Access::Enum access;
            Texture* texture;
            u16 mip;
        };

        // Frame descriptors
        VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;                  // manages allocation of descriptor sets
        std::vector<VkWriteDescriptorSet> m_writeDescriptorCache;            // to avoid reallocating space for every draw
        std::vector<VkDescriptorBufferInfo> m_bufferInfoCache;               // to avoid reallocating space for every draw
        std::vector<VkDescriptorImageInfo>  m_imageInfoCache;                // to avoid reallocating space for every draw

        // Frame render objects
        VkCommandPool m_renderCmdPool = VK_NULL_HANDLE;                      // handles allocation of render command buffers
        VkCommandBuffer m_renderCmdBuffer = VK_NULL_HANDLE;                  // buffer in which we record render commands
        VkSemaphore m_renderCompleteSemaphore = VK_NULL_HANDLE;              // signals when the rendering of the frame is complete
        VkSemaphore m_imageAvailableSemaphore = VK_NULL_HANDLE;              // signals when the rendered image is ready for presentation
        VkFence m_frameFence = VK_NULL_HANDLE;                               // handles CPU / GPU synchronization for frame rendering

        // Frame timestamp objects
        VkCommandPool m_timestampCmdPool = VK_NULL_HANDLE;                   // handles allocation of timestamp command buffers
        VkCommandBuffer m_timestampCmdBuffer = VK_NULL_HANDLE;               // buffer in which we record timestamp commands
        VkQueryPool m_timestampQueryPool = VK_NULL_HANDLE;                   // handles allocation of timestamp GPU queries
        VkFence m_timestampFence = VK_NULL_HANDLE;                           // handles CPU / GPU synchronization for timestamp queries

        // Frame states
        VkPipelineState m_pipelineState;                                     // holds infos for current bound pipeline
        DynamicState m_state, m_prevState;                                   // holds infos for dynamic states (viewport, scissor)
        bool m_forceApplyState = false;                                      // flag to force update the dynamic states

        // Frame shader bindings
        std::vector<ConstantBufferBinding> m_constantBufferBindings[ShaderStage::Count]; // constant buffers bound to the pipeline
        std::vector<SampledTextureBinding> m_sampledTextureBindings[ShaderStage::Count]; // sampled textures bound to the pipeline
        std::vector<StorageBufferBinding>  m_storageBufferBindings [ShaderStage::Count]; // storage buffers bound to the pipeline
        std::vector<StorageTextureBinding> m_storageTextureBindings[ShaderStage::Count]; // storage textures bound to the pipeline

        // Frame vertex/index buffer
        VertexBufferRange m_vertexBufferRange{};                             // vertex buffer bound to the pipeline
        bool m_vertexBufferChanged = false;                                  // flag to force update the vertex buffer
        IndexBufferRange m_indexBufferRange{};                               // index buffer bound to the pipeline
        bool m_indexBufferChanged = false;                                   // flag to force update the index buffer

        // Memory barriers
        std::vector<VkBufferMemoryBarrier> m_nextBufferBarriers;             // pending buffer barriers
        std::vector<VkImageMemoryBarrier> m_nextImageBarriers;               // pending image barriers
        u32 m_nextBufferBarrierCount = 0;                                    // number of pending buffer barriers
        u32 m_nextImageBarrierCount = 0;                                     // number of pending image barriers

        // Descriptors allocated per frame
        u32 m_constantBufferDescriptorsAllocated = 0;
        u32 m_sampleTextureDescriptorsAllocated  = 0;
        u32 m_storageBufferDescriptorsAllocated  = 0;
        u32 m_storageTextureDescriptorsAllocated = 0;

        // Runtime helpers
        bool _MapFrameBuffer();
        void _ApplyDynamicStates();
        void _BindVertexIndexBuffers();
        bool _UseProgram(ShaderProgram* _program);
        void _AddMemoryBarrier();
        void _EndRenderPass();

        // Static initialization helpers
        static void _CreateInstance();
        static void _CreateSurface();
        static void _PickPhysicalDevice();
        static void _CreateLogicalDevice();
        static void _CreateAllocator();
        static void _InitTransferObjects();
        static void _InitSwapchain();
        static void _CreateFramebuffers();
        static void _SetGpuDebugInfo();
        static void _ResizeDeletionQueues();

        // Instance helpers
        static bool _ContainsExtension(const std::vector<const char*>& _extensions, const char* _extension);
        static std::vector<const char*> _GetSupportedInstanceExtensions();
        static std::vector<const char*> _GetSupportedValidationLayers();
        static VKAPI_ATTR VkBool32 VKAPI_CALL _DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT _messageSeverity, VkDebugUtilsMessageTypeFlagsEXT _messageType, const VkDebugUtilsMessengerCallbackDataEXT* _pCallbackData, void* _pUserData);

        // Device helpers
        static bool _IsDeviceSuitable(VkPhysicalDevice _device);
        static u32 _FindQueueFamily(VkPhysicalDevice _device);
        static bool _CheckDeviceExtensionSupport(VkPhysicalDevice _device);
        static bool _CheckDeviceFeatureSupport(VkPhysicalDevice _device);
        static VkSwapchainSupportDetails _QuerySwapchainSupport(VkPhysicalDevice _device);
        static int _RateDeviceSuitability(VkPhysicalDevice _device);

        // Swapchain helpers
        static VkSurfaceFormatKHR _ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& _availableFormats);
        static VkPresentModeKHR _ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& _availablePresentModes);
        static VkExtent2D _ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& _capabilities);
        static void _RecreateSwapchain();
    };

    class GraphicsContextImpl : public VkGraphicsContext
    {
        using VkGraphicsContext::VkGraphicsContext;
    };

}