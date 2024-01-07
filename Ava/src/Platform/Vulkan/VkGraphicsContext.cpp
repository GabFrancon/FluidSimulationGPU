#include <avapch.h>
#include "VkGraphicsContext.h"

#include <Math/Math.h>
#include <Math/Hash.h>
#include <Debug/Log.h>
#include <Debug/Assert.h>
#include <Debug/Capture.h>
#include <Time/Profiler.h>
#include <UI/ImGuiTools.h>
#include <Graphics/Color.h>
#include <Strings/StringBuilder.h>
#include <Application/GUIApplication.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN

#define VMA_IMPLEMENTATION
#include <VMA/vk_mem_alloc.h>
#include <GLFW/glfw3.h>

AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    static constexpr u32 kVulkanIdxInvalid = 0xFFFFFFFF;
    static bool s_enableValidationLayers = false;

    // Extension requirements
    static VkDebugUtilsMessengerEXT s_debugMessenger = VK_NULL_HANDLE;
    static constexpr char const* kInstanceExtensionsRequired[] = { VK_KHR_WIN32_SURFACE_EXTENSION_NAME, VK_KHR_SURFACE_EXTENSION_NAME };
    static constexpr char const* kInstanceExtensionsOptional[] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
    static constexpr char const* kDeviceExtensionsRequired[]   = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME };
    static constexpr char const* kValidationLayers[]           = { "VK_LAYER_KHRONOS_validation", "VK_LAYER_KHRONOS_synchronization2" };

    // Vulkan additional debug functions
    static PFN_vkCreateDebugUtilsMessengerEXT  vkCreateDebugUtilsMessengerEXT = nullptr;
    static PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = nullptr;
    static PFN_vkCmdInsertDebugUtilsLabelEXT   vkCmdInsertDebugUtilsLabelEXT = nullptr;
    static PFN_vkCmdBeginDebugUtilsLabelEXT    vkCmdBeginDebugUtilsLabelEXT = nullptr;
    static PFN_vkCmdEndDebugUtilsLabelEXT      vkCmdEndDebugUtilsLabelEXT = nullptr;
    static PFN_vkSetDebugUtilsObjectNameEXT    vkSetDebugUtilsObjectNameEXT = nullptr;

    // Transfer objects
    static VkCommandPool s_transferCmdPool = VK_NULL_HANDLE;             // handles creation of transfer command buffers
    static VkCommandBuffer s_transferCmdBuffer = VK_NULL_HANDLE;         // the buffer in which we record memory transfer commands
    static VkFence s_transferFence = VK_NULL_HANDLE;                     // handles CPU / GPU synchronization for memory transfers

    // Cached Vulkan objects
    static std::unordered_map<u32, VkSampler> s_samplerCache;            // sampler cache to avoid duplicate them across textures
    static std::unordered_map<u32, VkRenderPass> s_passCache;            // pass cache to avoid duplicate them across framebuffers
    static std::unordered_map<u32, VkDescriptorSetLayout> s_layoutCache; // layout cache to avoid duplicate them across shaders

    // Device
    static VkInstance s_vulkanInstance = VK_NULL_HANDLE;                 // Vulkan library handle
    static VkSurfaceKHR s_surface = VK_NULL_HANDLE;                      // interface between Vulkan and the window
    static VkPhysicalDevice s_GPU = VK_NULL_HANDLE;                      // GPU chosen as graphic device
    static VkPhysicalDeviceLimits s_deviceLimits{};                      // device properties and limitations
    static VkDevice s_device = VK_NULL_HANDLE;                           // interface between Vulkan and the GPU
    static VkQueue s_graphicsQueue = VK_NULL_HANDLE;                     // GPU port where we submit commands
    static VmaAllocator s_allocator = VMA_NULL;                          // Vulkan memory allocator
    static u32 s_queueFamily = kVulkanIdxInvalid;                        // family of the chosen queue
    static bool s_frameCaptureTriggered = false;                         // flags to trigger frame capture

    // Swapchain
    static VkSwapchainKHR s_swapchain = VK_NULL_HANDLE;                  // swapchain handle
    static std::vector<Texture*> s_swapchainImages{};                    // swapchain target images
    static std::vector<FrameBuffer*> s_mainFramebuffers{};               // framebuffers bound to each swapchain image
    static u32 s_swapchainImageCount = 0;                                // number of images in the swapchain
    static u32 s_currentSwapchainImage = kVulkanIdxInvalid;              // gives the swapchain image to present on screen
    static bool s_swapchainUpToDate = false;                             // flags to trigger swapchain recreation

    // Deletion queues
    static std::vector<VkDeletionQueue> s_deletionQueues;
    static u32 s_currentDeletionQueue = 0;

    void VkGraphicsContext::Init()
    {
        _CreateInstance();
        _CreateSurface();
        _PickPhysicalDevice();
        _CreateLogicalDevice();
        _CreateAllocator();
        _InitTransferObjects();
        _InitSwapchain();
        _CreateFramebuffers();
        _SetGpuDebugInfo();
        _ResizeDeletionQueues();
    }

    void VkGraphicsContext::Shutdown()
    {
        WaitIdle();

        // Destroys main framebuffers
        for (size_t i = 0; i < s_swapchainImageCount; i++)
        {
            DestroyFrameBuffer(s_mainFramebuffers[i]);
            DestroyTexture(s_swapchainImages[i]);
        }

        // Destroys remaining vk objects
        for (auto& deletionQueue : s_deletionQueues)
        {
            deletionQueue.FlushAll();
        }

        // Destroys swapchain
        vkDestroySwapchainKHR(s_device, s_swapchain, nullptr);

        // Destroys samplers from cache
        for (const auto& pair : s_samplerCache)
        {
            vkDestroySampler(GetDevice(), pair.second, nullptr);
        }
        s_samplerCache.clear();

        // Destroys render passes from cache
        for (const auto& pair : s_passCache)
        {
            vkDestroyRenderPass(GetDevice(), pair.second, nullptr);
        }
        s_passCache.clear();

        // Destroys descriptor set layouts from cache
        for (const auto& pair : s_layoutCache)
        {
            vkDestroyDescriptorSetLayout(GetDevice(), pair.second, nullptr);
        }
        s_layoutCache.clear();

        // Destroys transfer objects
        vkDestroyCommandPool(s_device, s_transferCmdPool, nullptr);
        vkDestroyFence(s_device, s_transferFence, nullptr);

        // Destroys device
        vmaDestroyAllocator(s_allocator);
        vkDestroyDevice(s_device, nullptr);
        if (vkDestroyDebugUtilsMessengerEXT)
        {
            vkDestroyDebugUtilsMessengerEXT(s_vulkanInstance, s_debugMessenger, nullptr);
        }
        vkDestroySurfaceKHR(s_vulkanInstance, s_surface, nullptr);
        vkDestroyInstance(s_vulkanInstance, nullptr);
    }

    void VkGraphicsContext::Reset()
    {
        // Reset current framebuffer
        m_state.framebuffer = nullptr;
        m_state.viewport = {0, 0, 0, 0};
        m_state.scissor = {0, 0, 0, 0};
        m_state.lineWidth = 1.f;
        m_forceApplyState = true;

        // Reset pipeline states
        m_pipelineState.vertexLayout = AVA_DISABLE_STATE;
        m_pipelineState.rasterState = AVA_DISABLE_STATE;
        m_pipelineState.depthState = AVA_DISABLE_STATE;
        m_pipelineState.blendState = AVA_DISABLE_STATE;
        m_pipelineState.renderPass = VK_NULL_HANDLE;
        m_pipelineState.colorAttachments = 0;
        m_pipelineState.primitiveType = PrimitiveType::Triangle;
        m_pipelineState.wireframeView = false;
        m_pipelineState.pipelineStateDirty = true;

        // Reset vertex buffer
        m_vertexBufferRange.buffer = nullptr;
        m_vertexBufferRange.data = nullptr;
        m_vertexBufferRange.offset = 0;
        m_vertexBufferRange.size = 0;
        m_vertexBufferChanged = true;

        // Reset index buffer
        m_indexBufferRange.buffer = nullptr;
        m_indexBufferRange.data = nullptr;
        m_indexBufferRange.offset = 0;
        m_indexBufferRange.size = 0;
        m_indexBufferChanged = true;

        // Reset shader bindings
        for (u8 stage = 0; stage < ShaderStage::Count; stage++)
        {
            for (auto& bfBinding : m_constantBufferBindings[stage])
            {
                bfBinding.bfRange.buffer = nullptr;
                bfBinding.bfRange.data = nullptr;
            }
            for (auto& txBinding : m_sampledTextureBindings[stage])
            {
                txBinding.texture = nullptr;
            }
            for (auto& bfBinding : m_storageBufferBindings[stage])
            {
                bfBinding.buffer = nullptr;
                bfBinding.data = nullptr;
            }
            for (auto& txBinding : m_storageTextureBindings[stage])
            {
                txBinding.texture = nullptr;
            }
        }
    }

    VkGraphicsContext::VkGraphicsContext(const u32 _contextId)
    {
        const auto& settings = GraphicsContext::GetSettings();

        // Creates command pools
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = s_queueFamily;

        VkResult result = vkCreateCommandPool(s_device, &poolInfo, nullptr, &m_renderCmdPool);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create render command pool: %s", GetVkResultStr(result));
        
        result = vkCreateCommandPool(s_device, &poolInfo, nullptr, &m_timestampCmdPool);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create timestamp command pool: %s", GetVkResultStr(result));

        // Creates command buffers
        VkCommandBufferAllocateInfo renderCmdBufferInfo{};
        renderCmdBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        renderCmdBufferInfo.commandPool = m_renderCmdPool;
        renderCmdBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        renderCmdBufferInfo.commandBufferCount = 1;

        result = vkAllocateCommandBuffers(s_device, &renderCmdBufferInfo, &m_renderCmdBuffer);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to allocate render command buffer: %s", GetVkResultStr(result));

        VkCommandBufferAllocateInfo timestampCmdBufferInfo{};
        timestampCmdBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        timestampCmdBufferInfo.commandPool = m_timestampCmdPool;
        timestampCmdBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        timestampCmdBufferInfo.commandBufferCount = 1;

        result = vkAllocateCommandBuffers(s_device, &timestampCmdBufferInfo, &m_timestampCmdBuffer);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to allocate timestamp command buffer: %s", GetVkResultStr(result));

        // Creates semaphores
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        result = vkCreateSemaphore(s_device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphore);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create image available semaphore: %s", GetVkResultStr(result));

        result = vkCreateSemaphore(s_device, &semaphoreInfo, nullptr, &m_renderCompleteSemaphore);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create render complete semaphore: %s", GetVkResultStr(result));

        // Creates fences
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        result = vkCreateFence(s_device, &fenceInfo, nullptr, &m_frameFence);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create frame fence: %s", GetVkResultStr(result));

        result = vkCreateFence(s_device, &fenceInfo, nullptr, &m_timestampFence);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create timestamp fence: %s", GetVkResultStr(result));

        // Creates query pools
        VkQueryPoolCreateInfo queryCreateInfo = {};
        queryCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryCreateInfo.queryCount = 1;

        result = vkCreateQueryPool(s_device, &queryCreateInfo, nullptr, &m_timestampQueryPool);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to allocate timestamp query pool: %s", GetVkResultStr(result));

        // Creates descriptor pools
        std::array<VkDescriptorPoolSize, 4> poolSizes = {};
        poolSizes[0].descriptorCount = settings.maxDescriptorPerFrame;
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[1].descriptorCount = settings.maxDescriptorPerFrame;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[2].descriptorCount = settings.maxDescriptorPerFrame;
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[3].descriptorCount = settings.maxDescriptorPerFrame;
        poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

        VkDescriptorPoolCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.poolSizeCount = (u32)poolSizes.size();
        createInfo.pPoolSizes = poolSizes.data();
        createInfo.maxSets = settings.maxDescriptorPerFrame;

        result = vkCreateDescriptorPool(s_device, &createInfo, nullptr, &m_descriptorPool);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create descriptor pool: %s", GetVkResultStr(result));

        // Allocates resource bindings
        for (u8 stage = 0; stage < ShaderStage::Count; stage++)
        {
            m_constantBufferBindings[stage].resize(settings.nbBindingSlotPerStage);
            m_sampledTextureBindings[stage].resize(settings.nbBindingSlotPerStage);
            m_storageBufferBindings[stage].resize(settings.nbBindingSlotPerStage);
            m_storageTextureBindings[stage].resize(settings.nbBindingSlotPerStage);
        }

        // Max number of storage buffers used in a single draw or dispatch.
        const u32 maxBufferBarrierCount = 2 * settings.nbBindingSlotPerStage;
        m_nextBufferBarriers.resize(maxBufferBarrierCount);

        // Max number of storage textures used in a single draw or dispatch.
        const u32 maxImageBarrierCount = 2 * settings.nbBindingSlotPerStage;
        m_nextImageBarriers.resize(maxImageBarrierCount);

        constexpr VkAccessFlags allAccessFlags =
            VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
            VK_ACCESS_INDEX_READ_BIT |
            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
            VK_ACCESS_UNIFORM_READ_BIT |
            VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT |
            VK_ACCESS_TRANSFER_WRITE_BIT |
            VK_ACCESS_HOST_READ_BIT |
            VK_ACCESS_HOST_WRITE_BIT |
            VK_ACCESS_MEMORY_READ_BIT |
            VK_ACCESS_MEMORY_WRITE_BIT;

        // Init buffer barriers
        for (u32 i = 0; i < maxBufferBarrierCount; ++i)
        {
            VkBufferMemoryBarrier& barrier = m_nextBufferBarriers[i];
            barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.srcAccessMask = allAccessFlags;
            barrier.dstAccessMask = allAccessFlags;
        }

        // Init image barriers
        for (u32 i = 0; i < maxImageBarrierCount; ++i)
        {
            VkImageMemoryBarrier& barrier = m_nextImageBarriers[i];
            barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.srcAccessMask = allAccessFlags;
            barrier.dstAccessMask = allAccessFlags;
        }

        // Attributes debug names
        StringBuilder objectName;
        objectName.clear(); objectName.appendf("GraphicsContext[%zu]::m_renderCmdPool", _contextId);
        SetDebugObjectName(m_renderCmdPool, VK_OBJECT_TYPE_COMMAND_POOL, objectName.c_str());
        objectName.clear(); objectName.appendf("GraphicsContext[%zu]::m_timestampCmdPool", _contextId);
        SetDebugObjectName(m_timestampCmdPool, VK_OBJECT_TYPE_COMMAND_POOL, objectName.c_str());
        objectName.clear(); objectName.appendf("GraphicsContext[%zu]::m_renderCmdBuffer", _contextId);
        SetDebugObjectName(m_renderCmdBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER, objectName.c_str());
        objectName.clear(); objectName.appendf("GraphicsContext[%zu]::m_timestampCmdBuffer", _contextId);
        SetDebugObjectName(m_timestampCmdBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER, objectName.c_str());
        objectName.clear(); objectName.appendf("GraphicsContext[%zu]::m_renderCompleteSemaphore", _contextId);
        SetDebugObjectName(m_renderCompleteSemaphore, VK_OBJECT_TYPE_SEMAPHORE, objectName.c_str());
        objectName.clear(); objectName.appendf("GraphicsContext[%zu]::m_imageAvailableSemaphore", _contextId);
        SetDebugObjectName(m_imageAvailableSemaphore, VK_OBJECT_TYPE_SEMAPHORE, objectName.c_str());
        objectName.clear(); objectName.appendf("GraphicsContext[%zu]::m_frameFence", _contextId);
        SetDebugObjectName(m_frameFence, VK_OBJECT_TYPE_FENCE, objectName.c_str());
        objectName.clear(); objectName.appendf("GraphicsContext[%zu]::m_timestampFence", _contextId);
        SetDebugObjectName(m_timestampFence, VK_OBJECT_TYPE_FENCE, objectName.c_str());
        objectName.clear(); objectName.appendf("GraphicsContext[%zu]::m_timestampQueryPool", _contextId);
        SetDebugObjectName(m_timestampQueryPool, VK_OBJECT_TYPE_QUERY_POOL, objectName.c_str());
        objectName.clear(); objectName.appendf("GraphicsContext[%zu]::m_descriptorPool", _contextId);
        SetDebugObjectName(m_descriptorPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL,objectName.c_str());
    }

    VkGraphicsContext::~VkGraphicsContext()
    {
        WaitIdle();
        Reset();

        // Destroys frame objects
        vkDestroyDescriptorPool(s_device, m_descriptorPool, nullptr);
        vkDestroyQueryPool(s_device, m_timestampQueryPool, nullptr);
        vkDestroyCommandPool(s_device, m_renderCmdPool, nullptr);
        vkDestroyCommandPool(s_device, m_timestampCmdPool, nullptr);
        vkDestroySemaphore(s_device, m_renderCompleteSemaphore, nullptr);
        vkDestroySemaphore(s_device, m_imageAvailableSemaphore, nullptr);
        vkDestroyFence(s_device, m_timestampFence, nullptr);
        vkDestroyFence(s_device, m_frameFence, nullptr);

        // Clears resource bindings
        for (u8 stage = 0; stage < ShaderStage::Count; stage++)
        {
            m_constantBufferBindings[stage].clear();
            m_sampledTextureBindings[stage].clear();
            m_storageBufferBindings[stage].clear();
            m_storageTextureBindings[stage].clear();
        }

        // Clears memory barriers
        m_nextBufferBarriers.clear();
        m_nextImageBarriers.clear();
    }


    // ------- Getters -----------------------------------------------------------------------------------------------

    VkDevice VkGraphicsContext::GetDevice() 
    {
        return s_device;
    }

    VkPhysicalDeviceLimits VkGraphicsContext::GetDeviceLimits() 
    {
        return s_deviceLimits;
    }

    VkCommandBuffer VkGraphicsContext::GetCommandBuffer() const
    {
        return m_renderCmdBuffer;
    }

    VmaAllocator VkGraphicsContext::GetAllocator()
    {
        return s_allocator;
    }


    // ------- Time --------------------------------------------------------------------------------------------------

    u64 VkGraphicsContext::GetGpuTimestamp() const
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VkResult result = vkBeginCommandBuffer(m_timestampCmdBuffer, &beginInfo);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to start recording GPU timestamp command: %s", GetVkResultStr(result));

        // Resets query pool before we emit timestamp query
        vkCmdResetQueryPool(m_timestampCmdBuffer, m_timestampQueryPool, 0, 1);

        // Emits GPU timestamp command
        vkCmdWriteTimestamp(m_timestampCmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, m_timestampQueryPool, 0);

        result = vkEndCommandBuffer(m_timestampCmdBuffer);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to stop recording GPU timestamp command: %s", GetVkResultStr(result));

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_timestampCmdBuffer;

        // Resets fence before we submit timestamp command
        vkResetFences(s_device, 1, &m_timestampFence);

        // Submits timestamp command
        result = vkQueueSubmit(s_graphicsQueue, 1, &submitInfo, m_timestampFence);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to submit timestamp command: %s", GetVkResultStr(result));

        result = vkWaitForFences(s_device, 1, &m_timestampFence, VK_TRUE, UINT64_MAX);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] timeout while waiting for timestamp fence: %s", GetVkResultStr(result));

        // Fetches timestamp query result
        u64 gpuTime;
        vkGetQueryPoolResults(s_device, m_timestampQueryPool, 0, 1,
            sizeof(u64), &gpuTime, sizeof(u64), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        return gpuTime;
    }

    void VkGraphicsContext::WaitIdle()
    {
        vkDeviceWaitIdle(s_device);
    }


    // ------- Frame cycle -------------------------------------------------------------------------------------------

    void VkGraphicsContext::StartFrame()
    {
        Reset();
        VkResult result;

        // Waits until the gpu has finished rendering the last frame
        {
            AUTO_CPU_MARKER("VK Wait");

            result = vkWaitForFences(s_device, 1, &m_frameFence, VK_TRUE, UINT64_MAX);
            AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] timeout while waiting for frame fence: %s", GetVkResultStr(result));
        }

        // Resets render objects
        {
            AUTO_CPU_MARKER("VK Reset");

            // Processes vk objects waiting to be destroyed
            s_currentDeletionQueue = (s_currentDeletionQueue + 1) % s_deletionQueues.size();
            s_deletionQueues[s_currentDeletionQueue].FlushAll();

            // Recreates swapchain if needed
            if (!s_swapchainUpToDate)
            {
                _RecreateSwapchain();
            }

            vkResetFences(s_device, 1, &m_frameFence);
            vkResetDescriptorPool(s_device, m_descriptorPool, 0);
            vkResetCommandBuffer(m_renderCmdBuffer, 0);
        }

        // Acquires index of the target swapchain image
        {
            AUTO_CPU_MARKER("VK Acquire");

            result = vkAcquireNextImageKHR(
                s_device, s_swapchain, UINT64_MAX, m_imageAvailableSemaphore, VK_NULL_HANDLE, &s_currentSwapchainImage);

            // If the swapchain is out of date, recreates it
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
            {
                s_swapchainUpToDate = false;
            }
            else
            {
                AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to acquire swapchain image: %s", GetVkResultStr(result));
            }
        }

        // Starts recording draw commands
        {
            AUTO_CPU_MARKER("VK Begin");

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            result = vkBeginCommandBuffer(m_renderCmdBuffer, &beginInfo);
            AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to start recording render commands: %s", GetVkResultStr(result));
        }

        // Resets descriptor statistics
        m_constantBufferDescriptorsAllocated = 0;
        m_sampleTextureDescriptorsAllocated = 0;
        m_storageBufferDescriptorsAllocated = 0;
        m_storageTextureDescriptorsAllocated = 0;
    }

    void VkGraphicsContext::EndFrameAndSubmit()
    {
        VkResult result;

        // Stops recording draw commands
        {
            AUTO_CPU_MARKER("VK End");

            // Ends current render pass
            _EndRenderPass();

            result = vkEndCommandBuffer(m_renderCmdBuffer);
            AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to end recording render commands: %s", GetVkResultStr(result));
        }

        // Starts capturing the frame
        if (s_frameCaptureTriggered)
        {
            CaptureMgr::StartFrameCapture();
        }

        // Submits draw commands
        {
            AUTO_CPU_MARKER("VK Submit");

            VkCommandBufferSubmitInfo cmdBufferSubmitInfo{};
            cmdBufferSubmitInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
            cmdBufferSubmitInfo.commandBuffer = m_renderCmdBuffer;
            cmdBufferSubmitInfo.deviceMask = 0;

            VkSemaphoreSubmitInfo waitSemaphoreInfo{};
            waitSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            waitSemaphoreInfo.semaphore = m_imageAvailableSemaphore;
            waitSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            waitSemaphoreInfo.deviceIndex = 0;
            waitSemaphoreInfo.value = 1;

            VkSemaphoreSubmitInfo signalSemaphoreInfo{};
            signalSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            signalSemaphoreInfo.semaphore = m_renderCompleteSemaphore;
            signalSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            signalSemaphoreInfo.deviceIndex = 0;
            signalSemaphoreInfo.value = 2;

            VkSubmitInfo2 submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            submitInfo.commandBufferInfoCount = 1;
            submitInfo.pCommandBufferInfos = &cmdBufferSubmitInfo;
            submitInfo.waitSemaphoreInfoCount = 1;
            submitInfo.pWaitSemaphoreInfos = &waitSemaphoreInfo;
            submitInfo.signalSemaphoreInfoCount = 1;
            submitInfo.pSignalSemaphoreInfos = &signalSemaphoreInfo;

            result = vkQueueSubmit2(s_graphicsQueue, 1, &submitInfo, m_frameFence);
            AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to submit render commands: %s", GetVkResultStr(result));
        }

        // Presents image to swapchain
        {
            AUTO_CPU_MARKER("VK Present");

            VkPresentInfoKHR presentInfo{};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &m_renderCompleteSemaphore;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &s_swapchain;
            presentInfo.pImageIndices = &s_currentSwapchainImage;

            result = vkQueuePresentKHR(s_graphicsQueue, &presentInfo);

            // If the swapchain is out of date, recreates it
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
            {
                s_swapchainUpToDate = false;
            }
            else
            {
                 AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to present rendered image: %s", GetVkResultStr(result));
            }
        }

        // Stop capturing the frame
        if (s_frameCaptureTriggered)
        {
            CaptureMgr::EndFrameCapture();
            s_frameCaptureTriggered = false;
        }
    }


    // ------- Transfer commands -------------------------------------------------------------------------------------

    VkCommandBuffer VkGraphicsContext::BeginTransferCommand()
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        const VkResult result = vkBeginCommandBuffer(s_transferCmdBuffer, &beginInfo);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to start recording transfer command: %s", GetVkResultStr(result));

        return s_transferCmdBuffer;
    }

    void VkGraphicsContext::EndTransferCommand(const VkCommandBuffer _cmd)
    {
        VkResult result = vkEndCommandBuffer(_cmd);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to end recording transfer command: %s", GetVkResultStr(result));

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &_cmd;

        result = vkQueueSubmit(s_graphicsQueue, 1, &submitInfo, s_transferFence);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to submit transfer command: %s", GetVkResultStr(result));

        result = vkWaitForFences(s_device, 1, &s_transferFence, VK_TRUE, UINT64_MAX);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] timeout while waiting for transfer fence: %s", GetVkResultStr(result));

        vkResetFences(s_device, 1, &s_transferFence);
    }


    // ------- Draw commands -----------------------------------------------------------------------------------------

    void VkGraphicsContext::Draw(ShaderProgram* _program, const u32 _instanceCount/*= 1*/)
    {
        if (!_MapFrameBuffer())
        {
            AVA_ASSERT(false, "[GraphicsContext] invalid framebuffer bound to the pipeline.");
            return;
        }

        _ApplyDynamicStates();
        _BindVertexIndexBuffers();

        if (!_UseProgram(_program))
        {
            AVA_ASSERT(false, "[GraphicsContext] invalid shader program bound to the pipeline.");
            return;
        }

        if (m_indexBufferRange.buffer)
        {
            const u32 indexCount = m_indexBufferRange.Use32BitIndices()
                ? m_indexBufferRange.size >> 2
                : m_indexBufferRange.size >> 1;

            vkCmdDrawIndexed(m_renderCmdBuffer, indexCount, _instanceCount, 0, 0, 0);
        }
        else
        {
            const u32 vertexCount = m_vertexBufferRange.GetVertexCount();
            vkCmdDraw(m_renderCmdBuffer, vertexCount, _instanceCount, 0, 0);
        }

        _AddMemoryBarrier();
    }

    void VkGraphicsContext::Draw(ShaderProgram* _program, const IndirectBufferRange& _indirectRange)
    {
        if (_indirectRange.data)
        {
            _indirectRange.UploadData();
        }

        if (!_MapFrameBuffer())
        {
            AVA_ASSERT(false, "[GraphicsContext] invalid framebuffer bound to the pipeline.");
            return;
        }

        _ApplyDynamicStates();
        _BindVertexIndexBuffers();

        if (!_UseProgram(_program))
        {
            AVA_ASSERT(false, "[GraphicsContext] invalid shader program bound to the pipeline.");
            return;
        }

        const auto gpuBuffer = ((VkIndirectBuffer*)_indirectRange.buffer)->GetVkBuffer();
        const u32 bufferOffset = _indirectRange.offset;

        if (m_indexBufferRange.buffer)
        {
            constexpr u32 bufferStride = sizeof(IndirectDrawIndexedCmd);
            // const u32 drawCount = _indirectRange.size / bufferStride;
            vkCmdDrawIndexedIndirect(m_renderCmdBuffer, gpuBuffer, bufferOffset, 1, bufferStride);
        }
        else
        {
            constexpr u32 bufferStride = sizeof(IndirectDrawCmd);
            // const u32 drawCount = _indirectRange.size / sizeof(IndirectDrawCmd);
            vkCmdDrawIndirect(m_renderCmdBuffer, gpuBuffer, bufferOffset, 1, bufferStride);
        }

        _AddMemoryBarrier();
    }


    // ------- Dispatch commands -------------------------------------------------------------------------------------

    void VkGraphicsContext::Dispatch(ShaderProgram* _program, const u32 _groupCountX, const u32 _groupCountY, const u32 _groupCountZ)
    {
        _EndRenderPass();

        if (!_UseProgram(_program))
        {
            AVA_ASSERT(false, "[GraphicsContext] invalid shader program bound to the pipeline.");
            return;
        }

        vkCmdDispatch(m_renderCmdBuffer, _groupCountX, _groupCountY, _groupCountZ);
        _AddMemoryBarrier();
    }

    void VkGraphicsContext::Dispatch(ShaderProgram* _program, const IndirectBufferRange& _indirectRange)
    {
        _EndRenderPass();

        if (_indirectRange.data)
        {
            _indirectRange.UploadData();
        }

        if (!_UseProgram(_program))
        {
            AVA_ASSERT(false, "[GraphicsContext] invalid shader program bound to the pipeline.");
            return;
        }

        const auto gpuBuffer = ((VkIndirectBuffer*)_indirectRange.buffer)->GetVkBuffer();
        const auto bufferOffset = _indirectRange.offset;

        vkCmdDispatchIndirect(m_renderCmdBuffer, gpuBuffer, bufferOffset);
        _AddMemoryBarrier();
    }


    // ------- Constant buffers --------------------------------------------------------------------------------------

    u32 VkGraphicsContext::GetConstantBufferAlignment(const u32 _flags/*= 0*/)
    {
        if (_flags & AVA_BUFFER_READ_WRITE)
        {
            return GetStorageBufferAlignment();
        }
        return (u32)s_deviceLimits.minUniformBufferOffsetAlignment;
    }

    ConstantBuffer* VkGraphicsContext::CreateConstantBuffer(const u32 _size, const u32 _flags/*= 0*/)
    {
        return new VkConstantBuffer(_size, _flags);
    }

    void VkGraphicsContext::SetConstantBuffer(const ShaderStage::Enum _stage, const u8 _slot, const ConstantBufferRange& _bfRange)
    {
        if (_bfRange.data)
        {
            _bfRange.UploadData();
        }

        ConstantBufferBinding& binding = m_constantBufferBindings[_stage][_slot];
        binding.bfRange = _bfRange;
    }

    void VkGraphicsContext::DestroyBuffer(ConstantBuffer* _buffer)
    {
        s_deletionQueues[s_currentDeletionQueue].constantBuffersToDelete.Add((VkConstantBuffer*)_buffer);
    }


    // ------- Indirect buffers --------------------------------------------------------------------------------------

    u32 VkGraphicsContext::GetIndirectBufferAlignment(const u32 _flags/*= 0*/)
    {
        if (_flags & AVA_BUFFER_READ_WRITE)
        {
            return (u32)s_deviceLimits.minStorageBufferOffsetAlignment;
        }
        return 4u;
    }

    IndirectBuffer* VkGraphicsContext::CreateIndirectBuffer(const u32 _size, const u32 _flags/*= 0*/)
    {
        return new VkIndirectBuffer(_size, _flags);
    }

    void VkGraphicsContext::DestroyBuffer(IndirectBuffer* _buffer)
    {
        s_deletionQueues[s_currentDeletionQueue].indirectBuffersToDelete.Add((VkIndirectBuffer*)_buffer);
    }


    // ------- Vertex buffers ----------------------------------------------------------------------------------------

    u32 VkGraphicsContext::GetVertexBufferAlignment(const u32 _flags/*= 0*/)
    {
        if (_flags & AVA_BUFFER_READ_WRITE)
        {
            return GetStorageBufferAlignment();
        }
        return 4u;
    }

    VertexBuffer* VkGraphicsContext::CreateVertexBuffer(const u32 _size, const u32 _flags/*= 0*/)
    {
        return new VkVertexBuffer(_size, _flags);
    }

    VertexBuffer* VkGraphicsContext::CreateVertexBuffer(const VertexLayout& _vertexLayout, const u32 _vertexCount, const u32 _flags/*= 0*/)
    {
        return new VkVertexBuffer(_vertexLayout, _vertexCount, _flags);
    }

    void VkGraphicsContext::SetVertexBuffer(const VertexBufferRange& _bfRange)
    {
        if (_bfRange.data)
        {
            _bfRange.UploadData();
        }

        if (_bfRange != m_vertexBufferRange)
        {
            m_vertexBufferRange = _bfRange;
            m_vertexBufferChanged = true;
        }

        if (_bfRange.GetVertexLayout() != m_pipelineState.vertexLayout)
        {
            m_pipelineState.vertexLayout = _bfRange.GetVertexLayout();
            m_pipelineState.pipelineStateDirty = true;
        }
    }

    void VkGraphicsContext::DestroyBuffer(VertexBuffer* _buffer)
    {
        s_deletionQueues[s_currentDeletionQueue].vertexBuffersToDelete.Add((VkVertexBuffer*)_buffer);
    }


    // ------- Index buffers -----------------------------------------------------------------------------------------

    u32 VkGraphicsContext::GetIndexBufferAlignment(const u32 _flags/*= 0*/)
    {
        if (_flags & AVA_BUFFER_READ_WRITE)
        {
            return GetStorageBufferAlignment();
        }
        return 4u;
    }

    IndexBuffer* VkGraphicsContext::CreateIndexBuffer(const u32 _indexCount, const u32 _flags/*= 0*/)
    {
        return new VkIndexBuffer(_indexCount, _flags);
    }

    void VkGraphicsContext::SetIndexBuffer(const IndexBufferRange& _bfRange)
    {
        if (_bfRange.data)
        {
            _bfRange.UploadData();
        }

        if (_bfRange != m_indexBufferRange)
        {
            m_indexBufferRange = _bfRange;
            m_indexBufferChanged = true;
        }
    }

    void VkGraphicsContext::SetStorageBuffer(const ShaderStage::Enum _stage, const u8 _slot, const IndexBufferRange& _bfRange, const Access::Enum _access)
    {
        StorageBufferBinding& binding = m_storageBufferBindings[_stage][_slot];
        binding.bufferType = BufferType::Index;
        binding.buffer = _bfRange.buffer;
        binding.data = _bfRange.data;
        binding.size = _bfRange.size;
        binding.offset = _bfRange.offset;
        binding.access = _access;
    }

    void VkGraphicsContext::DestroyBuffer(IndexBuffer* _buffer)
    {
        s_deletionQueues[s_currentDeletionQueue].indexBuffersToDelete.Add((VkIndexBuffer*)_buffer);
    }


    // ------- Textures ----------------------------------------------------------------------------------------------

    Texture* VkGraphicsContext::CreateTexture(const TextureDescription& _desc)
    {
        return new VkTexture(_desc);
    }

    Texture* VkGraphicsContext::CreateTexture(const char* _path, const u32 _flags/*= 0*/)
    {
        return new VkTexture(_path, _flags);
    }

    Texture* VkGraphicsContext::CreateTexture(const TextureData& _data, const u32 _flags/*= 0*/)
    {
        return new VkTexture(_data, _flags);
    }

    Texture* VkGraphicsContext::ConstructTexture(void* _memory, const char* _path, const u32 _flags/*= 0*/)
    {
        return new (_memory) VkTexture(_path, _flags);
    }

    VkSampler VkGraphicsContext::CreateSampler(const VkSamplerCreateInfo* _info)
    {
        // Hashes sampler data to a 32 bits uint.
        u32 hash = HashU32(_info->minFilter);
        hash = HashU32Combine(_info->magFilter, hash);
        hash = HashU32Combine(_info->minFilter, hash);
        hash = HashU32Combine(_info->addressModeU, hash);
        hash = HashU32Combine(_info->addressModeV, hash);
        hash = HashU32Combine(_info->addressModeW, hash);
        hash = HashU32Combine(_info->mipmapMode, hash);
        hash = HashU32Combine(_info->minLod, hash);
        hash = HashU32Combine(_info->maxLod, hash);
        hash = HashU32Combine(_info->mipLodBias, hash);
        hash = HashU32Combine(_info->maxAnisotropy, hash);
        hash = HashU32Combine((int)_info->anisotropyEnable, hash);

        // If sampler doesn't exist, creates one and adds it to the cache.
        if (s_samplerCache.find(hash) == s_samplerCache.end())
        {
            VkSampler sampler{};
            const VkResult result = vkCreateSampler(GetDevice(), _info, nullptr, &sampler);
            AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create sampler: %s", GetVkResultStr(result));
            s_samplerCache[hash] = sampler;
        }

        return s_samplerCache[hash];
    }

    void VkGraphicsContext::SetTexture(const ShaderStage::Enum _stage, const u8 _slot, Texture* _texture)
    {
        SampledTextureBinding& binding = m_sampledTextureBindings[_stage][_slot];
        binding.texture = reinterpret_cast<VkTexture*>(_texture);
    }

    void VkGraphicsContext::DestructTexture(Texture* _texture)
    {
        ((VkTexture*)_texture)->~VkTexture();
    }

    void VkGraphicsContext::DestroyTexture(Texture* _texture)
    {
        s_deletionQueues[s_currentDeletionQueue].texturesToDelete.Add((VkTexture*)_texture);
    }


    // ------- Storage resources -------------------------------------------------------------------------------------

    template <class Buffer>
    static VkBuffer UpdateAndGetStorageBuffer(Buffer* _buffer, void*& _srcData, u32 _offset, u32 _size)
    {
        if (_srcData)
        {
            void* data = _buffer->Map(_offset, _size);
            memcpy(data, _srcData, _size);
            _buffer->Unmap();
            _srcData = nullptr;
        }

        return _buffer->GetVkBuffer();
    }

    u32 VkGraphicsContext::GetStorageBufferAlignment()
    {
        return (u32)s_deviceLimits.minStorageBufferOffsetAlignment;
    }

    void VkGraphicsContext::SetStorageBuffer(const ShaderStage::Enum _stage, const u8 _slot, const ConstantBufferRange& _bfRange, const Access::Enum _access)
    {
        StorageBufferBinding& binding = m_storageBufferBindings[_stage][_slot];
        binding.bufferType = BufferType::Constant;
        binding.buffer = _bfRange.buffer;
        binding.data = _bfRange.data;
        binding.size = _bfRange.size;
        binding.offset = _bfRange.offset;
        binding.access = _access;
    }

    void VkGraphicsContext::SetStorageBuffer(const ShaderStage::Enum _stage, const u8 _slot, const IndirectBufferRange& _bfRange, const Access::Enum _access)
    {
        StorageBufferBinding& binding = m_storageBufferBindings[_stage][_slot];
        binding.bufferType = BufferType::Indirect;
        binding.buffer = _bfRange.buffer;
        binding.data = _bfRange.data;
        binding.size = _bfRange.size;
        binding.offset = _bfRange.offset;
        binding.access = _access;
    }

    void VkGraphicsContext::SetStorageBuffer(const ShaderStage::Enum _stage, const u8 _slot, const VertexBufferRange& _bfRange, const Access::Enum _access)
    {
        StorageBufferBinding& binding = m_storageBufferBindings[_stage][_slot];
        binding.bufferType = BufferType::Vertex;
        binding.buffer = _bfRange.buffer;
        binding.data = _bfRange.data;
        binding.size = _bfRange.size;
        binding.offset = _bfRange.offset;
        binding.access = _access;
    }

    void VkGraphicsContext::SetStorageTexture(const ShaderStage::Enum _stage, const u8 _slot, Texture* _texture, const Access::Enum _access, const u16 _mip/*= 0*/)
    {
        StorageTextureBinding& binding = m_storageTextureBindings[_stage][_slot];
        binding.texture = reinterpret_cast<VkTexture*>(_texture);
        binding.access = _access;
        binding.mip = _mip;
    }


    // ------- Framebuffers ------------------------------------------------------------------------------------------

    FrameBuffer* VkGraphicsContext::GetMainFramebuffer() 
    {
        return s_mainFramebuffers[s_currentSwapchainImage];
    }

    const FrameBuffer* VkGraphicsContext::GetCurrentFramebuffer() const
    {
        return m_state.framebuffer;
    }

    FrameBuffer* VkGraphicsContext::CreateFrameBuffer(const FrameBufferDescription& _desc)
    {
        return new VkFrameBuffer(_desc);
    }

    VkRenderPass VkGraphicsContext::CreateRenderPass(const VkRenderPassCreateInfo* _info)
    {
        // Hashes render pass data to a 32 bits uint.
        u32 hash = HashU32Init();
        for (u32 i = 0; i < _info->attachmentCount; i++)
        {
            hash = HashU32Combine(_info->pAttachments[i].format, hash);
            hash = HashU32Combine(_info->pAttachments[i].samples, hash);
            hash = HashU32Combine(_info->pAttachments[i].initialLayout, hash);
            hash = HashU32Combine(_info->pAttachments[i].finalLayout, hash);
        }

        // If sampler doesn't exist, creates one and adds it to the cache.
        if (s_passCache.find(hash) == s_passCache.end())
        {
            VkRenderPass renderPass{};
            const VkResult result = vkCreateRenderPass(GetDevice(), _info, nullptr, &renderPass);
            AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create render pass: %s", GetVkResultStr(result));
            s_passCache[hash] = renderPass;
        }

        return s_passCache[hash];
    }

    void VkGraphicsContext::SetFramebuffer(FrameBuffer* _framebuffer/*= nullptr*/)
    {
        // If no framebuffer provided
        if (!_framebuffer)
        {
            // binds the main framebuffer
            m_state.framebuffer = GetMainFramebuffer();
            // updates the dynamic states to fit the main framebuffer extents
            SetViewport(m_state.framebuffer->GetWidth(), m_state.framebuffer->GetHeight());
        }
        else
        {
            m_state.framebuffer = _framebuffer;
        }

        const auto* framebuffer = (VkFrameBuffer*)m_state.framebuffer;
        m_pipelineState.renderPass = framebuffer->GetRenderPass();
        m_pipelineState.colorAttachments = framebuffer->GetColorAttachmentCount();
        m_pipelineState.pipelineStateDirty = true;
    }

    void VkGraphicsContext::Clear(const Color* _color, const float* _depth)
    {
        if (!m_state.framebuffer)
        {
            return;
        }

        if (!_MapFrameBuffer())
        {
            AVA_CORE_ERROR("[GraphicsContext] invalid render pass.");
            return;
        }
        std::vector<VkClearAttachment> clearAttachments;
        
        // clear color attachments
        if (_color && m_state.framebuffer)
        {
            for (u32 i = 0; i < m_state.framebuffer->GetColorAttachmentCount(); i++)
            {
                if (m_state.framebuffer->GetColorAttachment(i))
                {
                    const Color color = *_color;
                    VkClearAttachment colorClearAttachment{};
                    colorClearAttachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    colorClearAttachment.colorAttachment = i;
                    colorClearAttachment.clearValue.color = { color.r, color.g, color.b, color.a };
                    clearAttachments.push_back(colorClearAttachment);
                }
            }
        }
        // cleat depth stencil attachment
        if (_depth && m_state.framebuffer)
        {
            if (m_state.framebuffer->GetDepthAttachment())
            {
                VkClearAttachment depthStencilClearAttachment{};
                depthStencilClearAttachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                depthStencilClearAttachment.clearValue.depthStencil = { *_depth, 0 };
                clearAttachments.push_back(depthStencilClearAttachment);
            }
        }
        const u32 attachmentCount = static_cast<u32>(clearAttachments.size());
        if (attachmentCount == 0)
        {
            AVA_CORE_ERROR("[GraphicsContext] trying to clear a framebuffer that doesn't have the requested attachments.");
            return;
        }

        const auto* framebuffer = (VkFrameBuffer*)m_state.framebuffer;

        VkClearRect rect{};
        rect.rect.offset = { 0, 0 };
        rect.rect.extent = framebuffer->GetExtent();
        rect.baseArrayLayer = 0;
        rect.layerCount = 1;

        vkCmdClearAttachments(m_renderCmdBuffer, attachmentCount, clearAttachments.data(), 1, &rect);
    }

    void VkGraphicsContext::DestroyFrameBuffer(FrameBuffer* _framebuffer)
    {
        s_deletionQueues[s_currentDeletionQueue].framebuffersToDelete.Add((VkFrameBuffer*)_framebuffer);
    }


    // ------- Shaders -----------------------------------------------------------------------------------------------

    Shader* VkGraphicsContext::CreateShader(const ShaderStage::Enum _stage, const ShaderDescription& _desc)
    {
        return new VkShader(_stage, _desc);
    }

    Shader* VkGraphicsContext::CreateShader(const ShaderData& _data)
    {
        return new VkShader(_data);
    }

    Shader* VkGraphicsContext::ConstructShader(void* _memory, const ShaderStage::Enum _stage, const ShaderDescription& _desc)
    {
        return new (_memory) VkShader(_stage, _desc);
    }

    VkDescriptorSetLayout VkGraphicsContext::CreateShaderLayout(const VkDescriptorSetLayoutCreateInfo* _info)
    {
        // Hashes binding data to a 32 bits uint.
        u32 hash = HashU32Init();
        for (u32 i = 0; i < _info->bindingCount; i++)
        {
            hash = HashU32Combine(_info->pBindings[i].binding, hash);
            hash = HashU32Combine(_info->pBindings[i].descriptorType, hash);
            hash = HashU32Combine(_info->pBindings[i].descriptorCount, hash);
            hash = HashU32Combine(_info->pBindings[i].stageFlags, hash);
        }

        // If layout doesn't exist, creates one and adds it to the cache.
        if (s_layoutCache.find(hash) == s_layoutCache.end())
        {
            VkDescriptorSetLayout layout{};
            const VkResult result = vkCreateDescriptorSetLayout(GetDevice(), _info, nullptr, &layout);
            AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create descriptor set layout: %s", GetVkResultStr(result));
            s_layoutCache[hash] = layout;
        }

        return s_layoutCache[hash];
    }

    void VkGraphicsContext::DestructShader(Shader* _shader)
    {
        ((VkShader*)_shader)->~VkShader();
    }

    void VkGraphicsContext::DestroyShader(Shader* _shader)
    {
        s_deletionQueues[s_currentDeletionQueue].shadersToDelete.Add((VkShader*)_shader);
    }


    // ------- Shader programs ---------------------------------------------------------------------------------------

    ShaderProgram* VkGraphicsContext::CreateProgram(const ShaderProgramDescription& _desc)
    {
        return new VkShaderProgram(_desc);
    }

    ShaderProgram* VkGraphicsContext::CreateProgram(Shader* _stages[ShaderStage::Count], const ShaderResourceNames& _varNames)
    {
        return new VkShaderProgram(_stages, _varNames);
    }

    ShaderProgram* VkGraphicsContext::ConstructProgram(void* _memory, const ShaderProgramDescription& _desc)
    {
        return new (_memory) VkShaderProgram(_desc);
    }

    void VkGraphicsContext::DestructProgram(ShaderProgram* _program)
    {
        ((VkShaderProgram*)_program)->~VkShaderProgram();
    }

    void VkGraphicsContext::DestroyProgram(ShaderProgram* _program)
    {
        s_deletionQueues[s_currentDeletionQueue].programsToDelete.Add((VkShaderProgram*)_program);
    }


    // ------- States ------------------------------------------------------------------------------------------------

    void VkGraphicsContext::SetPrimitiveType(const PrimitiveType::Enum _type)
    {
        m_pipelineState.primitiveType = _type;
        m_pipelineState.pipelineStateDirty = true;
    }

    void VkGraphicsContext::SetRasterState(const RasterStateID _rasterState)
    {
        m_pipelineState.rasterState = _rasterState;
        m_pipelineState.pipelineStateDirty = true;
    }

    void VkGraphicsContext::SetDepthState(const DepthStateID _depthState)
    {
        m_pipelineState.depthState = _depthState;
        m_pipelineState.pipelineStateDirty = true;
    }

    void VkGraphicsContext::SetBlendState(const BlendStateID _blendState)
    {
        m_pipelineState.blendState = _blendState;
        m_pipelineState.pipelineStateDirty = true;
    }

    void VkGraphicsContext::SetWireframe(const bool _enabled)
    {
        m_pipelineState.wireframeView = _enabled;
        m_pipelineState.pipelineStateDirty = true;
    }

    void VkGraphicsContext::SetLineWidth(const float _lineWidth)
    {
        m_state.lineWidth = _lineWidth;
    }

    void VkGraphicsContext::SetViewport(const u16 _width, const u16 _height, const u16 _x/*= 0*/, const u16 _y/*= 0*/)
    {
        m_state.viewport.width = _width;
        m_state.viewport.height = _height;
        m_state.viewport.x = _x;
        m_state.viewport.y = _y;

        // For now we always want the scissor to be aligned with the viewport.
        SetScissor(_width, _height, _x, _y);
    }

    void VkGraphicsContext::SetScissor(const u16 _width, const u16 _height, const u16 _x/*= 0*/, const u16 _y/*= 0*/)
    {
        m_state.scissor.width = _width;
        m_state.scissor.height = _height;
        m_state.scissor.x = _x;
        m_state.scissor.y = _y;
    }


    // ------- Debug options -----------------------------------------------------------------------------------------

    void VkGraphicsContext::EnableGraphicsDebug(const bool _enable)
    {
    #if !defined(AVA_FINAL)
        // For vulkan, graphics debug = validation layers.
        s_enableValidationLayers = _enable;
    #endif
    }

    void VkGraphicsContext::ForceRecreateFramebuffer()
    {
    #if !defined(AVA_FINAL)
        s_swapchainUpToDate = false;
    #endif
    }

    void VkGraphicsContext::TriggerFrameCapture()
    {
    #if !defined(AVA_FINAL)
        s_frameCaptureTriggered = true;
    #endif
    }

    const char* VkGraphicsContext::GetVkResultStr(const VkResult _result)
    {
    #if !defined(AVA_FINAL)
        switch (_result)
        {
            case VK_SUCCESS : return "VK_SUCCESS";
            case VK_NOT_READY : return "VK_NOT_READY";
            case VK_TIMEOUT : return "VK_TIMEOUT";
            case VK_EVENT_SET : return "VK_EVENT_SET";
            case VK_EVENT_RESET : return "VK_EVENT_RESET";
            case VK_INCOMPLETE : return "VK_INCOMPLETE";
            case VK_ERROR_OUT_OF_HOST_MEMORY : return "VK_ERROR_OUT_OF_HOST_MEMORY";
            case VK_ERROR_OUT_OF_DEVICE_MEMORY : return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
            case VK_ERROR_INITIALIZATION_FAILED : return "VK_ERROR_INITIALIZATION_FAILED";
            case VK_ERROR_DEVICE_LOST : return "VK_ERROR_DEVICE_LOST";
            case VK_ERROR_MEMORY_MAP_FAILED : return "VK_ERROR_MEMORY_MAP_FAILED";
            case VK_ERROR_LAYER_NOT_PRESENT : return "VK_ERROR_LAYER_NOT_PRESENT";
            case VK_ERROR_EXTENSION_NOT_PRESENT : return "VK_ERROR_EXTENSION_NOT_PRESENT";
            case VK_ERROR_FEATURE_NOT_PRESENT : return "VK_ERROR_FEATURE_NOT_PRESENT";
            case VK_ERROR_INCOMPATIBLE_DRIVER : return "VK_ERROR_INCOMPATIBLE_DRIVER";
            case VK_ERROR_TOO_MANY_OBJECTS : return "VK_ERROR_TOO_MANY_OBJECTS";
            case VK_ERROR_FORMAT_NOT_SUPPORTED : return "VK_ERROR_FORMAT_NOT_SUPPORTED";
            case VK_ERROR_FRAGMENTED_POOL : return "VK_ERROR_FRAGMENTED_POOL";
            case VK_ERROR_OUT_OF_POOL_MEMORY : return "VK_ERROR_OUT_OF_POOL_MEMORY";
            case VK_ERROR_INVALID_EXTERNAL_HANDLE : return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
            case VK_ERROR_SURFACE_LOST_KHR : return "VK_ERROR_SURFACE_LOST_KHR";
            case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR : return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
            case VK_SUBOPTIMAL_KHR : return "VK_SUBOPTIMAL_KHR";
            case VK_ERROR_OUT_OF_DATE_KHR : return "VK_ERROR_OUT_OF_DATE_KHR";
            case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR : return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
            case VK_ERROR_VALIDATION_FAILED_EXT : return "VK_ERROR_VALIDATION_FAILED_EXT";
            case VK_ERROR_INVALID_SHADER_NV : return "VK_ERROR_INVALID_SHADER_NV";
            case VK_ERROR_FRAGMENTATION_EXT : return "VK_ERROR_FRAGMENTATION_EXT";
            case VK_ERROR_NOT_PERMITTED_EXT : return "VK_ERROR_NOT_PERMITTED_EXT";
            default: return "Invalid Vulkan Result";
        }
    #else
        return "";
    #endif
    }

    void VkGraphicsContext::SetDebugObjectName(void* _object, const VkObjectType _objectType, const char* _name)
    {
    #if !defined(AVA_FINAL)
        if (vkSetDebugUtilsObjectNameEXT)
        {
            VkDebugUtilsObjectNameInfoEXT nameInfo = {};
            nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            nameInfo.objectType = _objectType;
            nameInfo.objectHandle = (u64)_object;
            nameInfo.pObjectName = _name;
            vkSetDebugUtilsObjectNameEXT(s_device, &nameInfo);
        }
    #endif
    }

    void VkGraphicsContext::AddDebugMarker(const char* _label, const Color& _color) const
    {
    #if !defined(AVA_FINAL)
        if (vkCmdInsertDebugUtilsLabelEXT)
        {
            VkDebugUtilsLabelEXT labelInfo = {};
            labelInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
            labelInfo.pLabelName = _label;

            const float rawColor[4] = { _color.r, _color.g, _color.b, _color.a };
            memcpy(labelInfo.color, &rawColor[0], sizeof(float) * 4);

            vkCmdInsertDebugUtilsLabelEXT(m_renderCmdBuffer, &labelInfo);
        }
    #endif
    }

    void VkGraphicsContext::BeginDebugMarkerRegion(const char* _label, const Color& _color) const
    {
    #if !defined(AVA_FINAL)
        if (vkCmdBeginDebugUtilsLabelEXT)
        {
            VkDebugUtilsLabelEXT labelInfo = {};
            labelInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
            labelInfo.pLabelName = _label;

            const float rawColor[4] = { _color.r, _color.g, _color.b, _color.a };
            memcpy(labelInfo.color, &rawColor[0], sizeof(float) * 4);

            vkCmdBeginDebugUtilsLabelEXT(m_renderCmdBuffer, &labelInfo);
        }
    #endif
    }

    void VkGraphicsContext::EndDebugMarkerRegion() const
    {
    #if !defined(AVA_FINAL)
        if (vkCmdEndDebugUtilsLabelEXT)
        {
            vkCmdEndDebugUtilsLabelEXT(m_renderCmdBuffer);
        }
    #endif
    }

    void VkGraphicsContext::DisplayDebug() const
    {
        if (!AVA_VERIFY(ImGuiTools::WithinFrameScope(), 
            "[GraphicsContext] DisplayDebug() can only be called within ImGui frame scope."))
        {
            return;
        }

        ImGui::Text("Graphics API : Vulkan 1.3");
        ImGui::Spacing();

        if (ImGui::TreeNodeEx("Descriptors"))
        {
            ImGui::Text("Constant buffers : %u", m_constantBufferDescriptorsAllocated);
            ImGui::Text("Sampled textures : %u", m_sampleTextureDescriptorsAllocated);
            ImGui::Text("Storage buffers  : %u", m_storageBufferDescriptorsAllocated);
            ImGui::Text("Storage textures : %u", m_storageTextureDescriptorsAllocated);
            ImGui::TreePop();
        }
    }


    // ------- Runtime helpers ---------------------------------------------------------------------------------------

    bool VkGraphicsContext::_MapFrameBuffer()
    {
        const auto* framebuffer = (VkFrameBuffer*)m_state.framebuffer;

        // Checks if current framebuffer is valid
        if (framebuffer
            && framebuffer->GetRenderPass()
            && framebuffer->GetFramebuffer())
        {
            // Updates only if the framebuffer has changed
            if (m_state.framebuffer != m_prevState.framebuffer)
            {
                // End previous pass
                _EndRenderPass();

                // Begin new pass
                VkRenderPassBeginInfo renderPassInfo{};
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.renderPass = framebuffer->GetRenderPass();
                renderPassInfo.framebuffer = framebuffer->GetFramebuffer();
                renderPassInfo.renderArea.offset = { 0, 0 };
                renderPassInfo.renderArea.extent = framebuffer->GetExtent();

                vkCmdBeginRenderPass(m_renderCmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                m_prevState.framebuffer = m_state.framebuffer;
                m_forceApplyState = true;
                m_vertexBufferChanged = true;
                m_indexBufferChanged = true;
            }
            return true;
        }
        return false;
    }

    void VkGraphicsContext::_ApplyDynamicStates()
    {
        // Updates viewport if it was changed
        if (m_state.viewport != m_prevState.viewport || m_forceApplyState)
        {
            VkViewport viewport{};
            viewport.x = (float)m_state.viewport.x;
            viewport.y = (float)m_state.viewport.y;
            viewport.width = (float)m_state.viewport.width;
            viewport.height = (float)m_state.viewport.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            vkCmdSetViewport(m_renderCmdBuffer, 0, 1, &viewport);
            m_prevState.viewport = m_state.viewport;
        }

        // Updates scissor if it was changed
        if (m_state.scissor != m_prevState.scissor || m_forceApplyState)
        {
            VkRect2D scissor{};
            scissor.offset = { (int32_t)m_state.scissor.x, (int32_t)m_state.scissor.y };
            scissor.extent = { m_state.scissor.width, m_state.scissor.height };

            vkCmdSetScissor(m_renderCmdBuffer, 0, 1, &scissor);
            m_prevState.scissor = m_state.scissor;
        }

        // Updates line width if it was changed
        if (fabs(m_state.lineWidth - m_prevState.lineWidth) < FLT_EPSILON || m_forceApplyState)
        {
            vkCmdSetLineWidth(m_renderCmdBuffer, m_state.lineWidth);
            m_prevState.lineWidth = m_state.lineWidth;
        }

        m_forceApplyState = false;
    }

    void VkGraphicsContext::_BindVertexIndexBuffers()
    {
        if (m_vertexBufferChanged)
        {
            if (const auto* buffer = (VkVertexBuffer*)m_vertexBufferRange.buffer)
            {
                const VkBuffer vkBuffer = buffer->GetVkBuffer();
                const VkDeviceSize offset = m_vertexBufferRange.offset;
                vkCmdBindVertexBuffers(m_renderCmdBuffer, 0, 1, &vkBuffer, &offset);
            }
            m_vertexBufferChanged = false;
        }

        if (m_indexBufferChanged)
        {
            if (const auto* buffer = (VkIndexBuffer*)m_indexBufferRange.buffer)
            {
                const VkBuffer vkBuffer = buffer->GetVkBuffer();
                const VkDeviceSize offset = m_indexBufferRange.offset;
                const VkIndexType indexType = m_indexBufferRange.Use32BitIndices() ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
                vkCmdBindIndexBuffer(m_renderCmdBuffer, vkBuffer, offset, indexType);
            }
            m_indexBufferChanged = false;
        }
    }

    bool VkGraphicsContext::_UseProgram(ShaderProgram* _program)
    {
        auto* program = (VkShaderProgram*)_program;

        if (!_program || _program->GetResourceState() != ResourceState::Loaded) {
            return false;
        }

        // Binds pipeline
        const VkPipeline pipeline = program->IsCompute() ? program->GetComputePipeline() : program->GetGraphicsPipeline(m_pipelineState);
        VkPipelineLayout pipelineLayout = program->IsCompute() ? program->GetComputeLayout() : program->GetGraphicsLayout();
        const VkPipelineBindPoint pipelineBindPoint = program->IsCompute() ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;

        vkCmdBindPipeline(m_renderCmdBuffer, pipelineBindPoint, pipeline);

        // Allocates descriptor sets
        std::vector<VkDescriptorSet> descriptorSets{};
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts{};

        program->GetDescriptorSetLayouts(descriptorSetLayouts);
        descriptorSets.resize(descriptorSetLayouts.size());

        VkDescriptorSetAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = m_descriptorPool;
        allocateInfo.descriptorSetCount = (u32)descriptorSetLayouts.size();
        allocateInfo.pSetLayouts = descriptorSetLayouts.data();

        const VkResult result = vkAllocateDescriptorSets(s_device, &allocateInfo, descriptorSets.data());
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to allocate descriptor sets: %s", GetVkResultStr(result));

        // Prepares descriptor containers
        m_bufferInfoCache.clear();
        m_bufferInfoCache.resize(program->GetBufferBindingCount());
        u32 bufferCount = 0;

        m_imageInfoCache.clear();
        m_imageInfoCache.resize(program->GetTextureBindingCount());
        u32 textureCount = 0;

        m_writeDescriptorCache.clear();
        m_writeDescriptorCache.resize(program->GetBindingCount());
        u32 descriptorCount = 0;

        m_nextBufferBarrierCount = 0;
        m_nextImageBarrierCount = 0;

        const auto& varNames = _program->GetShaderVarNames();
        bool errored = false;

        // Builds descriptors
        for (u8 i = 0; i < ShaderStage::Count; i++)
        {
            const auto stage = ShaderStage::Enum(i);
            const auto* shader = (VkShader*)_program->GetShader(stage);

            if (!shader) {
                continue;
            }

            const int setIndex = VkShaderProgram::GetDescriptorSetIndex(stage);
            const VkDescriptorSet& descriptorSet = descriptorSets[setIndex];

            // Binds constant buffers
            for (const auto& shaderBinding : program->m_bufferBindings[stage])
            {
                const GpuBufferRange& bufferRange = m_constantBufferBindings[stage][shaderBinding.slot].bfRange;
                const auto gpuBuffer = (VkConstantBuffer*)bufferRange.buffer;

                if (!AVA_VERIFY(gpuBuffer && bufferRange.size > 0,"Missing uniform buffer '%s' for shader '%s'.",
                    varNames.GetConstantBufferName(stage, shaderBinding.slot), shader->GetResourcePath()))
                {
                    errored = true;
                    continue;
                }

                VkDescriptorBufferInfo& bufferInfo = m_bufferInfoCache[bufferCount];
                bufferInfo.buffer = gpuBuffer->GetVkBuffer();
                bufferInfo.offset = bufferRange.offset;
                bufferInfo.range = bufferRange.size;
                bufferCount++;

                // Creates the descriptor write
                VkWriteDescriptorSet& newWrite = m_writeDescriptorCache[descriptorCount];
                newWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                newWrite.pNext = nullptr;
                newWrite.descriptorCount = 1;
                newWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                newWrite.pBufferInfo = &bufferInfo;
                newWrite.dstBinding = shaderBinding.binding;
                newWrite.dstSet = descriptorSet;
                descriptorCount++;

                m_constantBufferDescriptorsAllocated++;
            }

            // Binds sampled textures
            for (const auto& shaderBinding : program->m_textureBindings[stage])
            {
                const SampledTextureBinding& texBinding = m_sampledTextureBindings[stage][shaderBinding.slot];
                const auto texture = (VkTexture*)texBinding.texture;

                if (!AVA_VERIFY(texture != nullptr, "Missing sampled texture '%s' for shader '%s'.",
                    varNames.GetSampledTextureName(stage, shaderBinding.slot), shader->GetResourcePath()))
                {
                    errored = true;
                    continue;
                }

                VkDescriptorImageInfo& imageInfo = m_imageInfoCache[textureCount];
                imageInfo.imageView = texture->GetImageView();
                imageInfo.sampler = texture->GetSampler();
                imageInfo.imageLayout = texture->GetDefaultLayout();
                textureCount++;

                // Creates the descriptor write
                VkWriteDescriptorSet& newWrite = m_writeDescriptorCache[descriptorCount];
                newWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                newWrite.pNext = nullptr;
                newWrite.descriptorCount = 1;
                newWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                newWrite.pImageInfo = &imageInfo;
                newWrite.dstBinding = shaderBinding.binding;
                newWrite.dstSet = descriptorSet;
                descriptorCount++;

                m_sampleTextureDescriptorsAllocated++;
            }

            // Binds storage buffers
            for (const auto& shaderBinding : program->m_storageBufferBindings[stage])
            {
                StorageBufferBinding& bufferBinding = m_storageBufferBindings[stage][shaderBinding.slot];

                if (!AVA_VERIFY(bufferBinding.buffer && bufferBinding.size > 0, "Missing storage buffer '%s' for shader '%s'.",
                    varNames.GetStorageBufferName(stage, shaderBinding.slot), shader->GetResourcePath()))
                {
                    errored = true;
                    continue;
                }

                VkBuffer vkBuffer = VK_NULL_HANDLE;
                switch (bufferBinding.bufferType)
                {
                    case BufferType::Constant:
                        vkBuffer = UpdateAndGetStorageBuffer((VkConstantBuffer*)bufferBinding.buffer, bufferBinding.data, bufferBinding.offset, bufferBinding.size);
                        break;
                    case BufferType::Indirect:
                        vkBuffer = UpdateAndGetStorageBuffer((VkIndirectBuffer*)bufferBinding.buffer, bufferBinding.data, bufferBinding.offset, bufferBinding.size);
                        break;
                    case BufferType::Vertex:
                        vkBuffer = UpdateAndGetStorageBuffer((VkVertexBuffer*)bufferBinding.buffer, bufferBinding.data, bufferBinding.offset, bufferBinding.size);
                        break;
                    case BufferType::Index:
                        vkBuffer = UpdateAndGetStorageBuffer((VkIndexBuffer*)bufferBinding.buffer, bufferBinding.data, bufferBinding.offset, bufferBinding.size);
                        break;
                    default:
                        AVA_ASSERT(false, "[GraphicsContext] storage buffer type is unkown.");
                        break;
                }

                VkDescriptorBufferInfo& bufferInfo = m_bufferInfoCache[bufferCount];
                bufferInfo.buffer = vkBuffer;
                bufferInfo.offset = bufferBinding.offset;
                bufferInfo.range = bufferBinding.size;
                bufferCount++;

                // Creates the descriptor write
                VkWriteDescriptorSet& newWrite = m_writeDescriptorCache[descriptorCount];
                newWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                newWrite.pNext = nullptr;
                newWrite.descriptorCount = 1;
                newWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                newWrite.pBufferInfo = &bufferInfo;
                newWrite.dstBinding = shaderBinding.binding;
                newWrite.dstSet = descriptorSet;
                descriptorCount++;

                // We only need a memory barrier if the buffer is written into.
                if (bufferBinding.access == Access::Write || bufferBinding.access == Access::ReadWrite)
                {
                    VkBufferMemoryBarrier& barrier = m_nextBufferBarriers[m_nextBufferBarrierCount];
                    barrier.buffer = vkBuffer;
                    barrier.offset = bufferBinding.offset;
                    barrier.size = bufferBinding.size;
                    m_nextBufferBarrierCount++;
                }

                m_storageBufferDescriptorsAllocated++;
            }

            // Binds storage textures
            for (const auto& shaderBinding : program->m_storageTextureBindings[stage])
            {
                const StorageTextureBinding& texBinding = m_storageTextureBindings[stage][shaderBinding.slot];
                auto* texture = (VkTexture*)texBinding.texture;

                if (!AVA_VERIFY(texture != nullptr, "Missing storage texture '%s' for shader '%s'.",
                    varNames.GetStorageTextureName(stage, shaderBinding.slot), shader->GetResourcePath()))
                {
                    errored = true;
                    continue;
                }

                VkDescriptorImageInfo& imageInfo = m_imageInfoCache[textureCount];
                imageInfo.imageView = texture->GetMipView(texBinding.mip);
                imageInfo.sampler = texture->GetSampler();
                imageInfo.imageLayout = texture->GetDefaultLayout();
                textureCount++;

                // Creates the descriptor write
                VkWriteDescriptorSet& newWrite = m_writeDescriptorCache[descriptorCount];
                newWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                newWrite.pNext = nullptr;
                newWrite.descriptorCount = 1;
                newWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                newWrite.pImageInfo = &imageInfo;
                newWrite.dstBinding = shaderBinding.binding;
                newWrite.dstSet = descriptorSet;
                descriptorCount++;

                // We only need a memory barrier if the texture is written into.
                if (texBinding.access == Access::Write || texBinding.access == Access::ReadWrite)
                {
                    VkImageMemoryBarrier& barrier = m_nextImageBarriers[m_nextImageBarrierCount];
                    barrier.image = texture->GetImage();
                    barrier.oldLayout = texture->GetLayout();
                    barrier.newLayout = texture->GetDefaultLayout();
                    barrier.subresourceRange.aspectMask = texture->GetAspectFlag();
                    barrier.subresourceRange.baseMipLevel = texBinding.mip;
                    barrier.subresourceRange.levelCount = 1;
                    barrier.subresourceRange.layerCount = texture->GetLayerCount();
                    m_nextImageBarrierCount++;
                }

                m_storageTextureDescriptorsAllocated++;
            }
        }

        if (!errored)
        {
            // Updates descriptors
            vkUpdateDescriptorSets(s_device, descriptorCount, m_writeDescriptorCache.data(), 0, nullptr);

            // Binds descriptors
            vkCmdBindDescriptorSets(
                m_renderCmdBuffer, pipelineBindPoint, pipelineLayout,
                0, (u32)descriptorSets.size(), descriptorSets.data(), 0, nullptr);
        }

        return !errored;
    }

    void VkGraphicsContext::_AddMemoryBarrier()
    {
        if (!m_nextBufferBarrierCount && !m_nextImageBarrierCount)
        {
            return;
        }

        if (m_nextBufferBarrierCount)
        {
            _EndRenderPass();
        }

        vkCmdPipelineBarrier(
            m_renderCmdBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0, nullptr,
            m_nextBufferBarrierCount, m_nextBufferBarriers.data(),
            m_nextImageBarrierCount, m_nextImageBarriers.data());
    }

    void VkGraphicsContext::_EndRenderPass()
    {
        if (m_prevState.framebuffer == nullptr)
        {
            return;
        }
        vkCmdEndRenderPass(m_renderCmdBuffer);

        // Adds a memory barrier for every texture attached to previous framebuffer
        std::vector<VkImageMemoryBarrier> barriers{};
        
        constexpr VkAccessFlags allImageReadAccessFlags =
            VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_TRANSFER_READ_BIT |
            VK_ACCESS_HOST_READ_BIT |
            VK_ACCESS_MEMORY_READ_BIT;
        
        const u32 colorAttachmentCount = m_prevState.framebuffer->GetColorAttachmentCount();
        for (u32 i = 0; i < colorAttachmentCount; i++)
        {
            const FramebufferAttachment* attachment = m_prevState.framebuffer->GetColorAttachment(i);
            const auto* texture = static_cast<VkTexture*>(attachment->texture);

            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.image = texture->GetImage();
            barrier.oldLayout = texture->GetLayout();
            barrier.newLayout = texture->GetDefaultLayout();
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange.baseMipLevel = attachment->mip;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = attachment->layer;
            barrier.subresourceRange.layerCount = 1;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = allImageReadAccessFlags;
            barriers.push_back(barrier);
        }
        
        if (m_prevState.framebuffer->HasDepthAttachment())
        {
            const FramebufferAttachment* attachment = m_prevState.framebuffer->GetDepthAttachment();
            const auto* texture = static_cast<VkTexture*>(attachment->texture);

            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.image = texture->GetImage();
            barrier.oldLayout = texture->GetLayout();
            barrier.newLayout = texture->GetDefaultLayout();
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange.baseMipLevel = attachment->mip;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = attachment->layer;
            barrier.subresourceRange.layerCount = 1;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = allImageReadAccessFlags;
            barriers.push_back(barrier);
        }
        
        vkCmdPipelineBarrier(
            m_renderCmdBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0, nullptr,
            0, nullptr,
            (u32)barriers.size(), barriers.data());

        m_prevState.framebuffer = nullptr;
    }


    // ------- Static initialization helpers -------------------------------------------------------------------------

    void VkGraphicsContext::_CreateInstance()
    {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Ava application";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "Ava Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        const auto extensions = _GetSupportedInstanceExtensions();
        const auto validationLayers = _GetSupportedValidationLayers();

        createInfo.enabledExtensionCount = (u32)extensions.size();
        createInfo.ppEnabledExtensionNames = extensions.data();

        if (s_enableValidationLayers)
        {
            createInfo.enabledLayerCount = (u32)validationLayers.size();
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }

        VkResult result = vkCreateInstance(&createInfo, nullptr, &s_vulkanInstance);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create vulkan instance: %s", GetVkResultStr(result));

        if (_ContainsExtension(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
        {
            // Gets all of the needed extension functions
            vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(s_vulkanInstance, "vkCreateDebugUtilsMessengerEXT");
            vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(s_vulkanInstance, "vkDestroyDebugUtilsMessengerEXT");

            vkCmdInsertDebugUtilsLabelEXT = (PFN_vkCmdInsertDebugUtilsLabelEXT)vkGetInstanceProcAddr(s_vulkanInstance, "vkCmdInsertDebugUtilsLabelEXT");
            vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(s_vulkanInstance, "vkCmdBeginDebugUtilsLabelEXT");
            vkCmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(s_vulkanInstance, "vkCmdEndDebugUtilsLabelEXT");
            vkSetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(s_vulkanInstance, "vkSetDebugUtilsObjectNameEXT");

            if (vkCreateDebugUtilsMessengerEXT)
            {
                // Initializes debug messenger
                VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
                debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
                debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
                debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                debugCreateInfo.pfnUserCallback = _DebugCallback;

                result = vkCreateDebugUtilsMessengerEXT(s_vulkanInstance, &debugCreateInfo, nullptr, &s_debugMessenger);
                AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create debug messenger: %s", GetVkResultStr(result));
            }
        }
    }

    void VkGraphicsContext::_CreateSurface()
    {
        auto* windowHandle = GUIApplication::GetInstance().GetWindow()->GetWindowHandle();
        AVA_ASSERT(windowHandle, "[Vulkan] window handle is invalid.");

        const VkResult result = glfwCreateWindowSurface(s_vulkanInstance, windowHandle, nullptr, &s_surface);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create window surface: %s", GetVkResultStr(result));
    }

    void VkGraphicsContext::_PickPhysicalDevice()
    {
        u32 deviceCount = 0;
        vkEnumeratePhysicalDevices(s_vulkanInstance, &deviceCount, nullptr);
        AVA_ASSERT(deviceCount > 0, "[GraphicsContext] failed to find GPUs with Vulkan support.");

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(s_vulkanInstance, &deviceCount, devices.data());

        // Sorts candidates by increasing score
        std::multimap<int, VkPhysicalDevice> candidates;

        for (const auto& device : devices)
        {
            if (_IsDeviceSuitable(device))
            {
                int score = _RateDeviceSuitability(device);
                candidates.insert(std::make_pair(score, device));
            }
        }
        // Selects best candidate
        AVA_ASSERT(candidates.rbegin()->first > 0, "[GraphicsContext] failed to find a suitable GPU.");
        s_GPU = candidates.rbegin()->second;
        s_queueFamily = _FindQueueFamily(s_GPU);

        // Saves device properties
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(s_GPU, &properties);
        s_deviceLimits = properties.limits;

        // Prints device properties
        const char* deviceName = properties.deviceName;

        const char* deviceType =
            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "Discrete GPU" :
            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "Integrated GPU" :
            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "CPU" : "Other";

        AVA_CORE_INFO("Configuring device '%s' (type : %s)", deviceName, deviceType);
    }

    void VkGraphicsContext::_CreateLogicalDevice()
    {
        constexpr float queuePriority = 1.f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = s_queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        // Vulkan core features
        VkPhysicalDeviceFeatures coreFeatures{};
        coreFeatures.geometryShader       = VK_TRUE; // enable geometry shader
        coreFeatures.fillModeNonSolid     = VK_TRUE; // enable wireframe view
        coreFeatures.wideLines            = VK_TRUE; // enable variable line width
        coreFeatures.shaderClipDistance   = VK_TRUE; // enable clip distances in shaders
        coreFeatures.samplerAnisotropy    = VK_TRUE; // enable anisotropy in texture sampling
        coreFeatures.textureCompressionBC = VK_TRUE; // enable block compression texture formats
        coreFeatures.depthClamp           = VK_TRUE; // enable clamping depth in interval [0,1]
        coreFeatures.depthBounds          = VK_TRUE; // enable culling fragments with depth outside of [0, 1]

        // Vulkan 1.1 features
        VkPhysicalDeviceVulkan11Features extraFeatures11 = {};
        extraFeatures11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        extraFeatures11.shaderDrawParameters = VK_TRUE; // enable vulkan shader built-in parameters

        // Vulkan 1.2 features
        VkPhysicalDeviceVulkan12Features extraFeatures12 = {};
        extraFeatures12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        extraFeatures12.shaderFloat16 = VK_TRUE; // enable 16 bits floats in shaders
        extraFeatures12.pNext = &extraFeatures11;

        // Vulkan 1.3 features
        VkPhysicalDeviceVulkan13Features extraFeatures13 = {};
        extraFeatures13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        extraFeatures13.synchronization2 = VK_TRUE; // enable vulkan synchronization 2
        extraFeatures13.pNext = &extraFeatures12;

        VkPhysicalDeviceFeatures2 allDeviceFeatures = {};
        allDeviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        allDeviceFeatures.features = coreFeatures;
        allDeviceFeatures.pNext = &extraFeatures13;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = &allDeviceFeatures;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.enabledExtensionCount = (u32)std::size(kDeviceExtensionsRequired);
        createInfo.ppEnabledExtensionNames = kDeviceExtensionsRequired;

        auto validationLayers = _GetSupportedValidationLayers();
        if (s_enableValidationLayers)
        {
            createInfo.enabledLayerCount = (u32)validationLayers.size();
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }

        VkResult result = vkCreateDevice(s_GPU, &createInfo, nullptr, &s_device);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create logical device: %s", GetVkResultStr(result));

        // Acquires the device queue where to submit commands
        vkGetDeviceQueue(s_device, s_queueFamily, 0, &s_graphicsQueue);
    }

    void VkGraphicsContext::_CreateAllocator()
    {
        VmaAllocatorCreateInfo createInfo{};
        createInfo.instance = s_vulkanInstance;
        createInfo.physicalDevice = s_GPU;
        createInfo.device = s_device;
        createInfo.frameInUseCount = GraphicsContext::GetContextCount();
        createInfo.pAllocationCallbacks = nullptr;
        createInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        vmaCreateAllocator(&createInfo, &s_allocator);
    }

    void VkGraphicsContext::_InitTransferObjects()
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = s_queueFamily;

        VkResult result = vkCreateCommandPool(s_device, &poolInfo, nullptr, &s_transferCmdPool);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create transfer command pool: %s", GetVkResultStr(result));

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = s_transferCmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        result = vkAllocateCommandBuffers(s_device, &allocInfo, &s_transferCmdBuffer);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to allocate transfer command buffer: %s", GetVkResultStr(result));

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = 0;

        result = vkCreateFence(s_device, &fenceInfo, nullptr, &s_transferFence);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create transfer fence: %s", GetVkResultStr(result));
    }

    void VkGraphicsContext::_InitSwapchain()
    {
        const VkSwapchainSupportDetails swapchainSupport = _QuerySwapchainSupport(s_GPU);
        const VkSurfaceFormatKHR surfaceFormat = _ChooseSwapSurfaceFormat(swapchainSupport.formats);
        const VkPresentModeKHR presentMode = _ChooseSwapPresentMode(swapchainSupport.presentModes);
        const VkExtent2D extent = _ChooseSwapExtent(swapchainSupport.capabilities);

        // Retrieves the minimal number of images the swapchain must hold
        u32 imageCount = swapchainSupport.capabilities.minImageCount + 1;
        if (swapchainSupport.capabilities.maxImageCount > 0 && imageCount > swapchainSupport.capabilities.maxImageCount)
        {
            imageCount = swapchainSupport.capabilities.maxImageCount;
        }

        // Creates the swapchain handle
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = s_surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        const VkResult result = vkCreateSwapchainKHR(s_device, &createInfo, nullptr, &s_swapchain);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create swapchain: %s", GetVkResultStr(result));

        // Creates swapchain images
        vkGetSwapchainImagesKHR(s_device, s_swapchain, &s_swapchainImageCount, nullptr);
        AVA_ASSERT(s_swapchainImageCount > 0, "[GraphicsContext] failed to find any image in the swapchain.");

        std::vector<VkImage> images(s_swapchainImageCount);
        vkGetSwapchainImagesKHR(s_device, s_swapchain, &s_swapchainImageCount, images.data());

        // Creates swapchain image views
        s_swapchainImages.resize(s_swapchainImageCount);
        for (u32 i = 0; i < s_swapchainImageCount; i++)
        {
            s_swapchainImages[i] = new VkTexture(extent, images[i], surfaceFormat.format);

            const std::string debugName = "TX_SWAPCHAIN_" + std::to_string(i);
            s_swapchainImages[i]->SetDebugName(debugName.c_str());
        }

        // Transfers swapchain images to their default layout
        const VkCommandBuffer cmd = BeginTransferCommand();
        for (const auto* image : s_swapchainImages)
        {
            auto* vkImage = (VkTexture*)image;
            vkImage->TransitionToDefaultLayout(cmd);
        }
        EndTransferCommand(cmd);

        s_currentSwapchainImage = 0;
        s_swapchainUpToDate = true;
    }

    void VkGraphicsContext::_CreateFramebuffers()
    {
        s_mainFramebuffers.resize(s_swapchainImageCount);
        for (size_t i = 0; i < s_swapchainImageCount; i++)
        {
            FrameBufferDescription desc{};
            desc.AddAttachment(s_swapchainImages[i]);

            s_mainFramebuffers[i] = CreateFrameBuffer(desc);
        }
    }

    void VkGraphicsContext::_SetGpuDebugInfo()
    {
        SetDebugObjectName(s_GPU, VK_OBJECT_TYPE_PHYSICAL_DEVICE, "GraphicsContext::s_GPU");
        SetDebugObjectName(s_device, VK_OBJECT_TYPE_DEVICE, "GraphicsContext::s_device");
        SetDebugObjectName(s_surface, VK_OBJECT_TYPE_SURFACE_KHR, "GraphicsContext::s_surface");
        SetDebugObjectName(s_graphicsQueue, VK_OBJECT_TYPE_QUEUE, "GraphicsContext::s_graphicQueue");
        SetDebugObjectName(s_swapchain, VK_OBJECT_TYPE_SWAPCHAIN_KHR, "GraphicsContext::s_swapchain");
        SetDebugObjectName(s_transferCmdBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER, "GraphicsContext::s_transferCmdBuffer");
        SetDebugObjectName(s_transferCmdPool, VK_OBJECT_TYPE_COMMAND_POOL, "GraphicsContext::s_transferCmdPool");
        SetDebugObjectName(s_transferFence, VK_OBJECT_TYPE_FENCE, "GraphicsContext::s_transferFence");
    }

    void VkGraphicsContext::_ResizeDeletionQueues()
    {
        // One more than the number of context, to reset the command polls earlier
        const u32 deletionQueueCount = GraphicsContext::GetContextCount() + 1u;
        s_deletionQueues.resize(deletionQueueCount);
    }


    // ------- Instance helpers --------------------------------------------------------------------------------------

    bool VkGraphicsContext::_ContainsExtension(const std::vector<const char*>& _extensions, const char* _extension)
    {
        for (const auto var : _extensions)
        {
            if (strcmp(var, _extension) == 0)
            {
                return true;
            }
        }
        return false;
    }

    std::vector<const char*> VkGraphicsContext::_GetSupportedInstanceExtensions()
    {
        std::vector<const char*> result{};

        u32 extensionCount;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> supportedExtensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, supportedExtensions.data());

        // Check if the required extensions are supported
        for (const char* requiredExtension : kInstanceExtensionsRequired)
        {
            bool extensionFound = false;

            for (const auto& extension : supportedExtensions)
            {
                if (strcmp(requiredExtension, extension.extensionName) == 0)
                {
                    result.push_back(requiredExtension);
                    extensionFound = true;
                    break;
                }
            }
            AVA_ASSERT(extensionFound, "[GraphicsContext] A Vulkan extension is required but not supported.");
        }
        // Then the optional ones
        for (auto optionalExtension : kInstanceExtensionsOptional)
        {
            for (const auto& extension : supportedExtensions)
            {
                if (strcmp(optionalExtension, extension.extensionName) == 0)
                {
                    result.push_back(optionalExtension);
                    break;
                }
            }
        }
        return result;
    }

    std::vector<const char*> VkGraphicsContext::_GetSupportedValidationLayers()
    {
        std::vector<const char*> result{};

        u32 layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layerProperties(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());

        // For each required validation layer
        for (auto validationLayer : kValidationLayers)
        {
            bool layerFound = false;

            // Search all supported layers for a corresponding one.
            for (const auto& layerProperty : layerProperties)
            {
                if (strcmp(layerProperty.layerName, validationLayer) == 0)
                {
                    result.push_back(validationLayer);
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound)
            {
                AVA_CORE_ERROR("[GraphicsContext] missing Vulkan validation layer '%s'.", validationLayer);
            }
        }
        return result;
    }
    
    VKAPI_ATTR VkBool32 VKAPI_CALL VkGraphicsContext::_DebugCallback(
        const VkDebugUtilsMessageSeverityFlagBitsEXT _messageSeverity,
        const VkDebugUtilsMessageTypeFlagsEXT _messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* _pCallbackData,
        void* _pUserData)
    {
        std::string typeStr;
        switch(_messageType)
        {
            case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
                typeStr = "General";
                break;
            case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
                typeStr = "Validation";
                break;
            case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
                typeStr  ="Performance";
                break;
            case VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT:
                typeStr = "Bindings";
                break;
            default:
                typeStr = "Unknown";
        }

        switch(_messageSeverity)
        {
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
                AVA_CORE_INFO("[Vk Info][%s] %s\n", typeStr.c_str(), _pCallbackData->pMessage);
                break;

            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
                AVA_CORE_WARN("[Vk Warning][%s] %s\n", typeStr.c_str(), _pCallbackData->pMessage);
                break;

            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
                AVA_CORE_ERROR("[Vk Error][%s] %s\n", typeStr.c_str(), _pCallbackData->pMessage);
                AVA_ASSERT(false, "[Vk Error][%s] %s\n", typeStr.c_str(), _pCallbackData->pMessage);
                break;
            default:
                AVA_CORE_TRACE("[Vk Unkown][%s] %s\n", typeStr.c_str(), _pCallbackData->pMessage);
        }

        return VK_FALSE;
    }


    // ------- Device helpers ----------------------------------------------------------------------------------------

    bool VkGraphicsContext::_IsDeviceSuitable(const VkPhysicalDevice _device)
    {
        // Device must contain a valid queue
        if (_FindQueueFamily(_device) == kVulkanIdxInvalid)
        {
            return false;
        }
        // Device must support required extensions
        if (!_CheckDeviceExtensionSupport(_device))
        {
            return false;
        }
        // Device must support required features
        if (!_CheckDeviceFeatureSupport(_device))
        {
            return false;
        }
        // Device must match the swapchain requirements
        const VkSwapchainSupportDetails swapChainSupport = _QuerySwapchainSupport(_device);
        if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty())
        {
            return false;
        }
        return true;
    }

    u32 VkGraphicsContext::_FindQueueFamily(const VkPhysicalDevice _device)
    {
        u32 queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(_device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(_device, &queueFamilyCount, queueFamilies.data());

        for (u32 queueFamilyIndex = 0; queueFamilyIndex < queueFamilyCount; queueFamilyIndex++)
        {
            const VkQueueFamilyProperties& queueProperties = queueFamilies[queueFamilyIndex];
            const VkBool32 graphicsSupported = queueProperties.queueFlags & VK_QUEUE_GRAPHICS_BIT;
            const VkBool32 computeSupported = queueProperties.queueFlags & VK_QUEUE_COMPUTE_BIT;
            const VkBool32 transferSupported = queueProperties.queueFlags & VK_QUEUE_TRANSFER_BIT;

            VkBool32 presentSupported = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(_device, queueFamilyIndex, s_surface, &presentSupported);

            if (graphicsSupported && computeSupported && transferSupported && presentSupported)
            {
                return queueFamilyIndex;
            }
        }
        return kVulkanIdxInvalid;
    }

    bool VkGraphicsContext::_CheckDeviceExtensionSupport(const VkPhysicalDevice _device)
    {
        u32 extensionCount;
        vkEnumerateDeviceExtensionProperties(_device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> supportedExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(_device, nullptr, &extensionCount, supportedExtensions.data());

        for (const char* requiredExtension : kDeviceExtensionsRequired)
        {
            bool extensionFound = false;
            for (const auto& extension : supportedExtensions)
            {
                if (strcmp(requiredExtension, extension.extensionName) == 0)
                {
                    extensionFound = true;
                    break;
                }
            }
            if (!extensionFound)
            {
                return false;
            }
        }
        return true;
    }

    bool VkGraphicsContext::_CheckDeviceFeatureSupport(const VkPhysicalDevice _device)
    {
        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(_device, &supportedFeatures);

        return
            supportedFeatures.samplerAnisotropy &&  // anisotropy 
            supportedFeatures.sampleRateShading &&  // sample shading  
            supportedFeatures.geometryShader &&     // geometry shader
            supportedFeatures.fillModeNonSolid &&   // wireframe view
            supportedFeatures.shaderClipDistance && // clip distances
            supportedFeatures.depthClamp;           // depth clamp
    }

    VkSwapchainSupportDetails VkGraphicsContext::_QuerySwapchainSupport(const VkPhysicalDevice _device)
    {
        VkSwapchainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_device, s_surface, &details.capabilities);

        u32 formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(_device, s_surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(_device, s_surface, &formatCount, details.formats.data());
        }

        u32 presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(_device, s_surface, &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(_device, s_surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    int VkGraphicsContext::_RateDeviceSuitability(const VkPhysicalDevice _device)
    {
        int score = 0;
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(_device, &deviceProperties);

        // Discrete GPUs have a significant performance advantage
        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            score += 1000;
        }

        // Maximum possible size of textures affects graphics quality
        score += deviceProperties.limits.maxImageDimension2D;

        // Maximum uniform buffer affects workflow
        score += deviceProperties.limits.maxPerStageDescriptorUniformBuffers;

        return score;
    }


    // ------- Swapchain helpers -------------------------------------------------------------------------------------

    VkSurfaceFormatKHR VkGraphicsContext::_ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& _availableFormats)
    {
        // If any surface surface format is supported, use VK_FORMAT_B8G8R8A8_UNORM.
        if (_availableFormats.size() == 1 && _availableFormats[0].format == VK_FORMAT_UNDEFINED)
        {
            return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        }
        // Otherwise if VK_FORMAT_B8G8R8A8_UNORM is supported, use it.
        for (const auto& availableFormat : _availableFormats)
        {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        // Otherwise use the first available surface format.
        return _availableFormats[0];
    }

    VkPresentModeKHR VkGraphicsContext::_ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& _availablePresentModes)
    {
        auto contains = [](const std::vector<VkPresentModeKHR>& _vec, const VkPresentModeKHR& _item)
        {
            return std::find(_vec.begin(), _vec.end(), _item) != _vec.end();
        };

        // FIFO relaxed = traditional v-sync
        if (GraphicsContext::GetVSyncMode() && contains(_availablePresentModes, VK_PRESENT_MODE_FIFO_RELAXED_KHR))
        {
            return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        }
        // Mailbox = run as fast as you can but only show the last generated image at each v-sync
        if (contains(_availablePresentModes, VK_PRESENT_MODE_MAILBOX_KHR))
        {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
        // Immediate = run as fast as you can, may result in screen tearing
        if (contains(_availablePresentModes, VK_PRESENT_MODE_IMMEDIATE_KHR))
        {
            return VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D VkGraphicsContext::_ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& _capabilities)
    {
        // If there is already an extent defined, use it.
        if (_capabilities.currentExtent.width != std::numeric_limits<u32>::max())
        {
            return _capabilities.currentExtent;
        }

        const Window* window = GUIApplication::GetInstance().GetWindow();

        VkExtent2D extent{};
        extent.width = (u32)Math::clamp(window->GetWidth(), (float)_capabilities.minImageExtent.width, (float)_capabilities.maxImageExtent.width);
        extent.height = (u32)Math::clamp(window->GetHeight(), (float)_capabilities.minImageExtent.height, (float)_capabilities.maxImageExtent.height);

        return extent;
    }

    void VkGraphicsContext::_RecreateSwapchain()
    {
        WaitIdle();

        const VkSwapchainSupportDetails support = _QuerySwapchainSupport(s_GPU);
        const VkExtent2D viewportExtent = support.capabilities.maxImageExtent;

        // Early returns if the window was minimized
        if (viewportExtent.width == 0 || viewportExtent.height == 0)
        {
            return;
        }

        // Cleanups main framebuffers
        for (size_t i = 0; i < s_swapchainImageCount; i++)
        {
            DestroyFrameBuffer(s_mainFramebuffers[i]);
            DestroyTexture(s_swapchainImages[i]);
        }

        // Destroys swapchain
        vkDestroySwapchainKHR(s_device, s_swapchain, nullptr);

        // Recreates swapchain
        _InitSwapchain();

        // Reallocates main framebuffers
        _CreateFramebuffers();
    }

}
