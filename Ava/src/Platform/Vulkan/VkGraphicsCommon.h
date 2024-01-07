#pragma once
/// @file VkGraphicsCommon.h
/// @brief file extending GraphicCommon.h to Vulkan.

#include <Graphics/GraphicsCommon.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <vulkan/vulkan.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    class VkShader;
    class VkTexture;
    class VkConstantBuffer;
    class VkIndirectBuffer;
    class VkVertexBuffer;
    class VkIndexBuffer;
    class VkFrameBuffer;
    class VkShaderProgram;

    /// @brief Contains everything needed to evaluate the swapchain of a specific GPU.
    struct VkSwapchainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    /// @brief Contains everything needed to create a VkPipeline object.
    struct VkPipelineState
    {
        // user-defined
        PrimitiveType::Enum primitiveType = PrimitiveType::Triangle;
        RasterStateID rasterState = AVA_DISABLE_STATE;
        BlendStateID blendState = AVA_DISABLE_STATE;
        DepthStateID depthState = AVA_DISABLE_STATE;
        bool wireframeView = false;

        // vertex buffer-defined
        VertexLayoutID vertexLayout = AVA_DISABLE_STATE;

        // framebuffer-defined
        VkRenderPass renderPass = VK_NULL_HANDLE;
        u32 colorAttachments = 0;

        // pipeline identification
        u32 pipelineID = 0;
        bool pipelineStateDirty = false;

        u32 GetPipelineID()
        {
            if (pipelineStateDirty && renderPass)
            {
                pipelineID = HashU32(primitiveType);
                pipelineID = HashU32Combine(vertexLayout, pipelineID);
                pipelineID = HashU32Combine(rasterState, pipelineID);
                pipelineID = HashU32Combine(depthState, pipelineID);
                pipelineID = HashU32Combine(blendState, pipelineID);
                pipelineID = HashU32Combine(wireframeView, pipelineID);
                pipelineID = HashU32Combine(renderPass, pipelineID);
                pipelineID = HashU32Combine(colorAttachments, pipelineID);
                pipelineStateDirty = false;
            }
            return pipelineID;
        }
    };

    /// @brief List of vulkan objects waiting for deletion.
    template <typename T>
    class VkDeletionList
    {
    public:
        void Add(T* _vkObject)
        {
            m_list.push_back(_vkObject);
        }

        void Flush()
        {
            while (!m_list.empty())
            {
                if (m_list.back() != nullptr)
                {
                    delete m_list.back();
                    m_list.back() = nullptr;
                }
                m_list.pop_back();
            }
        }

    private:
        std::vector<T*> m_list;
    };

    /// @brief Contains every vulkan objects registered for deletion.
    struct VkDeletionQueue
    {
        VkDeletionList<VkFrameBuffer> framebuffersToDelete;
        VkDeletionList<VkShaderProgram> programsToDelete;
        VkDeletionList<VkTexture> texturesToDelete;
        VkDeletionList<VkShader> shadersToDelete;
        VkDeletionList<VkVertexBuffer> vertexBuffersToDelete;
        VkDeletionList<VkIndexBuffer> indexBuffersToDelete;
        VkDeletionList<VkConstantBuffer> constantBuffersToDelete;
        VkDeletionList<VkIndirectBuffer> indirectBuffersToDelete;

        void FlushAll()
        {
            framebuffersToDelete.Flush();
            programsToDelete.Flush();
            texturesToDelete.Flush();
            shadersToDelete.Flush();
            vertexBuffersToDelete.Flush();
            indexBuffersToDelete.Flush();
            constantBuffersToDelete.Flush();
            indirectBuffersToDelete.Flush();
        }
    };
}

