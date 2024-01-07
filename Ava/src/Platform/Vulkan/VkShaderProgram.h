#pragma once
/// @file VkShaderProgram.h
/// @brief file implementing ShaderProgram.h for Vulkan.

#include <Graphics/ShaderProgram.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <vulkan/vulkan.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    struct VkPipelineState;

    class VkShaderProgram final : public ShaderProgram
    {
    public:
        VkShaderProgram() = default;
        ~VkShaderProgram() override;

        /// @brief for path-based shader resources.
        explicit VkShaderProgram(const ShaderProgramDescription& _desc);
        /// @brief for built-in shader objects.
        explicit VkShaderProgram(Shader* _stages[ShaderStage::Count], const ShaderResourceNames& _varNames);

        void SetDebugName(const char* _name) override;

        // For graphics shader programs
        VkPipelineLayout GetGraphicsLayout();
        VkPipeline GetGraphicsPipeline(VkPipelineState& _state);

        // For compute shader programs
        VkPipelineLayout GetComputeLayout();
        VkPipeline GetComputePipeline();

        u32 GetBufferBindingCount() const { return m_bufferBindingCount; }
        u32 GetTextureBindingCount() const { return m_textureBindingCount; }
        u32 GetBindingCount() const { return m_bufferBindingCount + m_textureBindingCount; }

        struct VarSlotBinding
        {
            u32 slot;    // CPU
            u32 binding; // GPU
        };

        std::vector<VarSlotBinding> m_bufferBindings[ShaderStage::Count];         // constant buffers
        std::vector<VarSlotBinding> m_textureBindings[ShaderStage::Count];        // sampled textures
        std::vector<VarSlotBinding> m_storageBufferBindings[ShaderStage::Count];  // storage buffers
        std::vector<VarSlotBinding> m_storageTextureBindings[ShaderStage::Count]; // storage textures

        /// @warning Must match the descriptor set indices declared in "shaders/common/Platform.glsl".
        static constexpr int kVertexSetIndex = 0;
        static constexpr int kFragmentSetIndex = 1;
        static constexpr int kGeometricSetIndex = 2;
        static constexpr int kComputeSetIndex = 0;

        void GetDescriptorSetLayouts(std::vector<VkDescriptorSetLayout>& _descriptorSetLayouts) const;
        static int GetDescriptorSetIndex(ShaderStage::Enum _stage);

    private:
        void _BuildSlotMap();

        // For graphics programs
        void _BuildGraphicsLayout();
        VkPipeline _BuildGraphicsPipeline(const VkPipelineState& _pipelineState);
        VkPipelineLayout m_graphicsLayout = VK_NULL_HANDLE;

        // For compute programs
        void _BuildComputeLayout();
        VkPipeline _BuildComputePipeline();
        VkPipelineLayout m_computeLayout = VK_NULL_HANDLE;

        std::unordered_map<u32, VkPipeline> m_pipelines{};
        VkPipeline m_lastPipeline = VK_NULL_HANDLE;
        u32 m_lastPipelineId = 0;

        u32 m_bufferBindingCount = 0;
        u32 m_textureBindingCount = 0;
    };

}
