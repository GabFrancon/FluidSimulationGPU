#pragma once
/// @file VkShader.h
/// @brief file implementing Shader.h for Vulkan.

#include <Graphics/Shader.h>
#include <Strings/StringHash.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <vulkan/vulkan.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    class VkShader final : public Shader
    {
    public:
        VkShader() = default;
        ~VkShader() override;

        /// @brief for path-based shader resources.
        explicit VkShader(ShaderStage::Enum _stage, const ShaderDescription& _desc);
        /// @brief for built-in shader data.
        explicit VkShader(const ShaderData& _data);

        void SetDebugName(const char* _name) override;
        void Load(const ShaderData& _data);

        // Reflected shader binding
        struct Binding
        {
            StringHash name;
            VkDescriptorType type;
            u32 bindingSlotID;
            u32 descriptorSetID;
        };

        // Reflected vertex attribute
        struct Attribute {
            StringHash name;
            VkFormat format;
            u32 locationID;
        };

        // Reflected fragment color output
        struct ColorOutput {
            StringHash name;
            VkFormat format;
            u32 locationID;
        };

        // Reflection data
        struct ReflectionData
        {
            ShaderStage::Enum stage;
            std::vector<Binding> bindings;
            std::vector<Attribute> attributes;
            std::vector<ColorOutput> colorOutputs;

            void Serialize(std::vector<char>& _bytes) const;
            void Deserialize(const ShaderData* _data);
        };

        const ReflectionData& GetReflectionData() const { return m_reflectionData; }
        VkDescriptorSetLayout GetDescriptorLayout() const { return m_shaderLayout; }
        VkShaderModule GetModule() const { return m_module; }

    private:
        void _LoadModule(const u32* _code, size_t _size);
        void _BuildShaderLayout();

        ReflectionData m_reflectionData;
        VkShaderModule m_module = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_shaderLayout = VK_NULL_HANDLE;
    };

}