#include <avapch.h>
#include "VkShader.h"

#include <Debug/Log.h>
#include <Debug/Assert.h>
#include <Resources/ShaderData.h>
#include <Platform/Vulkan/VkGraphicsContext.h>

namespace Ava {

    //-------------- Reflection data implementation -----------------------------------------

    void VkShader::ReflectionData::Serialize(std::vector<char>& _bytes) const
    {
        const size_t bufferSize = bindings.size() * sizeof(Binding)
                                + attributes.size() * sizeof(Attribute)
                                + colorOutputs.size() * sizeof(ColorOutput);

        _bytes.resize(bufferSize);
        char* cursor = _bytes.data();

        // copy shader bindings
        {
            const size_t count = bindings.size();
            const size_t rawSize = count * sizeof(Binding);

            memcpy(cursor, bindings.data(), rawSize);
            cursor += rawSize;
        }

        // copy vertex attributes
        {
            const size_t count = attributes.size();
            const size_t rawSize = count * sizeof(Attribute);

            memcpy(cursor, attributes.data(), rawSize);
            cursor += rawSize;
        }

        // copy fragment color outputs
        {
            const size_t count = colorOutputs.size();
            const size_t rawSize = count * sizeof(ColorOutput);

            memcpy(cursor, colorOutputs.data(), rawSize);
            // cursor += rawSize;
        }
    }

    void VkShader::ReflectionData::Deserialize(const ShaderData* _data)
    {
        stage = _data->stage;

        char* cursor = _data->reflectionData;
        AVA_ASSERT(cursor, "[Shader] invalid reflection data.");

        // copy shader bindings
        {
            const size_t count = _data->bindingsCount;
            const size_t rawSize = count * sizeof(Binding);

            bindings.resize(count);
            memcpy(bindings.data(), cursor, rawSize);
            cursor += rawSize;
        }

        // copy vertex attributes
        {
            const size_t count = _data->attributesCount;
            const size_t rawSize = count * sizeof(Attribute);

            attributes.resize(count);
            memcpy(attributes.data(), cursor, rawSize);
            cursor += rawSize;
        }

        // copy fragment color outputs
        {
            const size_t count = _data->colorOutputsCount;
            const size_t rawSize = count * sizeof(ColorOutput);

            colorOutputs.resize(count);
            memcpy(colorOutputs.data(), cursor, rawSize);
            // cursor += rawSize;
        }
    }


    //-------------- Vk shader implementation -----------------------------------------------

    VkShader::VkShader(const ShaderStage::Enum _stage, const ShaderDescription& _desc)
        : Shader(_stage, _desc)
    {
        const auto* path = GetResourcePath();
        ShaderData data;

        if (!ShaderLoader::Load(path, data))
        {
            AVA_CORE_ERROR("[Shader] failed to load '%s'.", path);
            SetResourceState(ResourceState::LoadFailed);
            return;
        }

        Load(data);
        ShaderLoader::Release(data);
        SetResourceState(ResourceState::Loaded);
    }

    VkShader::VkShader(const ShaderData& _data)
        : Shader(_data)
    {
        Load(_data);
        SetResourceState(ResourceState::Loaded);
    }

    VkShader::~VkShader()
    {
        vkDestroyShaderModule(VkGraphicsContext::GetDevice(), m_module, nullptr);
    }

    void VkShader::SetDebugName(const char* _name)
    {
        VkGraphicsContext::SetDebugObjectName(m_module, VK_OBJECT_TYPE_SHADER_MODULE, _name);
    }

    void VkShader::Load(const ShaderData& _data)
    {
        _LoadModule(_data.code, _data.codeSize);
        m_reflectionData.Deserialize(&_data);
        _BuildShaderLayout();
    }

    void VkShader::_LoadModule(const u32* _code, const size_t _size)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = _size;
        createInfo.pCode = _code;

        VkShaderModule shaderModule;
        const VkResult result = vkCreateShaderModule(VkGraphicsContext::GetDevice(), &createInfo, nullptr, &shaderModule);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create shader module: %s", VkGraphicsContext::GetVkResultStr(result));
        m_module = shaderModule;
    }

    void VkShader::_BuildShaderLayout()
    {
        const u32 bindingCount = (u32)m_reflectionData.bindings.size();
        if (bindingCount > 0)
        {
            std::vector<VkDescriptorSetLayoutBinding> vkBindings;
            vkBindings.resize(bindingCount);

            for (size_t i = 0; i < bindingCount; i++)
            {
                VkDescriptorSetLayoutBinding& layoutBinding = vkBindings[i];
                layoutBinding.binding = m_reflectionData.bindings[i].bindingSlotID;
                layoutBinding.descriptorCount = 1;
                layoutBinding.descriptorType = m_reflectionData.bindings[i].type;
                layoutBinding.stageFlags = ToVk(m_stage);
            }
            // Builds descriptor set layout
            VkDescriptorSetLayoutCreateInfo vkCreateInfo{};
            vkCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            vkCreateInfo.bindingCount = bindingCount;
            vkCreateInfo.pBindings = vkBindings.data();

            m_shaderLayout = VkGraphicsContext::CreateShaderLayout(&vkCreateInfo);
        }
        else
        {
            // Empty layout
            VkDescriptorSetLayoutCreateInfo emptySetLayoutInfo{};
            emptySetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            emptySetLayoutInfo.bindingCount = 0;
            emptySetLayoutInfo.flags = 0;
            emptySetLayoutInfo.pNext = nullptr;
            emptySetLayoutInfo.pBindings = nullptr;

            m_shaderLayout = VkGraphicsContext::CreateShaderLayout(&emptySetLayoutInfo);
        }
    }

}