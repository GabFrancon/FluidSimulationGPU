#include <avapch.h>
#include "VkShaderProgram.h"

#include <Debug/Assert.h>
#include <Platform/Vulkan/VkShader.h>
#include <Platform/Vulkan/VkGraphicsContext.h>

namespace Ava {

    VkShaderProgram::VkShaderProgram(const ShaderProgramDescription& _desc)
        : ShaderProgram(_desc)
    {
        _BuildSlotMap();
    }

    VkShaderProgram::VkShaderProgram(Shader* _stages[ShaderStage::Count], const ShaderResourceNames& _varNames)
        : ShaderProgram(_stages, _varNames)
    {
        _BuildSlotMap();
    }

    VkShaderProgram::~VkShaderProgram()
    {
        for (const auto& pair : m_pipelines)
        {
            vkDestroyPipeline(VkGraphicsContext::GetDevice(), pair.second, nullptr);
        }
        if (IsCompute())
        {
            vkDestroyPipelineLayout(VkGraphicsContext::GetDevice(), m_computeLayout, nullptr);
        }
        else
        {
            vkDestroyPipelineLayout(VkGraphicsContext::GetDevice(), m_graphicsLayout, nullptr);
        }
    }

    void VkShaderProgram::SetDebugName(const char* _name)
    {
        VkGraphicsContext::SetDebugObjectName(IsCompute() ? m_computeLayout : m_graphicsLayout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, _name);
    }

    VkPipelineLayout VkShaderProgram::GetGraphicsLayout()
    {
        AVA_ASSERT(!IsCompute(), "[ShaderProg] the program %s is not graphics compatible.", GetResourceName());

        if (!m_graphicsLayout)
        {
            _BuildGraphicsLayout();
        }
        return m_graphicsLayout;
    }

    VkPipeline VkShaderProgram::GetGraphicsPipeline(VkPipelineState& _state)
    {
        AVA_ASSERT(!IsCompute(), "[ShaderProg] the program %s is not graphics compatible.", GetResourceName());

        const u32 id = _state.GetPipelineID();
        if (id != m_lastPipelineId)
        {
            m_lastPipelineId = id;
            const auto findIterator = m_pipelines.find(id);
            if (findIterator != m_pipelines.end())
            {
                m_lastPipeline = findIterator->second;
            }
            else
            {
                m_lastPipeline = _BuildGraphicsPipeline(_state);
                m_pipelines[m_lastPipelineId] = m_lastPipeline;
            }
        }
        return m_lastPipeline;
    }

    VkPipelineLayout VkShaderProgram::GetComputeLayout()
    {
        AVA_ASSERT(IsCompute(), "[ShaderProg] the program %s is not compute compatible.", GetResourceName());

        if (!m_computeLayout)
        {
            _BuildComputeLayout();
        }
        return m_computeLayout;
    }

    VkPipeline VkShaderProgram::GetComputePipeline()
    {
        AVA_ASSERT(IsCompute(), "[ShaderProg] the %s program is not compute compatible.", GetResourceName());

        if (m_pipelines.empty())
        {
            // Compute programs only have one pipeline.
            m_lastPipelineId = 0;

            m_lastPipeline = _BuildComputePipeline();
            m_pipelines[m_lastPipelineId] = m_lastPipeline;
        }
        return m_pipelines[m_lastPipelineId];
    }

    void VkShaderProgram::GetDescriptorSetLayouts(std::vector<VkDescriptorSetLayout>& _descriptorSetLayouts) const
    {
        _descriptorSetLayouts.resize(GetShaderCount());

        if (IsCompute())
        {
            // only compute shader
            const auto* computeShader = (VkShader*)GetShader(ShaderStage::Compute);
            AVA_ASSERT(computeShader, "[ShaderProg] a compute program requires a valid compute shader.");
            _descriptorSetLayouts[kComputeSetIndex] = computeShader->GetDescriptorLayout();
        }
        else
        {
            // vertex shader is mandatory
            const auto* vertexShader = (VkShader*)GetShader(ShaderStage::Vertex);
            AVA_ASSERT(vertexShader, "[ShaderProg] a graphics program requires a valid vertex shader.");
            _descriptorSetLayouts[kVertexSetIndex] = vertexShader->GetDescriptorLayout();

            // fragment shader is mandatory
            const auto* fragmentShader = (VkShader*)GetShader(ShaderStage::Fragment);
            AVA_ASSERT(fragmentShader, "[ShaderProg] a graphics program requires a valid fragment shader.");
            _descriptorSetLayouts[kFragmentSetIndex] = fragmentShader->GetDescriptorLayout();

            // geometric shader is optional
            if (const auto* geometricShader = (VkShader*)GetShader(ShaderStage::Geometric))
            {
                _descriptorSetLayouts[kGeometricSetIndex] = geometricShader->GetDescriptorLayout();
            }
        }
    }

    int VkShaderProgram::GetDescriptorSetIndex(const ShaderStage::Enum _stage)
    {
        return
                _stage == ShaderStage::Vertex ? kVertexSetIndex :
                _stage == ShaderStage::Geometric ? kGeometricSetIndex :
                _stage == ShaderStage::Fragment ? kFragmentSetIndex :
                _stage == ShaderStage::Compute ? kComputeSetIndex :
                0;
    }

    void VkShaderProgram::_BuildSlotMap()
    {
        const u32 slotCountPerStage = GraphicsContext::GetSettings().nbBindingSlotPerStage;

        for (u8 i = 0; i < ShaderStage::Count; i++)
        {
            const auto stage = ShaderStage::Enum(i);

            if (const auto* shader = (VkShader*)GetShader(stage))
            {
                for (auto& reflectedBinding : shader->GetReflectionData().bindings)
                {
                    switch (reflectedBinding.type)
                    {
                        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                        {
                            for (u32 slot = 0; slot < slotCountPerStage; slot++)
                            {
                                if (m_varNames.GetConstantBufferName(stage, slot) == reflectedBinding.name)
                                {
                                    VarSlotBinding shaderBinding{};
                                    shaderBinding.slot = slot; // CPU
                                    shaderBinding.binding = reflectedBinding.bindingSlotID; // GPU

                                    m_bufferBindings[stage].push_back(shaderBinding);
                                    m_bufferBindingCount++;
                                    goto BindingFound;
                                }
                            }
                            AVA_ASSERT(false, "[ShaderProg] %s: the constant buffer %s is not linked to a slot!",
                                shader->GetResourcePath(), reflectedBinding.name.GetString());

                            break;
                        }

                        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                        {
                            for (u32 slot = 0; slot < slotCountPerStage; slot++)
                            {
                                const auto fromCpu = m_varNames.GetStorageBufferName(stage, slot);
                                const auto fromGpu = reflectedBinding.name;

                                if (fromCpu == fromGpu)
                                {
                                    VarSlotBinding shaderBinding{};
                                    shaderBinding.slot = slot; // CPU
                                    shaderBinding.binding = reflectedBinding.bindingSlotID; // GPU

                                    m_storageBufferBindings[stage].push_back(shaderBinding);
                                    m_bufferBindingCount++;
                                    goto BindingFound;
                                }
                            }
                            AVA_ASSERT(false, "[ShaderProg] %s: the storage buffer %s is not linked to a slot!",
                                shader->GetResourcePath(), reflectedBinding.name.GetString());

                            break;
                        }

                        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                        {
                            for (u32 slot = 0; slot < slotCountPerStage; slot++)
                            {
                                if (m_varNames.GetSampledTextureName(stage, slot) == reflectedBinding.name)
                                {
                                    VarSlotBinding shaderBinding{};
                                    shaderBinding.slot = slot; // CPU
                                    shaderBinding.binding = reflectedBinding.bindingSlotID; // GPU

                                    m_textureBindings[stage].push_back(shaderBinding);
                                    m_textureBindingCount++;
                                    goto BindingFound;
                                }
                            }
                            AVA_ASSERT(false, "[ShaderProg] %s: the sampled texture %s is not linked to a slot!",
                                shader->GetResourcePath(), reflectedBinding.name.GetString());

                            break;
                        }

                        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                        {
                            for (u32 slot = 0; slot < slotCountPerStage; slot++)
                            {
                                if (m_varNames.GetStorageTextureName(stage, slot) == reflectedBinding.name)
                                {
                                    VarSlotBinding shaderBinding{};
                                    shaderBinding.slot = slot; // CPU
                                    shaderBinding.binding = reflectedBinding.bindingSlotID; // GPU

                                    m_storageTextureBindings[stage].push_back(shaderBinding);
                                    m_textureBindingCount++;
                                    goto BindingFound;
                                }
                            }
                            AVA_ASSERT(false, "[ShaderProg] %s: the storage texture %s is not linked to a slot!",
                                shader->GetResourcePath(), reflectedBinding.name.GetString());

                            break;
                        }

                        default:
                            AVA_ASSERT(false, "[ShaderProg] %s: variable '%s' is not recognized.",
                                m_shaders[stage]->GetResourcePath(), reflectedBinding.name.GetString());
                    }

                    BindingFound:
                        continue;
                }
            }
        }
    }

    void VkShaderProgram::_BuildGraphicsLayout()
    {
        AVA_ASSERT(!IsCompute(), "[ShaderProg] the program %s is not graphics compatible.", GetResourceName());

        std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
        GetDescriptorSetLayouts(descriptorSetLayouts);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = (u32)descriptorSetLayouts.size();
        pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

        const VkResult result = vkCreatePipelineLayout(VkGraphicsContext::GetDevice(), &pipelineLayoutInfo, nullptr, &m_graphicsLayout);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create graphics pipeline layout: %s", VkGraphicsContext::GetVkResultStr(result));
    }

    VkPipeline VkShaderProgram::_BuildGraphicsPipeline(const VkPipelineState& _pipelineState)
    {
        AVA_ASSERT(!IsCompute(), "[ShaderProg] The program %s is not graphics compatible.", GetResourceName());

        // Makes sure the graphics layout was created
        if (!m_graphicsLayout)
        {
            _BuildGraphicsLayout();
        }

        // Settles shader stages
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages{};
        for (u8 i = 0; i < ShaderStage::Count; i++)
        {
            const auto stage = ShaderStage::Enum(i);

            if (const auto* shader = (VkShader*)GetShader(stage))
            {
                VkPipelineShaderStageCreateInfo shaderInfo{};
                shaderInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderInfo.stage = ToVk(stage);
                shaderInfo.module = shader->GetModule();
                shaderInfo.pName = "main";

                shaderStages.push_back(shaderInfo);
            }
        }

        // Settles input assembly state
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = ToVk(_pipelineState.primitiveType);
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // Settles vertex input state
        const VertexLayout& vertexLayout = GraphicsContext::GetVertexLayout(_pipelineState.vertexLayout);
        auto vertexShader = (VkShader*)GetShader(ShaderStage::Vertex);
        AVA_ASSERT(vertexShader, "[ShaderProg] a graphics program requires a valid vertex shader.");

        std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};

        // for each attribute required by the program
        for (auto& attribute : vertexShader->GetReflectionData().attributes)
        {
            bool attributeSlotFound = false;
            auto findAttribute = [&attribute, &vertexLayout, &attributeDescriptions]()
            {
                // identifies the semantic of the attribute via its name
                for (u8 i = 0; i < VertexSemantic::Count; i++)
                {
                    const auto semantic = VertexSemantic::Enum(i);
                    const StringHash semanticName = Stringify(semantic);
                    if (semanticName == attribute.name)
                    {
                        // searches for the corresponding element in the pipeline state
                        for (u8 j = 0; j < vertexLayout.attributesCount; j++)
                        {
                            const VertexAttribute& elem = vertexLayout.attributes[j];
                            if (elem.semantic == semantic)
                            {
                                VkVertexInputAttributeDescription attributeDesc{};
                                attributeDesc.binding = 0;
                                attributeDesc.location = attribute.locationID;
                                attributeDesc.offset = elem.offset;
                                attributeDesc.format = ToVk(elem.dataType, elem.dataCount);
                                attributeDescriptions.push_back(attributeDesc);
                                return true;
                            }
                        }
                    }
                }
                return false;
            };

            attributeSlotFound = findAttribute();

            AVA_ASSERT(attributeSlotFound,
                "[ShaderProg] %s is expecting data for attribute '%s' but current vertex buffer doesn't have it.",
                vertexShader->GetResourcePath(), attribute.name.GetString());
        }

        VkVertexInputBindingDescription bindingDesc{};
        VkPipelineVertexInputStateCreateInfo vertexInput{};

        if (attributeDescriptions.empty())
        {
            vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInput.vertexBindingDescriptionCount = 0;
            vertexInput.pVertexBindingDescriptions = nullptr;
            vertexInput.vertexAttributeDescriptionCount = 0;
            vertexInput.pVertexAttributeDescriptions = nullptr;
        }
        else
        {
            // We only have one vertex binding per program
            bindingDesc.binding = 0;
            bindingDesc.stride = vertexLayout.stride;
            bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInput.vertexBindingDescriptionCount = 1;
            vertexInput.pVertexBindingDescriptions = &bindingDesc;
            vertexInput.vertexAttributeDescriptionCount = (u32)attributeDescriptions.size();
            vertexInput.pVertexAttributeDescriptions = attributeDescriptions.data();
        }

        // Settles viewport state
        VkViewport viewport{};
        VkRect2D scissor{};

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        // viewport, scissor, and line width are made dynamic states
        std::vector dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.pDynamicStates = dynamicStates.data();
        dynamicState.dynamicStateCount = static_cast<u32>(dynamicStates.size());

        // Settles rasterization state
        const RasterState& rasterState = GraphicsContext::GetRasterState(_pipelineState.rasterState);
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = _pipelineState.wireframeView ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
        rasterizer.cullMode = rasterState.cullMode;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = rasterState.depthBiasEnable;
        rasterizer.depthBiasConstantFactor = rasterState.depthBiasConstantFactor;
        rasterizer.depthBiasSlopeFactor = rasterState.depthBiasSlopeFactor;

        // Settles depth stencil state
        const DepthState& depthState = GraphicsContext::GetDepthState(_pipelineState.depthState);
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = depthState.testEnable;
        depthStencil.depthWriteEnable = depthState.writeEnable;
        depthStencil.depthCompareOp = ToVk(depthState.compareFunc);
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        // Settles multisample state
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 0.2f;

        // Settles color blend state
        const BlendState& blendState = GraphicsContext::GetBlendState(_pipelineState.blendState);
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = blendState.blendEnable;
        colorBlendAttachment.srcColorBlendFactor = ToVk(blendState.rgbSrcFactor);
        colorBlendAttachment.dstColorBlendFactor = ToVk(blendState.rgbDstFactor);
        colorBlendAttachment.colorBlendOp = ToVk(blendState.blendFunc);
        colorBlendAttachment.srcAlphaBlendFactor = ToVk(blendState.alphaSrcFactor);
        colorBlendAttachment.dstAlphaBlendFactor = ToVk(blendState.alphaDstFactor);
        colorBlendAttachment.alphaBlendOp = ToVk(blendState.blendFunc);

        if (blendState.redWrite) {
            colorBlendAttachment.colorWriteMask |= VK_COLOR_COMPONENT_R_BIT;
        }
        if (blendState.greenWrite) {
            colorBlendAttachment.colorWriteMask |= VK_COLOR_COMPONENT_G_BIT;
        }
        if (blendState.blueWrite) {
            colorBlendAttachment.colorWriteMask |= VK_COLOR_COMPONENT_B_BIT;
        }
        if (blendState.alphaWrite) {
            colorBlendAttachment.colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;
        }

        auto fragmentShader = (VkShader*)GetShader(ShaderStage::Fragment);
        AVA_ASSERT(fragmentShader, "[ShaderProg] a graphics program requires a valid fragment shader.");

        // Checks the program contains the right number of color outputs
        const u32 attachmentsCount = (u32)fragmentShader->GetReflectionData().colorOutputs.size();

        AVA_ASSERT(attachmentsCount == _pipelineState.colorAttachments,
            "[ShaderProg] %s is expecting to write to %d color attachments, but current framebuffer only contains %d.",
            fragmentShader->GetResourcePath(), attachmentsCount, _pipelineState.colorAttachments);

        // We use the same blend parameters for each attachments
        std::vector blendAttachments(attachmentsCount, colorBlendAttachment);

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = attachmentsCount;
        colorBlending.pAttachments = blendAttachments.data();
        colorBlending.blendConstants[0] = 1.f;
        colorBlending.blendConstants[1] = 1.f;
        colorBlending.blendConstants[2] = 1.f;
        colorBlending.blendConstants[3] = 1.f;

        // Builds pipeline object
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = (u32)shaderStages.size();
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_graphicsLayout;
        pipelineInfo.renderPass = _pipelineState.renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        VkPipeline pipeline;
        const VkResult result = vkCreateGraphicsPipelines(VkGraphicsContext::GetDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create graphics pipeline: %s", VkGraphicsContext::GetVkResultStr(result));

        return pipeline;
    }

    void VkShaderProgram::_BuildComputeLayout()
    {
        AVA_ASSERT(IsCompute(), "[ShaderProg] the program %s is not compute compatible.", GetResourceName());

        std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
        GetDescriptorSetLayouts(descriptorSetLayouts);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = (u32)descriptorSetLayouts.size();
        pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        const VkResult result = vkCreatePipelineLayout(VkGraphicsContext::GetDevice(), &pipelineLayoutInfo, nullptr, &m_computeLayout);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create compute pipeline layout: %s", VkGraphicsContext::GetVkResultStr(result));
    }

    VkPipeline VkShaderProgram::_BuildComputePipeline()
    {
        AVA_ASSERT(IsCompute(), "[ShaderProg] the program %s is not compute compatible.", GetResourceName());

        if (!m_computeLayout)
        {
            _BuildComputeLayout();
        }

        const auto* shader = (VkShader*)GetShader(ShaderStage::Compute);
        AVA_ASSERT(shader, "[ShaderProg] a compute program requires a valid compute shader.");

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shader->GetModule();
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = m_computeLayout;
        pipelineInfo.basePipelineIndex = -1;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        VkPipeline pipeline;
        const VkResult result = vkCreateComputePipelines(VkGraphicsContext::GetDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create compute pipeline: %s", VkGraphicsContext::GetVkResultStr(result));

        return pipeline;
    }

}
