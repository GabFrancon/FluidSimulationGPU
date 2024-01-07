#include <avapch.h>
#include "ShaderProgram.h"

#include <Graphics/GraphicsContext.h>
#include <Debug/Assert.h>
#include <Debug/Log.h>

namespace Ava {

    //-------------- ShaderProgram implementation --------------------------------------------------------

    ShaderProgram::ShaderProgram(const ShaderProgramDescription& _desc)
    {
        const std::string resourceName = _desc.BuildProgramName();
        SetResourceName(resourceName.c_str());

        m_varNames = _desc.resourceNames;
        bool success = true;

        for (u8 i = 0; i < ShaderStage::Count; i++)
        {
            const auto stage = ShaderStage::Enum(i);
            const ShaderDescription& shaderDesc = _desc.shaders[stage];

            if (shaderDesc.IsValid())
            {
                Shader* shader = ShaderResH::UseLoad(stage, &shaderDesc);
                m_shaders[stage] = shader;

                if (!AVA_VERIFY(
                    shader->GetResourceState() == ResourceState::Loaded,
                    "Shader %s was not loaded.", shader->GetResourcePath()))
                {
                    success = false;
                    continue;
                }

                shader->SetDebugName(shaderDesc.path.c_str());
            }
            else
            {
                m_shaders[stage] = nullptr;
            }
        }

        SetResourceState(success ? ResourceState::Loaded : ResourceState::LoadFailed);
    }

    ShaderProgram::ShaderProgram(Shader* _stages[ShaderStage::Count], const ShaderResourceNames& _varNames)
    {
        m_varNames = _varNames;

        bool loadError = false;
        bool atLeastOneValid = false;
        ResourceID resId = HashU32Init();

        for (int i = 0; i < ShaderStage::Count; i++)
        {
            Shader* shader = _stages[i];

            if (shader == nullptr)
            {
                m_shaders[i] = nullptr;
                continue;
            }

            if (!AVA_VERIFY(shader->GetResourceState() == ResourceState::Loaded, 
                "[ShaderProg] shader %s was not loaded.", shader->GetResourcePath()))
            {
                loadError = true;
            }

            m_shaders[i] = shader;
            ShaderResH::UseLoad(shader);

            resId = HashU32Combine(shader->GetResourceID(), resId);
            atLeastOneValid = true;
        }

        const bool success = atLeastOneValid && !loadError;
        SetResourceState(success ? ResourceState::Loaded : ResourceState::LoadFailed);
        SetResourceID(resId);
    }

    ShaderProgram::~ShaderProgram()
    {
        for (u8 i = 0; i < ShaderStage::Count; i++)
        {
            const auto stage = ShaderStage::Enum(i);

            if (Shader* shader = GetShader(stage))
            {
                ShaderResH::ReleaseUnuse(shader);
            }
        }
    }

    u32 ShaderProgram::GetShaderCount() const
    {
        u32 count = 0;
        for (u8 i = 0; i < ShaderStage::Count; i++)
        {
            const auto stage = ShaderStage::Enum(i);
            if (GetShader(stage))
            {
                count++;
            }
        }
        return count;
    }

    Shader* ShaderProgram::GetShader(const ShaderStage::Enum _stage) const
    {
        return m_shaders[_stage];
    }

    bool ShaderProgram::IsUsingShader(const ShaderStage::Enum _stage, const ResourceID _shaderId) const
    {
        if (const Shader* shader = GetShader(_stage))
        {
            return shader->GetResourceID() == _shaderId;
        }
        return false;
    }

    bool ShaderProgram::IsCompute() const
    {
        return GetShader(ShaderStage::Compute) != nullptr;
    }


    //-------------- ShaderProgramResourceMgr implementation ----------------------------------------------

    Resource* ShaderProgramResourceMgr::CreateResource(const void* _constructionData)
    {
        const auto* programDesc = static_cast<const ShaderProgramDescription*>(_constructionData);
        ShaderProgram* program = GraphicsContext::CreateProgram(*programDesc);
        return program;
    }

    void ShaderProgramResourceMgr::DestroyResource(Resource* _resource)
    {
        auto* program = static_cast<ShaderProgram*>(_resource);
        GraphicsContext::DestroyProgram(program);
    }

    ResourceID ShaderProgramResourceMgr::BuildResourceID(const void* _constructionData)
    {
        const auto* programDesc = static_cast<const ShaderProgramDescription*>(_constructionData);
        return programDesc->BuildProgramID();
    }

    bool ShaderProgramResourceMgr::HotReloadResource(const ResourceID _resourceID)
    {
        // Checks if resource exists.
        auto* program = static_cast<ShaderProgram*>(GetResource(_resourceID));
        if (!program)
        {
            AVA_CORE_WARN("[ShaderProg] No program with ID '%u' found.", _resourceID);
            return false;
        }

        // Rebuilds shader program description.
        ShaderProgramDescription programDesc;
        programDesc.resourceNames = program->GetShaderVarNames();

        for (u8 i = 0; i < ShaderStage::Count; i++)
        {
            const auto stage = ShaderStage::Enum(i);
            if (const Shader* shader = program->GetShader(stage))
            {
                programDesc.shaders[stage] = shader->GetDescription();
            }
        }

        // Waits to sync with graphics context.
        GraphicsContext::WaitIdle();

        // Reconstructs program with new shader.
        ReleaseResource(program);
        GraphicsContext::DestructProgram(program);
        GraphicsContext::ConstructProgram(program, programDesc);
        program->SetResourceID(_resourceID);
        LoadResource(program);

        return true;
    }

}
