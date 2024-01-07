#include <avapch.h>
#include "Shader.h"

#include <Graphics/GraphicsContext.h>
#include <Debug/Assert.h>
#include <Debug/Log.h>

namespace Ava {

    //---------------- Shader implementation --------------------------------------------------------------------------

    Shader::Shader(const ShaderStage::Enum _stage, const ShaderDescription& _desc)
    {
        AVA_ASSERT(_desc.IsValid(), "[Shader] trying to load an invalid shader resource.");
        const std::string resourcePath = _desc.BuildResourcePath(_stage);
        SetResourceName(resourcePath.c_str());
        m_stage = _stage;
        m_desc = _desc;
    }

    Shader::Shader(const ShaderData& _data)
    {
        m_stage = _data.stage;
        const char* resourceName = _data.resourceName.c_str();

        SetResourceID(HashStr(resourceName));
        SetResourceName(resourceName);
    }


    //---------------- ShaderResourceMgr implementation ---------------------------------------------------------------

    Resource* ShaderResourceMgr::CreateResource(const void* _constructionData)
    {
        const auto* shaderDesc = static_cast<const ShaderDescription*>(_constructionData);
        Shader* shader = GraphicsContext::CreateShader(m_stage, *shaderDesc);
        return shader;
    }

    void ShaderResourceMgr::DestroyResource(Resource* _resource)
    {
        auto* shader = static_cast<Shader*>(_resource);
        GraphicsContext::DestroyShader(shader);
    }

    ResourceID ShaderResourceMgr::BuildResourceID(const void* _constructionData)
    {
        const auto* shaderDesc = static_cast<const ShaderDescription*>(_constructionData);
        return shaderDesc->BuildShaderId();
    }

    bool ShaderResourceMgr::HotReloadResource(const ResourceID _resourceID)
    {
        // Checks if resource exists.
        auto* shader = static_cast<Shader*>(GetResource(_resourceID));
        if (!shader)
        {
            AVA_CORE_WARN("[Shader] No shader with ID '%u' found.", _resourceID);
            return false;
        }

        // Rebuilds shader description.
        const ShaderDescription shaderDesc = shader->GetDescription();

        // Waits to sync with graphics context.
        GraphicsContext::WaitIdle();
        
        // Reconstructs shader resource.
        ReleaseResource(shader);
        GraphicsContext::DestructShader(shader);
        GraphicsContext::ConstructShader(shader, m_stage, shaderDesc);
        shader->SetResourceID(_resourceID);
        LoadResource(shader);

        return true;
    }


    //---------------- ShaderResH implementation ----------------------------------------------------------------------

    ShaderResourceMgr ShaderResH::s_resourceMgr[ShaderStage::Count] = {
        ShaderResourceMgr(ShaderStage::Vertex),
        ShaderResourceMgr(ShaderStage::Geometric),
        ShaderResourceMgr(ShaderStage::Fragment),
        ShaderResourceMgr(ShaderStage::Compute)
    };

    ResourceMgrBase* ShaderResH::GetResourceMgr(const ShaderStage::Enum _stage)
    {
        return &s_resourceMgr[_stage];
    }

    Shader* ShaderResH::Use(const ShaderStage::Enum _stage, const ShaderDescription* _desc)
    {
        return (Shader*)ResHBase::Use(GetResourceMgr(_stage), _desc);
    }

    void ShaderResH::Use(Shader* _shader)
    {
        ResHBase::Use(GetResourceMgr(_shader->GetStage()), _shader);
    }

    void ShaderResH::Load(const ShaderStage::Enum _stage, const ResourceID _resID)
    {
        ResHBase::Load(GetResourceMgr(_stage), _resID);
    }

    void ShaderResH::Load(Shader* _shader)
    {
        ResHBase::Load(GetResourceMgr(_shader->GetStage()), _shader);
    }

    void ShaderResH::Release(const ShaderStage::Enum _stage, const ResourceID _resID)
    {
        ResHBase::Load(GetResourceMgr(_stage), _resID);
    }

    void ShaderResH::Release(Shader* _shader)
    {
        ResHBase::Release(GetResourceMgr(_shader->GetStage()), _shader);
    }

    void ShaderResH::Unuse(const ShaderStage::Enum _stage, const ResourceID _resID)
    {
        ResHBase::Unuse(GetResourceMgr(_stage), _resID);
    }

    void ShaderResH::Unuse(Shader* _shader)
    {
        ResHBase::Unuse(GetResourceMgr(_shader->GetStage()), _shader);
    }

    Shader* ShaderResH::UseLoad(const ShaderStage::Enum _stage, const ShaderDescription* _desc)
    {
        return (Shader*)ResHBase::UseLoad(GetResourceMgr(_stage), _desc);
    }

    Shader* ShaderResH::UseLoad(Shader* _shader)
    {
        return (Shader*)ResHBase::UseLoad(GetResourceMgr(_shader->GetStage()), _shader);
    }

    void ShaderResH::ReleaseUnuse(const ShaderStage::Enum _stage, const ResourceID _resID)
    {
        ResHBase::ReleaseUnuse(GetResourceMgr(_stage), _resID);
    }

    void ShaderResH::ReleaseUnuse(Shader* _shader)
    {
        ResHBase::ReleaseUnuse(GetResourceMgr(_shader->GetStage()), _shader);
    }

    bool ShaderResH::HotReload(const ShaderStage::Enum _stage, const ResourceID _resID)
    {
        return ResHBase::HotReload(GetResourceMgr(_stage), _resID);
    }

    bool ShaderResH::ResourceExists(const ShaderStage::Enum _stage, const ResourceID _resID)
    {
        return ResHBase::ResourceExists(GetResourceMgr(_stage), _resID);
    }

}
