#pragma once
/// @file ShaderProgram.h
/// @brief interface for pipeline objects.

#include <Graphics/Shader.h>
#include <Resources/Resource.h>

namespace Ava {

    /// @brief List of shader stages a render pipeline can execute.
    class ShaderProgram : public Resource
    {
        friend class ShaderProgramResourceMgr;

    public:
        ShaderProgram() = default;
        ~ShaderProgram() override;

        /// @brief for path-based shader resources.
        explicit ShaderProgram(const ShaderProgramDescription& _desc);
        /// @brief for built-in shader objects.
        explicit ShaderProgram(Shader* _stages[ShaderStage::Count], const ShaderResourceNames& _varNames);

        virtual void SetDebugName(const char* _name) { }

        // Loading is handled by resource manager to stay sync with graphics context.
        void Load() override {}
        void Release() override {}

        u32 GetShaderCount() const;
        Shader* GetShader(ShaderStage::Enum _stage) const;
        const ShaderResourceNames& GetShaderVarNames() const { return m_varNames; }
        bool IsUsingShader(ShaderStage::Enum _stage, ResourceID _shaderId) const;
        bool IsCompute() const;

    protected:
        Shader* m_shaders[ShaderStage::Count];
        ShaderResourceNames m_varNames;
    };

    /// @brief Custom shader program resource manager.
    class ShaderProgramResourceMgr final : public ResourceMgr<ShaderProgram, ShaderProgramDescription>
    {
    public:
        Resource* CreateResource(const void* _constructionData) override;
        void DestroyResource(Resource* _resource) override;
        ResourceID BuildResourceID(const void* _constructionData) override;
        bool HotReloadResource(ResourceID _resourceID) override;
    };

    /// @brief Shader program resource helper.
    class ShaderProgramResH : public ResH<ShaderProgramResourceMgr>
    {
    };
}

