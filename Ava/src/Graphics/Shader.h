#pragma once
/// @file Shader.h
/// @brief interface for shaders.

#include <Graphics/GraphicsCommon.h>
#include <Resources/Resource.h>
#include <Resources/ShaderData.h>

namespace Ava {

    constexpr u8 kShaderSlotInvalid = 0xFF;

    /// @brief Single shader resource object.
    class Shader : public Resource
    {
        friend class ShaderResourceMgr;

    public:
        Shader() = default;
        ~Shader() override = default;

        /// @brief for path-based shader resources.
        explicit Shader(ShaderStage::Enum _stage, const ShaderDescription& _desc);
        /// @brief for built-in shader data.
        explicit Shader(const ShaderData& _data);

        virtual void SetDebugName(const char* _name) { }
        const char* GetResourcePath() const { return GetResourceName(); }

        // Loading is handled by resource manager to stay sync with graphics context.
        void Load() override {}
        void Release() override {}

        ShaderStage::Enum GetStage() const { return m_stage; }
        const ShaderDescription& GetDescription() const { return m_desc; }

    protected:
        ShaderStage::Enum m_stage;
        ShaderDescription m_desc;
    };

    /// @brief Custom shader resource manager.
    class ShaderResourceMgr final : public ResourceMgr<Shader, ShaderDescription>
    {
    public:
        ShaderResourceMgr() = default;
        ShaderResourceMgr(const ShaderStage::Enum _stage) : m_stage(_stage) {}
        ~ShaderResourceMgr() override = default;

        Resource* CreateResource(const void* _constructionData) override;
        void DestroyResource(Resource* _resource) override;
        ResourceID BuildResourceID(const void* _constructionData) override;
        bool HotReloadResource(ResourceID _resourceID) override;

    private:
        ShaderStage::Enum m_stage;
    };

    /// @brief Shader resource helper, to handle path based Shader resources.
    class ShaderResH : public ResH<ShaderResourceMgr>
    {
        typedef ResH<ShaderResourceMgr> Super;

    public:
        static ResourceMgrBase* GetResourceMgr(ShaderStage::Enum _stage);

        static Shader* Use(ShaderStage::Enum _stage, const ShaderDescription* _desc);
        static void Use(Shader* _shader);

        static void Load(ShaderStage::Enum _stage, ResourceID _resID);
        static void Load(Shader* _shader);

        static void Release(ShaderStage::Enum _stage, ResourceID _resID);
        static void Release(Shader* _shader);

        static void Unuse(ShaderStage::Enum _stage, ResourceID _resID);
        static void Unuse(Shader* _shader);

        static Shader* UseLoad(ShaderStage::Enum _stage, const ShaderDescription* _desc);
        static Shader* UseLoad(Shader* _shader);

        static void ReleaseUnuse(ShaderStage::Enum _stage, ResourceID _resID);
        static void ReleaseUnuse(Shader* _shader);

        static bool HotReload(ShaderStage::Enum _stage, ResourceID _resID);
        static bool ResourceExists(ShaderStage::Enum _stage, ResourceID _resID);

    private:
        static ShaderResourceMgr s_resourceMgr[ShaderStage::Count];
    };

}
