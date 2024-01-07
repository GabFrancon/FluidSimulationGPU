#pragma once
/// @file Resource.h
/// @brief

#include <Core/Base.h>
#include <Math/Hash.h>

namespace Ava {

    class Resource;
    class ResourceMgrBase;
    class ResHBase;

    enum class ResourceState
    {
        NotLoaded = 0,
        Loaded,
        LoadFailed,
        Released
    };

    using ResourceID = u32;
    static constexpr ResourceID kResourceIdxInvalid = 0xFFFFFFFF;

    /// @brief Resource interface.
    class Resource
    {
        friend class ResourceMgrBase;

    public:
        Resource() = default;
        virtual ~Resource() = default;

        Resource(const Resource&) = delete;
        Resource& operator=(const Resource&) = delete;

        virtual void Load() = 0;
        virtual void Release() = 0;

        const char* GetResourceName() const { return m_resourceName.c_str(); }
        ResourceID GetResourceID() const { return m_resourceID; }
        ResourceState GetResourceState() const { return m_resourceState; }

    protected:
        void SetResourceName(const char* _name) { m_resourceName = _name; }
        void SetResourceID(const ResourceID _resID) { m_resourceID = _resID; }
        void SetResourceState(const ResourceState _state) { m_resourceState = _state; }

        std::string m_resourceName;
        ResourceID m_resourceID = kResourceIdxInvalid;
        ResourceState m_resourceState = ResourceState::NotLoaded;
    };

    /// @brief Resource manager interface.
    class ResourceMgrBase
    {
    public:
        ResourceMgrBase() = default;
        virtual ~ResourceMgrBase() = default;

        virtual Resource* CreateResource(const void* _constructionData) = 0;
        virtual void DestroyResource(Resource* _resource) = 0;

        virtual ResourceID BuildResourceID(const void* _constructionData) = 0;
        virtual void ListAllResourceIDs(std::vector<ResourceID>& _resIDs);

        virtual bool ResourceExists(ResourceID _resID);
        virtual Resource* GetResource(ResourceID _resID);
        virtual bool HotReloadResource(ResourceID _resID);

        virtual void UseResource(Resource* _resource);
        virtual Resource* UseResource(const void* _constructionData);

        virtual void LoadResource(Resource* _resource);
        virtual void LoadResource(ResourceID _resID);

        virtual void ReleaseResource(Resource* _resource);
        virtual void ReleaseResource(ResourceID _resID);

        virtual void UnuseResource(Resource* _resource);
        virtual void UnuseResource(ResourceID _resID);

    protected:
        struct Allocation
        {
            int nbUse = 0;
            int nbLoad = 0;
            Resource* resource = nullptr;
        };
        std::map<ResourceID, Allocation> m_allocations;
    };

    /// @brief Base for custom resource managers.
    ///    @param T The resource type, must inherit from Resource class.
    ///    @param P The data structure used to build the resource.
    template <class T, class P>
    class ResourceMgr : public ResourceMgrBase
    {
        typedef ResourceMgrBase Super;

    public:
        typedef T ResourceType;
        typedef P ConstructionDataType;
    };

    /// @brief ResourceMgr implementing all pure functions remaining.
    ///    @param T The resource type, must inherit from Resource class.
    ///    @param P The data struct passed as a const pointer to the resource constructor.
    template <class T, class P>
    class DefaultResourceMgr final : public ResourceMgr<T, P>
    {
        typedef ResourceMgr<T, P> Super;

    public:
        using typename Super::ResourceType;
        using typename Super::ConstructionDataType;

        Resource* CreateResource(const void* _constructionData) override
        {
            const auto* data = static_cast<const ConstructionDataType*>(_constructionData);
            return new ResourceType(data);
        }

        void DestroyResource(Resource* _resource) override
        {
            delete _resource;
        }

        ResourceID BuildResourceID(const void* _constructionData) override
        {
            return HashStr(static_cast<const char*>(_constructionData));
        }

        bool HotReloadResource(const ResourceID _resID) override
        {
            auto* resource = static_cast<ResourceType*>(Super::GetResource(_resID));
            if (!resource)
            {
                return false;
            }
            resource->Release();
            resource->Load();
            return true;
        }
    };

    /// @brief Static resource helper.
    class ResHBase
    {
    public:
        static ResourceID GetResourceID(ResourceMgrBase* _resMgr, const void* _constructionData) { return _resMgr->BuildResourceID(_constructionData); }
        static void ListAllResourceIDs(ResourceMgrBase* _resMgr, std::vector<ResourceID>& _resIDs) { return _resMgr->ListAllResourceIDs(_resIDs); }

        static bool ResourceExists(ResourceMgrBase* _resMgr, ResourceID _resID) { return _resMgr->ResourceExists(_resID); }
        static Resource* GetResource(ResourceMgrBase* _resMgr, const ResourceID _resID) { return _resMgr->GetResource(_resID); }
        static bool HotReload(ResourceMgrBase* _resMgr, const ResourceID _resID) { return _resMgr->HotReloadResource(_resID); }

        static Resource* Use(ResourceMgrBase* _resMgr, const void* _constructionData) { return _resMgr->UseResource(_constructionData); }
        static void Use(ResourceMgrBase* _resMgr, Resource* _resource) { _resMgr->UseResource(_resource); }

        static void Load(ResourceMgrBase* _resMgr, const ResourceID _resID) { _resMgr->LoadResource(_resID); }
        static void Load(ResourceMgrBase* _resMgr, Resource* _resource) { _resMgr->LoadResource(_resource); }

        static void Release(ResourceMgrBase* _resMgr, const ResourceID _resID) { _resMgr->ReleaseResource(_resID); }
        static void Release(ResourceMgrBase* _resMgr, Resource* _resource) { _resMgr->ReleaseResource(_resource); }

        static void Unuse(ResourceMgrBase* _resMgr, const ResourceID _resID) { _resMgr->UnuseResource(_resID); }
        static void Unuse(ResourceMgrBase* _resMgr, Resource* _resource) { _resMgr->UnuseResource(_resource); }

        static Resource* UseLoad(ResourceMgrBase* _resMgr, const void* _constructionData);
        static Resource* UseLoad(ResourceMgrBase* _resMgr, Resource* _resource);

        static void ReleaseUnuse(ResourceMgrBase* _resMgr, ResourceID _resID);
        static void ReleaseUnuse(ResourceMgrBase* _resMgr, Resource* _resource);
    };

    /// @brief ResH template version, to be used with default or custom ResourceMgr.
    /// @param T the resource manager associated, can be a DefaultResourceMgr or a custom manager inherited from ResourceMgr.
    template <class T>
    class ResH : public ResHBase
    {
        typedef ResHBase Super;

    public:
        typedef T ResourceMgrType;
        typedef typename T::ResourceType ResourceType;
        typedef typename T::ConstructionDataType ConstructionDataType;

        static ResourceMgrBase* GetResourceMgr();
        static void SetResourceMgr(ResourceMgrBase* _resMgr);

        static ResourceID GetResourceID(const ConstructionDataType* _constructionData) { return Super::GetResourceID(GetResourceMgr(), (const void*)_constructionData); }
        static void ListAllResourceIDs(std::vector<ResourceID>& _resIDs) { return Super::ListAllResourceIDs(GetResourceMgr(), _resIDs); }

        static bool ResourceExists(const ResourceID _resID) { return Super::ResourceExists(GetResourceMgr(), _resID); }
        static ResourceType* GetResource(const ResourceID _resID) { return (ResourceType*)Super::GetResource(GetResourceMgr(), _resID); }
        static bool HotReload(const ResourceID _resID) { return Super::HotReload(GetResourceMgr(), _resID); }

        static ResourceType* Use(const ConstructionDataType* _constructionData) { return (ResourceType*)Super::Use(GetResourceMgr(), (const void*)_constructionData); }
        static void Use(const ResourceType* _resource) { if (_resource) Super::Use(GetResourceMgr(), (Resource*)_resource); }

        static void Load(const ResourceID _resID) { Super::Load(GetResourceMgr(), _resID); }
        static void Load(ResourceType* _resource) { if (_resource) Super::Load(GetResourceMgr(), (Resource*)_resource); }

        static void Release(const ResourceID _resID) { Super::Release(GetResourceMgr(), _resID); }
        static void Release(ResourceType* _resource) { if (_resource) Super::Release(GetResourceMgr(), (Resource*)_resource); }

        static void Unuse(const ResourceID _resID) { Super::Unuse(GetResourceMgr(), _resID); }
        static void Unuse(ResourceType* _resource) { if (_resource) Super::Unuse(GetResourceMgr(), (Resource*)_resource); }

        static ResourceType* UseLoad(const ConstructionDataType* _constructionData) { return (ResourceType*)Super::UseLoad(GetResourceMgr(), (const void*)_constructionData); }
        static ResourceType* UseLoad(const ResourceType* _resource) { return (ResourceType*)Super::UseLoad(GetResourceMgr(), (Resource*)_resource); }

        static void ReleaseUnuse(const ResourceID _resID) { Super::ReleaseUnuse(GetResourceMgr(), _resID); }
        static void ReleaseUnuse(ResourceType* _resource) { if (_resource) Super::ReleaseUnuse(GetResourceMgr(), (Resource*)_resource); }

    private:
        static ResourceMgrType s_defaultMgr;
        static ResourceMgrBase* s_customMgr;
    };

    template <class T>
    typename ResH<T>::ResourceMgrType ResH<T>::s_defaultMgr{};

    template <class T>
    ResourceMgrBase* ResH<T>::s_customMgr = nullptr;

    template <class T>
    ResourceMgrBase* ResH<T>::GetResourceMgr() 
    {
        return s_customMgr ? s_customMgr : &s_defaultMgr;
    }

    template <class T>
    void ResH<T>::SetResourceMgr(ResourceMgrBase* _resMgr)
    {
        s_customMgr = _resMgr;
    }

}
