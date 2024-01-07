#include <avapch.h>
#include "Resource.h"

#include <Debug/Assert.h>

namespace Ava {

    //---- Resource Manager -------------------------------------------------

    void ResourceMgrBase::ListAllResourceIDs(std::vector<ResourceID>& _resIDs)
    {
        _resIDs.reserve(m_allocations.size());
        for (const auto& alloc : m_allocations)
        {
            _resIDs.push_back(alloc.first);
        }
    }

    bool ResourceMgrBase::ResourceExists(ResourceID _resID)
    {
        return m_allocations.find(_resID) != m_allocations.end();
    }

    Resource* ResourceMgrBase::GetResource(const ResourceID _resID)
    {
        AVA_ASSERT(_resID != kResourceIdxInvalid);
        const auto it = m_allocations.find(_resID);

        if (it != m_allocations.end())
        {
            return it->second.resource;
        }
        return nullptr;
    }

    bool ResourceMgrBase::HotReloadResource(ResourceID _resID)
    {
        return false;
    }

    void ResourceMgrBase::UseResource(Resource* _resource)
    {
        AVA_ASSERT(_resource);
        Allocation& alloc = m_allocations[_resource->GetResourceID()];

        if (alloc.nbUse++ == 0)
        {
            AVA_ASSERT(!alloc.resource);
            alloc.resource = _resource;
        }
        AVA_ASSERT(alloc.resource == _resource);
    }

    Resource* ResourceMgrBase::UseResource(const void* _constructionData)
    {
        AVA_ASSERT(_constructionData);
        const ResourceID id = BuildResourceID(_constructionData);
        Allocation& alloc = m_allocations[id];

        if (alloc.nbUse++ == 0)
        {
            AVA_ASSERT(!alloc.resource);
            alloc.resource = CreateResource(_constructionData);
            alloc.resource->SetResourceID(id);
        }
        AVA_ASSERT(alloc.resource);
        return alloc.resource;
    }

    void ResourceMgrBase::LoadResource(Resource* _resource)
    {
        AVA_ASSERT(_resource);
        Allocation& alloc = m_allocations[_resource->GetResourceID()];

        AVA_ASSERT(alloc.resource == _resource);
        AVA_ASSERT(alloc.nbUse > 0);

        if (alloc.nbLoad++ == 0)
        {
            // Hack to make the resource system work for
            // assets that were already manually loaded.
            if (_resource->GetResourceState() == ResourceState::Loaded)
            {
                return;
            }

            _resource->Load();
        }
    }

    void ResourceMgrBase::LoadResource(const ResourceID _resID)
    {
        AVA_ASSERT(_resID != kResourceIdxInvalid);
        Allocation& alloc = m_allocations[_resID];

        AVA_ASSERT(alloc.resource);
        AVA_ASSERT(alloc.nbUse > 0);

        if (alloc.nbLoad++ == 0)
        {
            alloc.resource->Load();
        }
    }

    void ResourceMgrBase::ReleaseResource(Resource* _resource)
    {
        AVA_ASSERT(_resource);
        Allocation& alloc = m_allocations[_resource->GetResourceID()];

        AVA_ASSERT(alloc.resource == _resource);
        AVA_ASSERT(alloc.nbUse > 0 && alloc.nbLoad > 0);

        if (--alloc.nbLoad == 0)
        {
            _resource->Release();
        }
    }

    void ResourceMgrBase::ReleaseResource(const ResourceID _resID)
    {
        AVA_ASSERT(_resID != kResourceIdxInvalid);
        Allocation& alloc = m_allocations[_resID];

        AVA_ASSERT(alloc.resource);
        AVA_ASSERT(alloc.nbUse > 0 && alloc.nbLoad > 0);

        if (--alloc.nbLoad == 0)
        {
            alloc.resource->Release();
        }
    }

    void ResourceMgrBase::UnuseResource(Resource* _resource)
    {
        AVA_ASSERT(_resource);
        Allocation& alloc = m_allocations[_resource->GetResourceID()];

        AVA_ASSERT(alloc.resource == _resource);
        AVA_ASSERT(alloc.nbUse > 0);

        if (--alloc.nbUse == 0)
        {
            AVA_ASSERT(alloc.nbLoad == 0);
            m_allocations.erase(_resource->GetResourceID());
            DestroyResource(_resource);
        }
    }

    void ResourceMgrBase::UnuseResource(const ResourceID _resID)
    {
        AVA_ASSERT(_resID != kResourceIdxInvalid);
        Allocation& alloc = m_allocations[_resID];

        AVA_ASSERT(alloc.resource);
        AVA_ASSERT(alloc.nbUse > 0);

        if (--alloc.nbUse == 0)
        {
            AVA_ASSERT(alloc.nbLoad == 0);
            DestroyResource(alloc.resource);
            m_allocations.erase(_resID);
        }
    }


    //---- Resource Helper---------------------------------------------------

    Resource* ResHBase::UseLoad(ResourceMgrBase* _resMgr, const void* _constructionData)
    {
        Resource* resource = Use(_resMgr, _constructionData);
        Load(_resMgr, resource);
        return resource;
    }

    Resource* ResHBase::UseLoad(ResourceMgrBase* _resMgr, Resource* _resource)
    {
        Use(_resMgr, _resource);
        Load(_resMgr, _resource);
        return _resource;
    }

    void ResHBase::ReleaseUnuse(ResourceMgrBase* _resMgr, const ResourceID _resID)
    {
        Release(_resMgr, _resID);
        Unuse(_resMgr, _resID);
    }

    void ResHBase::ReleaseUnuse(ResourceMgrBase* _resMgr, Resource* _resource)
    {
        Release(_resMgr, _resource);
        Unuse(_resMgr, _resource);
    }


}