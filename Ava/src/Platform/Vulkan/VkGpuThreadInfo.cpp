#include <avapch.h>
#include "VkGpuThreadInfo.h"

#include <Math/Math.h>
#include <Debug/Assert.h>
#include <Graphics/Color.h>
#include <Time/TimeManager.h>
#include <Platform/Vulkan/VkGraphicsContext.h>

namespace Ava {

    void VkGpuThreadInfo::Init(const u32 _frameCount, const u32 _markerCount)
    {
        GpuThreadInfo::Init(_frameCount, _markerCount);

        m_cpuFrameStartTimes = new u64[m_numFramesToRecord];
        m_frameStartTimes = new u64[m_numFramesToRecord];
        m_markerStartTimes = new u64[m_totalMarkerCount];
        m_markerEndTimes = new u64[m_totalMarkerCount];

        memset(m_cpuFrameStartTimes, 0, m_numFramesToRecord * sizeof(u64));
        memset(m_frameStartTimes, 0, m_numFramesToRecord * sizeof(u64));
        memset(m_markerStartTimes, 0, m_totalMarkerCount * sizeof(u64));
        memset(m_markerEndTimes, 0, m_totalMarkerCount * sizeof(u64));

        const VkDevice device = VkGraphicsContext::GetDevice();

        VkQueryPoolCreateInfo queryCreateInfo{};
        queryCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;

        queryCreateInfo.queryCount = m_numFramesToRecord;
        VkResult result = vkCreateQueryPool(device, &queryCreateInfo, nullptr, &m_frameStartQueryPool);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create frame start query pool.");

        queryCreateInfo.queryCount = m_totalMarkerCount;
        result = vkCreateQueryPool(device, &queryCreateInfo, nullptr, &m_markerStartQueryPool);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create marker start query pool.");

        result = vkCreateQueryPool(device, &queryCreateInfo, nullptr, &m_markerEndQueryPool);
        AVA_ASSERT(result == VK_SUCCESS, "[Vulkan] failed to create marker end query pool.");

        VkGraphicsContext::SetDebugObjectName(m_frameStartQueryPool, VK_OBJECT_TYPE_QUERY_POOL, "GpuThreadInfo::m_frameStartQueryPool");
        VkGraphicsContext::SetDebugObjectName(m_markerStartQueryPool, VK_OBJECT_TYPE_QUERY_POOL, "GpuThreadInfo::m_markerStartQueryPool");
        VkGraphicsContext::SetDebugObjectName(m_markerEndQueryPool, VK_OBJECT_TYPE_QUERY_POOL, "GpuThreadInfo::m_markerEndQueryPool");

    }

    void VkGpuThreadInfo::StartFrame(const u32 _frameId, GraphicsContext* _ctx)
    {
        GpuThreadInfo::StartFrame(_frameId);
        m_ctx = _ctx;

        const VkDevice device = VkGraphicsContext::GetDevice();
        const VkCommandBuffer cmd = m_ctx->GetImpl()->GetCommandBuffer();

        // Updates CPU / GPU delta time
        _UpdateCpuGpuDeltaTime(m_ctx);

        // Lets the first frames run run without querying.
        m_gpuMarkersReady = m_currentFrame > 1;
        if (!m_gpuMarkersReady)
        {
            // Reset query pools
            vkCmdResetQueryPool(cmd, m_markerStartQueryPool, 0, m_totalMarkerCount);
            vkCmdResetQueryPool(cmd, m_markerEndQueryPool, 0, m_totalMarkerCount);
        }

        // Emits timestamp command
        const u32 frameStartIndex = _GetFrameStartIndex(m_currentFrame);
        vkCmdResetQueryPool(cmd, m_frameStartQueryPool, frameStartIndex, 1);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, m_frameStartQueryPool, frameStartIndex);

        // Saves current CPU timestamp
        m_cpuFrameStartTimes[frameStartIndex] = TimeMgr::GetTimeNS();

        // Tries to get times for the past frames
        if (m_currentFrame > m_numFramesToRecord)
        {
            u32 frame = Math::max(m_nextFrameToQuery, _GetOldestFrame());
            const u32 maxFrame = m_currentFrame - GraphicsContext::GetContextCount();

            for (; frame < maxFrame; ++frame)
            {
                const u32 firstMarker = GetFirstMarker(frame);

                // If no markers, nothing to do
                if (!IsMarkerValid(firstMarker))
                {
                    continue;
                }

                const u32 nextIdx = _GetFrameStartIndex(frame + 1);

                // Checks if the frame is fully processed by the GPU, by getting the first timestamp of the next frame
                const VkResult result = vkGetQueryPoolResults(device, m_frameStartQueryPool, nextIdx, 1,
                    sizeof(u64), m_frameStartTimes + nextIdx, sizeof(u64), VK_QUERY_RESULT_64_BIT);

                // Query result is ready
                if (result == VK_SUCCESS)
                {
                    const u32 lastMarker = GetLastMarker(frame);
                    AVA_ASSERT(IsMarkerValid(firstMarker) && IsMarkerValid(lastMarker));

                    if (lastMarker < firstMarker)
                    {
                        // the pool is a ring buffer so we'll loop over if we reach the max pool size, so 2 steps reset is required
                        _CollectAndResetQueriesRange(cmd, firstMarker, m_totalMarkerCount - 1);
                        _CollectAndResetQueriesRange(cmd, 0, lastMarker);
                    }
                    else
                    {
                        // easy case, all markers are continuous
                        _CollectAndResetQueriesRange(cmd, firstMarker, lastMarker);
                    }
                }
                else
                {
                    // No need to continue, next frames won't be available either
                    break;
                }
            }

            // Records the frame where we stopped for next time
            m_nextFrameToQuery = frame;
        }
    }

    void VkGpuThreadInfo::EndFrame()
    {
        // Invalidates reference to current graphics context
        m_ctx = nullptr;
    }

    void VkGpuThreadInfo::Close() const
    {
        GraphicsContext::WaitIdle();
        const VkDevice device = VkGraphicsContext::GetDevice();

        vkDestroyQueryPool(device, m_frameStartQueryPool, nullptr);
        vkDestroyQueryPool(device, m_markerStartQueryPool, nullptr);
        vkDestroyQueryPool(device, m_markerEndQueryPool, nullptr);

        delete[] m_cpuFrameStartTimes;
        delete[] m_frameStartTimes;
        delete[] m_markerStartTimes;
        delete[] m_markerEndTimes;

        GpuThreadInfo::Close();
    }

    u32 VkGpuThreadInfo::PushMarker(const char* _name)
    {
        AVA_ASSERT(m_ctx, "[Profiler] can't push marker in graphics queue outside frame rendering.");

        if (!m_ctx || !m_gpuMarkersReady)
        {
            return 0;
        }

        const u32 index = GpuThreadInfo::PushMarker(_name);
        if (IsMarkerValid(index))
        {
            m_ctx->BeginDebugMarkerRegion(_name, Color::Grey);

            const VkCommandBuffer cmd = m_ctx->GetImpl()->GetCommandBuffer();
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, m_markerStartQueryPool, index);
        }

        return index;
    }

    u32 VkGpuThreadInfo::PopMarker(const char* _name)
    {
        if (!m_ctx || !m_gpuMarkersReady)
        {
            return 0;
        }

        const u32 index = GpuThreadInfo::PopMarker(_name);
        if (IsMarkerValid(index))
        {
            const VkCommandBuffer cmd = m_ctx->GetImpl()->GetCommandBuffer();
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, m_markerEndQueryPool, index);

            m_ctx->EndDebugMarkerRegion();
        }

        return index;
    }

    u64 VkGpuThreadInfo::GetMarkerStartTimeNS(const u32 _frameId, const u32 _marker) const
    {
        const u64 markerStart = m_markerStartTimes[_marker] + m_cpuGpuDeltaTime;
        const u64 frameStart = m_cpuFrameStartTimes[_GetFrameStartIndex(_frameId)];
        return markerStart - frameStart;
    }

    u64 VkGpuThreadInfo::GetMarkerDurationNS(const u32 _marker) const
    {
        return m_markerEndTimes[_marker] - m_markerStartTimes[_marker];
    }

    u64 VkGpuThreadInfo::GetFrameStartTimeNS(const u32 _frameId) const
    {
        return m_cpuFrameStartTimes[_GetFrameStartIndex(_frameId)];
    }

    u64 VkGpuThreadInfo::GetFrameDurationNS(const u32 _frameId) const
    {
        const u64 nextFrameStart = m_cpuFrameStartTimes[_GetFrameStartIndex(_frameId + 1)];
        const u64 frameStart = m_cpuFrameStartTimes[_GetFrameStartIndex(_frameId)];
        return nextFrameStart - frameStart;
    }

    bool VkGpuThreadInfo::IsFrameAvailable(const u32 _frameId) const
    {
        return _frameId < m_nextFrameToQuery;
    }

    void VkGpuThreadInfo::_UpdateCpuGpuDeltaTime(const GraphicsContext* _ctx, const bool _forceUpdate/*= false*/)
    {
        // This operation can take up to 0.5 ms, so we only perform it once every 600 frames.
        if(_forceUpdate || _ctx->GetFrameId() % 600 == 0)
        {
            const u64 gpuTime = _ctx->GetGpuTimestamp();
            const u64 cpuTime = TimeMgr::GetTimeNS();
            m_cpuGpuDeltaTime = (s64)cpuTime - (s64)gpuTime;
        }
    }

    void VkGpuThreadInfo::_CollectAndResetQueriesRange(const VkCommandBuffer _cmd, const u32 _firstMarker, const u32 _lastMarker) const
    {
        const auto validRange = 
            _cmd != VK_NULL_HANDLE
            && IsMarkerValid(_firstMarker)
            && IsMarkerValid(_lastMarker)
            && _firstMarker <= _lastMarker;

        if (!AVA_VERIFY(validRange))
        {
            return;
        }

        const VkDevice device = VkGraphicsContext::GetDevice();
        const auto markerCount = _lastMarker - _firstMarker + 1;

        // Collects start marker query results
        vkGetQueryPoolResults(device, m_markerStartQueryPool, _firstMarker, markerCount, 
            sizeof(u64) * markerCount, m_markerStartTimes + _firstMarker, sizeof(u64), VK_QUERY_RESULT_64_BIT);

        // Collects end marker query results
        vkGetQueryPoolResults(device, m_markerEndQueryPool, _firstMarker, markerCount, 
            sizeof(u64) * markerCount, m_markerEndTimes + _firstMarker, sizeof(u64), VK_QUERY_RESULT_64_BIT);

        // Reset query pool command can only be submitted on a graphics queue
        vkCmdResetQueryPool(_cmd, m_markerStartQueryPool, _firstMarker, markerCount);
        vkCmdResetQueryPool(_cmd, m_markerEndQueryPool, _firstMarker, markerCount);
    }
}
