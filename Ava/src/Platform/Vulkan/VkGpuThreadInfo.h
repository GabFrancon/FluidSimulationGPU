#pragma once
/// @file VkGpuThreadInfo.h
/// @brief file implementing GpuThreadInfoBase for Vulkan.

#include <Time/ProfilerThreadInfo.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <vulkan/vulkan.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    class VkGpuThreadInfo final : public GpuThreadInfo
    {
    public:
        VkGpuThreadInfo() { SetName("GPU"); }

        void Init(u32 _frameCount, u32 _markerCount) override;
        void StartFrame(u32 _frameId, GraphicsContext* _ctx) override;
        void EndFrame() override;
        void Close() const override;

        u32 PushMarker(const char* _name) override;
        u32 PopMarker(const char* _name) override;

        u64 GetMarkerStartTimeNS(u32 _frameId, u32 _marker) const override;
        u64 GetMarkerDurationNS(u32 _marker) const override;

        u64 GetFrameStartTimeNS(u32 _frameId) const override;
        u64 GetFrameDurationNS(u32 _frameId) const override;
        bool IsFrameAvailable(u32 _frameId) const override;

    private:
        void _UpdateCpuGpuDeltaTime(const GraphicsContext* _ctx, bool _forceUpdate = false);
        void _CollectAndResetQueriesRange(VkCommandBuffer _cmd, u32 _firstMarker, u32 _lastMarker) const;

        u64* m_cpuFrameStartTimes = nullptr;
        u64* m_frameStartTimes = nullptr;
        u64* m_markerStartTimes = nullptr;
        u64* m_markerEndTimes = nullptr;

        u32 m_nextFrameToQuery = 0;
        VkQueryPool m_frameStartQueryPool = VK_NULL_HANDLE;
        VkQueryPool m_markerStartQueryPool = VK_NULL_HANDLE;
        VkQueryPool m_markerEndQueryPool = VK_NULL_HANDLE;

        s64 m_cpuGpuDeltaTime = 0;
        bool m_gpuMarkersReady = false;
        GraphicsContext* m_ctx = nullptr;
    };

}