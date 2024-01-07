#pragma once
/// @file ProfilerViewer.h
/// @brief

#include <Core/Base.h>
#include <UI/ImGuiTools.h>

namespace Ava {

    class ProfilerViewer
    {
    public:
        ProfilerViewer();
        ~ProfilerViewer();

        void SetPauseKey(ImGuiKey _key, const char* _keyName);
        void SetFullscreenKey(ImGuiKey _key, const char* _keyName);

        void SetPaused(bool _paused);
        bool IsPaused() const;

        void Display(bool* _windowOpened);

    private:
        /// Any marker smaller than this will not be displayed
        float m_minMarkerSize = 2.f;
        /// Any marker smaller than this will be grouped in a LOD marker
        float m_lodMarkerThreshold = 6.f;
        /// Time at the profiler left border
        float m_timeOffset = 0.f;
        /// Number of ImGui pixels per time unit
        float m_timeScale = 0.f;
        /// Width of the profiler in time unit
        float m_timeRange = 30.f;

        int m_cpuDisplayDepth;
        int m_gpuDisplayDepth;

        bool m_paused = false;
        ImGuiKey m_pauseKey = ImGuiKey_None;
        const char* m_pauseKeyName = "";

        bool m_fullscreen = false;
        ImGuiKey m_fullscreenKey = ImGuiKey_None;
        const char* m_fullscreenKeyName = "";

        // UI helpers
        ImVec2 m_savedWindowPos;
        ImVec2 m_savedWindowSize;
        u32 m_frameIdxNavigation = 0;
        char m_highlightFilter[256]{};

        // Logic helpers
        u32 m_guiOldestFrameIdx = 0;
        u32 m_nbFrameDisplayed = 8;
        u64 m_frameCount = 0;

        struct MarkerData;
        struct ThreadData;
        struct FrameData;

        std::vector<ThreadData> m_threads;
        std::vector<FrameData> m_frameRingBuffer;

        struct MarkerStats;
        std::vector<MarkerStats> m_gpuMarkerStats;
        std::vector<MarkerStats> m_cpuMarkerStats;

        // logic helpers
        void _UpdateGuiData();
        FrameData& _GetFrame(u32 _threadIdx, u32 _frameIndex);
        void _AddMarkerStats(u32 _threadIdx, const char *_markerName, u32 _nbFrames);
        void _UpdateMarkerStats(u32 _frameIdx);

        template <class ThreadInfoType>
        void _BuildFrameData(ThreadInfoType* _thread, u32 _frameIdx, FrameData& _frame, std::vector<MarkerStats>& _stats, bool _isGpuThread);

        // UI helpers
        void   _DisplayTimeline(const ImVec2& _pos, const ImVec2& _size);
        float  _GetTimeOffsetToNavigationFrame(u32 _offset, bool _positive);
        void   _DisplayMarkers(const ImVec4& _markerArea, const ImVec2& _frameStartPos, const FrameData& _frame, u32 _frameIdx, u32 _threadIdx, u32 _depth, u32 _maxDepth, u32& _markerIdx, ImGuiTextFilter& _filter);
        void   _ShowLodMarkerToolTip(u32 _lodMarkerCount, float _lodDuration, const FrameData& _frame, u32 _lastMarkerIdx, int _depth) const;
        float  _GetFrameDurationForMarker(const MarkerData& _marker, u32 _frameIdx, u32 _depth);
        ImVec2 _GetMarkerPos(const MarkerData& _marker, const ImVec2& _frameStartPos) const;
        void   _DisplayFrameDimensions(const ImVec2& _pos, const ImVec2& _size);
        void   _DisplayMarkerAreaHovered(const ImVec2& _delta, const ImColor& _leftColor, const ImColor& _rightColor) const;
        void   _DisplayMenuBar();
    };

}
