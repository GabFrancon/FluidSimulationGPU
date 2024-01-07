#pragma once
/// @file ProfilerThreadInfo.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    static constexpr u8  kMarkerMaxDepth = 4;
    static constexpr u32 kMarkerIdxInvalid = 0xFFFFFFFF;
    static constexpr u64 kMarkerOpenDuration = 0xFFFFFFFFFFFFFFFF;

    /// @brief Base class for profiler thread info structures.
    class ProfilerThreadInfo
    {
    public:
        ProfilerThreadInfo() = default;
        virtual ~ProfilerThreadInfo() {}

        void SetName(const char* _name);
        const char* GetName() const;

        virtual void Init(u32 _frameCount, u32 _markerCount);
        virtual void StartFrame(u32 _frameId);
        virtual void EndFrame();
        virtual void Close() const;

        /// @brief Returns the index of the first marker for this frame, or kMarkerIdxInvalid if no marker.
        u32 GetFirstMarker(u32 _frameId) const;
        /// @brief Returns the index of the next marker for this frame, or kMarkerIdxInvalid if marker was the last.
        u32 GetNextMarker(u32 _frameId, u32 _marker) const;
        /// @brief Returns the index of the last marker for this frame, or kMarkerIdxInvalid if no marker.
        u32 GetLastMarker(u32 _frameId) const;

        /// @brief Pushes a new marker on the stack. Push calls can be nested.
        virtual u32 PushMarker(const char* _name);
        /// @brief Pops the given marker from the stack.
        virtual u32 PopMarker(const char* _name);

        /// @brief Returns true if this index can be used to query info about a marker.
        static bool IsMarkerValid(u32 _marker);
        /// @brief Returns true if the marker has been pushed but not popped yet, and false otherwise.
        bool IsMarkerOpen(u32 _marker) const;

        /// @brief Returns the depth of the marker. 0 is the top level; 1 is a child of 0; etc.
        u8 GetMarkerDepth(const u32 _marker) const { return m_markerDepths[_marker]; }
        /// @brief Returns the hash of the name of the marker.
        u32 GetMarkerHash(const u32 _marker) const { return m_markerHashes[_marker]; }
        /// @brief Returns the name of the marker.
        const char* GetMarkerName(const u32 _marker) const { return m_markerNames[_marker]; }

        /// @brief Returns the elapsed time between the frame start and the beginning of the marker.
        virtual u64 GetMarkerStartTimeNS(u32 _frameId, u32 _marker) const { return 0u; }
        /// @brief Returns the duration of the marker, in nanoseconds
        virtual u64 GetMarkerDurationNS(u32 _marker) const { return 0u; }

        /// @brief Returns the time at which StartFrame(_frameId) was called.
        virtual u64 GetFrameStartTimeNS(u32 _frameId) const {return 0u; }
        /// @brief Returns the duration of the frame, in nanoseconds
        virtual u64 GetFrameDurationNS(u32 _frameId) const { return 0u; }
        /// @brief Returns true if the data of this frame are available.
        virtual bool IsFrameAvailable(u32 _frameId) const { return true; }

        u32 GetNumFramesToRecord() const { return m_numFramesToRecord; }
        u32 GetTotalMarkerCount() const { return m_totalMarkerCount; }

    protected:
        u32 _GetOldestFrame() const;
        u32 _GetFrameStartIndex(u32 _frameId) const;

        char m_name[32]{0};

        u32 m_totalMarkerCount = 0;
        u32 m_firstMarkerIndex = 0;
        u32 m_nextMarkerIndex = 0;

        u32 m_numFramesToRecord = 0;
        u32 m_currentFrame = 0;

        u8* m_markerDepths = nullptr;
        u32* m_markerHashes = nullptr;
        u32* m_frameFirstMarkers = nullptr;
        const char** m_markerNames = nullptr;

        struct PushedMarker {
            u32 frameId;
            u32 marker;
        };
        PushedMarker m_pushedMarkersStack[kMarkerMaxDepth]{};
        u32 m_pushedMarkersCount = 0;
    };

    /// @brief CPU-thread specific profiler info.
    class CpuThreadInfo : public ProfilerThreadInfo
    {
    public:
        CpuThreadInfo() { SetName("CPU"); }

        void Init(u32 _frameCount, u32 _markerCount) override;
        void StartFrame(u32 _frameId) override;
        void Close() const override;

        u32 PushMarker(const char* _name) override;
        u32 PopMarker(const char* _name) override;

        u64 GetMarkerStartTimeNS(u32 _frameId, u32 _marker) const override;
        u64 GetMarkerDurationNS(u32 _marker) const override;

        u64 GetFrameStartTimeNS(u32 _frameId) const override;
        u64 GetFrameDurationNS(u32 _frameId) const override;
        bool IsFrameAvailable(u32 _frameId) const override;

        static CpuThreadInfo* Create();

    private:
        u64* m_frameStartTimes = nullptr;
        u64* m_markerStartTimes = nullptr;
        u64* m_markerEndTimes = nullptr;
        u32* m_markerPastIndices = nullptr;
        bool m_cpuMarkerReady = false;
    };

    /// @brief GPU-thread specific profiler info.
    class GpuThreadInfo : public ProfilerThreadInfo
    {
    public:
        GpuThreadInfo() { SetName("GPU"); }

        void StartFrame(const u32 _frameId) override { return ProfilerThreadInfo::StartFrame(_frameId); }
        virtual void StartFrame(const u32 _frameId, GraphicsContext* _ctx) { }

        static GpuThreadInfo* Create();
    };

}
