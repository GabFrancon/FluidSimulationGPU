#include <avapch.h>
#include "ProfilerThreadInfo.h"

#include <Time/TimeManager.h>
#include <Debug/Assert.h>
#include <Math/Hash.h>

#if defined(AVA_GRAPHIC_API_VULKAN)
    #include <Platform/Vulkan/VkGpuThreadInfo.h>
#else
    #error Unknown graphics API
#endif

namespace Ava {

    //----- Profiler Thread Info base class ----------------------------------------------

    void ProfilerThreadInfo::SetName(const char* _name)
    {
        strncpy(m_name, _name, std::size(m_name));
        m_name[std::size(m_name) - 1] = 0;
    }

    const char* ProfilerThreadInfo::GetName() const
    {
        return m_name;
    }

    void ProfilerThreadInfo::Init(const u32 _frameCount, const u32 _markerCount)
    {
        m_numFramesToRecord = _frameCount;
        m_totalMarkerCount = _frameCount * _markerCount;
        m_firstMarkerIndex = m_totalMarkerCount - 1;

        m_frameFirstMarkers = new u32[m_numFramesToRecord];
        m_markerNames = new const char*[m_totalMarkerCount];
        m_markerHashes = new u32[m_totalMarkerCount];
        m_markerDepths = new u8[m_totalMarkerCount];

        for (size_t i = 0; i < _frameCount; i++)
        {
            m_frameFirstMarkers[i] = m_firstMarkerIndex;
        }
        memset(m_markerNames, 0, m_totalMarkerCount * sizeof(const char*));
        memset(m_markerHashes, 0, m_totalMarkerCount * sizeof(u32));
        memset(m_markerDepths, 0, m_totalMarkerCount * sizeof(u8));

        for (size_t i = 0; i < kMarkerMaxDepth; i++)
        {
            m_pushedMarkersStack[i].marker = kMarkerIdxInvalid;
            m_pushedMarkersStack[i].frameId = 0;
        }
    }

    void ProfilerThreadInfo::StartFrame(const u32 _frameId)
    {
        m_currentFrame = _frameId;
        m_frameFirstMarkers[_GetFrameStartIndex(_frameId)] = m_nextMarkerIndex;

        // m_firstMarkerIndex is the first marker of the oldest frame we have.
        const u32 oldestFrame = _GetOldestFrame();
        m_firstMarkerIndex = GetFirstMarker(oldestFrame);
    }

    void ProfilerThreadInfo::EndFrame()
    {
    }

    void ProfilerThreadInfo::Close() const
    {
        delete[] m_markerNames;
        delete[] m_markerHashes;
        delete[] m_markerDepths;
        delete[] m_frameFirstMarkers;
    }

    u32 ProfilerThreadInfo::GetFirstMarker(const u32 _frameId) const
    {
        const u32 firstMarker = m_frameFirstMarkers[_GetFrameStartIndex(_frameId)];
        const u32 nextFirstMarker = m_frameFirstMarkers[_GetFrameStartIndex(_frameId + 1)];

        // If the first marker of the next frame is the same, there is no markers for this frame
        return firstMarker != nextFirstMarker ? firstMarker : kMarkerIdxInvalid;
    }

    u32 ProfilerThreadInfo::GetNextMarker(const u32 _frameId, const u32 _marker) const
    {
        const u32 nextMarker = (_marker + 1) % m_totalMarkerCount;
        const u32 nextFirstMarker = m_frameFirstMarkers[_GetFrameStartIndex(_frameId + 1)];

        // If the next marker is the first of next frame, there is no next marker
        return nextMarker != nextFirstMarker ? nextMarker : kMarkerIdxInvalid;
    }

    u32 ProfilerThreadInfo::GetLastMarker(const u32 _frameId) const
    {
        u32 marker = GetFirstMarker(_frameId);
        u32 lastMarker = marker;

        while (IsMarkerValid(marker))
        {
            lastMarker = marker;
            marker = GetNextMarker(_frameId, marker);
        }
        return lastMarker;
    }

    u32 ProfilerThreadInfo::PushMarker(const char* _name)
    {
        const u32 stackIdx = m_nextMarkerIndex;

        if (AVA_VERIFY(stackIdx != m_firstMarkerIndex, "[Profiler] cannot push marker '%s', buffer full.", _name)
            && AVA_VERIFY(m_pushedMarkersCount < kMarkerMaxDepth, "[Profiler] cannot push marker '%s', max depth reached.", _name))
        {
            m_markerDepths[stackIdx] = m_pushedMarkersCount;
            m_markerNames[stackIdx] = _name;
            m_markerHashes[stackIdx] = HashStr(_name);

            m_pushedMarkersStack[m_pushedMarkersCount].marker = stackIdx;
            m_pushedMarkersStack[m_pushedMarkersCount].frameId = m_currentFrame;
            m_pushedMarkersCount++;

            m_nextMarkerIndex = (stackIdx + 1) % m_totalMarkerCount;
            return stackIdx;
        }
        return kMarkerIdxInvalid;
    }

    u32 ProfilerThreadInfo::PopMarker(const char* _name)
    {
        const u32 nameHash = HashStr(_name);
        int stackIdx = (int)m_pushedMarkersCount - 1;
        bool forgotToPop = false;
        u32 unpoppedIdx = 0;

        while (
            AVA_VERIFY(stackIdx >= 0,
            "[Profiler] cannot pop marker '%s'. This marker was never pushed or its parent was already popped.", _name))
        {
            const u32 index = m_pushedMarkersStack[stackIdx].marker;
            const u32 frame = m_pushedMarkersStack[stackIdx].frameId;
            stackIdx--;

            const u32 pushedNameHash = GetMarkerHash(index);

            // The name is the same, all good.
            if (pushedNameHash == nameHash)
            {
                AVA_ASSERT(!forgotToPop, "[Profiler] forgot to pop marker '%s' before popping '%s'.", m_markerNames[unpoppedIdx], _name);
                AVA_ASSERT(m_markerDepths[index] == stackIdx + 1);
                const u32 newStackSize = stackIdx + 1;

                // Reset pushed marker stack
                for (size_t i = newStackSize; i < m_pushedMarkersCount; i++)
                {
                    m_pushedMarkersStack[i].marker = kMarkerIdxInvalid;
                    m_pushedMarkersStack[i].frameId = 0;
                }
                m_pushedMarkersCount = newStackSize;
                return index;
            }
            // The name is different -> either this marker was overwritten
            // because pushed too long ago or we forgot to pop a child marker.
            const u32 oldestFrame = _GetOldestFrame();

            if (!AVA_VERIFY(frame >= oldestFrame, 
                "[Profiler] cannot pop marker '%s'. It was pushed %d frames ago.",
                _name, m_currentFrame - frame))
            {
                m_pushedMarkersCount--;
                break;
            }

            // Continue searching, the marker may be above in the stack.
            forgotToPop = true;
            unpoppedIdx = index;
        }

        return kMarkerIdxInvalid;
    }

    bool ProfilerThreadInfo::IsMarkerValid(const u32 _marker)
    {
        return _marker != kMarkerIdxInvalid;
    }

    bool ProfilerThreadInfo::IsMarkerOpen(const u32 _marker) const
    {
        return GetMarkerDurationNS(_marker) == kMarkerOpenDuration;
    }

    u32 ProfilerThreadInfo::_GetOldestFrame() const
    {
        return m_currentFrame >= m_numFramesToRecord ? m_currentFrame - m_numFramesToRecord + 1 : 0;
    }

    u32 ProfilerThreadInfo::_GetFrameStartIndex(const u32 _frameId) const
    {
        return _frameId % m_numFramesToRecord;
    }


    //----- CPU Thread Info implementation -----------------------------------------------

    void CpuThreadInfo::Init(const u32 _frameCount, const u32 _markerCount)
    {
        ProfilerThreadInfo::Init(_frameCount, _markerCount);

        m_frameStartTimes = new u64[m_numFramesToRecord];
        m_markerStartTimes = new u64[m_totalMarkerCount];
        m_markerEndTimes = new u64[m_totalMarkerCount];
        m_markerPastIndices = new u32[m_totalMarkerCount];
    }

    void CpuThreadInfo::StartFrame(const u32 _frameId)
    {
        ProfilerThreadInfo::StartFrame(_frameId);
        m_frameStartTimes[_GetFrameStartIndex(m_currentFrame)] = TimeMgr::GetTimeNS();

        // let the first frames run run without marker to let thread info being properly reset.
        m_cpuMarkerReady = m_currentFrame > 1;
    }

    void CpuThreadInfo::Close() const
    {
        delete[] m_frameStartTimes;
        delete[] m_markerStartTimes;
        delete[] m_markerEndTimes;
        delete[] m_markerPastIndices;

        ProfilerThreadInfo::Close();
    }

    u32 CpuThreadInfo::PushMarker(const char* _name)
    {
        if (!m_cpuMarkerReady)
        {
            return 0;
        }

        const u32 index = ProfilerThreadInfo::PushMarker(_name);
        if (IsMarkerValid(index))
        {
            m_markerStartTimes[index] = TimeMgr::GetTimeNS();
            m_markerEndTimes[index] = kMarkerOpenDuration;
            m_markerPastIndices[index] = kMarkerIdxInvalid;
        }
        return index;
    }

    u32 CpuThreadInfo::PopMarker(const char* _name)
    {
        if (!m_cpuMarkerReady)
        {
            return 0;
        }

        const u32 index = ProfilerThreadInfo::PopMarker(_name);
        if (IsMarkerValid(index))
        {
            m_markerEndTimes[index] = TimeMgr::GetTimeNS();
        }
        return index;
    }

    u64 CpuThreadInfo::GetMarkerStartTimeNS(const u32 _frameId, const u32 _marker) const
    {
        return m_markerStartTimes[_marker] - m_frameStartTimes[_GetFrameStartIndex(_frameId)];
    }

    u64 CpuThreadInfo::GetMarkerDurationNS(const u32 _marker) const
    {
        if (m_markerEndTimes[_marker] == kMarkerOpenDuration)
        {
            return kMarkerOpenDuration;
        }
        return m_markerEndTimes[_marker] - m_markerStartTimes[_marker];
    }

    u64 CpuThreadInfo::GetFrameStartTimeNS(const u32 _frameId) const
    {
        return m_frameStartTimes[_GetFrameStartIndex(_frameId)];
    }

    u64 CpuThreadInfo::GetFrameDurationNS(const u32 _frameId) const
    {
        return m_frameStartTimes[_GetFrameStartIndex(_frameId + 1)] - m_frameStartTimes[_GetFrameStartIndex(_frameId)];
    }

    bool CpuThreadInfo::IsFrameAvailable(u32 _frameId) const
    {
        // results are always available on the CPU.
        return true;
    }

    CpuThreadInfo* CpuThreadInfo::Create()
    {
    #if defined(AVA_PLATFORM_WINDOWS)
        return new CpuThreadInfo();
    #else
        return nullptr;
    #endif
    }


    //----- GPU Thread Info implementation -----------------------------------------------

    GpuThreadInfo* GpuThreadInfo::Create()
    {
    #if defined(AVA_GRAPHIC_API_VULKAN)
        return new VkGpuThreadInfo();
    #else
        return nullptr;
    #endif
    }


}
