#include <avapch.h>
#include "ProfilerViewer.h"

#include <Time/Profiler.h>
#include <Time/ProfilerThreadInfo.h>
#include <Strings/StringBuilder.h>
#include <Strings/StringHash.h>
#include <Debug/Im3D.h>
#include <Math/Math.h>
#include <Math/Hash.h>

namespace Ava {

    constexpr float kMarkerSpacing = 1.f;
    constexpr float kMarkerHeight = 40.f;
    constexpr float kFrameDimensionsHeight = 40.f;
    constexpr bool kDisplayColumnBorders = false;
    constexpr float kThreadNameColumnSize = 350.f;
    constexpr char const* kFrameMarkerStatsName = "Frame";
    constexpr ImVec2 kMarkerStatsWindowSize = ImVec2(600.f, 200.f);
    constexpr float kMarkerStatsColumnSize = 80.f;

    // Only one thread for CPU for now
    constexpr int kGpuThreadId = 0;
    constexpr int kCpuThreadId = 1;

    ImVec2 operator+(const ImVec2& _a, const ImVec2& _b)
    {
        ImVec2 res;
        res.x = _a.x + _b.x;
        res.y = _a.y + _b.y;
        return res;
    }

    ImVec2 operator-(const ImVec2& _a, const ImVec2& _b)
    {
        ImVec2 res;
        res.x = _a.x - _b.x;
        res.y = _a.y - _b.y;
        return res;
    }

    ImVec2 operator*(const ImVec2& _vec, const float _scalar)
    {
        ImVec2 res;
        res.x = _vec.x * _scalar;
        res.y = _vec.y * _scalar;
        return res;
    }

    ImVec2 operator/(const ImVec2& _vec, const float _scalar)
    {
        ImVec2 res;
        res.x = _vec.x / _scalar;
        res.y = _vec.y / _scalar;
        return res;
    }

    struct ProfilerViewer::MarkerData
    {
        const char* name;
        u32 markerIdx;
        float start;
        float duration;
        u16 firstChildIdx;
        u16 lastChildIdx;
        u16 parentIdx;
        u32 colorIdx;
    };

    struct ProfilerViewer::ThreadData
    {
        bool open;
        u32  maxMarkersPerFrame;
        char name[32];

        ThreadData()
        {
            open = false;
            name[0] = 0;
            maxMarkersPerFrame = 0;
        }

        void SetName(const char* _name)
        {
            strcpy(name, _name);
        }
    };

    struct ProfilerViewer::FrameData
   {
        float duration;
        u32 markerCount;
        u64 frameCount;
        std::vector<MarkerData> markers[kMarkerMaxDepth];

        void Clear()
        {
            for (u32 i = 0; i < kMarkerMaxDepth; ++i)
            {
                markers[i].clear();
            }
            duration = 0.f;
            frameCount = 0;
            markerCount = 0;
        }
   };

    struct ProfilerViewer::MarkerStats
    {
        enum StatsMode
        {
            AveragePerMarker,
            AveragePerFrame,
            Continuous,
        };

        u32           m_hash;
        std::string   m_name;
        bool          m_open;
        u32           m_numFramesPerGroup;
        u32           m_lastUpdatedFrame;
        StatsMode     m_mode;

        static constexpr u32 kArraySize = 100;

        // These arrays don't contain 1 value per frame but 1 value per "group" of frames.
        // By storing only one value per group and having one group corresponding to many frames,
        // we can display an approximate graph representing many many frames (eg. 10000) while keeping
        // the storage small (and constant).
        float         m_minDuration[kArraySize];
        float         m_maxDuration[kArraySize];
        float         m_avgDuration[kArraySize];
        float         m_nbMarkers  [kArraySize];

        float         m_totalDurationCurrentFrame;
        float         m_plotMin;

        void Init(const u32 _hash, const char* _name, const u32 _nbFrames)
        {
            m_hash = _hash;
            m_name = _name;
            m_open = true;
            m_numFramesPerGroup = _nbFrames / kArraySize;
            m_lastUpdatedFrame = 0;
            m_mode = AveragePerMarker;

            for (auto& minVal : m_minDuration) minVal = FLT_MAX;
            memset(m_maxDuration, 0, sizeof(m_maxDuration));
            memset(m_avgDuration, 0, sizeof(m_avgDuration));
            memset(m_nbMarkers,   0, sizeof(m_nbMarkers));

            m_totalDurationCurrentFrame = 0;

            m_plotMin = 0.0f;
        }

        void ClearFrameGroup(const u32 _groupIdx)
        {
            m_maxDuration  [_groupIdx] = 0;
            m_minDuration  [_groupIdx] = FLT_MAX;
            m_avgDuration  [_groupIdx] = 0;
            m_nbMarkers    [_groupIdx] = 0;
        }

        void StartFrame(const u32 _frameIdx)
        {
            // In continuous mode we ignore frame boundaries, nothing to do here.
            if (m_mode == Continuous)
                return;

            // If the profiler was paused, we may have missed lots of frames 
            // and the corresponding groups need to be cleared.
            if (_frameIdx > m_lastUpdatedFrame + 1)
            {
                // At least 1 frame was missed
                const int firstMissedFrame = m_lastUpdatedFrame + 1;
                const int lastMissedFrameIdx = _frameIdx - 1;

                const int firstGroup = (firstMissedFrame / m_numFramesPerGroup) % kArraySize;
                const int lastGroup  = (lastMissedFrameIdx / m_numFramesPerGroup) % kArraySize;
                
                for (int i = firstGroup; i <= lastGroup; ++i)
                {
                    ClearFrameGroup(i);
                }
            }

            // Every N frame, we start filling a new group.
            if (_frameIdx % m_numFramesPerGroup == 0)
            {
                // Clear the new group
                const int groupIdx = (_frameIdx / m_numFramesPerGroup) % kArraySize;
                ClearFrameGroup(groupIdx);

                // If we start a new group, it also means the previous group is complete.
                if (m_mode == AveragePerMarker)
                {
                    // Calculate the average per marker for the previous group.
                    const int prevGroupIdx = groupIdx == 0 ? kArraySize - 1 : groupIdx - 1;
                    if (m_nbMarkers[prevGroupIdx] > 0)
                    {
                        m_avgDuration[prevGroupIdx] = m_avgDuration[prevGroupIdx] / m_nbMarkers[prevGroupIdx];
                    }
                }
            }

            if (m_mode == AveragePerFrame)
            {
                // In this mode, the min/max values are per frame.
                // When starting frame N, we have all the values to update frame N-1.
                const u32 groupIdx = ((_frameIdx - 1) / m_numFramesPerGroup) % kArraySize;

                m_minDuration[groupIdx] = std::min(m_minDuration[groupIdx], m_totalDurationCurrentFrame);
                m_maxDuration[groupIdx] = std::max(m_maxDuration[groupIdx], m_totalDurationCurrentFrame);
            }

            m_totalDurationCurrentFrame = 0;

            m_lastUpdatedFrame = _frameIdx;
        }

        void AddStats(const MarkerData& _marker, const u32 _frameIdx)
        {
            if (m_mode != Continuous)
            {
                const u32 groupIdx = (_frameIdx / m_numFramesPerGroup) % kArraySize;

                m_nbMarkers[groupIdx] += 1.f / (float)m_numFramesPerGroup;
                m_avgDuration[groupIdx] += _marker.duration / (float)m_numFramesPerGroup;

                m_totalDurationCurrentFrame += _marker.duration;

                if (m_mode == AveragePerMarker)
                {
                    // In this mode, the min/max values are per marker
                    m_minDuration[groupIdx] = std::min(m_minDuration[groupIdx], _marker.duration);
                    m_maxDuration[groupIdx] = std::max(m_maxDuration[groupIdx], _marker.duration);
                }
            }
            else
            {
                // Continuous mode
                // In this case m_lastUpdatedFrame actually represents the current group.
                const u32 i = m_lastUpdatedFrame;

                m_nbMarkers    [i] += 1.f;
                m_minDuration  [i] = std::min(m_minDuration[i], _marker.duration);
                m_maxDuration  [i] = std::max(m_maxDuration[i], _marker.duration);
                m_avgDuration  [i] += _marker.duration;

                // Every N values, start using a new group.
                if (m_nbMarkers[i] >= (float)m_numFramesPerGroup)
                {
                    // Clear the new group
                    // In this mode m_lastUpdatedFrame actually represents the current group.
                    m_lastUpdatedFrame = (m_lastUpdatedFrame + 1) % kArraySize;
                    ClearFrameGroup(m_lastUpdatedFrame);

                    // If we start a new group, it also means the previous group is complete.
                    // Calculate the average for the previous group.
                    const int prevGroupIdx = m_lastUpdatedFrame == 0 ? kArraySize - 1 : m_lastUpdatedFrame - 1;
                    if (m_nbMarkers[prevGroupIdx] > 0)
                    {
                        m_avgDuration[prevGroupIdx] = m_avgDuration[prevGroupIdx] / m_nbMarkers[prevGroupIdx];
                    }
                }
            }
        }

        void Display(const char* _markerType, const u32 _frameIdx)
        {
            if (m_open)
            {
                int currentGroupIdx = (_frameIdx / m_numFramesPerGroup) % kArraySize;

                // In this case m_lastUpdatedFrame actually represents the current group.
                if (m_mode == Continuous)
                    currentGroupIdx = m_lastUpdatedFrame;

                float minVal = FLT_MAX;
                float maxVal = -FLT_MAX;
                float minAvg = 0.0f;
                float maxAvg = 0.0f;
                float avgAvg = 0.0f;
                float nbMarker = 0.0f;
                float validValueCount = 0;

                for (int i = 0; i < kArraySize; ++i)
                {
                    // Skip the current group because it's still "in construction" if m_numFramesPerGroup > 1
                    if (i == currentGroupIdx) continue;

                    // Skip groups that don't contain values yet
                    if (m_minDuration[i] == FLT_MAX) continue;

                    validValueCount += 1.0f;

                    maxVal = std::max(maxVal, m_maxDuration[i]);
                    minVal = std::min(minVal, m_minDuration[i]);
                    nbMarker = std::max(nbMarker, m_nbMarkers[i]);

                    minAvg += m_minDuration[i];
                    maxAvg += m_maxDuration[i];
                    avgAvg += m_avgDuration[i];
                }

                maxAvg /= validValueCount;
                minAvg /= validValueCount;
                avgAvg /= validValueCount;

                float maxVariance = 0;
                float minVariance = 0;
                for (int i = 0; i < kArraySize; ++i)
                {
                    // Skip the current group because it's still "in construction" if m_numFramesPerGroup > 1
                    if (i == currentGroupIdx) continue;

                    // Skip groups that don't contain values yet
                    if (m_minDuration[i] == FLT_MAX) continue;

                    maxVariance += (m_maxDuration[i] - maxAvg) * (m_maxDuration[i] - maxAvg);
                    minVariance += (m_minDuration[i] - minAvg) * (m_minDuration[i] - minAvg);
                }
                maxVariance /= validValueCount;
                const float maxStandardDeviation = sqrt(maxVariance);

                // Graphs make more sense if the min is 0
                float plotMin = 0.f;
                float plotMax = maxAvg + maxStandardDeviation;

                // However it can be useful to zoom to see more details
                if (m_plotMin > 0.f)
                {
                    plotMin = m_plotMin;
                    if (plotMin > plotMax)
                    {
                        plotMax = plotMin + plotMin * 0.01f;
                    }
                }

                nbMarker = (nbMarker > 0.9f && nbMarker < 1.1f) ? 1.0f : nbMarker;
                if (minVal == FLT_MAX)
                    minVal = 0.0f;
                if (maxVal == -FLT_MAX)
                    maxVal = 0.0f;

                char typeAndName[64];
                StrFormat(typeAndName, "%s - %s (%u frames)", _markerType, m_name.c_str(), m_numFramesPerGroup * kArraySize);

                ImGui::SetNextWindowSize(kMarkerStatsWindowSize, ImGuiCond_FirstUseEver);
                const ImVec2 graphSize = kMarkerStatsWindowSize / 2.f;

                if (ImGui::Begin(typeAndName, &m_open))
                {
                    ImGui::PushID(this);
                    // Max, min, average graph
                    ///////////////////////////////

                    ImGui::Columns(3, nullptr, false);
                    ImGui::SetColumnOffset(1, graphSize.x);
                    ImGui::SetColumnOffset(2, graphSize.x + kMarkerStatsColumnSize);

                    const ImVec2 pos = ImGui::GetCursorPos();

                    constexpr ImVec4 greenColor(0.0f, 1.0f, 0.0f, 1.0f);
                    constexpr ImVec4 yellowColor(1.0f, 1.0f, 0.0f, 1.0f);
                    constexpr ImVec4 redColor(1.0f, 60.0f / 255.0f, 60.0f / 255.0f, 1.0f);

                    // The first value is still "in construction", only show the ones that are complete.
                    constexpr int valueCount = kArraySize - 1;

                    // we render a simple graph
                    if (m_numFramesPerGroup == 1 && nbMarker == 1.0f)
                    {
                        ImGui::PushStyleColor(ImGuiCol_PlotLines, yellowColor);
                        ImGui::PlotLines("", m_maxDuration, valueCount, currentGroupIdx + 1, nullptr, plotMin, plotMax, graphSize);
                        ImGui::PopStyleColor();
                    }
                    else
                    {
                        ImGui::PushStyleColor(ImGuiCol_PlotLines, redColor);
                        ImGui::PlotLines("", m_maxDuration, valueCount, currentGroupIdx + 1, nullptr, plotMin, plotMax, graphSize);
                        ImGui::PopStyleColor(1);

                        ImGui::SetCursorPos(pos);
                        ImGui::PushStyleColor(ImGuiCol_PlotLines, greenColor);
                        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));

                        auto plotLines = [](void* _data, const int _index)
                        {
                            // If min is FLT_MAX, it means there are no values => display 0
                            const float* values = (float*)_data;
                            return values[_index] == FLT_MAX ? 0 : values[_index];
                        };

                        ImGui::PlotLines("", plotLines, m_minDuration, valueCount, currentGroupIdx + 1, nullptr, plotMin, plotMax, graphSize);
                        ImGui::PopStyleColor(2);

                        ImGui::SetCursorPos(pos);
                        ImGui::PushStyleColor(ImGuiCol_PlotLines, yellowColor);
                        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
                        ImGui::PlotLines("", m_avgDuration, valueCount, currentGroupIdx + 1, nullptr, plotMin, plotMax, graphSize);
                        ImGui::PopStyleColor(2);

                        ImGui::SetCursorPos(pos);
                        ImGui::InvisibleButton(m_name.c_str(), graphSize);

                        if (ImGui::IsItemHovered())
                        {
                            const float x = (graphSize.x - (ImGui::GetIO().MousePos.x - ImGui::GetItemRectMin().x)) / graphSize.x;
                            const int id = valueCount - (int)(x * valueCount);
                            const int itemIndex = (id + currentGroupIdx) % valueCount;

                            // Empty tooltip to override the last plot line's one
                            ImGui::SetTooltip("");
                            ImGui::BeginTooltip();
                            if (nbMarker > 1.0f || m_numFramesPerGroup > 1)
                            {
                                const float minDuration = m_minDuration[itemIndex] == FLT_MAX ? 0 : m_minDuration[itemIndex];
                                ImGui::TextColored(greenColor, "min:%.3f ms", minDuration);
                                ImGui::TextColored(yellowColor, "avg:%.3f ms", !isnan(m_avgDuration[itemIndex]) ? m_avgDuration[itemIndex] : 0.f);
                                ImGui::TextColored(redColor, "max:%.3f ms", m_maxDuration[itemIndex]);
                            }
                            else // Only one marker per frame so Avg = Max = Min
                            {
                                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0, 1.0f), "%g", m_avgDuration[itemIndex]);
                            }
                            ImGui::EndTooltip();

                            ///////// zoom / dezoom
                            const float wheel = ImGui::GetIO().MouseWheel;
                            if (wheel != 0.0f)
                            {
                                m_plotMin = Math::clamp(plotMin + avgAvg * 0.1f * wheel, 0.f, minAvg);
                            }
                        }
                    }

                    // Infos and reset
                    ///////////////////////////////
                    ImGui::NextColumn();
                    const float currentY = ImGui::GetCursorPosY();
                    ImGui::Text("%.3f", !isnan(plotMax) ? plotMax : 0.01f);
                    ImGui::SetCursorPosY(currentY + graphSize.y - ImGui::GetFontSize());
                    ImGui::Text("%.3f", plotMin);

                    ImGui::NextColumn();

                    bool reset = false;

                    const char* modeNames[] = {
                        "Per marker",
                        "Per frame",
                        "Continuous"
                    };

                    ImGui::PushItemWidth(180.f);
                    int mode = m_mode;
                    reset |= ImGui::Combo("##Mode", &mode, modeNames, (int)std::size(modeNames));
                    m_mode = (StatsMode)mode;
                    ImGui::PopItemWidth();

                    ImGui::Spacing();
                    reset |= ImGui::Button("Reset");

                    if (reset)
                    {
                        for (int i = 0; i < kArraySize; ++i)
                        {
                            ClearFrameGroup(i);
                        }
                        m_lastUpdatedFrame = 0;
                    }

                    if (m_plotMin != 0.0f && ImGui::Button("Reset Scale"))
                    {
                        m_plotMin = 0.0f;
                    }
                    ImGui::Columns(1);

                    if (m_numFramesPerGroup == 1 && nbMarker == 1.0f)
                    {
                        ImGui::TextColored(yellowColor, "avg:%.3f ms", !isnan(maxAvg) ? maxAvg : 0.f);
                    }
                    else
                    {
                        ImGui::TextColored(greenColor, "min:%.3f", minVal); ImGui::SameLine();
                        ImGui::TextColored(yellowColor, "avg:%.3f", !isnan(avgAvg) ? avgAvg : 0.f); ImGui::SameLine();
                        ImGui::TextColored(redColor, "max:%.3f", maxVal);
                    }

                    // Marker counter graph if more than 1 marker per frame
                    ///////////////////////////////
                    if (nbMarker > 1.0f && m_mode != Continuous)
                    {
                        ImGui::Columns(3, nullptr, false);
                        ImGui::SetColumnOffset(1, graphSize.x);
                        ImGui::SetColumnOffset(2, graphSize.x + kMarkerStatsColumnSize);
                        ImGui::PushStyleColor(ImGuiCol_PlotLines, yellowColor);
                        ImGui::PlotLines("", m_nbMarkers, (int)std::size(m_nbMarkers) - 1, currentGroupIdx + 1, nullptr, 0.0f, nbMarker, graphSize);
                        ImGui::PopStyleColor();
                        ImGui::NextColumn();
                        const float curY = ImGui::GetCursorPosY();
                        ImGui::Text("%g", nbMarker);
                        ImGui::SetCursorPosY(curY + graphSize.y - ImGui::GetFontSize());
                        ImGui::Text("%g", 0.0f);

                        ImGui::NextColumn();
                        ImGui::Text("Num Markers");
                        ImGui::Text("per Frame");
                        ImGui::Columns(1);
                    }
                    ImGui::PopID();
                }
                ImGui::End();
            }
        }
    };


    // ----- Profiler viewer implementation ----------------------------------------------

    ProfilerViewer::ProfilerViewer()
    {
        m_cpuDisplayDepth = kMarkerMaxDepth;
        m_gpuDisplayDepth = kMarkerMaxDepth;
    }

    ProfilerViewer::~ProfilerViewer()
    {
    }

    void ProfilerViewer::SetPauseKey(const ImGuiKey _key, const char* _keyName)
    {
        m_pauseKey = _key;
        m_pauseKeyName = _keyName;
    }

    void ProfilerViewer::SetFullscreenKey(const ImGuiKey _key, const char* _keyName)
    {
        m_fullscreenKey = _key;
        m_fullscreenKeyName = _keyName;
    }

    void ProfilerViewer::SetPaused(const bool _paused)
    {
        m_paused = _paused;

        if (!m_paused)
        {
            for (size_t i = 0; i < m_frameRingBuffer.size(); i++)
            {
                m_frameRingBuffer[i].Clear();
            }
        }
    }

    bool ProfilerViewer::IsPaused() const
    {
        return m_paused;
    }

    void ProfilerViewer::Display(bool* _windowOpened)
    {
        if (!AVA_VERIFY(ImGuiTools::WithinFrameScope(), 
            "[ProfilerViewer] Display() can only be called within ImGui frame scope."))
        {
            return;
        }

        _UpdateGuiData();

        const ImVec2 visibleRect = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowSize(ImVec2(visibleRect.x * 0.5f, visibleRect.y * 0.25f), ImGuiCond_FirstUseEver);

        u32 windowFlags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoFocusOnAppearing;

        if (m_fullscreen)
        {
            ImGui::SetNextWindowPos(ImVec2(0.f, 0.f));
            ImGui::SetNextWindowSize(ImVec2(visibleRect.x, visibleRect.y));

            windowFlags |= ImGuiWindowFlags_NoSavedSettings 
                        | ImGuiWindowFlags_NoResize 
                        | ImGuiWindowFlags_NoMove 
                        | ImGuiWindowFlags_NoCollapse;
        }

        ImGui::Begin("Profiler", _windowOpened, windowFlags);

        // Start 2 columns: the first one is for the thread names, the second one for the timeline and markers
        ImGui::Columns(2, nullptr, kDisplayColumnBorders);
        ImGui::SetColumnOffset(1, kThreadNameColumnSize);

        ImGui::NextColumn();
        const ImVec2 timelineAreaStart = ImGui::GetCursorPos();
        const float timeLineSizeX = ImGui::GetWindowSize().x - timelineAreaStart.x;
        ImGui::NextColumn();

        // Display timeline
        _DisplayTimeline(timelineAreaStart, ImVec2(timeLineSizeX, 0.f));

        // Add an invisible button to know if the mouse is hovering the timeline area.
        ImGui::NextColumn();
        const ImVec2 savedCursorPos = ImGui::GetCursorPos();
        ImGui::SetCursorPosY(timelineAreaStart.y);
        ImGui::InvisibleButton("Timeline Area", ImVec2(timeLineSizeX, savedCursorPos.y - timelineAreaStart.y));
        const bool timeLineAreaHoveredRect = ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly);
        ImGui::SetCursorPos(savedCursorPos);
        ImGui::NextColumn();

        // Go back to 1 column and start a child window to display the markers.
        // This way the timeline is always visible.
        ImGui::Columns(1);

        // No scroll with mouse wheel, scroll with left click and dragging
        u32 childWindowFlags = ImGuiWindowFlags_NoScrollWithMouse;
        if (ImGui::IsMouseDragging(0))
        {
            // Do not move the window when dragging in the time line
            childWindowFlags |= ImGuiWindowFlags_NoMove;
        }
        ImGui::BeginChild("MarkerArea", ImVec2(0,0), false, childWindowFlags);

        // Leave a little bit of space before drawing the marker area
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10.f);

        // And 2 columns again to display thread names / markers
        ImGui::Columns(2, nullptr, kDisplayColumnBorders);
        // Subtract padding to the column offset to be aligned the with the columns in the main window.
        ImGui::SetColumnOffset(1, kThreadNameColumnSize - ImGui::GetStyle().WindowPadding.x);

        for (u32 threadIdx = 0; threadIdx < m_threads.size(); ++threadIdx)
        {
            ThreadData& thread = m_threads[threadIdx];

            // If any of the frames use too many markers, the buffer if *probably* full.
            // (hard to tell for sure since new markers are not pushed when the buffer is actually full)
            bool markerBufferFull = false;
            for (u32 displayFrameIdx = 0; displayFrameIdx < m_nbFrameDisplayed; ++displayFrameIdx)
            {
                if (_GetFrame(threadIdx, displayFrameIdx).markerCount> thread.maxMarkersPerFrame)
                {
                    markerBufferFull = true;
                    break;
                }
            }

            if (markerBufferFull)
            {
                // Red text when the buffer is full.
                ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor(1.f, 0.f, 0.f));
            }

            // First column: the thread names
            ImGui::SetNextItemOpen(thread.open);
            const bool nodeOpen = ImGui::TreeNode((void*)(intptr_t)threadIdx, "%s%s", thread.name, markerBufferFull ? " (FULL)" : "");
            if (nodeOpen)
            {
                // Display number of markers (of the 1st frame) below the thread name
                const FrameData& frame = _GetFrame(threadIdx, 0);
                ImGui::Text("%s%d markers\nper frame", markerBufferFull ? "At least\n" : "", frame.markerCount);
                ImGui::TreePop();
            }

            thread.open = nodeOpen;

            if (markerBufferFull)
            {
                ImGui::PopStyleColor();
            }

            // Second column: the markers
            ImGui::NextColumn();

            u32 maxDepth;
            if (!m_threads[threadIdx].open)
            {
                maxDepth = 1;
            }
            else
            {
                maxDepth = threadIdx == 0 ? m_gpuDisplayDepth : m_cpuDisplayDepth;

                for (u32 depth = 0; depth<maxDepth; ++depth)
                {
                    u32 totalMarkers = 0;
                    for (u32 displayFrameIdx = 0; displayFrameIdx < m_nbFrameDisplayed; ++displayFrameIdx)
                    {
                        const FrameData& frame = _GetFrame(threadIdx, displayFrameIdx);
                        totalMarkers += (u32)frame.markers[depth].size();
                    }

                    if (totalMarkers == 0)
                    {
                        // No more markers at this depth, this is the real max depth
                        maxDepth = depth;
                        break;
                    }
                }
            }

            ImVec4 markerArea;
            markerArea.x = ImGui::GetCursorPosX();
            markerArea.y = ImGui::GetCursorPosY();
            markerArea.z = markerArea.x + timeLineSizeX;
            markerArea.w = 0.f;

            ImVec2 frameStartPos;
            frameStartPos.x = markerArea.x + m_timeOffset * m_timeScale;
            frameStartPos.y = markerArea.y;

            ImGuiTextFilter highlightFilter;
            strcpy(highlightFilter.InputBuf, m_highlightFilter);
            highlightFilter.Build();

            for (u32 displayFrameIdx = 0; displayFrameIdx < m_nbFrameDisplayed; ++displayFrameIdx)
            {
                FrameData& frame = _GetFrame(threadIdx, displayFrameIdx);
                for (u32 markerIdx = 0; markerIdx < frame.markers[0].size(); ++markerIdx)
                {
                    _DisplayMarkers(markerArea, frameStartPos, frame, displayFrameIdx, threadIdx, 0, maxDepth, markerIdx, highlightFilter);
                }
                const FrameData& cpuFrame = _GetFrame(kCpuThreadId, displayFrameIdx);
                frameStartPos.x += cpuFrame.duration * m_timeScale;
            }

            // Move cursor so that next thread will be below this thread's markers
            ImGui::SetCursorPosY(markerArea.y + maxDepth * (kMarkerHeight + 1.f) + 10.f);

            ImGui::NextColumn();
        }

        // Add an invisible button to know if the mouse is hovering the timeline area.
        ImGui::NextColumn();
        ImGui::SetCursorPosY(ImGui::GetScrollY());
        ImGui::InvisibleButton("Markers Area", ImGui::GetWindowSize());
        const bool markerAreaHovered = ImGui::IsItemHovered();
        const bool markerAreaHoveredRect = ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly);
        ImGui::NextColumn();

        ImGui::Columns(1);
        float newTimeOffset = m_timeOffset;

        if (markerAreaHovered || timeLineAreaHoveredRect)
        {
            // Time Scale
            if (const float wheel = ImGui::GetIO().MouseWheel)
            {
                const float mousePosX = ImGui::GetIO().MousePos.x - timelineAreaStart.x - ImGui::GetWindowPos().x;
                const float timeAtMouseBefore = mousePosX / m_timeScale;

                m_timeRange *= 1.f + wheel * -0.25f;
                const float timeScale = (ImGui::GetWindowSize().x - timelineAreaStart.x) / m_timeRange;

                const float timeAtMouseAfter = mousePosX / timeScale;
                newTimeOffset += (timeAtMouseAfter - timeAtMouseBefore);
            }

            if (ImGui::IsMouseDragging(0))
            {
                // Time offset
                if (ImGui::IsMouseDown(0) && ImGui::GetIO().MouseDelta.x != 0)
                {
                    newTimeOffset += ImGui::GetIO().MouseDelta.x / (m_timeScale);
                }
                // Scroll
                const float scrollY = ImGui::GetIO().MouseDelta.y * -1.0f + ImGui::GetScrollY();
                if (scrollY <= ImGui::GetScrollMaxY())
                {
                    ImGui::SetScrollY(scrollY);
                }
            }
        }

        // Navigating in frames
        if(ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
        {
            newTimeOffset = _GetTimeOffsetToNavigationFrame(1, false);
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow))
        {
            newTimeOffset = _GetTimeOffsetToNavigationFrame(1, true);
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))
        {
            newTimeOffset = _GetTimeOffsetToNavigationFrame(m_nbFrameDisplayed, false);
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))
        {
            newTimeOffset = _GetTimeOffsetToNavigationFrame(m_nbFrameDisplayed, true);
        }

        // Back to the main window
        ImGui::EndChild();

        ImGui::Columns(2, nullptr, kDisplayColumnBorders);
        ImGui::SetColumnOffset(1, kThreadNameColumnSize);

        ImGui::NextColumn();

        ImVec2 markerAreaEnd;
        markerAreaEnd.x = ImGui::GetWindowSize().x;
        markerAreaEnd.y = ImGui::GetCursorPosY();

        // Draw the lines separating the frames
        _DisplayFrameDimensions(timelineAreaStart, markerAreaEnd - timelineAreaStart);

        // Handle zooming/scrolling if the mouse is on the right part of the window.
        if (timeLineAreaHoveredRect || markerAreaHoveredRect)
        {
            // Mouse vertical line
            ImDrawList* imDrawList = ImGui::GetWindowDrawList();
            ImVec2 cursorLineFrom = ImGui::GetWindowPos();
            cursorLineFrom.x = ImGui::GetIO().MousePos.x;
            ImVec2 cursorLineTo = cursorLineFrom;
            cursorLineTo.y += ImGui::GetWindowHeight();
            imDrawList->AddLine(cursorLineFrom, cursorLineTo, ImColor(1.f,1.f,1.f,0.8f), 1.0f);
        }

        // We're hovering the timeline area or the marker area, but not clicking on a marker.
        if (markerAreaHovered)
        {
            constexpr auto rightMouseButton = 1;
            constexpr auto wheelMouseButton = 2;

            // Right mouse button
            if (ImGui::IsMouseDragging(rightMouseButton))
            {
                _DisplayMarkerAreaHovered(ImGui::GetMouseDragDelta(rightMouseButton), ImColor(0, 175, 201, 50), ImColor(0, 100, 201, 50));
            }
            // Mouse wheel button
            else if (ImGui::IsMouseDragging(wheelMouseButton))
            {
                const auto delta = ImGui::GetMouseDragDelta(wheelMouseButton);
                _DisplayMarkerAreaHovered(delta, ImColor(175, 201, 0, 50), ImColor(100, 201, 0, 50));
            }
        }
        _DisplayMenuBar();
        m_timeOffset = newTimeOffset;

        ImGui::End();

        // GPU marker stats widgets
        for (u32 i = 0; i < m_gpuMarkerStats.size(); ++i)
        {
            m_gpuMarkerStats[i].Display("GPU", m_guiOldestFrameIdx + m_nbFrameDisplayed - 1);
        }

        // CPU marker stats widgets
        for (u32 i = 0; i < m_cpuMarkerStats.size(); ++i)
        {
            m_cpuMarkerStats[i].Display("CPU", m_guiOldestFrameIdx + m_nbFrameDisplayed - 1);
        }
    }


    //----- Logic helpers -----------------------------------------------------------------

    void ProfilerViewer::_UpdateGuiData()
    {
        const auto* profiler = Profiler::GetInstance();
        if (profiler && !m_paused)
        {
            // Updates threads data (CPU + GPU)
            {
                static constexpr u32 threadCount = 2;
                m_threads.resize(threadCount);

                // Checks frame ring buffer size
                if ((u32)m_frameRingBuffer.size() != threadCount * m_nbFrameDisplayed)
                {
                    // The number of threads or the number of frames changed -> clear the ring buffer
                    for (u32 i = 0; i < m_frameRingBuffer.size(); ++i)
                    {
                        m_frameRingBuffer[i].Clear();
                    }
                    // And resize it
                    const u32 totalNbOfFrames = threadCount * m_nbFrameDisplayed;
                    m_frameRingBuffer.resize(totalNbOfFrames);
                }

                // Updates GPU thread name and marker budget
                {
                    const GpuThreadInfo* gpuThread = profiler->GetGpuThread();
                    m_threads[kGpuThreadId].SetName(gpuThread->GetName());
                    m_threads[kGpuThreadId].maxMarkersPerFrame = gpuThread->GetTotalMarkerCount() / gpuThread->GetNumFramesToRecord();
                }

                // Updates CPU thread name and marker budget
                {
                    const CpuThreadInfo* cpuThread = profiler->GetCpuThread();
                    m_threads[kCpuThreadId].SetName(cpuThread->GetName());
                    m_threads[kCpuThreadId].maxMarkersPerFrame = cpuThread->GetTotalMarkerCount() / cpuThread->GetNumFramesToRecord();
                }
            }

            // Avoids crash (infinite loop) when debugger runs early on start-up
            if (profiler->GetCurrentFrame() < profiler->GetNumFramesToRecord())
            {
                return;
            }

            const u32 profilerOldestFrame = profiler->GetCurrentFrame() - profiler->GetNumFramesToRecord() + 1;
            if (profilerOldestFrame < m_nbFrameDisplayed)
            {
                return;
            }

            // We put the profiler oldest frame data in the gui newest frame
            const u32 guiNewestFrame = profilerOldestFrame;
            m_guiOldestFrameIdx = guiNewestFrame - m_nbFrameDisplayed + 1;

            // And we display from the gui oldest frame to the gui newest.
            const u32 newestDisplayFrameIdx = m_nbFrameDisplayed - 1;

            // Updates the marker stats for the oldest frame.
            _UpdateMarkerStats(profilerOldestFrame);

            // First retrieves GPU markers
            const GpuThreadInfo* gpuThread = profiler->GetGpuThread();
            if (gpuThread->IsFrameAvailable(profilerOldestFrame))
            {
                FrameData& frame = _GetFrame(kGpuThreadId, newestDisplayFrameIdx);
                _BuildFrameData(gpuThread, profilerOldestFrame, frame, m_gpuMarkerStats, true);
            }

            // Then retrieves CPU markers
            const CpuThreadInfo* cpuThread = profiler->GetCpuThread();
            if (cpuThread->IsFrameAvailable(profilerOldestFrame))
            {
                FrameData& frame = _GetFrame(kCpuThreadId, newestDisplayFrameIdx);
                _BuildFrameData(cpuThread, profilerOldestFrame, frame, m_cpuMarkerStats, false);
            }
        }
        m_frameCount++;
    }

    ProfilerViewer::FrameData& ProfilerViewer::_GetFrame(const u32 _threadIdx, const u32 _frameIndex)
    {
        u32 finalFrameIdx = (m_guiOldestFrameIdx + _frameIndex) % m_nbFrameDisplayed;
        finalFrameIdx += _threadIdx * m_nbFrameDisplayed;
        return m_frameRingBuffer[finalFrameIdx];
    }

    void ProfilerViewer::_AddMarkerStats(const u32 _threadIdx, const char* _markerName, const u32 _nbFrames)
    {
        const u32 hash = HashStr(_markerName);
        std::vector<MarkerStats>& allStats = _threadIdx == kGpuThreadId ? m_gpuMarkerStats : m_cpuMarkerStats;

        // we check if it don't already exists
        for (auto it = allStats.begin(); it != allStats.end(); ++it)
        {
            if (it->m_hash == hash && it->m_numFramesPerGroup * MarkerStats::kArraySize == _nbFrames)
            {
                return;
            }
        }
        allStats.push_back(MarkerStats());
        allStats.back().Init(hash, _markerName, _nbFrames);
    }

    void ProfilerViewer::_UpdateMarkerStats(const u32 _frameIdx)
    {
        for (auto it = m_cpuMarkerStats.begin(); it != m_cpuMarkerStats.end();)
        {
            if (it->m_open)
            {
                it->StartFrame(_frameIdx);
                ++it;
            }
            else
            {
                it = m_cpuMarkerStats.erase(it);
            }
        }

        for (auto it = m_gpuMarkerStats.begin(); it != m_gpuMarkerStats.end();)
        {
            if (it->m_open)
            {
                it->StartFrame(_frameIdx);
                ++it;
            }
            else
            {
                it = m_gpuMarkerStats.erase(it);
            }
        }

    }

    template <class ThreadInfoType>
    void ProfilerViewer::_BuildFrameData(ThreadInfoType* _thread, u32 _frameIdx, FrameData& _frame, std::vector<MarkerStats>& _stats, const bool _isGpuThread)
    {
        _frame.Clear();
        _frame.duration = static_cast<float>(_thread->GetFrameDurationNS(_frameIdx) / 1000000.f);
        _frame.frameCount = m_frameCount;

        u32 markerIdx = _thread->GetFirstMarker(_frameIdx);
        while (ProfilerThreadInfo::IsMarkerValid(markerIdx))
        {
            MarkerData marker{};
            marker.markerIdx = markerIdx;
            marker.name = _thread->GetMarkerName(markerIdx);
            marker.start = static_cast<float>(_thread->GetMarkerStartTimeNS(_frameIdx, markerIdx) / 1000000.f);
            marker.duration = _thread->IsMarkerOpen(markerIdx) ? 1000000.f : static_cast<float>(_thread->GetMarkerDurationNS(markerIdx) / 1000000.f);

            marker.firstChildIdx = 0xFFFF;
            marker.lastChildIdx = 0xFFFF;

            u16 parentIdx;
            const int depth = _thread->GetMarkerDepth(markerIdx);

            // Sometimes the parent of this marker was invalidated, in this case we can't display this marker, we have to skip it.
            if (depth > 0)
            {
                // The parent is supposed to end after the child. If it's not the case, it's not the right parent.
                if (_frame.markers[depth-1].size() > 0
                    && (_frame.markers[depth-1].back().start 
                        + _frame.markers[depth-1].back().duration > marker.start))
                {
                    // Looks like this is the right parent!
                    parentIdx = (u16)_frame.markers[depth-1].size() - 1;
                }
                else
                {
                    // The parent is not there, we can't display this marker.
                    markerIdx = _thread->GetNextMarker(_frameIdx, markerIdx);
                    continue;
                }
            }
            else
            {
                // No parents at depth 0, it's normal.
                parentIdx = 0xFFFF;
            }

            marker.parentIdx = parentIdx;

            // At depth 0, marker color is based on the name hash.
            if (depth == 0)
            {
                marker.colorIdx = HashStr(marker.name);
            }
            // Other depths use the parent's color.
            else if (depth > 0 && depth < kMarkerMaxDepth)
            {
                const auto& markers = _frame.markers[depth-1];
                marker.colorIdx = markers.back().colorIdx;

                // Update first child / last child indices
                MarkerData& parent = _frame.markers[depth-1].back();
                const u32 childIdx = (u32)_frame.markers[depth].size();

                if (parent.firstChildIdx == 0xFFFF)
                {
                    parent.firstChildIdx = childIdx;
                }
                parent.lastChildIdx = childIdx+1;
            }
            if (depth >= 0 && depth < kMarkerMaxDepth)
            {
                _frame.markers[depth].push_back(marker);
            }

            // Check if we need stats for this marker
            if (!_stats.empty())
            {
                const u32 hash = HashStr(marker.name);

                for (u32 i = 0; i < _stats.size(); ++i)
                {
                    // We only update marker stats for non global markers
                    // For global GPU/CPU frame marker stats we compute the value after the loop using the last and the first marker
                    if (_stats[i].m_hash == hash)
                    {
                        _stats[i].AddStats(marker, _frameIdx);
                    }
                }
            }

            markerIdx = _thread->GetNextMarker(_frameIdx, markerIdx);
        }

        // Updates global frame stats (for GPU thread and main thread only)
        if (!_stats.empty() && !_frame.markers[0].empty())
        {
            const float frameStart = _frame.markers[0].front().start;
            const float frameEnd = _frame.markers[0].back().start + _frame.markers[0].back().duration;

            MarkerData marker{};
            marker.name = kFrameMarkerStatsName;
            marker.start = frameStart;
            marker.duration = frameEnd - frameStart;

            const u32 hash = HashStr(marker.name);

            for (u32 i = 0; i < _stats.size(); ++i)
            {
                if (_stats[i].m_hash == hash)
                {
                    _stats[i].AddStats(marker, _frameIdx);
                }
            }
        }

        // Special case for GPU markers: the markers order depend on the command buffer construction order,
        // not on their execution by the GPU, which means the markers are not necessarily sorted by time.
        if (_isGpuThread)
        {
            std::sort(_frame.markers[0].begin(), _frame.markers[0].end(), [](const MarkerData& _a, const MarkerData& _b) 
                {
                return _a.start < _b.start;
            });

            // Update the parentIdx of the child markers
            for (int parentIdx = 0; parentIdx < (int)_frame.markers[0].size(); ++parentIdx)
            {
                const MarkerData& parent = _frame.markers[0][parentIdx];
                for (int childIdx = parent.firstChildIdx; childIdx < parent.lastChildIdx; ++childIdx)
                {
                    _frame.markers[1][childIdx].parentIdx = parentIdx;
                }
            }
        }

        // Computes the total number of markers for this frame
        _frame.markerCount = 0;
        for (u32 i = 0; i < kMarkerMaxDepth; ++i)
        {
            _frame.markerCount += (u32)_frame.markers[i].size();
        }
    }


    //----- UI helpers --------------------------------------------------------------------

    struct GroupedMarker
    {
        const char* name;
        StringHash nameHash;
        float totalDuration;
        u32 colorIdx;
        u32 count;

        static bool CompareNameHashes(const GroupedMarker& _a, const GroupedMarker& _b)
        {
            return _a.nameHash < _b.nameHash;
        }

        static bool CompareDurations(const GroupedMarker& _a, const GroupedMarker& _b)
        {
            return _a.totalDuration > _b.totalDuration;
        }
    };

    static ImColor GetRandomColor(const u32 _seed)
    {
        static const ImColor kMarkerColors[] =
        {
            ImColor(0xFF, 0x3D, 0x31), // red
            ImColor(0x36, 0x5C, 0xE5), // blue
            ImColor(0x0F, 0xA1, 0x47), // green
            ImColor(0xDB, 0xA5, 0x28), // yellow
            ImColor(0x98, 0x33, 0xFF), // purple
        };
        return kMarkerColors[_seed % std::size(kMarkerColors)];
    }

    static int RoundUpToMultiple(const int _num, const int _multiple) 
    {
        return _num - _num % _multiple + (_num > 0 ? _multiple : 0); 
    }

    static void DrawTimeline(const ImVec2 _pos, const float _startTime, const float _endTime, const int _multiple, const int _skipMultiple, const float _barSize, const u32 _col, const float _timeScale)
    {
        const float spaceBetweenBars = _timeScale * (float)_multiple;

        if (spaceBetweenBars < 5.f)
        {
            return;
        }

        const int iStartTime = RoundUpToMultiple((int)_startTime, _multiple);
        const int iEndTime   = RoundUpToMultiple((int)_endTime  , _multiple);

        ImDrawList* drawList = ImGui::GetWindowDrawList();

        // Draw the bars
        {
            float lineX = _pos.x + (float)iStartTime * _timeScale;
            for (int i = iStartTime; i < iEndTime; i += _multiple, lineX += spaceBetweenBars)
            {
                if (_skipMultiple && (i % _skipMultiple == 0))
                    continue;

                drawList->AddLine(ImVec2(lineX, _pos.y), ImVec2(lineX, _pos.y + _barSize), _col);
            }
        }

        // Draw the text
        if (spaceBetweenBars > 100.f)
        {
            float lineX = _pos.x + (float)iStartTime * _timeScale;
            for (int i = iStartTime; i < iEndTime; i += _multiple, lineX += spaceBetweenBars)
            {
                if (_skipMultiple && (i % _skipMultiple == 0))
                {
                    continue;
                }

                char text[16];
                StrFormat(text, "%d.0ms", i);

                drawList->AddText(ImVec2(lineX, _pos.y + _barSize), _col, text);
            }
        }
    }

    static bool MarkerButton(const char* _name, const ImVec2 _pos, const ImVec2 _size, const ImColor _color)
    {
        ImGui::SetCursorPos(_pos);

        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)_color);

        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor(
            _color.Value.x * 0.8f,
            _color.Value.y * 0.8f,
            _color.Value.z * 0.8f));

        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor(
            _color.Value.x * 1.2f,
            _color.Value.y * 1.2f,
            _color.Value.z * 1.2f));

        bool clicked;

        // If the button is too small, don't display the text.
        if (_size.x > 10.f)
            clicked = ImGui::Button(_name, _size);
        else
            clicked = ImGui::Button("", _size);

        ImGui::PopStyleColor(3);

        return clicked;
    }

    void ProfilerViewer::_DisplayTimeline(const ImVec2& _pos, const ImVec2& _size)
    {
        if (!Profiler::IsEnabled())
        {
            ImGui::PushStyleColor(ImGuiCol_Text, IM3D_COL_YELLOW);
            ImGui::Text("Profiler not enabled");
            ImGui::Separator();
            ImGui::PopStyleColor();
        }

        float startTime = -m_timeOffset;
        const float endTime   = startTime + m_timeRange;

        ImGui::DragFloat("cursor", &startTime, 10.f / m_timeScale, -1000.f, 1000.f, "%.1fms");
        m_timeOffset = -startTime;

        ImGui::DragFloat("scope", &m_timeRange , 10.f / m_timeScale, 0.f, 1000.f, "%.1fms");
        m_timeScale = _size.x / m_timeRange;

        ImGui::PushFont(IMGUI_FONT_BOLD);

        if (ImGui::Button(ICON_FA_BACKWARD_FAST))
        {
            m_timeOffset = _GetTimeOffsetToNavigationFrame(m_nbFrameDisplayed, false);
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_BACKWARD))
        {
            m_timeOffset = _GetTimeOffsetToNavigationFrame(1, false);
        }
        ImGui::SameLine();
        if (ImGui::Button(m_paused ? ICON_FA_PLAY : ICON_FA_PAUSE))
        {
            SetPaused(!IsPaused());
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_FORWARD))
        {
            m_timeOffset = _GetTimeOffsetToNavigationFrame(1, true);
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_FORWARD_FAST))
        {
            m_timeOffset = _GetTimeOffsetToNavigationFrame(m_nbFrameDisplayed, true);
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_MINUS))
        {
            m_timeRange += 5;
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_PLUS))
        {
            m_timeRange -= 5;
        }

        ImGui::PopFont();
        ImGui::NextColumn();

        ImVec2 originPos;
        originPos.x = ImGui::GetWindowPos().x + _pos.x + m_timeOffset * m_timeScale;
        originPos.y = ImGui::GetWindowPos().y + _pos.y;

        DrawTimeline(originPos, startTime, endTime,  1,  5, 7.f, ImColor(ImGui::GetStyle().Colors[ImGuiCol_Separator]), m_timeScale);
        DrawTimeline(originPos, startTime, endTime,  5, 10, 11.f, ImColor(ImGui::GetStyle().Colors[ImGuiCol_Separator]), m_timeScale);
        DrawTimeline(originPos, startTime, endTime, 10,  0, 15.f, ImColor(ImGui::GetStyle().Colors[ImGuiCol_Separator]), m_timeScale);

        ImGui::NextColumn();
    }

    float ProfilerViewer::_GetTimeOffsetToNavigationFrame(const u32 _offset, const bool _positive)
    {
        if (!Profiler::IsEnabled() || _offset == 0)
        {
            return 0;
        }

        m_frameIdxNavigation = _positive
            ? Math::min(m_nbFrameDisplayed - 1, m_frameIdxNavigation + _offset)
            : m_frameIdxNavigation - Math::min(_offset, m_frameIdxNavigation);

        // find the timeOffset for the frame
        auto timeOffset = 0.0f;
        for (u32 i = 0; i < m_frameIdxNavigation; i++)
        {
            const FrameData& cpuFrame = _GetFrame(kCpuThreadId, i);
            timeOffset += cpuFrame.duration;
        }

        return -timeOffset;
    }

    void ProfilerViewer::_DisplayMarkers(const ImVec4& _markerArea, const ImVec2& _frameStartPos, const FrameData& _frame,
        const u32 _frameIdx, const u32 _threadIdx, const u32 _depth, const u32 _maxDepth, u32& _markerIdx, ImGuiTextFilter& _filter)
    {
        const MarkerData& marker = _frame.markers[_depth][_markerIdx];
        const ImVec2 markerPos = _GetMarkerPos(marker, _frameStartPos);
        const float markerSizeX = marker.duration * m_timeScale - kMarkerSpacing;

        if (markerPos.x + markerSizeX < _markerArea.x)
        {
            // Marker is off screen (on the left)
            return;
        }

        if (markerPos.x > _markerArea.z)
        {
            // Marker is off screen (on the right)
            // we don't need to check following markers because they will be offscreen too
            // so set the markerIdx to the end.
            _markerIdx = (u32)_frame.markers[_depth].size();
            return;
        }

        // Check if the button is big enough to be displayed
        if (markerSizeX < m_lodMarkerThreshold)
        {
            // It's too small, try to group it with siblings in a LOD marker. The rules are: 
            // - If next sibling is also too small and not too far away, add it to the LOD marker.
            // - When no more siblings can be grouped, if the resulting LOD marker is too small, don't display it.

            float lodEndTime = marker.start + marker.duration;
            u32   lodMarkerCount = 1;
            ImColor lodColor(0x88, 0x88, 0x88);
            
            if (_filter.IsActive())
            {
                lodColor = ImColor(30,30,30,128);
            }

            u32 nextMarkerIdx = _markerIdx + 1;

            while (
                nextMarkerIdx < _frame.markers[_depth].size() 
                && _frame.markers[_depth][nextMarkerIdx].parentIdx == marker.parentIdx)
            {
                const MarkerData& siblingMarker = _frame.markers[_depth][nextMarkerIdx];

                if (_filter.IsActive())
                {
                    if (_filter.PassFilter(siblingMarker.name))
                    {
                        lodColor = GetRandomColor(siblingMarker.colorIdx);
                    }
                }

                const float dist = (siblingMarker.start - lodEndTime) * m_timeScale;
                const float sizeX = siblingMarker.duration * m_timeScale;

                // If the next sibling is big enough or too far away, we stop the LOD here.
                if (sizeX >= m_lodMarkerThreshold || dist >= m_lodMarkerThreshold)
                {
                    break;
                }

                lodEndTime = siblingMarker.start + siblingMarker.duration;
                lodMarkerCount++;

                _markerIdx++;
                nextMarkerIdx++;
            }

            const float lodDuration = lodEndTime - marker.start;
            const float lodSizeX = lodDuration * m_timeScale - kMarkerSpacing;

            if (lodMarkerCount > 1)
            {
                // Check if the LOD is big enough to be displayed
                if (lodSizeX >= m_minMarkerSize)
                {
                    ImGui::PushID(&marker);

                    MarkerButton("...", markerPos, ImVec2(lodSizeX, kMarkerHeight), lodColor);

                    if (ImGui::IsItemHovered())
                    {
                        if (ImGui::IsMouseDoubleClicked(0))
                        {
                            m_threads[_threadIdx].open = !m_threads[_threadIdx].open;
                        }

                        _ShowLodMarkerToolTip(lodMarkerCount, lodDuration, _frame, _markerIdx, _depth);
                    }

                    ImGui::PopID();
                }

                // We just displayed an LOD marker (or we skipped it because it was too small).
                // Stop the recursion here, we don't want to display any child.
                return;
            }
        }

        // Either the marker is big enough not to be a LOD
        // or it's an LOD of 1 marker so we display it as a normal marker.

        // Check if the button is big enough to be displayed at all
        if (markerSizeX >= m_minMarkerSize)
        {
            ImVec4 color;
            if (_filter.IsActive())
            {
                if (_filter.PassFilter(marker.name))
                {
                    // retain color
                    color = GetRandomColor(marker.colorIdx);
                }
                else
                {
                    // gray out
                    color = ImColor(30,30,30,128);
                }
            }
            else
            {
                color = GetRandomColor(marker.colorIdx);
            }

            static constexpr u32 kNbShades = 5;
            static constexpr float kShadeDiff = 0.15f;
            static const float depthShade = 1.f - (_depth % kNbShades) * kShadeDiff;

            const ImColor colorShade (
                Math::saturate(color.x * depthShade),
                Math::saturate(color.y * depthShade),
                Math::saturate(color.z * depthShade));

            ImGui::PushID(&marker);

            MarkerButton(marker.name, markerPos, ImVec2(markerSizeX, kMarkerHeight), colorShade);

            if (m_paused && ImGui::BeginPopupContextItem("marker context menu"))
            {
                if (ImGui::MenuItem("Show stats (100 frames)"))
                {
                    _AddMarkerStats(_threadIdx, marker.name, 100);
                }
                if (ImGui::MenuItem("Show stats (1000 frames)")) 
                {
                    _AddMarkerStats(_threadIdx, marker.name, 1000);
                }
                if (ImGui::MenuItem("Show stats (10000 frames)")) 
                {
                    _AddMarkerStats(_threadIdx, marker.name, 10000);
                }
                ImGui::EndPopup();
            }
            else if (ImGui::IsItemHovered())
            {
                if (ImGui::IsMouseDoubleClicked(0))
                {
                    m_threads[_threadIdx].open = !m_threads[_threadIdx].open;
                }

                ImGui::BeginTooltip();
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::Text("%s", marker.name);
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::Text("(%.3fms)", marker.duration);
                ImGui::Text("%.3fms from frame start", marker.start);
                ImGui::Text("%.2f%% of frame time", marker.duration * 100.f / _frame.duration);
                ImGui::Text("%d child markers", marker.lastChildIdx - marker.firstChildIdx);
                ImGui::Text("%.3fms in entire frame", _GetFrameDurationForMarker(marker, _frameIdx, _depth));
                ImGui::EndTooltip();
            }

            ImGui::PopID();

            // Recurse on the children
            if (_depth + 1 < _maxDepth)
            {
                for (u32 childIdx = marker.firstChildIdx; childIdx < marker.lastChildIdx; ++childIdx)
                {
                    ImVec2 nextDepthStartPos = _frameStartPos;
                    nextDepthStartPos.y += kMarkerHeight + kMarkerSpacing;

                    _DisplayMarkers(_markerArea, nextDepthStartPos, _frame, _frameIdx, _threadIdx, _depth+1, _maxDepth, childIdx, _filter);
                }
            }
        }
    }

    void ProfilerViewer::_ShowLodMarkerToolTip(const u32 _lodMarkerCount, const float _lodDuration, const FrameData& _frame, const u32 _lastMarkerIdx, const int _depth) const
    {
        ImGui::BeginTooltip();
        ImGui::Text("%u markers", _lodMarkerCount);
        ImGui::SameLine();
        ImGui::Text("(%.3fms)", _lodDuration);

        const u32 firstMarkerIdx = _lastMarkerIdx - (_lodMarkerCount-1);
        const float lodStartTime = _frame.markers[_depth][firstMarkerIdx].start;
        ImGui::Text("%.3fms from frame start", lodStartTime);
        ImGui::Text("%.2f%% of frame time", _lodDuration * 100.f / _frame.duration);

        // Group markers in the LOD by names
        std::vector<GroupedMarker> groupedMarkers;
        for (u32 i = 0; i < _lodMarkerCount; ++i)
        {
            const u32 markerIdx = firstMarkerIdx + i;
            const MarkerData& marker = _frame.markers[_depth][markerIdx];

            GroupedMarker gm;
            gm.nameHash = StringHash(marker.name);
            gm.count = 1;
            gm.name = marker.name;
            gm.colorIdx = marker.colorIdx;
            gm.totalDuration = marker.duration;

            auto it = std::lower_bound(
                groupedMarkers.begin(),
                groupedMarkers.end(),
                gm, GroupedMarker::CompareNameHashes);

            if (it == groupedMarkers.end() || it->nameHash != gm.nameHash)
            {
                groupedMarkers.insert(it, gm);
            }
            else
            {
                it->count++;
                it->totalDuration += marker.duration;
            }
        }

        // Sort them by duration
        std::sort(groupedMarkers.begin(), groupedMarkers.end(), GroupedMarker::CompareDurations);

        // Display them
        ImGui::BeginGroup();
        for (auto it = groupedMarkers.begin(); it != groupedMarkers.end(); ++it)
        {
            const GroupedMarker& gm = *it;
            ImColor color = GetRandomColor(gm.colorIdx);

            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)color);
            ImGui::Text("%s", gm.name);
            ImGui::PopStyleColor();
        }
        ImGui::EndGroup();
        ImGui::SameLine();
        ImGui::BeginGroup();
        for (auto it = groupedMarkers.begin(); it != groupedMarkers.end(); ++it)
        {
            const GroupedMarker& gm = *it;
            ImGui::Text("x%d (%.3fms, %.2f%%)", gm.count, gm.totalDuration, gm.totalDuration * 100.f / _lodDuration);
        }
        ImGui::EndGroup();
        ImGui::EndTooltip();
    }

    float ProfilerViewer::_GetFrameDurationForMarker(const MarkerData& _marker, const u32 _frameIdx, const u32 _depth)
    {
        const auto* profiler = Profiler::GetInstance();
        auto frameDuration = 0.f;
        
        if (!profiler)
        {
            return frameDuration;
        }

        static auto getThreadDurationForMarker = [](const MarkerData& _threadMarker, const FrameData& _frame, const u32 _markerDepth)
        {
            const auto& markersAtDepth = _frame.markers[_markerDepth];
            auto duration = 0.f;

            for (const auto& markerAtDepth : markersAtDepth)
            {
                if (markerAtDepth.name == _threadMarker.name)
                {
                    duration += markerAtDepth.duration;
                }
            }
            return duration;
        };
        
        // GPU Frame
        const GpuThreadInfo* gpuThread = profiler->GetGpuThread();
        if (gpuThread && gpuThread->IsFrameAvailable(_frameIdx + m_guiOldestFrameIdx))
        {
            const FrameData& frame = _GetFrame(kGpuThreadId, _frameIdx);
            frameDuration += getThreadDurationForMarker(_marker, frame, _depth);
        }

        // CPU Frame
        const CpuThreadInfo* cpuThread = profiler->GetCpuThread();
        if (cpuThread && cpuThread->IsFrameAvailable(_frameIdx + m_guiOldestFrameIdx))
        {
            const FrameData& frame = _GetFrame(kCpuThreadId, _frameIdx);
            frameDuration += getThreadDurationForMarker(_marker, frame, _depth);
        }

        return frameDuration;
    }

    ImVec2 ProfilerViewer::_GetMarkerPos(const MarkerData& _marker, const ImVec2& _frameStartPos) const
    {
        ImVec2 markerPos;
        markerPos.x = _frameStartPos.x + _marker.start * m_timeScale;
        markerPos.y = _frameStartPos.y;

        return markerPos;
    }

    void ProfilerViewer::_DisplayFrameDimensions(const ImVec2& _pos, const ImVec2& _size)
    {
        if (!Profiler::IsEnabled())
        {
            return;
        }

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        const ImVec2 screenPos = ImVec2(ImGui::GetWindowPos()) + _pos;
        const auto separatorColor = ImColor(ImGui::GetStyle().Colors[ImGuiCol_Separator]);
        
        ImVec2 framePos;
        framePos.x = screenPos.x + m_timeOffset * m_timeScale;
        framePos.y = screenPos.y;

        // Draw the first vertical separator on the left
        drawList->AddLine(framePos, ImVec2(framePos.x, framePos.y + _size.y), separatorColor);

        for (u32 displayFrameIdx = 0; displayFrameIdx < m_nbFrameDisplayed; ++displayFrameIdx)
        {
            ImColor dimensionColor;
            ImVec2 txtPos;

            // Draws the CPU duration / arrows
            const FrameData& cpuFrame = _GetFrame(kCpuThreadId, displayFrameIdx);
            if (!cpuFrame.markers[0].empty())
            {
                // Draw the frame duration text
                char text[32];
                StrFormat(text, "FRAME %.2fms", cpuFrame.duration);

                const ImVec2 cpuTxtSize = ImGui::CalcTextSize(text);
                txtPos = ImVec2(framePos.x + cpuFrame.duration * m_timeScale * 0.5f - cpuTxtSize.x * 0.5f, framePos.y + kFrameDimensionsHeight);
                drawList->AddText(txtPos, ImGui::GetColorU32(ImGuiCol_Text), text);
                dimensionColor = ImGui::GetStyle().Colors[ImGuiCol_Text];

                // Draws the arrows left and right from the text
                {
                    ImVec2 leftArrow; 
                    leftArrow.x = framePos.x + 3.f;
                    leftArrow.y = floorf(txtPos.y + cpuTxtSize.y * 0.5f);
                    if (leftArrow.x < txtPos.x - 3.f)
                    {
                        drawList->AddLine(leftArrow + ImVec2(0.f, -5.f) , leftArrow + ImVec2(0.f, 4.f), dimensionColor);
                        drawList->AddLine(leftArrow, ImVec2(txtPos.x - 3.f, leftArrow.y), dimensionColor);

                        ImVec2 rightArrow; 
                        rightArrow.x = framePos.x + cpuFrame.duration * m_timeScale - 3.f;
                        rightArrow.y = leftArrow.y;
                        drawList->AddLine(rightArrow + ImVec2(0.f, -5.f) , rightArrow + ImVec2(0.f, 4.f), dimensionColor);
                        drawList->AddLine(rightArrow, ImVec2(txtPos.x + cpuTxtSize.x + 3.f, rightArrow.y), dimensionColor);
                    }
                }
            }

            // Draws the GPU duration / arrows
            const FrameData& gpuFrame = _GetFrame(kGpuThreadId, displayFrameIdx);
            if (!gpuFrame.markers[0].empty())
            {
                const float gpuStart = gpuFrame.markers[0].front().start;
                const float gpuEnd = gpuFrame.markers[0].back().start + gpuFrame.markers[0].back().duration;

                char gpuText[32];
                StrFormat(gpuText, "GPU %.2fms", gpuEnd - gpuStart);

                const ImVec2 gpuTxtSize = ImGui::CalcTextSize(gpuText);
                const auto gpuTxtPos = ImVec2(framePos.x + gpuStart * m_timeScale + (gpuEnd - gpuStart) * m_timeScale * 0.5f - gpuTxtSize.x * 0.5f, txtPos.y + kFrameDimensionsHeight);
                drawList->AddText(gpuTxtPos, ImGui::GetColorU32(ImGuiCol_Text), gpuText);

                // Draws the arrows left and right from the text
                ImVec2 leftArrow;
                leftArrow.x = framePos.x + gpuStart * m_timeScale;
                leftArrow.y = floorf(gpuTxtPos.y + gpuTxtSize.y * 0.5f);
                if (leftArrow.x < gpuTxtPos.x - 3.f)
                {
                    drawList->AddLine(leftArrow + ImVec2(0.f, -5.f) , leftArrow + ImVec2(0.f, 4.f), dimensionColor);
                    drawList->AddLine(leftArrow, ImVec2(gpuTxtPos.x - 3.f, leftArrow.y), dimensionColor);

                    ImVec2 rightArrow; 
                    rightArrow.x = framePos.x + gpuEnd * m_timeScale - 3.f;
                    rightArrow.y = leftArrow.y;
                    drawList->AddLine(rightArrow + ImVec2(0.f, -5.f) , rightArrow + ImVec2(0.f, 4.f), dimensionColor);
                    drawList->AddLine(rightArrow, ImVec2(gpuTxtPos.x + gpuTxtSize.x + 3.f, rightArrow.y), dimensionColor);
                }
            }

            // Draws the vertical separator
            framePos.x += cpuFrame.duration * m_timeScale;
            drawList->AddLine(framePos, ImVec2(framePos.x, framePos.y + _size.y), separatorColor);
        }
    }

    void ProfilerViewer::_DisplayMarkerAreaHovered(const ImVec2& _delta, const ImColor& _leftColor, const ImColor& _rightColor) const
    {
        if (_delta.x < FLT_EPSILON)
        {
            return;
        }

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 cursorFrom = ImGui::GetWindowPos();
        cursorFrom.x = ImGui::GetIO().MousePos.x - _delta.x;
        ImVec2 cursorTo = ImGui::GetWindowPos();
        cursorTo.x = ImGui::GetIO().MousePos.x;
        cursorTo.y += ImGui::GetWindowHeight();
        drawList->AddRectFilledMultiColor(cursorFrom, cursorTo, _leftColor, _rightColor, _rightColor, _leftColor);
        const ImVec2 lineFrom(cursorFrom.x, cursorTo.y - 10);
        const ImVec2 lineTo(cursorTo.x - 2.f, cursorTo.y - 10);
        drawList->AddLine(lineFrom, lineTo, ImColor(0, 175, 201, 250), 1.0f);
        drawList->AddLine(ImVec2(lineFrom.x, lineFrom.y - 3.f), ImVec2(lineFrom.x, lineFrom.y + 3.f), ImColor(0, 175, 201, 255));
        drawList->AddTriangleFilled(ImVec2(lineTo.x, lineTo.y), ImVec2(lineTo.x - 3.46f, lineTo.y - 2.f), ImVec2(lineTo.x - 3.46f, lineTo.y + 2.f), ImColor(0, 175, 201, 255));
        char text[20];
        StrFormat(text, "%.3fms", _delta.x / m_timeScale);
        const ImVec2 txtSize = ImGui::CalcTextSize(text);
        const ImVec2 txtPos(cursorFrom.x + (cursorTo.x - cursorFrom.x) / 2.0f - txtSize.x / 2.0f, cursorTo.y - 30.0f);
        drawList->AddText(txtPos, ImGui::GetColorU32(ImGuiCol_Text), text);
    }

    void ProfilerViewer::_DisplayMenuBar()
    {
        const bool prevFullScreen = m_fullscreen;

        if (ImGui::BeginMenuBar())
        {
            // Options menu
            if (ImGui::BeginMenu("Options"))
            {
                ImGui::MenuItem("Fullscreen", m_fullscreenKey != -1 ? m_fullscreenKeyName : nullptr, &m_fullscreen);
                if (ImGui::MenuItem("Pause", m_pauseKey != -1 ? m_pauseKeyName : nullptr, &m_paused))
                {
                    SetPaused(m_paused);
                }

                int nbFrameDisplayed = (int)m_nbFrameDisplayed;
                ImGui::DragInt("Frames Displayed", &nbFrameDisplayed, 0.1f, 1, 30, "%0.f");
                if (!m_paused)
                {
                    m_nbFrameDisplayed = nbFrameDisplayed;
                }
                else
                {
                    if (ImGui::IsItemHovered())
                    {
                        ImGui::SetTooltip("Resume to edit");
                    }
                }

                int lodThreshold = (int)m_lodMarkerThreshold;
                ImGui::DragInt("LOD Threshold", &lodThreshold, 0.1f, 0, 100, "%0.f pixels");
                m_lodMarkerThreshold = (float)lodThreshold;

                int markerMinSizeInt = (int)m_minMarkerSize;
                ImGui::DragInt("Marker Min Size", &markerMinSizeInt, 0.1f, 0, 100, "%0.f pixels");
                m_minMarkerSize = (float)markerMinSizeInt;

                ImGui::DragInt("GPU Max Depth", &m_gpuDisplayDepth, 0.1f, 1, kMarkerMaxDepth);
                ImGui::DragInt("CPU Max Depth", &m_cpuDisplayDepth, 0.1f, 1, kMarkerMaxDepth);

                ImGui::EndMenu();
            }

            // Marker stats menu
            if (ImGui::BeginMenu("Marker stats"))
            {
                if (ImGui::BeginMenu("Show CPU frame stats"))
                {
                    if (ImGui::MenuItem("100 frames"))
                    {
                        _AddMarkerStats(kCpuThreadId, kFrameMarkerStatsName, 100);
                    }
                    if (ImGui::MenuItem("1000 frames")) 
                    {
                        _AddMarkerStats(kCpuThreadId, kFrameMarkerStatsName, 1000);
                    }
                    if (ImGui::MenuItem("10000 frames")) 
                    {
                        _AddMarkerStats(kCpuThreadId, kFrameMarkerStatsName, 10000);
                    }
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Show GPU frame stats"))
                {
                    if (ImGui::MenuItem("100 frames"))
                    {
                        _AddMarkerStats(kGpuThreadId, kFrameMarkerStatsName, 100);
                    }
                    if (ImGui::MenuItem("1000 frames")) 
                    {
                        _AddMarkerStats(kGpuThreadId, kFrameMarkerStatsName, 1000);
                    }
                    if (ImGui::MenuItem("10000 frames")) 
                    {
                        _AddMarkerStats(kGpuThreadId, kFrameMarkerStatsName, 10000);
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenu();
            }

            ImGui::PushItemWidth(100);
            ImGui::InputText("Highlight", m_highlightFilter, std::size(m_highlightFilter));
            ImGui::PopItemWidth();

            ImGui::EndMenuBar();
        }

        if (m_fullscreenKey != -1)
        {
            m_fullscreen ^= ImGui::IsKeyPressed(m_fullscreenKey);
        }

        if (m_pauseKey != -1)
        {
            if (ImGui::IsKeyPressed(m_pauseKey, false))
            {
                SetPaused(!m_paused);
            }
        }

        if (prevFullScreen != m_fullscreen)
        {
            if (m_fullscreen)
            {
                m_savedWindowPos  = ImGui::GetWindowPos();
                m_savedWindowSize = ImGui::GetWindowSize();
            }
            else
            {
                ImGui::SetWindowSize(m_savedWindowSize);
                ImGui::SetWindowPos(m_savedWindowPos);
            }
        }
    }

}
