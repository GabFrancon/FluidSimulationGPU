#include <avapch.h>
#include "Profiler.h"

#include <Time/ProfilerThreadInfo.h>
#include <Debug/Assert.h>

namespace Ava {

    Profiler* Profiler::s_instance = nullptr;

    Profiler::Profiler(const ProfilerSettings& _settings)
    {
        m_cpuThread = CpuThreadInfo::Create();
        m_gpuThread = GpuThreadInfo::Create();

        AVA_ASSERT(_settings.numFramesToRecord >= 4, "[Profiler] needs to record at least 4 frames because GPU is late.");
        m_numFramesToRecord = _settings.numFramesToRecord;

        m_cpuThread->Init(_settings.numFramesToRecord, _settings.maxCpuMarkerCount);
        m_gpuThread->Init(_settings.numFramesToRecord, _settings.maxGpuMarkerCount);
    }

    Profiler::~Profiler()
    {
        m_cpuThread->Close();
        m_gpuThread->Close();
        m_numFramesToRecord = 0;

        delete m_cpuThread;
        delete m_gpuThread;
    }

    void Profiler::Init(const ProfilerSettings& _settings)
    {
        if (!s_instance)
        {
            s_instance = new Profiler(_settings);
        }
    }

    void Profiler::Close()
    {
        if (s_instance)
        {
            delete s_instance;
            s_instance = nullptr;
        }
    }

    void Profiler::StartFrame() const
    {
        m_cpuThread->StartFrame(m_currentFrame);
    }

    void Profiler::StartGpuFrame(GraphicsContext* _ctx) const
    {
        m_gpuThread->StartFrame(m_currentFrame, _ctx);
    }

    void Profiler::EndFrame()
    {
        m_cpuThread->EndFrame();
        m_gpuThread->EndFrame();
        m_currentFrame++;
    }

    void Profiler::PushCpuMarker(const char* _name) const
    {
        m_cpuThread->PushMarker(_name);
    }

    void Profiler::PopCpuMarker(const char* _name) const
    {
        m_cpuThread->PopMarker(_name);
    }

    void Profiler::PushGpuMarker(const char* _name) const
    {
        m_gpuThread->PushMarker(_name);
    }

    void Profiler::PopGpuMarker(const char* _name) const
    {
        m_gpuThread->PopMarker(_name);
    }
    
}
