#pragma once
/// @file Profiler.h
/// @brief

#include <Core/Base.h>

//----- Profiler class ------------------------------------

namespace Ava {

    class CpuThreadInfo;
    class GpuThreadInfo;

    /// @brief Profiler settings, required to create a Profiler.
    struct ProfilerSettings
    {
        u32 numFramesToRecord = 5;
        u32 maxCpuMarkerCount = 128;
        u32 maxGpuMarkerCount = 128;
    };

    /// @brief Singleton profiler for managing CPU and GPU markers across frames,
    /// offering useful profiling tools for performance analysis and optimization.
    class Profiler
    {
        static Profiler* s_instance;

    public:
        static void Init(const ProfilerSettings& _settings);
        static Profiler* GetInstance() { return s_instance; }
        static bool IsEnabled() { return s_instance != nullptr; }
        static void Close();

        /// @brief Marks the beginning of every frame.
        void StartFrame() const;
        /// @brief Marks the beginning of current frame rendering.
        void StartGpuFrame(GraphicsContext* _ctx) const;
        /// @brief Marks the ending of every frame.
        void EndFrame();

        /// @brief Starts measuring CPU time for the caller thread. Calls can be nested.
        void PushCpuMarker(const char* _name) const;
        /// @brief Stops measuring CPU time for the caller thread.
        void PopCpuMarker(const char* _name) const;

        /// @brief Starts measuring GPU time. Calls can be nested.
        void PushGpuMarker(const char* _name) const;
        /// @brief Stops measuring GPU time.
        void PopGpuMarker(const char* _name) const;

        /// @brief Returns the index of the current frame.
        u32 GetCurrentFrame() const { return m_currentFrame; }
        /// @brief Returns the number of frames buffered by the profiler.
        u32 GetNumFramesToRecord() const { return m_numFramesToRecord; }

        /// @brief Returns the collected data of the profiled CPU thread.
        const CpuThreadInfo* GetCpuThread() const { return m_cpuThread; }
        /// @brief Returns the collected data of the profiled GPU thread.
        const GpuThreadInfo* GetGpuThread() const { return m_gpuThread; }

    private:
        explicit Profiler(const ProfilerSettings& _settings);
        ~Profiler();

        u32 m_numFramesToRecord = 0;
        u32 m_currentFrame = 0;

        CpuThreadInfo* m_cpuThread = nullptr;
        GpuThreadInfo* m_gpuThread = nullptr;
    };

    /// @brief A scoped manager for CPU profiling, pushing a marker on creation and
    /// popping it upon scope exit. Simplifies CPU profiling within a limited scope.
    class ScopedCpuMarker
    {
    public:
        ScopedCpuMarker(const char* _name) : m_name(_name)
        {
            if (const auto* profiler = Profiler::GetInstance())
            {
                profiler->PushCpuMarker(m_name);
            }
        }

        ~ScopedCpuMarker()
        {
            if (const auto* profiler = Profiler::GetInstance())
            {
                profiler->PopCpuMarker(m_name);
            }
        }

    private:
        const char* m_name;
    };

    /// @brief A scoped manager for GPU profiling, pushing a marker on creation and
    /// popping it upon scope exit. Simplifies GPU profiling within a limited scope.
    class ScopedGpuMarker
    {
    public:
        ScopedGpuMarker(const char* _name) : m_name(_name)
        {
            if (const auto* profiler = Profiler::GetInstance())
            {
                profiler->PushGpuMarker(m_name);
            }
        }

        ~ScopedGpuMarker()
        {
            if (const auto* profiler = Profiler::GetInstance())
            {
                profiler->PopGpuMarker(m_name);
            }
        }

    private:
        const char* m_name;
    };

}


//----- Profiler macros ------------------------------------

#if defined(AVA_ENABLE_PROFILER)

    #define PUSH_CPU_MARKER(name) if (const Ava::Profiler* profiler = Ava::Profiler::GetInstance()) profiler->PushCpuMarker(name)
    #define PUSH_GPU_MARKER(name) if (const Ava::Profiler* profiler = Ava::Profiler::GetInstance()) profiler->PushGpuMarker(name)

    #define POP_CPU_MARKER(name)  if (const Ava::Profiler* profiler = Ava::Profiler::GetInstance()) profiler->PopCpuMarker(name)
    #define POP_GPU_MARKER(name)  if (const Ava::Profiler* profiler = Ava::Profiler::GetInstance()) profiler->PopGpuMarker(name)

    #define AUTO_CPU_MARKER(name) Ava::ScopedCpuMarker AVA_CONCAT(_cpuMarker, __LINE__) (name)
    #define AUTO_GPU_MARKER(name) Ava::ScopedGpuMarker AVA_CONCAT(_gpuMarker, __LINE__) (name)
    #define AUTO_CPU_GPU_MARKER(name) AUTO_CPU_MARKER(name); AUTO_GPU_MARKER(name)

#else

    #define PUSH_CPU_MARKER(name)
    #define PUSH_GPU_MARKER(name)

    #define POP_CPU_MARKER(name)
    #define POP_GPU_MARKER(name)

    #define AUTO_CPU_MARKER(name)
    #define AUTO_GPU_MARKER(name)
    #define AUTO_CPU_GPU_MARKER(name)

#endif
