#pragma once
/// @file TimeManager.h
/// @brief

#include <Core/Base.h>
#include <Time/Timestep.h>

namespace Ava {

    /// @brief Time manager, handles system clock.
    class TimeMgr
    {
        static TimeMgr* s_instance;

    public:
        static void Init();
        static TimeMgr* GetInstance() { return s_instance; }
        static void Shutdown();

        /// @brief Updates the current frame time.
        void Process();

        /// @brief Returns the time at which the last frame started.
        double GetLastFrameStartTime() const { return m_lastFrameStartTime; }
        /// @brief Returns the duration of the last frame.
        Timestep GetLastFrameDuration() const { return m_lastFrameDuration; }

        /// @brief Returns the time elapsed in seconds since the system was initialized.
        static double GetTime();
        /// @brief Sets the time in seconds thus overriding the engine internal clock.
        static void SetTime(double _time);

        /// @brief Returns current time in microseconds.
        static double GetTimeMS() { return GetTime() * 1e3; }
        /// @brief Returns current time in microseconds.
        static u64 GetTimeUS() { return static_cast<u64>(GetTime() * 1e6); }
        /// @brief Returns current time in nanoseconds.
        static u64 GetTimeNS() { return static_cast<u64>(GetTime() * 1e9); }

    private:
        TimeMgr() = default;
        ~TimeMgr() = default;

        double m_lastFrameStartTime = 0;
        Timestep m_lastFrameDuration;
    };

}

#define TIME_MGR Ava::TimeMgr::GetInstance()