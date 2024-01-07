#pragma once
/// @file PrecisionTimer.h
/// @brief

#include <Time/TimeManager.h>

namespace Ava {

    /// @brief High resolution timer for profiling.
    class PrecisionTimer
    {
    public:
        PrecisionTimer()
        {
            Start();
        }

        /// @brief Starts the timer.
        void Start()
        {
            m_start = TimeMgr::GetTime();
        }

        /// @brief Stops the timer.
        void Stop()
        {
            m_stop = TimeMgr::GetTime();
        }

        /// @brief Returns the time elapsed in seconds since Start() was called, or between Start() and Stop() if it was called.
        double Elapsed() const
        {
            double end;
            if (m_stop == 0.0)
            {
                end = TimeMgr::GetTime();
            }
            else
            {
                end = m_stop;
            }
            return end - m_start;
        }

        /// @brief Returns the time elapsed in milliseconds.
        double ElapsedMS() const
        {
            return Elapsed() * 1e3;
        }

        /// @brief Returns the time elapsed in microseconds.
        u64 ElapsedUS() const
        {
            return static_cast<u64>(Elapsed() * 1e6);
        }

        /// @brief Returns the time elapsed in nanoseconds.
        u64 ElapsedNS() const
        {
            return static_cast<u64>(Elapsed() * 1e9);
        }

    private:
        // in seconds
        double m_start = 0.0;
        double m_stop = 0.0;
    };

}
