#pragma once
/// @file Timestep.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    /// @brief Basic Timestep class.
    class Timestep
    {
    public:
        Timestep(const double _time = 0.0) : m_time(_time) {}

        double GetSeconds() const { return m_time; }
        double GetMilliSeconds() const { return m_time * 1e3; }
        u64 GetMicroSeconds() const { return static_cast<u64>(m_time * 1e6); }
        u64 GetNanoSeconds() const { return static_cast<u64>(m_time * 1e9); }

    private:
        double m_time;
    };

}