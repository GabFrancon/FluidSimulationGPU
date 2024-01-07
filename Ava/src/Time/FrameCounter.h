#pragma once
/// @file FrameCounter.h
/// @brief

namespace Ava {

    class FrameCounter
    {
    public:
        FrameCounter() = default;
        explicit FrameCounter(const u8 _totalFrameCount) : m_totalFrameCount(_totalFrameCount) {}

        u8 GetCurrentFrame() const { return m_currentFrame; }
        u8 GetTotalFrameCount() const { return m_totalFrameCount; }

        void NextFrame() { m_currentFrame = (m_currentFrame + 1) % m_totalFrameCount; }

    private:
        u8 m_currentFrame = 0;
        u8 m_totalFrameCount = 0;
    };

}
