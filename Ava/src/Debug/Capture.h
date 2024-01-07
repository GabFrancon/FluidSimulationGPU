#pragma once
/// @file Capture.h
/// @brief

namespace Ava {

    /// @brief Interface to record and save a single GPU frame capture.
    class CaptureMgr
    {
    public:
        static bool LoadRenderDoc();
        static bool StartFrameCapture();
        static bool EndFrameCapture();
        static bool IsEnabled();
    };

}
