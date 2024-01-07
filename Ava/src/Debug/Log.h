#pragma once
/// @file Log.h
/// @brief

#include <Core/Base.h>

//----- Log manager -----------------------------------

namespace Ava {

    /// @brief Interface to log an event to the console.
    class LogMgr
    {
    public:

        enum LogCategory
        {
            Core,
            App
        };

        enum LogLevel
        {
            LogTrace,
            LogInfo,
            LogWarning,
            LogError,
            LogCritical
        };

        /// @brief Logger will output to the default output stream (stdout, stderr).
        static void SetupSimpleLogger();
        /// @brief Logger will output colored logs to the console and report session to a .log file.
        static void SetupAppLogger();
        /// @brief Main logging function. Don't call it directly, use log macros instead.
        static void Log(LogCategory _category, LogLevel _level, const char* _msg, ...);
    };
}


//----- Log macros ------------------------------------

#if defined(AVA_ENABLE_LOG)

    #define AVA_CORE_TRACE(...)    Ava::LogMgr::Log(Ava::LogMgr::Core, Ava::LogMgr::LogTrace, __VA_ARGS__)
    #define AVA_CORE_INFO(...)     Ava::LogMgr::Log(Ava::LogMgr::Core, Ava::LogMgr::LogInfo, __VA_ARGS__)
    #define AVA_CORE_WARN(...)     Ava::LogMgr::Log(Ava::LogMgr::Core, Ava::LogMgr::LogWarning, __VA_ARGS__)
    #define AVA_CORE_ERROR(...)    Ava::LogMgr::Log(Ava::LogMgr::Core, Ava::LogMgr::LogError, __VA_ARGS__)
    #define AVA_CORE_CRITICAL(...) Ava::LogMgr::Log(Ava::LogMgr::Core, Ava::LogMgr::LogCritical, __VA_ARGS__)

    #define AVA_TRACE(...)         Ava::LogMgr::Log(Ava::LogMgr::App, Ava::LogMgr::LogTrace, __VA_ARGS__)
    #define AVA_INFO(...)          Ava::LogMgr::Log(Ava::LogMgr::App, Ava::LogMgr::LogInfo, __VA_ARGS__)
    #define AVA_WARN(...)          Ava::LogMgr::Log(Ava::LogMgr::App, Ava::LogMgr::LogWarning, __VA_ARGS__)
    #define AVA_ERROR(...)         Ava::LogMgr::Log(Ava::LogMgr::App, Ava::LogMgr::LogError, __VA_ARGS__)
    #define AVA_CRITICAL(...)      Ava::LogMgr::Log(Ava::LogMgr::App, Ava::LogMgr::LogCritical, __VA_ARGS__)

#else

    #define AVA_CORE_TRACE(...)
    #define AVA_CORE_INFO(...)
    #define AVA_CORE_WARN(...)
    #define AVA_CORE_ERROR(...)
    #define AVA_CORE_CRITICAL(...)

    #define AVA_TRACE(...)
    #define AVA_INFO(...)
    #define AVA_WARN(...)
    #define AVA_ERROR(...)
    #define AVA_CRITICAL(...)

#endif