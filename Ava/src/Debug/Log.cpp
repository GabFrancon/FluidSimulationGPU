#include <avapch.h>
#include "Log.h"

#include <Memory/Memory.h>
#include <Files/FilePath.h>
#include <Files/FileManager.h>
#include <Strings/StringBuilder.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <spdlog/spdlog.h>
#include "spdlog/sinks/base_sink.h"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace spdlog {

    namespace sinks {

        /// @brief Simple custom sink logger, suitable for tools.
        template<typename Mutex>
        class simple_sink : public base_sink <Mutex>
        {
        protected:
            void sink_it_(const details::log_msg& _msg) override
            {
                if (_msg.level == level::warn)
                {
                    std::cout << "[Warning] " << _msg.payload << std::endl;
                }
                else if (_msg.level == level::err)
                {
                    std::cerr << "[Error] " << _msg.payload << std::endl;
                }
                else
                {
                    std::cout << _msg.payload << std::endl;
                }
            }

            void flush_() override 
            {
               std::cout << std::flush;
            }
        };

        using simple_sink_t = simple_sink<details::null_mutex>;
        using simple_sink_mt = simple_sink<std::mutex>;
    }

}

namespace Ava {

    static Ref<spdlog::logger> s_coreLogger;
    static Ref<spdlog::logger> s_appLogger;

    void LogMgr::SetupSimpleLogger()
    {
        spdlog::sink_ptr logSink = CreateRef<spdlog::sinks::simple_sink_mt>();
        const Ref<spdlog::logger> logger = CreateRef<spdlog::logger>("console", logSink);

        logger->set_level(spdlog::level::trace);
        logger->flush_on(spdlog::level::trace);
        spdlog::register_logger(logger);

        s_coreLogger = logger;
        s_appLogger = logger;
    }

    void LogMgr::SetupAppLogger()
    {
        // Log file is saved next to the app executable.
        char logPath[MAX_PATH];
        FileMgr::GetExecutablePath(logPath);
        FilePath::ReplaceExtension(logPath, ".log");

        std::vector<spdlog::sink_ptr> logSinks;
        logSinks.emplace_back(CreateRef<spdlog::sinks::stdout_color_sink_mt>());
        logSinks.emplace_back(CreateRef<spdlog::sinks::basic_file_sink_mt>(logPath, true));

        logSinks[0]->set_pattern("%^[%T] %n: %v%$");
        logSinks[1]->set_pattern("[%T] [%l] %n: %v");

        s_coreLogger = CreateRef<spdlog::logger>("AVA", begin(logSinks), end(logSinks));
        spdlog::register_logger(s_coreLogger);
        s_coreLogger->set_level(spdlog::level::trace);
        s_coreLogger->flush_on(spdlog::level::trace);

        s_appLogger = CreateRef<spdlog::logger>("APP", begin(logSinks), end(logSinks));
        spdlog::register_logger(s_appLogger);
        s_appLogger->set_level(spdlog::level::trace);
        s_appLogger->flush_on(spdlog::level::trace);
    }

    void LogMgr::Log(const LogCategory _category, const LogLevel _level, const char* _msg, ...)
    {
        if (!_msg || !s_coreLogger || !s_appLogger)
        {
            return;
        }

        // Gets logger
        Ref<spdlog::logger> logger;
        switch(_category)
        {
            case Core:
                logger = s_coreLogger;
                break;
            case App:
                logger = s_appLogger;
                break;
            default:
                logger = s_appLogger;
        }

        // Formats log message
        static constexpr int kMsgBufferMaxSize = 1024;
        char formattedMsg[kMsgBufferMaxSize]{};

        va_list args;
        va_start(args, _msg);
        StrFormatV(formattedMsg, kMsgBufferMaxSize, _msg, args);
        formattedMsg[sizeof(formattedMsg) - 1] = 0;
        va_end(args);

        // Emits log
        switch(_level)
        {
            case LogTrace:
                logger->trace(formattedMsg);
                break;
            
            case LogInfo:
                logger->info(formattedMsg);
                break;

            case LogWarning:
                logger->warn(formattedMsg);
                break;

            case LogError:
                logger->error(formattedMsg);
                break;

            case LogCritical:
                logger->critical(formattedMsg);
                break;
        }
    }

}
