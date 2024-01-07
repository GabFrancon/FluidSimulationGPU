#pragma once
/// @file HeadlessApplication.h
/// @brief

#include <Application/Application.h>
#include <Memory/Memory.h>

namespace Ava {

    /// @brief app settings for HeadlessApplication instances.
    struct HeadlessAppSettings
    {
    };


    /// @brief Subclass of `Application` designed for headless applications. It focuses on functionality that
    /// does not require a traditional graphical user interface. Suitable for tasks like schema reflection,
    /// headless GPU computing, or any application that operates primarily through non-graphical interfaces,
    /// such as APIs and command-line inputs.
    class HeadlessApplication : public Application
    {
        static HeadlessApplication* s_instance;

    public:
        explicit HeadlessApplication(const HeadlessAppSettings& _settings);
        ~HeadlessApplication() override;

        static HeadlessApplication& GetInstance() { return *s_instance; }

        void Run() override;
        void Close();

    private:
        bool m_appRunning = true;
    };

}

