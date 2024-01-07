#pragma once
/// @file Application.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    /// @brief The base class for Ava applications, providing core functionality for managing the application's lifecycle.
    /// Subclasses of this class are tailored for specific application types, such as headless or GUI applications.
    class Application
    {
    public:
        explicit Application() = default;
        virtual ~Application() = default;

        virtual void Run() = 0;
    };

    /// @brief Function to be implemented by client to run their own Ava application.
    /// @details The basic usage is to instantiate and return a custom App class inherited from Ava::Application().
    /// @param _args The list of string arguments parsed from the command line.
    /// @return the Ava application to be run.
    Application* CreateApplication(const CmdLineParser& _args);

}
