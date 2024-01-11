#pragma once
/// @file EntryPoint.h
/// @brief

#include <Application/Application.h>
#include <Strings/CmdLineParser.h>
#include <Graphics/GraphicsContext.h>

#include <Debug/Log.h>
#include <Debug/Assert.h>
#include <Debug/Capture.h>

extern Ava::Application* Ava::CreateApplication(const CmdLineParser& _args);

int main(const int _argc, char** _argv)
{
    const Ava::CmdLineParser input(_argc, _argv);

#if !defined(AVA_FINAL)
    // Debug tools initialization
    if (!input.HasOption("-nolog"))
    {
        Ava::LogMgr::SetupAppLogger();
    }
    if (!input.HasOption("-noassert"))
    {
        Ava::AssertMgr::EnableAsserts();
    }
    if (input.HasOption("-renderdoc"))
    {
        Ava::CaptureMgr::LoadRenderDoc();
    }
    if (input.HasOption("-graphicsdebug"))
    {
        Ava::GraphicsContext::EnableGraphicsDebug(true);
    }
#endif

    // Startup
    auto* app = CreateApplication(input);

    // Runtime
    app->Run();

    // Shutdown
    delete app;

    return EXIT_SUCCESS;
}