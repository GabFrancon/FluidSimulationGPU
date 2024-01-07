#include <avapch.h>
#include "HeadlessApplication.h"

#include <Debug/Assert.h>

namespace Ava {

    HeadlessApplication* HeadlessApplication::s_instance = nullptr;

    HeadlessApplication::HeadlessApplication(const HeadlessAppSettings& _settings)
    {
        AVA_ASSERT(!s_instance, "Headless app already exists.");
        s_instance = this;
    }

    HeadlessApplication::~HeadlessApplication()
    {
    }

    void HeadlessApplication::Run()
    {
        // this is the main app loop
        while (m_appRunning)
        {
        }
    }

    void HeadlessApplication::Close()
    {
        m_appRunning = false;
    }

}

