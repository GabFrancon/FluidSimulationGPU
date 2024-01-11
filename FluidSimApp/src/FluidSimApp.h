#pragma once

#include <Application/GUIApplication.h>

namespace Ava {
    class UILayer;
    class DebugLayer;
}

class FluidSimApp final : public Ava::GUIApplication
{
public:
    FluidSimApp(const Ava::GUIAppSettings& _settings);
    ~FluidSimApp() override;

    static FluidSimApp& FluidSimApp::GetApp()
    {
        return static_cast<FluidSimApp&>(GetInstance());
    }

    Ava::UILayer* GetUILayer() const;
    Ava::DebugLayer* GetDebugLayer() const;

private:
    Ava::Ref<Ava::UILayer> m_UILayer;
    Ava::Ref<Ava::DebugLayer> m_debugLayer;
};

