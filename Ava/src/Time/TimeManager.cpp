#include <avapch.h>
#include "TimeManager.h"

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <GLFW/glfw3.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    TimeMgr* TimeMgr::s_instance = nullptr;

    void TimeMgr::Init()
    {
        if (!s_instance)
        {
            s_instance = new TimeMgr();
        }
    }

    void TimeMgr::Shutdown()
    {
        if (s_instance)
        {
            delete s_instance;
        }
    }

    void TimeMgr::Process()
    {
        const double previousTime = m_lastFrameStartTime;
        const double currentTime = GetTime();

        m_lastFrameStartTime = currentTime;
        m_lastFrameDuration = currentTime - previousTime;
    }

    double TimeMgr::GetTime()
    {
        return glfwGetTime();
    }

    void TimeMgr::SetTime(const double _time)
    {
        glfwSetTime(_time);
    }

}
