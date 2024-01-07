#include <avapch.h>

#include <Application/Window.h>
#include <Debug/Assert.h>

//--- Include Windows ------------------------
#include <dwmapi.h>
//--------------------------------------------

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    void* Window::_GetNativeHandle(GLFWwindow* _windowHandle)
    {
        return glfwGetWin32Window(_windowHandle);
    }

    void Window::_SetDarkMode(GLFWwindow* _windowHandle, const bool _enable)
    {
        const BOOL useDarkMode = _enable;
        auto* winWindowHandle =  glfwGetWin32Window(_windowHandle);
        DwmSetWindowAttribute(winWindowHandle, DWMWA_USE_IMMERSIVE_DARK_MODE, &useDarkMode, sizeof useDarkMode);
    }

}
