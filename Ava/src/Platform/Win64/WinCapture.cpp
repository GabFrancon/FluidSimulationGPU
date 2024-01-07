#include <avapch.h>

#include <Debug/Log.h>
#include <Debug/Capture.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <RenderDoc/renderdoc_app.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    // You will need to install RenderDoc 1.5 or newer version, and set the path to
    // your install (most likely C:/ProgramFiles/RenderDoc) in the PATH env variable.
    // See https://renderdoc.org/docs/in_application_api.html for more details.

    static constexpr char const* kRdocCaptureFilesTemplate = "captures/Ava";
    static RENDERDOC_API_1_5_0* s_rdocAPI = nullptr;

    bool CaptureMgr::LoadRenderDoc()
    {
        auto successfulInit = false;

        if (const HMODULE rdocDll = LoadLibraryA("renderdoc.dll"))
        {
            const auto getRenderDocApi = (pRENDERDOC_GetAPI)GetProcAddress(rdocDll, "RENDERDOC_GetAPI");
            const int ret = getRenderDocApi(eRENDERDOC_API_Version_1_5_0, (void**)&s_rdocAPI);

            if (s_rdocAPI && ret == 1)
            {
                s_rdocAPI->MaskOverlayBits(0, eRENDERDOC_Overlay_All);
                s_rdocAPI->SetCaptureFilePathTemplate(kRdocCaptureFilesTemplate);
                AVA_CORE_INFO("[RenderDoc] 1.5 version successfully loaded.");
                successfulInit = true;
            }
            else
            {
                AVA_CORE_ERROR("[RenderDoc] could not load 1.5 module.");
            }
        }
        else
        {
            AVA_CORE_ERROR("[RenderDoc] Could not find DLL, make sure RenderDoc is installed and up to 1.5 version.");
        }
        return successfulInit;
    }

    bool CaptureMgr::StartFrameCapture()
    {
        if (s_rdocAPI && !s_rdocAPI->IsFrameCapturing())
        {
            s_rdocAPI->StartFrameCapture(nullptr, nullptr);
            return true;
        }
        return false;
    }

    bool CaptureMgr::EndFrameCapture()
    {
        if (s_rdocAPI && s_rdocAPI->IsFrameCapturing())
        {
            const auto ret = s_rdocAPI->EndFrameCapture(nullptr, nullptr);
            if (ret == 1)
            {
                AVA_CORE_INFO("[RenderDoc] successfully captured the frame.");
                return true;
            }
            AVA_CORE_ERROR("[RenderDoc] failed to capture the frame.");
        }
        return false;
    }

    bool CaptureMgr::IsEnabled()
    {
        return s_rdocAPI != nullptr;
    }

}
