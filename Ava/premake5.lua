-- Ava library ----------------------------------------------------------------------

project "Ava"
    kind "StaticLib"
    language "C++"
    cppdialect "C++17"
    staticruntime "On"
    externalwarnings "Off"

    location  (ProjectDir .. "%{prj.name}")
    objdir    (ProjectDir .. "%{prj.name}" .. ObjRelativeDir)
    targetdir (ProjectDir .. "%{prj.name}" .. BinRelativeDir)

    pchheader "avapch.h"
    pchsource "src/avapch.cpp"

    files
    {
        -- Ava source code
        "src/**.h",
        "src/**.cpp",

        -- Debug views for Ava objects
        "utils/**.natvis",
    }

    includedirs
    {
        -- Ava include dirs
        "src",
        "data",

        -- External include dirs
        "%{IncludeDir.Vulkan}",
        "%{IncludeDir.GLFW}",
        "%{IncludeDir.ImGui}",
        "%{IncludeDir.rigtorp}",
        "%{IncludeDir.VMA}",
        "%{IncludeDir.RenderDoc}",
        "%{IncludeDir.spdlog}",
        "%{IncludeDir.glm}",
        "%{IncludeDir.LZ4}",
        "%{IncludeDir.stb}"
    }

    libdirs
    {
        "%{LibraryDir.Vulkan}"
    }

    links
    {
        -- Vulkan static lib
        "%{Library.Vulkan}",

        -- Built dependency libs
        "GLFW",
        "ImGui",
        "LZ4"
    }

    filter "system:windows"
        systemversion "latest"

    filter "system:linux"
        systemversion "latest"

    filter "configurations:Debug"
        defines "AVA_DEBUG"
        runtime "Debug"
        symbols "On"

    filter "configurations:Release"
        defines "AVA_RELEASE"
        runtime "Release"
        optimize "On"

    filter "configurations:Final"
        defines "AVA_FINAL"
        runtime "Release"
        optimize "On"
