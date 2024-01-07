-- Fluid Sim application ------------------------------------------------------

project "FluidSimApp"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    staticruntime "On"
    externalwarnings "Off"
    openmp "On"

    location  (ProjectDir .. "%{prj.name}")
    objdir    (ProjectDir .. "%{prj.name}" .. ObjRelativeDir)
    targetdir (ProjectDir .. "%{prj.name}" .. BinRelativeDir)

    files
    {
        "src/**.h",
        "src/**.cpp"
    }

    includedirs
    {
        "src",
        "%{IncludeDir.Ava}",
        "%{IncludeDir.rapidjson}",
        "%{IncludeDir.ImGui}",
        "%{IncludeDir.glm}"
    }

    links
    {
        "Ava"
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
        entrypoint "mainCRTStartup"
        kind "WindowedApp"
