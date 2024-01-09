-- Fluid Sim application ------------------------------------------------------

project "FluidSimApp"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    staticruntime "On"
    externalwarnings "Off"
    openmp "On"

    cudaKeep "On"
    cudaFastMath "On"
    cudaRelocatableCode "On"
    cudaVerbosePTXAS "On"
    cudaMaxRegCount "32"
    buildcustomizations "BuildCustomizations/CUDA 12.3"

    location   (ProjectDir .. "FluidSimApp")
    objdir     (ProjectDir .. "FluidSimApp" .. ObjRelativeDir)
    targetdir  (ProjectDir .. "FluidSimApp" .. BinRelativeDir)
    cudaIntDir (ProjectDir .. "FluidSimApp" .. CudaRelativeDir)

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
        cudaFiles { "src/**.cu" }

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
