project "LZ4"
    kind "StaticLib"
    language "C"
    staticruntime "On"
    externalwarnings "Off"
    warnings "Off"

    location  (ProjectDir .. "Externals/%{prj.name}")
    objdir    (ProjectDir .. "Externals/%{prj.name}" .. ObjRelativeDir)
    targetdir (ProjectDir .. "Externals/%{prj.name}" .. BinRelativeDir)

    files
    {
        "LZ4/lz4.h",
        "LZ4/lz4file.h",
        "LZ4/lz4frame.h",
        "LZ4/lz4hc.h",
        "LZ4/xxhash.h",

        "LZ4/lz4.c",
        "LZ4/lz4file.c",
        "LZ4/lz4frame.c",
        "LZ4/lz4hc.c",
        "LZ4/xxhash.c"
    }

    filter "system:windows"
        systemversion "latest"

    filter "system:linux"
        systemversion "latest"

    filter "configurations:Debug"
        runtime "Debug"
        symbols "On"

    filter "configurations:Release"
        runtime "Release"
        optimize "On"