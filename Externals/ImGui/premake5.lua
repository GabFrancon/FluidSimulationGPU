project "ImGui"
    kind "StaticLib"
    language "C++"
    cppdialect "C++17"
    staticruntime "On"
    externalwarnings "Off"
    warnings "Off"

    location  (ProjectDir .. "Externals/%{prj.name}")
    objdir    (ProjectDir .. "Externals/%{prj.name}" .. ObjRelativeDir)
    targetdir (ProjectDir .. "Externals/%{prj.name}" .. BinRelativeDir)

    files
    {
        "ImGui/imconfig.h",
        "ImGui/imgui.h",
        "ImGui/imgui_internal.h",
        "ImGui/imstb_rectpack.h",
        "ImGui/imstb_textedit.h",
        "ImGui/imstb_truetype.h",
        "ImGui/ImGuizmo.h",

        "ImGui/imgui.cpp",
        "ImGui/imgui_demo.cpp",
        "ImGui/imgui_draw.cpp",
        "ImGui/imgui_tables.cpp",
        "ImGui/imgui_widgets.cpp",
        "ImGui/ImGuizmo.cpp"
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

