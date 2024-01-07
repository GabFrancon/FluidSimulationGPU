-- Ava Solution -------------------------------------------------------------

include "Externals/Externals.lua"

workspace "FluidSim"
    architecture "x64"
    startproject "FluidSimApp"
    configurations { "Debug", "Release", "Final" }
    flags { "MultiProcessorCompile" }

    ProjectDir = "%{wks.location}/Projects/"
    ObjRelativeDir = "/obj/%{cfg.architecture}-%{cfg.buildcfg}"
    BinRelativeDir = "/bin/%{cfg.architecture}-%{cfg.buildcfg}"

    filter "action:vs*"
        -- Disable secure CRT warnings for Visual Studio
        defines "_CRT_SECURE_NO_WARNINGS"

    -- Externals --------
    group "Externals"
        include "Externals/GLFW"
        include "Externals/ImGui"
        include "Externals/LZ4"
    group ""

    -- App --------------
    include "Ava"
    include "FluidSimApp"
