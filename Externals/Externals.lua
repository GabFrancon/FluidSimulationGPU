-- Ava dependencies --------------------------------------------------

IncludeDir = {}
IncludeDir["Ava"]             = "%{wks.location}/Ava/src"
IncludeDir["Vulkan"]          = "%{wks.location}/Externals/Vulkan/Include"
IncludeDir["GLFW"]            = "%{wks.location}/Externals/GLFW/include"
IncludeDir["RenderDoc"]       = "%{wks.location}/Externals/RenderDoc"
IncludeDir["VMA"]             = "%{wks.location}/Externals/VMA"
IncludeDir["ImGui"]           = "%{wks.location}/Externals/ImGui"
IncludeDir["glm"]             = "%{wks.location}/Externals/glm"
IncludeDir["spdlog"]          = "%{wks.location}/Externals/spdlog"
IncludeDir["rapidjson"]       = "%{wks.location}/Externals/rapidjson"
IncludeDir["LZ4"]             = "%{wks.location}/Externals/LZ4"
IncludeDir["stb"]             = "%{wks.location}/Externals/stb"
IncludeDir["rigtorp"]         = "%{wks.location}/Externals/rigtorp"

LibraryDir = {}
LibraryDir["Vulkan"] = "%{wks.location}/Externals/Vulkan/Lib"

Library = {}
Library["Vulkan"] = "vulkan-1.lib"
