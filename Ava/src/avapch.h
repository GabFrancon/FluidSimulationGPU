#pragma once
/// @file avapch.h
/// @brief Precompiled header file to reduce compilation time and clean the include structure.

// ----- STL ----------------------------------------
#include <cassert>            // Assertions.
#include <cstdarg>            // Variable argument handling.
#include <cstdint>            // Fixed-size integer types.
#include <functional>         // Function objects and related utilities.
#include <iomanip>            // Manipulators for stream formatting.
#include <iostream>           // Input and output stream handling.
#include <limits>             // Characteristics of arithmetic types.
#include <memory>             // Smart pointers and memory management.
#include <mutex>              // Mutex and other synchronization primitives.
#include <numeric>            // Numeric operations.
#include <optional>           // Container for optional values.
#include <queue>              // Queue container.
#include <regex>              // Regular expressions.
#include <set>                // Sorted associative set container.
#include <map>                // Sorted associative map container.
#include <sstream>            // String stream processing.
#include <string>             // String class and related utilities.
#include <type_traits>        // Type traits and compile-time type information.
#include <unordered_map>      // Unordered associative map container.
#include <unordered_set>      // Unordered associative set container.
#include <vector>             // Dynamic array.
#include <array>              // Fixed-size array.
#include <algorithm>          // Algorithms on ranges (e.g., sorting).
#include <bitset>             // Fixed-size sequence of bits.

// ----- Memory -------------------------------------
#include <Memory/Memory.h>

// ----- Time ---------------------------------------
#include <Time/Timestep.h>

// ----- Files --------------------------------------
#include <Files/FilePath.h>

// ----- Graphics -----------------------------------
#include <Graphics/Color.h>

// ----- UI -----------------------------------------
#include <UI/UICommon.h>
#include <UI/ImGuiTools.h>

// ----- Math ---------------------------------------
#include <Math/Types.h>
#include <Math/Math.h>
#include <Math/Hash.h>

// ----- Strings ------------------------------------
#include <Strings/StringHash.h>
#include <Strings/StringBuilder.h>

// ----- Events -------------------------------------
#include <Events/WindowEvent.h>
#include <Events/FileEvent.h>
#include <Events/MouseEvent.h>
#include <Events/KeyEvent.h>
#include <Events/GamepadEvent.h>

// ----- Debug --------------------------------------
#include <Debug/Log.h>
#include <Debug/Assert.h>
#include <Debug/Im3D.h>
