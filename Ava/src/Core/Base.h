#pragma once
/// @file Base.h
/// @brief Root header file included by all Ava files.

// disables stupid Intellisense warnings
#if defined(_MSC_VER)
    #pragma warning(disable : 4146 26110 26812 26451 26495 28251)
#endif

// platform detection
#include <Platform/Platform.h>

// forward declaration
#include <Core/Forward.h>

// common std headers
#include <memory>
#include <string>
#include <cstdint>
#include <cstdarg>
#include <cassert>
#include <vector>
#include <array>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <type_traits>
#include <queue>
#include <bitset>
#include <optional>
#include <functional>
#include <numeric>
#include <mutex>

// common macros
#define AVA_EXPAND(x)    x
#define AVA_BIT(x)       (1 << x)
#define AVA_STRINGIFY(x) #x
#define AVA_CONCAT(x, y) x ## y

#define AVA_DISABLE_WARNINGS_BEGIN __pragma(warning(push, 0))
#define AVA_DISABLE_WARNINGS_END   __pragma(warning(pop))

#define AVA_BIND_FN(_func) \
    [this](auto&&... args) -> decltype(auto) { return this->_func(std::forward<decltype(args)>(args)...); }

// debug defines
#if !defined(AVA_FINAL)
    #define AVA_ENABLE_LOG
    #define AVA_ENABLE_ASSERT
    #define AVA_ENABLE_PROFILER
    #define AVA_STORE_STRING_HASH
#endif

// fixed-size integer types
namespace Ava {

    using u8  = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;

    using s8  = int8_t;
    using s16 = int16_t;
    using s32 = int32_t;
    using s64 = int64_t;

}
