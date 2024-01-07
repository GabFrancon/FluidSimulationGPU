#pragma once
/// @file Memory.h
/// @brief

#include <Core/Base.h>

// memory macros
#define AVA_MALLOC(size)                          malloc(size)
#define AVA_MALLOC_ALIGNED(size, alignment)       _aligned_malloc(size, alignment)
#define AVA_REALLOC(ptr, size)                    realloc(ptr, size)
#define AVA_REALLOC_ALIGNED(ptr, size, alignment) _aligned_realloc(ptr, size, alignment)
#define AVA_FREE(ptr)                             free(ptr)
#define AVA_FREE_ALIGNED(ptr)                     _aligned_free(ptr)
#define AVA_CACHE_ALIGN                           __declspec(align(AVA_CACHE_LINE_SIZE))
#define AVA_THREAD_LOCAL                          __declspec(thread)

// memory template definitions
namespace Ava {

    template <typename T>
    using Scope = std::unique_ptr<T>;
    template <typename T, typename ... Args>
    constexpr Scope<T> CreateScope(Args&& ... _args) { return std::make_unique<T>(std::forward<Args>(_args)...); }

    template <typename T>
    using Ref = std::shared_ptr<T>;
    template <typename T, typename ... Args>
    constexpr Ref<T> CreateRef(Args&& ... _args) { return std::make_shared<T>(std::forward<Args>(_args)...); }

}

