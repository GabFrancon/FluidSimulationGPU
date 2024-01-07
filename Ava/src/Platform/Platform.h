#pragma once
/// @file Platform.h
/// @brief File handling platform detection using predefined macros.

// Windows
#if defined(_WIN32)

    #if defined(_WIN64)
        #define AVA_PLATFORM_WINDOWS
        #define AVA_GRAPHIC_API_VULKAN
        #define AVA_BREAK() __debugbreak()
        #define AVA_YIELD_PROCESSOR() _mm_pause()
        #define AVA_BARRIER() _ReadWriteBarrier()
        #define AVA_CACHE_LINE_SIZE 64

        // missing _DEBUG macro in debug build
        #if defined(AVA_DEBUG) && !defined(_DEBUG)
            #define _DEBUG
        #endif

        // missing NDEBUG macro in release and final builds
        #if (defined(AVA_RELEASE) || defined (AVA_FINAL)) && !defined(NDEBUG)
            #define NDEBUG
        #endif

        // missing NOMINMAX macro
        #if !defined(NOMINMAX)
            #define NOMINMAX
        #endif

        #include <windows.h>

    #else
        #error "x86 Builds are not supported!"
    #endif

// Apple
#elif defined(__APPLE__) || defined(__MACH__)
    #include <targetconditionals.h>

    #if defined(TARGET_IPHONE_SIMULATOR)
        #error "IOS simulator is not supported!"

    #elif defined(TARGET_OS_IPHONE)
        #define AVA_PLATFORM_IOS
        #error "IOS is not supported!"

    #elif defined(TARGET_OS_MAC)
        #define AVA_PLATFORM_MACOS
        #error "MacOS is not supported!"

    #else
        #error "Unknown Apple platform!"
    #endif

// Android
#elif defined(__ANDROID__)

    #define AVA_PLATFORM_ANDROID
    #error "Android is not supported!"

// Linux
#elif defined(__linux__)

    #define AVA_PLATFORM_LINUX
    #error "Linux is not supported!"

// Unknown
#else
    #error "Unknown platform!"
#endif
