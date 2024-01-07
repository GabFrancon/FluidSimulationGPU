#pragma once
/// @file Assert.h
/// @brief

#include <Core/Base.h>

//----- Assert manager --------------------------------

namespace Ava {

    /// @brief Interface to emit a break point when a code assertion fails.
    class AssertMgr
    {
    public:
        /// @brief Different behaviors an assert can adopt.
        enum AssertAction
        {
            AssertAsk,   /// Asks the user to decide to break or not.
            AssertBreak, /// Always breaks.
            AssertIgnore /// Never breaks.
        };

        /// @brief Structure to store every assert that occur during execution.
        class AssertMap
        {
        public:
            AssertMap() = default;
            AssertAction Get(const char* _file, int _line, AssertAction _defaultAction);
            void Set(const char* _file, int _line, AssertAction _action);

        private:
            /// @brief File and line where an assert occurred. Used to uniquely identify the asserts.
            struct AssertLocation
            {
                const char* file;
                int line;

                AssertLocation(const char* _file, const int _line) : file(_file), line(_line) {}
                bool operator==(const AssertLocation& _other) const { return line == _other.line && file == _other.file; }
            };

            /// @brief Simple hasher to use AssertLocation in a unordered map.
            struct AssertLocationHasher
            {
                size_t operator()(AssertLocation _location) const;
            };

            std::unordered_map<AssertLocation, AssertAction, AssertLocationHasher> m_locationMap{};
        };

        /// @brief Platform dependent implementation of the assert function.
        typedef bool (*AssertHandler)(AssertAction& _action, const char* _condition, const char* _fileName, int _line, const char* _msg);

        static void EnableAsserts();
        static AssertMap& GetFailedAsserts();
        static AssertHandler& GetAssertHandler();

        /// @brief Main assert function.
        /// @return a boolean that tells the Assert manager if it should insert a breakpoint or not.
        /// @note Don't call this directly, use AVA_ASSERT(...) and AVA_VERIFY(...) macros instead.
        static bool Assert(const char* _cond, const char* _file, int _line, const char* _msg, ...);
    };

}


//----- Assert macros --------------------------------

#if defined(AVA_ENABLE_ASSERT)

    #define AVA_ASSERT_MSG(cond, msg, ...)                                                                \
        do {                                                                                              \
            (void)sizeof(cond);                                                                           \
            if (!(cond)) {                                                                                \
                if (Ava::AssertMgr::Assert(AVA_STRINGIFY(cond), __FILE__, __LINE__, msg, ##__VA_ARGS__))  \
                    AVA_BREAK();                                                                          \
            }                                                                                             \
        } while(0)

    #define AVA_VERIFY_MSG(cond, msg, ...)                                                                                      \
        ((cond) ? true :                                                                                                        \
            (Ava::AssertMgr::Assert(AVA_STRINGIFY(cond), __FILE__, __LINE__, msg, ##__VA_ARGS__) ? AVA_BREAK(), false : false))

#else

    #define AVA_ASSERT_MSG(cond, msg, ...) do { (void)sizeof(cond); } while(0)
    #define AVA_VERIFY_MSG(cond, msg, ...) (cond) ? true : false

#endif

#define AVA_ASSERT_NO_MSG(cond) AVA_ASSERT_MSG(cond, NULL)
#define AVA_VERIFY_NO_MSG(cond) AVA_VERIFY_MSG(cond, NULL)

#define AVA_INTERNAL_GET_MACRO_NAME( \
    ____1, ____2, ____3, ____4, ____5, ____6, ____7, ____8, ____9, ____10, \
    ____11,____12,____13,____14,____15,____16,____17,____18,____19,____20, \
    ____21,____22,____23,____24,____25,____26,____27,____28,____29,____30, \
    ____31,____32,____33,____34,____35,____36,____37,____38,____39,____40, \
    ____41,____42,____43,____44,____45,____46,____47,____48,____49,____50, \
    ____51,____52,____53,____54,____55,____56,____57,____58,____59,____60, \
    ____61,____62,____63,____N,...) ____N

#define AVA_INTERNAL_GET_ASSERT_MACRO(...)                                                                              \
    AVA_EXPAND(                                                                                                         \
        AVA_INTERNAL_GET_MACRO_NAME(                                                                                    \
            __VA_ARGS__##,                                                                                              \
            AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,   \
            AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,   \
            AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,   \
            AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,   \
            AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,   \
            AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,   \
            AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,   \
            AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,   \
            AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_MSG,AVA_ASSERT_NO_MSG \
        )                                                                                                               \
    )

#define AVA_INTERNAL_GET_VERIFY_MACRO(...)                                                                              \
    AVA_EXPAND(                                                                                                         \
        AVA_INTERNAL_GET_MACRO_NAME(                                                                                    \
            __VA_ARGS__##,                                                                                              \
            AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,   \
            AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,   \
            AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,   \
            AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,   \
            AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,   \
            AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,   \
            AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,   \
            AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,   \
            AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_MSG,AVA_VERIFY_NO_MSG \
        )                                                                                                               \
    )

#define AVA_ASSERT(...)                                          \
    AVA_EXPAND(                                                  \
        AVA_INTERNAL_GET_ASSERT_MACRO(__VA_ARGS__)(__VA_ARGS__)  \
     )

#define AVA_VERIFY(...)                                          \
    AVA_EXPAND(                                                  \
        AVA_INTERNAL_GET_VERIFY_MACRO(__VA_ARGS__)(__VA_ARGS__)  \
     )