#ifndef CLOBBER_DEBUG_HPP
#define CLOBBER_DEBUG_HPP

// (mostly) compiler agnostic warning macro
#if defined(_MSC_VER)
#define WARN(msg) __pragma(message(msg))
#elif defined(__GNUC__) || defined(__clang__)
#define WARN(msg) #warning msg
#else
#error Unknown compiler, could not initialize warning macro
#endif

// flag to enable windows CRT
#if !defined(NDEBUG) && defined(CLOBBER_USE_CRT) && defined(_WIN32)
#define CRT_ENABLED
#endif

// main CRT headers
#ifdef CRT_ENABLED
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
// #include <cstdlib>
#include <crtdbg.h>
#endif

// CRT definition to allow tracking of all allocations via replacement of the 'new' keyword
#ifdef CRT_ENABLED
#define new new (_NORMAL_BLOCK, __FILE__, __LINE__)
#define malloc(size) _malloc_dbg(size, _NORMAL_BLOCK, __FILE__, __LINE__)
#define calloc(count, size) _calloc_dbg(count, size, _NORMAL_BLOCK, __FILE__, __LINE__)

#define INIT_CRT_DEBUG()                                                                                                                   \
    do {                                                                                                                                   \
        _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF);                                            \
        _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);                                                                                   \
        _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);                                                                                 \
        _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);                                                                                  \
        _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);                                                                                \
        _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);                                                                                 \
        _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT);                                                                               \
    } while (0)
#endif

#endif // CLOBBER_DEBUG_HPP
