// native_runtime.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" void
print_cstr(const char *msg) {
    std::printf("%s", msg);
}

extern "C" void
print_i64(int64_t val) {
    std::printf("%lld", static_cast<long long>(val));
}

extern "C" void
print_f32(float val) {
    std::printf("%f", val);
}

extern "C" void
print_f64(double val) {
    std::printf("%lf", val);
}

// Allocates and returns a null-terminated string
extern "C" char *
i64_to_str(int64_t val) {
    auto str  = std::to_string(val);
    char *buf = static_cast<char *>(std::malloc(str.size() + 1));
    std::memcpy(buf, str.c_str(), str.size() + 1);
    return buf;
}

extern "C" char *
f32_to_str(float val) {
    auto str  = std::to_string(val);
    char *buf = static_cast<char *>(std::malloc(str.size() + 1));
    std::memcpy(buf, str.c_str(), str.size() + 1);
    return buf;
}

extern "C" char *
f64_to_str(double val) {
    auto str  = std::to_string(val);
    char *buf = static_cast<char *>(std::malloc(str.size() + 1));
    std::memcpy(buf, str.c_str(), str.size() + 1);
    return buf;
}
