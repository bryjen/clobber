#pragma once

#include <iostream>
#include <string>

namespace logging {
    inline void
    info(const std::string &message) {
        std::cout << message << std::endl;
    }

    inline void
    warn(const std::string &message) {
        std::cout << "[WARN] " << message << std::endl;
    }

    inline void
    error(const std::string &message) {
        std::cout << "[ERROR] " << message << std::endl;
    }

    inline void
    set_pattern(const std::string &pattern) {
        // No-op for std::cout, but kept for compatibility
    }

    inline void
    flush() {
        std::cout.flush();
    }

    inline void
    drop(const std::string &logger_name) {
        // No-op for std::cout, but kept for compatibility
    }
} // namespace logging