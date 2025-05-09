
#include <sstream>

#include "clobber/common/utils.hpp"

std::string
StringUtils::repeat(const std::string &s, size_t n) {
    std::string result;
    result.reserve(s.size() * n);
    for (size_t i = 0; i < n; ++i) {
        result += s;
    }
    return result;
}

std::string
StringUtils::spaces(size_t count) {
    return repeat(" ", count);
}

std::optional<int>
StringUtils::try_stoi(const std::string &str) {
    std::optional<int> opt = std::nullopt;
    try {
        opt = std::make_optional(std::stoi(str));
    } catch (...) {
        // ignored
    }
    return opt;
}

std::string
StringUtils::join(const std::string &delimiter, const std::vector<std::string> &lines) {
    std::ostringstream oss;
    for (size_t i = 0; i < lines.size(); ++i) {
        oss << lines[i];
        if (i + 1 < lines.size()) {
            oss << delimiter;
        }
    }
    return oss.str();
}