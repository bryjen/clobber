
#include <algorithm>
#include <cctype>
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
    try {
        return std::make_optional(std::stoi(str));
    } catch (...) {
        return std::nullopt;
    }
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

// credits: chatgpt for the 3 below:

std::string
StringUtils::trim(const std::string &str) {
    size_t start = 0;
    while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start])))
        ++start;

    size_t end = str.size();
    while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1])))
        --end;

    return str.substr(start, end - start);
}

std::string
StringUtils::remove_newlines(const std::string &str) {
    std::string result;
    result.reserve(str.size());

    std::copy_if(str.begin(), str.end(), std::back_inserter(result), [](char c) { return c != '\n' && c != '\r'; });

    return result;
}

std::string
StringUtils::normalize_whitespace(const std::string &input) {
    std::string result;
    bool in_whitespace = false;

    for (char c : input) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!in_whitespace) {
                result += ' ';
                in_whitespace = true;
            }
        } else {
            result += c;
            in_whitespace = false;
        }
    }

    return result;
}