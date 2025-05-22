#pragma once

#include <clobber/common/pch.hpp>

#define NOT_IMPLEMENTED() throw 69420;

namespace PointerUtils {

    template <typename T>
    std::vector<std::reference_wrapper<const T>>
    get_expr_views(const std::vector<std::unique_ptr<T>> &objs) {
        std::vector<std::reference_wrapper<const T>> views;
        for (const auto &obj : objs) {
            views.push_back(std::cref(*obj));
        }
        return views;
    }
}; // namespace PointerUtils

namespace StringUtils {
    std::string repeat(const std::string &s, size_t n);
    std::string spaces(size_t count);
    std::string join(const std::string &delimiter, const std::vector<std::string> &lines);

    std::string trim(const std::string &str);
    std::string remove_newlines(const std::string &str);
    std::string normalize_whitespace(const std::string &input);
    std::string norm(const std::string &str);

    std::optional<int> try_stoi(const std::string &str);
    std::optional<float> try_stof(const std::string &str);
    std::optional<double> try_stod(const std::string &str);
}; // namespace StringUtils

namespace ptr_utils = PointerUtils;
namespace str_utils = StringUtils;