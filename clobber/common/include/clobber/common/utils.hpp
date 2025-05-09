#ifndef UTILS_HPP
#define UTILS_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>

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
std::optional<int> try_stoi(const std::string &str);
std::string join(const std::string &delimiter, const std::vector<std::string> &lines);
}; // namespace StringUtils

namespace ptr_utils = PointerUtils;
namespace str_utils = StringUtils;

#endif // UTILS_HPP