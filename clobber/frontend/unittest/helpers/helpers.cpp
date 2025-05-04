#include "helpers.hpp"
#include <format>


template <typename T>
bool
assert_vectors_same_size(const std::vector<T> &actual, const std::vector<T> &expected, std::string *out_err_msg) {
    size_t actual_len   = actual.size();
    size_t expected_len = expected.size();

    if (actual_len == expected_len) {
        return true;
    }

    *out_err_msg =
        std::format("Mismatched actual & expected lengths, expected {} but got {}.", expected_len, actual_len);
    return false;
}