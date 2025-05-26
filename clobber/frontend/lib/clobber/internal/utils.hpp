#pragma once

#include <cstddef>
#include <typeindex>

#include "clobber/ast/ast.hpp"

namespace {
    size_t
    combine_hashes(const std::vector<std::size_t> &hashes) {
        std::size_t seed = 0;
        for (std::size_t h : hashes) {
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
} // namespace