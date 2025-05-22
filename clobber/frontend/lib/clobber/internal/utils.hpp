#pragma once

#include <cstddef>
#include <typeindex>

#include "clobber/ast/ast.hpp"

namespace {
    std::vector<std::unique_ptr<clobber::IdentifierExpr>>
    deepcopy_identifiers(const std::vector<std::unique_ptr<clobber::IdentifierExpr>> &identifiers) {
        std::vector<std::unique_ptr<clobber::IdentifierExpr>> copies;
        for (const auto &identifier : identifiers) {
            copies.push_back(identifier->clone_nowrap());
        }
        return copies;
    }

    std::vector<std::unique_ptr<clobber::Expr>>
    deepcopy_exprs(const std::vector<std::unique_ptr<clobber::Expr>> &exprs) {
        std::vector<std::unique_ptr<clobber::Expr>> copies;
        for (const auto &expr : exprs) {
            copies.push_back(expr->clone());
        }
        return copies;
    }

    size_t
    combine_hashes(const std::vector<std::size_t> &hashes) {
        std::size_t seed = 0;
        for (std::size_t h : hashes) {
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
} // namespace