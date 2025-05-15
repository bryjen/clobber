#pragma once

#include <clobber/common/debug.hpp> // common debug header

#include "pch.hpp"

namespace clobber {
    struct ExprBase;
    struct CompilationUnit;

    struct Diagnostic;
}; // namespace clobber

namespace clobber {
    struct Type {
        enum Kind {
            String,
            Char,
            Int,
            Float,
            Double,
            Bool,
            Func,
        } kind;
        std::vector<std::shared_ptr<Type>> params;
    };

    struct Symbol {
        std::string name;
        std::shared_ptr<Type> type;
    };

    using TypeMap = std::unordered_map<size_t, std::shared_ptr<Type>>;

    /* @brief */
    struct SemanticModel {
        std::unique_ptr<clobber::CompilationUnit> compilation_unit;
        std::unique_ptr<TypeMap> type_map;
    };

    /* @brief */
    std::unique_ptr<SemanticModel> get_semantic_model(std::unique_ptr<clobber::CompilationUnit> &&compilation_unit,
                                                      std::vector<clobber::Diagnostic> &);
}; // namespace clobber