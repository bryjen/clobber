#pragma once

#include <clobber/common/debug.hpp> // common debug header

#include "pch.hpp"

// TODO: Make all errors/warnings subclasses of `Diagnostic`

namespace clobber {
    struct ExprBase;
    struct CompilationUnit;
    struct Diagnostic {};

    /* @brief */
    struct SemanticWarning final : Diagnostic {
    public:
        SemanticWarning();
        SemanticWarning(size_t span_start, size_t span_len, const std::string &general_err_msg, const std::string &err_msg);
        ~SemanticWarning();

        std::string GetFormattedErrorMsg(const std::string &file, const std::string &source_text);

    protected:
        size_t span_start;
        size_t span_len;
        std::string general_err_msg;
        std::string err_msg;
    };

    /* @brief */
    struct SemanticError final : Diagnostic {
    public:
        SemanticError();
        SemanticError(size_t span_start, size_t span_len, const std::string &general_err_msg, const std::string &err_msg);
        ~SemanticError();

        std::string GetFormattedErrorMsg(const std::string &file, const std::string &source_text);

    protected:
        size_t span_start;
        size_t span_len;
        std::string general_err_msg;
        std::string err_msg;
    };

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
        std::vector<Diagnostic> diagnostics;
    };

    /* @brief */
    std::unique_ptr<SemanticModel> get_semantic_model(std::unique_ptr<clobber::CompilationUnit> &&compilation_unit);
}; // namespace clobber