#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ExprBase;
struct CompilationUnit;

// TODO: Make all errors/warnings subclasses of `Diagnostic`

struct Diagnostic {};

/* @brief */
struct SemanticWarning final : Diagnostic {
public:
    SemanticWarning();
    SemanticWarning(int span_start, int span_len, const std::string &general_err_msg, const std::string &err_msg);
    ~SemanticWarning();

    std::string GetFormattedErrorMsg(const std::string &file, const std::string &source_text);

protected:
    int span_start;
    int span_len;
    std::string general_err_msg;
    std::string err_msg;
};

/* @brief */
struct SemanticError final : Diagnostic {
public:
    SemanticError();
    SemanticError(int span_start, int span_len, const std::string &general_err_msg, const std::string &err_msg);
    ~SemanticError();

    std::string GetFormattedErrorMsg(const std::string &file, const std::string &source_text);

protected:
    int span_start;
    int span_len;
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
    std::unique_ptr<CompilationUnit> compilation_unit;
    std::unique_ptr<TypeMap> type_map;
    std::vector<Diagnostic> diagnostics;
};

namespace clobber {
/* @brief */
std::unique_ptr<SemanticModel> get_semantic_model(const std::string &source_text, std::unique_ptr<CompilationUnit> &&compilation_unit);
}; // namespace clobber