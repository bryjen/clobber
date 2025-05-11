#ifndef DIAGNOSTIC_FACTORY_HPP
#define DIAGNOSTIC_FACTORY_HPP

struct SemanticWarning; // clobber/semantics.hpp
struct SemanticError;   // clobber/semantics.hpp

namespace diagnostics {
namespace semantics {
namespace errors {
/* @brief */
SemanticError could_not_infer_type_error(int span_start, int span_len);

/* @brief */
SemanticError unresolved_symbol_error(int span_start, int span_len);

/* @brief */
SemanticError not_a_fn_error(int span_start, int span_len);

/* @brief */
SemanticError argument_type_mismatch_error(int span_start, int span_len);

/* @brief */
SemanticError mismatched_arity_error(int span_start, int span_len, size_t expected, size_t actual);
} // namespace errors
}; // namespace semantics
}; // namespace diagnostics

#endif DIAGNOSTIC_FACTORY_HPP