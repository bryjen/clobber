#pragma once

namespace clobber {
    struct SemanticWarning; // clobber/semantics.hpp
    struct SemanticError;   // clobber/semantics.hpp
} // namespace clobber

namespace diagnostics {
    namespace semantics {
        namespace errors {
            /* @brief */
            clobber::SemanticError could_not_infer_type_error(size_t span_start, size_t span_len);

            /* @brief */
            clobber::SemanticError unresolved_symbol_error(size_t span_start, size_t span_len);

            /* @brief */
            clobber::SemanticError not_a_fn_error(size_t span_start, size_t span_len);

            /* @brief */
            clobber::SemanticError argument_type_mismatch_error(size_t span_start, size_t span_len);

            /* @brief */
            clobber::SemanticError mismatched_arity_error(size_t span_start, size_t span_len, size_t expected, size_t actual);
        } // namespace errors
    }; // namespace semantics
}; // namespace diagnostics