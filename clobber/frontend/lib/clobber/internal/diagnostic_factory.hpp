#pragma once

namespace clobber {
    struct Diagnostic;
} // namespace clobber

namespace diagnostics {
    namespace semantics {
        namespace errors {
            /* @brief */
            clobber::Diagnostic could_not_infer_type_error(size_t span_start, size_t span_len);

            /* @brief */
            clobber::Diagnostic unresolved_symbol_error(size_t span_start, size_t span_len);

            /* @brief */
            clobber::Diagnostic not_a_fn_error(size_t span_start, size_t span_len);

            /* @brief */
            clobber::Diagnostic argument_type_mismatch_error(size_t span_start, size_t span_len);

            /* @brief */
            clobber::Diagnostic mismatched_arity_error(size_t span_start, size_t span_len, size_t expected, size_t actual);
        } // namespace errors
    }; // namespace semantics
}; // namespace diagnostics