#pragma once

#include "clobber/pch.hpp"

namespace clobber {
    struct Diagnostic;
} // namespace clobber

namespace diag {
    namespace parser {
        clobber::Diagnostic internal_err(size_t span_start, size_t span_len);

        clobber::Diagnostic internal_err(size_t span_start, size_t span_len, const std::string &msg);

        clobber::Diagnostic missing_closing_paren_err(size_t span_start, size_t span_len);
    }; // namespace parser

    namespace semantics {
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
    }; // namespace semantics
}; // namespace diag