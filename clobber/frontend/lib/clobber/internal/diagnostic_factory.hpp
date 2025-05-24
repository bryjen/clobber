#pragma once

#include "clobber/pch.hpp"

#include <clobber/ast/ast.hpp>
#include <clobber/common/diagnostic.hpp>

namespace {
    clobber::Diagnostic
    parser_err(size_t span_start, size_t span_len, const std::string &general_msg, const std::string &msg) {
        using clobber::Diagnostic;
        return Diagnostic(Diagnostic::Stage::Parser, Diagnostic::Severity::Error, span_len, span_len, general_msg, msg);
    }
}; // namespace

namespace diag {
    namespace parser {
        const std::string ierr_default_general_msg = "An internal error occurred.";
        const std::string ierr_default_msg         = "Could not parse, an internal error occurred.";

        clobber::Diagnostic
        internal_err(const clobber::Span &span) {
            return parser_err(span.start, span.length, ierr_default_general_msg, ierr_default_msg);
        }

        clobber::Diagnostic
        internal_err(const clobber::Span &span, const std::string &msg) {
            return parser_err(span.start, span.length, parser::ierr_default_general_msg, parser::ierr_default_msg);
        }

        clobber::Diagnostic
        missing_closing_paren_err(const clobber::Span &span) {
            return parser_err(span.start, span.length, parser::ierr_default_general_msg, parser::ierr_default_msg);
        }

        clobber::Diagnostic
        not_valid_type_token(const clobber::Span &span) {
            const std::string general_msg = "Invalid type expression";
            const std::string msg         = "Token is not a valid type";
            return parser_err(span.start, span.length, general_msg, msg);
        }
    }; // namespace parser

    namespace semantics {
        /* @brief */
        clobber::Diagnostic
        could_not_infer_type_error(size_t span_start, size_t span_len) {
            throw 0;
        }

        /* @brief */
        clobber::Diagnostic
        unresolved_symbol_error(size_t span_start, size_t span_len) {
            throw 0;
        }

        /* @brief */
        clobber::Diagnostic
        not_a_fn_error(size_t span_start, size_t span_len) {
            throw 0;
        }

        /* @brief */
        clobber::Diagnostic
        argument_type_mismatch_error(size_t span_start, size_t span_len) {
            throw 0;
        }

        /* @brief */
        clobber::Diagnostic
        mismatched_arity_error(size_t span_start, size_t span_len, size_t expected, size_t actual) {
            throw 0;
        }
    }; // namespace semantics
}; // namespace diag