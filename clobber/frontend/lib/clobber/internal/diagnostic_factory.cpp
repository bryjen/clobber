#include "clobber/internal/diagnostic_factory.hpp"

#include <clobber/common/diagnostic.hpp>

#include <clobber/ast/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

namespace diag {
    namespace parser {
        const std::string ierr_default_general_msg = "An internal error occurred.";
        const std::string ierr_default_msg         = "Could not parse, an internal error occurred.";
    } // namespace parser
}; // namespace diag

clobber::Diagnostic
diag::parser::internal_err(size_t span_start, size_t span_len) {
    using clobber::Diagnostic;
    Diagnostic err(Diagnostic::Stage::Parser, Diagnostic::Severity::Error, span_len, span_len, parser::ierr_default_general_msg,
                   parser::ierr_default_msg);
    return err;
}

clobber::Diagnostic
diag::parser::internal_err(size_t span_start, size_t span_len, const std::string &msg) {
    using clobber::Diagnostic;
    Diagnostic err(Diagnostic::Stage::Parser, Diagnostic::Severity::Error, span_start, span_len, parser::ierr_default_general_msg, msg);
    return err;
}

clobber::Diagnostic
diag::semantics::could_not_infer_type_error(size_t, size_t) {
    throw 0;
}

clobber::Diagnostic
diag::semantics::unresolved_symbol_error(size_t, size_t) {
    throw 0;
}

clobber::Diagnostic
diag::semantics::not_a_fn_error(size_t, size_t) {
    throw 0;
}

clobber::Diagnostic
diag::semantics::argument_type_mismatch_error(size_t, size_t) {
    throw 0;
}

clobber::Diagnostic
diag::semantics::mismatched_arity_error(size_t, size_t, size_t, size_t) {
    throw 0;
}