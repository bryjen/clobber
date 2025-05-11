#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include "clobber/internal/diagnostic_factory.hpp"

SemanticError
diagnostics::semantics::errors::could_not_infer_type_error(int span_start, int span_len) {
    throw 0;
}

SemanticError
diagnostics::semantics::errors::unresolved_symbol_error(int span_start, int span_len) {
    throw 0;
}

SemanticError
diagnostics::semantics::errors::not_a_fn_error(int span_start, int span_len) {
    throw 0;
}

SemanticError
diagnostics::semantics::errors::argument_type_mismatch_error(int span_start, int span_len) {
    throw 0;
}

SemanticError
diagnostics::semantics::errors::mismatched_arity_error(int span_start, int span_len, size_t expected, size_t actual) {
    throw 0;
}