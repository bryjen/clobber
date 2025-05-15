#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include "clobber/internal/diagnostic_factory.hpp"

clobber::SemanticError
diagnostics::semantics::errors::could_not_infer_type_error(size_t, size_t) {
    throw 0;
}

clobber::SemanticError
diagnostics::semantics::errors::unresolved_symbol_error(size_t, size_t) {
    throw 0;
}

clobber::SemanticError
diagnostics::semantics::errors::not_a_fn_error(size_t, size_t) {
    throw 0;
}

clobber::SemanticError
diagnostics::semantics::errors::argument_type_mismatch_error(size_t, size_t) {
    throw 0;
}

clobber::SemanticError
diagnostics::semantics::errors::mismatched_arity_error(size_t, size_t, size_t, size_t) {
    throw 0;
}