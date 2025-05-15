#include "clobber/internal/diagnostic_factory.hpp"

#include <clobber/common/diagnostic.hpp>

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

clobber::Diagnostic
diagnostics::semantics::errors::could_not_infer_type_error(size_t, size_t) {
    throw 0;
}

clobber::Diagnostic
diagnostics::semantics::errors::unresolved_symbol_error(size_t, size_t) {
    throw 0;
}

clobber::Diagnostic
diagnostics::semantics::errors::not_a_fn_error(size_t, size_t) {
    throw 0;
}

clobber::Diagnostic
diagnostics::semantics::errors::argument_type_mismatch_error(size_t, size_t) {
    throw 0;
}

clobber::Diagnostic
diagnostics::semantics::errors::mismatched_arity_error(size_t, size_t, size_t, size_t) {
    throw 0;
}