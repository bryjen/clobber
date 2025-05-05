#include "clobber/internal/parser_error_factory.hpp"
#include "clobber/parser.hpp"

namespace InternalErr {
const std::string default_general_err_msg = "An internal error occurred.";
const std::string default_err_msg         = "An unexpected error occurred.";
} // namespace InternalErr

ParserError
ParserErrorFactory::InternalErr(int span_start, int span_len) {
    return ParserError(span_start, span_len, InternalErr::default_general_err_msg, InternalErr::default_err_msg);
}
