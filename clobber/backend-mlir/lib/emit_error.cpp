#include "clobber/mlir-backend/emit_error.hpp"

EmitError::EmitError() {}
EmitError::EmitError(int span_start, int span_len, const std::string &general_err_msg, const std::string &err_msg) {}
EmitError::~EmitError() {}

std::string
EmitError::GetFormattedErrorMsg(const std::string &file, const std::string &source_text) {
    throw 0;
}