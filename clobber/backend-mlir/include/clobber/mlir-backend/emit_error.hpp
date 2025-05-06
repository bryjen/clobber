#ifndef EMIT_ERROR_HPP
#define EMIT_ERROR_HPP

#include <string>

struct EmitError {
public:
    EmitError();
    EmitError(int span_start, int span_len, const std::string &general_err_msg, const std::string &err_msg);
    ~EmitError();

    std::string GetFormattedErrorMsg(const std::string &file, const std::string &source_text);

protected:
    int span_start;
    int span_len;
    std::string general_err_msg;
    std::string err_msg;
};

#endif // EMIT_ERROR_HPP