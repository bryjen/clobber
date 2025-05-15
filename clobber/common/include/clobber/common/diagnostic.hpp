#pragma once

#include "pch.hpp"

namespace clobber {
    struct Diagnostic {
        enum class Stage {
            Tokenizer,
            Parser,
            Semantics,
            Emitter
        };

        enum class Severity {
            Info,
            Warning,
            Error
        };

        Diagnostic(Diagnostic::Stage, Diagnostic::Severity, size_t span_start, size_t span_len, const std::string &general_msg,
                   const std::string &msg);

        std::string GetFormattedErrorMsg(const std::string &file, const std::string &source_text);

        Diagnostic::Stage stage;
        Diagnostic::Severity severity;
        size_t span_start;
        size_t span_len;

        std::string general_msg; // a message indicating the type of the error, but not necessarily error specific info (ex. 'Invalid
                                 // numeric expression', but not what exactly wrong with it)
        std::string msg;
    };
}; // namespace clobber