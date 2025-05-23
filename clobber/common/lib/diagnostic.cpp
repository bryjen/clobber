#include <format>

#include "clobber/common/pch.hpp"

#include "clobber/common/diagnostic.hpp"

constexpr int padding = 1;

void
get_line_and_col(const std::string &source_text, size_t idx, size_t &out_line, size_t &out_col) {
    std::istringstream stream(source_text);
    std::string line;
    size_t remaining_chars = idx;
    size_t line_count      = 1;

    while (std::getline(stream, line)) {
        size_t len = line.length();
        if (len > remaining_chars) {
            break;
        }

        remaining_chars -= len;
        line_count++;
    }

    out_line = line_count;
    out_col  = remaining_chars;
}

std::string
get_line(const std::string &src, size_t line_num) {
    std::istringstream stream(src);
    std::string line;
    for (size_t i = 0; i <= line_num; ++i) {
        std::getline(stream, line);
    }
    return line;
}

std::string
repeat(const std::string &s, size_t n) {
    std::string result;
    result.reserve(s.size() * n);
    for (size_t i = 0; i < n; ++i) {
        result += s;
    }
    return result;
}

std::string
spaces(size_t count) {
    return repeat(" ", count);
}

std::string
base_format(const std::string &file, const std::string &source_text, size_t span_start, size_t span_length,
            const std::string &general_error_msg, const std::string &error_msg) {
    size_t line;
    size_t col;
    size_t margin;
    size_t line_num_chars; // the number of characters of the line number itself

    std::vector<std::string> lines;
    std::ostringstream out;

    get_line_and_col(source_text, span_start, line, col);
    line_num_chars = std::to_string(line).length();
    margin         = line_num_chars + 1;

    // clang-format off
    lines.push_back(std::format("Error: {}", general_error_msg));
    lines.push_back(std::format("{}┌─ {}:{}:{}", spaces(margin), file, line, col));
    lines.push_back(std::format("{}|", spaces(margin)));
    lines.push_back(std::format("{}{}|{}{}", line, spaces(margin - line_num_chars), spaces(padding), get_line(source_text, line - 1)));
    lines.push_back(std::format("{}|{}{}{} {}", spaces(margin), spaces(padding), spaces(col == 0 ? col : col - 1), repeat("^", span_length), error_msg));
    // clang-format on

    for (const auto &s : lines) {
        out << s << '\n';
    }
    return out.str();
}

clobber::Diagnostic::Diagnostic(Diagnostic::Stage stage, Diagnostic::Severity severity, size_t span_start, size_t span_len,
                                const std::string &general_msg, const std::string &msg)
    : stage(stage)
    , severity(severity)
    , span_start(span_start)
    , span_len(span_len)
    , general_msg(general_msg)
    , msg(msg) {}

std::string
clobber::Diagnostic::GetFormattedErrorMsg(const std::string &file, const std::string &source_text) {
    return base_format(file, source_text, this->span_start, this->span_len, this->general_msg, this->msg);
}