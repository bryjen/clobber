#include "pch.hpp"

#include <clobber/common/utils.hpp>

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include "type_tostring.hpp"

std::string
type_tostring(const clobber::Type &type) {
    std::string repr;
    switch (type.kind) {
    case clobber::Type::Int: {
        repr = "int";
        break;
    }
    case clobber::Type::Float: {
        repr = "float";
        break;
    }
    case clobber::Type::Double: {
        repr = "double";
        break;
    }
    case clobber::Type::String: {
        repr = "string";
        break;
    }
    case clobber::Type::Char: {
        repr = "string";
        break;
    }
    case clobber::Type::Bool: {
        repr = "bool";
        break;
    }
    case clobber::Type::Func: {
        std::vector<std::string> type_strs;
        /*
        for (const auto &type_param : type.params) {
        }
        */
        repr = std::format("({})", str_utils::join(" -> ", type_strs));
        break;
    }
    }

    return repr;
}