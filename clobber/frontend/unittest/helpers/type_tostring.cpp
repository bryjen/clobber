#include "pch.hpp"

#include <clobber/common/utils.hpp>

#include <clobber/semantics.hpp>

#include "type_tostring.hpp"

std::string
type_tostring(const Type &type) {
    std::string repr;
    switch (type.kind) {
    case Type::Int: {
        repr = "int";
        break;
    }
    case Type::Float: {
        repr = "float";
        break;
    }
    case Type::Double: {
        repr = "double";
        break;
    }
    case Type::String: {
        repr = "string";
        break;
    }
    case Type::Char: {
        repr = "string";
        break;
    }
    case Type::Bool: {
        repr = "bool";
        break;
    }
    case Type::Func: {
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