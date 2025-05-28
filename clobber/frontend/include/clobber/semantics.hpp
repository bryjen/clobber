#pragma once
#include <variant>

#include <clobber/common/debug.hpp> // common debug header

#include "pch.hpp"

namespace clobber {
    struct ExprBase;
    struct CompilationUnit;

    struct Diagnostic;
}; // namespace clobber

namespace clobber {
    struct TypeData;                        // opaque payload
    using Type = std::shared_ptr<TypeData>; // canonical handle

    struct NilType {
        auto operator<=>(const NilType &) const = default;
    };

    struct PrimitiveType {
        enum class Kind {
            I8,
            I16,
            I32,
            I64,
            F32,
            F64,
            Bool,
            Str, // TODO: Eval whether strings should be their own distinct type variant (and not a primitive)
            Char
        } kind;

        auto operator<=>(const PrimitiveType &) const = default;
    };

    struct VectorType {
        Type element;

        auto operator<=>(const VectorType &) const = default;
    };

    struct TensorType {
        Type element;
        std::vector<size_t> shape;

        auto operator<=>(const TensorType &) const = default;
    };

    struct FunctionType {
        std::vector<Type> params;
        Type ret;

        auto operator<=>(const FunctionType &) const = default;
    };

    struct UserDefinedType {
        std::string name;

        auto operator<=>(const UserDefinedType &) const = default;
    };

    // clang-format off
    /* @brief
     */
    using Variant = std::variant<
        NilType, 
        PrimitiveType, 
        VectorType, 
        TensorType, 
        FunctionType, 
        UserDefinedType
    >;
    // clang-format oon

    /* @brief
     */
    struct TypeData {
        Variant v;
    };

    /* @brief
     */
    struct Symbol {
        std::string name;
        Type type;
    };


    using TypeMap = std::unordered_map<size_t, Type>;

    /* @brief */
    struct SemanticModel {
        std::unique_ptr<clobber::CompilationUnit> compilation_unit;
        TypeMap type_map;
    };

    /* @brief */
    std::unique_ptr<SemanticModel> get_semantic_model(std::unique_ptr<clobber::CompilationUnit> &&compilation_unit,
                                                      std::vector<clobber::Diagnostic> &);
}; // namespace clobber