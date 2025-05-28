
#include <clobber/common/utils.hpp>
#include <clobber/common/variant.hpp>

#include "../clobber/internal/diagnostic_factory.hpp"
#include "clobber/pch.hpp"

#include "clobber/ast/ast.hpp"
#include "clobber/parser.hpp"
#include "clobber/semantics.hpp"

#include "semantics_impl.hpp"
#include "semantics_types.hpp"

struct SemanticContext; // forward dec

using InferTypeDelegate = clobber::Type (*)(SemanticContext &, const clobber::Expr &);

// macro to help defining forward declarations
#define INFER_TYPE_DELEGATE_FN(FN_NAME) clobber::Type FN_NAME(SemanticContext &, const clobber::Expr &);

INFER_TYPE_DELEGATE_FN(type_infer_expr_base)

INFER_TYPE_DELEGATE_FN(type_infer_num_literal_expr)
INFER_TYPE_DELEGATE_FN(type_infer_str_literal_expr)
INFER_TYPE_DELEGATE_FN(type_infer_char_literal_expr)

INFER_TYPE_DELEGATE_FN(type_infer_identifier_expr)
INFER_TYPE_DELEGATE_FN(type_infer_let_expr)
INFER_TYPE_DELEGATE_FN(type_infer_fn_expr)
INFER_TYPE_DELEGATE_FN(type_infer_def_expr)
INFER_TYPE_DELEGATE_FN(type_infer_do_expr)
INFER_TYPE_DELEGATE_FN(type_infer_call_expr)

namespace parser_to_semantic_types {
    clobber::Type
    convert_builtin_type_expr(TypePool &tp, const clobber::BuiltinTypeExpr &bte) {
        // clang-format off
        std::unordered_map<clobber::Token::Type, std::function<clobber::Type()>> builtin_type_map = {
            {clobber::Token::Type::CharKeywordToken,   [&tp]() { return tp._char(); }},
            {clobber::Token::Type::StringKeywordToken, [&tp]() { return tp.str(); }},
            {clobber::Token::Type::I8KeywordToken,     [&tp]() { return tp.i8(); }},
            {clobber::Token::Type::I16KeywordToken,    [&tp]() { return tp.i16(); }},
            {clobber::Token::Type::I32KeywordToken,    [&tp]() { return tp.i32(); }},
            {clobber::Token::Type::I64KeywordToken,    [&tp]() { return tp.i64(); }},
            {clobber::Token::Type::F32KeywordToken,    [&tp]() { return tp.f32(); }},
            {clobber::Token::Type::F64KeywordToken,    [&tp]() { return tp.f64(); }},
        };
        // clang-format on

        auto it = builtin_type_map.find(bte.type_keyword_token.type);
        return it != builtin_type_map.end() ? it->second() : nullptr;
    }

    clobber::Type
    convert_user_defined_type_expr(TypePool &tp, const clobber::UserDefinedTypeExpr &udte) {
        // return tp.intern<clobber::UserDefinedType>();
        NOT_IMPLEMENTED();
    }

    clobber::Type
    convert_parameterized_tensor_type_expr(TypePool &tp, const clobber::ParameterizedTypeExpr &pte) {
        NOT_IMPLEMENTED();
    }

    clobber::Type
    convert_type_expr(TypePool &tp, const clobber::TypeExpr &type_expr) {
        switch (type_expr.type_kind) {
        case clobber::TypeExpr::Type::BuiltinType: {
            const clobber::BuiltinTypeExpr &bte = static_cast<const clobber::BuiltinTypeExpr &>(type_expr);
            return convert_builtin_type_expr(tp, bte);
        }
        case clobber::TypeExpr::Type::UserDefinedType: {
            return nullptr;
        }
        case clobber::TypeExpr::Type::ParameterizedType: {
            return nullptr;
        }
        default: {
            return nullptr;
        }
        }
    }
}; // namespace parser_to_semantic_types

using namespace parser_to_semantic_types;

namespace builtins {
    std::unordered_map<std::string, std::function<clobber::Type(TypePool &)>> name_rtype_map = {
        {"itos", [](TypePool &tp) { return tp.func({tp.i32()}, tp.str()); }},
        {"ftos", [](TypePool &tp) { return tp.func({tp.f32()}, tp.str()); }},
        {"dtos", [](TypePool &tp) { return tp.func({tp.f64()}, tp.str()); }},
        {"println", [](TypePool &tp) { return tp.func({tp.str()}, tp.nil()); }},
        {"print", [](TypePool &tp) { return tp.func({tp.str()}, tp.nil()); }},
    };

    void
    init(TypePool &type_pool, SymbolTable &symbol_table) {
        for (const auto &[fn_name, get_type_callback] : name_rtype_map) {
            clobber::Type type = get_type_callback(type_pool);
            clobber::Symbol symbol{};
            symbol.name = fn_name;
            symbol.type = type;
            symbol_table.insert_symbol(symbol);
        }
    }
}; // namespace builtins

/* @brief */
struct SemanticContext {
    clobber::CompilationUnit &compilation_unit;
    SymbolTable &symbol_table;
    TypePool &type_pool;
    clobber::TypeMap &type_map;
    std::vector<clobber::Diagnostic> &diagnostics;
};

clobber::Type
type_infer_num_literal_expr(SemanticContext &ctx, const clobber::Expr &expr) {
    const std::string source_text               = ctx.compilation_unit.source_text;
    const clobber::NumLiteralExpr &num_lit_expr = static_cast<const clobber::NumLiteralExpr &>(expr);

    std::string value_string = source_text.substr(num_lit_expr.token.span.start, num_lit_expr.token.span.length);
    std::string value_string_no_postfix =
        (value_string.back() == 'd' || value_string.back() == 'f') ? value_string.substr(0, value_string.size() - 1) : value_string;

    if (value_string.empty()) {
        NOT_IMPLEMENTED(); // TODO: Throw here
        return nullptr;
    }

    clobber::Type type = nullptr;
    if (str_utils::try_stoi(value_string).has_value()) { // try parse as an int
        type = ctx.type_pool.i32();
    } else if (value_string.back() == 'f' && str_utils::try_stof(value_string_no_postfix).has_value()) {
        type = ctx.type_pool.f32();
    } else if (str_utils::try_stod(value_string_no_postfix).has_value()) {
        type = ctx.type_pool.f64();
    } else {
        NOT_IMPLEMENTED(); // TODO: Throw here
        return nullptr;
    }

    ctx.type_map.insert({expr.hash(), type});
    return type;
}

clobber::Type
type_infer_str_literal_expr(SemanticContext &context, const clobber::Expr &expr) {
    clobber::Type type = context.type_pool.str();
    context.type_map.insert({expr.hash(), type});
    return type;
}

clobber::Type
type_infer_char_literal_expr(SemanticContext &context, const clobber::Expr &expr) {
    clobber::Type type = context.type_pool._char();
    context.type_map.insert({expr.hash(), type});
    return type;
}

clobber::Type
type_infer_identifier_expr(SemanticContext &context, const clobber::Expr &expr) {
    const clobber::IdentifierExpr &ident_expr       = static_cast<const clobber::IdentifierExpr &>(expr);
    const std::optional<clobber::Symbol> symbol_opt = context.symbol_table.lookup_symbol(ident_expr.name);

    if (!symbol_opt) {
        NOT_IMPLEMENTED(); // TODO: log unresolved symbol error
        return nullptr;
    }

    clobber::Type type = symbol_opt.value().type;
    context.type_map.insert({expr.hash(), type});
    return type;
}

clobber::Type
type_infer_let_expr(SemanticContext &context, const clobber::Expr &expr) {
    const clobber::LetExpr &let_expr = static_cast<const clobber::LetExpr &>(expr);
    context.symbol_table.enter_scope();

    for (const auto &binding : let_expr.binding_vector_expr->bindings) {
        clobber::Type value_type = type_infer_expr_base(context, *binding->value);
        if (!value_type) {
            return nullptr;
        }

        // TODO: assert value's inferred type matches the type annotation

        clobber::Symbol symbol{};
        symbol.name = binding->identifier->name;
        symbol.type = value_type;
        context.symbol_table.insert_symbol(symbol);
    }

    clobber::Type last_type;
    for (const auto &body_expr : let_expr.body_exprs) {
        clobber::Type body_type = type_infer_expr_base(context, *body_expr);
        if (body_type) {
            context.type_map.insert({body_expr->hash(), body_type});
            last_type = body_type;
        }
    }

    context.symbol_table.exit_scope();

    if (last_type) {
        context.type_map.insert({let_expr.hash(), last_type});
    }

    return last_type;
}

clobber::Type
type_infer_fn_expr(SemanticContext &context, const clobber::Expr &expr) {
    const clobber::FnExpr &fn_expr = static_cast<const clobber::FnExpr &>(expr);
    context.symbol_table.enter_scope();

    std::vector<clobber::Type> param_types;
    for (const auto &parameter : fn_expr.parameter_vector_expr->parameters) {
        clobber::Type as_semantic_type = convert_type_expr(context.type_pool, *parameter->type_annot);

        clobber::Symbol symbol{};
        symbol.name = parameter->identifier->name;
        symbol.type = as_semantic_type;
        context.symbol_table.insert_symbol(symbol);

        param_types.push_back(as_semantic_type);
    }

    clobber::Type last_type;
    for (const auto &body_expr : fn_expr.body_exprs) {
        clobber::Type body_type = type_infer_expr_base(context, *body_expr);
        if (body_type) {
            context.type_map.insert({body_expr->hash(), body_type});
            last_type = body_type;
        }
    }

    context.symbol_table.exit_scope();

    clobber::Type fn_type = context.type_pool.func(std::move(param_types), last_type);
    context.type_map.insert({fn_expr.hash(), fn_type});
    return fn_type;
}

clobber::Type
type_infer_def_expr(SemanticContext &ctx, const clobber::Expr &expr) {
    const clobber::DefExpr &def_expr = static_cast<const clobber::DefExpr &>(expr);

    // infer the type of the given value
    auto value_type = type_infer_expr_base(ctx, *def_expr.value);
    if (!value_type) {
        return nullptr;
    }

    // add to symbol table
    clobber::Symbol symbol{};
    symbol.name = def_expr.identifier->name;
    symbol.type = value_type;
    ctx.symbol_table.insert_symbol(symbol);

    // insert into typemap
    ctx.type_map.insert({def_expr.hash(), value_type});
    return value_type;
}

clobber::Type
type_infer_do_expr(SemanticContext &ctx, const clobber::Expr &expr) {
    const clobber::DoExpr &do_expr = static_cast<const clobber::DoExpr &>(expr);

    clobber::Type last_type = nullptr;
    for (const auto &body_expr : do_expr.body_exprs) {
        clobber::Type body_type = type_infer_expr_base(ctx, *body_expr);
        if (body_type) {
            ctx.type_map.insert({body_expr->hash(), body_type});
            last_type = body_type;
        }
    }

    if (last_type) {
        ctx.type_map.insert({do_expr.hash(), last_type});
    }

    return last_type;
}

namespace call_expr {
    bool
    is_arithmetic_token(const clobber::Token &token) {
        // clang-format off
        std::unordered_set<clobber::Token::Type> valid_types = {
            clobber::Token::Type::PlusToken, 
            clobber::Token::Type::MinusToken,
            clobber::Token::Type::AsteriskToken, 
            clobber::Token::Type::SlashToken
        };
        // clang-format on
        return valid_types.contains(token.type);
    }

    /* @brief  */
    std::pair<clobber::Type, int>
    get_primitive_type_and_priority(TypePool &tp, clobber::PrimitiveType::Kind primitive_kind) {
        std::vector<std::pair<clobber::PrimitiveType::Kind, std::function<clobber::Type()>>> builtin_type_map = {
            {clobber::PrimitiveType::Kind::I8, [&tp]() { return tp.i8(); }},
            {clobber::PrimitiveType::Kind::I16, [&tp]() { return tp.i16(); }},
            {clobber::PrimitiveType::Kind::I32, [&tp]() { return tp.i32(); }},
            {clobber::PrimitiveType::Kind::I64, [&tp]() { return tp.i64(); }},
            {clobber::PrimitiveType::Kind::F32, [&tp]() { return tp.f32(); }},
            {clobber::PrimitiveType::Kind::F64, [&tp]() { return tp.f64(); }},
        };

        for (size_t i = 0; i < builtin_type_map.size(); i++) {
            if (primitive_kind == builtin_type_map[i].first) {
                return {builtin_type_map[i].second(), static_cast<int>(i)};
            }
        }

        return {nullptr, -1};
    }

    clobber::Type
    type_infer_arithmetic_expr(SemanticContext &context, const clobber::CallExpr &ce) {
        bool has_unsupported_arg_type = false;

        int current_priority   = 0;
        clobber::Type ret_type = context.type_pool.i8(); // 1i8 (lowest) by default

        for (const auto &argument : ptr_utils::get_expr_views(ce.arguments)) {
            clobber::Type arg_type = type_infer_expr_base(context, argument);
            if (!arg_type) {
                return nullptr;
            }

            clobber::PrimitiveType *p = std::get_if<clobber::PrimitiveType>(&(arg_type->v));
            if (!p) {
                has_unsupported_arg_type = true; // don't short circuit so we can also do type inference on the rest of the args
            }

            auto [type, priority] = get_primitive_type_and_priority(context.type_pool, p->kind);
            if (priority < 0 || !type) {
                // TODO: error here
                has_unsupported_arg_type = true;
            }

            if (priority > current_priority) {
                current_priority = priority;
                ret_type         = type;
            }
        }

        return has_unsupported_arg_type ? nullptr : ret_type;
    }

    /* TLDR; check if symbol exists -> symbol points to a function type -> assert that the arguments passed are equal types to the declared
     * parameters -> function's specified return type is the inferred type for the whole expression.
     * If ever we get partial application stuff, it should go here.
     */
    clobber::Type
    type_infer_identifier_call_expr(SemanticContext &context, const clobber::CallExpr &ce, const std::string &fn_name) {
        std::optional<clobber::Symbol> symbol_opt = context.symbol_table.lookup_symbol(fn_name);

        if (!symbol_opt) {
            NOT_IMPLEMENTED(); // TODO: log unresolved symbol err
            return nullptr;
        }

        auto fn_type_ptr = std::get_if<clobber::FunctionType>(&(symbol_opt.value().type->v));
        if (!fn_type_ptr) {
            NOT_IMPLEMENTED(); // TODO: log identifier not a function err
            return nullptr;
        }

        clobber::Type expected_return_type                  = fn_type_ptr->ret;
        std::vector<clobber::Type> expected_parameter_types = fn_type_ptr->params;

        size_t expected_num_arguments = expected_parameter_types.size();
        size_t actual_num_arguments   = ce.arguments.size();

        if (expected_num_arguments != actual_num_arguments) {
            NOT_IMPLEMENTED(); // TODO: log arity mismatch error
            return nullptr;
        }

        bool has_mismatched_argument = false;
        std::vector<std::shared_ptr<clobber::Type>> actual_types;

        for (size_t i = 0; i < ce.arguments.size(); i++) {
            clobber::Type expected_type = expected_parameter_types[i];
            clobber::Type arg_type      = type_infer_expr_base(context, *ce.arguments[i]);

            if (arg_type != expected_type) {
                // TODO: YOU NEED TO IMPLEMENT A SPAN METHOD FOR AN EXPR: `expr.get_span()`
                has_mismatched_argument = true; // don't short circuit so we can also do type inference on the rest of the args
            }
        }

        // if above holds true -> argument types match function parameter types -> return the return type of the function
        return has_mismatched_argument ? nullptr : expected_return_type;
    }

    clobber::Type
    type_infer_anonymous_call_expr(SemanticContext &context, const clobber::CallExpr &ce) {
        auto operator_type = type_infer_expr_base(context, *ce.operator_expr);

        if (!operator_type) {
            return nullptr;
        }

        auto fn_type_ptr = std::get_if<clobber::FunctionType>(&(operator_type->v));
        if (!fn_type_ptr) {
            NOT_IMPLEMENTED(); // TODO: log identifier not a function err
            return nullptr;
        }

        clobber::Type expected_return_type                         = fn_type_ptr->ret;
        const std::vector<clobber::Type> &expected_parameter_types = fn_type_ptr->params;

        size_t expected_num_arguments = expected_parameter_types.size();
        size_t actual_num_arguments   = ce.arguments.size();

        if (expected_num_arguments != actual_num_arguments) {
            NOT_IMPLEMENTED(); // TODO: log arity mismatch error
            return nullptr;
        }

        bool has_mismatched_argument = false;
        std::vector<std::shared_ptr<clobber::Type>> actual_types;

        for (size_t i = 0; i < ce.arguments.size(); i++) {
            clobber::Type expected_type = expected_parameter_types[i];
            clobber::Type arg_type      = type_infer_expr_base(context, *ce.arguments[i]);

            if (arg_type != expected_type) {
                // TODO: YOU NEED TO IMPLEMENT A SPAN METHOD FOR AN EXPR: `expr.get_span()`
                has_mismatched_argument = true; // don't short circuit so we can also do type inference on the rest of the args
            }
        }

        // if above holds true -> argument types match function parameter types -> return the return type of the function
        return has_mismatched_argument ? nullptr : expected_return_type;
    }
} // namespace call_expr

clobber::Type
type_infer_call_expr(SemanticContext &context, const clobber::Expr &expr) {
    using namespace call_expr;

    const clobber::CallExpr &call_expr = static_cast<const clobber::CallExpr &>(expr);

    if (call_expr.operator_expr->type == clobber::Expr::Type::IdentifierExpr) {
        auto ie = dynamic_cast<clobber::IdentifierExpr *>(call_expr.operator_expr.get());
        if (is_arithmetic_token(ie->token)) {
            return type_infer_arithmetic_expr(context, call_expr);
        } else {
            return type_infer_identifier_call_expr(context, call_expr, ie->name);
        }
    }

    return type_infer_anonymous_call_expr(context, call_expr);
}

clobber::Type
type_infer_expr_base(SemanticContext &context, const clobber::Expr &expr) {
    // clang-format off
    const std::unordered_map<clobber::Expr::Type, InferTypeDelegate> delegate_map = {
        { clobber::Expr::Type::NumericLiteralExpr, type_infer_num_literal_expr },
        { clobber::Expr::Type::StringLiteralExpr, type_infer_str_literal_expr },
        { clobber::Expr::Type::CharLiteralExpr, type_infer_char_literal_expr },
        { clobber::Expr::Type::IdentifierExpr, type_infer_identifier_expr },
        { clobber::Expr::Type::LetExpr, type_infer_let_expr },
        { clobber::Expr::Type::FnExpr, type_infer_fn_expr },
        { clobber::Expr::Type::DefExpr, type_infer_def_expr },
        { clobber::Expr::Type::DoExpr, type_infer_do_expr },
        { clobber::Expr::Type::CallExpr, type_infer_call_expr },
    };
    // clang-format on

    auto it                                       = delegate_map.find(expr.type);
    std::optional<InferTypeDelegate> delegate_opt = it != delegate_map.end() ? std::make_optional(it->second) : std::nullopt;

    clobber::Type inferred_type;
    if (delegate_opt) {
        inferred_type = delegate_opt.value()(context, expr);
    }

    return inferred_type;
}

std::unique_ptr<clobber::SemanticModel>
clobber::get_semantic_model(std::unique_ptr<clobber::CompilationUnit> &&compilation_unit, std::vector<clobber::Diagnostic> &diagnostics) {
    SymbolTable symbol_table{};
    TypePool type_pool{};
    clobber::TypeMap type_map{};

    builtins::init(type_pool, symbol_table);

    // clang-format off
    SemanticContext context{
        .compilation_unit = *compilation_unit, 
        .symbol_table = symbol_table, 
        .type_pool = type_pool, 
        .type_map = type_map, 
        .diagnostics = diagnostics
    };
    // clang-format on

    for (const auto &expr_view : compilation_unit->exprs) {
        clobber::Type inferred_type = type_infer_expr_base(context, *expr_view);
        if (inferred_type) {
            context.type_map.insert({expr_view->hash(), inferred_type});
        }
    }

    // clang-format off
    std::unique_ptr<clobber::SemanticModel> semantic_model = std::make_unique<clobber::SemanticModel>(SemanticModel{
        .compilation_unit = std::move(compilation_unit), 
        .type_map = std::move(context.type_map), 
    });
    // clang-format on

    return semantic_model;
}