#include "clobber/pch.hpp"

#include "clobber/internal/diagnostic_factory.hpp"

#include "clobber/common/utils.hpp"

#include "clobber/ast.hpp"
#include "clobber/parser.hpp"
#include "clobber/semantics.hpp"

struct SemanticContext; // forward dec

using InferTypeDelegate = std::shared_ptr<clobber::Type> (*)(SemanticContext &, const clobber::Expr &);

// macro to help defining forward declarations
#define INFER_TYPE_DELEGATE_FN(FN_NAME) std::shared_ptr<clobber::Type> FN_NAME(SemanticContext &, const clobber::Expr &);

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

/* @brief */
using Scope = std::unordered_map<std::string, clobber::Symbol>;

/* @brief */
class SymbolTable {
public:
    SymbolTable() { enter_scope(); }

    void
    enter_scope() {
        scopes.emplace_back();
    }

    void
    exit_scope() {
        if (!scopes.empty()) {
            scopes.pop_back();
        }
    }

    bool
    insert_symbol(const clobber::Symbol &symbol) {
        auto &current = scopes.back();
        return current.emplace(symbol.name, symbol).second;
    }

    std::optional<clobber::Symbol>
    lookup_symbol(const std::string &name) {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) {
                return std::make_optional(found->second);
            }
        }

        return std::nullopt;
    }

private:
    std::vector<Scope> scopes;
};

// hash function for type
namespace std {
    template <> struct hash<clobber::Type> {
        size_t
        operator()(const clobber::Type &type) const {
            size_t h = std::hash<int>()(type.kind);
            /*
            for (const auto &p : type.params)
                h ^= std::hash<int>()(p.kind) + 0x9e3779b9 + (h << 6) + (h >> 2);
            */
            return h;
        }
    };
} // namespace std

// For type interning
/* @brief */
class TypePool {
public:
    std::shared_ptr<clobber::Type>
    intern(const clobber::Type &type) {
        size_t hash = std::hash<clobber::Type>()(type);

        auto it = pool.find(hash);
        if (it != pool.end()) {
            return it->second;
        }

        std::shared_ptr<clobber::Type> ptr = std::make_shared<clobber::Type>(type);
        pool[hash]                         = ptr;
        return ptr;
    }

    // prebuilts + utils
    // TODO: change type to shorthand + bit precision; more descriptive + avoids keyword conflicts

    std::shared_ptr<clobber::Type>
    get_non_composite(clobber::Type::Kind kind) {
        switch (kind) {
        case clobber::Type::Int:
            return i32();
        case clobber::Type::Float:
            return f32();
        case clobber::Type::Double:
            return f64();
        case clobber::Type::String:
            return str();
        case clobber::Type::Char:
            return _char();
        case clobber::Type::Nil:
            return nil();
        default:
            return nullptr;
        }
    }

    std::shared_ptr<clobber::Type>
    i32() {
        clobber::Type type_desc = clobber::Type{};
        type_desc.kind          = clobber::Type::Int;
        type_desc.params.clear();
        return intern(type_desc);
    }

    std::shared_ptr<clobber::Type>
    f32() {
        clobber::Type type_desc = clobber::Type{};
        type_desc.kind          = clobber::Type::Float;
        type_desc.params.clear();
        return intern(type_desc);
    }

    std::shared_ptr<clobber::Type>
    f64() {
        clobber::Type type_desc = clobber::Type{};
        type_desc.kind          = clobber::Type::Double;
        type_desc.params.clear();
        return intern(type_desc);
    }

    std::shared_ptr<clobber::Type>
    str() {
        clobber::Type type_desc = clobber::Type{};
        type_desc.kind          = clobber::Type::String;
        type_desc.params.clear();
        return intern(type_desc);
    }

    std::shared_ptr<clobber::Type>
    _char() {
        clobber::Type type_desc = clobber::Type{};
        type_desc.kind          = clobber::Type::Char;
        type_desc.params.clear();
        return intern(type_desc);
    }

    std::shared_ptr<clobber::Type>
    nil() {
        clobber::Type type_desc = clobber::Type{};
        type_desc.kind          = clobber::Type::Nil;
        type_desc.params.clear();
        return intern(type_desc);
    }

    std::shared_ptr<clobber::Type>
    func(const std::vector<std::shared_ptr<clobber::Type>> &fn_types) {
        clobber::Type type_desc = clobber::Type{};
        type_desc.kind          = clobber::Type::Func;
        type_desc.params        = fn_types;
        return intern(type_desc);
    }

private:
    std::unordered_map<size_t, std::shared_ptr<clobber::Type>> pool;
};

/* @brief */
struct SemanticContext {
    clobber::CompilationUnit &compilation_unit;
    SymbolTable &symbol_table;
    TypePool &type_pool;
    clobber::TypeMap &type_map;
    std::vector<clobber::Diagnostic> &diagnostics;
};

namespace builtins {
    std::unordered_map<std::string, std::function<std::shared_ptr<clobber::Type>(TypePool &)>> name_rtype_map = {
        {"itos", [](TypePool &tp) { return tp.func({tp.i32(), tp.str()}); }},
        {"ftos", [](TypePool &tp) { return tp.func({tp.f32(), tp.str()}); }},
        {"dtos", [](TypePool &tp) { return tp.func({tp.f64(), tp.str()}); }},
        {"println", [](TypePool &tp) { return tp.func({tp.str(), tp.nil()}); }},
        {"print", [](TypePool &tp) { return tp.func({tp.str(), tp.nil()}); }},
    };

    void
    init(TypePool &type_pool, SymbolTable &symbol_table) {
        for (const auto &[fn_name, get_type_callback] : name_rtype_map) {
            std::shared_ptr<clobber::Type> type = get_type_callback(type_pool);
            clobber::Symbol symbol{};
            symbol.name = fn_name;
            symbol.type = type;
            symbol_table.insert_symbol(symbol);
        }
    }
} // namespace builtins

// TODO: remake this:
/*
// 'Could Not Infer Type' error; shorthand for less smelly code solely for the below function.
#define CNIT_ERROR()                                                                                                                       \
    clobber::SemanticError error =                                                                                                         \
        diagnostics::semantics::errors::could_not_infer_type_error(num_lit_expr.token.start, num_lit_expr.token.length);                   \
    context.diagnostics.push_back(error);
*/

std::shared_ptr<clobber::Type>
type_infer_num_literal_expr(SemanticContext &context, const clobber::Expr &expr) {
    const std::string source_text               = context.compilation_unit.source_text;
    const clobber::NumLiteralExpr &num_lit_expr = static_cast<const clobber::NumLiteralExpr &>(expr);

    std::string value_string = source_text.substr(num_lit_expr.token.start, num_lit_expr.token.length);
    std::string value_string_no_postfix =
        (value_string.back() == 'd' || value_string.back() == 'f') ? value_string.substr(0, value_string.size() - 1) : value_string;

    if (value_string.empty()) {
        NOT_IMPLEMENTED(); // TODO: Throw here
        return nullptr;
    }

    clobber::Type type_desc{};
    if (str_utils::try_stoi(value_string).has_value()) { // try parse as an int
        type_desc.kind = clobber::Type::Int;
    } else if (value_string.back() == 'f' && str_utils::try_stof(value_string_no_postfix).has_value()) {
        type_desc.kind = clobber::Type::Float;
    } else if (str_utils::try_stod(value_string_no_postfix).has_value()) {
        type_desc.kind = clobber::Type::Double;
    } else {
        NOT_IMPLEMENTED(); // TODO: Throw here
        return nullptr;
    }

    std::shared_ptr<clobber::Type> type = context.type_pool.intern(type_desc);
    context.type_map.insert({expr.hash(), type});
    return type;
}

std::shared_ptr<clobber::Type>
type_infer_str_literal_expr(SemanticContext &context, const clobber::Expr &expr) {
    clobber::Type type_desc{};
    type_desc.kind = clobber::Type::String;

    std::shared_ptr<clobber::Type> type = context.type_pool.intern(type_desc);
    context.type_map.insert({expr.hash(), type});
    return type;
}

std::shared_ptr<clobber::Type>
type_infer_char_literal_expr(SemanticContext &context, const clobber::Expr &expr) {
    clobber::Type type_desc{};
    type_desc.kind = clobber::Type::Char;

    std::shared_ptr<clobber::Type> type = context.type_pool.intern(type_desc);
    context.type_map.insert({expr.hash(), type});
    return type;
}

std::shared_ptr<clobber::Type>
type_infer_identifier_expr(SemanticContext &context, const clobber::Expr &expr) {
    const clobber::IdentifierExpr &ident_expr       = static_cast<const clobber::IdentifierExpr &>(expr);
    const std::optional<clobber::Symbol> symbol_opt = context.symbol_table.lookup_symbol(ident_expr.name);

    if (!symbol_opt) {
        NOT_IMPLEMENTED(); // TODO: Throw here
        /*
        clobber::SemanticError error =
            diagnostics::semantics::errors::unresolved_symbol_error(ident_expr.token.start, ident_expr.token.length);
        context.diagnostics.push_back(error);
        */
        return nullptr;
    }

    std::shared_ptr<clobber::Type> type = symbol_opt.value().type;
    context.type_map.insert({expr.hash(), type});
    return type;
}

std::shared_ptr<clobber::Type>
type_infer_let_expr(SemanticContext &context, const clobber::Expr &expr) {
    const clobber::LetExpr &let_expr = static_cast<const clobber::LetExpr &>(expr);
    context.symbol_table.enter_scope();

    const clobber::BindingVectorExpr &binding_vector_expr = std::cref(*let_expr.binding_vector_expr);
    for (size_t i = 0; i < binding_vector_expr.num_bindings; i++) {
        const clobber::IdentifierExpr &identifier_expr = std::cref(*binding_vector_expr.identifiers[i]);
        const clobber::Expr &identifier_value_expr     = std::cref(*binding_vector_expr.exprs[i]);

        std::shared_ptr<clobber::Type> identifier_type = type_infer_expr_base(context, identifier_value_expr);
        if (!identifier_type) {
            return nullptr;
        }

        clobber::Symbol symbol{};
        symbol.name = identifier_expr.name;
        symbol.type = identifier_type;
        context.symbol_table.insert_symbol(symbol);
    }

    std::shared_ptr<clobber::Type> last_type;
    auto expr_views = ptr_utils::get_expr_views(let_expr.body_exprs);
    for (const auto &expr_view : expr_views) {
        std::shared_ptr<clobber::Type> body_type = type_infer_expr_base(context, expr_view);
        if (body_type) {
            context.type_map.insert({expr_view.get().hash(), body_type});
            last_type = body_type;
        }
    }

    context.symbol_table.exit_scope();

    if (last_type) {
        context.type_map.insert({let_expr.hash(), last_type});
    }

    return last_type;
}

std::shared_ptr<clobber::Type>
type_infer_fn_expr(SemanticContext &context, const clobber::Expr &expr) {
    NOT_IMPLEMENTED();
}

std::shared_ptr<clobber::Type>
type_infer_def_expr(SemanticContext &context, const clobber::Expr &expr) {
    NOT_IMPLEMENTED();
}

std::shared_ptr<clobber::Type>
type_infer_do_expr(SemanticContext &context, const clobber::Expr &expr) {
    const clobber::DoExpr &do_expr = static_cast<const clobber::DoExpr &>(expr);

    std::shared_ptr<clobber::Type> last_type;
    auto expr_views = ptr_utils::get_expr_views(do_expr.body_exprs);
    for (const auto &expr_view : expr_views) {
        std::shared_ptr<clobber::Type> body_type = type_infer_expr_base(context, expr_view);
        if (body_type) {
            context.type_map.insert({expr_view.get().hash(), body_type});
            last_type = body_type;
        }
    }

    if (last_type) {
        context.type_map.insert({do_expr.hash(), last_type});
    }

    return last_type;
}

namespace call_expr {
    bool
    is_arithmetic_token(const clobber::Token &token) {
        std::unordered_set<clobber::Token::Type> valid_types = {clobber::Token::Type::PlusToken, clobber::Token::Type::MinusToken,
                                                                clobber::Token::Type::AsteriskToken, clobber::Token::Type::SlashToken};
        return valid_types.contains(token.type);
    }

    std::shared_ptr<clobber::Type>
    type_infer_arithmetic_expr(SemanticContext &context, const clobber::CallExpr &ce) {
        bool has_unsupported_arg_type = false;

        std::shared_ptr<clobber::Type> type = context.type_pool.i32(); // i32 by default
        for (const auto &argument : ptr_utils::get_expr_views(ce.arguments)) {
            auto arg_type = type_infer_expr_base(context, argument);
            if (!arg_type) {
                return nullptr;
            }

            if (arg_type->kind != clobber::Type::Int && arg_type->kind != clobber::Type::Float && arg_type->kind != clobber::Type::Double) {
                // TODO: error here
                has_unsupported_arg_type = true; // don't short circuit so we can also do type inference on the rest of the args
            }

            // promote inferred return type
            if (arg_type->kind == clobber::Type::Float) {
                type = context.type_pool.f32();
            } else if (arg_type->kind == clobber::Type::Double) {
                type = context.type_pool.f64();
            }
        }

        return has_unsupported_arg_type ? nullptr : type;
    }

    /* TLDR; check if symbol exists -> symbol points to a function type -> assert that the arguments passed are equal types to the declared
     * parameters -> function's specified return type is the inferred type for the whole expression.
     * If ever we get partial application stuff, it should go here.
     */
    std::shared_ptr<clobber::Type>
    type_infer_fn_call_expr(SemanticContext &context, const clobber::CallExpr &ce, const std::string &fn_name) {
        std::optional<clobber::Symbol> symbol_opt = context.symbol_table.lookup_symbol(fn_name);

        if (!symbol_opt) {
            NOT_IMPLEMENTED(); // TODO: log unresolved symbol err
            return nullptr;
        }

        if (symbol_opt.value().type->kind != clobber::Type::Func) {
            NOT_IMPLEMENTED(); // TODO: log identifier not a function err
            return nullptr;
        }

        clobber::Symbol symbol        = symbol_opt.value();
        auto parameter_types          = symbol.type->params;
        auto expected_return_type     = parameter_types.back();
        auto expected_parameter_types = std::vector<std::shared_ptr<clobber::Type>>(parameter_types.begin(), parameter_types.end() - 1);

        size_t expected_num_arguments = expected_parameter_types.size();
        size_t actual_num_arguments   = ce.arguments.size();

        if (expected_num_arguments != actual_num_arguments) {
            NOT_IMPLEMENTED(); // TODO: log arity mismatch error
            return nullptr;
        }

        bool has_mismatched_argument = false;
        std::vector<std::shared_ptr<clobber::Type>> actual_types;
        auto arg_expr_views = ptr_utils::get_expr_views(ce.arguments);
        for (size_t i = 0; i < arg_expr_views.size(); i++) {
            const clobber::Expr &arg_expr_view           = arg_expr_views[i].get();
            std::shared_ptr<clobber::Type> expected_type = expected_parameter_types[i];
            std::shared_ptr<clobber::Type> arg_type      = type_infer_expr_base(context, arg_expr_view);
            if (arg_type != expected_type) {
                // TODO: YOU NEED TO IMPLEMENT A SPAN METHOD FOR AN EXPR: `expr.get_span()`
                has_mismatched_argument = true; // don't short circuit so we can also do type inference on the rest of the args
            }
        }

        // if above holds true -> argument types match function parameter types -> return the return type of the funcion
        return has_mismatched_argument ? nullptr : expected_return_type;
    }
} // namespace call_expr

std::shared_ptr<clobber::Type>
type_infer_call_expr(SemanticContext &context, const clobber::Expr &expr) {
    using namespace call_expr;

    const clobber::CallExpr &call_expr = static_cast<const clobber::CallExpr &>(expr);

    /*
    if (is_arithmetic_token(call_expr.operator_token)) {
        return type_infer_arithmetic_expr(context, call_expr);
    } else {
        std::string fn_name = context.compilation_unit.source_text.substr(call_expr.operator_token.start, call_expr.operator_token.length);
        return type_infer_fn_call_expr(context, call_expr, fn_name);
    }
    */

    throw 0;
}

std::shared_ptr<clobber::Type>
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

    std::shared_ptr<clobber::Type> inferred_type;
    if (delegate_opt) {
        inferred_type = delegate_opt.value()(context, expr);
    }

    return inferred_type;
}

std::unique_ptr<clobber::SemanticModel>
clobber::get_semantic_model(std::unique_ptr<clobber::CompilationUnit> &&compilation_unit, std::vector<clobber::Diagnostic> &diagnostics) {
    SymbolTable symbol_table{};
    TypePool type_pool{};
    std::unique_ptr<clobber::TypeMap> type_map = std::make_unique<clobber::TypeMap>();

    builtins::init(type_pool, symbol_table);

    // clang-format off
    SemanticContext context{
        .compilation_unit = *compilation_unit, 
        .symbol_table = symbol_table, 
        .type_pool = type_pool, 
        .type_map = *type_map, 
        .diagnostics = diagnostics
    };
    // clang-format on

    auto expr_views = ptr_utils::get_expr_views(compilation_unit->exprs);
    for (const auto &expr_view : expr_views) {
        std::shared_ptr<Type> inferred_type = type_infer_expr_base(context, expr_view);
        if (inferred_type) {
            // context.type_map.insert({expr_view.get().id, inferred_type});
            context.type_map.insert({expr_view.get().hash(), inferred_type});
        }
    }

    // clang-format off
    std::unique_ptr<clobber::SemanticModel> semantic_model = std::make_unique<clobber::SemanticModel>(SemanticModel{
        .compilation_unit = std::move(compilation_unit), 
        .type_map = std::move(type_map), 
    });
    // clang-format on

    return semantic_model;
}