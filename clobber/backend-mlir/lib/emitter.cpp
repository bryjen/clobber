
#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include <clobber/common/utils.hpp>

#include "clobber/mlir-backend/emitter.hpp"

struct EmitterContext {
    const SemanticModel &semantic_model;
    mlir::OpBuilder &builder;
    std::vector<EmitError> errors;
};

template <typename T> using Option = std::optional<T>;

// using raw pointer types because ownership is handled by mlir
using EmitDelegate = mlir::Operation *(*)(EmitterContext &, const ExprBase &);
#define EMIT_DELEGATE_FN(FN_NAME) mlir::Operation *FN_NAME(EmitterContext &, const ExprBase &);

EMIT_DELEGATE_FN(emit_expr_base);
EMIT_DELEGATE_FN(emit_num_literal_expr);
EMIT_DELEGATE_FN(emit_string_literal_expr);
EMIT_DELEGATE_FN(emit_char_literal_expr);
EMIT_DELEGATE_FN(emit_identifier_expr);
EMIT_DELEGATE_FN(emit_let_expr);
EMIT_DELEGATE_FN(emit_fn_expr);
EMIT_DELEGATE_FN(emit_def_expr);
EMIT_DELEGATE_FN(emit_do_expr);
EMIT_DELEGATE_FN(emit_call_expr);

mlir::Operation *
emit_expr_base(EmitterContext &emitter_context, const ExprBase &expr_base) {
    // clang-format off
    std::unordered_map<ClobberExprType, EmitDelegate> delegates = {
        {ClobberExprType::NumericLiteralExpr, emit_num_literal_expr},
        {ClobberExprType::StringLiteralExpr, emit_string_literal_expr},
        {ClobberExprType::CharLiteralExpr, emit_char_literal_expr},
        {ClobberExprType::IdentifierExpr, emit_identifier_expr},
        {ClobberExprType::LetExpr, emit_let_expr},
        {ClobberExprType::FnExpr, emit_fn_expr},
        {ClobberExprType::DefExpr, emit_def_expr},
        {ClobberExprType::DoExpr, emit_do_expr},
        {ClobberExprType::CallExpr, emit_call_expr},
    };
    // clang-format on

    auto it = delegates.find(expr_base.expr_type);
    if (it == delegates.end()) {
        // TODO: throw error here
        return nullptr;
    }

    return it->second(emitter_context, expr_base);
}

mlir::Operation *
emit_num_literal_expr(EmitterContext &emitter_context, const ExprBase &expr_base) {
    mlir::OpBuilder &builder       = emitter_context.builder;
    const NumLiteralExpr &nle      = static_cast<const NumLiteralExpr &>(expr_base);
    const std::string &source_text = emitter_context.semantic_model.compilation_unit->source_text;

    const TypeMap &type_map = std::cref(*emitter_context.semantic_model.type_map);
    auto it                 = type_map.find(nle.id);

    if (it == type_map.end()) {
        // TODO: throw error here
        return nullptr;
    }

    mlir::arith::ConstantOp const_op;
    std::string value_str = source_text.substr(nle.token.start, nle.token.length);
    switch (it->second->kind) {
    case Type::Int: {
        try {
            mlir::IntegerType i32 = builder.getI32Type();
            int value             = std::stoi(value_str);
            const_op              = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerAttr(i32, value));
            break;
        } catch (...) {
            // TODO: throw error here
            return nullptr;
        }
    }
    case Type::Float: {
        try {
            mlir::FloatType f32 = builder.getF32Type();
            float value         = std::stof(value_str);
            const_op            = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerAttr(f32, value));
            break;
        } catch (...) {
            // TODO: throw error here
            return nullptr;
        }
    }
    case Type::Double: {
        try {
            mlir::FloatType f64 = builder.getF64Type();
            double value        = std::stod(value_str);
            const_op            = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerAttr(f64, value));
            break;
        } catch (...) {
            // TODO: throw error here
            return nullptr;
        }
    }
    default: {
        // TODO: throw error here
        return nullptr;
    }
    };

    return const_op;
}

mlir::Operation *
emit_string_literal_expr(EmitterContext &emitter_context, const ExprBase &expr_base) {
    throw 0;
}

mlir::Operation *
emit_char_literal_expr(EmitterContext &emitter_context, const ExprBase &expr_base) {
    mlir::OpBuilder &builder   = emitter_context.builder;
    const CharLiteralExpr &cle = static_cast<const CharLiteralExpr &>(expr_base);

    char value = cle.value[0]; // asserted to exist through semantic analysis

    mlir::IntegerType i8       = builder.getIntegerType(8); // chars are just 8 bit integers
    mlir::arith::ConstantOp ch = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerAttr(i8, value));
    return ch;
}

mlir::Operation *
emit_identifier_expr(EmitterContext &emitter_context, const ExprBase &expr_base) {
    throw 0;
}

mlir::Operation *
emit_let_expr(EmitterContext &emitter_context, const ExprBase &expr_base) {
    throw 0;
}

mlir::Operation *
emit_fn_expr(EmitterContext &emitter_context, const ExprBase &expr_base) {
    throw 0;
}

mlir::Operation *
emit_def_expr(EmitterContext &emitter_context, const ExprBase &expr_base) {
    throw 0;
}

mlir::Operation *
emit_do_expr(EmitterContext &emitter_context, const ExprBase &expr_base) {
    throw 0;
}

namespace utils {
    Option<mlir::Type>
    to_mlir_type(mlir::OpBuilder &builder, const Type &type) {
        switch (type.kind) {
        case Type::Char: {
            return std::make_optional(builder.getI8Type());
        }
        case Type::Double: {
            return std::make_optional(builder.getF64Type());
        }
        case Type::Float: {
            return std::make_optional(builder.getF32Type());
        }
        case Type::Int: {
            return std::make_optional(builder.getI32Type());
        }
            /* unsupported as of right now
        case Type::Bool: {
            break;
        }
        case Type::Func: {
            break;
        }
        case Type::String: {
            break;
        }
            */
        default: {
            return std::nullopt;
        }
        }
    }
}; // namespace utils

namespace fns {
    mlir::Operation *
    emit_add_expr(EmitterContext &emitter_context, const CallExpr &ce, const std::vector<mlir::Operation *> arguments) {
        mlir::OpBuilder &builder = emitter_context.builder;

        // TODO: Add support for: type specific ops (separate for floats) and variadic arguments (folded chains + minimum len assertions)

        auto fst_arg            = arguments[0]->getResult(0);
        auto snd_arg            = arguments[1]->getResult(0);
        mlir::arith::AddIOp sum = builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), fst_arg, snd_arg);
        return sum;
    }

    mlir::Operation *
    emit_mul_expr(EmitterContext &emitter_context, const CallExpr &ce, const std::vector<mlir::Operation *> arguments) {
        mlir::OpBuilder &builder = emitter_context.builder;

        // TODO: Add support for: type specific ops (separate for floats) and variadic arguments (folded chains + minimum len assertions)

        auto fst_arg            = arguments[0]->getResult(0);
        auto snd_arg            = arguments[1]->getResult(0);
        mlir::arith::MulIOp sum = builder.create<mlir::arith::MulIOp>(builder.getUnknownLoc(), fst_arg, snd_arg);
        return sum;
    }

    mlir::Operation *
    emit_user_defined_fn_call(EmitterContext &emitter_context, const CallExpr &ce, const std::vector<mlir::Operation *> arguments) {
        mlir::OpBuilder &builder       = emitter_context.builder;
        const std::string &source_text = emitter_context.semantic_model.compilation_unit->source_text;
        const std::string fn_name      = source_text.substr(ce.operator_token.start, ce.operator_token.length);

        const TypeMap &type_map = std::cref(*emitter_context.semantic_model.type_map);
        auto it                 = type_map.find(ce.id);
        if (it == type_map.end()) {
            // TODO: throw error here
            return nullptr;
        }
        Option<mlir::Type> type_opt = utils::to_mlir_type(builder, *it->second);
        if (!type_opt) {
            // TODO: throw error here
            return nullptr;
        }

        llvm::SmallVector<mlir::Value, 4> argv;
        for (const auto &arg : arguments) {
            argv.push_back(arg->getResult(0));
        }

        mlir::func::CallOp call = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), fn_name, type_opt.value(), argv);
        return call;
    }
}; // namespace fns

mlir::Operation *
emit_call_expr(EmitterContext &emitter_context, const ExprBase &expr_base) {
    // clang-format off
    using EmitBuiltinFnCallDelegate = mlir::Operation *(*)(EmitterContext &, const CallExpr &, const std::vector<mlir::Operation *>);
    std::unordered_map<std::string,EmitBuiltinFnCallDelegate> delegates = {
        { "+", fns::emit_add_expr },
        // { "-", builtin_fns::emit_subtract_expr },
        { "*", fns::emit_mul_expr },
        // { "/", builtin_fns::emit_divide_expr }
    };
    // clang-format on

    const CallExpr &ce             = static_cast<const CallExpr &>(expr_base);
    const std::string &source_text = emitter_context.semantic_model.compilation_unit->source_text;

    // we parse the arguments first:
    std::vector<mlir::Operation *> emitted_args;
    for (const auto &arg_expr_view : ptr_utils::get_expr_views(ce.arguments)) {
        mlir::Operation *arg_operation = emit_expr_base(emitter_context, arg_expr_view);
        if (!arg_operation) {
            return nullptr;
        }
        emitted_args.push_back(arg_operation);
    }

    const std::string operator_name = source_text.substr(ce.operator_token.start, ce.operator_token.length);
    auto it                         = delegates.find(operator_name);
    auto delegate                   = it != delegates.end() ? it->second : fns::emit_user_defined_fn_call;
    return delegate(emitter_context, ce, emitted_args);
}

mlir::ModuleOp
clobber::emit(mlir::MLIRContext &context, const SemanticModel &semantic_model) {

    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    mlir::FunctionType entry_point_type = builder.getFunctionType({}, {});
    mlir::func::FuncOp entry_point_fn   = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", entry_point_type);
    mlir::Block &entry_point            = *entry_point_fn.addEntryBlock(); // set the 'cursor' to inside the entry point fn
    builder.setInsertionPointToStart(&entry_point);

    // clang-format off
    EmitterContext em_ctx{
        .semantic_model = semantic_model,
        .builder = builder,
        .errors = {}
    };
    // clang-format on

    std::vector<mlir::Operation *> emitted_exprs;
    for (const auto &expr_view : ptr_utils::get_expr_views(semantic_model.compilation_unit->exprs)) {
        mlir::Operation *op = emit_expr_base(em_ctx, expr_view);
        if (op) {
            emitted_exprs.push_back(op);
        }
    }

    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(entry_point_fn);
    return module;

    // Sanity check block; just to test if the core mlir stuff works
    /*
    mlir::OpBuilder builder(&context);
    mlir::Location loc = builder.getUnknownLoc();

    // Create a new MLIR module
    // mlir::OwningOpRef<mlir::ModuleOp> _module = mlir::ModuleOp::create(loc);
    auto _module = mlir::ModuleOp::create(loc);

    // Create function type: () -> i32
    auto i32    = builder.getI32Type();
    auto fnType = builder.getFunctionType({}, i32);

    // Create @main function
    auto func          = builder.create<mlir::func::FuncOp>(loc, "main", fnType);
    mlir::Block *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Emit constants
    auto c1 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI32IntegerAttr(42));
    auto c2 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI32IntegerAttr(58));

    // Add them
    // auto sum = builder.create<mlir::arith::AddIOp>(loc, c1, c2);
    auto sum = builder.create<mlir::arith::AddIOp>(loc, c1, c2).getResult();
    builder.create<mlir::func::ReturnOp>(loc, sum);

    // Add function to module
    // _module->push_back(func);
    _module.push_back(func);
    return _module;
    */
}