#pragma warning(push)
#pragma warning(disable : 4267 4244 4996)
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#pragma warning(pop)

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include <clobber/common/utils.hpp>

#include "clobber/mlir-backend/emitter.hpp"

template <typename T> using Option = std::optional<T>;

/* @brief Used to store the operation for variables. Accounts for scopes via block ptr hashes. */
class VariableValueOpHasher {
public:
    void
    insert(mlir::Block *block, const std::string &var, mlir::Operation *op) {
        if (!block || !op) {
            return;
        }

        if (block_ptr_hash_map.find(block) == block_ptr_hash_map.end()) {
            block_ptr_hash_map[block] = current_block_ptr_hash++;
        }

        var_value_map[get_hash(block, var)] = op;
    }

    mlir::Operation *
    try_get_op(mlir::Block *block, const std::string &var) {
        if (!block) {
            return nullptr;
        }

        if (block_ptr_hash_map.find(block) == block_ptr_hash_map.end()) {
            block_ptr_hash_map[block] = current_block_ptr_hash++;
        }

        auto it = var_value_map.find(get_hash(block, var));
        return it != var_value_map.end() ? it->second : nullptr;
    }

private:
    size_t
    get_hash(mlir::Block *block, const std::string &var) { // assumed 'block' is not a nullptr
        size_t string_hash = std::hash<std::string>{}(var);
        size_t block_hash  = block_ptr_hash_map[block];
        return string_hash ^ (block_hash + 0x9e3779b9 + (string_hash << 6) + (string_hash >> 2));
    }

    size_t current_block_ptr_hash;
    std::unordered_map<mlir::Block *, size_t> block_ptr_hash_map;
    std::unordered_map<size_t, mlir::Operation *> var_value_map;
};

/* @brief */
struct EmitterContext {
    const SemanticModel &semantic_model;
    mlir::OpBuilder &builder;
    std::vector<EmitError> errors;

    std::vector<std::tuple<mlir::Block *, mlir::Block::iterator>> scopes;
    VariableValueOpHasher var_val_op_hasher;
};

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

    // sets insertion location to the start of the scope temporarily
    // mlir::OpBuilder::InsertionGuard _(builder);
    // builder.setInsertionPointToStart(emitter_context.scopes.back());

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
            const_op            = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getFloatAttr(f32, value));
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
            const_op            = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getFloatAttr(f64, value));
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
    mlir::OpBuilder &builder         = emitter_context.builder;
    const IdentifierExpr &ident_expr = static_cast<const IdentifierExpr &>(expr_base);
    mlir::Block *current_block       = builder.getInsertionBlock();

    mlir::Operation *value_op = emitter_context.var_val_op_hasher.try_get_op(current_block, ident_expr.name);
    if (!value_op) {
        // TODO: Emit error here
        throw 0;
    }

    return value_op;
}

mlir::Operation *
emit_let_expr(EmitterContext &emitter_context, const ExprBase &expr_base) {
    mlir::OpBuilder &builder = emitter_context.builder;
    const LetExpr &let_expr  = static_cast<const LetExpr &>(expr_base);

    const TypeMap &type_map = std::cref(*emitter_context.semantic_model.type_map);
    auto type_it            = type_map.find(let_expr.id);
    if (type_it == type_map.end()) {
        // TODO: throw internal error here
        return nullptr;
    }
    auto merge_ret_type_opt = utils::to_mlir_type(builder, *type_it->second);
    if (!merge_ret_type_opt) {
        // TODO: throw internal error here
        return nullptr;
    }

    // mlir::OpBuilder::InsertionGuard _(builder);
    mlir::Block *block   = builder.getInsertionBlock();
    mlir::Region *region = block->getParent();

    mlir::Block *sub_block   = builder.createBlock(region);
    mlir::Block *merge_block = builder.createBlock(region);
    merge_block->addArgument(merge_ret_type_opt.value(), builder.getUnknownLoc());

    {
        mlir::OpBuilder::InsertionGuard _(builder);
        builder.setInsertionPointToEnd(block);
        builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), sub_block);

        emitter_context.scopes.push_back({sub_block, builder.getInsertionPoint()});
    }

    builder.setInsertionPointToStart(sub_block);
    for (size_t i = 0; i < let_expr.binding_vector_expr->num_bindings; i++) {
        const std::string &name = let_expr.binding_vector_expr->identifiers[i]->name;
        const auto &expr_view   = std::cref(*let_expr.binding_vector_expr->exprs[i]);

        mlir::Operation *op = emit_expr_base(emitter_context, expr_view);
        if (!op) {
            return nullptr; // propagates error from function call
        }

        emitter_context.var_val_op_hasher.insert(sub_block, name, op);
    }

    mlir::Operation *last_operation = nullptr;
    for (const auto &body_expr_view : ptr_utils::get_expr_views(let_expr.body_exprs)) {
        last_operation = emit_expr_base(emitter_context, body_expr_view);
    }

    {
        mlir::OpBuilder::InsertionGuard _(builder);
        builder.setInsertionPointToEnd(sub_block);
        builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), merge_block, mlir::ValueRange{last_operation->getResult(0)});

        emitter_context.scopes.pop_back();
    }

    // builder.setInsertionPointToStart(merge_block);
    builder.setInsertionPointToEnd(merge_block);
    return last_operation;
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

namespace fns {
    mlir::Operation *
    emit_add_expr(EmitterContext &emitter_context, const CallExpr &ce, const std::vector<mlir::Operation *> arguments) {
        mlir::OpBuilder &builder = emitter_context.builder;

        if (arguments.size() < 2) {
            return nullptr; // silently 'error' semantic analyzer should point this out
        }

        // determining the type of add op to use
        const TypeMap &type_map = std::cref(*emitter_context.semantic_model.type_map);
        auto type_it            = type_map.find(ce.id);
        Type type{};
        if (type_it == type_map.end()) {
            // TODO: Temporary, just to test that delegates work, ensure that the semantic analyzer can understand overloaded types
            // TODO: throw error here
            // return nullptr;
            type.kind = Type::Kind::Float;
        } else {
            type = *type_it->second;
        }

        auto loc            = builder.getUnknownLoc();
        using delegate_type = std::function<mlir::Operation *(mlir::OpResult &, mlir::OpResult &)>;
        const std::unordered_map<Type::Kind, delegate_type> delegates{
            {Type::Kind::Int,
             [&builder, &loc](mlir::OpResult &fst, mlir::OpResult &snd) {
                 return (mlir::Operation *)builder.create<mlir::arith::AddIOp>(loc, fst, snd);
             }},
            {Type::Kind::Float,
             [&builder, &loc](mlir::OpResult &fst, mlir::OpResult &snd) {
                 return (mlir::Operation *)builder.create<mlir::arith::AddFOp>(loc, fst, snd);
             }},
        };

        auto delegate_it = delegates.find(type.kind);
        if (delegate_it == delegates.end()) {
            // TODO: throw error here
            return nullptr;
        }
        delegate_type delegate = delegate_it->second;

        // unfolding chains
        mlir::Operation *last_operation = nullptr;
        for (size_t i = 0; i < arguments.size() - 1; i++) {
            auto fst_arg   = arguments[i]->getResult(0);
            auto snd_arg   = arguments[i + 1]->getResult(0);
            last_operation = delegate(fst_arg, snd_arg);
        }
        return last_operation;
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
        .errors = {},
        .scopes = { { &entry_point, builder.getInsertionPoint() } },
        .var_val_op_hasher = {}
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
}