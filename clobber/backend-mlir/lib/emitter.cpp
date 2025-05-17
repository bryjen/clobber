#include "clobber/mlir-backend/emitter.hpp"

#pragma warning(push)
#pragma warning(disable : 4267 4244 4996)
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>

#include <mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMInterfaces.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>
#pragma warning(pop)

#include <clobber/common/diagnostic.hpp>

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include <clobber/common/utils.hpp>

template <typename T> using Option = std::optional<T>;

class ScopeStack {
public:
    ScopeStack() { enter_scope(); }

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
    insert_symbol(const std::string &name, mlir::Operation *op) {
        auto &current = scopes.back();
        return current.emplace(name, op).second;
    }

    Option<mlir::Operation *>
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
    using Scope = std::unordered_map<std::string, mlir::Operation *>;
    std::vector<Scope> scopes;
};

class TypeCache {
public:
    mlir::Type i1, i8, i32, i64;
    mlir::Type ptr; // opaque ptr
    mlir::Type unit;

    static TypeCache
    init(mlir::MLIRContext &ctx) {
        TypeCache cache{};
        cache.i1   = mlir::IntegerType::get(&ctx, 1);
        cache.i8   = mlir::IntegerType::get(&ctx, 8);
        cache.i32  = mlir::IntegerType::get(&ctx, 32);
        cache.i64  = mlir::IntegerType::get(&ctx, 64);
        cache.ptr  = mlir::LLVM::LLVMPointerType::get(&ctx, 0);
        cache.unit = mlir::LLVM::LLVMStructType::getLiteral(&ctx, {}, true);
        return std::move(cache);
    }
};

/* @brief */
struct EmitterContext {
    const clobber::SemanticModel &semantic_model;
    mlir::OpBuilder &builder;
    std::vector<clobber::Diagnostic> &diagnostics;

    std::vector<std::tuple<mlir::Block *, mlir::Block::iterator>> scopes;
    ScopeStack &scope_stack;
    TypeCache &type_cache;

    // VariableValueOpHasher var_val_op_hasher;
};

namespace builtins {
    mlir::Operation *
    emit_llvm_printf(mlir::ModuleOp &module, EmitterContext &emitter_context) {
        mlir::OpBuilder &builder = emitter_context.builder;
        mlir::MLIRContext *ctx   = builder.getContext();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto ptrTy    = mlir::LLVM::LLVMPointerType::get(ctx, 0); // !llvm.ptr
        auto funcType = mlir::LLVM::LLVMFunctionType::get(builder.getI32Type(), {ptrTy}, /*isVarArg=*/true);

        mlir::Operation *fn_op = nullptr;
        if (!module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
            fn_op = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "printf", funcType);
        }

        return fn_op;
    }

    mlir::Operation *
    emit_nil_type(mlir::ModuleOp &module, EmitterContext &emitter_context) {
        mlir::OpBuilder &builder = emitter_context.builder;
        mlir::MLIRContext *ctx   = builder.getContext();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        mlir::Type unitTy = mlir::LLVM::LLVMStructType::getLiteral(ctx, {}, true);
        throw 0;
    }

    mlir::Operation *
    emit_print_cstr(mlir::ModuleOp &module, EmitterContext &emitter_context) {
        mlir::OpBuilder &builder = emitter_context.builder;
        mlir::MLIRContext *ctx   = builder.getContext();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto ptrTy    = mlir::LLVM::LLVMPointerType::get(ctx, 0); // !llvm.ptr
        auto funcType = mlir::LLVM::LLVMFunctionType::get(emitter_context.type_cache.i64, {ptrTy}, /*isVarArg=*/true);

        mlir::Operation *fn_op = nullptr;
        if (!module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("print_cstr")) {
            fn_op = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "print_cstr", funcType);
        }

        return fn_op;
    }
}; // namespace builtins

namespace utils {
    Option<mlir::Type>
    to_mlir_type(mlir::OpBuilder &builder, const clobber::Type &type) {
        switch (type.kind) {
        case clobber::Type::Char: {
            return std::make_optional(builder.getI8Type());
        }
        case clobber::Type::Double: {
            return std::make_optional(builder.getF64Type());
        }
        case clobber::Type::Float: {
            return std::make_optional(builder.getF32Type());
        }
        case clobber::Type::Int: {
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
using EmitDelegate = mlir::Operation *(*)(EmitterContext &, const clobber::Expr &);
#define EMIT_DELEGATE_FN(FN_NAME) mlir::Operation *FN_NAME(EmitterContext &, const clobber::Expr &);

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
emit_expr_base(EmitterContext &emitter_context, const clobber::Expr &expr) {
    // clang-format off
    std::unordered_map<clobber::Expr::Type, EmitDelegate> delegates = {
        {clobber::Expr::Type::NumericLiteralExpr, emit_num_literal_expr},
        {clobber::Expr::Type::StringLiteralExpr, emit_string_literal_expr},
        {clobber::Expr::Type::CharLiteralExpr, emit_char_literal_expr},
        {clobber::Expr::Type::IdentifierExpr, emit_identifier_expr},
        {clobber::Expr::Type::LetExpr, emit_let_expr},
        {clobber::Expr::Type::FnExpr, emit_fn_expr},
        {clobber::Expr::Type::DefExpr, emit_def_expr},
        {clobber::Expr::Type::DoExpr, emit_do_expr},
        {clobber::Expr::Type::CallExpr, emit_call_expr},
    };
    // clang-format on

    auto it = delegates.find(expr.type);
    if (it == delegates.end()) {
        // TODO: throw error here
        return nullptr;
    }

    return it->second(emitter_context, expr);
}

mlir::Operation *
emit_num_literal_expr(EmitterContext &emitter_context, const clobber::Expr &expr_base) {
    mlir::OpBuilder &builder           = emitter_context.builder;
    const clobber::NumLiteralExpr &nle = static_cast<const clobber::NumLiteralExpr &>(expr_base);
    const std::string &source_text     = emitter_context.semantic_model.compilation_unit->source_text;

    const clobber::TypeMap &type_map = std::cref(*emitter_context.semantic_model.type_map);
    auto it                          = type_map.find(nle.hash());

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
    case clobber::Type::Int: {
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
    case clobber::Type::Float: {
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
    case clobber::Type::Double: {
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
emit_string_literal_expr(EmitterContext &emitter_context, const clobber::Expr &expr) {
    const clobber::StringLiteralExpr &sle = static_cast<const clobber::StringLiteralExpr &>(expr);
    const std::string str                 = sle.value;
    mlir::OpBuilder &builder              = emitter_context.builder;
    mlir::MLIRContext *ctx                = builder.getContext();
    mlir::Location loc                    = builder.getUnknownLoc();

    auto i8_type    = mlir::IntegerType::get(ctx, 8);
    auto i64_type   = mlir::IntegerType::get(ctx, 64);
    auto ptr_type   = mlir::LLVM::LLVMPointerType::get(ctx, 0); // opaque ptr
    auto array_type = mlir::LLVM::LLVMArrayType::get(i8_type, static_cast<uint64_t>(str.size() + 1));

    auto len_val = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64Type(), builder.getI64IntegerAttr(str.size()));
    auto alloca  = builder.create<mlir::LLVM::AllocaOp>(loc, mlir::LLVM::LLVMPointerType::get(ctx, 0), array_type, len_val, 0);
    auto basePtr = builder.create<mlir::LLVM::BitcastOp>(loc, ptr_type, alloca);

    // Store characters (including null terminator)
    for (size_t i = 0; i <= str.size(); ++i) {
        char ch = (i < str.size()) ? str[i] : '\0';

        auto charVal = builder.create<mlir::LLVM::ConstantOp>(loc, i8_type, builder.getI8IntegerAttr(ch));

        auto indexVal = builder.create<mlir::LLVM::ConstantOp>(loc, i64_type, builder.getI64IntegerAttr(i));

        auto gep = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, i8_type, basePtr, mlir::ValueRange{indexVal}, false);

        builder.create<mlir::LLVM::StoreOp>(loc, charVal, gep);
    }

    return basePtr;
}

mlir::Operation *
emit_char_literal_expr(EmitterContext &emitter_context, const clobber::Expr &expr) {
    mlir::OpBuilder &builder            = emitter_context.builder;
    const clobber::CharLiteralExpr &cle = static_cast<const clobber::CharLiteralExpr &>(expr);

    char value = cle.value[0]; // asserted to exist through semantic analysis

    mlir::IntegerType i8       = builder.getIntegerType(8); // chars are just 8 bit integers
    mlir::arith::ConstantOp ch = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerAttr(i8, value));
    return ch;
}

mlir::Operation *
emit_identifier_expr(EmitterContext &emitter_context, const clobber::Expr &expr) {
    mlir::OpBuilder &builder                  = emitter_context.builder;
    const clobber::IdentifierExpr &ident_expr = static_cast<const clobber::IdentifierExpr &>(expr);
    mlir::Block *current_block                = builder.getInsertionBlock();

    // mlir::Operation *value_op = emitter_context.var_val_op_hasher.try_get_op(current_block, ident_expr.name);
    Option<mlir::Operation *> value_op = emitter_context.scope_stack.lookup_symbol(ident_expr.name);
    if (!value_op) {
        // TODO: Emit error here
        throw 0;
    }

    return value_op.value();
}

mlir::Operation *
emit_let_expr(EmitterContext &emitter_context, const clobber::Expr &expr) {
    mlir::OpBuilder &builder         = emitter_context.builder;
    const clobber::LetExpr &let_expr = static_cast<const clobber::LetExpr &>(expr);

    const clobber::TypeMap &type_map = std::cref(*emitter_context.semantic_model.type_map);
    auto type_it                     = type_map.find(let_expr.hash());
    if (type_it == type_map.end()) {
        // TODO: throw internal error here
        return nullptr;
    }
    auto merge_ret_type_opt = utils::to_mlir_type(builder, *type_it->second);
    if (!merge_ret_type_opt) {
        // TODO: throw internal error here
        return nullptr;
    }

    // no need to isolate variables in scope, asserted by semantic analyzer

    emitter_context.scope_stack.enter_scope();

    for (size_t i = 0; i < let_expr.binding_vector_expr->num_bindings; i++) {
        const std::string &name = let_expr.binding_vector_expr->identifiers[i]->name;
        const auto &expr_view   = std::cref(*let_expr.binding_vector_expr->exprs[i]);

        mlir::Operation *op = emit_expr_base(emitter_context, expr_view);
        if (!op) {
            return nullptr; // propagates error from function call
        }

        emitter_context.scope_stack.insert_symbol(name, op);
    }

    mlir::Operation *last_operation = nullptr;
    for (const auto &body_expr_view : ptr_utils::get_expr_views(let_expr.body_exprs)) {
        last_operation = emit_expr_base(emitter_context, body_expr_view);
    }

    emitter_context.scope_stack.exit_scope();

    return last_operation;
}

mlir::Operation *
emit_fn_expr(EmitterContext &emitter_context, const clobber::Expr &expr) {
    throw 0;
}

mlir::Operation *
emit_def_expr(EmitterContext &emitter_context, const clobber::Expr &expr) {
    throw 0;
}

mlir::Operation *
emit_do_expr(EmitterContext &emitter_context, const clobber::Expr &expr) {
    throw 0;
}

namespace fns {
    mlir::Operation *
    emit_add_expr(EmitterContext &emitter_context, const clobber::CallExpr &ce, const std::vector<mlir::Operation *> arguments) {
        mlir::OpBuilder &builder = emitter_context.builder;

        if (arguments.size() < 2) {
            return nullptr; // silently 'error' semantic analyzer should point this out
        }

        // determining the type of add op to use
        const clobber::TypeMap &type_map = std::cref(*emitter_context.semantic_model.type_map);
        auto type_it                     = type_map.find(ce.hash());
        clobber::Type type{};
        if (type_it == type_map.end()) {
            // TODO: Temporary, just to test that delegates work, ensure that the semantic analyzer can understand overloaded types
            // TODO: throw error here
            // return nullptr;
            type.kind = clobber::Type::Kind::Float;
        } else {
            type = *type_it->second;
        }

        auto loc            = builder.getUnknownLoc();
        using delegate_type = std::function<mlir::Operation *(mlir::OpResult &, mlir::OpResult &)>;
        const std::unordered_map<clobber::Type::Kind, delegate_type> delegates{
            {clobber::Type::Kind::Int,
             [&builder, &loc](mlir::OpResult &fst, mlir::OpResult &snd) {
                 return (mlir::Operation *)builder.create<mlir::arith::AddIOp>(loc, fst, snd);
             }},
            {clobber::Type::Kind::Float,
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
    emit_mul_expr(EmitterContext &emitter_context, const clobber::CallExpr &ce, const std::vector<mlir::Operation *> arguments) {
        mlir::OpBuilder &builder = emitter_context.builder;

        // TODO: Add support for: type specific ops (separate for floats) and variadic arguments (folded chains + minimum len assertions)

        auto fst_arg            = arguments[0]->getResult(0);
        auto snd_arg            = arguments[1]->getResult(0);
        mlir::arith::MulIOp sum = builder.create<mlir::arith::MulIOp>(builder.getUnknownLoc(), fst_arg, snd_arg);
        return sum;
    }

    mlir::Operation *
    emit_user_defined_fn_call(EmitterContext &emitter_context, const clobber::CallExpr &ce,
                              const std::vector<mlir::Operation *> arguments) {
        mlir::OpBuilder &builder       = emitter_context.builder;
        const std::string &source_text = emitter_context.semantic_model.compilation_unit->source_text;
        const std::string fn_name      = source_text.substr(ce.operator_token.start, ce.operator_token.length);

        const clobber::TypeMap &type_map = std::cref(*emitter_context.semantic_model.type_map);
        auto it                          = type_map.find(ce.hash());
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
emit_call_expr(EmitterContext &emitter_context, const clobber::Expr &expr) {
    // clang-format off
    using EmitBuiltinFnCallDelegate = mlir::Operation *(*)(EmitterContext &, const clobber::CallExpr &, const std::vector<mlir::Operation *>);
    std::unordered_map<std::string,EmitBuiltinFnCallDelegate> delegates = {
        { "+", fns::emit_add_expr },
        // { "-", builtin_fns::emit_subtract_expr },
        { "*", fns::emit_mul_expr },
        // { "/", builtin_fns::emit_divide_expr }
    };
    // clang-format on

    const clobber::CallExpr &ce    = static_cast<const clobber::CallExpr &>(expr);
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
clobber::emit(mlir::MLIRContext &context, const SemanticModel &semantic_model, std::vector<clobber::Diagnostic> &diagnostics) {
    ScopeStack scope_stack;
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    TypeCache type_cache = TypeCache::init(context);

    mlir::FunctionType entry_point_type = builder.getFunctionType({}, {});
    mlir::func::FuncOp entry_point_fn   = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", entry_point_type);
    mlir::Block &entry_point            = *entry_point_fn.addEntryBlock(); // set the 'cursor' to inside the entry point fn
    builder.setInsertionPointToStart(&entry_point);

    // clang-format off
    EmitterContext em_ctx{
        .semantic_model = semantic_model,
        .builder = builder,
        .diagnostics = diagnostics,
        .scopes = { { &entry_point, builder.getInsertionPoint() } },
        .scope_stack = scope_stack,
        .type_cache = type_cache
        // .var_val_op_hasher = {}
    };
    // clang-format on

    builtins::emit_llvm_printf(module, em_ctx);
    builtins::emit_print_cstr(module, em_ctx);

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

bool
clobber::validate_target_config(const clobber::TargetConfig &target_config) {
    throw 0;
}

int
clobber::jit_execute(mlir::MLIRContext &context, mlir::ModuleOp &module, const TargetConfig &target_config, std::vector<std::string> argv) {
    // TODO: might need to rebuild the entirety of the monorepo also including all of llvm and clang
    /*  // via llvm lib
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

    if (!llvmModule) {
        llvm::errs() << "Translation failed.\n";
        return -1;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto jit = llvm::orc::LLJITBuilder().create();
    if (!jit) {
        llvm::errs() << "Failed to create JIT.\n";
        return -1;
    }

    auto &jitInstance = jit.get();
    if (auto err = jitInstance->addIRModule(llvm::orc::ThreadSafeModule(std::move(llvmModule), std::make_unique<llvm::LLVMContext>()))) {
        llvm::errs() << "Module load error.\n";
        return -1;
    }

    // Look up symbol
    auto sym = jitInstance->lookup("main");
    if (!sym) {
        llvm::errs() << "Symbol not found.\n";
        return -1;
    }

    // Call main
    using MainFn = int (*)();
    int result   = sym->toPtr<MainFn>()();
    llvm::outs() << "Returned: " << result << "\n";
    return result;
    */

    /*  // via mlir thing
    mlir::registerLLVMDialectTranslation(context);

    mlir::translateModuleToLLVMIR

        auto maybeEngine = mlir::ExecutionEngine::create(module, mlir::ExecutionEngineOptions()
                                                                     .setTransformer(mlir::makeOptimizingTransformer(3, 0, nullptr))
                                                                     .setSharedLibPaths({}) // add paths if using external .so/.dll
        );
    if (!maybeEngine) {
        llvm::errs() << "Failed to create ExecutionEngine\n";
        return;
    }
    auto &engine = maybeEngine.get();

    engine->registerSymbols([](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap map;
        map[interner("printf")] = llvm::JITEvaluatedSymbol(reinterpret_cast<uint64_t>(&printf), llvm::JITSymbolFlags::Exported);
        return map;
    });

    llvm::Expected<int32_t> result = engine->invoke("main");
    if (!result) {
        llvm::errs() << "Execution failed\n";
        return;
    }
    llvm::outs() << "Returned: " << result.get() << "\n";

    return 0;
    */
}