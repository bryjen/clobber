#pragma warning(push)
#pragma warning(disable : 4267 4244 4996)
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>

#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#pragma warning(pop)

#include <functional>
#include <optional>

#include "clobber/ast.hpp"
#include "clobber/mlir-backend/emit_error.hpp"
#include "clobber/mlir-backend/tosa_emitter.hpp"

template <typename T> using Option = std::optional<T>;

struct EmittedOp {
    mlir::Operation *op;
    enum class Kind {
        Add,
        Const,
        Unknown
    } kind;
};

using LoweringDelegate = bool (*)(mlir::OpBuilder &, std::vector<EmitError> &, const ExprBase &, EmittedOp &);

void
test_tosa_mlir_1() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tosa::TosaDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    // tensor<i32> scalar type
    auto type     = mlir::RankedTensorType::get({}, builder.getIntegerType(32));
    auto funcType = builder.getFunctionType({type, type}, {});

    auto func   = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", funcType);
    auto &entry = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entry);

    auto sum = builder.create<mlir::tosa::AddOp>(builder.getUnknownLoc(), type, entry.getArgument(0), entry.getArgument(1));

    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    module.push_back(func);

    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "TOSA MLIR verification failed\n";
    } else {
        llvm::outs() << "TOSA MLIR module:\n";
        module.dump();
    }
}

void
test_tosa_mlir_2() {
    mlir::MLIRContext context;
    context.loadDialect<mlir::tosa::TosaDialect, mlir::func::FuncDialect, mlir::tensor::TensorDialect>();

    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    mlir::Type i32Type                = builder.getIntegerType(32);
    mlir::RankedTensorType tensorType = mlir::RankedTensorType::get({1, 1}, i32Type);

    mlir::FunctionType funcType = builder.getFunctionType({}, {});
    mlir::func::FuncOp func     = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "entry", funcType);
    mlir::Block &entryBlock     = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Constant [[42]]
    mlir::DenseElementsAttr tensorAttr = mlir::DenseElementsAttr::get(tensorType, {42});
    mlir::tosa::ConstOp constOp        = builder.create<mlir::tosa::ConstOp>(builder.getUnknownLoc(), tensorType, tensorAttr);

    // Add [[42]] + [[42]]
    mlir::tosa::AddOp addOp =
        builder.create<mlir::tosa::AddOp>(builder.getUnknownLoc(), tensorType, constOp.getResult(), constOp.getResult());

    // Return
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    module.push_back(func);

    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "TOSA MLIR verification failed\n";
    } else {
        llvm::outs() << "TOSA MLIR module:\n";
        module.dump();
    }
}

void init_entry_fn(mlir::OpBuilder &, mlir::func::FuncOp &);
bool lower_numerical_literal_expr(mlir::OpBuilder &, std::vector<EmitError> &, const ExprBase &, EmittedOp &);
bool lower_call_expr(mlir::OpBuilder &, std::vector<EmitError> &, const ExprBase &, EmittedOp &);
bool lower_expr(mlir::OpBuilder &, std::vector<EmitError> &, const ExprBase &, EmittedOp &);

void
TosaEmitter::init_context(mlir::MLIRContext &context) {
    context.getOrLoadDialect<mlir::tosa::TosaDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
}

/* @brief Returns a list of readonly views from the expressions, which are unique ptrs.
 */
std::vector<std::reference_wrapper<const ExprBase>>
get_expr_views(const CompilationUnit &clobber_cu) {
    std::vector<std::reference_wrapper<const ExprBase>> expr_views;

    for (const auto &expr : clobber_cu.exprs) {
        expr_views.push_back(std::cref(*expr));
    }

    return expr_views;
}

mlir::ModuleOp
TosaEmitter::lower_ast_to_tosa(mlir::MLIRContext &context, const CompilationUnit &clobber_cu, std::vector<EmitError> &emit_errors) {
    mlir::func::FuncOp entry_point_fn;

    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
    init_entry_fn(builder, entry_point_fn);

    auto exprs = get_expr_views(clobber_cu);

    for (const ExprBase &expr : exprs) {
        EmittedOp op;

        switch (expr.expr_type) {
        case ExprType::NumericLiteralExpr: {
            auto something1 = lower_numerical_literal_expr(builder, emit_errors, expr, op);
            break;
        }
        case ExprType::CallExpr: {
            auto something2 = lower_call_expr(builder, emit_errors, expr, op);
            break;
        }
        default: {
            // TODO: emit error here
            throw 69420;
        }
        }
    }

    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(entry_point_fn);

    return module;
}

void
init_entry_fn(mlir::OpBuilder &builder, mlir::func::FuncOp &out_fn) {
    mlir::FunctionType entry_point_type = builder.getFunctionType({}, {});
    mlir::func::FuncOp entry_point_fn   = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", entry_point_type);
    mlir::Block &entry_point            = *entry_point_fn.addEntryBlock(); // set the 'cursor' to inside the entry point fn
    builder.setInsertionPointToStart(&entry_point);

    out_fn = entry_point_fn;
}

bool
lower_numerical_literal_expr(mlir::OpBuilder &builder, std::vector<EmitError> &errors, const ExprBase &expr, EmittedOp &out_op) {
    // TOSA doesn't natively support integers as is, so we have to use a 1x1 tensor type
    // auto *ptr = dynamic_cast<const NumLiteralExpr *>(&expr);
    const NumLiteralExpr &num_lit_expr = static_cast<const NumLiteralExpr &>(expr);

    mlir::RankedTensorType type  = mlir::RankedTensorType::get({1, 1}, builder.getIntegerType(32)); // int32 1x1 tensor
    mlir::DenseElementsAttr attr = mlir::DenseElementsAttr::get(type, {1});
    mlir::tosa::ConstOp const_op = builder.create<mlir::tosa::ConstOp>(builder.getUnknownLoc(), type, attr);

    out_op.kind = EmittedOp::Kind::Const;
    out_op.op   = const_op.getOperation();

    return true;
}

bool
lower_call_expr(mlir::OpBuilder &builder, std::vector<EmitError> &errors, const ExprBase &expr, EmittedOp &out_op) {
    // TODO: Make this shi arity agnostic, or atl take 2
    EmittedOp op1;
    EmittedOp op2;
    mlir::Value val1;
    mlir::Value val2;

    auto *call_expr_ptr = dynamic_cast<const CallExpr *>(&expr);

    const ExprBase &sm1 = std::cref(*call_expr_ptr->arguments[0]);
    const ExprBase &sm2 = std::cref(*call_expr_ptr->arguments[1]);

    if (!lower_expr(builder, errors, sm1, op1)) {
        return false;
    }
    if (!lower_expr(builder, errors, sm2, op2)) {
        return false;
    }

    val1 = op1.op->getResult(0);
    val2 = op2.op->getResult(0);

    mlir::RankedTensorType type = mlir::RankedTensorType::get({1}, builder.getIntegerType(32)); // int32 1x1 tensor
    mlir::tosa::AddOp addOp     = builder.create<mlir::tosa::AddOp>(builder.getUnknownLoc(), type, val1, val2);
    return addOp;
}

bool
lower_expr(mlir::OpBuilder &builder, std::vector<EmitError> &errors, const ExprBase &expr_base, EmittedOp &out_op) {
    LoweringDelegate callback;

    // clang-format off
    static std::unordered_map<ExprType, LoweringDelegate> lowering_fns = {
        { ExprType::NumericLiteralExpr, lower_numerical_literal_expr },
        { ExprType::CallExpr, lower_call_expr },
    };
    // clang-format on

    auto it                               = lowering_fns.find(expr_base.expr_type);
    Option<LoweringDelegate> callback_opt = (it != lowering_fns.end()) ? std::make_optional(it->second) : std::nullopt;

    if (!callback_opt) {
        // TODO: Internal error here
        return false;
    }

    callback = callback_opt.value();

    return callback(builder, errors, expr_base, out_op);
}