#ifndef LOWERING_HPP
#define LOWERING_HPP

#pragma warning(push)
#pragma warning(disable : 4267 4244 4996)
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#pragma warning(pop)

namespace Lowering {
/* @brief
 */
mlir::ModuleOp lower_to_spirv(mlir::MLIRContext &, const mlir::ModuleOp &, std::vector<EmitError> &);

/* @brief
 */
mlir::ModuleOp lower_to_llvm(mlir::MLIRContext &, const mlir::ModuleOp &, std::vector<EmitError> &);
}; // namespace Lowering
#endif // LOWERING_HPP