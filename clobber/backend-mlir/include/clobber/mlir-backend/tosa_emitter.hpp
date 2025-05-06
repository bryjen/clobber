#ifndef TOSA_EMITTER_HPP
#define TOSA_EMITTER_HPP

#pragma warning(push)
#pragma warning(disable : 4267 4244 4996)
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#pragma warning(pop)

struct CompilationUnit; // clobber/ast.hpp
struct EmitError;       // emit_error.hpp
struct EmitError;       // emit_error.hpp

void test_tosa_mlir_1();
void test_tosa_mlir_2();

namespace TosaEmitter {
/* @brief
 */
void init_context(mlir::MLIRContext &context);

/* @brief
 */
mlir::ModuleOp lower_ast_to_tosa(mlir::MLIRContext &, const CompilationUnit &, std::vector<EmitError> &);
}; // namespace TosaEmitter

#endif // TOSA_EMITTER_HPP