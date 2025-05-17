#pragma once

#pragma warning(push)
#pragma warning(disable : 4267 4244 4996)
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#pragma warning(pop)

#include "clobber/mlir-backend/pch.hpp"

namespace clobber {
    struct SemanticModel; // clobber/semantics.hpp
    struct Diagnostic;    // clobber/common/diagnostic.hpp
}; // namespace clobber

namespace clobber {
    mlir::ModuleOp emit(mlir::MLIRContext &context, const SemanticModel &semantic_model, std::vector<clobber::Diagnostic> &diagnostics);

    struct CompilationEnvironment {};

    struct TargetConfig {
        enum class AccelerationBackend {
            None, // executes accel blocks in the CPU
            GPU,
            NPU
        } acceleration_backend;
    };

    bool validate_target_config(const TargetConfig &target_config);

    int jit_execute(mlir::MLIRContext &context, mlir::ModuleOp &module, const TargetConfig &target_config, std::vector<std::string> argv);
}; // namespace clobber