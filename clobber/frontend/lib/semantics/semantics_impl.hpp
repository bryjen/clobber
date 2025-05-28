#pragma once

#include <clobber/common/utils.hpp>
#include <clobber/common/variant.hpp>

#include "../clobber/internal/diagnostic_factory.hpp"
#include "clobber/pch.hpp"

#include "clobber/ast/ast.hpp"
#include "clobber/parser.hpp"
#include "clobber/semantics.hpp"

#include "semantics_types.hpp"

namespace clobber {
    namespace semantic_utils {
        clobber::Type convert_type_expr(TypePool &tp, const clobber::TypeExpr &type_expr);
    };

}; // namespace clobber