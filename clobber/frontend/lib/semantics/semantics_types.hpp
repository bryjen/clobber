#pragma once

#include "clobber/pch.hpp"

#include "clobber/ast/ast.hpp"
#include "clobber/parser.hpp"
#include "clobber/semantics.hpp"

namespace clobber {
    using Scope = std::unordered_map<std::string, clobber::Symbol>;

    /* @brief Symbol table to keep track of variables and symbols across scopes. */
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

    /* @brief Zero overhead hash implementation for `clobber::Variant` using templated variant pattern matching with constexpr. */
    struct VariantHash {
        size_t
        operator()(clobber::Variant const &v) const noexcept {
            return std::visit(
                [](auto const &x) {
                    using T = std::decay_t<decltype(x)>;
                    if constexpr (std::is_same_v<T, clobber::PrimitiveType>)
                        return std::hash<int>{}(static_cast<int>(x.kind));

                    if constexpr (std::is_same_v<T, clobber::VectorType>)
                        return std::hash<clobber::Type>{}(x.element);

                    if constexpr (std::is_same_v<T, clobber::TensorType>) {
                        size_t h = std::hash<clobber::Type>{}(x.element);
                        for (auto s : x.shape)
                            h = h * 31 ^ std::hash<size_t>{}(s);
                        return h;
                    }

                    if constexpr (std::is_same_v<T, clobber::FunctionType>) {
                        size_t h = std::hash<clobber::Type>{}(x.ret);
                        for (auto const &p : x.params)
                            h = h * 31 ^ std::hash<clobber::Type>{}(p);
                        return h;
                    }

                    if constexpr (std::is_same_v<T, clobber::UserDefinedType>)
                        return std::hash<std::string>{}(x.name);

                    return size_t(0); // NilType
                },
                v);
        }
    };

    /* @brief Equals comparison implementation for `clobber::Variant`. */
    struct VariantEq {
        bool
        operator()(clobber::Variant const &a, clobber::Variant const &b) const noexcept {
            return a == b;
        }
    };

    /* @brief Data structure for type interning. */
    class TypePool {
    private:
        std::unordered_map<clobber::Variant, std::weak_ptr<clobber::TypeData>, VariantHash, VariantEq> pool;

    public:
        template <typename T, typename... Args>
        clobber::Type
        intern(Args &&...args) {
            clobber::Variant key{T{std::forward<Args>(args)...}};

            // lookup
            if (auto it = pool.find(key); it != pool.end()) {
                if (auto sp = it->second.lock()) // still alive
                    return sp;
            }

            // miss → create
            auto sp     = std::make_shared<clobber::TypeData>(clobber::TypeData{std::move(key)});
            pool[sp->v] = sp;
            return sp;
        }

        // clang-format off
        clobber::Type i8()   { return intern<clobber::PrimitiveType>(clobber::PrimitiveType::Kind::I8); }
        clobber::Type i16()  { return intern<clobber::PrimitiveType>(clobber::PrimitiveType::Kind::I16); }
        clobber::Type i32()  { return intern<clobber::PrimitiveType>(clobber::PrimitiveType::Kind::I32); }
        clobber::Type i64()  { return intern<clobber::PrimitiveType>(clobber::PrimitiveType::Kind::I64); }
        clobber::Type f32()  { return intern<clobber::PrimitiveType>(clobber::PrimitiveType::Kind::F32); }
        clobber::Type f64()  { return intern<clobber::PrimitiveType>(clobber::PrimitiveType::Kind::F64); }
        clobber::Type str()  { return intern<clobber::PrimitiveType>(clobber::PrimitiveType::Kind::Str); }
        clobber::Type _char(){ return intern<clobber::PrimitiveType>(clobber::PrimitiveType::Kind::Char); }
        clobber::Type nil()  { return intern<clobber::NilType>(); }
        // clang-format on

        clobber::Type
        func(std::vector<clobber::Type> param_types, clobber::Type ret_type) {
            return this->intern<clobber::FunctionType>(param_types, ret_type);
        }

        clobber::Type
        tensor(clobber::Type type, std::vector<size_t> shape) {
            return this->intern<clobber::TensorType>(type, shape);
        }
    };
}; // namespace clobber