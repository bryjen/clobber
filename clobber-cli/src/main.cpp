#include <iostream>
#include <optional>
#include <string>

#include <clobber/ast.hpp>
#include <clobber/cli/actions.hpp>
#include <clobber/cli/args.hpp>
#include <clobber/mlir-backend/tosa_emitter.hpp>

// clang-format off
#define ASSERT_PARSE_OK(x)                \
    do {                                  \
        if ((x) != 0)                     \
            return -1;                    \
    } while (0)
// clang-format on

int
main(int argc, char **argv) {
    int exit_code = 0;
    Args args{};

    ASSERT_PARSE_OK(cli11_parse_args(argc, argv, args));

    switch (args.parsed_subcommand) {
    case ParsedSubcommand::Build:
        exit_code = build(args.build_args);
        break;
    case ParsedSubcommand::New:
        exit_code = _new(args.new_args);
        break;
    case ParsedSubcommand::Run:
        exit_code = run(args.run_args);
        break;
    default:
        exit_code = -1;
        break;
    }
}