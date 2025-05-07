#ifndef ARGS_HPP
#define ARGS_HPP

#include <string>

enum class ParsedSubcommand {
    New,
    Build,
    Run
};

struct NewArgs {
    std::string name;
    std::string output_dir;
    bool init_git;
};

struct BuildArgs {
    std::string name;
    std::string output_dir;
};

struct RunArgs {
    std::string name;
    std::string output_dir;
};

struct Args {
    ParsedSubcommand parsed_subcommand;
    NewArgs new_args;
    BuildArgs build_args;
    RunArgs run_args;
};

namespace ArgsUtils {
void finalize(NewArgs &);
void finalize(BuildArgs &);
void finalize(RunArgs &);
}; // namespace ArgsUtils

int cli11_parse_args(int argc, char **argv, Args &out_args);

#endif // ARGS_HPP