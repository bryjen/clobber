#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "clobber/cli/actions.hpp"
#include "clobber/cli/args.hpp"

namespace fs                       = std::filesystem;
template <typename T> using Option = std::optional<T>;

struct ProjectConfig {
    // TODO: Add support for project file (ex. <PROJECTNAME>.toml)
};

Option<ProjectConfig>
get_project_config(const fs::path &path) {
    return std::nullopt;
}

std::vector<fs::path>
get_project_files() {
    throw 0;
}

int
build(const BuildArgs &build_args) {
    std::string name       = build_args.name;
    std::string output_dir = build_args.output_dir;

    return 0;
}

int
run(const RunArgs &run_args) {
    std::string name       = run_args.name;
    std::string output_dir = run_args.output_dir;

    return 0;
}