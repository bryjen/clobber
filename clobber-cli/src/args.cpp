#include <iostream>
#include <string>

#include <CLI/CLI.hpp>

#include "clobber/cli/args.hpp"

void
ArgsUtils::finalize(NewArgs &new_args) {
    throw 0;
}

void
ArgsUtils::finalize(BuildArgs &build_args) {
    throw 0;
}

void
ArgsUtils::finalize(RunArgs &run_args) {
    throw 0;
}

int
cli11_parse_args(int argc, char **argv, Args &args) {
    CLI::App app{"clobber"};

    CLI::App *new_subcommand   = app.add_subcommand("new", "Initialize a project.");
    CLI::App *build_subcommand = app.add_subcommand("build", "Build a project.");
    CLI::App *run_subcommand   = app.add_subcommand("run", "Build and run a project.");

    // new subcommand
    new_subcommand->add_option("-n,--name", args.new_args.name, "The name of the project.");
    new_subcommand->add_option("-o,--out", args.new_args.output_dir, "The directory to generate the project.");
    new_subcommand->add_flag("--initgit", args.new_args.init_git, "Initialize a git repo on creation.");

    // build subcommand
    build_subcommand->add_option("project", args.build_args.name, "The project file to build.");
    build_subcommand->add_option("-o,--out", args.build_args.output_dir, "The directory to generate build files.");

    // run subcommand
    run_subcommand->add_option("project", args.build_args.name, "The project file to build.");
    run_subcommand->add_option("-o,--out", args.build_args.output_dir, "The directory to generate build files.");

    app.require_subcommand(1); // require exactly one subcommand

    CLI11_PARSE(app, argc, argv);

    ParsedSubcommand parsed_subcommand;
    if (app.got_subcommand(new_subcommand)) {
        parsed_subcommand = ParsedSubcommand::New;
        ArgsUtils::finalize(args.new_args);

    } else if (app.got_subcommand(build_subcommand)) {
        parsed_subcommand = ParsedSubcommand::Build;
        ArgsUtils::finalize(args.build_args);

    } else if (app.got_subcommand(run_subcommand)) {
        parsed_subcommand = ParsedSubcommand::Run;
        ArgsUtils::finalize(args.run_args);

    } else {
        std::cout << "An unknown error occurred" << std::endl;
        throw 69420; // lands here if a subcommand couldn't be parsed, but should stop at 'CLI11_PARSE' anyways
    }

    args.parsed_subcommand = parsed_subcommand;
    return 0;
}
