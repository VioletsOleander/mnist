#include <iostream>
#include <string>

#include <CLI/CLI.hpp>
#include <toml++/toml.h>

#include "utils.hpp"

namespace mnist::utils {

int parse_args(Config &config, CLI::App &app, int argc, char *argv[]) {
    std::string config_path;
    app.add_option("config_path", config_path, "Path to the configuration file")
        ->required()
        ->check(CLI::ExistingFile);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &err) {
        // passing '-h' is identified as ParseError: CallForHelp and yield exit
        // code 0, so special handling is needed
        if (err.get_exit_code() == 0) {
            app.exit(err);
            return 1;
        } else {
            return app.exit(err);
        }
    }

    toml::table tbl;
    try {
        tbl = toml::parse_file(config_path);
    } catch (const toml::parse_error &err) {
        std::cerr << "Parsing failed: " << err << "\n";
        return 1;
    }

    std::string dataset_path = tbl["dataset_path"].value_or<std::string>("");
    if (dataset_path.empty()) {
        std::cerr
            << "Parsing error: dataset_path is required in the config file "
               "and should be a valid string\n";
        return 1;
    }

    std::string mode = tbl["mode"].value_or<std::string>("");
    if (mode.empty()) {
        std::cerr << "Parsing error: mode is required in the config file and "
                     "should be a valid string\n";
        return 1;
    } else if (mode != "train" && mode != "test") {
        std::cerr << "Parsing error: mode must be either 'train' or 'test'\n";
        return 1;
    }

    config.dataset_path = std::filesystem::path(dataset_path);
    config.mode = mode == "train" ? Mode::TRAIN : Mode::TEST;

    return 0;
}

} // namespace mnist::utils