#include <string>

#include <CLI/CLI.hpp>
#include <toml++/toml.h>

#include "utils.hpp"

namespace mnist::utils {

void parse_args(Config &config, CLI::App &app, int argc, char *argv[]) {
    std::string config_path;
    app.add_option("config_path", config_path, "Path to the configuration file")
        ->required()
        ->check(CLI::ExistingFile);

    app.parse(argc, argv);

    toml::table tbl;
    tbl = toml::parse_file(config_path);

    std::string dataset_path = tbl["dataset_path"].value_or<std::string>("");
    if (dataset_path.empty()) {
        throw std::runtime_error("dataset_path is required in the config file");
    }

    std::string mode = tbl["mode"].value_or<std::string>("");
    if (mode.empty()) {
        throw std::runtime_error("mode is required in the config file");
    } else if (mode != "train" && mode != "test") {
        throw std::runtime_error("mode must be either 'train' or 'test'");
    }

    config.dataset_path = std::filesystem::path(dataset_path);
    config.mode = mode == "train" ? Mode::TRAIN : Mode::TEST;
}

} // namespace mnist::utils