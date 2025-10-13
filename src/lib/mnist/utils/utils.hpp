#pragma once

#include <filesystem>

namespace CLI {

class App;

}

namespace mnist::utils {

enum class Mode { TRAIN, TEST };

struct Config {
    std::filesystem::path dataset_path;
    Mode mode;
};

int parse_args(Config &config, CLI::App &app, int argc, char *argv[]);

} // namespace mnist::utils