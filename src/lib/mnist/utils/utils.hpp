#pragma once

#include <filesystem>
#include <string>

namespace CLI {

class App;

}

namespace mnist::utils {

enum class Mode { TRAIN, TEST };

std::string mode_to_string(Mode mode);

struct Config {
    std::filesystem::path dataset_path;
    Mode mode;
    size_t batch_size;
    bool drop_last;
};

int parse_args(Config &config, CLI::App &app, int argc, char *argv[]);

} // namespace mnist::utils