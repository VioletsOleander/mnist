#pragma once

#include <filesystem>

#include <toml++/toml.h>

namespace CLI {

class App;

}

#ifndef MNIST_PARSE_OR_EXIT
#define MNIST_PARSE_OR_EXIT(config, app, argc, argv)                           \
    try {                                                                      \
        parse_args(config, app, argc, argv);                                   \
    } catch (const CLI::ParseError &err) {                                     \
        return app.exit(err);                                                  \
    } catch (const toml::parse_error &err) {                                   \
        std::cerr << "Parsing failed:\n" << err << "\n";                       \
        return 1;                                                              \
    } catch (const std::runtime_error &err) {                                  \
        std::cerr << "Runtime error: " << err.what() << "\n";                  \
        return 1;                                                              \
    } catch (const std::logic_error &err) {                                    \
        std::cerr << "Logic error: " << err.what() << "\n";                    \
        return 1;                                                              \
    }
#endif

namespace mnist::utils {

enum class Mode { TRAIN, TEST };

struct Config {
    std::filesystem::path dataset_path;
    Mode mode;
};

void parse_args(Config &config, CLI::App &app, int argc, char *argv[]);

} // namespace mnist::utils