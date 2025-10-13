#include <filesystem>
#include <string>

#include <CLI/CLI.hpp>
#include <mnist/mnist.hpp>
#include <toml++/toml.h>

int main(int argc, char *argv[]) {
    CLI::App app{"MNIST Dataset Viewer"};
    mnist::utils::Config config;
    MNIST_PARSE_OR_EXIT(config, app, argc, argv)

    auto dataset = mnist::data::MNISTDataset(config.dataset_path, config.mode);

    std::cout << "Dataset schema: " << dataset.schema() << std::endl;

    return 0;
}