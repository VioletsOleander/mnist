#include <string>

#include <CLI/CLI.hpp>
#include <mnist/mnist.hpp>

int main(int argc, char *argv[]) {
    CLI::App app{"MNIST Dataset Viewer"};
    mnist::utils::Config config;

    if (int result = mnist::utils::parse_args(config, app, argc, argv);
        result != 0) {
        return result;
    }

    auto dataset = mnist::data::MNISTDataset(config.dataset_path, config.mode);

    dataset.print(true);

    return 0;
}