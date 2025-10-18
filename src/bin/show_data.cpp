#include <memory>
#include <string>

#include <CLI/CLI.hpp>
#include <mnist/mnist.hpp>
#include <torch/torch.h>

int main(int argc, char *argv[]) {
    CLI::App app{"MNIST Dataset Viewer"};
    mnist::utils::Config config;

    if (int result = mnist::utils::parse_args(config, app, argc, argv);
        result != 0) {
        return result;
    }

    auto dataset = mnist::data::MNISTDataset(config.dataset_path, config.mode);

    dataset.print(true);

    auto dataloader = mnist::data::make_data_loader(std::move(dataset), config);

    for (auto batch : *dataloader) {
        std::cout << "\nBatch size: " << batch.size() << "\n";
        std::cout << "\nFirst sample in the batch:\n";
        std::cout << "\nImage Tensor:\n";
        std::cout << batch[0].data << "\n";
        std::cout << "\nLabel Tensor:\n";
        std::cout << batch[0].target << "\n";
        break; // only show the first batch
    }

    return 0;
}
