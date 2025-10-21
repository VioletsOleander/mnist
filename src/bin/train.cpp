#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>

#include <CLI/CLI.hpp>
#include <mnist/mnist.hpp>
#include <torch/torch.h>

int main(int argc, char *argv[]) {
    CLI::App app{"MNIST Training Application"};
    mnist::utils::Config config;

    if (int result = mnist::utils::parse_args(config, app, argc, argv);
        result != 0) {
        return result;
    }

    auto net = std::make_shared<mnist::model::SimpleNet>();
    std::cout << "Model structure:\n" << *net << "\n";

    auto dataset = mnist::data::MNISTDataset(config.dataset_path, config.mode);
    auto dataloader = mnist::data::make_data_loader(
        std::move(dataset.map(torch::data::transforms::Stack<>())), config);

    // the hyper parameter is simply determined by tab completion
    auto optim = torch::optim::SGD(
        net->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));

    net->train();
    std::cout << "Starting training for " << config.epochs << " epochs...\n";
    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        size_t batch_index = 0;
        for (auto &batch : *dataloader) {
            optim.zero_grad();
            auto prediction = net->forward(batch.data);
            auto loss =
                torch::nn::functional::cross_entropy(prediction, batch.target);
            loss.backward();
            optim.step();

            if (batch_index++ % 10 == 0) {
                std::cout << "Epoch: " << epoch
                          << " | Batch: " << batch_index - 1
                          << " | Loss: " << loss.item<float>() << "\n";
            }
        }
    }
    std::cout << "Training complete.\n";

    torch::save(net, config.model_path.string());
    std::cout << "Model saved to " << config.model_path << "\n";

    return 0;
}
