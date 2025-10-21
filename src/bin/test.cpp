#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>

#include <CLI/CLI.hpp>
#include <mnist/mnist.hpp>
#include <torch/torch.h>

int main(int argc, char *argv[]) {
    CLI::App app{"MNIST Evaluation Application"};
    mnist::utils::Config config;

    if (int result = mnist::utils::parse_args(config, app, argc, argv);
        result != 0) {
        return result;
    }

    auto net = std::make_shared<mnist::model::SimpleNet>();
    std::cout << "Model structure:\n" << *net << "\n";

    torch::load(net, config.model_path.string());
    std::cout << "Model weights loaded from " << config.model_path << "\n";

    auto dataset = mnist::data::MNISTDataset(config.dataset_path, config.mode);
    auto dataloader = mnist::data::make_data_loader(
        std::move(dataset.map(torch::data::transforms::Stack<>())), config);

    int64_t correct_samples = 0;
    int64_t total_samples = 0;

    net->eval();
    std::cout << "Starting evaluation...\n";
    {
        torch::NoGradGuard no_grad;
        for (auto &batch : *dataloader) {
            auto prediction = net->forward(batch.data);
            auto predicted_labels = prediction.argmax(1);
            auto correct =
                predicted_labels.eq(batch.target).sum().item<int64_t>();
            total_samples += batch.target.size(0);
            correct_samples += correct;
        }
    }
    if (total_samples == 0) {
        std::cout << "No samples to evaluate, accuracy is 0%.\n";
    } else {
        float accuracy = static_cast<float>(correct_samples) /
                         static_cast<float>(total_samples);
        std::cout << "Accuracy = " << correct_samples << " / " << total_samples
                  << " = " << accuracy * 100.0f << "%\n";
    }

    return 0;
}
