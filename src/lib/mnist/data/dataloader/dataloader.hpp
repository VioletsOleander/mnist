#pragma once

#include <memory>

#include <torch/torch.h>

#include "mnist/data/dataset/dataset.hpp"
#include "mnist/utils/utils.hpp"

namespace mnist::data {

// referenced lots of the implementation of torch::data::make_data_loader

template <typename Sampler = torch::data::samplers::RandomSampler>
std::unique_ptr<torch::data::StatelessDataLoader<MNISTDataset, Sampler>>
make_data_loader(MNISTDataset &&dataset, const mnist::utils::Config &config) {
    if (dataset.size().has_value() == false) {
        throw std::runtime_error(
            "Dataset size is unknown. Cannot create DataLoader.");
    }

    auto options = torch::data::DataLoaderOptions()
                       .drop_last(config.drop_last)
                       .workers(config.num_workers);

    if (config.batch_size >= dataset.size()) {
        options.batch_size(dataset.size().value());
        std::cout << "Warning: batch_size is larger than dataset size. "
                     "Setting batch_size to dataset size: "
                  << options.batch_size() << "\n";
    } else {
        options.batch_size(config.batch_size);
    }

    auto sampler = Sampler(dataset.size().value());

    return std::make_unique<
        torch::data::StatelessDataLoader<MNISTDataset, Sampler>>(
        std::move(dataset), std::move(sampler), options);
}

} // namespace mnist::data
