#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <torch/data/dataloader.h>
#include <utility>

#include <torch/torch.h>

#include "mnist/utils/utils.hpp"

namespace mnist::data {

// referenced lots of the implementation of torch::data::make_data_loader

template <typename Dataset,
          typename Sampler = torch::data::samplers::RandomSampler>
std::unique_ptr<torch::data::StatelessDataLoader<Dataset, Sampler>>
make_data_loader(Dataset &&dataset, const mnist::utils::Config &config) {
    if (dataset.size().has_value() == false) {
        throw std::runtime_error(
            "Dataset size is unknown. Cannot create DataLoader.");
    }

    auto options = torch::data::DataLoaderOptions()
                       .drop_last(config.drop_last)
                       .workers(config.num_workers);

    if (config.batch_size > dataset.size()) {
        options.batch_size(dataset.size().value());
        std::cerr << "Warning: batch_size is larger than dataset size. "
                     "Setting batch_size to dataset size: "
                  << options.batch_size() << "\n";
    } else {
        options.batch_size(config.batch_size);
    }

    auto sampler = Sampler(dataset.size().value());

    return std::make_unique<torch::data::StatelessDataLoader<Dataset, Sampler>>(
        std::move(dataset), std::move(sampler), options);
}

} // namespace mnist::data
