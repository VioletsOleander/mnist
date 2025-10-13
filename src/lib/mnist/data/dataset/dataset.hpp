#pragma once

#include <filesystem>
#include <string>

#include <torch/torch.h>

#include "mnist/utils/utils.hpp"

namespace arrow {

class Table;

}

namespace mnist::data {

class MNISTDataset : public torch::data::Dataset<MNISTDataset> {
  public:
    explicit MNISTDataset(
        const std::filesystem::path &dataset_path,
        const mnist::utils::Mode &mode = mnist::utils::Mode::TRAIN);

    torch::data::Example<torch::Tensor, torch::Tensor>
    get(size_t index) override;

    torch::optional<size_t> size() const override;

    std::string schema() const;

  private:
    torch::Tensor images_;
    torch::Tensor labels_;

    std::shared_ptr<arrow::Table> table_;
};

} // namespace mnist::data