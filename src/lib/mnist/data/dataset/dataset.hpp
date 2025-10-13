#pragma once

#include <arrow/table.h>
#include <filesystem>
#include <string>

#include <arrow/api.h>
#include <torch/torch.h>

#include "mnist/utils/utils.hpp"

namespace mnist::data {

class MNISTDataset : public torch::data::Dataset<MNISTDataset> {
  public:
    explicit MNISTDataset(
        const std::filesystem::path &dataset_path,
        const mnist::utils::Mode &mode = mnist::utils::Mode::TRAIN);

    torch::data::Example<torch::Tensor, torch::Tensor>
    get(size_t index) override;

    torch::optional<size_t> size() const override;

    std::string schema() const { return table_->schema()->ToString(true); };

  private:
    torch::Tensor images_;
    torch::Tensor labels_;

    std::shared_ptr<arrow::Table> table_;
};

} // namespace mnist::data