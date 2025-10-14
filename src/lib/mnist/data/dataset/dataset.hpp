#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "mnist/utils/utils.hpp"

namespace arrow {

class Table;
class Array;

} // namespace arrow

namespace mnist::data {

class MNISTDataset : public torch::data::Dataset<MNISTDataset> {
  public:
    explicit MNISTDataset(
        const std::filesystem::path &dataset_path,
        const mnist::utils::Mode &mode = mnist::utils::Mode::TRAIN);

    torch::data::Example<torch::Tensor, torch::Tensor>
    get(size_t index) override;

    torch::optional<size_t> size() const override;

    void print_table_info() const;
    void print_tensor_info() const;

  private:
    torch::Tensor image_tensor_;
    torch::Tensor label_tensor_;

    std::shared_ptr<const arrow::Table> table_;
};

} // namespace mnist::data