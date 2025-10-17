#pragma once

#include <filesystem>
#include <memory>

#include <torch/torch.h>

#include "mnist/utils/utils.hpp"

namespace arrow {

class Table;
class Array;

} // namespace arrow

namespace mnist::data {

namespace internal {

class MNISTRawDataset;

} // namespace internal

class MNISTDataset : public torch::data::Dataset<MNISTDataset> {
  public:
    explicit MNISTDataset(
        const std::filesystem::path &dataset_path,
        const mnist::utils::Mode &mode = mnist::utils::Mode::TRAIN);
    ~MNISTDataset() override;

    torch::data::Example<torch::Tensor, torch::Tensor>
    get(size_t index) override;

    torch::optional<size_t> size() const ov]erride;

    void print(bool verbose = false) const;

  private:
    torch::Tensor image_tensor_;
    torch::Tensor label_tensor_;

    std::unique_ptr<internal::MNISTRawDataset> raw_dataset_;
};

} // namespace mnist::data
