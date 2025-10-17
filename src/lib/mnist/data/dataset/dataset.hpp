#pragma once

#include <filesystem>
#include <memory>

#include <torch/torch.h>

namespace mnist::utils {

enum class Mode;

}

namespace mnist::data {

namespace internal {

class MNISTRawDataset;

} // namespace internal

class MNISTDataset : public torch::data::Dataset<MNISTDataset> {
  public:
    explicit MNISTDataset(const std::filesystem::path &dataset_path,
                          const mnist::utils::Mode &mode);
    ~MNISTDataset() override;

    torch::data::Example<torch::Tensor, torch::Tensor>
    get(size_t index) override;

    torch::optional<size_t> size() const override;

    void print(bool verbose = false) const;

  private:
    torch::Tensor image_tensor_;
    torch::Tensor label_tensor_;

    size_t num_samples_;
    std::unique_ptr<internal::MNISTRawDataset> raw_dataset_;
};

} // namespace mnist::data
