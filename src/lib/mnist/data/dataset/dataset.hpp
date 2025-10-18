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

    // explicit destructor let compiler defer the generation of destructor in
    // the source file, that is, after the declaration of MNISTRawDataset
    ~MNISTDataset() override;

    // since we have unique_ptr member, we delete copy constructor and copy
    // assignment operator, and use default move constructor and move assignment
    MNISTDataset(const MNISTDataset &) = delete;
    MNISTDataset &operator=(const MNISTDataset &) = delete;
    MNISTDataset(MNISTDataset &&) noexcept = default;
    MNISTDataset &operator=(MNISTDataset &&) noexcept = default;

    /// retrieve a single data sample given index (clone to own the memory)
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
