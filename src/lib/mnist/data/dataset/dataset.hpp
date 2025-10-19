#pragma once

#include <cstdint>
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

    // explicit destructor lets the compiler defer generation to the source
    // file, after the definition of MNISTRawDataset (needed for unique_ptr to
    // an incomplete type)
    ~MNISTDataset() override;

    MNISTDataset(const MNISTDataset &);
    MNISTDataset &operator=(const MNISTDataset &);
    MNISTDataset(MNISTDataset &&) noexcept;
    MNISTDataset &operator=(MNISTDataset &&) noexcept;

    /// retrieve a single data sample given index (clone to own the memory)
    torch::data::Example<torch::Tensor, torch::Tensor>
    get(uint64_t index) override;

    torch::optional<uint64_t> size() const override;

    void print(bool verbose = false) const;

  private:
    torch::Tensor image_tensor_;
    torch::Tensor label_tensor_;

    uint64_t num_samples_;
    std::shared_ptr<internal::MNISTRawDataset> raw_dataset_;
};

} // namespace mnist::data
