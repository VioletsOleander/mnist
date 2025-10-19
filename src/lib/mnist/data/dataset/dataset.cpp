#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "dataset.hpp"
#include "mnist/utils/utils.hpp"

constexpr size_t MNIST_IMAGE_WIDTH = 28;
constexpr size_t MNIST_IMAGE_HEIGHT = 28;
constexpr size_t MNIST_IMAGE_SIZE = MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT;

namespace fs = std::filesystem;

/// Declaration of class MNISTRawDataset

namespace mnist::data::internal {

class MNISTRawDataset {
  public:
    explicit MNISTRawDataset(const fs::path &dataset_path,
                             const mnist::utils::Mode &mode);

    torch::Tensor construct_image_tensor() const;
    torch::Tensor construct_label_tensor() const;

    size_t get_num_samples() const;

    void print(bool verbose) const;

  private:
    size_t num_images_;
    size_t num_labels_;

    std::vector<uint8_t> image_buffer_;
    std::vector<uint8_t> label_buffer_;

    mnist::utils::Mode mode_;
};

} // namespace mnist::data::internal

/// Implementation of class MNISTRawDataset

namespace mnist::data::internal {

std::pair<fs::path, fs::path> get_file_paths(const fs::path &dataset_path,
                                             const mnist::utils::Mode &mode) {
    std::string mode_str{mnist::utils::mode_to_string(mode)};

    return {dataset_path / mode_str / "images.bin",
            dataset_path / mode_str / "labels.txt"};
}

MNISTRawDataset::MNISTRawDataset(const fs::path &dataset_path,
                                 const mnist::utils::Mode &mode)
    : num_images_(0), num_labels_(0) {
    auto file_paths = get_file_paths(dataset_path, mode);

    std::ifstream image_file(file_paths.first, std::ios::binary);
    if (!image_file.is_open()) {
        throw std::runtime_error("Failed to open image file: " +
                                 file_paths.first.string());
    }

    image_file.seekg(0, std::ios::end);
    size_t file_size = image_file.tellg();
    image_file.seekg(0, std::ios::beg);

    assert(file_size % MNIST_IMAGE_SIZE == 0 &&
           "Corrupted MNIST image file detected");

    num_images_ = file_size / MNIST_IMAGE_SIZE;

    image_buffer_.resize(file_size);
    image_file.read(reinterpret_cast<char *>(image_buffer_.data()), file_size);
    image_file.close();

    // TODO: optimize label loading
    std::ifstream label_file(file_paths.second, std::ios::in);

    label_buffer_.reserve(num_images_);
    for (std::string line; std::getline(label_file, line);) {
        num_labels_++;
        label_buffer_.emplace_back(static_cast<uint8_t>(std::stoi(line)));
    }

    assert(num_images_ == num_labels_ &&
           "Number of images and labels do not match");
}

torch::Tensor MNISTRawDataset::construct_image_tensor() const {
    torch::TensorOptions options =
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    torch::Tensor image_tensor =
        torch::from_blob((void *)image_buffer_.data(),
                         {static_cast<int64_t>(num_images_), 1,
                          static_cast<int64_t>(MNIST_IMAGE_HEIGHT),
                          static_cast<int64_t>(MNIST_IMAGE_WIDTH)},
                         options)
            .clone()              // clone to own the memory
            .to(torch::kFloat32); // convert to float32

    image_tensor.div_(255.0f); // normalize to [0, 1]

    return image_tensor;
}

torch::Tensor MNISTRawDataset::construct_label_tensor() const {
    torch::TensorOptions options =
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    torch::Tensor label_tensor =
        torch::from_blob((void *)label_buffer_.data(),
                         {static_cast<int64_t>(num_labels_)}, options)
            .clone(); // clone to own the memory

    return label_tensor;
}

size_t MNISTRawDataset::get_num_samples() const { return num_images_; }

void MNISTRawDataset::print(bool verbose) const {
    std::cout << "Mode: " << utils::mode_to_string(mode_) << "\n";
    std::cout << "Number of samples: " << num_images_ << "\n";
    std::cout << "Image buffer size: " << image_buffer_.size() << " bytes\n";
    std::cout << "Label buffer size: " << label_buffer_.size() << " bytes\n";
}

} // namespace mnist::data::internal

/// Implementation of class MNISTDataset

namespace mnist::data {

MNISTDataset::MNISTDataset(const std::filesystem::path &dataset_path,
                           const mnist::utils::Mode &mode) {
    raw_dataset_ =
        std::make_shared<internal::MNISTRawDataset>(dataset_path, mode);

    image_tensor_ = raw_dataset_->construct_image_tensor();
    label_tensor_ = raw_dataset_->construct_label_tensor();
    num_samples_ = raw_dataset_->get_num_samples();
}

torch::data::Example<torch::Tensor, torch::Tensor>
MNISTDataset::get(size_t index) {
    torch::Tensor image = image_tensor_[index].clone();
    torch::Tensor label = label_tensor_[index].clone();

    return {image, label};
}

torch::optional<size_t> MNISTDataset::size() const { return num_samples_; }

void MNISTDataset::print(bool verbose) const {
    std::cout << "MNIST Dataset Info:\n";
    raw_dataset_->print(verbose);

    std::cout << "\nImage Tensor Info:\n";
    std::cout << "Device and Shape:";
    image_tensor_.print();
    std::cout << "Image Tensor Min: " << image_tensor_.min() << "\n";
    std::cout << "Image Tensor Max: " << image_tensor_.max() << "\n";

    std::cout << "\nLabel Tensor Info:\n";
    std::cout << "Device and Shape: ";
    label_tensor_.print();
    std::cout << "Labels:\n"
              << std::get<0>(torch::_unique(label_tensor_)) << "\n";
}

MNISTDataset::MNISTDataset(const MNISTDataset &other) = default;
MNISTDataset &MNISTDataset::operator=(const MNISTDataset &other) = default;
MNISTDataset::MNISTDataset(MNISTDataset &&other) noexcept = default;
MNISTDataset &MNISTDataset::operator=(MNISTDataset &&other) noexcept = default;

MNISTDataset::~MNISTDataset() = default;

} // namespace mnist::data
