#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <parquet/arrow/reader.h>

#include "dataset.hpp"
#include "mnist/utils/utils.hpp"

namespace fs = std::filesystem;
namespace utils = mnist::utils;

/// Declaration of class MNISTRawDataset

namespace mnist::data::internal {

class MNISTRawDataset {
  public:
    explicit MNISTRawDataset(const fs::path &dataset_path,
                             const utils::Mode &mode);
    void print(bool verbose) const;

  private:
    std::shared_ptr<arrow::Array> image_array_;
    std::shared_ptr<arrow::Array> label_array_;
};

} // namespace mnist::data::internal

/// Implementation of class MNISTRawDataset

namespace mnist::data::internal {

// Get train and test parquet file paths from the dataset directory
std::map<utils::Mode, fs::path>
get_parquet_file_paths(const fs::path &dataset_path) {
    std::map<utils::Mode, fs::path> file_paths;
    for (const auto &entry : fs::directory_iterator(dataset_path)) {
        if (entry.path().extension() == ".parquet") {
            auto filename = entry.path().filename().string();
            if (filename.find("train", 0) == 0) {
                file_paths[utils::Mode::TRAIN] = entry.path();
            } else if (filename.find("test", 0) == 0) {
                file_paths[utils::Mode::TEST] = entry.path();
            }
        }
    }
    return file_paths;
}

// Read a parquet file and return an arrow table
std::shared_ptr<arrow::Table> read_parquet(const std::string &file_path) {
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(file_path));

    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_ASSIGN_OR_THROW(
        reader, parquet::arrow::OpenFile(infile, arrow::default_memory_pool()));

    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
    return table;
}

MNISTRawDataset::MNISTRawDataset(const fs::path &dataset_path,
                                 const utils::Mode &mode) {
    auto file_paths = get_parquet_file_paths(dataset_path);
    auto table = read_parquet(file_paths[mode]);

    auto image_column = table->GetColumnByName("image");
    auto label_column = table->GetColumnByName("label");
    if (!image_column || !label_column) {
        throw std::runtime_error(
            "Parquet file must contain 'image' and 'label' columns.");
    }

    auto result = image_column->Flatten();
    if (!result.ok()) {
        throw std::runtime_error("Error flattening image column: " +
                                 result.status().ToString());
    }

    // image and label ChunkedArray are assumed to have only one chunk
    image_array_ = result.ValueOrDie()[0]->chunk(0);
    label_array_ = label_column->chunk(0);
    assert(image_array_->type_id() == arrow::Type::BINARY &&
           "Image array type must be BINARY");
    assert(label_array_->type_id() == arrow::Type::INT64 &&
           "Label array type must be INT64");

    auto image_array_data = image_array_->data();
    auto label_array_data = label_array_->data();
}

void MNISTRawDataset::print(bool verbose) const {
    std::cout << "Image array:\n"
              << "length: " << image_array_->length() << "\n"
              << "type: " << image_array_->type()->ToString()
              << ", with id: " << image_array_->type()->id()
              << ", with byte width: " << image_array_->type()->byte_width()
              << ", with bit width: " << image_array_->type()->bit_width()
              << "\n";
    std::cout << "\nLabel array:\n"
              << "length: " << label_array_->length() << "\n"
              << "type: " << label_array_->type()->ToString()
              << ", with id: " << label_array_->type()->id()
              << ", with byte width: " << label_array_->type()->byte_width()
              << ", with bit width: " << label_array_->type()->bit_width()
              << "\n";

    if (verbose) {
        auto image_array_buffers = image_array_->data()->buffers;
        auto label_array_buffers = label_array_->data()->buffers;

        std::cout << "\nImage array buffers:\n"
                  << "length: " << image_array_buffers.size() << "\n";
        std::cout << "\nLabel array buffers:\n"
                  << "length: " << label_array_buffers.size() << "\n";

        for (size_t i = 0; i < image_array_buffers.size(); i++) {
            if (image_array_buffers[i]) {
                std::cout << "\nImage array buffer " << i << ":\n"
                          << "size: " << image_array_buffers[i]->size() << "\n"
                          << "capacity: " << image_array_buffers[i]->capacity()
                          << "\n";
            } else {
                std::cout << "\nImage array buffer " << i << " is null.\n";
            }
        }

        for (size_t i = 0; i < label_array_buffers.size(); i++) {
            if (label_array_buffers[i]) {
                std::cout << "\nLabel array buffer " << i << ":\n"
                          << "size: " << label_array_buffers[i]->size() << "\n"
                          << "capacity: " << label_array_buffers[i]->capacity()
                          << "\n";
            } else {
                std::cout << "\nLabel array buffer " << i << " is null.\n";
            }
        }
    }
}

} // namespace mnist::data::internal

/// Implementation of class MNISTDataset

namespace mnist::data {

MNISTDataset::MNISTDataset(const std::filesystem::path &dataset_path,
                           const mnist::utils::Mode &mode) {
    raw_dataset_ =
        std::make_unique<internal::MNISTRawDataset>(dataset_path, mode);
}

torch::data::Example<torch::Tensor, torch::Tensor>
MNISTDataset::get(size_t index) {
    throw std::runtime_error("Not implemented yet");
}

torch::optional<size_t> MNISTDataset::size() const {
    throw std::runtime_error("Not implemented yet");
}

void MNISTDataset::print(bool verbose) const { raw_dataset_->print(verbose); }

MNISTDataset::~MNISTDataset() = default;

} // namespace mnist::data