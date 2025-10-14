#include <arrow/array/array_decimal.h>
#include <arrow/chunked_array.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <filesystem>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <cassert>
#include <cstring> // for std::memcpy
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <stdexcept>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include "dataset.hpp"
#include "mnist/utils/utils.hpp"

namespace fs = std::filesystem;
namespace utils = mnist::utils;

namespace {

// Helper function to get train and test parquet file paths from the dataset
// directory
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

// Helper function to read a Parquet file and return an Arrow Table
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

} // namespace

namespace mnist::data {

MNISTDataset::MNISTDataset(const fs::path &dataset_path,
                           const utils::Mode &mode) {
    /// Read the Parquet file
    auto file_paths = get_parquet_file_paths(dataset_path);

    table_ = read_parquet(file_paths[mode]);

    auto image_column = table_->GetColumnByName("image");
    auto label_column = table_->GetColumnByName("label");

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
    auto image_array = result.ValueOrDie()[0]->chunk(0);
    auto label_array = label_column->chunk(0);

    assert(image_array->type_id() == arrow::Type::BINARY &&
           "Image array type must be BINARY");
    assert(label_array->type_id() == arrow::Type::INT64 &&
           "Label array type must be INT64");

    /// Convert image array to torch tensor
    {
        // each item in image array is assume to be an (28 * 28) array of uint8
        // since MNIST images are 28x28 pixels, and each pixel is represented
        // as a gray scale value from 0 to 255
        const int64_t num_imgs = image_array->length();
        constexpr int32_t img_size = 28 * 28;

        auto binary_array =
            std::static_pointer_cast<arrow::BinaryArray>(image_array);

        // ensure each entry has expected length
        for (int64_t i = 0; i < num_imgs; ++i) {
            if (binary_array->value_length(i) != img_size) {
                std::cout << binary_array->value_length(i) << "\n";
                throw std::runtime_error("Binary entry size != 28*28");
            }
        }

        // total bytes for all images
        int64_t start_off = binary_array->value_offset(0);
        int64_t end_off = binary_array->value_offset(num_imgs);
        int64_t total_bytes = end_off - start_off;
        if (total_bytes != num_imgs * img_size) {
            throw std::runtime_error("Unexpected total image bytes");
        }

        image_tensor_ =
            torch::empty({(long)num_imgs, 1, 28, 28}, torch::kUInt8);

        auto values_buf = binary_array->value_data();
        const uint8_t *src = values_buf->data() + start_off;

        std::memcpy(image_tensor_.mutable_data_ptr(), src,
                    static_cast<size_t>(total_bytes));

        // convert to float and normalize
        image_tensor_ =
            image_tensor_.to(torch::kFloat32).div(255.0).contiguous();
    }

    /// Convert label array to torch tensor
    {
        const int64_t num_labels = label_array->length();

        auto int_array =
            std::static_pointer_cast<arrow::Int64Array>(label_array);

        const int64_t *src = int_array->raw_values();

        label_tensor_ = torch::empty({(long)num_labels}, torch::kInt64);

        std::memcpy(label_tensor_.mutable_data_ptr(), src,
                    static_cast<size_t>(num_labels * sizeof(int64_t)));
    }
}

torch::data::Example<torch::Tensor, torch::Tensor>
MNISTDataset::get(size_t index) {
    auto img = image_tensor_.select(0, static_cast<long>(index));
    auto lbl = label_tensor_.select(0, static_cast<long>(index));
    return {img, lbl};
}

torch::optional<size_t> MNISTDataset::size() const {
    return image_tensor_.size(0);
}

void MNISTDataset::print_table_info() const {
    std::cout << "Schema:\n" << table_->schema()->ToString(true) << "\n";

    auto image_column = table_->GetColumnByName("image");
    std::cout << "\nImage Column:\n"
              << "Num Chunks: " << image_column->num_chunks() << "\n"
              << "Length: " << image_column->length() << "\n"
              << "Type: " << image_column->type()->ToString() << "\n"
              << "String: " << image_column->ToString() << "\n";

    auto label_column = table_->GetColumnByName("label");
    std::cout << "\nLabel Column:\n"
              << "Num Chunks: " << label_column->num_chunks() << "\n"
              << "Length: " << label_column->length() << "\n"
              << "Type: " << label_column->type()->ToString() << "\n"
              << "String: " << label_column->ToString() << "\n";
}

void MNISTDataset::print_tensor_info() const {
    std::cout << "\nImage Tensor:\n"
              << "Size: " << image_tensor_.sizes() << "\n"
              << "Type: " << image_tensor_.dtype() << "\n"
              << "String: " << image_tensor_.toString() << "\n";

    std::cout << "\nLabel Tensor:\n"
              << "Size: " << label_tensor_.sizes() << "\n"
              << "Type: " << label_tensor_.dtype() << "\n"
              << "String: " << label_tensor_.toString() << "\n";
}

} // namespace mnist::data