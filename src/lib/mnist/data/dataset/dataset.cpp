#include <filesystem>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
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
    auto file_paths = get_parquet_file_paths(dataset_path);

    table_ = read_parquet(file_paths[mode]);

    //     // Extract image and label columns
    //     auto image_column = table->GetColumnByName("image");
    //     auto label_column = table->GetColumnByName("label");

    //     if (!image_column || !label_column) {
    //         throw std::runtime_error(
    //             "Parquet file must contain 'image' and 'label' columns.");
    //     }

    //     // Process images
    //     auto image_chunks = image_column->chunks();
    //     std::vector<uint8_t> image_data;
    //     long num_rows = table->num_rows();

    //     // Assuming images are stored as structs of bytes
    //     for (const auto &chunk : image_chunks) {
    //         auto struct_array =
    //         std::static_pointer_cast<arrow::StructArray>(chunk); auto
    //         byte_array_ptr = struct_array->GetFieldByName("bytes"); auto
    //         byte_array =
    //             std::static_pointer_cast<arrow::BinaryArray>(byte_array_ptr);

    //         for (int64_t i = 0; i < byte_array->length(); ++i) {
    //             auto value = byte_array->Value(i);
    //             image_data.insert(image_data.end(), value.begin(),
    //             value.end());
    //         }
    //     }

    //     images_ =
    //         torch::from_blob(image_data.data(), {num_rows, 1, 28, 28},
    //         torch::kU8)
    //             .clone();
    //     images_ = images_.to(torch::kFloat32).div(255.0);

    //     // Process labels
    //     auto label_chunks = label_column->chunks();
    //     std::vector<int64_t> label_data;
    //     for (const auto &chunk : label_chunks) {
    //         auto int_array =
    //         std::static_pointer_cast<arrow::Int64Array>(chunk); for (int64_t
    //         i = 0; i < int_array->length(); ++i) {
    //             label_data.push_back(int_array->Value(i));
    //         }
    //     }

    //     labels_ = torch::from_blob(label_data.data(),
    //     {(long)label_data.size()},
    //                                torch::kInt64)
    //                   .clone();
    // }

    // torch::data::Example<torch::Tensor, torch::Tensor>
    // MNISTDataset::get(size_t index) {
    //     return {images_[index], labels_[index]};
    // }

    // torch::optional<size_t> MNISTDataset::size() const { return
    // images_.size(0); }
}

torch::data::Example<torch::Tensor, torch::Tensor>
MNISTDataset::get(size_t index) {
    return {images_[index], labels_[index]};
}

torch::optional<size_t> MNISTDataset::size() const { return images_.size(0); }

std::string MNISTDataset::schema() const {
    return table_->schema()->ToString(true);
}

} // namespace mnist::data