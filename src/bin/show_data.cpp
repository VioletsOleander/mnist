#include <filesystem>
#include <string>

#include <toml++/toml.h>

#include <mnist/mnist.h>

using namespace mnist;

int main() {
    auto dataset_path = std::filesystem::path("../dataset/mnist");
    auto mode = utils::Mode::TRAIN;

    auto dataset = data::MNISTDataset(dataset_path, mode);

    std::cout << "Dataset schema: " << dataset.schema() << std::endl;

    return 0;
}