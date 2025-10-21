#pragma once

#include <torch/torch.h>

namespace mnist::model {

class SimpleNet : public torch::nn::Module {
  public:
    explicit SimpleNet();

    torch::Tensor forward(torch::Tensor x);

    void print_weights() const;

  private:
    static constexpr int kFlattenDim = 32 * 7 * 7;
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Linear fc{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
};

} // namespace mnist::model
