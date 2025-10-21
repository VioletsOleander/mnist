#include <iostream>

#include <torch/torch.h>

#include "model.hpp"

namespace mnist::model {

// the network structure and forward function here are just determined by tab
// completion

SimpleNet::SimpleNet()
    : conv1(torch::nn::Conv2dOptions(1, 16, /*kernel_size=*/3)
                .stride(1)
                .padding(1)),
      conv2(torch::nn::Conv2dOptions(16, 32, /*kernel_size=*/3)
                .stride(1)
                .padding(1)),
      fc(7 * 7 * 32, 10),
      pool(torch::nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("fc", fc);
    register_module("pool", pool);

    // Initialize weights and biases
    torch::nn::init::kaiming_normal_(conv1->weight);
    torch::nn::init::constant_(conv1->bias, 0.0);

    torch::nn::init::kaiming_normal_(conv2->weight);
    torch::nn::init::constant_(conv2->bias, 0.0);

    torch::nn::init::xavier_normal_(fc->weight);
    torch::nn::init::constant_(fc->bias, 0.0);
}

torch::Tensor SimpleNet::forward(torch::Tensor x) {
    // Input shape: [batch, 1, 28, 28]
    x = torch::relu(conv1(x));    // [batch, 16, 28, 28]
    x = pool(x);                  // [batch, 16, 14, 14]
    x = torch::relu(conv2(x));    // [batch, 32, 14, 14]
    x = pool(x);                  // [batch, 32, 7, 7]
    x = x.view({-1, 32 * 7 * 7}); // Flatten to [batch, 1568]
    x = fc(x);                    // [batch, 10] (raw logits)
    return x;
}

void SimpleNet::print_weights() {
    for (const auto &param : this->named_parameters()) {
        std::cout << param.key() << " - Shape: " << param.value().sizes()
                  << "\n"
                  << param.value() << "\n\n";
    }
}

} // namespace mnist::model
