#include "DenseLayer.h"

// Constructor for parameterized DenseLayer
DenseLayer::DenseLayer(const std::vector<float>& input_data,
                       const std::string& weights_file,
                       const std::string& bias_file,
                       const std::vector<int>& input_shape,
                       const std::vector<int>& output_shape,
                       const std::string& activation_type)
     : input_data_(input_data),
      input_shape_(input_shape),
      output_shape_(output_shape),
      activation_type_(activation_type) 
    {
    load_weights_and_bias(weights_file, bias_file);
}

// Load weights and bias from binary files
void DenseLayer::load_weights_and_bias(const std::string& weights_file,
                                       const std::string& bias_file) {
    // Load weights (kernel) from the binary file
    std::ifstream weights_stream(weights_file, std::ios::binary);
    weights_.resize(input_shape_[0] * output_shape_[0]); // weights size: input_shape[0] x output_shape[0]
    weights_stream.read(reinterpret_cast<char*>(weights_.data()), weights_.size() * sizeof(float));
    weights_stream.close();

    // Load bias from the binary file
    std::ifstream bias_stream(bias_file, std::ios::binary);
    bias_.resize(output_shape_[0]); // bias size: output_shape[0]
    bias_stream.read(reinterpret_cast<char*>(bias_.data()), bias_.size() * sizeof(float));
    bias_stream.close();
}

// Perform matrix-vector multiplication and apply activation
void DenseLayer::forward() {
    // Perform matrix-vector multiplication: output = input * weights + bias
    output_data_.resize(output_shape_[0]);
    for (int i = 0; i < output_shape_[0]; ++i) {
        output_data_[i] = 0.0f;
        for (int j = 0; j < input_shape_[0]; ++j) {
            output_data_[i] += input_data_[j] * weights_[i * input_shape_[0] + j];
        }
        output_data_[i] += bias_[i];
    }

    // Apply the activation function
    apply_activation();
}

// Apply the ReLU activation function
void DenseLayer::apply_activation() {
    if (activation_type_ == "relu") {
        relu();
    }
    else if(activation_type_ == "softmax"){
        softmax();
    } else {
        std::cerr << "Unsupported activation type: " << activation_type_ << std::endl;
    }
}

// Apply ReLU activation (output = max(0, input))
void DenseLayer::relu() {
    for (auto& value : output_data_) {
        value = std::max(0.0f, value);
    }
}

void DenseLayer::softmax() {
    float sum_exp = 0.0f;
    float max_value = *std::max_element(output_data_.begin(), output_data_.end());

     for (const auto& value : output_data_) {
        sum_exp += std::exp(value - max_value);  // Subtract max_value to avoid overflow
    }

    // Normalize each value to get the softmax result
    for (auto& value : output_data_) {
        value = std::exp(value - max_value) / sum_exp;  // Subtract max_value in the denominator as well
    }
    
    // float max_softmax = *std::max_element(output_data_.begin(), output_data_.end());

    // for (auto& value : output_data_) {
    //     value = (value == max_softmax) ? 1.0f : 0.0f;
    // }
}

// Getter for the output data
const std::vector<float>& DenseLayer::get_output() const {
    return output_data_;
}
