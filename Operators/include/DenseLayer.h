#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

class DenseLayer {
public:

    // Parameterized constructor
    DenseLayer(const std::vector<float>& input_data,
               const std::string& weights_file,
               const std::string& bias_file,
               const std::vector<int>& input_shape,
               const std::vector<int>& output_shape,
               const std::string& activation_type);

    // Method declarations
    void load_weights_and_bias(const std::string& weights_file,
                                const std::string& bias_file);
    void forward();
    void apply_activation();
    void relu();
    void softmax();
    const std::vector<float>& get_output() const;

private:
    // Private member variables
    std::vector<float> input_data_;
    std::vector<float> output_data_;
    std::vector<float> weights_;
    std::vector<float> bias_;
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
    std::string activation_type_;
};

#endif // DENSELAYER_H
