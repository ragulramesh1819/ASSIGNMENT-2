#ifndef CONV2D_H
#define CONV2D_H

#include <vector>

// Function to perform 2D convolution
std::vector<std::vector<std::vector<float>>> conv2D(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& weights,
    const std::vector<float>& biases,
    int stride = 1,
    int padding = 0);

#endif // CONV2D_H
