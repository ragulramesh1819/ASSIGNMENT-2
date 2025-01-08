


#ifndef MAX_POOLING2D_H
#define MAX_POOLING2D_H

#include <vector>
#include <array>
#include <string>

void max_pooling2d(const std::vector<float>& input, std::vector<float>& output,
                   const std::array<int, 4>& input_shape, const std::array<int, 4>& output_shape,
                   const std::array<int, 2>& pool_size, const std::array<int, 2>& strides,
                   const std::string& padding, const std::string& layer_name);

#endif // MAX_POOLING2D_H