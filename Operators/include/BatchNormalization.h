

#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H

#include <vector>
#include <string>



void batch_normalization(const std::vector<float>& input, std::vector<float>& output,
                         const std::vector<float>& gamma, const std::vector<float>& beta,
                         const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                         float epsilon, size_t channels, size_t height, size_t width, const std::string& layer_name);

#endif // BATCHNORMALIZATION_H