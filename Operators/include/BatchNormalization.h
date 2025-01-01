// #ifndef BATCHNORMALIZATION_H
// #define BATCHNORMALIZATION_H

// #include <vector>
// #include <string>

// class BatchNormalization {
// public:
//     // Constructor to initialize with input shape and weight file paths
//     BatchNormalization(const std::vector<int>& input_shape,
//                        const std::vector<std::string>& weight_file_paths);

//     // Load weights from the specified binary files
//     void loadWeights();

//     // Perform batch normalization on the input 3D tensor
//     void forward(const std::vector<std::vector<std::vector<float>>>& input);

//     // Save the output tensor to a binary file
//     void saveOutput(const std::string& output_file_path) const;

//     // Getter for the normalized output
//     const std::vector<std::vector<std::vector<float>>>& getOutput() const;

// private:
//     std::vector<int> input_shape;  // Input shape [height, width, channels]
//     std::vector<std::string> weight_file_paths;  // File paths for gamma, beta, moving_mean, moving_variance
//     std::vector<float> gamma;  // Scale parameter (gamma)
//     std::vector<float> beta;   // Shift parameter (beta)
//     std::vector<float> moving_mean;  // Moving mean parameter
//     std::vector<float> moving_variance;  // Moving variance parameter
//     std::vector<std::vector<std::vector<float>>> output;  // Output tensor

//     // Helper functions to read binary data
//     void readBinaryFile(const std::string& file_path, std::vector<float>& data);
// };

// #endif // BATCHNORMALIZATION_H
// batchnormalization.h
#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H

#include <vector>

// Function declaration for batch normalization
void batch_normalization(const std::vector<std::vector<std::vector<float>>>& input,
                         const std::vector<float>& gamma, const std::vector<float>& beta,
                         const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                         std::vector<std::vector<std::vector<float>>>& output);

#endif // BATCHNORMALIZATION_H
