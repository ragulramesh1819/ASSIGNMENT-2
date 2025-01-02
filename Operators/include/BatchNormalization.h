#ifndef BATCH_NORMALIZATION_H
#define BATCH_NORMALIZATION_H

#include <vector>
#include <string>

class BatchNormalization {
public:
    // Constructor and Destructor
    BatchNormalization();
    ~BatchNormalization();

    // Method to load weights (gamma, beta, moving mean, moving variance)
    void LoadWeights(const std::vector<std::string>& weight_paths);

    // Method to apply batch normalization on input
    void ApplyBatchNormalization(const std::vector<std::vector<std::vector<float>>>& input,
                                  std::vector<std::vector<std::vector<float>>>& output);

    // Setters and Getters for input and output shapes
    void SetInputShape(const std::vector<int>& shape);
    void SetOutputShape(const std::vector<int>& shape);
    std::vector<int> GetInputShape() const;
    std::vector<int> GetOutputShape() const;

private:
    std::vector<float> gamma;       // Scaling factor
    std::vector<float> beta;        // Shifting factor
    std::vector<float> moving_mean; // Moving mean for normalization
    std::vector<float> moving_variance; // Moving variance for normalization

    std::vector<int> input_shape;   // Input shape (height, width, channels)
    std::vector<int> output_shape;  // Output shape

    // Helper function to normalize a 3D vector
    void Normalize3DVector(std::vector<std::vector<std::vector<float>>>& input);
};

#endif // BATCH_NORMALIZATION_H
