// #ifndef MAXPOOLING3D_H
// #define MAXPOOLING3D_H

// #include <vector>

// class MaxPooling3D {
// public:
//     // Constructor: pool size, stride
//     MaxPooling3D(int poolHeight, int poolWidth, int stride);

//     // Method to apply max pooling on 3D input tensor
//     std::vector<std::vector<std::vector<float>>> applyMaxPooling(const std::vector<std::vector<std::vector<float>>>& input);

// private:
//     int poolHeight;
//     int poolWidth;
//     int stride;
// };

// #endif // MAXPOOLING3D_H

// max_pooling2d.h
#ifndef MAX_POOLING2D_H
#define MAX_POOLING2D_H

#include <vector>
#include <cstdint>

class MaxPooling2D {
public:
    MaxPooling2D(const std::vector<int>& inputShape, const std::vector<int>& outputShape, 
                 const std::vector<int>& strides, const std::string& padding);

    void applyPooling(const std::vector<std::vector<std::vector<float>>>& input,
                      std::vector<std::vector<std::vector<float>>>& output);

private:
    std::vector<int> inputShape;
    std::vector<int> outputShape;
    std::vector<int> strides;
    std::string padding;

    float getMaxValue(const std::vector<std::vector<std::vector<float>>>& input, 
                      int startRow, int startCol, int channel, int poolSize);
};

#endif // MAX_POOLING2D_H