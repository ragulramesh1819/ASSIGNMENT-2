
#ifndef MAX_POOLING2D_H
#define MAX_POOLING2D_H

#include <vector>
#include <cstdint>
#include<string>

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