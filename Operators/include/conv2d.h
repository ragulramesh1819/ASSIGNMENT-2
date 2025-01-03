#ifndef CONV2D_H
#define CONV2D_H

#include <vector>
#include <string>

class Conv2D {
public:
    Conv2D(const std::vector<int>& inputShape, const std::vector<int>& outputShape, 
           const std::vector<int>& kernelSize, const std::vector<int>& strides, 
           const std::string& padding, const std::string& activation);

    void loadWeights(const std::string& kernelFile, const std::string& biasFile);
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& input);

private:
    std::vector<int> inputShape;
    std::vector<int> outputShape;
    std::vector<int> kernelSize;
    std::vector<int> strides;
    std::string padding;
    std::string activation;

    std::vector<std::vector<std::vector<std::vector<float>>>> kernel;
    std::vector<float> bias;

    void applyActivation(std::vector<std::vector<std::vector<float>>>& output);
    std::vector<std::vector<std::vector<float>>> padInput(
        const std::vector<std::vector<std::vector<float>>>& input);
};

#endif // CONV2D_H




///// single vector 




