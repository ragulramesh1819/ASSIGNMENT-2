#ifndef FLATTEN_H
#define FLATTEN_H

#include <vector>
#include <string>

class Flatten {
public:
    // Constructor
    Flatten();

    // Set input and output shapes
    void SetInputShape(const std::vector<int>& shape);
    void SetOutputShape(const std::vector<int>& shape);

    // Perform the flatten operation
    void ApplyFlatten(const std::vector<std::vector<std::vector<float>>>& input,
                      std::vector<float>& output);

private:
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
};

#endif // FLATTEN_H
