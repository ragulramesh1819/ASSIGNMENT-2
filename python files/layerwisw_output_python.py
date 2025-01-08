

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def print_first_channel_outputs(model, input_data):
    layer_outputs = [layer.output for layer in model.layers] #if not isinstance(layer, tf.keras.layers.Dropout)]  # Skip dropout layers
    intermediate_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)
    # Get outputs for the input data
    outputs = intermediate_model.predict(input_data)
    for i, output in enumerate(outputs):
        print(f"Layer {i+1} - {model.layers[i].name}")
        if len(output.shape) > 2:  # For layers with multiple channels
            print(output[..., 0])  # Print only the first channel
        else:
            print(output)  # For 1D or scalar outputs, print as is
        print("-" * 50)
    return outputs

model = tf.keras.models.load_model("E:/Assignment-2-C++/Project_Root/cifar10_hyperParameter_model.h5")

# Load and preprocess the image
img_path = "E:\Assignment-2-C++\Project_Root\selected_image.png"
img = image.load_img(img_path, target_size=(32, 32))
input_data = image.img_to_array(img)
input_data = np.expand_dims(input_data, axis=0)

output = print_first_channel_outputs(model, input_data)
last_layer_output = output[-1]  # Get the last layer's output
y_pred = np.argmax(last_layer_output, axis=1)  # Get prediction from the last layer's output

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(class_names[y_pred[0]])