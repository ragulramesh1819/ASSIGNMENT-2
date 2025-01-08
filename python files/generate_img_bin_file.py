import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Select the image (e.g., index 66)// 280
image_index = 7777
image = x_train[image_index]
image_class = class_names[y_train[image_index][0]]

# Display the image with class name
plt.imshow(image)
plt.title(f"Selected Image: {image_class}")
plt.show()

# Save the image to the project root directory
image_file_path = "E:/Assignment-2-C++/Project_Root/selected_image.png"
plt.imsave(image_file_path, image)
print(f"Image saved as PNG at {image_file_path}")

# Resize the image to shape (1, 32, 32, 3)
resized_image = np.expand_dims(image, axis=0)

# Convert the resized image to binary format and save it
output_file_path = "E:/Assignment-2-C++/Project_Root/resized_image_binary.bin"
resized_image.astype(np.float32).tofile(output_file_path)
print(f"Image saved to binary format at {output_file_path}")
