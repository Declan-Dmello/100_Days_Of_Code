import tensorflow as tf
from tensorflow.keras.applications import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras.backend as K


class CNNVisualizer:
    def __init__(self):
        # Load pre-trained VGG16 model
        self.model = VGG16(weights='imagenet')

    def load_and_preprocess_image(self, img_path, target_size=(224, 224)):
        """Load and preprocess image for VGG16"""
        img = load_img(img_path, target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x, img

    def visualize_feature_maps(self, image_array, layer_name):
        """Visualize feature maps for a specific layer"""
        # Get the specified layer's output
        layer_output = self.model.get_layer(layer_name).output
        intermediate_model = Model(inputs=self.model.input,
                                   outputs=layer_output)

        # Get feature maps
        feature_maps = intermediate_model.predict(image_array)

        # Plot feature maps
        n_features = min(64, feature_maps.shape[-1])  # Display up to 64 features
        size = int(np.ceil(np.sqrt(n_features)))

        plt.figure(figsize=(20, 20))
        for i in range(n_features):
            plt.subplot(size, size, i + 1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def generate_grad_cam(self, image_array, layer_name, class_idx):
        """Generate Grad-CAM visualization"""
        # Create a model that maps the input image to:
        # 1. The target conv layer output
        # 2. The final model output
        grad_model = Model(inputs=self.model.input,
                           outputs=[self.model.get_layer(layer_name).output,
                                    self.model.output])

        # Compute gradient of top predicted class with respect to conv layer output
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(image_array)
            loss = predictions[:, class_idx]

        # Extract gradients and conv layer output
        grads = tape.gradient(loss, conv_output)
        conv_output = conv_output[0]
        grads = grads[0]

        # Global average pooling
        weights = tf.reduce_mean(grads, axis=(0, 1))

        # Create cam
        cam = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1)
        cam = tf.maximum(cam, 0)  # ReLU
        cam = cam / tf.reduce_max(cam)  # Normalize
        cam = cam.numpy()

        return cam

    def visualize_filters(self, layer_name, filter_index=0):
        """Visualize filters by generating patterns that maximize their activation"""
        layer = self.model.get_layer(layer_name)

        # Create a model that returns the layer output
        feature_extractor = Model(inputs=self.model.input,
                                  outputs=layer.output)

        # Start with a random image
        input_shape = self.model.input_shape[1:]
        input_img_data = np.random.random((1, *input_shape)) * 20 + 128.

        # Define the loss (maximize filter activation)
        loss = K.mean(feature_extractor.output[..., filter_index])

        # Compute gradients
        grads = K.gradients(loss, feature_extractor.input)[0]

        # Normalization trick
        grads = grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # Create function to compute loss and gradients
        iterate = K.function([feature_extractor.input], [loss, grads])

        # Gradient ascent
        for i in range(40):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 1.

        # Convert to valid image
        img = input_img_data[0]
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def visualize_saliency_map(self, image_array, class_idx):
        """Generate saliency map"""
        with tf.GradientTape() as tape:
            tape.watch(image_array)
            predictions = self.model(image_array)
            loss = predictions[:, class_idx]

        # Get gradients of input image
        grads = tape.gradient(loss, image_array)

        # Take maximum across channels
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)
        saliency = tf.squeeze(saliency)

        # Normalize
        saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))

        return saliency.numpy()


# Example usage
def run_demo(image_path):
    visualizer = CNNVisualizer()

    # Load and preprocess image
    img_array, original_img = visualizer.load_and_preprocess_image(image_path)

    # 1. Visualize feature maps
    print("Generating feature maps...")
    visualizer.visualize_feature_maps(img_array, 'block3_conv1')

    # 2. Generate Grad-CAM
    print("Generating Grad-CAM...")
    cam = visualizer.generate_grad_cam(img_array, 'block5_conv3',
                                       class_idx=visualizer.model.predict(img_array).argmax())

    # 3. Visualize filters
    print("Generating filter visualization...")
    filter_viz = visualizer.visualize_filters('block1_conv1')

    # 4. Generate saliency map
    print("Generating saliency map...")
    saliency = visualizer.visualize_saliency_map(img_array,
                                                 class_idx=visualizer.model.predict(img_array).argmax())

    # Plot results
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cam, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(filter_viz)
    plt.title('Filter Visualization')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(saliency, cmap='jet')
    plt.title('Saliency Map')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# To use the demo:
# run_demo('path_to_your_image.jpg')