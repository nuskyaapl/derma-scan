import tensorflow as tf
import numpy as np
import cv2
import os


# --------------------------------------------------
# Generate Grad-CAM Heatmap
# --------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    """
    Generate Grad-CAM heatmap for a given preprocessed image array and model.

    Parameters:
    - img_array: Preprocessed image (shape: 1,224,224,3)
    - model: Trained classification model
    - last_conv_layer_name: Name of last convolutional layer

    Returns:
    - heatmap: 2D array representing important regions (values 0–1)
    """

    # Create a new model that outputs:
    # 1. Feature maps from last convolutional layer
    # 2. Final prediction output
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Use GradientTape to record operations for gradient calculation
    with tf.GradientTape() as tape:

        # Get convolution layer outputs and predictions
        conv_outputs, predictions = grad_model(img_array)

        # Use prediction score as loss (target output)
        # predictions[0] -> first (and only) image in batch
        loss = predictions[0]

    # Compute gradients of loss w.r.t. convolution outputs
    grads = tape.gradient(loss, conv_outputs)

    # Perform global average pooling on gradients
    # This gives importance weight for each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Remove batch dimension from conv outputs
    conv_outputs = conv_outputs[0]

    # Multiply feature maps by importance weights
    # This produces the heatmap
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    # Remove extra dimensions
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: keep only positive values (important regions)
    heatmap = tf.maximum(heatmap, 0)

    # Find maximum value for normalization
    max_val = tf.math.reduce_max(heatmap)

    # If heatmap is all zeros, return blank heatmap
    if max_val == 0:
        return np.zeros((224, 224), dtype=np.float32)

    # Normalize heatmap values between 0 and 1
    heatmap = heatmap / max_val

    # Convert TensorFlow tensor to NumPy array
    return heatmap.numpy()


# --------------------------------------------------
# Overlay Heatmap on Original Image
# --------------------------------------------------
def overlay_heatmap(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image.

    Parameters:
    - image: Original PIL image
    - heatmap: Generated heatmap (0–1 values)
    - alpha: Transparency level of heatmap

    Returns:
    - superimposed_img: Image with heatmap overlay (BGR format)
    """

    img = np.array(image.convert("RGB"))

    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))

    # Convert heatmap values to 0–255 range
    heatmap = np.uint8(255 * heatmap)

    # Apply color map (JET gives red-yellow-blue visualization)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    # alpha controls visibility of heatmap
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img


# --------------------------------------------------
# Delete Old Heatmaps
# --------------------------------------------------
def clear_old_heatmaps(static_folder):
    """
    Delete previously generated heatmap images from static folder.

    This prevents storage from filling up and keeps only latest results.
    """

    # If folder does not exist, do nothing
    if not os.path.exists(static_folder):
        return

    # Loop through all files in static folder
    for file in os.listdir(static_folder):

        # Only delete files that start with "heatmap_"
        if file.startswith("heatmap_"):
            try:
                # Remove file
                os.remove(os.path.join(static_folder, file))
            except Exception:
                # Ignore errors (safe deletion)
                pass