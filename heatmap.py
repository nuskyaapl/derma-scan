import tensorflow as tf
import numpy as np
import cv2
import os


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    """
    Generate Grad-CAM heatmap for a given preprocessed image array and model.
    """

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)

    if max_val == 0:
        return np.zeros((224, 224), dtype=np.float32)

    heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on original PIL image.
    Returns BGR image for OpenCV saving.
    """

    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img


def clear_old_heatmaps(static_folder):
    """
    Delete old generated heatmaps from static folder.
    """

    if not os.path.exists(static_folder):
        return

    for file in os.listdir(static_folder):
        if file.startswith("heatmap_"):
            try:
                os.remove(os.path.join(static_folder, file))
            except Exception:
                pass